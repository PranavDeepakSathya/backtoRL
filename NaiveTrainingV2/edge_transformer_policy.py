import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _Block(nn.Module):
    """Pre-norm transformer block, 2× FFN. bias is (E, E) — static, broadcast over B and heads."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.d_h   = d // heads
        self.Wq    = nn.Linear(d, d, bias=False)
        self.Wk    = nn.Linear(d, d, bias=False)
        self.Wv    = nn.Linear(d, d, bias=False)
        self.Wo    = nn.Linear(d, d, bias=False)
        self.ff    = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

    def forward(self, x, bias):
        # x:    (B, E, d)
        # bias: (E, E) — additive, broadcast to (1, 1, E, E) for SDPA
        B, E, d = x.shape
        H, Dh   = self.heads, self.d_h

        xn = self.norm1(x)
        def split(t):
            return t.view(B, E, H, Dh).transpose(1, 2)  # (B, H, E, Dh)

        q, k, v = split(self.Wq(xn)), split(self.Wk(xn)), split(self.Wv(xn))
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=bias.unsqueeze(0).unsqueeze(0))
        x = x + self.Wo(out.transpose(1, 2).contiguous().view(B, E, d))
        x = x + self.ff(self.norm2(x))
        return x


class EdgeTransformerPolicy(nn.Module):
    """
    Edge-token transformer policy for Turán RL.

    Each of the E = n(n-1)/2 edge slots is a sequence token.
    Features per token: [edge_exists, deg_i/(n-1), deg_j/(n-1), common_nbrs/max(n-2,1)]
    Attention is biased by a static (E, E) share-a-node matrix so adjacent edges
    attend to each other — exactly the inductive bias needed for triangle detection.

    Interface matches MLPActorCritic: forward / get_action / evaluate.
    n is inferred from obs.shape[-1] at every call; proj_in is Linear(4, d) so
    the same weights work for any graph size.
    """

    _cache: dict = {}  # (n, device_str) → (ui, vi, edge_adj)

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.proj_in    = nn.Linear(4, d_model)
        self.blocks     = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.gamma      = nn.Parameter(torch.tensor(1.0))  # learned adj-bias scale
        self.logit_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

    # ------------------------------------------------------------------
    @classmethod
    def _get_cache(cls, n: int, device):
        key = (n, str(device))
        if key not in cls._cache:
            ui, vi = torch.triu_indices(n, n, offset=1, device=device)
            share = (
                (ui.unsqueeze(1) == ui.unsqueeze(0)) |
                (ui.unsqueeze(1) == vi.unsqueeze(0)) |
                (vi.unsqueeze(1) == ui.unsqueeze(0)) |
                (vi.unsqueeze(1) == vi.unsqueeze(0))
            )
            cls._cache[key] = (ui, vi, share.float())
        return cls._cache[key]

    @staticmethod
    def _infer_n(E: int) -> int:
        return int((1.0 + (1.0 + 8.0 * E) ** 0.5) / 2.0)

    # ------------------------------------------------------------------
    def forward(self, obs):
        """obs: (B, E) bool / bfloat16 / float32  →  logits (B, E), value (B,)"""
        B, E = obs.shape
        n    = self._infer_n(E)
        ui, vi, edge_adj = self._get_cache(n, obs.device)

        # Convert to float for feature computation
        x = obs.float()

        # Symmetric adjacency
        adj = x.new_zeros(B, n, n)
        adj[:, ui, vi] = x
        adj[:, vi, ui] = x

        # Normalised node degrees
        deg = adj.sum(-1) / (n - 1)                            # (B, n)

        # Common neighbours per edge (normalised)
        cn  = (adj[:, ui] * adj[:, vi]).sum(-1) / max(n - 2, 1)  # (B, E)

        feats = torch.stack([x, deg[:, ui], deg[:, vi], cn], dim=-1)  # (B, E, 4)

        h    = self.proj_in(feats)                              # (B, E, d)
        bias = (self.gamma * edge_adj).to(h.dtype)             # (E, E)

        for block in self.blocks:
            h = block(h, bias)

        logits = self.logit_head(h).squeeze(-1)                 # (B, E)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value  = self.value_head(h.mean(1)).squeeze(-1)         # (B,)
        return logits, value

    # ------------------------------------------------------------------
    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
