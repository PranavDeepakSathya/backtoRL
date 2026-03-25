import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _Block(nn.Module):
    """Pre-norm transformer block, 2x FFN."""

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
        B, E, d = x.shape
        H, Dh   = self.heads, self.d_h

        xn = self.norm1(x)
        def split(t):
            return t.view(B, E, H, Dh).transpose(1, 2)

        q, k, v = split(self.Wq(xn)), split(self.Wk(xn)), split(self.Wv(xn))
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=bias.unsqueeze(0).unsqueeze(0))
        x = x + self.Wo(out.transpose(1, 2).contiguous().view(B, E, d))
        x = x + self.ff(self.norm2(x))
        return x


class EdgeTransformerDegPolicy(nn.Module):
    """
    Edge-token transformer with degree-only features (no common neighbours).

    Per-token features: [edge_exists, deg_i/(n-1), deg_j/(n-1)]  — 3 features.
    Dropping common-neighbour computation removes the O(B*E*n) matmul and
    the C3-biased inductive prior, letting the model learn structure-agnostic
    representations.
    """

    _cache: dict = {}

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 obs_dim=None, action_dim=None):
        super().__init__()
        self.d_model = d_model
        self.proj_in    = nn.Linear(3, d_model)  # 3 features instead of 4
        self.blocks     = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.gamma      = nn.Parameter(torch.tensor(1.0))
        self.logit_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

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

    def forward(self, obs):
        B, E = obs.shape
        n    = self._infer_n(E)
        ui, vi, edge_adj = self._get_cache(n, obs.device)

        x = obs.float()
        adj = x.new_zeros(B, n, n)
        adj[:, ui, vi] = x
        adj[:, vi, ui] = x

        deg = adj.sum(-1) / (n - 1)

        feats = torch.stack([x, deg[:, ui], deg[:, vi]], dim=-1)  # (B, E, 3)

        h    = self.proj_in(feats)
        bias = (self.gamma * edge_adj).to(h.dtype)

        for block in self.blocks:
            h = block(h, bias)

        logits = self.logit_head(h).squeeze(-1)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value  = self.value_head(h.mean(1)).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
