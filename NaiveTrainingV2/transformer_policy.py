import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.q   = nn.Linear(d_model, d_model, bias=False)
        self.k   = nn.Linear(d_model, d_model, bias=False)
        self.v   = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, adj_bias):
        # x:        (B, N, D)
        # adj_bias: (B, N, N)  additive bias on attention logits
        B, N, D = x.shape
        H, Dh   = self.n_heads, self.d_head

        xn = self.norm1(x)
        q  = self.q(xn).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)
        k  = self.k(xn).view(B, N, H, Dh).transpose(1, 2)
        v  = self.v(xn).view(B, N, H, Dh).transpose(1, 2)

        # broadcast adj_bias across heads: (B, 1, N, N) → (B, H, N, N)
        bias = adj_bias.unsqueeze(1).expand(-1, H, -1, -1)

        # F.scaled_dot_product_attention uses FlashAttention when available
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)  # (B, H, N, Dh)
        attn = attn.transpose(1, 2).contiguous().view(B, N, D)
        x    = x + self.out(attn)

        x = x + self.ff(self.norm2(x))
        return x


class GraphTransformerPolicy(nn.Module):
    """
    Transformer policy for graph RL.

    Each graph vertex is a token.  Initial node features are derived from
    vertex degree — size-independent so the policy generalises to unseen n.
    Attention is biased by the adjacency matrix, making it equivalent to
    graph message passing.  Edge logits are bilinear scores between final
    node embeddings; value is a mean-pool scalar.

    Interface matches MLPActorCritic: forward / get_action / evaluate.
    Variable n is inferred from obs.shape[-1] at every forward call.
    """

    _triu_cache: dict = {}  # class-level cache shared across instances

    def __init__(self, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.d_model = d_model

        # Node input encoding (variable-n compatible):
        #   h_i^0 = e_node + degree(i) * e_nbr
        self.e_node = nn.Parameter(torch.empty(d_model))
        self.e_nbr  = nn.Parameter(torch.empty(d_model))
        nn.init.normal_(self.e_node, std=0.02)
        nn.init.normal_(self.e_nbr,  std=0.02)

        # Learned scale applied to adj matrix before using as attention bias.
        # 1.0 init: mild structural signal, learned from there.
        self.adj_bias_scale = nn.Parameter(torch.tensor(1.0))

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads) for _ in range(n_layers)
        ])

        # Bilinear edge scoring: logit(i,j) = (h_i ⊙ W_edge h_j).sum()
        self.W_edge = nn.Linear(d_model, d_model, bias=False)

        # Value head: mean-pool node embeddings → scalar
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _triu(self, n: int, device):
        key = (n, str(device))
        if key not in GraphTransformerPolicy._triu_cache:
            GraphTransformerPolicy._triu_cache[key] = \
                torch.triu_indices(n, n, offset=1, device=device)
        return GraphTransformerPolicy._triu_cache[key]

    @staticmethod
    def _infer_n(num_actions: int) -> int:
        """Recover n from num_actions = n*(n-1)//2."""
        return int((1.0 + (1.0 + 8.0 * num_actions) ** 0.5) / 2.0)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, obs):
        """
        obs: (B, num_actions)  bool / bfloat16 / float32
        returns: logits (B, num_actions),  value (B,)
        """
        B, na = obs.shape
        n     = self._infer_n(na)
        ui, vi = self._triu(n, obs.device)

        # Reconstruct (B, n, n) adjacency matrix
        adj = obs.new_zeros(B, n, n)
        adj[:, ui, vi] = obs
        adj[:, vi, ui] = obs

        # Initial node features
        degree = adj.sum(-1)                                    # (B, n)
        h = self.e_node + degree.unsqueeze(-1) * self.e_nbr    # (B, n, d_model)

        # Attention bias from adjacency
        adj_bias = adj * self.adj_bias_scale                    # (B, n, n)

        for layer in self.layers:
            h = layer(h, adj_bias)

        # Edge logits: bilinear score, scaled by 1/sqrt(d_model) to keep
        # magnitudes bounded and prevent entropy collapse.
        logits = (h[:, ui] * self.W_edge(h[:, vi])).sum(-1) / math.sqrt(self.d_model)

        # Value: mean-pool node embeddings
        value = self.value_head(h.mean(dim=1)).squeeze(-1)      # (B,)

        return logits, value

    # ------------------------------------------------------------------
    # RL interface (same as MLPActorCritic)
    # ------------------------------------------------------------------

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        logits = logits.masked_fill(obs.bool(), float('-inf'))
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        logits = logits.masked_fill(obs.bool(), float('-inf'))
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
