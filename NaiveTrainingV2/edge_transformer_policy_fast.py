"""
FastEdgeTransformerPolicy — all-bf16 feature computation, no adj matrix.

Changes vs edge_transformer_policy.py:
  1. No .float() cast — all feature math stays in bf16 under autocast.
  2. Degrees via scatter_add: avoids allocating (B, n, n) adj for degree computation.
  3. Common neighbours via precomputed edge-index lookup tensors (cn_a_idx, cn_b_idx):
     eliminates the adj matrix entirely — two (B, E, n) gathers replace the adj build + row gather.
  4. edge_adj bias cached in bf16 and shaped (1, 1, E, E) — no unsqueeze every forward,
     one gamma multiply per forward pass.
  5. reshape() instead of .contiguous().view() in attention output.
  6. gamma cast to obs.dtype before multiply so the bias stays in bf16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class _Block(nn.Module):
    """Pre-norm transformer block, 2× FFN.  bias: (1, 1, E, E) — broadcast over B and heads."""

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
        # bias: (1, 1, E, E) — already shaped for SDPA broadcast
        B, E, d = x.shape
        H, Dh   = self.heads, self.d_h

        xn = self.norm1(x)
        def split(t):
            return t.view(B, E, H, Dh).transpose(1, 2)   # (B, H, E, Dh)

        q, k, v = split(self.Wq(xn)), split(self.Wk(xn)), split(self.Wv(xn))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        # reshape avoids the separate .contiguous() allocation
        x = x + self.Wo(out.transpose(1, 2).reshape(B, E, d))
        x = x + self.ff(self.norm2(x))
        return x


class FastEdgeTransformerPolicy(nn.Module):
    """
    Speed-optimised EdgeTransformerPolicy.  Drop-in replacement — same
    forward / get_action / evaluate interface as EdgeTransformerPolicy.

    Forward pass is entirely in bf16 (no float32 intermediates):
      - degrees computed via scatter_add directly on the edge vector
      - common neighbours via precomputed (E, n) edge-index lookup tables
      - share-a-node bias cached as (1, 1, E, E) bf16 tensor per (n, device)
    """

    _cache: dict = {}   # (n, device_str, dtype) → (ui, vi, edge_adj, cn_a_idx, cn_b_idx, cn_mask)

    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model    = d_model
        self.proj_in    = nn.Linear(4, d_model)
        self.blocks     = nn.ModuleList([_Block(d_model, n_heads) for _ in range(n_layers)])
        self.gamma      = nn.Parameter(torch.tensor(1.0))
        self.logit_head = nn.Linear(d_model, 1)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1)
        )

    # ------------------------------------------------------------------
    @classmethod
    def _get_cache(cls, n: int, device, dtype):
        key = (n, str(device), dtype)
        if key not in cls._cache:
            E    = n * (n - 1) // 2
            ui, vi = torch.triu_indices(n, n, offset=1, device=device)   # (E,) each

            # ── share-a-node attention bias ──────────────────────────
            share = (
                (ui.unsqueeze(1) == ui.unsqueeze(0)) |
                (ui.unsqueeze(1) == vi.unsqueeze(0)) |
                (vi.unsqueeze(1) == ui.unsqueeze(0)) |
                (vi.unsqueeze(1) == vi.unsqueeze(0))
            ).to(dtype)                          # (E, E) bf16
            edge_adj = share.unsqueeze(0).unsqueeze(0)  # (1, 1, E, E) — SDPA-ready

            # ── edge-index lookup for common neighbours ──────────────
            # edge_map[i, j] = edge index for edge (i, j); both orders stored.
            # Diagonal stays 0 (dummy — always masked out by cn_mask below).
            edge_map = torch.zeros(n, n, dtype=torch.long, device=device)
            edge_map[ui, vi] = torch.arange(E, device=device)
            edge_map[vi, ui] = torch.arange(E, device=device)

            # For each edge e=(ui[e], vi[e]) and each node k,
            # cn_a_idx[e, k] = index of edge connecting ui[e] to k
            # cn_b_idx[e, k] = index of edge connecting vi[e] to k
            cn_a_idx = edge_map[ui]   # (E, n) long
            cn_b_idx = edge_map[vi]   # (E, n) long

            # Mask: zero out k == ui[e] (no self-loop) and k == vi[e] (the edge itself)
            k_range  = torch.arange(n, device=device)
            cn_mask  = (
                (k_range.unsqueeze(0) != ui.unsqueeze(1)) &
                (k_range.unsqueeze(0) != vi.unsqueeze(1))
            ).to(dtype)               # (E, n) bf16

            cls._cache[key] = (ui, vi, edge_adj, cn_a_idx, cn_b_idx, cn_mask)
        return cls._cache[key]

    @staticmethod
    def _infer_n(E: int) -> int:
        return int((1.0 + (1.0 + 8.0 * E) ** 0.5) / 2.0)

    # ------------------------------------------------------------------
    def forward(self, obs):
        """
        obs: (B, E) bfloat16   →   logits (B, E),  value (B,)

        Everything stays in bf16 — call under torch.autocast('cuda', torch.bfloat16).
        """
        B, E = obs.shape
        n    = self._infer_n(E)
        ui, vi, edge_adj, cn_a_idx, cn_b_idx, cn_mask = \
            self._get_cache(n, obs.device, obs.dtype)

        x = obs   # (B, E) bf16 — no .float() cast

        # ── Node degrees via scatter_add (no (B,n,n) adj needed) ────
        deg = x.new_zeros(B, n)
        deg.scatter_add_(1, ui.unsqueeze(0).expand(B, -1), x)
        deg.scatter_add_(1, vi.unsqueeze(0).expand(B, -1), x)
        deg = deg * (1.0 / (n - 1))   # (B, n) normalised

        # ── Common neighbours via precomputed lookup (no adj at all) ─
        # x[:, cn_a_idx]: (B, E, n) — presence of edges from ui[e] to each k
        # x[:, cn_b_idx]: (B, E, n) — presence of edges from vi[e] to each k
        # cn_mask zeros k==ui[e] and k==vi[e] entries
        cn = (x[:, cn_a_idx] * x[:, cn_b_idx] * cn_mask).sum(-1) \
             * (1.0 / max(n - 2, 1))   # (B, E) normalised

        feats = torch.stack([x, deg[:, ui], deg[:, vi], cn], dim=-1)   # (B, E, 4)

        h    = self.proj_in(feats)                                       # (B, E, d)
        bias = self.gamma.to(obs.dtype) * edge_adj                      # (1,1,E,E) bf16

        for block in self.blocks:
            h = block(h, bias)

        logits = self.logit_head(h).squeeze(-1)                         # (B, E)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value  = self.value_head(h.mean(1)).squeeze(-1)                 # (B,)
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
