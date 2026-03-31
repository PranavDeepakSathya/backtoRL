import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPFeaturedPolicy(nn.Module):
    """
    MLP policy with hand-crafted edge features for Turan RL.

    Features per edge (i,j): [edge_exists, deg_i/(n-1), deg_j/(n-1), common_nbrs/max(n-2,1)]
    These are concatenated into a flat vector of size 4*E and fed to shared actor/critic MLPs.

    n is inferred from obs.shape[-1] at every call, so the same weights work for any graph size.
    """

    _cache: dict = {}

    def __init__(self, hidden_dim: int = 256, n_layers: int = 2,
                 obs_dim=None, action_dim=None):
        super().__init__()
        assert obs_dim is not None, "obs_dim (= env.num_actions = E) must be provided"
        E = obs_dim

        def mlp(in_dim, out_dim):
            layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
            layers.append(nn.Linear(hidden_dim, out_dim))
            return nn.Sequential(*layers)

        self._actor  = mlp(4 * E, E)
        self._critic = mlp(4 * E, 1)

    def _build(self, E, device):
        pass  # kept for compat, no-op

    @classmethod
    def _get_cache(cls, n: int, device):
        key = (n, str(device))
        if key not in cls._cache:
            ui, vi = torch.triu_indices(n, n, offset=1, device=device)
            cls._cache[key] = (ui, vi)
        return cls._cache[key]

    @staticmethod
    def _infer_n(E: int) -> int:
        return int((1.0 + (1.0 + 8.0 * E) ** 0.5) / 2.0)

    def _featurize(self, obs):
        B, E = obs.shape
        n = self._infer_n(E)
        ui, vi = self._get_cache(n, obs.device)

        x = obs.float()
        adj = x.new_zeros(B, n, n)
        adj[:, ui, vi] = x
        adj[:, vi, ui] = x

        deg = adj.sum(-1) / (n - 1)
        cn  = (adj[:, ui] * adj[:, vi]).sum(-1) / max(n - 2, 1)

        # (B, E, 4) -> (B, 4*E)
        feats = torch.stack([x, deg[:, ui], deg[:, vi], cn], dim=-1)
        return feats.view(B, 4 * E)

    def forward(self, obs):
        B, E = obs.shape
        self._build(E, obs.device)
        feats  = self._featurize(obs)
        logits = self._actor(feats)
        value  = self._critic(feats).squeeze(-1)
        return logits, value

    def get_action(self, obs, deterministic=False):
        logits, value = self(obs)
        logits = logits.masked_fill(obs.bool(), float('-inf'))
        dist   = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate(self, obs, actions):
        logits, value = self(obs)
        logits = logits.masked_fill(obs.bool(), float('-inf'))
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
