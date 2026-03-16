import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLPActorCritic(nn.Module):
  def __init__(self, n, hidden=256, n_layers=2):
    super().__init__()
    obs_dim    = n * n
    action_dim = n * (n - 1) // 2
    self.n     = n

    def mlp(dims):
      layers = []
      for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.Tanh()]
      return nn.Sequential(*layers[:-1])

    dims         = [obs_dim] + [hidden] * n_layers
    self.actor   = nn.Sequential(*mlp(dims), nn.Linear(hidden, action_dim))
    self.critic  = nn.Sequential(*mlp(dims), nn.Linear(hidden, 1))

    self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.01)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def forward(self, obs):
    x      = obs.float()
    logits = self.actor(x)
    value  = self.critic(x).squeeze(-1)
    return logits, value

  def get_action(self, obs, deterministic=False):
    logits, value = self(obs)
    dist   = Categorical(logits=logits)
    action = dist.mode if deterministic else dist.sample()
    return action, dist.log_prob(action), dist.entropy(), value

  def evaluate(self, obs, actions):
    logits, value = self(obs)
    dist     = Categorical(logits=logits)
    return dist.log_prob(actions), dist.entropy(), value


class TransformerActorCritic(nn.Module):
  def __init__(self, n, d_model=128, n_heads=4, n_layers=4, dropout=0.0):
    super().__init__()
    self.n       = n
    self.d_model = d_model
    self.E       = n * (n - 1) // 2

    u_idx, v_idx = torch.triu_indices(n, n, offset=1)
    self.register_buffer('edge_u', u_idx)
    self.register_buffer('edge_v', v_idx)

    self.token_embed = nn.Linear(n, d_model)
    self.pos_embed   = nn.Embedding(n, d_model)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model         = d_model,
      nhead           = n_heads,
      dim_feedforward = 4 * d_model,
      dropout         = dropout,
      batch_first     = True,
      norm_first      = True,
    )
    self.transformer  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    self.action_proj  = nn.Linear(d_model, 1, bias=False)
    self.value_head   = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.Tanh(),
      nn.Linear(d_model, 1),
    )

    self._init_weights()

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=0.01)
        if m.bias is not None:
          nn.init.zeros_(m.bias)
    nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

  def encode(self, obs):
    batch  = obs.shape[0]
    tokens = obs.view(batch, self.n, self.n).to(torch.bfloat16)
    x      = self.token_embed(tokens)
    pos    = self.pos_embed(
      torch.arange(self.n, device=obs.device)
    ).unsqueeze(0)
    x = x + pos
    x = self.transformer(x)
    return x

  def forward(self, obs):
    x      = self.encode(obs)
    u_emb  = x[:, self.edge_u, :]
    v_emb  = x[:, self.edge_v, :]
    logits = self.action_proj(u_emb * v_emb).squeeze(-1).float()
    value  = self.value_head(x.mean(dim=1)).squeeze(-1).float()
    return logits, value

  def get_action(self, obs, deterministic=False):
    logits, value = self(obs)
    dist   = Categorical(logits=logits)
    action = dist.mode if deterministic else dist.sample()
    return action, dist.log_prob(action), dist.entropy(), value

  def evaluate(self, obs, actions):
    logits, value = self(obs)
    dist     = Categorical(logits=logits)
    return dist.log_prob(actions), dist.entropy(), value


def make_mlp(n, hidden=256, n_layers=2, device='cuda'):
  return MLPActorCritic(n=n, hidden=hidden, n_layers=n_layers).to(device)


def make_transformer(n, d_model=128, n_heads=4, n_layers=4, device='cuda', compile=True):
  policy = TransformerActorCritic(
    n=n, d_model=d_model, n_heads=n_heads, n_layers=n_layers
  ).to(device).to(torch.bfloat16)
  if compile:
    policy = torch.compile(policy, mode='reduce-overhead')
  return policy