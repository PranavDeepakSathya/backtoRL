# Custom Policies in PufferLib

## Policy Interface

PufferLib policies must implement two methods with specific signatures:

```python
class MyPolicy(torch.nn.Module):
    def forward(self, observations, state=None):
        """Used during TRAINING (PPO update).

        Args:
            observations: [batch_size, *obs_shape] tensor
                          For RNN: [segments, horizon, *obs_shape]
            state: dict with keys:
                'action': [batch_size, *atn_shape] - actions to evaluate
                'lstm_h': [batch, hidden] or None
                'lstm_c': [batch, hidden] or None

        Returns:
            logits: [batch_size, num_actions] or torch.distributions.Normal
            value:  [batch_size, 1] or [batch_size]
        """
        ...

    def forward_eval(self, observations, state=None):
        """Used during EVALUATION (rollout collection).

        Args:
            observations: [agents_per_batch, *obs_shape] tensor
            state: dict with keys:
                'reward': [batch] - current rewards
                'done':   [batch] - current dones
                'env_id': slice - which envs this batch corresponds to
                'mask':   [batch] - active agent mask
                'lstm_h': [batch, hidden] or None (if use_rnn)
                'lstm_c': [batch, hidden] or None (if use_rnn)

        Returns:
            logits: [batch_size, num_actions] or torch.distributions.Normal
            value:  [batch_size, 1] or [batch_size]
        """
        ...
```

**Important**: `forward()` and `forward_eval()` are separated because `torch.compile` has a major performance issue with `torch.no_grad()` context switching. During evaluation, `forward_eval` runs under `torch.no_grad()`, and during training, `forward` runs with gradients. Having them as separate methods lets `torch.compile` optimize each independently.

## Example 1: Simple MLP Policy

From `PufferLib/examples/pufferl.py`:

```python
class Policy(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_size = env.single_observation_space.shape[0]
        act_size = env.single_action_space.n

        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, 128)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(128, 128)),
        )
        self.action_head = torch.nn.Linear(128, act_size)
        self.value_head = torch.nn.Linear(128, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)
```

## Example 2: Policy with LSTM

```python
class RecurrentPolicy(torch.nn.Module):
    def __init__(self, env, hidden_size=256):
        super().__init__()
        obs_size = env.single_observation_space.shape[0]
        act_size = env.single_action_space.n
        self.hidden_size = hidden_size  # PufferLib reads this attribute!

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_size, hidden_size),
            torch.nn.ReLU(),
        )
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.action_head = torch.nn.Linear(hidden_size, act_size)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward_eval(self, observations, state=None):
        # observations: [batch, *obs_shape]
        hidden = self.encoder(observations)

        # LSTM expects [batch, seq=1, features]
        hidden = hidden.unsqueeze(1)
        h = state['lstm_h'].unsqueeze(0)  # [1, batch, hidden]
        c = state['lstm_c'].unsqueeze(0)
        hidden, (h, c) = self.lstm(hidden, (h, c))
        state['lstm_h'] = h.squeeze(0)
        state['lstm_c'] = c.squeeze(0)

        hidden = hidden.squeeze(1)
        return self.action_head(hidden), self.value_head(hidden)

    def forward(self, observations, state=None):
        # observations: [segments, horizon, *obs_shape]
        B, T, *obs_shape = observations.shape
        hidden = self.encoder(observations.reshape(B*T, *obs_shape))
        hidden = hidden.reshape(B, T, -1)

        # Process full sequence
        hidden, _ = self.lstm(hidden)
        hidden = hidden.reshape(B*T, -1)

        return self.action_head(hidden), self.value_head(hidden)
```

**IMPORTANT**: When using RNNs, set `use_rnn: True` in the config. PufferLib will:
- Manage LSTM hidden states across episodes
- Feed sequences of length `bptt_horizon` during training
- Reset hidden states on episode boundaries

## Example 3: Custom Policy for Turan Env (Edge Transformer)

Based on our existing `ModularTraining/policies/edge_transformer.py`:

```python
class TuranPolicy(torch.nn.Module):
    """Policy for graph construction (Turan-type problems).

    Obs shape:  (num_actions,) where num_actions = n*(n-1)/2
    Action:     Discrete(num_actions) — which edge to add
    """

    def __init__(self, env, hidden_size=256, num_heads=4, num_layers=2):
        super().__init__()
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.n

        # Encode binary edge observations
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        )

        # Actor and critic heads
        self.action_head = torch.nn.Linear(hidden_size, act_dim)
        self.value_head = torch.nn.Linear(hidden_size, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.encoder(observations.float())
        logits = self.action_head(hidden)
        value = self.value_head(hidden)
        return logits, value

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

    # Optional: mask invalid actions (edges already placed)
    def forward_eval_masked(self, observations, state=None):
        hidden = self.encoder(observations.float())
        logits = self.action_head(hidden)

        # Mask already-placed edges with -inf
        mask = observations.bool()
        logits = logits.masked_fill(mask, float('-inf'))

        value = self.value_head(hidden)
        return logits, value
```

## Plugging Custom Policies into PufferLib Training

### Option A: Using `pufferl` API

```python
import pufferlib.vector
from pufferlib import pufferl

env_name = 'turan'
# Register your env creator
env_creator = lambda **kwargs: TuranPufferEnv(**kwargs)

vecenv = pufferlib.vector.make(
    env_creator,
    num_envs=2, num_workers=2, batch_size=1,
    backend=pufferlib.vector.Multiprocessing,
    env_kwargs={'n': 20, 'num_envs': 4096, 'checker_id': 1}
)

# Create YOUR custom policy
policy = TuranPolicy(vecenv.driver_env, hidden_size=256).cuda()

# Load default config and customize
args = pufferl.load_config('default')
args['train']['env'] = env_name
args['train']['learning_rate'] = 0.03
args['train']['total_timesteps'] = 50_000_000

# Create trainer with your policy
trainer = pufferl.PuffeRL(args['train'], vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    logs = trainer.train()

trainer.close()
```

### Option B: Using a Config File

Create `configs/turan.ini`:

```ini
[env]
n = 20
checker_id = 1
num_envs = 4096
max_steps = 1000

[vec]
num_envs = 2
num_workers = 2

[policy]
hidden_size = 256

[train]
total_timesteps = 50000000
learning_rate = 0.03
batch_size = 65536
bptt_horizon = 16
update_epochs = 4
optimizer = muon
gamma = 0.999
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
```

Then:
```python
args = pufferl.load_config_file('configs/turan.ini', fill_in_default=True)
```

## `pufferlib.pytorch` Utilities

### `layer_init` — Orthogonal initialization (PPO standard)

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
```

### `sample_logits` — Action sampling

```python
def sample_logits(logits, action=None):
    """
    logits: [batch, num_actions]
    action: [batch] optional — if provided, evaluates log_prob of this action

    Returns:
        action:   [batch]
        logprob:  [batch]
        entropy:  [batch]
    """
    # If logits is Normal distribution → continuous actions
    # If logits is tensor → Categorical distribution
    dist = Categorical(logits=logits)
    if action is None:
        action = dist.sample()
    return action, dist.log_prob(action), dist.entropy()
```

## Architecture Recommendations (from blog)

- **Default**: 1-layer LSTM (128-512 hidden) replacing the main hidden layer
- **2D observations**: 2-3 conv layers with ReLU → flatten → hidden
- **Avoid**: Redundant FC layers when combining heterogeneous inputs
- **Depth**: "Deeper networks are not always better in RL, and they can sometimes be much harder to train"
- **Observation normalization**: Divide continuous values by their maximum (e.g., health/100), don't use running statistics
- **One-hot**: For discrete observations, use one-hot or learned embeddings
