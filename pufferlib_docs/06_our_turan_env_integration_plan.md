# Integrating Our Turan Env with PufferLib — Concrete Plan

## Current State

We have a working C environment (`ModularTraining/turan_env.c`) with:
- **Obs**: `uint8[num_envs, num_actions]` — upper triangle of adjacency matrix (bool)
- **Actions**: `int[num_envs, 2]` — (u, v) vertex pair
- **Rewards**: `float32[num_envs]` — +1 per safe edge, 0 on terminal
- **Terminal reward override**: on done, reward = total edge_count
- **Done**: `int[num_envs]` — 1 if forbidden subgraph would be created
- **Vectorized**: Handles `num_envs` in C with OpenMP parallelism
- **Python binding**: `turan_env_c.py` wraps via ctypes with zero-copy numpy views

Current training uses our custom `PPOTrainer` in `ModularTraining/trainer.py`.

## What Needs to Change for PufferLib

### Action Space Conversion

**Current**: Actions are edge indices `[0, num_actions)` → decoded to `(u, v)` in Python
**PufferLib needs**: `gymnasium.spaces.Discrete(num_actions)` — single integer per agent

This already matches! Our `CEnv.step()` takes edge indices and decodes them.

### Observation Casting

**Current**: `np.bool_` array
**PufferLib needs**: `gymnasium.spaces.Box(low=0, high=1, shape=(num_actions,), dtype=np.uint8)`

Minor change: cast bool to uint8 or change C to output uint8.

### Reward Handling

**Current**: Terminal reward = total edge count (sparse), step reward = 1.0
**PufferLib**: Clips rewards to `[-1, 1]` by default

Options:
1. Disable reward clipping in config
2. Normalize rewards (divide by expected max edges)
3. Keep dense +1 per step (already in [-1, 1])

Recommended: Keep the dense +1 per step as primary signal. The terminal edge_count can go into logging info.

## Option A: Quick Integration (PufferEnv wrapper)

Minimal changes, uses existing ctypes code:

```python
# turan_puffer.py
import gymnasium
import numpy as np
import pufferlib
from turan_env_c import CEnv, CHECKER_C4

class TuranPufferEnv(pufferlib.PufferEnv):
    def __init__(self, n=20, num_envs=1024, checker_id=CHECKER_C4,
                 buf=None, seed=0):
        num_actions = n * (n - 1) // 2

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(num_actions,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(num_actions)
        self.num_agents = num_envs

        super().__init__(buf)

        self.c_env = CEnv(n=n, num_envs=num_envs, checker_id=checker_id)
        self.n = n
        self.checker_id = checker_id
        self._episode_returns = np.zeros(num_envs, dtype=np.float32)

    def reset(self, seed=0):
        obs = self.c_env.reset()
        self.observations[:] = obs.astype(np.uint8)
        self._episode_returns[:] = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        obs, reward, done = self.c_env.step(actions)

        self.observations[:] = obs.astype(np.uint8)
        self.rewards[:] = reward          # +1 per edge, or edge_count on terminal
        self.terminals[:] = done

        # Track and report episode returns
        infos = []
        done_mask = done.astype(bool)
        if done_mask.any():
            edge_counts = self.c_env.edge_count.copy()
            mean_edges = edge_counts[done_mask].mean()
            max_edges = edge_counts[done_mask].max()
            infos = [{
                'episode_return': float(mean_edges),
                'max_edges': float(max_edges),
            }]

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        self.c_env.close()
```

## Option B: Full Ocean-style Integration

Refactor `turan_env.c` to conform to PufferLib's C API:

### Modified C struct

```c
typedef struct Log {
    float score;            // Best edge count this episode
    float episode_return;   // Cumulative reward
    float episode_length;   // Steps taken
    float n;                // Episode counter (MUST be last)
} Log;

typedef struct TuranEnv {
    // PufferLib-managed pointers (set by env_binding.h)
    void* observations;     // uint8[num_actions]
    void* actions;          // int32[1] (edge index)
    void* rewards;          // float32[1]
    unsigned char* terminals; // uint8[1]
    Log log;

    // Internal state
    int n;
    int rows;
    int num_actions;
    int checker_id;
    uint64_t* packed;
    int edge_count;
    int tick;

    // Precomputed edge→(u,v) lookup
    int* edge_u;
    int* edge_v;
} TuranEnv;
```

### Modified step function

```c
void c_step(TuranEnv* env) {
    int action = ((int*)env->actions)[0];
    int u = env->edge_u[action];
    int v = env->edge_v[action];

    uint8_t* obs = (uint8_t*)env->observations;
    float* rew = (float*)env->rewards;

    CheckerFn checker = get_checker(env->checker_id);

    if (obs[action]) {
        // Edge already exists → no-op
        rew[0] = 0.0f;
        env->terminals[0] = 0;
    } else if (checker(env->packed, env->n, env->rows, u, v)) {
        // Forbidden subgraph → terminal
        rew[0] = 0.0f;
        env->terminals[0] = 1;

        env->log.score = (float)env->edge_count;
        env->log.episode_return += env->edge_count;
        env->log.episode_length = (float)env->tick;
        env->log.n += 1.0f;

        c_reset(env);
    } else {
        // Safe edge addition
        obs[action] = 1;
        SET(env->packed + u * env->rows, v);
        SET(env->packed + v * env->rows, u);
        env->edge_count++;
        rew[0] = 1.0f;
        env->terminals[0] = 0;
    }
    env->tick++;
}
```

## Training Script

```python
import pufferlib
import pufferlib.vector
from pufferlib import pufferl
from turan_puffer import TuranPufferEnv
from policies.edge_transformer import EdgeTransformerPolicy

# Create vectorized env
env_creator = lambda **kwargs: TuranPufferEnv(**kwargs)
vecenv = pufferlib.vector.make(
    env_creator,
    num_envs=2, num_workers=2, batch_size=1,
    backend=pufferlib.vector.Multiprocessing,
    env_kwargs={'n': 20, 'num_envs': 4096, 'checker_id': 1}
)

# Create policy
policy = EdgeTransformerPolicy(vecenv.driver_env).cuda()

# Config
args = pufferl.load_config('default')
args['train'].update({
    'env': 'turan',
    'total_timesteps': 50_000_000,
    'learning_rate': 0.03,
    'batch_size': 65536,
    'bptt_horizon': 16,
    'update_epochs': 4,
    'optimizer': 'muon',
    'gamma': 0.999,
    'gae_lambda': 0.95,
    'compile': True,
})

# Train
trainer = pufferl.PuffeRL(args['train'], vecenv, policy)
while trainer.epoch < trainer.total_epochs:
    trainer.evaluate()
    trainer.train()
trainer.close()
```

## Hyperparameter Sweep with Protein

```python
from pufferlib.sweep import Protein, Log, Linear, Pow2

sweep_config = {
    'metric': 'episode_return',
    'metric_distribution': 'percentile',
    'goal': 'maximize',
    'downsample': 10,
    'use_gpu': True,
    'prune_pareto': True,
    'max_suggestion_cost': 3600,
    'early_stop_quantile': 0.3,

    'train': {
        'learning_rate': {
            'distribution': 'log_normal',
            'min': 0.0001, 'max': 0.1, 'scale': 'auto'
        },
        'gamma': {
            'distribution': 'logit_normal',
            'min': 0.95, 'max': 0.9999, 'scale': 'auto'
        },
        'gae_lambda': {
            'distribution': 'logit_normal',
            'min': 0.8, 'max': 0.99, 'scale': 'auto'
        },
        'ent_coef': {
            'distribution': 'log_normal',
            'min': 0.001, 'max': 0.1, 'scale': 'auto'
        },
        'total_timesteps': {
            'distribution': 'uniform_pow2',
            'min': 2**20, 'max': 2**26, 'scale': 'auto'
        },
    }
}

sweeper = Protein(sweep_config)

for trial in range(100):
    suggestion, info = sweeper.suggest(fill=args)
    # Run training with suggestion...
    # score = final_episode_return
    # cost = wall_clock_seconds
    sweeper.observe(suggestion, score, cost)
```

## Performance Expectations

| Config | Expected SPS | Notes |
|--------|-------------|-------|
| 2 workers × 4096 envs, n=20 | ~500K-1M | Turan is very fast in C |
| 2 workers × 4096 envs, n=40 | ~100K-300K | C4 checker is O(n²) |
| PufferEnv wrapper overhead | ~5-10% | One numpy copy per step |
| Ocean-style (zero-copy) | Baseline | Maximum throughput |
