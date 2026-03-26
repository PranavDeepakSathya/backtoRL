# Wrapping a C Environment for PufferLib Training

## Overview

PufferLib provides two ways to integrate C environments:
1. **Ocean-style**: Full PufferLib C API integration via `env_binding.h` (zero-copy, maximum performance)
2. **Python PufferEnv wrapper**: Wrap your existing C library with a Python class that inherits `PufferEnv`

This document covers both approaches, using our Turan graph environment as the concrete example.

---

## Approach 1: Ocean-Style (Full C API via `env_binding.h`)

This is the "PufferLib way" — your C code writes directly into shared memory buffers that the training loop reads from. Zero redundant copies.

### Step 1: Define Your C Environment Struct

Your env struct MUST have these fields (PufferLib reads/writes them directly):

```c
// Required by env_binding.h
typedef struct Log {
    float episode_return;   // Must be all floats
    float episode_length;
    float score;
    float n;                // MUST have 'n' as last field (episode counter)
} Log;

typedef struct TuranEnv {
    // ── PufferLib-required fields (env_binding.h reads these) ──
    void* observations;     // Pointer into shared numpy buffer
    void* actions;          // Pointer into shared numpy buffer
    void* rewards;          // Pointer into shared numpy buffer (float32)
    unsigned char* terminals;  // Pointer into shared numpy buffer (uint8)
    Log log;                // Aggregated logging struct

    // ── Your game state ──
    int n;                  // Graph size
    int num_actions;        // n*(n-1)/2
    uint64_t* packed;       // Adjacency bitset
    int edge_count;
    int checker_id;
    int tick;
    int max_steps;
} TuranEnv;
```

**Critical**: `observations`, `actions`, `rewards`, `terminals` are NOT allocated by your code. PufferLib allocates them as numpy arrays and passes pointers in during `vec_init`. Your C code reads actions from and writes observations/rewards/terminals to these buffers.

### Step 2: Implement Required C Functions

```c
// These four functions MUST exist — env_binding.h calls them
void c_reset(TuranEnv* env) {
    // Reset game state
    memset(env->observations, 0, env->num_actions * sizeof(uint8_t));
    // ... reset packed adjacency, edge_count, tick, etc.
    env->rewards[0] = 0.0f;       // Write reward directly
    env->terminals[0] = 0;         // Not done
}

void c_step(TuranEnv* env) {
    // Read action from shared buffer
    int action = ((int*)env->actions)[0];

    // Decode action to (u, v) edge
    int u = /* ... */, v = /* ... */;

    // Game logic
    if (creates_forbidden_subgraph(env, u, v)) {
        env->terminals[0] = 1;
        env->rewards[0] = 0.0f;

        // Log episode stats
        env->log.episode_return += env->edge_count;
        env->log.score = env->edge_count;
        env->log.episode_length = env->tick;
        env->log.n += 1;  // Increment episode counter

        c_reset(env);  // Auto-reset
    } else {
        // Add edge
        env->observations[edge_idx] = 1;  // Write obs directly
        env->edge_count++;
        env->rewards[0] = 1.0f;           // Write reward directly
        env->terminals[0] = 0;
    }
    env->tick++;
}

void c_render(TuranEnv* env) {
    // Optional: rendering code (raylib, terminal, etc.)
}

void c_close(TuranEnv* env) {
    // Free any internal allocations
    free(env->packed);
}
```

### Step 3: Create the Binding File

```c
// binding.c
#include "turan.h"    // Your header with TuranEnv struct + game logic

#define Env TuranEnv   // Tell env_binding.h your struct name
#include "../env_binding.h"  // PufferLib's binding machinery

// Initialize env from Python kwargs
static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->n = (int)unpack(kwargs, "n");
    env->checker_id = (int)unpack(kwargs, "checker_id");
    env->max_steps = (int)unpack(kwargs, "max_steps");
    env->num_actions = env->n * (env->n - 1) / 2;

    // Allocate internal state (NOT obs/actions/rewards — those come from PufferLib)
    int rows = (env->n + 63) / 64;
    env->packed = (uint64_t*)calloc(env->n * rows, sizeof(uint64_t));

    c_reset(env);
    return 0;
}

// Define what to log
static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    return 0;
}
```

### Step 4: Build System (setup.py)

```python
# In your setup.py or pyproject.toml
from setuptools import setup, Extension

turan_ext = Extension(
    'turan.binding',
    sources=['turan/binding.c'],
    include_dirs=[
        numpy.get_include(),
        'path/to/pufferlib/ocean/',  # For env_binding.h
    ],
    extra_compile_args=['-O3', '-march=native'],
)

setup(ext_modules=[turan_ext])
```

Or if using PufferLib's ocean build system, add to the ocean `__init__.py`.

### Step 5: Python Environment Wrapper

```python
import gymnasium
import numpy as np
import pufferlib

class TuranEnv(pufferlib.PufferEnv):
    def __init__(self, n=20, checker_id=1, max_steps=1000,
                 num_envs=1, buf=None, seed=0):
        self.n = n
        self.checker_id = checker_id
        self.max_steps = max_steps
        num_actions = n * (n - 1) // 2

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(num_actions,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(num_actions)
        self.num_agents = num_envs

        super().__init__(buf)  # MUST call after setting spaces + num_agents

        # Load C binding
        from turan import binding
        self.c_binding = binding

        # Initialize C environments
        self.c_envs = binding.vec_init(
            self.observations, self.actions, self.rewards,
            self.terminals, self.truncations,
            num_envs, seed,
            n=n, checker_id=checker_id, max_steps=max_steps
        )

    def reset(self, seed=0):
        self.c_binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.c_binding.vec_step(self.c_envs)
        info = self.c_binding.vec_log(self.c_envs)
        infos = [info] if info else []
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        self.c_binding.vec_close(self.c_envs)
```

### What `env_binding.h` Provides Automatically

When you `#include "env_binding.h"`, you get these Python-callable functions for free:

| Function | Signature | What it does |
|----------|-----------|--------------|
| `vec_init` | `(obs, act, rew, term, trunc, num_envs, seed, **kwargs)` | Allocates N Env structs, points each at a slice of the shared buffers |
| `vec_reset` | `(handle, seed)` | Calls `c_reset()` on each env |
| `vec_step` | `(handle)` | Calls `c_step()` on each env sequentially |
| `vec_log` | `(handle)` | Aggregates Log structs across envs, returns averaged dict |
| `vec_render` | `(handle, env_id)` | Calls `c_render()` on one env |
| `vec_close` | `(handle)` | Calls `c_close()` + free on each env |
| `env_init` | Single-env version | For non-vectorized use |
| `env_step` | Single-env version | |

### How Zero-Copy Works

```
Python (numpy array) ──shared memory──> C struct pointer
                                         │
  observations[agent_i] ────────────────> env->observations
  actions[agent_i]      ────────────────> env->actions
  rewards[agent_i]      ────────────────> env->rewards
  terminals[agent_i]    ────────────────> env->terminals

C code writes directly into the numpy arrays.
No memcpy. No serialization. No Python overhead per step.
```

In `vec_init`, each env gets a pointer offset into the shared numpy arrays:
```c
env->observations = (void*)((char*)PyArray_DATA(observations)
                   + i * PyArray_STRIDE(observations, 0));
env->actions      = (void*)((char*)PyArray_DATA(actions)
                   + i * PyArray_STRIDE(actions, 0));
// etc.
```

---

## Approach 2: Python PufferEnv Wrapper (Simpler, Still Fast)

If you already have a working C library with ctypes bindings (like our `turan_env_c.py`), you can wrap it as a PufferEnv without the full `env_binding.h` integration.

```python
import gymnasium
import numpy as np
import pufferlib
from turan_env_c import CEnv, CHECKER_C4

class TuranPufferEnv(pufferlib.PufferEnv):
    def __init__(self, n=20, num_envs=1024, checker_id=CHECKER_C4,
                 buf=None, seed=0):
        self.n = n
        self.checker_id = checker_id
        num_actions = n * (n - 1) // 2

        # 1. Set spaces and num_agents BEFORE super().__init__
        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=1, shape=(num_actions,), dtype=np.uint8)
        self.single_action_space = gymnasium.spaces.Discrete(num_actions)
        self.num_agents = num_envs

        super().__init__(buf)

        # 2. Create the C environment
        self.c_env = CEnv(n=n, num_envs=num_envs, checker_id=checker_id)

    def reset(self, seed=0):
        obs = self.c_env.reset()
        self.observations[:] = obs  # Copy into PufferLib's buffer
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        obs, reward, done = self.c_env.step(actions)

        # Copy results into PufferLib's shared buffers
        self.observations[:] = obs
        self.rewards[:] = reward
        self.terminals[:] = done

        # Auto-reset is handled by CEnv internally
        infos = []
        if done.any():
            edge_counts = self.c_env.edge_count.copy()
            infos = [{'episode_return': float(edge_counts[done].mean())}]

        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        self.c_env.close()
```

### Trade-offs

| Aspect | Ocean-style (env_binding.h) | Python PufferEnv wrapper |
|--------|---------------------------|--------------------------|
| Copies per step | 0 (zero-copy) | 1 (numpy → PufferLib buffer) |
| Complexity | Higher (C API contract) | Lower (Python glue) |
| Performance | Maximum | ~90-95% of maximum |
| Auto-reset | Must implement in C | Can use Python |
| Logging | Struct-based, auto-aggregated | Manual in Python |
| Multi-agent | Native via shared buffers | Manual indexing |

---

## Vectorization Backends

Once you have a PufferEnv, PufferLib can vectorize it:

```python
import pufferlib.vector

# Option 1: Serial (for debugging)
vecenv = pufferlib.vector.make(
    TuranPufferEnv,
    num_envs=1, num_workers=1,
    backend=pufferlib.vector.Serial,
    env_kwargs={'n': 20, 'num_envs': 1024}
)

# Option 2: Multiprocessing (production)
vecenv = pufferlib.vector.make(
    TuranPufferEnv,
    num_envs=2, num_workers=2,     # 2 worker processes
    batch_size=1,                   # Envs per send/recv batch
    backend=pufferlib.vector.Multiprocessing,
    env_kwargs={'n': 20, 'num_envs': 4096}  # Each worker runs 4096 envs
)
```

**Key pattern**: `num_envs` in `vector.make` = number of worker processes. `num_envs` in `env_kwargs` = environments per worker. Total agents = `num_envs * env_kwargs['num_envs']`.

For fast C environments, you often want few workers with many envs each (reduces IPC overhead):
```python
# Good: 2 workers × 4096 envs = 8192 total
vecenv = pufferlib.vector.make(..., num_envs=2, env_kwargs={'num_envs': 4096})

# Bad: 8192 workers × 1 env = 8192 total (massive IPC overhead)
vecenv = pufferlib.vector.make(..., num_envs=8192, env_kwargs={'num_envs': 1})
```
