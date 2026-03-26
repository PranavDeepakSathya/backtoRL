# Puffer Advantage — Custom GAE+VTrace Hybrid

## The Core Insight

PufferLib's "Puffer Advantage" unifies GAE (Generalized Advantage Estimation) and VTrace into a single formulation. From the blog:

> "GAE and VTrace are virtually identical! Puffer Advantage is equal to GAE when you set both clip coefficients to infinity and equal to VTrace with lambda=1, so our method is a strict generalization of both."

## Mathematical Formulation

### Standard GAE (for reference)

```
δ_t     = r_{t+1} + γ · V(s_{t+1}) · (1 - done_{t+1}) - V(s_t)
A_t^GAE = δ_t + γ · λ · (1 - done_{t+1}) · A_{t+1}^GAE
```

### Standard VTrace (for reference)

```
ρ_t = min(ρ̄, π(a_t|s_t) / μ(a_t|s_t))   # truncated importance weight
c_t = min(c̄, π(a_t|s_t) / μ(a_t|s_t))   # truncated trace coefficient
δ_t = ρ_t · (r_{t+1} + γ · V(s_{t+1}) - V(s_t))
A_t^VTrace = δ_t + γ · c_t · A_{t+1}^VTrace
```

### Puffer Advantage (the unified version)

```
ρ_t = min(importance_t, ρ_clip)     # importance = π_new / π_old
c_t = min(importance_t, c_clip)     # separate clip for trace coefficient
δ_t = ρ_t · (r_{t+1} + γ · V(s_{t+1}) · (1 - done_{t+1}) - V(s_t))
A_t^puffer = δ_t + γ · λ · c_t · (1 - done_{t+1}) · A_{t+1}^puffer
```

**Key parameters:**
- `ρ_clip = ∞` and `c_clip = ∞` → reduces to **standard GAE** (importance weights are 1 for on-policy)
- `ρ_clip = ρ̄` and `c_clip = c̄` with `λ = 1` → reduces to **VTrace**
- Intermediate values → a spectrum between GAE and VTrace

## Implementation: CUDA Kernel

The advantage is computed via a custom CUDA kernel for maximum throughput.

### Input Tensors

All tensors have shape `[num_segments, horizon]` and are `float32`:

| Tensor | Shape | Description |
|--------|-------|-------------|
| `values` | `[S, H]` | Value function predictions V(s) |
| `rewards` | `[S, H]` | Rewards received |
| `dones` | `[S, H]` | Terminal flags (0 or 1) |
| `importance` | `[S, H]` | Importance ratio π_new(a|s) / π_old(a|s) |
| `advantages` | `[S, H]` | Output buffer (written in-place) |

Where `S = segments`, `H = bptt_horizon`.

### CUDA Kernel Code (verbatim from `pufferlib.cu`)

```cuda
__host__ __device__ void puff_advantage_row_cuda(
    float* values, float* rewards, float* dones,
    float* importance, float* advantages,
    float gamma, float lambda,
    float rho_clip, float c_clip, int horizon)
{
    float lastpufferlam = 0;
    for (int t = horizon - 2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t   = fminf(importance[t], c_clip);
        float delta = rho_t * (rewards[t_next]
                              + gamma * values[t_next] * nextnonterminal
                              - values[t]);
        lastpufferlam = delta
                      + gamma * lambda * c_t * lastpufferlam * nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

// Each CUDA thread processes one segment row independently
__global__ void puff_advantage_kernel(
    float* values, float* rewards, float* dones,
    float* importance, float* advantages,
    float gamma, float lambda, float rho_clip, float c_clip,
    int num_steps, int horizon)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int offset = row * horizon;
    puff_advantage_row_cuda(
        values + offset, rewards + offset, dones + offset,
        importance + offset, advantages + offset,
        gamma, lambda, rho_clip, c_clip, horizon);
}
```

### Parallelism Strategy

- **Each CUDA thread** processes one segment row (one agent's trajectory of length `horizon`)
- **Thread blocks**: 256 threads per block
- **Grid**: `ceil(num_segments / 256)` blocks
- The backward scan within each row is sequential (inherent to GAE), but all rows run in parallel
- This is efficient because `horizon` is small (e.g. 16-32) while `segments` is large (e.g. 4096+)

### How Importance Ratios Flow

```
Epoch 0:
  ratio[:] = 1                            # Initialize all ratios to 1 (on-policy)
  advantages = compute_puff_advantage(... ratio ...)  # Uses ratio=1 → pure GAE

  for each minibatch:
    newlogprob = policy(mb_obs)
    ratio_mb = exp(newlogprob - mb_logprobs)
    self.ratio[idx] = ratio_mb.detach()   # Store updated ratios

Epoch 1+:
  # Now ratio != 1 for previously-updated segments
  # Puffer Advantage uses clipped ratios for off-policy correction
  advantages = compute_puff_advantage(... ratio ...)  # VTrace-style correction
```

This is the key innovation: within a single PPO update cycle (multiple epochs over the same batch), the advantage is **recomputed each minibatch** using the evolving importance ratios, providing soft off-policy correction without requiring a separate replay buffer.

## When to Use Which Settings

| Setting | `rho_clip` | `c_clip` | Effect |
|---------|-----------|----------|--------|
| Pure GAE | `inf` | `inf` | Standard PPO, no off-policy correction |
| Conservative | `1.0` | `1.0` | Strong off-policy correction (VTrace-like) |
| Moderate | `2.0` | `1.5` | Mild correction, good default |
| VTrace equivalent | `ρ̄` | `c̄`, `λ=1` | Full VTrace |

## Relationship to Prioritized Experience

After computing advantages, PufferLib selects minibatches proportional to advantage magnitude:

```python
adv = advantages.abs().sum(axis=1)           # [segments] - per-row total advantage
prio_weights = adv ** alpha                   # Prioritization exponent
idx = torch.multinomial(prio_probs, mb_size)  # Sample high-advantage segments more
mb_prio = (S * prio_probs[idx]) ** -beta      # Importance sampling correction
```

This means segments with high-magnitude advantages (where the policy can learn the most) are sampled more frequently, similar to Prioritized Experience Replay but within on-policy PPO.
