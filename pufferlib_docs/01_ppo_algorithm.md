# PufferLib PPO Algorithm — Full Pseudocode with Tensor Shapes

## Overview

PufferLib implements PPO (Proximal Policy Optimization) with several key innovations:
- **Puffer Advantage**: A unified GAE + VTrace advantage estimator (CUDA-accelerated)
- **Prioritized Experience Replay** within on-policy training
- **BPTT (Backpropagation Through Time) Horizons** for RNN support
- **Muon optimizer** support (alternative to Adam)
- **Cosine LR annealing** with configurable min ratio
- **Gradient accumulation** for large effective batch sizes

---

## 1. Data Layout and Tensor Shapes

PufferLib organizes experience in a **2D grid** of `[segments, horizon]` rather than the typical `[T, num_envs]` layout. This is critical for BPTT with LSTMs and the custom advantage kernel.

```
total_agents     = num_envs * agents_per_env  (across all vectorized workers)
batch_size       = total training batch size (e.g. 65536)
bptt_horizon     = temporal length of each segment (e.g. 16)
segments         = batch_size // bptt_horizon  (e.g. 4096)
obs_shape        = single_observation_space.shape  (e.g. (87,) or (84, 84))
atn_shape        = single_action_space.shape       (e.g. () for Discrete)
```

### Experience Buffers (allocated in `__init__`)

| Buffer         | Shape                              | Dtype     | Description |
|----------------|-------------------------------------|-----------|-------------|
| `observations` | `[segments, horizon, *obs_shape]`   | obs dtype | Stacked observations |
| `actions`      | `[segments, horizon, *atn_shape]`   | atn dtype | Actions taken |
| `values`       | `[segments, horizon]`               | float32   | V(s) predictions |
| `logprobs`     | `[segments, horizon]`               | float32   | log π(a|s) |
| `rewards`      | `[segments, horizon]`               | float32   | Clipped rewards (±1) |
| `terminals`    | `[segments, horizon]`               | float32   | Episode done flags |
| `truncations`  | `[segments, horizon]`               | float32   | Episode truncation flags |
| `ratio`        | `[segments, horizon]`               | float32   | Importance sampling ratio (init=1) |
| `importance`   | `[segments, horizon]`               | float32   | For VTrace clipping (init=1) |
| `ep_lengths`   | `[total_agents]`                    | int32     | Current episode length tracker |
| `ep_indices`   | `[total_agents]`                    | int32     | Maps agent → segment row |

### LSTM State (if `use_rnn=True`)

| Buffer   | Shape                | Description |
|----------|----------------------|-------------|
| `lstm_h` | dict: `{start_idx: [agents_per_batch, hidden_size]}` | Hidden states keyed by env batch offset |
| `lstm_c` | dict: `{start_idx: [agents_per_batch, hidden_size]}` | Cell states keyed by env batch offset |

---

## 2. Evaluate Phase (Rollout Collection)

```python
def evaluate(self):
    # Reset trackers
    full_rows = 0
    free_idx = total_agents

    while full_rows < segments:
        # ── Recv from vectorized env ──
        # o: [agents_per_batch, *obs_shape]    (numpy)
        # r: [agents_per_batch]                (numpy float32)
        # d: [agents_per_batch]                (numpy bool → terminal)
        # t: [agents_per_batch]                (numpy bool → truncation)
        # info: list of dicts
        # env_id: slice(start, stop)
        # mask: [agents_per_batch] bool
        o, r, d, t, info, env_id, mask = vecenv.recv()

        global_step += mask.sum()

        # ── Move to device ──
        o_device = torch.as_tensor(o).to(device)          # [batch, *obs_shape]
        r_device = torch.as_tensor(r).to(device)          # [batch]
        d_device = torch.as_tensor(d).to(device)          # [batch]

        # ── Policy forward (no grad) ──
        with torch.no_grad(), amp_context:
            state = {reward: r, done: d, env_id: env_id, mask: mask}
            if use_rnn:
                state['lstm_h'] = lstm_h[env_id.start]    # [batch, hidden]
                state['lstm_c'] = lstm_c[env_id.start]    # [batch, hidden]

            logits, value = policy.forward_eval(o_device, state)
            # logits: [batch, num_actions]  (or Normal distribution for continuous)
            # value:  [batch, 1] or [batch]

            action, logprob, _ = sample_logits(logits)
            # action:  [batch, *atn_shape]
            # logprob: [batch]

            r_device = torch.clamp(r_device, -1, 1)  # reward clipping

        # ── Store into experience buffer ──
        l = ep_lengths[env_id.start].item()  # current horizon position
        batch_rows = slice(...)  # maps env_id → segment indices

        observations[batch_rows, l] = o_device     # [num_envs_in_batch, *obs_shape]
        actions[batch_rows, l]      = action        # [num_envs_in_batch, *atn_shape]
        logprobs[batch_rows, l]     = logprob       # [num_envs_in_batch]
        rewards[batch_rows, l]      = r_device      # [num_envs_in_batch]
        terminals[batch_rows, l]    = d_device       # [num_envs_in_batch]
        values[batch_rows, l]       = value.flatten()# [num_envs_in_batch]

        ep_lengths[env_id] += 1

        # When a segment row is full (l+1 >= bptt_horizon):
        if l + 1 >= bptt_horizon:
            ep_indices[env_id] = free_idx + arange(num_in_batch)
            ep_lengths[env_id] = 0
            free_idx += num_in_batch
            full_rows += num_in_batch

        # ── Send actions back to env ──
        vecenv.send(action.cpu().numpy())
```

Key insight: agents fill segment rows one timestep at a time. When a row reaches `bptt_horizon`, it's "full" and a new row is started. This naturally handles variable-length episodes within a fixed BPTT window.

---

## 3. Train Phase (PPO Update)

```python
def train(self):
    # Config
    total_minibatches = update_epochs * batch_size / minibatch_size
    minibatch_segments = minibatch_size // bptt_horizon

    ratio[:] = 1  # Reset importance ratios

    for mb in range(total_minibatches):

        # ── 1. Compute advantages over ALL segments ──
        # advantages: [segments, horizon]
        advantages = compute_puff_advantage(
            values,        # [segments, horizon]
            rewards,       # [segments, horizon]
            terminals,     # [segments, horizon]
            ratio,         # [segments, horizon]  (importance sampling ratio)
            advantages,    # [segments, horizon]  (output buffer)
            gamma,         # float (e.g. 0.99)
            gae_lambda,    # float (e.g. 0.95)
            vtrace_rho_clip,  # float (e.g. 1.0, inf=standard GAE)
            vtrace_c_clip     # float (e.g. 1.0, inf=standard GAE)
        )

        # ── 2. Prioritized sampling ──
        # Sum advantage magnitude per segment row
        adv = advantages.abs().sum(axis=1)        # [segments]
        prio_weights = adv ** alpha                 # [segments]
        prio_probs = prio_weights / prio_weights.sum()  # [segments]

        # Sample minibatch_segments segment rows proportional to advantage
        idx = torch.multinomial(prio_probs, minibatch_segments)  # [mb_segments]

        # Importance sampling correction (annealed beta)
        mb_prio = (segments * prio_probs[idx, None]) ** -anneal_beta
        # mb_prio: [mb_segments, 1]  (broadcasts over horizon)

        # ── 3. Extract minibatch ──
        mb_obs       = observations[idx]    # [mb_segments, horizon, *obs_shape]
        mb_actions   = actions[idx]         # [mb_segments, horizon, *atn_shape]
        mb_logprobs  = logprobs[idx]        # [mb_segments, horizon]
        mb_values    = values[idx]          # [mb_segments, horizon]
        mb_returns   = advantages[idx] + mb_values  # [mb_segments, horizon]
        mb_advantages= advantages[idx]      # [mb_segments, horizon]

        # For non-RNN: flatten to [mb_segments * horizon, *obs_shape]
        if not use_rnn:
            mb_obs = mb_obs.reshape(-1, *obs_shape)

        # ── 4. Forward pass ──
        logits, newvalue = policy(mb_obs, state)
        # logits: [mb_size, num_actions]
        # newvalue: [mb_size, 1]
        actions, newlogprob, entropy = sample_logits(logits, action=mb_actions)
        # newlogprob: [mb_size] → reshaped to [mb_segments, horizon]
        # entropy: [mb_size]

        # ── 5. Compute ratio and update stored ratios ──
        newlogprob = newlogprob.reshape(mb_logprobs.shape)
        logratio = newlogprob - mb_logprobs
        ratio_mb = logratio.exp()              # [mb_segments, horizon]
        self.ratio[idx] = ratio_mb.detach()    # Store for next advantage computation

        # ── 6. Advantage normalization with priority weighting ──
        adv = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # ── 7. PPO clipped losses ──
        # Policy loss (clipped surrogate)
        pg_loss1 = -adv * ratio_mb
        pg_loss2 = -adv * torch.clamp(ratio_mb, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss (clipped)
        newvalue = newvalue.view(mb_returns.shape)
        v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
        v_loss = 0.5 * torch.max(
            (newvalue - mb_returns)**2,
            (v_clipped - mb_returns)**2
        ).mean()

        # Entropy bonus
        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

        # ── 8. Update stored values (for next advantage pass) ──
        values[idx] = newvalue.detach().float()

        # ── 9. Backward + optimizer step ──
        loss.backward()
        if (mb + 1) % accumulate_minibatches == 0:
            clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    # Cosine LR schedule step
    if anneal_lr:
        scheduler.step()
```

---

## 4. Full Training Loop

```python
trainer = PuffeRL(config, vecenv, policy)

while trainer.epoch < trainer.total_epochs:
    stats = trainer.evaluate()   # Collect rollout, fill buffers
    logs = trainer.train()       # PPO update on collected data
    # Dashboard/checkpoints happen inside train()

trainer.close()
```

---

## 5. Key Config Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | auto | Total transitions per update |
| `bptt_horizon` | auto | Temporal window per segment |
| `update_epochs` | 4 | PPO epochs per batch |
| `minibatch_size` | - | Minibatch for gradient computation |
| `learning_rate` | 0.03 | LR (often higher with Muon) |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `clip_coef` | 0.2 | PPO clip coefficient |
| `vf_coef` | 0.5 | Value function loss weight |
| `ent_coef` | 0.01 | Entropy bonus weight |
| `vtrace_rho_clip` | inf | VTrace rho clip (inf = GAE) |
| `vtrace_c_clip` | inf | VTrace c clip (inf = GAE) |
| `prio_alpha` | 0.0 | Prioritization exponent (0 = uniform) |
| `prio_beta0` | 0.4 | IS correction initial beta |
| `optimizer` | `"muon"` | `"adam"` or `"muon"` |
| `compile` | True | `torch.compile` the policy |
| `precision` | `"bfloat16"` | AMP precision |
| `cpu_offload` | False | Pin obs memory on CPU |
