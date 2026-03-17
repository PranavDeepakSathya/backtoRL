import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from turan_env_c import CEnv, CHECKER_C4
from policy import MLPActorCritic
import os
import time


CFG = dict(
    n             = 20,
    num_envs      = 1024,
    checker_id    = CHECKER_C4,
    n_steps       = 64,
    n_epochs      = 10,
    batch_size    = 2048,
    lr            = 3e-4,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_eps      = 0.2,
    vf_coef       = 0.5,
    ent_coef      = 0.01,
    max_grad_norm = 0.5,
    total_steps   = 20_000_000,
    device        = "cuda",
    log_interval  = 10,
    save_interval = 100,
    checkpoint_dir= "./checkpoints",
)


def compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
    n_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae   = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(n_steps)):
        next_value    = last_value if t == n_steps - 1 else values[t + 1]
        next_non_done = 1.0 - dones[t]
        delta         = rewards[t] + gamma * next_value * next_non_done - values[t]
        last_gae      = delta + gamma * gae_lambda * next_non_done * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def collect_rollout(env, policy, n_steps, device):
    num_envs = env.num_envs
    obs_dim  = env.num_actions
    # store obs as bool — 4x less memory than float32
    obs_buf  = torch.zeros(n_steps, num_envs, obs_dim, device=device, dtype=torch.bool)
    act_buf  = torch.zeros(n_steps, num_envs,          device=device, dtype=torch.long)
    rew_buf  = torch.zeros(n_steps, num_envs,          device=device)
    done_buf = torch.zeros(n_steps, num_envs,          device=device)
    val_buf  = torch.zeros(n_steps, num_envs,          device=device)
    logp_buf = torch.zeros(n_steps, num_envs,          device=device)

    obs = torch.from_numpy(env.reset()).to(device)  # numpy bool → cuda bool
    for t in range(n_steps):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            action, log_prob, _, value = policy.get_action(obs.bfloat16())
        next_obs_np, reward, done = env.step(action.cpu().numpy())
        obs_buf[t]  = obs
        act_buf[t]  = action
        rew_buf[t]  = torch.tensor(reward, device=device)
        done_buf[t] = torch.tensor(done.astype(np.float32), device=device)
        val_buf[t]  = value.float()
        logp_buf[t] = log_prob.float()
        obs = torch.from_numpy(next_obs_np).to(device)

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        _, last_value = policy(obs.bfloat16())
    return obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_value.float()


def ppo_update(policy, optimizer, obs_buf, act_buf, adv_buf, ret_buf, logp_old_buf, cfg):
    n        = obs_buf.shape[0]
    clip_eps = cfg["clip_eps"]
    adv      = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
    pg_losses, vf_losses, ent_losses, clip_fracs = [], [], [], []

    for _ in range(cfg["n_epochs"]):
        perm = torch.randperm(n, device=obs_buf.device)
        for start in range(0, n, cfg["batch_size"]):
            idx = perm[start:start + cfg["batch_size"]]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                log_prob, entropy, value = policy.evaluate(obs_buf[idx].bfloat16(), act_buf[idx])
            # cast to float32 before numerically sensitive ops
            log_prob = log_prob.float()
            entropy  = entropy.float()
            value    = value.float()
            ratio    = (log_prob - logp_old_buf[idx]).exp()
            pg_loss  = torch.max(
                         -adv[idx] * ratio,
                         -adv[idx] * ratio.clamp(1 - clip_eps, 1 + clip_eps)
                       ).mean()
            vf_loss  = (value - ret_buf[idx]).pow(2).mean()
            loss     = pg_loss + cfg["vf_coef"] * vf_loss - cfg["ent_coef"] * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(entropy.mean().item())
            clip_fracs.append(((ratio - 1).abs() > clip_eps).float().mean().item())

    return dict(pg=np.mean(pg_losses), vf=np.mean(vf_losses),
                ent=np.mean(ent_losses), clip=np.mean(clip_fracs))


def save_checkpoint(path, policy, optimizer, cfg, iteration, global_step, best_mean_ret):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # save the underlying module state dict (works with compiled or uncompiled policy)
    state_dict = policy._orig_mod.state_dict() if hasattr(policy, '_orig_mod') else policy.state_dict()
    torch.save({
        'policy':        state_dict,
        'optimizer':     optimizer.state_dict(),
        'cfg':           cfg,
        'iteration':     iteration,
        'global_step':   global_step,
        'best_mean_ret': best_mean_ret,
    }, path)


def load_checkpoint(path, device=None):
    ckpt      = torch.load(path, map_location='cpu')
    cfg       = ckpt['cfg']
    device    = device or cfg.get('device', 'cuda')
    n         = cfg['n']
    num_act   = n * (n - 1) // 2
    policy    = MLPActorCritic(num_act, num_act).to(device)
    optimizer = Adam(policy.parameters(), lr=cfg['lr'])
    policy.load_state_dict(ckpt['policy'])
    optimizer.load_state_dict(ckpt['optimizer'])
    policy = torch.compile(policy)
    print(f"loaded {path}  iter={ckpt['iteration']}  steps={ckpt['global_step']:,}  best={ckpt['best_mean_ret']:.2f}")
    return policy, optimizer, cfg, ckpt['iteration'], ckpt['global_step'], ckpt['best_mean_ret']


def run_episodes(policy, cfg, n_episodes=1, deterministic=True):
    n      = cfg['n']
    device = cfg['device']
    env    = CEnv(n=n, num_envs=n_episodes, checker_id=cfg['checker_id'])
    obs    = env.reset()  # numpy bool
    done_all = np.zeros(n_episodes, dtype=bool)
    last_obs = np.zeros((n_episodes, env.num_actions), dtype=np.float32)
    max_steps = env.num_actions + 10
    step = 0
    while not done_all.all() and step < max_steps:
        obs_t = torch.from_numpy(obs).to(device)  # bool tensor
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = policy(obs_t.bfloat16())
        logits = logits.float()
        logits[obs_t] = -1e9  # mask already-added edges
        action = logits.argmax(dim=-1) if deterministic else \
                 torch.distributions.Categorical(logits=logits).sample()
        obs, _, done_arr = env.step(action.cpu().numpy().astype(np.int32))
        newly_done = done_arr & ~done_all
        for e in np.where(newly_done)[0]:
            last_obs[e] = obs_t[e].cpu().numpy().astype(np.float32)
        done_all |= done_arr
        step += 1
    for e in np.where(~done_all)[0]:
        last_obs[e] = obs_t[e].cpu().numpy().astype(np.float32)
    env.close()
    # reconstruct symmetric n×n adjacency from upper-triangle obs
    us, vs = np.triu_indices(n, k=1)
    adjs = []
    for e in range(n_episodes):
        adj = np.zeros((n, n), dtype=np.float32)
        adj[us, vs] = last_obs[e]
        adj[vs, us] = last_obs[e]
        adjs.append(adj)
    return adjs


def train(cfg=CFG, resume=None):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    env    = CEnv(n=cfg["n"], num_envs=cfg["num_envs"], checker_id=cfg["checker_id"])
    policy = MLPActorCritic(env.num_actions, env.num_actions).to(device)
    policy = torch.compile(policy)
    optimizer = Adam(policy.parameters(), lr=cfg["lr"])

    start_iter    = 1
    global_step   = 0
    best_mean_ret = -np.inf
    history       = []

    if resume:
        caller_cfg = cfg   # save caller's values before load_checkpoint overwrites cfg
        policy, optimizer, cfg, start_iter, global_step, best_mean_ret = \
            load_checkpoint(resume, device=str(device))
        start_iter += 1
        # apply every key the caller explicitly set so notebook overrides actually work
        cfg.update(caller_cfg)

    steps_per_iter = cfg["n_steps"] * cfg["num_envs"]
    n_iters        = cfg["total_steps"] // steps_per_iter

    if start_iter > n_iters:
        raise ValueError(
            f"Resume: start_iter={start_iter} > n_iters={n_iters}. "
            f"Checkpoint already has {global_step:,} steps but cfg['total_steps']={cfg['total_steps']:,}. "
            f"Set cfg['total_steps'] > {global_step:,} before calling train()."
        )

    print(f"\nPPO  n={cfg['n']}  envs={cfg['num_envs']}  checker={cfg['checker_id']}  device={cfg['device']}")
    print(f"transitions/update: {steps_per_iter:,}   total iters: {n_iters:,}  resuming from iter {start_iter}"
          if resume else
          f"transitions/update: {steps_per_iter:,}   total iters: {n_iters:,}")
    print(f"params: {sum(p.numel() for p in policy.parameters()):,}\n")
    env.benchmark(steps=200)

    t_start = time.time()

    for iteration in range(start_iter, n_iters + 1):
        t_iter = time.time()
        policy.train()

        obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_val = \
            collect_rollout(env, policy, cfg["n_steps"], device)

        advantages, returns = compute_gae(
            rew_buf, done_buf, val_buf, last_val, cfg["gamma"], cfg["gae_lambda"])

        flat   = lambda t: t.reshape(-1, *t.shape[2:])
        stats  = ppo_update(policy, optimizer,
                             flat(obs_buf), flat(act_buf),
                             flat(advantages), flat(returns), flat(logp_buf), cfg)

        global_step  += steps_per_iter
        mean_ret      = returns.mean().item()
        fps           = steps_per_iter / (time.time() - t_iter)

        history.append(dict(iteration=iteration, global_step=global_step,
                            mean_ret=mean_ret, fps=fps, **stats))

        if mean_ret > best_mean_ret:
            best_mean_ret = mean_ret
            save_checkpoint(
                f"{cfg['checkpoint_dir']}/best.pt",
                policy, optimizer, cfg, iteration, global_step, best_mean_ret)

        if iteration % cfg["save_interval"] == 0:
            save_checkpoint(
                f"{cfg['checkpoint_dir']}/ckpt_{global_step//1000}k.pt",
                policy, optimizer, cfg, iteration, global_step, best_mean_ret)

        if iteration % cfg["log_interval"] == 0:
            print(f"iter {iteration:5d} | steps {global_step/1e6:.2f}M | "
                  f"fps {fps:6,.0f} | ret {mean_ret:.2f} | best {best_mean_ret:.2f} | "
                  f"pg {stats['pg']:.4f} | vf {stats['vf']:.3f} | "
                  f"ent {stats['ent']:.3f} | clip {stats['clip']:.3f} | "
                  f"t {time.time()-t_start:.0f}s")

    save_checkpoint(f"{cfg['checkpoint_dir']}/final.pt",
                    policy, optimizer, cfg, n_iters, global_step, best_mean_ret)
    print(f"\ndone. best return: {best_mean_ret:.2f}")
    env.close()
    return policy, history


def train_multi_n(n_values, policy, optimizer, cfg, n_cycles=1, global_step=0):
    """
    Multi-n curriculum training with a pre-built policy.
    Iterates through n_values n_cycles times; all share the same policy and
    optimizer, forcing cross-size generalisation.  Pass global_step>0 when
    resuming so step counts and checkpoint names stay consistent.
    Returns (policy, history).
    """
    device = torch.device(cfg['device'])
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

    steps_per_iter = cfg['n_steps'] * cfg['num_envs']
    log_interval   = cfg.get('log_interval', 50)
    history        = []
    best_per_n     = {n: -np.inf for n in n_values}
    t_start        = time.time()

    for cycle in range(1, n_cycles + 1):
        for n in n_values:
            env     = CEnv(n=n, num_envs=cfg['num_envs'], checker_id=cfg['checker_id'])
            n_iters = cfg['total_steps_per_n'] // steps_per_iter
            bound   = n * n // 4   # Turán/Mantel bound ex(n, C3) = floor(n²/4)
            cycle_tag = f"cycle {cycle}/{n_cycles}  " if n_cycles > 1 else ""
            print(f"\n── {cycle_tag}n={n}  iters={n_iters:,}  bound={bound} ──")

            for iteration in range(1, n_iters + 1):
                t_iter = time.time()
                policy.train()

                obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_val = \
                    collect_rollout(env, policy, cfg['n_steps'], device)

                advantages, returns = compute_gae(
                    rew_buf, done_buf, val_buf, last_val, cfg['gamma'], cfg['gae_lambda'])

                flat  = lambda t: t.reshape(-1, *t.shape[2:])
                stats = ppo_update(policy, optimizer,
                                   flat(obs_buf), flat(act_buf),
                                   flat(advantages), flat(returns), flat(logp_buf), cfg)

                global_step += steps_per_iter
                mean_ret     = returns.mean().item()
                fps          = steps_per_iter / (time.time() - t_iter)

                history.append(dict(n=n, cycle=cycle, iteration=iteration,
                                    global_step=global_step,
                                    mean_ret=mean_ret, fps=fps, **stats))

                if mean_ret > best_per_n[n]:
                    best_per_n[n] = mean_ret
                    save_checkpoint(
                        f"{cfg['checkpoint_dir']}/best_n{n}.pt",
                        policy, optimizer, {**cfg, 'n': n},
                        iteration, global_step, mean_ret)

                if iteration % log_interval == 0:
                    print(f"  iter {iteration:4d} | steps {global_step/1e6:.2f}M | "
                          f"fps {fps:5,.0f} | ret {mean_ret:.2f}/{bound} | "
                          f"best {best_per_n[n]:.2f} | "
                          f"pg {stats['pg']:.4f} | ent {stats['ent']:.3f} | "
                          f"t {time.time()-t_start:.0f}s")

            env.close()

    save_checkpoint(f"{cfg['checkpoint_dir']}/final.pt",
                    policy, optimizer, {**cfg, 'n': n_values[-1]},
                    n_cycles * sum(cfg['total_steps_per_n'] // steps_per_iter for _ in n_values),
                    global_step, max(best_per_n.values()))
    print(f"\nTraining done. best per n: {best_per_n}")
    return policy, history


if __name__ == "__main__":
    train()
