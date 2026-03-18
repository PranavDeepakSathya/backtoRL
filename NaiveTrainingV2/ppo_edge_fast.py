"""
PPO driver for FastEdgeTransformerPolicy.
Same structure as ppo_edge.py but wired to the fast policy and compiled with
mode='max-autotune'.  Also enables TF32 for any remaining float32 matmuls.
"""

from ppo3 import (
    CFG, compute_gae, collect_rollout, ppo_update,
    save_checkpoint, run_episodes,
)
from edge_transformer_policy_fast import FastEdgeTransformerPolicy

import os
import time
import numpy as np
import torch
from torch.optim import Adam
from turan_env_c import CEnv

# Allow TF32 for float32 matmuls (e.g. optimizer internals, GAE).
# ~3× faster than full FP32, negligible accuracy difference.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True


def train(cfg=CFG, resume=None):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    device = torch.device(cfg["device"])

    env    = CEnv(n=cfg["n"], num_envs=cfg["num_envs"], checker_id=cfg["checker_id"])
    policy = FastEdgeTransformerPolicy(
        d_model  = cfg.get("d_model",  64),
        n_heads  = cfg.get("n_heads",   4),
        n_layers = cfg.get("n_layers",  2),
    ).to(device)
    policy    = torch.compile(policy, mode="max-autotune")
    optimizer = Adam(policy.parameters(), lr=cfg["lr"])

    start_iter    = 1
    global_step   = 0
    best_mean_ret = -np.inf
    history       = []

    if resume:
        caller_cfg = cfg
        policy, optimizer, cfg, start_iter, global_step, best_mean_ret = \
            load_checkpoint(resume, device=str(device))
        start_iter += 1
        cfg.update(caller_cfg)

    steps_per_iter = cfg["n_steps"] * cfg["num_envs"]
    n_iters        = cfg["total_steps"] // steps_per_iter

    if start_iter > n_iters:
        raise ValueError(
            f"Resume: start_iter={start_iter} > n_iters={n_iters}. "
            f"Checkpoint already has {global_step:,} steps but "
            f"cfg['total_steps']={cfg['total_steps']:,}. "
            f"Set cfg['total_steps'] > {global_step:,} before calling train()."
        )

    print(f"\nPPO (fast)  n={cfg['n']}  envs={cfg['num_envs']}  checker={cfg['checker_id']}  device={cfg['device']}")
    print(f"transitions/update: {steps_per_iter:,}   total iters: {n_iters:,}" +
          (f"  resuming from iter {start_iter}" if resume else ""))
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

        flat  = lambda t: t.reshape(-1, *t.shape[2:])
        stats = ppo_update(policy, optimizer,
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


def train_multi_n_random(train_ns, policy, optimizer, cfg, n_cycles=3, global_step=0):
    """
    Curriculum training over multiple graph sizes with random cycle order.
    Same as ppo_edge.train_multi_n_random but wired to FastEdgeTransformerPolicy.
    """
    device         = torch.device(cfg['device'])
    steps_per_iter = cfg['n_steps'] * cfg['num_envs']
    log_interval   = cfg.get('log_interval', 50)
    save_interval  = cfg.get('save_interval', 200)
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)

    history    = []
    best_per_n = {n: -np.inf for n in train_ns}
    rng        = np.random.default_rng(seed=cfg.get('seed', 0))
    t_start    = time.time()

    for cycle in range(1, n_cycles + 1):
        order = rng.permutation(train_ns).tolist()
        print(f"\n══ Cycle {cycle}/{n_cycles}  order: {order} ══")

        for n in order:
            env     = CEnv(n=n, num_envs=cfg['num_envs'], checker_id=cfg['checker_id'])
            n_iters = cfg['total_steps_per_n'] // steps_per_iter
            bound   = n * n // 4

            print(f"\n── n={n}  iters={n_iters:,}  ref_bound~{bound} ──")

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

                if iteration % save_interval == 0:
                    save_checkpoint(
                        f"{cfg['checkpoint_dir']}/ckpt_{global_step//1000}k.pt",
                        policy, optimizer, {**cfg, 'n': n},
                        iteration, global_step, best_per_n[n])

                if iteration % log_interval == 0:
                    print(f"  iter {iteration:4d} | steps {global_step/1e6:.2f}M | "
                          f"fps {fps:5,.0f} | ret {mean_ret:.2f}/~{bound} | "
                          f"best {best_per_n[n]:.2f} | "
                          f"pg {stats['pg']:.4f} | ent {stats['ent']:.3f} | "
                          f"t {time.time()-t_start:.0f}s")

            env.close()

    save_checkpoint(f"{cfg['checkpoint_dir']}/final.pt",
                    policy, optimizer, cfg,
                    n_cycles * sum(cfg['total_steps_per_n'] // steps_per_iter
                                   for _ in train_ns),
                    global_step, max(best_per_n.values()))
    print(f"\nDone. best per n: { {k: f'{v:.2f}' for k, v in best_per_n.items()} }")
    return policy, history


def load_checkpoint(path, device=None):
    ckpt      = torch.load(path, map_location='cpu')
    cfg       = ckpt['cfg']
    device    = device or cfg.get('device', 'cuda')
    policy    = FastEdgeTransformerPolicy(
        d_model  = cfg.get('d_model',  64),
        n_heads  = cfg.get('n_heads',   4),
        n_layers = cfg.get('n_layers',  2),
    ).to(device)
    optimizer = Adam(policy.parameters(), lr=cfg['lr'])
    policy.load_state_dict(ckpt['policy'])
    optimizer.load_state_dict(ckpt['optimizer'])
    policy = torch.compile(policy, mode="max-autotune")
    print(f"loaded {path}  iter={ckpt['iteration']}  steps={ckpt['global_step']:,}  best={ckpt['best_mean_ret']:.2f}")
    return policy, optimizer, cfg, ckpt['iteration'], ckpt['global_step'], ckpt['best_mean_ret']
