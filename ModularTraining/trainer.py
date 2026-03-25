"""
Modular PPO trainer — pufferl-style evaluate/train loop.

Usage:
    from trainer import PPOTrainer
    from turan_env_c import CEnv, CHECKER_C3
    from policies import make_policy

    env     = CEnv(n=20, num_envs=1024, checker_id=CHECKER_C3)
    policy  = make_policy("mlp", obs_dim=env.num_actions, action_dim=env.num_actions)
    trainer = PPOTrainer(env, policy, checkpoint_dir="./my_ckpts", run_name="c3_n20")
    trainer.fit(total_steps=20_000_000)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass, field
from profiler import NullProfiler


@dataclass
class PPOConfig:
    # rollout
    n_steps: int = 64
    # optimisation
    n_epochs: int = 10
    batch_size: int = 2048
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # training
    total_steps: int = 20_000_000
    device: str = "cuda"
    compile: bool = True
    # logging
    log_interval: int = 10
    save_interval: int = 100
    # checkpointing
    checkpoint_dir: str = "./checkpoints"
    run_name: str = "run"


class PPOTrainer:
    """Modular PPO: plug any CEnv + any policy with (get_action, evaluate, forward)."""

    def __init__(self, env, policy, config=None, profiler=None, **overrides):
        self.cfg = config or PPOConfig()
        # apply any keyword overrides
        for k, v in overrides.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
            else:
                raise ValueError(f"Unknown config key: {k}")

        self.env = env
        self.device = torch.device(self.cfg.device)
        self.policy = policy.to(self.device)
        if self.cfg.compile:
            self.policy = torch.compile(self.policy)
        self.optimizer = Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.profiler = profiler if profiler is not None else NullProfiler()

        self.global_step = 0
        self.iteration = 0
        self.best_mean_ret = -np.inf
        self.history = []

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)

    # ── GAE ──────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_gae(rewards, dones, values, last_value, gamma, gae_lambda):
        n_steps, num_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(num_envs, device=rewards.device)
        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else values[t + 1]
            next_non_done = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_non_done - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_done * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    # ── evaluate (collect rollout) ───────────────────────────────────────

    def evaluate(self):
        """Collect one rollout of n_steps. Returns buffers."""
        cfg = self.cfg
        env = self.env
        device = self.device
        n_steps = cfg.n_steps
        num_envs = env.num_envs
        obs_dim = env.num_actions
        prof = self.profiler

        obs_buf  = torch.zeros(n_steps, num_envs, obs_dim, device=device, dtype=torch.bool)
        act_buf  = torch.zeros(n_steps, num_envs, device=device, dtype=torch.long)
        rew_buf  = torch.zeros(n_steps, num_envs, device=device)
        done_buf = torch.zeros(n_steps, num_envs, device=device)
        val_buf  = torch.zeros(n_steps, num_envs, device=device)
        logp_buf = torch.zeros(n_steps, num_envs, device=device)

        with prof.span("rollout"):
            obs = torch.from_numpy(env.reset()).to(device)
            for t in range(n_steps):
                with prof.span("rollout/policy_forward"):
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        action, log_prob, _, value = self.policy.get_action(obs.bfloat16())
                with prof.span("rollout/env_step"):
                    next_obs_np, reward, done = env.step(action.cpu().numpy())
                obs_buf[t]  = obs
                act_buf[t]  = action
                rew_buf[t]  = torch.tensor(reward, device=device)
                done_buf[t] = torch.tensor(done.astype(np.float32), device=device)
                val_buf[t]  = value.float()
                logp_buf[t] = log_prob.float()
                with prof.span("rollout/to_device"):
                    obs = torch.from_numpy(next_obs_np).to(device)

            with prof.span("rollout/last_value"):
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    _, last_value = self.policy(obs.bfloat16())

        return obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_value.float()

    # ── train (PPO update) ───────────────────────────────────────────────

    def train_step(self, obs_buf, act_buf, adv_buf, ret_buf, logp_old_buf):
        """Run n_epochs of minibatch PPO on collected data."""
        cfg = self.cfg
        prof = self.profiler
        n = obs_buf.shape[0]
        adv = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        pg_losses, vf_losses, ent_losses, clip_fracs = [], [], [], []

        with prof.span("train"):
            for _ in range(cfg.n_epochs):
                with prof.span("train/epoch"):
                    perm = torch.randperm(n, device=obs_buf.device)
                    for start in range(0, n, cfg.batch_size):
                        idx = perm[start:start + cfg.batch_size]
                        with prof.span("train/forward"):
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                log_prob, entropy, value = self.policy.evaluate(
                                    obs_buf[idx].bfloat16(), act_buf[idx])
                        with prof.span("train/loss"):
                            log_prob = log_prob.float()
                            entropy = entropy.float()
                            value = value.float()
                            ratio = (log_prob - logp_old_buf[idx]).exp()
                            pg_loss = torch.max(
                                -adv[idx] * ratio,
                                -adv[idx] * ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps)
                            ).mean()
                            vf_loss = (value - ret_buf[idx]).pow(2).mean()
                            loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy.mean()
                        self.optimizer.zero_grad()
                        with prof.span("train/backward"):
                            loss.backward()
                        with prof.span("train/grad_clip"):
                            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                        with prof.span("train/optimizer_step"):
                            self.optimizer.step()
                        pg_losses.append(pg_loss.item())
                        vf_losses.append(vf_loss.item())
                        ent_losses.append(entropy.mean().item())
                        clip_fracs.append(((ratio - 1).abs() > cfg.clip_eps).float().mean().item())

        return dict(pg=np.mean(pg_losses), vf=np.mean(vf_losses),
                    ent=np.mean(ent_losses), clip=np.mean(clip_fracs))

    # ── checkpointing ────────────────────────────────────────────────────

    def save(self, path):
        """Save checkpoint to path."""
        with self.profiler.span("save"):
            self._save_impl(path)

    def _save_impl(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state_dict = (self.policy._orig_mod.state_dict()
                      if hasattr(self.policy, '_orig_mod') else self.policy.state_dict())
        torch.save({
            'policy': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'iteration': self.iteration,
            'global_step': self.global_step,
            'best_mean_ret': self.best_mean_ret,
            'config': self.cfg.__dict__,
            'env_n': self.env.n,
            'env_checker_id': self.env.checker_id,
        }, path)

    def load(self, path):
        """Load checkpoint. Policy must already be created with correct architecture."""
        ckpt = torch.load(path, map_location='cpu')
        target = (self.policy._orig_mod if hasattr(self.policy, '_orig_mod')
                  else self.policy)
        target.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.iteration = ckpt['iteration']
        self.global_step = ckpt['global_step']
        self.best_mean_ret = ckpt['best_mean_ret']
        print(f"Loaded {path}  iter={self.iteration}  steps={self.global_step:,}  "
              f"best={self.best_mean_ret:.2f}")

    # ── main loop (pufferl-style) ────────────────────────────────────────

    def fit(self, total_steps=None, resume=None):
        """Run the full evaluate→train loop until total_steps."""
        if total_steps is not None:
            self.cfg.total_steps = total_steps
        if resume:
            self.load(resume)

        cfg = self.cfg
        steps_per_iter = cfg.n_steps * self.env.num_envs
        n_iters = cfg.total_steps // steps_per_iter
        start_iter = self.iteration + 1

        print(f"\n{'='*60}")
        print(f"PPO  n={self.env.n}  envs={self.env.num_envs}  "
              f"checker={self.env.checker_id}  device={cfg.device}")
        print(f"policy: {type(self.policy._orig_mod).__name__ if hasattr(self.policy, '_orig_mod') else type(self.policy).__name__}  "
              f"params: {sum(p.numel() for p in self.policy.parameters()):,}")
        print(f"transitions/update: {steps_per_iter:,}  total iters: {n_iters:,}")
        print(f"checkpoints: {cfg.checkpoint_dir}/{cfg.run_name}_*.pt")
        print(f"{'='*60}\n")

        self.env.benchmark(steps=200)
        t_start = time.time()

        prof = self.profiler
        for iteration in range(start_iter, n_iters + 1):
            with prof.span("iteration"):
                t_iter = time.time()
                self.iteration = iteration
                self.policy.train()

                # evaluate: collect rollout
                obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf, last_val = \
                    self.evaluate()

                # compute advantages
                with prof.span("gae"):
                    advantages, returns = self._compute_gae(
                        rew_buf, done_buf, val_buf, last_val, cfg.gamma, cfg.gae_lambda)

                # train: PPO update
                flat = lambda t: t.reshape(-1, *t.shape[2:])
                stats = self.train_step(
                    flat(obs_buf), flat(act_buf), flat(advantages),
                    flat(returns), flat(logp_buf))

                self.global_step += steps_per_iter
                mean_ret = returns.mean().item()
                fps = steps_per_iter / (time.time() - t_iter)

                self.history.append(dict(
                    iteration=iteration, global_step=self.global_step,
                    mean_ret=mean_ret, fps=fps, **stats))

                # save best
                if mean_ret > self.best_mean_ret:
                    self.best_mean_ret = mean_ret
                    self.save(f"{cfg.checkpoint_dir}/{cfg.run_name}_best.pt")

                # periodic save
                if iteration % cfg.save_interval == 0:
                    self.save(f"{cfg.checkpoint_dir}/{cfg.run_name}_{self.global_step // 1000}k.pt")

                # log
                if iteration % cfg.log_interval == 0:
                    print(f"iter {iteration:5d} | steps {self.global_step/1e6:.2f}M | "
                          f"fps {fps:6,.0f} | ret {mean_ret:.2f} | best {self.best_mean_ret:.2f} | "
                          f"pg {stats['pg']:.4f} | vf {stats['vf']:.3f} | "
                          f"ent {stats['ent']:.3f} | clip {stats['clip']:.3f} | "
                          f"t {time.time()-t_start:.0f}s")

        self.save(f"{cfg.checkpoint_dir}/{cfg.run_name}_final.pt")
        print(f"\nDone. best return: {self.best_mean_ret:.2f}")
        return self.history
