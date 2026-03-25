#!/usr/bin/env python3
"""
Modular Turan RL training — plug any env checker + any policy.

Examples:
    # MLP on C3, n=20
    python train.py --policy mlp --checker 0 --n 20

    # Edge transformer on C4, n=30, custom checkpoint
    python train.py --policy edge_transformer --checker 1 --n 30 \
                    --ckpt-dir ./my_runs --run-name c4_transformer_n30

    # Resume from checkpoint
    python train.py --policy mlp --checker 0 --n 20 --resume ./my_runs/best.pt

    # All checkers: C3=0 C4=1 C3C4=2 K23=3 THETA123=4 BULL=5 BOWTIE=6 K4=7
"""

import argparse
from turan_env_c import (
    CEnv, CHECKER_C3, CHECKER_C4, CHECKER_C3C4,
    CHECKER_K23, CHECKER_THETA123, CHECKER_BULL, CHECKER_BOWTIE, CHECKER_K4,
)
from policies import make_policy
from trainer import PPOTrainer, PPOConfig
from profiler import PPOProfiler, NullProfiler

CHECKER_NAMES = {
    CHECKER_C3: 'C3', CHECKER_C4: 'C4', CHECKER_C3C4: 'C3C4',
    CHECKER_K23: 'K23', CHECKER_THETA123: 'THETA123',
    CHECKER_BULL: 'BULL', CHECKER_BOWTIE: 'BOWTIE', CHECKER_K4: 'K4',
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Modular PPO for Turan environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    # env
    p.add_argument('--n', type=int, default=20, help='graph order')
    p.add_argument('--envs', type=int, default=1024, help='number of parallel envs')
    p.add_argument('--checker', type=int, default=CHECKER_C3,
                   help='checker id (0=C3 1=C4 2=C3C4 3=K23 4=THETA123 5=BULL 6=BOWTIE 7=K4)')
    # policy
    p.add_argument('--policy', type=str, default='mlp',
                   choices=['mlp', 'edge_transformer', 'edge_transformer_deg'],
                   help='policy architecture')
    p.add_argument('--hidden', type=int, nargs='+', default=[256, 256],
                   help='MLP hidden layer sizes (ignored for transformer)')
    p.add_argument('--d-model', type=int, default=64, help='transformer d_model')
    p.add_argument('--n-heads', type=int, default=4, help='transformer heads')
    p.add_argument('--n-layers', type=int, default=2, help='transformer layers')
    # training
    p.add_argument('--steps', type=int, default=20_000_000, help='total training steps')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--n-steps', type=int, default=64, help='rollout horizon')
    p.add_argument('--batch-size', type=int, default=2048, help='minibatch size')
    p.add_argument('--no-compile', action='store_true', help='disable torch.compile')
    # checkpointing
    p.add_argument('--ckpt-dir', type=str, default=None,
                   help='checkpoint dir (default: ./checkpoints_{checker}_{policy}_n{N})')
    p.add_argument('--run-name', type=str, default=None,
                   help='run name for checkpoint files (default: {checker}_{policy}_n{N})')
    p.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume')
    # logging
    p.add_argument('--log-interval', type=int, default=10)
    p.add_argument('--save-interval', type=int, default=100)
    # profiling
    p.add_argument('--profile', action='store_true', help='enable PPO profiler (writes trace.json)')
    p.add_argument('--trace-path', type=str, default=None,
                   help='path for trace file (default: {ckpt_dir}/trace.json)')
    return p.parse_args()


def main():
    args = parse_args()
    checker_name = CHECKER_NAMES.get(args.checker, str(args.checker))
    tag = f"{checker_name}_{args.policy}_n{args.n}"

    # env
    env = CEnv(n=args.n, num_envs=args.envs, checker_id=args.checker)

    # policy
    obs_dim = action_dim = env.num_actions
    if args.policy == 'mlp':
        policy = make_policy('mlp', obs_dim=obs_dim, action_dim=action_dim,
                             hidden=args.hidden)
    elif args.policy in ('edge_transformer', 'edge_transformer_deg'):
        policy = make_policy(args.policy, d_model=args.d_model,
                             n_heads=args.n_heads, n_layers=args.n_layers)

    # config
    cfg = PPOConfig(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        ent_coef=args.ent_coef,
        total_steps=args.steps,
        compile=not args.no_compile,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.ckpt_dir or f"./checkpoints_{tag}",
        run_name=args.run_name or tag,
    )

    # profiler
    if args.profile:
        profiler = PPOProfiler()
        trace_path = args.trace_path or f"{cfg.checkpoint_dir}/trace.json"
    else:
        profiler = NullProfiler()
        trace_path = None

    # trainer
    trainer = PPOTrainer(env, policy, config=cfg, profiler=profiler)
    trainer.fit(resume=args.resume)
    if trace_path:
        profiler.save(trace_path)
    env.close()


if __name__ == '__main__':
    main()
