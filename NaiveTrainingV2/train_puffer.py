"""
PuffeRL training script for TuranPufferEnv.
Run: python train_puffer.py [--n 20] [--envs 1024] [--checker 0] [--steps 20000000]
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib.pufferl import PuffeRL

from turan_puffer_env import TuranPufferEnv
from turan_env_c import CHECKER_C3, CHECKER_C4, CHECKER_K4

CHECKER_NAMES = {CHECKER_C3: 'C3', CHECKER_C4: 'C4', CHECKER_K4: 'K4'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n',        type=int, default=20)
    p.add_argument('--envs',     type=int, default=1024)
    p.add_argument('--checker',  type=int, default=CHECKER_C3,
                   help=f'checker id: C3={CHECKER_C3} C4={CHECKER_C4} K4={CHECKER_K4}')
    p.add_argument('--steps',    type=int, default=20_000_000)
    p.add_argument('--lr',       type=float, default=3e-4)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--hidden',   type=int,   default=256)
    p.add_argument('--ckpt-dir', type=str,   default=None,
                   help='checkpoint dir (default: ./checkpoints_puffer_{checker}_{n})')
    return p.parse_args()


def main():
    args   = parse_args()
    N      = args.n
    checker_name = CHECKER_NAMES.get(args.checker, str(args.checker))
    ckpt_dir = args.ckpt_dir or f'./checkpoints_puffer_{checker_name}_n{N}'
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"pufferlib {pufferlib.__version__}  torch {torch.__version__}")
    print(f"cuda={torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    print(f"n={N}  envs={args.envs}  checker={checker_name}  ckpt={ckpt_dir}")

    vecenv = pufferlib.vector.Serial(
        env_creators = [TuranPufferEnv],
        env_args     = [[]],
        env_kwargs   = [{'n': N, 'num_envs': args.envs, 'checker_id': args.checker}],
        num_envs     = 1,
    )

    policy = pufferlib.models.Default(
        env         = vecenv,
        hidden_size = args.hidden,
    ).to('cuda')

    print(f"params: {sum(p.numel() for p in policy.parameters()):,}")

    config = dict(
        env                = f'turan_{checker_name}_n{N}',
        total_timesteps    = args.steps,
        batch_size         = 'auto',
        bptt_horizon       = 64,
        update_epochs      = 10,
        minibatch_size     = 2048,
        max_minibatch_size = 2048,
        clip_coef          = 0.2,
        vf_clip_coef       = 10.0,
        vf_coef            = 0.5,
        ent_coef           = args.ent_coef,
        gamma              = 0.99,
        gae_lambda         = 0.95,
        vtrace_rho_clip    = 1.0,
        vtrace_c_clip      = 1.0,
        prio_alpha         = 0.0,
        prio_beta0         = 1.0,
        optimizer          = 'adam',
        learning_rate      = args.lr,
        adam_beta1         = 0.9,
        adam_beta2         = 0.999,
        adam_eps           = 1e-8,
        max_grad_norm      = 0.5,
        anneal_lr          = False,
        device             = 'cuda',
        precision          = 'bfloat16',
        cpu_offload        = False,
        use_rnn            = False,
        compile            = True,
        compile_mode       = 'default',
        compile_fullgraph  = False,
        torch_deterministic= False,
        seed               = 1,
        checkpoint_interval = 100,
        checkpoint_path    = ckpt_dir,
        run_name           = f'turan_{checker_name}_n{N}',
        run_dir            = ckpt_dir,
        data_dir           = ckpt_dir,
    )

    trainer  = PuffeRL(config=config, vecenv=vecenv, policy=policy)
    t_start  = time.time()
    log_every = 100   # log every N trainer.train() calls that return logs

    log_count = 0
    while trainer.global_step < config['total_timesteps']:
        trainer.evaluate()
        logs = trainer.train()
        if logs:
            log_count += 1
            if log_count % log_every == 0:
                elapsed = time.time() - t_start
                step_m  = trainer.global_step / 1e6
                ret     = logs.get('environment/episode_return', float('nan'))
                ploss   = logs.get('losses/policy_loss',         float('nan'))
                vloss   = logs.get('losses/value_loss',          float('nan'))
                ent     = logs.get('losses/entropy',             float('nan'))
                clip    = logs.get('losses/clipfrac',            float('nan'))
                print(f"steps {step_m:.2f}M | sps {trainer.sps:,.0f} | "
                      f"ret {ret:.3f} | pg {ploss:.4f} | vf {vloss:.3f} | "
                      f"ent {ent:.3f} | clip {clip:.3f} | t {elapsed:.0f}s")

    print(f"\nTraining done in {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
  