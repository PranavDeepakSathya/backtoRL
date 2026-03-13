import os
import warnings
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from turan_env import TuranEnv
from batched_env import BatchedTuranEnv

warnings.filterwarnings("ignore", message="Training and eval env are not of the same type")

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
CFG = dict(
    # env
    n               = 20,
    num_envs        = 1024,

    # training scale
    total_timesteps = 10_000_000,

    # PPO core
    learning_rate   = 3e-4,
    n_steps         = 64,          # 64 * 1024 = 65k transitions per update
    batch_size      = 512,
    n_epochs        = 10,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,

    # network
    net_arch        = [256, 256],

    # checkpointing
    checkpoint_dir  = "./checkpoints",
    checkpoint_freq = 1_000_000,
    eval_freq       = 500_000,
    n_eval_episodes = 50,
    device          = "cuda",
)
# ------------------------------------------------------------------


def train(cfg=CFG):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    train_env = BatchedTuranEnv(n=cfg["n"], num_envs=cfg["num_envs"])

    # eval env — same type as train env, small num_envs is fine
    eval_env = BatchedTuranEnv(n=cfg["n"], num_envs=32)

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(cfg["checkpoint_freq"] // cfg["num_envs"], 1),
        save_path   = cfg["checkpoint_dir"],
        name_prefix = f"turan_n{cfg['n']}",
        verbose     = 1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = cfg["checkpoint_dir"],
        log_path             = "./logs",
        eval_freq            = max(cfg["eval_freq"] // cfg["num_envs"], 1),
        n_eval_episodes      = cfg["n_eval_episodes"],
        deterministic        = True,
        verbose              = 1,
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = cfg["learning_rate"],
        n_steps         = cfg["n_steps"],
        batch_size      = cfg["batch_size"],
        n_epochs        = cfg["n_epochs"],
        gamma           = cfg["gamma"],
        gae_lambda      = cfg["gae_lambda"],
        clip_range      = cfg["clip_range"],
        ent_coef        = cfg["ent_coef"],
        vf_coef         = cfg["vf_coef"],
        max_grad_norm   = cfg["max_grad_norm"],
        policy_kwargs   = dict(net_arch=cfg["net_arch"]),
        tensorboard_log = "./logs",
        device          = cfg.get("device", "cuda"),
        verbose         = 1,
    )

    print(f"\nTraining  n={cfg['n']}  envs={cfg['num_envs']}  "
          f"steps={cfg['total_timesteps']:,}")
    print(f"transitions per update: {cfg['n_steps'] * cfg['num_envs']:,}\n")

    print("Benchmarking env...")
    train_env.benchmark(steps=200)

    model.learn(
        total_timesteps = cfg["total_timesteps"],
        callback        = CallbackList([checkpoint_cb, eval_cb]),
        progress_bar    = True,
    )

    final_path = os.path.join(cfg["checkpoint_dir"], f"turan_n{cfg['n']}_final")
    model.save(final_path)
    print(f"\nSaved → {final_path}.zip")

    train_env.close()
    eval_env.close()
    return model


def load(path, n=None):
    n = n or CFG["n"]
    env = TuranEnv(n=n)
    model = PPO.load(path, env=env)
    print(f"Loaded {path}")
    return model, env


if __name__ == "__main__":
    train()