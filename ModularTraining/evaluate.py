"""
Inference & visualization helpers for trained Turan policies.

Usage (script):
    python evaluate.py --checkpoint checkpoints/my_run_best.pt \
                       --policy edge_transformer --n 20 --checker 0 \
                       --k 32 --show

Usage (import):
    from evaluate import load_policy, best_of_k, draw_graph

    policy = load_policy("checkpoints/best.pt", "edge_transformer",
                         d_model=64, n_heads=4, n_layers=2)
    obs, edges, graph = best_of_k(policy, n=20, checker_id=0, k=32)
    draw_graph(graph, title="C3-free, n=20")
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

from turan_env_c import CEnv, CHECKER_C3, CHECKER_C4, CHECKER_K4
from policies import make_policy

CHECKER_NAMES = {
    CHECKER_C3: "C3", CHECKER_C4: "C4", CHECKER_K4: "K4",
}


# ── load ──────────────────────────────────────────────────────────────────────

def load_policy(checkpoint_path, policy_name, device="cuda", **policy_kwargs):
    """Load a trained policy from a checkpoint file."""
    policy = make_policy(policy_name, **policy_kwargs)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(ckpt["policy"])
    policy = policy.to(device).eval()
    info = {k: ckpt.get(k) for k in
            ("iteration", "global_step", "best_mean_ret", "env_n", "env_checker_id")}
    print(f"Loaded {checkpoint_path}  step={info['global_step']:,}  "
          f"best_ret={info['best_mean_ret']:.2f}")
    return policy, info


# ── rollout ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout(policy, n, checker_id, num_envs=1, deterministic=True, device="cuda"):
    """Run one full episode per env. Returns (obs, edge_counts) arrays."""
    env = CEnv(n=n, num_envs=num_envs, checker_id=checker_id)
    num_actions = n * (n - 1) // 2
    obs = torch.from_numpy(env.reset()).to(device)
    edge_counts = np.zeros(num_envs, dtype=np.int32)
    alive = np.ones(num_envs, dtype=bool)
    # save last-good obs per env before reset wipes it
    saved_obs = np.zeros((num_envs, num_actions), dtype=np.uint8)

    max_steps = num_actions
    for _ in range(max_steps):
        if not alive.any():
            break
        # snapshot obs BEFORE stepping — this is the full graph so far
        saved_obs[alive] = obs.cpu().numpy()[alive]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action, _, _, _ = policy.get_action(obs.bfloat16(), deterministic=deterministic)
        obs_np, reward, done = env.step(action.cpu().numpy())
        # on terminal step, CEnv returns edge_count as reward then resets
        newly_done = done & alive
        if newly_done.any():
            edge_counts[newly_done] = reward[newly_done].astype(np.int32)
        alive &= ~done
        obs = torch.from_numpy(obs_np).to(device)

    # for envs still alive (never hit terminal), count from obs
    if alive.any():
        saved_obs[alive] = obs.cpu().numpy()[alive]
        edge_counts[alive] = saved_obs[alive].sum(axis=1).astype(np.int32)

    env.close()
    return saved_obs, edge_counts


def best_of_k(policy, n, checker_id, k=32, deterministic=True, device="cuda"):
    """
    Run k rollouts and return the one with the most edges.
    Returns (obs_vector, edge_count, nx.Graph).
    """
    all_obs, all_counts = rollout(
        policy, n, checker_id, num_envs=k, deterministic=deterministic, device=device)

    best_idx = np.argmax(all_counts)
    best_obs = all_obs[best_idx]
    best_count = all_counts[best_idx]

    print(f"best_of_{k}: {best_count} edges  "
          f"(min={all_counts.min()}, mean={all_counts.mean():.1f}, max={all_counts.max()})")

    G = obs_to_graph(best_obs, n)
    return best_obs, best_count, G


# ── graph helpers ─────────────────────────────────────────────────────────────

def obs_to_graph(obs_vec, n):
    """Convert a flat upper-triangle obs vector to a networkx Graph."""
    us, vs = np.triu_indices(n, k=1)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u, v, e in zip(us, vs, obs_vec):
        if e:
            G.add_edge(int(u), int(v))
    return G


def draw_graph(G, title=None, ax=None, seed=42):
    """Draw a networkx graph."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos, ax=ax, with_labels=True,
            node_color="#1D9E75", node_size=500,
            font_color="white", edge_color="#444", width=1.2)
    title = title or f"{G.number_of_nodes()} vertices, {G.number_of_edges()} edges"
    ax.set_title(title, fontsize=14)
    return ax


def compare_k(policy, n, checker_id, ks=(1, 8, 32, 128), device="cuda"):
    """Run best-of-k for several k values and print a summary table."""
    print(f"\n{'k':>6}  {'best':>6}  {'mean':>8}  {'min':>6}  {'max':>6}")
    print("-" * 42)
    for k in ks:
        _, counts = rollout(policy, n, checker_id, num_envs=k, device=device)
        print(f"{k:>6}  {counts.max():>6}  {counts.mean():>8.1f}  "
              f"{counts.min():>6}  {counts.max():>6}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained Turan policy")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--policy", type=str, default="edge_transformer",
                   choices=["mlp", "edge_transformer", "edge_transformer_deg"])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--checker", type=int, default=CHECKER_C3)
    p.add_argument("--k", type=int, default=32, help="number of rollouts for best-of-k")
    # policy arch params
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    # output
    p.add_argument("--show", action="store_true", help="display the graph")
    p.add_argument("--save", type=str, default=None, help="save graph image to path")
    p.add_argument("--compare", action="store_true",
                   help="run compare_k with k=1,8,32,128,512")
    p.add_argument("--stochastic", action="store_true",
                   help="use stochastic sampling instead of greedy")
    args = p.parse_args()

    # build policy kwargs
    if args.policy == "mlp":
        num_actions = args.n * (args.n - 1) // 2
        pkw = dict(obs_dim=num_actions, action_dim=num_actions, hidden=args.hidden)
    else:
        pkw = dict(d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)

    policy, info = load_policy(args.checkpoint, args.policy, **pkw)
    checker_name = CHECKER_NAMES.get(args.checker, str(args.checker))

    if args.compare:
        compare_k(policy, args.n, args.checker, ks=(1, 8, 32, 128, 512))

    _, edge_count, G = best_of_k(
        policy, args.n, args.checker, k=args.k,
        deterministic=not args.stochastic)

    title = f"{checker_name}-free  n={args.n}  edges={edge_count}"
    ax = draw_graph(G, title=title)

    if args.save:
        ax.figure.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to {args.save}")
    if args.show:
        plt.show()
    elif not args.save:
        plt.show()


if __name__ == "__main__":
    main()
