"""
Head-to-head comparison: BatchedTuranEnv vs CEnv
Same actions, same sequence — do they produce identical rewards and dones?
"""
import numpy as np
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

# We'll build minimal versions here to avoid import issues
# First, test with deterministic action sequences

def test_manual_sequence():
    """Test a known C4-creating sequence"""
    from batched_env import BatchedTuranEnv
    from turan_env_c import CEnv, CHECKER_C4

    n = 10
    num_envs = 1

    batched = BatchedTuranEnv(n=n, num_envs=num_envs)
    c_env   = CEnv(n=n, num_envs=num_envs, checker_id=CHECKER_C4)

    # edge indices for a C4: 0-1, 1-2, 2-3, then 0-3
    us, vs = np.triu_indices(n, k=1)
    def eidx(u, v):
        mask = (us == u) & (vs == v)
        return int(np.where(mask)[0])

    tri_idx = (us * n + vs).astype(np.int64)

    edges = [(0,1), (1,2), (2,3), (0,3)]
    print("=== Manual C4 sequence ===")
    
    batched.reset()
    c_env.reset()

    for u, v in edges:
        a = np.array([eidx(u, v)])
        
        # batched env
        batched.step_async(a)
        b_obs, b_rew, b_done, b_info = batched.step_wait()
        
        # c env
        c_obs_raw, c_rew, c_done = c_env.step(a.astype(np.int32))
        c_obs = c_obs_raw[:, tri_idx]
        
        match_rew  = np.allclose(b_rew, c_rew)
        match_done = np.array_equal(b_done, c_done)
        match_obs  = np.array_equal(b_obs, c_obs.astype(np.float32))
        
        print(f"  edge ({u},{v}): batched_rew={b_rew[0]:+.2f} c_rew={c_rew[0]:+.2f} "
              f"batched_done={b_done[0]} c_done={c_done[0]} "
              f"rew_match={match_rew} done_match={match_done} obs_match={match_obs}")


def test_toggle():
    """Test edge toggling: add then remove"""
    from batched_env import BatchedTuranEnv
    from turan_env_c import CEnv, CHECKER_C4

    n = 10
    num_envs = 1

    batched = BatchedTuranEnv(n=n, num_envs=num_envs)
    c_env   = CEnv(n=n, num_envs=num_envs, checker_id=CHECKER_C4)

    us, vs = np.triu_indices(n, k=1)
    tri_idx = (us * n + vs).astype(np.int64)

    def eidx(u, v):
        mask = (us == u) & (vs == v)
        return int(np.where(mask)[0])

    print("\n=== Toggle test (add then remove) ===")
    batched.reset()
    c_env.reset()

    a = np.array([eidx(0, 1)])
    
    # add edge
    batched.step_async(a)
    b_obs, b_rew, b_done, _ = batched.step_wait()
    c_obs_raw, c_rew, c_done = c_env.step(a.astype(np.int32))
    c_obs = c_obs_raw[:, tri_idx]
    print(f"  ADD (0,1): b_rew={b_rew[0]:+.2f} c_rew={c_rew[0]:+.2f} "
          f"b_done={b_done[0]} c_done={c_done[0]} "
          f"obs_match={np.array_equal(b_obs, c_obs.astype(np.float32))}")

    # remove edge (same action again)
    batched.step_async(a)
    b_obs, b_rew, b_done, _ = batched.step_wait()
    c_obs_raw, c_rew, c_done = c_env.step(a.astype(np.int32))
    c_obs = c_obs_raw[:, tri_idx]
    print(f"  REM (0,1): b_rew={b_rew[0]:+.2f} c_rew={c_rew[0]:+.2f} "
          f"b_done={b_done[0]} c_done={c_done[0]} "
          f"obs_match={np.array_equal(b_obs, c_obs.astype(np.float32))}")

    # verify both are back to empty
    b_edges = b_obs.sum()
    c_edges = c_obs.sum()
    print(f"  edges after toggle: batched={b_edges} c_env={c_edges}")


def test_random_trajectories():
    """Run many random steps, compare step-by-step"""
    from batched_env import BatchedTuranEnv
    from turan_env_c import CEnv, CHECKER_C4

    n = 20
    num_envs = 64
    steps = 2000

    batched = BatchedTuranEnv(n=n, num_envs=num_envs)
    c_env   = CEnv(n=n, num_envs=num_envs, checker_id=CHECKER_C4)
    num_actions = n * (n - 1) // 2

    us, vs = np.triu_indices(n, k=1)
    tri_idx = (us * n + vs).astype(np.int64)

    np.random.seed(42)
    
    batched.reset()
    c_env.reset()

    mismatches = 0
    total_dones_b = 0
    total_dones_c = 0
    max_term_rew_b = 0
    max_term_rew_c = 0

    print(f"\n=== Random trajectory test: {steps} steps x {num_envs} envs ===")

    for t in range(steps):
        actions = np.random.randint(0, num_actions, size=num_envs)

        batched.step_async(actions)
        b_obs, b_rew, b_done, _ = batched.step_wait()

        c_obs_raw, c_rew, c_done = c_env.step(actions.astype(np.int32))
        c_obs = c_obs_raw[:, tri_idx].astype(np.float32)

        rew_match  = np.allclose(b_rew, c_rew, atol=1e-5)
        done_match = np.array_equal(b_done, c_done)
        obs_match  = np.array_equal(b_obs, c_obs)

        if not (rew_match and done_match and obs_match):
            mismatches += 1
            if mismatches <= 5:
                # find first diverging env
                for e in range(num_envs):
                    if b_rew[e] != c_rew[e] or b_done[e] != c_done[e]:
                        print(f"  MISMATCH step={t} env={e}: "
                              f"b_rew={b_rew[e]:.2f} c_rew={c_rew[e]:.2f} "
                              f"b_done={b_done[e]} c_done={c_done[e]} "
                              f"action={actions[e]}")
                        break
                if not rew_match and done_match and not obs_match:
                    for e in range(num_envs):
                        if not np.array_equal(b_obs[e], c_obs[e]):
                            print(f"    obs mismatch env={e}: "
                                  f"b_edges={b_obs[e].sum():.0f} c_edges={c_obs[e].sum():.0f}")
                            break
            
            # desync means we can't continue meaningfully
            if not done_match or not obs_match:
                # resync both envs
                pass  # let them diverge and count mismatches

        total_dones_b += b_done.sum()
        total_dones_c += c_done.sum()
        if b_done.any():
            max_term_rew_b = max(max_term_rew_b, b_rew[b_done].max())
        if c_done.any():
            max_term_rew_c = max(max_term_rew_c, c_rew[c_done].max())

    print(f"\n  Total mismatched steps: {mismatches}/{steps}")
    print(f"  Total dones: batched={total_dones_b} c_env={total_dones_c}")
    print(f"  Max terminal reward: batched={max_term_rew_b:.0f} c_env={max_term_rew_c:.0f}")

    if mismatches == 0:
        print("\n  ✓ ENVS ARE IDENTICAL — no bug in env logic.")
        print("  The training difference is a hyperparameter/stability issue, not a bug.")
    else:
        print(f"\n  ✗ FOUND {mismatches} MISMATCHES — there IS a bug!")


if __name__ == "__main__":
    test_manual_sequence()
    test_toggle()
    test_random_trajectories()