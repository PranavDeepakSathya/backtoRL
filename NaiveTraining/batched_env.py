import numpy as np
from stable_baselines3.common.vec_env import VecEnv
from gymnasium import spaces


class BatchedTuranEnv(VecEnv):
    """
    Fully vectorized Turán C4-free environment.
    All N envs live in one numpy array, stepped in a single call.
    No subprocess, no IPC, no Python loops in the hot path.

    adj shape:  (N, n, n)  int8
    obs shape:  (N, n*(n-1)//2)  float32
    """

    def __init__(self, n=20, num_envs=1024):
        self.n = n
        self.num_envs = num_envs
        self.num_actions = n * (n - 1) // 2
        self.STEP_COST = -0.01

        # precompute edge index <-> (u,v) mappings
        us, vs = np.triu_indices(n, k=1)
        self.edge_u = us.astype(np.int32)   # shape (num_actions,)
        self.edge_v = vs.astype(np.int32)   # shape (num_actions,)

        self.render_mode = None

        # all graph state in one array
        self.adj = np.zeros((num_envs, n, n), dtype=np.int8)
        self.edge_counts = np.zeros(num_envs, dtype=np.int32)

        obs_space = spaces.Box(0.0, 1.0, shape=(self.num_actions,), dtype=np.float32)
        act_space = spaces.Discrete(self.num_actions)

        super().__init__(num_envs, obs_space, act_space)

        # VecEnv requires these
        self.actions = None
        self.buf_obs = np.zeros((num_envs, self.num_actions), dtype=np.float32)
        self.buf_rews = np.zeros(num_envs, dtype=np.float32)
        self.buf_dones = np.zeros(num_envs, dtype=bool)
        self.buf_infos = [{} for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        self.adj[:] = 0
        self.edge_counts[:] = 0
        return self._obs_all()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        actions = self.actions
        n_envs = self.num_envs

        u = self.edge_u[actions]   # (N,)
        v = self.edge_v[actions]   # (N,)
        env_idx = np.arange(n_envs)

        # which envs are toggling an existing edge (remove)
        removing = self.adj[env_idx, u, v] == 1   # (N,) bool

        # for envs adding a new edge — check C4
        adding = ~removing

        # --- batched C4 check ---
        # C4 exists iff there is a path of length 3 between u and v:
        #   u -> w1 -> w2 -> v
        # i.e. some 2-hop neighbor of u is a direct neighbor of v
        row_u = self.adj[env_idx, u, :].astype(np.int32)   # (N, n)
        row_v = self.adj[env_idx, v, :].astype(np.int32)   # (N, n)

        # 2-hop neighbors of u via einsum: (N, n) x (N, n, n) -> (N, n)
        two_hop_u = np.einsum("bi,bij->bj", row_u, self.adj[env_idx].astype(np.int32))

        # exclude u and v themselves
        two_hop_u[env_idx, u] = 0
        two_hop_u[env_idx, v] = 0

        # overlap between 2-hop-from-u and direct-neighbors-of-v
        common = (two_hop_u * row_v).sum(axis=1)   # (N,)
        c4 = adding & (common > 0)                 # (N,) bool

        # --- apply transitions ---
        # removals
        self.adj[env_idx[removing], u[removing], v[removing]] = 0
        self.adj[env_idx[removing], v[removing], u[removing]] = 0
        self.edge_counts[removing] -= 1

        # safe additions (adding and no c4)
        safe = adding & ~c4
        self.adj[env_idx[safe], u[safe], v[safe]] = 1
        self.adj[env_idx[safe], v[safe], u[safe]] = 1
        self.edge_counts[safe] += 1

        # --- rewards ---
        rewards = np.full(n_envs, self.STEP_COST, dtype=np.float32)
        rewards[c4] = self.edge_counts[c4].astype(np.float32)

        # --- dones ---
        dones = c4.copy()

        # --- infos ---
        infos = [{"edges": self.edge_counts[i]} for i in range(n_envs)]
        for i in np.where(c4)[0]:
            infos[i]["c4"] = True

        # --- auto reset envs that are done ---
        if dones.any():
            self.adj[dones] = 0
            self.edge_counts[dones] = 0

        obs = self._obs_all()
        return obs, rewards, dones, infos

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        # SB3 calls this for render_mode during __init__
        indices = self._get_indices(indices)
        val = getattr(self, attr_name, None)
        return [val for _ in indices]

    def set_attr(self, attr_name, value, indices=None):
        indices = self._get_indices(indices)
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        indices = self._get_indices(indices)
        method = getattr(self, method_name)
        return [method(*method_args, **method_kwargs) for _ in indices]

    def seed(self, seed=None):
        return [None] * self.num_envs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _obs_all(self):
        # upper triangle of all envs at once
        # adj[:, us, vs] gives (N, num_actions) directly
        return self.adj[:, self.edge_u, self.edge_v].astype(np.float32)

    def turan_bound(self):
        n = self.n
        return int(0.5 * (1 + (n - 1) ** 0.5) * n)

    # ------------------------------------------------------------------
    # Sanity / benchmarking
    # ------------------------------------------------------------------

    def benchmark(self, steps=1000):
        import time
        self.reset()
        t0 = time.perf_counter()
        for _ in range(steps):
            actions = np.random.randint(0, self.num_actions, size=self.num_envs)
            self.step_async(actions)
            self.step_wait()
        dt = time.perf_counter() - t0
        sps = steps * self.num_envs / dt
        print(f"{steps} steps x {self.num_envs} envs in {dt:.2f}s")
        print(f"  {sps:,.0f} steps/sec")
        print(f"  {sps / 1e6:.2f}M steps/sec")
        return sps


if __name__ == "__main__":
    print("=== correctness check ===")
    env = BatchedTuranEnv(n=10, num_envs=4)
    obs = env.reset()
    print(f"obs shape:  {obs.shape}")
    print(f"num_actions: {env.num_actions}")
    print(f"turan bound: {env.turan_bound()}")

    # manually build C4 in env 0: edges 0-1, 1-2, 2-3, then 0-3 closes it
    # find action indices for these edges
    def edge_idx(env, u, v):
        mask = (env.edge_u == u) & (env.edge_v == v)
        return int(np.where(mask)[0])

    e = BatchedTuranEnv(n=10, num_envs=1)
    e.reset()
    for u, v in [(0,1),(1,2),(2,3)]:
        a = edge_idx(e, u, v)
        e.step_async(np.array([a]))
        obs, rew, done, info = e.step_wait()
        print(f"  add ({u},{v}): reward={rew[0]:+.2f} done={done[0]}")

    a = edge_idx(e, 0, 3)
    e.step_async(np.array([a]))
    obs, rew, done, info = e.step_wait()
    print(f"  add (0,3): reward={rew[0]:+.2f} done={done[0]} — should be terminal with reward=3")

    print("\n=== benchmark ===")
    bench_env = BatchedTuranEnv(n=20, num_envs=1024)
    bench_env.benchmark(steps=1000)