import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TuranEnv(gym.Env):
    """
    Turán environment: build a C4-free graph on n vertices with as many edges as possible.

    Action:  integer in [0, n*(n-1)//2)  — index into upper triangle
             if edge absent  → add it, check C4
             if edge present → remove it

    Reward:  0 every step (tiny time penalty)
             +edge_count on terminal (C4 hit)
             un-hackable: oscillating add/remove nets 0
    """

    metadata = {}
    STEP_COST = -0.01

    def __init__(self, n=20):
        super().__init__()
        self.n = n
        self.num_actions = n * (n - 1) // 2

        self.idx_to_edge = [(u, v) for u in range(n) for v in range(u + 1, n)]
        self.edge_to_idx = {(u, v): i for i, (u, v) in enumerate(self.idx_to_edge)}
        self._triu_idx = np.triu_indices(n, k=1)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_actions)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.adj = np.zeros((self.n, self.n), dtype=np.int8)
        self.edge_count = 0
        self.done = False
        return self._obs(), {}

    def step(self, action):
        u, v = self.idx_to_edge[action]

        if self.adj[u, v] == 1:
            # remove edge — costs a step, no terminal
            self.adj[u, v] = 0
            self.adj[v, u] = 0
            self.edge_count -= 1
            return self._obs(), self.STEP_COST, False, False, {"removed": (u, v)}

        # try to add edge
        if self._creates_c4(u, v):
            # terminal — reward is however many edges we managed to build
            return self._obs(), float(self.edge_count), True, False, {"c4": True, "edges": self.edge_count}

        self.adj[u, v] = 1
        self.adj[v, u] = 1
        self.edge_count += 1
        return self._obs(), self.STEP_COST, False, False, {"edges": self.edge_count}

    def _creates_c4(self, u, v):
        neighbors_u = np.where(self.adj[u] == 1)[0]
        neighbors_v = np.where(self.adj[v] == 1)[0]
        for w1 in neighbors_u:
            if w1 == v:
                continue
            for w2 in neighbors_v:
                if w2 == u or w2 == w1:
                    continue
                if self.adj[w1, w2] == 1:
                    return True
        return False

    def _obs(self):
        return self.adj[self._triu_idx].astype(np.float32)

    def turan_bound(self):
        return int(0.5 * (1 + (self.n - 1) ** 0.5) * self.n)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _draw_adj(self, adj, ax, title=None, highlight_c4=None):
        """Core draw — works on any adjacency matrix."""
        import matplotlib.pyplot as plt
        n = self.n
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v] == 1:
                    color, lw, zo = "#888780", 1.0, 1
                    if highlight_c4:
                        hi = highlight_c4
                        pairs = set(zip(hi, hi[1:] + hi[:1]))
                        if u in hi and v in hi and ((u, v) in pairs or (v, u) in pairs):
                            color, lw, zo = "#E24B4A", 2.5, 2
                    ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]],
                            color=color, lw=lw, zorder=zo, solid_capstyle="round")

        for i in range(n):
            c = "#E24B4A" if (highlight_c4 and i in highlight_c4) else "#1D9E75"
            ax.add_patch(plt.Circle(pos[i], 0.08, color=c, zorder=3))
            ax.text(pos[i, 0], pos[i, 1], str(i),
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=4)

        ec = int(adj.sum()) // 2
        bound = self.turan_bound()
        pct = ec / bound * 100 if bound > 0 else 0
        ax.set_title(title or f"edges={ec}  bound≈{bound}  ({pct:.0f}%)", fontsize=9, pad=6)
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.axis("off")

    def draw(self, obs=None, ax=None, title=None, highlight_c4=None):
        """
        Draw current state, or pass an obs vector to draw that instead.

            env.draw()                 # current state
            env.draw(obs=some_obs)     # any obs vector
            env.draw(ax=ax)            # into existing axes
        """
        import matplotlib.pyplot as plt

        if obs is not None:
            adj = np.zeros((self.n, self.n), dtype=np.int8)
            adj[self._triu_idx] = obs.astype(np.int8)
            adj = adj + adj.T
        else:
            adj = self.adj

        show = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        self._draw_adj(adj, ax, title=title, highlight_c4=highlight_c4)

        if show:
            plt.tight_layout()
            plt.show()

    def rollout(self, model, deterministic=True, draw=True):
        """Run one episode with a trained SB3 model, optionally draw final state."""
        obs, _ = self.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _, info = self.step(int(action))

        print(f"edges={self.edge_count}  bound~{self.turan_bound()}  "
              f"({self.edge_count / self.turan_bound() * 100:.0f}% of bound)")

        if draw:
            self.draw()
        return self.edge_count

    def animate_rollout(self, model, deterministic=True, interval=300):
        """
        Animate a full episode step by step in a notebook.
        Requires: %matplotlib notebook  OR  %matplotlib widget
        interval: ms between frames
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        obs, _ = self.reset()
        frames = [self.adj.copy()]
        done = False
        step = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, _, info = self.step(int(action))
            frames.append(self.adj.copy())
            step += 1

        fig, ax = plt.subplots(figsize=(5, 5))

        def update(i):
            ax.clear()
            bound = self.turan_bound()
            ec = int(frames[i].sum()) // 2
            title = f"step {i}  edges={ec}  bound≈{bound}"
            if i == len(frames) - 1:
                title += "  ← C4 hit" if info.get("c4") else "  ← done"
            self._draw_adj(frames[i], ax, title=title)

        ani = animation.FuncAnimation(
            fig, update, frames=len(frames),
            interval=interval, repeat=True
        )
        plt.tight_layout()
        return ani   # in notebook: display(ani) or just let it autoplay

    def __repr__(self):
        return (f"TuranEnv(n={self.n}, edges={self.edge_count}, "
                f"bound~{self.turan_bound()}, done={self.done})")