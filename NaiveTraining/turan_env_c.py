import numpy as np
import ctypes
import os
import subprocess

# checker ids matching C dispatch table
CHECKER_C3       = 0
CHECKER_C4       = 1
CHECKER_C3C4     = 2
CHECKER_K23      = 3
CHECKER_THETA123 = 4
CHECKER_BULL     = 5
CHECKER_BOWTIE   = 6
CHECKER_K4       = 7

def _build(src='turan_env.c', out='turan_env.so'):
  src_path = os.path.join(os.path.dirname(__file__), src)
  out_path = os.path.join(os.path.dirname(__file__), out)
  cmd = [
    'gcc', '-O3', '-march=native', '-fopenmp',
    '-shared', '-fPIC',
    '-o', out_path, src_path
  ]
  subprocess.run(cmd, check=True)
  return out_path

def _load(so_path='turan_env.so'):
  path = os.path.join(os.path.dirname(__file__), so_path)
  _build()
  lib = ctypes.CDLL(path)

  lib.create.restype  = ctypes.c_void_p
  lib.create.argtypes = [ctypes.c_int, ctypes.c_int]

  lib.destroy.restype  = None
  lib.destroy.argtypes = [ctypes.c_void_p]

  lib.reset_all.restype  = None
  lib.reset_all.argtypes = [ctypes.c_void_p]

  lib.reset_single.restype  = None
  lib.reset_single.argtypes = [ctypes.c_void_p, ctypes.c_int]

  lib.step.restype  = None
  lib.step.argtypes = [ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_int),
                       ctypes.c_int]

  lib.obs_ptr.restype    = ctypes.POINTER(ctypes.c_uint8)
  lib.obs_ptr.argtypes   = [ctypes.c_void_p]

  lib.reward_ptr.restype  = ctypes.POINTER(ctypes.c_float)
  lib.reward_ptr.argtypes = [ctypes.c_void_p]

  lib.done_ptr.restype    = ctypes.POINTER(ctypes.c_int)
  lib.done_ptr.argtypes   = [ctypes.c_void_p]

  lib.edge_count_ptr.restype  = ctypes.POINTER(ctypes.c_int)
  lib.edge_count_ptr.argtypes = [ctypes.c_void_p]
  
  return lib


class CEnv:
  def __init__(self, n, num_envs, checker_id=CHECKER_C4):
    self.n          = n
    self.num_envs   = num_envs
    self.checker_id = checker_id
    self.num_actions = n * (n - 1) // 2

    self._lib = _load()
    self._env = self._lib.create(n, num_envs)

    # zero copy numpy views into C buffers
    obs_raw    = self._lib.obs_ptr(self._env)
    rew_raw    = self._lib.reward_ptr(self._env)
    done_raw   = self._lib.done_ptr(self._env)
    ec_raw = self._lib.edge_count_ptr(self._env)
        
    self.obs    = np.frombuffer((ctypes.c_uint8 * (num_envs*n*n)).from_address(
                    ctypes.addressof(obs_raw.contents)), dtype=np.uint8
                  ).reshape(num_envs, n*n)
    self.reward = np.frombuffer((ctypes.c_float * num_envs).from_address(
                    ctypes.addressof(rew_raw.contents)), dtype=np.float32)
    self.done   = np.frombuffer((ctypes.c_int * num_envs).from_address(
                    ctypes.addressof(done_raw.contents)), dtype=np.int32)

    self.edge_count = np.frombuffer((ctypes.c_int * num_envs).from_address(
        ctypes.addressof(ec_raw.contents)), dtype=np.int32)

    # precompute upper triangle edge index → (u,v)
    us, vs = np.triu_indices(n, k=1)
    self._edge_u = us.astype(np.int32)
    self._edge_v = vs.astype(np.int32)

    self.reset()

  def step(self, actions):
    u  = self._edge_u[actions].astype(np.int32)
    v  = self._edge_v[actions].astype(np.int32)
    uv = np.stack([u, v], axis=1).ravel().copy()
    uv_ptr = uv.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    self._lib.step(self._env, uv_ptr, self.checker_id)
    done   = self.done.copy().astype(bool)
    reward = np.where(done, self.edge_count.astype(np.float32), self.reward)
    for e in np.where(done)[0]:
        self._lib.reset_single(self._env, int(e))
    return self.obs.copy(), reward, done

  def reset(self):
    self._lib.reset_all(self._env)
    return self.obs, self.reward, self.done.astype(bool)
        
  
  def close(self):
    self._lib.destroy(self._env)
    self._env = None

  def benchmark(self, steps=500):
    import time
    actions = np.random.randint(0, self.num_actions, self.num_envs).astype(np.uint32)
    self.reset()
    t0 = time.time()
    for _ in range(steps):
      self.step(actions)
    dt = time.time() - t0
    sps = steps * self.num_envs / dt
    print(f'{steps} steps x {self.num_envs} envs in {dt:.2f}s')
    print(f'  {sps:,.0f} steps/sec')
    print(f'  {sps/1e6:.2f}M steps/sec')
    
  def draw_obs(self, e=0):
    import matplotlib.pyplot as plt
    import networkx as nx
    adj = self.obs[e].reshape(self.n, self.n)
    G   = nx.from_numpy_array(adj.astype(float))
    pos = nx.spring_layout(G, seed=0)
    nx.draw(G, pos, with_labels=True, node_color='#1D9E75',
            node_size=500, font_color='white', edge_color='#444')
    plt.title(f'env={e}  edges={adj.sum()//2}')
    plt.show()