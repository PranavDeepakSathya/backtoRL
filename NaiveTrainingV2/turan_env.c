#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ROWS(n) (((n) + 63) / 64)
#define BIT(packed, i) (((packed)[(i)/64] >> ((i)%64)) & 1ULL)
#define SET(packed, i) ((packed)[(i)/64] |= (1ULL << ((i)%64)))
#define CLR(packed, i) ((packed)[(i)/64] &= ~(1ULL << ((i)%64)))

typedef struct {
  uint8_t*  obs;        /* upper triangle only, shape (num_envs, num_actions) */
  uint64_t* packed;     /* full n×n bitset for checker logic */
  float*    reward;
  int*      done;
  int*      edge_count;
  int       n;
  int       num_envs;
  int       rows;
  int       num_actions; /* n*(n-1)/2 */
} Env;

/* index of edge (u,v) with u < v in upper-triangle layout */
static inline int edge_idx(int n, int u, int v) {
  return u * (2*n - u - 1) / 2 + (v - u - 1);
}

Env* env_create(int n, int num_envs) {
  Env* env      = (Env*)malloc(sizeof(Env));
  int rows      = ROWS(n);
  int num_actions = n * (n - 1) / 2;
  env->n          = n;
  env->num_envs   = num_envs;
  env->rows       = rows;
  env->num_actions = num_actions;
  env->obs        = (uint8_t*)calloc(num_envs * num_actions, sizeof(uint8_t));
  env->packed     = (uint64_t*)calloc(num_envs * n * rows, sizeof(uint64_t));
  env->reward     = (float*)calloc(num_envs, sizeof(float));
  env->done       = (int*)calloc(num_envs, sizeof(int));
  env->edge_count = (int*)calloc(num_envs, sizeof(int));
  return env;
}

void env_destroy(Env* env) {
  free(env->obs);
  free(env->packed);
  free(env->reward);
  free(env->done);
  free(env->edge_count);
  free(env);
}

void env_reset_single(Env* env, int e) {
  int n           = env->n;
  int rows        = env->rows;
  int num_actions = env->num_actions;
  memset(env->obs    + e * num_actions, 0, num_actions * sizeof(uint8_t));
  memset(env->packed + e * n * rows,    0, n * rows * sizeof(uint64_t));
  env->reward[e]     = 0.0f;
  env->done[e]       = 0;
  env->edge_count[e] = 0;
}

void env_reset_all(Env* env) {
  int num_envs    = env->num_envs;
  int n           = env->n;
  int rows        = env->rows;
  int num_actions = env->num_actions;
  memset(env->obs,        0, num_envs * num_actions * sizeof(uint8_t));
  memset(env->packed,     0, num_envs * n * rows    * sizeof(uint64_t));
  memset(env->reward,     0, num_envs               * sizeof(float));
  memset(env->done,       0, num_envs               * sizeof(int));
  memset(env->edge_count, 0, num_envs               * sizeof(int));
}

static inline uint64_t* get_packed(Env* env, int e) {
  return env->packed + e * env->n * env->rows;
}

static inline uint8_t* get_obs(Env* env, int e) {
  return env->obs + e * env->num_actions;
}

// ── checkers ────────────────────────────────────────────────────────────────
// all called BEFORE adding edge (u,v)
// packed: pointer to this env's packed adj, shape (n, rows)

static int check_c3(uint64_t* packed, int n, int rows, int u, int v) {
  for (int w = 0; w < n; w++) {
    if (w == u || w == v) continue;
    if (BIT(packed + u*rows, w) && BIT(packed + v*rows, w))
      return 1;
  }
  return 0;
}

static int check_c4(uint64_t* packed, int n, int rows, int u, int v) {
  for (int x = 0; x < n; x++) {
    if (!BIT(packed + u*rows, x) || x == v) continue;
    for (int y = 0; y < n; y++) {
      if (!BIT(packed + v*rows, y) || y == u) continue;
      if (x != y && BIT(packed + x*rows, y))
        return 1;
    }
  }
  return 0;
}

static int check_c3c4(uint64_t* packed, int n, int rows, int u, int v) {
  return check_c3(packed, n, rows, u, v) || check_c4(packed, n, rows, u, v);
}

static int check_k23(uint64_t* packed, int n, int rows, int u, int v) {
  int pairs[2][2] = {{u,v},{v,u}};
  for (int p = 0; p < 2; p++) {
    int candidate = pairs[p][0];
    int other     = pairs[p][1];
    for (int w = 0; w < n; w++) {
      if (w == candidate || w == other) continue;
      if (!BIT(packed + other*rows, w)) continue;
      int common = 0;
      for (int i = 0; i < n; i++) {
        if (BIT(packed + candidate*rows, i) && BIT(packed + w*rows, i))
          common++;
      }
      if (common >= 2) return 1;
    }
  }
  return 0;
}

static int check_theta123(uint64_t* packed, int n, int rows, int u, int v) {
  // case1: u,v are poles
  for (int w1 = 0; w1 < n; w1++) {
    if (w1 == u || w1 == v) continue;
    if (!BIT(packed + u*rows, w1) || !BIT(packed + v*rows, w1)) continue;
    for (int w2 = 0; w2 < n; w2++) {
      if (w2 == u || w2 == v || w2 == w1) continue;
      if (!BIT(packed + u*rows, w2)) continue;
      for (int w3 = 0; w3 < n; w3++) {
        if (w3 == u || w3 == v || w3 == w1 || w3 == w2) continue;
        if (!BIT(packed + v*rows, w3)) continue;
        if (BIT(packed + w2*rows, w3)) return 1;
      }
    }
  }

  // case2: (u,v) is internal edge of length-2 path
  for (int p = 0; p < n; p++) {
    if (p == u || p == v) continue;
    if (!BIT(packed + u*rows, p) || !BIT(packed + v*rows, p)) continue;
    for (int w1 = 0; w1 < n; w1++) {
      if (w1 == u || w1 == v || w1 == p) continue;
      if (!BIT(packed + p*rows, w1)) continue;
      for (int w2 = 0; w2 < n; w2++) {
        if (w2 == u || w2 == v || w2 == p || w2 == w1) continue;
        if (BIT(packed + v*rows, w2) && BIT(packed + w1*rows, w2)) return 1;
        if (BIT(packed + u*rows, w2) && BIT(packed + w1*rows, w2)) return 1;
      }
    }
  }

  // case3a: (u,v) is middle edge of length-3 path
  for (int w1 = 0; w1 < n; w1++) {
    if (w1 == u || w1 == v) continue;
    if (!BIT(packed + u*rows, w1)) continue;
    for (int w2 = 0; w2 < n; w2++) {
      if (w2 == u || w2 == v || w2 == w1) continue;
      if (!BIT(packed + v*rows, w2)) continue;
      if (!BIT(packed + w1*rows, w2)) continue;
      for (int q = 0; q < n; q++) {
        if (q == u || q == v || q == w1 || q == w2) continue;
        if (BIT(packed + w1*rows, q) && BIT(packed + w2*rows, q)) return 1;
      }
    }
  }

  // case3b: (u,v) is first edge
  for (int m1 = 0; m1 < n; m1++) {
    if (m1 == u || m1 == v) continue;
    if (!BIT(packed + v*rows, m1)) continue;
    for (int m2 = 0; m2 < n; m2++) {
      if (m2 == u || m2 == v || m2 == m1) continue;
      if (!BIT(packed + m1*rows, m2)) continue;
      if (!BIT(packed + u*rows, m2)) continue;
      for (int q = 0; q < n; q++) {
        if (q == u || q == v || q == m1 || q == m2) continue;
        if (BIT(packed + u*rows, q) && BIT(packed + m2*rows, q)) return 1;
      }
    }
  }

  // case3b symmetric: (u,v) is last edge
  for (int m1 = 0; m1 < n; m1++) {
    if (m1 == u || m1 == v) continue;
    if (!BIT(packed + u*rows, m1)) continue;
    for (int m2 = 0; m2 < n; m2++) {
      if (m2 == u || m2 == v || m2 == m1) continue;
      if (!BIT(packed + m1*rows, m2)) continue;
      if (!BIT(packed + v*rows, m2)) continue;
      for (int q = 0; q < n; q++) {
        if (q == u || q == v || q == m1 || q == m2) continue;
        if (BIT(packed + v*rows, q) && BIT(packed + m2*rows, q)) return 1;
      }
    }
  }

  return 0;
}

static int check_bull(uint64_t* packed, int n, int rows, int u, int v) {
  // case1: (u,v) is pendant edge
  for (int a = 0; a < n; a++) {
    if (a == u || a == v) continue;
    for (int b = a+1; b < n; b++) {
      if (b == u || b == v) continue;
      if (!BIT(packed + a*rows, b)) continue;
      int hubs[2] = {u, v};
      for (int h = 0; h < 2; h++) {
        int hub = hubs[h];
        if (!BIT(packed + hub*rows, a) || !BIT(packed + hub*rows, b)) continue;
        int t2s[2] = {a, b};
        for (int t = 0; t < 2; t++) {
          for (int p = 0; p < n; p++) {
            if (p == u || p == v || p == a || p == b) continue;
            if (BIT(packed + t2s[t]*rows, p)) return 1;
          }
        }
      }
    }
  }

  // case2: (u,v) is triangle edge
  for (int w = 0; w < n; w++) {
    if (w == u || w == v) continue;
    if (!BIT(packed + u*rows, w) || !BIT(packed + v*rows, w)) continue;
    int tri[3][2] = {{u,v},{v,w},{u,w}};
    for (int e = 0; e < 3; e++) {
      int a = tri[e][0], b = tri[e][1];
      for (int p = 0; p < n; p++) {
        if (p == u || p == v || p == w) continue;
        if (!BIT(packed + a*rows, p)) continue;
        for (int q = 0; q < n; q++) {
          if (q == u || q == v || q == w || q == p) continue;
          if (BIT(packed + b*rows, q)) return 1;
        }
      }
    }
  }
  return 0;
}

static int check_bowtie(uint64_t* packed, int n, int rows, int u, int v) {
  for (int w = 0; w < n; w++) {
    if (w == u || w == v) continue;
    if (!BIT(packed + u*rows, w) || !BIT(packed + v*rows, w)) continue;
    int ts[3] = {u, v, w};
    for (int ti = 0; ti < 3; ti++) {
      int t = ts[ti];
      for (int a = 0; a < n; a++) {
        if (a == u || a == v || a == w) continue;
        if (!BIT(packed + t*rows, a)) continue;
        for (int b = a+1; b < n; b++) {
          if (b == u || b == v || b == w) continue;
          if (BIT(packed + t*rows, b) && BIT(packed + a*rows, b)) return 1;
        }
      }
    }
  }
  return 0;
}

static int check_k4(uint64_t* packed, int n, int rows, int u, int v) {
  for (int p = 0; p < n; p++) {
    if (p == u || p == v) continue;
    if (!BIT(packed + u*rows, p) || !BIT(packed + v*rows, p)) continue;
    for (int q = p+1; q < n; q++) {
      if (q == u || q == v) continue;
      if (BIT(packed + u*rows, q) && BIT(packed + v*rows, q) && BIT(packed + p*rows, q))
        return 1;
    }
  }
  return 0;
}

// ── checker dispatch ─────────────────────────────────────────────────────────

typedef int (*CheckerFn)(uint64_t*, int, int, int, int);

static CheckerFn get_checker(int checker_id) {
  switch(checker_id) {
    case 0: return check_c3;
    case 1: return check_c4;
    case 2: return check_c3c4;
    case 3: return check_k23;
    case 4: return check_theta123;
    case 5: return check_bull;
    case 6: return check_bowtie;
    case 7: return check_k4;
    default: return check_c4;
  }
}

// ── step (with edge toggling) ────────────────────────────────────────────────

void env_step(Env* env, int* actions, int checker_id) {
  int n        = env->n;
  int rows     = env->rows;
  int num_envs = env->num_envs;
  CheckerFn checker = get_checker(checker_id);

  #pragma omp parallel for schedule(static)
  for (int e = 0; e < num_envs; e++) {
    if (env->done[e]) {
      env->reward[e] = 0.0f;
      continue;
    }

    int u = actions[e * 2];
    int v = actions[e * 2 + 1];
    uint64_t* packed = get_packed(env, e);
    uint8_t*  obs    = get_obs(env, e);
    int       eidx   = edge_idx(n, u, v);

    if (BIT(packed + u*rows, v)) {
      /* ── edge exists → REMOVE (toggle off) ── */
      obs[eidx] = 0;
      CLR(packed + u*rows, v);
      CLR(packed + v*rows, u);
      env->edge_count[e]--;
      env->reward[e] = 0.0f;
    } else if (checker(packed, n, rows, u, v)) {
      /* ── adding would create forbidden subgraph → TERMINAL ── */
      env->done[e]   = 1;
      env->reward[e] = (float)env->edge_count[e];
    } else {
      /* ── safe addition ── */
      obs[eidx] = 1;
      SET(packed + u*rows, v);
      SET(packed + v*rows, u);
      env->edge_count[e]++;
      env->reward[e] = -0.01f;
    }
  }
}

// ── exposed API ──────────────────────────────────────────────────────────────

Env* create(int n, int num_envs)   { return env_create(n, num_envs); }
void destroy(Env* env)             { env_destroy(env); }
void reset_all(Env* env)           { env_reset_all(env); }
void reset_single(Env* env, int e) { env_reset_single(env, e); }

uint8_t* obs_ptr(Env* env)        { return env->obs; }
float*   reward_ptr(Env* env)     { return env->reward; }
int*     done_ptr(Env* env)       { return env->done; }
int*     edge_count_ptr(Env* env) { return env->edge_count; }
void step(Env* env, int* actions, int checker_id) {
  env_step(env, actions, checker_id);
}
