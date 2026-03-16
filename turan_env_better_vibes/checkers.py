import numpy as np


class SubgraphChecker:
  def __init__(self, n):
    self.n = n

  def check(self, adj, u, v) -> bool:
    raise NotImplementedError


class C3Checker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    return bool((adj[u] & adj[v]).sum() > 0)


class C4Checker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    assert u != v
    n = self.n
    for x in range(n):
      if not adj[u][x] or x == v: continue
      for y in range(n):
        if not adj[v][y] or y == u: continue
        if x != y and adj[x][y]:
          return True
    return False


class C3C4Checker(SubgraphChecker):
  def __init__(self, n):
    super().__init__(n)
    self.c3 = C3Checker(n)
    self.c4 = C4Checker(n)

  def check(self, adj, u, v) -> bool:
    return self.c3.check(adj, u, v) or self.c4.check(adj, u, v)




class K23Checker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    for candidate, other in [(u, v), (v, u)]:
      for w in range(self.n):
        if w == candidate or w == other: continue
        if adj[other][w] and (adj[candidate] & adj[w]).sum() >= 2:
          return True
    return False


class Theta123Checker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    return (
      self._case1(adj, u, v) or
      self._case2(adj, u, v) or
      self._case3(adj, u, v)
    )

  def _nbrs(self, adj, x, exclude):
    return set(w for w in range(self.n) if adj[x][w] and w not in exclude)

  def _case1(self, adj, u, v) -> bool:
    nbrs_u = self._nbrs(adj, u, {v})
    nbrs_v = self._nbrs(adj, v, {u})
    common = nbrs_u & nbrs_v
    if not common:
      return False
    for w1 in common:
      cands_u = nbrs_u - {w1}
      cands_v = nbrs_v - {w1}
      for w2 in cands_u:
        for w3 in cands_v:
          if w3 != w2 and adj[w2][w3]:
            return True
    return False

  def _case2(self, adj, u, v) -> bool:
    nbrs_u = self._nbrs(adj, u, {v})
    nbrs_v = self._nbrs(adj, v, {u})
    common = nbrs_u & nbrs_v
    for p in common:
      cands_w1 = self._nbrs(adj, p, {u, v, p})
      for w1 in cands_w1:
        for w2 in nbrs_v - {u, p}:
          if w2 != w1 and adj[w1][w2]:
            return True
      for w1 in cands_w1:
        for w2 in nbrs_u - {v, p}:
          if w2 != w1 and adj[w1][w2]:
            return True
    return False

  def _case3(self, adj, u, v) -> bool:
    nbrs_u = self._nbrs(adj, u, {v})
    nbrs_v = self._nbrs(adj, v, {u})
    for w1 in nbrs_u:
      for w2 in nbrs_v:
        if w2 == w1: continue
        if adj[w1][w2]:
          nbrs_w1 = self._nbrs(adj, w1, {u, v, w2})
          nbrs_w2 = self._nbrs(adj, w2, {u, v, w1})
          if nbrs_w1 & nbrs_w2:
            return True
    for m1 in nbrs_v:
      if m1 == u: continue
      for m2 in self._nbrs(adj, m1, {u, v}):
        if adj[u][m2]:
          if self._nbrs(adj, u, {v, m1, u, m2}) & self._nbrs(adj, m2, {v, m1, u, m2}):
            return True
    for m1 in nbrs_u:
      if m1 == v: continue
      for m2 in self._nbrs(adj, m1, {u, v}):
        if adj[v][m2]:
          if self._nbrs(adj, v, {u, m1, v, m2}) & self._nbrs(adj, m2, {u, m1, v, m2}):
            return True
    return False


class BullChecker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    n = self.n

    # case 1: (u,v) is a pendant edge
    # for each existing edge (a,b), if u or v is common neighbor of a,b
    # and one of a,b has another neighbor outside {u,v,a,b}
    for a in range(n):
      for b in range(a+1, n):
        if not adj[a][b]: continue
        if a in {u,v} or b in {u,v}: continue
        for hub in [u, v]:
          if adj[hub][a] and adj[hub][b]:
            # triangle (a, b, hub), pendant (u,v) off hub
            # need another pendant off a or b
            for t2 in [a, b]:
              for p in range(n):
                if p in {u, v, a, b}: continue
                if adj[t2][p]:
                  return True

    # case 2: (u,v) is a triangle edge
    # find common neighbor w, then for each triangle edge check if
    # it's the middle of a P3 with both ends outside {u,v,w}
    for w in range(n):
      if w == u or w == v: continue
      if not (adj[u][w] and adj[v][w]): continue
      # triangle (u,v,w), check each edge as middle of P3
      for a, b in [(u,v), (v,w), (u,w)]:
        for p in range(n):
          if p in {u,v,w}: continue
          if not adj[a][p]: continue
          for q in range(n):
            if q in {u,v,w} or q == p: continue
            if adj[b][q]:
              return True

    return False


class BowTieChecker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    n = self.n
    for w in range(n):
      if w == u or w == v: continue
      if not (adj[u][w] and adj[v][w]): continue
      for t in [u, v, w]:
        for a in range(n):
          if a in {u, v, w}: continue
          if not adj[t][a]: continue
          for b in range(a+1, n):
            if b in {u, v, w}: continue
            if adj[t][b] and adj[a][b]:
              return True
    return False
  
class K4Checker(SubgraphChecker):
  def check(self, adj, u, v) -> bool:
    n = self.n
    for p in range(n):
      if p == u or p == v: continue
      if not adj[u][p] or not adj[v][p]: continue
      for q in range(p+1, n):
        if q == u or q == v: continue
        if adj[u][q] and adj[v][q] and adj[p][q]:
          return True
    return False