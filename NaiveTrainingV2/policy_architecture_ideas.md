# Policy Network Architecture Ideas for Turan RL

**Context**: The environment observes a graph on `n` nodes as a flat boolean vector of length `E = n(n-1)/2` (upper-triangular adjacency). The action space is the same `E` indices — pick which edge to add. Episodes end when the forbidden subgraph property is satisfied. Current setup: small forbidden subgraphs (≤ 5-6 vertices), n likely staying small (≤ ~30).

---

## Shared Setup: Unpacking the Observation

All architectures below begin by reshaping the flat obs back into a symmetric adjacency matrix:

```python
# obs: (B, E) bool, E = n*(n-1)/2
# Reconstruct full n×n adjacency
i_idx, j_idx = torch.triu_indices(n, n, offset=1)   # shape (2, E)
A = torch.zeros(B, n, n, device=obs.device)          # (B, n, n)
A[:, i_idx, j_idx] = obs.float()
A[:, j_idx, i_idx] = obs.float()                     # symmetric
```

For variable N, `n` is passed as a parameter at construction or inferred from obs length via `n = (1 + sqrt(1 + 8*E)) / 2`.

---

## Architecture 1: Node Embedding + Bilinear Edge Scoring

### Mathematical Formulation

**Node features**: Initialize each node `i` from its adjacency row:

$$h_i^{(0)} = W_{\text{in}} A_{i,:} + b_{\text{in}} \in \mathbb{R}^d$$

Apply `L` rounds of a shared MLP update (no attention, pure aggregation):

$$h_i^{(\ell+1)} = \text{LayerNorm}\left( h_i^{(\ell)} + \text{MLP}\left( h_i^{(\ell)} \,\|\, \frac{1}{\deg_i} \sum_{j \sim i} h_j^{(\ell)} \right) \right)$$

where `||` is concatenation and `j ~ i` means edge `(i,j)` exists. This is a 1-layer Graph Convolutional Network (GCN) with residual connection and LayerNorm.

**Edge scoring**: After `L` rounds, score each candidate edge `(i,j)` with a bilinear form:

$$s_{ij} = h_i^{(L) \top} W_{\text{edge}} \, h_j^{(L)} \in \mathbb{R}$$

**Action logits**: Flatten the upper triangle, mask existing edges (set to `-inf`), then sample from Categorical.

**Critic**: Global mean pooling → MLP:

$$V = \text{MLP}_V\!\left(\frac{1}{n}\sum_i h_i^{(L)}\right) \in \mathbb{R}$$

### Torch Pseudocode

```python
class GCNActorCritic(nn.Module):
    def __init__(self, n, d=64, L=3):
        self.n = n
        self.proj_in = nn.Linear(n, d)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.Linear(d, d))
            for _ in range(L)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.W_edge = nn.Parameter(torch.randn(d, d) / d**0.5)
        self.critic_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        self.register_buffer('i_idx', i_idx)
        self.register_buffer('j_idx', j_idx)

    def forward(self, obs):
        B, E = obs.shape
        # Unpack adjacency
        A = torch.zeros(B, self.n, self.n)
        A[:, self.i_idx, self.j_idx] = obs
        A[:, self.j_idx, self.i_idx] = obs            # (B, n, n)

        # Initial node embeddings
        h = self.proj_in(A)                           # (B, n, d)

        # GCN rounds
        deg = A.sum(-1, keepdim=True).clamp(min=1)    # (B, n, 1)
        for layer, norm in zip(self.layers, self.norms):
            agg = (A @ h) / deg                       # (B, n, d) mean neighbor
            h = norm(h + layer(torch.cat([h, agg], dim=-1)))

        # Edge logits via bilinear scoring
        h_i = h[:, self.i_idx]                        # (B, E, d)
        h_j = h[:, self.j_idx]                        # (B, E, d)
        logits = (h_i @ self.W_edge * h_j).sum(-1)    # (B, E)

        # Mask existing edges
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        # Critic
        value = self.critic_head(h.mean(dim=1))       # (B, 1)
        return logits, value
```

**Properties**:
- O(n²d) per layer, very fast for small n
- Permutation equivariant (node relabeling → same policy behavior)
- Variable N: rebuild model with different `n`, or zero-pad to max N

---

## Architecture 2: Graph Transformer with Adjacency Bias

### Mathematical Formulation

Standard multi-head self-attention, but attention logits are biased by graph structure so already-connected node pairs attend more strongly.

**Input**: Same as above, `h_i^{(0)} = W_{\text{in}} A_{i,:} + b`.

**Biased attention**: For head `k` at layer `ℓ`:

$$e_{ij}^{(k,\ell)} = \frac{(W_Q^{(k)} h_i^{(\ell)})^\top (W_K^{(k)} h_j^{(\ell)})}{\sqrt{d_k}} + \beta^{(k)} A_{ij}$$

where `β^(k)` is a scalar learned bias per head (scalar, not per-pair). This lets each head decide whether to "focus on neighbors" or "look globally."

**Full update**:

$$\alpha_{ij}^{(k)} = \text{softmax}_j\left(e_{ij}^{(k)}\right)$$

$$\tilde{h}_i^{(\ell)} = \text{Concat}_k\left(\sum_j \alpha_{ij}^{(k)} W_V^{(k)} h_j^{(\ell)}\right)$$

$$h_i^{(\ell+1)} = \text{LayerNorm}\left(h_i^{(\ell)} + W_O \tilde{h}_i^{(\ell)}\right)$$

$$h_i^{(\ell+1)} \leftarrow \text{LayerNorm}\left(h_i^{(\ell+1)} + \text{FFN}(h_i^{(\ell+1)})\right)$$

**Why the bias matters for Turan**: Forbidden subgraphs are *local* (triangles, K4, etc). Biasing attention toward neighbors means the model naturally focuses on triangle-closing operations, which directly determine reward.

**Edge scoring and critic**: same bilinear head as Architecture 1.

### Torch Pseudocode

```python
class GraphTransformerActorCritic(nn.Module):
    def __init__(self, n, d=64, heads=4, L=3):
        self.n = n
        self.proj_in = nn.Linear(n, d)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(L)
        ])
        # Learned per-head adjacency bias scalar
        self.adj_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(heads)) for _ in range(L)
        ])
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
            for _ in range(L)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.W_edge = nn.Parameter(torch.randn(d, d) / d**0.5)
        self.critic_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        self.register_buffer('i_idx', i_idx)
        self.register_buffer('j_idx', j_idx)

    def forward(self, obs):
        B, E = obs.shape
        A = torch.zeros(B, self.n, self.n)
        A[:, self.i_idx, self.j_idx] = obs
        A[:, self.j_idx, self.i_idx] = obs            # (B, n, n)

        h = self.proj_in(A)                           # (B, n, d)

        for attn, bias_vec, ffn, norm in zip(
                self.attn_layers, self.adj_bias, self.ffn, self.norms):
            # Build adjacency attention bias: (B, heads, n, n)
            # bias_vec: (heads,) → broadcast over (B, n, n)
            attn_bias = (bias_vec.view(1, -1, 1, 1) * A.unsqueeze(1))
            # MultiheadAttention expects (B, n, n) attn_mask for additive bias
            # Merge heads dim into batch for manual attention
            # --- simplified: use attn_mask averaged over heads ---
            attn_mask = attn_bias.mean(dim=1)         # (B, n, n)

            # Run attention with additive bias (expand B into batch)
            # PyTorch MHA attn_mask adds to pre-softmax logits
            h_out, _ = attn(h, h, h,
                            attn_mask=attn_mask.view(B*1, self.n, self.n)
                                              .repeat_interleave(4, dim=0)
                                              .view(B, 4, self.n, self.n)
                                              .reshape(B*4, self.n, self.n),
                            need_weights=False)
            h = norm(h + h_out)
            h = h + ffn(h)

        # Edge logits
        h_i = h[:, self.i_idx]
        h_j = h[:, self.j_idx]
        logits = (h_i @ self.W_edge * h_j).sum(-1)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value = self.critic_head(h.mean(dim=1))
        return logits, value
```

> **Note**: The `attn_mask` reshaping above is fiddly. A cleaner implementation writes the attention manually with `einsum` to properly handle per-head biases. See Architecture 3 for a cleaner version.

**Properties**:
- O(n² d) attention, trivially fast for n ≤ 30
- Adjacency bias makes heads specialize: some heads follow edges (local), others ignore them (global)
- Permutation equivariant if no positional encodings are used

---

## Architecture 3: Clean Manual Multi-Head Attention with Per-Head Bias

This is the version you should actually implement — avoids the PyTorch MHA API awkwardness.

### Mathematical Formulation

Same as Architecture 2 but written cleanly. For layer `ℓ`, head `k`:

$$Q^{(k)} = H W_Q^{(k)}, \quad K^{(k)} = H W_K^{(k)}, \quad V^{(k)} = H W_V^{(k)}$$

$$\text{logits}^{(k)} = \frac{Q^{(k)} K^{(k)\top}}{\sqrt{d_k}} + \beta^{(k)} A \in \mathbb{R}^{n \times n}$$

$$\text{out}^{(k)} = \text{softmax}(\text{logits}^{(k)}) \cdot V^{(k)}$$

$$H' = \text{LayerNorm}(H + \text{Concat}(\text{out}^{(k)}_k) W_O)$$

### Torch Pseudocode

```python
class MHABlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.d_k = d // heads
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.beta = nn.Parameter(torch.zeros(heads))  # per-head adj bias
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, h, A):
        # h: (B, n, d),  A: (B, n, n)
        B, n, d = h.shape
        H = self.heads

        def split_heads(x):
            return x.view(B, n, H, self.d_k).transpose(1, 2)  # (B, H, n, d_k)

        Q, K, V = split_heads(self.Wq(h)), split_heads(self.Wk(h)), split_heads(self.Wv(h))

        # Scaled dot-product logits: (B, H, n, n)
        logits = Q @ K.transpose(-2, -1) / self.d_k**0.5

        # Adjacency bias: beta (H,) broadcast over (B, H, n, n)
        adj_bias = self.beta.view(1, H, 1, 1) * A.unsqueeze(1)
        logits = logits + adj_bias

        attn = logits.softmax(dim=-1)                  # (B, H, n, n)
        out = (attn @ V)                               # (B, H, n, d_k)
        out = out.transpose(1, 2).reshape(B, n, d)     # (B, n, d)
        out = self.Wo(out)

        h = self.norm1(h + out)
        h = self.norm2(h + self.ffn(h))
        return h


class TuranTransformerPolicy(nn.Module):
    def __init__(self, n, d=64, heads=4, L=3):
        super().__init__()
        self.n = n
        self.proj_in = nn.Linear(n, d)
        self.blocks = nn.ModuleList([MHABlock(d, heads) for _ in range(L)])
        self.W_edge = nn.Parameter(torch.randn(d, d) / d**0.5)
        self.critic_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        self.register_buffer('i_idx', i_idx)
        self.register_buffer('j_idx', j_idx)

    def forward(self, obs):
        B, E = obs.shape
        A = torch.zeros(B, self.n, self.n, device=obs.device)
        A[:, self.i_idx, self.j_idx] = obs.float()
        A[:, self.j_idx, self.i_idx] = obs.float()

        h = self.proj_in(A)                           # row of A as node features
        for block in self.blocks:
            h = block(h, A)                           # (B, n, d)

        # Bilinear edge scoring
        h_i = h[:, self.i_idx]                        # (B, E, d)
        h_j = h[:, self.j_idx]                        # (B, E, d)
        logits = (h_i @ self.W_edge * h_j).sum(-1)    # (B, E)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value = self.critic_head(h.mean(1))            # (B, 1)
        return logits, value
```

---

## Architecture 4: Edge Token Transformer

### Mathematical Formulation

Instead of node tokens, treat each **edge slot** `(i,j)` directly as a token. This makes the connection between tokens and actions trivial — the logit for action `k` is just the output of token `k`.

**Token feature for edge `(i,j)`**:

$$x_{ij} = \left[\mathbb{1}[\text{edge exists}],\; \deg_i / n,\; \deg_j / n,\; \text{common neighbors}(i,j) / n \right] \in \mathbb{R}^4$$

Then project: $h_{ij}^{(0)} = W_{\text{in}} x_{ij}$

**Attention bias between edge tokens**: Edges `(i,j)` and `(k,l)` are "adjacent" if they share a node. Define:

$$B_{(ij),(kl)} = \gamma_{\text{share}} \cdot \mathbb{1}[\{i,j\} \cap \{k,l\} \neq \emptyset]$$

This biases attention so that edges incident to the same node attend to each other — critical for detecting triangles.

**Output**: Final token representation `h_{ij}^{(L)}` → linear → scalar logit. Mask existing edges.

### Torch Pseudocode

```python
class EdgeTransformerPolicy(nn.Module):
    def __init__(self, n, d=64, heads=4, L=2):
        super().__init__()
        self.n = n
        E = n*(n-1)//2
        self.proj_in = nn.Linear(4, d)
        self.blocks = nn.ModuleList([MHABlock(d, heads) for _ in range(L)])
        # MHABlock from Architecture 3 but operating on E tokens instead of n
        self.logit_head = nn.Linear(d, 1)
        self.critic_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        i_idx, j_idx = torch.triu_indices(n, n, offset=1)   # (E,)
        self.register_buffer('i_idx', i_idx)
        self.register_buffer('j_idx', j_idx)

        # Precompute edge-edge adjacency (share a node): (E, E)
        share = (i_idx.unsqueeze(1) == i_idx.unsqueeze(0)) | \
                (i_idx.unsqueeze(1) == j_idx.unsqueeze(0)) | \
                (j_idx.unsqueeze(1) == i_idx.unsqueeze(0)) | \
                (j_idx.unsqueeze(1) == j_idx.unsqueeze(0))
        self.register_buffer('edge_adj', share.float())      # (E, E)
        self.gamma = nn.Parameter(torch.zeros(heads))

    def forward(self, obs):
        B, E = obs.shape

        # Node degrees from obs
        A = torch.zeros(B, self.n, self.n, device=obs.device)
        A[:, self.i_idx, self.j_idx] = obs.float()
        A[:, self.j_idx, self.i_idx] = obs.float()
        deg = A.sum(-1) / self.n                       # (B, n) normalized

        # Common neighbors for each edge pair
        cn = (A[:, self.i_idx] * A[:, self.j_idx]).sum(-1) / self.n  # (B, E)

        # Edge token features: [exists, deg_i, deg_j, common_neighbors]
        feats = torch.stack([
            obs.float(),
            deg[:, self.i_idx],
            deg[:, self.j_idx],
            cn
        ], dim=-1)                                     # (B, E, 4)

        h = self.proj_in(feats)                        # (B, E, d)

        # Edge-edge adjacency bias: (B=1, E, E) broadcast
        # Using edge_adj as the "A" argument to MHABlock
        edge_A = self.edge_adj.unsqueeze(0).expand(B, -1, -1)  # (B, E, E)
        for block in self.blocks:
            h = block(h, edge_A)

        logits = self.logit_head(h).squeeze(-1)        # (B, E)
        logits = logits.masked_fill(obs.bool(), float('-inf'))

        value = self.critic_head(h.mean(1))            # (B, 1)
        return logits, value
```

**Scaling**: E = n(n-1)/2 tokens. Attention is O(E²) = O(n⁴/4). For n=20: E=190, E²=36k — totally fine. For n=50: E=1225, E²=1.5M — getting expensive but doable with flash attention.

**Key advantage**: The `common_neighbors` feature directly encodes triangle-closure potential — the model sees exactly the information it needs to reason about forbidden subgraphs.

---

## The Variable-N Problem

The goal is a **single set of trained weights** that runs on any n. This is a hard constraint, not a table column.

The bottleneck is always the **input projection**. Architectures 1–3 as written use:

```python
self.proj_in = nn.Linear(n, d)   # weight shape (d, n) — BREAKS for different n
```

because they feed the adjacency row `A[i,:] ∈ ℝⁿ` as node features. If n changes, the weight matrix is the wrong shape.

Architecture 4 already sidesteps this: its input features are `[exists, deg_i/n, deg_j/n, common_neighbors/n]` — all scalars, `proj_in = nn.Linear(4, d)` is independent of n.

### Fix for Architectures 1–3: n-Independent Node Features

Replace `proj_in = nn.Linear(n, d)` with a small fixed-size feature vector per node:

```python
def node_features(A, n):
    # A: (B, n, n)
    deg = A.sum(-1)                                    # (B, n)
    deg_norm = deg / (n - 1)                           # fraction of possible edges

    # Clustering coefficient: triangles / (deg*(deg-1)/2)
    triangles = torch.diagonal(A @ A @ A, dim1=-2, dim2=-1) / 2  # (B, n)
    denom = deg * (deg - 1) / 2
    cc = triangles / denom.clamp(min=1)                # (B, n)

    # Mean and std of neighbor degrees
    neighbor_deg_sum = A @ deg                         # (B, n)
    mean_nbr_deg = neighbor_deg_sum / deg.clamp(min=1) / (n-1)

    return torch.stack([deg_norm, cc, mean_nbr_deg], dim=-1)  # (B, n, 3)
```

Then `proj_in = nn.Linear(3, d)` — fixed regardless of n. The rest of the architecture (attention layers, bilinear edge scoring) is already n-agnostic because attention over sequences of variable length is natural.

**The tradeoff**: you lose the exact adjacency structure in the input features. The attention layers recover it through the adjacency bias `β A`, but the initial node representation is purely local (degree, clustering, neighbor degree stats). For small forbidden subgraphs this is probably fine — triangles and K4s are detectable from these features within 2–3 attention layers.

**A middle ground**: concatenate n-independent features with a projected adjacency row, but normalize the projection:

$$h_i^{(0)} = W_{\text{local}} f_i + W_{\text{adj}} \frac{A_{i,:}}{\sqrt{n}}$$

where `f_i ∈ ℝ³` is the local feature vector and `W_adj` operates on a length-n input. This still ties the model to n — don't bother unless you're committed to a fixed n.

### Comparison Table

| Architecture | Input features | `proj_in` shape | Same weights for all N? | Attention cost |
|---|---|---|---|---|
| 1–3 as written | adj row `A[i,:]` | `(n → d)` | **No** | O(n²d) |
| 1–3 with n-indep features | `[deg, cc, mean_nbr_deg]` | `(3 → d)` | **Yes** | O(n²d) |
| 4: Edge Transformer | `[exists, deg_i, deg_j, cn]` | `(4 → d)` | **Yes** | O(n⁴) |

For your setting (n ≤ 30, small forbidden subgraphs):
- **Architecture 3 with n-independent features** is the right starting point: fast, principled, same weights across N
- **Architecture 4** has the most informative edge-level features (common neighbors directly encodes triangle potential) but costs O(n⁴) — still fast at n=30 (E=435 tokens)
- If you only ever train/test at a single n, using the adj row as features is fine and strictly more informative
