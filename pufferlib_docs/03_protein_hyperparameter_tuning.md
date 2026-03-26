# Protein — PufferLib's Hyperparameter Tuning Algorithm

## Overview

Protein is PufferLib's Bayesian hyperparameter optimization algorithm, a simplified successor to CARBS (Cost AwaRe Bayesian Search). It combines:
- **Two Gaussian Processes** (score GP + cost GP)
- **Pareto front tracking** (score vs. compute cost)
- **Sobol sequence** initialization
- **Smart cost targeting** to explore the cost-performance frontier
- **Early stopping** via robust log-cost regression

From the blog:
> "Protein is a substantial simplification of CARBS... reduces complexity from 2,500 lines to ~500."

## Core Algorithm

### Phase 1: Random Exploration (first N suggestions)

```python
def suggest(fill):
    if suggestion_idx <= num_random_samples:     # Default: 10
        # Use Sobol quasi-random sequence for structured coverage
        zero_one = sobol.random(1)[0]            # [num_hyperparams] in [0, 1)
        suggestion = 2 * zero_one - 1            # Scale to [-1, 1) normalized space

        # Constrain cost parameter to low values during exploration
        if cost_param_idx is not None:
            suggestion[cost_param_idx] = clip(-0.8 + 0.1 * randn(), -1, 1)

        return unnormalize(suggestion)
```

**Why Sobol?** Provides better coverage of the search space than pure random sampling, ensuring the GP has diverse training points.

### Phase 2: Bayesian Optimization (subsequent suggestions)

```python
def suggest(fill):
    # 1. Train both GP models on observed data
    score_loss, cost_loss = train_gp_models()

    # 2. Compute Pareto front (score vs cost)
    pareto_front = pareto_points(success_observations)
    pruned_front = prune_pareto_front(pareto_front)  # Remove inefficient tail

    # 3. Generate candidate suggestions around Pareto + top observations
    search_centers = stack([e['input'] for e in pareto_front])
    if top_observations:
        search_centers = vstack([search_centers, get_top_obs_params()])

    # Sample candidates: perturbations of Pareto points
    suggestions = hyperparameters.sample(
        len(search_centers) * 256,        # 256 candidates per center
        mu=search_centers                  # Centered on Pareto front
    )
    suggestions = filter_near_duplicates(suggestions)

    # 4. Score each candidate using GP predictions
    with torch.no_grad():
        gp_y_norm = gp_score.predict(suggestions)     # Predicted normalized score
        gp_log_c_norm = gp_cost.predict(suggestions)   # Predicted normalized log-cost

    # 5. Selection criterion (the core formula)
    # Maximize predicted score
    suggestion_scores = optimize_direction * gp_y_norm

    # Penalize by distance from random target cost
    target_cost = sample_target_cost_ratio(expansion_rate=0.1)
    weight = 1 - abs(target_cost - gp_log_c_norm)

    # Filter out suggestions exceeding max cost
    max_c_mask = exp(gp_log_c) < max_suggestion_cost

    suggestion_scores *= max_c_mask * weight

    # 6. Pick the best
    best_idx = argmax(suggestion_scores)
    return unnormalize(suggestions[best_idx])
```

### The Cost Targeting Formula (Blog version)

From the blog, the simplified core is:
```
w(x) = GP_y(x) · (1 - |α·U - GP_c(x)|)
```
Where:
- `GP_y(x)` = predicted score from score GP
- `GP_c(x)` = predicted normalized cost from cost GP
- `U ~ Uniform(0, 1)` = random target cost ratio
- `α = 1.25` = expansion rate (allows exploring beyond current Pareto)

**Why random target cost?** Each suggestion targets a different cost level, ensuring exploration across the full cost spectrum. The `α > 1` factor means ~20% of suggestions target costs above the current frontier, naturally expanding it.

## Gaussian Process Details

### Model Architecture

```python
class ExactGPModel(ExactGP):
    # Mean: Constant mean function
    mean = ConstantMean()

    # Kernel: Scale(Additive(Linear, Matern(3/2)))
    #   - Linear kernel captures global trends
    #   - Matern 3/2 captures local variations
    #   - ARD (Automatic Relevance Determination) per dimension
    kernel = ScaleKernel(
        AdditiveKernel(
            PolynomialKernel(power=1),        # Linear
            MaternKernel(nu=1.5, ard_num_dims=num_hyperparams)  # Matern 3/2
        )
    )

    # Noise prior from HEBO paper
    noise_prior = LogNormalPrior(log(0.01), 0.5)
```

### Training

```python
# Both GPs trained with Adam, 50 iterations per suggest() call
optimizer = Adam(gp.parameters(), lr=0.001, amsgrad=True)

# Data normalization:
#   Score: (y - y_min) / (y_max - y_min)    → [0, 1]
#   Cost:  (log(c) - log_c_min) / (log_c_max - log_c_min)  → [0, 1]
#   Hyperparams: already in [-1, 1] normalized space

# Max observations fed to GP: 750 (training time jumps after ~800)
# When more data available: keep 50% recent + 50% random sample from older
```

### Optimizer Reset

Every 50 suggestions, the GP optimizers are reset to avoid getting stuck in local optima of the marginal likelihood.

## Hyperparameter Space Definitions

PufferLib supports several parameter distributions, all normalized to `[-1, 1]`:

| Distribution | Class | Example Use | Normalization |
|-------------|-------|-------------|---------------|
| `uniform` | `Linear` | Reward coefficients | `2*(x-min)/(max-min) - 1` |
| `int_uniform` | `Linear` | Batch size | Same, then round |
| `uniform_pow2` | `Pow2` | Hidden sizes | Log2-space mapping |
| `log_normal` | `Log` | Learning rate | Log10-space mapping |
| `logit_normal` | `Logit` | Clip coefficient | Logit-space mapping |

Each space has a `scale` parameter controlling the search radius around the center.

## Pareto Front Management

### Computing the Pareto Front

```python
def pareto_points(observations):
    scores = [e['output'] for e in observations]
    costs = [e['cost'] for e in observations]

    # Sort by cost, keep only non-dominated points
    # Point A dominates B if: cost(A) <= cost(B) AND score(A) > score(B)
    sorted_by_cost = argsort(costs)
    pareto = []
    max_score = -inf
    for idx in sorted_by_cost:
        if scores[idx] > max_score:
            pareto.append(observations[idx])
            max_score = scores[idx]
    return pareto
```

### Pruning the Pareto Tail

Removes high-cost points with diminishing returns:

```python
def prune_pareto_front(pareto, efficiency_threshold=0.5):
    # Walk from highest cost backward
    # Remove points where:
    #   normalized_score_gain / normalized_cost_increase < threshold
    # Stop pruning below 98% of max score
```

## Early Stopping

Protein includes a `RobustLogCostModel` that fits:
```
Score_threshold = A + B * log(Cost)
```
using quantile regression (30th percentile). During training, if a run's score falls below this threshold at its current cost, it's stopped early to save compute.

```python
def should_stop(score, cost):
    threshold = A + B * log(cost)
    return score < threshold
```

## Observe (Recording Results)

```python
def observe(hypers, score, cost, is_failure=False):
    params = normalize(hypers)  # Convert to [-1, 1] space

    if is_failure or not isfinite(score):
        failure_observations.append(...)
        return

    # Dedup: if near-identical params exist, update in place
    # Track top N observations by score for search diversity
    success_observations.append({
        input: params,
        output: score,
        cost: cost,
    })
```

## Key Improvements Over CARBS

1. **No P_search bias** — CARBS had a separate "search distribution" that could fight the GP
2. **Proper normalization** — All hyperparams mapped to `[-1, 1]`, GPs work on normalized data
3. **Cost targeting** — Random cost targets instead of CARBS's complex cost modeling
4. **Simplicity** — ~500 lines vs ~2,500 lines
5. **Robustness to lucky seeds** — Random cost targeting ensures 1/5 of experiments push the frontier higher
6. **Sobol initialization** — Better space coverage than random starts
