# PufferLib Documentation for Turan RL Project

Detailed technical documentation of PufferLib internals, based on source code analysis of
the local `PufferLib/` folder and the [puffer.ai blog](https://puffer.ai/blog.html).

## Documents

1. **[PPO Algorithm](01_ppo_algorithm.md)** — Full pseudocode of PufferLib's PPO implementation with all tensor shapes, the evaluate/train loop, data layout, config parameters, and the prioritized experience mechanism.

2. **[Puffer Advantage](02_puffer_advantage.md)** — The custom GAE+VTrace hybrid advantage estimator. Mathematical formulation, CUDA kernel code, how importance ratios flow through training epochs, and the relationship to prioritized sampling.

3. **[Protein Hyperparameter Tuning](03_protein_hyperparameter_tuning.md)** — The Bayesian optimization algorithm (successor to CARBS). Dual Gaussian Processes, Pareto front tracking, cost targeting, Sobol initialization, early stopping, and the full suggest/observe loop.

4. **[Wrapping C Environments](04_wrapping_c_env_for_pufferlib.md)** — Two approaches: (A) Full Ocean-style integration via `env_binding.h` with zero-copy shared buffers, and (B) Python PufferEnv wrapper. Includes the complete C API contract, struct requirements, and vectorization backends.

5. **[Custom Policies](05_custom_policies.md)** — How to plug custom neural network architectures into PufferLib. The `forward` / `forward_eval` interface, MLP/LSTM/Transformer examples, action masking, and architecture recommendations from the blog.

6. **[Our Turan Env Integration Plan](06_our_turan_env_integration_plan.md)** — Concrete plan for wrapping our `turan_env.c` graph construction environment for PufferLib training. Includes both quick-wrapper and full Ocean-style approaches, a complete training script, and a Protein sweep config.
