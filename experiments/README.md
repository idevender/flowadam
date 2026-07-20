# Experiments

Standalone scripts used for the FlowAdam benchmark suite.

- `rosenbrock.py`: Rosenbrock curved-valley optimization benchmark.
- `two_spirals.py`: Two-spirals (extreme variant) classification benchmark.
- `ablation_injection.py`: Hard-vs-soft momentum injection ablation.
- `rotated_quadratic.py`: Rotated stiff quadratic benchmark.
- `cifar10.py`: CIFAR-10 ResNet-18 benchmark (mode A settings).
- `ill_conditioned_regression.py`: Ill-conditioned regression benchmark.
- `matrix_completion.py`: Synthetic matrix completion benchmark.
- `movielens.py`: MovieLens residualized matrix factorization benchmark.
- `jester.py`: Jester benchmark with tuned baselines and compute-matched comparisons.
- `jester_compute_matched.py`: Focused compute-matched Jester benchmark.
- `robust_factorization.py`: Robust low-rank plus sparse factorization benchmark.
- `tensor_completion.py`: CP-style tensor completion benchmark.
- `gnn_link_prediction.py`: GNN link prediction benchmark.
- `inverse_kinematics.py`: Multi-target inverse kinematics benchmark.
- `gamma_ablation.py`: Momentum blend gamma ablation.

## FlowAdam settings per experiment

Each script sets `switch_sensitivity`, `curvature_sensitivity`, and `ode_t_scale`
explicitly (overriding the `mode` preset). The values actually used are:

| Script | switch | curvature | ode_t_scale | Closest preset |
|---|---:|---:|---:|---|
| `matrix_completion.py` | 0.90 | 0.1 | 0.5 | B |
| `tensor_completion.py` | 0.90 | 0.1 | 0.5 | B |
| `robust_factorization.py` | 0.90 | 0.1 | 0.5 | B |
| `movielens.py` | 0.90 | 0.1 | 0.5 | B |
| `jester.py` | 0.90 | 0.1 | 0.5 | B |
| `rotated_quadratic.py` | 0.90 | 0.1 | 0.5 | B |
| `gamma_ablation.py` | 0.90 | 0.1 | 0.5 | B |
| `inverse_kinematics.py` | 0.90 | 1.5 | 0.5 | B (curvature tuned) |
| `gnn_link_prediction.py` | 0.50 | 1.5 | 1.0 | between A and B |
| `ill_conditioned_regression.py` | 0.40 | 2.5 | 1.0 | between A and B |
| `rosenbrock.py` | 0.50 | 2.0 | 1.0 | between A and B |
| `two_spirals.py` | 0.50 | 2.0 | 1.0 | between A and B |
| `ablation_injection.py` | 0.50 | 2.0 | 1.0 | between A and B |
| `cifar10.py` | 0.40 | 3.0 | 2.0 | A |
