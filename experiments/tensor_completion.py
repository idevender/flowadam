"""Tensor completion benchmark using CP-style factorization."""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse

from flowadam import FlowAdam



class TensorFactorization(nn.Module):
    """
    CP (CANDECOMP/PARAFAC) decomposition: T ~ sum_r U[:,r] x V[:,r] x W[:,r]
    
    3-way factorization where each factor matrix couples with all others,
    creating denser parameter correlations than 2D matrix completion.
    """
    
    def __init__(self, dims, rank, init_scale=0.1):
        super().__init__()
        I, J, K = dims
        self.U = nn.Parameter(torch.randn(I, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(J, rank) * init_scale)
        self.W = nn.Parameter(torch.randn(K, rank) * init_scale)
        self.rank = rank
        self.dims = dims
    
    def forward(self, i, j, k):
        """Predict entries at positions (i, j, k)."""
        return (self.U[i] * self.V[j] * self.W[k]).sum(dim=1)
    
    def full_tensor(self):
        """Reconstruct full tensor (for debugging)."""
        return torch.einsum('ir,jr,kr->ijk', self.U, self.V, self.W)



def generate_data(dims, true_rank, density, noise, seed):
    """
    Generate low-rank tensor completion data with train/test split.
    
    Args:
        dims: Tuple (I, J, K) for tensor dimensions
        true_rank: True CP rank
        density: Fraction of observed entries
        noise: Standard deviation of observation noise
        seed: Random seed
    
    Returns:
        Dictionary with training and test data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    I, J, K = dims
    
    U_true = torch.randn(I, true_rank) / np.sqrt(true_rank)
    V_true = torch.randn(J, true_rank) / np.sqrt(true_rank)
    W_true = torch.randn(K, true_rank) / np.sqrt(true_rank)
    T_true = torch.einsum('ir,jr,kr->ijk', U_true, V_true, W_true)
    
    mask = torch.rand(I, J, K) < density
    i_obs, j_obs, k_obs = torch.where(mask)
    values = T_true[i_obs, j_obs, k_obs] + noise * torch.randn(len(i_obs))
    
    test_mask = ~mask
    i_test, j_test, k_test = torch.where(test_mask)
    test_values = T_true[i_test, j_test, k_test]
    
    return {
        'T_true': T_true,
        'train_i': i_obs, 'train_j': j_obs, 'train_k': k_obs, 'train_v': values,
        'test_i': i_test, 'test_j': j_test, 'test_k': k_test, 'test_v': test_values,
        'n_train': len(values), 'n_test': len(test_values)
    }



def train_adam(data, config):
    """Train with Adam (gradient-based weight decay)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TensorFactorization(
        config['dims'], config['model_rank'], config['init_scale']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    ti, tj, tk, tv = data['train_i'], data['train_j'], data['train_k'], data['train_v']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        pred = model(ti, tj, tk)
        loss = ((pred - tv) ** 2).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        test_pred = model(data['test_i'], data['test_j'], data['test_k'])
        test_rmse = ((test_pred - data['test_v']) ** 2).mean().sqrt().item()
    
    return test_rmse, elapsed


def train_adamw(data, config, weight_decay_override=None):
    """Train with AdamW (decoupled weight decay)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TensorFactorization(
        config['dims'], config['model_rank'], config['init_scale']
    )
    
    wd = weight_decay_override if weight_decay_override is not None else config['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=wd
    )
    
    ti, tj, tk, tv = data['train_i'], data['train_j'], data['train_k'], data['train_v']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        pred = model(ti, tj, tk)
        loss = ((pred - tv) ** 2).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        test_pred = model(data['test_i'], data['test_j'], data['test_k'])
        test_rmse = ((test_pred - data['test_v']) ** 2).mean().sqrt().item()
    
    return test_rmse, elapsed


def train_flowadam(data, config):
    """Train with FlowAdam (regularization in the loss, not gradient-based)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TensorFactorization(
        config['dims'], config['model_rank'], config['init_scale']
    )
    
    optimizer = FlowAdam(
        model.parameters(),
        lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )
    
    ti, tj, tk, tv = data['train_i'], data['train_j'], data['train_k'], data['train_v']
    wd = config['weight_decay']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        def closure():
            optimizer.zero_grad()
            pred = model(ti, tj, tk)
            loss = ((pred - tv) ** 2).mean()
            if wd > 0:
                reg = wd * (model.U.pow(2).sum() + model.V.pow(2).sum() + model.W.pow(2).sum())
                loss = loss + reg
            loss.backward()
            return loss
        
        optimizer.step(closure)
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        test_pred = model(data['test_i'], data['test_j'], data['test_k'])
        test_rmse = ((test_pred - data['test_v']) ** 2).mean().sqrt().item()
    
    return test_rmse, optimizer.get_ode_count(), elapsed



def run_experiment(config, seeds=[42, 123, 456, 789, 999]):
    """Run experiment with multiple seeds."""
    
    adam_results = []
    adamw_results = []
    adamw100x_results = []
    flow_results = []
    
    adam_times = []
    adamw_times = []
    flow_times = []
    
    for seed in seeds:
        config['seed'] = seed
        
        data = generate_data(
            config['dims'], config['true_rank'],
            config['density'], config['noise'], seed
        )
        
        adam_rmse, adam_time = train_adam(data, config)
        adamw_rmse, adamw_time = train_adamw(data, config)
        adamw100x_rmse, _ = train_adamw(data, config, weight_decay_override=1e-3)
        flow_rmse, ode_count, flow_time = train_flowadam(data, config)
        
        adam_results.append(adam_rmse)
        adamw_results.append(adamw_rmse)
        adamw100x_results.append(adamw100x_rmse)
        flow_results.append(flow_rmse)
        
        adam_times.append(adam_time)
        adamw_times.append(adamw_time)
        flow_times.append(flow_time)
        
        best_baseline = min(adam_rmse, adamw_rmse, adamw100x_rmse)
        winner = "" if flow_rmse < best_baseline else ""
        
        print(f"  Seed {seed}: Adam={adam_rmse:.4f}, AdamW={adamw_rmse:.4f}, "
              f"AdamW(100x)={adamw100x_rmse:.4f}, FlowAdam={flow_rmse:.4f} (ODE={ode_count}) {winner}")
    
    return {
        'adam': (adam_results, adam_times),
        'adamw': (adamw_results, adamw_times),
        'adamw100x': (adamw100x_results, adamw_times),
        'flowadam': (flow_results, flow_times)
    }


def run_three_scenarios():
    """Run three scenarios for tensor completion (varying size, sparsity, noise)."""
    
    scenarios = {
        'small_sparse': {
            'dims': (30, 40, 50),
            'true_rank': 5, 'density': 0.10, 'noise': 0.1,
            'description': '30x40x50, rank=5, density=10%, noise=0.1'
        },
        'medium_sparse': {
            'dims': (40, 50, 60),
            'true_rank': 8, 'density': 0.08, 'noise': 0.1,
            'description': '40x50x60, rank=8, density=8%, noise=0.1'
        },
        'larger_sparse': {
            'dims': (50, 60, 70),
            'true_rank': 10, 'density': 0.08, 'noise': 0.1,
            'description': '50x60x70, rank=10, density=8%, noise=0.1'
        }
    }
    
    base_config = {
        'model_rank': None,  # Set to true_rank + 5
        'init_scale': 0.1,
        'n_steps': 1000,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }
    
    seeds = [42, 123, 456, 789, 999]
    
    print("=" * 100)
    print("TENSOR COMPLETION (CP Decomposition): ADAM vs ADAMW vs FLOWADAM")
    print("3D generalization of matrix completion - even more parameter coupling!")
    print("=" * 100)
    
    all_results = {}
    
    for name, scenario in scenarios.items():
        print(f"\n{'='*90}")
        print(f"SCENARIO: {name}")
        print(f"    {scenario['description']}")
        print("=" * 90)
        
        config = base_config.copy()
        config['dims'] = scenario['dims']
        config['true_rank'] = scenario['true_rank']
        config['model_rank'] = scenario['true_rank'] + 5
        config['density'] = scenario['density']
        config['noise'] = scenario['noise']
        
        results = run_experiment(config, seeds)
        
        adam_r, adam_t = results['adam']
        adamw_r, adamw_t = results['adamw']
        adamw100x_r, _ = results['adamw100x']
        flow_r, flow_t = results['flowadam']
        
        adam_mean, adam_std = np.mean(adam_r), np.std(adam_r)
        adamw_mean, adamw_std = np.mean(adamw_r), np.std(adamw_r)
        adamw100x_mean = np.mean(adamw100x_r)
        flow_mean, flow_std = np.mean(flow_r), np.std(flow_r)
        
        best_baseline = min(adam_mean, adamw_mean, adamw100x_mean)
        best_baseline_name = "Adam" if adam_mean == best_baseline else (
            "AdamW" if adamw_mean == best_baseline else "AdamW(100x)")
        
        improv_vs_adam = (adam_mean - flow_mean) / adam_mean * 100
        improv_vs_best = (best_baseline - flow_mean) / best_baseline * 100
        
        best_baseline_r = [min(a, aw, aw100) for a, aw, aw100 in zip(adam_r, adamw_r, adamw100x_r)]
        wins_vs_best = sum(1 for b, f in zip(best_baseline_r, flow_r) if f < b)
        
        all_results[name] = {
            'adam_mean': adam_mean, 'adam_std': adam_std,
            'adamw_mean': adamw_mean, 'adamw_std': adamw_std,
            'adamw100x_mean': adamw100x_mean,
            'flow_mean': flow_mean, 'flow_std': flow_std,
            'best_baseline': best_baseline, 'best_baseline_name': best_baseline_name,
            'improv_vs_adam': improv_vs_adam, 'improv_vs_best': improv_vs_best,
            'wins_vs_best': wins_vs_best,
            'description': scenario['description']
        }
        
        print(f"\n    --- Results ---")
        print(f"    Adam:          {adam_mean:.4f} +/- {adam_std:.4f}")
        print(f"    AdamW:         {adamw_mean:.4f} +/- {adamw_std:.4f}")
        print(f"    AdamW (100xlambda): {adamw100x_mean:.4f}")
        print(f"    FlowAdam:      {flow_mean:.4f} +/- {flow_std:.4f}")
        print(f"\n    Best Baseline: {best_baseline_name} ({best_baseline:.4f})")
        print(f"    FlowAdam vs Adam: {improv_vs_adam:+.1f}%")
        print(f"    FlowAdam vs Best: {improv_vs_best:+.1f}% ({wins_vs_best}/5 wins)")
        
        if improv_vs_best >= 10:
            print(f"    >10% IMPROVEMENT!")
        elif flow_mean < best_baseline:
            print(f"    FlowAdam WINS!")
    
    
    print("\n")
    print("=" * 110)
    print("TABLE: TENSOR COMPLETION RESULTS (Test RMSE)")
    print("=" * 110)
    print(f"{'Scenario':<20} {'Adam':<14} {'AdamW':<14} {'AdamW(100x)':<14} {'FlowAdam':<14} {'Improvement':<12} {'Wins'}")
    print("-" * 110)
    
    total_wins = 0
    for name, r in all_results.items():
        short_name = name.replace('_', ' ').title()
        adam_str = f"{r['adam_mean']:.4f}+/-{r['adam_std']:.3f}"
        adamw_str = f"{r['adamw_mean']:.4f}+/-{r['adamw_std']:.3f}"
        adamw100x_str = f"{r['adamw100x_mean']:.4f}"
        flow_str = f"{r['flow_mean']:.4f}+/-{r['flow_std']:.3f}"
        
        print(f"{short_name:<20} {adam_str:<14} {adamw_str:<14} {adamw100x_str:<14} {flow_str:<14} {r['improv_vs_best']:+.1f}%        {r['wins_vs_best']}/5")
        total_wins += r['wins_vs_best']
    
    print("-" * 110)
    print(f"TOTAL: FlowAdam wins {total_wins}/15 comparisons ({total_wins/15*100:.0f}%)")
    
    avg_improvement = np.mean([r['improv_vs_best'] for r in all_results.values()])
    print(f"\nAverage improvement over best baseline: {avg_improvement:+.1f}%")
    print("=" * 110)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark 15: Tensor Completion (CP Decomposition)')
    parser.add_argument('--three_scenarios', action='store_true', 
                        help='Run all three scenarios (recommended)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    if args.three_scenarios:
        run_three_scenarios()
    else:
        print("Run with --three_scenarios for full benchmark")
        run_three_scenarios()


if __name__ == "__main__":
    main()
