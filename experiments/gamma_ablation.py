"""Ablation study for FlowAdam momentum blend gamma."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import argparse

from flowadam import FlowAdam



class MatrixFactorization(nn.Module):
    """Low-rank matrix factorization: A ~ UV^T"""
    
    def __init__(self, n_users, n_items, rank, init_scale=0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_users, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(n_items, rank) * init_scale)
    
    def forward(self, user_ids, item_ids):
        """Predict ratings for given (user, item) pairs."""
        return (self.U[user_ids] * self.V[item_ids]).sum(dim=1)



def generate_data(n_users, n_items, true_rank, density, noise, seed):
    """Generate matrix completion data with train/test split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    U_true = np.random.randn(n_users, true_rank) / np.sqrt(true_rank)
    V_true = np.random.randn(n_items, true_rank) / np.sqrt(true_rank)
    A_true = torch.tensor(U_true @ V_true.T, dtype=torch.float32)
    
    mask = torch.rand(n_users, n_items) < density
    user_ids, item_ids = torch.where(mask)
    ratings = A_true[user_ids, item_ids] + noise * torch.randn(len(user_ids))
    
    test_mask = ~mask
    test_u, test_i = torch.where(test_mask)
    test_r = A_true[test_u, test_i]
    
    return {
        'train_u': user_ids, 'train_i': item_ids, 'train_r': ratings,
        'test_u': test_u, 'test_i': test_i, 'test_r': test_r,
    }



def train_flowadam(data, config, gamma):
    """
    Train with FlowAdam using specified gamma.
    
    Args:
        data: Training data dict
        config: Configuration dict
        gamma: Momentum blend gamma value to test
    """
    torch.manual_seed(config['seed'] + 1000)
    
    model = MatrixFactorization(
        config['n_users'], config['n_items'], 
        config['model_rank'], config['init_scale']
    )
    
    optimizer = FlowAdam(
        model.parameters(),
        lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale'],
        momentum_blend_gamma=gamma
    )
    
    train_u, train_i, train_r = data['train_u'], data['train_i'], data['train_r']
    test_u, test_i, test_r = data['test_u'], data['test_i'], data['test_r']
    wd = config['weight_decay']
    
    for step in range(config['n_steps']):
        def closure():
            optimizer.zero_grad()
            pred = model(train_u, train_i)
            loss = ((pred - train_r) ** 2).mean()
            if wd > 0:
                reg = wd * (model.U.pow(2).sum() + model.V.pow(2).sum())
                loss = loss + reg
            loss.backward()
            return loss
        
        optimizer.step(closure)
    
    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()
    
    return test_rmse, optimizer.get_ode_count()


def train_adam_baseline(data, config):
    """Train with Adam for comparison."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = MatrixFactorization(
        config['n_users'], config['n_items'], 
        config['model_rank'], config['init_scale']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    train_u, train_i, train_r = data['train_u'], data['train_i'], data['train_r']
    test_u, test_i, test_r = data['test_u'], data['test_i'], data['test_r']
    
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()
    
    return test_rmse



def run_gamma_ablation():
    """Run the gamma sensitivity ablation study."""
    
    print("=" * 80)
    print("GAMMA (gamma) SENSITIVITY ABLATION STUDY")
    print("Testing momentum blend factor: gamma in [0.1, 0.9]")
    print("=" * 80)
    
    config = {
        'n_users': 200, 'n_items': 300,
        'true_rank': 10, 'model_rank': 15,
        'density': 0.30, 'noise': 0.1,
        'init_scale': 0.1, 'n_steps': 1000, 'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }
    
    gamma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = [42, 123, 456]
    
    print(f"\nConfiguration:")
    print(f"  Matrix: {config['n_users']}x{config['n_items']}, rank={config['true_rank']}")
    print(f"  Density: {config['density']*100:.0f}%, Noise: {config['noise']}")
    print(f"  Seeds: {seeds}")
    print(f"  Gamma values: {gamma_values}")
    
    results = {}
    adam_results = []
    
    print("\n--- Adam Baseline ---")
    for seed in seeds:
        config['seed'] = seed
        data = generate_data(
            config['n_users'], config['n_items'],
            config['true_rank'], config['density'],
            config['noise'], seed
        )
        adam_rmse = train_adam_baseline(data, config)
        adam_results.append(adam_rmse)
        print(f"  Seed {seed}: Adam = {adam_rmse:.4f}")
    
    adam_mean = np.mean(adam_results)
    adam_std = np.std(adam_results)
    print(f"  Adam mean: {adam_mean:.4f} +/- {adam_std:.4f}")
    
    for gamma in gamma_values:
        print(f"\n--- gamma = {gamma:.1f} ---")
        gamma_rmses = []
        gamma_ode_counts = []
        
        for seed in seeds:
            config['seed'] = seed
            data = generate_data(
                config['n_users'], config['n_items'],
                config['true_rank'], config['density'],
                config['noise'], seed
            )
            rmse, ode_count = train_flowadam(data, config, gamma)
            gamma_rmses.append(rmse)
            gamma_ode_counts.append(ode_count)
            print(f"  Seed {seed}: RMSE = {rmse:.4f}, ODE = {ode_count}")
        
        mean_rmse = np.mean(gamma_rmses)
        std_rmse = np.std(gamma_rmses)
        mean_ode = np.mean(gamma_ode_counts)
        
        results[gamma] = {
            'rmses': gamma_rmses,
            'mean': mean_rmse,
            'std': std_rmse,
            'ode_count': mean_ode
        }
        
        improvement = (adam_mean - mean_rmse) / adam_mean * 100
        print(f"  Mean: {mean_rmse:.4f} +/- {std_rmse:.4f}, ODE: {mean_ode:.0f}")
        print(f"  vs Adam: {improvement:+.1f}%")
    
    
    print("\n" + "=" * 80)
    print("SUMMARY: GAMMA SENSITIVITY")
    print("=" * 80)
    print(f"\n{'gamma':<8} {'RMSE (mean+/-std)':<20} {'ODE Triggers':<15} {'vs Adam':<12} {'Status'}")
    print("-" * 75)
    
    best_gamma = min(results.keys(), key=lambda g: results[g]['mean'])
    best_rmse = results[best_gamma]['mean']
    default_gamma = 0.5
    default_rmse = results[default_gamma]['mean']
    
    for gamma in gamma_values:
        r = results[gamma]
        improvement = (adam_mean - r['mean']) / adam_mean * 100
        
        if gamma == best_gamma:
            status = "* BEST"
        elif gamma == default_gamma:
            status = "- DEFAULT"
        elif abs(r['mean'] - best_rmse) / best_rmse < 0.02:  # Within 2%
            status = "[OK] Good"
        else:
            status = ""
        
        print(f"{gamma:<8.1f} {r['mean']:.4f}+/-{r['std']:.4f}           "
              f"{r['ode_count']:<15.0f} {improvement:>+5.1f}%        {status}")
    
    print("-" * 75)
    print(f"Adam:    {adam_mean:.4f}+/-{adam_std:.4f}           ---             baseline")
    
    
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)
    
    relative_diff = abs(default_rmse - best_rmse) / best_rmse * 100
    print(f"\nBest gamma: {best_gamma} (RMSE = {best_rmse:.4f})")
    print(f"Default gamma: {default_gamma} (RMSE = {default_rmse:.4f})")
    print(f"Relative difference: {relative_diff:.2f}%")
    
    if relative_diff < 2.0:
        print("\n[OK] VALIDATION PASSED: gamma=0.5 is within 2% of optimal")
        print("  -> gamma=0.5 is a ROBUST default, NOT cherry-picked")
    else:
        print(f"\n[WARN] WARNING: gamma=0.5 differs by {relative_diff:.1f}% from optimal")
        print(f"  -> Consider updating default to gamma={best_gamma}")
    
    robust_gammas = [g for g in gamma_values 
                     if abs(results[g]['mean'] - best_rmse) / best_rmse < 0.02]
    if robust_gammas:
        print(f"\nRobust range: gamma in [{min(robust_gammas)}, {max(robust_gammas)}]")
    
    
    generate_gamma_plot(results, adam_mean, adam_std)
    
    return results, adam_mean


def generate_gamma_plot(results, adam_mean, adam_std):
    """Generate publication-quality plot for gamma sensitivity."""
    
    gamma_values = sorted(results.keys())
    means = [results[g]['mean'] for g in gamma_values]
    stds = [results[g]['std'] for g in gamma_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(gamma_values, means, yerr=stds, 
                fmt='o-', color='blue', linewidth=2, markersize=8,
                capsize=5, capthick=2, label='FlowAdam')
    
    ax.axhline(y=adam_mean, color='red', linestyle='--', linewidth=2, label='Adam')
    ax.axhspan(adam_mean - adam_std, adam_mean + adam_std, 
               alpha=0.2, color='red', label='Adam +/-1sigma')
    
    default_idx = gamma_values.index(0.5)
    ax.scatter([0.5], [means[default_idx]], s=200, c='green', marker='*', 
               zorder=5, label='Default (gamma=0.5)')
    
    best_rmse = min(means)
    robust_gammas = [g for g, m in zip(gamma_values, means) 
                     if abs(m - best_rmse) / best_rmse < 0.02]
    if len(robust_gammas) >= 2:
        ax.axvspan(min(robust_gammas), max(robust_gammas), 
                   alpha=0.1, color='green', label='Robust range (+/-2%)')
    
    ax.set_xlabel('Injection weight gamma (weight on ODE velocity)', fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title('FlowAdam: Sensitivity to gamma (ODE Injection Weight)\n(Matrix Completion)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    ax.set_xticks(gamma_values)
    ax.set_xlim(0.0, 1.0)
    
    plt.tight_layout()
    
    fig.savefig('gamma_sensitivity.png', dpi=150, bbox_inches='tight')
    fig.savefig('gamma_sensitivity.pdf', bbox_inches='tight')
    print("\n[plot] Plot saved: gamma_sensitivity.png / gamma_sensitivity.pdf")
    
    plt.close()


def run_extended_analysis():
    """
    Extended analysis: Test gamma on multiple scenarios.
    This provides stronger evidence for publication.
    """
    
    print("=" * 80)
    print("EXTENDED GAMMA ANALYSIS (Multiple Scenarios)")
    print("=" * 80)
    
    scenarios = {
        'small_dense': {
            'n_users': 200, 'n_items': 300, 
            'true_rank': 10, 'density': 0.30, 'noise': 0.1,
        },
        'medium_moderate': {
            'n_users': 300, 'n_items': 400, 
            'true_rank': 15, 'density': 0.20, 'noise': 0.1,
        },
    }
    
    base_config = {
        'init_scale': 0.1, 'n_steps': 1000, 'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }
    
    gamma_values = [0.3, 0.4, 0.5, 0.6, 0.7]  # Focus on middle range
    seeds = [42, 123, 456]
    
    all_scenario_results = {}
    
    for scenario_name, scenario in scenarios.items():
        print(f"\n--- Scenario: {scenario_name} ---")
        
        config = base_config.copy()
        config['n_users'] = scenario['n_users']
        config['n_items'] = scenario['n_items']
        config['true_rank'] = scenario['true_rank']
        config['model_rank'] = scenario['true_rank'] + 5
        config['density'] = scenario['density']
        config['noise'] = scenario['noise']
        
        scenario_results = {}
        
        for gamma in gamma_values:
            gamma_rmses = []
            
            for seed in seeds:
                config['seed'] = seed
                data = generate_data(
                    config['n_users'], config['n_items'],
                    config['true_rank'], config['density'],
                    config['noise'], seed
                )
                rmse, _ = train_flowadam(data, config, gamma)
                gamma_rmses.append(rmse)
            
            mean_rmse = np.mean(gamma_rmses)
            scenario_results[gamma] = mean_rmse
            print(f"  gamma={gamma}: {mean_rmse:.4f}")
        
        all_scenario_results[scenario_name] = scenario_results
    
    print("\n" + "=" * 80)
    print("CROSS-SCENARIO SUMMARY")
    print("=" * 80)
    
    validated = True
    for scenario_name, results in all_scenario_results.items():
        best_gamma = min(results.keys(), key=lambda g: results[g])
        best_rmse = results[best_gamma]
        default_rmse = results[0.5]
        diff = abs(default_rmse - best_rmse) / best_rmse * 100
        
        status = "[OK]" if diff < 2.0 else "[WARN]"
        print(f"{scenario_name}: Best gamma={best_gamma} ({best_rmse:.4f}), "
              f"gamma=0.5 ({default_rmse:.4f}), diff={diff:.1f}% {status}")
        
        if diff >= 2.0:
            validated = False
    
    if validated:
        print("\n[OK] gamma=0.5 VALIDATED across all scenarios")
    else:
        print("\n[WARN] gamma=0.5 may need reconsideration for some scenarios")
    
    return all_scenario_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark 9: Gamma Sensitivity Ablation')
    parser.add_argument('--run_ablation', action='store_true', 
                        help='Run the main gamma ablation study')
    parser.add_argument('--extended', action='store_true',
                        help='Run extended analysis on multiple scenarios')
    args = parser.parse_args()
    
    if args.run_ablation:
        run_gamma_ablation()
        if args.extended:
            run_extended_analysis()
    else:
        print("=" * 80)
        print("BENCHMARK 9: Gamma Sensitivity Ablation")
        print("=" * 80)
        print("\nUsage:")
        print("  python bench_9_gamma_ablation.py --run_ablation")
        print("  python bench_9_gamma_ablation.py --run_ablation --extended")
        print("\nThis script validates that gamma=0.5 is a robust default for FlowAdam.")
        print("=" * 80)


if __name__ == "__main__":
    main()
