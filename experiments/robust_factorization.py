"""Robust factorization benchmark (low-rank plus sparse structure)."""

import torch
import torch.nn as nn
import numpy as np
import time
import sys

from flowadam import FlowAdam


class RobustPCA(nn.Module):
    """M ~ UV^T (low-rank part only, sparse part handled via L1)"""
    
    def __init__(self, n, m, rank, init_scale=0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(m, rank) * init_scale)
    
    def forward(self):
        return self.U @ self.V.T


def generate_data(n, m, true_rank, density_sparse, magnitude_sparse, noise, seed):
    """Generate M = L_true + S_true + noise"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    U_true = np.random.randn(n, true_rank) / np.sqrt(true_rank)
    V_true = np.random.randn(m, true_rank) / np.sqrt(true_rank)
    L_true = torch.tensor(U_true @ V_true.T, dtype=torch.float32)
    
    S_true = torch.zeros(n, m)
    mask_sparse = torch.rand(n, m) < density_sparse
    S_true[mask_sparse] = magnitude_sparse * torch.randn(mask_sparse.sum())
    
    M = L_true + S_true + noise * torch.randn(n, m)
    
    return {'M': M, 'L_true': L_true, 'S_true': S_true}


def train_adam(data, config):
    """Train with Adam."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = RobustPCA(config['n'], config['m'], config['model_rank'], config['init_scale'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    M = data['M']
    
    for _ in range(config['n_steps']):
        optimizer.zero_grad()
        L_pred = model()
        diff = M - L_pred
        loss = torch.where(diff.abs() < 1.0, 0.5 * diff ** 2, diff.abs() - 0.5).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        L_pred = model()
        error = ((L_pred - data['L_true']) ** 2).mean().sqrt().item()
    
    return error


def train_adamw(data, config, weight_decay_override=None):
    """Train with AdamW."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = RobustPCA(config['n'], config['m'], config['model_rank'], config['init_scale'])
    wd = weight_decay_override if weight_decay_override is not None else config['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=wd)
    
    M = data['M']
    
    for _ in range(config['n_steps']):
        optimizer.zero_grad()
        L_pred = model()
        diff = M - L_pred
        loss = torch.where(diff.abs() < 1.0, 0.5 * diff ** 2, diff.abs() - 0.5).mean()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        L_pred = model()
        error = ((L_pred - data['L_true']) ** 2).mean().sqrt().item()
    
    return error


def train_flowadam(data, config):
    """Train with FlowAdam (regularization in the loss)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = RobustPCA(config['n'], config['m'], config['model_rank'], config['init_scale'])
    optimizer = FlowAdam(
        model.parameters(), lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )
    
    M = data['M']
    wd = config['weight_decay']
    
    for _ in range(config['n_steps']):
        def closure():
            optimizer.zero_grad()
            L_pred = model()
            diff = M - L_pred
            loss = torch.where(diff.abs() < 1.0, 0.5 * diff ** 2, diff.abs() - 0.5).mean()
            if wd > 0:
                reg = wd * (model.U.pow(2).sum() + model.V.pow(2).sum())
                loss = loss + reg
            loss.backward()
            return loss
        optimizer.step(closure)
    
    with torch.no_grad():
        L_pred = model()
        error = ((L_pred - data['L_true']) ** 2).mean().sqrt().item()
    
    return error, optimizer.get_ode_count()


print("=" * 70)
print("ROBUST PCA: ADAM vs AdamW vs FLOWADAM")
print("=" * 70)

scenarios = [
    {'n': 60, 'm': 80, 'true_rank': 5, 'density_sparse': 0.20, 'magnitude_sparse': 4.0, 'noise': 0.08, 'name': 'Small Heavy'},
    {'n': 80, 'm': 100, 'true_rank': 8, 'density_sparse': 0.20, 'magnitude_sparse': 4.5, 'noise': 0.10, 'name': 'Medium Heavy'},
    {'n': 100, 'm': 120, 'true_rank': 10, 'density_sparse': 0.18, 'magnitude_sparse': 5.0, 'noise': 0.12, 'name': 'Large Heavy'},
]

config_base = {
    'init_scale': 0.1, 'n_steps': 500, 'lr': 0.01, 'weight_decay': 1e-5,
    'switch_sensitivity': 0.90, 'curvature_sensitivity': 0.1, 'ode_t_scale': 0.5,
}

adamw_wds = [1e-5, 1e-4, 1e-3, 1e-2]
seeds = [42, 123, 456, 789, 999]

all_results = []

for scenario in scenarios:
    print(f"\n--- {scenario['name']}: {scenario['n']}x{scenario['m']}, rank={scenario['true_rank']} ---")
    
    adam_results = []
    adamw_results = {wd: [] for wd in adamw_wds}
    flow_results = []
    
    for seed in seeds:
        config = config_base.copy()
        config.update(scenario)
        config['seed'] = seed
        config['model_rank'] = scenario['true_rank'] + 3
        
        data = generate_data(
            scenario['n'], scenario['m'], scenario['true_rank'],
            scenario['density_sparse'], scenario['magnitude_sparse'],
            scenario['noise'], seed
        )
        
        adam_err = train_adam(data, config)
        flow_err, ode = train_flowadam(data, config)
        
        adam_results.append(adam_err)
        flow_results.append(flow_err)
        
        for wd in adamw_wds:
            adamw_err = train_adamw(data, config, weight_decay_override=wd)
            adamw_results[wd].append(adamw_err)
        
        winner = "[OK]" if flow_err < adam_err else "[FAIL]"
        print(f"  Seed {seed}: Adam={adam_err:.4f}, FlowAdam={flow_err:.4f} (ODE={ode}) {winner}")
    
    adam_mean, adam_std = np.mean(adam_results), np.std(adam_results)
    flow_mean, flow_std = np.mean(flow_results), np.std(flow_results)
    
    best_adamw_wd = None
    best_adamw_mean = float('inf')
    for wd in adamw_wds:
        m = np.mean(adamw_results[wd])
        if m < best_adamw_mean:
            best_adamw_mean = m
            best_adamw_wd = wd
    best_adamw_std = np.std(adamw_results[best_adamw_wd])
    
    improvement = (adam_mean - flow_mean) / adam_mean * 100
    wins = sum(1 for a, f in zip(adam_results, flow_results) if f < a)
    
    print(f"\n  Results:")
    print(f"    Adam:     {adam_mean:.3f} +/- {adam_std:.3f}")
    for wd in adamw_wds:
        m, s = np.mean(adamw_results[wd]), np.std(adamw_results[wd])
        marker = " <-- BEST" if wd == best_adamw_wd else ""
        print(f"    AdamW (lambda={wd}): {m:.3f} +/- {s:.3f}{marker}")
    print(f"    FlowAdam: {flow_mean:.3f} +/- {flow_std:.3f}")
    print(f"  Improvement vs Adam: {improvement:+.1f}%, Wins: {wins}/{len(seeds)}")
    
    if best_adamw_mean > adam_mean:
        print(f"  [WARN]  AdamW WORSE than Adam (consistent with matrix completion hypothesis)")
    
    all_results.append({
        'scenario': scenario['name'],
        'adam': f"{adam_mean:.3f}+/-{adam_std:.3f}",
        'adamw_best': f"{best_adamw_mean:.3f}+/-{best_adamw_std:.3f} (lambda={best_adamw_wd})",
        'flowadam': f"{flow_mean:.3f}+/-{flow_std:.3f}",
        'improv': f"{improvement:+.1f}%"
    })

print("\n" + "=" * 70)
print("SUMMARY TABLE (for paper if needed):")
print("=" * 70)
print(f"{'Scenario':<15} {'Adam':<15} {'AdamW*':<25} {'FlowAdam':<15} {'Improv'}")
print("-" * 80)
for r in all_results:
    print(f"{r['scenario']:<15} {r['adam']:<15} {r['adamw_best']:<25} {r['flowadam']:<15} {r['improv']}")
print("=" * 70)
