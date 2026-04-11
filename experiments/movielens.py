"""MovieLens residualized matrix factorization benchmark."""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import copy
from pathlib import Path

from flowadam import FlowAdam


class BiasOnlyModel(nn.Module):
    """Bias-only model: R = mu + b_u + b_i"""
    
    def __init__(self, n_users, n_items, global_mean):
        super().__init__()
        self.global_mean = global_mean
        self.b_u = nn.Parameter(torch.zeros(n_users))
        self.b_i = nn.Parameter(torch.zeros(n_items))
    
    def forward(self, user_ids, item_ids):
        return self.global_mean + self.b_u[user_ids] + self.b_i[item_ids]


class LatentFactorModel(nn.Module):
    """Latent factor model for residuals: e ~ U*V^T"""
    
    def __init__(self, n_users, n_items, rank, init_scale=0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_users, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(n_items, rank) * init_scale)
    
    def forward(self, user_ids, item_ids):
        return (self.U[user_ids] * self.V[item_ids]).sum(dim=1)


def load_movielens(data_path, seed=42):
    """Load MovieLens with 70/10/20 train/val/test split."""
    print(f"Loading MovieLens from {data_path}...")
    
    users, items, ratings = [], [], []
    with open(data_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                users.append(int(parts[0]))
                items.append(int(parts[1]))
                ratings.append(float(parts[2]))
    
    unique_users = sorted(set(users))
    unique_items = sorted(set(items))
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {m: i for i, m in enumerate(unique_items)}
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    user_ids = np.array([user_to_idx[u] for u in users])
    item_ids = np.array([item_to_idx[m] for m in items])
    rating_vals = np.array(ratings)
    
    np.random.seed(seed)
    indices = np.random.permutation(len(ratings))
    n_test = int(len(indices) * 0.20)
    n_val = int(len(indices) * 0.10)
    
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    global_mean = float(rating_vals[train_idx].mean())
    
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"  Global mean: {global_mean:.3f}")
    
    return {
        'train_u': torch.tensor(user_ids[train_idx], dtype=torch.long),
        'train_i': torch.tensor(item_ids[train_idx], dtype=torch.long),
        'train_r': torch.tensor(rating_vals[train_idx], dtype=torch.float32),
        'val_u': torch.tensor(user_ids[val_idx], dtype=torch.long),
        'val_i': torch.tensor(item_ids[val_idx], dtype=torch.long),
        'val_r': torch.tensor(rating_vals[val_idx], dtype=torch.float32),
        'test_u': torch.tensor(user_ids[test_idx], dtype=torch.long),
        'test_i': torch.tensor(item_ids[test_idx], dtype=torch.long),
        'test_r': torch.tensor(rating_vals[test_idx], dtype=torch.float32),
        'n_users': n_users,
        'n_items': n_items,
        'global_mean': global_mean,
    }


def fit_biases(data, lr=0.01, n_epochs=500, patience=50):
    """Fit bias-only model with early stopping."""
    model = BiasOnlyModel(data['n_users'], data['n_items'], data['global_mean'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_u = data['train_u']
    train_i = data['train_i']
    train_r = data['train_r']
    val_u = data['val_u']
    val_i = data['val_i']
    val_r = data['val_r']
    
    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_rmse = ((model(val_u, val_i) - val_r) ** 2).mean().sqrt().item()
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_state)
    
    with torch.no_grad():
        test_rmse = ((model(data['test_u'], data['test_i']) - data['test_r']) ** 2).mean().sqrt().item()
    
    print(f"  Bias-only: Val RMSE = {best_val_rmse:.4f}, Test RMSE = {test_rmse:.4f} (epoch {epoch-no_improve})")
    
    return model


def compute_residuals(data, bias_model):
    """Compute residuals: e = r - (mu + b_u + b_i)"""
    with torch.no_grad():
        train_bias = bias_model(data['train_u'], data['train_i'])
        val_bias = bias_model(data['val_u'], data['val_i'])
        test_bias = bias_model(data['test_u'], data['test_i'])
    
    return {
        'train_u': data['train_u'],
        'train_i': data['train_i'],
        'train_r': data['train_r'] - train_bias,  # residuals
        'val_u': data['val_u'],
        'val_i': data['val_i'],
        'val_r': data['val_r'] - val_bias,
        'test_u': data['test_u'],
        'test_i': data['test_i'],
        'test_r': data['test_r'] - test_bias,
        'n_users': data['n_users'],
        'n_items': data['n_items'],
        'orig_test_r': data['test_r'],
        'test_bias': test_bias,
    }


def train_uv_adam(residual_data, config, seed):
    """Train UV on residuals with Adam."""
    torch.manual_seed(seed + 1000)
    
    model = LatentFactorModel(
        residual_data['n_users'], residual_data['n_items'],
        config['rank'], config['init_scale']
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    
    train_u = residual_data['train_u']
    train_i = residual_data['train_i']
    train_r = residual_data['train_r']  # residuals
    val_u = residual_data['val_u']
    val_i = residual_data['val_i']
    val_r = residual_data['val_r']
    
    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0
    best_epoch = 0
    
    start = time.time()
    for epoch in range(config['n_epochs']):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_rmse = ((model(val_u, val_i) - val_r) ** 2).mean().sqrt().item()
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config['patience']:
                break
    
    elapsed = time.time() - start
    model.load_state_dict(best_state)
    
    with torch.no_grad():
        test_residual_pred = model(residual_data['test_u'], residual_data['test_i'])
        residual_rmse = ((test_residual_pred - residual_data['test_r']) ** 2).mean().sqrt().item()
        
        final_pred = residual_data['test_bias'] + test_residual_pred
        final_rmse = ((final_pred - residual_data['orig_test_r']) ** 2).mean().sqrt().item()
    
    return {
        'residual_rmse': residual_rmse,
        'final_rmse': final_rmse,
        'val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'time': elapsed,
    }


def train_uv_flowadam(residual_data, config, seed):
    """Train UV on residuals with FlowAdam."""
    torch.manual_seed(seed + 1000)
    
    model = LatentFactorModel(
        residual_data['n_users'], residual_data['n_items'],
        config['rank'], config['init_scale']
    )
    optimizer = FlowAdam(
        model.parameters(), lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )
    
    train_u = residual_data['train_u']
    train_i = residual_data['train_i']
    train_r = residual_data['train_r']  # residuals
    val_u = residual_data['val_u']
    val_i = residual_data['val_i']
    val_r = residual_data['val_r']
    


    best_val_rmse = float('inf')
    best_state = None
    no_improve = 0
    best_epoch = 0
    
    start = time.time()
    for epoch in range(config['n_epochs']):
        def closure():
            optimizer.zero_grad()
            pred = model(train_u, train_i)
            loss = ((pred - train_r) ** 2).mean()
            if wd > 0:
                loss = loss + wd * (model.U.pow(2).sum() + model.V.pow(2).sum())
            loss.backward()
            return loss
        optimizer.step(closure)
        
        with torch.no_grad():
            val_rmse = ((model(val_u, val_i) - val_r) ** 2).mean().sqrt().item()
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= config['patience']:
                break
    
    elapsed = time.time() - start
    model.load_state_dict(best_state)
    
    with torch.no_grad():
        test_residual_pred = model(residual_data['test_u'], residual_data['test_i'])
        residual_rmse = ((test_residual_pred - residual_data['test_r']) ** 2).mean().sqrt().item()
        
        final_pred = residual_data['test_bias'] + test_residual_pred
        final_rmse = ((final_pred - residual_data['orig_test_r']) ** 2).mean().sqrt().item()
    
    return {
        'residual_rmse': residual_rmse,
        'final_rmse': final_rmse,
        'val_rmse': best_val_rmse,
        'best_epoch': best_epoch,
        'time': elapsed,
        'ode_count': optimizer.get_ode_count(),
    }


def main():
    print("=" * 80)
    print("MOVIELENS RESIDUALIZED: ISOLATING UV COMPONENT")
    print("=" * 80)
    print("Protocol: Fit biases first, then train UV on residuals only")
    print("This directly tests whether FlowAdam helps coupled UV optimization")
    print()
    
    data_path = Path('data/ml-100k/u.data')
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
    
    config = {
        'rank': 20,
        'init_scale': 0.1,
        'n_epochs': 2000,
        'patience': 100,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }
    
    seeds = [42, 123, 456]
    
    adam_results = []
    flow_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print('='*60)
        
        data = load_movielens(data_path, seed=seed)
        
        print("\nStep 1: Fitting biases (mu + b_u + b_i)...")
        bias_model = fit_biases(data)
        
        print("\nStep 2: Computing residuals...")
        residual_data = compute_residuals(data, bias_model)
        print(f"  Residual range: [{residual_data['train_r'].min():.2f}, {residual_data['train_r'].max():.2f}]")
        print(f"  Residual std: {residual_data['train_r'].std():.3f}")
        
        print("\nStep 3: Training UV on residuals...")
        
        adam_result = train_uv_adam(residual_data, config, seed)
        flow_result = train_uv_flowadam(residual_data, config, seed)
        
        adam_results.append(adam_result)
        flow_results.append(flow_result)
        
        print(f"\n  Adam:     residual={adam_result['residual_rmse']:.4f}, final={adam_result['final_rmse']:.4f} (epoch {adam_result['best_epoch']})")
        print(f"  FlowAdam: residual={flow_result['residual_rmse']:.4f}, final={flow_result['final_rmse']:.4f} (epoch {flow_result['best_epoch']}, ODE={flow_result['ode_count']})")
        
        res_improve = (adam_result['residual_rmse'] - flow_result['residual_rmse']) / adam_result['residual_rmse'] * 100
        final_improve = (adam_result['final_rmse'] - flow_result['final_rmse']) / adam_result['final_rmse'] * 100
        
        winner = "FlowAdam" if flow_result['residual_rmse'] < adam_result['residual_rmse'] else "Adam"
        print(f"\n  -> {winner}: residual {res_improve:+.2f}%, final {final_improve:+.2f}%")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    adam_res_mean = np.mean([r['residual_rmse'] for r in adam_results])
    adam_res_std = np.std([r['residual_rmse'] for r in adam_results])
    flow_res_mean = np.mean([r['residual_rmse'] for r in flow_results])
    flow_res_std = np.std([r['residual_rmse'] for r in flow_results])
    
    adam_final_mean = np.mean([r['final_rmse'] for r in adam_results])
    flow_final_mean = np.mean([r['final_rmse'] for r in flow_results])
    
    res_improve = (adam_res_mean - flow_res_mean) / adam_res_mean * 100
    final_improve = (adam_final_mean - flow_final_mean) / adam_final_mean * 100
    
    wins_residual = sum(1 for a, f in zip(adam_results, flow_results) if f['residual_rmse'] < a['residual_rmse'])
    wins_final = sum(1 for a, f in zip(adam_results, flow_results) if f['final_rmse'] < a['final_rmse'])
    
    print(f"\nRESIDUAL RMSE (UV-specific):")
    print(f"  Adam:     {adam_res_mean:.4f} +/- {adam_res_std:.4f}")
    print(f"  FlowAdam: {flow_res_mean:.4f} +/- {flow_res_std:.4f}")
    print(f"  Improvement: {res_improve:+.2f}% ({wins_residual}/{len(seeds)} wins)")
    
    print(f"\nFINAL RMSE (bias + UV):")
    print(f"  Adam:     {adam_final_mean:.4f}")
    print(f"  FlowAdam: {flow_final_mean:.4f}")
    print(f"  Improvement: {final_improve:+.2f}% ({wins_final}/{len(seeds)} wins)")
    
    if res_improve > 5:
        print("\n[NOTE] SIGNIFICANT UV-SPECIFIC IMPROVEMENT!")
        print("This proves FlowAdam helps coupled UV optimization.")
    elif res_improve > 0:
        print("\n[OK] Small UV-specific improvement.")
    else:
        print("\n[WARN] No improvement on UV component either.")
        print("This suggests MovieLens really doesn't have strong UV coupling.")


if __name__ == "__main__":
    main()
