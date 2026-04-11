"""Jester compute-matched benchmark focused on fair gradient-eval budgets."""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from pathlib import Path
import urllib.request
import matplotlib
import matplotlib.pyplot as plt

from flowadam import FlowAdam

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,  # Ensures text is editable/selectable in PDF
    "ps.fonttype": 42,
    "figure.dpi": 150,
    "mathtext.fontset": "stix",  # Math font matching Times
})

COLOR_ADAM = "#C44E52"       # Muted red
COLOR_FLOWADAM = "#4C72B0"   # Muted blue
COLOR_IMPROVEMENT = "#2E7D32"  # Dark green for improvement annotations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MatrixFactorization(nn.Module):
    """Low-rank matrix factorization: R ~ UV^T"""

    def __init__(self, n_users, n_items, rank, init_scale=0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(n_users, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(n_items, rank) * init_scale)

    def forward(self, user_ids, item_ids):
        return (self.U[user_ids] * self.V[item_ids]).sum(dim=1)


def load_jester_data(data_dir='./data', seed=42, test_ratio=0.2):
    """
    Fast Jester loader (vectorized).
    Expects jester-data-1.xls in:
      ./data/jester/jester-data-1.xls  (preferred)
      ./data/jester-data-1.xls
    File format: first column = number of rated jokes, next 100 columns = ratings or 99 for missing.
    """
    import pandas as pd
    from pathlib import Path
    import numpy as np
    import torch

    paths_to_try = [
        Path(data_dir) / 'jester' / 'jester-data-1.xls',
        Path(data_dir) / 'jester-data-1.xls',
    ]

    jester_path = None
    for p in paths_to_try:
        if p.exists():
            jester_path = p
            break

    if jester_path is None:
        print("Jester data not found!")
        print("Please download from: http://eigentaste.berkeley.edu/dataset/")
        return None

    print(f"Loading Jester from {jester_path}...")

    df = pd.read_excel(jester_path, header=None, engine='xlrd')
    print(f"  Raw data shape: {df.shape}")

    R = df.iloc[:, 1:101].to_numpy(dtype=np.float32)  # shape: (n_users, 100)

    mask = (R != 99.0) & np.isfinite(R)

    user_ids, item_ids = np.where(mask)
    rating_vals = R[user_ids, item_ids].astype(np.float32)

    if rating_vals.size == 0:
        print("No ratings found!")
        return None

    global_mean = float(rating_vals.mean())
    centered = rating_vals - global_mean

    rng = np.random.default_rng(seed)
    idx = rng.permutation(rating_vals.size)
    n_test = int(rating_vals.size * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    n_users, n_items = R.shape

    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Total ratings: {rating_vals.size}")
    print(f"  Train: {train_idx.size}, Test: {test_idx.size}")
    print(f"  Rating range: [{rating_vals.min():.1f}, {rating_vals.max():.1f}]")
    print(f"  Mean rating: {global_mean:.2f}")

    return {
        'train_u': torch.tensor(user_ids[train_idx], dtype=torch.long),
        'train_i': torch.tensor(item_ids[train_idx], dtype=torch.long),
        'train_r': torch.tensor(centered[train_idx], dtype=torch.float32),
        'test_u': torch.tensor(user_ids[test_idx], dtype=torch.long),
        'test_i': torch.tensor(item_ids[test_idx], dtype=torch.long),
        'test_r': torch.tensor(centered[test_idx], dtype=torch.float32),
        'n_users': n_users,
        'n_items': n_items,
        'global_mean': global_mean,
    }


def train_adam(data, config, seed):
    """Train with Adam."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    train_u = data['train_u'].to(device)
    train_i = data['train_i'].to(device)
    train_r = data['train_r'].to(device)

    test_u = data['test_u'].to(device)
    test_i = data['test_i'].to(device)
    test_r = data['test_r'].to(device)

    start = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()

    return test_rmse, elapsed


def train_flowadam(data, config, seed):
    """Train with FlowAdam. Returns RMSE, ODE count, total grad evals, time."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = FlowAdam(
        model.parameters(), lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )

    train_u = data['train_u'].to(device)
    train_i = data['train_i'].to(device)
    train_r = data['train_r'].to(device)
    wd = config['weight_decay']

    test_u = data['test_u'].to(device)
    test_i = data['test_i'].to(device)
    test_r = data['test_r'].to(device)

    start = time.time()
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
    elapsed = time.time() - start

    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()

    return {
        'rmse': test_rmse,
        'ode_count': optimizer.get_ode_count(),
        'total_grad_evals': optimizer.get_total_grad_evals(),
        'time': elapsed
    }


def train_adam_extended(data, config, seed, n_steps):
    """Train with Adam for extended steps (compute-matched)."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )

    train_u = data['train_u'].to(device)
    train_i = data['train_i'].to(device)
    train_r = data['train_r'].to(device)
    test_u = data['test_u'].to(device)
    test_i = data['test_i'].to(device)
    test_r = data['test_r'].to(device)

    start = time.time()
    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()

    return test_rmse, elapsed


def train_adamw(data, config, seed, weight_decay_override=None):
    """Train with AdamW."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    
    wd = weight_decay_override if weight_decay_override is not None else config['weight_decay']
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=wd
    )

    train_u = data['train_u'].to(device)
    train_i = data['train_i'].to(device)
    train_r = data['train_r'].to(device)
    test_u = data['test_u'].to(device)
    test_i = data['test_i'].to(device)
    test_r = data['test_r'].to(device)

    start = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start

    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()

    return test_rmse, elapsed


def main():
    print("=" * 80)
    print("JESTER BENCHMARK: REAL-WORLD MATRIX COMPLETION")
    print("(Includes Compute-Matched Comparison)")
    print("=" * 80)

    data = load_jester_data(data_dir='./data', seed=0)

    if data is None:
        print("\nFailed to load Jester data. Please download manually.")
        return

    config = {
        'rank': 20,
        'init_scale': 0.1,
        'n_steps': 1000,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }

    seeds = [42, 123, 456, 789, 1024]
    adamw_wds = [1e-5, 1e-4, 1e-3, 1e-2]

    print(f"\nConfig: rank={config['rank']}, steps={config['n_steps']}, lr={config['lr']}")
    print(f"Seeds: {seeds}")

    adam_results = []
    adam_ext_results = []
    adamw_results = {wd: [] for wd in adamw_wds}
    flow_results = []
    flow_grad_evals = []

    for seed in seeds:
        data = load_jester_data(data_dir='./data', seed=seed)

        adam_rmse, adam_time = train_adam(data, config, seed)
        flow_result = train_flowadam(data, config, seed)

        adam_ext_rmse, adam_ext_time = train_adam_extended(
            data, config, seed, flow_result['total_grad_evals']
        )
        
        for wd in adamw_wds:
            adamw_rmse, _ = train_adamw(data, config, seed, weight_decay_override=wd)
            adamw_results[wd].append(adamw_rmse)

        adam_results.append(adam_rmse)
        adam_ext_results.append(adam_ext_rmse)
        flow_results.append(flow_result['rmse'])
        flow_grad_evals.append(flow_result['total_grad_evals'])

        imp_vs_adam = (adam_rmse - flow_result['rmse']) / adam_rmse * 100
        imp_vs_ext = (adam_ext_rmse - flow_result['rmse']) / adam_ext_rmse * 100

        print(f"\nSeed {seed}:")
        print(f"  Adam ({config['n_steps']} steps):     RMSE={adam_rmse:.4f} ({adam_time:.1f}s)")
        print(f"  Adam Ext ({flow_result['total_grad_evals']} grad evals): RMSE={adam_ext_rmse:.4f} ({adam_ext_time:.1f}s)")
        print(f"  FlowAdam ({config['n_steps']} steps): RMSE={flow_result['rmse']:.4f} ({flow_result['time']:.1f}s, ODE={flow_result['ode_count']})")
        print(f"  -> vs Adam: {imp_vs_adam:+.1f}%, vs Adam Ext (compute-matched): {imp_vs_ext:+.1f}%")

    adam_mean = np.mean(adam_results)
    adam_std = np.std(adam_results)
    adam_ext_mean = np.mean(adam_ext_results)
    adam_ext_std = np.std(adam_ext_results)
    flow_mean = np.mean(flow_results)
    flow_std = np.std(flow_results)
    grad_eval_mean = np.mean(flow_grad_evals)

    improvement = (adam_mean - flow_mean) / adam_mean * 100
    imp_vs_ext = (adam_ext_mean - flow_mean) / adam_ext_mean * 100
    wins = sum(1 for a, f in zip(adam_results, flow_results) if f < a)
    wins_ext = sum(1 for a, f in zip(adam_ext_results, flow_results) if f < a)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Adam:          {adam_mean:.4f} +/- {adam_std:.4f}")
    print(f"Adam Extended: {adam_ext_mean:.4f} +/- {adam_ext_std:.4f} (avg {grad_eval_mean:.0f} grad evals)")
    
    best_adamw_wd = None
    best_adamw_mean = float('inf')
    for wd in adamw_wds:
        m = np.mean(adamw_results[wd])
        if m < best_adamw_mean:
            best_adamw_mean = m
            best_adamw_wd = wd
    best_adamw_std = np.std(adamw_results[best_adamw_wd])
    
    print(f"\nAdamW results:")
    for wd in adamw_wds:
        m, s = np.mean(adamw_results[wd]), np.std(adamw_results[wd])
        marker = " <-- BEST" if wd == best_adamw_wd else ""
        print(f"  AdamW (lambda={wd}): {m:.4f} +/- {s:.4f}{marker}")
    
    print(f"\nFlowAdam:      {flow_mean:.4f} +/- {flow_std:.4f}")
    print(f"")
    print(f"Improvement vs Adam (same steps):              {improvement:+.1f}% ({wins}/{len(seeds)} wins)")
    print(f"Improvement vs Adam Ext (COMPUTE-MATCHED):     {imp_vs_ext:+.1f}% ({wins_ext}/{len(seeds)} wins)")
    
    if best_adamw_mean > adam_mean:
        print(f"\n[WARN]  AdamW WORSE than Adam (consistent with matrix completion hypothesis)")
    else:
        print(f"\nAdamW better than Adam by {(adam_mean - best_adamw_mean)/adam_mean*100:.1f}%")

    if improvement > 5:
        print("\n[SUCCESS] REAL-WORLD IMPROVEMENT! Add this to the paper.")
    elif improvement > 0:
        print("\n[OK] Small improvement. May be worth reporting.")
    else:
        print("\n[NOTE] No improvement. Jester may also be bias-dominated.")

    generate_jester_figures(seeds, adam_results, flow_results,
                           adam_mean, adam_std, flow_mean, flow_std)

    print("\n" + "=" * 80)
    print("COMPUTE-MATCHED SUMMARY (for paper)")
    print("=" * 80)
    print(f"FlowAdam vs Adam Extended (same compute budget):")
    print(f"  FlowAdam:      {flow_mean:.4f} +/- {flow_std:.4f}")
    print(f"  Adam Extended: {adam_ext_mean:.4f} +/- {adam_ext_std:.4f}")
    print(f"  Improvement:   {imp_vs_ext:+.1f}%")
    print(f"  Avg grad evals: {grad_eval_mean:.0f}")
    print("=" * 80)


def generate_jester_figures(seeds, adam_results, flow_results,
                            adam_mean, adam_std, flow_mean, flow_std):
    """
    Generate IEEE-style figures for Jester benchmark.
    Produces both per-seed comparison and summary bar charts.
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    os.makedirs('results', exist_ok=True)

    improvements = [(a - f) / a * 100 for a, f in zip(adam_results, flow_results)]
    overall_improvement = (adam_mean - flow_mean) / adam_mean * 100
    wins = sum(1 for a, f in zip(adam_results, flow_results) if f < a)

    fig, ax = plt.subplots(figsize=(8, 5))

    seed_labels = [f'Seed {s}' for s in seeds]
    x = np.arange(len(seeds))
    width = 0.35

    bars1 = ax.bar(x - width/2, adam_results, width,
                   label='Adam', color=COLOR_ADAM,
                   edgecolor='black', linewidth=0.8)

    bars2 = ax.bar(x + width/2, flow_results, width,
                   label='FlowAdam', color=COLOR_FLOWADAM,
                   edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Test RMSE (lower is better)')
    ax.set_title(f'Jester Joke Ratings: Real-World Recommender Benchmark\n'
                 f'(24,983 users x 100 jokes, {len(seeds)} seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(seed_labels)
    ax.legend(loc='upper left', framealpha=0.9)

    y_min = min(min(adam_results), min(flow_results)) - 0.2
    y_max = max(max(adam_results), max(flow_results)) + 0.3
    ax.set_ylim([y_min, y_max])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    for i, imp in enumerate(improvements):
        y_pos = flow_results[i] + 0.03
        ax.annotate(
            fr'$\Delta = -{imp:.1f}\%$',
            xy=(x[i] + width/2, y_pos),
            ha='center', va='bottom',
            fontsize=9, fontweight='bold', color=COLOR_IMPROVEMENT,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                     edgecolor=COLOR_IMPROVEMENT, alpha=0.9, linewidth=0.5)
        )

    plt.tight_layout()

    plt.savefig('results/jester_benchmark.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('results/jester_benchmark.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Generated: results/jester_benchmark.pdf/png")

    fig, ax = plt.subplots(figsize=(3.5, 4))  # Narrower for 2 bars

    methods = ['Adam', 'FlowAdam']
    means = [adam_mean, flow_mean]
    stds = [adam_std, flow_std]
    colors = [COLOR_ADAM, COLOR_FLOWADAM]

    x = np.arange(len(methods))
    width = 0.5  # Narrower bars for cleaner look

    bars = ax.bar(x, means, width, yerr=stds, color=colors,
                  capsize=5, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 1.5, 'capthick': 1.5})

    ax.set_ylabel('Test RMSE (lower is better)')
    ax.set_title(f'Jester: Real-World Benchmark\n'
                 f'(1.8M ratings, {len(seeds)} seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    y_min = min(means) - max(stds) - 0.3
    y_max = max(means) + max(stds) + 0.3
    ax.set_ylim([y_min, y_max])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    ax.annotate(
        fr'$\Delta = -{overall_improvement:.1f}\%$',
        xy=(1, flow_mean + flow_std + 0.05),
        ha='center', va='bottom',
        fontsize=11, fontweight='bold', color=COLOR_IMPROVEMENT,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                 edgecolor=COLOR_IMPROVEMENT, alpha=0.9, linewidth=1)
    )

    plt.tight_layout()

    plt.savefig('results/jester_summary.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('results/jester_summary.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("Generated: results/jester_summary.pdf/png")

    print()
    print("Data verification:")
    print("-" * 70)
    for i, seed in enumerate(seeds):
        print(f"  Seed {seed}: Adam={adam_results[i]:.4f}, "
              f"FlowAdam={flow_results[i]:.4f}, delta=-{improvements[i]:.1f}%")
    print("-" * 70)
    print(f"  Mean:    Adam={adam_mean:.4f}+/-{adam_std:.4f}, "
          f"FlowAdam={flow_mean:.4f}+/-{flow_std:.4f}")
    print(f"  Overall: delta=-{overall_improvement:.1f}%, {wins}/{len(seeds)} wins")
    print("-" * 70)
    print("[OK] Figures saved in IEEE-compatible format")


if __name__ == "__main__":
    main()
