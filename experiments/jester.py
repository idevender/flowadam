"""Jester benchmark with tuned baseline and compute-matched comparisons."""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from pathlib import Path
import urllib.request
import matplotlib
import matplotlib.pyplot as plt

from flowadam import FlowAdam
from torch.optim.optimizer import Optimizer
import math


class Lion(Optimizer):
    r"""
    Implements Lion algorithm.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if group['weight_decay'] > 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

class AdaBelief(Optimizer):
    """
    Implements AdaBelief algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay (Mary et al. 2019)
        fixed_decay (boolean, optional): (default: False) This is used when
            weight_decouple is set as True.
            When fixed_decay is True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay is False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$.
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, weight_decouple=weight_decouple, fixed_decay=fixed_decay, rectify=rectify)
        super(AdaBelief, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaBelief, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                amsgrad = group['amsgrad']
                state = self.state[p]

                beta1, beta2 = group['betas']

                if len(state) == 0:
                    state['rho_t'] = torch.tensor(0.0).to(p.device)
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        state['max_exp_avg_var'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_var = state['exp_avg'], state['exp_avg_var']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    if group['weight_decouple']:
                        if group['fixed_decay']:
                            p.data.mul_(1.0 - group['weight_decay'])
                        else:
                            p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    torch.max(max_exp_avg_var, exp_avg_var, out=max_exp_avg_var)
                    denom = (max_exp_avg_var.add(group['eps']).sqrt() / math.sqrt(bias_correction2)).add(group['eps'])
                else:
                    denom = (exp_avg_var.add(group['eps']).sqrt() / math.sqrt(bias_correction2)).add(group['eps'])

                if not group['rectify']:
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    state['rho_t'] = state['rho_t'] * beta2 + (1 - beta2)
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])

                    if rho_t > 4: # state['step'] > 4
                        step_size = group['lr'] * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
                        p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        step_size = group['lr'] / bias_correction1
                        p.data.add_(exp_avg, alpha=-step_size)
        return loss

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


def load_jester_data(data_dir='./data', seed=42, val_ratio=0.1, test_ratio=0.2):
    """
    Fast Jester loader (vectorized) with train/val/test split.

    Splits: train (70%), val (10%), test (20%)
    - Val set: used for hyperparameter sweeps
    - Test set: reported only once with final hyperparams

    Mean-centering uses TRAIN data only (no leakage).

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

    rng = np.random.default_rng(seed)
    idx = rng.permutation(rating_vals.size)
    n_test = int(rating_vals.size * test_ratio)
    n_val = int(rating_vals.size * val_ratio)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_mean = float(rating_vals[train_idx].mean())

    centered = rating_vals - train_mean

    n_users, n_items = R.shape

    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Total ratings: {rating_vals.size}")
    print(f"  Train: {train_idx.size}, Val: {val_idx.size}, Test: {test_idx.size}")
    print(f"  Rating range: [{rating_vals.min():.1f}, {rating_vals.max():.1f}]")
    print(f"  Train mean: {train_mean:.2f} (used for centering)")

    return {
        'train_u': torch.tensor(user_ids[train_idx], dtype=torch.long),
        'train_i': torch.tensor(item_ids[train_idx], dtype=torch.long),
        'train_r': torch.tensor(centered[train_idx], dtype=torch.float32),
        'val_u': torch.tensor(user_ids[val_idx], dtype=torch.long),
        'val_i': torch.tensor(item_ids[val_idx], dtype=torch.long),
        'val_r': torch.tensor(centered[val_idx], dtype=torch.float32),
        'test_u': torch.tensor(user_ids[test_idx], dtype=torch.long),
        'test_i': torch.tensor(item_ids[test_idx], dtype=torch.long),
        'test_r': torch.tensor(centered[test_idx], dtype=torch.float32),
        'n_users': n_users,
        'n_items': n_items,
        'global_mean': train_mean,  # For clarity, but it's train mean
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


def train_adam_extended(data, config, seed, lr, n_steps):
    """Train with Adam for extended steps (compute-matched). Uses tuned LR."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config['weight_decay']
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



def train_adamw(data, config, seed, weight_decay=None):
    """Train with AdamW (decoupled weight decay)."""
    torch.manual_seed(seed + 1000)

    wd = weight_decay if weight_decay is not None else config['weight_decay']

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
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


def train_lion(data, config, seed, lr=None):
    """Train with Lion optimizer."""
    torch.manual_seed(seed + 1000)

    learning_rate = lr if lr is not None else config['lr']

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = Lion(
        model.parameters(), lr=learning_rate, weight_decay=config['weight_decay']
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


def train_adabelief(data, config, seed, lr=None):
    """Train with AdaBelief optimizer."""
    torch.manual_seed(seed + 1000)

    learning_rate = lr if lr is not None else config['lr']

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = AdaBelief(
        model.parameters(), lr=learning_rate,
        weight_decay=config['weight_decay'],
        weight_decouple=True,  # AdamW-style
        rectify=False  # Disable rectification for full-batch
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


def train_lion_extended(data, config, seed, lr, n_steps):
    """Train Lion for extended steps (compute-matched)."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = Lion(
        model.parameters(), lr=lr, weight_decay=config['weight_decay']
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


def train_lbfgs(data, config, seed, max_iter=None):
    """
    Train with L-BFGS (PyTorch implementation).

    L-BFGS is a quasi-Newton method that approximates the Hessian.
    This provides a second-order baseline comparison.

    Uses torch.optim.LBFGS for vectorized ops + autograd (GPU-compatible).
    """
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)

    max_iterations = max_iter if max_iter is not None else 100  # L-BFGS converges faster

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,  # L-BFGS handles step size internally
        max_iter=20,  # Inner iterations per step
        history_size=10,
        line_search_fn='strong_wolfe'
    )

    train_u = data['train_u'].to(device)
    train_i = data['train_i'].to(device)
    train_r = data['train_r'].to(device)
    test_u = data['test_u'].to(device)
    test_i = data['test_i'].to(device)
    test_r = data['test_r'].to(device)
    wd = config['weight_decay']

    n_func_evals = 0

    def closure():
        nonlocal n_func_evals
        n_func_evals += 1
        optimizer.zero_grad()
        pred = model(train_u, train_i)
        loss = ((pred - train_r) ** 2).mean()
        if wd > 0:
            reg = wd * (model.U.pow(2).sum() + model.V.pow(2).sum())
            loss = loss + reg
        loss.backward()
        return loss

    start = time.time()
    for step in range(max_iterations):
        optimizer.step(closure)
    elapsed = time.time() - start

    with torch.no_grad():
        test_rmse = ((model(test_u, test_i) - test_r) ** 2).mean().sqrt().item()

    return test_rmse, elapsed, n_func_evals


def train_adam_with_lr(data, config, seed, lr):
    """Train with Adam using specified LR (for sweep)."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=config['weight_decay']
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


def train_adamw_with_lr_wd(data, config, seed, lr, wd):
    """Train with AdamW using specified LR and WD (for grid sweep)."""
    torch.manual_seed(seed + 1000)

    model = MatrixFactorization(
        data['n_users'], data['n_items'],
        config['rank'], config['init_scale']
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd
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


def run_baseline_sweep(data, config, seed):
    """
    Run LR sweeps for all optimizers to find optimal hyperparameters.

    Uses validation set for selection (not test set).
    Temporarily swaps val for test in the data dict to reuse existing train_* functions.

    Returns results dict with best configuration for each optimizer.
    """
    results = {}

    sweep_data = dict(data)
    sweep_data['test_u'] = data['val_u']
    sweep_data['test_i'] = data['val_i']
    sweep_data['test_r'] = data['val_r']

    print("\n--- Adam LR Sweep (on VAL set) ---")
    adam_lrs = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03]
    adam_best = {'lr': None, 'rmse': float('inf')}
    for lr in adam_lrs:
        rmse, t = train_adam_with_lr(sweep_data, config, seed, lr=lr)
        print(f"  LR={lr:.0e}: val_RMSE={rmse:.4f}")
        if rmse < adam_best['rmse']:
            adam_best = {'lr': lr, 'rmse': rmse}
    results['adam'] = adam_best
    print(f"  Best: LR={adam_best['lr']:.0e}")

    print("\n--- Lion LR Sweep (on VAL set) ---")
    lion_lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    lion_best = {'lr': None, 'rmse': float('inf')}
    for lr in lion_lrs:
        rmse, t = train_lion(sweep_data, config, seed, lr=lr)
        print(f"  LR={lr:.0e}: val_RMSE={rmse:.4f}")
        if rmse < lion_best['rmse']:
            lion_best = {'lr': lr, 'rmse': rmse}
    results['lion'] = lion_best
    print(f"  Best: LR={lion_best['lr']:.0e}")

    print("\n--- AdaBelief LR Sweep (on VAL set) ---")
    belief_lrs = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    belief_best = {'lr': None, 'rmse': float('inf')}
    for lr in belief_lrs:
        rmse, t = train_adabelief(sweep_data, config, seed, lr=lr)
        print(f"  LR={lr:.0e}: val_RMSE={rmse:.4f}")
        if rmse < belief_best['rmse']:
            belief_best = {'lr': lr, 'rmse': rmse}
    results['adabelief'] = belief_best
    print(f"  Best: LR={belief_best['lr']:.0e}")

    print("\n--- AdamW LR x WD Grid Sweep (on VAL set) ---")
    adamw_lrs = [1e-3, 3e-3, 0.01, 0.03]
    adamw_wds = [1e-5, 1e-4, 1e-3, 1e-2]
    adamw_best = {'lr': None, 'wd': None, 'rmse': float('inf')}
    print(f"  {'LR':<8} {'WD':<8} val_RMSE")
    for lr in adamw_lrs:
        for wd in adamw_wds:
            rmse, t = train_adamw_with_lr_wd(sweep_data, config, seed, lr=lr, wd=wd)
            print(f"  {lr:<8.0e} {wd:<8.0e} {rmse:.4f}")
            if rmse < adamw_best['rmse']:
                adamw_best = {'lr': lr, 'wd': wd, 'rmse': rmse}
    results['adamw'] = adamw_best
    print(f"  Best: LR={adamw_best['lr']:.0e}, WD={adamw_best['wd']:.0e}")

    return results


def main():
    print("=" * 80)
    print("JESTER BENCHMARK: COMPREHENSIVE BASELINE COMPARISON (ALL TUNED)")
    print("=" * 80)

    data = load_jester_data(data_dir='./data', seed=0)
    if data is None: return

    config = {
        'rank': 20,
        'init_scale': 0.1,
        'n_steps': 1000,
        'lr': 0.01, # Default placeholder
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 0.1,
        'ode_t_scale': 0.5,
    }

    seeds = [42, 123, 456, 789, 1024]
    sweep_seed = 42

    print("\n" + "=" * 80)
    print("PHASE 1: HYPERPARAMETER SWEEPS (Seed 42, on VALIDATION set)")
    print("=" * 80)

    data = load_jester_data(data_dir='./data', seed=sweep_seed)
    sweep_results = run_baseline_sweep(data, config, sweep_seed)

    print("\n--- FlowAdam LR Sweep (on VAL set) ---")
    flow_lrs = [1e-3, 3e-3, 0.01]
    flow_best = {'lr': None, 'rmse': float('inf')}

    sweep_data = dict(data)
    sweep_data['test_u'] = data['val_u']
    sweep_data['test_i'] = data['val_i']
    sweep_data['test_r'] = data['val_r']

    for lr in flow_lrs:
        cfg = config.copy()
        cfg['lr'] = lr
        rmse = train_flowadam(sweep_data, cfg, sweep_seed)['rmse']
        print(f"  LR={lr:.0e}: val_RMSE={rmse:.4f}")
        if rmse < flow_best['rmse']:
            flow_best = {'lr': lr, 'rmse': rmse}

    print(f"  Best FlowAdam LR: {flow_best['lr']:.0e}")



    adam_best = sweep_results['adam']
    lion_best = sweep_results['lion']
    belief_best = sweep_results['adabelief']
    adamw_best = sweep_results['adamw']

    print("\n" + "=" * 80)
    print("PHASE 2: MULTI-SEED EVALUATION (5 seeds)")
    print("=" * 80)
    print(f"Using best hyperparameters (all tuned on VAL):")
    print(f"  Adam:      LR={adam_best['lr']:.0e}")
    print(f"  FlowAdam:  LR={flow_best['lr']:.0e}")
    print(f"  Lion:      LR={lion_best['lr']:.0e}")
    print(f"  AdaBelief: LR={belief_best['lr']:.0e}")
    print(f"  AdamW:     LR={adamw_best['lr']:.0e}, WD={adamw_best['wd']:.0e}")

    results = {
        'adam': [], 'adam_ext': [], 'adamw': [], 'lion': [], 'lion_ext': [],
        'adabelief': [], 'lbfgs': [], 'flowadam': []
    }
    compute_stats = []

    for seed in seeds:
        data = load_jester_data(data_dir='./data', seed=seed)

        adam_rmse, _ = train_adam_with_lr(data, config, seed, lr=adam_best['lr'])
        results['adam'].append(adam_rmse)

        cfg_flow = config.copy()
        cfg_flow['lr'] = flow_best['lr']
        flow_res = train_flowadam(data, cfg_flow, seed)
        results['flowadam'].append(flow_res['rmse'])

        compute_stats.append({
            'seed': seed,
            'ode_count': flow_res['ode_count'],
            'total_grad_evals': flow_res['total_grad_evals'],
            'time': flow_res['time']
        })

        flow_total_grad = flow_res['total_grad_evals']

        adam_ext_rmse, adam_ext_time = train_adam_extended(data, config, seed, adam_best['lr'], flow_total_grad)
        results['adam_ext'].append(adam_ext_rmse)

        lion_ext_rmse, lion_ext_time = train_lion_extended(data, config, seed, lion_best['lr'], flow_total_grad)
        results['lion_ext'].append(lion_ext_rmse)

        lion_rmse, _ = train_lion(data, config, seed, lr=lion_best['lr'])
        results['lion'].append(lion_rmse)

        belief_rmse, _ = train_adabelief(data, config, seed, lr=belief_best['lr'])
        results['adabelief'].append(belief_rmse)

        adamw_rmse, _ = train_adamw_with_lr_wd(data, config, seed, lr=adamw_best['lr'], wd=adamw_best['wd'])
        results['adamw'].append(adamw_rmse)

        lbfgs_rmse, _, _ = train_lbfgs(data, config, seed)
        results['lbfgs'].append(lbfgs_rmse)

        print(f"\nSeed {seed}:")
        print(f"  Adam (tuned):     {adam_rmse:.4f}")
        print(f"  Adam Ext ({flow_total_grad} steps): {adam_ext_rmse:.4f}  [COMPUTE-MATCHED]")
        print(f"  Lion (tuned):     {lion_rmse:.4f}")
        print(f"  Lion Ext ({flow_total_grad} steps): {lion_ext_rmse:.4f}  [COMPUTE-MATCHED]")
        print(f"  FlowAdam:         {flow_res['rmse']:.4f}  (ODE={flow_res['ode_count']}, grad_evals={flow_total_grad})")
        print(f"  AdamW:            {adamw_rmse:.4f}")
        print(f"  AdaBelief:        {belief_rmse:.4f}")

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nTest RMSE (mean +/- std over 5 seeds):")
    print("-" * 60)
    for k in ['adam', 'adam_ext', 'lion', 'lion_ext', 'flowadam', 'adamw', 'adabelief', 'lbfgs']:
        v = results[k]
        mean_v = np.mean(v)
        std_v = np.std(v)
        print(f"  {k:<12}: {mean_v:.4f} +/- {std_v:.4f}")

    print("\n" + "-" * 60)
    print("FlowAdam Compute Statistics:")
    print("-" * 60)
    avg_ode = np.mean([c['ode_count'] for c in compute_stats])
    avg_grad = np.mean([c['total_grad_evals'] for c in compute_stats])
    avg_time = np.mean([c['time'] for c in compute_stats])
    print(f"  Avg ODE triggers:        {avg_ode:.1f}")
    print(f"  Avg total grad evals:    {avg_grad:.1f}")
    print(f"  Avg wall time:           {avg_time:.1f}s")
    print(f"  Compute overhead ratio:  {avg_grad / config['n_steps']:.2f}x vs Adam's {config['n_steps']} steps")

    print("\n" + "-" * 60)
    print("COMPUTE-MATCHED COMPARISON:")
    print("-" * 60)
    flow_mean = np.mean(results['flowadam'])
    adam_ext_mean = np.mean(results['adam_ext'])
    lion_ext_mean = np.mean(results['lion_ext'])
    imp_vs_adam_ext = (adam_ext_mean - flow_mean) / adam_ext_mean * 100
    imp_vs_lion_ext = (lion_ext_mean - flow_mean) / lion_ext_mean * 100
    print(f"  FlowAdam:              {flow_mean:.4f}")
    print(f"  Adam Ext (matched):    {adam_ext_mean:.4f}  -> FlowAdam is {imp_vs_adam_ext:+.1f}% better")
    print(f"  Lion Ext (matched):    {lion_ext_mean:.4f}  -> FlowAdam is {imp_vs_lion_ext:+.1f}% better")

    wins_adam_ext = sum(1 for a, f in zip(results['adam_ext'], results['flowadam']) if f < a)
    wins_lion_ext = sum(1 for a, f in zip(results['lion_ext'], results['flowadam']) if f < a)
    print(f"\n  Wins vs Adam Ext:  {wins_adam_ext}/5")
    print(f"  Wins vs Lion Ext:  {wins_lion_ext}/5")

    if imp_vs_adam_ext > 0 and imp_vs_lion_ext > 0:
        print("\n  FlowAdam wins EVEN with compute-matched baselines!")
    else:
        print("\n  Compute-matched baselines catch up - gains may be from extra compute.")


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
    print("Figures saved in IEEE-compatible format")


if __name__ == "__main__":
    main()
