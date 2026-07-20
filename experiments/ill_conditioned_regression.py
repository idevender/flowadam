"""Ill-conditioned regression benchmark comparing optimizers."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from flowadam import FlowAdam



def generate_ill_conditioned_data(n_samples=500, noise=0.1, seed=42):
    """
    Generate regression data with different scales to create ill-conditioning.
    
    Target: y = 100*sin(x1) + 0.01*x2^2 + 10*x3
    
    The different coefficients (100, 0.01, 10) create different gradient scales.
    """
    np.random.seed(seed)
    
    x1 = np.random.uniform(-1, 1, n_samples)
    x2 = np.random.uniform(-10, 10, n_samples)
    x3 = np.random.uniform(-1, 1, n_samples)
    
    y = 100 * np.sin(np.pi * x1) + 0.01 * x2**2 + 10 * x3
    y += noise * np.random.randn(n_samples)
    
    X = np.column_stack([x1, x2, x3]).astype(np.float32)
    y = y.astype(np.float32)
    
    return torch.from_numpy(X), torch.from_numpy(y).view(-1, 1)



class SmallMLP(nn.Module):
    """Small MLP with strong parameter interactions. Uses Tanh for smoothness."""
    def __init__(self, input_dim=3, hidden_size=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)



def train_regression(optimizer_name, config, X_train, y_train, X_test, y_test, verbose=True):
    """Train on ill-conditioned regression."""
    
    torch.manual_seed(config['seed'])
    
    model = SmallMLP(
        input_dim=X_train.shape[1],
        hidden_size=config['hidden_size']
    )
    
    criterion = nn.MSELoss()
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'] * 10, momentum=0.9)
    elif optimizer_name == 'FlowAdam':
        optimizer = FlowAdam(
            model.parameters(),
            lr=config['lr'],
            switch_sensitivity=config['switch_sensitivity'],
            curvature_sensitivity=config['curvature_sensitivity'],
            ode_t_scale=config['ode_t_scale']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    losses = []
    test_losses = []
    ode_triggers = []
    
    start_time = time.time()
    
    if verbose:
        print(f"\n--- Training {optimizer_name} ---")
    
    for step in range(config['n_steps'] + 1):
        def closure():
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + config['l2_lambda'] * l2_reg
            loss.backward()
            return loss
        
        if optimizer_name in ['Adam', 'SGD']:
            optimizer.zero_grad()
            pred = model(X_train)
            loss = criterion(pred, y_train)
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + config['l2_lambda'] * l2_reg
            loss.backward()
            optimizer.step()
        else:
            loss = optimizer.step(closure)
        
        losses.append(loss.item())
        
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
            test_losses.append(test_loss)
        
        if verbose and step % config['log_every'] == 0:
            ode_info = ""
            if optimizer_name == 'FlowAdam':
                n_ode = optimizer.get_ode_count()
                ode_info = f" | ODE={n_ode}"
            print(f"[{optimizer_name}] Step {step:4d}: Train Loss {loss.item():.4f}, "
                  f"Test Loss {test_loss:.4f}{ode_info}")
    
    elapsed = time.time() - start_time
    
    n_ode_final = 0
    if optimizer_name == 'FlowAdam':
        n_ode_final = optimizer.get_ode_count()
        ode_triggers = optimizer.state['global']['history_ode'].copy()
    
    if verbose:
        print(f"[{optimizer_name}] Done! Final Test Loss: {test_losses[-1]:.4f}, Time: {elapsed:.1f}s")
    
    return {
        'optimizer': optimizer_name,
        'losses': losses,
        'test_losses': test_losses,
        'ode_triggers': ode_triggers,
        'n_ode': n_ode_final,
        'elapsed': elapsed,
        'final_loss': losses[-1],
        'final_test_loss': test_losses[-1]
    }



def plot_results(results, config, save_path=None):
    """Plot training curves."""
    
    colors = {'Adam': 'blue', 'SGD': 'orange', 'FlowAdam': 'green'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    for name, r in results.items():
        ax1.plot(r['losses'], label=name, alpha=0.8, color=colors.get(name, 'gray'))
    if 'FlowAdam' in results and results['FlowAdam']['ode_triggers']:
        for t in results['FlowAdam']['ode_triggers']:
            ax1.axvline(x=t, color='green', alpha=0.1, linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for name, r in results.items():
        ax2.plot(r['test_losses'], label=name, alpha=0.8, color=colors.get(name, 'gray'))
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('Test Loss')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    names = list(results.keys())
    final_losses = [results[n]['final_test_loss'] for n in names]
    bars = ax3.bar(names, final_losses, color=[colors.get(n, 'gray') for n in names], alpha=0.8)
    ax3.set_ylabel('Final Test Loss')
    ax3.set_title('Final Test Loss Comparison')
    for bar, loss in zip(bars, final_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05, 
                 f'{loss:.2f}', ha='center', fontsize=10)
    
    plt.suptitle('Ill-Conditioned Regression Benchmark', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")
    
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def print_summary(results, config):
    """Print summary."""
    print("\n" + "=" * 70)
    print("   BENCHMARK 6: ILL-CONDITIONED REGRESSION - RESULTS")
    print("=" * 70)
    print(f"\nHidden size: {config['hidden_size']}, L2 lambda: {config['l2_lambda']}")
    print(f"Training: {config['n_steps']} steps, lr={config['lr']}")
    print("-" * 70)
    
    for name, r in results.items():
        ode_str = f", ODE={r['n_ode']}" if r['n_ode'] > 0 else ""
        print(f"  {name:12s}: TestLoss={r['final_test_loss']:.4f}, "
              f"TrainLoss={r['final_loss']:.4f}, Time={r['elapsed']:.1f}s{ode_str}")
    
    if 'Adam' in results and 'FlowAdam' in results:
        adam_loss = results['Adam']['final_test_loss']
        flow_loss = results['FlowAdam']['final_test_loss']
        
        print("-" * 70)
        if flow_loss < adam_loss:
            improvement = (adam_loss - flow_loss) / adam_loss * 100
            print(f"  FlowAdam WINS! {improvement:.1f}% lower test loss")
        elif flow_loss > adam_loss:
            diff = (flow_loss - adam_loss) / adam_loss * 100
            print(f"  Adam wins by {diff:.1f}%")
        else:
            print(f"  [TIE] Tie!")
    
    print("=" * 70)



def main():
    parser = argparse.ArgumentParser(description='Benchmark 6: Ill-Conditioned Regression')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden size (default: 16)')
    parser.add_argument('--steps', type=int, default=2000, help='Training steps (default: 2000)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--l2', type=float, default=0.001, help='L2 regularization (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting')
    args = parser.parse_args()
    
    X, y = generate_ill_conditioned_data(n_samples=500, noise=0.1, seed=args.seed)
    
    n_train = 400
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    print(f"Data: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}]")
    
    config = {
        'hidden_size': args.hidden,
        'n_steps': args.steps,
        'lr': args.lr,
        'l2_lambda': args.l2,
        
        'switch_sensitivity': 0.4,
        'curvature_sensitivity': 2.5,
        'ode_t_scale': 1.0,
        
        'log_every': 200,
        'seed': args.seed,
    }
    
    print("\n" + "=" * 60)
    print("BENCHMARK 6: ILL-CONDITIONED REGRESSION")
    print("=" * 60)
    print(f"Hidden size: {config['hidden_size']}")
    print(f"L2 regularization: {config['l2_lambda']}")
    print(f"Training: {config['n_steps']} steps, lr={config['lr']}")
    print()
    print("FlowAdam Hyperparameters:")
    print(f"  switch_sensitivity   = {config['switch_sensitivity']}")
    print(f"  curvature_sensitivity = {config['curvature_sensitivity']}")
    print(f"  ode_t_scale          = {config['ode_t_scale']}")
    print("=" * 60)
    
    results = {}
    
    results['Adam'] = train_regression('Adam', config, X_train, y_train, X_test, y_test)
    results['FlowAdam'] = train_regression('FlowAdam', config, X_train, y_train, X_test, y_test)
    
    print_summary(results, config)
    
    if not args.no_plot:
        plot_results(results, config, save_path='bench_6_regression_results.png')


if __name__ == "__main__":
    main()
