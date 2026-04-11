"""Rosenbrock benchmark comparing Adam and FlowAdam."""

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
from flowadam import FlowAdam


def rosenbrock(params):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    x, y = params[0], params[1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def train_rosenbrock(optimizer_name, start_point=(-1.5, 1.5), steps=500, lr=0.01):
    """Train on Rosenbrock function."""
    
    params = torch.tensor(list(start_point), requires_grad=True, dtype=torch.float32)
    
    if optimizer_name == "Adam":
        optimizer = optim.Adam([params], lr=lr)
    else:
        optimizer = FlowAdam([params], lr=lr, 
                             switch_sensitivity=0.5,   # Default
                             curvature_sensitivity=2.0, # Default
                             ode_t_scale=1.0)          # Default
    
    history = {'x': [], 'y': [], 'loss': [], 'ode_steps': []}
    
    def closure():
        optimizer.zero_grad()
        loss = rosenbrock(params)
        loss.backward()
        return loss
    
    for step in range(steps):
        loss = optimizer.step(closure)
        
        history['x'].append(params[0].item())
        history['y'].append(params[1].item())
        history['loss'].append(loss.item())
        
        if optimizer_name == "FlowAdam":
            history['ode_steps'] = optimizer.state['global']['history_ode'].copy()
    
    ode_count = len(history['ode_steps']) if optimizer_name == "FlowAdam" else 0
    return history, ode_count


def plot_results(hist_adam, hist_flow, ode_steps):
    """Plot optimization trajectories and loss curves."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1 = axes[0]
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    
    ax1.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis', alpha=0.5)
    ax1.plot(hist_adam['x'], hist_adam['y'], 'b.-', label='Adam', alpha=0.7, markersize=2)
    ax1.plot(hist_flow['x'], hist_flow['y'], 'r.-', label='FlowAdam', alpha=0.7, markersize=2)
    ax1.scatter([1], [1], c='green', s=100, marker='*', label='Minimum (1,1)', zorder=5)
    ax1.scatter(hist_adam['x'][0], hist_adam['y'][0], c='black', s=50, marker='o', label='Start')
    
    for step in ode_steps:
        if step < len(hist_flow['x']):
            ax1.scatter(hist_flow['x'][step], hist_flow['y'][step], 
                       c='yellow', s=80, marker='D', edgecolors='red', linewidths=2, zorder=5)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Optimization Trajectories')
    ax1.legend()
    
    ax2 = axes[1]
    ax2.semilogy(hist_adam['loss'], 'b-', label='Adam', alpha=0.8)
    ax2.semilogy(hist_flow['loss'], 'r-', label='FlowAdam', alpha=0.8)
    
    for step in ode_steps:
        ax2.axvline(x=step, color='orange', alpha=0.3, linestyle='--')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title('Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[2]
    dist_adam = [np.sqrt((x-1)**2 + (y-1)**2) for x, y in zip(hist_adam['x'], hist_adam['y'])]
    dist_flow = [np.sqrt((x-1)**2 + (y-1)**2) for x, y in zip(hist_flow['x'], hist_flow['y'])]
    
    ax3.semilogy(dist_adam, 'b-', label='Adam', alpha=0.8)
    ax3.semilogy(dist_flow, 'r-', label='FlowAdam', alpha=0.8)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Distance to (1,1)')
    ax3.set_title('Convergence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_rosenbrock.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to results_rosenbrock.png")


if __name__ == "__main__":
    print("="*60)
    print("BENCHMARK 1: ROSENBROCK FUNCTION")
    print("="*60)
    print("Start: (-1.5, 1.5) -> Target: (1, 1)")
    print()
    
    print("Training with Adam...")
    hist_adam, _ = train_rosenbrock("Adam")
    
    print("Training with FlowAdam...")
    hist_flow, ode_count = train_rosenbrock("FlowAdam")
    
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Adam:     Final Loss = {hist_adam['loss'][-1]:.6f}")
    print(f"          Final Position = ({hist_adam['x'][-1]:.4f}, {hist_adam['y'][-1]:.4f})")
    print()
    print(f"FlowAdam: Final Loss = {hist_flow['loss'][-1]:.6f}")
    print(f"          Final Position = ({hist_flow['x'][-1]:.4f}, {hist_flow['y'][-1]:.4f})")
    print(f"          ODE Triggers = {ode_count}")
    print("="*60)
    
    plot_results(hist_adam, hist_flow, hist_flow['ode_steps'])
