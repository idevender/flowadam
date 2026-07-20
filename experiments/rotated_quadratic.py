"""Rotated stiff quadratic benchmark for non-diagonal curvature."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from flowadam import FlowAdam


class RotatedStiffValley(nn.Module):
    """High-dimensional rotated stiff valley with off-diagonal curvature."""
    def __init__(self, dim=100, stiffness=1000.0):
        super().__init__()
        self.dim = dim
        
        H_diag = torch.ones(dim)
        H_diag[0] = stiffness
        H_diag[1] = stiffness
        
        torch.manual_seed(42)
        A = torch.randn(dim, dim)
        Q, _ = torch.linalg.qr(A)  # Q is orthogonal
        
        self.register_buffer('H', Q @ torch.diag(H_diag) @ Q.T)
        
        self.theta = nn.Parameter(torch.randn(dim) * 2.0)

    def forward(self):
        return 0.5 * (self.theta @ self.H @ self.theta)


def train_optimizer(opt_name, steps=500):
    """Train with specified optimizer."""
    torch.manual_seed(123)  # Different seed than model gen
    model = RotatedStiffValley(dim=50, stiffness=2000.0)
    
    ode_triggers = 0
    ode_history = []
    
    if opt_name == "Adam":
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
    elif opt_name == "FlowAdam":
        opt = FlowAdam(model.parameters(), lr=0.01, 
                      switch_sensitivity=0.90,
                      curvature_sensitivity=0.1,
                      ode_t_scale=0.5)
    
    losses = []
    start = time.time()
    
    print(f"--- Running {opt_name} ---")
    for i in range(steps):
        def closure():
            opt.zero_grad()
            loss = model()
            loss.backward()
            return loss
        
        loss = opt.step(closure)
        losses.append(loss.item())
        
        if i % 100 == 0:
            print(f"Step {i}: Loss {loss.item():.6f}")

    if opt_name == "FlowAdam":
        ode_history = opt.state['global']['history_ode'].copy()
        ode_triggers = len(ode_history)
        print(f"Total ODE Triggers: {ode_triggers}/{steps} ({(ode_triggers/steps)*100:.1f}%)")
            
    return losses, time.time() - start, ode_history


def plot_results(loss_adam, loss_flow, ode_history, steps):
    """Plot comparison showing FlowAdam advantage."""
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(loss_adam, label='Adam', linestyle='--', color='red', alpha=0.6)
    plt.plot(loss_flow, label='FlowAdam', color='blue', linewidth=1.5, alpha=0.8)
    
    for step in ode_history:
        plt.axvline(x=step, color='orange', alpha=0.2, linestyle='--')
    
    plt.yscale('log')
    plt.xlabel("Steps")
    plt.ylabel("Loss (Log Scale)")
    plt.title("Convergence on Rotated Stiff Valley\n(dim=50, stiffness=2000)")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.subplot(1, 2, 2)
    tail_start = int(steps * 0.5)
    plt.plot(range(tail_start, steps), loss_adam[tail_start:], label='Adam', linestyle='--', color='red')
    plt.plot(range(tail_start, steps), loss_flow[tail_start:], label='FlowAdam', color='blue')
    
    for step in ode_history:
        if step >= tail_start:
            plt.axvline(x=step, color='orange', alpha=0.2, linestyle='--')
    
    plt.yscale('log')
    plt.title("Tail Convergence (Zoomed)")
    plt.xlabel("Steps")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    """Run the stiff valley benchmark."""
    
    steps = 500
    loss_adam, t_adam, _ = train_optimizer("Adam", steps)
    loss_flow, t_flow, ode_history = train_optimizer("FlowAdam", steps)
    
    print(f"\n=== Results Summary ===")
    print(f"Adam:     Final Loss = {loss_adam[-1]:.6f}, Time = {t_adam:.2f}s")
    print(f"FlowAdam: Final Loss = {loss_flow[-1]:.6f}, Time = {t_flow:.2f}s")
    print(f"ODE Triggers: {len(ode_history)}")
    
    if loss_flow[-1] < loss_adam[-1]:
        improvement = (loss_adam[-1] - loss_flow[-1]) / loss_adam[-1] * 100
        print(f"\nFlowAdam WINS! {improvement:.1f}% better final loss")
    else:
        print(f"\nAdam had lower final loss")
    
    plot_results(loss_adam, loss_flow, ode_history, steps)


if __name__ == "__main__":
    main()
