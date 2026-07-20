"""Two-spirals classification benchmark (extreme variant)."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from flowadam import FlowAdam


def make_extreme_spirals(n_points, noise=0.5):
    """Generate extreme spirals with 1200 deg rotation factor."""
    n = np.sqrt(np.random.rand(n_points, 1)) * 1200 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))), 
            np.hstack((np.zeros(n_points), np.ones(n_points))))


class NarrowSpiralModel(nn.Module):
    """Narrow bottleneck MLP for spiral classification."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.net(x))


def train_model(optimizer_name, X, y, steps=4000):
    """Train model with specified optimizer."""
    torch.manual_seed(999)  # A seed where Adam typically struggles
    model = NarrowSpiralModel()
    criterion = nn.BCELoss()
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == "FlowAdam":
        optimizer = FlowAdam(model.parameters(), lr=0.005, 
                            switch_sensitivity=0.50,   # Conservative - only real plateaus
                            curvature_sensitivity=2.0, # Default curvature
                            ode_t_scale=1.0)           # Standard steps (not aggressive)
    
    losses = []
    accuracies = []
    ode_steps = []
    start_time = time.time()
    
    print(f"--- Training {optimizer_name} ---")
    for i in range(steps):
        def closure():
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.squeeze(), y.float())
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        losses.append(loss.item())
        
        if i % 100 == 0:
            with torch.no_grad():
                pred = (model(X).squeeze() > 0.5).long()
                acc = (pred == y).float().mean()
                accuracies.append(acc.item())
            print(f"[{optimizer_name}] Step {i}: Loss {loss.item():.4f}, Acc: {acc:.2f}")

    if optimizer_name == "FlowAdam":
        ode_steps = optimizer.state['global']['history_ode'].copy()
        print(f"Total ODE Triggers: {len(ode_steps)}/{steps} ({(len(ode_steps)/steps)*100:.1f}%)")

    return model, losses, accuracies, time.time() - start_time, ode_steps


def plot_results(model_adam, model_sgd, model_flow, 
                loss_adam, loss_sgd, loss_flow,
                acc_adam, acc_sgd, acc_flow,
                X, y, ode_steps):
    """Plot comparison results."""
    
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.plot(acc_adam, label='Adam', alpha=0.7, linestyle='--')
    plt.plot(acc_sgd, label='SGD', alpha=0.7, linestyle=':')
    plt.plot(acc_flow, label='FlowAdam', alpha=0.9, linewidth=2)
    
    for step in ode_steps:
        epoch_idx = step // 100
        if epoch_idx < len(acc_flow):
            plt.axvline(x=epoch_idx, color='orange', alpha=0.1, linestyle='--')
    
    plt.title("Test Accuracy")
    plt.xlabel("Epochs (x100)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    def plot_boundary(model, title, idx):
        plt.subplot(1, 4, idx)
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        with torch.no_grad():
            Z = model(grid).reshape(xx.shape)
        plt.contourf(xx, yy, Z.numpy(), alpha=0.8, cmap='RdBu')
        plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy(), edgecolors='k', s=20)
        plt.title(title)

    plot_boundary(model_adam, f"Adam Final: {acc_adam[-1]:.2f}", 2)
    plot_boundary(model_sgd, f"SGD Final: {acc_sgd[-1]:.2f}", 3)
    plot_boundary(model_flow, f"FlowAdam Final: {acc_flow[-1]:.2f}", 4)

    plt.tight_layout()
    plt.show()


def main():
    """Run the extreme spirals benchmark."""
    
    np.random.seed(42)
    X, y = make_extreme_spirals(1000, noise=0.1)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    X = (X - X.mean(dim=0)) / X.std(dim=0)  # Normalize
    
    steps = 4000  # Longer training for hard problem
    model_adam, loss_adam, acc_adam, t_adam, _ = train_model("Adam", X, y, steps)
    model_sgd, loss_sgd, acc_sgd, t_sgd, _ = train_model("SGD", X, y, steps)
    model_flow, loss_flow, acc_flow, t_flow, ode_steps = train_model("FlowAdam", X, y, steps)
    
    print(f"\n=== Timing Summary ===")
    print(f"Adam:     {t_adam:.2f}s")
    print(f"SGD:      {t_sgd:.2f}s")
    print(f"FlowAdam: {t_flow:.2f}s")
    
    print(f"\n=== Final Accuracy ===")
    print(f"Adam:     {acc_adam[-1]:.4f}")
    print(f"SGD:      {acc_sgd[-1]:.4f}")
    print(f"FlowAdam: {acc_flow[-1]:.4f}")
    
    plot_results(model_adam, model_sgd, model_flow,
                loss_adam, loss_sgd, loss_flow,
                acc_adam, acc_sgd, acc_flow,
                X, y, ode_steps)


if __name__ == "__main__":
    main()
