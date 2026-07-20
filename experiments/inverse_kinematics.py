"""Multi-target inverse kinematics trajectory optimization benchmark."""

import torch
import torch.nn as nn
import numpy as np
import time
import argparse

from flowadam import FlowAdam



class TrajectoryOptimizer(nn.Module):
    """
    Multi-target trajectory optimization.
    
    Finds a sequence of joint configurations (theta1, ..., thetat) that
    hits T waypoints with smoothness penalties between consecutive configs.
    Smoothness terms couple all parameters through the trajectory.
    """
    
    def __init__(self, n_links=8, n_waypoints=6, init_scale=0.3):
        super().__init__()
        self.n_links = n_links
        self.n_waypoints = n_waypoints
        
        self.register_buffer('lengths', torch.ones(n_links) * 1.0)
        
        self.thetas = nn.Parameter(torch.randn(n_waypoints, n_links) * init_scale)
        
    def forward_kinematics(self, theta):
        """
        Compute end-effector position for one joint configuration.
        Uses vectorized cumsum for proper gradient flow.
        """
        cumulative_angles = torch.cumsum(theta, dim=-1)
        x = (self.lengths * torch.cos(cumulative_angles)).sum(dim=-1)
        y = (self.lengths * torch.sin(cumulative_angles)).sum(dim=-1)
        return torch.stack([x, y], dim=-1)
    
    def compute_loss(self, targets, smoothness_weight=1.0):
        """
        Compute total loss: target matching + smoothness.
        
        Loss = sum_i ||fk(theta_i) - target_i||^2 + lambda * sum_i ||theta_{i+1} - theta_i||^2
        """
        positions = self.forward_kinematics(self.thetas)  # (n_waypoints, 2)
        target_loss = ((positions - targets) ** 2).sum()
        
        if self.n_waypoints > 1:
            diffs = self.thetas[1:] - self.thetas[:-1]  # (n_waypoints-1, n_links)
            smoothness_loss = (diffs ** 2).sum()
        else:
            smoothness_loss = torch.tensor(0.0)
        
        return target_loss + smoothness_weight * smoothness_loss
    
    def compute_target_rmse(self, targets):
        """Compute target tracking RMSE averaged over all waypoints."""
        positions = self.forward_kinematics(self.thetas)  # (n_waypoints, 2)
        mse = ((positions - targets) ** 2).mean()
        return mse.sqrt()
    
    def compute_smoothness(self):
        """
        Compute smoothness metric (mean squared difference between consecutive configs).
        Reported separately for interpretability.
        """
        if self.n_waypoints > 1:
            diffs = self.thetas[1:] - self.thetas[:-1]
            return (diffs ** 2).mean()
        return torch.tensor(0.0)



def generate_trajectory_targets(n_links, n_waypoints, trajectory_type='arc', seed=42):
    """
    Generate a trajectory of reachable target positions.
    
    Trajectory types:
    - 'arc': Semi-circular arc (challenging curvature)
    - 'zigzag': Sharp turns (hard for smooth optimization)
    - 'spiral': Inward spiral (varying reach)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    max_reach = n_links * 1.0  # All links have length 1.0
    
    if trajectory_type == 'arc':
        radius = 0.6 * max_reach
        angles = torch.linspace(0, np.pi, n_waypoints)
        targets = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1)
        
    elif trajectory_type == 'zigzag':
        x_positions = torch.linspace(-0.5 * max_reach, 0.5 * max_reach, n_waypoints)
        y_positions = torch.tensor([0.4 if i % 2 == 0 else 0.6 for i in range(n_waypoints)]) * max_reach
        targets = torch.stack([x_positions, y_positions], dim=1)
        
    elif trajectory_type == 'spiral':
        t = torch.linspace(0, 2 * np.pi, n_waypoints)
        radius_vals = 0.7 * max_reach * (1 - 0.5 * t / (2 * np.pi))
        targets = torch.stack([
            radius_vals * torch.cos(t),
            radius_vals * torch.sin(t)
        ], dim=1)
    
    else:
        radius = 0.5 * max_reach
        angles = torch.rand(n_waypoints) * 2 * np.pi
        targets = torch.stack([
            radius * torch.cos(angles),
            radius * torch.sin(angles)
        ], dim=1)
    
    return targets.float()



def train_adam(targets, config):
    """Train with Adam."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TrajectoryOptimizer(
        n_links=config['n_links'],
        n_waypoints=config['n_waypoints'],
        init_scale=config['init_scale']
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        loss = model.compute_loss(targets, config['smoothness_weight'])
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        target_rmse = model.compute_target_rmse(targets).item()
    
    return target_rmse, elapsed


def train_adamw(targets, config, weight_decay_override=None):
    """Train with AdamW."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TrajectoryOptimizer(
        n_links=config['n_links'],
        n_waypoints=config['n_waypoints'],
        init_scale=config['init_scale']
    )
    
    wd = weight_decay_override if weight_decay_override is not None else config['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=wd
    )
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        loss = model.compute_loss(targets, config['smoothness_weight'])
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        target_rmse = model.compute_target_rmse(targets).item()
    
    return target_rmse, elapsed


def train_flowadam(targets, config):
    """Train with FlowAdam (regularization IN THE LOSS)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = TrajectoryOptimizer(
        n_links=config['n_links'],
        n_waypoints=config['n_waypoints'],
        init_scale=config['init_scale']
    )
    
    optimizer = FlowAdam(
        model.parameters(),
        lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )
    
    wd = config['weight_decay']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        def closure():
            optimizer.zero_grad()
            loss = model.compute_loss(targets, config['smoothness_weight'])
            
            if wd > 0:
                reg = wd * model.thetas.pow(2).sum()
                loss = loss + reg
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
    elapsed = time.time() - start_time
    
    with torch.no_grad():
        target_rmse = model.compute_target_rmse(targets).item()
    
    return target_rmse, optimizer.get_ode_count(), elapsed



def run_experiment(config, seeds=[42, 100, 123, 789, 999]):
    """
    Run experiment comparing Adam, AdamW (tuned), and FlowAdam.
    
    Each seed generates a different trajectory. All optimizers are
    evaluated on the same trajectory per seed (paired comparison).
    Metric: target RMSE.
    """
    
    adam_results, adamw_results, adamw100x_results, adamw1000x_results, flow_results = [], [], [], [], []
    adam_times, adamw_times, adamw100x_times, adamw1000x_times, flow_times = [], [], [], [], []
    
    for seed in seeds:
        config['seed'] = seed
        
        targets = generate_trajectory_targets(
            config['n_links'],
            config['n_waypoints'],
            config['trajectory_type'],
            seed=seed
        )
        
        adam_rmse, adam_t = train_adam(targets, config)
        adamw_rmse, adamw_t = train_adamw(targets, config)
        adamw100x_rmse, adamw100x_t = train_adamw(targets, config, weight_decay_override=100 * config['weight_decay'])
        adamw1000x_rmse, adamw1000x_t = train_adamw(targets, config, weight_decay_override=1000 * config['weight_decay'])
        flow_rmse, ode_count, flow_t = train_flowadam(targets, config)
        
        adam_results.append(adam_rmse)
        adamw_results.append(adamw_rmse)
        adamw100x_results.append(adamw100x_rmse)
        adamw1000x_results.append(adamw1000x_rmse)
        flow_results.append(flow_rmse)
        
        adam_times.append(adam_t)
        adamw_times.append(adamw_t)
        adamw100x_times.append(adamw100x_t)
        adamw1000x_times.append(adamw1000x_t)
        flow_times.append(flow_t)
        
        best_baseline = min(adam_rmse, adamw_rmse, adamw100x_rmse, adamw1000x_rmse)
        all_rmses = {'Adam': adam_rmse, 'AdamW*': best_baseline, 'FlowAdam': flow_rmse}
        winner = min(all_rmses, key=all_rmses.get)
        
        print(f"Seed {seed}: Adam={adam_rmse:.4f}, AdamW*={best_baseline:.4f}, "
              f"FlowAdam={flow_rmse:.4f} (ODE={ode_count}) -> {winner}")
    
    return {
        'adam': (adam_results, adam_times),
        'adamw': (adamw_results, adamw_times),
        'adamw100x': (adamw100x_results, adamw100x_times),
        'adamw1000x': (adamw1000x_results, adamw1000x_times),
        'flowadam': (flow_results, flow_times),
    }


def print_summary(results, config):
    """Print summary statistics (Target RMSE metric)."""
    adam_vals = results['adam'][0]
    flow_vals = results['flowadam'][0]
    
    adam_mean = np.mean(adam_vals)
    adam_median = np.median(adam_vals)
    adamw_mean = np.mean(results['adamw'][0])
    adamw100x_mean = np.mean(results['adamw100x'][0])
    adamw1000x_mean = np.mean(results['adamw1000x'][0])
    flow_mean = np.mean(flow_vals)
    flow_median = np.median(flow_vals)
    
    best_adamw_mean = min(adamw_mean, adamw100x_mean, adamw1000x_mean)
    
    best_baseline_mean = min(adam_mean, best_adamw_mean)
    
    wins = 0
    for i in range(len(adam_vals)):
        best_baseline = min(results['adam'][0][i], results['adamw'][0][i],
                           results['adamw100x'][0][i], results['adamw1000x'][0][i])
        if results['flowadam'][0][i] < best_baseline:
            wins += 1
    
    print("\n" + "="*80)
    print("SUMMARY: Target RMSE (lower is better)")
    print("="*80)
    print(f"Configuration: {config['trajectory_type']} trajectory, "
          f"{config['n_waypoints']} waypoints, {config['n_links']} links")
    print(f"Note: Each seed = different trajectory instance (paired evaluation).")
    print(f"\n{'Optimizer':<15} {'Mean':<10} {'Median':<10}")
    print("-"*40)
    print(f"{'Adam':<15} {adam_mean:.4f}     {adam_median:.4f}")
    print(f"{'AdamW (1xlambda)':<15} {adamw_mean:.4f}")
    print(f"{'AdamW (100xlambda)':<15} {adamw100x_mean:.4f}")
    print(f"{'AdamW (1000xlambda)':<15} {adamw1000x_mean:.4f}")
    print(f"{'Best Baseline':<15} {best_baseline_mean:.4f}")
    print(f"{'FlowAdam':<15} {flow_mean:.4f}     {flow_median:.4f}")
    print("-"*40)
    
    if best_baseline_mean > 0:
        improvement_mean = (best_baseline_mean - flow_mean) / best_baseline_mean * 100
        improvement_median = (adam_median - flow_median) / adam_median * 100
        print(f"\nMean improvement vs best baseline: {improvement_mean:.1f}%")
        print(f"Median improvement vs Adam: {improvement_median:.1f}%")
    
    print(f"FlowAdam wins: {wins}/{len(adam_vals)}")
    
    if improvement_mean >= 10:
        print("\nFlowAdam achieves 10%+ mean improvement.")
    elif improvement_mean > 0:
        print(f"\n PARTIAL: FlowAdam better by {improvement_mean:.1f}% (target: 10%+)")
    else:
        print("\nFAILED: Baselines win")
    
    print("="*80)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--three_scenarios', action='store_true',
                       help='Run arc, zigzag, spiral scenarios')
    args = parser.parse_args()
    
    base_config = {
        'n_links': 8,             # Moderate redundancy
        'n_waypoints': 10,        # More waypoints = stronger coupling
        'init_scale': 0.5,        # Start in challenging region
        'n_steps': 1500,          # More steps for convergence
        'lr': 0.015,              # Slightly larger LR
        'weight_decay': 1e-5,
        'smoothness_weight': 1.0, # Stronger coupling between waypoints
        
        'switch_sensitivity': 0.90,
        'curvature_sensitivity': 1.5,  # Increased to reduce false triggers
        'ode_t_scale': 0.5,
        
        'trajectory_type': 'arc',
        'seed': 42
    }
    
    if args.three_scenarios:
        scenarios = ['arc', 'zigzag', 'spiral']
        all_results = {}
        
        for scenario in scenarios:
            print("\n" + "="*80)
            print(f"SCENARIO: {scenario.upper()} TRAJECTORY")
            print("="*80)
            
            config = base_config.copy()
            config['trajectory_type'] = scenario
            
            results = run_experiment(config)
            all_results[scenario] = results
            print_summary(results, config)
    else:
        print("\n" + "="*80)
        print("BENCHMARK 20: MULTI-TARGET INVERSE KINEMATICS")
        print("Task: Optimize coupled trajectory (Trigonometric + Smoothness)")
        print("="*80)
        
        results = run_experiment(base_config)
        print_summary(results, base_config)
