"""Ablation study comparing hard and soft momentum injection."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from torch.optim import Optimizer
from torchdiffeq import odeint

from flowadam import FlowAdam


class FlowAdamV1_HardInjection(Optimizer):
    """FlowAdam V1 - Hard momentum injection (replaces Adam's momentum after ODE)."""
    
    def __init__(self, params, 
                 lr=1e-3, 
                 betas=(0.9, 0.999), 
                 eps=1e-8,
                 ode_t_scale=1.0,
                 ode_method='dopri5',
                 ode_tol=1e-4,
                 switch_sensitivity=0.5, 
                 curvature_sensitivity=2.0): 
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        ode_t_scale=ode_t_scale,
                        ode_method=ode_method,
                        ode_tol=ode_tol,
                        switch_sensitivity=switch_sensitivity,
                        curvature_sensitivity=curvature_sensitivity)
        super(FlowAdamV1_HardInjection, self).__init__(params, defaults)
        
        self.state['global'] = {
            'avg_grad_norm': None,
            'avg_curvature': None,
            'step_count': 0,
            'history_ode': [] 
        }

    def _flatten(self, tensor_list):
        views = []
        for p in tensor_list:
            views.append(p.view(-1))
        return torch.cat(views) if views else torch.tensor([])

    def _unflatten_and_update(self, flat_params, target_params):
        offset = 0
        for p in target_params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def step(self, closure=None):
        if closure is None: 
            raise RuntimeError("Closure required.")
        
        loss = closure()
        
        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)
                    all_grads.append(p.grad)
        
        if not all_params: 
            return loss

        current_grad_vec = self._flatten(all_grads)
        current_norm = torch.norm(current_grad_vec).item()
        
        prev_grad_vec = self.state['global'].get('prev_grad_vec', torch.zeros_like(current_grad_vec))
        current_curvature = torch.norm(current_grad_vec - prev_grad_vec).item()
        
        stats = self.state['global']
        beta_stat = 0.9 
        
        if stats['avg_grad_norm'] is None:
            stats['avg_grad_norm'] = current_norm
            stats['avg_curvature'] = current_curvature
        else:
            stats['avg_grad_norm'] = beta_stat * stats['avg_grad_norm'] + (1-beta_stat) * current_norm
            stats['avg_curvature'] = beta_stat * stats['avg_curvature'] + (1-beta_stat) * current_curvature

        group = self.param_groups[0]
        is_plateau = current_norm < (stats['avg_grad_norm'] * group['switch_sensitivity'])
        is_stiff = current_curvature > (stats['avg_curvature'] * group['curvature_sensitivity'])
        
        use_ode = (is_plateau or is_stiff) and (stats['step_count'] > 10)

        if use_ode:
            stats['history_ode'].append(stats['step_count'])
            
            y0 = self._flatten(all_params)
            old_params_flat = y0.clone()

            def ode_func(t, y_flat):
                self._unflatten_and_update(y_flat, all_params)
                with torch.enable_grad():
                    self.zero_grad()
                    closure()
                new_grads = [p.grad.view(-1) for p in all_params]
                return -torch.clamp(torch.cat(new_grads), -1.0, 1.0)

            t_span = torch.tensor([0.0, group['lr'] * group['ode_t_scale']]).to(y0.device)
            
            try:
                solution = odeint(ode_func, y0, t_span, 
                                  method=group['ode_method'], 
                                  rtol=group['ode_tol'], 
                                  atol=group['ode_tol'])
                
                y_final = solution[-1]
                self._unflatten_and_update(y_final, all_params)
                
                displacement = y_final - old_params_flat
                offset = 0
                for p in all_params:
                    numel = p.numel()
                    p_disp = displacement[offset:offset+numel].view_as(p)
                    
                    state = self.state[p]
                    if 'exp_avg' in state:
                        state['exp_avg'] = -p_disp / group['lr']
                    
                    offset += numel

            except Exception as e:
                self._unflatten_and_update(y0, all_params)
                self._adam_step(all_params, group)
        else:
            self._adam_step(all_params, group)

        stats['prev_grad_vec'] = current_grad_vec.detach()
        stats['step_count'] += 1
        return loss

    def _adam_step(self, params, group):
        beta1, beta2 = group['betas']
        for p in params:
            if p.grad is None: 
                continue
            grad = p.grad.data
            state = self.state[p]
            
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            
            bias_corr1 = 1 - beta1 ** state['step']
            bias_corr2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_corr2) / bias_corr1
            
            denom = exp_avg_sq.sqrt().add_(group['eps'])
            p.data.addcdiv_(exp_avg, denom, value=-step_size)



def make_extreme_spirals(n_points, noise=0.5):
    """Generate extreme spirals with 1200 degree rotation factor."""
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



def train_model(optimizer_name, X, y, steps=4000, lr=0.005):
    """Train with specified optimizer."""
    torch.manual_seed(999)  # Same seed for fair comparison
    model = NarrowSpiralModel()
    criterion = nn.BCELoss()
    
    switch_sens = 0.50
    curv_sens = 2.0
    ode_scale = 1.0
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "FlowAdam_V1_Hard":
        optimizer = FlowAdamV1_HardInjection(
            model.parameters(), lr=lr,
            switch_sensitivity=switch_sens,
            curvature_sensitivity=curv_sens,
            ode_t_scale=ode_scale
        )
    elif optimizer_name == "FlowAdam_V2_Soft":
        optimizer = FlowAdam(
            model.parameters(), lr=lr,
            switch_sensitivity=switch_sens,
            curvature_sensitivity=curv_sens,
            ode_t_scale=ode_scale
        )
    
    losses = []
    accuracies = []
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
                acc = (pred == y).float().mean().item()
            accuracies.append(acc)
            
            ode_count = 0
            if optimizer_name != "Adam":
                ode_count = len(optimizer.state['global']['history_ode'])
            
            print(f"[{optimizer_name}] Step {i}: Loss {loss.item():.4f}, Acc: {acc:.2f} | ODE={ode_count}")
    
    with torch.no_grad():
        pred = (model(X).squeeze() > 0.5).long()
        final_acc = (pred == y).float().mean().item()
    
    elapsed = time.time() - start_time
    
    ode_triggers = 0
    if optimizer_name != "Adam":
        ode_triggers = len(optimizer.state['global']['history_ode'])
        print(f"Total ODE Triggers: {ode_triggers}/{steps}")
    
    return model, losses, accuracies, final_acc, elapsed, ode_triggers


def main():
    print("=" * 60)
    print("ABLATION STUDY: Extreme Spirals - Hard vs Soft Injection")
    print("=" * 60)
    print()
    print("This benchmark tests the SAME problem as bench_3,")
    print("but compares V1 (hard injection) vs V2 (soft injection).")
    print()
    
    np.random.seed(42)
    X, y = make_extreme_spirals(1000, noise=0.1)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    
    steps = 4000
    
    _, loss_adam, acc_adam, final_adam, t_adam, _ = train_model("Adam", X, y, steps)
    print()
    _, loss_v1, acc_v1, final_v1, t_v1, ode_v1 = train_model("FlowAdam_V1_Hard", X, y, steps)
    print()
    _, loss_v2, acc_v2, final_v2, t_v2, ode_v2 = train_model("FlowAdam_V2_Soft", X, y, steps)
    
    print()
    print("=" * 60)
    print("ABLATION RESULTS: Extreme Spirals")
    print("=" * 60)
    print()
    print(f"Adam (baseline):        Final Acc = {final_adam*100:.1f}%  Time = {t_adam:.1f}s")
    print(f"FlowAdam V1 (HARD):     Final Acc = {final_v1*100:.1f}%  Time = {t_v1:.1f}s  ODE = {ode_v1}")
    print(f"FlowAdam V2.1 (SOFT):   Final Acc = {final_v2*100:.1f}%  Time = {t_v2:.1f}s  ODE = {ode_v2}")
    print()
    
    def steps_to_threshold(accs, threshold=0.95):
        for i, a in enumerate(accs):
            if a >= threshold:
                return i * 100
        return -1  # Never reached
    
    steps_adam = steps_to_threshold(acc_adam)
    steps_v1 = steps_to_threshold(acc_v1)
    steps_v2 = steps_to_threshold(acc_v2)
    
    print("Steps to reach 95% accuracy:")
    print(f"  Adam:     {steps_adam if steps_adam > 0 else 'NEVER'}")
    print(f"  V1 Hard:  {steps_v1 if steps_v1 > 0 else 'NEVER'}")
    print(f"  V2 Soft:  {steps_v2 if steps_v2 > 0 else 'NEVER'}")
    print()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(loss_adam[::10], label='Adam', alpha=0.7)
    plt.plot(loss_v1[::10], label='V1 Hard', alpha=0.7)
    plt.plot(loss_v2[::10], label='V2 Soft', alpha=0.7)
    plt.title("Loss Convergence")
    plt.xlabel("Steps (x10)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    x_acc = [i * 100 for i in range(len(acc_adam))]
    plt.plot(x_acc, acc_adam, 'b-', label='Adam', linewidth=2)
    plt.plot(x_acc[:len(acc_v1)], acc_v1, 'r--', label='V1 Hard', linewidth=2)
    plt.plot(x_acc[:len(acc_v2)], acc_v2, 'g-', label='V2 Soft', linewidth=2)
    plt.axhline(y=0.95, color='gray', linestyle=':', label='95% threshold')
    plt.title("Accuracy Over Time")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    names = ['Adam', 'V1 Hard', 'V2 Soft']
    accs = [final_adam * 100, final_v1 * 100, final_v2 * 100]
    colors = ['blue', 'red', 'green']
    bars = plt.bar(names, accs, color=colors, alpha=0.7)
    plt.axhline(y=95, color='gray', linestyle=':', label='95% threshold')
    plt.title("Final Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ablation_extreme_spirals.png', dpi=150)
    print("Plot saved to ablation_extreme_spirals.png")
    plt.show()


if __name__ == "__main__":
    main()
