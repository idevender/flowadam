"""FlowAdam optimizer implementation with soft momentum injection and optional ODE integration."""

import torch
import math
from torch.optim import Optimizer
from torchdiffeq import odeint


def sync():
    """
    Synchronize CUDA operations for accurate timing.
    No-op if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FlowAdam(Optimizer):
    """
    FlowAdam V2.2 - Adaptive Hybrid Optimizer with Soft Momentum Injection.
    
    Combines Adam optimization with ODE gradient flow for navigating
    difficult loss landscape regions (plateaus and stiff curvature).
    
    Args:
        params: Model parameters
        lr: Learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Adam epsilon (default: 1e-8)
        ode_t_scale: ODE integration time scale (default: 1.0)
        ode_method: ODE solver method (default: 'dopri5')
        ode_tol: ODE solver tolerance (default: 1e-4)
        switch_sensitivity: Plateau detection threshold (default: 0.5)
        curvature_sensitivity: Stiffness detection threshold (default: 2.0)
        momentum_blend_gamma: Blending factor for ODE velocity injection (default: 0.5)
                              Controls: new_momentum = (1-gamma)*old_momentum + gamma*ode_velocity
                              gamma=0: Keep old momentum entirely (no ODE benefit)
                              gamma=1: Full ODE velocity (loses Adam stability)
                              gamma=0.5: Balanced blend (default, validated)
    
    Note: Requires closure that computes loss and calls backward().
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        mode="B",
        ode_t_scale=None,
        ode_method='dopri5',
        ode_tol=1e-4,
        switch_sensitivity=None,
        curvature_sensitivity=None,
        momentum_blend_gamma=0.5,
    ):
        presets = {
            "A": {
                "switch_sensitivity": 0.40,
                "curvature_sensitivity": 3.0,
                "ode_t_scale": 2.0,
            },
            "B": {
                "switch_sensitivity": 0.5,
                "curvature_sensitivity": 2.0,
                "ode_t_scale": 1.0,
            },
        }
        if mode not in presets:
            raise ValueError("mode must be 'A' or 'B'.")

        if switch_sensitivity is None:
            switch_sensitivity = presets[mode]["switch_sensitivity"]
        if curvature_sensitivity is None:
            curvature_sensitivity = presets[mode]["curvature_sensitivity"]
        if ode_t_scale is None:
            ode_t_scale = presets[mode]["ode_t_scale"]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        mode=mode,
                        ode_t_scale=ode_t_scale,
                        ode_method=ode_method,
                        ode_tol=ode_tol,
                        switch_sensitivity=switch_sensitivity,
                        curvature_sensitivity=curvature_sensitivity,
                        momentum_blend_gamma=momentum_blend_gamma)
        super(FlowAdam, self).__init__(params, defaults)
        
        self.state['global'] = {
            'avg_grad_norm': None,
            'avg_curvature': None,
            'step_count': 0,
            'history_ode': [],
            'grad_evals_total': 0,           # Outer-step gradient evaluations
            'ode_nfe_total': 0,              # Total ODE function evaluations
            'ode_nfe_per_trigger': [],       # NFE per trigger for stats
            '_current_ode_nfe': 0,           # Temporary counter for current ODE solve
        }

    def _flatten(self, tensor_list):
        """Flatten list of tensors into single vector."""
        views = []
        for p in tensor_list:
            views.append(p.view(-1))
        return torch.cat(views) if views else torch.tensor([])

    def _unflatten_and_update(self, flat_params, target_params):
        """Update parameters from flattened vector."""
        offset = 0
        for p in target_params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def get_ode_count(self):
        """Return number of ODE triggers so far."""
        return len(self.state['global']['history_ode'])
    
    def get_total_grad_evals(self):
        """
        Return total backward calls including ODE.
        grad_evals_total_including_ode = grad_evals_total + ode_nfe_total
        (Each ode_func call triggers closure() which calls backward())
        """
        stats = self.state['global']
        return stats['grad_evals_total'] + stats['ode_nfe_total']
    
    def get_total_ode_nfe(self):
        """Return total ODE function evaluations."""
        return self.state['global']['ode_nfe_total']
    
    def get_ode_nfe_stats(self):
        """
        Return statistics over ode_nfe_per_trigger.
        Returns dict with mean/median/min/max, or None values if empty.
        """
        nfe_list = self.state['global']['ode_nfe_per_trigger']
        if not nfe_list:
            return {
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'count': 0
            }
        
        import statistics
        return {
            'mean': statistics.mean(nfe_list),
            'median': statistics.median(nfe_list),
            'min': min(nfe_list),
            'max': max(nfe_list),
            'count': len(nfe_list)
        }

    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A callable that computes loss and calls backward().
                     Required for this optimizer.
        """
        if closure is None: 
            raise RuntimeError("FlowAdam requires a closure that computes loss.")
        
        stats = self.state['global']
        
        loss = closure()
        stats['grad_evals_total'] += 1
        
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
        
        prev_grad_vec = stats.get('prev_grad_vec', torch.zeros_like(current_grad_vec))
        current_curvature = torch.norm(current_grad_vec - prev_grad_vec).item()
        
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
            
            nfe_before = stats['_current_ode_nfe']
            stats['_current_ode_nfe'] = 0  # Reset counter for this trigger

            def ode_func(t, y_flat):
                """Gradient flow ODE: dtheta/dt = -nablaL(theta)"""
                stats['_current_ode_nfe'] += 1
                
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
                
                nfe_delta = stats['_current_ode_nfe']
                stats['ode_nfe_per_trigger'].append(nfe_delta)
                stats['ode_nfe_total'] += nfe_delta
                
                displacement = y_final - old_params_flat
                gamma = group['momentum_blend_gamma']  # Blending factor
                offset = 0
                for p in all_params:
                    numel = p.numel()
                    p_disp = displacement[offset:offset+numel].view_as(p)
                    
                    state = self.state[p]
                    if 'exp_avg' in state:
                        ode_velocity = -p_disp / group['lr']
                        
                        current_mom_norm = state['exp_avg'].norm()
                        ode_vel_norm = ode_velocity.norm()
                        scaling_factor = 1.0
                        if ode_vel_norm > current_mom_norm * 5.0:
                            scaling_factor = (current_mom_norm * 5.0) / (ode_vel_norm + 1e-8)

                        state['exp_avg'].mul_(1.0 - gamma).add_(ode_velocity * scaling_factor, alpha=gamma)
                    
                    offset += numel

            except Exception as e:
                nfe_delta = stats['_current_ode_nfe']
                if nfe_delta > 0:
                    stats['ode_nfe_per_trigger'].append(nfe_delta)
                    stats['ode_nfe_total'] += nfe_delta
                
                self._unflatten_and_update(y0, all_params)
                self._adam_step(all_params, group)
        else:
            self._adam_step(all_params, group)

        stats['prev_grad_vec'] = current_grad_vec.detach()
        stats['step_count'] += 1
        return loss

    def _adam_step(self, params, group):
        """Standard Adam update step."""
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


AdaptiveHybridOptimizer = FlowAdam
