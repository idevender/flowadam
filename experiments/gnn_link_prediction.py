"""GNN link prediction benchmark with message-passing coupling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import argparse

from flowadam import FlowAdam



class SimpleGCN(nn.Module):
    """
    Simple Graph Convolutional Network for node embeddings.
    
    Message passing: H^{(l+1)} = sigma(A_tilde * H^{(l)} * W^{(l)})
    where A_tilde is the normalized adjacency matrix. Gradients through W
    couple nodes sharing neighbors (topological coupling).
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for _ in range(n_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        
        self.dropout = dropout
        self.n_layers = n_layers
    
    def forward(self, x, adj_norm):
        """
        Args:
            x: Node features (n_nodes, input_dim)
            adj_norm: Normalized adjacency matrix (n_nodes, n_nodes)
        Returns:
            embeddings: Node embeddings (n_nodes, output_dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = adj_norm @ h  # Aggregate
            h = layer(h)       # Transform
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
    def predict_link(self, embeddings, edge_i, edge_j):
        """Predict link probability using dot product."""
        h_i = embeddings[edge_i]
        h_j = embeddings[edge_j]
        return (h_i * h_j).sum(dim=1)
    
    def get_regularization(self):
        """L2 regularization on all parameters."""
        reg = 0
        for layer in self.layers:
            reg += layer.weight.pow(2).sum()
        return reg



def generate_graph_data(n_nodes, avg_degree, feature_dim, seed=42, 
                        edge_ratio=10.0, feature_signal_strength=2.0):
    """
    Generate a random graph with community structure for link prediction.
    
    Creates a graph with latent communities - nodes in same community
    have higher probability of being connected.
    
    MEDIUM-SIGNAL REGIME:
    - edge_ratio: same-community / different-community edge probability ratio
      (default 10:1, matching standard SBM benchmarks)
    - feature_signal_strength: multiplier for community feature signal
      (default 2.0, giving clear but not trivial separation)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_communities = max(3, n_nodes // 50)  # ~50 nodes per community
    community_labels = torch.randint(0, n_communities, (n_nodes,))
    
    edges_i, edges_j = [], []
    
    p_same_pairs = 1.0 / n_communities  # approximate fraction of same-community pairs
    base_prob = avg_degree / n_nodes / (p_same_pairs * edge_ratio + (1 - p_same_pairs))
    
    edge_prob_same = base_prob * edge_ratio  # Higher prob within community
    edge_prob_diff = base_prob               # Lower prob across communities
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if community_labels[i] == community_labels[j]:
                prob = edge_prob_same
            else:
                prob = edge_prob_diff
            
            if np.random.random() < prob:
                edges_i.extend([i, j])
                edges_j.extend([j, i])
    
    edges_i = torch.tensor(edges_i, dtype=torch.long)
    edges_j = torch.tensor(edges_j, dtype=torch.long)
    
    adj = torch.zeros(n_nodes, n_nodes)
    adj[edges_i, edges_j] = 1.0
    
    adj = adj + torch.eye(n_nodes)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    adj_norm = degree_inv_sqrt.unsqueeze(1) * adj * degree_inv_sqrt.unsqueeze(0)
    
    
    one_hot_dim = n_communities
    random_dim = feature_dim - one_hot_dim if feature_dim > one_hot_dim else 0
    
    x_onehot = torch.zeros(n_nodes, one_hot_dim)
    for c in range(n_communities):
        mask = community_labels == c
        x_onehot[mask, c] = feature_signal_strength
        x_onehot[mask] += torch.randn(mask.sum(), one_hot_dim) * 0.3  # Add noise
    
    if random_dim > 0:
        x_random = torch.randn(n_nodes, random_dim) * 0.5
        x = torch.cat([x_onehot, x_random], dim=1)
    else:
        x = x_onehot[:, :feature_dim]
    
    n_edges = len(edges_i) // 2  # Undirected edges
    edge_indices = list(range(0, len(edges_i), 2))  # Take only one direction
    np.random.shuffle(edge_indices)
    
    n_train = int(0.8 * len(edge_indices))
    train_indices = edge_indices[:n_train]
    test_indices = edge_indices[n_train:]
    
    train_edges_i = edges_i[train_indices]
    train_edges_j = edges_j[train_indices]
    test_edges_i = edges_i[test_indices]
    test_edges_j = edges_j[test_indices]
    
    neg_edges_i, neg_edges_j = [], []
    existing_edges = set(zip(edges_i.tolist(), edges_j.tolist()))
    
    while len(neg_edges_i) < len(train_edges_i) + len(test_edges_i):
        i, j = np.random.randint(0, n_nodes, 2)
        if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
            neg_edges_i.append(i)
            neg_edges_j.append(j)
            existing_edges.add((i, j))
            existing_edges.add((j, i))
    
    neg_edges_i = torch.tensor(neg_edges_i, dtype=torch.long)
    neg_edges_j = torch.tensor(neg_edges_j, dtype=torch.long)
    
    train_neg_i = neg_edges_i[:len(train_edges_i)]
    train_neg_j = neg_edges_j[:len(train_edges_j)]
    test_neg_i = neg_edges_i[len(train_edges_i):len(train_edges_i) + len(test_edges_i)]
    test_neg_j = neg_edges_j[len(train_edges_j):len(train_edges_j) + len(test_edges_j)]
    
    train_adj = adj.clone()
    train_adj[test_edges_i, test_edges_j] = 0
    train_adj[test_edges_j, test_edges_i] = 0
    
    degree = train_adj.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[degree_inv_sqrt == float('inf')] = 0
    train_adj_norm = degree_inv_sqrt.unsqueeze(1) * train_adj * degree_inv_sqrt.unsqueeze(0)
    
    return {
        'x': x,
        'adj_norm': train_adj_norm,
        'train_pos_i': train_edges_i, 'train_pos_j': train_edges_j,
        'train_neg_i': train_neg_i, 'train_neg_j': train_neg_j,
        'test_pos_i': test_edges_i, 'test_pos_j': test_edges_j,
        'test_neg_i': test_neg_i, 'test_neg_j': test_neg_j,
        'n_nodes': n_nodes,
        'n_train': len(train_edges_i),
        'n_test': len(test_edges_i)
    }



def compute_auc(scores_pos, scores_neg):
    """Compute AUC-ROC score."""
    labels = torch.cat([torch.ones(len(scores_pos)), torch.zeros(len(scores_neg))])
    scores = torch.cat([scores_pos, scores_neg])
    
    sorted_indices = scores.argsort(descending=True)
    sorted_labels = labels[sorted_indices]
    
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    
    tp = 0
    auc = 0
    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            auc += tp
    
    return auc / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5


def train_adam(data, config):
    """Train with Adam."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = SimpleGCN(
        input_dim=data['x'].shape[1],
        hidden_dim=config['hidden_dim'],
        output_dim=config['embed_dim'],
        n_layers=config['n_layers']
    )
    model.train()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    x, adj = data['x'], data['adj_norm']
    train_pos_i, train_pos_j = data['train_pos_i'], data['train_pos_j']
    train_neg_i, train_neg_j = data['train_neg_i'], data['train_neg_j']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        
        embeddings = model(x, adj)
        
        pos_scores = model.predict_link(embeddings, train_pos_i, train_pos_j)
        neg_scores = model.predict_link(embeddings, train_neg_i, train_neg_j)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        embeddings = model(x, adj)
        test_pos_scores = torch.sigmoid(model.predict_link(embeddings, data['test_pos_i'], data['test_pos_j']))
        test_neg_scores = torch.sigmoid(model.predict_link(embeddings, data['test_neg_i'], data['test_neg_j']))
        test_auc = compute_auc(test_pos_scores, test_neg_scores)
    
    return test_auc, elapsed


def train_adamw(data, config, weight_decay_override=None):
    """Train with AdamW."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = SimpleGCN(
        input_dim=data['x'].shape[1],
        hidden_dim=config['hidden_dim'],
        output_dim=config['embed_dim'],
        n_layers=config['n_layers']
    )
    model.train()
    
    wd = weight_decay_override if weight_decay_override is not None else config['weight_decay']
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=wd
    )
    
    x, adj = data['x'], data['adj_norm']
    train_pos_i, train_pos_j = data['train_pos_i'], data['train_pos_j']
    train_neg_i, train_neg_j = data['train_neg_i'], data['train_neg_j']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        optimizer.zero_grad()
        
        embeddings = model(x, adj)
        pos_scores = model.predict_link(embeddings, train_pos_i, train_pos_j)
        neg_scores = model.predict_link(embeddings, train_neg_i, train_neg_j)
        
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss = pos_loss + neg_loss
        
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        embeddings = model(x, adj)
        test_pos_scores = torch.sigmoid(model.predict_link(embeddings, data['test_pos_i'], data['test_pos_j']))
        test_neg_scores = torch.sigmoid(model.predict_link(embeddings, data['test_neg_i'], data['test_neg_j']))
        test_auc = compute_auc(test_pos_scores, test_neg_scores)
    
    return test_auc, elapsed


def train_flowadam(data, config):
    """Train with FlowAdam (regularization in the loss)."""
    torch.manual_seed(config['seed'] + 1000)
    
    model = SimpleGCN(
        input_dim=data['x'].shape[1],
        hidden_dim=config['hidden_dim'],
        output_dim=config['embed_dim'],
        n_layers=config['n_layers']
    )
    model.train()
    
    optimizer = FlowAdam(
        model.parameters(),
        lr=config['lr'],
        switch_sensitivity=config['switch_sensitivity'],
        curvature_sensitivity=config['curvature_sensitivity'],
        ode_t_scale=config['ode_t_scale']
    )
    
    x, adj = data['x'], data['adj_norm']
    train_pos_i, train_pos_j = data['train_pos_i'], data['train_pos_j']
    train_neg_i, train_neg_j = data['train_neg_i'], data['train_neg_j']
    wd = config['weight_decay']
    
    start_time = time.time()
    for step in range(config['n_steps']):
        def closure():
            optimizer.zero_grad()
            
            embeddings = model(x, adj)
            pos_scores = model.predict_link(embeddings, train_pos_i, train_pos_j)
            neg_scores = model.predict_link(embeddings, train_neg_i, train_neg_j)
            
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            loss = pos_loss + neg_loss
            
            if wd > 0:
                reg = wd * model.get_regularization()
                loss = loss + reg
            
            loss.backward()
            return loss
        
        optimizer.step(closure)
    
    elapsed = time.time() - start_time
    
    model.eval()
    with torch.no_grad():
        embeddings = model(x, adj)
        test_pos_scores = torch.sigmoid(model.predict_link(embeddings, data['test_pos_i'], data['test_pos_j']))
        test_neg_scores = torch.sigmoid(model.predict_link(embeddings, data['test_neg_i'], data['test_neg_j']))
        test_auc = compute_auc(test_pos_scores, test_neg_scores)
    
    return test_auc, optimizer.get_ode_count(), elapsed



def run_experiment(config, seeds=[42, 123, 456, 789, 999]):
    """Run experiment with multiple seeds."""
    
    adam_results, adamw_results, adamw100x_results, adamw1000x_results, flow_results = [], [], [], [], []
    ode_counts = []
    
    for seed in seeds:
        config['seed'] = seed
        
        data = generate_graph_data(
            n_nodes=config['n_nodes'],
            avg_degree=config['avg_degree'],
            feature_dim=config['feature_dim'],
            seed=seed,
            edge_ratio=config.get('edge_ratio', 10.0),
            feature_signal_strength=config.get('feature_signal_strength', 2.0)
        )
        
        adam_auc, _ = train_adam(data, config)
        adamw_auc, _ = train_adamw(data, config)
        adamw100x_auc, _ = train_adamw(data, config, weight_decay_override=1e-3)
        adamw1000x_auc, _ = train_adamw(data, config, weight_decay_override=1e-2)
        flow_auc, ode_count, _ = train_flowadam(data, config)
        
        adam_results.append(adam_auc)
        adamw_results.append(adamw_auc)
        adamw100x_results.append(adamw100x_auc)
        adamw1000x_results.append(adamw1000x_auc)
        flow_results.append(flow_auc)
        ode_counts.append(ode_count)
        
        best_baseline = max(adam_auc, adamw_auc, adamw100x_auc, adamw1000x_auc)
        winner = 'FlowAdam' if flow_auc > best_baseline else 'Baseline'
        
        print(f"  Seed {seed}: Adam={adam_auc:.4f}, AdamW(1x)={adamw_auc:.4f}, "
              f"FlowAdam={flow_auc:.4f} (ODE={ode_count}) -> {winner}")
    
    return {
        'adam': adam_results,
        'adamw': adamw_results,
        'adamw100x': adamw100x_results,
        'adamw1000x': adamw1000x_results,
        'flowadam': flow_results,
        'ode_counts': ode_counts
    }


def run_all_scenarios():
    """Run all GNN scenarios with varying graph sizes and densities.
    
    MEDIUM-SIGNAL REGIME: Uses larger graphs and stronger community signal
    to produce AUC in the 0.75-0.85 range (comparable to standard benchmarks).
    """
    
    scenarios = {
        'medium_strong': {
            'n_nodes': 500,
            'avg_degree': 20,
            'feature_dim': 24,
            'hidden_dim': 64,
            'embed_dim': 32,
            'edge_ratio': 20.0,           # 20:1 same vs. diff (strong signal)
            'feature_signal_strength': 4.0,
            'description': '500 nodes, dense (avg_degree=20), 20:1 edge ratio'
        },
        'large_moderate': {
            'n_nodes': 600,
            'avg_degree': 18,
            'feature_dim': 28,
            'hidden_dim': 72,
            'embed_dim': 36,
            'edge_ratio': 18.0,           # 18:1 (still strong signal)
            'feature_signal_strength': 3.5,
            'description': '600 nodes, moderate (avg_degree=18), 18:1 edge ratio'
        },
        'larger_challenge': {
            'n_nodes': 800,
            'avg_degree': 20,
            'feature_dim': 32,
            'hidden_dim': 80,
            'embed_dim': 40,
            'edge_ratio': 15.0,           # 15:1 (harder but still reasonable)
            'feature_signal_strength': 3.0,
            'description': '800 nodes, moderate-sparse (avg_degree=15), 15:1 edge ratio'
        },
        'easy_calibration': {
            'n_nodes': 500,
            'avg_degree': 20,
            'feature_dim': 24,
            'hidden_dim': 64,
            'embed_dim': 32,
            'edge_ratio': 40.0,            # very strong signal (easy)
            'feature_signal_strength': 6.0,
            'description': '500 nodes, dense, high-signal calibration'
        }

    }
    
    base_config = {
        'n_layers': 2,
        'n_steps': 1000,
        'lr': 0.01,
        'weight_decay': 1e-5,
        'switch_sensitivity': 0.50,   # Between Mode A (0.4) and B (0.9)
        'curvature_sensitivity': 1.5,  # Between Mode A (3.0) and B (0.1)
        'ode_t_scale': 1.0,            # Between Mode A (2.0) and B (0.5)
    }
    
    seeds = [42, 123, 456, 789, 999]
    
    print("=" * 110)
    print("GNN LINK PREDICTION: TOPOLOGICAL COUPLING BENCHMARK")
    print("Coupling Type: Message passing through graph structure (NOT UV^T)")
    print("=" * 110)
    
    all_results = {}
    
    for name, scenario in scenarios.items():
        print(f"\n{'='*100}")
        print(f"SCENARIO: {name}")
        print(f"    {scenario['description']}")
        print("=" * 100)
        
        config = base_config.copy()
        config.update(scenario)
        
        results = run_experiment(config, seeds)
        
        adam_mean, adam_std = np.mean(results['adam']), np.std(results['adam'])
        adamw_mean = np.mean(results['adamw'])
        adamw100x_mean = np.mean(results['adamw100x'])
        adamw1000x_mean = np.mean(results['adamw1000x'])
        flow_mean, flow_std = np.mean(results['flowadam']), np.std(results['flowadam'])
        
        best_adamw = max(adamw_mean, adamw100x_mean, adamw1000x_mean)
        best_baseline = max(adam_mean, best_adamw)
        improv = (flow_mean - best_baseline) / best_baseline * 100 if flow_mean > best_baseline else -(best_baseline - flow_mean) / best_baseline * 100
        
        wins = sum(1 for a, f in zip(results['adam'], results['flowadam']) if f > a)
        
        all_results[name] = {
            'adam_mean': adam_mean, 'adam_std': adam_std,
            'adamw_best': best_adamw,
            'flow_mean': flow_mean, 'flow_std': flow_std,
            'improv': improv,
            'wins': wins,
            'ode_counts': results['ode_counts']
        }
        
        print(f"\n    Summary:")
        print(f"      Adam AUC:     {adam_mean:.4f}+/-{adam_std:.4f}")
        print(f"      AdamW (1xlambda):  {adamw_mean:.4f}")
        print(f"      AdamW (100xlambda):{adamw100x_mean:.4f}")
        print(f"      AdamW (1000xlambda):{adamw1000x_mean:.4f}")
        print(f"      Best AdamW:   {best_adamw:.4f}")
        print(f"      FlowAdam:     {flow_mean:.4f}+/-{flow_std:.4f}")
        print(f"    Improvement vs best baseline: {improv:+.1f}%, Wins: {wins}/5")
        print(f"    Avg ODE triggers: {np.mean(results['ode_counts']):.0f}")
        
        if best_adamw < adam_mean:
            print(f"    [WARN]  AdamW WORSE than Adam")
        else:
            print(f"    AdamW vs Adam: {(best_adamw - adam_mean)/adam_mean*100:+.1f}%")
    
    print("\n" + "=" * 100)
    print("SUMMARY: GNN TOPOLOGICAL COUPLING BENCHMARK")
    print("(Higher AUC is better)")
    print("=" * 100)
    print(f"{'Scenario':<25} {'Adam AUC':<15} {'FlowAdam AUC':<15} {'Improvement':<12} {'Wins'}")
    print("-" * 100)
    
    total_wins = 0
    for name, r in all_results.items():
        print(f"{name:<25} {r['adam_mean']:.4f}+/-{r['adam_std']:.3f}   {r['flow_mean']:.4f}+/-{r['flow_std']:.3f}   {r['improv']:+.1f}%        {r['wins']}/5")
        total_wins += r['wins']
    
    print("-" * 100)
    print(f"TOTAL: FlowAdam wins {total_wins}/15 comparisons")
    print("=" * 100)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark 17: GNN Link Prediction')
    parser.add_argument('--all', action='store_true', help='Run all scenarios')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.all:
        run_all_scenarios()
    else:
        print("=" * 80)
        print("GNN LINK PREDICTION - QUICK TEST (Medium-Signal Regime)")
        print("Run with --all for full benchmark")
        print("=" * 80)
        
        config = {
            'n_nodes': 500,              # Larger for stronger topological signal
            'avg_degree': 20,
            'feature_dim': 24,
            'hidden_dim': 64,
            'embed_dim': 32,
            'n_layers': 2,
            'n_steps': 800,
            'lr': 0.01,
            'weight_decay': 1e-5,
            'switch_sensitivity': 0.50,   # Tuned for GNNs
            'curvature_sensitivity': 1.5,  # Tuned for GNNs
            'ode_t_scale': 1.0,            # Tuned for GNNs
            'edge_ratio': 20.0,           # 20:1 same vs diff community (strong)
            'feature_signal_strength': 4.0,  # Strong community features
            'seed': args.seed
        }
        
        data = generate_graph_data(
            n_nodes=config['n_nodes'],
            avg_degree=config['avg_degree'],
            feature_dim=config['feature_dim'],
            seed=config['seed'],
            edge_ratio=config['edge_ratio'],
            feature_signal_strength=config['feature_signal_strength']
        )
        
        print(f"\nGraph: {config['n_nodes']} nodes, avg_degree={config['avg_degree']}")
        print(f"Train edges: {data['n_train']}, Test edges: {data['n_test']}")
        
        adam_auc, adam_time = train_adam(data, config)
        adamw_auc, adamw_time = train_adamw(data, config)
        flow_auc, ode, flow_time = train_flowadam(data, config)
        
        print(f"\n{'Optimizer':<15} {'Test AUC':<12} {'Time'}")
        print("-" * 40)
        print(f"{'Adam':<15} {adam_auc:.4f}       {adam_time:.1f}s")
        print(f"{'AdamW':<15} {adamw_auc:.4f}       {adamw_time:.1f}s")
        print(f"{'FlowAdam':<15} {flow_auc:.4f}       {flow_time:.1f}s (ODE={ode})")
        
        best = max(adam_auc, adamw_auc)
        if flow_auc > best:
            improv = (flow_auc - best) / best * 100
            print(f"\n[NOTE] FlowAdam WINS by {improv:.1f}%")
        else:
            print(f"\n[FAIL] Baseline wins")


if __name__ == "__main__":
    main()
