import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from models.gnn_utils import SimpleGCN, GNNEncoder, NetworkGraph
from models.ppo_agent import ActorCritic, PPOAgent
from models.dqn_agent import DQNNetwork, DQNAgent

def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    Counts trainable and total parameters for a PyTorch module,
    returning a layer-by-layer breakdown.
    """
    trainable_params = 0
    non_trainable_params = 0
    layer_breakdown = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        is_trainable = param.requires_grad
        if is_trainable:
            trainable_params += num_params
        else:
            non_trainable_params += num_params

        layer_breakdown.append({
            'name': name,
            'shape': list(param.shape),
            'parameters': num_params,
            'trainable': is_trainable
        })

    return {
        'total_parameters': trainable_params + non_trainable_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': non_trainable_params,
        'layer_breakdown': layer_breakdown
    }

def analyze_theoretical_complexity():
    """
    Derives theoretical computational complexity formulas based on actual CoTV-E implementation.
    GCN Layer: SimpleGCN performs H = ReLU(A * X * W)
    - Sparse matrix multiplication (A * X): O(|E| * Fin) where |E| is the number of directed edges.
    - Dense linear projection (X * W): O(|V| * Fin * Fout) where |V| is the number of nodes.
    - Total GCN Layer FLOPs: O(|E| * Fin + |V| * Fin * Fout)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_net_path = os.path.join(script_dir, "../network/network.net.xml")
    exp_net_path = os.path.join(script_dir, "../network/expanded_network.net.xml")

    networks = {}
    for name, path in [("Original (6x6)", orig_net_path), ("Expanded (8x8)", exp_net_path)]:
        if os.path.exists(path):
            graph = NetworkGraph(path)
            x, adj = graph.get_torch_data()
            num_nodes = x.shape[0]
            num_edges = int(torch.sum(adj > 0).item())
            networks[name] = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'adj_shape': list(adj.shape)
            }

    return networks

def profile_models():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Instantiate models
    ppo_model = ActorCritic(state_dim=5, action_dim=8, node_features=2, embed_dim=16, use_gnn=True)
    dqn_model = DQNNetwork(state_dim=5, action_dim=8, node_features=2, embed_dim=16, use_gnn=True)

    ppo_summary = count_parameters(ppo_model)
    dqn_summary = count_parameters(dqn_model)

    network_analysis = analyze_theoretical_complexity()

    print("=" * 80)
    print("CoTV-E MODEL ARCHITECTURE & COMPUTATIONAL COMPLEXITY REPORT")
    print("=" * 80)

    print("\n--- 1. PPO+GNN (ActorCritic) Architecture Summary ---")
    print(f"Total Trainable Parameters: {ppo_summary['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters:  {ppo_summary['non_trainable_parameters']:,}")
    print(f"{'Layer Name':<35} | {'Tensor Shape':<20} | {'Param Count':<12}")
    print("-" * 75)
    for layer in ppo_summary['layer_breakdown']:
        print(f"{layer['name']:<35} | {str(layer['shape']):<20} | {layer['parameters']:<12}")

    print("\n--- 2. DQN+GNN (DQNNetwork) Architecture Summary ---")
    print(f"Total Trainable Parameters: {dqn_summary['trainable_parameters']:,}")
    print(f"Non-Trainable Parameters:  {dqn_summary['non_trainable_parameters']:,}")
    print(f"{'Layer Name':<35} | {'Tensor Shape':<20} | {'Param Count':<12}")
    print("-" * 75)
    for layer in dqn_summary['layer_breakdown']:
        print(f"{layer['name']:<35} | {str(layer['shape']):<20} | {layer['parameters']:<12}")

    print("\n--- 3. Dynamic Network Topology Complexity ---")
    print(f"{'Network Scale':<20} | {'Nodes (|V|)':<12} | {'Edges (|E|)':<12} | {'GCN-1 FLOPs (Est)':<20} | {'GCN-2 FLOPs (Est)':<20}")
    print("-" * 88)

    for net_name, data in network_analysis.items():
        v = data['num_nodes']
        e = data['num_edges']
        # GCN-1: Fin=2, Fout=32 -> FLOPs = e*2 + v*2*32
        gcn1_flops = e * 2 + v * 2 * 32
        # GCN-2: Fin=32, Fout=16 -> FLOPs = e*32 + v*32*16
        gcn2_flops = e * 32 + v * 32 * 16
        print(f"{net_name:<20} | {v:<12} | {e:<12} | {gcn1_flops:<20,} | {gcn2_flops:<20,}")

    print("\n--- 4. Theoretical Complexity Expressions ---")
    print("Graph Convolution Layer (SimpleGCN):")
    print("  O(|E| * F_in + |V| * F_in * F_out)")
    print("  where |E| = number of directed edge connections, |V| = number of nodes (junctions),")
    print("        F_in = input feature dimension, F_out = output embedding dimension.")
    print("PPO Actor-Critic Decision Pass:")
    print("  O(GNN_Encoder + (D_state + D_embed) * H_hidden + H_hidden * A_action)")
    print("DQN Q-Network Decision Pass:")
    print("  O(GNN_Encoder + (D_state + D_embed) * H_hidden + H_hidden * A_action)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    profile_models()
