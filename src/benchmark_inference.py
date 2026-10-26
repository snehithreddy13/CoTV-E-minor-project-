import os
import sys
import time
import json
import csv
import platform
import argparse
import numpy as np
import torch
from typing import Dict, List, Any, Tuple

from models.gnn_utils import NetworkGraph, GNNEncoder
from models.ppo_agent import PPOAgent
from models.dqn_agent import DQNAgent

def get_system_metadata() -> Dict[str, Any]:
    return {
        'os': platform.system(),
        'os_release': platform.release(),
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Execution",
    }

def compute_statistics(samples: List[float]) -> Dict[str, float]:
    arr = np.array(samples)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p95_val = float(np.percentile(arr, 95))
    ci95_margin = float(1.96 * std_val / np.sqrt(n)) if n > 1 else 0.0

    return {
        'mean_ms': round(mean_val, 4),
        'median_ms': round(median_val, 4),
        'std_ms': round(std_val, 4),
        'min_ms': round(min_val, 4),
        'max_ms': round(max_val, 4),
        'p95_ms': round(p95_val, 4),
        'ci95_margin_ms': round(ci95_margin, 4)
    }

def benchmark_model_inference(
    model_name: str,
    agent: Any,
    graph: NetworkGraph,
    device: torch.device,
    warmup_iters: int = 50,
    benchmark_iters: int = 500
) -> Dict[str, Any]:
    """
    Benchmarks model inference latency with sub-component timing breakdown under torch.no_grad().
    """
    if hasattr(agent, 'model'):
        agent.model.eval()

    if "dqn" in model_name:
        agent.epsilon = 0.0  # Pure deterministic evaluation mode

    # Sample observation: [node_idx, target_idx, speed, dist_to_target, traffic_density]
    sample_obs = np.array([10.0, 45.0, 0.60, 0.25, 0.10], dtype=np.float32)

    # 1. Warm-up Phase
    with torch.no_grad():
        for _ in range(warmup_iters):
            g_data = graph.get_torch_data()
            if "ppo" in model_name:
                _ = agent.select_action(sample_obs, g_data)
            else:
                _ = agent.select_action(sample_obs, g_data)

    # 2. Timing Measurements
    obs_prep_samples = []
    graph_prep_samples = []
    gnn_forward_samples = []
    action_select_samples = []
    total_rl_samples = []

    # Extract model and GNN for isolated forward pass timing
    target_model = agent.model if hasattr(agent, 'model') else None

    with torch.no_grad():
        for _ in range(benchmark_iters):
            # A. Obs Preprocessing
            t0 = time.perf_counter()
            obs_t = torch.FloatTensor(sample_obs).unsqueeze(0).to(device)
            t1 = time.perf_counter()

            # B. Graph Feature & Adjacency Preprocessing
            t2 = time.perf_counter()
            x, adj = graph.get_torch_data()
            x_dev = x.to(device)
            adj_dev = adj.to(device)
            t3 = time.perf_counter()

            # C. GNN Forward Pass Latency
            t4 = time.perf_counter()
            if target_model and hasattr(target_model, 'gnn'):
                node_idx = obs_t[:, 0].long()
                node_idx = torch.clamp(node_idx, 0, x_dev.shape[0] - 1)
                node_embeds = target_model.gnn(x_dev, adj_dev)
                current_embed = node_embeds[node_idx]
                combined_x = torch.cat([obs_t, current_embed], dim=1)
            else:
                combined_x = obs_t
            t5 = time.perf_counter()

            # D. Policy / Action Selection Head Latency
            t6 = time.perf_counter()
            if "ppo" in model_name and target_model:
                probs = target_model.actor(combined_x)
                action = int(torch.argmax(probs).item())
            elif "dqn" in model_name and target_model:
                q_values = target_model(obs_t, x_dev, adj_dev)
                action = int(torch.argmax(q_values).item())
            else:
                action = 0
            t7 = time.perf_counter()

            # Record sub-millisecond samples in milliseconds
            obs_prep_samples.append((t1 - t0) * 1000.0)
            graph_prep_samples.append((t3 - t2) * 1000.0)
            gnn_forward_samples.append((t5 - t4) * 1000.0)
            action_select_samples.append((t7 - t6) * 1000.0)
            total_rl_samples.append(((t5 - t4) + (t7 - t6)) * 1000.0)

    # Compute parameter counts
    param_count = sum(p.numel() for p in target_model.parameters()) if target_model else 0

    return {
        'model_name': model_name.upper(),
        'parameter_count': param_count,
        'warmup_iterations': warmup_iters,
        'benchmark_iterations': benchmark_iters,
        'obs_preprocessing': compute_statistics(obs_prep_samples),
        'graph_construction': compute_statistics(graph_prep_samples),
        'gnn_forward_pass': compute_statistics(gnn_forward_samples),
        'action_selection': compute_statistics(action_select_samples),
        'total_rl_decision': compute_statistics(total_rl_samples),
        'raw_samples': {
            'gnn_forward_ms': gnn_forward_samples,
            'total_rl_ms': total_rl_samples
        }
    }

def main():
    parser = argparse.ArgumentParser(description="CoTV-E Inference Latency & Profiling Suite")
    parser.add_argument("--warmup", type=int, default=50, help="Warm-up iterations")
    parser.add_argument("--iterations", type=int, default=500, help="Benchmark iterations")
    parser.add_argument("--scenarios", nargs="*", default=["original", "expanded"], help="Scenarios to benchmark")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_dir = os.path.join(script_dir, "../database")
    os.makedirs(database_dir, exist_ok=True)

    sys_meta = get_system_metadata()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("CoTV-E INFERENCE LATENCY BENCHMARKING SUITE")
    print(f"Device: {sys_meta['device_name']} | PyTorch: {sys_meta['pytorch_version']} | OS: {sys_meta['os']}")
    print(f"Warm-up Steps: {args.warmup} | Benchmark Steps: {args.iterations}")
    print("=" * 80 + "\n")

    results_by_scenario = {}
    csv_rows = []

    for scenario in args.scenarios:
        net_filename = "expanded_network.net.xml" if scenario.lower() == "expanded" else "network.net.xml"
        net_path = os.path.join(script_dir, "../network", net_filename)
        if not os.path.exists(net_path):
            print(f"[WARNING] Network path {net_path} not found. Skipping.")
            continue

        graph = NetworkGraph(net_path)
        x_data, adj_data = graph.get_torch_data()
        num_nodes = x_data.shape[0]
        num_edges = int(torch.sum(adj_data > 0).item())

        print("-" * 80)
        print(f"SCENARIO: {scenario.upper()} NETWORK (|V|={num_nodes} nodes, |E|={num_edges} edges)")
        print("-" * 80)

        # Instantiate models
        ppo_agent = PPOAgent(state_dim=5, action_dim=8, use_gnn=True)
        dqn_agent = DQNAgent(state_dim=5, action_dim=8, use_gnn=True)

        scenario_res = {}
        for m_name, agent in [("dqn+gnn", dqn_agent), ("ppo+gnn", ppo_agent)]:
            print(f"Benchmarking {m_name.upper()}...")
            bench_res = benchmark_model_inference(
                model_name=m_name,
                agent=agent,
                graph=graph,
                device=device,
                warmup_iters=args.warmup,
                benchmark_iters=args.iterations
            )
            scenario_res[m_name.upper()] = bench_res

            tot_stats = bench_res['total_rl_decision']
            gnn_stats = bench_res['gnn_forward_pass']
            head_stats = bench_res['action_selection']

            print(f"  -> Total RL Latency: {tot_stats['mean_ms']:.4f} ms ± {tot_stats['std_ms']:.4f} ms "
                  f"(P95: {tot_stats['p95_ms']:.4f} ms | Min: {tot_stats['min_ms']:.4f} ms | Max: {tot_stats['max_ms']:.4f} ms)")
            print(f"     [GNN Forward: {gnn_stats['mean_ms']:.4f} ms | Head Selection: {head_stats['mean_ms']:.4f} ms]")

            csv_rows.append({
                'scenario': scenario.upper(),
                'model': m_name.upper(),
                'nodes': num_nodes,
                'edges': num_edges,
                'parameters': bench_res['parameter_count'],
                'device': sys_meta['device_name'],
                'mean_total_rl_ms': tot_stats['mean_ms'],
                'median_total_rl_ms': tot_stats['median_ms'],
                'p95_total_rl_ms': tot_stats['p95_ms'],
                'std_total_rl_ms': tot_stats['std_ms'],
                'gnn_forward_ms': gnn_stats['mean_ms'],
                'action_head_ms': head_stats['mean_ms'],
                'obs_prep_ms': bench_res['obs_preprocessing']['mean_ms'],
                'graph_prep_ms': bench_res['graph_construction']['mean_ms']
            })

        results_by_scenario[scenario.upper()] = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'models': scenario_res
        }

    # Save JSON benchmark report
    json_path = os.path.join(database_dir, "latency_benchmarks.json")
    with open(json_path, "w") as f:
        json.dump({
            'system_metadata': sys_meta,
            'scenarios': results_by_scenario
        }, f, indent=4)
    print(f"\n[SUCCESS] Saved detailed JSON benchmark report to: {json_path}")

    # Save CSV benchmark report
    csv_path = os.path.join(database_dir, "latency_benchmarks.csv")
    fieldnames = [
        'scenario', 'model', 'nodes', 'edges', 'parameters', 'device',
        'mean_total_rl_ms', 'median_total_rl_ms', 'p95_total_rl_ms',
        'std_total_rl_ms', 'gnn_forward_ms', 'action_head_ms',
        'obs_prep_ms', 'graph_prep_ms'
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[SUCCESS] Saved summary CSV benchmark table to: {csv_path}")

    print("\n" + "=" * 80)
    print("INFERENCE LATENCY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Scenario':<12} | {'Model':<10} | {'Nodes':<6} | {'Params':<8} | {'Mean Latency':<14} | {'P95 Latency':<14} | {'GNN Pass':<12}")
    print("-" * 80)
    for r in csv_rows:
        print(f"{r['scenario']:<12} | {r['model']:<10} | {r['nodes']:<6} | {r['parameters']:<8} | "
              f"{r['mean_total_rl_ms']:>10.4f} ms | {r['p95_total_rl_ms']:>10.4f} ms | {r['gnn_forward_ms']:>8.4f} ms")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
