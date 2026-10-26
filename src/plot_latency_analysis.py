import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_latency_charts():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "../database/latency_benchmarks.json")
    out_breakdown_png = os.path.join(script_dir, "latency_breakdown_chart.png")
    out_scaling_png = os.path.join(script_dir, "network_scaling_latency_chart.png")

    if not os.path.exists(json_path):
        print(f"[ERROR] Benchmark JSON file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    scenarios = data.get('scenarios', {})
    if not scenarios:
        print("[ERROR] No scenario data found in JSON.")
        return

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # 1. Component Latency Breakdown (Original Network)
    if "ORIGINAL" in scenarios:
        orig_models = scenarios["ORIGINAL"].get('models', {})
        models = ["DQN+GNN", "PPO+GNN"]

        obs_prep = [orig_models[m]['obs_preprocessing']['mean_ms'] for m in models if m in orig_models]
        gnn_pass = [orig_models[m]['gnn_forward_pass']['mean_ms'] for m in models if m in orig_models]
        act_head = [orig_models[m]['action_selection']['mean_ms'] for m in models if m in orig_models]

        x = np.arange(len(models))
        width = 0.55

        fig, ax = plt.subplots(figsize=(9, 6))

        p1 = ax.bar(x, obs_prep, width, label='Obs Preprocessing', color='#66c2a5', edgecolor='black')
        p2 = ax.bar(x, gnn_pass, width, bottom=obs_prep, label='GNN Forward Pass', color='#fc8d62', edgecolor='black')
        p3 = ax.bar(x, act_head, width, bottom=np.array(obs_prep)+np.array(gnn_pass), label='Action Selection Head', color='#8da0cb', edgecolor='black')

        ax.set_ylabel('Inference Time (milliseconds)', fontsize=12, fontweight='bold')
        ax.set_title('Sub-Component Inference Latency Breakdown (Original 6x6 Network)', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(np.array(obs_prep)+np.array(gnn_pass)+np.array(act_head)) * 1.3)
        ax.legend(fontsize=11, loc='upper right')

        for i in range(len(models)):
            tot = obs_prep[i] + gnn_pass[i] + act_head[i]
            ax.annotate(f'{tot:.4f} ms', xy=(x[i], tot), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_breakdown_png, dpi=300)
        print(f"[SUCCESS] Latency breakdown chart saved to: {out_breakdown_png}")
        plt.close()

    # 2. Network Scaling Comparison (Original vs Expanded)
    if "ORIGINAL" in scenarios and "EXPANDED" in scenarios:
        models = ["DQN+GNN", "PPO+GNN"]
        orig_lat = [scenarios["ORIGINAL"]['models'][m]['total_rl_decision']['mean_ms'] for m in models]
        exp_lat = [scenarios["EXPANDED"]['models'][m]['total_rl_decision']['mean_ms'] for m in models]

        orig_p95 = [scenarios["ORIGINAL"]['models'][m]['total_rl_decision']['p95_ms'] for m in models]
        exp_p95 = [scenarios["EXPANDED"]['models'][m]['total_rl_decision']['p95_ms'] for m in models]

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        r1 = ax.bar(x - width/2, orig_lat, width, label='Original (233 Nodes, 520 Edges)', color='#2b5c8f', edgecolor='black')
        r2 = ax.bar(x + width/2, exp_lat, width, label='Expanded (262 Nodes, 631 Edges)', color='#d95f02', edgecolor='black')

        ax.set_ylabel('Mean Total RL Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('RL Decision Latency Scaling Across Network Topology Sizes', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(max(orig_lat), max(exp_lat)) * 1.35)
        ax.legend(fontsize=11, loc='upper left')

        for rect in r1:
            h = rect.get_height()
            ax.annotate(f'{h:.4f} ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
        for rect in r2:
            h = rect.get_height()
            ax.annotate(f'{h:.4f} ms', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                        textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_scaling_png, dpi=300)
        print(f"[SUCCESS] Network scaling latency chart saved to: {out_scaling_png}")
        plt.close()

if __name__ == "__main__":
    generate_latency_charts()
