import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_plots():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../database/simulation_history.csv")
    output_png = os.path.join(script_dir, "network_comparison_analysis.png")

    if not os.path.isfile(csv_path):
        print(f"[ERROR] Database file not found: {csv_path}")
        return

    scenario_accs = {"ORIGINAL": {}, "EXPANDED": {}}
    scenario_resps = {"ORIGINAL": {}, "EXPANDED": {}}

    with open(csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scen = row.get('scenario', 'ORIGINAL').strip().upper()
            model = row.get('model', '').strip().upper()
            acc_str = row.get('accuracy', '').replace('%', '').strip()
            resp_str = row.get('avg_resp', '').strip()

            if not model or not acc_str or scen not in scenario_accs:
                continue

            try:
                acc_val = float(acc_str)
                resp_val = float(resp_str) if resp_str else 0.0
            except ValueError:
                continue

            if model not in scenario_accs[scen]:
                scenario_accs[scen][model] = []
                scenario_resps[scen][model] = []

            scenario_accs[scen][model].append(acc_val)
            if resp_val > 0:
                scenario_resps[scen][model].append(resp_val)

    models = ["TRACI", "DQN+GNN", "PPO+GNN"]
    orig_acc_means = [np.mean(scenario_accs["ORIGINAL"].get(m, [0])) for m in models]
    exp_acc_means = [np.mean(scenario_accs["EXPANDED"].get(m, [0])) for m in models]

    orig_resp_means = [np.mean(scenario_resps["ORIGINAL"].get(m, [0])) for m in models]
    exp_resp_means = [np.mean(scenario_resps["EXPANDED"].get(m, [0])) for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: RTR Accuracy Comparison
    rects1 = ax1.bar(x - width/2, orig_acc_means, width, label='Original (6x6 Grid, 1 Hosp)', color='#2b5c8f', edgecolor='black')
    rects2 = ax1.bar(x + width/2, exp_acc_means, width, label='Expanded (8x8 Grid, 2 Hosp)', color='#d95f02', edgecolor='black')

    ax1.set_ylabel('RTR Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('RTR Accuracy Across Network Scales', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('+GNN', '\n+GNN') for m in models], fontsize=11, fontweight='bold')
    ax1.set_ylim(50, 105)
    ax1.legend(fontsize=10, loc='upper left')

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Subplot 2: Average Response Time Comparison
    rects3 = ax2.bar(x - width/2, orig_resp_means, width, label='Original (6x6 Grid, 1 Hosp)', color='#2b5c8f', edgecolor='black')
    rects4 = ax2.bar(x + width/2, exp_resp_means, width, label='Expanded (8x8 Grid, 2 Hosp)', color='#d95f02', edgecolor='black')

    ax2.set_ylabel('Avg Response Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Response Time Across Network Scales', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace('+GNN', '\n+GNN') for m in models], fontsize=11, fontweight='bold')
    ax2.set_ylim(0, max(max(orig_resp_means), max(exp_resp_means)) * 1.25)
    ax2.legend(fontsize=10, loc='upper left')

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('CoTV-E Network Scalability Analysis: Original vs Expanded Road Network', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Network comparison plot saved to: {output_png}")

if __name__ == "__main__":
    generate_comparison_plots()
