import os
import numpy as np
import matplotlib.pyplot as plt

def calculate_rtr_accuracy(t_actual: float, t_ideal: float, alpha: float) -> float:
    """
    Calculate RTR Accuracy given actual travel time, ideal travel time, and alpha (overhead scale factor).
    
    Formula:
        delta_t = max(t_actual - t_ideal, 0)
        t_adj   = max(t_ideal + alpha * delta_t, t_ideal)
        RTR_Acc = min(t_ideal / t_adj, 1.0) * 100
    """
    delta_t = max(t_actual - t_ideal, 0.0)
    t_adj = max(t_ideal + alpha * delta_t, t_ideal)
    if t_adj <= 0:
        return 0.0
    rtr_acc = min(t_ideal / t_adj, 1.0) * 100.0
    return rtr_acc

def main():
    # Model data provided by user / empirical results
    models_data = {
        "TraCI": {
            "t_actual": 116.27,
            "t_ideal": 103.17,
            "alpha_default": 1.00,
            "color": "#1f77b4",  # Blue
            "linestyle": "-"
        },
        "DQN+GNN": {
            "t_actual": 143.08,
            "t_ideal": 103.02,
            "alpha_default": 0.60,
            "color": "#ff7f0e",  # Orange
            "linestyle": "--"
        },
        "PPO+GNN": {
            "t_actual": 147.69,
            "t_ideal": 106.60,
            "alpha_default": 0.28,
            "color": "#2ca02c",  # Green
            "linestyle": "-."
        }
    }

    # Generate alpha sweep from 0.0 to 1.0
    alphas = np.linspace(0.0, 1.0, 201)

    # Plot styling
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    print("=" * 65)
    print("SENSITIVITY ANALYSIS: OVERHEAD SCALE PARAMETER (alpha)")
    print("=" * 65)

    for name, data in models_data.items():
        t_actual = data["t_actual"]
        t_ideal = data["t_ideal"]
        alpha_def = data["alpha_default"]
        color = data["color"]
        ls = data["linestyle"]

        # Compute accuracy curve across alpha range
        acc_curve = [calculate_rtr_accuracy(t_actual, t_ideal, a) for a in alphas]
        
        # Calculate default operating point
        default_acc = calculate_rtr_accuracy(t_actual, t_ideal, alpha_def)

        print(f"Model: {name:<10} | Actual: {t_actual:>6.2f}s | Ideal: {t_ideal:>6.2f}s | "
              f"Default alpha: {alpha_def:.2f} => Accuracy: {default_acc:.2f}%")

        # Plot line
        ax.plot(alphas, acc_curve, label=f"{name} (t_act={t_actual}s, t_ideal={t_ideal}s)", 
                color=color, linewidth=2.5, linestyle=ls)

        # Plot scatter marker at hardcoded operating point
        ax.scatter([alpha_def], [default_acc], color=color, s=80, zorder=5, edgecolors='black', linewidth=1.5)

        # Annotate hardcoded point
        offset_y = 3 if name == "DQN+GNN" else (-4 if name == "TraCI" else 3)
        offset_x = -0.05 if alpha_def > 0.5 else 0.03
        ax.annotate(
            f"α = {alpha_def:.2f} ({default_acc:.1f}%)",
            xy=(alpha_def, default_acc),
            xytext=(alpha_def + offset_x, default_acc + offset_y),
            arrowprops=dict(facecolor=color, shrink=0.08, width=1, headwidth=6),
            fontsize=9,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8)
        )

    # Customize plot labels and title
    ax.set_title("Sensitivity Analysis of Overhead Scale Factor (α) on RTR Accuracy", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Overhead Scale Factor (α)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Response Time Ratio (RTR) Accuracy (%)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(65, 103)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=10)

    plt.tight_layout()

    # Save outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    out_path_src = os.path.join(script_dir, "alpha_sensitivity.png")
    out_path_root = os.path.join(project_root, "alpha_sensitivity.png")

    plt.savefig(out_path_src, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_root, dpi=300, bbox_inches='tight')
    plt.close()

    print("-" * 65)
    print(f"[SUCCESS] Plot saved to:")
    print(f"  1. {out_path_src}")
    print(f"  2. {out_path_root}")
    print("=" * 65)

if __name__ == "__main__":
    main()
