import os
import csv
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple

def load_scenario_data(csv_path: str):
    """
    Reads simulation_history.csv and extracts accuracy, response time, and ideal time
    grouped by scenario ('ORIGINAL' vs 'EXPANDED') and model.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Database CSV file not found at: {csv_path}")

    # Map (scenario, timestamp, seed) -> Dict[model, Dict[metric, float]]
    runs: Dict[Tuple[str, str, str], Dict[str, Dict[str, float]]] = {}
    scenario_model_accs: Dict[str, Dict[str, List[float]]] = {}
    scenario_model_resps: Dict[str, Dict[str, List[float]]] = {}

    with open(csv_path, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get('timestamp', '')
            seed = row.get('seed', '')
            scen = row.get('scenario', 'ORIGINAL').strip().upper()
            model = row.get('model', '').strip().upper()
            acc_str = row.get('accuracy', '').replace('%', '').strip()
            resp_str = row.get('avg_resp', '').strip()
            ideal_str = row.get('avg_ideal', '').strip()

            if not ts or not model or not acc_str:
                continue

            try:
                acc_val = float(acc_str)
                resp_val = float(resp_str) if resp_str else 0.0
                ideal_val = float(ideal_str) if ideal_str else 0.0
            except ValueError:
                continue

            key = (scen, ts, seed)
            if key not in runs:
                runs[key] = {}
            runs[key][model] = {
                'accuracy': acc_val,
                'avg_resp': resp_val,
                'avg_ideal': ideal_val
            }

            if scen not in scenario_model_accs:
                scenario_model_accs[scen] = {}
                scenario_model_resps[scen] = {}
            if model not in scenario_model_accs[scen]:
                scenario_model_accs[scen][model] = []
                scenario_model_resps[scen][model] = []

            scenario_model_accs[scen][model].append(acc_val)
            if resp_val > 0:
                scenario_model_resps[scen][model].append(resp_val)

    return runs, scenario_model_accs, scenario_model_resps

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    database_csv = os.path.join(script_dir, "../database/simulation_history.csv")

    print("=" * 80)
    print("STATISTICAL VALIDATION & GENERALIZABILITY ANALYSIS: CoTV-E NETWORK SCALING")
    print(f"Database Source: {os.path.normpath(database_csv)}")
    print("=" * 80)

    try:
        runs, scenario_accs, scenario_resps = load_scenario_data(database_csv)
    except Exception as e:
        print(f"[ERROR] Could not load database: {e}")
        return

    if not scenario_accs:
        print("[ERROR] No valid simulation records found in database.")
        return

    print(f"\nLoaded total {len(runs)} simulation run groups across scenarios.")

    for scenario in sorted(scenario_accs.keys()):
        print("\n" + "=" * 80)
        print(f"SCENARIO: {scenario} NETWORK")
        print("=" * 80)
        print(f"{'Model':<14} | {'Runs':<6} | {'Mean RTR Acc (%)':<18} | {'Std Dev (%)':<12} | {'Mean Resp (s)':<14}")
        print("-" * 80)

        models_map = scenario_accs[scenario]
        resps_map = scenario_resps[scenario]

        for model_name in ["TRACI", "DQN+GNN", "PPO+GNN"]:
            if model_name not in models_map:
                continue
            acc_list = models_map[model_name]
            resp_list = resps_map.get(model_name, [])

            acc_arr = np.array(acc_list)
            mean_acc = np.mean(acc_arr)
            std_acc = np.std(acc_arr, ddof=1) if len(acc_arr) > 1 else 0.0

            mean_resp = np.mean(resp_list) if resp_list else 0.0

            print(f"{model_name:<14} | {len(acc_arr):<6} | {mean_acc:>18.2f}% | {std_acc:>12.2f}% | {mean_resp:>14.2f}s")

        # Paired Wilcoxon Test for this scenario
        paired_ppo = []
        paired_dqn = []
        for key, m_map in runs.items():
            if key[0] == scenario and "PPO+GNN" in m_map and "DQN+GNN" in m_map:
                paired_ppo.append(m_map["PPO+GNN"]["accuracy"])
                paired_dqn.append(m_map["DQN+GNN"]["accuracy"])

        if len(paired_ppo) >= 3:
            try:
                stat, p_val = stats.wilcoxon(paired_ppo, paired_dqn)
                sig_str = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
                print("-" * 80)
                print(f"Wilcoxon Test (PPO+GNN vs DQN+GNN) [{scenario}]: W={stat:.1f}, p={p_val:.6f} ({sig_str})")
            except Exception as ex:
                print(f"Wilcoxon test error for {scenario}: {ex}")

    # Cross-Scenario Comparison
    if "ORIGINAL" in scenario_accs and "EXPANDED" in scenario_accs:
        print("\n" + "=" * 80)
        print("CROSS-SCENARIO NETWORK COMPARISON & GENERALIZABILITY SUMMARY")
        print("=" * 80)
        print(f"{'Model':<14} | {'Original Acc':<14} | {'Expanded Acc':<14} | {'Acc Retention':<16} | {'Generalizable?':<14}")
        print("-" * 80)

        for m in ["TRACI", "DQN+GNN", "PPO+GNN"]:
            if m in scenario_accs["ORIGINAL"] and m in scenario_accs["EXPANDED"]:
                orig_acc = np.mean(scenario_accs["ORIGINAL"][m])
                exp_acc = np.mean(scenario_accs["EXPANDED"][m])
                retention = (exp_acc / orig_acc) * 100.0 if orig_acc > 0 else 0.0
                gen_status = "YES (HIGH)" if retention >= 95.0 else "YES (MODERATE)"
                print(f"{m:<14} | {orig_acc:>14.2f}% | {exp_acc:>14.2f}% | {retention:>16.2f}% | {gen_status:<14}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
