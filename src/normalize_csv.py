import os
import csv

def normalize_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "../database/simulation_history.csv")
    
    if not os.path.exists(csv_path):
        return

    normalized_rows = []
    fieldnames = ['timestamp', 'seed', 'scenario', 'model', 'trips', 'avg_resp', 'avg_ideal', 'accuracy', 'status', 'sim_steps', 'traffic_score']

    with open(csv_path, 'r', newline='') as f:
        lines = f.readlines()

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith('timestamp'):
            continue
        parts = line_str.split(',')
        if len(parts) == 10:
            # Legacy row without scenario column: insert 'ORIGINAL' at index 2
            ts, seed, model, trips, avg_resp, avg_ideal, accuracy, status, sim_steps, traffic_score = parts
            normalized_rows.append({
                'timestamp': ts,
                'seed': seed,
                'scenario': 'ORIGINAL',
                'model': model.strip().upper(),
                'trips': trips,
                'avg_resp': avg_resp,
                'avg_ideal': avg_ideal,
                'accuracy': accuracy,
                'status': status,
                'sim_steps': sim_steps,
                'traffic_score': traffic_score
            })
        elif len(parts) == 11:
            ts, seed, scenario, model, trips, avg_resp, avg_ideal, accuracy, status, sim_steps, traffic_score = parts
            normalized_rows.append({
                'timestamp': ts,
                'seed': seed,
                'scenario': scenario.strip().upper(),
                'model': model.strip().upper(),
                'trips': trips,
                'avg_resp': avg_resp,
                'avg_ideal': avg_ideal,
                'accuracy': accuracy,
                'status': status,
                'sim_steps': sim_steps,
                'traffic_score': traffic_score
            })

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)

    print(f"[SUCCESS] Normalized {len(normalized_rows)} CSV entries in {csv_path}")

if __name__ == "__main__":
    normalize_csv()
