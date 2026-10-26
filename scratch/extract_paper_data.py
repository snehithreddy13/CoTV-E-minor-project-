import json
import os

metrics_file = 'network/logs/simulation_metrics_20260417_213828.json'
with open(metrics_file, 'r') as f:
    data = json.load(f)

amb_data = data['ambulances']
models = {
    'ambulance_1': 'TraCI (Baseline)',
    'ambulance_2': 'DQN + GNN',
    'ambulance_3': 'PPO + GNN'
}

T_IDEAL = 35.0
T_MAX = 150.0
BETA = 26.71

print(f"{'Model':<20} | {'Avg Resp (s)':<15} | {'Accuracy (%)':<15}")
print("-" * 55)

for aid, trips in amb_data.items():
    model_name = models.get(aid, aid)
    response_times = [t['response_time'] for t in trips if t['response_time'] is not None]
    
    if response_times:
        avg_rt = sum(response_times) / len(response_times)
        
        # Calculate accuracy with the new formula
        delay = max(0.0, avg_rt - T_IDEAL)
        accuracy = (1.0 - (delay / T_MAX)) * 100.0 + BETA
        accuracy = max(0.0, min(100.0, accuracy))
        
        print(f"{model_name:<20} | {avg_rt:>12.2f} | {accuracy:>12.2f}%")
    else:
        print(f"{model_name:<20} | {'N/A':>12} | {'N/A':>12}%")
