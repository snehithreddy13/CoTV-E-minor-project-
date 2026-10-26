# CoTV-E Multi-Model Ambulance Dispatch & Latency Analysis System

## Overview
Simulates emergency ambulance routing with three algorithms side-by-side in SUMO:
- **TraCI** — Dijkstra shortest-path baseline
- **DQN+GNN** — Deep Q-Network with Graph Neural Network routing
- **PPO+GNN** — Proximal Policy Optimization with GNN (SOTA)

Includes complete **Network Scalability Analysis** (Original 6x6 Grid vs Expanded 8x8 Grid with Dual Hospitals) and **Computational Cost & Inference Latency Profiling**.

---

## Usage Commands (Windows Command Prompt)

### 1. Model Profiling & Inference Latency Benchmarks
```cmd
:: Profile model parameter counts, FLOPs, and theoretical GCN complexity
python src/profile_complexity.py

:: Run unit tests verifying eval() mode and torch.no_grad() inference validity
python src/test_inference_validity.py

:: Run high-precision inference latency benchmark suite (50 warm-up, 500 iterations)
python src/benchmark_inference.py --warmup 50 --iterations 500

:: Generate latency breakdown and network scaling charts
python src/plot_latency_analysis.py
```

### 2. Multi-Scenario Simulation & Experiment Suite
```cmd
:: Run simulation on Original 6x6 Network
python src/ambulance_routing.py --scenario original --nogui --sim-steps 1800 --seed 42

:: Run simulation on Expanded 8x8 Dual-Hospital Network
python src/ambulance_routing.py --scenario expanded --nogui --sim-steps 1800 --seed 42

:: Run automated multi-seed experiment suite across all scenarios
python src/run_experiments.py --sim-steps 1800 --seeds 42 101 202 303 404
```

### 3. Statistical Analysis & Plot Generation
```cmd
:: Normalize simulation history database
python src/normalize_csv.py

:: Perform descriptive statistical analysis & paired Wilcoxon Signed-Rank tests
python src/statistical_validation.py

:: Generate comparative accuracy & response time plots
python src/plot_network_comparison.py
```

---

## Output Metrics & Report Files

- **`database/latency_benchmarks.json`**: JSON log containing system metadata, PyTorch versions, node/edge counts, and sub-component timing samples (preprocessing, GNN forward pass, action selection head).
- **`database/latency_benchmarks.csv`**: Summary CSV table of mean, median, P95, and standard deviation latency metrics.
- **`database/simulation_history.csv`**: Archived simulation run leaderboard results across network scenarios.
- **`src/latency_breakdown_chart.png`**: Visual chart of sub-component latency breakdown.
- **`src/network_scaling_latency_chart.png`**: Visual chart comparing RL decision latency scaling across network sizes.
- **`src/network_comparison_analysis.png`**: Visual chart comparing RTR accuracy and response time between Original and Expanded networks.
- **`PAPER_UPDATES.md`**: Complete paper revision report for Sections III, IV, V, and VI.

---

## Architectural Parameters & Inference Latency Summary

| Model | Parameters | Theoretical GCN Complexity | Original (6x6) Mean Latency | Expanded (8x8) Mean Latency | P95 Latency | Real-Time Feasibility |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **DQN+GNN** | 2,552 | $\mathcal{O}(\text{GNN} + (D_s+D_e)H + HA)$ | 0.3521 ms | 0.3819 ms | 0.5386 ms | Optimal ($<0.04\%$ of step) |
| **PPO+GNN** | **4,025** | $\mathcal{O}(\text{GNN} + 2(D_s+D_e)H + HA + H \cdot 1)$ | **0.2328 ms** | **0.2422 ms** | **0.3432 ms** | **Optimal ($<0.025\%$ of step)** |
