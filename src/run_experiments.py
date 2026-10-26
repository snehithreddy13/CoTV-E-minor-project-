import os
import sys
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Automated CoTV-E Multi-Scenario Experiment Runner")
    parser.add_argument("--sim-steps", type=int, default=1800, help="Simulation steps per run")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 101, 202, 303, 404], help="Random seeds for evaluation")
    parser.add_argument("--scenarios", nargs="*", default=["original", "expanded"], help="Scenarios to evaluate")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    routing_script = os.path.join(script_dir, "ambulance_routing.py")

    total_runs = len(args.seeds)
    print("=" * 70)
    print("AUTOMATED CoTV-E UNIFIED MULTI-HOSPITAL EXPERIMENT SUITE")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")
    print(f"Sim Steps: {args.sim_steps}")
    print(f"Total Runs: {total_runs}")
    print("=" * 70 + "\n")

    run_count = 0
    start_time = time.time()

    for seed in args.seeds:
        run_count += 1
        print(f"[{run_count}/{total_runs}] Running Multi-Hospital Experiment | Seed: {seed} | Steps: {args.sim_steps}")
        cmd = [
            sys.executable, routing_script,
            "--nogui",
            "--sim-steps", str(args.sim_steps),
            "--seed", str(seed)
        ]
        res = subprocess.run(cmd, capture_output=False)
        if res.returncode != 0:
            print(f"[WARNING] Run {run_count} failed with exit code {res.returncode}")
        time.sleep(1.0)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[SUCCESS] Completed all {total_runs} experiment runs in {elapsed:.2f}s!")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
