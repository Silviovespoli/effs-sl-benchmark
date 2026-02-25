#!/usr/bin/env python3
"""Generate small benchmark instances for MILP validation experiments.

Produces 9 instances: 3 sizes (n=10, 15, 20) x 3 replications (k=0, 1, 2).

The generation procedure mirrors the paper exactly:
- Processing times: Uniform(5, 20) per machine per job
- Due dates: sum(processing_times) + Exponential(media_delta)
- media_delta = 300 + 10*n (hardcoded, calibrated so that EDD at v=1.0
  yields varying baseline service levels across instances)
- Random seed: SEED + k*1000 + n_jobs, where SEED=100

These instances are used in Table 1 of the paper for the 5-method
comparison (Full MILP, Speed-Only MILP, IG, Matheuristic, DRL).
Each instance is tested at multiple SL targets (70%, 80%, 90%).

Usage
-----
    python scripts/generate_small_instances.py [--output data/small]
"""

import argparse
import random
from pathlib import Path

import numpy as np

# --- Constants (matching the paper) ---
SEED = 100
NUM_MACHINES = 3
PROC_TIME_MIN = 5.0
PROC_TIME_MAX = 20.0
INSTANCE_SIZES = [10, 15, 20]
N_REPLICATIONS = 3


def generate_instance(n_jobs, n_machines, media_delta, seed):
    """Generate a single flow-shop instance.

    Parameters
    ----------
    n_jobs : int
        Number of jobs.
    n_machines : int
        Number of machines.
    media_delta : float
        Mean of exponential distribution for due-date slack.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    processing_times : np.ndarray, shape (n_jobs, n_machines)
    due_dates : np.ndarray, shape (n_jobs,)
    """
    rng = random.Random(seed)

    processing_times = np.zeros((n_jobs, n_machines))
    due_dates = np.zeros(n_jobs)

    for j in range(n_jobs):
        times = [rng.uniform(PROC_TIME_MIN, PROC_TIME_MAX)
                 for _ in range(n_machines)]
        processing_times[j, :] = times
        total_time = sum(times)
        delta = rng.expovariate(1.0 / media_delta)
        due_dates[j] = total_time + delta

    return processing_times, due_dates


def main():
    parser = argparse.ArgumentParser(
        description="Generate small benchmark instances for MILP validation."
    )
    parser.add_argument(
        "--output", type=str, default="data/small",
        help="Output directory for CSV files (default: data/small/)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir.resolve()}")
    print(f"Generating {len(INSTANCE_SIZES)} sizes x "
          f"{N_REPLICATIONS} replications = "
          f"{len(INSTANCE_SIZES) * N_REPLICATIONS} instances")
    print()

    count = 0
    for n_jobs in INSTANCE_SIZES:
        media_delta = 300 + n_jobs * 10
        for k in range(N_REPLICATIONS):
            seed = SEED + k * 1000 + n_jobs
            processing_times, due_dates = generate_instance(
                n_jobs, NUM_MACHINES, media_delta, seed
            )

            filename = f"small_{n_jobs}jobs_k{k}.csv"
            filepath = output_dir / filename

            with open(filepath, "w") as f:
                f.write("job_id,time_m1,time_m2,time_m3,due_date\n")
                for j in range(n_jobs):
                    f.write(f"{j},"
                            f"{processing_times[j, 0]:.6f},"
                            f"{processing_times[j, 1]:.6f},"
                            f"{processing_times[j, 2]:.6f},"
                            f"{due_dates[j]:.6f}\n")

            count += 1
            print(f"  {filename} (n={n_jobs}, k={k}, "
                  f"seed={seed}, delta={media_delta})")

    print(f"\nGenerated {count} instances.")


if __name__ == "__main__":
    main()
