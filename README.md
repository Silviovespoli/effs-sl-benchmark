# EFFS-SL Benchmark

Benchmark instances for the **Energy-Efficient Permutation Flow-Shop Scheduling Problem with Service Level Constraints** (EFFS-SL).

## Overview

This repository provides benchmark instances, baseline results, and a pre-trained deep reinforcement learning model for the energy-efficient flow-shop scheduling problem with discrete speed scaling and a hard service level (SL) constraint. The problem consists of finding a job permutation and a speed assignment for each job on each machine that minimises total energy consumption while ensuring that at least a prescribed fraction of jobs are completed by their due dates.

No existing flow-shop benchmark incorporates the combination of due dates, discrete speed levels, and service level information required by this problem class. Standard benchmarks such as the [Taillard (1993)](https://doi.org/10.1016/0377-2217(93)90182-M) instances lack both due dates and speed scaling. This dataset provides the first common testbed for algorithmic comparison in energy-efficient flow-shop scheduling with service level constraints.

## Citation

If you use these instances in your research, please cite:

```bibtex
@article{Vespoli2025effs,
  title   = {Energy-Efficient Permutation Flow-Shop Scheduling with Service
             Level Constraints: A Speed Scaling Approach},
  author  = {Vespoli, Silvestro and De Martino, Maria and Diano, Vincenzo
             and Guizzi, Guido},
  journal = {Computers \& Industrial Engineering},
  year    = {2025},
  note    = {Under review}
}
```

## Problem Description

We consider a permutation flow-shop with $n$ jobs, $m = 3$ machines, and discrete speed levels $\mathcal{V} = \{0.6, 0.8, 1.0\}$. The power consumption of a machine at speed $v$ follows:

$$P(v) = P_{\text{idle}} + P_{\text{proc}} \cdot v^{\alpha}$$

with $P_{\text{idle}} = 2$ kW, $P_{\text{proc}} = 8$ kW, and $\alpha = 3$ (cubic law). The energy per operation is:

$$E(v) = P(v) \cdot \frac{p^{\text{base}}}{v}$$

At the three speed levels, the normalised energy per operation is:

| Speed | Power (kW) | Processing time | Energy / $p^{\text{base}}$ | Savings vs $v=1.0$ |
|-------|-----------|----------------|---------------------------|---------------------|
| 1.0   | 10.000    | $p$            | 10.000                    | 0%                  |
| 0.8   |  6.096    | $1.25p$        |  7.620                    | 23.8%               |
| 0.6   |  3.728    | $1.67p$        |  6.213                    | 37.9%               |

The objective is to minimise $E_{\text{total}} = \sum_{j,i} E_{ij}(v_{ij})$ subject to $\text{SL} \geq \text{SL}^*$, where $\text{SL} = \frac{1}{n}\sum_j \mathbf{1}[C_j \leq d_j]$.

## Instance Format

### Large instances (`data/sim1/`, `data/sim2/`)

CSV files with one row per job:

| Column | Description |
|--------|-------------|
| `job_id` | Job identifier |
| `time_m1` | Base processing time on machine 1 (minutes) |
| `time_m2` | Base processing time on machine 2 (minutes) |
| `time_m3` | Base processing time on machine 3 (minutes) |
| `due_date` | Due date (minutes) |
| `completion_time_edd` | Completion time under EDD at $v=1.0$ |
| `delay_edd` | Delay under EDD at $v=1.0$ (0 = on-time) |

The `sim2/` instances additionally include `arrival_time` for dynamic scheduling scenarios.

### Small instances (`data/small/`)

CSV files with columns: `job_id`, `time_m1`, `time_m2`, `time_m3`, `due_date`.

### Loading an instance (Python)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data/sim1/sim1_1000jobs_70sl.csv")
processing_times = df[["time_m1", "time_m2", "time_m3"]].values
due_dates = df["due_date"].values
```

## Instance Characteristics

### Large instances (sim1) — static arrivals

| Instance | $n$ | $m$ | BL SL (%) | Mean $p$ | $\sigma_p$ | $E_{\text{base}}$ (kW$\cdot$min) | $C_{\max}^{\text{EDD}}$ |
|----------|-----|-----|-----------|----------|------------|-----------------------------------|------------------------|
| sim1_1000jobs_70sl | 1,000 | 3 | 69.5 | 12.56 | 4.30 | 376,800 | 12,765 |
| sim1_1000jobs_80sl | 1,000 | 3 | 79.5 | 12.56 | 4.30 | 376,800 | 12,765 |
| sim1_1000jobs_99sl | 1,000 | 3 | 99.3 | 12.56 | 4.30 | 376,800 | 12,765 |
| sim1_3000jobs_70sl | 3,000 | 3 | 70.5 | 12.49 | 4.33 | 1,124,275 | 37,904 |
| sim1_3000jobs_80sl | 3,000 | 3 | 80.0 | 12.49 | 4.33 | 1,124,275 | 37,904 |
| sim1_3000jobs_99sl | 3,000 | 3 | 99.9 | 12.49 | 4.33 | 1,124,275 | 37,904 |
| sim1_5000jobs_70sl | 5,000 | 3 | 71.0 | 12.50 | 4.33 | 1,875,670 | 62,930 |
| sim1_5000jobs_80sl | 5,000 | 3 | 79.7 | 12.50 | 4.33 | 1,875,670 | 62,930 |
| sim1_5000jobs_99sl | 5,000 | 3 | 99.1 | 12.50 | 4.33 | 1,875,670 | 62,930 |

### Large instances (sim2) — dynamic arrivals

| Instance | $n$ | $m$ | BL SL (%) | Mean $p$ | $\sigma_p$ | $E_{\text{base}}$ (kW$\cdot$min) | $C_{\max}^{\text{EDD}}$ |
|----------|-----|-----|-----------|----------|------------|-----------------------------------|------------------------|
| sim2_1000jobs_70sl | 1,000 | 3 | 69.8 | 12.61 | 4.36 | 378,310 | 89,703 |
| sim2_1000jobs_80sl | 1,000 | 3 | 78.6 | 12.61 | 4.36 | 378,310 | 86,871 |
| sim2_1000jobs_99sl | 1,000 | 3 | 97.8 | 12.61 | 4.36 | 378,310 | 39,694 |
| sim2_3000jobs_70sl | 3,000 | 3 | 69.5 | 12.57 | 4.35 | 1,131,583 | 244,999 |
| sim2_3000jobs_80sl | 3,000 | 3 | 80.0 | 12.57 | 4.35 | 1,131,583 | 187,307 |
| sim2_3000jobs_99sl | 2,993 | 3 | 97.5 | 12.57 | 4.35 | 1,128,699 | 118,756 |
| sim2_5000jobs_70sl | 5,000 | 3 | 68.2 | 12.52 | 4.35 | 1,878,279 | 356,206 |
| sim2_5000jobs_80sl | 5,000 | 3 | 78.9 | 12.52 | 4.35 | 1,878,279 | 331,591 |
| sim2_5000jobs_99sl | 4,993 | 3 | 97.8 | 12.52 | 4.35 | 1,875,620 | 213,904 |

### Small instances (for MILP validation)

| Instance | $n$ | $m$ | BL SL (%) | $E_{\text{base}}$ (kW$\cdot$min) |
|----------|-----|-----|-----------|-----------------------------------|
| small_10jobs_k0 | 10 | 3 | 100.0 | 3,902 |
| small_10jobs_k1 | 10 | 3 | 100.0 | 3,426 |
| small_10jobs_k2 | 10 | 3 | 100.0 | 3,628 |
| small_15jobs_k0 | 15 | 3 | 100.0 | 6,370 |
| small_15jobs_k1 | 15 | 3 | 93.3 | 5,477 |
| small_15jobs_k2 | 15 | 3 | 100.0 | 5,040 |
| small_20jobs_k0 | 20 | 3 | 100.0 | 7,453 |
| small_20jobs_k1 | 20 | 3 | 100.0 | 6,707 |
| small_20jobs_k2 | 20 | 3 | 100.0 | 7,201 |

## Best Known Results

The file `results/best_known_results.csv` contains the best results obtained by each method on each instance and SL target. The five methods compared are:

1. **Full MILP** — joint sequence + speed optimisation (exact, $n \leq 20$)
2. **Speed-Only MILP** — speed optimisation for a fixed EDD sequence (exact)
3. **Iterated Greedy (IG)** — metaheuristic with slack-based speed adjustment
4. **Matheuristic** — IG sequence search + exact MILP speed assignment
5. **DRL** — deep reinforcement learning agent for speed assignment

### Summary of best savings (%) on large instances — IG

| Instance | SL*=50% | SL*=60% | SL*=70% |
|----------|---------|---------|---------|
| sim1_1000jobs_70sl | 28.4 | 26.6 | 25.5 |
| sim1_1000jobs_80sl | 32.5 | 27.2 | 27.1 |
| sim1_1000jobs_99sl | 36.5 | 28.4 | 31.2 |
| sim1_3000jobs_70sl | 24.1 | 23.6 | 23.5 |
| sim1_3000jobs_80sl | 25.2 | 24.7 | 24.4 |
| sim1_3000jobs_99sl | 27.5 | 26.4 | 26.2 |
| sim1_5000jobs_70sl | 24.5 | 23.9 | 23.8 |
| sim1_5000jobs_80sl | 26.1 | 25.2 | 25.2 |
| sim1_5000jobs_99sl | 30.2 | 28.5 | 28.4 |

### Summary of best savings (%) on large instances — Matheuristic

| Instance | SL*=50% | SL*=60% | SL*=70% |
|----------|---------|---------|---------|
| sim1_1000jobs_70sl | 28.2 | 26.1 | 25.0 |
| sim1_1000jobs_80sl | 31.0 | 27.2 | 27.1 |
| sim1_1000jobs_99sl | 34.4 | 28.3 | 29.3 |
| sim1_3000jobs_70sl | 24.0 | 23.3 | 23.3 |
| sim1_3000jobs_80sl | 25.0 | 24.6 | 24.3 |
| sim1_3000jobs_99sl | 27.2 | 26.4 | 26.2 |
| sim1_5000jobs_70sl | 24.1 | 23.8 | 23.8 |
| sim1_5000jobs_80sl | 25.9 | 25.1 | 25.1 |
| sim1_5000jobs_99sl | 30.2 | 28.4 | 28.4 |

**Note on DRL:** The DRL agent (trained on instances with $n \in [5, 200]$) satisfies the SL constraint on all small instances (Table 1) and on all large instances at SL\*=50%. At SL\*=70%, it violates the constraint on 7 out of 9 large instances. DRL results are included in `best_known_results.csv` with the achieved SL reported alongside savings. See the paper for a detailed discussion.

### Summary of DRL savings (%) on large instances

| Instance | SL\*=50% (savings / SL) | SL\*=60% (savings / SL) | SL\*=70% (savings / SL) |
|----------|------------------------|------------------------|------------------------|
| sim1_1000jobs_70sl | 22.4 / 55.6 | 22.3 / 55.6 | 22.3 / 57.0 |
| sim1_1000jobs_80sl | 23.9 / 60.7 | 23.9 / 63.7 | 24.1 / 67.2 |
| sim1_1000jobs_99sl | 26.2 / 68.8 | 26.5 / 70.6 | 26.6 / 75.5 |
| sim1_3000jobs_70sl | 21.8 / 55.6 | 22.3 / 57.9 | 22.8 / 61.3 |
| sim1_3000jobs_80sl | 23.1 / 59.3 | 23.3 / 61.3 | 23.6 / 63.4 |
| sim1_3000jobs_99sl | 24.2 / 54.9 | 24.5 / 58.1 | 24.8 / 62.5 |
| sim1_5000jobs_70sl | 23.0 / 57.6 | 23.1 / 59.6 | 23.4 / 62.6 |
| sim1_5000jobs_80sl | 23.8 / 56.7 | 24.0 / 60.8 | 24.3 / 64.5 |
| sim1_5000jobs_99sl | 26.7 / 66.0 | 27.0 / 69.1 | 27.2 / 71.3 |

### Summary of best savings (%) on small instances

| $n$ | SL* | Full MILP | Speed MILP | IG | Matheuristic | DRL |
|-----|-----|-----------|-----------|-----|-------------|-----|
| 10 | 70% | 37.9 | 37.9 | 37.9 | 37.9 | 31.2 |
| 10 | 80% | 37.9 | 37.9 | 37.9 | 36.8 | 31.3 |
| 10 | 90% | 37.9 | 37.9 | 34.8 | 34.4 | 31.3 |
| 15 | 70% | 37.9 | 36.9 | 37.9 | 37.9 | 26.1 |
| 15 | 80% | 37.9 | 34.6 | 35.7 | 35.4 | 26.1 |
| 15 | 90% | --- | 33.3 | 33.6 | 34.4 | 26.1 |
| 20 | 70% | 37.9 | 37.9 | 37.9 | 37.9 | 28.7 |
| 20 | 80% | 37.5 | 37.5 | 37.9 | 37.9 | 28.7 |
| 20 | 90% | --- | 36.0 | 37.9 | 36.8 | 28.7 |

## DRL Model

The directory `models/` contains:
- `ppo_speed_universal.zip` — a pre-trained PPO agent for per-operation speed assignment
- `vecnormalize.pkl` — reward normalisation statistics (required for correct inference)

The model was trained using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) on random instances with $n \in [5, 200]$, SL targets in $\{50\%, 55\%, \ldots, 95\%\}$, for 10 million timesteps with `ent_coef=0.005`, `gamma=0.999`, and `VecNormalize` (reward-only).

To use the model:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

model = PPO.load("models/ppo_speed_universal")
# See the paper for the 17-dimensional observation space definition
obs = ...  # construct observation vector
action, _ = model.predict(obs, deterministic=True)
speed = [0.6, 0.8, 1.0][action]
```

The observation space is a 17-dimensional vector; see Section 3.4 of the paper for the full feature definition.

## Reproducing Instances

### Small instances

The small instances can be reproduced deterministically:

```bash
python scripts/generate_small_instances.py --output data/small
```

Parameters: `Uniform(5, 20)` processing times, `Exponential(300 + 10n)` due-date slack, seeds `100 + k*1000 + n_jobs` for $k \in \{0, 1, 2\}$.

### Large instances — simulation logic

The large instances were generated by a discrete-event simulation (implemented in SimPy) of a three-machine permutation flow shop. Two simulation variants were used.

**sim1 (static arrivals).** All $n$ jobs are available at time zero. Each job's processing time on each machine is drawn independently from $\text{Uniform}(5, 20)$ minutes. Due dates are set as $d_j = \sum_{i=1}^{m} p_{ij} + \delta_j$, where $\delta_j \sim \text{Exponential}(\Delta)$ and $\Delta$ is a slack parameter calibrated via binary search so that the Earliest Due Date (EDD) rule, executed at maximum speed ($v = 1.0$), yields a service level close to the target (70%, 80%, or 99%). Each machine is modelled as a single-capacity resource; jobs are pre-sorted by due date and processed in that order without preemption.

**sim2 (dynamic arrivals).** Jobs arrive over time with inter-arrival intervals drawn from $\text{Exponential}(\mu)$, where $\mu$ is a second parameter calibrated jointly with $\Delta$ to achieve the target service level. Due dates are computed relative to each job's arrival time: $d_j = a_j + \sum_{i} p_{ij} + \delta_j$, where $a_j$ is the arrival time. A centralised EDD dispatcher assigns arriving jobs to machines as they become available, re-sorting the queue on each arrival. The dynamic arrivals and queueing effects make these instances substantially harder than their static counterparts and more representative of real manufacturing environments.

For both variants, the processing times within each instance group (e.g., all 1,000-job sim1 instances) are drawn from the same distribution with the same seed, and only the due-date slack parameter $\Delta$ varies to achieve different baseline service levels. The CSV files in this repository are the authoritative source; they cannot be regenerated from a simple script because the calibration procedure is iterative.

To convert the original Excel files to CSV:

```bash
python scripts/convert_excel_to_csv.py --source /path/to/Simulazioni
```

## Repository Structure

```
effs-sl-benchmark/
├── README.md                        # This file
├── LICENSE                          # MIT License
├── requirements.txt                 # numpy, pandas
├── energy_model.py                  # Standalone energy model implementation
├── instance_characteristics.csv     # Summary statistics for all instances
├── data/
│   ├── sim1/                        # 9 large static instances (CSV)
│   ├── sim2/                        # 9 large dynamic instances (CSV)
│   └── small/                       # 9 small validation instances (CSV)
├── results/
│   └── best_known_results.csv       # Best results from all five methods
├── scripts/
│   ├── convert_excel_to_csv.py      # Excel-to-CSV conversion
│   └── generate_small_instances.py  # Deterministic small instance generator
└── models/
    ├── ppo_speed_universal.zip      # Pre-trained DRL model (~1.7 MB)
    └── vecnormalize.pkl             # Reward normalisation statistics
```

## License

This benchmark is released under the [MIT License](LICENSE).
