"""Standalone energy model for variable-speed flow-shop machines.

Power model: P(v) = P_idle + P_proc * v^alpha
- At v=1.0: P = 2 + 8*1.0  = 10.000 kW
- At v=0.8: P = 2 + 8*0.512 =  6.096 kW
- At v=0.6: P = 2 + 8*0.216 =  3.728 kW

Processing time scales inversely with speed: p(v) = p_base / v

Energy per operation: E(v) = P(v) * p(v)
- At v=1.0: E = 10.000 * p     = 10.000 p  (baseline)
- At v=0.8: E =  6.096 * p/0.8 =  7.620 p  (savings 23.8%)
- At v=0.6: E =  3.728 * p/0.6 =  6.213 p  (savings 37.9%)

Reference
---------
Vespoli et al. (2025). Energy-Efficient Permutation Flow-Shop Scheduling
with Service Level Constraints: A Speed Scaling Approach.
Computers & Industrial Engineering.
"""

import numpy as np

# --- Energy model parameters ---
P_IDLE = 2.0          # kW, idle power draw (constant)
P_PROC = 8.0          # kW, speed-dependent processing power coefficient
ALPHA = 3             # speed-power exponent (cubic law)
SPEED_LEVELS = [0.6, 0.8, 1.0]


def power(v: float) -> float:
    """Instantaneous power at speed v [kW].

    P(v) = P_idle + P_proc * v^alpha
    """
    return P_IDLE + P_PROC * (v ** ALPHA)


def processing_time(p_base: float, v: float) -> float:
    """Actual processing time at speed v [minutes].

    p(v) = p_base / v
    """
    return p_base / v


def energy_operation(p_base: float, v: float) -> float:
    """Energy consumed by one operation [kW*min].

    E(v) = P(v) * p_base / v
    """
    return power(v) * processing_time(p_base, v)


def total_energy(processing_times: np.ndarray, speed_matrix: np.ndarray) -> float:
    """Total energy for a complete schedule [kW*min].

    Parameters
    ----------
    processing_times : np.ndarray, shape (n_jobs, n_machines)
        Base processing times at maximum speed (v=1.0).
    speed_matrix : np.ndarray, shape (n_jobs, n_machines)
        Speed assignment for each job on each machine.

    Returns
    -------
    float
        Total energy in kW*min.
    """
    P = P_IDLE + P_PROC * np.power(speed_matrix, ALPHA)
    t = processing_times / speed_matrix
    return float(np.sum(P * t))


def energy_at_max_speed(processing_times: np.ndarray) -> float:
    """Baseline energy: all operations at v=1.0 [kW*min]."""
    return float(np.sum(power(1.0) * processing_times))


def energy_savings_percent(energy_optimized: float, energy_baseline: float) -> float:
    """Percentage energy savings relative to baseline."""
    if energy_baseline == 0:
        return 0.0
    return (1.0 - energy_optimized / energy_baseline) * 100.0


# Precomputed energy factors for each speed level (energy per unit base processing time)
ENERGY_FACTORS = {v: energy_operation(1.0, v) for v in SPEED_LEVELS}
# {0.6: 6.213, 0.8: 7.620, 1.0: 10.0}


if __name__ == "__main__":
    print("Energy model parameters:")
    print(f"  P_idle = {P_IDLE} kW")
    print(f"  P_proc = {P_PROC} kW")
    print(f"  alpha  = {ALPHA}")
    print()
    print("Per-operation energy (normalised by p_base):")
    for v in SPEED_LEVELS:
        e = energy_operation(1.0, v)
        sav = energy_savings_percent(e, energy_operation(1.0, 1.0))
        print(f"  v={v:.1f}: P={power(v):.3f} kW, "
              f"p/p_base={1/v:.3f}, "
              f"E/p_base={e:.3f}, "
              f"savings={sav:.1f}%")
