#!/usr/bin/env python3
"""Convert thesis Excel benchmark files to portable CSV format.

Reads the 18 Excel files (9 sim1 + 9 sim2) from a source directory
and writes clean CSV files to data/sim1/ and data/sim2/.

Usage
-----
    python scripts/convert_excel_to_csv.py --source /path/to/Simulazioni

If --source is not provided, it looks for a ``Simulazioni/`` directory
in the current working directory.
"""

import argparse
from pathlib import Path

import pandas as pd

# --- File catalogue ---
SIM1_FILES = {
    (1000, 70): "SIMULAZIONE1_1000JOB_70%.xlsx",
    (1000, 80): "SIMULAZIONE1_1000JOB_80%.xlsx",
    (1000, 99): "SIMULAZIONE1_1000JOB_99%.xlsx",
    (3000, 70): "SIMULAZIONE1_3000JOB_70%.xlsx",
    (3000, 80): "SIMULAZIONE1_3000JOB_80%.xlsx",
    (3000, 99): "SIMULAZIONE1_3000JOB_99%.xlsx",
    (5000, 70): "SIMULAZIONE1_5000JOB_70%.xlsx",
    (5000, 80): "SIMULAZIONE1_5000JOB_80%.xlsx",
    (5000, 99): "SIMULAZIONE1_5000JOB_99%.xlsx",
}

SIM2_FILES = {
    (1000, 70): "1000job-70%.xlsx",
    (1000, 80): "1000job-80%.xlsx",
    (1000, 99): "1000job-99%.xlsx",
    (3000, 70): "3000job_70%.xlsx",
    (3000, 80): "3000job-80%.xlsx",
    (3000, 99): "3000job-99%.xlsx",
    (5000, 70): "5000job-70%.xlsx",
    (5000, 80): "5000job-80%.xlsx",
    (5000, 99): "5000job_99%.xlsx",
}


def convert_sim1(source_dir: Path, output_dir: Path) -> None:
    """Convert sim1 Excel files to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for (n_jobs, sl), filename in sorted(SIM1_FILES.items()):
        filepath = source_dir / "sim1" / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        df = pd.read_excel(filepath, sheet_name="Job Details")

        out_df = pd.DataFrame({
            "job_id": df["Job ID"],
            "time_m1": df["Time M1"],
            "time_m2": df["Time M2"],
            "time_m3": df["Time M3"],
            "due_date": df["Due Date"],
            "completion_time_edd": df["Completion Time"],
            "delay_edd": df["Delay"],
        })

        out_name = f"sim1_{n_jobs}jobs_{sl}sl.csv"
        out_df.to_csv(output_dir / out_name, index=False)
        print(f"  sim1: {out_name} ({len(out_df)} jobs)")


def convert_sim2(source_dir: Path, output_dir: Path) -> None:
    """Convert sim2 Excel files to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for (n_jobs, sl), filename in sorted(SIM2_FILES.items()):
        filepath = source_dir / "sim2" / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found, skipping")
            continue

        df = pd.read_excel(filepath, sheet_name="Job Details")

        out_df = pd.DataFrame({
            "job_id": df["Job ID"],
            "arrival_time": df["Arrival Time"],
            "time_m1": df["Time M1"],
            "time_m2": df["Time M2"],
            "time_m3": df["Time M3"],
            "due_date": df["Due Date"],
            "completion_time_edd": df["End Time"],
            "delay_edd": df["Delay"],
        })

        out_name = f"sim2_{n_jobs}jobs_{sl}sl.csv"
        out_df.to_csv(output_dir / out_name, index=False)
        print(f"  sim2: {out_name} ({len(out_df)} jobs)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert thesis Excel benchmark files to CSV."
    )
    parser.add_argument(
        "--source", type=str, default="Simulazioni",
        help="Path to directory containing sim1/ and sim2/ Excel files",
    )
    parser.add_argument(
        "--output", type=str, default="data",
        help="Output base directory (default: data/)",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_base = Path(args.output)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print(f"Source: {source_dir.resolve()}")
    print(f"Output: {output_base.resolve()}")
    print()

    print("Converting sim1 files...")
    convert_sim1(source_dir, output_base / "sim1")
    print()

    print("Converting sim2 files...")
    convert_sim2(source_dir, output_base / "sim2")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
