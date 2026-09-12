from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("results/tables/ml_dataset.csv")
OUT_DIR = Path("results/tables")
BY_DEVICE_OUT = OUT_DIR / "sensitivity_by_device.csv"
SUMMARY_OUT = OUT_DIR / "sensitivity_summary.csv"
LEVELS_OUT = OUT_DIR / "sensitivity_factor_levels.csv"

FACTORS = [
    "crossbar_size",
    "requested_weight_bits",
    "adc_bits",
]

TARGETS = [
    "accuracy",
    "relative_hardware_cost_proxy",
]


def safe_range(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.max() - s.min())


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run src/build_ml_dataset.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    required = {"device_id", *FACTORS, *TARGETS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []
    level_rows = []

    for device_id, g in df.groupby("device_id", sort=True):
        for factor in FACTORS:
            grouped = (
                g.groupby(factor, dropna=False)[TARGETS]
                .mean(numeric_only=True)
                .reset_index()
                .sort_values(factor)
            )

            for _, r in grouped.iterrows():
                level_rows.append(
                    {
                        "device_id": device_id,
                        "factor": factor,
                        "factor_level": r[factor],
                        "mean_accuracy": r["accuracy"],
                        "mean_relative_hardware_cost_proxy": r["relative_hardware_cost_proxy"],
                    }
                )

            acc_range = safe_range(grouped["accuracy"])
            cost_range = safe_range(grouped["relative_hardware_cost_proxy"])

            best_acc_row = grouped.loc[grouped["accuracy"].idxmax()]
            worst_acc_row = grouped.loc[grouped["accuracy"].idxmin()]

            rows.append(
                {
                    "device_id": device_id,
                    "factor": factor,
                    "accuracy_effect_range_pp": acc_range,
                    "cost_effect_range": cost_range,
                    "best_accuracy_level": best_acc_row[factor],
                    "best_level_mean_accuracy": best_acc_row["accuracy"],
                    "worst_accuracy_level": worst_acc_row[factor],
                    "worst_level_mean_accuracy": worst_acc_row["accuracy"],
                }
            )

    by_device = pd.DataFrame(rows)
    levels = pd.DataFrame(level_rows)

    summary = (
        by_device.groupby("factor", as_index=False)
        .agg(
            median_accuracy_effect_range_pp=("accuracy_effect_range_pp", "median"),
            mean_accuracy_effect_range_pp=("accuracy_effect_range_pp", "mean"),
            max_accuracy_effect_range_pp=("accuracy_effect_range_pp", "max"),
            median_cost_effect_range=("cost_effect_range", "median"),
            mean_cost_effect_range=("cost_effect_range", "mean"),
        )
        .sort_values("median_accuracy_effect_range_pp", ascending=False)
        .reset_index(drop=True)
    )

    by_device["accuracy_sensitivity_rank_within_device"] = (
        by_device.groupby("device_id")["accuracy_effect_range_pp"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    by_device = by_device.sort_values(
        ["device_id", "accuracy_sensitivity_rank_within_device", "factor"]
    ).reset_index(drop=True)

    by_device.to_csv(BY_DEVICE_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    levels.to_csv(LEVELS_OUT, index=False)

    print("\nSTAGE 7 — CONFIGURATION SENSITIVITY ANALYSIS")
    print("=" * 64)
    print(f"Rows analysed: {len(df)}")
    print(f"Device profiles: {df['device_id'].nunique()}")
    print(f"Factors: {', '.join(FACTORS)}")
    print("\nIMPORTANT INTERPRETATION")
    print("-" * 64)
    print(
        "This is an empirical sensitivity analysis over the existing simulated "
        "configuration grid. It does NOT prove physical causality."
    )
    print(
        "Sensitivity is measured as the range of grouped mean outcomes when one "
        "configuration factor changes within each device."
    )

    print("\nGLOBAL SENSITIVITY SUMMARY")
    print("-" * 64)
    print(summary.round(3).to_string(index=False))

    print("\nDOMINANT ACCURACY FACTOR PER DEVICE")
    print("-" * 64)
    dominant = (
        by_device[by_device["accuracy_sensitivity_rank_within_device"] == 1]
        [["device_id", "factor", "accuracy_effect_range_pp"]]
        .sort_values("device_id")
    )
    print(dominant.round(3).to_string(index=False))

    print("\nSAVED OUTPUTS")
    print("-" * 64)
    print(BY_DEVICE_OUT)
    print(SUMMARY_OUT)
    print(LEVELS_OUT)


if __name__ == "__main__":
    main()
