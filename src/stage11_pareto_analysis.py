from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


INPUT = Path("results/tables/ml_dataset.csv")
OUT_DIR = Path("results/tables")

FRONT_OUT = OUT_DIR / "pareto_front_by_device.csv"
SUMMARY_OUT = OUT_DIR / "pareto_summary_by_device.csv"
OVERVIEW_OUT = OUT_DIR / "pareto_overview.csv"
SCOPE_OUT = OUT_DIR / "pareto_scope.json"

ACCURACY_TOLERANCE_PP = 0.5


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maximize accuracy, minimize relative_hardware_cost_proxy.
    Returns a strict 2-objective non-dominated set.

    We first keep the highest-accuracy row at each identical cost, then
    sweep from low cost to high cost and keep only rows that improve accuracy.
    """
    x = df.copy()
    x = x.sort_values(
        ["relative_hardware_cost_proxy", "accuracy"],
        ascending=[True, False]
    )

    # At identical cost, only the best accuracy can be non-dominated.
    x = x.drop_duplicates(
        subset=["relative_hardware_cost_proxy"],
        keep="first"
    )

    keep = []
    best_accuracy_so_far = -np.inf

    for idx, row in x.iterrows():
        acc = float(row["accuracy"])
        if acc > best_accuracy_so_far + 1e-12:
            keep.append(idx)
            best_accuracy_so_far = acc

    return x.loc[keep].copy()


def select_best_accuracy(g: pd.DataFrame) -> pd.Series:
    return g.sort_values(
        ["accuracy", "relative_hardware_cost_proxy"],
        ascending=[False, True]
    ).iloc[0]


def select_cheapest_within_tolerance(
    g: pd.DataFrame,
    best_accuracy: float,
    tolerance_pp: float,
) -> pd.Series:
    eligible = g[g["accuracy"] >= best_accuracy - tolerance_pp].copy()
    if eligible.empty:
        return select_best_accuracy(g)

    return eligible.sort_values(
        ["relative_hardware_cost_proxy", "accuracy"],
        ascending=[True, False]
    ).iloc[0]


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run src/build_ml_dataset.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    required = {
        "device_id",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
        "accuracy",
        "relative_hardware_cost_proxy",
        "estimated_memristor_cells",
        "estimated_physical_crossbar_tiles",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_fronts = []
    summaries = []

    for device_id, g in df.groupby("device_id", sort=True):
        g = g.copy()
        front = pareto_front(g)
        front["device_id"] = device_id

        # Order front from cheapest to highest cost.
        front = front.sort_values(
            ["relative_hardware_cost_proxy", "accuracy"],
            ascending=[True, True]
        ).reset_index(drop=True)

        front["pareto_rank_by_cost"] = np.arange(1, len(front) + 1)
        front["is_pareto_optimal"] = True

        best = select_best_accuracy(g)
        best_acc = float(best["accuracy"])

        cheap = g.sort_values(
            ["relative_hardware_cost_proxy", "accuracy"],
            ascending=[True, False]
        ).iloc[0]

        tolerant = select_cheapest_within_tolerance(
            g,
            best_accuracy=best_acc,
            tolerance_pp=ACCURACY_TOLERANCE_PP,
        )

        # Mark notable reference points on the Pareto table.
        front["reference_best_accuracy"] = (
            (front["crossbar_size"] == best["crossbar_size"])
            & (front["requested_weight_bits"] == best["requested_weight_bits"])
            & (front["adc_bits"] == best["adc_bits"])
        )

        front["reference_cheapest_within_0p5pp"] = (
            (front["crossbar_size"] == tolerant["crossbar_size"])
            & (front["requested_weight_bits"] == tolerant["requested_weight_bits"])
            & (front["adc_bits"] == tolerant["adc_bits"])
        )

        all_fronts.append(front)

        cost_reduction = (
            100.0
            * (float(best["relative_hardware_cost_proxy"])
               - float(tolerant["relative_hardware_cost_proxy"]))
            / max(float(best["relative_hardware_cost_proxy"]), 1e-12)
        )

        summaries.append(
            {
                "device_id": device_id,
                "total_configurations": len(g),
                "pareto_points": len(front),
                "pareto_fraction": len(front) / len(g),
                "best_accuracy": best_acc,
                "best_accuracy_crossbar": int(best["crossbar_size"]),
                "best_accuracy_weight_bits": int(best["requested_weight_bits"]),
                "best_accuracy_adc_bits": int(best["adc_bits"]),
                "best_accuracy_cost_proxy": float(best["relative_hardware_cost_proxy"]),
                "absolute_min_cost_proxy": float(cheap["relative_hardware_cost_proxy"]),
                "absolute_min_cost_accuracy": float(cheap["accuracy"]),
                "within_0p5pp_accuracy": float(tolerant["accuracy"]),
                "within_0p5pp_accuracy_loss_pp": best_acc - float(tolerant["accuracy"]),
                "within_0p5pp_crossbar": int(tolerant["crossbar_size"]),
                "within_0p5pp_weight_bits": int(tolerant["requested_weight_bits"]),
                "within_0p5pp_adc_bits": int(tolerant["adc_bits"]),
                "within_0p5pp_cost_proxy": float(tolerant["relative_hardware_cost_proxy"]),
                "within_0p5pp_cost_reduction_vs_best_pct": cost_reduction,
                "within_0p5pp_memristor_cells": int(tolerant["estimated_memristor_cells"]),
                "within_0p5pp_crossbar_tiles": int(tolerant["estimated_physical_crossbar_tiles"]),
            }
        )

    fronts = pd.concat(all_fronts, ignore_index=True)
    summary = pd.DataFrame(summaries)

    overview = pd.DataFrame(
        [
            {
                "metric": "device_profiles",
                "value": int(summary["device_id"].nunique()),
            },
            {
                "metric": "simulator_rows_reused",
                "value": int(len(df)),
            },
            {
                "metric": "median_pareto_points_per_device",
                "value": float(summary["pareto_points"].median()),
            },
            {
                "metric": "mean_pareto_points_per_device",
                "value": float(summary["pareto_points"].mean()),
            },
            {
                "metric": "median_0p5pp_cost_reduction_vs_best_pct",
                "value": float(
                    summary["within_0p5pp_cost_reduction_vs_best_pct"].median()
                ),
            },
            {
                "metric": "mean_0p5pp_cost_reduction_vs_best_pct",
                "value": float(
                    summary["within_0p5pp_cost_reduction_vs_best_pct"].mean()
                ),
            },
            {
                "metric": "max_0p5pp_cost_reduction_vs_best_pct",
                "value": float(
                    summary["within_0p5pp_cost_reduction_vs_best_pct"].max()
                ),
            },
        ]
    )

    fronts.to_csv(FRONT_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)
    overview.to_csv(OVERVIEW_OUT, index=False)

    scope = {
        "stage": 11,
        "name": "Hardware-cost / accuracy Pareto analysis",
        "objectives": {
            "maximize": "simulated accuracy",
            "minimize": "relative_hardware_cost_proxy",
        },
        "important_limits": [
            "relative_hardware_cost_proxy is a heuristic, not measured energy, area, latency, power, or monetary cost",
            "Pareto fronts are computed from existing simulator outputs, not physical measurements",
            "the cheapest-within-0.5pp reference uses TRUE simulated accuracy and is therefore an oracle/reference benchmark, not an unseen-device ML recommendation",
            "do not use this stage to claim measured hardware efficiency",
        ],
        "accuracy_tolerance_pp": ACCURACY_TOLERANCE_PP,
        "outputs": [
            str(FRONT_OUT),
            str(SUMMARY_OUT),
            str(OVERVIEW_OUT),
        ],
    }
    SCOPE_OUT.write_text(json.dumps(scope, indent=2), encoding="utf-8")

    print("\nSTAGE 11 — HARDWARE-COST / ACCURACY PARETO ANALYSIS")
    print("=" * 76)
    print(f"Device profiles: {summary['device_id'].nunique()}")
    print(f"Existing simulator rows reused: {len(df)}")
    print("New exhaustive sweeps required: 0")

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 76)
    print(
        "relative_hardware_cost_proxy is a HEURISTIC design-complexity proxy, "
        "not measured energy, area, latency, power, or monetary cost."
    )
    print(
        "The cheapest-within-0.5pp point uses TRUE simulated accuracy. "
        "It is an oracle/reference benchmark, NOT an unseen-device ML recommendation."
    )

    print("\nPARETO SUMMARY BY DEVICE")
    print("-" * 76)
    cols = [
        "device_id",
        "pareto_points",
        "best_accuracy",
        "best_accuracy_cost_proxy",
        "within_0p5pp_accuracy",
        "within_0p5pp_accuracy_loss_pp",
        "within_0p5pp_cost_proxy",
        "within_0p5pp_cost_reduction_vs_best_pct",
    ]
    print(summary[cols].round(3).to_string(index=False))

    print("\nOVERALL")
    print("-" * 76)
    print(overview.round(3).to_string(index=False))

    print("\nSAVED OUTPUTS")
    print("-" * 76)
    print(FRONT_OUT)
    print(SUMMARY_OUT)
    print(OVERVIEW_OUT)
    print(SCOPE_OUT)


if __name__ == "__main__":
    main()
