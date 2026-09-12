from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("results/tables/ml_dataset.csv")
OUT_DIR = Path("results/tables")

DETAIL_OUT = OUT_DIR / "nonideality_proxy_by_device.csv"
SUMMARY_OUT = OUT_DIR / "nonideality_proxy_summary.csv"
SCOPE_OUT = OUT_DIR / "nonideality_proxy_scope.json"


SCENARIOS = [
    {
        "scenario_id": "BASELINE",
        "weight_bit_loss": 0,
        "adc_bit_loss": 0,
        "proxy_for": "none",
    },
    {
        "scenario_id": "WEIGHT_PRECISION_LOSS_1",
        "weight_bit_loss": 1,
        "adc_bit_loss": 0,
        "proxy_for": "reduced distinguishable conductance / effective weight precision",
    },
    {
        "scenario_id": "ADC_PRECISION_LOSS_1",
        "weight_bit_loss": 0,
        "adc_bit_loss": 1,
        "proxy_for": "reduced effective readout / ADC resolution",
    },
    {
        "scenario_id": "COMBINED_PRECISION_LOSS_1",
        "weight_bit_loss": 1,
        "adc_bit_loss": 1,
        "proxy_for": "combined mapping/readout precision degradation",
    },
    {
        "scenario_id": "COMBINED_PRECISION_LOSS_2",
        "weight_bit_loss": 2,
        "adc_bit_loss": 2,
        "proxy_for": "stronger combined mapping/readout precision degradation",
    },
]


def nearest_available_lower_or_equal(values, target):
    vals = sorted({int(v) for v in values if pd.notna(v)})
    if not vals:
        raise ValueError("No available precision values.")
    lower = [v for v in vals if v <= target]
    if lower:
        return max(lower)
    return min(vals)


def choose_stressed_row(device_df, crossbar, target_w, target_adc):
    same_crossbar = device_df[device_df["crossbar_size"] == crossbar].copy()
    if same_crossbar.empty:
        same_crossbar = device_df.copy()

    exact = same_crossbar[
        (same_crossbar["requested_weight_bits"] == target_w)
        & (same_crossbar["adc_bits"] == target_adc)
    ]
    if not exact.empty:
        # If duplicates ever exist, keep highest-accuracy representation.
        row = exact.sort_values("accuracy", ascending=False).iloc[0]
        return row, True

    tmp = same_crossbar.copy()
    tmp["_distance"] = (
        (tmp["requested_weight_bits"] - target_w).abs()
        + (tmp["adc_bits"] - target_adc).abs()
    )
    row = tmp.sort_values(
        ["_distance", "accuracy"],
        ascending=[True, False]
    ).iloc[0]
    return row, False


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
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows = []

    for device_id, g in df.groupby("device_id", sort=True):
        g = g.copy()

        # Robustness reference = best raw simulated accuracy, not a cost-aware optimum.
        baseline = g.sort_values(
            ["accuracy", "relative_hardware_cost_proxy"],
            ascending=[False, True]
        ).iloc[0]

        base_crossbar = int(baseline["crossbar_size"])
        base_w = int(baseline["requested_weight_bits"])
        base_adc = int(baseline["adc_bits"])
        base_acc = float(baseline["accuracy"])
        base_cost = float(baseline["relative_hardware_cost_proxy"])

        available_w = g["requested_weight_bits"].dropna().astype(int).unique()
        available_adc = g["adc_bits"].dropna().astype(int).unique()

        for s in SCENARIOS:
            requested_target_w = base_w - int(s["weight_bit_loss"])
            requested_target_adc = base_adc - int(s["adc_bit_loss"])

            target_w = nearest_available_lower_or_equal(
                available_w, requested_target_w
            )
            target_adc = nearest_available_lower_or_equal(
                available_adc, requested_target_adc
            )

            stressed, exact = choose_stressed_row(
                g,
                base_crossbar,
                target_w,
                target_adc,
            )

            stressed_acc = float(stressed["accuracy"])
            stressed_cost = float(stressed["relative_hardware_cost_proxy"])

            rows.append(
                {
                    "device_id": device_id,
                    "scenario_id": s["scenario_id"],
                    "proxy_for": s["proxy_for"],
                    "proxy_only": True,
                    "calibrated_to_device_nonideality_data": False,
                    "baseline_crossbar": base_crossbar,
                    "baseline_weight_bits": base_w,
                    "baseline_adc_bits": base_adc,
                    "baseline_accuracy": base_acc,
                    "baseline_relative_hardware_cost_proxy": base_cost,
                    "requested_weight_bit_loss": s["weight_bit_loss"],
                    "requested_adc_bit_loss": s["adc_bit_loss"],
                    "target_weight_bits": target_w,
                    "target_adc_bits": target_adc,
                    "selected_crossbar": int(stressed["crossbar_size"]),
                    "selected_weight_bits": int(stressed["requested_weight_bits"]),
                    "selected_adc_bits": int(stressed["adc_bits"]),
                    "exact_target_configuration_found": bool(exact),
                    "stressed_accuracy": stressed_acc,
                    "accuracy_drop_pp": base_acc - stressed_acc,
                    "stressed_relative_hardware_cost_proxy": stressed_cost,
                    "cost_proxy_change": stressed_cost - base_cost,
                }
            )

    detail = pd.DataFrame(rows)

    nonbaseline = detail[detail["scenario_id"] != "BASELINE"].copy()

    summary = (
        nonbaseline.groupby(
            ["scenario_id", "proxy_for"],
            as_index=False
        )
        .agg(
            devices=("device_id", "nunique"),
            mean_accuracy_drop_pp=("accuracy_drop_pp", "mean"),
            median_accuracy_drop_pp=("accuracy_drop_pp", "median"),
            max_accuracy_drop_pp=("accuracy_drop_pp", "max"),
            min_accuracy_drop_pp=("accuracy_drop_pp", "min"),
            exact_target_rate=("exact_target_configuration_found", "mean"),
        )
        .sort_values("median_accuracy_drop_pp", ascending=False)
        .reset_index(drop=True)
    )

    detail.to_csv(DETAIL_OUT, index=False)
    summary.to_csv(SUMMARY_OUT, index=False)

    scope = {
        "stage": 10,
        "name": "Nonideality robustness proxy",
        "status": "UNCALIBRATED_PROXY",
        "what_it_does": [
            "Reuses the existing simulated configuration grid.",
            "Represents nonideality stress as loss of effective weight precision and/or ADC precision.",
            "Measures the resulting accuracy change relative to each device's best raw simulated configuration.",
        ],
        "what_it_does_not_do": [
            "It does not inject measured cycle-to-cycle variability.",
            "It does not inject measured device-to-device variability.",
            "It does not model retention drift, endurance degradation, temperature dependence, or stochastic read noise directly.",
            "It is not calibrated to a probability distribution from experimental devices.",
            "It must not be presented as measured physical robustness.",
        ],
        "interpretation": (
            "This is a transparent stress-test proxy for how accelerator performance "
            "responds when nonideal hardware effectively reduces usable mapping or "
            "readout precision. Direct device-physics nonideality models require "
            "experimentally supported distributions and are future refinement."
        ),
        "scenarios": SCENARIOS,
    }

    SCOPE_OUT.write_text(json.dumps(scope, indent=2), encoding="utf-8")

    print("\nSTAGE 10 — NONIDEALITY ROBUSTNESS PROXY")
    print("=" * 72)
    print(f"Device profiles: {detail['device_id'].nunique()}")
    print(f"Existing simulator rows reused: {len(df)}")
    print("New exhaustive device sweeps required: 0")

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 72)
    print(
        "These are UNCALIBRATED EFFECTIVE-PRECISION STRESS TESTS, "
        "not measured physical noise/variability/drift models."
    )
    print(
        "Weight-bit loss is used only as a proxy for reduced usable conductance "
        "resolution; ADC-bit loss is used only as a proxy for reduced readout resolution."
    )

    print("\nSCENARIO SUMMARY")
    print("-" * 72)
    print(summary.round(3).to_string(index=False))

    print("\nMOST SENSITIVE DEVICE PER SCENARIO")
    print("-" * 72)
    worst_rows = []
    for scenario_id, g in nonbaseline.groupby("scenario_id", sort=True):
        r = g.sort_values("accuracy_drop_pp", ascending=False).iloc[0]
        worst_rows.append(
            {
                "scenario_id": scenario_id,
                "device_id": r["device_id"],
                "accuracy_drop_pp": r["accuracy_drop_pp"],
                "baseline_accuracy": r["baseline_accuracy"],
                "stressed_accuracy": r["stressed_accuracy"],
            }
        )
    worst = pd.DataFrame(worst_rows)
    print(worst.round(3).to_string(index=False))

    print("\nSAVED OUTPUTS")
    print("-" * 72)
    print(DETAIL_OUT)
    print(SUMMARY_OUT)
    print(SCOPE_OUT)


if __name__ == "__main__":
    main()
