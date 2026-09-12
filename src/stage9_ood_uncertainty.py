from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd


ML_INPUT = Path("results/tables/ml_dataset.csv")
PRED_INPUT = Path("results/tables/zero_shot_predictions.csv")
SUMMARY_INPUT = Path("results/tables/zero_shot_summary.csv")

OUT_DIR = Path("results/tables")
DEVICE_OUT = OUT_DIR / "ood_uncertainty_by_device.csv"
OVERVIEW_OUT = OUT_DIR / "ood_uncertainty_overview.csv"

NUMERIC_FEATURES = [
    "device_log10_on_off_ratio",
    "state_count_available",
    "physical_state_count",
]
CATEGORICAL_FEATURES = [
    "device_conductance_mode",
    "mapping_strategy",
]


def first_non_null(s: pd.Series):
    s = s.dropna()
    return s.iloc[0] if not s.empty else np.nan


def build_device_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "device_id",
        "study_id",
        "technology_family",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"ml_dataset.csv missing columns: {missing}")

    rows = []
    for device_id, g in df.groupby("device_id", sort=True):
        r = {"device_id": device_id}
        for c in cols:
            if c == "device_id":
                continue
            r[c] = first_non_null(g[c])
        rows.append(r)
    return pd.DataFrame(rows)


def normalized_distance(test_row: pd.Series, train_df: pd.DataFrame):
    # Training-only scaling to avoid using held-out device statistics.
    numeric_parts = []
    prepared = {}

    for c in NUMERIC_FEATURES:
        tr = pd.to_numeric(train_df[c], errors="coerce")
        median = float(tr.median()) if tr.notna().any() else 0.0
        std = float(tr.std(ddof=0)) if tr.notna().sum() > 1 else 0.0
        if not np.isfinite(std) or std < 1e-12:
            std = 1.0
        tv = pd.to_numeric(pd.Series([test_row[c]]), errors="coerce").iloc[0]
        if pd.isna(tv):
            tv = median
        prepared[c] = (tr.fillna(median), float(tv), median, std)

    distances = []
    for idx, tr_row in train_df.iterrows():
        num_diffs = []
        for c in NUMERIC_FEATURES:
            tr_values, tv, _, std = prepared[c]
            trv = float(tr_values.loc[idx])
            num_diffs.append(((tv - trv) / std) ** 2)

        numeric_distance = math.sqrt(sum(num_diffs) / max(len(num_diffs), 1))

        cat_mismatches = []
        for c in CATEGORICAL_FEATURES:
            a = str(test_row[c])
            b = str(tr_row[c])
            cat_mismatches.append(0.0 if a == b else 1.0)
        categorical_distance = (
            sum(cat_mismatches) / len(cat_mismatches)
            if cat_mismatches else 0.0
        )

        # Simple transparent heuristic, not a calibrated probability.
        combined = numeric_distance + categorical_distance
        distances.append(
            (
                idx,
                combined,
                numeric_distance,
                categorical_distance,
            )
        )

    distances.sort(key=lambda x: x[1])
    return distances[0]


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    if x[mask].std(ddof=0) < 1e-12 or y[mask].std(ddof=0) < 1e-12:
        return np.nan
    return float(x[mask].corr(y[mask]))


def main():
    for p in [ML_INPUT, PRED_INPUT, SUMMARY_INPUT]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(ML_INPUT)
    pred = pd.read_csv(PRED_INPUT)
    summary = pd.read_csv(SUMMARY_INPUT)

    devices = build_device_descriptors(ml)

    needed_summary = {"held_out_device", "baseline_regret_pp"}
    if not needed_summary.issubset(summary.columns):
        raise ValueError(
            "zero_shot_summary.csv must contain held_out_device and baseline_regret_pp"
        )

    pred_needed = {
        "held_out_device",
        "absolute_prediction_error_pp",
        "prediction_uncertainty",
    }
    if not pred_needed.issubset(pred.columns):
        raise ValueError(
            "zero_shot_predictions.csv is missing required prediction/uncertainty columns"
        )

    pred_agg = (
        pred.groupby("held_out_device", as_index=False)
        .agg(
            mean_absolute_prediction_error_pp=("absolute_prediction_error_pp", "mean"),
            max_absolute_prediction_error_pp=("absolute_prediction_error_pp", "max"),
            mean_tree_disagreement=("prediction_uncertainty", "mean"),
            max_tree_disagreement=("prediction_uncertainty", "max"),
        )
    )

    rows = []

    for _, test in devices.iterrows():
        held = test["device_id"]
        held_study = test["study_id"]

        train = devices[devices["study_id"] != held_study].copy()
        if train.empty:
            continue

        nearest_idx, combined, num_dist, cat_dist = normalized_distance(test, train)
        nearest = train.loc[nearest_idx]

        rows.append(
            {
                "held_out_device": held,
                "held_out_study": held_study,
                "held_out_family": test["technology_family"],
                "training_device_count": len(train),
                "nearest_training_device": nearest["device_id"],
                "nearest_training_family": nearest["technology_family"],
                "nearest_training_study": nearest["study_id"],
                "ood_numeric_distance": num_dist,
                "ood_categorical_mismatch_fraction": cat_dist,
                "ood_combined_score": combined,
            }
        )

    out = pd.DataFrame(rows)
    out = out.merge(
        summary[
            [
                "held_out_device",
                "baseline_regret_pp",
                "confidence_regret_pp",
                "baseline_near_optimal_success",
                "confidence_near_optimal_success",
            ]
        ],
        on="held_out_device",
        how="left",
    )
    out = out.merge(pred_agg, on="held_out_device", how="left")

    # Relative rank only. This is NOT a calibrated OOD probability.
    out["ood_relative_percentile"] = (
        out["ood_combined_score"].rank(method="average", pct=True) * 100.0
    )
    out["ood_relative_rank"] = (
        out["ood_combined_score"].rank(method="min", ascending=False).astype(int)
    )

    # A transparent warning flag for cases where recommendation regret is large
    # but ensemble disagreement is not correspondingly extreme.
    unc_med = float(out["mean_tree_disagreement"].median())
    out["possible_overconfident_failure"] = (
        (pd.to_numeric(out["baseline_regret_pp"], errors="coerce") > 2.0)
        & (pd.to_numeric(out["mean_tree_disagreement"], errors="coerce") <= unc_med)
    )

    out = out.sort_values(
        ["ood_combined_score", "baseline_regret_pp"],
        ascending=[False, False],
    ).reset_index(drop=True)

    overview_rows = [
        {
            "metric": "device_profiles",
            "value": len(out),
        },
        {
            "metric": "ood_score_vs_mean_abs_prediction_error_corr",
            "value": safe_corr(
                out["ood_combined_score"],
                out["mean_absolute_prediction_error_pp"],
            ),
        },
        {
            "metric": "ood_score_vs_baseline_regret_corr",
            "value": safe_corr(
                out["ood_combined_score"],
                out["baseline_regret_pp"],
            ),
        },
        {
            "metric": "tree_disagreement_vs_mean_abs_error_corr",
            "value": safe_corr(
                out["mean_tree_disagreement"],
                out["mean_absolute_prediction_error_pp"],
            ),
        },
        {
            "metric": "tree_disagreement_vs_baseline_regret_corr",
            "value": safe_corr(
                out["mean_tree_disagreement"],
                out["baseline_regret_pp"],
            ),
        },
        {
            "metric": "possible_overconfident_failures",
            "value": int(out["possible_overconfident_failure"].sum()),
        },
    ]
    overview = pd.DataFrame(overview_rows)

    out.to_csv(DEVICE_OUT, index=False)
    overview.to_csv(OVERVIEW_OUT, index=False)

    print("\nSTAGE 9 — OOD + UNCERTAINTY DIAGNOSTIC")
    print("=" * 68)
    print(f"Device profiles: {len(out)}")
    print("Primary split policy: study-blocked held-out device")
    print("\nIMPORTANT")
    print("-" * 68)
    print(
        "The OOD score is a relative descriptor-distance heuristic, NOT a "
        "calibrated probability and NOT a claim of physical novelty."
    )
    print(
        "Tree disagreement remains an uncalibrated ensemble heuristic."
    )

    print("\nRELATIVE OOD / ERROR TABLE")
    print("-" * 68)
    cols = [
        "held_out_device",
        "nearest_training_device",
        "ood_combined_score",
        "ood_relative_percentile",
        "baseline_regret_pp",
        "mean_absolute_prediction_error_pp",
        "mean_tree_disagreement",
        "possible_overconfident_failure",
    ]
    print(out[cols].round(3).to_string(index=False))

    print("\nOVERVIEW")
    print("-" * 68)
    print(overview.round(3).to_string(index=False))

    print("\nSAVED OUTPUTS")
    print("-" * 68)
    print(DEVICE_OUT)
    print(OVERVIEW_OUT)


if __name__ == "__main__":
    main()
