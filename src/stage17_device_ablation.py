from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

ML_FILE = RESULTS_DIR / "ml_dataset.csv"
OUT_BY_DEVICE = RESULTS_DIR / "stage17_ablation_by_device.csv"
OUT_SUMMARY = RESULTS_DIR / "stage17_ablation_summary.csv"
OUT_SCOPE = RESULTS_DIR / "stage17_ablation_scope.json"

NEAR_OPTIMAL_TOLERANCE_PP = 0.5
RANDOM_STATE = 42

MODEL_SPECS = {
    "HARDWARE_ONLY": {
        "numeric": [
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
        ],
        "categorical": [],
        "description": "Independent accelerator controls only.",
    },
    "HARDWARE_PLUS_ON_OFF": {
        "numeric": [
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
            "device_log10_on_off_ratio",
        ],
        "categorical": [],
        "description": "Hardware controls plus log10 ON/OFF ratio.",
    },
    "HARDWARE_PLUS_ON_OFF_MODE": {
        "numeric": [
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
            "device_log10_on_off_ratio",
        ],
        "categorical": [
            "device_conductance_mode",
        ],
        "description": "Hardware controls plus ON/OFF ratio and conductance mode.",
    },
    "FULL_DEVICE_AWARE": {
        "numeric": [
            "device_log10_on_off_ratio",
            "state_count_available",
            "physical_state_count",
            "crossbar_size",
            "requested_weight_bits",
            "effective_weight_levels",
            "slices_per_branch",
            "physical_cells_per_weight",
            "adc_bits",
        ],
        "categorical": [
            "device_conductance_mode",
            "mapping_strategy",
        ],
        "description": "Current full device-aware feature set.",
    },
}

def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def make_pipeline(numeric_features, categorical_features):
    transformers = []
    if numeric_features:
        transformers.append(("num", "passthrough", numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        )

    preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

def choose_fixed_training_config(train_df):
    grouped = (
        train_df
        .groupby(
            ["crossbar_size", "requested_weight_bits", "adc_bits"],
            as_index=False,
        )
        .agg(mean_training_accuracy=("accuracy", "mean"))
        .sort_values(
            [
                "mean_training_accuracy",
                "requested_weight_bits",
                "adc_bits",
                "crossbar_size",
            ],
            ascending=[False, True, True, True],
        )
    )
    best = grouped.iloc[0]
    return (
        int(best["crossbar_size"]),
        int(best["requested_weight_bits"]),
        int(best["adc_bits"]),
    )

def select_accuracy_first(test_df, predicted):
    ranked = test_df.copy()
    ranked["predicted_accuracy"] = predicted

    sort_cols = ["predicted_accuracy"]
    ascending = [False]

    if "relative_hardware_cost_proxy" in ranked.columns:
        sort_cols.append("relative_hardware_cost_proxy")
        ascending.append(True)

    sort_cols += [
        "requested_weight_bits",
        "adc_bits",
        "crossbar_size",
    ]
    ascending += [True, True, True]

    return ranked.sort_values(
        sort_cols,
        ascending=ascending,
    ).iloc[0]

def rmse(errors):
    arr = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))

def main():
    if not ML_FILE.exists():
        raise FileNotFoundError(f"Missing ML dataset: {ML_FILE}")

    df = pd.read_csv(ML_FILE)

    require_columns(
        df,
        [
            "device_id",
            "study_id",
            "technology_family",
            "accuracy",
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
        ],
    )

    for spec in MODEL_SPECS.values():
        require_columns(
            df,
            spec["numeric"] + spec["categorical"],
        )

    if "slices_per_branch" in df.columns:
        df["slices_per_branch"] = (
            pd.to_numeric(df["slices_per_branch"], errors="coerce")
            .fillna(0)
        )

    rows = []

    print("=" * 78)
    print("STAGE 17 — DEVICE-AWARE VS DEVICE-AGNOSTIC ABLATION")
    print("=" * 78)
    print(f"Rows: {len(df)}")
    print(f"Devices: {df['device_id'].nunique()}")
    print(f"Studies: {df['study_id'].nunique()}")
    print(f"Families: {df['technology_family'].nunique()}")
    print("Split: hold out device AND exclude every profile from the same study.")
    print()

    for device_id in sorted(df["device_id"].astype(str).unique()):
        test = df[df["device_id"] == device_id].copy()
        held_out_study = str(test["study_id"].iloc[0])
        held_out_family = str(test["technology_family"].iloc[0])

        train = df[
            df["study_id"].astype(str) != held_out_study
        ].copy()

        oracle_best = float(test["accuracy"].max())

        fixed_crossbar, fixed_weight, fixed_adc = (
            choose_fixed_training_config(train)
        )

        fixed = test[
            (test["crossbar_size"] == fixed_crossbar)
            & (test["requested_weight_bits"] == fixed_weight)
            & (test["adc_bits"] == fixed_adc)
        ].iloc[0]

        fixed_actual = float(fixed["accuracy"])
        fixed_regret = oracle_best - fixed_actual

        rows.append(
            {
                "held_out_device": device_id,
                "held_out_study": held_out_study,
                "held_out_family": held_out_family,
                "method": "FIXED_TRAINING_CONFIG",
                "candidate_mae_pp": np.nan,
                "candidate_rmse_pp": np.nan,
                "recommended_crossbar": fixed_crossbar,
                "recommended_weight_bits": fixed_weight,
                "recommended_adc_bits": fixed_adc,
                "recommended_predicted_accuracy": np.nan,
                "recommended_actual_accuracy": fixed_actual,
                "oracle_best_accuracy": oracle_best,
                "recommendation_regret_pp": fixed_regret,
                "near_optimal_success": (
                    fixed_regret <= NEAR_OPTIMAL_TOLERANCE_PP
                ),
                "training_rows": len(train),
                "training_devices": train["device_id"].nunique(),
                "training_studies": train["study_id"].nunique(),
            }
        )

        print(
            f"[{device_id}] FIXED_TRAINING_CONFIG  "
            f"regret={fixed_regret:.3f} pp"
        )

        for method, spec in MODEL_SPECS.items():
            features = spec["numeric"] + spec["categorical"]

            model = make_pipeline(
                spec["numeric"],
                spec["categorical"],
            )

            model.fit(
                train[features],
                train["accuracy"].astype(float),
            )

            predicted = model.predict(test[features])
            actual = test["accuracy"].astype(float).to_numpy()
            errors = predicted - actual

            rec = select_accuracy_first(test, predicted)

            rec_actual = float(rec["accuracy"])
            rec_pred = float(rec["predicted_accuracy"])
            regret = oracle_best - rec_actual

            rows.append(
                {
                    "held_out_device": device_id,
                    "held_out_study": held_out_study,
                    "held_out_family": held_out_family,
                    "method": method,
                    "candidate_mae_pp": float(np.mean(np.abs(errors))),
                    "candidate_rmse_pp": rmse(errors),
                    "recommended_crossbar": int(rec["crossbar_size"]),
                    "recommended_weight_bits": int(
                        rec["requested_weight_bits"]
                    ),
                    "recommended_adc_bits": int(rec["adc_bits"]),
                    "recommended_predicted_accuracy": rec_pred,
                    "recommended_actual_accuracy": rec_actual,
                    "oracle_best_accuracy": oracle_best,
                    "recommendation_regret_pp": regret,
                    "near_optimal_success": (
                        regret <= NEAR_OPTIMAL_TOLERANCE_PP
                    ),
                    "training_rows": len(train),
                    "training_devices": train["device_id"].nunique(),
                    "training_studies": train["study_id"].nunique(),
                }
            )

            print(
                f"    {method:<27} "
                f"MAE={np.mean(np.abs(errors)):.3f}  "
                f"RMSE={rmse(errors):.3f}  "
                f"regret={regret:.3f}  "
                f"{'PASS' if regret <= NEAR_OPTIMAL_TOLERANCE_PP else 'FAIL'}"
            )

        print()

    by_device = pd.DataFrame(rows)

    method_order = [
        "FIXED_TRAINING_CONFIG",
        "HARDWARE_ONLY",
        "HARDWARE_PLUS_ON_OFF",
        "HARDWARE_PLUS_ON_OFF_MODE",
        "FULL_DEVICE_AWARE",
    ]

    summary_rows = []

    for method in method_order:
        part = by_device[by_device["method"] == method].copy()

        summary_rows.append(
            {
                "method": method,
                "devices": part["held_out_device"].nunique(),
                "mean_candidate_mae_pp": (
                    float(part["candidate_mae_pp"].mean())
                    if part["candidate_mae_pp"].notna().any()
                    else np.nan
                ),
                "mean_candidate_rmse_pp": (
                    float(part["candidate_rmse_pp"].mean())
                    if part["candidate_rmse_pp"].notna().any()
                    else np.nan
                ),
                "mean_recommendation_regret_pp": float(
                    part["recommendation_regret_pp"].mean()
                ),
                "median_recommendation_regret_pp": float(
                    part["recommendation_regret_pp"].median()
                ),
                "max_recommendation_regret_pp": float(
                    part["recommendation_regret_pp"].max()
                ),
                "near_optimal_success_count": int(
                    part["near_optimal_success"].sum()
                ),
                "near_optimal_success_rate_pct": float(
                    100 * part["near_optimal_success"].mean()
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)

    by_device.to_csv(OUT_BY_DEVICE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    scope = {
        "stage": 17,
        "title": "Device-aware vs device-agnostic ablation",
        "rows": int(len(df)),
        "devices": int(df["device_id"].nunique()),
        "studies": int(df["study_id"].nunique()),
        "families": int(df["technology_family"].nunique()),
        "near_optimal_tolerance_pp": NEAR_OPTIMAL_TOLERANCE_PP,
        "split_policy": (
            "For each held-out device, every row from its source study "
            "is excluded from training."
        ),
        "recommendation_policy": (
            "Accuracy-first for every ML ablation so feature-set value is "
            "isolated from the deployed support gate."
        ),
        "models": {
            name: spec["description"]
            for name, spec in MODEL_SPECS.items()
        },
        "interpretation": [
            "Lower error/regret for device-aware variants supports useful transferable information from nanodevice descriptors.",
            "Failure to improve is also scientifically informative.",
            "This is a pilot study on a small literature-derived dataset.",
        ],
    }

    OUT_SCOPE.write_text(
        json.dumps(scope, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("STAGE 17 SUMMARY")
    print("=" * 78)
    print(summary.to_string(index=False))
    print()
    print(f"Saved: {OUT_BY_DEVICE}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_SCOPE}")

if __name__ == "__main__":
    main()
