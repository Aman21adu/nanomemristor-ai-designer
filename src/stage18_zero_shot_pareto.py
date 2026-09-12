from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
ML_FILE = RESULTS_DIR / "ml_dataset.csv"

OUT_BY_DEVICE = RESULTS_DIR / "stage18_pareto_by_device.csv"
OUT_SUMMARY = RESULTS_DIR / "stage18_pareto_summary.csv"
OUT_POINTS = RESULTS_DIR / "stage18_pareto_points.csv"
OUT_SCOPE = RESULTS_DIR / "stage18_pareto_scope.json"

RANDOM_STATE = 42
EPS = 1e-12

# Stage 18 intentionally compares a device-agnostic baseline against
# the full device-aware model. That directly tests whether nano descriptors
# improve prediction of the unseen-device trade-off frontier.
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

BUDGET_QUANTILES = [0.25, 0.50, 0.75, 1.00]


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def make_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
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

    preprocess = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

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


def pareto_mask(cost: np.ndarray, accuracy: np.ndarray) -> np.ndarray:
    """
    True for nondominated rows when lower cost and higher accuracy are better.

    A point is dominated if another point has cost <= it and accuracy >= it,
    with at least one strict improvement.
    """
    cost = np.asarray(cost, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)

    n = len(cost)
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue

        dominated = (
            (cost <= cost[i] + EPS)
            & (accuracy >= accuracy[i] - EPS)
            & (
                (cost < cost[i] - EPS)
                | (accuracy > accuracy[i] + EPS)
            )
        )

        if np.any(dominated):
            keep[i] = False

    return keep


def safe_range(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    if abs(span) < EPS:
        span = 1.0
    return lo, hi, span


def normalized_front_points(
    cost: np.ndarray,
    accuracy: np.ndarray,
    cost_lo: float,
    cost_span: float,
    acc_lo: float,
    acc_span: float,
) -> np.ndarray:
    cost_n = (np.asarray(cost, dtype=float) - cost_lo) / cost_span
    acc_n = (np.asarray(accuracy, dtype=float) - acc_lo) / acc_span
    return np.column_stack([cost_n, acc_n])


def symmetric_chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean nearest-neighbor distance in both directions, averaged.
    Coordinates are normalized before this function is called.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    dists = np.sqrt(
        np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    )

    a_to_b = np.min(dists, axis=1).mean()
    b_to_a = np.min(dists, axis=0).mean()

    return float((a_to_b + b_to_a) / 2.0)


def normalized_hypervolume(
    cost: np.ndarray,
    accuracy: np.ndarray,
    cost_lo: float,
    cost_span: float,
    acc_lo: float,
    acc_span: float,
) -> float:
    """
    Correct 2D normalized hypervolume with reference point (0, 0).

    Low cost is converted to a utility in [0, 1], where larger is better.
    Accuracy is normalized using the held-out device's true accuracy range.
    The dominated union area is then integrated exactly in 2D.

    This is a descriptive normalized metric only; the cost axis remains the
    project's relative hardware-cost proxy, not measured energy/area/latency.
    """
    if len(cost) == 0:
        return float("nan")

    utility = 1.0 - ((np.asarray(cost, dtype=float) - cost_lo) / cost_span)
    quality = (np.asarray(accuracy, dtype=float) - acc_lo) / acc_span

    utility = np.clip(utility, 0.0, 1.0)
    quality = np.clip(quality, 0.0, 1.0)

    points = pd.DataFrame({"x": utility, "y": quality})
    points = (
        points.groupby("x", as_index=False)["y"]
        .max()
        .sort_values("x")
    )

    x = points["x"].to_numpy(dtype=float)
    y = points["y"].to_numpy(dtype=float)

    # For rectangles anchored at (0, 0), at horizontal position t the
    # dominated height is max(y_i) over all points with x_i >= t.
    suffix_best_y = np.maximum.accumulate(y[::-1])[::-1]

    hv = 0.0
    prev_x = 0.0

    for xi, envelope_y in zip(x, suffix_best_y):
        if xi > prev_x:
            hv += (xi - prev_x) * envelope_y
            prev_x = xi

    return float(np.clip(hv, 0.0, 1.0))


def budget_regrets(
    test: pd.DataFrame,
    predicted_accuracy: np.ndarray,
    cost_col: str,
) -> tuple[float, float, list[dict]]:
    """
    At several cost budgets:
      1) choose the configuration with highest predicted accuracy under budget;
      2) reveal its true accuracy;
      3) compare against the true best accuracy under the same budget.
    """
    work = test.copy()
    work["_pred"] = np.asarray(predicted_accuracy, dtype=float)

    unique_costs = np.sort(work[cost_col].astype(float).unique())
    details = []
    regrets = []

    for q in BUDGET_QUANTILES:
        budget = float(np.quantile(unique_costs, q))

        feasible = work[work[cost_col].astype(float) <= budget + EPS].copy()

        if feasible.empty:
            continue

        predicted_choice = (
            feasible.sort_values(
                ["_pred", cost_col],
                ascending=[False, True],
            )
            .iloc[0]
        )

        oracle_accuracy = float(feasible["accuracy"].max())
        chosen_actual = float(predicted_choice["accuracy"])
        regret = oracle_accuracy - chosen_actual

        regrets.append(regret)

        details.append(
            {
                "budget_quantile": q,
                "budget": budget,
                "oracle_accuracy": oracle_accuracy,
                "chosen_actual_accuracy": chosen_actual,
                "chosen_predicted_accuracy": float(predicted_choice["_pred"]),
                "regret_pp": regret,
            }
        )

    if not regrets:
        return float("nan"), float("nan"), details

    return float(np.mean(regrets)), float(np.max(regrets)), details


def main() -> None:
    if not ML_FILE.exists():
        raise FileNotFoundError(f"Missing ML dataset: {ML_FILE}")

    df = pd.read_csv(ML_FILE)

    base_required = [
        "device_id",
        "study_id",
        "technology_family",
        "accuracy",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
    ]
    require_columns(df, base_required)

    cost_candidates = [
        "relative_hardware_cost_proxy",
        "hardware_cost_proxy",
        "relative_cost_proxy",
    ]
    cost_col = next((c for c in cost_candidates if c in df.columns), None)

    if cost_col is None:
        raise ValueError(
            "Could not find the relative hardware-cost proxy column. "
            f"Tried: {cost_candidates}"
        )

    for spec in MODEL_SPECS.values():
        require_columns(df, spec["numeric"] + spec["categorical"])

    if "slices_per_branch" in df.columns:
        df["slices_per_branch"] = (
            pd.to_numeric(df["slices_per_branch"], errors="coerce")
            .fillna(0)
        )

    print("=" * 84)
    print("STAGE 18 — ZERO-SHOT PARETO FRONTIER TRANSFER")
    print("=" * 84)
    print(f"Rows: {len(df)}")
    print(f"Devices: {df['device_id'].nunique()}")
    print(f"Studies: {df['study_id'].nunique()}")
    print(f"Families: {df['technology_family'].nunique()}")
    print(f"Cost axis: {cost_col}")
    print("Split: hold out device AND exclude every profile from the same study.")
    print("Frontier objective: maximize accuracy, minimize relative hardware-cost proxy.")
    print()

    metric_rows = []
    point_rows = []

    for device_id in sorted(df["device_id"].astype(str).unique()):
        test = df[df["device_id"] == device_id].copy()

        held_out_study = str(test["study_id"].iloc[0])
        held_out_family = str(test["technology_family"].iloc[0])

        train = df[
            df["study_id"].astype(str) != held_out_study
        ].copy()

        actual = test["accuracy"].astype(float).to_numpy()
        cost = test[cost_col].astype(float).to_numpy()

        true_mask = pareto_mask(cost, actual)

        cost_lo, _, cost_span = safe_range(cost)
        acc_lo, _, acc_span = safe_range(actual)

        true_front_geom = normalized_front_points(
            cost[true_mask],
            actual[true_mask],
            cost_lo,
            cost_span,
            acc_lo,
            acc_span,
        )

        true_hv = normalized_hypervolume(
            cost[true_mask],
            actual[true_mask],
            cost_lo,
            cost_span,
            acc_lo,
            acc_span,
        )

        print(
            f"[{device_id}] true Pareto points={int(true_mask.sum())}"
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

            pred_mask = pareto_mask(cost, predicted)

            overlap = int(np.sum(true_mask & pred_mask))
            true_count = int(true_mask.sum())
            pred_count = int(pred_mask.sum())

            recall = overlap / true_count if true_count else float("nan")
            precision = overlap / pred_count if pred_count else float("nan")

            if (
                np.isfinite(recall)
                and np.isfinite(precision)
                and (recall + precision) > 0
            ):
                f1 = 2.0 * recall * precision / (recall + precision)
            else:
                f1 = 0.0

            pred_front_geom = normalized_front_points(
                cost[pred_mask],
                predicted[pred_mask],
                cost_lo,
                cost_span,
                acc_lo,
                acc_span,
            )

            front_distance = symmetric_chamfer_distance(
                true_front_geom,
                pred_front_geom,
            )

            pred_hv = normalized_hypervolume(
                cost[pred_mask],
                predicted[pred_mask],
                cost_lo,
                cost_span,
                acc_lo,
                acc_span,
            )

            hv_abs_error = abs(pred_hv - true_hv)

            mean_budget_regret, max_budget_regret, budget_details = budget_regrets(
                test,
                predicted,
                cost_col,
            )

            metric_rows.append(
                {
                    "held_out_device": device_id,
                    "held_out_study": held_out_study,
                    "held_out_family": held_out_family,
                    "method": method,
                    "true_pareto_points": true_count,
                    "predicted_pareto_points": pred_count,
                    "exact_overlap_points": overlap,
                    "pareto_recall": recall,
                    "pareto_precision": precision,
                    "pareto_f1": f1,
                    "normalized_front_distance": front_distance,
                    "true_normalized_hypervolume": true_hv,
                    "predicted_normalized_hypervolume": pred_hv,
                    "hypervolume_abs_error": hv_abs_error,
                    "mean_cost_budget_regret_pp": mean_budget_regret,
                    "max_cost_budget_regret_pp": max_budget_regret,
                    "training_rows": len(train),
                    "training_devices": train["device_id"].nunique(),
                    "training_studies": train["study_id"].nunique(),
                    "budget_details_json": json.dumps(budget_details),
                }
            )

            for local_pos, (_, row) in enumerate(test.iterrows()):
                point_rows.append(
                    {
                        "held_out_device": device_id,
                        "held_out_study": held_out_study,
                        "held_out_family": held_out_family,
                        "method": method,
                        "row_position": local_pos,
                        "crossbar_size": row["crossbar_size"],
                        "requested_weight_bits": row["requested_weight_bits"],
                        "adc_bits": row["adc_bits"],
                        "mapping_strategy": row.get("mapping_strategy", ""),
                        "cost_proxy": float(row[cost_col]),
                        "actual_accuracy": float(row["accuracy"]),
                        "predicted_accuracy": float(predicted[local_pos]),
                        "is_true_pareto": bool(true_mask[local_pos]),
                        "is_predicted_pareto": bool(pred_mask[local_pos]),
                    }
                )

            print(
                f"    {method:<18} "
                f"pred_pts={pred_count:>2}  "
                f"recall={recall:6.3f}  "
                f"precision={precision:6.3f}  "
                f"F1={f1:6.3f}  "
                f"front_dist={front_distance:7.4f}  "
                f"HV_err={hv_abs_error:7.4f}  "
                f"budget_regret={mean_budget_regret:7.3f} pp"
            )

        print()

    by_device = pd.DataFrame(metric_rows)
    points = pd.DataFrame(point_rows)

    summary = (
        by_device
        .groupby("method", as_index=False)
        .agg(
            devices=("held_out_device", "nunique"),
            mean_pareto_recall=("pareto_recall", "mean"),
            mean_pareto_precision=("pareto_precision", "mean"),
            mean_pareto_f1=("pareto_f1", "mean"),
            mean_normalized_front_distance=("normalized_front_distance", "mean"),
            mean_hypervolume_abs_error=("hypervolume_abs_error", "mean"),
            mean_cost_budget_regret_pp=("mean_cost_budget_regret_pp", "mean"),
            max_cost_budget_regret_pp=("max_cost_budget_regret_pp", "max"),
        )
    )

    OUT_BY_DEVICE.parent.mkdir(parents=True, exist_ok=True)
    by_device.to_csv(OUT_BY_DEVICE, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    points.to_csv(OUT_POINTS, index=False)

    scope = {
        "stage": 18,
        "title": "Zero-shot Pareto frontier transfer",
        "research_question": (
            "Can a model trained on other experimentally reported memristor studies "
            "reconstruct the accuracy-versus-relative-cost trade-off frontier of a "
            "held-out device before seeing its simulation results?"
        ),
        "rows": int(len(df)),
        "devices": int(df["device_id"].nunique()),
        "studies": int(df["study_id"].nunique()),
        "families": int(df["technology_family"].nunique()),
        "cost_axis": cost_col,
        "split_policy": (
            "For each held-out device, all rows from that device's source study "
            "are excluded from training."
        ),
        "models": {
            name: spec["description"]
            for name, spec in MODEL_SPECS.items()
        },
        "metrics": {
            "pareto_recall": (
                "Fraction of true Pareto configurations also present on the predicted frontier."
            ),
            "pareto_precision": (
                "Fraction of predicted Pareto configurations that are truly Pareto-optimal."
            ),
            "normalized_front_distance": (
                "Symmetric nearest-neighbor distance between predicted and true frontiers "
                "after normalizing cost and accuracy within each held-out device."
            ),
            "hypervolume_abs_error": (
                "Absolute difference in bounded normalized 2D hypervolume. Descriptive only."
            ),
            "cost_budget_regret": (
                "At several cost budgets, actual accuracy loss from choosing the configuration "
                "with highest predicted accuracy versus the true best configuration under "
                "the same budget."
            ),
        },
        "important_limits": [
            (
                "The cost axis is the project's relative hardware-cost proxy; it is not "
                "measured energy, area, latency, power, or monetary cost."
            ),
            (
                "Exact Pareto overlap is intentionally strict; nearby frontier points can "
                "still be useful even if exact configuration overlap is low."
            ),
            (
                "The study contains only 10 device profiles from 8 independent primary studies "
                "and 5 technology families, so conclusions are pilot-scale."
            ),
            (
                "This stage tests transfer of a simulated architecture trade-off surface, "
                "not universal physical generalization across all memristor technologies."
            ),
        ],
    }

    OUT_SCOPE.write_text(
        json.dumps(scope, indent=2),
        encoding="utf-8",
    )

    print("=" * 84)
    print("STAGE 18 SUMMARY")
    print("=" * 84)
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print()
    print(f"Saved: {OUT_BY_DEVICE}")
    print(f"Saved: {OUT_SUMMARY}")
    print(f"Saved: {OUT_POINTS}")
    print(f"Saved: {OUT_SCOPE}")


if __name__ == "__main__":
    main()
