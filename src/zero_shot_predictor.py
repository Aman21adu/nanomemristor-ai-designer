from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Files
# ============================================================

INPUT_FILE = Path("results/tables/ml_dataset.csv")
MANIFEST_FILE = Path("results/tables/ml_dataset_manifest.json")

PRIMARY_PREDICTIONS_FILE = Path("results/tables/zero_shot_predictions.csv")
PRIMARY_SUMMARY_FILE = Path("results/tables/zero_shot_summary.csv")

STUDY_PREDICTIONS_FILE = Path("results/tables/zero_shot_study_predictions.csv")
STUDY_SUMMARY_FILE = Path("results/tables/zero_shot_study_summary.csv")

FAMILY_PREDICTIONS_FILE = Path("results/tables/zero_shot_family_predictions.csv")
FAMILY_SUMMARY_FILE = Path("results/tables/zero_shot_family_summary.csv")

VALIDATION_OVERVIEW_FILE = Path(
    "results/tables/zero_shot_validation_overview.csv"
)

RELIABILITY_PENALTY_FILE = Path(
    "results/tables/zero_shot_reliability_penalties.csv"
)


# ============================================================
# Experiment settings
# ============================================================

ACCURACY_TOLERANCE_PP = 0.5
RANDOM_STATE = 42
UNCERTAINTY_MULTIPLIER = 1.0
TARGET = "accuracy"

PRIMARY_MODE = "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT"
STUDY_MODE = "LEAVE_ONE_STUDY_OUT"
FAMILY_MODE = "LEAVE_ONE_FAMILY_OUT"

RELIABILITY_SOURCE = "NESTED_TRAINING_ONLY_STUDY_BLOCKED"


# ============================================================
# Model features
# ============================================================

NUMERIC_FEATURES = [
    "device_log10_on_off_ratio",
    "state_count_available",
    "physical_state_count",
    "crossbar_size",
    "requested_weight_bits",
    "effective_weight_levels",
    "slices_per_branch",
    "physical_cells_per_weight",
    "adc_bits",
]

CATEGORICAL_FEATURES = [
    "device_conductance_mode",
    "mapping_strategy",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

NOT_MODEL_FEATURES = [
    "device_id",
    "study_id",
    "technology_family",
    "device_state_count_status",
    "parameter_source",
    "precision_basis",
    "accuracy_loss",
    "relative_hardware_cost_proxy",
]


# ============================================================
# Reliability-profile cache
#
# A training-only nested study-blocked validation profile is
# calculated for each unique outer-training set. The cache
# prevents the same expensive nested profile being rebuilt for
# equivalent primary/study folds.
# ============================================================

_RELIABILITY_CACHE: dict[tuple[str, ...], tuple[pd.DataFrame, float, int]] = {}


# ============================================================
# Generic helpers
# ============================================================

def configuration_key(row) -> tuple[int, int, int]:
    return (
        int(row["crossbar_size"]),
        int(row["requested_weight_bits"]),
        int(row["adc_bits"]),
    )


def joined_values(values) -> str:
    return ";".join(sorted(str(v) for v in values))


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def get_random_forest_uncertainty(
    trained_pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Random-Forest ensemble mean and tree-prediction standard deviation.

    Tree disagreement is an uncalibrated ensemble heuristic.
    It is NOT a statistical confidence interval.
    """
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    forest = trained_pipeline.named_steps["regressor"]

    X_transformed = preprocessor.transform(X)

    tree_predictions = np.column_stack(
        [tree.predict(X_transformed) for tree in forest.estimators_]
    )

    return (
        np.mean(tree_predictions, axis=1),
        np.std(tree_predictions, axis=1, ddof=1),
    )


def get_near_optimal_candidates(
    group: pd.DataFrame,
    score_column: str,
    tolerance_pp: float,
):
    best_score = float(group[score_column].max())
    threshold = best_score - tolerance_pp

    candidates = group[group[score_column] >= threshold].copy()

    if candidates.empty:
        raise ValueError(f"No near-optimal candidates for {score_column}.")

    return candidates, best_score, threshold


def sort_cost_aware(
    candidates: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    """
    Prefer lower design-complexity descriptors after satisfying
    the near-optimal score requirement.

    relative_hardware_cost_proxy remains heuristic; it is not
    measured energy, area, latency, power, or monetary cost.
    """
    return candidates.sort_values(
        by=[
            "estimated_memristor_cells",
            "relative_hardware_cost_proxy",
            "requested_weight_bits",
            "adc_bits",
            score_column,
            "crossbar_size",
        ],
        ascending=[True, True, True, True, False, False],
    )


def select_near_optimal_configuration(
    group: pd.DataFrame,
    score_column: str,
    tolerance_pp: float,
):
    candidates, best_score, threshold = get_near_optimal_candidates(
        group,
        score_column,
        tolerance_pp,
    )

    ranked = sort_cost_aware(candidates, score_column)

    return ranked.iloc[0], best_score, threshold, candidates


def select_confidence_aware_configuration(
    group: pd.DataFrame,
    tolerance_pp: float,
):
    predicted_best = float(group["predicted_accuracy"].max())
    acceptance_threshold = predicted_best - tolerance_pp

    safe_candidates = group[
        group["confidence_lower_bound"] >= acceptance_threshold
    ].copy()

    if safe_candidates.empty:
        chosen, _, _, _ = select_near_optimal_configuration(
            group,
            "predicted_accuracy",
            tolerance_pp,
        )
        fallback_used = True
    else:
        chosen = sort_cost_aware(
            safe_candidates,
            "confidence_lower_bound",
        ).iloc[0]
        fallback_used = False

    return (
        chosen,
        predicted_best,
        acceptance_threshold,
        safe_candidates,
        fallback_used,
    )


def select_validation_aware_configuration(
    group: pd.DataFrame,
    tolerance_pp: float,
):
    """
    New recommendation policy.

    score =
        predicted_accuracy
        - nested training-only historical prediction-error penalty

    We first require a candidate to be within tolerance_pp of the
    BEST validation-adjusted score, then apply the same cost-aware
    sorting philosophy used by the original recommender.

    This is an empirical reliability heuristic, not a calibrated
    confidence interval.
    """
    return select_near_optimal_configuration(
        group,
        "validation_adjusted_score",
        tolerance_pp,
    )


def select_accuracy_first_configuration(
    group: pd.DataFrame,
):
    """
    Conservative fallback for descriptor extrapolation.

    Select the highest predicted-accuracy configuration first.
    Hardware-cost proxy is used only to break prediction ties.
    """
    return group.sort_values(
        by=[
            "predicted_accuracy",
            "estimated_memristor_cells",
            "relative_hardware_cost_proxy",
            "requested_weight_bits",
            "adc_bits",
            "crossbar_size",
        ],
        ascending=[False, True, True, True, True, False],
    ).iloc[0]


def assess_same_mode_descriptor_support(
    test_row: pd.Series,
    train_df: pd.DataFrame,
):
    """
    Decide whether aggressive cost optimization is supported by the
    OUTER training data for this device descriptor.

    Rule:
      - at least two training device profiles must share the same
        conductance mode, AND
      - the held-out device log10(ON/OFF) must lie inside the
        same-mode training envelope.

    If not, the recommendation falls back to accuracy-first selection.

    This is a transparent extrapolation guard, not a calibrated OOD score.
    """
    mode = str(test_row["device_conductance_mode"])
    test_ratio = float(test_row["device_log10_on_off_ratio"])

    descriptors = (
        train_df[
            [
                "device_id",
                "device_conductance_mode",
                "device_log10_on_off_ratio",
            ]
        ]
        .drop_duplicates("device_id")
        .copy()
    )

    same_mode = descriptors[
        descriptors["device_conductance_mode"].astype(str) == mode
    ].copy()

    same_mode_count = int(len(same_mode))

    if same_mode.empty:
        return {
            "same_mode_training_devices": 0,
            "same_mode_log10_ratio_min": np.nan,
            "same_mode_log10_ratio_max": np.nan,
            "inside_same_mode_ratio_envelope": False,
            "cost_optimization_supported": False,
            "support_reason": "NO_SAME_MODE_TRAINING_DEVICE",
        }

    ratio_min = float(same_mode["device_log10_on_off_ratio"].min())
    ratio_max = float(same_mode["device_log10_on_off_ratio"].max())

    inside = bool(
        (test_ratio >= ratio_min - 1e-12)
        and
        (test_ratio <= ratio_max + 1e-12)
    )

    enough_same_mode = same_mode_count >= 2
    supported = bool(enough_same_mode and inside)

    if not enough_same_mode:
        reason = "INSUFFICIENT_SAME_MODE_DEVICE_COVERAGE"
    elif not inside:
        reason = "SAME_MODE_ON_OFF_RATIO_EXTRAPOLATION"
    else:
        reason = "SAME_MODE_INTERPOLATION_SUPPORTED"

    return {
        "same_mode_training_devices": same_mode_count,
        "same_mode_log10_ratio_min": ratio_min,
        "same_mode_log10_ratio_max": ratio_max,
        "inside_same_mode_ratio_envelope": inside,
        "cost_optimization_supported": supported,
        "support_reason": reason,
    }


def get_raw_best_configuration(group: pd.DataFrame) -> pd.Series:
    return group.sort_values(
        by=[
            "accuracy",
            "estimated_memristor_cells",
            "relative_hardware_cost_proxy",
            "requested_weight_bits",
            "adc_bits",
            "crossbar_size",
        ],
        ascending=[False, True, True, True, True, False],
    ).iloc[0]


def same_configuration(row_a, row_b) -> bool:
    return configuration_key(row_a) == configuration_key(row_b)


def get_cost_aware_rank(
    true_candidates: pd.DataFrame,
    recommended,
):
    ranked = sort_cost_aware(
        true_candidates,
        "accuracy",
    ).reset_index(drop=True)

    target_key = configuration_key(recommended)

    for index, row in ranked.iterrows():
        if configuration_key(row) == target_key:
            return int(index + 1)

    return np.nan


def candidate_overlap_metrics(
    true_candidates: pd.DataFrame,
    predicted_candidates: pd.DataFrame,
):
    true_keys = {
        configuration_key(row)
        for _, row in true_candidates.iterrows()
    }

    predicted_keys = {
        configuration_key(row)
        for _, row in predicted_candidates.iterrows()
    }

    overlap = true_keys.intersection(predicted_keys)

    precision = (
        len(overlap) / len(predicted_keys)
        if predicted_keys
        else 0.0
    )

    recall = (
        len(overlap) / len(true_keys)
        if true_keys
        else 0.0
    )

    return float(precision), float(recall), int(len(overlap))


# ============================================================
# Validation-aware reliability penalty
# ============================================================

def build_training_only_reliability_profile(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, int]:
    """
    Build a reliability penalty using ONLY the current OUTER training set.

    Inside the outer training set, each source study is held out once.
    The resulting out-of-study prediction errors are aggregated by
    (requested_weight_bits, adc_bits).

    Therefore the outer held-out device/study is never used to construct
    its own reliability penalty.

    Returns:
        penalty_table
        fallback_global_mean_abs_error_pp
        nested_prediction_rows
    """
    cache_key = tuple(sorted(train_df["device_id"].astype(str).unique()))

    if cache_key in _RELIABILITY_CACHE:
        table, fallback, rows = _RELIABILITY_CACHE[cache_key]
        return table.copy(), fallback, rows

    parts = []

    for held_out_study in sorted(train_df["study_id"].unique()):
        inner_test = train_df[
            train_df["study_id"] == held_out_study
        ].copy()

        inner_train = train_df[
            train_df["study_id"] != held_out_study
        ].copy()

        if inner_train.empty or inner_test.empty:
            continue

        model = build_model()
        model.fit(
            inner_train[ALL_FEATURES],
            inner_train[TARGET],
        )

        inner_pred, _ = get_random_forest_uncertainty(
            model,
            inner_test[ALL_FEATURES],
        )

        part = inner_test[
            [
                "requested_weight_bits",
                "adc_bits",
                "accuracy",
            ]
        ].copy()

        part["nested_predicted_accuracy"] = inner_pred
        part["nested_absolute_prediction_error_pp"] = (
            part["nested_predicted_accuracy"]
            - part["accuracy"]
        ).abs()

        parts.append(part)

    if not parts:
        raise ValueError(
            "Unable to build nested training-only reliability profile."
        )

    nested = pd.concat(parts, ignore_index=True)

    fallback = float(
        nested["nested_absolute_prediction_error_pp"].mean()
    )

    table = (
        nested.groupby(
            ["requested_weight_bits", "adc_bits"],
            as_index=False,
        )
        .agg(
            validation_reliability_penalty_pp=(
                "nested_absolute_prediction_error_pp",
                "mean",
            ),
            validation_reliability_median_error_pp=(
                "nested_absolute_prediction_error_pp",
                "median",
            ),
            validation_reliability_max_error_pp=(
                "nested_absolute_prediction_error_pp",
                "max",
            ),
            validation_reliability_rows=(
                "nested_absolute_prediction_error_pp",
                "size",
            ),
        )
    )

    _RELIABILITY_CACHE[cache_key] = (
        table.copy(),
        fallback,
        int(len(nested)),
    )

    return table, fallback, int(len(nested))


def apply_validation_aware_scores(
    predicted_df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    penalty_table, fallback, nested_rows = (
        build_training_only_reliability_profile(train_df)
    )

    out = predicted_df.merge(
        penalty_table,
        on=["requested_weight_bits", "adc_bits"],
        how="left",
        validate="many_to_one",
    )

    out["validation_reliability_penalty_pp"] = (
        out["validation_reliability_penalty_pp"].fillna(fallback)
    )

    out["validation_reliability_median_error_pp"] = (
        out["validation_reliability_median_error_pp"].fillna(fallback)
    )

    out["validation_reliability_max_error_pp"] = (
        out["validation_reliability_max_error_pp"].fillna(fallback)
    )

    out["validation_reliability_rows"] = (
        out["validation_reliability_rows"].fillna(0).astype(int)
    )

    out["validation_adjusted_score"] = (
        out["predicted_accuracy"]
        - out["validation_reliability_penalty_pp"]
    )

    out["validation_reliability_source"] = RELIABILITY_SOURCE

    return out, nested_rows


def build_deployment_reliability_table(
    primary_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full-dataset deployment penalty table from PRIMARY study-blocked
    predictions. Every error contributing here came from a model that did not
    train on the corresponding source study.

    This table is suitable for later Custom Device / website recommendation
    heuristics. It is still empirical and uncalibrated.
    """
    table = (
        primary_predictions.groupby(
            ["requested_weight_bits", "adc_bits"],
            as_index=False,
        )
        .agg(
            historical_mean_abs_error_pp=(
                "absolute_prediction_error_pp",
                "mean",
            ),
            historical_median_abs_error_pp=(
                "absolute_prediction_error_pp",
                "median",
            ),
            historical_max_abs_error_pp=(
                "absolute_prediction_error_pp",
                "max",
            ),
            historical_validation_rows=(
                "absolute_prediction_error_pp",
                "size",
            ),
        )
        .sort_values(
            ["requested_weight_bits", "adc_bits"]
        )
        .reset_index(drop=True)
    )

    table["penalty_source"] = (
        "PRIMARY_STUDY_BLOCKED_HELD_OUT_PREDICTIONS"
    )

    table["calibrated_confidence"] = False

    return table


# ============================================================
# Evaluate one held-out physical device
# ============================================================

def evaluate_device_predictions(
    device_test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_mode: str,
    held_out_group: str,
    held_out_group_rows: int,
    excluded_device_ids: list[str],
    nested_reliability_rows: int,
):
    test_df = device_test_df.copy()

    held_out_device = str(test_df["device_id"].iloc[0])
    held_out_study = str(test_df["study_id"].iloc[0])
    held_out_family = str(test_df["technology_family"].iloc[0])

    # --------------------------------------------------------
    # Prediction quality
    # --------------------------------------------------------

    global_mae = mean_absolute_error(
        test_df["accuracy"],
        test_df["predicted_accuracy"],
    )

    global_rmse = np.sqrt(
        mean_squared_error(
            test_df["accuracy"],
            test_df["predicted_accuracy"],
        )
    )

    global_r2 = r2_score(
        test_df["accuracy"],
        test_df["predicted_accuracy"],
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    raw_best = get_raw_best_configuration(test_df)
    true_best_accuracy = float(raw_best["accuracy"])

    (
        true_cost_aware,
        _,
        true_threshold,
        true_candidates,
    ) = select_near_optimal_configuration(
        test_df,
        "accuracy",
        ACCURACY_TOLERANCE_PP,
    )

    test_df["is_true_near_optimal"] = (
        test_df["accuracy"] >= true_threshold
    )

    near_region = test_df[test_df["is_true_near_optimal"]].copy()

    near_region_mae = mean_absolute_error(
        near_region["accuracy"],
        near_region["predicted_accuracy"],
    )

    near_region_rmse = np.sqrt(
        mean_squared_error(
            near_region["accuracy"],
            near_region["predicted_accuracy"],
        )
    )

    # --------------------------------------------------------
    # Original baseline recommendation
    # --------------------------------------------------------

    (
        baseline_recommended,
        predicted_best_accuracy,
        predicted_threshold,
        predicted_candidates,
    ) = select_near_optimal_configuration(
        test_df,
        "predicted_accuracy",
        ACCURACY_TOLERANCE_PP,
    )

    (
        candidate_precision,
        candidate_recall,
        candidate_overlap,
    ) = candidate_overlap_metrics(
        true_candidates,
        predicted_candidates,
    )

    # --------------------------------------------------------
    # Tree-disagreement confidence-aware recommendation
    # --------------------------------------------------------

    (
        confidence_recommended,
        _,
        confidence_acceptance_threshold,
        safe_candidates,
        confidence_fallback_used,
    ) = select_confidence_aware_configuration(
        test_df,
        ACCURACY_TOLERANCE_PP,
    )

    (
        safe_precision,
        safe_recall,
        safe_overlap,
    ) = candidate_overlap_metrics(
        true_candidates,
        safe_candidates,
    )

    # --------------------------------------------------------
    # NEW: validation-aware recommendation
    # --------------------------------------------------------

    (
        validation_recommended,
        validation_best_score,
        validation_threshold,
        validation_candidates,
    ) = select_validation_aware_configuration(
        test_df,
        ACCURACY_TOLERANCE_PP,
    )

    (
        validation_candidate_precision,
        validation_candidate_recall,
        validation_candidate_overlap,
    ) = candidate_overlap_metrics(
        true_candidates,
        validation_candidates,
    )

    # --------------------------------------------------------
    # NEW: support-gated recommendation
    #
    # Cost optimization is allowed only when the held-out device lies
    # inside the same-conductance-mode ON/OFF envelope represented by
    # at least two OUTER training device profiles.
    #
    # Otherwise use the highest predicted-accuracy configuration.
    # --------------------------------------------------------

    support = assess_same_mode_descriptor_support(
        test_df.iloc[0],
        train_df,
    )

    if support["cost_optimization_supported"]:
        guarded_recommended = validation_recommended
        guarded_policy = "VALIDATION_AWARE_COST"
    else:
        guarded_recommended = select_accuracy_first_configuration(
            test_df
        )
        guarded_policy = "ACCURACY_FIRST_EXTRAPOLATION_FALLBACK"

    # --------------------------------------------------------
    # Selection metrics helper
    # --------------------------------------------------------

    def metrics_for_selection(row):
        actual = float(row["accuracy"])
        predicted = float(row["predicted_accuracy"])
        uncertainty = float(row["prediction_uncertainty"])
        lower = float(row["confidence_lower_bound"])
        regret = true_best_accuracy - actual
        success = bool(regret <= ACCURACY_TOLERANCE_PP)
        exact = bool(same_configuration(row, true_cost_aware))
        cost_rank = get_cost_aware_rank(true_candidates, row)

        return {
            "actual": actual,
            "predicted": predicted,
            "uncertainty": uncertainty,
            "lower": lower,
            "regret": float(regret),
            "success": success,
            "exact": exact,
            "cost_rank": cost_rank,
            "top3": bool(np.isfinite(cost_rank) and cost_rank <= 3),
            "top5": bool(np.isfinite(cost_rank) and cost_rank <= 5),
        }

    baseline = metrics_for_selection(baseline_recommended)
    confidence = metrics_for_selection(confidence_recommended)
    validation = metrics_for_selection(validation_recommended)
    guarded = metrics_for_selection(guarded_recommended)

    # --------------------------------------------------------
    # Uncertainty diagnostic
    # --------------------------------------------------------

    uncertainty_values = test_df[
        "prediction_uncertainty"
    ].to_numpy()

    absolute_errors = test_df[
        "absolute_prediction_error_pp"
    ].to_numpy()

    if (
        np.std(uncertainty_values) > 0
        and np.std(absolute_errors) > 0
    ):
        uncertainty_error_correlation = float(
            np.corrcoef(
                uncertainty_values,
                absolute_errors,
            )[0, 1]
        )
    else:
        uncertainty_error_correlation = np.nan

    # --------------------------------------------------------
    # Training metadata / leakage checks
    # --------------------------------------------------------

    training_device_ids = sorted(train_df["device_id"].unique())
    training_study_ids = sorted(train_df["study_id"].unique())
    training_family_ids = sorted(train_df["technology_family"].unique())

    same_study_training_leakage = (
        held_out_study in set(training_study_ids)
    )

    same_family_present_in_training = (
        held_out_family in set(training_family_ids)
    )

    if same_study_training_leakage:
        raise ValueError(
            f"Study leakage detected for {held_out_device}: "
            f"{held_out_study} is present in training."
        )

    search_reduction_pct = (
        1.0 - (1.0 / len(test_df))
    ) * 100.0

    # --------------------------------------------------------
    # Prediction-row metadata
    # --------------------------------------------------------

    test_df["validation_mode"] = validation_mode
    test_df["held_out_group"] = str(held_out_group)
    test_df["held_out_device"] = held_out_device
    test_df["held_out_study"] = held_out_study
    test_df["held_out_family"] = held_out_family
    test_df["training_devices"] = len(training_device_ids)
    test_df["training_studies"] = len(training_study_ids)
    test_df["training_families"] = len(training_family_ids)
    test_df["same_study_training_leakage"] = False

    # --------------------------------------------------------
    # Summary row
    # --------------------------------------------------------

    summary_row = {
        "validation_mode": validation_mode,
        "held_out_group": str(held_out_group),
        "held_out_device": held_out_device,
        "held_out_study": held_out_study,
        "held_out_family": held_out_family,
        "training_devices": int(len(training_device_ids)),
        "training_studies": int(len(training_study_ids)),
        "training_families": int(len(training_family_ids)),
        "training_device_ids": joined_values(training_device_ids),
        "training_study_ids": joined_values(training_study_ids),
        "training_family_ids": joined_values(training_family_ids),
        "excluded_from_training_device_ids": joined_values(
            excluded_device_ids
        ),
        "same_study_training_leakage": False,
        "same_family_present_in_training": same_family_present_in_training,
        "training_rows": int(len(train_df)),
        "held_out_rows": int(len(test_df)),
        "held_out_group_rows": int(held_out_group_rows),

        "global_mae_pp": float(global_mae),
        "global_rmse_pp": float(global_rmse),
        "global_r2": float(global_r2),
        "near_region_mae_pp": float(near_region_mae),
        "near_region_rmse_pp": float(near_region_rmse),

        "mean_prediction_uncertainty_pp": float(
            test_df["prediction_uncertainty"].mean()
        ),
        "uncertainty_error_correlation": (
            float(uncertainty_error_correlation)
            if np.isfinite(uncertainty_error_correlation)
            else np.nan
        ),

        "safe_candidate_count": int(len(safe_candidates)),
        "safe_candidate_precision": float(safe_precision),
        "safe_candidate_recall": float(safe_recall),
        "safe_candidate_overlap": int(safe_overlap),

        "predicted_candidate_count": int(len(predicted_candidates)),
        "true_near_optimal_candidate_count": int(len(true_candidates)),
        "candidate_precision": float(candidate_precision),
        "candidate_recall": float(candidate_recall),
        "candidate_overlap": int(candidate_overlap),

        "validation_candidate_count": int(len(validation_candidates)),
        "validation_candidate_precision": float(
            validation_candidate_precision
        ),
        "validation_candidate_recall": float(
            validation_candidate_recall
        ),
        "validation_candidate_overlap": int(
            validation_candidate_overlap
        ),

        "true_raw_best_accuracy": true_best_accuracy,
        "true_cost_aware_crossbar": int(
            true_cost_aware["crossbar_size"]
        ),
        "true_cost_aware_weight_bits": int(
            true_cost_aware["requested_weight_bits"]
        ),
        "true_cost_aware_adc_bits": int(
            true_cost_aware["adc_bits"]
        ),
        "true_cost_aware_accuracy": float(
            true_cost_aware["accuracy"]
        ),
        "true_near_optimal_threshold": float(true_threshold),

        # Original baseline
        "baseline_crossbar": int(
            baseline_recommended["crossbar_size"]
        ),
        "baseline_weight_bits": int(
            baseline_recommended["requested_weight_bits"]
        ),
        "baseline_adc_bits": int(
            baseline_recommended["adc_bits"]
        ),
        "baseline_predicted_accuracy": baseline["predicted"],
        "baseline_uncertainty_pp": baseline["uncertainty"],
        "baseline_lower_bound": baseline["lower"],
        "baseline_passes_confidence_gate": bool(
            baseline["lower"] >= confidence_acceptance_threshold
        ),
        "baseline_actual_accuracy": baseline["actual"],
        "baseline_regret_pp": baseline["regret"],
        "baseline_near_optimal_success": baseline["success"],
        "baseline_exact_match": baseline["exact"],
        "baseline_cost_aware_rank": (
            int(baseline["cost_rank"])
            if np.isfinite(baseline["cost_rank"])
            else np.nan
        ),
        "baseline_cost_aware_top3": baseline["top3"],
        "baseline_cost_aware_top5": baseline["top5"],
        "predicted_best_accuracy": float(predicted_best_accuracy),
        "predicted_near_optimal_threshold": float(predicted_threshold),

        # Tree-disagreement confidence-aware
        "confidence_crossbar": int(
            confidence_recommended["crossbar_size"]
        ),
        "confidence_weight_bits": int(
            confidence_recommended["requested_weight_bits"]
        ),
        "confidence_adc_bits": int(
            confidence_recommended["adc_bits"]
        ),
        "confidence_predicted_accuracy": confidence["predicted"],
        "confidence_uncertainty_pp": confidence["uncertainty"],
        "confidence_lower_bound": confidence["lower"],
        "confidence_fallback_used": confidence_fallback_used,
        "confidence_actual_accuracy": confidence["actual"],
        "confidence_regret_pp": confidence["regret"],
        "confidence_near_optimal_success": confidence["success"],
        "confidence_exact_match": confidence["exact"],
        "confidence_cost_aware_rank": (
            int(confidence["cost_rank"])
            if np.isfinite(confidence["cost_rank"])
            else np.nan
        ),
        "confidence_cost_aware_top3": confidence["top3"],
        "confidence_cost_aware_top5": confidence["top5"],
        "confidence_acceptance_threshold": float(
            confidence_acceptance_threshold
        ),

        # NEW validation-aware
        "validation_aware_crossbar": int(
            validation_recommended["crossbar_size"]
        ),
        "validation_aware_weight_bits": int(
            validation_recommended["requested_weight_bits"]
        ),
        "validation_aware_adc_bits": int(
            validation_recommended["adc_bits"]
        ),
        "validation_aware_predicted_accuracy": validation["predicted"],
        "validation_aware_reliability_penalty_pp": float(
            validation_recommended[
                "validation_reliability_penalty_pp"
            ]
        ),
        "validation_aware_adjusted_score": float(
            validation_recommended["validation_adjusted_score"]
        ),
        "validation_aware_uncertainty_pp": validation["uncertainty"],
        "validation_aware_actual_accuracy": validation["actual"],
        "validation_aware_regret_pp": validation["regret"],
        "validation_aware_near_optimal_success": validation["success"],
        "validation_aware_exact_match": validation["exact"],
        "validation_aware_cost_aware_rank": (
            int(validation["cost_rank"])
            if np.isfinite(validation["cost_rank"])
            else np.nan
        ),
        "validation_aware_cost_aware_top3": validation["top3"],
        "validation_aware_cost_aware_top5": validation["top5"],
        "validation_adjusted_best_score": float(
            validation_best_score
        ),
        "validation_adjusted_threshold": float(
            validation_threshold
        ),
        "validation_reliability_source": RELIABILITY_SOURCE,
        "nested_reliability_prediction_rows": int(
            nested_reliability_rows
        ),

        # NEW support-gated final recommendation
        "guarded_policy": guarded_policy,
        "guarded_support_reason": support["support_reason"],
        "guarded_same_mode_training_devices": int(
            support["same_mode_training_devices"]
        ),
        "guarded_same_mode_log10_ratio_min": support[
            "same_mode_log10_ratio_min"
        ],
        "guarded_same_mode_log10_ratio_max": support[
            "same_mode_log10_ratio_max"
        ],
        "guarded_inside_same_mode_ratio_envelope": bool(
            support["inside_same_mode_ratio_envelope"]
        ),
        "guarded_cost_optimization_supported": bool(
            support["cost_optimization_supported"]
        ),
        "guarded_crossbar": int(
            guarded_recommended["crossbar_size"]
        ),
        "guarded_weight_bits": int(
            guarded_recommended["requested_weight_bits"]
        ),
        "guarded_adc_bits": int(
            guarded_recommended["adc_bits"]
        ),
        "guarded_predicted_accuracy": guarded["predicted"],
        "guarded_actual_accuracy": guarded["actual"],
        "guarded_regret_pp": guarded["regret"],
        "guarded_near_optimal_success": guarded["success"],
        "guarded_exact_match": guarded["exact"],
        "guarded_cost_aware_rank": (
            int(guarded["cost_rank"])
            if np.isfinite(guarded["cost_rank"])
            else np.nan
        ),
        "guarded_cost_aware_top3": guarded["top3"],
        "guarded_cost_aware_top5": guarded["top5"],

        "search_reduction_pct": float(search_reduction_pct),
    }

    return test_df, summary_row


# ============================================================
# Fit one outer model and evaluate held-out physical devices
# ============================================================

def fit_predict_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    validation_mode: str,
    held_out_group: str,
    excluded_device_ids: list[str],
):
    if train_df.empty:
        raise ValueError(
            f"{validation_mode}: training set is empty for "
            f"held-out group {held_out_group}."
        )

    if test_df.empty:
        raise ValueError(
            f"{validation_mode}: test set is empty for "
            f"held-out group {held_out_group}."
        )

    train_studies = set(train_df["study_id"].unique())
    test_studies = set(test_df["study_id"].unique())

    study_overlap = train_studies.intersection(test_studies)

    if study_overlap:
        raise ValueError(
            f"{validation_mode}: study leakage for "
            f"{held_out_group}: {sorted(study_overlap)}"
        )

    if validation_mode == FAMILY_MODE:
        family_overlap = set(
            train_df["technology_family"].unique()
        ).intersection(
            set(test_df["technology_family"].unique())
        )

        if family_overlap:
            raise ValueError(
                f"{validation_mode}: family leakage for "
                f"{held_out_group}: {sorted(family_overlap)}"
            )

    model = build_model()

    model.fit(
        train_df[ALL_FEATURES],
        train_df[TARGET],
    )

    predicted_accuracy, prediction_uncertainty = (
        get_random_forest_uncertainty(
            model,
            test_df[ALL_FEATURES],
        )
    )

    predicted_df = test_df.copy()
    predicted_df["predicted_accuracy"] = predicted_accuracy
    predicted_df["prediction_uncertainty"] = prediction_uncertainty
    predicted_df["confidence_lower_bound"] = (
        predicted_df["predicted_accuracy"]
        - UNCERTAINTY_MULTIPLIER
        * predicted_df["prediction_uncertainty"]
    )
    predicted_df["prediction_error_pp"] = (
        predicted_df["predicted_accuracy"]
        - predicted_df["accuracy"]
    )
    predicted_df["absolute_prediction_error_pp"] = (
        predicted_df["prediction_error_pp"].abs()
    )

    # NEW: build reliability penalties strictly from the OUTER training set.
    predicted_df, nested_reliability_rows = (
        apply_validation_aware_scores(
            predicted_df,
            train_df,
        )
    )

    prediction_parts = []
    summary_rows = []
    held_out_group_rows = int(len(predicted_df))

    for held_out_device in sorted(
        predicted_df["device_id"].unique()
    ):
        device_test_df = predicted_df[
            predicted_df["device_id"] == held_out_device
        ].copy()

        evaluated_predictions, summary_row = (
            evaluate_device_predictions(
                device_test_df=device_test_df,
                train_df=train_df,
                validation_mode=validation_mode,
                held_out_group=held_out_group,
                held_out_group_rows=held_out_group_rows,
                excluded_device_ids=excluded_device_ids,
                nested_reliability_rows=nested_reliability_rows,
            )
        )

        prediction_parts.append(evaluated_predictions)
        summary_rows.append(summary_row)

    return (
        pd.concat(prediction_parts, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


# ============================================================
# Validation runners
# ============================================================

def run_primary_validation(df: pd.DataFrame):
    all_predictions = []
    all_summaries = []

    for held_out_device in sorted(df["device_id"].unique()):
        device_rows = df[
            df["device_id"] == held_out_device
        ].copy()

        held_out_studies = device_rows["study_id"].unique()

        if len(held_out_studies) != 1:
            raise ValueError(
                f"{held_out_device} maps to multiple study IDs."
            )

        held_out_study = str(held_out_studies[0])

        test_df = device_rows
        train_df = df[
            df["study_id"] != held_out_study
        ].copy()

        excluded_device_ids = sorted(
            df.loc[
                df["study_id"] == held_out_study,
                "device_id",
            ].unique()
        )

        print()
        print("=" * 48)
        print("PRIMARY HELD-OUT DEVICE:", held_out_device)
        print("HELD-OUT STUDY:", held_out_study)
        print(
            "EXCLUDED SAME-STUDY DEVICES:",
            excluded_device_ids,
        )
        print(
            "TRAINING DEVICES:",
            sorted(train_df["device_id"].unique()),
        )
        print("=" * 48)

        predictions, summary = fit_predict_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            validation_mode=PRIMARY_MODE,
            held_out_group=held_out_device,
            excluded_device_ids=excluded_device_ids,
        )

        all_predictions.append(predictions)
        all_summaries.append(summary)

    return (
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
    )


def run_study_validation(df: pd.DataFrame):
    all_predictions = []
    all_summaries = []

    for held_out_study in sorted(df["study_id"].unique()):
        test_df = df[
            df["study_id"] == held_out_study
        ].copy()

        train_df = df[
            df["study_id"] != held_out_study
        ].copy()

        excluded_device_ids = sorted(
            test_df["device_id"].unique()
        )

        print()
        print("=" * 48)
        print("LEAVE-ONE-STUDY-OUT:", held_out_study)
        print("TEST DEVICES:", excluded_device_ids)
        print("=" * 48)

        predictions, summary = fit_predict_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            validation_mode=STUDY_MODE,
            held_out_group=held_out_study,
            excluded_device_ids=excluded_device_ids,
        )

        all_predictions.append(predictions)
        all_summaries.append(summary)

    return (
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
    )


def run_family_validation(df: pd.DataFrame):
    all_predictions = []
    all_summaries = []

    for held_out_family in sorted(
        df["technology_family"].unique()
    ):
        test_df = df[
            df["technology_family"] == held_out_family
        ].copy()

        held_out_studies = set(
            test_df["study_id"].unique()
        )

        train_df = df[
            (
                df["technology_family"] != held_out_family
            )
            &
            (
                ~df["study_id"].isin(held_out_studies)
            )
        ].copy()

        excluded_device_ids = sorted(
            df.loc[
                ~df.index.isin(train_df.index),
                "device_id",
            ].unique()
        )

        print()
        print("=" * 48)
        print("LEAVE-ONE-FAMILY-OUT:", held_out_family)
        print(
            "TEST DEVICES:",
            sorted(test_df["device_id"].unique()),
        )
        print(
            "HELD-OUT STUDIES:",
            sorted(held_out_studies),
        )
        print("=" * 48)

        predictions, summary = fit_predict_and_evaluate(
            train_df=train_df,
            test_df=test_df,
            validation_mode=FAMILY_MODE,
            held_out_group=held_out_family,
            excluded_device_ids=excluded_device_ids,
        )

        all_predictions.append(predictions)
        all_summaries.append(summary)

    return (
        pd.concat(all_predictions, ignore_index=True),
        pd.concat(all_summaries, ignore_index=True),
    )


# ============================================================
# Aggregate / display
# ============================================================

def summarize_validation_mode(
    validation_mode: str,
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
):
    overall_near = predictions_df[
        predictions_df["is_true_near_optimal"]
    ].copy()

    near_mae = mean_absolute_error(
        overall_near["accuracy"],
        overall_near["predicted_accuracy"],
    )

    near_rmse = np.sqrt(
        mean_squared_error(
            overall_near["accuracy"],
            overall_near["predicted_accuracy"],
        )
    )

    uncertainty = predictions_df[
        "prediction_uncertainty"
    ].to_numpy()

    abs_error = predictions_df[
        "absolute_prediction_error_pp"
    ].to_numpy()

    if np.std(uncertainty) > 0 and np.std(abs_error) > 0:
        uncertainty_error_corr = float(
            np.corrcoef(
                uncertainty,
                abs_error,
            )[0, 1]
        )
    else:
        uncertainty_error_corr = np.nan

    return {
        "validation_mode": validation_mode,
        "evaluated_devices": int(
            summary_df["held_out_device"].nunique()
        ),
        "summary_rows": int(len(summary_df)),

        "baseline_mean_regret_pp": float(
            summary_df["baseline_regret_pp"].mean()
        ),
        "baseline_near_optimal_success_pct": float(
            100.0
            * summary_df[
                "baseline_near_optimal_success"
            ].mean()
        ),

        "confidence_mean_regret_pp": float(
            summary_df["confidence_regret_pp"].mean()
        ),
        "confidence_near_optimal_success_pct": float(
            100.0
            * summary_df[
                "confidence_near_optimal_success"
            ].mean()
        ),

        "validation_aware_mean_regret_pp": float(
            summary_df[
                "validation_aware_regret_pp"
            ].mean()
        ),
        "validation_aware_near_optimal_success_pct": float(
            100.0
            * summary_df[
                "validation_aware_near_optimal_success"
            ].mean()
        ),
        "validation_aware_exact_match_pct": float(
            100.0
            * summary_df[
                "validation_aware_exact_match"
            ].mean()
        ),
        "validation_aware_top3_pct": float(
            100.0
            * summary_df[
                "validation_aware_cost_aware_top3"
            ].mean()
        ),

        "guarded_mean_regret_pp": float(
            summary_df["guarded_regret_pp"].mean()
        ),
        "guarded_near_optimal_success_pct": float(
            100.0
            * summary_df[
                "guarded_near_optimal_success"
            ].mean()
        ),
        "guarded_exact_match_pct": float(
            100.0
            * summary_df["guarded_exact_match"].mean()
        ),
        "guarded_accuracy_first_fallback_pct": float(
            100.0
            * (
                summary_df["guarded_policy"]
                == "ACCURACY_FIRST_EXTRAPOLATION_FALLBACK"
            ).mean()
        ),

        "near_region_mae_pp": float(near_mae),
        "near_region_rmse_pp": float(near_rmse),
        "uncertainty_error_correlation": (
            float(uncertainty_error_corr)
            if np.isfinite(uncertainty_error_corr)
            else np.nan
        ),
    }


def print_mode_summary(
    title: str,
    summary_df: pd.DataFrame,
    overview_row: dict,
):
    print()
    print()
    print("=" * 48)
    print(title)
    print("=" * 48)

    display_columns = [
        "held_out_device",
        "held_out_study",
        "held_out_family",
        "training_devices",
        "training_studies",
        "training_families",

        "baseline_crossbar",
        "baseline_weight_bits",
        "baseline_adc_bits",
        "baseline_regret_pp",
        "baseline_near_optimal_success",

        "confidence_crossbar",
        "confidence_weight_bits",
        "confidence_adc_bits",
        "confidence_regret_pp",
        "confidence_near_optimal_success",

        "validation_aware_crossbar",
        "validation_aware_weight_bits",
        "validation_aware_adc_bits",
        "validation_aware_reliability_penalty_pp",
        "validation_aware_regret_pp",
        "validation_aware_near_optimal_success",

        "guarded_policy",
        "guarded_crossbar",
        "guarded_weight_bits",
        "guarded_adc_bits",
        "guarded_regret_pp",
        "guarded_near_optimal_success",
    ]

    print(
        summary_df[display_columns].to_string(index=False)
    )

    print()
    print(
        "Baseline mean regret: "
        f"{overview_row['baseline_mean_regret_pp']:.3f} pp"
    )
    print(
        "Baseline near-optimal success: "
        f"{overview_row['baseline_near_optimal_success_pct']:.1f}%"
    )

    print(
        "Tree-disagreement mean regret: "
        f"{overview_row['confidence_mean_regret_pp']:.3f} pp"
    )
    print(
        "Tree-disagreement near-optimal success: "
        f"{overview_row['confidence_near_optimal_success_pct']:.1f}%"
    )

    print(
        "Validation-aware mean regret: "
        f"{overview_row['validation_aware_mean_regret_pp']:.3f} pp"
    )
    print(
        "Validation-aware near-optimal success: "
        f"{overview_row['validation_aware_near_optimal_success_pct']:.1f}%"
    )

    print(
        "Support-gated mean regret: "
        f"{overview_row['guarded_mean_regret_pp']:.3f} pp"
    )
    print(
        "Support-gated near-optimal success: "
        f"{overview_row['guarded_near_optimal_success_pct']:.1f}%"
    )
    print(
        "Support-gated accuracy-first fallback rate: "
        f"{overview_row['guarded_accuracy_first_fallback_pct']:.1f}%"
    )

    print(
        "Near-region MAE: "
        f"{overview_row['near_region_mae_pp']:.3f} pp"
    )
    print(
        "Near-region RMSE: "
        f"{overview_row['near_region_rmse_pp']:.3f} pp"
    )

    corr = overview_row["uncertainty_error_correlation"]

    if np.isfinite(corr):
        print(
            "Tree-disagreement/error correlation: "
            f"{corr:.3f}"
        )
    else:
        print(
            "Tree-disagreement/error correlation: N/A"
        )


# ============================================================
# Input validation
# ============================================================

def load_and_validate_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing dataset: {INPUT_FILE}"
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest = json.load(f)

    split_rule = str(
        manifest.get("split_rule", "")
    ).upper()

    if "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT" not in split_rule:
        raise ValueError(
            "Manifest must specify "
            "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT."
        )

    if "NEVER_RANDOM_ROW_SPLIT" not in split_rule:
        raise ValueError(
            "Manifest must explicitly prohibit random row-level splitting."
        )

    manifest_model_features = manifest.get(
        "model_input_features",
        [],
    )

    if manifest_model_features:
        if set(manifest_model_features) != set(ALL_FEATURES):
            raise ValueError(
                "Manifest model_input_features do not match "
                "zero_shot_predictor.py ALL_FEATURES.\n"
                f"Manifest: {sorted(manifest_model_features)}\n"
                f"Predictor: {sorted(ALL_FEATURES)}"
            )

    required_columns = [
        "device_id",
        "study_id",
        "technology_family",
        "device_state_count_status",
        "parameter_source",
        "precision_basis",
        TARGET,
        "estimated_memristor_cells",
        "estimated_physical_crossbar_tiles",
        "relative_hardware_cost_proxy",
    ] + ALL_FEATURES

    missing_columns = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "ML dataset is missing columns:\n"
            f"{missing_columns}"
        )

    missing_values = (
        df[ALL_FEATURES]
        .isna()
        .sum()
    )
    missing_values = missing_values[
        missing_values > 0
    ]

    if not missing_values.empty:
        raise ValueError(
            "Model features contain missing values:\n"
            f"{missing_values}"
        )

    device_group_check = (
        df.groupby("device_id")
        .agg(
            studies=("study_id", "nunique"),
            families=("technology_family", "nunique"),
        )
    )

    if (
        (device_group_check["studies"] != 1).any()
        or
        (device_group_check["families"] != 1).any()
    ):
        raise ValueError(
            "Every device must map to exactly one study "
            "and one technology family."
        )

    rows_per_device = df["device_id"].value_counts()

    if rows_per_device.nunique() != 1:
        raise ValueError(
            "Current experiment expects an equal "
            "configuration count per device."
        )

    return df, manifest


# ============================================================
# Main
# ============================================================

def main():
    df, _ = load_and_validate_dataset()

    configs_per_device = int(
        df["device_id"].value_counts().iloc[0]
    )
    device_count = int(df["device_id"].nunique())
    study_count = int(df["study_id"].nunique())
    family_count = int(
        df["technology_family"].nunique()
    )

    print()
    print(
        "STUDY-AWARE ZERO-SHOT + VALIDATION-AWARE "
        "RECOMMENDATION EXPERIMENT"
    )
    print("=" * 64)
    print("Rows:", len(df))
    print("Device profiles:", device_count)
    print("Distinct source studies:", study_count)
    print("Technology families:", family_count)
    print(
        "Configurations per device:",
        configs_per_device,
    )

    print()
    print("PRIMARY SPLIT:")
    print(
        "  STUDY-BLOCKED LEAVE-ONE-DEVICE-OUT"
    )

    print()
    print("ADDITIONAL SPLITS:")
    print("  LEAVE-ONE-STUDY-OUT")
    print("  LEAVE-ONE-FAMILY-OUT")

    print()
    print("RECOMMENDATION METHODS:")
    print("  1. Original predicted-accuracy + cost")
    print(
        "  2. Tree-disagreement heuristic + cost"
    )
    print(
        "  3. Validation-aware score + cost"
    )
    print(
        "  4. NEW support-gated final recommendation"
    )

    print()
    print(
        "Validation-aware score = predicted accuracy "
        "- training-only nested study-blocked historical error penalty"
    )
    print(
        "The penalty is empirical and uncalibrated; "
        "it is NOT a confidence interval."
    )
    print(
        "The support gate allows cost optimization only when the "
        "held-out device is inside the same-mode ON/OFF training envelope "
        "represented by at least two training device profiles. "
        "Otherwise it falls back to accuracy-first selection."
    )

    print()
    print("MODEL FEATURES")
    print("-" * 64)
    for feature in ALL_FEATURES:
        print(" +", feature)

    print()
    print("NOT MODEL FEATURES")
    print("-" * 64)
    for feature in NOT_MODEL_FEATURES:
        print(" -", feature)

    primary_predictions, primary_summary = (
        run_primary_validation(df)
    )

    study_predictions, study_summary = (
        run_study_validation(df)
    )

    family_predictions, family_summary = (
        run_family_validation(df)
    )

    primary_overview = summarize_validation_mode(
        PRIMARY_MODE,
        primary_predictions,
        primary_summary,
    )

    study_overview = summarize_validation_mode(
        STUDY_MODE,
        study_predictions,
        study_summary,
    )

    family_overview = summarize_validation_mode(
        FAMILY_MODE,
        family_predictions,
        family_summary,
    )

    overview_df = pd.DataFrame(
        [
            primary_overview,
            study_overview,
            family_overview,
        ]
    )

    deployment_reliability = (
        build_deployment_reliability_table(
            primary_predictions
        )
    )

    Path("results/tables").mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_predictions.to_csv(
        PRIMARY_PREDICTIONS_FILE,
        index=False,
    )
    primary_summary.to_csv(
        PRIMARY_SUMMARY_FILE,
        index=False,
    )

    study_predictions.to_csv(
        STUDY_PREDICTIONS_FILE,
        index=False,
    )
    study_summary.to_csv(
        STUDY_SUMMARY_FILE,
        index=False,
    )

    family_predictions.to_csv(
        FAMILY_PREDICTIONS_FILE,
        index=False,
    )
    family_summary.to_csv(
        FAMILY_SUMMARY_FILE,
        index=False,
    )

    overview_df.to_csv(
        VALIDATION_OVERVIEW_FILE,
        index=False,
    )

    deployment_reliability.to_csv(
        RELIABILITY_PENALTY_FILE,
        index=False,
    )

    print_mode_summary(
        "PRIMARY: STUDY-BLOCKED DEVICE HOLDOUT",
        primary_summary,
        primary_overview,
    )

    print_mode_summary(
        "SECONDARY: LEAVE-ONE-STUDY-OUT",
        study_summary,
        study_overview,
    )

    print_mode_summary(
        "SECONDARY: LEAVE-ONE-FAMILY-OUT",
        family_summary,
        family_overview,
    )

    print()
    print("RELIABILITY PENALTY TABLE")
    print("-" * 64)
    print(
        deployment_reliability.round(3).to_string(
            index=False
        )
    )

    print()
    print("SEARCH EFFICIENCY")
    print("-" * 64)
    print(
        f"One selected validation instead of "
        f"{configs_per_device} exhaustive unseen-device "
        "configuration simulations."
    )
    print(
        f"Expensive-search reduction: "
        f"{(1 - 1 / configs_per_device) * 100:.2f}%"
    )
    print(
        "This refers only to expensive unseen-device "
        "configuration evaluations, not total training, "
        "fabrication, energy, or monetary cost."
    )

    print()
    print("CURRENT EVIDENCE / METHOD LIMITATION")
    print("-" * 64)
    print(
        f"{device_count} device profiles come from "
        f"{study_count} source studies across "
        f"{family_count} technology families."
    )
    print(
        "TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt "
        "share one experimental study."
    )
    print(
        "The primary split blocks same-study siblings "
        "from training."
    )
    print(
        "The Random-Forest tree disagreement remains "
        "an uncalibrated ensemble heuristic."
    )
    print(
        "The validation-aware penalty is also an "
        "empirical heuristic. For fair held-out evaluation, "
        "it is constructed only from nested study-blocked "
        "predictions inside the outer training set."
    )
    print(
        "The support gate is a transparent descriptor-extrapolation "
        "safeguard, not a calibrated OOD detector. It prevents aggressive "
        "cost optimization when same-mode device support is insufficient "
        "or the ON/OFF ratio lies outside the training envelope."
    )
    print(
        "These experiments remain pilot validation, "
        "not proof of broad physical generalization."
    )

    if "HfOx_01" in set(df["device_id"].astype(str)):
        print(
            "HfOx_01 uses its 2024 variability paper as "
            "the primary study_id; same-stack multilevel "
            "support is retained separately in provenance."
        )

    if "HfZrOx_01" in set(df["device_id"].astype(str)):
        print(
            "HfZrOx_01 is an independent low-ON/OFF "
            "ANALOG profile with reported post-wake-up "
            "ON/OFF=2.0; absolute conductance is normalized "
            "from the ratio."
        )

    if "TiOx_04" in set(df["device_id"].astype(str)):
        print(
            "TiOx_04 is an independent TiOx/TiOy "
            "GRADUAL_MULTILEVEL profile with reported "
            "ON/OFF approximately 76 at 0.2 V; absolute "
            "conductance is normalized from the ratio."
        )

    print()
    print("SAVED OUTPUTS")
    print("-" * 64)
    print(PRIMARY_PREDICTIONS_FILE)
    print(PRIMARY_SUMMARY_FILE)
    print(STUDY_PREDICTIONS_FILE)
    print(STUDY_SUMMARY_FILE)
    print(FAMILY_PREDICTIONS_FILE)
    print(FAMILY_SUMMARY_FILE)
    print(VALIDATION_OVERVIEW_FILE)
    print(RELIABILITY_PENALTY_FILE)


if __name__ == "__main__":
    main()
