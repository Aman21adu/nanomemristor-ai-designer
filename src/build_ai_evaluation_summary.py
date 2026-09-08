from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


# ============================================================
# PURPOSE
# ============================================================
#
# Build a mentor-ready evaluation of the AI recommender.
#
# Main question:
#
#     "How capable / accurate is the AI?"
#
# Evaluation:
#
#     hold out one entire physical memristor device
#                       ->
#        train on the other devices only
#                       ->
#      predict the unseen device's configurations
#                       ->
#           recommend one configuration
#                       ->
# compare recommendation against exhaustive ground truth
#
#
# IMPORTANT
# ============================================================
#
# Dataset size is read dynamically from the generated
# zero-shot result files.
#
# Configuration rows are NOT independent physical-device
# samples.
#
# The independent evidence unit is the held-out physical
# device.
#
# This remains pilot validation, not proof of broad
# cross-device generalization.
# ============================================================


# ============================================================
# INPUT FILES
# ============================================================

SUMMARY_INPUT = (
    "results/tables/"
    "zero_shot_summary.csv"
)


PREDICTIONS_INPUT = (
    "results/tables/"
    "zero_shot_predictions.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_CSV = (
    "results/tables/"
    "ai_evaluation_evidence.csv"
)


OUTPUT_TEXT = (
    "results/tables/"
    "ai_evaluation_summary.txt"
)


# ============================================================
# EXPERIMENT DEFINITION
# ============================================================

NEAR_OPTIMAL_TOLERANCE_PP = 0.5


# ============================================================
# HELPERS
# ============================================================

def require_file(
    file_path,
):

    path = Path(
        file_path
    )


    if not path.exists():

        raise FileNotFoundError(

            f"Required input file not found: "
            f"{file_path}"

        )


    return path


def bool_series(
    series,
):

    # --------------------------------------------------------
    # Robust conversion in case CSV booleans are read as
    # strings instead of native bool values.
    # --------------------------------------------------------

    if series.dtype == bool:

        return series


    converted = (

        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({

            "true": True,
            "1": True,
            "yes": True,

            "false": False,
            "0": False,
            "no": False,

        })

    )


    if converted.isna().any():

        bad_values = (

            series[
                converted.isna()
            ]
            .astype(str)
            .unique()
            .tolist()

        )


        raise ValueError(

            "Could not safely convert boolean values: "
            f"{bad_values}"

        )


    return converted.astype(
        bool
    )


def safe_rank_text(
    value,
):

    if pd.isna(
        value
    ):

        return (
            "outside true near-optimal region"
        )


    return str(
        int(
            value
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "BUILDING AI ZERO-SHOT EVALUATION SUMMARY"
    )

    print(
        "========================================"
    )


    # ========================================================
    # Validate inputs
    # ========================================================

    summary_path = require_file(
        SUMMARY_INPUT
    )


    predictions_path = require_file(
        PREDICTIONS_INPUT
    )


    # ========================================================
    # Load data
    # ========================================================

    summary_df = pd.read_csv(
        summary_path
    )


    predictions_df = pd.read_csv(
        predictions_path
    )


    # ========================================================
    # Basic structural validation
    # ========================================================

    required_summary_columns = [

        "held_out_device",
        "held_out_family",

        "training_devices",
        "training_rows",
        "held_out_rows",

        "near_region_mae_pp",
        "near_region_rmse_pp",

        "uncertainty_error_correlation",

        "true_raw_best_accuracy",

        "true_cost_aware_crossbar",
        "true_cost_aware_weight_bits",
        "true_cost_aware_adc_bits",
        "true_cost_aware_accuracy",

        "baseline_crossbar",
        "baseline_weight_bits",
        "baseline_adc_bits",
        "baseline_predicted_accuracy",
        "baseline_actual_accuracy",
        "baseline_regret_pp",
        "baseline_near_optimal_success",
        "baseline_exact_match",
        "baseline_cost_aware_rank",
        "baseline_cost_aware_top3",

        "confidence_crossbar",
        "confidence_weight_bits",
        "confidence_adc_bits",
        "confidence_actual_accuracy",
        "confidence_regret_pp",
        "confidence_near_optimal_success",
        "confidence_exact_match",
        "confidence_cost_aware_rank",
        "confidence_cost_aware_top3",

        "confidence_fallback_used",

        "search_reduction_pct",

    ]


    missing_summary_columns = [

        column

        for column
        in required_summary_columns

        if column
        not in summary_df.columns

    ]


    if missing_summary_columns:

        raise ValueError(

            "zero_shot_summary.csv is missing required "
            f"columns: {missing_summary_columns}"

        )


    required_prediction_columns = [

        "held_out_device",
        "accuracy",
        "predicted_accuracy",
        "prediction_uncertainty",
        "absolute_prediction_error_pp",
        "is_true_near_optimal",

    ]


    missing_prediction_columns = [

        column

        for column
        in required_prediction_columns

        if column
        not in predictions_df.columns

    ]


    if missing_prediction_columns:

        raise ValueError(

            "zero_shot_predictions.csv is missing required "
            f"columns: {missing_prediction_columns}"

        )


    # ========================================================
    # Independent evidence size
    # ========================================================

    independent_devices = int(

        summary_df[
            "held_out_device"
        ]
        .nunique()

    )


    if independent_devices < 2:

        raise ValueError(

            "At least two held-out devices are required "
            "for leave-one-device-out evaluation."

        )


    held_out_counts = (

        summary_df[
            "held_out_rows"
        ]
        .astype(int)
        .unique()

    )


    if len(
        held_out_counts
    ) != 1:

        raise ValueError(

            "Held-out devices do not all contain the same "
            "number of configurations.\n"
            f"Found held_out_rows values: "
            f"{sorted(held_out_counts.tolist())}"

        )


    configurations_per_device = int(
        held_out_counts[0]
    )


    total_configuration_rows = (

        independent_devices
        *
        configurations_per_device

    )


    if (
        len(
            predictions_df
        )
        !=
        total_configuration_rows
    ):

        raise ValueError(

            "Prediction-row count does not match the "
            "expected leave-one-device-out structure.\n"
            f"Expected: {total_configuration_rows}\n"
            f"Found:    {len(predictions_df)}"

        )


    # ========================================================
    # Training-device count
    # ========================================================

    training_device_counts = (

        summary_df[
            "training_devices"
        ]
        .astype(int)
        .unique()

    )


    if len(
        training_device_counts
    ) != 1:

        raise ValueError(

            "Training-device count is inconsistent "
            "across zero-shot folds.\n"
            f"Found: {sorted(training_device_counts.tolist())}"

        )


    training_devices_per_fold = int(
        training_device_counts[0]
    )


    # ========================================================
    # Convert booleans safely
    # ========================================================

    baseline_success = bool_series(

        summary_df[
            "baseline_near_optimal_success"
        ]

    )


    baseline_exact = bool_series(

        summary_df[
            "baseline_exact_match"
        ]

    )


    baseline_top3 = bool_series(

        summary_df[
            "baseline_cost_aware_top3"
        ]

    )


    confidence_success = bool_series(

        summary_df[
            "confidence_near_optimal_success"
        ]

    )


    confidence_exact = bool_series(

        summary_df[
            "confidence_exact_match"
        ]

    )


    confidence_top3 = bool_series(

        summary_df[
            "confidence_cost_aware_top3"
        ]

    )


    confidence_fallback = bool_series(

        summary_df[
            "confidence_fallback_used"
        ]

    )


    # ========================================================
    # Baseline recommendation metrics
    # ========================================================

    baseline_mean_regret = float(

        summary_df[
            "baseline_regret_pp"
        ]
        .mean()

    )


    baseline_median_regret = float(

        summary_df[
            "baseline_regret_pp"
        ]
        .median()

    )


    baseline_worst_regret = float(

        summary_df[
            "baseline_regret_pp"
        ]
        .max()

    )


    baseline_success_count = int(

        baseline_success
        .sum()

    )


    baseline_success_rate = (

        100.0
        *
        baseline_success_count
        /
        independent_devices

    )


    baseline_exact_count = int(

        baseline_exact
        .sum()

    )


    baseline_exact_rate = (

        100.0
        *
        baseline_exact_count
        /
        independent_devices

    )


    baseline_top3_count = int(

        baseline_top3
        .sum()

    )


    baseline_top3_rate = (

        100.0
        *
        baseline_top3_count
        /
        independent_devices

    )


    # ========================================================
    # Confidence-aware recommendation metrics
    # ========================================================

    confidence_mean_regret = float(

        summary_df[
            "confidence_regret_pp"
        ]
        .mean()

    )


    confidence_success_count = int(

        confidence_success
        .sum()

    )


    confidence_success_rate = (

        100.0
        *
        confidence_success_count
        /
        independent_devices

    )


    confidence_exact_count = int(

        confidence_exact
        .sum()

    )


    confidence_exact_rate = (

        100.0
        *
        confidence_exact_count
        /
        independent_devices

    )


    confidence_top3_count = int(

        confidence_top3
        .sum()

    )


    confidence_top3_rate = (

        100.0
        *
        confidence_top3_count
        /
        independent_devices

    )


    fallback_count = int(

        confidence_fallback
        .sum()

    )


    fallback_rate = (

        100.0
        *
        fallback_count
        /
        independent_devices

    )


    # ========================================================
    # Pooled near-optimal-region prediction metrics
    # ========================================================

    near_mask = bool_series(

        predictions_df[
            "is_true_near_optimal"
        ]

    )


    overall_near = predictions_df[

        near_mask
        ==
        True

    ].copy()


    if len(
        overall_near
    ) == 0:

        raise ValueError(

            "No true near-optimal configurations were "
            "found in zero_shot_predictions.csv."

        )


    overall_near_mae = float(

        mean_absolute_error(

            overall_near[
                "accuracy"
            ],

            overall_near[
                "predicted_accuracy"
            ],

        )

    )


    overall_near_rmse = float(

        np.sqrt(

            mean_squared_error(

                overall_near[
                    "accuracy"
                ],

                overall_near[
                    "predicted_accuracy"
                ],

            )

        )

    )


    near_region_rows = int(
        len(
            overall_near
        )
    )


    # ========================================================
    # Uncertainty diagnostic
    # ========================================================

    uncertainty_values = (

        predictions_df[
            "prediction_uncertainty"
        ]
        .to_numpy(
            dtype=float
        )

    )


    absolute_errors = (

        predictions_df[
            "absolute_prediction_error_pp"
        ]
        .to_numpy(
            dtype=float
        )

    )


    if (

        np.std(
            uncertainty_values
        ) > 0

        and

        np.std(
            absolute_errors
        ) > 0

    ):

        overall_uncertainty_error_correlation = float(

            np.corrcoef(

                uncertainty_values,
                absolute_errors,

            )[0, 1]

        )


    else:

        overall_uncertainty_error_correlation = float(
            "nan"
        )


    # ========================================================
    # Search reduction
    # ========================================================

    exhaustive_evaluations = int(
        configurations_per_device
    )


    recommended_evaluations = 1


    search_reduction_pct = (

        100.0
        *
        (

            1.0

            -

            (
                recommended_evaluations
                /
                exhaustive_evaluations
            )

        )

    )


    # ========================================================
    # Cross-check search reduction stored by predictor
    # ========================================================

    stored_search_reduction_values = (

        summary_df[
            "search_reduction_pct"
        ]
        .astype(float)
        .to_numpy()

    )


    if not np.allclose(

        stored_search_reduction_values,
        search_reduction_pct,
        atol=0.01,

    ):

        raise ValueError(

            "Search-reduction value in "
            "zero_shot_summary.csv does not match "
            "the current configuration count.\n"
            f"Calculated: {search_reduction_pct:.4f}%\n"
            f"Stored values: "
            f"{stored_search_reduction_values.tolist()}"

        )


    # ========================================================
    # Identify failure cases
    # ========================================================

    failure_df = summary_df[

        baseline_success
        ==
        False

    ].copy()


    failure_count = int(
        len(
            failure_df
        )
    )


    # ========================================================
    # Build evidence table
    # ========================================================

    evidence_rows = []


    evidence_rows.append({

        "evidence_id":
            "BASELINE_RECOMMENDATION_QUALITY",

        "metric_category":
            "DESIGN_RECOMMENDATION",

        "metric_name":
            "Mean recommendation regret",

        "value":
            baseline_mean_regret,

        "unit":
            "percentage_points",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                "Average difference between the true exhaustive "
                "raw-best accuracy and the accuracy of the "
                "configuration recommended for each completely "
                "held-out device."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "BASELINE_NEAR_OPTIMAL_SUCCESS",

        "metric_category":
            "DESIGN_RECOMMENDATION",

        "metric_name":
            (
                f"Recommendation within "
                f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp "
                f"of exhaustive raw best"
            ),

        "value":
            baseline_success_rate,

        "unit":
            "percent",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                f"{baseline_success_count}/{independent_devices} "
                f"held-out devices received a recommendation "
                f"within {NEAR_OPTIMAL_TOLERANCE_PP:.1f} "
                f"percentage points of the true exhaustive "
                f"best accuracy."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "BASELINE_EXACT_COST_AWARE_MATCH",

        "metric_category":
            "DESIGN_RECOMMENDATION",

        "metric_name":
            "Exact cost-aware optimum match",

        "value":
            baseline_exact_rate,

        "unit":
            "percent",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                f"{baseline_exact_count}/{independent_devices} "
                f"held-out devices matched the exhaustive "
                f"cost-aware selected configuration exactly."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "BASELINE_COST_AWARE_TOP3",

        "metric_category":
            "DESIGN_RECOMMENDATION",

        "metric_name":
            "Cost-aware Top-3 success",

        "value":
            baseline_top3_rate,

        "unit":
            "percent",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                f"{baseline_top3_count}/{independent_devices} "
                f"held-out recommendations were within the "
                f"Top-3 cost-aware candidates of the true "
                f"near-optimal region."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "NEAR_REGION_MAE",

        "metric_category":
            "PREDICTION_ACCURACY",

        "metric_name":
            "Pooled near-optimal-region MAE",

        "value":
            overall_near_mae,

        "unit":
            "percentage_points",

        "supporting_count":
            near_region_rows,

        "interpretation":
            (
                "Prediction error evaluated only on configurations "
                "that are truly within the defined near-optimal "
                "accuracy region across all held-out folds."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "NEAR_REGION_RMSE",

        "metric_category":
            "PREDICTION_ACCURACY",

        "metric_name":
            "Pooled near-optimal-region RMSE",

        "value":
            overall_near_rmse,

        "unit":
            "percentage_points",

        "supporting_count":
            near_region_rows,

        "interpretation":
            (
                "Root-mean-square prediction error in the "
                "decision-relevant near-optimal region."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "UNCERTAINTY_ERROR_CORRELATION",

        "metric_category":
            "UNCERTAINTY_DIAGNOSTIC",

        "metric_name":
            "Tree-disagreement / absolute-error correlation",

        "value":
            overall_uncertainty_error_correlation,

        "unit":
            "pearson_correlation",

        "supporting_count":
            len(
                predictions_df
            ),

        "interpretation":
            (
                "Positive correlation means Random-Forest tree "
                "disagreement tends to rise when prediction error "
                "rises. This is a diagnostic heuristic, not a "
                "calibrated confidence interval."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "CONFIDENCE_MEAN_REGRET",

        "metric_category":
            "UNCERTAINTY_AWARE_RECOMMENDATION",

        "metric_name":
            "Confidence-aware mean regret",

        "value":
            confidence_mean_regret,

        "unit":
            "percentage_points",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                "The uncertainty-aware rule changes mean regret "
                "but does not improve the current near-optimal "
                "success rate."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "CONFIDENCE_SUCCESS_RATE",

        "metric_category":
            "UNCERTAINTY_AWARE_RECOMMENDATION",

        "metric_name":
            "Confidence-aware near-optimal success",

        "value":
            confidence_success_rate,

        "unit":
            "percent",

        "supporting_count":
            independent_devices,

        "interpretation":
            (
                f"{confidence_success_count}/{independent_devices} "
                f"held-out devices succeed under the "
                f"confidence-aware recommendation rule."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "SEARCH_REDUCTION",

        "metric_category":
            "SEARCH_EFFICIENCY",

        "metric_name":
            "Unseen-device expensive-evaluation reduction",

        "value":
            search_reduction_pct,

        "unit":
            "percent",

        "supporting_count":
            exhaustive_evaluations,

        "interpretation":
            (
                f"One recommended configuration is evaluated "
                f"instead of {exhaustive_evaluations} exhaustive "
                f"unseen-device configurations. This is not a "
                f"claim of total computational-cost reduction."
            ),

    })


    evidence_rows.append({

        "evidence_id":
            "INDEPENDENT_DEVICE_COUNT",

        "metric_category":
            "EVIDENCE_LIMITATION",

        "metric_name":
            "Independent physical devices",

        "value":
            independent_devices,

        "unit":
            "devices",

        "supporting_count":
            total_configuration_rows,

        "interpretation":
            (
                f"The dataset contains "
                f"{total_configuration_rows} simulated "
                f"configuration rows, but only "
                f"{independent_devices} independent physical "
                f"devices. Generalization evidence is therefore "
                f"still limited."
            ),

    })


    evidence_df = pd.DataFrame(
        evidence_rows
    )


    # ========================================================
    # Save evidence CSV
    # ========================================================

    Path(
        "results/tables"
    ).mkdir(

        parents=True,
        exist_ok=True,

    )


    evidence_df.to_csv(

        OUTPUT_CSV,
        index=False,

    )


    # ========================================================
    # Build mentor-ready text report
    # ========================================================

    lines = []


    lines.append(
        "AI ZERO-SHOT EVALUATION SUMMARY"
    )

    lines.append(
        "================================"
    )

    lines.append("")


    # ========================================================
    # Evaluation protocol
    # ========================================================

    lines.append(
        "1. HOW THE AI WAS TESTED"
    )

    lines.append(
        "------------------------"
    )

    lines.append("")


    lines.append(

        "The model was evaluated with leave-one-device-out "
        "zero-shot testing."

    )


    lines.append("")


    lines.append(

        "For each fold, one complete physical memristor "
        "device was hidden from training."

    )


    lines.append(

        f"The model trained on the remaining "
        f"{training_devices_per_fold} devices and then "
        f"predicted the {configurations_per_device} "
        f"accelerator configurations for the unseen device."

    )


    lines.append("")


    lines.append(

        "The AI then selected one accelerator configuration, "
        "which was compared against exhaustive simulation "
        f"of all {configurations_per_device} configurations "
        "for that held-out device."

    )


    lines.append("")


    lines.append(

        f"Independent physical devices: "
        f"{independent_devices}."

    )


    lines.append(

        f"Configurations per device: "
        f"{configurations_per_device}."

    )


    lines.append(

        f"Total simulated rows: "
        f"{total_configuration_rows}."

    )


    lines.append("")


    lines.append(

        f"The {total_configuration_rows} configuration rows "
        f"are not {total_configuration_rows} independent "
        "physical samples."

    )


    lines.append("")


    # ========================================================
    # Per-device recommendation results
    # ========================================================

    lines.append(
        "2. HELD-OUT DEVICE RESULTS"
    )

    lines.append(
        "--------------------------"
    )

    lines.append("")


    for _, row in summary_df.iterrows():

        device_id = str(
            row[
                "held_out_device"
            ]
        )


        success = bool(

            baseline_success.loc[
                row.name
            ]

        )


        exact = bool(

            baseline_exact.loc[
                row.name
            ]

        )


        top3 = bool(

            baseline_top3.loc[
                row.name
            ]

        )


        lines.append(
            f"{device_id}:"
        )


        lines.append(

            f"  True exhaustive raw-best accuracy: "
            f"{row['true_raw_best_accuracy']:.2f}%"

        )


        lines.append(

            f"  AI recommendation: "
            f"{int(row['baseline_crossbar'])}x"
            f"{int(row['baseline_crossbar'])}, "
            f"W{int(row['baseline_weight_bits'])}, "
            f"ADC{int(row['baseline_adc_bits'])}"

        )


        lines.append(

            f"  Predicted accuracy: "
            f"{row['baseline_predicted_accuracy']:.2f}%"

        )


        lines.append(

            f"  Actual accuracy: "
            f"{row['baseline_actual_accuracy']:.2f}%"

        )


        lines.append(

            f"  Regret: "
            f"{row['baseline_regret_pp']:.2f} pp"

        )


        lines.append(

            f"  Within {NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp "
            f"of raw best: "
            f"{'YES' if success else 'NO'}"

        )


        lines.append(

            f"  Exact cost-aware optimum: "
            f"{'YES' if exact else 'NO'}"

        )


        lines.append(

            f"  Cost-aware rank: "
            f"{safe_rank_text(row['baseline_cost_aware_rank'])}"

        )


        lines.append(

            f"  Cost-aware Top-3: "
            f"{'YES' if top3 else 'NO'}"

        )


        lines.append("")


    # ========================================================
    # Overall recommendation quality
    # ========================================================

    lines.append(
        "3. OVERALL RECOMMENDATION QUALITY"
    )

    lines.append(
        "---------------------------------"
    )

    lines.append("")


    lines.append(

        f"Mean recommendation regret: "
        f"{baseline_mean_regret:.3f} pp."

    )


    lines.append(

        f"Median recommendation regret: "
        f"{baseline_median_regret:.3f} pp."

    )


    lines.append(

        f"Worst recommendation regret: "
        f"{baseline_worst_regret:.3f} pp."

    )


    lines.append("")


    lines.append(

        f"Near-optimal success: "
        f"{baseline_success_count}/{independent_devices} "
        f"devices = {baseline_success_rate:.1f}%."

    )


    lines.append(

        f"Exact cost-aware optimum: "
        f"{baseline_exact_count}/{independent_devices} "
        f"devices = {baseline_exact_rate:.1f}%."

    )


    lines.append(

        f"Cost-aware Top-3: "
        f"{baseline_top3_count}/{independent_devices} "
        f"devices = {baseline_top3_rate:.1f}%."

    )


    lines.append("")


    # ========================================================
    # Prediction accuracy
    # ========================================================

    lines.append(
        "4. PREDICTION ACCURACY WHERE IT MATTERS"
    )

    lines.append(
        "---------------------------------------"
    )

    lines.append("")


    lines.append(

        "Global prediction metrics are not the main evidence "
        "because very poor configurations can make global "
        "R-squared appear extremely high."

    )


    lines.append("")


    lines.append(

        "The more decision-relevant measurement is prediction "
        "error in the true near-optimal region."

    )


    lines.append("")


    lines.append(

        f"Pooled near-region MAE: "
        f"{overall_near_mae:.3f} pp."

    )


    lines.append(

        f"Pooled near-region RMSE: "
        f"{overall_near_rmse:.3f} pp."

    )


    lines.append(

        f"Near-region configuration rows evaluated: "
        f"{near_region_rows}."

    )


    lines.append("")


    # ========================================================
    # Failure
    # ========================================================

    lines.append(
        "5. DOCUMENTED FAILURE"
    )

    lines.append(
        "---------------------"
    )

    lines.append("")


    if failure_count == 0:

        lines.append(

            "No held-out device exceeded the "
            f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp "
            "near-optimal tolerance."

        )


    else:

        for _, row in failure_df.iterrows():

            lines.append(

                f"{row['held_out_device']} failed the "
                f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp "
                f"criterion with "
                f"{row['baseline_regret_pp']:.2f} pp regret."

            )


            lines.append(

                f"The recommended configuration achieved "
                f"{row['baseline_actual_accuracy']:.2f}% "
                f"versus a raw exhaustive best of "
                f"{row['true_raw_best_accuracy']:.2f}%."

            )


            lines.append("")


    lines.append(

        "This failure is retained as part of the evaluation "
        "rather than removed from the reported results."

    )


    lines.append("")


    # ========================================================
    # Uncertainty
    # ========================================================

    lines.append(
        "6. UNCERTAINTY DIAGNOSTIC"
    )

    lines.append(
        "-------------------------"
    )

    lines.append("")


    lines.append(

        f"Overall tree-disagreement / absolute-error "
        f"correlation: "
        f"{overall_uncertainty_error_correlation:.3f}."

    )


    lines.append("")


    lines.append(

        "Positive correlation means tree disagreement tends "
        "to become larger when prediction error becomes larger."

    )


    lines.append("")


    lines.append(

        "However, this is an uncalibrated ensemble-disagreement "
        "heuristic and must not be presented as a statistically "
        "calibrated confidence interval."

    )


    lines.append("")


    lines.append(

        f"Baseline mean regret: "
        f"{baseline_mean_regret:.3f} pp."

    )


    lines.append(

        f"Confidence-aware mean regret: "
        f"{confidence_mean_regret:.3f} pp."

    )


    lines.append(

        f"Baseline near-optimal success: "
        f"{baseline_success_rate:.1f}%."

    )


    lines.append(

        f"Confidence-aware near-optimal success: "
        f"{confidence_success_rate:.1f}%."

    )


    lines.append("")


    if (
        confidence_success_rate
        >
        baseline_success_rate
    ):

        lines.append(

            "The confidence-aware rule improves the current "
            "near-optimal success rate."

        )


    elif (
        confidence_success_rate
        ==
        baseline_success_rate

        and

        confidence_mean_regret
        <
        baseline_mean_regret
    ):

        lines.append(

            "The confidence-aware rule keeps the same success "
            "rate while slightly reducing mean regret."

        )


    else:

        lines.append(

            "The confidence-aware rule does not improve the "
            "current baseline recommendation performance, "
            "so the baseline predictor remains the primary "
            "recommender."

        )


    lines.append("")


    lines.append(

        f"Confidence fallback rate: "
        f"{fallback_rate:.1f}%."

    )


    lines.append("")


    # ========================================================
    # Search efficiency
    # ========================================================

    lines.append(
        "7. SEARCH EFFICIENCY"
    )

    lines.append(
        "--------------------"
    )

    lines.append("")


    lines.append(

        f"Exhaustive unseen-device search: "
        f"{exhaustive_evaluations} configuration evaluations."

    )


    lines.append(

        f"AI recommendation workflow: "
        f"{recommended_evaluations} selected configuration "
        f"evaluation."

    )


    lines.append("")


    lines.append(

        f"Reduction in expensive unseen-device "
        f"simulation/validation evaluations: "
        f"{search_reduction_pct:.2f}%."

    )


    lines.append("")


    lines.append(

        f"This is not a claim of {search_reduction_pct:.2f}% "
        "reduction in total training or computational cost."

    )


    lines.append("")


    # ========================================================
    # Mentor answer
    # ========================================================

    lines.append(
        "MENTOR ANSWER: HOW ACCURATE IS THE AI?"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append("")


    lines.append(

        "In the current pilot evaluation, each memristor "
        "device is completely hidden during its test fold."

    )


    lines.append("")


    lines.append(

        f"The baseline recommender selects a configuration "
        f"within {NEAR_OPTIMAL_TOLERANCE_PP:.1f} percentage "
        f"points of the exhaustive raw best for "
        f"{baseline_success_count}/{independent_devices} "
        f"held-out devices ({baseline_success_rate:.1f}%)."

    )


    lines.append("")


    lines.append(

        f"Mean recommendation regret is "
        f"{baseline_mean_regret:.2f} percentage points, "
        f"and the recommendation is within the cost-aware "
        f"Top-3 for {baseline_top3_count}/{independent_devices} "
        f"devices ({baseline_top3_rate:.1f}%)."

    )


    lines.append("")


    lines.append(

        f"In the decision-relevant near-optimal region, "
        f"the pooled prediction MAE is "
        f"{overall_near_mae:.3f} percentage points."

    )


    lines.append("")


    if failure_count > 0:

        lines.append(

            f"{failure_count} held-out device"
            f"{'s' if failure_count != 1 else ''} "
            f"currently fail"
            f"{'s' if failure_count == 1 else ''} "
            "the chosen near-optimal criterion, so the "
            "system is not presented as perfectly reliable."

        )


    else:

        lines.append(

            "No held-out device currently fails the chosen "
            "near-optimal criterion, but the evidence set is "
            "still too small for a broad reliability claim."

        )


    lines.append("")


    # ========================================================
    # Evidence limitation
    # ========================================================

    lines.append(
        "CURRENT EVIDENCE LIMITATION"
    )

    lines.append(
        "---------------------------"
    )

    lines.append("")


    lines.append(

        f"Only {independent_devices} independent physical "
        f"devices are currently available."

    )


    lines.append(

        f"Each zero-shot fold therefore trains on only "
        f"{training_devices_per_fold} physical devices."

    )


    lines.append("")


    lines.append(

        f"The {total_configuration_rows} simulator rows "
        f"come from those {independent_devices} devices "
        f"across {configurations_per_device} accelerator "
        f"configurations each."

    )


    lines.append("")


    lines.append(

        "Therefore the current results should be described "
        "as pilot proof-of-concept validation."

    )


    lines.append("")


    lines.append(

        "They are not yet sufficient evidence for broad "
        "cross-technology generalization to arbitrary "
        "future memristor devices."

    )


    # ========================================================
    # Save text
    # ========================================================

    with open(

        OUTPUT_TEXT,
        "w",
        encoding="utf-8",

    ) as f:

        f.write(

            "\n".join(
                lines
            )

        )


    # ========================================================
    # Console
    # ========================================================

    print()

    print(
        "Independent devices:",
        independent_devices
    )


    print(
        "Training devices per fold:",
        training_devices_per_fold
    )


    print(
        "Configurations per device:",
        configurations_per_device
    )


    print(
        "Total simulation rows:",
        total_configuration_rows
    )


    print()

    print(
        "Baseline mean regret:",
        f"{baseline_mean_regret:.3f} pp"
    )


    print(
        "Baseline near-optimal success:",
        f"{baseline_success_rate:.1f}%"
    )


    print(
        "Baseline exact cost-aware match:",
        f"{baseline_exact_rate:.1f}%"
    )


    print(
        "Baseline cost-aware Top-3:",
        f"{baseline_top3_rate:.1f}%"
    )


    print()

    print(
        "Overall near-region MAE:",
        f"{overall_near_mae:.3f} pp"
    )


    print(
        "Overall near-region RMSE:",
        f"{overall_near_rmse:.3f} pp"
    )


    print(
        "Near-region rows:",
        near_region_rows
    )


    print()

    print(
        "Uncertainty/error correlation:",
        f"{overall_uncertainty_error_correlation:.3f}"
    )


    print()

    print(
        "Baseline mean regret:",
        f"{baseline_mean_regret:.3f} pp"
    )


    print(
        "Confidence-aware mean regret:",
        f"{confidence_mean_regret:.3f} pp"
    )


    print()

    print(
        "Expensive-search reduction:",
        f"{search_reduction_pct:.2f}%"
    )


    print()

    print(
        "Documented baseline failures:",
        failure_count
    )


    print()

    print(
        "Evidence rows:",
        len(
            evidence_df
        )
    )


    print()

    print(
        "Saved CSV:"
    )

    print(
        OUTPUT_CSV
    )


    print()

    print(
        "Saved mentor summary:"
    )

    print(
        OUTPUT_TEXT
    )


    print()

    print(
        "CURRENT CLAIM"
    )

    print(
        "-------------"
    )


    print()

    print(

        f"Leave-one-device-out pilot validation across "
        f"{configurations_per_device} configurations per "
        f"device shows useful zero-shot recommendation "
        f"capability, with {failure_count} documented "
        f"failure{'s' if failure_count != 1 else ''} and "
        f"only {independent_devices} independent physical "
        f"devices."

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()