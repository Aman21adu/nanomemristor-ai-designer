from pathlib import Path

import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# Files
# ============================================================

INPUT_FILE = (
    "results/tables/"
    "ml_dataset.csv"
)

MANIFEST_FILE = (
    "results/tables/"
    "ml_dataset_manifest.json"
)


# Existing primary-output filenames are preserved so the
# current dashboard can continue reading them.
PRIMARY_PREDICTIONS_FILE = (
    "results/tables/"
    "zero_shot_predictions.csv"
)

PRIMARY_SUMMARY_FILE = (
    "results/tables/"
    "zero_shot_summary.csv"
)


# Additional validation outputs.
STUDY_PREDICTIONS_FILE = (
    "results/tables/"
    "zero_shot_study_predictions.csv"
)

STUDY_SUMMARY_FILE = (
    "results/tables/"
    "zero_shot_study_summary.csv"
)

FAMILY_PREDICTIONS_FILE = (
    "results/tables/"
    "zero_shot_family_predictions.csv"
)

FAMILY_SUMMARY_FILE = (
    "results/tables/"
    "zero_shot_family_summary.csv"
)

VALIDATION_OVERVIEW_FILE = (
    "results/tables/"
    "zero_shot_validation_overview.csv"
)


# ============================================================
# Experiment settings
# ============================================================

ACCURACY_TOLERANCE_PP = 0.5

RANDOM_STATE = 42

UNCERTAINTY_MULTIPLIER = 1.0

TARGET = "accuracy"


# ============================================================
# Model features
#
# IMPORTANT:
#
# device_id, study_id and technology_family are NEVER model
# features. They exist only for grouping, provenance and
# leakage-safe validation.
# ============================================================

NUMERIC_FEATURES = [

    # Device properties
    "device_log10_on_off_ratio",
    "state_count_available",
    "physical_state_count",

    # Accelerator configuration
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


ALL_FEATURES = (

    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES

)


NOT_MODEL_FEATURES = [

    # Grouping / provenance
    "device_id",
    "study_id",
    "technology_family",

    # Evidence/provenance descriptors
    "device_state_count_status",
    "parameter_source",
    "precision_basis",

    # Targets / cost descriptors
    "accuracy_loss",
    "relative_hardware_cost_proxy",

]


# ============================================================
# Validation mode names
# ============================================================

PRIMARY_MODE = (
    "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT"
)

STUDY_MODE = (
    "LEAVE_ONE_STUDY_OUT"
)

FAMILY_MODE = (
    "LEAVE_ONE_FAMILY_OUT"
)


# ============================================================
# Configuration identity
# ============================================================

def configuration_key(
    row
):

    return (

        int(
            row[
                "crossbar_size"
            ]
        ),

        int(
            row[
                "requested_weight_bits"
            ]
        ),

        int(
            row[
                "adc_bits"
            ]
        ),

    )


# ============================================================
# Near-optimal candidate set
# ============================================================

def get_near_optimal_candidates(
    group,
    score_column,
    tolerance_pp,
):

    best_score = float(

        group[
            score_column
        ].max()

    )


    threshold = (

        best_score
        -
        tolerance_pp

    )


    candidates = group[

        group[
            score_column
        ]
        >=
        threshold

    ].copy()


    if candidates.empty:

        raise ValueError(

            f"No near-optimal candidates "
            f"for {score_column}."

        )


    return (

        candidates,
        best_score,
        threshold,

    )


# ============================================================
# Cost-aware sorting rule
#
# Same selection philosophy as the existing predictor:
# first satisfy the near-optimal accuracy requirement,
# then prefer lower architecture-cost descriptors.
#
# The cost proxy remains heuristic and is not measured
# area, power, energy or latency.
# ============================================================

def sort_cost_aware(
    candidates,
    score_column,
):

    return (

        candidates

        .sort_values(

            by=[

                "estimated_memristor_cells",
                "relative_hardware_cost_proxy",
                "requested_weight_bits",
                "adc_bits",
                score_column,
                "crossbar_size",

            ],

            ascending=[

                True,
                True,
                True,
                True,
                False,
                False,

            ],

        )

    )


def select_near_optimal_configuration(
    group,
    score_column,
    tolerance_pp,
):

    (

        candidates,
        best_score,
        threshold,

    ) = get_near_optimal_candidates(

        group,
        score_column,
        tolerance_pp,

    )


    ranked = sort_cost_aware(

        candidates,
        score_column,

    )


    return (

        ranked.iloc[0],
        best_score,
        threshold,
        candidates,

    )


# ============================================================
# Confidence-aware selection
#
# Tree disagreement is a heuristic, NOT a calibrated
# confidence interval.
# ============================================================

def select_confidence_aware_configuration(
    group,
    tolerance_pp,
):

    predicted_best = float(

        group[
            "predicted_accuracy"
        ].max()

    )


    acceptance_threshold = (

        predicted_best
        -
        tolerance_pp

    )


    safe_candidates = group[

        group[
            "confidence_lower_bound"
        ]
        >=
        acceptance_threshold

    ].copy()


    if not safe_candidates.empty:

        ranked = sort_cost_aware(

            safe_candidates,
            "confidence_lower_bound",

        )


        chosen = ranked.iloc[0]

        fallback_used = False


    else:

        (

            chosen,
            _,
            _,
            _,

        ) = select_near_optimal_configuration(

            group,
            "predicted_accuracy",
            tolerance_pp,

        )


        fallback_used = True


    return (

        chosen,
        predicted_best,
        acceptance_threshold,
        safe_candidates,
        fallback_used,

    )


# ============================================================
# Ground-truth helpers
# ============================================================

def get_raw_best_configuration(
    group
):

    ranked = (

        group

        .sort_values(

            by=[

                "accuracy",
                "estimated_memristor_cells",
                "relative_hardware_cost_proxy",
                "requested_weight_bits",
                "adc_bits",
                "crossbar_size",

            ],

            ascending=[

                False,
                True,
                True,
                True,
                True,
                False,

            ],

        )

    )


    return ranked.iloc[0]


def same_configuration(
    row_a,
    row_b,
):

    return (

        configuration_key(
            row_a
        )

        ==

        configuration_key(
            row_b
        )

    )


def get_cost_aware_rank(
    true_candidates,
    recommended,
):

    ranked = (

        sort_cost_aware(

            true_candidates,
            "accuracy",

        )

        .reset_index(
            drop=True
        )

    )


    target_key = configuration_key(
        recommended
    )


    for index, row in ranked.iterrows():

        if (

            configuration_key(
                row
            )

            ==

            target_key

        ):

            return int(
                index + 1
            )


    return np.nan


def candidate_overlap_metrics(
    true_candidates,
    predicted_candidates,
):

    true_keys = {

        configuration_key(
            row
        )

        for _, row
        in true_candidates.iterrows()

    }


    predicted_keys = {

        configuration_key(
            row
        )

        for _, row
        in predicted_candidates.iterrows()

    }


    overlap = (

        true_keys
        .intersection(
            predicted_keys
        )

    )


    precision = (

        len(
            overlap
        )
        /
        len(
            predicted_keys
        )

        if len(
            predicted_keys
        ) > 0

        else 0.0

    )


    recall = (

        len(
            overlap
        )
        /
        len(
            true_keys
        )

        if len(
            true_keys
        ) > 0

        else 0.0

    )


    return (

        float(
            precision
        ),

        float(
            recall
        ),

        int(
            len(
                overlap
            )
        ),

    )


# ============================================================
# Random-Forest ensemble uncertainty
# ============================================================

def get_random_forest_uncertainty(
    trained_pipeline,
    X,
):

    """
    Return the mean tree prediction and standard deviation
    across Random-Forest trees.

    Standard deviation is used only as an ensemble
    disagreement heuristic.
    """

    preprocessor = (

        trained_pipeline
        .named_steps[
            "preprocessor"
        ]

    )


    forest = (

        trained_pipeline
        .named_steps[
            "regressor"
        ]

    )


    X_transformed = (

        preprocessor.transform(
            X
        )

    )


    tree_predictions = np.column_stack(

        [

            tree.predict(
                X_transformed
            )

            for tree
            in forest.estimators_

        ]

    )


    ensemble_mean = np.mean(

        tree_predictions,
        axis=1,

    )


    ensemble_std = np.std(

        tree_predictions,
        axis=1,
        ddof=1,

    )


    return (

        ensemble_mean,
        ensemble_std,

    )


# ============================================================
# Model construction
# ============================================================

def build_model():

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "numeric",
                "passthrough",
                NUMERIC_FEATURES,

            ),

            (

                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

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

            (
                "preprocessor",
                preprocessor,
            ),

            (
                "regressor",
                regressor,
            ),

        ]

    )


# ============================================================
# String helper for metadata columns
# ============================================================

def joined_values(
    values
):

    return ";".join(

        sorted(

            str(
                value
            )

            for value
            in values

        )

    )


# ============================================================
# Evaluate one held-out device after predictions exist
# ============================================================

def evaluate_device_predictions(
    device_test_df,
    train_df,
    validation_mode,
    held_out_group,
    held_out_group_rows,
    excluded_device_ids,
):

    test_df = device_test_df.copy()


    held_out_device = str(

        test_df[
            "device_id"
        ].iloc[0]

    )


    held_out_study = str(

        test_df[
            "study_id"
        ].iloc[0]

    )


    held_out_family = str(

        test_df[
            "technology_family"
        ].iloc[0]

    )


    # --------------------------------------------------------
    # Prediction quality
    # --------------------------------------------------------

    global_mae = mean_absolute_error(

        test_df[
            "accuracy"
        ],

        test_df[
            "predicted_accuracy"
        ],

    )


    global_rmse = np.sqrt(

        mean_squared_error(

            test_df[
                "accuracy"
            ],

            test_df[
                "predicted_accuracy"
            ],

        )

    )


    global_r2 = r2_score(

        test_df[
            "accuracy"
        ],

        test_df[
            "predicted_accuracy"
        ],

    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    raw_best = get_raw_best_configuration(
        test_df
    )


    true_best_accuracy = float(

        raw_best[
            "accuracy"
        ]

    )


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


    test_df[
        "is_true_near_optimal"
    ] = (

        test_df[
            "accuracy"
        ]
        >=
        true_threshold

    )


    near_region = test_df[

        test_df[
            "is_true_near_optimal"
        ]

    ].copy()


    near_region_mae = mean_absolute_error(

        near_region[
            "accuracy"
        ],

        near_region[
            "predicted_accuracy"
        ],

    )


    near_region_rmse = np.sqrt(

        mean_squared_error(

            near_region[
                "accuracy"
            ],

            near_region[
                "predicted_accuracy"
            ],

        )

    )


    # --------------------------------------------------------
    # Baseline recommendation
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
    # Confidence-aware recommendation
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


    # --------------------------------------------------------
    # Baseline metrics
    # --------------------------------------------------------

    baseline_actual_accuracy = float(

        baseline_recommended[
            "accuracy"
        ]

    )


    baseline_predicted_accuracy = float(

        baseline_recommended[
            "predicted_accuracy"
        ]

    )


    baseline_uncertainty = float(

        baseline_recommended[
            "prediction_uncertainty"
        ]

    )


    baseline_lower_bound = float(

        baseline_recommended[
            "confidence_lower_bound"
        ]

    )


    baseline_regret = (

        true_best_accuracy
        -
        baseline_actual_accuracy

    )


    baseline_success = bool(

        baseline_regret
        <=
        ACCURACY_TOLERANCE_PP

    )


    baseline_exact_match = bool(

        same_configuration(

            baseline_recommended,
            true_cost_aware,

        )

    )


    baseline_cost_rank = get_cost_aware_rank(

        true_candidates,
        baseline_recommended,

    )


    baseline_top3 = bool(

        np.isfinite(
            baseline_cost_rank
        )

        and

        baseline_cost_rank <= 3

    )


    baseline_top5 = bool(

        np.isfinite(
            baseline_cost_rank
        )

        and

        baseline_cost_rank <= 5

    )


    baseline_passes_confidence_gate = bool(

        baseline_lower_bound
        >=
        confidence_acceptance_threshold

    )


    # --------------------------------------------------------
    # Confidence-aware metrics
    # --------------------------------------------------------

    confidence_actual_accuracy = float(

        confidence_recommended[
            "accuracy"
        ]

    )


    confidence_predicted_accuracy = float(

        confidence_recommended[
            "predicted_accuracy"
        ]

    )


    confidence_uncertainty = float(

        confidence_recommended[
            "prediction_uncertainty"
        ]

    )


    confidence_lower_bound = float(

        confidence_recommended[
            "confidence_lower_bound"
        ]

    )


    confidence_regret = (

        true_best_accuracy
        -
        confidence_actual_accuracy

    )


    confidence_success = bool(

        confidence_regret
        <=
        ACCURACY_TOLERANCE_PP

    )


    confidence_exact_match = bool(

        same_configuration(

            confidence_recommended,
            true_cost_aware,

        )

    )


    confidence_cost_rank = get_cost_aware_rank(

        true_candidates,
        confidence_recommended,

    )


    confidence_top3 = bool(

        np.isfinite(
            confidence_cost_rank
        )

        and

        confidence_cost_rank <= 3

    )


    confidence_top5 = bool(

        np.isfinite(
            confidence_cost_rank
        )

        and

        confidence_cost_rank <= 5

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
    # Uncertainty diagnostic
    # --------------------------------------------------------

    uncertainty_values = (

        test_df[
            "prediction_uncertainty"
        ].to_numpy()

    )


    absolute_errors = (

        test_df[
            "absolute_prediction_error_pp"
        ].to_numpy()

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

        uncertainty_error_correlation = float(

            np.corrcoef(

                uncertainty_values,
                absolute_errors,

            )[0, 1]

        )


    else:

        uncertainty_error_correlation = np.nan


    # --------------------------------------------------------
    # Training-group metadata
    # --------------------------------------------------------

    training_device_ids = sorted(

        train_df[
            "device_id"
        ].unique()

    )


    training_study_ids = sorted(

        train_df[
            "study_id"
        ].unique()

    )


    training_family_ids = sorted(

        train_df[
            "technology_family"
        ].unique()

    )


    same_study_training_leakage = bool(

        held_out_study
        in
        set(
            training_study_ids
        )

    )


    same_family_present_in_training = bool(

        held_out_family
        in
        set(
            training_family_ids
        )

    )


    if same_study_training_leakage:

        raise ValueError(

            f"Study leakage detected for "
            f"{held_out_device}: "
            f"{held_out_study} is present in training."

        )


    search_reduction_pct = (

        1.0
        -
        (
            1.0
            /
            len(
                test_df
            )
        )

    ) * 100.0


    # --------------------------------------------------------
    # Add metadata to every prediction row
    # --------------------------------------------------------

    test_df[
        "validation_mode"
    ] = validation_mode


    test_df[
        "held_out_group"
    ] = str(
        held_out_group
    )


    test_df[
        "held_out_device"
    ] = held_out_device


    test_df[
        "held_out_study"
    ] = held_out_study


    test_df[
        "held_out_family"
    ] = held_out_family


    test_df[
        "training_devices"
    ] = int(
        len(
            training_device_ids
        )
    )


    test_df[
        "training_studies"
    ] = int(
        len(
            training_study_ids
        )
    )


    test_df[
        "training_families"
    ] = int(
        len(
            training_family_ids
        )
    )


    test_df[
        "same_study_training_leakage"
    ] = False


    # --------------------------------------------------------
    # Summary row
    #
    # Existing column names are preserved for dashboard
    # compatibility. New study/family metadata is added.
    # --------------------------------------------------------

    summary_row = {

        # New validation metadata
        "validation_mode":
            validation_mode,

        "held_out_group":
            str(
                held_out_group
            ),

        "held_out_device":
            held_out_device,

        "held_out_study":
            held_out_study,

        "held_out_family":
            held_out_family,

        "training_devices":
            int(
                len(
                    training_device_ids
                )
            ),

        "training_studies":
            int(
                len(
                    training_study_ids
                )
            ),

        "training_families":
            int(
                len(
                    training_family_ids
                )
            ),

        "training_device_ids":
            joined_values(
                training_device_ids
            ),

        "training_study_ids":
            joined_values(
                training_study_ids
            ),

        "training_family_ids":
            joined_values(
                training_family_ids
            ),

        "excluded_from_training_device_ids":
            joined_values(
                excluded_device_ids
            ),

        "same_study_training_leakage":
            False,

        "same_family_present_in_training":
            same_family_present_in_training,

        "training_rows":
            int(
                len(
                    train_df
                )
            ),

        # Preserved meaning: rows for this held-out device.
        "held_out_rows":
            int(
                len(
                    test_df
                )
            ),

        # Entire held-out study/family group can contain more
        # than one device.
        "held_out_group_rows":
            int(
                held_out_group_rows
            ),


        # Prediction metrics
        "global_mae_pp":
            float(
                global_mae
            ),

        "global_rmse_pp":
            float(
                global_rmse
            ),

        "global_r2":
            float(
                global_r2
            ),

        "near_region_mae_pp":
            float(
                near_region_mae
            ),

        "near_region_rmse_pp":
            float(
                near_region_rmse
            ),


        # Uncertainty diagnostic
        "mean_prediction_uncertainty_pp":
            float(

                test_df[
                    "prediction_uncertainty"
                ].mean()

            ),

        "uncertainty_error_correlation":
            (

                float(
                    uncertainty_error_correlation
                )

                if np.isfinite(
                    uncertainty_error_correlation
                )

                else np.nan

            ),

        "safe_candidate_count":
            int(
                len(
                    safe_candidates
                )
            ),

        "safe_candidate_precision":
            float(
                safe_precision
            ),

        "safe_candidate_recall":
            float(
                safe_recall
            ),

        "safe_candidate_overlap":
            int(
                safe_overlap
            ),

        "predicted_candidate_count":
            int(
                len(
                    predicted_candidates
                )
            ),

        "true_near_optimal_candidate_count":
            int(
                len(
                    true_candidates
                )
            ),

        "candidate_precision":
            float(
                candidate_precision
            ),

        "candidate_recall":
            float(
                candidate_recall
            ),

        "candidate_overlap":
            int(
                candidate_overlap
            ),


        # True optimum
        "true_raw_best_accuracy":
            true_best_accuracy,

        "true_cost_aware_crossbar":
            int(
                true_cost_aware[
                    "crossbar_size"
                ]
            ),

        "true_cost_aware_weight_bits":
            int(
                true_cost_aware[
                    "requested_weight_bits"
                ]
            ),

        "true_cost_aware_adc_bits":
            int(
                true_cost_aware[
                    "adc_bits"
                ]
            ),

        "true_cost_aware_accuracy":
            float(
                true_cost_aware[
                    "accuracy"
                ]
            ),

        "true_near_optimal_threshold":
            float(
                true_threshold
            ),


        # Baseline recommendation
        "baseline_crossbar":
            int(
                baseline_recommended[
                    "crossbar_size"
                ]
            ),

        "baseline_weight_bits":
            int(
                baseline_recommended[
                    "requested_weight_bits"
                ]
            ),

        "baseline_adc_bits":
            int(
                baseline_recommended[
                    "adc_bits"
                ]
            ),

        "baseline_predicted_accuracy":
            baseline_predicted_accuracy,

        "baseline_uncertainty_pp":
            baseline_uncertainty,

        "baseline_lower_bound":
            baseline_lower_bound,

        "baseline_passes_confidence_gate":
            baseline_passes_confidence_gate,

        "baseline_actual_accuracy":
            baseline_actual_accuracy,

        "baseline_regret_pp":
            float(
                baseline_regret
            ),

        "baseline_near_optimal_success":
            baseline_success,

        "baseline_exact_match":
            baseline_exact_match,

        "baseline_cost_aware_rank":
            (

                int(
                    baseline_cost_rank
                )

                if np.isfinite(
                    baseline_cost_rank
                )

                else np.nan

            ),

        "baseline_cost_aware_top3":
            baseline_top3,

        "baseline_cost_aware_top5":
            baseline_top5,

        "predicted_best_accuracy":
            float(
                predicted_best_accuracy
            ),

        "predicted_near_optimal_threshold":
            float(
                predicted_threshold
            ),


        # Confidence-aware recommendation
        "confidence_crossbar":
            int(
                confidence_recommended[
                    "crossbar_size"
                ]
            ),

        "confidence_weight_bits":
            int(
                confidence_recommended[
                    "requested_weight_bits"
                ]
            ),

        "confidence_adc_bits":
            int(
                confidence_recommended[
                    "adc_bits"
                ]
            ),

        "confidence_predicted_accuracy":
            confidence_predicted_accuracy,

        "confidence_uncertainty_pp":
            confidence_uncertainty,

        "confidence_lower_bound":
            confidence_lower_bound,

        "confidence_fallback_used":
            confidence_fallback_used,

        "confidence_actual_accuracy":
            confidence_actual_accuracy,

        "confidence_regret_pp":
            float(
                confidence_regret
            ),

        "confidence_near_optimal_success":
            confidence_success,

        "confidence_exact_match":
            confidence_exact_match,

        "confidence_cost_aware_rank":
            (

                int(
                    confidence_cost_rank
                )

                if np.isfinite(
                    confidence_cost_rank
                )

                else np.nan

            ),

        "confidence_cost_aware_top3":
            confidence_top3,

        "confidence_cost_aware_top5":
            confidence_top5,

        "confidence_acceptance_threshold":
            float(
                confidence_acceptance_threshold
            ),


        # Efficiency
        "search_reduction_pct":
            float(
                search_reduction_pct
            ),

    }


    return (

        test_df,
        summary_row,

    )


# ============================================================
# Train one model for one holdout group and evaluate every
# test device separately.
# ============================================================

def fit_predict_and_evaluate(
    train_df,
    test_df,
    validation_mode,
    held_out_group,
    excluded_device_ids,
):

    if train_df.empty:

        raise ValueError(

            f"{validation_mode}: training set is empty "
            f"for held-out group {held_out_group}."

        )


    if test_df.empty:

        raise ValueError(

            f"{validation_mode}: test set is empty "
            f"for held-out group {held_out_group}."

        )


    # --------------------------------------------------------
    # Leakage checks before model fitting
    # --------------------------------------------------------

    train_studies = set(

        train_df[
            "study_id"
        ].unique()

    )


    test_studies = set(

        test_df[
            "study_id"
        ].unique()

    )


    study_overlap = (

        train_studies
        .intersection(
            test_studies
        )

    )


    if study_overlap:

        raise ValueError(

            f"{validation_mode}: study leakage for "
            f"{held_out_group}: "
            f"{sorted(study_overlap)}"

        )


    if (
        validation_mode
        ==
        FAMILY_MODE
    ):

        family_overlap = (

            set(
                train_df[
                    "technology_family"
                ].unique()
            )

            .intersection(

                set(
                    test_df[
                        "technology_family"
                    ].unique()
                )

            )

        )


        if family_overlap:

            raise ValueError(

                f"{validation_mode}: family leakage for "
                f"{held_out_group}: "
                f"{sorted(family_overlap)}"

            )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()


    X_train = train_df[
        ALL_FEATURES
    ]


    y_train = train_df[
        TARGET
    ]


    X_test = test_df[
        ALL_FEATURES
    ]


    model.fit(

        X_train,
        y_train,

    )


    (

        predicted_accuracy,
        prediction_uncertainty,

    ) = get_random_forest_uncertainty(

        model,
        X_test,

    )


    predicted_df = test_df.copy()


    predicted_df[
        "predicted_accuracy"
    ] = predicted_accuracy


    predicted_df[
        "prediction_uncertainty"
    ] = prediction_uncertainty


    predicted_df[
        "confidence_lower_bound"
    ] = (

        predicted_df[
            "predicted_accuracy"
        ]

        -

        UNCERTAINTY_MULTIPLIER

        *
        predicted_df[
            "prediction_uncertainty"
        ]

    )


    predicted_df[
        "prediction_error_pp"
    ] = (

        predicted_df[
            "predicted_accuracy"
        ]

        -

        predicted_df[
            "accuracy"
        ]

    )


    predicted_df[
        "absolute_prediction_error_pp"
    ] = (

        predicted_df[
            "prediction_error_pp"
        ]
        .abs()

    )


    # --------------------------------------------------------
    # Evaluate each held-out physical device independently.
    #
    # This prevents a multi-device family/study test from
    # incorrectly selecting one accelerator configuration
    # across several different physical devices.
    # --------------------------------------------------------

    prediction_parts = []

    summary_rows = []


    held_out_group_rows = int(
        len(
            predicted_df
        )
    )


    for held_out_device in sorted(

        predicted_df[
            "device_id"
        ].unique()

    ):

        device_test_df = predicted_df[

            predicted_df[
                "device_id"
            ]
            ==
            held_out_device

        ].copy()


        (

            evaluated_predictions,
            summary_row,

        ) = evaluate_device_predictions(

            device_test_df=device_test_df,

            train_df=train_df,

            validation_mode=validation_mode,

            held_out_group=held_out_group,

            held_out_group_rows=held_out_group_rows,

            excluded_device_ids=excluded_device_ids,

        )


        prediction_parts.append(
            evaluated_predictions
        )


        summary_rows.append(
            summary_row
        )


    return (

        pd.concat(

            prediction_parts,
            ignore_index=True,

        ),

        pd.DataFrame(
            summary_rows
        ),

    )


# ============================================================
# Primary validation:
# STUDY-BLOCKED LEAVE-ONE-DEVICE-OUT
#
# Test only one device.
#
# Training removes:
#
#   held-out device
#   +
#   every sibling device from the same study
#
# This is the primary output written to the historical
# zero_shot_predictions.csv and zero_shot_summary.csv files.
# ============================================================

def run_primary_validation(
    df
):

    all_predictions = []

    all_summaries = []


    devices = sorted(

        df[
            "device_id"
        ].unique()

    )


    for held_out_device in devices:

        device_rows = df[

            df[
                "device_id"
            ]
            ==
            held_out_device

        ].copy()


        held_out_studies = (

            device_rows[
                "study_id"
            ]
            .unique()

        )


        if (
            len(
                held_out_studies
            )
            != 1
        ):

            raise ValueError(

                f"{held_out_device} maps to multiple "
                "study IDs."

            )


        held_out_study = str(
            held_out_studies[0]
        )


        test_df = device_rows


        train_df = df[

            df[
                "study_id"
            ]
            !=
            held_out_study

        ].copy()


        excluded_device_ids = sorted(

            df.loc[

                df[
                    "study_id"
                ]
                ==
                held_out_study,

                "device_id",

            ]
            .unique()

        )


        print()

        print(
            "========================================"
        )

        print(
            "PRIMARY HELD-OUT DEVICE:",
            held_out_device
        )

        print(
            "HELD-OUT STUDY:",
            held_out_study
        )

        print(
            "EXCLUDED SAME-STUDY DEVICES:",
            excluded_device_ids
        )

        print(
            "TRAINING DEVICES:",
            sorted(
                train_df[
                    "device_id"
                ].unique()
            )
        )

        print(
            "========================================"
        )


        (

            predictions,
            summary,

        ) = fit_predict_and_evaluate(

            train_df=train_df,

            test_df=test_df,

            validation_mode=PRIMARY_MODE,

            held_out_group=held_out_device,

            excluded_device_ids=excluded_device_ids,

        )


        all_predictions.append(
            predictions
        )


        all_summaries.append(
            summary
        )


    return (

        pd.concat(

            all_predictions,
            ignore_index=True,

        ),

        pd.concat(

            all_summaries,
            ignore_index=True,

        ),

    )


# ============================================================
# Leave-One-Study-Out
#
# All devices from one experimental study are held out at
# the same time. Each held-out device is scored separately.
# ============================================================

def run_study_validation(
    df
):

    all_predictions = []

    all_summaries = []


    studies = sorted(

        df[
            "study_id"
        ].unique()

    )


    for held_out_study in studies:

        test_df = df[

            df[
                "study_id"
            ]
            ==
            held_out_study

        ].copy()


        train_df = df[

            df[
                "study_id"
            ]
            !=
            held_out_study

        ].copy()


        excluded_device_ids = sorted(

            test_df[
                "device_id"
            ].unique()

        )


        print()

        print(
            "========================================"
        )

        print(
            "LEAVE-ONE-STUDY-OUT:",
            held_out_study
        )

        print(
            "TEST DEVICES:",
            excluded_device_ids
        )

        print(
            "========================================"
        )


        (

            predictions,
            summary,

        ) = fit_predict_and_evaluate(

            train_df=train_df,

            test_df=test_df,

            validation_mode=STUDY_MODE,

            held_out_group=held_out_study,

            excluded_device_ids=excluded_device_ids,

        )


        all_predictions.append(
            predictions
        )


        all_summaries.append(
            summary
        )


    return (

        pd.concat(

            all_predictions,
            ignore_index=True,

        ),

        pd.concat(

            all_summaries,
            ignore_index=True,

        ),

    )


# ============================================================
# Leave-One-Family-Out
#
# All devices in one technology family are held out.
#
# Extra safety:
# any study represented in the held-out family is also
# excluded from training. This prevents cross-family
# same-paper leakage if such a study is added in the future.
# ============================================================

def run_family_validation(
    df
):

    all_predictions = []

    all_summaries = []


    families = sorted(

        df[
            "technology_family"
        ].unique()

    )


    for held_out_family in families:

        test_df = df[

            df[
                "technology_family"
            ]
            ==
            held_out_family

        ].copy()


        held_out_studies = set(

            test_df[
                "study_id"
            ].unique()

        )


        train_df = df[

            (
                df[
                    "technology_family"
                ]
                !=
                held_out_family
            )

            &

            (
                ~df[
                    "study_id"
                ].isin(
                    held_out_studies
                )
            )

        ].copy()


        excluded_device_ids = sorted(

            df.loc[

                ~df.index.isin(
                    train_df.index
                ),

                "device_id",

            ]
            .unique()

        )


        print()

        print(
            "========================================"
        )

        print(
            "LEAVE-ONE-FAMILY-OUT:",
            held_out_family
        )

        print(
            "TEST DEVICES:",
            sorted(
                test_df[
                    "device_id"
                ].unique()
            )
        )

        print(
            "HELD-OUT STUDIES:",
            sorted(
                held_out_studies
            )
        )

        print(
            "========================================"
        )


        (

            predictions,
            summary,

        ) = fit_predict_and_evaluate(

            train_df=train_df,

            test_df=test_df,

            validation_mode=FAMILY_MODE,

            held_out_group=held_out_family,

            excluded_device_ids=excluded_device_ids,

        )


        all_predictions.append(
            predictions
        )


        all_summaries.append(
            summary
        )


    return (

        pd.concat(

            all_predictions,
            ignore_index=True,

        ),

        pd.concat(

            all_summaries,
            ignore_index=True,

        ),

    )


# ============================================================
# Aggregate one validation mode
# ============================================================

def summarize_validation_mode(
    validation_mode,
    predictions_df,
    summary_df,
):

    baseline_mean_regret = float(

        summary_df[
            "baseline_regret_pp"
        ].mean()

    )


    baseline_success_rate = float(

        100.0

        *
        summary_df[
            "baseline_near_optimal_success"
        ].mean()

    )


    baseline_exact_rate = float(

        100.0

        *
        summary_df[
            "baseline_exact_match"
        ].mean()

    )


    baseline_top3_rate = float(

        100.0

        *
        summary_df[
            "baseline_cost_aware_top3"
        ].mean()

    )


    confidence_mean_regret = float(

        summary_df[
            "confidence_regret_pp"
        ].mean()

    )


    confidence_success_rate = float(

        100.0

        *
        summary_df[
            "confidence_near_optimal_success"
        ].mean()

    )


    confidence_exact_rate = float(

        100.0

        *
        summary_df[
            "confidence_exact_match"
        ].mean()

    )


    confidence_top3_rate = float(

        100.0

        *
        summary_df[
            "confidence_cost_aware_top3"
        ].mean()

    )


    fallback_rate = float(

        100.0

        *
        summary_df[
            "confidence_fallback_used"
        ].mean()

    )


    overall_near = predictions_df[

        predictions_df[
            "is_true_near_optimal"
        ]

    ].copy()


    overall_near_mae = mean_absolute_error(

        overall_near[
            "accuracy"
        ],

        overall_near[
            "predicted_accuracy"
        ],

    )


    overall_near_rmse = np.sqrt(

        mean_squared_error(

            overall_near[
                "accuracy"
            ],

            overall_near[
                "predicted_accuracy"
            ],

        )

    )


    overall_uncertainty = (

        predictions_df[
            "prediction_uncertainty"
        ].to_numpy()

    )


    overall_absolute_error = (

        predictions_df[
            "absolute_prediction_error_pp"
        ].to_numpy()

    )


    if (

        np.std(
            overall_uncertainty
        ) > 0

        and

        np.std(
            overall_absolute_error
        ) > 0

    ):

        uncertainty_error_correlation = float(

            np.corrcoef(

                overall_uncertainty,
                overall_absolute_error,

            )[0, 1]

        )


    else:

        uncertainty_error_correlation = np.nan


    return {

        "validation_mode":
            validation_mode,

        "evaluated_devices":
            int(
                summary_df[
                    "held_out_device"
                ].nunique()
            ),

        "summary_rows":
            int(
                len(
                    summary_df
                )
            ),

        "baseline_mean_regret_pp":
            baseline_mean_regret,

        "baseline_near_optimal_success_pct":
            baseline_success_rate,

        "baseline_exact_cost_aware_match_pct":
            baseline_exact_rate,

        "baseline_cost_aware_top3_pct":
            baseline_top3_rate,

        "confidence_mean_regret_pp":
            confidence_mean_regret,

        "confidence_near_optimal_success_pct":
            confidence_success_rate,

        "confidence_exact_cost_aware_match_pct":
            confidence_exact_rate,

        "confidence_cost_aware_top3_pct":
            confidence_top3_rate,

        "confidence_fallback_rate_pct":
            fallback_rate,

        "near_region_mae_pp":
            float(
                overall_near_mae
            ),

        "near_region_rmse_pp":
            float(
                overall_near_rmse
            ),

        "uncertainty_error_correlation":
            (

                float(
                    uncertainty_error_correlation
                )

                if np.isfinite(
                    uncertainty_error_correlation
                )

                else np.nan

            ),

    }


# ============================================================
# Display one mode summary
# ============================================================

def print_mode_summary(
    title,
    summary_df,
    overview_row,
):

    print()

    print()

    print(
        "========================================"
    )

    print(
        title
    )

    print(
        "========================================"
    )


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

    ]


    print(

        summary_df[
            display_columns
        ].to_string(
            index=False
        )

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

        "Confidence-aware mean regret: "
        f"{overview_row['confidence_mean_regret_pp']:.3f} pp"

    )


    print(

        "Confidence-aware near-optimal success: "
        f"{overview_row['confidence_near_optimal_success_pct']:.1f}%"

    )


    print(

        "Near-region MAE: "
        f"{overview_row['near_region_mae_pp']:.3f} pp"

    )


    print(

        "Near-region RMSE: "
        f"{overview_row['near_region_rmse_pp']:.3f} pp"

    )


    correlation = overview_row[
        "uncertainty_error_correlation"
    ]


    if np.isfinite(
        correlation
    ):

        print(

            "Uncertainty/error correlation: "
            f"{correlation:.3f}"

        )


    else:

        print(

            "Uncertainty/error correlation: N/A"

        )


# ============================================================
# Load dataset + manifest
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


manifest_path = Path(
    MANIFEST_FILE
)


if not manifest_path.exists():

    raise FileNotFoundError(

        f"Missing manifest: "
        f"{MANIFEST_FILE}"

    )


with open(

    manifest_path,

    "r",

    encoding="utf-8",

) as f:

    manifest = json.load(
        f
    )


# ============================================================
# Validate manifest split policy
# ============================================================

split_rule = str(

    manifest.get(
        "split_rule",
        ""
    )

).upper()


if (

    "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT"
    not in split_rule

):

    raise ValueError(

        "Manifest must specify "
        "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT."

    )


if (

    "NEVER_RANDOM_ROW_SPLIT"
    not in split_rule

):

    raise ValueError(

        "Manifest must explicitly prohibit "
        "random row-level splitting."

    )


# ============================================================
# Validate manifest model-feature policy
# ============================================================

manifest_model_features = manifest.get(
    "model_input_features",
    []
)


if manifest_model_features:

    if (

        set(
            manifest_model_features
        )

        !=

        set(
            ALL_FEATURES
        )

    ):

        raise ValueError(

            "Manifest model_input_features do not match "
            "zero_shot_predictor.py ALL_FEATURES.\n"
            f"Manifest: {sorted(manifest_model_features)}\n"
            f"Predictor: {sorted(ALL_FEATURES)}"

        )


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [

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

    column

    for column
    in REQUIRED_COLUMNS

    if column
    not in df.columns

]


if missing_columns:

    raise ValueError(

        "ML dataset is missing columns:\n"
        f"{missing_columns}"

    )


# ============================================================
# Missing model-feature check
# ============================================================

missing_values = (

    df[
        ALL_FEATURES
    ]
    .isna()
    .sum()

)


missing_values = (

    missing_values[
        missing_values > 0
    ]

)


if not missing_values.empty:

    raise ValueError(

        "Model features contain missing values:\n"
        f"{missing_values}"

    )


# ============================================================
# Dataset grouping checks
# ============================================================

device_group_check = (

    df

    .groupby(
        "device_id"
    )

    .agg(

        studies=(
            "study_id",
            "nunique",
        ),

        families=(
            "technology_family",
            "nunique",
        ),

    )

)


if (

    (
        device_group_check[
            "studies"
        ]
        != 1
    ).any()

    or

    (
        device_group_check[
            "families"
        ]
        != 1
    ).any()

):

    raise ValueError(

        "Every device must map to exactly one study "
        "and one technology family."

    )


rows_per_device = (

    df[
        "device_id"
    ]
    .value_counts()

)


if (
    rows_per_device.nunique()
    != 1
):

    raise ValueError(

        "Current experiment expects an equal "
        "configuration count per device."

    )


CONFIGS_PER_DEVICE = int(

    rows_per_device.iloc[0]

)


DEVICE_COUNT = int(

    df[
        "device_id"
    ].nunique()

)


STUDY_COUNT = int(

    df[
        "study_id"
    ].nunique()

)


FAMILY_COUNT = int(

    df[
        "technology_family"
    ].nunique()

)


# ============================================================
# Experiment header
# ============================================================

print()

print(
    "STUDY-AWARE ZERO-SHOT + UNCERTAINTY EXPERIMENT"
)

print(
    "========================================"
)


print(
    "Rows:",
    len(
        df
    )
)


print(
    "Device profiles:",
    DEVICE_COUNT
)


print(
    "Distinct source studies:",
    STUDY_COUNT
)


print(
    "Technology families:",
    FAMILY_COUNT
)


print(
    "Configurations per device:",
    CONFIGS_PER_DEVICE
)


print()

print(
    "PRIMARY SPLIT:"
)


print(
    "  STUDY-BLOCKED LEAVE-ONE-DEVICE-OUT"
)


print()

print(
    "ADDITIONAL SPLITS:"
)


print(
    "  LEAVE-ONE-STUDY-OUT"
)


print(
    "  LEAVE-ONE-FAMILY-OUT"
)


print()

print(
    "Uncertainty:"
)


print(

    "  Random-Forest tree-prediction standard deviation"

)


print(

    "  Conservative score = predicted accuracy - "
    f"{UNCERTAINTY_MULTIPLIER:.1f} × tree disagreement"

)


print()

print(

    "Tree disagreement is an uncalibrated heuristic, "
    "NOT a statistical confidence interval."

)


print()

print(
    "MODEL FEATURES"
)

print(
    "----------------------------------------"
)


for feature in ALL_FEATURES:

    print(
        " +",
        feature
    )


print()

print(
    "NOT MODEL FEATURES"
)

print(
    "----------------------------------------"
)


for feature in NOT_MODEL_FEATURES:

    print(
        " -",
        feature
    )


# ============================================================
# Run the three validations
# ============================================================

(

    primary_predictions,
    primary_summary,

) = run_primary_validation(
    df
)


(

    study_predictions,
    study_summary,

) = run_study_validation(
    df
)


(

    family_predictions,
    family_summary,

) = run_family_validation(
    df
)


# ============================================================
# Aggregate summaries
# ============================================================

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


# ============================================================
# Save all outputs
# ============================================================

Path(
    "results/tables"
).mkdir(

    parents=True,
    exist_ok=True,

)


# Primary / backward-compatible output
primary_predictions.to_csv(

    PRIMARY_PREDICTIONS_FILE,
    index=False,

)


primary_summary.to_csv(

    PRIMARY_SUMMARY_FILE,
    index=False,

)


# Study-level output
study_predictions.to_csv(

    STUDY_PREDICTIONS_FILE,
    index=False,

)


study_summary.to_csv(

    STUDY_SUMMARY_FILE,
    index=False,

)


# Family-level output
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


# ============================================================
# Display final summaries
# ============================================================

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


# ============================================================
# Search efficiency
# ============================================================

print()

print(
    "SEARCH EFFICIENCY"
)

print(
    "----------------------------------------"
)


print(

    f"One selected validation instead of "
    f"{CONFIGS_PER_DEVICE} exhaustive unseen-device "
    "configuration simulations."

)


print(

    f"Expensive-search reduction: "
    f"{(1 - 1 / CONFIGS_PER_DEVICE) * 100:.2f}%"

)


print()

print(

    "This is a reduction in expensive unseen-device "
    "configuration evaluations, NOT total training, "
    "simulation, fabrication, energy or monetary cost."

)


# ============================================================
# Evidence limitation
# ============================================================

print()

print(
    "CURRENT EVIDENCE LIMITATION"
)

print(
    "----------------------------------------"
)


print(

    f"{DEVICE_COUNT} device profiles come from "
    f"{STUDY_COUNT} source studies across "
    f"{FAMILY_COUNT} technology families."

)


print(

    "TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt "
    "share one experimental study."

)


if "HfOx_01" in set(df["device_id"].astype(str)):

    print(

        "HfOx_01 uses its 2024 variability paper as the "
        "primary study_id; same-stack multilevel behavior "
        "support is retained separately in provenance."

    )


print(

    "The primary split blocks same-study siblings from "
    "training, preventing that direct leakage."

)


print(

    "The Random-Forest uncertainty remains an "
    "uncalibrated ensemble-disagreement heuristic."

)


print(

    "These experiments remain pilot validation, "
    "not proof of broad cross-device or cross-family "
    "physical generalization."

)


# ============================================================
# Saved files
# ============================================================

print()

print(
    "SAVED OUTPUTS"
)

print(
    "----------------------------------------"
)


print(
    PRIMARY_PREDICTIONS_FILE
)


print(
    PRIMARY_SUMMARY_FILE
)


print(
    STUDY_PREDICTIONS_FILE
)


print(
    STUDY_SUMMARY_FILE
)


print(
    FAMILY_PREDICTIONS_FILE
)


print(
    FAMILY_SUMMARY_FILE
)


print(
    VALIDATION_OVERVIEW_FILE
)
