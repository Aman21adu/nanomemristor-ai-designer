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

PREDICTIONS_FILE = (
    "results/tables/"
    "zero_shot_predictions.csv"
)

SUMMARY_FILE = (
    "results/tables/"
    "zero_shot_summary.csv"
)


# ============================================================
# Experiment settings
# ============================================================

ACCURACY_TOLERANCE_PP = 0.5

RANDOM_STATE = 42


# ============================================================
# Uncertainty heuristic
# ============================================================
#
# Random Forest consists of many decision trees.
#
# If the trees strongly disagree on the predicted accuracy,
# we treat that as larger model uncertainty.
#
# uncertainty =
#     standard deviation of individual tree predictions
#
#
# Conservative lower bound:
#
#     predicted accuracy
#     -
#     UNCERTAINTY_MULTIPLIER * tree disagreement
#
#
# IMPORTANT:
#
# This is NOT a calibrated statistical confidence interval.
#
# It is an ensemble-disagreement heuristic that we are
# experimentally testing.
# ============================================================

UNCERTAINTY_MULTIPLIER = 1.0


# ============================================================
# Prediction target
# ============================================================

TARGET = "accuracy"


# ============================================================
# Model features
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
    + CATEGORICAL_FEATURES

)


# ============================================================
# Explicitly excluded from AI inputs
# ============================================================

NOT_MODEL_FEATURES = [

    "device_id",
    "technology_family",

    "device_state_count_status",
    "parameter_source",
    "precision_basis",

    "accuracy_loss",

    "relative_hardware_cost_proxy",

]


# ============================================================
# Configuration identity
# ============================================================

def configuration_key(row):

    return (

        int(
            row["crossbar_size"]
        ),

        int(
            row["requested_weight_bits"]
        ),

        int(
            row["adc_bits"]
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
        - tolerance_pp

    )


    candidates = group[

        group[
            score_column
        ]
        >= threshold

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
# Same rule as select_optimal.py.
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


# ============================================================
# Standard predicted / actual cost-aware selection
# ============================================================

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
# ============================================================
#
# Step 1:
# Find the predicted best accuracy.
#
# Step 2:
# Define the same predicted 0.5-pp requirement.
#
# Step 3:
# A configuration is considered "safe" only if:
#
# confidence_lower_bound >= predicted_best - tolerance
#
#
# Step 4:
# Among safe candidates use the same hardware-cost rule.
#
#
# If no safe candidate exists:
# do NOT invent a result.
#
# We fall back to the ordinary predictor recommendation and
# explicitly mark confidence_fallback_used = True.
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
        - tolerance_pp

    )


    safe_candidates = group[

        group[
            "confidence_lower_bound"
        ]
        >= acceptance_threshold

    ].copy()


    if not safe_candidates.empty:

        ranked = sort_cost_aware(

            safe_candidates,

            "confidence_lower_bound",

        )


        chosen = (
            ranked.iloc[0]
        )


        fallback_used = False


    else:

        # ----------------------------------------------------
        # No configuration satisfies the conservative rule.
        #
        # Keep experiment executable but explicitly report
        # that confidence-aware selection could not safely
        # identify a candidate.
        # ----------------------------------------------------

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
# Raw exhaustive maximum-accuracy configuration
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


    return (
        ranked.iloc[0]
    )


# ============================================================
# Same configuration?
# ============================================================

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


# ============================================================
# Cost-aware rank inside TRUE near-optimal region
# ============================================================

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


# ============================================================
# Candidate-set precision / recall
# ============================================================

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


    if len(predicted_keys) > 0:

        precision = (

            len(overlap)
            / len(predicted_keys)

        )

    else:

        precision = 0.0


    if len(true_keys) > 0:

        recall = (

            len(overlap)
            / len(true_keys)

        )

    else:

        recall = 0.0


    return (

        float(
            precision
        ),

        float(
            recall
        ),

        int(
            len(overlap)
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
    Return:

        mean tree prediction
        tree-prediction standard deviation

    The standard deviation is used only as an ensemble
    disagreement heuristic.

    It is NOT a calibrated confidence interval.
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
# Load dataset
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# Load manifest
# ============================================================

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
# Validate split rule
# ============================================================

split_rule = str(

    manifest.get(
        "split_rule",
        ""
    )

).upper()


if (

    "LEAVE_ONE_DEVICE_OUT"
    not in split_rule

):

    raise ValueError(

        "Manifest must specify "
        "LEAVE_ONE_DEVICE_OUT."

    )


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [

    "device_id",
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
# Missing model feature check
# ============================================================

missing_values = (

    df[
        ALL_FEATURES
    ]

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


# ============================================================
# Device structure
# ============================================================

devices = sorted(

    df[
        "device_id"
    ].unique()

)


rows_per_device = (

    df[
        "device_id"
    ]
    .value_counts()

)


if rows_per_device.nunique() != 1:

    raise ValueError(

        "Current experiment expects an equal "
        "configuration count per device."

    )


CONFIGS_PER_DEVICE = int(

    rows_per_device.iloc[0]

)


# ============================================================
# Current family structure
# ============================================================

device_family_pairs = (

    df[

        [
            "device_id",
            "technology_family",
        ]

    ]

    .drop_duplicates()

)


family_counts = (

    device_family_pairs[
        "technology_family"
    ]
    .value_counts()

)


one_device_per_family = bool(

    (
        family_counts == 1
    ).all()

)


# ============================================================
# Experiment header
# ============================================================

print()

print(
    "ZERO-SHOT + UNCERTAINTY EXPERIMENT"
)

print(
    "========================================"
)


print(
    "Rows:",
    len(df)
)


print(
    "Physical devices:",
    df[
        "device_id"
    ].nunique()
)


print(
    "Technology families:",
    df[
        "technology_family"
    ].nunique()
)


print(
    "Configurations per device:",
    CONFIGS_PER_DEVICE
)


print()

print(
    "Uncertainty method:"
)


print(

    "  Standard deviation across "
    "Random-Forest tree predictions"

)


print(

    "  Conservative score = "
    "predicted accuracy - "
    f"{UNCERTAINTY_MULTIPLIER:.1f} × tree disagreement"

)


print()

print(
    "IMPORTANT:"
)


print(

    "This is an ensemble-disagreement heuristic, "
    "NOT a calibrated confidence interval."

)


if one_device_per_family:

    print()

    print(

        "Each current family contains one device, "
        "so device-holdout and family-holdout "
        "are currently equivalent."

    )


# ============================================================
# Display feature policy
# ============================================================

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
# Containers
# ============================================================

all_predictions = []

summary_rows = []


# ============================================================
# Leave-One-Device-Out
# ============================================================

for held_out_device in devices:

    train_df = df[

        df[
            "device_id"
        ]
        != held_out_device

    ].copy()


    test_df = df[

        df[
            "device_id"
        ]
        == held_out_device

    ].copy()


    held_out_family = str(

        test_df[
            "technology_family"
        ].iloc[0]

    )


    print()

    print(
        "========================================"
    )

    print(

        "HELD-OUT DEVICE:",
        held_out_device

    )

    print(

        "HELD-OUT FAMILY:",
        held_out_family

    )

    print(
        "========================================"
    )


    print(

        "Training devices:",

        sorted(

            train_df[
                "device_id"
            ].unique()

        )

    )


    # ========================================================
    # Matrices
    # ========================================================

    X_train = train_df[
        ALL_FEATURES
    ]


    y_train = train_df[
        TARGET
    ]


    X_test = test_df[
        ALL_FEATURES
    ]


    y_test = test_df[
        TARGET
    ]


    # ========================================================
    # Preprocessor
    # ========================================================

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


    # ========================================================
    # Random Forest
    # ========================================================

    regressor = RandomForestRegressor(

        n_estimators=500,

        random_state=RANDOM_STATE,

        min_samples_leaf=2,

        n_jobs=-1,

    )


    model = Pipeline(

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


    # ========================================================
    # Train
    # ========================================================

    model.fit(

        X_train,

        y_train,

    )


    # ========================================================
    # Obtain mean prediction + ensemble disagreement
    # ========================================================

    (

        predicted_accuracy,
        prediction_uncertainty,

    ) = get_random_forest_uncertainty(

        model,

        X_test,

    )


    test_df[
        "predicted_accuracy"
    ] = predicted_accuracy


    test_df[
        "prediction_uncertainty"
    ] = prediction_uncertainty


    test_df[
        "confidence_lower_bound"
    ] = (

        test_df[
            "predicted_accuracy"
        ]

        -

        UNCERTAINTY_MULTIPLIER

        * test_df[
            "prediction_uncertainty"
        ]

    )


    test_df[
        "prediction_error_pp"
    ] = (

        test_df[
            "predicted_accuracy"
        ]

        -

        test_df[
            "accuracy"
        ]

    )


    test_df[
        "absolute_prediction_error_pp"
    ] = (

        test_df[
            "prediction_error_pp"
        ]
        .abs()

    )


    test_df[
        "held_out_device"
    ] = held_out_device


    test_df[
        "held_out_family"
    ] = held_out_family


    # ========================================================
    # Global prediction metrics
    # ========================================================

    global_mae = mean_absolute_error(

        y_test,

        predicted_accuracy,

    )


    global_rmse = np.sqrt(

        mean_squared_error(

            y_test,

            predicted_accuracy,

        )

    )


    global_r2 = r2_score(

        y_test,

        predicted_accuracy,

    )


    # ========================================================
    # Raw exhaustive best
    # ========================================================

    raw_best = get_raw_best_configuration(
        test_df
    )


    true_best_accuracy = float(

        raw_best[
            "accuracy"
        ]

    )


    # ========================================================
    # True cost-aware optimum
    # ========================================================

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

        >= true_threshold

    )


    # ========================================================
    # Near-optimal-region prediction metrics
    # ========================================================

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


    # ========================================================
    # BASELINE zero-shot recommendation
    # ========================================================

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


    # ========================================================
    # Candidate identification quality
    # ========================================================

    (

        candidate_precision,

        candidate_recall,

        candidate_overlap,

    ) = candidate_overlap_metrics(

        true_candidates,

        predicted_candidates,

    )


    # ========================================================
    # CONFIDENCE-AWARE recommendation
    # ========================================================

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


    # ========================================================
    # Evaluate BASELINE recommendation
    # ========================================================

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
        - baseline_actual_accuracy

    )


    baseline_success = bool(

        baseline_regret
        <= ACCURACY_TOLERANCE_PP

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

        and baseline_cost_rank <= 3

    )


    baseline_top5 = bool(

        np.isfinite(
            baseline_cost_rank
        )

        and baseline_cost_rank <= 5

    )


    # ========================================================
    # Was baseline recommendation considered safe?
    # ========================================================

    baseline_passes_confidence_gate = bool(

        baseline_lower_bound
        >= confidence_acceptance_threshold

    )


    # ========================================================
    # Evaluate CONFIDENCE-AWARE recommendation
    # ========================================================

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
        - confidence_actual_accuracy

    )


    confidence_success = bool(

        confidence_regret
        <= ACCURACY_TOLERANCE_PP

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

        and confidence_cost_rank <= 3

    )


    confidence_top5 = bool(

        np.isfinite(
            confidence_cost_rank
        )

        and confidence_cost_rank <= 5

    )


    # ========================================================
    # Safe candidate quality
    # ========================================================

    (

        safe_precision,

        safe_recall,

        safe_overlap,

    ) = candidate_overlap_metrics(

        true_candidates,

        safe_candidates,

    )


    # ========================================================
    # Uncertainty diagnostic
    #
    # Does larger tree disagreement tend to correspond to
    # larger absolute prediction error?
    #
    # This is descriptive only.
    # ========================================================

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


    # ========================================================
    # Search reduction
    # ========================================================

    search_reduction_pct = (

        1.0

        -

        (
            1
            / len(test_df)
        )

    ) * 100.0


    # ========================================================
    # Display fold results
    # ========================================================

    print()

    print(
        "PREDICTION QUALITY"
    )

    print(
        "----------------------------------------"
    )


    print(

        f"Global MAE: "
        f"{global_mae:.3f} pp"

    )


    print(

        f"Global RMSE: "
        f"{global_rmse:.3f} pp"

    )


    print(

        f"Global R2: "
        f"{global_r2:.3f}"

    )


    print(

        f"Near-region MAE: "
        f"{near_region_mae:.3f} pp"

    )


    print(

        f"Near-region RMSE: "
        f"{near_region_rmse:.3f} pp"

    )


    print()

    print(
        "UNCERTAINTY DIAGNOSTIC"
    )

    print(
        "----------------------------------------"
    )


    print(

        f"Mean tree disagreement: "
        f"{test_df['prediction_uncertainty'].mean():.3f} pp"

    )


    print(

        f"Uncertainty/error correlation: "
        f"{uncertainty_error_correlation:.3f}"

    )


    print(

        f"Safe candidate count: "
        f"{len(safe_candidates)}"

    )


    print(

        f"Safe-candidate precision: "
        f"{100.0 * safe_precision:.1f}%"

    )


    print(

        f"Safe-candidate recall: "
        f"{100.0 * safe_recall:.1f}%"

    )


    # ========================================================
    # Ground truth
    # ========================================================

    print()

    print(
        "GROUND TRUTH"
    )

    print(
        "----------------------------------------"
    )


    print(

        f"Raw best: "
        f"{int(raw_best['crossbar_size'])}x"
        f"{int(raw_best['crossbar_size'])}, "
        f"W{int(raw_best['requested_weight_bits'])}, "
        f"ADC{int(raw_best['adc_bits'])} "
        f"-> {true_best_accuracy:.2f}%"

    )


    print(

        f"Cost-aware optimum: "
        f"{int(true_cost_aware['crossbar_size'])}x"
        f"{int(true_cost_aware['crossbar_size'])}, "
        f"W{int(true_cost_aware['requested_weight_bits'])}, "
        f"ADC{int(true_cost_aware['adc_bits'])} "
        f"-> {float(true_cost_aware['accuracy']):.2f}%"

    )


    # ========================================================
    # Baseline recommendation
    # ========================================================

    print()

    print(
        "BASELINE ZERO-SHOT RECOMMENDATION"
    )

    print(
        "----------------------------------------"
    )


    print(

        f"Config: "
        f"{int(baseline_recommended['crossbar_size'])}x"
        f"{int(baseline_recommended['crossbar_size'])}, "
        f"W{int(baseline_recommended['requested_weight_bits'])}, "
        f"ADC{int(baseline_recommended['adc_bits'])}"

    )


    print(

        f"Predicted accuracy: "
        f"{baseline_predicted_accuracy:.2f}%"

    )


    print(

        f"Tree disagreement: "
        f"{baseline_uncertainty:.3f} pp"

    )


    print(

        f"Conservative lower bound: "
        f"{baseline_lower_bound:.2f}%"

    )


    print(

        f"Passes confidence gate: "
        f"{baseline_passes_confidence_gate}"

    )


    print(

        f"Actual accuracy: "
        f"{baseline_actual_accuracy:.2f}%"

    )


    print(

        f"Regret: "
        f"{baseline_regret:.2f} pp"

    )


    print(

        f"Near-optimal success: "
        f"{baseline_success}"

    )


    # ========================================================
    # Confidence-aware recommendation
    # ========================================================

    print()

    print(
        "CONFIDENCE-AWARE RECOMMENDATION"
    )

    print(
        "----------------------------------------"
    )


    print(

        f"Config: "
        f"{int(confidence_recommended['crossbar_size'])}x"
        f"{int(confidence_recommended['crossbar_size'])}, "
        f"W{int(confidence_recommended['requested_weight_bits'])}, "
        f"ADC{int(confidence_recommended['adc_bits'])}"

    )


    print(

        f"Predicted accuracy: "
        f"{confidence_predicted_accuracy:.2f}%"

    )


    print(

        f"Tree disagreement: "
        f"{confidence_uncertainty:.3f} pp"

    )


    print(

        f"Conservative lower bound: "
        f"{confidence_lower_bound:.2f}%"

    )


    print(

        f"Fallback used: "
        f"{confidence_fallback_used}"

    )


    print(

        f"Actual accuracy: "
        f"{confidence_actual_accuracy:.2f}%"

    )


    print(

        f"Regret: "
        f"{confidence_regret:.2f} pp"

    )


    print(

        f"Near-optimal success: "
        f"{confidence_success}"

    )


    if np.isfinite(
        confidence_cost_rank
    ):

        print(

            f"Cost-aware rank: "
            f"{int(confidence_cost_rank)}/"
            f"{len(true_candidates)}"

        )

    else:

        print(

            "Cost-aware rank: N/A "
            "(outside true near-optimal region)"

        )


    # ========================================================
    # Summary row
    # ========================================================

    summary_rows.append({

        "held_out_device":
            held_out_device,

        "held_out_family":
            held_out_family,

        "training_devices":
            int(
                train_df[
                    "device_id"
                ].nunique()
            ),

        "training_rows":
            int(
                len(train_df)
            ),

        "held_out_rows":
            int(
                len(test_df)
            ),


        # ----------------------------------------------------
        # Prediction metrics
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Uncertainty diagnostic
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # True optimum
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Baseline recommendation
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Confidence-aware recommendation
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        "search_reduction_pct":
            float(
                search_reduction_pct
            ),

    })


    all_predictions.append(
        test_df
    )


# ============================================================
# Combine all folds
# ============================================================

predictions_df = pd.concat(

    all_predictions,

    ignore_index=True,

)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Save files
# ============================================================

Path(
    "results/tables"
).mkdir(

    parents=True,

    exist_ok=True,

)


predictions_df.to_csv(

    PREDICTIONS_FILE,

    index=False,

)


summary_df.to_csv(

    SUMMARY_FILE,

    index=False,

)


# ============================================================
# Overall BASELINE results
# ============================================================

baseline_mean_regret = float(

    summary_df[
        "baseline_regret_pp"
    ].mean()

)


baseline_success_rate = (

    100.0

    * summary_df[
        "baseline_near_optimal_success"
    ].mean()

)


baseline_exact_rate = (

    100.0

    * summary_df[
        "baseline_exact_match"
    ].mean()

)


baseline_top3_rate = (

    100.0

    * summary_df[
        "baseline_cost_aware_top3"
    ].mean()

)


# ============================================================
# Overall CONFIDENCE-AWARE results
# ============================================================

confidence_mean_regret = float(

    summary_df[
        "confidence_regret_pp"
    ].mean()

)


confidence_success_rate = (

    100.0

    * summary_df[
        "confidence_near_optimal_success"
    ].mean()

)


confidence_exact_rate = (

    100.0

    * summary_df[
        "confidence_exact_match"
    ].mean()

)


confidence_top3_rate = (

    100.0

    * summary_df[
        "confidence_cost_aware_top3"
    ].mean()

)


fallback_rate = (

    100.0

    * summary_df[
        "confidence_fallback_used"
    ].mean()

)


# ============================================================
# Overall near-region prediction
# ============================================================

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


# ============================================================
# Overall uncertainty diagnostic
# ============================================================

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

    overall_uncertainty_error_correlation = float(

        np.corrcoef(

            overall_uncertainty,

            overall_absolute_error,

        )[0, 1]

    )

else:

    overall_uncertainty_error_correlation = np.nan


# ============================================================
# Final summary
# ============================================================

print()

print()

print(
    "========================================"
)

print(
    "OVERALL ZERO-SHOT + UNCERTAINTY RESULTS"
)

print(
    "========================================"
)


DISPLAY_COLUMNS = [

    "held_out_device",

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

    "confidence_fallback_used",

]


print(

    summary_df[

        DISPLAY_COLUMNS

    ].to_string(
        index=False
    )

)


# ============================================================
# Prediction quality
# ============================================================

print()

print(
    "NEAR-OPTIMAL REGION PREDICTION"
)

print(
    "----------------------------------------"
)


print(

    f"Overall near-region MAE: "
    f"{overall_near_mae:.3f} pp"

)


print(

    f"Overall near-region RMSE: "
    f"{overall_near_rmse:.3f} pp"

)


# ============================================================
# Uncertainty quality
# ============================================================

print()

print(
    "UNCERTAINTY HEURISTIC"
)

print(
    "----------------------------------------"
)


print(

    f"Overall uncertainty/error correlation: "
    f"{overall_uncertainty_error_correlation:.3f}"

)


print(

    f"Confidence fallback rate: "
    f"{fallback_rate:.1f}%"

)


print()

print(

    "Higher positive correlation means tree disagreement "
    "tends to increase when prediction error increases."

)


print()

print(

    "This does NOT make the uncertainty estimate "
    "statistically calibrated."

)


# ============================================================
# Baseline versus confidence-aware
# ============================================================

print()

print(
    "BASELINE RECOMMENDATION"
)

print(
    "----------------------------------------"
)


print(

    f"Mean regret: "
    f"{baseline_mean_regret:.3f} pp"

)


print(

    f"Near-optimal success: "
    f"{baseline_success_rate:.1f}%"

)


print(

    f"Exact cost-aware match: "
    f"{baseline_exact_rate:.1f}%"

)


print(

    f"Cost-aware Top-3: "
    f"{baseline_top3_rate:.1f}%"

)


print()

print(
    "CONFIDENCE-AWARE RECOMMENDATION"
)

print(
    "----------------------------------------"
)


print(

    f"Mean regret: "
    f"{confidence_mean_regret:.3f} pp"

)


print(

    f"Near-optimal success: "
    f"{confidence_success_rate:.1f}%"

)


print(

    f"Exact cost-aware match: "
    f"{confidence_exact_rate:.1f}%"

)


print(

    f"Cost-aware Top-3: "
    f"{confidence_top3_rate:.1f}%"

)


# ============================================================
# Search reduction
# ============================================================

print()

print(
    "SEARCH EFFICIENCY"
)

print(
    "----------------------------------------"
)


print(

    f"One validation instead of "
    f"{CONFIGS_PER_DEVICE} exhaustive simulations"

)


print(

    f"Expensive-search reduction: "
    f"{(1 - 1 / CONFIGS_PER_DEVICE) * 100:.2f}%"

)


print()

print(

    "This is reduction in expensive unseen-device "
    "simulation/validation evaluations, NOT total "
    "computational cost."

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

    "Only 4 independent physical devices are available."

)


print(

    "Each fold therefore trains on only 3 devices."

)


print(

    "The uncertainty method is currently an "
    "uncalibrated ensemble-disagreement heuristic."

)


print(

    "The experiment should be treated as pilot validation, "
    "not proof of broad cross-device generalization."

)


# ============================================================
# Saved files
# ============================================================

print()

print(
    "Saved predictions to:"
)

print(
    PREDICTIONS_FILE
)


print()

print(
    "Saved summary to:"
)

print(
    SUMMARY_FILE
)