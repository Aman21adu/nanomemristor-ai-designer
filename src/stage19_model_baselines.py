from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ============================================================
# Files
# ============================================================

DATA_FILE = Path("results/tables/ml_dataset.csv")

OUTPUT_DIR = Path("results/tables")
OUTPUT_BY_DEVICE = OUTPUT_DIR / "stage19_model_baselines_by_device.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "stage19_model_baselines_summary.csv"
OUTPUT_PREDICTIONS = OUTPUT_DIR / "stage19_model_baseline_predictions.csv"
OUTPUT_SCOPE = OUTPUT_DIR / "stage19_model_baselines_scope.json"


# ============================================================
# Features
#
# Same full device-aware feature set used by the current study.
# IDs/family labels are NOT model features.
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

TARGET = "accuracy"

RANDOM_STATE = 42


# ============================================================
# Preprocessing
# ============================================================

def make_preprocessor(scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipe,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# Models
#
# These are baseline-screening models, not a claim that each
# model has been exhaustively hyperparameter optimized.
# ============================================================

def build_models():
    return {
        "RIDGE": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        scale_numeric=True
                    ),
                ),
                (
                    "model",
                    Ridge(alpha=1.0),
                ),
            ]
        ),

        "RANDOM_FOREST": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        scale_numeric=False
                    ),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "EXTRA_TREES": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        scale_numeric=False
                    ),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        random_state=RANDOM_STATE,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "GRADIENT_BOOSTING": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        scale_numeric=False
                    ),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                        loss="squared_error",
                    ),
                ),
            ]
        ),

        "MLP": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        scale_numeric=True
                    ),
                ),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=1500,
                        early_stopping=True,
                        validation_fraction=0.15,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# ============================================================
# Study-blocked held-out-device evaluation
# ============================================================

def main():

    df = pd.read_csv(DATA_FILE)

    required = (
        ["device_id", "study_id", "technology_family", TARGET]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    devices = sorted(
        df["device_id"].astype(str).unique()
    )

    print("=" * 70)
    print("STAGE 19 — MODEL BASELINE COMPARISON")
    print("=" * 70)
    print("Rows:", len(df))
    print("Devices:", len(devices))
    print(
        "Studies:",
        df["study_id"].nunique(),
    )
    print(
        "Families:",
        df["technology_family"].nunique(),
    )
    print()
    print(
        "Split: hold out one device and exclude "
        "ALL profiles from the same study."
    )
    print(
        "No random row split is used."
    )
    print("=" * 70)

    by_device_rows = []
    prediction_rows = []

    for held_out_device in devices:

        test_df = df[
            df["device_id"].astype(str)
            == held_out_device
        ].copy()

        if test_df.empty:
            continue

        held_out_study = str(
            test_df["study_id"].iloc[0]
        )

        train_df = df[
            df["study_id"].astype(str)
            != held_out_study
        ].copy()

        print()
        print(
            f"Held out: {held_out_device}"
        )
        print(
            f"Study:    {held_out_study}"
        )
        print(
            "Training devices:",
            train_df["device_id"].nunique(),
        )
        print(
            "Training rows:",
            len(train_df),
        )
        print(
            "Test rows:",
            len(test_df),
        )

        X_train = train_df[
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        ]

        y_train = train_df[TARGET].astype(float)

        X_test = test_df[
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        ]

        y_test = test_df[TARGET].astype(float)

        actual_best_accuracy = float(
            y_test.max()
        )

        models = build_models()

        for model_name, model in models.items():

            print(
                f"  Training {model_name}..."
            )

            model.fit(
                X_train,
                y_train,
            )

            pred = model.predict(X_test)

            mae = mean_absolute_error(
                y_test,
                pred,
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    pred,
                )
            )

            predicted_best_position = int(
                np.argmax(pred)
            )

            selected_actual_accuracy = float(
                y_test.iloc[
                    predicted_best_position
                ]
            )

            selected_predicted_accuracy = float(
                pred[
                    predicted_best_position
                ]
            )

            regret_pp = (
                actual_best_accuracy
                - selected_actual_accuracy
            )

            selected_row = test_df.iloc[
                predicted_best_position
            ]

            by_device_rows.append(
                {
                    "method": model_name,
                    "held_out_device":
                        held_out_device,
                    "held_out_study":
                        held_out_study,
                    "training_devices":
                        train_df[
                            "device_id"
                        ].nunique(),
                    "training_rows":
                        len(train_df),
                    "test_rows":
                        len(test_df),
                    "candidate_mae_pp":
                        float(mae),
                    "candidate_rmse_pp":
                        float(rmse),
                    "actual_best_accuracy":
                        actual_best_accuracy,
                    "selected_actual_accuracy":
                        selected_actual_accuracy,
                    "selected_predicted_accuracy":
                        selected_predicted_accuracy,
                    "regret_pp":
                        float(regret_pp),
                    "selected_crossbar_size":
                        selected_row[
                            "crossbar_size"
                        ],
                    "selected_weight_bits":
                        selected_row[
                            "requested_weight_bits"
                        ],
                    "selected_adc_bits":
                        selected_row[
                            "adc_bits"
                        ],
                }
            )

            for i, (_, row) in enumerate(
                test_df.iterrows()
            ):
                prediction_rows.append(
                    {
                        "method":
                            model_name,
                        "held_out_device":
                            held_out_device,
                        "held_out_study":
                            held_out_study,
                        "crossbar_size":
                            row[
                                "crossbar_size"
                            ],
                        "requested_weight_bits":
                            row[
                                "requested_weight_bits"
                            ],
                        "adc_bits":
                            row[
                                "adc_bits"
                            ],
                        "actual_accuracy":
                            float(
                                row[TARGET]
                            ),
                        "predicted_accuracy":
                            float(pred[i]),
                    }
                )

    by_device = pd.DataFrame(
        by_device_rows
    )

    predictions = pd.DataFrame(
        prediction_rows
    )

    summary_rows = []

    for method, g in by_device.groupby(
        "method"
    ):

        regrets = g["regret_pp"]

        summary_rows.append(
            {
                "method": method,
                "devices": len(g),
                "mean_candidate_mae_pp":
                    g[
                        "candidate_mae_pp"
                    ].mean(),
                "median_candidate_mae_pp":
                    g[
                        "candidate_mae_pp"
                    ].median(),
                "mean_candidate_rmse_pp":
                    g[
                        "candidate_rmse_pp"
                    ].mean(),
                "median_candidate_rmse_pp":
                    g[
                        "candidate_rmse_pp"
                    ].median(),
                "mean_regret_pp":
                    regrets.mean(),
                "median_regret_pp":
                    regrets.median(),
                "max_regret_pp":
                    regrets.max(),
                "success_rate_le_0_5pp":
                    float(
                        (
                            regrets <= 0.5
                        ).mean()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "mean_candidate_mae_pp",
            "mean_candidate_rmse_pp",
        ]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_device.to_csv(
        OUTPUT_BY_DEVICE,
        index=False,
    )

    predictions.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    scope = {
        "stage": 19,
        "name":
            "Study-blocked ML model baseline comparison",
        "rows": int(len(df)),
        "devices": int(
            df["device_id"].nunique()
        ),
        "studies": int(
            df["study_id"].nunique()
        ),
        "families": int(
            df[
                "technology_family"
            ].nunique()
        ),
        "split_rule":
            "Hold out one device and exclude every profile sharing the held-out study_id.",
        "random_row_split_used": False,
        "models": [
            "RIDGE",
            "RANDOM_FOREST",
            "EXTRA_TREES",
            "GRADIENT_BOOSTING",
            "MLP",
        ],
        "important_note":
            "This is a baseline screening experiment with fixed reasonable hyperparameters, not exhaustive hyperparameter optimization.",
    }

    OUTPUT_SCOPE.write_text(
        json.dumps(
            scope,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:",
        OUTPUT_SUMMARY,
    )


if __name__ == "__main__":
    main()
