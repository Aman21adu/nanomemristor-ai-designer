from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

DATA_FILE = Path("results/tables/ml_dataset.csv")
OUTPUT_DIR = Path("results/tables")

OUT_DEVICE = OUTPUT_DIR / "stage20_seed_stability_by_device.csv"
OUT_SEED = OUTPUT_DIR / "stage20_seed_stability_by_seed.csv"
OUT_SUMMARY = OUTPUT_DIR / "stage20_seed_stability_summary.csv"
OUT_SCOPE = OUTPUT_DIR / "stage20_seed_stability_scope.json"

SEEDS = [7, 21, 42, 84, 123]

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


def make_preprocessor():
    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipe = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
            (
                "numeric",
                numeric_pipe,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipe,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_models(seed):
    return {
        "RANDOM_FOREST": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "EXTRA_TREES": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "GRADIENT_BOOSTING": Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        loss="squared_error",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def main():

    df = pd.read_csv(DATA_FILE)

    devices = sorted(
        df["device_id"].astype(str).unique()
    )

    print("=" * 72)
    print("STAGE 20 — RANDOM-SEED STABILITY")
    print("=" * 72)

    print("Rows:", len(df))
    print("Devices:", df["device_id"].nunique())
    print("Studies:", df["study_id"].nunique())
    print("Families:", df["technology_family"].nunique())
    print("Seeds:", SEEDS)

    print()
    print(
        "Split: held-out device with all same-study "
        "profiles excluded from training."
    )
    print("No random row split.")
    print("=" * 72)

    rows = []

    feature_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    for seed in SEEDS:

        print()
        print("#" * 72)
        print("SEED:", seed)
        print("#" * 72)

        for device in devices:

            test_df = df[
                df["device_id"].astype(str)
                == device
            ].copy()

            held_study = str(
                test_df["study_id"].iloc[0]
            )

            train_df = df[
                df["study_id"].astype(str)
                != held_study
            ].copy()

            X_train = train_df[
                feature_columns
            ]

            y_train = train_df[
                TARGET
            ].astype(float)

            X_test = test_df[
                feature_columns
            ]

            y_test = test_df[
                TARGET
            ].astype(float)

            actual_best = float(
                y_test.max()
            )

            models = build_models(seed)

            for model_name, model in models.items():

                print(
                    f"Seed {seed} | "
                    f"{device} | "
                    f"{model_name}"
                )

                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(
                    X_test
                )

                mae = float(
                    mean_absolute_error(
                        y_test,
                        pred,
                    )
                )

                rmse = float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            pred,
                        )
                    )
                )

                best_idx = int(
                    np.argmax(pred)
                )

                selected_actual = float(
                    y_test.iloc[
                        best_idx
                    ]
                )

                regret = float(
                    actual_best
                    - selected_actual
                )

                rows.append(
                    {
                        "seed": seed,
                        "method": model_name,
                        "held_out_device": device,
                        "held_out_study":
                            held_study,
                        "training_devices":
                            int(
                                train_df[
                                    "device_id"
                                ].nunique()
                            ),
                        "training_rows":
                            len(train_df),
                        "candidate_mae_pp":
                            mae,
                        "candidate_rmse_pp":
                            rmse,
                        "regret_pp":
                            regret,
                        "success_le_0_5pp":
                            int(
                                regret <= 0.5
                            ),
                    }
                )

    device_results = pd.DataFrame(
        rows
    )

    seed_rows = []

    for (method, seed), g in (
        device_results.groupby(
            ["method", "seed"]
        )
    ):

        seed_rows.append(
            {
                "method": method,
                "seed": seed,
                "devices": len(g),
                "mean_mae_pp":
                    g[
                        "candidate_mae_pp"
                    ].mean(),
                "mean_rmse_pp":
                    g[
                        "candidate_rmse_pp"
                    ].mean(),
                "mean_regret_pp":
                    g[
                        "regret_pp"
                    ].mean(),
                "max_regret_pp":
                    g[
                        "regret_pp"
                    ].max(),
                "success_rate_le_0_5pp":
                    g[
                        "success_le_0_5pp"
                    ].mean(),
            }
        )

    by_seed = pd.DataFrame(
        seed_rows
    )

    summary_rows = []

    for method, g in by_seed.groupby(
        "method"
    ):

        summary_rows.append(
            {
                "method": method,
                "seeds": len(g),

                "mean_of_seed_mae_pp":
                    g[
                        "mean_mae_pp"
                    ].mean(),

                "sd_of_seed_mae_pp":
                    g[
                        "mean_mae_pp"
                    ].std(ddof=1),

                "min_seed_mae_pp":
                    g[
                        "mean_mae_pp"
                    ].min(),

                "max_seed_mae_pp":
                    g[
                        "mean_mae_pp"
                    ].max(),

                "mean_of_seed_rmse_pp":
                    g[
                        "mean_rmse_pp"
                    ].mean(),

                "sd_of_seed_rmse_pp":
                    g[
                        "mean_rmse_pp"
                    ].std(ddof=1),

                "mean_of_seed_regret_pp":
                    g[
                        "mean_regret_pp"
                    ].mean(),

                "sd_of_seed_regret_pp":
                    g[
                        "mean_regret_pp"
                    ].std(ddof=1),

                "worst_regret_across_seeds_pp":
                    g[
                        "max_regret_pp"
                    ].max(),

                "mean_success_rate_le_0_5pp":
                    g[
                        "success_rate_le_0_5pp"
                    ].mean(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "mean_of_seed_mae_pp"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device_results.to_csv(
        OUT_DEVICE,
        index=False,
    )

    by_seed.to_csv(
        OUT_SEED,
        index=False,
    )

    summary.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    scope = {
        "stage": 20,
        "name":
            "Random-seed stability of leading ML baselines",
        "seeds": SEEDS,
        "models": [
            "RANDOM_FOREST",
            "EXTRA_TREES",
            "GRADIENT_BOOSTING",
        ],
        "device_profiles":
            int(
                df["device_id"].nunique()
            ),
        "independent_studies":
            int(
                df["study_id"].nunique()
            ),
        "families":
            int(
                df[
                    "technology_family"
                ].nunique()
            ),
        "rows":
            int(len(df)),
        "split_rule":
            "Study-blocked held-out-device evaluation; every same-study sibling is excluded from training.",
        "interpretation":
            "Seed variation measures algorithmic stochastic stability. It does not increase the number of independent experimental devices or studies.",
    }

    OUT_SCOPE.write_text(
        json.dumps(
            scope,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FINAL SEED-STABILITY SUMMARY")
    print("=" * 72)

    print(
        summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
