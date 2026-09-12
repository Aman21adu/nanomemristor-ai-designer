from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ML_INPUT = Path("results/tables/ml_dataset.csv")
VALIDATION_INPUT = Path("results/tables/zero_shot_predictions.csv")
OUT_DIR = Path("results/tables")

RANDOM_STATE = 42
TOTAL_NETWORK_WEIGHTS = 101632

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

SUPPORTED_MODES = [
    "ANALOG",
    "GRADUAL_MULTILEVEL",
    "DISCRETE_BINARY",
    "DISCRETE_MULTILEVEL",
]


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


def tree_predictions(model: Pipeline, X: pd.DataFrame):
    pre = model.named_steps["preprocessor"]
    forest = model.named_steps["regressor"]
    Xt = pre.transform(X)

    per_tree = np.column_stack(
        [tree.predict(Xt) for tree in forest.estimators_]
    )

    return (
        per_tree.mean(axis=1),
        per_tree.std(axis=1, ddof=1),
    )


def infer_mapping(mode: str):
    mode = mode.upper()

    if mode == "ANALOG":
        return "IDEALIZED_SINGLE_PAIR_ANALOG", "IDEALIZED_ANALOG_MAPPING"

    if mode == "GRADUAL_MULTILEVEL":
        return "IDEALIZED_SINGLE_PAIR_GRADUAL", "IDEALIZED_GRADUAL_MAPPING"

    if mode == "DISCRETE_BINARY":
        return "BINARY_BIT_SLICED_DIFFERENTIAL", "BIT_SLICED_BINARY_MAPPING"

    if mode == "DISCRETE_MULTILEVEL":
        return "SINGLE_DIFFERENTIAL_PAIR", "DERIVED_DEVICE_STATE_CAP"

    raise ValueError(f"Unsupported conductance mode: {mode}")


def base_crossbar_tiles(crossbar_size: int) -> int:
    layers = [(784, 128), (128, 10)]
    total = 0

    for inp, out in layers:
        total += math.ceil(inp / crossbar_size) * math.ceil(out / crossbar_size)

    return total


def mapping_descriptors(mode: str, states: int, weight_bits: int):
    requested_levels = (2 ** int(weight_bits)) - 1

    if mode in {"ANALOG", "GRADUAL_MULTILEVEL"}:
        return requested_levels, 0, 2

    if mode == "DISCRETE_BINARY":
        slices = max(int(weight_bits) - 1, 1)
        return requested_levels, slices, 2 * slices

    if mode == "DISCRETE_MULTILEVEL":
        # Differential pair with S states/branch provides at most 2S-1
        # signed differential levels in this simplified mapping model.
        effective = min(requested_levels, (2 * int(states)) - 1)
        return effective, 0, 2

    raise ValueError(mode)


def build_candidate_grid(
    on_off_ratio: float,
    mode: str,
    states: int,
    config_reference: pd.DataFrame,
) -> pd.DataFrame:
    mapping_strategy, precision_basis = infer_mapping(mode)

    config_keys = (
        config_reference[
            ["crossbar_size", "requested_weight_bits", "adc_bits"]
        ]
        .drop_duplicates()
        .sort_values(["crossbar_size", "requested_weight_bits", "adc_bits"])
    )

    rows = []

    for _, cfg in config_keys.iterrows():
        cb = int(cfg["crossbar_size"])
        wb = int(cfg["requested_weight_bits"])
        adc = int(cfg["adc_bits"])

        levels, slices, cells_per_weight = mapping_descriptors(
            mode, states, wb
        )

        physical_tiles = base_crossbar_tiles(cb) * cells_per_weight
        memristor_cells = TOTAL_NETWORK_WEIGHTS * cells_per_weight
        adc_levels = 2 ** adc
        cost_proxy = physical_tiles * adc_levels

        rows.append(
            {
                "device_on_off_ratio": float(on_off_ratio),
                "device_log10_on_off_ratio": math.log10(float(on_off_ratio)),
                "state_count_available": int(states >= 2),
                "physical_state_count": int(states if states >= 2 else 0),
                "device_conductance_mode": mode,
                "mapping_strategy": mapping_strategy,
                "precision_basis": precision_basis,
                "crossbar_size": cb,
                "requested_weight_bits": wb,
                "effective_weight_levels": int(levels),
                "slices_per_branch": int(slices),
                "physical_cells_per_weight": int(cells_per_weight),
                "adc_bits": adc,
                "estimated_memristor_cells": int(memristor_cells),
                "estimated_physical_crossbar_tiles": int(physical_tiles),
                "adc_levels": int(adc_levels),
                "relative_hardware_cost_proxy": int(cost_proxy),
            }
        )

    return pd.DataFrame(rows)


def reliability_table(validation: pd.DataFrame) -> pd.DataFrame:
    required = {
        "requested_weight_bits",
        "adc_bits",
        "absolute_prediction_error_pp",
    }

    if not required.issubset(validation.columns):
        raise ValueError(
            "zero_shot_predictions.csv does not contain the fields "
            "needed for the empirical reliability penalty."
        )

    return (
        validation.groupby(
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
    )


def device_descriptor_table(ml: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "device_id",
        "study_id",
        "technology_family",
        "device_log10_on_off_ratio",
        "state_count_available",
        "physical_state_count",
        "device_conductance_mode",
        "mapping_strategy",
    ]

    return ml[cols].drop_duplicates("device_id").reset_index(drop=True)


def nearest_descriptor(
    custom_row: pd.Series,
    known: pd.DataFrame,
):
    numeric = [
        "device_log10_on_off_ratio",
        "state_count_available",
        "physical_state_count",
    ]
    categorical = [
        "device_conductance_mode",
        "mapping_strategy",
    ]

    prepared = {}

    for col in numeric:
        s = pd.to_numeric(known[col], errors="coerce")
        median = float(s.median())
        std = float(s.std(ddof=0))

        if not np.isfinite(std) or std < 1e-12:
            std = 1.0

        prepared[col] = (s.fillna(median), median, std)

    distances = []

    for idx, row in known.iterrows():
        squared = []

        for col in numeric:
            vals, _, std = prepared[col]
            test = float(custom_row[col])
            train = float(vals.loc[idx])
            squared.append(((test - train) / std) ** 2)

        numeric_distance = math.sqrt(sum(squared) / len(squared))

        cat_mismatch = np.mean(
            [
                0.0 if str(custom_row[c]) == str(row[c]) else 1.0
                for c in categorical
            ]
        )

        combined = numeric_distance + float(cat_mismatch)
        distances.append((idx, combined, numeric_distance, cat_mismatch))

    distances.sort(key=lambda x: x[1])
    idx, combined, num, cat = distances[0]

    return known.loc[idx], combined, num, cat


def validate_args(args):
    if args.on_off_ratio <= 1.0:
        raise ValueError("--on-off-ratio must be greater than 1.")

    mode = args.mode.upper()

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"--mode must be one of: {', '.join(SUPPORTED_MODES)}"
        )

    if mode == "DISCRETE_BINARY":
        states = 2
    elif mode == "DISCRETE_MULTILEVEL":
        if args.states is None or args.states < 2:
            raise ValueError(
                "DISCRETE_MULTILEVEL requires --states >= 2."
            )
        states = int(args.states)
    else:
        # For analog/gradual behavior a missing fixed state count must remain
        # unknown. Zero is only the machine-readable placeholder.
        if args.states is None:
            states = 0
        else:
            if args.states < 2:
                raise ValueError("--states must be >=2 when supplied.")
            states = int(args.states)

    return mode, states


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 13 custom-device accelerator predictor. "
            "This predicts simulator-derived accuracy for a user-specified "
            "device descriptor; it does not fabricate new physical evidence."
        )
    )

    parser.add_argument(
        "--on-off-ratio",
        type=float,
        required=True,
        help="Reported or user-assumed ON/OFF resistance ratio (>1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help=(
            "ANALOG, GRADUAL_MULTILEVEL, "
            "DISCRETE_BINARY, or DISCRETE_MULTILEVEL"
        ),
    )
    parser.add_argument(
        "--states",
        type=int,
        default=None,
        help=(
            "Fixed physical conductance-state count. "
            "Omit for analog/gradual devices when not reported."
        ),
    )
    parser.add_argument(
        "--name",
        type=str,
        default="CustomDevice",
        help="Label used only in output files.",
    )

    args = parser.parse_args()
    mode, states = validate_args(args)

    for path in [ML_INPUT, VALIDATION_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(ML_INPUT)
    validation = pd.read_csv(VALIDATION_INPUT)

    model = build_model()
    model.fit(ml[ALL_FEATURES], ml["accuracy"])

    candidates = build_candidate_grid(
        on_off_ratio=args.on_off_ratio,
        mode=mode,
        states=states,
        config_reference=ml,
    )

    mean_pred, tree_std = tree_predictions(
        model,
        candidates[ALL_FEATURES],
    )

    candidates["predicted_accuracy"] = mean_pred
    candidates["tree_disagreement"] = tree_std

    rel = reliability_table(validation)

    candidates = candidates.merge(
        rel,
        on=["requested_weight_bits", "adc_bits"],
        how="left",
        validate="many_to_one",
    )

    if candidates["historical_mean_abs_error_pp"].isna().any():
        fallback = float(
            validation["absolute_prediction_error_pp"].mean()
        )
        candidates["historical_mean_abs_error_pp"] = (
            candidates["historical_mean_abs_error_pp"].fillna(fallback)
        )

    # Validation-aware score:
    # predicted simulated accuracy minus historical study-blocked MAE for
    # the same weight/ADC precision region.
    #
    # This is empirical and uncalibrated; it is NOT a confidence bound.
    candidates["validation_adjusted_score"] = (
        candidates["predicted_accuracy"]
        - candidates["historical_mean_abs_error_pp"]
    )

    accuracy_first = candidates.sort_values(
        [
            "predicted_accuracy",
            "relative_hardware_cost_proxy",
        ],
        ascending=[False, True],
    ).iloc[0]

    validation_aware = candidates.sort_values(
        [
            "validation_adjusted_score",
            "predicted_accuracy",
            "relative_hardware_cost_proxy",
        ],
        ascending=[False, False, True],
    ).iloc[0]

    known = device_descriptor_table(ml)

    descriptor = validation_aware.copy()
    nearest, combined, num_dist, cat_dist = nearest_descriptor(
        descriptor,
        known,
    )

    candidates["custom_device_name"] = args.name
    candidates["input_on_off_ratio"] = float(args.on_off_ratio)
    candidates["input_fixed_state_count"] = int(states)
    candidates["descriptor_distance_to_nearest_known"] = float(combined)
    candidates["nearest_known_device"] = str(nearest["device_id"])

    safe_name = "".join(
        c if c.isalnum() or c in {"_", "-"} else "_"
        for c in args.name
    ).strip("_") or "CustomDevice"

    candidates_out = OUT_DIR / f"custom_device_{safe_name}_candidates.csv"
    summary_out = OUT_DIR / f"custom_device_{safe_name}_summary.csv"

    candidates.to_csv(candidates_out, index=False)

    summary = pd.DataFrame(
        [
            {
                "custom_device_name": args.name,
                "on_off_ratio": float(args.on_off_ratio),
                "conductance_mode": mode,
                "fixed_state_count": int(states),
                "fixed_state_count_is_reported_by_user": bool(args.states is not None),
                "candidate_configurations": len(candidates),
                "accuracy_first_crossbar": int(accuracy_first["crossbar_size"]),
                "accuracy_first_weight_bits": int(
                    accuracy_first["requested_weight_bits"]
                ),
                "accuracy_first_adc_bits": int(accuracy_first["adc_bits"]),
                "accuracy_first_predicted_accuracy": float(
                    accuracy_first["predicted_accuracy"]
                ),
                "validation_aware_crossbar": int(
                    validation_aware["crossbar_size"]
                ),
                "validation_aware_weight_bits": int(
                    validation_aware["requested_weight_bits"]
                ),
                "validation_aware_adc_bits": int(
                    validation_aware["adc_bits"]
                ),
                "validation_aware_predicted_accuracy": float(
                    validation_aware["predicted_accuracy"]
                ),
                "validation_aware_historical_error_penalty_pp": float(
                    validation_aware["historical_mean_abs_error_pp"]
                ),
                "validation_adjusted_score": float(
                    validation_aware["validation_adjusted_score"]
                ),
                "tree_disagreement": float(
                    validation_aware["tree_disagreement"]
                ),
                "nearest_known_device": str(nearest["device_id"]),
                "nearest_known_family": str(nearest["technology_family"]),
                "descriptor_distance_to_nearest_known": float(combined),
                "numeric_descriptor_distance": float(num_dist),
                "categorical_mismatch_fraction": float(cat_dist),
            }
        ]
    )

    summary.to_csv(summary_out, index=False)

    print("\nSTAGE 13 — CUSTOM DEVICE INPUT")
    print("=" * 72)
    print(f"Custom device: {args.name}")
    print(f"ON/OFF ratio: {args.on_off_ratio:g}")
    print(f"Conductance mode: {mode}")

    if states > 0:
        print(f"Fixed state count supplied: {states}")
    else:
        print("Fixed state count: NOT SUPPLIED / UNKNOWN")

    print(f"Candidate configurations generated: {len(candidates)}")
    print(f"Training rows used: {len(ml)}")
    print(f"Training device profiles: {ml['device_id'].nunique()}")

    print("\nACCURACY-FIRST PREDICTION")
    print("-" * 72)
    print(
        f"{int(accuracy_first['crossbar_size'])}x"
        f"{int(accuracy_first['crossbar_size'])}, "
        f"W={int(accuracy_first['requested_weight_bits'])}, "
        f"ADC={int(accuracy_first['adc_bits'])}"
    )
    print(
        f"Predicted simulated accuracy: "
        f"{float(accuracy_first['predicted_accuracy']):.3f}%"
    )

    print("\nVALIDATION-AWARE RECOMMENDATION")
    print("-" * 72)
    print(
        f"{int(validation_aware['crossbar_size'])}x"
        f"{int(validation_aware['crossbar_size'])}, "
        f"W={int(validation_aware['requested_weight_bits'])}, "
        f"ADC={int(validation_aware['adc_bits'])}"
    )
    print(
        f"Predicted simulated accuracy: "
        f"{float(validation_aware['predicted_accuracy']):.3f}%"
    )
    print(
        f"Historical study-blocked error penalty: "
        f"{float(validation_aware['historical_mean_abs_error_pp']):.3f} pp"
    )
    print(
        f"Validation-adjusted score: "
        f"{float(validation_aware['validation_adjusted_score']):.3f}"
    )
    print(
        f"Tree disagreement: "
        f"{float(validation_aware['tree_disagreement']):.3f} "
        "(uncalibrated)"
    )

    print("\nDESCRIPTOR-DISTANCE WARNING")
    print("-" * 72)
    print(
        f"Nearest known device: {nearest['device_id']} "
        f"({nearest['technology_family']})"
    )
    print(
        f"Relative descriptor distance: {combined:.3f}"
    )
    print(
        "This distance is a heuristic only; it is not an OOD probability."
    )

    print("\nIMPORTANT LIMITATION")
    print("-" * 72)
    print(
        "This tool predicts simulator-derived accelerator performance from "
        "user-supplied descriptors. It does NOT validate the custom material "
        "physically, does NOT create missing literature evidence, and does NOT "
        "replace fabrication or experimental characterization."
    )
    print(
        "The validation-adjusted score is an empirical reliability heuristic "
        "derived from prior study-blocked prediction errors; it is NOT a "
        "calibrated confidence interval."
    )

    print("\nSAVED OUTPUTS")
    print("-" * 72)
    print(candidates_out)
    print(summary_out)


if __name__ == "__main__":
    main()
