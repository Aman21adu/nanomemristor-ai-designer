from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stage13_custom_device import (
    ALL_FEATURES,
    ML_INPUT,
    VALIDATION_INPUT,
    OUT_DIR,
    build_model,
    tree_predictions,
    build_candidate_grid,
    reliability_table,
    device_descriptor_table,
    nearest_descriptor,
)


CANDIDATES_OUT = OUT_DIR / "reverse_design_candidates.csv"
REQUIREMENTS_OUT = OUT_DIR / "reverse_design_requirements.csv"
SUMMARY_OUT = OUT_DIR / "reverse_design_summary.csv"


def descriptor_templates():
    # Unknown state count for analog/gradual remains unknown (0 placeholder).
    templates = [
        ("ANALOG", 0),
        ("GRADUAL_MULTILEVEL", 0),
        ("DISCRETE_BINARY", 2),
        ("DISCRETE_MULTILEVEL", 4),
        ("DISCRETE_MULTILEVEL", 8),
        ("DISCRETE_MULTILEVEL", 16),
    ]
    return templates


def ratio_grid(ml: pd.DataFrame, n_log_points: int = 14):
    ratios = pd.to_numeric(
        ml["device_on_off_ratio"],
        errors="coerce",
    )
    ratios = ratios[(ratios > 1) & np.isfinite(ratios)]

    if ratios.empty:
        raise ValueError("No valid ON/OFF ratios found in ml_dataset.csv.")

    lo = float(ratios.min())
    hi = float(ratios.max())

    log_grid = np.geomspace(lo, hi, n_log_points)
    exact_known = ratios.drop_duplicates().to_numpy(dtype=float)

    grid = np.unique(
        np.round(
            np.concatenate([log_grid, exact_known]),
            6,
        )
    )
    grid = grid[grid > 1.0]

    return sorted(grid.tolist())


def add_descriptor_distance(
    candidates: pd.DataFrame,
    known: pd.DataFrame,
) -> pd.DataFrame:
    descriptor_cols = [
        "device_on_off_ratio",
        "device_log10_on_off_ratio",
        "state_count_available",
        "physical_state_count",
        "device_conductance_mode",
        "mapping_strategy",
    ]

    unique = candidates[descriptor_cols].drop_duplicates().copy()

    rows = []
    for _, d in unique.iterrows():
        nearest, combined, num_dist, cat_dist = nearest_descriptor(
            d,
            known,
        )

        rows.append(
            {
                **{c: d[c] for c in descriptor_cols},
                "nearest_known_device": str(nearest["device_id"]),
                "nearest_known_family": str(nearest["technology_family"]),
                "descriptor_distance_to_nearest_known": float(combined),
                "numeric_descriptor_distance": float(num_dist),
                "categorical_mismatch_fraction": float(cat_dist),
            }
        )

    dist = pd.DataFrame(rows)

    return candidates.merge(
        dist,
        on=descriptor_cols,
        how="left",
        validate="many_to_one",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 14 reverse design: search device-property/configuration "
            "candidates that satisfy requested accelerator constraints."
        )
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        required=True,
        help=(
            "Minimum validation-adjusted predicted simulator accuracy in percent."
        ),
    )
    parser.add_argument(
        "--max-cost-proxy",
        type=float,
        required=True,
        help=(
            "Maximum relative_hardware_cost_proxy. "
            "This is a heuristic, not measured hardware cost."
        ),
    )
    parser.add_argument(
        "--max-descriptor-distance",
        type=float,
        default=None,
        help=(
            "Optional maximum relative descriptor-distance heuristic. "
            "Not an OOD probability."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of reverse-design recommendations to print.",
    )
    args = parser.parse_args()

    if not ML_INPUT.exists():
        raise FileNotFoundError(f"{ML_INPUT} not found.")
    if not VALIDATION_INPUT.exists():
        raise FileNotFoundError(f"{VALIDATION_INPUT} not found.")
    if args.max_cost_proxy <= 0:
        raise ValueError("--max-cost-proxy must be > 0.")
    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ml = pd.read_csv(ML_INPUT)
    validation = pd.read_csv(VALIDATION_INPUT)

    model = build_model()
    model.fit(ml[ALL_FEATURES], ml["accuracy"])

    rel = reliability_table(validation)
    known = device_descriptor_table(ml)

    ratios = ratio_grid(ml)
    templates = descriptor_templates()

    all_candidates = []

    for ratio in ratios:
        for mode, states in templates:
            grid = build_candidate_grid(
                on_off_ratio=float(ratio),
                mode=mode,
                states=int(states),
                config_reference=ml,
            )

            mean_pred, tree_std = tree_predictions(
                model,
                grid[ALL_FEATURES],
            )

            grid["predicted_accuracy"] = mean_pred
            grid["tree_disagreement"] = tree_std
            grid["candidate_on_off_ratio"] = float(ratio)
            grid["candidate_mode"] = mode
            grid["candidate_state_count"] = int(states)

            all_candidates.append(grid)

    candidates = pd.concat(all_candidates, ignore_index=True)

    candidates = candidates.merge(
        rel,
        on=["requested_weight_bits", "adc_bits"],
        how="left",
        validate="many_to_one",
    )

    fallback_penalty = float(
        pd.to_numeric(
            validation["absolute_prediction_error_pp"],
            errors="coerce",
        ).mean()
    )

    candidates["historical_mean_abs_error_pp"] = (
        candidates["historical_mean_abs_error_pp"].fillna(
            fallback_penalty
        )
    )

    candidates["validation_adjusted_score"] = (
        candidates["predicted_accuracy"]
        - candidates["historical_mean_abs_error_pp"]
    )

    candidates = add_descriptor_distance(
        candidates,
        known,
    )

    candidates["accuracy_margin_pp"] = (
        candidates["validation_adjusted_score"]
        - float(args.target_accuracy)
    )

    candidates["passes_accuracy"] = (
        candidates["validation_adjusted_score"]
        >= float(args.target_accuracy)
    )

    candidates["passes_cost"] = (
        candidates["relative_hardware_cost_proxy"]
        <= float(args.max_cost_proxy)
    )

    if args.max_descriptor_distance is None:
        candidates["passes_distance"] = True
    else:
        candidates["passes_distance"] = (
            candidates["descriptor_distance_to_nearest_known"]
            <= float(args.max_descriptor_distance)
        )

    candidates["feasible"] = (
        candidates["passes_accuracy"]
        & candidates["passes_cost"]
        & candidates["passes_distance"]
    )

    candidates.to_csv(CANDIDATES_OUT, index=False)

    feasible = candidates[candidates["feasible"]].copy()

    descriptor_key = [
        "candidate_on_off_ratio",
        "candidate_mode",
        "candidate_state_count",
    ]

    if not feasible.empty:
        # For each candidate device descriptor, retain its cheapest feasible
        # accelerator configuration, breaking ties by stronger adjusted score.
        req = (
            feasible.sort_values(
                [
                    *descriptor_key,
                    "relative_hardware_cost_proxy",
                    "validation_adjusted_score",
                    "predicted_accuracy",
                ],
                ascending=[
                    True, True, True,
                    True, False, False,
                ],
            )
            .drop_duplicates(
                subset=descriptor_key,
                keep="first",
            )
            .copy()
        )

        # Rank most evidence-supported first, then cheaper designs, then margin.
        req = req.sort_values(
            [
                "descriptor_distance_to_nearest_known",
                "relative_hardware_cost_proxy",
                "accuracy_margin_pp",
            ],
            ascending=[True, True, False],
        ).reset_index(drop=True)

        req["reverse_design_rank"] = np.arange(1, len(req) + 1)

    else:
        # Preserve useful closest-miss information.
        req = candidates.copy()
        req["constraint_violation_score"] = (
            np.maximum(
                0.0,
                float(args.target_accuracy)
                - req["validation_adjusted_score"],
            )
            + np.maximum(
                0.0,
                (
                    req["relative_hardware_cost_proxy"]
                    - float(args.max_cost_proxy)
                )
                / max(float(args.max_cost_proxy), 1.0),
            )
        )

        if args.max_descriptor_distance is not None:
            req["constraint_violation_score"] += np.maximum(
                0.0,
                req["descriptor_distance_to_nearest_known"]
                - float(args.max_descriptor_distance),
            )

        req = (
            req.sort_values(
                [
                    "constraint_violation_score",
                    "descriptor_distance_to_nearest_known",
                    "relative_hardware_cost_proxy",
                ],
                ascending=[True, True, True],
            )
            .drop_duplicates(
                subset=descriptor_key,
                keep="first",
            )
            .head(max(args.top_k, 15))
            .reset_index(drop=True)
        )

        req["reverse_design_rank"] = np.arange(1, len(req) + 1)

    req.to_csv(REQUIREMENTS_OUT, index=False)

    summary = pd.DataFrame(
        [
            {
                "target_accuracy": float(args.target_accuracy),
                "max_cost_proxy": float(args.max_cost_proxy),
                "max_descriptor_distance": (
                    np.nan
                    if args.max_descriptor_distance is None
                    else float(args.max_descriptor_distance)
                ),
                "training_rows": len(ml),
                "training_devices": ml["device_id"].nunique(),
                "searched_on_off_ratios": len(ratios),
                "searched_device_templates": len(templates),
                "total_candidate_configurations": len(candidates),
                "feasible_candidate_configurations": int(
                    candidates["feasible"].sum()
                ),
                "feasible_device_descriptor_sets": (
                    int(
                        feasible[descriptor_key]
                        .drop_duplicates()
                        .shape[0]
                    )
                    if not feasible.empty
                    else 0
                ),
            }
        ]
    )

    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nSTAGE 14 — REVERSE DESIGN")
    print("=" * 76)
    print(
        f"Target validation-adjusted simulated accuracy: "
        f">= {args.target_accuracy:.3f}%"
    )
    print(
        f"Maximum relative hardware-cost proxy: "
        f"{args.max_cost_proxy:g}"
    )

    if args.max_descriptor_distance is None:
        print("Descriptor-distance constraint: none")
    else:
        print(
            f"Maximum descriptor-distance heuristic: "
            f"{args.max_descriptor_distance:g}"
        )

    print(f"Training rows: {len(ml)}")
    print(f"Training device profiles: {ml['device_id'].nunique()}")
    print(f"ON/OFF ratios searched: {len(ratios)}")
    print(f"Device descriptor templates searched: {len(templates)}")
    print(f"Candidate accelerator designs searched: {len(candidates)}")
    print(
        f"Feasible candidate configurations: "
        f"{int(candidates['feasible'].sum())}"
    )

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 76)
    print(
        "Reverse design searches the learned simulator surrogate for DEVICE "
        "DESCRIPTOR REQUIREMENTS that could satisfy accelerator constraints."
    )
    print(
        "It does NOT invent a new material, prove fabricability, or guarantee "
        "physical performance."
    )
    print(
        "The cost proxy is heuristic. The validation-adjusted score and "
        "descriptor distance are also uncalibrated heuristics."
    )

    if not feasible.empty:
        print("\nTOP REVERSE-DESIGN REQUIREMENTS")
        print("-" * 76)

        cols = [
            "reverse_design_rank",
            "candidate_on_off_ratio",
            "candidate_mode",
            "candidate_state_count",
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
            "predicted_accuracy",
            "historical_mean_abs_error_pp",
            "validation_adjusted_score",
            "accuracy_margin_pp",
            "relative_hardware_cost_proxy",
            "nearest_known_device",
            "descriptor_distance_to_nearest_known",
        ]

        print(
            req[cols]
            .head(args.top_k)
            .round(3)
            .to_string(index=False)
        )
    else:
        print("\nNO FEASIBLE DESIGN FOUND")
        print("-" * 76)
        print(
            "No candidate satisfied all requested constraints. "
            "The table below shows the closest descriptor-level misses."
        )

        cols = [
            "reverse_design_rank",
            "candidate_on_off_ratio",
            "candidate_mode",
            "candidate_state_count",
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
            "validation_adjusted_score",
            "relative_hardware_cost_proxy",
            "descriptor_distance_to_nearest_known",
            "constraint_violation_score",
        ]

        print(
            req[cols]
            .head(args.top_k)
            .round(3)
            .to_string(index=False)
        )

    print("\nSAVED OUTPUTS")
    print("-" * 76)
    print(CANDIDATES_OUT)
    print(REQUIREMENTS_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
