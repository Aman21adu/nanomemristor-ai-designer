from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQ_INPUT = Path("results/tables/reverse_design_requirements.csv")
MATCH_INPUT = Path("results/tables/reverse_design_experiment_matches.csv")
OUT_DIR = Path("results/tables")

TARGETS_OUT = OUT_DIR / "future_research_targets.csv"
SUMMARY_OUT = OUT_DIR / "future_research_target_summary.csv"
SCOPE_OUT = OUT_DIR / "future_research_target_scope.json"

# We deliberately exclude exact / near-duplicate descriptor regions.
MIN_GAP_DISTANCE = 0.15


def research_tier(distance: float) -> str:
    if distance < 0.15:
        return "EXISTING_OR_NEAR_DUPLICATE"
    if distance <= 0.50:
        return "TIER_A_INTERPOLATIVE_GAP"
    if distance <= 1.00:
        return "TIER_B_MODERATE_GAP"
    return "TIER_C_HIGH_EXTRAPOLATION"


def direction_text(row: pd.Series) -> str:
    mode = str(row["candidate_mode"])
    ratio = float(row["candidate_on_off_ratio"])
    states = int(row["candidate_state_count"])
    anchor = str(row["matched_device_id"])

    if mode == "ANALOG":
        return (
            f"Explore an analog-switching device region near ON/OFF≈{ratio:g}, "
            f"using {anchor} only as the nearest literature anchor. "
            "Experimentally verify analog update behavior, absolute conductance, "
            "variability and retention before accelerator claims."
        )

    if mode == "GRADUAL_MULTILEVEL":
        return (
            f"Explore gradual-multilevel switching near ON/OFF≈{ratio:g}, "
            f"using {anchor} as the nearest literature anchor. "
            "Preserve gradual conductance tuning while experimentally checking "
            "absolute RON/ROFF, update linearity, variability and retention."
        )

    if mode == "DISCRETE_BINARY":
        return (
            f"Explore a binary switching device near ON/OFF≈{ratio:g}. "
            f"Nearest literature anchor: {anchor}. "
            "Verify stable two-state separation, endurance, retention and read margin."
        )

    if mode == "DISCRETE_MULTILEVEL":
        return (
            f"Explore a discrete-multilevel device near ON/OFF≈{ratio:g} "
            f"with about {states} stable states. Nearest literature anchor: {anchor}. "
            "The state count must be demonstrated experimentally; verify state "
            "separation, programming repeatability, retention and variability."
        )

    return (
        f"Explore the descriptor region around ON/OFF≈{ratio:g}; "
        f"nearest literature anchor: {anchor}."
    )


def main():
    for path in [REQ_INPUT, MATCH_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    req = pd.read_csv(REQ_INPUT)
    matches = pd.read_csv(MATCH_INPUT)

    required_req = {
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
    }
    missing_req = sorted(required_req - set(req.columns))
    if missing_req:
        raise ValueError(
            f"reverse_design_requirements.csv missing columns: {missing_req}"
        )

    required_match = {
        "reverse_design_rank",
        "match_rank",
        "matched_device_id",
        "matched_study_id",
        "matched_family",
        "matched_on_off_ratio",
        "matched_mode",
        "match_class",
        "match_distance",
    }
    missing_match = sorted(required_match - set(matches.columns))
    if missing_match:
        raise ValueError(
            f"reverse_design_experiment_matches.csv missing columns: {missing_match}"
        )

    if "feasible" in req.columns:
        req = req[req["feasible"] == True].copy()  # noqa: E712

    best_match = matches[matches["match_rank"] == 1].copy()

    cols = [
        "reverse_design_rank",
        "matched_device_id",
        "matched_study_id",
        "matched_family",
        "matched_on_off_ratio",
        "matched_mode",
        "match_class",
        "match_distance",
    ]

    optional_match_cols = [
        "matched_evidence_coverage_fraction",
        "matched_scenario_status",
        "matched_scenario_reasons",
    ]
    for c in optional_match_cols:
        if c in best_match.columns:
            cols.append(c)

    merged = req.merge(
        best_match[cols],
        on="reverse_design_rank",
        how="inner",
        validate="one_to_one",
    )

    merged["research_tier"] = merged["match_distance"].apply(research_tier)

    # Exclude exact and very-near duplicates. Stage 16 is about research gaps.
    future = merged[merged["match_distance"] >= MIN_GAP_DISTANCE].copy()

    if future.empty:
        raise ValueError(
            "No descriptor-gap candidates remain after excluding existing/near-duplicate profiles."
        )

    tier_order = {
        "TIER_A_INTERPOLATIVE_GAP": 1,
        "TIER_B_MODERATE_GAP": 2,
        "TIER_C_HIGH_EXTRAPOLATION": 3,
    }
    future["_tier_order"] = future["research_tier"].map(tier_order).fillna(9)

    # Transparent ranking:
    # 1) prefer the nearest genuine gap,
    # 2) then stronger validation-adjusted performance,
    # 3) then lower heuristic cost.
    #
    # This avoids inventing an opaque "novelty score".
    future = future.sort_values(
        [
            "_tier_order",
            "match_distance",
            "validation_adjusted_score",
            "relative_hardware_cost_proxy",
        ],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    future["future_research_rank"] = np.arange(1, len(future) + 1)
    future["experimental_direction"] = future.apply(direction_text, axis=1)

    future["is_material_identity_prediction"] = False
    future["is_structure_prediction"] = False
    future["requires_new_experimental_validation"] = True

    output_cols = [
        "future_research_rank",
        "research_tier",
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
        "matched_device_id",
        "matched_study_id",
        "matched_family",
        "matched_on_off_ratio",
        "matched_mode",
        "match_class",
        "match_distance",
    ]

    for c in optional_match_cols:
        if c in future.columns:
            output_cols.append(c)

    output_cols += [
        "experimental_direction",
        "is_material_identity_prediction",
        "is_structure_prediction",
        "requires_new_experimental_validation",
    ]

    future[output_cols].to_csv(TARGETS_OUT, index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "feasible_reverse_design_requirements",
                "value": int(len(merged)),
            },
            {
                "metric": "existing_or_near_duplicate_excluded",
                "value": int((merged["match_distance"] < MIN_GAP_DISTANCE).sum()),
            },
            {
                "metric": "future_gap_targets_retained",
                "value": int(len(future)),
            },
            {
                "metric": "tier_A_interpolative_gap_targets",
                "value": int(
                    (future["research_tier"] == "TIER_A_INTERPOLATIVE_GAP").sum()
                ),
            },
            {
                "metric": "tier_B_moderate_gap_targets",
                "value": int(
                    (future["research_tier"] == "TIER_B_MODERATE_GAP").sum()
                ),
            },
            {
                "metric": "tier_C_high_extrapolation_targets",
                "value": int(
                    (future["research_tier"] == "TIER_C_HIGH_EXTRAPOLATION").sum()
                ),
            },
            {
                "metric": "median_gap_distance",
                "value": float(future["match_distance"].median()),
            },
            {
                "metric": "max_gap_distance",
                "value": float(future["match_distance"].max()),
            },
        ]
    )
    summary.to_csv(SUMMARY_OUT, index=False)

    scope = {
        "stage": 16,
        "name": "Future experimental descriptor targets / research-gap generation",
        "minimum_gap_distance": MIN_GAP_DISTANCE,
        "what_this_stage_does": [
            "Starts from reverse-design candidates that satisfy the accelerator target.",
            "Removes exact and near-duplicate descriptor regions already represented in the literature dataset.",
            "Ranks remaining descriptor gaps by evidence proximity, predicted simulator performance, and heuristic cost.",
            "Generates experimentally testable DEVICE-PROPERTY TARGETS.",
        ],
        "what_this_stage_does_not_do": [
            "It does not predict a new chemical composition.",
            "It does not predict a new electrode/oxide stack.",
            "It does not claim a novel material has been discovered.",
            "It does not prove fabricability.",
            "It does not replace experimental device characterization.",
        ],
        "why_material_identity_is_not_predicted": (
            "The current ML feature space contains device descriptors such as ON/OFF ratio, "
            "conductance mode, state count and accelerator configuration, but it does not "
            "encode sufficient composition/structure/process descriptors to learn a reliable "
            "mapping from accelerator requirements back to a new material or stack."
        ),
        "recommended_next_research_extension": (
            "Add composition, electrode materials, oxide thickness, fabrication process, "
            "forming/set/reset voltages, endurance, retention, variability and temperature "
            "features across substantially more independent studies before attempting "
            "material/structure recommendation."
        ),
        "outputs": [
            str(TARGETS_OUT),
            str(SUMMARY_OUT),
        ],
    }
    SCOPE_OUT.write_text(json.dumps(scope, indent=2), encoding="utf-8")

    print("\nSTAGE 16 — FUTURE EXPERIMENTAL RESEARCH TARGETS")
    print("=" * 84)
    print(f"Feasible reverse-design requirements: {len(merged)}")
    print(
        "Existing/near-duplicate descriptor targets excluded: "
        f"{int((merged['match_distance'] < MIN_GAP_DISTANCE).sum())}"
    )
    print(f"Future descriptor-gap targets retained: {len(future)}")

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 84)
    print(
        "This stage generates FUTURE DEVICE-PROPERTY TARGETS, not new material identities."
    )
    print(
        "The current dataset is not rich enough to infer a new chemical composition, "
        "electrode stack or fabrication process reliably."
    )
    print(
        "A retained target means: this device-property region is predicted to be useful "
        "for the accelerator objective and is not already represented closely in the "
        "current literature dataset."
    )

    print("\nTOP FUTURE RESEARCH TARGETS")
    print("-" * 84)

    display_cols = [
        "future_research_rank",
        "research_tier",
        "candidate_on_off_ratio",
        "candidate_mode",
        "candidate_state_count",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
        "validation_adjusted_score",
        "relative_hardware_cost_proxy",
        "matched_device_id",
        "match_distance",
    ]

    print(
        future[display_cols]
        .head(15)
        .round(3)
        .to_string(index=False)
    )

    print("\nSUMMARY")
    print("-" * 84)
    print(summary.round(3).to_string(index=False))

    print("\nNANOTECHNOLOGY INTERPRETATION")
    print("-" * 84)
    print(
        "Use the target ON/OFF ratio, conductance behavior and state-count requirement "
        "as experimentally testable specifications. Material/stack engineering would "
        "then try to realize those specifications, but the current model must not name "
        "a new material as if it had discovered one."
    )

    print("\nSAVED OUTPUTS")
    print("-" * 84)
    print(TARGETS_OUT)
    print(SUMMARY_OUT)
    print(SCOPE_OUT)


if __name__ == "__main__":
    main()
