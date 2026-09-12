from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd

REQ_INPUT = Path("results/tables/reverse_design_requirements.csv")
ML_INPUT = Path("results/tables/ml_dataset.csv")
EVIDENCE_INPUT = Path("results/tables/evidence_coverage_by_device.csv")

OUT_DIR = Path("results/tables")
MATCH_OUT = OUT_DIR / "reverse_design_experiment_matches.csv"
SUMMARY_OUT = OUT_DIR / "reverse_design_experiment_match_summary.csv"


def first_non_null(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if not s.empty else np.nan


def device_profiles(ml: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "device_id", "study_id", "technology_family",
        "device_on_off_ratio", "device_log10_on_off_ratio",
        "state_count_available", "physical_state_count",
        "device_conductance_mode", "mapping_strategy",
        "parameter_source", "precision_basis",
    ]
    missing = [c for c in cols if c not in ml.columns]
    if missing:
        raise ValueError(f"ml_dataset.csv missing required columns: {missing}")

    rows = []
    for device_id, g in ml.groupby("device_id", sort=True):
        row = {"device_id": device_id}
        for col in cols:
            if col != "device_id":
                row[col] = first_non_null(g[col])
        rows.append(row)
    return pd.DataFrame(rows)


def required_state_count(req: pd.Series):
    mode = str(req["candidate_mode"])
    states = int(req["candidate_state_count"])
    if mode in {"ANALOG", "GRADUAL_MULTILEVEL"} and states == 0:
        return None
    return states


def profile_distance(req: pd.Series, prof: pd.Series):
    req_log = math.log10(float(req["candidate_on_off_ratio"]))
    prof_log = float(prof["device_log10_on_off_ratio"])
    ratio_distance = abs(req_log - prof_log)

    mode_mismatch = (
        0.0
        if str(req["candidate_mode"]) == str(prof["device_conductance_mode"])
        else 1.0
    )

    req_states = required_state_count(req)

    if req_states is None:
        state_distance = 0.0
        state_status = "NOT_REQUIRED_OR_UNKNOWN"
    else:
        available = int(prof["state_count_available"]) == 1
        prof_states = int(prof["physical_state_count"])
        if available and prof_states > 0:
            state_distance = abs(math.log2(req_states) - math.log2(prof_states))
            state_status = "AVAILABLE"
        else:
            state_distance = 1.5
            state_status = "MISSING_IN_MATCHED_PROFILE"

    combined = 1.5 * mode_mismatch + ratio_distance + 0.5 * state_distance

    return {
        "match_distance": combined,
        "ratio_log10_distance": ratio_distance,
        "mode_mismatch": mode_mismatch,
        "state_count_distance": state_distance,
        "state_requirement_status": state_status,
    }


def classify_match(distance, ratio_distance, mode_mismatch, state_distance):
    if mode_mismatch == 0 and ratio_distance < 1e-6 and state_distance < 1e-6:
        return "EXACT_DESCRIPTOR_MATCH"
    if distance <= 0.15:
        return "VERY_CLOSE_EXISTING_PROFILE"
    if distance <= 0.50:
        return "CLOSE_EXISTING_PROFILE"
    if distance <= 1.00:
        return "MODERATE_GAP"
    return "LARGE_GAP"


def main():
    for path in [REQ_INPUT, ML_INPUT]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    req = pd.read_csv(REQ_INPUT)
    ml = pd.read_csv(ML_INPUT)
    profiles = device_profiles(ml)

    required_req = {
        "reverse_design_rank",
        "candidate_on_off_ratio",
        "candidate_mode",
        "candidate_state_count",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
        "validation_adjusted_score",
        "relative_hardware_cost_proxy",
    }

    missing = sorted(required_req - set(req.columns))
    if missing:
        raise ValueError(f"reverse_design_requirements.csv missing columns: {missing}")

    if "feasible" in req.columns:
        req = req[req["feasible"] == True].copy()  # noqa: E712

    if req.empty:
        raise ValueError("No feasible reverse-design requirements are available to match.")

    evidence = pd.read_csv(EVIDENCE_INPUT) if EVIDENCE_INPUT.exists() else None

    rows = []

    for _, r in req.iterrows():
        candidates = []
        for _, p in profiles.iterrows():
            d = profile_distance(r, p)
            candidates.append({"profile": p, **d})

        candidates.sort(
            key=lambda x: (
                x["match_distance"],
                x["mode_mismatch"],
                x["ratio_log10_distance"],
            )
        )

        for match_rank, item in enumerate(candidates[:3], start=1):
            p = item["profile"]

            row = {
                "reverse_design_rank": int(r["reverse_design_rank"]),
                "match_rank": match_rank,
                "candidate_on_off_ratio": float(r["candidate_on_off_ratio"]),
                "candidate_mode": str(r["candidate_mode"]),
                "candidate_state_count": int(r["candidate_state_count"]),
                "required_crossbar_size": int(r["crossbar_size"]),
                "required_weight_bits": int(r["requested_weight_bits"]),
                "required_adc_bits": int(r["adc_bits"]),
                "validation_adjusted_score": float(r["validation_adjusted_score"]),
                "relative_hardware_cost_proxy": float(r["relative_hardware_cost_proxy"]),
                "matched_device_id": str(p["device_id"]),
                "matched_study_id": str(p["study_id"]),
                "matched_family": str(p["technology_family"]),
                "matched_on_off_ratio": float(p["device_on_off_ratio"]),
                "matched_mode": str(p["device_conductance_mode"]),
                "matched_state_count_available": int(p["state_count_available"]),
                "matched_physical_state_count": int(p["physical_state_count"]),
                "matched_parameter_source": str(p["parameter_source"]),
                "matched_precision_basis": str(p["precision_basis"]),
                "match_distance": float(item["match_distance"]),
                "ratio_log10_distance": float(item["ratio_log10_distance"]),
                "mode_mismatch": float(item["mode_mismatch"]),
                "state_count_distance": float(item["state_count_distance"]),
                "state_requirement_status": str(item["state_requirement_status"]),
            }

            row["match_class"] = classify_match(
                row["match_distance"],
                row["ratio_log10_distance"],
                row["mode_mismatch"],
                row["state_count_distance"],
            )

            if evidence is not None:
                ev = evidence[evidence["device_id"] == row["matched_device_id"]]
                if not ev.empty:
                    e = ev.iloc[0]
                    row["matched_evidence_coverage_fraction"] = float(
                        e["evidence_coverage_fraction"]
                    )
                    row["matched_scenario_status"] = str(e["scenario_status"])
                    row["matched_scenario_reasons"] = str(e["scenario_reasons"])

            rows.append(row)

    matches = pd.DataFrame(rows).sort_values(
        ["reverse_design_rank", "match_rank"]
    ).reset_index(drop=True)
    matches.to_csv(MATCH_OUT, index=False)

    best = matches[matches["match_rank"] == 1].copy()

    summary = pd.DataFrame([
        {"metric": "reverse_design_requirements_matched",
         "value": int(best["reverse_design_rank"].nunique())},
        {"metric": "exact_descriptor_matches",
         "value": int((best["match_class"] == "EXACT_DESCRIPTOR_MATCH").sum())},
        {"metric": "very_close_or_exact_matches",
         "value": int(best["match_class"].isin(
             ["EXACT_DESCRIPTOR_MATCH", "VERY_CLOSE_EXISTING_PROFILE"]
         ).sum())},
        {"metric": "moderate_or_large_gap_matches",
         "value": int(best["match_class"].isin(
             ["MODERATE_GAP", "LARGE_GAP"]
         ).sum())},
        {"metric": "unique_matched_source_studies",
         "value": int(best["matched_study_id"].nunique())},
        {"metric": "unique_matched_device_profiles",
         "value": int(best["matched_device_id"].nunique())},
        {"metric": "median_best_match_distance",
         "value": float(best["match_distance"].median())},
        {"metric": "max_best_match_distance",
         "value": float(best["match_distance"].max())},
    ])
    summary.to_csv(SUMMARY_OUT, index=False)

    print("\nSTAGE 15 — MATCH REVERSE DESIGN TO EXPERIMENTAL PROFILES")
    print("=" * 80)
    print(f"Reverse-design requirements matched: {best['reverse_design_rank'].nunique()}")
    print(f"Literature device profiles available: {len(profiles)}")
    print(f"Independent primary studies represented: {profiles['study_id'].nunique()}")

    print("\nIMPORTANT INTERPRETATION")
    print("-" * 80)
    print(
        "This stage does NOT claim that a paper experimentally demonstrated "
        "the accelerator configuration proposed by reverse design."
    )
    print(
        "It only checks whether the REQUIRED DEVICE DESCRIPTORS resemble "
        "experimental memristor profiles already represented in the literature dataset."
    )
    print(
        "A descriptor match supports plausibility of those device properties; "
        "it does NOT validate accelerator-level performance."
    )

    print("\nBEST EXPERIMENTAL MATCH FOR EACH REVERSE-DESIGN REQUIREMENT")
    print("-" * 80)

    cols = [
        "reverse_design_rank",
        "candidate_on_off_ratio",
        "candidate_mode",
        "candidate_state_count",
        "matched_device_id",
        "matched_study_id",
        "matched_family",
        "matched_on_off_ratio",
        "matched_mode",
        "match_class",
        "match_distance",
    ]

    if "matched_evidence_coverage_fraction" in best.columns:
        cols += [
            "matched_evidence_coverage_fraction",
            "matched_scenario_status",
        ]

    print(
        best[cols]
        .sort_values("reverse_design_rank")
        .round(3)
        .to_string(index=False)
    )

    print("\nSUMMARY")
    print("-" * 80)
    print(summary.round(3).to_string(index=False))

    print("\nHOW TO READ THE RESULT")
    print("-" * 80)
    print(
        "EXACT_DESCRIPTOR_MATCH = the reverse-design descriptor matches an "
        "existing literature profile on modeled ON/OFF ratio, conductance mode "
        "and required state-count information."
    )
    print(
        "VERY_CLOSE/CLOSE = experimentally related descriptor region, but not "
        "an exact literature profile."
    )
    print(
        "MODERATE/LARGE_GAP = stronger extrapolation; targeted future "
        "fabrication/characterization would be most useful there."
    )

    print("\nSAVED OUTPUTS")
    print("-" * 80)
    print(MATCH_OUT)
    print(SUMMARY_OUT)


if __name__ == "__main__":
    main()
