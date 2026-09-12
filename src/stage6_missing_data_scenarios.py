from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("results/tables/ml_dataset.csv")
OUTPUT_DIR = Path("results/tables")
DEVICE_OUT = OUTPUT_DIR / "evidence_coverage_by_device.csv"
SUMMARY_OUT = OUTPUT_DIR / "evidence_coverage_summary.csv"
SCENARIO_OUT = OUTPUT_DIR / "scenario_registry.csv"
POLICY_OUT = OUTPUT_DIR / "missing_data_policy.json"

KEY_DEVICE_COLUMNS = [
    "device_id",
    "study_id",
    "technology_family",
    "device_on_off_ratio",
    "device_conductance_mode",
    "device_state_count_status",
    "state_count_available",
    "physical_state_count",
    "parameter_source",
    "mapping_strategy",
    "precision_basis",
]


def first_value(group: pd.DataFrame, col: str):
    if col not in group.columns:
        return np.nan
    s = group[col].dropna()
    if s.empty:
        return np.nan
    return s.iloc[0]


def text(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def bool_from_value(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def build_device_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for device_id, g in df.groupby("device_id", sort=True):
        row = {"device_id": device_id}

        for col in KEY_DEVICE_COLUMNS:
            if col == "device_id":
                continue
            row[col] = first_value(g, col)

        ratio = pd.to_numeric(pd.Series([row.get("device_on_off_ratio")]), errors="coerce").iloc[0]
        ratio_available = bool(pd.notna(ratio) and ratio > 0)

        parameter_source = text(row.get("parameter_source")).lower()
        absolute_available = parameter_source == "absolute"
        normalized_ratio_only = parameter_source == "normalized_from_ratio"

        state_available = bool_from_value(row.get("state_count_available"))
        state_count = pd.to_numeric(
            pd.Series([row.get("physical_state_count")]), errors="coerce"
        ).iloc[0]
        reported_state_count = bool(state_available and pd.notna(state_count) and state_count > 0)

        state_status = text(row.get("device_state_count_status")).upper()
        if reported_state_count:
            state_evidence = "REPORTED_OR_DERIVED_FIXED_COUNT"
        elif "NOT_REPORTED" in state_status or not state_available:
            state_evidence = "NOT_REPORTED"
        else:
            state_evidence = "UNRESOLVED"

        mapping = text(row.get("mapping_strategy")).upper()
        precision = text(row.get("precision_basis")).upper()
        idealized_mapping = ("IDEALIZED" in mapping) or ("IDEALIZED" in precision)

        evidence_items = {
            "on_off_ratio_available": ratio_available,
            "absolute_electrical_values_available": absolute_available,
            "fixed_state_count_available": reported_state_count,
            "nonideal_mapping_available": not idealized_mapping,
        }
        coverage_count = sum(int(v) for v in evidence_items.values())

        reasons = []
        if not ratio_available:
            reasons.append("ON/OFF ratio missing")
        if normalized_ratio_only:
            reasons.append("absolute RON/ROFF unavailable; normalized from ratio")
        elif not absolute_available:
            reasons.append("absolute electrical parameter basis unresolved")
        if not reported_state_count:
            reasons.append("fixed physical state count not reported")
        if idealized_mapping:
            reasons.append("accelerator mapping is idealized")

        if reasons:
            scenario_status = "ASSUMPTION_SENSITIVE"
        else:
            scenario_status = "LITERATURE_CONSTRAINED"

        row.update(
            {
                "on_off_ratio_available": ratio_available,
                "absolute_electrical_values_available": absolute_available,
                "normalized_ratio_only": normalized_ratio_only,
                "fixed_state_count_available": reported_state_count,
                "state_count_evidence": state_evidence,
                "idealized_mapping": idealized_mapping,
                "evidence_items_available": coverage_count,
                "evidence_items_total": len(evidence_items),
                "evidence_coverage_fraction": coverage_count / len(evidence_items),
                "scenario_status": scenario_status,
                "scenario_reasons": "; ".join(reasons) if reasons else "none",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_scenario_registry(device_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in device_df.iterrows():
        device = r["device_id"]

        # Scenario 1: exactly what the current literature-grounded pipeline supports.
        rows.append(
            {
                "device_id": device,
                "scenario_id": "EVIDENCE_BASELINE",
                "scenario_class": "BASELINE",
                "parameter": "all_current_supported_parameters",
                "scenario_value": "current_pipeline_value",
                "source_type": "REPORTED_OR_DERIVED_OR_EXPLICITLY_NORMALIZED",
                "allowed_for_main_results": True,
                "interpretation": (
                    "Use the current literature-grounded device description. "
                    "Unknown parameters remain unknown and are not silently imputed."
                ),
            }
        )

        if bool(r["normalized_ratio_only"]):
            rows.append(
                {
                    "device_id": device,
                    "scenario_id": "ABSOLUTE_CONDUCTANCE_MISSING",
                    "scenario_class": "MISSING_DATA",
                    "parameter": "absolute_RON_ROFF",
                    "scenario_value": "DO_NOT_IMPUTE",
                    "source_type": "MISSING",
                    "allowed_for_main_results": True,
                    "interpretation": (
                        "Only the reported ON/OFF ratio is used. Absolute resistance "
                        "must not be fabricated from an unsupported point estimate."
                    ),
                }
            )

        if not bool(r["fixed_state_count_available"]):
            rows.append(
                {
                    "device_id": device,
                    "scenario_id": "STATE_COUNT_UNKNOWN",
                    "scenario_class": "MISSING_DATA",
                    "parameter": "physical_state_count",
                    "scenario_value": "DO_NOT_IMPUTE_POINT_VALUE",
                    "source_type": "MISSING",
                    "allowed_for_main_results": True,
                    "interpretation": (
                        "Do not convert gradual/analog behavior into an invented fixed "
                        "state count. A bounded hypothetical sweep belongs in sensitivity analysis."
                    ),
                }
            )
            rows.append(
                {
                    "device_id": device,
                    "scenario_id": "STATE_COUNT_SENSITIVITY",
                    "scenario_class": "HYPOTHETICAL_SENSITIVITY",
                    "parameter": "physical_state_count",
                    "scenario_value": "2|4|8|16|32|64",
                    "source_type": "ASSUMED_RANGE",
                    "allowed_for_main_results": False,
                    "interpretation": (
                        "Exploratory range only. These values are NOT literature measurements "
                        "for this device and must be labeled as hypothetical."
                    ),
                }
            )

        if bool(r["idealized_mapping"]):
            rows.append(
                {
                    "device_id": device,
                    "scenario_id": "IDEALIZED_MAPPING_BASELINE",
                    "scenario_class": "MODEL_ASSUMPTION",
                    "parameter": "mapping_nonideality",
                    "scenario_value": "idealized",
                    "source_type": "ASSUMED_BASELINE",
                    "allowed_for_main_results": True,
                    "interpretation": (
                        "Current baseline assumes idealized mapping. Noise, variability and drift "
                        "are not claimed to be represented here; they are deferred to the nonideality stage."
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_summary(device_df: pd.DataFrame) -> pd.DataFrame:
    n = len(device_df)

    metrics = [
        ("device_profiles", n),
        ("distinct_source_studies", device_df["study_id"].nunique(dropna=True)),
        ("technology_families", device_df["technology_family"].nunique(dropna=True)),
        ("ratio_available_devices", int(device_df["on_off_ratio_available"].sum())),
        ("absolute_parameter_devices", int(device_df["absolute_electrical_values_available"].sum())),
        ("normalized_ratio_only_devices", int(device_df["normalized_ratio_only"].sum())),
        ("fixed_state_count_devices", int(device_df["fixed_state_count_available"].sum())),
        ("idealized_mapping_devices", int(device_df["idealized_mapping"].sum())),
        (
            "assumption_sensitive_devices",
            int((device_df["scenario_status"] == "ASSUMPTION_SENSITIVE").sum()),
        ),
        (
            "literature_constrained_devices",
            int((device_df["scenario_status"] == "LITERATURE_CONSTRAINED").sum()),
        ),
    ]

    return pd.DataFrame(metrics, columns=["metric", "value"])


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run src/build_ml_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    if "device_id" not in df.columns:
        raise ValueError("ml_dataset.csv does not contain device_id")

    device_df = build_device_table(df)
    scenario_df = build_scenario_registry(device_df)
    summary_df = build_summary(device_df)

    device_df.to_csv(DEVICE_OUT, index=False)
    scenario_df.to_csv(SCENARIO_OUT, index=False)
    summary_df.to_csv(SUMMARY_OUT, index=False)

    policy = {
        "stage": 6,
        "name": "Missing-data and scenario handling",
        "principles": [
            "Never silently impute a literature parameter that is not reported.",
            "Keep reported, derived, normalized, assumed and missing information distinguishable.",
            "A hypothetical sensitivity range is not a measured device property.",
            "Unknown fixed state count for analog/gradual devices remains unknown in baseline results.",
            "Normalized conductance preserves a reported ON/OFF ratio but is not evidence of absolute RON/ROFF.",
            "Idealized mapping is a modeling assumption, not measured hardware behavior.",
            "Noise, variability and drift are deferred to the nonideality stage unless directly supported by evidence.",
        ],
        "hypothetical_state_count_range": [2, 4, 8, 16, 32, 64],
        "outputs": [
            str(DEVICE_OUT),
            str(SUMMARY_OUT),
            str(SCENARIO_OUT),
        ],
    }
    POLICY_OUT.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    print("\nSTAGE 6 — MISSING-DATA / SCENARIO AUDIT")
    print("=" * 60)
    print(f"Device profiles: {len(device_df)}")
    print(f"Source studies: {device_df['study_id'].nunique(dropna=True)}")
    print(f"Technology families: {device_df['technology_family'].nunique(dropna=True)}")

    print("\nEVIDENCE COVERAGE BY DEVICE")
    print("-" * 60)
    cols = [
        "device_id",
        "parameter_source",
        "state_count_evidence",
        "idealized_mapping",
        "evidence_items_available",
        "evidence_items_total",
        "scenario_status",
    ]
    print(device_df[cols].to_string(index=False))

    print("\nSUMMARY")
    print("-" * 60)
    print(summary_df.to_string(index=False))

    print("\nSCENARIO POLICY")
    print("-" * 60)
    print("Baseline: keep unknown values unknown.")
    print("Hypothetical ranges: label explicitly; never present as literature values.")
    print("Nonidealities: deferred to Stage 10 unless supported by evidence.")

    print("\nSAVED OUTPUTS")
    print("-" * 60)
    print(DEVICE_OUT)
    print(SUMMARY_OUT)
    print(SCENARIO_OUT)
    print(POLICY_OUT)


if __name__ == "__main__":
    main()
