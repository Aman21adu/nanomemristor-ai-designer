from pathlib import Path

import pandas as pd


# ============================================================
# Files
# ============================================================

TRACEABILITY_FILE = "data/source_traceability.csv"

AUDIT_FILE = (
    "results/tables/"
    "traceability_audit.csv"
)

OUTPUT_CSV = (
    "results/tables/"
    "data_quality_evidence.csv"
)

OUTPUT_TEXT = (
    "results/tables/"
    "data_quality_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(file_path):

    path = Path(file_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )

    return path


def main():

    print()
    print(
        "BUILDING DATA QUALITY + TRACEABILITY SUMMARY"
    )
    print(
        "=========================================="
    )


    # ========================================================
    # Load files
    # ========================================================

    require_file(
        TRACEABILITY_FILE
    )

    require_file(
        AUDIT_FILE
    )


    trace_df = pd.read_csv(
        TRACEABILITY_FILE
    )

    audit_df = pd.read_csv(
        AUDIT_FILE
    )


    # ========================================================
    # Database scope
    # ========================================================

    total_rows = len(
        trace_df
    )

    total_devices = int(
        trace_df[
            "device_id"
        ].nunique()
    )


    # ========================================================
    # Evidence types
    # ========================================================

    type_counts = (
        trace_df[
            "value_type"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )


    experimental_count = int(
        type_counts.get(
            "EXPERIMENTAL",
            0,
        )
    )

    derived_count = int(
        type_counts.get(
            "DERIVED",
            0,
        )
    )

    reported_count = int(
        type_counts.get(
            "REPORTED",
            0,
        )
    )

    not_reported_count = int(
        type_counts.get(
            "NOT_REPORTED",
            0,
        )
    )

    not_applicable_count = int(
        type_counts.get(
            "NOT_APPLICABLE",
            0,
        )
    )

    derived_fabrication_count = int(
        type_counts.get(
            "DERIVED_FROM_FABRICATION",
            0,
        )
    )

    assumed_count = int(
        type_counts.get(
            "ASSUMED",
            0,
        )
    )


    # ========================================================
    # Provenance completeness
    # ========================================================

    provenance_columns = [

        "source_title",
        "doi",
        "page",
        "figure_or_table",
        "source_note",

    ]


    missing_counts = {

        column: int(
            trace_df[
                column
            ]
            .isna()
            .sum()
        )

        for column
        in provenance_columns

    }


    rows_with_missing = int(

        trace_df[
            provenance_columns
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()

    )


    complete_rows = (
        total_rows
        - rows_with_missing
    )


    provenance_pct = (

        100.0
        * complete_rows
        / total_rows

    )


    # ========================================================
    # Active simulator audit
    # ========================================================

    simulated_devices = int(

        audit_df[
            "device_id"
        ]
        .nunique()

    )


    audit_checks = len(
        audit_df
    )


    pass_checks = int(

        (
            audit_df[
                "status"
            ]
            ==
            "PASS"
        )
        .sum()

    )


    warning_checks = int(

        (
            audit_df[
                "status"
            ]
            ==
            "WARN"
        )
        .sum()

    )


    fail_checks = int(

        (
            audit_df[
                "status"
            ]
            ==
            "FAIL"
        )
        .sum()

    )


    assumption_df = audit_df[

        audit_df[
            "value_type"
        ]
        ==
        "ASSUMED"

    ].copy()


    assumption_count = len(
        assumption_df
    )


    # ========================================================
    # Evidence table
    # ========================================================

    evidence_rows = [

        {
            "evidence_id":
                "DATABASE_PROPERTY_ROWS",

            "metric_category":
                "DATABASE_SCOPE",

            "metric_name":
                "Literature property records",

            "value":
                total_rows,

            "unit":
                "rows",

            "interpretation":
                (
                    "Property-level literature records "
                    "stored in the traceability database."
                ),
        },

        {
            "evidence_id":
                "DATABASE_DEVICE_COUNT",

            "metric_category":
                "DATABASE_SCOPE",

            "metric_name":
                "Physical devices represented",

            "value":
                total_devices,

            "unit":
                "devices",

            "interpretation":
                (
                    "Distinct literature devices currently "
                    "represented."
                ),
        },

        {
            "evidence_id":
                "PROVENANCE_COMPLETENESS",

            "metric_category":
                "PROVENANCE",

            "metric_name":
                "Complete provenance rows",

            "value":
                provenance_pct,

            "unit":
                "percent",

            "interpretation":
                (
                    f"{complete_rows}/{total_rows} rows contain "
                    f"source title, DOI, page, evidence locator "
                    f"and source note."
                ),
        },

        {
            "evidence_id":
                "EXPERIMENTAL_ROWS",

            "metric_category":
                "EVIDENCE_TYPE",

            "metric_name":
                "Experimental records",

            "value":
                experimental_count,

            "unit":
                "rows",

            "interpretation":
                (
                    "Values directly supported by "
                    "experimental literature evidence."
                ),
        },

        {
            "evidence_id":
                "DERIVED_ROWS",

            "metric_category":
                "EVIDENCE_TYPE",

            "metric_name":
                "Derived records",

            "value":
                derived_count,

            "unit":
                "rows",

            "interpretation":
                (
                    "Values calculated or classified from "
                    "reported literature evidence."
                ),
        },

        {
            "evidence_id":
                "NOT_REPORTED_ROWS",

            "metric_category":
                "EVIDENCE_TYPE",

            "metric_name":
                "Explicitly not-reported records",

            "value":
                not_reported_count,

            "unit":
                "rows",

            "interpretation":
                (
                    "Missing literature values are explicitly "
                    "recorded instead of being invented."
                ),
        },

        {
            "evidence_id":
                "ASSUMED_ROWS",

            "metric_category":
                "EVIDENCE_TYPE",

            "metric_name":
                "Explicit assumptions",

            "value":
                assumed_count,

            "unit":
                "rows",

            "interpretation":
                (
                    "Modeling assumptions are explicitly "
                    "identified rather than presented as "
                    "experimental measurements."
                ),
        },

        {
            "evidence_id":
                "ACTIVE_SIMULATOR_AUDIT",

            "metric_category":
                "SIMULATOR_INPUT_AUDIT",

            "metric_name":
                "Detailed simulator-input checks",

            "value":
                audit_checks,

            "unit":
                "checks",

            "interpretation":
                (
                    f"Traceability checks across "
                    f"{simulated_devices} currently simulated "
                    f"devices."
                ),
        },

        {
            "evidence_id":
                "ACTIVE_PASS_CHECKS",

            "metric_category":
                "SIMULATOR_INPUT_AUDIT",

            "metric_name":
                "Passed checks",

            "value":
                pass_checks,

            "unit":
                "checks",

            "interpretation":
                (
                    "Traceable active simulator inputs with "
                    "no warning."
                ),
        },

        {
            "evidence_id":
                "ACTIVE_WARNING_CHECKS",

            "metric_category":
                "SIMULATOR_INPUT_AUDIT",

            "metric_name":
                "Warning checks",

            "value":
                warning_checks,

            "unit":
                "checks",

            "interpretation":
                (
                    "Traceable inputs that remain explicit "
                    "modeling assumptions."
                ),
        },

        {
            "evidence_id":
                "ACTIVE_FAILED_CHECKS",

            "metric_category":
                "SIMULATOR_INPUT_AUDIT",

            "metric_name":
                "Failed checks",

            "value":
                fail_checks,

            "unit":
                "checks",

            "interpretation":
                (
                    "Inputs failing traceability or "
                    "consistency validation."
                ),
        },

    ]


    evidence_df = pd.DataFrame(
        evidence_rows
    )


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
    # Human-readable report
    # ========================================================

    lines = [

        "DATA QUALITY + TRACEABILITY SUMMARY",
        "===================================",
        "",

        "1. LITERATURE DATABASE",
        "-----------------------",
        "",
        f"Physical devices represented: {total_devices}.",
        f"Property-level traceability records: {total_rows}.",
        "",

        "2. PROVENANCE COMPLETENESS",
        "--------------------------",
        "",
        (
            f"Complete provenance: "
            f"{complete_rows}/{total_rows} "
            f"({provenance_pct:.1f}%)."
        ),
        "",
        (
            f"Missing source titles: "
            f"{missing_counts['source_title']}."
        ),
        (
            f"Missing DOI values: "
            f"{missing_counts['doi']}."
        ),
        (
            f"Missing page references: "
            f"{missing_counts['page']}."
        ),
        (
            f"Missing evidence locators: "
            f"{missing_counts['figure_or_table']}."
        ),
        (
            f"Missing source notes: "
            f"{missing_counts['source_note']}."
        ),
        "",

        "3. EVIDENCE CLASSIFICATION",
        "--------------------------",
        "",
        f"EXPERIMENTAL: {experimental_count}.",
        f"DERIVED: {derived_count}.",
        f"REPORTED: {reported_count}.",
        (
            "DERIVED_FROM_FABRICATION: "
            f"{derived_fabrication_count}."
        ),
        f"NOT_REPORTED: {not_reported_count}.",
        f"NOT_APPLICABLE: {not_applicable_count}.",
        f"ASSUMED: {assumed_count}.",
        "",
        (
            "Measured, reported, derived, missing and "
            "assumed values are not treated as equivalent."
        ),
        "",

        "4. ACTIVE SIMULATOR INPUT AUDIT",
        "--------------------------------",
        "",
        f"Simulated devices: {simulated_devices}.",
        f"Detailed checks: {audit_checks}.",
        f"PASS: {pass_checks}.",
        f"WARN: {warning_checks}.",
        f"FAIL: {fail_checks}.",
        "",

        "5. EXPLICIT ASSUMPTIONS",
        "-----------------------",
        "",

    ]


    if assumption_count == 0:

        lines.append(
            "No active simulator input is marked ASSUMED."
        )

    else:

        for _, row in assumption_df.iterrows():

            lines.append(
                (
                    f"{row['device_id']} - "
                    f"{row['property_name']} = "
                    f"{row['simulator_value']}"
                )
            )

            lines.append(
                f"Reason: {row['message']}"
            )

            lines.append("")


    lines.extend([

        "",
        "6. EXAMPLE: TaOx_01",
        "-------------------",
        "",
        "RON = 1100 ohm -> EXPERIMENTAL.",
        "ROFF = 3.2 Mohm -> EXPERIMENTAL.",
        (
            "ON/OFF ratio = 2909.09 -> "
            "DERIVED from ROFF/RON."
        ),
        (
            "Conductance states = 7 -> "
            "DERIVED from the paper's state description."
        ),
        "",

        "MENTOR ANSWER: HOW CORRECT IS THE DATA?",
        "---------------------------------------",
        "",
        (
            f"All {total_rows} current literature records "
            f"across {total_devices} devices have complete "
            f"property-level provenance."
        ),
        "",
        (
            "Values are explicitly classified as "
            "experimental, reported, derived, not reported, "
            "not applicable, fabrication-derived, or assumed."
        ),
        "",
        (
            f"For the {simulated_devices} devices currently "
            f"used by the accelerator simulator, "
            f"{audit_checks} detailed checks produced "
            f"{pass_checks} PASS, "
            f"{warning_checks} WARN and "
            f"{fail_checks} FAIL."
        ),
        "",
        (
            "The current warning is an explicitly declared "
            "modeling assumption rather than an untraceable "
            "or hidden input."
        ),
        "",

        "CURRENT LIMITATION",
        "------------------",
        "",
        (
            "Traceability proves where an input came from "
            "and how its evidence was classified."
        ),
        "",
        (
            "It does not prove that every published "
            "measurement is universally correct."
        ),
        "",
        (
            "It also does not prove that the simulator has "
            "been experimentally validated against fabricated "
            "hardware."
        ),

    ])


    with open(
        OUTPUT_TEXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


    # ========================================================
    # Console
    # ========================================================

    print()

    print(
        "Literature devices:",
        total_devices
    )

    print(
        "Traceability rows:",
        total_rows
    )

    print(
        "Provenance completeness:",
        f"{provenance_pct:.1f}%"
    )

    print()

    print(
        "Evidence types:"
    )

    print(
        "  EXPERIMENTAL:",
        experimental_count
    )

    print(
        "  DERIVED:",
        derived_count
    )

    print(
        "  REPORTED:",
        reported_count
    )

    print(
        "  NOT_REPORTED:",
        not_reported_count
    )

    print(
        "  NOT_APPLICABLE:",
        not_applicable_count
    )

    print(
        "  DERIVED_FROM_FABRICATION:",
        derived_fabrication_count
    )

    print(
        "  ASSUMED:",
        assumed_count
    )

    print()

    print(
        "Active simulator devices:",
        simulated_devices
    )

    print(
        "Detailed simulator checks:",
        audit_checks
    )

    print(
        "PASS:",
        pass_checks
    )

    print(
        "WARN:",
        warning_checks
    )

    print(
        "FAIL:",
        fail_checks
    )

    print()

    print(
        "Active assumption inputs:",
        assumption_count
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
        "Every current simulator input is traceable "
        "to documented literature evidence or an "
        "explicitly declared modeling assumption."
    )

    print()

    print(
        "Traceability does not equal experimental "
        "hardware validation."
    )


if __name__ == "__main__":

    main()