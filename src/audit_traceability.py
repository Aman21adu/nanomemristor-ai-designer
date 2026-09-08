from pathlib import Path
import math

import numpy as np
import pandas as pd


# ============================================================
# Files
# ============================================================

DEVICE_FILE = (
    "data/device_profiles.csv"
)

TRACE_FILE = (
    "data/source_traceability.csv"
)

RESULT_FILE = (
    "results/tables/"
    "all_device_config_results.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "traceability_audit.csv"
)


# ============================================================
# Current validated simulator devices
#
# These are read from the combined simulator-result file
# rather than hard-coded as the permanent project dataset.
# ============================================================

devices = pd.read_csv(
    DEVICE_FILE
)

trace = pd.read_csv(
    TRACE_FILE
)

results = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# Basic schema validation
# ============================================================

DEVICE_REQUIRED_COLUMNS = [

    "device_id",
    "technology_family",

    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",

    "conductance_states",
    "conductance_mode",
    "state_count_status",

]


TRACE_REQUIRED_COLUMNS = [

    "device_id",
    "property_name",
    "value",
    "unit",
    "value_type",

    "source_title",
    "doi",
    "page",
    "figure_or_table",
    "source_note",

]


RESULT_REQUIRED_COLUMNS = [

    "device_id",
    "parameter_source",

]


missing_device_columns = [

    column

    for column
    in DEVICE_REQUIRED_COLUMNS

    if column not in devices.columns

]


missing_trace_columns = [

    column

    for column
    in TRACE_REQUIRED_COLUMNS

    if column not in trace.columns

]


missing_result_columns = [

    column

    for column
    in RESULT_REQUIRED_COLUMNS

    if column not in results.columns

]


if missing_device_columns:

    raise ValueError(

        "device_profiles.csv is missing columns:\n"
        f"{missing_device_columns}"

    )


if missing_trace_columns:

    raise ValueError(

        "source_traceability.csv is missing columns:\n"
        f"{missing_trace_columns}"

    )


if missing_result_columns:

    raise ValueError(

        "all_device_config_results.csv is missing columns:\n"
        f"{missing_result_columns}"

    )


# ============================================================
# Devices actually used by the current simulator experiment
# ============================================================

used_device_ids = sorted(

    results[
        "device_id"
    ]
    .unique()

)


used_devices = devices[

    devices[
        "device_id"
    ].isin(
        used_device_ids
    )

].copy()


# ============================================================
# Validate one profile row per simulated device
# ============================================================

for device_id in used_device_ids:

    count = int(

        (
            used_devices[
                "device_id"
            ]
            ==
            device_id
        ).sum()

    )


    if count != 1:

        raise ValueError(

            f"{device_id} has {count} profile rows. "
            "Expected exactly one."

        )


# ============================================================
# Get simulator parameter source
#
# It should be constant across all configurations of a
# particular device.
# ============================================================

parameter_source_map = {}


for device_id, group in results.groupby(
    "device_id"
):

    sources = (

        group[
            "parameter_source"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()

    )


    if len(sources) != 1:

        raise ValueError(

            f"{device_id} has inconsistent "
            f"parameter_source values: {sources}"

        )


    parameter_source_map[
        device_id
    ] = sources[0]


# ============================================================
# Helper:
# normalize strings
# ============================================================

def clean_text(value):

    if pd.isna(value):

        return ""

    return str(
        value
    ).strip()


# ============================================================
# Helper:
# numeric equality with tolerance
# ============================================================

def values_close(
    a,
    b,
    relative_tolerance=1e-4,
    absolute_tolerance=1e-8,
):

    try:

        a = float(a)
        b = float(b)

    except (
        TypeError,
        ValueError,
    ):

        return False


    return math.isclose(

        a,
        b,

        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,

    )


# ============================================================
# Helper:
# retrieve one traceability row
# ============================================================

def get_trace_row(
    device_id,
    property_name,
):

    matches = trace[

        (
            trace[
                "device_id"
            ]
            ==
            device_id
        )

        &

        (
            trace[
                "property_name"
            ]
            ==
            property_name
        )

    ].copy()


    if matches.empty:

        return None


    if len(matches) > 1:

        raise ValueError(

            f"Duplicate traceability rows found for "
            f"{device_id} / {property_name}."

        )


    return (
        matches.iloc[0]
    )


# ============================================================
# Helper:
# classify evidence strength
# ============================================================

def evidence_class(
    value_type,
):

    value_type = clean_text(
        value_type
    ).upper()


    if value_type == "EXPERIMENTAL":

        return (
            "DIRECT"
        )


    if value_type in {

        "DERIVED",
        "REPORTED",
        "DERIVED_FROM_FABRICATION",

    }:

        return (
            "SUPPORTED"
        )


    if value_type == "ASSUMED":

        return (
            "ASSUMPTION"
        )


    if value_type in {

        "NOT_REPORTED",
        "NOT_APPLICABLE",

    }:

        return (
            "MISSING_OR_NA"
        )


    return (
        "UNKNOWN"
    )


# ============================================================
# Helper:
# general traceability check
# ============================================================

def audit_property(
    device_id,
    property_name,
    simulator_value,
    required,
    expected_numeric=None,
    expected_text=None,
    allow_missing_value=False,
):

    trace_row = get_trace_row(

        device_id,
        property_name,

    )


    # --------------------------------------------------------
    # Missing traceability row
    # --------------------------------------------------------

    if trace_row is None:

        if required:

            status = "FAIL"

            message = (
                "Simulator-relevant property has no "
                "traceability row."
            )

        else:

            status = "INFO"

            message = (
                "Optional property has no traceability row."
            )


        return {

            "device_id":
                device_id,

            "property_name":
                property_name,

            "simulator_value":
                simulator_value,

            "trace_value":
                np.nan,

            "value_type":
                "",

            "evidence_class":
                "NONE",

            "status":
                status,

            "doi_present":
                False,

            "page_present":
                False,

            "figure_or_table_present":
                False,

            "message":
                message,

        }


    # --------------------------------------------------------
    # Evidence metadata
    # --------------------------------------------------------

    value_type = clean_text(

        trace_row[
            "value_type"
        ]

    ).upper()


    evidence = evidence_class(
        value_type
    )


    doi_present = bool(

        clean_text(
            trace_row[
                "doi"
            ]
        )

    )


    page_present = bool(

        clean_text(
            trace_row[
                "page"
            ]
        )

    )


    figure_present = bool(

        clean_text(
            trace_row[
                "figure_or_table"
            ]
        )

    )


    trace_value = trace_row[
        "value"
    ]


    status = "PASS"

    message = (
        "Traceability row found."
    )


    # --------------------------------------------------------
    # Numeric comparison
    # --------------------------------------------------------

    if expected_numeric is not None:

        if not values_close(

            trace_value,

            expected_numeric,

        ):

            status = "FAIL"

            message = (

                "Traceability value does not match "
                "the simulator input."

            )


    # --------------------------------------------------------
    # Text comparison
    # --------------------------------------------------------

    if expected_text is not None:

        trace_text = clean_text(
            trace_value
        ).upper()


        expected_text_clean = clean_text(
            expected_text
        ).upper()


        if (
            trace_text
            != expected_text_clean
        ):

            status = "FAIL"

            message = (

                "Traceability text does not match "
                "the simulator input."

            )


    # --------------------------------------------------------
    # Missing value may be correct for properties explicitly
    # reported as unavailable in the literature.
    # --------------------------------------------------------

    if (

        allow_missing_value

        and

        pd.isna(
            trace_value
        )

        and

        value_type
        in {

            "NOT_REPORTED",
            "NOT_APPLICABLE",

        }

    ):

        status = "PASS"

        message = (

            "Literature explicitly does not provide "
            "a fixed numeric value."

        )


    # --------------------------------------------------------
    # Assumptions are allowed, but must remain visible.
    # --------------------------------------------------------

    if (

        status == "PASS"

        and

        value_type == "ASSUMED"

    ):

        status = "WARN"

        message = (

            "Simulator value is explicitly marked "
            "as an assumption/modeling approximation."

        )


    # --------------------------------------------------------
    # Unknown evidence vocabulary
    # --------------------------------------------------------

    if (

        status == "PASS"

        and

        evidence == "UNKNOWN"

    ):

        status = "WARN"

        message = (

            "Traceability exists, but value_type "
            "is not part of the recognized audit vocabulary."

        )


    # --------------------------------------------------------
    # DOI should exist for simulator-relevant literature data.
    # --------------------------------------------------------

    if (

        required

        and

        not doi_present

        and

        status == "PASS"

    ):

        status = "WARN"

        message = (

            "Traceability exists, but DOI/source identifier "
            "is missing."

        )


    return {

        "device_id":
            device_id,

        "property_name":
            property_name,

        "simulator_value":
            simulator_value,

        "trace_value":
            trace_value,

        "value_type":
            value_type,

        "evidence_class":
            evidence,

        "status":
            status,

        "doi_present":
            doi_present,

        "page_present":
            page_present,

        "figure_or_table_present":
            figure_present,

        "message":
            message,

    }


# ============================================================
# Audit rows
# ============================================================

audit_rows = []


# ============================================================
# Audit each simulation-used device
# ============================================================

for _, device in used_devices.iterrows():

    device_id = clean_text(

        device[
            "device_id"
        ]

    )


    conductance_mode = clean_text(

        device[
            "conductance_mode"
        ]

    ).upper()


    state_count_status = clean_text(

        device[
            "state_count_status"
        ]

    ).upper()


    parameter_source = clean_text(

        parameter_source_map[
            device_id
        ]

    )


    ron = device[
        "ron_ohm"
    ]


    roff = device[
        "roff_ohm"
    ]


    ratio = device[
        "on_off_ratio"
    ]


    states = device[
        "conductance_states"
    ]


    # ========================================================
    # 1. Conductance behavior mode
    # ========================================================

    audit_rows.append(

        audit_property(

            device_id=device_id,

            property_name="conductance_mode",

            simulator_value=conductance_mode,

            required=True,

            expected_text=conductance_mode,

        )

    )


    # ========================================================
    # 2. Conductance-state information
    # ========================================================

    if pd.notna(
        states
    ):

        audit_rows.append(

            audit_property(

                device_id=device_id,

                property_name="conductance_states",

                simulator_value=states,

                required=True,

                expected_numeric=states,

            )

        )


    else:

        audit_rows.append(

            audit_property(

                device_id=device_id,

                property_name="conductance_states",

                simulator_value="NO_FIXED_COUNT",

                required=True,

                allow_missing_value=True,

            )

        )


    # ========================================================
    # 3. Conductance-range parameters
    #
    # ABSOLUTE:
    # simulator uses RON and ROFF.
    #
    # normalized_from_ratio:
    # simulator uses ON/OFF ratio.
    # ========================================================

    if parameter_source == "absolute":

        if (
            pd.isna(ron)
            or pd.isna(roff)
        ):

            audit_rows.append({

                "device_id":
                    device_id,

                "property_name":
                    "conductance_range",

                "simulator_value":
                    "absolute",

                "trace_value":
                    np.nan,

                "value_type":
                    "",

                "evidence_class":
                    "NONE",

                "status":
                    "FAIL",

                "doi_present":
                    False,

                "page_present":
                    False,

                "figure_or_table_present":
                    False,

                "message":
                    (
                        "parameter_source=absolute but "
                        "RON/ROFF are missing."
                    ),

            })


        else:

            audit_rows.append(

                audit_property(

                    device_id=device_id,

                    property_name="ron_ohm",

                    simulator_value=ron,

                    required=True,

                    expected_numeric=ron,

                )

            )


            audit_rows.append(

                audit_property(

                    device_id=device_id,

                    property_name="roff_ohm",

                    simulator_value=roff,

                    required=True,

                    expected_numeric=roff,

                )

            )


            # ------------------------------------------------
            # If an ON/OFF ratio is also stored in the profile,
            # verify that it is numerically consistent with
            # ROFF/RON.
            # ------------------------------------------------

            calculated_ratio = (

                float(roff)
                / float(ron)

            )


            if pd.notna(
                ratio
            ):

                if values_close(

                    ratio,

                    calculated_ratio,

                    relative_tolerance=1e-3,

                ):

                    ratio_status = "PASS"

                    ratio_message = (

                        "Stored ON/OFF ratio agrees with "
                        "ROFF/RON."

                    )

                else:

                    ratio_status = "FAIL"

                    ratio_message = (

                        "Stored ON/OFF ratio does not agree "
                        "with ROFF/RON."

                    )


                audit_rows.append({

                    "device_id":
                        device_id,

                    "property_name":
                        "profile_ratio_consistency",

                    "simulator_value":
                        ratio,

                    "trace_value":
                        calculated_ratio,

                    "value_type":
                        "DERIVED_CHECK",

                    "evidence_class":
                        "INTERNAL_CHECK",

                    "status":
                        ratio_status,

                    "doi_present":
                        True,

                    "page_present":
                        True,

                    "figure_or_table_present":
                        True,

                    "message":
                        ratio_message,

                })


                # --------------------------------------------
                # Also require provenance for the stored ratio
                # if it exists in device_profiles.csv.
                # --------------------------------------------

                audit_rows.append(

                    audit_property(

                        device_id=device_id,

                        property_name="on_off_ratio",

                        simulator_value=ratio,

                        required=True,

                        expected_numeric=ratio,

                    )

                )


    elif parameter_source == "normalized_from_ratio":

        if pd.isna(
            ratio
        ):

            audit_rows.append({

                "device_id":
                    device_id,

                "property_name":
                    "on_off_ratio",

                "simulator_value":
                    np.nan,

                "trace_value":
                    np.nan,

                "value_type":
                    "",

                "evidence_class":
                    "NONE",

                "status":
                    "FAIL",

                "doi_present":
                    False,

                "page_present":
                    False,

                "figure_or_table_present":
                    False,

                "message":
                    (
                        "Simulator uses normalized ratio, "
                        "but profile ratio is missing."
                    ),

            })


        else:

            audit_rows.append(

                audit_property(

                    device_id=device_id,

                    property_name="on_off_ratio",

                    simulator_value=ratio,

                    required=True,

                    expected_numeric=ratio,

                )

            )


    else:

        audit_rows.append({

            "device_id":
                device_id,

            "property_name":
                "parameter_source",

            "simulator_value":
                parameter_source,

            "trace_value":
                np.nan,

            "value_type":
                "",

            "evidence_class":
                "NONE",

            "status":
                "FAIL",

            "doi_present":
                False,

            "page_present":
                False,

            "figure_or_table_present":
                False,

            "message":
                (
                    "Unsupported or insufficient "
                    "conductance parameter source."
                ),

        })


    # ========================================================
    # 4. State-count provenance consistency
    #
    # Compare device_profiles.state_count_status against
    # source_traceability.value_type.
    # ========================================================

    state_trace = get_trace_row(

        device_id,

        "conductance_states",

    )


    if state_trace is not None:

        trace_state_type = clean_text(

            state_trace[
                "value_type"
            ]

        ).upper()


        expected_pairs = {

            "REPORTED": {
                "REPORTED",
                "EXPERIMENTAL",
            },

            "DERIVED": {
                "DERIVED",
            },

            "NOT_REPORTED": {
                "NOT_REPORTED",
            },

            "NOT_APPLICABLE": {
                "NOT_APPLICABLE",
            },

        }


        accepted_types = expected_pairs.get(

            state_count_status,

            set(),

        )


        if (
            trace_state_type
            in accepted_types
        ):

            consistency_status = "PASS"

            consistency_message = (

                "state_count_status agrees with "
                "traceability evidence type."

            )


        else:

            consistency_status = "WARN"

            consistency_message = (

                "state_count_status and traceability "
                "value_type should be reviewed."

            )


        audit_rows.append({

            "device_id":
                device_id,

            "property_name":
                "state_count_status_consistency",

            "simulator_value":
                state_count_status,

            "trace_value":
                trace_state_type,

            "value_type":
                "CONSISTENCY_CHECK",

            "evidence_class":
                "INTERNAL_CHECK",

            "status":
                consistency_status,

            "doi_present":
                True,

            "page_present":
                True,

            "figure_or_table_present":
                True,

            "message":
                consistency_message,

        })


# ============================================================
# Build audit dataframe
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)


# ============================================================
# Device-level summary
# ============================================================

device_summary_rows = []


for device_id, group in audit_df.groupby(
    "device_id"
):

    fail_count = int(

        (
            group[
                "status"
            ]
            ==
            "FAIL"
        ).sum()

    )


    warn_count = int(

        (
            group[
                "status"
            ]
            ==
            "WARN"
        ).sum()

    )


    pass_count = int(

        (
            group[
                "status"
            ]
            ==
            "PASS"
        ).sum()

    )


    assumption_count = int(

        (
            group[
                "evidence_class"
            ]
            ==
            "ASSUMPTION"
        ).sum()

    )


    if fail_count > 0:

        overall_status = "FAIL"


    elif warn_count > 0:

        overall_status = "PASS_WITH_WARNINGS"


    else:

        overall_status = "PASS"


    device_summary_rows.append({

        "device_id":
            device_id,

        "overall_status":
            overall_status,

        "pass_checks":
            pass_count,

        "warning_checks":
            warn_count,

        "failed_checks":
            fail_count,

        "assumption_inputs":
            assumption_count,

    })


device_summary = pd.DataFrame(
    device_summary_rows
)


# ============================================================
# Save detailed report
# ============================================================

Path(
    "results/tables"
).mkdir(

    parents=True,

    exist_ok=True,

)


audit_df.to_csv(

    OUTPUT_FILE,

    index=False,

)


# ============================================================
# Display
# ============================================================

print()

print(
    "SIMULATOR INPUT TRACEABILITY AUDIT"
)

print(
    "========================================"
)


print(
    "Simulated devices:",
    len(
        used_device_ids
    )
)


print(
    "Detailed checks:",
    len(
        audit_df
    )
)


print()

print(
    "DEVICE SUMMARY"
)

print(
    "----------------------------------------"
)


print(

    device_summary.to_string(
        index=False
    )

)


# ============================================================
# Detailed checks
# ============================================================

print()

print(
    "DETAILED CHECKS"
)

print(
    "----------------------------------------"
)


display_columns = [

    "device_id",

    "property_name",

    "simulator_value",

    "trace_value",

    "value_type",

    "status",

    "message",

]


print(

    audit_df[

        display_columns

    ].to_string(
        index=False
    )

)


# ============================================================
# Evidence-type summary
# ============================================================

print()

print(
    "TRACEABILITY EVIDENCE USED"
)

print(
    "----------------------------------------"
)


evidence_summary = (

    audit_df[

        ~audit_df[
            "value_type"
        ].isin(

            [
                "CONSISTENCY_CHECK",
                "DERIVED_CHECK",
            ]

        )

    ]

    [
        "value_type"
    ]

    .value_counts(
        dropna=False
    )

)


print(
    evidence_summary
)


# ============================================================
# Assumption warning
# ============================================================

assumed_rows = audit_df[

    audit_df[
        "value_type"
    ]
    ==
    "ASSUMED"

]


print()

print(
    "ASSUMPTION-BASED SIMULATOR INPUTS"
)

print(
    "----------------------------------------"
)


if assumed_rows.empty:

    print(
        "None."
    )


else:

    print(

        assumed_rows[

            [
                "device_id",
                "property_name",
                "simulator_value",
                "message",
            ]

        ].to_string(
            index=False
        )

    )


# ============================================================
# Final project-level status
# ============================================================

total_failures = int(

    (
        audit_df[
            "status"
        ]
        ==
        "FAIL"
    ).sum()

)


total_warnings = int(

    (
        audit_df[
            "status"
        ]
        ==
        "WARN"
    ).sum()

)


print()

print(
    "PROJECT TRACEABILITY STATUS"
)

print(
    "----------------------------------------"
)


if total_failures > 0:

    print(
        "FAIL"
    )

    print(

        f"{total_failures} simulator-input checks "
        "still lack valid traceability."

    )


elif total_warnings > 0:

    print(
        "PASS WITH WARNINGS"
    )

    print(

        "All required simulator inputs are traceable, "
        "but some rely on explicit modeling assumptions "
        "or metadata that should remain visible."

    )


else:

    print(
        "PASS"
    )

    print(

        "All currently used simulator inputs passed "
        "the automated traceability checks."

    )


print()

print(
    "IMPORTANT:"
)


print(

    "PASS means the simulator input can be traced to "
    "a documented experimental, reported, derived, "
    "or explicitly declared modeling source."

)


print(

    "It does NOT mean the simulator itself has been "
    "experimentally validated against fabricated hardware."

)


print()

print(
    "Saved detailed audit to:"
)

print(
    OUTPUT_FILE
)