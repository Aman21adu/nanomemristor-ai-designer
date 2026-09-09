from pathlib import Path
import math

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_FILE = PROJECT_ROOT / "data" / "device_profiles.csv"
TRACE_FILE = PROJECT_ROOT / "data" / "source_traceability.csv"
DERIVATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "tiox02_derived_operating_points.csv"
)


# ============================================================
# SOURCE INFORMATION
# ============================================================

PAPER_TITLE = (
    "An electrical characterisation methodology for "
    "identifying the switching mechanism in TiO2 "
    "memristive stacks"
)

PAPER_DOI = "10.1038/s41598-019-44607-3"
RAW_DATA_DOI = "10.5258/SOTON/D0930"

# The three electrode variants come from the SAME study.
STUDY_ID = PAPER_DOI

# The paper states that HRS/LRS should be evaluated in the
# non-switching regime and that the analysis uses |V| < 0.5 V.
#
# +0.10 V is therefore selected here as a consistent analysis
# point. It is NOT claimed to be a paper-reported standard
# read voltage.
ANALYSIS_TEMPERATURE_K = 300.0
ANALYSIS_READ_VOLTAGE_V = 0.10


# ============================================================
# DERIVATION DATA
#
# Raw values come from the openly available workbook:
#
# Data_for_An_electrical_characterisation_methodology_for_
# identifying_the_switching_mechanism_in_TiO2_memristive_
# stacks.xlsx
#
# Positive-bias branches at 300 K are linearly interpolated
# to +0.10 V.
#
# I(0.10) = I1 + (I2-I1)*(0.10-V1)/(V2-V1)
# R = V / I
#
# RON  = lower of the two branch resistances
# ROFF = higher of the two branch resistances
# ON/OFF = ROFF / RON
# ============================================================

DERIVATIONS = {
    "TiOx_02_Au": {
        "sheet": "Fig.2a",
        "stack": "Au/TiO2/Au",
        "top_electrode": "Au",

        "branch1_v1": 0.09670,
        "branch1_i1": 1.00223e-05,
        "branch1_v2": 0.14771,
        "branch1_i2": 1.69436e-05,

        "branch2_v1": 0.09646,
        "branch2_i1": 5.89325e-06,
        "branch2_v2": 0.14771,
        "branch2_i2": 9.42612e-06,
    },

    "TiOx_02_Ni": {
        "sheet": "Fig.2b",
        "stack": "Au/TiO2/Ni",
        "top_electrode": "Ni",

        "branch1_v1": 0.09848,
        "branch1_i1": 1.47150e-05,
        "branch1_v2": 0.14949,
        "branch1_i2": 2.31153e-05,

        "branch2_v1": 0.09856,
        "branch2_i1": 1.62994e-05,
        "branch2_v2": 0.14965,
        "branch2_i2": 2.59014e-05,
    },

    "TiOx_02_Pt": {
        "sheet": "Fig.2c",
        "stack": "Au/TiO2/Pt",
        "top_electrode": "Pt",

        "branch1_v1": 0.09662,
        "branch1_i1": 8.38973e-06,
        "branch1_v2": 0.14763,
        "branch1_i2": 1.38392e-05,

        "branch2_v1": 0.09646,
        "branch2_i1": 1.79445e-05,
        "branch2_v2": 0.14715,
        "branch2_i2": 2.77278e-05,
    },
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def interpolate_current(
    v1,
    i1,
    v2,
    i2,
    target_voltage,
):
    if not (
        min(v1, v2)
        <= target_voltage
        <= max(v1, v2)
    ):
        raise ValueError(
            "Target voltage is outside the interpolation "
            "interval."
        )

    return (
        i1
        +
        (
            i2 - i1
        )
        *
        (
            target_voltage - v1
        )
        /
        (
            v2 - v1
        )
    )


def derive_operating_point(
    device_id,
    record,
):
    branch1_current = interpolate_current(
        record["branch1_v1"],
        record["branch1_i1"],
        record["branch1_v2"],
        record["branch1_i2"],
        ANALYSIS_READ_VOLTAGE_V,
    )

    branch2_current = interpolate_current(
        record["branch2_v1"],
        record["branch2_i1"],
        record["branch2_v2"],
        record["branch2_i2"],
        ANALYSIS_READ_VOLTAGE_V,
    )

    branch1_resistance = (
        ANALYSIS_READ_VOLTAGE_V
        /
        branch1_current
    )

    branch2_resistance = (
        ANALYSIS_READ_VOLTAGE_V
        /
        branch2_current
    )

    ron_ohm = min(
        branch1_resistance,
        branch2_resistance,
    )

    roff_ohm = max(
        branch1_resistance,
        branch2_resistance,
    )

    on_off_ratio = (
        roff_ohm
        /
        ron_ohm
    )

    return {
        "device_id":
            device_id,

        "study_id":
            STUDY_ID,

        "device_stack":
            record["stack"],

        "top_electrode":
            record["top_electrode"],

        "raw_workbook_sheet":
            record["sheet"],

        "temperature_K":
            ANALYSIS_TEMPERATURE_K,

        "selected_read_voltage_V":
            ANALYSIS_READ_VOLTAGE_V,

        "branch1_v1_V":
            record["branch1_v1"],

        "branch1_i1_A":
            record["branch1_i1"],

        "branch1_v2_V":
            record["branch1_v2"],

        "branch1_i2_A":
            record["branch1_i2"],

        "branch1_interpolated_current_A":
            branch1_current,

        "branch1_resistance_ohm":
            branch1_resistance,

        "branch2_v1_V":
            record["branch2_v1"],

        "branch2_i1_A":
            record["branch2_i1"],

        "branch2_v2_V":
            record["branch2_v2"],

        "branch2_i2_A":
            record["branch2_i2"],

        "branch2_interpolated_current_A":
            branch2_current,

        "branch2_resistance_ohm":
            branch2_resistance,

        "ron_ohm":
            ron_ohm,

        "roff_ohm":
            roff_ohm,

        "on_off_ratio":
            on_off_ratio,

        "paper_doi":
            PAPER_DOI,

        "raw_dataset_doi":
            RAW_DATA_DOI,

        "derivation_method":
            (
                "300 K raw Figure-2 I-V data; "
                "linear interpolation of the two "
                "positive-bias non-switching branches "
                "to +0.10 V; R=V/I; RON=min(branch R); "
                "ROFF=max(branch R)."
            ),

        "read_voltage_evidence":
            (
                "ASSUMED analysis point: +0.10 V was "
                "selected inside the paper's |V|<0.5 V "
                "non-switching regime; it is not a "
                "paper-reported standard read voltage."
            ),
    }


def ensure_study_id_column(
    dataframe,
):
    if "study_id" not in dataframe.columns:

        dataframe.insert(
            1,
            "study_id",
            "",
        )

    for index, row in dataframe.iterrows():

        current = clean_text(
            row.get(
                "study_id",
                "",
            )
        )

        if current:
            continue

        doi = clean_text(
            row.get(
                "doi",
                "",
            )
        )

        source_title = clean_text(
            row.get(
                "source_title",
                "",
            )
        )

        if doi:

            dataframe.at[
                index,
                "study_id",
            ] = doi

        elif source_title:

            dataframe.at[
                index,
                "study_id",
            ] = (
                "TITLE::"
                + source_title
            )

    return dataframe


# ============================================================
# LOAD CURRENT PROJECT DATA
# ============================================================

if not DEVICE_FILE.exists():
    raise FileNotFoundError(
        f"Missing: {DEVICE_FILE}"
    )

if not TRACE_FILE.exists():
    raise FileNotFoundError(
        f"Missing: {TRACE_FILE}"
    )


devices = pd.read_csv(
    DEVICE_FILE
)

trace = pd.read_csv(
    TRACE_FILE
)


required_device_columns = {
    "device_id",
    "technology_family",
    "device_stack",
    "active_material",
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "conductance_states",
    "read_voltage_v",
    "source_title",
    "doi",
    "year",
    "notes",
    "conductance_mode",
    "state_count_status",
    "state_count_notes",
}


required_trace_columns = {
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
}


missing_device_columns = (
    required_device_columns
    -
    set(
        devices.columns
    )
)


missing_trace_columns = (
    required_trace_columns
    -
    set(
        trace.columns
    )
)


if missing_device_columns:
    raise ValueError(
        "device_profiles.csv is missing columns: "
        f"{sorted(missing_device_columns)}"
    )


if missing_trace_columns:
    raise ValueError(
        "source_traceability.csv is missing columns: "
        f"{sorted(missing_trace_columns)}"
    )


# ============================================================
# ADD / POPULATE STUDY ID
# ============================================================

devices = ensure_study_id_column(
    devices
)

trace = ensure_study_id_column(
    trace
)


# ============================================================
# DERIVE THE THREE OPERATING POINTS
# ============================================================

derived_rows = []


for (
    device_id,
    record,
) in DERIVATIONS.items():

    derived_rows.append(
        derive_operating_point(
            device_id,
            record,
        )
    )


derived_df = pd.DataFrame(
    derived_rows
)


# ============================================================
# UPDATE DEVICE PROFILES
# ============================================================

for _, derived in derived_df.iterrows():

    device_id = derived[
        "device_id"
    ]


    matches = devices[
        "device_id"
    ] == device_id


    count = int(
        matches.sum()
    )


    if count != 1:

        raise ValueError(
            f"{device_id} has {count} rows in "
            "device_profiles.csv; expected exactly one."
        )


    devices.loc[
        matches,
        "study_id",
    ] = STUDY_ID


    devices.loc[
        matches,
        "ron_ohm",
    ] = round(
        float(
            derived[
                "ron_ohm"
            ]
        ),
        2,
    )


    devices.loc[
        matches,
        "roff_ohm",
    ] = round(
        float(
            derived[
                "roff_ohm"
            ]
        ),
        2,
    )


    devices.loc[
        matches,
        "on_off_ratio",
    ] = round(
        float(
            derived[
                "on_off_ratio"
            ]
        ),
        6,
    )


    devices.loc[
        matches,
        "read_voltage_v",
    ] = ANALYSIS_READ_VOLTAGE_V


    old_notes = clean_text(
        devices.loc[
            matches,
            "notes",
        ].iloc[0]
    )


    derivation_note = (
        "RON/ROFF and ON/OFF ratio derived from the "
        f"{derived['raw_workbook_sheet']} 300 K raw I-V "
        "dataset by linear interpolation to +0.10 V "
        "inside the non-switching regime; +0.10 V is "
        "an explicit analysis assumption rather than a "
        "paper-reported standard read voltage; "
        f"raw dataset DOI {RAW_DATA_DOI}"
    )


    if derivation_note not in old_notes:

        if old_notes:

            new_notes = (
                old_notes
                + "; "
                + derivation_note
            )

        else:

            new_notes = (
                derivation_note
            )

        devices.loc[
            matches,
            "notes",
        ] = new_notes


# ============================================================
# ADD / REPLACE TRACEABILITY ROWS
#
# One row per device/property is required by the audit script.
# ============================================================

new_trace_rows = []


for _, derived in derived_df.iterrows():

    device_id = derived[
        "device_id"
    ]

    sheet_name = derived[
        "raw_workbook_sheet"
    ]


    common_note = (
        f"Derived from openly available raw dataset "
        f"DOI {RAW_DATA_DOI}, sheet {sheet_name}, "
        f"300 K. Two positive-bias non-switching "
        f"branches were linearly interpolated to "
        f"+{ANALYSIS_READ_VOLTAGE_V:.2f} V. "
        f"Detailed calculation is stored in "
        f"data/tiox02_derived_operating_points.csv."
    )


    new_trace_rows.extend(
        [
            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "ron_ohm",

                "value":
                    round(
                        float(
                            derived[
                                "ron_ohm"
                            ]
                        ),
                        2,
                    ),

                "unit":
                    "ohm",

                "value_type":
                    "DERIVED",

                "source_title":
                    PAPER_TITLE,

                "doi":
                    PAPER_DOI,

                "page":
                    "6",

                "figure_or_table":
                    (
                        f"Figure 2 / raw sheet "
                        f"{sheet_name}"
                    ),

                "source_note":
                    (
                        common_note
                        + " RON is the lower of the two "
                        "derived branch resistances."
                    ),
            },

            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "roff_ohm",

                "value":
                    round(
                        float(
                            derived[
                                "roff_ohm"
                            ]
                        ),
                        2,
                    ),

                "unit":
                    "ohm",

                "value_type":
                    "DERIVED",

                "source_title":
                    PAPER_TITLE,

                "doi":
                    PAPER_DOI,

                "page":
                    "6",

                "figure_or_table":
                    (
                        f"Figure 2 / raw sheet "
                        f"{sheet_name}"
                    ),

                "source_note":
                    (
                        common_note
                        + " ROFF is the higher of the two "
                        "derived branch resistances."
                    ),
            },

            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "on_off_ratio",

                "value":
                    round(
                        float(
                            derived[
                                "on_off_ratio"
                            ]
                        ),
                        6,
                    ),

                "unit":
                    "ROFF/RON",

                "value_type":
                    "DERIVED",

                "source_title":
                    PAPER_TITLE,

                "doi":
                    PAPER_DOI,

                "page":
                    "6",

                "figure_or_table":
                    (
                        f"Figure 2 / raw sheet "
                        f"{sheet_name}"
                    ),

                "source_note":
                    (
                        common_note
                        + " ON/OFF ratio = ROFF/RON."
                    ),
            },

            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "read_voltage",

                "value":
                    ANALYSIS_READ_VOLTAGE_V,

                "unit":
                    "V",

                "value_type":
                    "ASSUMED",

                "source_title":
                    PAPER_TITLE,

                "doi":
                    PAPER_DOI,

                "page":
                    "6-7",

                "figure_or_table":
                    "Figure 2; Discussion",

                "source_note":
                    (
                        "+0.10 V is a project-selected "
                        "analysis/read point inside the "
                        "paper's stated |V|<0.5 V "
                        "non-switching regime. The paper "
                        "explicitly notes that read-out "
                        "resistance depends on read voltage. "
                        "This value is therefore an "
                        "assumption, not a directly reported "
                        "standard read voltage."
                    ),
            },

            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "analysis_temperature",

                "value":
                    ANALYSIS_TEMPERATURE_K,

                "unit":
                    "K",

                "value_type":
                    "ASSUMED",

                "source_title":
                    PAPER_TITLE,

                "doi":
                    PAPER_DOI,

                "page":
                    "6",

                "figure_or_table":
                    (
                        f"Figure 2 / raw sheet "
                        f"{sheet_name}"
                    ),

                "source_note":
                    (
                        "300 K is one experimentally measured "
                        "temperature in the source dataset and "
                        "is selected here as the common "
                        "operating-point temperature for the "
                        "three device variants."
                    ),
            },

            {
                "device_id":
                    device_id,

                "study_id":
                    STUDY_ID,

                "property_name":
                    "raw_dataset_doi",

                "value":
                    RAW_DATA_DOI,

                "unit":
                    "",

                "value_type":
                    "REPORTED",

                "source_title":
                    (
                        "Data for "
                        + PAPER_TITLE
                    ),

                "doi":
                    RAW_DATA_DOI,

                "page":
                    "Data repository",

                "figure_or_table":
                    sheet_name,

                "source_note":
                    (
                        "Open raw dataset used for the "
                        "300 K operating-point derivation."
                    ),
            },
        ]
    )


new_trace_df = pd.DataFrame(
    new_trace_rows
)


# Align columns:
# keep every original/current column and allow study_id.
for column in trace.columns:

    if column not in new_trace_df.columns:

        new_trace_df[
            column
        ] = ""


for column in new_trace_df.columns:

    if column not in trace.columns:

        trace[
            column
        ] = ""


new_trace_df = new_trace_df[
    trace.columns
]


# Remove only the properties this script owns, then append
# fresh deterministic rows. This makes the script idempotent.
owned_properties = {
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "read_voltage",
    "analysis_temperature",
    "raw_dataset_doi",
}


target_device_ids = set(
    DERIVATIONS.keys()
)


remove_mask = (
    trace[
        "device_id"
    ].isin(
        target_device_ids
    )
    &
    trace[
        "property_name"
    ].isin(
        owned_properties
    )
)


trace = trace[
    ~remove_mask
].copy()


trace = pd.concat(
    [
        trace,
        new_trace_df,
    ],
    ignore_index=True,
)


# ============================================================
# VALIDATION
# ============================================================

# No duplicate device/profile rows.
duplicate_devices = (
    devices[
        "device_id"
    ]
    .duplicated()
    .sum()
)


if duplicate_devices:

    raise ValueError(
        "Duplicate device_id rows detected after update."
    )


# No duplicate traceability row per device/property.
duplicate_trace = trace.duplicated(
    subset=[
        "device_id",
        "property_name",
    ]
).sum()


if duplicate_trace:

    duplicate_rows = trace[
        trace.duplicated(
            subset=[
                "device_id",
                "property_name",
            ],
            keep=False,
        )
    ][
        [
            "device_id",
            "property_name",
        ]
    ]

    raise ValueError(
        "Duplicate device/property traceability rows "
        "detected:\n"
        + duplicate_rows.to_string(
            index=False
        )
    )


# Stored ratios must agree with rounded ROFF/RON.
for device_id in sorted(
    target_device_ids
):

    row = devices[
        devices[
            "device_id"
        ]
        ==
        device_id
    ].iloc[0]


    calculated_ratio = (
        float(
            row[
                "roff_ohm"
            ]
        )
        /
        float(
            row[
                "ron_ohm"
            ]
        )
    )


    stored_ratio = float(
        row[
            "on_off_ratio"
        ]
    )


    if not math.isclose(
        calculated_ratio,
        stored_ratio,
        rel_tol=1e-3,
        abs_tol=1e-8,
    ):

        raise ValueError(
            f"{device_id}: stored ON/OFF ratio "
            "does not agree with ROFF/RON."
        )


# All three variants must share one study ID.
study_ids = set(
    devices[
        devices[
            "device_id"
        ].isin(
            target_device_ids
        )
    ][
        "study_id"
    ]
    .astype(str)
    .str.strip()
    .tolist()
)


if study_ids != {
    STUDY_ID
}:

    raise ValueError(
        "The three TiOx_02 variants do not share "
        "the expected study_id."
    )


# ============================================================
# WRITE UPDATED DATA
# ============================================================

devices.to_csv(
    DEVICE_FILE,
    index=False,
)


trace.to_csv(
    TRACE_FILE,
    index=False,
)


derived_df.to_csv(
    DERIVATION_FILE,
    index=False,
)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print()
print(
    "TiOx_02 literature-data expansion complete."
)
print()


summary_columns = [
    "device_id",
    "study_id",
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "read_voltage_v",
    "conductance_mode",
    "state_count_status",
]


print(
    devices[
        devices[
            "device_id"
        ].isin(
            sorted(
                target_device_ids
            )
        )
    ][
        summary_columns
    ]
    .sort_values(
        "device_id"
    )
    .to_string(
        index=False
    )
)


print()
print(
    "Shared study_id:",
    STUDY_ID,
)


print(
    "Raw dataset DOI:",
    RAW_DATA_DOI,
)


print(
    "Derivation evidence:",
    DERIVATION_FILE,
)


print()
print(
    "IMPORTANT:"
)


print(
    "  +0.10 V and 300 K are explicitly documented "
    "analysis choices."
)


print(
    "  The three TiOx_02 variants are distinct device "
    "stacks from ONE experimental study."
)


print(
    "  Do not rebuild the ML model until the new sweeps "
    "and study-aware validation changes are ready."
)
