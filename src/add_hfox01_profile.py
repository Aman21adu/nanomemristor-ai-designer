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
    / "hfox01_derived_operating_point.csv"
)


# ============================================================
# PRIMARY EXPERIMENTAL STUDY
# ============================================================

PRIMARY_TITLE = (
    "Modeling and Simulation of Correlated Cycle-to-Cycle "
    "Variability in the Current-Voltage Hysteresis Loops "
    "of RRAM Devices"
)

PRIMARY_DOI = "10.1109/TNANO.2024.3485213"
PRIMARY_STUDY_ID = PRIMARY_DOI


# ============================================================
# SUPPORTING SAME-STACK MULTILEVEL STUDY
# ============================================================

SUPPORTING_TITLE = (
    "Investigation of the multilevel capability of "
    "TiN/Ti/HfO2/W resistive switching devices by sweep "
    "and pulse programming"
)

SUPPORTING_DOI = "10.1016/j.mee.2017.11.007"
SUPPORTING_STUDY_ID = SUPPORTING_DOI


# ============================================================
# FIGURE-DERIVED OPERATING POINT
#
# The 2024 paper states that HRS and LRS currents are
# evaluated at V = -0.2 V.
#
# Figure 9 shows the experimental time series and dashed
# experimental mean values.
#
# The currents below are VISUAL APPROXIMATIONS from the
# dashed mean lines in Figure 9:
#
#   LRS mean current ~ 3.59 mA
#   HRS mean current ~ 0.079 mA
#
# These are therefore DERIVED APPROXIMATIONS, not directly
# tabulated/reported numerical values.
# ============================================================

READ_VOLTAGE_V = -0.2

FIGURE_LRS_MEAN_CURRENT_MA = 3.59
FIGURE_HRS_MEAN_CURRENT_MA = 0.079


def resistance_from_current(
    voltage_v,
    current_ma,
):
    current_a = abs(float(current_ma)) * 1e-3

    if current_a <= 0:
        raise ValueError(
            "Current must be positive."
        )

    return abs(float(voltage_v)) / current_a


RON_OHM = resistance_from_current(
    READ_VOLTAGE_V,
    FIGURE_LRS_MEAN_CURRENT_MA,
)

ROFF_OHM = resistance_from_current(
    READ_VOLTAGE_V,
    FIGURE_HRS_MEAN_CURRENT_MA,
)

ON_OFF_RATIO = ROFF_OHM / RON_OHM


# Store conservative precision because the source values were
# visually estimated from a plotted dashed mean line.
RON_STORED = round(RON_OHM, 1)
ROFF_STORED = round(ROFF_OHM, -1)
ON_OFF_STORED = round(ON_OFF_RATIO, 1)


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def ensure_study_id_column(
    dataframe,
):
    if "study_id" not in dataframe.columns:
        dataframe.insert(
            1,
            "study_id",
            "",
        )

    return dataframe


# ============================================================
# LOAD DATA
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


devices = ensure_study_id_column(
    devices
)

trace = ensure_study_id_column(
    trace
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_device_columns = {
    "device_id",
    "study_id",
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
    "study_id",
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
    set(devices.columns)
)


missing_trace_columns = (
    required_trace_columns
    -
    set(trace.columns)
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
# LOCATE HfOx_01
# ============================================================

DEVICE_ID = "HfOx_01"

matches = (
    devices["device_id"] == DEVICE_ID
)


if int(matches.sum()) != 1:
    raise ValueError(
        f"{DEVICE_ID} must appear exactly once in "
        "device_profiles.csv."
    )


# ============================================================
# UPDATE DEVICE PROFILE
#
# study_id remains the PRIMARY 2024 experimental study.
#
# The supporting 2018 study is recorded at the PROPERTY level
# in source_traceability.csv for conductance_mode.
# ============================================================

devices.loc[
    matches,
    "study_id",
] = PRIMARY_STUDY_ID


devices.loc[
    matches,
    "ron_ohm",
] = RON_STORED


devices.loc[
    matches,
    "roff_ohm",
] = ROFF_STORED


devices.loc[
    matches,
    "on_off_ratio",
] = ON_OFF_STORED


devices.loc[
    matches,
    "read_voltage_v",
] = READ_VOLTAGE_V


devices.loc[
    matches,
    "conductance_mode",
] = "GRADUAL_MULTILEVEL"


# No fixed physical state count is invented.
devices.loc[
    matches,
    "conductance_states",
] = pd.NA


devices.loc[
    matches,
    "state_count_status",
] = "NOT_REPORTED"


devices.loc[
    matches,
    "state_count_notes",
] = (
    "The 2024 primary study evaluates HRS and LRS as "
    "the two extreme conduction states but does not report "
    "a fixed physical conductance-state count. A 2018 "
    "same-stack TiN/Ti/HfO2/W study experimentally "
    "demonstrates multilevel conductance tuning, but no "
    "single fixed state count is imposed here."
)


old_notes = clean_text(
    devices.loc[
        matches,
        "notes",
    ].iloc[0]
)


new_note = (
    "RON/ROFF and ON/OFF ratio are approximate values "
    "derived from the experimental mean-current dashed "
    "lines in Figure 9 of DOI "
    f"{PRIMARY_DOI} at -0.2 V "
    f"(LRS ~{FIGURE_LRS_MEAN_CURRENT_MA} mA, "
    f"HRS ~{FIGURE_HRS_MEAN_CURRENT_MA} mA). "
    "GRADUAL_MULTILEVEL behavior is supported by the "
    "same-stack multilevel study DOI "
    f"{SUPPORTING_DOI}. "
    "The figure-derived electrical values are approximate "
    "and must not be described as directly tabulated."
)


if new_note not in old_notes:
    if old_notes:
        devices.loc[
            matches,
            "notes",
        ] = old_notes + "; " + new_note
    else:
        devices.loc[
            matches,
            "notes",
        ] = new_note


# ============================================================
# BUILD DERIVATION RECORD
# ============================================================

derivation = pd.DataFrame(
    [
        {
            "device_id":
                DEVICE_ID,

            "primary_study_id":
                PRIMARY_STUDY_ID,

            "primary_paper_doi":
                PRIMARY_DOI,

            "supporting_study_id":
                SUPPORTING_STUDY_ID,

            "supporting_paper_doi":
                SUPPORTING_DOI,

            "read_voltage_v":
                READ_VOLTAGE_V,

            "figure":
                "Figure 9",

            "figure_lrs_mean_current_ma_approx":
                FIGURE_LRS_MEAN_CURRENT_MA,

            "figure_hrs_mean_current_ma_approx":
                FIGURE_HRS_MEAN_CURRENT_MA,

            "derived_ron_ohm_unrounded":
                RON_OHM,

            "derived_roff_ohm_unrounded":
                ROFF_OHM,

            "derived_on_off_ratio_unrounded":
                ON_OFF_RATIO,

            "stored_ron_ohm":
                RON_STORED,

            "stored_roff_ohm":
                ROFF_STORED,

            "stored_on_off_ratio":
                ON_OFF_STORED,

            "conductance_mode":
                "GRADUAL_MULTILEVEL",

            "state_count_status":
                "NOT_REPORTED",

            "derivation_method":
                (
                    "Visual estimate of the experimental "
                    "mean-current dashed lines in Figure 9; "
                    "R=|V|/|I|; RON from LRS mean current; "
                    "ROFF from HRS mean current; "
                    "ON/OFF=ROFF/RON."
                ),

            "precision_warning":
                (
                    "Figure-derived approximate values. "
                    "Stored values are intentionally rounded "
                    "to avoid false numerical precision."
                ),
        }
    ]
)


# ============================================================
# TRACEABILITY ROWS
#
# Existing HfOx_01 rows not owned by this updater are kept.
# ============================================================

new_trace_rows = [
    {
        "device_id":
            DEVICE_ID,

        "study_id":
            PRIMARY_STUDY_ID,

        "property_name":
            "ron_ohm",

        "value":
            RON_STORED,

        "unit":
            "ohm",

        "value_type":
            "DERIVED",

        "source_title":
            PRIMARY_TITLE,

        "doi":
            PRIMARY_DOI,

        "page":
            "763",

        "figure_or_table":
            "Figure 9(a)",

        "source_note":
            (
                "Approximate RON derived from the "
                f"experimental LRS mean-current dashed line "
                f"in Figure 9 at -0.2 V. "
                f"Visual mean estimate ~"
                f"{FIGURE_LRS_MEAN_CURRENT_MA} mA; "
                "R=|V|/|I|. This is figure-derived, "
                "not a directly tabulated resistance."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            PRIMARY_STUDY_ID,

        "property_name":
            "roff_ohm",

        "value":
            ROFF_STORED,

        "unit":
            "ohm",

        "value_type":
            "DERIVED",

        "source_title":
            PRIMARY_TITLE,

        "doi":
            PRIMARY_DOI,

        "page":
            "763",

        "figure_or_table":
            "Figure 9(b)",

        "source_note":
            (
                "Approximate ROFF derived from the "
                f"experimental HRS mean-current dashed line "
                f"in Figure 9 at -0.2 V. "
                f"Visual mean estimate ~"
                f"{FIGURE_HRS_MEAN_CURRENT_MA} mA; "
                "R=|V|/|I|. This is figure-derived, "
                "not a directly tabulated resistance."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            PRIMARY_STUDY_ID,

        "property_name":
            "on_off_ratio",

        "value":
            ON_OFF_STORED,

        "unit":
            "ROFF/RON",

        "value_type":
            "DERIVED",

        "source_title":
            PRIMARY_TITLE,

        "doi":
            PRIMARY_DOI,

        "page":
            "763",

        "figure_or_table":
            "Figure 9",

        "source_note":
            (
                "Approximate ON/OFF resistance ratio derived "
                "as ROFF/RON from the Figure-9 mean-current "
                "estimates. This value is approximate."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            SUPPORTING_STUDY_ID,

        "property_name":
            "conductance_mode",

        "value":
            "GRADUAL_MULTILEVEL",

        "unit":
            "",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SUPPORTING_TITLE,

        "doi":
            SUPPORTING_DOI,

        "page":
            "148-153",

        "figure_or_table":
            "Multilevel sweep/pulse experiments",

        "source_note":
            (
                "Supporting same-stack TiN/Ti/HfO2/W "
                "experimental study reports clearly "
                "distinguishable multilevel states and "
                "continuous dependence of filamentary "
                "conductance on programming conditions. "
                "Used only to support the simulator behavior "
                "classification, not to replace the primary "
                "2024 electrical operating-point data."
            ),
    },
]


new_trace_df = pd.DataFrame(
    new_trace_rows
)


# Align to current traceability schema.
for column in trace.columns:
    if column not in new_trace_df.columns:
        new_trace_df[column] = ""


for column in new_trace_df.columns:
    if column not in trace.columns:
        trace[column] = ""


new_trace_df = new_trace_df[
    trace.columns
]


# Remove only the rows this updater owns.
owned_properties = {
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "conductance_mode",
}


remove_mask = (
    (trace["device_id"] == DEVICE_ID)
    &
    trace["property_name"].isin(
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

duplicate_profiles = (
    devices[
        "device_id"
    ]
    .duplicated()
    .sum()
)


if duplicate_profiles:
    raise ValueError(
        "Duplicate device_id rows detected."
    )


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
            "study_id",
        ]
    ]

    raise ValueError(
        "Duplicate device/property traceability rows "
        "detected:\n"
        + duplicate_rows.to_string(
            index=False
        )
    )


profile = devices[
    devices["device_id"] == DEVICE_ID
].iloc[0]


calculated_ratio = (
    float(profile["roff_ohm"])
    /
    float(profile["ron_ohm"])
)


if not math.isclose(
    calculated_ratio,
    float(profile["on_off_ratio"]),
    rel_tol=5e-3,
    abs_tol=1e-8,
):
    raise ValueError(
        "Stored HfOx_01 ON/OFF ratio does not agree "
        "with stored ROFF/RON within rounding tolerance."
    )


if (
    clean_text(
        profile["study_id"]
    )
    != PRIMARY_STUDY_ID
):
    raise ValueError(
        "HfOx_01 primary study_id changed unexpectedly."
    )


mode_rows = trace[
    (trace["device_id"] == DEVICE_ID)
    &
    (trace["property_name"] == "conductance_mode")
]


if len(mode_rows) != 1:
    raise ValueError(
        "Expected exactly one conductance_mode "
        "traceability row for HfOx_01."
    )


if (
    clean_text(
        mode_rows.iloc[0]["study_id"]
    )
    != SUPPORTING_STUDY_ID
):
    raise ValueError(
        "The conductance-mode evidence is not linked "
        "to the supporting study."
    )


# ============================================================
# WRITE FILES
# ============================================================

devices.to_csv(
    DEVICE_FILE,
    index=False,
)


trace.to_csv(
    TRACE_FILE,
    index=False,
)


derivation.to_csv(
    DERIVATION_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "HfOx_01 evidence-backed profile expansion complete."
)
print()


print(
    devices[
        devices["device_id"] == DEVICE_ID
    ][
        [
            "device_id",
            "study_id",
            "ron_ohm",
            "roff_ohm",
            "on_off_ratio",
            "read_voltage_v",
            "conductance_mode",
            "state_count_status",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "Primary electrical study:",
    PRIMARY_DOI,
)


print(
    "Supporting behavior study:",
    SUPPORTING_DOI,
)


print(
    "Derived evidence file:",
    DERIVATION_FILE,
)


print()
print(
    "IMPORTANT:"
)


print(
    "  RON, ROFF and ON/OFF are approximate "
    "Figure-9-derived values, not tabulated measurements."
)


print(
    "  GRADUAL_MULTILEVEL is supported by a separate "
    "same-stack experimental study."
)


print(
    "  conductance_states remains NOT_REPORTED."
)


print(
    "  The profile's primary study_id remains the "
    "2024 variability study."
)
