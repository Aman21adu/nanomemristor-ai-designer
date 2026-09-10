from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_FILE = PROJECT_ROOT / "data" / "device_profiles.csv"
TRACE_FILE = PROJECT_ROOT / "data" / "source_traceability.csv"


# ============================================================
# DEVICE / STUDY
# ============================================================

DEVICE_ID = "HfZrOx_01"
STUDY_ID = "10.1002/pssr.202000524"

TECHNOLOGY_FAMILY = "HfZrOx"

DEVICE_STACK = "TiN/HfZrO4/TiOx/TiN"
ACTIVE_MATERIAL = "HfZrO4/TiOx"

SOURCE_TITLE = (
    "Ferroelectric, Analog Resistive Switching in "
    "Back-End-of-Line Compatible TiN/HfZrO4/TiOx Junctions"
)

DOI = STUDY_ID
YEAR = 2021


# ============================================================
# EVIDENCE-BACKED OPERATING POINT
#
# Published paper:
#   DOI 10.1002/pssr.202000524
#
# Same-authors NVMW extended abstract explicitly states:
#
# - resistance is measured with Vread = 100 mV
#   (non-destructive read-out)
# - switching is non-volatile, analog, reversible
# - resistance gradually increases/decreases during RESET/SET
# - intermediate states are reached by partial domain switching
# - after wake-up, ON/OFF increases from 1.3 to 2.0
#
# We use the post-wake-up ON/OFF = 2.0 operating regime.
#
# IMPORTANT:
# No absolute RON/ROFF values are inserted here because the
# text source used for this update does not provide a clean
# tabulated pair. The current simulator can normalize from the
# reported ON/OFF ratio.
# ============================================================

ON_OFF_RATIO = 2.0
READ_VOLTAGE_V = 0.1

CONDUCTANCE_MODE = "ANALOG"

STATE_COUNT_STATUS = "NOT_REPORTED"


# ============================================================
# LOAD CURRENT DATA
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


# ============================================================
# REQUIRED COLUMNS
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
    "variation_pct",
    "read_noise",
    "drift",
    "read_voltage_v",
    "set_voltage_v",
    "reset_voltage_v",
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
# BUILD / REPLACE DEVICE PROFILE
# ============================================================

profile = {
    "device_id":
        DEVICE_ID,

    "study_id":
        STUDY_ID,

    "technology_family":
        TECHNOLOGY_FAMILY,

    "device_stack":
        DEVICE_STACK,

    "active_material":
        ACTIVE_MATERIAL,

    # Absolute RON/ROFF intentionally remain unknown.
    "ron_ohm":
        pd.NA,

    "roff_ohm":
        pd.NA,

    # Explicitly reported after wake-up.
    "on_off_ratio":
        ON_OFF_RATIO,

    # Intermediate states are demonstrated but no fixed
    # physical state count is reported.
    "conductance_states":
        pd.NA,

    "variation_pct":
        pd.NA,

    "read_noise":
        pd.NA,

    # Drift is qualitatively reported, but no single scalar
    # compatible with the current schema is inserted.
    "drift":
        pd.NA,

    "read_voltage_v":
        READ_VOLTAGE_V,

    # Switching/write behavior is reported, but we do not
    # collapse the varying write thresholds/pulse conditions
    # into one potentially misleading scalar SET/RESET value.
    "set_voltage_v":
        pd.NA,

    "reset_voltage_v":
        pd.NA,

    "source_title":
        SOURCE_TITLE,

    "doi":
        DOI,

    "year":
        YEAR,

    "notes":
        (
            "BEOL-compatible ferroelectric analog resistive "
            "memory; 4.5 nm HfZrO4 and 3.5 nm TiOx layers; "
            "100 mV non-destructive read-out; reversible "
            "gradual SET/RESET with intermediate resistance "
            "states; after ferroelectric wake-up the reported "
            "ON/OFF ratio increases from 1.3 to 2.0. "
            "The profile uses the post-wake-up ON/OFF=2.0 "
            "regime. Absolute RON/ROFF are not inserted, so "
            "the simulator must use normalized conductance "
            "from the reported ratio."
        ),

    # The source repeatedly describes the device as an
    # analog resistive memory. This follows the project's
    # existing taxonomy for no-fixed-state analog devices.
    "conductance_mode":
        CONDUCTANCE_MODE,

    "state_count_status":
        STATE_COUNT_STATUS,

    "state_count_notes":
        (
            "The study demonstrates intermediate resistance "
            "states and analog reversible switching, including "
            "states reached by switching only a fraction of "
            "ferroelectric domains, but it does not report a "
            "fixed physical conductance-state count."
        ),
}


existing_mask = (
    devices["device_id"] == DEVICE_ID
)


if int(existing_mask.sum()) > 1:
    raise ValueError(
        f"{DEVICE_ID} appears more than once in "
        "device_profiles.csv."
    )


profile_df = pd.DataFrame(
    [profile]
)[devices.columns]


if int(existing_mask.sum()) == 1:

    devices = devices[
        ~existing_mask
    ].copy()


devices = pd.concat(
    [
        devices,
        profile_df,
    ],
    ignore_index=True,
)


# ============================================================
# TRACEABILITY
# ============================================================

trace_rows = [
    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "device_stack",

        "value":
            DEVICE_STACK,

        "unit":
            "",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Device fabrication / Figure 1",

        "source_note":
            (
                "TiN(10 nm)/HfZrO4(4.5 nm)/"
                "TiOx(3.5 nm)/TiN(20 nm) asymmetric "
                "stack is reported."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "active_material",

        "value":
            ACTIVE_MATERIAL,

        "unit":
            "",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Device fabrication",

        "source_note":
            (
                "Ferroelectric HfZrO4 is combined with a "
                "semiconducting TiOx interlayer."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "hzo_thickness_nm",

        "value":
            4.5,

        "unit":
            "nm",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Device fabrication",

        "source_note":
            "Reported HfZrO4 thickness.",
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "tiox_thickness_nm",

        "value":
            3.5,

        "unit":
            "nm",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Device fabrication",

        "source_note":
            "Reported TiOx interlayer thickness.",
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "read_voltage",

        "value":
            READ_VOLTAGE_V,

        "unit":
            "V",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Figure 1 / write-read experiment",

        "source_note":
            (
                "Resistance is measured approximately ten "
                "seconds after programming using a 100 mV "
                "non-destructive read voltage."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "on_off_ratio",

        "value":
            ON_OFF_RATIO,

        "unit":
            "HRS/LRS resistance ratio",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "2",

        "figure_or_table":
            "Figure 1 / wake-up discussion",

        "source_note":
            (
                "After ferroelectric wake-up, the study "
                "reports that the ON/OFF ratio increases "
                "from 1.3 to 2.0. The profile uses the "
                "post-wake-up value 2.0."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "conductance_mode",

        "value":
            CONDUCTANCE_MODE,

        "unit":
            "",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1-2",

        "figure_or_table":
            "Figure 1",

        "source_note":
            (
                "The device is explicitly described as an "
                "analog resistive memory. Resistance changes "
                "gradually during RESET and SET, and "
                "intermediate states are experimentally "
                "observed."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "conductance_states",

        "value":
            pd.NA,

        "unit":
            "",

        "value_type":
            "NOT_REPORTED",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1-2",

        "figure_or_table":
            "Figure 1 / mechanism discussion",

        "source_note":
            (
                "Intermediate states are reached by partial "
                "ferroelectric-domain switching, but no "
                "fixed physical conductance-state count is "
                "reported."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "switching_mechanism",

        "value":
            "FERROELECTRIC_POLARIZATION_WITH_OXYGEN_VACANCY_REDISTRIBUTION",

        "unit":
            "",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "2",

        "figure_or_table":
            "Figure 3 / mechanism discussion",

        "source_note":
            (
                "Resistive switching is linked to "
                "ferroelectric polarization; wake-up behavior "
                "is explained with redistribution of oxygen "
                "vacancies, and partial domain switching "
                "produces intermediate states."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "wake_up_cycles",

        "value":
            "1E5",

        "unit":
            "cycles",

        "value_type":
            "EXPERIMENTAL",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "2",

        "figure_or_table":
            "Figure 2",

        "source_note":
            (
                "The reported wake-up comparison includes "
                "the device after 1E5 cycles."
            ),
    },

    {
        "device_id":
            DEVICE_ID,

        "study_id":
            STUDY_ID,

        "property_name":
            "drift",

        "value":
            pd.NA,

        "unit":
            "",

        "value_type":
            "NOT_REPORTED",

        "source_title":
            SOURCE_TITLE,

        "doi":
            DOI,

        "page":
            "1",

        "figure_or_table":
            "Figure 1 discussion",

        "source_note":
            (
                "The study qualitatively reports cycle-to-"
                "cycle LRS drift of a few tens of kOhms and "
                "a longer-term resistance increase of about "
                "0.5 GOhm after six days, but these are not "
                "collapsed into the project's current single "
                "drift scalar because the schema does not "
                "encode the required time/condition context."
            ),
    },
]


new_trace = pd.DataFrame(
    trace_rows
)


# Align traceability columns.
for column in trace.columns:

    if column not in new_trace.columns:

        new_trace[column] = ""


for column in new_trace.columns:

    if column not in trace.columns:

        trace[column] = ""


new_trace = new_trace[
    trace.columns
]


# Idempotent replacement: this script owns all traceability
# entries for the newly added device.
trace = trace[
    trace["device_id"] != DEVICE_ID
].copy()


trace = pd.concat(
    [
        trace,
        new_trace,
    ],
    ignore_index=True,
)


# ============================================================
# VALIDATION
# ============================================================

if devices["device_id"].duplicated().any():

    duplicate_ids = (
        devices.loc[
            devices["device_id"].duplicated(
                keep=False
            ),
            "device_id",
        ]
        .tolist()
    )

    raise ValueError(
        "Duplicate device profiles detected: "
        f"{duplicate_ids}"
    )


device_trace = trace[
    trace["device_id"] == DEVICE_ID
]


duplicate_trace = device_trace.duplicated(
    subset=[
        "device_id",
        "property_name",
    ]
).sum()


if duplicate_trace:

    raise ValueError(
        f"{DEVICE_ID} contains duplicate "
        "device/property traceability rows."
    )


saved_profile = devices[
    devices["device_id"] == DEVICE_ID
].iloc[0]


if float(
    saved_profile["on_off_ratio"]
) != ON_OFF_RATIO:

    raise ValueError(
        "Stored ON/OFF ratio changed unexpectedly."
    )


if float(
    saved_profile["read_voltage_v"]
) != READ_VOLTAGE_V:

    raise ValueError(
        "Stored read voltage changed unexpectedly."
    )


if (
    str(
        saved_profile["conductance_mode"]
    ).strip().upper()
    != CONDUCTANCE_MODE
):

    raise ValueError(
        "Stored conductance mode changed unexpectedly."
    )


# ============================================================
# WRITE
# ============================================================

devices.to_csv(
    DEVICE_FILE,
    index=False,
)


trace.to_csv(
    TRACE_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print(
    "HfZrOx_01 low-ON/OFF literature profile added."
)
print()


print(
    devices[
        devices["device_id"] == DEVICE_ID
    ][
        [
            "device_id",
            "study_id",
            "technology_family",
            "device_stack",
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
    "Primary study:",
    STUDY_ID,
)


print(
    "Reported post-wake-up ON/OFF:",
    ON_OFF_RATIO,
)


print(
    "Reported read voltage:",
    READ_VOLTAGE_V,
    "V",
)


print(
    "Absolute RON/ROFF:",
    "not inserted"
)


print()
print(
    "IMPORTANT:"
)


print(
    "  The simulator should therefore use "
    "normalized conductance from the reported ratio."
)


print(
    "  Analog/intermediate-state behavior is "
    "experimentally supported; no fixed state count "
    "is invented."
)


print(
    "  This device intentionally extends literature "
    "coverage into the low ON/OFF ~2 regime."
)
