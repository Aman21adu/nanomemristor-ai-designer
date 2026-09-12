from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEVICE_FILE = PROJECT_ROOT / "data" / "device_profiles.csv"
TRACE_FILE = PROJECT_ROOT / "data" / "source_traceability.csv"


# ============================================================
# PROFILE IDENTITY
# ============================================================

DEVICE_ID = "TiOx_04"

PRIMARY_STUDY_ID = "10.1007/s42835-019-00107-y"
PRIMARY_DOI = PRIMARY_STUDY_ID

PRIMARY_TITLE = (
    "Mimicking Synaptic Behaviors with Cross-Point Structured "
    "TiOx/TiOy-Based Filamentary RRAM for Neuromorphic Applications"
)

SUPPORTING_THESIS_ID = "10.23186/korea.000000081642.11009.0000815"

SUPPORTING_TITLE = (
    "Experimental study of TiOx/TiOy based filamentary RRAM "
    "as synaptic device for neuromorphic systems"
)

TECHNOLOGY_FAMILY = "TiOx"
ACTIVE_MATERIAL = "TiOx/TiOy"

# Primary/related sources establish Au top/bottom electrodes with
# the TiOx/TiOy bilayer. We do not encode an additional Ti layer
# because the available abstract-level evidence does not justify
# a more specific unambiguous slash-stack sequence.
DEVICE_STACK = "Au/TiOx/TiOy/Au"

YEAR = 2019


# ============================================================
# EVIDENCE-BACKED OPERATING POINT
# ============================================================

ON_OFF_RATIO = 76.0
READ_VOLTAGE_V = 0.2
FORMING_VOLTAGE_V = 5.62

CONDUCTANCE_MODE = "GRADUAL_MULTILEVEL"

# The related thesis demonstrates 50 conductance values under a
# 50-pulse LTP/LTD protocol, but this is protocol-dependent and
# is NOT treated as a fixed intrinsic physical state count.
CONDUCTANCE_STATES = pd.NA
STATE_COUNT_STATUS = "NOT_REPORTED"


# ============================================================
# LOAD DATA
# ============================================================

if not DEVICE_FILE.exists():
    raise FileNotFoundError(f"Missing: {DEVICE_FILE}")

if not TRACE_FILE.exists():
    raise FileNotFoundError(f"Missing: {TRACE_FILE}")

devices = pd.read_csv(DEVICE_FILE)
trace = pd.read_csv(TRACE_FILE)


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

missing_device_columns = required_device_columns - set(devices.columns)
missing_trace_columns = required_trace_columns - set(trace.columns)

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
# BUILD DEVICE PROFILE
# ============================================================

profile = {
    "device_id": DEVICE_ID,
    "study_id": PRIMARY_STUDY_ID,
    "technology_family": TECHNOLOGY_FAMILY,
    "device_stack": DEVICE_STACK,
    "active_material": ACTIVE_MATERIAL,

    # Absolute HRS/LRS resistance values are not safely available
    # from the evidence used here. The simulator should normalize
    # from the reported ON/OFF ratio.
    "ron_ohm": pd.NA,
    "roff_ohm": pd.NA,

    "on_off_ratio": ON_OFF_RATIO,
    "conductance_states": CONDUCTANCE_STATES,

    "variation_pct": pd.NA,
    "read_noise": pd.NA,
    "drift": pd.NA,

    "read_voltage_v": READ_VOLTAGE_V,

    # Do not invent scalar SET/RESET voltages from pulse protocols.
    "set_voltage_v": pd.NA,
    "reset_voltage_v": pd.NA,

    "source_title": PRIMARY_TITLE,
    "doi": PRIMARY_DOI,
    "year": YEAR,

    "notes": (
        "Cross-point TiOx/TiOy bilayer filamentary RRAM for "
        "neuromorphic synaptic operation. Reported active area "
        "20 x 20 um^2. Electroforming occurs at approximately "
        "5.62 V. Reported ON/OFF ratio is approximately 76 at "
        "0.2 V. LTP/LTD and STDP are demonstrated. The related "
        "2018 thesis for the same experimental platform reports "
        "gradual conductance modulation and 50 protocol-generated "
        "conductance values under a 50-pulse LTP/LTD experiment. "
        "Those 50 values are NOT treated as a fixed intrinsic "
        "physical state count. Absolute RON/ROFF are not inserted, "
        "so normalized conductance from ratio is required."
    ),

    "conductance_mode": CONDUCTANCE_MODE,

    "state_count_status": STATE_COUNT_STATUS,

    "state_count_notes": (
        "Gradual multi-level behavior is experimentally demonstrated. "
        "The related thesis reports 50 different conductance values "
        "during a 50-pulse LTP/LTD protocol, but this is a programming-"
        "protocol outcome rather than a reported fixed intrinsic state "
        "capacity. Therefore no fixed physical state count is imposed."
    ),
}


existing = devices["device_id"] == DEVICE_ID

if int(existing.sum()) > 1:
    raise ValueError(
        f"{DEVICE_ID} appears more than once in device_profiles.csv."
    )

if int(existing.sum()) == 1:
    devices = devices[~existing].copy()

profile_df = pd.DataFrame([profile])[devices.columns]

devices = pd.concat(
    [devices, profile_df],
    ignore_index=True,
)


# ============================================================
# TRACEABILITY
# ============================================================

trace_rows = [
    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "active_material",
        "value": ACTIVE_MATERIAL,
        "unit": "",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "Abstract / device description",
        "source_note": (
            "Primary journal article reports a TiOx/TiOy bilayer "
            "insulator in a cross-point filamentary RRAM."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": SUPPORTING_THESIS_ID,
        "property_name": "device_stack",
        "value": DEVICE_STACK,
        "unit": "",
        "value_type": "EXPERIMENTAL",
        "source_title": SUPPORTING_TITLE,
        "doi": SUPPORTING_THESIS_ID,
        "page": "Abstract; experimental section starts p.36",
        "figure_or_table": "Device structure / fabrication",
        "source_note": (
            "Related thesis for the same experimental platform states "
            "that Au thin films are used as both top and bottom "
            "electrodes with a TiOx/TiOy bilayer switching structure. "
            "The profile encodes the core Au/TiOx/TiOy/Au stack and "
            "does not add a more specific auxiliary Ti-layer position "
            "without unambiguous source text."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": SUPPORTING_THESIS_ID,
        "property_name": "device_area",
        "value": "20 x 20",
        "unit": "um^2",
        "value_type": "EXPERIMENTAL",
        "source_title": SUPPORTING_TITLE,
        "doi": SUPPORTING_THESIS_ID,
        "page": "Abstract",
        "figure_or_table": "Device description",
        "source_note": "Reported cross-point active area.",
    },

    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "forming_voltage",
        "value": FORMING_VOLTAGE_V,
        "unit": "V",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "Abstract / I-V characterization",
        "source_note": (
            "Journal article reports electroforming at approximately "
            "5.62 V."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "read_voltage",
        "value": READ_VOLTAGE_V,
        "unit": "V",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "Abstract / DC sweep characterization",
        "source_note": (
            "The reported ON/OFF ratio is evaluated at 0.2 V."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "on_off_ratio",
        "value": ON_OFF_RATIO,
        "unit": "ratio",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "Abstract / DC sweep characterization",
        "source_note": (
            "Journal article reports ON/OFF approximately 76 at 0.2 V "
            "after electroforming."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": SUPPORTING_THESIS_ID,
        "property_name": "switching_mechanism",
        "value": "FILAMENTARY_OXYGEN_REDox",
        "unit": "",
        "value_type": "EXPERIMENTAL",
        "source_title": SUPPORTING_TITLE,
        "doi": SUPPORTING_THESIS_ID,
        "page": "Abstract",
        "figure_or_table": "Mechanism description",
        "source_note": (
            "Conductance evolution is attributed to conductive-filament "
            "formation together with oxygen-driven redox reactions in "
            "the TiOx/TiOy bilayer."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "conductance_mode",
        "value": CONDUCTANCE_MODE,
        "unit": "",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "LTP/LTD pulse experiments",
        "source_note": (
            "The journal article reports LTP/LTD using repeated pulses "
            "and further uses 25 linearly varied pulses so conductance "
            "changes linearly. This supports gradual multilevel mapping."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": SUPPORTING_THESIS_ID,
        "property_name": "demonstrated_protocol_states",
        "value": 50,
        "unit": "conductance values",
        "value_type": "EXPERIMENTAL",
        "source_title": SUPPORTING_TITLE,
        "doi": SUPPORTING_THESIS_ID,
        "page": "Abstract; LTP/LTD section starts p.56",
        "figure_or_table": "50-pulse LTP/LTD experiment",
        "source_note": (
            "The related thesis reports 50 different conductance values "
            "for LTP/LTD under 50 identical pulses. This is retained as "
            "protocol evidence only and is not used as an intrinsic "
            "physical state-count cap."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": SUPPORTING_THESIS_ID,
        "property_name": "conductance_states",
        "value": pd.NA,
        "unit": "",
        "value_type": "NOT_REPORTED",
        "source_title": SUPPORTING_TITLE,
        "doi": SUPPORTING_THESIS_ID,
        "page": "Abstract",
        "figure_or_table": "Multi-level / LTP-LTD discussion",
        "source_note": (
            "No fixed intrinsic maximum number of physical conductance "
            "states is established. The reported 50 states are tied to "
            "a 50-pulse experimental protocol."
        ),
    },

    {
        "device_id": DEVICE_ID,
        "study_id": PRIMARY_STUDY_ID,
        "property_name": "stpd_conductance_change",
        "value": "-77.79 to 96.07",
        "unit": "%",
        "value_type": "EXPERIMENTAL",
        "source_title": PRIMARY_TITLE,
        "doi": PRIMARY_DOI,
        "page": "869-875",
        "figure_or_table": "STDP experiment",
        "source_note": (
            "Reported STDP conductance change range using a "
            "time-division multiplexing approach."
        ),
    },
]

new_trace = pd.DataFrame(trace_rows)

for column in trace.columns:
    if column not in new_trace.columns:
        new_trace[column] = ""

for column in new_trace.columns:
    if column not in trace.columns:
        trace[column] = ""

new_trace = new_trace[trace.columns]

# Idempotent: replace all traceability rows for this new profile.
trace = trace[trace["device_id"] != DEVICE_ID].copy()

trace = pd.concat(
    [trace, new_trace],
    ignore_index=True,
)


# ============================================================
# VALIDATION
# ============================================================

if devices["device_id"].duplicated().any():
    dup = devices.loc[
        devices["device_id"].duplicated(keep=False),
        "device_id",
    ].tolist()
    raise ValueError(f"Duplicate device IDs detected: {dup}")

device_trace = trace[trace["device_id"] == DEVICE_ID]

duplicate_trace = device_trace.duplicated(
    subset=["device_id", "property_name"]
).sum()

if duplicate_trace:
    raise ValueError(
        f"{DEVICE_ID} contains duplicate property traceability rows."
    )

saved = devices[devices["device_id"] == DEVICE_ID].iloc[0]

if float(saved["on_off_ratio"]) != ON_OFF_RATIO:
    raise ValueError("Unexpected ON/OFF ratio after update.")

if float(saved["read_voltage_v"]) != READ_VOLTAGE_V:
    raise ValueError("Unexpected read voltage after update.")

if str(saved["conductance_mode"]).strip() != CONDUCTANCE_MODE:
    raise ValueError("Unexpected conductance mode after update.")

if pd.notna(saved["conductance_states"]):
    raise ValueError(
        "TiOx_04 must not receive a fixed intrinsic state count."
    )


# ============================================================
# WRITE
# ============================================================

devices.to_csv(DEVICE_FILE, index=False)
trace.to_csv(TRACE_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print()
print("TiOx_04 evidence-backed profile added.")
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
            "conductance_states",
            "state_count_status",
        ]
    ].to_string(index=False)
)

print()
print("Primary journal study:", PRIMARY_STUDY_ID)
print("Supporting thesis:", SUPPORTING_THESIS_ID)
print("Reported ON/OFF:", ON_OFF_RATIO)
print("Read voltage:", READ_VOLTAGE_V, "V")
print("Forming voltage:", FORMING_VOLTAGE_V, "V")
print("Conductance mode:", CONDUCTANCE_MODE)
print()

print("IMPORTANT:")
print(
    "  50 experimentally demonstrated protocol states are "
    "recorded as evidence but NOT used as a fixed physical "
    "state-count cap."
)
print(
    "  Absolute RON/ROFF remain unknown; the simulator should "
    "use normalized conductance from ON/OFF ratio."
)
print(
    "  The thesis is supporting evidence for the same experimental "
    "platform and must NOT be counted as an independent primary study."
)
