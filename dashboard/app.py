from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NanoMemristor AI Designer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
DATA_DIR = PROJECT_ROOT / "data"

SUMMARY_FILE = RESULTS_DIR / "zero_shot_summary.csv"
PREDICTIONS_FILE = RESULTS_DIR / "zero_shot_predictions.csv"
DEVICE_FILE = DATA_DIR / "device_profiles.csv"
TRACE_FILE = DATA_DIR / "source_traceability.csv"
AUDIT_FILE = RESULTS_DIR / "traceability_audit.csv"
OPTIMAL_FILE = RESULTS_DIR / "device_optimal_configs.csv"
BASELINE_FILE = PROJECT_ROOT / "results" / "mnist_baseline_accuracy.txt"


# ============================================================
# SETTINGS
# ============================================================

NEAR_OPTIMAL_TOLERANCE_PP = 0.5

DEFAULT_DEVICE_ID = "TaOx_01"


MODEL_FEATURE_LABELS = [
    "ON/OFF ratio",
    "State-count availability",
    "Physical state count",
    "Crossbar size",
    "Requested weight precision",
    "Effective weight levels",
    "Bit slices per branch",
    "Physical cells per weight",
    "ADC precision",
    "Conductance behavior",
    "Weight-mapping strategy",
]


# ============================================================
# PUBLIC MATERIAL INFORMATION
# ============================================================

MATERIAL_INFO = {

    "TiOx_03": {
        "symbol": "TiO₂",
        "name": "Titanium Dioxide Memristor",
        "short_name": "Titanium Dioxide",

        "description": (
            "Titanium dioxide is an oxide material used "
            "as the switching layer in resistive-memory "
            "devices. This literature profile shows "
            "analog conductance behavior."
        ),

        "nano_note": (
            "Its oxide switching layer produces the "
            "electrical behavior used by the accelerator "
            "model."
        ),
    },

    "HfOx_02": {
        "symbol": "HfO₂",
        "name": "Hafnium Oxide Memristor",
        "short_name": "Hafnium Oxide",

        "description": (
            "Hafnium oxide is an important electronic "
            "oxide widely used in nanoscale electronic "
            "devices. This profile exhibits gradual "
            "analog conductance behavior."
        ),

        "nano_note": (
            "Its conductance window constrains how "
            "neural-network weights are represented "
            "inside the simulated accelerator."
        ),
    },

    "TaOx_01": {
        "symbol": "TaOₓ",
        "name": "Tantalum Oxide Multilevel Memristor",
        "short_name": "Tantalum Oxide",

        "description": (
            "Tantalum oxide memristors can support "
            "multiple resistance states. This literature "
            "profile supports seven physical conductance "
            "states."
        ),

        "nano_note": (
            "The finite state capability directly limits "
            "the effective neural-weight precision that "
            "can be represented."
        ),
    },

    "ZnO_01": {
        "symbol": "ZnO",
        "name": "Zinc Oxide Memristor",
        "short_name": "Zinc Oxide",

        "description": (
            "Zinc oxide is a semiconducting oxide used "
            "in resistive-memory research. This source "
            "reports nanocrystalline ZnO and binary "
            "resistance switching."
        ),

        "nano_note": (
            "Because the device is binary, higher "
            "neural-weight precision is obtained through "
            "bit slicing across multiple physical cells."
        ),
    },

}


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1420px;
        padding-top: 1.05rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    /* --------------------------------------------------------
       Important:
       text must wrap instead of being cut with ...
       -------------------------------------------------------- */

    .wrap-text,
    .hero,
    .process-card,
    .material-card,
    .info-card,
    .callout,
    .success-card,
    .warning-card,
    .status-card,
    .progress-box {

        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
    }

    /* --------------------------------------------------------
       HERO
       -------------------------------------------------------- */

    .hero {
        border-radius: 22px;
        padding: 1.55rem 1.9rem;
        margin-bottom: 1.1rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        background:
            linear-gradient(
                135deg,
                rgba(82, 69, 210, 0.16),
                rgba(20, 145, 165, 0.08)
            );
    }

    .hero-kicker {
        font-size: 0.76rem;
        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.12em;

        opacity: 0.65;

        margin-bottom: 0.45rem;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 850;
        line-height: 1.08;
    }

    .hero-subtitle {
        margin-top: 0.7rem;

        max-width: 950px;

        line-height: 1.55;

        font-size: 1.03rem;

        opacity: 0.80;
    }

    /* --------------------------------------------------------
       PROCESS CARDS
       No fixed height:
       content is never cut off.
       -------------------------------------------------------- */

    .process-card {
        min-height: 175px;

        border-radius: 16px;

        padding: 1rem 1.05rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.22);

        background:
            rgba(120, 120, 120, 0.04);
    }

    .process-number {
        font-size: 0.72rem;
        font-weight: 800;

        letter-spacing: 0.11em;

        opacity: 0.52;
    }

    .process-title {
        font-weight: 800;

        font-size: 1rem;

        margin-top: 0.45rem;

        line-height: 1.3;
    }

    .process-text {
        margin-top: 0.45rem;

        opacity: 0.72;

        font-size: 0.86rem;

        line-height: 1.45;
    }

    /* --------------------------------------------------------
       MATERIAL CARDS
       -------------------------------------------------------- */

    .material-card {
        min-height: 225px;

        border-radius: 17px;

        padding: 1.15rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        background:
            linear-gradient(
                160deg,
                rgba(120, 120, 120, 0.05),
                rgba(60, 130, 170, 0.04)
            );
    }

    .material-symbol {
        font-size: 1.75rem;

        font-weight: 850;

        letter-spacing: -0.03em;
    }

    .material-name {
        font-weight: 750;

        margin-top: 0.15rem;

        line-height: 1.35;
    }

    .material-description {
        opacity: 0.70;

        font-size: 0.85rem;

        line-height: 1.48;

        margin-top: 0.65rem;
    }

    /* --------------------------------------------------------
       INFORMATION CARDS
       Used instead of st.metric() for long text.
       -------------------------------------------------------- */

    .info-card {
        min-height: 145px;

        border-radius: 15px;

        padding: 1rem 1.1rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        background:
            rgba(120, 120, 120, 0.035);
    }

    .info-label {
        font-size: 0.72rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.08em;

        opacity: 0.55;

        line-height: 1.35;
    }

    .info-value {
        font-size: 1.22rem;

        font-weight: 800;

        margin-top: 0.32rem;

        line-height: 1.30;

        white-space: normal !important;

        overflow-wrap: anywhere !important;
    }

    .info-note {
        font-size: 0.80rem;

        opacity: 0.68;

        margin-top: 0.34rem;

        line-height: 1.40;
    }

    /* --------------------------------------------------------
       CALLOUTS
       -------------------------------------------------------- */

    .zero-shot-box {
        border-radius: 18px;

        padding: 1.15rem 1.3rem;

        border:
            1px solid
            rgba(75, 95, 220, 0.34);

        background:
            linear-gradient(
                135deg,
                rgba(80, 80, 200, 0.12),
                rgba(20, 145, 165, 0.06)
            );

        margin-top: 0.75rem;
        margin-bottom: 1.05rem;

        line-height: 1.5;
    }

    .zero-shot-kicker {
        font-size: 0.74rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        opacity: 0.62;
        margin-bottom: 0.45rem;
    }

    .zero-shot-title {
        font-size: 1.12rem;
        font-weight: 850;
        line-height: 1.35;
    }

    .zero-shot-row {
        margin-top: 0.7rem;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .zero-shot-label {
        font-weight: 800;
    }

    .zero-shot-note {
        margin-top: 0.85rem;
        padding-top: 0.8rem;
        border-top:
            1px solid
            rgba(110, 110, 110, 0.18);
        font-size: 0.88rem;
        opacity: 0.78;
        line-height: 1.5;
    }

    .callout {
        border-radius: 14px;

        padding: 1rem 1.15rem;

        border-left:
            4px solid
            rgba(70, 90, 210, 0.75);

        background:
            rgba(80, 80, 190, 0.07);

        margin-top: 0.8rem;

        margin-bottom: 1rem;

        line-height: 1.5;
    }

    .success-card {
        border-radius: 16px;

        padding: 1.1rem;

        border:
            1px solid
            rgba(40, 150, 90, 0.34);

        background:
            rgba(40, 150, 90, 0.07);

        margin-top: 0.7rem;

        margin-bottom: 1rem;

        line-height: 1.45;
    }

    .warning-card {
        border-radius: 16px;

        padding: 1.1rem;

        border:
            1px solid
            rgba(210, 140, 30, 0.34);

        background:
            rgba(210, 140, 30, 0.07);

        margin-top: 0.7rem;

        margin-bottom: 1rem;

        line-height: 1.45;
    }

    /* --------------------------------------------------------
       STATUS CARDS
       -------------------------------------------------------- */

    .status-card {
        min-height: 120px;

        border-radius: 14px;

        padding: 1rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        background:
            rgba(120, 120, 120, 0.035);
    }

    .status-label {
        font-size: 0.72rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.07em;

        opacity: 0.55;
    }

    .status-value {
        font-size: 1.08rem;

        font-weight: 800;

        margin-top: 0.35rem;

        line-height: 1.35;

        white-space: normal !important;

        overflow-wrap: anywhere !important;
    }

    /* --------------------------------------------------------
       DESIGN SPACE PROGRESSION
       -------------------------------------------------------- */

    .progress-row {
        display: flex;

        align-items: center;

        gap: 0.7rem;

        margin-top: 0.8rem;

        flex-wrap: wrap;
    }

    .progress-box {
        border-radius: 14px;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        padding: 0.85rem 1.15rem;

        min-width: 155px;

        text-align: center;

        background:
            rgba(120, 120, 120, 0.035);
    }

    .progress-value {
        font-size: 1.55rem;

        font-weight: 850;
    }

    .progress-label {
        font-size: 0.78rem;

        opacity: 0.62;

        margin-top: 0.2rem;

        line-height: 1.35;
    }

    .progress-arrow {
        font-size: 1.7rem;

        opacity: 0.38;
    }


    /* --------------------------------------------------------
       WHY THIS DESIGN
       -------------------------------------------------------- */

    .why-card {
        min-height: 170px;

        border-radius: 15px;

        padding: 1rem 1.05rem;

        border:
            1px solid
            rgba(110, 110, 110, 0.20);

        background:
            rgba(120, 120, 120, 0.035);

        white-space: normal !important;
        overflow: visible !important;
        overflow-wrap: anywhere !important;
    }

    .why-kicker {
        font-size: 0.70rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.08em;

        opacity: 0.55;

        margin-bottom: 0.35rem;
    }

    .why-title {
        font-size: 1.03rem;

        font-weight: 800;

        line-height: 1.30;
    }

    .why-text {
        margin-top: 0.45rem;

        font-size: 0.84rem;

        line-height: 1.46;

        opacity: 0.72;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HTML RENDERING
# ============================================================

def render_html(
    markup,
):

    cleaned = textwrap.dedent(
        markup
    ).strip()

    cleaned = " ".join(

        line.strip()

        for line
        in cleaned.splitlines()

        if line.strip()

    )

    st.markdown(
        cleaned,
        unsafe_allow_html=True,
    )


# ============================================================
# HELPERS
# ============================================================

def require_files(
    paths,
):

    missing = [

        str(path)

        for path
        in paths

        if not path.exists()

    ]

    if missing:

        st.error(
            "Some generated research files are missing."
        )

        st.code(
            "\n".join(
                missing
            )
        )

        st.info(
            "Run the analysis pipeline and reload the website."
        )

        st.stop()


def bool_series(
    series,
):

    if series.dtype == bool:

        return series

    converted = (

        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,

                "false": False,
                "0": False,
                "no": False,
            }
        )

    )

    if converted.isna().any():

        bad_values = (

            series[
                converted.isna()
            ]
            .astype(str)
            .unique()
            .tolist()

        )

        raise ValueError(

            "Unexpected boolean values: "
            f"{bad_values}"

        )

    return converted.astype(
        bool
    )


def bool_value(
    value,
):

    return bool(

        bool_series(

            pd.Series(
                [value]
            )

        ).iloc[0]

    )


def clean_text(
    value,
    fallback="Not available",
):

    if pd.isna(
        value
    ):

        return fallback

    text = str(
        value
    ).strip()

    if not text:

        return fallback

    return text


def format_ratio(
    value,
):

    if pd.isna(
        value
    ):

        return "Not available"

    value = float(
        value
    )

    if value >= 1000:

        return f"{value:,.0f}"

    if value >= 100:

        return f"{value:,.1f}"

    return f"{value:,.2f}"


def public_info(
    device_id,
):

    if device_id in MATERIAL_INFO:

        return MATERIAL_INFO[
            device_id
        ]

    return {
        "symbol": "Memristor",
        "name": "Memristor Device",
        "short_name": "Memristor",

        "description": (
            "Literature-derived memristor device "
            "used in the accelerator experiment."
        ),

        "nano_note": (
            "Its electrical behavior constrains "
            "accelerator design."
        ),
    }


def public_device_label(
    device_id,
):

    info = public_info(
        device_id
    )

    return (
        f"{info['symbol']} — "
        f"{info['short_name']}"
    )


def public_mapping_name(
    conductance_mode,
):

    mode = clean_text(
        conductance_mode,
        fallback=""
    ).upper()

    if mode == "DISCRETE_BINARY":

        return (
            "Bit-sliced binary differential mapping"
        )

    if mode == "DISCRETE_MULTILEVEL":

        return (
            "Multilevel differential-pair mapping"
        )

    if mode == "ANALOG":

        return (
            "Idealized analog differential-pair mapping"
        )

    if mode == "GRADUAL":

        return (
            "Idealized gradual-conductance mapping"
        )

    return (
        "Device-aware conductance mapping"
    )


def public_mode_name(
    mode,
):

    mode = clean_text(
        mode,
        fallback=""
    ).upper()

    names = {

        "DISCRETE_BINARY":
            "Binary switching",

        "DISCRETE_MULTILEVEL":
            "Multilevel switching",

        "ANALOG":
            "Analog conductance",

        "GRADUAL":
            "Gradual conductance",

    }

    return names.get(
        mode,
        mode.replace(
            "_",
            " "
        ).title(),
    )


def config_label(
    crossbar,
    weight_bits,
    adc_bits,
):

    return (
        f"{int(crossbar)} × "
        f"{int(crossbar)} | "
        f"{int(weight_bits)}-bit weights | "
        f"{int(adc_bits)}-bit ADC"
    )


def find_candidate(
    candidate_df,
    crossbar,
    weight_bits,
    adc_bits,
):

    matches = candidate_df[

        (
            candidate_df[
                "crossbar_size"
            ]
            ==
            int(
                crossbar
            )
        )

        &

        (
            candidate_df[
                "requested_weight_bits"
            ]
            ==
            int(
                weight_bits
            )
        )

        &

        (
            candidate_df[
                "adc_bits"
            ]
            ==
            int(
                adc_bits
            )
        )

    ]

    if matches.empty:

        return None

    return matches.iloc[0]


def provenance_complete_percentage(
    trace_df,
):

    provenance_columns = [
        "source_title",
        "doi",
        "page",
        "figure_or_table",
        "source_note",
    ]

    for column in provenance_columns:

        if column not in trace_df.columns:

            return float(
                "nan"
            )

    valid = pd.DataFrame(
        index=trace_df.index
    )

    for column in provenance_columns:

        valid[
            column
        ] = (

            trace_df[
                column
            ]
            .notna()

            &

            trace_df[
                column
            ]
            .astype(str)
            .str.strip()
            .ne("")

        )

    return float(

        100.0
        *
        valid.all(
            axis=1
        ).mean()

    )


def human_property_name(
    name,
):

    name = clean_text(
        name,
        fallback=""
    )

    special = {

        "conductance_mode":
            "Conductance Behavior",

        "conductance_states":
            "Conductance States",

        "on_off_ratio":
            "ON/OFF Ratio",

        "ron_ohm":
            "Low-Resistance State",

        "roff_ohm":
            "High-Resistance State",

        "state_count_status_consistency":
            "State-Count Evidence Check",

        "profile_ratio_consistency":
            "ON/OFF Ratio Consistency",

    }

    if name in special:

        return special[
            name
        ]

    return (

        name
        .replace(
            "_",
            " "
        )
        .title()

    )


def human_evidence_type(
    value,
):

    value = clean_text(
        value,
        fallback=""
    )

    return (

        value
        .replace(
            "_",
            " "
        )
        .title()

    )


def evidence_category(
    value_type,
    simulator_value=None,
    trace_value=None,
):

    raw = clean_text(
        value_type,
        fallback=""
    ).strip().upper()

    normalized = (
        raw
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized in {
        "REPORTED",
        "MEASURED",
        "EXPERIMENTAL",
        "DIRECTLY_REPORTED",
    }:
        return "Reported"

    if normalized in {
        "DERIVED",
        "CALCULATED",
        "COMPUTED",
        "INFERRED",
    }:
        return "Derived"

    if normalized in {
        "ASSUMED",
        "MODEL_ASSUMPTION",
        "SIMULATOR_ASSUMPTION",
    }:
        return "Assumed"

    if normalized in {
        "MISSING",
        "NOT_REPORTED",
        "UNREPORTED",
        "NOT_AVAILABLE",
    }:
        return "Missing"

    simulator_missing = (
        simulator_value is None
        or pd.isna(simulator_value)
        or str(simulator_value).strip() == ""
    )

    trace_missing = (
        trace_value is None
        or pd.isna(trace_value)
        or str(trace_value).strip() == ""
    )

    if simulator_missing and trace_missing:
        return "Missing"

    return (
        human_evidence_type(
            value_type
        )
        if raw
        else "Unclassified"
    )


def source_location_text(
    page=None,
    figure_or_table=None,
):

    parts = []

    if (
        page is not None
        and not pd.isna(page)
        and str(page).strip()
    ):
        parts.append(
            f"p. {str(page).strip()}"
        )

    if (
        figure_or_table is not None
        and not pd.isna(figure_or_table)
        and str(figure_or_table).strip()
    ):
        parts.append(
            str(figure_or_table).strip()
        )

    return (
        " • ".join(parts)
        if parts
        else "Not specified"
    )


def is_integrity_check_record(
    row,
):

    property_name = clean_text(
        row.get(
            "property_name"
        ),
        fallback="",
    ).strip().lower()

    value_type = clean_text(
        row.get(
            "value_type"
        ),
        fallback="",
    ).strip().upper()

    audit_only_properties = {
        "profile_ratio_consistency",
        "state_count_status_consistency",
    }

    audit_only_types = {
        "DERIVED_CHECK",
        "CONSISTENCY_CHECK",
    }

    return (
        property_name in audit_only_properties
        or value_type in audit_only_types
    )


def format_evidence_value(
    value,
    property_name,
    unit="",
):

    if (
        value is None
        or pd.isna(value)
        or str(value).strip() == ""
    ):
        return "—"

    property_name = clean_text(
        property_name,
        fallback="",
    ).strip().lower()

    unit = clean_text(
        unit,
        fallback="",
    ).strip()

    raw_text = str(
        value
    ).strip()

    # Keep dimensionless ON/OFF ratio free of derivation notation
    # such as "R4/LRS", which is not a physical unit.
    if property_name == "on_off_ratio":

        try:
            numeric = float(
                value
            )

            return f"{numeric:.2f}"

        except (
            TypeError,
            ValueError,
        ):
            return raw_text

    # State counts are easier to read as integers when integral.
    if property_name == "conductance_states":

        try:
            numeric = float(
                value
            )

            if numeric.is_integer():
                return str(
                    int(
                        numeric
                    )
                )

        except (
            TypeError,
            ValueError,
        ):
            pass

    # Improve readability of resistance units.
    if unit.lower() == "ohm":
        unit = "Ω"

    if unit:
        return (
            f"{raw_text} {unit}"
        )

    return raw_text


def render_process_card(
    number,
    title,
    text,
):

    render_html(
        f"""
        <div class="process-card">
            <div class="process-number">
                STEP {number}
            </div>

            <div class="process-title">
                {title}
            </div>

            <div class="process-text">
                {text}
            </div>
        </div>
        """
    )


def render_info_card(
    label,
    value,
    note,
):

    render_html(
        f"""
        <div class="info-card">
            <div class="info-label">
                {label}
            </div>

            <div class="info-value">
                {value}
            </div>

            <div class="info-note">
                {note}
            </div>
        </div>
        """
    )


def render_status_card(
    label,
    value,
):

    render_html(
        f"""
        <div class="status-card">
            <div class="status-label">
                {label}
            </div>

            <div class="status-value">
                {value}
            </div>
        </div>
        """
    )


def render_accuracy_validation_chart(
    predicted_accuracy,
    actual_accuracy,
    exhaustive_best_accuracy,
    baseline_accuracy,
):

    chart_df = pd.DataFrame(
        {
            "Result": [
                "Predicted",
                "Actual recommended",
                "Exhaustive best",
                "Software baseline",
            ],
            "Accuracy": [
                float(predicted_accuracy),
                float(actual_accuracy),
                float(exhaustive_best_accuracy),
                float(baseline_accuracy),
            ],
        }
    )

    chart_df["Label"] = chart_df["Accuracy"].map(
        lambda value: f"{value:.2f}%"
    )

    low = float(chart_df["Accuracy"].min())
    high = float(chart_df["Accuracy"].max())
    spread = max(high - low, 0.10)
    padding = max(0.12, spread * 0.35)

    spec = {
        "height": 320,
        "title": "Prediction → Exhaustive Validation (zoomed accuracy scale)",
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "opacity": 0.35,
                },
                "encoding": {
                    "x": {
                        "field": "Result",
                        "type": "nominal",
                        "sort": [
                            "Predicted",
                            "Actual recommended",
                            "Exhaustive best",
                            "Software baseline",
                        ],
                        "axis": {
                            "title": None,
                            "labelAngle": 0,
                        },
                    },
                    "y": {
                        "field": "Accuracy",
                        "type": "quantitative",
                        "scale": {
                            "domain": [low - padding, high + padding],
                            "zero": False,
                        },
                        "axis": {
                            "title": "Accuracy (%)",
                            "format": ".2f",
                        },
                    },
                },
            },
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 120,
                },
                "encoding": {
                    "x": {
                        "field": "Result",
                        "type": "nominal",
                        "sort": [
                            "Predicted",
                            "Actual recommended",
                            "Exhaustive best",
                            "Software baseline",
                        ],
                    },
                    "y": {
                        "field": "Accuracy",
                        "type": "quantitative",
                        "scale": {
                            "domain": [low - padding, high + padding],
                            "zero": False,
                        },
                    },
                    "tooltip": [
                        {"field": "Result", "type": "nominal"},
                        {
                            "field": "Accuracy",
                            "type": "quantitative",
                            "format": ".3f",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "text",
                    "dy": -16,
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {
                        "field": "Result",
                        "type": "nominal",
                        "sort": [
                            "Predicted",
                            "Actual recommended",
                            "Exhaustive best",
                            "Software baseline",
                        ],
                    },
                    "y": {
                        "field": "Accuracy",
                        "type": "quantitative",
                        "scale": {
                            "domain": [low - padding, high + padding],
                            "zero": False,
                        },
                    },
                    "text": {
                        "field": "Label",
                        "type": "nominal",
                    },
                },
            },
        ],
    }

    st.vega_lite_chart(
        chart_df,
        spec,
        use_container_width=True,
    )


def render_regret_threshold_chart(
    regret_pp,
    threshold_pp,
):

    regret_pp = float(regret_pp)
    threshold_pp = float(threshold_pp)
    passed = regret_pp <= threshold_pp
    result_text = "PASS" if passed else "FAIL"

    axis_max = max(
        threshold_pp * 1.25,
        regret_pp * 1.20,
        threshold_pp + 0.10,
    )

    bar_df = pd.DataFrame(
        {
            "Metric": ["Regret"],
            "Start": [0.0],
            "Value": [regret_pp],
            "Label": [f"{regret_pp:.2f} pp — {result_text}"],
        }
    )

    spec = {
        "height": 170,
        "title": "Regret against the near-optimal success threshold",
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "size": 30,
                },
                "encoding": {
                    "x": {
                        "field": "Value",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, axis_max],
                            "zero": True,
                        },
                        "axis": {
                            "title": "Accuracy regret (percentage points)",
                            "format": ".2f",
                        },
                    },
                    "x2": {
                        "field": "Start",
                    },
                    "y": {
                        "field": "Metric",
                        "type": "nominal",
                        "axis": {"title": None},
                    },
                    "tooltip": [
                        {
                            "field": "Value",
                            "type": "quantitative",
                            "format": ".3f",
                            "title": "Regret (pp)",
                        }
                    ],
                },
            },
            {
                "data": {
                    "values": [
                        {"Threshold": threshold_pp}
                    ]
                },
                "mark": {
                    "type": "rule",
                    "strokeDash": [6, 5],
                    "size": 2,
                },
                "encoding": {
                    "x": {
                        "field": "Threshold",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, axis_max],
                            "zero": True,
                        },
                    }
                },
            },
            {
                "data": {
                    "values": [
                        {
                            "Threshold": threshold_pp,
                            "ThresholdLabel": f"Threshold {threshold_pp:.2f} pp",
                        }
                    ]
                },
                "mark": {
                    "type": "text",
                    "angle": 270,
                    "dx": 44,
                    "dy": -6,
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {
                        "field": "Threshold",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, axis_max],
                            "zero": True,
                        },
                    },
                    "text": {
                        "field": "ThresholdLabel",
                        "type": "nominal",
                    },
                },
            },
            {
                "mark": {
                    "type": "text",
                    "align": "left",
                    "dx": 8,
                    "dy": -24,
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {
                        "field": "Value",
                        "type": "quantitative",
                        "scale": {
                            "domain": [0, axis_max],
                            "zero": True,
                        },
                    },
                    "y": {
                        "field": "Metric",
                        "type": "nominal",
                    },
                    "text": {
                        "field": "Label",
                        "type": "nominal",
                    },
                },
            },
        ],
    }

    st.vega_lite_chart(
        bar_df,
        spec,
        use_container_width=True,
    )


def render_predicted_vs_actual_chart(
    candidate_df,
    recommended_row,
    raw_best_row,
):

    chart_df = candidate_df[
        [
            "predicted_accuracy",
            "accuracy",
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
        ]
    ].copy()

    chart_df = chart_df.rename(
        columns={
            "predicted_accuracy": "Predicted Accuracy",
            "accuracy": "Actual Accuracy",
            "crossbar_size": "Crossbar",
            "requested_weight_bits": "Weight Bits",
            "adc_bits": "ADC Bits",
        }
    )

    min_axis = float(
        min(
            chart_df["Predicted Accuracy"].min(),
            chart_df["Actual Accuracy"].min(),
        )
    )

    max_axis = float(
        max(
            chart_df["Predicted Accuracy"].max(),
            chart_df["Actual Accuracy"].max(),
        )
    )

    padding = max(0.05, (max_axis - min_axis) * 0.04)
    low = min_axis - padding
    high = max_axis + padding

    recommended_point = {
        "Predicted Accuracy": float(
            recommended_row["predicted_accuracy"]
        ),
        "Actual Accuracy": float(
            recommended_row["accuracy"]
        ),
        "Label": "AI recommendation",
    }

    best_point = {
        "Predicted Accuracy": float(
            raw_best_row["predicted_accuracy"]
        ),
        "Actual Accuracy": float(
            raw_best_row["accuracy"]
        ),
        "Label": "Exhaustive best",
    }

    ideal_line = [
        {
            "Predicted Accuracy": low,
            "Actual Accuracy": low,
        },
        {
            "Predicted Accuracy": high,
            "Actual Accuracy": high,
        },
    ]

    x_encoding = {
        "field": "Predicted Accuracy",
        "type": "quantitative",
        "scale": {
            "domain": [low, high],
            "zero": False,
        },
        "axis": {
            "title": "Predicted accuracy (%)",
            "format": ".2f",
        },
    }

    y_encoding = {
        "field": "Actual Accuracy",
        "type": "quantitative",
        "scale": {
            "domain": [low, high],
            "zero": False,
        },
        "axis": {
            "title": "Actual simulated accuracy (%)",
            "format": ".2f",
        },
    }

    spec = {
        "height": 430,
        "title": (
            "Predicted vs actual accuracy across "
            f"{len(chart_df)} held-out-device configurations"
        ),
        "layer": [
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 48,
                    "opacity": 0.48,
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                    "tooltip": [
                        {
                            "field": "Predicted Accuracy",
                            "type": "quantitative",
                            "format": ".3f",
                        },
                        {
                            "field": "Actual Accuracy",
                            "type": "quantitative",
                            "format": ".3f",
                        },
                        {"field": "Crossbar", "type": "quantitative"},
                        {"field": "Weight Bits", "type": "quantitative"},
                        {"field": "ADC Bits", "type": "quantitative"},
                    ],
                },
            },
            {
                "data": {"values": ideal_line},
                "mark": {
                    "type": "line",
                    "strokeDash": [7, 5],
                    "opacity": 0.75,
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                },
            },
            {
                "data": {"values": [recommended_point]},
                "mark": {
                    "type": "point",
                    "filled": True,
                    "shape": "diamond",
                    "size": 230,
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                },
            },
            {
                "data": {"values": [recommended_point]},
                "mark": {
                    "type": "text",
                    "dx": 10,
                    "dy": -12,
                    "align": "left",
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                    "text": {"field": "Label", "type": "nominal"},
                },
            },
            {
                "data": {"values": [best_point]},
                "mark": {
                    "type": "point",
                    "filled": True,
                    "shape": "triangle-up",
                    "size": 220,
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                },
            },
            {
                "data": {"values": [best_point]},
                "mark": {
                    "type": "text",
                    "dx": 10,
                    "dy": 14,
                    "align": "left",
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": x_encoding,
                    "y": y_encoding,
                    "text": {"field": "Label", "type": "nominal"},
                },
            },
        ],
    }

    st.vega_lite_chart(
        chart_df,
        spec,
        use_container_width=True,
    )


# ============================================================
# VERIFY FILES
# ============================================================

require_files(
    [
        SUMMARY_FILE,
        PREDICTIONS_FILE,
        DEVICE_FILE,
        TRACE_FILE,
        AUDIT_FILE,
        OPTIMAL_FILE,
        BASELINE_FILE,
    ]
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    devices = pd.read_csv(
        DEVICE_FILE
    )

    trace = pd.read_csv(
        TRACE_FILE
    )

    audit = pd.read_csv(
        AUDIT_FILE
    )

    optimal = pd.read_csv(
        OPTIMAL_FILE
    )

    return (
        summary,
        predictions,
        devices,
        trace,
        audit,
        optimal,
    )


(
    summary,
    predictions,
    devices,
    trace,
    audit,
    optimal,
) = load_data()


software_baseline_accuracy = float(
    BASELINE_FILE.read_text(encoding="utf-8").strip()
)


# ============================================================
# PROJECT METRICS
# ============================================================

active_device_ids = sorted(

    summary[
        "held_out_device"
    ]
    .astype(str)
    .unique()
    .tolist()

)


independent_device_count = len(
    active_device_ids
)


configurations_per_device = int(

    summary[
        "held_out_rows"
    ]
    .iloc[0]

)


total_simulation_rows = (

    independent_device_count
    *
    configurations_per_device

)


training_devices_per_fold = int(

    summary[
        "training_devices"
    ]
    .iloc[0]

)


baseline_success = bool_series(

    summary[
        "baseline_near_optimal_success"
    ]

)


baseline_success_count = int(

    baseline_success.sum()

)


baseline_success_rate = (

    100.0
    *
    baseline_success_count
    /
    independent_device_count

)


baseline_mean_regret = float(

    summary[
        "baseline_regret_pp"
    ]
    .mean()

)


baseline_top3 = bool_series(

    summary[
        "baseline_cost_aware_top3"
    ]

)


baseline_top3_count = int(

    baseline_top3.sum()

)


baseline_top3_rate = (

    100.0
    *
    baseline_top3_count
    /
    independent_device_count

)


near_mask = bool_series(

    predictions[
        "is_true_near_optimal"
    ]

)


near_predictions = predictions[

    near_mask

].copy()


near_region_mae = float(

    near_predictions[
        "absolute_prediction_error_pp"
    ]
    .mean()

)


near_region_rmse = float(

    np.sqrt(

        np.mean(

            np.square(

                near_predictions[
                    "prediction_error_pp"
                ]
                .astype(float)

            )

        )

    )

)


search_reduction_pct = float(

    summary[
        "search_reduction_pct"
    ]
    .mean()

)


literature_device_count = int(

    devices[
        "device_id"
    ]
    .nunique()

)


traceability_row_count = int(
    len(
        trace
    )
)


provenance_completeness = (
    provenance_complete_percentage(
        trace
    )
)


audit_pass_count = int(

    (
        audit[
            "status"
        ]
        ==
        "PASS"
    ).sum()

)


audit_warn_count = int(

    (
        audit[
            "status"
        ]
        ==
        "WARN"
    ).sum()

)


audit_fail_count = int(

    (
        audit[
            "status"
        ]
        ==
        "FAIL"
    ).sum()

)


# ============================================================
# DESIGN SPACE
# ============================================================

crossbar_values = sorted(

    predictions[
        "crossbar_size"
    ]
    .astype(int)
    .unique()
    .tolist()

)


weight_values = sorted(

    predictions[
        "requested_weight_bits"
    ]
    .astype(int)
    .unique()
    .tolist()

)


adc_values = sorted(

    predictions[
        "adc_bits"
    ]
    .astype(int)
    .unique()
    .tolist()

)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## NanoMemristor AI Designer"
)


st.sidebar.caption(
    "Choose a literature-derived memristor material "
    "and explore the accelerator design recommended "
    "by the zero-shot AI model."
)


public_device_options = {

    public_device_label(
        device_id
    ):
        device_id

    for device_id
    in active_device_ids

}


public_labels = list(
    public_device_options.keys()
)


default_index = 0


for index, label in enumerate(
    public_labels
):

    if (
        public_device_options[
            label
        ]
        ==
        DEFAULT_DEVICE_ID
    ):

        default_index = index

        break


selected_public_label = (
    st.sidebar.selectbox(
        "Choose a memristor material",
        public_labels,
        index=default_index,
    )
)


selected_device = (

    public_device_options[
        selected_public_label
    ]

)


selected_material = public_info(
    selected_device
)


selected_summary = (

    summary[

        summary[
            "held_out_device"
        ]
        ==
        selected_device

    ]
    .iloc[0]

)


selected_family = clean_text(

    selected_summary[
        "held_out_family"
    ]

)


candidate_df = predictions[

    predictions[
        "held_out_device"
    ]
    ==
    selected_device

].copy()


profile_matches = devices[

    devices[
        "device_id"
    ]
    ==
    selected_device

]


if profile_matches.empty:

    profile = None

else:

    profile = (
        profile_matches.iloc[0]
    )


st.sidebar.markdown(
    "---"
)


st.sidebar.markdown(
    f"### {selected_material['symbol']}"
)


st.sidebar.markdown(
    f"**{selected_material['name']}**"
)


st.sidebar.caption(
    selected_material[
        "description"
    ]
)


st.sidebar.markdown(
    "---"
)


st.sidebar.write(
    f"**Candidate designs:** "
    f"{len(candidate_df)}"
)


st.sidebar.write(
    f"**Training devices in this test:** "
    f"{training_devices_per_fold}"
)


st.sidebar.write(
    f"**Technology family:** "
    f"{selected_family}"
)


st.sidebar.caption(
    "Results shown here come directly from "
    "the recorded zero-shot evaluation."
)


# ============================================================
# SELECTED DEVICE DESCRIPTORS
# ============================================================

device_ratio = float(

    candidate_df[
        "device_on_off_ratio"
    ]
    .iloc[0]

)


conductance_mode = clean_text(

    candidate_df[
        "device_conductance_mode"
    ]
    .iloc[0]

)


public_conductance_mode = (
    public_mode_name(
        conductance_mode
    )
)


state_count_available = int(

    candidate_df[
        "state_count_available"
    ]
    .iloc[0]

)


physical_state_count = int(

    candidate_df[
        "physical_state_count"
    ]
    .iloc[0]

)


mapping_public_name = (
    public_mapping_name(
        conductance_mode
    )
)


# ============================================================
# AI RECOMMENDATION
# ============================================================

recommended_crossbar = int(

    selected_summary[
        "baseline_crossbar"
    ]

)


recommended_weight_bits = int(

    selected_summary[
        "baseline_weight_bits"
    ]

)


recommended_adc_bits = int(

    selected_summary[
        "baseline_adc_bits"
    ]

)


recommended_predicted_accuracy = float(

    selected_summary[
        "baseline_predicted_accuracy"
    ]

)


recommended_uncertainty = float(

    selected_summary[
        "baseline_uncertainty_pp"
    ]

)


recommended_lower_bound = float(

    selected_summary[
        "baseline_lower_bound"
    ]

)


recommended_actual_accuracy = float(

    selected_summary[
        "baseline_actual_accuracy"
    ]

)


recommended_regret = float(

    selected_summary[
        "baseline_regret_pp"
    ]

)


recommended_success = bool_value(

    selected_summary[
        "baseline_near_optimal_success"
    ]

)


recommended_exact = bool_value(

    selected_summary[
        "baseline_exact_match"
    ]

)


recommended_top3 = bool_value(

    selected_summary[
        "baseline_cost_aware_top3"
    ]

)


recommended_rank = (

    selected_summary[
        "baseline_cost_aware_rank"
    ]

)


recommended_row = find_candidate(

    candidate_df,

    recommended_crossbar,

    recommended_weight_bits,

    recommended_adc_bits,

)


if recommended_row is None:

    st.error(
        "The saved recommendation could not be found "
        "inside the candidate design table."
    )

    st.stop()


recommended_effective_levels = int(

    recommended_row[
        "effective_weight_levels"
    ]

)


recommended_cells_per_weight = int(

    recommended_row[
        "physical_cells_per_weight"
    ]

)


recommended_cost_proxy = float(

    recommended_row[
        "relative_hardware_cost_proxy"
    ]

)


# ============================================================
# EXHAUSTIVE RESULTS
# ============================================================

raw_best_row = (

    candidate_df
    .sort_values(

        by=[
            "accuracy",
            "relative_hardware_cost_proxy",
            "requested_weight_bits",
            "adc_bits",
        ],

        ascending=[
            False,
            True,
            True,
            True,
        ],

    )
    .iloc[0]

)


raw_best_accuracy = float(

    raw_best_row[
        "accuracy"
    ]

)


cost_aware_crossbar = int(

    selected_summary[
        "true_cost_aware_crossbar"
    ]

)


cost_aware_weight_bits = int(

    selected_summary[
        "true_cost_aware_weight_bits"
    ]

)


cost_aware_adc_bits = int(

    selected_summary[
        "true_cost_aware_adc_bits"
    ]

)


cost_aware_accuracy = float(

    selected_summary[
        "true_cost_aware_accuracy"
    ]

)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-kicker">
            Nanotechnology + AI accelerator design
        </div>

        <div class="hero-title">
            NanoMemristor AI Designer
        </div>

        <div class="hero-subtitle">
            Explore how literature-derived memristor device
            properties influence neural-network accelerator
            design, then see how zero-shot AI recommends a
            near-optimal hardware configuration for a device
            that was completely excluded from model training.
        </div>

    </div>
    """
)


# ============================================================
# PROCESS
# ============================================================

process_columns = st.columns(
    5
)


with process_columns[0]:

    render_process_card(
        "01",
        "Choose Material",
        (
            "Select a literature-derived "
            "oxide memristor device."
        ),
    )


with process_columns[1]:

    render_process_card(
        "02",
        "Read Device Behavior",
        (
            "ON/OFF ratio and physical state "
            "capability constrain neural-weight "
            "representation."
        ),
    )


with process_columns[2]:

    render_process_card(
        "03",
        "Explore 245 Designs",
        (
            "Crossbar size, weight precision and "
            "ADC precision define the current "
            "accelerator search space."
        ),
    )


with process_columns[3]:

    render_process_card(
        "04",
        "Zero-Shot AI",
        (
            "The chosen device is hidden while "
            "the AI learns from the other "
            "physical devices."
        ),
    )


with process_columns[4]:

    render_process_card(
        "05",
        "Verify Result",
        (
            "The recommendation is checked "
            "against all 245 simulated designs "
            "for the hidden device."
        ),
    )


st.write("")


# ============================================================
# NAVIGATION
# ============================================================

(
    home_tab,
    forward_tab,
    nano_tab,
    evidence_tab,
    sources_tab,
) = st.tabs(
    [
        "Home",
        "Forward Design",
        "Why Nano?",
        "Research Evidence",
        "Sources & Limitations",
    ]
)


# ============================================================
# HOME
# ============================================================

with home_tab:

    st.subheader(
        "What is this website for?"
    )


    left, right = st.columns(
        [1.2, 0.8]
    )


    with left:

        st.write(
            """
            A memristor-based AI accelerator can be designed
            in many different ways. The best accelerator
            architecture depends partly on the electrical
            behavior of the memristor device.

            This website connects **published memristor
            device data** with an accelerator simulator and a
            **zero-shot AI recommender**.

            The objective is to identify a promising
            accelerator configuration for a memristor device
            that the AI did not see during training.
            """
        )


    with right:

        render_html(
            """
            <div class="callout">

                <strong>Scientific scope</strong>

                <br><br>

                Literature-grounded memristor
                device-to-accelerator co-design simulator.

                <br><br>

                This is a pilot research prototype rather
                than a complete material-physics-to-accelerator
                simulator.

            </div>
            """
        )


    st.divider()


    st.subheader(
        "Explore the Memristor Materials"
    )


    material_columns = st.columns(
        len(
            active_device_ids
        )
    )


    for index, device_id in enumerate(
        active_device_ids
    ):

        info = public_info(
            device_id
        )


        with material_columns[
            index
        ]:

            render_html(
                f"""
                <div class="material-card">

                    <div class="material-symbol">
                        {info['symbol']}
                    </div>

                    <div class="material-name">
                        {info['short_name']}
                    </div>

                    <div class="material-description">
                        {info['description']}
                    </div>

                </div>
                """
            )


    st.caption(
        "These are the four literature devices currently "
        "ready for the accelerator experiment. "
        f"The broader literature database contains "
        f"{literature_device_count} device profiles."
    )


    st.divider()


    st.subheader(
        "Current Research Snapshot"
    )


    snapshot_columns = st.columns(
        5
    )


    snapshot_columns[0].metric(
        "Independent Physical Devices",
        independent_device_count,
    )


    snapshot_columns[1].metric(
        "Configurations per Device",
        configurations_per_device,
    )


    snapshot_columns[2].metric(
        "Simulation Cases",
        total_simulation_rows,
    )


    snapshot_columns[3].metric(
        "Near-Optimal Recommendations",
        (
            f"{baseline_success_count}/"
            f"{independent_device_count}"
        ),
    )


    snapshot_columns[4].metric(
        "Near-Optimal Success",
        f"{baseline_success_rate:.0f}%",
    )


    render_html(
        f"""
        <div class="callout">

            <strong>
                What does {baseline_success_rate:.0f}% mean?
            </strong>

            <br><br>

            The AI recommendation was within
            {NEAR_OPTIMAL_TOLERANCE_PP:.1f}
            percentage points of the exhaustive
            raw-best accuracy for
            {baseline_success_count} of the
            {independent_device_count}
            completely held-out devices.

            <br><br>

            This is a
            <strong>
                near-optimal recommendation success rate
            </strong>,
            not a statement that the AI is simply
            "{baseline_success_rate:.0f}% accurate."

        </div>
        """
    )


# ============================================================
# FORWARD DESIGN
# ============================================================

with forward_tab:

    st.subheader(
        "Forward Design"
    )


    st.caption(
        "Start from a memristor material and ask: "
        "Which accelerator configuration should I use?"
    )


    st.markdown(
        f"## {selected_material['symbol']} — "
        f"{selected_material['name']}"
    )


    st.write(
        selected_material[
            "description"
        ]
    )


    st.divider()


    # ========================================================
    # DEVICE BEHAVIOR
    # ========================================================

    st.markdown(
        "### 1. Device Behavior"
    )


    if state_count_available == 1:

        physical_state_display = str(
            physical_state_count
        )

        physical_state_note = (
            "Literature-supported physical "
            "conductance-state capability."
        )

    else:

        physical_state_display = (
            "No fixed count"
        )

        physical_state_note = (
            "The literature does not provide "
            "a fixed discrete state count."
        )


    device_columns = st.columns(
        4
    )


    with device_columns[0]:

        render_info_card(
            "Material",
            selected_material[
                "symbol"
            ],
            selected_material[
                "short_name"
            ],
        )


    with device_columns[1]:

        render_info_card(
            "Conductance Behavior",
            public_conductance_mode,
            (
                "Electrical switching behavior "
                "used by the accelerator simulator."
            ),
        )


    with device_columns[2]:

        render_info_card(
            "ON/OFF Ratio",
            format_ratio(
                device_ratio
            ),
            (
                "Ratio between the higher- and "
                "lower-conductance operating states."
            ),
        )


    with device_columns[3]:

        render_info_card(
            "Physical Conductance States",
            physical_state_display,
            physical_state_note,
        )


    st.write("")


    if profile is not None:

        detail_left, detail_right = st.columns(
            2
        )


        with detail_left:

            st.write(
                "**Device stack:** "
                + clean_text(
                    profile[
                        "device_stack"
                    ]
                )
            )


            st.write(
                "**Active material:** "
                + clean_text(
                    profile[
                        "active_material"
                    ]
                )
            )


        with detail_right:

            st.write(
                "**Published source:** "
                + clean_text(
                    profile[
                        "source_title"
                    ]
                )
            )


            st.write(
                "**Publication year:** "
                + clean_text(
                    profile[
                        "year"
                    ]
                )
            )


    st.info(
        selected_material[
            "nano_note"
        ]
    )


    st.divider()


    # ========================================================
    # EVIDENCE-AWARE DEVICE PROFILE
    # ========================================================

    st.markdown(
        "### Evidence-Aware Device Profile"
    )


    st.write(
        "The simulator does not treat every input as equally "
        "supported. Device parameters are separated from "
        "consistency checks so that evidence quality and audit "
        "quality are not mixed together."
    )


    selected_trace_forward = trace[

        trace[
            "device_id"
        ]
        ==
        selected_device

    ].copy()


    selected_audit_forward = audit[

        audit[
            "device_id"
        ]
        ==
        selected_device

    ].copy()


    input_audit_forward = selected_audit_forward[

        ~selected_audit_forward.apply(
            is_integrity_check_record,
            axis=1,
        )

    ].copy()


    integrity_audit_forward = selected_audit_forward[

        selected_audit_forward.apply(
            is_integrity_check_record,
            axis=1,
        )

    ].copy()


    trace_lookup = {}


    for _, trace_row in selected_trace_forward.iterrows():

        property_name = clean_text(
            trace_row.get(
                "property_name"
            ),
            fallback="",
        )

        if (
            property_name
            and property_name not in trace_lookup
        ):

            trace_lookup[
                property_name
            ] = trace_row


    evidence_rows = []


    for _, audit_row in input_audit_forward.iterrows():

        property_name = clean_text(
            audit_row.get(
                "property_name"
            ),
            fallback="",
        )


        trace_row = trace_lookup.get(
            property_name
        )


        simulator_value = audit_row.get(
            "simulator_value"
        )


        trace_value = audit_row.get(
            "trace_value"
        )


        value_type = audit_row.get(
            "value_type"
        )


        category = evidence_category(
            value_type,
            simulator_value=simulator_value,
            trace_value=trace_value,
        )


        if trace_row is not None:

            unit = clean_text(
                trace_row.get(
                    "unit"
                ),
                fallback="",
            )

            source_title = clean_text(
                trace_row.get(
                    "source_title"
                ),
                fallback="Not specified",
            )

            doi = clean_text(
                trace_row.get(
                    "doi"
                ),
                fallback="Not specified",
            )

            source_note = clean_text(
                trace_row.get(
                    "source_note"
                ),
                fallback="Not specified",
            )

            source_location = (
                source_location_text(
                    trace_row.get(
                        "page"
                    ),
                    trace_row.get(
                        "figure_or_table"
                    ),
                )
            )

        else:

            unit = ""
            source_title = "Not linked"
            doi = "Not linked"
            source_note = "Not linked"
            source_location = "Not linked"


        simulator_display = format_evidence_value(
            simulator_value,
            property_name,
            unit,
        )


        trace_display = format_evidence_value(
            trace_value,
            property_name,
            unit,
        )


        evidence_rows.append(
            {

                "Parameter":
                    human_property_name(
                        property_name
                    ),

                "Simulator Value":
                    simulator_display,

                "Literature / Source Value":
                    trace_display,

                "Evidence Status":
                    category,

                "Source Location":
                    source_location,

                "Audit":
                    clean_text(
                        audit_row.get(
                            "status"
                        ),
                        fallback="Not checked",
                    ),

                "Explanation":
                    clean_text(
                        audit_row.get(
                            "message"
                        ),
                        fallback="No audit note stored.",
                    ),

                "Source Note":
                    source_note,

                "Source":
                    source_title,

                "DOI / Identifier":
                    doi,

            }
        )


    evidence_df = pd.DataFrame(
        evidence_rows
    )


    category_order = [
        "Reported",
        "Derived",
        "Assumed",
        "Missing",
    ]


    evidence_counts = {

        category: int(
            (
                evidence_df[
                    "Evidence Status"
                ]
                ==
                category
            ).sum()
        )
        if not evidence_df.empty
        else 0

        for category
        in category_order

    }


    evidence_metric_columns = st.columns(
        4
    )


    evidence_metric_columns[0].metric(
        "Reported",
        evidence_counts[
            "Reported"
        ],
    )


    evidence_metric_columns[1].metric(
        "Derived",
        evidence_counts[
            "Derived"
        ],
    )


    evidence_metric_columns[2].metric(
        "Assumed",
        evidence_counts[
            "Assumed"
        ],
    )


    evidence_metric_columns[3].metric(
        "Missing",
        evidence_counts[
            "Missing"
        ],
    )


    if evidence_df.empty:

        st.warning(
            "No simulator-input evidence records are stored "
            "for this device."
        )

    else:

        compact_evidence_columns = [
            "Parameter",
            "Simulator Value",
            "Literature / Source Value",
            "Evidence Status",
            "Source Location",
            "Audit",
        ]


        st.dataframe(
            evidence_df[
                compact_evidence_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


        with st.expander(
            "View full evidence trail and source details"
        ):

            st.dataframe(
                evidence_df,
                use_container_width=True,
                hide_index=True,
            )


    render_html(
        """
        <div class="callout">

            <strong>
                How to read the evidence status
            </strong>

            <br><br>

            <strong>Reported</strong> — directly supported by
            the stored literature/experimental record.

            <br>

            <strong>Derived</strong> — calculated or inferred
            from reported values.

            <br>

            <strong>Assumed</strong> — explicitly introduced by
            the simulator or modeling workflow.

            <br>

            <strong>Missing</strong> — not available in the
            stored evidence and not silently presented as an
            experimental fact.

        </div>
        """
    )


    if evidence_counts[
        "Assumed"
    ] > 0:

        st.warning(
            f"This device currently uses "
            f"{evidence_counts['Assumed']} explicitly "
            f"assumption-based simulator input(s)."
        )


    if evidence_counts[
        "Missing"
    ] > 0:

        st.warning(
            f"This device currently has "
            f"{evidence_counts['Missing']} parameter(s) "
            f"classified as missing in the stored evidence."
        )


    st.markdown(
        "#### Evidence Integrity Checks"
    )


    st.caption(
        "These rows do not represent additional device "
        "parameters. They verify that derived/profile values "
        "remain internally consistent with the stored evidence."
    )


    if integrity_audit_forward.empty:

        st.info(
            "No separate integrity-check records are stored "
            "for this device."
        )

    else:

        integrity_rows = []


        for _, check_row in integrity_audit_forward.iterrows():

            property_name = clean_text(
                check_row.get(
                    "property_name"
                ),
                fallback="",
            )


            integrity_rows.append(
                {

                    "Check":
                        human_property_name(
                            property_name
                        ),

                    "Computed / Profile Value":
                        clean_text(
                            check_row.get(
                                "simulator_value"
                            ),
                            fallback="—",
                        ),

                    "Reference Value":
                        clean_text(
                            check_row.get(
                                "trace_value"
                            ),
                            fallback="—",
                        ),

                    "Status":
                        clean_text(
                            check_row.get(
                                "status"
                            ),
                            fallback="Not checked",
                        ),

                    "Explanation":
                        clean_text(
                            check_row.get(
                                "message"
                            ),
                            fallback="No audit note stored.",
                        ),

                }
            )


        integrity_df = pd.DataFrame(
            integrity_rows
        )


        st.dataframe(
            integrity_df,
            use_container_width=True,
            hide_index=True,
        )


        integrity_pass_count = int(
            (
                integrity_df[
                    "Status"
                ]
                ==
                "PASS"
            ).sum()
        )


        if (
            integrity_pass_count
            ==
            len(
                integrity_df
            )
        ):

            st.success(
                f"All {integrity_pass_count} stored "
                f"evidence-integrity checks PASS."
            )

        else:

            st.warning(
                f"{integrity_pass_count} of "
                f"{len(integrity_df)} stored "
                f"evidence-integrity checks PASS."
            )


    st.divider()



    # ========================================================
    # ZERO-SHOT AI
    # ========================================================

    st.markdown(
        "### 2. Zero-Shot AI Recommendation"
    )


    training_public_names = [

        public_device_label(
            device_id
        )

        for device_id
        in active_device_ids

        if device_id != selected_device

    ]


    hidden_device_name = (
        f"{selected_material['symbol']} — "
        f"{selected_material['short_name']}"
    )


    training_device_text = " • ".join(
        training_public_names
    )


    render_html(
        f"""
        <div class="zero-shot-box">

            <div class="zero-shot-kicker">
                Zero-shot test
            </div>

            <div class="zero-shot-title">
                The selected memristor is completely hidden
                from model training.
            </div>

            <div class="zero-shot-row">
                <span class="zero-shot-label">
                    Hidden device:
                </span>
                {hidden_device_name}
            </div>

            <div class="zero-shot-row">
                <span class="zero-shot-label">
                    Training devices:
                </span>
                {training_device_text}
            </div>

            <div class="zero-shot-note">
                No simulated accuracy values from
                <strong>{hidden_device_name}</strong>
                are used to train the model in this fold.
                The AI must recommend an accelerator using
                design knowledge learned from the other
                memristor technologies.
            </div>

        </div>
        """
    )


    st.markdown(
        "#### AI Recommendation"
    )


    recommendation_columns = st.columns(
        5
    )


    recommendation_columns[0].metric(
        "Crossbar",
        (
            f"{recommended_crossbar}"
            f" × "
            f"{recommended_crossbar}"
        ),
    )


    recommendation_columns[1].metric(
        "Weight Precision",
        f"{recommended_weight_bits}-bit",
    )


    recommendation_columns[2].metric(
        "ADC Precision",
        f"{recommended_adc_bits}-bit",
    )


    recommendation_columns[3].metric(
        "Effective Weight Levels",
        recommended_effective_levels,
    )


    recommendation_columns[4].metric(
        "Physical Cells per Weight",
        recommended_cells_per_weight,
    )


    prediction_left, prediction_center, prediction_right = (
        st.columns(
            [1, 1.15, 1]
        )
    )


    with prediction_center:

        st.metric(
            "Predicted Accuracy",
            f"{recommended_predicted_accuracy:.2f}%",
        )


    with st.expander(
        "Prediction uncertainty details"
    ):

        uncertainty_columns = st.columns(
            2
        )


        uncertainty_columns[0].metric(
            "Random-Forest Tree Disagreement",
            f"{recommended_uncertainty:.3f} pp",
        )


        uncertainty_columns[1].metric(
            "Conservative Prediction Score",
            f"{recommended_lower_bound:.2f}%",
        )


        st.caption(
            "Conservative prediction score = predicted accuracy "
            "minus Random-Forest tree disagreement. "
            "Tree disagreement is an uncertainty heuristic, "
            "not a calibrated confidence interval and not a "
            "measured accuracy value."
        )


    recommendation_details = pd.DataFrame(
        [
            {
                "Crossbar": (
                    f"{recommended_crossbar} × "
                    f"{recommended_crossbar}"
                ),
                "Weight Precision": (
                    f"{recommended_weight_bits}-bit"
                ),
                "ADC Precision": (
                    f"{recommended_adc_bits}-bit"
                ),
                "Effective Levels": recommended_effective_levels,
                "Cells / Weight": recommended_cells_per_weight,
                "Mapping": mapping_public_name,
                "Relative Cost Proxy": int(
                    round(recommended_cost_proxy)
                ),
            }
        ]
    )


    st.dataframe(
        recommendation_details,
        use_container_width=True,
        hide_index=True,
    )


    render_html(
        f"""
        <div class="callout">

            <strong>AI-selected design</strong>

            <br><br>

            {selected_material['symbol']}
            → {recommended_crossbar} ×
            {recommended_crossbar} crossbar
            → {recommended_weight_bits}-bit weights
            → {recommended_adc_bits}-bit ADC

        </div>
        """
    )


    # ========================================================
    # WHY THIS DESIGN
    # ========================================================

    st.markdown(
        "### Why this design?"
    )


    st.caption(
        "This explanation separates device/simulator constraints "
        "from the AI model's learned selection. It does not claim "
        "that the Random Forest proves physical causation."
    )


    if state_count_available == 1:

        state_explanation_title = (
            f"{physical_state_count} physical states "
            f"→ {recommended_effective_levels} effective levels"
        )

        state_explanation_text = (
            "The literature-supported physical state capability "
            "is used by the simulator's weight-mapping model. "
            "For this selected configuration, that mapping yields "
            f"{recommended_effective_levels} effective signed "
            "weight levels."
        )

    else:

        state_explanation_title = (
            f"No fixed physical state count "
            f"→ {recommended_effective_levels} effective levels"
        )

        state_explanation_text = (
            "This literature profile does not report a fixed "
            "discrete state count. The displayed effective levels "
            "therefore come from the simulator's idealized "
            "accelerator mapping and must not be interpreted as "
            "measured physical conductance states."
        )


    why_row_1 = st.columns(
        2
    )


    with why_row_1[0]:

        render_html(
            f"""
            <div class="why-card">

                <div class="why-kicker">
                    Physical device constraint
                </div>

                <div class="why-title">
                    {state_explanation_title}
                </div>

                <div class="why-text">
                    {state_explanation_text}
                </div>

            </div>
            """
        )


    with why_row_1[1]:

        render_html(
            f"""
            <div class="why-card">

                <div class="why-kicker">
                    Weight representation
                </div>

                <div class="why-title">
                    {recommended_weight_bits}-bit requested precision
                </div>

                <div class="why-text">
                    The accelerator requests
                    {recommended_weight_bits}-bit neural weights,
                    but the physically realizable representation
                    is constrained by the selected device mapping.
                    This configuration uses
                    {recommended_cells_per_weight}
                    physical memristor cells per weight.
                </div>

            </div>
            """
        )


    st.write("")


    why_row_2 = st.columns(
        2
    )


    with why_row_2[0]:

        render_html(
            f"""
            <div class="why-card">

                <div class="why-kicker">
                    Readout precision
                </div>

                <div class="why-title">
                    {recommended_adc_bits}-bit ADC
                </div>

                <div class="why-text">
                    ADC precision is one of the accelerator
                    variables explored by the model.
                    The AI selected {recommended_adc_bits} bits
                    because this complete configuration lies in
                    its predicted near-optimal region.
                    This does not mean {recommended_adc_bits} bits
                    is universally optimal for
                    {selected_material['symbol']}.
                </div>

            </div>
            """
        )


    with why_row_2[1]:

        render_html(
            f"""
            <div class="why-card">

                <div class="why-kicker">
                    Crossbar architecture
                </div>

                <div class="why-title">
                    {recommended_crossbar} ×
                    {recommended_crossbar} crossbar
                </div>

                <div class="why-text">
                    Crossbar size changes the accelerator
                    configuration evaluated by the simulator.
                    The model placed this
                    {recommended_crossbar} ×
                    {recommended_crossbar} option inside the
                    predicted near-optimal design region for
                    the completely held-out device.
                </div>

            </div>
            """
        )


    render_html(
        f"""
        <div class="callout">

            <strong>
                How the final recommendation is chosen
            </strong>

            <br><br>

            The Random-Forest model predicts accuracy across
            the candidate design space. The recommender then
            selects a lower-cost design from the
            <strong>predicted near-optimal region</strong>,
            rather than simply taking the configuration with
            the single highest predicted accuracy.

            <br><br>

            Recorded relative hardware-cost proxy for this
            recommendation:
            <strong>{recommended_cost_proxy:,.0f}</strong>.

        </div>
        """
    )



    # ========================================================
    # TOP CANDIDATES
    # ========================================================

    st.markdown(
        "### Top 5 Predicted Candidates"
    )


    top_five = (

        candidate_df
        .sort_values(

            by=[
                "predicted_accuracy",
                "relative_hardware_cost_proxy",
            ],

            ascending=[
                False,
                True,
            ],

        )
        .head(
            5
        )
        .copy()

    )


    top_five_display = pd.DataFrame(
        {

            "Configuration": [

                config_label(
                    row[
                        "crossbar_size"
                    ],
                    row[
                        "requested_weight_bits"
                    ],
                    row[
                        "adc_bits"
                    ],
                )

                for _, row
                in top_five.iterrows()

            ],

            "Predicted Accuracy": [

                f"{float(value):.2f}%"

                for value
                in top_five[
                    "predicted_accuracy"
                ]

            ],

            "Effective Levels": (

                top_five[
                    "effective_weight_levels"
                ]
                .astype(int)
                .tolist()

            ),

            "Cells per Weight": (

                top_five[
                    "physical_cells_per_weight"
                ]
                .astype(int)
                .tolist()

            ),

            "Relative Cost Proxy": (

                top_five[
                    "relative_hardware_cost_proxy"
                ]
                .round(0)
                .astype(int)
                .tolist()

            ),

        }
    )


    st.dataframe(
        top_five_display,
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "The final AI recommendation is selected from the "
        "predicted near-optimal region using hardware-cost "
        "criteria. Therefore it does not necessarily have "
        "the single highest predicted accuracy."
    )


    with st.expander(
        f"Explore all "
        f"{configurations_per_device} candidates"
    ):

        full_candidates = pd.DataFrame(
            {

                "Crossbar": (

                    candidate_df[
                        "crossbar_size"
                    ]
                    .astype(int)

                ),

                "Weight Bits": (

                    candidate_df[
                        "requested_weight_bits"
                    ]
                    .astype(int)

                ),

                "Effective Levels": (

                    candidate_df[
                        "effective_weight_levels"
                    ]
                    .astype(int)

                ),

                "Cells per Weight": (

                    candidate_df[
                        "physical_cells_per_weight"
                    ]
                    .astype(int)

                ),

                "ADC Bits": (

                    candidate_df[
                        "adc_bits"
                    ]
                    .astype(int)

                ),

                "Predicted Accuracy": (

                    candidate_df[
                        "predicted_accuracy"
                    ]
                    .round(
                        3
                    )

                ),

                "Prediction Disagreement": (

                    candidate_df[
                        "prediction_uncertainty"
                    ]
                    .round(
                        3
                    )

                ),

                "Relative Cost Proxy": (

                    candidate_df[
                        "relative_hardware_cost_proxy"
                    ]
                    .round(
                        0
                    )
                    .astype(int)

                ),

            }
        )


        full_candidates = (

            full_candidates
            .sort_values(

                by=[
                    "Predicted Accuracy",
                    "Relative Cost Proxy",
                ],

                ascending=[
                    False,
                    True,
                ],

            )

        )


        st.dataframe(
            full_candidates,
            use_container_width=True,
            hide_index=True,
        )


    st.divider()


    # ========================================================
    # EXHAUSTIVE VALIDATION
    # ========================================================

    st.markdown(
        "### 3. Exhaustive Validation"
    )


    st.write(
        "The recommendation can now be checked against "
        f"all {configurations_per_device} simulated designs "
        "for this held-out device."
    )


    reveal_key = (
        f"validation_"
        f"{selected_device}"
    )


    if reveal_key not in st.session_state:

        st.session_state[
            reveal_key
        ] = False


    if st.button(
        "Reveal Validation Result",
        type="primary",
        key=(
            f"reveal_"
            f"{selected_device}"
        ),
    ):

        st.session_state[
            reveal_key
        ] = True


    if not st.session_state[
        reveal_key
    ]:

        st.info(
            "Validation is hidden so the zero-shot "
            "recommendation can be viewed before revealing "
            "its exhaustive ground truth."
        )


    else:

        validation_columns = st.columns(
            4
        )


        validation_columns[0].metric(
            "Predicted Accuracy",
            f"{recommended_predicted_accuracy:.2f}%",
        )


        validation_columns[1].metric(
            "Actual Recommended Accuracy",
            f"{recommended_actual_accuracy:.2f}%",
        )


        validation_columns[2].metric(
            "Exhaustive Best",
            f"{raw_best_accuracy:.2f}%",
        )


        validation_columns[3].metric(
            "Software Baseline",
            f"{software_baseline_accuracy:.2f}%",
        )


        if recommended_success:

            render_html(
                f"""
                <div class="success-card">

                    <strong>
                        Near-optimal recommendation — PASS
                    </strong>

                    <br><br>

                    The recommendation is within
                    {NEAR_OPTIMAL_TOLERANCE_PP:.1f}
                    percentage points of the exhaustive
                    raw-best accuracy.

                </div>
                """
            )

        else:

            render_html(
                f"""
                <div class="warning-card">

                    <strong>
                        Near-optimal recommendation — FAIL
                    </strong>

                    <br><br>

                    The recommendation is more than
                    {NEAR_OPTIMAL_TOLERANCE_PP:.1f}
                    percentage points below the exhaustive
                    raw-best accuracy.

                </div>
                """
            )


        st.markdown(
            "#### Validation Visuals"
        )


        visual_left, visual_right = st.columns(
            [1.15, 0.85]
        )


        with visual_left:

            render_accuracy_validation_chart(
                recommended_predicted_accuracy,
                recommended_actual_accuracy,
                raw_best_accuracy,
                software_baseline_accuracy,
            )

            st.caption(
                "The y-axis is intentionally zoomed so the "
                "small accuracy differences are readable. "
                "The numeric labels should be used when "
                "judging the magnitude of the differences."
            )


        with visual_right:

            render_regret_threshold_chart(
                recommended_regret,
                NEAR_OPTIMAL_TOLERANCE_PP,
            )

            st.caption(
                f"Regret = {recommended_regret:.2f} pp; "
                f"success threshold = "
                f"{NEAR_OPTIMAL_TOLERANCE_PP:.2f} pp. "
                f"Result: "
                f"{'PASS' if recommended_success else 'FAIL'}."
            )


        st.markdown(
            f"#### Predicted vs Actual Across "
            f"{configurations_per_device} Configurations"
        )


        render_predicted_vs_actual_chart(
            candidate_df,
            recommended_row,
            raw_best_row,
        )


        st.caption(
            "Each point is one accelerator configuration for "
            "the held-out device. The dashed diagonal is ideal "
            "prediction (predicted = actual). The diamond marks "
            "the AI recommendation and the triangle marks the "
            "exhaustive raw-accuracy optimum."
        )


        comparison_rows = [

            {
                "Design":
                    "AI Recommendation",

                "Configuration":
                    config_label(
                        recommended_crossbar,
                        recommended_weight_bits,
                        recommended_adc_bits,
                    ),

                "Accuracy":
                    f"{recommended_actual_accuracy:.2f}%",

                "Purpose":
                    "AI-selected configuration",
            },

            {
                "Design":
                    "Raw Accuracy Best",

                "Configuration":
                    config_label(
                        raw_best_row[
                            "crossbar_size"
                        ],
                        raw_best_row[
                            "requested_weight_bits"
                        ],
                        raw_best_row[
                            "adc_bits"
                        ],
                    ),

                "Accuracy":
                    f"{raw_best_accuracy:.2f}%",

                "Purpose":
                    "Maximum simulated accuracy",
            },

            {
                "Design":
                    "Cost-Aware Optimum",

                "Configuration":
                    config_label(
                        cost_aware_crossbar,
                        cost_aware_weight_bits,
                        cost_aware_adc_bits,
                    ),

                "Accuracy":
                    f"{cost_aware_accuracy:.2f}%",

                "Purpose":
                    (
                        "Lower-cost design inside "
                        "near-optimal region"
                    ),
            },

        ]


        st.dataframe(
            pd.DataFrame(
                comparison_rows
            ),
            use_container_width=True,
            hide_index=True,
        )


        if pd.isna(
            recommended_rank
        ):

            rank_text = (
                "Outside the true near-optimal region"
            )

        else:

            rank_text = str(
                int(
                    recommended_rank
                )
            )


        status_columns = st.columns(
            3
        )


        with status_columns[0]:

            render_status_card(
                "Exact Cost-Aware Match",
                (
                    "Yes"
                    if recommended_exact
                    else "No"
                ),
            )


        with status_columns[1]:

            render_status_card(
                "Within Cost-Aware Top 3",
                (
                    "Yes"
                    if recommended_top3
                    else "No"
                ),
            )


        with status_columns[2]:

            render_status_card(
                "Cost-Aware Rank",
                rank_text,
            )


        st.caption(
            "Near-optimal success measures accuracy. "
            "Cost-aware rank instead compares hardware-cost "
            "ordering among designs inside the true "
            "near-optimal accuracy region."
        )


# ============================================================
# WHY NANO
# ============================================================

with nano_tab:

    st.subheader(
        "Why is this a nanotechnology project?"
    )


    st.write(
        """
        The AI is not simply ranking material names.

        Literature-derived memristor electrical
        characteristics affect how neural-network weights
        can be represented, which then changes accelerator
        behavior.
        """
    )


    st.markdown(
        f"## {selected_material['symbol']} — "
        f"{selected_material['short_name']}"
    )


    st.write(
        selected_material[
            "description"
        ]
    )


    nano_columns = st.columns(
        4
    )


    with nano_columns[0]:

        render_info_card(
            "Nanomaterial",
            selected_material[
                "symbol"
            ],
            selected_material[
                "short_name"
            ],
        )


    with nano_columns[1]:

        render_info_card(
            "Device Behavior",
            public_conductance_mode,
            (
                "Electrical switching behavior "
                "used by the accelerator model."
            ),
        )


    with nano_columns[2]:

        render_info_card(
            "Weight Representation",
            (
                f"{recommended_effective_levels} "
                f"effective levels"
            ),
            mapping_public_name,
        )


    with nano_columns[3]:

        render_info_card(
            "Physical Cell Cost",
            (
                f"{recommended_cells_per_weight} "
                f"cells / weight"
            ),
            (
                "Physical memristor cells required "
                "by the selected mapping."
            ),
        )


    st.write("")


    if conductance_mode.upper() == "DISCRETE_BINARY":

        st.info(
            "This is a binary memristor. Each physical "
            "cell still has only two physical conductance "
            "states. Higher neural-weight precision is "
            "created by bit slicing across multiple cells."
        )


    elif conductance_mode.upper() == "DISCRETE_MULTILEVEL":

        st.info(
            "This multilevel device has a finite physical "
            "state capability. That state capability directly "
            "caps the effective neural-weight precision "
            "available to the accelerator."
        )


    else:

        st.warning(
            "This literature profile does not report a fixed "
            "discrete state count. The displayed effective "
            "weight levels come from an idealized accelerator "
            "mapping and must not be interpreted as measured "
            "physical conductance states."
        )


    st.divider()


    st.markdown(
        "### How Device Properties Affect the Accelerator"
    )


    effect_columns = st.columns(
        3
    )


    with effect_columns[0]:

        render_info_card(
            "Physical State Capability",
            "Weight Precision",
            (
                "Too few available states can strongly "
                "reduce effective neural-weight precision."
            ),
        )


    with effect_columns[1]:

        render_info_card(
            "ON/OFF Ratio",
            "ADC Interaction",
            (
                "Conductance-window importance changes "
                "depending on ADC resolution."
            ),
        )


    with effect_columns[2]:

        render_info_card(
            "Binary Devices",
            "Cell-Cost Tradeoff",
            (
                "Higher precision requires additional "
                "bit-sliced physical cells."
            ),
        )


    render_html(
        """
        <div class="callout">

            <strong>Current model limitation</strong>

            <br><br>

            Absolute conductance magnitude is effectively
            normalized away when the ON/OFF ratio is preserved.

            <br><br>

            Real hardware can additionally depend on line
            resistance, current magnitude, sensing limits,
            power and heating.

        </div>
        """
    )


# ============================================================
# RESEARCH EVIDENCE
# ============================================================

with evidence_tab:

    st.subheader(
        "How strong is the current evidence?"
    )


    st.markdown(
        "### Current Research Scope"
    )


    scope_columns_1 = st.columns(
        4
    )


    with scope_columns_1[0]:

        render_info_card(
            "Simulation-Ready Devices",
            independent_device_count,
            "Independent physical devices used in the current zero-shot experiment.",
        )


    with scope_columns_1[1]:

        render_info_card(
            "Broader Literature Profiles",
            literature_device_count,
            "Literature-derived device profiles currently stored in the project database.",
        )


    with scope_columns_1[2]:

        render_info_card(
            "Configurations per Device",
            configurations_per_device,
            "Accelerator configurations evaluated for each active simulation-ready device.",
        )


    with scope_columns_1[3]:

        render_info_card(
            "Simulation Cases",
            total_simulation_rows,
            "Device-configuration simulation cases; not independent experimental devices.",
        )


    st.write("")


    scope_columns_2 = st.columns(
        4
    )


    with scope_columns_2[0]:

        render_info_card(
            "Workload",
            "MNIST",
            "Current neural-network workload used by the recorded accelerator evaluation.",
        )


    with scope_columns_2[1]:

        render_info_card(
            "Recommendation Model",
            "Random Forest",
            "Cross-device model used to predict accelerator accuracy for a completely held-out device.",
        )


    with scope_columns_2[2]:

        render_info_card(
            "Hardware Validation",
            "Not yet",
            "The current simulator has not been validated against a fabricated memristor crossbar accelerator.",
        )


    with scope_columns_2[3]:

        render_info_card(
            "Uncertainty",
            "Heuristic",
            "Random-Forest tree disagreement is shown as an uncalibrated uncertainty indicator.",
        )


    render_html(
        f"""
        <div class="callout">

            <strong>
                How to interpret the current evidence
            </strong>

            <br><br>

            The project currently evaluates
            <strong>{independent_device_count} independent
            simulation-ready devices</strong> and
            <strong>{total_simulation_rows} device-configuration
            simulation cases</strong>.

            <br><br>

            The simulation cases are repeated accelerator designs
            derived from those physical devices. They must not be
            described as {total_simulation_rows} independent
            experimental devices.

            <br><br>

            The present results are therefore
            <strong>pilot proof-of-concept evidence</strong>
            for literature-grounded device-to-accelerator
            recommendation, not proof of broad physical
            generalization across memristor technologies.

        </div>
        """
    )


    st.divider()


    evidence_metrics = st.columns(
        5
    )


    evidence_metrics[0].metric(
        "Literature Profiles",
        literature_device_count,
    )


    evidence_metrics[1].metric(
        "Simulation-Ready Devices",
        independent_device_count,
    )


    evidence_metrics[2].metric(
        "Simulation Cases",
        total_simulation_rows,
    )


    evidence_metrics[3].metric(
        "Near-Optimal Success",
        (
            f"{baseline_success_count}/"
            f"{independent_device_count}"
        ),
    )


    evidence_metrics[4].metric(
        "Mean Regret",
        f"{baseline_mean_regret:.3f} pp",
    )


    render_html(
        f"""
        <div class="callout">

            The project contains
            <strong>
                {total_simulation_rows}
                device-configuration simulation cases
            </strong>,
            but those cases come from only
            <strong>
                {independent_device_count}
                independent physical devices
            </strong>.

            <br><br>

            The current results are therefore treated as
            pilot proof-of-concept evidence rather than proof
            of broad cross-device generalization.

        </div>
        """
    )


    st.divider()


    st.markdown(
        "### Zero-Shot Results by Held-Out Device"
    )


    result_rows = []


    for _, row in summary.iterrows():

        device_id = clean_text(
            row[
                "held_out_device"
            ]
        )


        info = public_info(
            device_id
        )


        success = bool_value(
            row[
                "baseline_near_optimal_success"
            ]
        )


        result_rows.append(
            {

                "Material":
                    (
                        f"{info['symbol']} — "
                        f"{info['short_name']}"
                    ),

                "AI Recommendation":
                    config_label(
                        row[
                            "baseline_crossbar"
                        ],
                        row[
                            "baseline_weight_bits"
                        ],
                        row[
                            "baseline_adc_bits"
                        ],
                    ),

                "Actual Accuracy":
                    (
                        f"{float(row['baseline_actual_accuracy']):.2f}%"
                    ),

                "Regret":
                    (
                        f"{float(row['baseline_regret_pp']):.2f} pp"
                    ),

                "Result":
                    (
                        "PASS"
                        if success
                        else "FAIL"
                    ),
            }
        )


    st.dataframe(
        pd.DataFrame(
            result_rows
        ),
        use_container_width=True,
        hide_index=True,
    )


    st.caption(
        "Regret = exhaustive raw-best accuracy minus "
        "the actual accuracy of the AI-recommended design. "
        f"PASS means regret ≤ "
        f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} percentage points."
    )


    st.divider()


    st.markdown(
        "### What does the AI actually use?"
    )


    feature_left, feature_right = st.columns(
        2
    )


    with feature_left:

        render_info_card(
            "AI Input Features",
            len(
                MODEL_FEATURE_LABELS
            ),
            (
                "Device and accelerator descriptors "
                "used by the Random-Forest predictor."
            ),
        )


        st.write("")


        for feature in MODEL_FEATURE_LABELS:

            st.write(
                f"• {feature}"
            )


    with feature_right:

        render_info_card(
            "Independent Design Variables",
            3,
            (
                "These variables generate the accelerator "
                "configuration search space."
            ),
        )


        st.write("")


        st.write(
            "• Crossbar size"
        )

        st.write(
            "• Requested weight precision"
        )

        st.write(
            "• ADC precision"
        )


        st.info(
            "245 configurations does not mean "
            "245 independent parameters."
        )


    st.divider()


    st.markdown(
        "### Design-Space Expansion"
    )


    render_html(
        f"""
        <div class="progress-row">

            <div class="progress-box">

                <div class="progress-value">
                    27
                </div>

                <div class="progress-label">
                    Original pilot
                </div>

            </div>

            <div class="progress-arrow">
                →
            </div>

            <div class="progress-box">

                <div class="progress-value">
                    64
                </div>

                <div class="progress-label">
                    First expansion
                </div>

            </div>

            <div class="progress-arrow">
                →
            </div>

            <div class="progress-box">

                <div class="progress-value">
                    {configurations_per_device}
                </div>

                <div class="progress-label">
                    Current configurations / device
                </div>

            </div>

        </div>
        """
    )


    st.write("")


    st.write(
        f"Current design space: "
        f"**{len(crossbar_values)} crossbar choices × "
        f"{len(weight_values)} weight choices × "
        f"{len(adc_values)} ADC choices = "
        f"{configurations_per_device} configurations "
        f"per device.**"
    )


    st.caption(
        "The expanded search uses denser values along "
        "existing simulator-supported axes. Unsupported "
        "physical variables were not invented simply to "
        "increase the number of cases."
    )


    st.divider()


    st.markdown(
        "### Prediction Quality"
    )


    quality_columns = st.columns(
        4
    )


    quality_columns[0].metric(
        "Near-Region MAE",
        f"{near_region_mae:.3f} pp",
    )


    quality_columns[1].metric(
        "Near-Region RMSE",
        f"{near_region_rmse:.3f} pp",
    )


    quality_columns[2].metric(
        "Cost-Aware Top 3",
        (
            f"{baseline_top3_count}/"
            f"{independent_device_count}"
        ),
    )


    quality_columns[3].metric(
        "Search Reduction",
        f"{search_reduction_pct:.2f}%",
    )


    st.caption(
        "Search reduction means evaluating one selected "
        "configuration for an unseen device instead of "
        f"exhaustively evaluating all "
        f"{configurations_per_device}. "
        "It is not a claim about total computational cost."
    )


    st.divider()


    st.markdown(
        "### Research Upgrade Path"
    )


    roadmap_columns = st.columns(
        5
    )


    with roadmap_columns[0]:

        render_process_card(
            "01",
            "More Independent Devices",
            "Expand from the current small set of simulation-ready devices to multiple independent devices per material family.",
        )


    with roadmap_columns[1]:

        render_process_card(
            "02",
            "Stronger Validation",
            "Use leave-one-device-out and leave-one-family-out testing with baseline model comparisons.",
        )


    with roadmap_columns[2]:

        render_process_card(
            "03",
            "Nonidealities + Uncertainty",
            "Add variability, noise, drift and missing-data scenarios with calibrated confidence/OOD checks.",
        )


    with roadmap_columns[3]:

        render_process_card(
            "04",
            "Hardware Metrics",
            "Extend beyond accuracy toward energy, latency, area and peripheral overhead trade-offs.",
        )


    with roadmap_columns[4]:

        render_process_card(
            "05",
            "Reverse Nano-Design",
            "Map accelerator requirements back to required device properties and match them to experimental literature devices.",
        )


    st.caption(
        "These are planned research extensions. They are not presented as capabilities of the current prototype."
    )



# ============================================================
# SOURCES AND LIMITATIONS
# ============================================================

with sources_tab:

    st.subheader(
        "Data Sources and Scientific Transparency"
    )


    trust_columns = st.columns(
        4
    )


    trust_columns[0].metric(
        "Literature Profiles",
        literature_device_count,
    )


    trust_columns[1].metric(
        "Traceability Records",
        traceability_row_count,
    )


    trust_columns[2].metric(
        "Provenance Fields Complete",
        (
            f"{provenance_completeness:.1f}%"
            if np.isfinite(
                provenance_completeness
            )
            else "N/A"
        ),
    )


    trust_columns[3].metric(
        "Input Audit",
        f"{audit_pass_count} PASS",
    )


    st.caption(
        f"Audit details: "
        f"{audit_pass_count} PASS, "
        f"{audit_warn_count} warning, "
        f"{audit_fail_count} fail."
    )


    render_html(
        """
        <div class="callout">

            <strong>
                What does provenance completeness mean?
            </strong>

            <br><br>

            It means the stored literature records contain
            the source title, DOI/source identifier, page,
            figure/table information and a source note.

            <br><br>

            It does not mean that every model value is
            experimentally perfect, and it does not mean
            that the accelerator simulator has been validated
            against fabricated hardware.

        </div>
        """
    )


    st.divider()


    st.markdown(
        "### Evidence Classification"
    )


    input_audit_records = audit[

        ~audit.apply(
            is_integrity_check_record,
            axis=1,
        )

    ].copy()


    all_evidence_categories = []


    for _, row in input_audit_records.iterrows():

        all_evidence_categories.append(
            evidence_category(
                row.get(
                    "value_type"
                ),
                simulator_value=row.get(
                    "simulator_value"
                ),
                trace_value=row.get(
                    "trace_value"
                ),
            )
        )


    all_evidence_series = pd.Series(
        all_evidence_categories,
        dtype="object",
    )


    classification_columns = st.columns(
        4
    )


    classification_columns[0].metric(
        "Reported Inputs",
        int(
            (
                all_evidence_series
                ==
                "Reported"
            ).sum()
        ),
    )


    classification_columns[1].metric(
        "Derived Inputs",
        int(
            (
                all_evidence_series
                ==
                "Derived"
            ).sum()
        ),
    )


    classification_columns[2].metric(
        "Assumed Inputs",
        int(
            (
                all_evidence_series
                ==
                "Assumed"
            ).sum()
        ),
    )


    classification_columns[3].metric(
        "Missing Inputs",
        int(
            (
                all_evidence_series
                ==
                "Missing"
            ).sum()
        ),
    )


    st.caption(
        "These counts include device/simulator inputs only. "
        "Consistency-check rows are shown separately and are "
        "not counted as physical input parameters. "
        "Evidence status describes provenance, not measurement "
        "accuracy."
    )



    st.divider()


    st.markdown(
        f"### Source for "
        f"{selected_material['symbol']} — "
        f"{selected_material['short_name']}"
    )


    if profile is not None:

        source_left, source_right = st.columns(
            2
        )


        with source_left:

            st.write(
                "**Publication**"
            )


            st.write(
                clean_text(
                    profile[
                        "source_title"
                    ]
                )
            )


            st.write(
                "**DOI**"
            )


            st.code(
                clean_text(
                    profile[
                        "doi"
                    ]
                )
            )


        with source_right:

            st.write(
                "**Device stack**"
            )


            st.write(
                clean_text(
                    profile[
                        "device_stack"
                    ]
                )
            )


            st.write(
                "**Active material**"
            )


            st.write(
                clean_text(
                    profile[
                        "active_material"
                    ]
                )
            )


    selected_trace = trace[

        trace[
            "device_id"
        ]
        ==
        selected_device

    ].copy()


    with st.expander(
        "View literature provenance records"
    ):

        provenance_display = pd.DataFrame(
            {

                "Property": [

                    human_property_name(
                        value
                    )

                    for value
                    in selected_trace[
                        "property_name"
                    ]

                ],

                "Value":
                    selected_trace[
                        "value"
                    ],

                "Unit":
                    selected_trace[
                        "unit"
                    ],

                "Evidence Type": [

                    human_evidence_type(
                        value
                    )

                    for value
                    in selected_trace[
                        "value_type"
                    ]

                ],

                "Page":
                    selected_trace[
                        "page"
                    ],

                "Figure or Table":
                    selected_trace[
                        "figure_or_table"
                    ],

                "Source Note":
                    selected_trace[
                        "source_note"
                    ],

            }
        )


        st.dataframe(
            provenance_display,
            use_container_width=True,
            hide_index=True,
        )


    selected_audit = audit[

        audit[
            "device_id"
        ]
        ==
        selected_device

    ].copy()


    with st.expander(
        "View simulator-input audit"
    ):

        audit_display = pd.DataFrame(
            {

                "Property": [

                    human_property_name(
                        value
                    )

                    for value
                    in selected_audit[
                        "property_name"
                    ]

                ],

                "Simulator Value":
                    selected_audit[
                        "simulator_value"
                    ],

                "Literature Value":
                    selected_audit[
                        "trace_value"
                    ],

                "Evidence Type": [

                    human_evidence_type(
                        value
                    )

                    for value
                    in selected_audit[
                        "value_type"
                    ]

                ],

                "Status":
                    selected_audit[
                        "status"
                    ],

                "Explanation":
                    selected_audit[
                        "message"
                    ],

            }
        )


        st.dataframe(
            audit_display,
            use_container_width=True,
            hide_index=True,
        )


    assumption_rows = audit[

        audit[
            "value_type"
        ]
        ==
        "ASSUMED"

    ].copy()


    if not assumption_rows.empty:

        st.warning(
            "The current simulator contains "
            f"{len(assumption_rows)} explicitly declared "
            "assumption-based input."
        )


        assumption_display = []


        for _, row in assumption_rows.iterrows():

            info = public_info(

                clean_text(
                    row[
                        "device_id"
                    ]
                )

            )


            assumption_display.append(
                {

                    "Material":
                        (
                            f"{info['symbol']} — "
                            f"{info['short_name']}"
                        ),

                    "Property":
                        human_property_name(
                            row[
                                "property_name"
                            ]
                        ),

                    "Value":
                        row[
                            "simulator_value"
                        ],

                    "Status":
                        row[
                            "status"
                        ],

                    "Explanation":
                        row[
                            "message"
                        ],

                }
            )


        st.dataframe(
            pd.DataFrame(
                assumption_display
            ),
            use_container_width=True,
            hide_index=True,
        )


    st.divider()


    st.markdown(
        "### Current Limitations"
    )


    st.write(
        f"""
        - The current accelerator experiment uses only
          **{independent_device_count} independent
          simulation-ready physical devices**.

        - The **{total_simulation_rows} simulation cases are
          not {total_simulation_rows} independent device
          samples**.

        - Each zero-shot fold trains on only
          **{training_devices_per_fold} physical devices**.

        - Analog profiles without a fixed reported state count
          use an **idealized accelerator precision mapping**.

        - Random-Forest tree disagreement is an
          **uncalibrated uncertainty heuristic**.

        - Absolute conductance magnitude is currently
          normalized away when the conductance ratio is
          preserved.

        - The present model does not yet fully propagate
          thickness, variability, read noise, drift,
          line resistance, power, heating, switching kinetics
          or filament dynamics into accelerator behavior.

        - The simulator has not yet been experimentally
          validated against fabricated memristor crossbar
          accelerator hardware.
        """
    )


    st.info(
        "The current results should therefore be interpreted "
        "as pilot proof-of-concept evidence for "
        "literature-grounded memristor "
        "device-to-accelerator co-design."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "NanoMemristor AI Designer — "
    "Literature-grounded memristor "
    "device-to-accelerator co-design research prototype."
)