from __future__ import annotations

import math
import sys
import textwrap
from pathlib import Path

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
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SUMMARY_FILE = RESULTS_DIR / "zero_shot_summary.csv"
PREDICTIONS_FILE = RESULTS_DIR / "zero_shot_predictions.csv"
STUDY_SUMMARY_FILE = RESULTS_DIR / "zero_shot_study_summary.csv"
FAMILY_SUMMARY_FILE = RESULTS_DIR / "zero_shot_family_summary.csv"
VALIDATION_OVERVIEW_FILE = RESULTS_DIR / "zero_shot_validation_overview.csv"
RELIABILITY_FILE = RESULTS_DIR / "zero_shot_reliability_penalties.csv"

DEVICE_FILE = DATA_DIR / "device_profiles.csv"
TRACE_FILE = DATA_DIR / "source_traceability.csv"
AUDIT_FILE = RESULTS_DIR / "traceability_audit.csv"
BASELINE_FILE = PROJECT_ROOT / "results" / "mnist_baseline_accuracy.txt"
ML_FILE = RESULTS_DIR / "ml_dataset.csv"

EVIDENCE_COVERAGE_FILE = RESULTS_DIR / "evidence_coverage_by_device.csv"
EVIDENCE_SUMMARY_FILE = RESULTS_DIR / "evidence_coverage_summary.csv"

SENSITIVITY_BY_DEVICE_FILE = RESULTS_DIR / "sensitivity_by_device.csv"
SENSITIVITY_SUMMARY_FILE = RESULTS_DIR / "sensitivity_summary.csv"

OOD_DEVICE_FILE = RESULTS_DIR / "ood_uncertainty_by_device.csv"
OOD_OVERVIEW_FILE = RESULTS_DIR / "ood_uncertainty_overview.csv"

NONIDEALITY_DETAIL_FILE = RESULTS_DIR / "nonideality_proxy_by_device.csv"
NONIDEALITY_SUMMARY_FILE = RESULTS_DIR / "nonideality_proxy_summary.csv"

PARETO_FRONT_FILE = RESULTS_DIR / "pareto_front_by_device.csv"
PARETO_SUMMARY_FILE = RESULTS_DIR / "pareto_summary_by_device.csv"
PARETO_OVERVIEW_FILE = RESULTS_DIR / "pareto_overview.csv"

REVERSE_CANDIDATES_FILE = RESULTS_DIR / "reverse_design_candidates.csv"
REVERSE_REQUIREMENTS_FILE = RESULTS_DIR / "reverse_design_requirements.csv"
REVERSE_MATCHES_FILE = RESULTS_DIR / "reverse_design_experiment_matches.csv"
REVERSE_MATCH_SUMMARY_FILE = RESULTS_DIR / "reverse_design_experiment_match_summary.csv"

FUTURE_TARGETS_FILE = RESULTS_DIR / "future_research_targets.csv"
FUTURE_TARGET_SUMMARY_FILE = RESULTS_DIR / "future_research_target_summary.csv"


# ============================================================
# SETTINGS
# ============================================================

NEAR_OPTIMAL_TOLERANCE_PP = 0.5
DEFAULT_DEVICE_ID = "TaOx_01"

PAGES = [
    "Home",
    "Forward Design",
    "Custom Device",
    "Reverse Design",
    "Research Targets",
    "Why Nano?",
    "Research Evidence",
    "Sources & Limitations",
]

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
    "HfOx_01": {
        "symbol": "HfO₂",
        "short_name": "Hafnium Oxide — Variability Profile",
        "name": "Hafnium Oxide Memristor",
        "description": (
            "A literature-derived HfO₂ device profile using a TiN/Ti/HfO₂/W "
            "stack. Its operating resistance values are derived from the "
            "reported experimental figure and retained with explicit provenance."
        ),
    },
    "HfOx_02": {
        "symbol": "HfO₂",
        "short_name": "Hafnium Oxide — Analog Profile",
        "name": "Hafnium Oxide Memristor",
        "description": (
            "An independent hafnium-oxide profile represented as analog "
            "conductance behavior in the accelerator model."
        ),
    },
    "HfZrOx_01": {
        "symbol": "HfZrO₄",
        "short_name": "Hafnium–Zirconium Oxide",
        "name": "Hafnium–Zirconium Oxide Memristor",
        "description": (
            "An independent HfZrO₄-based profile with reported low ON/OFF "
            "contrast and analog behavior. Absolute conductance is normalized "
            "from the reported ratio."
        ),
    },
    "TaOx_01": {
        "symbol": "TaOₓ",
        "short_name": "Tantalum Oxide",
        "name": "Tantalum Oxide Multilevel Memristor",
        "description": (
            "A literature-constrained multilevel tantalum-oxide profile with "
            "a finite experimentally supported conductance-state capability."
        ),
    },
    "TiOx_02_Au": {
        "symbol": "TiOₓ",
        "short_name": "Titanium Oxide — Au Electrode",
        "name": "Titanium Oxide Memristor (Au)",
        "description": (
            "One electrode variant from a shared TiOₓ experimental study. "
            "It is a separate device profile, not an independent source study."
        ),
    },
    "TiOx_02_Ni": {
        "symbol": "TiOₓ",
        "short_name": "Titanium Oxide — Ni Electrode",
        "name": "Titanium Oxide Memristor (Ni)",
        "description": (
            "The Ni-electrode TiOₓ variant. This profile exposed the original "
            "low-ADC recommendation failure and motivated the support-gated "
            "recommendation policy."
        ),
    },
    "TiOx_02_Pt": {
        "symbol": "TiOₓ",
        "short_name": "Titanium Oxide — Pt Electrode",
        "name": "Titanium Oxide Memristor (Pt)",
        "description": (
            "The Pt-electrode variant from the same TiOₓ study as the Au and Ni "
            "profiles. It is handled with study-blocked validation."
        ),
    },
    "TiOx_03": {
        "symbol": "TiO₂",
        "short_name": "Titanium Dioxide — Analog Profile",
        "name": "Titanium Dioxide Memristor",
        "description": (
            "A literature-derived titanium-dioxide profile represented with "
            "analog conductance behavior."
        ),
    },
    "TiOx_04": {
        "symbol": "TiOₓ/TiOᵧ",
        "short_name": "Titanium Oxide — Gradual Profile",
        "name": "TiOₓ/TiOᵧ Gradual Memristor",
        "description": (
            "An independent cross-point TiOₓ/TiOᵧ profile with gradual "
            "multilevel behavior and reported LTP/LTD-like conductance tuning."
        ),
    },
    "ZnO_01": {
        "symbol": "ZnO",
        "short_name": "Zinc Oxide",
        "name": "Zinc Oxide Memristor",
        "description": (
            "A zinc-oxide binary switching profile. Higher neural-weight "
            "precision is modeled through bit slicing across multiple cells."
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
        max-width: 1450px;
        padding-top: 2.35rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    .hero,
    .welcome-shell,
    .process-card,
    .material-card,
    .info-card,
    .callout,
    .warning-card,
    .success-card,
    .status-card,
    .why-card,
    .workflow-card {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }

    .hero {
        border-radius: 22px;
        padding: 1.9rem 2rem 1.65rem 2rem;
        margin-top: .35rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(110,110,110,.20);
        background:
            linear-gradient(
                135deg,
                rgba(82,69,210,.16),
                rgba(20,145,165,.08)
            );
    }

    .hero-kicker,
    .card-kicker {
        font-size: .73rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .10em;
        opacity: .58;
    }

    .hero-title {
        font-size: 2.4rem;
        font-weight: 900;
        line-height: 1.08;
    }

    .hero-subtitle {
        margin-top: .72rem;
        max-width: 900px;
        line-height: 1.55;
        opacity: .80;
    }

    .hero-flow {
        display: flex;
        align-items: center;
        gap: .45rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .hero-flow-step {
        border-radius: 999px;
        padding: .32rem .62rem;
        border: 1px solid rgba(110,110,110,.18);
        background: rgba(120,120,120,.04);
        font-size: .76rem;
        font-weight: 760;
        opacity: .82;
    }

    .hero-flow-arrow {
        opacity: .35;
        font-size: .86rem;
    }

    .welcome-shell {
        max-width: 1120px;
        margin: .5rem auto 1.5rem auto;
        border-radius: 24px;
        padding: 2rem 2.2rem;
        border: 1px solid rgba(110,110,110,.22);
        background:
            linear-gradient(
                145deg,
                rgba(82,69,210,.12),
                rgba(20,145,165,.05),
                rgba(120,120,120,.025)
            );
    }

    .welcome-title {
        font-size: 2.55rem;
        font-weight: 900;
        line-height: 1.05;
    }

    .welcome-text {
        margin-top: .8rem;
        line-height: 1.6;
        opacity: .80;
        max-width: 900px;
    }

    .workflow-card,
    .process-card,
    .info-card,
    .material-card,
    .why-card,
    .status-card {
        border-radius: 16px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(110,110,110,.20);
        background: rgba(120,120,120,.035);
    }

    .workflow-card {
        min-height: 178px;
        transition: border-color .16s ease, transform .16s ease;
    }

    .workflow-card:hover {
        border-color: rgba(100,120,235,.48);
        transform: translateY(-1px);
    }

    .workflow-title,
    .process-title,
    .material-name,
    .why-title {
        margin-top: .35rem;
        font-weight: 850;
        line-height: 1.3;
    }

    .workflow-text,
    .process-text,
    .material-description,
    .why-text {
        margin-top: .45rem;
        opacity: .72;
        font-size: .86rem;
        line-height: 1.48;
    }

    .material-card {
        min-height: 210px;
        margin-bottom: .8rem;
        background:
            linear-gradient(
                160deg,
                rgba(120,120,120,.05),
                rgba(60,130,170,.04)
            );
    }

    .material-symbol {
        font-size: 1.55rem;
        font-weight: 900;
    }

    .info-card {
        min-height: 135px;
    }

    .info-label,
    .status-label {
        font-size: .70rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: .07em;
        opacity: .55;
    }

    .info-value,
    .status-value {
        font-size: 1.18rem;
        font-weight: 850;
        margin-top: .32rem;
        line-height: 1.3;
    }

    .info-note {
        font-size: .79rem;
        opacity: .68;
        margin-top: .34rem;
        line-height: 1.4;
    }

    .callout,
    .warning-card,
    .success-card {
        border-radius: 15px;
        padding: 1rem 1.15rem;
        margin: .75rem 0 1rem 0;
        line-height: 1.52;
    }

    .callout {
        border-left: 4px solid rgba(70,90,210,.78);
        background: rgba(80,80,190,.07);
    }

    .warning-card {
        border: 1px solid rgba(210,140,30,.34);
        background: rgba(210,140,30,.07);
    }

    .success-card {
        border: 1px solid rgba(40,150,90,.34);
        background: rgba(40,150,90,.07);
    }

    .policy-badge {
        display: inline-block;
        padding: .32rem .58rem;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 850;
        border: 1px solid rgba(110,110,110,.20);
        background: rgba(120,120,120,.06);
    }

    .tiny-note {
        font-size: .78rem;
        opacity: .64;
        line-height: 1.45;
    }

    .stage-chip {
        display: inline-block;
        margin: .15rem .25rem .15rem 0;
        padding: .30rem .55rem;
        border-radius: 999px;
        border: 1px solid rgba(110,110,110,.20);
        background: rgba(120,120,120,.04);
        font-size: .77rem;
        font-weight: 750;
    }

    .step-row {
        display: flex;
        gap: .55rem;
        flex-wrap: wrap;
        margin-top: .8rem;
    }

    .step-pill {
        flex: 1 1 150px;
        min-width: 145px;
        text-align: center;
        border-radius: 13px;
        padding: .78rem .85rem;
        border: 1px solid rgba(110,110,110,.18);
        background: rgba(120,120,120,.035);
    }

    .step-pill b {
        display: block;
        margin-bottom: .2rem;
    }

    .onboarding-tip {
        margin-top: .45rem;
        margin-bottom: .35rem;
        font-size: .78rem;
        line-height: 1.4;
        opacity: .58;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(110,110,110,.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# BASIC HELPERS
# ============================================================

def render_html(markup: str):
    cleaned = textwrap.dedent(markup).strip()
    cleaned = " ".join(
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def clean_text(value, fallback="Not available"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    return text if text else fallback


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    converted = (
        series.astype(str)
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
        bad = series[converted.isna()].astype(str).unique().tolist()
        raise ValueError(f"Unexpected boolean values: {bad}")
    return converted.astype(bool)


def bool_value(value) -> bool:
    return bool(bool_series(pd.Series([value])).iloc[0])


def row_value(row, preferred, fallback=None, default=np.nan):
    if preferred in row.index and not pd.isna(row[preferred]):
        return row[preferred]
    if fallback and fallback in row.index and not pd.isna(row[fallback]):
        return row[fallback]
    return default


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def require_core_files():
    core = [
        SUMMARY_FILE,
        PREDICTIONS_FILE,
        DEVICE_FILE,
        TRACE_FILE,
        AUDIT_FILE,
        BASELINE_FILE,
    ]
    missing = [str(path) for path in core if not path.exists()]
    if missing:
        st.error("Some core generated research files are missing.")
        st.code("\n".join(missing))
        st.info("Run the analysis pipeline and reload the website.")
        st.stop()


def format_ratio(value):
    if pd.isna(value):
        return "Not available"
    value = float(value)
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:,.1f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def public_info(device_id: str):
    return MATERIAL_INFO.get(
        device_id,
        {
            "symbol": "Memristor",
            "name": f"Memristor Profile {device_id}",
            "short_name": device_id,
            "description": (
                "A literature-derived memristor profile used in the "
                "device-to-accelerator research pipeline."
            ),
        },
    )


def public_device_label(device_id: str):
    info = public_info(device_id)
    return f"{info['symbol']} — {info['short_name']}"


def public_mode_name(mode):
    mode = clean_text(mode, "").upper()
    names = {
        "DISCRETE_BINARY": "Binary switching",
        "DISCRETE_MULTILEVEL": "Discrete multilevel switching",
        "ANALOG": "Analog conductance",
        "GRADUAL": "Gradual conductance",
        "GRADUAL_MULTILEVEL": "Gradual multilevel switching",
    }
    return names.get(mode, mode.replace("_", " ").title())


def public_mapping_name(mode):
    mode = clean_text(mode, "").upper()
    names = {
        "DISCRETE_BINARY": "Bit-sliced binary differential mapping",
        "DISCRETE_MULTILEVEL": "Multilevel differential-pair mapping",
        "ANALOG": "Idealized analog differential-pair mapping",
        "GRADUAL": "Idealized gradual-conductance mapping",
        "GRADUAL_MULTILEVEL": "Idealized gradual-conductance mapping",
    }
    return names.get(mode, "Device-aware conductance mapping")


def config_label(crossbar, weight_bits, adc_bits):
    return (
        f"{int(crossbar)} × {int(crossbar)} | "
        f"{int(weight_bits)}-bit weights | {int(adc_bits)}-bit ADC"
    )


def find_candidate(candidate_df, crossbar, weight_bits, adc_bits):
    matches = candidate_df[
        (candidate_df["crossbar_size"] == int(crossbar))
        & (candidate_df["requested_weight_bits"] == int(weight_bits))
        & (candidate_df["adc_bits"] == int(adc_bits))
    ]
    return None if matches.empty else matches.iloc[0]


def render_info_card(label, value, note):
    render_html(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
            <div class="info-note">{note}</div>
        </div>
        """
    )


def render_workflow_card(kicker, title, text):
    render_html(
        f"""
        <div class="workflow-card">
            <div class="card-kicker">{kicker}</div>
            <div class="workflow-title">{title}</div>
            <div class="workflow-text">{text}</div>
        </div>
        """
    )


def render_material_card(info):
    render_html(
        f"""
        <div class="material-card">
            <div class="material-symbol">{info['symbol']}</div>
            <div class="material-name">{info['short_name']}</div>
            <div class="material-description">{info['description']}</div>
        </div>
        """
    )


def info_popover(title, beginner_text, researcher_text=None):
    text = researcher_text if (
        st.session_state.get("app_mode") == "Researcher"
        and researcher_text
    ) else beginner_text

    if hasattr(st, "popover"):
        with st.popover("ⓘ"):
            st.markdown(f"**{title}**")
            st.write(text)
    else:
        with st.expander(f"ⓘ {title}"):
            st.write(text)


def section_header(title, beginner_help=None, researcher_help=None, level=3):
    if beginner_help:
        left, right = st.columns([0.94, 0.06])
        with left:
            st.markdown("#" * level + f" {title}")
        with right:
            info_popover(title, beginner_help, researcher_help)
    else:
        st.markdown("#" * level + f" {title}")


def match_class(distance: float):
    if distance < 1e-6:
        return "EXACT_DESCRIPTOR_MATCH"
    if distance <= 0.15:
        return "VERY_CLOSE_EXISTING_PROFILE"
    if distance <= 0.50:
        return "CLOSE_EXISTING_PROFILE"
    if distance <= 1.00:
        return "MODERATE_GAP"
    return "LARGE_GAP"


def research_tier(distance: float):
    if distance < 0.15:
        return "EXISTING_OR_NEAR_DUPLICATE"
    if distance <= 0.50:
        return "TIER_A_INTERPOLATIVE_GAP"
    if distance <= 1.00:
        return "TIER_B_MODERATE_GAP"
    return "TIER_C_HIGH_EXTRAPOLATION"


def format_policy(policy: str):
    if policy == "VALIDATION_AWARE_COST":
        return "Validation-aware cost optimization"
    if policy == "ACCURACY_FIRST_EXTRAPOLATION_FALLBACK":
        return "Accuracy-first safety fallback"
    return clean_text(policy)


def human_mode(mode):
    return public_mode_name(mode)


def human_research_tier(value):
    names = {
        "TIER_A_INTERPOLATIVE_GAP": "Tier A — Interpolative gap",
        "TIER_B_MODERATE_GAP": "Tier B — Moderate gap",
        "TIER_C_HIGH_EXTRAPOLATION": "Tier C — High extrapolation",
        "EXISTING_OR_NEAR_DUPLICATE": "Existing / near duplicate",
    }
    return names.get(clean_text(value, ""), clean_text(value))


def human_match_class(value):
    names = {
        "EXACT_DESCRIPTOR_MATCH": "Exact descriptor match",
        "VERY_CLOSE_EXISTING_PROFILE": "Very-close existing profile",
        "CLOSE_EXISTING_PROFILE": "Close existing profile",
        "MODERATE_GAP": "Moderate gap",
        "LARGE_GAP": "Large gap",
    }
    return names.get(
        clean_text(value, ""),
        clean_text(value).replace("_", " ").title(),
    )


def display_target_states(mode, state_count):
    mode = clean_text(mode, "").upper()

    if mode in {"ANALOG", "GRADUAL", "GRADUAL_MULTILEVEL"}:
        try:
            numeric = int(float(state_count))
        except Exception:
            numeric = 0

        if numeric <= 0:
            return "Not fixed"

    try:
        numeric = int(float(state_count))
        return str(numeric) if numeric > 0 else "Not reported"
    except Exception:
        return clean_text(state_count, "Not reported")


def safe_numeric(df, column):
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def evidence_bucket(value):
    raw = clean_text(value, "").strip().upper().replace("-", "_").replace(" ", "_")
    if raw in {"REPORTED", "MEASURED", "EXPERIMENTAL", "DIRECTLY_REPORTED"}:
        return "Reported"
    if raw in {"DERIVED", "CALCULATED", "COMPUTED", "INFERRED"}:
        return "Derived"
    if raw in {"ASSUMED", "MODEL_ASSUMPTION", "SIMULATOR_ASSUMPTION"}:
        return "Assumed"
    if raw in {"MISSING", "NOT_REPORTED", "UNREPORTED", "NOT_AVAILABLE"}:
        return "Missing"
    return "Other"


def render_accuracy_validation_chart(
    predicted_accuracy,
    actual_accuracy,
    exhaustive_best_accuracy,
    baseline_accuracy,
):
    chart_df = pd.DataFrame(
        {
            "Result": [
                "AI predicted",
                "Recommended actual",
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
        lambda x: f"{x:.2f}%"
    )

    low = float(chart_df["Accuracy"].min())
    high = float(chart_df["Accuracy"].max())
    spread = max(high - low, 0.10)
    padding = max(0.12, spread * 0.35)

    spec = {
        "height": 300,
        "title": "Prediction → exhaustive validation",
        "layer": [
            {
                "mark": {"type": "line", "opacity": 0.35},
                "encoding": {
                    "x": {
                        "field": "Result",
                        "type": "nominal",
                        "sort": [
                            "AI predicted",
                            "Recommended actual",
                            "Exhaustive best",
                            "Software baseline",
                        ],
                        "axis": {"title": None, "labelAngle": 0},
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
                            "AI predicted",
                            "Recommended actual",
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
                            "AI predicted",
                            "Recommended actual",
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


def render_regret_threshold_chart(regret_pp, threshold_pp):
    regret_pp = float(regret_pp)
    threshold_pp = float(threshold_pp)
    passed = regret_pp <= threshold_pp

    axis_max = max(
        threshold_pp * 1.35,
        regret_pp * 1.20,
        threshold_pp + 0.10,
    )

    chart_df = pd.DataFrame(
        {
            "Metric": ["Regret"],
            "Value": [regret_pp],
            "Start": [0.0],
            "Label": [
                f"{regret_pp:.2f} pp — {'PASS' if passed else 'FAIL'}"
            ],
        }
    )

    spec = {
        "height": 165,
        "title": "Regret against near-optimal threshold",
        "layer": [
            {
                "mark": {"type": "bar", "size": 30},
                "encoding": {
                    "x": {
                        "field": "Value",
                        "type": "quantitative",
                        "scale": {"domain": [0, axis_max]},
                        "axis": {
                            "title": "Accuracy regret (percentage points)",
                            "format": ".2f",
                        },
                    },
                    "x2": {"field": "Start"},
                    "y": {
                        "field": "Metric",
                        "type": "nominal",
                        "axis": {"title": None},
                    },
                },
            },
            {
                "data": {
                    "values": [{"Threshold": threshold_pp}]
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
                        "scale": {"domain": [0, axis_max]},
                    },
                },
            },
            {
                "mark": {
                    "type": "text",
                    "dx": 8,
                    "dy": -24,
                    "align": "left",
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": {
                        "field": "Value",
                        "type": "quantitative",
                        "scale": {"domain": [0, axis_max]},
                    },
                    "y": {"field": "Metric", "type": "nominal"},
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

    low = float(
        min(
            chart_df["Predicted Accuracy"].min(),
            chart_df["Actual Accuracy"].min(),
        )
    )
    high = float(
        max(
            chart_df["Predicted Accuracy"].max(),
            chart_df["Actual Accuracy"].max(),
        )
    )
    pad = max(0.05, (high - low) * 0.04)

    rec_point = {
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
            "Predicted Accuracy": low - pad,
            "Actual Accuracy": low - pad,
        },
        {
            "Predicted Accuracy": high + pad,
            "Actual Accuracy": high + pad,
        },
    ]

    enc_x = {
        "field": "Predicted Accuracy",
        "type": "quantitative",
        "scale": {"domain": [low - pad, high + pad], "zero": False},
        "axis": {"title": "Predicted accuracy (%)", "format": ".2f"},
    }

    enc_y = {
        "field": "Actual Accuracy",
        "type": "quantitative",
        "scale": {"domain": [low - pad, high + pad], "zero": False},
        "axis": {"title": "Actual simulated accuracy (%)", "format": ".2f"},
    }

    spec = {
        "height": 410,
        "title": (
            f"Predicted vs actual across {len(chart_df)} "
            "held-out configurations"
        ),
        "layer": [
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 48,
                    "opacity": 0.45,
                },
                "encoding": {
                    "x": enc_x,
                    "y": enc_y,
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
                    "opacity": 0.7,
                },
                "encoding": {"x": enc_x, "y": enc_y},
            },
            {
                "data": {"values": [rec_point]},
                "mark": {
                    "type": "point",
                    "filled": True,
                    "shape": "diamond",
                    "size": 230,
                },
                "encoding": {"x": enc_x, "y": enc_y},
            },
            {
                "data": {"values": [rec_point]},
                "mark": {
                    "type": "text",
                    "dx": 10,
                    "dy": -12,
                    "align": "left",
                    "fontWeight": "bold",
                },
                "encoding": {
                    "x": enc_x,
                    "y": enc_y,
                    "text": {"field": "Label"},
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
                "encoding": {"x": enc_x, "y": enc_y},
            },
        ],
    }

    st.vega_lite_chart(
        chart_df,
        spec,
        use_container_width=True,
    )


def render_regret_comparison_chart(summary_df):
    rows = []

    for _, r in summary_df.iterrows():
        device = str(r["held_out_device"])

        if "baseline_regret_pp" in r.index:
            rows.append(
                {
                    "Device": device,
                    "Policy": "Original baseline",
                    "Regret": float(r["baseline_regret_pp"]),
                }
            )

        if "guarded_regret_pp" in r.index:
            rows.append(
                {
                    "Device": device,
                    "Policy": "Support-gated",
                    "Regret": float(r["guarded_regret_pp"]),
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        return

    spec = {
        "height": 330,
        "title": "Held-out-device regret: original vs support-gated policy",
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "Device",
                "type": "nominal",
                "axis": {"title": None, "labelAngle": -35},
            },
            "xOffset": {"field": "Policy"},
            "color": {
                "field": "Policy",
                "type": "nominal",
                "legend": {"title": None},
            },
            "y": {
                "field": "Regret",
                "type": "quantitative",
                "axis": {
                    "title": "Regret (percentage points)",
                    "format": ".2f",
                },
            },
            "tooltip": [
                {"field": "Device", "type": "nominal"},
                {"field": "Policy", "type": "nominal"},
                {
                    "field": "Regret",
                    "type": "quantitative",
                    "format": ".3f",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


def render_sensitivity_chart(selected_sens):
    if selected_sens.empty:
        return

    df = selected_sens[
        ["factor", "accuracy_effect_range_pp"]
    ].copy()

    df["Factor"] = (
        df["factor"]
        .astype(str)
        .str.replace("_", " ", regex=False)
        .str.title()
    )

    spec = {
        "height": 250,
        "title": "Selected-device empirical sensitivity",
        "mark": {"type": "bar"},
        "encoding": {
            "x": {
                "field": "accuracy_effect_range_pp",
                "type": "quantitative",
                "axis": {
                    "title": "Grouped mean accuracy range (pp)",
                    "format": ".2f",
                },
            },
            "y": {
                "field": "Factor",
                "type": "nominal",
                "sort": "-x",
                "axis": {"title": None},
            },
            "tooltip": [
                {"field": "Factor", "type": "nominal"},
                {
                    "field": "accuracy_effect_range_pp",
                    "type": "quantitative",
                    "format": ".3f",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


def render_pareto_chart(
    candidate_df,
    pareto_df,
    recommended_row,
):
    if candidate_df.empty:
        return

    all_points = candidate_df[
        [
            "relative_hardware_cost_proxy",
            "accuracy",
        ]
    ].copy()

    all_points = all_points.rename(
        columns={
            "relative_hardware_cost_proxy": "Cost",
            "accuracy": "Accuracy",
        }
    )

    pareto_points = pd.DataFrame()

    if not pareto_df.empty:
        pareto_points = pareto_df[
            [
                "relative_hardware_cost_proxy",
                "accuracy",
            ]
        ].copy()

        pareto_points = pareto_points.rename(
            columns={
                "relative_hardware_cost_proxy": "Cost",
                "accuracy": "Accuracy",
            }
        ).sort_values("Cost")

    rec = {
        "Cost": float(
            recommended_row["relative_hardware_cost_proxy"]
        ),
        "Accuracy": float(
            recommended_row["accuracy"]
        ),
        "Label": "AI recommendation",
    }

    layers = [
        {
            "data": {"values": all_points.to_dict("records")},
            "mark": {
                "type": "point",
                "filled": True,
                "size": 34,
                "opacity": 0.22,
            },
            "encoding": {
                "x": {
                    "field": "Cost",
                    "type": "quantitative",
                    "scale": {"type": "log"},
                    "axis": {
                        "title": "Relative hardware-cost proxy (log scale)"
                    },
                },
                "y": {
                    "field": "Accuracy",
                    "type": "quantitative",
                    "axis": {"title": "Simulated accuracy (%)"},
                },
            },
        }
    ]

    if not pareto_points.empty:
        layers.append(
            {
                "data": {
                    "values": pareto_points.to_dict("records")
                },
                "mark": {
                    "type": "line",
                    "point": True,
                    "size": 2,
                },
                "encoding": {
                    "x": {
                        "field": "Cost",
                        "type": "quantitative",
                        "scale": {"type": "log"},
                    },
                    "y": {
                        "field": "Accuracy",
                        "type": "quantitative",
                    },
                },
            }
        )

    layers += [
        {
            "data": {"values": [rec]},
            "mark": {
                "type": "point",
                "filled": True,
                "shape": "diamond",
                "size": 230,
            },
            "encoding": {
                "x": {
                    "field": "Cost",
                    "type": "quantitative",
                    "scale": {"type": "log"},
                },
                "y": {
                    "field": "Accuracy",
                    "type": "quantitative",
                },
            },
        },
        {
            "data": {"values": [rec]},
            "mark": {
                "type": "text",
                "dx": 10,
                "dy": -10,
                "align": "left",
                "fontWeight": "bold",
            },
            "encoding": {
                "x": {
                    "field": "Cost",
                    "type": "quantitative",
                    "scale": {"type": "log"},
                },
                "y": {
                    "field": "Accuracy",
                    "type": "quantitative",
                },
                "text": {"field": "Label"},
            },
        },
    ]

    spec = {
        "height": 360,
        "title": "Accuracy vs relative hardware-cost proxy",
        "layer": layers,
    }

    st.vega_lite_chart(
        all_points,
        spec,
        use_container_width=True,
    )


def provenance_complete_percentage(trace_df):
    required = [
        "source_title",
        "doi",
        "page",
        "figure_or_table",
        "source_note",
    ]
    if any(c not in trace_df.columns for c in required):
        return np.nan
    valid = pd.DataFrame(index=trace_df.index)
    for c in required:
        valid[c] = (
            trace_df[c].notna()
            & trace_df[c].astype(str).str.strip().ne("")
        )
    return float(100 * valid.all(axis=1).mean())


# ============================================================
# LOAD DATA
# ============================================================

require_core_files()


@st.cache_data
def load_all_data():
    return {
        "summary": pd.read_csv(SUMMARY_FILE),
        "predictions": pd.read_csv(PREDICTIONS_FILE),
        "devices": pd.read_csv(DEVICE_FILE),
        "trace": pd.read_csv(TRACE_FILE),
        "audit": pd.read_csv(AUDIT_FILE),
        "study_summary": read_csv_optional(STUDY_SUMMARY_FILE),
        "family_summary": read_csv_optional(FAMILY_SUMMARY_FILE),
        "validation_overview": read_csv_optional(VALIDATION_OVERVIEW_FILE),
        "reliability": read_csv_optional(RELIABILITY_FILE),
        "ml": read_csv_optional(ML_FILE),
        "evidence_coverage": read_csv_optional(EVIDENCE_COVERAGE_FILE),
        "evidence_summary": read_csv_optional(EVIDENCE_SUMMARY_FILE),
        "sensitivity_by_device": read_csv_optional(SENSITIVITY_BY_DEVICE_FILE),
        "sensitivity_summary": read_csv_optional(SENSITIVITY_SUMMARY_FILE),
        "ood_device": read_csv_optional(OOD_DEVICE_FILE),
        "ood_overview": read_csv_optional(OOD_OVERVIEW_FILE),
        "nonideality_detail": read_csv_optional(NONIDEALITY_DETAIL_FILE),
        "nonideality_summary": read_csv_optional(NONIDEALITY_SUMMARY_FILE),
        "pareto_front": read_csv_optional(PARETO_FRONT_FILE),
        "pareto_summary": read_csv_optional(PARETO_SUMMARY_FILE),
        "pareto_overview": read_csv_optional(PARETO_OVERVIEW_FILE),
        "reverse_candidates": read_csv_optional(REVERSE_CANDIDATES_FILE),
        "reverse_requirements": read_csv_optional(REVERSE_REQUIREMENTS_FILE),
        "reverse_matches": read_csv_optional(REVERSE_MATCHES_FILE),
        "reverse_match_summary": read_csv_optional(REVERSE_MATCH_SUMMARY_FILE),
        "future_targets": read_csv_optional(FUTURE_TARGETS_FILE),
        "future_target_summary": read_csv_optional(FUTURE_TARGET_SUMMARY_FILE),
    }


DATA = load_all_data()

summary = DATA["summary"]
predictions = DATA["predictions"]
devices = DATA["devices"]
trace = DATA["trace"]
audit = DATA["audit"]
ml_df = DATA["ml"]

software_baseline_accuracy = float(
    BASELINE_FILE.read_text(encoding="utf-8").strip()
)


# ============================================================
# PROJECT METRICS
# ============================================================

active_device_ids = sorted(
    summary["held_out_device"].astype(str).unique().tolist()
)

device_count = len(active_device_ids)

study_count = (
    int(summary["held_out_study"].nunique())
    if "held_out_study" in summary.columns
    else np.nan
)

family_count = (
    int(summary["held_out_family"].nunique())
    if "held_out_family" in summary.columns
    else np.nan
)

configurations_per_device = int(summary["held_out_rows"].iloc[0])
total_simulation_rows = int(len(predictions))

primary_success_col = (
    "guarded_near_optimal_success"
    if "guarded_near_optimal_success" in summary.columns
    else "baseline_near_optimal_success"
)

primary_regret_col = (
    "guarded_regret_pp"
    if "guarded_regret_pp" in summary.columns
    else "baseline_regret_pp"
)

primary_success = bool_series(summary[primary_success_col])
primary_success_count = int(primary_success.sum())
primary_success_rate = float(100 * primary_success.mean())
primary_mean_regret = float(summary[primary_regret_col].mean())

guarded_fallback_rate = (
    float(
        100
        * (
            summary["guarded_policy"]
            == "ACCURACY_FIRST_EXTRAPOLATION_FALLBACK"
        ).mean()
    )
    if "guarded_policy" in summary.columns
    else np.nan
)

literature_device_count = int(devices["device_id"].nunique())
provenance_completeness = provenance_complete_percentage(trace)


# ============================================================
# SESSION / FIRST PAGE
# ============================================================

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Beginner"

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


def open_app():
    st.session_state.show_welcome = False
    st.rerun()


def open_welcome():
    st.session_state.show_welcome = True
    st.rerun()


def enter_page(page_name: str):
    st.session_state.current_page = page_name
    st.session_state.show_welcome = False
    st.rerun()


# ============================================================
# WELCOME / QUICK START
# ============================================================

if st.session_state.show_welcome:
    render_html(
        """
        <div class="welcome-shell">
            <div class="card-kicker">Quick Start</div>
            <div class="welcome-title">NanoMemristor AI Designer</div>
            <div class="welcome-text">
                Connect literature-derived memristor behavior to simulated
                AI-accelerator design.<br><br>
                Choose a workflow below, or continue to the main dashboard
                and explore at your own pace.
            </div>
        </div>
        """
    )

    section_header(
        "Choose a workflow",
        (
            "Each workflow below can take you directly to that part of the tool. "
            "You can also enter the main Home page using Continue."
        ),
        level=2,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_workflow_card(
            "Workflow 1",
            "Known Memristor → Accelerator",
            (
                "Choose a literature-derived memristor and receive a "
                "support-gated accelerator recommendation."
            ),
        )
        if st.button(
            "Start Forward Design →",
            key="welcome_forward",
            use_container_width=True,
        ):
            enter_page("Forward Design")

    with c2:
        render_workflow_card(
            "Workflow 2",
            "My Device → Accelerator",
            (
                "Enter ON/OFF ratio, conductance behavior and state-count "
                "information for your own device descriptor."
            ),
        )
        if st.button(
            "Start Custom Device →",
            key="welcome_custom",
            use_container_width=True,
        ):
            enter_page("Custom Device")

    with c3:
        render_workflow_card(
            "Workflow 3",
            "Desired Performance → Device Properties",
            (
                "Use reverse design to search for device-property requirements "
                "that satisfy an accelerator target."
            ),
        )
        if st.button(
            "Start Reverse Design →",
            key="welcome_reverse",
            use_container_width=True,
        ):
            enter_page("Reverse Design")

    with c4:
        render_workflow_card(
            "Workflow 4",
            "Future Research Targets",
            (
                "Explore useful descriptor regions not already closely represented "
                "in the current literature dataset."
            ),
        )
        if st.button(
            "Start Research Targets →",
            key="welcome_targets",
            use_container_width=True,
        ):
            enter_page("Research Targets")

    st.write("")

    researcher_mode = st.toggle(
        "Researcher mode",
        value=(st.session_state.app_mode == "Researcher"),
        help=(
            "Off = Beginner mode: interpretation first and fewer raw details. "
            "On = Researcher mode: validation, provenance, diagnostics and raw tables."
        ),
    )
    st.session_state.app_mode = (
        "Researcher" if researcher_mode else "Beginner"
    )

    render_html(
        """
        <div class="onboarding-tip">
            ⓘ Tip: inside the application, use the information icons for short,
            contextual explanations. You can reopen this Quick Start page from
            the sidebar at any time.
        </div>
        """
    )

    spacer_left, action_col, spacer_right = st.columns([1.2, 1, 1.2])

    with action_col:
        if st.button(
            "Continue to Home →",
            type="primary",
            use_container_width=True,
        ):
            enter_page("Home")

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚡ NanoMemristor AI Designer")

if st.sidebar.button(
    "🏠 Quick Start / How to Use",
    use_container_width=True,
):
    open_welcome()

researcher_mode = st.sidebar.toggle(
    "Researcher mode",
    value=(st.session_state.app_mode == "Researcher"),
    help=(
        "Off = Beginner mode. On = Researcher mode with provenance, "
        "validation diagnostics and detailed tables."
    ),
)
st.session_state.app_mode = (
    "Researcher" if researcher_mode else "Beginner"
)

st.sidebar.markdown("---")

public_device_options = {
    public_device_label(device_id): device_id
    for device_id in active_device_ids
}

labels = list(public_device_options)
default_label_index = 0

for i, label in enumerate(labels):
    if public_device_options[label] == DEFAULT_DEVICE_ID:
        default_label_index = i
        break

selected_label = st.sidebar.selectbox(
    "Literature device",
    labels,
    index=default_label_index,
    help=(
        "Global device filter used by Forward Design, Why Nano?, "
        "Research Evidence and Sources & Limitations."
    ),
)

selected_device = public_device_options[selected_label]
selected_material = public_info(selected_device)

st.sidebar.caption(
    f"{device_count} profiles • {study_count} source studies • "
    f"{family_count} technology families"
)

st.sidebar.markdown("---")
st.sidebar.caption("NAVIGATION")

for nav_page in PAGES:
    is_active = nav_page == st.session_state.current_page

    if st.sidebar.button(
        nav_page,
        key=f"nav_{nav_page}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.session_state.current_page = nav_page
        st.rerun()

page = st.session_state.current_page

selected_summary = summary[
    summary["held_out_device"] == selected_device
].iloc[0]

candidate_df = predictions[
    predictions["held_out_device"] == selected_device
].copy()

profile_matches = devices[devices["device_id"] == selected_device]
profile = None if profile_matches.empty else profile_matches.iloc[0]

st.sidebar.caption(
    "ⓘ Use contextual help inside the pages whenever a term is unclear."
)


# ============================================================
# SELECTED DEVICE DESCRIPTORS + FINAL RECOMMENDATION
# ============================================================

device_ratio = float(candidate_df["device_on_off_ratio"].iloc[0])
conductance_mode = clean_text(
    candidate_df["device_conductance_mode"].iloc[0]
)
state_count_available = int(
    candidate_df["state_count_available"].iloc[0]
)
physical_state_count = int(
    candidate_df["physical_state_count"].iloc[0]
)

rec_crossbar = int(
    row_value(
        selected_summary,
        "guarded_crossbar",
        "baseline_crossbar",
    )
)
rec_weight = int(
    row_value(
        selected_summary,
        "guarded_weight_bits",
        "baseline_weight_bits",
    )
)
rec_adc = int(
    row_value(
        selected_summary,
        "guarded_adc_bits",
        "baseline_adc_bits",
    )
)

rec_predicted = float(
    row_value(
        selected_summary,
        "guarded_predicted_accuracy",
        "baseline_predicted_accuracy",
    )
)
rec_actual = float(
    row_value(
        selected_summary,
        "guarded_actual_accuracy",
        "baseline_actual_accuracy",
    )
)
rec_regret = float(
    row_value(
        selected_summary,
        "guarded_regret_pp",
        "baseline_regret_pp",
    )
)
rec_success = bool(
    row_value(
        selected_summary,
        "guarded_near_optimal_success",
        "baseline_near_optimal_success",
    )
)
rec_policy = clean_text(
    row_value(
        selected_summary,
        "guarded_policy",
        default="LEGACY_BASELINE",
    )
)
rec_support_reason = clean_text(
    row_value(
        selected_summary,
        "guarded_support_reason",
        default="Not available",
    )
)

recommended_row = find_candidate(
    candidate_df,
    rec_crossbar,
    rec_weight,
    rec_adc,
)

if recommended_row is None:
    st.error(
        "The saved recommendation could not be found inside "
        "the held-out-device candidate table."
    )
    st.stop()

rec_levels = int(recommended_row["effective_weight_levels"])
rec_cells = int(recommended_row["physical_cells_per_weight"])
rec_cost = float(recommended_row["relative_hardware_cost_proxy"])

raw_best_row = candidate_df.sort_values(
    [
        "accuracy",
        "relative_hardware_cost_proxy",
        "requested_weight_bits",
        "adc_bits",
    ],
    ascending=[False, True, True, True],
).iloc[0]

raw_best_accuracy = float(raw_best_row["accuracy"])


# ============================================================
# PAGE HERO
# ============================================================

render_html(
    """
    <div class="hero">
        <div class="hero-kicker">Nanotechnology + AI accelerator co-design</div>
        <div class="hero-title">NanoMemristor AI Designer</div>
        <div class="hero-subtitle">
            From experimentally reported memristor behavior to AI-assisted
            accelerator recommendations, reverse design and research-gap discovery.
        </div>
        <div class="hero-flow">
            <span class="hero-flow-step">Literature evidence</span>
            <span class="hero-flow-arrow">→</span>
            <span class="hero-flow-step">Device descriptors</span>
            <span class="hero-flow-arrow">→</span>
            <span class="hero-flow-step">Simulation</span>
            <span class="hero-flow-arrow">→</span>
            <span class="hero-flow-step">Study-aware AI</span>
            <span class="hero-flow-arrow">→</span>
            <span class="hero-flow-step">Design decision</span>
        </div>
    </div>
    """
)


# ============================================================
# HOME
# ============================================================

if page == "Home":
    section_header(
        "What is this project?",
        (
            "The tool links memristor device behavior reported in the literature "
            "to a simulated AI accelerator. It then asks whether knowledge learned "
            "from other devices can recommend a useful accelerator design for a "
            "device that was excluded from model training."
        ),
        (
            "The current system is a cross-layer research prototype. It uses "
            "study-aware held-out validation and explicitly distinguishes "
            "literature evidence, simulator assumptions and heuristic reliability "
            "signals."
        ),
        level=2,
    )

    left, right = st.columns([1.2, 0.8])

    with left:
        st.write(
            """
            Instead of treating **nanotechnology** and **AI** as separate topics,
            this project connects them directly:
            """
        )

        render_html(
            """
            <div class="step-row">
                <div class="step-pill"><b>1. Literature</b>Experimental memristor evidence</div>
                <div class="step-pill"><b>2. Nano Device</b>Electrical device descriptors</div>
                <div class="step-pill"><b>3. Accelerator</b>245 hardware configurations</div>
                <div class="step-pill"><b>4. AI</b>Cross-device prediction</div>
                <div class="step-pill"><b>5. Validation</b>Held-out ground truth</div>
            </div>
            """
        )

    with right:
        render_html(
            f"""
            <div class="callout">
                <strong>Current evidence base</strong><br><br>
                <strong>{device_count}</strong> simulation-ready device profiles<br>
                <strong>{study_count}</strong> independent primary studies<br>
                <strong>{family_count}</strong> technology families<br>
                <strong>{total_simulation_rows:,}</strong> device-configuration
                simulation rows
            </div>
            """
        )

    st.divider()

    section_header(
        "Current headline result",
        (
            "A recommendation is counted as near-optimal when the actual simulated "
            "accuracy is within 0.5 percentage points of the best exhaustive result."
        ),
        (
            "The primary result now uses the support-gated policy. Cost optimization "
            "is allowed only in a sufficiently supported same-mode descriptor region; "
            "otherwise the policy falls back to accuracy-first selection."
        ),
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Near-optimal success", f"{primary_success_count}/{device_count}")
    m2.metric("Success rate", f"{primary_success_rate:.1f}%")
    m3.metric("Mean regret", f"{primary_mean_regret:.3f} pp")
    m4.metric(
        "Safety fallback rate",
        (
            f"{guarded_fallback_rate:.1f}%"
            if np.isfinite(guarded_fallback_rate)
            else "N/A"
        ),
    )

    render_html(
        """
        <div class="warning-card">
            <strong>Important:</strong> the 100% near-optimal result is a pilot
            result on the current small literature-derived dataset. The high
            safety-fallback rate means the framework often refuses aggressive
            cost optimization because descriptor support is still sparse.
        </div>
        """
    )

    st.divider()

    section_header("Explore the current memristor profiles")

    cols_per_row = 4

    for start in range(0, len(active_device_ids), cols_per_row):
        row_ids = active_device_ids[start:start + cols_per_row]
        cols = st.columns(cols_per_row)

        for j, device_id in enumerate(row_ids):
            with cols[j]:
                render_material_card(public_info(device_id))

    st.caption(
        "TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt are three device "
        "profiles from one shared experimental study, not three independent studies."
    )


# ============================================================
# FORWARD DESIGN
# ============================================================

elif page == "Forward Design":
    section_header(
        "Forward Design",
        (
            "Start with a memristor device and ask: which accelerator "
            "configuration should I use?"
        ),
        (
            "The final policy combines a study-aware Random-Forest prediction, "
            "a historical validation penalty and a descriptor-support gate."
        ),
        level=2,
    )

    st.markdown(
        f"## {selected_material['symbol']} — {selected_material['name']}"
    )
    st.write(selected_material["description"])

    section_header(
        "1. Device behavior",
        (
            "These descriptors connect the nanodevice to the accelerator model."
        ),
        (
            "ON/OFF ratio, conductance mode and state capability enter the "
            "cross-device model together with accelerator configuration features."
        ),
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_info_card(
            "Material / profile",
            selected_material["symbol"],
            "Selected literature profile",
        )

    with c2:
        render_info_card(
            "Conductance behavior",
            public_mode_name(conductance_mode),
            "Electrical switching mode",
        )

    with c3:
        render_info_card(
            "ON/OFF ratio",
            format_ratio(device_ratio),
            "Conductance contrast",
        )

    with c4:
        if state_count_available:
            state_value = str(physical_state_count)
            state_note = "Fixed state count represented in the dataset."
        else:
            state_value = "Not fixed / unknown"
            state_note = "Missing fixed count is kept explicitly unknown."
        render_info_card(
            "Physical states",
            state_value,
            "Literature state-count status",
        )

    if profile is not None and st.session_state.app_mode == "Researcher":
        with st.expander("Device profile details"):
            detail_cols = [
                c
                for c in [
                    "device_id",
                    "technology_family",
                    "device_stack",
                    "active_material",
                    "source_title",
                    "doi",
                    "year",
                    "parameter_source",
                ]
                if c in profile.index
            ]
            detail_df = pd.DataFrame(
                [{"Field": c, "Value": profile[c]} for c in detail_cols]
            )
            st.dataframe(detail_df, use_container_width=True, hide_index=True)

        selected_audit_forward = audit[
            audit["device_id"] == selected_device
        ].copy()

        if not selected_audit_forward.empty and "value_type" in selected_audit_forward.columns:
            selected_audit_forward["Evidence"] = (
                selected_audit_forward["value_type"].map(evidence_bucket)
            )

            section_header(
                "Evidence status",
                (
                    "The simulator distinguishes reported, derived, assumed "
                    "and missing information rather than presenting every value "
                    "as equally experimental."
                ),
            )

            ec1, ec2, ec3, ec4 = st.columns(4)

            for col, label in zip(
                [ec1, ec2, ec3, ec4],
                ["Reported", "Derived", "Assumed", "Missing"],
            ):
                col.metric(
                    label,
                    int(
                        (
                            selected_audit_forward["Evidence"]
                            == label
                        ).sum()
                    ),
                )

            with st.expander("View selected-device evidence audit"):
                audit_cols = [
                    c
                    for c in [
                        "property_name",
                        "simulator_value",
                        "trace_value",
                        "value_type",
                        "status",
                        "message",
                    ]
                    if c in selected_audit_forward.columns
                ]

                st.dataframe(
                    selected_audit_forward[audit_cols],
                    use_container_width=True,
                    hide_index=True,
                )

    st.divider()

    section_header(
        "2. Support-gated AI recommendation",
        (
            "The system first checks whether aggressive cost optimization is "
            "supported. If not, it chooses the highest predicted-accuracy region "
            "instead of trusting a cheap but risky configuration."
        ),
        (
            "The gate requires at least two same-conductance-mode training profiles "
            "and the held-out device's log10(ON/OFF) to lie inside that training "
            "envelope. This is a transparent safeguard, not a calibrated OOD probability."
        ),
    )

    policy_label = format_policy(rec_policy)

    render_html(
        f"""
        <div class="callout">
            <span class="policy-badge">{policy_label}</span><br><br>
            <strong>Support decision:</strong> {rec_support_reason}
        </div>
        """
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Crossbar", f"{rec_crossbar} × {rec_crossbar}")
    r2.metric("Weight precision", f"{rec_weight}-bit")
    r3.metric("ADC precision", f"{rec_adc}-bit")
    r4.metric("Effective levels", rec_levels)
    r5.metric("Cells / weight", rec_cells)

    p1, p2, p3 = st.columns(3)
    p1.metric("Predicted accuracy", f"{rec_predicted:.2f}%")
    p2.metric("Actual simulated accuracy", f"{rec_actual:.2f}%")
    p3.metric("Regret", f"{rec_regret:.2f} pp")

    if rec_success:
        st.success(
            f"PASS — this recommendation is within "
            f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp of the exhaustive raw-best accuracy."
        )
    else:
        st.error(
            f"FAIL — regret exceeds the "
            f"{NEAR_OPTIMAL_TOLERANCE_PP:.1f} pp near-optimal threshold."
        )

    section_header(
        "Why this design?",
        (
            "The explanation below separates device constraints from the AI's "
            "selection rule."
        ),
        (
            "The Random Forest does not establish physical causality. "
            "Sensitivity analysis is empirical over the simulated grid."
        ),
    )

    w1, w2 = st.columns(2)

    with w1:
        render_workflow_card(
            "Device constraint",
            f"{public_mode_name(conductance_mode)}",
            (
                f"The device mapping provides {rec_levels} effective signed "
                f"weight levels and uses {rec_cells} physical cells per weight "
                f"for this configuration."
            ),
        )

    with w2:
        render_workflow_card(
            "Recommendation policy",
            policy_label,
            (
                "The policy either searches a validation-aware near-optimal region "
                "for lower-cost designs or falls back to accuracy-first when "
                "descriptor support is insufficient."
            ),
        )

    if st.session_state.app_mode == "Researcher":
        st.divider()
        section_header(
            "Researcher diagnostics",
            "Additional diagnostics used to understand the recommendation.",
        )

        diag1, diag2, diag3 = st.columns(3)

        if not DATA["evidence_coverage"].empty:
            ev = DATA["evidence_coverage"][
                DATA["evidence_coverage"]["device_id"] == selected_device
            ]
            if not ev.empty:
                e = ev.iloc[0]
                with diag1:
                    render_info_card(
                        "Evidence coverage",
                        (
                            f"{100*float(e['evidence_coverage_fraction']):.0f}%"
                            if "evidence_coverage_fraction" in e.index
                            else "N/A"
                        ),
                        clean_text(e.get("scenario_status", "Unknown")),
                    )

        if not DATA["ood_device"].empty:
            od = DATA["ood_device"][
                DATA["ood_device"]["held_out_device"] == selected_device
            ]
            if not od.empty:
                o = od.iloc[0]
                with diag2:
                    render_info_card(
                        "Relative OOD score",
                        f"{float(o['ood_combined_score']):.3f}",
                        "Relative descriptor-distance heuristic; not a probability.",
                    )

        if not DATA["sensitivity_by_device"].empty:
            s = DATA["sensitivity_by_device"][
                DATA["sensitivity_by_device"]["device_id"] == selected_device
            ]
            if not s.empty:
                dom = s.sort_values(
                    "accuracy_sensitivity_rank_within_device"
                ).iloc[0]
                with diag3:
                    render_info_card(
                        "Dominant grid factor",
                        clean_text(dom["factor"]).replace("_", " ").title(),
                        (
                            f"Grouped mean accuracy range: "
                            f"{float(dom['accuracy_effect_range_pp']):.2f} pp"
                        ),
                    )

        if not DATA["pareto_summary"].empty:
            ps = DATA["pareto_summary"][
                DATA["pareto_summary"]["device_id"] == selected_device
            ]
            if not ps.empty:
                st.markdown("#### Pareto trade-off reference")
                p = ps.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Pareto points", int(p["pareto_points"]))
                c2.metric(
                    "Best simulated accuracy",
                    f"{float(p['best_accuracy']):.2f}%",
                )
                c3.metric(
                    "Oracle 0.5-pp proxy cost reduction",
                    f"{float(p['within_0p5pp_cost_reduction_vs_best_pct']):.1f}%",
                )
                st.caption(
                    "The 0.5-pp Pareto reference uses true simulated accuracy and is "
                    "therefore an oracle/reference benchmark, not an unseen-device AI recommendation."
                )

    st.divider()

    section_header(
        "3. Exhaustive validation",
        (
            "Because this is a research evaluation, the held-out device already has "
            "245 simulated configurations. They are revealed only after the AI "
            "recommendation so we can measure regret."
        ),
    )

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("AI predicted", f"{rec_predicted:.2f}%")
    v2.metric("Recommended actual", f"{rec_actual:.2f}%")
    v3.metric("Exhaustive raw best", f"{raw_best_accuracy:.2f}%")
    v4.metric("Software baseline", f"{software_baseline_accuracy:.2f}%")

    visual_left, visual_right = st.columns([1.15, 0.85])

    with visual_left:
        render_accuracy_validation_chart(
            rec_predicted,
            rec_actual,
            raw_best_accuracy,
            software_baseline_accuracy,
        )

        st.caption(
            "The accuracy axis is zoomed so small differences are visible. "
            "Use the numeric labels when judging the magnitude."
        )

    with visual_right:
        render_regret_threshold_chart(
            rec_regret,
            NEAR_OPTIMAL_TOLERANCE_PP,
        )

        st.caption(
            f"PASS means regret ≤ {NEAR_OPTIMAL_TOLERANCE_PP:.2f} pp."
        )

    section_header(
        "Top predicted candidates",
        (
            "The model predicts all 245 held-out configurations before "
            "the final policy chooses one of them."
        ),
    )

    top_five = (
        candidate_df
        .sort_values(
            [
                "predicted_accuracy",
                "relative_hardware_cost_proxy",
            ],
            ascending=[False, True],
        )
        .head(5)
        .copy()
    )

    top_display_cols = [
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
        "predicted_accuracy",
        "accuracy",
        "relative_hardware_cost_proxy",
    ]

    if "validation_adjusted_score" in top_five.columns:
        top_display_cols.insert(4, "validation_adjusted_score")

    st.dataframe(
        top_five[top_display_cols].round(3),
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.app_mode == "Researcher":
        section_header(
            "Predicted vs actual across the full held-out design space",
            (
                "Each point is one accelerator configuration. "
                "The diagonal is perfect prediction."
            ),
        )

        render_predicted_vs_actual_chart(
            candidate_df,
            recommended_row,
            raw_best_row,
        )

        st.caption(
            "The diamond is the final AI recommendation. "
            "The triangle is the exhaustive raw-accuracy optimum."
        )

        true_cost_crossbar = int(selected_summary["true_cost_aware_crossbar"])
        true_cost_weight = int(selected_summary["true_cost_aware_weight_bits"])
        true_cost_adc = int(selected_summary["true_cost_aware_adc_bits"])
        true_cost_accuracy = float(selected_summary["true_cost_aware_accuracy"])

        comparison_df = pd.DataFrame(
            [
                {
                    "Design": "Support-gated AI recommendation",
                    "Configuration": config_label(
                        rec_crossbar,
                        rec_weight,
                        rec_adc,
                    ),
                    "Actual Accuracy": f"{rec_actual:.2f}%",
                    "Role": "Unseen-device recommendation",
                },
                {
                    "Design": "Raw accuracy best",
                    "Configuration": config_label(
                        raw_best_row["crossbar_size"],
                        raw_best_row["requested_weight_bits"],
                        raw_best_row["adc_bits"],
                    ),
                    "Actual Accuracy": f"{raw_best_accuracy:.2f}%",
                    "Role": "Exhaustive reference",
                },
                {
                    "Design": "Oracle cost-aware optimum",
                    "Configuration": config_label(
                        true_cost_crossbar,
                        true_cost_weight,
                        true_cost_adc,
                    ),
                    "Actual Accuracy": f"{true_cost_accuracy:.2f}%",
                    "Role": "Uses true held-out accuracy",
                },
            ]
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "The oracle cost-aware optimum is a reference benchmark because "
            "it uses true held-out simulated accuracy. It is not available to "
            "the unseen-device recommender."
        )
        top_cols = [
            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",
            "predicted_accuracy",
            "validation_adjusted_score",
            "accuracy",
            "relative_hardware_cost_proxy",
        ]
        top_cols = [c for c in top_cols if c in candidate_df.columns]

        with st.expander(f"View all {len(candidate_df)} held-out configurations"):
            st.dataframe(
                candidate_df[top_cols].sort_values(
                    "predicted_accuracy",
                    ascending=False,
                ),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# CUSTOM DEVICE
# ============================================================

elif page == "Custom Device":
    section_header(
        "Custom Device",
        (
            "Enter a few device descriptors and ask what accelerator configuration "
            "the current surrogate would recommend."
        ),
        (
            "This is descriptor-level prediction. It does not create missing "
            "literature evidence or validate a fabricated device."
        ),
        level=2,
    )

    render_html(
        """
        <div class="warning-card">
            <strong>Scope:</strong> this workflow predicts simulator-derived
            accelerator performance from user-supplied descriptors. It is not a
            material-characterization tool and does not prove physical performance.
        </div>
        """
    )

    if ml_df.empty:
        st.error("ml_dataset.csv is missing, so Custom Device cannot run.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            custom_ratio = st.number_input(
                "ON/OFF ratio",
                min_value=1.001,
                value=50.0,
                step=1.0,
                help=(
                    "Ratio between high- and low-conductance states. "
                    "Use a value from measurement/literature when possible."
                ),
            )

        with col2:
            custom_mode = st.selectbox(
                "Conductance behavior",
                [
                    "ANALOG",
                    "GRADUAL_MULTILEVEL",
                    "DISCRETE_BINARY",
                    "DISCRETE_MULTILEVEL",
                ],
                help=(
                    "ANALOG/GRADUAL do not require a fixed state count. "
                    "DISCRETE modes do."
                ),
            )

        with col3:
            if custom_mode == "DISCRETE_BINARY":
                custom_states = 2
                st.number_input(
                    "Physical states",
                    value=2,
                    disabled=True,
                    help="Binary switching uses two physical states.",
                )
            elif custom_mode == "DISCRETE_MULTILEVEL":
                custom_states = int(
                    st.number_input(
                        "Physical states",
                        min_value=2,
                        value=8,
                        step=1,
                        help="Fixed stable conductance-state count.",
                    )
                )
            else:
                state_known = st.checkbox(
                    "Fixed state count is known",
                    value=False,
                    help=(
                        "Leave unchecked if the literature does not report a fixed "
                        "discrete state count."
                    ),
                )
                if state_known:
                    custom_states = int(
                        st.number_input(
                            "Physical states",
                            min_value=2,
                            value=8,
                            step=1,
                        )
                    )
                else:
                    custom_states = 0

        if st.button(
            "Predict Custom Device",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Training the surrogate and evaluating 245 candidate designs..."):
                try:
                    from stage13_custom_device import (
                        ALL_FEATURES as CUSTOM_FEATURES,
                        build_candidate_grid,
                        build_model,
                        device_descriptor_table,
                        nearest_descriptor,
                        reliability_table,
                        tree_predictions,
                    )

                    model = build_model()
                    model.fit(
                        ml_df[CUSTOM_FEATURES],
                        ml_df["accuracy"],
                    )

                    custom_candidates = build_candidate_grid(
                        on_off_ratio=float(custom_ratio),
                        mode=custom_mode,
                        states=int(custom_states),
                        config_reference=ml_df,
                    )

                    mean_pred, tree_std = tree_predictions(
                        model,
                        custom_candidates[CUSTOM_FEATURES],
                    )

                    custom_candidates["predicted_accuracy"] = mean_pred
                    custom_candidates["tree_disagreement"] = tree_std

                    rel = reliability_table(predictions)

                    custom_candidates = custom_candidates.merge(
                        rel,
                        on=["requested_weight_bits", "adc_bits"],
                        how="left",
                        validate="many_to_one",
                    )

                    fallback_penalty = float(
                        predictions["absolute_prediction_error_pp"].mean()
                    )

                    custom_candidates["historical_mean_abs_error_pp"] = (
                        custom_candidates["historical_mean_abs_error_pp"]
                        .fillna(fallback_penalty)
                    )

                    custom_candidates["validation_adjusted_score"] = (
                        custom_candidates["predicted_accuracy"]
                        - custom_candidates["historical_mean_abs_error_pp"]
                    )

                    descriptor_profiles = device_descriptor_table(ml_df)
                    nearest, distance, _, _ = nearest_descriptor(
                        custom_candidates.iloc[0],
                        descriptor_profiles,
                    )

                    same_mode_devices = (
                        ml_df[
                            [
                                "device_id",
                                "device_conductance_mode",
                                "device_log10_on_off_ratio",
                            ]
                        ]
                        .drop_duplicates("device_id")
                    )

                    same_mode = same_mode_devices[
                        same_mode_devices["device_conductance_mode"]
                        == custom_mode
                    ]

                    log_ratio = math.log10(float(custom_ratio))

                    if same_mode.empty:
                        supported = False
                        support_reason = "No same-mode training device"
                    else:
                        min_ratio = float(
                            same_mode["device_log10_on_off_ratio"].min()
                        )
                        max_ratio = float(
                            same_mode["device_log10_on_off_ratio"].max()
                        )
                        supported = (
                            len(same_mode) >= 2
                            and min_ratio <= log_ratio <= max_ratio
                        )
                        if len(same_mode) < 2:
                            support_reason = "Insufficient same-mode device coverage"
                        elif not (min_ratio <= log_ratio <= max_ratio):
                            support_reason = "Same-mode ON/OFF extrapolation"
                        else:
                            support_reason = "Same-mode interpolation supported"

                    if supported:
                        best_score = float(
                            custom_candidates["validation_adjusted_score"].max()
                        )
                        near = custom_candidates[
                            custom_candidates["validation_adjusted_score"]
                            >= best_score - NEAR_OPTIMAL_TOLERANCE_PP
                        ].copy()

                        chosen = near.sort_values(
                            [
                                "estimated_memristor_cells",
                                "relative_hardware_cost_proxy",
                                "requested_weight_bits",
                                "adc_bits",
                                "validation_adjusted_score",
                                "crossbar_size",
                            ],
                            ascending=[True, True, True, True, False, False],
                        ).iloc[0]
                        custom_policy = "VALIDATION_AWARE_COST"
                    else:
                        chosen = custom_candidates.sort_values(
                            [
                                "predicted_accuracy",
                                "estimated_memristor_cells",
                                "relative_hardware_cost_proxy",
                            ],
                            ascending=[False, True, True],
                        ).iloc[0]
                        custom_policy = "ACCURACY_FIRST_EXTRAPOLATION_FALLBACK"

                    st.session_state["custom_result"] = {
                        "candidates": custom_candidates,
                        "chosen": chosen,
                        "nearest": nearest,
                        "distance": float(distance),
                        "policy": custom_policy,
                        "support_reason": support_reason,
                    }

                except ModuleNotFoundError as exc:
                    if getattr(exc, "name", "") == "sklearn":
                        st.warning(
                            "Custom prediction is temporarily unavailable because "
                            "the deployed environment is missing scikit-learn. "
                            "Add scikit-learn to requirements.txt and redeploy."
                        )
                    else:
                        st.warning(
                            "Custom prediction could not start because a required "
                            "Python package is unavailable."
                        )

                    if st.session_state.app_mode == "Researcher":
                        with st.expander("Technical detail"):
                            st.code(str(exc))

                except Exception as exc:
                    st.warning(
                        "Custom prediction could not complete. "
                        "The rest of the website is still available."
                    )

                    if st.session_state.app_mode == "Researcher":
                        with st.expander("Technical detail"):
                            st.code(
                                f"{type(exc).__name__}: {exc}"
                            )

        result = st.session_state.get("custom_result")

        if result:
            chosen = result["chosen"]
            nearest = result["nearest"]

            section_header(
                "Recommendation",
                (
                    "The same support-gated idea used in the held-out validation "
                    "is applied to this custom descriptor."
                ),
            )

            render_html(
                f"""
                <div class="callout">
                    <span class="policy-badge">
                        {format_policy(result['policy'])}
                    </span><br><br>
                    <strong>Support decision:</strong> {result['support_reason']}
                </div>
                """
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(
                "Crossbar",
                f"{int(chosen['crossbar_size'])} × {int(chosen['crossbar_size'])}",
            )
            c2.metric(
                "Weight precision",
                f"{int(chosen['requested_weight_bits'])}-bit",
            )
            c3.metric(
                "ADC precision",
                f"{int(chosen['adc_bits'])}-bit",
            )
            c4.metric(
                "Predicted accuracy",
                f"{float(chosen['predicted_accuracy']):.2f}%",
            )

            d1, d2, d3 = st.columns(3)
            d1.metric(
                "Validation-adjusted score",
                f"{float(chosen['validation_adjusted_score']):.2f}",
            )
            d2.metric(
                "Nearest known device",
                str(nearest["device_id"]),
            )
            d3.metric(
                "Descriptor distance",
                f"{result['distance']:.3f}",
            )

            st.caption(
                "Descriptor distance is a relative heuristic only. It is not an "
                "OOD probability and not proof that the custom material is physically valid."
            )

            if st.session_state.app_mode == "Researcher":
                with st.expander("View top custom-device candidates"):
                    show = result["candidates"].sort_values(
                        "validation_adjusted_score",
                        ascending=False,
                    ).head(25)
                    cols = [
                        "crossbar_size",
                        "requested_weight_bits",
                        "adc_bits",
                        "predicted_accuracy",
                        "historical_mean_abs_error_pp",
                        "validation_adjusted_score",
                        "tree_disagreement",
                        "relative_hardware_cost_proxy",
                    ]
                    st.dataframe(
                        show[cols],
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# REVERSE DESIGN
# ============================================================

elif page == "Reverse Design":
    section_header(
        "Reverse Design",
        (
            "Instead of starting from a memristor, start from the accelerator "
            "performance you want and search backward for useful device properties."
        ),
        (
            "The reverse search operates over descriptor scenarios and the trained "
            "surrogate. It returns device-property requirements, not a newly "
            "discovered chemical material."
        ),
        level=2,
    )

    render_html(
        """
        <div class="step-row">
            <div class="step-pill"><b>Desired performance</b>Accuracy + cost constraint</div>
            <div class="step-pill"><b>Search backward</b>Surrogate descriptor space</div>
            <div class="step-pill"><b>Device target</b>ON/OFF + behavior + states</div>
            <div class="step-pill"><b>Literature anchor</b>Nearest known profile</div>
        </div>
        """
    )

    reverse_df = DATA["reverse_candidates"]

    if reverse_df.empty:
        st.error(
            "reverse_design_candidates.csv is missing. Run Stage 14 first."
        )
    else:
        f1, f2, f3 = st.columns(3)

        with f1:
            target_accuracy = st.number_input(
                "Minimum validation-adjusted accuracy",
                min_value=0.0,
                max_value=100.0,
                value=95.5,
                step=0.1,
                help=(
                    "Predicted simulated accuracy minus the empirical historical "
                    "study-blocked error penalty."
                ),
            )

        with f2:
            max_cost = st.number_input(
                "Maximum relative hardware-cost proxy",
                min_value=1.0,
                value=5000.0,
                step=100.0,
                help=(
                    "A heuristic comparison metric. It is not measured energy, "
                    "area, latency, power or monetary cost."
                ),
            )

        with f3:
            use_distance = st.checkbox(
                "Limit descriptor distance",
                value=False,
                help=(
                    "Optional relative descriptor-distance filter. "
                    "This is not an OOD probability."
                ),
            )
            max_distance = (
                st.number_input(
                    "Maximum descriptor distance",
                    min_value=0.0,
                    value=1.0,
                    step=0.05,
                )
                if use_distance
                else None
            )

        feasible = reverse_df[
            (reverse_df["validation_adjusted_score"] >= float(target_accuracy))
            & (
                reverse_df["relative_hardware_cost_proxy"]
                <= float(max_cost)
            )
        ].copy()

        if use_distance:
            feasible = feasible[
                feasible["descriptor_distance_to_nearest_known"]
                <= float(max_distance)
            ]

        descriptor_key = [
            "candidate_on_off_ratio",
            "candidate_mode",
            "candidate_state_count",
        ]

        if feasible.empty:
            st.warning(
                "No candidate satisfies all current constraints. "
                "Lower the target accuracy, raise the cost limit, or relax "
                "the descriptor-distance limit."
            )
        else:
            req = (
                feasible.sort_values(
                    descriptor_key
                    + [
                        "relative_hardware_cost_proxy",
                        "validation_adjusted_score",
                    ],
                    ascending=[True, True, True, True, False],
                )
                .drop_duplicates(descriptor_key, keep="first")
                .sort_values(
                    [
                        "descriptor_distance_to_nearest_known",
                        "relative_hardware_cost_proxy",
                        "validation_adjusted_score",
                    ],
                    ascending=[True, True, False],
                )
                .reset_index(drop=True)
            )

            req["Match"] = req[
                "descriptor_distance_to_nearest_known"
            ].map(match_class)

            m1, m2, m3 = st.columns(3)
            m1.metric("Feasible configurations", f"{len(feasible):,}")
            m2.metric("Feasible descriptor targets", len(req))
            m3.metric(
                "Closest literature distance",
                f"{float(req['descriptor_distance_to_nearest_known'].min()):.3f}",
            )

            section_header(
                "Top reverse-design requirements",
                (
                    "Each row says: if a device approximately had these properties, "
                    "the surrogate predicts that the shown accelerator configuration "
                    "could satisfy your target."
                ),
            )

            display = req.head(15).copy()

            display["candidate_mode"] = display[
                "candidate_mode"
            ].map(human_mode)

            display["candidate_state_count"] = [
                display_target_states(mode, states)
                for mode, states in zip(
                    req.head(15)["candidate_mode"],
                    req.head(15)["candidate_state_count"],
                )
            ]

            display["Match"] = display["Match"].map(
                human_match_class
            )

            display = display.rename(
                columns={
                    "candidate_on_off_ratio": "Target ON/OFF",
                    "candidate_mode": "Conductance Mode",
                    "candidate_state_count": "Target States",
                    "crossbar_size": "Crossbar",
                    "requested_weight_bits": "Weight Bits",
                    "adc_bits": "ADC Bits",
                    "validation_adjusted_score": "Adjusted Score",
                    "relative_hardware_cost_proxy": "Cost Proxy",
                    "nearest_known_device": "Nearest Literature Device",
                    "descriptor_distance_to_nearest_known": "Descriptor Distance",
                }
            )

            cols = [
                "Target ON/OFF",
                "Conductance Mode",
                "Target States",
                "Crossbar",
                "Weight Bits",
                "ADC Bits",
                "Adjusted Score",
                "Cost Proxy",
                "Nearest Literature Device",
                "Descriptor Distance",
                "Match",
            ]

            st.dataframe(
                display[cols].round(3),
                use_container_width=True,
                hide_index=True,
            )

            section_header(
                "Experimental literature match",
                (
                    "For each reverse-designed device-property target, "
                    "the tool checks which experimental profile in the current "
                    "literature dataset is closest."
                ),
                (
                    "This is Stage 15 descriptor matching. "
                    "A close descriptor match supports plausibility of the "
                    "modeled device properties only; it does not mean the paper "
                    "demonstrated the proposed accelerator configuration."
                ),
            )

            try:
                from stage15_match_to_experiments import (
                    device_profiles as stage15_device_profiles,
                    profile_distance as stage15_profile_distance,
                    classify_match as stage15_classify_match,
                )

                experimental_profiles = stage15_device_profiles(
                    ml_df
                )

                match_rows = []

                for _, requirement_row in req.head(15).iterrows():
                    best_profile = None
                    best_distance = None
                    best_terms = None

                    for _, literature_profile in experimental_profiles.iterrows():
                        terms = stage15_profile_distance(
                            requirement_row,
                            literature_profile,
                        )

                        if (
                            best_distance is None
                            or terms["match_distance"] < best_distance
                        ):
                            best_profile = literature_profile
                            best_distance = float(
                                terms["match_distance"]
                            )
                            best_terms = terms

                    if best_profile is None:
                        continue

                    match_label = stage15_classify_match(
                        best_terms["match_distance"],
                        best_terms["ratio_log10_distance"],
                        best_terms["mode_mismatch"],
                        best_terms["state_count_distance"],
                    )

                    source_title = "Not available"

                    source_match = devices[
                        devices["device_id"]
                        == str(best_profile["device_id"])
                    ]

                    if (
                        not source_match.empty
                        and "source_title" in source_match.columns
                    ):
                        source_title = clean_text(
                            source_match.iloc[0]["source_title"]
                        )

                    coverage_text = "N/A"

                    if not DATA["evidence_coverage"].empty:
                        ev_match = DATA["evidence_coverage"][
                            DATA["evidence_coverage"]["device_id"]
                            == str(best_profile["device_id"])
                        ]

                        if (
                            not ev_match.empty
                            and "evidence_coverage_fraction"
                            in ev_match.columns
                        ):
                            coverage_text = (
                                f"{100 * float(ev_match.iloc[0]['evidence_coverage_fraction']):.0f}%"
                            )

                    match_rows.append(
                        {
                            "Target ON/OFF": float(
                                requirement_row["candidate_on_off_ratio"]
                            ),
                            "Target Mode": str(
                                requirement_row["candidate_mode"]
                            ),
                            "Target States": int(
                                requirement_row["candidate_state_count"]
                            ),
                            "Matched Device": str(
                                best_profile["device_id"]
                            ),
                            "Family": str(
                                best_profile["technology_family"]
                            ),
                            "Study / DOI": str(
                                best_profile["study_id"]
                            ),
                            "Match Class": match_label,
                            "Distance": best_distance,
                            "Evidence Coverage": coverage_text,
                            "Source": source_title,
                        }
                    )

                match_df = pd.DataFrame(match_rows)

                if not match_df.empty:
                    match_df["Target Mode"] = match_df[
                        "Target Mode"
                    ].map(human_mode)

                    match_df["Target States"] = [
                        display_target_states(mode, states)
                        for mode, states in zip(
                            [row["candidate_mode"] for _, row in req.head(15).iterrows()],
                            match_df["Target States"],
                        )
                    ]

                    match_df["Match Class"] = match_df[
                        "Match Class"
                    ].map(human_match_class)

                    match_df = match_df.rename(
                        columns={
                            "Evidence Coverage": "Core Evidence Coverage"
                        }
                    )

                    st.dataframe(
                        match_df.round(3),
                        use_container_width=True,
                        hide_index=True,
                    )

                    exact_or_very_close = int(
                        match_df["Match Class"].isin(
                            [
                                "EXACT_DESCRIPTOR_MATCH",
                                "VERY_CLOSE_EXISTING_PROFILE",
                            ]
                        ).sum()
                    )

                    st.caption(
                        f"{exact_or_very_close} of the displayed targets have "
                        "an exact or very-close descriptor match in the current "
                        "experimental literature dataset."
                    )

            except Exception as exc:
                st.info(
                    "Experimental matching data are available in the saved "
                    "Stage-15 results, but the live matching helper could not "
                    "be loaded in this session."
                )

                if (
                    st.session_state.app_mode == "Researcher"
                    and not DATA["reverse_matches"].empty
                ):
                    with st.expander(
                        "View saved Stage-15 experimental matches"
                    ):
                        st.dataframe(
                            DATA["reverse_matches"],
                            use_container_width=True,
                            hide_index=True,
                        )

            render_html(
                """
                <div class="warning-card">
                    <strong>Do not read this as material discovery.</strong>
                    A reverse-design target is a device-property specification.
                    Experimental matching checks whether similar descriptors have
                    appeared in the literature; it does not prove that a chemical
                    composition or fabrication process will realize the target or
                    that the paper demonstrated the proposed accelerator.
                </div>
                """
            )

            if st.session_state.app_mode == "Researcher":
                with st.expander("View full feasible reverse-design table"):
                    st.dataframe(
                        req,
                        use_container_width=True,
                        hide_index=True,
                    )

                if not DATA["reverse_match_summary"].empty:
                    st.markdown("#### Saved Stage-15 literature matching summary")
                    st.dataframe(
                        DATA["reverse_match_summary"],
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# RESEARCH TARGETS
# ============================================================

elif page == "Research Targets":
    section_header(
        "Future Experimental Research Targets",
        (
            "Explore device-property regions identified by the completed "
            "Stage-16 pipeline as useful accelerator-oriented research gaps."
        ),
        (
            "These targets are generated from Stage-14 reverse-design "
            "requirements matched against Stage-15 experimental profiles. "
            "The saved Stage-16 table is the authoritative source shown here."
        ),
        level=2,
    )

    future_targets = DATA["future_targets"].copy()

    if future_targets.empty:
        st.error(
            "future_research_targets.csv is missing. Run Stage 16 before "
            "using this page."
        )
    else:
        default_min_score = 95.5
        default_max_cost = 5000.0

        c1, c2 = st.columns(2)

        with c1:
            future_accuracy = st.number_input(
                "Minimum validation-adjusted accuracy",
                value=default_min_score,
                step=0.1,
                min_value=0.0,
                max_value=100.0,
                key="future_accuracy",
                help=(
                    "This filter narrows the saved Stage-16 research-target set. "
                    "It does not regenerate new targets."
                ),
            )

        with c2:
            future_cost = st.number_input(
                "Maximum relative hardware-cost proxy",
                value=default_max_cost,
                step=100.0,
                min_value=1.0,
                key="future_cost",
                help=(
                    "Relative comparison proxy only; not measured energy, area, "
                    "latency, power or monetary cost."
                ),
            )

        future = future_targets[
            (
                future_targets["validation_adjusted_score"]
                >= float(future_accuracy)
            )
            & (
                future_targets["relative_hardware_cost_proxy"]
                <= float(future_cost)
            )
        ].copy()

        if future.empty:
            st.warning(
                "No saved Stage-16 research target satisfies the current filters."
            )
        else:
            if "future_research_rank" in future.columns:
                future = future.sort_values(
                    "future_research_rank"
                ).reset_index(drop=True)
            else:
                future = future.sort_values(
                    [
                        "match_distance",
                        "validation_adjusted_score",
                        "relative_hardware_cost_proxy",
                    ],
                    ascending=[True, False, True],
                ).reset_index(drop=True)

            a = int(
                (
                    future["research_tier"]
                    == "TIER_A_INTERPOLATIVE_GAP"
                ).sum()
            )
            b = int(
                (
                    future["research_tier"]
                    == "TIER_B_MODERATE_GAP"
                ).sum()
            )
            c = int(
                (
                    future["research_tier"]
                    == "TIER_C_HIGH_EXTRAPOLATION"
                ).sum()
            )

            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Research-gap targets", len(future))
            x2.metric("Tier A", a)
            x3.metric("Tier B", b)
            x4.metric("Tier C", c)

            st.markdown(
                """
                **Tier A** = closest genuine research gap and best near-term target.  
                **Tier B** = moderate extrapolation; stronger validation required.  
                **Tier C** = high extrapolation; hypothesis-generating only.
                """
            )

            if (
                abs(float(future_accuracy) - default_min_score) < 1e-9
                and abs(float(future_cost) - default_max_cost) < 1e-9
            ):
                st.caption(
                    "Default view = the saved Stage-16 result set. "
                    "This should reproduce the official Stage-16 counts."
                )
            else:
                st.caption(
                    "You are viewing a filtered subset of the saved Stage-16 "
                    "research targets."
                )

            display = future.head(15).copy()

            display["research_tier"] = display[
                "research_tier"
            ].map(human_research_tier)

            raw_modes = display["candidate_mode"].copy()

            display["candidate_mode"] = raw_modes.map(
                human_mode
            )

            display["candidate_state_count"] = [
                display_target_states(mode, states)
                for mode, states in zip(
                    raw_modes,
                    display["candidate_state_count"],
                )
            ]

            if "match_class" in display.columns:
                display["match_class"] = display[
                    "match_class"
                ].map(human_match_class)

            display = display.rename(
                columns={
                    "future_research_rank": "Rank",
                    "research_tier": "Research Tier",
                    "candidate_on_off_ratio": "Target ON/OFF",
                    "candidate_mode": "Conductance Mode",
                    "candidate_state_count": "Target States",
                    "crossbar_size": "Crossbar",
                    "requested_weight_bits": "Weight Bits",
                    "adc_bits": "ADC Bits",
                    "validation_adjusted_score": "Adjusted Score",
                    "relative_hardware_cost_proxy": "Cost Proxy",
                    "matched_device_id": "Nearest Literature Device",
                    "matched_study_id": "Study / DOI",
                    "matched_family": "Matched Family",
                    "match_class": "Match Class",
                    "match_distance": "Gap Distance",
                }
            )

            cols = [
                c
                for c in [
                    "Rank",
                    "Research Tier",
                    "Target ON/OFF",
                    "Conductance Mode",
                    "Target States",
                    "Crossbar",
                    "Weight Bits",
                    "ADC Bits",
                    "Adjusted Score",
                    "Cost Proxy",
                    "Nearest Literature Device",
                    "Matched Family",
                    "Gap Distance",
                ]
                if c in display.columns
            ]

            st.dataframe(
                display[cols].round(3),
                use_container_width=True,
                hide_index=True,
            )

            render_html(
                """
                <div class="callout">
                    <strong>Nanotechnology meaning:</strong> use the target ON/OFF
                    ratio, conductance behavior and state-count requirement as
                    experimentally testable device specifications. Material/stack
                    engineering can then investigate how to realize those
                    specifications.
                </div>
                """
            )

            render_html(
                """
                <div class="warning-card">
                    <strong>Future direction, not current capability:</strong>
                    predicting a new chemical composition, electrode stack, oxide
                    thickness or fabrication recipe requires a much richer dataset
                    containing composition, structure, process, endurance, retention,
                    variability and temperature information.
                </div>
                """
            )

            if st.session_state.app_mode == "Researcher":
                with st.expander(
                    "View all filtered Stage-16 research targets"
                ):
                    full_display = future.copy()

                    full_display["research_tier"] = full_display[
                        "research_tier"
                    ].map(human_research_tier)

                    raw_full_modes = full_display[
                        "candidate_mode"
                    ].copy()

                    full_display["candidate_mode"] = raw_full_modes.map(
                        human_mode
                    )

                    full_display["candidate_state_count"] = [
                        display_target_states(mode, states)
                        for mode, states in zip(
                            raw_full_modes,
                            full_display["candidate_state_count"],
                        )
                    ]

                    if "match_class" in full_display.columns:
                        full_display["match_class"] = full_display[
                            "match_class"
                        ].map(human_match_class)

                    st.dataframe(
                        full_display,
                        use_container_width=True,
                        hide_index=True,
                    )


# ============================================================
# WHY NANO
# ============================================================

elif page == "Why Nano?":
    section_header(
        "Why is this a nanotechnology project?",
        (
            "The project is not simply ranking material names. "
            "Nanoscale device behavior constrains how neural-network weights "
            "can be represented in a memristor accelerator."
        ),
        level=2,
    )

    st.markdown(
        f"## {selected_material['symbol']} — {selected_material['short_name']}"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_info_card(
            "Nano device",
            selected_material["symbol"],
            selected_material["short_name"],
        )

    with c2:
        render_info_card(
            "Electrical behavior",
            public_mode_name(conductance_mode),
            "Literature-derived switching behavior used by the model.",
        )

    with c3:
        render_info_card(
            "Weight representation",
            f"{rec_levels} effective levels",
            public_mapping_name(conductance_mode),
        )

    with c4:
        render_info_card(
            "Physical mapping",
            f"{rec_cells} cells / weight",
            "Physical memristor-cell requirement of the selected mapping.",
        )

    render_html(
        """
        <div class="step-row">
            <div class="step-pill"><b>Nanomaterial / stack</b>Determines electrical switching behavior</div>
            <div class="step-pill"><b>Device descriptors</b>ON/OFF, state capability, conductance mode</div>
            <div class="step-pill"><b>Weight mapping</b>How neural weights use physical cells</div>
            <div class="step-pill"><b>Accelerator</b>Accuracy / precision / cost trade-off</div>
        </div>
        """
    )

    st.divider()

    section_header(
        "What is modeled — and what is not",
        (
            "The current model captures a limited device-to-accelerator pathway. "
            "It does not yet simulate every physical nonideality."
        ),
    )

    st.markdown(
        """
        **Currently represented**
        - ON/OFF ratio
        - conductance behavior
        - fixed state count when available
        - mapping strategy
        - weight precision
        - ADC precision
        - crossbar size

        **Not yet directly calibrated to measured device physics**
        - cycle-to-cycle and device-to-device variability distributions
        - retention drift
        - endurance degradation
        - line resistance and IR drop
        - switching kinetics
        - temperature dependence
        - measured energy, area, latency and peripheral power
        """
    )

    if st.session_state.app_mode == "Researcher":
        render_html(
            """
            <div class="warning-card">
                Stage 10 therefore uses an <strong>uncalibrated effective-precision
                stress-test proxy</strong>. Stage 11 uses a
                <strong>relative hardware-cost proxy</strong>. Neither should be
                presented as measured physical robustness or measured hardware cost.
            </div>
            """
        )


# ============================================================
# RESEARCH EVIDENCE
# ============================================================

elif page == "Research Evidence":
    section_header(
        "Research Evidence",
        (
            "This page shows how well the recommendation has been tested and "
            "where the current evidence is still weak."
        ),
        (
            "Study-blocked, leave-one-study-out and leave-one-family-out validation "
            "are separated from uncertainty/OOD diagnostics and proxy analyses."
        ),
        level=2,
    )

    if st.session_state.app_mode == "Researcher":
        render_html(
            """
            <span class="stage-chip">Evidence-aware</span>
            <span class="stage-chip">Study-blocked validation</span>
            <span class="stage-chip">Support-gated recommendation</span>
            <span class="stage-chip">Sensitivity</span>
            <span class="stage-chip">OOD diagnostic</span>
            <span class="stage-chip">Robustness proxy</span>
            <span class="stage-chip">Pareto</span>
            <span class="stage-chip">Reverse design</span>
            """
        )
        st.write("")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Device profiles", device_count)
    e2.metric("Independent studies", study_count)
    e3.metric("Technology families", family_count)
    e4.metric("Simulation rows", f"{total_simulation_rows:,}")

    section_header("Primary recommendation performance")

    p1, p2, p3 = st.columns(3)
    p1.metric("Near-optimal success", f"{primary_success_rate:.1f}%")
    p2.metric("Mean regret", f"{primary_mean_regret:.3f} pp")
    p3.metric(
        "Accuracy-first fallback",
        (
            f"{guarded_fallback_rate:.1f}%"
            if np.isfinite(guarded_fallback_rate)
            else "N/A"
        ),
    )

    render_html(
        """
        <div class="warning-card">
            The support-gated result is intentionally conservative. A high
            fallback rate means the dataset often does not support aggressive
            cost optimization for a held-out descriptor region.
        </div>
        """
    )

    result_cols = [
        "held_out_device",
        "held_out_study",
        "held_out_family",
        "baseline_regret_pp",
        "confidence_regret_pp",
        "validation_aware_regret_pp",
        "guarded_policy",
        "guarded_regret_pp",
        "guarded_near_optimal_success",
    ]
    result_cols = [c for c in result_cols if c in summary.columns]

    result_display = summary[result_cols].copy()

    if "guarded_policy" in result_display.columns:
        result_display["guarded_policy"] = result_display[
            "guarded_policy"
        ].map(format_policy)

    st.dataframe(
        result_display.round(3),
        use_container_width=True,
        hide_index=True,
    )

    render_regret_comparison_chart(summary)

    st.caption(
        "The support-gated policy fixed the large TiOx_02_Ni failure by "
        "falling back to accuracy-first selection in an extrapolative region. "
        "This does not make the low-ADC predictions themselves accurate."
    )

    if st.session_state.app_mode == "Researcher":
        st.divider()

        section_header(
            "Validation overview",
            (
                "Compare primary study-blocked validation with leave-one-study "
                "and leave-one-family evaluation."
            ),
        )

        if not DATA["validation_overview"].empty:
            st.dataframe(
                DATA["validation_overview"].round(3),
                use_container_width=True,
                hide_index=True,
            )

        section_header(
            "Evidence-aware missing-data coverage",
            (
                "Reported, derived, assumed and missing information is kept explicit."
            ),
        )

        if not DATA["evidence_coverage"].empty:
            st.dataframe(
                DATA["evidence_coverage"],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Stage-6 evidence coverage file not found.")

        section_header(
            "Sensitivity",
            (
                "This measures grouped outcome ranges over the existing simulated "
                "configuration grid. It does not establish physical causality."
            ),
        )

        if not DATA["sensitivity_summary"].empty:
            st.dataframe(
                DATA["sensitivity_summary"].round(3),
                use_container_width=True,
                hide_index=True,
            )

            selected_sens = DATA["sensitivity_by_device"][
                DATA["sensitivity_by_device"]["device_id"]
                == selected_device
            ]

            if not selected_sens.empty:
                st.caption(
                    f"Selected-device sensitivity: {selected_device}"
                )

                render_sensitivity_chart(
                    selected_sens
                )

                st.dataframe(
                    selected_sens.round(3),
                    use_container_width=True,
                    hide_index=True,
                )

        section_header(
            "OOD + uncertainty diagnostic",
            (
                "OOD score and Random-Forest tree disagreement are diagnostics only. "
                "They are not calibrated confidence probabilities."
            ),
        )

        if not DATA["ood_device"].empty:
            st.dataframe(
                DATA["ood_device"].round(3),
                use_container_width=True,
                hide_index=True,
            )

        section_header(
            "Nonideality robustness proxy",
            (
                "This reuses the existing grid and stresses effective weight/ADC "
                "precision. It is not measured variability, drift or noise."
            ),
        )

        if not DATA["nonideality_summary"].empty:
            st.dataframe(
                DATA["nonideality_summary"].round(3),
                use_container_width=True,
                hide_index=True,
            )

        section_header(
            "Hardware-cost / accuracy Pareto analysis",
            (
                "The cost proxy is heuristic. The oracle 0.5-pp reference uses "
                "true simulated accuracy and is not an unseen-device recommendation."
            ),
        )

        if not DATA["pareto_summary"].empty:
            selected_pareto = DATA["pareto_front"][
                DATA["pareto_front"]["device_id"]
                == selected_device
            ].copy()

            render_pareto_chart(
                candidate_df,
                selected_pareto,
                recommended_row,
            )

            st.caption(
                "Dots are simulated configurations; the connected frontier "
                "shows Pareto-efficient accuracy/cost trade-offs. "
                "The cost axis is a relative hardware-cost proxy."
            )

            st.dataframe(
                DATA["pareto_summary"].round(3),
                use_container_width=True,
                hide_index=True,
            )

        section_header(
            "Model inputs",
            (
                "These are the features used by the cross-device Random Forest."
            ),
        )

        for feature in MODEL_FEATURE_LABELS:
            st.write(f"• {feature}")


# ============================================================
# SOURCES + LIMITATIONS
# ============================================================

elif page == "Sources & Limitations":
    section_header(
        "Sources & Scientific Transparency",
        (
            "See where the selected device data came from and which parts are "
            "reported, derived, assumed or missing."
        ),
        level=2,
    )

    t1, t2, t3 = st.columns(3)
    t1.metric("Literature profiles", literature_device_count)
    t2.metric("Traceability records", len(trace))
    t3.metric(
        "Provenance fields complete",
        (
            f"{provenance_completeness:.1f}%"
            if np.isfinite(provenance_completeness)
            else "N/A"
        ),
    )

    st.markdown(
        f"### {selected_material['symbol']} — {selected_material['short_name']}"
    )

    if profile is not None:
        profile_fields = [
            c
            for c in [
                "source_title",
                "doi",
                "year",
                "device_stack",
                "active_material",
                "parameter_source",
            ]
            if c in profile.index
        ]

        source_table = pd.DataFrame(
            [{"Field": c, "Value": profile[c]} for c in profile_fields]
        )
        st.dataframe(
            source_table,
            use_container_width=True,
            hide_index=True,
        )

    selected_trace = trace[trace["device_id"] == selected_device].copy()
    selected_audit = audit[audit["device_id"] == selected_device].copy()

    with st.expander("View literature provenance records", expanded=False):
        st.dataframe(
            selected_trace,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("View simulator-input audit", expanded=False):
        st.dataframe(
            selected_audit,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    section_header(
        "Current limitations",
        (
            "These limits are part of the scientific interpretation, not hidden "
            "implementation details."
        ),
    )

    st.markdown(
        f"""
        - The current evidence base contains **{device_count} device profiles from
          {study_count} independent primary studies across {family_count} families**.
        - TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt come from one shared study.
        - The **{total_simulation_rows:,} simulation rows are not independent
          physical experiments**.
        - The support gate is a transparent extrapolation safeguard, not a
          calibrated confidence model.
        - Random-Forest tree disagreement and descriptor distance are heuristic
          diagnostics.
        - Stage 10 is an **uncalibrated effective-precision robustness proxy**,
          not measured cycle-to-cycle/device-to-device variability, drift,
          endurance or temperature behavior.
        - Stage 11 uses a **relative hardware-cost proxy**, not measured energy,
          area, power, latency or monetary cost.
        - Reverse design generates **device-property targets**, not guaranteed
          materials, stacks or fabrication recipes.
        - The simulator has not yet been validated against a fabricated
          memristor crossbar accelerator.
        """
    )

    render_html(
        """
        <div class="callout">
            <strong>Future direction:</strong> a true material/structure
            recommendation engine would require a substantially larger dataset
            containing composition, electrode materials, thickness, fabrication
            process, set/reset/forming voltages, endurance, retention, variability
            and temperature behavior.
        </div>
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "NanoMemristor AI Designer — literature-grounded memristor "
    "device-to-accelerator co-design research prototype."
)
