from __future__ import annotations

from textwrap import dedent

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


def _html(content: str):
    """Render trusted local UI HTML without Markdown code-block parsing."""
    st.html(dedent(content).strip())



# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

STAGE37 = ROOT / "results" / "frozen" / "stage37"
HARDWARE = ROOT / "results" / "frozen" / "hardware"
STAGE40 = ROOT / "results" / "stage40_hardware_finalization"

MODEL_SUMMARY_FILE = STAGE37 / "model_comparison_summary.csv"
DEVICE_RESULTS_FILE = STAGE37 / "device_level_results.csv"
PAIRWISE_FILE = STAGE37 / "stage37C_pairwise_agreement.csv"
DEVICE_AGREEMENT_FILE = STAGE37 / "stage37C_device_agreement.csv"
SEED_SUMMARY_FILE = STAGE37 / "stage37D_seed_summary.csv"
SEED_ROBUSTNESS_FILE = STAGE37 / "stage37D_seed_robustness_summary.csv"

HW_STRATEGIC_FILE = HARDWARE / "stage34G_strategic_hardware_dataset.csv"
HW_JOINT_FILE = HARDWARE / "stage35B_joint_accuracy_hardware.csv"
HW_PARETO_CANDIDATES_FILE = HARDWARE / "stage35C_pareto_candidates.csv"
HW_PARETO_FRONT_FILE = HARDWARE / "stage35C_pareto_front.csv"

HW_SUMMARY_FILE = STAGE40 / "stage40_hardware_summary.csv"


MODEL_LABELS = {
    "RandomForest": "Random Forest",
    "ExtraTrees": "Extra Trees",
    "GradientBoosting": "Gradient Boosting",
    "HistGradientBoosting": "HistGradientBoosting",
}


# ============================================================
# STYLE
# ============================================================

def inject_publication_style():
    _html("""
        <style>
        .pub-shell {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 20px;
            padding: 1.25rem 1.35rem;
            margin: .8rem 0 1.2rem 0;
            background:
                linear-gradient(
                    135deg,
                    rgba(80,70,210,.07),
                    rgba(20,150,165,.04)
                );
        }

        .pub-kicker {
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
            opacity: .72;
            margin-bottom: .35rem;
        }

        .pub-title {
            font-size: 1.5rem;
            font-weight: 800;
            line-height: 1.22;
            margin-bottom: .5rem;
        }

        .pub-text {
            font-size: .96rem;
            line-height: 1.62;
            opacity: .90;
        }

        .evidence-layer {
            border: 1px solid rgba(120,120,120,.20);
            border-radius: 16px;
            padding: 1rem 1.05rem;
            min-height: 150px;
        }

        .evidence-layer b {
            font-size: 1.03rem;
        }

        .scientific-note {
            border-left: 4px solid rgba(70,120,210,.75);
            padding: .85rem 1rem;
            margin: 1rem 0;
            background: rgba(70,120,210,.06);
            border-radius: 0 12px 12px 0;
        }

        .hardware-note {
            border-left: 4px solid rgba(20,150,120,.75);
            padding: .85rem 1rem;
            margin: 1rem 0;
            background: rgba(20,150,120,.06);
            border-radius: 0 12px 12px 0;
        }
        </style>
        """)


# ============================================================
# DATA
# ============================================================

def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required publication evidence file not found:\n{path}"
        )
    return pd.read_csv(path)


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_publication_evidence():
    data = {
        "model_summary": _read_required(MODEL_SUMMARY_FILE),
        "device_results": _read_required(DEVICE_RESULTS_FILE),
        "pairwise": _read_required(PAIRWISE_FILE),
        "device_agreement": _read_optional(DEVICE_AGREEMENT_FILE),
        "seed_summary": _read_required(SEED_SUMMARY_FILE),
        "seed_robustness": _read_required(SEED_ROBUSTNESS_FILE),

        "hardware": _read_required(HW_STRATEGIC_FILE),
        "hardware_joint": _read_optional(HW_JOINT_FILE),
        "pareto_candidates": _read_required(HW_PARETO_CANDIDATES_FILE),
        "pareto_front": _read_required(HW_PARETO_FRONT_FILE),
        "hardware_summary": _read_optional(HW_SUMMARY_FILE),
    }

    # --------------------------------------------------------
    # Publication-evidence QA
    # --------------------------------------------------------

    assert len(data["model_summary"]) == 4
    assert len(data["device_results"]) == 40
    assert data["device_results"]["held_out_device"].nunique() == 10

    assert len(data["seed_summary"]) == 10
    assert set(data["seed_summary"]["seed"].astype(int)) == {
        0, 1, 7, 21, 42
    }

    assert len(data["hardware"]) == 32
    assert data["hardware"]["device_id"].nunique() == 4

    assert len(data["pareto_candidates"]) == 32
    assert len(data["pareto_front"]) == 20

    return data


# ============================================================
# HELPERS
# ============================================================

def _pretty_model(value):
    return MODEL_LABELS.get(str(value), str(value))


def _section(title, text=None):
    st.markdown(f"### {title}")
    if text:
        st.caption(text)


def _model_summary_display(df):
    out = df.copy()

    out["Model"] = out["model"].map(_pretty_model)

    keep = {
        "Model": "Model",
        "pooled_mae_pp": "MAE (pp)",
        "baseline_mean_regret_pp": "Baseline regret (pp)",
        "guarded_mean_regret_pp": "Guarded regret (pp)",
        "guarded_success_devices": "Guarded success",
        "target7_guarded_mean_regret_pp": "Target regret (pp)",
        "target7_guarded_success_devices": "Target success",
    }

    cols = ["Model"]

    for source, label in keep.items():
        if source == "Model":
            continue
        if source in out.columns:
            out[label] = out[source]
            cols.append(label)

    out = out[cols]

    if "Guarded success" in out:
        out["Guarded success"] = (
            out["Guarded success"].astype(int).astype(str) + "/10"
        )

    if "Target success" in out:
        out["Target success"] = (
            out["Target success"].astype(int).astype(str) + "/7"
        )

    return out


# ============================================================
# GRAPH 1 — BASELINE VS GUARDED REGRET
# ============================================================

def render_regret_graph(model_summary):
    rows = []

    for _, r in model_summary.iterrows():
        model = _pretty_model(r["model"])

        rows.append({
            "Model": model,
            "Policy": "Baseline",
            "Mean regret": float(r["baseline_mean_regret_pp"]),
        })

        rows.append({
            "Model": model,
            "Policy": "Support-gated",
            "Mean regret": float(r["guarded_mean_regret_pp"]),
        })

    df = pd.DataFrame(rows)

    spec = {
        "height": 360,
        "title": {
            "text": "Baseline vs support-gated recommendation regret",
            "subtitle": [
                "Lower is better. Near-optimal threshold = 0.5 percentage points."
            ],
        },
        "layer": [
            {
                "mark": {
                    "type": "bar",
                    "cornerRadiusTopLeft": 3,
                    "cornerRadiusTopRight": 3,
                },
                "encoding": {
                    "x": {
                        "field": "Model",
                        "type": "nominal",
                        "sort": [
                            "Random Forest",
                            "Extra Trees",
                            "Gradient Boosting",
                            "HistGradientBoosting",
                        ],
                        "axis": {
                            "title": None,
                            "labelAngle": 0,
                        },
                    },
                    "xOffset": {
                        "field": "Policy",
                    },
                    "y": {
                        "field": "Mean regret",
                        "type": "quantitative",
                        "axis": {
                            "title": "Mean recommendation regret (pp)"
                        },
                    },
                    "color": {
                        "field": "Policy",
                        "type": "nominal",
                        "legend": {
                            "title": None,
                            "orient": "top",
                        },
                    },
                    "tooltip": [
                        {"field": "Model"},
                        {"field": "Policy"},
                        {
                            "field": "Mean regret",
                            "format": ".3f",
                            "title": "Regret (pp)",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "rule",
                    "strokeDash": [6, 4],
                    "opacity": 0.75,
                },
                "encoding": {
                    "y": {
                        "datum": 0.5,
                        "type": "quantitative",
                    },
                },
            },
        ],
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


# ============================================================
# GRAPH 2 — MODEL SUCCESS
# ============================================================

def render_success_graph(model_summary):
    rows = []

    for _, r in model_summary.iterrows():
        model = _pretty_model(r["model"])

        rows.append({
            "Model": model,
            "Evaluation": "All held-out devices",
            "Success": float(r["guarded_success_pct"]),
        })

        rows.append({
            "Model": model,
            "Evaluation": "Target 7 devices",
            "Success": float(r["target7_guarded_success_pct"]),
        })

    df = pd.DataFrame(rows)

    spec = {
        "height": 340,
        "title": {
            "text": "Near-optimal recommendation success by model",
            "subtitle": [
                "Success means regret ≤ 0.5 percentage points."
            ],
        },
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 3,
            "cornerRadiusTopRight": 3,
        },
        "encoding": {
            "x": {
                "field": "Model",
                "type": "nominal",
                "sort": [
                    "Random Forest",
                    "Extra Trees",
                    "Gradient Boosting",
                    "HistGradientBoosting",
                ],
                "axis": {
                    "title": None,
                    "labelAngle": 0,
                },
            },
            "xOffset": {
                "field": "Evaluation",
            },
            "y": {
                "field": "Success",
                "type": "quantitative",
                "scale": {
                    "domain": [0, 100]
                },
                "axis": {
                    "title": "Near-optimal success (%)"
                },
            },
            "color": {
                "field": "Evaluation",
                "legend": {
                    "title": None,
                    "orient": "top",
                },
            },
            "tooltip": [
                {"field": "Model"},
                {"field": "Evaluation"},
                {
                    "field": "Success",
                    "format": ".1f",
                    "title": "Success (%)",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


# ============================================================
# GRAPH 3 — SEED ROBUSTNESS
# ============================================================

def render_seed_graph(seed_summary):
    df = seed_summary.copy()

    df = df[
        df["model"].isin(["RandomForest", "ExtraTrees"])
    ].copy()

    df["Model"] = df["model"].map(_pretty_model)
    df["Seed"] = df["seed"].astype(int)
    df["Target mean regret"] = df[
        "target7_mean_regret_pp"
    ].astype(float)

    spec = {
        "height": 330,
        "title": {
            "text": "Random-seed robustness",
            "subtitle": [
                "Target-device mean recommendation regret across five seeds."
            ],
        },
        "mark": {
            "type": "line",
            "point": {
                "filled": True,
                "size": 85,
            },
            "strokeWidth": 2.5,
        },
        "encoding": {
            "x": {
                "field": "Seed",
                "type": "ordinal",
                "axis": {
                    "title": "Random seed"
                },
            },
            "y": {
                "field": "Target mean regret",
                "type": "quantitative",
                "axis": {
                    "title": "Target mean regret (pp)"
                },
                "scale": {
                    "zero": False
                },
            },
            "color": {
                "field": "Model",
                "legend": {
                    "title": None,
                    "orient": "top",
                },
            },
            "tooltip": [
                {"field": "Model"},
                {"field": "Seed"},
                {
                    "field": "Target mean regret",
                    "format": ".4f",
                    "title": "Mean regret (pp)",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


# ============================================================
# GRAPH 4 — CROSS-MODEL AGREEMENT
# ============================================================

def render_agreement_graph(pairwise):
    df = pairwise.copy()

    df["Comparison"] = (
        df["model_a"].map(_pretty_model)
        + " vs "
        + df["model_b"].map(_pretty_model)
    )

    df["Exact agreement"] = df[
        "exact_config_agreement_pct"
    ].astype(float)

    df = df.sort_values(
        "Exact agreement",
        ascending=True,
    )

    spec = {
        "height": 350,
        "title": {
            "text": "Exact architecture agreement across models",
            "subtitle": [
                "Exact configuration agreement is stricter than near-optimal success."
            ],
        },
        "mark": {
            "type": "bar",
            "cornerRadiusEnd": 3,
        },
        "encoding": {
            "y": {
                "field": "Comparison",
                "type": "nominal",
                "sort": "-x",
                "axis": {
                    "title": None,
                },
            },
            "x": {
                "field": "Exact agreement",
                "type": "quantitative",
                "scale": {
                    "domain": [0, 100]
                },
                "axis": {
                    "title": "Exact configuration agreement (%)"
                },
            },
            "tooltip": [
                {"field": "Comparison"},
                {
                    "field": "Exact agreement",
                    "format": ".1f",
                    "title": "Exact agreement (%)",
                },
                {
                    "field": "both_success_devices",
                    "title": "Both successful devices",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


# ============================================================
# GRAPH 5 — HARDWARE PARETO
# ============================================================

def render_hardware_graph(candidates):
    df = candidates.copy()

    if "device_pareto_optimal" in df.columns:
        pareto_bool = (
            df["device_pareto_optimal"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    else:
        pareto_bool = pd.Series(
            False,
            index=df.index,
        )

    df["Design status"] = np.where(
        pareto_bool,
        "Pareto-optimal",
        "Evaluated candidate",
    )

    df["Device"] = df["device_id"].astype(str)

    tooltip = [
        {"field": "Device"},
        {"field": "Design status"},
        {
            "field": "latency_us",
            "format": ".3f",
            "title": "Latency (µs)",
        },
        {
            "field": "dynamic_energy_uJ",
            "format": ".3f",
            "title": "Dynamic energy (µJ)",
        },
    ]

    optional_tooltips = [
        ("chip_area_mm2", ".3f", "Area (mm²)"),
        (
            "energy_efficiency_TOPS_W",
            ".3f",
            "Energy efficiency (TOPS/W)",
        ),
        ("fps", ".1f", "FPS"),
        ("crossbar_size", None, "Crossbar"),
        ("requested_weight_bits", None, "Weight bits"),
        ("adc_bits", None, "ADC bits"),
    ]

    for field, fmt, title in optional_tooltips:
        if field in df.columns:
            item = {
                "field": field,
                "title": title,
            }
            if fmt is not None:
                item["format"] = fmt
            tooltip.append(item)

    spec = {
        "height": 430,
        "title": {
            "text": "NeuroSim hardware trade-off space",
            "subtitle": [
                "32 strategic circuit-level points; Pareto status shown explicitly."
            ],
        },
        "mark": {
            "type": "point",
            "filled": True,
            "size": 120,
        },
        "encoding": {
            "x": {
                "field": "latency_us",
                "type": "quantitative",
                "axis": {
                    "title": "Latency (µs)"
                },
            },
            "y": {
                "field": "dynamic_energy_uJ",
                "type": "quantitative",
                "axis": {
                    "title": "Dynamic energy (µJ)"
                },
            },
            "color": {
                "field": "Device",
                "type": "nominal",
                "legend": {
                    "title": "Device",
                    "orient": "right",
                },
            },
            "shape": {
                "field": "Design status",
                "type": "nominal",
                "legend": {
                    "title": None,
                },
            },
            "opacity": {
                "condition": {
                    "test": "datum['Design status'] === 'Pareto-optimal'",
                    "value": 1.0,
                },
                "value": 0.38,
            },
            "tooltip": tooltip,
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


# ============================================================
# MAIN WEBSITE COMPONENT
# ============================================================

def render_publication_results(mode="Beginner"):
    inject_publication_style()

    try:
        data = load_publication_evidence()
    except Exception as exc:
        st.error(
            "Final publication evidence could not be loaded."
        )
        st.code(str(exc))
        return

    model_summary = data["model_summary"]
    device_results = data["device_results"]
    seed_summary = data["seed_summary"]
    pairwise = data["pairwise"]
    hardware = data["hardware"]
    pareto_candidates = data["pareto_candidates"]
    pareto_front = data["pareto_front"]

    rf = model_summary[
        model_summary["model"] == "RandomForest"
    ].iloc[0]

    et = model_summary[
        model_summary["model"] == "ExtraTrees"
    ].iloc[0]

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    _html("""
        <div class="pub-shell">
            <div class="pub-kicker">Final frozen research evidence</div>
            <div class="pub-title">
                Publication-level validation and hardware evidence
            </div>
            <div class="pub-text">
                This section presents the frozen multi-model, seed-robustness,
                cross-model and NeuroSim evidence used for the final research
                interpretation. It is separate from exploratory diagnostics.
            </div>
        </div>
        """)

    # --------------------------------------------------------
    # Evidence base
    # --------------------------------------------------------

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Held-out devices",
        device_results["held_out_device"].nunique(),
    )

    k2.metric(
        "ML models evaluated",
        model_summary["model"].nunique(),
    )

    k3.metric(
        "Random seeds",
        seed_summary["seed"].nunique(),
    )

    k4.metric(
        "NeuroSim points",
        len(hardware),
    )

    st.caption(
        "The complete project contains 10 device profiles from 8 independent "
        "studies across 5 technology families and 2,450 device–configuration cases."
    )

    # --------------------------------------------------------
    # Main outcome
    # --------------------------------------------------------

    _section(
        "1. Final zero-shot recommendation performance",
        "Recommendation quality is evaluated on completely held-out devices "
        "using regret and a 0.5-percentage-point near-optimal threshold.",
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Random Forest",
        f"{int(rf['guarded_success_devices'])}/10",
        help="Held-out devices within the near-optimal threshold.",
    )

    r2.metric(
        "RF mean regret",
        f"{float(rf['guarded_mean_regret_pp']):.3f} pp",
    )

    r3.metric(
        "Extra Trees",
        f"{int(et['guarded_success_devices'])}/10",
        help="Held-out devices within the near-optimal threshold.",
    )

    r4.metric(
        "ET mean regret",
        f"{float(et['guarded_mean_regret_pp']):.3f} pp",
    )

    display = _model_summary_display(model_summary)

    st.dataframe(
        display.round(4),
        use_container_width=True,
        hide_index=True,
    )

    render_regret_graph(model_summary)

    _html("""
        <div class="scientific-note">
        <b>Interpretation:</b> support-aware guarding is highly effective for
        Random Forest and Extra Trees in the current dataset, but it is not a
        universal improvement for every regression model. HistGradientBoosting
        is an important counterexample.
        </div>
        """)

    render_success_graph(model_summary)

    # --------------------------------------------------------
    # Seed robustness
    # --------------------------------------------------------

    _section(
        "2. Random-seed robustness",
        "Random Forest and Extra Trees were repeated using seeds "
        "0, 1, 7, 21 and 42.",
    )

    s1, s2 = st.columns(2)

    with s1:
        st.metric(
            "RF target-device robustness",
            "7/7 × 5 seeds",
            help="All seven target devices were near-optimal for every tested seed.",
        )

    with s2:
        st.metric(
            "Extra Trees target robustness",
            "7/7 × 5 seeds",
            help="All seven target devices were near-optimal for every tested seed.",
        )

    render_seed_graph(seed_summary)

    # --------------------------------------------------------
    # Cross-model agreement
    # --------------------------------------------------------

    _section(
        "3. Cross-model agreement",
        "Exact architecture agreement is deliberately separated from "
        "near-optimal recommendation success.",
    )

    rf_et = pairwise[
        (
            (pairwise["model_a"] == "RandomForest")
            & (pairwise["model_b"] == "ExtraTrees")
        )
        |
        (
            (pairwise["model_a"] == "ExtraTrees")
            & (pairwise["model_b"] == "RandomForest")
        )
    ]

    if not rf_et.empty:
        row = rf_et.iloc[0]

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "RF ↔ ET exact agreement",
            f"{float(row['exact_config_agreement_pct']):.1f}%",
        )

        a2.metric(
            "Both successful",
            f"{int(row['both_success_devices'])}/10",
        )

        a3.metric(
            "Target both-success",
            f"{float(row['target7_both_success_pct']):.1f}%",
        )

    render_agreement_graph(pairwise)

    st.caption(
        "Two models can choose different exact accelerator configurations "
        "while both remain inside the near-optimal region."
    )

    # --------------------------------------------------------
    # Evidence layers
    # --------------------------------------------------------

    _section(
        "4. Three evidence layers",
        "These layers are complementary, but they must not be described "
        "as one end-to-end physical experiment.",
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        _html("""
            <div class="evidence-layer">
                <b>1 — Zero-shot ML</b><br><br>
                Held-out-device recommendation and regret.<br><br>
                <small>
                Evidence type: machine-learning prediction and decision quality.
                </small>
            </div>
            """)

    with e2:
        _html("""
            <div class="evidence-layer">
                <b>2 — Software accuracy</b><br><br>
                Quantized VGG8 / CIFAR-10 inference for selected configurations.
                <br><br>
                <small>
                Evidence type: software-level inference accuracy.
                </small>
            </div>
            """)

    with e3:
        _html("""
            <div class="evidence-layer">
                <b>3 — NeuroSim hardware</b><br><br>
                Area, latency, dynamic energy, throughput and efficiency.
                <br><br>
                <small>
                Evidence type: circuit/system-level hardware estimation.
                </small>
            </div>
            """)

    # --------------------------------------------------------
    # Hardware
    # --------------------------------------------------------

    _section(
        "5. Circuit-level hardware evidence",
        "The main hardware evidence contains 32 strategic NeuroSim points "
        "across four device profiles.",
    )

    h1, h2, h3, h4 = st.columns(4)

    h1.metric(
        "Strategic points",
        len(hardware),
    )

    h2.metric(
        "Hardware devices",
        hardware["device_id"].nunique(),
    )

    h3.metric(
        "Pareto-front points",
        len(pareto_front),
    )

    h4.metric(
        "Max observed TOPS/W",
        f"{hardware['energy_efficiency_TOPS_W'].max():.2f}",
    )

    render_hardware_graph(pareto_candidates)

    _html("""
        <div class="hardware-note">
        <b>Hardware evidence boundary:</b> these values are NeuroSim
        circuit/system-level estimates for the validated strategic hardware
        subset. They are not fabricated-chip measurements, and the 32 points
        do not directly validate every Stage-37 zero-shot recommendation.
        </div>
        """)

    if not data["hardware_summary"].empty:
        hw_table = data["hardware_summary"].copy()

        rename = {
            "device_id": "Device",
            "validated_hardware_points": "Validated points",
            "pareto_points": "Pareto points",
            "min_area_mm2": "Min area (mm²)",
            "min_latency_us": "Min latency (µs)",
            "min_dynamic_energy_uJ": "Min energy (µJ)",
            "max_energy_efficiency_TOPS_W": "Max TOPS/W",
            "max_throughput_TOPS": "Max TOPS",
            "max_fps": "Max FPS",
        }

        hw_table = hw_table.rename(columns=rename)

        st.dataframe(
            hw_table.round(4),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # Scientific boundary
    # --------------------------------------------------------

    _section("6. What these results establish")

    st.success(
        "The strongest supported result is evidence-aware, study-blocked "
        "cross-device transfer to near-optimal accelerator recommendations "
        "for held-out devices, with particularly robust results from Random "
        "Forest and Extra Trees."
    )

    st.warning(
        "The study does not establish universal memristor generalization, "
        "fabricated-chip performance, or direct NeuroSim validation of every "
        "final zero-shot recommendation."
    )

    # --------------------------------------------------------
    # Researcher details
    # --------------------------------------------------------

    if mode == "Researcher":
        st.divider()

        _section(
            "Publication evidence tables",
            "Frozen tables used to generate the final scientific results.",
        )

        with st.expander(
            "Model comparison — frozen Stage 37",
            expanded=False,
        ):
            st.dataframe(
                model_summary.round(5),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "Seed-level results",
            expanded=False,
        ):
            st.dataframe(
                seed_summary.round(5),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "Cross-model pairwise agreement",
            expanded=False,
        ):
            st.dataframe(
                pairwise.round(5),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander(
            "Hardware Pareto front",
            expanded=False,
        ):
            st.dataframe(
                pareto_front.round(5),
                use_container_width=True,
                hide_index=True,
            )

# ============================================================
# FINAL PUBLIC DASHBOARD V2
# ============================================================


def render_model_decision_graph(model_summary):
    df = model_summary.copy()

    df["Model"] = df["model"].map(_pretty_model)
    df["Prediction MAE"] = df["pooled_mae_pp"].astype(float)
    df["Decision regret"] = df["guarded_mean_regret_pp"].astype(float)
    df["Success"] = df["guarded_success_pct"].astype(float)

    spec = {
        "height": 390,
        "title": {
            "text": "Prediction error vs design-decision quality",
            "subtitle": [
                "A model can have similar prediction error yet very different recommendation regret."
            ],
        },
        "layer": [
            {
                "mark": {
                    "type": "point",
                    "filled": True,
                    "size": 180,
                },
                "encoding": {
                    "x": {
                        "field": "Prediction MAE",
                        "type": "quantitative",
                        "axis": {
                            "title": "Pooled prediction MAE (pp)"
                        },
                        "scale": {
                            "zero": False
                        },
                    },
                    "y": {
                        "field": "Decision regret",
                        "type": "quantitative",
                        "axis": {
                            "title": "Guarded mean recommendation regret (pp)"
                        },
                    },
                    "color": {
                        "field": "Model",
                        "type": "nominal",
                        "legend": {
                            "title": None,
                            "orient": "right",
                        },
                    },
                    "tooltip": [
                        {"field": "Model"},
                        {
                            "field": "Prediction MAE",
                            "format": ".3f",
                            "title": "MAE (pp)",
                        },
                        {
                            "field": "Decision regret",
                            "format": ".3f",
                            "title": "Regret (pp)",
                        },
                        {
                            "field": "Success",
                            "format": ".1f",
                            "title": "Success (%)",
                        },
                    ],
                },
            },
            {
                "mark": {
                    "type": "rule",
                    "strokeDash": [6, 4],
                    "opacity": .65,
                },
                "encoding": {
                    "y": {
                        "datum": 0.5,
                        "type": "quantitative",
                    }
                },
            },
        ],
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


def render_hardware_device_comparison(hardware_summary):
    if hardware_summary.empty:
        return

    df = hardware_summary.copy()

    df["Device"] = df["device_id"].astype(str)

    df["Max energy efficiency"] = df[
        "max_energy_efficiency_TOPS_W"
    ].astype(float)

    spec = {
        "height": 330,
        "title": {
            "text": "Best observed energy efficiency by hardware device",
            "subtitle": [
                "Maximum TOPS/W among the eight strategic NeuroSim points evaluated for each device."
            ],
        },
        "mark": {
            "type": "bar",
            "cornerRadiusTopLeft": 4,
            "cornerRadiusTopRight": 4,
        },
        "encoding": {
            "x": {
                "field": "Device",
                "type": "nominal",
                "axis": {
                    "title": None,
                    "labelAngle": 0,
                },
            },
            "y": {
                "field": "Max energy efficiency",
                "type": "quantitative",
                "axis": {
                    "title": "Maximum energy efficiency (TOPS/W)"
                },
            },
            "color": {
                "field": "Device",
                "legend": None,
            },
            "tooltip": [
                {"field": "Device"},
                {
                    "field": "Max energy efficiency",
                    "format": ".4f",
                    "title": "TOPS/W",
                },
            ],
        },
    }

    st.vega_lite_chart(
        df,
        spec,
        use_container_width=True,
    )


def _question_card(question, answer):
    _html(f"""
        <div style="
            border:1px solid rgba(120,120,140,.16);
            border-radius:16px;
            padding:1rem 1.1rem;
            margin:.35rem 0 1rem 0;
            background:rgba(120,120,140,.035);
        ">
            <div style="
                font-size:.72rem;
                text-transform:uppercase;
                letter-spacing:.08em;
                opacity:.62;
                font-weight:800;
                margin-bottom:.35rem;
            ">
                Question this graph answers
            </div>

            <div style="
                font-weight:800;
                font-size:1rem;
                margin-bottom:.3rem;
            ">
                {question}
            </div>

            <div style="
                font-size:.9rem;
                line-height:1.55;
                opacity:.82;
            ">
                {answer}
            </div>
        </div>
        """)


def render_publication_results_v2(mode="Beginner"):
    inject_publication_style()

    try:
        data = load_publication_evidence()
    except Exception as exc:
        st.error(
            "Final publication evidence could not be loaded."
        )
        st.code(str(exc))
        return

    model_summary = data["model_summary"]
    device_results = data["device_results"]
    pairwise = data["pairwise"]
    seed_summary = data["seed_summary"]

    hardware = data["hardware"]
    candidates = data["pareto_candidates"]
    pareto_front = data["pareto_front"]
    hardware_summary = data["hardware_summary"]

    rf = model_summary[
        model_summary["model"] == "RandomForest"
    ].iloc[0]

    et = model_summary[
        model_summary["model"] == "ExtraTrees"
    ].iloc[0]

    gb = model_summary[
        model_summary["model"] == "GradientBoosting"
    ].iloc[0]

    hgb = model_summary[
        model_summary["model"] == "HistGradientBoosting"
    ].iloc[0]

    devices = int(
        device_results["held_out_device"].nunique()
    )

    studies = int(
        device_results["held_out_study"].nunique()
    )

    configs_per_device = int(
        device_results["test_rows"].iloc[0]
    )

    design_cases = devices * configs_per_device

    seeds = int(
        seed_summary["seed"].nunique()
    )

    # --------------------------------------------------------
    # Main header
    # --------------------------------------------------------

    _html("""
        <div class="pub-shell">

            <div class="pub-kicker">
                Frozen final research evidence
            </div>

            <div class="pub-title">
                From prediction quality
                to hardware trade-offs
            </div>

            <div class="pub-text">
                The final evaluation combines multi-model zero-shot
                recommendation, random-seed robustness, cross-model
                agreement and a separate strategic NeuroSim hardware study.
                Each evidence layer is shown independently to avoid
                overstating what was physically validated.
            </div>

        </div>
        """)

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Held-out devices",
        devices,
    )

    m2.metric(
        "Design cases",
        f"{design_cases:,}",
    )

    m3.metric(
        "ML models",
        len(model_summary),
    )

    m4.metric(
        "Random seeds",
        seeds,
    )

    st.caption(
        f"{studies} independent source studies • "
        f"{len(hardware)} strategic NeuroSim points • "
        f"{len(pareto_front)} hardware Pareto-front points"
    )

    st.write("")

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    (
        tab_overview,
        tab_models,
        tab_robustness,
        tab_hardware,
        tab_boundary,
    ) = st.tabs(
        [
            "Overview",
            "Model comparison",
            "Robustness",
            "Hardware evidence",
            "Scientific boundary",
        ]
    )

    # ========================================================
    # OVERVIEW
    # ========================================================

    with tab_overview:

        st.markdown("### Headline recommendation result")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Random Forest",
            f"{int(rf['guarded_success_devices'])}/10",
            f"{float(rf['guarded_mean_regret_pp']):.3f} pp",
        )

        c2.metric(
            "Extra Trees",
            f"{int(et['guarded_success_devices'])}/10",
            f"{float(et['guarded_mean_regret_pp']):.3f} pp",
        )

        c3.metric(
            "Gradient Boosting",
            f"{int(gb['guarded_success_devices'])}/10",
            f"{float(gb['guarded_mean_regret_pp']):.3f} pp",
        )

        c4.metric(
            "HistGradientBoosting",
            f"{int(hgb['guarded_success_devices'])}/10",
            f"{float(hgb['guarded_mean_regret_pp']):.3f} pp",
        )

        st.caption(
            "Top number = held-out devices inside the 0.5-pp "
            "near-optimal threshold. Delta line = guarded mean regret."
        )

        st.success(
            "Random Forest and Extra Trees produced near-optimal "
            "recommendations for all 10 held-out devices in the final "
            "evaluation."
        )

        st.write("")

        st.markdown("### Evidence architecture")

        e1, e2, e3 = st.columns(3)

        with e1:
            _html("""
                <div class="evidence-layer">
                    <b>① Zero-shot ML recommendation</b>
                    <br><br>
                    Held-out-device prediction, support gating,
                    regret and recommendation success.
                    <br><br>
                    <small>
                    Evidence: machine-learning decision quality
                    </small>
                </div>
                """)

        with e2:
            _html("""
                <div class="evidence-layer">
                    <b>② Quantized software accuracy</b>
                    <br><br>
                    VGG8 / CIFAR-10 software inference for selected
                    configurations.
                    <br><br>
                    <small>
                    Evidence: software-level model accuracy
                    </small>
                </div>
                """)

        with e3:
            _html("""
                <div class="evidence-layer">
                    <b>③ NeuroSim hardware estimates</b>
                    <br><br>
                    Area, latency, dynamic energy, throughput,
                    FPS and energy efficiency.
                    <br><br>
                    <small>
                    Evidence: circuit/system-level estimates
                    </small>
                </div>
                """)

        st.info(
            "These three layers support the same research story, but "
            "they are not one end-to-end measured hardware experiment."
        )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    with tab_models:

        st.markdown(
            "### Does the result depend on one convenient ML model?"
        )

        _question_card(
            "Does support-aware guarding improve every model?",
            (
                "No. It is highly effective for Random Forest and "
                "Extra Trees, improves decision success for Gradient "
                "Boosting, but HistGradientBoosting remains an important "
                "counterexample. This prevents an overly broad claim."
            ),
        )

        render_regret_graph(
            model_summary
        )

        _question_card(
            "Which models actually return near-optimal designs?",
            (
                "Random Forest and Extra Trees reach 10/10 guarded "
                "success. Gradient Boosting reaches 9/10, while "
                "HistGradientBoosting reaches 6/10."
            ),
        )

        render_success_graph(
            model_summary
        )

        _question_card(
            "Is low regression error the same as good design selection?",
            (
                "No. Recommendation regret evaluates the actual design "
                "decision. The graph below separates prediction-surface "
                "error from decision quality."
            ),
        )

        render_model_decision_graph(
            model_summary
        )

        st.markdown("### Final comparison table")

        st.dataframe(
            _model_summary_display(
                model_summary
            ).round(4),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # ROBUSTNESS
    # ========================================================

    with tab_robustness:

        st.markdown(
            "### Does the recommendation remain stable when randomness changes?"
        )

        r1, r2 = st.columns(2)

        r1.metric(
            "Random Forest target success",
            "7/7 × 5 seeds",
        )

        r2.metric(
            "Extra Trees target success",
            "7/7 × 5 seeds",
        )

        _question_card(
            "Does a different random seed change the conclusion?",
            (
                "Across seeds 0, 1, 7, 21 and 42, both leading "
                "models retained near-optimal success for all seven "
                "target devices."
            ),
        )

        render_seed_graph(
            seed_summary
        )

        st.divider()

        st.markdown(
            "### Do independent models choose the same design?"
        )

        rf_et = pairwise[
            (
                (pairwise["model_a"] == "RandomForest")
                & (pairwise["model_b"] == "ExtraTrees")
            )
            |
            (
                (pairwise["model_a"] == "ExtraTrees")
                & (pairwise["model_b"] == "RandomForest")
            )
        ]

        if not rf_et.empty:
            row = rf_et.iloc[0]

            a1, a2, a3 = st.columns(3)

            a1.metric(
                "RF ↔ ET exact match",
                f"{float(row['exact_config_agreement_pct']):.1f}%",
            )

            a2.metric(
                "Both near-optimal",
                f"{int(row['both_success_devices'])}/10",
            )

            a3.metric(
                "Target both-success",
                f"{float(row['target7_both_success_pct']):.1f}%",
            )

        _question_card(
            "Why can exact agreement be below 100% while both models succeed?",
            (
                "The near-optimal region can contain more than one useful "
                "accelerator configuration. Exact architecture equality is "
                "therefore a stricter requirement than successful design "
                "recommendation."
            ),
        )

        render_agreement_graph(
            pairwise
        )

    # ========================================================
    # HARDWARE
    # ========================================================

    with tab_hardware:

        st.markdown(
            "### What happens when selected designs are evaluated at circuit/system level?"
        )

        h1, h2, h3, h4 = st.columns(4)

        h1.metric(
            "Strategic points",
            len(hardware),
        )

        h2.metric(
            "Hardware devices",
            hardware["device_id"].nunique(),
        )

        h3.metric(
            "Pareto-front points",
            len(pareto_front),
        )

        h4.metric(
            "Max observed efficiency",
            f"{hardware['energy_efficiency_TOPS_W'].max():.2f} TOPS/W",
        )

        _question_card(
            "Is there one universally best accelerator configuration?",
            (
                "No. The 32-point hardware study shows trade-offs among "
                "latency, dynamic energy, area, throughput and efficiency. "
                "The Pareto front represents designs for which one objective "
                "cannot be improved without worsening another."
            ),
        )

        render_hardware_graph(
            candidates
        )

        if not hardware_summary.empty:

            _question_card(
                "Do device technologies produce different hardware operating regions?",
                (
                    "Yes. The evaluated device profiles reach different "
                    "best observed energy-efficiency values under the "
                    "strategic NeuroSim configurations."
                ),
            )

            render_hardware_device_comparison(
                hardware_summary
            )

            st.markdown(
                "### Hardware comparison table"
            )

            hw_table = hardware_summary.rename(
                columns={
                    "device_id": "Device",
                    "validated_hardware_points": "Points",
                    "pareto_points": "Pareto",
                    "min_area_mm2": "Min area (mm²)",
                    "min_latency_us": "Min latency (µs)",
                    "min_dynamic_energy_uJ": "Min energy (µJ)",
                    "max_energy_efficiency_TOPS_W": "Max TOPS/W",
                    "max_throughput_TOPS": "Max TOPS",
                    "max_fps": "Max FPS",
                }
            )

            st.dataframe(
                hw_table.round(4),
                use_container_width=True,
                hide_index=True,
            )

        st.warning(
            "Hardware metrics are NeuroSim circuit/system-level estimates "
            "for the validated 32-point subset. They are not fabricated-chip "
            "measurements and do not directly validate every Stage-37 "
            "zero-shot recommendation."
        )

    # ========================================================
    # SCIENTIFIC BOUNDARY
    # ========================================================

    with tab_boundary:

        st.markdown(
            "### What the current research supports"
        )

        st.success(
            "Evidence-aware, study-blocked cross-device learning can "
            "recommend near-optimal accelerator configurations for the "
            "held-out devices in the current literature-derived dataset, "
            "with particularly robust performance from Random Forest and "
            "Extra Trees."
        )

        st.markdown(
            "### What the current research does not establish"
        )

        st.markdown(
            """
            - Universal generalization to all memristor technologies.
            - Fabricated-chip validation.
            - Measured energy, latency or area.
            - Direct NeuroSim validation of every final zero-shot recommendation.
            - Equivalence between software inference accuracy and hardware-measured accuracy.
            - Replacement of NeuroSim, SPICE or detailed EDA.
            """
        )

        st.markdown(
            "### Publication positioning"
        )

        st.info(
            "The strongest contribution is the integrated problem formulation: "
            "device evidence → study-blocked cross-device learning → unseen "
            "device recommendation → support/extrapolation handling → "
            "near-optimal decision evaluation → separate circuit-level "
            "hardware trade-off evidence."
        )

        if mode == "Researcher":

            st.divider()

            st.markdown(
                "### Frozen evidence tables"
            )

            with st.expander(
                "Stage 37 — model comparison",
                expanded=False,
            ):
                st.dataframe(
                    model_summary.round(5),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander(
                "Stage 37 — seed-level robustness",
                expanded=False,
            ):
                st.dataframe(
                    seed_summary.round(5),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander(
                "Stage 37 — cross-model agreement",
                expanded=False,
            ):
                st.dataframe(
                    pairwise.round(5),
                    use_container_width=True,
                    hide_index=True,
                )

            with st.expander(
                "Stage 35 — hardware Pareto front",
                expanded=False,
            ):
                st.dataframe(
                    pareto_front.round(5),
                    use_container_width=True,
                    hide_index=True,
                )
