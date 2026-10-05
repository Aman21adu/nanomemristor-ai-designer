from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


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
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )


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

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown(
        """
        <div class="scientific-note">
        <b>Interpretation:</b> support-aware guarding is highly effective for
        Random Forest and Extra Trees in the current dataset, but it is not a
        universal improvement for every regression model. HistGradientBoosting
        is an important counterexample.
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.markdown(
            """
            <div class="evidence-layer">
                <b>1 — Zero-shot ML</b><br><br>
                Held-out-device recommendation and regret.<br><br>
                <small>
                Evidence type: machine-learning prediction and decision quality.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            """
            <div class="evidence-layer">
                <b>2 — Software accuracy</b><br><br>
                Quantized VGG8 / CIFAR-10 inference for selected configurations.
                <br><br>
                <small>
                Evidence type: software-level inference accuracy.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e3:
        st.markdown(
            """
            <div class="evidence-layer">
                <b>3 — NeuroSim hardware</b><br><br>
                Area, latency, dynamic energy, throughput and efficiency.
                <br><br>
                <small>
                Evidence type: circuit/system-level hardware estimation.
                </small>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    st.markdown(
        """
        <div class="hardware-note">
        <b>Hardware evidence boundary:</b> these values are NeuroSim
        circuit/system-level estimates for the validated strategic hardware
        subset. They are not fabricated-chip measurements, and the 32 points
        do not directly validate every Stage-37 zero-shot recommendation.
        </div>
        """,
        unsafe_allow_html=True,
    )

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
