from __future__ import annotations

from textwrap import dedent

import streamlit as st


def _html(content: str):
    """Render trusted local UI HTML without Markdown code-block parsing."""
    st.html(dedent(content).strip())


from publication_evidence import load_publication_evidence


def inject_final_public_style():
    _html("""
        <style>

        /* --------------------------------------------------
           Global spacing
        -------------------------------------------------- */

        .block-container {
            max-width: 1420px;
            padding-top: 1.5rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.025em;
        }

        /* --------------------------------------------------
           Main public hero
        -------------------------------------------------- */

        .final-hero {
            position: relative;
            overflow: hidden;
            border-radius: 28px;
            padding: 2.6rem 2.7rem 2.35rem 2.7rem;
            margin: .4rem 0 1.4rem 0;

            border:
                1px solid rgba(120, 120, 150, .22);

            background:
                radial-gradient(
                    circle at 92% 12%,
                    rgba(66, 208, 255, .18),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 10% 80%,
                    rgba(139, 92, 246, .16),
                    transparent 34%
                ),
                linear-gradient(
                    135deg,
                    rgba(74, 66, 180, .12),
                    rgba(10, 150, 170, .07)
                );
        }

        .final-badge {
            display: inline-block;
            padding: .35rem .72rem;
            border-radius: 999px;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;

            background: rgba(102, 88, 230, .12);
            border: 1px solid rgba(102, 88, 230, .28);

            margin-bottom: 1rem;
        }

        .final-title {
            font-size: clamp(2.25rem, 5vw, 4rem);
            line-height: 1.04;
            font-weight: 850;
            max-width: 1000px;

            letter-spacing: -.045em;
        }

        .final-subtitle {
            margin-top: 1rem;
            max-width: 900px;

            font-size: 1.12rem;
            line-height: 1.65;
            opacity: .84;
        }

        .flow-strip {
            margin-top: 1.6rem;

            display: flex;
            gap: .65rem;
            flex-wrap: wrap;
            align-items: center;
        }

        .flow-node {
            padding: .55rem .85rem;
            border-radius: 12px;

            background: rgba(120,120,140,.08);
            border: 1px solid rgba(120,120,140,.18);

            font-weight: 750;
            font-size: .88rem;
        }

        .flow-arrow {
            opacity: .55;
            font-weight: 900;
        }

        /* --------------------------------------------------
           Public explanation cards
        -------------------------------------------------- */

        .public-card {
            border-radius: 18px;

            border:
                1px solid rgba(120,120,140,.18);

            padding: 1.15rem 1.2rem;
            min-height: 175px;

            background:
                linear-gradient(
                    145deg,
                    rgba(125,125,150,.055),
                    rgba(125,125,150,.018)
                );
        }

        .public-card-kicker {
            font-size: .69rem;
            text-transform: uppercase;
            letter-spacing: .09em;
            font-weight: 850;
            opacity: .60;
            margin-bottom: .45rem;
        }

        .public-card-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: .45rem;
        }

        .public-card-text {
            font-size: .91rem;
            line-height: 1.55;
            opacity: .82;
        }

        /* --------------------------------------------------
           Publication cards
        -------------------------------------------------- */

        .pub-position {
            border-radius: 20px;
            padding: 1.3rem 1.4rem;

            border:
                1px solid rgba(120,120,140,.18);

            background:
                linear-gradient(
                    135deg,
                    rgba(35,145,165,.07),
                    rgba(100,75,200,.05)
                );

            margin: .8rem 0 1.2rem 0;
        }

        .pub-position-title {
            font-size: 1.15rem;
            font-weight: 820;
            margin-bottom: .4rem;
        }

        .pub-position-text {
            font-size: .94rem;
            line-height: 1.62;
            opacity: .86;
        }

        /* --------------------------------------------------
           Streamlit metrics
        -------------------------------------------------- */

        div[data-testid="stMetric"] {
            border:
                1px solid rgba(120,120,140,.14);

            border-radius: 16px;
            padding: .8rem 1rem;

            background:
                rgba(120,120,140,.035);
        }

        /* --------------------------------------------------
           Tabs
        -------------------------------------------------- */

        button[data-baseweb="tab"] {
            font-weight: 720;
        }

        /* --------------------------------------------------
           Sidebar
        -------------------------------------------------- */

        section[data-testid="stSidebar"] {
            border-right:
                1px solid rgba(120,120,140,.12);
        }

        </style>
        """)





# GLOBAL NO-TRUNCATION STYLE
def inject_no_truncation_style():
    _html("""
    <style>

    /* ======================================================
       NEVER SHOW ... IN METRIC CARDS
       Wrap complete words instead.
       ====================================================== */

    div[data-testid="stMetric"] {
        min-width: 0 !important;
        overflow: visible !important;
        height: auto !important;
    }

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] *,
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        max-width: 100% !important;
    }

    div[data-testid="stMetricLabel"] p {
        line-height: 1.25 !important;
        min-height: 1.5rem !important;
    }

    div[data-testid="stMetricValue"] {
        line-height: 1.08 !important;
    }

    /*
       Same rule for our custom cards.
       Never intentionally shorten text with ellipsis.
    */

    .public-card,
    .public-card *,
    .pub-position,
    .pub-position *,
    .workflow-card,
    .workflow-card *,
    .why-card,
    .why-card *,
    .info-card,
    .info-card *,
    .callout,
    .callout * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
    }

    </style>
    """)


def render_home_hero(
    device_count: int,
    study_count: int,
    family_count: int,
    total_simulation_rows: int,
):
    data = load_publication_evidence()

    model_summary = data["model_summary"]
    seed_summary = data["seed_summary"]
    hardware = data["hardware"]
    pareto_front = data["pareto_front"]

    rf = model_summary[
        model_summary["model"] == "RandomForest"
    ].iloc[0]

    et = model_summary[
        model_summary["model"] == "ExtraTrees"
    ].iloc[0]

    # ============================================================
    # HERO
    # ============================================================

    _html("""
    <div class="final-hero">

        <div class="final-badge">
            RESEARCH PROTOTYPE • NANOTECHNOLOGY + AI + CHIP DESIGN
        </div>

        <div class="final-title">
            From a memristor device<br>
            to an accelerator design recommendation.
        </div>

        <div class="final-subtitle">
            NanoMemristor AI Designer uses evidence from characterized
            memristor technologies to screen accelerator design choices
            for a held-out or new device — then checks whether the
            recommendation is sufficiently supported before detailed
            hardware validation.
        </div>

        <div class="flow-strip">
            <span class="flow-node">Device evidence</span>
            <span class="flow-arrow">→</span>

            <span class="flow-node">Cross-device AI</span>
            <span class="flow-arrow">→</span>

            <span class="flow-node">Support check</span>
            <span class="flow-arrow">→</span>

            <span class="flow-node">Prioritize</span>
            <span class="flow-arrow">→</span>

            <span class="flow-node">Validate</span>
        </div>

    </div>
    """)

    # ============================================================
    # DATA SNAPSHOT
    # ============================================================

    st.markdown("### Research evidence at a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Device profiles",
        device_count,
        help="Literature-derived simulation-ready memristor profiles.",
    )

    c2.metric(
        "Independent studies",
        study_count,
    )

    c3.metric(
        "Device × design cases",
        f"{total_simulation_rows:,}",
        help="10 devices × 245 accelerator configurations.",
    )

    c4.metric(
        "ML models evaluated",
        model_summary["model"].nunique(),
    )

    c5, c6, c7 = st.columns(3)

    c5.metric(
        "Random seeds",
        seed_summary["seed"].nunique(),
    )

    c6.metric(
        "NeuroSim points",
        len(hardware),
    )

    c7.metric(
        "Hardware Pareto points",
        len(pareto_front),
    )

    st.caption(
        f"{family_count} technology families represented in the current evidence base."
    )

    st.divider()

    # ============================================================
    # PROBLEM → APPROACH → OUTPUT
    # ============================================================

    st.markdown("### Why the problem matters")

    p1, p2, p3 = st.columns(3)

    with p1:
        _html("""
        <div class="public-card">
            <div class="public-card-kicker">01 • Problem</div>

            <div class="public-card-title">
                A better memristor is not automatically
                a better AI accelerator.
            </div>

            <div class="public-card-text">
                Device behavior changes the accelerator design space.
                Crossbar size, weight precision and ADC precision can
                become better or worse depending on the device.
            </div>
        </div>
        """)

    with p2:
        _html("""
        <div class="public-card">
            <div class="public-card-kicker">02 • Approach</div>

            <div class="public-card-title">
                Cross-device AI + support check
            </div>

            <div class="public-card-text">
                Learn transferable relationships from characterized
                devices, leave the target device out of training,
                generate a recommendation, then check whether the
                target lies inside supported evidence space.
            </div>
        </div>
        """)

    with p3:
        _html("""
        <div class="public-card">
            <div class="public-card-kicker">03 • Decision</div>

            <div class="public-card-title">
                Decide what deserves detailed validation next.
            </div>

            <div class="public-card-text">
                The system proposes a promising accelerator starting
                configuration for screening. Detailed simulators and
                EDA remain the final validation layer.
            </div>
        </div>
        """)

    st.divider()

    # ============================================================
    # HEADLINE RESULTS
    # ============================================================

    st.markdown("### Final recommendation results")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Random Forest",
        f"{int(rf['guarded_success_devices'])}/10",
        f"{float(rf['guarded_mean_regret_pp']):.3f} pp regret",
        help="Held-out devices within the 0.5 percentage-point near-optimal threshold.",
    )

    r2.metric(
        "Extra Trees",
        f"{int(et['guarded_success_devices'])}/10",
        f"{float(et['guarded_mean_regret_pp']):.3f} pp regret",
        help="Held-out devices within the 0.5 percentage-point near-optimal threshold.",
    )

    r3.metric(
        "Seed robustness",
        "7/7 × 5",
        "target devices",
        help="Near-optimal target-device success across five tested random seeds.",
    )

    r4.metric(
        "Hardware evidence",
        f"{len(hardware)} points",
        f"{len(pareto_front)} Pareto",
        help="Strategic NeuroSim circuit/system-level evaluations.",
    )

    st.caption(
        "Near-optimal = recommendation regret ≤ 0.5 percentage points "
        "in the current study-blocked evaluation."
    )

    st.divider()

    # ============================================================
    # PRODUCT / RESEARCH POSITIONING
    # ============================================================

    st.markdown("### Where NanoMemristor AI Designer fits")

    a, b = st.columns([1.15, 1])

    with a:
        _html("""
        <div class="pub-position">

            <div class="public-card-kicker">
                SCREEN → PRIORITIZE → VALIDATE
            </div>

            <div class="pub-position-title">
                Reduce the search space before expensive validation.
            </div>

            <div class="pub-position-text">
                The goal is not to replace detailed circuit simulators.
                It is to help researchers identify which device–architecture
                combinations deserve deeper simulation or EDA-level
                validation first.
            </div>

        </div>
        """)

    with b:
        st.markdown(
            """
            **Use this site to:**

            - check a known memristor profile,
            - explore a custom device,
            - reverse-search desirable device properties,
            - inspect research gaps,
            - compare ML robustness,
            - inspect NeuroSim hardware trade-offs,
            - review provenance and limitations.
            """
        )

    st.info(
        "Scientific boundary: zero-shot recommendations are ML predictions; "
        "VGG8/CIFAR-10 results are software accuracy; NeuroSim values are "
        "circuit/system-level estimates. They are complementary evidence layers, "
        "not one fabricated-chip experiment."
    )



def render_publication_positioning():
    _html("""
        <div class="pub-position">

            <div class="public-card-kicker">
                Publication contribution
            </div>

            <div class="pub-position-title">
                Evidence-aware cross-device memristor-to-accelerator co-design
            </div>

            <div class="pub-position-text">
                The research evaluates whether information learned from
                characterized memristor technologies can transfer to
                near-optimal accelerator recommendations for a previously
                unseen device under study-blocked validation, while explicitly
                handling evidence provenance, support/extrapolation,
                multi-model robustness and separate circuit-level
                hardware evidence.
            </div>

        </div>
        """)

    c1, c2, c3 = st.columns(3)

    with c1:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">New contribution</div>
                <div class="public-card-title">
                    Cross-device zero-shot recommendation
                </div>
                <div class="public-card-text">
                    A target device/study is excluded from training and
                    receives an accelerator recommendation from knowledge
                    learned from other technologies.
                </div>
            </div>
            """)

    with c2:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">Existing technology</div>
                <div class="public-card-title">
                    Detailed simulator remains essential
                </div>
                <div class="public-card-text">
                    NeuroSim and related tools remain the circuit/system
                    validation layer. The contribution is not a replacement
                    simulator.
                </div>
            </div>
            """)

    with c3:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">Scientific status</div>
                <div class="public-card-title">
                    Research prototype
                </div>
                <div class="public-card-text">
                    Results are computational and literature-grounded.
                    Hardware values are simulation estimates, not fabricated
                    chip measurements.
                </div>
            </div>
            """)


# ============================================================
# PUBLICATION + SOURCES V2
# ============================================================

def render_publication_positioning_v2():

    data = load_publication_evidence()

    model_summary = data["model_summary"]
    device_results = data["device_results"]
    seed_summary = data["seed_summary"]
    hardware = data["hardware"]
    pareto_front = data["pareto_front"]

    rf = model_summary[
        model_summary["model"] == "RandomForest"
    ].iloc[0]

    et = model_summary[
        model_summary["model"] == "ExtraTrees"
    ].iloc[0]


    # ========================================================
    # CONTRIBUTION
    # ========================================================

    st.markdown("### Research contribution")

    st.info(
        "Evidence-aware, study-blocked zero-shot transfer from memristor "
        "device descriptors to near-optimal accelerator configurations for "
        "previously unseen devices, with validation-aware support handling, "
        "cross-model robustness and separate circuit-level hardware evidence."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("#### Contribution")
            st.markdown(
                """
                **Cross-device device → architecture recommendation**

                The target device is excluded from training, and the model
                must recommend an accelerator configuration using evidence
                learned from other characterized technologies.
                """
            )

    with c2:
        with st.container(border=True):
            st.markdown("#### Evidence-aware safeguard")
            st.markdown(
                """
                **Support before aggressive optimization**

                The workflow checks whether the target is sufficiently
                represented before trusting cost-oriented selection.
                Unsupported regions use a conservative accuracy-first fallback.
                """
            )

    with c3:
        with st.container(border=True):
            st.markdown("#### Hardware connection")
            st.markdown(
                """
                **ML recommendation + separate circuit evidence**

                Strategic configurations are evaluated with NeuroSim to
                expose latency, energy, area, throughput and efficiency
                trade-offs without claiming fabricated-chip validation.
                """
            )


    st.divider()


    # ========================================================
    # WHAT IS NEW VS WHAT IS USED
    # ========================================================

    st.markdown("### What is the contribution — and what is not claimed as new?")

    n1, n2 = st.columns(2)

    with n1:
        with st.container(border=True):
            st.markdown("#### Contribution of this research")

            st.markdown(
                """
                - study-blocked unseen-device transfer
                - memristor descriptors → accelerator recommendation
                - validation-aware support handling
                - recommendation-quality evaluation using regret
                - multi-model and multi-seed robustness
                - separate circuit/system hardware evidence
                """
            )

    with n2:
        with st.container(border=True):
            st.markdown("#### Existing methods used as building blocks")

            st.markdown(
                """
                - Random Forest / Extra Trees / boosting regressors
                - quantized VGG8 / CIFAR-10 software evaluation
                - Pareto analysis
                - NeuroSim circuit/system simulation
                - standard regression metrics such as MAE and RMSE
                """
            )

    st.caption(
        "The research contribution is the integrated evidence-aware "
        "cross-device co-design formulation and validation framework — "
        "not a claim that the individual ML algorithms or NeuroSim are new."
    )


    st.divider()


    # ========================================================
    # FINAL EVIDENCE SNAPSHOT
    # ========================================================

    st.markdown("### Final publication evidence")

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Held-out devices",
        device_results["held_out_device"].nunique(),
    )

    p2.metric(
        "ML models",
        model_summary["model"].nunique(),
    )

    p3.metric(
        "Random seeds",
        seed_summary["seed"].nunique(),
    )

    p4.metric(
        "NeuroSim points",
        len(hardware),
    )

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Random Forest",
        f"{int(rf['guarded_success_devices'])}/10",
        f"{float(rf['guarded_mean_regret_pp']):.3f} pp regret",
    )

    r2.metric(
        "Extra Trees",
        f"{int(et['guarded_success_devices'])}/10",
        f"{float(et['guarded_mean_regret_pp']):.3f} pp regret",
    )

    r3.metric(
        "Hardware Pareto points",
        len(pareto_front),
    )

    r4.metric(
        "Target robustness",
        "7/7 × 5 seeds",
    )


    st.divider()


    # ========================================================
    # EVIDENCE CHAIN
    # ========================================================

    st.markdown("### Evidence chain")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        with st.container(border=True):
            st.markdown("#### ① Device evidence")
            st.caption(
                "Literature-derived memristor profiles with source, "
                "stack, descriptors and provenance."
            )

    with e2:
        with st.container(border=True):
            st.markdown("#### ② Zero-shot ML")
            st.caption(
                "Target device/study is withheld while the model learns "
                "from the remaining evidence."
            )

    with e3:
        with st.container(border=True):
            st.markdown("#### ③ Decision validation")
            st.caption(
                "Regret, near-optimal success, model agreement and "
                "random-seed robustness evaluate the recommendation."
            )

    with e4:
        with st.container(border=True):
            st.markdown("#### ④ Hardware evidence")
            st.caption(
                "Selected strategic points are evaluated with NeuroSim "
                "for circuit/system-level trade-offs."
            )


    st.divider()


    # ========================================================
    # CLAIM MATRIX
    # ========================================================

    st.markdown("### Scientific claim boundary")

    yes_col, no_col = st.columns(2)

    with yes_col:
        st.success(
            "SUPPORTED BY THE CURRENT STUDY"
        )

        st.markdown(
            """
            - study-blocked zero-shot recommendation on the current dataset
            - near-optimal decision performance for RF and Extra Trees
            - robustness across the tested five seeds
            - cross-model comparison
            - 32-point strategic NeuroSim hardware evidence
            - multi-objective Pareto trade-offs
            """
        )

    with no_col:
        st.error(
            "NOT ESTABLISHED BY THE CURRENT STUDY"
        )

        st.markdown(
            """
            - fabricated-chip validation
            - measured chip energy, area or latency
            - universal generalization to all memristor technologies
            - direct NeuroSim validation of every ML recommendation
            - software accuracy as hardware-measured accuracy
            - replacement of NeuroSim, SPICE or EDA
            """
        )


    st.divider()


    # ========================================================
    # RESEARCH EVIDENCE STATUS
    # ========================================================

    st.markdown("### Research evidence status")

    s1, s2, s3 = st.columns(3)

    with s1:
        with st.container(border=True):
            st.markdown("#### Cross-device validation")
            st.success("10 held-out device profiles")
            st.caption(
                "The target device is excluded from training and evaluated "
                "using study-blocked zero-shot recommendation."
            )

    with s2:
        with st.container(border=True):
            st.markdown("#### Robustness evaluation")
            st.success("4 models • 5 random seeds")
            st.caption(
                "The final study compares multiple regressors, random-seed "
                "stability and cross-model recommendation agreement."
            )

    with s3:
        with st.container(border=True):
            st.markdown("#### Hardware evidence")
            st.success("32 NeuroSim points • 20 Pareto")
            st.caption(
                "Selected strategic configurations are evaluated for "
                "circuit/system-level energy, latency, area and efficiency trade-offs."
            )

    st.caption(
        "Current evidence is computational and literature-grounded. "
        "NeuroSim values are simulation estimates, not fabricated-chip measurements."
    )
