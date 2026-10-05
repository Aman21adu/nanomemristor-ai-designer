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

    _html("""
        <div class="final-hero">

            <div class="final-badge">
                Research prototype • Nano + AI + chip design
            </div>

            <div class="final-title">
                From a memristor device
                to an accelerator design recommendation.
            </div>

            <div class="final-subtitle">
                NanoMemristor AI Designer learns transferable relationships
                across experimentally characterized memristor technologies,
                screens accelerator configurations for a held-out or new
                device, checks whether the recommendation is supported,
                and helps prioritize what should receive detailed
                circuit-level validation next.
            </div>

            <div class="flow-strip">
                <span class="flow-node">Device evidence</span>
                <span class="flow-arrow">→</span>

                <span class="flow-node">Cross-device AI</span>
                <span class="flow-arrow">→</span>

                <span class="flow-node">Support check</span>
                <span class="flow-arrow">→</span>

                <span class="flow-node">Prioritize design</span>
                <span class="flow-arrow">→</span>

                <span class="flow-node">Validate</span>
            </div>

        </div>
        """)

    # --------------------------------------------------------
    # Core evidence
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Device profiles",
        device_count,
        help="Simulation-ready literature-derived memristor profiles.",
    )

    c2.metric(
        "Independent studies",
        study_count,
    )

    c3.metric(
        "Design cases",
        f"{total_simulation_rows:,}",
        help="245 accelerator configurations for each of 10 device profiles.",
    )

    c4.metric(
        "ML models evaluated",
        model_summary["model"].nunique(),
    )

    st.caption(
        f"{family_count} technology families • "
        f"{seed_summary['seed'].nunique()} random seeds • "
        f"{len(hardware)} strategic NeuroSim points • "
        f"{len(pareto_front)} Pareto-front hardware points"
    )

    st.write("")

    # --------------------------------------------------------
    # Problem → solution → output
    # --------------------------------------------------------

    p1, p2, p3 = st.columns(3)

    with p1:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">Problem</div>
                <div class="public-card-title">
                    A better device is not automatically a better accelerator.
                </div>
                <div class="public-card-text">
                    Memristor technologies differ in switching behavior,
                    dynamic range and available states. Each new device can
                    change the preferred crossbar, weight and ADC design.
                </div>
            </div>
            """)

    with p2:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">Approach</div>
                <div class="public-card-title">
                    Cross-device AI + support check
                </div>
                <div class="public-card-text">
                    Learn from characterized technologies, hold the target
                    device out of training, then check whether the recommendation
                    lies inside supported descriptor space before trusting
                    cost-oriented optimization.
                </div>
            </div>
            """)

    with p3:
        _html("""
            <div class="public-card">
                <div class="public-card-kicker">Output</div>
                <div class="public-card-title">
                    What deserves detailed validation next?
                </div>
                <div class="public-card-text">
                    The system proposes a crossbar size, weight precision
                    and ADC precision as an early-stage screening decision.
                    Detailed simulators and EDA remain the validation layer.
                </div>
            </div>
            """)

    st.write("")

    # --------------------------------------------------------
    # Main final results
    # --------------------------------------------------------

    st.markdown("### Final research snapshot")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        "Random Forest",
        f"{int(rf['guarded_success_devices'])}/10 near-optimal",
        f"{float(rf['guarded_mean_regret_pp']):.3f} pp regret",
    )

    r2.metric(
        "Extra Trees",
        f"{int(et['guarded_success_devices'])}/10 near-optimal",
        f"{float(et['guarded_mean_regret_pp']):.3f} pp regret",
    )

    r3.metric(
        "Seed robustness",
        "7/7 × 5 seeds",
        help=(
            "RF and Extra Trees each maintained near-optimal "
            "target-device success across all five tested seeds."
        ),
    )

    r4.metric(
        "Hardware evidence",
        f"{len(hardware)} NeuroSim points",
        f"{len(pareto_front)} Pareto points",
    )

    st.info(
        "Positioning: NanoMemristor AI Designer does not replace NeuroSim, "
        "SPICE or EDA. It screens and prioritizes promising configurations "
        "so detailed validation can focus on better candidates."
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
