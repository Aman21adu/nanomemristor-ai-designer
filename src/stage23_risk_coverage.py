from pathlib import Path
import json
import numpy as np
import pandas as pd


SUMMARY_FILE = Path(
    "results/tables/zero_shot_study_summary.csv"
)

PRED_FILE = Path(
    "results/tables/zero_shot_study_predictions.csv"
)

OUT_POLICY = Path(
    "results/tables/stage23_risk_coverage_summary.csv"
)

OUT_DEVICE = Path(
    "results/tables/stage23_risk_coverage_by_device.csv"
)

OUT_SCOPE = Path(
    "results/tables/stage23_risk_coverage_scope.json"
)


def as_bool(value):
    if isinstance(value, bool):
        return value

    return (
        str(value).strip().lower()
        in {"true", "1", "yes"}
    )


summary = pd.read_csv(SUMMARY_FILE)
pred = pd.read_csv(PRED_FILE)

device_rows = []


for _, row in summary.iterrows():

    device = str(row["held_out_device"])

    p = pred[
        pred["held_out_device"].astype(str)
        == device
    ].copy()

    if p.empty:
        raise ValueError(
            f"No prediction rows found for {device}"
        )

    # --------------------------------------------------------
    # Accuracy-first fallback:
    # choose architecture with maximum predicted accuracy.
    # --------------------------------------------------------

    fallback_idx = (
        p["predicted_accuracy"]
        .astype(float)
        .idxmax()
    )

    fallback_row = p.loc[fallback_idx]

    true_best = float(
        p["accuracy"].astype(float).max()
    )

    fallback_actual = float(
        fallback_row["accuracy"]
    )

    fallback_regret = (
        true_best - fallback_actual
    )

    fallback_cost = float(
        fallback_row[
            "relative_hardware_cost_proxy"
        ]
    )

    # --------------------------------------------------------
    # Validation-aware cost recommendation already stored
    # in zero_shot_study_summary.csv.
    # --------------------------------------------------------

    cb = float(
        row["validation_aware_crossbar"]
    )

    wb = float(
        row["validation_aware_weight_bits"]
    )

    adc = float(
        row["validation_aware_adc_bits"]
    )

    chosen = p[
        (
            p["crossbar_size"].astype(float)
            == cb
        )
        &
        (
            p[
                "requested_weight_bits"
            ].astype(float)
            == wb
        )
        &
        (
            p["adc_bits"].astype(float)
            == adc
        )
    ]

    if len(chosen) != 1:
        raise ValueError(
            f"{device}: expected one "
            f"validation-aware configuration, "
            f"found {len(chosen)}"
        )

    chosen = chosen.iloc[0]

    cost_regret = float(
        row["validation_aware_regret_pp"]
    )

    cost_proxy = float(
        chosen[
            "relative_hardware_cost_proxy"
        ]
    )

    same_mode_count = int(
        float(
            row[
                "guarded_same_mode_training_devices"
            ]
        )
    )

    inside = as_bool(
        row[
            "guarded_inside_same_mode_ratio_envelope"
        ]
    )

    device_rows.append(
        {
            "held_out_device": device,
            "held_out_study":
                row["held_out_study"],
            "held_out_family":
                row["held_out_family"],

            "same_mode_training_devices":
                same_mode_count,

            "inside_same_mode_ratio_envelope":
                inside,

            "fallback_regret_pp":
                fallback_regret,

            "fallback_cost_proxy":
                fallback_cost,

            "validation_aware_regret_pp":
                cost_regret,

            "validation_aware_cost_proxy":
                cost_proxy,
        }
    )


devices = pd.DataFrame(device_rows)


# ------------------------------------------------------------
# Fixed sensitivity policies.
#
# IMPORTANT:
# These are descriptive sensitivity tests.
# We are NOT selecting the best threshold from test results.
# ------------------------------------------------------------

def policy_accuracy_first(df):
    return np.zeros(
        len(df),
        dtype=bool,
    )


def policy_current(df):
    return (
        (
            df["same_mode_training_devices"]
            >= 2
        )
        &
        (
            df[
                "inside_same_mode_ratio_envelope"
            ]
        )
    ).to_numpy()


def policy_one_inside(df):
    return (
        (
            df["same_mode_training_devices"]
            >= 1
        )
        &
        (
            df[
                "inside_same_mode_ratio_envelope"
            ]
        )
    ).to_numpy()


def policy_two_any_ratio(df):
    return (
        df["same_mode_training_devices"]
        >= 2
    ).to_numpy()


def policy_one_any_ratio(df):
    return (
        df["same_mode_training_devices"]
        >= 1
    ).to_numpy()


def policy_no_gate(df):
    return np.ones(
        len(df),
        dtype=bool,
    )


POLICIES = {
    "ACCURACY_FIRST_ALWAYS":
        policy_accuracy_first,

    "CURRENT_STRICT":
        policy_current,

    "COUNT1_PLUS_RATIO_ENVELOPE":
        policy_one_inside,

    "COUNT2_ANY_RATIO":
        policy_two_any_ratio,

    "COUNT1_ANY_RATIO":
        policy_one_any_ratio,

    "NO_GATE_COST_ALWAYS":
        policy_no_gate,
}


policy_rows = []
expanded_rows = []


baseline_mean_cost = float(
    devices[
        "fallback_cost_proxy"
    ].mean()
)


for name, func in POLICIES.items():

    accepted = func(devices)

    result = devices.copy()

    result["policy"] = name
    result["cost_optimization_accepted"] = (
        accepted
    )

    result["selected_regret_pp"] = np.where(
        accepted,
        result[
            "validation_aware_regret_pp"
        ],
        result[
            "fallback_regret_pp"
        ],
    )

    result["selected_cost_proxy"] = np.where(
        accepted,
        result[
            "validation_aware_cost_proxy"
        ],
        result[
            "fallback_cost_proxy"
        ],
    )

    result["success_le_0_5pp"] = (
        result["selected_regret_pp"]
        <= 0.5
    )

    coverage = float(
        accepted.mean()
    )

    mean_regret = float(
        result[
            "selected_regret_pp"
        ].mean()
    )

    max_regret = float(
        result[
            "selected_regret_pp"
        ].max()
    )

    success = float(
        result[
            "success_le_0_5pp"
        ].mean()
    )

    mean_cost = float(
        result[
            "selected_cost_proxy"
        ].mean()
    )

    cost_reduction = (
        100.0
        * (
            baseline_mean_cost
            - mean_cost
        )
        / baseline_mean_cost
    )

    accepted_rows = result[
        result[
            "cost_optimization_accepted"
        ]
    ]

    if len(accepted_rows) > 0:

        accepted_mean_regret = float(
            accepted_rows[
                "selected_regret_pp"
            ].mean()
        )

        accepted_max_regret = float(
            accepted_rows[
                "selected_regret_pp"
            ].max()
        )

        accepted_success = float(
            accepted_rows[
                "success_le_0_5pp"
            ].mean()
        )

    else:
        accepted_mean_regret = np.nan
        accepted_max_regret = np.nan
        accepted_success = np.nan

    # --------------------------------------------------------
    # Independent-study weighted results.
    # The three TiOx_02 profiles count as one study group here.
    # --------------------------------------------------------

    study_group = (
        result
        .groupby(
            "held_out_study",
            as_index=False,
        )
        .agg(
            mean_regret_pp=(
                "selected_regret_pp",
                "mean",
            ),
            all_devices_success=(
                "success_le_0_5pp",
                "all",
            ),
        )
    )

    study_weighted_regret = float(
        study_group[
            "mean_regret_pp"
        ].mean()
    )

    study_group_success = float(
        study_group[
            "all_devices_success"
        ].mean()
    )

    policy_rows.append(
        {
            "policy": name,

            "profiles": len(result),

            "coverage":
                coverage,

            "accepted_profiles":
                int(accepted.sum()),

            "mean_regret_pp":
                mean_regret,

            "max_regret_pp":
                max_regret,

            "success_rate_le_0_5pp":
                success,

            "accepted_mean_regret_pp":
                accepted_mean_regret,

            "accepted_max_regret_pp":
                accepted_max_regret,

            "accepted_success_rate":
                accepted_success,

            "mean_relative_hardware_cost_proxy":
                mean_cost,

            "cost_proxy_reduction_vs_accuracy_first_pct":
                float(cost_reduction),

            "study_weighted_mean_regret_pp":
                study_weighted_regret,

            "study_group_allpass_rate":
                study_group_success,
        }
    )

    expanded_rows.append(result)


policy_summary = pd.DataFrame(
    policy_rows
)

expanded = pd.concat(
    expanded_rows,
    ignore_index=True,
)


policy_summary.to_csv(
    OUT_POLICY,
    index=False,
)

expanded.to_csv(
    OUT_DEVICE,
    index=False,
)


scope = {
    "stage": "23B",
    "name":
        "Support-gate risk-coverage sensitivity analysis",

    "evaluation_basis":
        "existing leave-one-study-out predictions",

    "profiles": int(len(devices)),

    "independent_studies":
        int(
            devices[
                "held_out_study"
            ].nunique()
        ),

    "near_optimal_threshold_pp":
        0.5,

    "hardware_metric":
        "relative_hardware_cost_proxy",

    "hardware_metric_warning":
        (
            "The cost proxy is heuristic and must not "
            "be described as measured energy, area, "
            "latency, power, or monetary cost."
        ),

    "important_methodological_note":
        (
            "Alternative gate rules are evaluated only "
            "as sensitivity analyses. They must not be "
            "selected as a new final gate merely because "
            "they perform best on these same held-out "
            "test profiles."
        ),
}


OUT_SCOPE.write_text(
    json.dumps(
        scope,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 90)
print("STAGE 23B — SUPPORT-GATE RISK-COVERAGE")
print("=" * 90)

cols = [
    "policy",
    "coverage",
    "mean_regret_pp",
    "max_regret_pp",
    "success_rate_le_0_5pp",
    "cost_proxy_reduction_vs_accuracy_first_pct",
    "study_weighted_mean_regret_pp",
    "study_group_allpass_rate",
]

print(
    policy_summary[
        cols
    ].to_string(
        index=False
    )
)
