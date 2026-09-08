from pathlib import Path

import pandas as pd


# ============================================================
# PURPOSE
# ============================================================
#
# Consolidate controlled device-to-accelerator experiments
# into one defensible evidence summary.
#
#
# The summary distinguishes:
#
#   1. What device properties affect the accelerator
#   2. What properties currently do NOT affect it
#   3. Device-circuit interactions
#   4. Device-architecture tradeoffs
#   5. Current simulator limitations
#
#
# Intended especially for the mentor question:
#
#   "What makes this nanotechnology rather than only AI?"
#
#
# CURRENT SCIENTIFIC POSITION
# ============================================================
#
# The simulator is:
#
#   LITERATURE-GROUNDED
#   DEVICE-ELECTRICAL-BEHAVIOR-AWARE
#
#
# It is NOT yet:
#
#   FULL MATERIAL-TO-DEVICE PHYSICS SIMULATION
#
#
# Thickness, material transport, filament physics,
# geometry-dependent IR drop, variability, noise, drift,
# switching kinetics, power and heating are not yet fully
# propagated into accelerator behavior.
# ============================================================


# ============================================================
# INPUT FILES
# ============================================================

CONDUCTANCE_SCALE_FILE = (
    "results/tables/"
    "conductance_scale_invariance.csv"
)


RATIO_ADC4_FILE = (
    "results/tables/"
    "ratio_accuracy_sensitivity.csv"
)


RATIO_ADC6_FILE = (
    "results/tables/"
    "ratio_accuracy_sensitivity_adc6.csv"
)


STATE_COUNT_FILE = (
    "results/tables/"
    "state_count_accuracy_sensitivity.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_CSV = (
    "results/tables/"
    "nano_device_accelerator_evidence.csv"
)


OUTPUT_TEXT = (
    "results/tables/"
    "nano_device_accelerator_evidence_summary.txt"
)


# ============================================================
# HELPER:
# require input file
# ============================================================

def require_file(
    file_path,
):

    path = Path(
        file_path
    )


    if not path.exists():

        raise FileNotFoundError(

            f"Required experiment file not found: "
            f"{file_path}"

        )


    return path


# ============================================================
# HELPER:
# summarize absolute conductance-scale experiment
# ============================================================

def summarize_conductance_scale_experiment(
    dataframe,
):

    total_cases = len(
        dataframe
    )


    passed_cases = int(

        dataframe[
            "scale_invariance_pass"
        ]

        .astype(bool)

        .sum()

    )


    all_pass = (
        passed_cases
        ==
        total_cases
    )


    minimum_scale = float(

        dataframe[
            "conductance_scale_factor"
        ]
        .min()

    )


    maximum_scale = float(

        dataframe[
            "conductance_scale_factor"
        ]
        .max()

    )


    maximum_difference = float(

        dataframe[
            "max_abs_output_difference"
        ]
        .max()

    )


    maximum_mean_difference = float(

        dataframe[
            "mean_abs_output_difference"
        ]
        .max()

    )


    all_argmax_same = bool(

        dataframe[
            "argmax_same"
        ]
        .astype(bool)
        .all()

    )


    all_levels_same = bool(

        dataframe[
            "levels_same"
        ]
        .astype(bool)
        .all()

    )


    device_count = int(

        dataframe[
            "device_id"
        ]
        .nunique()

    )


    return [{

        "evidence_id":
            "ABSOLUTE_CONDUCTANCE_SCALE_INVARIANCE",

        "device_template":
            "MULTIPLE_DEVICES",

        "device_property":
            "Absolute conductance magnitude",

        "property_category":
            "CURRENT_SIMULATOR_LIMITATION",

        "conductance_mode":
            "MULTIPLE",

        "accelerator_mechanism":
            (
                "Gmin and Gmax are scaled together while "
                "their ratio is preserved. In the present "
                "equations, physical current and ADC "
                "full-scale scale together and the result "
                "is divided by conductance range."
            ),

        "application_metric":
            "Crossbar numerical output",

        "fixed_crossbar_size":
            int(
                dataframe[
                    "crossbar_size"
                ]
                .iloc[0]
            ),

        "fixed_weight_bits":
            int(
                dataframe[
                    "requested_weight_bits"
                ]
                .iloc[0]
            ),

        "fixed_adc_bits":
            int(
                dataframe[
                    "adc_bits"
                ]
                .iloc[0]
            ),

        "low_test_value":
            minimum_scale,

        "high_test_value":
            maximum_scale,

        "low_value_accuracy":
            float("nan"),

        "high_value_accuracy":
            float("nan"),

        "accuracy_change_pp":
            float("nan"),

        "best_test_value":
            float("nan"),

        "best_test_accuracy":
            float("nan"),

        "original_device_value":
            float("nan"),

        "controlled_or_measured":
            "CONTROLLED_ABLATION",

        "tested_cases":
            total_cases,

        "passed_cases":
            passed_cases,

        "tested_devices":
            device_count,

        "min_scale_factor":
            minimum_scale,

        "max_scale_factor":
            maximum_scale,

        "max_abs_output_difference":
            maximum_difference,

        "max_mean_abs_output_difference":
            maximum_mean_difference,

        "all_argmax_same":
            all_argmax_same,

        "all_effective_levels_same":
            all_levels_same,

        "scientific_interpretation":
            (
                "Absolute conductance magnitude is effectively "
                "normalized away by the current simulator when "
                "the ON/OFF ratio is preserved."
            ),

        "what_it_supports":
            (
                f"{passed_cases}/{total_cases} controlled cases "
                f"passed from {minimum_scale:g}x to "
                f"{maximum_scale:g}x conductance scaling. "
                f"The largest numerical output difference was "
                f"{maximum_difference:.3e}."
            ),

        "what_it_does_not_support":
            (
                "Does not imply that real memristor hardware "
                "is independent of absolute conductance. "
                "IR drop, wire resistance, power, current "
                "limits, sensing, heating and related effects "
                "are not yet modeled."
            ),

    }]


# ============================================================
# HELPER:
# summarize one ON/OFF-ratio experiment
# ============================================================

def summarize_ratio_experiment(
    dataframe,
    adc_bits,
):

    rows = []


    device_ids = (

        dataframe[
            "device_template"
        ]

        .dropna()

        .unique()

        .tolist()

    )


    for device_id in device_ids:

        subset = dataframe[

            dataframe[
                "device_template"
            ]
            ==
            device_id

        ].copy()


        subset = subset.sort_values(
            "tested_on_off_ratio"
        )


        lowest = subset.iloc[0]

        highest = subset.iloc[-1]


        best_index = (

            subset[
                "hardware_accuracy"
            ]

            .idxmax()

        )


        best = dataframe.loc[
            best_index
        ]


        low_ratio = float(

            lowest[
                "tested_on_off_ratio"
            ]

        )


        high_ratio = float(

            highest[
                "tested_on_off_ratio"
            ]

        )


        low_accuracy = float(

            lowest[
                "hardware_accuracy"
            ]

        )


        high_accuracy = float(

            highest[
                "hardware_accuracy"
            ]

        )


        accuracy_change = (

            high_accuracy
            - low_accuracy

        )


        best_ratio = float(

            best[
                "tested_on_off_ratio"
            ]

        )


        best_accuracy = float(

            best[
                "hardware_accuracy"
            ]

        )


        conductance_mode = str(

            lowest[
                "conductance_mode"
            ]

        )


        original_ratio = float(

            lowest[
                "original_device_ratio"
            ]

        )


        if adc_bits == 4:

            interpretation = (

                "ON/OFF ratio can strongly affect "
                "application-level accuracy when ADC "
                "precision is coarse."

            )


        else:

            interpretation = (

                "ON/OFF ratio remains an active device "
                "parameter, but its application-level "
                "effect becomes much smaller with the "
                "higher ADC precision."

            )


        rows.append({

            "evidence_id":
                (
                    f"ON_OFF_RATIO_ADC{adc_bits}_"
                    f"{device_id}"
                ),

            "device_template":
                device_id,

            "device_property":
                "ON/OFF conductance ratio",

            "property_category":
                "DEVICE_ELECTRICAL_CHARACTERISTIC",

            "conductance_mode":
                conductance_mode,

            "accelerator_mechanism":
                (
                    "Changes Gmin relative to Gmax, "
                    "altering branch currents before "
                    "ADC quantization."
                ),

            "application_metric":
                "MNIST classification accuracy",

            "fixed_crossbar_size":
                int(
                    lowest[
                        "crossbar_size"
                    ]
                ),

            "fixed_weight_bits":
                int(
                    lowest[
                        "requested_weight_bits"
                    ]
                ),

            "fixed_adc_bits":
                adc_bits,

            "low_test_value":
                low_ratio,

            "high_test_value":
                high_ratio,

            "low_value_accuracy":
                low_accuracy,

            "high_value_accuracy":
                high_accuracy,

            "accuracy_change_pp":
                accuracy_change,

            "best_test_value":
                best_ratio,

            "best_test_accuracy":
                best_accuracy,

            "original_device_value":
                original_ratio,

            "controlled_or_measured":
                "CONTROLLED_ABLATION",

            "tested_cases":
                float("nan"),

            "passed_cases":
                float("nan"),

            "tested_devices":
                1,

            "min_scale_factor":
                float("nan"),

            "max_scale_factor":
                float("nan"),

            "max_abs_output_difference":
                float("nan"),

            "max_mean_abs_output_difference":
                float("nan"),

            "all_argmax_same":
                float("nan"),

            "all_effective_levels_same":
                float("nan"),

            "scientific_interpretation":
                interpretation,

            "what_it_supports":
                (
                    "A memristor device electrical parameter "
                    "can propagate through the crossbar model "
                    "to application-level neural-network "
                    "accuracy."
                ),

            "what_it_does_not_support":
                (
                    "Does not prove a direct causal "
                    "relationship from material thickness "
                    "or atomic-scale physics to accuracy. "
                    "The tested ratios are hypothetical "
                    "controlled ablation values."
                ),

        })


    return rows


# ============================================================
# HELPER:
# summarize physical state-count experiment
# ============================================================

def summarize_state_count_experiment(
    dataframe,
):

    subset = dataframe.sort_values(
        "physical_conductance_states"
    )


    lowest = subset.iloc[0]

    highest = subset.iloc[-1]


    low_states = int(

        lowest[
            "physical_conductance_states"
        ]

    )


    high_states = int(

        highest[
            "physical_conductance_states"
        ]

    )


    low_levels = int(

        lowest[
            "effective_weight_levels"
        ]

    )


    high_levels = int(

        highest[
            "effective_weight_levels"
        ]

    )


    low_accuracy = float(

        lowest[
            "hardware_accuracy"
        ]

    )


    high_accuracy = float(

        highest[
            "hardware_accuracy"
        ]

    )


    accuracy_change = (

        high_accuracy
        - low_accuracy

    )


    requested_levels = int(

        lowest[
            "requested_signed_weight_levels"
        ]

    )


    saturation_rows = subset[

        subset[
            "effective_weight_levels"
        ]
        ==
        requested_levels

    ]


    if len(
        saturation_rows
    ) > 0:

        minimum_saturating_states = int(

            saturation_rows[
                "physical_conductance_states"
            ]

            .min()

        )


        saturation_accuracy = float(

            saturation_rows[

                saturation_rows[
                    "physical_conductance_states"
                ]
                ==
                minimum_saturating_states

            ][
                "hardware_accuracy"
            ]

            .iloc[0]

        )


    else:

        minimum_saturating_states = None

        saturation_accuracy = float(
            "nan"
        )


    return [{

        "evidence_id":
            "PHYSICAL_STATE_COUNT",

        "device_template":
            str(
                lowest[
                    "template_device"
                ]
            ),

        "device_property":
            "Physical conductance-state capability",

        "property_category":
            "DEVICE_SWITCHING_CHARACTERISTIC",

        "conductance_mode":
            str(
                lowest[
                    "conductance_mode"
                ]
            ),

        "accelerator_mechanism":
            (
                "Physical state count limits the number "
                "of representable signed neural-weight "
                "levels in a single differential-pair "
                "mapping."
            ),

        "application_metric":
            "MNIST classification accuracy",

        "fixed_crossbar_size":
            int(
                lowest[
                    "crossbar_size"
                ]
            ),

        "fixed_weight_bits":
            int(
                lowest[
                    "requested_weight_bits"
                ]
            ),

        "fixed_adc_bits":
            int(
                lowest[
                    "adc_bits"
                ]
            ),

        "low_test_value":
            low_states,

        "high_test_value":
            high_states,

        "low_value_accuracy":
            low_accuracy,

        "high_value_accuracy":
            high_accuracy,

        "accuracy_change_pp":
            accuracy_change,

        "best_test_value":
            minimum_saturating_states,

        "best_test_accuracy":
            saturation_accuracy,

        "original_device_value":
            7.0,

        "controlled_or_measured":
            "CONTROLLED_ABLATION",

        "tested_cases":
            len(
                subset
            ),

        "passed_cases":
            float("nan"),

        "tested_devices":
            1,

        "min_scale_factor":
            float("nan"),

        "max_scale_factor":
            float("nan"),

        "max_abs_output_difference":
            float("nan"),

        "max_mean_abs_output_difference":
            float("nan"),

        "all_argmax_same":
            float("nan"),

        "all_effective_levels_same":
            float("nan"),

        "scientific_interpretation":
            (
                f"In this W=4 experiment, increasing "
                f"physical state capability changes "
                f"effective neural-weight precision "
                f"from {low_levels} to {high_levels} "
                f"signed levels. Accuracy changes from "
                f"{low_accuracy:.2f}% to "
                f"{high_accuracy:.2f}%. Precision "
                f"saturates once "
                f"{minimum_saturating_states} physical "
                f"states provide the requested "
                f"{requested_levels} signed levels."
            ),

        "what_it_supports":
            (
                "Physical memristor state capability is "
                "a direct device-level constraint on "
                "accelerator precision. Insufficient "
                "state capability can strongly degrade "
                "application accuracy."
            ),

        "what_it_does_not_support":
            (
                "The tested state counts are "
                "hypothetical controlled values. "
                "They must not be presented as measured "
                "properties of the TaOx template device."
            ),

    }]


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "BUILDING NANO-DEVICE -> ACCELERATOR EVIDENCE SUMMARY"
    )

    print(
        "====================================================="
    )


    # ========================================================
    # Validate files
    # ========================================================

    conductance_scale_path = require_file(
        CONDUCTANCE_SCALE_FILE
    )


    ratio_adc4_path = require_file(
        RATIO_ADC4_FILE
    )


    ratio_adc6_path = require_file(
        RATIO_ADC6_FILE
    )


    state_count_path = require_file(
        STATE_COUNT_FILE
    )


    # ========================================================
    # Read experiments
    # ========================================================

    conductance_scale = pd.read_csv(
        conductance_scale_path
    )


    ratio_adc4 = pd.read_csv(
        ratio_adc4_path
    )


    ratio_adc6 = pd.read_csv(
        ratio_adc6_path
    )


    state_count = pd.read_csv(
        state_count_path
    )


    # ========================================================
    # Build evidence rows
    # ========================================================

    evidence_rows = []


    evidence_rows.extend(

        summarize_conductance_scale_experiment(
            conductance_scale
        )

    )


    evidence_rows.extend(

        summarize_ratio_experiment(

            ratio_adc4,
            adc_bits=4,

        )

    )


    evidence_rows.extend(

        summarize_ratio_experiment(

            ratio_adc6,
            adc_bits=6,

        )

    )


    evidence_rows.extend(

        summarize_state_count_experiment(
            state_count
        )

    )


    evidence_df = pd.DataFrame(
        evidence_rows
    )


    # ========================================================
    # Save evidence CSV
    # ========================================================

    Path(
        "results/tables"
    ).mkdir(

        parents=True,
        exist_ok=True,

    )


    evidence_df.to_csv(

        OUTPUT_CSV,
        index=False,

    )


    # ========================================================
    # Conductance-scale summary values
    # ========================================================

    scale_row = evidence_df[

        evidence_df[
            "evidence_id"
        ]
        ==
        "ABSOLUTE_CONDUCTANCE_SCALE_INVARIANCE"

    ].iloc[0]


    # ========================================================
    # State summary
    # ========================================================

    state_row = evidence_df[

        evidence_df[
            "evidence_id"
        ]
        ==
        "PHYSICAL_STATE_COUNT"

    ].iloc[0]


    # ========================================================
    # Human-readable mentor summary
    # ========================================================

    lines = []


    lines.append(
        "NANO-DEVICE TO ACCELERATOR EVIDENCE SUMMARY"
    )

    lines.append(
        "=========================================="
    )

    lines.append("")


    lines.append(
        "CURRENT SCIENTIFIC POSITION"
    )

    lines.append(
        "---------------------------"
    )

    lines.append("")


    lines.append(

        "The current simulator is "
        "device-electrical-behavior-aware."

    )


    lines.append(

        "It is not yet a complete "
        "material-to-device physics simulator."

    )


    lines.append("")


    lines.append(

        "Literature-derived memristor electrical "
        "characteristics constrain how neural-network "
        "weights are represented and how crossbar "
        "currents are quantized."

    )


    lines.append("")


    # ========================================================
    # 1. Absolute conductance scale
    # ========================================================

    lines.append(
        "1. ABSOLUTE CONDUCTANCE SCALE"
    )

    lines.append(
        "-----------------------------"
    )

    lines.append("")


    lines.append(

        "Controlled experiment: Gmin and Gmax were "
        "multiplied by the same factor while the "
        "ON/OFF ratio was preserved."

    )


    lines.append("")


    lines.append(

        f"Tested "
        f"{int(scale_row['tested_cases'])} cases across "
        f"{int(scale_row['tested_devices'])} devices."

    )


    lines.append(

        f"Scale range: "
        f"{scale_row['min_scale_factor']:g}x to "
        f"{scale_row['max_scale_factor']:g}x."

    )


    lines.append(

        f"Passed cases: "
        f"{int(scale_row['passed_cases'])}/"
        f"{int(scale_row['tested_cases'])}."

    )


    lines.append(

        f"Maximum numerical output difference: "
        f"{scale_row['max_abs_output_difference']:.3e}."

    )


    lines.append("")


    lines.append(

        "Interpretation: absolute conductance magnitude "
        "is effectively normalized away by the current "
        "simulator when the ON/OFF ratio is unchanged."

    )


    lines.append("")


    lines.append(

        "This is a CURRENT MODEL LIMITATION, not a "
        "claim about real hardware."

    )


    lines.append("")


    lines.append(

        "Real arrays can depend on absolute conductance "
        "through IR drop, line resistance, current "
        "magnitude, power, sensing limits and heating."

    )


    lines.append("")


    # ========================================================
    # 2. ON/OFF ratio
    # ========================================================

    lines.append(
        "2. ON/OFF RATIO EFFECT"
    )

    lines.append(
        "----------------------"
    )

    lines.append("")


    lines.append(

        "Controlled experiment: only the ON/OFF "
        "conductance ratio was changed while network, "
        "crossbar size, weight precision, ADC precision, "
        "mapping mode and dataset were held fixed."

    )


    lines.append("")


    for device_id in [

        "ZnO_01",
        "TaOx_01",
        "HfOx_02",

    ]:

        adc4_row = evidence_df[

            evidence_df[
                "evidence_id"
            ]
            ==
            f"ON_OFF_RATIO_ADC4_{device_id}"

        ].iloc[0]


        adc6_row = evidence_df[

            evidence_df[
                "evidence_id"
            ]
            ==
            f"ON_OFF_RATIO_ADC6_{device_id}"

        ].iloc[0]


        lines.append(
            f"{device_id}:"
        )


        lines.append(

            f"  ADC 4: ratio "
            f"{adc4_row['low_test_value']:.0f} -> "
            f"{adc4_row['high_test_value']:.0f}, "
            f"accuracy "
            f"{adc4_row['low_value_accuracy']:.2f}% -> "
            f"{adc4_row['high_value_accuracy']:.2f}% "
            f"({adc4_row['accuracy_change_pp']:+.2f} pp)."

        )


        lines.append(

            f"  ADC 6: ratio "
            f"{adc6_row['low_test_value']:.0f} -> "
            f"{adc6_row['high_test_value']:.0f}, "
            f"accuracy "
            f"{adc6_row['low_value_accuracy']:.2f}% -> "
            f"{adc6_row['high_value_accuracy']:.2f}% "
            f"({adc6_row['accuracy_change_pp']:+.2f} pp)."

        )


        lines.append("")


    lines.append(

        "Interpretation: ON/OFF ratio is an active "
        "device parameter, but its accelerator-level "
        "importance depends strongly on ADC precision."

    )


    lines.append("")


    lines.append(

        "Higher ADC resolution compensates for much "
        "of the conductance-window limitation in the "
        "present model."

    )


    lines.append("")


    # ========================================================
    # 3. Physical state capability
    # ========================================================

    lines.append(
        "3. PHYSICAL CONDUCTANCE-STATE CAPABILITY"
    )

    lines.append(
        "----------------------------------------"
    )

    lines.append("")


    lines.append(

        "Controlled experiment: physical conductance-state "
        "count was varied while ON/OFF ratio, ADC precision, "
        "weight precision, crossbar size, network and dataset "
        "were fixed."

    )


    lines.append("")


    lines.append(

        f"2 physical states -> "
        f"3 effective signed weight levels -> "
        f"{state_row['low_value_accuracy']:.2f}% accuracy."

    )


    lines.append(

        f"16 physical states -> "
        f"15 effective signed weight levels -> "
        f"{state_row['high_value_accuracy']:.2f}% accuracy."

    )


    lines.append("")


    lines.append(

        f"Accuracy difference: "
        f"{state_row['accuracy_change_pp']:.2f} "
        f"percentage points."

    )


    lines.append("")


    lines.append(

        f"For W=4, "
        f"{int(state_row['best_test_value'])} "
        f"physical states are sufficient to reach "
        f"the requested 15 signed levels."

    )


    lines.append("")


    lines.append(

        "Additional physical states beyond that point "
        "do not increase precision in this W=4 mapping."

    )


    lines.append("")


    # ========================================================
    # 4. Binary device tradeoff
    # ========================================================

    lines.append(
        "4. BINARY DEVICE ARCHITECTURE TRADEOFF"
    )

    lines.append(
        "--------------------------------------"
    )

    lines.append("")


    lines.append(

        "A binary memristor still has only two "
        "physical states."

    )


    lines.append(

        "The project does not treat one binary cell "
        "as having 15, 63 or 255 physical states."

    )


    lines.append("")


    lines.append(

        "Higher neural-weight precision is instead "
        "created through bit slicing across multiple "
        "binary cells."

    )


    lines.append("")


    lines.append(

        "For W=4: 3 binary slices per differential "
        "branch = 6 physical cells per neural-network "
        "weight, producing 15 effective signed "
        "accelerator levels."

    )


    lines.append("")


    lines.append(

        "Therefore device technology creates an "
        "architecture-level tradeoff: multilevel "
        "devices can encode more precision per cell, "
        "while binary devices recover precision using "
        "additional physical cells."

    )


    lines.append("")


    # ========================================================
    # Mentor answer
    # ========================================================

    lines.append(
        "MENTOR ANSWER: WHAT MAKES THIS NANO?"
    )

    lines.append(
        "-----------------------------------"
    )

    lines.append("")


    lines.append(

        "The nanotechnology contribution is not simply "
        "the material name supplied to an AI model."

    )


    lines.append("")


    lines.append(

        "Literature-derived memristor device "
        "characteristics constrain the physical "
        "representation of neural weights and therefore "
        "change accelerator behavior."

    )


    lines.append("")


    lines.append(

        "The strongest current evidence is physical "
        "conductance-state capability: insufficient "
        "states severely reduce effective weight "
        "precision and application accuracy."

    )


    lines.append("")


    lines.append(

        "ON/OFF ratio is also an active device "
        "parameter, but its importance interacts "
        "strongly with ADC resolution."

    )


    lines.append("")


    lines.append(

        "That interaction makes this a device-circuit-"
        "architecture co-design problem rather than "
        "simply an AI model ranking material names."

    )


    lines.append("")


    # ========================================================
    # Current limitation
    # ========================================================

    lines.append(
        "CURRENT LIMITATION"
    )

    lines.append(
        "------------------"
    )

    lines.append("")


    lines.append(

        "The present simulator starts from device "
        "electrical characteristics rather than "
        "directly predicting those characteristics "
        "from material physics."

    )


    lines.append("")


    lines.append(

        "The absolute-conductance experiment also "
        "shows that absolute conductance magnitude "
        "is currently normalized away."

    )


    lines.append("")


    lines.append(

        "Thickness, crystallite size, material stack, "
        "device area, switching mechanism, line "
        "resistance, variability, read noise, drift, "
        "filament dynamics, switching kinetics, power "
        "and heating are not yet fully propagated into "
        "accelerator behavior."

    )


    lines.append("")


    lines.append(
        "Therefore the current claim should be:"
    )


    lines.append("")


    lines.append(

        "\"Literature-grounded memristor "
        "device-to-accelerator co-design simulator.\""

    )


    lines.append("")


    lines.append(
        "Do not yet claim:"
    )


    lines.append("")


    lines.append(

        "\"Full material-physics-to-accelerator simulator.\""

    )


    # ========================================================
    # Save text
    # ========================================================

    with open(

        OUTPUT_TEXT,
        "w",
        encoding="utf-8",

    ) as f:

        f.write(

            "\n".join(
                lines
            )

        )


    # ========================================================
    # Console summary
    # ========================================================

    print()

    print(
        "Evidence rows:",
        len(
            evidence_df
        )
    )


    print()

    print(
        "Saved CSV:"
    )

    print(
        OUTPUT_CSV
    )


    print()

    print(
        "Saved mentor summary:"
    )

    print(
        OUTPUT_TEXT
    )


    print()

    print(
        "KEY CURRENT FINDINGS"
    )

    print(
        "--------------------"
    )


    print()

    print(

        "1. Absolute conductance scale is effectively "
        "normalized away in the current simulator."

    )


    print()

    print(

        "2. ON/OFF ratio affects accelerator behavior, "
        "but its importance strongly interacts with "
        "ADC resolution."

    )


    print()

    print(

        "3. Physical memristor state capability directly "
        "constrains accelerator weight precision."

    )


    print()

    print(

        "4. Binary devices recover higher precision "
        "through additional bit-sliced physical cells."

    )


    print()

    print(

        "This supports literature-grounded "
        "device-to-accelerator co-design, not yet "
        "full material-physics simulation."

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()