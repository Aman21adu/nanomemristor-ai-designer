from pathlib import Path
import copy

import pandas as pd
import torch
import torch.nn.functional as F

from device_model import load_device
from memristor_sim import crossbar_linear


# ============================================================
# Purpose
# ============================================================
#
# Controlled sensitivity test:
#
#     change ONLY the device ON/OFF ratio
#
# while keeping:
#
#     neural-network weights
#     inputs
#     crossbar size
#     weight precision
#     ADC precision
#     conductance mode
#     physical state capability
#
# unchanged.
#
#
# This complements test_conductance_scale.py.
#
# Previous test:
#
#     same ratio
#     different absolute conductance scale
#
#     -> effectively invariant
#
#
# Current test:
#
#     same absolute Gmax
#     different Gmin
#
#     -> changes Gmax/Gmin ratio
#
#
# We want to determine whether ON/OFF ratio materially affects
# the CURRENT simulator output.
# ============================================================


# ============================================================
# Output
# ============================================================

OUTPUT_FILE = (
    "results/tables/"
    "ratio_sensitivity.csv"
)


# ============================================================
# Representative device modes
# ============================================================
#
# ZnO_01:
#     binary + bit slicing
#
# TaOx_01:
#     fixed multilevel
#
# HfOx_02:
#     analog mapping
#
# We intentionally test several mapping regimes.
# ============================================================

DEVICE_IDS = [

    "ZnO_01",
    "TaOx_01",
    "HfOx_02",

]


# ============================================================
# Ratios to test
# ============================================================
#
# These are controlled hypothetical values.
#
# They are NOT being written into device_profiles.csv.
#
# They are used only for sensitivity analysis.
# ============================================================

TEST_RATIOS = [

    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    500.0,
    1000.0,
    3000.0,

]


# ============================================================
# High-ratio reference
# ============================================================
#
# This approximates:
#
#     Gmin -> 0
#
# while keeping:
#
#     Gmax = 1
#
# It is only a mathematical reference for this diagnostic.
# ============================================================

REFERENCE_RATIO = 1_000_000.0


# ============================================================
# Accelerator configuration
# ============================================================
#
# ADC = 4 is deliberately chosen because limited ADC
# resolution makes conductance-window effects easier to see.
#
# We are testing sensitivity, not reproducing the final
# optimization experiment here.
# ============================================================

CROSSBAR_SIZE = 32
WEIGHT_BITS = 4
ADC_BITS = 4


# ============================================================
# Deterministic synthetic layer
# ============================================================

torch.manual_seed(
    42
)


BATCH_SIZE = 64
INPUT_FEATURES = 64
OUTPUT_FEATURES = 16


x = torch.rand(

    BATCH_SIZE,
    INPUT_FEATURES,

    dtype=torch.float32,

)


weight = (

    torch.randn(

        OUTPUT_FEATURES,
        INPUT_FEATURES,

        dtype=torch.float32,

    )

    * 0.10

)


bias = (

    torch.randn(

        OUTPUT_FEATURES,

        dtype=torch.float32,

    )

    * 0.01

)


# ============================================================
# Software reference
# ============================================================
#
# This contains no memristor conductance mapping or ADC.
#
# It is useful only as an additional error reference.
# ============================================================

software_output = F.linear(

    x,
    weight,
    bias,

)


software_argmax = torch.argmax(

    software_output,

    dim=1,

)


# ============================================================
# Helper:
# create controlled profile with chosen ON/OFF ratio
# ============================================================

def make_ratio_profile(
    original_profile,
    ratio,
):

    if ratio <= 1.0:

        raise ValueError(

            "ON/OFF ratio must be greater than 1."

        )


    profile = copy.deepcopy(
        original_profile
    )


    # --------------------------------------------------------
    # Keep Gmax fixed at 1.
    #
    # Change only Gmin:
    #
    #     Gmax / Gmin = ratio
    #
    # --------------------------------------------------------

    gmax = 1.0

    gmin = (

        gmax
        / ratio

    )


    profile[
        "gmax"
    ] = gmax


    profile[
        "gmin"
    ] = gmin


    profile[
        "ratio"
    ] = ratio


    # --------------------------------------------------------
    # Descriptive resistance equivalents
    #
    # These are not used directly by crossbar_linear().
    # --------------------------------------------------------

    profile[
        "ron"
    ] = (

        1.0
        / gmax

    )


    profile[
        "roff"
    ] = (

        1.0
        / gmin

    )


    return profile


# ============================================================
# Results
# ============================================================

result_rows = []


# ============================================================
# Header
# ============================================================

print()

print(
    "ON/OFF RATIO SENSITIVITY TEST"
)

print(
    "========================================"
)


print()

print(
    "Controlled variables:"
)


print(
    f"Crossbar size = {CROSSBAR_SIZE}"
)


print(
    f"Weight bits = {WEIGHT_BITS}"
)


print(
    f"ADC bits = {ADC_BITS}"
)


print()

print(

    "Gmax is fixed at 1.0."

)


print(

    "Only Gmin is changed to create "
    "different Gmax/Gmin ratios."

)


print()

print(

    "No device database values are modified."

)


# ============================================================
# Test each mapping regime
# ============================================================

for device_id in DEVICE_IDS:

    original_profile = load_device(
        device_id
    )


    conductance_mode = str(

        original_profile[
            "conductance_mode"
        ]

    )


    physical_states = (

        original_profile.get(
            "states",
            float("nan")
        )

    )


    print()

    print(
        "========================================"
    )

    print(
        "DEVICE TEMPLATE:",
        device_id
    )

    print(
        "========================================"
    )


    print(
        "Conductance mode:",
        conductance_mode
    )


    print(
        "Physical states:",
        physical_states
    )


    # ========================================================
    # High-ratio mathematical reference
    # ========================================================

    reference_profile = make_ratio_profile(

        original_profile,

        REFERENCE_RATIO,

    )


    (

        reference_output,
        reference_levels,

    ) = crossbar_linear(

        x=x,

        weight=weight,

        bias=bias,

        crossbar_size=CROSSBAR_SIZE,

        weight_bits=WEIGHT_BITS,

        adc_bits=ADC_BITS,

        device_profile=reference_profile,

    )


    reference_argmax = torch.argmax(

        reference_output,

        dim=1,

    )


    print()

    print(

        f"{'Ratio':>10} "
        f"{'Gmin':>12} "
        f"{'Mean diff':>14} "
        f"{'Max diff':>14} "
        f"{'Ref argmax':>12} "
        f"{'SW argmax':>11} "
        f"{'SW MAE':>12}"

    )


    # ========================================================
    # Ratio sweep
    # ========================================================

    for ratio in TEST_RATIOS:

        profile = make_ratio_profile(

            original_profile,

            ratio,

        )


        (

            hardware_output,
            actual_levels,

        ) = crossbar_linear(

            x=x,

            weight=weight,

            bias=bias,

            crossbar_size=CROSSBAR_SIZE,

            weight_bits=WEIGHT_BITS,

            adc_bits=ADC_BITS,

            device_profile=profile,

        )


        # ----------------------------------------------------
        # Difference from high-ratio hardware reference
        # ----------------------------------------------------

        reference_difference = (

            hardware_output
            - reference_output

        ).abs()


        mean_difference = float(

            reference_difference
            .mean()
            .item()

        )


        max_difference = float(

            reference_difference
            .max()
            .item()

        )


        # ----------------------------------------------------
        # Argmax agreement with high-ratio hardware reference
        # ----------------------------------------------------

        hardware_argmax = torch.argmax(

            hardware_output,

            dim=1,

        )


        reference_argmax_agreement = float(

            (
                hardware_argmax
                ==
                reference_argmax
            )

            .float()

            .mean()

            .item()

        )


        # ----------------------------------------------------
        # Argmax agreement with ideal software layer
        # ----------------------------------------------------

        software_argmax_agreement = float(

            (
                hardware_argmax
                ==
                software_argmax
            )

            .float()

            .mean()

            .item()

        )


        # ----------------------------------------------------
        # Error versus ideal software output
        # ----------------------------------------------------

        software_mae = float(

            (

                hardware_output
                - software_output

            )

            .abs()

            .mean()

            .item()

        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        result_rows.append({

            "device_template":
                device_id,

            "conductance_mode":
                conductance_mode,

            "physical_states":
                physical_states,

            "crossbar_size":
                CROSSBAR_SIZE,

            "weight_bits":
                WEIGHT_BITS,

            "adc_bits":
                ADC_BITS,

            "on_off_ratio":
                ratio,

            "gmax":
                float(
                    profile[
                        "gmax"
                    ]
                ),

            "gmin":
                float(
                    profile[
                        "gmin"
                    ]
                ),

            "effective_weight_levels":
                int(
                    actual_levels
                ),

            "mean_abs_difference_vs_high_ratio":
                mean_difference,

            "max_abs_difference_vs_high_ratio":
                max_difference,

            "argmax_agreement_vs_high_ratio":
                reference_argmax_agreement,

            "argmax_agreement_vs_software":
                software_argmax_agreement,

            "mean_abs_error_vs_software":
                software_mae,

        })


        print(

            f"{ratio:10.1f} "

            f"{profile['gmin']:12.6f} "

            f"{mean_difference:14.6e} "

            f"{max_difference:14.6e} "

            f"{100.0 * reference_argmax_agreement:11.1f}% "

            f"{100.0 * software_argmax_agreement:10.1f}% "

            f"{software_mae:12.6f}"

        )


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(
    result_rows
)


Path(
    "results/tables"
).mkdir(

    parents=True,

    exist_ok=True,

)


results_df.to_csv(

    OUTPUT_FILE,

    index=False,

)


# ============================================================
# Summary
# ============================================================

print()

print(
    "========================================"
)

print(
    "SENSITIVITY SUMMARY"
)

print(
    "========================================"
)


for device_id in DEVICE_IDS:

    subset = results_df[

        results_df[
            "device_template"
        ]
        ==
        device_id

    ].copy()


    lowest_ratio = (

        subset

        .sort_values(
            "on_off_ratio"
        )

        .iloc[0]

    )


    highest_test_ratio = (

        subset

        .sort_values(
            "on_off_ratio"
        )

        .iloc[-1]

    )


    print()

    print(
        device_id
    )


    print(

        f"  At ratio "
        f"{lowest_ratio['on_off_ratio']:.1f}:"

    )


    print(

        f"    Mean difference vs high-ratio reference = "
        f"{lowest_ratio['mean_abs_difference_vs_high_ratio']:.6f}"

    )


    print(

        f"    Argmax agreement vs high-ratio reference = "
        f"{100.0 * lowest_ratio['argmax_agreement_vs_high_ratio']:.1f}%"

    )


    print(

        f"  At ratio "
        f"{highest_test_ratio['on_off_ratio']:.1f}:"

    )


    print(

        f"    Mean difference vs high-ratio reference = "
        f"{highest_test_ratio['mean_abs_difference_vs_high_ratio']:.6f}"

    )


    print(

        f"    Argmax agreement vs high-ratio reference = "
        f"{100.0 * highest_test_ratio['argmax_agreement_vs_high_ratio']:.1f}%"

    )


# ============================================================
# Scientific interpretation
# ============================================================

print()

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "----------------------------------------"
)


print(

    "Unlike the absolute-conductance-scale test, "
    "this experiment changes the conductance ratio itself."

)


print()

print(

    "If output/error metrics change as the ratio changes, "
    "then ON/OFF ratio is a genuine device-dependent input "
    "to the CURRENT simulator."

)


print()

print(

    "This still does not make the simulator a complete "
    "material-physics model."

)


print()

print(

    "Thickness, geometry, line resistance, variability, "
    "noise, drift, switching kinetics, and other physical "
    "effects are not yet represented by this diagnostic."

)


print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)