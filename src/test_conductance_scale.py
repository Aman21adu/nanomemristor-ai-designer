from pathlib import Path
import copy
import math

import pandas as pd
import torch

from device_model import load_device
from memristor_sim import crossbar_linear


# ============================================================
# PURPOSE
# ============================================================
#
# Test whether the CURRENT simulator is sensitive to:
#
#     absolute conductance magnitude
#
# when:
#
#     Gmin and Gmax are scaled together
#
# and therefore:
#
#     Gmax / Gmin
#
# remains unchanged.
#
#
# If outputs remain essentially identical, then absolute
# conductance scale is effectively cancelled by the present
# simulator equations.
#
#
# IMPORTANT:
#
# This describes the CURRENT mathematical simulator only.
#
# It does NOT mean real memristor hardware is independent
# of absolute conductance.
#
# Effects such as:
#
#     line resistance
#     IR drop
#     current magnitude
#     power
#     sensing limits
#     device heating
#
# can make absolute conductance important in real hardware.
# ============================================================


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    "results/tables/"
    "conductance_scale_invariance.csv"
)


# ============================================================
# DEVICES
# ============================================================

DEVICE_IDS = [

    "ZnO_01",
    "TaOx_01",
    "HfOx_02",
    "TiOx_03",

]


# ============================================================
# ACCELERATOR CONFIGURATION
# ============================================================

CROSSBAR_SIZE = 32

WEIGHT_BITS = 4

ADC_BITS = 6


# ============================================================
# CONDUCTANCE SCALE FACTORS
# ============================================================
#
# Both Gmin and Gmax are multiplied by the same factor.
#
# Therefore the ON/OFF ratio remains unchanged.
# ============================================================

SCALE_FACTORS = [

    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,

]


# ============================================================
# NUMERICAL TOLERANCE
# ============================================================

MAX_ACCEPTABLE_DIFFERENCE = 1e-5


# ============================================================
# DETERMINISTIC SYNTHETIC LAYER
# ============================================================

torch.manual_seed(
    42
)


BATCH_SIZE = 32

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
# HELPER:
# safely scale one device profile
# ============================================================

def make_scaled_profile(
    original_profile,
    scale_factor,
):

    profile = copy.deepcopy(
        original_profile
    )


    original_gmin = float(
        profile["gmin"]
    )


    original_gmax = float(
        profile["gmax"]
    )


    if (

        not math.isfinite(
            original_gmin
        )

        or

        not math.isfinite(
            original_gmax
        )

        or

        original_gmin <= 0

        or

        original_gmax <= original_gmin

    ):

        raise ValueError(

            f"{profile['device_id']} does not have "
            "a valid conductance range."

        )


    # --------------------------------------------------------
    # Scale both conductances
    # --------------------------------------------------------

    profile[
        "gmin"
    ] = (

        original_gmin
        * scale_factor

    )


    profile[
        "gmax"
    ] = (

        original_gmax
        * scale_factor

    )


    # --------------------------------------------------------
    # Keep descriptive resistance values consistent.
    #
    # R = 1 / G
    #
    # If G is multiplied by scale_factor,
    # R is divided by scale_factor.
    #
    # crossbar_linear() itself currently uses Gmin/Gmax.
    # --------------------------------------------------------

    ron = profile.get(
        "ron",
        float("nan")
    )


    roff = profile.get(
        "roff",
        float("nan")
    )


    try:

        ron = float(
            ron
        )

    except (
        TypeError,
        ValueError,
    ):

        ron = float(
            "nan"
        )


    try:

        roff = float(
            roff
        )

    except (
        TypeError,
        ValueError,
    ):

        roff = float(
            "nan"
        )


    if math.isfinite(
        ron
    ):

        profile[
            "ron"
        ] = (

            ron
            / scale_factor

        )


    if math.isfinite(
        roff
    ):

        profile[
            "roff"
        ] = (

            roff
            / scale_factor

        )


    # --------------------------------------------------------
    # Verify conductance ratio did not change
    # --------------------------------------------------------

    original_ratio = (

        original_gmax
        / original_gmin

    )


    scaled_ratio = (

        float(
            profile[
                "gmax"
            ]
        )

        /

        float(
            profile[
                "gmin"
            ]
        )

    )


    if not math.isclose(

        scaled_ratio,
        original_ratio,

        rel_tol=1e-10,
        abs_tol=1e-12,

    ):

        raise ValueError(

            "Conductance scaling accidentally changed "
            "the ON/OFF ratio."

        )


    return profile


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "ABSOLUTE CONDUCTANCE SCALE TEST"
    )

    print(
        "========================================"
    )


    print()

    print(
        "Configuration:"
    )


    print(

        f"Crossbar = "
        f"{CROSSBAR_SIZE}x{CROSSBAR_SIZE}"

    )


    print(

        f"Weight bits = "
        f"{WEIGHT_BITS}"

    )


    print(

        f"ADC bits = "
        f"{ADC_BITS}"

    )


    print()

    print(

        "Both Gmin and Gmax are scaled together, "
        "so their ratio remains unchanged."

    )


    print()

    print(

        "If output differences stay near floating-point "
        "noise, absolute conductance magnitude is "
        "effectively cancelled by the current simulator."

    )


    # ========================================================
    # RESULTS
    # ========================================================

    result_rows = []

    overall_pass = True


    # ========================================================
    # DEVICE LOOP
    # ========================================================

    for device_id in DEVICE_IDS:

        original_profile = load_device(
            device_id
        )


        original_gmin = float(
            original_profile[
                "gmin"
            ]
        )


        original_gmax = float(
            original_profile[
                "gmax"
            ]
        )


        original_ratio = (

            original_gmax
            / original_gmin

        )


        conductance_mode = str(

            original_profile.get(
                "conductance_mode",
                "UNKNOWN",
            )

        )


        print()

        print(
            "========================================"
        )

        print(
            "DEVICE:",
            device_id
        )

        print(
            "========================================"
        )


        print(
            "Mode:",
            conductance_mode
        )


        print(
            "Original Gmin:",
            original_gmin
        )


        print(
            "Original Gmax:",
            original_gmax
        )


        print(
            "Gmax/Gmin ratio:",
            original_ratio
        )


        # ====================================================
        # REFERENCE OUTPUT
        #
        # scale = 1.0
        # ====================================================

        reference_profile = (
            make_scaled_profile(

                original_profile,
                1.0,

            )
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
            "Scale comparison:"
        )


        print(

            f"{'Scale':>10} "
            f"{'Max abs diff':>16} "
            f"{'Mean abs diff':>16} "
            f"{'Argmax same':>14} "
            f"{'Levels':>8} "
            f"{'Pass':>8}"

        )


        device_pass = True


        # ====================================================
        # SCALE SWEEP
        # ====================================================

        for scale_factor in SCALE_FACTORS:

            scaled_profile = (
                make_scaled_profile(

                    original_profile,
                    scale_factor,

                )
            )


            (
                scaled_output,
                scaled_levels,

            ) = crossbar_linear(

                x=x,

                weight=weight,

                bias=bias,

                crossbar_size=CROSSBAR_SIZE,

                weight_bits=WEIGHT_BITS,

                adc_bits=ADC_BITS,

                device_profile=scaled_profile,

            )


            # ------------------------------------------------
            # Numerical output difference
            # ------------------------------------------------

            difference = (

                scaled_output
                - reference_output

            ).abs()


            max_difference = float(

                difference
                .max()
                .item()

            )


            mean_difference = float(

                difference
                .mean()
                .item()

            )


            # ------------------------------------------------
            # Argmax comparison
            # ------------------------------------------------

            scaled_argmax = torch.argmax(

                scaled_output,

                dim=1,

            )


            argmax_same = bool(

                torch.equal(

                    reference_argmax,
                    scaled_argmax,

                )

            )


            # ------------------------------------------------
            # Effective precision comparison
            # ------------------------------------------------

            levels_same = (

                int(
                    scaled_levels
                )

                ==

                int(
                    reference_levels
                )

            )


            # ------------------------------------------------
            # Test result
            # ------------------------------------------------

            scale_pass = (

                max_difference
                <= MAX_ACCEPTABLE_DIFFERENCE

                and

                argmax_same

                and

                levels_same

            )


            if not scale_pass:

                device_pass = False


            # ------------------------------------------------
            # Save evidence row
            # ------------------------------------------------

            result_rows.append({

                "device_id":
                    device_id,

                "conductance_mode":
                    conductance_mode,

                "experiment_type":
                    "ABSOLUTE_CONDUCTANCE_SCALE_ABLATION",

                "crossbar_size":
                    CROSSBAR_SIZE,

                "requested_weight_bits":
                    WEIGHT_BITS,

                "adc_bits":
                    ADC_BITS,

                "original_gmin":
                    original_gmin,

                "original_gmax":
                    original_gmax,

                "original_on_off_ratio":
                    original_ratio,

                "conductance_scale_factor":
                    scale_factor,

                "scaled_gmin":
                    float(
                        scaled_profile[
                            "gmin"
                        ]
                    ),

                "scaled_gmax":
                    float(
                        scaled_profile[
                            "gmax"
                        ]
                    ),

                "scaled_on_off_ratio":
                    (

                        float(
                            scaled_profile[
                                "gmax"
                            ]
                        )

                        /

                        float(
                            scaled_profile[
                                "gmin"
                            ]
                        )

                    ),

                "effective_weight_levels":
                    int(
                        scaled_levels
                    ),

                "max_abs_output_difference":
                    max_difference,

                "mean_abs_output_difference":
                    mean_difference,

                "argmax_same":
                    argmax_same,

                "levels_same":
                    levels_same,

                "numerical_tolerance":
                    MAX_ACCEPTABLE_DIFFERENCE,

                "scale_invariance_pass":
                    scale_pass,

                "scientific_interpretation":
                    (
                        "Tests whether the current simulator "
                        "is invariant to absolute conductance "
                        "scale when ON/OFF ratio is preserved."
                    ),

                "limitation":
                    (
                        "Does not imply real hardware is "
                        "independent of absolute conductance; "
                        "IR drop, line resistance, power and "
                        "sensing effects are not modeled here."
                    ),

            })


            print(

                f"{scale_factor:10.3g} "

                f"{max_difference:16.8e} "

                f"{mean_difference:16.8e} "

                f"{str(argmax_same):>14} "

                f"{int(scaled_levels):8d} "

                f"{str(scale_pass):>8}"

            )


        # ====================================================
        # DEVICE RESULT
        # ====================================================

        print()


        if device_pass:

            print(

                "RESULT: SCALE-INVARIANT "
                "within numerical tolerance."

            )


        else:

            print(

                "RESULT: Conductance scaling changed "
                "the simulator output materially."

            )

            overall_pass = False


    # ========================================================
    # SAVE CSV
    # ========================================================

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


    # ========================================================
    # OVERALL RESULT
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "OVERALL RESULT"
    )

    print(
        "========================================"
    )


    print()


    if overall_pass:

        print(
            "PASS"
        )


        print()

        print(

            "For every tested device and scale factor, "
            "multiplying Gmin and Gmax together did not "
            "materially change the crossbar output."

        )


        print()

        print(

            "The CURRENT simulator is therefore "
            "effectively invariant to absolute "
            "conductance scale when ON/OFF ratio "
            "is preserved."

        )


    else:

        print(
            "NOT SCALE-INVARIANT"
        )


        print()

        print(

            "At least one device showed a material "
            "output change when absolute conductance "
            "was scaled."

        )


    # ========================================================
    # QUANTITATIVE SUMMARY
    # ========================================================

    print()

    print(
        "Maximum difference observed:"
    )


    print(

        f"{results_df['max_abs_output_difference'].max():.8e}"

    )


    print()

    print(
        "Rows saved:",
        len(
            results_df
        )
    )


    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


    # ========================================================
    # SCIENTIFIC INTERPRETATION
    # ========================================================

    print()

    print(
        "IMPORTANT INTERPRETATION"
    )

    print(
        "----------------------------------------"
    )


    print()

    print(

        "This experiment describes the CURRENT "
        "mathematical simulator."

    )


    print()

    print(

        "It does NOT mean that real memristor "
        "hardware is independent of absolute "
        "conductance."

    )


    print()

    print(

        "Absolute conductance can influence line "
        "resistance, IR drop, current magnitude, "
        "power, sensing limits, heating, and other "
        "nonidealities that are not represented by "
        "the present simplified model."

    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()