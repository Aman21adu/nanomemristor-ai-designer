from pathlib import Path
import copy
import math

import pandas as pd
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from memristor_sim import MLP, hardware_forward
from device_model import load_device


# ============================================================
# PURPOSE
# ============================================================
#
# End-to-end controlled sensitivity experiment:
#
#     memristor ON/OFF ratio
#              ↓
#       crossbar behavior
#              ↓
#        neural network
#              ↓
#       MNIST accuracy
#
#
# This version uses:
#
#     ADC_BITS = 6
#
# so we can compare against the previous ADC-4 sensitivity
# experiment.
#
#
# Only the conductance ratio is changed.
#
# We keep fixed:
#
#     trained neural network
#     MNIST test set
#     crossbar size
#     weight precision
#     ADC precision
#     device conductance mode
#     device state capability
#
#
# IMPORTANT:
#
# The tested ratios are CONTROLLED HYPOTHETICAL VALUES.
#
# They are NOT claimed to be experimentally measured values
# for these devices.
#
# Nothing in device_profiles.csv is modified.
# ============================================================


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    "results/tables/"
    "ratio_accuracy_sensitivity_adc6.csv"
)


# ============================================================
# DEVICE TEMPLATES
# ============================================================

DEVICE_IDS = [

    "ZnO_01",
    "TaOx_01",
    "HfOx_02",

]


# ============================================================
# CONTROLLED ON/OFF RATIOS
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
# ACCELERATOR CONFIGURATION
# ============================================================
#
# Same configuration as the ADC-4 experiment except:
#
#     ADC_BITS = 6
#
# This lets us test whether ON/OFF-ratio sensitivity remains
# visible at a higher ADC precision that is closer to the
# configurations selected by the project optimizer.
# ============================================================

CROSSBAR_SIZE = 32
WEIGHT_BITS = 4
ADC_BITS = 6


# ============================================================
# HELPER:
# create profile with controlled ON/OFF ratio
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
    # Fix Gmax.
    #
    # Change Gmin only.
    #
    # Therefore:
    #
    #     Gmax / Gmin = desired ratio
    # --------------------------------------------------------

    gmax = 1.0

    gmin = (
        gmax
        / ratio
    )


    profile["gmax"] = gmax
    profile["gmin"] = gmin

    profile["ratio"] = ratio


    # --------------------------------------------------------
    # Equivalent normalized resistances
    #
    # These values are kept consistent for descriptive
    # purposes.
    #
    # crossbar_linear() currently works from Gmin/Gmax.
    # --------------------------------------------------------

    profile["ron"] = (
        1.0
        / gmax
    )

    profile["roff"] = (
        1.0
        / gmin
    )


    return profile


# ============================================================
# HELPER:
# evaluate full MNIST test set
# ============================================================

def evaluate_hardware_accuracy(
    model,
    test_loader,
    torch_device,
    device_profile,
):

    correct = 0
    total = 0
    actual_levels = None


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                torch_device
            )

            labels = labels.to(
                torch_device
            )


            (
                outputs,
                actual_levels,

            ) = hardware_forward(

                model,
                images,

                CROSSBAR_SIZE,
                WEIGHT_BITS,
                ADC_BITS,

                device_profile,

            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (

                predictions
                == labels

            ).sum().item()


            total += labels.size(
                0
            )


    accuracy = (

        100.0
        * correct
        / total

    )


    return (
        accuracy,
        actual_levels,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "END-TO-END ON/OFF RATIO ACCURACY TEST"
    )

    print(
        "ADC = 6 BITS"
    )

    print(
        "========================================"
    )


    # ========================================================
    # CPU / GPU
    # ========================================================

    torch_device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"

    )


    print()

    print(
        "Using:",
        torch_device
    )


    # ========================================================
    # Software baseline accuracy
    # ========================================================

    with open(

        "results/mnist_baseline_accuracy.txt",
        "r",

    ) as f:

        baseline_accuracy = float(

            f.read().strip()

        )


    print()

    print(
        "Software MNIST baseline:",
        f"{baseline_accuracy:.2f}%"
    )


    # ========================================================
    # Load trained model
    # ========================================================

    model = MLP().to(
        torch_device
    )


    model.load_state_dict(

        torch.load(

            "results/mnist_mlp_baseline.pt",

            map_location=torch_device,

        )

    )


    model.eval()


    # ========================================================
    # MNIST test data
    # ========================================================

    test_data = datasets.MNIST(

        root="data",

        train=False,

        download=False,

        transform=transforms.ToTensor(),

    )


    test_loader = DataLoader(

        test_data,

        batch_size=256,

        shuffle=False,

    )


    print()

    print(
        "MNIST test samples:",
        len(test_data)
    )


    print()

    print(
        "Fixed accelerator configuration:"
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

        "Only ON/OFF ratio is changed "
        "inside each device template."

    )


    print()

    print(

        "The test ratios are hypothetical "
        "controlled ablation values."

    )


    # ========================================================
    # Results
    # ========================================================

    result_rows = []


    # ========================================================
    # Device loop
    # ========================================================

    for device_id in DEVICE_IDS:

        original_profile = load_device(
            device_id
        )


        conductance_mode = str(

            original_profile.get(
                "conductance_mode",
                "UNKNOWN",
            )

        )


        physical_states = (

            original_profile.get(
                "states",
                float("nan"),
            )

        )


        original_gmin = float(
            original_profile["gmin"]
        )


        original_gmax = float(
            original_profile["gmax"]
        )


        original_ratio = (

            original_gmax
            / original_gmin

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


        print(
            "Original literature/model ratio:",
            f"{original_ratio:.6g}"
        )


        print()

        print(

            f"{'Ratio':>10} "
            f"{'Gmin':>12} "
            f"{'Levels':>8} "
            f"{'Accuracy':>12} "
            f"{'Loss':>12}"

        )


        # ====================================================
        # Ratio loop
        # ====================================================

        for ratio in TEST_RATIOS:

            controlled_profile = (
                make_ratio_profile(

                    original_profile,
                    ratio,

                )
            )


            (
                accuracy,
                actual_levels,

            ) = evaluate_hardware_accuracy(

                model,
                test_loader,
                torch_device,
                controlled_profile,

            )


            accuracy_loss = (

                baseline_accuracy
                - accuracy

            )


            # ------------------------------------------------
            # Descriptive flag:
            #
            # Is this controlled ratio approximately equal to
            # the original literature/model ratio?
            # ------------------------------------------------

            matches_original_ratio = bool(

                math.isclose(

                    ratio,
                    original_ratio,

                    rel_tol=0.02,
                    abs_tol=0.01,

                )

            )


            # ------------------------------------------------
            # Save row
            # ------------------------------------------------

            result_rows.append({

                "device_template":
                    device_id,

                "conductance_mode":
                    conductance_mode,

                "physical_states":
                    physical_states,

                "original_device_ratio":
                    original_ratio,

                "tested_on_off_ratio":
                    ratio,

                "matches_original_ratio":
                    matches_original_ratio,

                "gmax":
                    controlled_profile[
                        "gmax"
                    ],

                "gmin":
                    controlled_profile[
                        "gmin"
                    ],

                "crossbar_size":
                    CROSSBAR_SIZE,

                "requested_weight_bits":
                    WEIGHT_BITS,

                "adc_bits":
                    ADC_BITS,

                "effective_weight_levels":
                    int(
                        actual_levels
                    ),

                "software_baseline_accuracy":
                    baseline_accuracy,

                "hardware_accuracy":
                    accuracy,

                "accuracy_loss":
                    accuracy_loss,

                "experiment_type":
                    (
                        "CONTROLLED_RATIO_ABLATION_ADC6"
                    ),

            })


            print(

                f"{ratio:10.1f} "

                f"{controlled_profile['gmin']:12.6f} "

                f"{int(actual_levels):8d} "

                f"{accuracy:11.2f}% "

                f"{accuracy_loss:11.2f}"

            )


    # ========================================================
    # Save table
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
    # Summary
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "ADC-6 ACCURACY SENSITIVITY SUMMARY"
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


        best = results_df.loc[
            best_index
        ]


        print()

        print(
            device_id
        )


        print(

            f"  Ratio "
            f"{lowest['tested_on_off_ratio']:.1f}: "
            f"{lowest['hardware_accuracy']:.2f}%"

        )


        print(

            f"  Ratio "
            f"{highest['tested_on_off_ratio']:.1f}: "
            f"{highest['hardware_accuracy']:.2f}%"

        )


        print(

            f"  Accuracy difference "
            f"(high ratio - low ratio): "
            f"{highest['hardware_accuracy'] - lowest['hardware_accuracy']:.2f} pp"

        )


        print(

            f"  Best tested accuracy: "
            f"{best['hardware_accuracy']:.2f}% "
            f"at ratio "
            f"{best['tested_on_off_ratio']:.1f}"

        )


    # ========================================================
    # Scientific interpretation
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "IMPORTANT INTERPRETATION"
    )

    print(
        "========================================"
    )


    print()

    print(

        "This experiment repeats the ON/OFF-ratio "
        "ablation with a 6-bit ADC."

    )


    print()

    print(

        "If accuracy still changes substantially, then "
        "the device ON/OFF ratio remains an active "
        "accelerator parameter even with the higher "
        "ADC precision."

    )


    print()

    print(

        "The tested ratios are controlled hypothetical "
        "ablation values and are not new experimental "
        "measurements for the named devices."

    )


    print()

    print(

        "This demonstrates device-electrical-property "
        "sensitivity, not a complete material-to-device "
        "physics model."

    )


    print()

    print(

        "Thickness, geometry, variability, noise, drift, "
        "line resistance, switching kinetics, and energy "
        "effects still require separate physical models."

    )


    print()

    print(
        "Saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()