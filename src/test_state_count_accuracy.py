from pathlib import Path
import copy

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
# Controlled end-to-end sensitivity experiment:
#
#     physical memristor conductance-state count
#                     ↓
#       effective signed neural-weight levels
#                     ↓
#              crossbar computation
#                     ↓
#               MNIST accuracy
#
#
# We vary ONLY:
#
#     physical conductance states
#
#
# We keep fixed:
#
#     trained neural network
#     MNIST test set
#     ON/OFF ratio
#     absolute normalized conductance range
#     crossbar size
#     requested weight bits
#     ADC bits
#     mapping strategy
#
#
# IMPORTANT
# ============================================================
#
# The tested state counts are CONTROLLED HYPOTHETICAL VALUES.
#
# They are NOT claimed to be measured state counts for
# TaOx_01 or any other physical device.
#
# TaOx_01 is used only as a DISCRETE_MULTILEVEL mapping
# template.
#
# Nothing in device_profiles.csv is modified.
# ============================================================


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FILE = (
    "results/tables/"
    "state_count_accuracy_sensitivity.csv"
)


# ============================================================
# TEMPLATE DEVICE
# ============================================================
#
# TaOx_01 is useful because the current simulator already
# treats it as:
#
#     DISCRETE_MULTILEVEL
#
# with a single differential pair.
#
# We replace its state count during this controlled ablation.
# ============================================================

TEMPLATE_DEVICE_ID = "TaOx_01"


# ============================================================
# CONTROLLED PHYSICAL STATE COUNTS
# ============================================================
#
# For a differential pair:
#
#     N physical states
#
# gives a maximum of:
#
#     2N - 1 signed levels
#
#
# With requested W = 4:
#
#     requested signed levels = 15
#
#
# Therefore:
#
# N = 2  -> 3 levels
# N = 3  -> 5 levels
# N = 4  -> 7 levels
# N = 5  -> 9 levels
# N = 6  -> 11 levels
# N = 7  -> 13 levels
# N = 8  -> 15 levels
# N = 16 -> still 15 because requested W=4 caps it
# ============================================================

TEST_STATE_COUNTS = [

    2,
    3,
    4,
    5,
    6,
    7,
    8,
    16,

]


# ============================================================
# FIXED DEVICE CONDUCTANCE WINDOW
# ============================================================
#
# Keep:
#
#     Gmax / Gmin = 1000
#
# for every state-count experiment.
#
# This prevents ON/OFF ratio from becoming a confounding
# variable.
# ============================================================

FIXED_ON_OFF_RATIO = 1000.0

FIXED_GMAX = 1.0

FIXED_GMIN = (
    FIXED_GMAX
    / FIXED_ON_OFF_RATIO
)


# ============================================================
# FIXED ACCELERATOR CONFIGURATION
# ============================================================
#
# ADC = 6 because our previous experiment showed that
# conductance-ratio effects become relatively small here.
#
# This makes it a good configuration for asking whether
# physical state capability still matters.
# ============================================================

CROSSBAR_SIZE = 32

WEIGHT_BITS = 4

ADC_BITS = 6


# ============================================================
# CREATE CONTROLLED DEVICE PROFILE
# ============================================================

def make_state_count_profile(
    original_profile,
    physical_states,
):

    if physical_states < 2:

        raise ValueError(
            "Physical state count must be at least 2."
        )


    profile = copy.deepcopy(
        original_profile
    )


    # --------------------------------------------------------
    # Force discrete multilevel behavior for this controlled
    # experiment.
    # --------------------------------------------------------

    profile[
        "conductance_mode"
    ] = "DISCRETE_MULTILEVEL"


    # --------------------------------------------------------
    # Controlled physical state count
    # --------------------------------------------------------

    profile[
        "states"
    ] = float(
        physical_states
    )


    # --------------------------------------------------------
    # Fixed conductance range
    # --------------------------------------------------------

    profile[
        "gmax"
    ] = FIXED_GMAX


    profile[
        "gmin"
    ] = FIXED_GMIN


    profile[
        "ratio"
    ] = FIXED_ON_OFF_RATIO


    # --------------------------------------------------------
    # Equivalent normalized resistances
    #
    # These are kept internally consistent.
    #
    # The current crossbar implementation directly uses
    # Gmin and Gmax.
    # --------------------------------------------------------

    profile[
        "ron"
    ] = (
        1.0
        / FIXED_GMAX
    )


    profile[
        "roff"
    ] = (
        1.0
        / FIXED_GMIN
    )


    # --------------------------------------------------------
    # This is a hypothetical controlled ablation, not
    # literature evidence.
    # --------------------------------------------------------

    profile[
        "state_count_status"
    ] = "CONTROLLED_ABLATION"


    return profile


# ============================================================
# FULL MNIST EVALUATION
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
        "PHYSICAL STATE-COUNT ACCURACY TEST"
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
    # SOFTWARE BASELINE
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
    # LOAD TRAINED MODEL
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
    # LOAD MNIST TEST SET
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


    # ========================================================
    # LOAD TEMPLATE DEVICE
    # ========================================================

    original_profile = load_device(
        TEMPLATE_DEVICE_ID
    )


    print()

    print(
        "Template device:",
        TEMPLATE_DEVICE_ID
    )


    print(

        "Template mode:",
        original_profile[
            "conductance_mode"
        ]

    )


    print(

        "Original template physical states:",
        original_profile[
            "states"
        ]

    )


    # ========================================================
    # CONTROL VARIABLES
    # ========================================================

    print()

    print(
        "Fixed experiment parameters:"
    )


    print(

        f"ON/OFF ratio = "
        f"{FIXED_ON_OFF_RATIO:.0f}"

    )


    print(

        f"Gmax = "
        f"{FIXED_GMAX}"

    )


    print(

        f"Gmin = "
        f"{FIXED_GMIN}"

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

        "Only physical conductance-state count "
        "is changed."

    )


    print()

    print(

        "State counts are controlled hypothetical "
        "ablation values."

    )


    # ========================================================
    # EXPECTED REQUESTED LEVELS
    # ========================================================

    requested_signed_levels = (

        (2 ** WEIGHT_BITS)
        - 1

    )


    print()

    print(

        "Requested signed weight levels:",
        requested_signed_levels

    )


    # ========================================================
    # RESULTS
    # ========================================================

    result_rows = []


    print()

    print(

        f"{'States':>8} "
        f"{'Physical cap':>14} "
        f"{'Effective':>12} "
        f"{'Accuracy':>12} "
        f"{'Loss':>12}"

    )


    # ========================================================
    # STATE-COUNT SWEEP
    # ========================================================

    for physical_states in TEST_STATE_COUNTS:

        controlled_profile = (
            make_state_count_profile(

                original_profile,
                physical_states,

            )
        )


        # ----------------------------------------------------
        # Theoretical maximum signed levels
        # ----------------------------------------------------

        physical_signed_level_cap = (

            (2 * physical_states)
            - 1

        )


        expected_effective_levels = min(

            requested_signed_levels,
            physical_signed_level_cap,

        )


        # ----------------------------------------------------
        # Full MNIST hardware evaluation
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Consistency check
        # ----------------------------------------------------

        if (

            int(
                actual_levels
            )

            !=

            int(
                expected_effective_levels
            )

        ):

            raise ValueError(

                "Simulator effective-level result does not "
                "match the expected physical state cap."

            )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        result_rows.append({

            "template_device":
                TEMPLATE_DEVICE_ID,

            "experiment_type":
                "CONTROLLED_STATE_COUNT_ABLATION",

            "conductance_mode":
                "DISCRETE_MULTILEVEL",

            "physical_conductance_states":
                physical_states,

            "physical_signed_level_cap":
                physical_signed_level_cap,

            "requested_weight_bits":
                WEIGHT_BITS,

            "requested_signed_weight_levels":
                requested_signed_levels,

            "effective_weight_levels":
                int(
                    actual_levels
                ),

            "fixed_on_off_ratio":
                FIXED_ON_OFF_RATIO,

            "gmax":
                FIXED_GMAX,

            "gmin":
                FIXED_GMIN,

            "crossbar_size":
                CROSSBAR_SIZE,

            "adc_bits":
                ADC_BITS,

            "software_baseline_accuracy":
                baseline_accuracy,

            "hardware_accuracy":
                accuracy,

            "accuracy_loss":
                accuracy_loss,

        })


        print(

            f"{physical_states:8d} "

            f"{physical_signed_level_cap:14d} "

            f"{int(actual_levels):12d} "

            f"{accuracy:11.2f}% "

            f"{accuracy_loss:11.2f}"

        )


    # ========================================================
    # SAVE
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
    # SUMMARY
    # ========================================================

    print()

    print(
        "========================================"
    )

    print(
        "STATE-COUNT SENSITIVITY SUMMARY"
    )

    print(
        "========================================"
    )


    lowest = (

        results_df

        .sort_values(
            "physical_conductance_states"
        )

        .iloc[0]

    )


    highest = (

        results_df

        .sort_values(
            "physical_conductance_states"
        )

        .iloc[-1]

    )


    best_index = (

        results_df[
            "hardware_accuracy"
        ]

        .idxmax()

    )


    best = results_df.loc[
        best_index
    ]


    print()

    print(

        f"Lowest tested state count: "
        f"{int(lowest['physical_conductance_states'])}"

    )


    print(

        f"  Effective levels: "
        f"{int(lowest['effective_weight_levels'])}"

    )


    print(

        f"  Accuracy: "
        f"{lowest['hardware_accuracy']:.2f}%"

    )


    print()

    print(

        f"Highest tested state count: "
        f"{int(highest['physical_conductance_states'])}"

    )


    print(

        f"  Effective levels: "
        f"{int(highest['effective_weight_levels'])}"

    )


    print(

        f"  Accuracy: "
        f"{highest['hardware_accuracy']:.2f}%"

    )


    print()

    print(

        f"Accuracy difference "
        f"(highest states - lowest states): "
        f"{highest['hardware_accuracy'] - lowest['hardware_accuracy']:.2f} pp"

    )


    print()

    print(

        f"Best tested accuracy: "
        f"{best['hardware_accuracy']:.2f}%"

    )


    print(

        f"Best physical state count: "
        f"{int(best['physical_conductance_states'])}"

    )


    print(

        f"Best effective weight levels: "
        f"{int(best['effective_weight_levels'])}"

    )


    # ========================================================
    # SATURATION CHECK
    # ========================================================

    saturation_rows = results_df[

        results_df[
            "effective_weight_levels"
        ]
        ==
        requested_signed_levels

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


        print()

        print(

            "Minimum physical state count needed "
            "to reach the requested 15 signed levels:",
            minimum_saturating_states

        )


    # ========================================================
    # INTERPRETATION
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

        "This experiment changes only the assumed "
        "physical conductance-state capability."

    )


    print()

    print(

        "If MNIST accuracy changes as physical state "
        "count changes, then state capability directly "
        "propagates through the current accelerator "
        "model to application-level accuracy."

    )


    print()

    print(

        "Once the physical state count is high enough "
        "to satisfy the requested neural-weight "
        "precision, additional physical states should "
        "not provide extra precision in this W=4 test."

    )


    print()

    print(

        "The tested state counts are controlled "
        "hypothetical values."

    )


    print()

    print(

        "They must NOT be presented as measured "
        "properties of TaOx_01."

    )


    print()

    print(

        "This remains a device-level electrical "
        "abstraction, not a complete material-physics "
        "simulation."

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