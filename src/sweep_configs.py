import argparse
import itertools

import pandas as pd
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from memristor_sim import MLP, hardware_forward
from device_model import load_device
from config_space import (
    CROSSBAR_SIZES,
    WEIGHT_BITS,
    ADC_BITS,
)


def main():

    # ============================================================
    # Command-line argument
    # ============================================================

    parser = argparse.ArgumentParser(
        description="Sweep accelerator configurations for one memristor device."
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device ID from data/device_profiles.csv",
    )

    args = parser.parse_args()


    # ============================================================
    # Load device
    # ============================================================

    device_profile = load_device(
        args.device
    )

    print()
    print("Device:")
    print("-----------------------------")
    print("ID:", device_profile["device_id"])
    print("Family:", device_profile["family"])
    print("Stack:", device_profile["stack"])
    print("Gmin:", device_profile["gmin"])
    print("Gmax:", device_profile["gmax"])
    print("States:", device_profile["states"])
    print(
        "Conductance mode:",
        device_profile["conductance_mode"],
    )
    print(
        "State count status:",
        device_profile["state_count_status"],
    )
    print(
        "Parameter source:",
        device_profile["parameter_source"],
    )


    # ============================================================
    # Basic parameter validation
    # ============================================================

    if (
        device_profile["parameter_source"]
        == "insufficient"
    ):

        raise ValueError(
            f"{args.device} does not currently have enough "
            f"conductance-range parameters."
        )


    conductance_mode = str(
        device_profile.get(
            "conductance_mode",
            "UNKNOWN",
        )
    ).upper()


    state_count_status = str(
        device_profile.get(
            "state_count_status",
            "UNKNOWN",
        )
    ).upper()


    reported_states = (
        device_profile["states"]
    )


    # ============================================================
    # Reject unsupported device modes
    # ============================================================

    if (
        conductance_mode
        == "CONTINUOUS_QUANTIZED"
    ):

        raise ValueError(
            f"{args.device} uses CONTINUOUS_QUANTIZED behavior "
            f"and requires a dedicated instability/noise model."
        )


    if (
        conductance_mode
        == "UNKNOWN"
    ):

        raise ValueError(
            f"{args.device} has UNKNOWN conductance behavior "
            f"and cannot currently be swept defensibly."
        )


    supported_modes = {
        "DISCRETE_BINARY",
        "DISCRETE_MULTILEVEL",
        "ANALOG",
        "GRADUAL_MULTILEVEL",
    }


    if (
        conductance_mode
        not in supported_modes
    ):

        raise ValueError(
            f"{args.device} has unsupported conductance mode: "
            f"{conductance_mode}"
        )


    # ============================================================
    # CPU / GPU
    # ============================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Using:", device)


    # ============================================================
    # Baseline accuracy
    # ============================================================

    with open(
        "results/mnist_baseline_accuracy.txt",
        "r",
    ) as f:

        baseline_accuracy = float(
            f.read().strip()
        )


    # ============================================================
    # Load trained model
    # ============================================================

    model = MLP().to(
        device
    )

    model.load_state_dict(
        torch.load(
            "results/mnist_mlp_baseline.pt",
            map_location=device,
        )
    )

    model.eval()


    # ============================================================
    # MNIST test data
    # ============================================================

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


    # ============================================================
    # Search space
    # ============================================================

    configs = list(
        itertools.product(
            CROSSBAR_SIZES,
            WEIGHT_BITS,
            ADC_BITS,
        )
    )

    total_configs = len(
        configs
    )

    print()
    print(
        "Total configurations:",
        total_configs,
    )


    results = []


    # ============================================================
    # Sweep configurations
    # ============================================================

    for number, (
        crossbar,
        weight_bits,
        adc_bits,

    ) in enumerate(
        configs,
        start=1,
    ):

        correct = 0
        total = 0
        actual_levels = None


        # --------------------------------------------------------
        # Full MNIST evaluation
        # --------------------------------------------------------

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(
                    device
                )

                labels = labels.to(
                    device
                )

                (
                    outputs,
                    actual_levels,

                ) = hardware_forward(
                    model,
                    images,
                    crossbar,
                    weight_bits,
                    adc_bits,
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


        # ========================================================
        # Accuracy
        # ========================================================

        accuracy = (
            100.0
            * correct
            / total
        )

        accuracy_loss = (
            baseline_accuracy
            - accuracy
        )


        # ========================================================
        # Mapping metadata
        # ========================================================

        mapping_strategy = None
        precision_basis = None
        slices_per_branch = None
        physical_cells_per_weight = None


        # --------------------------------------------------------
        # Binary device
        #
        # One cell has only 2 physical states.
        #
        # Higher accelerator precision is created by combining
        # several binary cells through bit slicing.
        # --------------------------------------------------------

        if (
            conductance_mode
            == "DISCRETE_BINARY"
        ):

            mapping_strategy = (
                "BINARY_BIT_SLICED_DIFFERENTIAL"
            )

            precision_basis = (
                "BIT_SLICED_BINARY_MAPPING"
            )

            slices_per_branch = (
                weight_bits - 1
            )

            physical_cells_per_weight = (
                2
                * slices_per_branch
            )


        # --------------------------------------------------------
        # Fixed discrete multilevel device
        # --------------------------------------------------------

        elif (
            conductance_mode
            == "DISCRETE_MULTILEVEL"
        ):

            mapping_strategy = (
                "SINGLE_DIFFERENTIAL_PAIR"
            )

            physical_cells_per_weight = 2


            if (
                pd.notna(
                    reported_states
                )
                and float(
                    reported_states
                ) >= 2
            ):

                if (
                    state_count_status
                    == "REPORTED"
                ):

                    precision_basis = (
                        "REPORTED_DEVICE_STATE_CAP"
                    )


                elif (
                    state_count_status
                    == "DERIVED"
                ):

                    precision_basis = (
                        "DERIVED_DEVICE_STATE_CAP"
                    )


                else:

                    precision_basis = (
                        "DEVICE_STATE_CAP"
                    )


            else:

                precision_basis = (
                    "UNRESOLVED"
                )


        # --------------------------------------------------------
        # Analog device
        # --------------------------------------------------------

        elif (
            conductance_mode
            == "ANALOG"
        ):

            mapping_strategy = (
                "IDEALIZED_SINGLE_PAIR_ANALOG"
            )

            precision_basis = (
                "IDEALIZED_ANALOG_MAPPING"
            )

            physical_cells_per_weight = 2


        # --------------------------------------------------------
        # Gradual multilevel device
        # --------------------------------------------------------

        elif (
            conductance_mode
            == "GRADUAL_MULTILEVEL"
        ):

            mapping_strategy = (
                "IDEALIZED_SINGLE_PAIR_GRADUAL"
            )

            precision_basis = (
                "IDEALIZED_GRADUAL_MAPPING"
            )

            physical_cells_per_weight = 2


        # ========================================================
        # Save one configuration
        # ========================================================

        results.append({

            "device_id":
                device_profile["device_id"],

            "technology_family":
                device_profile["family"],

            "conductance_mode":
                conductance_mode,

            "state_count_status":
                state_count_status,

            "reported_conductance_states":
                reported_states,

            "mapping_strategy":
                mapping_strategy,

            "precision_basis":
                precision_basis,

            "parameter_source":
                device_profile[
                    "parameter_source"
                ],

            "crossbar_size":
                crossbar,

            "requested_weight_bits":
                weight_bits,

            # -----------------------------------------------
            # Accelerator-level effective signed levels.
            #
            # For binary devices this is NOT the physical
            # state count of one memristor.
            # -----------------------------------------------

            "effective_weight_levels":
                actual_levels,

            # -----------------------------------------------
            # Old column retained temporarily for compatibility
            # with existing project scripts.
            # -----------------------------------------------

            "effective_conductance_levels":
                actual_levels,

            "slices_per_branch":
                slices_per_branch,

            "physical_cells_per_weight":
                physical_cells_per_weight,

            "adc_bits":
                adc_bits,

            "accuracy":
                accuracy,

            "accuracy_loss":
                accuracy_loss,
        })


        # ========================================================
        # Console summary
        # ========================================================

        mapping_text = ""


        if (
            conductance_mode
            == "DISCRETE_BINARY"
        ):

            mapping_text = (
                f", slices={slices_per_branch}"
                f", cells/weight={physical_cells_per_weight}"
            )


        print(
            f"[{number:02d}/{total_configs}] "
            f"{crossbar}x{crossbar}, "
            f"W={weight_bits}, "
            f"ADC={adc_bits}, "
            f"levels={actual_levels}, "
            f"basis={precision_basis}"
            f"{mapping_text} "
            f"-> {accuracy:.2f}%"
        )


    # ============================================================
    # DataFrame
    # ============================================================

    df = pd.DataFrame(
        results
    )

    df = df.sort_values(
        by="accuracy",
        ascending=False,
    )


    # ============================================================
    # Save CSV
    # ============================================================

    output_path = (
        "results/tables/"
        f"{args.device}_config_sweep.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )


    # ============================================================
    # Top configurations
    # ============================================================

    print()
    print(
        "TOP CONFIGURATIONS"
    )

    print(
        df.head(
            10
        ).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":

    main()