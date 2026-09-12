from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

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


RESULT_COLUMNS = [
    "device_id",
    "technology_family",
    "conductance_mode",
    "state_count_status",
    "reported_conductance_states",
    "mapping_strategy",
    "precision_basis",
    "parameter_source",
    "crossbar_size",
    "requested_weight_bits",
    "effective_weight_levels",
    "effective_conductance_levels",
    "slices_per_branch",
    "physical_cells_per_weight",
    "adc_bits",
    "accuracy",
    "accuracy_loss",
]


def config_key(crossbar, weight_bits, adc_bits):
    return (
        int(crossbar),
        int(weight_bits),
        int(adc_bits),
    )


def expected_config_keys(configs):
    return {
        config_key(crossbar, weight_bits, adc_bits)
        for crossbar, weight_bits, adc_bits in configs
    }


def dataframe_config_keys(df):
    return {
        config_key(
            row.crossbar_size,
            row.requested_weight_bits,
            row.adc_bits,
        )
        for row in df.itertuples(index=False)
    }


def validate_saved_rows(
    df,
    *,
    device_id,
    valid_config_keys,
    path,
    allow_partial,
):
    missing_columns = [
        column
        for column in RESULT_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} is missing required columns: "
            f"{missing_columns}"
        )

    if df.empty:
        if allow_partial:
            return
        raise ValueError(f"{path} is empty.")

    device_ids = set(
        df["device_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    if device_ids != {str(device_id)}:
        raise ValueError(
            f"{path} contains unexpected device IDs: "
            f"{sorted(device_ids)}"
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "crossbar_size",
                "requested_weight_bits",
                "adc_bits",
            ]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{path} contains {duplicate_count} duplicate "
            "accelerator configurations."
        )

    actual_keys = dataframe_config_keys(df)

    unexpected = actual_keys - valid_config_keys

    if unexpected:
        raise ValueError(
            f"{path} contains configurations outside "
            f"config_space.py: {sorted(unexpected)[:10]}"
        )

    if not allow_partial and actual_keys != valid_config_keys:
        missing = valid_config_keys - actual_keys
        raise ValueError(
            f"{path} is incomplete. Missing "
            f"{len(missing)} configurations. "
            f"Preview: {sorted(missing)[:10]}"
        )


def atomic_write_csv(df, path):
    """
    Save through a temporary file and atomically replace the target.

    If the process is interrupted while writing, the previous valid
    checkpoint remains available.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = path.with_name(
        path.name + ".tmp"
    )

    df.to_csv(
        temp_path,
        index=False,
    )

    os.replace(
        temp_path,
        path,
    )


def mapping_metadata(
    conductance_mode,
    state_count_status,
    reported_states,
    weight_bits,
):
    mapping_strategy = None
    precision_basis = None
    slices_per_branch = None
    physical_cells_per_weight = None

    if conductance_mode == "DISCRETE_BINARY":
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
            2 * slices_per_branch
        )

    elif conductance_mode == "DISCRETE_MULTILEVEL":
        mapping_strategy = (
            "SINGLE_DIFFERENTIAL_PAIR"
        )

        physical_cells_per_weight = 2

        if (
            pd.notna(reported_states)
            and float(reported_states) >= 2
        ):
            if state_count_status == "REPORTED":
                precision_basis = (
                    "REPORTED_DEVICE_STATE_CAP"
                )
            elif state_count_status == "DERIVED":
                precision_basis = (
                    "DERIVED_DEVICE_STATE_CAP"
                )
            else:
                precision_basis = (
                    "DEVICE_STATE_CAP"
                )
        else:
            precision_basis = "UNRESOLVED"

    elif conductance_mode == "ANALOG":
        mapping_strategy = (
            "IDEALIZED_SINGLE_PAIR_ANALOG"
        )

        precision_basis = (
            "IDEALIZED_ANALOG_MAPPING"
        )

        physical_cells_per_weight = 2

    elif conductance_mode == "GRADUAL_MULTILEVEL":
        mapping_strategy = (
            "IDEALIZED_SINGLE_PAIR_GRADUAL"
        )

        precision_basis = (
            "IDEALIZED_GRADUAL_MAPPING"
        )

        physical_cells_per_weight = 2

    else:
        raise ValueError(
            f"Unsupported conductance mode: "
            f"{conductance_mode}"
        )

    return (
        mapping_strategy,
        precision_basis,
        slices_per_branch,
        physical_cells_per_weight,
    )


def print_top_configurations(df, output_path):
    print()
    print("TOP CONFIGURATIONS")
    print()

    print(
        df.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep accelerator configurations for one memristor "
            "device with checkpoint/resume support."
        )
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device ID from data/device_profiles.csv",
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Discard any existing checkpoint/final sweep for this "
            "device and start all configurations again."
        ),
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

    if conductance_mode == "CONTINUOUS_QUANTIZED":
        raise ValueError(
            f"{args.device} uses CONTINUOUS_QUANTIZED behavior "
            f"and requires a dedicated instability/noise model."
        )

    if conductance_mode == "UNKNOWN":
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

    if conductance_mode not in supported_modes:
        raise ValueError(
            f"{args.device} has unsupported conductance mode: "
            f"{conductance_mode}"
        )

    # ============================================================
    # Search space + output paths
    # ============================================================

    configs = list(
        itertools.product(
            CROSSBAR_SIZES,
            WEIGHT_BITS,
            ADC_BITS,
        )
    )

    total_configs = len(configs)
    valid_config_keys = expected_config_keys(
        configs
    )

    output_directory = Path(
        "results/tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / f"{args.device}_config_sweep.csv"
    )

    checkpoint_path = (
        output_directory
        / f"{args.device}_config_sweep.checkpoint.csv"
    )

    print()
    print(
        "Total configurations:",
        total_configs,
    )

    # ============================================================
    # Optional clean restart
    # ============================================================

    if args.restart:
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        if output_path.exists():
            output_path.unlink()

        print(
            "Restart requested: previous checkpoint/final "
            "output removed."
        )

    # ============================================================
    # If a complete final sweep already exists, do not rerun it
    # ============================================================

    if output_path.exists():
        final_df = pd.read_csv(
            output_path
        )

        try:
            validate_saved_rows(
                final_df,
                device_id=args.device,
                valid_config_keys=valid_config_keys,
                path=output_path,
                allow_partial=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"Existing final sweep is invalid or incomplete:\n"
                f"{exc}\n"
                "Use --restart only if you intentionally want to "
                "discard it and rerun the complete sweep."
            ) from exc

        if checkpoint_path.exists():
            checkpoint_path.unlink()

        final_df = final_df.sort_values(
            by="accuracy",
            ascending=False,
        )

        print()
        print(
            "Complete sweep already exists. "
            "No configurations need to be rerun."
        )

        print_top_configurations(
            final_df,
            output_path,
        )

        return

    # ============================================================
    # Load checkpoint, if present
    # ============================================================

    if checkpoint_path.exists():
        checkpoint_df = pd.read_csv(
            checkpoint_path
        )

        validate_saved_rows(
            checkpoint_df,
            device_id=args.device,
            valid_config_keys=valid_config_keys,
            path=checkpoint_path,
            allow_partial=True,
        )

        results = (
            checkpoint_df[
                RESULT_COLUMNS
            ]
            .to_dict(
                orient="records"
            )
        )

        completed_keys = dataframe_config_keys(
            checkpoint_df
        )

        print()
        print(
            f"Checkpoint found: "
            f"{len(completed_keys)}/{total_configs} "
            "configurations already completed."
        )
        print(
            f"Remaining configurations: "
            f"{total_configs - len(completed_keys)}"
        )

    else:
        results = []
        completed_keys = set()

        print()
        print(
            "No checkpoint found. "
            "Starting a new sweep."
        )

    # ============================================================
    # If checkpoint itself already contains everything, finalize
    # ============================================================

    if completed_keys == valid_config_keys:
        df = pd.DataFrame(
            results,
            columns=RESULT_COLUMNS,
        )

        df = df.sort_values(
            by="accuracy",
            ascending=False,
        )

        atomic_write_csv(
            df,
            output_path,
        )

        checkpoint_path.unlink(
            missing_ok=True
        )

        print()
        print(
            "Checkpoint already contained the complete sweep. "
            "Final CSV created without rerunning simulations."
        )

        print_top_configurations(
            df,
            output_path,
        )

        return

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
    # Sweep only missing configurations
    # ============================================================

    for number, (
        crossbar,
        weight_bits,
        adc_bits,
    ) in enumerate(
        configs,
        start=1,
    ):
        key = config_key(
            crossbar,
            weight_bits,
            adc_bits,
        )

        if key in completed_keys:
            continue

        correct = 0
        total = 0
        actual_levels = None

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

        accuracy = (
            100.0
            * correct
            / total
        )

        accuracy_loss = (
            baseline_accuracy
            - accuracy
        )

        (
            mapping_strategy,
            precision_basis,
            slices_per_branch,
            physical_cells_per_weight,
        ) = mapping_metadata(
            conductance_mode=conductance_mode,
            state_count_status=state_count_status,
            reported_states=reported_states,
            weight_bits=weight_bits,
        )

        result_row = {
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

            "effective_weight_levels":
                actual_levels,

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
        }

        results.append(
            result_row
        )

        completed_keys.add(
            key
        )

        checkpoint_df = pd.DataFrame(
            results,
            columns=RESULT_COLUMNS,
        )

        atomic_write_csv(
            checkpoint_df,
            checkpoint_path,
        )

        mapping_text = ""

        if conductance_mode == "DISCRETE_BINARY":
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
            f"-> {accuracy:.2f}% "
            f"[checkpoint "
            f"{len(completed_keys)}/{total_configs}]"
        )

    # ============================================================
    # Validate + finalize
    # ============================================================

    df = pd.DataFrame(
        results,
        columns=RESULT_COLUMNS,
    )

    validate_saved_rows(
        df,
        device_id=args.device,
        valid_config_keys=valid_config_keys,
        path=checkpoint_path,
        allow_partial=False,
    )

    df = df.sort_values(
        by="accuracy",
        ascending=False,
    )

    atomic_write_csv(
        df,
        output_path,
    )

    checkpoint_path.unlink(
        missing_ok=True
    )

    print()
    print(
        "Sweep complete. "
        "Checkpoint removed after final CSV was saved."
    )

    print_top_configurations(
        df,
        output_path,
    )


if __name__ == "__main__":
    main()
