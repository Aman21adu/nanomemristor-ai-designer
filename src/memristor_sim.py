import argparse
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from device_model import load_device


# ============================================================
# Neural-network architecture
# ============================================================

class MLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                28 * 28,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                10
            )
        )


    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


# ============================================================
# Helper: validate physical state count
# ============================================================

def valid_state_count(
    value
):

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return False


    return (
        math.isfinite(value)
        and value >= 2
    )


# ============================================================
# Weight -> conductance mapping
#
# Used for:
#
#   DISCRETE_MULTILEVEL
#   ANALOG
#   GRADUAL_MULTILEVEL
#
# DISCRETE_BINARY uses a separate bit-sliced path inside
# crossbar_linear().
# ============================================================

def map_weights_to_conductance(
    weights,
    requested_bits,
    device_profile
):

    # --------------------------------------------------------
    # Device conductance range
    # --------------------------------------------------------

    gmin = float(
        device_profile["gmin"]
    )

    gmax = float(
        device_profile["gmax"]
    )


    if (
        not math.isfinite(gmin)
        or not math.isfinite(gmax)
        or gmax <= gmin
    ):

        raise ValueError(

            f"{device_profile['device_id']} does not have "
            f"a valid conductance range."

        )


    conductance_range = (
        gmax - gmin
    )


    # --------------------------------------------------------
    # Requested signed weight levels
    #
    # W=2 -> 3
    # W=4 -> 15
    # W=6 -> 63
    # W=8 -> 255
    # --------------------------------------------------------

    requested_levels = (
        (2 ** requested_bits) - 1
    )


    if requested_levels < 3:

        raise ValueError(
            "requested_bits must be at least 2."
        )


    # --------------------------------------------------------
    # Device behavior
    # --------------------------------------------------------

    conductance_mode = str(

        device_profile.get(
            "conductance_mode",
            "UNKNOWN"
        )

    ).upper()


    states = device_profile.get(
        "states",
        float("nan")
    )


    # --------------------------------------------------------
    # FIXED PHYSICAL STATE COUNT
    #
    # If a device has N physical conductance states:
    #
    # differential signed levels = 2N - 1
    #
    # Example:
    #
    # TaOx_01:
    # N = 7
    #
    # maximum signed levels:
    #
    # 2(7) - 1 = 13
    # --------------------------------------------------------

    if valid_state_count(
        states
    ):

        physical_states = int(
            float(states)
        )

        physical_signed_levels = (
            (2 * physical_states) - 1
        )

        actual_levels = min(
            requested_levels,
            physical_signed_levels
        )


    # --------------------------------------------------------
    # ANALOG / GRADUAL DEVICE
    #
    # There is no fixed literature-reported state count.
    #
    # Therefore requested precision is treated as an
    # idealized accelerator mapping assumption.
    #
    # It must NOT be described as a measured physical
    # state count.
    # --------------------------------------------------------

    elif conductance_mode in {

        "ANALOG",
        "GRADUAL_MULTILEVEL"

    }:

        actual_levels = (
            requested_levels
        )


    # --------------------------------------------------------
    # Unsupported physical model
    # --------------------------------------------------------

    else:

        raise ValueError(

            f"{device_profile['device_id']} has conductance mode "
            f"{conductance_mode} but no supported physical "
            f"state-count model."

        )


    # --------------------------------------------------------
    # Number of positive magnitude codes
    # --------------------------------------------------------

    max_code = (
        (actual_levels - 1) // 2
    )


    if max_code < 1:

        raise ValueError(
            "Effective signed conductance levels must be >= 3."
        )


    # --------------------------------------------------------
    # Layer weight scale
    # --------------------------------------------------------

    weight_scale = (
        weights.abs().max().item()
    )


    # --------------------------------------------------------
    # Zero-weight layer protection
    # --------------------------------------------------------

    if weight_scale == 0:

        g_plus = torch.full_like(
            weights,
            gmin
        )

        g_minus = torch.full_like(
            weights,
            gmin
        )

        return (
            g_plus,
            g_minus,
            0.0,
            actual_levels
        )


    # --------------------------------------------------------
    # Normalize trained weights into [-1, +1]
    # --------------------------------------------------------

    normalized_weight = (

        weights
        / weight_scale

    ).clamp(
        -1.0,
        1.0
    )


    # --------------------------------------------------------
    # Quantize neural-network weight
    # --------------------------------------------------------

    quantized_normalized = (

        torch.round(

            normalized_weight
            * max_code

        )

        / max_code

    )


    # --------------------------------------------------------
    # Differential representation
    #
    # Positive weight:
    #
    # G+ > Gmin
    # G- = Gmin
    #
    # Negative weight:
    #
    # G+ = Gmin
    # G- > Gmin
    # --------------------------------------------------------

    positive_component = torch.clamp(
        quantized_normalized,
        min=0.0
    )


    negative_component = torch.clamp(
        -quantized_normalized,
        min=0.0
    )


    g_plus = (

        gmin

        + positive_component
        * conductance_range

    )


    g_minus = (

        gmin

        + negative_component
        * conductance_range

    )


    return (
        g_plus,
        g_minus,
        weight_scale,
        actual_levels
    )


# ============================================================
# ADC quantization
# ============================================================

def adc_quantize_current(
    current,
    bits,
    full_scale
):

    if bits < 1:

        raise ValueError(
            "ADC bits must be >= 1."
        )


    adc_levels = (
        (2 ** bits) - 1
    )


    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    epsilon = torch.finfo(
        current.dtype
    ).eps


    safe_full_scale = torch.clamp(
        full_scale,
        min=epsilon
    )


    # --------------------------------------------------------
    # Normalize physical current to ADC range
    # --------------------------------------------------------

    normalized = (

        current
        / safe_full_scale

    ).clamp(
        0.0,
        1.0
    )


    # --------------------------------------------------------
    # Quantize
    # --------------------------------------------------------

    quantized_normalized = (

        torch.round(

            normalized
            * adc_levels

        )

        / adc_levels

    )


    # --------------------------------------------------------
    # Convert back into current units
    # --------------------------------------------------------

    quantized_current = (

        quantized_normalized
        * full_scale

    )


    return quantized_current


# ============================================================
# Crossbar linear layer
# ============================================================

def crossbar_linear(
    x,
    weight,
    bias,
    crossbar_size,
    weight_bits,
    adc_bits,
    device_profile
):

    # --------------------------------------------------------
    # Device parameters
    # --------------------------------------------------------

    gmin = float(
        device_profile["gmin"]
    )

    gmax = float(
        device_profile["gmax"]
    )


    if (
        not math.isfinite(gmin)
        or not math.isfinite(gmax)
        or gmax <= gmin
    ):

        raise ValueError(

            f"{device_profile['device_id']} does not have "
            f"enough conductance-range information "
            f"for the current simulation."

        )


    conductance_range = (
        gmax - gmin
    )


    conductance_mode = str(

        device_profile.get(
            "conductance_mode",
            "UNKNOWN"
        )

    ).upper()


    # ========================================================
    # BINARY MEMRISTOR BIT-SLICED PATH
    # ========================================================
    #
    # A binary physical memristor still has only:
    #
    #   Gmin
    #   Gmax
    #
    # It is NOT converted into a 15-state or 255-state cell.
    #
    # Multiple binary cells are combined instead.
    #
    #
    # Example: requested W=4
    #
    # signed codes:
    #
    #   -7 ... 0 ... +7
    #
    # magnitude requires:
    #
    #   bit0 = 1
    #   bit1 = 2
    #   bit2 = 4
    #
    # Therefore:
    #
    #   3 binary slices on positive branch
    #   3 binary slices on negative branch
    #
    #   = 6 physical cells / NN weight
    #
    # Effective accelerator levels = 15
    #
    # Physical states of ONE device = still 2
    # ========================================================

    if conductance_mode == "DISCRETE_BINARY":

        if weight_bits < 2:

            raise ValueError(

                "Binary bit-sliced mapping requires "
                "weight_bits >= 2."

            )


        # ----------------------------------------------------
        # Requested signed weight levels
        #
        # 2 bits -> 3
        # 4 bits -> 15
        # 6 bits -> 63
        # 8 bits -> 255
        # ----------------------------------------------------

        actual_levels = (
            (2 ** weight_bits) - 1
        )


        # ----------------------------------------------------
        # Maximum integer magnitude
        #
        # 2 bits -> 1
        # 4 bits -> 7
        # 6 bits -> 31
        # 8 bits -> 127
        # ----------------------------------------------------

        max_magnitude_code = (
            (actual_levels - 1) // 2
        )


        # ----------------------------------------------------
        # Number of physical binary slices per branch
        #
        # 2 bits -> 1
        # 4 bits -> 3
        # 6 bits -> 5
        # 8 bits -> 7
        # ----------------------------------------------------

        slices_per_branch = (
            weight_bits - 1
        )


        # ----------------------------------------------------
        # Original layer weight scale
        # ----------------------------------------------------

        weight_scale = (
            weight.abs().max().item()
        )


        # ----------------------------------------------------
        # Zero-weight layer protection
        # ----------------------------------------------------

        if weight_scale == 0:

            output = torch.zeros(

                x.size(0),
                weight.size(0),

                device=x.device,
                dtype=x.dtype

            )


            if bias is not None:

                output += bias


            return (
                output,
                actual_levels
            )


        # ----------------------------------------------------
        # Normalize original trained weights
        # ----------------------------------------------------

        normalized_weight = (

            weight
            / weight_scale

        ).clamp(
            -1.0,
            1.0
        )


        # ----------------------------------------------------
        # Quantize into integer codes
        #
        # Example W=4:
        #
        # -7 ... 0 ... +7
        # ----------------------------------------------------

        signed_code = torch.round(

            normalized_weight
            * max_magnitude_code

        ).clamp(

            -max_magnitude_code,
            max_magnitude_code

        ).to(
            torch.int64
        )


        # ----------------------------------------------------
        # Separate differential branches
        # ----------------------------------------------------

        positive_code = torch.clamp(
            signed_code,
            min=0
        )


        negative_code = torch.clamp(
            -signed_code,
            min=0
        )


        # ----------------------------------------------------
        # Digital accumulator
        # ----------------------------------------------------

        output_integer = torch.zeros(

            x.size(0),
            weight.size(0),

            device=x.device,
            dtype=x.dtype

        )


        input_features = (
            weight.size(1)
        )


        # ====================================================
        # Crossbar tiling
        # ====================================================

        for start in range(

            0,
            input_features,
            crossbar_size

        ):

            end = min(

                start + crossbar_size,
                input_features

            )


            # ------------------------------------------------
            # Inputs are non-negative
            #
            # Layer 1: MNIST pixels
            # Layer 2: ReLU output
            # ------------------------------------------------

            x_tile = torch.clamp(

                x[:, start:end],

                min=0

            )


            positive_tile = (
                positive_code[:, start:end]
            )


            negative_tile = (
                negative_code[:, start:end]
            )


            # ------------------------------------------------
            # Full-scale branch current for ONE slice
            #
            # sum(V) * Gmax
            # ------------------------------------------------

            full_scale = (

                x_tile.sum(
                    dim=1,
                    keepdim=True
                )

                * gmax

            )


            # =================================================
            # Process every binary slice independently
            # =================================================

            for bit_index in range(
                slices_per_branch
            ):

                # ---------------------------------------------
                # Digital significance
                # ---------------------------------------------

                bit_significance = (
                    2 ** bit_index
                )


                # ---------------------------------------------
                # Extract positive branch bit
                # ---------------------------------------------

                positive_bit = (

                    (
                        positive_tile
                        >> bit_index
                    )

                    & 1

                ).to(
                    x.dtype
                )


                # ---------------------------------------------
                # Extract negative branch bit
                # ---------------------------------------------

                negative_bit = (

                    (
                        negative_tile
                        >> bit_index
                    )

                    & 1

                ).to(
                    x.dtype
                )


                # ---------------------------------------------
                # Program physical binary cells
                #
                # 0 -> Gmin
                # 1 -> Gmax
                # ---------------------------------------------

                g_plus_slice = (

                    gmin

                    + positive_bit
                    * conductance_range

                )


                g_minus_slice = (

                    gmin

                    + negative_bit
                    * conductance_range

                )


                # ---------------------------------------------
                # Physical current-domain MVM
                # ---------------------------------------------

                i_plus = torch.matmul(

                    x_tile,
                    g_plus_slice.T

                )


                i_minus = torch.matmul(

                    x_tile,
                    g_minus_slice.T

                )


                # ---------------------------------------------
                # ADC occurs for each physical slice
                # BEFORE differential subtraction
                # ---------------------------------------------

                i_plus_q = adc_quantize_current(

                    i_plus,
                    adc_bits,
                    full_scale

                )


                i_minus_q = adc_quantize_current(

                    i_minus,
                    adc_bits,
                    full_scale

                )


                # ---------------------------------------------
                # Differential current
                # ---------------------------------------------

                differential_current = (

                    i_plus_q
                    - i_minus_q

                )


                # ---------------------------------------------
                # Convert physical conductance response
                # back into binary integer-domain sum
                # ---------------------------------------------

                slice_integer_sum = (

                    differential_current
                    / conductance_range

                )


                # ---------------------------------------------
                # Digital shift-and-add
                # ---------------------------------------------

                output_integer += (

                    slice_integer_sum
                    * bit_significance

                )


        # ----------------------------------------------------
        # Return to neural-network weight scale
        # ----------------------------------------------------

        output = (

            output_integer

            * (
                weight_scale
                / max_magnitude_code
            )

        )


        # ----------------------------------------------------
        # Bias remains digital
        # ----------------------------------------------------

        if bias is not None:

            output += bias


        return (
            output,
            actual_levels
        )


    # ========================================================
    # ANALOG / MULTILEVEL PATH
    # ========================================================

    (
        g_plus,
        g_minus,
        weight_scale,
        actual_levels

    ) = map_weights_to_conductance(

        weight,
        weight_bits,
        device_profile

    )


    # --------------------------------------------------------
    # Output tensor
    # --------------------------------------------------------

    output = torch.zeros(

        x.size(0),
        weight.size(0),

        device=x.device,
        dtype=x.dtype

    )


    input_features = (
        weight.size(1)
    )


    # ========================================================
    # Crossbar tiling
    # ========================================================

    for start in range(

        0,
        input_features,
        crossbar_size

    ):

        end = min(

            start + crossbar_size,
            input_features

        )


        # ----------------------------------------------------
        # Inputs are non-negative
        # ----------------------------------------------------

        x_tile = torch.clamp(

            x[:, start:end],

            min=0

        )


        gp_tile = (
            g_plus[:, start:end]
        )


        gm_tile = (
            g_minus[:, start:end]
        )


        # ----------------------------------------------------
        # Physical MVM
        # ----------------------------------------------------

        i_plus = torch.matmul(

            x_tile,
            gp_tile.T

        )


        i_minus = torch.matmul(

            x_tile,
            gm_tile.T

        )


        # ----------------------------------------------------
        # ADC full scale
        # ----------------------------------------------------

        full_scale = (

            x_tile.sum(
                dim=1,
                keepdim=True
            )

            * gmax

        )


        # ----------------------------------------------------
        # ADC before differential subtraction
        # ----------------------------------------------------

        i_plus_q = adc_quantize_current(

            i_plus,
            adc_bits,
            full_scale

        )


        i_minus_q = adc_quantize_current(

            i_minus,
            adc_bits,
            full_scale

        )


        # ----------------------------------------------------
        # Differential current
        # ----------------------------------------------------

        differential_current = (

            i_plus_q
            - i_minus_q

        )


        # ----------------------------------------------------
        # Convert physical result back into
        # neural-network partial sum
        # ----------------------------------------------------

        partial_sum = (

            differential_current

            / conductance_range

            * weight_scale

        )


        output += partial_sum


    # --------------------------------------------------------
    # Bias remains digital
    # --------------------------------------------------------

    if bias is not None:

        output += bias


    return (
        output,
        actual_levels
    )


# ============================================================
# Hardware forward pass
# ============================================================

def hardware_forward(
    model,
    images,
    crossbar_size,
    weight_bits,
    adc_bits,
    device_profile
):

    # --------------------------------------------------------
    # Flatten MNIST image
    # --------------------------------------------------------

    images = images.view(

        images.size(0),
        -1

    )


    # --------------------------------------------------------
    # Network layers
    # --------------------------------------------------------

    layer1 = (
        model.network[1]
    )


    layer2 = (
        model.network[3]
    )


    # ========================================================
    # Layer 1
    # ========================================================

    x, levels1 = crossbar_linear(

        images,

        layer1.weight,
        layer1.bias,

        crossbar_size,
        weight_bits,
        adc_bits,

        device_profile

    )


    # --------------------------------------------------------
    # ReLU remains digital
    # --------------------------------------------------------

    x = torch.relu(
        x
    )


    # ========================================================
    # Layer 2
    # ========================================================

    x, levels2 = crossbar_linear(

        x,

        layer2.weight,
        layer2.bias,

        crossbar_size,
        weight_bits,
        adc_bits,

        device_profile

    )


    return (

        x,

        min(
            levels1,
            levels2
        )

    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Memristor crossbar simulation for MNIST MLP."
        )

    )


    parser.add_argument(

        "--device",

        type=str,

        required=True

    )


    parser.add_argument(

        "--crossbar",

        type=int,

        default=32

    )


    parser.add_argument(

        "--weight_bits",

        type=int,

        default=4

    )


    parser.add_argument(

        "--adc_bits",

        type=int,

        default=6

    )


    parser.add_argument(

        "--batch_size",

        type=int,

        default=256

    )


    args = parser.parse_args()


    # ========================================================
    # Load device profile
    # ========================================================

    device_profile = load_device(
        args.device
    )


    # --------------------------------------------------------
    # Device information
    # --------------------------------------------------------

    print()

    print(
        "Device:"
    )

    print(
        "-----------------------------"
    )


    print(
        f"ID: {device_profile['device_id']}"
    )


    print(
        f"Family: {device_profile['family']}"
    )


    print(
        f"Stack: {device_profile['stack']}"
    )


    print(
        f"RON: {device_profile['ron']}"
    )


    print(
        f"ROFF: {device_profile['roff']}"
    )


    print(
        f"Gmin: {device_profile['gmin']}"
    )


    print(
        f"Gmax: {device_profile['gmax']}"
    )


    print(
        f"ON/OFF ratio: {device_profile['ratio']}"
    )


    print(
        f"Reported conductance states: "
        f"{device_profile['states']}"
    )


    print(
        f"Conductance mode: "
        f"{device_profile['conductance_mode']}"
    )


    print(
        f"State count status: "
        f"{device_profile['state_count_status']}"
    )


    print(
        f"Parameter source: "
        f"{device_profile['parameter_source']}"
    )


    # ========================================================
    # Conductance-range validation
    # ========================================================

    if (
        device_profile["parameter_source"]
        == "insufficient"
    ):

        raise ValueError(

            f"{device_profile['device_id']} does not have "
            f"enough conductance-range information "
            f"for the current simulation."

        )


    conductance_mode = str(

        device_profile[
            "conductance_mode"
        ]

    ).upper()


    # ========================================================
    # Modeling mode information
    # ========================================================

    if conductance_mode == "DISCRETE_BINARY":

        signed_levels = (
            (2 ** args.weight_bits) - 1
        )

        slices_per_branch = (
            args.weight_bits - 1
        )

        physical_cells_per_weight = (
            2 * slices_per_branch
        )


        print()

        print(
            "Binary bit-sliced mapping:"
        )


        print(
            f"Requested weight bits: "
            f"{args.weight_bits}"
        )


        print(
            f"Effective signed weight levels: "
            f"{signed_levels}"
        )


        print(
            f"Binary slices per branch: "
            f"{slices_per_branch}"
        )


        print(
            f"Physical memristor cells per weight: "
            f"{physical_cells_per_weight}"
        )


        print(
            "NOTE: these effective levels are created "
            "by combining multiple binary cells."
        )


        print(
            "They are NOT physical states of one memristor."
        )


    elif conductance_mode in {

        "ANALOG",
        "GRADUAL_MULTILEVEL"

    } and not valid_state_count(

        device_profile["states"]

    ):

        print()

        print(
            "Mapping assumption:"
        )


        print(

            f"{conductance_mode} device has no fixed "
            f"literature-reported conductance-state count."

        )


        print(

            "Requested digital weight levels are treated "
            "as an idealized accelerator mapping assumption."

        )


    elif (
        conductance_mode
        == "CONTINUOUS_QUANTIZED"
    ):

        raise ValueError(

            f"{device_profile['device_id']} uses "
            f"CONTINUOUS_QUANTIZED behavior and requires "
            f"a dedicated instability/noise model."

        )


    elif (
        conductance_mode
        == "UNKNOWN"
    ):

        raise ValueError(

            f"{device_profile['device_id']} has UNKNOWN "
            f"conductance behavior and cannot yet be "
            f"simulated defensibly."

        )


    # ========================================================
    # CPU / GPU
    # ========================================================

    compute_device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print()

    print(
        f"Using: {compute_device}"
    )


    # ========================================================
    # Load trained baseline
    # ========================================================

    model = MLP().to(
        compute_device
    )


    state_dict = torch.load(

        "results/mnist_mlp_baseline.pt",

        map_location=compute_device

    )


    model.load_state_dict(
        state_dict
    )


    model.eval()


    # ========================================================
    # MNIST test set
    # ========================================================

    test_dataset = datasets.MNIST(

        root="data",

        train=False,

        download=False,

        transform=transforms.ToTensor()

    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=args.batch_size,

        shuffle=False

    )


    # ========================================================
    # Hardware simulation
    # ========================================================

    correct = 0

    total = 0

    effective_levels = None


    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                compute_device
            )

            labels = labels.to(
                compute_device
            )


            (
                outputs,
                effective_levels

            ) = hardware_forward(

                model,

                images,

                args.crossbar,

                args.weight_bits,

                args.adc_bits,

                device_profile

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


    # ========================================================
    # Results
    # ========================================================

    print()

    print(
        "Simulation:"
    )


    print(
        "-----------------------------"
    )


    print(
        f"Crossbar size: "
        f"{args.crossbar}x{args.crossbar}"
    )


    print(
        f"Requested weight bits: "
        f"{args.weight_bits}"
    )


    print(
        f"Effective signed levels: "
        f"{effective_levels}"
    )


    print(
        f"ADC bits: "
        f"{args.adc_bits}"
    )


    print(
        f"Hardware-sim accuracy: "
        f"{accuracy:.2f}%"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()