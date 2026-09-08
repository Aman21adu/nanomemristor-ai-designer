from pathlib import Path

import math

import pandas as pd

from config_space import generate_configs


# ============================================================
# Validated simulation-ready devices
#
# IMPORTANT:
#
# Only devices that currently have:
#
# 1. usable conductance-range information
# 2. a supported simulator behavior model
# 3. regenerated results using the current schema
#
# are included here.
#
# TaOx_02 is intentionally excluded because its
# CONTINUOUS_QUANTIZED behavior requires a dedicated
# instability/noise model.
# ============================================================

VALIDATED_DEVICES = [

    "ZnO_01",
    "TaOx_01",
    "HfOx_02",
    "TiOx_03",

]


# ============================================================
# Expected accelerator configuration space
#
# IMPORTANT:
#
# Do NOT hard-code 64, 245, or any future configuration count.
#
# The authoritative configuration grid comes from:
#
#     src/config_space.py
#
# ============================================================

EXPECTED_CONFIGS = generate_configs()


EXPECTED_CONFIGS_PER_DEVICE = len(
    EXPECTED_CONFIGS
)


EXPECTED_CONFIG_KEYS = {

    (
        int(
            config[
                "crossbar_size"
            ]
        ),

        int(
            config[
                "weight_bits"
            ]
        ),

        int(
            config[
                "adc_bits"
            ]
        ),
    )

    for config
    in EXPECTED_CONFIGS

}


if (
    len(
        EXPECTED_CONFIG_KEYS
    )
    !=
    EXPECTED_CONFIGS_PER_DEVICE
):

    raise ValueError(

        "config_space.py generated duplicate "
        "accelerator configurations."

    )


EXPECTED_TOTAL_ROWS = (

    EXPECTED_CONFIGS_PER_DEVICE
    *
    len(
        VALIDATED_DEVICES
    )

)


# ============================================================
# Expected current sweep schema
# ============================================================

REQUIRED_COLUMNS = [

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


# ============================================================
# Current neural-network architecture
#
# MNIST MLP:
#
# 784 -> 128 -> 10
#
# Only the two Linear layers contain memristor-mapped weights.
# Bias remains digital.
# ============================================================

NETWORK_LAYERS = [

    # input_features, output_features
    (784, 128),
    (128, 10),

]


# ------------------------------------------------------------
# Total neural-network weights stored in crossbars
# ------------------------------------------------------------

TOTAL_NETWORK_WEIGHTS = sum(

    input_features
    * output_features

    for input_features, output_features
    in NETWORK_LAYERS

)


# ============================================================
# Crossbar tile calculation
# ============================================================

def calculate_base_crossbar_tiles(
    crossbar_size
):

    """

    Calculate how many square crossbar tiles are required
    for the two neural-network Linear layers for ONE
    physical conductance plane.

    Example:

    For each layer:

        row tiles
        =
        ceil(input_features / crossbar_size)

        column tiles
        =
        ceil(output_features / crossbar_size)

        tile count
        =
        row tiles * column tiles

    The result does NOT yet include differential branches
    or binary bit slices.

    """

    total_tiles = 0


    for (
        input_features,
        output_features

    ) in NETWORK_LAYERS:

        input_tiles = math.ceil(

            input_features
            /
            crossbar_size

        )


        output_tiles = math.ceil(

            output_features
            /
            crossbar_size

        )


        total_tiles += (

            input_tiles
            *
            output_tiles

        )


    return total_tiles


# ============================================================
# Build sweep-file list
# ============================================================

files = [

    Path(
        "results/tables/"
        f"{device_id}_config_sweep.csv"
    )

    for device_id
    in VALIDATED_DEVICES

]


# ============================================================
# Load and validate every device sweep
# ============================================================

dataframes = []


for path in files:

    # --------------------------------------------------------
    # File must exist
    # --------------------------------------------------------

    if not path.exists():

        raise FileNotFoundError(

            f"Missing validated sweep file: "
            f"{path}"

        )


    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        path
    )


    # --------------------------------------------------------
    # Check schema
    # --------------------------------------------------------

    missing_columns = [

        column

        for column
        in REQUIRED_COLUMNS

        if column
        not in df.columns

    ]


    if missing_columns:

        raise ValueError(

            f"{path} is using an outdated or incomplete "
            f"schema.\n"
            f"Missing columns: {missing_columns}"

        )


    # --------------------------------------------------------
    # Each current device sweep must contain the complete
    # configuration space defined in config_space.py.
    # --------------------------------------------------------

    if (
        len(
            df
        )
        !=
        EXPECTED_CONFIGS_PER_DEVICE
    ):

        raise ValueError(

            f"{path} contains {len(df)} rows. "
            f"Expected {EXPECTED_CONFIGS_PER_DEVICE} "
            f"from config_space.py."

        )


    # --------------------------------------------------------
    # Make sure one file contains only one device
    # --------------------------------------------------------

    unique_devices = (

        df[
            "device_id"
        ]
        .dropna()
        .unique()

    )


    if len(
        unique_devices
    ) != 1:

        raise ValueError(

            f"{path} contains multiple device IDs: "
            f"{list(unique_devices)}"

        )


    file_device = (
        unique_devices[0]
    )


    expected_device = (

        path.name
        .replace(
            "_config_sweep.csv",
            ""
        )

    )


    if (
        file_device
        !=
        expected_device
    ):

        raise ValueError(

            f"Device mismatch in {path}.\n"
            f"File implies: {expected_device}\n"
            f"CSV contains: {file_device}"

        )


    # --------------------------------------------------------
    # Check that accelerator configurations are unique
    # --------------------------------------------------------

    duplicate_configs = df.duplicated(

        subset=[

            "crossbar_size",
            "requested_weight_bits",
            "adc_bits",

        ]

    ).sum()


    if duplicate_configs > 0:

        raise ValueError(

            f"{path} contains "
            f"{duplicate_configs} duplicated "
            f"accelerator configurations."

        )


    # --------------------------------------------------------
    # Verify that the file contains the EXACT configuration
    # grid defined by config_space.py.
    #
    # This is stronger than checking the row count alone.
    # --------------------------------------------------------

    actual_config_keys = {

        (
            int(
                row.crossbar_size
            ),

            int(
                row.requested_weight_bits
            ),

            int(
                row.adc_bits
            ),
        )

        for row
        in df.itertuples(
            index=False
        )

    }


    missing_configs = (

        EXPECTED_CONFIG_KEYS
        -
        actual_config_keys

    )


    unexpected_configs = (

        actual_config_keys
        -
        EXPECTED_CONFIG_KEYS

    )


    if (
        missing_configs
        or
        unexpected_configs
    ):

        message_lines = [

            (
                f"{path} does not contain the exact "
                "configuration grid defined in "
                "config_space.py."
            ),

        ]


        if missing_configs:

            preview = sorted(
                missing_configs
            )[:10]


            message_lines.append(

                "Missing configurations "
                "(crossbar, weight_bits, adc_bits): "
                f"{preview}"

            )


        if unexpected_configs:

            preview = sorted(
                unexpected_configs
            )[:10]


            message_lines.append(

                "Unexpected configurations "
                "(crossbar, weight_bits, adc_bits): "
                f"{preview}"

            )


        raise ValueError(

            "\n".join(
                message_lines
            )

        )


    # --------------------------------------------------------
    # Append validated sweep
    # --------------------------------------------------------

    dataframes.append(
        df
    )


# ============================================================
# Confirm identical column order
# ============================================================

reference_columns = list(
    dataframes[0].columns
)


for (
    device_id,
    df

) in zip(

    VALIDATED_DEVICES,
    dataframes

):

    if (
        list(
            df.columns
        )
        !=
        reference_columns
    ):

        raise ValueError(

            f"{device_id} does not have the same "
            f"column order as the other sweep files."

        )


# ============================================================
# Combine validated device sweeps
# ============================================================

combined = pd.concat(

    dataframes,

    ignore_index=True

)


# ============================================================
# Validate combined size
# ============================================================

if (
    len(
        combined
    )
    !=
    EXPECTED_TOTAL_ROWS
):

    raise ValueError(

        "Unexpected combined dataset size.\n"
        f"Expected: {EXPECTED_TOTAL_ROWS}\n"
        f"Found:    {len(combined)}"

    )


# ============================================================
# Architecture-level quantities
# ============================================================

# ------------------------------------------------------------
# Total neural-network weights
#
# Same model for every configuration.
# ------------------------------------------------------------

combined[
    "network_weight_count"
] = TOTAL_NETWORK_WEIGHTS


# ------------------------------------------------------------
# Estimated number of actual memristor cells required
#
# neural-network weights
# x
# physical cells per weight
#
# Examples:
#
# single differential pair:
#
#   101632 x 2
#
# binary W=4:
#
#   101632 x 6
#
# binary W=6:
#
#   101632 x 10
#
# This excludes spare/redundant cells and peripheral circuits.
# ------------------------------------------------------------

combined[
    "estimated_memristor_cells"
] = (

    combined[
        "network_weight_count"
    ]

    *
    combined[
        "physical_cells_per_weight"
    ]

)


# ------------------------------------------------------------
# Number of square crossbar tiles needed for ONE
# conductance plane.
# ------------------------------------------------------------

combined[
    "base_crossbar_tiles"
] = combined[

    "crossbar_size"

].apply(

    calculate_base_crossbar_tiles

)


# ------------------------------------------------------------
# Estimated physical crossbar tiles
#
# Includes:
#
# differential branches
#
# and, for binary devices,
# multiple bit-sliced planes.
#
# Example:
#
# physical_cells_per_weight = 2
#
# means two physical conductance planes:
#
# G+
# G-
# ------------------------------------------------------------

combined[
    "estimated_physical_crossbar_tiles"
] = (

    combined[
        "base_crossbar_tiles"
    ]

    *
    combined[
        "physical_cells_per_weight"
    ]

)


# ------------------------------------------------------------
# ADC quantization levels
#
# 4 bit  -> 16
# 5 bit  -> 32
# 6 bit  -> 64
# 7 bit  -> 128
# 8 bit  -> 256
# 9 bit  -> 512
# 10 bit -> 1024
#
# This is NOT an ADC area or power estimate.
# ------------------------------------------------------------

combined[
    "adc_levels"
] = (

    2
    **
    combined[
        "adc_bits"
    ]

)


# ============================================================
# Heuristic architecture-cost proxy
# ============================================================
#
# IMPORTANT:
#
# This is NOT:
#
#   measured power
#   measured energy
#   measured area
#   measured latency
#
# It is only a relative design-space proxy.
#
# It increases when:
#
# - more physical crossbar planes/tiles are needed
# - ADC precision increases
#
# We keep this explicitly labeled as a proxy so it cannot
# accidentally be presented as real hardware energy or area.
# ============================================================

combined[
    "relative_hardware_cost_proxy"
] = (

    combined[
        "estimated_physical_crossbar_tiles"
    ]

    *
    combined[
        "adc_levels"
    ]

)


# ============================================================
# Save combined dataset
# ============================================================

output = Path(

    "results/tables/"
    "all_device_config_results.csv"

)


combined.to_csv(

    output,

    index=False

)


# ============================================================
# Summary
# ============================================================

print()

print(
    "VALIDATED COMBINED DATASET"
)

print(
    "========================================"
)


print(
    "Devices:",
    len(
        VALIDATED_DEVICES
    )
)


print(
    "Configurations per device:",
    EXPECTED_CONFIGS_PER_DEVICE
)


print(
    "Expected combined rows:",
    EXPECTED_TOTAL_ROWS
)


print(
    "Combined rows:",
    len(
        combined
    )
)


print(
    "Columns:",
    len(
        combined.columns
    )
)


print(
    "Network weights:",
    TOTAL_NETWORK_WEIGHTS
)


print(
    "Configuration-space source:",
    "src/config_space.py"
)


print()

print(
    "Rows per device:"
)


print(

    combined[
        "device_id"
    ]
    .value_counts()
    .sort_index()

)


# ============================================================
# Mapping strategies
# ============================================================

print()

print(
    "Mapping strategies:"
)


print(

    combined[

        [
            "device_id",
            "conductance_mode",
            "mapping_strategy",
            "precision_basis",

        ]

    ]
    .drop_duplicates()
    .sort_values(
        "device_id"
    )
    .to_string(
        index=False
    )

)


# ============================================================
# Best raw-accuracy configuration per device
#
# IMPORTANT:
#
# This is only maximum simulated accuracy.
#
# It is NOT yet the final accelerator optimum because
# hardware cost must later be included in the objective.
# ============================================================

best_accuracy = (

    combined

    .sort_values(

        [
            "accuracy",
            "relative_hardware_cost_proxy",

        ],

        ascending=[
            False,
            True,
        ]

    )

    .groupby(
        "device_id",
        as_index=False
    )

    .head(1)

)


print()

print(
    "BEST RAW ACCURACY PER DEVICE"
)

print(
    "(not yet cost-aware optimum)"
)


print(

    best_accuracy[

        [
            "device_id",
            "technology_family",
            "mapping_strategy",
            "crossbar_size",
            "requested_weight_bits",
            "effective_weight_levels",
            "physical_cells_per_weight",
            "adc_bits",
            "accuracy",
            "accuracy_loss",
            "estimated_memristor_cells",
            "estimated_physical_crossbar_tiles",
            "relative_hardware_cost_proxy",

        ]

    ].to_string(
        index=False
    )

)


# ============================================================
# Save confirmation
# ============================================================

print()

print(
    "Saved to:",
    output
)