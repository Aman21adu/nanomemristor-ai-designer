from pathlib import Path

import math

import pandas as pd

from config_space import generate_configs


# ============================================================
# Files
# ============================================================

DEVICE_PROFILE_FILE = Path(
    "data/device_profiles.csv"
)

SWEEP_DIRECTORY = Path(
    "results/tables"
)

OUTPUT_FILE = SWEEP_DIRECTORY / (
    "all_device_config_results.csv"
)


# ============================================================
# Validated simulation-ready device profiles
#
# IMPORTANT:
#
# These are seven simulator-ready DEVICE PROFILES.
#
# TiOx_02_Au, TiOx_02_Ni, and TiOx_02_Pt are distinct
# electrode/device variants from the SAME experimental study.
# Therefore:
#
#   device count != independent-study count
#
# The study relationship is carried forward through study_id.
#
# HfOx_01 uses the 2024 variability paper as its PRIMARY
# study_id. Its GRADUAL_MULTILEVEL behavior has additional
# same-stack literature support recorded at property level in
# source_traceability.csv; that supporting paper is not treated
# as a second physical profile or a second primary study here.
#
#
# HfZrOx_01 is an independent 2021 primary study with
# reported post-wake-up ON/OFF = 2.0 and ANALOG behavior.
# Absolute RON/ROFF are unavailable, so the simulator uses
# normalized conductance derived from the reported ratio.
#
# TaOx_02 remains intentionally excluded because its
# CONTINUOUS_QUANTIZED behavior needs a dedicated
# instability/noise model.
# ============================================================

VALIDATED_DEVICES = [

    "ZnO_01",
    "TaOx_01",

    "HfOx_01",
    "HfOx_02",

    "HfZrOx_01",

    "TiOx_03",

    "TiOx_02_Au",
    "TiOx_02_Ni",
    "TiOx_02_Pt",

]


# ============================================================
# Expected accelerator configuration space
#
# The authoritative configuration grid comes from:
#
#     src/config_space.py
#
# No configuration count is hard-coded here.
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
# Required current sweep schema
# ============================================================

REQUIRED_SWEEP_COLUMNS = [

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
# Required device-profile metadata
# ============================================================

REQUIRED_PROFILE_COLUMNS = [

    "device_id",
    "study_id",
    "technology_family",

]


# ============================================================
# Current neural-network architecture
#
# MNIST MLP:
#
#     784 -> 128 -> 10
#
# Only Linear-layer weights are mapped to memristor crossbars.
# Bias remains digital.
# ============================================================

NETWORK_LAYERS = [

    # input_features, output_features
    (784, 128),

    (128, 10),

]


TOTAL_NETWORK_WEIGHTS = sum(

    input_features
    * output_features

    for input_features, output_features
    in NETWORK_LAYERS

)


# ============================================================
# Helpers
# ============================================================

def clean_text(
    value
):

    if pd.isna(
        value
    ):

        return ""


    return str(
        value
    ).strip()


def calculate_base_crossbar_tiles(
    crossbar_size
):

    """
    Number of square crossbar tiles required for the two
    Linear layers for ONE physical conductance plane.

    Differential branches / bit-sliced planes are added later
    through physical_cells_per_weight.
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
# Load device-profile metadata
# ============================================================

if not DEVICE_PROFILE_FILE.exists():

    raise FileNotFoundError(

        f"Missing device profile file: "
        f"{DEVICE_PROFILE_FILE}"

    )


profiles = pd.read_csv(
    DEVICE_PROFILE_FILE
)


missing_profile_columns = [

    column

    for column
    in REQUIRED_PROFILE_COLUMNS

    if column
    not in profiles.columns

]


if missing_profile_columns:

    raise ValueError(

        "device_profiles.csv is missing columns:\n"
        f"{missing_profile_columns}"

    )


validated_profiles = profiles[

    profiles[
        "device_id"
    ].isin(
        VALIDATED_DEVICES
    )

].copy()


profile_counts = (

    validated_profiles[
        "device_id"
    ]
    .value_counts()

)


missing_profile_devices = [

    device_id

    for device_id
    in VALIDATED_DEVICES

    if int(
        profile_counts.get(
            device_id,
            0
        )
    ) != 1

]


if missing_profile_devices:

    raise ValueError(

        "Every validated device must have exactly one "
        "device-profile row.\n"
        "Problem devices: "
        f"{missing_profile_devices}"

    )


validated_profiles[
    "study_id"
] = (

    validated_profiles[
        "study_id"
    ]
    .apply(
        clean_text
    )

)


missing_study_ids = (

    validated_profiles[
        "study_id"
    ]
    ==
    ""

)


if missing_study_ids.any():

    bad_devices = (

        validated_profiles.loc[

            missing_study_ids,

            "device_id",

        ]
        .tolist()

    )


    raise ValueError(

        "Validated devices must have non-empty study_id "
        "values.\n"
        f"Missing study_id for: {bad_devices}"

    )


profile_metadata = (

    validated_profiles[

        [
            "device_id",
            "study_id",
            "technology_family",
        ]

    ]

    .copy()

)


study_by_device = dict(

    zip(

        profile_metadata[
            "device_id"
        ],

        profile_metadata[
            "study_id"
        ],

    )

)


family_by_device = dict(

    zip(

        profile_metadata[
            "device_id"
        ],

        profile_metadata[
            "technology_family"
        ],

    )

)


# ============================================================
# Build sweep-file list
# ============================================================

files = [

    SWEEP_DIRECTORY / (
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
        in REQUIRED_SWEEP_COLUMNS

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
    # Complete configuration count
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
    # One file -> one device
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


    file_device = str(
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
    # Family in sweep must agree with profile metadata
    # --------------------------------------------------------

    unique_families = (

        df[
            "technology_family"
        ]
        .dropna()
        .astype(str)
        .unique()

    )


    if len(
        unique_families
    ) != 1:

        raise ValueError(

            f"{path} must contain exactly one "
            "technology_family."

        )


    expected_family = str(
        family_by_device[
            file_device
        ]
    )


    if (
        str(
            unique_families[0]
        )
        !=
        expected_family
    ):

        raise ValueError(

            f"Technology-family mismatch for "
            f"{file_device}.\n"
            f"Profile: {expected_family}\n"
            f"Sweep:   {unique_families[0]}"

        )


    # --------------------------------------------------------
    # Accelerator configurations must be unique
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
    # Exact configuration grid
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

            message_lines.append(

                "Missing configurations "
                "(crossbar, weight_bits, adc_bits): "
                f"{sorted(missing_configs)[:10]}"

            )


        if unexpected_configs:

            message_lines.append(

                "Unexpected configurations "
                "(crossbar, weight_bits, adc_bits): "
                f"{sorted(unexpected_configs)[:10]}"

            )


        raise ValueError(

            "\n".join(
                message_lines
            )

        )


    # --------------------------------------------------------
    # study_id is provenance/grouping metadata.
    #
    # It is attached here from the evidence-aware device
    # profile rather than duplicated manually in each sweep.
    # --------------------------------------------------------

    if (
        "study_id"
        in df.columns
    ):

        existing_studies = (

            df[
                "study_id"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()

        )


        if (

            len(
                existing_studies
            ) > 0

            and

            (
                len(
                    existing_studies
                ) != 1

                or

                existing_studies[0]
                !=
                study_by_device[
                    file_device
                ]
            )

        ):

            raise ValueError(

                f"Conflicting study_id values in {path}."

            )


        df[
            "study_id"
        ] = study_by_device[
            file_device
        ]


    else:

        df.insert(

            1,

            "study_id",

            study_by_device[
                file_device
            ],

        )


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
            "column order as the other sweep files."

        )


# ============================================================
# Combine validated device sweeps
# ============================================================

combined = pd.concat(

    dataframes,

    ignore_index=True

)


# ============================================================
# Validate combined structure
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


combined_devices = set(

    combined[
        "device_id"
    ].unique()

)


if (
    combined_devices
    !=
    set(
        VALIDATED_DEVICES
    )
):

    raise ValueError(

        "Combined device set does not match "
        "VALIDATED_DEVICES."

    )


# ============================================================
# Architecture-level quantities
# ============================================================

combined[
    "network_weight_count"
] = TOTAL_NETWORK_WEIGHTS


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


combined[
    "base_crossbar_tiles"
] = (

    combined[
        "crossbar_size"
    ]

    .apply(
        calculate_base_crossbar_tiles
    )

)


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
#
# NOT measured power / energy / area / latency.
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

SWEEP_DIRECTORY.mkdir(

    parents=True,

    exist_ok=True,

)


combined.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# Summary
# ============================================================

device_count = int(

    combined[
        "device_id"
    ].nunique()

)


study_count = int(

    combined[
        "study_id"
    ].nunique()

)


family_count = int(

    combined[
        "technology_family"
    ].nunique()

)


print()

print(
    "VALIDATED COMBINED DATASET"
)

print(
    "========================================"
)


print(
    "Device profiles:",
    device_count
)


print(
    "Distinct source studies:",
    study_count
)


print(
    "Technology families:",
    family_count
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
# Study structure
# ============================================================

print()

print(
    "STUDY GROUPS"
)

print(
    "----------------------------------------"
)


study_structure = (

    combined[

        [
            "device_id",
            "study_id",
            "technology_family",
        ]

    ]

    .drop_duplicates()

    .sort_values(

        [
            "study_id",
            "device_id",
        ]

    )

)


print(

    study_structure.to_string(
        index=False
    )

)


print()

print(

    "NOTE: TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt "
    "are distinct device profiles from one study; "
    "they are not counted as three independent studies."

)


# ============================================================
# Mapping strategies
# ============================================================

print()

print(
    "MAPPING STRATEGIES"
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
# This is maximum simulated accuracy only.
# It is NOT yet a measured-hardware optimum.
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
        ],

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
    "(not yet cost-aware measured-hardware optimum)"
)


print(

    best_accuracy[

        [
            "device_id",
            "study_id",
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


print()

print(
    "Saved to:",
    OUTPUT_FILE
)
