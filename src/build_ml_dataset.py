from pathlib import Path

import json
import math

import pandas as pd

from config_space import generate_configs


# ============================================================
# Files
# ============================================================

DEVICE_FILE = (
    "data/device_profiles.csv"
)

RESULT_FILE = (
    "results/tables/"
    "all_device_config_results.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "ml_dataset.csv"
)

MANIFEST_FILE = (
    "results/tables/"
    "ml_dataset_manifest.json"
)


# ============================================================
# Expected validated devices
#
# These should match the devices currently included in
# all_device_config_results.csv.
# ============================================================

EXPECTED_DEVICES = {

    "ZnO_01",
    "TaOx_01",
    "HfOx_02",
    "TiOx_03",

}


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
# This allows future configuration-space expansion without
# manually changing this file again.
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
        EXPECTED_DEVICES
    )

)


# ============================================================
# Required device-profile columns
# ============================================================

DEVICE_REQUIRED_COLUMNS = [

    "device_id",
    "technology_family",
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "conductance_states",
    "conductance_mode",
    "state_count_status",

]


# ============================================================
# Required simulator-result columns
# ============================================================

RESULT_REQUIRED_COLUMNS = [

    "device_id",
    "technology_family",

    "conductance_mode",
    "state_count_status",

    "mapping_strategy",
    "precision_basis",
    "parameter_source",

    "crossbar_size",
    "requested_weight_bits",
    "effective_weight_levels",

    "slices_per_branch",
    "physical_cells_per_weight",

    "adc_bits",

    "accuracy",
    "accuracy_loss",

    "estimated_memristor_cells",
    "estimated_physical_crossbar_tiles",
    "adc_levels",
    "relative_hardware_cost_proxy",

]


# ============================================================
# Load data
# ============================================================

devices = pd.read_csv(
    DEVICE_FILE
)

results = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# Validate schemas
# ============================================================

missing_device_columns = [

    column

    for column
    in DEVICE_REQUIRED_COLUMNS

    if column
    not in devices.columns

]


if missing_device_columns:

    raise ValueError(

        "device_profiles.csv is missing columns:\n"
        f"{missing_device_columns}"

    )


missing_result_columns = [

    column

    for column
    in RESULT_REQUIRED_COLUMNS

    if column
    not in results.columns

]


if missing_result_columns:

    raise ValueError(

        "all_device_config_results.csv "
        "is missing columns:\n"
        f"{missing_result_columns}"

    )


# ============================================================
# Validate device set
# ============================================================

used_devices = set(

    results[
        "device_id"
    ].unique()

)


if (
    used_devices
    != EXPECTED_DEVICES
):

    raise ValueError(

        "Unexpected device set in combined results.\n"
        f"Expected: {sorted(EXPECTED_DEVICES)}\n"
        f"Found:    {sorted(used_devices)}"

    )


# ============================================================
# Every device must contribute the complete configuration
# space currently defined in config_space.py.
# ============================================================

rows_per_device = (

    results[
        "device_id"
    ]
    .value_counts()

)


for device_id in EXPECTED_DEVICES:

    count = int(

        rows_per_device.get(
            device_id,
            0
        )

    )


    if (
        count
        !=
        EXPECTED_CONFIGS_PER_DEVICE
    ):

        raise ValueError(

            f"{device_id} has {count} configurations. "
            f"Expected {EXPECTED_CONFIGS_PER_DEVICE} "
            f"from config_space.py."

        )


# ============================================================
# Ensure there are no duplicate accelerator configurations
# inside one device.
# ============================================================

duplicate_count = results.duplicated(

    subset=[

        "device_id",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",

    ]

).sum()


if duplicate_count > 0:

    raise ValueError(

        f"Found {duplicate_count} duplicate "
        f"device/configuration rows."

    )


# ============================================================
# Verify that every device contains the exact configuration
# grid defined by config_space.py.
#
# This is stronger than checking the row count alone.
#
# Example:
# 245 rows with one missing configuration and one wrong
# configuration would still have the correct row count.
# This check catches that problem.
# ============================================================

for device_id in EXPECTED_DEVICES:

    device_result_rows = results[

        results[
            "device_id"
        ]
        ==
        device_id

    ]


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
        in device_result_rows.itertuples(
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
                f"{device_id} does not contain "
                "the exact configuration grid "
                "defined in config_space.py."
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


# ============================================================
# Validate total simulator-result size
# ============================================================

if (
    len(
        results
    )
    !=
    EXPECTED_TOTAL_ROWS
):

    raise ValueError(

        "Unexpected total simulator-result row count.\n"
        f"Expected: {EXPECTED_TOTAL_ROWS}\n"
        f"Found:    {len(results)}"

    )


# ============================================================
# Keep only devices with validated simulation results
# ============================================================

devices = devices[

    devices[
        "device_id"
    ].isin(
        used_devices
    )

].copy()


# ============================================================
# Ensure exactly one profile row per device
# ============================================================

profile_counts = (

    devices[
        "device_id"
    ]
    .value_counts()

)


duplicated_profiles = (

    profile_counts[
        profile_counts != 1
    ]

)


if not duplicated_profiles.empty:

    raise ValueError(

        "Each simulated device must have exactly "
        "one device-profile row.\n"
        f"{duplicated_profiles}"

    )


# ============================================================
# Build device-level physical descriptors
# ============================================================

device_rows = []


for _, row in devices.iterrows():

    device_id = str(
        row["device_id"]
    )


    # --------------------------------------------------------
    # Resistance / ratio information
    # --------------------------------------------------------

    ron = row[
        "ron_ohm"
    ]

    roff = row[
        "roff_ohm"
    ]

    ratio = row[
        "on_off_ratio"
    ]


    # --------------------------------------------------------
    # Determine usable ON/OFF ratio
    #
    # Priority:
    #
    # 1. directly available ratio
    # 2. derive from ROFF / RON
    #
    # We do NOT invent one if neither exists.
    # --------------------------------------------------------

    if (
        pd.notna(ratio)
        and float(ratio) > 0
    ):

        usable_ratio = float(
            ratio
        )


    elif (

        pd.notna(ron)

        and pd.notna(roff)

        and float(ron) > 0

        and float(roff) > 0

    ):

        usable_ratio = (

            float(roff)
            /
            float(ron)

        )


    else:

        raise ValueError(

            f"{device_id} has no usable "
            f"ON/OFF ratio."

        )


    # --------------------------------------------------------
    # Physical conductance-state information
    #
    # IMPORTANT:
    #
    # Missing fixed state count is NOT replaced by an
    # invented physical state count.
    #
    # Instead:
    #
    # state_count_available = 1
    # physical_state_count  = actual count
    #
    # or:
    #
    # state_count_available = 0
    # physical_state_count  = 0
    #
    # The zero is a machine-readable placeholder only.
    # The availability flag tells the model what it means.
    # --------------------------------------------------------

    states = row[
        "conductance_states"
    ]


    if (

        pd.notna(states)

        and float(states) >= 2

    ):

        state_count_available = 1

        physical_state_count = int(
            float(states)
        )


    else:

        state_count_available = 0

        physical_state_count = 0


    # --------------------------------------------------------
    # Device behavior provenance
    # --------------------------------------------------------

    conductance_mode = str(

        row[
            "conductance_mode"
        ]

    ).upper()


    state_count_status = str(

        row[
            "state_count_status"
        ]

    ).upper()


    # --------------------------------------------------------
    # Save one device descriptor row
    # --------------------------------------------------------

    device_rows.append({

        "device_id":
            device_id,

        "technology_family":
            row[
                "technology_family"
            ],

        "device_on_off_ratio":
            usable_ratio,

        "device_log10_on_off_ratio":
            math.log10(
                usable_ratio
            ),

        "state_count_available":
            state_count_available,

        "physical_state_count":
            physical_state_count,

        "device_conductance_mode":
            conductance_mode,

        "device_state_count_status":
            state_count_status,

    })


device_features = pd.DataFrame(
    device_rows
)


# ============================================================
# Merge device-level X with architecture configuration Y
# and simulated performance S.
# ============================================================

ml = results.merge(

    device_features,

    on=[

        "device_id",
        "technology_family",

    ],

    how="inner",

    validate="many_to_one",

)


# ============================================================
# Check merged row count
# ============================================================

if len(ml) != len(results):

    raise ValueError(

        "ML merge changed the number of simulation rows.\n"
        f"Results rows: {len(results)}\n"
        f"ML rows:      {len(ml)}"

    )


# ============================================================
# Semantic cleanup of bit-slicing information
#
# For non-bit-sliced architectures:
#
# slices_per_branch = 0
#
# means:
#
# "not applicable"
#
# not:
#
# "missing experimental data".
# ============================================================

ml[
    "slices_per_branch"
] = (

    ml[
        "slices_per_branch"
    ]
    .fillna(0)
    .astype(int)

)


# ============================================================
# Dataset columns
# ============================================================

columns = [

    # ========================================================
    # GROUPING / PROVENANCE
    #
    # These are NOT normal model features.
    #
    # device_id is the holdout unit.
    # ========================================================

    "device_id",
    "technology_family",


    # ========================================================
    # DEVICE FEATURES X
    # ========================================================

    "device_on_off_ratio",
    "device_log10_on_off_ratio",

    "state_count_available",
    "physical_state_count",

    "device_conductance_mode",
    "device_state_count_status",

    "parameter_source",


    # ========================================================
    # ARCHITECTURE / CONFIGURATION FEATURES Y
    # ========================================================

    "mapping_strategy",
    "precision_basis",

    "crossbar_size",

    "requested_weight_bits",

    "effective_weight_levels",

    "slices_per_branch",

    "physical_cells_per_weight",

    "adc_bits",


    # ========================================================
    # ARCHITECTURE COST DESCRIPTORS
    #
    # These are derived design quantities.
    #
    # They are NOT measured area / power / energy.
    # ========================================================

    "estimated_memristor_cells",

    "estimated_physical_crossbar_tiles",

    "adc_levels",

    "relative_hardware_cost_proxy",


    # ========================================================
    # TARGET PERFORMANCE S
    # ========================================================

    "accuracy",
    "accuracy_loss",

]


ml = ml[
    columns
].copy()


# ============================================================
# Validate that required ML fields contain no NaN
# ============================================================

missing_values = (

    ml
    .isna()
    .sum()

)


missing_values = missing_values[
    missing_values > 0
]


if not missing_values.empty:

    raise ValueError(

        "Unexpected missing values remain "
        "in the ML dataset:\n"
        f"{missing_values}"

    )


# ============================================================
# Important leakage checks
# ============================================================

# ------------------------------------------------------------
# device_id appears many times intentionally.
#
# Therefore random row-level train/test splitting would leak
# information from the same physical device into both sets.
# ------------------------------------------------------------

if (
    ml[
        "device_id"
    ].nunique()
    >=
    len(
        ml
    )
):

    raise ValueError(

        "Unexpected dataset structure: "
        "device grouping was lost."

    )


# ============================================================
# Explicit feature roles
#
# This makes later scripts less likely to accidentally use
# device_id as an AI input.
# ============================================================

GROUP_COLUMNS = [

    "device_id",

]


PROVENANCE_COLUMNS = [

    "technology_family",

]


DEVICE_MODEL_FEATURES = [

    "device_on_off_ratio",

    "device_log10_on_off_ratio",

    "state_count_available",

    "physical_state_count",

    "device_conductance_mode",

    "device_state_count_status",

    "parameter_source",

]


CONFIG_MODEL_FEATURES = [

    "mapping_strategy",

    "precision_basis",

    "crossbar_size",

    "requested_weight_bits",

    "effective_weight_levels",

    "slices_per_branch",

    "physical_cells_per_weight",

    "adc_bits",

]


COST_COLUMNS = [

    "estimated_memristor_cells",

    "estimated_physical_crossbar_tiles",

    "adc_levels",

    "relative_hardware_cost_proxy",

]


TARGET_COLUMNS = [

    "accuracy",
    "accuracy_loss",

]


# ============================================================
# Save dataset
# ============================================================

Path(
    "results/tables"
).mkdir(
    parents=True,
    exist_ok=True
)


ml.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# Save machine-readable manifest
#
# Later training scripts can read this instead of guessing
# which columns are allowed as model inputs.
# ============================================================

manifest = {

    "dataset_file":
        OUTPUT_FILE,

    "rows":
        int(
            len(ml)
        ),

    "devices":
        int(
            ml[
                "device_id"
            ].nunique()
        ),

    "configurations_per_device":
        int(
            EXPECTED_CONFIGS_PER_DEVICE
        ),

    "configuration_space_source":
        "src/config_space.py",

    "group_holdout_column":
        "device_id",

    "group_columns":
        GROUP_COLUMNS,

    "provenance_columns":
        PROVENANCE_COLUMNS,

    "device_model_features":
        DEVICE_MODEL_FEATURES,

    "configuration_model_features":
        CONFIG_MODEL_FEATURES,

    "cost_columns":
        COST_COLUMNS,

    "target_columns":
        TARGET_COLUMNS,

    "recommended_target":
        "accuracy",

    "split_rule":
        (
            "LEAVE_ONE_DEVICE_OUT; "
            "NEVER_RANDOM_ROW_SPLIT"
        ),

    "important_notes": [

        (
            "device_id must be used only for grouping "
            "and held-out evaluation, not as a model feature."
        ),

        (
            "technology_family is retained for provenance "
            "and reporting."
        ),

        (
            "The configuration count is read dynamically "
            "from src/config_space.py."
        ),

        (
            "physical_state_count=0 means no fixed physical "
            "state count is available; consult "
            "state_count_available."
        ),

        (
            "effective_weight_levels describes accelerator "
            "mapping precision, not necessarily the physical "
            "state count of one memristor."
        ),

        (
            "relative_hardware_cost_proxy is heuristic and "
            "must not be reported as measured energy, power, "
            "latency, or silicon area."
        ),

        (
            "With only four physical device profiles, "
            "this dataset is suitable for pipeline testing "
            "but not yet for strong generalization claims."
        ),

    ],

}


with open(

    MANIFEST_FILE,

    "w",

    encoding="utf-8",

) as f:

    json.dump(

        manifest,

        f,

        indent=4,

    )


# ============================================================
# Display summary
# ============================================================

print()

print(
    "ML DATASET CREATED"
)

print(
    "========================================"
)


print(
    "Rows:",
    len(
        ml
    )
)


print(
    "Physical devices:",
    ml[
        "device_id"
    ].nunique()
)


print(
    "Technology families:",
    ml[
        "technology_family"
    ].nunique()
)


print(
    "Configurations per device:",
    EXPECTED_CONFIGS_PER_DEVICE
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

    ml[
        "device_id"
    ]
    .value_counts()
    .sort_index()

)


# ============================================================
# Device descriptor summary
# ============================================================

print()

print(
    "DEVICE DESCRIPTORS"
)


summary = (

    ml[

        [
            "device_id",
            "technology_family",
            "device_conductance_mode",
            "device_on_off_ratio",
            "state_count_available",
            "physical_state_count",
            "parameter_source",
        ]

    ]

    .drop_duplicates()

    .sort_values(
        "device_id"
    )

)


print(

    summary.to_string(
        index=False
    )

)


# ============================================================
# Mapping summary
# ============================================================

print()

print(
    "MAPPING STRATEGIES"
)


mapping_summary = (

    ml[

        [
            "device_id",
            "mapping_strategy",
            "precision_basis",
        ]

    ]

    .drop_duplicates()

    .sort_values(
        "device_id"
    )

)


print(

    mapping_summary.to_string(
        index=False
    )

)


# ============================================================
# Leakage warning
# ============================================================

print()

print(
    "IMPORTANT ML SPLIT RULE"
)

print(
    "----------------------------------------"
)


print(

    f"DO NOT randomly split the {len(ml)} rows "
    "into train and test sets."

)


print(

    f"All {EXPECTED_CONFIGS_PER_DEVICE} "
    "configurations of one device "
    "must stay together."

)


print()

print(

    "Required evaluation: "
    "LEAVE ONE DEVICE OUT."

)


print()

print(

    "Example: train on ZnO_01 + TaOx_01 + HfOx_02, "
    "then test only on TiOx_03."

)


# ============================================================
# Dataset-size limitation
# ============================================================

independent_device_count = int(

    ml[
        "device_id"
    ].nunique()

)


print()

print(
    "CURRENT LIMITATION"
)

print(
    "----------------------------------------"
)


print(

    f"There are {len(ml)} simulator rows, "
    f"but only {independent_device_count} "
    "independent physical devices."

)


print(

    f"The {len(ml)} rows must NOT be presented as "
    f"{len(ml)} independent device samples."

)


print(

    "This is currently a pilot dataset for validating "
    "the zero-shot pipeline."

)


# ============================================================
# Saved files
# ============================================================

print()

print(
    "Saved dataset to:",
    OUTPUT_FILE
)


print(
    "Saved manifest to:",
    MANIFEST_FILE
)
