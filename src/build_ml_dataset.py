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
# Expected validated device profiles
# ============================================================

EXPECTED_DEVICES = {

    "ZnO_01",
    "TaOx_01",

    "HfOx_01",
    "HfOx_02",

    "HfZrOx_01",

    "TiOx_03",
    "TiOx_04",

    "TiOx_02_Au",
    "TiOx_02_Ni",
    "TiOx_02_Pt",

}


# ============================================================
# Expected accelerator configuration space
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
# Required columns
# ============================================================

DEVICE_REQUIRED_COLUMNS = [

    "device_id",
    "study_id",
    "technology_family",
    "ron_ohm",
    "roff_ohm",
    "on_off_ratio",
    "conductance_states",
    "conductance_mode",
    "state_count_status",

]


RESULT_REQUIRED_COLUMNS = [

    "device_id",
    "study_id",
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
    !=
    EXPECTED_DEVICES
):

    raise ValueError(

        "Unexpected device set in combined results.\n"
        f"Expected: {sorted(EXPECTED_DEVICES)}\n"
        f"Found:    {sorted(used_devices)}"

    )


# ============================================================
# Complete configuration space per device
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
# Duplicate configuration check
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
# Exact configuration-grid check
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


# ============================================================
# Validate total result size
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
# Keep only validated profiles
# ============================================================

devices = devices[

    devices[
        "device_id"
    ].isin(
        used_devices
    )

].copy()


profile_counts = (

    devices[
        "device_id"
    ]
    .value_counts()

)


bad_profile_counts = (

    profile_counts[
        profile_counts != 1
    ]

)


missing_profile_ids = (

    EXPECTED_DEVICES
    -
    set(
        devices[
            "device_id"
        ].unique()
    )

)


if (

    not bad_profile_counts.empty

    or

    missing_profile_ids

):

    raise ValueError(

        "Each simulated device must have exactly "
        "one device-profile row.\n"
        f"Bad counts:\n{bad_profile_counts}\n"
        f"Missing devices: {sorted(missing_profile_ids)}"

    )


# ============================================================
# Validate study_id
# ============================================================

devices[
    "study_id"
] = (

    devices[
        "study_id"
    ]
    .apply(
        clean_text
    )

)


missing_study = (

    devices[
        "study_id"
    ]
    ==
    ""

)


if missing_study.any():

    raise ValueError(

        "Every simulated device must have a non-empty "
        "study_id.\n"
        f"Missing for: "
        f"{devices.loc[missing_study, 'device_id'].tolist()}"

    )


results[
    "study_id"
] = (

    results[
        "study_id"
    ]
    .apply(
        clean_text
    )

)


# ============================================================
# Cross-check profile metadata against combined results
# ============================================================

profile_metadata = (

    devices[

        [
            "device_id",
            "study_id",
            "technology_family",
        ]

    ]

    .copy()

)


result_metadata = (

    results[

        [
            "device_id",
            "study_id",
            "technology_family",
        ]

    ]

    .drop_duplicates()

)


if (
    len(
        result_metadata
    )
    !=
    len(
        EXPECTED_DEVICES
    )
):

    raise ValueError(

        "Combined results contain inconsistent "
        "device/study/family metadata."

    )


metadata_check = profile_metadata.merge(

    result_metadata,

    on="device_id",

    how="outer",

    suffixes=(
        "_profile",
        "_result",
    ),

    indicator=True,

)


metadata_problem = (

    (
        metadata_check[
            "_merge"
        ]
        !=
        "both"
    )

    |

    (
        metadata_check[
            "study_id_profile"
        ]
        !=
        metadata_check[
            "study_id_result"
        ]
    )

    |

    (
        metadata_check[
            "technology_family_profile"
        ]
        !=
        metadata_check[
            "technology_family_result"
        ]
    )

)


if metadata_problem.any():

    raise ValueError(

        "study_id or technology_family mismatch between "
        "device_profiles.csv and combined results.\n"
        f"{metadata_check.loc[metadata_problem].to_string(index=False)}"

    )


# ============================================================
# Build one physical descriptor row per device
# ============================================================

device_rows = []


for _, row in devices.iterrows():

    device_id = str(
        row[
            "device_id"
        ]
    )


    study_id = clean_text(
        row[
            "study_id"
        ]
    )


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
    # Usable ON/OFF ratio
    #
    # 1. use stored ratio when present
    # 2. otherwise derive ROFF / RON
    # 3. never invent a value
    # --------------------------------------------------------

    if (

        pd.notna(
            ratio
        )

        and

        float(
            ratio
        ) > 0

    ):

        usable_ratio = float(
            ratio
        )


    elif (

        pd.notna(
            ron
        )

        and

        pd.notna(
            roff
        )

        and

        float(
            ron
        ) > 0

        and

        float(
            roff
        ) > 0

    ):

        usable_ratio = (

            float(
                roff
            )

            /

            float(
                ron
            )

        )


    else:

        raise ValueError(

            f"{device_id} has no usable "
            "ON/OFF ratio."

        )


    # --------------------------------------------------------
    # Physical conductance-state information
    #
    # physical_state_count = 0 is only a machine-readable
    # placeholder when no fixed state count is reported.
    # state_count_available carries the meaning.
    # --------------------------------------------------------

    states = row[
        "conductance_states"
    ]


    if (

        pd.notna(
            states
        )

        and

        float(
            states
        ) >= 2

    ):

        state_count_available = 1

        physical_state_count = int(
            float(
                states
            )
        )


    else:

        state_count_available = 0

        physical_state_count = 0


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


    device_rows.append({

        # Grouping/provenance only
        "device_id":
            device_id,

        "study_id":
            study_id,

        "technology_family":
            row[
                "technology_family"
            ],


        # Device features
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
# Merge device-level X with architecture Y and performance S
# ============================================================

ml = results.merge(

    device_features,

    on=[

        "device_id",
        "study_id",
        "technology_family",

    ],

    how="inner",

    validate="many_to_one",

)


if (
    len(
        ml
    )
    !=
    len(
        results
    )
):

    raise ValueError(

        "ML merge changed the number of simulation rows.\n"
        f"Results rows: {len(results)}\n"
        f"ML rows:      {len(ml)}"

    )


# ============================================================
# Semantic cleanup of bit-slicing
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
    # NONE of these columns are model inputs.
    # ========================================================

    "device_id",
    "study_id",
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
    # DERIVED COST DESCRIPTORS
    #
    # NOT measured area / power / energy / latency.
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
# Required-field missing-value check
# ============================================================

missing_values = (

    ml
    .isna()
    .sum()

)


missing_values = (

    missing_values[
        missing_values > 0
    ]

)


if not missing_values.empty:

    raise ValueError(

        "Unexpected missing values remain "
        "in the ML dataset:\n"
        f"{missing_values}"

    )


# ============================================================
# Leakage structure checks
# ============================================================

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


# Every device must map to exactly one study and one family.
device_group_check = (

    ml

    .groupby(
        "device_id"
    )

    .agg(

        studies=(
            "study_id",
            "nunique",
        ),

        families=(
            "technology_family",
            "nunique",
        ),

    )

)


if (

    (
        device_group_check[
            "studies"
        ]
        != 1
    ).any()

    or

    (
        device_group_check[
            "families"
        ]
        != 1
    ).any()

):

    raise ValueError(

        "A device maps to multiple study IDs or "
        "technology families."

    )


# ============================================================
# Explicit feature roles
# ============================================================

GROUP_COLUMNS = [

    "device_id",
    "study_id",
    "technology_family",

]


PROVENANCE_COLUMNS = [

    "study_id",
    "technology_family",

]


DEVICE_MODEL_FEATURES = [

    "device_log10_on_off_ratio",

    "state_count_available",
    "physical_state_count",

    "device_conductance_mode",

]


CONFIG_MODEL_FEATURES = [

    "mapping_strategy",

    "crossbar_size",
    "requested_weight_bits",
    "effective_weight_levels",

    "slices_per_branch",
    "physical_cells_per_weight",

    "adc_bits",

]


MODEL_INPUT_FEATURES = (

    DEVICE_MODEL_FEATURES
    +
    CONFIG_MODEL_FEATURES

)


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

    exist_ok=True,

)


ml.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# Dataset structure
# ============================================================

device_count = int(

    ml[
        "device_id"
    ].nunique()

)


study_count = int(

    ml[
        "study_id"
    ].nunique()

)


family_count = int(

    ml[
        "technology_family"
    ].nunique()

)


study_device_counts = (

    ml[

        [
            "study_id",
            "device_id",
        ]

    ]

    .drop_duplicates()

    .groupby(
        "study_id"
    )[
        "device_id"
    ]
    .nunique()

)


# ============================================================
# Save machine-readable manifest
# ============================================================

manifest = {

    "dataset_file":
        OUTPUT_FILE,

    "rows":
        int(
            len(
                ml
            )
        ),

    "device_profiles":
        device_count,

    # Backward-friendly alias.
    "devices":
        device_count,

    "distinct_source_studies":
        study_count,

    "technology_families":
        family_count,

    "configurations_per_device":
        int(
            EXPECTED_CONFIGS_PER_DEVICE
        ),

    "configuration_space_source":
        "src/config_space.py",

    "device_holdout_column":
        "device_id",

    "study_holdout_column":
        "study_id",

    "family_holdout_column":
        "technology_family",

    "group_columns":
        GROUP_COLUMNS,

    "provenance_columns":
        PROVENANCE_COLUMNS,

    "device_model_features":
        DEVICE_MODEL_FEATURES,

    "configuration_model_features":
        CONFIG_MODEL_FEATURES,

    "model_input_features":
        MODEL_INPUT_FEATURES,

    "cost_columns":
        COST_COLUMNS,

    "target_columns":
        TARGET_COLUMNS,

    "recommended_target":
        "accuracy",

    # Keep split_rule for downstream compatibility.
    "split_rule":
        (
            "STUDY_BLOCKED_LEAVE_ONE_DEVICE_OUT; "
            "EXCLUDE_ALL_SAME_STUDY_SIBLINGS_FROM_TRAINING; "
            "NEVER_RANDOM_ROW_SPLIT"
        ),

    "additional_validation_rules": [

        "LEAVE_ONE_STUDY_OUT",

        "LEAVE_ONE_FAMILY_OUT",

    ],

    "important_notes": [

        (
            "device_id, study_id and technology_family "
            "are grouping/provenance variables and must "
            "not be used as model features."
        ),

        (
            "Primary device-level evaluation must exclude "
            "every training device sharing the held-out "
            "device's study_id. This prevents same-study "
            "sibling leakage."
        ),

        (
            "TiOx_02_Au, TiOx_02_Ni and TiOx_02_Pt "
            "are three device profiles from one experimental "
            "study, not three independent studies."
        ),

        (
            "HfOx_01 keeps the 2024 variability paper as its "
            "primary study_id. Its GRADUAL_MULTILEVEL "
            "classification has separate same-stack supporting "
            "literature recorded in source_traceability.csv."
        ),

        (
            "HfZrOx_01 is an independent low-ON/OFF ANALOG "
            "profile with reported post-wake-up ON/OFF=2.0. "
            "Its absolute RON/ROFF are not available, so "
            "parameter_source is normalized_from_ratio."
        ),

        (
            "TiOx_04 is an independent TiOx/TiOy "
            "GRADUAL_MULTILEVEL profile with reported "
            "ON/OFF approximately 76 at 0.2 V. Its absolute "
            "RON/ROFF are not available, so parameter_source "
            "is normalized_from_ratio. Protocol-generated "
            "multi-level states are not treated as a fixed "
            "intrinsic physical state count."
        ),

        (
            "Leave-one-study-out evaluates transfer to an "
            "entire unseen experimental study."
        ),

        (
            "Leave-one-family-out evaluates transfer to an "
            "unseen material technology family."
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
            "Seven device profiles from five source studies "
            "remain a small research dataset. Results should "
            "be presented as pilot validation rather than "
            "proof of broad physical generalization."
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
# Study membership
# ============================================================

print()

print(
    "STUDY MEMBERSHIP"
)

print(
    "----------------------------------------"
)


study_membership = (

    ml[

        [
            "study_id",
            "device_id",
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

    study_membership.to_string(
        index=False
    )

)


print()

print(
    "Devices per study:"
)


print(
    study_device_counts
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
            "study_id",
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
# Split policy
# ============================================================

print()

print(
    "IMPORTANT ML SPLIT POLICY"
)

print(
    "----------------------------------------"
)


print(

    f"DO NOT randomly split the {len(ml)} rows "
    "into train and test sets."

)


print(

    f"All {EXPECTED_CONFIGS_PER_DEVICE} configurations "
    "of one device must stay together."

)


print()

print(

    "PRIMARY: STUDY-BLOCKED LEAVE-ONE-DEVICE-OUT"

)


print(

    "When one device is tested, every sibling device "
    "from the same study_id is also removed from training."

)


print()

print(

    "ADDITIONAL: LEAVE-ONE-STUDY-OUT"

)


print(

    "ADDITIONAL: LEAVE-ONE-FAMILY-OUT"

)


# ============================================================
# Dataset-size limitation
# ============================================================

print()

print(
    "CURRENT LIMITATION"
)

print(
    "----------------------------------------"
)


print(

    f"There are {len(ml)} simulator rows generated from "
    f"{device_count} device profiles, "
    f"{study_count} source studies and "
    f"{family_count} technology families."

)


print(

    f"The {len(ml)} rows must NOT be presented as "
    f"{len(ml)} independent physical-device samples."

)


print(

    "This remains a pilot literature-to-simulator dataset "
    "for testing cross-device and cross-family transfer."

)


print()

print(
    "Saved dataset to:",
    OUTPUT_FILE
)


print(
    "Saved manifest to:",
    MANIFEST_FILE
)
