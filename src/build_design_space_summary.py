from pathlib import Path

import pandas as pd

from config_space import (
    CROSSBAR_SIZES,
    WEIGHT_BITS,
    ADC_BITS,
    generate_configs,
)


# ============================================================
# PURPOSE
# ============================================================
#
# Build mentor-ready evidence for:
#
#     "Can the original 27 configurations be increased?"
#
#
# IMPORTANT TERMINOLOGY
# ============================================================
#
# The project does NOT have 245 independent parameters.
#
# It currently varies THREE accelerator-design parameters:
#
#     1. crossbar size
#     2. requested weight precision
#     3. ADC precision
#
# Their Cartesian product creates 245 requested
# accelerator configurations per device.
#
#
# The design-space expansion is:
#
#     original pilot: 27 configurations
#     V2:             64 configurations
#     V3:            245 configurations
#
#
# No fake temperature, noise, thickness, or material
# variables are added merely to inflate the search space.
# ============================================================


# ============================================================
# Historical project milestones
# ============================================================

ORIGINAL_CONFIGURATIONS = 27

V2_CONFIGURATIONS = 64


# ============================================================
# Files
# ============================================================

COMBINED_RESULTS_FILE = (
    "results/tables/"
    "all_device_config_results.csv"
)


OPTIMAL_RESULTS_FILE = (
    "results/tables/"
    "device_optimal_configs.csv"
)


OUTPUT_CSV = (
    "results/tables/"
    "design_space_evidence.csv"
)


OUTPUT_TEXT = (
    "results/tables/"
    "design_space_summary.txt"
)


# ============================================================
# Helpers
# ============================================================

def require_file(
    file_path,
):

    path = Path(
        file_path
    )


    if not path.exists():

        raise FileNotFoundError(

            f"Required file not found: "
            f"{file_path}"

        )


    return path


# ============================================================
# Main
# ============================================================

def main():

    print()

    print(
        "BUILDING DESIGN-SPACE EXPANSION SUMMARY"
    )

    print(
        "========================================"
    )


    # ========================================================
    # Current configuration space
    # ========================================================

    current_configs = generate_configs()


    configurations_per_device = len(
        current_configs
    )


    design_parameter_count = 3


    crossbar_choice_count = len(
        CROSSBAR_SIZES
    )


    weight_choice_count = len(
        WEIGHT_BITS
    )


    adc_choice_count = len(
        ADC_BITS
    )


    expected_count = (

        crossbar_choice_count
        *
        weight_choice_count
        *
        adc_choice_count

    )


    if (
        configurations_per_device
        !=
        expected_count
    ):

        raise ValueError(

            "Configuration-space size does not match "
            "the Cartesian product of the three "
            "design axes."

        )


    # ========================================================
    # Ensure generated configurations are unique
    # ========================================================

    configuration_keys = {

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
        in current_configs

    }


    if (
        len(
            configuration_keys
        )
        !=
        configurations_per_device
    ):

        raise ValueError(

            "Duplicate configurations were generated "
            "by config_space.py."

        )


    # ========================================================
    # Load current V3 results
    # ========================================================

    require_file(
        COMBINED_RESULTS_FILE
    )


    require_file(
        OPTIMAL_RESULTS_FILE
    )


    results = pd.read_csv(
        COMBINED_RESULTS_FILE
    )


    optimal = pd.read_csv(
        OPTIMAL_RESULTS_FILE
    )


    # ========================================================
    # Required result columns
    # ========================================================

    required_result_columns = [

        "device_id",
        "crossbar_size",
        "requested_weight_bits",
        "effective_weight_levels",
        "adc_bits",
        "accuracy",
        "physical_cells_per_weight",

    ]


    missing_columns = [

        column

        for column
        in required_result_columns

        if column
        not in results.columns

    ]


    if missing_columns:

        raise ValueError(

            "Combined results are missing required "
            f"columns: {missing_columns}"

        )


    # ========================================================
    # Device structure
    # ========================================================

    physical_devices = int(

        results[
            "device_id"
        ]
        .nunique()

    )


    rows_per_device = (

        results[
            "device_id"
        ]
        .value_counts()

    )


    if rows_per_device.nunique() != 1:

        raise ValueError(

            "Not every device contributes the same "
            "number of configurations."

        )


    actual_configurations_per_device = int(

        rows_per_device.iloc[0]

    )


    if (

        actual_configurations_per_device
        !=
        configurations_per_device

    ):

        raise ValueError(

            "Current combined results do not match "
            "config_space.py.\n"
            f"Expected per device: "
            f"{configurations_per_device}\n"
            f"Found per device: "
            f"{actual_configurations_per_device}"

        )


    total_simulation_rows = len(
        results
    )


    expected_total_rows = (

        physical_devices
        *
        configurations_per_device

    )


    if (
        total_simulation_rows
        !=
        expected_total_rows
    ):

        raise ValueError(

            "Unexpected total simulation-row count.\n"
            f"Expected: {expected_total_rows}\n"
            f"Found:    {total_simulation_rows}"

        )


    # ========================================================
    # Expansion factors
    # ========================================================

    growth_vs_original = (

        configurations_per_device
        /
        ORIGINAL_CONFIGURATIONS

    )


    growth_vs_v2 = (

        configurations_per_device
        /
        V2_CONFIGURATIONS

    )


    increase_vs_original_pct = (

        100.0
        *
        (
            configurations_per_device
            -
            ORIGINAL_CONFIGURATIONS
        )
        /
        ORIGINAL_CONFIGURATIONS

    )


    increase_vs_v2_pct = (

        100.0
        *
        (
            configurations_per_device
            -
            V2_CONFIGURATIONS
        )
        /
        V2_CONFIGURATIONS

    )


    # ========================================================
    # Device-dependent effective configuration behavior
    #
    # Requested precision and effective precision are not
    # always identical.
    #
    # Example:
    #
    # TaOx_01 has a device-state cap.
    #
    # Therefore W4, W5, W6, W7 and W8 may all map to the
    # same maximum effective signed-level count.
    #
    # This is important:
    #
    # 245 is the REQUESTED accelerator design space.
    #
    # Device physics can collapse some requested designs
    # into equivalent effective behavior.
    # ========================================================

    effective_summary_rows = []


    for (
        device_id,
        device_df

    ) in results.groupby(
        "device_id"
    ):

        requested_weight_choices = int(

            device_df[
                "requested_weight_bits"
            ]
            .nunique()

        )


        effective_level_choices = int(

            device_df[
                "effective_weight_levels"
            ]
            .nunique()

        )


        max_effective_levels = int(

            device_df[
                "effective_weight_levels"
            ]
            .max()

        )


        effective_summary_rows.append({

            "device_id":
                device_id,

            "requested_weight_precision_choices":
                requested_weight_choices,

            "effective_weight_level_choices":
                effective_level_choices,

            "maximum_effective_weight_levels":
                max_effective_levels,

        })


    effective_summary = pd.DataFrame(
        effective_summary_rows
    )


    # ========================================================
    # Cost-aware optimum check
    # ========================================================

    required_optimal_columns = [

        "device_id",
        "crossbar_size",
        "requested_weight_bits",
        "adc_bits",
        "accuracy",
        "regret_pp",

    ]


    missing_optimal_columns = [

        column

        for column
        in required_optimal_columns

        if column
        not in optimal.columns

    ]


    if missing_optimal_columns:

        raise ValueError(

            "device_optimal_configs.csv is missing "
            f"required columns: {missing_optimal_columns}"

        )


    optimal_device_count = int(

        optimal[
            "device_id"
        ]
        .nunique()

    )


    if (
        optimal_device_count
        !=
        physical_devices
    ):

        raise ValueError(

            "Cost-aware optimum table does not contain "
            "one selected result for every simulated device."

        )


    mean_selected_regret = float(

        optimal[
            "regret_pp"
        ]
        .mean()

    )


    worst_selected_regret = float(

        optimal[
            "regret_pp"
        ]
        .max()

    )


    # ========================================================
    # Build evidence table
    # ========================================================

    evidence_rows = [

        {
            "evidence_id":
                "DESIGN_PARAMETER_COUNT",

            "metric_category":
                "DESIGN_SPACE_STRUCTURE",

            "metric_name":
                "Independent accelerator design parameters",

            "value":
                design_parameter_count,

            "unit":
                "parameters",

            "interpretation":
                (
                    "The current search varies crossbar size, "
                    "requested weight precision and ADC precision."
                ),
        },

        {
            "evidence_id":
                "CROSSBAR_CHOICES",

            "metric_category":
                "DESIGN_SPACE_STRUCTURE",

            "metric_name":
                "Crossbar-size choices",

            "value":
                crossbar_choice_count,

            "unit":
                "choices",

            "interpretation":
                str(
                    CROSSBAR_SIZES
                ),
        },

        {
            "evidence_id":
                "WEIGHT_PRECISION_CHOICES",

            "metric_category":
                "DESIGN_SPACE_STRUCTURE",

            "metric_name":
                "Requested weight-precision choices",

            "value":
                weight_choice_count,

            "unit":
                "choices",

            "interpretation":
                str(
                    WEIGHT_BITS
                ),
        },

        {
            "evidence_id":
                "ADC_PRECISION_CHOICES",

            "metric_category":
                "DESIGN_SPACE_STRUCTURE",

            "metric_name":
                "ADC-precision choices",

            "value":
                adc_choice_count,

            "unit":
                "choices",

            "interpretation":
                str(
                    ADC_BITS
                ),
        },

        {
            "evidence_id":
                "CONFIGURATIONS_PER_DEVICE",

            "metric_category":
                "DESIGN_SPACE_SIZE",

            "metric_name":
                "Requested configurations per device",

            "value":
                configurations_per_device,

            "unit":
                "configurations",

            "interpretation":
                (
                    f"{crossbar_choice_count} crossbar choices "
                    f"x {weight_choice_count} weight choices "
                    f"x {adc_choice_count} ADC choices."
                ),
        },

        {
            "evidence_id":
                "TOTAL_SIMULATION_ROWS",

            "metric_category":
                "DESIGN_SPACE_SIZE",

            "metric_name":
                "Current device-configuration simulation rows",

            "value":
                total_simulation_rows,

            "unit":
                "rows",

            "interpretation":
                (
                    f"{physical_devices} physical devices "
                    f"x {configurations_per_device} "
                    f"configurations per device."
                ),
        },

        {
            "evidence_id":
                "EXPANSION_FROM_ORIGINAL",

            "metric_category":
                "DESIGN_SPACE_GROWTH",

            "metric_name":
                "Growth versus original 27-case space",

            "value":
                growth_vs_original,

            "unit":
                "times",

            "interpretation":
                (
                    f"27 -> {configurations_per_device} "
                    f"configurations per device."
                ),
        },

        {
            "evidence_id":
                "EXPANSION_FROM_V2",

            "metric_category":
                "DESIGN_SPACE_GROWTH",

            "metric_name":
                "Growth versus V2 64-case space",

            "value":
                growth_vs_v2,

            "unit":
                "times",

            "interpretation":
                (
                    f"64 -> {configurations_per_device} "
                    f"configurations per device."
                ),
        },

        {
            "evidence_id":
                "COST_AWARE_MEAN_REGRET",

            "metric_category":
                "DESIGN_SPACE_USEFULNESS",

            "metric_name":
                "Mean regret of exhaustive cost-aware selections",

            "value":
                mean_selected_regret,

            "unit":
                "percentage_points",

            "interpretation":
                (
                    "The expanded space supports selection of "
                    "lower-cost configurations while remaining "
                    "within the defined 0.5 percentage-point "
                    "near-optimal accuracy region."
                ),
        },

    ]


    evidence_df = pd.DataFrame(
        evidence_rows
    )


    # ========================================================
    # Save evidence
    # ========================================================

    Path(
        "results/tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


    evidence_df.to_csv(
        OUTPUT_CSV,
        index=False,
    )


    # ========================================================
    # Human-readable report
    # ========================================================

    lines = [

        "DESIGN-SPACE EXPANSION SUMMARY",
        "==============================",
        "",

        "1. TERMINOLOGY",
        "--------------",
        "",
        (
            "The project currently varies 3 independent "
            "accelerator-design parameters."
        ),
        "",
        (
            f"These 3 parameters generate "
            f"{configurations_per_device} requested "
            f"accelerator configurations per device."
        ),
        "",
        (
            f"Therefore, the correct wording is "
            f"'{configurations_per_device} configurations', "
            "not "
            f"'{configurations_per_device} parameters'."
        ),
        "",

        "2. CURRENT DESIGN AXES",
        "----------------------",
        "",
        (
            f"Crossbar sizes: "
            f"{CROSSBAR_SIZES}"
        ),
        (
            f"Requested weight bits: "
            f"{WEIGHT_BITS}"
        ),
        (
            f"ADC bits: "
            f"{ADC_BITS}"
        ),
        "",
        (
            f"Configuration count = "
            f"{crossbar_choice_count} x "
            f"{weight_choice_count} x "
            f"{adc_choice_count} "
            f"= {configurations_per_device}."
        ),
        "",

        "3. DESIGN-SPACE EXPANSION",
        "-------------------------",
        "",
        (
            f"Original pilot: "
            f"{ORIGINAL_CONFIGURATIONS} configurations."
        ),
        (
            f"V2: "
            f"{V2_CONFIGURATIONS} configurations."
        ),
        (
            f"Current V3: "
            f"{configurations_per_device} configurations."
        ),
        "",
        (
            f"Growth from original: "
            f"{growth_vs_original:.2f}x "
            f"({increase_vs_original_pct:.1f}% increase)."
        ),
        (
            f"Growth from V2: "
            f"{growth_vs_v2:.2f}x "
            f"({increase_vs_v2_pct:.1f}% increase)."
        ),
        "",

        "4. CURRENT SIMULATION SCALE",
        "---------------------------",
        "",
        (
            f"Simulation-ready physical devices: "
            f"{physical_devices}."
        ),
        (
            f"Configurations per device: "
            f"{configurations_per_device}."
        ),
        (
            f"Total device-configuration rows: "
            f"{total_simulation_rows}."
        ),
        "",
        (
            f"The {total_simulation_rows} rows are not "
            f"{total_simulation_rows} independent "
            "physical-device samples."
        ),
        "",
        (
            f"There are still only "
            f"{physical_devices} independent physical "
            "devices."
        ),
        "",

        "5. WHY THE EXPANSION IS MEANINGFUL",
        "-----------------------------------",
        "",
        (
            "The larger search space was created by making "
            "the existing simulator-supported accelerator "
            "axes denser."
        ),
        "",
        (
            "No unsupported temperature, noise, material "
            "thickness, or other artificial parameter was "
            "added simply to inflate the number of cases."
        ),
        "",
        (
            "The expanded search therefore explores more "
            "crossbar-size, weight-precision, and ADC-precision "
            "tradeoffs while using the same validated "
            "simulation pipeline."
        ),
        "",

        "6. DEVICE PHYSICS STILL CONSTRAINS THE SPACE",
        "---------------------------------------------",
        "",

    ]


    for _, row in effective_summary.iterrows():

        lines.append(

            (
                f"{row['device_id']}: "
                f"{int(row['requested_weight_precision_choices'])} "
                f"requested weight-precision choices -> "
                f"{int(row['effective_weight_level_choices'])} "
                f"distinct effective-level counts; "
                f"maximum effective signed levels = "
                f"{int(row['maximum_effective_weight_levels'])}."
            )

        )


    lines.extend([

        "",
        (
            "This means 245 is the requested accelerator "
            "configuration space."
        ),
        "",
        (
            "A physical device may make several requested "
            "configurations effectively equivalent."
        ),
        "",
        (
            "For example, TaOx_01 is limited by its "
            "7-state device capability, giving a maximum "
            "of 13 signed differential weight levels."
        ),
        "",

        "7. EXPANDED SEARCH USEFULNESS",
        "-----------------------------",
        "",
        (
            f"Mean regret of the exhaustive cost-aware "
            f"selected configurations: "
            f"{mean_selected_regret:.3f} percentage points."
        ),
        (
            f"Worst selected regret: "
            f"{worst_selected_regret:.3f} percentage points."
        ),
        "",
        (
            "The expanded search therefore supports "
            "lower-cost near-optimal design selection rather "
            "than searching only for maximum simulated "
            "accuracy."
        ),
        "",

        "MENTOR ANSWER: CAN 27 BE INCREASED?",
        "------------------------------------",
        "",
        (
            "Yes. The original 27-configuration search space "
            f"has been expanded to "
            f"{configurations_per_device} configurations "
            "per device."
        ),
        "",
        (
            "The expansion uses 5 crossbar sizes, "
            "7 requested weight precisions, and "
            "7 ADC precisions."
        ),
        "",
        (
            f"Across the {physical_devices} currently "
            f"simulation-ready devices, this produces "
            f"{total_simulation_rows} device-configuration "
            "simulation rows."
        ),
        "",
        (
            "The important point is that these are "
            "configurations, not independent parameters."
        ),
        "",
        (
            "The project still varies 3 accelerator-design "
            "parameters, and the new values were added only "
            "where the simulator already supports them."
        ),
        "",

        "CURRENT LIMITATION",
        "------------------",
        "",
        (
            "A larger configuration grid strengthens "
            "design-space exploration, but it does not "
            "increase the number of independent physical "
            "memristor devices."
        ),
        "",
        (
            f"The project still has only "
            f"{physical_devices} independent simulation-ready "
            "device profiles."
        ),
        "",
        (
            "Future expansion should prioritize additional "
            "literature-supported physical devices and "
            "additional validated physics, rather than "
            "inflating the configuration count alone."
        ),

    ])


    with open(
        OUTPUT_TEXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )


    # ========================================================
    # Console
    # ========================================================

    print()

    print(
        "Independent design parameters:",
        design_parameter_count
    )


    print(
        "Crossbar choices:",
        crossbar_choice_count
    )


    print(
        "Weight-bit choices:",
        weight_choice_count
    )


    print(
        "ADC-bit choices:",
        adc_choice_count
    )


    print()

    print(
        "Configurations per device:",
        configurations_per_device
    )


    print(
        "Simulation-ready devices:",
        physical_devices
    )


    print(
        "Total simulation rows:",
        total_simulation_rows
    )


    print()

    print(
        "Growth vs original 27:",
        f"{growth_vs_original:.2f}x"
    )


    print(
        "Growth vs V2 64:",
        f"{growth_vs_v2:.2f}x"
    )


    print()

    print(
        "Mean cost-aware selected regret:",
        f"{mean_selected_regret:.3f} pp"
    )


    print(
        "Worst cost-aware selected regret:",
        f"{worst_selected_regret:.3f} pp"
    )


    print()

    print(
        "Saved CSV:"
    )

    print(
        OUTPUT_CSV
    )


    print()

    print(
        "Saved mentor summary:"
    )

    print(
        OUTPUT_TEXT
    )


    print()

    print(
        "CURRENT CLAIM"
    )

    print(
        "-------------"
    )

    print()

    print(

        f"The project varies 3 accelerator-design "
        f"parameters to create "
        f"{configurations_per_device} requested "
        f"configurations per device, producing "
        f"{total_simulation_rows} current "
        f"device-configuration simulation rows."

    )


if __name__ == "__main__":

    main()