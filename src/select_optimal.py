import pandas as pd


# ============================================================
# Files
# ============================================================

INPUT_FILE = (
    "results/tables/"
    "all_device_config_results.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "device_optimal_configs.csv"
)


# ============================================================
# Near-optimal accuracy tolerance
#
# A configuration is considered near-optimal if its simulated
# accuracy is no more than this many percentage points below
# the exhaustive maximum for that device.
# ============================================================

ACCURACY_TOLERANCE_PP = 0.5


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [

    "device_id",
    "technology_family",
    "conductance_mode",
    "mapping_strategy",
    "precision_basis",

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


# ============================================================
# Load validated combined dataset
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)


# ============================================================
# Validate schema
# ============================================================

missing_columns = [

    column

    for column
    in REQUIRED_COLUMNS

    if column
    not in df.columns

]


if missing_columns:

    raise ValueError(

        "Combined dataset is missing required columns:\n"
        f"{missing_columns}"

    )


# ============================================================
# Selected results
# ============================================================

selected_rows = []


# ============================================================
# Process every memristor device separately
# ============================================================

for device_id, group in df.groupby(
    "device_id"
):

    group = group.copy()


    # ========================================================
    # Exhaustive maximum accuracy
    # ========================================================

    best_accuracy = (
        group["accuracy"].max()
    )


    # ========================================================
    # Near-optimal accuracy threshold
    #
    # Example:
    #
    # exhaustive best = 96.30%
    # tolerance       = 0.50 percentage points
    #
    # minimum accepted accuracy:
    #
    # 95.80%
    # ========================================================

    minimum_allowed_accuracy = (

        best_accuracy
        - ACCURACY_TOLERANCE_PP

    )


    # ========================================================
    # Keep only configurations satisfying accuracy tolerance
    # ========================================================

    near_optimal = group[

        group["accuracy"]
        >= minimum_allowed_accuracy

    ].copy()


    if near_optimal.empty:

        raise ValueError(

            f"No near-optimal configurations "
            f"were found for {device_id}."

        )


    # ========================================================
    # Raw maximum-accuracy reference configuration
    #
    # This is NOT automatically considered the best hardware
    # design.
    #
    # If multiple configurations have identical maximum
    # accuracy, choose the one with lower architecture cost.
    # ========================================================

    raw_best = (

        group

        .sort_values(

            by=[

                "accuracy",
                "estimated_memristor_cells",
                "relative_hardware_cost_proxy",
                "requested_weight_bits",
                "adc_bits",

            ],

            ascending=[

                False,
                True,
                True,
                True,
                True,

            ]

        )

        .iloc[0]

    )


    # ========================================================
    # COST-AWARE NEAR-OPTIMAL SELECTION
    # ========================================================
    #
    # We deliberately avoid inventing a weighted objective
    # such as:
    #
    # score =
    # accuracy
    # - alpha * area
    # - beta  * energy
    #
    # because we do not yet have calibrated physical area,
    # energy, power, or latency models.
    #
    #
    # Instead, once a configuration satisfies the required
    # accuracy tolerance, we use transparent priorities:
    #
    #
    # PRIORITY 1
    # ----------
    # Fewest physical memristor cells.
    #
    #
    # PRIORITY 2
    # ----------
    # Lowest current relative hardware-cost proxy.
    #
    # This proxy captures:
    #
    #   physical crossbar tile count
    #   ADC precision
    #
    #
    # PRIORITY 3
    # ----------
    # Lowest requested weight precision.
    #
    # This is important particularly for analog devices.
    #
    # Our current cell-count model gives an analog W=4 and
    # analog W=8 design the same number of cells.
    #
    # However, higher weight precision can require greater
    # programming/control precision and should not be treated
    # as automatically free.
    #
    # Until a calibrated programming/peripheral model exists,
    # lower requested precision is preferred when previous
    # hardware-cost quantities tie.
    #
    #
    # PRIORITY 4
    # ----------
    # Lowest ADC precision.
    #
    #
    # PRIORITY 5
    # ----------
    # Highest simulated accuracy.
    #
    #
    # PRIORITY 6
    # ----------
    # Larger crossbar if everything above ties.
    #
    # This last rule only makes the result deterministic.
    # ========================================================

    near_optimal = (

        near_optimal

        .sort_values(

            by=[

                "estimated_memristor_cells",
                "relative_hardware_cost_proxy",
                "requested_weight_bits",
                "adc_bits",
                "accuracy",
                "crossbar_size",

            ],

            ascending=[

                True,
                True,
                True,
                True,
                False,
                False,

            ]

        )

    )


    # ========================================================
    # Chosen near-optimal configuration
    # ========================================================

    chosen = (
        near_optimal.iloc[0].copy()
    )


    # ========================================================
    # Accuracy evaluation information
    # ========================================================

    chosen[
        "best_exhaustive_accuracy"
    ] = best_accuracy


    chosen[
        "near_optimal_threshold"
    ] = minimum_allowed_accuracy


    chosen[
        "regret_pp"
    ] = (

        best_accuracy
        - chosen["accuracy"]

    )


    chosen[
        "candidate_configs"
    ] = len(
        near_optimal
    )


    # ========================================================
    # Raw-best configuration information
    # ========================================================

    chosen[
        "raw_best_crossbar_size"
    ] = raw_best[
        "crossbar_size"
    ]


    chosen[
        "raw_best_weight_bits"
    ] = raw_best[
        "requested_weight_bits"
    ]


    chosen[
        "raw_best_adc_bits"
    ] = raw_best[
        "adc_bits"
    ]


    chosen[
        "raw_best_accuracy"
    ] = raw_best[
        "accuracy"
    ]


    chosen[
        "raw_best_memristor_cells"
    ] = raw_best[
        "estimated_memristor_cells"
    ]


    chosen[
        "raw_best_hardware_cost_proxy"
    ] = raw_best[
        "relative_hardware_cost_proxy"
    ]


    # ========================================================
    # Memristor-cell savings versus raw maximum-accuracy design
    # ========================================================

    raw_cells = float(

        raw_best[
            "estimated_memristor_cells"
        ]

    )


    chosen_cells = float(

        chosen[
            "estimated_memristor_cells"
        ]

    )


    if raw_cells > 0:

        cell_savings_pct = (

            100.0

            * (
                raw_cells
                - chosen_cells
            )

            / raw_cells

        )

    else:

        cell_savings_pct = 0.0


    chosen[
        "memristor_cell_savings_vs_raw_best_pct"
    ] = cell_savings_pct


    # ========================================================
    # Relative proxy savings versus raw-best design
    #
    # IMPORTANT:
    #
    # This is NOT measured:
    #
    #   energy
    #   area
    #   power
    #   latency
    #
    # It is only a relative design-space comparison.
    # ========================================================

    raw_proxy = float(

        raw_best[
            "relative_hardware_cost_proxy"
        ]

    )


    chosen_proxy = float(

        chosen[
            "relative_hardware_cost_proxy"
        ]

    )


    if raw_proxy > 0:

        proxy_savings_pct = (

            100.0

            * (
                raw_proxy
                - chosen_proxy
            )

            / raw_proxy

        )

    else:

        proxy_savings_pct = 0.0


    chosen[
        "relative_cost_proxy_savings_vs_raw_best_pct"
    ] = proxy_savings_pct


    # ========================================================
    # Explicit selection-rule provenance
    # ========================================================

    chosen[
        "selection_rule"
    ] = (

        "WITHIN_0.5PP_"
        "THEN_FEWEST_CELLS_"
        "THEN_LOWEST_COST_PROXY_"
        "THEN_LOWEST_WEIGHT_BITS_"
        "THEN_LOWEST_ADC_BITS"

    )


    selected_rows.append(
        chosen
    )


# ============================================================
# Create final table
# ============================================================

optimal_df = pd.DataFrame(
    selected_rows
)


# ============================================================
# Output columns
# ============================================================

columns = [

    # --------------------------------------------------------
    # Device identity
    # --------------------------------------------------------

    "device_id",
    "technology_family",
    "conductance_mode",
    "mapping_strategy",
    "precision_basis",


    # --------------------------------------------------------
    # Selected accelerator configuration
    # --------------------------------------------------------

    "crossbar_size",
    "requested_weight_bits",
    "effective_weight_levels",
    "physical_cells_per_weight",
    "adc_bits",


    # --------------------------------------------------------
    # Selected performance
    # --------------------------------------------------------

    "accuracy",
    "best_exhaustive_accuracy",
    "near_optimal_threshold",
    "regret_pp",


    # --------------------------------------------------------
    # Selected architecture estimates
    # --------------------------------------------------------

    "estimated_memristor_cells",
    "estimated_physical_crossbar_tiles",
    "relative_hardware_cost_proxy",


    # --------------------------------------------------------
    # Exhaustive raw-best reference
    # --------------------------------------------------------

    "raw_best_crossbar_size",
    "raw_best_weight_bits",
    "raw_best_adc_bits",
    "raw_best_accuracy",
    "raw_best_memristor_cells",
    "raw_best_hardware_cost_proxy",


    # --------------------------------------------------------
    # Hardware savings
    # --------------------------------------------------------

    "memristor_cell_savings_vs_raw_best_pct",
    "relative_cost_proxy_savings_vs_raw_best_pct",


    # --------------------------------------------------------
    # Number of qualifying configurations
    # --------------------------------------------------------

    "candidate_configs",


    # --------------------------------------------------------
    # Selection provenance
    # --------------------------------------------------------

    "selection_rule",

]


optimal_df = optimal_df[
    columns
]


# ============================================================
# Sort final output
# ============================================================

optimal_df = optimal_df.sort_values(
    "device_id"
)


# ============================================================
# Save
# ============================================================

optimal_df.to_csv(

    OUTPUT_FILE,

    index=False

)


# ============================================================
# Display
# ============================================================

print()

print(
    "COST-AWARE NEAR-OPTIMAL CONFIGURATIONS"
)

print(
    "========================================"
)


print()

print(

    "Accuracy requirement: within "
    f"{ACCURACY_TOLERANCE_PP:.1f} percentage points "
    "of the exhaustive best."

)


print()

print(
    "Selection priority:"
)

print(
    "1. Fewest physical memristor cells"
)

print(
    "2. Lowest relative hardware-cost proxy"
)

print(
    "3. Lowest requested weight bits"
)

print(
    "4. Lowest ADC bits"
)

print(
    "5. Highest simulated accuracy"
)


print()

print(

    optimal_df[

        [

            "device_id",

            "crossbar_size",

            "requested_weight_bits",

            "effective_weight_levels",

            "physical_cells_per_weight",

            "adc_bits",

            "accuracy",

            "best_exhaustive_accuracy",

            "regret_pp",

            "estimated_memristor_cells",

            "relative_hardware_cost_proxy",

            "memristor_cell_savings_vs_raw_best_pct",

            "relative_cost_proxy_savings_vs_raw_best_pct",

        ]

    ].to_string(
        index=False
    )

)


print()

print(
    "IMPORTANT:"
)


print(

    "The hardware-cost proxy is a heuristic "
    "design-space metric."

)


print(

    "It must not be presented as measured "
    "energy, power, latency, or silicon area."

)


print()

print(
    "Saved to:",
    OUTPUT_FILE
)