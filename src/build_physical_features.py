from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# Files
# ============================================================

DEVICE_FILE = (
    "data/device_profiles.csv"
)

TRACE_FILE = (
    "data/source_traceability.csv"
)

OUTPUT_FILE = (
    "results/tables/"
    "device_physical_features.csv"
)


# ============================================================
# Load data
# ============================================================

devices = pd.read_csv(
    DEVICE_FILE
)

trace = pd.read_csv(
    TRACE_FILE
)


# ============================================================
# Required columns
# ============================================================

DEVICE_REQUIRED_COLUMNS = [

    "device_id",
    "technology_family",
    "device_stack",
    "active_material",

]


TRACE_REQUIRED_COLUMNS = [

    "device_id",
    "property_name",
    "value",
    "unit",
    "value_type",

    "source_title",
    "doi",
    "page",
    "figure_or_table",
    "source_note",

]


missing_device_columns = [

    column

    for column
    in DEVICE_REQUIRED_COLUMNS

    if column
    not in devices.columns

]


missing_trace_columns = [

    column

    for column
    in TRACE_REQUIRED_COLUMNS

    if column
    not in trace.columns

]


if missing_device_columns:

    raise ValueError(

        "device_profiles.csv is missing columns:\n"
        f"{missing_device_columns}"

    )


if missing_trace_columns:

    raise ValueError(

        "source_traceability.csv is missing columns:\n"
        f"{missing_trace_columns}"

    )


# ============================================================
# Helpers
# ============================================================

def clean_text(value):

    if pd.isna(value):

        return ""

    return str(
        value
    ).strip()


# ============================================================
# Unit conversion
# ============================================================

def unit_to_nm_multiplier(
    unit
):

    unit = (
        clean_text(
            unit
        )
        .lower()
        .replace("μ", "u")
        .replace("µ", "u")
    )


    if unit == "nm":

        return 1.0


    if unit in {

        "um",
        "µm",
        "μm",

    }:

        return 1000.0


    raise ValueError(

        f"Unsupported thickness unit: "
        f"{unit}"

    )


# ============================================================
# Numeric parser
#
# Supported literature formats:
#
#   5.5
#   100
#   10-18
#   10–18
#   24.1 ± 0.27
#   29 ± 5%
#
#
# IMPORTANT:
#
# A reported range such as 10-18 does NOT receive
# an invented midpoint.
# ============================================================

def parse_physical_value(
    raw_value,
    raw_unit,
):

    text = clean_text(
        raw_value
    )


    if not text:

        return {

            "representation":
                "MISSING",

            "nominal":
                np.nan,

            "lower":
                np.nan,

            "upper":
                np.nan,

            "uncertainty":
                np.nan,

            "uncertainty_type":
                "",

        }


    # --------------------------------------------------------
    # Normalize dash characters
    # --------------------------------------------------------

    normalized = (

        text

        .replace("–", "-")
        .replace("—", "-")

    )


    multiplier = unit_to_nm_multiplier(
        raw_unit
    )


    # ========================================================
    # Pattern 1:
    # nominal ± percent
    #
    # Example:
    #
    # 29 ± 5%
    # ========================================================

    percent_match = re.fullmatch(

        r"\s*"
        r"([-+]?\d*\.?\d+)"
        r"\s*±\s*"
        r"(\d*\.?\d+)"
        r"\s*%\s*",

        normalized,

    )


    if percent_match:

        nominal_original = float(
            percent_match.group(1)
        )


        percent = float(
            percent_match.group(2)
        )


        nominal_nm = (

            nominal_original
            * multiplier

        )


        delta_nm = (

            nominal_nm
            * percent
            / 100.0

        )


        return {

            "representation":
                "NOMINAL_PLUS_MINUS_PERCENT",

            "nominal":
                nominal_nm,

            "lower":
                nominal_nm - delta_nm,

            "upper":
                nominal_nm + delta_nm,

            "uncertainty":
                percent,

            "uncertainty_type":
                "PERCENT",

        }


    # ========================================================
    # Pattern 2:
    # nominal ± absolute uncertainty
    #
    # Example:
    #
    # 24.1 ± 0.27
    # ========================================================

    absolute_uncertainty_match = re.fullmatch(

        r"\s*"
        r"([-+]?\d*\.?\d+)"
        r"\s*±\s*"
        r"(\d*\.?\d+)"
        r"\s*",

        normalized,

    )


    if absolute_uncertainty_match:

        nominal_original = float(

            absolute_uncertainty_match
            .group(1)

        )


        uncertainty_original = float(

            absolute_uncertainty_match
            .group(2)

        )


        nominal_nm = (

            nominal_original
            * multiplier

        )


        uncertainty_nm = (

            uncertainty_original
            * multiplier

        )


        return {

            "representation":
                "NOMINAL_PLUS_MINUS_ABSOLUTE",

            "nominal":
                nominal_nm,

            "lower":
                nominal_nm
                - uncertainty_nm,

            "upper":
                nominal_nm
                + uncertainty_nm,

            "uncertainty":
                uncertainty_nm,

            "uncertainty_type":
                "ABSOLUTE_NM",

        }


    # ========================================================
    # Pattern 3:
    # reported interval / range
    #
    # Example:
    #
    # 10-18
    #
    # IMPORTANT:
    #
    # nominal remains NaN.
    #
    # We do NOT silently replace the interval by 14 nm.
    # ========================================================

    range_match = re.fullmatch(

        r"\s*"
        r"([-+]?\d*\.?\d+)"
        r"\s*-\s*"
        r"([-+]?\d*\.?\d+)"
        r"\s*",

        normalized,

    )


    if range_match:

        lower_original = float(
            range_match.group(1)
        )


        upper_original = float(
            range_match.group(2)
        )


        lower_nm = (

            lower_original
            * multiplier

        )


        upper_nm = (

            upper_original
            * multiplier

        )


        if lower_nm > upper_nm:

            raise ValueError(

                f"Invalid physical range: "
                f"{raw_value} {raw_unit}"

            )


        return {

            "representation":
                "REPORTED_RANGE",

            "nominal":
                np.nan,

            "lower":
                lower_nm,

            "upper":
                upper_nm,

            "uncertainty":
                np.nan,

            "uncertainty_type":
                "RANGE",

        }


    # ========================================================
    # Pattern 4:
    # single reported numeric value
    # ========================================================

    single_match = re.fullmatch(

        r"\s*"
        r"([-+]?\d*\.?\d+)"
        r"\s*",

        normalized,

    )


    if single_match:

        value_original = float(
            single_match.group(1)
        )


        value_nm = (

            value_original
            * multiplier

        )


        return {

            "representation":
                "SINGLE_VALUE",

            "nominal":
                value_nm,

            "lower":
                value_nm,

            "upper":
                value_nm,

            "uncertainty":
                0.0,

            "uncertainty_type":
                "NONE",

        }


    raise ValueError(

        "Could not parse physical value: "
        f"'{raw_value}' {raw_unit}"

    )


# ============================================================
# Find exactly one traceability row from a set of
# acceptable property names
# ============================================================

def get_property_row(
    device_id,
    property_names,
):

    matches = trace[

        (
            trace[
                "device_id"
            ]
            ==
            device_id
        )

        &

        (
            trace[
                "property_name"
            ].isin(
                property_names
            )
        )

    ].copy()


    if matches.empty:

        return None


    if len(matches) > 1:

        raise ValueError(

            f"{device_id} has multiple matching rows "
            f"for {property_names}:\n"
            f"{matches[['property_name', 'value']]}"

        )


    return (
        matches.iloc[0]
    )


# ============================================================
# Build one standardized row per device
# ============================================================

physical_rows = []


for _, device in devices.iterrows():

    device_id = clean_text(

        device[
            "device_id"
        ]

    )


    # ========================================================
    # Active / switching-layer thickness
    #
    # Current source vocabulary contains:
    #
    #   thickness_nm
    #   film_thickness
    #
    # We retain which source property was used.
    # ========================================================

    thickness_row = get_property_row(

        device_id,

        [

            "thickness_nm",
            "film_thickness",

        ],

    )


    if thickness_row is None:

        thickness_source_property = ""

        thickness_original_value = ""

        thickness_original_unit = ""

        thickness_value_type = ""

        thickness_representation = "MISSING"

        thickness_nominal_nm = np.nan

        thickness_lower_nm = np.nan

        thickness_upper_nm = np.nan

        thickness_uncertainty = np.nan

        thickness_uncertainty_type = ""

        thickness_source_title = ""

        thickness_doi = ""

        thickness_page = ""

        thickness_figure_or_table = ""

        thickness_source_note = ""


    else:

        parsed_thickness = parse_physical_value(

            thickness_row[
                "value"
            ],

            thickness_row[
                "unit"
            ],

        )


        thickness_source_property = clean_text(

            thickness_row[
                "property_name"
            ]

        )


        thickness_original_value = clean_text(

            thickness_row[
                "value"
            ]

        )


        thickness_original_unit = clean_text(

            thickness_row[
                "unit"
            ]

        )


        thickness_value_type = clean_text(

            thickness_row[
                "value_type"
            ]

        ).upper()


        thickness_representation = (

            parsed_thickness[
                "representation"
            ]

        )


        thickness_nominal_nm = (

            parsed_thickness[
                "nominal"
            ]

        )


        thickness_lower_nm = (

            parsed_thickness[
                "lower"
            ]

        )


        thickness_upper_nm = (

            parsed_thickness[
                "upper"
            ]

        )


        thickness_uncertainty = (

            parsed_thickness[
                "uncertainty"
            ]

        )


        thickness_uncertainty_type = (

            parsed_thickness[
                "uncertainty_type"
            ]

        )


        thickness_source_title = clean_text(

            thickness_row[
                "source_title"
            ]

        )


        thickness_doi = clean_text(

            thickness_row[
                "doi"
            ]

        )


        thickness_page = clean_text(

            thickness_row[
                "page"
            ]

        )


        thickness_figure_or_table = clean_text(

            thickness_row[
                "figure_or_table"
            ]

        )


        thickness_source_note = clean_text(

            thickness_row[
                "source_note"
            ]

        )


    # ========================================================
    # Crystallite size
    #
    # This is deliberately kept SEPARATE from layer thickness.
    #
    # Example:
    #
    # ZnO_01:
    #
    #   film thickness = 34 um = 34,000 nm
    #
    #   crystallite size = 62 nm
    #
    # They describe different physical quantities.
    # ========================================================

    crystallite_row = get_property_row(

        device_id,

        [
            "crystallite_size"
        ],

    )


    if crystallite_row is None:

        crystallite_size_nm = np.nan

        crystallite_value_type = ""

        crystallite_source_title = ""

        crystallite_doi = ""

        crystallite_page = ""

        crystallite_figure_or_table = ""

        crystallite_source_note = ""


    else:

        parsed_crystallite = parse_physical_value(

            crystallite_row[
                "value"
            ],

            crystallite_row[
                "unit"
            ],

        )


        crystallite_size_nm = (

            parsed_crystallite[
                "nominal"
            ]

        )


        crystallite_value_type = clean_text(

            crystallite_row[
                "value_type"
            ]

        ).upper()


        crystallite_source_title = clean_text(

            crystallite_row[
                "source_title"
            ]

        )


        crystallite_doi = clean_text(

            crystallite_row[
                "doi"
            ]

        )


        crystallite_page = clean_text(

            crystallite_row[
                "page"
            ]

        )


        crystallite_figure_or_table = clean_text(

            crystallite_row[
                "figure_or_table"
            ]

        )


        crystallite_source_note = clean_text(

            crystallite_row[
                "source_note"
            ]

        )


    # ========================================================
    # Does thickness have one directly usable nominal value?
    #
    # TRUE:
    #
    #   5.5 nm
    #   24.1 ± 0.27 nm
    #   29 ± 5%
    #
    # FALSE:
    #
    #   10-18 nm
    #
    # because choosing a midpoint would be an additional
    # modeling assumption.
    # ========================================================

    thickness_nominal_available = bool(

        pd.notna(
            thickness_nominal_nm
        )

    )


    # ========================================================
    # Store row
    # ========================================================

    physical_rows.append({

        # ----------------------------------------------------
        # Device identity
        # ----------------------------------------------------

        "device_id":
            device_id,

        "technology_family":
            clean_text(
                device[
                    "technology_family"
                ]
            ),

        "active_material":
            clean_text(
                device[
                    "active_material"
                ]
            ),

        "device_stack":
            clean_text(
                device[
                    "device_stack"
                ]
            ),


        # ----------------------------------------------------
        # Standardized layer-thickness feature
        # ----------------------------------------------------

        "thickness_source_property":
            thickness_source_property,

        "thickness_original_value":
            thickness_original_value,

        "thickness_original_unit":
            thickness_original_unit,

        "thickness_representation":
            thickness_representation,

        "thickness_nominal_available":
            thickness_nominal_available,

        "active_layer_thickness_nominal_nm":
            thickness_nominal_nm,

        "active_layer_thickness_lower_nm":
            thickness_lower_nm,

        "active_layer_thickness_upper_nm":
            thickness_upper_nm,

        "thickness_uncertainty":
            thickness_uncertainty,

        "thickness_uncertainty_type":
            thickness_uncertainty_type,

        "thickness_value_type":
            thickness_value_type,


        # ----------------------------------------------------
        # Layer-thickness provenance
        # ----------------------------------------------------

        "thickness_source_title":
            thickness_source_title,

        "thickness_doi":
            thickness_doi,

        "thickness_page":
            thickness_page,

        "thickness_figure_or_table":
            thickness_figure_or_table,

        "thickness_source_note":
            thickness_source_note,


        # ----------------------------------------------------
        # Separate nanoscale structural descriptor
        # ----------------------------------------------------

        "crystallite_size_nm":
            crystallite_size_nm,

        "crystallite_value_type":
            crystallite_value_type,


        # ----------------------------------------------------
        # Crystallite provenance
        # ----------------------------------------------------

        "crystallite_source_title":
            crystallite_source_title,

        "crystallite_doi":
            crystallite_doi,

        "crystallite_page":
            crystallite_page,

        "crystallite_figure_or_table":
            crystallite_figure_or_table,

        "crystallite_source_note":
            crystallite_source_note,

    })


# ============================================================
# Create dataframe
# ============================================================

physical = pd.DataFrame(
    physical_rows
)


# ============================================================
# Validation
# ============================================================

if physical[
    "device_id"
].duplicated().any():

    raise ValueError(

        "Duplicate device rows found in "
        "physical-feature table."

    )


# ============================================================
# Ensure every database device is represented
# ============================================================

expected_devices = set(

    devices[
        "device_id"
    ]

)


output_devices = set(

    physical[
        "device_id"
    ]

)


if expected_devices != output_devices:

    raise ValueError(

        "Physical feature table does not contain "
        "the same device set as device_profiles.csv."

    )


# ============================================================
# Save
# ============================================================

Path(
    "results/tables"
).mkdir(

    parents=True,

    exist_ok=True,

)


physical.to_csv(

    OUTPUT_FILE,

    index=False,

)


# ============================================================
# Display standardized thickness table
# ============================================================

print()

print(
    "STANDARDIZED PHYSICAL DEVICE FEATURES"
)

print(
    "========================================"
)


print(
    "Devices:",
    len(
        physical
    )
)


print()

print(
    "ACTIVE / SWITCHING-LAYER THICKNESS"
)

print(
    "----------------------------------------"
)


display_columns = [

    "device_id",

    "technology_family",

    "thickness_source_property",

    "thickness_original_value",

    "thickness_original_unit",

    "thickness_representation",

    "active_layer_thickness_nominal_nm",

    "active_layer_thickness_lower_nm",

    "active_layer_thickness_upper_nm",

    "thickness_value_type",

]


print(

    physical[

        display_columns

    ]

    .sort_values(
        "device_id"
    )

    .to_string(
        index=False
    )

)


# ============================================================
# Range warning
# ============================================================

range_rows = physical[

    physical[
        "thickness_representation"
    ]
    ==
    "REPORTED_RANGE"

]


print()

print(
    "REPORTED THICKNESS RANGES"
)

print(
    "----------------------------------------"
)


if range_rows.empty:

    print(
        "None."
    )


else:

    print(

        range_rows[

            [
                "device_id",
                "thickness_original_value",
                "thickness_original_unit",
                "active_layer_thickness_lower_nm",
                "active_layer_thickness_upper_nm",
            ]

        ].to_string(
            index=False
        )

    )


    print()

    print(

        "No midpoint was invented for these rows."

    )


# ============================================================
# Unit conversion check
# ============================================================

converted_rows = physical[

    physical[
        "thickness_original_unit"
    ]
    .str.lower()
    .isin(

        [
            "um",
            "µm",
            "μm",
        ]

    )

]


print()

print(
    "MICROMETER-TO-NANOMETER CONVERSIONS"
)

print(
    "----------------------------------------"
)


if converted_rows.empty:

    print(
        "None."
    )


else:

    print(

        converted_rows[

            [
                "device_id",
                "thickness_original_value",
                "thickness_original_unit",
                "active_layer_thickness_nominal_nm",
            ]

        ].to_string(
            index=False
        )

    )


# ============================================================
# Separate crystallite information
# ============================================================

print()

print(
    "CRYSTALLITE SIZE"
)

print(
    "----------------------------------------"
)


crystallite_rows = physical[

    physical[
        "crystallite_size_nm"
    ].notna()

]


if crystallite_rows.empty:

    print(
        "No crystallite-size data available."
    )


else:

    print(

        crystallite_rows[

            [
                "device_id",
                "crystallite_size_nm",
                "crystallite_value_type",
            ]

        ].to_string(
            index=False
        )

    )


# ============================================================
# ML-readiness summary
# ============================================================

print()

print(
    "THICKNESS DATA READINESS"
)

print(
    "----------------------------------------"
)


single_nominal_count = int(

    physical[
        "thickness_nominal_available"
    ].sum()

)


range_count = int(

    (
        physical[
            "thickness_representation"
        ]
        ==
        "REPORTED_RANGE"
    ).sum()

)


missing_count = int(

    (
        physical[
            "thickness_representation"
        ]
        ==
        "MISSING"
    ).sum()

)


print(

    "Devices with usable reported nominal thickness:",
    single_nominal_count

)


print(

    "Devices represented only by a reported range:",
    range_count

)


print(

    "Devices with missing thickness evidence:",
    missing_count

)


# ============================================================
# Scientific caution
# ============================================================

print()

print(
    "IMPORTANT"
)

print(
    "----------------------------------------"
)


print(

    "Layer thickness and crystallite size are "
    "different physical quantities and are kept separate."

)


print()

print(

    "A reported interval is stored as lower/upper bounds. "
    "No unsupported midpoint is created."

)


print()

print(

    "This table standardizes literature-derived physical "
    "device descriptors. It does not yet mean thickness "
    "should automatically be used as an ML feature."

)


print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)