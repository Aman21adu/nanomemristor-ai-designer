from itertools import product


# ============================================================
# V3 CONFIGURATION SPACE
# ============================================================
#
# Accelerator design parameters:
#
#   Crossbar size:
#       16, 32, 64, 128, 256
#
#   Requested neural-weight precision:
#       2 through 8 bits
#
#   ADC precision:
#       4 through 10 bits
#
#
# Total:
#
#   5 crossbar sizes
# x 7 weight precisions
# x 7 ADC precisions
#
# = 245 configurations per device
#
#
# IMPORTANT:
#
# These are CONFIGURATIONS, not 245 independent parameters.
#
# The simulator still varies three accelerator-design
# parameters.
# ============================================================


CROSSBAR_SIZES = [
    16,
    32,
    64,
    128,
    256,
]


WEIGHT_BITS = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
]


ADC_BITS = [
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]


# ============================================================
# Generate complete configuration grid
# ============================================================

def generate_configs():

    configs = []


    for (
        crossbar,
        weight_bits,
        adc_bits,

    ) in product(

        CROSSBAR_SIZES,
        WEIGHT_BITS,
        ADC_BITS,

    ):

        configs.append({

            "crossbar_size":
                crossbar,

            "weight_bits":
                weight_bits,

            "adc_bits":
                adc_bits,

        })


    return configs


# ============================================================
# Standalone verification
# ============================================================

if __name__ == "__main__":

    configs = generate_configs()


    print(
        "Crossbar choices:",
        len(
            CROSSBAR_SIZES
        )
    )


    print(
        "Weight-bit choices:",
        len(
            WEIGHT_BITS
        )
    )


    print(
        "ADC-bit choices:",
        len(
            ADC_BITS
        )
    )


    print()


    print(
        "Total configurations:",
        len(
            configs
        )
    )


    print(
        "Expected total:",
        (
            len(
                CROSSBAR_SIZES
            )
            *
            len(
                WEIGHT_BITS
            )
            *
            len(
                ADC_BITS
            )
        )
    )


    print()


    print(
        "First configuration:",
        configs[0]
    )


    print(
        "Last configuration:",
        configs[-1]
    )