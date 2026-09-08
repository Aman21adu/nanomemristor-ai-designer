import pandas as pd
import numpy as np


def load_device(device_id):

    df = pd.read_csv("data/device_profiles.csv")

    row = df[df["device_id"] == device_id]

    if row.empty:
        raise ValueError(f"Device {device_id} not found")

    row = row.iloc[0]

    ron = row["ron_ohm"]
    roff = row["roff_ohm"]
    ratio = row["on_off_ratio"]

    # --------------------------------------------------
    # Determine conductance range
    # --------------------------------------------------

    # If real RON and ROFF exist
    if pd.notna(ron) and pd.notna(roff):

        source = "absolute"

        gmax = 1.0 / ron
        gmin = 1.0 / roff

    # If only ON/OFF ratio exists,
    # use normalized resistance.
    elif pd.notna(ratio):

        source = "normalized_from_ratio"

        ron = 1.0
        roff = ratio

        gmax = 1.0
        gmin = 1.0 / ratio

    else:

        source = "insufficient"

        gmax = np.nan
        gmin = np.nan

    # --------------------------------------------------
    # Return device profile
    # --------------------------------------------------

    return {
        "device_id": row["device_id"],
        "family": row["technology_family"],
        "stack": row["device_stack"],

        "ron": ron,
        "roff": roff,
        "ratio": ratio,

        "gmin": gmin,
        "gmax": gmax,

        "states": row["conductance_states"],

        "conductance_mode": row["conductance_mode"],
        "state_count_status": row["state_count_status"],
        "state_count_notes": row["state_count_notes"],

        "read_voltage": row["read_voltage_v"],

        "parameter_source": source
    }


if __name__ == "__main__":

    devices = [
        "ZnO_01",
        "HfOx_02",
        "TiOx_01",
        "TaOx_01",
        "TaOx_02"
    ]

    for device_id in devices:

        d = load_device(device_id)

        print("\n-----------------------------")
        print("Device:", d["device_id"])
        print("Family:", d["family"])
        print("Stack:", d["stack"])
        print("RON:", d["ron"])
        print("ROFF:", d["roff"])
        print("Gmin:", d["gmin"])
        print("Gmax:", d["gmax"])
        print("States:", d["states"])
        print(
            "Conductance mode:",
            d["conductance_mode"]
        )
        print(
            "State count status:",
            d["state_count_status"]
        )
        print(
            "Parameter source:",
            d["parameter_source"]
        )