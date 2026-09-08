from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

SUMMARY_FILE = "results/tables/zero_shot_summary.csv"
PREDICTIONS_FILE = "results/tables/zero_shot_predictions.csv"
ML_DATASET_FILE = "results/tables/ml_dataset.csv"

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

summary = pd.read_csv(SUMMARY_FILE)
predictions = pd.read_csv(PREDICTIONS_FILE)
ml = pd.read_csv(ML_DATASET_FILE)


summary["label"] = (
    summary["device_id"]
    + "\n("
    + summary["held_out_family"]
    + ")"
)


# ============================================================
# FIGURE 1
# Exhaustive best vs zero-shot recommendation
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(summary))
width = 0.36

ax.bar(
    x - width / 2,
    summary["true_best_accuracy"],
    width,
    label="Exhaustive best"
)

ax.bar(
    x + width / 2,
    summary["actual_accuracy"],
    width,
    label="Zero-shot recommended"
)

ax.set_xticks(x)
ax.set_xticklabels(summary["label"])

ax.set_ylabel("MNIST Accuracy (%)")

ax.set_title(
    "Exhaustive Best vs Zero-Shot Recommended Configuration"
)

ax.legend()
ax.grid(axis="y", alpha=0.25)

for i, row in summary.iterrows():

    ax.text(
        i - width / 2,
        row["true_best_accuracy"] + 0.4,
        f"{row['true_best_accuracy']:.2f}",
        ha="center",
        fontsize=8
    )

    ax.text(
        i + width / 2,
        row["actual_accuracy"] + 0.4,
        f"{row['actual_accuracy']:.2f}",
        ha="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure1_exhaustive_vs_zero_shot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 2
# Recommendation regret
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.bar(
    summary["label"],
    summary["regret_pp"]
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="Near-optimal threshold = 0.5 pp"
)

ax.set_ylabel("Regret (percentage points)")
ax.set_title("Zero-Shot Recommendation Regret")

ax.legend()
ax.grid(axis="y", alpha=0.25)

for i, value in enumerate(summary["regret_pp"]):

    ax.text(
        i,
        value + 0.015,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure2_zero_shot_regret.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3
# Predicted vs actual performance
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

for family, group in predictions.groupby("held_out_family"):

    ax.scatter(
        group["accuracy"],
        group["predicted_accuracy"],
        alpha=0.65,
        label=family
    )


minimum = min(
    predictions["accuracy"].min(),
    predictions["predicted_accuracy"].min()
)

maximum = max(
    predictions["accuracy"].max(),
    predictions["predicted_accuracy"].max()
)

margin = (maximum - minimum) * 0.03

minimum -= margin
maximum += margin

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=1.5,
    label="Ideal prediction"
)

ax.set_xlim(minimum, maximum)
ax.set_ylim(minimum, maximum)

ax.set_xlabel("Actual Simulated Accuracy (%)")
ax.set_ylabel("Predicted Accuracy (%)")

ax.set_title(
    "Zero-Shot Predicted vs Actual Configuration Performance"
)

ax.legend()
ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure3_predicted_vs_actual.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4
# Selected accelerator configurations
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6.5))

levels = summary["recommended_effective_levels"]

bars = ax.bar(
    summary["label"],
    levels
)

ax.set_ylabel("Effective Signed Weight Levels")

ax.set_title(
    "Zero-Shot Selected Accelerator Configurations",
    pad=20
)

ax.grid(axis="y", alpha=0.25)

ax.set_ylim(
    0,
    max(levels) * 1.30
)

for i, bar in enumerate(bars):

    row = summary.iloc[i]

    config = (
        f"{int(row['recommended_crossbar'])}×"
        f"{int(row['recommended_crossbar'])}\n"
        f"W{int(row['recommended_weight_bits'])} / "
        f"ADC{int(row['recommended_adc_bits'])}"
    )

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.4,
        config,
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure4_selected_configurations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5
# ON/OFF ratio alone does NOT determine performance
# ============================================================

device_features = (
    ml[
        [
            "device_id",
            "technology_family",
            "on_off_ratio"
        ]
    ]
    .drop_duplicates()
    .rename(
        columns={
            "technology_family": "held_out_family"
        }
    )
)


device_summary = summary.merge(
    device_features,
    on=[
        "device_id",
        "held_out_family"
    ],
    how="left"
)


fig, ax = plt.subplots(figsize=(9, 6.5))

ax.scatter(
    device_summary["on_off_ratio"],
    device_summary["actual_accuracy"],
    s=100
)


for _, row in device_summary.iterrows():

    label = (
        f"{row['device_id']}\n"
        f"{int(row['recommended_effective_levels'])} levels"
    )

    # High-accuracy points:
    # move labels downward so they do not overlap title
    if row["actual_accuracy"] > 80:

        offset = (0, -16)
        vertical_alignment = "top"

    else:

        offset = (7, 7)
        vertical_alignment = "bottom"

    ax.annotate(
        label,
        (
            row["on_off_ratio"],
            row["actual_accuracy"]
        ),
        xytext=offset,
        textcoords="offset points",
        fontsize=9,
        ha="center",
        va=vertical_alignment
    )


ax.set_xscale("log")

ax.set_xlabel(
    "Device ON/OFF Ratio (log scale)"
)

ax.set_ylabel(
    "Accuracy of Recommended Configuration (%)"
)

ax.set_title(
    "ON/OFF Ratio Alone Does Not Determine AI Accuracy",
    pad=16
)

ax.set_ylim(
    5,
    102
)

ax.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "figure5_device_properties_vs_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6
# Overall project result
# ============================================================

mean_regret = summary["regret_pp"].mean()

near_optimal_rate = (
    100
    * summary["near_optimal_success"].mean()
)

search_reduction = (
    summary["search_reduction_pct"].mean()
)


fig, ax = plt.subplots(figsize=(7.5, 5.5))

names = [
    "Near-optimal\nsuccess",
    "Exhaustive-search\nreduction"
]

values = [
    near_optimal_rate,
    search_reduction
]


bars = ax.bar(
    names,
    values
)

ax.set_ylim(0, 108)

ax.set_ylabel("Percentage (%)")

ax.set_title(
    "Overall Zero-Shot Design-Space Search Performance\n"
    f"Mean recommendation regret = {mean_regret:.3f} pp",
    pad=14
)

ax.grid(axis="y", alpha=0.25)


for bar, value in zip(bars, values):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        fontsize=11
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "figure6_overall_zero_shot_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 7
# Prediction error of recommended configurations
# ============================================================

summary["absolute_prediction_error"] = (
    summary["predicted_accuracy"]
    - summary["actual_accuracy"]
).abs()


fig, ax = plt.subplots(figsize=(9, 5.5))

bars = ax.bar(
    summary["label"],
    summary["absolute_prediction_error"]
)

ax.set_ylabel(
    "Absolute Prediction Error (percentage points)"
)

ax.set_title(
    "Prediction Error of Recommended Configurations"
)

ax.grid(axis="y", alpha=0.25)


for i, value in enumerate(
    summary["absolute_prediction_error"]
):

    ax.text(
        i,
        value + 0.01,
        f"{value:.2f}",
        ha="center",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "figure7_recommendation_prediction_error.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

recommended_mae = (
    summary[
        "absolute_prediction_error"
    ].mean()
)


print("\nFINAL FIGURES UPDATED")
print("=====================================")

print("1. figure1_exhaustive_vs_zero_shot.png")
print("2. figure2_zero_shot_regret.png")
print("3. figure3_predicted_vs_actual.png")
print("4. figure4_selected_configurations.png")
print("5. figure5_device_properties_vs_accuracy.png")
print("6. figure6_overall_zero_shot_summary.png")
print("7. figure7_recommendation_prediction_error.png")


print("\nFINAL HEADLINE RESULTS")
print("-------------------------------------")

print(
    f"Mean regret: "
    f"{mean_regret:.3f} pp"
)

print(
    f"Near-optimal success: "
    f"{near_optimal_rate:.1f}%"
)

print(
    f"Exhaustive-search reduction: "
    f"{search_reduction:.1f}%"
)

print(
    f"Recommended-config prediction MAE: "
    f"{recommended_mae:.3f} pp"
)