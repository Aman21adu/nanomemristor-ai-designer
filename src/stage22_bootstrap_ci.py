from pathlib import Path
import json
import numpy as np
import pandas as pd

INPUT = Path("results/tables/stage21_study_group_summary.csv")
OUT = Path("results/tables/stage22_study_bootstrap_ci.csv")
SCOPE = Path("results/tables/stage22_bootstrap_scope.json")

N_BOOT = 20000
SEED = 42

df = pd.read_csv(INPUT)

metrics = [
    "mean_global_mae_pp",
    "mean_baseline_regret_pp",
    "mean_guarded_regret_pp",
]

rng = np.random.default_rng(SEED)

rows = []

for metric in metrics:
    values = df[metric].astype(float).to_numpy()

    n = len(values)

    observed = float(np.mean(values))

    samples = rng.choice(
        values,
        size=(N_BOOT, n),
        replace=True,
    )

    boot_means = samples.mean(axis=1)

    ci_low = float(
        np.percentile(boot_means, 2.5)
    )

    ci_high = float(
        np.percentile(boot_means, 97.5)
    )

    rows.append(
        {
            "metric": metric,
            "independent_groups": n,
            "observed_mean": observed,
            "bootstrap_sd": float(
                np.std(boot_means, ddof=1)
            ),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_resamples": N_BOOT,
        }
    )

result = pd.DataFrame(rows)

result.to_csv(
    OUT,
    index=False,
)

scope = {
    "stage": 22,
    "name": "Independent-study bootstrap confidence intervals",
    "sampling_unit": "independent held-out study",
    "independent_studies": int(len(df)),
    "bootstrap_resamples": N_BOOT,
    "random_seed": SEED,
    "ci_method": "percentile bootstrap",
    "important_note": (
        "Architecture configurations within a device are not "
        "treated as independent statistical samples. "
        "Bootstrap resampling is performed over the eight "
        "independent held-out study groups."
    ),
}

SCOPE.write_text(
    json.dumps(scope, indent=2),
    encoding="utf-8",
)

print("=" * 70)
print("STAGE 22 — STUDY-LEVEL BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)
print(result.to_string(index=False))
print()
print("Saved:", OUT)
