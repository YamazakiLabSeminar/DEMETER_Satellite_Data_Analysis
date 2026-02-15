import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

FIXED_BASELINE = 0.48


def gaussian_with_fixed_baseline(x: np.ndarray, amplitude: float, mu: float, sigma: float) -> np.ndarray:
    # Inverted Gaussian (downward dip) with fixed baseline at y=0.48
    return FIXED_BASELINE - amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# [ Path Settings ]
# ==================================================================================================
DATA_PATH = Path(r"E:\figures\sea_analysis_results_with_moving_average\t_mean_with_moving_average.csv")
OUTPUT_DIR = Path(r"E:\figures\gaussian_fit_results")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"File not found: {DATA_PATH}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# [ Load ]
# ==================================================================================================
df = pd.read_csv(DATA_PATH)
required_cols = {"time_s", "mean_E_norm_ma"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Required columns are missing. Need: {required_cols}, got: {set(df.columns)}")

fit_df = df[["time_s", "mean_E_norm_ma"]].dropna().copy()
x = fit_df["time_s"].to_numpy(dtype=float)
y = fit_df["mean_E_norm_ma"].to_numpy(dtype=float)
if len(x) < 5:
    raise ValueError("Not enough data points for fitting.")

FIT_RANGE_S = 120.0
fit_mask = np.abs(x) <= FIT_RANGE_S
x_fit = x[fit_mask]
y_fit_target = y[fit_mask]
if len(x_fit) < 5:
    raise ValueError("Not enough data points inside fit range.")

# [ Initial guess ]
# ==================================================================================================
amplitude0 = float(np.nanmax(y_fit_target) - np.nanmin(y_fit_target))
mu0 = float(x_fit[np.nanargmin(y_fit_target)])
sigma0 = 40.0
p0 = [amplitude0, mu0, sigma0]

# [ Fitting ]
# ==================================================================================================
bounds = (
    [0.0, -FIT_RANGE_S, 5.0],     # amplitude, mu, sigma
    [1.0, FIT_RANGE_S, 200.0],
)
params, cov = curve_fit(
    gaussian_with_fixed_baseline,
    x_fit,
    y_fit_target,
    p0=p0,
    bounds=bounds,
    maxfev=20000,
)
amplitude, mu, sigma = [float(v) for v in params]
y_fit = gaussian_with_fixed_baseline(x, amplitude, mu, sigma)
baseline_line = np.full_like(x, FIXED_BASELINE)

# [ Save fit parameters ]
# ==================================================================================================
result = {
    "amplitude": amplitude,
    "mu": mu,
    "sigma": sigma,
    "baseline": FIXED_BASELINE,
    "fit_range_s": FIT_RANGE_S,
}
with open(OUTPUT_DIR / "gaussian_fit_params.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

fit_output_df = pd.DataFrame(
    {
        "time_s": x,
        "mean_E_norm_ma": y,
        "gaussian_fit": y_fit,
        "baseline_line": baseline_line,
    }
)
fit_output_df.to_csv(OUTPUT_DIR / "gaussian_fit_timeseries.csv", index=False)

# [ Plot ]
# ==================================================================================================
fig, ax = plt.subplots(layout="constrained")
ax.plot(x, y, color="blue", linewidth=2, label="Moving average (from Step4)")
ax.plot(x, y_fit, color="red", linewidth=2, linestyle="--", label="Gaussian fit")
ax.plot(x, baseline_line, color="black", linewidth=1.2, linestyle=":", label="Baseline")
ax.axvspan(-FIT_RANGE_S, FIT_RANGE_S, color="gray", alpha=0.08, label="Fit range")
ax.set_xlabel("Time (s)", fontsize=18)
ax.set_ylabel("Mean E_norm", fontsize=18)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

text = f"A={amplitude:.4f}, mu={mu:.2f}, sigma={sigma:.2f}, base={FIXED_BASELINE:.4f}"
ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=11)

fig.savefig(OUTPUT_DIR / "gaussian_fit_from_step4_ma.png", dpi=300)
plt.close(fig)

print("[Info] Gaussian fitting completed.")
print(f"[Info] amplitude={amplitude:.6f}, mu={mu:.6f}, sigma={sigma:.6f}, baseline={FIXED_BASELINE:.6f}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_from_step4_ma.png'}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_params.json'}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_timeseries.csv'}")
