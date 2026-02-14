import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def gaussian_with_offset(x: np.ndarray, amplitude: float, mu: float, sigma: float, baseline: float) -> np.ndarray:
    return baseline + amplitude * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


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

# [ Initial guess ]
# ==================================================================================================
baseline0 = float(np.nanmin(y))
amplitude0 = float(np.nanmax(y) - np.nanmin(y))
mu0 = float(x[np.nanargmax(y)])
sigma0 = 50.0
p0 = [amplitude0, mu0, sigma0, baseline0]

# [ Fitting ]
# ==================================================================================================
bounds = (
    [0.0, -250.0, 1e-6, 0.0],   # amplitude, mu, sigma, baseline lower bounds
    [1.0, 250.0, 500.0, 1.0],   # upper bounds
)
params, cov = curve_fit(
    gaussian_with_offset,
    x,
    y,
    p0=p0,
    bounds=bounds,
    maxfev=20000,
)
amplitude, mu, sigma, baseline = [float(v) for v in params]
y_fit = gaussian_with_offset(x, amplitude, mu, sigma, baseline)

# [ Save fit parameters ]
# ==================================================================================================
result = {
    "amplitude": amplitude,
    "mu": mu,
    "sigma": sigma,
    "baseline": baseline,
}
with open(OUTPUT_DIR / "gaussian_fit_params.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

fit_output_df = pd.DataFrame(
    {
        "time_s": x,
        "mean_E_norm_ma": y,
        "gaussian_fit": y_fit,
    }
)
fit_output_df.to_csv(OUTPUT_DIR / "gaussian_fit_timeseries.csv", index=False)

# [ Plot ]
# ==================================================================================================
fig, ax = plt.subplots(layout="constrained")
ax.plot(x, y, color="blue", linewidth=2, label="Moving average (from Step4)")
ax.plot(x, y_fit, color="red", linewidth=2, linestyle="--", label="Gaussian fit")
ax.set_xlabel("Time (s)", fontsize=18)
ax.set_ylabel("Mean E_norm", fontsize=18)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

text = f"A={amplitude:.4f}, mu={mu:.2f}, sigma={sigma:.2f}, base={baseline:.4f}"
ax.text(0.02, 0.03, text, transform=ax.transAxes, fontsize=11)

fig.savefig(OUTPUT_DIR / "gaussian_fit_from_step4_ma.png", dpi=300)
plt.close(fig)

print("[Info] Gaussian fitting completed.")
print(f"[Info] amplitude={amplitude:.6f}, mu={mu:.6f}, sigma={sigma:.6f}, baseline={baseline:.6f}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_from_step4_ma.png'}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_params.json'}")
print(f"[Info] Saved: {OUTPUT_DIR / 'gaussian_fit_timeseries.csv'}")