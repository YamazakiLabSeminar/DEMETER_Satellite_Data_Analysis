import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# [ Path Settings ]
#==================================================================================================
DATA_DIR = Path(r"E:\tables\preprocessing_for_sea_analysis\preprocessing_for_sea_analysis_results.csv")
OUTPUT_DIR = Path(r"E:\figures\sea_analysis_results_with_moving_average")
if not DATA_DIR.exists():
    raise FileNotFoundError(f"File not found: {DATA_DIR}")
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#--------------------------------------------------------------------------------------------------

# [ SEA Analysis ]
#==================================================================================================
MOVING_AVG_WINDOW = 51  # 2秒刻みデータで約100秒幅

sea_df = pd.read_csv(DATA_DIR)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
print("[Info]   DataFrame of SEA Analysis Results:\n==================================================")
sea_df.info()
print("--------------------------------------------------")
print(sea_df)
print("**************************************************")

# t_-194 から t_194 の各列平均を計算
t_cols = [c for c in sea_df.columns if c.startswith("t_")]
t_mean = sea_df[t_cols].mean(axis=0)

print("[Info]   Mean of each time column:\n==================================================")
print(t_mean)
print("**************************************************")

# 必要に応じて後続処理で使いやすいようDataFrame化
t_mean_df = t_mean.rename("mean_E_norm").reset_index().rename(columns={"index": "time_col"})
t_mean_df["time_s"] = t_mean_df["time_col"].str.replace("t_", "", regex=False).astype(int)
t_mean_df = t_mean_df.sort_values("time_s")
print("[Info]   DataFrame of Mean E_norm for each time column:\n==================================================")
t_mean_df.info()
print("--------------------------------------------------")
print(t_mean_df)
print("**************************************************")

# 時系列を滑らかにする移動平均
t_mean_df["mean_E_norm_ma"] = t_mean_df["mean_E_norm"].rolling(
    window=MOVING_AVG_WINDOW, center=True, min_periods=1
).mean()

# 結果を保存
t_mean_df.to_csv(OUTPUT_DIR / "t_mean_with_moving_average.csv", index=False)

# グラフ作成（元系列 + 移動平均）
fig, ax = plt.subplots(layout="constrained")
ax.plot(t_mean_df["time_s"], t_mean_df["mean_E_norm"], label="Time-series variations for all earthquake orbits", color="gray", alpha=0.6)
ax.plot(
    t_mean_df["time_s"],
    t_mean_df["mean_E_norm_ma"],
    label=f"Moving Average",
    color="blue",
    linewidth=2,
)

ax.tick_params(axis="both", labelsize=12)
ax.set_xticks([-200, -150, -100, -50, 0, 50, 100, 150, 200])
# ax.set_yticks([0.46, 0.47, 0.48, 0.49, 0.50])
# ax.set_ylim(0.46, 0.50)
ax.set_xlabel("Time (s)", fontsize=18)
ax.set_ylabel("Mean E_norm", fontsize=18)
ax.legend(fontsize=12)
plt.tight_layout()

fig.savefig(OUTPUT_DIR / "sea_analysis_with_moving_average_ver5.png", dpi=300)
plt.close(fig)
