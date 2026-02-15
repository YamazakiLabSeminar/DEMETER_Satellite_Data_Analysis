import math
from pathlib import Path

import pandas as pd

# === Thresholds / constants ===
DAYS_THRESHOLD = 30       # days
DISTANCE_THRESHOLD = 30   # km

EARTH_RADIUS = 6378.140   # km (equatorial radius)
POLAR_RADIUS = 6356.755   # km (polar radius)


def timedelta_in_days(time1: pd.Timestamp, time2: pd.Timestamp) -> float:
    time_diff = abs(time1 - time2)
    return time_diff.total_seconds() / 86400.0


def calculate_distance(lon_a: float, lat_a: float, lon_b: float, lat_b: float) -> float:
    if lon_a == lon_b and lat_a == lat_b:
        return 0.0

    f = (EARTH_RADIUS - POLAR_RADIUS) / EARTH_RADIUS
    rad_lat_a = math.radians(lat_a)
    rad_lon_a = math.radians(lon_a)
    rad_lat_b = math.radians(lat_b)
    rad_lon_b = math.radians(lon_b)

    pa = math.atan((POLAR_RADIUS / EARTH_RADIUS) * math.tan(rad_lat_a))
    pb = math.atan((POLAR_RADIUS / EARTH_RADIUS) * math.tan(rad_lat_b))

    cos_xx = (
        math.sin(pa) * math.sin(pb)
        + math.cos(pa) * math.cos(pb) * math.cos(rad_lon_a - rad_lon_b)
    )
    cos_xx = min(1.0, max(-1.0, cos_xx))
    xx = math.acos(cos_xx)

    sin_half = math.sin(xx / 2.0)
    cos_half = math.cos(xx / 2.0)
    if sin_half == 0.0 or cos_half == 0.0:
        return EARTH_RADIUS * xx

    c1 = ((math.sin(xx) - xx) * (math.sin(pa) + math.sin(pb)) ** 2) / (cos_half ** 2)
    c2 = ((math.sin(xx) + xx) * (math.sin(pa) - math.sin(pb)) ** 2) / (sin_half ** 2)
    dr = (f / 8.0) * (c1 - c2)
    rho = EARTH_RADIUS * (xx + dr)
    return rho


def convert_to_datetime(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=df["day"],
            hour=df["hour"],
            minute=df["minute"],
            second=df["second"],
        ),
        errors="coerce",
    )


def main() -> None:
    input_csv_path = Path(r"E:\tables\earthquake_catalog\add_time_column\eq_m4.8above_depth40kmbelow_2004-2010_add-time-row.csv")

    output_csv_name = f"all-eq-declustering-{DAYS_THRESHOLD}day-{DISTANCE_THRESHOLD}km.csv"

    output_dir = Path(r"E:\tables\earthquake_catalog\declustered")
    output_csv_path = output_dir / output_csv_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")

    print(f"[Info] Reading input CSV: {input_csv_path}")
    df_original = pd.read_csv(
        input_csv_path,
        usecols=["year", "month", "day", "hour", "minute", "second", "latitude", "longitude", "mag"],
    )

    df_original["datetime"] = convert_to_datetime(df_original)
    df_original = df_original.dropna(subset=["datetime"]).copy()
    df_original.sort_values(by="datetime", ascending=True, inplace=True)
    df_original.reset_index(drop=True, inplace=True)
    df_original["event_id"] = df_original.index

    df_mag = df_original.sort_values("mag", ascending=False).reset_index(drop=True)
    df_time = df_original.sort_values("datetime", ascending=True).reset_index(drop=True)

    n_events = len(df_original)
    print(f"[Info] Total events = {n_events}")

    removed_flags = [False] * n_events
    dt_series = df_time["datetime"]

    total_events = len(df_mag)
    for primary_eq_index, primary_eq in df_mag.iterrows():
        e_id = int(primary_eq["event_id"])
        if removed_flags[e_id]:
            continue

        t_a = primary_eq["datetime"]
        cutoff_time = t_a + pd.Timedelta(days=DAYS_THRESHOLD)

        start_idx = int(dt_series.searchsorted(t_a, side="right"))
        end_idx = int(dt_series.searchsorted(cutoff_time, side="right"))

        lon_a = float(primary_eq["longitude"])
        lat_a = float(primary_eq["latitude"])

        for j in range(start_idx, end_idx):
            sec_event_id = int(df_time.at[j, "event_id"])
            if removed_flags[sec_event_id]:
                continue

            t_b = df_time.at[j, "datetime"]
            if timedelta_in_days(t_a, t_b) > DAYS_THRESHOLD:
                continue

            lon_b = float(df_time.at[j, "longitude"])
            lat_b = float(df_time.at[j, "latitude"])
            dist = calculate_distance(lon_a, lat_a, lon_b, lat_b)
            if dist <= DISTANCE_THRESHOLD:
                removed_flags[sec_event_id] = True

    remaining_data = df_original[[not f for f in removed_flags]].copy()
    remaining_data.to_csv(output_csv_path, index=False)

    print(f"[Info] Saved final declustered data to {output_csv_path}")


if __name__ == "__main__":
    main()
