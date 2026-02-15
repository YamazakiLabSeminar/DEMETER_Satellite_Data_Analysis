from pathlib import Path

import pandas as pd


def main() -> None:
    input_csv = Path(r"F:\external\eq_catalog\eq_m4.8above_depth40kmbelow_2004-2010.csv")
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    time_col = df.columns[0]
    time_data = pd.to_datetime(df[time_col], errors="coerce")

    time_parts = pd.DataFrame(
        {
            "year": time_data.dt.year,
            "month": time_data.dt.month,
            "day": time_data.dt.day,
            "hour": time_data.dt.hour,
            "minute": time_data.dt.minute,
            "second": time_data.dt.second,
        }
    )

    df_processed = pd.concat([time_parts, df], axis=1)

    output_folder = Path(r"E:\tables\earthquake_catalog\add_time_column")
    output_folder.mkdir(parents=True, exist_ok=True)

    output_csv = output_folder / f"{input_csv.stem}_add-time-row.csv"
    df_processed.to_csv(output_csv, index=False)

    print(f"入力ファイル: {input_csv}")
    print(f"出力ファイル: {output_csv}")


if __name__ == "__main__":
    main()
