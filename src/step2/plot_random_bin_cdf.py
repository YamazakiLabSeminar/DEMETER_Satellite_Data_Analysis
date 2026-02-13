from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config_loader import load_yaml_config
from src.paths import CONFIG_DIR, FIGURES_DIR, INTERIM_DIR, ensure_dirs


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot CDF of E_norm for a random bin_id")
    p.add_argument(
        "--config",
        type=str,
        default="step2_normalization.yaml",
        help="Step2 YAML config path. If relative, it is resolved under CONFIG_DIR.",
    )
    p.add_argument(
        "--step2_dir",
        type=str,
        default="",
        help="Override Step2 output dir (full path). If empty, use config io.step2_dirname under INTERIM_DIR.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for choosing bin_id (0 means no fixed seed).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. If empty, save under FIGURES_DIR.",
    )
    p.add_argument(
        "--logx",
        action="store_true",
        help="Use log scale for x-axis (E_1700band_mean).",
    )
    return p


def resolve_config_path(cfg_arg: str, config_dir: Path) -> Path:
    p = Path(cfg_arg)
    if p.is_absolute():
        return p
    return config_dir / p


def pick_random_bin_id(files: list[Path], seed: int | None) -> str:
    """
    Unique bin_id から一様ランダムで1つ選ぶ（reservoir sampling）。
    """
    if seed is not None and seed != 0:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    seen: set[str] = set()
    n = 0
    chosen: str | None = None

    for f in files:
        df = pd.read_csv(f, usecols=["bin_id"])
        bin_ids = df["bin_id"].dropna().astype(str).unique()
        for b in bin_ids:
            if b in seen:
                continue
            seen.add(b)
            n += 1
            if rng.random() < (1.0 / n):
                chosen = b

    if chosen is None:
        raise RuntimeError("No valid bin_id found in Step2 outputs.")
    return chosen


def collect_pairs(files: list[Path], bin_id: str) -> tuple[np.ndarray, np.ndarray]:
    e_vals: list[np.ndarray] = []
    en_vals: list[np.ndarray] = []
    for f in files:
        df = pd.read_csv(f, usecols=["bin_id", "E_1700band_mean", "E_norm"])
        sub = df[df["bin_id"].astype(str) == bin_id][["E_1700band_mean", "E_norm"]]
        sub["E_1700band_mean"] = pd.to_numeric(sub["E_1700band_mean"], errors="coerce")
        sub["E_norm"] = pd.to_numeric(sub["E_norm"], errors="coerce")
        sub = sub.dropna()
        if len(sub) > 0:
            e_vals.append(sub["E_1700band_mean"].to_numpy(dtype=float))
            en_vals.append(sub["E_norm"].to_numpy(dtype=float))
    if not e_vals:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(e_vals), np.concatenate(en_vals)


def plot_scatter(
    e_vals: np.ndarray,
    en_vals: np.ndarray,
    bin_id: str,
    out_path: Path,
    logx: bool,
) -> None:
    fig, ax = plt.subplots(layout="constrained")
    ax.scatter(e_vals, en_vals, s=6, alpha=0.6, edgecolors="none")
    ax.set_title(f"E_1700band_mean vs E_norm (bin_id={bin_id})", fontsize=18)
    ax.set_xlabel("E_1700band_mean", fontsize=18)
    ax.set_ylabel("E_norm", fontsize=18)
    ax.tick_params(axis="both", labelsize=12)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    args = build_argparser().parse_args()
    ensure_dirs()

    cfg_path = resolve_config_path(args.config, CONFIG_DIR)
    cfg = load_yaml_config(cfg_path)

    if args.step2_dir:
        step2_dir = Path(args.step2_dir)
    else:
        step2_dirname = cfg.get("io", {}).get("step2_dirname", "step2_normalized")
        step2_dir = INTERIM_DIR / step2_dirname

    if not step2_dir.exists():
        raise FileNotFoundError(f"Step2 directory not found: {step2_dir}")

    files = sorted(step2_dir.glob("*_step2.csv"))
    if not files:
        raise FileNotFoundError(f"No Step2 csv found in: {step2_dir}")

    bin_id = pick_random_bin_id(files, seed=args.seed)
    e_vals, en_vals = collect_pairs(files, bin_id)
    if len(e_vals) == 0:
        raise RuntimeError(f"No E_1700band_mean / E_norm found for bin_id={bin_id}")

    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = FIGURES_DIR / f"cdf_enorm_{ts}.png"

    plot_scatter(e_vals, en_vals, bin_id, out_path, logx=args.logx)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
