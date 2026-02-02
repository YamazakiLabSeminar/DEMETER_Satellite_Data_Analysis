from __future__ import annotations
from pathlib import Path

def load_yaml_config(path: Path) -> dict:
    """
    YAML設定ファイルを読み込んで dict を返す関数。
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError("pyyaml が必要です。`pip install pyyaml` を実行してください。") from e

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return cfg
