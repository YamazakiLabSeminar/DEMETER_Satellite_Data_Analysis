from pathlib import Path
from orbit_index import build_orbit_index

build_orbit_index(
    in_dir=Path(r"E:\interim\step2_normalized"),
    out_csv=Path(r"E:\tables\step3_orbit_index.csv"),
)
