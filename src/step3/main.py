from pathlib import Path
from candidate_orbit import extract_candidate_orbits

extract_candidate_orbits(
    orbit_index_csv= Path(r"E:\tables\step3_orbit_index.csv"),
    step2_dir= Path(r"E:\interim\step2_normalized")
)
