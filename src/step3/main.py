from pathlib import Path
from orbit_eq_matchi_time import match_orbit_to_eq_time

match_orbit_to_eq_time(
    orbit_index_csv= Path(r"E:\tables\step3_orbit_index.csv"),
    eq_csv= Path(r"E:\interim\earthquake_catalog\eq_m4.8above_depth40kmbelow_2004-2010_declustred_standardized.csv")
)
