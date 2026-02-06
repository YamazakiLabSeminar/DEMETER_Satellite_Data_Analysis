from pathlib import Path
from orbit_index import build_orbit_index
from orbit_eq_matchi_time import match_orbit_to_eq_time
from orbit_eq_matchi_distance import match_orbit_distance

# orbit_index表の作成
#    build_orbit_index(
#        in_dir= Path(r"E:\interim\step2_normalized"),
#        out_csv= Path(r"E:\tables\step3_orbit_index.csv")
#    )
#===================================================================
#    match_orbit_to_eq_time(
#        orbit_index_csv= Path(r"E:\tables\step3_orbit_index.csv"),
#        eq_csv= Path(r"E:\interim\earthquake_catalog\eq_m4.8above_depth40kmbelow_2004-2010_declustred_standardized.csv")
#    )

match_orbit_distance(
    orbit_index_csv= Path(r"E:\tables\step3_orbit_index.csv"),
    candidate_dir= Path(r"E:\interim\step3_candidate")
)