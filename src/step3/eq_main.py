from pathlib import Path
from eq_catalog_standarlize import standardize_eq_catalog

standardize_eq_catalog(
    in_csv= Path(r"E:\interim\earthquake_catalog\eq_m4.8above_depth40kmbelow_2004-2010_declustred.csv"),
    out_csv= Path(r"E:\interim\earthquake_catalog\eq_m4.8above_depth40kmbelow_2004-2010_declustred_standardized.csv")
)