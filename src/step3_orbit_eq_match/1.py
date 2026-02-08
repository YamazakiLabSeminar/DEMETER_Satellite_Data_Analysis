import pandas as pd

df = pd.read_csv(r"E:\tables\orbit_quake_ver4.csv")
cand_cols = [c for c in df.columns if c.startswith("orbit_meet_time_")]

time_only_orbits = set()
for _, row in df[cand_cols].iterrows():
    for x in row.dropna():
        time_only_orbits.add(x)

print("unique_orbits_time_only =", len(time_only_orbits))
