import sys, glob
import pandas as pd
d = sys.argv[1] + "/trajectories"
df = pd.read_parquet(sorted(glob.glob(d + "/episode_*.parquet"))[0])
df["t"] = df["frame_idx"] * 0.02
print("frames:", len(df))
cols = ["t", "commands_0", "commands_3", "root_lin_vel_robot_0", "root_pos_2"]
for t in [1, 4.5, 7.5, 12, 21, 26, 31]:
    row = df.iloc[(df["t"] - t).abs().argmin()]
    print("  ".join(f"{c}={row[c]:.2f}" for c in cols))
