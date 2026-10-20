import sys, json, glob
import numpy as np, pandas as pd
d = sys.argv[1] + "/trajectories"
names = json.load(open(d + "/metadata.json"))["joint_names"]
files = sorted(glob.glob(d + "/episode_*.parquet"))
df = pd.read_parquet(files[0])
print("episode files:", len(files), "| frames in episode 0:", len(df))
for n in ["left_ankle_pitch_joint", "left_knee_joint", "left_hip_pitch_joint"]:
    w = df[f"joint_vel_{names.index(n)}"].to_numpy()
    flips = np.mean(np.sign(w[1:]) != np.sign(w[:-1]))
    print(f"{n:24s} mean|w| = {np.abs(w).mean():5.2f} rad/s | sign flips per frame = {flips:.2f}")
col = f"joint_vel_{names.index('left_ankle_pitch_joint')}"
print("\nleft ankle pitch velocity, first 20 frames:")
print(np.round(df[col].to_numpy()[:20], 2))
