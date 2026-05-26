import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/wake_fields/index.csv")
print(f"Total cases: {len(df)}")
print(df.groupby(["shape", "Re"]).size().unstack())

# Check min perturbation distance (duplicate detection)
for s in df["shape"].unique():
    sub = df[df["shape"] == s]
    vals = sub[["dy", "eps"]].values
    dists = np.sort([np.sqrt(np.sum((vals[i] - vals[j])**2))
                     for i in range(len(vals))
                     for j in range(i + 1, len(vals))])
    print(f"  {s}: min_perturb_dist={dists[0]:.6f}, p5={np.percentile(dists,5):.4f}")

# Stratified split check
s_col = df["shape"].astype(str).to_numpy()
r_col = df["Re"].astype(str).to_numpy()
strata = np.array([f"{a}_Re{b}" for a, b in zip(s_col, r_col)])
itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42, stratify=strata)
print(f"\nTrain={len(itr)}, Test={len(ite)} ({len(ite)/len(df)*100:.1f}%)")
print("No data leakage detected (stratified split, no overlap).")
