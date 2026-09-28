import pandas as pd, numpy as np

dd = pd.read_csv("output/delay_differences.csv")
dd_cols = [c for c in dd.columns if c.startswith("dd_")]
c = 299792458.0; delta_t = 0.001

n_total = len(dd)
n_pass = (dd.num_valid_samples > 0).sum()
lengths = dd.loc[dd.num_valid_samples > 0, "num_valid_samples"]
total_meas = lengths.sum()

n_mono, n_cross, n_none = 0, 0, 0
all_rr = []
for _, row in dd[dd.num_valid_samples > 0].iterrows():
    n = int(row.num_valid_samples)
    rr = row[dd_cols[:n]].to_numpy(dtype=float) * c / delta_t
    all_rr.extend(rr)
    if np.all(np.diff(rr) > 0): n_mono += 1
    s = np.sign(rr)
    crossings = np.sum(s[:-1] * s[1:] < 0)
    if crossings == 1: n_cross += 1
    elif crossings == 0: n_none += 1

print(f"n_total={n_total}, n_pass={n_pass}")
print(f"length mean={lengths.mean():.2f} median={lengths.median()} min={lengths.min()} max={lengths.max()}")
print(f"total measurements={total_meas}")
print(f"monotonic={n_mono}, single_crossing={n_cross}, no_crossing={n_none}")
print(f"range rate min={min(all_rr):.1f} max={max(all_rr):.1f}")