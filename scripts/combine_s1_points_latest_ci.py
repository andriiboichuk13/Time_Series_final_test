# Combine: point forecast from the Session-1 file (optionally blended with the latest file), 95% interval from the latest file.
# Winkler depends only on lower/upper and the actuals, so the interval can come from one file and the points from another.
# Colab: run this cell, upload the two CSVs when asked (or put them next to the notebook and set S1_FILE / LATEST_FILE).
import numpy as np, pandas as pd

S1_FILE, LATEST_FILE = "s1.csv", "latest.csv"      # Session 1 file, and the latest file (the one with the interval you like)
OUT_FILE = "Boichuk_project.csv"                    # name the template asks for
A1, A2 = 1.0, 1.0       # weight of the S1 point forecast on day 1 (hours 1-24) / day 2 (hours 25-48); 1.0 = pure S1, 0.5 = average with latest
NF, DF = 1.0, 1.0       # extra multiplier on night / day hours of the final point (the latest file already has night x0.85 inside; S1 does not)
APPLY_NF_TO_S1 = True   # S1 has no night factor yet: apply 0.85 to the S1 part so MAPE stays low (set False to keep S1 exactly as submitted)
NIGHT = None            # night = hours with sunrise/daylight 0; None = use hours 19-06 local (set to a boolean array of 48 for exact)

try:
    from google.colab import files
    import os
    if not (os.path.exists(S1_FILE) and os.path.exists(LATEST_FILE)):
        print("upload the two files: first the Session-1 file, then the latest file")
        up = files.upload(); names = list(up)
        S1_FILE, LATEST_FILE = names[0], names[1]
except ImportError:
    pass

s1, lt = pd.read_csv(S1_FILE), pd.read_csv(LATEST_FILE)
cols = ["ds", "forecast", "lower_95", "upper_95"]
assert list(s1.columns) == cols and list(lt.columns) == cols and len(s1) == len(lt) == 48 and (s1.ds == lt.ds).all(), "files must have the 4 columns and the same 48 timestamps"

hour = pd.to_datetime(lt.ds).dt.hour.to_numpy()
night = ((hour >= 19) | (hour <= 6)) if NIGHT is None else np.asarray(NIGHT, bool)
a = np.where(np.arange(48) < 24, A1, A2)
p_s1 = s1.forecast.to_numpy(float) * (np.where(night, 0.85, 1.0) if APPLY_NF_TO_S1 else 1.0)
p_lt = lt.forecast.to_numpy(float)
fc = a * p_s1 + (1 - a) * p_lt
fc = np.clip(fc * np.where(night, NF, DF), 0, None)

lo, hi = lt.lower_95.to_numpy(float), lt.upper_95.to_numpy(float)       # the interval comes from the latest file
moved = (fc < lo) | (fc > hi)
lo, hi = np.minimum(lo, fc), np.maximum(hi, fc)                          # the template requires lower <= forecast <= upper
sub = pd.DataFrame({"ds": lt.ds, "forecast": fc, "lower_95": np.clip(lo, 0, None), "upper_95": hi})

assert sub.notna().all().all() and (sub.lower_95 >= 0).all() and (sub.lower_95 <= sub.forecast).all() and (sub.forecast <= sub.upper_95).all()
w0, w1 = (lt.upper_95 - lt.lower_95), (sub.upper_95 - sub.lower_95)
print(f"{moved.sum()} of 48 hours had the S1 point outside the latest band -> band stretched there; mean width {w0.mean():.0f} -> {w1.mean():.0f}")
tot = lambda v: f"{v[:24].sum():.0f} / {v[24:].sum():.0f}"
print("day totals (day 1 / day 2): S1", tot(s1.forecast.to_numpy()), "| latest", tot(p_lt), "| combined", tot(fc))
print("max |combined - latest| per hour:", round(np.abs(fc - p_lt).max()), "| max |combined - S1|:", round(np.abs(fc - s1.forecast.to_numpy()).max()))
sub.assign(forecast=sub.forecast.round(3), lower_95=sub.lower_95.round(3), upper_95=sub.upper_95.round(3)).to_csv(OUT_FILE, index=False)
print("saved", OUT_FILE)
try:
    import matplotlib.pyplot as plt
    t = pd.to_datetime(sub.ds)
    plt.figure(figsize=(12, 4)); plt.fill_between(t, sub.lower_95, sub.upper_95, alpha=0.2, label="band (latest, stretched if needed)")
    plt.plot(t, s1.forecast, "--", label="S1 point"); plt.plot(t, p_lt, "--", label="latest point"); plt.plot(t, sub.forecast, lw=2, color="C0", label="combined point")
    plt.legend(); plt.show()
    files.download(OUT_FILE)
except Exception:
    pass
