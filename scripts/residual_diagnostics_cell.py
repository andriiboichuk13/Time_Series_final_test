# Residual diagnostics of the final blend (run after the section 4 blend cell; needs actual, Pbl, SEGA, SEGB, W, tpos, K)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

res = actual - Pbl                                   # backtest origins x 48 hours, after the night factor
dl = W["daylight"][tpos]                             # 1 = day, 0 = night
hod = (np.arange(48)[None, :] + 1) % 24              # hour of day of each horizon step (forecast starts at 01:00)

def day1_series(seg):                                # hours 1-24 of consecutive midnights = one real hourly series
    return res[seg][:, :24].reshape(-1)

rows = []
for name, seg in (("recent (A)", SEGA), ("last autumn (B)", SEGB)):
    r = day1_series(seg)
    lb = acorr_ljungbox(r, lags=[24, 48], return_df=True)
    rows.append({"segment": name, "n hours": len(r), "mean residual": r.mean(), "std": r.std(),
                 "lag-1 autocorr": pd.Series(r).autocorr(1), "lag-24 autocorr": pd.Series(r).autocorr(24),
                 "Ljung-Box p, lag 24": lb["lb_pvalue"].iloc[0], "Ljung-Box p, lag 48": lb["lb_pvalue"].iloc[1]})
tab = pd.DataFrame(rows).set_index("segment").round(3)
display(tab)
print("p below 0.05 = the residuals still have autocorrelation (a pattern the model did not capture)")

fig, ax = plt.subplots(2, 3, figsize=(17, 8))
r = day1_series(SEGA)
ax[0, 0].plot(r, lw=0.7); ax[0, 0].axhline(0, c="k", lw=0.5); ax[0, 0].set_title("residuals, recent 60 days, day-1 hours in time order")
plot_acf(r, lags=72, ax=ax[0, 1]); ax[0, 1].set_title("ACF of those residuals (lags in hours)")
stats.probplot(res[:, :24].reshape(-1), dist="norm", plot=ax[0, 2]); ax[0, 2].set_title("normal QQ plot, all 120 origins, day 1")
ax[1, 0].bar(range(24), [res[:, :24][:, k].mean() for k in range(24)])
ax[1, 0].set_title("mean residual by forecast hour (day 1): bias"); ax[1, 0].set_xlabel("hours since origin - 1")
ax[1, 1].scatter(Pbl[:, :24].reshape(-1), res[:, :24].reshape(-1), s=4, alpha=0.4); ax[1, 1].axhline(0, c="k", lw=0.5)
ax[1, 1].set_title("residual vs forecast level: wider errors at the peak"); ax[1, 1].set_xlabel("forecast")
rmse_h = np.sqrt((res ** 2).mean(axis=0)); ax[1, 2].plot(np.arange(1, 49), rmse_h); ax[1, 2].axvline(24.5, c="gray", ls="--")
ax[1, 2].set_title("RMSE by horizon (hour 1 to 48)"); ax[1, 2].set_xlabel("horizon")
plt.tight_layout(); plt.show()

g = pd.DataFrame({"night": np.where(dl == 0, "night", "day").reshape(-1), "day": np.where(np.arange(48)[None, :] < 24, "day 1", "day 2").repeat(len(res), 0).reshape(-1), "res": res.reshape(-1), "abs": np.abs(res).reshape(-1)})
print(g.groupby(["night", "day"])["res"].agg(mean="mean", std="std").round(1).join(g.groupby(["night", "day"])["abs"].quantile(0.95).rename("95th pct |error|").round(0)))
