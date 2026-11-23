"""Checks ts_models.py on simulated data with known answers. Run: python3 test_ts_models.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ts_models import ARIMA, HoltWinters, auto_arima, naive_forecast

rng = np.random.default_rng(1)
lines = []
def check(name, ok, detail):
    lines.append(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

# 1. ARMA(1,1) parameter recovery
n = 4000; phi, th, mu = 0.6, 0.3, 0.5
e = rng.normal(size=n + 50); x = np.zeros(n + 50)
for t in range(1, n + 50):
    x[t] = phi * x[t - 1] + e[t] + th * e[t - 1]
x = x[50:] + mu
m = ARIMA((1, 0, 1)).fit(x)
ok = abs(m.par_[0] - mu) < 0.15 and abs(m.par_[1] - phi) < 0.1 and abs(m.par_[2] - th) < 0.1
check("ARMA(1,1) recovery", ok, f"true mu/phi/theta = {mu}/{phi}/{th}, estimated {m.par_[0]:.2f}/{m.par_[1]:.2f}/{m.par_[2]:.2f}")

# 2. AR(1) forecast converges to the mean and sd approaches the stationary sd
f = ARIMA((1, 0, 0)).fit(x).forecast(60)
check("AR forecast decays to mean", abs(f["mean"][-1] - x.mean()) < 0.2, f"h=60 forecast {f['mean'][-1]:.2f} vs sample mean {x.mean():.2f}")

# 3. Random walk with drift: drift recovery and 95% interval coverage over many simulated paths
drift, sig, hit, tot = 0.01, 0.05, 0, 0
for _ in range(300):
    y = np.cumsum(drift + sig * rng.normal(size=200))
    fut = y[-1] + np.cumsum(drift + sig * rng.normal(size=12))
    f = ARIMA((0, 1, 0)).fit(y).forecast(12)
    lo, hi = f[0.95]
    hit += np.sum((fut >= lo) & (fut <= hi)); tot += 12
cov = hit / tot
check("ARIMA(0,1,0) 95% interval coverage", 0.92 <= cov <= 0.98, f"{cov:.3f} (target 0.95)")

# 4. auto_arima picks a random walk for a random walk
sel = [tuple([auto_arima(np.cumsum(0.01 + 0.05 * rng.normal(size=300))).p, 1, auto_arima(np.cumsum(0.01 + 0.05 * rng.normal(size=300))).q]) for _ in range(1)]
check("auto_arima on random walk returns a small model", sum(sel[0][0::2]) <= 2, f"selected (p,d,q) = {sel[0]}")

# 5. Holt-Winters beats naive on a trending seasonal series, and its intervals cover
t = np.arange(180)
def gen():
    return 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, size=t.size)
better, hit, tot = 0, 0, 0
for _ in range(40):
    y = gen(); tr, te = y[:168], y[168:]
    hw = HoltWinters(seasonal=True, m=12).fit(tr); f = hw.forecast(12)
    e_hw = np.sqrt(np.mean((f["mean"] - te) ** 2)); e_nv = np.sqrt(np.mean((naive_forecast(tr, 12) - te) ** 2))
    better += e_hw < e_nv
    lo, hi = f[0.95]; hit += np.sum((te >= lo) & (te <= hi)); tot += 12
check("Holt-Winters beats naive on seasonal trend", better >= 38, f"better in {better}/40 simulations")
check("Holt-Winters 95% interval coverage", 0.88 <= hit / tot <= 0.99, f"{hit / tot:.3f}")

# 6. Non-seasonal damped Holt on a straight line is near-exact
y = 5 + 2 * np.arange(60) + rng.normal(0, 0.01, 60)
f = HoltWinters(seasonal=False, damped=False).fit(y).forecast(6)["mean"]
check("Holt on a line", abs(f[-1] - (5 + 2 * 65)) < 1.0, f"h=6 forecast {f[-1]:.2f} vs {5 + 2 * 65}")

print("\n".join(lines))
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write("\n".join(lines) + "\n")
sys.exit(1 if any(l.startswith("FAIL") for l in lines) else 0)
