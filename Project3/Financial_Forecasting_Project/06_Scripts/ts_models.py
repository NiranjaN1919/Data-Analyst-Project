"""
Small time-series toolkit (ARIMA and Holt-Winters/ETS) built on numpy + scipy.

Why this exists: statsmodels could not be installed in the environment where this project was built.
The models below follow the standard textbook definitions (conditional-sum-of-squares ARIMA;
additive-error damped-trend / seasonal exponential smoothing, Hyndman et al. 2008) and are checked on
simulated data with known answers in test_ts_models.py. If statsmodels is available you can swap in
statsmodels.tsa.arima.model.ARIMA / ExponentialSmoothing; the interfaces used by the analysis are tiny.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm


# ----------------------------------------------------------------------------------- ARIMA
class ARIMA:
    """ARIMA(p, d, q), d in {0, 1}, with a constant (mean for d=0, drift for d=1). CSS estimation."""

    def __init__(self, order):
        self.p, self.d, self.q = order
        if self.d not in (0, 1):
            raise ValueError("d must be 0 or 1")

    def _resid(self, par, z):
        p = self.p
        mu, phi, th = par[0], par[1:1 + p], par[1 + p:]
        return lfilter(np.r_[1.0, -phi], np.r_[1.0, th], z - mu)

    def _ok(self, par):
        p = self.p
        phi, th = par[1:1 + p], par[1 + p:]
        if p and np.abs(np.roots(np.r_[1.0, -phi])).max() >= 0.999:
            return False
        if self.q and np.abs(np.roots(np.r_[1.0, th])).max() >= 0.999:
            return False
        return True

    def fit(self, y):
        y = np.asarray(y, float)
        z = np.diff(y) if self.d == 1 else y.copy()
        p, q = self.p, self.q
        self.y_, self.z_ = y, z
        n = len(z)

        def sse(par):
            if not self._ok(par):
                return 1e12
            e = self._resid(par, z)[p:]
            return float(e @ e)

        x0 = np.zeros(1 + p + q)
        x0[0] = z.mean()
        if p:                                    # OLS start for the AR part
            X = np.column_stack([z[p - i - 1:n - i - 1] - z.mean() for i in range(p)])
            x0[1:1 + p] = np.clip(np.linalg.lstsq(X, z[p:] - z.mean(), rcond=None)[0], -0.9, 0.9)
        if q:
            x0[1 + p:] = 0.05
        bounds = [(None, None)] + [(-0.98, 0.98)] * (p + q)
        if p + q:
            r = minimize(sse, x0, method="L-BFGS-B", bounds=bounds)
            self.par_ = r.x if r.fun < sse(x0) else x0
        else:
            self.par_ = x0
        e = self._resid(self.par_, z)[p:]
        self.n_eff_ = len(e)
        self.sigma2_ = float(e @ e) / max(self.n_eff_ - (p + q + 1), 1)
        self.aic_ = self.n_eff_ * np.log(max(float(e @ e) / self.n_eff_, 1e-300)) + 2 * (p + q + 2)
        self.resid_ = self._resid(self.par_, z)
        return self

    def forecast(self, h, levels=(0.8, 0.95)):
        """Return dict: mean, and (lo, hi) arrays per confidence level, on the scale of the input series."""
        p, q = self.p, self.q
        mu, phi, th = self.par_[0], self.par_[1:1 + p], self.par_[1 + p:]
        x = list(self.z_ - mu)
        e = list(self.resid_)
        zf = []
        for _ in range(h):
            xh = sum(phi[i] * x[-1 - i] for i in range(p) if len(x) > i) + \
                 sum(th[j] * e[-1 - j] for j in range(q) if len(e) > j)
            x.append(xh)
            e.append(0.0)
            zf.append(xh + mu)
        zf = np.array(zf)
        mean = self.y_[-1] + np.cumsum(zf) if self.d == 1 else zf
        psi = np.zeros(h)
        psi[0] = 1.0
        for j in range(1, h):
            psi[j] = (th[j - 1] if j <= q else 0.0) + sum(phi[i - 1] * psi[j - i] for i in range(1, min(j, p) + 1))
        w = np.cumsum(psi) if self.d == 1 else psi
        sd = np.sqrt(self.sigma2_ * np.cumsum(w ** 2))
        out = {"mean": mean, "sd": sd}
        for lv in levels:
            zq = norm.ppf(0.5 + lv / 2)
            out[lv] = (mean - zq * sd, mean + zq * sd)
        return out


def auto_arima(y, d=1, grid=None):
    """Pick the (p, q) with the lowest AIC from a small grid."""
    if grid is None:
        grid = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    best = None
    for p, q in grid:
        try:
            m = ARIMA((p, d, q)).fit(y)
        except Exception:
            continue
        if best is None or m.aic_ < best.aic_:
            best = m
    return best


# ----------------------------------------------------------------------------- Holt-Winters
class HoltWinters:
    """Additive-error exponential smoothing: damped additive trend, optional additive seasonality (period m)."""

    def __init__(self, seasonal=False, m=12, damped=True):
        self.seasonal, self.m, self.damped = seasonal, m, damped

    def _init(self, y):
        m = self.m
        if self.seasonal:
            l0 = y[:m].mean()
            b0 = (y[m:2 * m].mean() - y[:m].mean()) / m
            s0 = y[:m] - l0
        else:
            l0, b0, s0 = y[0], y[1] - y[0], np.zeros(1)
        return l0, b0, s0

    def _run(self, par, y):
        a, bb, gg, ph = par
        b_, g_ = bb * a, gg * (1 - a)
        l, b, s = self._init(y)
        m = self.m if self.seasonal else 1
        s = list(s)
        sse = 0.0
        start = m if self.seasonal else 1
        for t in range(start if self.seasonal else 0, len(y)):
            st = s[t % m] if self.seasonal else 0.0
            yh = l + ph * b + st
            e = y[t] - yh
            sse += e * e
            l = l + ph * b + a * e
            b = ph * b + b_ * e
            if self.seasonal:
                s[t % m] = st + g_ * e
        return sse, l, b, s

    def fit(self, y):
        y = np.asarray(y, float)
        self.y_ = y
        x0 = [0.3, 0.1, 0.1, 0.9 if self.damped else 1.0]
        bounds = [(0.01, 0.99), (0.0, 1.0), (0.0, 1.0), (0.8, 0.98) if self.damped else (1.0, 1.0)]
        r = minimize(lambda p_: self._run(p_, y)[0], x0, method="L-BFGS-B", bounds=bounds)
        self.par_ = r.x
        self.sse_, self.l_, self.b_, self.s_ = self._run(self.par_, y)
        n = len(y) - (self.m if self.seasonal else 0)
        self.sigma2_ = self.sse_ / n
        self.n_ = len(y)
        return self

    def forecast(self, h, levels=(0.8, 0.95)):
        a, bb, gg, ph = self.par_
        mean = np.zeros(h)
        for k in range(1, h + 1):
            damp = sum(ph ** i for i in range(1, k + 1))
            seas = self.s_[(self.n_ + k - 1) % self.m] if self.seasonal else 0.0
            mean[k - 1] = self.l_ + damp * self.b_ + seas
        c = np.zeros(h)                          # c_j weights for the forecast-error variance
        for j in range(1, h):
            c[j] = a + bb * a * sum(ph ** i for i in range(1, j + 1)) + \
                   (gg * (1 - a) if self.seasonal and j % self.m == 0 else 0.0)
        var = self.sigma2_ * (1 + np.cumsum(c ** 2))
        sd = np.sqrt(var)
        out = {"mean": mean, "sd": sd}
        for lv in levels:
            zq = norm.ppf(0.5 + lv / 2)
            out[lv] = (mean - zq * sd, mean + zq * sd)
        return out


# --------------------------------------------------------------------------------- baselines
def naive_forecast(y, h):
    return np.repeat(y[-1], h)


def drift_forecast(y, h):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)
