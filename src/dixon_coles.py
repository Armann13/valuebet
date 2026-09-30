"""Zaman ağırlıklı Dixon-Coles gol modeli.

Ev sahibi golleri ~ Poisson(lambda), deplasman golleri ~ Poisson(nu)
    log lambda = mu + home + atak[ev] + defans[deplasman]
    log nu     = mu +        atak[dep] + defans[ev]
defans burada "zayıflık"tır: büyük değer = çok gol yiyen takım.
Düşük skorlar (0-0, 1-0, 0-1, 1-1) için Dixon-Coles tau düzeltmesi (rho) uygulanır.
Geçmiş maçlar exp(-xi * gün) ile ağırlıklandırılır; atak/defansa L2 (ridge) çekimi uygulanır.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import config


@dataclass
class DixonColes:
    xi: float = config.XI
    ridge: float = config.RIDGE
    lookback_days: int = config.LOOKBACK_DAYS
    max_goals: int = config.MAX_GOALS

    teams: list = field(default_factory=list)
    mu: float = 0.0
    home: float = 0.0
    rho: float = 0.0
    attack: np.ndarray = field(default_factory=lambda: np.zeros(0))
    defence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_matches: dict = field(default_factory=dict)  # eğitim penceresinde takım başına maç sayısı
    ref_date: pd.Timestamp | None = None
    converged: bool = False

    # ------------------------------------------------------------------ eğitim
    def fit(self, df: pd.DataFrame, ref_date=None, warm_start: "DixonColes | None" = None) -> "DixonColes":
        """df: date, home, away, fthg, ftag sütunları. Sadece ref_date'ten ÖNCEKİ maçlar kullanılır."""
        ref_date = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max() + pd.Timedelta(days=1)
        d = df[(df["date"] < ref_date) & (df["date"] >= ref_date - pd.Timedelta(days=self.lookback_days))]
        d = d.dropna(subset=["fthg", "ftag"])
        if len(d) < 30:
            raise ValueError(f"eğitim için yeterli maç yok ({len(d)})")

        teams = sorted(set(d["home"]) | set(d["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d["home"].map(idx).to_numpy()
        ai = d["away"].map(idx).to_numpy()
        x = d["fthg"].to_numpy(dtype=float)
        y = d["ftag"].to_numpy(dtype=float)
        days = (ref_date - d["date"]).dt.days.to_numpy(dtype=float)
        w = np.exp(-self.xi * days)

        theta0 = np.zeros(3 + 2 * n)
        theta0[0] = np.log(max((x.mean() + y.mean()) / 2, 0.1))
        theta0[1] = 0.25
        if warm_start is not None and warm_start.teams:
            theta0[0], theta0[1], theta0[2] = warm_start.mu, warm_start.home, warm_start.rho
            prev = {t: i for i, t in enumerate(warm_start.teams)}
            for t, i in idx.items():
                if t in prev:
                    theta0[3 + i] = warm_start.attack[prev[t]]
                    theta0[3 + n + i] = warm_start.defence[prev[t]]

        bounds = [(None, None), (None, None), (-0.25, 0.25)] + [(None, None)] * (2 * n)
        res = minimize(_objective, theta0, args=(hi, ai, x, y, w, n, self.ridge),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "gtol": 1e-7})
        th = res.x
        self.teams, self.mu, self.home, self.rho = teams, th[0], th[1], th[2]
        self.attack, self.defence = th[3:3 + n].copy(), th[3 + n:].copy()
        cnt = pd.concat([d["home"], d["away"]]).value_counts()
        self.n_matches = cnt.to_dict()
        self.ref_date, self.converged = ref_date, bool(res.success)
        return self

    # ------------------------------------------------------------------ tahmin
    def _team_params(self, team: str) -> tuple[float, float, bool]:
        """Bilinmeyen takım (ör. uzun süredir bu ligde olmayan yükselen takım): zayıf varsayılan."""
        if team in self.teams:
            i = self.teams.index(team)
            return self.attack[i], self.defence[i], True
        return float(np.quantile(self.attack, 0.25)), float(np.quantile(self.defence, 0.75)), False

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        ah, dh, _ = self._team_params(home)
        aa, da, _ = self._team_params(away)
        return float(np.exp(self.mu + self.home + ah + da)), float(np.exp(self.mu + aa + dh))

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        lam, nu = self.expected_goals(home, away)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, nu))
        m[0, 0] *= 1 - lam * nu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + nu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def predict(self, home: str, away: str) -> dict:
        lam, nu = self.expected_goals(home, away)
        m = self.score_matrix(home, away)
        g = np.arange(self.max_goals + 1)
        total = g[:, None] + g[None, :]
        flat = np.argsort(m, axis=None)[::-1][:3]
        top = [f"{i // m.shape[1]}-{i % m.shape[1]} ({m.flat[i]:.0%})" for i in flat]
        return {
            "exp_home": lam, "exp_away": nu,
            "p_home": float(np.tril(m, -1).sum()),
            "p_draw": float(np.trace(m)),
            "p_away": float(np.triu(m, 1).sum()),
            "p_over25": float(m[total > 2.5].sum()),
            "p_under25": float(m[total < 2.5].sum()),
            "p_btts": float(m[1:, 1:].sum()),
            "top_scores": ", ".join(top),
            "known_home": home in self.teams, "known_away": away in self.teams,
            "n_home": int(self.n_matches.get(home, 0)), "n_away": int(self.n_matches.get(away, 0)),
        }

    def ratings(self) -> pd.DataFrame:
        r = pd.DataFrame({"team": self.teams, "attack": self.attack, "defence_weakness": self.defence})
        r["net"] = r["attack"] - r["defence_weakness"]
        r["matches"] = r["team"].map(self.n_matches)
        return r.sort_values("net", ascending=False).reset_index(drop=True)


def _objective(theta, hi, ai, x, y, w, n, ridge):
    """Ağırlıklı negatif log-olabilirlik + ridge ve analitik gradyanı."""
    mu, h, rho = theta[0], theta[1], theta[2]
    a, d = theta[3:3 + n], theta[3 + n:]
    eta1 = mu + h + a[hi] + d[ai]
    eta2 = mu + a[ai] + d[hi]
    lam, nu = np.exp(eta1), np.exp(eta2)

    tau = np.ones_like(lam)
    dt1 = np.zeros_like(lam)   # d tau / d eta1
    dt2 = np.zeros_like(lam)   # d tau / d eta2
    dtr = np.zeros_like(lam)   # d tau / d rho
    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)
    ln00 = lam[m00] * nu[m00]
    tau[m00] = 1 - ln00 * rho
    dt1[m00] = dt2[m00] = -ln00 * rho
    dtr[m00] = -ln00
    tau[m01] = 1 + lam[m01] * rho
    dt1[m01] = lam[m01] * rho
    dtr[m01] = lam[m01]
    tau[m10] = 1 + nu[m10] * rho
    dt2[m10] = nu[m10] * rho
    dtr[m10] = nu[m10]
    tau[m11] = 1 - rho
    dtr[m11] = -1.0
    tau = np.maximum(tau, 1e-10)

    ll = w * (np.log(tau) + x * eta1 - lam + y * eta2 - nu)
    g1 = w * (dt1 / tau + x - lam)
    g2 = w * (dt2 / tau + y - nu)
    gr = w * (dtr / tau)

    W = w.sum()
    obj = (-ll.sum() + ridge * (a @ a + d @ d)) / W
    grad = np.empty_like(theta)
    grad[0] = -(g1.sum() + g2.sum())
    grad[1] = -g1.sum()
    grad[2] = -gr.sum()
    grad[3:3 + n] = -(np.bincount(hi, g1, n) + np.bincount(ai, g2, n)) + 2 * ridge * a
    grad[3 + n:] = -(np.bincount(ai, g1, n) + np.bincount(hi, g2, n)) + 2 * ridge * d
    return obj, grad / W
