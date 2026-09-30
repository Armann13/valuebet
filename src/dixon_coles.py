"""Zaman ağırlıklı Dixon-Coles gol modeli (tek lig ya da birden çok lig birlikte).

Ev sahibi golleri ~ Poisson(lambda), deplasman golleri ~ Poisson(nu); maç L liginde oynanıyorsa
    log lambda = mu[L] + home[L] + atak[ev] + defans[deplasman]
    log nu     = mu[L] +           atak[dep] + defans[ev]
defans burada "zayıflık"tır: büyük değer = çok gol yiyen takım.
Düşük skorlar (0-0, 1-0, 0-1, 1-1) için Dixon-Coles tau düzeltmesi (rho) uygulanır.
Geçmiş maçlar exp(-xi * gün) ile ağırlıklandırılır; atak/defansa L2 (ridge) çekimi uygulanır.

Birden çok lig (SP1 + SP2) birlikte eğitilince takım güçleri ortak ölçektedir; ligler arası ölçek, lig
değiştiren (düşen / çıkan) takımlar üzerinden bağlanır. Gol seviyesi (mu) ve ev avantajı lige özeldir.
Tek ligle eğitilince eski tek lig modeliyle birebir aynıdır.

Çekim iki yönde ayrı ayarlanır:
    güç    s = atak - defans  (1X2'yi belirler)                                  -> ridge
    eğilim t = atak + defans  (takımın maçlarındaki toplam gol, Ü/A'yı belirler) -> ridge_total, hedef 0
ceza = ridge/2 * Σ(s - hedef)² + ridge_total/2 * Σt²

Güç çekiminin hedefi takımın ligine (pencerede son maçını oynadığı lig) ve durumuna bağlıdır:
    hedef = lig_gücü[lig] + yeni_gelen[durum]
lig_gücü: ligin ortalama gücü (ilk lig 0 referans). yeni_gelen: bu sezon üst lige çıkmış ("çıkan") ya da alt
lige düşmüş ("düşen") takımların lig ortalamasından ortalama sapması. Bu olmadan, üst lige yeni çıkan takım
(ligin alt sıralarında olması beklenirken) üst lig ortalamasına doğru yukarı çekilir ve olduğundan güçlü
görünür; düşen takım da aşağı çekilir. Hepsi cezasız parametrelerdir; lig sırası adlara göredir (SP1 üst).
Durum için sezon sütunu gerekir; yoksa herkes "yeni değil" sayılır.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

import config

SINGLE = "_"  # df'te div sütunu yoksa tek ligin adı


@dataclass
class DixonColes:
    xi: float = config.XI
    ridge: float = config.RIDGE
    ridge_total: float | None = config.RIDGE_TOTAL  # None = ridge ile aynı
    lookback_days: int = config.LOOKBACK_DAYS
    max_goals: int = config.MAX_GOALS

    teams: list = field(default_factory=list)
    leagues: list = field(default_factory=list)       # ilk lig referans (gücü 0)
    mu: dict = field(default_factory=dict)            # lig -> gol seviyesi
    home: dict = field(default_factory=dict)          # lig -> ev avantajı
    league_strength: dict = field(default_factory=dict)  # lig -> ortalama güç (atak - defans)
    newcomer_offset: dict = field(default_factory=dict)  # "çıkan" / "düşen" -> lig ortalamasından sapma
    rho: float = 0.0
    attack: np.ndarray = field(default_factory=lambda: np.zeros(0))
    defence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    team_league: dict = field(default_factory=dict)   # takım -> pencerede son maçını oynadığı lig
    team_status: dict = field(default_factory=dict)   # takım -> +1 bu sezon çıktı, -1 düştü, 0 değil
    n_matches: dict = field(default_factory=dict)     # eğitim penceresinde takım başına maç sayısı
    ref_date: pd.Timestamp | None = None
    converged: bool = False

    # ------------------------------------------------------------------ eğitim
    def fit(self, df: pd.DataFrame, ref_date=None, warm_start: "DixonColes | None" = None) -> "DixonColes":
        """df: date, home, away, fthg, ftag (+ isteğe bağlı div, season) sütunları. Sadece ref_date'ten
        ÖNCEKİ maçlar kullanılır. df'te birden çok lig varsa hepsi tek modelde, ortak ölçekte eğitilir."""
        ref_date = pd.Timestamp(ref_date) if ref_date is not None else df["date"].max() + pd.Timedelta(days=1)
        d = df[(df["date"] < ref_date) & (df["date"] >= ref_date - pd.Timedelta(days=self.lookback_days))]
        d = d.dropna(subset=["fthg", "ftag"]).sort_values("date", kind="stable")
        if len(d) < 30:
            raise ValueError(f"eğitim için yeterli maç yok ({len(d)})")

        div = d["div"].astype(str) if "div" in d else pd.Series(SINGLE, index=d.index)
        leagues = sorted(div.unique())
        k = len(leagues)
        lidx = {lg: j for j, lg in enumerate(leagues)}
        teams = sorted(set(d["home"]) | set(d["away"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d["home"].map(idx).to_numpy()
        ai = d["away"].map(idx).to_numpy()
        li = div.map(lidx).to_numpy()
        x = d["fthg"].to_numpy(dtype=float)
        y = d["ftag"].to_numpy(dtype=float)
        w = np.exp(-self.xi * (ref_date - d["date"]).dt.days.to_numpy(dtype=float))
        team_league, team_status = _team_context(df[df["date"] < ref_date], d, div, leagues)
        tl = np.array([lidx[team_league[t]] for t in teams])
        ts = np.array([team_status.get(t, 0) for t in teams])

        # [mu(k), home(k), rho, lig_gücü(k-1), yeni_gelen çıkan & düşen (k>1 ise), atak(n), defans(n)]
        off = 3 * k + 2 * (k > 1)
        theta0 = np.zeros(off + 2 * n)
        for j in range(k):
            theta0[j] = np.log(max((x[li == j].mean() + y[li == j].mean()) / 2, 0.1))
            theta0[k + j] = 0.25
        if warm_start is not None and warm_start.teams:
            theta0[2 * k] = warm_start.rho
            for lg, j in lidx.items():
                theta0[j] = warm_start.mu.get(lg, theta0[j])
                theta0[k + j] = warm_start.home.get(lg, theta0[k + j])
                if j > 0:
                    theta0[2 * k + j] = warm_start.league_strength.get(lg, 0.0)
            if k > 1:
                theta0[3 * k] = warm_start.newcomer_offset.get("çıkan", 0.0)
                theta0[3 * k + 1] = warm_start.newcomer_offset.get("düşen", 0.0)
            prev = {t: i for i, t in enumerate(warm_start.teams)}
            for t, i in idx.items():
                if t in prev:
                    theta0[off + i] = warm_start.attack[prev[t]]
                    theta0[off + n + i] = warm_start.defence[prev[t]]

        bounds = [(None, None)] * (2 * k) + [(-0.25, 0.25)] + [(None, None)] * (off - 2 * k - 1 + 2 * n)
        res = minimize(_objective, theta0, args=(hi, ai, x, y, w, n, self.ridge, self.ridge_total, li, tl, k, ts),
                       jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 2000, "gtol": 1e-7})
        th = res.x
        self.teams, self.leagues = teams, leagues
        self.team_league, self.team_status = team_league, team_status
        self.mu = {lg: float(th[j]) for lg, j in lidx.items()}
        self.home = {lg: float(th[k + j]) for lg, j in lidx.items()}
        self.rho = float(th[2 * k])
        self.league_strength = {lg: (0.0 if j == 0 else float(th[2 * k + j])) for lg, j in lidx.items()}
        self.newcomer_offset = {"çıkan": float(th[3 * k]), "düşen": float(th[3 * k + 1])} if k > 1 else {}
        self.attack, self.defence = th[off:off + n].copy(), th[off + n:].copy()
        self.n_matches = pd.concat([d["home"], d["away"]]).value_counts().to_dict()
        self.ref_date, self.converged = ref_date, bool(res.success)
        return self

    # ------------------------------------------------------------------ tahmin
    def _league(self, div) -> str:
        if div in self.mu:
            return div
        if len(self.leagues) == 1:
            return self.leagues[0]
        raise ValueError(f"model bu ligi tanımıyor: {div} (ligler: {self.leagues})")

    def _team_params(self, team: str, league: str) -> tuple[float, float, bool]:
        """Bilinmeyen takım (ör. alt ligden yeni çıkan): maçın ligindeki takımlara göre zayıf varsayılan."""
        if team in self.teams:
            i = self.teams.index(team)
            return self.attack[i], self.defence[i], True
        same = np.array([self.team_league[t] == league for t in self.teams])
        a, d = (self.attack[same], self.defence[same]) if same.sum() >= 4 else (self.attack, self.defence)
        return float(np.quantile(a, 0.25)), float(np.quantile(d, 0.75)), False

    def expected_goals(self, home: str, away: str, div: str | None = None) -> tuple[float, float]:
        lg = self._league(div)
        ah, dh, _ = self._team_params(home, lg)
        aa, da, _ = self._team_params(away, lg)
        return (float(np.exp(self.mu[lg] + self.home[lg] + ah + da)),
                float(np.exp(self.mu[lg] + aa + dh)))

    def score_matrix(self, home: str, away: str, div: str | None = None) -> np.ndarray:
        lam, nu = self.expected_goals(home, away, div)
        g = np.arange(self.max_goals + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, nu))
        m[0, 0] *= 1 - lam * nu * self.rho
        m[0, 1] *= 1 + lam * self.rho
        m[1, 0] *= 1 + nu * self.rho
        m[1, 1] *= 1 - self.rho
        m = np.clip(m, 0, None)
        return m / m.sum()

    def predict(self, home: str, away: str, div: str | None = None) -> dict:
        lam, nu = self.expected_goals(home, away, div)
        m = self.score_matrix(home, away, div)
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
        r = pd.DataFrame({"team": self.teams, "league": [self.team_league[t] for t in self.teams],
                          "attack": self.attack, "defence_weakness": self.defence})
        r["net"] = r["attack"] - r["defence_weakness"]
        r["matches"] = r["team"].map(self.n_matches)
        return r.sort_values("net", ascending=False).reset_index(drop=True)


def _team_context(hist: pd.DataFrame, d: pd.DataFrame, div: pd.Series, leagues: list) -> tuple[dict, dict]:
    """Takımın ligi (penceredeki son maçı) ve durumu: +1 bu sezon üst lige çıktı, -1 alt lige düştü, 0 değil.
    'Bu sezon' = takımın son maçının sezonu; bir önceki sezonki ligi hist'ten (ref_date öncesi tüm maçlar)."""
    last = pd.DataFrame({"team": np.r_[d["home"], d["away"]], "div": np.r_[div, div],
                         "date": np.r_[d["date"], d["date"]],
                         "season": np.r_[d["season"], d["season"]] if "season" in d else 0}
                        ).sort_values("date", kind="stable").groupby("team").last()
    team_league = last["div"].to_dict()
    if len(leagues) < 2 or "season" not in d:
        return team_league, {}
    prev = {}
    for col in ("home", "away"):
        prev.update(zip(zip(hist[col], hist["season"]), hist["div"].astype(str)))
    rank = {lg: j for j, lg in enumerate(leagues)}
    status = {}
    for t, r in last.iterrows():
        p = prev.get((t, int(r["season"]) - 1))
        if p is not None and p != r["div"] and p in rank:
            status[t] = 1 if rank[r["div"]] < rank[p] else -1
    return team_league, status


def _objective(theta, hi, ai, x, y, w, n, ridge, ridge_total=None, li=None, tl=None, k=1, ts=None):
    """Ağırlıklı negatif log-olabilirlik + ridge ve analitik gradyanı.
    li: maçın lig indeksi, tl: takımın lig indeksi, k: lig sayısı, ts: takımın durumu (+1 çıktı, -1 düştü, 0)."""
    rt = ridge if ridge_total is None else ridge_total
    li = np.zeros(len(x), dtype=int) if li is None else li
    tl = np.zeros(n, dtype=int) if tl is None else tl
    ts = np.zeros(n, dtype=int) if ts is None else ts
    mu, h, rho = theta[:k], theta[k:2 * k], theta[2 * k]
    lstr = np.r_[0.0, theta[2 * k + 1:3 * k]]
    up, down = (ts == 1), (ts == -1)
    off = 3 * k + 2 * (k > 1)
    target = lstr[tl] + (theta[3 * k] * up + theta[3 * k + 1] * down if k > 1 else 0.0)
    a, d = theta[off:off + n], theta[off + n:]
    eta1 = mu[li] + h[li] + a[hi] + d[ai]
    eta2 = mu[li] + a[ai] + d[hi]
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
    e, t = a - d - target, a + d   # güç: hedeften sapma; eğilim
    obj = (-ll.sum() + 0.5 * (ridge * (e @ e) + rt * (t @ t))) / W
    grad = np.empty_like(theta)
    grad[:k] = -np.bincount(li, g1 + g2, k)
    grad[k:2 * k] = -np.bincount(li, g1, k)
    grad[2 * k] = -gr.sum()
    grad[2 * k + 1:3 * k] = -ridge * np.bincount(tl, e, k)[1:]
    if k > 1:
        grad[3 * k] = -ridge * e[up].sum()
        grad[3 * k + 1] = -ridge * e[down].sum()
    grad[off:off + n] = -(np.bincount(hi, g1, n) + np.bincount(ai, g2, n)) + ridge * e + rt * t
    grad[off + n:] = -(np.bincount(ai, g1, n) + np.bincount(hi, g2, n)) - ridge * e + rt * t
    return obj, grad / W
