"""Kart, faul, isabetli şut ve şut sayısı tahmini (sadece kulüp ligleri; football-data'da maç başı sayılar var).

Korner modeliyle aynı yöntem (bkz. korner.py): gol modelinin aynısı ilgili sayılarla (ev / deplasman, takımın
"üretme" ve "yeme" eğilimi, zaman ağırlıklı, ülke grubu başına birleşik model), takım başına negatif binom.
Kart = sarı + kırmızı (İddaa: "toplam kart sayısı"). Ayarlar walk-forward doğrulamayla seçildi (STATS).
Milli maçlarda bu veriler yok: analizde İddaa'nın oranı gösterilir.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom

import config
from src.dixon_coles import DixonColes

# ad -> ev / deplasman sütunları (toplanır), model ayarı, negatif binom k, olasılık tablosunun üst sınırı.
# Walk-forward 2024/25+ (aylık yeniden eğitim, ~15.500 maç, 2026-10-02), toplam "x+" barajlarında log loss model /
# lig ortalaması: kart 0.597 / 0.601 (hakem verisi yok, takım farkı az), faul 0.570 / 0.589, isabetli şut
# 0.615 / 0.624, şut 0.561 / 0.572. k: kartta 20, diğerlerinde 20 ile 100 arası düz.
STATS = {
    "kart": {"home": ["hy", "hr"], "away": ["ay", "ar"], "xi": 0.0019, "ridge": 8.0, "rt": 64.0, "k": 20.0, "max": 20},
    "faul": {"home": ["hf"], "away": ["af"], "xi": 0.0019, "ridge": 8.0, "rt": 64.0, "k": 100.0, "max": 45},
    "isabetli": {"home": ["hst"], "away": ["ast"], "xi": 0.0019, "ridge": 8.0, "rt": 64.0, "k": 50.0, "max": 30},
    "şut": {"home": ["hsh"], "away": ["ash"], "xi": 0.0019, "ridge": 8.0, "rt": 64.0, "k": 50.0, "max": 60},
}


def frame(matches: pd.DataFrame, stat: str) -> pd.DataFrame:
    """İlgili sayıların olduğu maçlar; gol sütunlarının yerine bu sayılar (model aynı sütunları okur)."""
    s = STATS[stat]
    cols = s["home"] + s["away"]
    if not set(cols) <= set(matches.columns):
        return pd.DataFrame()
    d = matches.dropna(subset=cols).drop(columns=["fthg", "ftag", "mot", "hxg", "axg"], errors="ignore")
    return d.assign(fthg=d[s["home"]].sum(axis=1).astype(int), ftag=d[s["away"]].sum(axis=1).astype(int))


def model(stat: str, **kw) -> DixonColes:
    s = {**STATS[stat], **kw}
    return DixonColes(xi=s["xi"], ridge=s["ridge"], ridge_total=s["rt"], xg_weight=0.0, fit_rho=False,
                      motivation_fixed=None)


def fit_models(matches: pd.DataFrame, ref_dates: dict, stat: str) -> dict:
    """Lig -> model (ülke grubu başına bir birleşik model; ligin verisi yoksa yok)."""
    sf = frame(matches, stat)
    models = {}
    if sf.empty:
        return models
    for gdivs in config.LEAGUE_GROUPS.values():
        refs = [ref_dates[d] for d in gdivs if d in ref_dates]
        g = sf[sf["div"].isin(gdivs)]
        if not refs or g.empty:
            continue
        try:
            m = model(stat).fit(g, ref_date=min(refs))
        except ValueError:
            continue
        models.update({d: m for d in gdivs if d in ref_dates and d in m.mu})
    return models


def pmf(mean: float, stat: str) -> np.ndarray:
    s = STATS[stat]
    g = np.arange(s["max"] + 1)
    p = nbinom.pmf(g, s["k"], s["k"] / (s["k"] + mean))
    return p / p.sum()


def probs(lam: float, nu: float, stat: str) -> dict:
    """Ev, deplasman ve toplam için P(sayı >= x) tabloları ve "en çok" 1 / X / 2."""
    ph, pa = pmf(lam, stat), pmf(nu, stat)
    tot = np.convolve(ph, pa)
    m = np.outer(ph, pa)
    g = np.arange(len(ph))
    h, a = np.meshgrid(g, g, indexing="ij")
    at_least = lambda p: np.cumsum(p[::-1])[::-1]  # noqa: E731  # P(X >= x)
    return {"ev": at_least(ph), "dep": at_least(pa), "toplam": at_least(tot),
            "1": float(m[h > a].sum()), "X": float(m[h == a].sum()), "2": float(m[h < a].sum()),
            "ort": (lam, nu)}


# ------------------------------------------------------------------ doğrulama
def walk_forward(matches: pd.DataFrame, stat: str, start: str, refit_days: int = 28, **kw) -> pd.DataFrame:
    sf = frame(matches, stat)
    rows = []
    for gdivs in config.LEAGUE_GROUPS.values():
        g = sf[sf["div"].isin(gdivs)]
        if g.empty:
            continue
        ev = g[g["date"] >= pd.Timestamp(start)]
        prev = None
        for ref in pd.date_range(pd.Timestamp(start), ev["date"].max() + pd.Timedelta(days=1), freq=f"{refit_days}D"):
            part = ev[(ev["date"] >= ref) & (ev["date"] < ref + pd.Timedelta(days=refit_days))]
            if part.empty:
                continue
            try:
                prev = model(stat, **kw).fit(g, ref_date=ref, warm_start=prev)
            except ValueError:
                continue
            for r in part.itertuples(index=False):
                if r.div in prev.mu:
                    lam, nu = prev.expected_goals(r.home, r.away, r.div)
                    rows.append({"div": r.div, "h": int(r.fthg), "a": int(r.ftag), "lam": lam, "nu": nu})
    return pd.DataFrame(rows)


def evaluate(p: pd.DataFrame, stat: str, ks, lines) -> pd.DataFrame:
    """Toplam için "x+" barajlarında log loss: model ve lig ortalaması, dağılım parametresine göre."""
    tot = (p["h"] + p["a"]).to_numpy()
    lm = p.groupby("div")[["h", "a"]].transform("mean")
    out = []
    for name, L, N in [("model", p["lam"], p["nu"]), ("lig ortalaması", lm["h"], lm["a"])]:
        for k in ks:
            old = STATS[stat]["k"]
            STATS[stat]["k"] = k
            ll = []
            for lam, nu, t in zip(L, N, tot):
                pr = probs(lam, nu, stat)["toplam"]
                for x in lines:
                    q = pr[x] if x < len(pr) else 0.0
                    ll.append(-np.log(max(q if t >= x else 1 - q, 1e-12)))
            STATS[stat]["k"] = old
            out.append({"tahmin": name, "k": k, "log loss": float(np.mean(ll))})
    return pd.DataFrame(out)
