"""Korner tahmini (sadece kulüp ligleri: football-data'da maç başı korner sayısı HC / AC var; milli maçlarda yok).

Model: gol modelinin aynısı (zaman ağırlıklı Dixon-Coles, ülke grubu başına birleşik model), gol yerine korner
sayısıyla ve düşük skor düzeltmesi olmadan (rho = 0):
    log E[ev kornerleri] = mu[lig] + ev[lig] + atak[ev] + defans[dep]   (atak = korner kazandırma eğilimi)
Kornerler Poisson'dan daha dağınık (aynı ortalamada daha çok uç değer); bu yüzden her takımın korner sayısı
negatif binom kabul edilir: varyans = ortalama + ortalama² / KORNER_DISPERSIYON. Dağılım parametresi ve model
ayarları walk-forward doğrulamayla seçildi (bkz. config.KORNER_*, python main.py korner).

Marketler: toplam korner alt / üst (İddaa'nın barajında) ve en çok korner (1 / X / 2).
Keskin piyasa oranı olmadığı için korner tahminleri sadece modeldir (💎 çıkmaz); isabet sonuç takibinden izlenir.
"""
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

import config
from src.dixon_coles import DixonColes

MAX_CORNERS = 30


def corner_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Korner sayısı olan maçlar, gol sütunlarının yerine korner sayılarıyla (model aynı sütunları okur)."""
    if "hc" not in matches:
        return pd.DataFrame()
    d = matches.dropna(subset=["hc", "ac"]).drop(columns=["fthg", "ftag", "mot", "hxg", "axg"], errors="ignore")
    return d.rename(columns={"hc": "fthg", "ac": "ftag"})


def model(xi=None, ridge=None, ridge_total=None) -> DixonColes:
    return DixonColes(xi=config.KORNER_XI if xi is None else xi,
                      ridge=config.KORNER_RIDGE if ridge is None else ridge,
                      ridge_total=config.KORNER_RIDGE_TOTAL if ridge_total is None else ridge_total,
                      xg_weight=0.0, fit_rho=False, motivation_fixed=None)


def fit_models(matches: pd.DataFrame, ref_dates: dict) -> dict:
    """Lig -> korner modeli (ülke grubu başına bir birleşik model, grubun en erken tahmin tarihiyle)."""
    cf = corner_frame(matches)
    models = {}
    if cf.empty:
        return models
    for gdivs in config.LEAGUE_GROUPS.values():
        refs = [ref_dates[d] for d in gdivs if d in ref_dates]
        g = cf[cf["div"].isin(gdivs)]
        if not refs or g.empty:
            continue
        try:
            m = model().fit(g, ref_date=min(refs))
        except ValueError as e:
            print(f"  ! korner {', '.join(gdivs)}: {e}")
            continue
        models.update({d: m for d in gdivs if d in ref_dates and d in m.mu})  # korner verisi olan ligler
    return models


def _pmf(mean: float, k: float | None) -> np.ndarray:
    g = np.arange(MAX_CORNERS + 1)
    if k is None:
        p = poisson.pmf(g, mean)
    else:  # negatif binom: n = k, p = k / (k + ortalama)
        p = nbinom.pmf(g, k, k / (k + mean))
    return p / p.sum()


def matrix(lam: float, nu: float, k: float | None = None) -> np.ndarray:
    """m[ev kornerleri, deplasman kornerleri] (bağımsız)."""
    k = config.KORNER_DISPERSIYON if k is None else (None if k == 0 else k)
    return np.outer(_pmf(lam, k), _pmf(nu, k))


def market_probs(m: np.ndarray, line: float | None) -> dict:
    """Toplam korner alt / üst (line, ör. 10.5) ve en çok korner 1 / X / 2."""
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    out = {"En çok korner 1": float(m[h > a].sum()), "En çok korner X": float(m[h == a].sum()),
           "En çok korner 2": float(m[h < a].sum())}
    if line is not None and np.isfinite(line):
        out[f"Korner {line:g} Üst"] = float(m[(h + a) > line].sum())
        out[f"Korner {line:g} Alt"] = float(m[(h + a) < line].sum())
    return out


def bet_won(bet: str, hc: int, ac: int) -> bool:
    """Korner tahmini tuttu mu ("Korner 10.5 Üst", "En çok korner 1" ...)."""
    if bet.startswith("En çok korner"):
        side = bet.split()[-1]
        return {"1": hc > ac, "X": hc == ac, "2": hc < ac}[side]
    _, line, side = bet.split()
    return (hc + ac > float(line)) if side == "Üst" else (hc + ac < float(line))


def is_corner_bet(bet: str) -> bool:
    return str(bet).startswith(("Korner ", "En çok korner"))


# ------------------------------------------------------------------ doğrulama
def walk_forward(matches: pd.DataFrame, start: str, refit_days: int = 28, xi=None, ridge=None,
                 ridge_total=None) -> pd.DataFrame:
    """Her refit_days günde bir (grup grup) yeniden eğitilir; sonraki maçların beklenen kornerleri (lam, nu)."""
    cf = corner_frame(matches)
    rows = []
    for gdivs in config.LEAGUE_GROUPS.values():
        g = cf[cf["div"].isin(gdivs)]
        if g.empty:
            continue
        ev = g[g["date"] >= pd.Timestamp(start)]
        prev = None
        for ref in pd.date_range(pd.Timestamp(start), ev["date"].max() + pd.Timedelta(days=1), freq=f"{refit_days}D"):
            part = ev[(ev["date"] >= ref) & (ev["date"] < ref + pd.Timedelta(days=refit_days))]
            if part.empty:
                continue
            try:
                prev = model(xi, ridge, ridge_total).fit(g, ref_date=ref, warm_start=prev)
            except ValueError:
                continue
            for r in part.itertuples(index=False):
                if r.div not in prev.mu:  # ligin pencerede korner verisi yok (ör. EC bazı sezonlar)
                    continue
                lam, nu = prev.expected_goals(r.home, r.away, r.div)
                rows.append({"div": r.div, "date": r.date, "home": r.home, "away": r.away,
                             "hc": int(r.fthg), "ac": int(r.ftag), "lam": lam, "nu": nu})
        print(f"  korner walk-forward: {', '.join(gdivs)} tamam")
    return pd.DataFrame(rows)


def evaluate(p: pd.DataFrame, ks=(0, 50, 20, 12, 8, 5), lines=(8.5, 9.5, 10.5, 11.5)) -> pd.DataFrame:
    """Dağılım parametresine göre log loss: toplam alt / üst (her baraj) ve en çok korner 1 / X / 2.
    Kıyas: ligin geçmiş ortalamasıyla (takım farkı olmadan) aynı dağılım."""
    tot = (p["hc"] + p["ac"]).to_numpy()
    res = np.select([p["hc"] > p["ac"], p["hc"] == p["ac"]], [0, 1], 2)
    league_mean = p.groupby("div")[["hc", "ac"]].transform("mean")  # basit kıyas (iyimser: aynı dönemin ortalaması)
    out = []
    for name, lam_s, nu_s in [("model", p["lam"], p["nu"]), ("lig ortalaması", league_mean["hc"], league_mean["ac"])]:
        for k in ks:
            ll_ou, ll_1x2 = [], []
            for lam, nu, t, y in zip(lam_s, nu_s, tot, res):
                m = matrix(lam, nu, k)
                g = np.arange(m.shape[0])
                h, a = np.meshgrid(g, g, indexing="ij")
                for line in lines:
                    po = m[(h + a) > line].sum()
                    ll_ou.append(-np.log(max(po if t > line else 1 - po, 1e-12)))
                p3 = [m[h > a].sum(), m[h == a].sum(), m[h < a].sum()]
                ll_1x2.append(-np.log(max(p3[y], 1e-12)))
            out.append({"tahmin": name, "dağılım": "Poisson" if k == 0 else f"NB k={k}",
                        "alt/üst log loss": np.mean(ll_ou), "en çok korner log loss": np.mean(ll_1x2)})
    return pd.DataFrame(out)


def run(start: str = "2024-07-01") -> str:
    """Doğrulama raporu (reports/korner_summary.md)."""
    from src import db
    p = walk_forward(db.load_matches(), start)
    if p.empty:
        return "Korner verisi yok (python main.py build; football-data HC / AC sütunları)."
    ev = evaluate(p, ks=(0, config.KORNER_DISPERSIYON))
    cal = calibration(p, 9.5)

    def table(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                         ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    cal = cal.assign(dilim=cal["dilim"].astype(str), tahmin=cal["tahmin"].map("{:.1%}".format),
                     gerçek=cal["gerçek"].map("{:.1%}".format))
    return "\n".join([
        "# Korner modeli — doğrulama", "",
        f"Walk-forward {start}+ (28 günde bir yeniden eğitim), {len(p)} maç. Ayar: xi={config.KORNER_XI}, "
        f"ridge={config.KORNER_RIDGE}, ridge_total={config.KORNER_RIDGE_TOTAL}, negatif binom k={config.KORNER_DISPERSIYON}.",
        "", table(ev.round(4)), "",
        "Alt / üst log loss: 8.5, 9.5, 10.5 ve 11.5 barajlarının ortalaması. 'Lig ortalaması' = takım farkı olmadan "
        "ligin ortalama korner sayısı (aynı dönemin ortalaması: iyimser bir kıyas).", "",
        "## Kalibrasyon (toplam korner 9.5 üst)", "", table(cal)])


def calibration(p: pd.DataFrame, line: float = 9.5, k: float | None = None) -> pd.DataFrame:
    """Toplam korner üst olasılığı dilimlerine göre gerçekleşme oranı."""
    po = np.array([market_probs(matrix(l, n, k), line)[f"Korner {line:g} Üst"] for l, n in zip(p["lam"], p["nu"])])
    hit = (p["hc"] + p["ac"]).to_numpy() > line
    d = pd.DataFrame({"p": po, "hit": hit})
    d["dilim"] = pd.cut(d["p"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    return d.groupby("dilim", observed=True).agg(tahmin=("p", "mean"), gerçek=("hit", "mean"), n=("p", "size")).reset_index()
