"""Faz 2: XI (zaman sönümü) ve ridge ayarı.

Her (xi, ridge) çifti için walk-forward tahminleri bir kez üretilir. En iyi ayar SADECE ayar dönemindeki
maçlara (sezon <= tune_to) bakılarak seçilir (1X2 log loss). Seçilen ayar ile varsayılan ayar, seçimde hiç
kullanılmamış test dönemindeki aynı maçlarda eşleşmiş olarak kıyaslanır; böylece iyileşme tahmini iyimser
olmaz. Log loss maç sonucuna karşı ölçüldüğü için kapanış oranı gerektirmez.
"""
import contextlib
import io
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config
from src import db
from src.backtest import MARKETS, _ll_each, _md_table, _paired_diff, _sig, outcomes, walk_forward
from src.odds import devig_frame

XI_GRID = [0.0005, 0.001, 0.0015, 0.0019, 0.0025, 0.003, 0.004, 0.005]
RIDGE_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
KEYS = {"1X2": "1x2", "OU2.5": "ou"}

_DATA = {}  # her işçi süreçte bir kez yüklenir


def _init(divs):
    for div in divs:
        _DATA[div] = db.load_matches(div)


def _per_match(p: pd.DataFrame) -> pd.DataFrame:
    """Maç başına model ve keskin kapanış log loss'u (kapanış yoksa NaN)."""
    out = p[["date", "season", "home", "away"]].copy()
    for mk, spec in MARKETS.items():
        y = outcomes(p, mk)
        out[f"ll_{KEYS[mk]}"] = _ll_each(p[spec["model"]].to_numpy(dtype=float), y)
        close = devig_frame(p, [f"{config.CLOSING_BOOK}_{s}" for s in spec["suffix"]], config.DEVIG_METHOD)
        out[f"ll_close_{KEYS[mk]}"] = _ll_each(close, y)
    return out.reset_index(drop=True)


def _one(job):
    div, xi, ridge, eval_from = job
    with contextlib.redirect_stdout(io.StringIO()):
        preds = walk_forward(_DATA[div], eval_from, xi, ridge)
    return job, _per_match(preds)


def _half_life(xi: float) -> str:
    return "∞" if xi == 0 else f"{np.log(2) / xi:.0f} g"


def _grid_md(grid: pd.DataFrame, col: str, base: float, best: tuple) -> str:
    """xi satır, ridge sütun; hücre = (log loss - varsayılan) x 10^4. En iyi hücre kalın."""
    xis, ridges = sorted(grid["xi"].unique()), sorted(grid["ridge"].unique())
    rows = []
    for xi in xis:
        row = {"xi (yarı ömür)": f"{xi:g} ({_half_life(xi)})"}
        for r in ridges:
            v = grid.loc[(grid["xi"] == xi) & (grid["ridge"] == r), col].iloc[0]
            s = f"{(v - base) * 1e4:+.0f}"
            row[f"ridge {r:g}"] = f"**{s}**" if (xi, r) == best else s
        rows.append(row)
    return _md_table(pd.DataFrame(rows))


def _compare(a: pd.DataFrame, b: pd.DataFrame, tune_to: int) -> pd.DataFrame:
    """Test döneminde a (ayarlı) vs b (varsayılan), aynı maçlarda."""
    m = a.merge(b, on=["date", "home", "away"], suffixes=("", "_base"))
    test = m[m["season"] > tune_to]
    rows = []
    for mk, k in KEYS.items():
        d, se = _paired_diff(test[f"ll_{k}"], test[f"ll_{k}_base"])
        c = test[test[f"ll_close_{k}"].notna()]
        gb, sb = _paired_diff(c[f"ll_{k}_base"], c[f"ll_close_{k}"])
        ga, sa = _paired_diff(c[f"ll_{k}"], c[f"ll_close_{k}"])
        rows.append({"market": mk, "n": len(test), "varsayılan": test[f"ll_{k}_base"].mean(),
                     "ayarlı": test[f"ll_{k}"].mean(), "fark ± %95 GA": f"{d:+.4f} ± {1.96 * se:.4f}",
                     "sonuç": _sig(d, se).replace("model", "ayarlı"),
                     "kapanışa uzaklık (vars.)": f"{gb:+.4f} ± {1.96 * sb:.4f}",
                     "kapanışa uzaklık (ayarlı)": f"{ga:+.4f} ± {1.96 * sa:.4f}"})
    return pd.DataFrame(rows)


def run(divs: list, xi0: float, ridge0: float, eval_from: int, tune_to: int, workers: int,
        xi_grid: list | None = None, ridge_grid: list | None = None) -> tuple[str, pd.DataFrame]:
    xis = sorted(set(xi_grid or XI_GRID) | {xi0})
    ridges = sorted(set(ridge_grid or RIDGE_GRID) | {ridge0})
    jobs = [(div, xi, r, eval_from) for div in divs for xi in xis for r in ridges]
    print(f"  {len(jobs)} walk-forward, {workers} paralel işçi ...", flush=True)
    t0, res = time.time(), {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(divs,)) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            job, pm = f.result()
            res[job[:3]] = pm
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} ({time.time() - t0:.0f} sn)", flush=True)

    md = ["# XI / ridge ayarı", "",
          f"Walk-forward {eval_from}'dan itibaren. **Ayar dönemi:** {eval_from}–{tune_to} sezonları (seçim sadece "
          f"buna bakar). **Test dönemi:** {tune_to + 1} ve sonrası (seçimde hiç kullanılmadı). "
          f"Varsayılan: xi={xi0:g}, ridge={ridge0:g}. Seçim ölçütü: 1X2 log loss.", ""]
    all_rows = []
    for div in divs:
        rows = []
        for xi in xis:
            for r in ridges:
                pm = res[(div, xi, r)]
                tune, test = pm[pm["season"] <= tune_to], pm[pm["season"] > tune_to]
                rows.append({"div": div, "xi": xi, "ridge": r, "n_ayar": len(tune),
                             "ll_1x2_ayar": tune["ll_1x2"].mean(), "ll_ou_ayar": tune["ll_ou"].mean(),
                             "n_test": len(test), "ll_1x2_test": test["ll_1x2"].mean(),
                             "ll_ou_test": test["ll_ou"].mean()})
        grid = pd.DataFrame(rows)
        all_rows.append(grid)
        base = grid[(grid["xi"] == xi0) & (grid["ridge"] == ridge0)].iloc[0]
        b1 = grid.loc[grid["ll_1x2_ayar"].idxmin()]
        bo = grid.loc[grid["ll_ou_ayar"].idxmin()]
        best = (b1["xi"], b1["ridge"])
        edge = []  # 3'ten az değerli bir eksende "kenar" anlamsız
        if len(xis) >= 3 and best[0] in (xis[0], xis[-1]):
            edge.append("xi")
        if len(ridges) >= 3 and best[1] in (ridges[0], ridges[-1]):
            edge.append("ridge")

        md += [f"## {config.LEAGUES.get(div, div)} ({div})", "",
               f"**En iyi (ayar dönemi, 1X2):** xi={best[0]:g} (yarı ömür {_half_life(best[0])}), "
               f"ridge={best[1]:g} — ayar döneminde varsayılana göre 1X2 log loss "
               f"{b1['ll_1x2_ayar'] - base['ll_1x2_ayar']:+.4f}. "
               f"Ü/A için en iyi: xi={bo['xi']:g}, ridge={bo['ridge']:g}.", ""]
        if edge:
            md += [f"⚠ En iyi değer ızgaranın kenarında ({', '.join(edge)}); aralığı genişletmek gerekebilir.", ""]
        md += ["### Test döneminde (seçimde kullanılmadı) ayarlı vs varsayılan",
               _md_table(_compare(res[(div, *best)], res[(div, xi0, ridge0)], tune_to)), "",
               "Kapanışa uzaklık = model − keskin kapanış log loss farkı (0'a ne kadar yakınsa o kadar iyi).", "",
               "### Izgara — 1X2 log loss, ayar dönemi (varsayılana göre fark × 10⁻⁴, negatif = daha iyi)",
               _grid_md(grid, "ll_1x2_ayar", base["ll_1x2_ayar"], best), "",
               "### Izgara — Ü/A 2.5 log loss, ayar dönemi (varsayılana göre fark × 10⁻⁴)",
               _grid_md(grid, "ll_ou_ayar", base["ll_ou_ayar"], (bo["xi"], bo["ridge"])), ""]
    md.append(f"Süre: {time.time() - t0:.0f} sn")
    return "\n".join(md), pd.concat(all_rows, ignore_index=True)
