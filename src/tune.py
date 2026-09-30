"""Faz 2: model ayarı (xi = zaman sönümü, ridge = güç yönünde çekim, ridge_total = eğilim yönünde çekim).

Her ayar için walk-forward tahminleri bir kez üretilir. En iyi ayar SADECE ayar dönemindeki maçlara
(sezon <= tune_to) bakılarak seçilir. Seçilen ayar ile varsayılan ayar, seçimde hiç kullanılmamış test
dönemindeki aynı maçlarda eşleşmiş olarak kıyaslanır; böylece iyileşme tahmini iyimser olmaz. Log loss maç
sonucuna karşı ölçüldüğü için kapanış oranı gerektirmez. Bir çalıştırmada en fazla iki eksen değişebilir.
"""
import contextlib
import io
import itertools
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
AXES = ["xi", "ridge", "ridge_total"]
AXIS_NAMES = {"xi": "xi (yarı ömür)", "ridge": "ridge (güç)", "ridge_total": "ridge_total (eğilim)"}
CRITERIA = {"1x2": ("ll_1x2", "1X2 log loss"), "ou": ("ll_ou", "Ü/A 2.5 log loss"),
            "toplam": ("ll_sum", "1X2 + Ü/A log loss toplamı")}

_DATA = {}  # her işçi süreçte bir kez yüklenir


def _init(divs, joint):
    if joint:  # birleşik modelde eğitim her zaman tüm liglerle; değerlendirme divs ile sınırlanır
        _DATA["ALL"] = db.load_matches()
    else:
        for div in divs:
            _DATA[div] = db.load_matches(div)


def _per_match(p: pd.DataFrame) -> pd.DataFrame:
    """Maç başına model ve keskin kapanış log loss'u (kapanış yoksa NaN)."""
    out = p[["div", "date", "season", "home", "away"]].copy()
    for mk, spec in MARKETS.items():
        y = outcomes(p, mk)
        out[f"ll_{KEYS[mk]}"] = _ll_each(p[spec["model"]].to_numpy(dtype=float), y)
        close = devig_frame(p, [f"{config.CLOSING_BOOK}_{s}" for s in spec["suffix"]], config.DEVIG_METHOD)
        out[f"ll_close_{KEYS[mk]}"] = _ll_each(close, y)
    out["ll_sum"] = out["ll_1x2"] + out["ll_ou"]
    out["ll_close_sum"] = out["ll_close_1x2"] + out["ll_close_ou"]
    return out.reset_index(drop=True)


def _one(job):
    div, xi, ridge, ridge_total, eval_from = job
    with contextlib.redirect_stdout(io.StringIO()):
        preds = walk_forward(_DATA[div], eval_from, xi, ridge, ridge_total=ridge_total)
    return job, _per_match(preds)


def _half_life(xi: float) -> str:
    return "∞" if xi == 0 else f"{np.log(2) / xi:.0f} g"


def _label(axis: str, v: float) -> str:
    return f"{v:g} ({_half_life(v)})" if axis == "xi" else f"{v:g}"


def _fmt_rt(rt) -> str:
    return "=ridge" if rt is None else f"{rt:g}"


def _key_str(key) -> str:
    xi, r, rt = key
    return f"xi={xi:g} (yarı ömür {_half_life(xi)}), ridge={r:g}, ridge_total={_fmt_rt(rt)}"


def _grid_md(grid: pd.DataFrame, rows_ax: str, cols_ax: str, col: str, base: float) -> str:
    """Hücre = (log loss - varsayılan) x 10^4. Tablonun en iyi hücresi kalın."""
    best = grid.loc[grid[col].idxmin(), "key"]
    out = []
    for rv in sorted(grid[rows_ax].unique()):
        row = {AXIS_NAMES[rows_ax]: _label(rows_ax, rv)}
        for cv in sorted(grid[cols_ax].unique()):
            cell = grid[(grid[rows_ax] == rv) & (grid[cols_ax] == cv)]
            s = f"{(cell[col].iloc[0] - base) * 1e4:+.0f}" if len(cell) else ""
            row[f"{cols_ax} {_label(cols_ax, cv)}"] = f"**{s}**" if len(cell) and cell["key"].iloc[0] == best else s
        out.append(row)
    return _md_table(pd.DataFrame(out))


def _compare(a: pd.DataFrame, b: pd.DataFrame, tune_to: int) -> pd.DataFrame:
    """Test döneminde a (ayarlı) vs b (varsayılan), aynı maçlarda."""
    m = a.merge(b, on=["date", "home", "away"], suffixes=("", "_base"))
    test = m[m["season"] > tune_to]
    rows = []
    for mk, k in [*KEYS.items(), ("toplam", "sum")]:
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


def run(divs: list, xi0: float, ridge0: float, rt0: float | None, eval_from: int, tune_to: int, workers: int,
        xi_grid: list | None = None, ridge_grid: list | None = None, rt_grid: list | None = None,
        criterion: str = "toplam", joint: bool = False) -> tuple[str, pd.DataFrame]:
    """rt_grid verilmezse ridge_total varsayılanda (rt0; None = ridge ile aynı) sabit kalır."""
    xis = sorted(set(xi_grid or XI_GRID) | {xi0})
    ridges = sorted(set(ridge_grid or RIDGE_GRID) | {ridge0})
    # ridge_total ızgarası sayısal olunca varsayılan (None = ridge ile aynı) sayıya çevrilir; model aynıdır
    rt_base = (ridge0 if rt0 is None else rt0) if rt_grid else rt0
    rts = sorted(set(rt_grid) | {rt_base}) if rt_grid else [rt0]
    base_key = (xi0, ridge0, rt_base)
    varying = [ax for ax, vals in zip(AXES, (xis, ridges, rts)) if len(vals) > 1]
    if len(varying) > 2:
        raise SystemExit("En fazla iki eksen değişebilir; birini sabitle (ör. --xi-grid 0.0019).")
    rows_ax, cols_ax = (varying + [a for a in AXES if a not in varying])[:2]
    crit_col, crit_name = CRITERIA[criterion]

    keys = ["ALL"] if joint else divs
    jobs = [(key, xi, r, rt, eval_from) for key in keys for xi, r, rt in itertools.product(xis, ridges, rts)]
    print(f"  {len(jobs)} walk-forward, {workers} paralel işçi ...", flush=True)
    t0, res = time.time(), {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(divs, joint)) as ex:
        futures = [ex.submit(_one, j) for j in jobs]
        for k, f in enumerate(as_completed(futures), 1):
            job, pm = f.result()
            for div in (divs if job[0] == "ALL" else [job[0]]):
                res[(div, *job[1:4])] = pm[pm["div"] == div].reset_index(drop=True)
            if k % 10 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)} ({time.time() - t0:.0f} sn)", flush=True)

    md = ["# Model ayarı (xi / ridge / ridge_total)", "",
          f"Walk-forward {eval_from}'dan itibaren. **Ayar dönemi:** {eval_from}–{tune_to} sezonları (seçim sadece "
          f"buna bakar). **Test dönemi:** {tune_to + 1} ve sonrası (seçimde hiç kullanılmadı). "
          f"Varsayılan: xi={xi0:g}, ridge={ridge0:g}, ridge_total={_fmt_rt(rt0)}. Seçim ölçütü: {crit_name}. "
          f"Model: {'birleşik (tüm ligler)' if joint else 'lige özel'}.", ""]
    all_rows = []
    for div in divs:
        rows = []
        for xi, r, rt in itertools.product(xis, ridges, rts):
            pm = res[(div, xi, r, rt)]
            tune, test = pm[pm["season"] <= tune_to], pm[pm["season"] > tune_to]
            rows.append({"div": div, "key": (xi, r, rt), "xi": xi, "ridge": r, "ridge_total": r if rt is None else rt,
                         "n_ayar": len(tune), "n_test": len(test),
                         **{f"{c}_ayar": tune[c].mean() for c in ("ll_1x2", "ll_ou", "ll_sum")},
                         **{f"{c}_test": test[c].mean() for c in ("ll_1x2", "ll_ou", "ll_sum")}})
        grid = pd.DataFrame(rows)
        all_rows.append(grid.drop(columns="key"))
        base = grid[grid["key"].map(lambda k: k == base_key)].iloc[0]
        best = grid.loc[grid[f"{crit_col}_ayar"].idxmin()]
        edge = [ax for ax, vals in zip(AXES, (xis, ridges, rts))
                if len(vals) >= 3 and best[ax] in (min(vals), max(vals))]
        others = "; ".join(f"sadece {name} için en iyi: {_key_str(grid.loc[grid[c + '_ayar'].idxmin(), 'key'])}"
                           for c, name in [("ll_1x2", "1X2"), ("ll_ou", "Ü/A")])
        gain = best[f"{crit_col}_ayar"] - base[f"{crit_col}_ayar"]
        md += [f"## {config.LEAGUES.get(div, div)} ({div})", "",
               f"**En iyi (ayar dönemi, {crit_name}):** {_key_str(best['key'])} — ayar döneminde varsayılana "
               f"göre {gain:+.4f}.", "", f"({others})", ""]
        if edge:
            md += [f"⚠ En iyi değer ızgaranın kenarında ({', '.join(edge)}); aralığı genişletmek gerekebilir.", ""]
        md += ["### Test döneminde (seçimde kullanılmadı) ayarlı vs varsayılan",
               _md_table(_compare(res[(div, *best["key"])], res[(div, *base_key)], tune_to)), "",
               "Kapanışa uzaklık = model − keskin kapanış log loss farkı (0'a ne kadar yakınsa o kadar iyi).", ""]
        for c, name in [("ll_sum", "1X2 + Ü/A toplamı"), ("ll_1x2", "1X2"), ("ll_ou", "Ü/A 2.5")]:
            md += [f"### Izgara — {name} log loss, ayar dönemi (varsayılana göre fark × 10⁻⁴, negatif = daha iyi)",
                   _grid_md(grid, rows_ax, cols_ax, f"{c}_ayar", base[f"{c}_ayar"]), ""]
    md.append(f"Süre: {time.time() - t0:.0f} sn")
    return "\n".join(md), pd.concat(all_rows, ignore_index=True)
