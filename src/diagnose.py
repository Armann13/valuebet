"""Açık analizi: modelin keskin kapanışa göre log loss açığı hangi maçlarda yoğunlaşıyor?

Son backtest'in tahminleri (reports/backtest_preds_{lig}.csv) dilimlere ayrılır: sezon içindeki tur, lige
yeni gelen takım, dinlenme, favorinin gücü, sezon. base_dir verilirse aynı maçlarda baz tahminlerle eşleşmiş
kıyas da yapılır; böylece bir model değişikliğinin hangi dilimde işe yaradığı görülür.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import db
from src.backtest import MARKETS, _ll_each, _md_table, outcomes
from src.odds import devig_frame

KEYS = {"1X2": "1x2", "OU2.5": "ou"}


def _losses(p: pd.DataFrame) -> pd.DataFrame:
    out = p[["div", "date", "season", "home", "away"]].copy()
    for mk, spec in MARKETS.items():
        y = outcomes(p, mk)
        close = devig_frame(p, [f"{config.CLOSING_BOOK}_{s}" for s in spec["suffix"]], config.DEVIG_METHOD)
        out[f"ll_{KEYS[mk]}"] = _ll_each(p[spec["model"]].to_numpy(dtype=float), y)
        out[f"gap_{KEYS[mk]}"] = out[f"ll_{KEYS[mk]}"] - _ll_each(close, y)
        if mk == "1X2":
            out["fav_p"] = close.max(axis=1)
    return out


def _context(all_matches: pd.DataFrame) -> tuple[dict, dict]:
    """(lig, sezon) -> takımlar ve (takım, tarih) -> önceki lig maçından bu yana gün."""
    teams = {}
    for r in all_matches[["div", "season", "home", "away"]].itertuples():
        teams.setdefault((r.div, r.season), set()).update((r.home, r.away))
    long = pd.concat([all_matches[["date", "home"]].set_axis(["date", "team"], axis=1),
                      all_matches[["date", "away"]].set_axis(["date", "team"], axis=1)]).sort_values("date")
    long["rest"] = (long["date"] - long.groupby("team")["date"].shift()).dt.days
    return teams, {(t, d): r for t, d, r in zip(long["team"], long["date"], long["rest"])}


def _origin(teams: dict, div: str, season: int, team: str) -> str:
    if team in teams.get((div, season - 1), set()):
        return "eski"
    if any(team in teams.get((other, season - 1), set()) for other in config.LEAGUES if other != div):
        return "diğer ligden"
    return "veride yok"


def _segments(d: pd.DataFrame, teams: dict, rest: dict) -> pd.DataFrame:
    d = d.sort_values("date").copy()
    long = pd.concat([d[["season", "date", "home"]].set_axis(["season", "date", "team"], axis=1),
                      d[["season", "date", "away"]].set_axis(["season", "date", "team"], axis=1)]).sort_values("date")
    long["n"] = long.groupby(["season", "team"]).cumcount() + 1
    md = {(s, t, dt): n for s, t, dt, n in zip(long["season"], long["team"], long["date"], long["n"])}
    tur = np.array([min(md[(s, h, dt)], md[(s, a, dt)]) for s, h, a, dt in zip(d["season"], d["home"], d["away"], d["date"])])
    d["tur"] = pd.cut(tur, [0, 3, 6, 10, 19, 29, 99], labels=["1-3", "4-6", "7-10", "11-19", "20-29", "30+"])
    orig = [{_origin(teams, dv, s, h), _origin(teams, dv, s, a)} - {"eski"}
            for dv, s, h, a in zip(d["div"], d["season"], d["home"], d["away"])]
    d["yeni takım"] = ["diğer ligden gelen var" if "diğer ligden" in o else "alt ligden (veride yok) gelen var" if o
                       else "ikisi de eski" for o in orig]
    d["yeni takım x dönem"] = [("ikisi de eski" if y == "ikisi de eski" else "yeni var") +
                               (" / tur 1-10" if t <= 10 else " / tur 11+") for y, t in zip(d["yeni takım"], tur)]
    short = [(rest.get((h, dt), np.nan) <= 4) or (rest.get((a, dt), np.nan) <= 4)
             for h, a, dt in zip(d["home"], d["away"], d["date"])]
    d["dinlenme"] = np.where(short, "biri ≤4 gün", "ikisi de 5+ gün")
    d["favori (kapanış)"] = pd.cut(d["fav_p"], [0, 0.45, 0.6, 0.75, 1], labels=["<%45", "%45-60", "%60-75", ">%75"])
    d["sezon"] = d["season"].astype(str)
    d["genel"] = "tüm maçlar"
    return d


def _table(d: pd.DataFrame, by: str, has_base: bool) -> pd.DataFrame:
    rows = []
    for (div, key), g in d.groupby(["div", by], observed=True, sort=True):
        tot = d.loc[d["div"] == div, "gap_1x2"].sum()
        row = {"lig": div, by: key, "n": len(g), "maç_%": f"{100 * len(g) / (d['div'] == div).sum():.1f}",
               "1X2 açık/maç": g["gap_1x2"].mean(), "±GA": 1.96 * g["gap_1x2"].std(ddof=1) / np.sqrt(len(g)),
               "1X2 açığın_%": f"{100 * g['gap_1x2'].sum() / tot:.1f}", "Ü/A açık/maç": g["gap_ou"].mean()}
        if has_base:
            for k, lab in KEYS.items():
                diff = g[f"ll_{lab}"] - g[f"ll_{lab}_base"]
                row[f"{k} fark (şimdi−baz)"] = f"{diff.mean():+.4f} ± {1.96 * diff.std(ddof=1) / np.sqrt(len(g)):.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def run(divs: list, base_dir: str | None = None) -> str:
    teams, rest = _context(db.load_matches())
    frames = []
    for div in divs:
        f = config.REPORTS_DIR / f"backtest_preds_{div}.csv"
        if not f.exists():
            raise SystemExit(f"{f} yok; önce: python main.py backtest")
        cur = _losses(pd.read_csv(f, parse_dates=["date"], low_memory=False))
        if base_dir:
            base = _losses(pd.read_csv(Path(base_dir) / f"backtest_preds_{div}.csv", parse_dates=["date"],
                                       low_memory=False))
            cur = cur.merge(base[["date", "home", "away", "ll_1x2", "ll_ou"]], on=["date", "home", "away"],
                            suffixes=("", "_base"))
        frames.append(cur[np.isfinite(cur["gap_1x2"])])  # Ü/A kapanışı 2019'dan önce yok; Ü/A sütunları NaN atlar
    d = _segments(pd.concat(frames, ignore_index=True), teams, rest)

    md = ["# Açık analizi (model − keskin kapanış log loss)", "",
          "Açık/maç: modelin kapanıştan ne kadar kötü olduğu (0 = kapanış kadar iyi). Açığın_%: toplam 1X2 "
          "açığının yüzde kaçının bu dilimden geldiği; maç_%'sinden büyükse açık bu dilimde yoğunlaşıyor."]
    if base_dir:
        md.append(f"Fark (şimdi−baz): aynı maçlarda şimdiki tahminler ile `{base_dir}` tahminleri arasındaki log "
                  "loss farkı; negatif = şimdiki daha iyi.")
    md.append("")
    for by in ["genel", "yeni takım", "yeni takım x dönem", "tur", "dinlenme", "favori (kapanış)", "sezon"]:
        md += [f"## {by}", _md_table(_table(d, by, bool(base_dir)), "{:.4f}"), ""]
    return "\n".join(md)
