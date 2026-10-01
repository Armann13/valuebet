"""Faz 4: kağıt üzerinde canlı takip.

Her çalıştırmada (GitHub Actions veri güncellemesinden sonra ya da elle `python main.py track`):
  1) Kapanış: sonucu ve keskin kapanış oranı veriye düşmüş kayıtlar tamamlanır (sonuç, kapanış olasılığı, CLV).
  2) Kayıt: yaklaşan maçlar ilk görüldükleri anda kaydedilir; sonradan görülen oran değişiklikleri kaydı
     değiştirmez (gerçekte bahsi gördüğün anda oynarsın).
       tracking/predictions.csv  tüm maçların model olasılıkları + o anki B365 ve keskin açılış oranları
       tracking/bets.csv         backtest'teki üç kuralın kağıt bahisleri (B365 oranıyla, 1 birim)
  3) Özet: tracking/summary.md (canlı log loss model − kapanış, kural bazında CLV ve ROI).
Maçlar (lig, sezon, ev, deplasman) ile eşleştirilir; ertelenen maçın tarihi değişse de kayıt bulunur.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config
from src import db
from src.backtest import MARKETS, RULES, _ll_each, _md_table, _paired_diff, find_bets, outcomes
from src.odds import devig_frame

KEY = ["div", "season", "home", "away"]
SUFFIX = {"1X2": ["h", "d", "a"], "OU2.5": ["o25", "u25"]}
LABEL_IDX = {mk: {lab: i for i, lab in enumerate(spec["labels"])} for mk, spec in MARKETS.items()}
PRED_COLS = ["p_home", "p_draw", "p_away", "p_over25", "p_under25"]


def _paths():
    d = config.TRACKING_DIR
    return d / "predictions.csv", d / "bets.csv", d / "summary.md"


def _read(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()


def _new_predictions(fx: pd.DataFrame, now: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """fx: predict çıktısı (fikstür satırları). Döner: tahmin kayıtları ve üç kuralın bahisleri."""
    odds = [f"{b}_{s}" for b in ("b365", "sharp") for s in ("h", "d", "a", "o25", "u25")]
    preds = fx[KEY + ["date"] + PRED_COLS + ["mot"] * ("mot" in fx) + odds].copy()
    preds.insert(0, "model", config.MODEL_VERSION)
    preds.insert(0, "recorded_at", now)

    bets = []
    for rule in RULES:
        b = find_bets(fx, "b365", config.CLOSING_BOOK, config.DEVIG_METHOD, config.EDGE_MIN, config.MAX_ODDS,
                      config.MIN_TEAM_MATCHES, rule)
        if not b.empty:
            bets.append(b[KEY + ["date", "market", "pick", "price", "p_model"]].assign(**{f"kural_{rule}": True}))
    if not bets:
        return preds, pd.DataFrame()
    b = bets[0]
    for other in bets[1:]:
        b = b.merge(other, on=KEY + ["date", "market", "pick", "price", "p_model"], how="outer")
    for rule in RULES:
        b[f"kural_{rule}"] = b.get(f"kural_{rule}", pd.Series(False, index=b.index)).fillna(False).astype(bool)
    # bahis anındaki keskin açılış adil olasılığı
    fair = {mk: pd.DataFrame(devig_frame(fx, [f"sharp_{s}" for s in SUFFIX[mk]], config.DEVIG_METHOD),
                             index=fx.index) for mk in MARKETS}
    keyed = fx.set_index(KEY).index
    b["p_sharp"] = [fair[r.market].iloc[keyed.get_loc(tuple(getattr(r, k) for k in KEY)),
                                        LABEL_IDX[r.market][r.pick]] for r in b.itertuples()]
    b["edge_model"] = b["p_model"] * b["price"] - 1
    b["edge_keskin"] = b["p_sharp"] * b["price"] - 1
    b.insert(0, "model", config.MODEL_VERSION)
    b.insert(0, "recorded_at", now)
    return preds, b


def _append_new(old: pd.DataFrame, new: pd.DataFrame, key: list) -> tuple[pd.DataFrame, int]:
    if new.empty:
        return old, 0
    if old.empty:
        return new.reset_index(drop=True), len(new)
    seen = set(map(tuple, old[key].astype(str).to_numpy()))
    fresh = new[[tuple(r) not in seen for r in new[key].astype(str).to_numpy()]]
    return pd.concat([old, fresh], ignore_index=True), len(fresh)


def _results_close(results: pd.DataFrame) -> pd.DataFrame:
    """Sonucu belli maçlar: KEY + gol + keskin kapanış adil olasılıkları (close_h, close_d, ...)."""
    res = results.dropna(subset=["fthg", "ftag"]).copy()
    for mk in MARKETS:
        f = devig_frame(res, [f"sharpc_{s}" for s in SUFFIX[mk]], config.DEVIG_METHOD)
        for i, s in enumerate(SUFFIX[mk]):
            res[f"close_{s}"] = f[:, i]
    res = res[KEY + ["fthg", "ftag"] + [f"close_{s}" for mk in MARKETS for s in SUFFIX[mk]]]
    res["season"] = res["season"].astype(int)
    return res.drop_duplicates(subset=KEY, keep="last")  # anahtar tekrar ederse satırlar çoğalmasın


def _settle(preds: pd.DataFrame, bets: pd.DataFrame, res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sonucu belli maçlar için (res = _results_close) sonuç, keskin kapanış adil olasılıkları, CLV ve kâr."""
    fill = lambda df: df.drop(columns=[c for c in res.columns if c not in KEY and c in df]).merge(  # noqa: E731
        res, on=KEY, how="left")
    if not preds.empty:
        preds = fill(preds)
    if not bets.empty:
        bets = fill(bets)
        won, p_close = [], []
        for r in bets.itertuples():
            if pd.isna(r.fthg):
                won.append(np.nan)
                p_close.append(np.nan)
                continue
            y = outcomes(pd.DataFrame({"fthg": [r.fthg], "ftag": [r.ftag]}), r.market)[0]
            j = LABEL_IDX[r.market][r.pick]
            won.append(float(y[j]))
            p_close.append(getattr(r, f"close_{SUFFIX[r.market][j]}"))
        bets["won"], bets["p_close"] = won, p_close
        bets["clv"] = bets["price"] * bets["p_close"] - 1
        bets["profit"] = np.where(bets["won"].isna(), np.nan, np.where(bets["won"] == 1, bets["price"] - 1, -1.0))
    return preds, bets


IDDAA_OUTCOMES = [("1X2", "1", "odd_1", "h", "p_home"), ("1X2", "X", "odd_x", "d", "p_draw"),
                  ("1X2", "2", "odd_2", "a", "p_away"), ("OU2.5", "Üst", "odd_ust", "o25", "p_over25"),
                  ("OU2.5", "Alt", "odd_alt", "u25", "p_under25")]


def _iddaa_clv(preds: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """İddaa oran geçmişinden her maç-sonuç için ilk ve son (başlama öncesi) oran; maç sonuçlanmışsa (res)
    keskin kapanışa göre CLV; tahmin kaydı varsa model olasılığı ve keskin açılış adil olasılığı."""
    from src import iddaa
    odds = iddaa.load_odds()
    if odds.empty:
        return pd.DataFrame()
    odds = odds.dropna(subset=["home", "away"]).sort_values("snapshot_utc")
    odds = odds[odds["div"] != config.MILLI]  # milli maçlarda keskin kapanış yok: CLV ölçülemez
    odds["season"] = odds["season"].astype(int)
    g = odds.groupby(KEY)
    first, last = g.head(1).set_index(KEY), g.tail(1).set_index(KEY)
    n_snap = g.size()
    p = preds.set_index(KEY) if not preds.empty else pd.DataFrame()
    sharp = {mk: pd.DataFrame(devig_frame(preds, [f"sharp_{s}" for s in SUFFIX[mk]], config.DEVIG_METHOD),
                              index=p.index) for mk in MARKETS} if not preds.empty else {}
    r_ = res.set_index(KEY)
    rows = []
    for key in first.index:
        f, l = first.loc[key], last.loc[key]
        pr = p.loc[key] if key in p.index else None
        rr = r_.loc[key] if key in r_.index else None
        for mk, pick, col, suf, pcol in IDDAA_OUTCOMES:
            if mk == "OU2.5" and not (f.get("ou_line") == 2.5 and l.get("ou_line") == 2.5):
                continue
            j = SUFFIX[mk].index(suf)
            rows.append({**dict(zip(KEY, key)), "kickoff_utc": l["kickoff_utc"], "market": mk, "pick": pick,
                         "mbs": l.get("mbs"), "kayıt_sayısı": int(n_snap.loc[key]), "odd_ilk": f[col], "odd_son": l[col],
                         "p_model": pr[pcol] if pr is not None else np.nan,
                         "p_keskin_açılış": sharp[mk].loc[key].iloc[j] if pr is not None else np.nan,
                         "fthg": rr["fthg"] if rr is not None else np.nan,
                         "ftag": rr["ftag"] if rr is not None else np.nan,
                         "p_close": rr[f"close_{suf}"] if rr is not None else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["edge_model"] = t["p_model"] * t["odd_ilk"] - 1
    t["edge_keskin"] = t["p_keskin_açılış"] * t["odd_ilk"] - 1
    t["clv_ilk"] = t["odd_ilk"] * t["p_close"] - 1
    t["clv_son"] = t["odd_son"] * t["p_close"] - 1
    won = []
    for r in t.itertuples():
        won.append(np.nan if pd.isna(r.fthg) else
                   float(outcomes(pd.DataFrame({"fthg": [r.fthg], "ftag": [r.ftag]}), r.market)[0][LABEL_IDX[r.market][r.pick]]))
    t["won"] = won
    t["profit_ilk"] = np.where(t["won"].isna(), np.nan, np.where(t["won"] == 1, t["odd_ilk"] - 1, -1.0))
    return t


def _iddaa_summary(t: pd.DataFrame) -> list:
    md = ["## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)", ""]
    if t.empty:
        return md + ["Henüz İddaa oranı kaydı eşleşen maç yok.", ""]
    s = t.dropna(subset=["clv_ilk"])
    groups = [("Tüm İddaa fiyatları", s),
              ("İddaa > keskin açılış adil fiyatı", s[s["edge_keskin"] > 0]),
              (f"Model edge ≥ %{100 * config.EDGE_MIN:.0f} (onaysız)", s[s["edge_model"] >= config.EDGE_MIN]),
              ("Model edge + keskin onay (VALUE)", s[(s["edge_model"] >= config.EDGE_MIN) & (s["edge_keskin"] > 0)])]
    rows = []
    for name, g in groups:
        rows.append({"grup": name, "sonuç sayısı": len(g), "maç": g[KEY].drop_duplicates().shape[0],
                     "ort. oran": g["odd_ilk"].mean() if len(g) else np.nan,
                     "CLV ilk oran_%": 100 * g["clv_ilk"].mean() if len(g) else np.nan,
                     "±GA_%": 196 * g["clv_ilk"].std() / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                     "CLV son oran_%": 100 * g["clv_son"].mean() if len(g) else np.nan,
                     "CLV>0_%": 100 * (g["clv_ilk"] > 0).mean() if len(g) else np.nan,
                     "ROI_%": 100 * g["profit_ilk"].mean() if len(g) else np.nan})
    open_n = t[t["clv_ilk"].isna()][KEY].drop_duplicates().shape[0]
    return md + [_md_table(pd.DataFrame(rows), "{:.2f}"), "",
                 f"Sonuçlanmamış (açık) maç: {open_n}. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan "
                 "önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv", ""]


def _summary(preds: pd.DataFrame, bets: pd.DataFrame) -> str:
    last = preds["recorded_at"].max() if not preds.empty else "—"  # zaman damgası sadece veri değişince değişsin
    md = ["# Canlı takip (kağıt üzerinde)", "",
          f"Son kayıt: {last} UTC. Model: {config.MODEL_VERSION}. Bahisler B365 oranıyla, 1 birim.",
          "Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.", ""]
    rows = []
    if not preds.empty:
        done = preds.dropna(subset=["fthg"])
        for div, g in [*done.groupby("div"), ("toplam", done)]:
            row = {"lig": div, "kayıtlı": int((preds["div"] == div).sum()) if div != "toplam" else len(preds),
                   "sonuçlanan": len(g)}
            for mk, spec in MARKETS.items():
                cc = [f"close_{s}" for s in SUFFIX[mk]]
                h = g.dropna(subset=cc)
                if len(h) >= 2:
                    y = outcomes(h, mk)
                    d, se = _paired_diff(_ll_each(h[spec["model"]].to_numpy(float), y),
                                         _ll_each(h[cc].to_numpy(float), y))
                    row[f"{mk} log loss (model − kapanış)"] = f"{d:+.4f} ± {1.96 * se:.4f} (n={len(h)})"
                else:
                    row[f"{mk} log loss (model − kapanış)"] = "—"
            rows.append(row)
    md += ["## Tahmin kalitesi", _md_table(pd.DataFrame(rows)) if rows else "Henüz kayıt yok.", ""]

    rows = []
    if not bets.empty:
        for rule, label in RULES.items():
            g = bets[bets[f"kural_{rule}"]]
            s = g.dropna(subset=["won"])
            c = s["clv"].dropna()
            rows.append({"kural": label, "bahis": len(g), "açık": len(g) - len(s), "sonuçlanan": len(s),
                         "ort. oran": g["price"].mean() if len(g) else np.nan,
                         "ROI_%": 100 * s["profit"].mean() if len(s) else np.nan,
                         "CLV_ort_%": 100 * c.mean() if len(c) else np.nan,
                         "CLV ±GA_%": 196 * c.std() / np.sqrt(len(c)) if len(c) > 1 else np.nan,
                         "CLV>0_%": 100 * (c > 0).mean() if len(c) else np.nan})
    md += ["## Kağıt bahisler", _md_table(pd.DataFrame(rows), "{:.2f}") if rows else "Henüz bahis yok.", "",
           "Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması)."]
    return "\n".join(md)


def run(upcoming: pd.DataFrame | None) -> str:
    """upcoming: predict.run çıktısı (kaynak == fixtures satırları) ya da None."""
    pred_path, bet_path, sum_path = _paths()
    config.TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    preds, bets = _read(pred_path), _read(bet_path)
    n_p = n_b = 0
    if upcoming is not None and not upcoming.empty:
        # sadece henüz başlamamış maçlar (saat İngiltere saati; UTC sayılır, en fazla 1 saat sapar)
        day = pd.to_datetime(upcoming["date"]).dt.strftime("%Y-%m-%d")
        time = upcoming["time"].fillna("23:59").astype(str) if "time" in upcoming else "23:59"
        kick = pd.to_datetime(day + " " + time, errors="coerce").fillna(pd.to_datetime(day) + pd.Timedelta("23:59:00"))
        fx = upcoming[kick > pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))].reset_index(drop=True)
        if not fx.empty:
            new_p, new_b = _new_predictions(fx, now)
            preds, n_p = _append_new(preds, new_p, KEY)
            bets, n_b = _append_new(bets, new_b, KEY + ["market", "pick"])
    res = _results_close(db.load_matches())
    preds, bets = _settle(preds, bets, res)
    if not preds.empty:
        preds.to_csv(pred_path, index=False)
    if not bets.empty:
        bets.to_csv(bet_path, index=False)
    md = _summary(preds, bets)
    ic = _iddaa_clv(preds, res)
    if not ic.empty:
        ic.to_csv(config.TRACKING_DIR / "iddaa_clv.csv", index=False)
    md += "\n\n" + "\n".join(_iddaa_summary(ic))
    sum_path.write_text(md, encoding="utf-8")
    settled = int(preds["fthg"].notna().sum()) if not preds.empty else 0
    print(f"  yeni tahmin: {n_p}, yeni bahis: {n_b}, sonuçlanan tahmin: {settled}")
    return md
