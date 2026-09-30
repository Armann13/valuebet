"""Walk-forward backtest.

Her hafta için model SADECE o haftadan önceki maçlarla yeniden eğitilir ve o haftanın maçları tahmin
edilir (geleceği görme / sızıntı yok). Sonra:
  1) Tahmin kalitesi: model vs keskin kapanış oranları (Pinnacle, yoksa Betfair borsası; log loss, Brier,
     RPS, kalibrasyon)
  2) Value simülasyonu: edge >= eşik olan bahisler, düz bahis ROI, çeyrek Kelly kasa, CLV
"""
import numpy as np
import pandas as pd

import config
from src.dixon_coles import DixonColes
from src.odds import devig_frame

MARKETS = {
    "1X2":   {"labels": ["1", "X", "2"], "model": ["p_home", "p_draw", "p_away"], "suffix": ["h", "d", "a"]},
    "OU2.5": {"labels": ["Üst", "Alt"], "model": ["p_over25", "p_under25"], "suffix": ["o25", "u25"]},
}


def outcomes(df: pd.DataFrame, market: str) -> np.ndarray:
    hg, ag = df["fthg"].to_numpy(), df["ftag"].to_numpy()
    if market == "1X2":
        return np.column_stack([hg > ag, hg == ag, hg < ag]).astype(float)
    tot = hg + ag
    return np.column_stack([tot > 2.5, tot < 2.5]).astype(float)


# ---------------------------------------------------------------------------- walk-forward
def walk_forward(df: pd.DataFrame, eval_from: int, xi: float, ridge: float,
                 lookback: int = config.LOOKBACK_DAYS, ridge_total: float | None = None) -> pd.DataFrame:
    """df'te birden çok lig varsa her hafta tek bir birleşik model eğitilir (bkz. DixonColes)."""
    df = df.sort_values("date").reset_index(drop=True)
    ev = df[df["season"] >= eval_from].copy()
    iso = ev["date"].dt.isocalendar()
    ev["round_key"] = iso["year"].astype(str) + "-" + iso["week"].astype(str).str.zfill(2)
    rows, prev, cur_season = [], None, None
    for _, g in ev.groupby("round_key", sort=True):
        season = int(g["season"].iloc[0])
        if season != cur_season:
            cur_season = season
            print(f"    sezon {season}/{(season + 1) % 100:02d} ...", flush=True)
        try:
            model = DixonColes(xi=xi, ridge=ridge, ridge_total=ridge_total, lookback_days=lookback).fit(
                df, ref_date=g["date"].min(), warm_start=prev)
        except ValueError:
            continue
        prev = model
        for rec in g.to_dict("records"):
            rows.append({**rec, **model.predict(rec["home"], rec["away"], rec.get("div"))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- metrikler
def _ll_each(p, y):
    return -np.log(np.clip((p * y).sum(axis=1), 1e-12, 1))


def _log_loss(p, y):
    return float(np.mean(_ll_each(p, y)))


def _paired_diff(a, b) -> tuple[float, float]:
    """Aynı maçlarda maç başına fark (a - b): ortalama ve standart hata. İki tahminciyi aynı maçlar
    üzerinde kıyasladığı için ayrı ayrı log loss'lara bakmaktan çok daha hassastır."""
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def _sig(mean: float, se: float) -> str:
    if mean - 1.96 * se > 0:
        return "model anlamlı derecede kötü"
    if mean + 1.96 * se < 0:
        return "model anlamlı derecede iyi"
    return "fark anlamlı değil"


def _brier(p, y):
    return float(np.mean(((p - y) ** 2).sum(axis=1)))


def _rps(p, y):
    cp, cy = np.cumsum(p, axis=1)[:, :-1], np.cumsum(y, axis=1)[:, :-1]
    return float(np.mean(((cp - cy) ** 2).sum(axis=1) / (p.shape[1] - 1)))


def evaluate(preds: pd.DataFrame, bet_book: str, closing_book: str, method: str) -> dict:
    out = {}
    for mk, spec in MARKETS.items():
        pm = preds[spec["model"]].to_numpy(dtype=float)
        pc = devig_frame(preds, [f"{closing_book}_{s}" for s in spec["suffix"]], method)
        pb = devig_frame(preds, [f"{bet_book}_{s}" for s in spec["suffix"]], method)
        ok = np.isfinite(pc).all(axis=1) & np.isfinite(pb).all(axis=1)
        if ok.sum() < 20:
            continue
        y = outcomes(preds, mk)[ok]
        table = []
        for name, p in [("Model (Dixon-Coles)", pm[ok]), (f"{config.book_label(bet_book)} (marjsız)", pb[ok]),
                        (f"{config.book_label(closing_book)} (marjsız)", pc[ok])]:
            row = {"kaynak": name, "log_loss": _log_loss(p, y), "brier": _brier(p, y)}
            if mk == "1X2":
                row["rps"] = _rps(p, y)
            table.append(row)
        ll_model = _ll_each(pm[ok], y)
        src_col = f"{closing_book}_src_{'1x2' if mk == '1X2' else 'ou'}"
        out[mk] = {"n": int(ok.sum()), "table": pd.DataFrame(table),
                   "close_sources": preds.loc[ok, src_col].value_counts().to_dict() if src_col in preds else {},
                   "mean_abs_diff_vs_close": float(np.mean(np.abs(pm[ok] - pc[ok]))),
                   "vs_bet": _paired_diff(ll_model, _ll_each(pb[ok], y)),
                   "vs_close": _paired_diff(ll_model, _ll_each(pc[ok], y))}
    return out


def calibration(preds: pd.DataFrame, bins=np.linspace(0, 1, 11)) -> pd.DataFrame:
    spec = MARKETS["1X2"]
    p = preds[spec["model"]].to_numpy(dtype=float).ravel()
    y = outcomes(preds, "1X2").ravel()
    b = np.clip(np.digitize(p, bins) - 1, 0, len(bins) - 2)
    rows = []
    for i in range(len(bins) - 1):
        m = b == i
        if m.sum() >= 10:
            rows.append({"aralık": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "n": int(m.sum()),
                         "tahmin_ort": p[m].mean(), "gerçekleşen": y[m].mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- value simülasyonu
# Bahis seçme kuralları. "sharp" modeli hiç kullanmaz: sadece bahis anında bilinen keskin açılış adil
# fiyatını (Pinnacle, yoksa Betfair borsası) oynanan oranla kıyaslar. Model ancak bu kuraldan daha iyi CLV
# veriyorsa bilgi katıyordur.
RULES = {
    "model": "Model",
    "sharp": "Keskin açılış adil fiyatı (model yok)",
    "both": "Model + keskin piyasa onayı (predict'teki VALUE kuralı)",
}


def find_bets(preds: pd.DataFrame, bet_book: str, closing_book: str, method: str,
              edge_min: float, max_odds: float, min_matches: int, rule: str = "model") -> pd.DataFrame:
    bets = []
    for mk, spec in MARKETS.items():
        pm = preds[spec["model"]].to_numpy(dtype=float)
        price = preds[[f"{bet_book}_{s}" for s in spec["suffix"]]].to_numpy(dtype=float)
        pc = devig_frame(preds, [f"{closing_book}_{s}" for s in spec["suffix"]], method)
        ps = devig_frame(preds, [f"sharp_{s}" for s in spec["suffix"]], method)
        y = outcomes(preds, mk)
        edge_sharp = ps * price - 1
        edge = edge_sharp.copy() if rule == "sharp" else pm * price - 1
        edge[~np.isfinite(edge) | (price > max_odds)] = -np.inf
        data_ok = (preds["n_home"] >= min_matches) & (preds["n_away"] >= min_matches)
        for i in np.where(data_ok.to_numpy())[0]:
            j = int(np.argmax(edge[i]))
            if edge[i, j] < edge_min:
                continue
            if rule == "both" and not edge_sharp[i, j] > 0:
                continue
            r = preds.iloc[i]
            bets.append({
                "date": r["date"], "season": r["season"], "div": r["div"], "home": r["home"], "away": r["away"],
                "market": mk, "pick": spec["labels"][j], "p_model": pm[i, j], "price": price[i, j],
                "edge": edge[i, j], "p_close": pc[i, j],
                "clv": price[i, j] * pc[i, j] - 1 if np.isfinite(pc[i, j]) else np.nan,
                "won": bool(y[i, j]),
            })
    b = pd.DataFrame(bets)
    if b.empty:
        return b
    b = b.sort_values(["date", "home"]).reset_index(drop=True)
    b["profit_flat"] = np.where(b["won"], b["price"] - 1, -1.0)
    return b


def kelly_bankroll(b: pd.DataFrame, fraction: float, cap: float, start: float) -> pd.DataFrame:
    bank, stakes, banks = start, [], []
    for r in b.itertuples():
        f = min(cap, max(0.0, fraction * r.edge / (r.price - 1)))
        stake = bank * f
        bank += stake * (r.price - 1) if r.won else -stake
        stakes.append(stake)
        banks.append(bank)
    b = b.copy()
    b["stake_kelly"], b["bankroll"] = stakes, banks
    return b


def summarize(b: pd.DataFrame, start: float | None = None) -> dict:
    if b.empty:
        return {"bahis": 0}
    n = len(b)
    pf = b["profit_flat"]
    clv = b["clv"].dropna()
    s = {
        "bahis": n,
        "isabet_%": 100 * b["won"].mean(),
        "ort_oran": b["price"].mean(),
        "ROI_%": 100 * pf.mean(),
        "kâr_birim": pf.sum(),
        "t_istatistiği": pf.mean() / (pf.std(ddof=1) / np.sqrt(n)) if n > 1 and pf.std() > 0 else np.nan,
        "model_edge_ort_%": 100 * b["edge"].mean(),
        "CLV_ort_%": 100 * clv.mean(),
        "CLV_t": clv.mean() / (clv.std(ddof=1) / np.sqrt(len(clv))) if len(clv) > 1 and clv.std() > 0 else np.nan,
        "CLV>0_%": 100 * (clv > 0).mean(),
    }
    if start is not None and "bankroll" in b:
        peak = np.maximum.accumulate(np.r_[start, b["bankroll"].to_numpy()])
        s["kelly_kasa_son"] = b["bankroll"].iloc[-1]
        s["max_düşüş_%"] = 100 * np.max(1 - np.r_[start, b["bankroll"].to_numpy()] / peak)
    return s


# ---------------------------------------------------------------------------- rapor
def _md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def verdict(ev: dict, summ: dict) -> list[str]:
    lines = []
    e = ev.get("1X2")
    if e is not None:
        m, c = e["table"].iloc[0]["log_loss"], e["table"].iloc[2]["log_loss"]
        d, se = e["vs_close"]
        info = f"log loss {m:.4f} vs {c:.4f}, fark {d:+.4f} ± {1.96 * se:.4f}"
        if d - 1.96 * se > 0:
            lines.append(f"- 1X2'de model kapanış çizgisinden **anlamlı derecede kötü** ({info}). "
                         "Modelin bulduğu 'value'ların büyük kısmı muhtemelen model hatası; tek başına "
                         "bahis sinyali olarak kullanılmamalı.")
        elif d > 0:
            lines.append(f"- 1X2'de model kapanıştan biraz kötü ama fark anlamlı değil ({info}).")
        else:
            lines.append(f"- 1X2'de model kapanış çizgisi kadar ya da daha iyi ({info}). "
                         "Bu çok nadir görülür; önce sızıntı (leakage) olup olmadığını kontrol et.")
    if summ.get("bahis", 0) > 0:
        clv, clv_t, tstat = summ["CLV_ort_%"], summ["CLV_t"], summ["t_istatistiği"]
        sig = "anlamlı" if np.isfinite(clv_t) and abs(clv_t) >= 2 else "anlamlı değil"
        if clv > 0:
            lines.append(f"- Ortalama CLV pozitif (%{clv:.2f}, t = {clv_t:.2f}, {sig}): bahisler kapanış oranını "
                         "yenmiş. En umut verici sinyal bu; ama aşağıdaki strateji tablosunda keskin açılış "
                         "kuralıyla kıyasla.")
        else:
            lines.append(f"- Ortalama CLV negatif (%{clv:.2f}, t = {clv_t:.2f}, {sig}): bahisler kapanıştan kötü "
                         "oranlara yapılmış. Uzun vadede kârlılık beklenmez.")
        if np.isfinite(tstat) and abs(tstat) < 2:
            lines.append(f"- ROI istatistiksel olarak anlamlı değil (t = {tstat:.2f}); sonuç büyük ölçüde şans olabilir.")
    return lines


def run(div: str, df: pd.DataFrame, xi: float, ridge: float, edge_min: float, bet_book: str,
        closing_book: str = config.CLOSING_BOOK, method: str = config.DEVIG_METHOD,
        eval_from: int = config.EVAL_FROM_SEASON, ridge_total: float | None = config.RIDGE_TOTAL,
        preds: pd.DataFrame | None = None) -> str:
    """preds verilmezse df (bu ligin maçları) ile walk-forward yapılır. Birleşik modelde walk-forward tüm
    ligler için bir kez yapılır ve bu ligin satırları preds olarak verilir."""
    rt = "=ridge" if ridge_total is None else f"{ridge_total:g}"
    joint = preds is not None
    if preds is None:
        print(f"  [{div}] walk-forward başlıyor (xi={xi}, ridge={ridge}, ridge_total={rt}) ...")
        preds = walk_forward(df, eval_from, xi, ridge, ridge_total=ridge_total)
    if preds.empty:
        return f"## {config.LEAGUES.get(div, div)}\n\nDeğerlendirilecek maç yok.\n"
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(config.REPORTS_DIR / f"backtest_preds_{div}.csv", index=False)

    ev = evaluate(preds, bet_book, closing_book, method)
    cal = calibration(preds)
    bets = find_bets(preds, bet_book, closing_book, method, edge_min, config.MAX_ODDS, config.MIN_TEAM_MATCHES)
    if not bets.empty:
        bets = kelly_bankroll(bets, config.KELLY_FRACTION, config.KELLY_CAP, config.START_BANKROLL)
        bets.to_csv(config.REPORTS_DIR / f"backtest_bets_{div}.csv", index=False)
    summ = summarize(bets, config.START_BANKROLL)

    md = [f"## {config.LEAGUES.get(div, div)} ({div})",
          f"Değerlendirilen maç: {len(preds)} | sezonlar: {preds['season'].min()}–{preds['season'].max()} | "
          f"model={'birleşik (tüm ligler)' if joint else 'lige özel'}, "
          f"xi={xi}, ridge={ridge}, ridge_total={rt}, bahis oranı={config.book_label(bet_book)}, "
          f"kıyas={config.book_label(closing_book)}, edge eşiği=%{100 * edge_min:.0f}", ""]
    md += ["### Değerlendirme", *verdict(ev, summ), ""]
    for mk, e in ev.items():
        md += [f"### Tahmin kalitesi — {mk} (n={e['n']}, düşük = iyi)", _md_table(e["table"]), ""]
        for ref, key in [(config.book_label(bet_book), "vs_bet"), (config.book_label(closing_book), "vs_close")]:
            d, se = e[key]
            md.append(f"- Log loss farkı (model − {ref}): {d:+.4f} ± {1.96 * se:.4f} (%95 GA) → {_sig(d, se)}")
        md.append(f"- Model ile kapanış olasılıkları arasındaki ortalama mutlak fark: {e['mean_abs_diff_vs_close']:.3f}")
        if e["close_sources"]:
            md.append("- Kapanış kaynağı: " + ", ".join(f"{k.upper()} {v} maç" for k, v in e["close_sources"].items()))
        md.append("")
    if not cal.empty:
        md += ["### Kalibrasyon (1X2, tüm sonuçlar)", _md_table(cal, "{:.3f}"), ""]

    if bets.empty:
        md += ["### Value simülasyonu", "Eşiği geçen bahis yok.", ""]
    else:
        md += ["### Value simülasyonu (tüm marketler)", _md_table(pd.DataFrame([summ]), "{:.2f}"), ""]
        by_mk = pd.DataFrame([{"market": mk, **summarize(g)} for mk, g in bets.groupby("market")])
        md += ["### Markete göre", _md_table(by_mk, "{:.2f}"), ""]
        by_season = pd.DataFrame([{"sezon": s, **summarize(g)} for s, g in bets.groupby("season")])
        md += ["### Sezona göre", _md_table(by_season, "{:.2f}"), ""]

    def short(s: dict) -> dict:
        return {"bahis": s.get("bahis", 0), "ort_oran": s.get("ort_oran", np.nan), "ROI_%": s.get("ROI_%", np.nan),
                "t": s.get("t_istatistiği", np.nan), "CLV_ort_%": s.get("CLV_ort_%", np.nan),
                "CLV_t": s.get("CLV_t", np.nan)}

    strat = [{"strateji": label, **short(summarize(find_bets(preds, bet_book, closing_book, method, edge_min,
                                                             config.MAX_ODDS, config.MIN_TEAM_MATCHES, rule)))}
             for rule, label in RULES.items()]
    md += [f"### Strateji karşılaştırması (edge eşiği %{100 * edge_min:.0f}, oran {config.book_label(bet_book)})",
           _md_table(pd.DataFrame(strat), "{:.2f}"),
           "\nKeskin açılış = Pinnacle, yoksa Betfair borsası. Model, ancak CLV'si 'Keskin açılış' satırından "
           "belirgin şekilde yüksekse bahis anındaki piyasa bilgisine bir şey ekliyor demektir.", ""]
    # Farklı edge eşiklerinde ne olurdu?
    thr = []
    for e in [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
        s = short(summarize(find_bets(preds, bet_book, closing_book, method, e, config.MAX_ODDS,
                                      config.MIN_TEAM_MATCHES)))
        thr.append({"edge_eşiği_%": 100 * e, **s})
    md += ["### Edge eşiğine duyarlılık", _md_table(pd.DataFrame(thr), "{:.2f}"), ""]
    return "\n".join(md)
