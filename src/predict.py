"""Yaklaşan maçlar için olasılık, adil oran ve value tespiti.

İki kaynak:
  1) football-data.co.uk fixtures.csv (Bet365, Betfair borsası, piyasa ortalaması ve en iyi oranlarla)
  2) İsteğe bağlı: senin elle doldurduğun İddaa oranları (iddaa.csv)

Her olası bahis için iki ayrı edge hesaplanır:
  edge_model  = model_olasılığı * oran - 1       (modelin görüşü)
  edge_keskin = keskin_adil_olasılık * oran - 1  (oran, keskin piyasanın adil fiyatını geçiyor mu?
                                                  keskin = Pinnacle, yoksa Betfair borsası)
İkisinin birlikte pozitif olması en güvenilir sinyaldir.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import db
from src.dixon_coles import DixonColes
from src.download import download_fixtures
from src.features import add_motivation
from src.odds import MARKET_SUFFIXES, add_sharp, devig_frame
from src.teams import TeamResolver

MODEL_NAME = config.MODEL_VERSION

MARKET_COLS = [("1", "p_home", "h"), ("X", "p_draw", "d"), ("2", "p_away", "a"),
               ("Üst2.5", "p_over25", "o25"), ("Alt2.5", "p_under25", "u25")]


def _market_block(df: pd.DataFrame, prefix: str, book: str) -> pd.DataFrame:
    """Bir bürodan 1X2 ve Ü/A 2.5 oranlarını {prefix}_{etiket} sütunlarına kopyalar."""
    out = pd.DataFrame(index=df.index)
    for lab, _, suf in MARKET_COLS:
        c = f"{book}_{suf}"
        out[f"{prefix}_{lab}"] = df[c] if c in df else np.nan
    return out


def _add_edges(res: pd.DataFrame, price_prefix: str) -> pd.DataFrame:
    """Her satır için en iyi value adayını bulur (model ve keskin piyasa edge'leri ile)."""
    picks = []
    for _, r in res.iterrows():
        best = None
        for lab, pcol, _ in MARKET_COLS:
            price = r.get(f"{price_prefix}_{lab}", np.nan)
            if not np.isfinite(price) or price > config.MAX_ODDS:
                continue
            e_model = r[pcol] * price - 1
            p_sharp = r.get(f"sharp_fair_{lab}", np.nan)
            e_sharp = p_sharp * price - 1 if np.isfinite(p_sharp) else np.nan
            if best is None or e_model > best[2]:
                best = (lab, price, e_model, e_sharp)
        if best is None:
            picks.append((None, np.nan, np.nan, np.nan, ""))
            continue
        lab, price, e_model, e_sharp = best
        low_data = min(r["n_home"], r["n_away"]) < config.MIN_TEAM_MATCHES
        if e_model >= config.EDGE_MIN and not low_data:
            flag = "VALUE" if (np.isfinite(e_sharp) and e_sharp > 0) else "sadece model"
        else:
            flag = ""
        picks.append((lab, price, e_model, e_sharp, flag))
    res[[f"{price_prefix}_seçim", f"{price_prefix}_oran", f"{price_prefix}_edge_model",
         f"{price_prefix}_edge_keskin", f"{price_prefix}_sinyal"]] = pd.DataFrame(picks, index=res.index)
    return res


def _sharp_fair(df: pd.DataFrame) -> pd.DataFrame:
    """Keskin açılış oranlarının (sharp_*, bkz. odds.add_sharp) marjsız olasılıkları."""
    out = pd.DataFrame(index=df.index)
    for labs, sufs in [(["1", "X", "2"], MARKET_SUFFIXES["1x2"]), (["Üst2.5", "Alt2.5"], MARKET_SUFFIXES["ou"])]:
        cols = [f"sharp_{s}" for s in sufs]
        if all(c in df for c in cols):
            fair = devig_frame(df, cols, config.DEVIG_METHOD)
        else:
            fair = np.full((len(df), len(sufs)), np.nan)
        for k, lab in enumerate(labs):
            out[f"sharp_fair_{lab}"] = fair[:, k]
    return out


def _fit_models(matches: pd.DataFrame, ref_dates: dict, model_kw: dict, joint: bool) -> dict:
    """Lig -> model. Birleşik modelde her ülke grubu (config.LEAGUE_GROUPS) için tek model, grubun en erken
    maç tarihine göre eğitilir. Yeterli verisi olmayan lig / grup atlanır."""
    models = {}
    if joint:
        for gdivs in config.LEAGUE_GROUPS.values():
            refs = [ref_dates[d] for d in gdivs if d in ref_dates]
            if not refs:
                continue
            try:
                m = DixonColes(**model_kw).fit(matches[matches["div"].isin(gdivs)], ref_date=min(refs))
            except ValueError as e:
                print(f"  ! {', '.join(gdivs)}: {e}")
                continue
            models.update({d: m for d in gdivs if d in ref_dates})
        return models
    for div, ref in ref_dates.items():
        d = matches[matches["div"] == div]
        if d.empty:
            continue
        try:
            models[div] = DixonColes(**model_kw).fit(d, ref_date=ref)
        except ValueError as e:
            print(f"  ! {div}: {e}")
    return models


def _predict_rows(fx: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for rec in fx.to_dict("records"):
        m = models.get(rec["div"])
        if m is None:
            continue
        p = m.predict(rec["home"], rec["away"], rec["div"], rec.get("mot", 0))
        rows.append({**rec, **p})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    for lab, pcol, _ in MARKET_COLS:
        res[f"adil_oran_{lab}"] = 1 / res[pcol]
    return res


def run(fixtures_path: str | None, iddaa_path: str | None, xi: float, ridge: float,
        ridge_total: float | None = config.RIDGE_TOTAL, joint: bool = config.JOINT,
        motivation: bool = config.MOTIVATION, save: bool = True) -> pd.DataFrame:
    model_kw = {"xi": xi, "ridge": ridge, "ridge_total": ridge_total}
    matches = db.load_matches()
    if matches.empty:
        raise SystemExit("Veritabanı boş. Önce: python main.py download && python main.py build")
    if motivation:  # eğitim maçları ve tahmin edilecek maçlar için, maç öncesi puan durumundan
        matches = add_motivation(matches)
    today = pd.Timestamp(date.today())

    frames = []
    # ---- 1) football-data fixtures.csv
    path = Path(fixtures_path) if fixtures_path else download_fixtures()
    if path and Path(path).exists():
        fx = add_sharp(db.normalize(db.read_raw_csv(path), require_result=False))
        fx = fx[fx["div"].isin(config.LEAGUES) & (fx["date"] >= today - pd.Timedelta(days=1))]
        if not fx.empty:
            if motivation:
                fx = add_motivation(matches, fx)
            ref = fx.groupby("div")["date"].min().to_dict()
            models = _fit_models(matches, ref, model_kw, joint)
            res = _predict_rows(fx, models)
            if not res.empty:
                res = pd.concat([res, _sharp_fair(res), _market_block(res, "max", "max"),
                                 _market_block(res, "b365", "b365")], axis=1)
                res = _add_edges(res, "max")
                res["kaynak"] = "fixtures"
                frames.append(res)
    else:
        print("  ! fixtures.csv bulunamadı; sadece İddaa dosyası (varsa) kullanılacak.")

    # ---- 2) İddaa oranları
    if iddaa_path:
        frames.append(_run_iddaa(iddaa_path, matches, frames[0] if frames else None, model_kw, joint, motivation,
                                 today))

    if not frames:
        raise SystemExit("Tahmin edilecek maç bulunamadı.")
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("Tahmin edilecek maç bulunamadı.")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["kaynak", "div", "date", "home"]).reset_index(drop=True)
    if save:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        f = config.REPORTS_DIR / f"predictions_{today:%Y-%m-%d}.csv"
        out.to_csv(f, index=False)
        db.save_predictions(out.drop_duplicates(subset=["div", "date", "home", "away"]), MODEL_NAME)
        print(f"  kaydedildi: {f}")
    return out


def _latest_iddaa(today: pd.Timestamp) -> pd.DataFrame:
    """Otomatik toplanan İddaa oranlarından (src/iddaa.py) henüz başlamamış maçların son oranları,
    iddaa.csv biçiminde. Ü/A 2.5 sadece İddaa'nın ana barajı 2.5 ise."""
    from src import iddaa
    odds = iddaa.load_odds()
    if odds.empty:
        return odds
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    last = odds[odds["kickoff_utc"] > now].sort_values("snapshot_utc").groupby("event_id").tail(1)
    last = last.dropna(subset=["home", "away"])
    two_five = last["ou_line"] == 2.5
    print(f"  İddaa: otomatik toplanan oranlar, {len(last)} maç (son kayıt {odds['snapshot_utc'].max():%d.%m %H:%M} UTC)")
    return pd.DataFrame({"home": last["home"], "away": last["away"], "odd_1": last["odd_1"], "odd_x": last["odd_x"],
                         "odd_2": last["odd_2"], "odd_over25": last["odd_ust"].where(two_five),
                         "odd_under25": last["odd_alt"].where(two_five), "mbs": last["mbs"],
                         "date": last["kickoff_utc"].dt.strftime("%d.%m.%Y")})


def _run_iddaa(path: str, matches: pd.DataFrame, fixtures_res: pd.DataFrame | None,
               model_kw: dict, joint: bool, motivation: bool, today: pd.Timestamp) -> pd.DataFrame:
    """iddaa.csv sütunları: home,away,odd_1,odd_x,odd_2[,odd_over25,odd_under25][,date].
    path == "auto": dosya yerine otomatik toplanan son İddaa oranları kullanılır."""
    raw = _latest_iddaa(today) if path == "auto" else pd.read_csv(path)
    if raw.empty:
        return raw
    latest = matches.sort_values("date").groupby("home")["div"].last().to_dict()
    resolve = TeamResolver(list(matches["home"].unique()))
    recs = []
    for r in raw.to_dict("records"):
        h, a = resolve(r["home"]), resolve(r["away"])
        if not h or not a:
            print(f"  ! takım eşleşmedi: '{r['home']}' / '{r['away']}' -> team_aliases.csv'ye ekle "
                  f"(python main.py teams ile doğru isimleri gör)")
            continue
        d = pd.to_datetime(r.get("date"), dayfirst=True, errors="coerce") if r.get("date") else pd.NaT
        d = d if pd.notna(d) else today
        recs.append({"div": latest.get(h), "date": d, "season": d.year if d.month >= 7 else d.year - 1,
                     "home": h, "away": a,
                     "iddaa_1": r.get("odd_1"), "iddaa_X": r.get("odd_x"), "iddaa_2": r.get("odd_2"),
                     "iddaa_Üst2.5": r.get("odd_over25", np.nan), "iddaa_Alt2.5": r.get("odd_under25", np.nan),
                     "mbs": r.get("mbs", np.nan)})
    fx = pd.DataFrame(recs)
    if fx.empty:
        return fx
    for c in [c for c in fx if c.startswith("iddaa_")]:
        fx[c] = pd.to_numeric(fx[c], errors="coerce")
    if motivation:
        fx = add_motivation(matches, fx)
    ref = {div: today for div in fx["div"].dropna().unique()}
    models = _fit_models(matches, ref, model_kw, joint)
    res = _predict_rows(fx, models)
    if res.empty:
        return res
    # Aynı maç fixtures.csv'de varsa keskin piyasa oranlarını oradan al
    if fixtures_res is not None and not fixtures_res.empty:
        sharp_cols = [f"sharp_{s}" for sufs in MARKET_SUFFIXES.values() for s in sufs]
        res = res.merge(fixtures_res[["home", "away", *sharp_cols]].drop_duplicates(["home", "away"]),
                        on=["home", "away"], how="left")
    res = pd.concat([res, _sharp_fair(res)], axis=1)
    res = _add_edges(res, "iddaa")
    res["kaynak"] = "iddaa"
    return res


LABEL_TR = {"1": "1 (ev)", "X": "X (beraberlik)", "2": "2 (deplasman)", "Üst2.5": "Üst 2.5", "Alt2.5": "Alt 2.5"}
PCOL = {lab: pcol for lab, pcol, _ in MARKET_COLS}


def _reliability() -> dict:
    """(lig, kural) -> backtest özeti metni (reports/strateji_ozet.csv; python main.py backtest yazar)."""
    f = config.REPORTS_DIR / "strateji_ozet.csv"
    if not f.exists():
        return {}
    return {(r["lig"], r["kural"]): (f"{r['CLV_ort_%']:+.1f}% (t {r['CLV_t']:+.1f}, n {r['bahis']})"
                                     if r["bahis"] > 1 else "—")
            for r in pd.read_csv(f).to_dict("records")}


def recommendations(res: pd.DataFrame, min_prob: float = 0.60) -> str:
    """Öneri raporu (markdown). Maç hem İddaa'da hem fikstürde varsa İddaa oranı esas alınır.
      1) Keskin fiyat sinyali: oran keskin piyasanın (Pinnacle / Betfair) adil fiyatını >= EDGE_MIN geçiyor
         (model kullanılmaz; backtest'te en tutarlı pozitif CLV bu kuralda).
      2) VALUE: model edge >= EDGE_MIN ve oran keskin adil fiyatın üstünde (bkz. _add_edges).
      3) En olası sonuçlar: modelin >= min_prob verdiği 1X2 sonuçları (value DEĞİL).
    Her sinyalin yanında o ligde o kuralın backtest CLV'si (B365 oranıyla) gösterilir."""
    today = pd.Timestamp(date.today())
    rel = _reliability()
    md = [f"# Öneriler — {today:%d.%m.%Y}", "",
          f"Model: {config.MODEL_VERSION}. Oran: İddaa (varsa), yoksa piyasadaki en iyi oran. 'Backtest' sütunu: "
          "bu kuralın o ligdeki geçmiş CLV'si (B365 ile; t ≥ 2 anlamlı). İddaa'da canlı ölçüm yeni başladı; "
          "gerçek parayla oynamadan önce tracking/summary.md'deki İddaa CLV'sine bak.", ""]
    seen, sharp, value, likely = set(), [], [], []
    for src, pre in (("iddaa", "iddaa"), ("fixtures", "max")):
        for r in res[res["kaynak"] == src].to_dict("records"):
            key = (r["div"], r["home"], r["away"])
            if key in seen:
                continue
            seen.add(key)
            match = f"{r['home']} - {r['away']}"
            when = pd.to_datetime(r["date"]).strftime("%d.%m")
            where = "İddaa" if src == "iddaa" else "piyasa (en iyi)"
            enough = min(r["n_home"], r["n_away"]) >= config.MIN_TEAM_MATCHES
            for lab, pcol, _ in MARKET_COLS:  # keskin fiyat sinyali: her sonuç ayrı
                price, ps = r.get(f"{pre}_{lab}", np.nan), r.get(f"sharp_fair_{lab}", np.nan)
                if not (np.isfinite(price) and np.isfinite(ps)) or price > config.MAX_ODDS or not enough:
                    continue
                if ps * price - 1 >= config.EDGE_MIN:
                    sharp.append({"lig": r["div"], "tarih": when, "maç": match, "seçim": LABEL_TR[lab], "oran": price,
                                  "keskin adil oran": 1 / ps, "edge keskin_%": 100 * (ps * price - 1),
                                  "model olasılığı_%": 100 * r[pcol], "MBS": r.get("mbs", np.nan),
                                  "backtest": rel.get((r["div"], "sharp"), "—"), "oran kaynağı": where})
            if r.get(f"{pre}_sinyal") == "VALUE":
                lab = r[f"{pre}_seçim"]
                p = r[PCOL[lab]]
                value.append({"lig": r["div"], "tarih": when, "maç": match, "seçim": LABEL_TR[lab],
                              "oran": r[f"{pre}_oran"], "adil oran (model)": 1 / p, "model olasılığı_%": 100 * p,
                              "keskin piyasa_%": 100 * r.get(f"sharp_fair_{lab}", np.nan),
                              "edge model_%": 100 * r[f"{pre}_edge_model"], "edge keskin_%": 100 * r[f"{pre}_edge_keskin"],
                              "MBS": r.get("mbs", np.nan), "backtest": rel.get((r["div"], "both"), "—"),
                              "oran kaynağı": where})
            probs = {"1": r["p_home"], "X": r["p_draw"], "2": r["p_away"]}
            lab = max(probs, key=probs.get)
            if probs[lab] >= min_prob:
                price = r.get(f"{pre}_{lab}", np.nan)
                likely.append({"lig": r["div"], "tarih": when, "maç": match, "en olası": LABEL_TR[lab],
                               "olasılık_%": 100 * probs[lab], "adil oran": 1 / probs[lab], "oran": price,
                               "oran − adil_%": 100 * (price * probs[lab] - 1) if np.isfinite(price) else np.nan,
                               "oran kaynağı": where})
    from src.backtest import _md_table
    md += ["## 1) Keskin fiyat sinyali (model yok)", "",
           f"Oran, keskin piyasanın (Pinnacle / Betfair) adil fiyatını ≥ %{100 * config.EDGE_MIN:.0f} geçiyor. "
           "Backtest'te 22 ligin 20'sinde pozitif CLV veren kural bu (yumuşak büro keskin piyasanın gerisinde "
           "kalınca). Keskin fiyat fikstürdeki son kayıttan; İddaa oranı daha yeni olabilir. MBS > 1 ise tek oynanamaz.", ""]
    md += [_md_table(pd.DataFrame(sharp).sort_values("edge keskin_%", ascending=False), "{:.2f}") if sharp
           else "Bu çalıştırmada keskin fiyat sinyali yok.", ""]
    md += ["## 2) VALUE (model + keskin onay)", "",
           f"Model edge ≥ %{100 * config.EDGE_MIN:.0f} **ve** oran keskin adil fiyatın üstünde. Backtest'te bazı "
           "liglerde pozitif CLV, ama keskin fiyat sinyalinden genelde daha iyi değil.", ""]
    md += [_md_table(pd.DataFrame(value).sort_values("edge model_%", ascending=False), "{:.2f}") if value
           else "Bu çalıştırmada VALUE yok.", ""]
    md += [f"## 3) En olası sonuçlar (model ≥ %{100 * min_prob:.0f}) — value DEĞİL", "",
           "İsabet oranı yüksek olur ama kâr göstergesi değildir: 'oran − adil' negatifse bu oranla uzun vadede "
           "kaybettirir.", ""]
    md += [_md_table(pd.DataFrame(likely).sort_values("olasılık_%", ascending=False), "{:.2f}") if likely
           else "Bu eşiği geçen maç yok.", ""]
    return "\n".join(md)


def print_table(res: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for src, g in res.groupby("kaynak"):
        pre = "max" if src == "fixtures" else "iddaa"
        cols = ["div", "date", "home", "away", "exp_home", "exp_away", "p_home", "p_draw", "p_away",
                "p_over25", "top_scores", f"{pre}_seçim", f"{pre}_oran", f"{pre}_edge_model",
                f"{pre}_edge_keskin", f"{pre}_sinyal"]
        show = g[[c for c in cols if c in g]].copy()
        show["date"] = pd.to_datetime(show["date"]).dt.strftime("%d.%m")
        for c in ["exp_home", "exp_away"]:
            show[c] = show[c].round(2)
        for c in ["p_home", "p_draw", "p_away", "p_over25"]:
            show[c] = (100 * show[c]).round(0).astype(int).astype(str) + "%"
        for c in [f"{pre}_edge_model", f"{pre}_edge_keskin"]:
            if c in show:
                show[c] = (100 * show[c]).round(1)
        title = "football-data fixtures (en iyi piyasa oranına göre)" if src == "fixtures" else "İddaa oranlarına göre"
        print(f"\n=== {title} ===")
        print(show.to_string(index=False))
