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
import difflib
import unicodedata
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src import db
from src.dixon_coles import DixonColes
from src.download import download_fixtures
from src.odds import MARKET_SUFFIXES, add_sharp, devig_frame

MODEL_NAME = "dixon_coles_v1"

# İddaa / yaygın isim -> football-data.co.uk ismi. Eksik olanları team_aliases.csv'ye ekleyebilirsin.
BUILTIN_ALIASES = {
    "atletico madrid": "Ath Madrid", "atl. madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao", "bilbao": "Ath Bilbao",
    "real betis": "Betis", "celta vigo": "Celta", "real sociedad": "Sociedad",
    "rayo vallecano": "Vallecano", "rayo": "Vallecano", "espanyol": "Espanol",
    "deportivo la coruna": "La Coruna", "deportivo": "La Coruna", "sporting gijon": "Sp Gijon",
    "real valladolid": "Valladolid", "deportivo alaves": "Alaves", "real oviedo": "Oviedo",
    "racing santander": "Santander", "real zaragoza": "Zaragoza", "rcd mallorca": "Mallorca",
    "real mallorca": "Mallorca", "ud las palmas": "Las Palmas", "granada cf": "Granada",
    "sd huesca": "Huesca", "cd tenerife": "Tenerife", "sd eibar": "Eibar", "malaga cf": "Malaga",
    "cd leganes": "Leganes", "girona fc": "Girona", "getafe cf": "Getafe", "sevilla fc": "Sevilla",
    "valencia cf": "Valencia", "villarreal cf": "Villarreal", "fc barcelona": "Barcelona",
}

MARKET_COLS = [("1", "p_home", "h"), ("X", "p_draw", "d"), ("2", "p_away", "a"),
               ("Üst2.5", "p_over25", "o25"), ("Alt2.5", "p_under25", "u25")]


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


class TeamResolver:
    def __init__(self, known: list[str]):
        self.known = sorted(set(known))
        self.by_norm = {_norm(t): t for t in self.known}
        self.aliases = {k: v for k, v in BUILTIN_ALIASES.items() if v in self.known}
        f = config.ROOT / "team_aliases.csv"
        if f.exists():
            for r in pd.read_csv(f).itertuples(index=False):
                self.aliases[_norm(r.alias)] = str(r.team).strip()

    def __call__(self, name: str) -> str | None:
        if name in self.known:
            return name
        n = _norm(name)
        if n in self.aliases:
            return self.aliases[n]
        if n in self.by_norm:
            return self.by_norm[n]
        hit = difflib.get_close_matches(n, list(self.by_norm), n=1, cutoff=0.75)
        return self.by_norm[hit[0]] if hit else None


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


def _fit_models(matches: pd.DataFrame, ref_dates: dict, xi: float, ridge: float) -> dict:
    models = {}
    for div, ref in ref_dates.items():
        d = matches[matches["div"] == div]
        if d.empty:
            continue
        models[div] = DixonColes(xi=xi, ridge=ridge).fit(d, ref_date=ref)
    return models


def _predict_rows(fx: pd.DataFrame, models: dict) -> pd.DataFrame:
    rows = []
    for rec in fx.to_dict("records"):
        m = models.get(rec["div"])
        if m is None:
            continue
        p = m.predict(rec["home"], rec["away"])
        rows.append({**rec, **p})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    for lab, pcol, _ in MARKET_COLS:
        res[f"adil_oran_{lab}"] = 1 / res[pcol]
    return res


def run(fixtures_path: str | None, iddaa_path: str | None, xi: float, ridge: float,
        save: bool = True) -> pd.DataFrame:
    matches = db.load_matches()
    if matches.empty:
        raise SystemExit("Veritabanı boş. Önce: python main.py download && python main.py build")
    today = pd.Timestamp(date.today())

    frames = []
    # ---- 1) football-data fixtures.csv
    path = Path(fixtures_path) if fixtures_path else download_fixtures()
    if path and Path(path).exists():
        fx = add_sharp(db.normalize(db.read_raw_csv(path), require_result=False))
        fx = fx[fx["div"].isin(config.LEAGUES) & (fx["date"] >= today - pd.Timedelta(days=1))]
        if not fx.empty:
            ref = fx.groupby("div")["date"].min().to_dict()
            models = _fit_models(matches, ref, xi, ridge)
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
        frames.append(_run_iddaa(iddaa_path, matches, frames[0] if frames else None, xi, ridge, today))

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


def _run_iddaa(path: str, matches: pd.DataFrame, fixtures_res: pd.DataFrame | None,
               xi: float, ridge: float, today: pd.Timestamp) -> pd.DataFrame:
    """iddaa.csv sütunları: home,away,odd_1,odd_x,odd_2[,odd_over25,odd_under25][,date]"""
    raw = pd.read_csv(path)
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
        recs.append({"div": latest.get(h), "date": d if pd.notna(d) else today, "home": h, "away": a,
                     "iddaa_1": r.get("odd_1"), "iddaa_X": r.get("odd_x"), "iddaa_2": r.get("odd_2"),
                     "iddaa_Üst2.5": r.get("odd_over25", np.nan), "iddaa_Alt2.5": r.get("odd_under25", np.nan)})
    fx = pd.DataFrame(recs)
    if fx.empty:
        return fx
    for c in [c for c in fx if c.startswith("iddaa_")]:
        fx[c] = pd.to_numeric(fx[c], errors="coerce")
    ref = {div: today for div in fx["div"].dropna().unique()}
    models = _fit_models(matches, ref, xi, ridge)
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
