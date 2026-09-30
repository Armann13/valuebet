"""Risk seviyesine göre tavsiye raporu (yaklaşan maçlar).

Her maç için modelin skor olasılık tablosu (Dixon-Coles) kurulur. Keskin piyasanın (Pinnacle / Betfair) 1X2
olasılıkları varsa tablo onlara göre ayarlanır: ev galibiyeti / beraberlik / deplasman bölgeleri ayrı ayrı
ölçeklenir (toplamlar piyasayla aynı olur, skor dağılımı modelden gelir). Piyasa modelden isabetli olduğu için
olasılıklar böylece daha doğru, ve tüm marketler (MS, çifte şans, alt/üst, KG, kombinasyon, kesin skor) aynı
tablodan geldiği için birbiriyle tutarlıdır.

Risk seviyesi = tahminin TUTMAMA olasılığı:
  Tier 1 düşük (olasılık >= %70), Tier 2 orta (%50-70), Tier 3 yüksek (%25-50: beraberlik, sürpriz, kombinasyon),
  Tier 4 çok yüksek (en olası kesin skor, kısa liste), Tier 5 aşırı (her maçın en olası kesin skoru).
Değer: İddaa oranı x olasılık - 1. Pozitifse fiyat adil fiyatın üstünde (uzun vadede kârlı olması beklenir);
negatifse tahmin tutsa bile bu oranla uzun vadede kaybettirir. Risk seviyesi ile değer farklı şeylerdir.
"""
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

import config
from src import db
from src.features import add_motivation
from src.odds import add_sharp, devig_frame

TIERS = [("🟢 Tier 1 — Düşük risk", 0.70, 1.01), ("🟡 Tier 2 — Orta risk", 0.50, 0.70),
         ("🔴 Tier 3 — Yüksek risk", 0.25, 0.50)]
MAX_PER_TIER = {0: 8, 1: 10, 2: 10}
MAX_SHORT_SCORES = 6          # Tier 4: en olası kesin skoru en yüksek olan maçlar
IDDAA_COL = {"MS 1": "odd_1", "MS X": "odd_x", "MS 2": "odd_2", "ÇŞ 1-X": "odd_1x", "ÇŞ 1-2": "odd_12",
             "ÇŞ X-2": "odd_x2", "2.5 Alt": "odd_alt", "2.5 Üst": "odd_ust", "KG Var": "odd_kg_var",
             "KG Yok": "odd_kg_yok"}


def market_probs(m: np.ndarray) -> dict:
    """Skor tablosundan (m[ev golü, deplasman golü]) market olasılıkları."""
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    home, draw, away = h > a, h == a, h < a
    over, btts = (h + a) >= 3, (h >= 1) & (a >= 1)
    p = lambda mask: float(m[mask].sum())  # noqa: E731
    return {"MS 1": p(home), "MS X": p(draw), "MS 2": p(away),
            "ÇŞ 1-X": p(home | draw), "ÇŞ X-2": p(draw | away), "ÇŞ 1-2": p(home | away),
            "2.5 Üst": p(over), "2.5 Alt": p(~over), "KG Var": p(btts), "KG Yok": p(~btts),
            "MS 1 & KG Var": p(home & btts), "MS 2 & KG Var": p(away & btts), "MS X & KG Var": p(draw & btts),
            "MS 1 & 2.5 Üst": p(home & over), "MS 2 & 2.5 Üst": p(away & over)}


def calibrate(m: np.ndarray, target: tuple) -> np.ndarray:
    """1X2 bölgelerini piyasa olasılıklarına (target = (p1, px, p2)) ölçekler."""
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    out = m.copy()
    for mask, t in ((h > a, target[0]), (h == a, target[1]), (h < a, target[2])):
        s = m[mask].sum()
        if s > 0 and np.isfinite(t):
            out[mask] *= t / s
    return out / out.sum()


def _upcoming(fixtures_path) -> pd.DataFrame:
    """Henüz başlamamış maçlar: İddaa'da listelenenler (son oranlarıyla) + fikstürdekiler (keskin oranlarla)."""
    from src import iddaa
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows = []
    odds = iddaa.load_odds()
    if not odds.empty:
        last = odds[odds["kickoff_utc"] > now].sort_values("snapshot_utc").groupby("event_id").tail(1)
        last = last.dropna(subset=["home", "away"])
        for c in ["odd_1x", "odd_12", "odd_x2", "odd_kg_var", "odd_kg_yok", "mbs"]:
            last[c] = last[c] if c in last else np.nan
        last = last.assign(date=last["kickoff_utc"].dt.normalize(), kaynak="İddaa")
        rows.append(last)
    if fixtures_path and fixtures_path.exists():
        fx = add_sharp(db.normalize(db.read_raw_csv(fixtures_path), require_result=False))
        fx = fx[fx["div"].isin(config.LEAGUES) & (fx["date"] >= now.normalize())]
        if not fx.empty:
            for i, (lab, s) in enumerate(zip(("1", "x", "2"), ("h", "d", "a"))):
                fx[f"keskin_{lab}"] = devig_frame(fx, ["sharp_h", "sharp_d", "sharp_a"], config.DEVIG_METHOD)[:, i]
            rows.append(fx.assign(kaynak="fikstür"))
    if not rows:
        return pd.DataFrame()
    up = pd.concat(rows, ignore_index=True, sort=False)
    # aynı maç iki kaynakta: keskin oranları fikstürden, İddaa oranlarını İddaa'dan al
    keep = [c for c in up if c.startswith(("odd_", "keskin_")) or c in ("mbs", "ou_line", "kickoff_utc")]
    merged = up.groupby(["div", "home", "away"], as_index=False).agg(
        {**{c: "first" for c in keep}, "date": "min", "kaynak": lambda s: " + ".join(sorted(set(s)))})
    merged["season"] = np.where(merged["date"].dt.month >= 7, merged["date"].dt.year, merged["date"].dt.year - 1)
    return merged


def run(fixtures_path=None, xi=config.XI, ridge=config.RIDGE, ridge_total=config.RIDGE_TOTAL,
        joint=config.JOINT, motivation=config.MOTIVATION) -> str:
    from src.predict import _fit_models
    up = _upcoming(fixtures_path)
    today = date.today()
    md = [f"# Tavsiyeler — {today:%d.%m.%Y}", "",
          "Her satır: tahmin — kazanma olasılığı, adil oran, İddaa oranı ve **değer** (İddaa oranı adil oranın "
          "üstündeyse +). Risk seviyesi tahminin tutma olasılığına göredir; **değer ayrı bir şeydir**: düşük "
          "riskli bir tahmin değeri negatifse uzun vadede kaybettirir. Olasılıklar: keskin piyasa (Pinnacle / "
          "Betfair) varsa ona ayarlanmış model, yoksa sadece model (daha az isabetli). Kombinasyon ve kesin skor "
          "olasılıkları modelin skor tablosundan gelir.", ""]
    if up.empty:
        return "\n".join(md + ["Yaklaşan maç bulunamadı (İddaa kaydı ya da fikstür yok)."])
    matches = db.load_matches()
    if motivation:
        matches = add_motivation(matches)
        up = add_motivation(matches, up)
    models = _fit_models(matches, up.groupby("div")["date"].min().to_dict(),
                         {"xi": xi, "ridge": ridge, "ridge_total": ridge_total}, joint)
    cands, scores = [], []
    for r in up.to_dict("records"):
        mdl = models.get(r["div"])
        if mdl is None:
            continue
        m = mdl.score_matrix(r["home"], r["away"], r["div"], r.get("mot", 0))
        target = tuple(r.get(f"keskin_{k}", np.nan) for k in ("1", "x", "2"))
        src = "piyasa" if all(np.isfinite(target)) else "model"
        if src == "piyasa":
            m = calibrate(m, target)
        info = {"maç": f"{r['home']} – {r['away']}", "lig": r["div"], "tarih": pd.Timestamp(r["date"]).strftime("%d.%m"),
                "mbs": r.get("mbs"), "kaynak": src}
        for bet, p in market_probs(m).items():
            col = IDDAA_COL.get(bet)
            odd = r.get(col, np.nan) if col else np.nan
            if bet.startswith("2.5") and r.get("ou_line") != 2.5:
                odd = np.nan  # İddaa'nın ana barajı 2.5 değilse bu oran başka baraj içindir
            cands.append({**info, "tahmin": bet, "p": p, "odd": odd})
        i, j = np.unravel_index(np.argmax(m), m.shape)
        scores.append({**info, "tahmin": f"{i}-{j}", "p": float(m[i, j]), "odd": np.nan})
    c = pd.DataFrame(cands)
    c["değer"] = c["odd"] * c["p"] - 1

    def line(x) -> str:
        s = f"• {x['maç']} ({x['lig']}, {x['tarih']}): **{x['tahmin']}** — olasılık %{100 * x['p']:.0f}, adil oran {1 / x['p']:.2f}"
        if np.isfinite(x["odd"]):
            s += f", İddaa {x['odd']:.2f} (değer {100 * x['değer']:+.0f}%{' ✅' if x['değer'] > 0 else ''})"
        else:
            s += ", İddaa oranı yok"
        if pd.notna(x.get("mbs")) and x.get("mbs") > 1:
            s += f", MBS {int(x['mbs'])}"
        return s + ("" if x["kaynak"] == "piyasa" else " _(sadece model)_")

    for k, (title, lo, hi) in enumerate(TIERS):
        t = c[(c["p"] >= lo) & (c["p"] < hi)].copy()
        if k == 0:  # düşük riskte çifte şans ancak MS zaten >= %70 değilse
            t = t[~t["tahmin"].str.startswith("ÇŞ") | ~t["maç"].isin(t.loc[t["tahmin"].str.startswith("MS "), "maç"])]
        # maç başına en iyi tek tahmin: önce değeri pozitif olanlar, sonra olasılık
        t["sıra"] = np.where(t["değer"] > 0, 1, 0)
        t = t.sort_values(["sıra", "değer", "p"], ascending=False).groupby("maç").head(1)
        t = t.sort_values(["sıra", "p"], ascending=False).head(MAX_PER_TIER[k])
        md += [f"## {title}", ""] + ([line(x) for x in t.to_dict("records")] or ["Bu seviyede tahmin yok."]) + [""]
    s = pd.DataFrame(scores).sort_values("p", ascending=False)
    md += ["## ⚡ Tier 4 — Çok yüksek risk (kısa kesin skor)", ""]
    md += [line(x) for x in s.head(MAX_SHORT_SCORES).to_dict("records")] + [""]
    md += ["## 🔥 Tier 5 — Aşırı risk (tüm maçlarda kesin skor)", ""]
    md += [line(x) for x in s.to_dict("records")] + [""]
    pos = c[(c["değer"] > 0) & (c["p"] >= 0.25)].sort_values("değer", ascending=False)
    md += ["## ✅ Değeri pozitif olanlar (risk seviyesinden bağımsız)", "",
           "Uzun vadede kâr beklentisi olan tek grup bu; olasılık keskin piyasaya ayarlıysa daha güvenilir.", ""]
    md += [line(x) for x in pos.head(15).to_dict("records")] or ["İddaa oranı adil fiyatın üstünde olan tahmin yok."]
    return "\n".join(md)
