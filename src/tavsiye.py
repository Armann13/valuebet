"""Risk seviyesine göre tavsiye raporu (yaklaşan maçlar ya da geçmiş bir hafta sonunun provası).

Her maç için modelin skor olasılık tablosu (Dixon-Coles) kurulur. Keskin piyasanın (Pinnacle / Betfair) 1X2
ve 2.5 alt/üst olasılıkları varsa tablo onlara göre ayarlanır (bölgeler sırayla ölçeklenir; toplamlar piyasayla
aynı olur, skor dağılımının geri kalanı modelden gelir). Piyasa modelden isabetli olduğu için
olasılıklar böylece daha doğru, ve tüm marketler (MS, çifte şans, alt/üst, KG, kombinasyon, kesin skor) aynı
tablodan geldiği için birbiriyle tutarlıdır.

Risk seviyesi = tahminin TUTMAMA olasılığı:
  Tier 1 düşük (olasılık >= %70), Tier 2 orta (%50-70), Tier 3 yüksek (%25-50: beraberlik, sürpriz, kombinasyon),
  Tier 4 çok yüksek (en olası kesin skor, kısa liste), Tier 5 aşırı (her maçın en olası kesin skoru).
Değer: oran x olasılık - 1. Pozitifse fiyat adil fiyatın üstünde (uzun vadede kârlı olması beklenir); negatifse
tahmin tutsa bile bu oranla uzun vadede kaybettirir. Risk seviyesi ile değer farklı şeylerdir.

Prova (as_of): model sadece o tarihten önceki maçlarla eğitilir; o tarihten itibaren `days` gündeki maçlar
için tavsiye üretilir ve gerçekleşen sonuçlarla karşılaştırılır. Geçmiş İddaa oranı olmadığı için fiyat olarak
Bet365 kullanılır; keskin olasılık açılış oranından (bahis anında bilinen), kapanıştan değil.
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
ODD_COL = {"MS 1": "odd_1", "MS X": "odd_x", "MS 2": "odd_2", "ÇŞ 1-X": "odd_1x", "ÇŞ 1-2": "odd_12",
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


def bet_won(bet: str, hg: int, ag: int) -> bool:
    """Tahminin tutup tutmadığı (kesin skor "2-0" biçiminde)."""
    if "-" in bet and bet[0].isdigit():
        return bet == f"{hg}-{ag}"
    res = {"1": hg > ag, "X": hg == ag, "2": hg < ag}
    checks = {"ÇŞ 1-X": hg >= ag, "ÇŞ X-2": hg <= ag, "ÇŞ 1-2": hg != ag, "2.5 Üst": hg + ag >= 3,
              "2.5 Alt": hg + ag <= 2, "KG Var": hg >= 1 and ag >= 1, "KG Yok": hg == 0 or ag == 0}
    ok = True
    for part in bet.split(" & "):
        ok &= res[part.split()[1]] if part.startswith("MS ") else checks[part]
    return bool(ok)


def calibrate(m: np.ndarray, target: tuple, target_ou: tuple | None = None, iters: int = 50) -> np.ndarray:
    """Skor tablosunu piyasa olasılıklarına ayarlar: 1X2 bölgeleri target = (p1, px, p2)'ye, verilirse 2.5 alt/üst
    bölgeleri target_ou = (p_üst, p_alt)'a. İki kısıt varsa sırayla, yakınsayana kadar uygulanır (IPF)."""
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    groups = [[(h > a, target[0]), (h == a, target[1]), (h < a, target[2])]]
    if target_ou is not None and all(np.isfinite(target_ou)):
        groups.append([((h + a) >= 3, target_ou[0]), ((h + a) <= 2, target_ou[1])])
    out = m / m.sum()
    for _ in range(iters if len(groups) > 1 else 1):
        for grp in groups:
            for mask, t in grp:
                s = out[mask].sum()
                if s > 0 and np.isfinite(t):
                    out[mask] *= t / s
            out /= out.sum()
    return out


def _sharp_1x2(df: pd.DataFrame) -> pd.DataFrame:
    """Keskin açılış (PS, yoksa Betfair) adil olasılıkları: 1X2 ve 2.5 alt/üst."""
    fair = devig_frame(df, ["sharp_h", "sharp_d", "sharp_a"], config.DEVIG_METHOD)
    ou = devig_frame(df, ["sharp_o25", "sharp_u25"], config.DEVIG_METHOD)
    return df.assign(keskin_1=fair[:, 0], keskin_x=fair[:, 1], keskin_2=fair[:, 2], keskin_ust=ou[:, 0],
                     keskin_alt=ou[:, 1])


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
        rows.append(last.assign(date=last["kickoff_utc"].dt.normalize(), kaynak="İddaa"))
    if fixtures_path and fixtures_path.exists():
        fx = add_sharp(db.normalize(db.read_raw_csv(fixtures_path), require_result=False))
        fx = fx[fx["div"].isin(config.LEAGUES) & (fx["date"] >= now.normalize())]
        if not fx.empty:
            rows.append(_sharp_1x2(fx).assign(kaynak="fikstür"))
    if not rows:
        return pd.DataFrame()
    up = pd.concat(rows, ignore_index=True, sort=False)
    # aynı maç iki kaynakta: keskin oranları fikstürden, İddaa oranlarını İddaa'dan al ("first" boşları atlar)
    keep = [c for c in up if c.startswith(("odd_", "keskin_")) or c in ("mbs", "ou_line", "kickoff_utc")]
    merged = up.groupby(["div", "home", "away"], as_index=False).agg(
        {**{c: "first" for c in keep}, "date": "min", "kaynak": lambda s: " + ".join(sorted(set(s)))})
    merged["season"] = np.where(merged["date"].dt.month >= 7, merged["date"].dt.year, merged["date"].dt.year - 1)
    return merged


def _history(matches: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Prova: [start, start+days) arasındaki oynanmış maçlar; fiyat Bet365, keskin olasılık açılış oranından."""
    up = matches[(matches["date"] >= start) & (matches["date"] < start + pd.Timedelta(days=days))].copy()
    up = _sharp_1x2(up)
    up["odd_1"], up["odd_x"], up["odd_2"] = up["b365_h"], up["b365_d"], up["b365_a"]
    up["odd_ust"], up["odd_alt"] = up["b365_o25"], up["b365_u25"]
    up["ou_line"] = np.where(up["b365_o25"].notna(), 2.5, np.nan)
    for c in ["odd_1x", "odd_12", "odd_x2", "odd_kg_var", "odd_kg_yok", "mbs"]:
        up[c] = np.nan
    return up.assign(kaynak="prova")


def run(fixtures_path=None, xi=config.XI, ridge=config.RIDGE, ridge_total=config.RIDGE_TOTAL,
        joint=config.JOINT, motivation=config.MOTIVATION, as_of: str | None = None, days: int = 4) -> str:
    from src.predict import _fit_models
    matches = db.load_matches()
    if as_of:
        start = pd.Timestamp(as_of)
        up, train, price = _history(matches, start, days), matches[matches["date"] < start], "B365"
        title = f"# Tavsiye provası — {start:%d.%m.%Y} ve sonraki {days} gün (sonuçlarla)"
    else:
        up, train, price = _upcoming(fixtures_path), matches, "İddaa"
        title = f"# Tavsiyeler — {date.today():%d.%m.%Y}"
    md = [title, "",
          f"Her satır: tahmin — kazanma olasılığı, adil oran, {price} oranı ve **değer** ({price} oranı adil oranın "
          "üstündeyse +). Risk seviyesi tahminin tutma olasılığına göredir; **değer ayrı bir şeydir**: düşük "
          "riskli bir tahmin değeri negatifse uzun vadede kaybettirir. Olasılıklar: keskin piyasa (Pinnacle / "
          "Betfair) varsa ona ayarlanmış model, yoksa sadece model (daha az isabetli). Kombinasyon ve kesin skor "
          "olasılıkları modelin skor tablosundan gelir.", ""]
    if as_of:
        md += ["**Prova:** model sadece bu tarihten önceki maçları gördü; keskin olasılık o anki açılış oranından. "
               "Geçmiş İddaa oranı olmadığı için fiyat Bet365. Tek hafta sonu çok küçük bir örnek — sonuçlar "
               "şanstan çok etkilenir; amaç sistemin nasıl çalıştığını göstermek.", ""]
    if up.empty:
        return "\n".join(md + ["Maç bulunamadı."])
    if motivation:
        train = add_motivation(train)
        up = add_motivation(train, up)
    refs = {d: (start if as_of else t) for d, t in up.groupby("div")["date"].min().items()}
    models = _fit_models(train, refs, {"xi": xi, "ridge": ridge, "ridge_total": ridge_total}, joint)
    cands, scores = [], []
    for r in up.to_dict("records"):
        mdl = models.get(r["div"])
        if mdl is None:
            continue
        m = mdl.score_matrix(r["home"], r["away"], r["div"], r.get("mot", 0))
        target = tuple(r.get(f"keskin_{k}", np.nan) for k in ("1", "x", "2"))
        target_ou = (r.get("keskin_ust", np.nan), r.get("keskin_alt", np.nan))
        src = "piyasa" if all(np.isfinite(target)) else "model"
        if src == "piyasa":
            m = calibrate(m, target, target_ou)
        info = {"maç": f"{r['home']} – {r['away']}", "lig": r["div"], "tarih": pd.Timestamp(r["date"]).strftime("%d.%m"),
                "mbs": r.get("mbs"), "kaynak": src, "hg": r.get("fthg", np.nan), "ag": r.get("ftag", np.nan)}
        for bet, p in market_probs(m).items():
            col = ODD_COL.get(bet)
            odd = r.get(col, np.nan) if col else np.nan
            if bet.startswith("2.5") and r.get("ou_line") != 2.5:
                odd = np.nan  # ana baraj 2.5 değilse bu oran başka baraj içindir
            cands.append({**info, "tahmin": bet, "p": p, "odd": odd})
        i, j = np.unravel_index(np.argmax(m), m.shape)
        scores.append({**info, "tahmin": f"{i}-{j}", "p": float(m[i, j]), "odd": np.nan})
    c = pd.DataFrame(cands)
    c["değer"] = c["odd"] * c["p"] - 1
    # 💎 sadece olasılık piyasaya ayarlıysa: sadece modelin gördüğü "değer" backtest'te kapanışı yenmedi
    c["gerçek_değer"] = (c["değer"] > 0) & (c["kaynak"] == "piyasa")
    judged = as_of is not None

    def line(x) -> str:
        s = f"• {x['maç']} ({x['lig']}, {x['tarih']}): **{x['tahmin']}** — olasılık %{100 * x['p']:.0f}, adil oran {1 / x['p']:.2f}"
        if np.isfinite(x["odd"]):
            s += f", {price} {x['odd']:.2f} (değer {100 * x['değer']:+.0f}%{' 💎' if x.get('gerçek_değer') else ''})"
        else:
            s += f", {price} oranı yok"
        if pd.notna(x.get("mbs")) and x.get("mbs") > 1:
            s += f", MBS {int(x['mbs'])}"
        s += "" if x["kaynak"] == "piyasa" else " _(sadece model)_"
        if judged:
            s += f" → {int(x['hg'])}-{int(x['ag'])} {'✅' if bet_won(x['tahmin'], int(x['hg']), int(x['ag'])) else '❌'}"
        return s

    picked = []
    for k, (tname, lo, hi) in enumerate(TIERS):
        t = c[(c["p"] >= lo) & (c["p"] < hi)].copy()
        if k == 0:  # düşük riskte çifte şans ancak MS zaten >= %70 değilse
            t = t[~t["tahmin"].str.startswith("ÇŞ") | ~t["maç"].isin(t.loc[t["tahmin"].str.startswith("MS "), "maç"])]
        # maç başına tek tahmin: önce (piyasaya göre) değeri pozitif olanlar, sonra değer, sonra olasılık
        t["sıra"] = t["gerçek_değer"].astype(int)
        t = t.sort_values(["sıra", "değer", "p"], ascending=False).groupby("maç").head(1)
        t = t.sort_values(["sıra", "p"], ascending=False).head(MAX_PER_TIER[k])
        picked.append((tname, t))
        md += [f"## {tname}", ""] + ([line(x) for x in t.to_dict("records")] or ["Bu seviyede tahmin yok."]) + [""]
    s = pd.DataFrame(scores).sort_values("p", ascending=False)
    picked += [("⚡ Tier 4 — Çok yüksek risk (kısa kesin skor)", s.head(MAX_SHORT_SCORES)),
               ("🔥 Tier 5 — Aşırı risk (tüm maçlarda kesin skor)", s)]
    for tname, t in picked[3:]:
        md += [f"## {tname}", ""] + [line(x) for x in t.to_dict("records")] + [""]
    pos = c[c["gerçek_değer"] & (c["p"] >= 0.25)].sort_values("değer", ascending=False).head(15)
    picked.append(("💎 Değeri pozitif olanlar", pos))
    md += ["## 💎 Değeri pozitif olanlar (risk seviyesinden bağımsız)", "",
           "Uzun vadede kâr beklentisi olan tek grup bu: oran, keskin piyasanın (Pinnacle / Betfair) adil fiyatının "
           "üstünde. Sadece modelin gördüğü değerler buraya alınmaz (backtest'te kapanışı yenmedi).", ""]
    md += [line(x) for x in pos.to_dict("records")] or [f"{price} oranı adil fiyatın üstünde olan tahmin yok."]
    if judged:
        md += ["", "## Sonuç özeti", "", "| seviye | tahmin | tuttu | isabet | oranı olan | kâr/zarar (1 birim) | beklenen isabet |",
               "|---|---|---|---|---|---|---|"]
        for tname, t in picked:
            if t.empty:
                continue
            won = np.array([bet_won(b, int(h), int(a)) for b, h, a in zip(t["tahmin"], t["hg"], t["ag"])])
            has = np.isfinite(t["odd"].to_numpy(float))
            pl = float(np.sum(np.where(won[has], t["odd"].to_numpy(float)[has] - 1, -1.0))) if has.any() else np.nan
            md.append(f"| {tname} | {len(t)} | {int(won.sum())} | %{100 * won.mean():.0f} | {int(has.sum())} | "
                      f"{'—' if not has.any() else f'{pl:+.2f}'} | %{100 * t['p'].mean():.0f} |")
        md += ["", "'Beklenen isabet' = seçilen tahminlerin ortalama olasılığı. Gerçek isabet buna yakınsa olasılıklar "
               "doğru demektir; kâr ise isabetten değil fiyattan (değer) gelir."]
    return "\n".join(md)
