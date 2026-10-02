"""Seçilen maçların ayrıntılı analizi (ör. sadece 21:45 maçları): MS, çifte şans, alt / üst (tüm barajlar), KG,
MS + toplam gol, MS + KG, korner (tüm barajlar, en çok korner); her biri olasılık ve İddaa oranıyla, maç başına
risk seviyesine göre öneri.

Gol marketleri: modelin skor tablosu, İddaa'nın marjsız MS ve 2.5 alt / üst olasılıklarına ayarlı (tavsiye raporuyla
aynı yöntem). Korner: kulüp maçlarında korner modeli; milli maçlarda korner verisi olmadığından İddaa'nın marjı
arındırılmış oranı (her baraj için alt / üst). Oranlar o anki İddaa programından (canlı) çekilir.

    python main.py analiz --saat 21:45 [--gun 2026-10-02] [--telegram]
"""
import html
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import config
from src import iddaa, istatistik, korner, milli
from src.odds import devig
from src.tavsiye import calibrate

SIDE = {"1": "1", "0": "X", "2": "2"}


STAT_MARKETS = {912: "Şut", 913: "İsabetli şut", 914: "Faul"}   # "Takım Şut / İsabetli Şut / Faul" (X+ barajları)
STAT_OF = {"Şut": "şut", "İsabetli şut": "isabetli", "Faul": "faul", "Kart": "kart"}


def detail_markets(detail: dict, home: str | None = None, away: str | None = None) -> dict:
    """Maç detayından tahmin adı -> İddaa oranı (kapalı sonuçlar hariç). home / away: İddaa'daki takım adları
    (takım bazlı şut / faul barajlarını ev / deplasmana eşlemek için)."""
    out = {}
    for m in detail.get("m", []):
        t, st, line = m.get("t"), m.get("st"), iddaa._line(m)
        o = {x.get("n"): iddaa._odd(x.get("odd")) for x in m.get("o", [])}
        if t == 2 and st == 658 and np.isfinite(line):  # toplam kart alt / üst
            out.update({f"Kart {line:g} Alt": o.get("Alt"), f"Kart {line:g} Üst": o.get("Üst")})
        elif t == 2 and st == 824:  # kim daha çok kart görür
            out.update({f"En çok kart {SIDE[k]}": v for k, v in o.items() if k in SIDE})
        elif t == 2 and st in STAT_MARKETS:  # "İspanya 8+", "Toplam 10+" (ilk yarı barajları atlanır)
            for k, v in o.items():
                mt = re.fullmatch(r"(.+) (\d+)\+", str(k))
                if not mt or mt[1].startswith("İY "):
                    continue
                side = {home: "Ev", away: "Dep", "Toplam": "Toplam"}.get(mt[1])
                if side:
                    out[f"{STAT_MARKETS[st]} {side} {mt[2]}+"] = v
        elif t == 2 and st == 12:  # oyuncu gol atar
            out.update({f"Golcü: {k}": v for k, v in o.items()})
        if t == 1 and st == 1:
            out.update({f"MS {SIDE[k]}": v for k, v in o.items() if k in SIDE})
        elif t == 2 and st == 92:
            out.update({"ÇŞ 1-X": o.get("1 ve 0"), "ÇŞ 1-2": o.get("1 ve 2"), "ÇŞ X-2": o.get("0 ve 2")})
        elif t == 2 and st == 101 and np.isfinite(line):
            out.update({f"{line:g} Alt": o.get("Alt"), f"{line:g} Üst": o.get("Üst")})
        elif t == 2 and st == 89:
            out.update({"KG Var": o.get("Var"), "KG Yok": o.get("Yok")})
        elif t == 2 and st == 7 and np.isfinite(line):
            for k, v in o.items():
                side, _, ou = k.partition(" ve ")
                if side in SIDE:
                    out[f"MS {SIDE[side]} & {line:g} {ou}"] = v
        elif t == 2 and st == 698:
            for k, v in o.items():
                side, _, kg = k.partition(" ve ")
                if side in SIDE:
                    out[f"MS {SIDE[side]} & KG {kg}"] = v
        elif t == 2 and st == 48 and np.isfinite(line):
            out.update({f"Korner {line:g} Alt": o.get("Alt"), f"Korner {line:g} Üst": o.get("Üst")})
        elif t == 2 and st == 52:
            out.update({f"En çok korner {SIDE[k]}": v for k, v in o.items() if k in SIDE})
    return {k: v for k, v in out.items() if v is not None and np.isfinite(v)}


def _mask(bet: str, h: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Gol tahmini (ör. "MS 1 & 2.5 Üst", "ÇŞ X-2", "KG Var") -> skor tablosunda tuttuğu hücreler."""
    out = np.ones_like(h, dtype=bool)
    for part in bet.split(" & "):
        if part.startswith("MS "):
            out &= {"1": h > a, "X": h == a, "2": h < a}[part[3:]]
        elif part.startswith("ÇŞ "):
            out &= {"1-X": h >= a, "X-2": h <= a, "1-2": h != a}[part[3:]]
        elif part.startswith("KG "):
            out &= ((h >= 1) & (a >= 1)) if part.endswith("Var") else ((h == 0) | (a == 0))
        else:
            line, side = re.fullmatch(r"([\d.]+) (Alt|Üst)", part).groups()
            out &= (h + a > float(line)) if side == "Üst" else (h + a < float(line))
    return out


def _is_goal_bet(b: str) -> bool:
    return not (korner.is_corner_bet(b) or b.startswith(("Kart ", "En çok kart", "Golcü:")) or
                b.split()[0] in ("Şut", "İsabetli", "Faul"))


def goal_probs(m: np.ndarray, bets) -> dict:
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    return {b: float(m[_mask(b, h, a)].sum()) for b in bets if _is_goal_bet(b)}


def stat_probs(odds: dict, smod: dict) -> tuple[dict, set]:
    """Kart / şut / isabetli şut / faul olasılıkları: kulüplerde model (smod: istatistik adı -> probs çıktısı),
    milli maçlarda İddaa'dan: iki yönlü marketler marjı arındırılarak, tek yönlü "X+" barajları ham (1 / oran, marj
    dahil; öneriye alınmaz). Döner: olasılıklar ve ham (yaklaşık) olanların kümesi."""
    out, approx = {}, set()
    for b, odd in odds.items():
        if b.startswith("Kart "):
            _, line, side = b.split()
            if "kart" in smod:
                pr = smod["kart"]["toplam"]
                x = int(np.floor(float(line))) + 1  # 4.5 Üst = en az 5
                p_ust = float(pr[x]) if x < len(pr) else 0.0
                out[b] = p_ust if side == "Üst" else 1 - p_ust
            else:
                pair = [odds.get(f"Kart {line} Alt"), odds.get(f"Kart {line} Üst")]
                if all(v is not None for v in pair):
                    fair = devig(pair, config.DEVIG_METHOD)
                    out[b] = float(fair[0] if side == "Alt" else fair[1])
        elif b.startswith("En çok kart"):
            if "kart" in smod:
                out[b] = smod["kart"][b[-1]]
            else:
                trio = [odds.get(f"En çok kart {s}") for s in ("1", "X", "2")]
                if all(v is not None for v in trio):
                    out[b] = float(devig(trio, config.DEVIG_METHOD)["1X2".index(b[-1])])
        elif b.split()[0] in ("Şut", "İsabetli", "Faul"):
            label, side, x = b.rsplit(" ", 2)
            stat, x = STAT_OF[label], int(x[:-1])
            if stat in smod:
                pr = smod[stat][{"Ev": "ev", "Dep": "dep", "Toplam": "toplam"}[side]]
                out[b] = float(pr[x]) if x < len(pr) else 0.0
            else:
                out[b] = 1 / odd
                approx.add(b)
        elif b.startswith("Golcü:"):
            out[b] = 1 / odd
            approx.add(b)
    return out, approx


def corner_probs(odds: dict, cmat: np.ndarray | None) -> dict:
    """Korner olasılıkları: korner modeli varsa ondan (kulüp), yoksa İddaa'nın marjsız iki yönlü oranından (milli)."""
    out = {}
    for b in [b for b in odds if korner.is_corner_bet(b)]:
        if cmat is not None:
            line = float(b.split()[1]) if b.startswith("Korner") else None
            out[b] = korner.market_probs(cmat, line)[b]
        elif b.startswith("Korner"):
            _, line, side = b.split()
            pair = [odds.get(f"Korner {line} Alt"), odds.get(f"Korner {line} Üst")]
            if all(v is not None for v in pair):
                fair = devig(pair, config.DEVIG_METHOD)
                out[b] = float(fair[0] if side == "Alt" else fair[1])
        else:
            trio = [odds.get(f"En çok korner {s}") for s in ("1", "X", "2")]
            if all(v is not None for v in trio):
                out[b] = float(devig(trio, config.DEVIG_METHOD)["1X2".index(b[-1])])
    return out


NAME_W = 14   # tablolarda bahis adı sütunu (telefonda tek satıra sığsın)


def _row(label: str, probs: dict, odds: dict, bet: str, approx: set = frozenset()) -> str | None:
    """Tablo satırı: "2.5 Üst          %68   1.37" (eşit aralıklı yazıyla hizalı)."""
    if bet not in probs:
        return None
    pct = ("~" if bet in approx else "") + f"%{100 * probs[bet]:.0f}"
    odd = f"{odds[bet]:.2f}" if bet in odds else "—"
    return f"{label[:NAME_W]:<{NAME_W}}{pct:>5}  {odd:>5}"


def _short_name(name: str) -> str:
    """Uzun oyuncu adı: "Roman Yaremchuk" -> "R. Yaremchuk" (tablo sütununa sığsın)."""
    parts = name.split()
    return name if len(name) <= NAME_W or len(parts) < 2 else f"{parts[0][0]}. {' '.join(parts[1:])}"


def _table(title: str, rows) -> list:
    rows = [r for r in rows if r]
    return ["", title, "<pre>" + html.escape("\n".join(rows)) + "</pre>"] if rows else []


def _picks(probs: dict, odds: dict) -> list:
    """Maç başına öneri: güvenli (%70+, oran 1.20+), orta (%50-70, oran 1.40+), riskli (%25-50)."""
    out = []
    for icon, lo, hi, min_odd in [("🟢", 0.70, 1.01, 1.20), ("🟡", 0.50, 0.70, 1.40), ("🔴", 0.25, 0.50, 1.0)]:
        cand = [(p, b) for b, p in probs.items() if lo <= p < hi and odds.get(b, 0) >= min_odd]
        if cand:
            p, b = max(cand)
            out.append(f"{icon} <b>{html.escape(b)}</b> — %{100 * p:.0f} · oran <b>{odds[b]:.2f}</b>")
    return out


def match_block(r: dict, odds: dict, m: np.ndarray, cmat: np.ndarray | None, now_tr: datetime,
                smod: dict | None = None) -> str:
    """Tek maçın mesajı: başlık, öneriler, sonra konu konu hizalı tablolar (bahis · olasılık · İddaa oranı)."""
    smod = smod or {}
    sp, approx = stat_probs(odds, smod)
    probs = {**goal_probs(m, odds), **corner_probs(odds, cmat), **sp}
    probs.update({b: p for b, p in goal_probs(m, [f"{x} Üst" for x in ("1.5", "2.5", "3.5")] + ["KG Var"]).items()
                  if b not in probs})  # İddaa'da kapalı olsa da olasılığı göster
    fav = max(("1", "X", "2"), key=lambda s: probs.get(f"MS {s}", 0))
    src = lambda model: "model" if model else "İddaa'ya göre"  # noqa: E731
    out = [f"⚽ <b>{html.escape(r['maç'])}</b>", f"🕘 {r['saat']} · {html.escape(r['lig'])}", "", "⭐ <b>ÖNERİLER</b>"]
    out += _picks({b: p for b, p in probs.items() if b in odds and b not in approx}, odds) or ["uygun öneri yok"]
    out += _table("🎯 <b>Maç sonucu</b>", [_row(f"MS {s}", probs, odds, f"MS {s}") for s in ("1", "X", "2")] +
                  [_row(b, probs, odds, b) for b in ("ÇŞ 1-X", "ÇŞ X-2", "ÇŞ 1-2")])
    out += _table("⚽ <b>Goller</b>", [_row(f"{x} Üst", probs, odds, f"{x} Üst") for x in ("1.5", "2.5", "3.5", "4.5")] +
                  [_row("2.5 Alt", probs, odds, "2.5 Alt"), _row("KG Var", probs, odds, "KG Var"),
                   _row("KG Yok", probs, odds, "KG Yok")])
    combos = sorted((b for b in probs if b.startswith(f"MS {fav} & ") and b in odds and probs[b] >= 0.15),
                    key=lambda b: -probs[b])[:5]
    out += _table(f"🔗 <b>MS {fav} + gol</b>", [_row(b[3:], probs, odds, b) for b in combos])
    kor = sorted((b for b in probs if re.fullmatch(r"Korner [\d.]+ Üst", b) and 0.1 <= probs[b] <= 0.95),
                 key=lambda b: float(b.split()[1]))
    kfav = max(("1", "X", "2"), key=lambda s: probs.get(f"En çok korner {s}", -1))
    out += _table(f"🚩 <b>Korner</b> <i>({src(cmat is not None)})</i>",
                  [_row(b[7:], probs, odds, b) for b in kor] +
                  [_row(f"En çok: {kfav}", probs, odds, f"En çok korner {kfav}")])
    kart = sorted((b for b in probs if re.fullmatch(r"Kart [\d.]+ Üst", b)), key=lambda b: float(b.split()[1]))
    cfav = max(("1", "X", "2"), key=lambda s: probs.get(f"En çok kart {s}", -1))
    out += _table(f"🟨 <b>Kart</b> <i>({src('kart' in smod)})</i>",
                  [_row(b[5:], probs, odds, b) for b in kart] + [_row(f"En çok: {cfav}", probs, odds, f"En çok kart {cfav}")])
    stat_rows = []
    for label, short in (("İsabetli şut", "İsabetli"), ("Şut", "Şut"), ("Faul", "Faul")):
        tot = [b for b in probs if b.startswith(f"{label} Toplam ")][:3]  # toplam barajları
        stat_rows += [_row(f"{short} {b.split()[-1]}", probs, odds, b, approx) for b in tot]
    out += _table(f"👟 <b>Şut / faul (toplam)</b> <i>({src(bool({'şut', 'isabetli', 'faul'} & set(smod)))})</i>",
                  stat_rows)
    scorers = sorted((b for b in probs if b.startswith("Golcü:")), key=lambda b: -probs[b])[:4]
    out += _table("🥅 <b>Gol atabilir</b> <i>(İddaa'ya göre)</i>",
                  [_row(_short_name(b[7:]), probs, odds, b, approx) for b in scorers])
    return "\n".join(out)


def run(saat: str, gun: str | None = None) -> list[str]:
    """Döner: maç başına Telegram (HTML) metinleri; ilk eleman başlık."""
    from src import db
    from src.predict import _fit_models
    now_tr = datetime.now(timezone.utc) + timedelta(hours=3)
    day = pd.Timestamp(gun).date() if gun else now_tr.date()
    raw = iddaa._get("competitions")
    comps = iddaa.resolve_competitions(raw)
    names = {c["i"]: str(c.get("n", "")).strip() for c in raw}
    events = iddaa._get("events?st=1&type=0&version=0")["events"]
    df = iddaa.parse_events(events, comps, names)
    if df.empty:
        return [f"{saat}: takip edilen maç yok"]
    tr = pd.to_datetime(df["kickoff_utc"]) + pd.Timedelta(hours=3)
    df = df[(tr.dt.date == day) & (tr.dt.strftime("%H:%M") == saat)].copy()
    df, _ = iddaa._map_teams(df)
    df = df.dropna(subset=["home", "away"])
    if df.empty:
        return [f"{day:%d.%m} {saat}: sistemin takip ettiği maç yok"]
    ref = pd.Timestamp(day)
    models = {}
    if (df["div"] == config.MILLI).any():
        models[config.MILLI] = milli.fit_current(ref)
    club_divs = sorted(set(df["div"]) - {config.MILLI})
    cmodels, smodels = {}, {}
    if club_divs:
        matches = db.load_matches()
        refs = {d: ref for d in club_divs}
        models.update(_fit_models(matches, refs,
                                  {"xi": config.XI, "ridge": config.RIDGE, "ridge_total": config.RIDGE_TOTAL}, True))
        cmodels = korner.fit_models(matches, refs)
        smodels = {s: istatistik.fit_models(matches, refs, s) for s in istatistik.STATS}
    blocks = []
    for r in df.to_dict("records"):
        mdl = models.get(r["div"])
        if mdl is None:
            continue
        odds = detail_markets(iddaa._get(f"event/{int(r['event_id'])}"), r["home_iddaa"], r["away_iddaa"])
        smod = {s: istatistik.probs(*ms[r["div"]].expected_goals(r["home"], r["away"], r["div"]), s)
                for s, ms in smodels.items() if r["div"] in ms}
        m = mdl.score_matrix(r["home"], r["away"], r["div"])
        fair = devig([odds.get(f"MS {s}", np.nan) for s in ("1", "X", "2")], config.DEVIG_METHOD)
        ou = devig([odds.get("2.5 Üst", np.nan), odds.get("2.5 Alt", np.nan)], config.DEVIG_METHOD)
        if np.isfinite(fair).all() or np.isfinite(ou).all():
            m = calibrate(m, tuple(fair), tuple(ou))  # tavsiye raporuyla aynı: İddaa'nın marjsız MS ve 2.5'ine ayarlı
        cm = cmodels.get(r["div"])
        cmat = korner.matrix(*cm.expected_goals(r["home"], r["away"], r["div"])) if cm is not None else None
        comp = r.get("competition")
        label = {"maç": f"{r['home_iddaa']} – {r['away_iddaa']}", "saat": saat,
                 "lig": comp.split(",")[0] if isinstance(comp, str) and comp else r["div"]}
        blocks.append(match_block(label, odds, m, cmat, now_tr, smod))
    head = "\n".join([
        f"🔎 <b>{day:%d.%m} · {saat} maçları</b> ({len(blocks)} maç)", "",
        "Her maç ayrı mesajda:",
        "⭐ en üstte öneriler: 🟢 güvenli (%70+) · 🟡 orta (%50-70) · 🔴 riskli (%25-50)",
        "📋 tablolarda: <b>bahis · tutma olasılığı · İddaa oranı</b>", "",
        "<i>Gol olasılıkları model (İddaa'nın marjsız oranına ayarlı). Korner / kart / şut: kulüplerde model, milli "
        "maçlarda İddaa'nın oranından. ~ = İddaa'nın oranından, marj dahil (gerçekte biraz daha düşük).</i>"])
    return [head] + blocks
