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
from src import iddaa, korner, milli
from src.odds import devig
from src.tavsiye import calibrate

SIDE = {"1": "1", "0": "X", "2": "2"}


def detail_markets(detail: dict) -> dict:
    """Maç detayından tahmin adı -> İddaa oranı (kapalı sonuçlar hariç)."""
    out = {}
    for m in detail.get("m", []):
        t, st, line = m.get("t"), m.get("st"), iddaa._line(m)
        o = {x.get("n"): iddaa._odd(x.get("odd")) for x in m.get("o", [])}
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


def goal_probs(m: np.ndarray, bets) -> dict:
    g = np.arange(m.shape[0])
    h, a = np.meshgrid(g, g, indexing="ij")
    return {b: float(m[_mask(b, h, a)].sum()) for b in bets if not korner.is_corner_bet(b)}


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


def _fmt(bet: str, probs: dict, odds: dict, short: str | None = None) -> str | None:
    if bet not in probs:
        return None
    s = f"{short or bet} %{100 * probs[bet]:.0f}"
    return s + (f" ({odds[bet]:.2f})" if bet in odds else "")


def _picks(probs: dict, odds: dict) -> list:
    """Maç başına öneri: güvenli (%70+, oran 1.20+), orta (%50-70, oran 1.40+), riskli (%25-50)."""
    out = []
    for name, lo, hi, min_odd in [("🟢 Güvenli", 0.70, 1.01, 1.20), ("🟡 Orta", 0.50, 0.70, 1.40),
                                  ("🔴 Riskli", 0.25, 0.50, 1.0)]:
        cand = [(p, b) for b, p in probs.items() if lo <= p < hi and odds.get(b, 0) >= min_odd]
        if cand:
            p, b = max(cand)
            out.append(f"{name}: <b>{html.escape(b)}</b> %{100 * p:.0f} ({odds[b]:.2f})")
    return out


def match_block(r: dict, odds: dict, m: np.ndarray, cmat: np.ndarray | None, now_tr: datetime) -> str:
    probs = {**goal_probs(m, odds), **corner_probs(odds, cmat)}
    probs.update({b: p for b, p in goal_probs(m, [f"{x} Üst" for x in ("1.5", "2.5", "3.5")] + ["KG Var"]).items()
                  if b not in probs})  # İddaa'da kapalı olsa da olasılığı göster
    fav = max(("1", "X", "2"), key=lambda s: probs.get(f"MS {s}", 0))
    j = lambda items: " · ".join(x for x in items if x)  # noqa: E731
    gol_lines = sorted({float(b.split()[0]) for b in probs if re.fullmatch(r"[\d.]+ Üst", b)})
    kor_lines = sorted({float(b.split()[1]) for b in probs if re.fullmatch(r"Korner [\d.]+ Üst", b)})
    lines = [f"⚽ <b>{r['saat']} · {html.escape(r['maç'])}</b> <i>({html.escape(r['lig'])})</i>",
             "<b>MS:</b> " + j(_fmt(f"MS {s}", probs, odds, s) for s in ("1", "X", "2")),
             "<b>Çifte şans:</b> " + j(_fmt(b, probs, odds, b[3:]) for b in ("ÇŞ 1-X", "ÇŞ X-2", "ÇŞ 1-2")),
             "<b>Gol üst:</b> " + j(_fmt(f"{x:g} Üst", probs, odds, f"{x:g}") for x in gol_lines),
             "<b>KG:</b> " + j(_fmt(b, probs, odds, b[3:]) for b in ("KG Var", "KG Yok")),
             f"<b>MS {fav} + gol:</b> " + j(_fmt(f"MS {fav} & {x:g} {s}", probs, odds, f"{x:g} {s}")
                                          for s in ("Üst", "Alt") for x in gol_lines),
             f"<b>MS {fav} + KG:</b> " + j(_fmt(f"MS {fav} & KG {s}", probs, odds, s) for s in ("Var", "Yok"))]
    if kor_lines or any(b.startswith("En çok korner") for b in probs):
        src = "model" if cmat is not None else "İddaa'ya göre"
        lines.append(f"<b>Korner üst</b> <i>({src})</i>: " + j(_fmt(f"Korner {x:g} Üst", probs, odds, f"{x:g}")
                                                              for x in kor_lines))
        lines.append("<b>En çok korner:</b> " + j(_fmt(f"En çok korner {s}", probs, odds, s) for s in ("1", "X", "2")))
    else:
        lines.append("<b>Korner:</b> İddaa'da henüz açık değil")
    lines += _picks({b: p for b, p in probs.items() if b in odds}, odds)
    return "\n".join(x for x in lines if not x.endswith(": "))


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
    cmodels = {}
    if club_divs:
        matches = db.load_matches()
        models.update(_fit_models(matches, {d: ref for d in club_divs},
                                  {"xi": config.XI, "ridge": config.RIDGE, "ridge_total": config.RIDGE_TOTAL}, True))
        cmodels = korner.fit_models(matches, {d: ref for d in club_divs})
    blocks = []
    for r in df.to_dict("records"):
        mdl = models.get(r["div"])
        if mdl is None:
            continue
        odds = detail_markets(iddaa._get(f"event/{int(r['event_id'])}"))
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
        blocks.append(match_block(label, odds, m, cmat, now_tr))
    head = (f"🔎 <b>{day:%d.%m} {saat} maçları — ayrıntılı analiz</b>\n<i>% = tutma olasılığı, parantezde İddaa oranı. "
            "Gol olasılıkları modelin skor tablosundan, İddaa'nın marjsız MS ve 2.5 alt/üst oranına ayarlı. "
            "Milli maçlarda korner verisi olmadığından korner olasılığı İddaa'nın oranından. "
            "Öneri: maç başına güvenli (%70+, oran 1.20+), orta (%50-70, oran 1.40+), riskli (%25-50).</i>")
    return [head] + blocks
