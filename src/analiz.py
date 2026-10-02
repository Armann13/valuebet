"""İddaa maç detayındaki marketlerin ayrıştırılması ve olasılıkları (kart, şut, faul, korner, gol kombinasyonları)
ile seçilen saatteki maçların Tier listesi.

    python main.py analiz --saat 21:45 [--gun 2026-10-02] [--telegram]

Analiz, günlük özetle aynı Tier listesi biçimindedir (tavsiye.run, sadece o saatte başlayan maçlarla): Tier 1-3
(merdiven: Tier 1'deki tahminin Tier 2'de aynı maçta devamı, ör. MS 1 -> MS 1 & KG Var), kesin skor, İY/MS, korner
(kulüp maçları) ve kuponlar. Sadece İddaa'da oynanabilen (oranı olan, kapalı olmayan) tahminler.
"""
import re

import numpy as np
import pandas as pd

import config
from src import iddaa, korner
from src.odds import devig

SIDE = {"1": "1", "0": "X", "2": "2"}


STAT_MARKETS = {912: "Şut", 913: "İsabetli şut", 914: "Faul"}   # "Takım Şut / İsabetli Şut / Faul" (X+ barajları)
STAT_OF = {"Şut": "şut", "İsabetli şut": "isabetli", "Faul": "faul", "Kart": "kart"}


def detail_markets(detail: dict, home: str | None = None, away: str | None = None, keep_closed: bool = False) -> dict:
    """Maç detayından tahmin adı -> İddaa oranı. home / away: İddaa'daki takım adları (takım bazlı şut / faul
    barajlarını ev / deplasmana eşlemek için). Kapalı sonuçlar (1.00) atlanır; keep_closed=True ise NaN olarak kalır."""
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
    return {k: v for k, v in out.items() if v is not None and (keep_closed or np.isfinite(v))}


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


def run(saat: str, gun: str | None = None) -> list[str]:
    """Seçilen saatteki maçların Tier listesi: Telegram mesajları (günlük özetle aynı biçim)."""
    from src import tavsiye, telegram
    _, picks = tavsiye.run(config.RAW_DIR / "fixtures.csv", saat=saat, gun=gun)
    now_tr = telegram._tr_now()
    day = pd.Timestamp(gun).date() if gun else now_tr.date()
    if picks.empty:
        return [f"🔎 <b>{day:%d.%m} · {saat}</b>: sistemin takip ettiği maç yok"]
    header = f"🔎 <b>{day:%d.%m} · {saat} maçları</b> ({picks['maç'].nunique()} maç)"
    return telegram.digest_parts(picks, now_tr, header, [])
