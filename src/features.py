"""Maç öncesi puan durumundan türetilen özellikler.

Motivasyon: sezon sonuna doğru bir takımın hâlâ "hedefi" var mı? Her lig için sıralama sınırları tanımlıdır
(ör. La Liga'da şampiyonluk 1|2, Şampiyonlar Ligi 4|5, Avrupa 7|8, küme 17|18; bkz. BOUNDARIES). Takım, kalan maç başına
`reach` puan içinde yetişebileceği ya da kaybedebileceği bir sınır varsa hedeflidir. Sezonun büyük kısmında
herkes hedeflidir; ayrım son haftalarda ortaya çıkar. mot = ev hedefli - deplasman hedefli ∈ {-1, 0, +1}.

Puan durumu sadece maç gününden ÖNCEKİ sonuçlardan hesaplanır (geleceği görme yok). Sıralama puan, averaj,
atılan gol ile yapılır (La Liga'nın ikili averaj kuralı yaklaşık olarak atlanır).
"""
import numpy as np
import pandas as pd

import config

# lig -> {sınır adı: p}; sınır p ile p+1. sıra arasıdır. p < 0 ise takım sayısından sayılır (-3 = son 3 düşer).
# Kotalar yaklaşıktır (Avrupa kotası, play-off biçimi yıllara göre değişir); amaç "hâlâ bir hedefi var mı".
BOUNDARIES = {
    "E0": {"şampiyonluk": 1, "Şampiyonlar Ligi": 4, "Avrupa": 7, "küme": -3},
    "E1": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 6, "küme": -3},
    "E2": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 6, "küme": -4},
    "E3": {"şampiyonluk": 1, "direkt çıkış": 3, "play-off": 7, "küme": -2},
    "EC": {"şampiyonluk": 1, "play-off": 7, "küme": -4},
    "SC0": {"şampiyonluk": 1, "Avrupa": 4, "küme": -2},
    "SC1": {"şampiyonluk": 1, "play-off": 4, "küme": -2},
    "SC2": {"şampiyonluk": 1, "play-off": 4, "küme": -2},
    "SC3": {"şampiyonluk": 1, "play-off": 4, "küme": -1},
    "D1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 4, "Avrupa": 7, "küme": -3},
    "D2": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 3, "küme": -3},
    "I1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 4, "Avrupa": 7, "küme": -3},
    "I2": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 8, "küme": -5},
    "SP1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 4, "Avrupa": 7, "küme": -3},
    "SP2": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 6, "küme": -4},
    "F1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 3, "Avrupa": 6, "küme": -3},
    "F2": {"şampiyonluk": 1, "direkt çıkış": 2, "play-off": 5, "küme": -3},
    "N1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 2, "Avrupa": 8, "küme": -3},
    "B1": {"şampiyonluk": 1, "şampiyonluk grubu": 6, "küme": -2},
    "P1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 2, "Avrupa": 5, "küme": -3},
    "T1": {"şampiyonluk": 1, "Şampiyonlar Ligi": 2, "Avrupa": 4, "küme": -4},
    "G1": {"şampiyonluk": 1, "şampiyonluk grubu": 4, "küme": -2},
}


def _live(div: str, pts: int, pos: int, remaining: int, table: list, reach: float) -> bool:
    """table: puanlar, sıraya göre azalan. Takım herhangi bir sınıra kalan maç başına reach puan uzaklıkta mı?"""
    if div not in BOUNDARIES:
        return True
    r, n = reach * remaining, len(table)
    for b in BOUNDARIES[div].values():
        b = b if b > 0 else n + b
        if not 1 <= b < n:
            continue
        if pos <= b and pts - table[b] <= r:       # sınırın üstünde; alttaki yetişebilir
            return True
        if pos > b and table[b - 1] - pts <= r:    # sınırın altında; üsttekine yetişebilir
            return True
    return False


def add_motivation(results: pd.DataFrame, queries: pd.DataFrame | None = None,
                   reach: float = config.MOTIVATION_REACH) -> pd.DataFrame:
    """queries'e (verilmezse results'a) live_home, live_away ve mot sütunlarını ekler. results: sonucu belli
    maçlar (div, season, date, home, away, fthg, ftag); queries: aynı sütunlar, sonuç gerekmez (fikstür)."""
    q = (results if queries is None else queries).copy()
    played = results.dropna(subset=["fthg", "ftag"])
    live = {}
    for (div, season), qg in q.groupby(["div", "season"]):
        rg = played[(played["div"] == div) & (played["season"] == season)]
        teams = sorted(set(rg["home"]) | set(rg["away"]) | set(qg["home"]) | set(qg["away"]))
        total = 2 * (len(teams) - 1)
        pts, gd, gf, n_played = (dict.fromkeys(teams, 0) for _ in range(4))
        by_date = {d: g for d, g in rg.groupby("date")}
        for date in sorted(set(by_date) | set(qg["date"])):
            order = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))
            table = [pts[t] for t in order]
            pos = {t: i + 1 for i, t in enumerate(order)}
            for t in set(qg.loc[qg["date"] == date, "home"]) | set(qg.loc[qg["date"] == date, "away"]):
                live[(div, season, date, t)] = _live(div, pts[t], pos[t], total - n_played[t], table, reach)
            for r in by_date.get(date, pd.DataFrame()).itertuples():
                hg, ag = int(r.fthg), int(r.ftag)
                pts[r.home] += 3 if hg > ag else 1 if hg == ag else 0
                pts[r.away] += 3 if ag > hg else 1 if hg == ag else 0
                gd[r.home] += hg - ag
                gd[r.away] += ag - hg
                gf[r.home] += hg
                gf[r.away] += ag
                n_played[r.home] += 1
                n_played[r.away] += 1
    q["live_home"] = [live.get(k, True) for k in zip(q["div"], q["season"], q["date"], q["home"])]
    q["live_away"] = [live.get(k, True) for k in zip(q["div"], q["season"], q["date"], q["away"])]
    q["mot"] = q["live_home"].astype(int) - q["live_away"].astype(int)
    return q
