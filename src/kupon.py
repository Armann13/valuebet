"""İY/MS (ilk yarı / maç sonu) olasılıkları ve sistem kupon hesabı.

İY/MS: maç sonu skor tablosu (piyasaya ayarlı) verildiğinde her golün ilk yarıda atılma olasılığı ILK_YARI (veride
%44.6; ev ve deplasman aynı); ilk yarı golleri binom dağılımıyla bölünür. Bu varsayım geçmiş maçlarda tutuyor (ör. 2-1
biten maçlarda ilk yarı X / 1 / 2 gerçek %38 / %46 / %16, hesap %39 / %47 / %14).

Sistem kupon "n'de k": maçlar bağımsız sayılır; n maçın her k'lı kombinasyonu ayrı kolon (1'er birim); en az k maç
tutarsa tutan kolonların oranları toplamı kadar ödeme. Olasılıklar n maçın tüm sonuç kombinasyonları (2^n) üzerinden.
"""
from itertools import combinations, product
from math import comb

import numpy as np

ILK_YARI = 0.446
SONUC = {0: "X", 1: "1", 2: "2"}
# Binom ayrımı "dönüşleri" az veriyor: geride kalan ikinci yarıda yükleniyor (oyun durumu etkisi). 2016-2026 kulüp
# maçlarında (24.000, skor tablosu B365 1X2 + 2.5'e ayarlı) gerçek / beklenen. 2016-22'de öğrenilip 2022/23+'da
# test: 9 sonuçlu log loss 1.8910 -> 1.8889; 1/2 beklenen 251 -> 334 (gerçek 345), 2/X 687 -> 757 (gerçek 758).
IYMS_DUZELTME = {"1/1": 0.953, "1/X": 1.116, "1/2": 1.359, "X/1": 0.997, "X/X": 0.996, "X/2": 0.973,
                 "2/1": 1.216, "2/X": 1.118, "2/2": 0.969}


def _res(h, a):
    return np.where(h > a, 1, np.where(h == a, 0, 2))


def iyms_probs(m: np.ndarray, f: float = ILK_YARI, duzelt: bool = True) -> dict:
    """{"İY/MS 1/1": p, ...}: m[ev golü, deplasman golü] maç sonu skor tablosu. duzelt: dönüşler için IYMS_DUZELTME
    (maç sonu olasılıkları korunur)."""
    n = m.shape[0]
    g = np.arange(n)
    # B[k, j] = P(ilk yarıda j gol | maçta k gol)
    B = np.array([[comb(k, j) * f ** j * (1 - f) ** (k - j) if j <= k else 0.0 for j in g] for k in g])
    hj, aj = np.meshgrid(g, g, indexing="ij")
    ht_res = _res(hj, aj)
    out = np.zeros((3, 3))
    for H in range(n):
        for A in range(n):
            if m[H, A] < 1e-12:
                continue
            ht = np.outer(B[H], B[A])
            ft = int(_res(np.array(H), np.array(A)))
            for r in (0, 1, 2):
                out[r, ft] += m[H, A] * ht[ht_res == r].sum()
    if duzelt:
        for j in (0, 1, 2):  # maç sonu sütunu içinde yeniden dağıt: maç sonu olasılığı aynı kalır
            adj = out[:, j] * np.array([IYMS_DUZELTME[f"{SONUC[i]}/{SONUC[j]}"] for i in (0, 1, 2)])
            if adj.sum() > 0:
                out[:, j] = adj * out[:, j].sum() / adj.sum()
    return {f"İY/MS {SONUC[i]}/{SONUC[j]}": float(out[i, j]) for i in (1, 0, 2) for j in (1, 0, 2)}


def iyms_won(bet: str, ihg: int, iag: int, hg: int, ag: int) -> bool:
    ht, ft = bet.split()[1].split("/")
    side = lambda h, a: "1" if h > a else "X" if h == a else "2"  # noqa: E731
    return side(ihg, iag) == ht and side(hg, ag) == ft


def is_iyms(bet: str) -> bool:
    return str(bet).startswith("İY/MS ")


def system_table(ps: list[float], odds: list[float], ks=None) -> list[dict]:
    """Sistem kupon "n'de k" için: kolon sayısı, en az k tutma olasılığı, kâr olasılığı, beklenen dönüş (1 birim
    başına). Tüm 2^n sonuç kombinasyonu üzerinden."""
    n = len(ps)
    ks = ks or [k for k in range(max(2, n - 3), n + 1)]
    ps, odds = np.asarray(ps, float), np.asarray(odds, float)
    rows = []
    for k in ks:
        cols = list(combinations(range(n), k))
        p_any, p_profit, exp_ret = 0.0, 0.0, 0.0
        for outcome in product((0, 1), repeat=n):
            w = np.array(outcome, bool)
            prob = float(np.prod(np.where(w, ps, 1 - ps)))
            if w.sum() < k:
                continue
            payout = sum(float(np.prod(odds[list(c)])) for c in cols if w[list(c)].all())
            p_any += prob
            exp_ret += prob * payout / len(cols)
            p_profit += prob * (payout > len(cols))
        rows.append({"k": k, "kolon": len(cols), "en_az_k": p_any, "kar_olasiligi": p_profit,
                     "beklenen_donus": exp_ret})
    return rows


def best_system(ps: list[float], odds: list[float]) -> dict:
    """Önerilen "n'de k": kâra geçme olasılığı en yüksek olan (eşitse daha büyük k, daha az kolon)."""
    return max(system_table(ps, odds), key=lambda s: (round(s["kar_olasiligi"], 3), s["k"]))


def system_payout(won: list[bool], odds: list[float], k: int) -> float:
    """Sistem kuponun dönüşü (kolon başına 1 birim): tüm ayakları tutan k'lı kolonların oran çarpımları toplamı."""
    w, odds = np.asarray(won, bool), np.asarray(odds, float)
    return sum(float(np.prod(odds[list(c)])) for c in combinations(range(len(w)), k) if w[list(c)].all())
