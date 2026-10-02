"""İY/MS (ilk yarı / maç sonu) olasılıkları ve kombine / sistem kupon hesabı.

İY/MS: maç sonu skor tablosu (piyasaya ayarlı) verildiğinde her golün ilk yarıda atılma olasılığı ILK_YARI (veride
%44.6; ev ve deplasman aynı); ilk yarı golleri binom dağılımıyla bölünür. Bu varsayım geçmiş maçlarda tutuyor (ör. 2-1
biten maçlarda ilk yarı X / 1 / 2 gerçek %38 / %46 / %16, hesap %39 / %47 / %14).

Kupon: maçlar bağımsız sayılır. Kombine kuponun tutma olasılığı = olasılıkların çarpımı, oranı = oranların çarpımı.
Sistem kupon "n'de k": n maçın her k'lı kombinasyonu ayrı kolon (1'er birim); en az k maç tutarsa tutan kolonların
oranları toplamı kadar ödeme. Olasılıklar n maçın tüm sonuç kombinasyonları (2^n) üzerinden tam hesaplanır.
"""
from itertools import combinations, product
from math import comb

import numpy as np

ILK_YARI = 0.446
SONUC = {0: "X", 1: "1", 2: "2"}


def _res(h, a):
    return np.where(h > a, 1, np.where(h == a, 0, 2))


def iyms_probs(m: np.ndarray, f: float = ILK_YARI) -> dict:
    """{"İY/MS 1/1": p, ...}: m[ev golü, deplasman golü] maç sonu skor tablosu."""
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
    return {f"İY/MS {SONUC[i]}/{SONUC[j]}": float(out[i, j]) for i in (1, 0, 2) for j in (1, 0, 2)}


def iyms_won(bet: str, ihg: int, iag: int, hg: int, ag: int) -> bool:
    ht, ft = bet.split()[1].split("/")
    side = lambda h, a: "1" if h > a else "X" if h == a else "2"  # noqa: E731
    return side(ihg, iag) == ht and side(hg, ag) == ft


def is_iyms(bet: str) -> bool:
    return str(bet).startswith("İY/MS ")


def best_combo(cands: list[dict], sizes=(3, 4), min_odds: float = 3.0, max_odds: float = 6.0) -> list[dict]:
    """Orta riskli kombine: farklı maçlardan sizes kadar tahmin; toplam oran [min_odds, max_odds] içinde, tutma
    olasılığı en yüksek olan. MBS (minimum maç sayısı) kurala uyar. cands: {"maç", "p", "odd", "mbs", ...}."""
    best, best_p = [], -1.0
    for n in sizes:
        for combo in combinations(cands, n):
            if len({c["maç"] for c in combo}) < n:
                continue
            if max(int(c.get("mbs") or 1) for c in combo) > n:
                continue
            odds = float(np.prod([c["odd"] for c in combo]))
            p = float(np.prod([c["p"] for c in combo]))
            if min_odds <= odds <= max_odds and p > best_p:
                best, best_p = list(combo), p
    return best


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
