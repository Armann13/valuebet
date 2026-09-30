"""Oran yardımcıları: ima edilen olasılık, marj, marjı temizleme (de-vig) ve keskin piyasa sütunları."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

import config

MARKET_SUFFIXES = {"1x2": ["h", "d", "a"], "ou": ["o25", "u25"]}


def margin(odds) -> float:
    return float(np.sum(1.0 / np.asarray(odds, dtype=float)) - 1.0)


def devig(odds, method: str = "power") -> np.ndarray:
    """Bir marketin oranlarından (ör. [H, D, A]) marjsız 'adil' olasılıkları çıkarır.

    proportional: 1/oran değerlerini toplamları 1 olacak şekilde ölçekler.
    power:        p_i = (1/oran_i)^k, sum p_i = 1 olacak k'yı bulur. Favori-sürpriz yanlılığını
                  proportional yönteme göre daha iyi yakalar.
    """
    o = np.asarray(odds, dtype=float)
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        return np.full(o.shape, np.nan)
    p = 1.0 / o
    if method == "proportional":
        return p / p.sum()
    if method == "power":
        k = brentq(lambda k: np.sum(p ** k) - 1.0, 0.05, 50.0)
        q = p ** k
        return q / q.sum()
    raise ValueError(f"bilinmeyen yöntem: {method}")


def devig_frame(df, cols, method: str = "power") -> np.ndarray:
    """DataFrame satırları için toplu (vektörel) de-vig; sonuç satır satır devig() ile aynıdır.
    Eksik ya da hatalı (<= 1.0) oranlı satırlar NaN döner."""
    arr = df[cols].to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr).all(axis=1) & (arr > 1.0).all(axis=1)
    p = 1.0 / arr[ok]
    if method == "proportional":
        out[ok] = p / p.sum(axis=1, keepdims=True)
    elif method == "power":
        # sum(p^k) k'da azalır; her satırın kökü devig()'deki aralıkta ikiye bölmeyle bulunur
        lo, hi = np.full(len(p), 0.05), np.full(len(p), 50.0)
        for _ in range(60):
            k = (lo + hi) / 2
            big = (p ** k[:, None]).sum(axis=1) > 1.0
            lo, hi = np.where(big, k, lo), np.where(big, hi, k)
        q = p ** ((lo + hi) / 2)[:, None]
        out[ok] = q / q.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"bilinmeyen yöntem: {method}")
    return out


def add_sharp(df: pd.DataFrame) -> pd.DataFrame:
    """Keskin piyasa oranlarını ekler: sharp_* (açılış) ve sharpc_* (kapanış), bkz. config.SHARP_SOURCES.
    Her market için satır bazında ilk tam ve geçerli kaynak alınır (bir marketin oranları hep aynı
    kaynaktan gelir). Marjı EXCHANGE_MAX_MARGIN'i aşan borsa fiyatı geçersiz sayılır. Kullanılan kaynak
    {ad}_src_{market} sütununda (ör. sharpc_src_1x2 = "psc" / "bfec" / None)."""
    new = {}
    for name, sources in config.SHARP_SOURCES.items():
        for mk, sufs in MARKET_SUFFIXES.items():
            out = np.full((len(df), len(sufs)), np.nan)
            src = np.full(len(df), None, dtype=object)
            filled = np.zeros(len(df), dtype=bool)
            for book in sources:
                cols = [f"{book}_{s}" for s in sufs]
                if not all(c in df for c in cols):
                    continue
                arr = df[cols].to_numpy(dtype=float)
                ok = np.isfinite(arr).all(axis=1) & (arr > 1.0).all(axis=1)
                if book in config.EXCHANGE_BOOKS:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ok &= (1 / arr).sum(axis=1) - 1 <= config.EXCHANGE_MAX_MARGIN
                take = ok & ~filled
                out[take], src[take], filled = arr[take], book, filled | take
            for k, s in enumerate(sufs):
                new[f"{name}_{s}"] = out[:, k]
            new[f"{name}_src_{mk}"] = src
    return pd.concat([df.drop(columns=[c for c in new if c in df]), pd.DataFrame(new, index=df.index)], axis=1)
