"""Risk seviyesine göre tavsiye raporu (yaklaşan maçlar ya da geçmiş bir hafta sonunun provası).

Her maç için modelin skor olasılık tablosu (Dixon-Coles) kurulur. Keskin piyasanın (Pinnacle / Betfair) 1X2
ve 2.5 alt/üst olasılıkları varsa tablo onlara göre ayarlanır (bölgeler sırayla ölçeklenir; toplamlar piyasayla
aynı olur, skor dağılımının geri kalanı modelden gelir). Piyasa modelden isabetli olduğu için
olasılıklar böylece daha doğru, ve tüm marketler (MS, çifte şans, alt/üst, KG, kombinasyon, kesin skor) aynı
tablodan geldiği için birbiriyle tutarlıdır. Keskin oran yoksa (ör. milli maçlar) tablo fiyatın (İddaa) marjı
arındırılmış oranlarına ayarlanır: kulüp backtest'inde yumuşak büronun (B365) marjsız oranları da modelden
isabetliydi (1X2 log loss farkı 0.013-0.017). Bu durumda değer ölçülemez (fiyat kendisiyle kıyaslanır).
Milli maçlar ayrı bir modelle (src/milli.py) tahmin edilir.

Risk seviyesi = tahminin TUTMAMA olasılığı:
  Tier 1 düşük (olasılık >= %70), Tier 2 orta (%50-70), Tier 3 yüksek (%25-50: beraberlik, sürpriz, kombinasyon),
  Tier 4 çok yüksek (%18-38, oran en az 2.30: MS & KG / Üst, sürpriz MS, İY/MS, yüksek korner barajları),
  Tier 5 aşırı (en olası kesin skor, kısa liste), Tier 6 (her maçın en olası kesin skoru).
  Tier 1, 2 ve 4'te oranı MIN_ODDS'un altındaki tahminler (ör. 1.06'lık çifte şans) listelenmez.
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
from src import db, korner, milli
from src.features import add_motivation
from src.iddaa import KOR_LINES_COL, ODD_COLS, SCORE_COL, corner_lines, score_odds
from src.odds import add_sharp, devig, devig_frame

TIERS = [("🟢 Tier 1 — Düşük risk", 0.70, 1.01), ("🟡 Tier 2 — Orta risk", 0.50, 0.70),
         ("🔴 Tier 3 — Yüksek risk", 0.25, 0.50), ("🟣 Tier 4 — Çok yüksek risk", 0.18, 0.38)]
MAX_PER_TIER = {0: 8, 1: 12, 2: 12, 3: 10}
# En düşük oran (İddaa; yoksa adil oran). Tier 1 / 2: 1.06 gibi oranlar oynanmaya değmez. Tier 4: Tier 3'ün (~1.8-2.6)
# üstündeki oran aralığı (kullanıcı isteği, 2026-10-05: Tier 1-2 düşük oranlı, daha yüksek riskli seçenek de olsun)
MIN_ODDS = {0: 1.20, 1: 1.40, 3: 2.30}
MAX_SHORT_SCORES = 6          # Tier 5: en olası kesin skoru en yüksek olan maçlar
SKOR_TIERS = ("⚡ Tier 5 — Aşırı risk (kısa kesin skor)", "🔥 Tier 6 — En yüksek risk (tüm maçlarda kesin skor)")
ODD_COL = {"MS 1": "odd_1", "MS X": "odd_x", "MS 2": "odd_2", "ÇŞ 1-X": "odd_1x", "ÇŞ 1-2": "odd_12",
           "ÇŞ X-2": "odd_x2", "2.5 Alt": "odd_alt", "2.5 Üst": "odd_ust", "KG Var": "odd_kg_var",
           "KG Yok": "odd_kg_yok", "MS 1 & 2.5 Üst": "odd_ms1_ust", "MS 2 & 2.5 Üst": "odd_ms2_ust",
           "MS 1 & KG Var": "odd_ms1_kg", "MS X & KG Var": "odd_msx_kg", "MS 2 & KG Var": "odd_ms2_kg"}
PRICE_COLS = ODD_COLS + ["mbs", SCORE_COL, KOR_LINES_COL]   # İddaa'dan gelen fiyat sütunları (eksikse boş)
KORNER_PER_TIER = {0: 4, 1: 4, 2: 4, 3: 3}   # Tier 1-4'te korner tahmini sayısı (gol tahminlerine ek)
SISTEM_TIER = "🎰 Sistem kupon"
SISTEM_N = 8                  # sistem kuponda maç sayısı
SISTEM_MIN_ODDS = 1.30
# Çılgın kupon (kullanıcı isteği, 2026-10-24): İY/MS "dönüşleri" (ilk yarıyı önde kapatan berabere kalır / kaybeder;
# oran ~12-30) + günün 💎'leri. Piyango gibi: tutma ihtimali çok düşük, fiyat yine de en iyi olanlardan seçilir.
CILGIN_TIER = "🎲 Çılgın kupon"
CILGIN_DONUS = ("İY/MS 1/X", "İY/MS 2/X", "İY/MS 1/2", "İY/MS 2/1")
CILGIN_N = 4                  # dönüş ayağı (farklı maçlar)
CILGIN_DEGERLI = 2            # yanına en fazla bu kadar 💎


def _iyms_col(bet: str) -> str:
    a, b = bet.split()[1].lower().split("/")
    return f"odd_iyms_{a}{b}"


def _system(picks: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Bugün ve yarının Tier 1-2 (ve korner) tahminlerinden SISTEM_N maçlık sistem kupon: maç başına en olası
    tahmin, oranı SISTEM_MIN_ODDS+; en az 5 maç yoksa boş."""
    p = picks[np.isfinite(picks["odd"].astype(float)) & (picks["odd"] >= SISTEM_MIN_ODDS)].copy()
    when = pd.to_datetime(p["baslama"].where(p["baslama"].astype(str).str.len() >= 16), errors="coerce")
    p = p[(when > now) & (when.dt.normalize() <= now.normalize() + pd.Timedelta(days=1))]
    sistem = p.sort_values("p", ascending=False).drop_duplicates("maç").head(SISTEM_N)
    return sistem if len(sistem) >= 5 else sistem.iloc[:0]


def _crazy(c: pd.DataFrame, pos: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Çılgın kupon: bugün / yarının İY/MS dönüşlerinden olasılık x İddaa oranı en yüksek olanlar (maç başına bir,
    CILGIN_N maç; dönüş olasılıkları kupon.IYMS_DUZELTME ile düzeltilmiş) + başka maçlardan en fazla CILGIN_DEGERLI 💎.
    CILGIN_N dönüş yoksa boş."""
    def soon(d):
        when = pd.to_datetime(d["baslama"].where(d["baslama"].astype(str).str.len() >= 16), errors="coerce")
        return d[(when > now) & (when.dt.normalize() <= now.normalize() + pd.Timedelta(days=1)) &
                 (d["odd"].astype(float) > 1)]
    legs = soon(c[c["tahmin"].isin(CILGIN_DONUS)]).sort_values("değer", ascending=False)
    legs = legs.drop_duplicates("maç").head(CILGIN_N)
    if len(legs) < CILGIN_N:
        return legs.iloc[:0]
    value = soon(pos[~pos["maç"].isin(legs["maç"])]).drop_duplicates("maç").head(CILGIN_DEGERLI)
    return pd.concat([legs, value], ignore_index=True)


def _corner_cands(r: dict, info: dict, cm) -> list[dict]:
    """Maçın İddaa'daki tüm korner barajları (alt / üst) ve en çok korner için aday tahminler. Olasılık: kulüp
    maçlarında korner modeli (cm), yoksa (milli maçlar) İddaa'nın marjı arındırılmış oranı."""
    lines = corner_lines(r.get(KOR_LINES_COL), r)
    trio = [r.get(f"odd_kor_{s}", np.nan) for s in ("1", "x", "2")]
    cmat = korner.matrix(*cm.expected_goals(r["home"], r["away"], r["div"])) if cm is not None else None
    src = "korner" if cmat is not None else "oran"
    out = []
    for line, (a, u) in lines.items():
        if cmat is not None:
            pu = korner.market_probs(cmat, line)[f"Korner {line:g} Üst"]
            pa = 1 - pu
        else:
            pa, pu = devig([a, u], config.DEVIG_METHOD)
        for bet, p, odd in ((f"Korner {line:g} Alt", pa, a), (f"Korner {line:g} Üst", pu, u)):
            if odd > 1.0 and np.isfinite(p):
                out.append({**info, "kaynak": src, "tahmin": bet, "p": float(p), "odd": odd})
    ps = (korner.market_probs(cmat, None) if cmat is not None else
          dict(zip([f"En çok korner {s}" for s in "1X2"], devig(trio, config.DEVIG_METHOD))))
    for s, odd in zip("1X2", trio):
        p = ps.get(f"En çok korner {s}", np.nan)
        if odd > 1.0 and np.isfinite(p):
            out.append({**info, "kaynak": src, "tahmin": f"En çok korner {s}", "p": float(p), "odd": odd})
    return out


def _corner_narrower(a: str, b: str) -> bool:
    """Korner merdiveni: "Korner 8.5 Üst" "Korner 7.5 Üst"ün devamı (tutarsa o da tutar); alt için tersi."""
    pa, pb = a.split(), b.split()
    if not (a.startswith("Korner ") and b.startswith("Korner ") and pa[2] == pb[2]):
        return False
    return float(pa[1]) > float(pb[1]) if pa[2] == "Üst" else float(pa[1]) < float(pb[1])


def _select_tiers(c: pd.DataFrame, max_per_tier: dict, narrower) -> tuple[list, dict]:
    """Tier 1-4 seçimi (bir tahmin ailesi için: gol ya da korner). Maç başına seviye başına tek tahmin; bir maç üst
    seviyedeyse alt seviyede sadece onun daha dar devamı (narrower(yeni, önceki)) gelir (merdiven, ↳), seviyenin en
    fazla yarısı. Döner: seviye başına seçilenler ve maç -> son basamaktaki tahmin."""
    frames, rung, order = [], {}, []  # order: maçların ilk görülme sırası
    for k, (_, lo, hi) in enumerate(TIERS):
        band = c[(c["p"] >= lo) & (c["p"] < hi)].copy()
        if k in MIN_ODDS:  # çok düşük oran (ör. 1.06'lık çifte şans) önerilmez; oran yoksa adil oranla
            price_or_fair = band["odd"].astype(float).fillna(1 / band["p"])
            band = band[price_or_fair >= MIN_ODDS[k]]
        if k == 0:  # düşük riskte çifte şans ancak MS zaten >= %70 değilse
            band = band[~band["tahmin"].str.startswith("ÇŞ") |
                        ~band["maç"].isin(band.loc[band["tahmin"].str.startswith("MS "), "maç"])]
        band["sıra"] = band["gerçek_değer"].astype(int)
        # önce üst seviyedeki maçların aynı maçta bir basamak riskli devamı (en olası olanı)
        ladder = []
        for maç in order:
            nxt = band[(band["maç"] == maç) & band["tahmin"].map(lambda a, b=rung[maç]: narrower(a, b))]
            if len(nxt):
                ladder.append(nxt.sort_values(["p", "değer"], ascending=False).index[0])
        lad = band.loc[ladder].head(max_per_tier[k] // 2).assign(merdiven=True)  # en fazla yarısı merdiven
        # gerisi yeni maçlar (merdivendeki maça ilgisiz ikinci tahmin yok), maç başına tek tahmin: önce (piyasaya
        # göre) değeri pozitif olanlar, sonra değer, sonra olasılık
        rest = band[~band["maç"].isin(order)]
        rest = rest.sort_values(["sıra", "değer", "p"], ascending=False).groupby("maç").head(1)
        rest = rest.sort_values(["sıra", "p"], ascending=False).head(max_per_tier[k] - len(lad)).assign(merdiven=False)
        t = pd.concat([lad, rest]).sort_values(["sıra", "p"], ascending=False)
        for maç, bet in zip(t["maç"], t["tahmin"]):
            if maç not in rung:
                order.append(maç)
            rung[maç] = bet
        frames.append(t)
    return frames, rung
CAND_COLS = ["maç", "lig", "tarih", "saat", "baslama", "gün", "mbs", "kaynak", "hg", "ag", "div", "home", "away",
             "tahmin", "p", "odd"]


def bet_masks(n: int) -> dict:
    """Her tahminin tuttuğu skorlar: n x n tabloda (ev golü, deplasman golü) maske."""
    g = np.arange(n)
    h, a = np.meshgrid(g, g, indexing="ij")
    home, draw, away = h > a, h == a, h < a
    over, btts = (h + a) >= 3, (h >= 1) & (a >= 1)
    return {"MS 1": home, "MS X": draw, "MS 2": away,
            "ÇŞ 1-X": home | draw, "ÇŞ X-2": draw | away, "ÇŞ 1-2": home | away,
            "2.5 Üst": over, "2.5 Alt": ~over, "KG Var": btts, "KG Yok": ~btts,
            "MS 1 & KG Var": home & btts, "MS 2 & KG Var": away & btts, "MS X & KG Var": draw & btts,
            "MS 1 & 2.5 Üst": home & over, "MS 2 & 2.5 Üst": away & over}


def market_probs(m: np.ndarray) -> dict:
    """Skor tablosundan (m[ev golü, deplasman golü]) market olasılıkları."""
    return {bet: float(m[mask].sum()) for bet, mask in bet_masks(m.shape[0]).items()}


IYMS_BETS = [f"İY/MS {a}/{b}" for a in "1X2" for b in "1X2"]


def _ft_mask(bet: str, n: int) -> np.ndarray:
    """Tahminin maç sonu skor tablosundaki karşılığı (İY/MS a/b için maç sonu b)."""
    return bet_masks(n)[f"MS {bet[-1]}" if bet.startswith("İY/MS") else bet]


def _narrower() -> dict:
    """Tahmin -> onun daha dar (daha riskli) devamları: tuttuklarında o tahmin de tutan tahminler
    (ör. MS 1 -> MS 1 & KG Var, MS 1 -> İY/MS 1/1). Merdiven bunlarla kurulur. İY/MS a/b, maç sonu b'yi içeren her
    tahminin (MS b, b'yi içeren çifte şans) devamıdır; kendisinin devamı yok (sadece kesin skor)."""
    masks = bet_masks(config.MAX_GOALS + 1)
    out = {b: [a for a in masks if a != b and (~masks[a] | masks[b]).all() and (masks[a] != masks[b]).any()]
           for b in masks}
    for iy in IYMS_BETS:
        ft = masks[f"MS {iy[-1]}"]
        for b in masks:
            if (~ft | masks[b]).all():
                out[b].append(iy)
        out[iy] = []
    return out


NARROWER = _narrower()


def is_ladder(x) -> bool:
    """Satır merdiven devamı mı (↳)? CSV'den okununca sütun boş / numpy bool olabilir."""
    v = x.get("merdiven")
    return isinstance(v, (bool, np.bool_)) and bool(v)


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


def _upcoming(fixtures_path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Henüz başlamamış maçlar: İddaa'da listelenenler (son oranlarıyla) + fikstürdekiler (keskin oranlarla).
    Döner: maçlar ve İddaa oranlarının son kayıt zamanı (UTC)."""
    from src import iddaa
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    rows = []
    odds = iddaa.load_odds()
    last_snap = odds["snapshot_utc"].max() if not odds.empty else None
    if not odds.empty:
        last = odds[odds["kickoff_utc"] > now].sort_values("snapshot_utc").groupby("event_id").tail(1)
        last = last.dropna(subset=["home", "away"])
        for c in PRICE_COLS:  # eski kayıtlarda yeni sütunlar yok
            last[c] = last[c] if c in last else np.nan
        rows.append(last.assign(date=last["kickoff_utc"].dt.normalize(), kaynak="İddaa"))
    if fixtures_path and fixtures_path.exists():
        fx = add_sharp(db.normalize(db.read_raw_csv(fixtures_path), require_result=False))
        fx = fx[fx["div"].isin(config.LEAGUES) & (fx["date"] >= now.normalize())]
        if not fx.empty:
            rows.append(_sharp_1x2(fx).assign(kaynak="fikstür"))
    if not rows:
        return pd.DataFrame(), last_snap
    up = pd.concat(rows, ignore_index=True, sort=False)
    # aynı maç iki kaynakta: keskin oranları fikstürden, İddaa oranlarını İddaa'dan al ("first" boşları atlar)
    keep = [c for c in up if c.startswith(("odd_", "keskin_")) or
            c in ("mbs", "ou_line", "kor_line", "kickoff_utc", "competition", "home_iddaa", "away_iddaa")]
    merged = up.groupby(["div", "home", "away"], as_index=False).agg(
        {**{c: "first" for c in keep}, "date": "min", "kaynak": lambda s: " + ".join(sorted(set(s)))})
    merged["season"] = np.where(merged["date"].dt.month >= 7, merged["date"].dt.year, merged["date"].dt.year - 1)
    return merged, last_snap


def _history(matches: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    """Prova: [start, start+days) arasındaki oynanmış maçlar; fiyat Bet365, keskin olasılık açılış oranından.
    Milli maçlar da (oran yok: sadece model)."""
    end = start + pd.Timedelta(days=days)
    up = matches[(matches["date"] >= start) & (matches["date"] < end)].copy()
    up = _sharp_1x2(up)
    up["odd_1"], up["odd_x"], up["odd_2"] = up["b365_h"], up["b365_d"], up["b365_a"]
    up["odd_ust"], up["odd_alt"] = up["b365_o25"], up["b365_u25"]
    up["ou_line"] = np.where(up["b365_o25"].notna(), 2.5, np.nan)
    for c in PRICE_COLS:  # B365'te sadece 1X2 ve 2.5 alt/üst var
        if c not in up:
            up[c] = np.nan
    nat = milli.load_results()
    if not nat.empty:
        nat = nat[(nat["date"] >= start) & (nat["date"] < end)].rename(columns={"tournament": "competition"})
        up = pd.concat([up, nat], ignore_index=True, sort=False) if not nat.empty else up
    return up.assign(kaynak="prova")


def _price_fair(r: dict) -> tuple[tuple, tuple]:
    """Fiyatın (İddaa / provada B365) marjı arındırılmış 1X2 ve 2.5 alt/üst olasılıkları (yoksa NaN)."""
    fair = devig([r.get("odd_1", np.nan), r.get("odd_x", np.nan), r.get("odd_2", np.nan)], config.DEVIG_METHOD)
    ou = (devig([r.get("odd_ust", np.nan), r.get("odd_alt", np.nan)], config.DEVIG_METHOD)
          if r.get("ou_line") == 2.5 else np.full(2, np.nan))
    return tuple(fair), tuple(ou)


def market_matrix(r: dict, n: int = config.MAX_GOALS + 1) -> np.ndarray | None:
    """Avrupa kupası maçı (model yok): fiyatın marjsız 1X2 ve 2.5 alt/üst olasılıklarına en yakın bağımsız Poisson
    skor tablosu (λ ızgarası); sonra diğer maçlardaki gibi calibrate ile tam ayarlanır. 1X2 oranı yoksa None."""
    from scipy.stats import poisson
    target, target_ou = _price_fair(r)
    if not all(np.isfinite(target)):
        return None
    lam = np.arange(0.15, 4.01, 0.05)
    pmf = poisson.pmf(np.arange(n)[None, :], lam[:, None])           # (λ, gol)
    m = pmf[:, None, :, None] * pmf[None, :, None, :]                # (λ ev, λ dep, ev golü, dep golü)
    m /= m.sum(axis=(-2, -1), keepdims=True)                          # tablo n golde kesilir: toplam 1 olsun
    g = np.arange(n)
    h, a = np.meshgrid(g, g, indexing="ij")
    loss = sum((m[..., mask].sum(-1) - t) ** 2 for mask, t in zip((h > a, h == a, h < a), target))
    if np.isfinite(target_ou[0]):
        loss = loss + (m[..., (h + a) >= 3].sum(-1) - target_ou[0]) ** 2
    i, j = np.unravel_index(np.argmin(loss), loss.shape)
    return m[i, j]


def _match_label(r: dict) -> str:
    """Maç adı: İddaa'daki adlarla (kuponda aranacak ad), yoksa veri setindeki adlarla."""
    h, a = r.get("home_iddaa"), r.get("away_iddaa")
    if isinstance(h, str) and isinstance(a, str):
        return f"{h} – {a}"
    return f"{r['home']} – {r['away']}"


def _league_label(r: dict) -> str:
    if r["div"] not in (config.MILLI, config.AVRUPA):
        return r["div"]
    comp = r.get("competition")
    if isinstance(comp, str) and comp.strip():
        return comp.split(",")[0].strip()
    return "Milli" if r["div"] == config.MILLI else "Avrupa kupası"


def run(fixtures_path=None, xi=config.XI, ridge=config.RIDGE, ridge_total=config.RIDGE_TOTAL,
        joint=config.JOINT, motivation=config.MOTIVATION, as_of: str | None = None,
        days: int = 4, saat: str | None = None, gun: str | None = None) -> tuple[str, pd.DataFrame]:
    """Döner: markdown rapor ve seçilen tahminler tablosu (seviye, maç, tahmin, olasılık, oran, değer ...).
    saat (ör. "21:45") verilirse sadece o saatte (TR) başlayan maçlar (gun: YYYY-AA-GG, varsayılan bugün)."""
    from src.predict import _fit_models
    matches = db.load_matches()
    if as_of:
        start = pd.Timestamp(as_of)
        up, train, price = _history(matches, start, days), matches[matches["date"] < start], "B365"
        title = f"# Tavsiye provası — {start:%d.%m.%Y} ve sonraki {days} gün (sonuçlarla)"
    else:
        (up, last_snap), train, price = _upcoming(fixtures_path), matches, "İddaa"
        title = f"# Tavsiyeler — {date.today():%d.%m.%Y}"
        if saat:  # sadece seçilen saatteki maçlar (İddaa'da başlama saati olanlar)
            day = pd.Timestamp(gun).date() if gun else (datetime.now(timezone.utc) + pd.Timedelta(hours=3)).date()
            if up.empty or "kickoff_utc" not in up:
                up = pd.DataFrame()
            else:
                ko = pd.to_datetime(up["kickoff_utc"]) + pd.Timedelta(hours=3)
                up = up[(ko.dt.strftime("%H:%M") == saat) & (ko.dt.date == day)]
            title = f"# Tavsiyeler — {day:%d.%m.%Y} {saat} maçları"
        if last_snap is not None:
            title += f"\n\nİddaa oranlarının son kaydı: {last_snap + pd.Timedelta(hours=3):%d.%m %H:%M} (TR saati)"
    md = [title, "",
          f"Her satır: tahmin — kazanma olasılığı, adil oran, {price} oranı ve **değer** ({price} oranı adil oranın "
          "üstündeyse +). Risk seviyesi tahminin tutma olasılığına göredir; **değer ayrı bir şeydir**: düşük "
          "riskli bir tahmin değeri negatifse uzun vadede kaybettirir. Olasılıklar: keskin piyasa (Pinnacle / "
          f"Betfair) varsa ona ayarlanmış model; yoksa {price} oranının marjı arındırılmış haline ayarlı _({price} "
          "oranına ayarlı)_ — bu durumda değer ölçülemez, fiyat kendisiyle kıyaslanır; oran da yoksa _sadece model_ "
          "(en az isabetli). Kombinasyon ve kesin skor olasılıkları modelin skor tablosundan gelir. İddaa'da "
          "listelenen maçlarda oranı olmayan (kapalı) bahisler listelenmez.", "",
          "**↳** = aynı maçta bir üst seviyedeki tahminin daha riskli (daha yüksek oranlı) devamı: bu tutarsa "
          "üstteki de tutar (ör. MS 1 → MS 1 & 2.5 Üst → MS 1 & KG Var). Tier 2 ve 3'ün en fazla yarısı böyle, "
          "gerisi diğer maçlar.", ""]
    if as_of:
        md += ["**Prova:** model sadece bu tarihten önceki maçları gördü; keskin olasılık o anki açılış oranından. "
               "Geçmiş İddaa oranı olmadığı için fiyat Bet365. Tek hafta sonu çok küçük bir örnek — sonuçlar "
               "şanstan çok etkilenir; amaç sistemin nasıl çalıştığını göstermek.", ""]
    if up.empty:
        return "\n".join(md + ["Maç bulunamadı."]), pd.DataFrame()
    is_nat, is_eu = up["div"] == config.MILLI, up["div"] == config.AVRUPA
    club, nat, eu = up[~is_nat & ~is_eu], up[is_nat], up[is_eu]
    if motivation and not club.empty:
        train = add_motivation(train)
        club = add_motivation(train, club)
    up = pd.concat([club, nat, eu], ignore_index=True, sort=False)
    refs = {d: (start if as_of else t) for d, t in club.groupby("div")["date"].min().items()}
    models = _fit_models(train, refs, {"xi": xi, "ridge": ridge, "ridge_total": ridge_total}, joint) if refs else {}
    # korner: sadece İddaa'da korner oranı açık kulüp maçlarının ligleri için (geçmiş korner verisi kulüplerde var)
    has_kor = club[[c for c in ("kor_line", "odd_kor_1") if c in club]].notna().any(axis=1) if not as_of else []
    kor_divs = set(club.loc[has_kor, "div"]) if len(has_kor) else set()
    cmodels = korner.fit_models(train, {d: refs[d] for d in kor_divs if d in refs}) if kor_divs else {}
    if not nat.empty:
        mdl = milli.fit_current(start if as_of else nat["date"].min())
        if mdl is not None:
            models[config.MILLI] = mdl
        md += ["**Milli maçlar** ayrı bir modelle (milli takım sonuçları, tarafsız saha ve hazırlık maçı ağırlığıyla) "
               "tahmin edilir. Milli maçlar için keskin piyasa oranı olmadığından 💎 çıkmaz. İddaa sahayı "
               "bildirmediği için ev sahibi kendi sahasında sayılır.", ""]
    if not eu.empty:
        md += ["**Avrupa kupaları** (Şampiyonlar / Avrupa / Konferans Ligi): takımlar farklı liglerden olduğu için "
               "modelimiz yok; olasılıklar İddaa oranlarının marjı arındırılmış halinden. 💎 çıkmaz.", ""]
    from src import kupon
    cands, scores, corners, mats = [], [], [], {}
    for r in up.to_dict("records"):
        if r["div"] == config.AVRUPA:
            m = market_matrix(r)  # sonra fiyatın marjsız oranına ayarlanır (kaynak: oran)
            if m is None:
                continue
        else:
            mdl = models.get(r["div"])
            if mdl is None:
                continue
            neutral = r.get("neutral") is True or r.get("neutral") is np.True_
            m = mdl.score_matrix(r["home"], r["away"], r["div"], r.get("mot", 0), neutral)
        target = tuple(r.get(f"keskin_{k}", np.nan) for k in ("1", "x", "2"))
        target_ou = (r.get("keskin_ust", np.nan), r.get("keskin_alt", np.nan))
        src = "piyasa" if all(np.isfinite(target)) else "model"
        if src == "piyasa":
            m = calibrate(m, target, target_ou)
        else:  # keskin oran yok: fiyatın marjsız oranı (modelden isabetli)
            target, target_ou = _price_fair(r)
            src = "oran" if all(np.isfinite(target)) else "model"
            if all(np.isfinite(target)) or all(np.isfinite(target_ou)):
                m = calibrate(m, target, target_ou)  # 1X2'de kapalı sonuç varsa sadece alt/üst
        # İddaa'da listelenen maçta oranı olmayan market (kapalı / açılmamış) oynanamaz: önerilmez
        listed = not as_of and "İddaa" in str(r.get("kaynak", ""))
        ko = r.get("kickoff_utc")  # İddaa'daki maçlarda başlama saati var (UTC); TR = UTC+3
        tr = pd.Timestamp(ko) + pd.Timedelta(hours=3) if pd.notna(ko) else None
        info = {"maç": _match_label(r), "lig": _league_label(r),
                "tarih": (tr if tr is not None else pd.Timestamp(r["date"])).strftime("%d.%m"),
                "saat": tr.strftime("%H:%M") if tr is not None else "",
                "baslama": tr.strftime("%Y-%m-%d %H:%M") if tr is not None else "",  # TR saati
                "gün": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                "mbs": r.get("mbs"), "kaynak": src, "hg": r.get("fthg", np.nan), "ag": r.get("ftag", np.nan),
                "div": r["div"], "home": r["home"], "away": r["away"]}  # sonuç eşleştirmesi için (bkz. sonuclar.py)
        for bet, p in market_probs(m).items():
            col = ODD_COL.get(bet)
            odd = r.get(col, np.nan) if col else np.nan
            if not odd > 1.0:  # eski kayıtlarda 1.00 = oynanamaz
                odd = np.nan
            if bet.startswith("2.5") and r.get("ou_line") != 2.5:
                odd = np.nan  # ana baraj 2.5 değilse bu oran başka baraj içindir
            if listed and col and not np.isfinite(odd):
                continue
            cands.append({**info, "tahmin": bet, "p": p, "odd": odd})
        if not as_of:  # İY/MS: olasılığına göre Tier 1-4'e girer (sadece İddaa'da oranı varsa)
            for bet, p in kupon.iyms_probs(m).items():
                odd = r.get(_iyms_col(bet), np.nan)
                if odd > 1.0:
                    cands.append({**info, "tahmin": bet, "p": p, "odd": odd})
        # kesin skor tierlardan sonra seçilir: maçın merdiveniyle uyumlu en olası skor
        mats[info["maç"]] = (m, info, score_odds(r.get(SCORE_COL)), listed)
        if listed:  # korner: İddaa'da oranı olan tüm barajlar ve en çok korner (Tier 1-4'e girer)
            corners += _corner_cands(r, info, cmodels.get(r["div"]))
    c = pd.DataFrame(cands, columns=CAND_COLS)
    c["değer"] = c["odd"] * c["p"] - 1
    # 💎 sadece olasılık piyasaya ayarlıysa: sadece modelin gördüğü "değer" backtest'te kapanışı yenmedi
    c["gerçek_değer"] = (c["değer"] > 0) & (c["kaynak"] == "piyasa")
    judged = as_of is not None

    def line(x) -> str:
        when = f"{x['tarih']} {x['saat']}" if x.get("saat") else x["tarih"]
        s = (f"• {'↳ ' if is_ladder(x) else ''}{x['maç']} ({x['lig']}, {when}): **{x['tahmin']}** — "
             f"olasılık %{100 * x['p']:.0f}, adil oran {1 / x['p']:.2f}")
        if np.isfinite(x["odd"]):
            s += f", {price} {x['odd']:.2f} (değer {100 * x['değer']:+.0f}%{' 💎' if x.get('gerçek_değer') else ''})"
        else:
            s += f", {price} oranı yok"
        if pd.notna(x.get("mbs")) and x.get("mbs") > 1:
            s += f", MBS {int(x['mbs'])}"
        s += {"piyasa": "", "oran": f" _({price} oranına ayarlı)_", "korner": " _(korner modeli)_"}.get(
            x["kaynak"], " _(sadece model)_")
        if judged and not str(x["tahmin"]).startswith("İY/MS"):  # provada İY/MS skoru (ilk yarı) yok
            s += f" → {int(x['hg'])}-{int(x['ag'])} {'✅' if bet_won(x['tahmin'], int(x['hg']), int(x['ag'])) else '❌'}"
        return s

    # Tier 1-4: gol tahminleri (İY/MS dahil) ve korner ayrı merdivenlerle seçilir, aynı seviyede yan yana durur
    goal_t, rung = _select_tiers(c, MAX_PER_TIER, lambda a, b: a in NARROWER.get(b, ()))
    k = pd.DataFrame(corners, columns=CAND_COLS)
    k["değer"], k["gerçek_değer"] = k["odd"] * k["p"] - 1, False  # keskin korner oranı yok: değer ölçülemez
    kor_t, _ = _select_tiers(k, KORNER_PER_TIER, _corner_narrower)
    picked = []
    for (tname, _, _), gt, kt in zip(TIERS, goal_t, kor_t):
        t = pd.concat([x for x in (gt, kt) if len(x)]) if len(gt) or len(kt) else gt
        picked.append((tname, t))
        md += [f"## {tname}", ""] + ([line(x) for x in t.to_dict("records")] or ["Bu seviyede tahmin yok."]) + [""]
    # kesin skor: maçın merdiveninin son basamağıyla uyumlu en olası skor (ör. MS 1 & KG Var -> 2-1); tier'da
    # olmayan maçta genel en olası skor. İddaa'da skor bahsi açık değilse önerilmez.
    for maç, (m, info, sc, listed) in mats.items():
        mm = np.where(_ft_mask(rung[maç], m.shape[0]), m, 0.0) if maç in rung else m
        i, j = np.unravel_index(np.argmax(mm), mm.shape)
        s_odd = sc.get(f"{i}-{j}", np.nan) if sc is not None else np.nan
        if not (listed and sc is not None and not np.isfinite(s_odd)):
            scores.append({**info, "tahmin": f"{i}-{j}", "p": float(m[i, j]), "odd": s_odd})
    s = pd.DataFrame(scores, columns=CAND_COLS).sort_values("p", ascending=False)
    s["değer"], s["gerçek_değer"] = s["odd"] * s["p"] - 1, False  # kesin skor olasılığı piyasaya ayarlı değil
    picked += [(SKOR_TIERS[0], s.head(MAX_SHORT_SCORES)), (SKOR_TIERS[1], s)]
    for tname, t in picked[len(TIERS):]:
        md += [f"## {tname}", ""] + [line(x) for x in t.to_dict("records")] + [""]
    if not as_of:
        from src import kupon
        legs = pd.concat([t for name, t in picked if name.startswith(("🟢", "🟡"))], ignore_index=True)
        legs = legs.reindex(columns=list(dict.fromkeys([*CAND_COLS, *legs.columns])))
        sistem = _system(legs, pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + pd.Timedelta(hours=3))
        picked.append((SISTEM_TIER, sistem))
        md += [f"## {SISTEM_TIER} ({len(sistem)} maç)", ""]
        if sistem.empty:
            md += ["Bugün ve yarın için en az 5 uygun maç yok.", ""]
        else:
            md += [line(x) for x in sistem.to_dict("records")] + ["", "| sistem | kolon | en az k tutma | kâr olasılığı | "
                                                                    "beklenen dönüş (birim başına) |", "|---|---|---|---|---|"]
            for s in kupon.system_table(list(sistem["p"]), list(sistem["odd"])):
                md.append(f"| {len(sistem)}'de {s['k']} | {s['kolon']} | %{100 * s['en_az_k']:.0f} | "
                          f"%{100 * s['kar_olasiligi']:.0f} | {s['beklenen_donus']:.2f} |")
            md += [""]
    pos = c[c["gerçek_değer"] & (c["p"] >= 0.25)].sort_values("değer", ascending=False).head(15)
    picked.append(("💎 Değeri pozitif olanlar", pos))
    md += ["## 💎 Değeri pozitif olanlar (risk seviyesinden bağımsız)", "",
           "Uzun vadede kâr beklentisi olan tek grup bu: oran, keskin piyasanın (Pinnacle / Betfair) adil fiyatının "
           "üstünde. Sadece modelin gördüğü değerler buraya alınmaz (backtest'te kapanışı yenmedi).", ""]
    md += [line(x) for x in pos.to_dict("records")] or [f"{price} oranı adil fiyatın üstünde olan tahmin yok."]
    if not as_of:
        crazy = _crazy(c, pos, pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + pd.Timedelta(hours=3))
        picked.append((CILGIN_TIER, crazy))
        md += ["", f"## {CILGIN_TIER}", ""]
        if crazy.empty:
            md += [f"Bugün ve yarın için en az {CILGIN_N} maçta İY/MS dönüş oranı yok.", ""]
        else:
            md += [line(x) for x in crazy.to_dict("records")]
            tl = lambda x, d=2: f"{x:,.{d}f}".replace(",", "#").replace(".", ",").replace("#", ".")  # noqa: E731
            md += ["", f"Toplam oran **{tl(np.prod(crazy['odd'].astype(float)))}**, tutma olasılığı yaklaşık "
                       f"1/{tl(1 / np.prod(crazy['p'].astype(float)), 0)}. Piyango gibidir; ayaklar İddaa oranına göre "
                       "en değerli dönüşlerden (dönüş olasılıkları geçmiş maçlara göre düzeltilmiş) seçildi.", ""]
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
    frames = [t.assign(seviye=name) for name, t in picked if not t.empty]
    return "\n".join(md), (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
