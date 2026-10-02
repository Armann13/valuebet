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
  Tier 4 çok yüksek (en olası kesin skor, kısa liste), Tier 5 aşırı (her maçın en olası kesin skoru).
  Tier 1 ve 2'de oranı MIN_ODDS'un altındaki tahminler (ör. 1.06'lık çifte şans) listelenmez.
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
from src.iddaa import ODD_COLS, SCORE_COL, score_odds
from src.odds import add_sharp, devig, devig_frame

TIERS = [("🟢 Tier 1 — Düşük risk", 0.70, 1.01), ("🟡 Tier 2 — Orta risk", 0.50, 0.70),
         ("🔴 Tier 3 — Yüksek risk", 0.25, 0.50)]
MAX_PER_TIER = {0: 8, 1: 12, 2: 12}
MIN_ODDS = {0: 1.20, 1: 1.40}   # Tier 1 / 2'de en düşük oran (İddaa; yoksa adil oran): 1.06 gibi oranlar oynanmaya değmez
MAX_SHORT_SCORES = 6          # Tier 4: en olası kesin skoru en yüksek olan maçlar
ODD_COL = {"MS 1": "odd_1", "MS X": "odd_x", "MS 2": "odd_2", "ÇŞ 1-X": "odd_1x", "ÇŞ 1-2": "odd_12",
           "ÇŞ X-2": "odd_x2", "2.5 Alt": "odd_alt", "2.5 Üst": "odd_ust", "KG Var": "odd_kg_var",
           "KG Yok": "odd_kg_yok", "MS 1 & 2.5 Üst": "odd_ms1_ust", "MS 2 & 2.5 Üst": "odd_ms2_ust",
           "MS 1 & KG Var": "odd_ms1_kg", "MS X & KG Var": "odd_msx_kg", "MS 2 & KG Var": "odd_ms2_kg"}
PRICE_COLS = ODD_COLS + ["mbs", SCORE_COL]   # İddaa'dan gelen fiyat sütunları (eksikse boş)
CORNER_TIER = "🚩 Korner — kulüp maçları"
MAX_CORNERS = 10
IYMS_TIER = "🔀 Tier 8 — İY/MS (ilk yarı / maç sonu)"
MAX_IYMS = 8
KUPON_TIER = "🎫 Kupon — orta risk (kombine)"
SISTEM_TIER = "🎰 Sistem kupon"
SISTEM_N = 8                  # sistem kuponda maç sayısı
SISTEM_MIN_ODDS = 1.30


def _iyms_col(bet: str) -> str:
    a, b = bet.split()[1].lower().split("/")
    return f"odd_iyms_{a}{b}"


def _coupons(picks: pd.DataFrame, now: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bugün ve yarının Tier 1-2 (ve korner) tahminlerinden: orta riskli kombine (3-4 maç, toplam oran 3-6, en
    yüksek tutma olasılığı) ve SISTEM_N maçlık sistem kupon (en olası, oranı SISTEM_MIN_ODDS+ tahminler)."""
    from src import kupon
    p = picks[np.isfinite(picks["odd"].astype(float)) & (picks["odd"] >= MIN_ODDS[0])].copy()
    when = pd.to_datetime(p["baslama"].where(p["baslama"].astype(str).str.len() >= 16), errors="coerce")
    p = p[(when > now) & (when.dt.normalize() <= now.normalize() + pd.Timedelta(days=1))]
    p = p.sort_values("p", ascending=False).drop_duplicates("maç")  # maç başına en olası tahmin
    combo = kupon.best_combo(p.head(12).to_dict("records"))
    sistem = p[p["odd"] >= SISTEM_MIN_ODDS].head(SISTEM_N)
    return pd.DataFrame(combo, columns=p.columns), (sistem if len(sistem) >= 5 else sistem.iloc[:0])


def _corner_col(bet: str) -> str:
    """Korner bahsi -> İddaa oran sütunu."""
    if bet.startswith("En çok korner"):
        return {"1": "odd_kor_1", "X": "odd_kor_x", "2": "odd_kor_2"}[bet.split()[-1]]
    return "odd_kor_ust" if bet.endswith("Üst") else "odd_kor_alt"
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


def _narrower() -> dict:
    """Tahmin -> onun daha dar (daha riskli) devamları: tuttuklarında o tahmin de tutan tahminler
    (ör. MS 1 -> MS 1 & 2.5 Üst -> MS 1 & KG Var). Merdiven bunlarla kurulur."""
    masks = bet_masks(config.MAX_GOALS + 1)
    return {b: [a for a in masks if a != b and (~masks[a] | masks[b]).all() and (masks[a] != masks[b]).any()]
            for b in masks}


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


def _match_label(r: dict) -> str:
    """Maç adı: İddaa'daki adlarla (kuponda aranacak ad), yoksa veri setindeki adlarla."""
    h, a = r.get("home_iddaa"), r.get("away_iddaa")
    if isinstance(h, str) and isinstance(a, str):
        return f"{h} – {a}"
    return f"{r['home']} – {r['away']}"


def _league_label(r: dict) -> str:
    if r["div"] != config.MILLI:
        return r["div"]
    comp = r.get("competition")
    return comp.split(",")[0].strip() if isinstance(comp, str) and comp.strip() else "Milli"


def run(fixtures_path=None, xi=config.XI, ridge=config.RIDGE, ridge_total=config.RIDGE_TOTAL,
        joint=config.JOINT, motivation=config.MOTIVATION, as_of: str | None = None,
        days: int = 4) -> tuple[str, pd.DataFrame]:
    """Döner: markdown rapor ve seçilen tahminler tablosu (seviye, maç, tahmin, olasılık, oran, değer ...)."""
    from src.predict import _fit_models
    matches = db.load_matches()
    if as_of:
        start = pd.Timestamp(as_of)
        up, train, price = _history(matches, start, days), matches[matches["date"] < start], "B365"
        title = f"# Tavsiye provası — {start:%d.%m.%Y} ve sonraki {days} gün (sonuçlarla)"
    else:
        (up, last_snap), train, price = _upcoming(fixtures_path), matches, "İddaa"
        title = f"# Tavsiyeler — {date.today():%d.%m.%Y}"
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
    is_nat = up["div"] == config.MILLI
    club, nat = up[~is_nat], up[is_nat]
    if motivation and not club.empty:
        train = add_motivation(train)
        club = add_motivation(train, club)
    up = pd.concat([club, nat], ignore_index=True, sort=False)
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
    from src import kupon
    cands, scores, corners, iyms = [], [], [], []
    for r in up.to_dict("records"):
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
        i, j = np.unravel_index(np.argmax(m), m.shape)
        sc = score_odds(r.get(SCORE_COL))  # None = bilinmiyor, {} = İddaa'da skor bahsi açık değil
        s_odd = sc.get(f"{i}-{j}", np.nan) if sc is not None else np.nan
        if not (listed and sc is not None and not np.isfinite(s_odd)):
            scores.append({**info, "tahmin": f"{i}-{j}", "p": float(m[i, j]), "odd": s_odd})
        for bet, p in kupon.iyms_probs(m).items():  # Tier 8: ilk yarı / maç sonu
            odd = r.get(_iyms_col(bet), np.nan)
            if odd > 1.0 or not listed:
                iyms.append({**info, "tahmin": bet, "p": p, "odd": odd if odd > 1.0 else np.nan})
        cm = cmodels.get(r["div"])
        if cm is not None and listed:  # korner: sadece İddaa'da oranı olan bahisler
            cmat = korner.matrix(*cm.expected_goals(r["home"], r["away"], r["div"]))
            for bet, p in korner.market_probs(cmat, r.get("kor_line", np.nan)).items():
                odd = r.get(_corner_col(bet), np.nan)
                if odd > 1.0 and odd >= config.KORNER_MIN_ODDS:
                    corners.append({**info, "kaynak": "korner", "tahmin": bet, "p": p, "odd": odd})
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

    picked, rung, order = [], {}, []  # merdiven: maç -> son seviyedeki tahmini (görülme sırasıyla)
    for k, (tname, lo, hi) in enumerate(TIERS):
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
            nxt = band[(band["maç"] == maç) & band["tahmin"].isin(NARROWER[rung[maç]])]
            if len(nxt):
                ladder.append(nxt.sort_values(["p", "değer"], ascending=False).index[0])
        lad = band.loc[ladder].head(MAX_PER_TIER[k] // 2).assign(merdiven=True)  # en fazla yarısı merdiven
        # gerisi yeni maçlar (merdivendeki maça ilgisiz ikinci tahmin yok), maç başına tek tahmin: önce (piyasaya
        # göre) değeri pozitif olanlar, sonra değer, sonra olasılık
        rest = band[~band["maç"].isin(order)]
        rest = rest.sort_values(["sıra", "değer", "p"], ascending=False).groupby("maç").head(1)
        rest = rest.sort_values(["sıra", "p"], ascending=False).head(MAX_PER_TIER[k] - len(lad)).assign(merdiven=False)
        t = pd.concat([lad, rest]).sort_values(["sıra", "p"], ascending=False)
        for maç, bet in zip(t["maç"], t["tahmin"]):
            if maç not in rung:
                order.append(maç)
            rung[maç] = bet
        picked.append((tname, t))
        md += [f"## {tname}", ""] + ([line(x) for x in t.to_dict("records")] or ["Bu seviyede tahmin yok."]) + [""]
    s = pd.DataFrame(scores, columns=CAND_COLS).sort_values("p", ascending=False)
    s["değer"], s["gerçek_değer"] = s["odd"] * s["p"] - 1, False  # kesin skor olasılığı piyasaya ayarlı değil
    picked += [("⚡ Tier 4 — Çok yüksek risk (kısa kesin skor)", s.head(MAX_SHORT_SCORES)),
               ("🔥 Tier 5 — Aşırı risk (tüm maçlarda kesin skor)", s)]
    for tname, t in picked[3:]:
        md += [f"## {tname}", ""] + [line(x) for x in t.to_dict("records")] + [""]
    k = pd.DataFrame(corners, columns=CAND_COLS)
    k["değer"], k["gerçek_değer"] = k["odd"] * k["p"] - 1, False  # keskin korner oranı yok: değer ölçülemez
    k = k[k["p"] >= 0.5].sort_values("p", ascending=False).groupby("maç").head(1).head(MAX_CORNERS)
    picked.append((CORNER_TIER, k))
    md += [f"## {CORNER_TIER}", "",
           "Kulüp maçlarında toplam korner alt / üst (İddaa'nın barajında) ve en çok korner (1 / X / 2); maç başına en "
           f"olası bahis (oran {config.KORNER_MIN_ODDS:.2f}+). Olasılıklar korner modelinden (geçmiş maçlarda "
           "kalibrasyonu iyi; en çok kornerde lig ortalamasından belirgin isabetli, toplam kornerde fark küçük). "
           "Milli maçlarda korner verisi olmadığından yok.", ""]
    md += [line(x) for x in k.to_dict("records")] or ["İddaa'da korner oranı açık kulüp maçı yok."]
    md += [""]
    t8 = pd.DataFrame(iyms, columns=CAND_COLS)
    t8["değer"], t8["gerçek_değer"] = t8["odd"] * t8["p"] - 1, False
    t8 = t8.sort_values("p", ascending=False).groupby("maç").head(1).head(MAX_IYMS)  # maç başına en olası İY/MS
    picked.append((IYMS_TIER, t8))
    md += [f"## {IYMS_TIER}", "",
           "Maç başına en olası ilk yarı / maç sonu sonucu (ör. 1/1 = ilk yarıyı da maçı da ev sahibi önde bitirir; 2/1 "
           "= ilk yarı deplasman önde, maçı ev sahibi kazanır). Olasılıklar maç sonu skor tablosundan, her golün ilk "
           "yarıda atılma olasılığı %44.6 (geçmiş maçlarda ölçüldü).", ""]
    md += [line(x) for x in t8.to_dict("records")] or ["İY/MS oranı açık maç yok."]
    md += [""]
    if not as_of:
        from src import kupon
        legs = pd.concat([t for name, t in picked if name.startswith(("🟢", "🟡", "🚩"))], ignore_index=True)
        legs = legs.reindex(columns=list(dict.fromkeys([*CAND_COLS, *legs.columns])))
        combo, sistem = _coupons(legs, pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
                                 + pd.Timedelta(hours=3))
        picked += [(KUPON_TIER, combo), (SISTEM_TIER, sistem)]
        md += [f"## {KUPON_TIER}", ""]
        if combo.empty:
            md += ["Bugün ve yarın için uygun kombine bulunamadı.", ""]
        else:
            md += [f"Toplam oran **{np.prod(combo['odd']):.2f}**, tutma olasılığı **%{100 * np.prod(combo['p']):.0f}** "
                   "(maçlar bağımsız sayılır).", ""] + [line(x) for x in combo.to_dict("records")] + [""]
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
    if judged:
        md += ["", "## Sonuç özeti", "", "| seviye | tahmin | tuttu | isabet | oranı olan | kâr/zarar (1 birim) | beklenen isabet |",
               "|---|---|---|---|---|---|---|"]
        for tname, t in picked:
            if t.empty or tname == IYMS_TIER:  # provada ilk yarı skoru yok
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
