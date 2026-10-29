"""Telegram'da gönderilen tavsiyelerin sonuçları: tutup tutmadığı, seviye bazında isabet ve kâr/zarar.

Kayıt: telegram.notify her gönderimde mesajdaki tahminleri tavsiyeler/gecmis.csv'ye ekler (ilk gönderildiği andaki
olasılık ve İddaa oranıyla; aynı maç ve tahmin bir kez).
Sonuç: kulüp maçları football-data'dan (veritabanı), milli maçlar milli sonuç verisinden (src/milli.py; son maçlar
ESPN'den). Veri güncellendikçe kayıtlar kapanır: kulüplerde 1-3 gün, milli maçlarda birkaç saat.
Rapor: bir günün tüm tahminleri sonuçlanınca, en geç ertesi sabah (RAPOR_SAATI'nden sonra; sonucu henüz gelmeyenler
⏳ ile) o günün sonuç mesajı Telegram'a bir kez gider; geç gelen sonuçlar toplamlara sonradan eklenir.
tavsiyeler/sonuclar.md tüm zamanların özetidir; her seviyenin tüm maçları tavsiyeler/sonuclar/<seviye>.md'de.
Kâr/zarar: her tahmine tek başına 1 birim, İddaa oranıyla (MBS / kombine hesaba katılmaz; oranı olmayan tahmin
kâr/zarara girmez). "Beklenen" = tahminlerin ortalama olasılığı: gerçek isabet buna yakınsa olasılıklar doğrudur.
Sistem kupon: gönderilen her kupon (aynı anda gönderilen 🎰 ayakları) bir bütün olarak, önerilen "n'de k" ile:
yatırılan kolon sayısı ve dönen; son ayağının günü raporunda.
"""
import html
import json
import re
from datetime import datetime
from math import comb

import numpy as np
import pandas as pd

import config
from src import korner, kupon
from src.tavsiye import KUPON_IKON, KUPON_TURLERI, bet_won

LOG_COLS = ["gönderildi", "gün", "tarih", "baslama", "div", "home", "away", "maç", "lig", "seviye", "tahmin", "p",
            "odd", "değer", "kaynak", "merdiven", "hg", "ag", "hk", "ak", "ihg", "iag",
            "iptal", "mesaj"]  # hk/ak korner, ihg/iag ilk yarı; iptal: geçersiz kayıt (bkz. _void); mesaj: "kadro"
# Kadro güncellemesi (mesaj="kadro") tahminleri ayrı tutulur: seviye istatistikleri ilk verilen tahminlerle; kadro
# öncesi / sonrası karşılaştırması ayrı (kadro_compare). Bu zamandan önceki kadro mesajlarında sadece değişen tahminler
# kaydedildi (aynı kalanlar sabahki kayıtla birleşti): o maçlarda kadro kaydı olmayan seviye "aynı kaldı" sayılır.
KADRO_TAM_KAYIT = "2026-10-25 00:00"
SEVIYELER = [("🟢", "Tier 1"), ("🟡", "Tier 2"), ("🔴", "Tier 3"), ("🟣", "Tier 4"), ("⚡", "Tier 5"), ("🔥", "Tier 6"),
             ("🔀", "Tier 8 İY/MS"), ("🚩", "Korner (tüm seviyeler)"), ("🎫", "Kupon ayakları"),
             ("🎰", "Sistem ayakları"), ("🎯", "İdeal kupon ayakları"),
             *[(i, f"{n.split(' ', 1)[1]} ayakları") for i, n in zip(KUPON_IKON, (t[0] for t in KUPON_TURLERI))],
             ("🎲", "Çılgın kupon ayakları"), ("💎", "💎 Değerli")]   # 🚩: seviyeye değil bahis türüne göre (korner)
AYRI = ("💎", "🎫", "🎰", "🎯", *KUPON_IKON, "🎲")   # bu bölümlerdeki tahmin başka seviyede de gönderildiyse ayrı sayılır
KUPONLAR = ("🎰", "🎯", *KUPON_IKON, "🎲")   # her gönderilen kupon ayrı (bir bütün olarak sonuçlanır)
# n'de n: hepsi tutmalı, dönen = oranların çarpımı. 2'li / 3'lü / 4'lü 2026-10-29'dan beri (ideal kuponun yerine)
TEK_KOLON = {**{i: t[0].split(" ", 1)[1] for i, t in zip(KUPON_IKON, KUPON_TURLERI)}, "🎲": "Çılgın kupon",
             "🎯": "İdeal kupon"}
SAYFA = {"🟢": "tier1", "🟡": "tier2", "🔴": "tier3", "🟣": "tier4", "⚡": "tier5", "🔥": "tier6", "🔀": "iyms",
         "🚩": "korner", "🎫": "kupon", "🎰": "sistem", "🎯": "ideal",
         **{i: f"kupon{t[1]}" for i, t in zip(KUPON_IKON, KUPON_TURLERI)},
         "🎲": "cilgin", "💎": "degerli"}   # tavsiyeler/sonuclar/<ad>.md
GUNLER = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")
RAPOR_SAATI = 8          # TR saati: bir günün raporu en geç ertesi gün bu saatten sonra gider (eksik sonuç ⏳)
RAPOR_LINKI = "https://github.com/Armann13/valuebet/blob/main/tavsiyeler/sonuclar.md"


def log_path():
    return config.TAVSIYE_DIR / "gecmis.csv"


def load_log() -> pd.DataFrame:
    p = log_path()
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_csv(p, dtype={"tarih": str, "gün": str, "baslama": str})


def _key(df: pd.DataFrame) -> pd.Series:
    """Aynı maçın aynı tahmini bir kez sayılır (sabah ve akşam özetinde tekrar gelse de); 💎, kupon ve sistem
    ayakları ayrı tutulur. Her gönderilen sistem kuponu ayrıdır (aynı ayak iki kuponda olabilir; kupon eksik kalmasın)."""
    group = df["seviye"].map(lambda s: next((i for i in AYRI if str(s).startswith(i)), ""))
    group = group.where(~group.isin(KUPONLAR), group + df["gönderildi"].astype(str))
    group = group.where(~_kadro(df), "📋" + group)  # kadro güncellemesi ilk tahminlerden ayrı
    return df["gün"] + "|" + df["maç"] + "|" + df["tahmin"] + "|" + group


def _kadro(df: pd.DataFrame) -> pd.Series:
    """Kadro güncellemesi mesajından gelen kayıtlar."""
    if "mesaj" not in df:
        return pd.Series(False, index=df.index)
    return df["mesaj"].astype(str).eq("kadro")


def record(rows: list, sent_at: datetime) -> int:
    """Telegram'da gönderilen tahmin satırlarını geçmişe ekler. Döner: yeni kayıt sayısı."""
    if not rows:
        return 0
    new = pd.DataFrame(rows).assign(gönderildi=f"{sent_at:%Y-%m-%d %H:%M}").reindex(columns=LOG_COLS)
    old = load_log()
    new = new[~_key(new).isin(set(_key(old))) & ~_key(new).duplicated()]
    if new.empty:
        return 0
    config.TAVSIYE_DIR.mkdir(parents=True, exist_ok=True)
    (pd.concat([old, new], ignore_index=True) if len(old) else new).to_csv(log_path(), index=False)
    return len(new)


def load_results() -> pd.DataFrame:
    """Oynanmış maçlar: kulüp (veritabanı; henüz gelmeyenler ESPN'den) + milli. Sütunlar: div, date, home, away,
    fthg, ftag, hc, ac (korner), hthg, htag (ilk yarı). Aynı maç iki kaynakta varsa önce gelen (football-data) kullanılır."""
    from src import db, kulup_sonuc, milli
    cols = ["div", "date", "home", "away", "fthg", "ftag", "hc", "ac", "hthg", "htag"]
    parts = []
    try:
        parts.append(db.load_matches().reindex(columns=cols))
    except Exception as e:  # veritabanı yoksa (python main.py build) sadece milli
        print(f"  ! sonuçlar: kulüp maçları okunamadı ({str(e)[:100]})")
    club = kulup_sonuc.load()
    if not club.empty:
        parts.append(club.reindex(columns=cols))
    nat = milli.load_results(since="2026-01-01")
    if not nat.empty:
        parts.append(nat.reindex(columns=cols))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def settle(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Sonucu olmayan kayıtlara skor (hg, ag) ve korner sayısı (hk, ak) ekler (aynı lig, aynı takımlar, tarih ±1
    gün: saat dilimi farkı). Milli maçlarda tarafsız sahada ev / deplasman sırası farklı yazılmış olabilir: ters
    sıra da denenir."""
    log = log.reindex(columns=list(dict.fromkeys([*log.columns, "hk", "ak", "ihg", "iag"])))
    res = results.reindex(columns=["div", "date", "home", "away", "fthg", "ftag", "hc", "ac", "hthg", "htag"]
                          ).dropna(subset=["fthg", "ftag"])
    idx = {}
    for div, d, h, a, hg, ag, hc, ac, ih, ia in zip(res["div"], pd.to_datetime(res["date"]), res["home"],
                                                    res["away"], res["fthg"], res["ftag"], res["hc"], res["ac"],
                                                    res["hthg"], res["htag"]):
        idx.setdefault((div, h, a), []).append((d, (int(hg), int(ag)), (hc, ac), (ih, ia)))
    corner = log["tahmin"].map(korner.is_corner_bet)
    half = log["tahmin"].map(kupon.is_iyms)
    for i in log.index[log["hg"].isna() | (corner & log["hk"].isna()) | (half & log["ihg"].isna())]:
        r, day = log.loc[i], pd.Timestamp(log.at[i, "gün"])
        for h, a, swap in ((r["home"], r["away"], False), (r["away"], r["home"], True)):
            hit = next(((g, k, t) for d, g, k, t in idx.get((r["div"], h, a), []) if abs((d - day).days) <= 1), None)
            if hit:
                (hg, ag), (hc, ac), (ih, ia) = hit
                log.at[i, "hg"], log.at[i, "ag"] = (ag, hg) if swap else (hg, ag)
                log.at[i, "hk"], log.at[i, "ak"] = (ac, hc) if swap else (hc, ac)
                log.at[i, "ihg"], log.at[i, "iag"] = (ia, ih) if swap else (ih, ia)
                break
    return log


def _void(df: pd.DataFrame) -> pd.Series:
    """İptal edilen tahminler (iptal sütunu; ör. kadın maçının erkek takımına eşlendiği kayıt): isabete ve kâra
    girmez; sistem kuponda İddaa'daki gibi 1.00 oranla tutmuş sayılır."""
    if "iptal" not in df:
        return pd.Series(False, index=df.index)
    return df["iptal"].astype(str).str.lower().isin(["true", "1"])


def _mark(r, w) -> str:
    if str(r.get("iptal")).lower() in ("true", "1"):
        return "🚫"
    return "⏳" if pd.isna(w) else ("✅" if w else "❌")


def _won(df: pd.DataFrame) -> pd.Series:
    """Tahmin tuttu mu (True / False; sonuç yoksa ya da iptalse NaN). Korner bahisleri korner sayılarıyla."""
    col = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)  # noqa: E731
    ok = lambda *v: all(pd.notna(x) for x in v)  # noqa: E731
    out = []
    for t, h, a, kh, ka, ih, ia in zip(df["tahmin"], df["hg"], df["ag"], col("hk"), col("ak"), col("ihg"), col("iag")):
        if korner.is_corner_bet(t):
            out.append(korner.bet_won(t, int(kh), int(ka)) if ok(kh, ka) else np.nan)
        elif kupon.is_iyms(t):
            if ok(ih, ia, h, a):
                out.append(kupon.iyms_won(t, int(ih), int(ia), int(h), int(a)))
            elif ok(h, a) and not bet_won(f"MS {t.split('/')[-1]}", int(h), int(a)):
                out.append(False)  # maç sonu tutmadı: ilk yarı skoru olmadan da kaybetti
            else:
                out.append(np.nan)
        else:
            out.append(bet_won(t, int(h), int(a)) if ok(h, a) else np.nan)
    return pd.Series(out, index=df.index, dtype=object).mask(_void(df), np.nan)


def _score_text(r) -> str:
    """Skor ("2-1"); korner bahsinde korner sayısı ("korner 6-4"), İY/MS'de ilk yarı da ("İY 1-0, MS 2-1")."""
    if korner.is_corner_bet(r["tahmin"]):
        return f"korner {int(r['hk'])}-{int(r['ak'])}" if pd.notna(r.get("hk")) and pd.notna(r.get("ak")) else ""
    if kupon.is_iyms(r["tahmin"]):
        if pd.notna(r.get("ihg")) and pd.notna(r.get("iag")) and pd.notna(r["hg"]):
            return f"İY {int(r['ihg'])}-{int(r['iag'])}, MS {int(r['hg'])}-{int(r['ag'])}"
        return ""
    return f"{int(r['hg'])}-{int(r['ag'])}" if pd.notna(r["hg"]) else ""


def systems(log: pd.DataFrame, icon: str = "🎰") -> list[dict]:
    """Gönderilen her sistem kuponu (aynı anda gönderilen 🎰 ayakları) bir bütün olarak: önerilen n'de k (gönderirken
    seviyeye yazılır: "🎰 Sistem kupon · 7'de 6"; eski kayıtlarda aynı hesapla bulunur), yatırılan kolon sayısı ve
    dönen (birim). gün: son ayağın günü (kupon o günün raporunda). icon TEK_KOLON'daysa (🎯 ideal, 🎲 çılgın) n'de n
    (tek kolon, hepsi tutmalı; dönen = oranların çarpımı). oran: ayakların oran çarpımı."""
    s = log[log["seviye"].astype(str).str.startswith(icon)]
    won, out = _won(s), []
    for sent, g in s.groupby("gönderildi"):
        n = len(g)
        m = re.search(r"(\d+)'[dt][ae] (\d+)", " ".join(g["seviye"].astype(str)))
        if icon in TEK_KOLON:
            k = n
        elif m:
            k = int(m[2])
        elif g["p"].notna().all() and g["odd"].notna().all():
            k = kupon.best_system(list(g["p"]), list(g["odd"]))["k"]
        else:
            continue
        w, void = won[g.index], _void(g)
        done = bool((w.notna() | void).all())
        legs, odds = w.eq(True) | void, g["odd"].astype(float).mask(void, 1.0)  # iptal ayak: 1.00 oranla tutmuş
        pay = kupon.system_payout(list(legs), list(odds), k) if done else np.nan
        out.append({"gönderildi": sent, "gün": g["gün"].max(), "n": n, "k": k, "kolon": comb(n, k),
                    "tutan": int(w.eq(True).sum()), "bitti": done, "dönen": pay, "oran": float(odds.prod())})
    return out


def _tl(x: float) -> str:
    return f"{x:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def _coupon_line(s: dict, icon: str = "🎲") -> str:
    """Tek kolonluk kuponun (🎯 / 🎲) sonuç satırı."""
    head = (f"{icon} {TEK_KOLON[icon]} ({pd.Timestamp(s['gönderildi']):%d.%m %H:%M} gönderilen, {s['n']} maç, oran "
            f"{_tl(s['oran'])})")
    if not s["bitti"]:
        return f"{head}: sonuç bekleniyor ({s['tutan']} maç tuttu, diğerleri ⏳)"
    if s["dönen"] > 0:
        return f"{head}: ✅ TUTTU! 1 birim → {_tl(s['dönen'])} birim"
    return f"{head}: ❌ {s['tutan']}/{s['n']} maç tuttu"


def _system_line(s: dict) -> str:
    head = f"🎰 Sistem kupon {s['n']}'de {s['k']} ({pd.Timestamp(s['gönderildi']):%d.%m %H:%M} gönderilen)"
    if not s["bitti"]:
        return f"{head}: sonuç bekleniyor ({s['tutan']} maç tuttu, diğerleri ⏳)"
    net = s["dönen"] - s["kolon"]
    return (f"{head}: {s['tutan']}/{s['n']} maç tuttu · {s['kolon']} birim yatırıldı, {s['dönen']:.2f} döndü "
            f"({net:+.2f}) {'✅' if net > 0 else '❌'}")


def _select(df: pd.DataFrame, icon: str) -> pd.Series:
    """Bir seviye satırına giren tahminler: seviye simgesine göre; 🚩 bahis türüne göre (tüm korner tahminleri)."""
    if icon == "🚩":
        return df["tahmin"].map(korner.is_corner_bet).astype(bool)
    return df["seviye"].astype(str).str.startswith(icon)


def stats(df: pd.DataFrame) -> list[dict]:
    """Seviye bazında: sonuçlanan, tutan, isabet, beklenen isabet, kâr/zarar (1 birim, İddaa oranı). İptaller ve kadro
    güncellemesi tahminleri hariç (ilk verilen tahminler; kadro ayrı: kadro_compare)."""
    out = []
    df = df[~_void(df) & ~_kadro(df)]
    won = _won(df)
    for icon, name in SEVIYELER:
        sel = _select(df, icon)
        done = sel & won.notna()
        if not sel.any():
            continue
        w = won[done].astype(bool)
        odd = df.loc[done, "odd"].astype(float)
        has = odd.notna()
        pl = float(np.where(w[has], odd[has] - 1, -1.0).sum()) if has.any() else np.nan
        out.append({"icon": icon, "seviye": name, "tahmin": int(sel.sum()), "sonuçlanan": int(done.sum()),
                    "tutan": int(w.sum()), "isabet": w.mean() if len(w) else np.nan,
                    "beklenen": df.loc[done, "p"].mean() if done.any() else np.nan,
                    "oranlı": int(has.sum()), "kâr_zarar": pl, "roi": pl / has.sum() if has.any() else np.nan})
    return out


def _stat_line(s: dict) -> str:
    if s["sonuçlanan"] == 0:
        return f"{s['icon']} {s['seviye']}: {s['tahmin']} tahmin, henüz sonuç yok"
    txt = (f"{s['icon']} {s['seviye']}: {s['tutan']}/{s['sonuçlanan']} tuttu (%{100 * s['isabet']:.0f}, beklenen "
           f"%{100 * s['beklenen']:.0f})")
    if s["oranlı"]:
        txt += f", {s['kâr_zarar']:+.2f} birim"
    return txt


def _detail(r, w, prefix: str = "") -> str:
    """Sonuç mesajında bir tahmin: işaret, (önek), maç: tahmin (skor) @oran."""
    mark = _mark(r, w)
    score = _score_text(r)
    score = f" ({score})" if score else (" (iptal)" if mark == "🚫" else " (sonuç yok)")
    odd = f" @{r['odd']:.2f}" if pd.notna(r["odd"]) else ""
    return f"{prefix}{mark} {html.escape(str(r['maç']))}: {html.escape(str(r['tahmin']))}{score}{odd}"


def day_message(day: str, df: pd.DataFrame, allt: pd.DataFrame) -> str:
    """Bir günün sonuç mesajı (Telegram HTML): önce kuponlar ayaklarıyla (2026-10-29'dan beri Telegram kupon
    gönderiyor), 💎'ler, kadro öncesi / sonrası, seviye özeti (Tier listeleri tam raporda; ayrıntı GitHub
    sayfalarında), başlangıçtan beri toplam."""
    df = df[~_kadro(df)]  # kadro güncellemesi tahminleri ayrı satırda (kadro_compare)
    won, allw = _won(df), _won(allt)
    lines = [f"📊 <b>Sonuçlar — {pd.Timestamp(day):%d.%m.%Y}</b>"]
    for icon in TEK_KOLON:
        for s in systems(allt, icon):
            if s["gün"] != day:
                continue
            legs = allt[allt["seviye"].astype(str).str.startswith(icon) & (allt["gönderildi"] == s["gönderildi"])]
            lines += ["", _coupon_line(s, icon)] + [_detail(r, allw[i], "   ") for i, r in legs.iterrows()]
    lines += [_system_line(s) for s in systems(allt) if s["gün"] == day]
    v = df[df["seviye"].astype(str).str.startswith("💎")]
    if len(v):
        lines += ["", "<b>💎 Değerli</b>"] + [_detail(r, won[i]) for i, r in v.iterrows()]
    cmp = kadro_compare(allt[allt["gün"] == day])
    if cmp:
        lines += ["", _kadro_line(cmp)]
    lines += ["", "<b>Seviye özeti</b> <i>(her tahmin tek başına 1 birim, İddaa oranıyla)</i>"]
    lines += [_stat_line(s) for s in stats(df)]
    done = _won(allt).notna().sum()
    lines += ["", f"<b>Başlangıçtan beri</b> ({done} sonuçlanan tahmin)"] + [_stat_line(s) for s in stats(allt)]
    lines += ["", f'<a href="{RAPOR_LINKI}">Tüm sonuçlar (Tier Tier her maç)</a>']
    return "\n".join(lines)


KADRO_SEVIYE = ("🟢", "🟡", "🔴", "🟣")   # karşılaştırılan seviyeler (kesin skor ve kuponlar hariç)


def kadro_compare(log: pd.DataFrame) -> dict | None:
    """Kadro güncellemesi gelen maçlarda ilk verilen tahminler (önce) ile kadro mesajındakiler (sonra), Tier 1-4.
    Döner: {"maç": [maç başına seviyeler: önce / sonra tahminleri ve sonuçları], "önce" / "sonra": özet (tahmin,
    sonuçlanan, tutan, kâr/zarar), "değişen": eski / yeni tutan sayısı}; kadro kaydı yoksa None."""
    k = _kadro(log)
    if not k.any():
        return None
    won, icon = _won(log), log["seviye"].astype(str).str[:1]
    base = icon.isin(KADRO_SEVIYE) & ~_void(log)
    sent = log[k].groupby(["gün", "maç"])["gönderildi"].min()  # maçın kadro mesajı zamanı
    rows, once, sonra, changed = [], [], [], {"n": 0, "eski": 0, "yeni": 0, "bitti": 0}
    for (day, maç), when in sent.items():
        same = (log["gün"] == day) & (log["maç"] == maç) & base
        tiers = []
        for i in KADRO_SEVIYE:
            old = log[same & ~k & (icon == i)]
            new = log[same & k & (icon == i)]
            if new.empty and str(when) < KADRO_TAM_KAYIT:  # eski kayıt: aynı kalan tahmin ayrıca kaydedilmedi
                new = old
            if old.empty and new.empty:
                continue
            o = [(t, won[x], odd) for x, t, odd in zip(old.index, old["tahmin"], old["odd"])]
            n = [(t, won[x], odd) for x, t, odd in zip(new.index, new["tahmin"], new["odd"])]
            once += o
            sonra += n
            if o and n and {t for t, _, _ in o} != {t for t, _, _ in n}:
                changed["n"] += 1
                done = all(pd.notna(w) for _, w, _ in o + n)
                if done:
                    changed["bitti"] += 1
                    changed["eski"] += any(bool(w) for _, w, _ in o)
                    changed["yeni"] += any(bool(w) for _, w, _ in n)
            tiers.append({"icon": i, "önce": o, "sonra": n})
        rows.append({"gün": day, "maç": maç, "zaman": when, "seviyeler": tiers})

    def summary(picks):
        done = [(w, odd) for _, w, odd in picks if pd.notna(w)]
        pl = sum((float(odd) - 1 if w else -1.0) for w, odd in done if pd.notna(odd))
        return {"tahmin": len(picks), "sonuçlanan": len(done), "tutan": sum(bool(w) for w, _ in done), "kâr": pl}
    return {"maç": rows, "önce": summary(once), "sonra": summary(sonra), "değişen": changed}


def _kadro_line(c: dict) -> str:
    """Sonuç mesajında kadro öncesi / sonrası satırı."""
    o, s, d = c["önce"], c["sonra"], c["değişen"]
    txt = (f"📋 Kadro güncellemesi ({len(c['maç'])} maç): ilk tahminler {o['tutan']}/{o['sonuçlanan']} tuttu "
           f"({o['kâr']:+.2f}), kadro sonrası liste {s['tutan']}/{s['sonuçlanan']} tuttu ({s['kâr']:+.2f})")
    if d["bitti"]:
        txt += f" · değişen {d['bitti']} tahminde eski {d['eski']}, yeni {d['yeni']} tuttu"
    return txt


def _kadro_page(c: dict) -> str:
    """tavsiyeler/sonuclar/kadro.md: maç maç kadro öncesi / sonrası tahminler."""
    o, s, d = c["önce"], c["sonra"], c["değişen"]
    md = ["# 📋 Kadro öncesi / sonrası", "", "[← Tüm sonuçlar](../sonuclar.md)", "",
          "Kadro güncellemesi gelen maçlarda ilk verilen tahmin (önce) ile kadrolar açıklandıktan sonra son oranlarla "
          "verilen tahmin (sonra), Tier 1-4. Her tahmin 1 birim, İddaa oranıyla. Sabah mesajında maç başına birkaç "
          "tahmin, kadro mesajında o maçların tam listesi vardır: birebir kıyas için **değişen** satırına bak "
          "(aynı maçta aynı seviyede eski ve yeni tahmin).", "",
          "| | tahmin | sonuçlanan | tutan | kâr/zarar |", "|---|---|---|---|---|",
          f"| önce | {o['tahmin']} | {o['sonuçlanan']} | {o['tutan']} | {o['kâr']:+.2f} |",
          f"| sonra | {s['tahmin']} | {s['sonuçlanan']} | {s['tutan']} | {s['kâr']:+.2f} |", "",
          f"Değişen tahmin: {d['n']} ({d['bitti']} sonuçlandı: eski {d['eski']}, yeni {d['yeni']} tuttu).", "",
          "<i>✅ tuttu · ❌ tutmadı · ⏳ sonuç bekleniyor · @ İddaa oranı</i>"]
    name = dict(SEVIYELER)
    fmt = lambda picks: ", ".join(f"{'⏳' if pd.isna(w) else '✅' if w else '❌'} {t}" +  # noqa: E731
                                  (f" @{float(odd):.2f}" if pd.notna(odd) else "") for t, w, odd in picks) or "—"
    for m in sorted(c["maç"], key=lambda x: str(x["zaman"]), reverse=True):
        md += ["", f"## {pd.Timestamp(m['gün']):%d.%m} · {m['maç']}", "",
               f"Kadro mesajı: {pd.Timestamp(m['zaman']):%d.%m %H:%M}", ""]
        for t in m["seviyeler"]:
            a, b = {x for x, _, _ in t["önce"]}, {x for x, _, _ in t["sonra"]}
            what = "aynı" if a == b else "yeni" if not a else "kalktı" if not b else "değişti"
            md.append(f"- {t['icon']} {name[t['icon']]}: {fmt(t['önce'])} → {fmt(t['sonra'])} ({what})")
    return "\n".join(md) + "\n"


def _pick_line(r, w, prefix: str = "") -> str:
    """Sayfada bir tahmin: sonuç, (önek), tahmin (↳ merdiven), İddaa oranı, skor."""
    mark = _mark(r, w)
    parts = [f"{mark} {prefix}" + ("↳ " if str(r.get("merdiven")) == "True" else "") + str(r["tahmin"])]
    if pd.notna(r["odd"]):
        parts.append(f"@{r['odd']:.2f}")
    if _score_text(r):
        parts.append(_score_text(r))
    if mark == "🚫":
        parts.append("iptal")
    return "- " + " · ".join(parts)


def _by_day(sel: pd.DataFrame, won: pd.Series, show_tier: bool) -> list[str]:
    """Gün gün (en yeni gün üstte), gün içinde maç saatine göre; her maçın altında tahminleri."""
    t = pd.to_datetime(sel["baslama"], errors="coerce")
    day = t.dt.strftime("%Y-%m-%d").fillna(sel["gün"])
    out = []
    for d in sorted(day.unique(), reverse=True):
        g = sel[day == d].assign(_t=t).sort_values("_t", kind="stable")
        out += ["", f"## {pd.Timestamp(d):%d.%m.%Y} {GUNLER[pd.Timestamp(d).dayofweek]}"]
        for m in dict.fromkeys(g["maç"]):
            mg = g[g["maç"] == m]
            when = f"{mg['_t'].iloc[0]:%H:%M} · " if pd.notna(mg["_t"].iloc[0]) else ""
            out += ["", f"**{when}{m}**", ""]
            out += [_pick_line(r, won[i], f"{str(r['seviye'])[:1]} " if show_tier else "") for i, r in mg.iterrows()]
    return out


def _by_coupon(sel: pd.DataFrame, won: pd.Series, log: pd.DataFrame, icon: str = "🎰") -> list[str]:
    """Kupon sayfası (sistem / çılgın): her gönderilen kupon (en yeni üstte), sonucu ve maçları."""
    coupons = {c["gönderildi"]: c for c in systems(log, icon)}
    out = []
    for sent, g in sorted(sel.groupby("gönderildi"), key=lambda x: x[0], reverse=True):
        c = coupons.get(sent)
        what = (f" · {c['n']} maç, oran {_tl(c['oran'])}" if icon in TEK_KOLON else f" · {c['n']}'de {c['k']}") if c else ""
        out += ["", f"## {pd.Timestamp(sent):%d.%m %H:%M} gönderilen kupon{what}", ""]
        if c and icon in TEK_KOLON:
            out += [_coupon_line(c, icon).split("): ", 1)[1], ""]
        elif c and c["bitti"]:
            net = c["dönen"] - c["kolon"]
            out += [f"**{c['tutan']}/{c['n']} maç tuttu · {c['kolon']} birim yatırıldı, {c['dönen']:.2f} döndü "
                    f"({net:+.2f})** {'✅' if net > 0 else '❌'}", ""]
        elif c:
            out += [f"Sonuç bekleniyor ({c['tutan']} maç tuttu, diğerleri ⏳)", ""]
        t = pd.to_datetime(g["baslama"], errors="coerce")
        for i, r in g.loc[t.sort_values(kind="stable").index].iterrows():
            when = f"{t[i]:%d.%m %H:%M} " if pd.notna(t[i]) else ""
            out.append(_pick_line(r, won[i], f"{when}{r['maç']} · "))
    return out


def tier_pages(log: pd.DataFrame) -> dict:
    """Seviye sayfaları (tavsiyeler/sonuclar/<ad>.md): o seviyenin tüm tahminleri maç maç, gün gün; sistem kuponda
    kupon kupon; kadro öncesi / sonrası (kadro.md). Döner: sayfa adı -> markdown."""
    won, pages = _won(log), {}
    cmp = kadro_compare(log)
    if cmp:
        pages["kadro"] = _kadro_page(cmp)
    for s in stats(log):
        icon = s["icon"]
        sel = log[_select(log, icon) & ~_kadro(log)]  # ilk verilen tahminler (kadro sayfası ayrı)
        pending = s["tahmin"] - s["sonuçlanan"]
        if s["sonuçlanan"]:
            head = (f"**{s['tutan']}/{s['sonuçlanan']} tuttu** (isabet %{100 * s['isabet']:.0f}, beklenen "
                    f"%{100 * s['beklenen']:.0f})")
            if s["oranlı"]:
                head += f" · {s['kâr_zarar']:+.2f} birim"
        else:
            head = "Henüz sonuçlanan yok"
        if pending:
            head += f" · {pending} tahmin sonuç bekliyor"
        md = [f"# {icon} {s['seviye']}", "", "[← Tüm sonuçlar](../sonuclar.md)", "", head, "",
              "<i>✅ tuttu · ❌ tutmadı · ⏳ sonuç bekleniyor · 🚫 iptal (hesaba girmez) · ↳ bir üst seviyedeki "
              "tahminin aynı maçta devamı · @ İddaa oranı</i>"]
        md += (_by_coupon(sel, won, log, icon) if icon in KUPONLAR else
               _by_day(sel, won, show_tier=icon == "🚩"))
        pages[SAYFA[icon]] = "\n".join(md) + "\n"
    return pages


def write_pages(log: pd.DataFrame) -> None:
    """Seviye sayfalarını tavsiyeler/sonuclar/ altına yazar (sonuclar.md'deki tablodan bağlantılı)."""
    folder = config.TAVSIYE_DIR / "sonuclar"
    folder.mkdir(parents=True, exist_ok=True)
    pages = tier_pages(log)
    for name, text in pages.items():
        (folder / f"{name}.md").write_text(text, encoding="utf-8")
    for old in folder.glob("*.md"):  # artık olmayan seviyenin sayfası (ör. numara kayınca) eski içerikle kalmasın
        if old.stem not in pages:
            old.unlink()


def summary_md(log: pd.DataFrame) -> str:
    """tavsiyeler/sonuclar.md: tüm zamanlar (seviyeler kendi sayfasına bağlantılı), sistem kuponları, gün gün özet."""
    md = ["# Tavsiye sonuçları", "",
          f"Son güncelleme: {datetime.now():%d.%m.%Y %H:%M}. Telegram mesajına giren tahminler (ilk gönderildiği andaki "
          "olasılık ve İddaa oranıyla; 29.10'dan beri Telegram'da kuponlar ve 💎 gösteriliyor, Tier'lar tam raporda ve "
          "aynı ölçütle burada izleniyor). Kâr/zarar: her tahmine tek başına 1 birim (kombine / MBS "
          "hesaba katılmaz). **Beklenen** = tahminlerin ortalama olasılığı; gerçek isabet buna yakınsa olasılıklar "
          "doğru demektir. Kâr isabetten değil fiyattan gelir: 💎 dışındaki tahminlerin İddaa fiyatı adil fiyatın "
          "altındadır, uzun vadede zarar beklenir.", "", "## Tüm zamanlar", "",
          "**Seviyeye tıkla:** o seviyedeki tüm maçlar tek tek, gün gün açılır.", "",
          "| seviye | tahmin | sonuçlanan | tutan | isabet | beklenen | kâr/zarar (birim) | ROI |",
          "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, p=False: "—" if pd.isna(v) else (f"%{100 * v:.0f}" if p else f"{v:+.2f}")  # noqa: E731
    for s in stats(log):
        md.append(f"| [{s['icon']} {s['seviye']}](sonuclar/{SAYFA[s['icon']]}.md) | {s['tahmin']} | {s['sonuçlanan']} | "
                  f"{s['tutan']} | {fmt(s['isabet'], True)} | {fmt(s['beklenen'], True)} | {fmt(s['kâr_zarar'])} | "
                  f"{fmt(s['roi'], True)} |")
    sis = systems(log)
    if sis:
        md += ["", "## Sistem kuponları", "", "Her gönderilen kupon bir bütün olarak, önerilen n'de k ile (kolon başına "
               "1 birim). Yukarıdaki \"Sistem ayakları\" satırı ayakları tek tek sayar.", "",
               "| gönderildi | kupon | tutan maç | yatırılan | dönen | kâr/zarar |", "|---|---|---|---|---|---|"]
        for s in sis:
            pay = "⏳" if not s["bitti"] else f"{s['dönen']:.2f}"
            net = "⏳" if not s["bitti"] else fmt(s["dönen"] - s["kolon"])
            md.append(f"| {pd.Timestamp(s['gönderildi']):%d.%m %H:%M} | {s['n']}'de {s['k']} | {s['tutan']}/{s['n']} | "
                      f"{s['kolon']} | {pay} | {net} |")
        fin = [s for s in sis if s["bitti"]]
        if fin:
            md.append(f"| **toplam** ({len(fin)} kupon) | | | {sum(s['kolon'] for s in fin)} | "
                      f"{sum(s['dönen'] for s in fin):.2f} | {fmt(sum(s['dönen'] - s['kolon'] for s in fin))} |")
    cmp = kadro_compare(log)
    if cmp:
        o, s, d = cmp["önce"], cmp["sonra"], cmp["değişen"]
        md += ["", "## [📋 Kadro öncesi / sonrası](sonuclar/kadro.md)", "",
               f"Kadro güncellemesi gelen {len(cmp['maç'])} maçta (Tier 1-4) ilk verilen tahmin ile kadrolar "
               "açıklandıktan sonraki tahmin. Maç maç ayrıntı için başlığa tıkla.", "",
               "| | tahmin | sonuçlanan | tutan | kâr/zarar |", "|---|---|---|---|---|",
               f"| önce | {o['tahmin']} | {o['sonuçlanan']} | {o['tutan']} | {o['kâr']:+.2f} |",
               f"| sonra | {s['tahmin']} | {s['sonuçlanan']} | {s['tutan']} | {s['kâr']:+.2f} |", "",
               f"Değişen tahmin: {d['n']} ({d['bitti']} sonuçlandı: eski {d['eski']}, yeni {d['yeni']} tuttu)."]
    notes = {**{i: f"29.10'dan beri; {lo:.2f}-{hi:.2f} oranlı {n} ayak (MS, MS & 2.5 Üst, MS & KG Var, en çok korner; "
                   "değere göre), hepsi farklı maç" for i, (_, n, lo, hi) in zip(KUPON_IKON, KUPON_TURLERI)},
             "🎯": "3 banko (Tier 1 MS / çifte şans) + oran yükseltici (Tier 3-4 en çok korner / MS & KG Var; "
                   "25.10'dan beri 2, 26.10'dan beri 2.50+ oranlı 2-3 tane); 29.10'da yerini 2'li kupon aldı",
             "🎲": "İY/MS dönüşleri + 💎"}
    for icon, name in TEK_KOLON.items():
        cr = systems(log, icon)
        if not cr:
            continue
        md += ["", f"## {name}lar", "", f"{notes[icon]}; hepsi tutmalı (1 birim, dönen = toplam oran).", "",
               "| gönderildi | maç | toplam oran | tutan maç | sonuç |", "|---|---|---|---|---|"]
        for s in cr:
            res = "⏳" if not s["bitti"] else ("✅ " + _tl(s["dönen"]) if s["dönen"] > 0 else "❌")
            md.append(f"| {pd.Timestamp(s['gönderildi']):%d.%m %H:%M} | {s['n']} | {_tl(s['oran'])} | "
                      f"{s['tutan']}/{s['n']} | {res} |")
        fin = [s for s in cr if s["bitti"]]
        if fin:
            md.append(f"| **toplam** ({len(fin)} kupon) | | | | {len(fin)} birim → {_tl(sum(s['dönen'] for s in fin))} |")
    md += ["", "## Gün gün", "", "| gün | tahmin | sonuçlanan | tutan | isabet | beklenen | kâr/zarar |",
           "|---|---|---|---|---|---|---|"]
    won = _won(log)
    for day, g in log[~_kadro(log)].groupby("gün", sort=False):
        w = won[g.index].dropna().astype(bool)
        odd = g.loc[w.index, "odd"].astype(float)
        has = odd.notna()
        pl = np.where(w[has], odd[has] - 1, -1.0).sum() if has.any() else np.nan
        md.append(f"| {pd.Timestamp(day):%d.%m.%Y} | {len(g)} | {len(w)} | {int(w.sum())} | "
                  f"{fmt(w.mean() if len(w) else np.nan, True)} | {fmt(g.loc[w.index, 'p'].mean() if len(w) else np.nan, True)} | "
                  f"{fmt(pl)} |")
    return "\n".join(md)


def due_days(log: pd.DataFrame, done_days: set, now_tr: pd.Timestamp) -> list:
    """Sonuç mesajı gidecek günler: geçmiş, henüz gönderilmemiş ve ya tamamı sonuçlanmış ya da ertesi gün
    RAPOR_SAATI geçmiş."""
    won, out = _won(log), []
    for day, g in log.groupby("gün"):
        d = pd.Timestamp(day)
        if day in done_days or d >= now_tr.normalize():
            continue
        if (won[g.index].notna() | _void(g)).all() or now_tr >= d + pd.Timedelta(days=1, hours=RAPOR_SAATI):
            out.append(day)
    return out


def run(send: bool = True, now_tr: pd.Timestamp | None = None) -> int:
    """Kayıtları sonuçlarla kapatır, sonuclar.md'yi yazar, zamanı gelen günlerin sonuç mesajını gönderir.
    Döner: gönderilen mesaj sayısı."""
    from src import telegram
    log = load_log()
    if log.empty:
        print("  sonuçlar: henüz kayıtlı tahmin yok (Telegram'da gönderilenler kaydedilir)")
        return 0
    try:  # football-data'ya henüz gelmemiş kulüp sonuçları
        from src import kulup_sonuc
        kulup_sonuc.download(log)
    except Exception as e:
        print(f"  ! kulüp sonuçları (ESPN) alınamadı: {str(e)[:150]}")
    log = settle(log, load_results())
    log.to_csv(log_path(), index=False)
    (config.TAVSIYE_DIR / "sonuclar.md").write_text(summary_md(log), encoding="utf-8")
    write_pages(log)
    state_path = config.TAVSIYE_DIR / "telegram_durum.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    done_days = set(state.get("sonuc_gonderilen", []))
    now_tr = now_tr if now_tr is not None else pd.Timestamp(telegram._tr_now().replace(tzinfo=None))
    won = _won(log)
    sent = 0
    for day in due_days(log, done_days, now_tr):
        msg = day_message(day, log[log["gün"] == day], log)
        if not send:
            print(msg)
            continue
        if telegram.send(msg):
            done_days.add(day)
            sent += 1
    if sent:
        state["sonuc_gonderilen"] = sorted(done_days)[-200:]
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    n_done = int(won.notna().sum())
    print(f"  sonuçlar: {len(log)} kayıt, {n_done} sonuçlandı, {sent} gün raporu gönderildi")
    return sent
