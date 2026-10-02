"""Telegram'da gönderilen tavsiyelerin sonuçları: tutup tutmadığı, seviye bazında isabet ve kâr/zarar.

Kayıt: telegram.notify her gönderimde mesajdaki tahminleri tavsiyeler/gecmis.csv'ye ekler (ilk gönderildiği andaki
olasılık ve İddaa oranıyla; aynı maç ve tahmin bir kez).
Sonuç: kulüp maçları football-data'dan (veritabanı), milli maçlar milli sonuç verisinden (src/milli.py; son maçlar
ESPN'den). Veri güncellendikçe kayıtlar kapanır: kulüplerde 1-3 gün, milli maçlarda birkaç saat.
Rapor: bir günün tüm tahminleri sonuçlanınca, en geç ertesi sabah (RAPOR_SAATI'nden sonra; sonucu henüz gelmeyenler
⏳ ile) o günün sonuç mesajı Telegram'a bir kez gider; geç gelen sonuçlar toplamlara sonradan eklenir.
tavsiyeler/sonuclar.md tüm zamanların özetidir.
Kâr/zarar: her tahmine tek başına 1 birim, İddaa oranıyla (MBS / kombine hesaba katılmaz; oranı olmayan tahmin
kâr/zarara girmez). "Beklenen" = tahminlerin ortalama olasılığı: gerçek isabet buna yakınsa olasılıklar doğrudur.
"""
import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src import korner
from src.tavsiye import bet_won

LOG_COLS = ["gönderildi", "gün", "tarih", "baslama", "div", "home", "away", "maç", "lig", "seviye", "tahmin", "p",
            "odd", "değer", "kaynak", "merdiven", "hg", "ag", "hk", "ak"]   # hk / ak: korner sayıları
SEVIYELER = [("🟢", "Tier 1"), ("🟡", "Tier 2"), ("🔴", "Tier 3"), ("⚡", "Tier 4"), ("🔥", "Tier 5"),
             ("🚩", "Korner"), ("💎", "💎 Değerli")]
RAPOR_SAATI = 9          # TR saati: bir günün raporu en geç ertesi gün bu saatten sonra gider (eksik sonuç ⏳)
RAPOR_LINKI = "https://github.com/Armann13/valuebet/blob/main/tavsiyeler/sonuclar.md"


def log_path():
    return config.TAVSIYE_DIR / "gecmis.csv"


def load_log() -> pd.DataFrame:
    p = log_path()
    if not p.exists():
        return pd.DataFrame(columns=LOG_COLS)
    return pd.read_csv(p, dtype={"tarih": str, "gün": str, "baslama": str})


def _key(df: pd.DataFrame) -> pd.Series:
    """Aynı maçın aynı tahmini bir kez sayılır (sabah ve akşam özetinde tekrar gelse de); 💎 ayrı tutulur."""
    return df["gün"] + "|" + df["maç"] + "|" + df["tahmin"] + "|" + df["seviye"].str.startswith("💎").astype(str)


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
    """Oynanmış maçlar: kulüp (veritabanı) + milli. Sütunlar: div, date, home, away, fthg, ftag, hc, ac (korner;
    milli maçlarda yok)."""
    from src import db, milli
    cols = ["div", "date", "home", "away", "fthg", "ftag", "hc", "ac"]
    parts = []
    try:
        parts.append(db.load_matches().reindex(columns=cols))
    except Exception as e:  # veritabanı yoksa (python main.py build) sadece milli
        print(f"  ! sonuçlar: kulüp maçları okunamadı ({str(e)[:100]})")
    nat = milli.load_results(since="2026-01-01")
    if not nat.empty:
        parts.append(nat.reindex(columns=cols))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def settle(log: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Sonucu olmayan kayıtlara skor (hg, ag) ve korner sayısı (hk, ak) ekler (aynı lig, aynı takımlar, tarih ±1
    gün: saat dilimi farkı). Milli maçlarda tarafsız sahada ev / deplasman sırası farklı yazılmış olabilir: ters
    sıra da denenir."""
    log = log.reindex(columns=list(dict.fromkeys([*log.columns, "hk", "ak"])))
    res = results.reindex(columns=["div", "date", "home", "away", "fthg", "ftag", "hc", "ac"]).dropna(
        subset=["fthg", "ftag"])
    idx = {}
    for div, d, h, a, hg, ag, hc, ac in zip(res["div"], pd.to_datetime(res["date"]), res["home"], res["away"],
                                            res["fthg"], res["ftag"], res["hc"], res["ac"]):
        idx.setdefault((div, h, a), []).append((d, (int(hg), int(ag)), (hc, ac)))
    corner = log["tahmin"].map(korner.is_corner_bet)
    for i in log.index[log["hg"].isna() | (corner & log["hk"].isna())]:
        r, day = log.loc[i], pd.Timestamp(log.at[i, "gün"])
        for h, a, swap in ((r["home"], r["away"], False), (r["away"], r["home"], True)):
            hit = next(((g, k) for d, g, k in idx.get((r["div"], h, a), []) if abs((d - day).days) <= 1), None)
            if hit:
                (hg, ag), (hc, ac) = hit
                log.at[i, "hg"], log.at[i, "ag"] = (ag, hg) if swap else (hg, ag)
                log.at[i, "hk"], log.at[i, "ak"] = (ac, hc) if swap else (hc, ac)
                break
    return log


def _won(df: pd.DataFrame) -> pd.Series:
    """Tahmin tuttu mu (True / False; sonuç yoksa NaN). Korner bahisleri korner sayılarıyla."""
    hk = df["hk"] if "hk" in df else pd.Series(np.nan, index=df.index)
    ak = df["ak"] if "ak" in df else pd.Series(np.nan, index=df.index)
    out = []
    for t, h, a, kh, ka in zip(df["tahmin"], df["hg"], df["ag"], hk, ak):
        if korner.is_corner_bet(t):
            out.append(korner.bet_won(t, int(kh), int(ka)) if pd.notna(kh) and pd.notna(ka) else np.nan)
        else:
            out.append(bet_won(t, int(h), int(a)) if pd.notna(h) and pd.notna(a) else np.nan)
    return pd.Series(out, index=df.index, dtype=object)


def _score_text(r) -> str:
    """Skor ("2-1"); korner bahsinde korner sayısı ("korner 6-4"); sonuç yoksa boş."""
    if korner.is_corner_bet(r["tahmin"]):
        return f"korner {int(r['hk'])}-{int(r['ak'])}" if pd.notna(r.get("hk")) and pd.notna(r.get("ak")) else ""
    return f"{int(r['hg'])}-{int(r['ag'])}" if pd.notna(r["hg"]) else ""


def stats(df: pd.DataFrame) -> list[dict]:
    """Seviye bazında: sonuçlanan, tutan, isabet, beklenen isabet, kâr/zarar (1 birim, İddaa oranı)."""
    out = []
    won = _won(df)
    for icon, name in SEVIYELER:
        sel = df["seviye"].str.startswith(icon)
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


def day_message(day: str, df: pd.DataFrame, allt: pd.DataFrame) -> str:
    """Bir günün sonuç mesajı (Telegram HTML): seviye özeti, tek tek tahminler, başlangıçtan beri toplam."""
    won = _won(df)
    lines = [f"📊 <b>Sonuçlar — {pd.Timestamp(day):%d.%m.%Y}</b>",
             f"<i>Telegram'da gönderilen {len(df)} tahmin; her biri tek başına 1 birim, İddaa oranıyla</i>", ""]
    lines += [_stat_line(s) for s in stats(df)]
    lines += ["", "<b>Ayrıntı</b>"]
    for icon, _ in SEVIYELER:
        for (_, r), w in zip(df[df["seviye"].str.startswith(icon)].iterrows(), won[df["seviye"].str.startswith(icon)]):
            mark = "⏳" if pd.isna(w) else ("✅" if w else "❌")
            score = _score_text(r)
            score = f" ({score})" if score else " (sonuç yok)"
            odd = f" @{r['odd']:.2f}" if pd.notna(r["odd"]) else ""
            lines.append(f"{mark} {icon} {html.escape(str(r['maç']))}: {html.escape(str(r['tahmin']))}{score}{odd}")
    done = _won(allt).notna().sum()
    lines += ["", f"<b>Başlangıçtan beri</b> ({done} sonuçlanan tahmin)"] + [_stat_line(s) for s in stats(allt)]
    lines += ["", f'<a href="{RAPOR_LINKI}">Tüm sonuçlar</a>']
    return "\n".join(lines)


def summary_md(log: pd.DataFrame) -> str:
    """tavsiyeler/sonuclar.md: tüm zamanlar, gün gün özet ve son tahminler."""
    md = ["# Tavsiye sonuçları", "",
          f"Son güncelleme: {datetime.now():%d.%m.%Y %H:%M}. Sadece Telegram'da gönderilen tahminler (ilk "
          "gönderildiği andaki olasılık ve İddaa oranıyla). Kâr/zarar: her tahmine tek başına 1 birim (kombine / MBS "
          "hesaba katılmaz). **Beklenen** = tahminlerin ortalama olasılığı; gerçek isabet buna yakınsa olasılıklar "
          "doğru demektir. Kâr isabetten değil fiyattan gelir: 💎 dışındaki tahminlerin İddaa fiyatı adil fiyatın "
          "altındadır, uzun vadede zarar beklenir.", "", "## Tüm zamanlar", "",
          "| seviye | tahmin | sonuçlanan | tutan | isabet | beklenen | kâr/zarar (birim) | ROI |",
          "|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, p=False: "—" if pd.isna(v) else (f"%{100 * v:.0f}" if p else f"{v:+.2f}")  # noqa: E731
    for s in stats(log):
        md.append(f"| {s['icon']} {s['seviye']} | {s['tahmin']} | {s['sonuçlanan']} | {s['tutan']} | "
                  f"{fmt(s['isabet'], True)} | {fmt(s['beklenen'], True)} | {fmt(s['kâr_zarar'])} | {fmt(s['roi'], True)} |")
    md += ["", "## Gün gün", "", "| gün | tahmin | sonuçlanan | tutan | isabet | beklenen | kâr/zarar |",
           "|---|---|---|---|---|---|---|"]
    won = _won(log)
    for day, g in log.groupby("gün", sort=False):
        w = won[g.index].dropna().astype(bool)
        odd = g.loc[w.index, "odd"].astype(float)
        has = odd.notna()
        pl = np.where(w[has], odd[has] - 1, -1.0).sum() if has.any() else np.nan
        md.append(f"| {pd.Timestamp(day):%d.%m.%Y} | {len(g)} | {len(w)} | {int(w.sum())} | "
                  f"{fmt(w.mean() if len(w) else np.nan, True)} | {fmt(g.loc[w.index, 'p'].mean() if len(w) else np.nan, True)} | "
                  f"{fmt(pl)} |")
    md += ["", "## Son 200 tahmin", "", "| gün | seviye | maç | tahmin | olasılık | İddaa | skor | sonuç |",
           "|---|---|---|---|---|---|---|---|"]
    for (_, r), w in list(zip(log.iterrows(), won))[::-1][:200]:
        odd = "—" if pd.isna(r["odd"]) else f"{r['odd']:.2f}"
        score = _score_text(r) or "—"
        mark = "⏳" if pd.isna(w) else ("✅" if w else "❌")
        md.append(f"| {r['tarih']} | {r['seviye'][:1]} | {r['maç']} | {r['tahmin']} | %{100 * r['p']:.0f} | {odd} | "
                  f"{score} | {mark} |")
    return "\n".join(md)


def due_days(log: pd.DataFrame, done_days: set, now_tr: pd.Timestamp) -> list:
    """Sonuç mesajı gidecek günler: geçmiş, henüz gönderilmemiş ve ya tamamı sonuçlanmış ya da ertesi gün
    RAPOR_SAATI geçmiş."""
    won, out = _won(log), []
    for day, g in log.groupby("gün"):
        d = pd.Timestamp(day)
        if day in done_days or d >= now_tr.normalize():
            continue
        if won[g.index].notna().all() or now_tr >= d + pd.Timedelta(days=1, hours=RAPOR_SAATI):
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
    log = settle(log, load_results())
    log.to_csv(log_path(), index=False)
    (config.TAVSIYE_DIR / "sonuclar.md").write_text(summary_md(log), encoding="utf-8")
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
