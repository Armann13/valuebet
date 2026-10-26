"""Telegram bildirimi: tavsiyeler/son.csv'den (python main.py tavsiye --yayinla) mesaj.

  Özet: günde iki kez (TR saatiyle OZET_SAATLERI'nin her birinden sonraki ilk çalıştırmada: sabah ve akşam maçlarından
        önce) bugün ve yarının maçları için 💎 değerli bahisler + Tier 1-6'dan öne çıkanlar (Tier 5-6: kesin skor) +
        tam raporun linki.
  Anlık uyarı: daha önce gönderilmemiş yeni bir 💎 (oran keskin piyasanın adil fiyatının üstünde) çıkınca.
Gönderilenler tavsiyeler/telegram_durum.json'da tutulur (aynı bahis iki kez gönderilmez).

Kurulum: bot anahtarı ve sohbet numarası ortam değişkenlerinden okunur (GitHub'da gizli ayar / secret):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Tanımlı değilse hiçbir şey gönderilmez. Bkz. `python main.py telegram --kurulum`.
"""
import html
import json
import os
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config
from src import korner
from src.tavsiye import is_ladder

API = "https://api.telegram.org/bot{token}/{method}"
# TR saati -> başlık; o saatten sonraki ilk çalıştırmada. Sabah 08: GitHub sabah işini sık sık geciktirip (08:2x-08:4x)
# 09:xx işlerini atlıyor; geç gelen iş de özeti göndersin.
OZET_SAATLERI = {8: "Günün tavsiyeleri", 18: "Akşam güncellemesi"}
MAX_LEN = 4000          # Telegram sınırı 4096 karakter (etiketler hariç, bkz. _visible_len)
RAPOR_LINKI = "https://github.com/Armann13/valuebet/blob/main/tavsiyeler/son.md"
KORNER_MAC = 2   # maç başına en fazla bu kadar korner tahmini (en düşük riskliler)
BASLIK = {  # simge -> (başlık, açıklama)
    "💎": ("DEĞERLİ", "oran adil fiyatın üstünde"), "🟢": ("TIER 1", "düşük risk"), "🟡": ("TIER 2", "orta risk"),
    "🔴": ("TIER 3", "yüksek risk"), "🟣": ("TIER 4", "çok yüksek risk"), "⚡": ("TIER 5", "kesin skor"),
    "🔥": ("TIER 6", "kesin skor · yukarıdaki maçlar"),
    "🔀": ("TIER 8", "ilk yarı / maç sonu"), "🚩": ("KORNER", "kulüp maçları"),
    "🎫": ("KOMBİNE KUPON", "orta risk"), "🎰": ("SİSTEM KUPON", ""),
    "🎯": ("İDEAL KUPON", "3 banko + 2-3 oran yükseltici (2.50+)"),
    "🎲": ("ÇILGIN KUPON", "İY/MS dönüşleri + 💎 · piyango gibi")}
CIZGI = "━━━━━━━━━━━━━━━"


def _tr_now() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=3)


def _visible_len(text: str) -> int:
    """Telegram'ın saydığı uzunluk: HTML etiketleri çıkmış, &amp; gibi karakterler açılmış metin, UTF-16 birimi
    (emoji 2 sayılır). 4096 sınırı buna uygulanır."""
    return len(html.unescape(re.sub(r"<[^>]+>", "", text)).encode("utf-16-le")) // 2


def send(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilmedi")
        return False
    parts, cur = [], ""
    for line in text.split("\n"):  # uzun mesajı satır sınırından böl (satırlar etiketleri kendi içinde kapatır)
        if cur and _visible_len(cur + line) + 1 > MAX_LEN:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    parts.append(cur)
    for p in parts:
        r = requests.post(API.format(token=token, method="sendMessage"), timeout=30,
                          data={"chat_id": chat, "text": p, "parse_mode": "HTML", "disable_web_page_preview": True})
        if not r.ok:
            print(f"  ! telegram hatası: {r.status_code} {r.text[:200]}")
            return False
    return True


def _key(r) -> str:
    return f"{r['gün']}|{r['maç']}|{r['tahmin']}"


def _when(r, now_tr: datetime) -> str:
    """Maç saati (TR): bugünse "21:45", yarınsa "Yarın 19:00", saat bilinmiyorsa tarih."""
    b = r.get("baslama")
    try:
        t = datetime.strptime(str(b), "%Y-%m-%d %H:%M")
    except ValueError:
        return str(r.get("tarih", ""))
    if t.date() == now_tr.date():
        return f"{t:%H:%M}"
    if t.date() == (now_tr + timedelta(days=1)).date():
        return f"Yarın {t:%H:%M}"
    return f"{t:%d.%m %H:%M}"


def _sorted(t: pd.DataFrame) -> pd.DataFrame:
    """Seçilen tahminler başlama saatine göre (saati olmayanlar sonda)."""
    if "baslama" not in t:
        return t
    key = t["baslama"].where(t["baslama"].astype(str).str.len() >= 16, "9999")
    return t.assign(_k=key.astype(str)).sort_values("_k", kind="stable").drop(columns="_k")


def _fmt(r, now_tr: datetime | None = None) -> str:
    """İki satır: "🕘 saat · maç" ve "➜ tahmin · İddaa oranı" (↳ = üst seviyedeki tahminin aynı maçta devamı)."""
    now_tr = now_tr or _tr_now()
    head = f"🕘 {_when(r, now_tr)} · <b>{html.escape(str(r['maç']))}</b>"
    bet = f"{'↳' if is_ladder(r) else '➜'} {html.escape(str(r['tahmin']))}"
    if pd.notna(r.get("odd")):
        bet += f" · <b>{r['odd']:.2f}</b>"
    if r.get("gerçek_değer") is True or str(r.get("gerçek_değer")) == "True":
        bet += " 💎"
    if pd.notna(r.get("mbs")) and r["mbs"] > 1:
        bet += f" · MBS {int(r['mbs'])}"
    return f"{head}\n{bet}"


def _playable(rows: pd.DataFrame) -> pd.DataFrame:
    """Sadece İddaa'da oynanabilen tahminler: oranı olan (kapalı ya da henüz açılmamış bahis listeye girmez)."""
    return rows[pd.to_numeric(rows["odd"], errors="coerce") > 1.0]


def _section(icon: str, rows: pd.DataFrame, now_tr: datetime) -> list:
    """Bölüm: çizgi, başlık, açıklama; tahminler arasında boş satır."""
    title, sub = BASLIK[icon]
    out = ["", CIZGI, f"{icon} <b>{title}</b>" + (f" · <i>{sub}</i>" if sub else "")]
    for r in _sorted(rows).to_dict("records"):
        out += ["", _fmt(r, now_tr)]
    return out


def _tl(x: float) -> str:
    """Türkçe sayı biçimi: 23815.78 -> "23.815,78"."""
    return f"{x:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def _coupon_lines(soon: pd.DataFrame, now_tr: datetime, log: list, icon: str) -> list:
    """Kupon bölümü (🎯 ideal, 🎲 çılgın): ayaklar ve toplam oran; hepsi tutmalı. Bir ayağın oranı kapandıysa
    (oynanamaz) kupon eksik kalmasın diye hiç gösterilmez."""
    from src.tavsiye import CILGIN_N, IDEAL_BANKO, IDEAL_YUKSELTICI
    rows = soon[soon["seviye"].str.startswith(icon)]
    t = _playable(rows)
    need = IDEAL_BANKO + IDEAL_YUKSELTICI[0] if icon == "🎯" else CILGIN_N
    if len(t) < len(rows) or (t["tahmin"].str.startswith("İY/MS").sum() if icon == "🎲" else len(t)) < need:
        return []
    log += t.to_dict("records")
    total = float(t["odd"].astype(float).prod())
    return _section(icon, t, now_tr) + ["", f"🧾 <b>{len(t)} maç · toplam oran {_tl(total)}</b> (hepsi tutmalı)"]


TIER_SIRA = ("🟢", "🟡", "🔴", "🟣", "⚡", "🔥", "💎")   # maçın altında tahminlerin sırası
EFSANE = ("<i>🟢 Tier 1 düşük · 🟡 Tier 2 orta · 🔴 Tier 3 yüksek · 🟣 Tier 4 çok yüksek risk · ⚡🔥 kesin skor · "
          "💎 değerli (oran adil fiyatın üstünde) · ↳ bir üst seviyedeki tahminin aynı maçta devamı (o tutarsa "
          "üstteki de tutar)</i>")


def _day_label(day, now_tr: datetime) -> str:
    """"BUGÜN · 25.10 Cumartesi", "YARIN · 26.10 Pazar", sonrası "27.10 Pazartesi"."""
    from src.sonuclar import GUNLER
    name = f"{day:%d.%m} {GUNLER[day.weekday()]}"
    if day == now_tr.date():
        return f"BUGÜN · {name}"
    if day == (now_tr + timedelta(days=1)).date():
        return f"YARIN · {name}"
    return name


def _bet_line(r) -> str:
    """Maçın altında bir tahmin: Tier simgesi, (↳ merdiven) tahmin · İddaa oranı (💎 değerliyse)."""
    icon = str(r["seviye"])[:1]
    s = f"   {icon} {'↳ ' if is_ladder(r) else ''}{html.escape(str(r['tahmin']))}"
    if pd.notna(r.get("odd")):
        s += f" · <b>{float(r['odd']):.2f}</b>"
    if icon != "💎" and (r.get("gerçek_değer") is True or str(r.get("gerçek_değer")) == "True"):
        s += " 💎"
    return s


def _by_day(rows: pd.DataFrame, now_tr: datetime) -> list[list[str]]:
    """Gün -> saat -> maç; her maçın altında tahminleri Tier sırasıyla (kullanıcı isteği, 2026-10-25). Döner: gün
    başına satırlar."""
    if rows.empty:
        return []
    t = pd.to_datetime(rows["baslama"], errors="coerce") if "baslama" in rows else pd.Series(pd.NaT, index=rows.index)
    day = t.dt.date.where(t.notna(), pd.to_datetime(rows["gün"]).dt.date)  # saati olmayan: kayıttaki gün
    order = rows["seviye"].astype(str).str[:1].map({k: i for i, k in enumerate(TIER_SIRA)}).fillna(len(TIER_SIRA))
    d = rows.assign(_t=t, _gun=day, _saat=t.dt.strftime("%H:%M").fillna("saat belli değil"), _sira=order)
    d = d.sort_values(["_gun", "_t", "maç", "_sira"], kind="stable", na_position="last")
    days = []
    for gun, dg in d.groupby("_gun", sort=False):
        out = ["", CIZGI, f"📅 <b>{_day_label(gun, now_tr)}</b>"]
        for saat, sg in dg.groupby("_saat", sort=False):
            out += ["", f"🕘 <b>{saat}</b>"]
            for maç, mg in sg.groupby("maç", sort=False):
                mbs = pd.to_numeric(mg["mbs"], errors="coerce").dropna() if "mbs" in mg else pd.Series(dtype=float)
                out.append(f"⚽ <b>{html.escape(str(maç))}</b>" + (f" · MBS {int(mbs.iloc[0])}"
                                                                     if len(mbs) and mbs.iloc[0] > 1 else ""))
                out += [_bet_line(r) for r in mg.to_dict("records")]
        days.append(out)
    return days


def _select(soon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mesaja girecek tahminler: her maçın Tier 1-4 gol tahminleri (seviye başına tek), en düşük riskli KORNER_MAC korner
    tahmini, bir kesin skor (Tier 5 kısa listedeyse o, yoksa Tier 6); ayrıca 💎'ler. Sadece oynanabilenler."""
    play = _playable(soon)
    icon = play["seviye"].astype(str).str[:1]
    rank = icon.map({k: i for i, k in enumerate(TIER_SIRA)})
    tiers = play[icon.isin(TIER_SIRA[:4])]
    corner = tiers["tahmin"].map(korner.is_corner_bet).astype(bool)  # boşken de bool maske olsun
    kor = tiers[corner].assign(_r=rank).sort_values("_r", kind="stable").groupby("maç").head(KORNER_MAC)
    score = play[icon.isin(("⚡", "🔥"))].assign(_r=rank).sort_values("_r", kind="stable").drop_duplicates("maç")
    rows = pd.concat([tiers[~corner], kor.drop(columns="_r"), score.drop(columns="_r")], ignore_index=True)
    return rows, play[icon == "💎"]


def digest_parts(soon: pd.DataFrame, now_tr: datetime, header: str, log: list) -> list:
    """Gün -> saat -> maç düzeninde, her gün ayrı mesaj: bugün / yarın oranı olan her maç, altında Tier 1-4 (İY/MS ve
    korner dahil, merdivenle), kesin skor ve 💎 (_select); son mesajın altında 🎯 ideal ve 🎲 çılgın kupon. Uzun gün
    Telegram sınırında satır sonundan bölünür (send)."""
    rows, v = _select(soon)
    log += rows.to_dict("records") + v.to_dict("records")
    listed = set(zip(rows["maç"], rows["tahmin"]))
    v = v[[(m, b) not in listed for m, b in zip(v["maç"], v["tahmin"])]] if len(v) else v  # listedeki 💎 satır sonunda
    parts = [p for p in (rows, v) if len(p)]
    days = _by_day(pd.concat(parts, ignore_index=True), now_tr) if parts else []
    intro = [header, "<i>📅 gün · 🕘 saat · ⚽ maç; altında tahminler: Tier simgesi, tahmin · İddaa oranı</i>"]
    tail = _coupon_lines(soon, now_tr, log, "🎯") + _coupon_lines(soon, now_tr, log, "🎲")
    tail += ["", CIZGI, EFSANE, f'<a href="{RAPOR_LINKI}">Tam rapor</a>']
    if not days:
        return ["\n".join(intro + ["", "Oynanabilir tahmin yok."] + tail)]
    first = header.split("\n")[0]  # kadro mesajında başlığın altında kadro bilgisi var: devamda sadece ilk satır
    return ["\n".join((intro if i == 0 else [f"{first} · devam"]) + d + (tail if i == len(days) - 1 else []))
            for i, d in enumerate(days)]


def _no_match_note(picks: pd.DataFrame, now_tr: datetime) -> str:
    """Bugün ve yarın tavsiye yoksa (ör. milli maç arası bitti, ligler hafta sonu başlıyor): kısa not ve sıradaki
    maç günleri."""
    from src.sonuclar import GUNLER
    lines = [f"⚽ <b>{OZET_SAATLERI[min(OZET_SAATLERI)]} — {now_tr:%d.%m}</b>", "",
             "Bugün ve yarın takip edilen liglerde tavsiye çıkan maç yok."]
    if not picks.empty:
        days = picks.drop_duplicates(["gün", "maç"]).groupby("gün").size().head(3)
        lines += ["", "<b>Sıradaki maç günleri</b>"] + [
            f"• {pd.Timestamp(d):%d.%m} {GUNLER[pd.Timestamp(d).dayofweek]}: {n} maç" for d, n in days.items()]
    lines += ["", "<i>Tavsiyeler maç gününden bir gün önce sabah listeye girer.</i>"]
    return "\n".join(lines)


def build_messages(picks: pd.DataFrame, state: dict, now_tr: datetime, force: bool = False,
                   log: list | None = None) -> tuple[list, dict]:
    """Gönderilecek mesajlar ve güncellenmiş durum (saf fonksiyon: test edilebilir).
    force: özet bu saat dilimi için zaten gönderildiyse de şimdi gönder (elle istenen güncel özet).
    log: verilirse mesajlara giren tahmin satırları buna eklenir (sonuç takibi, bkz. sonuclar.py)."""
    log = [] if log is None else log
    msgs = []
    state = {**state, "son_ozet": state.get("son_ozet"), "gonderilen": list(state.get("gonderilen", []))}
    passed = [h for h in OZET_SAATLERI if now_tr.hour >= h]
    slot = f"{now_tr:%Y-%m-%d} {max(passed):02d}" if passed else None  # bugünün son geçilen özet saati
    due = force or (slot is not None and state["son_ozet"] != slot)
    # bugün ve yarın maç yoksa sabah (ya da elle istenince) kısa not: sessiz kalınca mesaj gelmedi sanılıyor
    note_due = due and (force or max(passed) == min(OZET_SAATLERI))
    if picks.empty:
        if note_due:
            msgs.append(_no_match_note(picks, now_tr))
            state["son_ozet"] = slot or state["son_ozet"]
        return msgs, state
    soon = picks[picks["gün"] <= (now_tr + timedelta(days=1)).strftime("%Y-%m-%d")]
    value = picks[picks["seviye"].str.startswith("💎")].drop_duplicates("maç")
    sent = set(state["gonderilen"])
    if due and not soon.empty:
        title = "Güncel tavsiyeler" if force else OZET_SAATLERI[max(passed)]
        stamp = f"{now_tr:%d.%m %H:%M}" if force else f"{now_tr:%d.%m}"
        msgs += digest_parts(soon, now_tr, f"⚽ <b>{title} — {stamp}</b>", log)
        state["son_ozet"] = slot or state["son_ozet"]
        v = soon[soon["seviye"].str.startswith("💎")]
        sent |= {_key(r) for r in v.to_dict("records")}  # özette gönderilen 💎 anlık uyarıda tekrar gelmesin
    elif note_due:
        msgs.append(_no_match_note(picks, now_tr))
        state["son_ozet"] = slot or state["son_ozet"]
    new = [r for r in _playable(value).to_dict("records") if _key(r) not in sent]
    if new:
        msgs.append("\n".join(["💎 <b>Yeni değerli bahis</b>", ""] + [_fmt(r, now_tr) for r in new] +
                              ["", f'<a href="{RAPOR_LINKI}">Tam rapor</a>']))
        sent |= {_key(r) for r in new}
        log += new
    state["gonderilen"] = sorted(sent)[-500:]  # dosya şişmesin
    return msgs, state


def notify(force: bool = False) -> int:
    """tavsiyeler/son.csv'den mesajları gönderir; durum dosyasını günceller. Döner: gönderilen mesaj sayısı.
    force: özeti şimdi gönder (bugün zaten gittiyse de)."""
    path, state_path = config.TAVSIYE_DIR / "son.csv", config.TAVSIYE_DIR / "telegram_durum.json"
    if not path.exists() or path.stat().st_size == 0:
        print("  telegram: tavsiyeler/son.csv yok")
        return 0
    try:
        picks = pd.read_csv(path, dtype={"tarih": str, "gün": str, "saat": str, "baslama": str})  # "01.10" sayı sanılmasın
    except pd.errors.EmptyDataError:
        picks = pd.DataFrame()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    rows, now = [], _tr_now()
    msgs, new_state = build_messages(picks, state, now, force, log=rows)
    if not msgs:
        print("  telegram: gönderilecek yeni bir şey yok")
        return 0
    if all(send(m) for m in msgs):  # gönderilemezse durum güncellenmez, bir sonraki çalıştırmada tekrar dener
        state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=1), encoding="utf-8")
        from src import sonuclar
        sonuclar.record(rows, now)  # gönderilen tahminler: maçlar bitince tutup tutmadığı raporlanır
        print(f"  telegram: {len(msgs)} mesaj gönderildi")
        return len(msgs)
    return 0


def test() -> None:
    """GitHub'daki gizli ayarlarla (secrets) mesaj gidiyor mu: tek bir test mesajı."""
    if not send("✅ GitHub'dan test mesajı: tavsiyeler buraya otomatik gelecek (günlük özet + yeni 💎)."):
        raise SystemExit(1)
    print("  telegram: test mesajı gönderildi")


def setup() -> None:
    """Kurulum yardımı: TELEGRAM_BOT_TOKEN ortam değişkeniyle botuna son yazan sohbetin numarasını bulur ve
    test mesajı gönderir. Önce Telegram'da botuna bir mesaj yaz."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip().strip('"').strip("'")
    if not token:
        raise SystemExit('Önce: $env:TELEGRAM_BOT_TOKEN = "BotFather\'ın verdiği anahtar"')
    me = requests.get(API.format(token=token, method="getMe"), timeout=30).json()
    if not me.get("ok"):
        raise SystemExit("Anahtar geçersiz (Telegram kabul etmedi). BotFather'daki anahtarı eksiksiz kopyaladığından "
                         "emin ol: 123456789:ABC... biçiminde, iki nokta dahil.")
    bot = me["result"]["username"]
    print(f"  bu anahtar @{bot} botuna ait -> Telegram'da https://t.me/{bot} adresini aç, Başlat'a bas, bir mesaj yaz")
    r = requests.get(API.format(token=token, method="getUpdates"), timeout=30).json()
    if not r.get("ok"):
        raise SystemExit(f"Telegram yanıtı: {r}")
    chats = {}
    for u in r.get("result", []):
        msg = u.get("message") or u.get("edited_message") or u.get("my_chat_member") or u.get("channel_post")
        if msg and "chat" in msg:
            chats[msg["chat"]["id"]] = msg["chat"].get("first_name") or msg["chat"].get("title", "")
    if not chats:
        raise SystemExit(f"@{bot} botuna henüz mesaj gelmemiş. Telegram'da https://t.me/{bot} adresini aç "
                         "(BotFather'ı değil, kendi botunu), Başlat'a bas, 'merhaba' yaz, sonra bu komutu tekrar çalıştır.")
    for cid, name in chats.items():
        print(f"  sohbet numarası: {cid} ({name})")
    cid = list(chats)[-1]
    os.environ["TELEGRAM_CHAT_ID"] = str(cid)
    if send("✅ valuebet bağlantısı çalışıyor. Tavsiyeler buraya gelecek."):
        print(f"  test mesajı gönderildi. Şimdi: gh secret set TELEGRAM_CHAT_ID --body {cid}")
