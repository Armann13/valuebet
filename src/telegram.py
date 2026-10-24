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
TIER_LIMIT = {"🟢": 5, "🟡": 6, "🔴": 4, "🟣": 4, "⚡": 4}   # ⚡ = Tier 5, en olası kesin skorlar
BASLIK = {  # simge -> (başlık, açıklama)
    "💎": ("DEĞERLİ", "oran adil fiyatın üstünde"), "🟢": ("TIER 1", "düşük risk"), "🟡": ("TIER 2", "orta risk"),
    "🔴": ("TIER 3", "yüksek risk"), "🟣": ("TIER 4", "çok yüksek risk"), "⚡": ("TIER 5", "kesin skor"),
    "🔥": ("TIER 6", "kesin skor · yukarıdaki maçlar"),
    "🔀": ("TIER 8", "ilk yarı / maç sonu"), "🚩": ("KORNER", "kulüp maçları"),
    "🎫": ("KOMBİNE KUPON", "orta risk"), "🎰": ("SİSTEM KUPON", ""),
    "🎲": ("ÇILGIN KUPON", "İY/MS dönüşleri + 💎 · piyango gibi")}
KORNER_LIMIT = 3   # Tier 1-4'te gol tahminlerine ek en fazla bu kadar korner tahmini
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


def _system_lines(soon: pd.DataFrame, now_tr: datetime, log: list) -> list:
    """Sistem kupon: maçlar ve önerilen "n'de k" (kâra geçme şansı en yüksek olan)."""
    from src import kupon
    sistem = _playable(soon[soon["seviye"].str.startswith("🎰")])
    if len(sistem) < 5:
        return []
    n = len(sistem)
    best = kupon.best_system(list(sistem["p"]), list(sistem["odd"]))
    # önerilen k kayda yazılır: sonuç raporunda kupon bir bütün olarak (yatırılan / dönen) hesaplanır
    log += sistem.assign(seviye=f"{sistem['seviye'].iloc[0]} · {n}'de {best['k']}").to_dict("records")
    return _section("🎰", sistem, now_tr) + ["", f"🧾 <b>{n}'de {best['k']} · {best['kolon']} kolon</b> "
                                                  "(kolon başına 1 birim)"]


def _tl(x: float) -> str:
    """Türkçe sayı biçimi: 23815.78 -> "23.815,78"."""
    return f"{x:,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")


def _crazy_lines(soon: pd.DataFrame, now_tr: datetime, log: list) -> list:
    """Çılgın kupon: İY/MS dönüşleri + 💎, toplam oranıyla (tüm ayaklar tutmalı)."""
    from src.tavsiye import CILGIN_N
    t = _playable(soon[soon["seviye"].str.startswith("🎲")])
    if t["tahmin"].str.startswith("İY/MS").sum() < CILGIN_N:  # oranı kapanan dönüş varsa kupon eksik kalmasın
        return []
    log += t.to_dict("records")
    total = float(t["odd"].astype(float).prod())
    return _section("🎲", t, now_tr) + ["", f"🧾 <b>{len(t)} maç · toplam oran {_tl(total)}</b> (hepsi tutmalı)"]


def digest_parts(soon: pd.DataFrame, now_tr: datetime, header: str, log: list) -> list:
    """Tek mesaj: 💎, Tier 1-4 (İY/MS ve korner dahil, merdivenle), Tier 5-6 kesin skor, sistem kupon. Sadece
    İddaa'da oynanabilen (oranı olan) tahminler."""
    out = [header, "<i>🕘 saat · maç   ➜ tahmin · İddaa oranı</i>"]
    v = _playable(soon[soon["seviye"].str.startswith("💎")])
    out += _section("💎", v, now_tr) if not v.empty else ["", CIZGI, "💎 <b>DEĞERLİ</b> · <i>bugün yok</i>"]
    log += v.to_dict("records")
    shown = {}  # seviye simgesi -> mesajda geçen maçlar
    for icon, n in TIER_LIMIT.items():
        t = _playable(soon[soon["seviye"].str.startswith(icon)])
        corner = t["tahmin"].map(korner.is_corner_bet).astype(bool)  # boşken de bool maske olsun
        t = pd.concat([t[~corner].head(n), t[corner].head(KORNER_LIMIT)])  # gol + korner, saat sırasıyla gösterilir
        shown[icon] = set(t["maç"])
        log += t.to_dict("records")
        if not t.empty:
            out += _section(icon, t, now_tr)
    # Tier 6: yukarıdaki (Tier 1-4) maçların kesin skoru; Tier 5'te zaten olanlar hariç
    above = shown["🟢"] | shown["🟡"] | shown["🔴"] | shown["🟣"]
    t5 = _playable(soon[soon["seviye"].str.startswith("🔥") & soon["maç"].isin(above - shown["⚡"])])
    log += t5.to_dict("records")
    if not t5.empty:
        out += _section("🔥", t5, now_tr)
    out += _system_lines(soon, now_tr, log)
    out += _crazy_lines(soon, now_tr, log)
    out += ["", CIZGI, "<i>↳ = bir üst seviyedeki tahminin aynı maçta devamı (bu tutarsa üstteki de tutar)</i>",
            f'<a href="{RAPOR_LINKI}">Tam rapor</a>']
    return ["\n".join(out)]


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
