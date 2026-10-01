"""Telegram bildirimi: tavsiyeler/son.csv'den (python main.py tavsiye --yayinla) mesaj.

  Özet: günde iki kez (TR saatiyle OZET_SAATLERI'nin her birinden sonraki ilk çalıştırmada: sabah ve akşam maçlarından
        önce) bugün ve yarının maçları için 💎 değerli bahisler + Tier 1-4'ten öne çıkanlar (Tier 4: kesin skor) +
        tam raporun linki.
  Anlık uyarı: daha önce gönderilmemiş yeni bir 💎 (oran keskin piyasanın adil fiyatının üstünde) çıkınca.
Gönderilenler tavsiyeler/telegram_durum.json'da tutulur (aynı bahis iki kez gönderilmez).

Kurulum: bot anahtarı ve sohbet numarası ortam değişkenlerinden okunur (GitHub'da gizli ayar / secret):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID. Tanımlı değilse hiçbir şey gönderilmez. Bkz. `python main.py telegram --kurulum`.
"""
import html
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config

API = "https://api.telegram.org/bot{token}/{method}"
OZET_SAATLERI = {9: "Günün tavsiyeleri", 18: "Akşam güncellemesi"}  # TR saati -> başlık; o saatten sonraki ilk çalıştırmada
MAX_LEN = 4000          # Telegram sınırı 4096 karakter
RAPOR_LINKI = "https://github.com/Armann13/valuebet/blob/main/tavsiyeler/son.md"
TIER_LIMIT = {"🟢": 5, "🟡": 5, "🔴": 3, "⚡": 4}   # ⚡ = Tier 4, en olası kesin skorlar


def _tr_now() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=3)


def send(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("  telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID tanımlı değil, mesaj gönderilmedi")
        return False
    parts, cur = [], ""
    for line in text.split("\n"):  # uzun mesajı satır sınırından böl
        if len(cur) + len(line) + 1 > MAX_LEN:
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


def _fmt(r) -> str:
    s = (f"• <b>{html.escape(r['maç'])}</b> ({r['lig']}, {r['tarih']}): <b>{html.escape(r['tahmin'])}</b> — "
         f"%{100 * r['p']:.0f}, adil {1 / r['p']:.2f}")
    if pd.notna(r.get("odd")):
        s += f", İddaa {r['odd']:.2f} ({100 * r['değer']:+.0f}%{' 💎' if r.get('gerçek_değer') else ''})"
    if pd.notna(r.get("mbs")) and r["mbs"] > 1:
        s += f", MBS {int(r['mbs'])}"
    return s


def build_messages(picks: pd.DataFrame, state: dict, now_tr: datetime) -> tuple[list, dict]:
    """Gönderilecek mesajlar ve güncellenmiş durum (saf fonksiyon: test edilebilir)."""
    msgs, state = [], {"son_ozet": state.get("son_ozet"), "gonderilen": list(state.get("gonderilen", []))}
    if picks.empty:
        return msgs, state
    soon = picks[picks["gün"] <= (now_tr + timedelta(days=1)).strftime("%Y-%m-%d")]
    value = picks[picks["seviye"].str.startswith("💎")].drop_duplicates("maç")
    sent = set(state["gonderilen"])
    passed = [h for h in OZET_SAATLERI if now_tr.hour >= h]
    slot = f"{now_tr:%Y-%m-%d} {max(passed):02d}" if passed else None  # bugünün son geçilen özet saati
    if slot and state["son_ozet"] != slot and not soon.empty:
        lines = [f"⚽ <b>{OZET_SAATLERI[max(passed)]} — {now_tr:%d.%m}</b>", ""]
        v = soon[soon["seviye"].str.startswith("💎")]
        lines += ["💎 <b>Değerli (oran adil fiyatın üstünde)</b>"] + ([_fmt(r) for r in v.to_dict("records")] or ["yok"])
        for icon, n in TIER_LIMIT.items():
            t = soon[soon["seviye"].str.startswith(icon)].head(n)
            if not t.empty:
                lines += ["", f"<b>{html.escape(t['seviye'].iloc[0])}</b>"] + [_fmt(r) for r in t.to_dict("records")]
        lines += ["", "Olasılık = tutma şansı; 💎 = fiyatı iyi. Sadece 💎 uzun vadede kârlı olmayı bekler.",
                  f'<a href="{RAPOR_LINKI}">Tam rapor</a>']
        msgs.append("\n".join(lines))
        state["son_ozet"] = slot
        sent |= {_key(r) for r in v.to_dict("records")}  # özette gönderilen 💎 anlık uyarıda tekrar gelmesin
    new = [r for r in value.to_dict("records") if _key(r) not in sent]
    if new:
        msgs.append("\n".join(["💎 <b>Yeni değerli bahis</b>", ""] + [_fmt(r) for r in new] +
                              ["", f'<a href="{RAPOR_LINKI}">Tam rapor</a>']))
        sent |= {_key(r) for r in new}
    state["gonderilen"] = sorted(sent)[-500:]  # dosya şişmesin
    return msgs, state


def notify() -> int:
    """tavsiyeler/son.csv'den mesajları gönderir; durum dosyasını günceller. Döner: gönderilen mesaj sayısı."""
    path, state_path = config.TAVSIYE_DIR / "son.csv", config.TAVSIYE_DIR / "telegram_durum.json"
    if not path.exists() or path.stat().st_size == 0:
        print("  telegram: tavsiyeler/son.csv yok")
        return 0
    try:
        picks = pd.read_csv(path, dtype={"tarih": str, "gün": str})  # "01.10" sayı (1.1) olarak okunmasın
    except pd.errors.EmptyDataError:
        picks = pd.DataFrame()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    msgs, new_state = build_messages(picks, state, _tr_now())
    if not msgs:
        print("  telegram: gönderilecek yeni bir şey yok")
        return 0
    if all(send(m) for m in msgs):  # gönderilemezse durum güncellenmez, bir sonraki çalıştırmada tekrar dener
        state_path.write_text(json.dumps(new_state, ensure_ascii=False, indent=1), encoding="utf-8")
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
