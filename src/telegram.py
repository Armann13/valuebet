"""Telegram bildirimi: tavsiyeler/son.csv'den (python main.py tavsiye --yayinla) mesaj.

  Özet: günde iki kez (TR saatiyle OZET_SAATLERI'nin her birinden sonraki ilk çalıştırmada: sabah ve akşam maçlarından
        önce) bugün ve yarının maçlarından kuponlar (2'li, 3'lü, 4'lü, çılgın; kullanıcı isteği 2026-10-29: Tier
        listeleri yerine), kuponlarda olmayan 💎'ler ve tam raporun linki. Tier listeleri Telegram'da gösterilmez
        (tam raporda); sonuç takibi için yine kaydedilir.
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
OZET_SAATLERI = {8: "Günün kuponları", 18: "Akşam kuponları"}
MAX_LEN = 4000          # Telegram sınırı 4096 karakter (etiketler hariç, bkz. _visible_len)
RAPOR_LINKI = "https://github.com/Armann13/valuebet/blob/main/tavsiyeler/son.md"
KORNER_MAC = 2   # maç başına en fazla bu kadar korner tahmini (en düşük riskliler)
ADAY_SAYISI = 3  # kadro / analiz mesajında maç başına gösterilen kupon adayı (📌)
BASLIK = {  # simge -> (başlık, açıklama)
    "💎": ("DEĞERLİ", "oran adil fiyatın üstünde"), "🟢": ("TIER 1", "düşük risk"), "🟡": ("TIER 2", "orta risk"),
    "🔴": ("TIER 3", "yüksek risk"), "🟣": ("TIER 4", "çok yüksek risk"), "⚡": ("TIER 5", "kesin skor"),
    "🔥": ("TIER 6", "kesin skor · yukarıdaki maçlar"),
    "🔀": ("TIER 8", "ilk yarı / maç sonu"), "🚩": ("KORNER", "kulüp maçları"),
    "🎫": ("KOMBİNE KUPON", "orta risk"), "🎰": ("SİSTEM KUPON", ""),
    "2️⃣": ("2'Lİ KUPON", "ayaklar 1.70-2.50"), "3️⃣": ("3'LÜ KUPON", "ayaklar 2.00-3.20"),
    "4️⃣": ("4'LÜ KUPON", "ayaklar 2.20-3.50"),
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


def _kac(x: float) -> str:
    """Yaklaşık "kaç kuponda bir": 3.6 -> "4", 23815.4 -> "23.815"."""
    return f"{max(1, round(x)):,}".replace(",", ".")


def _coupon_lines(soon: pd.DataFrame, now_tr: datetime, log: list, icon: str, seen: set) -> list:
    """Kupon bölümü (2️⃣ 3️⃣ 4️⃣, 🎲 çılgın): ayaklar, toplam oran ve yaklaşık kaç kuponda bir tuttuğu; hepsi tutmalı.
    Bir ayağın oranı kapandıysa (oynanamaz) ya da ayak eksikse kupon gösterilmez. Aynı kupon (aynı ayaklar) daha önce
    gönderildiyse yine gösterilir ama sonuç takibine ikinci kez girmez (seen: gönderilen kuponların imzaları)."""
    from src.tavsiye import CILGIN_N, KUPON_TURLERI
    rows = soon[soon["seviye"].astype(str).str.startswith(icon)]
    t = _playable(rows)
    need = {name.split()[0]: n for name, n, _, _ in KUPON_TURLERI}.get(icon, CILGIN_N)
    if len(t) < len(rows) or (t["tahmin"].str.startswith("İY/MS").sum() if icon == "🎲" else len(t)) < need:
        return []
    sig = icon + "|" + "|".join(sorted(t["maç"].astype(str) + ":" + t["tahmin"].astype(str)))
    if sig not in seen:
        log += t.to_dict("records")
        seen.add(sig)
    total = float(t["odd"].astype(float).prod())
    p = pd.to_numeric(t["p"], errors="coerce") if "p" in t else pd.Series(dtype=float)
    chance = f" · yaklaşık {_kac(1 / p.prod())} kuponda 1 tutar" if len(p) and p.notna().all() and p.prod() > 0 else ""
    return _section(icon, t, now_tr) + ["", f"🧾 <b>{len(t)} maç · toplam oran {_tl(total)}</b>{chance}"]


TIER_SIRA = ("🟢", "🟡", "🔴", "🟣", "⚡", "🔥", "💎")   # Tier'lar (takip için kaydedilen, _select)
EFSANE = ("<i>➜ tahmin · İddaa oranı · 💎 değerli (oran adil fiyatın üstünde) · MBS: İddaa'nın en az maç kuralı. "
          "Tier listeleri ve tüm kupon adayları tam raporda.</i>")


def _select(soon: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sonuç takibine giren Tier tahminleri (2026-10-29'dan beri Telegram'da gösterilmez, tam raporda): her maçın Tier
    1-4 gol tahminleri (seviye başına tek), en düşük riskli KORNER_MAC korner tahmini, bir kesin skor (Tier 5 kısa
    listedeyse o, yoksa Tier 6); ayrıca 💎'ler. Sadece oynanabilenler."""
    play = _playable(soon)
    icon = play["seviye"].astype(str).str[:1]
    rank = icon.map({k: i for i, k in enumerate(TIER_SIRA)})
    tiers = play[icon.isin(TIER_SIRA[:4])]
    corner = tiers["tahmin"].map(korner.is_corner_bet).astype(bool)  # boşken de bool maske olsun
    # _r: boş tabloya bütün seri atanırsa pandas serinin satırlarını (boş) ekler; sadece seçilen satırların sırası
    kor = tiers[corner].assign(_r=rank[corner.index[corner]]).sort_values("_r", kind="stable")
    kor = kor.groupby("maç").head(KORNER_MAC)
    sc = icon.isin(("⚡", "🔥"))
    score = play[sc].assign(_r=rank[sc]).sort_values("_r", kind="stable").drop_duplicates("maç")
    rows = pd.concat([tiers[~corner], kor.drop(columns="_r"), score.drop(columns="_r")], ignore_index=True)
    return rows, play[icon == "💎"]


def coupon_lines(label: str, picks: pd.DataFrame, sent: pd.DataFrame) -> list[str]:
    """Bir maçın kupon satırları (kadro mesajı, analiz): önceden gönderilen kuponlardaki ayağı (sent; gönderildiği oran
    -> son oran) ve kupona uygun en değerli ADAY_SAYISI ayak (📌 kupon adayları, tavsiye.kupon_adaylari sırasıyla)."""
    from src.tavsiye import ADAY_TIER, KUPON_MIN_ORAN
    mine = picks[picks["maç"] == label] if len(picks) else picks
    now = {} if mine.empty else dict(zip(mine["tahmin"], mine["odd"].astype(float)))
    out = []
    old = sent[sent["maç"] == label].drop_duplicates(["seviye", "tahmin"]) if len(sent) else sent
    for r in old.to_dict("records"):
        cur, odd = now.get(r["tahmin"]), float(r["odd"])
        txt = f"{odd:.2f}" + (f" → şimdi {cur:.2f}" if pd.notna(cur) and abs(cur - odd) > 1e-9 else "")
        out.append(f"🧾 {html.escape(str(r['seviye']).split(' ', 1)[-1], quote=False)}daki ayak: "
                   f"{html.escape(str(r['tahmin']))} · {txt}")
    cand = mine[mine["seviye"].astype(str).str.startswith(ADAY_TIER.split()[0])].head(ADAY_SAYISI) if len(mine) else mine
    if cand.empty:
        return out + [f"📌 Kupona uygun ayak yok (oranı {KUPON_MIN_ORAN:.2f}+, değeri yeterli)"]
    out.append("📌 Kupona uygun ayaklar (son oranlarla):")
    for r in cand.to_dict("records"):
        value = r.get("gerçek_değer") is True or str(r.get("gerçek_değer")) == "True"
        out.append(f"   ➜ {html.escape(str(r['tahmin']))} · <b>{float(r['odd']):.2f}</b>{' 💎' if value else ''}")
    return out


def tracked(soon: pd.DataFrame) -> list:
    """Sonuç takibine kaydedilen Tier ve 💎 satırları (Telegram'da gösterilmeseler de: _select)."""
    rows, v = _select(soon)
    return rows.to_dict("records") + v.to_dict("records")


def digest_parts(soon: pd.DataFrame, now_tr: datetime, header: str, log: list, seen: set | None = None,
                 bos: str | None = None) -> list:
    """Kupon mesajı (kullanıcı isteği, 2026-10-29: "tier tier değil kupon önersin, düşük oran istemiyorum"): 2'li, 3'lü,
    4'lü ve 🎲 çılgın kupon, altında kuponlarda olmayan 💎'ler. Tier listeleri tam raporda; sonuç takibi için yine
    kaydedilir (tracked). bos: kupon da 💎 de yoksa yazılacak not. Uzun mesaj Telegram sınırında satır sonundan
    bölünür (send)."""
    from src.tavsiye import KUPON_IKON, KUPON_MIN_ORAN
    log += tracked(soon)
    seen = set() if seen is None else seen
    icons = (*KUPON_IKON, "🎲")
    body = [x for icon in icons for x in _coupon_lines(soon, now_tr, log, icon, seen)]
    legs = soon[soon["seviye"].astype(str).str.startswith(icons)]
    in_coupon = set(zip(legs["maç"], legs["tahmin"]))
    v = _playable(soon[soon["seviye"].astype(str).str.startswith("💎")])
    v = v[[(m, b) not in in_coupon for m, b in zip(v["maç"], v["tahmin"])]] if len(v) else v  # kupondaki 💎 bir kez
    if len(v):
        body += _section("💎", v, now_tr)
    if not body:
        body = ["", bos or f"Bugün ve yarın kupon kuracak kadar uygun ayak (oranı {KUPON_MIN_ORAN:.2f}+) yok."]
    intro = [header, "<i>Her kuponda maçların hepsi tutmalı · ayaklar farklı maçlardan</i>"]
    tail = ["", CIZGI, EFSANE, f'<a href="{RAPOR_LINKI}">Tam rapor</a>']
    return ["\n".join(intro + body + tail)]


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
        title = "Güncel kuponlar" if force else OZET_SAATLERI[max(passed)]
        stamp = f"{now_tr:%d.%m %H:%M}" if force else f"{now_tr:%d.%m}"
        old = list(state.get("kuponlar", []))
        seen = set(old)
        msgs += digest_parts(soon, now_tr, f"⚽ <b>{title} — {stamp}</b>", log, seen)
        state["kuponlar"] = (old + sorted(seen - set(old)))[-100:]  # aynı kupon yine gelirse takibe iki kez girmez
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
