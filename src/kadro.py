"""Kadro açıklanınca güncelleme (kullanıcı isteği, 2026-10-24).

Maçtan ~1 saat önce ilk 11'ler açıklanınca (ESPN'in skor servisi: maç özetindeki kadrolar) o maçların Tier listesi son
İddaa oranlarıyla yeniden hesaplanır ve Telegram'a tek mesaj gider: dizilişler, son resmi maçlarında düzenli ilk 11'e
çıkıp bu maçta olmayanlar (yedekte mi), sabahki listeye göre değişen tahminler ve güncel Tier listesi.

Kadro oyuncu bazında modele girmez (oyuncu verimiz yok): kadro haberini piyasa fiyatlar; İddaa oranı değişince
olasılıklar (milli ve Avrupa maçlarında doğrudan İddaa'ya, kulüplerde keskin oran yoksa İddaa'ya ayarlı) ve liste de
değişir. Eksik oyuncu listesi bilgi içindir.

GitHub'da 15 dakikada bir (.github/workflows/kadro.yml): başlamasına PENCERE dakika kalan, mesajı gitmemiş maçlar
kontrol edilir; kadro açıklanmışsa güncelleme gönderilir. Gönderilen maçlar tavsiyeler/kadro_durum.json'da.
Mesajdaki tahminler sonuç takibine ilk verilenlerden ayrı (mesaj="kadro") girer: sonuclar.kadro_compare kadro öncesi /
sonrası isabeti ve kârı karşılaştırır.
"""
import html
import json
import time
from collections import Counter

import pandas as pd
import requests

import config
from src import kulup_sonuc

SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary"
PENCERE = (15, 90)    # başlamasına bu kadar dakika kalan maçlar (kadrolar genelde ~60 dakika önce açıklanır)
DUZENLI = 3           # son 5 resmi maçın en az bu kadarında ilk 11'de: düzenli oyuncu
SEVIYE = ("🟢", "🟡", "🔴", "🟣", "⚡", "🔥")   # değişiklik karşılaştırmasına giren seviyeler


def state_path():
    return config.TAVSIYE_DIR / "kadro_durum.json"


def _state() -> dict:
    p = state_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _get(url: str, **params) -> dict:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(0.3)  # siteye nazik davran
    return r.json()


def summary(event_id: str) -> dict:
    """ESPN maç özeti (lig fark etmez: "all")."""
    return _get(SUMMARY_URL.format(league="all"), event=event_id)


def candidates(now_utc: pd.Timestamp | None = None, pencere: tuple = PENCERE) -> pd.DataFrame:
    """Başlamasına pencere (dakika) kadar kalan, kadro mesajı gönderilmemiş maçlar: İddaa oran kayıtlarındaki son
    satırlar (veritabanı gerekmez: kayıttaki eşlenmiş adlarla)."""
    files = sorted((config.RAW_DIR / "iddaa").glob("odds_*.csv"))
    if not files:
        return pd.DataFrame()
    cols = ["snapshot_utc", "event_id", "div", "kickoff_utc", "home_iddaa", "away_iddaa", "home", "away"]
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in files], ignore_index=True)
    last = df.sort_values("snapshot_utc").groupby("event_id").tail(1).dropna(subset=["home", "away"])
    now = now_utc if now_utc is not None else pd.Timestamp.now("UTC").tz_localize(None)
    mins = (pd.to_datetime(last["kickoff_utc"]) - now).dt.total_seconds() / 60
    last = last[(mins >= pencere[0]) & (mins <= pencere[1])]
    sent = set(map(str, _state().get("gonderilen", [])))
    return last[~last["event_id"].astype(str).isin(sent)].assign(kickoff_utc=lambda d: pd.to_datetime(d["kickoff_utc"]))


def wait_seconds(now_utc: pd.Timestamp | None = None, horizon_h: float = 8) -> int:
    """GitHub'daki uzun işin döngüsü için: kontrol penceresine (PENCERE) girmiş maç varsa 0; yoksa önümüzdeki horizon_h
    saatte ilk maçın pencereye girmesine kalan saniye; hiç maç yoksa -1 (iş kapanabilir)."""
    now = now_utc if now_utc is not None else pd.Timestamp.now("UTC").tz_localize(None)
    c = candidates(now, (PENCERE[0], horizon_h * 60))
    if c.empty:
        return -1
    to_window = (c["kickoff_utc"].min() - pd.Timedelta(minutes=PENCERE[1]) - now).total_seconds()
    return max(0, int(to_window))


def _codes(div: str) -> list:
    """Lig kodu -> ESPN turnuva kodları."""
    from src import milli
    if div == config.MILLI:
        return list(milli.ESPN_LEAGUES)
    c = config.ESPN_KULUP.get(div)
    return [] if c is None else [c] if isinstance(c, str) else list(c)


def _scheduled(data: dict) -> list[dict]:
    out = []
    for e in data.get("events", []):
        side = {c.get("homeAway"): c for c in e["competitions"][0].get("competitors", [])}
        if set(side) == {"home", "away"}:
            out.append({"id": e["id"], "date": pd.Timestamp(e["date"]).tz_localize(None),
                        "home_espn": side["home"]["team"]["displayName"],
                        "away_espn": side["away"]["team"]["displayName"]})
    return out


def find_events(cand: pd.DataFrame) -> dict:
    """Aday maçların ESPN karşılıkları: event_id -> ESPN maç no. Ad (football-data / İddaa adı) ve saat (±3 saat)."""
    boards, out = {}, {}
    for r in cand.to_dict("records"):
        ko = pd.Timestamp(r["kickoff_utc"])
        days = {ko.strftime("%Y%m%d"), (ko + pd.Timedelta(hours=3)).strftime("%Y%m%d")}  # UTC ve TR günü
        events = []
        for code in _codes(r["div"]):
            for d in days:
                if (code, d) not in boards:
                    try:
                        boards[code, d] = _scheduled(_get(config.ESPN_URL.format(league=code), dates=d))
                    except Exception:  # bir turnuva / gün alınamazsa diğerleri
                        boards[code, d] = []
                events += boards[code, d]
        near = [e for e in events if abs((e["date"] - ko).total_seconds()) <= 3 * 3600]
        e = kulup_sonuc.match(near, [r["home"], r["home_iddaa"]], [r["away"], r["away_iddaa"]])
        if e:
            out[r["event_id"]] = e["id"]
    return out


def lineup(espn_id: str, fetch=summary, regulars: bool = True) -> dict | None:
    """Açıklanmış kadro: {"home"/"away": ad, diziliş, ilk 11, yedekler, eksik düzenliler}; açıklanmadıysa None.
    Düzenli: son 5 resmi maçın (hazırlık hariç) en az DUZENLI'sinde ilk 11 (en az 3 maç varsa)."""
    s = fetch(espn_id)
    teams = {}
    for t in s.get("rosters", []):
        roster = t.get("roster", [])
        teams[t.get("homeAway")] = {
            "id": str(t.get("team", {}).get("id")), "ad": t.get("team", {}).get("displayName", ""),
            "dizilis": t.get("formation"),
            "ilk11": {str(p["athlete"]["id"]): p["athlete"]["displayName"] for p in roster if p.get("starter")},
            "yedek": {str(p["athlete"]["id"]) for p in roster if not p.get("starter")}, "eksik": None}
    if not all(len(teams.get(k, {}).get("ilk11", {})) >= 11 for k in ("home", "away")):
        return None
    if regulars:
        for l5 in s.get("lastFiveGames", []):
            team = next((t for t in teams.values() if t["id"] == str(l5.get("team", {}).get("id"))), None)
            if team is None:
                continue
            starts, names, n = Counter(), {}, 0
            for e in l5.get("events", []):
                if "friendly" in str(e.get("leagueAbbreviation", "")).lower():
                    continue
                try:
                    past = fetch(e["id"])
                except Exception:
                    continue
                r = next((x for x in past.get("rosters", []) if str(x.get("team", {}).get("id")) == team["id"]), None)
                xi = [p for p in (r or {}).get("roster", []) if p.get("starter")]
                if len(xi) >= 11:
                    n += 1
                    starts.update(str(p["athlete"]["id"]) for p in xi)
                    names.update({str(p["athlete"]["id"]): p["athlete"]["displayName"] for p in xi})
            if n >= 3:
                team["eksik"] = [(names[i], i in team["yedek"]) for i, k in starts.most_common()
                                 if k >= min(DUZENLI, n) and i not in team["ilk11"]]
    return teams


def lineup_lines(label: str, lu: dict) -> list[str]:
    """Mesajda bir maçın kadro satırları."""
    h, a = lu["home"], lu["away"]
    form = f" · {h['dizilis']} / {a['dizilis']}" if h.get("dizilis") and a.get("dizilis") else ""
    out = [f"🏟 <b>{html.escape(label)}</b>{form}"]
    for t in (h, a):
        if t["eksik"] is None:
            continue
        if t["eksik"]:
            names = ", ".join(html.escape(n) + (" (yedekte)" if bench else "") for n, bench in t["eksik"])
            out.append(f"• {html.escape(t['ad'])}: düzenli ilk 11'den yok: {names}")
        else:
            out.append(f"• {html.escape(t['ad'])}: düzenli ilk 11 tam")
    return out


def changes(old: pd.DataFrame, new_rows: list) -> list[str]:
    """Önceden gönderilen tahminlere göre değişenler: aynı maçta aynı seviyede önceden verilen tahmin şimdi farklıysa
    "🔄 eski → yeni", kalktıysa "➖". Önceden o seviyede tahmin olmayan yeni satırlar sayılmaz (sabah mesajında her
    maçın sadece bir kısmı vardı). Kesin skor tek grup (⚽): kısa listeye girip çıkması (Tier 5 / 6) değişiklik değil."""
    def groups(rows) -> dict:
        out = {}
        for r in rows:
            icon = str(r["seviye"])[:1]
            if icon in SEVIYE:
                out.setdefault((r["maç"], "⚽" if icon in ("⚡", "🔥") else icon), set()).add(r["tahmin"])
        return out
    before, after = groups(old.to_dict("records")), groups(new_rows)
    out = []
    for (maç, icon), bets in sorted(before.items()):
        now = after.get((maç, icon), set())
        gone, new = sorted(bets - now), sorted(now - bets)
        name = f"{icon} {html.escape(maç)}"
        if gone and new:
            out.append(f"🔄 {name}: {html.escape(', '.join(gone))} → {html.escape(', '.join(new))}")
        elif gone:
            out.append(f"➖ {name}: {html.escape(', '.join(gone))} (artık listede yok)")
    return out


def run(send: bool = True, now_utc: pd.Timestamp | None = None) -> dict:
    """Kadrosu açıklanan aday maçlar için güncelleme mesajları. Döner: {"satirlar": sonuç takibine girecek tahminler,
    "maclar": mesajı giden İddaa maç numaraları, "zaman": gönderim zamanı (TR)} (kayıt: kaydet)."""
    from src import sonuclar, tavsiye, telegram
    out = {"satirlar": [], "maclar": [], "zaman": f"{telegram._tr_now():%Y-%m-%d %H:%M}"}
    cand = candidates(now_utc)
    if cand.empty:
        print(f"  kadro: başlamasına {PENCERE[0]}-{PENCERE[1]} dakika kalan yeni maç yok")
        return out
    espn = find_events(cand)
    ready = []
    for r in cand.to_dict("records"):
        lu = lineup(espn[r["event_id"]]) if r["event_id"] in espn else None
        if lu:
            ready.append((r, lu))
    print(f"  kadro: {len(cand)} aday maç, ESPN'de {len(espn)}, kadrosu açıklanan {len(ready)}")
    if not ready:
        return out
    log = sonuclar.load_log()
    now_tr = telegram._tr_now()
    slots = {}
    for r, lu in ready:
        slots.setdefault(pd.Timestamp(r["kickoff_utc"]) + pd.Timedelta(hours=3), []).append((r, lu))
    for ko, items in sorted(slots.items()):
        _, picks = tavsiye.run(config.RAW_DIR / "fixtures.csv", saat=f"{ko:%H:%M}", gun=f"{ko:%Y-%m-%d}")
        labels = [tavsiye._match_label(r) for r, _ in items]
        if not picks.empty:  # ideal / çılgın kupon günlük özette; her kadro mesajında yeni kupon açılmasın
            picks = picks[picks["maç"].isin(labels) & ~picks["seviye"].str.startswith(("🎰", "🎯", "🎲"))]
        head = [f"📋 <b>Kadrolar açıklandı · {ko:%d.%m %H:%M}</b>", ""]
        for label, (_, lu) in zip(labels, items):
            head += lineup_lines(label, lu) + [""]
        rows = []
        if not picks.empty:
            telegram.digest_parts(picks, now_tr, "", rows)  # önce yeni liste (değişiklikleri bulmak için)
        days = {pd.Timestamp(r["kickoff_utc"]).strftime("%Y-%m-%d") for r, _ in items}  # kayıttaki gün: UTC tarihi
        old = (log[log["maç"].isin(labels) & log["gün"].isin(days) & ~sonuclar._kadro(log)]  # ilk verilenler
               if len(log) else log)
        diff = changes(old, rows) if len(old) else []
        head += ["🔄 <b>Sabahki listeye göre</b>"] + (diff or ["Değişiklik yok"]) if len(old) else []
        rows = []
        if picks.empty:
            msgs = ["\n".join(head + ["", "Bu maçlar için tavsiye çıkmadı (oran yok ya da kapalı)."])]
        else:
            msgs = telegram.digest_parts(picks, now_tr, "\n".join(head).rstrip(), rows)
        if not send:
            print("\n\n".join(msgs))
        elif not all([telegram.send(m) for m in msgs]):
            continue  # gönderilemedi: bir sonraki çalıştırmada tekrar denenir
        # mesajdaki tüm tahminler (aynı kalanlar da) "kadro" olarak ayrı kaydedilir: kadro öncesi / sonrası karşılaştırma
        out["satirlar"] += [{**r, "mesaj": "kadro"} for r in rows]
        out["maclar"] += [str(r["event_id"]) for r, _ in items]
    return out


def kaydet(res: dict) -> int:
    """Gönderilen güncellemenin tahminlerini sonuç takibine, maçları kadro_durum.json'a yazar. Döner: yeni kayıt."""
    from src import sonuclar
    n = sonuclar.record(res["satirlar"], pd.Timestamp(res["zaman"])) if res["satirlar"] else 0
    if res["maclar"]:
        st = _state()
        st["gonderilen"] = (st.get("gonderilen", []) + res["maclar"])[-500:]
        config.TAVSIYE_DIR.mkdir(parents=True, exist_ok=True)
        state_path().write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return n


def check(now_utc: pd.Timestamp | None = None) -> bool:
    """GitHub işinin ucuz ön kontrolü: kadrosu açıklanmış aday maç var mı (eksik oyuncu hesabı yok)."""
    cand = candidates(now_utc)
    if cand.empty:
        print(f"  kadro: başlamasına {PENCERE[0]}-{PENCERE[1]} dakika kalan yeni maç yok")
        return False
    espn = find_events(cand)
    ready = sum(lineup(e, regulars=False) is not None for e in espn.values())
    print(f"  kadro: {len(cand)} aday maç, ESPN'de {len(espn)}, kadrosu açıklanan {ready}")
    return ready > 0
