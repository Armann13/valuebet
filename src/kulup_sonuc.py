"""Kulüp maçlarının sonuçları ESPN'den: sadece tavsiye takibi için (sonuclar.py).

football-data.co.uk sonuçları haftada birkaç kez günceller (hafta sonu maçları pazartesi gelir); ESPN maçtan birkaç
saat sonra verir. Sadece sonucu bekleyen tahminlerin ligi ve günü sorgulanır (config.ESPN_KULUP). Takımlar ESPN
adıyla football-data ya da İddaa adı karşılaştırılarak eşlenir (aynı lig, aynı gün: aday sayısı az, iki takım da
tutmalı). Model eğitimi football-data ile kalır; burada sadece sonuç kaydı (data/raw/kulup/espn.csv).
ESPN alt liglerde korner vermeyebilir (0 görünür): şut sayısı da yoksa korner boş bırakılır.
"""
import difflib
import re
import time

import pandas as pd
import requests

import config
from src import korner
from src.teams import _norm

COLS = ["date", "div", "home", "away", "fthg", "ftag", "hthg", "htag", "hc", "ac"]


def path():
    return config.RAW_DIR / "kulup" / "espn.csv"


def load() -> pd.DataFrame:
    p = path()
    if not p.exists():
        return pd.DataFrame(columns=COLS).astype({"date": "datetime64[ns]"})
    return pd.read_csv(p, parse_dates=["date"])


def _tokens(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", _norm(s)).split()


def _same(espn: str, names: list[str]) -> float:
    """ESPN adı bizim adlardan birine ne kadar benziyor (1 = bir adın tüm kelimeleri diğerinde: "Ayr" ~ "Ayr United")."""
    e = _tokens(espn)
    best = 0.0
    for n in names:
        t = _tokens(n)
        if t and e and (set(t) <= set(e) or set(e) <= set(t)):
            return 1.0
        best = max(best, difflib.SequenceMatcher(None, " ".join(e), " ".join(t)).ratio())
    return best


def _events(data: dict) -> list[dict]:
    """ESPN scoreboard'dan biten maçlar: ev / deplasman adı, skor, ilk yarı ve korner (varsa)."""
    out = []
    for e in data.get("events", []):
        if not e.get("status", {}).get("type", {}).get("completed"):
            continue
        comp = e["competitions"][0]
        side = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if set(side) != {"home", "away"}:
            continue
        st = {k: {s.get("name"): pd.to_numeric(s.get("displayValue"), errors="coerce")
                  for s in c.get("statistics", [])} for k, c in side.items()}
        try:
            score = {k: int(float(c["score"])) for k, c in side.items()}
        except (KeyError, TypeError, ValueError):
            continue
        try:
            ht = {k: int(float(c["linescores"][0]["value"])) for k, c in side.items()}
        except (KeyError, IndexError, TypeError, ValueError):
            ht = {"home": None, "away": None}
        corners = {k: st[k].get("wonCorners") for k in side}
        shots = sum(float(st[k].get("totalShots") or 0) for k in side)
        if any(pd.isna(v) for v in corners.values()) or (sum(corners.values()) == 0 and shots == 0):
            corners = {"home": None, "away": None}  # istatistik yok (alt liglerde 0 görünür)
        out.append({"home_espn": side["home"]["team"]["displayName"], "away_espn": side["away"]["team"]["displayName"],
                    "fthg": score["home"], "ftag": score["away"], "hthg": ht["home"], "htag": ht["away"],
                    "hc": corners["home"], "ac": corners["away"]})
    return out


def match(events: list[dict], home: list[str], away: list[str], cutoff: float = 0.75) -> dict | None:
    """Bir maçın ESPN karşılığı: iki takım da benzemeli; en iyi aday tek olmalı (eşitlikte karar verilmez)."""
    scored = sorted(((min(_same(e["home_espn"], home), _same(e["away_espn"], away)), i) for i, e in enumerate(events)),
                    reverse=True)
    if not scored or scored[0][0] < cutoff or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return events[scored[0][1]]


def download(log: pd.DataFrame, today: pd.Timestamp | None = None) -> int:
    """Sonucu bekleyen kulüp tahminlerinin maçlarını ESPN'den çekip kaydeder. Döner: yeni eklenen maç sayısı."""
    today = (today or pd.Timestamp.now()).normalize()
    log = log.reindex(columns=list(dict.fromkeys([*log.columns, "hg", "hk"])))
    corner = log["tahmin"].map(korner.is_corner_bet).astype(bool)
    need = log[log["div"].isin(config.ESPN_KULUP) & (log["hg"].isna() | (corner & log["hk"].isna()))
               & (pd.to_datetime(log["gün"]) <= today)]
    if need.empty:
        return 0
    old = load()
    have = set(zip(old["date"].dt.strftime("%Y-%m-%d"), old["div"], old["home"], old["away"]))
    rows, failed = [], 0
    for (div, day), g in need.groupby(["div", "gün"]):
        games = g.drop_duplicates(["home", "away"])
        if all((day, div, h, a) in have for h, a in zip(games["home"], games["away"])):
            continue
        try:
            r = requests.get(config.ESPN_URL.format(league=config.ESPN_KULUP[div]),
                             params={"dates": day.replace("-", "")}, timeout=30)
            r.raise_for_status()
            events = _events(r.json())
        except Exception:  # bir lig / gün alınamazsa diğerleri devam etsin
            failed += 1
            continue
        for _, x in games.iterrows():
            ih, _, ia = str(x["maç"]).partition(" – ")  # İddaa adları (football-data adından farklı olabilir)
            e = match(events, [x["home"], ih], [x["away"], ia])
            if e:
                rows.append({"date": day, "div": div, "home": x["home"], "away": x["away"],
                             **{k: e[k] for k in COLS[4:]}})
        time.sleep(0.3)  # siteye nazik davran
    if rows:
        new = pd.concat([old, pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d["date"]))], ignore_index=True)
        new = new.drop_duplicates(["date", "div", "home", "away"], keep="last").sort_values("date")
        path().parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path(), index=False, date_format="%Y-%m-%d")
    print(f"  + kulüp sonuçları (ESPN): {len(rows)} maç" + (f", {failed} sorgu başarısız" if failed else ""))
    return len(rows)
