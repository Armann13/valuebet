"""İddaa oranlarının otomatik toplanması (config.LEAGUES'teki tüm ligler).

iddaa.com'un sitesinin kullandığı JSON uç noktasından (resmi API değil) maç öncesi futbol programı çekilir:
  maç sonucu (market t=1, st=1; sonuçlar "1" / "0" / "2") ve alt/üst (t=2, st=101; baraj "sov").
Her sonuç için iki oran vardır: odd (ekranda gösterilen, normal kupon) ve wodd (sitenin kodunda "king odds";
~%4 düşük). Hesaplarda odd kullanılır; ikisi de kaydedilir. mbs: maçın minimum kombinasyon sayısı.

Kayıtlar data/raw/iddaa/odds_{sezon}.csv'ye eklenir; bir maçın oranları son kayıttan beri değişmediyse
yeni satır yazılmaz. Böylece her maçın oran geçmişi (ilk görülen, son görülen) tutulur.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (valuebet research script)", "Origin": "https://www.iddaa.com",
           "Referer": "https://www.iddaa.com/"}
ODD_COLS = ["odd_1", "odd_x", "odd_2", "ou_line", "odd_alt", "odd_ust"]
WODD_COLS = ["wodd_1", "wodd_x", "wodd_2", "wodd_alt", "wodd_ust"]
EVENT_KEY = "event_id"


def _get(path: str):
    r = requests.get(config.IDDAA_API + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("isSuccess", False):
        raise RuntimeError(f"iddaa yanıtı başarısız: {j.get('message')}")
    return j["data"]


def resolve_competitions(comps: list) -> dict:
    """iddaa lig no -> bizim lig kodumuz. config.IDDAA_LEAGUE_NAMES ile: ad tam aynı ya da "ad," ile başlıyor
    (sezon aşaması eki, ör. "İskoçya Premiership, Şampiyona Turu"). "... Beşiktaş Özel" gibi özel marketler dışarıda."""
    out = {}
    for c in comps:
        if str(c.get("si")) != "1":  # futbol
            continue
        name = str(c.get("n", "")).strip()
        for div, base in config.IDDAA_LEAGUE_NAMES.items():
            if name == base or name.startswith(base + ","):
                out[c["i"]] = div
    return out


def parse_events(events: list, competitions: dict) -> pd.DataFrame:
    """events: /sportsbook/events yanıtındaki liste. competitions: iddaa lig no -> bizim lig kodumuz."""
    rows = []
    for e in events:
        div = competitions.get(e.get("ci"))
        if div is None:
            continue
        row = {"event_id": e["i"], "div": div, "kickoff_utc": datetime.fromtimestamp(e["d"], timezone.utc)
               .strftime("%Y-%m-%d %H:%M"), "home_iddaa": e["hn"], "away_iddaa": e["an"], "mbs": e.get("mbc")}
        for m in e.get("m", []):
            o = {x["n"]: x for x in m.get("o", [])}
            if m.get("t") == 1 and m.get("st") == 1:  # maç sonucu
                for lab, col in (("1", "1"), ("0", "x"), ("2", "2")):
                    if lab in o:
                        row[f"odd_{col}"], row[f"wodd_{col}"] = o[lab].get("odd"), o[lab].get("wodd")
            elif m.get("t") == 2 and m.get("st") == 101:  # alt / üst (ana baraj)
                row["ou_line"] = float(m["sov"]) if m.get("sov") not in (None, "") else np.nan
                for lab, col in (("Alt", "alt"), ("Üst", "ust")):
                    if lab in o:
                        row[f"odd_{col}"], row[f"wodd_{col}"] = o[lab].get("odd"), o[lab].get("wodd")
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ODD_COLS + WODD_COLS:
        if c not in df:
            df[c] = np.nan
    return df


def _map_teams(df: pd.DataFrame, report: bool = False) -> tuple[pd.DataFrame, set]:
    from src import db
    from src.teams import TeamResolver
    known = db.load_matches()
    unknown = set()
    for div in df["div"].unique():
        resolve = TeamResolver(known.loc[known["div"] == div, "home"].unique())
        if report:
            df.attrs.setdefault("resolvers", []).append(resolve)
        for side in ("home", "away"):
            sel = df["div"] == div
            names = {n: resolve(n) for n in df.loc[sel, f"{side}_iddaa"].dropna().unique()}  # boş ad: eşleme yok
            unknown |= {n for n, v in names.items() if v is None}
            df.loc[sel, side] = df.loc[sel, f"{side}_iddaa"].map(names)
    return df, unknown


def _season(kickoff: pd.Series) -> pd.Series:
    t = pd.to_datetime(kickoff)
    return np.where(t.dt.month >= 7, t.dt.year, t.dt.year - 1)


def collect() -> int:
    """Anlık oranları çeker ve değişenleri data/raw/iddaa/odds_{sezon}.csv'ye ekler. Döner: yeni satır sayısı."""
    comps = resolve_competitions(_get("competitions"))
    events = _get("events?st=1&type=0&version=0")["events"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    df = parse_events(events, comps)
    if df.empty:
        print(f"  iddaa: takip edilen liglerde maç yok ({len(comps)} lig eşleşti)")
        return 0
    df, unknown = _map_teams(df, report=True)
    if unknown:
        print(f"  ! iddaa'da eşleşmeyen takım adları (team_aliases.csv'ye ekle): {', '.join(sorted(unknown))}")
    fuzzy = {k: v for r in df.attrs.pop("resolvers", []) for k, v in r.fuzzy.items()}
    if fuzzy:  # yanlış eşleşme riski: gözden geçir, yanlışsa team_aliases.csv ile düzelt
        print("  ? iddaa bulanık eşleşmeler: " + ", ".join(f"{k} -> {v}" for k, v in sorted(fuzzy.items())))
    df.insert(0, "snapshot_utc", now)
    df["season"] = _season(df["kickoff_utc"])
    folder = config.RAW_DIR / "iddaa"
    folder.mkdir(parents=True, exist_ok=True)
    added = 0
    for season, g in df.groupby("season"):
        path = folder / f"odds_{season}.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=g.columns)
        last = old.sort_values("snapshot_utc").groupby(EVENT_KEY).tail(1).set_index(EVENT_KEY) if len(old) else None
        keep = []
        for r in g.itertuples(index=False):
            if last is not None and r.event_id in last.index:
                prev = last.loc[r.event_id, ODD_COLS].astype(float).to_numpy()
                cur = np.array([getattr(r, c) for c in ODD_COLS], dtype=float)
                if np.allclose(prev, cur, equal_nan=True):
                    continue
            keep.append(r._asdict())
        if keep:
            pd.concat([old, pd.DataFrame(keep)], ignore_index=True).to_csv(path, index=False)
            added += len(keep)
    per = ", ".join(f"{d} {n}" for d, n in df["div"].value_counts().sort_index().items())
    print(f"  iddaa: {len(df)} maç görüldü ({per}), {added} yeni / değişen oran kaydı")
    return added


def load_odds() -> pd.DataFrame:
    """Tüm oran kayıtları (her maçın oran geçmişi)."""
    files = sorted((config.RAW_DIR / "iddaa").glob("odds_*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "home_iddaa" in df:  # ham adlardan yeniden eşle: sonradan eklenen eşleştirmeler eski kayıtlara da uygulanır
        mapped, _ = _map_teams(df.copy())
        for side in ("home", "away"):  # yeniden eşleme başarısızsa kayıtlı eşleşme kalır
            df[side] = mapped[side].where(mapped[side].notna(), df[side] if side in df else None)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"])
    df["snapshot_utc"] = pd.to_datetime(df["snapshot_utc"])
    return df[df["snapshot_utc"] < df["kickoff_utc"]]  # sadece maç başlamadan önceki oranlar
