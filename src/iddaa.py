"""İddaa oranlarının otomatik toplanması (config.LEAGUES'teki tüm ligler).

iddaa.com'un sitesinin kullandığı JSON uç noktasından (resmi API değil) maç öncesi futbol programı çekilir:
  maç sonucu (market t=1, st=1; sonuçlar "1" / "0" / "2"), alt/üst (t=2, st=101; baraj "sov"; bir maçta birden
  çok baraj olabilir, 2.5 varsa o alınır),
  çifte şans (t=2, st=92; "1 ve 0" / "1 ve 2" / "0 ve 2") ve karşılıklı gol (t=2, st=89; "Var" / "Yok").
Her sonuç için iki oran vardır: odd (ekranda gösterilen, normal kupon) ve wodd (sitenin kodunda "king odds";
~%4 düşük). Hesaplarda odd kullanılır; ikisi de kaydedilir. mbs: maçın minimum kombinasyon sayısı.

Kayıtlar data/raw/iddaa/odds_{sezon}.csv'ye eklenir; bir maçın oranları son kayıttan beri değişmediyse
yeni satır yazılmaz. Böylece her maçın oran geçmişi (ilk görülen, son görülen) tutulur.
"""
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (valuebet research script)", "Origin": "https://www.iddaa.com",
           "Referer": "https://www.iddaa.com/"}
ODD_COLS = ["odd_1", "odd_x", "odd_2", "ou_line", "odd_alt", "odd_ust",
            "odd_1x", "odd_12", "odd_x2", "odd_kg_var", "odd_kg_yok"]
WODD_COLS = ["wodd_1", "wodd_x", "wodd_2", "wodd_alt", "wodd_ust"]
EVENT_KEY = "event_id"


def _get(path: str):
    r = requests.get(config.IDDAA_API + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    j = r.json()
    if not j.get("isSuccess", False):
        raise RuntimeError(f"iddaa yanıtı başarısız: {j.get('message')}")
    return j["data"]


def _odd(x) -> float:
    """İddaa oranı; 1.00 ve altı "oynanamaz" (kapalı sonuç) demektir, boş sayılır."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if v > 1.0 else np.nan


def resolve_competitions(comps: list) -> dict:
    """iddaa lig no -> bizim lig kodumuz. config.IDDAA_LEAGUE_NAMES ile: ad tam aynı ya da "ad," ile başlıyor
    (sezon aşaması eki, ör. "İskoçya Premiership, Şampiyona Turu"). "... Beşiktaş Özel" gibi özel marketler dışarıda.
    Milli takım turnuvaları (config.IDDAA_MILLI_KEYWORDS; gençler / kadınlar / kulüpler hariç) -> config.MILLI."""
    out = {}
    for c in comps:
        if str(c.get("si")) != "1":  # futbol
            continue
        name = str(c.get("n", "")).strip()
        for div, base in config.IDDAA_LEAGUE_NAMES.items():
            if name == base or name.startswith(base + ","):
                out[c["i"]] = div
        if (c["i"] not in out and any(k in name for k in config.IDDAA_MILLI_KEYWORDS)
                and not re.search(config.IDDAA_MILLI_EXCLUDE, name)):
            out[c["i"]] = config.MILLI
    return out


def parse_events(events: list, competitions: dict, names: dict | None = None) -> pd.DataFrame:
    """events: /sportsbook/events yanıtındaki liste. competitions: iddaa lig no -> bizim lig kodumuz.
    names: iddaa lig no -> iddaa'daki turnuva adı (kaydedilir; milli maçlarda raporda gösterilir)."""
    rows = []
    for e in events:
        div = competitions.get(e.get("ci"))
        if div is None:
            continue
        row = {"event_id": e["i"], "div": div, "kickoff_utc": datetime.fromtimestamp(e["d"], timezone.utc)
               .strftime("%Y-%m-%d %H:%M"), "home_iddaa": e["hn"], "away_iddaa": e["an"], "mbs": e.get("mbc"),
               "competition": (names or {}).get(e.get("ci"))}
        for m in e.get("m", []):
            o = {x["n"]: {k: _odd(x.get(k)) for k in ("odd", "wodd")} for x in m.get("o", [])}
            if m.get("t") == 1 and m.get("st") == 1:  # maç sonucu
                for lab, col in (("1", "1"), ("0", "x"), ("2", "2")):
                    if lab in o:
                        row[f"odd_{col}"], row[f"wodd_{col}"] = o[lab]["odd"], o[lab]["wodd"]
            elif m.get("t") == 2 and m.get("st") == 101:  # alt / üst: birden çok baraj olabilir (1.5, 2.5, 3.5)
                line = float(m["sov"]) if m.get("sov") not in (None, "") else np.nan
                if "ou_line" in row and not (line == 2.5 and row["ou_line"] != 2.5):
                    continue  # 2.5 varsa o, yoksa ilk gelen baraj
                row["ou_line"] = line
                for lab, col in (("Alt", "alt"), ("Üst", "ust")):
                    if lab in o:
                        row[f"odd_{col}"], row[f"wodd_{col}"] = o[lab]["odd"], o[lab]["wodd"]
            elif m.get("t") == 2 and m.get("st") == 92:  # çifte şans
                for lab, col in (("1 ve 0", "1x"), ("1 ve 2", "12"), ("0 ve 2", "x2")):
                    if lab in o:
                        row[f"odd_{col}"] = o[lab]["odd"]
            elif m.get("t") == 2 and m.get("st") == 89:  # karşılıklı gol
                for lab, col in (("Var", "kg_var"), ("Yok", "kg_yok")):
                    if lab in o:
                        row[f"odd_{col}"] = o[lab]["odd"]
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ODD_COLS + WODD_COLS:
        if c not in df:
            df[c] = np.nan
    return df


def _map_teams(df: pd.DataFrame, report: bool = False) -> tuple[pd.DataFrame, set]:
    from src import db, milli
    from src.teams import TeamResolver
    known = db.load_matches()
    unknown = set()
    for div in df["div"].unique():
        if div == config.MILLI:  # Türkçe ülke adı -> veri setindeki ad; bulanık eşleşme yok
            resolve = TeamResolver(sorted(milli.TEAMS), aliases=milli.ULKELER, allow_fuzzy=False)
        else:
            resolve = TeamResolver(known.loc[known["div"] == div, "home"].unique())
        if report:
            df.attrs.setdefault("resolvers", []).append(resolve)
        for side in ("home", "away"):
            sel = df["div"] == div
            names = {n: resolve(n) for n in df.loc[sel, f"{side}_iddaa"].dropna().unique()}  # boş ad: eşleme yok
            unknown |= {n for n, v in names.items() if v is None}
            df.loc[sel, side] = df.loc[sel, f"{side}_iddaa"].map(names)
    return df, unknown


def _report_names(df: pd.DataFrame, unknown: set, fuzzy: dict, now: str) -> None:
    """Eşleşmeyen ve bulanık eşleşen adları data/raw/iddaa/eslesme_raporu.csv'de biriktirir (gözden geçirmek için;
    düzeltme team_aliases.csv'ye 'alias,team' satırıyla)."""
    rows = []
    for side in ("home", "away"):
        for div, name in df[["div", f"{side}_iddaa"]].drop_duplicates().itertuples(index=False):
            if name in unknown:
                rows.append({"iddaa_adı": name, "lig": div, "durum": "eşleşmedi", "eşlenen": ""})
            elif name in fuzzy:
                rows.append({"iddaa_adı": name, "lig": div, "durum": "bulanık", "eşlenen": fuzzy[name]})
    if not rows:
        return
    path = config.RAW_DIR / "iddaa" / "eslesme_raporu.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["iddaa_adı", "lig", "durum", "eşlenen", "ilk_görülme"])
    new = pd.DataFrame(rows).drop_duplicates().assign(ilk_görülme=now)
    new = new[~new.set_index(["iddaa_adı", "lig"]).index.isin(old.set_index(["iddaa_adı", "lig"]).index)]
    if len(new):
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)


def _season(kickoff: pd.Series) -> pd.Series:
    t = pd.to_datetime(kickoff)
    return np.where(t.dt.month >= 7, t.dt.year, t.dt.year - 1)


def collect() -> int:
    """Anlık oranları çeker ve değişenleri data/raw/iddaa/odds_{sezon}.csv'ye ekler. Döner: yeni satır sayısı."""
    raw = _get("competitions")
    comps = resolve_competitions(raw)
    events = _get("events?st=1&type=0&version=0")["events"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    df = parse_events(events, comps, {c["i"]: str(c.get("n", "")).strip() for c in raw})
    if df.empty:
        print(f"  iddaa: takip edilen liglerde maç yok ({len(comps)} lig eşleşti)")
        return 0
    df, unknown = _map_teams(df, report=True)
    if unknown:
        print(f"  ! iddaa'da eşleşmeyen takım adları (team_aliases.csv'ye ekle): {', '.join(sorted(unknown))}")
    fuzzy = {k: v for r in df.attrs.pop("resolvers", []) for k, v in r.fuzzy.items()}
    if fuzzy:  # yanlış eşleşme riski: gözden geçir, yanlışsa team_aliases.csv ile düzelt
        print("  ? iddaa bulanık eşleşmeler: " + ", ".join(f"{k} -> {v}" for k, v in sorted(fuzzy.items())))
    _report_names(df, unknown, fuzzy, now)
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
                prev = last.loc[r.event_id].reindex(ODD_COLS).astype(float).to_numpy()  # eski dosyada sütun eksik olabilir
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
