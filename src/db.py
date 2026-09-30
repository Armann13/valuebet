"""Ham CSV'leri temizleyip SQLite veritabanına yazar; model ve backtest veriyi buradan okur."""
import re
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

import config
from src.odds import add_sharp

ODDS_COLS = list(dict.fromkeys(config.odds_column_map().values()))

MATCHES_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS matches (
    div     TEXT NOT NULL,
    season  INTEGER NOT NULL,
    date    TEXT NOT NULL,
    time    TEXT,
    home    TEXT NOT NULL,
    away    TEXT NOT NULL,
    fthg    INTEGER, ftag INTEGER, ftr TEXT,
    hthg    INTEGER, htag INTEGER,
    hxg     REAL, axg REAL,
    {", ".join(f"{c} REAL" for c in ODDS_COLS)},
    PRIMARY KEY (div, date, home, away)
);
"""
SCHEMA = MATCHES_SCHEMA + """
CREATE TABLE IF NOT EXISTS predictions (
    created_at TEXT NOT NULL,
    model      TEXT NOT NULL,
    div        TEXT NOT NULL,
    date       TEXT NOT NULL,
    home       TEXT NOT NULL,
    away       TEXT NOT NULL,
    exp_home REAL, exp_away REAL,
    p_home REAL, p_draw REAL, p_away REAL,
    p_over25 REAL, p_under25 REAL, p_btts REAL,
    PRIMARY KEY (model, div, date, home, away, created_at)
);
"""


def connect() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.executescript(SCHEMA)
    return con


def parse_dates(s: pd.Series) -> pd.Series:
    """football-data tarihleri: 'dd/mm/yyyy' veya eski dosyalarda 'dd/mm/yy'."""
    s = s.astype(str).str.strip()
    long_fmt = s.str.len() == 10
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out[long_fmt] = pd.to_datetime(s[long_fmt], format="%d/%m/%Y", errors="coerce")
    out[~long_fmt] = pd.to_datetime(s[~long_fmt], format="%d/%m/%y", errors="coerce")
    return out


def read_raw_csv(path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"okunamadı: {path}")


def normalize(raw: pd.DataFrame, div: str | None = None, season: int | None = None,
              require_result: bool = True) -> pd.DataFrame:
    """football-data formatındaki bir tabloyu bizim sütun adlarımıza çevirir."""
    raw = raw.rename(columns=lambda c: str(c).strip().lstrip("﻿"))
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
    df = pd.DataFrame(index=raw.index)
    df["div"] = raw["Div"].astype(str).str.strip() if "Div" in raw else div
    if div is not None:
        df["div"] = div
    df["date"] = parse_dates(raw["Date"])
    df["time"] = raw["Time"].astype(str).str.strip() if "Time" in raw else None
    df["home"] = raw["HomeTeam"].astype(str).str.strip()
    df["away"] = raw["AwayTeam"].astype(str).str.strip()
    for src, dst in [("FTHG", "fthg"), ("FTAG", "ftag"), ("HTHG", "hthg"), ("HTAG", "htag")]:
        df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    df["ftr"] = raw["FTR"].astype(str).str.strip() if "FTR" in raw else None
    colmap = config.odds_column_map()
    for src, dst in colmap.items():
        if dst in df:  # aynı hedefe iki kaynak düşerse ilk dolu olanı tut
            if src in raw:
                df[dst] = df[dst].fillna(pd.to_numeric(raw[src], errors="coerce"))
            continue
        df[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan
    # 1.0 veya altı oranlar hatalı kayıttır
    for c in ODDS_COLS:
        df.loc[df[c] <= 1.0, c] = np.nan
    df = df.dropna(subset=["date"])
    if require_result:
        df = df.dropna(subset=["fthg", "ftag"])
        df["fthg"] = df["fthg"].astype(int)
        df["ftag"] = df["ftag"].astype(int)
    if season is not None:
        df["season"] = season
    else:
        df["season"] = np.where(df["date"].dt.month >= 7, df["date"].dt.year, df["date"].dt.year - 1)
    return df.reset_index(drop=True)


def build() -> int:
    """data/raw içindeki tüm sezon dosyalarını veritabanına yükler (tekrar çalıştırmak güvenli).

    Dosya adı 'SP1_2526.csv' biçimindeyse lig ve sezon addan alınır; değilse (ör. tarayıcıdan elle
    indirilmiş 'SP1 (3).csv') dosyanın içindeki Div ve Date sütunlarından bulunur.
    """
    files = sorted(f for f in config.RAW_DIR.glob("*.csv") if f.name.lower() != "fixtures.csv")
    pat = re.compile(r"^(?P<div>[A-Z0-9]+)_(?P<code>\d{4})\.csv$")
    frames = []
    for f in files:
        m = pat.match(f.name)
        try:
            raw = read_raw_csv(f)
            if m:
                if m["div"] not in config.LEAGUES:
                    continue
                df = normalize(raw, div=m["div"], season=2000 + int(m["code"][:2]))
            else:
                if not {"Div", "HomeTeam", "AwayTeam", "Date"} <= set(c.strip().lstrip("﻿") for c in raw.columns):
                    print(f"  ? {f.name}: football-data formatında değil, atlandı")
                    continue
                df = normalize(raw)
                df = df[df["div"].isin(config.LEAGUES)]
        except Exception as e:  # bozuk dosya tüm yüklemeyi durdurmasın
            print(f"  ! {f.name}: okunamadı ({e})")
            continue
        if df.empty:
            continue
        frames.append(df)
        seasons = ", ".join(f"{s}/{(s + 1) % 100:02d}" for s in sorted(df["season"].unique()))
        print(f"  {f.name}: {len(df)} maç ({', '.join(sorted(df['div'].unique()))} {seasons})")
    if not frames:
        print("  ! data/raw içinde sezon dosyası yok. Önce: python main.py download")
        return 0
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["div", "date", "home", "away"], keep="last")
    all_df = attach_xg(all_df)
    all_df["date"] = all_df["date"].dt.strftime("%Y-%m-%d")
    cols = ["div", "season", "date", "time", "home", "away", "fthg", "ftag", "ftr", "hthg", "htag",
            "hxg", "axg"] + ODDS_COLS
    con = connect()
    with con:
        # matches her build'de sıfırdan yazılır; tabloyu yeniden kurmak yeni oran sütunlarını da ekler
        con.execute("DROP TABLE matches")
        con.execute(MATCHES_SCHEMA)
        con.executemany(
            f"INSERT INTO matches ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            all_df[cols].astype(object).where(all_df[cols].notna(), None).itertuples(index=False, name=None),
        )
    con.close()
    return len(all_df)


def attach_xg(df: pd.DataFrame) -> pd.DataFrame:
    """data/raw/xg/{lig}_{sezon}.csv (Understat) dosyalarındaki maç xG'sini (hxg, axg) maçlara ekler.
    Eşleştirme (lig, sezon, ev, deplasman) ile; Understat takım adları football-data adlarına çevrilir.
    Gollerin tutmadığı eşleşmeler atılır (yanlış eşleşme koruması)."""
    from src.teams import TeamResolver
    df = df.assign(hxg=np.nan, axg=np.nan)
    pat = re.compile(r"^(?P<div>[A-Z0-9]+)_(?P<code>\d{4})\.csv$")
    parts, unknown = [], set()
    for f in sorted((config.RAW_DIR / "xg").glob("*.csv")):
        m = pat.match(f.name)
        if not m or m["div"] not in config.LEAGUES:
            continue
        div, season = m["div"], 2000 + int(m["code"][:2])
        x = pd.read_csv(f)
        resolve = TeamResolver(df.loc[df["div"] == div, "home"].unique())
        for col in ("home", "away"):
            names = {n: resolve(n) for n in x[col].unique()}
            unknown |= {n for n, v in names.items() if v is None}
            x[col] = x[col].map(names)
        parts.append(x.dropna(subset=["home", "away"]).assign(div=div, season=season))
    if not parts:
        return df
    key = ["div", "season", "home", "away"]
    xg = pd.concat(parts, ignore_index=True).drop_duplicates(subset=key, keep=False)  # belirsiz eşleşme: hiçbiri
    merged = df.drop(columns=["hxg", "axg"]).merge(xg[key + ["hg", "ag", "hxg", "axg"]], on=key, how="left")
    bad = merged["hg"].notna() & ((merged["hg"] != merged["fthg"]) | (merged["ag"] != merged["ftag"]))
    merged.loc[bad, ["hxg", "axg"]] = np.nan
    n_ok = int(merged["hxg"].notna().sum())
    print(f"  xG: {n_ok} maça eklendi" + (f", {int(bad.sum())} maçta gol tutmadı (atıldı)" if bad.any() else ""))
    if unknown:
        print(f"  ! xG'de eşleşmeyen takım adları (team_aliases.csv'ye ekle): {', '.join(sorted(unknown))}")
    return merged.drop(columns=["hg", "ag"])


COVERAGE_COLS = {"B365 1X2": "b365_h", "keskin aç. 1X2": "sharp_h", "keskin kap. 1X2": "sharpc_h",
                 "B365 Ü/A": "b365_o25", "keskin aç. Ü/A": "sharp_o25", "keskin kap. Ü/A": "sharpc_o25",
                 "xG": "hxg"}


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Lig ve sezon başına maç sayısı ve oran sütunlarının doluluğu (%). Backtest'in hangi sezonlarda
    (keskin kapanış oranı olan) anlamlı olduğunu gösterir. df: load_matches() çıktısı."""
    cols = list(COVERAGE_COLS.values())
    g = df[cols].notna().groupby([df["div"], df["season"]])
    out = (100 * g.mean()).round().astype(int).rename(columns={v: k for k, v in COVERAGE_COLS.items()})
    out.insert(0, "maç", g.size())
    out = out.reset_index()
    out["season"] = out["season"].map(lambda s: f"{s}/{(s + 1) % 100:02d}")
    return out.rename(columns={"div": "lig", "season": "sezon"})


def load_matches(div: str | None = None) -> pd.DataFrame:
    con = connect()
    q = "SELECT * FROM matches" + (" WHERE div = ?" if div else "") + " ORDER BY date, home"
    df = pd.read_sql_query(q, con, params=(div,) if div else None)
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    return add_sharp(df)


def save_predictions(df: pd.DataFrame, model_name: str) -> None:
    cols = ["div", "date", "home", "away", "exp_home", "exp_away", "p_home", "p_draw", "p_away",
            "p_over25", "p_under25", "p_btts"]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.insert(0, "model", model_name)
    out.insert(0, "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    con = connect()
    with con:
        out.to_sql("predictions", con, if_exists="append", index=False)
    con.close()
