"""football-data.co.uk formatında SAHTE veri üretir (internetsiz test için).

Takım güçleri bilinen bir Dixon-Coles sürecinden gelir, oranlar gerçek olasılıklara gürültü + marj
eklenerek üretilir. Bu veriyle alınan backtest sonuçlarının gerçek dünya için HİÇBİR anlamı yoktur;
amaç sadece kodun uçtan uca çalıştığını doğrulamak.

    python tests/make_synthetic.py <hedef_klasör>
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

RNG = np.random.default_rng(42)
HOME, RHO = 0.27, -0.08
BASE = {"SP1": np.log(1.30), "SP2": np.log(1.12)}


def dc_matrix(lam, nu, rho=RHO, g=10):
    x = np.arange(g + 1)
    m = np.outer(poisson.pmf(x, lam), poisson.pmf(x, nu))
    m[0, 0] *= 1 - lam * nu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + nu * rho
    m[1, 1] *= 1 - rho
    return m / m.sum()


def noisy(p, sd):
    q = p * np.exp(RNG.normal(0, sd, size=len(p)))
    return q / q.sum()


def to_odds(p, margin):
    return np.round(1 / (p * (1 + margin)), 2)


def round_robin(teams):
    t = list(teams)
    if len(t) % 2:
        t.append(None)
    n = len(t)
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        pairs = [(a, b) if r % 2 == 0 else (b, a) for a, b in pairs if a and b]
        rounds.append(pairs)
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def match_row(div, d, h, a, strength, with_result=True, with_closing=True):
    ah, dh = strength[h]
    aa, da = strength[a]
    lam = np.exp(BASE[div] + HOME + ah + da)
    nu = np.exp(BASE[div] + aa + dh)
    m = dc_matrix(lam, nu)
    p1x2 = np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])
    tot = np.add.outer(np.arange(11), np.arange(11))
    pou = np.array([m[tot > 2.5].sum(), m[tot < 2.5].sum()])
    row = {"Div": div, "Date": d.strftime("%d/%m/%Y" if d.year >= 2017 else "%d/%m/%y"),
           "Time": "21:00", "HomeTeam": h, "AwayTeam": a}
    if with_result:
        k = RNG.choice(m.size, p=m.ravel())
        hg, ag = divmod(k, m.shape[1])
        row.update({"FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else "D" if hg == ag else "A",
                    "HTHG": min(hg, RNG.integers(0, 2)), "HTAG": min(ag, RNG.integers(0, 2))})
    books = {"B365": (0.15, 0.055, "B365"), "PS": (0.08, 0.025, "P"), "Avg": (0.12, 0.06, "Avg")}
    if with_closing:
        books.update({"B365C": (0.10, 0.055, "B365C"), "PSC": (0.04, 0.025, "PC"), "AvgC": (0.08, 0.06, "AvgC")})
    for b, (sd, mg, oup) in books.items():
        o = to_odds(noisy(p1x2, sd), mg)
        row[f"{b}H"], row[f"{b}D"], row[f"{b}A"] = o
        ou = to_odds(noisy(pou, sd), mg)
        row[f"{oup}>2.5"], row[f"{oup}<2.5"] = ou
    for pre, oup, src, srcou in [("Max", "Max", "Avg", "Avg")] + ([("MaxC", "MaxC", "AvgC", "AvgC")] if with_closing else []):
        for s in "HDA":
            row[f"{pre}{s}"] = round(row[f"{src}{s}"] * 1.05, 2)
        row[f"{oup}>2.5"] = round(row[f"{srcou}>2.5"] * 1.05, 2)
        row[f"{oup}<2.5"] = round(row[f"{srcou}<2.5"] * 1.05, 2)
    return row


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    cur = config.current_season()
    names = {"SP1": [f"Liga Takim {i:02d}" for i in range(1, 21)],
             "SP2": [f"Segunda Takim {i:02d}" for i in range(1, 23)]}
    strength = {}
    for t in names["SP1"]:
        strength[t] = [RNG.normal(0, 0.25), RNG.normal(0, 0.2)]
    for t in names["SP2"]:
        strength[t] = [RNG.normal(-0.3, 0.2), RNG.normal(0.25, 0.15)]
    fixtures = []
    for season in range(config.FIRST_SEASON, cur + 1):
        start = date(season, 8, 15)
        start += timedelta(days=(5 - start.weekday()) % 7)  # ilk cumartesi
        for div in ("SP1", "SP2"):
            rows = []
            for r, pairs in enumerate(round_robin(names[div])):
                for h, a in pairs:
                    d = start + timedelta(weeks=r, days=int(RNG.integers(-1, 3)))
                    if d < today:
                        rows.append(match_row(div, d, h, a, strength, with_closing=season >= 2019))
                    elif season == cur and d < today + timedelta(days=10):
                        fixtures.append(match_row(div, d, h, a, strength, with_result=False, with_closing=False))
            if rows:
                pd.DataFrame(rows).to_csv(out_dir / f"{div}_{season % 100:02d}{(season + 1) % 100:02d}.csv", index=False)
        # sezon sonu: güçler kayar, 3 takım düşer / 3 takım çıkar
        for t in strength:
            strength[t][0] += RNG.normal(0, 0.08)
            strength[t][1] += RNG.normal(0, 0.06)
        net = lambda t: strength[t][0] - strength[t][1]  # noqa: E731
        down = sorted(names["SP1"], key=net)[:3]
        up = sorted(names["SP2"], key=net)[-3:]
        names["SP1"] = [t for t in names["SP1"] if t not in down] + up
        names["SP2"] = [t for t in names["SP2"] if t not in up] + down
    pd.DataFrame(fixtures).to_csv(out_dir / "fixtures.csv", index=False)
    print(f"sentetik veri yazıldı: {out_dir} ({len(fixtures)} fikstür)")
    return names


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else config.ROOT / "tests" / "_synthetic_data" / "raw")
