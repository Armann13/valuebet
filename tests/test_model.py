"""Testler. Çalıştırma:  python tests/test_model.py   (veya pytest yüklüyse: python -m pytest tests)"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import check_grad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src import db  # noqa: E402
from src.dixon_coles import DixonColes, _objective  # noqa: E402
from src.odds import add_sharp, devig, devig_frame, margin  # noqa: E402


def _simulate(n_teams=20, seasons=3, home=0.28, rho=-0.1, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    rows, day = [], pd.Timestamp("2020-08-15")
    from tests.make_synthetic import dc_matrix
    for s in range(seasons):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lam = np.exp(0.2 + home + att[i] + dfn[j])
                nu = np.exp(0.2 + att[j] + dfn[i])
                m = dc_matrix(lam, nu, rho)
                k = rng.choice(m.size, p=m.ravel())
                rows.append({"date": day + pd.Timedelta(days=int(rng.integers(0, 300)) + 365 * s),
                             "home": teams[i], "away": teams[j], "fthg": k // 11, "ftag": k % 11})
    return pd.DataFrame(rows).sort_values("date"), att, dfn


def test_gradient():
    rng = np.random.default_rng(0)
    n, m = 6, 200
    hi, ai = rng.integers(0, n, m), rng.integers(0, n, m)
    x, y = rng.poisson(1.4, m).astype(float), rng.poisson(1.1, m).astype(float)
    w = rng.uniform(0.3, 1, m)
    th = rng.normal(0, 0.2, 3 + 2 * n)
    th[2] = -0.07
    for r, rt in [(0.5, None), (0.5, 8.0), (3.0, 0.2)]:
        err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, r, rt)[0],
                         lambda t: _objective(t, hi, ai, x, y, w, n, r, rt)[1], th)
        assert err < 1e-5, (r, rt, err)
    # iki lig: [mu(2), home(2), rho, lig_gücü(1), yeni_gelen çıkan & düşen, atak(n), defans(n)]
    li, tl, ts = rng.integers(0, 2, m), rng.integers(0, 2, n), rng.integers(-1, 2, n)
    th2 = np.r_[0.1, 0.3, 0.25, 0.2, -0.07, -0.4, -0.15, 0.2, th[3:]]
    err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts)[0],
                     lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts)[1], th2)
    assert err < 1e-5, err
    # motivasyonlu: parametre yeni gelen sapmalarından sonra, takımlardan önce
    mot = rng.integers(-1, 2, m).astype(float)
    th3 = np.r_[th2[:8], 0.3, th2[8:]]
    err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot)[0],
                     lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot)[1], th3)
    assert err < 1e-5, err
    th4 = np.r_[th[:3], 0.3, th[3:]]  # tek lig + motivasyon
    err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, None, None, 1, None, mot)[0],
                     lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, None, None, 1, None, mot)[1], th4)
    assert err < 1e-5, err
    # xG karışımı: Poisson hedefi tam sayı değil, tau gerçek skorla
    xp, yp = x * 0.5 + rng.uniform(0, 2, m) * 0.5, y * 0.5 + rng.uniform(0, 2, m) * 0.5
    err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot, xp, yp)[0],
                     lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot, xp, yp)[1], th3)
    assert err < 1e-5, err
    # tarafsız saha (milli maçlar): ev avantajı sadece hm = 1 olan maçlarda
    hm = (rng.uniform(size=m) > 0.3).astype(float)
    err = check_grad(lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot, xp, yp, hm)[0],
                     lambda t: _objective(t, hi, ai, x, y, w, n, 2.0, 8.0, li, tl, 2, ts, mot, xp, yp, hm)[1], th3)
    assert err < 1e-5, err
    # ridge = ridge_total iken ceza eski haliyle aynı: ridge * (Σa² + Σd²)
    a, d = th[3:3 + n], th[3 + n:]
    pen = (_objective(th, hi, ai, x, y, w, n, 2.0, 2.0)[0] - _objective(th, hi, ai, x, y, w, n, 0.0, 0.0)[0]) * w.sum()
    assert abs(pen - 2.0 * (a @ a + d @ d)) < 1e-9, pen


def test_motivation_live():
    from src.features import _live
    table = [80, 75, 70, 65, 60, 58, 55, 50, 48, 46, 45, 44, 43, 42, 41, 40, 39, 38, 30, 20]  # 20 takım
    assert not _live("SP1", 46, 10, 2, table, 1.0)   # orta sıra, 2 maç kala hiçbir sınıra yetişemez
    assert _live("SP1", 39, 17, 2, table, 1.0)       # 18.'nin 1 puan önünde: küme hattında
    assert not _live("SP1", 80, 1, 2, table, 1.0)    # şampiyonluk kesinleşmiş
    assert _live("SP1", 46, 10, 10, table, 1.0)      # 10 maç kala Avrupa'ya 9 puan: hâlâ hedefli


def test_motivation_recovery():
    """Hedefli takım m kadar güçlü oynuyorsa (puan durumuna göre tur tur üretilmiş lig), model m'yi bulur."""
    from src.features import add_motivation
    from tests.make_synthetic import dc_matrix, round_robin
    rng = np.random.default_rng(2)
    teams = [f"T{i:02d}" for i in range(20)]
    s = dict(zip(teams, rng.normal(0, 0.35, 20)))
    true_m, rows = 0.6, []
    for season in range(2019, 2024):
        start = pd.Timestamp(f"{season}-08-17")
        for r, pairs in enumerate(round_robin(teams)):
            fx = pd.DataFrame([{"div": "SP1", "season": season, "date": start + pd.Timedelta(weeks=r),
                                "home": h, "away": a} for h, a in pairs])
            past = pd.DataFrame(rows) if rows else fx.assign(fthg=np.nan, ftag=np.nan).iloc[:0]
            for rec in add_motivation(past, fx, reach=1.0).itertuples():
                ms = 0.5 * true_m * rec.mot
                mm = dc_matrix(np.exp(0.45 + (s[rec.home] - s[rec.away]) / 2 + ms),
                               np.exp(0.2 + (s[rec.away] - s[rec.home]) / 2 - ms))
                k = rng.choice(mm.size, p=mm.ravel())
                rows.append({"div": "SP1", "season": season, "date": rec.date, "home": rec.home, "away": rec.away,
                             "fthg": k // 11, "ftag": k % 11, "mot": rec.mot})
    df = pd.DataFrame(rows)
    assert (df["mot"] != 0).sum() > 150, (df["mot"] != 0).sum()
    m = DixonColes(xi=0.0, ridge=1.0, ridge_total=16.0, lookback_days=5000, motivation_fixed=None).fit(df)
    assert abs(m.motivation - true_m) < 0.3, m.motivation
    fixed = DixonColes(xi=0.0, ridge=1.0, ridge_total=16.0, lookback_days=5000, motivation_fixed=0.25).fit(df)
    assert fixed.motivation == 0.25
    assert m.predict("T00", "T01", mot=1)["p_home"] > m.predict("T00", "T01", mot=0)["p_home"]


def test_ridge_total_shrinks_totals():
    df, _, _ = _simulate(seasons=2)
    m = DixonColes(xi=0.0, ridge=0.5, ridge_total=1e6, lookback_days=5000).fit(df)
    assert np.abs(m.attack + m.defence).max() < 1e-3  # eğilim sıfıra çekildi
    assert np.abs(m.attack - m.defence).std() > 0.1  # güç farkları duruyor


def _two_leagues(seed, seasons=3, n=12, moves=2):
    """Ortak ölçekte iki lig (L1 üst, L2 alt). Her sezon arası alt ligin en iyi `moves` takımı çıkar, üst ligin
    en kötüleri düşer. Döner: maçlar, gerçek güçler, son sezondaki ligler, son sezon çıkan / düşen takımlar."""
    from tests.make_synthetic import dc_matrix
    rng = np.random.default_rng(seed)
    s = {f"A{i:02d}": 0.3 + rng.normal(0, 0.2) for i in range(n)}      # güç (atak - defans)
    s.update({f"B{i:02d}": -0.3 + rng.normal(0, 0.2) for i in range(n)})
    league = {t: ("L1" if t[0] == "A" else "L2") for t in s}
    rows, day, moved = [], pd.Timestamp("2020-08-15"), {}
    for season in range(seasons):
        for lg in ("L1", "L2"):
            ts = [t for t in s if league[t] == lg]
            for h in ts:
                for a in ts:
                    if h != a:
                        m = dc_matrix(np.exp(0.45 + (s[h] - s[a]) / 2), np.exp(0.2 + (s[a] - s[h]) / 2))
                        k = rng.choice(m.size, p=m.ravel())
                        rows.append({"div": lg, "season": 2020 + season, "home": h, "away": a,
                                     "date": day + pd.Timedelta(days=int(rng.integers(0, 280)) + 365 * season),
                                     "fthg": k // 11, "ftag": k % 11})
        if season < seasons - 1:
            moved = {}
            l1 = sorted([t for t in s if league[t] == "L1"], key=s.get)
            l2 = sorted([t for t in s if league[t] == "L2"], key=s.get)
            for t in l1[:moves]:
                league[t], moved[t] = "L2", -1
            for t in l2[-moves:]:
                league[t], moved[t] = "L1", 1
    return pd.DataFrame(rows).sort_values("date"), s, league, moved


def test_joint_leagues():
    """İki lig ortak ölçekte: lig farkı tahmin edilir, lig değiştiren takımın geçmişi korunur."""
    df, s, league, _ = _two_leagues(seed=0)
    m = DixonColes(xi=0.0, ridge=4.0, ridge_total=16.0, lookback_days=5000).fit(df)
    net = dict(zip(m.ratings()["team"], m.ratings()["net"]))
    gap = lambda v: np.mean([v[t] for t in s if league[t] == "L2"]) - np.mean([v[t] for t in s if league[t] == "L1"])  # noqa: E731
    assert abs(gap(net) - gap(s)) < 0.15, (gap(net), gap(s))  # gerçek fark ≈ -0.65
    old = {t: v for t, v in net.items() if not m.team_status.get(t)}  # lig gücü = yeni gelmemişlerin ortalaması
    old_gap = np.mean([v for t, v in old.items() if league[t] == "L2"]) - np.mean([v for t, v in old.items() if league[t] == "L1"])
    assert abs(m.league_strength["L2"] - old_gap) < 0.05, (m.league_strength, old_gap)
    assert np.corrcoef([net[t] for t in s], list(s.values()))[0, 1] > 0.85  # tek ölçekte sıralama
    # lig değiştiren takım bilinir; alt ligde bilinmeyen takım alt ligin zayıflarına göre kurulur
    p = m.predict("Yeni", "A00", "L2")
    assert not p["known_home"] and p["known_away"] and p["p_home"] < 0.5


def test_newcomer_offset():
    """Üst lige yeni çıkanlar o ligin ortalamasının altında, düşenler alt ligin üstünde olmalı; çekim hedefi
    bunu öğrenir (yoksa çıkan takım üst lig ortalamasına doğru yukarı çekilirdi)."""
    df, s, league, moved = _two_leagues(seed=0, moves=3)
    m = DixonColes(xi=0.0, ridge=4.0, ridge_total=16.0, lookback_days=5000).fit(df)
    assert {t: m.team_status.get(t, 0) for t in moved} == moved
    assert sum(1 for v in m.team_status.values() if v) == len(moved)
    assert m.newcomer_offset["çıkan"] < -0.1 and m.newcomer_offset["düşen"] > 0.1, m.newcomer_offset


def test_parameter_recovery():
    df, att, dfn = _simulate()
    model = DixonColes(xi=0.0, ridge=0.1, lookback_days=5000).fit(df)
    assert abs(model.home[model.leagues[0]] - 0.28) < 0.08, model.home
    assert -0.25 < model.rho < 0.05, model.rho
    order = [int(t[1:]) for t in model.teams]
    assert np.corrcoef(model.attack, att[order])[0, 1] > 0.85
    assert np.corrcoef(model.defence, dfn[order])[0, 1] > 0.8


def test_probabilities():
    df, _, _ = _simulate(seasons=1)
    model = DixonColes().fit(df)
    p = model.predict("T00", "T01")
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-9
    assert abs(p["p_over25"] + p["p_under25"] - 1) < 1e-9
    q = model.predict("Yeni Takim", "T01")  # bilinmeyen takım çökmemeli
    assert not q["known_home"] and 0 < q["p_home"] < 1


def test_devig():
    odds = [1.80, 3.60, 4.75]
    assert margin(odds) > 0
    for method in ("proportional", "power"):
        p = devig(odds, method)
        assert abs(p.sum() - 1) < 1e-9 and np.all(p > 0)
    # power yöntemi marjı sürprizlere daha çok yükler -> favorinin adil olasılığı daha yüksek
    assert devig(odds, "power")[0] > devig(odds, "proportional")[0]


def test_devig_frame_matches_rowwise():
    rng = np.random.default_rng(3)
    p = rng.dirichlet([4, 3, 3], 300)
    odds = pd.DataFrame(1 / (p * rng.uniform(0.97, 1.15, (300, 1))), columns=list("hda"))  # negatif marj da var
    odds.iloc[5, 1], odds.iloc[7, 0] = np.nan, 1.0
    for method in ("proportional", "power"):
        vec = devig_frame(odds, list("hda"), method)
        row = np.array([devig(r, method) for r in odds.to_numpy()])
        assert np.allclose(vec, row, atol=1e-9, equal_nan=True), method
        assert np.isnan(vec[[5, 7]]).all()


def test_add_sharp():
    nan = np.nan
    df = pd.DataFrame({  # satırlar: Pinnacle tam | Pinnacle eksik -> borsa | borsa sığ (marj > %10) | hiçbiri
        "ps_h": [2.0, 2.0, nan, nan], "ps_d": [3.4, nan, nan, nan], "ps_a": [4.0, 4.0, nan, nan],
        "bfe_h": [2.1, 2.1, 1.5, nan], "bfe_d": [3.6, 3.6, 2.5, nan], "bfe_a": [4.2, 4.2, 3.0, nan],
        "ps_o25": [1.9, 1.9, nan, nan], "ps_u25": [1.95, 1.95, nan, nan],
    })
    s = add_sharp(df)
    src = s["sharp_src_1x2"]  # eksik kaynak: pandas 2'de None, pandas 3'te NaN
    assert list(src[:2]) == ["ps", "bfe"] and src[2:].isna().all()
    assert s.loc[0, "sharp_d"] == 3.4 and s.loc[1, "sharp_d"] == 3.6  # bir marketin oranları tek kaynaktan
    assert np.isnan(s.loc[2:, ["sharp_h", "sharp_d", "sharp_a"]].to_numpy()).all()
    assert list(s["sharp_src_ou"][:2]) == ["ps", "ps"] and np.isnan(s["sharpc_h"]).all()


def test_parse_understat():
    from src.download import parse_understat
    data = {"dates": [
        {"id": "1", "isResult": True, "h": {"id": "1", "title": "Athletic Club"}, "a": {"id": "2", "title": "Getafe"},
         "goals": {"h": "1", "a": "1"}, "xG": {"h": "0.989", "a": "0.475"}, "datetime": "2024-08-15 17:00:00"},
        {"id": "2", "isResult": False, "h": {"id": "3", "title": "Betis"}, "a": {"id": "4", "title": "Girona"},
         "goals": {"h": None, "a": None}, "xG": {"h": None, "a": None}, "datetime": "2024-08-15 19:30:00"}]}
    df = parse_understat(data)
    assert len(df) == 1 and df.loc[0, "hxg"] == 0.989 and df.loc[0, "ag"] == 1
    from src.teams import TeamResolver
    assert TeamResolver(["Ath Bilbao", "Getafe", "Betis"])("Athletic Club") == "Ath Bilbao"
    r = TeamResolver(["Sutton", "Man City", "Man United", "Paris SG", "Paris FC"])
    assert r("Sutton United") == "Sutton" and r.fuzzy == {"Sutton United": "Sutton"}  # tek aday: önek kuralı
    assert r("Manchester United") == "Man United" and r("Paris Saint Germain") == "Paris SG"  # eşleştirme tablosu
    assert r("Paris") is None  # iki aday ("Paris SG", "Paris FC"): tahmin etme


def test_parse_iddaa():
    from src.iddaa import parse_events
    ev = [{"i": 1, "hn": "Real Madrid", "an": "Villarreal", "ci": 129, "d": 1790870400, "mbc": 1,
           "m": [{"t": 1, "st": 1, "o": [{"no": 1, "odd": 1.45, "wodd": 1.39, "n": "1"},
                                         {"no": 2, "odd": 4.4, "wodd": 4.23, "n": "0"},
                                         {"no": 3, "odd": 5.9, "wodd": 5.67, "n": "2"}]},
                 {"t": 2, "st": 101, "sov": "1.5", "o": [{"no": 1, "odd": 4.0, "n": "Alt"}, {"no": 2, "odd": 1.2, "n": "Üst"}]},
                 {"t": 2, "st": 101, "sov": "2.5", "o": [{"no": 1, "odd": 2.1, "wodd": 2.02, "n": "Alt"},
                                                         {"no": 2, "odd": 1.62, "wodd": 1.56, "n": "Üst"}]},
                 {"t": 2, "st": 101, "sov": "3.5", "o": [{"no": 1, "odd": 1.3, "n": "Alt"}, {"no": 2, "odd": 3.1, "n": "Üst"}]},
                 {"t": 2, "st": 92, "o": [{"n": "1 ve 0", "odd": 1.0}, {"n": "1 ve 2", "odd": 1.12}]},  # 1.00 = kapalı
                 {"t": 2, "st": 7, "sov": "3.5", "o": [{"n": "1 ve Üst", "odd": 3.3}]},  # başka baraj: alınmaz
                 {"t": 2, "st": 7, "sov": "2.5", "o": [{"n": "1 ve Üst", "odd": 2.05}, {"n": "2 ve Üst", "odd": 9.5}]},
                 {"t": 2, "st": 698, "o": [{"n": "1 ve Var", "odd": 2.9}, {"n": "0 ve Var", "odd": 7.0},
                                           {"n": "2 ve Var", "odd": 1.0}]},
                 {"t": 2, "st": 90, "o": [{"n": "1/1", "odd": 2.05}, {"n": "2/1", "odd": 21.1}, {"n": "0/0", "odd": 6.8}]},
                 {"t": 2, "st": 89, "o": [{"no": 1, "odd": 1.7, "n": "Var"}, {"no": 2, "odd": 2.0, "n": "Yok"}]}]},
          {"i": 2, "hn": "X", "an": "Y", "ci": 999, "d": 1790870400, "m": []}]  # başka lig: atlanır
    df = parse_events(ev, {129: "SP1"})
    assert len(df) == 1
    r = df.iloc[0]
    assert (r["div"], r["odd_1"], r["odd_x"], r["odd_2"], r["ou_line"], r["odd_ust"], r["mbs"]) == \
        ("SP1", 1.45, 4.4, 5.9, 2.5, 1.62, 1)  # birden çok baraj varsa 2.5
    assert np.isnan(r["odd_1x"]) and r["odd_12"] == 1.12 and r["odd_kg_var"] == 1.7
    assert (r["odd_ms1_ust"], r["odd_ms2_ust"], r["odd_ms1_kg"], r["odd_msx_kg"]) == (2.05, 9.5, 2.9, 7.0)
    assert np.isnan(r["odd_ms2_kg"])
    assert (r["odd_iyms_11"], r["odd_iyms_21"], r["odd_iyms_xx"]) == (2.05, 21.1, 6.8) and np.isnan(r["odd_iyms_12"])
    from src.iddaa import parse_scores, score_odds
    detail = {"m": [{"t": 2, "st": 717, "o": [{"n": "1:0", "odd": 3.4}]},  # ilk yarı skoru: alınmaz
                    {"t": 2, "st": 86, "o": [{"n": "1:0", "odd": 10.15}, {"n": "2:0", "odd": 6.97},
                                             {"n": "3:3", "odd": 1.0}, {"n": "diğer", "odd": 5.0}]}]}
    assert parse_scores(detail) == "1:0=10.15;2:0=6.97" and parse_scores({"m": []}) == "yok"
    assert score_odds("1:0=10.15;2:0=6.97") == {"1-0": 10.15, "2-0": 6.97}
    assert score_odds("yok") == {} and score_odds(float("nan")) is None
    assert r["kickoff_utc"] == "2026-10-01 16:00"
    from src.iddaa import resolve_competitions
    comps = [{"i": 129, "si": "1", "n": "İspanya La Liga"}, {"i": 1951, "si": "1", "n": "İspanya La Liga 2"},
             {"i": 19843, "si": "1", "n": "İskoçya Premiership, Şampiyona Turu"},
             {"i": 80676, "si": "1", "n": "Türkiye Süper Lig Beşiktaş Özel"}, {"i": 584, "si": "1", "n": "Türkiye Süper Lig"},
             {"i": 5, "si": "5", "n": "İspanya La Liga"}]  # si=5: başka spor
    assert resolve_competitions(comps) == {129: "SP1", 1951: "SP2", 19843: "SC0", 584: "T1"}
    # milli turnuvalar; gençler, kadınlar ve kulüp turnuvaları hariç
    milli = [{"i": 1407, "si": "1", "n": "UEFA Uluslar Ligi"}, {"i": 1484, "si": "1", "n": "Uluslararası Hazırlık Maçları"},
             {"i": 2902, "si": "1", "n": "CONCACAF Uluslar A Ligi, Grp A"},
             {"i": 1432, "si": "1", "n": "Afrika Uluslar Kup.,Eleme,Gr.D"},
             {"i": 7, "si": "1", "n": "Dünya Kupası Elemeleri, Avrupa"},
             {"i": 126, "si": "1", "n": "UEFA Şampiyonlar Ligi"}, {"i": 958, "si": "1", "n": "Kulüplerarası Hazırlık Maçları"},
             {"i": 8, "si": "1", "n": "U21 Avrupa Şampiyonası Elemeleri"}, {"i": 9, "si": "1", "n": "Kadınlar Dünya Kupası"},
             {"i": 10, "si": "1", "n": "Dünya Kupası U20"}, {"i": 11, "si": "1", "n": "Kulüpler Dünya Kupası"}]
    assert resolve_competitions(milli) == {i: config.MILLI for i in (1407, 1484, 2902, 1432, 7)}


def test_milli_names():
    """Türkçe ülke adları veri setindeki adlara; bulanık eşleşme yok (Avusturya ≠ Avustralya)."""
    from src.milli import TEAMS, ULKELER
    from src.teams import TeamResolver
    r = TeamResolver(sorted(TEAMS), aliases=ULKELER, allow_fuzzy=False)
    cases = {"Avusturya": "Austria", "Avustralya": "Australia", "Sırbistan": "Serbia", "Kuzey İrlanda": "Northern Ireland",
             "İrlanda": "Republic of Ireland", "Çekya": "Czech Republic", "Türkiye": "Turkey", "Curacao": "Curaçao",
             "Kuzey Makedonya": "North Macedonia", "Fildişi Sahili": "Ivory Coast", "Kongo DC": "DR Congo",
             "Kongo": "Congo", "Gine": "Guinea", "Spain": "Spain", "Kıbrıs Rum Kesimi": "Cyprus",
             "ABD Virgin Adaları": "United States Virgin Islands", "ST Kitts ve Nevis": "Saint Kitts and Nevis",
             "Türkiye U21": None, "Bask Ülkesi": None}
    got = {k: r(k) for k in cases}
    assert got == cases, {k: v for k, v in got.items() if v != cases[k]}
    assert r.fuzzy == {}


def test_espn_results():
    """ESPN skorları: biten maçlar, ad eşleştirme, uzatmalı maçta 90 dakika skoru, ana veri setiyle tekrar yok."""
    import tempfile
    from src import milli
    from src.teams import TeamResolver
    team = lambda n, ha, s, ls=None: {"homeAway": ha, "team": {"displayName": n}, "score": s,  # noqa: E731
                                      **({"linescores": [{"value": v} for v in ls]} if ls else {})}
    ev = lambda st, c, d="2026-09-29T18:45Z": {"date": d, "status": {"type": {"completed": st != "pre", "name": st}},  # noqa: E731
                                                "competitions": [{"competitors": c}]}
    data = {"events": [ev("STATUS_FULL_TIME", [team("Czechia", "home", "0"), team("England", "away", "2")]),
                       ev("pre", [team("Spain", "home", "0"), team("Croatia", "away", "0")]),
                       ev("STATUS_FINAL_AET", [team("Wales", "home", "2", [1, 0, 1]), team("Norway", "away", "1", [0, 1, 0])]),
                       ev("STATUS_FULL_TIME", [team("Atlantis", "home", "1"), team("Spain", "away", "1")])]}
    resolve = TeamResolver(sorted(milli.TEAMS), aliases=milli.ESPN_ADLAR, allow_fuzzy=False)
    rows, unknown = milli.parse_espn(data, "uefa.nations", resolve)
    assert [(r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in rows] == \
        [("Czech Republic", "England", 0, 2), ("Wales", "Norway", 1, 1)]  # uzatma golü sayılmaz
    assert unknown == {"Atlantis"} and rows[0]["tournament"] == "UEFA Nations League"
    old = config.RAW_DIR
    with tempfile.TemporaryDirectory() as d:
        config.RAW_DIR = Path(d)
        try:
            (Path(d) / "milli").mkdir()
            base = pd.DataFrame([{"date": "2026-09-28", "home_team": "Czech Republic", "away_team": "England",
                                  "home_score": 0, "away_score": 2, "tournament": "UEFA Nations League", "city": "",
                                  "country": "", "neutral": False}])
            base.to_csv(milli.results_path(), index=False)
            pd.DataFrame(rows).to_csv(milli.espn_path(), index=False)
            got = milli.load_results(since="2026-01-01")
        finally:
            config.RAW_DIR = old
    assert len(got) == 2 and (got["home"] == "Czech Republic").sum() == 1  # aynı maç ±1 gün: ana veri seti öncelikli


def test_neutral_venue():
    """Tarafsız sahada ev avantajı yok: model ev avantajını sadece ev sahibinin kendi sahasındaki maçlardan öğrenir;
    tarafsız sahada A-B tahmini B-A'nın aynası."""
    df, _, _ = _simulate(seed=7)
    rng = np.random.default_rng(7)
    df["neutral"] = rng.uniform(size=len(df)) < 0.4
    # tarafsız maçların yarısında ev / deplasman rolleri değişir (gerçekteki gibi "ev" adı rastgele)
    swap = df["neutral"] & (rng.uniform(size=len(df)) < 0.5)
    df.loc[swap, ["home", "away"]] = df.loc[swap, ["away", "home"]].to_numpy()
    df.loc[swap, ["fthg", "ftag"]] = df.loc[swap, ["ftag", "fthg"]].to_numpy()
    m = DixonColes(xi=0.0, ridge=0.1, lookback_days=5000).fit(df)
    p = m.predict("T00", "T01", neutral=True)
    q = m.predict("T01", "T00", neutral=True)
    assert abs(p["p_home"] - q["p_away"]) < 1e-9 and abs(p["p_draw"] - q["p_draw"]) < 1e-9
    assert m.predict("T00", "T01")["p_home"] > p["p_home"]  # kendi sahasında daha olası
    w = DixonColes(xi=0.0, ridge=0.1, lookback_days=5000).fit(df.assign(weight=np.where(df["neutral"], 0.5, 1.0)))
    assert w.converged and np.isfinite(w.attack).all()


def test_tavsiye_markets():
    from src.tavsiye import calibrate, market_probs
    from tests.make_synthetic import dc_matrix
    m = dc_matrix(1.6, 0.9)
    p = market_probs(m)
    assert abs(p["MS 1"] + p["MS X"] + p["MS 2"] - 1) < 1e-9
    assert abs(p["ÇŞ 1-X"] - (p["MS 1"] + p["MS X"])) < 1e-9 and abs(p["2.5 Alt"] + p["2.5 Üst"] - 1) < 1e-9
    assert p["MS 1 & KG Var"] <= min(p["MS 1"], p["KG Var"])
    c = market_probs(calibrate(m, (0.40, 0.30, 0.30)))  # piyasaya ayarlı: 1X2 toplamları hedefle aynı
    assert abs(c["MS 1"] - 0.40) < 1e-9 and abs(c["MS X"] - 0.30) < 1e-9 and abs(c["MS 2"] - 0.30) < 1e-9
    c = market_probs(calibrate(m, (0.40, 0.30, 0.30), (0.55, 0.45)))  # 1X2 + alt/üst birlikte
    assert abs(c["MS 1"] - 0.40) < 1e-6 and abs(c["2.5 Üst"] - 0.55) < 1e-6
    from src.tavsiye import NARROWER, bet_won
    assert bet_won("MS 1 & KG Var", 2, 1) and not bet_won("MS 1 & KG Var", 2, 0)
    # merdiven: daha dar tahmin tutarsa geniş olan da tutar
    assert {"MS 1 & 2.5 Üst", "MS 1 & KG Var"} <= set(NARROWER["MS 1"])
    assert "MS 1 & KG Var" in NARROWER["MS 1 & 2.5 Üst"] and "MS 2" in NARROWER["ÇŞ X-2"]
    assert "MS 1" not in NARROWER["2.5 Üst"] and "MS 1" not in NARROWER["MS 1"] and NARROWER["KG Yok"] == []
    assert bet_won("ÇŞ X-2", 1, 1) and bet_won("2.5 Alt", 1, 1) and bet_won("1-1", 1, 1) and not bet_won("2-0", 2, 1)


def test_telegram_messages():
    from datetime import datetime
    from src.telegram import build_messages
    row = {"maç": "A – B", "lig": "E3", "tarih": "03.10", "gün": "2026-10-03", "mbs": 1, "kaynak": "piyasa"}
    picks = pd.DataFrame([
        {**row, "seviye": "💎 Değeri pozitif olanlar", "tahmin": "MS 1", "p": 0.55, "odd": 2.0, "değer": 0.10,
         "gerçek_değer": True},
        {**row, "seviye": "🟢 Tier 1 — Düşük risk", "tahmin": "ÇŞ 1-X", "p": 0.80, "odd": 1.25,
         "değer": float("nan"), "gerçek_değer": False},
        {**row, "seviye": "⚡ Tier 4 — Çok yüksek risk (kısa kesin skor)", "tahmin": "1-0", "p": 0.18,
         "odd": 5.5, "değer": float("nan"), "gerçek_değer": False}])
    t5 = "🔥 Tier 5 — Aşırı risk (tüm maçlarda kesin skor)"
    extra = pd.DataFrame([  # E – F Tier 2'de; Tier 5'te A – B (Tier 4'te var), E – F ve mesajda olmayan G – H
        {**row, "maç": "E – F", "seviye": "🟡 Tier 2 — Orta risk", "tahmin": "MS 2", "p": 0.6, "odd": 1.5,
         "değer": -0.1, "gerçek_değer": False},
        {**row, "maç": "K – L", "seviye": "🟢 Tier 1 — Düşük risk", "tahmin": "MS 1", "p": 0.9, "odd": float("nan"),
         "değer": float("nan"), "gerçek_değer": False},  # İddaa'da kapalı: listeye girmez
        *[{**row, "maç": m, "seviye": t5, "tahmin": sc, "p": 0.12, "odd": 7.0, "değer": float("nan"),
           "gerçek_değer": False} for m, sc in [("A – B", "1-0"), ("E – F", "0-2"), ("G – H", "3-3")]]])
    msgs, _ = build_messages(pd.concat([picks, extra], ignore_index=True), {}, datetime(2026, 10, 3, 10, 0))
    allt = "\n".join(msgs)
    assert "TIER 5" in allt and "➜ 0-2 ·" in allt and "G – H" not in allt and "K – L" not in allt
    assert allt.count("➜ 1-0 ·") == 1 and "%" not in allt  # skor tekrar etmez; yüzde yok
    msgs, st = build_messages(picks, {}, datetime(2026, 10, 3, 10, 0))  # 10:00: günlük özet (tek mesaj)
    assert len(msgs) == 1 and "Günün tavsiyeleri" in msgs[0] and "💎" in msgs[0] and "TIER 1" in msgs[0]
    assert "TIER 4" in msgs[0] and "➜ 1-0 ·" in msgs[0] and "Tam rapor" in msgs[0] and "KUPON" not in msgs[0]
    # maç saati: bugün "21:45", yarın "Yarın 19:00"; tahmin satırı ayrı
    from src.telegram import _fmt
    r = {**row, "tahmin": "MS 1", "p": 0.7, "odd": 1.4, "değer": -0.02, "gerçek_değer": False, "mbs": 3}
    assert _fmt({**r, "baslama": "2026-10-03 21:45"}, datetime(2026, 10, 3, 10, 0)) == \
        "🕘 21:45 · <b>A – B</b>\n➜ MS 1 · <b>1.40</b> · MBS 3"
    assert _fmt({**r, "baslama": "2026-10-04 19:00"}, datetime(2026, 10, 3, 10, 0)).startswith("🕘 Yarın 19:00")
    assert "💎" in _fmt({**r, "gerçek_değer": True, "değer": 0.05}, datetime(2026, 10, 3, 10, 0))
    msgs, st = build_messages(picks, st, datetime(2026, 10, 3, 13, 0))  # aynı gün tekrar: hiçbir şey
    assert msgs == []
    new = pd.concat([picks, picks.iloc[[0]].assign(maç="C – D")], ignore_index=True)
    msgs, st = build_messages(new, st, datetime(2026, 10, 3, 16, 0))  # yeni 💎: anlık uyarı
    assert len(msgs) == 1 and "Yeni değerli bahis" in msgs[0] and "C – D" in msgs[0] and "A – B" not in msgs[0]
    msgs, st = build_messages(picks, st, datetime(2026, 10, 3, 18, 30))  # akşam özeti (💎 tekrar uyarı olarak gelmez)
    assert len(msgs) == 1 and "Akşam güncellemesi" in msgs[0]
    msgs, st = build_messages(picks, st, datetime(2026, 10, 3, 21, 0))  # aynı akşam tekrar: hiçbir şey
    assert msgs == []
    msgs, st2 = build_messages(picks, st, datetime(2026, 10, 3, 21, 5), force=True)  # elle istenen güncel özet
    assert len(msgs) == 1 and "Güncel tavsiyeler" in msgs[0] and st2["son_ozet"] == st["son_ozet"]
    msgs, _ = build_messages(picks, {}, datetime(2026, 10, 3, 7, 0))  # 07:00: özet yok ama 💎 uyarısı var
    assert len(msgs) == 1 and "Yeni değerli bahis" in msgs[0]
    msgs, _ = build_messages(picks, {"son_ozet": "2026-10-03"}, datetime(2026, 10, 3, 12, 0))  # eski durum biçimi
    assert "Günün tavsiyeleri" in msgs[0]


def test_kupon():
    """İY/MS olasılıkları maç sonuyla tutarlı; kombine ve sistem kupon hesabı."""
    from src import kupon
    from src.tavsiye import market_probs
    from tests.make_synthetic import dc_matrix
    m = dc_matrix(1.6, 1.0)
    iy = kupon.iyms_probs(m)
    ms = market_probs(m)
    assert abs(sum(iy.values()) - 1) < 1e-9 and len(iy) == 9
    for s in ("1", "X", "2"):  # maç sonu marjinali aynı
        assert abs(sum(v for k, v in iy.items() if k.endswith("/" + s)) - ms[f"MS {s}"]) < 1e-9
    assert iy["İY/MS 1/1"] > iy["İY/MS 2/1"] > 0
    assert kupon.iyms_won("İY/MS 2/1", 0, 1, 2, 1) and not kupon.iyms_won("İY/MS 1/1", 0, 0, 1, 0)
    # merdiven: İY/MS a/b, maç sonu b'yi içeren tahminlerin devamı; kesin skor maç sonuyla uyumlu
    from src.tavsiye import NARROWER, _ft_mask
    assert "İY/MS 1/1" in NARROWER["MS 1"] and "İY/MS X/1" in NARROWER["ÇŞ 1-X"] and "İY/MS 2/2" not in NARROWER["MS 1"]
    assert "İY/MS 1/1" not in NARROWER["MS 1 & KG Var"] and NARROWER["İY/MS 1/1"] == []
    assert (_ft_mask("İY/MS 2/1", 11) == _ft_mask("MS 1", 11)).all()
    ps, os_ = [0.7, 0.6, 0.8, 0.65, 0.75], [1.4, 1.6, 1.25, 1.5, 1.3]
    tab = {r["k"]: r for r in kupon.system_table(ps, os_)}
    assert abs(tab[5]["en_az_k"] - np.prod(ps)) < 1e-12 and abs(tab[5]["beklenen_donus"] - np.prod(np.multiply(ps, os_))) < 1e-12
    assert tab[2]["kolon"] == 10 and tab[2]["en_az_k"] > tab[4]["en_az_k"]


def test_analiz():
    """Maç analizi: İddaa detayından market adları ve skor tablosundan olasılıklar."""
    from src.analiz import corner_probs, detail_markets, goal_probs
    from tests.make_synthetic import dc_matrix
    detail = {"m": [{"t": 1, "st": 1, "o": [{"n": "1", "odd": 1.5}, {"n": "0", "odd": 4.0}, {"n": "2", "odd": 5.0}]},
                    {"t": 2, "st": 92, "o": [{"n": "1 ve 0", "odd": 1.1}, {"n": "0 ve 2", "odd": 2.1}]},
                    {"t": 2, "st": 101, "sov": "2.5", "o": [{"n": "Alt", "odd": 2.2}, {"n": "Üst", "odd": 1.6}]},
                    {"t": 2, "st": 7, "sov": "2.5", "o": [{"n": "1 ve Üst", "odd": 2.0}, {"n": "0 ve Alt", "odd": 6.0}]},
                    {"t": 2, "st": 698, "o": [{"n": "1 ve Var", "odd": 2.8}]},
                    {"t": 2, "st": 48, "sov": "7.5", "o": [{"n": "Alt", "odd": 2.6}, {"n": "Üst", "odd": 1.28}]}]}
    odds = detail_markets(detail)
    assert odds["MS X"] == 4.0 and odds["ÇŞ 1-X"] == 1.1 and odds["2.5 Üst"] == 1.6
    assert odds["MS 1 & 2.5 Üst"] == 2.0 and odds["MS X & 2.5 Alt"] == 6.0 and odds["MS 1 & KG Var"] == 2.8
    assert odds["Korner 7.5 Üst"] == 1.28
    p = goal_probs(dc_matrix(1.6, 1.0), list(odds) + ["2.5 Alt"])
    assert abs(p["2.5 Üst"] + p["2.5 Alt"] - 1) < 1e-9 and p["MS 1 & 2.5 Üst"] < p["MS 1"] < p["ÇŞ 1-X"]
    c = corner_probs(odds, None)  # milli: İddaa'nın marjsız oranı
    assert abs(c["Korner 7.5 Üst"] + c["Korner 7.5 Alt"] - 1) < 1e-9 and c["Korner 7.5 Üst"] > 0.6
    # kart / şut / faul / golcü
    from src import istatistik
    from src.analiz import stat_probs
    det = {"m": [{"t": 2, "st": 658, "sov": "4.5", "o": [{"n": "Alt", "odd": 1.97}, {"n": "Üst", "odd": 1.52}]},
                 {"t": 2, "st": 913, "o": [{"n": "İspanya 8+", "odd": 1.71}, {"n": "Çekya 3+", "odd": 1.76},
                                           {"n": "Toplam 10+", "odd": 1.48}, {"n": "İY İspanya 4+", "odd": 2.0}]},
                 {"t": 2, "st": 12, "o": [{"n": "Roberto Fernandez", "odd": 1.54}]}]}
    so = detail_markets(det, "İspanya", "Çekya")
    assert so["Kart 4.5 Üst"] == 1.52 and so["İsabetli şut Ev 8+"] == 1.71 and so["İsabetli şut Dep 3+"] == 1.76
    assert so["İsabetli şut Toplam 10+"] == 1.48 and so["Golcü: Roberto Fernandez"] == 1.54 and len(so) == 6
    sp, approx = stat_probs(so, {})  # milli: kart marjsız, X+ barajları ham
    assert abs(sp["Kart 4.5 Üst"] + sp["Kart 4.5 Alt"] - 1) < 1e-9 and "İsabetli şut Ev 8+" in approx
    pr = istatistik.probs(5.0, 3.5, "isabetli")  # kulüp: model
    sp, approx = stat_probs(so, {"isabetli": pr, "kart": istatistik.probs(2.2, 2.4, "kart")})
    assert abs(sp["İsabetli şut Toplam 10+"] - pr["toplam"][10]) < 1e-12 and "İsabetli şut Ev 8+" not in approx
    assert pr["toplam"][0] == 1.0 or abs(pr["toplam"][0] - 1) < 1e-9 and pr["1"] > pr["2"]


def test_korner():
    """Korner modeli: rho 0, dağılım tutarlılığı, bahis sonuçları; İddaa korner marketleri."""
    from src import korner
    from src.iddaa import parse_corners
    df, att, _ = _simulate(seasons=1, seed=3)
    rng = np.random.default_rng(3)
    order = {f"T{i:02d}": i for i in range(len(att))}
    df["hc"] = [rng.poisson(5.5 * np.exp(0.5 * att[order[h]])) for h in df["home"]]
    df["ac"] = [rng.poisson(4.5 * np.exp(0.5 * att[order[a]])) for a in df["away"]]
    m = korner.model(xi=0.0, ridge=1.0, ridge_total=1.0).fit(korner.corner_frame(df.assign(div="SP1")))
    assert m.rho == 0.0 and m.converged
    lam, nu = m.expected_goals("T00", "T01", "SP1")
    assert 3 < lam < 9 and 2 < nu < 8
    mat = korner.matrix(lam, nu)
    assert abs(mat.sum() - 1) < 1e-9
    p = korner.market_probs(mat, 9.5)
    assert abs(p["Korner 9.5 Üst"] + p["Korner 9.5 Alt"] - 1) < 1e-9
    assert abs(p["En çok korner 1"] + p["En çok korner X"] + p["En çok korner 2"] - 1) < 1e-9
    assert korner.bet_won("Korner 9.5 Üst", 6, 4) and not korner.bet_won("Korner 10.5 Üst", 6, 4)
    assert korner.bet_won("En çok korner 2", 3, 7) and korner.bet_won("En çok korner X", 5, 5)
    assert korner.is_corner_bet("Korner 9.5 Alt") and not korner.is_corner_bet("MS 1")
    detail = {"m": [{"t": 2, "st": 48, "sov": "11.5", "o": [{"n": "Alt", "odd": 1.16}, {"n": "Üst", "odd": 3.33}]},
                    {"t": 2, "st": 48, "sov": "10.5", "o": [{"n": "Alt", "odd": 1.5}, {"n": "Üst", "odd": 2.4}]},
                    {"t": 2, "st": 52, "o": [{"n": "1", "odd": 1.65}, {"n": "0", "odd": 7.28}, {"n": "2", "odd": 2.32}]},
                    {"t": 2, "st": 49, "sov": "5.5", "o": [{"n": "Alt", "odd": 1.2}, {"n": "Üst", "odd": 2.9}]}]}
    c = parse_corners(detail)
    assert (c["kor_line"], c["odd_kor_alt"], c["odd_kor_ust"]) == (10.5, 1.5, 2.4)  # en dengeli baraj
    assert (c["odd_kor_1"], c["odd_kor_x"], c["odd_kor_2"]) == (1.65, 7.28, 2.32)
    from src.iddaa import KOR_LINES_COL, corner_lines
    assert c[KOR_LINES_COL] == "10.5=1.5/2.4;11.5=1.16/3.33"  # tüm barajlar
    assert corner_lines("7.5=-/1.28;8.5=1.95/1.54")[8.5] == (1.95, 1.54) and np.isnan(corner_lines("7.5=-/1.28")[7.5][0])
    assert corner_lines(float("nan"), {"kor_line": 9.5, "odd_kor_alt": 1.8, "odd_kor_ust": 1.9}) == {9.5: (1.8, 1.9)}
    # Tier listesinde korner: milli maçta İddaa'nın marjsız oranı, kulüpte model; merdiven 7.5 Üst -> 8.5 Üst
    from src.tavsiye import _corner_cands, _corner_narrower
    row = {KOR_LINES_COL: "7.5=2.6/1.28;8.5=1.95/1.54", "odd_kor_1": 1.33, "odd_kor_x": 7.28, "odd_kor_2": 3.39,
           "home": "T00", "away": "T01", "div": "SP1"}
    nat = {x["tahmin"]: x for x in _corner_cands(row, {"maç": "A – B"}, None)}
    assert abs(nat["Korner 7.5 Üst"]["p"] + nat["Korner 7.5 Alt"]["p"] - 1) < 1e-9 and nat["Korner 7.5 Üst"]["p"] > 0.6
    assert nat["Korner 7.5 Üst"]["kaynak"] == "oran" and nat["En çok korner 1"]["odd"] == 1.33
    club = {x["tahmin"]: x for x in _corner_cands(row, {"maç": "A – B"}, m)}
    assert club["Korner 8.5 Üst"]["kaynak"] == "korner" and club["Korner 8.5 Üst"]["p"] < club["Korner 7.5 Üst"]["p"]
    assert _corner_narrower("Korner 8.5 Üst", "Korner 7.5 Üst") and not _corner_narrower("Korner 7.5 Üst", "Korner 8.5 Üst")
    assert _corner_narrower("Korner 8.5 Alt", "Korner 9.5 Alt") and not _corner_narrower("Korner 8.5 Alt", "Korner 7.5 Üst")


def test_sonuclar():
    """Gönderilen tahminlerin kaydı, sonuçla kapanması, seviye özeti ve gün mesajı."""
    import tempfile
    from datetime import datetime
    from src import sonuclar
    base = {"tarih": "01.10", "kaynak": "oran", "merdiven": False, "hg": np.nan, "ag": np.nan, "değer": -0.05}
    rows = [{**base, "gün": "2026-10-01", "div": "SP1", "home": "Betis", "away": "Getafe", "maç": "Betis – Getafe",
             "lig": "SP1", "seviye": "🟢 Tier 1 — Düşük risk", "tahmin": "ÇŞ 1-X", "p": 0.8, "odd": 1.25},
            {**base, "gün": "2026-10-01", "div": "SP1", "home": "Betis", "away": "Getafe", "maç": "Betis – Getafe",
             "lig": "SP1", "seviye": "🟡 Tier 2 — Orta risk", "tahmin": "MS 1", "p": 0.55, "odd": 1.7, "merdiven": True},
            {**base, "gün": "2026-10-02", "div": "MILLI", "home": "Spain", "away": "Czech Republic",
             "maç": "İspanya – Çekya", "lig": "UEFA Uluslar Ligi", "seviye": "🔥 Tier 5", "tahmin": "2-0", "p": 0.14,
             "odd": float("nan")},
            {**base, "gün": "2026-10-02", "div": "MILLI", "home": "Wales", "away": "Norway", "maç": "Galler – Norveç",
             "lig": "UEFA Uluslar Ligi", "seviye": "🔴 Tier 3 — Yüksek risk", "tahmin": "MS 2 & KG Var", "p": 0.3,
             "odd": 3.2},
            {**base, "gün": "2026-10-01", "div": "SP1", "home": "Betis", "away": "Getafe", "maç": "Betis – Getafe",
             "lig": "SP1", "seviye": "🚩 Korner — kulüp maçları", "tahmin": "Korner 9.5 Üst", "p": 0.55, "odd": 1.8},
            {**base, "gün": "2026-10-01", "div": "SP1", "home": "Betis", "away": "Getafe", "maç": "Betis – Getafe",
             "lig": "SP1", "seviye": "🔀 Tier 8 — İY/MS", "tahmin": "İY/MS X/X", "p": 0.2, "odd": 4.5},
            {**base, "gün": "2026-10-01", "div": "SP1", "home": "Betis", "away": "Getafe", "maç": "Betis – Getafe",
             "lig": "SP1", "seviye": "🎫 Kupon — orta risk", "tahmin": "ÇŞ 1-X", "p": 0.8, "odd": 1.25}]
    old_dir = config.TAVSIYE_DIR
    with tempfile.TemporaryDirectory() as d:
        config.TAVSIYE_DIR = Path(d)
        try:
            assert sonuclar.record(rows, datetime(2026, 10, 1, 9, 20)) == 7  # kupon ayağı Tier 1'den ayrı sayılır
            assert sonuclar.record(rows[:1], datetime(2026, 10, 1, 18, 20)) == 0  # akşam özetinde tekrar: sayılmaz
            log = sonuclar.load_log()
        finally:
            config.TAVSIYE_DIR = old_dir
    res = pd.DataFrame({"div": ["SP1", "MILLI", "MILLI"], "date": pd.to_datetime(["2026-10-01", "2026-10-03", "2026-09-20"]),
                        "home": ["Betis", "Spain", "Norway"], "away": ["Getafe", "Czech Republic", "Wales"],
                        "fthg": [1, 2, 1], "ftag": [1, 0, 1], "hc": [7, np.nan, np.nan],
                        "ac": [4, np.nan, np.nan], "hthg": [0, 1, np.nan], "htag": [0, 0, np.nan]})  # İspanya maçı ertesi gün (saat dilimi)
    log = sonuclar.settle(log, res)
    assert (log["hg"].iloc[:3].tolist(), log["ag"].iloc[:3].tolist()) == ([1, 1, 2], [1, 1, 0])
    assert np.isnan(log["hg"].iloc[3])  # Galler maçının sonucu henüz yok (eski karşılaşma tarih tutmuyor)
    assert (log["hk"].iloc[4], log["ak"].iloc[4]) == (7, 4)  # korner: 11 > 9.5
    s = {x["icon"]: x for x in sonuclar.stats(log)}
    assert (s["🟢"]["tutan"], s["🟡"]["tutan"], s["🔥"]["tutan"], s["🚩"]["tutan"]) == (1, 0, 1, 1)  # 🚩: bahis türü
    assert s["🔀"]["tutan"] == 1 and s["🎫"]["tutan"] == 1  # İY 0-0, MS 1-1: X/X tuttu
    msg = sonuclar.day_message("2026-10-01", log[log["gün"] == "2026-10-01"], log)
    assert "korner 7-4" in msg and "İY 0-0, MS 1-1" in msg and "🎫 Kupon (1 maç, oran 1.25): ✅ tuttu" in msg, msg
    assert abs(s["🟢"]["kâr_zarar"] - 0.25) < 1e-9 and s["🟡"]["kâr_zarar"] == -1 and s["🔴"]["sonuçlanan"] == 0
    msg = sonuclar.day_message("2026-10-01", log[log["gün"] == "2026-10-01"], log)
    assert "Sonuçlar — 01.10.2026" in msg and "✅ 🟢 Betis – Getafe: ÇŞ 1-X (1-1) @1.25" in msg and "❌" in msg
    md = sonuclar.summary_md(log)
    assert "## Tüm zamanlar" in md and "⏳" in md
    # rapor zamanı: gün tamamsa hemen, eksikse ertesi sabah 09:00'dan sonra (eksikler ⏳)
    due = lambda t: sonuclar.due_days(log, set(), pd.Timestamp(t))  # noqa: E731
    assert due("2026-10-02 12:00") == ["2026-10-01"]       # 01.10 tamam; 02.10 henüz bugün
    assert due("2026-10-03 08:00") == ["2026-10-01"]       # 02.10'da Galler maçı eksik, sabah 09:00 olmadı
    assert due("2026-10-03 09:30") == ["2026-10-01", "2026-10-02"]
    assert sonuclar.due_days(log, {"2026-10-01"}, pd.Timestamp("2026-10-03 09:30")) == ["2026-10-02"]


def test_xg_fit():
    """xG hedefiyle (tam sayı olmayan Poisson hedefi) eğitim yakınsar ve anlamlı güçler verir."""
    df, att, dfn = _simulate(seasons=1, seed=4)
    rng = np.random.default_rng(4)
    df["hxg"] = df["fthg"] * 0.3 + rng.gamma(2, 0.6, len(df))  # gürültülü ama bilgili
    df["axg"] = df["ftag"] * 0.3 + rng.gamma(2, 0.5, len(df))
    order = None
    for w_ in (0.0, 0.7):
        m = DixonColes(xi=0.0, ridge=0.1, ridge_total=0.1, lookback_days=5000, xg_weight=w_).fit(df)
        order = [int(t[1:]) for t in m.teams]
        assert m.converged and np.isfinite(m.attack).all()
    assert np.corrcoef(m.attack, att[order])[0, 1] > 0.5


def test_real_header_parsing():
    raw = db.read_raw_csv(ROOT / "tests" / "fixtures" / "real_header_sample.csv")
    df = db.normalize(raw, div="SP1", season=2025)
    r = df.iloc[0]
    assert r["date"] == pd.Timestamp("2025-08-15") and r["home"] == "Girona"
    assert (r["ps_h"], r["psc_h"], r["psc_a"], r["b365_o25"], r["psc_u25"]) == (2.3, 2.29, 3.45, 2.2, 1.72)


def test_end_to_end():
    """Sentetik veriyle: build -> backtest -> predict (+İddaa dosyası). Gerçek veriye dokunmaz."""
    tmp = ROOT / "tests" / "_e2e"
    shutil.rmtree(tmp, ignore_errors=True)
    env = {**os.environ, "VALUEBET_DATA_DIR": str(tmp / "data"), "VALUEBET_REPORTS_DIR": str(tmp / "reports"),
           "VALUEBET_TRACKING_DIR": str(tmp / "tracking"), "VALUEBET_TAVSIYE_DIR": str(tmp / "tavsiyeler"),
           "PYTHONIOENCODING": "utf-8"}
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True,  # noqa: E731
                                    text=True, encoding="utf-8", errors="replace", timeout=900)
    r = run("tests/make_synthetic.py", str(tmp / "data" / "raw"))
    assert r.returncode == 0, r.stderr
    # elle indirilmiş gibi isimlendirilmiş dosya da okunabilmeli
    raw = tmp / "data" / "raw"
    (raw / "SP1_2526.csv").rename(raw / "SP1 (3).csv")
    r = run("main.py", "build")
    assert r.returncode == 0, r.stderr
    assert "SP1 (3).csv: 380 maç (SP1 2025/26)" in r.stdout, r.stdout
    assert "Oran doluluğu" in r.stdout, r.stdout
    assert "xG:" in r.stdout and "maça eklendi" in r.stdout and "gol tutmadı" not in r.stdout, r.stdout
    r = run("main.py", "milli", "--eval-from", "2024")  # milli takım modeli: doğrulama raporu
    assert r.returncode == 0, r.stderr
    rep = (tmp / "reports" / "milli_summary.md").read_text(encoding="utf-8")
    assert "Kalibrasyon" in rep and "| tümü |" in rep, rep[:800]
    assert "Basque Country" not in pd.read_csv(tmp / "reports" / "milli_preds.csv")["home"].values  # FIFA dışı
    r = run("main.py", "backtest", "--no-joint", "--div", "SP1", "--eval-from", "2024")
    assert r.returncode == 0, r.stderr
    report = (tmp / "reports" / "backtest_summary.md").read_text(encoding="utf-8")
    assert "Log loss farkı (model − Keskin kapanış PSC/BFEC)" in report and "Strateji karşılaştırması" in report
    assert "BFEC" in report.split("Kapanış kaynağı:")[1].splitlines()[0], "son sezon borsa kapanışına geçmeli"
    base = tmp / "baz"
    base.mkdir()
    shutil.copy(tmp / "reports" / "backtest_preds_SP1.csv", base)  # lige özel modelin tahminleri
    r = run("main.py", "backtest", "--joint", "--eval-from", "2026")
    assert r.returncode == 0, r.stderr
    report = (tmp / "reports" / "backtest_summary.md").read_text(encoding="utf-8")
    assert report.count("model=birleşik") == 2, report[:500]
    r = run("main.py", "diagnose", "--div", "SP1", "--base", str(base))
    assert r.returncode == 0, r.stderr
    assert "fark (şimdi−baz)" in (tmp / "reports" / "diagnose_summary.md").read_text(encoding="utf-8")
    r = run("main.py", "backtest", "--joint", "--motivation", "--eval-from", "2026")
    assert r.returncode == 0, r.stderr
    assert "mot" in pd.read_csv(tmp / "reports" / "backtest_preds_SP1.csv", nrows=1).columns
    r = run("main.py", "tune", "--div", "SP1", "--eval-from", "2024", "--tune-to", "2024", "--workers", "2",
            "--xi-grid", str(config.XI), "--ridge-grid", str(config.RIDGE + 1),
            "--ridge-total-grid", str((config.RIDGE_TOTAL or config.RIDGE) + 1))
    assert r.returncode == 0, r.stderr
    assert "Test döneminde" in (tmp / "reports" / "tune_summary.md").read_text(encoding="utf-8")
    assert len(pd.read_csv(tmp / "reports" / "tune_grid.csv")) == 4  # {ridge, ridge+1} x {total, total+1}
    iddaa = tmp / "iddaa.csv"
    iddaa.write_text("home,away,odd_1,odd_x,odd_2\nLiga Takim 01,liga takım 02,2.1,3.3,3.4\n", encoding="utf-8")
    r = run("main.py", "predict", "--motivation", "--fixtures", str(tmp / "data" / "raw" / "fixtures.csv"),
            "--iddaa", str(iddaa))
    assert r.returncode == 0, r.stderr
    assert "İddaa oranlarına göre" in r.stdout, r.stdout[-2000:]
    rec = next((tmp / "reports").glob("oneriler_*.md")).read_text(encoding="utf-8")
    assert all(s in rec for s in ("## 1) Keskin fiyat sinyali", "## 2) VALUE", "## 3) En olası sonuçlar")), rec[:500]
    pred = pd.read_csv(next((tmp / "reports").glob("predictions_*.csv")))
    fx = pred[pred["kaynak"] == "fixtures"]
    assert fx["max_edge_keskin"].notna().any(), "fikstürde Pinnacle yok; borsa fiyatı kullanılmalı"

    # canlı takip: yaklaşan maçlar kaydedilir; ikinci çalıştırma aynı maçları tekrar eklemez
    fixtures = str(tmp / "data" / "raw" / "fixtures.csv")
    r = run("main.py", "track", "--fixtures", fixtures)
    assert r.returncode == 0, r.stderr
    trk = tmp / "tracking"
    tp = pd.read_csv(trk / "predictions.csv")
    assert len(tp) > 0 and tp["fthg"].isna().all(), tp.head()
    # geçmişte oynanmış bir maç için elle kayıt ekle: sonraki çalıştırmada kapanmalı
    played = pd.read_csv(tmp / "data" / "raw" / "SP1_2425.csv").iloc[0]
    old = tp.iloc[[0]].assign(div="SP1", season=2024, home=played["HomeTeam"], away=played["AwayTeam"])
    pd.concat([tp, old]).to_csv(trk / "predictions.csv", index=False)
    bet = pd.DataFrame([{"recorded_at": "2024-08-01 00:00", "model": "test", "div": "SP1", "season": 2024,
                         "home": played["HomeTeam"], "away": played["AwayTeam"], "date": "2024-08-17",
                         "market": "1X2", "pick": "1", "price": 2.0, "p_model": 0.5, "kural_model": True,
                         "kural_sharp": False, "kural_both": False, "p_sharp": 0.45, "edge_model": 0.0,
                         "edge_keskin": -0.1}])
    bets_path = trk / "bets.csv"
    (pd.concat([pd.read_csv(bets_path), bet]) if bets_path.exists() else bet).to_csv(bets_path, index=False)
    # İddaa oran geçmişi: aynı maç için iki kayıt (ilk ve son) + başladıktan sonraki kayıt (sayılmamalı)
    kick = pd.to_datetime(played["Date"], dayfirst=True).strftime("%Y-%m-%d") + " 19:00"
    iddaa_dir = tmp / "data" / "raw" / "iddaa"
    iddaa_dir.mkdir(parents=True)
    pd.DataFrame([{"snapshot_utc": s, "event_id": 7, "div": "SP1", "kickoff_utc": kick, "home_iddaa": "x",
                   "away_iddaa": "y", "mbs": 1, "odd_1": o1, "odd_x": 3.3, "odd_2": 3.1, "ou_line": 2.5,
                   "odd_alt": 1.8, "odd_ust": 1.9, "home": played["HomeTeam"], "away": played["AwayTeam"],
                   "season": 2024} for s, o1 in [("2024-08-10 09:00", 2.2), ("2024-08-16 09:00", 2.0),
                                                 ("2099-01-01 00:00", 9.9)]]).to_csv(iddaa_dir / "odds_2024.csv",
                                                                                    index=False)
    r = run("main.py", "track", "--fixtures", fixtures)
    assert r.returncode == 0, r.stderr
    tp2 = pd.read_csv(trk / "predictions.csv")
    assert len(tp2) == len(tp) + 1, (len(tp2), len(tp))  # tekrar eklenmedi
    done = tp2[(tp2["season"] == 2024) & (tp2["home"] == played["HomeTeam"])].iloc[0]
    assert done["fthg"] == played["FTHG"] and np.isfinite(done["close_h"])
    tb = pd.read_csv(bets_path)
    b = tb[tb["model"] == "test"].iloc[0]
    assert b["won"] == float(played["FTHG"] > played["FTAG"]) and np.isfinite(b["clv"])
    assert b["profit"] == (1.0 if b["won"] else -1.0)
    assert "Kağıt bahisler" in (trk / "summary.md").read_text(encoding="utf-8")
    ic = pd.read_csv(trk / "iddaa_clv.csv")
    one = ic[(ic["market"] == "1X2") & (ic["pick"] == "1")].iloc[0]
    assert (one["odd_ilk"], one["odd_son"], one["kayıt_sayısı"]) == (2.2, 2.0, 2)  # başladıktan sonraki sayılmaz
    assert abs(one["clv_ilk"] - (2.2 * done["close_h"] - 1)) < 1e-9
    assert set(ic["pick"]) == {"1", "X", "2", "Üst", "Alt"}
    assert "İddaa fiyatlarının değeri" in (trk / "summary.md").read_text(encoding="utf-8")

    # predict --iddaa auto: yaklaşan bir maç için otomatik toplanmış İddaa oranı kullanılır
    up = pd.read_csv(fixtures).iloc[0]
    kick_up = pd.to_datetime(up["Date"], dayfirst=True)
    season_up = kick_up.year if kick_up.month >= 7 else kick_up.year - 1
    pd.DataFrame([{"snapshot_utc": "2000-01-01 00:00", "event_id": 8, "div": up["Div"],
                   "kickoff_utc": f"{kick_up:%Y-%m-%d} 20:00", "mbs": 1, "odd_1": 2.5, "odd_x": 3.2, "odd_2": 2.9,
                   "ou_line": 1.5, "odd_alt": 2.4, "odd_ust": 1.5, "home": up["HomeTeam"], "away": up["AwayTeam"],
                   "season": season_up, "odd_skor": "yok",  # İddaa'da skor bahsi açık değil
                   "kor_line": 9.5, "odd_kor_alt": 1.85, "odd_kor_ust": 1.85, "odd_kor_1": 1.6, "odd_kor_x": 7.0,
                   "odd_kor_2": 2.6},
                  # milli maç: Türkçe adlar ülke adlarına eşlenir; keskin oran yok -> İddaa oranına ayarlı
                  {"snapshot_utc": "2000-01-01 00:00", "event_id": 9, "div": config.MILLI,
                   "kickoff_utc": f"{kick_up + pd.Timedelta(days=1):%Y-%m-%d} 18:45", "mbs": 1, "odd_1": 1.35, "odd_x": 4.6, "odd_2": 8.0,
                   "ou_line": 2.5, "odd_alt": 2.0, "odd_ust": 1.7, "home_iddaa": "İspanya", "away_iddaa": "Çekya",
                   "competition": "UEFA Uluslar Ligi", "season": season_up,
                   "odd_skor": ";".join(f"{h}:{a}=9" for h in range(6) for a in range(6))}]
                 ).to_csv(iddaa_dir / f"odds_{season_up}.csv", index=False)
    r = run("main.py", "predict", "--fixtures", fixtures, "--iddaa", "auto")
    assert r.returncode == 0, r.stderr
    assert "otomatik toplanan oranlar, 1 maç" in r.stdout and "İddaa oranlarına göre" in r.stdout, r.stdout[-1500:]
    r = run("main.py", "tavsiye", "--fixtures", fixtures, "--yayinla")
    assert r.returncode == 0, r.stderr
    assert all(s in r.stdout for s in ("Tier 1", "Tier 5", "Değeri pozitif")), r.stdout[-1500:]
    son = (tmp / "tavsiyeler" / "son.md").read_text(encoding="utf-8")
    assert "Tier 1" in son and "İddaa oranlarının son kaydı" in son, son[:300]
    assert "İspanya – Çekya (UEFA Uluslar Ligi" in son and "İddaa oranına ayarlı" in son, son[-2500:]  # İddaa'daki adlar
    picks = pd.read_csv(tmp / "tavsiyeler" / "son.csv", dtype={"tarih": str})
    assert picks["tarih"].str.fullmatch(r"\d{2}\.\d{2}").all()  # "01.10" sayı olarak okunmamalı
    nat = picks[picks["maç"] == "İspanya – Çekya"]
    assert len(nat) and (nat["kaynak"] == "oran").all() and not nat["gerçek_değer"].any()  # milli maçta 💎 yok
    from src.tavsiye import MAX_PER_TIER, NARROWER, TIERS
    last_bet = {}  # merdiven: ↳ satırı aynı maçın bir önceki seviyedeki tahmininin daha dar devamı
    for k, (name, _, _) in enumerate(TIERS):
        tier = picks[picks["seviye"] == name]
        assert (tier["merdiven"] == True).sum() <= MAX_PER_TIER[k] // 2  # noqa: E712
        for maç, bet, lad in zip(tier["maç"], tier["tahmin"], tier["merdiven"]):
            if lad is True:
                assert bet in NARROWER[last_bet[maç]], (maç, last_bet[maç], bet)
        last_bet.update(zip(tier["maç"], tier["tahmin"]))
    t5 = picks[picks["seviye"].str.startswith("🔥")]
    nat5 = t5.loc[t5["maç"] == "İspanya – Çekya", "odd"]
    assert len(nat5) == 1 and nat5.iloc[0] == 9.0  # skor oranı İddaa'dan
    assert f"{up['HomeTeam']} – {up['AwayTeam']}" not in set(t5["maç"])  # skor bahsi kapalı: kesin skor önerilmez
    from src import korner as kmod
    kor = picks[picks["tahmin"].map(kmod.is_corner_bet)]  # korner Tier 1-3'te, İddaa korner oranıyla
    assert (kor["maç"] == f"{up['HomeTeam']} – {up['AwayTeam']}").any() and kor["seviye"].str[0].isin(
        ["🟢", "🟡", "🔴"]).all(), kor[["seviye", "tahmin", "odd"]]
    assert "korner modeli" in son
    ms1 = nat.loc[nat["tahmin"] == "MS 1", "p"]
    assert ms1.empty or abs(ms1.iloc[0] - devig([1.35, 4.6, 8.0], config.DEVIG_METHOD)[0]) < 1e-4
    day9 = f"{kick_up + pd.Timedelta(days=1):%Y-%m-%d}"  # milli maç 18:45 UTC = 21:45 TR
    r = run("main.py", "analiz", "--saat", "21:45", "--gun", day9)  # sadece o saatteki maçların Tier listesi
    assert r.returncode == 0 and "21:45 maçları" in r.stdout and "İspanya – Çekya" in r.stdout, r.stdout[-1500:]
    assert "%" not in r.stdout and up["HomeTeam"] not in r.stdout
    r = run("main.py", "sonuc", "--gonderme")  # henüz Telegram'da gönderilen tahmin yok
    assert r.returncode == 0 and "henüz kayıtlı tahmin yok" in r.stdout, r.stdout + r.stderr
    r = run("main.py", "tavsiye", "--prova", "2025-09-01", "--gun", "7")  # geçmiş hafta: sonuçlarla karşılaştırma
    assert r.returncode == 0, r.stderr
    assert "Sonuç özeti" in r.stdout and ("✅" in r.stdout or "❌" in r.stdout), r.stdout[-1500:]
    assert "Milli maçlar" in r.stdout and "(UEFA Nations League" in r.stdout, r.stdout[-1500:]
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("Tüm testler geçti.")
