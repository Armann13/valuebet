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
    from src.tavsiye import bet_won
    assert bet_won("MS 1 & KG Var", 2, 1) and not bet_won("MS 1 & KG Var", 2, 0)
    assert bet_won("ÇŞ X-2", 1, 1) and bet_won("2.5 Alt", 1, 1) and bet_won("1-1", 1, 1) and not bet_won("2-0", 2, 1)


def test_telegram_messages():
    from datetime import datetime
    from src.telegram import build_messages
    row = {"maç": "A – B", "lig": "E3", "tarih": "03.10", "gün": "2026-10-03", "mbs": 1, "kaynak": "piyasa"}
    picks = pd.DataFrame([
        {**row, "seviye": "💎 Değeri pozitif olanlar", "tahmin": "MS 1", "p": 0.55, "odd": 2.0, "değer": 0.10,
         "gerçek_değer": True},
        {**row, "seviye": "🟢 Tier 1 — Düşük risk", "tahmin": "ÇŞ 1-X", "p": 0.80, "odd": float("nan"),
         "değer": float("nan"), "gerçek_değer": False},
        {**row, "seviye": "⚡ Tier 4 — Çok yüksek risk (kısa kesin skor)", "tahmin": "1-0", "p": 0.18,
         "odd": float("nan"), "değer": float("nan"), "gerçek_değer": False}])
    msgs, st = build_messages(picks, {}, datetime(2026, 10, 3, 10, 0))  # 10:00: günlük özet
    assert len(msgs) == 1 and "Günün tavsiyeleri" in msgs[0] and "💎" in msgs[0] and "Tier 1" in msgs[0]
    assert "Tier 4" in msgs[0] and "<b>1-0</b>" in msgs[0]  # kesin skor da özette
    msgs, st = build_messages(picks, st, datetime(2026, 10, 3, 13, 0))  # aynı gün tekrar: hiçbir şey
    assert msgs == []
    new = pd.concat([picks, picks.iloc[[0]].assign(maç="C – D")], ignore_index=True)
    msgs, st = build_messages(new, st, datetime(2026, 10, 3, 16, 0))  # yeni 💎: anlık uyarı
    assert len(msgs) == 1 and "Yeni değerli bahis" in msgs[0] and "C – D" in msgs[0] and "A – B" not in msgs[0]
    msgs, _ = build_messages(picks, {}, datetime(2026, 10, 3, 7, 0))  # 07:00: özet yok ama 💎 uyarısı var
    assert len(msgs) == 1 and "Yeni değerli bahis" in msgs[0]


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
                   "season": season_up},
                  # milli maç: Türkçe adlar ülke adlarına eşlenir; keskin oran yok -> İddaa oranına ayarlı
                  {"snapshot_utc": "2000-01-01 00:00", "event_id": 9, "div": config.MILLI,
                   "kickoff_utc": f"{kick_up + pd.Timedelta(days=1):%Y-%m-%d} 18:45", "mbs": 1, "odd_1": 1.35, "odd_x": 4.6, "odd_2": 8.0,
                   "ou_line": 2.5, "odd_alt": 2.0, "odd_ust": 1.7, "home_iddaa": "İspanya", "away_iddaa": "Çekya",
                   "competition": "UEFA Uluslar Ligi", "season": season_up}]
                 ).to_csv(iddaa_dir / f"odds_{season_up}.csv", index=False)
    r = run("main.py", "predict", "--fixtures", fixtures, "--iddaa", "auto")
    assert r.returncode == 0, r.stderr
    assert "otomatik toplanan oranlar, 1 maç" in r.stdout and "İddaa oranlarına göre" in r.stdout, r.stdout[-1500:]
    r = run("main.py", "tavsiye", "--fixtures", fixtures, "--yayinla")
    assert r.returncode == 0, r.stderr
    assert all(s in r.stdout for s in ("Tier 1", "Tier 5", "Değeri pozitif")), r.stdout[-1500:]
    son = (tmp / "tavsiyeler" / "son.md").read_text(encoding="utf-8")
    assert "Tier 1" in son and "İddaa oranlarının son kaydı" in son, son[:300]
    assert "Spain – Czech Republic (UEFA Uluslar Ligi" in son and "İddaa oranına ayarlı" in son, son[-2500:]
    picks = pd.read_csv(tmp / "tavsiyeler" / "son.csv")
    nat = picks[picks["maç"] == "Spain – Czech Republic"]
    assert len(nat) and (nat["kaynak"] == "oran").all() and not nat["gerçek_değer"].any()  # milli maçta 💎 yok
    ms1 = nat.loc[nat["tahmin"] == "MS 1", "p"]
    assert ms1.empty or abs(ms1.iloc[0] - devig([1.35, 4.6, 8.0], config.DEVIG_METHOD)[0]) < 1e-4
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
