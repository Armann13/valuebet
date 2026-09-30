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
    assert list(s["sharp_src_1x2"]) == ["ps", "bfe", None, None]
    assert s.loc[0, "sharp_d"] == 3.4 and s.loc[1, "sharp_d"] == 3.6  # bir marketin oranları tek kaynaktan
    assert np.isnan(s.loc[2:, ["sharp_h", "sharp_d", "sharp_a"]].to_numpy()).all()
    assert list(s["sharp_src_ou"][:2]) == ["ps", "ps"] and np.isnan(s["sharpc_h"]).all()


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
    pred = pd.read_csv(next((tmp / "reports").glob("predictions_*.csv")))
    fx = pred[pred["kaynak"] == "fixtures"]
    assert fx["max_edge_keskin"].notna().any(), "fikstürde Pinnacle yok; borsa fiyatı kullanılmalı"
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
    print("Tüm testler geçti.")
