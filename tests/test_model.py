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
    # ridge = ridge_total iken ceza eski haliyle aynı: ridge * (Σa² + Σd²)
    a, d = th[3:3 + n], th[3 + n:]
    pen = (_objective(th, hi, ai, x, y, w, n, 2.0, 2.0)[0] - _objective(th, hi, ai, x, y, w, n, 0.0, 0.0)[0]) * w.sum()
    assert abs(pen - 2.0 * (a @ a + d @ d)) < 1e-9, pen


def test_ridge_total_shrinks_totals():
    df, _, _ = _simulate(seasons=2)
    m = DixonColes(xi=0.0, ridge=0.5, ridge_total=1e6, lookback_days=5000).fit(df)
    assert np.abs(m.attack + m.defence).max() < 1e-3  # eğilim sıfıra çekildi
    assert np.abs(m.attack - m.defence).std() > 0.1  # güç farkları duruyor


def test_parameter_recovery():
    df, att, dfn = _simulate()
    model = DixonColes(xi=0.0, ridge=0.1, lookback_days=5000).fit(df)
    assert abs(model.home - 0.28) < 0.08, model.home
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
    r = run("main.py", "backtest", "--div", "SP1", "--eval-from", "2024")
    assert r.returncode == 0, r.stderr
    report = (tmp / "reports" / "backtest_summary.md").read_text(encoding="utf-8")
    assert "Log loss farkı (model − Keskin kapanış PSC/BFEC)" in report and "Strateji karşılaştırması" in report
    assert "BFEC" in report.split("Kapanış kaynağı:")[1].splitlines()[0], "son sezon borsa kapanışına geçmeli"
    r = run("main.py", "tune", "--div", "SP1", "--eval-from", "2024", "--tune-to", "2024", "--workers", "2",
            "--xi-grid", str(config.XI), "--ridge-grid", str(config.RIDGE + 1),
            "--ridge-total-grid", str((config.RIDGE_TOTAL or config.RIDGE) + 1))
    assert r.returncode == 0, r.stderr
    assert "Test döneminde" in (tmp / "reports" / "tune_summary.md").read_text(encoding="utf-8")
    assert len(pd.read_csv(tmp / "reports" / "tune_grid.csv")) == 4  # {ridge, ridge+1} x {total, total+1}
    iddaa = tmp / "iddaa.csv"
    iddaa.write_text("home,away,odd_1,odd_x,odd_2\nLiga Takim 01,liga takım 02,2.1,3.3,3.4\n", encoding="utf-8")
    r = run("main.py", "predict", "--fixtures", str(tmp / "data" / "raw" / "fixtures.csv"), "--iddaa", str(iddaa))
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
