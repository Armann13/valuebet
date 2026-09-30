"""Value bet sistemi — komut satırı.

    python main.py download          # football-data.co.uk'tan sezon dosyalarını indir
    python main.py build             # CSV'leri SQLite veritabanına yükle
    python main.py backtest          # walk-forward backtest + rapor (reports/backtest_summary.md)
    python main.py tune              # XI / ridge ayarı (reports/tune_summary.md)
    python main.py diagnose          # açık nerede yoğunlaşıyor (reports/diagnose_summary.md)
    python main.py predict           # yaklaşan maçlar için olasılıklar ve value adayları
    python main.py predict --iddaa iddaa.csv
    python main.py ratings           # güncel takım güç sıralaması
    python main.py teams             # veritabanındaki takım isimleri (İddaa eşleştirmesi için)
    python main.py all               # download + build + backtest
"""
import argparse
import os
import sys
import time

import config

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def cmd_download(args):
    from src.download import download_fixtures, download_seasons
    print("Sezon dosyaları indiriliyor ...")
    ok = download_seasons(force=args.force)
    if ok:
        print("Yaklaşan maçlar indiriliyor ...")
        download_fixtures()
    return ok


def cmd_build(args):
    from src import db
    print("Veritabanı oluşturuluyor ...")
    n = db.build()
    print(f"Toplam {n} maç -> {config.DB_PATH}")
    if n:
        print("\nOran doluluğu (%):")
        print(db.coverage(db.load_matches()).to_string(index=False))


def cmd_backtest(args):
    from src import backtest, db
    divs = [args.div] if args.div else list(config.LEAGUES)
    parts = ["# Backtest raporu", ""]
    t0 = time.time()
    kw = dict(xi=args.xi, ridge=args.ridge, edge_min=args.edge, bet_book=args.book, eval_from=args.eval_from,
              ridge_total=args.ridge_total)
    if args.joint:  # tek walk-forward, tüm ligler birlikte; rapor lig lig
        print("  [birleşik] walk-forward başlıyor ...")
        preds = backtest.walk_forward(db.load_matches(), args.eval_from, args.xi, args.ridge,
                                      ridge_total=args.ridge_total)
        for div in divs:
            parts.append(backtest.run(div, None, preds=preds[preds["div"] == div].reset_index(drop=True), **kw))
    else:
        for div in divs:
            df = db.load_matches(div)
            if df.empty:
                print(f"  [{div}] veri yok, atlanıyor")
                continue
            parts.append(backtest.run(div, df, **kw))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "backtest_summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n" + "\n".join(parts))
    print(f"\nRapor: {out}  ({time.time() - t0:.0f} sn)")


def cmd_tune(args):
    from src import tune
    divs = [args.div] if args.div else list(config.LEAGUES)
    grid = lambda s: [float(v) for v in s.split(",")] if s else None  # noqa: E731
    md, table = tune.run(divs, args.xi, args.ridge, args.ridge_total, args.eval_from, args.tune_to, args.workers,
                         grid(args.xi_grid), grid(args.ridge_grid), grid(args.ridge_total_grid), args.criterion,
                         joint=args.joint)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "tune_summary.md"
    out.write_text(md, encoding="utf-8")
    table.to_csv(config.REPORTS_DIR / "tune_grid.csv", index=False)
    print("\n" + md)
    print(f"\nRapor: {out}")


def cmd_diagnose(args):
    from src import diagnose
    md = diagnose.run([args.div] if args.div else list(config.LEAGUES), args.base)
    out = config.REPORTS_DIR / "diagnose_summary.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nRapor: {out}")


def cmd_predict(args):
    from src import predict
    res = predict.run(args.fixtures, args.iddaa, xi=args.xi, ridge=args.ridge, ridge_total=args.ridge_total,
                      joint=args.joint)
    predict.print_table(res)


def cmd_ratings(args):
    from src import db
    from src.dixon_coles import DixonColes
    kw = dict(xi=args.xi, ridge=args.ridge, ridge_total=args.ridge_total)
    joint = DixonColes(**kw).fit(db.load_matches()) if args.joint else None
    for div in ([args.div] if args.div else list(config.LEAGUES)):
        df = db.load_matches(div)
        if df.empty:
            continue
        cur = df[df["season"] == df["season"].max()]
        m = joint if joint is not None else DixonColes(**kw).fit(df)
        r = m.ratings()
        r = r[r["team"].isin(set(cur["home"]) | set(cur["away"]))]
        extra = (f", lig gücü {m.league_strength[div]:+.3f}, yeni gelen sapması: çıkan "
                 f"{m.newcomer_offset['çıkan']:+.3f} / düşen {m.newcomer_offset['düşen']:+.3f}") if joint else ""
        print(f"\n=== {config.LEAGUES[div]} — ev avantajı {m.home[div]:.3f}, rho {m.rho:.3f}{extra} ===")
        print(r.round(3).to_string(index=False))


def cmd_teams(args):
    from src import db
    df = db.load_matches()
    last = df[df["season"] >= df["season"].max() - 1]
    for div, g in last.groupby("div"):
        print(f"\n{config.LEAGUES.get(div, div)}: " + ", ".join(sorted(set(g["home"]))))


def cmd_all(args):
    cmd_download(args)
    cmd_build(args)
    cmd_backtest(args)


def main():
    p = argparse.ArgumentParser(description="La Liga value bet sistemi (Faz 0-1)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--xi", type=float, default=config.XI, help="zaman sönümü (günlük)")
        sp.add_argument("--ridge", type=float, default=config.RIDGE, help="güç yönünde L2 çekim katsayısı")
        sp.add_argument("--ridge-total", type=float, default=config.RIDGE_TOTAL, dest="ridge_total",
                        help="eğilim (toplam gol) yönünde L2 çekimi (verilmezse config)")
        sp.add_argument("--div", choices=list(config.LEAGUES), help="tek lig (varsayılan: hepsi)")
        sp.add_argument("--joint", action=argparse.BooleanOptionalAction, default=config.JOINT,
                        help="ligleri tek modelde birlikte eğit (--no-joint: her lig ayrı)")

    for name, fn in [("download", cmd_download), ("build", cmd_build), ("backtest", cmd_backtest),
                     ("tune", cmd_tune), ("diagnose", cmd_diagnose), ("predict", cmd_predict),
                     ("ratings", cmd_ratings), ("teams", cmd_teams), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
        common(sp)
        if name in ("download", "all"):
            sp.add_argument("--force", action="store_true", help="bitmiş sezonları da yeniden indir")
        if name in ("backtest", "all"):
            sp.add_argument("--edge", type=float, default=config.EDGE_MIN, help="min edge (0.03 = %%3)")
            sp.add_argument("--book", default=config.BET_BOOK, choices=["b365", "ps", "avg", "max"],
                            help="backtest'te bahis oynanan oranlar")
            sp.add_argument("--eval-from", type=int, default=config.EVAL_FROM_SEASON, dest="eval_from")
        if name == "tune":  # --xi / --ridge burada kıyas alınan varsayılan ayardır
            sp.add_argument("--eval-from", type=int, default=config.EVAL_FROM_SEASON, dest="eval_from")
            sp.add_argument("--tune-to", type=int, default=2021, dest="tune_to",
                            help="seçim bu sezona kadarki maçlarla yapılır; sonrası test dönemi")
            sp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
            sp.add_argument("--xi-grid", dest="xi_grid", help="virgülle ayrılmış xi değerleri (ör. 0.001,0.002)")
            sp.add_argument("--ridge-grid", dest="ridge_grid", help="virgülle ayrılmış ridge değerleri")
            sp.add_argument("--ridge-total-grid", dest="ridge_total_grid",
                            help="virgülle ayrılmış ridge_total değerleri (verilmezse ridge_total sabit)")
            sp.add_argument("--criterion", default="toplam", choices=["1x2", "ou", "toplam"],
                            help="seçim ölçütü: 1X2, Ü/A ya da ikisinin toplamı log loss")
        if name == "diagnose":
            sp.add_argument("--base", help="kıyas için baz tahminlerin klasörü (içinde backtest_preds_*.csv)")
        if name == "predict":
            sp.add_argument("--fixtures", help="yerel fixtures.csv yolu (verilmezse indirilir)")
            sp.add_argument("--iddaa", help="İddaa oranlarını içeren CSV (bkz. iddaa_ornek.csv)")
    args = p.parse_args()
    if args.fn(args) is False:  # ör. siteye ulaşılamadı; GitHub Actions işi başarısız görünsün
        sys.exit(1)


if __name__ == "__main__":
    main()
