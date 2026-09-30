"""Value bet sistemi — komut satırı.

    python main.py download          # football-data.co.uk'tan sezon dosyalarını indir
    python main.py build             # CSV'leri SQLite veritabanına yükle
    python main.py backtest          # walk-forward backtest + rapor (reports/backtest_summary.md)
    python main.py tune              # XI / ridge ayarı (reports/tune_summary.md)
    python main.py diagnose          # açık nerede yoğunlaşıyor (reports/diagnose_summary.md)
    python main.py predict           # yaklaşan maçlar için olasılıklar ve value adayları
    python main.py predict --iddaa iddaa.csv   # ya da --iddaa auto (otomatik toplanan oranlar)
    python main.py track             # kağıt üzerinde canlı takip (tracking/)
    python main.py iddaa             # İddaa oranlarını topla (data/raw/iddaa/)
    python main.py tavsiye           # risk seviyesine göre tavsiyeler (reports/tavsiye_{tarih}.md)
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
    from src.download import download_fixtures, download_seasons, download_xg
    print("Sezon dosyaları indiriliyor ...")
    ok = download_seasons(force=args.force)
    if ok:
        print("Yaklaşan maçlar indiriliyor ...")
        download_fixtures()
        print("xG (Understat) indiriliyor ...")
        download_xg(force=args.force)  # başarısızlığı işi durdurmaz (ikincil veri)
    return ok


def cmd_build(args):
    from src import db
    print("Veritabanı oluşturuluyor ...")
    n = db.build()
    print(f"Toplam {n} maç -> {config.DB_PATH}")
    if n:
        df = db.load_matches()
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        db.coverage(df).to_csv(config.REPORTS_DIR / "coverage.csv", index=False)
        print("\nOran doluluğu (%, tüm sezonlar; sezon sezon: reports/coverage.csv):")
        print(db.coverage_summary(df).to_string(index=False))


def _matches(args, div: str | list | None = None):
    """Maçlar (div: tek lig, lig listesi ya da hepsi); --motivation açıksa maç öncesi puan durumundan
    motivasyon sütunu (mot) eklenir."""
    from src import db
    from src.features import add_motivation
    df = db.load_matches(div)
    return add_motivation(df) if args.motivation and not df.empty else df


def _groups(divs: list) -> dict:
    """İstenen liglerin grupları; eğitim her zaman grubun tüm ligleriyle yapılır."""
    return {g: gd for g, gd in config.LEAGUE_GROUPS.items() if set(gd) & set(divs)}


def cmd_backtest(args):
    import pandas as pd
    from src import backtest, db
    divs = [args.div] if args.div else db.available_divs()
    parts = ["# Backtest raporu", ""]
    t0 = time.time()
    kw = dict(xi=args.xi, ridge=args.ridge, edge_min=args.edge, bet_book=args.book, eval_from=args.eval_from,
              ridge_total=args.ridge_total)
    if args.joint:  # her ülke grubu tek birleşik model; gruplar paralel; rapor lig lig
        groups = _groups(divs)
        print(f"  [birleşik] walk-forward: {len(groups)} grup, {args.workers} paralel işçi ...")
        df = _matches(args, [d for gd in groups.values() for d in gd])
        preds = backtest.walk_forward_groups(df, groups, args.eval_from, args.xi, args.ridge, args.ridge_total,
                                             args.workers)
        for div in divs:
            p = preds[preds["div"] == div].reset_index(drop=True) if not preds.empty else preds
            if p.empty:
                print(f"  [{div}] veri yok, atlanıyor")
                continue
            parts.append(backtest.run(div, None, preds=p, **kw))
    else:
        for div in divs:
            df = _matches(args, div)
            if df.empty:
                print(f"  [{div}] veri yok, atlanıyor")
                continue
            parts.append(backtest.run(div, df, **kw))
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    strat = backtest.strategy_summary(divs, args.book)
    if not strat.empty:  # lig x kural özeti: öneri raporu sinyallerin güvenilirliğini buradan gösterir
        strat.to_csv(config.REPORTS_DIR / "strateji_ozet.csv", index=False)
        wide = strat.pivot(index="lig", columns="kural", values="CLV_ort_%").reindex(
            [d for d in divs if d in set(strat["lig"])])
        tstat = strat.pivot(index="lig", columns="kural", values="CLV_t").reindex(wide.index)
        nbet = strat.pivot(index="lig", columns="kural", values="bahis").reindex(wide.index)
        table = pd.DataFrame({f"{backtest.RULES[r]}": [f"{c:+.2f}% (n={n:.0f}, t={t:+.1f})" if n > 1 else "—"
                                                       for c, t, n in zip(wide[r], tstat[r], nbet[r])]
                              for r in backtest.RULES}, index=wide.index).reset_index()
        parts[2:2] = ["## Lig özeti — kural bazında CLV (B365 oranıyla)", backtest._md_table(table), ""]
    out = config.REPORTS_DIR / "backtest_summary.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n" + "\n".join(parts))
    print(f"\nRapor: {out}  ({time.time() - t0:.0f} sn)")


def cmd_tune(args):
    from src import db, tune
    divs = [args.div] if args.div else db.available_divs()
    grid = lambda s: [float(v) for v in s.split(",")] if s else None  # noqa: E731
    md, table = tune.run(divs, args.xi, args.ridge, args.ridge_total, args.eval_from, args.tune_to, args.workers,
                         grid(args.xi_grid), grid(args.ridge_grid), grid(args.ridge_total_grid), args.criterion,
                         joint=args.joint, motivation=args.motivation)
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
                      joint=args.joint, motivation=args.motivation)
    predict.print_table(res)
    md = predict.recommendations(res)
    out = config.REPORTS_DIR / f"oneriler_{time.strftime('%Y-%m-%d')}.md"
    out.write_text(md, encoding="utf-8")
    print("\n" + md + f"\nÖneriler: {out}")


def cmd_tavsiye(args):
    from pathlib import Path

    from src import tavsiye
    md = tavsiye.run(Path(args.fixtures) if args.fixtures else config.RAW_DIR / "fixtures.csv", xi=args.xi,
                     ridge=args.ridge, ridge_total=args.ridge_total, joint=args.joint, motivation=args.motivation,
                     as_of=args.prova, days=args.gun)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"tavsiye_prova_{args.prova}" if args.prova else f"tavsiye_{time.strftime('%Y-%m-%d')}"
    out = config.REPORTS_DIR / f"{name}.md"
    out.write_text(md, encoding="utf-8")
    print(md + f"\n\nRapor: {out}")


def cmd_iddaa(args):
    from src import iddaa
    print("İddaa oranları toplanıyor ...")
    iddaa.collect()


def cmd_track(args):
    from src import predict, tracking
    upcoming = None
    try:
        res = predict.run(args.fixtures or str(config.RAW_DIR / "fixtures.csv"), None, xi=args.xi, ridge=args.ridge,
                          ridge_total=args.ridge_total, joint=args.joint, motivation=args.motivation, save=False)
        upcoming = res[res["kaynak"] == "fixtures"]
    except SystemExit as e:  # yaklaşan maç yok: sadece kapanış ve özet
        print(f"  {e}")
    print(tracking.run(upcoming))


def cmd_ratings(args):
    from src.dixon_coles import DixonColes
    kw = dict(xi=args.xi, ridge=args.ridge, ridge_total=args.ridge_total)
    models = {}  # birleşik: grup başına bir model
    for div in ([args.div] if args.div else list(config.LEAGUES)):
        df = _matches(args, div)
        if df.empty:
            continue
        cur = df[df["season"] == df["season"].max()]
        joint = None
        if args.joint:
            g = config.group_of(div)
            if g not in models:
                models[g] = DixonColes(**kw).fit(_matches(args, config.LEAGUE_GROUPS[g]))
            joint = models[g]
        m = joint if joint is not None else DixonColes(**kw).fit(df)
        r = m.ratings()
        r = r[r["team"].isin(set(cur["home"]) | set(cur["away"]))]
        extra = (f", lig gücü {m.league_strength[div]:+.3f}, yeni gelen sapması: çıkan "
                 f"{m.newcomer_offset['çıkan']:+.3f} / düşen {m.newcomer_offset['düşen']:+.3f}"
                 ) if joint is not None and m.newcomer_offset else ""
        extra += f", motivasyon {m.motivation:+.3f}" if args.motivation else ""
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
        sp.add_argument("--motivation", action=argparse.BooleanOptionalAction, default=config.MOTIVATION,
                        help="sezon sonu motivasyonunu (puan durumundan) modele kat")

    for name, fn in [("download", cmd_download), ("build", cmd_build), ("backtest", cmd_backtest),
                     ("tune", cmd_tune), ("diagnose", cmd_diagnose), ("predict", cmd_predict), ("track", cmd_track), ("iddaa", cmd_iddaa), ("tavsiye", cmd_tavsiye),
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
            sp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                            help="paralel işçi (birleşik modelde ülke grupları paralel)")
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
        if name in ("track", "tavsiye"):
            sp.add_argument("--fixtures", help="fixtures.csv yolu (varsayılan: data/raw/fixtures.csv)")
        if name == "tavsiye":
            sp.add_argument("--prova", help="geçmiş bir tarihten (YYYY-AA-GG) itibaren prova: model o tarihe kadarki "
                                            "veriyle, tavsiyeler gerçekleşen sonuçlarla karşılaştırılır")
            sp.add_argument("--gun", type=int, default=4, help="provada kaç günlük maç (varsayılan 4)")
        if name == "predict":
            sp.add_argument("--fixtures", help="yerel fixtures.csv yolu (verilmezse indirilir)")
            sp.add_argument("--iddaa", help="İddaa oranları: CSV yolu (bkz. iddaa_ornek.csv) ya da 'auto' "
                                            "(otomatik toplanan son oranlar)")
    args = p.parse_args()
    if args.fn(args) is False:  # ör. siteye ulaşılamadı; GitHub Actions işi başarısız görünsün
        sys.exit(1)


if __name__ == "__main__":
    main()
