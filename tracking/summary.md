# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-20 23:55 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 41 | 32 | -0.0051 ± 0.0562 (n=32) | +0.0274 ± 0.0547 (n=32) |
| D1 | 48 | 40 | -0.0053 ± 0.0515 (n=40) | +0.0129 ± 0.0320 (n=40) |
| D2 | 42 | 35 | +0.0012 ± 0.0400 (n=35) | -0.0060 ± 0.0443 (n=35) |
| E0 | 58 | 48 | +0.0195 ± 0.0465 (n=48) | -0.0065 ± 0.0263 (n=48) |
| E1 | 74 | 62 | +0.0282 ± 0.0422 (n=62) | +0.0308 ± 0.0325 (n=62) |
| E2 | 69 | 57 | +0.0601 ± 0.0411 (n=57) | +0.0121 ± 0.0425 (n=57) |
| E3 | 77 | 65 | +0.0246 ± 0.0478 (n=65) | -0.0068 ± 0.0390 (n=65) |
| EC | 71 | 59 | +0.0616 ± 0.0353 (n=59) | +0.0238 ± 0.0323 (n=59) |
| F1 | 46 | 38 | +0.0228 ± 0.0582 (n=38) | +0.0191 ± 0.0354 (n=38) |
| F2 | 23 | 19 | +0.0368 ± 0.0836 (n=19) | +0.0306 ± 0.0554 (n=19) |
| G1 | 41 | 34 | +0.1132 ± 0.0696 (n=34) | +0.0359 ± 0.0574 (n=34) |
| I1 | 64 | 54 | +0.0145 ± 0.0454 (n=54) | -0.0002 ± 0.0525 (n=54) |
| I2 | 58 | 49 | +0.0306 ± 0.0362 (n=49) | +0.0003 ± 0.0325 (n=49) |
| N1 | 50 | 41 | -0.0282 ± 0.0617 (n=41) | +0.0061 ± 0.0440 (n=41) |
| P1 | 32 | 31 | -0.0238 ± 0.0700 (n=31) | -0.0067 ± 0.0521 (n=31) |
| SC0 | 39 | 33 | +0.0026 ± 0.0499 (n=33) | -0.0060 ± 0.0630 (n=33) |
| SC1 | 33 | 28 | +0.0604 ± 0.0913 (n=28) | +0.0050 ± 0.0788 (n=28) |
| SC2 | 40 | 34 | +0.0433 ± 0.0662 (n=34) | +0.0295 ± 0.0320 (n=34) |
| SC3 | 35 | 30 | +0.0493 ± 0.0588 (n=30) | +0.0052 ± 0.0446 (n=30) |
| SP1 | 55 | 45 | +0.0784 ± 0.0425 (n=45) | +0.0107 ± 0.0423 (n=45) |
| SP2 | 80 | 70 | +0.0050 ± 0.0388 (n=70) | +0.0080 ± 0.0296 (n=70) |
| T1 | 49 | 40 | +0.0821 ± 0.0559 (n=40) | -0.0495 ± 0.1090 (n=40) |
| toplam | 1125 | 944 | +0.0304 ± 0.0111 (n=944) | +0.0076 ± 0.0101 (n=944) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 1242 | 203 | 1039 | 3.08 | -14.72 | -7.75 | 0.54 | 16.55 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 13 | 2 | 11 | 4.43 | 25.45 | 0.43 | 5.81 | 63.64 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 5496 | 1108 | 2.62 | -14.05 | 0.21 | -14.15 | 2.20 | -16.29 |
| İddaa > keskin açılış adil fiyatı | 7 | 6 | 2.97 | 0.43 | 7.27 | -13.38 | 42.86 | -63.71 |
| Model edge ≥ %3 (onaysız) | 632 | 453 | 4.13 | -16.66 | 0.83 | -17.09 | 5.06 | -25.54 |
| Model edge + keskin onay (VALUE) | 3 | 3 | 2.95 | 7.46 | 9.06 | -9.95 | 66.67 | -15.33 |

Sonuçlanmamış (açık) maç: 372. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
