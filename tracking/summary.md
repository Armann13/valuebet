# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-30 22:11 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 24 | 24 | -0.0211 ± 0.0704 (n=24) | +0.0377 ± 0.0625 (n=24) |
| D1 | 32 | 32 | -0.0286 ± 0.0595 (n=32) | +0.0081 ± 0.0365 (n=32) |
| D2 | 28 | 28 | -0.0075 ± 0.0414 (n=28) | -0.0198 ± 0.0505 (n=28) |
| E0 | 39 | 38 | +0.0300 ± 0.0497 (n=38) | -0.0121 ± 0.0288 (n=38) |
| E1 | 48 | 48 | +0.0436 ± 0.0375 (n=48) | +0.0241 ± 0.0362 (n=48) |
| E2 | 52 | 52 | +0.0476 ± 0.0429 (n=52) | +0.0047 ± 0.0430 (n=52) |
| E3 | 55 | 55 | +0.0182 ± 0.0540 (n=55) | -0.0023 ± 0.0433 (n=55) |
| EC | 49 | 48 | +0.0586 ± 0.0407 (n=48) | +0.0302 ± 0.0360 (n=48) |
| F1 | 30 | 29 | +0.0128 ± 0.0653 (n=29) | +0.0294 ± 0.0376 (n=29) |
| F2 | 16 | 14 | +0.0360 ± 0.1077 (n=14) | +0.0482 ± 0.0695 (n=14) |
| G1 | 27 | 25 | +0.1160 ± 0.0798 (n=25) | +0.0219 ± 0.0714 (n=25) |
| I1 | 45 | 43 | +0.0287 ± 0.0494 (n=43) | +0.0152 ± 0.0576 (n=43) |
| I2 | 40 | 40 | +0.0348 ± 0.0409 (n=40) | +0.0026 ± 0.0374 (n=40) |
| N1 | 33 | 33 | -0.0349 ± 0.0736 (n=33) | -0.0119 ± 0.0514 (n=33) |
| P1 | 24 | 22 | -0.0751 ± 0.0897 (n=22) | +0.0091 ± 0.0455 (n=22) |
| SC0 | 27 | 27 | +0.0059 ± 0.0552 (n=27) | +0.0213 ± 0.0627 (n=27) |
| SC1 | 21 | 20 | +0.0690 ± 0.1200 (n=20) | +0.0349 ± 0.0920 (n=20) |
| SC2 | 25 | 24 | +0.0818 ± 0.0749 (n=24) | +0.0312 ± 0.0391 (n=24) |
| SC3 | 20 | 20 | +0.0881 ± 0.0643 (n=20) | -0.0018 ± 0.0573 (n=20) |
| SP1 | 37 | 35 | +0.0631 ± 0.0503 (n=35) | +0.0235 ± 0.0446 (n=35) |
| SP2 | 52 | 48 | +0.0014 ± 0.0498 (n=48) | +0.0091 ± 0.0345 (n=48) |
| T1 | 32 | 31 | +0.0753 ± 0.0638 (n=31) | -0.0694 ± 0.1384 (n=31) |
| toplam | 756 | 736 | +0.0290 ± 0.0126 (n=736) | +0.0090 ± 0.0116 (n=736) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 839 | 25 | 814 | 3.08 | -12.58 | -7.38 | 0.60 | 17.20 |
| Keskin açılış adil fiyatı (model yok) | 8 | 1 | 7 | 3.48 | -58.86 | -2.97 | 5.53 | 42.86 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 10 | 0 | 10 | 4.40 | 38.00 | 1.71 | 5.79 | 70.00 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 4289 | 866 | 2.63 | -13.82 | 0.24 | -13.93 | 2.38 | -15.75 |
| İddaa > keskin açılış adil fiyatı | 7 | 6 | 2.97 | 0.43 | 7.27 | -13.38 | 42.86 | -63.71 |
| Model edge ≥ %3 (onaysız) | 512 | 367 | 4.12 | -15.70 | 0.91 | -16.67 | 6.25 | -22.27 |
| Model edge + keskin onay (VALUE) | 3 | 3 | 2.95 | 7.46 | 9.06 | -9.95 | 66.67 | -15.33 |

Sonuçlanmamış (açık) maç: 209. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
