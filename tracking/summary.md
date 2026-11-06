# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-06 22:57 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 33 | 24 | -0.0211 ± 0.0704 (n=24) | +0.0377 ± 0.0625 (n=24) |
| D1 | 40 | 32 | -0.0286 ± 0.0595 (n=32) | +0.0081 ± 0.0365 (n=32) |
| D2 | 35 | 28 | -0.0075 ± 0.0414 (n=28) | -0.0198 ± 0.0505 (n=28) |
| E0 | 48 | 39 | +0.0285 ± 0.0485 (n=39) | -0.0132 ± 0.0281 (n=39) |
| E1 | 62 | 51 | +0.0373 ± 0.0400 (n=51) | +0.0271 ± 0.0345 (n=51) |
| E2 | 53 | 52 | +0.0476 ± 0.0429 (n=52) | +0.0047 ± 0.0430 (n=52) |
| E3 | 55 | 55 | +0.0182 ± 0.0540 (n=55) | -0.0023 ± 0.0433 (n=55) |
| EC | 49 | 48 | +0.0586 ± 0.0407 (n=48) | +0.0302 ± 0.0360 (n=48) |
| F1 | 38 | 30 | +0.0134 ± 0.0631 (n=30) | +0.0354 ± 0.0382 (n=30) |
| F2 | 19 | 16 | +0.0401 ± 0.0967 (n=16) | +0.0383 ± 0.0640 (n=16) |
| G1 | 34 | 27 | +0.1096 ± 0.0744 (n=27) | +0.0266 ± 0.0665 (n=27) |
| I1 | 54 | 45 | +0.0224 ± 0.0501 (n=45) | +0.0218 ± 0.0560 (n=45) |
| I2 | 49 | 40 | +0.0348 ± 0.0409 (n=40) | +0.0026 ± 0.0374 (n=40) |
| N1 | 41 | 33 | -0.0349 ± 0.0736 (n=33) | -0.0119 ± 0.0514 (n=33) |
| P1 | 32 | 24 | -0.0591 ± 0.0849 (n=24) | -0.0187 ± 0.0613 (n=24) |
| SC0 | 33 | 27 | +0.0059 ± 0.0552 (n=27) | +0.0213 ± 0.0627 (n=27) |
| SC1 | 25 | 20 | +0.0690 ± 0.1200 (n=20) | +0.0349 ± 0.0920 (n=20) |
| SC2 | 30 | 24 | +0.0818 ± 0.0749 (n=24) | +0.0312 ± 0.0391 (n=24) |
| SC3 | 25 | 20 | +0.0881 ± 0.0643 (n=20) | -0.0018 ± 0.0573 (n=20) |
| SP1 | 46 | 36 | +0.0643 ± 0.0490 (n=36) | +0.0183 ± 0.0445 (n=36) |
| SP2 | 60 | 51 | +0.0013 ± 0.0493 (n=51) | +0.0200 ± 0.0346 (n=51) |
| T1 | 40 | 32 | +0.0739 ± 0.0618 (n=32) | -0.0637 ± 0.1345 (n=32) |
| toplam | 901 | 754 | +0.0285 ± 0.0125 (n=754) | +0.0097 ± 0.0115 (n=754) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 996 | 159 | 837 | 3.09 | -12.53 | -7.41 | 0.60 | 17.20 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 11 | 1 | 10 | 4.36 | 38.00 | 1.71 | 5.79 | 70.00 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 4454 | 899 | 2.62 | -13.81 | 0.23 | -13.92 | 2.38 | -15.60 |
| İddaa > keskin açılış adil fiyatı | 7 | 6 | 2.97 | 0.43 | 7.27 | -13.38 | 42.86 | -63.71 |
| Model edge ≥ %3 (onaysız) | 521 | 375 | 4.14 | -15.84 | 0.91 | -16.74 | 6.14 | -22.07 |
| Model edge + keskin onay (VALUE) | 3 | 3 | 2.95 | 7.46 | 9.06 | -9.95 | 66.67 | -15.33 |

Sonuçlanmamış (açık) maç: 336. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
