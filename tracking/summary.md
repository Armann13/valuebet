# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-03 22:56 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 25 | 24 | -0.0211 ± 0.0704 (n=24) | +0.0377 ± 0.0625 (n=24) |
| D1 | 32 | 32 | -0.0286 ± 0.0595 (n=32) | +0.0081 ± 0.0365 (n=32) |
| D2 | 28 | 28 | -0.0075 ± 0.0414 (n=28) | -0.0198 ± 0.0505 (n=28) |
| E0 | 39 | 39 | +0.0285 ± 0.0485 (n=39) | -0.0132 ± 0.0281 (n=39) |
| E1 | 51 | 48 | +0.0436 ± 0.0375 (n=48) | +0.0241 ± 0.0362 (n=48) |
| E2 | 53 | 52 | +0.0476 ± 0.0429 (n=52) | +0.0047 ± 0.0430 (n=52) |
| E3 | 55 | 55 | +0.0182 ± 0.0540 (n=55) | -0.0023 ± 0.0433 (n=55) |
| EC | 49 | 48 | +0.0586 ± 0.0407 (n=48) | +0.0302 ± 0.0360 (n=48) |
| F1 | 30 | 30 | +0.0134 ± 0.0631 (n=30) | +0.0354 ± 0.0382 (n=30) |
| F2 | 16 | 16 | +0.0401 ± 0.0967 (n=16) | +0.0383 ± 0.0640 (n=16) |
| G1 | 27 | 27 | +0.1096 ± 0.0744 (n=27) | +0.0266 ± 0.0665 (n=27) |
| I1 | 45 | 45 | +0.0224 ± 0.0501 (n=45) | +0.0218 ± 0.0560 (n=45) |
| I2 | 40 | 40 | +0.0348 ± 0.0409 (n=40) | +0.0026 ± 0.0374 (n=40) |
| N1 | 33 | 33 | -0.0349 ± 0.0736 (n=33) | -0.0119 ± 0.0514 (n=33) |
| P1 | 24 | 24 | -0.0591 ± 0.0849 (n=24) | -0.0187 ± 0.0613 (n=24) |
| SC0 | 27 | 27 | +0.0059 ± 0.0552 (n=27) | +0.0213 ± 0.0627 (n=27) |
| SC1 | 21 | 20 | +0.0690 ± 0.1200 (n=20) | +0.0349 ± 0.0920 (n=20) |
| SC2 | 25 | 24 | +0.0818 ± 0.0749 (n=24) | +0.0312 ± 0.0391 (n=24) |
| SC3 | 20 | 20 | +0.0881 ± 0.0643 (n=20) | -0.0018 ± 0.0573 (n=20) |
| SP1 | 37 | 36 | +0.0643 ± 0.0490 (n=36) | +0.0183 ± 0.0445 (n=36) |
| SP2 | 52 | 51 | +0.0013 ± 0.0493 (n=51) | +0.0200 ± 0.0346 (n=51) |
| T1 | 32 | 32 | +0.0739 ± 0.0618 (n=32) | -0.0637 ± 0.1345 (n=32) |
| toplam | 761 | 751 | +0.0289 ± 0.0125 (n=751) | +0.0094 ± 0.0115 (n=751) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 844 | 12 | 832 | 3.08 | -12.79 | -7.42 | 0.60 | 17.19 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 10 | 0 | 10 | 4.40 | 38.00 | 1.71 | 5.79 | 70.00 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 4364 | 881 | 2.63 | -13.82 | 0.24 | -13.94 | 2.38 | -15.69 |
| İddaa > keskin açılış adil fiyatı | 7 | 6 | 2.97 | 0.43 | 7.27 | -13.38 | 42.86 | -63.71 |
| Model edge ≥ %3 (onaysız) | 518 | 372 | 4.14 | -15.82 | 0.91 | -16.75 | 6.18 | -22.36 |
| Model edge + keskin onay (VALUE) | 3 | 3 | 2.95 | 7.46 | 9.06 | -9.95 | 66.67 | -15.33 |

Sonuçlanmamış (açık) maç: 194. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
