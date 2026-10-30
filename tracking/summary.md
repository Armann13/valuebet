# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-30 22:11 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 24 | 16 | +0.0021 ± 0.0653 (n=16) | +0.0695 ± 0.0741 (n=16) |
| D1 | 32 | 24 | -0.0428 ± 0.0743 (n=24) | +0.0088 ± 0.0448 (n=24) |
| D2 | 28 | 21 | -0.0073 ± 0.0387 (n=21) | -0.0422 ± 0.0589 (n=21) |
| E0 | 39 | 29 | +0.0407 ± 0.0548 (n=29) | +0.0028 ± 0.0330 (n=29) |
| E1 | 48 | 37 | +0.0202 ± 0.0408 (n=37) | -0.0010 ± 0.0399 (n=37) |
| E2 | 52 | 40 | +0.0545 ± 0.0526 (n=40) | +0.0014 ± 0.0469 (n=40) |
| E3 | 55 | 44 | +0.0082 ± 0.0589 (n=44) | -0.0214 ± 0.0450 (n=44) |
| EC | 49 | 37 | +0.0574 ± 0.0477 (n=37) | +0.0339 ± 0.0429 (n=37) |
| F1 | 30 | 24 | +0.0049 ± 0.0786 (n=24) | +0.0359 ± 0.0444 (n=24) |
| F2 | 16 | 12 | +0.0454 ± 0.1156 (n=12) | +0.0421 ± 0.0754 (n=12) |
| G1 | 27 | 20 | +0.1191 ± 0.0868 (n=20) | +0.0422 ± 0.0801 (n=20) |
| I1 | 45 | 35 | +0.0101 ± 0.0562 (n=35) | +0.0386 ± 0.0587 (n=35) |
| I2 | 40 | 30 | +0.0298 ± 0.0538 (n=30) | +0.0007 ± 0.0445 (n=30) |
| N1 | 33 | 24 | -0.0360 ± 0.0885 (n=24) | +0.0053 ± 0.0600 (n=24) |
| P1 | 24 | 16 | -0.0877 ± 0.1050 (n=16) | -0.0020 ± 0.0594 (n=16) |
| SC0 | 27 | 24 | +0.0218 ± 0.0560 (n=24) | +0.0370 ± 0.0652 (n=24) |
| SC1 | 21 | 16 | +0.1175 ± 0.1035 (n=16) | +0.0025 ± 0.1028 (n=16) |
| SC2 | 25 | 19 | +0.0803 ± 0.0818 (n=19) | +0.0405 ± 0.0440 (n=19) |
| SC3 | 20 | 15 | +0.0831 ± 0.0707 (n=15) | +0.0211 ± 0.0665 (n=15) |
| SP1 | 37 | 27 | +0.0748 ± 0.0476 (n=27) | +0.0306 ± 0.0528 (n=27) |
| SP2 | 52 | 41 | +0.0089 ± 0.0557 (n=41) | +0.0153 ± 0.0379 (n=41) |
| T1 | 32 | 24 | +0.1006 ± 0.0745 (n=24) | +0.0097 ± 0.0689 (n=24) |
| toplam | 756 | 575 | +0.0303 ± 0.0142 (n=575) | +0.0143 ± 0.0116 (n=575) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 839 | 210 | 629 | 3.08 | -12.58 | -7.70 | 0.66 | 15.90 |
| Keskin açılış adil fiyatı (model yok) | 8 | 3 | 5 | 3.48 | -42.40 | -4.22 | 6.77 | 40.00 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 10 | 6 | 4 | 4.40 | 170.00 | -1.52 | 5.23 | 50.00 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 3397 | 687 | 2.62 | -13.75 | 0.25 | -13.88 | 2.21 | -15.83 |
| İddaa > keskin açılış adil fiyatı | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |
| Model edge ≥ %3 (onaysız) | 395 | 285 | 4.13 | -15.95 | 0.99 | -16.69 | 4.56 | -28.78 |
| Model edge + keskin onay (VALUE) | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |

Sonuçlanmamış (açık) maç: 316. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
