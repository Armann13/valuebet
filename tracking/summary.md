# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-27 21:50 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 16 | 16 | +0.0021 ± 0.0653 (n=16) | +0.0695 ± 0.0741 (n=16) |
| D1 | 24 | 24 | -0.0428 ± 0.0743 (n=24) | +0.0088 ± 0.0448 (n=24) |
| D2 | 21 | 21 | -0.0073 ± 0.0387 (n=21) | -0.0422 ± 0.0589 (n=21) |
| E0 | 29 | 29 | +0.0407 ± 0.0548 (n=29) | +0.0028 ± 0.0330 (n=29) |
| E1 | 37 | 37 | +0.0202 ± 0.0408 (n=37) | -0.0010 ± 0.0399 (n=37) |
| E2 | 40 | 40 | +0.0545 ± 0.0526 (n=40) | +0.0014 ± 0.0469 (n=40) |
| E3 | 44 | 44 | +0.0082 ± 0.0589 (n=44) | -0.0214 ± 0.0450 (n=44) |
| EC | 38 | 34 | +0.0544 ± 0.0515 (n=34) | +0.0388 ± 0.0449 (n=34) |
| F1 | 24 | 24 | +0.0049 ± 0.0786 (n=24) | +0.0359 ± 0.0444 (n=24) |
| F2 | 12 | 12 | +0.0454 ± 0.1156 (n=12) | +0.0421 ± 0.0754 (n=12) |
| G1 | 20 | 20 | +0.1191 ± 0.0868 (n=20) | +0.0422 ± 0.0801 (n=20) |
| I1 | 35 | 28 | +0.0283 ± 0.0515 (n=28) | +0.0309 ± 0.0642 (n=28) |
| I2 | 30 | 27 | +0.0260 ± 0.0590 (n=27) | +0.0084 ± 0.0445 (n=27) |
| N1 | 24 | 24 | -0.0360 ± 0.0885 (n=24) | +0.0053 ± 0.0600 (n=24) |
| P1 | 16 | 16 | -0.0877 ± 0.1050 (n=16) | -0.0020 ± 0.0594 (n=16) |
| SC0 | 24 | 18 | +0.0521 ± 0.0652 (n=18) | +0.0406 ± 0.0822 (n=18) |
| SC1 | 17 | 16 | +0.1175 ± 0.1035 (n=16) | +0.0025 ± 0.1028 (n=16) |
| SC2 | 20 | 19 | +0.0803 ± 0.0818 (n=19) | +0.0405 ± 0.0440 (n=19) |
| SC3 | 15 | 15 | +0.0831 ± 0.0707 (n=15) | +0.0211 ± 0.0665 (n=15) |
| SP1 | 28 | 27 | +0.0748 ± 0.0476 (n=27) | +0.0306 ± 0.0528 (n=27) |
| SP2 | 41 | 41 | +0.0089 ± 0.0557 (n=41) | +0.0153 ± 0.0379 (n=41) |
| T1 | 24 | 24 | +0.1006 ± 0.0745 (n=24) | +0.0097 ± 0.0689 (n=24) |
| toplam | 579 | 556 | +0.0321 ± 0.0144 (n=556) | +0.0141 ± 0.0117 (n=556) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 635 | 26 | 609 | 3.07 | -14.02 | -7.74 | 0.67 | 15.60 |
| Keskin açılış adil fiyatı (model yok) | 5 | 0 | 5 | 3.45 | -42.40 | -4.22 | 6.77 | 40.00 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 4 | 1 | 3 | 4.01 | 68.33 | 0.74 | 3.93 | 66.67 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 3172 | 642 | 2.63 | -13.69 | 0.26 | -13.84 | 2.21 | -15.62 |
| İddaa > keskin açılış adil fiyatı | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |
| Model edge ≥ %3 (onaysız) | 383 | 275 | 4.16 | -16.04 | 1.00 | -16.79 | 4.18 | -31.45 |
| Model edge + keskin onay (VALUE) | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |

Sonuçlanmamış (açık) maç: 183. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
