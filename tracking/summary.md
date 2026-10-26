# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-23 21:17 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 16 | 8 | +0.0463 ± 0.0715 (n=8) | +0.0292 ± 0.1136 (n=8) |
| D1 | 24 | 16 | -0.0749 ± 0.1057 (n=16) | +0.0268 ± 0.0529 (n=16) |
| D2 | 21 | 14 | -0.0067 ± 0.0494 (n=14) | -0.0241 ± 0.0795 (n=14) |
| E0 | 29 | 20 | +0.0439 ± 0.0682 (n=20) | +0.0153 ± 0.0346 (n=20) |
| E1 | 37 | 26 | +0.0249 ± 0.0506 (n=26) | +0.0123 ± 0.0442 (n=26) |
| E2 | 40 | 28 | +0.0548 ± 0.0689 (n=28) | -0.0058 ± 0.0532 (n=28) |
| E3 | 44 | 32 | -0.0139 ± 0.0679 (n=32) | -0.0366 ± 0.0500 (n=32) |
| EC | 35 | 24 | +0.0653 ± 0.0669 (n=24) | +0.0157 ± 0.0581 (n=24) |
| F1 | 24 | 16 | -0.0142 ± 0.1071 (n=16) | +0.0781 ± 0.0406 (n=16) |
| F2 | 12 | 8 | +0.0605 ± 0.1644 (n=8) | +0.0530 ± 0.0879 (n=8) |
| G1 | 20 | 13 | +0.0752 ± 0.1131 (n=13) | +0.0307 ± 0.0834 (n=13) |
| I1 | 28 | 19 | +0.0081 ± 0.0687 (n=19) | +0.0321 ± 0.0713 (n=19) |
| I2 | 27 | 18 | +0.0179 ± 0.0861 (n=18) | +0.0060 ± 0.0588 (n=18) |
| N1 | 24 | 16 | -0.0656 ± 0.1257 (n=16) | -0.0085 ± 0.0810 (n=16) |
| P1 | 16 | 8 | -0.0007 ± 0.0819 (n=8) | -0.0253 ± 0.0853 (n=8) |
| SC0 | 18 | 12 | +0.0113 ± 0.0813 (n=12) | +0.0209 ± 0.1050 (n=12) |
| SC1 | 17 | 12 | +0.0580 ± 0.1139 (n=12) | +0.0029 ± 0.1018 (n=12) |
| SC2 | 20 | 15 | +0.0700 ± 0.1005 (n=15) | +0.0459 ± 0.0465 (n=15) |
| SC3 | 15 | 15 | +0.0831 ± 0.0707 (n=15) | +0.0211 ± 0.0665 (n=15) |
| SP1 | 28 | 18 | +0.0742 ± 0.0625 (n=18) | -0.0007 ± 0.0662 (n=18) |
| SP2 | 41 | 31 | +0.0063 ± 0.0619 (n=31) | +0.0259 ± 0.0416 (n=31) |
| T1 | 24 | 16 | +0.0879 ± 0.0949 (n=16) | +0.0261 ± 0.0902 (n=16) |
| toplam | 560 | 385 | +0.0259 ± 0.0179 (n=385) | +0.0128 ± 0.0136 (n=385) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 615 | 187 | 428 | 3.07 | -11.60 | -7.57 | 0.83 | 17.99 |
| Keskin açılış adil fiyatı (model yok) | 5 | 2 | 3 | 3.45 | -100.00 | 0.66 | 4.41 | 66.67 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 3 | 0 | 3 | 3.43 | 68.33 | 0.74 | 3.93 | 66.67 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 2234 | 454 | 2.63 | -13.58 | 0.32 | -13.82 | 2.37 | -15.49 |
| İddaa > keskin açılış adil fiyatı | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |
| Model edge ≥ %3 (onaysız) | 265 | 194 | 4.19 | -16.05 | 1.24 | -16.92 | 4.91 | -29.04 |
| Model edge + keskin onay (VALUE) | 1 | 1 | 2.33 | 10.63 | — | -8.36 | 100.00 | -100.00 |

Sonuçlanmamış (açık) maç: 304. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
