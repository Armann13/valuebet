# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-20 23:55 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 41 | 40 | +0.0224 ± 0.0565 (n=40) | +0.0245 ± 0.0513 (n=40) |
| D1 | 48 | 48 | +0.0056 ± 0.0482 (n=48) | +0.0089 ± 0.0305 (n=48) |
| D2 | 42 | 42 | +0.0260 ± 0.0420 (n=42) | +0.0047 ± 0.0403 (n=42) |
| E0 | 58 | 57 | +0.0026 ± 0.0430 (n=57) | -0.0104 ± 0.0253 (n=57) |
| E1 | 74 | 74 | +0.0335 ± 0.0374 (n=74) | +0.0283 ± 0.0305 (n=74) |
| E2 | 69 | 68 | +0.0444 ± 0.0414 (n=68) | +0.0203 ± 0.0379 (n=68) |
| E3 | 77 | 77 | +0.0280 ± 0.0436 (n=77) | -0.0074 ± 0.0362 (n=77) |
| EC | 71 | 70 | +0.0510 ± 0.0334 (n=70) | +0.0254 ± 0.0287 (n=70) |
| F1 | 46 | 46 | +0.0266 ± 0.0506 (n=46) | +0.0372 ± 0.0347 (n=46) |
| F2 | 23 | 21 | +0.0560 ± 0.0809 (n=21) | +0.0179 ± 0.0545 (n=21) |
| G1 | 41 | 40 | +0.1136 ± 0.0613 (n=40) | +0.0264 ± 0.0553 (n=40) |
| I1 | 64 | 63 | +0.0226 ± 0.0415 (n=63) | +0.0100 ± 0.0493 (n=63) |
| I2 | 58 | 58 | +0.0326 ± 0.0362 (n=58) | -0.0084 ± 0.0303 (n=58) |
| N1 | 50 | 50 | -0.0233 ± 0.0543 (n=50) | +0.0107 ± 0.0379 (n=50) |
| P1 | 32 | 31 | -0.0238 ± 0.0700 (n=31) | -0.0067 ± 0.0521 (n=31) |
| SC0 | 39 | 39 | +0.0199 ± 0.0466 (n=39) | +0.0153 ± 0.0563 (n=39) |
| SC1 | 33 | 31 | +0.0566 ± 0.0825 (n=31) | +0.0088 ± 0.0721 (n=31) |
| SC2 | 40 | 39 | +0.0585 ± 0.0668 (n=39) | +0.0349 ± 0.0306 (n=39) |
| SC3 | 35 | 34 | +0.0484 ± 0.0529 (n=34) | +0.0091 ± 0.0418 (n=34) |
| SP1 | 55 | 53 | +0.0729 ± 0.0398 (n=53) | +0.0204 ± 0.0375 (n=53) |
| SP2 | 80 | 79 | +0.0041 ± 0.0364 (n=79) | +0.0102 ± 0.0278 (n=79) |
| T1 | 49 | 48 | +0.0840 ± 0.0540 (n=48) | -0.0450 ± 0.0957 (n=48) |
| toplam | 1125 | 1108 | +0.0331 ± 0.0103 (n=1108) | +0.0104 ± 0.0092 (n=1108) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 1242 | 18 | 1224 | 3.08 | -15.63 | -7.78 | 0.49 | 16.42 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 13 | 0 | 13 | 4.43 | 6.15 | 0.70 | 5.31 | 61.54 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 6385 | 1287 | 2.63 | -14.26 | 0.20 | -14.33 | 2.00 | -16.25 |
| İddaa > keskin açılış adil fiyatı | 9 | 8 | 2.87 | -0.93 | 6.02 | -14.27 | 33.33 | -71.78 |
| Model edge ≥ %3 (onaysız) | 767 | 535 | 4.21 | -17.28 | 0.77 | -17.50 | 4.17 | -28.96 |
| Model edge + keskin onay (VALUE) | 4 | 4 | 2.93 | 3.00 | 10.83 | -11.77 | 50.00 | -36.50 |

Sonuçlanmamış (açık) maç: 293. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
