# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-25 01:56 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 41 | 40 | +0.0224 ± 0.0565 (n=40) | +0.0245 ± 0.0513 (n=40) |
| D1 | 48 | 48 | +0.0056 ± 0.0482 (n=48) | +0.0089 ± 0.0305 (n=48) |
| D2 | 42 | 42 | +0.0260 ± 0.0420 (n=42) | +0.0047 ± 0.0403 (n=42) |
| E0 | 58 | 58 | +0.0028 ± 0.0423 (n=58) | -0.0099 ± 0.0249 (n=58) |
| E1 | 78 | 74 | +0.0335 ± 0.0374 (n=74) | +0.0283 ± 0.0305 (n=74) |
| E2 | 70 | 68 | +0.0444 ± 0.0414 (n=68) | +0.0203 ± 0.0379 (n=68) |
| E3 | 77 | 77 | +0.0280 ± 0.0436 (n=77) | -0.0074 ± 0.0362 (n=77) |
| EC | 74 | 70 | +0.0510 ± 0.0334 (n=70) | +0.0254 ± 0.0287 (n=70) |
| F1 | 46 | 46 | +0.0266 ± 0.0506 (n=46) | +0.0372 ± 0.0347 (n=46) |
| F2 | 23 | 23 | +0.0535 ± 0.0799 (n=23) | +0.0153 ± 0.0519 (n=23) |
| G1 | 41 | 41 | +0.1130 ± 0.0598 (n=41) | +0.0280 ± 0.0540 (n=41) |
| I1 | 64 | 64 | +0.0207 ± 0.0410 (n=64) | +0.0114 ± 0.0486 (n=64) |
| I2 | 62 | 58 | +0.0326 ± 0.0362 (n=58) | -0.0084 ± 0.0303 (n=58) |
| N1 | 50 | 50 | -0.0233 ± 0.0543 (n=50) | +0.0107 ± 0.0379 (n=50) |
| P1 | 32 | 31 | -0.0238 ± 0.0700 (n=31) | -0.0067 ± 0.0521 (n=31) |
| SC0 | 39 | 39 | +0.0199 ± 0.0466 (n=39) | +0.0153 ± 0.0563 (n=39) |
| SC1 | 33 | 31 | +0.0566 ± 0.0825 (n=31) | +0.0088 ± 0.0721 (n=31) |
| SC2 | 40 | 39 | +0.0585 ± 0.0668 (n=39) | +0.0349 ± 0.0306 (n=39) |
| SC3 | 35 | 34 | +0.0484 ± 0.0529 (n=34) | +0.0091 ± 0.0418 (n=34) |
| SP1 | 55 | 54 | +0.0725 ± 0.0391 (n=54) | +0.0182 ± 0.0371 (n=54) |
| SP2 | 80 | 80 | +0.0024 ± 0.0361 (n=80) | +0.0094 ± 0.0275 (n=80) |
| T1 | 49 | 49 | +0.0808 ± 0.0533 (n=49) | -0.0443 ± 0.0937 (n=49) |
| toplam | 1137 | 1116 | +0.0328 ± 0.0103 (n=1116) | +0.0104 ± 0.0092 (n=1116) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 1255 | 23 | 1232 | 3.07 | -15.14 | -7.75 | 0.49 | 16.56 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 14 | 1 | 13 | 4.43 | 6.15 | 0.70 | 5.31 | 61.54 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 6425 | 1295 | 2.63 | -14.26 | 0.19 | -14.33 | 1.99 | -16.29 |
| İddaa > keskin açılış adil fiyatı | 9 | 8 | 2.87 | -0.93 | 6.02 | -14.27 | 33.33 | -71.78 |
| Model edge ≥ %3 (onaysız) | 771 | 537 | 4.21 | -17.28 | 0.77 | -17.49 | 4.15 | -28.74 |
| Model edge + keskin onay (VALUE) | 4 | 4 | 2.93 | 3.00 | 10.83 | -11.77 | 50.00 | -36.50 |

Sonuçlanmamış (açık) maç: 285. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
