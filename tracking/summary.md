# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-16 20:50 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 8 | 8 | +0.0463 ± 0.0715 (n=8) | +0.0292 ± 0.1136 (n=8) |
| D1 | 16 | 8 | -0.1015 ± 0.1951 (n=8) | +0.0372 ± 0.0769 (n=8) |
| D2 | 14 | 7 | +0.0414 ± 0.0592 (n=7) | +0.0526 ± 0.0730 (n=7) |
| E0 | 20 | 10 | +0.0100 ± 0.0969 (n=10) | -0.0030 ± 0.0325 (n=10) |
| E1 | 26 | 15 | +0.0310 ± 0.0564 (n=15) | -0.0014 ± 0.0606 (n=15) |
| E2 | 27 | 15 | +0.0684 ± 0.1021 (n=15) | +0.0017 ± 0.0684 (n=15) |
| E3 | 32 | 20 | -0.0292 ± 0.0911 (n=20) | -0.0530 ± 0.0618 (n=20) |
| EC | 24 | 24 | +0.0653 ± 0.0669 (n=24) | +0.0157 ± 0.0581 (n=24) |
| F1 | 16 | 8 | -0.0191 ± 0.1786 (n=8) | +0.0729 ± 0.0725 (n=8) |
| F2 | 8 | 4 | +0.1687 ± 0.1474 (n=4) | -0.0197 ± 0.0974 (n=4) |
| G1 | 13 | 7 | +0.0627 ± 0.1565 (n=7) | +0.0282 ± 0.0882 (n=7) |
| I1 | 19 | 10 | -0.0178 ± 0.0520 (n=10) | +0.0021 ± 0.1139 (n=10) |
| I2 | 18 | 9 | +0.0656 ± 0.1154 (n=9) | -0.0116 ± 0.0937 (n=9) |
| N1 | 16 | 8 | -0.0829 ± 0.1256 (n=8) | -0.0870 ± 0.1314 (n=8) |
| P1 | 8 | 7 | +0.0175 ± 0.0852 (n=7) | -0.0167 ± 0.0966 (n=7) |
| SC0 | 12 | 6 | +0.0373 ± 0.1185 (n=6) | -0.0427 ± 0.1661 (n=6) |
| SC1 | 12 | 8 | +0.0858 ± 0.1463 (n=8) | +0.0591 ± 0.1309 (n=8) |
| SC2 | 15 | 10 | +0.0606 ± 0.1336 (n=10) | +0.0512 ± 0.0629 (n=10) |
| SC3 | 15 | 10 | +0.0713 ± 0.0976 (n=10) | +0.0387 ± 0.0707 (n=10) |
| SP1 | 18 | 8 | +0.1283 ± 0.0437 (n=8) | -0.0194 ± 0.1009 (n=8) |
| SP2 | 31 | 21 | +0.0176 ± 0.0736 (n=21) | +0.0327 ± 0.0546 (n=21) |
| T1 | 16 | 8 | +0.1568 ± 0.0947 (n=8) | +0.0812 ± 0.1252 (n=8) |
| toplam | 384 | 231 | +0.0352 ± 0.0227 (n=231) | +0.0099 ± 0.0179 (n=231) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 427 | 181 | 246 | 3.06 | -21.96 | -7.96 | 1.10 | 16.26 |
| Keskin açılış adil fiyatı (model yok) | 3 | 0 | 3 | 3.72 | -100.00 | 0.66 | 4.41 | 66.67 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 3 | 1 | 2 | 3.43 | 60.00 | 0.14 | 6.50 | 50.00 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 1070 | 220 | 2.67 | -13.61 | 0.45 | -13.55 | 1.68 | -17.27 |
| İddaa > keskin açılış adil fiyatı | 0 | 0 | — | — | — | — | — | — |
| Model edge ≥ %3 (onaysız) | 136 | 100 | 4.43 | -16.41 | 1.88 | -16.67 | 5.15 | -35.56 |
| Model edge + keskin onay (VALUE) | 0 | 0 | — | — | — | — | — | — |

Sonuçlanmamış (açık) maç: 201. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
