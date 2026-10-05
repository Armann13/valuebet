# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-02 22:04 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| E2 | 3 | 3 | +0.1401 ± 0.2158 (n=3) | -0.0247 ± 0.1733 (n=3) |
| E3 | 8 | 8 | -0.0278 ± 0.1119 (n=8) | -0.0447 ± 0.1127 (n=8) |
| EC | 13 | 13 | +0.0449 ± 0.0966 (n=13) | +0.0320 ± 0.0726 (n=13) |
| SC1 | 3 | 3 | +0.1371 ± 0.2881 (n=3) | +0.2082 ± 0.0945 (n=3) |
| SC2 | 5 | 5 | +0.1107 ± 0.1271 (n=5) | +0.0860 ± 0.0886 (n=5) |
| SC3 | 5 | 5 | +0.0511 ± 0.0553 (n=5) | +0.0049 ± 0.0892 (n=5) |
| SP2 | 10 | 9 | +0.0166 ± 0.1073 (n=9) | +0.0198 ± 0.1066 (n=9) |
| toplam | 47 | 46 | +0.0468 ± 0.0473 (n=46) | +0.0270 ± 0.0414 (n=46) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 38 | 1 | 37 | 3.07 | -18.81 | -10.06 | 3.34 | 8.11 |
| Keskin açılış adil fiyatı (model yok) | 1 | 0 | 1 | 1.70 | -100.00 | 2.31 | — | 100.00 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 0 | 0 | 0 | — | — | — | — | — |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 196 | 40 | 2.52 | -15.27 | 1.06 | -14.98 | 2.04 | -16.09 |
| İddaa > keskin açılış adil fiyatı | 0 | 0 | — | — | — | — | — | — |
| Model edge ≥ %3 (onaysız) | 19 | 15 | 4.20 | -20.68 | 4.74 | -20.59 | 5.26 | -20.05 |
| Model edge + keskin onay (VALUE) | 0 | 0 | — | — | — | — | — | — |

Sonuçlanmamış (açık) maç: 13. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
