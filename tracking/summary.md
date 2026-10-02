# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-10-02 22:04 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| EC | 13 | 2 | -0.1823 ± 0.2385 (n=2) | -0.1338 ± 0.2164 (n=2) |
| toplam | 47 | 2 | -0.1823 ± 0.2385 (n=2) | -0.1338 ± 0.2164 (n=2) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 38 | 37 | 1 | 3.07 | -100.00 | -1.55 | — | 0.00 |
| Keskin açılış adil fiyatı (model yok) | 1 | 0 | 1 | 1.70 | -100.00 | 2.31 | — | 100.00 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 0 | 0 | 0 | — | — | — | — | — |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 6 | 2 | 2.78 | -14.15 | 9.83 | -13.80 | 16.67 | -7.17 |
| İddaa > keskin açılış adil fiyatı | 0 | 0 | — | — | — | — | — | — |
| Model edge ≥ %3 (onaysız) | 0 | 0 | — | — | — | — | — | — |
| Model edge + keskin onay (VALUE) | 0 | 0 | — | — | — | — | — | — |

Sonuçlanmamış (açık) maç: 52. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
