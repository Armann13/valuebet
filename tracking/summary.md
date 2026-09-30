# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-09-30 13:13 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| toplam | 2 | 0 | — | — |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 1 | 1 | 0 | 4.10 | — | — | — | — |
| Keskin açılış adil fiyatı (model yok) | 1 | 1 | 0 | 1.70 | — | — | — | — |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 0 | 0 | 0 | — | — | — | — | — |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

Henüz İddaa oranı kaydı eşleşen maç yok.
