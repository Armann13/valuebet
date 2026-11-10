# Canlı takip (kağıt üzerinde)

Son kayıt: 2026-11-06 22:57 UTC. Model: dc-v3 (birleşik + yeni gelen + motivasyon + xG 0.25, ridge 4/16). Bahisler B365 oranıyla, 1 birim.
Az veriyle sonuçlar çok gürültülüdür; asıl ölçüt birkaç yüz maçtan sonra anlam kazanır.

## Tahmin kalitesi
| lig | kayıtlı | sonuçlanan | 1X2 log loss (model − kapanış) | OU2.5 log loss (model − kapanış) |
|---|---|---|---|---|
| B1 | 33 | 32 | -0.0051 ± 0.0562 (n=32) | +0.0274 ± 0.0547 (n=32) |
| D1 | 40 | 40 | -0.0053 ± 0.0515 (n=40) | +0.0129 ± 0.0320 (n=40) |
| D2 | 35 | 35 | +0.0012 ± 0.0400 (n=35) | -0.0060 ± 0.0443 (n=35) |
| E0 | 48 | 48 | +0.0195 ± 0.0465 (n=48) | -0.0065 ± 0.0263 (n=48) |
| E1 | 62 | 62 | +0.0282 ± 0.0422 (n=62) | +0.0308 ± 0.0325 (n=62) |
| E2 | 53 | 52 | +0.0476 ± 0.0429 (n=52) | +0.0047 ± 0.0430 (n=52) |
| E3 | 55 | 55 | +0.0182 ± 0.0540 (n=55) | -0.0023 ± 0.0433 (n=55) |
| EC | 49 | 48 | +0.0586 ± 0.0407 (n=48) | +0.0302 ± 0.0360 (n=48) |
| F1 | 38 | 38 | +0.0228 ± 0.0582 (n=38) | +0.0191 ± 0.0354 (n=38) |
| F2 | 19 | 19 | +0.0368 ± 0.0836 (n=19) | +0.0306 ± 0.0554 (n=19) |
| G1 | 34 | 34 | +0.1132 ± 0.0696 (n=34) | +0.0359 ± 0.0574 (n=34) |
| I1 | 54 | 54 | +0.0145 ± 0.0454 (n=54) | -0.0002 ± 0.0525 (n=54) |
| I2 | 49 | 49 | +0.0306 ± 0.0362 (n=49) | +0.0003 ± 0.0325 (n=49) |
| N1 | 41 | 41 | -0.0282 ± 0.0617 (n=41) | +0.0061 ± 0.0440 (n=41) |
| P1 | 32 | 31 | -0.0238 ± 0.0700 (n=31) | -0.0067 ± 0.0521 (n=31) |
| SC0 | 33 | 33 | +0.0026 ± 0.0499 (n=33) | -0.0060 ± 0.0630 (n=33) |
| SC1 | 25 | 24 | +0.0668 ± 0.1033 (n=24) | +0.0156 ± 0.0840 (n=24) |
| SC2 | 30 | 29 | +0.0487 ± 0.0742 (n=29) | +0.0346 ± 0.0348 (n=29) |
| SC3 | 25 | 25 | +0.0795 ± 0.0556 (n=25) | +0.0123 ± 0.0491 (n=25) |
| SP1 | 46 | 45 | +0.0784 ± 0.0425 (n=45) | +0.0107 ± 0.0423 (n=45) |
| SP2 | 60 | 60 | +0.0051 ± 0.0437 (n=60) | +0.0205 ± 0.0310 (n=60) |
| T1 | 40 | 40 | +0.0821 ± 0.0559 (n=40) | -0.0495 ± 0.1090 (n=40) |
| toplam | 901 | 894 | +0.0298 ± 0.0114 (n=894) | +0.0091 ± 0.0104 (n=894) |

## Kağıt bahisler
| kural | bahis | açık | sonuçlanan | ort. oran | ROI_% | CLV_ort_% | CLV ±GA_% | CLV>0_% |
|---|---|---|---|---|---|---|---|---|
| Model | 996 | 8 | 988 | 3.09 | -14.66 | -7.66 | 0.55 | 16.80 |
| Keskin açılış adil fiyatı (model yok) | 8 | 0 | 8 | 3.48 | -64.00 | -3.18 | 4.81 | 37.50 |
| Model + keskin piyasa onayı (predict'teki VALUE kuralı) | 11 | 0 | 11 | 4.36 | 25.45 | 0.43 | 5.81 | 63.64 |

Kurallar backtest'tekiyle aynı (bkz. reports/backtest_summary.md → Strateji karşılaştırması).

## İddaa fiyatlarının değeri (keskin kapanışa göre CLV)

| grup | sonuç sayısı | maç | ort. oran | CLV ilk oran_% | ±GA_% | CLV son oran_% | CLV>0_% | ROI_% |
|---|---|---|---|---|---|---|---|---|
| Tüm İddaa fiyatları | 5243 | 1057 | 2.63 | -13.93 | 0.22 | -14.02 | 2.29 | -16.00 |
| İddaa > keskin açılış adil fiyatı | 7 | 6 | 2.97 | 0.43 | 7.27 | -13.38 | 42.86 | -63.71 |
| Model edge ≥ %3 (onaysız) | 610 | 435 | 4.16 | -16.57 | 0.85 | -17.03 | 5.25 | -25.12 |
| Model edge + keskin onay (VALUE) | 3 | 3 | 2.95 | 7.46 | 9.06 | -9.95 | 66.67 | -15.33 |

Sonuçlanmamış (açık) maç: 179. 'İlk oran' = İddaa'da ilk görülen, 'son oran' = başlamadan önceki son kayıt. Oran = `odd` (ekrandaki; `wodd` ayrıca kayıtlı). Ayrıntı: tracking/iddaa_clv.csv
