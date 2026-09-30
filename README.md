# La Liga Value Bet Sistemi — Faz 0 + Faz 1

Tarihsel La Liga ve La Liga 2 verisini indirir, zaman ağırlıklı **Dixon-Coles** gol modelini kurar,
modeli **keskin kapanış oranlarına** karşı dürüstçe test eder (walk-forward backtest) ve yaklaşan maçlar
için olasılık, adil oran ve value adaylarını listeler.

**Keskin piyasa** = Pinnacle, o yoksa Betfair Exchange (borsa). football-data.co.uk'taki Pinnacle oranları
12.01.2026'da kesildi; borsa oranları 2024/25'ten beri var. İkisinin çakıştığı ~1.200 maçta borsa kapanışı
Pinnacle kapanışı kadar keskin çıktı. Ayrıntı ve marj filtresi: `config.py` (`SHARP_SOURCES`).

## 1. Kurulum (bir kez)

Gereken: **Python 3.10+** (python.org'dan). Proje klasörünü VS Code'da aç, menüden **Terminal → New Terminal** ile terminali aç:

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kurulumu doğrula (internet gerekmez, sahte veriyle çalışır, ~15 sn):
```bash
python tests/test_model.py
```

## 2. Kullanım

```bash
python main.py download     # football-data.co.uk'tan 2014/15'ten bugüne SP1 + SP2 (~1 dk)
python main.py build        # CSV'ler -> data/football.db (SQLite) + sezon başına oran doluluk tablosu
python main.py backtest     # walk-forward backtest -> reports/backtest_summary.md (~30 sn)
python main.py tune         # XI/ridge ızgara taraması (ayar 2016-2021, test 2022+) -> reports/tune_summary.md (~2,5 dk)
python main.py diagnose     # model-kapanış açığı hangi maçlarda yoğun -> reports/diagnose_summary.md
python main.py predict      # yaklaşan maçlar: olasılıklar, adil oranlar, value adayları
python main.py ratings      # güncel takım güç sıralaması (iki lig ortak ölçekte)
python main.py teams        # veritabanındaki takım isimleri
```

Kısayol: `python main.py all` = download + build + backtest.

Yeni haftada güncellemek için: `python main.py download && python main.py build && python main.py predict`
(bitmiş sezonlar tekrar indirilmez, sadece içinde bulunulan sezon güncellenir).

**Türkiye'den (football-data.co.uk BTK engelli):** `download` yerelde çalışmaz. Dosyaları GitHub Actions
(`.github/workflows/veri-guncelle.yml`) Pazartesi, Salı, Perşembe ve Cuma günleri indirip `data/raw`'a commit eder.
Yerelde sadece:
```bash
git pull
python main.py build
```
Hemen güncel veri gerekirse: `gh workflow run veri-guncelle.yml`, bitince `git pull`.

Faydalı seçenekler:
```bash
python main.py backtest --div SP1 --edge 0.05 --book ps     # sadece La Liga, %5 eşik, Pinnacle ön-oranları
python main.py backtest --xi 0.003 --ridge 1.0 --ridge-total 32   # model parametrelerini dene
python main.py tune --xi-grid 0.0019 --ridge-grid 2,4,8 --ridge-total-grid 8,16,32   # en fazla 2 eksen değişir
python main.py backtest --no-joint                           # her lig ayrı model (varsayılan: SP1+SP2 birlikte)
python main.py diagnose --base reports/baz                   # iki model sürümünü aynı maçlarda dilim dilim kıyasla
```

**Model:** SP1 ve SP2 tek bir Dixon-Coles modelinde, ortak güç ölçeğinde eğitilir; lig değiştiren takımların
geçmişi kaybolmaz. Her takım kendi liginin ortalamasına çekilir; bu sezon lige yeni çıkan / düşen takımlar için
ayrıca öğrenilen bir sapma vardır (yoksa çıkan takımlar sistematik olarak fazla güçlü görünür). Ayrıntı:
`src/dixon_coles.py` başındaki açıklama.

**Motivasyon:** maç öncesi puan durumundan her takımın hâlâ bir "hedefi" (şampiyonluk, Avrupa, çıkış, play-off,
küme hattı) olup olmadığı hesaplanır (`src/features.py`). Sezon sonunda sadece bir takımın hedefi varsa, o takımın
gücü sabit bir miktar (`MOTIVATION_M`) artırılır. Piyasa bunu fiyatlıyordu, model bilmiyordu; kapatmak modelin
kapanışa uzaklığını belirgin azalttı. `--no-motivation` ile kapatılabilir.

Bir model değişikliğini denerken: önce `reports/backtest_preds_*.csv` dosyalarını bir klasöre kopyala (baz),
değişikliği yapıp `backtest` çalıştır, sonra `diagnose --base <klasör>` ile değişikliğin hangi maçlarda işe
yaradığını gör.

### Canlı takip (kağıt üzerinde)

GitHub Actions her veri güncellemesinden sonra `python main.py track` çalıştırır ve sonuçları depoya yazar:

- `tracking/predictions.csv`: yaklaşan her maç, **ilk görüldüğü anda**: model olasılıkları, o anki B365 oranları,
  keskin açılış (Pinnacle / Betfair) oranları. Maç oynanıp veri gelince sonuç ve keskin kapanış eklenir.
- `tracking/bets.csv`: backtest'teki üç kuralın kağıt bahisleri (B365 oranıyla, 1 birim); sonuçlanınca CLV ve kâr.
- `tracking/summary.md`: canlı log loss (model − kapanış) ve kural bazında CLV / ROI.

Görmek için `git pull` yeterli. Model değişince `config.MODEL_VERSION`'ı artır; kayıtlar hangi modelden geldiğini
tutar. Birkaç yüz maç birikmeden sonuçlar çok gürültülüdür.

### İddaa oranlarıyla value arama

`iddaa_ornek.csv` dosyasını kopyala, oranları İddaa'dan elle gir (örnekteki oranlar temsilidir):

```csv
home,away,odd_1,odd_x,odd_2,odd_over25,odd_under25,date
Real Madrid,Villarreal,1.45,4.40,5.90,1.62,2.10,04.10.2026
```

```bash
python main.py predict --iddaa iddaa.csv
```

Takım isimleri football-data.co.uk'taki isimlerle eşleştirilir (ör. "Atletico Madrid" → "Ath Madrid").
Eşleşmeyen isim olursa uyarı verir; `team_aliases.csv` dosyasına `alias,team` satırı eklemen yeterli.
Doğru isimleri `python main.py teams` gösterir.

## 3. Çıktıları okumak

**`reports/backtest_summary.md`**, en önemli dosya:

| Metrik | Anlamı |
|---|---|
| log loss / Brier / RPS | Olasılık tahmininin kalitesi. **Düşük = iyi.** Modeli, B365'in ve keskin kapanışın (PSC/BFEC) marjsız olasılıklarıyla aynı maçlarda karşılaştırır. "Kapanış kaynağı" satırı kaç maçta hangisinin kullanıldığını gösterir. |
| Log loss farkı ± %95 GA | Maç başına eşleşmiş fark (model − kıyas). Aralık 0'ı içeriyorsa fark anlamlı değil. Model sürümlerini kıyaslarken asıl bakılacak sayı. |
| Kalibrasyon | Model %40 dediğinde gerçekten ~%40 mı oluyor? |
| ROI / t istatistiği | Düz 1 birim bahisle kâr. \|t\| < 2 ise sonuç büyük ihtimalle şans. |
| **CLV** / CLV_t | Oynanan oranın keskin kapanış adil fiyatına göre değeri. **Uzun vadeli kârlılığın en güvenilir erken göstergesi.** ROI'den çok daha az gürültülü; \|CLV_t\| ≥ 2 anlamlı. |
| Strateji karşılaştırması | Aynı eşikte üç kural: sadece model, sadece keskin açılış adil fiyatı (model yok), ikisi birden (`predict`'teki VALUE kuralı). Modelin CLV'si keskin açılış satırını geçmiyorsa, model bahis anındaki piyasa bilgisine bir şey eklemiyor. |
| Edge eşiğine duyarlılık | Eşik yükseldikçe CLV artıyor mu? Artmıyorsa modelin "büyük edge" dediği yerler aslında model hatası. |

Dürüst beklenti: saf Dixon-Coles modeli büyük ihtimalle keskin kapanıştan **daha kötü** çıkacak.
Bu normal ve Faz 1'in amacı da bunu ölçmek. Sonraki fazlardaki iyileştirmelerin (xG, kadro, ajan) katkısı
bu taban çizgisine göre ölçülecek.

**`python main.py predict`** her maç için iki edge verir:

- `edge_model` = model olasılığı × oran − 1 → modelin görüşü
- `edge_keskin` = keskin piyasa (Pinnacle, yoksa borsa) adil olasılığı × oran − 1 → oran, en keskin piyasanın
  adil fiyatını geçiyor mu?

`VALUE` işareti ikisi birden pozitifse konur. `sadece model` = model value görüyor ama keskin piyasa görmüyor;
bu durumda hata büyük ihtimalle modelde.

## 4. Ayarlar

Tüm ayarlar `config.py` içinde: ligler, ilk sezon, zaman sönümü (`XI`), ridge, edge eşiği, maksimum oran,
Kelly oranı vb.

## 5. Klasör yapısı

```
main.py              komut satırı
config.py            ayarlar
src/download.py      veri indirme
src/db.py            SQLite şeması, CSV temizleme/yükleme
src/dixon_coles.py   model (analitik gradyanlı, zaman ağırlıklı, ridge'li, birleşik ligler, motivasyon)
src/features.py      maç öncesi puan durumu ve motivasyon
src/tracking.py      kağıt üzerinde canlı takip (tracking/ klasörüne yazar)
src/odds.py          marj temizleme (proportional / power)
src/backtest.py      walk-forward backtest, metrikler, value simülasyonu, rapor
src/tune.py          XI/ridge ayarı (paralel ızgara, ayar/test dönemi ayrımı)
src/diagnose.py      açık analizi (tur, yeni takım, dinlenme, favori, sezon dilimleri; baz kıyası)
src/predict.py       yaklaşan maçlar + İddaa eşleştirme
tests/               testler + sahte veri üretici (internetsiz test için)
data/                indirilen veri ve veritabanı (otomatik oluşur)
reports/             raporlar ve tahmin CSV'leri (otomatik oluşur)
```

Veritabanındaki `predictions` tablosu, `predict` her çalıştığında tahminleri zaman damgasıyla saklar;
Faz 4-5'teki canlı takip buradan beslenecek.

## 6. Sonraki fazlar

- **Faz 2:** xG tabanlı güç (Understat), dinlenme günü, Avrupa kupası yoğunluğu, `XI`/ridge optimizasyonu
- **Faz 3:** Haber/sakatlık okuyan ajan; modele sınırlı ve gerekçeli düzeltme, katkısı ayrıca ölçülür
- **Faz 4:** Kağıt üzerinde canlı takip, CLV kaydı, kasa yönetimi
- **Faz 5:** Günlük otomatik çalışma + Telegram bildirimi

Veri kaynağı: [football-data.co.uk](https://www.football-data.co.uk) (ücretsiz). Sayfayı gereksiz yere
sık indirme; haftada bir iki kez yeterli.
