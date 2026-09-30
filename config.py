"""Proje ayarları. Parametrelerin çoğu komut satırından da ezilebilir (python main.py --help)."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Veri klasörü. Testler sentetik veriyi ayrı bir klasöre yazmak için VALUEBET_DATA_DIR kullanır;
# gerçek verin bu sayede asla karışmaz.
DATA_DIR = Path(os.environ.get("VALUEBET_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "football.db"
REPORTS_DIR = Path(os.environ.get("VALUEBET_REPORTS_DIR", ROOT / "reports"))

# ---------------------------------------------------------------- Veri
LEAGUES = {
    "SP1": "La Liga",
    "SP2": "La Liga 2 (Segunda)",
}
FIRST_SEASON = 2014  # 2014/15 sezonu. Model zaman ağırlıklı olduğu için daha eskisine gerek yok.


def current_season() -> int:
    """Sezon başlangıç yılı: Temmuz ve sonrası yeni sezon sayılır."""
    t = date.today()
    return t.year if t.month >= 7 else t.year - 1


SEASON_URL = "https://www.football-data.co.uk/mmz4281/{code}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# football-data.co.uk oran sütunları -> bizim kısa adlarımız.
#   b365 = Bet365 (maçtan 1-3 gün önce), ps = Pinnacle (maçtan 1-3 gün önce)
#   bfe = Betfair Exchange (borsa), avg / max = piyasa ortalaması / en iyi oran
#   sonunda "c" olanlar kapanış (closing) oranları.
# Kapsam (SP1/SP2): Pinnacle 1X2 kapanışı 2014/15'ten, Ü/A kapanışı 2019/20'den itibaren var, ama
# football-data'da Pinnacle 12.01.2026'da kesildi. Betfair Exchange (1X2 + Ü/A) 2024/25'ten itibaren var.
BOOKS = ["b365", "ps", "bfe", "avg", "max", "b365c", "psc", "bfec", "avgc", "maxc"]
_1X2_PREFIX = {"b365": "B365", "ps": "PS", "bfe": "BFE", "avg": "Avg", "max": "Max",
               "b365c": "B365C", "psc": "PSC", "bfec": "BFEC", "avgc": "AvgC", "maxc": "MaxC"}
_OU_PREFIX = {"b365": "B365", "ps": "P", "bfe": "BFE", "avg": "Avg", "max": "Max",
              "b365c": "B365C", "psc": "PC", "bfec": "BFEC", "avgc": "AvgC", "maxc": "MaxC"}


def odds_column_map() -> dict:
    """Kaynak CSV sütun adı -> veritabanı sütun adı. Birden fazla kaynak aynı hedefe gidebilir;
    ilk dolu olan kullanılır (site bazı sezonlarda farklı adlar kullanmış)."""
    m = {}
    for book in BOOKS:
        p = _1X2_PREFIX[book]
        m[f"{p}H"], m[f"{p}D"], m[f"{p}A"] = f"{book}_h", f"{book}_d", f"{book}_a"
        q = _OU_PREFIX[book]
        m[f"{q}>2.5"], m[f"{q}<2.5"] = f"{book}_o25", f"{book}_u25"
        if book.endswith("c"):  # notes.txt'deki alternatif yazım: PSHC, AvgHC, MaxHC ...
            base = _1X2_PREFIX[book[:-1]]
            m[f"{base}HC"], m[f"{base}DC"], m[f"{base}AC"] = f"{book}_h", f"{book}_d", f"{book}_a"
    # 2019 öncesi dosyalarda ortalama/en iyi oranlar Betbrain sütunlarında
    for src, dst in [("BbAv", "avg"), ("BbMx", "max")]:
        m[f"{src}H"], m[f"{src}D"], m[f"{src}A"] = f"{dst}_h", f"{dst}_d", f"{dst}_a"
        m[f"{src}>2.5"], m[f"{src}<2.5"] = f"{dst}_o25", f"{dst}_u25"
    return m


# ---------------------------------------------------------------- Model (Dixon-Coles)
# XI/RIDGE `python main.py tune` ile seçildi (2016-2021 ayar, 2022+ test; 2026-09-30). 1X2 için en iyi
# bölge iki ligde de xi 0.0019-0.0025, ridge 2-8 ve oldukça düz; ortak ayar xi=0.0019, ridge=4. Test
# döneminde eski ridge=0.5'e göre 1X2 log loss: SP2 -0.0064 ± 0.0042 (anlamlı), SP1 -0.0004 ± 0.0029.
# Ridge = atak/defans parametrelerine sd = 1/sqrt(2*ridge) olan normal önsel; 4 -> sd ≈ 0.35.
# Güç ve eğilim çekimi ayrı ayarlandı (ölçüt 1X2 + Ü/A toplamı): 1X2 neredeyse sadece RIDGE'e, Ü/A neredeyse
# sadece RIDGE_TOTAL'a bağlı; Ü/A için en iyi RIDGE_TOTAL iki ligde de 16. Test döneminde RIDGE_TOTAL=RIDGE'e
# göre Ü/A log loss: SP2 -0.0041 ± 0.0029 (anlamlı), SP1 -0.0004 ± 0.0030.
XI = 0.0019           # günlük zaman sönümü; yarı ömür = ln(2)/XI ≈ 365 gün
RIDGE = 4.0           # güç (atak - defans) yönünde L2 çekimi (az maçlı takımları ortalamaya çeker)
RIDGE_TOTAL = 16.0    # eğilim (atak + defans, Ü/A) yönünde L2 çekimi; None = RIDGE ile aynı
LOOKBACK_DAYS = 3 * 365
MAX_GOALS = 10        # skor matrisi 0..MAX_GOALS

# ---------------------------------------------------------------- Backtest / value
EVAL_FROM_SEASON = 2016   # 1X2 kapanışı 2014'ten, Ü/A kapanışı 2019'dan var; 2 sezon eğitim geçmişi bırakılır
BET_BOOK = "b365"         # backtest'te "oynadığımız" oranlar (İddaa geçmişi olmadığı için yumuşak bir büro vekil)
CLOSING_BOOK = "sharpc"   # kıyas noktası: keskin kapanış (piyasanın en keskin tahmini, aşağıya bak)

# Keskin piyasa = Pinnacle, yoksa Betfair Exchange. Market bazında, satır satır ilk tam ve geçerli kaynak
# alınır (sharp_* = açılış, sharpc_* = kapanış). Pinnacle ile borsanın çakıştığı ~1.200 maçta (2024/25 -
# Oca 2026) ikisi aynı derecede keskin çıktı (log loss farkı ±0,002 içinde, anlamsız). Maçtan günler önce
# toplanan borsa fiyatları La Liga 2'de sığ olabiliyor (marj %90'a kadar); marjı EXCHANGE_MAX_MARGIN'i aşan
# borsa fiyatı yok sayılır (%10'da SP2 açılışlarının %85'i kalıyor ve Pinnacle kadar iyi).
SHARP_SOURCES = {"sharp": ["ps", "bfe"], "sharpc": ["psc", "bfec"]}
EXCHANGE_BOOKS = {"bfe", "bfec"}
EXCHANGE_MAX_MARGIN = 0.10
BOOK_LABELS = {"sharp": "Keskin açılış PS/BFE", "sharpc": "Keskin kapanış PSC/BFEC"}


def book_label(book: str) -> str:
    return BOOK_LABELS.get(book, book.upper())


DEVIG_METHOD = "power"    # "power" veya "proportional"
EDGE_MIN = 0.03           # model_olasılığı * oran - 1 >= %3 ise value say
MAX_ODDS = 6.0            # çok yüksek oranlı sürprizleri dışarıda bırak (model hatası burada en büyük)
MIN_TEAM_MATCHES = 5      # eğitim penceresinde bu kadar maçı olmayan takımlara bahis yok
KELLY_FRACTION = 0.25     # çeyrek Kelly
KELLY_CAP = 0.02          # tek bahiste kasanın en fazla %2'si
START_BANKROLL = 1000.0
