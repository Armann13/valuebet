"""football-data.co.uk'tan sezon CSV'lerini ve yaklaşan maç (fixtures) dosyasını indirir."""
import time
from pathlib import Path

import requests

import config

HEADERS = {"User-Agent": "Mozilla/5.0 (valuebet research script)"}


class ConnectionProblem(Exception):
    pass


def season_code(start_year: int) -> str:
    """2024 -> '2425'"""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def _explain(err: Exception) -> str:
    msg = str(err)
    if "btk.gov.tr" in msg:
        return ("football-data.co.uk bu internet bağlantısından erişime kapalı (BTK erişim engeli). "
                "Kodla ilgili bir sorun değil.\n"
                "  Dosyaları başka bir yoldan edinip data/raw klasörüne koyabilirsin; dosya adları önemli değil,\n"
                "  'python main.py build' klasördeki tüm CSV'leri okuyup lig ve sezonu kendisi bulur.")
    return f"football-data.co.uk'a bağlanılamadı: {msg[:200]}"


def _get(url: str, dest: Path) -> bool:
    """True = indirildi, False = dosya yok (404 vb.), ConnectionProblem = siteye hiç ulaşılamadı."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        raise ConnectionProblem(_explain(e)) from e
    if r.status_code != 200 or len(r.content) < 200:
        print(f"  - bulunamadı ({r.status_code}): {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def download_seasons(force: bool = False) -> bool:
    cur = config.current_season()
    for div in config.LEAGUES:
        for y in range(config.FIRST_SEASON, cur + 1):
            dest = config.RAW_DIR / f"{div}_{season_code(y)}.csv"
            # Bitmiş sezonlar bir kez indirilir; içinde bulunduğumuz sezon her seferinde güncellenir.
            if dest.exists() and y < cur and not force:
                continue
            url = config.SEASON_URL.format(code=season_code(y), div=div)
            try:
                if _get(url, dest):
                    print(f"  + {dest.name}")
            except ConnectionProblem as e:
                print(f"  ! {e}")
                return False
            time.sleep(1.0)  # siteye nazik davran
    return True


def download_fixtures() -> Path | None:
    dest = config.RAW_DIR / "fixtures.csv"
    try:
        if _get(config.FIXTURES_URL, dest):
            print(f"  + {dest.name}")
            return dest
    except ConnectionProblem as e:
        print(f"  ! {e}")
    return dest if dest.exists() else None
