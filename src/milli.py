"""Milli takımlar: sonuç verisi, Türkçe ülke adları, Dixon-Coles (tarafsız saha + hazırlık maçı ağırlığı),
walk-forward doğrulama ve ayar.

Veri: config.MILLI_URL -> data/raw/milli/results.csv (python main.py download). Sadece FIFA üyeleri ve CONCACAF'ın
FIFA dışı üyeleri (Guadeloupe, Martinik ...) arasındaki maçlar kullanılır; CONIFA / bölge takımları (Bask Ülkesi,
Jersey ...) dışarıda. Bu veri seti haftalarca gecikebildiği için son maçlar ESPN'den (config.ESPN_URL) alınıp
data/raw/milli/espn.csv'de biriktirilir (python main.py milli --espn); aynı maç ana veri setine eklenince onunki
kullanılır. ESPN'de saha bilgisi yok: büyük turnuvalarda tarafsız, diğerlerinde ev sahibi kendi sahasında sayılır.

Model: kulüp modeliyle aynı Dixon-Coles, tek lig (config.MILLI). Tarafsız sahada ev avantajı yok; hazırlık maçları
(ve davet turnuvaları) MILLI_FRIENDLY_WEIGHT ağırlıklı (kadro denemesi, düşük motivasyon). Milli takımlar yılda
~10 maç oynadığı için pencere (6 yıl) ve yarı ömür kulüp modelinden uzun.

Doğrulama: milli maçlar için keskin kapanış oranı yok; ölçüt gerçekleşen sonuçlara göre log loss (1X2 ve 2.5 Ü/A),
geçmiş yüzdelerle yapılan sabit bir tahmine karşı. Ayar 2016-2021 maçlarıyla, test 2022 ve sonrası.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
import requests

import config
from src.dixon_coles import DixonColes

# İddaa'daki Türkçe ad -> veri setindeki (martj42) ad. Bir ülkenin birden çok yazımı olabilir.
ULKELER = {
    # UEFA
    "Arnavutluk": "Albania", "Andorra": "Andorra", "Ermenistan": "Armenia", "Avusturya": "Austria",
    "Azerbaycan": "Azerbaijan", "Belarus": "Belarus", "Beyaz Rusya": "Belarus", "Belçika": "Belgium",
    "Bosna Hersek": "Bosnia and Herzegovina", "Bosna-Hersek": "Bosnia and Herzegovina", "Bulgaristan": "Bulgaria",
    "Hırvatistan": "Croatia", "Kıbrıs": "Cyprus", "Güney Kıbrıs": "Cyprus", "Güney Kıbrıs Rum Kesimi": "Cyprus",
    "Kıbrıs Rum Kesimi": "Cyprus", "GKRY": "Cyprus", "Çekya": "Czech Republic", "Çek Cumhuriyeti": "Czech Republic",
    "Danimarka": "Denmark", "İngiltere": "England", "Estonya": "Estonia", "Faroe Adaları": "Faroe Islands",
    "Faroe": "Faroe Islands", "Finlandiya": "Finland", "Fransa": "France", "Gürcistan": "Georgia",
    "Almanya": "Germany", "Cebelitarık": "Gibraltar", "Yunanistan": "Greece", "Macaristan": "Hungary",
    "İzlanda": "Iceland", "İrlanda": "Republic of Ireland", "İrlanda Cumhuriyeti": "Republic of Ireland",
    "İsrail": "Israel", "İtalya": "Italy", "Kazakistan": "Kazakhstan", "Kosova": "Kosovo", "Letonya": "Latvia",
    "Lihtenştayn": "Liechtenstein", "Litvanya": "Lithuania", "Lüksemburg": "Luxembourg", "Malta": "Malta",
    "Moldova": "Moldova", "Karadağ": "Montenegro", "Hollanda": "Netherlands", "Kuzey Makedonya": "North Macedonia",
    "Makedonya": "North Macedonia", "Kuzey İrlanda": "Northern Ireland", "Norveç": "Norway", "Polonya": "Poland",
    "Portekiz": "Portugal", "Romanya": "Romania", "Rusya": "Russia", "San Marino": "San Marino",
    "İskoçya": "Scotland", "Sırbistan": "Serbia", "Slovakya": "Slovakia", "Slovenya": "Slovenia",
    "İspanya": "Spain", "İsveç": "Sweden", "İsviçre": "Switzerland", "Türkiye": "Turkey", "Ukrayna": "Ukraine",
    "Galler": "Wales",
    # CONMEBOL
    "Arjantin": "Argentina", "Bolivya": "Bolivia", "Brezilya": "Brazil", "Şili": "Chile", "Kolombiya": "Colombia",
    "Ekvador": "Ecuador", "Paraguay": "Paraguay", "Peru": "Peru", "Uruguay": "Uruguay", "Venezuela": "Venezuela",
    # CONCACAF (son altısı FIFA üyesi değil ama CONCACAF turnuvalarında oynuyor)
    "Anguilla": "Anguilla", "Antigua ve Barbuda": "Antigua and Barbuda", "Antigua & Barbuda": "Antigua and Barbuda",
    "Aruba": "Aruba", "Bahamalar": "Bahamas", "Bahamas": "Bahamas", "Barbados": "Barbados", "Belize": "Belize",
    "Bermuda": "Bermuda", "Britanya Virjin Adaları": "British Virgin Islands",
    "İngiliz Virjin Adaları": "British Virgin Islands", "İngiliz Virgin Adaları": "British Virgin Islands",
    "Britanya Virgin Adaları": "British Virgin Islands", "Kanada": "Canada", "Cayman Adaları": "Cayman Islands",
    "Kosta Rika": "Costa Rica", "Küba": "Cuba", "Curaçao": "Curaçao", "Curacao": "Curaçao", "Dominika": "Dominica",
    "Dominik Cumhuriyeti": "Dominican Republic", "El Salvador": "El Salvador", "Grenada": "Grenada",
    "Guatemala": "Guatemala", "Guyana": "Guyana", "Haiti": "Haiti", "Honduras": "Honduras", "Jamaika": "Jamaica",
    "Meksika": "Mexico", "Montserrat": "Montserrat", "Nikaragua": "Nicaragua", "Panama": "Panama",
    "Porto Riko": "Puerto Rico", "Saint Kitts ve Nevis": "Saint Kitts and Nevis",
    "St. Kitts ve Nevis": "Saint Kitts and Nevis", "St Kitts ve Nevis": "Saint Kitts and Nevis",
    "Saint Lucia": "Saint Lucia", "St. Lucia": "Saint Lucia", "St Lucia": "Saint Lucia",
    "Saint Vincent ve Grenadinler": "Saint Vincent and the Grenadines",
    "St. Vincent ve Grenadinler": "Saint Vincent and the Grenadines",
    "St Vincent ve Grenadinler": "Saint Vincent and the Grenadines", "Surinam": "Suriname",
    "Trinidad ve Tobago": "Trinidad and Tobago", "Trinidad & Tobago": "Trinidad and Tobago",
    "Trinidad Tobago": "Trinidad and Tobago", "Turks ve Caicos Adaları": "Turks and Caicos Islands",
    "ABD": "United States", "Amerika Birleşik Devletleri": "United States", "Amerika": "United States",
    "ABD Virjin Adaları": "United States Virgin Islands", "Amerika Virjin Adaları": "United States Virgin Islands",
    "ABD Virgin Adaları": "United States Virgin Islands", "Amerika Virgin Adaları": "United States Virgin Islands",
    "Guadeloupe": "Guadeloupe", "Guadalupe": "Guadeloupe", "Martinik": "Martinique",
    "Fransız Guyanası": "French Guiana", "Sint Maarten": "Sint Maarten", "Bonaire": "Bonaire",
    "Saint Martin": "Saint Martin",
    # CAF
    "Cezayir": "Algeria", "Angola": "Angola", "Benin": "Benin", "Botsvana": "Botswana", "Burkina Faso": "Burkina Faso",
    "Burundi": "Burundi", "Kamerun": "Cameroon", "Yeşil Burun Adaları": "Cape Verde", "Yeşil Burun": "Cape Verde",
    "Cape Verde": "Cape Verde", "Orta Afrika Cumhuriyeti": "Central African Republic", "Çad": "Chad",
    "Komorlar": "Comoros", "Komor Adaları": "Comoros", "Kongo": "Congo", "Kongo DC": "DR Congo",
    "DR Kongo": "DR Congo", "Demokratik Kongo Cumhuriyeti": "DR Congo", "Kongo Demokratik Cumhuriyeti": "DR Congo",
    "Cibuti": "Djibouti", "Mısır": "Egypt", "Ekvator Ginesi": "Equatorial Guinea", "Eritre": "Eritrea",
    "Esvatini": "Eswatini", "Svaziland": "Eswatini", "Etiyopya": "Ethiopia", "Gabon": "Gabon", "Gambiya": "Gambia",
    "Gana": "Ghana", "Gine": "Guinea", "Gine-Bissau": "Guinea-Bissau", "Fildişi Sahili": "Ivory Coast",
    "Kenya": "Kenya", "Lesoto": "Lesotho", "Liberya": "Liberia", "Libya": "Libya", "Madagaskar": "Madagascar",
    "Malavi": "Malawi", "Mali": "Mali", "Moritanya": "Mauritania", "Mauritius": "Mauritius", "Morityus": "Mauritius",
    "Fas": "Morocco", "Mozambik": "Mozambique", "Namibya": "Namibia", "Nijer": "Niger", "Nijerya": "Nigeria",
    "Ruanda": "Rwanda", "Sao Tome ve Principe": "São Tomé and Príncipe", "Senegal": "Senegal",
    "Seyşeller": "Seychelles", "Sierra Leone": "Sierra Leone", "Somali": "Somalia", "Güney Afrika": "South Africa",
    "Güney Sudan": "South Sudan", "Sudan": "Sudan", "Tanzanya": "Tanzania", "Togo": "Togo", "Tunus": "Tunisia",
    "Uganda": "Uganda", "Zambiya": "Zambia", "Zimbabve": "Zimbabwe",
    # AFC
    "Afganistan": "Afghanistan", "Avustralya": "Australia", "Bahreyn": "Bahrain", "Bangladeş": "Bangladesh",
    "Butan": "Bhutan", "Brunei": "Brunei", "Kamboçya": "Cambodia", "Çin": "China", "Çin Halk Cumhuriyeti": "China",
    "Çin Taipei": "Taiwan", "Tayvan": "Taiwan", "Guam": "Guam", "Hong Kong": "Hong Kong", "Hindistan": "India",
    "Endonezya": "Indonesia", "İran": "Iran", "Irak": "Iraq", "Japonya": "Japan", "Ürdün": "Jordan",
    "Kuveyt": "Kuwait", "Kırgızistan": "Kyrgyzstan", "Laos": "Laos", "Lübnan": "Lebanon", "Makao": "Macau",
    "Malezya": "Malaysia", "Maldivler": "Maldives", "Moğolistan": "Mongolia", "Myanmar": "Myanmar", "Nepal": "Nepal",
    "Kuzey Kore": "North Korea", "Umman": "Oman", "Pakistan": "Pakistan", "Filistin": "Palestine",
    "Filipinler": "Philippines", "Katar": "Qatar", "Suudi Arabistan": "Saudi Arabia", "Singapur": "Singapore",
    "Güney Kore": "South Korea", "Sri Lanka": "Sri Lanka", "Suriye": "Syria", "Tacikistan": "Tajikistan",
    "Tayland": "Thailand", "Doğu Timor": "Timor-Leste", "Türkmenistan": "Turkmenistan",
    "Birleşik Arap Emirlikleri": "United Arab Emirates", "BAE": "United Arab Emirates", "Özbekistan": "Uzbekistan",
    "Vietnam": "Vietnam", "Yemen": "Yemen",
    # OFC
    "Amerikan Samoası": "American Samoa", "Cook Adaları": "Cook Islands", "Fiji": "Fiji",
    "Yeni Kaledonya": "New Caledonia", "Yeni Zelanda": "New Zealand", "Papua Yeni Gine": "Papua New Guinea",
    "Samoa": "Samoa", "Solomon Adaları": "Solomon Islands", "Tahiti": "Tahiti", "Tonga": "Tonga", "Vanuatu": "Vanuatu",
}
TEAMS = set(ULKELER.values())

# Hazırlık maçı sayılan (sonucu daha az bilgi taşıyan) turnuvalar: dostluk maçları ve davet turnuvaları
FRIENDLY_LIKE = {"Friendly", "FIFA Series", "Kirin Cup", "Kirin Challenge Cup", "King's Cup", "CONCACAF Series",
                 "Unity Cup", "Navruz Cup", "Tri Nation Tournament", "Merdeka Tournament", "Three Nations Cup",
                 "Tri-Nations Series", "Tri-Nations Cup", "Mauritius Four Nations Cup", "Jordan International Tournament",
                 "Al Ain International Cup", "Mukuru 4 Nations", "Canadian Shield", "Mahinda Rajapaksa Cup",
                 "Superclásico de las Américas", "Morocco, Capital of African Football", "Mapinduzi Cup",
                 "Intercontinental Cup", "Hungary Heritage Cup", "Atlantic Heritage Cup", "Soccer Ashes"}


# ESPN turnuva kodu -> veri setindeki turnuva adı (hazırlık maçı tespiti ve rapor için)
ESPN_LEAGUES = {
    "uefa.nations": "UEFA Nations League", "fifa.friendly": "Friendly",
    "concacaf.nations.league": "CONCACAF Nations League", "fifa.worldq.uefa": "FIFA World Cup qualification",
    "fifa.worldq.conmebol": "FIFA World Cup qualification", "fifa.worldq.concacaf": "FIFA World Cup qualification",
    "fifa.worldq.afc": "FIFA World Cup qualification", "fifa.worldq.caf": "FIFA World Cup qualification",
    "fifa.worldq.ofc": "FIFA World Cup qualification", "uefa.euroq": "UEFA Euro qualification",
    "caf.nations_qual": "African Cup of Nations qualification", "afc.cupq": "AFC Asian Cup qualification",
    "fifa.world": "FIFA World Cup", "uefa.euro": "UEFA Euro", "conmebol.america": "Copa América",
    "concacaf.gold": "Gold Cup", "caf.nations": "African Cup of Nations", "afc.asian.cup": "AFC Asian Cup",
    "aff.championship": "AFF Championship",
}
ESPN_FINALS = {"fifa.world", "uefa.euro", "conmebol.america", "concacaf.gold", "caf.nations", "afc.asian.cup",
               "aff.championship"}   # büyük turnuva: maçlar tarafsız sahada sayılır (ESPN saha bilgisi vermiyor)
# ESPN'deki ad -> veri setindeki ad (aynı olanlar gerekmez)
ESPN_ADLAR = {"Czechia": "Czech Republic", "US Virgin Islands": "United States Virgin Islands",
              "Brunei Darussalam": "Brunei", "Chinese Taipei": "Taiwan", "Türkiye": "Turkey", "Turkiye": "Turkey",
              "Korea Republic": "South Korea", "Korea DPR": "North Korea", "IR Iran": "Iran", "China PR": "China",
              "Congo DR": "DR Congo", "DR Congo": "DR Congo", "Côte d'Ivoire": "Ivory Coast", "Cote d'Ivoire": "Ivory Coast",
              "Cape Verde Islands": "Cape Verde", "Cabo Verde": "Cape Verde", "Kyrgyz Republic": "Kyrgyzstan",
              "Bosnia-Herzegovina": "Bosnia and Herzegovina", "St Kitts and Nevis": "Saint Kitts and Nevis",
              "St. Kitts and Nevis": "Saint Kitts and Nevis", "St Lucia": "Saint Lucia", "St. Lucia": "Saint Lucia",
              "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
              "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines", "USA": "United States",
              "Ireland": "Republic of Ireland", "Swaziland": "Eswatini", "East Timor": "Timor-Leste",
              "Gambia": "Gambia", "The Gambia": "Gambia", "Sao Tome and Principe": "São Tomé and Príncipe",
              "Turks and Caicos": "Turks and Caicos Islands", "Macao": "Macau", "Palestinian Territory": "Palestine",
              "St. Martin": "Saint Martin", "St Martin": "Saint Martin"}


def results_path():
    return config.RAW_DIR / "milli" / "results.csv"


def espn_path():
    return config.RAW_DIR / "milli" / "espn.csv"


def parse_espn(data: dict, league: str, resolve) -> tuple[list, set]:
    """ESPN scoreboard yanıtından biten maçlar (veri seti biçiminde). resolve: ESPN adı -> veri setindeki ad (ya da
    None). Uzatma / penaltıyla biten maçta skor 90 dakikanınkidir (devre skorlarından); yoksa maç atlanır.
    Döner: satırlar, eşleşmeyen adlar."""
    rows, unknown = [], set()
    for e in data.get("events", []):
        st = e.get("status", {}).get("type", {})
        if not st.get("completed"):
            continue
        comp = e["competitions"][0]
        side = {c.get("homeAway"): c for c in comp.get("competitors", [])}
        if set(side) != {"home", "away"}:
            continue
        names = {k: resolve(c["team"]["displayName"]) for k, c in side.items()}
        unknown |= {side[k]["team"]["displayName"] for k, v in names.items() if v is None}
        if None in names.values():
            continue
        try:
            if any(x in st.get("name", "") for x in ("AET", "PEN")):  # 90 dakika: ilk iki devre
                score = {k: sum(float(p["value"]) for p in c["linescores"][:2]) for k, c in side.items()}
            else:
                score = {k: float(c["score"]) for k, c in side.items()}
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({"date": e["date"][:10], "home_team": names["home"], "away_team": names["away"],
                     "home_score": int(score["home"]), "away_score": int(score["away"]),
                     "tournament": ESPN_LEAGUES[league], "city": "", "country": "",
                     "neutral": "TRUE" if league in ESPN_FINALS else "FALSE"})
    return rows, unknown


def download_espn(start: date | None = None) -> bool:
    """Son milli maç sonuçlarını ESPN'den çekip data/raw/milli/espn.csv'ye ekler. start verilmezse: dosya yoksa ana
    veri setinin son maçından 7 gün öncesi, varsa son 10 gün (aylık sorgu: ilgili aylar)."""
    from src.teams import TeamResolver
    today = date.today()
    if start is None:
        start = today - timedelta(days=10)
        if not espn_path().exists() and results_path().exists():
            last = pd.Timestamp(pd.read_csv(results_path(), usecols=["date"])["date"].max()).date()
            start = min(start, last - timedelta(days=7))
    months = sorted({(start + timedelta(days=i)).strftime("%Y%m") for i in range((today - start).days + 1)})
    resolve = TeamResolver(sorted(TEAMS), aliases=ESPN_ADLAR, allow_fuzzy=False)
    rows, unknown, failed = [], set(), 0
    for league in ESPN_LEAGUES:
        for m in months:
            try:
                r = requests.get(config.ESPN_URL.format(league=league), params={"dates": m}, timeout=30)
                r.raise_for_status()
                got, unk = parse_espn(r.json(), league, resolve)
                rows += got
                unknown |= unk
            except Exception:  # tek turnuva / ay alınamazsa diğerleri devam etsin
                failed += 1
            time.sleep(0.3)  # siteye nazik davran
    if failed == len(ESPN_LEAGUES) * len(months):
        print("  ! ESPN'e ulaşılamadı; milli sonuçlar güncellenmedi")
        return False
    new = pd.DataFrame(rows)
    path = espn_path()
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    if not new.empty:
        new = new.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last").sort_values("date")
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_csv(path, index=False)
    print(f"  + milli/espn.csv: {len(rows)} biten maç ({', '.join(months)}), toplam {len(new)}"
          + (f"; eşleşmeyen ad: {', '.join(sorted(unknown))}" if unknown else "")
          + (f"; {failed} sorgu başarısız" if failed else ""))
    return True


def _merge_espn(r: pd.DataFrame) -> pd.DataFrame:
    """Ana veri setine ESPN'deki ek maçlar (ana veri setinde ±1 gün içinde aynı eşleşme yoksa)."""
    if not espn_path().exists():
        return r
    e = pd.read_csv(espn_path())
    if e.empty:
        return r
    seen = set()
    recent = r[pd.to_datetime(r["date"]) >= pd.Timestamp(e["date"].min()) - pd.Timedelta(days=2)]
    for d, h, a in zip(pd.to_datetime(recent["date"]), recent["home_team"], recent["away_team"]):
        for k in (-1, 0, 1):
            day = (d + pd.Timedelta(days=k)).strftime("%Y-%m-%d")
            seen |= {(day, h, a), (day, a, h)}
    e = e[[(d, h, a) not in seen for d, h, a in zip(e["date"], e["home_team"], e["away_team"])]]
    return pd.concat([r, e], ignore_index=True) if len(e) else r


def download() -> bool:
    """data/raw/milli/results.csv'yi günceller (tek dosya, tüm tarih)."""
    try:
        r = requests.get(config.MILLI_URL, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! milli maç sonuçları indirilemedi: {str(e)[:200]}")
        return False
    if len(r.content) < 100_000:
        print(f"  ! milli maç sonuçları beklenenden kısa ({len(r.content)} bayt), eski dosya korunuyor")
        return False
    dest = results_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    last = pd.read_csv(dest, usecols=["date"])["date"].max()
    print(f"  + milli/results.csv (son maç {last})")
    return True


def load_results(path=None, since: str = "2008-01-01") -> pd.DataFrame:
    """Oynanmış milli maçlar (FIFA üyeleri arasında): date, div, home, away, fthg, ftag, tournament, neutral,
    friendly."""
    path = path or results_path()
    if not path.exists():
        return pd.DataFrame()
    r = _merge_espn(pd.read_csv(path))
    r = r.dropna(subset=["home_score", "away_score"])
    r = r[r["home_team"].isin(TEAMS) & r["away_team"].isin(TEAMS) & (r["date"] >= since)]
    df = pd.DataFrame({"date": pd.to_datetime(r["date"]), "div": config.MILLI, "home": r["home_team"],
                       "away": r["away_team"], "fthg": r["home_score"].astype(int),
                       "ftag": r["away_score"].astype(int), "tournament": r["tournament"],
                       "neutral": r["neutral"].astype(str).str.upper().eq("TRUE")})
    df["friendly"] = df["tournament"].isin(FRIENDLY_LIKE)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def model(xi=None, ridge=None, ridge_total=None) -> DixonColes:
    return DixonColes(xi=config.MILLI_XI if xi is None else xi, ridge=config.MILLI_RIDGE if ridge is None else ridge,
                      ridge_total=config.MILLI_RIDGE_TOTAL if ridge_total is None else ridge_total,
                      lookback_days=config.MILLI_LOOKBACK_DAYS)


def fit(df: pd.DataFrame, ref_date, xi=None, ridge=None, ridge_total=None, fw=None,
        warm_start: DixonColes | None = None) -> DixonColes:
    fw = config.MILLI_FRIENDLY_WEIGHT if fw is None else fw
    return model(xi, ridge, ridge_total).fit(df.assign(weight=np.where(df["friendly"], fw, 1.0)),
                                             ref_date=ref_date, warm_start=warm_start)


def fit_current(ref_date) -> DixonColes | None:
    """Tavsiye için: ref_date'ten önceki tüm maçlarla güncel ayarlı model (veri yoksa None)."""
    df = load_results()
    if df.empty:
        print("  ! milli maç verisi yok (python main.py download)")
        return None
    return fit(df, ref_date)


def _probs(m: np.ndarray) -> tuple:
    g = np.arange(m.shape[0])
    tot = g[:, None] + g[None, :]
    return float(np.tril(m, -1).sum()), float(np.trace(m)), float(np.triu(m, 1).sum()), float(m[tot > 2.5].sum())


def walk_forward(df: pd.DataFrame, eval_from: int = config.EVAL_FROM_SEASON, xi=None, ridge=None, ridge_total=None,
                 fw=None) -> pd.DataFrame:
    """Haftalık: her hafta (Pzt-Paz) o haftanın ilk maç gününden önceki maçlarla eğitilir, haftanın maçları
    tahmin edilir. Döner: maç başına p_home, p_draw, p_away, p_over25."""
    ev = df[df["date"].dt.year >= eval_from]
    rows, prev = [], None
    for _, g in ev.groupby(ev["date"].dt.to_period("W-SUN")):
        prev = fit(df, g["date"].min(), xi, ridge, ridge_total, fw, warm_start=prev)
        for r in g.itertuples(index=False):
            ph, pd_, pa, po = _probs(prev.score_matrix(r.home, r.away, config.MILLI, 0, r.neutral))
            rows.append({"date": r.date, "home": r.home, "away": r.away, "fthg": r.fthg, "ftag": r.ftag,
                         "tournament": r.tournament, "neutral": r.neutral, "friendly": r.friendly,
                         "p_home": ph, "p_draw": pd_, "p_away": pa, "p_over25": po,
                         "known": r.home in prev.teams and r.away in prev.teams})
    return pd.DataFrame(rows)


def _losses(p: pd.DataFrame, base: dict) -> pd.DataFrame:
    """Maç başına log loss: model ve sabit tahmin (geçmiş yüzdeler; tarafsız sahada simetrik)."""
    y = np.select([p["fthg"] > p["ftag"], p["fthg"] == p["ftag"]], [0, 1], 2)
    pm = p[["p_home", "p_draw", "p_away"]].to_numpy()
    over = (p["fthg"] + p["ftag"] > 2.5).to_numpy()
    b = np.array([base["neutral"] if n else base["home"] for n in p["neutral"]])
    return pd.DataFrame({
        "model_1x2": -np.log(np.clip(pm[np.arange(len(p)), y], 1e-12, None)),
        "sabit_1x2": -np.log(b[np.arange(len(p)), y]),
        "model_ou": -np.log(np.clip(np.where(over, p["p_over25"], 1 - p["p_over25"]), 1e-12, None)),
        "sabit_ou": -np.log(np.where(over, base["over"], 1 - base["over"])),
    }, index=p.index)


def base_rates(df: pd.DataFrame, before) -> dict:
    """Sabit tahmin: before'dan önceki maçlarda ev / beraberlik / deplasman ve 2.5 üst yüzdeleri."""
    h = df[df["date"] < pd.Timestamp(before)]
    res = np.select([h["fthg"] > h["ftag"], h["fthg"] == h["ftag"]], [0, 1], 2)
    home = np.bincount(res[~h["neutral"].to_numpy()], minlength=3) / (~h["neutral"]).sum()
    neu = np.bincount(res[h["neutral"].to_numpy()], minlength=3) / h["neutral"].sum()
    neu = np.array([(neu[0] + neu[2]) / 2, neu[1], (neu[0] + neu[2]) / 2])  # tarafsızda "ev" adı rastgele
    return {"home": home, "neutral": neu, "over": float((h["fthg"] + h["ftag"] > 2.5).mean())}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def _diff(a: pd.Series, b: pd.Series) -> str:
    d = a - b
    return f"{d.mean():+.4f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.4f}" if len(d) > 1 else "—"


def report(df: pd.DataFrame, preds: pd.DataFrame, eval_from: int, title: str) -> str:
    """Log loss (model ve sabit tahmin), dilimler ve kalibrasyon."""
    loss = _losses(preds, base_rates(df, f"{eval_from}-01-01"))
    segs = {"tümü": preds.index, "resmi maçlar": preds.index[~preds["friendly"]],
            "hazırlık maçları": preds.index[preds["friendly"]], "tarafsız saha": preds.index[preds["neutral"]],
            "2016-2021 (ayar dönemi)": preds.index[preds["date"].dt.year <= 2021],
            "2022+ (test dönemi)": preds.index[preds["date"].dt.year >= 2022]}
    rows = []
    for name, ix in segs.items():
        if len(ix) == 0:
            continue
        lo = loss.loc[ix]
        rows.append({"dilim": name, "maç": len(ix), "1X2 model": f"{lo['model_1x2'].mean():.4f}",
                     "1X2 sabit": f"{lo['sabit_1x2'].mean():.4f}", "fark (model − sabit)":
                     _diff(lo["model_1x2"], lo["sabit_1x2"]), "Ü/A model": f"{lo['model_ou'].mean():.4f}",
                     "Ü/A sabit": f"{lo['sabit_ou'].mean():.4f}"})
    # kalibrasyon: her maçın üç sonucu için tahmin edilen olasılık dilimine göre gerçekleşme oranı
    y = np.select([preds["fthg"] > preds["ftag"], preds["fthg"] == preds["ftag"]], [0, 1], 2)
    pm = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    flat = pd.DataFrame({"p": pm.ravel(), "hit": (np.arange(3)[None, :] == y[:, None]).ravel()})
    flat["dilim"] = pd.cut(flat["p"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cal = flat.groupby("dilim", observed=True).agg(tahmin=("p", "mean"), gerçek=("hit", "mean"), n=("p", "size"))
    cal = cal.reset_index().assign(dilim=lambda c: c["dilim"].astype(str), tahmin=lambda c: c["tahmin"].map("{:.1%}".format),
                                   gerçek=lambda c: c["gerçek"].map("{:.1%}".format))
    return "\n".join([title, "", _md_table(pd.DataFrame(rows)), "",
                      "Log loss küçük = iyi. 'Sabit' = her maça geçmiş yüzdeleri (ev / beraberlik / deplasman, "
                      "tarafsız sahada simetrik) veren tahmin; model en azından bunu açıkça yenmeli. Milli maçlar "
                      "için keskin kapanış oranı olmadığından kulüp raporlarındaki gibi piyasayla kıyas yapılamıyor.",
                      "", "### Kalibrasyon (1X2, tüm sonuçlar)", "", _md_table(cal), "",
                      "Tahmin edilen olasılık ile gerçekleşme oranı yakınsa olasılıklar güvenilir demektir."])


def _wf_job(args):
    df, eval_from, xi, ridge, rt, fw = args
    p = walk_forward(df, eval_from, xi, ridge, rt, fw)
    return (xi, ridge, rt, fw), p


def tune(df: pd.DataFrame, eval_from: int, tune_to: int, xis, ridges, rts, fws, workers: int) -> tuple[str, pd.DataFrame]:
    """Izgara: her ayar için walk-forward; seçim tune_to'ya kadarki maçlarda (1X2 + Ü/A log loss), sonrası test."""
    jobs = [(df, eval_from, *c) for c in product(xis, ridges, rts, fws)]
    base = (config.MILLI_XI, config.MILLI_RIDGE, config.MILLI_RIDGE_TOTAL, config.MILLI_FRIENDLY_WEIGHT)
    if base not in [j[2:] for j in jobs]:
        jobs.append((df, eval_from, *base))
    print(f"  milli ayar: {len(jobs)} ayar, {workers} paralel işçi ...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = dict(ex.map(_wf_job, jobs))
    br = base_rates(df, f"{eval_from}-01-01")
    rows = []
    for (xi, ridge, rt, fw), p in results.items():
        lo = _losses(p, br)
        sel, test = p["date"].dt.year <= tune_to, p["date"].dt.year > tune_to
        rows.append({"xi": xi, "ridge": ridge, "ridge_total": rt, "hazırlık_ağırlığı": fw,
                     "ayar_1x2": lo.loc[sel, "model_1x2"].mean(), "ayar_ou": lo.loc[sel, "model_ou"].mean(),
                     "test_1x2": lo.loc[test, "model_1x2"].mean(), "test_ou": lo.loc[test, "model_ou"].mean()})
    t = pd.DataFrame(rows)
    t["ayar_toplam"] = t["ayar_1x2"] + t["ayar_ou"]
    t = t.sort_values("ayar_toplam").reset_index(drop=True)
    best = tuple(t.loc[0, ["xi", "ridge", "ridge_total", "hazırlık_ağırlığı"]])
    pb, p0 = results[best], results[base]
    test = pb["date"].dt.year > tune_to
    lb, l0 = _losses(pb, br), _losses(p0, br)
    md = ["# Milli takım modeli — ayar", "",
          f"Seçim {eval_from}-{tune_to} maçlarında (1X2 + Ü/A log loss toplamı), test {tune_to + 1}+. "
          f"Pencere {config.MILLI_LOOKBACK_DAYS // 365} yıl.", "",
          f"**En iyi:** xi={best[0]}, ridge={best[1]}, ridge_total={best[2]}, hazırlık ağırlığı={best[3]}",
          f"**Mevcut ayar:** xi={base[0]}, ridge={base[1]}, ridge_total={base[2]}, hazırlık ağırlığı={base[3]}", "",
          f"Test döneminde en iyi − mevcut: 1X2 {_diff(lb.loc[test, 'model_1x2'], l0.loc[test, 'model_1x2'])}, "
          f"Ü/A {_diff(lb.loc[test, 'model_ou'], l0.loc[test, 'model_ou'])} (negatif = en iyi daha iyi)", "",
          "## Tüm ayarlar (ayar dönemine göre sıralı)", "", _md_table(t.round(4).head(40))]
    return "\n".join(md), t


def run(eval_from: int = config.EVAL_FROM_SEASON) -> tuple[str, pd.DataFrame]:
    """Mevcut ayarla doğrulama raporu."""
    df = load_results()
    if df.empty:
        raise SystemExit("Milli maç verisi yok: önce python main.py download")
    p = walk_forward(df, eval_from)
    title = (f"# Milli takım modeli — doğrulama\n\nWalk-forward {eval_from}+, {len(p)} maç (FIFA üyeleri). Ayar: "
             f"xi={config.MILLI_XI}, ridge={config.MILLI_RIDGE}, ridge_total={config.MILLI_RIDGE_TOTAL}, hazırlık "
             f"ağırlığı={config.MILLI_FRIENDLY_WEIGHT}, pencere {config.MILLI_LOOKBACK_DAYS // 365} yıl.")
    return report(df, p, eval_from, title), p


def ratings(ref_date=None, top: int = 50) -> pd.DataFrame:
    df = load_results()
    m = fit(df, ref_date or df["date"].max() + pd.Timedelta(days=1))
    r = m.ratings().drop(columns=["league"])
    return r[r["matches"] >= 5].head(top).reset_index(drop=True)


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)
