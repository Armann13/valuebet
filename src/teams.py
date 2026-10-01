"""Takım adı eşleştirme: İddaa / Understat / yaygın adlar -> football-data.co.uk adları."""
import difflib
import unicodedata

import pandas as pd

import config

# Yaygın isim -> football-data.co.uk ismi. Eksik olanları team_aliases.csv'ye ekleyebilirsin.
BUILTIN_ALIASES = {
    "atletico madrid": "Ath Madrid", "atl. madrid": "Ath Madrid", "atletico": "Ath Madrid",
    "athletic bilbao": "Ath Bilbao", "athletic club": "Ath Bilbao", "bilbao": "Ath Bilbao",
    "real betis": "Betis", "celta vigo": "Celta", "real sociedad": "Sociedad",
    "rayo vallecano": "Vallecano", "rayo": "Vallecano", "espanyol": "Espanol",
    "deportivo la coruna": "La Coruna", "deportivo": "La Coruna", "sporting gijon": "Sp Gijon",
    "real valladolid": "Valladolid", "deportivo alaves": "Alaves", "real oviedo": "Oviedo",
    "racing santander": "Santander", "real zaragoza": "Zaragoza", "rcd mallorca": "Mallorca",
    "real mallorca": "Mallorca", "ud las palmas": "Las Palmas", "granada cf": "Granada",
    "sd huesca": "Huesca", "cd tenerife": "Tenerife", "sd eibar": "Eibar", "malaga cf": "Malaga",
    "cd leganes": "Leganes", "girona fc": "Girona", "getafe cf": "Getafe", "sevilla fc": "Sevilla",
    "valencia cf": "Valencia", "villarreal cf": "Villarreal", "fc barcelona": "Barcelona",
    # Understat (büyük 5 lig)
    "manchester city": "Man City", "manchester united": "Man United", "newcastle united": "Newcastle",
    "queens park rangers": "QPR", "west bromwich albion": "West Brom", "wolverhampton wanderers": "Wolves",
    "arminia bielefeld": "Bielefeld", "borussia dortmund": "Dortmund", "borussia m.gladbach": "M'gladbach",
    "fc cologne": "FC Koln", "hamburger sv": "Hamburg", "hertha berlin": "Hertha",
    "rasenballsport leipzig": "RB Leipzig", "paris saint germain": "Paris SG", "parma calcio 1913": "Parma",
    "spal 2013": "Spal",
}


def _norm(s: str) -> str:
    s = str(s).replace("ı", "i").replace("İ", "I")  # Türkçe noktasız ı NFKD'de ayrışmaz, yoksa silinirdi
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


class TeamResolver:
    """aliases: ek eşleştirmeler (ör. Türkçe ülke adları). allow_fuzzy=False: sadece tam / takma ad / normalize
    eşleşme (ülke adlarında bulanık eşleşme tehlikeli: "Avusturya" ~ "Australia")."""

    def __init__(self, known: list[str], aliases: dict | None = None, allow_fuzzy: bool = True):
        self.known = sorted(set(known))
        self.by_norm = {_norm(t): t for t in self.known}
        self.allow_fuzzy = allow_fuzzy
        self.fuzzy = {}  # bulanık eşleşmeyle verilen kararlar (ad -> takım): gözden geçirmek için
        extra = {_norm(k): v for k, v in (aliases or {}).items()}
        self.aliases = {k: v for k, v in {**BUILTIN_ALIASES, **extra}.items() if v in self.known}
        f = config.ROOT / "team_aliases.csv"
        if f.exists():
            for r in pd.read_csv(f).itertuples(index=False):
                if str(r.team).strip() in self.known:  # başka listenin (kulüp / milli) adı karışmasın
                    self.aliases[_norm(r.alias)] = str(r.team).strip()

    def __call__(self, name: str) -> str | None:
        if name in self.known:
            return name
        n = _norm(name)
        if n in self.aliases:
            return self.aliases[n]
        if n in self.by_norm:
            return self.by_norm[n]
        if not self.allow_fuzzy:
            return None
        # "Sutton United" -> "Sutton": ad, bilinen TEK bir takımın adıyla (kelime sınırında) başlıyorsa
        pre = [k for k in self.by_norm if n.startswith(k + " ") or k.startswith(n + " ")]
        if len(pre) == 1:
            self.fuzzy[name] = self.by_norm[pre[0]]
            return self.by_norm[pre[0]]
        if len(pre) > 1:  # belirsiz (ör. "Paris" -> "Paris SG" / "Paris FC"): tahmin etme
            return None
        hit = difflib.get_close_matches(n, list(self.by_norm), n=1, cutoff=0.75)
        if hit:
            self.fuzzy[name] = self.by_norm[hit[0]]
        return self.by_norm[hit[0]] if hit else None
