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
}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return " ".join(s.lower().replace("-", " ").split())


class TeamResolver:
    def __init__(self, known: list[str]):
        self.known = sorted(set(known))
        self.by_norm = {_norm(t): t for t in self.known}
        self.fuzzy = {}  # bulanık eşleşmeyle verilen kararlar (ad -> takım): gözden geçirmek için
        self.aliases = {k: v for k, v in BUILTIN_ALIASES.items() if v in self.known}
        f = config.ROOT / "team_aliases.csv"
        if f.exists():
            for r in pd.read_csv(f).itertuples(index=False):
                self.aliases[_norm(r.alias)] = str(r.team).strip()

    def __call__(self, name: str) -> str | None:
        if name in self.known:
            return name
        n = _norm(name)
        if n in self.aliases:
            return self.aliases[n]
        if n in self.by_norm:
            return self.by_norm[n]
        hit = difflib.get_close_matches(n, list(self.by_norm), n=1, cutoff=0.75)
        if hit:
            self.fuzzy[name] = self.by_norm[hit[0]]
        return self.by_norm[hit[0]] if hit else None
