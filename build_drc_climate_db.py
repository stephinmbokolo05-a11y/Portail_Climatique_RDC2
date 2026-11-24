#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_drc_climate_db.py
=======================
Télécharge en une seule commande toutes les données nécessaires à l'Atlas
climatique de la RDC et les range dans UNE base SQLite (drc_climate_atlas.db).

SOURCES (toutes gratuites, sans clé d'API)
  nasa       NASA POWER daily point  : observations/réanalyse 1981→ (14 variables)
  era5       Open-Meteo Archive      : réanalyse ERA5 (contrôle croisé de NASA POWER)
  cmip6      Open-Meteo Climate API  : modèles CMIP6 HighResMIP, 1950→2050
  boundaries Limites des 26 provinces (GADM 4.1, repli : fichier de l'Atlas)
  atlas_ref  CSV CMIP6 « prêts » de l'Atlas original (idiouf, Hugging Face)

SAUVEGARDE : tout est écrit dans D:\\Portail_Climatique_RDC\\ATLAS (modifiable
avec --base-dir) : la base .db, le GeoJSON des provinces (drc_data\\) et les
exports CSV (drc_data\\csv\\). Le dossier est créé automatiquement.

USAGE
  pip install requests
  python build_drc_climate_db.py                      # tout, valeurs par défaut
  python build_drc_climate_db.py --sources nasa,boundaries
  python build_drc_climate_db.py --provinces Kinshasa,Ituri --start-year 1991
  python build_drc_climate_db.py --export-csv         # exporte aussi en CSV

Le script est REPRENABLE : chaque tronçon téléchargé est enregistré dans la
table download_log. Si une limite d'API est atteinte, relancez simplement la
même commande plus tard : seuls les tronçons manquants seront retéléchargés.

ATTRIBUTION : NASA POWER (NASA LaRC), ERA5 (Copernicus/ECMWF) et CMIP6 via
Open-Meteo (CC BY 4.0, usage non commercial), GADM 4.1 (usage académique).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import sys
import time
import unicodedata
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Module manquant : exécutez  pip install requests")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# (code, libellé, latitude, longitude, zone) — centroïdes repris de l'Atlas
PROVINCES = [
    ("Bas-Uele", "Bas-Uélé", 3.7662, 25.2509, "Nord"),
    ("Equateur", "Équateur", 0.1705, 19.1985, "Nord-Ouest"),
    ("Haut-Katanga", "Haut-Katanga", -10.6904, 27.7960, "Sud"),
    ("Haut-Lomami", "Haut-Lomami", -8.2178, 25.3761, "Sud"),
    ("Haut-Uele", "Haut-Uélé", 3.1843, 28.3522, "Nord"),
    ("Ituri", "Ituri", 2.1429, 30.2567, "Est"),
    ("Kasai", "Kasaï", -5.1948, 21.0937, "Centre-Sud"),
    ("Kasai-Central", "Kasaï-Central", -6.1670, 22.5110, "Centre-Sud"),
    ("Kasai-Oriental", "Kasaï-Oriental", -6.1821, 23.5896, "Centre-Sud"),
    ("Kinshasa", "Kinshasa", -4.4419, 15.2663, "Ouest"),
    ("Kongo-Central", "Kongo-Central", -5.0114, 14.6629, "Ouest"),
    ("Kwango", "Kwango", -5.9646, 17.6446, "Ouest"),
    ("Kwilu", "Kwilu", -4.8554, 18.3252, "Ouest"),
    ("Lomami", "Lomami", -6.5336, 24.2436, "Centre-Sud"),
    ("Lualaba", "Lualaba", -9.9397, 24.4193, "Sud"),
    ("Mai-Ndombe", "Maï-Ndombe", -3.0662, 18.3635, "Ouest"),
    ("Maniema", "Maniema", -3.1219, 26.9479, "Est"),
    ("Mongala", "Mongala", 2.2611, 21.4179, "Nord-Ouest"),
    ("Nord-Kivu", "Nord-Kivu", -0.6797, 29.2337, "Est"),
    ("Nord-Ubangi", "Nord-Ubangi", 3.9045, 21.0359, "Nord-Ouest"),
    ("Sankuru", "Sankuru", -3.8448, 23.8135, "Centre"),
    ("Sud-Kivu", "Sud-Kivu", -3.0381, 28.1650, "Est"),
    ("Sud-Ubangi", "Sud-Ubangi", 3.2376, 19.5657, "Nord-Ouest"),
    ("Tanganyika", "Tanganyika", -6.1000, 27.2000, "Est"),
    ("Tshopo", "Tshopo", 0.7978, 25.3788, "Centre"),
    ("Tshuapa", "Tshuapa", -0.4831, 21.5719, "Centre"),
]

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_COLS = {  # paramètre NASA POWER -> colonne SQL
    "PRECTOTCORR": "precip_mm", "T2M": "t2m_c", "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c", "T2MDEW": "dewpoint_c", "RH2M": "rh_pct",
    "QV2M": "qv2m_gkg", "WS2M": "ws2m_ms", "WS10M": "ws10m_ms",
    "PS": "pressure_kpa", "ALLSKY_SFC_SW_DWN": "srad",
    "EVPTRNS": "evptrns_mm", "GWETROOT": "gwet_root", "GWETTOP": "gwet_top",
}
NASA_CORE = ["PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M"]

ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
ERA5_COLS = {
    "temperature_2m_mean": "temp_mean_c", "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c", "precipitation_sum": "precip_mm",
    "wind_speed_10m_max": "wind_max_ms", "shortwave_radiation_sum": "srad_mj_m2",
    "et0_fao_evapotranspiration": "et0_mm",
}

CMIP_URL = "https://climate-api.open-meteo.com/v1/climate"
CMIP_COLS = {
    "temperature_2m_mean": "temp_mean_c", "temperature_2m_max": "tmax_c",
    "temperature_2m_min": "tmin_c", "precipitation_sum": "precip_mm",
    "rain_sum": "rain_mm", "relative_humidity_2m_mean": "rh_pct",
    "dew_point_2m_mean": "dewpoint_c", "wind_speed_10m_mean": "wind_ms",
    "shortwave_radiation_sum": "srad_mj_m2", "pressure_msl_mean": "pressure_hpa",
    "cloud_cover_mean": "cloud_pct",
    "soil_moisture_0_to_10cm_mean": "soil_moisture",
}
OM_CORE = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
           "precipitation_sum"]
DEFAULT_CMIP_MODELS = ["EC_Earth3P_HR", "MPI_ESM1_2_XR", "CMCC_CM2_VHR4",
                       "FGOALS_f3_H", "MRI_AGCM3_2_S", "HiRAM_SIT_HR", "NICAM16_8S"]

GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_COD_1.json.zip"
ATLAS_BASE = ("https://huggingface.co/spaces/idiouf/"
              "DRC_Climate_Risk_and_Future_Projections_Atlas/resolve/main/")
ATLAS_GEOJSON = ATLAS_BASE + "DRC_Admin_1_Provinces.geojson"
ATLAS_CSVS = {
    "atlas_cmip6_monthly": ATLAS_BASE + "drc_cmip6_monthly_ready.csv",
    "atlas_cmip6_indicators": ATLAS_BASE + "drc_cmip6_indicators_ready.csv",
}

# Dossier racine où TOUS les fichiers sauvegardés seront placés (base, GeoJSON, CSV)
BASE_DIR = Path(r"D:\Portail_Climatique_RDC\ATLAS")

UA = {"User-Agent": "drc-climate-atlas-builder/1.0 (research)"}


class DailyLimit(Exception):
    """Quota journalier d'une API atteint : reprendre plus tard."""


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Base de données
# ---------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS provinces(
    code TEXT PRIMARY KEY, label TEXT, lat REAL, lon REAL, zone TEXT,
    geometry_geojson TEXT);
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS download_log(
    source TEXT, key TEXT, status TEXT, n_rows INTEGER, detail TEXT, ts TEXT,
    PRIMARY KEY(source, key));
CREATE TABLE IF NOT EXISTS obs_nasa_daily(
    province TEXT, date TEXT, year INTEGER, month INTEGER,
    precip_mm REAL, t2m_c REAL, tmax_c REAL, tmin_c REAL, dewpoint_c REAL,
    rh_pct REAL, qv2m_gkg REAL, ws2m_ms REAL, ws10m_ms REAL, pressure_kpa REAL,
    srad REAL, evptrns_mm REAL, gwet_root REAL, gwet_top REAL,
    PRIMARY KEY(province, date));
CREATE TABLE IF NOT EXISTS obs_era5_daily(
    province TEXT, date TEXT, year INTEGER, month INTEGER,
    temp_mean_c REAL, tmax_c REAL, tmin_c REAL, precip_mm REAL,
    wind_max_ms REAL, srad_mj_m2 REAL, et0_mm REAL,
    PRIMARY KEY(province, date));
CREATE TABLE IF NOT EXISTS cmip6_daily(
    province TEXT, model TEXT, date TEXT, year INTEGER, month INTEGER,
    temp_mean_c REAL, tmax_c REAL, tmin_c REAL, precip_mm REAL, rain_mm REAL,
    rh_pct REAL, dewpoint_c REAL, wind_ms REAL, srad_mj_m2 REAL,
    pressure_hpa REAL, cloud_pct REAL, soil_moisture REAL,
    PRIMARY KEY(province, model, date));

DROP VIEW IF EXISTS v_nasa_monthly;
CREATE VIEW v_nasa_monthly AS
SELECT province, year, month, COUNT(precip_mm) AS n_days,
       SUM(precip_mm) AS precip_mm, AVG(t2m_c) AS t2m_c, AVG(tmax_c) AS tmax_c,
       AVG(tmin_c) AS tmin_c, AVG(rh_pct) AS rh_pct, AVG(ws2m_ms) AS ws2m_ms
FROM obs_nasa_daily GROUP BY province, year, month;

DROP VIEW IF EXISTS v_nasa_annual;
CREATE VIEW v_nasa_annual AS
SELECT province, year, COUNT(precip_mm) AS n_days, SUM(precip_mm) AS prcptot_mm,
       AVG(t2m_c) AS tmean_c, AVG(tmax_c) AS tmax_mean_c, AVG(tmin_c) AS tmin_mean_c,
       MAX(tmax_c) AS txx_c, MIN(tmin_c) AS tnn_c,
       SUM(precip_mm >= 20) AS r20mm_days, SUM(precip_mm >= 50) AS r50mm_days,
       SUM(precip_mm < 1) AS dry_days, SUM(tmax_c >= 35) AS hot_days_35,
       SUM(tmin_c >= 25) AS warm_nights_25, MAX(precip_mm) AS rx1day_mm,
       AVG(rh_pct) AS rh_mean_pct
FROM obs_nasa_daily GROUP BY province, year HAVING n_days >= 330;

DROP VIEW IF EXISTS v_nasa_normals_1991_2020;
CREATE VIEW v_nasa_normals_1991_2020 AS
SELECT province, month, AVG(precip_mm) AS precip_mm, AVG(t2m_c) AS t2m_c,
       AVG(tmax_c) AS tmax_c, AVG(tmin_c) AS tmin_c, AVG(rh_pct) AS rh_pct,
       COUNT(*) AS n_years
FROM v_nasa_monthly WHERE year BETWEEN 1991 AND 2020 AND n_days >= 25
GROUP BY province, month;

DROP VIEW IF EXISTS v_cmip6_annual;
CREATE VIEW v_cmip6_annual AS
SELECT province, model, year, COUNT(precip_mm) AS n_days,
       SUM(precip_mm) AS prcptot_mm, AVG(temp_mean_c) AS tmean_c,
       AVG(tmax_c) AS tmax_mean_c, SUM(precip_mm >= 20) AS r20mm_days,
       MAX(precip_mm) AS rx1day_mm
FROM cmip6_daily GROUP BY province, model, year;
"""


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT OR IGNORE INTO provinces(code,label,lat,lon,zone) VALUES(?,?,?,?,?)",
        PROVINCES)
    conn.commit()
    return conn


def logged_ok(conn, source: str, key: str) -> bool:
    r = conn.execute("SELECT status FROM download_log WHERE source=? AND key=?",
                     (source, key)).fetchone()
    return bool(r and r[0] == "ok")


def mark(conn, source, key, status, n=0, detail=""):
    conn.execute("INSERT OR REPLACE INTO download_log VALUES(?,?,?,?,?,?)",
                 (source, key, status, n, detail[:500],
                  datetime.now(timezone.utc).isoformat(timespec="seconds")))
    conn.commit()


def insert_rows(conn, table: str, cols: list[str], rows: list[tuple]) -> None:
    sql = (f"INSERT OR REPLACE INTO {table}({','.join(cols)}) "
           f"VALUES({','.join('?' * len(cols))})")
    conn.executemany(sql, rows)
    conn.commit()


# ---------------------------------------------------------------------------
# HTTP robuste
# ---------------------------------------------------------------------------
def http_get(url: str, params: dict | None = None, tries: int = 5,
             timeout: int = 180, binary: bool = False):
    """GET avec nouvelles tentatives. Lève DailyLimit si quota journalier atteint."""
    last = ""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.content if binary else r.json()
            body = r.text[:300]
            last = f"HTTP {r.status_code}: {body}"
            if r.status_code == 429:
                if "daily" in body.lower():
                    raise DailyLimit(body)
                wait = 65 * attempt
                log(f"  limite de débit (429) → pause {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in (400, 404, 422):
                raise ValueError(last)          # requête invalide : inutile de répéter
            time.sleep(5 * attempt)
        except (requests.ConnectionError, requests.Timeout) as e:
            last = f"réseau : {e}"
            time.sleep(8 * attempt)
    raise RuntimeError(f"Échec après {tries} essais — {last}")


def year_chunks(start: int, end: int, size: int):
    y = start
    while y <= end:
        yield y, min(y + size - 1, end)
        y += size


def clean(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if v <= -990 or v != v else v


# ---------------------------------------------------------------------------
# 1. NASA POWER
# ---------------------------------------------------------------------------
def fetch_nasa(conn, provs, start, end, sleep, force):
    cols = ["province", "date", "year", "month"] + list(NASA_COLS.values())
    for code, _, lat, lon, _ in provs:
        for y0, y1 in year_chunks(start, end, 15):
            key = f"{code}:{y0}-{y1}"
            if not force and logged_ok(conn, "nasa", key):
                continue
            log(f"NASA POWER  {code}  {y0}-{y1}")
            base = {"community": "AG", "longitude": lon, "latitude": lat,
                    "start": f"{y0}0101", "end": f"{y1}1231", "format": "JSON"}
            try:
                try:
                    data = http_get(NASA_URL, {**base, "parameters": ",".join(NASA_COLS)})
                except ValueError:  # un paramètre indisponible → jeu minimal
                    log("  jeu étendu refusé, repli sur les 6 variables de base")
                    data = http_get(NASA_URL, {**base, "parameters": ",".join(NASA_CORE)})
                par = data["properties"]["parameter"]
                dates = sorted(next(iter(par.values())).keys())
                rows = []
                for d in dates:
                    iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
                    rows.append((code, iso, int(d[:4]), int(d[4:6])) + tuple(
                        clean(par.get(p, {}).get(d)) for p in NASA_COLS))
                insert_rows(conn, "obs_nasa_daily", cols, rows)
                mark(conn, "nasa", key, "ok", len(rows))
            except Exception as e:
                log(f"  ✗ {e}")
                mark(conn, "nasa", key, "error", 0, str(e))
            time.sleep(sleep)


# ---------------------------------------------------------------------------
# 2 & 3. Open-Meteo (ERA5 et CMIP6)
# ---------------------------------------------------------------------------
def pick_series(daily: dict, var: str):
    """Retrouve la série d'une variable, avec ou sans suffixe de modèle."""
    if var in daily:
        return daily[var]
    for k, v in daily.items():
        if k.startswith(var + "_"):
            return v
    return None


def fetch_open_meteo(conn, source, url, colmap, provs, start, end, models,
                     table, sleep, force):
    variants = [list(colmap), OM_CORE]  # jeu complet, puis repli minimal
    for code, _, lat, lon, _ in provs:
        for model in models:
            for y0, y1 in year_chunks(start, end, 10):
                y1c = min(y1, 2050) if source == "cmip6" else y1
                key = f"{code}:{model or 'era5'}:{y0}-{y1c}"
                if not force and logged_ok(conn, source, key):
                    continue
                log(f"{source.upper():5s} {code}  {model or ''}  {y0}-{y1c}")
                base = {"latitude": lat, "longitude": lon,
                        "start_date": f"{y0}-01-01", "end_date": f"{y1c}-12-31",
                        "wind_speed_unit": "ms", "timezone": "GMT"}
                if model:
                    base["models"] = model
                try:
                    data = None
                    for vs in variants:
                        try:
                            data = http_get(url, {**base, "daily": ",".join(vs)})
                            break
                        except ValueError as e:
                            log(f"  variables refusées ({str(e)[:80]}), repli")
                    if data is None:
                        raise RuntimeError("aucune combinaison de variables acceptée")
                    daily = data["daily"]
                    times = daily["time"]
                    series = {v: pick_series(daily, v) for v in colmap}
                    rows = []
                    for i, d in enumerate(times):
                        vals = tuple(clean(series[v][i]) if series[v] else None
                                     for v in colmap)
                        head = (code, model, d) if model else (code, d)
                        rows.append(head + (int(d[:4]), int(d[5:7])) + vals)
                    cols = (["province", "model", "date"] if model else
                            ["province", "date"]) + ["year", "month"] + list(colmap.values())
                    insert_rows(conn, table, cols, rows)
                    mark(conn, source, key, "ok", len(rows))
                except DailyLimit as e:
                    log(f"✗ Quota journalier atteint ({e}). Relancez demain : reprise automatique.")
                    mark(conn, source, key, "quota", 0, str(e))
                    return
                except Exception as e:
                    log(f"  ✗ {e}")
                    mark(conn, source, key, "error", 0, str(e))
                time.sleep(sleep)


# ---------------------------------------------------------------------------
# 4. Limites administratives
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "".join(c for c in s.lower() if c.isalnum())


def fetch_boundaries(conn, outdir: Path):
    gj = None
    try:
        log("Limites provinciales : GADM 4.1")
        raw = http_get(GADM_URL, binary=True, timeout=300)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = next(n for n in z.namelist() if n.endswith(".json"))
            gj = json.loads(z.read(name))
    except Exception as e:
        log(f"  GADM indisponible ({e}) → repli sur le fichier de l'Atlas")
        try:
            gj = http_get(ATLAS_GEOJSON, timeout=300)
        except Exception as e2:
            log(f"  ✗ limites non téléchargées : {e2}")
            mark(conn, "boundaries", "adm1", "error", 0, str(e2))
            return
    (outdir / "DRC_Admin_1_Provinces.geojson").write_text(
        json.dumps(gj), encoding="utf-8")
    index = {norm(p[0]): p[0] for p in PROVINCES}
    index.update({norm(p[1]): p[0] for p in PROVINCES})
    matched = set()
    for feat in gj.get("features", []):
        props = feat.get("properties", {})
        label = next((props[k] for k in ("NAME_1", "name", "NAME", "province",
                                         "shapeName", "ADM1_FR") if k in props), "")
        code = index.get(norm(label))
        if code:
            conn.execute("UPDATE provinces SET geometry_geojson=? WHERE code=?",
                         (json.dumps(feat.get("geometry")), code))
            matched.add(code)
    conn.commit()
    missing = [p[0] for p in PROVINCES if p[0] not in matched]
    log(f"  {len(matched)}/26 provinces appariées" +
        (f" — manquantes : {', '.join(missing)}" if missing else ""))
    mark(conn, "boundaries", "adm1", "ok", len(matched), ",".join(missing))


# ---------------------------------------------------------------------------
# 5. CSV de référence de l'Atlas original
# ---------------------------------------------------------------------------
def fetch_atlas_ref(conn):
    for table, url in ATLAS_CSVS.items():
        if logged_ok(conn, "atlas_ref", table):
            continue
        try:
            log(f"Atlas (référence) : {table}")
            raw = http_get(url, binary=True, timeout=300).decode("utf-8-sig")
            rd = csv.reader(io.StringIO(raw))
            header = [h.strip().replace('"', "") for h in next(rd)]
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(f'CREATE TABLE "{table}"({",".join(chr(34)+h+chr(34) for h in header)})')

            def conv(x):
                try:
                    return float(x)
                except ValueError:
                    return x
            rows = [tuple(conv(x) for x in r) for r in rd if len(r) == len(header)]
            conn.executemany(
                f'INSERT INTO "{table}" VALUES({",".join("?" * len(header))})', rows)
            conn.commit()
            mark(conn, "atlas_ref", table, "ok", len(rows))
        except Exception as e:
            log(f"  ✗ {e}")
            mark(conn, "atlas_ref", table, "error", 0, str(e))


# ---------------------------------------------------------------------------
# Rapport de contrôle qualité
# ---------------------------------------------------------------------------
def report(conn):
    print("\n" + "=" * 72 + "\nRAPPORT DE LA BASE\n" + "=" * 72)
    for t in ("provinces", "obs_nasa_daily", "obs_era5_daily", "cmip6_daily",
              "atlas_cmip6_monthly", "atlas_cmip6_indicators"):
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  {t:26s} {n:>12,d} lignes".replace(",", " "))
        except sqlite3.OperationalError:
            print(f"  {t:26s} (absente)")
    print("\nJournal de téléchargement :")
    for s, st, n in conn.execute(
            "SELECT source,status,COUNT(*) FROM download_log GROUP BY 1,2 ORDER BY 1,2"):
        print(f"  {s:12s} {st:8s} {n}")
    print("\nContrôle de plausibilité NASA (pluie annuelle moyenne, bornes 500–2600 mm) :")
    q = """SELECT province, ROUND(AVG(prcptot_mm)), COUNT(*) FROM v_nasa_annual
           GROUP BY province ORDER BY 2"""
    for p, mm, ny in conn.execute(q):
        flag = "  ⚠ hors bornes" if mm and not 500 <= mm <= 2600 else ""
        print(f"  {p:16s} {mm:>6.0f} mm/an  ({ny} années complètes){flag}")
    errs = conn.execute(
        "SELECT source,key FROM download_log WHERE status!='ok'").fetchall()
    if errs:
        print(f"\n{len(errs)} tronçon(s) à reprendre : relancez la même commande.")


def export_csv(conn, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    for t in ("provinces", "obs_nasa_daily", "obs_era5_daily", "cmip6_daily",
              "v_nasa_annual", "v_nasa_monthly", "v_nasa_normals_1991_2020",
              "v_cmip6_annual"):
        cur = conn.execute(f"SELECT * FROM {t}" if t != "provinces" else
                           "SELECT code,label,lat,lon,zone FROM provinces")
        with open(outdir / f"{t}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur)
        log(f"export → {outdir / (t + '.csv')}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(BASE_DIR),
                    help="dossier racine de sauvegarde (défaut : %(default)s)")
    ap.add_argument("--db", default="drc_climate_atlas.db",
                    help="nom ou chemin de la base (relatif → dans --base-dir)")
    ap.add_argument("--outdir", default="drc_data",
                    help="dossier des fichiers annexes (relatif → dans --base-dir)")
    ap.add_argument("--sources", default="boundaries,atlas_ref,nasa,era5,cmip6")
    ap.add_argument("--provinces", default="", help="codes séparés par des virgules")
    ap.add_argument("--start-year", type=int, default=1981)
    ap.add_argument("--end-year", type=int, default=date.today().year - 1,
                    help="dernière année complète (défaut : année précédente)")
    ap.add_argument("--cmip-start", type=int, default=1985)
    ap.add_argument("--cmip-end", type=int, default=2050)
    ap.add_argument("--cmip-models", default=",".join(DEFAULT_CMIP_MODELS[:3]),
                    help="liste Open-Meteo ; ex. all pour les 7 modèles")
    ap.add_argument("--sleep", type=float, default=1.5, help="pause entre requêtes (s)")
    ap.add_argument("--force", action="store_true", help="retélécharger même si déjà fait")
    ap.add_argument("--export-csv", action="store_true")
    # parse_known_args : ignore les arguments injectés par Jupyter/Spyder (-f kernel.json)
    a, _inconnus = ap.parse_known_args()

    base = Path(a.base_dir)
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        base = Path.cwd() / "ATLAS"
        base.mkdir(parents=True, exist_ok=True)
        log(f"⚠ Dossier {a.base_dir} inaccessible ({e}) → repli sur {base}")
    db_path = Path(a.db) if Path(a.db).is_absolute() else base / a.db
    outdir = Path(a.outdir) if Path(a.outdir).is_absolute() else base / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"Dossier de sauvegarde : {base}")
    conn = open_db(db_path)
    sources = {s.strip() for s in a.sources.split(",") if s.strip()}
    wanted = {p.strip() for p in a.provinces.split(",") if p.strip()}
    provs = [p for p in PROVINCES if not wanted or p[0] in wanted]
    unknown = wanted - {p[0] for p in PROVINCES}
    if unknown:
        sys.exit(f"Provinces inconnues : {', '.join(sorted(unknown))}")
    models = (DEFAULT_CMIP_MODELS if a.cmip_models == "all"
              else [m.strip() for m in a.cmip_models.split(",") if m.strip()])

    conn.executemany("INSERT OR REPLACE INTO metadata VALUES(?,?)", [
        ("built_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("obs_period", f"{a.start_year}-{a.end_year}"),
        ("cmip6_period", f"{a.cmip_start}-{a.cmip_end}"),
        ("cmip6_models", ",".join(models)),
        ("nasa_source", NASA_URL), ("era5_source", ERA5_URL), ("cmip6_source", CMIP_URL),
        ("note", "CMIP6 HighResMIP via Open-Meteo : différent du CNRM-CM6-1-HR/SSP5-8.5 "
                 "de l'Atlas original ; la table atlas_cmip6_* conserve ce dernier."),
    ])
    conn.commit()

    t0 = time.time()
    try:
        if "boundaries" in sources:
            fetch_boundaries(conn, outdir)
        if "atlas_ref" in sources:
            fetch_atlas_ref(conn)
        if "nasa" in sources:
            fetch_nasa(conn, provs, a.start_year, a.end_year, a.sleep, a.force)
        if "era5" in sources:
            fetch_open_meteo(conn, "era5", ERA5_URL, ERA5_COLS, provs, a.start_year,
                             a.end_year, [None], "obs_era5_daily", a.sleep, a.force)
        if "cmip6" in sources:
            fetch_open_meteo(conn, "cmip6", CMIP_URL, CMIP_COLS, provs, a.cmip_start,
                             a.cmip_end, models, "cmip6_daily", a.sleep, a.force)
    except KeyboardInterrupt:
        log("Interrompu : la progression est sauvegardée, relancez pour reprendre.")

    if a.export_csv:
        export_csv(conn, outdir / "csv")
    report(conn)
    log(f"Terminé en {(time.time() - t0) / 60:.1f} min → {db_path.resolve()}")
    conn.close()


if __name__ == "__main__":
    main()
