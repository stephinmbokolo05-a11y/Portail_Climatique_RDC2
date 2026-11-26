#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_hf_space.py
=================
ÉTAPE 4 : produit, EN UNE COMMANDE, un Hugging Face Space (application Streamlit) de même
type que l'Atlas de référence, à partir de votre base drc_climate_atlas.db :

    D:\\Portail_Climatique_RDC\\ATLAS\\space_hf\\
        app.py  Dockerfile  README.md  requirements.txt
        data\\*.csv.gz  provinces.geojson  meta.json      (quelques Mo au total)

L'application contient : portrait national · carte (≈40 couches) · fiche province (climat,
séries + tendances, sources de pluie, SPI, exposition, fiche à télécharger) · comparaison de
provinces · tendances et fiabilité (NASA/ERA5/GPCC/TAMSAT) · projections 2026-2040 ·
problèmes et risques (chaleur, pluies fortes, saisons sèches, variabilité, paludisme) ·
téléchargement des données · méthodes et limites.

USAGE
  python build_hf_space.py                        # fabrique le dossier space_hf
  pip install streamlit plotly                    # (pour tester en local)
  cd space_hf && streamlit run app.py             # test local : http://localhost:8501

  Publication sur Hugging Face (compte gratuit + jeton « write » : huggingface.co/settings/tokens) :
  pip install huggingface_hub
  set HF_TOKEN=hf_xxx                             # (PowerShell : $env:HF_TOKEN="hf_xxx")
  python build_hf_space.py --upload --repo-id VOTRE_NOM/Atlas_Climatique_RDC
"""
import argparse
import json
import os
import re
import sqlite3
import sys

# Garde-fou : si ce fichier est envoye sur Streamlit Cloud par erreur sous le nom app.py, afficher un message clair
if __name__ == "__main__" and "streamlit" in sys.modules:
    import streamlit as st
    st.error("**Ce fichier est le GÉNÉRATEUR `build_hf_space.py`, pas l'application.**  \n"
             "Sur GitHub, `app.py` doit être le fichier **`pour_github\\app.py`** créé par ce générateur "
             "(environ 1 Mo, il commence par `# -*- coding: utf-8 -*-` et contient `def page_portrait`). "
             "Lancez `python build_hf_space.py` sur votre PC, puis envoyez `pour_github\\app.py` et "
             "`pour_github\\requirements.txt` dans le dépôt.")
    st.stop()
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Portail_Climatique_RDC\ATLAS")
SOURCES = ["nasa", "era5", "gpcc", "tamsat", "chirps"]

def norm(s):
    s = unicodedata.normalize("NFD", str(s or ""))
    return "".join(c for c in s if unicodedata.category(c) != "Mn" and c.isalnum()).lower()


def tbl(conn, name):
    try:
        return pd.read_sql(f"SELECT * FROM {name}", conn)
    except Exception:
        return pd.DataFrame()


def num(x, d=3):
    try:
        x = float(x)
        return round(x, d) if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def rdp(pts, tol):
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 4:
        return pts
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        seg = pts[i + 1:j]
        if len(seg) == 0:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        d = (np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L) if L > 0 \
            else np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def geometry(g, tol=0.012):
    if not isinstance(g, str) or not g:
        return []
    gj = json.loads(g)
    polys = gj["coordinates"] if gj["type"] == "MultiPolygon" else [gj["coordinates"]]
    out = []
    for poly in polys:
        r = np.asarray(poly[0], float)
        if (r[:, 0].max() - r[:, 0].min()) < 0.06 and (r[:, 1].max() - r[:, 1].min()) < 0.06:
            continue                                   # micro-îlots
        s = rdp(r, tol)
        if len(s) >= 4:
            out.append([[round(x, 3), round(y, 3)] for x, y in s])
    return out


def anchor(label):
    s = str(label).lower()
    yrs = [int(y) for y in re.findall(r"(?:19|20|21)\d{2}", s)]
    if len(yrs) >= 2:
        return (yrs[0] + yrs[1]) / 2 + 0.5
    if "hist" in s or "base" in s or "ref" in s:
        return 1999.5
    if "near" in s or "proche" in s or "mid" in s:
        return 2032.5
    if "far" in s or "long" in s or "lointain" in s:
        return 2075.5
    return None


def projections(conn, provs, normals, rain_base):
    ci = tbl(conn, "atlas_cmip6_indicators")
    if ci.empty:
        return None, "table atlas_cmip6_indicators absente"
    cols = list(ci.columns)
    pc = next((c for c in cols if re.search(r"prov|admin|name|region", c, re.I)), None)
    strc = [c for c in cols if c != pc and not pd.api.types.is_numeric_dtype(ci[c]) and 2 <= ci[c].nunique() <= 6]
    if pc is None or not strc:
        return None, f"colonnes province/période non reconnues : {cols}"
    per = strc[0]
    anchors = {v: anchor(v) for v in ci[per].unique()}
    if any(a is None for a in anchors.values()):
        return None, f"périodes non reconnues : {list(anchors)}"
    numc = [c for c in cols if c not in (pc, per) and pd.api.types.is_numeric_dtype(ci[c])
            and not re.search(r"^(lat|lon|id|year)", c, re.I)]
    idx = {norm(r.label): r.code for r in provs.itertuples()}
    idx.update({norm(r.code): r.code for r in provs.itertuples()})
    years = list(range(2026, 2041))
    order = sorted(anchors, key=anchors.get)
    xs = [anchors[o] for o in order]
    out, base, meta = {}, {}, []
    for c in numc:
        low = c.lower()
        kind = ("temp" if re.search(r"temp|tas|t2m", low) and not re.search(r"dry|wet|freq", low) else
                "rain" if re.search(r"precip|rain|prcp|^pr", low)
                and not re.search(r"intens|heavy|freq|dry|wet|cv|var|extreme|day", low) else "raw")
        meta.append(dict(id=c, kind=kind))
    for r_code, g in ci.groupby(ci[pc].map(lambda v: idx.get(norm(v)))):
        if r_code is None:
            continue
        g = g.set_index(per)
        out[r_code], base[r_code] = {}, {}
        for m in meta:
            ys = [num(g.loc[o, m["id"]], 6) if o in g.index else None for o in order]
            if any(v is None for v in ys):
                continue
            vals = np.interp(years, xs, ys)
            hist = ys[0]
            if m["kind"] == "temp" and (r_code, "TMEAN") in normals:
                b = normals[(r_code, "TMEAN")]
                vals = b + (vals - hist)
                base[r_code][m["id"]] = b
            elif m["kind"] == "rain" and r_code in rain_base and hist:
                b = rain_base[r_code]
                vals = b * vals / hist
                base[r_code][m["id"]] = b
            else:
                base[r_code][m["id"]] = hist
            out[r_code][m["id"]] = [round(float(v), 3) for v in vals]
    if not out:
        return None, "aucune province appariée dans atlas_cmip6_indicators"
    lab = {"temp": "Température moyenne (°C), corrigée du biais",
           "rain": "Pluie annuelle (mm), corrigée du biais"}
    for m in meta:
        m["label"] = lab.get(m["kind"], m["id"].replace("_", " "))
        m["unit"] = {"temp": "°C", "rain": "mm"}.get(m["kind"], "valeur du jeu")
        m["dec"] = {"temp": 2, "rain": 0}.get(m["kind"], 2)
    meta = [m for m in meta if any(m["id"] in v for v in out.values())]
    return dict(years=years, cols=meta, data=out, base=base,
                periods={str(k): v for k, v in anchors.items()}), "ok"


README = """---
title: Atlas climatique RDC
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Climat, tendances et projections des provinces de la RDC
---

# Atlas climatique de la RDC

Normales 1991-2020 · tendances 1981-2024 · fiabilité (NASA POWER, ERA5, GPCC, TAMSAT) · projections 2026-2040 (CMIP6).
Application Streamlit : portrait national, carte, fiches provinces, comparaison, risques, données téléchargeables.

**Limites** : la tendance de la pluie n'est pas établie (sources en désaccord) ; projections issues d'un seul modèle et d'un scénario très émetteur (SSP5-8.5).
"""

DOCKERFILE = """FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STREAMLIT_SERVER_HEADLESS=true
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false", "--theme.base=light", "--theme.primaryColor=#1f7a4d", "--theme.backgroundColor=#f7f6f1", "--theme.secondaryBackgroundColor=#ecebe3", "--theme.textColor=#1d2a24"]
"""

STREAMLIT_CONFIG = """[theme]
base = "light"
primaryColor = "#1f7a4d"
backgroundColor = "#f7f6f1"
secondaryBackgroundColor = "#ecebe3"
textColor = "#1d2a24"

[server]
headless = true
enableXsrfProtection = false
enableCORS = false

[browser]
gatherUsageStats = false
"""

REQUIREMENTS = "streamlit>=1.37,<1.50\npandas>=2.0\nnumpy>=1.26\nplotly>=5.20\n"

APP_SOURCE = r'''# -*- coding: utf-8 -*-
"""Atlas climatique de la RDC : application Streamlit (Hugging Face Space)."""
import streamlit as st

st.set_page_config(page_title="Atlas climatique RDC", page_icon="🌍", layout="wide")
_boot = st.empty()
_boot.info("⏳ Chargement de l'atlas climatique de la RDC… (la première ouverture peut prendre une minute)")
try:
    import streamlit.components.v1 as _components
    _components.html("<script>try{var d=window.parent.document.documentElement;d.setAttribute('lang','fr');"
                     "d.setAttribute('translate','no');d.classList.add('notranslate');}catch(e){}</script>", height=0)
except Exception:
    pass

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent


def embedded_bytes():
    """Donnees integrees a la fin de ce fichier (mode autonome : un seul fichier app.py suffit)."""
    import base64
    begin, end = "# ===ATLAS" + "_DATA_BEGIN===", "# ===ATLAS" + "_DATA_END==="
    try:
        src = Path(__file__).read_text(encoding="utf-8")
    except Exception:
        return None
    if begin not in src:
        return None
    body = src.split(begin, 1)[1].split(end, 1)[0]
    b64 = "".join(l[1:].strip() for l in body.splitlines() if l.startswith("#"))
    return base64.b64decode(b64) if b64 else None


def find_data():
    """Retrouve les donnees : integrees dans app.py, dossier data/, space_hf/data/, racine, ou archive .zip."""
    import hashlib
    import tempfile
    import zipfile
    blob = embedded_bytes()
    if blob:
        d = Path(tempfile.gettempdir()) / ("atlas_emb_" + hashlib.md5(blob[:4000]).hexdigest()[:12])
        if not (d.exists() and any(d.rglob("meta.json"))):
            d.mkdir(parents=True, exist_ok=True)
            import io
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                zf.extractall(d)
        for p in sorted(d.rglob("meta.json")):
            return p.parent
    for c in (HERE / "data", HERE / "space_hf" / "data", HERE):
        if (c / "meta.json").exists():
            return c
    for p in sorted(HERE.rglob("meta.json")):
        if (p.parent / "provinces.geojson").exists():
            return p.parent
    for zn in ("atlas_data.zip", "data.zip"):
        z = HERE / zn
        if z.exists():
            d = Path(tempfile.gettempdir()) / ("atlas_" + zn.replace(".zip", ""))
            if not (d.exists() and any(d.rglob("meta.json"))):
                d.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
            for p in sorted(d.rglob("meta.json")):
                return p.parent
    return HERE / "data"


DATA = find_data()

# ------------------------------------------------------------------ apparence
pio.templates["atlas"] = go.layout.Template(layout=dict(
    font=dict(family="Segoe UI, system-ui, Arial, sans-serif", size=13, color="#1d2a24"),
    colorway=["#1f7a4d", "#c0482b", "#2a6fb0", "#b8860b", "#7a4bb0", "#2b9bb5", "#8a6d3b"],
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor="rgba(120,120,110,.18)", zeroline=False, linecolor="rgba(120,120,110,.4)"),
    yaxis=dict(gridcolor="rgba(120,120,110,.18)", zeroline=False, linecolor="rgba(120,120,110,.4)"),
    hoverlabel=dict(font_size=13), legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "plotly+atlas"

CSS = """
<style>
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif !important; }
.block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1320px; }
h1, h2, h3 { letter-spacing: -0.02em; }
.hero { position: relative; overflow: hidden; background: linear-gradient(115deg, #0b4a32 0%, #1f7a4d 42%, #2a6fb0 100%);
        color: #fff; padding: 30px 34px; border-radius: 20px; margin: 4px 0 18px; box-shadow: 0 10px 30px rgba(15,80,55,.25); }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
              background: radial-gradient(circle, rgba(255,255,255,.22), rgba(255,255,255,0) 70%); }
.hero h1 { color: #fff; font-size: 2.05rem; font-weight: 800; margin: 0 0 6px; padding: 0; }
.hero p { margin: 0; opacity: .92; font-size: 1.02rem; max-width: 820px; }
.hero .tags { margin-top: 14px; }
.tag { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3); color: #fff;
       padding: 3px 12px; border-radius: 999px; font-size: .78rem; margin: 0 6px 6px 0; }
.phead { display: flex; align-items: center; gap: 14px; padding: 14px 20px; border-radius: 16px; margin: 2px 0 16px;
         background: linear-gradient(100deg, rgba(31,122,77,.14), rgba(42,111,176,.12)); border: 1px solid rgba(31,122,77,.22); }
.phead .ic { font-size: 1.8rem; } .phead h2 { margin: 0; padding: 0; font-size: 1.4rem; font-weight: 700; }
.phead p { margin: 0; opacity: .75; font-size: .92rem; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 6px 0 18px; }
.kpi { background: rgba(255,255,255,.75); border: 1px solid rgba(120,120,110,.2); border-left: 6px solid #1f7a4d; border-radius: 14px;
       padding: 14px 18px; box-shadow: 0 2px 10px rgba(0,0,0,.05); }
.kpi .kl { font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; color: #5d6b64; font-weight: 600; }
.kpi .kv { font-size: 1.85rem; font-weight: 800; line-height: 1.15; margin: 3px 0 2px; color: #1d2a24; }
.kpi .ks { font-size: .82rem; color: #5d6b64; }
.callouts { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 8px 0 14px; }
.callout { border-radius: 14px; padding: 14px 18px; border: 1px solid; background: rgba(255,255,255,.7); }
.callout h4 { margin: 0 0 6px; font-size: 1rem; } .callout ul { margin: 0; padding-left: 18px; font-size: .92rem; }
.callout.ok { border-color: #1f7a4d; background: rgba(31,122,77,.09); } .callout.mid { border-color: #b8860b; background: rgba(184,134,11,.10); }
.callout.bad { border-color: #b23a2a; background: rgba(178,58,42,.09); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b4a32 0%, #14613f 55%, #1d5f8f 100%); }
[data-testid="stSidebar"] * { color: #f2f7f4 !important; }
[data-testid="stSidebar"] .brand { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 2px; }
[data-testid="stSidebar"] .brand-sub { font-size: .8rem; opacity: .8; margin-bottom: 14px; }
div[data-testid="stMetric"] { background: rgba(255,255,255,.7); border: 1px solid rgba(120,120,110,.2); border-radius: 14px; padding: 10px 14px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.foot { text-align: center; opacity: .6; font-size: .8rem; margin-top: 28px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, tags=()):
    t = "".join(f'<span class="tag">{x}</span>' for x in tags if x)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="tags">{t}</div></div>', unsafe_allow_html=True)


def page_header(icon, title, sub=""):
    st.markdown(f'<div class="phead"><div class="ic">{icon}</div><div><h2>{title}</h2><p>{sub}</p></div></div>', unsafe_allow_html=True)


def kpi_row(items):
    """items : (libelle, valeur, sous-texte, couleur)."""
    h = "".join(f'<div class="kpi" style="border-left-color:{c}"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div></div>'
                for l, v, s, c in items)
    st.markdown(f'<div class="kpis">{h}</div>', unsafe_allow_html=True)


def callouts(blocks):
    """blocks : (classe ok/mid/bad, titre, [puces])."""
    h = "".join(f'<div class="callout {k}"><h4>{t}</h4><ul>{"".join(f"<li>{i}</li>" for i in its)}</ul></div>' for k, t, its in blocks)
    st.markdown(f'<div class="callouts">{h}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ données
@st.cache_data
def load(name):
    p = DATA / f"{name}.csv.gz"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def load_geo():
    return json.loads((DATA / "provinces.geojson").read_text(encoding="utf-8"))


if not (DATA / "meta.json").exists():
    st.error("**Dossier `data/` introuvable.** Cette application a besoin des fichiers de données (fichiers .csv.gz, provinces.geojson, "
             "meta.json). Envoyez dans le dépôt GitHub, à côté de `app.py`, soit le dossier `data`, soit une archive `data.zip` contenant ce dossier.")
    seen = sorted(p.name + ("/" if p.is_dir() else "") for p in HERE.iterdir() if not p.name.startswith("."))
    st.write("Ce que l'application voit dans le dépôt :", seen)
    st.write("Dossier cherché :", str(HERE / "data"))
    st.stop()
META = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
GEO = load_geo()
PROV = load("provinces")
LABEL = dict(zip(PROV.code, PROV.label))
LABEL["NATIONAL"] = "National (moyenne des provinces)"
CODES = list(PROV.code)
NORM, CLIM, REG, TR = load("normals"), load("clim"), load("regime"), load("trends")
ANN, SPI, SPIA, HZ = load("annual"), load("spi"), load("spi_annual"), load("hazard")
RAIN, CONS, VAL, BRK = load("rain_sources"), load("consensus"), load("validation"), load("breaks")
NMON, PROJ = load("nasa_monthly"), load("proj")
STATUS = META.get("rain_status", "inconnue")
_boot.empty()

SRC = {"nasa": ("NASA POWER", "#c0482b"), "era5": ("ERA5", "#7a4bb0"), "gpcc": ("GPCC", "#2a6fb0"),
       "tamsat": ("TAMSAT", "#1f8a5b"), "chirps": ("CHIRPS", "#b8860b")}
LB = {"PRCPTOT": ("Pluie annuelle", "mm"), "TMEAN": ("Température moyenne", "°C"),
      "TXMEAN": ("Température maximale moyenne", "°C"), "TNMEAN": ("Température minimale moyenne", "°C"),
      "RX1DAY": ("Pluie max. en 1 jour", "mm"), "RX5DAY": ("Pluie max. en 5 jours", "mm"),
      "R10MM": ("Jours ≥ 10 mm", "jours"), "R20MM": ("Jours ≥ 20 mm", "jours"), "R50MM": ("Jours ≥ 50 mm", "jours"),
      "CDD": ("Jours secs consécutifs", "jours"), "CWD": ("Jours humides consécutifs", "jours"),
      "TXX": ("Température max. absolue", "°C"), "TNN": ("Température min. absolue", "°C"),
      "TX90P": ("Jours chauds (TX90p)", "% jours"), "TN90P": ("Nuits chaudes (TN90p)", "% jours"),
      "SU30": ("Jours > 30 °C", "jours"), "TXGE35": ("Jours ≥ 35 °C", "jours"), "TR20": ("Nuits > 20 °C", "jours"),
      "WET_DAYS": ("Jours de pluie", "jours"), "SDII": ("Intensité des jours pluvieux", "mm/jour"),
      "DTR": ("Amplitude thermique journalière", "°C"), "RH_MEAN": ("Humidité relative", "%"),
      "R95PTOT": ("Pluie des jours très humides", "mm"), "R95P_SHARE": ("Part des pluies très fortes", "%")}
RAIN_IND = {"PRCPTOT", "RX1DAY", "RX5DAY", "R10MM", "R20MM", "R50MM", "WET_DAYS", "SDII", "R95PTOT",
            "R95P_SHARE", "CDD", "CWD"}
MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
SCALE = {"seq-blue": "Blues", "seq-warm": "YlOrRd", "seq-dry": "YlOrBr", "seq-red": "Reds",
         "div-temp": "RdBu_r", "div-rain": "BrBG"}


def lab(k):
    return LB.get(k, (k, ""))[0]


def f(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def sg(x, d=1):
    return "–" if x is None or pd.isna(x) else ("+" if x >= 0 else "") + f(x, d)


def conf(ind):
    """Fiabilité d'une tendance : (niveau, texte)."""
    if ind in RAIN_IND:
        return ("🟢", "à confirmer") if STATUS == "corroboree" else ("🔴", "non établie")
    v = VAL[(VAL.province == "NATIONAL") & (VAL.indicator == ind)] if len(VAL) and "indicator" in VAL else VAL
    if len(v) and pd.notna(v.iloc[0].get("sen_slope_per_decade_era5")):
        same = np.sign(v.iloc[0].sen_slope_per_decade_nasa) == np.sign(v.iloc[0].sen_slope_per_decade_era5)
        return ("🟠", "sens robuste, ampleur incertaine") if same else ("🔴", "sources divergentes")
    return "🟠", "source unique"


def theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    if not sl:
        return 0.0, float(np.median(y))
    s = float(np.median(sl))
    return s, float(np.median(y - s * x))


def banner():
    msg = {"non_etablie": ("warning", "**Tendance de pluie non établie.** Les réanalyses (NASA POWER, ERA5) montrent une forte "
                          "baisse avec un saut brusque vers 1998–2001 que les sources indépendantes (GPCC, TAMSAT) ne "
                          "confirment pas. Les normales de pluie sont utilisables ; la tendance ne doit pas être citée comme un fait."),
           "partielle": ("warning", "**Baisse de pluie partiellement corroborée** : probablement réelle mais surestimée par les réanalyses."),
           "corroboree": ("success", "**Baisse de pluie corroborée** par des sources indépendantes."),
           "inconnue": ("warning", "**Fiabilité de la tendance de pluie non vérifiée** (aucune source indépendante).")}[STATUS]
    getattr(st, msg[0])(msg[1])


# ------------------------------------------------------------------ graphiques
def fig_clim(code):
    g = CLIM[CLIM.province == code].sort_values("month")
    if g.empty:
        return None
    fg = make_subplots(specs=[[{"secondary_y": True}]])
    fg.add_trace(go.Bar(x=MOIS, y=g.precip_mm, name="Pluie (mm/mois)", marker_color="#3a7cbf", opacity=.8), secondary_y=False)
    fg.add_trace(go.Scatter(x=MOIS, y=g.t2m_c, name="Température moy. (°C)", line=dict(color="#c0482b", width=3)), secondary_y=True)
    if "tmax_c" in g:
        fg.add_trace(go.Scatter(x=MOIS, y=g.tmax_c, name="Tmax", line=dict(color="#c0482b", dash="dot")), secondary_y=True)
        fg.add_trace(go.Scatter(x=MOIS, y=g.tmin_c, name="Tmin", line=dict(color="#e0a090", dash="dot")), secondary_y=True)
    fg.update_yaxes(title_text="mm/mois", secondary_y=False)
    fg.update_yaxes(title_text="°C", secondary_y=True)
    fg.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15))
    return fg


def fig_sources(code, height=360):
    g = RAIN[RAIN.province == code]
    if g.empty:
        return None
    g = g.assign(Source=g.source.map(lambda s: SRC.get(s, (s,))[0]))
    fg = px.line(g, x="year", y="mm", color="Source", color_discrete_map={v[0]: v[1] for v in SRC.values()},
                 labels={"year": "Année", "mm": "Pluie annuelle (mm)"})
    fg.add_vline(x=2000, line_dash="dot", line_color="grey")
    fg.update_layout(height=height, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    return fg


def choropleth(values, unit, scale, mid0=False, height=560):
    d = pd.DataFrame({"code": list(values), "v": list(values.values())}).dropna()
    d["Province"] = d.code.map(LABEL)
    kw = dict(color_continuous_midpoint=0) if mid0 else {}
    fg = px.choropleth(d, geojson=GEO, locations="code", featureidkey="properties.code", color="v",
                       hover_name="Province", color_continuous_scale=scale, labels={"v": unit}, **kw)
    fg.update_traces(marker_line_color="white", marker_line_width=0.8)
    fg.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
    fg.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height,
                     coloraxis_colorbar=dict(title=unit, thickness=14, len=0.7))
    return fg


# ------------------------------------------------------------------ couches de la carte
def build_layers():
    L = {}
    P = NORM[NORM.province != "NATIONAL"]
    piv = P.pivot(index="province", columns="indicator", values="normal")
    for k in ["PRCPTOT", "TMEAN", "TXMEAN", "TNMEAN", "RX1DAY", "R20MM", "CDD", "TXX", "SU30", "RH_MEAN", "SDII"]:
        if k in piv:
            L[f"Normales · {lab(k)} ({LB[k][1]})"] = dict(v=piv[k].to_dict(), u=LB[k][1], p=SCALE["seq-dry" if k == "CDD" else
                                                          "seq-blue" if k in RAIN_IND or k == "RH_MEAN" else "seq-warm"], d=1, mid=False, note="")
    T = TR[TR.province != "NATIONAL"]
    for k in ["TMEAN", "TXMEAN", "TNMEAN", "CDD"]:
        s = T[T.indicator == k].set_index("province").sen_slope_per_decade
        if len(s):
            L[f"Tendance · {lab(k)}"] = dict(v=s.to_dict(), u=f"{LB[k][1]}/décennie", p=SCALE["div-temp"], d=2, mid=True,
                                              note="Source : NASA POWER. " + conf(k)[1].capitalize() + ".")
    if len(CONS):
        for s in SRC:
            c = f"{s}_pct_dec"
            if c in CONS:
                L[f"Pluie annuelle, tendance · {SRC[s][0]}"] = dict(v=dict(zip(CONS.province, CONS[c])), u="%/décennie",
                                                                    p=SCALE["div-rain"], d=1, mid=True,
                                                                    note="Les sources ne s'accordent pas : cette couche montre le désaccord.")
    for c, name in [("heat_score", "chaleur"), ("flood_score", "inondations"), ("drought_score", "sécheresse"), ("composite", "composite")]:
        if len(HZ) and c in HZ:
            L[f"Exposition · {name}"] = dict(v=dict(zip(HZ.province, HZ[c])), u="score 0-100", p=SCALE["seq-red"], d=0, mid=False,
                                             note="Comparaison relative entre provinces, pas un risque absolu.")
    if len(PROJ):
        for ind, g in PROJ.groupby("indicator"):
            kind = g.kind.iloc[0]
            if kind not in ("temp", "rain"):
                continue
            a = g[g.year == g.year.min()].set_index("province").value
            z = g[g.year == g.year.max()].set_index("province").value
            if kind == "temp":
                L["Projection · hausse de température 2026→2040"] = dict(v=(z - a).to_dict(), u="°C", p=SCALE["seq-warm"], d=2, mid=False,
                                                                        note="Modèle unique, SSP5-8.5 : haut de fourchette.")
            else:
                L["Projection · variation de la pluie 2026→2040"] = dict(v=((z - a) / a * 100).to_dict(), u="%", p=SCALE["div-rain"], d=1,
                                                                         mid=True, note="Modèle unique, SSP5-8.5 : haut de fourchette.")
    return L


# ------------------------------------------------------------------ rapport province
def province_report(code):
    n = NORM[NORM.province == code].set_index("indicator")
    t = TR[TR.province == code].set_index("indicator")
    out = [f"# Fiche climatique : {LABEL[code]}", f"Généré le {META.get('built', '')}", "", "## Normales 1991–2020"]
    for k in ["PRCPTOT", "TMEAN", "TXMEAN", "TNMEAN", "RX1DAY", "CDD", "TXX"]:
        if k in n.index:
            out.append(f"- {lab(k)} : {f(n.loc[k, 'normal'], 1)} {LB[k][1]}")
    out += ["", "## Tendances 1981–2024 (NASA POWER)"]
    for k in ["TMEAN", "TXMEAN", "TNMEAN", "PRCPTOT", "RX1DAY", "CDD"]:
        if k in t.index:
            r = t.loc[k]
            out.append(f"- {lab(k)} : {sg(r.sen_slope_per_decade, 2)} {LB[k][1]}/décennie ({r.significance}) — fiabilité : {conf(k)[1]}")
    if len(HZ) and code in set(HZ.province):
        h = HZ[HZ.province == code].iloc[0]
        out += ["", "## Exposition relative (0-100)", f"- Chaleur {f(h.heat_score, 0)} · Inondations {f(h.flood_score, 0)} · Sécheresse {f(h.drought_score, 0)}"]
    out += ["", "## Limites", "La tendance de pluie n'est pas établie (sources en désaccord). Projections : modèle unique, SSP5-8.5."]
    return "\n".join(out)


# ------------------------------------------------------------------ pages
def page_portrait():
    hero("🌍 Atlas climatique de la République démocratique du Congo",
         "Normales 1991-2020, tendances 1981-2024, fiabilité des données et projections 2026-2040 pour les 26 provinces.",
         [META.get("analysis", ""), "NASA POWER", "ERA5", "GPCC", "TAMSAT", "CMIP6", f"mis à jour le {META.get('built', '')}"])
    banner()
    n = NORM[NORM.province == "NATIONAL"].set_index("indicator").normal
    t = TR[TR.province == "NATIONAL"].set_index("indicator")
    items = [("Pluie annuelle", f"{f(n.get('PRCPTOT'), 0)} mm", "normale 1991-2020", "#2a6fb0"),
             ("Température moyenne", f"{f(n.get('TMEAN'))} °C", "normale 1991-2020", "#c0482b"),
             ("Maximales / minimales", f"{f(n.get('TXMEAN'))} / {f(n.get('TNMEAN'))} °C", "moyennes annuelles", "#e0763a")]
    if "TMEAN" in t.index:
        items.append(("Réchauffement", f"{sg(t.loc['TMEAN', 'sen_slope_per_decade'], 2)} °C/déc.", conf("TMEAN")[1], "#b8860b"))
    kpi_row(items)
    a, b = st.columns(2)
    with a:
        st.subheader("Régime climatique national")
        fg = fig_clim("NATIONAL")
        if fg:
            st.plotly_chart(fg, use_container_width=True)
    with b:
        st.subheader("Pluie annuelle selon la source")
        fg = fig_sources("NATIONAL")
        if fg:
            st.plotly_chart(fg, use_container_width=True)
        st.caption("Un saut simultané des seules réanalyses (NASA, ERA5), sans équivalent dans les autres sources, évoque un changement du système d'observation plutôt que du climat.")
    st.subheader("Ce qui est fiable, ce qui ne l'est pas")
    callouts([("ok", "✅ Fiable", ["Cycle saisonnier et ordres de grandeur des normales", "Sens du réchauffement (NASA POWER et ERA5)",
                                   "Contrastes régionaux : relief de l'Est, saison sèche du Sud"]),
              ("mid", "⚠️ À nuancer", ["Ampleur du réchauffement (écart entre sources)", "Maximales et minimales", "Pluies extrêmes (réanalyses seules)"]),
              ("bad", "⛔ Non établi", ["Tendance de la pluie annuelle 1981-2024", "Classement des provinces « qui s'assèchent »"])])


def page_carte():
    page_header("🗺️", "Carte des 26 provinces", "Une quarantaine de couches : normales, tendances, exposition, projections")
    L = build_layers()
    if not L:
        st.warning("Aucune couche disponible.")
        return
    name = st.selectbox("Couche", list(L))
    lay = L[name]
    st.plotly_chart(choropleth(lay["v"], lay["u"], lay["p"], lay["mid"]), use_container_width=True)
    if lay["note"]:
        st.caption(lay["note"])
    d = pd.DataFrame({"Province": [LABEL.get(k, k) for k in lay["v"]], f"Valeur ({lay['u']})": list(lay["v"].values())})
    st.dataframe(d.sort_values(d.columns[1], ascending=False), hide_index=True, use_container_width=True)


def page_province():
    page_header("📍", "Fiche province", "Climat, tendances, sources de pluie, sécheresse et exposition")
    code = st.selectbox("Province", ["NATIONAL"] + CODES, format_func=lambda c: LABEL.get(c, c))
    r = REG[REG.province == code]
    if len(r):
        r = r.iloc[0]
        st.caption(f"Saison sèche : {f(r.longest_dry_season_months, 0)} mois (< 50 mm) · mois le plus pluvieux : {MOIS[int(r.wettest_month) - 1]} · "
                   f"amplitude thermique annuelle : {f(r.temp_amplitude_c)} °C")
    t1, t2, t3, t4, t5 = st.tabs(["Climat", "Séries et tendances", "Sources de pluie", "Sécheresse (SPI)", "Exposition"])
    with t1:
        fg = fig_clim(code)
        if fg:
            st.plotly_chart(fg, use_container_width=True)
        n = NORM[NORM.province == code]
        if len(n):
            st.dataframe(n.assign(Indicateur=n.indicator.map(lab))[["Indicateur", "unit", "normal", "std", "min", "max", "cv_pct"]]
                         .rename(columns={"unit": "Unité", "normal": "Normale", "std": "Écart-type", "cv_pct": "CV %"}).round(2),
                         hide_index=True, use_container_width=True)
    with t2:
        inds = [k for k in LB if k in set(ANN.columns)]
        k = st.selectbox("Indicateur", inds, format_func=lab, index=inds.index("TMEAN") if "TMEAN" in inds else 0)
        g = ANN[ANN.province == code][["year", k]].dropna()
        if len(g):
            fg = px.line(g, x="year", y=k, markers=True, labels={"year": "Année", k: f"{lab(k)} ({LB[k][1]})"})
            tt = TR[(TR.province == code) & (TR.indicator == k)]
            if len(tt):
                s, i0 = theil_sen(g.year, g[k])
                fg.add_trace(go.Scatter(x=g.year, y=s * g.year + i0, name="Tendance (Theil-Sen)", line=dict(color="#c0482b", dash="dash")))
                rr = tt.iloc[0]
                st.markdown(f"**Tendance :** {sg(rr.sen_slope_per_decade, 2)} {LB[k][1]}/décennie · {rr.significance} · "
                            f"fiabilité {conf(k)[0]} {conf(k)[1]}")
            fg.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fg, use_container_width=True)
        tr = TR[TR.province == code]
        if len(tr):
            tr = tr.assign(Indicateur=tr.indicator.map(lab), Fiabilité=tr.indicator.map(lambda i: conf(i)[0] + " " + conf(i)[1]))
            st.dataframe(tr[["Indicateur", "sen_slope_per_decade", "pct_per_decade", "significance", "Fiabilité"]].rename(
                columns={"sen_slope_per_decade": "Pente /décennie", "pct_per_decade": "% /décennie", "significance": "Signif."}).round(2),
                hide_index=True, use_container_width=True)
    with t3:
        fg = fig_sources(code, 420)
        if fg:
            st.plotly_chart(fg, use_container_width=True)
        else:
            st.info("Une seule source disponible pour cette province.")
        c = CONS[CONS.province == code] if len(CONS) else CONS
        if len(c):
            row = c.iloc[0]
            st.write("Tendance (% par décennie) : " + " · ".join(
                f"{SRC[s][0]} {sg(row[s + '_pct_dec'])}" for s in SRC if s + "_pct_dec" in c and pd.notna(row[s + "_pct_dec"])))
    with t4:
        sc = st.radio("Échelle", [3, 12], format_func=lambda s: f"SPI-{s} mois", horizontal=True)
        g = SPI[(SPI.province == code) & (SPI.scale == sc)].copy() if len(SPI) else SPI
        if len(g):
            g["date"] = pd.to_datetime(dict(year=g.year, month=g.month, day=1))
            fg = go.Figure(go.Bar(x=g.date, y=g.spi, marker_color=np.where(g.spi < 0, "#c08a2b", "#2a6fb0")))
            fg.add_hline(y=-1, line_dash="dot", line_color="grey")
            fg.update_layout(height=360, yaxis_title="SPI", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fg, use_container_width=True)
            st.caption("SPI ≤ −1 : sécheresse modérée ; ≤ −1,5 : sévère. Calibré sur 1991-2020. Le SPI repose sur la pluie des réanalyses : mêmes réserves que pour la tendance.")
    with t5:
        h = HZ[HZ.province == code] if len(HZ) else HZ
        if len(h):
            h = h.iloc[0]
            cols = ["heat_score", "flood_score", "drought_score"]
            fg = go.Figure(go.Scatterpolar(r=[h[c] for c in cols] + [h[cols[0]]], theta=["Chaleur", "Inondation", "Sécheresse", "Chaleur"], fill="toself"))
            fg.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=360, margin=dict(l=30, r=30, t=20, b=20))
            st.plotly_chart(fg, use_container_width=True)
            st.caption("Exposition relative entre provinces (70 % niveau, 30 % tendance), pas un risque absolu.")
        else:
            st.info("Scores non disponibles pour ce territoire.")
    if code != "NATIONAL":
        st.download_button("📄 Télécharger la fiche (Markdown)", province_report(code), file_name=f"fiche_{code}.md")


def page_comparer():
    page_header("⚖️", "Comparer des provinces", "Superposez le climat et les séries de 2 à 6 provinces")
    default = [c for c in ["Kinshasa", "Ituri", "Haut-Katanga"] if c in CODES] or CODES[:2]
    sel = st.multiselect("Provinces (2 à 6)", CODES, default=default, format_func=lambda c: LABEL[c], max_selections=6)
    if len(sel) < 2:
        st.info("Choisissez au moins deux provinces.")
        return
    c1, c2 = st.columns(2)
    d = CLIM[CLIM.province.isin(sel)].assign(Province=lambda x: x.province.map(LABEL), Mois=lambda x: x.month.map(lambda m: MOIS[m - 1]))
    c1.plotly_chart(px.line(d, x="Mois", y="precip_mm", color="Province", markers=True, category_orders={"Mois": MOIS}, labels={"precip_mm": "Pluie (mm/mois)"}).update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10)), use_container_width=True)
    c2.plotly_chart(px.line(d, x="Mois", y="t2m_c", color="Province", markers=True, category_orders={"Mois": MOIS}, labels={"t2m_c": "Température (°C)"}).update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10)), use_container_width=True)
    inds = [k for k in LB if k in set(ANN.columns)]
    k = st.selectbox("Indicateur annuel", inds, format_func=lab, index=inds.index("TMEAN") if "TMEAN" in inds else 0)
    a = ANN[ANN.province.isin(sel)].assign(Province=lambda x: x.province.map(LABEL))
    st.plotly_chart(px.line(a, x="year", y=k, color="Province", labels={"year": "Année", k: f"{lab(k)} ({LB[k][1]})"}).update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10)), use_container_width=True)
    n = NORM[NORM.province.isin(sel) & NORM.indicator.isin(["PRCPTOT", "TMEAN", "TXMEAN", "RX1DAY", "CDD", "TXX"])]
    pv = n.pivot(index="indicator", columns="province", values="normal").rename(index=lab, columns=LABEL).round(1)
    st.dataframe(pv, use_container_width=True)


def page_tendances():
    page_header("📈", "Tendances et fiabilité", "Que peut-on affirmer ? Comparaison de cinq sources de données")
    banner()
    t = TR[TR.province == "NATIONAL"]
    order = ["TMEAN", "TXMEAN", "TNMEAN", "TXX", "TX90P", "TN90P", "SU30", "PRCPTOT", "RX1DAY", "RX5DAY", "R20MM", "CDD", "SDII"]
    t = t[t.indicator.isin(order)].set_index("indicator").loc[[o for o in order if o in set(t.indicator)]].reset_index()
    era = {}
    if len(VAL) and "indicator" in VAL:
        v = VAL[VAL.province == "NATIONAL"].set_index("indicator")
        era = v.sen_slope_per_decade_era5.to_dict() if "sen_slope_per_decade_era5" in v else {}
    t = t.assign(Indicateur=t.indicator.map(lab), ERA5=t.indicator.map(lambda i: era.get(i)),
                 Fiabilité=t.indicator.map(lambda i: conf(i)[0] + " " + conf(i)[1]))
    st.subheader("National 1981–2024")
    st.dataframe(t[["Indicateur", "unit", "sen_slope_per_decade", "significance", "ERA5", "Fiabilité"]].rename(
        columns={"unit": "Unité", "sen_slope_per_decade": "NASA POWER /décennie", "significance": "Signif.", "ERA5": "ERA5 /décennie"}).round(3),
        hide_index=True, use_container_width=True)
    if len(CONS):
        st.subheader("Pluie annuelle : tendance par province et par source (% par décennie)")
        cols = [f"{s}_pct_dec" for s in SRC if f"{s}_pct_dec" in CONS]
        m = CONS.set_index("province")[cols].rename(columns={c: SRC[c[:-8]][0] for c in cols}).rename(index=LABEL)
        fg = px.imshow(m.round(1), color_continuous_scale="BrBG", color_continuous_midpoint=0, aspect="auto", text_auto=True,
                       labels=dict(color="%/décennie"))
        fg.update_layout(height=max(420, 24 * len(m)), margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fg, use_container_width=True)
        st.caption("Si les colonnes se contredisent, la tendance n'est pas établie.")
        if META.get("verdict"):
            st.info(f"Verdict automatique : {META['verdict']}")
    if len(BRK):
        st.subheader("Ruptures de la pluie annuelle (test de Pettitt)")
        b = BRK[BRK.indicator == "PRCPTOT"]
        sm = b.groupby("source").agg(provinces=("province", "count"), significatives=("pettitt_p", lambda s: int((s < .05).sum())),
                                     annee_mediane=("break_year", "median"), saut_median_pct=("change_pct", "median")).round(1)
        st.dataframe(sm, use_container_width=True)


def proj_series():
    return PROJ.pivot_table(index=["province", "year"], columns="indicator", values="value").reset_index() if len(PROJ) else PROJ


def malaria(t, p):
    ft = np.clip((t - 18) * (34 - t) / 63, 0, 1)
    return ft * np.clip((p / 12 - 15) / 65, 0, 1)


def page_projections():
    page_header("🔮", "Projections 2026–2040", "Modèle CMIP6, SSP5-8.5, appliqué comme changement sur la référence observée")
    if PROJ.empty:
        st.warning("Projections indisponibles (table CMIP6 de l'Atlas absente).")
        return
    st.warning("**Lecture prudente.** Un seul modèle (CNRM-CM6-1-HR), scénario SSP5-8.5 très émetteur : haut de fourchette. "
               "Interpolation entre périodes de 30 à 50 ans : pas de variabilité d'une année à l'autre. Température et pluie sont "
               "exprimées comme un **changement appliqué à la référence observée** (correction de biais « delta »).")
    meta = PROJ.drop_duplicates("indicator")[["indicator", "label", "unit", "kind"]].set_index("indicator")
    c1, c2 = st.columns(2)
    ind = c1.selectbox("Indicateur", list(meta.index), format_func=lambda i: meta.loc[i, "label"])
    terr = c2.selectbox("Territoire", ["NATIONAL"] + [c for c in CODES if c in set(PROJ.province)], format_func=lambda c: LABEL.get(c, c))
    g = PROJ[PROJ.indicator == ind]
    s = g.groupby("year").value.mean() if terr == "NATIONAL" else g[g.province == terr].set_index("year").value
    base = g.groupby("province").base.first().mean() if terr == "NATIONAL" else g[g.province == terr].base.iloc[0]
    fg = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines+markers", line=dict(color="#c0482b", width=3)))
    fg.add_hline(y=base, line_dash="dot", line_color="grey", annotation_text="référence")
    fg.update_layout(height=380, yaxis_title=f"{meta.loc[ind, 'label']}", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fg, use_container_width=True)
    k = st.columns(4)
    k[0].metric("Référence", f(base, 2))
    for col, y in zip(k[1:], [2026, 2033, 2040]):
        v = s.get(y)
        col.metric(str(y), f(v, 2), sg(v - base, 2) if v is not None else None, delta_color="off")
    d = g.pivot(index="province", columns="year", values="value")
    d = pd.DataFrame({"Province": [LABEL.get(c, c) for c in d.index], "2026": d[2026].values, "2040": d[2040].values})
    d["Évolution"] = d["2040"] - d["2026"]
    st.dataframe(d.sort_values("Évolution", ascending=False).round(2), hide_index=True, use_container_width=True)


def page_risques():
    page_header("⚠️", "Problèmes et risques 2026–2040", "Chaleur, pluies fortes, saisons sèches, variabilité, paludisme")
    st.caption("Chaque « problème » est suivi par l'indicateur climatique correspondant. Valeurs corrigées du biais pour température et pluie ; autres indicateurs : valeurs brutes du modèle (changements, pas niveaux absolus).")
    if PROJ.empty:
        st.warning("Projections indisponibles.")
        return
    W = proj_series()
    inds = list(PROJ.indicator.unique())
    pick = lambda rg: next((i for i in inds if re.search(rg, i, re.I) and PROJ[PROJ.indicator == i].kind.iloc[0] == "raw"), None)
    temp = next((i for i in inds if PROJ[PROJ.indicator == i].kind.iloc[0] == "temp"), None)
    rain = next((i for i in inds if PROJ[PROJ.indicator == i].kind.iloc[0] == "rain"), None)
    probs = {}
    if temp:
        probs["Chaleur (température moyenne, °C)"] = W.set_index(["province", "year"])[temp]
    if pick(r"heavy|intens|extreme"):
        probs["Pluies fortes et inondations (intensité)"] = W.set_index(["province", "year"])[pick(r"heavy|intens|extreme")]
    if pick(r"dry"):
        probs["Saisons sèches (fréquence de mois secs)"] = W.set_index(["province", "year"])[pick(r"dry")]
    if pick(r"(^|_)cv|var"):
        probs["Variabilité de la pluie (CV)"] = W.set_index(["province", "year"])[pick(r"(^|_)cv|var")]
    if temp and rain:
        probs["Aptitude climatique au paludisme (proxy 0-1)"] = malaria(W.set_index(["province", "year"])[temp], W.set_index(["province", "year"])[rain])
    if not probs:
        st.info("Indicateurs non reconnus dans la table de projections.")
        return
    yr = st.slider("Année", 2026, 2040, 2040)
    cols = st.columns(min(3, len(probs)))
    for i, (name, ser) in enumerate(probs.items()):
        a = ser.xs(2026, level="year")
        z = ser.xs(yr, level="year")
        with cols[i % len(cols)]:
            st.metric(name, f(z.mean(), 2), f"{sg(z.mean() - a.mean(), 2)} depuis 2026", delta_color="off")
            top = (z - a).sort_values(ascending=False).head(3)
            st.caption("Plus forte hausse : " + ", ".join(f"{LABEL.get(p, p)} ({sg(v, 2)})" for p, v in top.items()))
    st.subheader("Évolution nationale (indice 100 = 2026)")
    rows = []
    for name, ser in probs.items():
        m = ser.groupby(level="year").mean()
        rows += [dict(Problème=name, Année=y, Indice=100 * v / m.loc[2026]) for y, v in m.items()]
    st.plotly_chart(px.line(pd.DataFrame(rows), x="Année", y="Indice", color="Problème").update_layout(height=400, legend=dict(orientation="h", y=-0.25)), use_container_width=True)
    if len(HZ):
        st.subheader("Exposition actuelle relative (observations)")
        m = HZ.assign(Province=HZ.province.map(LABEL)).sort_values("composite", ascending=False)
        st.plotly_chart(px.bar(m, x="Province", y=["heat_score", "flood_score", "drought_score"], barmode="group",
                               labels={"value": "Score 0-100", "variable": ""}).update_layout(height=420, legend=dict(orientation="h", y=-0.4)), use_container_width=True)


def page_donnees():
    page_header("📥", "Données et téléchargements", "Toutes les tables de cet atlas en CSV")
    tabs = {"Normales": NORM, "Climatologie mensuelle": CLIM, "Régime": REG, "Tendances": TR, "Indices annuels": ANN,
            "Pluie annuelle par source": RAIN, "Comparaison des sources": CONS, "Exposition": HZ, "SPI annuel": SPIA,
            "Projections": PROJ, "Validation NASA/ERA5": VAL, "Ruptures": BRK}
    name = st.selectbox("Table", [k for k, v in tabs.items() if len(v)])
    df = tabs[name]
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{name.lower().replace(' ', '_')}.csv")


def page_methodes():
    page_header("ℹ️", "Méthodes, sources et limites", "Ce qui est affirmé, et ce qui ne peut pas l être")
    st.markdown("""
**Sources** : NASA POWER (quotidien 1981–2025, base des normales et tendances) · ERA5 (réanalyse, contrôle croisé) ·
GPCC, TAMSAT, CHIRPS (pluie indépendante des réanalyses) · CMIP6 CNRM-CM6-1-HR, SSP5-8.5 (Atlas d'origine) · limites GADM 4.1.

**Méthodes** : indices ETCCDI ; tendances de Theil-Sen avec test de Mann-Kendall (sans correction d'autocorrélation) ;
percentiles sur 1991-2020 ; SPI calibré sur 1991-2020 ; ruptures par test de Pettitt ; cohérence entre sources sur période commune.
Scores d'exposition = comparaison relative entre provinces (70 % niveau, 30 % tendance). « National » = moyenne simple des provinces.

**Limites** : aucun jeu n'est une vérité de terrain (réseau de stations très clairsemé en RDC) ; la tendance de la pluie n'est pas établie ;
les projections reposent sur un seul modèle et un scénario très émetteur ; interpolation entre périodes, sans variabilité interannuelle.
""")
    st.json({k: v for k, v in META.items() if k != "verdict"})


PAGES = {"🌍 Portrait national": page_portrait, "🗺️ Carte": page_carte, "📍 Fiche province": page_province,
         "⚖️ Comparer": page_comparer, "📈 Tendances et fiabilité": page_tendances, "🔮 Projections 2026–2040": page_projections,
         "⚠️ Problèmes et risques": page_risques, "📥 Données": page_donnees, "ℹ️ Méthodes": page_methodes}
st.sidebar.markdown('<div class="brand">🌍 Atlas climatique RDC</div><div class="brand-sub">26 provinces · 1981-2040</div>', unsafe_allow_html=True)
PAGES[st.sidebar.radio("Navigation", list(PAGES), label_visibility="collapsed")]()
st.sidebar.markdown("---")
st.sidebar.caption("Données : NASA POWER, ERA5, GPCC, TAMSAT, CMIP6. Voir « Méthodes » pour les limites.")
st.markdown('<div class="foot">Atlas climatique de la RDC · tendances de pluie non établies · projections indicatives (modèle et scénario uniques)</div>', unsafe_allow_html=True)

with st.sidebar.expander("Diagnostic technique"):
    import plotly
    st.caption(f"Streamlit {st.__version__} · pandas {pd.__version__} · plotly {plotly.__version__}")
    st.caption(f"Données : {DATA}")
    st.caption(f"Tables : normales {len(NORM)}, climat {len(CLIM)}, tendances {len(TR)}, pluie par source {len(RAIN)}, projections {len(PROJ)}")
_boot.empty()
'''


def log(m):
    print(m, flush=True)


def export(conn, out):
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)

    def save(name, df):
        if df is not None and len(df):
            df.to_csv(data / f"{name}.csv.gz", index=False, compression="gzip", float_format="%.5g")
            log(f"  data/{name}.csv.gz  ({len(df)} lignes)")
        else:
            log(f"  (absent) {name}")

    prov = tbl(conn, "provinces")
    if prov.empty:
        sys.exit("Table provinces vide : lancez build_drc_climate_db.py")
    save("provinces", prov[["code", "label", "zone"]])
    feats = []
    for r in prov.itertuples():
        rings = geometry(r.geometry_geojson)
        if rings:
            feats.append(dict(type="Feature", properties=dict(code=r.code, label=r.label, zone=r.zone),
                              geometry=dict(type="MultiPolygon", coordinates=[[ring] for ring in rings])))
    (data / "provinces.geojson").write_text(json.dumps(dict(type="FeatureCollection", features=feats)), encoding="utf-8")
    log(f"  data/provinces.geojson  ({len(feats)} provinces)")

    nm = tbl(conn, "an_normals")
    if nm.empty:
        sys.exit("Tables d'analyse absentes : lancez analyze_drc_climate.py")
    save("normals", nm)
    save("clim", tbl(conn, "an_monthly_climatology"))
    save("regime", tbl(conn, "an_regime"))
    save("trends", tbl(conn, "an_trends"))
    save("annual", tbl(conn, "an_annual_indices"))
    spi = tbl(conn, "an_spi")
    save("spi", spi[spi.scale.isin([3, 12])] if len(spi) else spi)
    save("spi_annual", tbl(conn, "an_spi_annual"))
    save("hazard", tbl(conn, "an_hazard_scores"))
    cons = tbl(conn, "an_rain_consensus")
    save("consensus", cons)
    save("validation", tbl(conn, "an_validation_trends"))
    save("breaks", tbl(conn, "an_validation_breaks"))
    save("nasa_monthly", tbl(conn, "v_nasa_monthly"))

    # pluie annuelle par source (long) + NATIONAL
    ser = {}
    na = tbl(conn, "v_nasa_annual")
    if len(na):
        ser["nasa"] = na.rename(columns={"prcptot_mm": "mm"})[["province", "year", "mm"]]
    try:
        era = pd.read_sql("""SELECT province,year,SUM(precip_mm) mm,COUNT(precip_mm) n FROM obs_era5_daily
                             GROUP BY province,year HAVING n>=330""", conn)
        if len(era):
            ser["era5"] = era[["province", "year", "mm"]]
    except Exception:
        pass
    ind = tbl(conn, "obs_rain_indep_monthly")
    for s, g in ind.groupby("source") if len(ind) else []:
        c = g.groupby(["province", "year"]).precip_mm.agg(["sum", "count"]).reset_index()
        ser[s] = c[c["count"] == 12].rename(columns={"sum": "mm"})[["province", "year", "mm"]]
    rows = []
    common = sorted(set.intersection(*[set(d.province) for d in ser.values()])) if ser else []
    for s, d in ser.items():
        rows.append(d.assign(source=s))
        nat = d[d.province.isin(common)].groupby("year").mm.mean().reset_index()
        rows.append(nat.assign(province="NATIONAL", source=s))
    rain = pd.concat(rows, ignore_index=True)[["source", "province", "year", "mm"]] if rows else pd.DataFrame()
    save("rain_sources", rain)

    # projections (long)
    nbase = {(r.province, "TMEAN"): float(r.normal) for r in nm.itertuples() if r.indicator == "TMEAN"}
    rb = {}
    for r in nm[nm.indicator == "PRCPTOT"].itertuples():
        vals = [float(r.normal)]
        for s in ("gpcc", "tamsat"):
            if s in ser:
                x = ser[s][(ser[s].province == r.province) & ser[s].year.between(1991, 2020)].mm
                if len(x):
                    vals.append(float(x.mean()))
        rb[r.province] = float(np.median(vals))
    proj, why = projections(conn, prov, nbase, rb)
    log(f"  projections : {why}")
    if proj:
        recs = []
        for code, cols in proj["data"].items():
            for m in proj["cols"]:
                if m["id"] in cols:
                    for y, v in zip(proj["years"], cols[m["id"]]):
                        recs.append(dict(province=code, indicator=m["id"], kind=m["kind"], label=m["label"], unit=m["unit"],
                                         year=y, value=v, base=proj["base"][code].get(m["id"])))
        save("proj", pd.DataFrame(recs))

    verdict = cons.verdict_global.iloc[0] if len(cons) else None
    v = (verdict or "").upper()
    status = ("non_etablie" if ("NON" in v or "INCONCLUANT" in v) else "partielle" if "PARTIELLEMENT" in v
              else "corroboree" if "CORROB" in v else "inconnue")
    md = tbl(conn, "metadata")
    md = dict(zip(md.key, md.value)) if len(md) else {}
    (data / "meta.json").write_text(json.dumps(dict(
        built=datetime.now().strftime("%d/%m/%Y %H:%M"), analysis=md.get("analysis_periods", ""),
        verdict=verdict, rain_status=status, sources=sorted(ser)), ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--db", default="drc_climate_atlas.db")
    ap.add_argument("--out", default="space_hf")
    ap.add_argument("--upload", action="store_true", help="publier sur Hugging Face")
    ap.add_argument("--repo-id", default="", help="ex. moncompte/Atlas_Climatique_RDC")
    a, _ = ap.parse_known_args()
    base = Path(a.base_dir)
    dbp = Path(a.db) if Path(a.db).is_absolute() else base / a.db
    if not dbp.exists():
        sys.exit(f"Base introuvable : {dbp}")
    out = Path(a.out) if Path(a.out).is_absolute() else base / a.out
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(dbp)
    log("Export des données ...")
    export(conn, out)
    conn.close()
    (out / "app.py").write_text(APP_SOURCE, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / ".streamlit").mkdir(exist_ok=True)
    (out / ".streamlit" / "config.toml").write_text(STREAMLIT_CONFIG, encoding="utf-8")
    import base64
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out / "data").iterdir()):
            z.write(p, "data/" + p.name)
    (out / "data.zip").write_bytes(buf.getvalue())
    # Version AUTONOME pour GitHub / Streamlit Cloud : donnees integrees dans app.py (un seul fichier a envoyer)
    gh = out.parent / "pour_github"
    gh.mkdir(parents=True, exist_ok=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    begin, end = "# ===ATLAS" + "_DATA_BEGIN===", "# ===ATLAS" + "_DATA_END==="
    body = "\n".join("# " + b64[i:i + 100] for i in range(0, len(b64), 100))
    (gh / "app.py").write_text(APP_SOURCE + "\n\n" + begin + "\n" + body + "\n" + end + "\n", encoding="utf-8")
    (gh / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    mo = (gh / "app.py").stat().st_size / 1e6
    log("\n" + "=" * 74)
    log("TERMINE. Envoyez sur GitHub UNIQUEMENT ces 2 fichiers (le dossier va s'ouvrir) :")
    log(f"   {gh}\\app.py   ({mo:.1f} Mo)   <- doit faire environ 1 Mo")
    log(f"   {gh}\\requirements.txt")
    log("ATTENTION : ne jamais envoyer build_hf_space.py sous le nom app.py.")
    if mo < 0.3:
        log("!!! app.py est trop petit : les donnees ne sont pas integrees. Envoyez-moi ce message.")
    log("=" * 74)
    try:
        os.startfile(str(gh))          # Windows : ouvre le dossier pour vous
    except Exception:
        pass
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    log(f"\nSpace prêt : {out}  ({size:.1f} Mo)")
    log("Test local :  pip install streamlit plotly  puis  cd \"" + str(out) + "\"  &&  streamlit run app.py")
    if a.upload:
        if not a.repo_id or "/" not in a.repo_id:
            sys.exit("--repo-id requis, de la forme  votrenom/Atlas_Climatique_RDC")
        try:
            from huggingface_hub import HfApi
        except ImportError:
            sys.exit("pip install huggingface_hub  puis relancez avec --upload")
        api = HfApi(token=os.environ.get("HF_TOKEN") or None)
        api.create_repo(a.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=a.repo_id, repo_type="space",
                          commit_message="Atlas climatique RDC")
        log(f"Publié : https://huggingface.co/spaces/{a.repo_id}  (la construction prend 2 à 5 minutes)")


if __name__ == "__main__":
    main()
