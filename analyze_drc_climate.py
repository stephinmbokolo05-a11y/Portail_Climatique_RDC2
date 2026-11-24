#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_drc_climate.py
======================
ÉTAPE 2 de l'Atlas climatique RDC : lit la base construite par
build_drc_climate_db.py et calcule, à partir des observations NASA POWER :

  an_annual_indices      indices annuels par province (ETCCDI) + NATIONAL
  an_normals             normales 1991-2020 (moyenne, écart-type, CV, min, max)
  an_monthly_climatology climatologie mensuelle 1991-2020 (régime pluviométrique)
  an_regime              synthèse du régime : mois le plus pluvieux, mois secs...
  an_trends              tendances 1981-2024 : pente de Theil-Sen + test de Mann-Kendall
  an_spi                 SPI mensuel (3, 6, 12 mois), calibré sur 1991-2020
  an_spi_annual          synthèse annuelle de la sécheresse (SPI)
  an_hazard_scores       scores d'exposition 0-100 : chaleur, inondation, sécheresse
  an_nasa_vs_era5        contrôle croisé NASA POWER / ERA5 (si ERA5 est présent)

Dépendances : pandas, numpy, scipy (inclus dans Anaconda).

USAGE
  python analyze_drc_climate.py
  %run analyze_drc_climate.py --export-csv        (dans Jupyter)
  python analyze_drc_climate.py --trend-end 2025

NOTES MÉTHODOLOGIQUES
  * Année retenue seulement si >= 330 jours valides ; mois si >= 25 jours.
  * Percentiles (TX90p, TN90p, R95p, R99p) : fenêtre de 5 jours, base 1991-2020,
    sans rééchantillonnage « bootstrap » (léger biais dans la période de base).
  * Mann-Kendall sans pré-blanchiment (autocorrélation non corrigée).
  * « NATIONAL » = moyenne simple des provinces, non pondérée par la surface.
  * Les scores d'aléas mesurent l'exposition RELATIVE entre provinces, pas un
    risque absolu.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(r"D:\Portail_Climatique_RDC\ATLAS")
MIN_DAYS_YEAR, MIN_DAYS_MONTH = 330, 25


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------
def load_daily(conn, table="obs_nasa_daily"):
    if table == "obs_nasa_daily":
        q = ("SELECT province,date,year,month,precip_mm,t2m_c,tmax_c,tmin_c,rh_pct "
             "FROM obs_nasa_daily")
    else:
        q = ("SELECT province,date,year,month,precip_mm,temp_mean_c AS t2m_c,"
             "tmax_c,tmin_c FROM obs_era5_daily")
    df = pd.read_sql(q, conn, parse_dates=["date"])
    df = df.sort_values(["province", "date"]).reset_index(drop=True)
    doy = df["date"].dt.dayofyear
    leap = df["date"].dt.is_leap_year
    df["idx"] = np.where(leap & (doy > 59), doy - 1, doy).clip(1, 365)  # 29 fév → 28 fév
    return df


def max_run(flags: np.ndarray) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def pct_thresholds(g: pd.DataFrame, col: str, q: float, b0: int, b1: int):
    """Seuil de percentile par jour calendaire (fenêtre de 5 jours), base b0-b1."""
    base = g[(g["year"] >= b0) & (g["year"] <= b1)]
    M = base.pivot_table(index="year", columns="idx", values=col).reindex(
        columns=range(1, 366)).values
    ext = np.concatenate([M[:, -2:], M, M[:, :2]], axis=1)
    return np.array([np.nanpercentile(ext[:, d:d + 5], q) for d in range(365)])


# ---------------------------------------------------------------------------
# Indices annuels (ETCCDI)
# ---------------------------------------------------------------------------
def annual_indices(df, b0, b1):
    out = []
    for prov, g in df.groupby("province"):
        tx90 = pct_thresholds(g, "tmax_c", 90, b0, b1)
        tn90 = pct_thresholds(g, "tmin_c", 90, b0, b1)
        base = g[(g["year"] >= b0) & (g["year"] <= b1)]["precip_mm"]
        wetb = base[base >= 1]
        r95, r99 = (np.nanpercentile(wetb, 95), np.nanpercentile(wetb, 99)) \
            if len(wetb) > 50 else (np.nan, np.nan)
        for yr, y in g.groupby("year"):
            p = y["precip_mm"].to_numpy(float)
            if np.isfinite(p).sum() < MIN_DAYS_YEAR:
                continue
            wet = p >= 1
            tx, tn, tm = (y["tmax_c"].to_numpy(float), y["tmin_c"].to_numpy(float),
                          y["t2m_c"].to_numpy(float))
            ii = y["idx"].to_numpy() - 1
            ptot = np.nansum(p)
            out.append(dict(
                province=prov, year=int(yr),
                PRCPTOT=ptot,
                RX1DAY=np.nanmax(p),
                RX5DAY=pd.Series(p).rolling(5, min_periods=5).sum().max(),
                R10MM=int(np.nansum(p >= 10)), R20MM=int(np.nansum(p >= 20)),
                R50MM=int(np.nansum(p >= 50)),
                WET_DAYS=int(np.nansum(wet)),
                SDII=(np.nansum(p[wet]) / wet.sum()) if wet.sum() else np.nan,
                CDD=max_run(np.nan_to_num(p, nan=99) < 1),
                CWD=max_run(wet),
                R95PTOT=np.nansum(p[p > r95]) if np.isfinite(r95) else np.nan,
                R99PTOT=np.nansum(p[p > r99]) if np.isfinite(r99) else np.nan,
                R95P_SHARE=(100 * np.nansum(p[p > r95]) / ptot) if ptot > 0 and np.isfinite(r95) else np.nan,
                TMEAN=np.nanmean(tm), TXMEAN=np.nanmean(tx), TNMEAN=np.nanmean(tn),
                DTR=np.nanmean(tx - tn),
                TXX=np.nanmax(tx), TNN=np.nanmin(tn),
                TX90P=100 * np.nanmean(tx > tx90[ii]),
                TN90P=100 * np.nanmean(tn > tn90[ii]),
                SU30=int(np.nansum(tx > 30)), TXGE35=int(np.nansum(tx >= 35)),
                TR20=int(np.nansum(tn > 20)),
                RH_MEAN=np.nanmean(y["rh_pct"].to_numpy(float)) if "rh_pct" in y else np.nan,
            ))
    ann = pd.DataFrame(out)
    nat = ann.drop(columns="province").groupby("year").mean(numeric_only=True).reset_index()
    nat.insert(0, "province", "NATIONAL")
    return pd.concat([ann, nat], ignore_index=True)


INDICATORS = [c for c in ["PRCPTOT", "RX1DAY", "RX5DAY", "R10MM", "R20MM", "R50MM", "WET_DAYS",
                          "SDII", "CDD", "CWD", "R95PTOT", "R95P_SHARE", "TMEAN", "TXMEAN",
                          "TNMEAN", "DTR", "TXX", "TNN", "TX90P", "TN90P", "SU30", "TXGE35",
                          "TR20", "RH_MEAN"]]
UNITS = {"PRCPTOT": "mm", "RX1DAY": "mm", "RX5DAY": "mm", "SDII": "mm/jour humide",
         "TMEAN": "°C", "TXMEAN": "°C", "TNMEAN": "°C", "DTR": "°C", "TXX": "°C", "TNN": "°C",
         "TX90P": "% jours", "TN90P": "% jours", "R95P_SHARE": "%", "RH_MEAN": "%",
         "R95PTOT": "mm", "R99PTOT": "mm"}
RAIN_LIKE = {"PRCPTOT", "RX1DAY", "RX5DAY", "R10MM", "R20MM", "R50MM", "WET_DAYS", "SDII",
             "R95PTOT", "R99PTOT"}


# ---------------------------------------------------------------------------
# Normales, climatologie, régime
# ---------------------------------------------------------------------------
def normals(ann, n0, n1):
    sub = ann[(ann.year >= n0) & (ann.year <= n1)]
    rows = []
    for prov, g in sub.groupby("province"):
        for ind in INDICATORS:
            v = g[ind].dropna()
            if len(v) >= 20:
                rows.append(dict(province=prov, indicator=ind, unit=UNITS.get(ind, "jours"),
                                 normal=v.mean(), std=v.std(), min=v.min(), max=v.max(),
                                 cv_pct=100 * v.std() / v.mean() if v.mean() else np.nan,
                                 n_years=len(v), period=f"{n0}-{n1}"))
    return pd.DataFrame(rows)


def monthly_clim(df, n0, n1):
    d = df[(df.year >= n0) & (df.year <= n1)]
    m = d.groupby(["province", "year", "month"]).agg(
        n=("precip_mm", "count"), precip_mm=("precip_mm", "sum"), t2m_c=("t2m_c", "mean"),
        tmax_c=("tmax_c", "mean"), tmin_c=("tmin_c", "mean"), rh_pct=("rh_pct", "mean"),
        wet_days=("precip_mm", lambda s: (s >= 1).sum())).reset_index()
    m = m[m.n >= MIN_DAYS_MONTH]
    clim = m.groupby(["province", "month"]).agg(
        precip_mm=("precip_mm", "mean"), precip_std=("precip_mm", "std"), t2m_c=("t2m_c", "mean"),
        tmax_c=("tmax_c", "mean"), tmin_c=("tmin_c", "mean"), rh_pct=("rh_pct", "mean"),
        wet_days=("wet_days", "mean"), n_years=("year", "count")).reset_index()
    clim["period"] = f"{n0}-{n1}"
    return clim


def regime(clim):
    rows = []
    for prov, g in clim.groupby("province"):
        g = g.sort_values("month")
        p = g["precip_mm"].to_numpy()
        dry = p < 50
        # saison sèche = plus longue série circulaire de mois < 50 mm
        best, cur = 0, 0
        for f in np.concatenate([dry, dry]):
            cur = cur + 1 if f else 0
            best = max(best, min(cur, 12))
        rows.append(dict(province=prov, annual_mm=p.sum(), wettest_month=int(g.iloc[p.argmax()]["month"]),
                         driest_month=int(g.iloc[p.argmin()]["month"]), dry_months_lt50mm=int(dry.sum()),
                         longest_dry_season_months=int(best),
                         seasonality_index=float(np.abs(p - p.mean()).sum() / max(p.sum(), 1e-9)),
                         warmest_month=int(g.iloc[g["t2m_c"].to_numpy().argmax()]["month"]),
                         annual_temp_c=g["t2m_c"].mean(),
                         temp_amplitude_c=g["t2m_c"].max() - g["t2m_c"].min()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tendances : Mann-Kendall + Theil-Sen
# ---------------------------------------------------------------------------
def mann_kendall(y):
    y = np.asarray(y, float)
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(y, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18
    z = 0.0 if s == 0 or var <= 0 else (s - np.sign(s)) / np.sqrt(var)
    return s / (n * (n - 1) / 2), 2 * (1 - stats.norm.cdf(abs(z)))


def trends(ann, t0, t1, norm):
    nm = norm.set_index(["province", "indicator"])["normal"].to_dict()
    rows = []
    for prov, g in ann[(ann.year >= t0) & (ann.year <= t1)].groupby("province"):
        for ind in INDICATORS:
            d = g[["year", ind]].dropna()
            if len(d) < 25:
                continue
            slope, icpt, lo, hi = stats.theilslopes(d[ind].to_numpy(), d["year"].to_numpy())
            tau, p = mann_kendall(d[ind].to_numpy())
            base = nm.get((prov, ind))
            rows.append(dict(
                province=prov, indicator=ind, unit=UNITS.get(ind, "jours"), n_years=len(d),
                period=f"{t0}-{t1}", mean=d[ind].mean(),
                sen_slope_per_decade=slope * 10, ci95_low_per_decade=lo * 10,
                ci95_high_per_decade=hi * 10,
                pct_per_decade=(100 * slope * 10 / base) if (ind in RAIN_LIKE and base) else np.nan,
                change_total=slope * (t1 - t0), mk_tau=tau, mk_p=p,
                significance=("p<0,01" if p < .01 else "p<0,05" if p < .05 else "p<0,10" if p < .10 else "ns"),
                direction="hausse" if slope > 0 else "baisse" if slope < 0 else "stable"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SPI
# ---------------------------------------------------------------------------
def spi_series(x: pd.Series, months: pd.Series, calib: pd.Series, min_pos=15):
    """SPI par mois calendaire : gamma sur valeurs > 0 + probabilité de zéro."""
    out = pd.Series(np.nan, index=x.index)
    for m in range(1, 13):
        sel = (months == m) & x.notna()
        cal = x[sel & calib]
        pos = cal[cal > 0]
        if len(cal) < 20 or len(pos) < min_pos:
            continue
        q0 = 1 - len(pos) / len(cal)
        a, loc, sc = stats.gamma.fit(pos, floc=0)
        h = q0 + (1 - q0) * stats.gamma.cdf(x[sel], a, loc=loc, scale=sc)
        out[sel] = stats.norm.ppf(np.clip(h, 1e-4, 1 - 1e-4))
    return out.clip(-3.5, 3.5)


def spi_all(df, n0, n1, scales):
    d = df.groupby(["province", "year", "month"]).agg(n=("precip_mm", "count"),
                                                       p=("precip_mm", "sum")).reset_index()
    d.loc[d.n < MIN_DAYS_MONTH, "p"] = np.nan
    rows = []
    for prov, g in d.groupby("province"):
        g = g.sort_values(["year", "month"]).reset_index(drop=True)
        calib = (g.year >= n0) & (g.year <= n1)
        for sc in scales:
            acc = g["p"].rolling(sc, min_periods=sc).sum()
            s = spi_series(acc, g["month"], calib)
            rows.append(pd.DataFrame(dict(province=prov, scale=sc, year=g.year, month=g.month, spi=s)))
    spi = pd.concat(rows, ignore_index=True).dropna(subset=["spi"])
    a = []
    for (prov, yr), g in spi.groupby(["province", "year"]):
        s3, s12 = g[g.scale == 3], g[(g.scale == 12) & (g.month == 12)]
        a.append(dict(province=prov, year=int(yr),
                      spi12_dec=s12.spi.iloc[0] if len(s12) else np.nan,
                      months_dry_spi3=int((s3.spi <= -1).sum()),
                      months_severe_spi3=int((s3.spi <= -1.5).sum()),
                      months_wet_spi3=int((s3.spi >= 1).sum())))
    return spi, pd.DataFrame(a)


# ---------------------------------------------------------------------------
# Scores d'exposition
# ---------------------------------------------------------------------------
def minmax(s):
    return 100 * (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0


def hazard_scores(norm, regm, trend):
    n = norm[norm.province != "NATIONAL"].pivot(index="province", columns="indicator", values="normal")
    r = regm.set_index("province")
    t = trend[trend.province != "NATIONAL"].pivot(index="province", columns="indicator",
                                                  values="sen_slope_per_decade")
    sc = pd.DataFrame(index=n.index)
    sc["heat_level"] = (minmax(n["TXMEAN"]) + minmax(n["TXX"]) + minmax(n["SU30"])) / 3
    sc["heat_trend"] = minmax(t["TXMEAN"])
    sc["flood_level"] = (minmax(n["RX1DAY"]) * .35 + minmax(n["R50MM"]) * .25 +
                         minmax(n["R95P_SHARE"]) * .2 + minmax(n["PRCPTOT"]) * .2)
    sc["flood_trend"] = (minmax(t["RX1DAY"]) + minmax(t["R20MM"])) / 2
    sc["drought_level"] = (minmax(n["CDD"]) * .4 + minmax(r["dry_months_lt50mm"]) * .3 +
                           minmax(n["PRCPTOT"].max() - n["PRCPTOT"]) * .3)
    sc["drought_trend"] = (minmax(t["CDD"]) + minmax(-t["PRCPTOT"])) / 2
    for k in ("heat", "flood", "drought"):
        sc[f"{k}_score"] = .7 * sc[f"{k}_level"] + .3 * sc[f"{k}_trend"]
    sc["composite"] = sc[["heat_score", "flood_score", "drought_score"]].mean(axis=1)
    sc = sc.round(1).reset_index()
    sc["method"] = "exposition relative entre provinces : 70 % niveau (normale) + 30 % tendance"
    return sc


# ---------------------------------------------------------------------------
# Contrôle croisé NASA / ERA5
# ---------------------------------------------------------------------------
def nasa_vs_era5(conn, nasa):
    n = conn.execute("SELECT COUNT(*) FROM obs_era5_daily").fetchone()[0]
    if n == 0:
        return pd.DataFrame()
    era = load_daily(conn, "obs_era5_daily")
    f = lambda d: d.groupby(["province", "year", "month"]).agg(
        n=("precip_mm", "count"), p=("precip_mm", "sum"), t=("t2m_c", "mean")).reset_index()
    m = f(nasa).merge(f(era), on=["province", "year", "month"], suffixes=("_nasa", "_era5"))
    m = m[(m.n_nasa >= MIN_DAYS_MONTH) & (m.n_era5 >= MIN_DAYS_MONTH)]
    rows = []
    for prov, g in m.groupby("province"):
        if len(g) < 36:
            continue
        rows.append(dict(province=prov, n_months=len(g),
                         precip_ratio_era5_over_nasa=g.p_era5.sum() / max(g.p_nasa.sum(), 1e-9),
                         precip_corr_monthly=g.p_nasa.corr(g.p_era5),
                         temp_bias_era5_minus_nasa_c=(g.t_era5 - g.t_nasa).mean(),
                         temp_corr_monthly=g.t_nasa.corr(g.t_era5)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--db", default="drc_climate_atlas.db")
    ap.add_argument("--normal-start", type=int, default=1991)
    ap.add_argument("--normal-end", type=int, default=2020)
    ap.add_argument("--trend-start", type=int, default=1981)
    ap.add_argument("--trend-end", type=int, default=2024)
    ap.add_argument("--spi-scales", default="3,6,12")
    ap.add_argument("--export-csv", action="store_true")
    a, _ = ap.parse_known_args()   # tolère les arguments injectés par Jupyter

    base = Path(a.base_dir)
    dbp = Path(a.db) if Path(a.db).is_absolute() else base / a.db
    if not dbp.exists():
        sys.exit(f"Base introuvable : {dbp}\nLancez d'abord build_drc_climate_db.py")
    conn = sqlite3.connect(dbp)
    n0, n1, t0, t1 = a.normal_start, a.normal_end, a.trend_start, a.trend_end

    log("1/7 Chargement des observations NASA POWER ...")
    df = load_daily(conn)
    if df.empty:
        sys.exit("Table obs_nasa_daily vide : lancez d'abord le téléchargement.")
    log(f"    {len(df):,} lignes, {df.province.nunique()} provinces".replace(",", " "))
    log("2/7 Indices annuels (ETCCDI) ...")
    ann = annual_indices(df, n0, n1)
    log("3/7 Normales et régime pluviométrique ...")
    norm = normals(ann, n0, n1)
    clim = monthly_clim(df, n0, n1)
    regm = regime(clim)
    nat = clim.drop(columns=["period"]).groupby("month").mean(numeric_only=True).reset_index()
    nat.insert(0, "province", "NATIONAL")
    nat["period"] = f"{n0}-{n1}"
    clim = pd.concat([clim, nat], ignore_index=True)
    regm = pd.concat([regm, regime(nat)], ignore_index=True)
    log("4/7 Tendances (Theil-Sen, Mann-Kendall) ...")
    trend = trends(ann, t0, t1, norm)
    log("5/7 SPI ...")
    spi, spia = spi_all(df, n0, n1, [int(x) for x in a.spi_scales.split(",")])
    log("6/7 Scores d'exposition ...")
    haz = hazard_scores(norm, regm, trend)
    log("7/7 Contrôle croisé NASA / ERA5 ...")
    cmp_ = nasa_vs_era5(conn, df)

    for name, t in [("an_annual_indices", ann), ("an_normals", norm),
                    ("an_monthly_climatology", clim), ("an_regime", regm), ("an_trends", trend),
                    ("an_spi", spi), ("an_spi_annual", spia), ("an_hazard_scores", haz),
                    ("an_nasa_vs_era5", cmp_)]:
        if len(t):
            t.to_sql(name, conn, if_exists="replace", index=False)
            log(f"    table {name:24s} {len(t):>8,d} lignes".replace(",", " "))
    conn.execute("INSERT OR REPLACE INTO metadata VALUES('analysis_periods', ?)",
                 (f"normales {n0}-{n1}; tendances {t0}-{t1}",))
    conn.commit()

    if a.export_csv:
        out = base / "drc_data" / "analysis_csv"
        out.mkdir(parents=True, exist_ok=True)
        for name in ("an_annual_indices", "an_normals", "an_monthly_climatology", "an_regime",
                     "an_trends", "an_spi_annual", "an_hazard_scores", "an_nasa_vs_era5"):
            try:
                pd.read_sql(f"SELECT * FROM {name}", conn).to_csv(out / f"{name}.csv", index=False,
                                                                  encoding="utf-8-sig")
            except Exception:
                pass
        log(f"CSV exportés → {out}")

    # ---- Rapport ----
    nn = norm[norm.province == "NATIONAL"].set_index("indicator")["normal"]
    tt = trend[trend.province == "NATIONAL"].set_index("indicator")
    print("\n" + "=" * 72 + f"\nPORTRAIT NATIONAL — normales {n0}-{n1}, tendances {t0}-{t1}\n" + "=" * 72)
    print(f"  Pluie annuelle : {nn['PRCPTOT']:.0f} mm | Température moyenne : {nn['TMEAN']:.1f} °C | "
          f"Tmax moy. : {nn['TXMEAN']:.1f} °C | Tmin moy. : {nn['TNMEAN']:.1f} °C")
    for ind, lab in [("TMEAN", "Température moyenne"), ("TXMEAN", "Tmax moyenne"),
                     ("TNMEAN", "Tmin moyenne"), ("PRCPTOT", "Pluie annuelle"),
                     ("RX1DAY", "Pluie max. en 1 jour"), ("CDD", "Jours secs consécutifs"),
                     ("TX90P", "Jours chauds (TX90p)")]:
        r = tt.loc[ind]
        pc = f" ({r.pct_per_decade:+.1f} %/décennie)" if pd.notna(r.pct_per_decade) else ""
        print(f"  {lab:26s} {r.sen_slope_per_decade:+.3f} {r.unit}/décennie{pc}  [{r.significance}]")
    tp = trend[(trend.indicator == "TMEAN") & (trend.province != "NATIONAL")].sort_values(
        "sen_slope_per_decade", ascending=False)
    print("\n  Réchauffement le plus rapide :", ", ".join(
        f"{r.province} ({r.sen_slope_per_decade:+.2f} °C/déc.)" for r in tp.head(3).itertuples()))
    pp = trend[(trend.indicator == "PRCPTOT") & (trend.province != "NATIONAL")].sort_values("pct_per_decade")
    print("  Plus forte baisse de pluie    :", ", ".join(
        f"{r.province} ({r.pct_per_decade:+.1f} %/déc.)" for r in pp.head(3).itertuples()))
    print("  Plus forte hausse de pluie    :", ", ".join(
        f"{r.province} ({r.pct_per_decade:+.1f} %/déc.)" for r in pp.tail(3).iloc[::-1].itertuples()))
    for k, lab in [("heat", "Chaleur"), ("flood", "Inondation"), ("drought", "Sécheresse")]:
        top = haz.sort_values(f"{k}_score", ascending=False).head(3)
        print(f"  Exposition {lab:11s}:", ", ".join(f"{r.province} ({getattr(r, k + '_score'):.0f})"
                                                    for r in top.itertuples()))
    if len(cmp_):
        print(f"\n  Contrôle ERA5 : rapport pluie ERA5/NASA médian = "
              f"{cmp_.precip_ratio_era5_over_nasa.median():.2f}, biais température médian = "
              f"{cmp_.temp_bias_era5_minus_nasa_c.median():+.2f} °C ({len(cmp_)} provinces)")
    log(f"\nTerminé → {dbp}")
    conn.close()


if __name__ == "__main__":
    main()
