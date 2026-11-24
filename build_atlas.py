#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_atlas.py
==============
ÉTAPE 3 : génère l'ATLAS CLIMATIQUE DE LA RDC (un seul fichier HTML autonome, sans
internet) à partir de la base drc_climate_atlas.db.

Contenu de l'atlas
  1. Portrait national : normales 1991-2020, régime pluviométrique, tendances 1981-2024
     avec un indicateur de FIABILITÉ (robuste / ampleur incertaine / non établie)
  2. Carte des 26 provinces : ~40 couches (normales, tendances, scores d'aléas, projections)
     + fiche détaillée par province
  3. Tendances et robustesse : comparaison NASA POWER / ERA5 / GPCC / TAMSAT / CHIRPS
  4. Projections 2026-2040 : modèle CMIP6 de l'Atlas (CNRM-CM6-1-HR, SSP5-8.5) appliqué
     comme CHANGEMENT sur la référence observée (correction de biais « delta »)
  5. Méthodes, sources et limites

Tables lues si présentes : provinces, an_normals, an_monthly_climatology, an_regime, an_trends,
an_annual_indices, an_spi_annual, an_hazard_scores, an_rain_consensus, an_validation_trends,
obs_rain_indep_monthly, v_nasa_annual, obs_era5_daily, atlas_cmip6_indicators.

USAGE
  python build_atlas.py
  -> D:\\Portail_Climatique_RDC\\ATLAS\\atlas_climatique_rdc.html   (double-clic pour ouvrir)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:\Portail_Climatique_RDC\ATLAS")
SOURCES = ["nasa", "era5", "gpcc", "tamsat", "chirps"]


TEMPLATE = r'''<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Atlas climatique de la RDC</title>
<style>
:root{--bg:#f6f5f0;--card:#fff;--tx:#1d2a24;--mu:#5d6b64;--ln:#dcd9ce;--ac:#1f6f4a;--ok:#1f7a4d;--mid:#b8860b;--bad:#b23a2a}
@media(prefers-color-scheme:dark){:root{--bg:#121815;--card:#1b2420;--tx:#e8eee9;--mu:#9aaa9f;--ln:#2e3a34;--ac:#5cc596;--ok:#5cc596;--mid:#e0b44a;--bad:#ee8a6b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}h1{font-size:1.5rem;margin:.1em 0}h2{font-size:1.1rem;margin:1.1em 0 .4em}h3{font-size:.95rem;margin:.8em 0 .3em}
.sub{color:var(--mu);margin:0 0 10px}.l{color:var(--mu);font-size:.85rem}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}nav button{border:1px solid var(--ln);background:var(--card);color:var(--tx);padding:8px 14px;border-radius:20px;cursor:pointer;font:inherit}
nav button.on{background:var(--ac);color:#fff;border-color:var(--ac)}
section{display:none}section.on{display:block}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.g2{display:grid;grid-template-columns:minmax(300px,1.1fr) minmax(300px,1fr);gap:14px}@media(max-width:820px){.g2{grid-template-columns:1fr}}
.c{background:var(--card);border:1px solid var(--ln);border-radius:12px;padding:12px 14px}
.k{font-size:1.45rem;font-weight:700}
.warn{background:color-mix(in srgb,var(--mid) 14%,var(--card));border:1px solid var(--mid);border-radius:10px;padding:10px 12px;margin:10px 0;font-size:.9rem}
.good{background:color-mix(in srgb,var(--ok) 12%,var(--card));border:1px solid var(--ok);border-radius:10px;padding:10px 12px;margin:10px 0;font-size:.9rem}
.w{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{padding:5px 8px;border-bottom:1px solid var(--ln);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
.b{display:inline-block;padding:1px 9px;border-radius:10px;font-size:.78rem;color:#fff}.b.ok{background:var(--ok)}.b.mid{background:var(--mid)}.b.bad{background:var(--bad)}
select{font:inherit;padding:6px;border-radius:8px;border:1px solid var(--ln);background:var(--card);color:var(--tx);max-width:100%}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0}
#pp td,#pp th{white-space:normal}svg{width:100%;height:auto;display:block}path.pv{stroke:var(--bg);stroke-width:.8;cursor:pointer}path.pv:hover{stroke:var(--tx);stroke-width:1.6}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;background:var(--ln);font-size:.78rem;margin:2px 3px 2px 0}
.lg{display:flex;align-items:center;gap:8px;font-size:.8rem;color:var(--mu);margin-top:6px}.lg i{flex:1;height:10px;border-radius:5px}
</style></head><body><main>
<h1>Atlas climatique de la République démocratique du Congo</h1>
<p class="sub" id="sub"></p>
<nav id="nav"></nav>

<section id="s0"><div id="vbanner"></div><div class="g" id="kpi"></div>
<div class="g2" style="margin-top:12px"><div class="c"><h3>Régime climatique national (1991–2020)</h3><svg id="chClim" viewBox="0 0 640 260"></svg></div>
<div class="c"><h3>Pluie annuelle nationale selon la source</h3><svg id="chRain" viewBox="0 0 640 260"></svg><div class="l" id="rainNote"></div></div></div>
<h2>Ce qui est fiable, ce qui ne l’est pas</h2><div class="g2" id="reliab"></div></section>

<section id="s1"><div class="row"><label>Couche <select id="layer"></select></label></div>
<div class="g2"><div class="c"><svg id="map"></svg><div class="lg"><span id="lmin"></span><i id="lbar"></i><span id="lmax"></span></div><div class="l" id="lnote"></div></div>
<div class="c" id="pp"></div></div></section>

<section id="s2"><h2>Tendances nationales 1981–2024</h2><div class="w"><table id="tt"></table></div>
<h2>Pluie annuelle : tendance selon la source (% par décennie)</h2><div class="w"><table id="tc"></table></div>
<p class="l">Un désaccord entre sources indépendantes signifie que la tendance de pluie n’est pas établie, même si chaque source est « significative ».</p></section>

<section id="s3"><div id="projBody"></div></section>

<section id="s4"><div class="c" id="meth"></div></section>
</main>
<script>
const D=__DATA__;
const $=s=>document.querySelector(s);
const f=(x,d=1)=>(x==null||!isFinite(x))?"–":Number(x).toLocaleString("fr-FR",{minimumFractionDigits:d,maximumFractionDigits:d});
const sg=(x,d=1)=>(x==null||!isFinite(x))?"–":(x>=0?"+":"")+f(x,d);
const MN=["Jan","Fév","Mar","Avr","Mai","Juin","Juil","Août","Sep","Oct","Nov","Déc"];
const SRC={nasa:["NASA POWER","#c0482b"],era5:["ERA5","#7a4bb0"],gpcc:["GPCC","#2a6fb0"],tamsat:["TAMSAT","#1f8a5b"],chirps:["CHIRPS","#b8860b"]};
const LB={PRCPTOT:["Pluie annuelle","mm"],TMEAN:["Température moyenne","°C"],TXMEAN:["Température maximale moyenne","°C"],TNMEAN:["Température minimale moyenne","°C"],RX1DAY:["Pluie max. en 1 jour","mm"],R20MM:["Jours ≥ 20 mm","jours"],CDD:["Jours secs consécutifs","jours"],TXX:["Température max. absolue","°C"],TX90P:["Jours chauds (TX90p)","% jours"],TN90P:["Nuits chaudes (TN90p)","% jours"],SU30:["Jours > 30 °C","jours"],R95P_SHARE:["Part des pluies très fortes","%"],WET_DAYS:["Jours de pluie","jours"],SDII:["Intensité des jours pluvieux","mm/jour"],DTR:["Amplitude thermique journalière","°C"],CWD:["Jours humides consécutifs","jours"],R10MM:["Jours ≥ 10 mm","jours"],R50MM:["Jours ≥ 50 mm","jours"],RX5DAY:["Pluie max. en 5 jours","mm"],TNN:["Température min. absolue","°C"],TXGE35:["Jours ≥ 35 °C","jours"],TR20:["Nuits > 20 °C","jours"],RH_MEAN:["Humidité relative","%"],R95PTOT:["Pluie des jours très humides","mm"]};
const isRain=k=>["PRCPTOT","RX1DAY","RX5DAY","R10MM","R20MM","R50MM","WET_DAYS","SDII","R95PTOT","R95P_SHARE","CDD","CWD"].includes(k);
const PAL={"seq-blue":["#eef5fb","#9cc3e3","#3a7cbf","#12426e"],"seq-warm":["#fff4e0","#f6c273","#e0763a","#8f2a14"],"seq-dry":["#fbf3dc","#e6c46a","#c08a2b","#6f4a0b"],"seq-red":["#fdeee8","#f3a98c","#d4573a","#7d1f12"],"div-temp":["#2a6fb0","#d8e6f2","#f7f3ee","#f1c3ad","#c0482b"],"div-rain":["#9a5b17","#e6c79a","#f7f3ee","#a9cde8","#1f6fb0"]};
const hex=h=>[1,3,5].map(i=>parseInt(h.substr(i,2),16));
function col(pal,t){const s=PAL[pal],n=s.length-1;t=Math.max(0,Math.min(1,t));const i=Math.min(Math.floor(t*n),n-1),u=t*n-i,a=hex(s[i]),b=hex(s[i+1]);return "rgb("+a.map((x,j)=>Math.round(x+(b[j]-x)*u)).join(",")+")"}
const gradient=pal=>"linear-gradient(90deg,"+PAL[pal].join(",")+")";
const PN=c=>c==="NATIONAL"?"National (moyenne des provinces)":(D.provinces.find(p=>p.code===c)||{label:c}).label;

/* ---------- graphiques ---------- */
function lineChart(el,series,o={}){const W=640,H=260,m={l:50,r:12,t:22,b:26};let xs=[],ys=[];series.forEach(s=>s.pts.forEach(p=>{xs.push(p[0]);ys.push(p[1])}));
if(!xs.length){el.innerHTML="";return}let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);if(y1-y0<1e-9){y1+=1}const pd=(y1-y0)*.1;y0-=pd;y1+=pd;
const X=x=>m.l+(x-x0)/((x1-x0)||1)*(W-m.l-m.r),Y=y=>H-m.b-(y-y0)/(y1-y0)*(H-m.t-m.b);let s="";
for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--ln)"/><text x="${m.l-4}" y="${Y(v)+4}" text-anchor="end" font-size="11" fill="var(--mu)">${f(v,o.d??0)}</text>`}
const step=Math.max(1,Math.round((x1-x0)/8));for(let x=Math.ceil(x0);x<=x1;x+=step)s+=`<text x="${X(x)}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--mu)">${x}</text>`;
(o.marks||[]).forEach(k=>{if(k.x>=x0&&k.x<=x1)s+=`<line x1="${X(k.x)}" x2="${X(k.x)}" y1="${m.t}" y2="${H-m.b}" stroke="var(--mu)" stroke-dasharray="4 3"/><text x="${X(k.x)+3}" y="${m.t+10}" font-size="10" fill="var(--mu)">${k.t}</text>`});
series.forEach(se=>{s+=`<polyline fill="none" stroke="${se.c}" stroke-width="${se.w||2}" points="${se.pts.map(p=>X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)).join(" ")}"/>`});
let lx=m.l;series.forEach(se=>{s+=`<rect x="${lx}" y="5" width="10" height="10" fill="${se.c}"/><text x="${lx+14}" y="14" font-size="11" fill="var(--tx)">${se.n}</text>`;lx+=se.n.length*6.4+30});
el.innerHTML=s}
function barLine(el,cl){if(!cl){el.innerHTML="";return}const W=640,H=260,m={l:46,r:46,t:18,b:26},bw=(W-m.l-m.r)/12;const p=cl.map(r=>r[0]),t=cl.map(r=>r[1]);const pm=Math.max(...p)*1.1||1,t0=Math.min(...t)-1,t1=Math.max(...t)+1;
let s="";p.forEach((v,i)=>{const h=(v||0)/pm*(H-m.t-m.b);s+=`<rect x="${m.l+i*bw+4}" y="${H-m.b-h}" width="${bw-8}" height="${h}" fill="#3a7cbf" opacity=".75"><title>${MN[i]} : ${f(v,0)} mm</title></rect><text x="${m.l+i*bw+bw/2}" y="${H-8}" text-anchor="middle" font-size="11" fill="var(--mu)">${MN[i]}</text>`});
const Y=v=>H-m.b-(v-t0)/(t1-t0)*(H-m.t-m.b);s+=`<polyline fill="none" stroke="#c0482b" stroke-width="2.5" points="${t.map((v,i)=>(m.l+i*bw+bw/2)+","+Y(v)).join(" ")}"/>`;
for(let i=0;i<=4;i++){s+=`<text x="${m.l-4}" y="${H-m.b-i/4*(H-m.t-m.b)+4}" text-anchor="end" font-size="11" fill="#3a7cbf">${f(pm*i/4,0)}</text><text x="${W-m.r+4}" y="${Y(t0+(t1-t0)*i/4)+4}" font-size="11" fill="#c0482b">${f(t0+(t1-t0)*i/4,0)}</text>`}
s+=`<text x="${m.l}" y="12" font-size="11" fill="#3a7cbf">Pluie (mm/mois)</text><text x="${W-m.r}" y="12" text-anchor="end" font-size="11" fill="#c0482b">Température (°C)</text>`;el.innerHTML=s}
const rainSeries=c=>Object.entries((D.P[c]||{}).rain||{}).map(([k,v])=>({n:SRC[k][0],c:SRC[k][1],pts:v}));

/* ---------- onglets ---------- */
const TABS=["Portrait national","Carte et provinces","Tendances et fiabilité","Projections 2026–2040","Méthodes"];
TABS.forEach((t,i)=>{const b=document.createElement("button");b.textContent=t;b.onclick=()=>show(i);$("#nav").appendChild(b)});
function show(i){[...$("#nav").children].forEach((b,j)=>b.classList.toggle("on",i==j));for(let j=0;j<5;j++)$("#s"+j).classList.toggle("on",i==j);try{location.hash="t"+i}catch(e){}}
$("#sub").textContent=`Normales ${(D.meta.analysis.match(/normales ([\d-]+)/)||[,"1991-2020"])[1]} · tendances ${(D.meta.analysis.match(/tendances ([\d-]+)/)||[,"1981-2024"])[1]} · sources : ${D.meta.sources.map(s=>SRC[s]?SRC[s][0]:s).join(", ")} · généré le ${D.meta.built}`;

/* ---------- 0 portrait ---------- */
const N=D.P.NATIONAL||{},nn=k=>(N.n&&N.n[k])?N.n[k][0]:null,tt=k=>(N.t&&N.t[k])?N.t[k]:null;
const stMsg={non_etablie:["warn","<b>Tendance de pluie non établie.</b> Les réanalyses (NASA POWER, ERA5) montrent une forte baisse avec un saut brusque vers 1998–2001, que les sources indépendantes ne confirment pas. Les normales de pluie restent utilisables ; la tendance de pluie ne doit pas être publiée comme un fait."],partielle:["warn","<b>Baisse de pluie partiellement corroborée</b> : probablement réelle mais surestimée par les réanalyses."],corroboree:["good","<b>Baisse de pluie corroborée</b> par des sources indépendantes."],inconnue:["warn","<b>Fiabilité de la tendance de pluie non vérifiée</b> (aucune source indépendante en base)."]}[D.rainStatus];
$("#vbanner").innerHTML=`<div class="${stMsg[0]}">${stMsg[1]}${D.verdict?`<div class="l">Verdict automatique : ${D.verdict}</div>`:""}</div>`;
const T=tt("TMEAN"),tc=D.tempconf.TMEAN;
$("#kpi").innerHTML=[["Pluie annuelle",f(nn("PRCPTOT"),0)+" mm","normale"],["Température moyenne",f(nn("TMEAN"))+" °C","normale"],["Tmax moyenne",f(nn("TXMEAN"))+" °C","normale"],["Tmin moyenne",f(nn("TNMEAN"))+" °C","normale"],["Réchauffement",T?sg(T[0],2)+" °C/déc.":"–",tc&&tc.era5!=null?`NASA ${sg(tc.nasa,2)} · ERA5 ${sg(tc.era5,2)}`:(T?T[3]:"")],["Saison sèche",N.reg?f(N.reg.longest_dry_season_months,0)+" mois":"–","mois < 50 mm consécutifs"]].map(a=>`<div class="c"><div class="l">${a[0]}</div><div class="k">${a[1]}</div><div class="l">${a[2]}</div></div>`).join("");
barLine($("#chClim"),N.clim);
lineChart($("#chRain"),rainSeries("NATIONAL"),{marks:[{x:2000,t:"2000"}]});
$("#rainNote").textContent="Moyenne nationale des provinces communes à toutes les sources. Un saut simultané des seules réanalyses, sans équivalent dans les autres sources, évoque un changement du système d’observation plutôt que du climat.";
const rel=[];
rel.push(["good","<b>Fiable</b>",`<ul><li>Cycle saisonnier et ordres de grandeur des normales (à comparer entre sources dans l’onglet Tendances).</li><li>Sens du réchauffement : NASA POWER et ERA5 s’accordent sur une hausse.</li><li>Contrastes régionaux visibles sur la carte (relief de l’Est, saison sèche du Sud).</li></ul>`]);
rel.push(["warn","<b>À nuancer</b>",`<ul><li>Ampleur du réchauffement : ${tc&&tc.era5!=null?`entre ${sg(Math.min(tc.nasa,tc.era5),2)} (ERA5) et ${sg(Math.max(tc.nasa,tc.era5),2)} °C/décennie (NASA)`:"une seule source disponible"}.</li><li>Maximales et minimales : l’écart entre elles diffère selon la source.</li><li>Pluies extrêmes : tendances issues des seules réanalyses.</li></ul>`]);
rel.push(["warn","<b>Non établi</b>",`<ul><li>Tendance de la pluie annuelle 1981–2024 (sources en désaccord).</li><li>Tout classement de provinces « qui s’assèchent le plus ».</li></ul>`]);
$("#reliab").innerHTML=rel.map(r=>`<div class="${r[0]}">${r[1]}${r[2]}</div>`).join("");

/* ---------- 1 carte ---------- */
const geo=D.provinces.filter(p=>p.geom&&p.geom.length);let lon0=1e9,lon1=-1e9,lat0=1e9,lat1=-1e9;
geo.forEach(p=>p.geom.forEach(r=>r.forEach(q=>{lon0=Math.min(lon0,q[0]);lon1=Math.max(lon1,q[0]);lat0=Math.min(lat0,q[1]);lat1=Math.max(lat1,q[1])})));
const kx=Math.cos((lat0+lat1)/2*Math.PI/180),Wm=520,Sc=Wm/((lon1-lon0)*kx||1),Hm=(lat1-lat0)*Sc;
const px=q=>((q[0]-lon0)*kx*Sc).toFixed(1)+","+((lat1-q[1])*Sc).toFixed(1);
const paths={};geo.forEach(p=>paths[p.code]=p.geom.map(r=>"M"+r.map(px).join("L")+"Z").join(""));
$("#map").setAttribute("viewBox",`0 0 ${Wm} ${Hm||400}`);
const groups={};D.layers.forEach(l=>(groups[l.group]=groups[l.group]||[]).push(l));
$("#layer").innerHTML=Object.entries(groups).map(([g,ls])=>`<optgroup label="${g}">${ls.map(l=>`<option value="${l.id}">${l.label}</option>`).join("")}</optgroup>`).join("");
let selP="NATIONAL";
function drawMap(){const l=D.layers.find(x=>x.id===$("#layer").value);if(!l)return;const v=Object.values(l.vals).filter(x=>x!=null);let mn=Math.min(...v),mx=Math.max(...v);const dv=l.pal.startsWith("div");if(dv){const m=Math.max(Math.abs(mn),Math.abs(mx))||1;mn=-m;mx=m}
$("#map").innerHTML=geo.map(p=>{const x=l.vals[p.code];const t=x==null?null:(x-mn)/((mx-mn)||1);return `<path class="pv" data-c="${p.code}" d="${paths[p.code]}" fill="${t==null?"#bbb":col(l.pal,t)}"${p.code===selP?' style="stroke:var(--tx);stroke-width:2.2"':""}><title>${p.label} : ${f(x,l.dec)} ${l.unit}</title></path>`}).join("");
$("#lbar").style.background=gradient(l.pal);$("#lmin").textContent=f(mn,l.dec);$("#lmax").textContent=f(mx,l.dec)+" "+l.unit;$("#lnote").textContent=l.note||""}
$("#map").onclick=e=>{const c=e.target.dataset&&e.target.dataset.c;if(c){selP=c;drawMap();panel(c)}};$("#layer").onchange=drawMap;
const conf=(k,p)=>{if(isRain(k))return D.rainStatus==="corroboree"?["mid","à confirmer"]:["bad","non établie"];const a=D.tempconf[k];return a&&a.era5!=null&&Math.sign(a.nasa)===Math.sign(a.era5)?["mid","ampleur incertaine"]:["mid","source unique"]};
function panel(c){const P=D.P[c]||{},n=k=>P.n&&P.n[k]?P.n[k][0]:null;
const rows=["PRCPTOT","TMEAN","TXMEAN","TNMEAN","RX1DAY","CDD","TXX","R20MM"].map(k=>`<tr><td>${LB[k][0]}</td><td>${f(n(k),k==="PRCPTOT"||k==="CDD"?0:1)} ${LB[k][1]}</td></tr>`).join("");
const trs=["TMEAN","TXMEAN","PRCPTOT","RX1DAY","CDD"].filter(k=>P.t&&P.t[k]).map(k=>{const r=P.t[k],b=conf(k);return `<tr><td>${LB[k][0]}</td><td>${sg(r[0],k==="PRCPTOT"?0:2)} ${r[6]}/déc.</td><td>${r[3]}</td><td><span class="b ${b[0]}">${b[1]}</span></td></tr>`}).join("");
const h=P.hz?`<div>${[["Chaleur",P.hz.heat_score],["Inondation",P.hz.flood_score],["Sécheresse",P.hz.drought_score]].map(a=>`<span class="pill">${a[0]} ${f(a[1],0)}/100</span>`).join("")}</div>`:"";
$("#pp").innerHTML=`<h3 style="margin-top:0">${PN(c)}</h3><div class="l">${(D.provinces.find(p=>p.code===c)||{}).zone||""}${P.reg?` · saison sèche ${f(P.reg.longest_dry_season_months,0)} mois · mois le plus pluvieux : ${MN[(P.reg.wettest_month||1)-1]}`:""}</div>${h}
<div class="w"><table><tr><td>Normale 1991–2020</td><td></td></tr>${rows}</table></div><svg id="pc" viewBox="0 0 640 260"></svg><svg id="pr" viewBox="0 0 640 260"></svg>
<h3>Tendances 1981–2024</h3><div class="w"><table><tr><th>Indicateur</th><th>Tendance</th><th>Signif.</th><th>Fiabilité</th></tr>${trs}</table></div>${P.spi_dry_years!=null?`<div class="l">Années de sécheresse (SPI-12 ≤ −1) : ${P.spi_dry_years}</div>`:""}`;
barLine($("#pc"),P.clim);lineChart($("#pr"),rainSeries(c),{marks:[{x:2000,t:"2000"}]})}
drawMap();panel("NATIONAL");

/* ---------- 2 tendances ---------- */
const ord=["TMEAN","TXMEAN","TNMEAN","TXX","TX90P","TN90P","SU30","PRCPTOT","RX1DAY","RX5DAY","R20MM","CDD","SDII"];
const nt=D.natTrends.slice().sort((a,b)=>ord.indexOf(a.ind)-ord.indexOf(b.ind)).filter(r=>ord.includes(r.ind));
$("#tt").innerHTML="<tr><th>Indicateur</th><th>NASA POWER /déc.</th><th>Signif.</th><th>ERA5 /déc.</th><th>Fiabilité</th></tr>"+nt.map(r=>{const b=conf(r.ind),a=r.alt;return `<tr><td>${(LB[r.ind]||[r.ind])[0]}</td><td>${sg(r.slope,2)} ${r.unit}${r.pct!=null?` (${sg(r.pct,1)} %)`:""}</td><td>${r.sig}</td><td>${a&&a.era5!=null?sg(a.era5,2):"–"}</td><td><span class="b ${b[0]}">${b[1]}</span></td></tr>`}).join("");
const srcs=Object.keys(SRC).filter(s=>D.consensus.length&&D.consensus[0][s+"_pct_dec"]!=null);
$("#tc").innerHTML=D.consensus.length?"<tr><th>Province</th>"+srcs.map(s=>`<th>${SRC[s][0]}</th>`).join("")+"<th>Consensus</th></tr>"+D.consensus.map(r=>`<tr><td>${PN(r.province)}</td>${srcs.map(s=>`<td>${sg(r[s+"_pct_dec"],1)}</td>`).join("")}<td>${sg(r.consensus_pct_dec,1)}</td></tr>`).join(""):"<tr><td>Comparaison multi-sources non disponible : lancez fetch_rain_independent.py</td></tr>";

/* ---------- 3 projections ---------- */
function projView(){const PR=D.proj,B=$("#projBody");if(!PR){B.innerHTML=`<div class="warn">Projections indisponibles : ${D.projWhy}.</div>`;return}
B.innerHTML=`<div class="warn"><b>Lecture prudente.</b> Un seul modèle (CNRM-CM6-1-HR) et le scénario SSP5-8.5, très émetteur : à lire comme un haut de fourchette. Les valeurs sont une interpolation entre périodes de 30 à 50 ans, sans variabilité d’une année à l’autre. Température et pluie sont exprimées comme <b>changement appliqué à la référence observée</b> (correction de biais « delta »).</div>
<div class="row"><label>Indicateur <select id="pi"></select></label><label>Territoire <select id="pt"></select></label></div><div class="c"><svg id="pch" viewBox="0 0 640 260"></svg></div><div class="g" id="pk" style="margin-top:10px"></div><h2>Provinces : évolution 2026 → 2040</h2><div class="w"><table id="ptb"></table></div>`;
PR.cols.forEach(m=>$("#pi").add(new Option(m.label,m.id)));$("#pt").add(new Option("National (moyenne)","NATIONAL"));D.provinces.forEach(p=>{if(PR.data[p.code])$("#pt").add(new Option(p.label,p.code))});
const series=(id,c)=>{if(c!=="NATIONAL")return (PR.data[c]||{})[id];const L=Object.values(PR.data).map(d=>d[id]).filter(Boolean);return L.length?PR.years.map((_,i)=>L.reduce((s,a)=>s+a[i],0)/L.length):null};
function draw(){const id=$("#pi").value,c=$("#pt").value,m=PR.cols.find(x=>x.id===id),v=series(id,c);if(!v)return;
lineChart($("#pch"),[{n:m.label+" — "+PN(c),c:"#c0482b",pts:v.map((y,i)=>[PR.years[i],y])}],{d:m.dec});
const hist=c==="NATIONAL"?Object.values(PR.base).map(b=>b[id]).filter(x=>x!=null).reduce((s,x,_,a)=>s+x/a.length,0):(PR.base[c]||{})[id];
$("#pk").innerHTML=[["Référence",f(hist,m.dec),m.kind==="raw"?"modèle 1985–2014":"observée"],["2026",f(v[0],m.dec),sg(v[0]-hist,m.dec)],["2033",f(v[7],m.dec),sg(v[7]-hist,m.dec)],["2040",f(v[14],m.dec),sg(v[14]-hist,m.dec)+" · "+sg(v[14]-v[0],m.dec)+" depuis 2026"]].map(a=>`<div class="c"><div class="l">${a[0]}</div><div class="k">${a[1]}</div><div class="l">${a[2]}</div></div>`).join("");
const rows=Object.keys(PR.data).filter(k=>(PR.data[k][id])).map(k=>{const s=PR.data[k][id];return {k,a:s[0],z:s[14]}}).sort((x,y)=>(y.z-y.a)-(x.z-x.a));
$("#ptb").innerHTML=`<tr><th>Province</th><th>2026</th><th>2040</th><th>Évolution</th></tr>`+rows.map(r=>`<tr><td>${PN(r.k)}</td><td>${f(r.a,m.dec)}</td><td>${f(r.z,m.dec)}</td><td>${sg(r.z-r.a,m.dec)}</td></tr>`).join("")}
$("#pi").onchange=$("#pt").onchange=draw;draw()}
projView();

/* ---------- 4 méthodes ---------- */
const cn=D.meta.counts;
$("#meth").innerHTML=`<h2 style="margin-top:0">Sources</h2><ul><li><b>NASA POWER</b> (quotidien, 1981–2025, ${f(cn.obs_nasa_daily||0,0)} lignes) : base des normales et tendances.</li><li><b>ERA5</b> (réanalyse) : contrôle croisé.</li><li><b>GPCC, TAMSAT, CHIRPS</b> : pluie indépendante des réanalyses (pluviomètres / satellite).</li><li><b>CMIP6</b> : CNRM-CM6-1-HR, SSP5-8.5 (Atlas d’origine, idiouf).</li><li><b>Limites provinciales</b> : GADM 4.1.</li></ul>
<h2>Méthodes</h2><ul><li>Indices ETCCDI ; tendances de Theil-Sen avec test de Mann-Kendall (sans correction d’autocorrélation) ; percentiles sur 1991–2020.</li><li>Ruptures : test de Pettitt. Cohérence entre sources évaluée sur une période commune.</li><li>Scores d’exposition : comparaison relative entre provinces (70 % niveau, 30 % tendance).</li><li>« National » = moyenne simple des provinces (non pondérée par la surface ou la population).</li></ul>
<h2>Limites</h2><ul><li>Aucun jeu n’est une vérité de terrain : le réseau de stations est très clairsemé en RDC.</li><li>La tendance de la pluie n’est pas établie ; ne pas la citer comme un résultat.</li><li>Les projections sont indicatives (modèle et scénario uniques).</li></ul>`;
show(+(location.hash.slice(2))||0);
</script></body></html>
'''


def log(m):
    print(m, flush=True)


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


# ---------------------------------------------------------------------------
# Projections : CMIP6 de l'Atlas, changement appliqué à la référence observée
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
def build_data(conn):
    prov = tbl(conn, "provinces")
    if prov.empty:
        sys.exit("Table provinces vide : lancez build_drc_climate_db.py")
    nm, cl, rg = tbl(conn, "an_normals"), tbl(conn, "an_monthly_climatology"), tbl(conn, "an_regime")
    tr, ai, spi = tbl(conn, "an_trends"), tbl(conn, "an_annual_indices"), tbl(conn, "an_spi_annual")
    hz, cons, vt = tbl(conn, "an_hazard_scores"), tbl(conn, "an_rain_consensus"), tbl(conn, "an_validation_trends")
    if nm.empty or tr.empty:
        sys.exit("Tables d'analyse absentes : lancez analyze_drc_climate.py")
    codes = list(prov.code) + ["NATIONAL"]
    P = {c: {} for c in codes}

    for r in nm.itertuples():
        if r.province in P:
            P[r.province].setdefault("n", {})[r.indicator] = [num(r.normal, 2), num(r.std, 2)]
    for c, g in cl.groupby("province"):
        if c in P:
            g = g.sort_values("month")
            P[c]["clim"] = [[num(a, 1) for a in (r.precip_mm, r.t2m_c, r.tmax_c, r.tmin_c)]
                            for r in g.itertuples()]
    for r in rg.itertuples():
        if r.province in P:
            P[r.province]["reg"] = {k: num(getattr(r, k), 2) for k in (
                "annual_mm", "wettest_month", "driest_month", "dry_months_lt50mm",
                "longest_dry_season_months", "annual_temp_c", "temp_amplitude_c")}
    for r in tr.itertuples():
        if r.province in P:
            P[r.province].setdefault("t", {})[r.indicator] = [
                num(r.sen_slope_per_decade, 3), num(r.pct_per_decade, 2), num(r.mk_p, 4),
                r.significance, num(r.ci95_low_per_decade, 3), num(r.ci95_high_per_decade, 3), r.unit]
    keep = ["PRCPTOT", "TMEAN", "TXMEAN", "TNMEAN", "RX1DAY", "CDD"]
    for c, g in ai.groupby("province"):
        if c in P:
            P[c]["ser"] = {k: [[int(y), num(v, 2)] for y, v in zip(g.year, g[k]) if pd.notna(v)]
                           for k in keep if k in g}
    for r in hz.itertuples():
        P[r.province]["hz"] = {k: num(getattr(r, k), 1) for k in
                               ("heat_score", "flood_score", "drought_score", "composite")}
    if len(spi):
        for c, g in spi.groupby("province"):
            if c in P:
                P[c]["spi_dry_years"] = int((g.spi12_dec <= -1).sum())

    # pluie annuelle par source
    ser = {}
    na = tbl(conn, "v_nasa_annual")
    if len(na):
        ser["nasa"] = na.rename(columns={"prcptot_mm": "p"})[["province", "year", "p"]]
    try:
        era = pd.read_sql("""SELECT province,year,SUM(precip_mm) p,COUNT(precip_mm) n FROM obs_era5_daily
                             GROUP BY province,year HAVING n>=330""", conn)
        if len(era):
            ser["era5"] = era[["province", "year", "p"]]
    except Exception:
        pass
    ind = tbl(conn, "obs_rain_indep_monthly")
    for s, g in ind.groupby("source") if len(ind) else []:
        c = g.groupby(["province", "year"]).precip_mm.agg(["sum", "count"]).reset_index()
        ser[s] = c[c["count"] == 12].rename(columns={"sum": "p"})[["province", "year", "p"]]
    for s, df in ser.items():
        for c, g in df.groupby("province"):
            if c in P:
                P[c].setdefault("rain", {})[s] = [[int(y), num(v, 0)] for y, v in zip(g.year, g.p)]
    common = sorted(set.intersection(*[set(d.province) for d in ser.values()])) if ser else []
    for s, df in ser.items():
        nat = df[df.province.isin(common)].groupby("year").p.mean()
        P["NATIONAL"].setdefault("rain", {})[s] = [[int(y), num(v, 0)] for y, v in nat.items()]

    # fiabilité
    verdict = cons.verdict_global.iloc[0] if len(cons) else None
    v = (verdict or "").upper()
    rain_status = ("non_etablie" if ("NON" in v or "INCONCLUANT" in v) else
                   "partielle" if "PARTIELLEMENT" in v else "corroboree" if "CORROB" in v else "inconnue")
    vtn = vt[(vt.province == "NATIONAL")] if len(vt) else vt
    tempconf = {}
    for r in vtn.itertuples():
        tempconf[r.indicator] = dict(
            nasa=num(r.sen_slope_per_decade_nasa, 3), era5=num(r.sen_slope_per_decade_era5, 3),
            verdict=r.verdict)

    # couches cartographiques
    layers = []

    def add(id_, group, label, unit, vals, pal, dec=1, note=""):
        vals = {k: num(x, 4) for k, x in vals.items() if k != "NATIONAL"}
        if any(x is not None for x in vals.values()):
            layers.append(dict(id=id_, group=group, label=label, unit=unit, vals=vals, pal=pal,
                               dec=dec, note=note))
    NL = {"PRCPTOT": ("Pluie annuelle", "mm", "seq-blue", 0), "TMEAN": ("Température moyenne", "°C", "seq-warm", 1),
          "TXMEAN": ("Température maximale moyenne", "°C", "seq-warm", 1),
          "TNMEAN": ("Température minimale moyenne", "°C", "seq-warm", 1),
          "RX1DAY": ("Pluie maximale en 1 jour", "mm", "seq-blue", 0), "R20MM": ("Jours de pluie ≥ 20 mm", "jours", "seq-blue", 1),
          "CDD": ("Jours secs consécutifs (max annuel)", "jours", "seq-dry", 0), "TXX": ("Température max. absolue", "°C", "seq-warm", 1),
          "SU30": ("Jours > 30 °C", "jours", "seq-warm", 0), "RH_MEAN": ("Humidité relative", "%", "seq-blue", 0),
          "SDII": ("Intensité des jours pluvieux", "mm/jour", "seq-blue", 1)}
    for k, (lab, u, pal, d) in NL.items():
        add("n_" + k, "Normales 1991-2020", lab, u, {c: P[c].get("n", {}).get(k, [None])[0] for c in codes}, pal, d)
    TL = {"TMEAN": ("Tendance température moyenne", "°C/décennie", "div-temp", 2),
          "TXMEAN": ("Tendance température maximale", "°C/décennie", "div-temp", 2),
          "TNMEAN": ("Tendance température minimale", "°C/décennie", "div-temp", 2),
          "CDD": ("Tendance jours secs consécutifs", "jours/décennie", "div-temp", 1)}
    for k, (lab, u, pal, d) in TL.items():
        add("t_" + k, "Tendances 1981-2024", lab, u, {c: P[c].get("t", {}).get(k, [None])[0] for c in codes}, pal, d)
    add("t_RX1", "Tendances 1981-2024", "Tendance pluie max. 1 jour (NASA) ⚠", "%/décennie",
        {c: P[c].get("t", {}).get("RX1DAY", [None, None])[1] for c in codes}, "div-rain", 1,
        "Basée sur NASA POWER seul : voir onglet Tendances pour la fiabilité.")
    if len(cons):
        for s in SOURCES:
            col = f"{s}_pct_dec"
            if col in cons:
                add("r_" + s, "Tendance pluie annuelle, par source", f"Pluie annuelle : {s.upper()}", "%/décennie",
                    dict(zip(cons.province, cons[col])), "div-rain", 1,
                    "Les sources ne s'accordent pas : cette couche montre le désaccord.")
    else:
        add("r_nasa", "Tendance pluie annuelle, par source", "Pluie annuelle : NASA POWER ⚠", "%/décennie",
            {c: P[c].get("t", {}).get("PRCPTOT", [None, None])[1] for c in codes}, "div-rain", 1,
            "Non vérifiée par une source indépendante.")
    for k, lab in [("heat_score", "Exposition à la chaleur"), ("flood_score", "Exposition aux inondations"),
                   ("drought_score", "Exposition à la sécheresse"), ("composite", "Exposition composite")]:
        add("h_" + k, "Scores d'exposition (0-100, relatifs)", lab, "score", {c: P[c].get("hz", {}).get(k) for c in codes},
            "seq-red", 0, "Comparaison entre provinces, pas un risque absolu.")

    # projections
    nbase = {(c, "TMEAN"): P[c]["n"]["TMEAN"][0] for c in codes if "TMEAN" in P[c].get("n", {})}
    rb = {}
    for c in prov.code:
        vals = [P[c]["n"]["PRCPTOT"][0]] if "PRCPTOT" in P[c].get("n", {}) else []
        for s in ("gpcc", "tamsat"):
            sr = [v for y, v in P[c].get("rain", {}).get(s, []) if 1991 <= y <= 2020 and v]
            if sr:
                vals.append(float(np.mean(sr)))
        if vals:
            rb[c] = float(np.median(vals))
    proj, why = projections(conn, prov, nbase, rb)
    if proj:
        for m in proj["cols"]:
            if m["kind"] in ("temp", "rain"):
                dd = {c: (proj["data"][c][m["id"]][-1] - proj["data"][c][m["id"]][0]) / (
                    1 if m["kind"] == "temp" else proj["data"][c][m["id"]][0] / 100)
                      for c in proj["data"] if m["id"] in proj["data"][c]}
                add("p_" + m["id"], "Projection 2026 → 2040", ("Hausse de température" if m["kind"] == "temp"
                    else "Variation de la pluie annuelle"), "°C" if m["kind"] == "temp" else "%", dd,
                    "seq-warm" if m["kind"] == "temp" else "div-rain", 2 if m["kind"] == "temp" else 1,
                    "Modèle CMIP6 unique, scénario SSP5-8.5 (haut de fourchette).")
    log(f"  projections : {why}")

    nt = []
    for k, r in P["NATIONAL"].get("t", {}).items():
        nt.append(dict(ind=k, slope=r[0], pct=r[1], p=r[2], sig=r[3], lo=r[4], hi=r[5], unit=r[6],
                       alt=tempconf.get(k)))
    meta = tbl(conn, "metadata")
    md = dict(zip(meta.key, meta.value)) if len(meta) else {}
    counts = {}
    for t in ("obs_nasa_daily", "obs_era5_daily", "cmip6_daily", "obs_rain_indep_monthly"):
        try:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception:
            pass
    return dict(
        provinces=[dict(code=r.code, label=r.label, zone=r.zone, geom=geometry(r.geometry_geojson))
                   for r in prov.itertuples()],
        P=P, layers=layers, natTrends=nt, tempconf=tempconf, verdict=verdict, rainStatus=rain_status,
        proj=proj, projWhy=why, consensus=cons.round(2).to_dict("records") if len(cons) else [],
        meta=dict(built=datetime.now().strftime("%d/%m/%Y %H:%M"), analysis=md.get("analysis_periods", ""),
                  counts=counts, sources=sorted(ser)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", default=str(BASE_DIR))
    ap.add_argument("--db", default="drc_climate_atlas.db")
    ap.add_argument("--out", default="atlas_climatique_rdc.html")
    a, _ = ap.parse_known_args()
    base = Path(a.base_dir)
    dbp = Path(a.db) if Path(a.db).is_absolute() else base / a.db
    if not dbp.exists():
        sys.exit(f"Base introuvable : {dbp}")
    conn = sqlite3.connect(dbp)
    log("Lecture de la base ...")
    D = build_data(conn)
    conn.close()
    html_tpl = TEMPLATE
    for d in ([Path(__file__).resolve().parent] if "__file__" in globals() else []) + [Path.cwd(), base]:
        if (d / "atlas_template.html").exists():          # modèle modifié par vos soins : prioritaire
            html_tpl = (d / "atlas_template.html").read_text(encoding="utf-8")
            log(f"  modèle externe utilisé : {d / 'atlas_template.html'}")
            break
    html = html_tpl.replace(
        "__DATA__", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
    out = Path(a.out) if Path(a.out).is_absolute() else base / a.out
    out.write_text(html, encoding="utf-8")
    log(f"Atlas généré : {out}  ({out.stat().st_size / 1e6:.1f} Mo, {len(D['layers'])} couches)")


if __name__ == "__main__":
    main()
