r"""La lambda dels TIRS REALS: el metode dels llindars i el metode dels veins.

Pla i criteris: documentacio/pla-analisi-caos.md, «17/10 · La lambda dels tirs reals»,
escrits ABANS d'aquest codi. Els dos metodes es diuen «dels llindars» i «dels veins»;
el tercer cami, la comparacio amb el model, NO es fa aqui: demana la firma de X, Y i Z.

LA DISTANCIA. Nomes els angles, d = sqrt(dth1^2 + dth2^2) amb dif_angular. Al repos
omega = 0, o sigui que la d0 dels angles de repos es la de la norma sencera.

EL METODE DELS LLINDARS. Per a cada parella de tirs, alineats a l'amollada, el temps
T(D) que triguen a separar-se una distancia D gran: lambda = ln(D/d0)/T(D). Amb
diversos D, una separacio exponencial dona una lambda(D) PLANA, i una de lineal (el
pendol simple) una que CAU. La d0 es la dels angles de repos, on el soroll es petit.

EL METODE DELS VEINS. Parelles d'estats de QUALSEVOL tir a QUALSEVOL instant, triades
a una separacio inicial per sobre del soroll, que se segueixen unes decimes de segon.
No alinea res: l'error de l'amollada no hi entra.

QUE MIRA DE LES PRESES REALS, I QUAN
  descriu   el pas 0: les preses, SENSE cap corba de separacio. Llegeix l'index, el CSV
            i el JSON de cada presa (dades.carrega) i el .trk (per al soroll, amb les
            funcions de eines/soroll_en_moviment.py).
  valida    el pas 1, NOMES SIMULACIO. De dades/ nomes obre el JSON del pas 0 (numeros
            del soroll, de l'amollada i dels angles de repos: cap trajectoria real), els
            dos JSON de soroll i el de regla2_amb_soroll (la lambda de Benettin, info).
  reals     el pas 3. TANCAT AMB PANY: es nega a funcionar sense
            dades/divergencia/lyapunov_real_aprovat.json, que nomes s'escriu quan en
            Guillem aprova grups i metodes.
  horitzo   (23/10) l'horitzo de prediccio t = ln(D_tol/D0)/lambda amb la lambda real del
            R1, a cada remostreig del bootstrap. Darrere del mateix pany. Real sol: no
            simula res (ho comprova) i no es compara amb el feix.
  corregit  (24/10) el R2 i el R1 amb el repos final de cada presa a (0, 0): la marca de
            l'articulacio desplacada de costat, pel cami de la conversio i sense tocar ni
            el .trk ni el CSV. Darrere del mateix pany. Els numeros publicats no canvien.
            MODEL DESCARTAT el 24/10 (l'eix de gir de debo es per sobre del pivot de la
            conversio; documentacio/pla-correccio-biaix.md): els seus numeros no valen.
  dues_parts (25/10) el R2 i el R1 amb la correccio que es firmara, comparar.corregeix_presa
            (l'eix de gir del cercle de cada presa i el repos final; rho del regle). PER
            INFORMACIO: als 150 graus la correccio no esta validada. Darrere del mateix pany.
            Puja d'esborranys/_contra_corregeix_presa.py --regle amb la sortida identica (DQ1).
  z_r2      (26/10) LA Z amb la lambda* dels 150 graus: la del JSON real (no obre cap presa) contra el R2 simulat des de
            les mateixes condicions mesurades, amb el mateix soroll, alineacio i metode. TANCAT AMB EL PANY DE LA FIRMA
            (firma.json amb X, Y i Z), que es mira abans d'obrir res. --sigma-ci-graus 0,0 es l'opcio (a); > 0, la (b):
            els tirs surten de les mesurades mes N(0, s). --autoprova-z: ZR1-ZR5, nomes simulacio.
  z_diagnostic (27/10 al vespre) ELS DIAGNOSTICS DE LA Z FIRMADA, fets despres del resultat i SENSE CAP VEREDICTE, sense
            friccio: D4 (la s), D4b (les condicions de debo amb la contraccio d'empirical Bayes), D2 (sense la correccio),
            D3 (el muntatge) i D1 (la lambda(D)). Darrere del pany de la firma; no obre cap presa real (DG1), i la
            referencia es bit a bit el costat simulat de la Z (DG0).
  z_bessons (29/10) l'error de cada besso de la Z (els 20 de la ZR3', la C3) al costat del de la presa real (12 %), i
            en quin percentil cau. NOMES SIMULACIO, mes el JSON de la Z: cap presa, cap veredicte. ZB1 i ZB2.

Us:
    .venv\\Scripts\\python eines\\lyapunov_real.py --autoprova
    .venv\\Scripts\\python eines\\lyapunov_real.py descriu --json dades/divergencia/lyapunov_real_pas0_2026-10-17.json
    .venv\\Scripts\\python eines\\lyapunov_real.py valida --pas0 dades/divergencia/lyapunov_real_pas0_2026-10-17.json --json dades/divergencia/lyapunov_real_validacio_2026-10-17.json
    .venv\\Scripts\\python eines\\lyapunov_real.py horitzo --pas02 dades/divergencia/lyapunov_real_pas0b_2026-10-17.json --reals dades/divergencia/lyapunov_real_reals_2026-10-17.json --json dades/divergencia/lyapunov_real_horitzo_2026-10-23.json
    .venv\\Scripts\\python eines\\lyapunov_real.py corregit --pas0 dades/divergencia/lyapunov_real_pas0_2026-10-17.json --pas02 dades/divergencia/lyapunov_real_pas0b_2026-10-17.json --reals dades/divergencia/lyapunov_real_reals_2026-10-17.json --json dades/divergencia/lyapunov_real_corregit_2026-10-24.json
    .venv\\Scripts\\python eines\\lyapunov_real.py dues_parts --pas0 dades/divergencia/lyapunov_real_pas0_2026-10-17.json --pas02 dades/divergencia/lyapunov_real_pas0b_2026-10-17.json --reals dades/divergencia/lyapunov_real_reals_2026-10-24.json --json dades/divergencia/lyapunov_real_dues_parts_2026-10-25.json
    .venv\\Scripts\\python eines\\lyapunov_real.py --autoprova-z
    .venv\\Scripts\\python eines\\lyapunov_real.py z_r2 --pas0 dades/divergencia/lyapunov_real_pas0_2026-10-17.json --pas02 dades/divergencia/lyapunov_real_pas0b_2026-10-17.json --reals dades/divergencia/lyapunov_real_reals_2026-10-24.json --sigma-ci-graus 1.45,1.53 --json dades/divergencia/lyapunov_real_z_r2_<data>.json
    .venv\\Scripts\\python eines\\lyapunov_real.py z_diagnostic --processos 2 --json dades/divergencia/lyapunov_real_z_diagnostic_2026-10-27.json
    .venv\\Scripts\\python eines\\lyapunov_real.py z_bessons --processos 2 --json dades/divergencia/lyapunov_real_z_bessons_2026-10-29.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol import caos, dades  # noqa: E402
from pendol.fisica import Dissipacio, Friccio, Parametres  # noqa: E402
from pendol.mesures import dif_angular, estats_de_taula, omega_zero  # noqa: E402
import regla2_amb_soroll as rg  # noqa: E402

NET = ARREL / "dades" / "net"
INDEX = NET / "preses.json"
DIVERGENCIA = ARREL / "dades" / "divergencia"
APROVAT = DIVERGENCIA / "lyapunov_real_aprovat.json"
REGLA2_JSON = DIVERGENCIA / "regla2_amb_soroll_2026-10-03.json"

# ------------------------------------------------------------- les tries
# Totes surten del pla del 17/10. Si en canvia alguna, es diu al pla amb la data.

GRUPS = ("45", "90", "150")
FPS = 240.0
DT = 1.0 / FPS
T_MAX_S = 3.0                 # mes enlla, una parella que no ha creuat es censurada
DURADA_SIM_S = 6.0
REPOS_S = 0.5                 # repos davant de cada tir simulat, perque el filtre no tingui vora
N_D = 5                       # valors de D, repartits logaritmicament
FACTORS_TERRA = (3.0, 2.0)    # D_min = 3 x terra; si no queden 1,2 e de rang, 2 x terra
RANG_MINIM_LN = 1.2           # corregit el 17/10 abans d'executar (abans 1): vegeu el pla
FRACCIO_CREUEN = 0.75
RATIO_EXP, RATIO_LIN = 0.7, 0.5
SIGMA_AMOLLADA_MAX_S = 0.020  # fora dels llindars: 150_4, 150_12, 45_10
TRAM_S = (0.1, 1.0)           # el soroll i la |omega| del primer segon
SOSTRE_FINESTRA_S = (0.5, 2.0)
SOSTRE_MOSTRES = 200
REMOSTREJOS = 1000
REALITZACIONS = 5
LLAVOR = 20261017
PROCESSOS = 6
DELTA = 1e-8                  # les pertorbacions infinitesimals de les referencies

# Els veins
BANDA_D0 = (3.0, 4.5)         # x el terra de seguiment
FRANGES_E = {"45": (0.8, 1.03), "90": (1.7, math.inf), "150": (1.7, math.inf)}
DE_MAX = 0.1                  # |dE|/E_v2
THEILER_S = 2.0
CADA_REF = 10
VEINS_DES_DE_S = 0.1
VEINS_FINS_S = 1.5
CREIXEMENT_MINIM = 1.0
VEINS_REF_MAX = 400           # parelles amb referencia infinitesimal, per grup

# L'alineacio relativa
AL_FINESTRA_S = 0.08
AL_PAS_S = 1e-4
AL_ITERACIONS = 2

COTES = {"LL1": 0.20, "LL2": 0.25, "VN1": 0.20, "VN2": 0.25, "boot": 0.20, "VN4": 0.3,
         "AL1_s": 0.002, "AL2_s": 0.0005}


# ------------------------------------------------------------- la distancia

def dist_ang(a, b):
    """sqrt(dth1^2 + dth2^2) amb les diferencies plegades. a, b: (>=2, ...)."""
    return np.hypot(dif_angular(a[0], b[0]), dif_angular(a[1], b[1]))


def _parametres():
    return Parametres.des_de_json(ARREL / "parametres.json")


# =============================================================== els llindars

def creuaments(t, d, Ds):
    """T (P, nD): el primer instant amb d >= D, interpolant ln d. nan si no creua.

    Si ja hi es a la primera mostra, T = t[0].
    """
    t = np.asarray(t, dtype=float)
    d = np.atleast_2d(np.asarray(d, dtype=float))
    lnd = np.log(np.maximum(d, 1e-300))
    T = np.full((d.shape[0], len(Ds)), np.nan)
    for q, D in enumerate(Ds):
        sobre = d >= D
        alguna = sobre.any(axis=1)
        k = np.argmax(sobre, axis=1)
        lD = math.log(D)
        for p in np.nonzero(alguna)[0]:
            kk = int(k[p])
            if kk == 0:
                T[p, q] = t[0]
                continue
            a, b = lnd[p, kk - 1], lnd[p, kk]
            f = (lD - a) / (b - a) if b != a else 1.0
            T[p, q] = t[kk - 1] + f * (t[kk] - t[kk - 1])
    return T


def lambdes_parelles(T, d0, Ds, t_max):
    """(lam, fita, censurada): lam = ln(D/d0)/T, i per a les censurades ln(D/d0)/t_max."""
    lnR = np.log(np.asarray(Ds)[None, :] / np.asarray(d0)[:, None])
    cens = ~np.isfinite(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(cens, np.nan, np.where(T > 0, lnR / np.where(T > 0, T, 1.0), np.inf))
    return lam, lnR / t_max, cens


def agrega(lam, fita, cens, idx=None):
    """La mediana de les lambdes de les parelles (les censurades, amb la seva fita) i
    la fraccio que creua, per a cada D."""
    v = np.where(cens, fita, lam)
    c = cens
    if idx is not None:
        v, c = v[idx], c[idx]
    return np.median(v, axis=0), 1.0 - c.mean(axis=0)


def veredicte_llindars(lamD, frac, Ds) -> dict:
    """Exponencial, lineal, no es pot distingir, o no creua. Regles del pla del 17/10."""
    lamD, frac, Ds = np.asarray(lamD), np.asarray(frac), np.asarray(Ds)
    if frac[0] < FRACCIO_CREUEN:
        return {"veredicte": "no creua", "ratio": None, "j_max": None, "rang_ln": None}
    j = int(max(k for k in range(len(Ds)) if frac[k] >= FRACCIO_CREUEN))
    rang = float(math.log(Ds[j] / Ds[0]))
    ratio_sencera = float(lamD[-1] / lamD[0])
    if rang < RANG_MINIM_LN - 1e-12:
        # D_max ha hagut de baixar massa. Amb les fites de les censurades la ratio
        # sencera es una fita SUPERIOR: si ja es <= 0,5, es lineal de debo.
        v = "lineal" if ratio_sencera <= RATIO_LIN else "rang curt"
        return {"veredicte": v, "ratio": ratio_sencera, "j_max": j, "rang_ln": rang}
    ratio = float(lamD[j] / lamD[0])
    v = "exponencial" if ratio >= RATIO_EXP else ("lineal" if ratio <= RATIO_LIN else "no es pot distingir")
    return {"veredicte": v, "ratio": ratio, "j_max": j, "rang_ln": rang}


def _mapa_parelles(parelles):
    return {(min(i, j), max(i, j)): k for k, (i, j) in enumerate(parelles)}


def avalua_llindars(t, d, d0, parelles, n_tirs, Ds, *, n_boot=REMOSTREJOS, llavor=0,
                    t_max=T_MAX_S) -> dict:
    """Tot el metode sobre un conjunt de parelles: lambda(D), veredicte, lambda_inc i bootstrap."""
    T = creuaments(t, d, Ds)
    lam, fita, cens = lambdes_parelles(T, d0, Ds, t_max)
    lamD, frac = agrega(lam, fita, cens)
    v = veredicte_llindars(lamD, frac, Ds)
    # lambda_inc: de D_min a D_max (sencera), nomes les parelles que creuen totes dues
    ok = np.isfinite(T[:, 0]) & np.isfinite(T[:, -1]) & (T[:, -1] > T[:, 0])
    lam_inc = (float(np.median(math.log(Ds[-1] / Ds[0]) / (T[ok, -1] - T[ok, 0])))
               if ok.sum() >= 3 else None)
    r = {"D_graus": [math.degrees(x) for x in Ds], "lambda_D": lamD.tolist(),
         "fraccio_creuen": frac.tolist(), **v, "lambda_inc": lam_inc,
         "parelles": len(parelles), "creuen_d_entrada": int(np.sum(T[:, 0] == t[0])),
         "lambda_mitjana_escala": float(np.mean(lamD))}
    if n_boot:
        mapa = _mapa_parelles(parelles)
        rng = np.random.default_rng(llavor)
        mostres, exp = [], 0
        for _ in range(n_boot):
            tirs = rng.integers(0, n_tirs, n_tirs).tolist()
            idx = [mapa[(min(a, b), max(a, b))] for a, b in caos._parelles_de(tirs)
                   if (min(a, b), max(a, b)) in mapa]
            if len(idx) < 3:
                continue
            lb, fb = agrega(lam, fita, cens, np.array(idx))
            mostres.append(lb)
            exp += veredicte_llindars(lb, fb, Ds)["veredicte"] == "exponencial"
        m = np.array(mostres)
        r["bootstrap"] = {"remostrejos": len(m), "sigma_D": m.std(axis=0, ddof=1).tolist(),
                          "p16_84_D": np.percentile(m, [16, 84], axis=0).T.tolist(),
                          "fraccio_exponencial": exp / max(len(m), 1)}
    return r


# =============================================================== els veins

def troba_veins(Q, valids, E, *, periodic, sep, vel, banda, theiler, cada,
                de_max=None, temps=None, separacio_creuada=None) -> list:
    """(i, ka, j, kb) per a cada estat de referencia amb un vei admissible.

    Q: llista de (M_i, n) coordenades; valids: llista de (M_i,) bool (pot ser punt de
    partida: dins de la franja d'energia, prou futur...). sep: les coordenades de la
    separacio; vel: les altres, que han de complir |dq| <= d_sep. E: llista d'energies
    (o None). El vei es el MES PROPER en totes les coordenades, dins de la franja de d.
    """
    periodic = np.asarray(periodic, bool)
    n = Q[0].shape[1]
    sep, vel = list(sep), list(vel)
    if sorted(sep + vel) != list(range(n)):
        raise ValueError("totes les coordenades han de ser de separacio o restringides")

    def embed(x):
        cols = []
        for c in range(n):
            if periodic[c]:
                cols += [np.cos(x[:, c]), np.sin(x[:, c])]
            else:
                cols.append(x[:, c])
        return np.column_stack(cols)

    qi, qk = [], []
    for i, (q, v) in enumerate(zip(Q, valids)):
        k = np.nonzero(v)[0]
        qi.append(np.full(len(k), i))
        qk.append(k)
    qi, qk = np.concatenate(qi), np.concatenate(qk)
    pool = np.vstack([Q[i][k] for i, k in zip(qi, qk)]) if len(qi) else np.zeros((0, n))
    if not len(pool):
        return []
    Epool = np.concatenate([E[i][k:k + 1] for i, k in zip(qi, qk)]) if E is not None else None
    # temps / separacio_creuada (17/10, segona part): dos estats de tirs DIFERENTS han de ser
    # a instants que difereixin >= separacio_creuada, comptats des de l'amollada de cada tir.
    # Treu els «tirs que van sortir junts». Sense aquests arguments, res no canvia.
    Tpool = (np.concatenate([temps[i][k:k + 1] for i, k in zip(qi, qk)])
             if temps is not None and separacio_creuada is not None else None)
    arbre = cKDTree(embed(pool))
    lo, hi = banda
    radi = hi * math.sqrt(1 + len(vel)) * 1.0001
    refs = np.nonzero(qk % cada == 0)[0]
    sortida = []
    for r in refs:
        cand = np.array(arbre.query_ball_point(embed(pool[r:r + 1])[0], radi), dtype=int)
        if not len(cand):
            continue
        dq = pool[cand] - pool[r]
        dq[:, periodic] = dif_angular(pool[cand][:, periodic], pool[r][periodic])
        dsep = np.sqrt(np.sum(dq[:, sep] ** 2, axis=1))
        ok = (dsep >= lo) & (dsep <= hi)
        for c in vel:
            ok &= np.abs(dq[:, c]) <= dsep
        mateix = qi[cand] == qi[r]
        ok &= ~(mateix & (np.abs(qk[cand] - qk[r]) < theiler))
        if Tpool is not None:
            ok &= ~(~mateix & (np.abs(Tpool[cand] - Tpool[r]) < separacio_creuada))
        if Epool is not None and de_max is not None:
            ok &= np.abs(Epool[cand] - Epool[r]) <= de_max
        if not ok.any():
            continue
        dfull = np.sqrt(np.sum(dq ** 2, axis=1))
        b = cand[ok][np.argmin(dfull[ok])]
        sortida.append((int(qi[r]), int(qk[r]), int(qi[b]), int(qk[b])))
    return sortida


def segueix(Q, parelles, *, periodic, sep, H):
    """(P, H+1): la distancia de separacio de cada parella al llarg de H mostres."""
    periodic = np.asarray(periodic, bool)
    D = np.empty((len(parelles), H + 1))
    for p, (i, ka, j, kb) in enumerate(parelles):
        a, b = Q[i][ka:ka + H + 1], Q[j][kb:kb + H + 1]
        dq = b - a
        dq[:, periodic] = dif_angular(b[:, periodic], a[:, periodic])
        D[p] = np.sqrt(np.sum(dq[:, sep] ** 2, axis=1))
    return D


def pendent_veins(lnr, lnd, w, h0, dt, ln_sostre_3) -> dict:
    """La recta sobre <ln(d/d0)> de h0 fins que <ln d> arriba a ln(sostre/3)."""
    w = np.asarray(w, dtype=float)
    m = (w @ lnr) / w.sum()
    ml = (w @ lnd) / w.sum()
    H = len(m) - 1
    dalt = np.nonzero(ml[h0:] >= ln_sostre_3)[0]
    h1 = int(h0 + dalt[0]) if len(dalt) else H
    if h1 - h0 < 2:
        return {"ok": False, "lambda": 0.0, "creixement_ln": 0.0, "finestra_s": [h0 * dt, h1 * dt],
                "motiu": "la finestra te menys de tres punts"}
    tt = np.arange(h0, h1 + 1) * dt
    lam = float(np.polyfit(tt, m[h0:h1 + 1], 1)[0])
    creix = float(ml[h1] - ml[h0])
    return {"ok": bool(creix >= CREIXEMENT_MINIM), "lambda": lam, "creixement_ln": creix,
            "finestra_s": [h0 * dt, h1 * dt],
            "motiu": None if creix >= CREIXEMENT_MINIM else f"dins de la finestra creix {creix:.2f} (cal 1)"}


def pesos_bootstrap(parelles_tirs, tirs, n_tirs):
    m = np.bincount(tirs, minlength=n_tirs).astype(float)
    i = np.array([a for a, _ in parelles_tirs])
    j = np.array([b for _, b in parelles_tirs])
    return np.where(i == j, m[i], m[i] * m[j])


def avalua_veins(Q, valids, E, cfg, *, n_boot=REMOSTREJOS, llavor=0, guarda_valors=False) -> dict:
    """Tot el metode: les parelles, la corba, la lambda i el bootstrap per tirs.

    guarda_valors (23/10, per a l'horitzo): torna tambe les lambdes de cada remostreig.
    """
    par = troba_veins(Q, valids, E, periodic=cfg["periodic"], sep=cfg["sep"], vel=cfg["vel"],
                      banda=cfg["banda"], theiler=cfg["theiler"], cada=cfg["cada"],
                      de_max=cfg.get("de_max"), temps=cfg.get("temps"),
                      separacio_creuada=cfg.get("separacio_creuada"))
    base = {"parelles": len(par), "tirs": len({p[0] for p in par} | {p[2] for p in par}),
            "_parelles": par}
    if len(par) < 3:
        return {**base, "ok": False, "lambda": 0.0, "creixement_ln": 0.0, "finestra_s": None,
                "motiu": "menys de tres parelles"}
    D = segueix(Q, par, periodic=cfg["periodic"], sep=cfg["sep"], H=cfg["H"])
    lnd = np.log(np.maximum(D, 1e-300))
    lnr = lnd - lnd[:, :1]
    r = {**base, **pendent_veins(lnr, lnd, np.ones(len(par)), cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])}
    r["_lnr_mitjana"] = lnr.mean(axis=0)
    if n_boot:
        rng = np.random.default_rng(llavor)
        pt = [(p[0], p[2]) for p in par]
        n_tirs = len(Q)
        v, falla = [], 0
        for _ in range(n_boot):
            tirs = rng.integers(0, n_tirs, n_tirs)
            w = pesos_bootstrap(pt, tirs, n_tirs)
            if w.sum() <= 0:
                falla += 1
                continue
            b = pendent_veins(lnr, lnd, w, cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])
            v.append(b["lambda"])
            falla += not b["ok"]
        r["bootstrap"] = {"remostrejos": len(v), "sigma": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                          "sense_lambda": falla}
        if guarda_valors:
            r["bootstrap"]["valors"] = [float(x) for x in v]
    return r


# =============================================================== l'alineacio relativa

def alinea_relatiu(series, th1_repos, sigmes):
    """s_i perque cada tir, avaluat a t + s_i, s'assembli a la mediana del grup a [0, 0,08] s.

    series: llista de (t, th1) amb t comptat des de l'amollada actual. Sense model: la
    plantilla es la mediana dels tirs mateixos. La mitjana dels s_i es posa a zero.
    """
    tt = np.arange(0.0, AL_FINESTRA_S + 0.5 * DT, DT)
    s = np.zeros(len(series))
    for _ in range(AL_ITERACIONS):
        corbes = np.array([np.interp(tt + s[i], t, th) - th1_repos[i] for i, (t, th) in enumerate(series)])
        plantilla = np.median(corbes, axis=0)
        for i, (t, th) in enumerate(series):
            amp = 3.0 * sigmes[i]
            cand = np.arange(-amp, amp + 0.5 * AL_PAS_S, AL_PAS_S)
            v = np.interp(tt[None, :] + cand[:, None], t, th) - th1_repos[i]
            s[i] = cand[int(np.argmin(np.sum((v - plantilla) ** 2, axis=1)))]
        s -= s.mean()
    return s


# =============================================================== el soroll fabricat

@lru_cache(maxsize=64)
def _guany_filtre(rho: float) -> float:
    """La desviacio d'un AR(1) unitari despres del Savitzky-Golay de la conversio."""
    rng = np.random.default_rng(LLAVOR + int(round(rho * 1e6)))
    x = rg.ar1(rng, rho, int(200.0 * FPS))
    f = savgol_filter(x, rg.FINESTRA, rg.ORDRE)
    return float(f[50:-50].std(ddof=1))


def soroll_cru(rng, n, sd, rho):
    """(2, n) soroll cru d'angle perque, filtrat, cada component tingui la sd demanada."""
    return np.vstack([sd[k] / _guany_filtre(float(rho[k])) * rg.ar1(rng, float(rho[k]), n) for k in range(2)])


def observa(y, x0, sd, rho, rng):
    """Un tir com el donaria la conversio: repos + moviment, soroll cru, filtre.

    Torna (t des de l'amollada de debo, estats (4, M)), amb el repos inclos.
    """
    n_rep = int(round(REPOS_S * FPS))
    th = np.hstack([np.repeat(np.asarray(x0, dtype=float)[:2, None], n_rep, axis=1), y[:2]])
    if sd is not None:
        th = th + soroll_cru(rng, th.shape[1], sd, rho)
    return (np.arange(th.shape[1]) - n_rep) / FPS, rg.com_la_conversio(th)


def a_la_graella(t, X, zero, tc):
    """Els estats a tc comptat des de `zero` (interpolacio lineal, angles desembolicats)."""
    return np.vstack([np.interp(tc + zero, t, X[k]) for k in range(4)])


# =============================================================== les simulacions

_P = None


def _p():
    global _P
    if _P is None:
        _P = _parametres()
    return _P


def _sim_doble(args):
    from pendol.simulador import simula
    x0, temps, amb_friccio = args
    t = np.arange(0.0, temps + 0.5 * DT, DT)
    dis = None
    if amb_friccio:   # DE PROVA: C i eps provisionals, com a la resta del projecte
        dis = Dissipacio.des_de_friccio(Friccio.des_de_json(ARREL / "parametres.json"), C=0.00794, eps=1e-3)
    return simula(_p(), np.asarray(x0, dtype=float), t_eval=t, dis=dis)[1]


def _sim_simple(args):
    from pendol import simple
    th0, temps = args
    t = np.arange(0.0, temps + 0.5 * DT, DT)
    s = simple.Simple.des_de_parametres(_p())
    return simple.com_a_doble(simple.simula(s, [th0, 0.0], t_eval=t)[1])


def _ref_ll(args):
    """ln(d/delta) en angles d'una pertorbacio infinitesimal des de x0, en la direccio u."""
    from pendol.simulador import simula
    x0, u, temps = args
    t = np.arange(0.0, temps + 0.5 * DT, DT)
    x0 = np.asarray(x0, dtype=float)
    u = np.asarray(u, dtype=float)
    u = u / float(dist_ang(u, np.zeros(4)))
    ya = simula(_p(), x0, t_eval=t)[1]
    yb = simula(_p(), x0 + DELTA * u, t_eval=t)[1]
    return dist_ang(yb, ya) / DELTA


# =============================================================== pas 0: les preses reals

def preses_reals() -> list[dict]:
    """Totes les preses de l'index amb amollada (la 45_7 fora). NOMES amb dades.carrega."""
    sortida = []
    for f in json.loads(INDEX.read_text(encoding="utf-8"))["preses"]:
        if f.get("fora_del_balanc"):
            continue
        taula, meta = dades.carrega(NET / f"{f['nom']}.csv")
        a = meta.get("amollada") or {}
        if not a.get("hi_ha_amollada"):
            continue
        sortida.append({"nom": f["nom"], "grup": f"{f['grup_graus']:.0f}", "meta": meta, "taula": taula,
                        "zero_s": float(a["amollada_s"]), "sigma_s": float(a["incertesa_s"]),
                        "repos_rad": np.radians(a["angles_amollada_graus"]),
                        "t": taula["t"].to_numpy() - float(a["amollada_s"]),
                        "estats": estats_de_taula(taula)})
    return sortida


def soroll_del_tram(pr, w0):
    """Les components d'angle del soroll de seguiment al tram de 0,1-1 s (total i sense
    geometria), amb les funcions de soroll_en_moviment.py i el seu R0."""
    import soroll_en_moviment as sm
    meta = pr["meta"]
    t_fot, J, G, corregida = sm.posicions_del_trk(meta)
    c = sm.cadena(t_fot, J, G, bool(meta.get("signe_invertit", False)))
    r0 = sm.comprova_r0(c, meta, pr["taula"], corregida)
    fin, ordre = int(meta["savgol"]["finestra_mostres"]), int(meta["savgol"]["ordre"])
    am, t, fps = pr["zero_s"], c["t"], c["fps"]
    s, _ = sm.filtra(c["e"], fin, ordre, fps)
    tot = (t >= am + sm.DES_DE_S) & (t <= am + sm.FINS_A_S)
    tram = (t >= am + TRAM_S[0]) & (t <= am + TRAM_S[1])
    resid, _ = sm.treu_geometria(c, tot)
    sr, _ = sm.filtra(resid, fin, ordre, fps)
    # L'rms del tram AMB l'offset respecte de la llargada mediana de tota la presa: un
    # soroll lent te, en 0,9 s, una desviacio de finestra molt mes petita que el seu nivell
    # (autoprova LLa7, 17/10). Es publica al costat; quina va a la validacio ho decideix en Guillem.
    return {"r0_ok": bool(r0["ok"]),
            "tram_rms_rad": [float(np.sqrt(np.mean(s[k][tram] ** 2))) for k in range(2)],
            "tram_rms_sense_geometria_rad": [float(np.sqrt(np.mean(sr[k][tram] ** 2))) for k in range(2)],
            "tram_total_rad": [float(s[k][tram].std(ddof=1)) for k in range(2)],
            "tram_sense_geometria_rad": [float(sr[k][tram].std(ddof=1)) for k in range(2)],
            "finestra_total_rad": [float(s[k][tot].std(ddof=1)) for k in range(2)]}


def _sostre_desaparellat(series, finestres, rng, mostres=SOSTRE_MOSTRES):
    """La mediana de d entre estats de tirs DIFERENTS a instants independents."""
    v = []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            ki = rng.choice(finestres[i], mostres)
            kj = rng.choice(finestres[j], mostres)
            v.append(dist_ang(series[i][:, ki], series[j][:, kj]))
    return float(np.median(np.concatenate(v)))


def escala(ingr: dict, noms, sigma_alin: dict, sostre: float) -> dict:
    """D_min, D_max, els 5 D i les parelles, per la regla del pla. Tot en radians."""
    noms = list(noms)
    # L'rms del tram (decidit el 17/10 despres de la fallada del LLa7), no la desviacio.
    s = {n: math.hypot(*ingr[n]["soroll"]["tram_rms_rad"]) for n in noms}
    w = {n: ingr[n]["omega_tram_rad_s"] for n in noms}
    terres, seg, alin = [], [], []
    for a in range(len(noms)):
        for b in range(a + 1, len(noms)):
            i, j = noms[a], noms[b]
            sg = math.hypot(s[i], s[j])
            al = 0.5 * (w[i] + w[j]) * math.hypot(sigma_alin[i], sigma_alin[j])
            seg.append(sg), alin.append(al), terres.append(math.hypot(sg, al))
    terra = float(np.median(terres))
    D_max = sostre / 3.0
    tria = None
    for f in FACTORS_TERRA:
        if math.log(D_max / (f * terra)) >= RANG_MINIM_LN:
            tria = f
            break
    r = {"terra_rad": terra, "seguiment_rad": float(np.median(seg)), "alineacio_rad": float(np.median(alin)),
         "sostre_rad": sostre, "D_max_rad": D_max, "factor": tria,
         "rang_ln_3x": math.log(D_max / (3 * terra)), "rang_ln_2x": math.log(D_max / (2 * terra)),
         "avaluable": tria is not None}
    if tria is None:
        return r
    D_min = tria * terra
    r["D_rad"] = np.geomspace(D_min, D_max, N_D).tolist()
    return r


def parelles_de_lescala(d0_de, n, D_min):
    """Les parelles (i < j) amb d0 <= D_min / e."""
    return [(i, j) for i in range(n) for j in range(i + 1, n) if d0_de(i, j) <= D_min / math.e]


def descriu() -> dict:
    """El pas 0: tot el que es pot dir de les preses SENSE cap corba de separacio."""
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    preses = preses_reals()
    caracter = rg.caracter_mesurat()
    rng = np.random.default_rng(LLAVOR)
    ingr, grups = {}, {}
    for pr in preses:
        print(f"  {pr['nom']} ...", end="", flush=True)
        t, X = pr["t"], pr["estats"]
        E = np.asarray(caos.energia_relativa(X, p)) / Ev2
        tram = (t >= TRAM_S[0]) & (t <= TRAM_S[1])
        franja = FRANGES_E[pr["grup"]]
        dins = (t >= VEINS_DES_DE_S) & (E >= franja[0]) & (E <= franja[1])
        dt = float(np.median(np.diff(t)))
        ingr[pr["nom"]] = {
            "grup": pr["grup"], "sigma_amollada_s": pr["sigma_s"],
            "angles_repos_graus": np.degrees(pr["repos_rad"]).tolist(),
            "E_sobre_Ev2": {f"{x:g}": float(np.interp(x, t, E)) for x in (0.0, 1.5, 3.0)},
            "temps_a_la_franja_s": float(dins.sum() * dt),
            "omega_tram_rad_s": float(np.median(np.hypot(X[2][tram], X[3][tram]))),
            "durada_s": float(t[-1]), "fps": 1.0 / dt,
            "soroll": soroll_del_tram(pr, w0),
            "inclosa_llindars": pr["sigma_s"] <= SIGMA_AMOLLADA_MAX_S,
        }
        print(f" soroll al tram {np.degrees(ingr[pr['nom']]['soroll']['tram_total_rad'])} graus, "
              f"R0 {'ok' if ingr[pr['nom']]['soroll']['r0_ok'] else 'FALLA'}", flush=True)
    for g in GRUPS:
        pg = [pr for pr in preses if pr["grup"] == g]
        noms = [pr["nom"] for pr in pg]
        tc = np.arange(0.0, SOSTRE_FINESTRA_S[1] + 0.5 * DT, DT)
        series = [a_la_graella(pr["t"], pr["estats"], 0.0, tc) for pr in pg]
        fin = [np.nonzero(tc >= SOSTRE_FINESTRA_S[0])[0]] * len(pg)
        sostre = _sostre_desaparellat(series, fin, rng)
        # El sostre dels veins: estats de dins de la franja d'energia.
        franja = FRANGES_E[g]
        sv, fv = [], []
        for pr in pg:
            E = np.asarray(caos.energia_relativa(pr["estats"], p)) / Ev2
            k = np.nonzero((pr["t"] >= VEINS_DES_DE_S) & (E >= franja[0]) & (E <= franja[1]))[0]
            if len(k):
                sv.append(pr["estats"]), fv.append(k)
        sostre_v = _sostre_desaparellat(sv, fv, rng)
        d0 = {}
        for a in range(len(pg)):
            for b in range(a + 1, len(pg)):
                d0[(noms[a], noms[b])] = float(dist_ang(pg[a]["repos_rad"], pg[b]["repos_rad"]))
        incl = [n for n in noms if ingr[n]["inclosa_llindars"]]
        esc = escala(ingr, incl, {n: ingr[n]["sigma_amollada_s"] for n in incl}, sostre)
        if esc["avaluable"]:
            Dmin = esc["D_rad"][0]
            esc["parelles"] = sum(1 for a in range(len(incl)) for b in range(a + 1, len(incl))
                                  if d0[(incl[a], incl[b])] <= Dmin / math.e)
        seg = [math.hypot(math.hypot(*ingr[a]["soroll"]["tram_rms_rad"]),
                          math.hypot(*ingr[b]["soroll"]["tram_rms_rad"]))
               for a, b in d0]
        terra_seg = float(np.median(seg))
        # El nivell de la validacio: mitjana quadratica del grup, component a component.
        mq = lambda clau: np.sqrt(np.mean([np.square(ingr[n]["soroll"][clau]) for n in noms], axis=0)).tolist()  # noqa: E731
        grups[g] = {
            "preses": noms, "inclosos_llindars": incl, "rho": caracter[g]["rho"],
            "d0_graus": {"mediana": math.degrees(np.median(list(d0.values()))),
                         "min": math.degrees(min(d0.values())), "max": math.degrees(max(d0.values()))},
            "sostre_llindars_rad": sostre, "sostre_veins_rad": sostre_v,
            "terra_seguiment_veins_rad": terra_seg,
            "soroll_validacio_rad": {"total": mq("tram_rms_rad"),
                                     "sense_geometria": mq("tram_rms_sense_geometria_rad")},
            "banda_veins_rad": [BANDA_D0[0] * terra_seg, BANDA_D0[1] * terra_seg],
            "escala_video": esc,
        }
    return {"eina": "eines/lyapunov_real.py descriu", "pla": "documentacio/pla-analisi-caos.md (17/10)",
            "que_es": "PAS 0: les preses reals SENSE cap corba de separacio",
            "omega0_rad_s": w0, "E_volteig_2_J": Ev2, "per_presa": ingr, "per_grup": grups}


def imprimeix_pas0(r):
    g = math.degrees
    for k, v in r["per_grup"].items():
        e = v["escala_video"]
        pp = [r["per_presa"][n] for n in v["preses"]]
        print(f"\n== {k} graus: {len(v['preses'])} preses, {len(v['inclosos_llindars'])} als llindars")
        print(f"   d0: mediana {v['d0_graus']['mediana']:.2f} graus ({v['d0_graus']['min']:.2f}-{v['d0_graus']['max']:.2f})")
        print(f"   E/E_v2 a 0 s: {min(x['E_sobre_Ev2']['0'] for x in pp):.2f}-{max(x['E_sobre_Ev2']['0'] for x in pp):.2f}; "
              f"a 1,5 s: {min(x['E_sobre_Ev2']['1.5'] for x in pp):.2f}-{max(x['E_sobre_Ev2']['1.5'] for x in pp):.2f}; "
              f"a 3 s: {min(x['E_sobre_Ev2']['3'] for x in pp):.2f}-{max(x['E_sobre_Ev2']['3'] for x in pp):.2f}")
        print(f"   temps a la franja dels veins {FRANGES_E[k]}: {min(x['temps_a_la_franja_s'] for x in pp):.1f}-"
              f"{max(x['temps_a_la_franja_s'] for x in pp):.1f} s per tir")
        hy = lambda c: g(math.hypot(*c))  # noqa: E731
        print(f"   soroll de seguiment en angles per tir (mediana): desviacio a 0,1-1 s "
              f"{np.median([hy(x['soroll']['tram_total_rad']) for x in pp]):.2f}, rms a 0,1-1 s "
              f"{np.median([hy(x['soroll']['tram_rms_rad']) for x in pp]):.2f}, desviacio a 0,1-10 s "
              f"{np.median([hy(x['soroll']['finestra_total_rad']) for x in pp]):.2f} graus")
        print(f"   |omega| al tram 0,1-1 s: mediana {np.median([x['omega_tram_rad_s'] for x in pp]):.1f} rad/s; "
              f"sigma d'amollada: mediana {1000 * np.median([x['sigma_amollada_s'] for x in pp]):.1f} ms")
        print(f"   llindars (video): seguiment {g(e['seguiment_rad']):.2f}, alineacio {g(e['alineacio_rad']):.2f}, "
              f"terra {g(e['terra_rad']):.2f} graus; sostre {g(e['sostre_rad']):.1f} -> D_max {g(e['D_max_rad']):.1f}; "
              f"rang ln amb 3x {e['rang_ln_3x']:.2f}, amb 2x {e['rang_ln_2x']:.2f}")
        if e["avaluable"]:
            print(f"      -> {e['factor']:g} x terra; D = {', '.join(f'{g(x):.1f}' for x in e['D_rad'])} graus; "
                  f"{e['parelles']} parelles amb d0 <= D_min/e")
        else:
            print("      -> NO AVALUABLE amb l'amollada del video: no queden 1,2 e de rang")
        print(f"   veins: terra de seguiment {g(v['terra_seguiment_veins_rad']):.2f} graus, franja de d0 "
              f"{g(v['banda_veins_rad'][0]):.2f}-{g(v['banda_veins_rad'][1]):.2f}, sostre a la franja "
              f"{g(v['sostre_veins_rad']):.1f} -> sostre/3 {g(v['sostre_veins_rad']) / 3:.1f} graus "
              f"({math.log(v['sostre_veins_rad'] / 3 / np.mean(v['banda_veins_rad'])):.2f} e des del mig de la franja)")
    r0 = [n for n, x in r["per_presa"].items() if not x["soroll"]["r0_ok"]]
    print(f"\nR0 (el cami de la conversio dona el CSV): {'ok a totes' if not r0 else 'FALLA a ' + ', '.join(r0)}")


# =============================================================== pas 1: la validacio

def _veins_cfg(g, pas0, dt=DT):
    v = pas0["per_grup"][g]
    return {"periodic": [True, True, False, False], "sep": [0, 1], "vel": [2, 3],
            "banda": tuple(v["banda_veins_rad"]), "theiler": int(round(THEILER_S / dt)),
            "cada": CADA_REF, "de_max": DE_MAX, "H": int(round(VEINS_FINS_S / dt)),
            "h0": int(round(VEINS_DES_DE_S / dt)), "dt": dt,
            "ln_sostre_3": math.log(v["sostre_veins_rad"] / 3.0)}


def _veins_entrada(obs, p, Ev2, w0, franja, dt=DT):
    """De tirs (t, X) a les coordenades dels veins, valids i energies."""
    Q, V, E = [], [], []
    H = int(round(VEINS_FINS_S / dt))
    for t, X in obs:
        q = np.column_stack([X[0], X[1], X[2] / w0, X[3] / w0])
        e = np.asarray(caos.energia_relativa(X, p)) / Ev2
        v = (t >= VEINS_DES_DE_S) & (e >= franja[0]) & (e <= franja[1])
        v[max(len(t) - H, 0):] = False
        Q.append(q), V.append(v), E.append(e)
    return Q, V, E


def _resum_criteri(nom, ok, text):
    return {"criteri": nom, "passa": bool(ok), "detall": text}


def valida(pas0: dict, processos: int = PROCESSOS) -> dict:
    """El pas 1, NOMES SIMULACIO, amb els criteris del pla del 17/10."""
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    ref_b = json.loads(REGLA2_JSON.read_text(encoding="utf-8"))
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    sortida = {"grups": {}}
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processos) as pool:
        for g in GRUPS:
            ara = time.perf_counter()
            G = pas0["per_grup"][g]
            noms = G["preses"]
            ing = [pas0["per_presa"][n] for n in noms]
            N = len(noms)
            x0s = [np.radians(x["angles_repos_graus"] + [0.0, 0.0]) for x in ing]
            x0s = [np.array([a[0], a[1], 0.0, 0.0]) for a in x0s]
            th_s = [0.5 * (x[0] + x[1]) for x in x0s]
            sig = np.array([x["sigma_amollada_s"] for x in ing])
            rho = G["rho"]
            print(f"\n== {g} graus: simulant {N} tirs del doble, {N} del simple i {N} amb friccio DE PROVA ...",
                  flush=True)
            Y = pool.map(_sim_doble, [(x, DURADA_SIM_S, False) for x in x0s])
            Ys = pool.map(_sim_simple, [(th, DURADA_SIM_S) for th in th_s])
            Yf = pool.map(_sim_doble, [(x, DURADA_SIM_S, True) for x in x0s])
            x0s_s = [np.array([th, th, 0.0, 0.0]) for th in th_s]
            sv = G["soroll_validacio_rad"]
            nivells = {"total": [np.array(sv["total"])] * N,
                       "sense_geometria": [np.array(sv["sense_geometria"])] * N,
                       "doble_total": [2.0 * np.array(sv["total"])] * N}

            # Les observacions: ideal (sense soroll ni error d'amollada) i 5 per nivell.
            def observacions(Yset, x0set, sd_list, r, nivell, que):
                if sd_list is None:
                    return [observa(Y_[:, :], x0, None, rho, None) for Y_, x0 in zip(Yset, x0set)]
                return [observa(Y_, x0, sd_list[i], rho,
                                np.random.default_rng(rg.llavor_de(float(g), r, i, len(nivell), que)))
                        for i, (Y_, x0) in enumerate(zip(Yset, x0set))]

            dts = [np.random.default_rng(rg.llavor_de(float(g), r, 77)).normal(0.0, sig) for r in range(REALITZACIONS)]
            casos = [("ideal", None, None)]
            for niv in nivells:
                for r in range(REALITZACIONS):
                    casos.append((niv, r, nivells[niv]))
            obs = {}
            for niv, r, sd in casos:
                o_d = observacions(Y, x0s, sd, r if r is not None else 0, niv, 1)
                o_s = observacions(Ys, x0s_s, sd, r if r is not None else 0, niv, 2)
                dt_i = np.zeros(N) if r is None else dts[r]
                obs[(niv, r)] = (o_d, o_s, dt_i)
            o_f = observacions(Yf, x0s, None, 0, "ideal", 3)

            # ---------- l'alineacio relativa: AL1 i AL2
            def relativa(o, x0set, dt_i):
                ser = [(t - dt_i[i], X[0]) for i, (t, X) in enumerate(o)]
                return alinea_relatiu(ser, [x[0] for x in x0set], sig)
            incl = [k for k, n in enumerate(noms) if pas0["per_presa"][n]["inclosa_llindars"]]
            s_ideal = relativa(obs[("ideal", None)][0], x0s, np.zeros(N))
            al2 = float(np.max(np.abs(s_ideal[incl])))
            rms_al, rms_tots = {}, {}
            s_cas = {("ideal", None): (s_ideal, relativa(obs[("ideal", None)][1], x0s_s, np.zeros(N)))}
            for niv, r, sd in casos[1:]:
                o_d, o_s, dt_i = obs[(niv, r)]
                sd_ = relativa(o_d, x0s, dt_i)
                ss_ = relativa(o_s, x0s_s, dt_i)
                s_cas[(niv, r)] = (sd_, ss_)
                res = (sd_ + dt_i)[incl]
                rms_al.setdefault(niv, []).append(float(np.sqrt(np.mean((res - res.mean()) ** 2))))
                rt = sd_ + dt_i
                rms_tots.setdefault(niv, []).append(float(np.sqrt(np.mean((rt - rt.mean()) ** 2))))
            al1_n = sum(x <= COTES["AL1_s"] for x in rms_al["total"])
            AL = {"AL1": _resum_criteri("AL1", al1_n >= 4, f"rms del residu relatiu (total, preses dels llindars) "
                                        f"{', '.join(f'{1000 * x:.2f}' for x in rms_al['total'])} ms; cal <= 2 ms a >= 4 de 5"),
                  "AL2": _resum_criteri("AL2", al2 <= COTES["AL2_s"], f"sense soroll, |s| maxim {1000 * al2:.3f} ms; cal <= 0,5 ms"),
                  "rms_ms": {k: [1000 * x for x in v] for k, v in rms_al.items()},
                  "rms_ms_amb_totes_les_preses_info": {k: [1000 * x for x in v] for k, v in rms_tots.items()}}
            relativa_ok = AL["AL1"]["passa"] and AL["AL2"]["passa"]
            sigma_rel = max(rms_al["total"])

            # ---------- les escales: video (del pas 0) i relativa (amb la sigma de AL1)
            nincl = [noms[k] for k in incl]
            escales = {"video": G["escala_video"]}
            if relativa_ok:
                escales["relativa"] = escala(pas0["per_presa"], nincl, {n: sigma_rel for n in nincl},
                                             G["sostre_llindars_rad"])
            else:
                escales["relativa"] = {"avaluable": False, "motiu": "AL1 o AL2 fallen"}

            # ---------- la referencia linealitzada, per a cada parella dels inclosos
            parelles_tots = [(a, b) for a in range(len(incl)) for b in range(a + 1, len(incl))]
            feines = []
            for a, b in parelles_tots:
                i, j = incl[a], incl[b]
                u = np.array([dif_angular(x0s[j][0], x0s[i][0]), dif_angular(x0s[j][1], x0s[i][1]), 0.0, 0.0])
                feines.append((x0s[i], u, T_MAX_S))
            G_lin = np.array(pool.map(_ref_ll, feines))
            d0_d = np.array([float(dist_ang(x0s[incl[a]], x0s[incl[b]])) for a, b in parelles_tots])
            d0_s = np.array([float(dist_ang(x0s_s[incl[a]], x0s_s[incl[b]])) for a, b in parelles_tots])

            blocs_ll = {}
            for var, esc in escales.items():
                if not esc.get("avaluable"):
                    blocs_ll[var] = {"escala": esc, "avaluable": False}
                    continue
                Ds = np.array(esc["D_rad"])
                sel_d = [k for k, (a, b) in enumerate(parelles_tots) if d0_d[k] <= Ds[0] / math.e]
                sel_s = [k for k, (a, b) in enumerate(parelles_tots) if d0_s[k] <= Ds[0] / math.e]
                par_d = [parelles_tots[k] for k in sel_d]
                par_s = [parelles_tots[k] for k in sel_s]
                # La referencia: d0 x g(t), el seu primer creuament.
                T_lin = creuaments(tc, d0_d[sel_d, None] * G_lin[sel_d], Ds)
                lam, fita, cens = lambdes_parelles(T_lin, d0_d[sel_d], Ds, T_MAX_S)
                ref_D, ref_frac = agrega(lam, fita, cens)

                def corre(o, s_rel, dt_i, par, d0v, xs_sel, llavor, simple=False):
                    zeros = dt_i + (s_rel if var == "relativa" else 0.0)
                    X = [a_la_graella(t, Xo, zeros[i], tc) for i, (t, Xo) in enumerate(o)]
                    Xi = [X[k] for k in incl]
                    d = np.array([dist_ang(Xi[a], Xi[b]) for a, b in par])
                    return avalua_llindars(tc, d, d0v, par, len(incl), Ds, llavor=llavor)

                res = {}
                for (niv, r), (o_d, o_s, dt_i) in obs.items():
                    s_d, s_s = s_cas[(niv, r)]
                    ll = rg.llavor_de(float(g), r if r is not None else 9, len(niv))
                    res[(niv, r)] = (corre(o_d, s_d, dt_i, par_d, d0_d[sel_d], None, ll),
                                     corre(o_s, s_s, dt_i, par_s, d0_s[sel_s], None, ll + 1) if par_s else None)
                fr = corre(o_f, np.zeros(N), np.zeros(N), par_d, d0_d[sel_d], None, 5)
                # ------- criteris LL
                ideal_d = res[("ideal", None)][0]
                rel_ideal = np.array(ideal_d["lambda_D"]) / ref_D - 1
                tot = [res[("total", r)] for r in range(REALITZACIONS)]

                def ll2_ok(rr):
                    rel = np.array(rr["lambda_D"]) / ref_D - 1
                    sb = rr["bootstrap"]["sigma_D"]
                    return (bool(np.all(np.abs(rel) <= COTES["LL2"]))
                            and sb[0] <= COTES["boot"] * rr["lambda_D"][0]
                            and sb[-1] <= COTES["boot"] * rr["lambda_D"][-1])
                n_ll2 = sum(ll2_ok(d) for d, _ in tot)
                n_ll3 = sum(d["veredicte"] == "exponencial" for d, _ in tot)
                simples = [s for _, s in tot]
                simple_exp = sum(1 for s in simples if s and s["veredicte"] == "exponencial")
                simple_lin = sum(1 for s in simples if (s is None) or s["veredicte"] in ("lineal", "no creua"))
                crit = {
                    "LL1": _resum_criteri("LL1", bool(np.all(np.abs(rel_ideal) <= COTES["LL1"])),
                                          "sense soroll / referencia - 1: " + ", ".join(f"{100 * x:+.0f} %" for x in rel_ideal)),
                    "LL2": _resum_criteri("LL2", n_ll2 >= 4, f"{n_ll2} de 5 a +-25 % de la referencia amb sigma_boot <= 20 %"),
                    "LL3": _resum_criteri("LL3", ideal_d["veredicte"] == "exponencial" and n_ll3 >= 4,
                                          f"sense soroll {ideal_d['veredicte']} (ratio {ideal_d['ratio']}); "
                                          f"amb soroll, exponencial a {n_ll3} de 5"),
                    "LL4": _resum_criteri("LL4", simple_exp == 0 and simple_lin >= 4,
                                          f"el simple: exponencial a {simple_exp} de 5, lineal o no creua a {simple_lin} de 5"),
                    "LL5": _resum_criteri("LL5", ideal_d["fraccio_creuen"][-1] >= FRACCIO_CREUEN,
                                          f"sense soroll creuen D_max {100 * ideal_d['fraccio_creuen'][-1]:.0f} %"),
                }
                blocs_ll[var] = {
                    "avaluable": True, "escala": esc, "parelles_doble": len(par_d), "parelles_simple": len(par_s),
                    "referencia_D": ref_D.tolist(), "referencia_fraccio": ref_frac.tolist(),
                    "criteris": crit,
                    "passa": all(c["passa"] for c in crit.values()),
                    "corregudes": {f"{niv}:{r}": {"doble": d, "simple": s} for (niv, r), (d, s) in res.items()},
                    "friccio_de_prova_ideal": fr,
                }

            # ---------- els veins
            franja = FRANGES_E[g]
            cfg = _veins_cfg(g, pas0)
            vres = {}
            for (niv, r), (o_d, o_s, _) in obs.items():
                cut = lambda o: [(t[t >= 0], X[:, t >= 0]) for t, X in o]  # noqa: E731
                Qd, Vd, Ed = _veins_entrada(cut(o_d), p, Ev2, w0, franja)
                Qs, Vs, Es = _veins_entrada(cut(o_s), p, Ev2, w0, franja)
                ll = rg.llavor_de(float(g), r if r is not None else 9, len(niv), 3)
                vres[(niv, r)] = (avalua_veins(Qd, Vd, Ed, cfg, llavor=ll), avalua_veins(Qs, Vs, Es, cfg, llavor=ll + 1))
            cut = lambda o: [(t[t >= 0], X[:, t >= 0]) for t, X in o]  # noqa: E731
            Qf, Vf, Ef = _veins_entrada(cut(o_f), p, Ev2, w0, franja)
            vfr = avalua_veins(Qf, Vf, Ef, cfg, llavor=6)
            # La referencia: FTLE infinitesimal des dels estats (de debo) de les parelles ideals.
            par_i = vres[("ideal", None)][0]["_parelles"]
            rr = np.random.default_rng(LLAVOR + int(g))
            mostra = (rr.choice(len(par_i), VEINS_REF_MAX, replace=False) if len(par_i) > VEINS_REF_MAX
                      else np.arange(len(par_i)))
            feines = []
            for m in mostra:
                i, ka, j, kb = par_i[m]
                xa, xb = Y[i][:, ka], Y[j][:, kb]
                u = np.array([dif_angular(xb[0], xa[0]), dif_angular(xb[1], xa[1]), xb[2] - xa[2], xb[3] - xa[3]])
                feines.append((xa, u, VEINS_FINS_S))
            Gv = np.log(np.array(pool.map(_ref_ll, feines))) if feines else np.zeros((0, cfg["H"] + 1))
            ref_corba = Gv.mean(axis=0) if len(Gv) else None

            def ref_pendent(fin):
                if ref_corba is None or fin is None:
                    return None
                a, b = int(round(fin[0] / DT)), int(round(fin[1] / DT))
                return float(np.polyfit(np.arange(a, b + 1) * DT, ref_corba[a:b + 1], 1)[0]) if b - a >= 2 else None

            vi = vres[("ideal", None)][0]
            ref_i = ref_pendent(vi["finestra_s"])
            vt = [vres[("total", r)] for r in range(REALITZACIONS)]

            def vn2(d):
                rf = ref_pendent(d["finestra_s"])
                sb = (d.get("bootstrap") or {}).get("sigma")
                return (rf is not None and d["ok"] and abs(d["lambda"] / rf - 1) <= COTES["VN2"]
                        and sb is not None and sb <= COTES["boot"] * abs(d["lambda"]))

            def vn4(d, s):
                sd = (d.get("bootstrap") or {}).get("sigma") or 0.0
                ss = (s.get("bootstrap") or {}).get("sigma") or 0.0
                ls = s["lambda"] if s["parelles"] >= 3 else 0.0
                return ls <= COTES["VN4"] * d["lambda"] and d["lambda"] - ls >= 2.0 * math.hypot(sd, ss)
            n_vn2 = sum(vn2(d) for d, _ in vt)
            n_vn3 = sum(d["ok"] for d, _ in vt)
            n_vn4 = sum(vn4(d, s) for d, s in vt)
            n_vn5 = sum(d["parelles"] >= 100 and d["tirs"] >= 8 for d, _ in vt)
            vcrit = {
                "VN1": _resum_criteri("VN1", ref_i is not None and abs(vi["lambda"] / ref_i - 1) <= COTES["VN1"],
                                      f"sense soroll {vi['lambda']:.3f} contra la referencia {ref_i}"),
                "VN2": _resum_criteri("VN2", n_vn2 >= 4, f"{n_vn2} de 5 a +-25 % de la referencia amb sigma_boot <= 20 %"),
                "VN3": _resum_criteri("VN3", vi["ok"] and n_vn3 >= 4,
                                      f"creixement sense soroll {vi['creixement_ln']:.2f}; >= 1 a {n_vn3} de 5"),
                "VN4": _resum_criteri("VN4", n_vn4 >= 4, f"{n_vn4} de 5 amb el simple <= 0,3 x el doble i a >= 2 sigma"),
                "VN5": _resum_criteri("VN5", vi["parelles"] >= 100 and vi["tirs"] >= 8 and n_vn5 >= 4,
                                      f"sense soroll {vi['parelles']} parelles de {vi['tirs']} tirs; >= 100 de >= 8 a {n_vn5} de 5"),
            }

            def neta(d):
                return {k: v for k, v in d.items() if not k.startswith("_")}
            bloc_v = {"config": {k: v for k, v in cfg.items()}, "criteris": vcrit,
                      "passa": all(c["passa"] for c in vcrit.values()),
                      "referencia_pendent_ideal": ref_i, "referencia_parelles": int(len(Gv)),
                      "corregudes": {f"{niv}:{r}": {"doble": {**neta(d), "referencia": ref_pendent(d["finestra_s"])},
                                                     "simple": neta(s)} for (niv, r), (d, s) in vres.items()},
                      "friccio_de_prova_ideal": neta(vfr)}
            bm = ref_b["grups"][g].get("benettin", {})
            sortida["grups"][g] = {"alineacio_relativa": AL, "sigma_relativa_s": sigma_rel,
                                   "llindars": blocs_ll, "veins": bloc_v,
                                   "benettin_nominal_info": {"lambda_mediana_caotiques": bm.get("lambda_mediana_caotiques"),
                                                             "caotiques": bm.get("caotiques"), "regulars": bm.get("regulars")},
                                   "segons": time.perf_counter() - ara}
            imprimeix_grup(g, sortida["grups"][g])
    return sortida


def imprimeix_grup(g, b):
    print(f"\n-- {g} graus ({b['segons']:.0f} s). NOMES SIMULACIO, sense friccio.")
    for k in ("AL1", "AL2"):
        c = b["alineacio_relativa"][k]
        print(f"   {'ok ' if c['passa'] else 'NO '} {k}: {c['detall']}")
    for var, bl in b["llindars"].items():
        if not bl.get("avaluable"):
            print(f"   llindars ({var}): NO AVALUABLE ({bl['escala'].get('motiu', 'no queden 1,2 e de rang')})")
            continue
        e = bl["escala"]
        print(f"   llindars ({var}): {e['factor']:g} x terra, D = {', '.join(f'{math.degrees(x):.1f}' for x in e['D_rad'])} "
              f"graus; {bl['parelles_doble']} parelles. Referencia {', '.join(f'{x:.2f}' for x in bl['referencia_D'])}")
        idl = bl["corregudes"]["ideal:None"]["doble"]
        print(f"      sense soroll: {', '.join(f'{x:.2f}' for x in idl['lambda_D'])}  ({idl['veredicte']}, ratio "
              f"{idl['ratio']}, lambda_inc {idl['lambda_inc']})")
        for r in range(REALITZACIONS):
            c = bl["corregudes"][f"total:{r}"]
            d, s = c["doble"], c["simple"]
            print(f"      total {r}: doble {', '.join(f'{x:.2f}' for x in d['lambda_D'])} ({d['veredicte']}, "
                  f"{d['ratio'] if d['ratio'] is None else round(d['ratio'], 2)}; sigma {d['bootstrap']['sigma_D'][0]:.2f}/"
                  f"{d['bootstrap']['sigma_D'][-1]:.2f}) | simple {s['veredicte'] if s else 'cap parella'} "
                  f"({None if not s or s['ratio'] is None else round(s['ratio'], 2)})")
        for k, c in bl["criteris"].items():
            print(f"      {'ok ' if c['passa'] else 'NO '} {k}: {c['detall']}")
        fr = bl["friccio_de_prova_ideal"]
        print(f"      (info) amb friccio DE PROVA, sense soroll: {', '.join(f'{x:.2f}' for x in fr['lambda_D'])} ({fr['veredicte']})")
    v = b["veins"]
    vi = v["corregudes"]["ideal:None"]
    print(f"   veins: franja {math.degrees(v['config']['banda'][0]):.2f}-{math.degrees(v['config']['banda'][1]):.2f} graus. "
          f"Sense soroll: doble {vi['doble']['lambda']:.3f} (ref {vi['doble']['referencia']}, creix "
          f"{vi['doble']['creixement_ln']:.2f}, {vi['doble']['parelles']} parelles), simple {vi['simple']['lambda']:.3f}")
    for r in range(REALITZACIONS):
        c = v["corregudes"][f"total:{r}"]
        d, s = c["doble"], c["simple"]
        print(f"      total {r}: doble {d['lambda']:.3f} +- {(d.get('bootstrap') or {}).get('sigma')} (ref {d['referencia']}, "
              f"creix {d['creixement_ln']:.2f}, {d['parelles']} parelles) | simple {s['lambda']:.3f}")
    for k, c in v["criteris"].items():
        print(f"      {'ok ' if c['passa'] else 'NO '} {k}: {c['detall']}")
    print(f"      (info) amb friccio DE PROVA, sense soroll: {v['friccio_de_prova_ideal']['lambda']:.3f}")
    bn = b["benettin_nominal_info"]
    print(f"   (info) Benettin, 12 orbites nominals de 1000 s: {bn['caotiques']} caotiques, {bn['regulars']} regulars, "
          f"lambda mediana {bn['lambda_mediana_caotiques']}")


# =============================================================== segona part (17/10)
# «Un resultat amb les preses d'ara». Criteris al pla, «17/10 (segona part)», escrits
# ABANS d'aquest codi. Resultat 2: el temps fins a una sola D gran. Resultat 1: la lambda
# per franges d'energia amb veins de debo, barrejant totes les preses.

FRANGES_R1 = {"0,8-1,1": (0.8, 1.1), "1,7-3,5": (1.7, 3.5), "3,5-6,3": (3.5, 6.3)}
RHO_DE_FRANJA = {"0,8-1,1": "45", "1,7-3,5": "90", "3,5-6,3": "150"}   # el grup que hi passa mes temps
ENERGIES_LAMBDA_E = {"0,8-1,1": (0.8, 0.9, 0.97, 1.03, 1.1), "1,7-3,5": (1.7, 2.5, 3.3, 3.45),
                     "3,5-6,3": (4.5, 6.2)}
LAMBDA_E_ORBITES = ARREL / "dades" / "lyapunov" / "lambda_E_2026-10-01" / "orbites.jsonl"
BANDA_D0_R1 = (3.0, 6.0)
SEPARACIO_CREUADA_S = 0.5
DURADA_R1_S = (6.0, 30.0)
TIRS_R1 = 12
FACTOR_SOROLL_FORA = 2.0
FRACCIO_CAOTIQUES_VE2 = 0.8
COTES2 = {"LU1": 0.20, "LU2": 0.25, "VE1": 0.20, "VE2": 0.30, "VE3": 0.25, "VE5": 0.3}


def fora_per_soroll(pas0) -> dict:
    """Les preses amb el soroll del tram (rms, norma d'angles) > 2 x la mediana del grup."""
    fora = {}
    for g, G in pas0["per_grup"].items():
        v = {n: math.hypot(*pas0["per_presa"][n]["soroll"]["tram_rms_rad"]) for n in G["preses"]}
        med = float(np.median(list(v.values())))
        for n, x in v.items():
            if x > FACTOR_SOROLL_FORA * med:
                fora[n] = f"soroll {math.degrees(x):.2f} graus > 2 x la mediana del grup ({math.degrees(med):.2f})"
    return fora


def inclosos_r2(pas0, fora_s):
    return {g: [n for n in G["preses"] if pas0["per_presa"][n]["sigma_amollada_s"] <= SIGMA_AMOLLADA_MAX_S
                and n not in fora_s] for g, G in pas0["per_grup"].items()}


def soroll_per_franja(pr, p, Ev2, franges=None) -> dict:
    """Suma de quadrats de la e filtrada (angles) als fotogrames de cada franja d'energia."""
    import soroll_en_moviment as sm
    meta = pr["meta"]
    t_fot, J, G, _ = sm.posicions_del_trk(meta)
    c = sm.cadena(t_fot, J, G, bool(meta.get("signe_invertit", False)))
    fin, ordre = int(meta["savgol"]["finestra_mostres"]), int(meta["savgol"]["ordre"])
    s, _ = sm.filtra(c["e"], fin, ordre, c["fps"])
    if s.shape[1] != pr["estats"].shape[1]:
        raise RuntimeError(f"{pr['nom']}: la graella de la llargada no es la del CSV")
    E = np.asarray(caos.energia_relativa(pr["estats"], p)) / Ev2
    sortida = {}
    for nom, (a, b) in (franges or FRANGES_R1).items():
        sel = (pr["t"] >= VEINS_DES_DE_S) & (E >= a) & (E < b)
        sortida[nom] = {"n": int(sel.sum()), "ss": [float(np.sum(s[k][sel] ** 2)) for k in range(2)],
                        "temps_s": float(sel.sum() / c["fps"])}
    return sortida


def _cfg_r1(banda, sostre, temps, dt=DT):
    return {"periodic": [True, True, False, False], "sep": [0, 1], "vel": [2, 3], "banda": tuple(banda),
            "theiler": int(round(THEILER_S / dt)), "cada": CADA_REF, "de_max": DE_MAX,
            "H": int(round(VEINS_FINS_S / dt)), "h0": int(round(VEINS_DES_DE_S / dt)), "dt": dt,
            "ln_sostre_3": math.log(sostre / 3.0), "temps": temps, "separacio_creuada": SEPARACIO_CREUADA_S}


def _entrada_r1(tirs, p, Ev2, w0, franja, dt=DT):
    """(Q, V, E, temps) de tirs (t des de l'amollada, X)."""
    Q, V, E = _veins_entrada(tirs, p, Ev2, w0, franja, dt)
    for v, e in zip(V, E):
        v &= e < franja[1]          # la franja es [a, b): el limit de dalt no hi entra
    return Q, V, E, [t for t, _ in tirs]


def descriu2(pas0) -> dict:
    """El pas 0 dels dos resultats. Cap corba de separacio: soroll, temps, sostres i recomptes."""
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    fora_s = fora_per_soroll(pas0)
    incl2 = inclosos_r2(pas0, fora_s)
    preses = preses_reals()
    r2 = {}
    for g, G in pas0["per_grup"].items():
        noms = incl2[g]
        Dstar = G["sostre_llindars_rad"] / 3.0
        rep = {n: np.radians(pas0["per_presa"][n]["angles_repos_graus"]) for n in noms}
        d0 = [float(dist_ang(rep[a], rep[b])) for i, a in enumerate(noms) for b in noms[i + 1:]]
        esc = escala(pas0["per_presa"], noms, {n: pas0["per_presa"][n]["sigma_amollada_s"] for n in noms},
                     G["sostre_llindars_rad"])
        sd = np.sqrt(np.mean([np.square(pas0["per_presa"][n]["soroll"]["tram_rms_rad"]) for n in noms], axis=0))
        r2[g] = {"preses": noms, "fora": [n for n in G["preses"] if n not in noms],
                 "D_estrella_rad": Dstar, "terra_rad": esc["terra_rad"],
                 "parelles": sum(x <= Dstar / math.e for x in d0), "parelles_totals": len(d0),
                 "D_diagnostic_rad": (np.geomspace(2 * esc["terra_rad"], Dstar, N_D).tolist()
                                      if 2 * esc["terra_rad"] < Dstar else None),
                 "soroll_validacio_rad": sd.tolist(), "rho": G["rho"]}
    # Resultat 1
    noms1 = [pr["nom"] for pr in preses if pr["nom"] not in fora_s]
    pr1 = [pr for pr in preses if pr["nom"] in noms1]
    per_presa = {}
    for pr in pr1:
        print(f"  {pr['nom']} ...", flush=True)
        per_presa[pr["nom"]] = soroll_per_franja(pr, p, Ev2)
    rng = np.random.default_rng(LLAVOR + 2)
    r1 = {}
    for nom, franja in FRANGES_R1.items():
        n = sum(per_presa[k][nom]["n"] for k in per_presa)
        if n == 0:
            r1[nom] = {"avaluable": False, "motiu": "cap presa hi passa"}
            continue
        rms = np.sqrt(np.array([sum(per_presa[k][nom]["ss"][c] for k in per_presa) for c in range(2)]) / n)
        terra = math.sqrt(2.0) * float(np.hypot(*rms))
        banda = [BANDA_D0_R1[0] * terra, BANDA_D0_R1[1] * terra]
        tirs = [(pr["t"], pr["estats"]) for pr in pr1]
        Q, V, E, T = _entrada_r1(tirs, p, Ev2, w0, franja)
        sv = [pr["estats"] for pr, v in zip(pr1, V) if v.any()]
        fv = [np.nonzero(v)[0] for v in V if v.any()]
        sostre = _sostre_desaparellat(sv, fv, rng) if len(sv) >= 2 else float("nan")
        cfg = _cfg_r1(banda, sostre, T)
        par = troba_veins(Q, V, E, periodic=cfg["periodic"], sep=cfg["sep"], vel=cfg["vel"], banda=cfg["banda"],
                          theiler=cfg["theiler"], cada=cfg["cada"], de_max=cfg["de_max"], temps=T,
                          separacio_creuada=SEPARACIO_CREUADA_S)
        r1[nom] = {"avaluable": True, "soroll_rms_rad": rms.tolist(), "terra_rad": terra, "banda_rad": banda,
                   "sostre_rad": sostre, "rang_e": math.log(sostre / 3 / np.mean(banda)),
                   "temps_total_s": float(sum(per_presa[k][nom]["temps_s"] for k in per_presa)),
                   "preses_amb_temps": sum(per_presa[k][nom]["n"] > 0 for k in per_presa),
                   "parelles": len(par), "tirs_a_les_parelles": len({x[0] for x in par} | {x[2] for x in par}),
                   "parelles_de_grups_diferents": sum(pr1[a]["grup"] != pr1[c]["grup"] for a, _, c, _ in par)}
    return {"eina": "eines/lyapunov_real.py descriu2", "pla": "documentacio/pla-analisi-caos.md (17/10, segona part)",
            "que_es": "PAS 0 dels dos resultats: SENSE cap corba de separacio (soroll, temps, sostres, recomptes)",
            "fora_per_soroll": fora_s, "resultat2": r2, "resultat1": r1}


def imprimeix_pas02(r):
    g = math.degrees
    print("\nFora per la regla del soroll: " + (", ".join(f"{k} ({v})" for k, v in r["fora_per_soroll"].items()) or "cap"))
    print("\nRESULTAT 2 (una sola D):")
    for k, v in r["resultat2"].items():
        print(f"  {k} graus: {len(v['preses'])} preses (fora: {', '.join(v['fora']) or 'cap'}); D* = {g(v['D_estrella_rad']):.1f} "
              f"graus, terra {g(v['terra_rad']):.2f}; {v['parelles']} de {v['parelles_totals']} parelles amb d0 <= D*/e; "
              f"soroll de validacio {', '.join(f'{g(x):.2f}' for x in v['soroll_validacio_rad'])} graus")
    print("\nRESULTAT 1 (franges d'energia, totes les preses barrejades):")
    for k, v in r["resultat1"].items():
        if not v["avaluable"]:
            print(f"  {k}: {v['motiu']}")
            continue
        print(f"  E/E_v2 {k}: {v['temps_total_s']:.0f} s de {v['preses_amb_temps']} preses; soroll rms "
              f"{', '.join(f'{g(x):.2f}' for x in v['soroll_rms_rad'])} graus, terra {g(v['terra_rad']):.2f}; franja de d0 "
              f"{g(v['banda_rad'][0]):.1f}-{g(v['banda_rad'][1]):.1f}; sostre {g(v['sostre_rad']):.1f} (rang {v['rang_e']:.2f} e); "
              f"{v['parelles']} parelles de {v['tirs_a_les_parelles']} tirs ({v['parelles_de_grups_diferents']} entre grups)")


def _orbites_franja(nom):
    """12 orbites de lambda(E) a les energies de la franja, triades a l'atzar amb llavor fixa."""
    ens = ENERGIES_LAMBDA_E[nom]
    o = [json.loads(x) for x in LAMBDA_E_ORBITES.read_text(encoding="utf-8").splitlines() if x.strip()]
    o = sorted([x for x in o if any(abs(x["energia_rao"] - e) < 1e-9 for e in ens)],
               key=lambda x: (x["energia_rao"], x["index"]))
    tria = np.random.default_rng(LLAVOR + len(nom)).choice(len(o), TIRS_R1, replace=False)
    return [o[k] for k in sorted(tria)]


def _resum_veins(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def valida2(pas0, pas02, processos=PROCESSOS) -> dict:
    """El pas 1 dels dos resultats. NOMES SIMULACIO."""
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    ref_b = json.loads(REGLA2_JSON.read_text(encoding="utf-8"))
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    out = {"resultat2": {}, "resultat1": {}}
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        # ------------------------------------------------ resultat 2
        for g, R in pas02["resultat2"].items():
            noms = R["preses"]
            N = len(noms)
            x0s = [np.array([*np.radians(pas0["per_presa"][n]["angles_repos_graus"]), 0.0, 0.0]) for n in noms]
            sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
            Dstar = R["D_estrella_rad"]
            sd, rho = np.array(R["soroll_validacio_rad"]), R["rho"]
            Y = pool.map(_sim_doble, [(x, DURADA_SIM_S, False) for x in x0s])
            Yf = pool.map(_sim_doble, [(x, DURADA_SIM_S, True) for x in x0s])
            th_s = [0.5 * (x[0] + x[1]) for x in x0s]
            Ys = pool.map(_sim_simple, [(th, DURADA_SIM_S) for th in th_s])
            xs_s = [np.array([th, th, 0.0, 0.0]) for th in th_s]
            par_all = [(a, b) for a in range(N) for b in range(a + 1, N)]
            d0 = np.array([float(dist_ang(x0s[a], x0s[b])) for a, b in par_all])
            d0s = np.array([float(dist_ang(xs_s[a], xs_s[b])) for a, b in par_all])
            sel = [k for k in range(len(par_all)) if d0[k] <= Dstar / math.e]
            sels = [k for k in range(len(par_all)) if d0s[k] <= Dstar / math.e]
            par, pars = [par_all[k] for k in sel], [par_all[k] for k in sels]
            G_lin = np.array(pool.map(_ref_ll, [(x0s[a], np.array([dif_angular(x0s[b][0], x0s[a][0]),
                                                                    dif_angular(x0s[b][1], x0s[a][1]), 0, 0]),
                                                 T_MAX_S) for a, b in par]))
            lam, fita, cens = lambdes_parelles(creuaments(tc, d0[sel, None] * G_lin, [Dstar]), d0[sel], [Dstar], T_MAX_S)
            ref = float(agrega(lam, fita, cens)[0][0])

            def corre(Yset, xset, prs, d0v, r, que):
                obs = [observa(y, x, None if r is None else sd, rho,
                               None if r is None else np.random.default_rng(rg.llavor_de(float(g), r, i, que, 21)))
                       for i, (y, x) in enumerate(zip(Yset, xset))]
                dts = np.zeros(N) if r is None else np.random.default_rng(rg.llavor_de(float(g), r, 22)).normal(0.0, sig)
                X = [a_la_graella(t, Xo, dts[i], tc) for i, (t, Xo) in enumerate(obs)]
                d = np.array([dist_ang(X[a], X[b]) for a, b in prs])
                return avalua_llindars(tc, d, d0v, prs, N, [Dstar], llavor=rg.llavor_de(float(g), 9 if r is None else r, que))
            ideal = corre(Y, x0s, par, d0[sel], None, 1)
            tots = [corre(Y, x0s, par, d0[sel], r, 1) for r in range(REALITZACIONS)]
            simples = [corre(Ys, xs_s, pars, d0s[sels], r, 2) for r in range(REALITZACIONS)] if pars else []
            fr = corre(Yf, x0s, par, d0[sel], None, 3)
            rel_i = ideal["lambda_D"][0] / ref - 1
            rel_t = [x["lambda_D"][0] / ref - 1 for x in tots]
            crit = {"LU1": _resum_criteri("LU1", abs(rel_i) <= COTES2["LU1"], f"sense soroll {ideal['lambda_D'][0]:.3f} "
                                          f"contra la referencia {ref:.3f} ({100 * rel_i:+.0f} %)"),
                    "LU2": _resum_criteri("LU2", sum(abs(x) <= COTES2["LU2"] for x in rel_t) >= 4,
                                          "amb soroll i amollada: " + ", ".join(f"{100 * x:+.0f} %" for x in rel_t)),
                    "LU3": _resum_criteri("LU3", ideal["fraccio_creuen"][0] >= FRACCIO_CREUEN,
                                          f"sense soroll creuen {100 * ideal['fraccio_creuen'][0]:.0f} %")}
            bm = ref_b["grups"][g].get("benettin", {})
            out["resultat2"][g] = {
                "D_estrella_graus": math.degrees(Dstar), "parelles": len(par), "referencia": ref,
                "ideal": ideal, "amb_soroll": tots, "simple_amb_soroll_info": simples, "friccio_de_prova_info": fr,
                "benettin_info": bm.get("lambda_mediana_caotiques"), "criteris": crit,
                "passa": all(c["passa"] for c in crit.values())}
            imprimeix_r2(g, out["resultat2"][g])

        # ------------------------------------------------ resultat 1
        for nom, R in pas02["resultat1"].items():
            if not R["avaluable"]:
                out["resultat1"][nom] = {"avaluable": False}
                continue
            franja = FRANGES_R1[nom]
            orb = _orbites_franja(nom)
            x0s = [np.array([*np.radians(o["angles_graus"]), 0.0, 0.0]) for o in orb]
            lam_b = [o["lambda_T"] for o in orb]
            f_caot = float(np.mean([o["mena"] == "caotica" for o in orb]))
            dur = float(np.clip(R["temps_total_s"] / TIRS_R1, *DURADA_R1_S))
            Y = pool.map(_sim_doble, [(x, dur, False) for x in x0s])
            Md = p.A + p.B
            th_s = [math.acos(max(-1.0, 1.0 - float(caos.energia_relativa(x, p)) / (p.g * Md))) for x in x0s]
            Ys = pool.map(_sim_simple, [(th, dur) for th in th_s])
            xs_s = [np.array([th, th, 0.0, 0.0]) for th in th_s]
            g_rho = RHO_DE_FRANJA[nom]
            rho = pas0["per_grup"][g_rho]["rho"]
            sd = np.array(R["soroll_rms_rad"])

            def tirs(Yset, xset, r, que):
                obs = [observa(y, x, None if r is None else sd, rho,
                               None if r is None else np.random.default_rng(rg.llavor_de(len(nom), r, i, que, 31)))
                       for i, (y, x) in enumerate(zip(Yset, xset))]
                return [(t[t >= 0], X[:, t >= 0]) for t, X in obs]

            def corre(Yset, xset, r, que):
                tt = tirs(Yset, xset, r, que)
                Q, V, E, T = _entrada_r1(tt, p, Ev2, w0, franja)
                cfg = _cfg_r1(R["banda_rad"], R["sostre_rad"], T)
                return avalua_veins(Q, V, E, cfg, llavor=rg.llavor_de(len(nom), 9 if r is None else r, que))
            ideal = corre(Y, x0s, None, 1)
            tots = [corre(Y, x0s, r, 1) for r in range(REALITZACIONS)]
            simples = [corre(Ys, xs_s, r, 2) for r in range(REALITZACIONS)]
            # La referencia: FTLE infinitesimal des dels estats de debo de les parelles ideals.
            pi = ideal["_parelles"]
            mostra = (np.random.default_rng(LLAVOR + 7).choice(len(pi), VEINS_REF_MAX, replace=False)
                      if len(pi) > VEINS_REF_MAX else np.arange(len(pi)))
            feines = []
            for m in mostra:
                i, ka, j, kb = pi[m]
                xa, xb = Y[i][:, ka], Y[j][:, kb]
                feines.append((xa, np.array([dif_angular(xb[0], xa[0]), dif_angular(xb[1], xa[1]),
                                             xb[2] - xa[2], xb[3] - xa[3]]), VEINS_FINS_S))
            Gv = np.log(np.array(pool.map(_ref_ll, feines))) if feines else None
            corba = Gv.mean(axis=0) if Gv is not None else None

            def refp(fin):
                if corba is None or not fin:
                    return None
                a, b = int(round(fin[0] / DT)), int(round(fin[1] / DT))
                return float(np.polyfit(np.arange(a, b + 1) * DT, corba[a:b + 1], 1)[0]) if b - a >= 2 else None
            ri = refp(ideal["finestra_s"])
            lb = float(np.mean(lam_b))

            def dins(x, ref, cota):
                return ref is not None and ref != 0 and abs(x / ref - 1) <= cota
            n3 = sum(d["ok"] and dins(d["lambda"], refp(d["finestra_s"]), COTES2["VE3"]) for d in tots)
            n4 = sum(d["ok"] for d in tots)

            def ve5(d, s):
                sd_ = (d.get("bootstrap") or {}).get("sigma") or 0.0
                ss_ = (s.get("bootstrap") or {}).get("sigma") or 0.0
                ls = s["lambda"] if s["parelles"] >= 3 else 0.0
                return ls <= COTES2["VE5"] * d["lambda"] and d["lambda"] - ls >= 2.0 * math.hypot(sd_, ss_)
            n5 = sum(ve5(d, s) for d, s in zip(tots, simples))
            n6 = sum(d["parelles"] >= 100 and d["tirs"] >= 8 for d in tots)
            crit = {
                "VE1": _resum_criteri("VE1", dins(ideal["lambda"], ri, COTES2["VE1"]),
                                      f"sense soroll {ideal['lambda']:.3f} contra la FTLE {ri}"),
                "VE2": (_resum_criteri("VE2", dins(ideal["lambda"], lb, COTES2["VE2"]),
                                       f"sense soroll {ideal['lambda']:.3f} contra Benettin {lb:.3f} ({100 * f_caot:.0f} % caotiques)")
                        if f_caot >= FRACCIO_CAOTIQUES_VE2 else
                        {"criteri": "VE2", "passa": None, "detall": f"no s'aplica: {100 * f_caot:.0f} % caotiques "
                                                                    f"(Benettin {lb:.3f})"}),
                "VE3": _resum_criteri("VE3", n3 >= 4, f"{n3} de 5 a +-25 % de la FTLE"),
                "VE4": _resum_criteri("VE4", ideal["ok"] and n4 >= 4,
                                      f"creixement sense soroll {ideal['creixement_ln']:.2f}; >= 1 a {n4} de 5"),
                "VE5": _resum_criteri("VE5", n5 >= 4, f"{n5} de 5 amb el simple <= 0,3 x el doble i a >= 2 sigma"),
                "VE6": _resum_criteri("VE6", ideal["parelles"] >= 100 and ideal["tirs"] >= 8 and n6 >= 4
                                      and R["parelles"] >= 100 and R["tirs_a_les_parelles"] >= 8,
                                      f"simulacio: {ideal['parelles']} parelles de {ideal['tirs']} tirs (>= 100 de >= 8 a "
                                      f"{n6} de 5); preses reals (recompte): {R['parelles']} de {R['tirs_a_les_parelles']}"),
            }
            out["resultat1"][nom] = {
                "avaluable": True, "durada_tirs_s": dur, "energies": sorted({o["energia_rao"] for o in orb}),
                "benettin_mitjana": lb, "fraccio_caotiques": f_caot, "referencia_ftle_ideal": ri,
                "ideal": {**_resum_veins(ideal), "referencia": ri},
                "amb_soroll": [{**_resum_veins(d), "referencia": refp(d["finestra_s"])} for d in tots],
                "simple_amb_soroll": [_resum_veins(s) for s in simples], "criteris": crit,
                "passa": all(c["passa"] is not False for c in crit.values())}
            imprimeix_r1(nom, out["resultat1"][nom])
    return out


def imprimeix_r2(g, b):
    print(f"\n-- RESULTAT 2, {g} graus: D* = {b['D_estrella_graus']:.1f} graus, {b['parelles']} parelles. NOMES SIMULACIO.")
    print(f"   referencia linealitzada {b['referencia']:.3f}; sense soroll {b['ideal']['lambda_D'][0]:.3f}; amb soroll i "
          f"amollada: {', '.join(f'{x['lambda_D'][0]:.3f}+-{x['bootstrap']['sigma_D'][0]:.3f}' for x in b['amb_soroll'])}")
    print(f"   (info) simple amb soroll: {', '.join(f'{x['lambda_D'][0]:.2f} ({100 * x['fraccio_creuen'][0]:.0f} % creuen)' for x in b['simple_amb_soroll_info'])}")
    print(f"   (info) friccio DE PROVA, sense soroll: {b['friccio_de_prova_info']['lambda_D'][0]:.3f}; Benettin {b['benettin_info']}")
    for k, c in b["criteris"].items():
        print(f"   {'ok ' if c['passa'] else 'NO '} {k}: {c['detall']}")


def imprimeix_r1(nom, b):
    print(f"\n-- RESULTAT 1, franja E/E_v2 {nom}: 12 orbites de lambda(E) a {b['energies']}, {b['durada_tirs_s']:.1f} s cada tir. "
          "NOMES SIMULACIO, sense friccio.")
    i = b["ideal"]
    print(f"   Benettin {b['benettin_mitjana']:.3f} ({100 * b['fraccio_caotiques']:.0f} % caotiques); FTLE de la finestra "
          f"{b['referencia_ftle_ideal']}; sense soroll {i['lambda']:.3f} (finestra {i['finestra_s']}, creix "
          f"{i['creixement_ln']:.2f}, {i['parelles']} parelles de {i['tirs']} tirs)")
    for d, s in zip(b["amb_soroll"], b["simple_amb_soroll"]):
        print(f"      amb soroll: {d['lambda']:.3f} +- {(d.get('bootstrap') or {}).get('sigma')} (FTLE {d['referencia']}, creix "
              f"{d['creixement_ln']:.2f}, {d['parelles']} parelles) | simple {s['lambda']:.3f} ({s['parelles']} parelles)")
    for k, c in b["criteris"].items():
        e = "-- " if c["passa"] is None else ("ok " if c["passa"] else "NO ")
        print(f"   {e} {k}: {c['detall']}")


FIRMA = ARREL / "dades" / "versemblant" / "firma.json"


def z_veins(pas0, pas02, reals_json, *, amb_friccio_de_prova=False, processos=PROCESSOS) -> dict:
    """La Z amb el metode dels veins: real (el resultat del pas 3) contra simulat des de les
    mateixes condicions inicials, amb el mateix soroll i el mateix metode.

    PANY: es nega a funcionar, ABANS D'OBRIR RES, si no hi ha la firma de X, Y i Z. La regla del
    veredicte es la de eines/criteri_z.py (2 sigma; «no avaluable» si l'error passa del 20 %).
    La friccio: sense, o DE PROVA; amb els coeficients decidits caldra afegir-los aqui.
    """
    if not FIRMA.exists():
        raise SystemExit(f"Pany tancat: no hi ha {FIRMA.relative_to(ARREL)}. La Z compara simulacio i "
                         "realitat, i aixo no es fa abans de firmar X, Y i Z.")
    import criteri_z
    sim = _r1_simulat(pas0, pas02, amb_friccio=amb_friccio_de_prova, processos=processos)
    real = reals_json["resultat1"]["1,7-3,5"]
    v = criteri_z.veredicte(real["lambda"], real["bootstrap"]["sigma"], sim["lambda"], sim["sigma"])
    return {"real": {"lambda": real["lambda"], "sigma": real["bootstrap"]["sigma"]}, "simulat": sim, **v,
            "friccio": "DE PROVA" if amb_friccio_de_prova else "cap"}


def _r1_simulat(pas0, pas02, *, amb_friccio, franja_nom="1,7-3,5", durada=30.0, processos=PROCESSOS) -> dict:
    """El R1 sobre tirs simulats des dels angles de repos de les preses del R1, amb el soroll de la
    franja: la mediana de 5 realitzacions i la sigma del bootstrap de la primera. NOMES SIMULACIO."""
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    R = pas02["resultat1"][franja_nom]
    fora_s = pas02["fora_per_soroll"]
    noms = [n for g in ("90", "150") for n in pas0["per_grup"][g]["preses"] if n not in fora_s]
    x0s = [np.array([*np.radians(pas0["per_presa"][n]["angles_repos_graus"]), 0.0, 0.0]) for n in noms]
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        Y = pool.map(_sim_doble, [(x, durada, amb_friccio) for x in x0s])
    rho = pas0["per_grup"][RHO_DE_FRANJA[franja_nom]]["rho"]
    sd = np.array(R["soroll_rms_rad"])
    res = []
    for r in range(REALITZACIONS):
        obs = [observa(y, x, sd, rho, np.random.default_rng(rg.llavor_de(51, r, i))) for i, (y, x) in enumerate(zip(Y, x0s))]
        Q, V, E, T = _entrada_r1([(t[t >= 0], X[:, t >= 0]) for t, X in obs], p, Ev2, w0, FRANGES_R1[franja_nom])
        res.append(avalua_veins(Q, V, E, _cfg_r1(R["banda_rad"], R["sostre_rad"], T), llavor=rg.llavor_de(52, r)))
    return {"lambda": float(np.median([x["lambda"] for x in res])), "sigma": res[0]["bootstrap"]["sigma"],
            "realitzacions": [x["lambda"] for x in res], "tirs": noms}


def friccio_r1(pas0, pas02, validacio2, franja_nom="1,7-3,5", durada=30.0, processos=PROCESSOS) -> dict:
    """La sensibilitat del R1 a la friccio DE PROVA (17/10). NOMES SIMULACIO.

    Els tirs surten dels angles de repos de les preses del R1 dels grups que hi passen (90 i
    150), i travessen la franja perdent energia, com les reals. Criteri: sense soroll, a +-15 %
    de la lambda sense friccio de la validacio.
    """
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    R = pas02["resultat1"][franja_nom]
    franja = FRANGES_R1[franja_nom]
    fora_s = pas02["fora_per_soroll"]
    noms = [n for g in ("90", "150") for n in pas0["per_grup"][g]["preses"] if n not in fora_s]
    x0s = [np.array([*np.radians(pas0["per_presa"][n]["angles_repos_graus"]), 0.0, 0.0]) for n in noms]
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        Y = pool.map(_sim_doble, [(x, durada, True) for x in x0s])
    rho = pas0["per_grup"][RHO_DE_FRANJA[franja_nom]]["rho"]
    sd = np.array(R["soroll_rms_rad"])

    def corre(r):
        obs = [observa(y, x, None if r is None else sd, rho,
                       None if r is None else np.random.default_rng(rg.llavor_de(41, r, i)))
               for i, (y, x) in enumerate(zip(Y, x0s))]
        tirs = [(t[t >= 0], X[:, t >= 0]) for t, X in obs]
        Q, V, E, T = _entrada_r1(tirs, p, Ev2, w0, franja)
        return avalua_veins(Q, V, E, _cfg_r1(R["banda_rad"], R["sostre_rad"], T),
                            llavor=rg.llavor_de(42, 9 if r is None else r))
    ideal = corre(None)
    tots = [corre(r) for r in range(REALITZACIONS)]
    sense = validacio2["resultat1"][franja_nom]["ideal"]["lambda"]
    rel = ideal["lambda"] / sense - 1
    temps = [float(np.sum(v) * DT) for v in _entrada_r1([(t[t >= 0], X[:, t >= 0]) for t, X in
                                                          [observa(y, x, None, rho, None) for y, x in zip(Y, x0s)]],
                                                         p, Ev2, w0, franja)[1]]
    return {"que_es": "SENSIBILITAT a la friccio DE PROVA (C = 0,00794, eps = 1e-3): NOMES SIMULACIO, no per a la memoria",
            "franja": franja_nom, "tirs": noms, "temps_a_la_franja_s": temps,
            "ideal": _resum_veins(ideal), "amb_soroll": [_resum_veins(d) for d in tots],
            "sense_friccio_validacio": sense, "canvi_relatiu": rel,
            "criteri": _resum_criteri("FR1", abs(rel) <= 0.15, f"sense soroll {ideal['lambda']:.3f} amb friccio de prova "
                                      f"contra {sense:.3f} sense friccio ({100 * rel:+.1f} %); cal +-15 %")}


def avalua_parelles(Q, par, cfg, n_tirs, *, n_boot=REMOSTREJOS, llavor=0) -> dict:
    """La corba, la lambda i el bootstrap per tirs d'un conjunt de parelles JA TRIADES."""
    base = {"parelles": len(par), "tirs": len({p[0] for p in par} | {p[2] for p in par})}
    if len(par) < 3:
        return {**base, "ok": False, "lambda": 0.0, "creixement_ln": 0.0, "finestra_s": None,
                "motiu": "menys de tres parelles"}
    D = segueix(Q, par, periodic=cfg["periodic"], sep=cfg["sep"], H=cfg["H"])
    lnd = np.log(np.maximum(D, 1e-300))
    lnr = lnd - lnd[:, :1]
    r = {**base, **pendent_veins(lnr, lnd, np.ones(len(par)), cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])}
    rng = np.random.default_rng(llavor)
    pt = [(p[0], p[2]) for p in par]
    v, falla = [], 0
    for _ in range(n_boot):
        w = pesos_bootstrap(pt, rng.integers(0, n_tirs, n_tirs), n_tirs)
        if w.sum() <= 0:
            falla += 1
            continue
        b = pendent_veins(lnr, lnd, w, cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])
        v.append(b["lambda"])
        falla += not b["ok"]
    r["bootstrap"] = {"remostrejos": len(v), "sigma": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                      "sense_lambda": falla}
    return r


def robustesa(pas0, pas02, principal: float) -> dict:
    """La robustesa del R1 (1,7-3,5) a les preses reals. Criteris al pla, 22/10. Darrere del pany."""
    if not APROVAT.exists() or "1,7-3,5" not in json.loads(APROVAT.read_text(encoding="utf-8")).get("resultat1", []):
        raise SystemExit("Pany tancat: el R1 a 1,7-3,5 no esta aprovat per a les preses reals.")
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    fora_s = pas02["fora_per_soroll"]
    pr1 = [pr for pr in preses_reals() if pr["nom"] not in fora_s]
    tirs = [(pr["t"], pr["estats"]) for pr in pr1]
    grup = [pr["grup"] for pr in pr1]
    R = pas02["resultat1"]["1,7-3,5"]
    franja = FRANGES_R1["1,7-3,5"]
    Q, V, E, T = _entrada_r1(tirs, p, Ev2, w0, franja)
    sortida = {}

    def resum(nom, r):
        s = (r.get("bootstrap") or {}).get("sigma")
        aval = r["parelles"] >= 100 and r["ok"] and s is not None
        comp = aval and abs(r["lambda"] - principal) <= 2.0 * s
        sortida[nom] = {"lambda": r["lambda"], "sigma": s, "parelles": r["parelles"], "tirs": r["tirs"],
                        "creixement_ln": r["creixement_ln"], "finestra_s": r["finestra_s"],
                        "avaluable": bool(aval), "compatible": bool(comp) if aval else None}
        print(f"  {nom:<28} lambda {r['lambda']:.3f} +- {s if s is None else round(s, 3)}  "
              f"({r['parelles']} parelles de {r['tirs']} tirs, creix {r['creixement_ln']:.2f})  -> "
              f"{'no avaluable' if not aval else ('compatible' if comp else 'NO COMPATIBLE')}", flush=True)

    # 1. Per grups: les MATEIXES parelles del principal, partides.
    cfg = _cfg_r1(R["banda_rad"], R["sostre_rad"], T)
    par = troba_veins(Q, V, E, periodic=cfg["periodic"], sep=cfg["sep"], vel=cfg["vel"], banda=cfg["banda"],
                      theiler=cfg["theiler"], cada=cfg["cada"], de_max=cfg["de_max"], temps=T,
                      separacio_creuada=SEPARACIO_CREUADA_S)
    tot = avalua_parelles(Q, par, cfg, len(Q), llavor=LLAVOR)
    print(f"  (el principal, refet: {tot['lambda']:.3f}; publicat {principal:.3f})")
    for nom, f in (("parelles 90-90", lambda a, c: grup[a] == grup[c] == "90"),
                   ("parelles 150-150", lambda a, c: grup[a] == grup[c] == "150"),
                   ("parelles creuades 90-150", lambda a, c: grup[a] != grup[c])):
        resum(nom, avalua_parelles(Q, [x for x in par if f(x[0], x[2])], cfg, len(Q), llavor=LLAVOR))
    # 2. Per energia: cada subfranja amb el seu soroll, la seva franja de d0 i el seu sostre.
    sub = {"1,7-2,5": (1.7, 2.5), "2,5-3,5": (2.5, 3.5)}
    soroll = [soroll_per_franja(pr, p, Ev2, sub) for pr in pr1]
    rng = np.random.default_rng(LLAVOR + 3)
    for nom, fr in sub.items():
        n = sum(s[nom]["n"] for s in soroll)
        rms = np.sqrt(np.array([sum(s[nom]["ss"][c] for s in soroll) for c in range(2)]) / n)
        terra = math.sqrt(2.0) * float(np.hypot(*rms))
        Qs, Vs, Es, Ts = _entrada_r1(tirs, p, Ev2, w0, fr)
        sv = [pr["estats"] for pr, v in zip(pr1, Vs) if v.any()]
        fv = [np.nonzero(v)[0] for v in Vs if v.any()]
        cfg_s = _cfg_r1([BANDA_D0_R1[0] * terra, BANDA_D0_R1[1] * terra], _sostre_desaparellat(sv, fv, rng), Ts)
        resum(f"subfranja {nom}", avalua_veins(Qs, Vs, Es, cfg_s, llavor=LLAVOR))
    # 3. Per les tries.
    terra = R["terra_rad"]
    for nom, banda, sep_c in (("d0 a [3; 4,5] x terra", [3.0 * terra, 4.5 * terra], SEPARACIO_CREUADA_S),
                              ("d0 a [4,5; 6] x terra", [4.5 * terra, 6.0 * terra], SEPARACIO_CREUADA_S),
                              ("1,0 s entre tirs", R["banda_rad"], 1.0)):
        c = _cfg_r1(banda, R["sostre_rad"], T)
        c["separacio_creuada"] = sep_c
        resum(nom, avalua_veins(Q, V, E, c, llavor=LLAVOR))
    av = [v for v in sortida.values() if v["avaluable"]]
    veredicte = "robust" if av and all(v["compatible"] for v in av) else "no robust"
    print(f"\n  VEREDICTE: {veredicte} ({sum(v['compatible'] for v in av)} de {len(av)} variants avaluables "
          f"compatibles; {len(sortida) - len(av)} no avaluables)")
    return {"que_es": "ROBUSTESA del R1 (E/E_v2 1,7-3,5) a les preses reals, real contra real (pla, 22/10)",
            "principal": principal, "principal_refet": tot["lambda"], "variants": sortida, "veredicte": veredicte}


def biaix(pas0, pas02, reals_json, sortejos_r2=200, sortejos_r1=50) -> dict:
    """La sensibilitat del R1 i del R2 a un biaix de theta1 constant dins de cada presa (pla, 22/10).

    A cada sorteig, cada presa rep o_i ~ N(0, sigma_b) a theta1 de TOTA la presa (repos i moviment).
    Real contra real, darrere del pany. No corregeix res: mesura quant es mouen els resultats.
    """
    if not APROVAT.exists():
        raise SystemExit("Pany tancat: no hi ha l'aprovacio de les preses reals.")
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    preses = {pr["nom"]: pr for pr in preses_reals()}
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    rng = np.random.default_rng(LLAVOR + 22)
    out = {"que_es": "SENSIBILITAT al biaix de theta1 (pla, 22/10): real contra real", "R2_150": {}, "R1": {}}

    # ---------- R2, 150 graus
    R = pas02["resultat2"]["150"]
    noms = R["preses"]
    N = len(noms)
    Dstar = R["D_estrella_rad"]
    rep0 = [preses[n]["repos_rad"].copy() for n in noms]
    d0_nom = np.array([float(dist_ang(rep0[a], rep0[b])) for a in range(N) for b in range(a + 1, N)])
    par = [(a, b) for a in range(N) for b in range(a + 1, N)]
    sel = [k for k in range(len(par)) if d0_nom[k] <= Dstar / math.e]
    par = [par[k] for k in sel]
    X0 = [a_la_graella(preses[n]["t"], preses[n]["estats"], 0.0, tc) for n in noms]
    publicat_r2 = reals_json["resultat2"]["150"]["lambda"]

    def lam_r2(o, angles=(0,)):
        X = [x.copy() for x in X0]
        rp = [r.copy() for r in rep0]
        for i in range(N):
            for k in angles:
                X[i][k] += o[i, k]
                rp[i][k] += o[i, k]
        d0 = np.array([float(dist_ang(rp[a], rp[b])) for a, b in par])
        d = np.array([dist_ang(X[a], X[b]) for a, b in par])
        return avalua_llindars(tc, d, np.maximum(d0, 1e-6), par, N, [Dstar], n_boot=0)["lambda_D"][0]
    for sb, angles, nom in ((1.0, (0,), "1"), (2.0, (0,), "2"), (3.0, (0,), "3"), (2.0, (0, 1), "2_th1_th2")):
        v = [lam_r2(np.radians(rng.normal(0.0, sb, (N, 2))), angles) for _ in range(sortejos_r2)]
        out["R2_150"][nom] = {"mitjana": float(np.mean(v)), "desplacament": float(np.mean(v) - publicat_r2),
                              "dispersio": float(np.std(v, ddof=1)), "p16_84": np.percentile(v, [16, 84]).tolist()}
        print(f"  R2 150, sigma_b {nom}: mitjana {np.mean(v):.3f} (publicat {publicat_r2:.3f}, desplacament "
              f"{np.mean(v) - publicat_r2:+.3f}), dispersio {np.std(v, ddof=1):.3f}", flush=True)

    # ---------- R1, veins a 1,7-3,5
    fora_s = pas02["fora_per_soroll"]
    pr1 = [pr for n, pr in preses.items() if n not in fora_s]
    R1 = pas02["resultat1"]["1,7-3,5"]
    publicat_r1 = reals_json["resultat1"]["1,7-3,5"]["lambda"]

    def lam_r1(o, angles=(0,)):
        tirs = []
        for i, pr in enumerate(pr1):
            X = pr["estats"].copy()
            for k in angles:
                X[k] = X[k] + o[i, k]
            tirs.append((pr["t"], X))
        Q, V, E, T = _entrada_r1(tirs, p, Ev2, w0, FRANGES_R1["1,7-3,5"])
        r = avalua_veins(Q, V, E, _cfg_r1(R1["banda_rad"], R1["sostre_rad"], T), n_boot=0)
        return r["lambda"], r["parelles"]
    for sb, angles, nom in ((1.0, (0,), "1"), (2.0, (0,), "2"), (3.0, (0,), "3"), (2.0, (0, 1), "2_th1_th2")):
        res = [lam_r1(np.radians(rng.normal(0.0, sb, (len(pr1), 2))), angles) for _ in range(sortejos_r1)]
        v = [x[0] for x in res]
        out["R1"][nom] = {"mitjana": float(np.mean(v)), "desplacament": float(np.mean(v) - publicat_r1),
                          "dispersio": float(np.std(v, ddof=1)), "p16_84": np.percentile(v, [16, 84]).tolist(),
                          "parelles_mediana": float(np.median([x[1] for x in res]))}
        print(f"  R1 veins, sigma_b {nom}: mitjana {np.mean(v):.3f} (publicat {publicat_r1:.3f}, desplacament "
              f"{np.mean(v) - publicat_r1:+.3f}), dispersio {np.std(v, ddof=1):.3f}, parelles "
              f"{np.median([x[1] for x in res]):.0f}", flush=True)
    sig = {"R2_150": reals_json["resultat2"]["150"]["bootstrap"]["sigma_D"][0],
           "R1": reals_json["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"]}
    out["dominant"] = {k: {nom: bool(abs(x["desplacament"]) > sig[k] or x["dispersio"] > sig[k])
                           for nom, x in out[k].items()} for k in ("R2_150", "R1")}
    out["sigma_estadistica"] = sig
    return out


def r2_de_preses(preses, noms, sig, Dstar, D_diagnostic=None, *, complet=True) -> dict:
    """El R2 d'un grup sobre unes preses {nom: {t, estats, repos_rad}}, amb t des de l'amollada mesurada: la lambda*,
    el bootstrap per tirs, la sensibilitat a l'alineacio i, si complet, la de la d0 i el diagnostic de D.

    Era el cos de reals2. Surt a part el 26/10 perque els bessons de la Z (z_r2) passin pel mateix estimador que les
    preses reals; la ZR0 comprova que reals2 hi dona el mateix bit a bit.
    """
    N = len(noms)
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    rep = [preses[n]["repos_rad"] for n in noms]
    par_all = [(a, b) for a in range(N) for b in range(a + 1, N)]
    d0 = np.array([float(dist_ang(rep[a], rep[b])) for a, b in par_all])
    sel = [k for k in range(len(par_all)) if d0[k] <= Dstar / math.e]
    par = [par_all[k] for k in sel]

    def lam_amb(zeros, d0v, n_boot=0, Ds=(Dstar,)):
        X = [a_la_graella(preses[n]["t"], preses[n]["estats"], zeros[i], tc) for i, n in enumerate(noms)]
        d = np.array([dist_ang(X[a], X[b]) for a, b in par])
        return avalua_llindars(tc, d, d0v, par, N, list(Ds), n_boot=n_boot, llavor=LLAVOR)
    base = lam_amb(np.zeros(N), d0[sel], REMOSTREJOS)
    rng = np.random.default_rng(LLAVOR + 11)
    mc_al = [lam_amb(rng.normal(0.0, sig), d0[sel])["lambda_D"][0] for _ in range(200)]
    r = {"lambda": base["lambda_D"][0], "fraccio_creuen": base["fraccio_creuen"][0], "bootstrap": base["bootstrap"],
         "mc_alineacio": {"desviacio": float(np.std(mc_al, ddof=1)), "mitjana": float(np.mean(mc_al))}}
    if complet:
        mc_d0 = {}
        for sb in (0.5, 1.0):
            v = []
            for _ in range(200):
                rp = [x + np.radians(rng.normal(0.0, sb, 2)) for x in rep]
                d0p = np.array([float(dist_ang(rp[a], rp[b])) for a, b in par_all])[sel]
                v.append(lam_amb(np.zeros(N), np.maximum(d0p, 1e-6))["lambda_D"][0])
            mc_d0[f"{sb:g}"] = {"mitjana": float(np.mean(v)), "desviacio": float(np.std(v, ddof=1))}
        r["mc_d0"] = mc_d0
        r["diagnostic_lambda_D"] = lam_amb(np.zeros(N), d0[sel], 0, D_diagnostic) if D_diagnostic else None
    r["D_estrella_graus"] = math.degrees(Dstar)
    r["parelles"] = len(par)
    return r


def reals2(pas0, pas02, preses=None, guarda_valors=False) -> dict:
    """El pas 3 dels dos resultats. TANCAT AMB PANY: nomes el que diu l'aprovacio.

    preses (24/10): {nom: presa} ja corregides (corregit); si no, preses_reals(). guarda_valors: el
    bootstrap del R1 torna tambe els valors (per a l'horitzo).
    """
    if not APROVAT.exists():
        raise SystemExit(f"Pany tancat: no hi ha {APROVAT.relative_to(ARREL)}. Les corbes de separacio reals nomes "
                         "es miren quan en Guillem aprova els resultats (pas 2).")
    aprovat = json.loads(APROVAT.read_text(encoding="utf-8"))
    p = _parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    if preses is None:
        preses = {pr["nom"]: pr for pr in preses_reals()}
    out = {"aprovacio": aprovat, "resultat2": {}, "resultat1": {}}
    for g in aprovat.get("resultat2", []):
        R = pas02["resultat2"][g]
        sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in R["preses"]])
        out["resultat2"][g] = r2_de_preses(preses, R["preses"], sig, R["D_estrella_rad"], R["D_diagnostic_rad"])
    fora_s = pas02["fora_per_soroll"]
    pr1 = [pr for n, pr in preses.items() if n not in fora_s]
    for nom in aprovat.get("resultat1", []):
        R = pas02["resultat1"][nom]
        Q, V, E, T = _entrada_r1([(pr["t"], pr["estats"]) for pr in pr1], p, Ev2, w0, FRANGES_R1[nom])
        r = avalua_veins(Q, V, E, _cfg_r1(R["banda_rad"], R["sostre_rad"], T), llavor=LLAVOR,
                         guarda_valors=guarda_valors)
        out["resultat1"][nom] = {**_resum_veins(r), "grups_de_les_parelles": sorted(
            {f"{pr1[a]['grup']}-{pr1[c]['grup']}" for a, _, c, _ in r["_parelles"]})}
    return out


# =============================================================== 23/10: l'horitzo de prediccio

HORITZO_D0_GRAUS = (0.3, 1.0)      # la precisio de les condicions inicials
HORITZO_TOL_GRAUS = (3.0, 5.0)     # la X provisional, i 5 graus
HORITZO_PERCENTILS = (2.5, 16.0, 50.0, 84.0, 97.5)


def horitzons(lams, central, d0s_graus=HORITZO_D0_GRAUS, tols_graus=HORITZO_TOL_GRAUS) -> dict:
    """t = ln(D_tol/D0)/lambda a cada remostreig. Una lambda <= 0 es un horitzo INFINIT: no es llenca.

    Els percentils son de la mostra (inverted_cdf), perque hi pot haver infinits.
    """
    lams = np.asarray(lams, dtype=float)
    positives = lams > 0
    out = {"remostrejos": int(len(lams)), "fraccio_lambda_no_positiva": float(np.mean(~positives))}
    for d0 in d0s_graus:
        for tol in tols_graus:
            lnR = math.log(tol / d0)
            t = np.full(len(lams), math.inf)
            t[positives] = lnR / lams[positives]
            q = np.percentile(t, HORITZO_PERCENTILS, method="inverted_cdf")
            out[f"d0_{d0:g}_tol_{tol:g}"] = {
                "D0_graus": d0, "D_tol_graus": tol, "ln_R": lnR,
                "central_s": lnR / central if central > 0 else math.inf,
                **{f"p{p:g}_s": float(x) for p, x in zip(HORITZO_PERCENTILS, q)}}
    return out


def horitzo(pas02, reals_json, corregit=False) -> dict:
    """L'horitzo de prediccio amb la lambda real del R1 (pla, 23/10). Real sol, darrere del pany. Amb corregit (28/10),
    les preses de preses_corregides(), i la H1 es mira contra el JSON de dues_parts del 27/10."""
    global _OBERTS
    if not APROVAT.exists() or "1,7-3,5" not in json.loads(APROVAT.read_text(encoding="utf-8")).get("resultat1", []):
        raise SystemExit("Pany tancat: el R1 a 1,7-3,5 no esta aprovat per a les preses reals.")
    # H3: res no pot simular, i s'apunta que s'obre.
    from pendol import simple, simulador

    def prohibit(*_a, **_k):
        raise RuntimeError("H3: l'horitzo real ha intentat simular")
    originals = (simulador.simula, simple.simula)
    simulador.simula = simple.simula = prohibit
    sys.addaudithook(_registra)
    _OBERTS = []
    try:
        p = _parametres()
        w0 = omega_zero(p)
        Ev2 = caos.energia_volteig_2(p)
        fora_s = pas02["fora_per_soroll"]
        totes = list(preses_corregides()[0].values()) if corregit else preses_reals()
        pr1 = [pr for pr in totes if pr["nom"] not in fora_s]
        R = pas02["resultat1"]["1,7-3,5"]
        Q, V, E, T = _entrada_r1([(pr["t"], pr["estats"]) for pr in pr1], p, Ev2, w0, FRANGES_R1["1,7-3,5"])
        r = avalua_veins(Q, V, E, _cfg_r1(R["banda_rad"], R["sostre_rad"], T), llavor=LLAVOR, guarda_valors=True)
    finally:
        simulador.simula, simple.simula = originals
        oberts, _OBERTS = _OBERTS, None
    pub = reals_json["resultat1"]["1,7-3,5"]
    lam, sig = r["lambda"], r["bootstrap"]["sigma"]
    e_lam, e_sig = abs(lam / pub["lambda"] - 1), abs(sig / pub["bootstrap"]["sigma"] - 1)
    h1 = e_lam <= 1e-12 and e_sig <= 1e-12
    print(f"H1. El R1 refet: lambda {lam:.16f} (publicada {pub['lambda']:.16f}, {e_lam:.1e}); sigma {sig:.16f} "
          f"(publicada {pub['bootstrap']['sigma']:.16f}, {e_sig:.1e}) -> {'ok' if h1 else 'FALLA'}", flush=True)
    arrel = ARREL.resolve()
    rel = sorted({str(Path(x).resolve().relative_to(arrel)).replace("\\", "/") for x in oberts
                  if Path(x).resolve().is_relative_to(arrel)})
    prohibits = [x for x in rel if x.startswith(("dades/sim/", "figures/"))]
    h3 = not prohibits
    print(f"H3. Sense simular (la simulacio estava prohibida i ha acabat); de dades/sim/ i figures/ ha obert: "
          f"{prohibits or 'res'} -> {'ok' if h3 else 'FALLA'}", flush=True)
    out = {"eina": "eines/lyapunov_real.py horitzo", "pla": "documentacio/pla-analisi-caos.md (23/10, l'horitzo)",
           "que_es": "L'HORITZO DE PREDICCIO amb la lambda real del R1 (veins, E/E_v2 1,7-3,5): t = ln(D_tol/D0)/lambda "
                     "a cada remostreig del bootstrap de tirs. REAL SOL: cap simulacio, i no es compara amb el feix",
           "angles": ("corregits amb comparar.corregeix_presa (preses_corregides, 28/10)" if corregit
                      else "sense corregir"),
           "criteris": {"H1": {"passa": bool(h1), "lambda": lam, "sigma": sig, "error_lambda": e_lam,
                               "error_sigma": e_sig},
                        "H3": {"passa": bool(h3), "fitxers_oberts": rel}}}
    if not h1:
        print("\nH1 FALLA: no es calcula cap horitzo. Un criteri que falla NO es relaxa: s'atura i s'explica.")
        return out
    b = r["bootstrap"]
    out["lambda"] = {"valor": lam, "sigma": sig, "remostrejos": b["remostrejos"], "sense_creixement": b["sense_lambda"],
                     "parelles": r["parelles"], "tirs": r["tirs"], "finestra_s": r["finestra_s"]}
    out["horitzons"] = horitzons(b["valors"], lam)
    out["limitacions"] = [
        "la lambda es va mesurar en separacions de ~6,6-13 graus fins a ~40 (sostre/3); aqui s'extrapola a 0,3-5 "
        "graus. Ho aguanta la validacio (nomes simulacio): el metode hi donava la lambda de Benettin (2,16 contra 2,13)",
        "la lambda es el ritme de la distancia en angles sqrt(dth1^2 + dth2^2); la X es de theta2",
        "l'horitzo d'un model perfecte amb nomes l'error de les condicions inicials, a 1,7-3,5 E_v2: NO es el de "
        "l'experiment 1 ni el del feix"]
    print(f"\nLa lambda: {lam:.3f} +- {sig:.3f} s^-1 ({b['remostrejos']} remostrejos; "
          f"{100 * out['horitzons']['fraccio_lambda_no_positiva']:.1f} % amb lambda <= 0)")
    print(f"{'D0':>6} {'D_tol':>6} {'central':>8} {'mediana':>8} {'p16-p84':>16} {'p2,5-p97,5':>16}")
    for k, h in out["horitzons"].items():
        if not k.startswith("d0_"):
            continue
        print(f"{h['D0_graus']:>6g} {h['D_tol_graus']:>6g} {h['central_s']:>7.2f}s {h['p50_s']:>7.2f}s "
              f"{h['p16_s']:>7.2f}-{h['p84_s']:<7.2f}s {h['p2.5_s']:>7.2f}-{h['p97.5_s']:<7.2f}s")
    return out


# =============================================================== 24/10: el repos final a (0, 0)

RESIDU_VERTICAL_GRAUS = 0.15      # la a, decidida per en Guillem el 23/10
TOL_TH2_REPOS_GRAUS = 0.05        # BR2


def treu_desplacament(th1, th2, L1, L2, c):
    """Els angles de debo si la marca seguida es M = J + c*n1, amb n1 la perpendicular a la BARRA 1.

    th1, L1: l'angle i la llargada de la marca (pivot -> M); th2, L2: els de M -> gomet. Exacte:
    |J| = sqrt(|M|^2 - c^2) i th1 = th_M - asin(c/|M|). Tot al conveni de fisica.py (y amunt).
    """
    th1 = np.asarray(th1, dtype=float)
    d1 = np.arcsin(c / np.asarray(L1, dtype=float))
    t1 = th1 - d1
    L1v = np.sqrt(np.asarray(L1, dtype=float) ** 2 - c * c)
    Gx = L1 * np.sin(th1) + L2 * np.sin(th2)
    Gy = -L1 * np.cos(th1) - L2 * np.cos(th2)
    a2 = np.arctan2(Gx - L1v * np.sin(t1), -(Gy + L1v * np.cos(t1)))
    return t1, th2 + dif_angular(a2, th2)


def corregeix_posicions(t_fot, J, G, inverteix, n_cua) -> dict:
    """La c que fa que la cua doni theta1 = 0 (modul 2 pi), i els angles corregits als fotogrames.

    J, G: les posicions de la conversio (relatives al pivot, y avall), com posicions_del_trk.
    """
    from scipy.optimize import brentq
    from pendol import tracker
    J = np.asarray(J, dtype=float)
    G = np.asarray(G, dtype=float)
    th1, th2 = tracker.a_angles(t_fot, J[:, 0], J[:, 1], G[:, 0], G[:, 1], (0.0, 0.0), True, inverteix)
    L1 = np.hypot(J[:, 0], J[:, 1])
    L2 = np.hypot(G[:, 0] - J[:, 0], G[:, 1] - J[:, 1])
    cua = slice(len(th1) - n_cua, None)

    def repos1(c):
        return float(np.mean(dif_angular(th1[cua] - np.arcsin(c / L1[cua]), 0.0)))
    lim = 0.5 * float(L1[cua].min())
    c = brentq(repos1, -lim, lim, xtol=1e-12, rtol=1e-15)
    t1, t2 = treu_desplacament(th1, th2, L1, L2, c)
    return {"c_px": float(c), "th1": th1, "th2": th2, "L1": L1, "L2": L2, "th1_c": t1, "th2_c": t2,
            "repos_abans_rad": [float(np.mean(dif_angular(x[cua], 0.0))) for x in (th1, th2)],
            "repos_despres_rad": [float(np.mean(dif_angular(x[cua], 0.0))) for x in (t1, t2)]}


def correccio_repos(pr, c_forcada=None) -> dict:
    """La presa amb el repos final a (0, 0), pel cami de la conversio. No toca ni el .trk ni el CSV.

    c_forcada = 0 es el BR1: el cami ha de tornar el CSV.
    """
    import soroll_en_moviment as sm
    meta = pr["meta"]
    t_fot, J, G, _ = sm.posicions_del_trk(meta)
    inv = bool(meta.get("signe_invertit", False))
    fps_fot = 1.0 / float(np.median(np.diff(t_fot)))
    n_cua = max(10, int(round(float(meta["geometria"]["cua_s"]) * fps_fot)))
    k = corregeix_posicions(t_fot, J, G, inv, n_cua)
    if c_forcada is not None:
        k["th1_c"], k["th2_c"] = treu_desplacament(k["th1"], k["th2"], k["L1"], k["L2"], float(c_forcada))
    t, fps, (a1, a2), _ = sm.a_la_graella(t_fot, k["th1_c"], k["th2_c"])
    fin, ordre = int(meta["savgol"]["finestra_mostres"]), int(meta["savgol"]["ordre"])
    X = np.vstack([savgol_filter(a1, fin, ordre), savgol_filter(a2, fin, ordre),
                   savgol_filter(a1, fin, ordre, deriv=1, delta=1.0 / fps),
                   savgol_filter(a2, fin, ordre, deriv=1, delta=1.0 / fps)])
    c = k["c_px"] if c_forcada is None else float(c_forcada)
    L1m, L2m = float(np.median(k["L1"])), float(np.median(k["L2"]))
    r1, r2 = treu_desplacament(pr["repos_rad"][0], pr["repos_rad"][1], L1m, L2m, c)
    ref = pr["estats"]
    dev = (float(np.max(np.abs(X - ref) / np.maximum(np.abs(ref), 1.0))) if X.shape == ref.shape else math.inf)
    return {"c_px": c, "estats": X, "repos_rad": np.array([float(r1), float(r2)]), "desviacio_del_csv": dev,
            "repos_final_abans_graus": [math.degrees(x) for x in k["repos_abans_rad"]],
            "repos_final_despres_graus": [math.degrees(x) for x in k["repos_despres_rad"]],
            "amollada_abans_graus": [math.degrees(x) for x in pr["repos_rad"]],
            "amollada_despres_graus": [math.degrees(float(r1)), math.degrees(float(r2))],
            "fotogrames_de_cua": n_cua}


def _r2_amb_gir(preses, pas02, rng, sigma_graus, sortejos):
    """El R2 dels 150 graus amb un gir o_i ~ N(0, sigma), el mateix a theta1 i theta2 de cada presa."""
    R = pas02["resultat2"]["150"]
    noms = R["preses"]
    N = len(noms)
    Dstar = R["D_estrella_rad"]
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    rep0 = [preses[n]["repos_rad"] for n in noms]
    d0 = np.array([float(dist_ang(rep0[a], rep0[b])) for a in range(N) for b in range(a + 1, N)])
    par = [(a, b) for a in range(N) for b in range(a + 1, N)]
    par = [par[k] for k in range(len(par)) if d0[k] <= Dstar / math.e]
    X0 = [a_la_graella(preses[n]["t"], preses[n]["estats"], 0.0, tc) for n in noms]
    v = []
    for _ in range(sortejos):
        o = np.radians(rng.normal(0.0, sigma_graus, N))
        X = [x.copy() for x in X0]
        rp = [r.copy() for r in rep0]
        for i in range(N):
            X[i][:2] += o[i]
            rp[i] = rp[i] + o[i]
        d0p = np.array([float(dist_ang(rp[a], rp[b])) for a, b in par])
        d = np.array([dist_ang(X[a], X[b]) for a, b in par])
        v.append(avalua_llindars(tc, d, np.maximum(d0p, 1e-6), par, N, [Dstar], n_boot=0)["lambda_D"][0])
    return v


def corregit(pas0, pas02, reals_json) -> dict:
    """El R2 i el R1 amb el repos final a (0, 0) (pla, 24/10). Real contra real, darrere del pany."""
    if not APROVAT.exists():
        raise SystemExit("Pany tancat: no hi ha l'aprovacio de les preses reals.")
    criteris = {}

    # BR0: sense correccio, el que es va publicar el 17/10.
    print("BR0. El pas 3 sense correccio, contra el publicat el 17/10", flush=True)
    r0 = reals2(pas0, pas02)
    pub = reals_json
    e = {"R1_lambda": abs(r0["resultat1"]["1,7-3,5"]["lambda"] / pub["resultat1"]["1,7-3,5"]["lambda"] - 1),
         "R1_sigma": abs(r0["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"]
                         / pub["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"] - 1),
         "R2_lambda": abs(r0["resultat2"]["150"]["lambda"] / pub["resultat2"]["150"]["lambda"] - 1),
         "R2_sigma": abs(r0["resultat2"]["150"]["bootstrap"]["sigma_D"][0]
                         / pub["resultat2"]["150"]["bootstrap"]["sigma_D"][0] - 1)}
    # Reformulada el 24/10 DESPRES de fallar (la 90_10 es va tornar a convertir el 23/10, i entra al R1
    # pero no al R2), amb la decisio d'en Guillem: el R2 a 1e-12 i el R1 a 1e-3. Vegeu el pla.
    passa = max(e["R2_lambda"], e["R2_sigma"]) <= 1e-12 and max(e["R1_lambda"], e["R1_sigma"]) <= 1e-3
    criteris["BR0"] = {"passa": bool(passa), "errors": e, "cotes": {"R2": 1e-12, "R1": 1e-3}}
    print(f"   {'ok ' if passa else 'NO '} R2 {max(e['R2_lambda'], e['R2_sigma']):.1e} (cal 1e-12); "
          f"R1 {max(e['R1_lambda'], e['R1_sigma']):.1e} (cal 1e-3)", flush=True)

    # BR1 i BR2, presa a presa.
    preses = {pr["nom"]: pr for pr in preses_reals()}
    per_presa, preses_c = {}, {}
    for nom, pr in preses.items():
        zero = correccio_repos(pr, c_forcada=0.0)
        k = correccio_repos(pr)
        th1d, th2d = k["repos_final_despres_graus"]
        per_presa[nom] = {kk: v for kk, v in k.items() if kk not in ("estats", "repos_rad")}
        per_presa[nom]["desviacio_del_csv_amb_c_zero"] = zero["desviacio_del_csv"]
        per_presa[nom]["signe_invertit"] = bool(pr["meta"].get("signe_invertit", False))
        preses_c[nom] = {**pr, "estats": k["estats"], "repos_rad": k["repos_rad"]}
        print(f"   {nom:7} c {k['c_px']:+7.2f} px; repos final abans ({k['repos_final_abans_graus'][0]:+6.2f}, "
              f"{k['repos_final_abans_graus'][1]:+6.2f}) -> despres ({th1d:+.1e}, {th2d:+6.3f}) graus; "
              f"amb c = 0, del CSV {zero['desviacio_del_csv']:.1e}", flush=True)
    br1 = max(v["desviacio_del_csv_amb_c_zero"] for v in per_presa.values())
    criteris["BR1"] = {"passa": bool(br1 <= 1e-9), "desviacio_maxima": br1}
    th1m = max(abs(v["repos_final_despres_graus"][0]) for v in per_presa.values())
    th2m = max(abs(v["repos_final_despres_graus"][1]) for v in per_presa.values())
    criteris["BR2"] = {"passa": bool(math.radians(th1m) <= 1e-9 and th2m <= TOL_TH2_REPOS_GRAUS),
                       "theta1_maxim_graus": th1m, "theta2_maxim_graus": th2m}
    print(f"BR1. Amb c = 0, el cami torna el CSV: desviacio maxima {br1:.1e} (cal <= 1e-9) -> "
          f"{'ok' if criteris['BR1']['passa'] else 'FALLA'}")
    print(f"BR2. Repos final corregit: |theta1| maxim {th1m:.1e} graus, |theta2| maxim {th2m:.3f} graus "
          f"(cal <= {TOL_TH2_REPOS_GRAUS}) -> {'ok' if criteris['BR2']['passa'] else 'FALLA'}", flush=True)
    out = {"eina": "eines/lyapunov_real.py corregit", "pla": "documentacio/pla-analisi-caos.md (24/10)",
           "que_es": "el R2 i el R1 amb el repos final de cada presa a (0, 0): REAL CONTRA REAL, cap simulacio. "
                     "Els numeros publicats no canvien",
           "criteris": criteris, "per_presa": per_presa}
    if not all(v["passa"] for v in criteris.values()):
        print("\nUn criteri FALLA: no es calcula cap lambda corregida. Un criteri que falla NO es relaxa.")
        return out

    print("\nEl pas 3 amb la correccio", flush=True)
    rc = reals2(pas0, pas02, preses=preses_c, guarda_valors=True)
    valors = rc["resultat1"]["1,7-3,5"]["bootstrap"].pop("valors")
    out["resultat2"], out["resultat1"] = rc["resultat2"], rc["resultat1"]
    out["sense_correccio"] = {"R2_150_lambda": r0["resultat2"]["150"]["lambda"],
                              "R2_150_sigma": r0["resultat2"]["150"]["bootstrap"]["sigma_D"][0],
                              "R1_lambda": r0["resultat1"]["1,7-3,5"]["lambda"],
                              "R1_sigma": r0["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"],
                              "que_es": "el pas 3 sense corregir amb les preses d'ara: la base nova del 24/10"}
    rng = np.random.default_rng(LLAVOR + 24)
    v = _r2_amb_gir(preses_c, pas02, rng, RESIDU_VERTICAL_GRAUS, 200)
    out["resultat2"]["150"]["mc_residu_vertical"] = {
        "sigma_graus": RESIDU_VERTICAL_GRAUS, "sortejos": 200, "mitjana": float(np.mean(v)),
        "desviacio": float(np.std(v, ddof=1))}
    out["horitzons_amb_R1_corregit"] = horitzons(valors, rc["resultat1"]["1,7-3,5"]["lambda"])
    a, b = r0["resultat2"]["150"], rc["resultat2"]["150"]
    print(f"   R2 150: lambda* {a['lambda']:.3f} -> {b['lambda']:.3f} +- {b['bootstrap']['sigma_D'][0]:.3f} "
          f"(alineacio +- {b['mc_alineacio']['desviacio']:.3f}; residu de la vertical +- "
          f"{out['resultat2']['150']['mc_residu_vertical']['desviacio']:.3f}); {b['parelles']} parelles")
    a, b = r0["resultat1"]["1,7-3,5"], rc["resultat1"]["1,7-3,5"]
    print(f"   R1 veins: lambda {a['lambda']:.3f} -> {b['lambda']:.3f} +- {b['bootstrap']['sigma']:.3f} "
          f"({b['parelles']} parelles de {b['tirs']} tirs, creix {b['creixement_ln']:.2f})")
    h = out["horitzons_amb_R1_corregit"]["d0_0.3_tol_3"]
    print(f"   horitzo (0,3 -> 3 graus) amb el R1 corregit: {h['central_s']:.2f} s (p16-p84 {h['p16_s']:.2f}-"
          f"{h['p84_s']:.2f} s)")
    return out


def preses_corregides():
    """(preses_c, per_presa, rep0): les preses de preses_reals() amb la correccio que es firma,
    comparar.corregeix_presa. Era el cos de dues_parts; surt a part el 28/10 perque l'horitzo i les figures corregides
    en facin servir la mateixa (la DQ5 comprova que dues_parts dona el mateix bit a bit).

    Com esborranys/_contra_corregeix_presa.py: els angles, mes el que mou la correccio; les omega, mes el seu
    gradient (la derivada de l'angle corregit, filtrat una sola vegada); l'amollada, mes la mitjana del que mou
    la correccio al repos d'abans d'amollar."""
    from pendol import comparar
    preses_c, per_presa, rep0 = {}, {}, {}
    for pr in preses_reals():
        rep0[pr["nom"]] = pr["repos_rad"]
        taula, meta = pr["taula"], pr["meta"]
        tc = taula["t"].to_numpy()
        th1, th2 = taula["theta1"].to_numpy(), taula["theta2"].to_numpy()
        a1, a2, info = comparar.corregeix_presa(taula, meta, pr["nom"])
        mou = np.vstack([a1 - th1, a2 - th2])
        X = pr["estats"].copy()
        X[:2] = X[:2] + mou
        X[2:] = X[2:] + np.gradient(mou, tc, axis=1)
        ta = meta["amollada"]["amollada_s"]
        ini = (tc >= max(0.0, ta - 0.4)) & (tc <= ta - 0.02)
        rep = pr["repos_rad"] + mou[:, ini].mean(axis=1)
        preses_c[pr["nom"]] = {**pr, "estats": X, "repos_rad": rep}
        per_presa[pr["nom"]] = {"amollada_abans_graus": np.degrees(pr["repos_rad"]).tolist(),
                                "amollada_despres_graus": np.degrees(rep).tolist(), "rho": info["rho"]}
    return preses_c, per_presa, rep0


def dues_parts(pas0, pas02, base) -> dict:
    """El R2 i el R1 amb la correccio que es firmara, comparar.corregeix_presa (pla, 25/10, «Tres feines»), PER
    INFORMACIO: cap numero adoptat, i als 150 graus la correccio no esta validada. Real contra real, darrere del pany.
    Puja d'esborranys/_contra_corregeix_presa.py --regle (DQ1, comprovada fora). `base` es el JSON del 24/10 (DQ2)."""
    if not APROVAT.exists():
        raise SystemExit("Pany tancat: no hi ha l'aprovacio de les preses reals.")

    print("DQ2. Sense correccio, la base del 24/10", flush=True)
    r0 = reals2(pas0, pas02)
    e = max(abs(r0["resultat2"]["150"]["lambda"] / base["resultat2"]["150"]["lambda"] - 1),
            abs(r0["resultat2"]["150"]["bootstrap"]["sigma_D"][0] / base["resultat2"]["150"]["bootstrap"]["sigma_D"][0] - 1),
            abs(r0["resultat1"]["1,7-3,5"]["lambda"] / base["resultat1"]["1,7-3,5"]["lambda"] - 1),
            abs(r0["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"] / base["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"] - 1))
    criteris = {"DQ2": {"passa": bool(e <= 1e-12), "error": e}}
    print(f"   {'ok ' if e <= 1e-12 else 'NO '} error relatiu maxim {e:.1e} (cal <= 1e-12)", flush=True)
    out = {"eina": "eines/lyapunov_real.py dues_parts", "pla": "documentacio/pla-analisi-caos.md (25/10, «Tres feines»)",
           "que_es": "el R2 i el R1 amb comparar.corregeix_presa (la correccio en dues parts, rho del regle), PER "
                     "INFORMACIO: REAL CONTRA REAL, cap simulacio, cap numero adoptat; als 150 graus la correccio no "
                     "esta validada",
           "criteris": criteris}
    if not criteris["DQ2"]["passa"]:
        print("\nLa DQ2 FALLA: no es calcula res amb la correccio. Un criteri que falla NO es relaxa.")
        return out

    preses_c, per_presa, rep0 = preses_corregides()
    rc = reals2(pas0, pas02, preses=preses_c)
    noms = pas02["resultat2"]["150"]["preses"]
    d0 = {}
    for clau, pp in (("abans", rep0), ("despres", {n: preses_c[n]["repos_rad"] for n in noms})):
        d0[clau] = math.degrees(float(np.median([dist_ang(pp[noms[a]], pp[noms[b]])
                                                 for a in range(len(noms)) for b in range(a + 1, len(noms))])))
    out.update({"resultat2": rc["resultat2"], "resultat1": rc["resultat1"], "per_presa": per_presa,
                "d0_mediana_150_graus": d0,
                "sense_correccio": {"R2_150_lambda": r0["resultat2"]["150"]["lambda"],
                                    "R2_150_sigma": r0["resultat2"]["150"]["bootstrap"]["sigma_D"][0],
                                    "R1_lambda": r0["resultat1"]["1,7-3,5"]["lambda"],
                                    "R1_sigma": r0["resultat1"]["1,7-3,5"]["bootstrap"]["sigma"],
                                    "que_es": "el pas 3 sense corregir: la base del 24/10"}})
    a, b = r0["resultat2"]["150"], rc["resultat2"]["150"]
    print(f"   R2 150: lambda* {a['lambda']:.4f} -> {b['lambda']:.4f} +- {b['bootstrap']['sigma_D'][0]:.4f} "
          f"(alineacio +- {b['mc_alineacio']['desviacio']:.3f}); {b['parelles']} parelles; d0 mediana "
          f"{d0['abans']:.2f} -> {d0['despres']:.2f} graus")
    a, b = r0["resultat1"]["1,7-3,5"], rc["resultat1"]["1,7-3,5"]
    print(f"   R1 veins: lambda {a['lambda']:.4f} -> {b['lambda']:.4f} +- {b['bootstrap']['sigma']:.4f} "
          f"({b['parelles']} parelles de {b['tirs']} tirs)")
    return out


# =============================================================== 26/10: la Z de la lambda* (z_r2)
# Pla: «26/10 · La Z de la λ* dels 150°», escrit abans d'aquest codi. Real contra simulat amb el MATEIX metode (el R2),
# des de les mateixes condicions inicials mesurades i amb la mateixa cadena de mesura. La regla, la de criteri_z.

Z_GRUP = "150"
Z_REALITZACIONS = 20
Z_BESSONS = 20
Z_BESSONS_K = 10
Z_K = (0.7, 0.8, 0.9, 1.1, 1.2, 1.3)
Z_S_CANDIDATA = (1.45, 1.53)      # l'eix (1,01 / 1,03) i la creu dels 90 (1,045 / 1,127), en quadratura: vegeu el pla


def condicions_mesurades(pas0, pas02, reals_json, g=Z_GRUP):
    """(noms, x_mes, d'on): les condicions inicials mesurades de les preses del R2, les del resultat real triat.

    Amb el JSON de dues_parts, les corregides (per_presa.amollada_despres_graus); si no, les del pas 0 (sense corregir).
    """
    noms = pas02["resultat2"][g]["preses"]
    pp = reals_json.get("per_presa") or {}
    if noms and all("amollada_despres_graus" in (pp.get(n) or {}) for n in noms):
        ang = {n: pp[n]["amollada_despres_graus"] for n in noms}
        de = "per_presa.amollada_despres_graus del JSON real (corregides)"
    else:
        ang = {n: pas0["per_presa"][n]["angles_repos_graus"] for n in noms}
        de = "pas0.angles_repos_graus (sense corregir)"
    return noms, [np.array([*np.radians(ang[n]), 0.0, 0.0]) for n in noms], de


def _veritats(x_mes, s_graus, llavor):
    """Les condicions de debo d'una realitzacio: les mesurades, mes N(0, s) a cada angle (l'opcio b). Amb s = 0, les
    mesurades tal qual (l'opcio a)."""
    if not any(s_graus):
        return [x.copy() for x in x_mes]
    e = np.random.default_rng(llavor).normal(0.0, 1.0, (len(x_mes), 2)) * np.radians(np.asarray(s_graus, dtype=float))
    return [np.r_[x[:2] + e[i], x[2:]] for i, x in enumerate(x_mes)]


def _sim_doble_k(args):
    """El doble k vegades mes de pressa: y(k t) a la graella de 240 fps, o sigui la lambda per k. NOMES per a la ZR4."""
    from pendol.simulador import simula
    x0, temps, k = args
    t = np.arange(0.0, temps + 0.5 * DT, DT)
    return simula(_p(), np.asarray(x0, dtype=float), t_eval=k * t)[1]


def _trajectories(pool, conjunts, escala=1.0, amb_friccio=False):
    """Una llista de conjunts de condicions inicials -> les trajectories de cada conjunt, en una sola tanda del pool."""
    pla = [x for c in conjunts for x in c]
    if escala == 1.0:
        Y = pool.map(_sim_doble, [(x, DURADA_SIM_S, amb_friccio) for x in pla])
    else:
        Y = pool.map(_sim_doble_k, [(x, DURADA_SIM_S, escala) for x in pla])
    sortida, k = [], 0
    for c in conjunts:
        sortida.append(Y[k:k + len(c)])
        k += len(c)
    return sortida


def _realitzacio(pool, x_mes, s_graus, g, rs, que, escala=1.0, amb_friccio=False, veritats=None):
    """(condicions de debo, trajectories) de cada realitzacio. Amb s = 0 totes surten de les mesurades: es simula un cop.
    veritats (27/10, nomes per als diagnostics): una altra manera de sortejar les de debo, amb les mateixes llavors."""
    xvs = [(veritats or _veritats)(x_mes, s_graus, rg.llavor_de(float(g), r, que, 23)) for r in rs]
    if any(s_graus):
        return xvs, _trajectories(pool, xvs, escala, amb_friccio)
    return xvs, _trajectories(pool, [xvs[0]], escala, amb_friccio) * len(rs)


def _observa_tirs(Y, xv, sd, rho, sig, g, r, que):
    """Els tirs com els dona la conversio (el soroll, filtrat) i l'error d'amollada de cada presa. Amb que = 1, les
    llavors de valida2 (la ZR2 ho comprova bit a bit)."""
    obs = [observa(y, x, sd, rho, np.random.default_rng(rg.llavor_de(float(g), r, i, que, 21)))
           for i, (y, x) in enumerate(zip(Y, xv))]
    alin = rg.llavor_de(float(g), r, 22) if que == 1 else rg.llavor_de(float(g), r, que, 22)
    return obs, np.random.default_rng(alin).normal(0.0, sig)


def r2_simulat(pool, x_mes, sig, R, *, g=Z_GRUP, s_graus=(0.0, 0.0), rs=range(Z_REALITZACIONS), que=1,
               amb_friccio=False, Ds_diagnostic=None, veritats=None) -> list:
    """El costat simulat de la Z: el R2 d'amb_soroll de valida2 a cada realitzacio. NOMES SIMULACIO.

    La d0 i les parelles son les de les condicions MESURADES, com a la presa real; els tirs surten de les de debo
    (_veritats). Sense friccio, o DE PROVA. Per als diagnostics (27/10): Ds_diagnostic afegeix la lambda(D) a part
    (sense bootstrap, i sense tocar la resta), i veritats canvia com se sortegen les de debo.
    """
    N = len(x_mes)
    Dstar = R["D_estrella_rad"]
    sd, rho = np.array(R["soroll_validacio_rad"]), R["rho"]
    tc = np.arange(0.0, T_MAX_S + 0.5 * DT, DT)
    par_all = [(a, b) for a in range(N) for b in range(a + 1, N)]
    d0 = np.array([float(dist_ang(x_mes[a], x_mes[b])) for a, b in par_all])
    sel = [k for k in range(len(par_all)) if d0[k] <= Dstar / math.e]
    par = [par_all[k] for k in sel]
    rs = list(rs)
    xvs, Ys = _realitzacio(pool, x_mes, s_graus, g, rs, que, amb_friccio=amb_friccio, veritats=veritats)
    out = []
    for r, xv, Y in zip(rs, xvs, Ys):
        obs, dts = _observa_tirs(Y, xv, sd, rho, sig, g, r, que)
        X = [a_la_graella(t, Xo, dts[i], tc) for i, (t, Xo) in enumerate(obs)]
        d = np.array([dist_ang(X[a], X[b]) for a, b in par])
        out.append(avalua_llindars(tc, d, d0[sel], par, N, [Dstar], llavor=rg.llavor_de(float(g), r, que)))
        if Ds_diagnostic:
            out[-1]["diagnostic_lambda_D"] = avalua_llindars(tc, d, d0[sel], par, N, list(Ds_diagnostic), n_boot=0)["lambda_D"]
    return out


def resum_simulat(res) -> dict:
    """La lambda_sim (mediana de les realitzacions) i sigma_sim = sqrt(mediana dels sigma_boot^2 + dispersio^2)."""
    lam = np.array([x["lambda_D"][0] for x in res])
    sb2 = np.array([x["bootstrap"]["sigma_D"][0] for x in res]) ** 2
    disp = float(np.std(lam, ddof=1)) if len(lam) > 1 else 0.0
    return {"lambda": float(np.median(lam)), "sigma_boot": math.sqrt(float(np.median(sb2))), "dispersio": disp,
            "sigma": math.sqrt(float(np.median(sb2)) + disp ** 2), "realitzacions": len(res),
            "lambda_realitzacions": lam.tolist(), "fraccio_creuen": [x["fraccio_creuen"][0] for x in res],
            "parelles": res[0]["parelles"]}


def bessons(pool, x_mes, sig, R, noms, *, g=Z_GRUP, s_graus=(0.0, 0.0), rs=range(Z_BESSONS), que=4,
            escala=1.0) -> list:
    """Bessons de la presa real: el mateix model (o k vegades mes de pressa), mesurats com la presa i passats per
    r2_de_preses, l'estimador de les preses reals. La lambda i la sigma_real de cada un. NOMES SIMULACIO."""
    sd, rho = np.array(R["soroll_validacio_rad"]), R["rho"]
    rs = list(rs)
    xvs, Ys = _realitzacio(pool, x_mes, s_graus, g, rs, que, escala)
    out = []
    for r, xv, Y in zip(rs, xvs, Ys):
        obs, dts = _observa_tirs(Y, xv, sd, rho, sig, g, r, que)
        # El zero de cada besso es la seva amollada «mesurada»: la de debo mes l'error.
        preses = {n: {"t": t - dts[i], "estats": Xo, "repos_rad": x_mes[i][:2]}
                  for i, (n, (t, Xo)) in enumerate(zip(noms, obs))}
        b = r2_de_preses(preses, noms, sig, R["D_estrella_rad"], complet=False)
        out.append({"lambda": b["lambda"], "sigma": math.hypot(b["bootstrap"]["sigma_D"][0], b["mc_alineacio"]["desviacio"])})
    return out


SIGMES_SIM = ("regla_04_10", "dispersio")   # 27/10: la dispersio sola es l'opcio 2 (pla, «L'opcio 2 a z_r2»)


def _firma_z(nom_reals=None, s_graus=None, realitzacions=None, sigma_sim=None):
    """EL PANY de la Z. Sense firma.json, o amb la X, la Y o la Z buides, s'atura ABANS D'OBRIR RES. Si la Z firmada
    porta parametres (sigmes, error_maxim, reals, sigma_ci_graus, realitzacions, sigma_sim), els d'ara hi han de
    coincidir."""
    if not FIRMA.exists():
        raise SystemExit(f"Pany tancat: no hi ha {FIRMA.relative_to(ARREL)}. La Z compara simulacio i "
                         "realitat, i aixo no es fa abans de firmar X, Y i Z.")
    f = json.loads(FIRMA.read_text(encoding="utf-8"))
    buits = [k for k in ("X", "Y", "Z") if not f.get(k)]
    if buits:
        raise SystemExit(f"Pany tancat: a {FIRMA.relative_to(ARREL)} hi ha buit: {', '.join(buits)}.")
    Z = f["Z"] if isinstance(f["Z"], dict) else {}
    import criteri_z
    cami = lambda c: (Path(c) if Path(c).is_absolute() else ARREL / c).resolve()  # noqa: E731
    xoca = [k for k, v in (("sigmes", criteri_z.SIGMES), ("error_maxim", criteri_z.ERROR_MAXIM)) if k in Z and float(Z[k]) != v]
    if nom_reals is not None and "reals" in Z and cami(Z["reals"]) != cami(nom_reals):
        xoca.append("reals")
    if s_graus is not None and "sigma_ci_graus" in Z and [float(x) for x in Z["sigma_ci_graus"]] != [float(x) for x in s_graus]:
        xoca.append("sigma_ci_graus")
    if realitzacions is not None and "realitzacions" in Z and int(Z["realitzacions"]) != int(realitzacions):
        xoca.append("realitzacions")
    if sigma_sim is not None and "sigma_sim" in Z and Z["sigma_sim"] != sigma_sim:
        xoca.append("sigma_sim")
    if xoca:
        raise SystemExit(f"La Z firmada no diu el mateix que aquesta crida: {', '.join(xoca)}. No es corre.")
    return f["Z"]


def z_r2(pas0, pas02, reals_json, *, sigma_ci_graus, nom_reals=None, realitzacions=Z_REALITZACIONS,
         amb_friccio_de_prova=False, sigma_sim="regla_04_10", processos=PROCESSOS) -> dict:
    """La Z amb la lambda* dels 150 graus: la del JSON real contra la del R2 simulat des de les mateixes condicions
    mesurades, amb el mateix soroll, la mateixa alineacio i el mateix metode.

    PANY: es nega a funcionar, ABANS D'OBRIR RES, sense la firma de X, Y i Z. La presa real no s'obre: la lambda, el
    sigma del bootstrap i el de l'alineacio surten del JSON. sigma_ci_graus = (0, 0) es l'opcio (a); > 0, la (b).
    sigma_sim (27/10): «regla_04_10» = sqrt(mediana sigma_boot^2 + dispersio^2); «dispersio» = l'opcio 2, la dispersio
    entre realitzacions sola (els dos costats comparteixen les condicions inicials).
    """
    if sigma_sim not in SIGMES_SIM:
        raise ValueError(f"sigma_sim ha de ser {' o '.join(SIGMES_SIM)}")
    firma_z = _firma_z(nom_reals, sigma_ci_graus, realitzacions, sigma_sim)
    import criteri_z
    g = Z_GRUP
    R = pas02["resultat2"][g]
    noms, x_mes, de = condicions_mesurades(pas0, pas02, reals_json, g)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    rr = reals_json["resultat2"][g]
    N = len(noms)
    d0 = [float(dist_ang(x_mes[a], x_mes[b])) for a in range(N) for b in range(a + 1, N)]
    par_sim = sum(x <= R["D_estrella_rad"] / math.e for x in d0)
    if par_sim != rr["parelles"]:
        raise SystemExit(f"Les parelles no coincideixen ({par_sim} simulades, {rr['parelles']} reals): les condicions "
                         "inicials no son les del resultat real. No es corre.")
    real = {"lambda": rr["lambda"], "sigma_boot": rr["bootstrap"]["sigma_D"][0],
            "sigma_alineacio": rr["mc_alineacio"]["desviacio"]}
    real["sigma"] = math.hypot(real["sigma_boot"], real["sigma_alineacio"])
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        sim = resum_simulat(r2_simulat(pool, x_mes, sig, R, s_graus=sigma_ci_graus, rs=range(realitzacions)))
        fr = (resum_simulat(r2_simulat(pool, x_mes, sig, R, s_graus=sigma_ci_graus, rs=range(realitzacions),
                                       amb_friccio=True)) if amb_friccio_de_prova else None)
    tria = (lambda x: x["sigma"]) if sigma_sim == "regla_04_10" else (lambda x: x["dispersio"])  # noqa: E731
    sim["sigma_z"] = tria(sim)
    v = criteri_z.veredicte(real["lambda"], real["sigma"], sim["lambda"], sim["sigma_z"])
    out = {"eina": "eines/lyapunov_real.py z_r2", "pla": "documentacio/pla-analisi-caos.md (26/10, la Z de la lambda*)",
           "que_es": "LA Z amb la lambda* dels 150 graus: real (del JSON) contra el R2 simulat des de les mateixes "
                     "condicions mesurades, amb el mateix soroll, alineacio i metode. Sense friccio",
           "firma_Z": firma_z, "reals": nom_reals, "condicions_inicials_de": de, "preses": noms,
           "sigma_ci_graus": [float(x) for x in sigma_ci_graus],
           "opcio": "a (les mesurades)" if not any(sigma_ci_graus) else "b (les mesurades mes N(0, s))",
           "real": real, "simulat": sim, **v, "friccio": "cap", "sigma_sim_definicio": sigma_sim,
           "regla": {"sigmes": criteri_z.SIGMES, "error_maxim": criteri_z.ERROR_MAXIM,
                     "sigma_real": "sqrt(sigma_boot^2 + sigma_alineacio^2) del JSON real",
                     "sigma_sim": ("sqrt(mediana dels sigma_boot^2 + dispersio entre realitzacions^2)"
                                   if sigma_sim == "regla_04_10" else
                                   "la dispersio entre realitzacions sola (opcio 2, 27/10)")}}
    if fr is not None:
        fr["sigma_z"] = tria(fr)
        out["friccio_de_prova_info"] = {"simulat": fr, **criteri_z.veredicte(real["lambda"], real["sigma"], fr["lambda"],
                                                                             fr["sigma_z"])}
    print(f"La Z (lambda* dels 150 graus), opcio {out['opcio']}, s = {out['sigma_ci_graus']} graus, sigma_sim = {sigma_sim}")
    print(f"   real:     {real['lambda']:.3f} +- {real['sigma']:.3f} ({100 * v['error_relatiu_real']:.0f} %)")
    print(f"   simulat:  {sim['lambda']:.3f} +- {sim['sigma_z']:.3f} ({100 * v['error_relatiu_sim']:.0f} %; "
          f"{sim['realitzacions']} realitzacions, dispersio {sim['dispersio']:.3f})")
    print(f"   diferencia {v['diferencia']:+.3f}, 2 sigma = {2 * v['sigma_combinada']:.3f} -> {v['veredicte'].upper()}")
    if fr is not None:
        print(f"   (info) amb la friccio DE PROVA: {fr['lambda']:.3f} +- {fr['sigma_z']:.3f} -> "
              f"{out['friccio_de_prova_info']['veredicte']}")
    return out


def z_costat_simulat(pas0, pas02, reals_json, *, sigma_ci_graus, realitzacions=Z_REALITZACIONS,
                     processos=PROCESSOS) -> dict:
    """NOMES SIMULACIO (27/10): l'error relatiu del costat simulat de la Z, amb la sigma_sim del 04/10 i amb la de
    l'opcio 2 (nomes la dispersio entre realitzacions), per decidir la regla abans de la firma.

    Del JSON real nomes en llegeix les condicions inicials (condicions_mesurades), i no escriu cap lambda absoluta: la
    simulada no es posa al costat de la real fins que hi hagi firma.json. Per aixo no demana el pany.
    """
    g = Z_GRUP
    R = pas02["resultat2"][g]
    noms, x_mes, de = condicions_mesurades(pas0, pas02, reals_json, g)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        s = resum_simulat(r2_simulat(pool, x_mes, sig, R, s_graus=sigma_ci_graus, rs=range(realitzacions)))
    rel = {"bootstrap": s["sigma_boot"] / s["lambda"], "dispersio": s["dispersio"] / s["lambda"],
           "regla_04_10": s["sigma"] / s["lambda"], "opcio_2": s["dispersio"] / s["lambda"]}
    print(f"   s = {list(sigma_ci_graus)} graus, condicions de {de}: error relatiu amb la regla del 04/10 "
          f"{100 * rel['regla_04_10']:.1f} % (bootstrap {100 * rel['bootstrap']:.1f} %, dispersio {100 * rel['dispersio']:.1f} %); "
          f"amb l'opcio 2, {100 * rel['opcio_2']:.1f} %", flush=True)
    return {"condicions_inicials_de": de, "sigma_ci_graus": [float(x) for x in sigma_ci_graus],
            "realitzacions": s["realitzacions"], "parelles": s["parelles"], "error_relatiu": rel}


# Els tres casos del pla (27/10, «La Z amb la regla canviada»), i el que va imprimir l'autoprova del 26/10 (ZS1).
Z_CASOS = (("C0a", "lyapunov_real_reals_2026-10-24.json", (0.0, 0.0)),
           ("C0b", "lyapunov_real_reals_2026-10-24.json", Z_S_CANDIDATA),
           ("C3", "lyapunov_real_dues_parts_2026-10-27.json", (1.045, 1.127)))
ZS1_AUTOPROVA = {"C0a": (28.3, 27.2, 7.8), "C0b": (33.4, 23.9, 23.3)}


def z_casos(processos=PROCESSOS) -> dict:
    """NOMES SIMULACIO: els tres casos de z_costat_simulat, amb ZS1-ZS3 (pla, 27/10)."""
    global _OBERTS
    sys.addaudithook(_registra)
    _OBERTS = []
    carrega = lambda c: json.loads(Path(c).read_text(encoding="utf-8"))  # noqa: E731
    pas0, pas02 = carrega(PAS0_JSON), carrega(PAS0B_JSON)
    casos = {}
    for nom, fitxer, s in Z_CASOS:
        print(f"{nom}:", flush=True)
        casos[nom] = {"reals": f"dades/divergencia/{fitxer}",
                      **z_costat_simulat(pas0, pas02, carrega(DIVERGENCIA / fitxer), sigma_ci_graus=s, processos=processos)}
    oberts, _OBERTS = _OBERTS, None
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    permesos = {"divergencia/" + f for f in {PAS0_JSON.name, PAS0B_JSON.name, *(c[1] for c in Z_CASOS)}}
    zs1 = {k: [round(100 * casos[k]["error_relatiu"][q], 1) for q in ("regla_04_10", "bootstrap", "dispersio")]
           for k in ZS1_AUTOPROVA}
    crit = {"ZS1": {"passa": all(tuple(zs1[k]) == ZS1_AUTOPROVA[k] for k in zs1), "ara": zs1,
                    "autoprova_26_10": {k: list(v) for k, v in ZS1_AUTOPROVA.items()}},
            "ZS2": {"passa": set(dins) <= permesos, "fitxers_oberts_de_dades": dins},
            "ZS3": {"passa": '"lambda' not in json.dumps(_json_net(casos))}}
    for k, c in crit.items():
        print(f"   {'ok ' if c['passa'] else 'NO '} {k}: " + json.dumps({x: y for x, y in c.items() if x != 'passa'},
                                                                        ensure_ascii=False))
    c3 = casos["C3"]["error_relatiu"]["opcio_2"]
    resultat = ("avaluable: abans de fer-la servir, una ZR3' i una ZR4' amb criteris escrits abans" if c3 <= 0.20
                else "no avaluable")
    print(f"\nC3 amb l'opcio 2: {100 * c3:.1f} % -> {resultat}")
    return {"eina": "eines/lyapunov_real.py z_costat_simulat",
            "pla": "documentacio/pla-analisi-caos.md (27/10, la Z amb la regla canviada)",
            "que_es": "NOMES SIMULACIO: l'error relatiu del costat simulat de la Z (cap lambda absoluta), amb la sigma_sim "
                      "del 04/10 i amb la de l'opcio 2 (nomes la dispersio entre realitzacions)",
            "casos": casos, "criteris": crit, "C3_opcio_2": {"error_relatiu": c3, "resultat": resultat}}


# =============================================================== pas 3: el pany

# 27/10 al vespre: els diagnostics de la Z firmada, SENSE FRICCIO i SENSE CAP VEREDICTE (pla, «Els diagnostics de la Z»).
Z_R2_JSON = DIVERGENCIA / "lyapunov_real_z_r2_2026-10-27.json"


def _posa_p(q):
    """Inicialitzador del pool del D3: els processos simulen amb la variant del muntatge q."""
    global _P
    _P = q


def _contraccio(x_mes, s_graus):
    """D4b: (mu, v, tau^2, c) de cada angle, en radiants, de les condicions mesurades (ddof = 1)."""
    s = np.radians(np.asarray(s_graus, dtype=float))
    M = np.array([x[:2] for x in x_mes])
    mu, v = M.mean(axis=0), M.var(axis=0, ddof=1)
    tau2 = np.maximum(0.0, v - s ** 2)
    return mu, v, tau2, tau2 / (tau2 + s ** 2)


def _veritats_contretes(x_mes, s_graus, llavor):
    """D4b: les de debo amb la contraccio d'empirical Bayes, mu + c (mesurada - mu) + sqrt(c) s z, amb les mateixes z
    que _veritats. Aixi les distancies de sortida son, de mitjana, les de debo (no les mesurades mes l'error)."""
    s = np.radians(np.asarray(s_graus, dtype=float))
    mu, _, _, c = _contraccio(x_mes, s_graus)
    z = np.random.default_rng(llavor).normal(0.0, 1.0, (len(x_mes), 2))
    return [np.r_[mu + c * (x[:2] - mu) + np.sqrt(c) * s * z[i], x[2:]] for i, x in enumerate(x_mes)]


def z_diagnostic(processos=PROCESSOS) -> dict:
    """Els diagnostics de la Z firmada: D4 (la s), D4b (la contraccio), D2 (sense la correccio), D3 (el muntatge) i D1
    (la lambda(D)). SENSE FRICCIO i SENSE CAP VEREDICTE: el «no passa» firmat no es mou. Darrere del pany de firma.json;
    les lambda reals surten dels JSON, i no s'obre cap presa (DG1). La referencia es el costat simulat de la Z (DG0)."""
    global _OBERTS
    firma_z = _firma_z()
    import incerteses_muntatge as im
    sys.addaudithook(_registra)
    _OBERTS = []
    carrega = lambda c: json.loads(Path(c).read_text(encoding="utf-8"))  # noqa: E731
    pas0, pas02 = carrega(PAS0_JSON), carrega(PAS0B_JSON)
    dp, base, zr = carrega(DUES_PARTS_2710_JSON), carrega(REALS_BASE_JSON), carrega(Z_R2_JSON)
    g = Z_GRUP
    R = pas02["resultat2"][g]
    s = tuple(float(x) for x in firma_z["sigma_ci_graus"])
    rs = range(int(firma_z["realitzacions"]))
    noms, x_c, de_c = condicions_mesurades(pas0, pas02, dp, g)
    _, x_u, de_u = condicions_mesurades(pas0, pas02, base, g)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    sigma_de = lambda r2: math.hypot(r2["bootstrap"]["sigma_D"][0], r2["mc_alineacio"]["desviacio"])  # noqa: E731
    real = {"lambda": dp["resultat2"][g]["lambda"], "sigma": sigma_de(dp["resultat2"][g])}
    real_u = {"lambda": base["resultat2"][g]["lambda"], "sigma": sigma_de(base["resultat2"][g])}
    p0 = _parametres()
    ara = time.perf_counter()
    print(f"Els diagnostics de la Z (pla, 27/10 al vespre). SENSE FRICCIO i SENSE CAP VEREDICTE. s = {list(s)} graus, "
          f"{len(rs)} realitzacions; condicions de {de_c}\n", flush=True)
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processos) as pool:
        ref = r2_simulat(pool, x_c, sig, R, s_graus=s, rs=rs, Ds_diagnostic=R["D_diagnostic_rad"])
        d4 = {"s_0": r2_simulat(pool, x_c, sig, R, s_graus=(0.0, 0.0), rs=rs),
              "s_meitat": r2_simulat(pool, x_c, sig, R, s_graus=(s[0] / 2, s[1] / 2), rs=rs)}
        d4b = r2_simulat(pool, x_c, sig, R, s_graus=s, rs=rs, veritats=_veritats_contretes)
        d2 = r2_simulat(pool, x_u, sig, R, s_graus=s, rs=rs)
    d3 = {}
    for nom, q in im.variants(p0).items():
        if "cargols" in nom:      # fixats a 12,00 cm (en Guillem, 22/10): no son una incertesa del feix firmat
            continue
        with ctx.Pool(processos, initializer=_posa_p, initargs=(q,)) as pool:
            d3[nom] = r2_simulat(pool, x_c, sig, R, s_graus=s, rs=rs)
    print(f"   ({time.perf_counter() - ara:.0f} s)\n", flush=True)

    lam_ref = [x["lambda_D"][0] for x in ref]
    criteris = {"DG0": {"passa": bool(lam_ref == zr["simulat"]["lambda_realitzacions"]),
                        "que": "les 20 lambda de la referencia son exactament les de lyapunov_real_z_r2_2026-10-27.json"}}
    sR = resum_simulat(ref)
    lamR = sR["lambda"]

    def fila(res, rl=real):
        sm = resum_simulat(res)
        sc = math.hypot(rl["sigma"], sm["dispersio"])
        F = (lamR - sm["lambda"]) / (lamR - real["lambda"])
        nom = ("hi passa per sobre" if F > 1 else "n'explica la major part" if F >= 0.5 else
               "n'explica una part" if F >= 0.2 else "no l'explica")
        return {"lambda": sm["lambda"], "dispersio": sm["dispersio"], "lambda_realitzacions": sm["lambda_realitzacions"],
                "diferencia_sigmes": (rl["lambda"] - sm["lambda"]) / sc, "fraccio_explicada": F, "lectura": nom}

    # les distancies de sortida (graus, mediana de les parelles del R2 i de les realitzacions): mesurades, i de debo
    par_all = [(a, b) for a in range(len(x_c)) for b in range(a + 1, len(x_c))]
    sel = [k for k, (a, b) in enumerate(par_all) if dist_ang(x_c[a], x_c[b]) <= R["D_estrella_rad"] / math.e]

    def sortida(fv):
        v = []
        for r in rs:
            xv = fv(x_c, s, rg.llavor_de(float(g), r, 1, 23))
            v += [math.degrees(float(dist_ang(xv[par_all[k][0]], xv[par_all[k][1]]))) for k in sel]
        return float(np.median(v))
    mu, v, tau2, c = _contraccio(x_c, s)
    out = {"eina": "eines/lyapunov_real.py z_diagnostic",
           "pla": "documentacio/pla-analisi-caos.md (27/10 al vespre, «Els diagnostics de la Z»)",
           "que_es": "DIAGNOSTICS DE LA Z FIRMADA, fets despres del resultat: SENSE CAP VEREDICTE (el «no passa» firmat no "
                     "es mou), sense friccio. Real (dels JSON, cap presa oberta) contra simulat",
           "s_graus": list(s), "realitzacions": len(rs), "real": real, "real_sense_corregir": real_u,
           "referencia": {"lambda": lamR, "dispersio": sR["dispersio"], "lambda_realitzacions": lam_ref,
                          "diferencia_sigmes": (real["lambda"] - lamR) / math.hypot(real["sigma"], sR["dispersio"])},
           "D4": {k: fila(x) for k, x in d4.items()},
           "D4b": {**fila(d4b), "contraccio": {"mu_graus": np.degrees(mu).tolist(), "sd_graus": np.degrees(np.sqrt(v)).tolist(),
                                               "tau_graus": np.degrees(np.sqrt(tau2)).tolist(), "c": c.tolist(),
                                               "c_zero": bool(np.any(c == 0))}},
           "distancies_de_sortida_graus": {"mesurades": sortida(lambda x, _s, _l: x), "referencia": sortida(_veritats),
                                           "D4b": sortida(_veritats_contretes)},
           "D3": {k: fila(x) for k, x in d3.items()}}
    # D2: sense corregir, contra el 3,43
    s2 = resum_simulat(d2)
    gF = 1 - real["lambda"] / lamR
    g2 = 1 - real_u["lambda"] / s2["lambda"]
    out["D2"] = {"lambda": s2["lambda"], "dispersio": s2["dispersio"], "lambda_realitzacions": s2["lambda_realitzacions"],
                 "diferencia_sigmes": (real_u["lambda"] - s2["lambda"]) / math.hypot(real_u["sigma"], s2["dispersio"]),
                 "g_firmada": gF, "g2": g2,
                 "lectura": ("la correccio no hi te gaire a veure" if g2 >= 0.75 * gF else
                             "la diferencia ve sobretot de la correccio" if g2 <= 0.25 * gF else "en part")}
    # D1: la lambda(D) del model (mediana de les realitzacions de la referencia) contra la real corregida
    lsD = np.median(np.array([x["diagnostic_lambda_D"] for x in ref]), axis=0)
    lrD = np.array(dp["resultat2"][g]["diagnostic_lambda_D"]["lambda_D"])
    delta = 1 - lrD / lsD
    q = float(delta[0] / delta[-1])
    out["D1"] = {"D_graus": [math.degrees(x) for x in R["D_diagnostic_rad"]], "lambda_real_D": lrD.tolist(),
                 "lambda_model_D": lsD.tolist(), "delta_D": delta.tolist(), "rao_delta_primera_ultima": q,
                 "lectura": ("com un problema de d0" if q >= 1.15 else
                             "com un problema de ritme (del model)" if q <= 1.05 else "no decideix")}

    oberts, _OBERTS = _OBERTS, None
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    permesos = {"divergencia/" + c.name for c in (PAS0_JSON, PAS0B_JSON, DUES_PARTS_2710_JSON, REALS_BASE_JSON, Z_R2_JSON)}
    permesos.add("versemblant/firma.json")
    criteris["DG1"] = {"passa": bool(set(dins) <= permesos and not any(x.startswith("net/") for x in dins)),
                       "fitxers_oberts_de_dades": dins}
    out["criteris"] = criteris

    # la sortida
    print(f"   real (corregit) {real['lambda']:.3f} +- {real['sigma']:.3f}; sense corregir {real_u['lambda']:.3f}")
    print(f"   referencia (la Z firmada): {lamR:.3f}, dispersio {sR['dispersio']:.3f}, "
          f"{out['referencia']['diferencia_sigmes']:+.2f} sigma")
    ds = out["distancies_de_sortida_graus"]
    print(f"   distancies de sortida (mediana): mesurades {ds['mesurades']:.2f}, referencia {ds['referencia']:.2f}, "
          f"D4b {ds['D4b']:.2f} graus; contraccio c = {[round(x, 3) for x in c.tolist()]}")
    for nom, f in [("D4 s = 0", out["D4"]["s_0"]), ("D4 s/2", out["D4"]["s_meitat"]), ("D4b contraccio", out["D4b"]),
                   *[(f"D3 {k}", x) for k, x in out["D3"].items()]]:
        print(f"   {nom:<44} {f['lambda']:.3f} (disp {f['dispersio']:.3f}), {f['diferencia_sigmes']:+.2f} sigma, "
              f"F = {f['fraccio_explicada']:+.2f}: {f['lectura']}")
    d2o = out["D2"]
    print(f"   D2 sense corregir: model {d2o['lambda']:.3f} (disp {d2o['dispersio']:.3f}) contra {real_u['lambda']:.3f}, "
          f"{d2o['diferencia_sigmes']:+.2f} sigma; g2 = {d2o['g2']:.3f} contra g = {d2o['g_firmada']:.3f}: {d2o['lectura']}")
    d1 = out["D1"]
    print("   D1 lambda(D): " + "; ".join(f"{dg:.1f} graus real {a:.2f} model {b:.2f} (delta {dd:.3f})" for dg, a, b, dd in
                                         zip(d1["D_graus"], d1["lambda_real_D"], d1["lambda_model_D"], d1["delta_D"])))
    print(f"      delta(primera)/delta(ultima) = {q:.3f}: {d1['lectura']}")
    for k, cr in criteris.items():
        print(f"   {'ok ' if cr['passa'] else 'NO '} {k}: {cr.get('que', ', '.join(cr.get('fitxers_oberts_de_dades', [])))}")
    return out


def reals(*_a, **_k):
    """EL PANY: sense l'aprovacio d'en Guillem no es mira cap corba real."""
    if not APROVAT.exists():
        raise SystemExit(f"Pany tancat: no hi ha {APROVAT.relative_to(ARREL)}. Les corbes de separacio "
                         "reals nomes es miren quan en Guillem aprova els grups i els metodes (pas 2).")
    raise NotImplementedError("El pas 3 s'escriu quan hi hagi l'aprovacio: hi diu quins grups i metodes.")


# =============================================================== l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def _fabricats(llavor, N=12, lam=2.5, sd_tir=0.3, sd_par=0.2, d0_mig=0.03, lineal=False,
               t=None, frac_quiet=0.0):
    """Corbes de separacio fabricades: (t, d (P, M), d0 (P,), parelles, ritmes)."""
    rng = np.random.default_rng(llavor)
    t = np.arange(0.0, T_MAX_S + 0.5 * DT, DT) if t is None else t
    a = rng.normal(0.0, sd_tir, N)
    par = [(i, j) for i in range(N) for j in range(i + 1, N)]
    d, d0, rit = [], [], []
    quiets = set(rng.choice(len(par), int(round(frac_quiet * len(par))), replace=False).tolist())
    for k, (i, j) in enumerate(par):
        x0 = d0_mig * math.exp(rng.normal(0.0, 0.3))
        r = lam + 0.5 * (a[i] + a[j]) + rng.normal(0.0, sd_par)
        if k in quiets:
            d.append(np.full(len(t), x0))
        elif lineal:
            d.append(x0 * (1.0 + r * t))
        else:
            d.append(x0 * np.exp(r * t))
        d0.append(x0), rit.append(r)
    return t, np.array(d), np.array(d0), par, np.array(rit)


def _cobertura_una(llavor):
    """Una repeticio de LLa5: exponencials amb soroll multiplicatiu, lambda de poblacio 2,5."""
    t, d, d0, par, _ = _fabricats(llavor)
    d = d * np.exp(np.random.default_rng(llavor + 1).normal(0.0, 0.05, d.shape))
    Ds = np.geomspace(0.03 * math.e ** 1.2, 0.03 * math.e ** 2.4, N_D)
    r = avalua_llindars(t, d, d0, par, 12, Ds, n_boot=200, llavor=llavor + 2)
    return r["lambda_D"][2], r["bootstrap"]["sigma_D"][2]


def _presa_fabricada(c, inverteix, voltes=0, soroll_px=0.0, llavor=0):
    """BC1: un pendol que es mou 6 s i acaba quiet 2 s a (2 pi voltes, 2 pi voltes), amb la marca a c px
    de costat (fixa a la barra 1). Torna (t, marca, gomet) com posicions_del_trk (y avall) i els angles."""
    t = np.arange(0.0, 8.0, DT)
    env = np.clip(1.0 - t / 6.0, 0.0, None) ** 3
    th1 = 2.0 * np.cos(2 * np.pi * 0.9 * t) * env + 2 * np.pi * voltes * (1 - env)
    th2 = 2.5 * np.sin(2 * np.pi * 1.3 * t + 0.4) * env + 2 * np.pi * voltes * (1 - env)
    L1, L2 = 194.0, 168.0
    J = np.column_stack([L1 * np.sin(th1), -L1 * np.cos(th1)])
    M = J + c * np.column_stack([np.cos(th1), np.sin(th1)])
    G = J + np.column_stack([L2 * np.sin(th2), -L2 * np.cos(th2)])
    s = -1.0 if inverteix else 1.0
    a_imatge = lambda P: np.column_stack([s * P[:, 0], -P[:, 1]])  # noqa: E731
    M, G = a_imatge(M), a_imatge(G)
    if soroll_px:
        rng = np.random.default_rng(llavor)
        M = M + rng.normal(0.0, soroll_px, M.shape)
        G = G + rng.normal(0.0, soroll_px, G.shape)
    return t, M, G, (th1, th2)


def autoprova() -> int:
    """LLa1-LLa7 i VNa1-VNa5, escrits al pla el 17/10 abans d'escriure el codi (i H2, BC1, BC1b i BC2)."""
    global _OBERTS, APROVAT
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    sys.addaudithook(_registra)
    _OBERTS = []
    print("Autoprova de eines/lyapunov_real.py (criteris del pla, 17/10). Dades FABRICADES.\n")

    print("LLa1. El creuament d'una exponencial mostrejada a 240 fps, exacte a 1e-9 s")
    t = np.arange(0.0, 3.0 + 0.5 * DT, DT)
    err = 0.0
    for lam, d0, D in ((2.5, 0.02, 0.3), (1.3, 0.01, 0.05), (4.0, 0.03, 1.0)):
        T = creuaments(t, d0 * np.exp(lam * t), [D])[0, 0]
        err = max(err, abs(T - math.log(D / d0) / lam))
    mira(err < 1e-9, f"error maxim {err:.1e} s")

    print("\nLLa2. 12 tirs exponencials fabricats: lambda(D) a 2 % de la mediana dels ritmes, ratio 0,95-1,05")
    t, d, d0, par, rit = _fabricats(1)
    Ds = np.geomspace(0.03 * math.e ** 1.2, 0.03 * math.e ** 2.4, N_D)
    r = avalua_llindars(t, d, d0, par, 12, Ds, n_boot=0)
    rel = np.abs(np.array(r["lambda_D"]) / np.median(rit) - 1)
    mira(rel.max() < 0.02 and 0.95 <= r["ratio"] <= 1.05,
         f"lambda(D) {', '.join(f'{x:.4f}' for x in r['lambda_D'])} contra {np.median(rit):.4f}; ratio {r['ratio']:.4f} "
         f"({r['veredicte']})")

    print("\nLLa3. Separacio lineal amb 1,2 e de rang de D: ratio <= 0,5, amb D_min/d0 = e, 6 i 20")
    for R1 in (math.e, 6.0, 20.0):
        # 66 parelles amb la mateixa d0 i d = d0 (1 + c t): D_min/d0 es exactament R1.
        par = [(i, j) for i in range(12) for j in range(i + 1, 12)]
        d0m = 0.03
        d0 = np.full(len(par), d0m)
        Dl = np.geomspace(R1 * d0m, R1 * d0m * math.exp(RANG_MINIM_LN), N_D)
        tt = np.arange(0.0, 30.0 + 0.5 * DT, DT)
        dd = d0m * (1.0 + 2.5 * tt)[None, :].repeat(len(par), 0)
        r = avalua_llindars(tt, dd, d0, par, 12, Dl, n_boot=0, t_max=30.0)
        mira(r["ratio"] is not None and r["ratio"] <= RATIO_LIN,
             f"D_min/d0 = {R1:.2f}: ratio {r['ratio']:.3f} ({r['veredicte']})")

    print("\nLLa4. Amb un 30 % de parelles que no creuen, aquella D queda invalidada")
    t, d, d0, par, _ = _fabricats(3, frac_quiet=0.30)
    r = avalua_llindars(t, d, d0, par, 12, Ds, n_boot=0)
    mira(r["veredicte"] == "no creua" and abs(r["fraccio_creuen"][0] - 0.70) < 0.02,
         f"creuen {100 * r['fraccio_creuen'][0]:.0f} % a D_min -> {r['veredicte']}")

    print("\nLLa5. Cobertura del bootstrap per tirs (200 repeticions): 55-80 % dins d'1 sigma")
    ara = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(PROCESSOS) as pool:
        res = pool.map(_cobertura_una, [5000 + k for k in range(200)])
    # La lambda certa del metode a aquestes corbes es la mediana de la poblacio de
    # ritmes de parella, 2,5 (simetrica).
    dins = sum(1 for l, s in res if abs(l - 2.5) <= s)
    mira(0.55 <= dins / 200 <= 0.80, f"{dins} de 200 ({dins / 2:.0f} %), {time.perf_counter() - ara:.0f} s")

    print("\nLLa6. Reproduible bit a bit")
    t, d, d0, par, _ = _fabricats(4)
    a = json.dumps(avalua_llindars(t, d, d0, par, 12, Ds, n_boot=100, llavor=1))
    b = json.dumps(avalua_llindars(t, d, d0, par, 12, Ds, n_boot=100, llavor=1))
    mira(a == b, "dues passades identiques")

    # Reformulat el 17/10, despres de la fallada, amb la decisio d'en Guillem (l'rms del tram).
    # La versio d'abans comparava el nivell estacionari amb la desviacio de 6,5 s, que en un
    # soroll de rho 0,99 surt un 13 % mes petita: vegeu el pla.
    print("\nLLa7. El soroll fabricat: (a) 200 s, sigma a < 10 % i rho a +-0,02; (b) mesurat com les preses")
    print("      (rms de 0,9 s, mitjana quadratica de 200 finestres), el nivell a < 10 %")
    for sd, rho in (((0.0262, 0.0244), (0.990, 0.992)), ((0.0012, 0.0057), (0.70, 0.90)),
                    ((0.0087, 0.0123), (0.972, 0.972))):
        rng = np.random.default_rng(7)
        cru = soroll_cru(rng, int(200.0 * FPS), sd, rho)
        f = rg.com_la_conversio(cru)
        rel_a = np.array([f[k][50:-50].std(ddof=1) for k in range(2)]) / np.array(sd) - 1
        drho = np.array([float(np.corrcoef(x[:-1], x[1:])[0, 1]) for x in cru]) - np.array(rho)
        n_f = int(round((TRAM_S[1] - TRAM_S[0]) * FPS))
        n = int((DURADA_SIM_S + REPOS_S) * FPS) + 1
        ms = []
        for _ in range(200):
            g_ = rg.com_la_conversio(soroll_cru(rng, n, sd, rho))
            a0 = int(round((REPOS_S + TRAM_S[0]) * FPS))
            ms.append([np.mean(g_[k][a0:a0 + n_f] ** 2) for k in range(2)])
        rel_b = np.sqrt(np.mean(ms, axis=0)) / np.array(sd) - 1
        mira(np.all(np.abs(rel_a) < 0.10) and np.all(np.abs(drho) < 0.02) and np.all(np.abs(rel_b) < 0.10),
             f"sd {sd}, rho {rho}: (a) {100 * rel_a[0]:+.1f} / {100 * rel_a[1]:+.1f} %, rho "
             f"{drho[0]:+.3f} / {drho[1]:+.3f}; (b) {100 * rel_b[0]:+.1f} / {100 * rel_b[1]:+.1f} %")

    print("\nVNa1. Troba un vei plantat, i respecta el Theiler i l'energia")
    tt = np.arange(0.0, 10.0, DT)
    M = len(tt)
    A = np.column_stack([0.3 * tt, np.zeros(M), np.full(M, 0.3), np.zeros(M)])
    B = np.column_stack([0.3 * (tt + 4.0), np.full(M, 0.02), np.full(M, 0.3), np.zeros(M)])
    C = B.copy()
    C[:, 1] = 0.019          # mes a prop que el B: sense filtre d'energia el guanya
    Dd = np.column_stack([0.5 * np.sin(2 * np.pi * tt), 5.0 + 0.02 * tt, np.pi * np.cos(2 * np.pi * tt),
                          np.zeros(M)])
    Q = [A, B, C, Dd]
    E = [np.ones(M), np.ones(M), np.full(M, 5.0), np.ones(M)]
    H = int(round(0.2 / DT))
    V = [np.r_[np.ones(M - H, bool), np.zeros(H, bool)] for _ in Q]
    kw = dict(periodic=[True, True, False, False], sep=[0, 1], vel=[2, 3], banda=(0.015, 0.025), cada=10)
    par = troba_veins(Q, V, None, theiler=int(2.0 / DT), de_max=None, **kw)
    par_e = troba_veins(Q, V, [x / 1.0 for x in E], theiler=int(2.0 / DT), de_max=0.1, **kw)
    de_a = [p for p in par_e if p[0] == 0]
    # El B plantat nomes existeix per a les referencies de l'A a partir de 4 s (amb un
    # marge de 0,1 s): abans, el vei mes proper es la vora del B, i tambe es legitim.
    k4 = int(round(4.0 / DT))
    bons = all(p[2] == 1 and (p[1] < k4 + 24 or abs(p[3] - (p[1] - k4)) <= 1) for p in de_a)
    de_d = [p for p in par_e if p[0] == 3]
    par_t = troba_veins(Q, V, E, theiler=int(0.5 / DT), de_max=0.1, **kw)
    mira(len(de_a) > 0 and bons and any(p[2] == 2 for p in par) and not any(p[2] == 2 or p[0] == 2 for p in par_e),
         f"de l'A: {len(de_a)} referencies, totes amb el B 4 s abans; el C (energia fora) nomes surt sense filtre")
    mira(len(de_d) == 0 and any(p[0] == 3 and p[2] == 3 for p in par_t),
         f"el D (torna a si mateix a 1 s): {len(de_d)} parelles amb Theiler de 2 s, "
         f"{sum(p[0] == 3 for p in par_t)} amb 0,5 s")

    print("\nVNa6. (17/10, segona part) Dos tirs que surten junts no fan parella; desfasats 4 s, si")
    B0 = A.copy()
    B0[:, 1] = 0.02                       # el mateix tir, 0,02 mes enlla, AL MATEIX INSTANT
    TT = [tt, tt]
    p_junts = troba_veins([A, B0], V[:2], None, theiler=int(2.0 / DT), temps=TT, separacio_creuada=0.5, **kw)
    p_junts0 = troba_veins([A, B0], V[:2], None, theiler=int(2.0 / DT), **kw)
    p_desf = troba_veins([A, B], V[:2], None, theiler=int(2.0 / DT), temps=TT, separacio_creuada=0.5, **kw)
    mira(len(p_junts) == 0 and len(p_junts0) > 0 and len(p_desf) > 0,
         f"al mateix instant: {len(p_junts0)} parelles sense la restriccio i {len(p_junts)} amb; desfasats 4 s: {len(p_desf)}")

    print("\nVNa2. Lorenz, amb 10 trajectories: 0,906 a 10 %")
    from scipy.integrate import solve_ivp

    def lorenz(_, v):
        x, y, z = v
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]
    ara = time.perf_counter()
    dtl = 0.01
    rngl = np.random.default_rng(11)
    QL = []
    for _ in range(10):
        x0 = solve_ivp(lorenz, (0, 30), rngl.normal(0, 5, 3) + [1, 1, 20], method="DOP853", rtol=1e-10, atol=1e-10).y[:, -1]
        s = solve_ivp(lorenz, (0, 100), x0, method="DOP853", rtol=1e-10, atol=1e-12, t_eval=np.arange(0, 100, dtl))
        QL.append(s.y.T.copy())
    HL = int(8.0 / dtl)
    VL = [np.r_[np.ones(len(q) - HL, bool), np.zeros(HL, bool)] for q in QL]
    rs = np.random.default_rng(12)
    sostre = float(np.median([np.linalg.norm(QL[rs.integers(10)][rs.integers(len(QL[0]))] -
                                             QL[rs.integers(10)][rs.integers(len(QL[0]))]) for _ in range(4000)]))
    cfgL = {"periodic": [False] * 3, "sep": [0, 1, 2], "vel": [], "banda": (0.05, 0.075), "theiler": int(5 / dtl),
            "cada": 25, "H": HL, "h0": int(0.5 / dtl), "dt": dtl, "ln_sostre_3": math.log(sostre / 3)}
    rL = avalua_veins(QL, VL, None, cfgL, n_boot=0)
    errL = rL["lambda"] / 0.9056 - 1
    # 17/10: el criteri del 10 % va fallar (-10,5 %). En Guillem tria l'opcio 2 DESPRES de
    # veure-ho: la finestra es queda fins a sostre/3, el biaix es publica, i el VN1 de la
    # validacio (+-20 % contra la FTLE de la mateixa finestra) es el que decideix al pendol.
    print(f"   (info, biaix declarat) lambda {rL['lambda']:.4f} ({100 * errL:+.1f} %), {rL['parelles']} "
          f"parelles, finestra {rL['finestra_s'][0]:.2f}-{rL['finestra_s'][1]:.2f}, "
          f"{time.perf_counter() - ara:.0f} s")

    print("\nVNa3. Un oscil.lador isocron: |lambda| < 0,05")
    rngo = np.random.default_rng(13)
    to = np.arange(0.0, 50.0, DT)
    QO = []
    for _ in range(10):
        Am, ph = rngo.uniform(1.0, 1.2), rngo.uniform(0, 2 * np.pi)
        QO.append(np.column_stack([Am * np.cos(to + ph), -Am * np.sin(to + ph)]))
    HO = int(1.5 / DT)
    VO = [np.r_[np.ones(len(q) - HO, bool), np.zeros(HO, bool)] for q in QO]
    cfgO = {"periodic": [False, False], "sep": [0, 1], "vel": [], "banda": (0.03, 0.045), "theiler": int(2.0 / DT),
            "cada": 50, "H": HO, "h0": int(0.1 / DT), "dt": DT, "ln_sostre_3": math.log(1.5 / 3)}
    rO = avalua_veins(QO, VO, None, cfgO, n_boot=0)
    mira(abs(rO["lambda"]) < 0.05 and rO["parelles"] > 10, f"lambda {rO['lambda']:+.2e}, {rO['parelles']} parelles")

    print("\nVNa4. El bootstrap amb tots els pesos a 1 dona la mostra sencera")
    D_ = segueix(QL, rL["_parelles"], periodic=cfgL["periodic"], sep=cfgL["sep"], H=cfgL["H"])
    lnd = np.log(D_)
    lnr = lnd - lnd[:, :1]
    w = pesos_bootstrap([(p_[0], p_[2]) for p_ in rL["_parelles"]], np.arange(10), 10)
    b = pendent_veins(lnr, lnd, w, cfgL["h0"], dtl, cfgL["ln_sostre_3"])
    mira(np.all(w == 1) and abs(b["lambda"] / rL["lambda"] - 1) < 1e-12,
         f"amb els pesos d'un remostreig identic: {b['lambda']:.12f} contra {rL['lambda']:.12f}")

    print("\nVNa5. Reproduible bit a bit")
    a = avalua_veins(QO, VO, None, cfgO, n_boot=50, llavor=3)
    b = avalua_veins(QO, VO, None, cfgO, n_boot=50, llavor=3)
    mira(a["_parelles"] == b["_parelles"] and a["lambda"] == b["lambda"] and a["bootstrap"] == b["bootstrap"],
         "dues passades identiques")

    print("\nH2. (23/10) L'horitzo: (a) lambda constant, a 1e-15; (b) ln lambda ~ N(ln 2; 0,4), 1e5 mostres:")
    print("    mediana a 1 %, p16 i p84 a 2 %; (c) un 30 % de lambda <= 0 son horitzons infinits")
    h = horitzons(np.full(50, 2.18), 2.18)
    err = max(abs(v["p50_s"] / (v["ln_R"] / 2.18) - 1) + abs(v["central_s"] / (v["ln_R"] / 2.18) - 1)
              for k, v in h.items() if k.startswith("d0_"))
    mira(err <= 1e-15, f"(a) error maxim {err:.1e}")
    mu, s = math.log(2.0), 0.4
    h = horitzons(np.exp(np.random.default_rng(23).normal(mu, s, 100000)), 2.0)
    e = []
    for k, v in h.items():
        if k.startswith("d0_"):
            e += [abs(v["p50_s"] / (v["ln_R"] * math.exp(-mu)) - 1),
                  abs(v["p16_s"] / (v["ln_R"] * math.exp(-mu - s)) - 1),
                  abs(v["p84_s"] / (v["ln_R"] * math.exp(-mu + s)) - 1)]
    mira(e[0::3] and max(e[0::3]) <= 0.01 and max(e[1::3] + e[2::3]) <= 0.02,
         f"(b) mediana {100 * max(e[0::3]):.2f} %, p16/p84 {100 * max(e[1::3] + e[2::3]):.2f} %")
    lc = np.r_[np.full(30, -0.5), np.full(70, 2.0)]
    h = horitzons(lc, 2.0)
    v = h["d0_0.3_tol_3"]
    mira(abs(h["fraccio_lambda_no_positiva"] - 0.30) < 1e-12 and h["remostrejos"] == 100 and math.isinf(v["p84_s"])
         and math.isfinite(v["p16_s"]),
         f"(c) fraccio {h['fraccio_lambda_no_positiva']:.2f}, {h['remostrejos']} remostrejos, p16 {v['p16_s']:.3f} s, "
         f"p84 {v['p84_s']}")

    print("\nBC1. (24/10) El repos final a (0, 0), presa fabricada sense soroll: la c a 1e-9 px i els angles")
    print("     de debo a cada fotograma a 1e-9 rad; c = -10, -3, 0, +4 i +10,6 px, els dos signes, i 3 voltes")
    pitjor = [0.0, 0.0]
    for inv in (False, True):
        for c in (-10.0, -3.0, 0.0, 4.0, 10.6):
            for voltes in ((0, 3) if c == 10.6 else (0,)):
                t, M, G, th = _presa_fabricada(c, inv, voltes)
                k = corregeix_posicions(t, M, G, inv, int(round(2.0 * FPS)))
                e_ang = max(float(np.max(np.abs(dif_angular(k["th1_c"], th[0])))),
                            float(np.max(np.abs(dif_angular(k["th2_c"], th[1])))))
                pitjor = [max(pitjor[0], abs(k["c_px"] - c)), max(pitjor[1], e_ang)]
    mira(pitjor[0] <= 1e-9 and pitjor[1] <= 1e-9,
         f"error maxim de la c {pitjor[0]:.1e} px; dels angles {pitjor[1]:.1e} rad")

    print("\nBC1b. Amb soroll blanc de 0,5 px: la c a 0,1 px, i el biaix dels angles corregits <= 0,05 graus")
    pitjor = [0.0, 0.0]
    for inv, c, ll in ((False, 10.6, 1), (True, -6.0, 2), (False, 3.0, 3)):
        t, M, G, th = _presa_fabricada(c, inv, 0, soroll_px=0.5, llavor=ll)
        k = corregeix_posicions(t, M, G, inv, int(round(2.0 * FPS)))
        b = max(abs(float(np.mean(dif_angular(k["th1_c"], th[0])))),
                abs(float(np.mean(dif_angular(k["th2_c"], th[1])))))
        pitjor = [max(pitjor[0], abs(k["c_px"] - c)), max(pitjor[1], math.degrees(b))]
    mira(pitjor[0] <= 0.1 and pitjor[1] <= 0.05, f"error maxim de la c {pitjor[0]:.3f} px; biaix {pitjor[1]:.4f} graus")

    oberts, _OBERTS = _OBERTS, None
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)) for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    mira(not dins, "de dades/ no obre res" + ("" if not dins else ": " + ", ".join(dins)))

    print("\nVNa7. (22/10) Sense la firma de X, Y i Z, la Z dels veins es nega a funcionar abans d'obrir res")
    if FIRMA.exists():
        mira(True, "(hi ha firma.json: la prova del pany no s'hi pot fer)")
    else:
        _OBERTS = []
        try:
            z_veins({}, {}, {})
            negat = False
        except SystemExit:
            negat = True
        ob, _OBERTS = _OBERTS, None
        mira(negat and not ob, f"es nega: {negat}; fitxers oberts: {len(ob)}")

    print("\nBC2. (24/10) Sense l'aprovacio, corregit es nega a funcionar abans d'obrir res")
    desat, APROVAT = APROVAT, ARREL / "dades" / "divergencia" / "_no_existeix_aprovat.json"
    _OBERTS = []
    try:
        corregit({}, {}, {})
        negat = False
    except SystemExit:
        negat = True
    finally:
        APROVAT = desat
    ob, _OBERTS = _OBERTS, None
    mira(negat and not ob, f"es nega: {negat}; fitxers oberts: {len(ob)}")

    print("\nDQ3. (25/10) Sense l'aprovacio, dues_parts es nega a funcionar abans d'obrir res")
    desat, APROVAT = APROVAT, ARREL / "dades" / "divergencia" / "_no_existeix_aprovat.json"
    _OBERTS = []
    try:
        dues_parts({}, {}, {})
        negat = False
    except SystemExit:
        negat = True
    finally:
        APROVAT = desat
    ob, _OBERTS = _OBERTS, None
    mira(negat and not ob, f"es nega: {negat}; fitxers oberts: {len(ob)}")
    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> el creuament, la lambda(D), el veredicte, el bootstrap i els veins fan el que diuen.")
    return 0


PAS0_JSON = DIVERGENCIA / "lyapunov_real_pas0_2026-10-17.json"
PAS0B_JSON = DIVERGENCIA / "lyapunov_real_pas0b_2026-10-17.json"
VALIDACIO2_JSON = DIVERGENCIA / "lyapunov_real_validacio2_2026-10-17.json"
REALS_BASE_JSON = DIVERGENCIA / "lyapunov_real_reals_2026-10-24.json"


def autoprova_z(processos=PROCESSOS) -> int:
    """ZR1-ZR5, i la ZR4 i la ZR4b informatives, escrits al pla el 26/10 abans del codi. NOMES SIMULACIO.

    No escriu cap lambda simulada absoluta: nomes errors relatius i veredictes (la simulada no es posa al costat de la
    real fins que hi hagi firma.json).
    """
    global _OBERTS
    import criteri_z
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    sys.addaudithook(_registra)
    print("Autoprova de la Z de la lambda* (z_r2; pla, 26/10). NOMES SIMULACIO.\n")

    print("ZR1. Sense firma.json, z_r2 (directa i des de la linia d'ordres) i z_veins (linia d'ordres) s'aturen sense obrir res")
    if FIRMA.exists():
        mira(True, "(hi ha firma.json: la prova del pany no s'hi pot fer)")
    else:
        _OBERTS = []
        negats = []
        base = ["--pas0", str(PAS0_JSON), "--pas02", str(PAS0B_JSON), "--reals", str(REALS_BASE_JSON)]
        for crida in (lambda: z_r2({}, {}, {}, sigma_ci_graus=(0.0, 0.0)),
                      lambda: main(["z_r2", *base, "--sigma-ci-graus", "0,0"]),
                      lambda: main(["z_veins", *base])):
            try:
                crida()
                negats.append(False)
            except SystemExit as e:
                negats.append("Pany tancat" in str(e.code))
        ob, _OBERTS = _OBERTS, None
        mira(all(negats) and not ob, f"es neguen: {negats}; fitxers oberts: {len(ob)}")

    _OBERTS = []
    pas0 = json.loads(PAS0_JSON.read_text(encoding="utf-8"))
    pas02 = json.loads(PAS0B_JSON.read_text(encoding="utf-8"))
    val2 = json.loads(VALIDACIO2_JSON.read_text(encoding="utf-8"))
    g = Z_GRUP
    R = pas02["resultat2"][g]
    noms, x_mes, _ = condicions_mesurades(pas0, pas02, {}, g)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    S0, SC = (0.0, 0.0), Z_S_CANDIDATA
    ara = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        print("\nZR2. Amb s = 0, les 5 primeres realitzacions, bit a bit les amb_soroll del R2 de la validacio del 17/10")
        a = r2_simulat(pool, x_mes, sig, R, s_graus=S0)
        ref = val2["resultat2"][g]["amb_soroll"]
        iguals = [a[r]["lambda_D"][0] == ref[r]["lambda_D"][0] and a[r]["bootstrap"]["sigma_D"][0] == ref[r]["bootstrap"]["sigma_D"][0]
                  for r in range(5)]
        mira(all(iguals), f"lambda i sigma del bootstrap identiques a {sum(iguals)} de 5")
        xa = _veritats(x_mes, S0, rg.llavor_de(float(g), 0, 1, 23))
        xb = _veritats(x_mes, SC, rg.llavor_de(float(g), 0, 1, 23))
        mira(all(np.array_equal(u, v) for u, v in zip(xa, x_mes))
             and all(not np.array_equal(u[:2], v[:2]) and np.array_equal(u[2:], v[2:]) for u, v in zip(xb, x_mes)),
             "amb s = 0 surten de les mesurades tal qual; amb s > 0, de condicions diferents a tots els tirs (i omega = 0)")

        b = r2_simulat(pool, x_mes, sig, R, s_graus=SC)
        simA, simB = resum_simulat(a), resum_simulat(b)
        for nom, s in (("s = 0", simA), (f"s = {SC}", simB)):
            print(f"   (info) costat simulat, {nom}: error relatiu {100 * s['sigma'] / s['lambda']:.1f} % (bootstrap "
                  f"{100 * s['sigma_boot'] / s['lambda']:.1f} %, dispersio entre {s['realitzacions']} realitzacions "
                  f"{100 * s['dispersio'] / s['lambda']:.1f} %)")

        print("\nZR5. Reproduible: s > 0 amb 3 realitzacions, dues passades")
        b3 = r2_simulat(pool, x_mes, sig, R, s_graus=SC, rs=range(3))
        mira(json.dumps(_json_net(b3)) == json.dumps(_json_net(b[:3])), "identiques bit a bit")

        print("\nZR3. Un model perfecte passa: 20 bessons, dels avaluables 'no passa' com a molt el 10 %")
        bes, buides = {}, []
        for nom, s, sim in (("s = 0", S0, simA), (f"s = {SC}", SC, simB)):
            bes[nom] = bessons(pool, x_mes, sig, R, noms, s_graus=s, que=4)
            v = [criteri_z.veredicte(x["lambda"], x["sigma"], sim["lambda"], sim["sigma"])["veredicte"] for x in bes[nom]]
            n_av, n_np = sum(x != "no avaluable" for x in v), v.count("no passa")
            err = np.median([x["sigma"] / x["lambda"] for x in bes[nom]])
            if n_av < 10:
                print(f"   --  {nom}: nomes {n_av} de 20 avaluables: la ZR3 no es pot fer amb aquesta s, i la Z hi es "
                      f"'no avaluable' per construccio (error mitja dels bessons {100 * err:.0f} %)")
                buides.append(nom)
                continue
            mira(n_np <= 0.10 * n_av, f"{nom}: {v.count('passa')} passen, {n_np} no passen, {20 - n_av} no avaluables "
                                      f"(error mitja dels bessons {100 * err:.0f} %)")

        print("\nZR4b (informativa). Bessons amb error de mesura, s = " + str(SC) + ", contra el costat simulat de l'opcio (a)")
        v = [criteri_z.veredicte(x["lambda"], x["sigma"], simA["lambda"], simA["sigma"])["veredicte"] for x in bes[f"s = {SC}"]]
        print(f"   --  {v.count('passa')} passen, {v.count('no passa')} no passen, {v.count('no avaluable')} no avaluables, de 20")

        print("\nZR4 (informativa). Bessons d'un pendol k vegades mes rapid (la lambda per k): quants 'no passa'")
        for nom, s, sim in (("s = 0", S0, simA), (f"s = {SC}", SC, simB)):
            fila = []
            for j, k in enumerate(Z_K):
                bk = bessons(pool, x_mes, sig, R, noms, s_graus=s, rs=range(100 * j, 100 * j + Z_BESSONS_K), que=5,
                             escala=k)
                v = [criteri_z.veredicte(x["lambda"], x["sigma"], sim["lambda"], sim["sigma"])["veredicte"] for x in bk]
                fila.append(f"k {k:g}: {v.count('no passa')}/{Z_BESSONS_K - v.count('no avaluable')}")
            print(f"   --  {nom}: " + "; ".join(fila) + "  (no passen / avaluables)")
    print(f"\n   ({time.perf_counter() - ara:.0f} s)")

    oberts, _OBERTS = _OBERTS, None
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    permesos = {str(c.resolve().relative_to(arrel_dades)).replace("\\", "/") for c in (PAS0_JSON, PAS0B_JSON, VALIDACIO2_JSON)}
    mira(set(dins) <= permesos, "de dades/ nomes obre el pas 0, el pas 0b i la validacio del 17/10: " + ", ".join(dins))
    print()
    if problemes:
        print("AUTOPROVA DE LA Z FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> el pany tanca, el costat simulat es el de la validacio i es reproduible.")
    if buides:
        print(f"     PERO la ZR3 no s'ha pogut fer amb {' ni amb '.join(buides)}: cap besso no es avaluable, o sigui que")
        print("     amb la regla tal com es (20 %) la Z de la lambda* surt 'no avaluable' per construccio.")
    else:
        print("     I un model perfecte passa.")
    return 0


DUES_PARTS_2710_JSON = DIVERGENCIA / "lyapunov_real_dues_parts_2026-10-27.json"
Z_COSTAT_SIMULAT_JSON = DIVERGENCIA / "lyapunov_real_z_costat_simulat_2026-10-27.json"
Z_S_CREU = (1.045, 1.127)       # la creu dels 90 graus, com a suposicio declarada als 150 (cas C3)


def autoprova_z2(processos=PROCESSOS) -> int:
    """ZR3'-ZR6' de l'opcio 2 (pla, 27/10, «L'opcio 2 a z_r2»), escrits abans del codi. NOMES SIMULACIO, amb la
    configuracio C3: les condicions corregides i s = la creu dels 90 graus. No escriu cap lambda absoluta."""
    global _OBERTS, FIRMA
    import criteri_z
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    sys.addaudithook(_registra)
    print("Autoprova de l'opcio 2 de la Z (pla, 27/10). NOMES SIMULACIO.\n")

    print("ZR6'. El pany amb sigma_sim, amb una firma de prova a esborranys/ (z_r2 no es crida mai)")
    desat, prova = FIRMA, ARREL / "esborranys" / "_firma_de_prova_zr6.json"

    def deixa(f):
        try:
            _firma_z(sigma_sim=f)
            return True
        except SystemExit:
            return False
    try:
        FIRMA = prova
        prova.write_text(json.dumps({"X": 3.0, "Y": "no avaluable", "Z": {"sigma_sim": "dispersio"}}), encoding="utf-8")
        refusa_altra, deixa_igual = not deixa("regla_04_10"), deixa("dispersio")
        prova.write_text(json.dumps({"X": 3.0, "Y": "", "Z": {"sigma_sim": "dispersio"}}), encoding="utf-8")
        refusa_y_buida = not deixa("dispersio")
    finally:
        FIRMA = desat
        prova.unlink(missing_ok=True)
    mira(refusa_altra and deixa_igual and refusa_y_buida and not prova.exists(),
         f"refusa una sigma_sim diferent: {refusa_altra}; deixa passar la mateixa: {deixa_igual}; refusa la Y buida: "
         f"{refusa_y_buida}; la firma de prova, esborrada: {not prova.exists()}")

    _OBERTS = []
    carrega = lambda c: json.loads(Path(c).read_text(encoding="utf-8"))  # noqa: E731
    pas0, pas02 = carrega(PAS0_JSON), carrega(PAS0B_JSON)
    zc = carrega(Z_COSTAT_SIMULAT_JSON)
    g = Z_GRUP
    R = pas02["resultat2"][g]
    noms, x_mes, de = condicions_mesurades(pas0, pas02, carrega(DUES_PARTS_2710_JSON), g)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    print(f"\nLa configuracio C3: condicions de {de}; s = {Z_S_CREU} graus")
    ara = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        print("\nZR5'. El costat simulat de la C3 dona exactament els errors relatius desats el 27/10")
        s3 = resum_simulat(r2_simulat(pool, x_mes, sig, R, s_graus=Z_S_CREU))
        rel = {"bootstrap": s3["sigma_boot"] / s3["lambda"], "dispersio": s3["dispersio"] / s3["lambda"],
               "regla_04_10": s3["sigma"] / s3["lambda"], "opcio_2": s3["dispersio"] / s3["lambda"]}
        mira(rel == zc["casos"]["C3"]["error_relatiu"], f"iguals: {rel == zc['casos']['C3']['error_relatiu']} "
                                                         f"(opcio 2: {100 * rel['opcio_2']:.1f} %)")

        print("\nZR3'. Un model perfecte passa: 20 bessons, dels avaluables 'no passa' com a molt el 10 %")
        bes = bessons(pool, x_mes, sig, R, noms, s_graus=Z_S_CREU, que=4)
        v = [criteri_z.veredicte(x["lambda"], x["sigma"], s3["lambda"], s3["dispersio"])["veredicte"] for x in bes]
        n_av, n_np = sum(x != "no avaluable" for x in v), v.count("no passa")
        err = [x["sigma"] / x["lambda"] for x in bes]
        detall = (f"{v.count('passa')} passen, {n_np} no passen, {20 - n_av} no avaluables; error dels bessons: mediana "
                  f"{100 * np.median(err):.0f} %, {sum(e <= criteri_z.ERROR_MAXIM for e in err)} de 20 per sota del 20 %")
        buida = n_av < 10
        if buida:
            print(f"   --  nomes {n_av} de 20 avaluables: la ZR3' no es pot fer ({detall})")
        else:
            mira(n_np <= 0.10 * n_av, detall)

        print("\nZR4b' (informativa). Els mateixos bessons contra el costat simulat de la C3 amb s = 0 (l'error de les")
        print("      condicions inicials, ignorat)")
        s0 = resum_simulat(r2_simulat(pool, x_mes, sig, R, s_graus=(0.0, 0.0)))
        v0 = [criteri_z.veredicte(x["lambda"], x["sigma"], s0["lambda"], s0["dispersio"])["veredicte"] for x in bes]
        print(f"   --  {v0.count('passa')} passen, {v0.count('no passa')} no passen, {v0.count('no avaluable')} no avaluables, "
              f"de 20 (amb s = 0, l'opcio 2 dona {100 * s0['dispersio'] / s0['lambda']:.1f} %)")

        print("\nZR4' (informativa). Bessons d'un pendol k vegades mes rapid (la lambda per k): quants 'no passa'")
        fila = []
        for j, k in enumerate(Z_K):
            bk = bessons(pool, x_mes, sig, R, noms, s_graus=Z_S_CREU, rs=range(100 * j, 100 * j + Z_BESSONS_K), que=5,
                         escala=k)
            vk = [criteri_z.veredicte(x["lambda"], x["sigma"], s3["lambda"], s3["dispersio"])["veredicte"] for x in bk]
            fila.append(f"k {k:g}: {vk.count('no passa')}/{Z_BESSONS_K - vk.count('no avaluable')}")
        print("   --  " + "; ".join(fila) + "  (no passen / avaluables)")
    print(f"\n   ({time.perf_counter() - ara:.0f} s)")

    oberts, _OBERTS = _OBERTS, None
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    permesos = {"divergencia/" + c.name for c in (PAS0_JSON, PAS0B_JSON, DUES_PARTS_2710_JSON, Z_COSTAT_SIMULAT_JSON)}
    mira(set(dins) <= permesos, "de dades/ nomes obre el que diu el pla: " + ", ".join(dins))
    print()
    if problemes:
        print("AUTOPROVA DE L'OPCIO 2 FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> el pany mira la sigma_sim i el costat simulat es el desat.")
    print("     PERO la ZR3' no s'ha pogut fer: no hi ha prou bessons avaluables." if buida else
          "     I un model perfecte passa amb l'opcio 2.")
    return 0


Z_R2_JSON = DIVERGENCIA / "lyapunov_real_z_r2_2026-10-27.json"


def z_bessons(processos=PROCESSOS) -> dict:
    """(29/10) L'error de cada besso de la Z (pla, «29/10 · Sis feines», F6): els 20 bessons de la ZR3' (la C3, amb les
    mateixes llavors), amb la lambda, la sigma i l'error relatiu de cada un, al costat del de la presa real, que surt
    del JSON de la Z. NOMES SIMULACIO, mes aquell JSON: no obre cap presa, i SENSE CAP VEREDICTE.

    ZB1: reprodueix el que va imprimir la ZR3' el 27/10 (8 de 20 per sota del 20 %, mediana del 21 %). ZB2: de dades/
    nomes obre els JSON de l'--autoprova-z2 i el de la Z."""
    global _OBERTS
    import criteri_z
    sys.addaudithook(_registra)
    _OBERTS = []
    carrega = lambda c: json.loads(Path(c).read_text(encoding="utf-8"))  # noqa: E731
    pas0, pas02, z = carrega(PAS0_JSON), carrega(PAS0B_JSON), carrega(Z_R2_JSON)
    R = pas02["resultat2"][Z_GRUP]
    noms, x_mes, de = condicions_mesurades(pas0, pas02, carrega(DUES_PARTS_2710_JSON), Z_GRUP)
    sig = np.array([pas0["per_presa"][n]["sigma_amollada_s"] for n in noms])
    ara = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        bes = bessons(pool, x_mes, sig, R, noms, s_graus=Z_S_CREU, que=4)
    oberts, _OBERTS = _OBERTS, None
    err = [x["sigma"] / x["lambda"] for x in bes]
    e_real = z["real"]["sigma"] / z["real"]["lambda"]
    per_sota = sum(e < e_real for e in err)
    percentil = 100.0 * per_sota / len(err)
    lectura = ("per sota del percentil 10: els tirs reals es dispersen menys del que el model fa esperar des d'aquestes "
               "condicions (coherent amb la D4b: van sortir mes junts que la mesura)" if percentil < 10 else
               "per sobre del percentil 90: els tirs reals es dispersen mes del que el model fa esperar" if percentil > 90
               else "entre el percentil 10 i el 90: dins del que el model fa esperar")
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    permesos = {"divergencia/" + c.name for c in (PAS0_JSON, PAS0B_JSON, DUES_PARTS_2710_JSON, Z_COSTAT_SIMULAT_JSON,
                                                  Z_R2_JSON)}
    sota_20 = sum(e <= criteri_z.ERROR_MAXIM for e in err)
    criteris = {"ZB1": {"passa": bool(f"{100 * np.median(err):.0f}" == "21" and sota_20 == 8),
                        "que": "reprodueix la ZR3' del 27/10: 8 de 20 per sota del 20 %, mediana del 21 %",
                        "detall": f"{sota_20} de 20 per sota del 20 %, mediana {100 * np.median(err):.1f} %"},
                "ZB2": {"passa": set(dins) <= permesos, "que": "de dades/ nomes obre el que diu el pla",
                        "fitxers_oberts_de_dades": dins}}
    print(f"Els 20 bessons de la ZR3' (condicions de {de}; s = {Z_S_CREU} graus; {time.perf_counter() - ara:.0f} s)")
    for i, (x, e) in enumerate(zip(bes, err)):
        print(f"   {i:2d}  lambda {x['lambda']:.3f}  sigma {x['sigma']:.3f}  error {100 * e:5.1f} %")
    print(f"\nLa presa real: error {100 * e_real:.1f} %. Hi ha {per_sota} bessons de 20 amb menys error (percentil "
          f"{percentil:.0f}): {lectura}")
    for k, c in criteris.items():
        print(f"   {'ok ' if c['passa'] else 'NO '} {k}: {c['que']} ({c.get('detall', ', '.join(dins))})")
    return {"eina": "eines/lyapunov_real.py z_bessons", "pla": "documentacio/pla-analisi-caos.md (29/10, «Sis feines», F6)",
            "que_es": "l'error de cada besso de la Z (la C3), al costat del de la presa real: NOMES SIMULACIO, mes el JSON "
                      "de la Z; cap presa oberta i CAP VEREDICTE",
            "condicions_de": de, "s_graus": list(Z_S_CREU),
            "bessons": [{"lambda": x["lambda"], "sigma": x["sigma"], "error_relatiu": e} for x, e in zip(bes, err)],
            "error_bessons": {"mediana": float(np.median(err)), "minim": min(err), "maxim": max(err),
                              "per_sota_del_20_pc": sota_20},
            "real": {"lambda": z["real"]["lambda"], "sigma": z["real"]["sigma"], "error_relatiu": e_real,
                     "de": "dades/divergencia/" + Z_R2_JSON.name},
            "bessons_amb_menys_error": per_sota, "percentil": percentil, "lectura": lectura,
            "regla_de_lectura": "escrita abans (pla, 29/10): < 10 per sota; 10-90 dins; > 90 per sobre",
            "criteris": criteris}


# =============================================================== main

def _json_net(v):
    if isinstance(v, dict):
        return {str(k): _json_net(x) for k, x in v.items() if not str(k).startswith("_")}
    if isinstance(v, (list, tuple)):
        return [_json_net(x) for x in v]
    if isinstance(v, np.ndarray):
        return _json_net(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def main(argv=None) -> int:
    global _OBERTS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ordre", nargs="?", choices=("descriu", "valida", "descriu2", "valida2", "friccio2", "reals",
                                                 "robustesa", "z_veins", "biaix", "horitzo", "corregit", "dues_parts",
                                                 "z_r2", "z_costat_simulat", "z_diagnostic", "z_bessons"))
    ap.add_argument("--friccio-de-prova", action="store_true",
                    help="z_veins i z_r2: el costat simulat amb friccio DE PROVA (a z_r2, nomes com a informacio)")
    ap.add_argument("--sigma-ci-graus", help="z_r2, obligatori: la sigma de les condicions de debo al voltant de les "
                                             "mesurades, 's1,s2' en graus (0,0 es l'opcio a)")
    ap.add_argument("--realitzacions", type=int, default=Z_REALITZACIONS, help="z_r2: les realitzacions del costat simulat")
    ap.add_argument("--autoprova-z", action="store_true", help="ZR1-ZR5 de la Z de la lambda* (nomes simulacio)")
    ap.add_argument("--sigma-sim", choices=SIGMES_SIM, default="regla_04_10",
                    help="z_r2: la sigma del costat simulat (regla_04_10, o dispersio = l'opcio 2 del 27/10)")
    ap.add_argument("--autoprova-z2", action="store_true", help="ZR3'-ZR6' de l'opcio 2 (nomes simulacio)")
    ap.add_argument("--corregit", action="store_true",
                    help="horitzo: amb les preses corregides de preses_corregides() (28/10); --reals, el dues_parts del 27/10")
    ap.add_argument("--reals", help="el JSON de reals (per a robustesa)")
    ap.add_argument("--validacio2", help="el JSON de valida2 (per a friccio2)")
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--pas0", help="el JSON del pas 0 (per a valida, descriu2, valida2 i reals)")
    ap.add_argument("--pas02", help="el JSON del pas 0 de la segona part (per a valida2 i reals)")
    ap.add_argument("--processos", type=int, default=PROCESSOS)
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.processos > 6:
        ap.error("com a molt 6 processos")
    if args.autoprova_z:
        return autoprova_z(args.processos)
    if args.autoprova_z2:
        return autoprova_z2(args.processos)
    s_ci = None
    if args.ordre == "z_r2":
        if not args.sigma_ci_graus:
            ap.error("z_r2 demana --sigma-ci-graus s1,s2 (0,0 es l'opcio a): ho decideix la firma")
        s_ci = tuple(float(x) for x in args.sigma_ci_graus.split(","))
        if len(s_ci) != 2 or min(s_ci) < 0:
            ap.error("--sigma-ci-graus: dos valors no negatius, s1,s2")
    # EL PANY de la Z, abans d'obrir cap fitxer (ZR1): els JSON es carreguen despres.
    if args.ordre == "z_r2":
        _firma_z(args.reals, s_ci, args.realitzacions, args.sigma_sim)
    elif args.ordre in ("z_veins", "z_diagnostic"):
        _firma_z()
    carrega = lambda c: json.loads(Path(c).read_text(encoding="utf-8"))  # noqa: E731
    if args.ordre == "z_diagnostic":
        r = z_diagnostic(args.processos)
        if not all(c["passa"] for c in r["criteris"].values()):
            args.json = None
            print("Un control FALLA: no es desa res. Un criteri que falla NO es relaxa.")
    elif args.ordre == "z_bessons":
        r = z_bessons(args.processos)
        if not all(c["passa"] for c in r["criteris"].values()):
            args.json = None
            print("Un criteri FALLA: no es desa res. Un criteri que falla NO es relaxa.")
    elif args.ordre == "z_costat_simulat":
        r = z_casos(args.processos)
        if not all(c["passa"] for c in r["criteris"].values()):
            args.json = None
            print("Un criteri FALLA: no es desa res. Un criteri que falla NO es relaxa.")
    elif args.ordre == "z_r2":
        r = z_r2(carrega(args.pas0), carrega(args.pas02), carrega(args.reals), sigma_ci_graus=s_ci,
                 nom_reals=args.reals, realitzacions=args.realitzacions,
                 amb_friccio_de_prova=args.friccio_de_prova, sigma_sim=args.sigma_sim, processos=args.processos)
    elif args.ordre == "reals":
        if not APROVAT.exists():
            reals()
        r = reals2(carrega(args.pas0), carrega(args.pas02))
        print(json.dumps(_json_net(r), indent=1, ensure_ascii=False))
    elif args.ordre == "biaix":
        r = biaix(carrega(args.pas0), carrega(args.pas02), carrega(args.reals))
    elif args.ordre == "horitzo":
        r = horitzo(carrega(args.pas02), carrega(args.reals), corregit=args.corregit)
        if not r["criteris"]["H1"]["passa"] or not r["criteris"]["H3"]["passa"]:
            args.json = None
            print("No es desa res.")
    elif args.ordre == "corregit":
        r = corregit(carrega(args.pas0), carrega(args.pas02), carrega(args.reals))
    elif args.ordre == "dues_parts":
        r = dues_parts(carrega(args.pas0), carrega(args.pas02), carrega(args.reals))
        if not r["criteris"]["DQ2"]["passa"]:
            args.json = None
            print("No es desa res.")
    elif args.ordre == "z_veins":
        r = z_veins(carrega(args.pas0), carrega(args.pas02), carrega(args.reals),
                    amb_friccio_de_prova=args.friccio_de_prova, processos=args.processos)
        print(json.dumps(_json_net(r), indent=1, ensure_ascii=False))
    elif args.ordre == "robustesa":
        principal = carrega(args.reals)["resultat1"]["1,7-3,5"]["lambda"]
        r = robustesa(carrega(args.pas0), carrega(args.pas02), principal)
    elif args.ordre == "friccio2":
        r = friccio_r1(carrega(args.pas0), carrega(args.pas02), carrega(args.validacio2), processos=args.processos)
        i = r["ideal"]
        print(f"Franja {r['franja']}, {len(r['tirs'])} tirs amb friccio DE PROVA; temps a la franja "
              f"{min(r['temps_a_la_franja_s']):.1f}-{max(r['temps_a_la_franja_s']):.1f} s per tir")
        print(f"   sense soroll: {i['lambda']:.3f} (finestra {i['finestra_s']}, creix {i['creixement_ln']:.2f}, "
              f"{i['parelles']} parelles de {i['tirs']} tirs)")
        for d in r["amb_soroll"]:
            print(f"   amb soroll: {d['lambda']:.3f} +- {(d.get('bootstrap') or {}).get('sigma')} "
                  f"(creix {d['creixement_ln']:.2f}, {d['parelles']} parelles)")
        c = r["criteri"]
        print(f"   {'ok ' if c['passa'] else 'NO '} {c['criteri']}: {c['detall']}")
    elif args.ordre == "descriu2":
        r = descriu2(carrega(args.pas0))
        imprimeix_pas02(r)
    elif args.ordre == "valida2":
        sys.addaudithook(_registra)
        _OBERTS = []
        r = valida2(carrega(args.pas0), carrega(args.pas02), args.processos)
        oberts, _OBERTS = _OBERTS, None
        arrel_dades = (ARREL / "dades").resolve()
        r["fitxers_oberts_de_dades"] = sorted({str(Path(x).resolve().relative_to(arrel_dades)) for x in oberts
                                              if Path(x).resolve().is_relative_to(arrel_dades)})
        print("\nDe dades/ ha obert: " + ", ".join(r["fitxers_oberts_de_dades"]))
        r = {"eina": "eines/lyapunov_real.py valida2", "pla": "documentacio/pla-analisi-caos.md (17/10, segona part)",
             "que_es": "PAS 1 dels dos resultats: NOMES SIMULACIO, sense friccio (la de prova, nomes informativa)", **r}
    elif args.ordre == "descriu":
        r = descriu()
        imprimeix_pas0(r)
    elif args.ordre == "valida":
        if not args.pas0:
            ap.error("valida demana --pas0")
        sys.addaudithook(_registra)
        _OBERTS = []
        pas0 = json.loads(Path(args.pas0).read_text(encoding="utf-8"))
        r = valida(pas0, args.processos)
        oberts, _OBERTS = _OBERTS, None
        arrel_dades = (ARREL / "dades").resolve()
        r["fitxers_oberts_de_dades"] = sorted({str(Path(x).resolve().relative_to(arrel_dades)) for x in oberts
                                              if Path(x).resolve().is_relative_to(arrel_dades)})
        print("\nDe dades/ ha obert: " + ", ".join(r["fitxers_oberts_de_dades"]))
        r = {"eina": "eines/lyapunov_real.py valida", "pla": "documentacio/pla-analisi-caos.md (17/10)",
             "que_es": "PAS 1: NOMES SIMULACIO, sense friccio (la friccio, DE PROVA i nomes informativa); "
                       "cap trajectoria real", **r}
    else:
        ap.error("digues una ordre: descriu, valida o reals (o --autoprova)")
    if args.json:
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        cami.write_text(json.dumps(_json_net(r), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
