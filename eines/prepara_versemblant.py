"""Els numeros per firmar X, Y i Z, sense mirar cap comparacio.

Per que existeix. El deure 4 de documentacio/tancar-el-treball.html (seccio
11) demana escriure que vol dir «versemblant» ABANS de comparar cap simulacio
amb la realitat: la simulacio segueix el tir dins de X graus de theta2 durant
Y segons, i la Z de les divergencies. Els numeros d'orientacio que hi havia
eren de preses velles, d'abans de corregir la marca de l'articulacio, i
l'amplada del feix (+-0,3 graus) no sortia de cap incertesa mesurada. Aqui es
mesuren tres coses, i la tria continua sent d'en Guillem:

  A. EL SOROLL DE SEGUIMENT de cada grup. Per sota de 3 sigmes del soroll
     filtrat de theta2, el criteri X ni tan sols es podria comprovar.
  B. LA INCERTESA DE LES CONDICIONS INICIALS de cada presa, que es el que ha
     de fer d'amplada del feix. L'estadistica, de la mitjana del repos, i a
     part els termes sistematics, SENSE SUMAR-LOS. El biaix de theta1 es
     mesura de dues maneres que no quadren del tot -la marca de costat, per
     rigidesa, i el theta1 del repos final, on el model diu 0- i no esta
     resolt.
  C. L'HORITZO DEL FEIX, NOMES DE SIMULACIO. Si el tir real fos un membre
     qualsevol del feix d'un model perfecte, quant de temps seguiria theta2
     dins de X. Es el sostre de Y: amb un Y mes llarg, el criteri no el
     passaria ni el model perfecte.

LA BARRERA. Aixo no mira cap comparacio entre simulacio i realitat. De les
preses reals nomes es llegeix el JSON i, del CSV, el REPOS d'abans d'amollar:
repos_de_la_presa() es l'unica funcio que obre un CSV real, i l'autoprova
comprova que no en torna res de despres. Les simulacions surten dels angles
de repos mesurats, pero no es comparen amb res real.

Z NO ES AQUI. Demana l'estimador de l'experiment 4 -<ln(d/d0)> sobre
parelles- i la lambda no s'ha de comencar sense que en Guillem ho demani.

EL LIMIT DE C: el model es SENSE FRICCIO, que es l'unic que hi ha
(pendol/fisica.py, pas 2). Amb friccio el pendol perd energia i els feixos
s'haurien de separar mes a poc a poc. C es un sostre orientatiu, no una
prediccio.

Les decisions que hi ha a sota -els blocs de dues finestres, els 60 membres,
la llavor pel nom- es van prendre abans de mirar cap resultat, i son a
documentacio/pla-numeros-versemblant.md.

Us:
    .venv\\Scripts\\python eines\\prepara_versemblant.py
    .venv\\Scripts\\python eines\\prepara_versemblant.py --json dades/versemblant/numeros.json
    .venv\\Scripts\\python eines\\prepara_versemblant.py --membres 20 --processos 4
    .venv\\Scripts\\python eines\\prepara_versemblant.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import dades  # noqa: E402
from pendol.fisica import Dissipacio, Friccio, Parametres  # noqa: E402
from pendol.simulador import ATOL, RTOL, simula  # noqa: E402

INDEX = ARREL / "dades" / "net" / "preses.json"

# Les 10 preses de 45 del 25/09/2026, marcades al centre del cargol. El seu
# soroll va a part: es cinc vegades mes petit i barrejar-lo amb el de les
# velles faria una mediana que no descriu cap de les dues maneres de marcar.
NOVES_45 = ("45_1", "45_2", "45_3", "45_4", "45_5", "45_6", "45_7", "45_8",
            "45_10", "45_11")

# C. L'horitzo del feix.
X_GRAUS = (2.0, 3.0, 5.0)
AMPLADES_GRAUS = (0.03, 0.1, 0.3, 1.0)
MEMBRES = 200                # el que diu el criteri: 200 simulacions per feix (26/09)
MEMBRES_C = 60               # C es va fer el 25/09 amb 60, i amb els mateixos 60 es continua
TEMPS_S = 15.0
FPS_SIM = 240.0              # la graella de les preses
LLAVOR = 20260925

# B. L'estimador de l'error de la mitjana del repos.
FINESTRES_PER_BLOC = 2       # amb una sola, dos blocs veins compartirien filtre
BLOCS_MINIMS = 4
SIGMES_AMOLLADA = 3.0        # marge del repos respecte de l'amollada


# ----------------------------------------------------------- A. el soroll

def guany_savgol(finestra: int, ordre: int) -> float:
    """Quant redueix el soroll blanc el filtre de la conversio, al centre."""
    return float(np.sqrt(np.sum(savgol_coeffs(finestra, ordre) ** 2)))


def soroll_per_grup(index: list[dict], metes: dict) -> dict:
    grups: dict[str, list] = {}
    for fila in index:
        nom = fila["nom"]
        grup = f"{fila['grup_graus']:.0f}"
        if grup == "45":
            grup = "45 noves" if nom in NOVES_45 else "45 velles"
        sg = metes[nom]["savgol"]
        g = guany_savgol(sg["finestra_mostres"], sg["ordre"])
        s1, s2 = fila["soroll_angular_graus"]
        grups.setdefault(grup, []).append((nom, s1, s2, g))

    fora = {}
    for grup, files in grups.items():
        s1 = np.array([f[1] for f in files])
        s2 = np.array([f[2] for f in files])
        g = np.array([f[3] for f in files])
        fora[grup] = {
            "preses": [f[0] for f in files],
            "fotograma_theta1_graus": _estad(s1),
            "fotograma_theta2_graus": _estad(s2),
            "guany_del_filtre": _estad(g),
            "filtrat_theta1_graus": _estad(s1 * g),
            "filtrat_theta2_graus": _estad(s2 * g),
            "tres_sigmes_filtrat_theta2_graus": _estad(3.0 * s2 * g),
        }
    return fora


def _estad(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"n": int(len(v)), "mediana": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max())}


def _estad_o_res(v) -> dict | None:
    return _estad(v) if len(v) else None


# ------------------------------------- B. les condicions inicials

def repos_de_la_presa(cami_csv, amollada_s: float, incertesa_s: float,
                      finestra_mostres: int, fps: float):
    """(t, theta1, theta2) del REPOS d'una presa real, i res mes.

    ES L'UNICA FUNCIO D'AQUEST FITXER QUE OBRE UN CSV REAL, i nomes en torna
    les mostres d'abans d'amollar: la barrera de no mirar cap comparacio
    depen d'aixo, i l'autoprova ho comprova. Es deixa fora:
      - la primera mitja finestra, on el filtre treballa a la vora;
      - l'ultima mitja finestra abans d'amollar, que el filtre ja barreja amb
        el moviment, i a mes tres vegades la incertesa de l'amollada.
    """
    taula, _ = dades.carrega(cami_csv)
    t = taula["t"].to_numpy()
    mitja = 0.5 * finestra_mostres / fps
    fi = amollada_s - mitja - SIGMES_AMOLLADA * incertesa_s
    sel = (t >= mitja) & (t < fi)
    return (t[sel], taula["theta1"].to_numpy()[sel], taula["theta2"].to_numpy()[sel])


def error_de_la_mitjana(x, bloc: int):
    """(mitjana, error de la mitjana) per blocs, o (mitjana, None) si no n'hi ha prou.

    Les mostres filtrades estan correlacionades, o sigui que std/sqrt(N) faria
    l'error massa petit. Amb mitjanes de blocs de dues finestres de filtre, els
    blocs ja son gairebe independents.
    """
    x = np.asarray(x, dtype=float)
    n = len(x) // bloc
    if n < BLOCS_MINIMS:
        return (float(x.mean()) if len(x) else None), None
    mitjanes = x[: n * bloc].reshape(n, bloc).mean(axis=1)
    return float(mitjanes.mean()), float(mitjanes.std(ddof=1) / math.sqrt(n))


def condicions_inicials(nom: str, fila: dict, meta: dict) -> dict:
    a = meta["amollada"]
    fps = float(meta["fps"])
    sg = meta["savgol"]
    finestra = int(sg["finestra_mostres"])
    g = guany_savgol(finestra, int(sg["ordre"]))
    t, th1, th2 = repos_de_la_presa(ARREL / "dades" / "net" / f"{nom}.csv",
                                    a["amollada_s"], a["incertesa_s"], finestra, fps)
    bloc = FINESTRES_PER_BLOC * finestra
    _, e1 = error_de_la_mitjana(np.degrees(th1), bloc)
    _, e2 = error_de_la_mitjana(np.degrees(th2), bloc)
    s1, s2 = fila["soroll_angular_graus"]
    curt = e1 is None or e2 is None
    if curt:
        # La cota conservadora: el soroll filtrat d'un sol fotograma.
        e1, e2 = s1 * g, s2 * g

    m = meta["geometria"]["marca_articulacio"]
    costat = math.degrees(math.atan2(m["de_costat_px"], m["L1_px"]))
    # La segona mesura del mateix error, i sense cap model de marques: el
    # theta1 que dona la conversio quan el pendol penja quiet al final. Alli el
    # model diu theta1 = 0, o sigui que tot el que en surti es biaix de theta1.
    # Es el mateix numero que tracker.py escriu com «de costat, al repos».
    repos_final = math.degrees(math.atan2(fila["articulacio_delta_px"],
                                          meta["geometria"]["L1_px"]))
    delta_px = m["incertesa_delta_px"] if m["corregida"] else abs(m["delta_px"])
    terme_delta = math.degrees(delta_px / m["gomet_px"])

    return {
        "angles_graus": [float(v) for v in a["angles_amollada_graus"]],
        "repos_llegit_s": float(len(t) / fps),
        "mostres_de_repos": int(len(t)),
        "curt": bool(curt),
        "estadistica_graus": [float(e1), float(e2)],
        "costat_theta1_graus": float(costat),
        "costat_px": float(m["de_costat_px"]),
        "repos_final_theta1_graus": float(repos_final),
        "delta_theta2_graus": float(terme_delta),
        "marca_corregida": bool(m["corregida"]),
        "vertical_graus": float(meta["geometria"]["estabilitat_graus"]),
    }


def ci_per_grup(ci: dict, grup_de: dict) -> dict:
    fora = {}
    for grup in sorted(set(grup_de.values()), key=float):
        noms = [n for n in ci if grup_de[n] == grup]
        # Les mesurades i les cotes, per separat: barrejar-les faria una
        # mediana que no es ni una incertesa ni una cota.
        mesurades = [n for n in noms if not ci[n]["curt"]]
        curtes = [n for n in noms if ci[n]["curt"]]
        costat = [ci[n]["costat_theta1_graus"] for n in noms]
        final = [ci[n]["repos_final_theta1_graus"] for n in noms]
        fora[grup] = {
            "preses": noms,
            "mesurades": mesurades,
            "curtes": curtes,
            "estadistica_theta1_graus": _estad_o_res([ci[n]["estadistica_graus"][0] for n in mesurades]),
            "estadistica_theta2_graus": _estad_o_res([ci[n]["estadistica_graus"][1] for n in mesurades]),
            "cota_theta1_graus": _estad_o_res([ci[n]["estadistica_graus"][0] for n in curtes]),
            "cota_theta2_graus": _estad_o_res([ci[n]["estadistica_graus"][1] for n in curtes]),
            "costat_theta1_graus": _estad(costat),
            "costat_theta1_abs_graus": _estad(np.abs(costat)),
            "repos_final_theta1_graus": _estad(final),
            "repos_final_theta1_abs_graus": _estad(np.abs(final)),
            "delta_theta2_graus": _estad([ci[n]["delta_theta2_graus"] for n in noms]),
            "vertical_graus": _estad([ci[n]["vertical_graus"] for n in noms]),
        }
    return fora


# ----------------------------------------------- C. l'horitzo del feix

def llavor_de(nom: str) -> int:
    """Pel nom, no per la posicio: una presa nova no ha de canviar les altres."""
    return LLAVOR + zlib.crc32(nom.encode("utf-8"))


def temps_de_sortida(t, th2_central, th2_membre, x_rad: float) -> float:
    """El primer instant en que |th2 - th2 central| > X. inf si no en surt.

    theta2 va DESEMBOLICAT, com surt de la integracio: una volta de
    diferencia no es seguir el tir, encara que el pendol sigui al mateix lloc.
    """
    fora = np.nonzero(np.abs(th2_membre - th2_central) > x_rad)[0]
    return round(float(t[fora[0]]), 6) if len(fora) else math.inf


def primer_instant(t, v, llindar: float) -> float:
    """El primer instant en que v passa del llindar. inf si no en passa mai."""
    fora = np.nonzero(v > llindar)[0]
    return round(float(t[fora[0]]), 6) if len(fora) else math.inf


# EL MUNTATGE (06/10/2026, documentacio/pla-feix-amb-muntatge.md). Decidit per en
# Guillem: el feix de la prova del model porta, a mes de la incertesa de les
# condicions inicials, la del muntatge, amb les quatre variants
# d'eines/incerteses_muntatge.py. Cada membre sorteja cada parametre UNIFORME
# entre els dos extrems, independents. Declarat: el banc i els cargols poden
# comptar el mateix error dues vegades. El biaix de theta1 NO hi es.
CARGOLS_LLUNY_M = 0.1278      # on poden caure els caps dels cargols (CLAUDE.md)
I2_DEL_PERIODE = 1.0298       # la barra 2 com diu el seu periode
G_DEL_BANC = 1.0 - 0.0089     # tot el pendol com el banc
L1_REGLE_M = 0.001            # la precisio del regle
LLAVOR_MUNTATGE = 6102026     # la seva propia font: no toca la de les condicions inicials


def muntatge_de(p: Parametres, collapsat: bool = False) -> dict:
    """Els rangs uniformes de cada parametre. Collapsat: tots d'amplada zero."""
    if collapsat:
        return {"rs": (p.rs, p.rs), "I2_factor": (1.0, 1.0), "g_factor": (1.0, 1.0),
                "L1": (p.L1, p.L1)}
    return {"rs": (p.rs, CARGOLS_LLUNY_M), "I2_factor": (1.0, I2_DEL_PERIODE),
            "g_factor": (G_DEL_BANC, 1.0), "L1": (p.L1 - L1_REGLE_M, p.L1 + L1_REGLE_M)}


def mostra_muntatge(p: Parametres, rangs: dict, llavor: int, n: int) -> list:
    """Els Parametres de n membres, cadascun amb el seu muntatge sortejat.

    La I2 es puja allargant R2, com fa incerteses_muntatge.py: aixi nomes canvia
    la inercia de la barra 2 i no res mes de la dinamica. Amb rangs d'amplada
    zero, cada membre te exactament els parametres nominals.
    """
    u = np.random.default_rng([llavor, LLAVOR_MUNTATGE]).random((n, 4))
    sortida = []
    for a, b, c, d in u:
        (rs0, rs1), (f0, f1) = rangs["rs"], rangs["I2_factor"]
        (g0, g1), (l0, l1) = rangs["g_factor"], rangs["L1"]
        f = f0 + b * (f1 - f0)
        R2 = p.R2 * math.sqrt(1.0 + (f - 1.0) * p.I2 / ((1 / 12) * p.m2 * p.R2**2))
        sortida.append(replace(p, rs=rs0 + a * (rs1 - rs0), R2=R2,
                               g=p.g * (g0 + c * (g1 - g0)), L1=l0 + d * (l1 - l0)))
    return sortida


def horitzo_de_la_presa(feina):
    """Un feix per amplada al voltant d'una condicio inicial. Va en un proces.

    Tots els membres de totes les amplades fan servir els MATEIXOS nombres
    normals, escalats per l'amplada: aixi la diferencia entre amplades no
    barreja l'efecte de l'amplada amb el de la sort.

    Torna dues coses:
      C  el temps de sortida de cada un dels primers membres_c membres
         (|theta2 - theta2 central| > X), com el 25/09;
      D  L'HORITZO SEGONS LA DEFINICIO CORREGIDA D'EN GUILLEM (26/09): el
         primer instant en que el FEIX MATEIX s'eixampla mes de X, amb
         l'amplada com a maxim - minim entre tots els membres i el central.
         Es dona per a theta2 i theta1, i a mes l'amplada de la banda del
         2,5 al 97,5 %, que es la que dibuixen les figures.
    Els primers membres_c nombres normals son els mateixos que si nomes se'n
    generessin membres_c, o sigui que C no depen de quants membres te D.
    """
    nom, p, estat0, amplades, xs, membres, membres_c, temps, fps, llavor, *resta = feina
    # La dissipacio es opcional i va al final: sense, la feina es la d'abans del
    # 27/09 i dona el mateix. Amb, es NOMES una prova (--friccio).
    dis = resta[0] if resta else None
    # El muntatge (06/10) tambe es opcional, i va darrere de la dissipacio. El
    # central es sempre el nominal; cada membre te els seus parametres, els
    # mateixos a totes les amplades.
    rangs = resta[1] if len(resta) > 1 else None
    pm = mostra_muntatge(p, rangs, llavor, membres) if rangs is not None else [p] * membres
    estat0 = np.asarray(estat0, dtype=float)
    t, y0 = simula(p, estat0, temps, fps, rtol=RTOL, atol=ATOL, dis=dis)
    z = np.random.default_rng(llavor).standard_normal((membres, 2))
    fora_c, fora_d = {}, {}
    for amplada in amplades:
        sigma = math.radians(amplada)
        sortides = {x: [] for x in xs}
        th1s, th2s = [y0[0]], [y0[1]]
        for i in range(membres):
            estat = estat0.copy()
            estat[0] += sigma * z[i, 0]
            estat[1] += sigma * z[i, 1]
            _, y = simula(pm[i], estat, temps, fps, rtol=RTOL, atol=ATOL, dis=dis)
            th1s.append(y[0])
            th2s.append(y[1])
            if i < membres_c:
                for x in xs:
                    sortides[x].append(temps_de_sortida(t, y0[1], y[1], math.radians(x)))
        a1, a2 = np.array(th1s), np.array(th2s)
        w1 = a1.max(axis=0) - a1.min(axis=0)
        w2 = a2.max(axis=0) - a2.min(axis=0)
        b2 = np.percentile(a2, 97.5, axis=0) - np.percentile(a2, 2.5, axis=0)
        fora_c[amplada] = sortides
        fora_d[amplada] = {
            x: {"minmax_theta2_s": primer_instant(t, w2, math.radians(x)),
                "minmax_theta1_s": primer_instant(t, w1, math.radians(x)),
                "banda95_theta2_s": primer_instant(t, b2, math.radians(x))}
            for x in xs
        }
    return nom, fora_c, fora_d


def quantil(v, q: float) -> float:
    """Sense interpolar: el valor d'un membre de debo, i inf si es inf.

    Interpolar entre un temps i un inf donaria inf o nan segons el cas.
    """
    return float(np.percentile(np.asarray(v, dtype=float), q, method="inverted_cdf"))


def resum_de_presa(sortides: dict) -> dict:
    return {
        amplada: {x: {"mediana_s": quantil(v, 50), "q25_s": quantil(v, 25)}
                  for x, v in per_x.items()}
        for amplada, per_x in sortides.items()
    }


def horitzo_per_grup(resums: dict, grup_de: dict) -> dict:
    fora = {}
    for grup in sorted(set(grup_de[n] for n in resums), key=float):
        noms = [n for n in resums if grup_de[n] == grup]
        fora[grup] = {}
        for amplada in AMPLADES_GRAUS:
            fora[grup][amplada] = {}
            for x in X_GRAUS:
                med = [resums[n][amplada][x]["mediana_s"] for n in noms]
                q25 = [resums[n][amplada][x]["q25_s"] for n in noms]
                fora[grup][amplada][x] = {
                    "n": len(noms),
                    "mediana_de_medianes_s": quantil(med, 50),
                    "min_mediana_s": float(min(med)),
                    "max_mediana_s": float(max(med)),
                    "mediana_de_q25_s": quantil(q25, 50),
                }
    return fora


def horitzo_d_per_grup(d_preses: dict, grup_de: dict) -> dict:
    fora = {}
    for grup in sorted(set(grup_de[n] for n in d_preses), key=float):
        noms = [n for n in d_preses if grup_de[n] == grup]
        fora[grup] = {}
        for amplada in AMPLADES_GRAUS:
            fora[grup][amplada] = {}
            for x in X_GRAUS:
                fila = {"n": len(noms)}
                for clau in ("minmax_theta2_s", "minmax_theta1_s", "banda95_theta2_s"):
                    v = [d_preses[n][amplada][x][clau] for n in noms]
                    fila[clau] = {"mediana": quantil(v, 50), "min": float(min(v)), "max": float(max(v))}
                fora[grup][amplada][x] = fila
    return fora


def calcula_horitzons(p, ci: dict, membres: int, processos: int, dis=None, rangs=None):
    if rangs is not None:
        extra = (dis, rangs)
    else:
        extra = (dis,) if dis is not None else ()
    feines = [
        (nom, p, [*np.radians(c["angles_graus"]), 0.0, 0.0], AMPLADES_GRAUS,
         X_GRAUS, membres, min(MEMBRES_C, membres), TEMPS_S, FPS_SIM, llavor_de(nom), *extra)
        for nom, c in ci.items()
    ]
    total = len(feines) * (1 + membres * len(AMPLADES_GRAUS))
    print(f"\nC i D. {len(feines)} preses x (1 + {membres} membres x {len(AMPLADES_GRAUS)} "
          f"amplades) = {total} simulacions de {TEMPS_S:.0f} s, amb {processos} processos",
          flush=True)
    t0 = time.perf_counter()
    sortides, amplades_d = {}, {}
    with multiprocessing.get_context("spawn").Pool(processos) as pool:
        for k, (nom, s, d) in enumerate(pool.imap_unordered(horitzo_de_la_presa, feines), 1):
            sortides[nom], amplades_d[nom] = s, d
            print(f"   {k:>2}/{len(feines)}  {nom:<7} {time.perf_counter() - t0:6.0f} s",
                  flush=True)
    return {nom: sortides[nom] for nom in ci}, {nom: amplades_d[nom] for nom in ci}


# ------------------------------------------------------------- sortida

def _s(v: float) -> str:
    return f"> {TEMPS_S:.0f}" if math.isinf(v) else f"{v:.2f}"


def _json_net(v):
    """inf -> None, i claus numeriques -> text, perque el JSON sigui valid."""
    if isinstance(v, dict):
        return {(f"{k:g}" if isinstance(k, float) else str(k)): _json_net(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_net(x) for x in v]
    if isinstance(v, float) and math.isinf(v):
        return None
    return v


def imprimeix(soroll: dict, ci: dict, ci_grup: dict, hor_grup: dict | None) -> None:
    print("\nA. SOROLL DE SEGUIMENT (graus). Fotograma: de la rigidesa de les barres;")
    print("   filtrat: x el guany del Savitzky-Golay, suposant soroll blanc.")
    print(f"   {'grup':<10} {'n':>3} {'th1 fotog.':>17} {'th2 fotog.':>17} "
          f"{'th1 filtr.':>11} {'th2 filtr.':>11} {'3 sigma th2 filtr.':>19}")
    for grup in sorted(soroll, key=lambda g: (float(g.split()[0]), g)):
        s = soroll[grup]
        f1, f2 = s["fotograma_theta1_graus"], s["fotograma_theta2_graus"]
        print(f"   {grup:<10} {f1['n']:>3} {f1['mediana']:6.2f} ({f1['min']:.2f}-{f1['max']:.2f})"
              f" {f2['mediana']:6.2f} ({f2['min']:.2f}-{f2['max']:.2f})"
              f" {s['filtrat_theta1_graus']['mediana']:>11.3f}"
              f" {s['filtrat_theta2_graus']['mediana']:>11.3f}"
              f"   {s['tres_sigmes_filtrat_theta2_graus']['mediana']:.2f}"
              f" (max {s['tres_sigmes_filtrat_theta2_graus']['max']:.2f})")

    print("\nB. INCERTESA DE LES CONDICIONS INICIALS (graus), preses que arrenquen del repos.")
    print("   Estadistica: error de la mitjana del repos; si el repos es curt, la cota d'un")
    print("   fotograma filtrat. Els sistematics van A PART i no se sumen. El de theta1 es")
    print("   mesura de dues maneres i NO ESTA RESOLT:")
    print("     costat      la marca de l'articulacio fora de l'eix, de costat (rigidesa);")
    print("     repos final el theta1 que dona la conversio amb el pendol penjant quiet al")
    print("                 final, on el model diu 0: el biaix total, sense model de marques.")
    print(f"   {'presa':<7} {'repos s':>7} {'estad. th1':>10} {'th2':>7} {'costat th1':>11} "
          f"{'(px)':>7} {'repos final th1':>16} {'delta th2':>10} {'vertical':>9}")
    for nom, c in ci.items():
        e1, e2 = c["estadistica_graus"]
        marca = " curt: cota" if c["curt"] else ""
        print(f"   {nom:<7} {c['repos_llegit_s']:>7.2f} {e1:>10.4f} {e2:>7.4f} "
              f"{c['costat_theta1_graus']:>+11.2f} {c['costat_px']:>+7.1f} "
              f"{c['repos_final_theta1_graus']:>+16.2f} "
              f"{c['delta_theta2_graus']:>10.3f} {c['vertical_graus']:>9.4f}{marca}")

    def _mm(e, fmt=".4f"):
        return "-" if e is None else f"{e['mediana']:{fmt}} ({e['max']:{fmt}})"

    print(f"\n   {'grup':<5} {'mesurades':>9} {'estad. th1 med (max)':>21} {'th2':>17}"
          f" {'curtes':>7} {'cota th1':>17}   {'|costat| th1':>13} {'repos final th1':>21}"
          f" {'delta th2 max':>14}")
    for grup, g in ci_grup.items():
        rf = g["repos_final_theta1_graus"]
        print(f"   {grup:<5} {len(g['mesurades']):>9} {_mm(g['estadistica_theta1_graus']):>21}"
              f" {_mm(g['estadistica_theta2_graus']):>17} {len(g['curtes']):>7}"
              f" {_mm(g['cota_theta1_graus']):>17}   {_mm(g['costat_theta1_abs_graus'], '.2f'):>13}"
              f"   {rf['min']:+.2f} a {rf['max']:+.2f} (|med| {g['repos_final_theta1_abs_graus']['mediana']:.2f})"
              f" {g['delta_theta2_graus']['max']:>9.3f}")

    if hor_grup is None:
        return
    print("\nC. HORITZO DEL FEIX, NOMES SIMULACIO I SENSE FRICCIO (segons).")
    print("   Quant de temps segueix theta2 dins de X un membre qualsevol del feix.")
    print("   Mediana sobre les preses de la mediana de cada presa [rang de medianes];")
    print("   entre claudators rodons, la mediana del quartil 25 (el membre que surt aviat).")
    for grup, per_amplada in hor_grup.items():
        n = next(iter(next(iter(per_amplada.values())).values()))["n"]
        print(f"\n   {grup} graus ({n} preses)")
        print(f"   {'amplada':>9}" + "".join(f"{f'X = {x:g} graus':>28}" for x in X_GRAUS))
        for amplada, per_x in per_amplada.items():
            cel = []
            for x in X_GRAUS:
                h = per_x[x]
                cel.append(f"{_s(h['mediana_de_medianes_s'])} [{_s(h['min_mediana_s'])}-"
                           f"{_s(h['max_mediana_s'])}] ({_s(h['mediana_de_q25_s'])})")
            print(f"   {amplada:>7g} g" + "".join(f"{c:>28}" for c in cel))


def imprimeix_d(d_grup: dict, membres: int, amb_friccio: bool = False, titol: str | None = None) -> None:
    if titol:
        print("\n" + "=" * 78 + f"\n {titol}\n" + "=" * 78)
    print(f"\nD. HORITZO DE PREDICCIO: quan el FEIX MATEIX s'eixampla mes de X (segons).")
    print(f"   Amplada = maxim - minim de theta2 entre els {membres} membres i el central.")
    print("   Mediana sobre les preses [rang]; entre parentesis, la banda del 2,5 al 97,5 %")
    print("   (la de les figures), que arriba mes tard. Nomes simulacio i "
          + ("AMB FRICCIO DE PROVA." if amb_friccio else "sense friccio."))
    for grup, per_amplada in d_grup.items():
        n = next(iter(next(iter(per_amplada.values())).values()))["n"]
        print(f"\n   {grup} graus ({n} preses)")
        print(f"   {'amplada':>9}" + "".join(f"{f'X = {x:g} graus':>28}" for x in X_GRAUS))
        for amplada, per_x in per_amplada.items():
            cel = []
            for x in X_GRAUS:
                m, b = per_x[x]["minmax_theta2_s"], per_x[x]["banda95_theta2_s"]
                cel.append(f"{_s(m['mediana'])} [{_s(m['min'])}-{_s(m['max'])}] ({_s(b['mediana'])})")
            print(f"   {amplada:>7g} g" + "".join(f"{c:>28}" for c in cel))


# ------------------------------------------------------------ autoprova

def autoprova() -> int:
    """Els criteris del pla, escrits abans de mirar (1-4 el 25/09, 5-6 el 26/09)."""
    problemes = []
    p = Parametres.des_de_json(ARREL / "parametres.json")

    # 1. L'estimador recupera l'error de la mitjana d'un soroll blanc filtrat
    #    com les preses (13 mostres, grau 3, 1 s de repos a 240 fps), dins
    #    del 20 %.
    rng = np.random.default_rng(1)
    finestra, bloc, mitja = 13, FINESTRES_PER_BLOC * 13, 13 // 2
    mitjanes, errors = [], []
    for _ in range(400):
        y = savgol_filter(rng.normal(0.0, 0.3, 240), finestra, 3)[mitja:-mitja]
        m, e = error_de_la_mitjana(y, bloc)
        mitjanes.append(m)
        errors.append(e)
    rao = float(np.mean(errors)) / float(np.std(mitjanes, ddof=1))
    ok = 0.8 <= rao <= 1.2
    print(f"1. error de la mitjana: estimat / de debo = {rao:.3f} (0,8-1,2)  "
          f"{'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append(f"l'estimador del repos erra un {100 * abs(rao - 1):.0f} %")

    # 2. Un feix d'amplada 0 no surt mai. I (5, del 26/09) tampoc no s'eixampla mai.
    _, s, d0 = horitzo_de_la_presa(("prova", p, np.radians([150.0, 150.0, 0.0, 0.0]),
                                    (0.0,), X_GRAUS, 4, 4, 5.0, FPS_SIM, 7))
    tots = [v for per_x in s.values() for vs in per_x.values() for v in vs]
    ok = all(math.isinf(v) for v in tots)
    print(f"2. feix d'amplada 0: {sum(math.isinf(v) for v in tots)}/{len(tots)} "
          f"no en surten  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("un feix d'amplada 0 surt: hi ha alguna cosa que no es el feix")
    dtots = [v for per_x in d0.values() for fila in per_x.values() for v in fila.values()]
    ok = all(math.isinf(v) for v in dtots)
    print(f"5. feix d'amplada 0: {sum(math.isinf(v) for v in dtots)}/{len(dtots)} amplades "
          f"que no passen mai de X  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("un feix d'amplada 0 s'eixampla: l'amplada no es la del feix")

    # 3. Amb una X mes gran surt mes tard, i amb una amplada mes gran, abans.
    #    Perque no passi en buit, el feix d'1 grau ha de sortir dins dels 8 s.
    _, s, d = horitzo_de_la_presa(("prova", p, np.radians([150.0, 150.0, 0.0, 0.0]),
                                   (0.03, 1.0), (2.0, 5.0), 12, 12, 8.0, FPS_SIM, 11))
    # 6 (26/09). El central es dins del feix, o sigui que l'amplada sempre es com
    #    a minim la desviacio de qualsevol membre: D amb X no pot arribar mes
    #    tard que el primer membre que surt a C amb la mateixa X.
    ok6 = all(d[a][x]["minmax_theta2_s"] <= min(s[a][x]) for a in (0.03, 1.0) for x in (2.0, 5.0))
    print("6. D <= primer membre de C: " + ", ".join(
        f"{a:g} g X={x:g}: {_s(d[a][x]['minmax_theta2_s'])} <= {_s(min(s[a][x]))}"
        for a in (0.03, 1.0) for x in (2.0, 5.0)) + f"  {'OK' if ok6 else 'FALLA'}")
    if not ok6:
        problemes.append("D arriba mes tard que el primer membre de C: l'amplada esta mal calculada")
    med = {(a, x): quantil(s[a][x], 50) for a in (0.03, 1.0) for x in (2.0, 5.0)}
    ok_x = all(med[(a, 5.0)] >= med[(a, 2.0)] for a in (0.03, 1.0))
    ok_a = all(med[(1.0, x)] <= med[(0.03, x)] for x in (2.0, 5.0))
    ok_n = all(math.isfinite(med[(1.0, x)]) for x in (2.0, 5.0))
    ok = ok_x and ok_a and ok_n
    print("3. medianes (s): " + ", ".join(f"{a:g} g X={x:g}: {_s(v)}" for (a, x), v in med.items())
          + f"  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append(f"l'ordre no es el que toca (X {ok_x}, amplada {ok_a}, surt {ok_n})")

    # 4. La funcio que llegeix el repos no torna res de despres d'amollar.
    desti = ARREL / "esborranys" / "_autoprova_versemblant"
    t = np.arange(0.0, 3.0, 1.0 / 240.0)
    amollada = 1.0
    th = np.where(t < amollada, 0.5, 0.5 - 2.0 * (t - amollada) ** 2)
    cami = dades.desa(desti / "fals.csv", t, th, th, np.zeros_like(t), np.zeros_like(t),
                      {"origen": "autoprova de prepara_versemblant"})
    tr, r1, r2 = repos_de_la_presa(cami, amollada, 0.005, 13, 240.0)
    limit = amollada - 0.5 * 13 / 240.0 - SIGMES_AMOLLADA * 0.005
    ok = len(tr) > 0 and tr.max() < limit and np.all(r1 == 0.5) and np.all(r2 == 0.5)
    print(f"4. repos llegit: {len(tr)} mostres, fins a {tr.max():.4f} s (amollada a "
          f"{amollada} s, limit {limit:.4f})  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("la funcio del repos torna mostres de despres d'amollar")

    # 7-10 (06/10/2026): el muntatge. Criteris de documentacio/pla-feix-amb-muntatge.md,
    # escrits abans de mirar.
    ci150 = np.radians([150.0, 150.0, 0.0, 0.0])
    feina = ("prova", p, ci150, (0.03, 0.3), X_GRAUS, 6, 6, 4.0, FPS_SIM, 13)
    sense = horitzo_de_la_presa(feina)
    collapsat = horitzo_de_la_presa((*feina, None, muntatge_de(p, collapsat=True)))
    ok = sense[1] == collapsat[1] and sense[2] == collapsat[2]
    print(f"7. muntatge collapsat: el mateix que sense muntatge  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("amb el muntatge collapsat no surt el mateix que sense: la font "
                         "del muntatge toca alguna cosa mes")

    _, _, d_m = horitzo_de_la_presa(("prova", p, ci150, (0.0,), (2.0,), 12, 12, 8.0, FPS_SIM, 17,
                                     None, muntatge_de(p)))
    t2 = d_m[0.0][2.0]["minmax_theta2_s"]
    ok = math.isfinite(t2)
    print(f"8. amplada 0 amb muntatge: el feix passa de 2 graus a {_s(t2)} s  "
          f"{'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("amb el muntatge actiu i amplada zero el feix no s'eixampla: "
                         "el muntatge no hi entra")

    rangs = muntatge_de(p)
    pm = mostra_muntatge(p, rangs, 99, 200)
    valors = {"rs": [q.rs for q in pm], "I2_factor": [q.I2 / p.I2 for q in pm],
              "g_factor": [q.g / p.g for q in pm], "L1": [q.L1 for q in pm]}
    mal = []
    for k, (lo, hi) in rangs.items():
        v = np.array(valors[k])
        marge = 1e-12 * max(1.0, abs(hi))
        dins = v.min() >= lo - marge and v.max() <= hi + marge
        cobreix = v.min() - lo <= 0.05 * (hi - lo) and hi - v.max() <= 0.05 * (hi - lo)
        if not (dins and cobreix):
            mal.append(k)
    ok = not mal
    print(f"9. els parametres sortejats son dins dels rangs i els cobreixen "
          f"{'' if ok else '(' + ', '.join(mal) + ') '} {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append(f"el mostreig del muntatge no compleix el rang: {mal}")

    feina_m = (*feina, None, rangs)
    ok = horitzo_de_la_presa(feina_m)[1:] == horitzo_de_la_presa(feina_m)[1:]
    print(f"10. amb muntatge, dues passades donen el mateix  {'OK' if ok else 'FALLA'}")
    if not ok:
        problemes.append("amb el muntatge no es determinista")

    if problemes:
        print("\nAUTOPROVA FALLADA:")
        for pr in problemes:
            print(f"  - {pr}")
        return 1
    print("\nAutoprova OK.")
    return 0


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    global TEMPS_S, AMPLADES_GRAUS   # --temps i --amplades els canvien (27/09)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar els numeros")
    ap.add_argument("--membres", type=int, default=MEMBRES,
                    help=f"membres de cada feix (per defecte {MEMBRES})")
    ap.add_argument("--processos", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)),
                    help="processos per a les simulacions (per defecte, la meitat dels nuclis, com a molt 6)")
    ap.add_argument("--sense-feix", action="store_true",
                    help="nomes A i B, sense simular res")
    # Les quatre de sota son del 27/09 (documentacio/pla-feina-sense-intervencio.md,
    # punt 3): per mirar com canvia l'horitzo amb la friccio. Els coeficients hi
    # van A MA i son de prova; sense cap d'aquestes opcions, l'eina es la d'abans.
    ap.add_argument("--temps", type=float, default=TEMPS_S,
                    help=f"segons que es simulen (per defecte {TEMPS_S:g})")
    ap.add_argument("--amplades", help="amplades del feix en graus, separades per comes "
                    f"(per defecte {','.join(f'{a:g}' for a in AMPLADES_GRAUS)})")
    ap.add_argument("--friccio", action="store_true",
                    help="simula amb friccio: rodaments de parametres.json i la C i la eps "
                         "que es donin. Coeficients de PROVA, no adoptats")
    ap.add_argument("--aire-C", type=float, help="amb --friccio, obligatori")
    ap.add_argument("--epsilon", type=float, help="amb --friccio, obligatori")
    ap.add_argument("--sense-viscos", action="store_true",
                    help="amb --friccio, posa b1 = b2 = 0")
    ap.add_argument("--muntatge", action="store_true",
                    help="calcula tambe la D amb les incerteses del muntatge sortejades a "
                         "cada membre (uniforme entre extrems, 06/10), i dona les dues")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()

    TEMPS_S = args.temps
    if args.amplades:
        AMPLADES_GRAUS = tuple(float(a) for a in args.amplades.split(","))
    dis = None
    if args.friccio:
        if args.aire_C is None or args.epsilon is None:
            ap.error("amb --friccio cal dir --aire-C i --epsilon")
        dis = Dissipacio.des_de_friccio(Friccio.des_de_json(ARREL / "parametres.json"),
                                        args.aire_C, args.epsilon)
        if args.sense_viscos:
            from dataclasses import replace
            dis = replace(dis, b1=0.0, b2=0.0)
        print(f"AMB FRICCIO DE PROVA: {dis}")
    elif args.aire_C is not None or args.epsilon is not None or args.sense_viscos:
        ap.error("--aire-C, --epsilon i --sense-viscos nomes van amb --friccio")

    index = json.loads(INDEX.read_text(encoding="utf-8"))["preses"]
    # Les preses marcades fora del balanc (dades/retalls.json) tampoc no entren
    # aqui: la 45_7 te un fotograma trencat que li infla el soroll.
    fora_balanc = [f["nom"] for f in index if f.get("fora_del_balanc")]
    index = [f for f in index if not f.get("fora_del_balanc")]
    metes = {f["nom"]: json.loads((ARREL / "dades" / "net" / f"{f['nom']}.json")
                                  .read_text(encoding="utf-8")) for f in index}
    print(f"{len(index)} preses a l'index" + (f" (fora del balanc: {', '.join(fora_balanc)})"
                                              if fora_balanc else "")
          + ". De les reals, nomes es llegeix el JSON i el repos d'abans d'amollar.")

    soroll = soroll_per_grup(index, metes)

    del_repos = [f for f in index if f["arrenca_del_repos"]]
    grup_de = {f["nom"]: f"{f['grup_graus']:.0f}" for f in del_repos}
    ci = {f["nom"]: condicions_inicials(f["nom"], f, metes[f["nom"]]) for f in del_repos}
    ci_grup = ci_per_grup(ci, grup_de)

    hor_preses = hor_grup = d_preses = d_grup = None
    d_preses_m = d_grup_m = rangs = None
    if not args.sense_feix:
        p = Parametres.des_de_json(ARREL / "parametres.json")
        sortides, d_preses = calcula_horitzons(p, ci, args.membres, args.processos, dis)
        hor_preses = {n: resum_de_presa(s) for n, s in sortides.items()}
        hor_grup = horitzo_per_grup(hor_preses, grup_de)
        d_grup = horitzo_d_per_grup(d_preses, grup_de)
        if args.muntatge:
            rangs = muntatge_de(p)
            print("\nI ARA AMB EL MUNTATGE: " + ", ".join(f"{k} {v[0]:.6g}-{v[1]:.6g}"
                                                           for k, v in rangs.items()))
            _, d_preses_m = calcula_horitzons(p, ci, args.membres, args.processos, dis, rangs)
            d_grup_m = horitzo_d_per_grup(d_preses_m, grup_de)

    imprimeix(soroll, ci, ci_grup, hor_grup)
    if d_grup is not None:
        imprimeix_d(d_grup, args.membres, amb_friccio=dis is not None,
                    titol="SENSE MUNTATGE: nomes la incertesa de les condicions inicials"
                    if args.muntatge else None)
    if d_grup_m is not None:
        imprimeix_d(d_grup_m, args.membres, amb_friccio=dis is not None,
                    titol="AMB MUNTATGE: condicions inicials + muntatge sortejat a cada membre")

    if args.json:
        sortida = {
            "eina": "eines/prepara_versemblant.py",
            "pla": "documentacio/pla-numeros-versemblant.md",
            "preses_a_l_index": len(index),
            "config": {
                "x_graus": X_GRAUS, "amplades_graus": AMPLADES_GRAUS,
                "membres": args.membres, "membres_C": min(MEMBRES_C, args.membres),
                "fora_del_balanc": fora_balanc, "temps_s": TEMPS_S, "fps": FPS_SIM,
                "llavor": LLAVOR, "llavor_per_presa": "LLAVOR + crc32(nom)",
                "tolerancia": {"rtol": RTOL, "atol": ATOL},
                "finestres_per_bloc": FINESTRES_PER_BLOC, "blocs_minims": BLOCS_MINIMS,
                "friccio": False if dis is None else {**dis.com_a_dict(),
                                                      "_estat": "coeficients de PROVA"},
                "muntatge": None if rangs is None else {
                    "rangs": rangs, "mostreig": "uniforme entre extrems, independents",
                    "llavor": f"[llavor de la presa, {LLAVOR_MUNTATGE}]",
                    "pla": "documentacio/pla-feix-amb-muntatge.md",
                    "_declarat": "el banc i els cargols poden comptar el mateix error dues "
                                 "vegades; el biaix de theta1 no hi es"},
            },
            "A_soroll": soroll,
            "B_condicions_inicials": {"per_presa": ci, "per_grup": ci_grup},
        }
        if hor_grup is not None:
            sortida["C_horitzo"] = {
                "definicio": "primer instant en que |theta2 membre - theta2 central| > X, "
                             "theta2 desembolicat; null = no en surt dins de temps_s",
                "per_grup": hor_grup,
                "per_presa": hor_preses,
                "sortides_per_membre": sortides,
            }
            sortida["D_horitzo_amplada_del_feix"] = {
                "definicio": "primer instant en que el feix mateix s'eixampla mes de X: amplada = "
                             "maxim - minim entre tots els membres i el central (minmax_theta2_s, "
                             "minmax_theta1_s); banda95_theta2_s = amplada entre els percentils "
                             "2,5 i 97,5; null = no passa de X dins de temps_s",
                "per_grup": d_grup,
                "per_presa": d_preses,
            }
        if d_grup_m is not None:
            sortida["D_amb_muntatge"] = {
                "definicio": "la mateixa D, amb cada membre amb el seu muntatge sortejat "
                             "(config.muntatge); el central es el nominal",
                "per_grup": d_grup_m,
                "per_presa": d_preses_m,
            }
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        cami.write_text(json.dumps(_json_net(sortida), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
