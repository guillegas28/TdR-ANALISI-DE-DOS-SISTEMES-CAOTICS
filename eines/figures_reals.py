r"""Figures NOMES AMB PRESES REALS: espaguetis, espai de fases i seccio de Poincare.

Autoritzades per en Guillem el 03/10: cap simulacio al costat, en cap figura
ni en cap numero, fins que es firmin X, Y i Z. Aqui no s'importa el simulador
i no s'obre res de dades/sim/ (F1 ho comprova).

  espaguetis  theta1(t) i theta2(t) de tots els tirs d'un grup superposats, de
              -0,2 a 10 s. Figura QUALITATIVA (quatre-experiments.html): el
              numero de l'experiment 4 es <ln(d/d0)>, no aixo.
  fases       (theta plegat, omega) de cada brac, als trams de 0-3 s i 3-10 s;
              una figura per grup. Es una OMBRA de l'espai de quatre dimensions:
              que les corbes es creuin no vol dir res (CLAUDE.md).
  poincare    theta2 = 0 amb omega2 > 0 (caos.creuaments_interpolats), cada tir
              sencer des de l'amollada, per franges d'E/E_v2. L'energia surt de
              les funcions del model (caos.energia_relativa) aplicades als
              estats REALS; no se simula res. Amb friccio l'energia baixa, i per
              aixo es parteix per franges. Des del 25/10, AMB EL FILTRE DE
              CREUAMENTS (calcula_poincare): res als ultims cua_s de la presa, i
              un sol punt per passada de -0,5 a +0,5 graus (criteris FP1-FP5;
              la FP2'' es compleix per construccio, i en queda la llista
              declarada al JSON, filtre.enmig_del_moviment). Amb --corregit,
              els angles de comparar.corregeix_presa (FC1-FC5).
  poincare_energia  (25/10) els mateixos punts, un panell per grup, pintats per
              E/E_v2 amb estil.RAMPA_REAL i la mateixa escala als tres. Criteris
              P1-P6 al pla («25/10 · El Poincare real pintat per energia»).
              FIGURA DE TREBALL: els angles no porten la correccio del biaix.

LES TRIES, escrites al pla abans d'escriure el codi:
  - totes les preses de l'index amb amollada; la 45_7 fora, com al balanc
    (el salt de 341 px en ple moviment);
  - el zero de cada tir es l'amollada_s del VIDEO (el bloc amollada del seu
    JSON), mai l'afinada, que depen del model;
  - un sol color, el de les dades reals (estil.REAL), amb transparencia: un
    tir no es una serie amb identitat propia, i estil.py no te mes colors.

Criteris F1-F3 a documentacio/pla-analisi-caos.md («Cinc feines sense
intervencio»), escrits el 03/10 abans d'escriure aixo.

Us:
    .venv\\Scripts\\python eines\\figures_reals.py espaguetis|fases|poincare|poincare_energia|totes
    .venv\\Scripts\\python eines\\figures_reals.py --autoprova
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

import peus  # noqa: E402
from pendol import caos, dades  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import dif_angular, estats_de_taula  # noqa: E402

NET = ARREL / "dades" / "net"
INDEX = NET / "preses.json"
CENTRES = ARREL / "dades" / "versemblant" / "centres_cercle.json"   # el llegeix comparar.corregeix_presa
CORRECCIO_CLICS = ARREL / "dades" / "versemblant" / "correccio_clics.json"   # tambe (des del pas D, 27/10)
DEFINITIVA = True     # les corregides, definitives des del 29/10 (en Guillem: «si estan a la ultima, definitives»)
GRUPS = ("45", "90", "150")
ESPAGUETIS_S = (-0.2, 10.0)
TRAMS_FASES_S = ((0.0, 3.0), (3.0, 10.0))
FRANGES_E = ((3.5, math.inf, "E/E_v2 ≥ 3,5"), (1.7, 3.5, "1,7 ≤ E/E_v2 < 3,5"),
             (1.0, 1.7, "1 ≤ E/E_v2 < 1,7"), (-math.inf, 1.0, "E/E_v2 < 1"))
# Quatre franges des del 03/10 (abans tres, amb >= 1,7 tota junta): dins d'aquella
# franja l'energia anava d'1,7 a 6,2, un factor 3,6. Partir-la al 3,5 deixa a
# sobre el principi dels 150 (6,2) i el dels 90 (3,4-3,5) al seu llindar, i
# a sota el caos ple de despres de la primera perdua d'energia. Les mateixes
# franges per als tres grups, perque les files es comparin.


def _coma(x: float, fmt: str = ".2f") -> str:
    return format(x, fmt).replace(".", ",")


# ------------------------------------------------------------- les dades

def preses_reals(corregit: bool = False, noms=None) -> list[dict]:
    """Totes les preses que entren: t des de l'amollada, estats (4, N) i el zero.

    ES L'UNICA FUNCIO QUE OBRE DADES. Llegeix l'index i, de cada presa, el CSV
    i el JSON amb dades.carrega.

    Amb corregit=True (25/10), els angles de cada presa son els de
    comparar.corregeix_presa, que obre dades/versemblant/centres_cercle.json i
    parametres.json. Les omega son les del CSV mes la derivada del que mou la
    correccio: la derivada de l'angle corregit, filtrat una sola vegada.

    Amb noms (28/10, per a eines/figures_experiment2.py), nomes s'obren aquestes
    preses; sense, totes, com abans.
    """
    if corregit:
        from pendol import comparar
    sortida = []
    for f in json.loads(INDEX.read_text(encoding="utf-8"))["preses"]:
        if f.get("fora_del_balanc") or (noms is not None and f["nom"] not in noms):
            continue
        taula, meta = dades.carrega(NET / f"{f['nom']}.csv")
        a = meta.get("amollada") or {}
        if not a.get("hi_ha_amollada"):
            continue
        zero = a["amollada_s"]
        t = taula["t"].to_numpy()
        estats, rho = estats_de_taula(taula), None
        if corregit:
            th1, th2, info = comparar.corregeix_presa(taula, meta, f["nom"])
            mou = np.vstack([th1 - estats[0], th2 - estats[1]])
            estats[0], estats[1] = th1, th2
            estats[2:] = estats[2:] + np.gradient(mou, t, axis=1)
            rho = info["rho"]
        sortida.append({"nom": f["nom"], "grup": f"{f['grup_graus']:.0f}", "zero_s": zero,
                        "incertesa_zero_s": float(a["incertesa_s"]),
                        "t": t - zero, "estats": estats,
                        "cua_s": float(meta["geometria"]["cua_s"]), "corregit": corregit, "rho": rho})
    return sortida


def _sufix(preses) -> str:
    return "_corregit" if preses and preses[0]["corregit"] else ""


def _text_correccio(preses) -> str:
    """La frase del peu de les figures corregides, amb un espai al davant (buida si no ho son)."""
    if not _sufix(preses):
        return ""
    grups = " i ".join(f"{g}°" for g in _grups_amb_clics())
    return (f" ANGLES CORREGITS amb comparar.corregeix_presa: l'eix de gir del cercle de cada presa i el repòs final, "
            f"amb la ρ del regle ({_coma(preses[0]['rho'], '.4f')}); als {grups}, amb els clics de la creu del 27/10, i "
            f"als 150°, sense clics. Les velocitats, les de la conversió més la derivada del que mou la correcció. Als "
            f"150° la correcció no està validada."
            + ("" if DEFINITIVA else " PROVISIONAL: falta que en Guillem les doni per definitives."))


def _grups_amb_clics() -> list[str]:
    """Els grups que porten els clics de la creu a la correccio (els de correccio_clics.json)."""
    noms = json.loads(CORRECCIO_CLICS.read_text(encoding="utf-8"))["preses"]
    return sorted({n.split("_")[0] for n in noms}, key=float)


def _plega_graus(th):
    return np.degrees(dif_angular(th, 0.0))


def _amb_talls(x, y):
    """Insereix nan on l'angle plegat salta de +180 a -180, perque la linia no travessi."""
    salt = np.nonzero(np.abs(np.diff(x)) > 180.0)[0] + 1
    return np.insert(x, salt, np.nan), np.insert(y, salt, np.nan)


def calcula_espaguetis(preses) -> dict:
    a, b = ESPAGUETIS_S
    r = {}
    for g in GRUPS:
        tirs = []
        for p in preses:
            if p["grup"] != g:
                continue
            sel = (p["t"] >= a) & (p["t"] <= b)
            tirs.append({"nom": p["nom"], "t": p["t"][sel], "th1": np.degrees(p["estats"][0][sel]),
                         "th2": np.degrees(p["estats"][1][sel])})
        r[g] = tirs
    return r


def calcula_fases(preses) -> dict:
    r = {}
    for g in GRUPS:
        r[g] = {}
        for a, b in TRAMS_FASES_S:
            tirs = []
            for p in preses:
                if p["grup"] != g:
                    continue
                sel = (p["t"] >= a) & (p["t"] <= b)
                tirs.append({"nom": p["nom"], "th1": _plega_graus(p["estats"][0][sel]),
                             "w1": p["estats"][2][sel], "th2": _plega_graus(p["estats"][1][sel]),
                             "w2": p["estats"][3][sel]})
            r[g][f"{a:g}-{b:g}"] = tirs
    return r


# 25/10, el filtre de creuaments (pla, «25/10 · El filtre de creuaments del Poincare real»). La C2 de coherencia
# va trobar creuaments al repos final: caos.creuaments_interpolats no te llindar, i el soroll del seguiment fa
# travessar theta2 = 0 cap amunt quan el pendol ja es quiet. Dues regles:
#   1. res al tram de repos de la conversio (els ultims cua_s de cada presa);
#   2. histeresi: una PASSADA cap amunt va de l'ultim fotograma amb theta2 plegat <= -H al primer amb >= +H, i de
#      cada passada es queda un sol punt (l'ultim creuament que hi cau a dins).
# H = 0,5 graus: a les 29 preses quietes als ultims 2 s, theta2 fa <= 0,47 graus de pic a pic. Preu declarat: es
# perden les oscil.lacions de menys de 0,5 graus, que son els ultims instants de cada tir, i els FREGAMENTS: el brac
# 2 toca la vertical i es gira, a menys de 0,5 graus (sis enmig del moviment a les preses del 25/10, sense corregir),
# i els passos en que theta2 ronda la vertical (un a la 150_2, amb els angles corregits). Si el brac va passar la
# vertical o nomes la va tocar, les dades no ho poden dir. En Guillem va triar aquesta opcio (la A, i despres l'A''),
# la mateixa regla a tot arreu, despres que la FP2 i la FP2' fallessin.
H_PASSADA = math.radians(0.5)


def passades_amunt(a, h: float = H_PASSADA) -> list[tuple[int, int]]:
    """Els fotogrames (i0, i1) de cada passada cap amunt de l'angle plegat `a`: de l'ultim amb a <= -h al primer
    amb a >= +h. Un volteig (de +pi a -pi i amunt per zero) es una passada normal."""
    out, ultim_neg = [], None
    for i, x in enumerate(np.asarray(a, dtype=float)):
        if x <= -h:
            ultim_neg = i
        elif x >= h and ultim_neg is not None:
            out.append((ultim_neg, i))
            ultim_neg = None
    return out


def filtra_creuaments(t, estats, tall_s: float, h: float = H_PASSADA) -> dict:
    """Els creuaments de caos.creuaments_interpolats amb el filtre. Torna el que torna aquella funcio i, per a cada
    creuament, "classe": "" si es queda; "a" si es al tram de repos (t >= tall_s); "b" si es un duplicat dins d'una
    passada que compta; "c" si no es dins de cap passada completa."""
    c = caos.creuaments_interpolats(t, estats)
    passades = [(t[i0], t[i1]) for i0, i1 in passades_amunt(dif_angular(np.asarray(estats)[1], 0.0), h)]
    classe = ["c"] * len(c["t_s"])
    for k, tk in enumerate(c["t_s"]):
        if tk >= tall_s:
            classe[k] = "a"
    for t0, t1 in passades:
        dins = [k for k, tk in enumerate(c["t_s"]) if t0 <= tk <= t1 and classe[k] != "a"]
        for k in dins[:-1]:
            classe[k] = "b"
        if dins:
            classe[dins[-1]] = ""
    return {**c, "classe": classe}


def calcula_poincare(preses, p_mod, trets: dict | None = None) -> dict:
    """Els punts de les dues figures de Poincare, amb el filtre. Si es dona `trets`, s'hi posa, per grup, cada
    creuament tret: (nom, t, classe, E/E_v2, t de l'ultim punt que es queda del seu tir)."""
    Ev2 = caos.energia_volteig_2(p_mod)
    r = {}
    for g in GRUPS:
        punts, fora = [], []
        for p in preses:
            if p["grup"] != g:
                continue
            mou = p["t"] >= 0.0
            c = filtra_creuaments(p["t"][mou], p["estats"][:, mou], float(p["t"][-1]) - p["cua_s"])
            if not len(c["t_s"]):
                continue
            E = np.atleast_1d(caos.energia_relativa(c["estats"].T, p_mod)) / Ev2
            queden = [k for k, x in enumerate(c["classe"]) if not x]
            ultim = float(c["t_s"][queden[-1]]) if queden else -math.inf
            for k in range(len(c["t_s"])):
                if c["classe"][k]:
                    fora.append((p["nom"], float(c["t_s"][k]), c["classe"][k], float(E[k]), ultim))
                    continue
                punts.append((p["nom"], float(c["t_s"][k]), float(np.degrees(dif_angular(c["estats"][k, 0], 0.0))),
                              float(c["estats"][k, 2]), float(E[k])))
        r[g] = punts
        if trets is not None:
            trets[g] = fora
    return r


def resum_filtre(preses, trets: dict) -> dict:
    """El bloc «filtre» del JSON de metriques: les regles, quants creuaments treu cada una, per grup, i la llista
    declarada dels que treu enmig del moviment."""
    per_nom = {p["nom"]: p for p in preses}
    return {"regles": "1) res als ultims cua_s de cada presa (el repos de la conversio); 2) histeresi: una passada "
                      "cap amunt va de l'ultim fotograma amb theta2 <= -h al primer amb >= +h, i en queda un sol punt",
            "h_graus": math.degrees(H_PASSADA), "cua_s": sorted({p["cua_s"] for p in preses}),
            "trets": {g: {"a_repos": sum(q[2] == "a" for q in trets[g]),
                          "b_duplicats": sum(q[2] == "b" for q in trets[g]),
                          "c_sense_passada": sum(q[2] == "c" for q in trets[g]),
                          # els (c) d'abans de l'ultim punt que es queda: fregaments i passos que ronden la vertical (FP2'')
                          "c_enmig_del_moviment": sum(q[2] == "c" and q[1] <= q[4] for q in trets[g])}
                      for g in GRUPS},
            # 25/10, les declaracions per a la memoria (en Guillem): cada creuament tret enmig del moviment.
            "enmig_del_moviment": [
                {"grup": g, "nom": q[0], "t_s": round(q[1], 3), "E_sobre_Ev2": round(q[3], 4),
                 "mena": _mena(*_costats_de_la_franja(per_nom[q[0]], q[1]))}
                for g in GRUPS for q in trets[g] if q[2] == "c" and q[1] <= q[4]]}


def _text_filtre(preses) -> str:
    cues = sorted({p["cua_s"] for p in preses})
    cua = f"dels últims {_coma(cues[0], 'g')} s" if len(cues) == 1 else "de la cua de repòs"
    return (f"Filtre: no hi entren els creuaments {cua} de cada presa (el repòs de la conversió), i un creuament "
            f"només compta si θ2 travessa de −{_coma(math.degrees(H_PASSADA), 'g')}° a "
            f"+{_coma(math.degrees(H_PASSADA), 'g')}° (un sol punt per passada): en queden fora el soroll del "
            f"seguiment al repòs, les oscil·lacions de menys de {_coma(math.degrees(H_PASSADA), 'g')}° i els "
            f"fregaments, en què el braç 2 toca la vertical i es gira a menys de "
            f"{_coma(math.degrees(H_PASSADA), 'g')}°.")


# ------------------------------------------------------------- les figures

def _peu(fig, text):
    from pendol import estil
    peus.peu(fig, 0.01, -0.005, text, fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)


def _comptes(preses):
    return {g: sum(p["grup"] == g for p in preses) for g in GRUPS}


def dibuixa_espaguetis(r, preses, directori):
    import matplotlib.pyplot as plt
    from pendol import estil

    estil.aplica()
    fig, eixos = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
    n = _comptes(preses)
    for fila, g in enumerate(GRUPS):
        for col, (clau, nom) in enumerate((("th1", "θ1"), ("th2", "θ2"))):
            ax = eixos[fila, col]
            for tir in r[g]:
                ax.plot(tir["t"], tir[clau], color=estil.REAL, lw=0.7, alpha=0.45, rasterized=True)
            ax.set_title(f"{g}° · {nom} ({n[g]} tirs)", loc="left")
            ax.set_ylabel(f"{nom} (graus, desembolicat)")
            ax.axvline(0.0, color=estil.EIX, lw=0.8, zorder=0)
    for ax in eixos[-1]:
        ax.set_xlabel("t des de l'amollada del vídeo (s)")
    inc = [1000 * p["incertesa_zero_s"] for p in preses]
    _peu(fig, f"DADES REALS, cap simulació. Cada línia és un tir del Tracker ({len(preses)} preses; la "
              f"45_7 fora, com al balanç). Zero: l'amollada del vídeo de cada presa (incertesa de "
              f"{min(inc):.0f} a {max(inc):.0f} ms). Angles des de la vertical inferior, θ2 absolut. "
              f"Figura qualitativa: el número de l'experiment 4 és ⟨ln(d/d₀)⟩, no aquesta dispersió."
              + _text_correccio(preses))
    fig.tight_layout()
    peus.desa(fig, "espaguetis_reals" + _sufix(preses), directori)


def dibuixa_fases(r, preses, directori):
    import matplotlib.pyplot as plt
    from pendol import estil

    estil.aplica()
    n = _comptes(preses)
    for g in GRUPS:
        fig, eixos = plt.subplots(2, 2, figsize=(10.0, 8.0), sharex=True)
        for col, tram in enumerate(r[g]):
            for fila, (th, w, nom) in enumerate((("th1", "w1", "1"), ("th2", "w2", "2"))):
                ax = eixos[fila, col]
                for tir in r[g][tram]:
                    x, y = _amb_talls(tir[th], tir[w])
                    ax.plot(x, y, color=estil.REAL, lw=0.5, alpha=0.35, rasterized=True)
                ax.set_title(f"Braç {nom} · de {tram.replace('-', ' a ')} s", loc="left")
                ax.set_ylabel(f"ω{nom} (rad/s)")
                ax.set_xlim(-180, 180)
                ax.set_xticks(range(-180, 181, 90))
        for ax in eixos[-1]:
            ax.set_xlabel("angle plegat (graus)")
        fig.suptitle(f"Espai de fases real, {g}° ({n[g]} tirs)", x=0.01, ha="left",
                     fontweight="bold")
        _peu(fig, "DADES REALS, cap simulació. Cada línia és un tir; el zero, l'amollada del vídeo. "
                  "Cada panell és l'OMBRA d'un espai de quatre dimensions (θ1, θ2, ω1, ω2): que les "
                  "corbes es creuin aquí no vol dir que es creuin de debò. Les velocitats surten de la "
                  "derivada de Savitzky-Golay de la conversió i porten el soroll de seguiment mesurat "
                  "el 03/10 (eines/soroll_en_moviment.py)." + _text_correccio(preses))
        fig.tight_layout()
        peus.desa(fig, f"fases_reals_{g}" + _sufix(preses), directori)


def dibuixa_poincare(r, preses, directori):
    import matplotlib.pyplot as plt
    from pendol import estil

    estil.aplica()
    n_col = len(FRANGES_E)
    fig, eixos = plt.subplots(3, n_col, figsize=(3.4 * n_col, 9.5))
    for fila, g in enumerate(GRUPS):
        pts = r[g]
        for col, (lo, hi, etiqueta) in enumerate(FRANGES_E):
            ax = eixos[fila, col]
            sel = [q for q in pts if lo <= q[4] < hi]
            ax.set_title(f"{g}° · {etiqueta}", loc="left", fontsize=10, pad=15)
            if not sel:
                # Un panell sense punts no te eixos: no hi ha res a llegir-hi.
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"cap creuament\nals {g}° a aquesta energia", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=estil.TINTA_SUAU)
                continue
            ax.plot([q[2] for q in sel], [q[3] for q in sel], "o", ms=3.0, color=estil.REAL,
                    alpha=0.6, mec="none")
            E = [q[4] for q in sel]
            # A sota del titol i fora de l'area de dades: a dins tapava punts.
            ax.text(0.0, 1.01, f"{len(sel)} punts · E/E_v2 de {_coma(min(E))} a {_coma(max(E))}",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=estil.TINTA_2)
            ax.set_xlim(-180, 180)
            ax.set_xticks(range(-180, 181, 90))
            ax.set_ylabel("ω1 (rad/s)")
            ax.set_xlabel("θ1 plegat (graus)")
    _peu(fig, "DADES REALS, cap simulació. Secció de Poincaré: el braç 2 passa per la vertical de sota "
              "girant en sentit positiu (θ2 = 0, ω2 > 0), per interpolació lineal entre fotogrames "
              "(caos.creuaments_interpolats). Cada tir sencer des de l'amollada del vídeo. L'energia de "
              "cada punt es calcula amb el model sobre l'estat mesurat; com que el pèndol frega, baixa, "
              "i per això es parteix per franges. Dins de cada franja l'energia encara varia (el rang "
              "de cada panell és sota el seu títol). Les seccions conservatives (figures/poincare_E*) són "
              "a energia fixa; el model amb la fricció firmada es compara amb aquesta a figures/poincare_friccio "
              "(comparació no firmada, des de la firma del 27/10). "
              + _text_filtre(preses) + _text_correccio(preses))
    fig.tight_layout()
    peus.desa(fig, "poincare_real" + _sufix(preses), directori)


def escala_energia(r) -> tuple[float, float]:
    """La mateixa escala de color als tres panells: de la E mes baixa a la mes alta de tots els punts."""
    E = [q[4] for g in GRUPS for q in r[g]]
    return min(E), max(E)


def figura_poincare_energia(r, preses):
    """La figura del Poincare pintat per energia, sense desar. Torna (fig, {grup: (collection, punts
    en l'ordre dibuixat)}, norm): l'autoprova en llegeix els colors (P4)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from pendol import estil

    estil.aplica()
    mapa, norm = estil.mapa_real(), Normalize(*escala_energia(r))
    n = _comptes(preses)
    fig, eixos = plt.subplots(1, len(GRUPS), figsize=(13.0, 4.8), layout="constrained")
    dibuixats = {}
    for ax, g in zip(eixos, GRUPS):
        # Ordenats per energia: els d'energia alta, a sobre.
        pts = sorted(r[g], key=lambda q: q[4])
        sc = ax.scatter([q[2] for q in pts], [q[3] for q in pts], c=[q[4] for q in pts], cmap=mapa, norm=norm,
                        s=7, linewidths=0, rasterized=True)
        dibuixats[g] = (sc, pts)
        E = [q[4] for q in pts]
        ax.set_title(f"{g}° ({n[g]} tirs)", loc="left", pad=15)
        ax.text(0.0, 1.01, f"{len(pts)} punts · E/E_v2 de {_coma(min(E))} a {_coma(max(E))}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=estil.TINTA_2)
        ax.set_xlim(-180, 180)
        ax.set_xticks(range(-180, 181, 90))
        ax.set_xlabel("θ1 plegat (graus)")
        ax.set_ylabel("ω1 (rad/s)")
    fig.colorbar(dibuixats[GRUPS[-1]][0], ax=list(eixos), shrink=0.85, pad=0.02, label="E/E_v2 a cada creuament")
    peus.peu(fig, 0.0, -0.02, "DADES REALS, cap simulació. Secció de Poincaré: el braç 2 passa per la vertical de sota "
                         "girant en sentit positiu (θ2 = 0, ω2 > 0), per interpolació lineal entre fotogrames, cada "
                         "tir sencer des de l'amollada del vídeo. El color és l'energia de cada punt, calculada amb "
                         "el model sobre l'estat mesurat, amb la mateixa escala als tres panells; com que el pèndol "
                         "frega, l'energia baixa al llarg de cada tir. Els punts d'energia alta es dibuixen a sobre. "
                         + _text_filtre(preses)
                         + (_text_correccio(preses) if _sufix(preses) else
                            " FIGURA DE TREBALL: els angles no porten la correcció del biaix de la marca."),
             fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    return fig, dibuixats, norm


def dibuixa_poincare_energia(r, preses, directori):
    from pendol import estil

    fig, _, _ = figura_poincare_energia(r, preses)
    peus.desa(fig, "poincare_real_energia" + _sufix(preses), directori)


# ------------------------------------------------------------- les metriques

def _empremta(obj) -> str:
    """Un hash de totes les dades dibuixades: per a F2 i per al JSON."""
    h = hashlib.sha256()

    def passa(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                passa(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                passa(v)
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(repr(x).encode())
    passa(obj)
    return h.hexdigest()


def metriques(preses, esp, fas, poi, trets) -> dict:
    from pendol import estil

    filtre = resum_filtre(preses, trets)
    base = {
        "que_es": "figures NOMES amb preses reals (autoritzat el 03/10); cap simulacio",
        "eina": "eines/figures_reals.py",
        "preses": [{"nom": p["nom"], "grup": p["grup"], "zero_s": p["zero_s"],
                    "incertesa_zero_s": p["incertesa_zero_s"]} for p in preses],
        "fora": "45_7 (fora_del_balanc)",
        "zero": "amollada_s del video (bloc amollada del JSON de cada presa); mai l'afinada",
    }
    poinc = {}
    for g in GRUPS:
        poinc[g] = {"punts_total": len(poi[g])}
        for lo, hi, etiqueta in FRANGES_E:
            E = [q[4] for q in poi[g] if lo <= q[4] < hi]
            poinc[g][etiqueta] = len(E)
            poinc[g][f"{etiqueta}: rang E/E_v2"] = [min(E), max(E)] if E else None
        poinc[g]["per_presa"] = {}
        for q in poi[g]:
            poinc[g]["per_presa"][q[0]] = poinc[g]["per_presa"].get(q[0], 0) + 1
    out = {
        "espaguetis_reals": {**base, "finestra_s": list(ESPAGUETIS_S), "empremta": _empremta(esp)},
        "fases_reals": {**base, "trams_s": [list(x) for x in TRAMS_FASES_S], "empremta": _empremta(fas)},
        "poincare_real": {**base, "condicio": "theta2 = 0 (plegat) amb omega2 > 0, interpolacio lineal",
                          "filtre": filtre, "franges_E_sobre_Ev2": [x[2] for x in FRANGES_E],
                          "punts": poinc, "empremta": _empremta(poi)},
        # 25/10: els mateixos punts que poincare_real (la mateixa empremta), pintats per energia.
        "poincare_real_energia": {
            **base, "condicio": "theta2 = 0 (plegat) amb omega2 > 0, interpolacio lineal", "filtre": filtre,
            "color": "E/E_v2 de cada punt, estil.RAMPA_REAL, la mateixa escala als tres panells",
            "rampa": list(estil.RAMPA_REAL), "escala_E_sobre_Ev2": list(escala_energia(poi)),
            "angles": "corregits (vegeu correccio)" if _sufix(preses) else "sense corregir: FIGURA DE TREBALL",
            "punts": {g: {"punts_total": len(poi[g]),
                          "rang_E_sobre_Ev2": [min(q[4] for q in poi[g]), max(q[4] for q in poi[g])]}
                      for g in GRUPS},
            "empremta": _empremta(poi)},
    }
    if _sufix(preses):
        # 25/10: les figures corregides porten de quina correccio surten. Sense corregir, el JSON no canvia.
        correccio = {"funcio": "pendol.comparar.corregeix_presa (correccio en dues parts: l'eix de gir del cercle "
                               "de cada presa i el repos final)",
                     "rho": preses[0]["rho"], "rho_de": "el regle (r_gomet / L1 de parametres.json)",
                     "fitxers_sha256": {c.relative_to(ARREL).as_posix(): hashlib.sha256(c.read_bytes()).hexdigest()
                                        for c in (CENTRES, CORRECCIO_CLICS)},
                     "grups_amb_clics": _grups_amb_clics(),
                     "omega": "les del CSV mes np.gradient del que mou la correccio",
                     "validada_als_150": False, "definitiva": DEFINITIVA,
                     "espera": None if DEFINITIVA else "que en Guillem les doni per definitives"}
        for v in out.values():
            v["correccio"] = correccio
    return out


def fes(quines, directori="figures", corregit: bool = False) -> int:
    p_mod = Parametres.des_de_json(ARREL / "parametres.json")
    preses = preses_reals(corregit)
    sx = _sufix(preses)
    trets = {}
    esp, fas, poi = calcula_espaguetis(preses), calcula_fases(preses), calcula_poincare(preses, p_mod, trets)
    m = metriques(preses, esp, fas, poi, trets)
    d = Path(directori)
    d.mkdir(parents=True, exist_ok=True)
    if "espaguetis" in quines:
        dibuixa_espaguetis(esp, preses, directori)
        (d / f"espaguetis_reals{sx}.json").write_text(json.dumps(m["espaguetis_reals"], indent=1, ensure_ascii=False), encoding="utf-8")
    if "fases" in quines:
        dibuixa_fases(fas, preses, directori)
        for g in GRUPS:
            (d / f"fases_reals_{g}{sx}.json").write_text(json.dumps(m["fases_reals"], indent=1, ensure_ascii=False), encoding="utf-8")
    if "poincare" in quines:
        dibuixa_poincare(poi, preses, directori)
        (d / f"poincare_real{sx}.json").write_text(json.dumps(m["poincare_real"], indent=1, ensure_ascii=False), encoding="utf-8")
        for g in GRUPS:
            pg = m["poincare_real"]["punts"][g]
            ft = m["poincare_real"]["filtre"]["trets"][g]
            print(f"  Poincare {g}: {pg['punts_total']} punts; "
                  + "; ".join(f"{e.replace('≥', '>=').replace('≤', '<=')}: {pg[e]}"
                              for _, _, e in FRANGES_E)
                  + f"; trets pel filtre: {ft['a_repos']} al repos, {ft['b_duplicats']} duplicats, "
                    f"{ft['c_sense_passada']} sense passada")
    if "poincare_energia" in quines:
        dibuixa_poincare_energia(poi, preses, directori)
        (d / f"poincare_real_energia{sx}.json").write_text(json.dumps(m["poincare_real_energia"], indent=1, ensure_ascii=False), encoding="utf-8")
        a, b = m["poincare_real_energia"]["escala_E_sobre_Ev2"]
        print(f"  Poincare per energia: escala E/E_v2 de {a:.3f} a {b:.3f}; "
              + "; ".join(f"{g}: {m['poincare_real_energia']['punts'][g]['punts_total']} punts" for g in GRUPS))
    return 0


# --------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """F1-F3, escrits al pla el 03/10/2026 abans d'escriure el codi; P3-P4, FP1 i FP1b, el 25/10; la FP2', el 25/10,
    reescrita despres de fallar per decisio d'en Guillem (l'opcio A); FC2 i FC3 (i la part de la FC4 que hi va), el
    25/10, amb els angles corregits; la FP2'', reescrita per segona vegada despres de fallar (l'A''), es compleix per
    construccio."""
    global _OBERTS
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    print("Autoprova de eines/figures_reals.py (criteris escrits al pla el 03/10/2026)\n")
    p_mod = Parametres.des_de_json(ARREL / "parametres.json")
    sys.addaudithook(_registra)
    _OBERTS = []
    preses = preses_reals()
    trets = {}
    esp, fas, poi = calcula_espaguetis(preses), calcula_fases(preses), calcula_poincare(preses, p_mod, trets)
    m1 = json.dumps(metriques(preses, esp, fas, poi, trets), sort_keys=True)
    oberts, _OBERTS = _OBERTS, None

    print("F1. De dades/ nomes obre net/preses.json i el JSON i el CSV de cada presa; res de dades/sim/")
    arrel_dades = (ARREL / "dades").resolve()
    dins = {str(Path(x).resolve()) for x in oberts if Path(x).resolve().is_relative_to(arrel_dades)}
    esperats = {str(INDEX.resolve())}
    for p in preses:
        esperats |= {str((NET / f"{p['nom']}.csv").resolve()), str((NET / f"{p['nom']}.json").resolve())}
    sim = [x for x in dins if "sim" in Path(x).relative_to(arrel_dades).parts]
    mira(dins == esperats and not sim,
         f"{len(dins)} fitxers oberts a dades/, {len(esperats)} esperats; de dades/sim/: {len(sim)}"
         + ("" if dins == esperats else f"; de mes: {sorted(dins - esperats)[:3]}; falten: {sorted(esperats - dins)[:3]}"))
    print(f"     ({len(preses)} preses: " + " / ".join(f"{g}: {sum(p['grup'] == g for p in preses)}" for g in GRUPS)
          + f"; la 45_7 {'NO ' if any(p['nom'] == '45_7' for p in preses) else ''}hi es fora)")

    print("\nF2. Dues passades: el JSON de metriques i les dades dibuixades, iguals bit a bit")
    preses2 = preses_reals()
    trets2 = {}
    poi2 = calcula_poincare(preses2, p_mod, trets2)
    m2 = json.dumps(metriques(preses2, calcula_espaguetis(preses2), calcula_fases(preses2), poi2, trets2),
                    sort_keys=True)
    mira(m1 == m2, f"empremta {hashlib.sha256(m1.encode()).hexdigest()[:16]} les dues vegades")

    print("\nF3. El zero de cada tir es bit a bit l'amollada_s del seu JSON (la del video)")
    dolents = []
    for p in preses:
        meta = json.loads((NET / f"{p['nom']}.json").read_text(encoding="utf-8"))
        zero = meta["amollada"]["amollada_s"]
        taula, _ = dades.carrega(NET / f"{p['nom']}.csv")
        if not (p["zero_s"] == zero and np.array_equal(p["t"], taula["t"].to_numpy() - zero)):
            dolents.append(p["nom"])
    mira(not dolents, f"{len(preses)} preses, {len(dolents)} amb un altre zero" + (f": {dolents}" if dolents else ""))

    _comprova_p3_p4(poi, preses, mira)

    # FP1, FP1b i FP2 (25/10, el filtre de creuaments): documentacio/pla-analisi-caos.md, escrits abans.
    print("\nFP1. Un tir fabricat (theta2 = 5 graus * e^(-t/2,17 s) * sin(2 pi t / 0,8 s), 10 s, i 3 s de soroll sol), "
          "amb el soroll real del repos d'una presa; nomes la histeresi")
    A0, T, TAU, T_OSC, T_SOL = math.radians(5.0), 0.8, 2.17, 10.0, 3.0
    w = 2.0 * math.pi / T
    for nom in ("45_2", "45_8"):
        s_th, s_w, dt = _soroll_del_repos(next(x for x in preses if x["nom"] == nom))
        t = np.arange(0.0, T_OSC + T_SOL, dt)
        osc = t < T_OSC
        env = np.where(osc, A0 * np.exp(-t / TAU), 0.0)
        th = env * np.sin(w * t)
        om = np.where(osc, A0 * np.exp(-t / TAU) * (w * np.cos(w * t) - np.sin(w * t) / TAU), 0.0)
        c = filtra_creuaments(t, _tir_fabricat(s_th, s_w, th, om), math.inf)
        queden = np.array([c["t_s"][k] for k, x in enumerate(c["classe"]) if not x])
        veritat = T * np.arange(1, int(round(T_OSC / T)) + 1)
        veritat = veritat[veritat < T_OSC]
        grans = veritat[A0 * np.exp(-veritat / TAU) >= math.radians(1.0)]
        un_cada = all(np.sum(np.abs(queden - tv) <= 0.02) == 1 for tv in grans)
        env_q = np.where(queden < T_OSC, A0 * np.exp(-queden / TAU), 0.0)
        # Mes estricte que el criteri escrit, i declarat: a la zona d'envolupant >= 1 grau, cap punt lluny d'un
        # creuament de debo.
        lluny = [q for q, e in zip(queden, env_q) if e >= math.radians(1.0) and np.min(np.abs(veritat - q)) > 0.02]
        petits = queden[env_q <= math.radians(0.25)]
        mira(un_cada and not lluny and not len(petits),
             f"soroll de la {nom}: {len(grans)} creuaments amb envolupant >= 1 grau, cadascun amb "
             f"{'un sol punt' if un_cada else 'NO UN SOL PUNT'} a <= 0,02 s; punts lluny d'un creuament: {len(lluny)}; "
             f"punts amb envolupant <= 0,25 graus o al soroll sol: {len(petits)}; "
             f"({len(queden)} punts, {sum(x == 'c' for x in c['classe'])} creuaments sense passada, "
             f"{sum(x == 'b' for x in c['classe'])} duplicats)")

    print("\nFP1b. Un volteig fabricat (theta2 girant a 1 volta/s, 5,2 s), amb el mateix soroll: un punt per pas")
    s_th, s_w, dt = _soroll_del_repos(next(x for x in preses if x["nom"] == "45_2"))
    t = np.arange(0.0, 5.2, dt)
    wv = 2.0 * math.pi
    for sentit in (1, -1):
        c = filtra_creuaments(t, _tir_fabricat(s_th, s_w, sentit * (-3.0 + wv * t), np.full(len(t), sentit * wv)),
                              math.inf)
        queden = np.array([c["t_s"][k] for k, x in enumerate(c["classe"]) if not x])
        if sentit == 1:
            veritat = (3.0 + 2.0 * math.pi * np.arange(5)) / wv
            mira(len(queden) == len(veritat) and all(np.sum(np.abs(queden - tv) <= 0.02) == 1 for tv in veritat),
                 f"cap amunt: {len(queden)} punts, {len(veritat)} passos per zero")
        else:
            print(f"   (informatiu) en sentit contrari no hi ha cap pas cap amunt: {len(queden)} punts")

    _comprova_fp2(trets, preses, mira)

    # FC2 i FC3 (25/10, les figures corregides): documentacio/pla-analisi-caos.md, escrits abans.
    print("\n--- Amb els angles corregits (comparar.corregeix_presa) ---")
    from pendol import comparar
    _OBERTS = []
    pc = preses_reals(corregit=True)
    trets_c = {}
    poi_c = calcula_poincare(pc, p_mod, trets_c)
    mc1 = json.dumps(metriques(pc, calcula_espaguetis(pc), calcula_fases(pc), poi_c, trets_c), sort_keys=True)
    oberts_c, _OBERTS = _OBERTS, None

    print("\nFC2. theta1 i theta2 son bit a bit els de corregeix_presa, i les omega, les del CSV mes el gradient del "
          "que mou la correccio")
    dolents = []
    for p in pc:
        taula, meta = dades.carrega(NET / f"{p['nom']}.csv")
        th1, th2, _ = comparar.corregeix_presa(taula, meta, p["nom"])
        X = estats_de_taula(taula)
        w = X[2:] + np.gradient(np.vstack([th1 - X[0], th2 - X[1]]), taula["t"].to_numpy(), axis=1)
        if not (np.array_equal(p["estats"][0], th1) and np.array_equal(p["estats"][1], th2)
                and np.array_equal(p["estats"][2:], w)):
            dolents.append(p["nom"])
    mira(not dolents, f"{len(pc)} preses, {len(dolents)} diferents" + (f": {dolents}" if dolents else ""))

    # 28/10, reescrita ABANS de correr (pla, «28/10 · El pas D a les figures reals»): des del pas D, corregeix_presa
    # llegeix tambe correccio_clics.json.
    print("\nFC3. F1: de dades/ nomes obre el mateix que sense corregir, centres_cercle.json i correccio_clics.json; "
          "res de dades/sim/")
    dins_c = {str(Path(x).resolve()) for x in oberts_c if Path(x).resolve().is_relative_to(arrel_dades)}
    esperats_c = esperats | {str(CENTRES.resolve()), str(CORRECCIO_CLICS.resolve())}
    sim_c = [x for x in dins_c if "sim" in Path(x).relative_to(arrel_dades).parts]
    mira(dins_c == esperats_c and not sim_c,
         f"{len(dins_c)} fitxers oberts a dades/, {len(esperats_c)} esperats; de dades/sim/: {len(sim_c)}"
         + ("" if dins_c == esperats_c else f"; de mes: {sorted(dins_c - esperats_c)[:3]}; "
                                            f"falten: {sorted(esperats_c - dins_c)[:3]}"))
    print("FC3. F2: dues passades amb els angles corregits, iguals bit a bit")
    pc2 = preses_reals(corregit=True)
    trets_c2 = {}
    poi_c2 = calcula_poincare(pc2, p_mod, trets_c2)
    mc2 = json.dumps(metriques(pc2, calcula_espaguetis(pc2), calcula_fases(pc2), poi_c2, trets_c2), sort_keys=True)
    mira(mc1 == mc2, f"empremta {hashlib.sha256(mc1.encode()).hexdigest()[:16]} les dues vegades")
    print("FC3. F3: el zero i el temps de cada tir, bit a bit els de sense corregir (l'amollada del video)")
    mateixos = [a["nom"] for a, b in zip(pc, preses)
                if a["nom"] == b["nom"] and a["zero_s"] == b["zero_s"] and np.array_equal(a["t"], b["t"])]
    mira(len(mateixos) == len(preses) == len(pc), f"{len(mateixos)} de {len(preses)} preses")
    print("FC3. P3 i P4 sobre els punts corregits:", end="")
    _comprova_p3_p4(poi_c, pc, mira)
    print("\nFC4 (la part de l'autoprova). La FP2'' amb els angles corregits:", end="")
    _comprova_fp2(trets_c, pc, mira)
    print("     (informatiu) punts amb el filtre, sense corregir / corregits: "
          + "; ".join(f"{g}: {len(poi[g])} / {len(poi_c[g])}" for g in GRUPS))
    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> nomes llegeix preses reals, es reproduible, el zero es l'amollada del video, el Poincare per "
          "energia dibuixa els mateixos punts amb el color de la seva E, i el filtre treu el soroll i els "
          "fregaments de menys de 0,5 graus (declarats) sense tocar cap pas de debo; i amb --corregit, els angles son "
          "els de comparar.corregeix_presa i tot aixo continua valent.")
    return 0


def _comprova_p3_p4(poi, preses, mira):
    """P3 i P4 sobre uns punts i les seves preses (amb els angles corregits o sense)."""
    # P3 i P4 (25/10, el Poincare pintat per energia): documentacio/pla-analisi-caos.md, escrits abans.
    import matplotlib
    matplotlib.use("Agg")
    from pendol import estil
    fig, dib, norm = figura_poincare_energia(poi, preses)
    fig.canvas.draw()
    print("\nP3. Els punts dibuixats son exactament els de calcula_poincare, amb el mateix nombre per grup")
    for g in GRUPS:
        sc, pts = dib[g]
        xy = np.array([[q[2], q[3]] for q in pts])
        mira(len(pts) == len(poi[g]) and sorted(pts) == sorted(poi[g]) and np.array_equal(sc.get_offsets(), xy),
             f"{g}: {len(pts)} punts dibuixats, {len(poi[g])} a calcula_poincare")
    print("\nP4. El color de cada punt es el mapa aplicat a la seva E, amb la mateixa escala als tres panells")
    a, b = escala_energia(poi)
    mapa = estil.mapa_real()
    for g in GRUPS:
        sc, pts = dib[g]
        esperat = mapa(norm(np.array([q[4] for q in pts])))
        dif = float(np.max(np.abs(sc.get_facecolors() - esperat)))
        mateixa = sc.norm is norm and (sc.norm.vmin, sc.norm.vmax) == (a, b)
        mira(dif <= 1e-6 and mateixa, f"{g}: diferencia de color {dif:.1e} (cal <= 1e-6); escala E/E_v2 "
                                      f"{sc.norm.vmin:.3f}-{sc.norm.vmax:.3f}, {'la comuna' if mateixa else 'UNA ALTRA'}")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _comprova_fp2(trets, preses, mira):
    """La FP2'' sobre els creuaments trets d'unes preses (amb els angles corregits o sense)."""
    # La FP2 va fallar el 25/10 (sis fregaments de debo enmig del moviment): en Guillem va triar l'opcio A, la
    # mateixa regla a tot arreu. La FP2' (fregaments: entren i surten de la franja pel mateix costat) va fallar amb
    # els angles corregits (la 150_2, on theta2 ronda la vertical): en Guillem va triar l'A''. FP2'', reescrita per
    # segona vegada despres de fallar i declarada al pla: es compleix PER CONSTRUCCIO, perque es exactament el que
    # treu la regla 2. El que demostra el filtre son la FP1 i la FP1b; d'aqui en queda la llista declarada.
    print("\nFP2''. Amb les preses reals, cada creuament tret enmig del moviment te un pas que no arriba a -0,5 graus "
          "abans o a +0,5 graus despres (PER CONSTRUCCIO); es declaren tots")
    per_nom = {p["nom"]: p for p in preses}
    for g in GRUPS:
        q = trets[g]
        enmig = [x for x in q if x[2] == "c" and x[1] <= x[4]]
        dins = []
        for x in enmig:
            pr = per_nom[x[0]]
            mou = pr["t"] >= 0.0
            tm = pr["t"][mou]
            pas = passades_amunt(dif_angular(pr["estats"][1][mou], 0.0))
            if any(tm[i0] <= x[1] <= tm[i1] for i0, i1 in pas):
                dins.append(x)
        n = {k: sum(x[2] == k for x in q) for k in "abc"}
        Es = [x[3] for x in q]
        mira(not dins, f"{g}: trets {n['a']} al repos, {n['b']} duplicats, {n['c']} sense passada; enmig del "
                       f"moviment {len(enmig)}, dins d'un pas complet {len(dins)}"
                       + (f" {[(x[0], round(x[1], 2)) for x in dins]}" if dins else ""))
        for x in enmig:
            print(f"     {_mena(*_costats_de_la_franja(per_nom[x[0]], x[1]))}: {x[0]} a {x[1]:.2f} s, "
                  f"E/E_v2 {x[3]:.4f}")
        if Es:
            print(f"     (informatiu) E/E_v2 dels trets: de {min(Es):.5f} a {max(Es):.5f}")
        dup = [f"{x[0]} a {x[1]:.2f} s ({x[3]:.5f})" for x in q if x[2] == "b"]
        if dup:
            print(f"     (informatiu) els duplicats: " + "; ".join(dup))


def _mena(abans: int, despres: int) -> str:
    """La mena d'un creuament tret enmig del moviment, de per on entra theta2 a la franja de +-h i per on en surt."""
    return {(-1, -1): "fregament per sota", (1, 1): "fregament per sobre",
            (1, -1): "ronda la vertical (entra per dalt i surt per baix)",
            (-1, 1): "entra per baix i surt per dalt"}.get((abans, despres), "a l'inici o al final de la presa")


def _costats_de_la_franja(p, tc: float, h: float = H_PASSADA) -> tuple[int, int]:
    """Per on entra theta2 a la franja de +-h on hi ha el creuament de l'instant tc, i per on en surt: -1 per sota,
    +1 per sobre, 0 si no en surt abans de l'inici o del final de la presa."""
    a = dif_angular(p["estats"][1], 0.0)
    i = int(np.searchsorted(p["t"], tc))   # el creuament es entre els fotogrames i - 1 (a < 0) i i (a >= 0)
    j0 = max(i - 1, 0)
    while j0 > 0 and abs(a[j0]) < h:
        j0 -= 1
    j1 = min(i, len(a) - 1)
    while j1 < len(a) - 1 and abs(a[j1]) < h:
        j1 += 1
    return (int(np.sign(a[j0])) if abs(a[j0]) >= h else 0), (int(np.sign(a[j1])) if abs(a[j1]) >= h else 0)


def _soroll_del_repos(p):
    """El soroll dels ultims cua_s d'una presa real: theta2 amb la mitjana treta, la omega2 del CSV i el pas."""
    sel = p["t"] > p["t"][-1] - p["cua_s"]
    th2 = p["estats"][1][sel]
    return th2 - th2.mean(), p["estats"][3][sel], float(np.median(np.diff(p["t"])))


def _tir_fabricat(s_th, s_w, theta2, omega2):
    """estats (4, N) d'un tir fabricat: theta1 = omega1 = 0, i el soroll real repetit fins a cobrir el temps."""
    n = len(theta2)
    return np.vstack([np.zeros(n), theta2 + np.resize(s_th, n), np.zeros(n), omega2 + np.resize(s_w, n)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("quines", nargs="?", default="totes",
                    choices=("espaguetis", "fases", "poincare", "poincare_energia", "totes"))
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--corregit", action="store_true",
                    help="angles corregits amb comparar.corregeix_presa; figures amb el sufix _corregit")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a <figures>/memoria/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.memoria:
        peus.memoria()
        args.figures = str(Path(args.figures) / "memoria")
    quines = (("espaguetis", "fases", "poincare", "poincare_energia") if args.quines == "totes"
              else (args.quines,))
    sortida = fes(quines, args.figures, args.corregit)
    peus.escriu(args.figures)
    return sortida


if __name__ == "__main__":
    sys.exit(main())
