r"""El soroll de seguiment EN MOVIMENT, amb la norma sencera (pas C de l'experiment 4).

Per que existeix. eines/terra_de_soroll.py va mesurar el soroll al REPOS
d'abans d'amollar (0,16-0,31 graus per parella) i va trobar que no serveix:
les marques quietes es segueixen gairebe perfectament, i el soroll de l'index
(soroll_angular_graus, 0,06-1 grau per fotograma) surt sobretot del moviment.
I la diferencia es enorme si aquest soroll es blanc: la derivada de
Savitzky-Golay que dona les velocitats el multiplica per unes 25 vegades en la
norma, i el terra passaria a ser tan gran com la d0. Aqui es mesura de debo.

COM. Les barres son rigides: en pixels, L1 = |articulacio - pivot| i
L2 = |gomet - articulacio| haurien de ser constants, i tot el que varien es
error de seguiment. e = L/L_mediana - 1 fa de soroll d'angle (un error d de
pixels al llarg de la barra es d/L de llargada relativa; de traves, d/L
d'angle). Aquesta e es passa pel MATEIX Savitzky-Golay (13, 3) i per la
MATEIXA derivada que la conversio, o sigui que no cal suposar si el soroll es
blanc o lent: el filtre fa amb la e el que fa amb els angles. En surten les
quatre components de la norma de mesures.distancia_fases i sigma_d.

L'APARELLAMENT AMB EL CSV. Les posicions surten de tracker.llegeix_del_trk, la
mateixa funcio que fa servir la conversio, amb el retall de dades/retalls.json
i la cua del JSON de la presa: aixi la marca de l'articulacio ja hi va
corregida exactament igual (J = A - delta*u1, nomes al llarg de la barra 1). La
graella del CSV es (k - k0)*dt; a les cinc preses amb forats la conversio
interpola els angles linealment, i aqui s'interpolen les llargades igual.
R0 comprova, presa a presa, que amb aquestes posicions el cami de la conversio
reprodueix el CSV a l'arrodoniment.

LA FINESTRA: de 0,1 a 10 s despres de l'amollada del JSON. Es llegeix tot el
.trk perque el filtre ha de veure la serie sencera, com a la conversio, pero
nomes es mesura dins de la finestra.

TRES LIMITS, i van a la memoria:
  - la llargada nomes veu l'error AL LLARG de la barra, i l'estela va de
    traves: l'articulacio es mou sempre perpendicular a la barra 1, o sigui
    que L1 es cega a la seva estela. D4 compara la part rapida de traves i la
    de llarg, que no ho resol pero ho fita;
  - una part de la variacio de L pot ser GEOMETRIA (el pivot fora del centre,
    perspectiva, la marca de l'articulacio de costat, que no es corregeix):
    depen de la posicio del pendol i no es soroll de seguiment. D1 la separa i
    es publiquen les dues xifres; quina va a la regla 2 ho decideix en Guillem;
  - el biaix de theta1, constant dins de cada presa, no hi surt.

Criteris de l'autoprova, fixats el 03/10/2026 abans d'escriure el codi
(documentacio/pla-analisi-caos.md, «03/10 · El soroll en moviment»):
  R0  amb les posicions d'aqui, el cami de la conversio reprodueix les quatre
      columnes del CSV a l'arrodoniment (10 xifres), i la mateixa graella;
  R1  soroll blanc isotrop conegut (1,5 px) amb 20 fotogrames perduts: cada
      component i sigma_d a < 15 % del soroll de debo;
  R2  el mateix amb soroll lent (AR(1), rho = 0,9): a < 15 %;
  R3  reproduible bit a bit, i de dades/ nomes obre el .trk, el JSON i el CSV
      de la presa;
  R4  la regressio geometrica (D1) explica > 90 % d'un pivot desplacat 3 px i
      una perspectiva del 2 % sense soroll, i < 5 % del soroll blanc sol.

Us:
    .venv\\Scripts\\python eines\\soroll_en_moviment.py
    .venv\\Scripts\\python eines\\soroll_en_moviment.py --json dades/divergencia/soroll_moviment_2026-10-03.json
    .venv\\Scripts\\python eines\\soroll_en_moviment.py --autoprova
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol import dades, tracker  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import distancia_fases, omega_zero  # noqa: E402
from terra_de_soroll import soroll_esperat  # noqa: E402  el mateix diagnostic de blancor

NET = ARREL / "dades" / "net"
INDEX = NET / "preses.json"

DES_DE_S = 0.1     # despres de l'amollada: fora l'amollada mateixa i la vora del filtre
FINS_A_S = 10.0    # despres de l'amollada
TRAMS_S = ((0.1, 1.0), (1.0, 3.0), (3.0, 10.0))  # D3: la lambda es llegiria al primer
TOLERANCIA_R0 = 1e-9  # relativa (sobre max(|x|, 1)): el CSV desa %.10g, 5e-10 de mitja xifra
COMPONENTS = ("theta1", "theta2", "omega1/omega0", "omega2/omega0")


# ------------------------------------------------------------- la cadena

def a_la_graella(t, *senyals):
    """Els senyals d'una presa sobre la graella del CSV, com ho fa tracker.processa.

    COPIA de les linies de processa() que fan la graella (tracker.py no es
    toca). Si els fotogrames tenen forats, la conversio remostreja amb
    interpolacio lineal a 1/mediana(dt); si no, la graella son els fotogrames
    mateixos i els fps surten de la durada. R0 comprova que dona el mateix.
    Torna (t, fps, senyals, remostrejat).
    """
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    dt_med = float(np.median(np.diff(t)))
    if np.ptp(np.diff(t)) > 0.05 * dt_med:
        fps = 1.0 / dt_med
        t_unif = np.arange(t[0], t[-1] + 0.5 / fps, 1.0 / fps)
        return t_unif, fps, [np.interp(t_unif, t, s) for s in senyals], True
    fps = (len(t) - 1) / (t[-1] - t[0])
    return t, fps, [np.asarray(s, dtype=float) for s in senyals], False


def cadena(t_fot, J, G, inverteix: bool = False) -> dict:
    """De les posicions (relatives al pivot, y avall) als angles i les llargades crus.

    Tot sobre la graella del CSV i sense filtrar. J i G son (N, 2) en pixels
    sobre els fotogrames t_fot. e = L/mediana(L) - 1 es el soroll d'angle
    substitut de cada barra.
    """
    J = np.asarray(J, dtype=float)
    G = np.asarray(G, dtype=float)
    th1, th2 = tracker.a_angles(t_fot, J[:, 0], J[:, 1], G[:, 0], G[:, 1], (0.0, 0.0), True, inverteix)
    L1 = np.hypot(J[:, 0], J[:, 1])
    L2 = np.hypot(G[:, 0] - J[:, 0], G[:, 1] - J[:, 1])
    t, fps, (th1, th2, L1, L2), remostrejat = a_la_graella(t_fot, th1, th2, L1, L2)

    # Quins punts de la graella no tenen fotograma al darrere (els forats).
    t0 = np.asarray(t_fot, dtype=float) - float(t_fot[0])
    i = np.clip(np.searchsorted(t0, t), 1, len(t0) - 1)
    dist = np.minimum(np.abs(t - t0[i - 1]), np.abs(t0[i] - t))
    return {
        "t": t, "fps": fps, "remostrejat": remostrejat,
        "th": np.vstack([th1, th2]),
        "e": np.vstack([L1 / np.median(L1) - 1.0, L2 / np.median(L2) - 1.0]),
        "interpolat": dist > 0.25 / fps,
    }


def filtra(x, finestra: int, ordre: int, fps: float):
    """El suavitzat i la derivada de la conversio, fila a fila (com a processa)."""
    s = np.vstack([savgol_filter(r, finestra, ordre) for r in x])
    d = np.vstack([savgol_filter(r, finestra, ordre, deriv=1, delta=1.0 / fps) for r in x])
    return s, d


def components(s, d, sel, w0: float):
    """Les quatre components de la norma (radians d'angle equivalent) i sigma_d."""
    c = np.array([s[0][sel].std(ddof=1), s[1][sel].std(ddof=1),
                  d[0][sel].std(ddof=1) / w0, d[1][sel].std(ddof=1) / w0])
    return c, float(np.sqrt(np.sum(c * c)))


def regressors(th):
    """D1: el que la geometria pot fer a cada llargada, en funcio de la posicio.

    L1: un pivot fora del centre la fa variar com cos/sin(theta1); una
    perspectiva o una lent, com cos/sin(2 theta1).
    L2: la marca de l'articulacio fora de l'eix (el que queda de delta i la
    de costat, que no es corregeix) com cos/sin(theta2 - theta1); la
    perspectiva, com cos/sin(2 theta2).
    """
    th1, th2 = th
    r = th2 - th1
    u = np.ones_like(th1)
    X1 = np.column_stack([u, np.cos(th1), np.sin(th1), np.cos(2 * th1), np.sin(2 * th1)])
    X2 = np.column_stack([u, np.cos(r), np.sin(r), np.cos(2 * th2), np.sin(2 * th2)])
    return X1, X2


def treu_geometria(c: dict, sel):
    """La e sense la part geometrica (ajustada a la finestra), i quina fraccio en treu."""
    resid, frac = [], []
    for X, e in zip(regressors(c["th"]), c["e"]):
        coef = np.linalg.lstsq(X[sel], e[sel], rcond=None)[0]
        geo = X @ coef - coef[0]
        r = e - geo
        frac.append(float(1.0 - r[sel].var() / e[sel].var()))
        resid.append(r)
    return np.vstack(resid), frac


def _graus(v):
    return [float(math.degrees(x)) for x in v]


def mesura(c: dict, amollada_s: float, w0: float, finestra: int, ordre: int,
           des_de: float = DES_DE_S, fins_a: float = FINS_A_S, trams=TRAMS_S) -> dict:
    """El soroll d'una serie (real o fabricada): total, sense geometria i els diagnostics."""
    t, fps = c["t"], c["fps"]
    sel = (t >= amollada_s + des_de) & (t <= amollada_s + fins_a)

    s, d = filtra(c["e"], finestra, ordre, fps)
    comp, sd = components(s, d, sel, w0)

    r, frac = treu_geometria(c, sel)
    sr, dr = filtra(r, finestra, ordre, fps)
    comp_g, sd_g = components(sr, dr, sel, w0)

    # D2: blanc o lent. Sobre la e crua sense geometria, dins de la finestra.
    cru = r[:, sel] - r[:, sel].mean(axis=1, keepdims=True)
    lag1 = [float(np.corrcoef(x[:-1], x[1:])[0, 1]) for x in cru]
    sig_cru = cru.std(axis=1, ddof=1)
    g0, g1 = soroll_esperat(1.0, finestra, ordre, fps)
    esperat = np.array([sig_cru[0] * g0, sig_cru[1] * g0, sig_cru[0] * g1 / w0, sig_cru[1] * g1 / w0])

    # D3: per trams, amb la velocitat tipica de cada tram (de la mateixa derivada).
    ths, thd = filtra(c["th"], finestra, ordre, fps)
    per_tram = []
    for a, b in trams:
        st = (t >= amollada_s + a) & (t <= amollada_s + b)
        _, sd_t = components(s, d, st, w0)
        _, sd_tg = components(sr, dr, st, w0)
        per_tram.append({"de_s": a, "fins_s": b, "sigma_d_graus": math.degrees(sd_t),
                         "sigma_d_sense_geometria_graus": math.degrees(sd_tg),
                         "omega_mediana_rad_s": [float(np.median(np.abs(thd[0][st]))),
                                                 float(np.median(np.abs(thd[1][st])))]})

    # D4: la part rapida de traves (angle cru - filtrat) i la de llarg (e crua -
    # filtrada). La de traves porta a mes el que el cubic no segueix del moviment.
    traves = (c["th"] - ths)[:, sel].std(axis=1, ddof=1)
    llarg = (c["e"] - s)[:, sel].std(axis=1, ddof=1)

    return {
        "mostres": int(sel.sum()),
        "forats_a_la_finestra": int(c["interpolat"][sel].sum()),
        "total": {"components_graus": _graus(comp), "sigma_d_graus": math.degrees(sd)},
        "sense_geometria": {"components_graus": _graus(comp_g), "sigma_d_graus": math.degrees(sd_g)},
        "fraccio_geometrica": frac,
        "lag1": lag1,
        "soroll_cru_per_fotograma_graus": _graus(sig_cru),
        "esperat_si_blanc_graus": _graus(esperat),
        "per_tram": per_tram,
        "rapid_de_traves_graus": _graus(traves),
        "rapid_de_llarg_graus": _graus(llarg),
        "_components_rad": comp, "_components_geo_rad": comp_g,
    }


# ------------------------------------------------------------- les preses reals

def posicions_del_trk(meta: dict):
    """(t, J, G, marca_corregida) d'una presa, tal com les fa servir la conversio.

    ES L'UNICA FUNCIO QUE OBRE UN .trk. Amb el retall i la cua del JSON, i la
    correccio de la marca de l'articulacio que fa llegeix_del_trk per defecte.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        t, xj, yj, xg, yg, m = tracker.llegeix_del_trk(
            ARREL / meta["fitxer_trk"], meta["geometria"]["cua_s"],
            meta.get("articulacio_corregida", False), True,
            meta.get("retall_des_del_fotograma"))
    corregida = bool(m["geometria"]["marca_articulacio"].get("corregida"))
    return t, np.column_stack([xj, yj]), np.column_stack([xg, yg]), corregida


def comprova_r0(c: dict, meta: dict, taula, marca_corregida: bool) -> dict:
    """R0: el cami de la conversio, amb aquestes posicions, dona el CSV."""
    fin, ordre = int(meta["savgol"]["finestra_mostres"]), int(meta["savgol"]["ordre"])
    fps = c["fps"]
    # La regla de la finestra de processa(), per si un dia algu la canvia.
    f = int(round(0.05 * fps))
    f = max(ordre + 2, f + (f % 2 == 0))
    dt = 1.0 / fps
    calc = {
        "t": c["t"],
        "theta1": savgol_filter(c["th"][0], fin, ordre),
        "theta2": savgol_filter(c["th"][1], fin, ordre),
        "omega1": savgol_filter(c["th"][0], fin, ordre, deriv=1, delta=dt),
        "omega2": savgol_filter(c["th"][1], fin, ordre, deriv=1, delta=dt),
    }
    n_ok = len(taula) == len(c["t"])
    dev = {}
    if n_ok:
        for k, v in calc.items():
            ref = taula[k].to_numpy()
            dev[k] = float(np.max(np.abs(v - ref) / np.maximum(np.abs(ref), 1.0)))
    marca_ok = marca_corregida == bool(meta["geometria"]["marca_articulacio"].get("corregida"))
    ok = (n_ok and f == fin and abs(fps / float(meta["fps"]) - 1) < 1e-12 and marca_ok
          and all(v <= TOLERANCIA_R0 for v in dev.values()))
    return {"ok": bool(ok), "mostres_iguals": bool(n_ok), "finestra_igual": bool(f == fin),
            "fps_relatiu": float(fps / float(meta["fps"]) - 1), "marca_igual": bool(marca_ok),
            "desviacio_relativa_maxima": dev}


def mesura_presa(fila: dict, w0: float) -> dict | None:
    """Tot el d'una presa real. Obre el seu JSON, el seu .trk i el seu CSV (per a R0)."""
    nom = fila["nom"]
    meta = json.loads((NET / f"{nom}.json").read_text(encoding="utf-8"))
    a = meta.get("amollada") or {}
    if not a.get("hi_ha_amollada"):
        return None
    t_fot, J, G, corregida = posicions_del_trk(meta)
    c = cadena(t_fot, J, G, bool(meta.get("signe_invertit", False)))
    taula, _ = dades.carrega(NET / f"{nom}.csv")
    r0 = comprova_r0(c, meta, taula, corregida)
    sg = meta["savgol"]
    m = mesura(c, float(a["amollada_s"]), w0, int(sg["finestra_mostres"]), int(sg["ordre"]))
    for k in ("_components_rad", "_components_geo_rad"):
        m.pop(k)
    return {"grup": f"{fila['grup_graus']:.0f}", "angles_amollada_graus": list(a["angles_amollada_graus"]),
            "remostrejada": bool(c["remostrejat"]), "marca_corregida": corregida, "r0": r0, **m}


def _estad(v) -> dict | None:
    v = np.asarray(v, dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mediana": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


VARIANTS = (("total", lambda p: p["total"]["sigma_d_graus"]),
            ("sense_geometria", lambda p: p["sense_geometria"]["sigma_d_graus"]),
            ("tram_0.1-1s", lambda p: p["per_tram"][0]["sigma_d_graus"]))


def per_grup(preses: dict, w0: float) -> dict:
    """El terra de cada parella, sqrt(sA^2 + sB^2), i d0/terra, amb les tres variants."""
    grups = {}
    for g in sorted({v["grup"] for v in preses.values()}, key=float):
        noms = [n for n, v in preses.items() if v["grup"] == g and v["r0"]["ok"]]
        bloc = {"preses": noms,
                "fora_per_r0": [n for n, v in preses.items() if v["grup"] == g and not v["r0"]["ok"]]}
        for clau, f in VARIANTS:
            bloc[f"sigma_d_presa_{clau}_graus"] = _estad([f(preses[n]) for n in noms])
        comp = np.array([preses[n]["total"]["components_graus"] for n in noms])
        compg = np.array([preses[n]["sense_geometria"]["components_graus"] for n in noms])
        bloc["components_mediana_total_graus"] = dict(zip(COMPONENTS, np.median(comp, axis=0).tolist()))
        bloc["components_mediana_sense_geometria_graus"] = dict(zip(COMPONENTS, np.median(compg, axis=0).tolist()))
        bloc["fraccio_geometrica_mediana"] = np.median([preses[n]["fraccio_geometrica"] for n in noms], axis=0).tolist()
        bloc["lag1_mediana"] = np.median([preses[n]["lag1"] for n in noms], axis=0).tolist()
        d0s, terres = [], {k: [] for k, _ in VARIANTS}
        for i, a in enumerate(noms):
            for b in noms[i + 1:]:
                ea = np.radians([*preses[a]["angles_amollada_graus"], 0.0, 0.0])
                eb = np.radians([*preses[b]["angles_amollada_graus"], 0.0, 0.0])
                d0s.append(math.degrees(float(distancia_fases(ea, eb, w0))))
                for k, f in VARIANTS:
                    terres[k].append(math.hypot(f(preses[a]), f(preses[b])))
        d0 = np.array(d0s)
        bloc["parelles"] = len(d0s)
        bloc["d0_parella_graus"] = _estad(d0)
        for k, _ in VARIANTS:
            tr = np.array(terres[k])
            rao = d0 / tr if len(tr) else tr
            bloc[f"terra_parella_{k}_graus"] = _estad(tr)
            bloc[f"rao_d0_terra_{k}"] = _estad(rao)
            bloc[f"fraccio_d0_per_sobre_del_terra_{k}"] = float(np.mean(rao > 1)) if len(rao) else None
            bloc[f"fraccio_d0_per_sobre_de_3_terres_{k}"] = float(np.mean(rao > 3)) if len(rao) else None
        grups[g] = bloc
    return grups


def mesura_tot(index: list, p) -> dict:
    w0 = omega_zero(p)
    preses = {}
    for fila in index:
        if fila.get("fora_del_balanc"):
            continue
        print(f"  {fila['nom']} ...", end="", flush=True)
        r = mesura_presa(fila, w0)
        if r is None:
            print(" sense amollada, fora")
            continue
        preses[fila["nom"]] = r
        print(f" sigma_d {r['total']['sigma_d_graus']:.3f} graus"
              f"{'' if r['r0']['ok'] else '   R0 FALLA: ' + json.dumps(r['r0'])}")
    return {"omega0_rad_s": w0, "per_presa": preses, "per_grup": per_grup(preses, w0)}


def imprimeix(r: dict) -> None:
    print(f"\nomega0 = {r['omega0_rad_s']:.3f} rad/s. Tot en graus d'angle equivalent; finestra "
          f"{DES_DE_S}-{FINS_A_S} s despres de l'amollada.\n")
    print("Per presa: sigma_d total / sense geometria; components totals (th1 th2 w1/w0 w2/w0); "
          "fraccio geometrica (L1 L2); lag1; forats")
    for n, v in r["per_presa"].items():
        c = v["total"]["components_graus"]
        print(f"  {n:>6}  {v['total']['sigma_d_graus']:6.3f} / {v['sense_geometria']['sigma_d_graus']:6.3f}   "
              f"{c[0]:.3f} {c[1]:.3f} {c[2]:6.3f} {c[3]:6.3f}   "
              f"geo {v['fraccio_geometrica'][0]:.2f} {v['fraccio_geometrica'][1]:.2f}   "
              f"lag1 {v['lag1'][0]:.2f} {v['lag1'][1]:.2f}   forats {v['forats_a_la_finestra']}"
              f"{'' if v['r0']['ok'] else '   R0 FALLA'}")
    print()
    for g, v in r["per_grup"].items():
        print(f"Grup {g}: {len(v['preses'])} preses, {v['parelles']} parelles"
              + (f"  (fora per R0: {', '.join(v['fora_per_r0'])})" if v["fora_per_r0"] else ""))
        print("   components, mediana total:           "
              + "  ".join(f"{k} {x:.3f}" for k, x in v["components_mediana_total_graus"].items()))
        print("   components, mediana sense geometria: "
              + "  ".join(f"{k} {x:.3f}" for k, x in v["components_mediana_sense_geometria_graus"].items()))
        print(f"   fraccio geometrica (L1, L2): {v['fraccio_geometrica_mediana'][0]:.2f}, "
              f"{v['fraccio_geometrica_mediana'][1]:.2f}    lag1: {v['lag1_mediana'][0]:.2f}, "
              f"{v['lag1_mediana'][1]:.2f}")
        d0 = v["d0_parella_graus"]
        print(f"   d0 parella: {d0['mediana']:.3f} ({d0['min']:.3f}-{d0['max']:.3f})")
        for k, _ in VARIANTS:
            tp, ra = v[f"terra_parella_{k}_graus"], v[f"rao_d0_terra_{k}"]
            print(f"   terra {k:<16} {tp['mediana']:6.3f} ({tp['min']:.3f}-{tp['max']:.3f})   d0/terra "
                  f"{ra['mediana']:5.2f} ({ra['min']:.2f}-{ra['max']:.2f})   >1: "
                  f"{100 * v[f'fraccio_d0_per_sobre_del_terra_{k}']:3.0f} %   >3: "
                  f"{100 * v[f'fraccio_d0_per_sobre_de_3_terres_{k}']:3.0f} %")
        print()


# --------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    """Hook d'auditoria de Python: apunta cada fitxer que s'obre mentre _OBERTS no es None."""
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def _trajectoria(p, temps: float, fps: float):
    """Un pendol fabricat (nomes simulacio, sense friccio), amollat de 150 / 150 graus."""
    from pendol.simulador import simula
    t, y = simula(p, np.radians([150.0, 150.0, 0.0, 0.0]), temps=temps, fps=fps)
    return t, y[0], y[1]


def _posicions(th1, th2, L1=195.0, L2=227.0):
    """Les marques en pixels, relatives al pivot i amb la y avall, com les dona llegeix_del_trk."""
    J = np.column_stack([L1 * np.sin(th1), L1 * np.cos(th1)])
    G = J + np.column_stack([L2 * np.sin(th2), L2 * np.cos(th2)])
    return J, G


def _soroll_de_debo(c_net, c_brut, sel, w0, fin, ordre):
    """El soroll de veritat: angles amb soroll menys angles nets, despres del mateix filtre."""
    s0, d0 = filtra(c_net["th"], fin, ordre, c_net["fps"])
    s1, d1 = filtra(c_brut["th"], fin, ordre, c_brut["fps"])
    return components(s1 - s0, d1 - d0, sel, w0)


def autoprova() -> int:
    """R0-R4, fixats el 03/10/2026 abans d'escriure el codi."""
    global _OBERTS
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    fps, fin, ordre = 240.0, 13, 3
    print("Autoprova de eines/soroll_en_moviment.py (criteris fixats el 03/10/2026)\n")

    # La trajectoria fabricada. 50 s en comptes dels 10 de les preses: el que es
    # prova es l'estimador, que no depen de la llargada, i amb mes mostres
    # l'error estadistic de cada desviacio (soroll filtrat, correlacionat) baixa
    # al ~3 %, ben per sota del 15 % del criteri.
    t, th1, th2 = _trajectoria(p, 50.5, fps)
    rng = np.random.default_rng(20261003)
    n = len(t)
    perduts = np.concatenate([np.arange(3000, 3005), rng.choice(np.arange(24, 12000), 15, replace=False)])
    queden = np.setdiff1d(np.arange(n), perduts)
    t_f = t[queden]
    J0, G0 = _posicions(th1, th2)
    c_net = cadena(t_f, J0[queden], G0[queden])
    sel = (c_net["t"] >= DES_DE_S) & (c_net["t"] <= 50.0)
    print(f"Pendol fabricat: 150/150 graus, {n} fotogrames, {len(perduts)} perduts; finestra 0,1-50 s\n")

    def compara(nom, J, G):
        c = cadena(t_f, J[queden], G[queden])
        veritat, sd_v = _soroll_de_debo(c_net, c, sel, w0, fin, ordre)
        m = mesura(c, 0.0, w0, fin, ordre, fins_a=50.0, trams=((0.1, 50.0),))
        est = m["_components_rad"]
        rel = np.append(est / veritat, math.radians(m["total"]["sigma_d_graus"]) / sd_v) - 1
        mira(bool(np.all(np.abs(rel) < 0.15)),
             f"{nom}: estimat / de debo - 1 = " + ", ".join(f"{100 * x:+.1f} %" for x in rel)
             + f"  (sigma_d de debo {math.degrees(sd_v):.3f} graus)")
        return c, m

    # R1. Soroll blanc isotrop d'1,5 px a les dues marques.
    print("R1. Soroll blanc d'1,5 px, amb forats: cada component i sigma_d a < 15 %")
    nJ, nG = rng.normal(0.0, 1.5, (n, 2)), rng.normal(0.0, 1.5, (n, 2))
    c_blanc, m_blanc = compara("blanc", J0 + nJ, G0 + nG)
    print(f"        (forats a la finestra: {m_blanc['forats_a_la_finestra']}; lag1 "
          f"{m_blanc['lag1'][0]:.2f} {m_blanc['lag1'][1]:.2f})")

    # R2. Soroll lent: AR(1) amb rho = 0,9 (uns 10 fotogrames), la mateixa sigma.
    print("\nR2. Soroll lent (AR(1), rho = 0,9): cada component i sigma_d a < 15 %")
    rho = 0.9

    def ar1(forma):
        x = np.empty(forma)
        z = rng.normal(0.0, 1.5, forma)
        x[0] = z[0]
        for k in range(1, forma[0]):
            x[k] = rho * x[k - 1] + math.sqrt(1 - rho * rho) * z[k]
        return x
    _, m_lent = compara("lent", J0 + ar1((n, 2)), G0 + ar1((n, 2)))
    print(f"        (lag1 {m_lent['lag1'][0]:.2f} {m_lent['lag1'][1]:.2f})")

    # R4. La geometria: pivot 3 px fora i perspectiva del 2 % (y x 0,98), sense soroll.
    print("\nR4. La regressio geometrica: > 90 % de la geometria sola, < 5 % del soroll blanc sol")
    p0 = np.array([3.0, -2.0])
    persp = np.array([1.0, 0.98])
    Jg, Gg = J0 * persp - p0, G0 * persp - p0
    c_geo = cadena(t_f, Jg[queden], Gg[queden])
    m_geo = mesura(c_geo, 0.0, w0, fin, ordre, fins_a=50.0, trams=((0.1, 50.0),))
    fg, fb = m_geo["fraccio_geometrica"], m_blanc["fraccio_geometrica"]
    mira(min(fg) > 0.90, f"geometria sola: explica {100 * fg[0]:.2f} % (L1) i {100 * fg[1]:.2f} % (L2)")
    mira(max(fb) < 0.05, f"soroll blanc sol: explica {100 * fb[0]:.3f} % (L1) i {100 * fb[1]:.3f} % (L2)")
    # Mesurat i publicat, sense criteri: geometria + soroll blanc, la xifra sense geometria.
    c_gb = cadena(t_f, (Jg + nJ)[queden], (Gg + nG)[queden])
    _, sd_v = _soroll_de_debo(c_geo, c_gb, sel, w0, fin, ordre)
    m_gb = mesura(c_gb, 0.0, w0, fin, ordre, fins_a=50.0, trams=((0.1, 50.0),))
    print(f"        (sense criteri) geometria + blanc: sigma_d total {m_gb['total']['sigma_d_graus']:.3f}, "
          f"sense geometria {m_gb['sense_geometria']['sigma_d_graus']:.3f}, de debo {math.degrees(sd_v):.3f} graus")
    # D4, el sostre del cubic: la part rapida de traves d'una trajectoria SENSE soroll.
    m_net = mesura(c_net, 0.0, w0, fin, ordre, fins_a=50.0, trams=((0.1, 50.0),))
    print(f"        (D4, sense criteri) el cubic sobre aquesta trajectoria neta deixa de traves "
          f"{m_net['rapid_de_traves_graus'][0]:.4f} / {m_net['rapid_de_traves_graus'][1]:.4f} graus; "
          f"amb el blanc, traves {m_blanc['rapid_de_traves_graus'][0]:.3f} / "
          f"{m_blanc['rapid_de_traves_graus'][1]:.3f} i llarg {m_blanc['rapid_de_llarg_graus'][0]:.3f} / "
          f"{m_blanc['rapid_de_llarg_graus'][1]:.3f}")

    # R0. Tres preses reals que toquen els tres camins: marca corregida (150_3),
    # retall (45_3) i 49 fotogrames perduts (90_11).
    print("\nR0. El cami de la conversio, amb les posicions d'aqui, reprodueix el CSV")
    index = {f["nom"]: f for f in json.loads(INDEX.read_text(encoding="utf-8"))["preses"]}
    for nom in ("150_3", "45_3", "90_11"):
        r = mesura_presa(index[nom], w0)
        dv = r["r0"]["desviacio_relativa_maxima"]
        mira(r["r0"]["ok"], f"{nom}: desviacio relativa maxima "
                            + ", ".join(f"{k} {v:.1e}" for k, v in dv.items())
                            + f"; remostrejada {r['remostrejada']}, marca corregida {r['marca_corregida']}")

    # R3. Bit a bit, i que de dades/ nomes obre el que toca.
    print("\nR3. Reproduible bit a bit, i de dades/ nomes obre el .trk, el JSON i el CSV de la presa")
    m1 = mesura(cadena(t_f, (J0 + nJ)[queden], (G0 + nG)[queden]), 0.0, w0, fin, ordre, fins_a=50.0)
    m2 = mesura(cadena(t_f, (J0 + nJ)[queden], (G0 + nG)[queden]), 0.0, w0, fin, ordre, fins_a=50.0)
    netes = [{k: v for k, v in m.items() if not k.startswith("_")} for m in (m1, m2)]
    mira(json.dumps(netes[0]) == json.dumps(netes[1]), "fabricada: dues passades identiques")
    sys.addaudithook(_registra)
    _OBERTS = []
    a = mesura_presa(index["150_3"], w0)
    oberts, _OBERTS = _OBERTS, None
    b = mesura_presa(index["150_3"], w0)
    mira(json.dumps(a) == json.dumps(b), "150_3: dues passades identiques")
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve()) for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    esperats = sorted(str(x.resolve()) for x in (ARREL / "dades/video/tracker/150/150_3.trk",
                                                 NET / "150_3.json", NET / "150_3.csv"))
    mira(dins == esperats, "de dades/ obre: " + ", ".join(str(Path(x).relative_to(arrel_dades)) for x in dins))

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> recupera un soroll blanc i un de lent coneguts darrere del filtre, aparella")
    print("     els fotogrames del .trk amb el CSV com la conversio, es reproduible, nomes llegeix")
    print("     la presa, i la regressio geometrica separa la geometria del soroll.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    p = Parametres.des_de_json(ARREL / "parametres.json")
    index = json.loads(INDEX.read_text(encoding="utf-8"))["preses"]
    r = mesura_tot(index, p)
    imprimeix(r)
    falla = [n for n, v in r["per_presa"].items() if not v["r0"]["ok"]]
    if args.json:
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        sortida = {"eina": "eines/soroll_en_moviment.py", "pla": "documentacio/pla-analisi-caos.md",
                   "que_es": "soroll de seguiment EN MOVIMENT (0,1-10 s despres de l'amollada), de la "
                             "variacio de les llargades de les barres passada pel filtre de la conversio, "
                             "en la norma de mesures.distancia_fases; nomes veu l'error al llarg de la "
                             "barra; sense el biaix de theta1",
                   "finestra_s": [DES_DE_S, FINS_A_S], **r}
        cami.write_text(json.dumps(sortida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Desat: {cami}")
    if falla:
        print(f"R0 FALLA a {', '.join(falla)}: aquestes preses no entren als grups.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
