r"""La prova del banc de la joganza: el cargol de l'articulacio es menja energia a cada volta?

Pla, amb els criteris escrits ABANS de mirar: documentacio/pla-joganza.md (22/10/2026).

Muntatge de les preses 2.x: la barra 1 lligada al suport, la barra 2 lliure al voltant del
cargol de l'articulacio. Sis preses a 239,9 fps (dades/video/tracker/jogança/):

    IMG_0348  control, des de 150 graus          IMG_0351  gir, 1,5 voltes
    IMG_0349  des de 174 graus, sense passar     IMG_0352  gir, 8,5 voltes
                                                 IMG_0350  gir, 17,5 voltes
                                                 IMG_0353  gir, 33,5 voltes

El metode es el de friccio_pendol_doble.py --energia, amb les seves mateixes funcions:

    e = (4 pi^2 / T0^2)(1 - cos th) + w^2/2          (energia per unitat d'inercia)
    de/dt = -( s|w| + v w^2 + a|w|^3 )

La referencia (s, v, a) s'ajusta sobre el control. A les de gir, l'EXCES de cada volta es la
caiguda d'energia mesurada entre dos passos pel punt de baix menys la que dona la referencia
integrada amb la w mesurada. A la de 174 graus, el mateix de mig cicle en mig cicle mentre
l'amplitud passa de 150 graus. NO s'adopta cap coeficient: aixo produeix numeros.

    .venv\Scripts\python.exe eines\joganza.py --autoprova
    .venv\Scripts\python.exe eines\joganza.py --analitza
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import friccio_pendol_doble as fpd  # noqa: E402
from pendol.trk import llegeix_trk  # noqa: E402

DIR = ARREL / "dades" / "video" / "tracker" / "jogança"
FRICCIO = ARREL / "dades" / "friccio_doble"
CONTROL, ESCOMBRADA = "IMG_0348", "IMG_0349"
GIR = ("IMG_0351", "IMG_0352", "IMG_0350", "IMG_0353")

I2 = 3.162e-4               # kg m2: la barra 2 sobre el seu pivot, la de les 2.x ("inercia_kg_m2")
FINESTRA_INICI_S = 2.0      # el maxim d'energia que fa d'inici es busca en aquests primers segons
TALL_S = 0.1                # i l'analisi comenca aixo despres
RADI_MAX_PX = 3.0           # una volta queda fora si extrem-rodament s'aparta mes d'aixo
AMPLITUD_ESC = math.radians(150.0)
AMPLITUD_SOROLL = math.radians(90.0)   # el soroll per mig cicle, del control per sobre d'aixo
BOOT = 2000

# Segona part (22/10): el joc del cargol dins del forat.
D2_M = 0.051                # el centre de gravetat de la barra 2, des del seu pivot (CLAUDE.md)
L0_JOC = 0.5                # m/s2: l'amplada del tanh que fa de signe suau de la carrega
GOMET_MM = 139.0            # el gomet, a 13,9 cm del pivot (calibre, 9/09): fa l'escala px -> mm
BLOC_S = 1.0


# ------------------------------------------------------------------ lectura

_OBERTS: list[str] = []


def _registra(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.path.abspath(os.fsdecode(args[0])))


def fora_de_lloc(oberts) -> list[str]:
    """A5: els fitxers de dades/ oberts que no son ni de la joganza ni de friccio_doble."""
    dades = ARREL / "dades"
    mal = []
    for c in oberts:
        try:
            r = Path(c).relative_to(dades).as_posix()
        except ValueError:
            continue
        if not (r.startswith("video/tracker/jogança/") or r.startswith("friccio_doble/")):
            mal.append(r)
    return sorted(set(mal))


def llegeix(nom: str):
    """(t, xe, ye, xr, yr) en pixels d'imatge (y avall), nomes els fotogrames amb totes dues
    pistes. Diu tambe si hi falta cap fotograma."""
    d = llegeix_trk(DIR / f"{nom}.trk")
    ps = {k.lower(): v for k, v in d["pistes"].items()}
    fe, xe, ye = (np.asarray(v, float) for v in ps["extrem"])
    fr, xr, yr = (np.asarray(v, float) for v in ps["rodament"])
    com = np.intersect1d(fe, fr)
    ie, ir = np.searchsorted(fe, com), np.searchsorted(fr, com)
    t = (com - com[0]) * d["delta_t_s"]
    return t, xe[ie], ye[ie], xr[ir], yr[ir], {"forats": int((np.diff(com) > 1).sum()),
                                             "fps": float(d["fps"])}


def angle_i_energia(t, xe, ye, xr, yr, T0: float | None = None):
    """L'angle al voltant del CENTRE DEL CERCLE que fa l'extrem, amb zero al repos final.

    El centre s'ajusta (Kasa, fpd.ajusta_circumferencia, com l'eina de friccio a les 2.x):
    decidit per en Guillem el 22/10, perque la marca del rodament es a ~3,5 mm de l'eix de
    gir a totes les preses. Les pistes del rodament i del fix no es fan servir per al centre.

    Es gira el marc perque el repos caigui a +90 graus, que es el que espera
    fpd.angle_i_velocitat, i se'n fa servir la derivada de Savitzky-Golay tal com es."""
    cx, cy, _, _ = fpd.ajusta_circumferencia(xe, ye)
    x, y = xe - cx, ye - cy
    n1 = max(10, int(round(1.0 / np.median(np.diff(t)))))
    phi = math.atan2(float(np.median(y[-n1:])), float(np.median(x[-n1:])))
    al = math.pi / 2.0 - phi
    xg = math.cos(al) * x - math.sin(al) * y
    yg = math.sin(al) * x + math.cos(al) * y
    th, w, _, _ = fpd.angle_i_velocitat(t, xg, yg)
    radi = np.hypot(x, y)
    # El zero es el repos final: si l'ultim segon no es quiet, el zero es fals i l'energia
    # oscil.la amb l'angle. Es diu, i qui crida decideix.
    angle_i_energia.ultim_segon_graus = float(np.degrees(np.ptp(th[-n1:])))
    e = None if T0 is None else (4.0 * math.pi ** 2 / T0 ** 2) * (1.0 - np.cos(th)) + 0.5 * w ** 2
    return th, w, e, radi


def punts_de_retorn(t, th, w, des_de: float = 0.0):
    """(instant, angle) de cada canvi de signe de w, interpolats."""
    i = np.nonzero((w[:-1] * w[1:] < 0) & (t[:-1] >= des_de))[0]
    f = w[i] / (w[i] - w[i + 1])
    return t[i] + f * (t[i + 1] - t[i]), th[i] + f * (th[i + 1] - th[i])


def T0_de(t, th, w, des_de: float = 0.0) -> float:
    """El periode a amplitud zero, amb la funcio de l'eina de friccio (12-25 graus)."""
    tk, ak = punts_de_retorn(t, th, w, des_de)
    T0, _, _ = fpd.periode_a_amplitud_zero(tk, np.abs(ak))
    return float(T0)


def inici(t, e) -> float:
    """0,1 s despres del maxim de l'energia en els primers 2 s: la ma ja no hi es."""
    m = t <= t[0] + FINESTRA_INICI_S
    return float(t[m][int(np.argmax(e[m]))] + TALL_S)


def amollada_quieta(t, th) -> tuple[float, float]:
    """Per a les preses que s'amollen des del repos (el control i la de 174 graus).

    El maxim de l'energia no serveix: en amollar, l'energia es gairebe plana, i el biaix de la
    derivada el fa caure mes tard. Aqui: el primer instant en que la barra s'ha mogut 1 grau
    de l'angle on la ma l'aguantava (la mediana dels primers 0,5 s), menys 0,1 s; caure 1 grau
    des de 150-174 graus costa ~0,06 s, o sigui que aquell instant encara es quiet.
    Torna (instant de l'amollada, des d'on es busquen els punts de retorn)."""
    th_ma = float(np.median(th[t <= t[0] + 0.5]))
    i = int(np.argmax(np.abs(th - th_ma) > math.radians(1.0)))
    return float(t[i] - TALL_S), float(t[i] + TALL_S)


def referencia(t, th, w, T0: float, des_de: float) -> dict:
    """s, v i a amb fpd.ajusta_per_energia tal com es, sobre el registre sencer."""
    s = t >= des_de
    L_eq = fpd.G_VIC * T0 ** 2 / (4.0 * math.pi ** 2)
    r = fpd.ajusta_per_energia(t[s], th[s], w[s], L_eq, sostre_graus=None)
    return {"s": r["s_sec_rad_s-2"], "v": r["v_viscos_s-1"], "a": r["a_aerodinamic_rad-1"],
            "errors": r["errors"], "ajust": r}


def _integrals(t, w):
    aw = np.abs(w)
    return (cumulative_trapezoid(aw, t, initial=0.0), cumulative_trapezoid(aw ** 2, t, initial=0.0),
            cumulative_trapezoid(aw ** 3, t, initial=0.0))


# ------------------------------------------------------------------ les mesures

def per_volta(t, th, w, e, radi, ref, des_de: float) -> dict:
    """L'exces de cada volta sencera: entre dos passos seguits pel punt de baix (th = 2 pi n)
    amb n que avanca en el mateix sentit."""
    m = th / (2.0 * math.pi)
    n = np.floor(m)
    i = np.nonzero((n[:-1] != n[1:]) & (t[:-1] >= des_de))[0]
    nint = np.maximum(n[i], n[i + 1])            # l'enter que es creua
    f = (nint - m[i]) / (m[i + 1] - m[i])
    tc = t[i] + f * (t[i + 1] - t[i])
    sentit = np.sign(m[i + 1] - m[i])
    I = _integrals(t, w)
    ec = np.interp(tc, t, e)
    R = float(np.median(radi[t >= des_de]))
    voltes, fora = [], {"radi": 0}
    for j in range(len(tc) - 1):
        if not (nint[j + 1] == nint[j] + sentit[j] and sentit[j + 1] == sentit[j]):
            continue
        a, b = tc[j], tc[j + 1]
        dins = (t >= a) & (t <= b)
        if np.abs(radi[dins] - R).max() > RADI_MAX_PX:
            fora["radi"] += 1
            continue
        ints = [float(np.interp(b, t, x) - np.interp(a, t, x)) for x in I]
        mes = float(ec[j] - ec[j + 1])
        pred = ref["s"] * ints[0] + ref["v"] * ints[1] + ref["a"] * ints[2]
        voltes.append({"t0": float(a), "t1": float(b), "w_mitjana": float(np.abs(w[dins]).mean()),
                       "mesurada_mJ": 1e3 * I2 * mes, "prevista_mJ": 1e3 * I2 * pred,
                       "exces_mJ": 1e3 * I2 * (mes - pred),
                       "derivades_mJ": [-1e3 * I2 * x for x in ints]})
    return {"voltes": voltes, "fora": fora, "radi_px": R}


def per_mig_cicle(t, th, w, e, ref, des_de: float, amplitud_min: float,
                  t_amollada: float | None = None) -> list[dict]:
    """L'exces de cada mig cicle entre dos punts de retorn seguits, tots dos per sobre
    d'amplitud_min. Al punt de retorn w = 0: l'energia es tota potencial.

    L'amollada (la barra quieta a la ma) es el primer punt de retorn. Els altres es
    busquen a partir de des_de: abans, el soroll de la ma aguantant faria punts falsos."""
    tk, ak = punts_de_retorn(t, th, w, des_de)
    if t_amollada is not None:
        tk = np.concatenate([[t_amollada], tk])
        ak = np.concatenate([[np.interp(t_amollada, t, th)], ak])
    I = _integrals(t, w)
    ek = np.interp(tk, t, e)
    fora = []
    for j in range(len(tk) - 1):
        if abs(ak[j]) < amplitud_min or abs(ak[j + 1]) < amplitud_min:
            continue
        ints = [float(np.interp(tk[j + 1], t, x) - np.interp(tk[j], t, x)) for x in I]
        mes = float(ek[j] - ek[j + 1])
        pred = ref["s"] * ints[0] + ref["v"] * ints[1] + ref["a"] * ints[2]
        fora.append({"t0": float(tk[j]), "amplitud_graus": math.degrees(abs(ak[j])),
                     "mesurada_mJ": 1e3 * I2 * mes, "prevista_mJ": 1e3 * I2 * pred,
                     "exces_mJ": 1e3 * I2 * (mes - pred),
                     "derivades_mJ": [-1e3 * I2 * x for x in ints]})
    return fora


def resum_gir(per_presa: dict, ref, llavor: int = 0) -> dict:
    """L'exces mitja per volta i el pendent contra w, amb bootstrap per preses i voltes, i la
    incertesa de la referencia propagada sense correlacions (sobreestimada: v i a hi van
    anticorrelacionats)."""
    tot = [v for vs in per_presa.values() for v in vs]
    if len(tot) < 3:
        return {"n": len(tot)}
    x = np.array([v["w_mitjana"] for v in tot])
    y = np.array([v["exces_mJ"] for v in tot])
    q, (pend, _) = float(y.mean()), np.polyfit(x, y, 1)
    rng = np.random.default_rng(llavor)
    blocs = [np.array([[v["w_mitjana"], v["exces_mJ"]] for v in vs]) for vs in per_presa.values() if vs]
    bq, bp = [], []
    for _ in range(BOOT):
        tria = [blocs[k] for k in rng.integers(len(blocs), size=len(blocs))]
        mostra = np.vstack([b[rng.integers(len(b), size=len(b))] for b in tria])
        bq.append(mostra[:, 1].mean())
        bp.append(np.polyfit(mostra[:, 0], mostra[:, 1], 1)[0] if np.ptp(mostra[:, 0]) > 0 else np.nan)
    D = np.array([v["derivades_mJ"] for v in tot])        # d(exces)/d(s, v, a)
    err = np.array(ref["errors"])
    sq_ref = float(np.sqrt(np.sum((D.mean(axis=0) * err) ** 2)))
    sp_ref = float(np.sqrt(np.sum([(np.polyfit(x, D[:, k], 1)[0] * err[k]) ** 2 for k in range(3)])))
    sq = float(np.hypot(np.std(bq, ddof=1), sq_ref))
    sp = float(np.hypot(np.nanstd(bp, ddof=1), sp_ref))
    return {"n": len(tot), "exces_mitja_mJ": q, "sigma_mJ": sq, "pendent_mJ_per_rad_s": float(pend),
            "sigma_pendent": sp, "sigma_ref_mJ": sq_ref, "sigma_ref_pendent": sp_ref}


def resum_escombrada(mig: list[dict], sigma_mig: float, ref) -> dict:
    if not mig:
        return {"n": 0}
    exc = sum(m["exces_mJ"] for m in mig)
    pred = sum(m["prevista_mJ"] for m in mig)
    D = np.array([m["derivades_mJ"] for m in mig]).sum(axis=0)
    s_ref = float(np.sqrt(np.sum((D * np.array(ref["errors"])) ** 2)))
    s = float(np.hypot(sigma_mig * math.sqrt(len(mig)), s_ref))
    return {"n": len(mig), "exces_mJ": float(exc), "prevista_mJ": float(pred), "sigma_mJ": s}


def veredicte(g: dict, esc: dict) -> dict:
    """Les quatre condicions del pla; exactament una -> aquella, si no V5."""
    sense_esc = abs(esc["exces_mJ"]) <= 2 * esc["sigma_mJ"]
    c = {
        "V1": bool(g["exces_mitja_mJ"] > 3 * g["sigma_mJ"]
                   and abs(g["pendent_mJ_per_rad_s"]) <= 2 * g["sigma_pendent"] and sense_esc),
        "V2": bool(esc["exces_mJ"] > 2 * esc["sigma_mJ"] and esc["exces_mJ"] > 0.10 * esc["prevista_mJ"]),
        "V3": bool(abs(g["exces_mitja_mJ"]) <= 2 * g["sigma_mJ"] and sense_esc),
        "V4": bool(g["pendent_mJ_per_rad_s"] > 3 * g["sigma_pendent"]),
    }
    certes = [k for k, v in c.items() if v]
    return {"condicions": c, "veredicte": certes[0] if len(certes) == 1 else "V5", "certes": certes}


def analitza_preses(dades: dict, ref_donada: dict | None = None) -> dict:
    """El circuit sencer sobre un diccionari {nom: (t, xe, ye, xr, yr)}: el que fa --analitza
    i el que fa l'autoprova amb preses fabricades."""
    tC, *posC = dades[CONTROL]
    thC, wC, _, radiC = angle_i_energia(tC, *posC)
    T0 = T0_de(tC, thC, wC)
    _, _, eC, _ = angle_i_energia(tC, *posC, T0=T0)
    amC, iniC = amollada_quieta(tC, thC)
    ref = ref_donada or referencia(tC, thC, wC, T0, amC)
    migC = per_mig_cicle(tC, thC, wC, eC, ref, iniC, AMPLITUD_SOROLL, t_amollada=amC)
    sigma_mig = float(np.std([m["exces_mJ"] for m in migC], ddof=1)) if len(migC) >= 3 else float("nan")

    tE, *posE = dades[ESCOMBRADA]
    thE, wE, eE, _ = angle_i_energia(tE, *posE, T0=T0)
    amE, iniE = amollada_quieta(tE, thE)
    migE = per_mig_cicle(tE, thE, wE, eE, ref, iniE, AMPLITUD_ESC, t_amollada=amE)
    esc = resum_escombrada(migE, sigma_mig, ref)

    gir, info = {}, {}
    for nom in GIR:
        t, *pos = dades[nom]
        th, w, e, radi = angle_i_energia(t, *pos, T0=T0)
        i0 = inici(t, e)
        pv = per_volta(t, th, w, e, radi, ref, i0)
        gir[nom] = pv["voltes"]
        info[nom] = {"inici_s": i0, "voltes_bones": len(pv["voltes"]), "fora": pv["fora"],
                     "radi_px": pv["radi_px"], "t_maxim_energia_global_s": float(t[int(np.argmax(e))])}
    g = resum_gir(gir, ref)
    out = {"T0_s": T0, "referencia": {k: ref[k] for k in ("s", "v", "a", "errors")},
           "inici_control_s": iniC, "inici_175_s": iniE, "sigma_mig_cicle_mJ": sigma_mig,
           "escombrada": esc, "mig_cicles_175": migE, "gir": g, "voltes": gir, "info_gir": info}
    if g.get("n", 0) >= 3 and esc.get("n", 0) > 0:
        out["veredicte"] = veredicte(g, esc)
    else:
        # Sense cap mig cicle de la 0349 per sobre de 150 graus no es pot avaluar la V2, ni el
        # «cap exces a la 0349» de la V1 i la V3: el veredicte es V5, amb el motiu.
        out["veredicte"] = {"veredicte": "V5", "certes": [], "condicions": {},
                            "motiu": (f"voltes de gir: {g.get('n', 0)}; migs cicles de la 0349 per "
                                      f"sobre de 150 graus: {esc.get('n', 0)}")}
    return out


# ------------------------------------------------------------------ l'autoprova

def fabrica(th0, w0, coefs, T0, q_mJ=0.0, temps=40.0, fps=239.9, repos_s=1.0, empenta_s=0.0,
            soroll_px=0.3, R=227.0, llavor=0, rodament_fora_px=0.0, joc_px=0.0, elipse_px=0.0):
    """Una barra 2 fabricada: repos, (empenta), i decaiment amb s, v i a, i un cop de q a cada
    pas per dalt. Torna (t, xe, ye, xr, yr) en pixels, i l'instant en que s'acaba l'empenta."""
    s, v, a = coefs
    k = 4.0 * math.pi ** 2 / T0 ** 2
    q = 1e-3 * q_mJ / I2
    eps = 1e-3

    def f(_, y, tau=0.0):
        th, w = y
        return [w, -k * math.sin(th) - s * math.tanh(w / eps) - v * w - a * w * abs(w) + tau]

    tt, yy = [np.array([0.0])], [np.array([[th0], [0.0]])]
    t_lliure = repos_s
    if empenta_s > 0:
        tau = w0 / empenta_s
        sol = solve_ivp(lambda t_, y: f(t_, y, tau), (repos_s, repos_s + empenta_s), [th0, 0.0],
                        rtol=1e-10, atol=1e-12, dense_output=True)
        y0 = sol.y[:, -1]
        t_lliure = repos_s + empenta_s
        tt.append(np.array([repos_s, t_lliure]))
        segm = [(sol, repos_s, t_lliure)]
    else:
        y0 = np.array([th0, w0])
        segm = []

    def dalt(_, y):
        return math.cos(y[0] / 2.0)
    dalt.terminal = True

    t_ara = t_lliure
    while t_ara < temps:
        sol = solve_ivp(f, (t_ara, temps), y0, rtol=1e-10, atol=1e-12, dense_output=True, events=dalt)
        segm.append((sol, t_ara, sol.t[-1]))
        if sol.status == 1:
            th_, w_ = sol.y_events[0][0]
            ec = 0.5 * w_ ** 2 - q          # la cinetica (per unitat d'inercia) despres del cop
            w_ = math.copysign(math.sqrt(max(2.0 * ec, 0.0)), w_)
            y0 = np.array([th_ + 1e-9 * math.copysign(1.0, w_), w_])
            t_ara = sol.t[-1]
        else:
            break

    t = np.arange(0.0, temps, 1.0 / fps)
    th = np.full_like(t, th0)
    w = np.zeros_like(t)
    for sol, a_, b_ in segm:
        m = (t >= a_) & (t <= b_)
        th[m] = sol.sol(t[m])[0]
        w[m] = sol.sol(t[m])[1]
    rng = np.random.default_rng(llavor)
    cx, cy = 500.0, 400.0
    # El joc (segona part, 22/10): el radi salta +-joc_px segons el signe de la carrega L, i la
    # camera hi posa una el.lipse (harmonic 2). Amb tots dos a zero, les preses d'abans.
    Rt = R + joc_px * np.sign(w ** 2 * D2_M + fpd.G_VIC * np.cos(th)) + elipse_px * np.cos(2 * th)
    xe = cx + Rt * np.sin(th) + soroll_px * rng.standard_normal(len(t))
    ye = cy + Rt * np.cos(th) + soroll_px * rng.standard_normal(len(t))
    xr = cx + rodament_fora_px + soroll_px * rng.standard_normal(len(t))
    yr = cy + soroll_px * rng.standard_normal(len(t))
    return (t, xe, ye, xr, yr), t_lliure


def joc_fabricat(coefs, T0, q_mJ=0.0, a_gir=1.0, a_tots=1.0, llavor=0, rodament_fora_px=0.0):
    """El control (150 graus), la de 174 i quatre de gir, totes fabricades."""
    s, v, a = coefs
    base = (s, v, a * a_tots)
    rf = rodament_fora_px
    d = {CONTROL: fabrica(math.radians(150), 0.0, base, T0, temps=80.0, llavor=llavor,
                          rodament_fora_px=rf)[0],
         ESCOMBRADA: fabrica(math.radians(174), 0.0, base, T0, temps=80.0, llavor=llavor + 1,
                             rodament_fora_px=rf)[0]}
    girc = (s, v, a * a_gir * a_tots)
    # L'empenta lluita contra la gravetat: amb aquests objectius, la barra surt a ~19-38 rad/s,
    # com les preses de debo (19-41).
    for j, (nom, w0) in enumerate(zip(GIR, (30.0, 38.0, 46.0, 60.0))):
        d[nom] = fabrica(0.0, w0, girc, T0, q_mJ=q_mJ, temps=80.0, empenta_s=0.2,
                         llavor=llavor + 10 + j, rodament_fora_px=rf)[0]
    return d


def autoprova() -> int:
    coefs = (0.43, 0.0, 0.0065)     # de l'ordre de les 2.x (energia, registre sencer)
    T0 = 0.657
    mal = []
    print("Autoprova de l'eina de la joganza (preses fabricades)\n")

    r = analitza_preses(joc_fabricat(coefs, T0, q_mJ=0.0, llavor=100))
    g, ver = r["gir"], r.get("veredicte", {})
    ok = abs(g["exces_mitja_mJ"]) <= 2 * g["sigma_mJ"] and ver.get("veredicte") != "V1"
    print(f"   {'ok ' if ok else 'NO '} A1 sense cops: exces {g['exces_mitja_mJ']:+.3f} +- {g['sigma_mJ']:.3f} mJ, "
          f"pendent {g['pendent_mJ_per_rad_s']:+.4f} +- {g['sigma_pendent']:.4f}; veredicte {ver.get('veredicte')} "
          f"({ver.get('certes')})")
    if not ok:
        mal.append("A1")

    r = analitza_preses(joc_fabricat(coefs, T0, q_mJ=2.8, llavor=200))
    r_a2 = r
    g, ver = r["gir"], r.get("veredicte", {})
    ok = (abs(g["exces_mitja_mJ"] / 2.8 - 1) <= 0.15 and abs(g["pendent_mJ_per_rad_s"]) <= 2 * g["sigma_pendent"]
          and ver.get("veredicte") == "V1")
    print(f"   {'ok ' if ok else 'NO '} A2 cops de 2,8 mJ: exces {g['exces_mitja_mJ']:+.3f} +- {g['sigma_mJ']:.3f} mJ, "
          f"pendent {g['pendent_mJ_per_rad_s']:+.4f} +- {g['sigma_pendent']:.4f}; veredicte {ver.get('veredicte')} "
          f"({ver.get('certes')})")
    if not ok:
        mal.append("A2")

    # La referencia «feta amb x1,0» (pla): ajustada sobre el control fabricat, que te l'aire
    # x1,0; nomes les de gir el tenen x1,5. (22/10: abans s'hi donava una referencia inventada.)
    r = analitza_preses(joc_fabricat(coefs, T0, a_gir=1.5, llavor=300))
    g, ver = r["gir"], r.get("veredicte", {})
    ok = ver.get("veredicte") == "V4"
    print(f"   {'ok ' if ok else 'NO '} A3 aire x1,5 a les de gir: pendent {g['pendent_mJ_per_rad_s']:+.4f} +- "
          f"{g['sigma_pendent']:.4f}; veredicte {ver.get('veredicte')} ({ver.get('certes')})")
    if not ok:
        mal.append("A3")
    r = analitza_preses(joc_fabricat(coefs, T0, a_tots=1.5, llavor=400))
    ver = r.get("veredicte", {})
    print(f"   --  A3b (informativa) aire x1,5 a totes: veredicte {ver.get('veredicte')} ({ver.get('certes')})")

    (t, *pos), t_lliure = fabrica(0.0, 25.0, coefs, T0, empenta_s=0.2, temps=80.0, llavor=500)
    th, w, e, _ = angle_i_energia(t, *pos, T0=T0)
    i0 = inici(t, e)
    ok = t_lliure <= i0 <= t_lliure + 0.15
    print(f"   {'ok ' if ok else 'NO '} A4 l'empenta s'acaba a {t_lliure:.3f} s i l'analisi comenca a {i0:.3f} s")
    if not ok:
        mal.append("A4")

    prova = [str(ARREL / "dades" / "net" / "preses.json"), str(DIR / "IMG_0348.trk"),
             str(FRICCIO / "Friccio_2.1.json")]
    ok = fora_de_lloc(prova) == ["net/preses.json"]
    print(f"   {'ok ' if ok else 'NO '} A5 el guardia d'obertures marca nomes el que no toca: {fora_de_lloc(prova)}")
    if not ok:
        mal.append("A5")

    # A6 (22/10, abans de mirar cap resultat amb el centre ajustat): el rodament marcat a 6 px de
    # l'eix dona EXACTAMENT el mateix, perque el centre ja no surt del rodament.
    r6 = analitza_preses(joc_fabricat(coefs, T0, q_mJ=2.8, llavor=200, rodament_fora_px=6.0))
    ok = (r6["gir"]["exces_mitja_mJ"] == r_a2["gir"]["exces_mitja_mJ"]
          and r6["veredicte"]["veredicte"] == r_a2["veredicte"]["veredicte"])
    print(f"   {'ok ' if ok else 'NO '} A6 rodament a 6 px de l'eix: exces {r6['gir']['exces_mitja_mJ']:+.4f} "
          f"contra {r_a2['gir']['exces_mitja_mJ']:+.4f} mJ, veredicte {r6['veredicte']['veredicte']}")
    if not ok:
        mal.append("A6")

    print("\n" + ("OK -> A1-A6" if not mal else f"AUTOPROVA FALLADA: {mal}"))
    return 1 if mal else 0


# ------------------------------------------------------------------ segona part: el joc del cargol

def joc_de(t, xe, ye, xr, yr, des_de: float, llavor: int = 0) -> dict:
    """La meitat del joc, k (px), de la regressio del radi:

        r - R = a1 cos th + b1 sin th + a2 cos 2th + b2 sin 2th + k tanh(L/L0)
        L = w^2 d2 + g cos th          (la carrega de la barra sobre el cargol, per unitat de massa)

    amb error per bootstrap de blocs d'1 s."""
    th, w, _, radi = angle_i_energia(t, xe, ye, xr, yr)
    m = t >= des_de
    th, w, r, tt = th[m], w[m], radi[m], t[m]
    L = w ** 2 * D2_M + fpd.G_VIC * np.cos(th)
    X = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th), np.cos(2 * th), np.sin(2 * th),
                         np.tanh(L / L0_JOC)])
    beta = np.linalg.lstsq(X, r, rcond=None)[0]
    bloc = np.floor((tt - tt[0]) / BLOC_S).astype(int)
    blocs = [np.nonzero(bloc == b)[0] for b in np.unique(bloc)]
    rng = np.random.default_rng(llavor)
    ks = []
    for _ in range(500):
        idx = np.concatenate([blocs[i] for i in rng.integers(len(blocs), size=len(blocs))])
        ks.append(np.linalg.lstsq(X[idx], r[idx], rcond=None)[0][5])
    R = float(beta[0])
    return {"k_px": float(beta[5]), "sigma_px": float(np.std(ks, ddof=1)), "R_px": R,
            "elipse_px": float(np.hypot(beta[3], beta[4])),
            "fotogrames_L_negativa": int((L < 0).sum()), "fotogrames": int(m.sum()),
            "px_per_mm": R / GOMET_MM}


def resum_joc(per_presa: dict) -> dict:
    """La k de les preses amb prou carrega negativa, combinada amb pesos 1/sigma^2."""
    us = {n: d for n, d in per_presa.items() if d["fotogrames_L_negativa"] >= 240}
    if not us:
        return {"veredicte": "no concloent", "motiu": "cap presa amb 1 s de carrega negativa"}
    k = np.array([d["k_px"] for d in us.values()])
    s = np.array([d["sigma_px"] for d in us.values()])
    wgt = 1 / s ** 2
    km, sm = float((k * wgt).sum() / wgt.sum()), float(1 / math.sqrt(wgt.sum()))
    mateix_signe = bool(np.all(np.sign(k) == np.sign(km)))
    pxmm = float(np.mean([d["px_per_mm"] for d in us.values()]))
    if km > 3 * sm and mateix_signe:
        ver = "joc detectat"
    elif abs(km) <= 2 * sm:
        ver = "no detectat"
    else:
        ver = "no concloent"
    return {"preses": sorted(us), "k_px": km, "sigma_px": sm, "joc_mm": 2 * km / pxmm,
            "sigma_joc_mm": 2 * sm / pxmm, "sostre_2sigma_mm": 2 * (abs(km) + 2 * sm) / pxmm,
            "mateix_signe": mateix_signe, "veredicte": ver}


def autoprova_joc() -> int:
    coefs, T0 = (0.43, 0.0, 0.0065), 0.657
    mal = []
    print("Autoprova del joc del cargol (preses fabricades)\n")
    for nom_cas, kfab in (("J1 joc de 0,8 px", 0.8), ("J2 sense joc", 0.0)):
        per = {}
        for j, (th0, w0, emp) in enumerate(((math.radians(150), 0.0, 0.0), (math.radians(174), 0.0, 0.0),
                                            (0.0, 30.0, 0.2))):
            (t, *pos), tl = fabrica(th0, w0, coefs, T0, temps=80.0, empenta_s=emp, llavor=700 + j,
                                    joc_px=kfab, elipse_px=1.0, rodament_fora_px=6.0)
            th, _, _, _ = angle_i_energia(t, *pos)
            des = amollada_quieta(t, th)[1] if emp == 0 else tl + TALL_S
            per[f"f{j}"] = joc_de(t, *pos, des, llavor=j)
        r = resum_joc(per)
        if kfab > 0:
            ok = abs(r["k_px"] / kfab - 1) <= 0.25
        else:
            ok = abs(r["k_px"]) <= 2 * r["sigma_px"]
        print(f"   {'ok ' if ok else 'NO '} {nom_cas}: k = {r['k_px']:+.3f} +- {r['sigma_px']:.3f} px "
              f"(fabricat {kfab}); {r['veredicte']}")
        if not ok:
            mal.append(nom_cas.split()[0])
    print("\n" + ("OK -> J1-J2" if not mal else f"AUTOPROVA FALLADA: {mal}"))
    return 1 if mal else 0


def analitza_joc(sortida: Path) -> int:
    sys.addaudithook(_registra)
    per = {}
    for nom in (CONTROL, ESCOMBRADA) + GIR:
        t, xe, ye, xr, yr, _ = llegeix(nom)
        th, w, _, _ = angle_i_energia(t, xe, ye, xr, yr)
        if nom in (CONTROL, ESCOMBRADA):
            des = amollada_quieta(t, th)[1]
        else:
            T0 = 0.6584   # el del control (primera part); nomes serveix per trobar l'inici
            _, _, e, _ = angle_i_energia(t, xe, ye, xr, yr, T0=T0)
            des = inici(t, e)
        per[nom] = joc_de(t, xe, ye, xr, yr, des)
    r = {"per_presa": per, "resum": resum_joc(per), "fora_de_lloc": fora_de_lloc(_OBERTS)}
    sortida.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    for nom, d in per.items():
        print(f"   {nom}: k {d['k_px']:+.3f} +- {d['sigma_px']:.3f} px; el.lipse {d['elipse_px']:.2f} px; "
              f"carrega negativa {d['fotogrames_L_negativa']} de {d['fotogrames']} fotogrames")
    s = r["resum"]
    print(f"\nRESUM ({s.get('preses')}): k {s.get('k_px', float('nan')):+.3f} +- {s.get('sigma_px', float('nan')):.3f} px"
          f" -> joc {s.get('joc_mm', float('nan')):+.3f} +- {s.get('sigma_joc_mm', float('nan')):.3f} mm; "
          f"sostre a 2 sigma {s.get('sostre_2sigma_mm', float('nan')):.3f} mm; {s['veredicte']}")
    print(f"A5 fora de lloc: {r['fora_de_lloc'] or 'cap'}\n-> {sortida}")
    return 0


# ------------------------------------------------------------------ l'analisi de debo

def r0(ref) -> dict:
    """R0: el control dins de la dispersio de les tres 2.x (energia, registre sencer) a 2 sigma."""
    v = []
    for n in ("2.1", "2.2", "2.3"):
        d = json.loads((FRICCIO / f"Friccio_{n}.json").read_text(encoding="utf-8"))["energia_sencera"]
        v.append([d["s_sec_rad_s-2"], d["v_viscos_s-1"], d["a_aerodinamic_rad-1"]])
    v = np.array(v)
    mitja, sd = v.mean(axis=0), v.std(axis=0, ddof=1)
    c = np.array([ref["s"], ref["v"], ref["a"]])
    lim = 2 * np.sqrt(np.array(ref["errors"]) ** 2 + sd ** 2)
    return {"2x_mitjana": mitja.tolist(), "2x_sd": sd.tolist(), "control": c.tolist(),
            "dins": [bool(abs(c[k] - mitja[k]) <= lim[k]) for k in range(3)],
            "nota": "la 2.1 es provisional fins que es resolgui el criteri F1 (22/10)"}


def analitza(sortida: Path) -> int:
    sys.addaudithook(_registra)
    dades, lect = {}, {}
    for nom in (CONTROL, ESCOMBRADA) + GIR:
        t, xe, ye, xr, yr, info = llegeix(nom)
        dades[nom] = (t, xe, ye, xr, yr)
        lect[nom] = info
    # El tall de la ma es busca en els primers 2 s (pla). Si el maxim d'energia de tota la presa
    # cau mes tard, el criteri no serveix per a aquella presa: s'atura abans de cap resultat.
    for nom in (CONTROL, ESCOMBRADA) + GIR:
        t, *pos = dades[nom]
        th, w, _, _ = angle_i_energia(t, *pos)
        if angle_i_energia.ultim_segon_graus > 1.0:
            print(f"ATURAT: {nom} no acaba quiet (l'ultim segon es mou "
                  f"{angle_i_energia.ultim_segon_graus:.1f} graus): el zero de l'angle seria fals.")
            return 2
    for nom in GIR:
        t, *pos = dades[nom]
        th, w, _, _ = angle_i_energia(t, *pos)
        e = 0.5 * w ** 2 + 90.0 * (1 - np.cos(th))
        tm = float(t[int(np.argmax(e))])
        if tm > FINESTRA_INICI_S:
            print(f"ATURAT: a {nom} el maxim d'energia cau a {tm:.2f} s, despres dels primers "
                  f"{FINESTRA_INICI_S:g} s. El criteri del tall no hi val, i s'ha de decidir.")
            return 2
    r = analitza_preses(dades)
    r["R0"] = r0(r["referencia"])
    r["lectura"] = lect
    mal = fora_de_lloc(_OBERTS)
    r["A5_fora_de_lloc"] = mal
    sortida.write_text(json.dumps(r, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    g, esc = r["gir"], r["escombrada"]
    ref = r["referencia"]
    print(f"T0 del control {r['T0_s']:.4f} s;  referencia s {ref['s']:.4f}, v {ref['v']:.4g}, a {ref['a']:.5f}"
          f"  (errors {', '.join(f'{x:.3g}' for x in ref['errors'])})")
    print(f"R0 (control dins de les 2.x a 2 sigma, s / v / a): {r['R0']['dins']}  [{r['R0']['nota']}]")
    for nom, inf in r["info_gir"].items():
        print(f"   {nom}: inici {inf['inici_s']:.2f} s, voltes bones {inf['voltes_bones']}, fora {inf['fora']}")
    print(f"GIR: {g.get('n')} voltes; exces mitja {g.get('exces_mitja_mJ', float('nan')):+.3f} +- "
          f"{g.get('sigma_mJ', float('nan')):.3f} mJ/volta; pendent contra w "
          f"{g.get('pendent_mJ_per_rad_s', float('nan')):+.4f} +- {g.get('sigma_pendent', float('nan')):.4f} mJ/(rad/s)")
    print(f"174 graus: {esc.get('n')} migs cicles per sobre de 150; exces {esc.get('exces_mJ', float('nan')):+.3f} +- "
          f"{esc.get('sigma_mJ', float('nan')):.3f} mJ sobre {esc.get('prevista_mJ', float('nan')):.3f} previstos")
    if "veredicte" in r:
        print(f"VEREDICTE: {r['veredicte']['veredicte']}  (condicions certes: {r['veredicte']['certes']})")
    print(f"A5 fitxers de dades/ fora de lloc: {mal or 'cap'}")
    print(f"\n-> {sortida}")
    return 1 if mal else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    ap.add_argument("--analitza", action="store_true")
    ap.add_argument("--joc", action="store_true",
                    help="la segona part: el joc del cargol (amb --autoprova o --analitza)")
    ap.add_argument("--sortida", default=str(ARREL / "esborranys" / "joganza.json"))
    a = ap.parse_args(argv)
    if a.autoprova:
        return autoprova_joc() if a.joc else autoprova()
    if a.analitza:
        return analitza_joc(Path(a.sortida)) if a.joc else analitza(Path(a.sortida))
    ap.error("digues --autoprova o --analitza (i --joc per a la segona part)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
