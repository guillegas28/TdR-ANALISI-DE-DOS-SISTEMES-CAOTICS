r"""L'experiment 1: el feix contra cada tir real, i la Y de la dotzena llegida en un sol sentit (la D4, 27/10/2026).

Pla i criteris (escrits abans del codi): documentacio/pla-d4-experiment1.md. La regla: documentacio/per-firmar-24-10.md,
seccions 1, 3 i 3e.

EL FEIX d'una presa es porta bit a bit d'esborranys/_conflicte_y.py (simula_feix_r, amb els rangs del «cargol mesurat», el
repos encallat i l'error del centre de l'onzena tanda), des de les condicions inicials de la presa corregida
(comparar.corregeix_presa: als 45 i 90 graus, les dels clics). El zero del rellotge (simulador.afina_amollada, 0,085 s), el
soroll del tir dins del feix, l'horitzo, la Y d'un tir i el veredicte del grup son els de pendol/comparar.py.

LA LECTURA, EN UN SOL SENTIT: un grup que passa es un resultat; un que falla no ho es. La creu del Tracker fa fallar pendols
bons, pero no amaga els defectes (la catorzena tanda). Els 150 graus, «no avaluables».

EL PANY: sense dades/versemblant/firma.json, o si l'ordre no demana el que s'ha firmat (la X, la c, el fora, la finestra del
zero i els grups), s'atura abans d'obrir res de dades/net/.

    .venv\Scripts\python.exe eines\experiment1.py --autoprova
    .venv\Scripts\python.exe eines\experiment1.py y --x 5 --json dades/experiment1/y_<data>.json [--processos N]
    .venv\Scripts\python.exe eines\experiment1.py cap_on --json dades/experiment1/cap_on_<data>.json   (posterior, NO firmada)
    .venv\Scripts\python.exe eines\experiment1.py figura   (les figures 1 i 2, a figures/; pla-figura-i-eines-experiment1.md)
    .venv\Scripts\python.exe eines\experiment1.py salt --json dades/experiment1/salt_<data>.json   (mirat despres del veredicte)
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import zlib
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol import comparar, dades  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.simulador import afina_amollada, simula  # noqa: E402

import prepara_versemblant as pv  # noqa: E402

FIRMA = ARREL / "dades" / "versemblant" / "firma.json"
NUMEROS = ARREL / "dades" / "versemblant" / "numeros_2026-10-06_muntatge.json"   # quines preses fan cada grup
SOROLL_MOV = ARREL / "dades" / "divergencia" / "soroll_moviment_2026-10-03.json"
PARAMETRES = ARREL / "parametres.json"

# El feix firmat (secció 1 de per-firmar-24-10.md), tal com el construeix l'esborrany a la dotzena tanda.
MEMBRES = pv.MEMBRES                     # 200, i el nominal
AMPLADA_GRAUS = 0.1                      # condicions inicials
CARGOL_MESURAT = (0.0, 1.0, 1.0, 1.0)    # el cargol a 12,00 cm; la I2, la g i la L1, amb el rang sencer
AMPLADA_ENCALLAT_GRAUS = (0.15, 0.30)    # el repos encallat, theta1 i theta2 (en Guillem, 24/10)
SIGMA_CENTRE_PX = {"45": (0.05, 0.34), "90": (0.16, 0.23)}   # l'error del centre, de costat i vertical (24/10)
T_FEIX_S = 1.5                           # fins on arriba el feix
T_SOROLL_S = 1.0                         # les primeres mostres del soroll surten d'una font, la resta d'una altra
PAS_FEIX_S = 2.5e-4                      # graella densa del feix
GRUPS_Y = ("45", "90")                   # els 150, «no avaluables»
N_GRUP = {"45": 12, "90": 11}


def llavor(*parts) -> int:
    return zlib.crc32("|".join(str(x) for x in parts).encode("utf-8"))


# ------------------------------------------------------------ el feix (bit a bit de l'esborrany)

def mesura_amb_repos_encallat(th1, th2, e1: float, e2: float, L1: float, Lg: float):
    """El que es mesura d'un pendol si al repos final era a (e1, e2) -dins de la zona morta del fregament sec- i la presa
    es corregeix perque el repos doni (0, 0) (desena tanda, 24/10). Amb e1 = e2 = 0, els mateixos objectes."""
    if e1 == 0.0 and e2 == 0.0:
        return th1, th2
    th1, th2 = np.asarray(th1, dtype=float), np.asarray(th2, dtype=float)
    P2r = np.array([L1 * math.sin(e1), -L1 * math.cos(e1)])
    Gr = P2r + Lg * np.array([math.sin(e2), -math.cos(e2)])
    v = math.atan2(Gr[0], -Gr[1])
    u = np.array([math.sin(v), -math.cos(v)])
    n1r = np.array([math.cos(e1), math.sin(e1)])        # de costat de la barra 1, al repos
    creu = lambda a, b: a[0] * b[1] - a[1] * b[0]       # noqa: E731
    d = -creu(P2r, u) / creu(n1r, u)
    x1 = L1 * np.sin(th1) + d * np.cos(th1)
    y1 = -L1 * np.cos(th1) + d * np.sin(th1)
    xg = L1 * np.sin(th1) + Lg * np.sin(th2)
    yg = -L1 * np.cos(th1) - Lg * np.cos(th2)
    plega = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi   # noqa: E731
    a1 = np.arctan2(x1, -y1) - v
    a2 = np.arctan2(xg - x1, -(yg - y1)) - v
    return th1 + plega(a1 - th1), th2 + plega(a2 - th2)


def mesura_general(th1, th2, e1: float, e2: float, L1: float, Lg: float, ex: float = 0.0, ey: float = 0.0):
    """mesura_amb_repos_encallat mes un error del centre del cercle (ex, ey), en metres (onzena tanda, 24/10)."""
    if ex == 0.0 and ey == 0.0:
        return mesura_amb_repos_encallat(th1, th2, e1, e2, L1, Lg)
    th1, th2 = np.asarray(th1, dtype=float), np.asarray(th2, dtype=float)
    P = np.array([ex, ey])
    P2r = np.array([L1 * math.sin(e1), -L1 * math.cos(e1)])
    Gr = P2r + Lg * np.array([math.sin(e2), -math.cos(e2)])
    v = math.atan2(Gr[0] - P[0], -(Gr[1] - P[1]))
    u = np.array([math.sin(v), -math.cos(v)])
    n1r = np.array([math.cos(e1), math.sin(e1)])
    creu = lambda a, b: a[0] * b[1] - a[1] * b[0]       # noqa: E731
    d = -creu(P2r - P, u) / creu(n1r, u)
    jx, jy = L1 * np.sin(th1) + d * np.cos(th1), -L1 * np.cos(th1) + d * np.sin(th1)
    xg = L1 * np.sin(th1) + Lg * np.sin(th2)
    yg = -L1 * np.cos(th1) - Lg * np.cos(th2)
    plega = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi   # noqa: E731
    a1 = np.arctan2(jx - P[0], -(jy - P[1])) - v
    a2 = np.arctan2(xg - jx, -(yg - jy)) - v
    return th1 + plega(a1 - th1), th2 + plega(a2 - th2)


def inverteix_general(m1, m2, e1, e2, L1, Lg, ex=0.0, ey=0.0):
    """Els angles de debo que, passats per mesura_general, donen (m1, m2)."""
    t = np.array([m1, m2], dtype=float)
    for _ in range(8):
        a, b = mesura_general(t[0], t[1], e1, e2, L1, Lg, ex, ey)
        t = t - (np.array([float(a), float(b)]) - np.array([m1, m2]))
    return float(t[0]), float(t[1])


def rangs_de(p: Parametres, conf) -> dict:
    """Els rangs del muntatge: conf = (rs, I2, g, L1), on 1 vol dir el rang sencer de prepara_versemblant."""
    frs, fi2, fg, fl = conf
    m = pv.muntatge_de(p)
    return {"rs": m["rs"] if frs == 1 else (p.rs, p.rs + frs * (pv.CARGOLS_LLUNY_M - p.rs)),
            "I2_factor": m["I2_factor"] if fi2 == 1 else (1.0, 1.0 + fi2 * (pv.I2_DEL_PERIODE - 1.0)),
            "g_factor": m["g_factor"] if fg == 1 else (1.0 - fg * (1.0 - pv.G_DEL_BANC), 1.0),
            "L1": m["L1"] if fl == 1 else (p.L1 - fl * pv.L1_REGLE_M, p.L1 + fl * pv.L1_REGLE_M)}


def simula_feix_r(p: Parametres, estat0, nom: str, tg, rangs, vertical=None,
                  amplada_graus: float = AMPLADA_GRAUS) -> tuple[np.ndarray, np.ndarray]:
    """El nominal i els 200 membres. vertical = (a1, a2, sigma de costat, sigma vertical): cada membre sorteja el seu repos
    encallat (uniforme dins de +-a1 i +-a2, en radians) i el seu error del centre (normal, en metres); arrenca dels angles de
    debo que donen la condicio inicial mesurada, i la seva sortida passa per mesura_general."""
    pm = pv.mostra_muntatge(p, rangs, pv.llavor_de(nom), MEMBRES)
    z = np.random.default_rng(pv.llavor_de(nom)).standard_normal((MEMBRES, 2))
    if vertical is not None:
        ev = np.random.default_rng([pv.llavor_de(nom), 24102026]).uniform(-1.0, 1.0, (MEMBRES, 2)) \
            * np.array(vertical[:2])
        ec = (np.random.default_rng([pv.llavor_de(nom), 24102026, 2]).standard_normal((MEMBRES, 2))
              * np.array(vertical[2:4])) if len(vertical) > 2 else np.zeros((MEMBRES, 2))
    sigma = math.radians(amplada_graus)
    A1 = np.empty((MEMBRES + 1, len(tg)))
    A2 = np.empty((MEMBRES + 1, len(tg)))
    _, y = simula(p, estat0, t_eval=tg)
    A1[0], A2[0] = y[0], y[1]
    for i in range(MEMBRES):
        e = np.array(estat0, dtype=float).copy()
        e[0] += sigma * z[i, 0]
        e[1] += sigma * z[i, 1]
        if vertical is not None:
            e[0], e[1] = inverteix_general(e[0], e[1], ev[i, 0], ev[i, 1], p.L1, p.r_gomet, ec[i, 0], ec[i, 1])
        _, y = simula(pm[i], e, t_eval=tg)
        if vertical is not None:
            A1[i + 1], A2[i + 1] = mesura_general(y[0], y[1], ev[i, 0], ev[i, 1], p.L1, p.r_gomet, ec[i, 0], ec[i, 1])
        else:
            A1[i + 1], A2[i + 1] = y[0], y[1]
    return A1, A2


def feix(p: Parametres, estat0, nom: str, grup: str, collapsat: bool = False, t_fins: float = T_FEIX_S):
    """El feix firmat d'una presa: (tg, A1, A2), en temps des de l'amollada. Collapsat (E2): totes les amplades a zero.
    t_fins: fins on (el de la Y, 1,5 s; l'analisi posterior cap_on, 3 s)."""
    tg = np.arange(0.0, t_fins + PAS_FEIX_S / 2, PAS_FEIX_S)
    if collapsat:
        A1, A2 = simula_feix_r(p, estat0, nom, tg, pv.muntatge_de(p, collapsat=True), (0.0, 0.0, 0.0, 0.0), 0.0)
        return tg, A1, A2
    sc = np.array(SIGMA_CENTRE_PX[grup]) * p.L1 / 200.0
    vertical = tuple(math.radians(a) for a in AMPLADA_ENCALLAT_GRAUS) + (float(sc[0]), float(sc[1]))
    A1, A2 = simula_feix_r(p, estat0, nom, tg, rangs_de(p, CARGOL_MESURAT), vertical)
    return tg, A1, A2


def soroll(p: Parametres, nom: str, finestra_mostres: int, soroll_index_graus, lent_graus, lent_rho, fps: float,
           t_fins: float = T_FEIX_S):
    """El soroll del tir per a cada fila del feix (comparar.soroll_dels_membres), amb la llavor de les tandes."""
    ps = comparar.parametres_del_soroll(p, int(finestra_mostres), [float(v) for v in soroll_index_graus],
                                        [float(v) for v in lent_graus], [float(v) for v in lent_rho])
    n1 = int(round(T_SOROLL_S * fps)) + 3
    n = int(round(t_fins * fps)) + 3
    return comparar.soroll_dels_membres(ps, n1, n, llavor("soroll del feix", nom, "S1"))


def y_del_tir(t, th1, th2, zero: float, tg, A1, A2, N, x_graus: float) -> dict:
    """L'horitzo amb la X i la Y d'un tir (comparar.avalua_y amb el soroll dins del feix)."""
    H = comparar.horitzo_feix(tg, A2, x_graus)
    return {"H_s": H, "y": comparar.avalua_y(t, th1, th2, zero, tg, A1, A2, H, limit=T_FEIX_S, soroll_feix=N)}


# ------------------------------------------------------------ la lectura i el pany

def lectura(grup: str, per_tir: dict | None) -> dict:
    """El veredicte d'un grup, llegit en un sol sentit (secció 3e de per-firmar-24-10.md)."""
    if grup not in GRUPS_Y:
        return {"lectura": "no avaluable", "per_que": "els 150 graus no es van poder avaluar a cap tanda"}
    v = comparar.veredicte_grup(per_tir)
    return {**v, "lectura": "falla: no concloent" if v["falla"] else "passa: és un resultat"}


def llegeix_firma(cami: Path = FIRMA) -> dict | None:
    return json.loads(cami.read_text(encoding="utf-8")) if cami.exists() else None


def pany(firma: dict | None, x_graus: float, grups) -> dict:
    """Sense firma, o si l'ordre no demana el que s'ha firmat, s'atura ABANS d'obrir res de dades/net/."""
    if firma is None:
        raise SystemExit(f"Pany tancat: no hi ha {FIRMA.relative_to(ARREL)}. L'experiment 1 compara simulacio i realitat, "
                         "i aixo no es fa abans de firmar X, Y i Z.")
    buits = [k for k in ("X", "Y", "Z") if not firma.get(k)]
    if buits:
        raise SystemExit(f"Pany tancat: la firma no diu {', '.join(buits)}.")
    Y = firma["Y"] if isinstance(firma["Y"], dict) else {}
    xoca = []
    if not isinstance(firma["X"], (int, float)) or float(firma["X"]) != float(x_graus):
        xoca.append(f"la X (firmada {firma['X']!r}, demanada {x_graus:g})")
    for clau, valor in (("c", comparar.C_GRUP), ("fora_s", comparar.CONFIRMA_S), ("finestra_s", comparar.FINESTRA_Y_S)):
        if Y.get(clau) is None or float(Y[clau]) != float(valor):
            xoca.append(f"{clau} (firmada {Y.get(clau)!r}, la regla {valor:g})")
    for g in grups:
        if g not in Y.get("en_un_sol_sentit", []):
            xoca.append(f"el grup {g} (no es a en_un_sol_sentit)")
    if xoca:
        raise SystemExit("Pany tancat: l'ordre no demana el que s'ha firmat: " + "; ".join(xoca) + ".")
    return firma


# ------------------------------------------------------------ una presa real (nomes darrere del pany)

def preses_del_grup(grup: str) -> list[str]:
    noms = json.loads(NUMEROS.read_text(encoding="utf-8"))["B_condicions_inicials"]["per_presa"]
    fora = sorted((n for n in noms if n.split("_")[0] == grup), key=lambda n: int(n.split("_")[1]))
    if len(fora) != N_GRUP[grup]:
        raise SystemExit(f"Al grup {grup} hi ha {len(fora)} preses, i la regla es va provar amb {N_GRUP[grup]}.")
    return fora


def presa_corregida(nom: str, p: Parametres) -> dict:
    """La presa real corregida, les condicions inicials del feix (les dels clics) i el zero afinat. Nomes despres del pany."""
    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{nom}.csv")
    th1, th2, info = comparar.corregeix_presa(taula, meta, nom)
    t = taula["t"].to_numpy(dtype=float)
    r = meta["amollada"]
    ta = float(r["amollada_s"])
    ini = (t >= max(0.0, ta - 0.4)) & (t <= ta - 0.02)
    estat0 = np.array([float(np.mean(th1[ini])), float(np.mean(th2[ini])), 0.0, 0.0])
    # Les condicions inicials del feix han de ser les dels clics (la K3 d'eines/correccio_amb_clics.py).
    clics = json.loads(comparar.CLICS.read_text(encoding="utf-8"))["preses"].get(nom)
    if not info["clics"] or clics is None:
        raise SystemExit(f"{nom}: no te la correccio amb clics, i als 45 i 90 graus la regla la demana.")
    dif = max(abs(estat0[k] - math.radians(clics["amollada_amb_clics_graus"][k])) for k in range(2))
    if dif >= 1e-12:
        raise SystemExit(f"{nom}: les condicions inicials no son les dels clics ({dif:.1e} rad).")
    taula_c = taula.copy()
    taula_c["theta1"], taula_c["theta2"] = th1, th2
    af = afina_amollada(taula_c, r, p, estat0, finestra_s=comparar.FINESTRA_Y_S)
    zero = float(af["amollada_afinada_s"]) if af["ok"] else ta
    return {"t": t, "th1": th1, "th2": th2, "meta": meta, "ta": ta, "zero": zero, "afinat": bool(af["ok"]), "estat0": estat0}


def soroll_de_la_presa(p: Parametres, nom: str, meta: dict, t_fins: float = T_FEIX_S):
    m = json.loads(SOROLL_MOV.read_text(encoding="utf-8"))["per_presa"][nom]
    return soroll(p, nom, meta["savgol"]["finestra_mostres"], meta["geometria"]["rigidesa"]["soroll_angular_graus"],
                  m["total"]["components_graus"][:2], m["lag1"], float(meta["fps"]), t_fins)


def avalua_presa(feina) -> dict:
    """La Y d'una presa real. Nomes es crida des de l'ordre y, despres del pany."""
    nom, x_graus = feina
    grup = nom.split("_")[0]
    p = Parametres.des_de_json(PARAMETRES)
    q = presa_corregida(nom, p)
    tg, A1, A2 = feix(p, q["estat0"], nom, grup)
    N = soroll_de_la_presa(p, nom, q["meta"])
    fora = y_del_tir(q["t"], q["th1"], q["th2"], q["zero"], tg, A1, A2, N, x_graus)
    return {"nom": nom, "grup": grup,
            "condicions_inicials_graus": [math.degrees(q["estat0"][0]), math.degrees(q["estat0"][1])],
            "zero_video_s": q["ta"], "zero_s": q["zero"], "zero_afinat": q["afinat"], **fora}


# ------------------------------------------------------------ l'analisi posterior: cap on van els tirs (NO firmada)

T_CAP_ON_S = 3.0
D_CAP_ON_GRAUS = (10.0, 5.0, 20.0)       # la principal, 10; 5 i 20, sensibilitat (fixades abans de calcular)
BOOTSTRAP_CAP_ON = 10000
LLAVOR_CAP_ON = 27102026


def temps_separacio(ts, d, D: float, confirma_s: float = comparar.CONFIRMA_S) -> np.ndarray:
    """Per a cada fila de d (separacions sobre ts), el primer instant en que passa de D i s'hi queda confirma_s seguits (com
    avalua_y); inf si no passa mai."""
    d = np.atleast_2d(np.asarray(d, dtype=float))
    fora = d > D
    c = np.concatenate([np.zeros((len(d), 1), dtype=int), np.cumsum(fora, axis=1)], axis=1)
    fi = np.searchsorted(ts, ts + confirma_s, side="right")
    i = np.arange(len(ts))
    ok = fora & ((c[:, fi] - c[:, i]) == (fi - i))
    t = np.full(len(d), np.inf)
    hi = ok.any(axis=1)
    t[hi] = np.asarray(ts)[ok.argmax(axis=1)[hi]]
    return t


def percentil(t_tir: float, t_membres) -> float:
    """La fraccio de membres que se separen abans que el tir; els empats (tambe els que no se separen mai), la meitat."""
    t_membres = np.asarray(t_membres)
    return float((np.sum(t_membres < t_tir) + 0.5 * np.sum(t_membres == t_tir)) / len(t_membres))


def separacions(ts, tg, A1, A2, N, th1=None, th2=None):
    """d = max(|dtheta1|, |dtheta2|) respecte del nominal (la fila 0, sense soroll): dels membres (amb el seu soroll) i, si es
    dona, del tir. Torna (d_membres, d_tir)."""
    nom1, nom2 = np.interp(ts, tg, A1[0]), np.interp(ts, tg, A2[0])
    M1 = np.array([np.interp(ts, tg, A1[i]) for i in range(1, len(A1))]) + N[0][1:, :len(ts)]
    M2 = np.array([np.interp(ts, tg, A2[i]) for i in range(1, len(A2))]) + N[1][1:, :len(ts)]
    dm = np.maximum(np.abs(M1 - nom1), np.abs(M2 - nom2))
    dt = None if th1 is None else np.maximum(np.abs(np.asarray(th1) - nom1), np.abs(np.asarray(th2) - nom2))
    return dm, dt


def cap_on_presa(nom: str) -> dict:
    """L'analisi posterior d'una presa real. Nomes es crida des de l'ordre cap_on, despres del pany."""
    grup = nom.split("_")[0]
    p = Parametres.des_de_json(PARAMETRES)
    q = presa_corregida(nom, p)
    tg, A1, A2 = feix(p, q["estat0"], nom, grup, t_fins=T_CAP_ON_S)
    N = soroll_de_la_presa(p, nom, q["meta"], T_CAP_ON_S)
    tau = q["t"] - q["zero"]
    sel = (tau >= 0.0) & (tau <= T_CAP_ON_S)
    ts = tau[sel]
    dm, dt = separacions(ts, tg, A1, A2, N, q["th1"][sel], q["th2"][sel])
    fora = {"nom": nom, "grup": grup, "per_D": {}}
    for D in D_CAP_ON_GRAUS:
        tm = temps_separacio(ts, dm, math.radians(D))
        tr = float(temps_separacio(ts, dt, math.radians(D))[0])
        fora["per_D"][f"{D:g}"] = {"t_tir_s": tr, "percentil": percentil(tr, tm),
                                   "t_membres_mediana_s": float(np.median(tm)),
                                   "membres_que_no_se_separen": int(np.sum(~np.isfinite(tm)))}
    return fora


def resum_cap_on(percentils) -> dict:
    v = np.asarray(percentils, dtype=float)
    rng = np.random.default_rng(LLAVOR_CAP_ON)
    med = np.median(v[rng.integers(0, len(v), (BOOTSTRAP_CAP_ON, len(v)))], axis=1)
    lo, hi = (float(x) for x in np.percentile(med, [2.5, 97.5]))
    lectura = ("els tirs reals se separen abans que el feix" if hi < 0.5 else
               "els tirs reals se separen despres que el feix" if lo > 0.5 else
               "compatible amb el feix, amb molt poca precisio")
    return {"n": len(v), "mediana": float(np.median(v)), "interval_95": [lo, hi],
            "per_sota_de_0_1": int(np.sum(v < 0.1)), "lectura": lectura}


def ordre_cap_on(firma: dict | None, json_sortida: Path | None, processos: int = 1) -> int:
    x = float(firma["X"]) if firma and isinstance(firma.get("X"), (int, float)) else math.nan
    pany(firma, x, GRUPS_Y)
    noms = [n for g in GRUPS_Y for n in preses_del_grup(g)]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            res = pool.map(cap_on_presa, noms)
    else:
        res = [cap_on_presa(n) for n in noms]
    fora = {"_que_es": "ANALISI POSTERIOR, NO FIRMADA (documentacio/pla-d4-experiment1.md, «L'analisi posterior»): cap on "
                       "van els tirs, amb una tolerancia gran. Es va pensar despres de veure que la Y firmada no es concloent.",
            "t_fins_s": T_CAP_ON_S, "D_graus": D_CAP_ON_GRAUS, "per_grup": {}}
    for g in GRUPS_Y:
        rg = [r for r in res if r["grup"] == g]
        fora["per_grup"][g] = {"per_tir": {r["nom"]: r["per_D"] for r in rg},
                               "per_D": {f"{D:g}": resum_cap_on([r["per_D"][f"{D:g}"]["percentil"] for r in rg])
                                         for D in D_CAP_ON_GRAUS}}
        for D in D_CAP_ON_GRAUS:
            s = fora["per_grup"][g]["per_D"][f"{D:g}"]
            print(f"{g} graus, D = {D:>4g}: mediana del percentil {s['mediana']:.2f} (95 %: {s['interval_95'][0]:.2f}-"
                  f"{s['interval_95'][1]:.2f}), {s['per_sota_de_0_1']} de {s['n']} per sota de 0,1 -> {s['lectura']}")
    if json_sortida is not None:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(pv._json_net(fora), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"-> {json_sortida}")
    return 0


def ordre_y(x_graus: float, grups, firma: dict | None, json_sortida: Path | None, processos: int = 1) -> int:
    pany(firma, x_graus, grups)
    feines = [(n, x_graus) for g in grups for n in preses_del_grup(g)]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            res = pool.map(avalua_presa, feines)
    else:
        res = [avalua_presa(f) for f in feines]
    fora = {"_que_es": "L'experiment 1: la Y de la dotzena, llegida en un sol sentit (eines/experiment1.py).",
            "firma": {"data": firma.get("data"), "X": firma["X"], "Y": firma["Y"]}, "x_graus": x_graus,
            "per_grup": {}, "150": lectura("150", None)}
    for g in grups:
        per_tir = {r["nom"]: r["y"] for r in res if r["grup"] == g}
        fora["per_grup"][g] = {"lectura": lectura(g, per_tir), "per_tir": {r["nom"]: r for r in res if r["grup"] == g}}
        L = fora["per_grup"][g]["lectura"]
        print(f"{g} graus: {L['lectura'].upper()} ({len(L['fallen'])} tirs fora de {L['avaluables']} avaluables; "
              f"c = {L['c']}; no avaluables: {L['no_avaluables'] or 'cap'})")
    print("150 graus: no avaluables")
    if json_sortida is not None:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(pv._json_net(fora), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"-> {json_sortida}")
    return 0


# ------------------------------------------------------------ el salt a l'amollada (mirat despres del veredicte)

FINESTRA_SALT_S = (-0.015, -0.005)       # respecte de l'amollada del video: just abans, on posa_clics no toca la serie


def salt_presa(nom: str) -> dict:
    """La serie corregida just abans d'amollar menys les condicions inicials del feix (les dels clics): el salt de la creu
    (pla-d4-experiment1.md, «Mirat despres del veredicte»). Nomes despres del pany."""
    p = Parametres.des_de_json(PARAMETRES)
    q = presa_corregida(nom, p)
    w = (q["t"] >= q["ta"] + FINESTRA_SALT_S[0]) & (q["t"] <= q["ta"] + FINESTRA_SALT_S[1])
    s = [math.degrees(float(np.mean(th[w])) - float(q["estat0"][k])) for k, th in enumerate((q["th1"], q["th2"]))]
    return {"nom": nom, "grup": nom.split("_")[0], "salt_graus": s, "mostres": int(np.sum(w))}


def ordre_salt(firma: dict | None, json_sortida: Path | None, processos: int = 1) -> int:
    x = float(firma["X"]) if firma and isinstance(firma.get("X"), (int, float)) else math.nan
    pany(firma, x, GRUPS_Y)
    noms = [n for g in GRUPS_Y for n in preses_del_grup(g)]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            res = pool.map(salt_presa, noms)
    else:
        res = [salt_presa(n) for n in noms]
    fora = {"_que_es": "El salt de la creu a l'amollada (eines/experiment1.py salt; pla-d4-experiment1.md, «Mirat despres del "
                       "veredicte»; documentacio/pla-figura-i-eines-experiment1.md, pas 3.3): la serie corregida a [ta - 15; "
                       "ta - 5] ms menys les condicions inicials del feix. Mirat despres del veredicte; no en canvia cap.",
            "finestra_s": FINESTRA_SALT_S, "per_grup": {}}
    for g in GRUPS_Y:
        rg = [r for r in res if r["grup"] == g]
        rms = [float(np.sqrt(np.mean([r["salt_graus"][k] ** 2 for r in rg]))) for k in range(2)]
        fora["per_grup"][g] = {"rms_graus": rms, "per_tir": {r["nom"]: r for r in rg}}
        print(f"{g} graus: rms theta1 {rms[0]:.2f}, theta2 {rms[1]:.2f} graus")
        for r in rg:
            print(f"   {r['nom']:6} theta1 {r['salt_graus'][0]:+.2f}  theta2 {r['salt_graus'][1]:+.2f}  ({r['mostres']} mostres)")
    if json_sortida is not None:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(pv._json_net(fora), indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"-> {json_sortida}")
    return 0


# ------------------------------------------------------------ les figures (documentacio/pla-figura-i-eines-experiment1.md)

Y_FIRMADA = ARREL / "dades" / "experiment1" / "y_2026-10-27.json"
T_FIGURA_S = 1.0
YLIM_FIGURA_GRAUS = 10.0


def tir_de_la_mediana(per_tir: dict) -> str:
    """La regla de la figura 1 (fixada abans de dibuixar): el tir de la mediana de l'instant de sortida del grup (un que no surt,
    o no avaluable, compta com a infinit); amb un nombre parell de tirs, el de sota. Els empats, per numero de presa."""
    t = {n: (r["y"]["t_sortida_s"] if r["y"].get("falla") else math.inf) for n, r in per_tir.items()}
    ordre = sorted(t, key=lambda n: (t[n], int(n.split("_")[1])))
    return ordre[(len(ordre) - 1) // 2]


def franja_del_feix(q: dict, tg, A1, A2, N, t_fins: float) -> dict:
    """El tir, el nominal (la fila 0) i el minim i el maxim dels membres amb el soroll del tir, el mateix que mira
    comparar.avalua_y, des del zero i fins a t_fins. En radians."""
    tau = q["t"] - q["zero"]
    sel = (tau >= 0.0) & (tau <= t_fins)
    ts = tau[sel]
    fora = {"t_s": ts}
    for j, (A, real) in enumerate(((A1, q["th1"]), (A2, q["th2"])), start=1):
        M = np.array([np.interp(ts, tg, A[i]) for i in range(len(A))]) + N[j - 1][:, :len(ts)]
        fora[f"real{j}"], fora[f"nominal{j}"] = np.asarray(real)[sel], np.interp(ts, tg, A[0])
        fora[f"min{j}"], fora[f"max{j}"] = M.min(axis=0), M.max(axis=0)
    return fora


def dibuixa_feix_tir(franges: dict, ys: dict, per_tir: dict):
    """Figura 1: un tir per grup (files) i theta1, theta2 (columnes), l'angle menys el nominal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pendol import estil
    estil.aplica()
    fig, eixos = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey=True)
    for fila, g in enumerate(GRUPS_Y):
        nom, f, r = franges[g]["nom"], franges[g], per_tir[g]
        ts = f["t_s"]
        for col, j in enumerate((1, 2)):
            ax = eixos[fila, col]
            nom_j = f[f"nominal{j}"]
            ax.fill_between(ts, np.degrees(f[f"min{j}"] - nom_j), np.degrees(f[f"max{j}"] - nom_j), color=estil.FEIX, lw=0,
                            label="feix: 200 simulacions, amb el soroll del tir")
            ax.plot(ts, np.degrees(f[f"real{j}"] - nom_j), color=estil.REAL, ls=estil.TRAC_REAL, lw=1.3, label="tir real")
            ax.axvline(r["H_s"], color=estil.TINTA_2, ls=(0, (4, 3)), lw=0.9, label="horitzó del feix (X = 5°)")
            if ys[g].get("falla") and ys[g]["angle"] in ("tots dos", f"theta{j}"):
                k = int(np.argmin(np.abs(ts - ys[g]["t_sortida_s"])))
                ax.plot(ts[k], np.degrees(f[f"real{j}"][k] - nom_j[k]), "o", ms=5.5, color=estil.REAL, mec=estil.TINTA,
                        mew=0.7, zorder=5, label="on surt del feix")
            ax.axhline(0.0, color=estil.EIX, lw=0.6, zorder=0)
            ax.set_ylim(-YLIM_FIGURA_GRAUS, YLIM_FIGURA_GRAUS)
            ax.set_xlim(0.0, T_FIGURA_S)
            ax.set_title(f"{g}° · θ{j}  (tir {nom})", loc="left")
            if col == 0:
                ax.set_ylabel("angle − simulació nominal (°)")
            if fila == 1:
                ax.set_xlabel("temps des del zero (s)")
    eixos[0, 0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def dibuixa_sortides(y: dict):
    """Figura 2: tots els tirs, una fila per tir, del zero fins on surt (o fins a l'horitzo, si no surt)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pendol import estil
    estil.aplica()
    fig, eixos = plt.subplots(1, 2, figsize=(9.6, 4.6), sharex=True)
    for ax, g in zip(eixos, GRUPS_Y):
        per_tir = y["per_grup"][g]["per_tir"]
        noms = sorted(per_tir, key=lambda n: int(n.split("_")[1]))
        for fila, nom in enumerate(noms):
            r = per_tir[nom]
            ax.plot([r["H_s"]] * 2, [fila - 0.32, fila + 0.32], color=estil.TINTA_2, lw=1.2, zorder=3)
            if r["y"].get("falla"):
                ts = r["y"]["t_sortida_s"]
                ax.plot([0.0, ts], [fila, fila], color=estil.REAL, lw=3.0, solid_capstyle="butt")
                ax.plot(ts, fila, "o", ms=5.5, color=estil.REAL, mec=estil.TINTA, mew=0.7, zorder=4)
                txt = f"{ts * 1000:.1f} ms" if ts < 0.01 else f"{ts * 1000:.0f} ms" if ts < 0.1 else f"{ts:.2f} s"
                ax.text(ts + 0.012, fila, txt.replace(".", ","), va="center", fontsize=7.5, color=estil.TINTA_2)
            else:
                ax.plot([0.0, r["H_s"]], [fila, fila], color=estil.REAL, lw=3.0, alpha=0.35, solid_capstyle="butt")
                ax.text(r["H_s"] + 0.012, fila, "no surt", va="center", fontsize=7.5, color=estil.TINTA_2)
        L = y["per_grup"][g]["lectura"]
        ax.set_yticks(range(len(noms)), noms)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 0.7)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{g}°: {len(L['fallen'])} de {L['avaluables']} surten abans de l'horitzó", loc="left")
        ax.set_xlabel("temps des del zero (s)")
    eixos[0].plot([], [], color=estil.REAL, lw=3.0, label="dins del feix")
    eixos[0].plot([], [], "o", ms=5.5, color=estil.REAL, mec=estil.TINTA, mew=0.7, label="surt del feix")
    eixos[0].plot([], [], color=estil.TINTA_2, lw=1.2, label="horitzó del feix")
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.legend(*eixos[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8.5)
    return fig


def ordre_figura(firma: dict | None, cami_y: Path = Y_FIRMADA, directori: Path = ARREL / "figures") -> int:
    """Les figures 1 i 2 de l'experiment 1, amb la Y firmada del 27/10. Darrere del pany. FE1: la Y dels dos tirs de la figura 1,
    tornada a calcular, ha de ser identica a la del JSON; J: de dades/net/ nomes s'obren aquests dos tirs."""
    import hashlib
    from pendol import estil
    y = json.loads(cami_y.read_text(encoding="utf-8"))
    pany(firma, float(y["x_graus"]), GRUPS_Y)
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    p = Parametres.des_de_json(PARAMETRES)
    franges, ys, per_tir, problemes = {}, {}, {}, []
    for g in GRUPS_Y:
        nom = tir_de_la_mediana(y["per_grup"][g]["per_tir"])
        r = y["per_grup"][g]["per_tir"][nom]
        q = presa_corregida(nom, p)
        tg, A1, A2 = feix(p, q["estat0"], nom, g)
        N = soroll_de_la_presa(p, nom, q["meta"])
        nou = y_del_tir(q["t"], q["th1"], q["th2"], q["zero"], tg, A1, A2, N, float(y["x_graus"]))
        nou = json.loads(json.dumps(pv._json_net(nou)))
        iguals = nou["H_s"] == r["H_s"] and nou["y"] == r["y"] and q["zero"] == r["zero_s"]
        print(f"   {'ok ' if iguals else 'NO '} FE1 {g} graus, el tir {nom}: horitzo {nou['H_s']:.4f} s, surt "
              f"{nou['y'].get('t_sortida_s')} ({nou['y'].get('angle')}); identic al JSON: {iguals}")
        if not iguals:
            problemes.append(f"FE1 {nom}")
        franges[g] = {"nom": nom, **franja_del_feix(q, tg, A1, A2, N, T_FIGURA_S)}
        ys[g], per_tir[g] = nou["y"], r
    mirant[0] = False
    net = sorted({Path(o).name for o in oberts if "net" in Path(o).parts and o.endswith(".csv")})
    esperats = sorted(f"{franges[g]['nom']}.csv" for g in GRUPS_Y)
    print(f"   {'ok ' if net == esperats else 'NO '} J de dades/net/ nomes s'obren {esperats}: {net}")
    if net != esperats:
        problemes.append("J")
    if problemes:
        raise SystemExit(f"FALLA {problemes}: no es dibuixa res.")
    estil.desa(dibuixa_feix_tir(franges, ys, per_tir), "experiment1_feix_tir", directori)
    estil.desa(dibuixa_sortides(y), "experiment1_sortides", directori)
    metr = {"_que_es": "Les figures 1 i 2 de l'experiment 1 (eines/experiment1.py figura; "
                       "documentacio/pla-figura-i-eines-experiment1.md). Amb el Tracker: el resultat firmat.",
            "y": {"fitxer": cami_y.relative_to(ARREL).as_posix(), "sha256": hashlib.sha256(cami_y.read_bytes()).hexdigest()},
            "regla_del_tir": "la mediana de l'instant de sortida del grup; amb un nombre parell, el de sota",
            "figura_1": {g: {"tir": franges[g]["nom"], "H_s": per_tir[g]["H_s"], "y": ys[g]} for g in GRUPS_Y},
            "figura_2": {g: {"lectura": y["per_grup"][g]["lectura"]["lectura"],
                             "surten": len(y["per_grup"][g]["lectura"]["fallen"]),
                             "avaluables": y["per_grup"][g]["lectura"]["avaluables"]} for g in GRUPS_Y}}
    cami = directori / "experiment1_figures.json"
    cami.write_text(json.dumps(pv._json_net(metr), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {cami}")
    return 0


# ------------------------------------------------------------ l'autoprova (E2-E6 i J; la E1, a l'esborrany)

def autoprova() -> int:
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova d'eines/experiment1.py (criteris de documentacio/pla-d4-experiment1.md; la E1, a "
          "esborranys/_comprova_experiment1.py)")
    p = Parametres.des_de_json(PARAMETRES)
    estat0 = np.array([math.radians(91.0), math.radians(90.0), 0.0, 0.0])

    # E2: amb les amplades a zero, 201 files identiques i cap horitzo; amb les firmades, el feix s'eixampla.
    tg, A1, A2 = feix(p, estat0, "90_1", "90", collapsat=True)
    iguals = bool(all(np.array_equal(A1[i], A1[0]) and np.array_equal(A2[i], A2[0]) for i in range(len(A1))))
    H0 = comparar.horitzo_feix(tg, A2, 5.0)
    tg, B1, B2 = feix(p, estat0, "90_1", "90")
    w = B2.max(axis=0) - B2.min(axis=0)
    mira(iguals and not math.isfinite(H0) and w[0] > 0.0 and w[-1] > w[0],
         f"E2 amplades a zero: {len(A1)} files identiques ({iguals}), horitzo {H0}; amb les firmades, amplada de theta2 "
         f"{math.degrees(w[0]):.3f} graus a t = 0 (les condicions inicials, 0,1 graus) i {math.degrees(w[-1]):.1f} a "
         f"{tg[-1]:g} s")

    # E3: la lectura en un sol sentit.
    tir = lambda f: {"avaluable": True, "falla": f}   # noqa: E731
    l0 = lectura("90", {f"t{i}": tir(False) for i in range(11)})
    l1 = lectura("90", {f"t{i}": tir(i == 0) for i in range(11)})
    l2 = lectura("45", {f"t{i}": tir(i < 2) for i in range(12)})
    l3 = lectura("150", None)
    mira(l0["lectura"] == "passa: és un resultat" and l1["lectura"] == "passa: és un resultat"
         and l2["lectura"] == "falla: no concloent" and l3["lectura"] == "no avaluable",
         f"E3 la lectura: cap fora -> {l0['lectura']!r}; 1 fora -> {l1['lectura']!r}; 2 fora -> {l2['lectura']!r}; "
         f"150 -> {l3['lectura']!r}")

    # E4 i E5: el pany, amb firmes en memoria. Cap ha d'arribar a obrir res de dades/net/. Barrera: mentrestant, la funcio
    # que obriria les preses s'atura sempre, o sigui que ni amb un error del pany no s'obriria cap presa.
    bona = {"data": "prova", "X": 5.0, "Z": {"que": "prova"},
            "Y": {"c": 2, "fora_s": 0.2, "finestra_s": 0.085, "en_un_sol_sentit": ["45", "90"]}}

    class Passaria(Exception):
        pass

    def no_obris(*_a, **_k):
        raise Passaria()

    g = globals()
    desa = g["preses_del_grup"]
    g["preses_del_grup"] = no_obris
    n0 = len(oberts)
    casos = {"sense firma": (None, 5.0, ["90"]),
             "una altra X": (bona, 3.0, ["90"]),
             "una altra c": (dict(bona, Y=dict(bona["Y"], c=3)), 5.0, ["90"]),
             "un altre fora": (dict(bona, Y=dict(bona["Y"], fora_s=0.3)), 5.0, ["90"]),
             "una altra finestra": (dict(bona, Y=dict(bona["Y"], finestra_s=0.25)), 5.0, ["90"]),
             "un grup que no hi es": (dict(bona, Y=dict(bona["Y"], en_un_sol_sentit=["90"])), 5.0, ["45"]),
             "la Z buida": (dict(bona, Z=None), 5.0, ["90"])}
    atura = {}
    try:
        for k, (f, x, gr) in casos.items():
            try:
                ordre_y(x, gr, f, None)
                atura[k] = False
            except SystemExit:
                atura[k] = True
            except Passaria:
                atura[k] = False
    finally:
        g["preses_del_grup"] = desa
    net = [o for o in oberts[n0:] if "net" in Path(o).parts]
    mira(atura["sense firma"] and not net, f"E4 sense firma, l'ordre y s'atura ({atura['sense firma']}) i no obre res de "
                                           f"dades/net/ ({len(net)} fitxers)")
    mira(all(atura[k] for k in casos if k != "sense firma") and not net,
         f"E5 amb una firma que no diu el mateix, s'atura: {', '.join(k for k in casos if k != 'sense firma' and atura[k])}"
         f"; dades/net/ obert: {len(net)}")
    try:
        pany(bona, 5.0, ["45", "90"])
        bona_passa = True
    except SystemExit:
        bona_passa = False
    mira(bona_passa, "E5 (control) amb la firma que diu el mateix, el pany deixa passar")

    # E6: reproduible.
    tg, C1, C2 = feix(p, estat0, "90_1", "90")
    ps = (13, [0.3, 0.6], [0.5, 0.7], [0.97, 0.98], 239.9)
    N1, N2 = soroll(p, "90_1", *ps), soroll(p, "90_1", *ps)
    mira(np.array_equal(B1, C1) and np.array_equal(B2, C2) and np.array_equal(N1[0], N2[0])
         and np.array_equal(N1[1], N2[1]), "E6 dues vegades el feix i el soroll d'una presa: el mateix bit a bit")

    # P0 (l'analisi posterior): el temps de separacio sobre series fabricades.
    ts0 = np.arange(0.0, 3.0, 1.0 / 240.0)
    D0 = math.radians(10.0)
    a = np.where(ts0 >= 1.0, 2 * D0, 0.0)
    b = np.where(((ts0 >= 0.5) & (ts0 < 0.6)) | (ts0 >= 2.0), 2 * D0, 0.0)
    tA, tB, tC = temps_separacio(ts0, np.vstack([a, b, np.zeros_like(ts0)]), D0)
    mira(tA == ts0[np.argmax(ts0 >= 1.0)] and tB == ts0[np.argmax(ts0 >= 2.0)] and not math.isfinite(tC),
         f"P0 el temps de separacio: passa a 1,0 s -> {tA:.4f}; 0,1 s per sobre i despres des de 2,0 s -> {tB:.4f}; "
         f"mai -> {tC}")

    # P1: el percentil, amb els membres d'un feix de prova (90 graus, D = 10) presos un a un com a tir contra els altres.
    tg3, F1, F2 = feix(p, estat0, "90_1", "90", t_fins=T_CAP_ON_S)
    Np = soroll(p, "90_1", *ps[:4], ps[4], T_CAP_ON_S)
    ts1 = np.arange(0.0, T_CAP_ON_S, 1.0 / ps[4])
    dm, _ = separacions(ts1, tg3, F1, F2, Np)
    tm = temps_separacio(ts1, dm, math.radians(10.0))
    qs = np.array([percentil(tm[k], np.delete(tm, k)) for k in range(len(tm))])
    med, baix = float(np.median(qs)), float(np.mean(qs < 0.1))
    mira(0.45 <= med <= 0.55 and 0.05 <= baix <= 0.15,
         f"P1 el percentil dels membres contra els altres: mediana {med:.3f} (0,45-0,55), per sota de 0,1 {baix:.3f} "
         f"(0,05-0,15); {int(np.sum(~np.isfinite(tm)))} de {len(tm)} no se separen en {T_CAP_ON_S:g} s")

    # FE2 (pla-figura-i-eines-experiment1.md): la regla del tir de la figura 1, amb resultats fabricats.
    fab = lambda ts: {f"90_{k + 1}": {"y": ({"avaluable": True, "falla": True, "t_sortida_s": t} if math.isfinite(t)   # noqa: E731
                                           else {"avaluable": True, "falla": False})} for k, t in enumerate(ts)}
    m4 = tir_de_la_mediana(fab([0.3, math.inf, 0.1, 0.2]))
    m5 = tir_de_la_mediana(fab([0.3, math.inf, 0.1, 0.2, 0.05]))
    m3 = tir_de_la_mediana(fab([0.2, 0.2, 0.2]))
    # El 5 esperat era 90_3 a la primera versio: un error de Claude en escriure la prova (ordenats, 90_5, 90_3, 90_4, 90_1 i
    # 90_2: la mediana es la 90_4). Corregit el 29/10 i declarat a pla-figura-i-eines-experiment1.md.
    mira(m4 == "90_4" and m5 == "90_4" and m3 == "90_2",
         f"FE2 la regla de la mediana: 4 tirs (0,3 / no surt / 0,1 / 0,2) -> {m4} (90_4); 5 (i 0,05) -> {m5} (90_4); "
         f"3 empatats -> {m3} (90_2)")

    mirant[0] = False
    dolents = [o for o in oberts if ("net" in Path(o).parts and o.endswith(".csv")) or "sim" in Path(o).parts]
    mira(not dolents, f"J no obre cap CSV de dades/net/ ni res de dades/sim/: {dolents or 'cap'}")
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    sub = ap.add_subparsers(dest="ordre")
    ay = sub.add_parser("y", help="la Y de la dotzena, en un sol sentit, sobre les preses reals (darrere del pany)")
    ay.add_argument("--x", type=float, required=True, help="la X, en graus: ha de ser la firmada")
    ay.add_argument("--grups", nargs="+", default=list(GRUPS_Y), choices=GRUPS_Y)
    ay.add_argument("--json", type=Path, default=None)
    ay.add_argument("--processos", type=int, default=1)
    ac = sub.add_parser("cap_on", help="ANALISI POSTERIOR, NO FIRMADA: cap on van els tirs, amb una tolerancia gran")
    ac.add_argument("--json", type=Path, default=None)
    ac.add_argument("--processos", type=int, default=1)
    sub.add_parser("figura", help="les figures 1 i 2 de l'experiment 1, amb la Y firmada (darrere del pany)")
    asl = sub.add_parser("salt", help="mirat despres del veredicte: el salt de la creu a l'amollada (darrere del pany)")
    asl.add_argument("--json", type=Path, default=None)
    asl.add_argument("--processos", type=int, default=1)
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.ordre == "y":
        return ordre_y(args.x, args.grups, llegeix_firma(), args.json, min(args.processos, 6))
    if args.ordre == "cap_on":
        return ordre_cap_on(llegeix_firma(), args.json, min(args.processos, 6))
    if args.ordre == "figura":
        return ordre_figura(llegeix_firma())
    if args.ordre == "salt":
        return ordre_salt(llegeix_firma(), args.json, min(args.processos, 6))
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
