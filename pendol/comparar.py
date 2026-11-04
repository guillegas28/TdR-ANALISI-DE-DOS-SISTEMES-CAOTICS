"""Compara la trajectoria real (Tracker) amb la simulacio.

Idea central del treball
------------------------
Comparar UNA simulacio amb UN tir real punt a punt esta condemnat al fracas:
passats uns segons ja no s'assemblen. Aixo NO vol dir que el simulador
estigui malament: es exactament el que prediu la teoria del caos, perque no
coneixes l'angle inicial amb precisio infinita.

La comparacio honesta es contra un FEIX de simulacions generades dins la
teva incertesa de mesura (--feix). La pregunta deixa de ser "coincideixen?"
i passa a ser "durant quant de temps la realitat es queda dins del feix?".
Aquest temps ES l'horitzo de predictibilitat del teu pendol, i es un
resultat quantitatiu i propi.

La Y (la prova del model, 23/10/2026)
-------------------------------------
avalua_y(), horitzo_feix(), soroll_dels_membres() i veredicte_grup() son la
regla confirmada a la setena tanda (documentacio/pla-conflicte-y-afinament.md),
portada bit a bit d'esborranys/_conflicte_y.py. Encara NO esta firmada. El
feix no es fa aqui: avalua_y el rep fet. Pla: documentacio/pla-firma-i-y.md.

EL PANY. Comparar una presa real amb la simulacio no es fa abans de firmar X, Y
i Z: sense dades/versemblant/firma.json, la linia d'ordres s'atura amb qualsevol
--real abans d'obrir cap fitxer.

Us:
    .venv\\Scripts\\python -m pendol.comparar --real dades/net/tir01.csv \\
        --feix dades/sim/feix_120graus/ --nom tir01
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter, savgol_coeffs, savgol_filter

from pendol import dades, estil

ARREL = Path(__file__).resolve().parent.parent
FIRMA = ARREL / "dades" / "versemblant" / "firma.json"

# La regla de la Y confirmada el 23/10 (setena tanda). NO FIRMADA.
X_GRAUS = 3.0          # l'amplada del feix (max - min de theta2) que fa l'horitzo
CONFIRMA_S = 0.2       # fora del feix 0,2 s seguits = n'ha sortit (26/09)
C_GRUP = 2             # el grup falla si en surten c tirs o mes
FINESTRA_Y_S = 0.085   # la finestra de l'afinament de l'amollada amb que es va confirmar
FILES_FEIX = 201       # el nominal i 200 membres

# Viu a pendol/mesures.py, que es el modul prim que poden importar les eines
# sense arrossegar-hi matplotlib. Es continua podent escriure
# comparar.dif_angular: aixi no hi ha dues versions de la mateixa cosa.
from pendol.mesures import dif_angular  # noqa: F401


def troba_desfasament(t_r, th_r, t_s, th_s, finestra=2.0, maxim=1.0, pas=0.002):
    """Cerca el desfasament temporal que millor encaixa l'inici de les dues corbes.

    Sempre hi ha uns fotogrames de diferencia entre el moment en que
    engegues el video i el t=0 de la simulacio. S'ajusta minimitzant l'error
    quadratic NOMES als primers segons, quan encara no ha entrat el caos.
    """
    dins = t_r <= t_r[0] + finestra
    millor_s, millor_err = 0.0, np.inf
    for s in np.arange(-maxim, maxim + pas / 2, pas):
        v = np.interp(t_r[dins] + s, t_s, th_s, left=np.nan, right=np.nan)
        ok = ~np.isnan(v)
        if ok.sum() < 0.8 * dins.sum():
            continue
        err = np.sqrt(np.mean((v[ok] - th_r[dins][ok]) ** 2))
        if err < millor_err:
            millor_s, millor_err = float(s), float(err)
    return millor_s, millor_err


# --------------------------------------------------------------- el pany

def pany(firma=FIRMA) -> None:
    """Sense la firma de X, Y i Z no es compara res real. S'atura ABANS d'obrir res."""
    if not Path(firma).exists():
        raise SystemExit(f"Pany tancat: no hi ha {Path(firma).name}. Comparar una presa amb la "
                         "simulacio no es fa abans de firmar X, Y i Z.")


# --------------------------------------------------------------- la Y
#
# Tot el que segueix es d'esborranys/_conflicte_y.py, sense canviar cap operacio
# ni cap llavor: la regla que es firma ha de ser la que es va provar. Ho
# comprova esborranys/_comprova_y_comparar.py (el criteri Y8).

def guany_savgol(finestra: int, ordre: int) -> float:
    """Quant redueix el soroll blanc el filtre de la conversio, al centre."""
    return float(np.sqrt(np.sum(savgol_coeffs(finestra, ordre) ** 2)))


def guany_ar1(finestra: int, ordre: int, rho: float) -> float:
    """Quant redueix el filtre de la conversio un AR(1) de parametre rho."""
    h = savgol_coeffs(finestra, ordre)
    i = np.arange(len(h))
    return float(np.sqrt(h @ (rho ** np.abs(i[:, None] - i[None, :])) @ h))


def soroll_blanc_m(p, s_graus) -> tuple[float, float]:
    """Les sigmes de posicio de la marca i del gomet que donen el soroll de l'index."""
    s1, s2 = (math.radians(v) for v in s_graus)
    sig1 = s1 * p.L1
    sig2 = math.sqrt(max((s2 * p.r_gomet) ** 2 - sig1**2, 0.0))
    return sig1, sig2


def parametres_del_soroll(p, finestra_mostres: int, soroll_index_graus, lent_total_graus,
                          lent_rho) -> dict:
    """El model del soroll d'una presa, com el dels bessons de la setena tanda.

    soroll_index_graus  meta["geometria"]["rigidesa"]["soroll_angular_graus"]
    lent_total_graus    el soroll lent mesurat en moviment, filtrat (theta1, theta2)
    lent_rho            la seva autocorrelacio a un fotograma
    Torna el blanc i el lent CRUS (abans del filtre), en graus, que es el que sorteja
    soroll_dels_membres.
    """
    sig1, sig2 = soroll_blanc_m(p, soroll_index_graus)
    blanc = [float(soroll_index_graus[0]), math.degrees(math.hypot(sig1, sig2) / p.r_gomet)]
    gw = guany_savgol(finestra_mostres, 3)
    lent = []
    for k in range(2):
        T, w = lent_total_graus[k], blanc[k] * gw
        lent.append(math.sqrt(max(T * T - w * w, 0.0)) / guany_ar1(finestra_mostres, 3, lent_rho[k]))
    return {"finestra_mostres": int(finestra_mostres), "blanc_graus": blanc, "lent_graus": lent,
            "rho": [float(r) for r in lent_rho]}


def soroll_dels_membres(ps: dict, n1: int, n: int, llavor: int, files: int = FILES_FEIX):
    """Una realitzacio del soroll del tir per a cada fila del feix, passada pel filtre de la
    conversio, en radians: (theta1, theta2), cadascun de forma (files, n). Les n1 primeres
    mostres crues surten d'una font i les que passen d'n1, d'una altra (com a la setena tanda)."""
    rng = np.random.default_rng(llavor)
    rng2 = np.random.default_rng([llavor, 22102026])
    fin = ps["finestra_mostres"]
    fora = []
    for k in range(2):
        cru = math.radians(ps["blanc_graus"][k]) * rng.standard_normal((files, n1))
        e = rng.standard_normal((files, n1))
        if n > n1:
            cru = np.hstack([cru, math.radians(ps["blanc_graus"][k]) * rng2.standard_normal((files, n - n1))])
            e = np.hstack([e, rng2.standard_normal((files, n - n1))])
        rho = ps["rho"][k]
        cru = cru + math.radians(ps["lent_graus"][k]) * lfilter([math.sqrt(1.0 - rho * rho)], [1.0, -rho], e, axis=1)
        fora.append(savgol_filter(cru, fin, 3, axis=1))
    return fora[0], fora[1]


def horitzo_feix(tg, A2, x_graus: float = X_GRAUS) -> float:
    """L'horitzo: el primer instant en que maxim - minim de theta2 del feix passa de X."""
    w = A2.max(axis=0) - A2.min(axis=0)
    fora = np.nonzero(w > math.radians(x_graus))[0]
    return float(tg[fora[0]]) if len(fora) else math.inf


def avalua_y(t, th1, th2, zero: float, tg, A1, A2, H: float, des_de: float = 0.0,
             limit: float = 1.0, soroll_feix=None, confirma_s: float = CONFIRMA_S) -> dict:
    """La Y d'un tir: dins si theta1 I theta2 son entre el minim i el maxim del feix (amb el
    soroll del tir, si es dona); n'ha sortit a t_s si hi roman fora confirma_s seguits; falla
    si t_s cau abans de l'horitzo H. `tg`, A1 i A2 son el feix (files = membres), en temps des
    de l'amollada; `zero` es l'amollada de la presa al seu rellotge; `limit`, fins on arriba
    el feix."""
    if not math.isfinite(H) or H > limit - confirma_s:
        return {"avaluable": False, "motiu": "l'horitzo cau fora de l'abast del feix"}
    if H <= des_de:
        return {"avaluable": False, "motiu": "l'horitzo cau abans del tram mirat"}
    tau = np.asarray(t, dtype=float) - zero
    sel = (tau >= 0.0) & (tau <= limit)
    ts, a1, a2 = tau[sel], np.asarray(th1)[sel], np.asarray(th2)[sel]
    M1 = np.array([np.interp(ts, tg, A1[i]) for i in range(len(A1))])
    M2 = np.array([np.interp(ts, tg, A2[i]) for i in range(len(A2))])
    if soroll_feix is not None:
        # Cada membre amb la seva realitzacio del soroll, fotograma a fotograma des del
        # primer fotograma despres del zero.
        M1 = M1 + soroll_feix[0][:, :len(ts)]
        M2 = M2 + soroll_feix[1][:, :len(ts)]
    in1 = (a1 >= M1.min(axis=0)) & (a1 <= M1.max(axis=0))
    in2 = (a2 >= M2.min(axis=0)) & (a2 <= M2.max(axis=0))
    fora = ~(in1 & in2)
    c = np.concatenate([[0], np.cumsum(fora)])
    fi = np.searchsorted(ts, ts + confirma_s, side="right")
    tram = (ts >= des_de) & (ts < H)
    sortida = None
    for i in np.nonzero(tram & fora)[0]:
        if c[fi[i]] - c[i] == fi[i] - i:
            sortida = int(i)
            break
    r = {"avaluable": True, "falla": sortida is not None,
         "fraccio_dins": float(1.0 - fora[tram].mean()) if tram.any() else None}
    if sortida is not None:
        r["t_sortida_s"] = float(ts[sortida])
        r["angle"] = ("tots dos" if not in1[sortida] and not in2[sortida]
                      else "theta1" if not in1[sortida] else "theta2")
    return r


def veredicte_grup(per_tir: dict, c: int = C_GRUP) -> dict:
    """El veredicte d'un grup: falla si c tirs o mes surten del feix abans de l'horitzo.
    `per_tir` es {nom: resultat d'avalua_y}. Els no avaluables no compten, i es diuen."""
    avaluables = {n: r for n, r in per_tir.items() if r.get("avaluable")}
    fallen = sorted(n for n, r in avaluables.items() if r["falla"])
    return {"falla": len(fallen) >= c, "c": c, "fallen": fallen, "avaluables": len(avaluables),
            "no_avaluables": sorted(n for n in per_tir if n not in avaluables)}


def corregeix_repos(theta1, theta2, rho: float, n_cua: int):
    """La correccio del biaix de theta1 (24/10, documentacio/pla-correccio-biaix.md): el repos final
    ha de donar (0, 0).

    La conversio agafa com a vertical la recta pivot -> extrem del repos, pero la marca de
    l'articulacio pot ser fora de l'eix de costat (el gomet del rodament es va tornar a enganxar), i
    llavors theta1 no dona zero al repos. Supòsit: la marca es a sigma = tan(theta1 del repos) de costat
    de l'articulacio (en unitats de L1), fixa a la barra 1. Llavors theta1 = theta1 mesurat - r1, i
    l'articulacio es el peu de la perpendicular de la marca a l'eix de la barra, a distancia 1.
    theta2 es la direccio articulacio -> gomet, amb |articulacio -> gomet| = rho: la distancia marca ->
    gomet, que canvia amb el plegament, surt d'aquesta condicio (25/10: la primera versio posava
    l'articulacio a sqrt(1 + d^2) i prenia la distancia marca -> gomet com a rho, un error de segon
    ordre a theta2; la Y14 ho prova). rho = r_gomet / L1. n_cua: les mostres del final que fan el repos.
    Torna (theta1, theta2, theta1 del repos que s'ha tret, en rad)."""
    th1, th2 = np.asarray(theta1, dtype=float), np.asarray(theta2, dtype=float)
    plega = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi   # noqa: E731
    r1 = float(np.mean(plega(th1[-n_cua:])))
    s = math.tan(r1)                                    # la marca, de costat, en unitats de L1
    n1 = th1 - r1
    pu = np.cos(n1) * np.sin(th2) - np.sin(n1) * np.cos(th2)    # p (perpendicular de la barra 1) . u (theta2 mesurat)
    L = -s * pu + np.sqrt(rho * rho - s * s * (1.0 - pu * pu))   # la distancia marca -> gomet
    xg = s * np.cos(n1) + L * np.sin(th2)               # articulacio -> gomet
    yg = s * np.sin(n1) - L * np.cos(th2)
    n2 = th2 + plega(np.arctan2(xg, -yg) - th2)
    return n1, n2, r1


def corregeix_dues_parts(theta1, theta2, d, rho_g: float, rho: float, n_cua: int, r: float = 1.0):
    """La correccio en dues parts (24/10, documentacio/pla-correccio-biaix.md, «L'opcio 1»).

    A, l'error fix a la imatge: els angles es tornen a calcular respecte del centre del cercle (l'eix de
    debo) en lloc del pivot de la conversio. d = (centre - pivot)/L1 al marc de la conversio (x dreta, y
    amunt); rho_g = |extrem - pivot|/L1 al repos final, que la conversio posa a la vertical (0, -rho_g).
    Suposa que l'error es del pivot: theta2 nomes canvia per la vertical nova.
    La marca es posa exactament sobre el cercle de radi r (en unitats de L1) al voltant del centre, al llarg
    del raig que veu la conversio des del pivot (24/10: la primera versio la posava a distancia 1 del pivot,
    i la Y13 hi va fallar per aquesta aproximacio de segon ordre).
    B, l'error enganxat a la barra: corregeix_repos a sobre del resultat d'A.
    Torna (theta1, theta2, theta1 del repos que B ha tret)."""
    th1, th2 = np.asarray(theta1, dtype=float), np.asarray(theta2, dtype=float)
    dx, dy = float(d[0]), float(d[1])
    if dx != 0.0 or dy != 0.0:
        plega = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi   # noqa: E731
        phi = math.atan2(-dx, rho_g + dy)                       # la vertical nova: centre -> extrem al repos
        ux, uy = np.sin(th1), -np.cos(th1)
        ud = ux * dx + uy * dy
        lam = ud + np.sqrt(ud * ud - (dx * dx + dy * dy) + r * r)
        mx, my = lam * ux - dx, lam * uy - dy
        th1 = th1 + plega(np.arctan2(mx, -my) - phi - th1)
        th2 = th2 - phi
    return corregeix_repos(th1, th2, rho, n_cua)


CENTRES = ARREL / "dades" / "versemblant" / "centres_cercle.json"
CLICS = ARREL / "dades" / "versemblant" / "correccio_clics.json"


def posa_clics(theta1, theta2, t, ta: float, n_cua: int, c: dict):
    """La correccio amb clics (27/10, documentacio/pla-correccio-biaix.md, «La correccio amb clics» i «27/10: l'endavant»):
    la vertical del clic de l'extrem al repos final (un gir rigid, c["dth_rad"]), i el repos final (les ultimes n_cua mostres)
    i la finestra de l'amollada ([ta - 0,4; ta - 0,02]) posats als angles dels clics, desembolicats al de la serie. Les
    constants les escriu eines/correccio_amb_clics.py. Torna (theta1, theta2)."""
    th1 = np.array(theta1, dtype=float) + c["dth_rad"]
    th2 = np.array(theta2, dtype=float) + c["dth_rad"]
    ini = (t >= max(0.0, ta - 0.4)) & (t <= ta - 0.02)
    cua = np.arange(len(t) - n_cua, len(t))
    for (a1, a2), idx in ((c["theta_inicial_rad"], ini), (c["theta_final_rad"], cua)):
        for serie, a in ((th1, a1), (th2, a2)):
            serie[idx] = a + 2.0 * math.pi * round((float(np.mean(serie[idx])) - a) / (2.0 * math.pi))
    return th1, th2


def corregeix_presa(taula, meta: dict, nom: str, amb_clics: bool = True):
    """L'UNICA manera de corregir una presa real (25/10, documentacio/pla-correccio-biaix.md, «La T6 del
    cercle» i «La rho de la part B»): corregeix_dues_parts amb la part A de dades/versemblant/centres_cercle.json
    (de eines/centre_del_cercle.py), la cua de la conversio (meta["geometria"]["cua_s"]) i la rho del regle
    (r_gomet / L1 de parametres.json). Nomes obre el JSON dels centres, el dels clics i parametres.json.
    La rho no surt de la imatge: la part B la vol respecte del centre del cercle, i L2_px / L1_px es respecte del
    pivot de la conversio (fins a un 9 % de diferencia); i ben posada al centre, s'aparta del regle d'un 1,5 a un
    6 %, perque arrossega l'efecte del tope i la incertesa vertical del centre. El regle es independent, el mateix
    a totes les preses, i el de la simulacio.
    27/10: si la presa es a dades/versemblant/correccio_clics.json (les 23 preses de 45 i 90 graus clicades), la
    correccio amb clics (posa_clics, i la part A refeta amb els clics); si no, o amb amb_clics=False, la d'abans, bit a
    bit. Torna (theta1, theta2, {"r1": ..., "rho": ..., "n_cua": ..., "clics": ...})."""
    from pendol.fisica import Parametres
    pa = json.loads(CENTRES.read_text(encoding="utf-8"))["preses"][nom]["part_a"]
    geo = meta["geometria"]
    t = taula["t"].to_numpy()
    n_cua = max(10, int(round(float(geo["cua_s"]) / float(np.median(np.diff(t))))))
    p = Parametres.des_de_json(ARREL / "parametres.json")
    rho = p.r_gomet / p.L1
    th1, th2 = taula["theta1"].to_numpy(), taula["theta2"].to_numpy()
    c = json.loads(CLICS.read_text(encoding="utf-8"))["preses"].get(nom) if amb_clics and CLICS.exists() else None
    if c is not None:
        th1, th2 = posa_clics(th1, th2, t, float(meta["amollada"]["amollada_s"]), n_cua, c)
        pa = c["part_a"]
    th1, th2, r1 = corregeix_dues_parts(th1, th2, pa["d"], pa["rho_g"], rho, n_cua, pa["r"])
    return th1, th2, {"r1": r1, "rho": rho, "n_cua": n_cua, "clics": c is not None}


# --------------------------------------------------------------- l'autoprova

def autoprova_y() -> bool:
    """Y1-Y7 de documentacio/pla-firma-i-y.md, escrits abans del codi."""
    import sys as _sys

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    print("\nAutoprova de la Y (criteris Y1-Y7 de documentacio/pla-firma-i-y.md)")
    # Un feix fabricat: 21 files que s'obren linealment al voltant d'una corba, fins a 1,5 s.
    # L'amplada de theta2 val 20*w(t), i la X es tria perque l'horitzo caigui a ~1,0 s.
    FPS, LIMIT = 240.0, 1.5
    tg = np.arange(0.0, LIMIT + 1.25e-4, 2.5e-4)
    w = 1e-3 * (1.0 + 5.0 * tg)
    k = np.arange(21)[:, None] - 10
    A1 = np.sin(2.0 * tg) + k * w
    A2 = 0.5 * np.cos(3.0 * tg) + k * w
    x = math.degrees(20 * 1e-3 * (1.0 + 5.0 * 1.0))
    H = horitzo_feix(tg, A2, x)
    zero = 0.37
    ts = np.arange(0.0, LIMIT + 0.3, 1.0 / FPS)
    t = zero + ts

    def tir(salt_des_de=None, durada=math.inf, fila=10):
        th1 = np.interp(ts, tg, A1[fila], right=np.nan)
        th2 = np.interp(ts, tg, A2[fila], right=np.nan)
        if salt_des_de is not None:
            fora = (ts >= salt_des_de) & (ts < salt_des_de + durada)
            th2 = np.where(fora, th2 + 0.2, th2)
        return t, th1, th2

    print(f"   (feix fabricat: horitzo {H:.4f} s amb X = {x:.3f} graus, {FPS:.0f} fps)")
    r = avalua_y(*tir(fila=3), zero, tg, A1, A2, H, limit=LIMIT)
    mira(r["avaluable"] and not r["falla"] and r["fraccio_dins"] == 1.0,
         f"Y1 un membre del feix no en surt mai: {r}")
    r = avalua_y(*tir(0.4), zero, tg, A1, A2, H, limit=LIMIT)
    mira(r["falla"] and abs(r["t_sortida_s"] - 0.4) <= 1.0 / FPS and r["angle"] == "theta2",
         f"Y2 surt a 0,4 s i s'hi queda: falla a {r.get('t_sortida_s', float('nan')):.4f} s")
    r19 = avalua_y(*tir(0.4, 0.19), zero, tg, A1, A2, H, limit=LIMIT)
    r21 = avalua_y(*tir(0.4, 0.21), zero, tg, A1, A2, H, limit=LIMIT)
    mira(not r19["falla"] and r21["falla"],
         f"Y3 fora 0,19 s no falla ({r19['falla']}), fora 0,21 s falla ({r21['falla']})")
    r = avalua_y(*tir(H + 0.05), zero, tg, A1, A2, H, limit=LIMIT)
    r_inf = avalua_y(*tir(), zero, tg, A1, A2, math.inf, limit=LIMIT)
    r_lim = avalua_y(*tir(), zero, tg, A1, A2, LIMIT - 0.1, limit=LIMIT)
    mira(r["avaluable"] and not r["falla"] and not r_inf["avaluable"] and not r_lim["avaluable"],
         f"Y4 fora nomes despres de l'horitzo no falla ({r['falla']}); horitzo infinit o a "
         f"menys de 0,2 s del limit: no avaluable ({r_inf['avaluable']}, {r_lim['avaluable']})")

    bo, dolent = {"avaluable": True, "falla": False}, {"avaluable": True, "falla": True}
    no = {"avaluable": False, "motiu": "prova"}
    g1 = veredicte_grup({**{f"t{i}": bo for i in range(11)}, "t11": dolent})
    g2 = veredicte_grup({**{f"t{i}": bo for i in range(10)}, "t10": dolent, "t11": dolent})
    g3 = veredicte_grup({**{f"t{i}": bo for i in range(10)}, "t10": dolent, "t11": no})
    mira(not g1["falla"] and g2["falla"] and not g3["falla"] and g3["no_avaluables"] == ["t11"]
         and g3["avaluables"] == 11, "Y5 grup de 12: 1 fallat passa, 2 fallen; el no avaluable es diu i no compta")

    ps = {"finestra_mostres": 13, "blanc_graus": [0.3, 0.7], "lent_graus": [1.1, 2.0], "rho": [0.97, 0.99]}
    a = soroll_dels_membres(ps, 250, 400, 1234)
    b = soroll_dels_membres(ps, 250, 400, 1234)
    curt = soroll_dels_membres(ps, 250, 250, 1234)
    # La versio curta de l'esborrany (soroll_dels_membres de _conflicte_y.py), escrita aqui.
    rng = np.random.default_rng(1234)
    ref = []
    for kk in range(2):
        cru = math.radians(ps["blanc_graus"][kk]) * rng.standard_normal((FILES_FEIX, 250))
        e = rng.standard_normal((FILES_FEIX, 250))
        rho = ps["rho"][kk]
        cru = cru + math.radians(ps["lent_graus"][kk]) * lfilter([math.sqrt(1.0 - rho * rho)], [1.0, -rho], e, axis=1)
        ref.append(savgol_filter(cru, 13, 3, axis=1))
    mira(all(np.array_equal(u, v) for u, v in zip(a, b)) and all(np.array_equal(u, v) for u, v in zip(curt, ref)),
         "Y6 el soroll: la mateixa llavor, bit a bit; amb n = n1, la versio curta bit a bit")

    if FIRMA.exists():
        mira(True, "Y7 (hi ha firma.json: la prova del pany no s'hi pot fer)")
    else:
        oberts = []

        def registra(event, args):
            if event == "open" and isinstance(args[0], (str, bytes)) and actiu[0]:
                oberts.append(str(args[0]))

        actiu = [True]
        _sys.addaudithook(registra)
        try:
            main(["--real", "dades/net/90_1.csv", "--feix", "dades/sim/no_hi_es/"])
            negat = False
        except SystemExit as e:
            negat = "Pany tancat" in str(e)
        actiu[0] = False
        net = [o for o in oberts if "net" in Path(o).parts]
        mira(negat and not net, f"Y7 sense firma, --real s'atura ({negat}) sense obrir res de dades/net ({len(net)})")
    return not problemes


def autoprova_correccio() -> bool:
    """Y9-Y10 de documentacio/pla-correccio-biaix.md (pas 1 bis), escrits abans del codi; Y14 (25/10), la part B exacta."""
    import contextlib
    import io
    import tempfile
    from pendol import amollada, tracker
    from pendol.fisica import Parametres

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    print("\nAutoprova de la correccio del repos (Y9-Y10 i Y14 de documentacio/pla-correccio-biaix.md)")
    p = Parametres.des_de_json(ARREL / "parametres.json")
    FPS, QUIET, MOU, REPOS = 240.0, 0.5, 3.0, 2.0
    t, th1, th2 = amollada.tir_fals_amb_repos(p, np.radians([40.0, 45.0, 0.0, 0.0]), QUIET, MOU, FPS)
    nr = int(round(REPOS * FPS))
    t = np.concatenate([t, t[-1] + np.arange(1, nr + 1) / FPS])
    th1 = np.concatenate([th1, np.zeros(nr)])           # el repos final, a (0, 0) de veritat
    th2 = np.concatenate([th2, np.zeros(nr)])
    rho = p.r_gomet / p.L1
    n_cua = int(round(1.0 * FPS))

    def per_la_conversio(s_m):
        x1 = p.L1 * np.sin(th1) + s_m * np.cos(th1)     # la marca, s de costat (fixa a la barra 1)
        y1 = -p.L1 * np.cos(th1) + s_m * np.sin(th1)
        x2 = p.L1 * np.sin(th1) + p.r_gomet * np.sin(th2)
        y2 = -p.L1 * np.cos(th1) - p.r_gomet * np.cos(th2)
        with tempfile.TemporaryDirectory() as dd:
            cru, net = Path(dd) / "cru.csv", Path(dd) / "net.csv"
            amollada._escriu_cru(cru, t, x1, y1, x2, y2)
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False)
            taula, _ = dades.carrega(net)
        return taula["theta1"].to_numpy(), taula["theta2"].to_numpy()

    m1, m2 = per_la_conversio(0.003)
    c1, c2, r1 = corregeix_repos(m1, m2, rho, n_cua)
    mou = (t >= QUIET + 0.2) & (t <= QUIET + MOU - 0.2)
    err = math.degrees(max(np.max(np.abs(c1[mou] - th1[mou])), np.max(np.abs(c2[mou] - th2[mou]))))
    rep = math.degrees(max(abs(float(np.mean(c1[-n_cua:]))), abs(float(np.mean(c2[-n_cua:])))))
    mira(err < 0.05 and rep < 0.01,
         f"Y9 marca a 3 mm de costat (theta1 del repos {math.degrees(r1):+.2f} graus): corregida, error "
         f"{err:.4f} graus en moviment (< 0,05) i repos a {rep:.5f} graus (< 0,01)")
    m1, m2 = per_la_conversio(0.0)
    c1, c2, _ = corregeix_repos(m1, m2, rho, n_cua)
    mou_res = max(np.max(np.abs(c1 - m1)), np.max(np.abs(c2 - m2)))
    mira(mou_res < 1e-9, f"Y10 marca a l'eix: la correccio mou com a molt {mou_res:.1e} rad (< 1e-9)")

    # Y14 (25/10): geometria pura, sense filtre, amb els bracos plegats i la marca a tan 4 graus de costat.
    rng = np.random.default_rng(25102026)
    nm, nr14, sig = 4000, 500, math.tan(math.radians(4.0))
    v1 = np.concatenate([rng.uniform(-2.5, 2.5, nm), np.zeros(nr14)])
    v2 = np.concatenate([rng.uniform(-np.pi, np.pi, nm), np.zeros(nr14)])
    jx, jy = np.sin(v1), -np.cos(v1)                       # l'articulacio, a L1 = 1
    mx, my = jx + sig * np.cos(v1), jy + sig * np.sin(v1)   # la marca, de costat, fixa a la barra 1
    gx, gy = jx + rho * np.sin(v2), jy - rho * np.cos(v2)
    w1 = np.arctan2(mx, -my)                               # el que veu la conversio (vertical pivot -> gomet al repos)
    w2 = np.arctan2(gx - mx, -(gy - my))
    plega = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi   # noqa: E731
    c1, c2, _ = corregeix_repos(w1, w2, rho, nr14)
    e14 = max(np.max(np.abs(plega(c1 - v1))), np.max(np.abs(plega(c2 - v2))))
    # informatiu: la versio d'abans del 25/10
    d = -math.tan(float(np.mean(w1[-nr14:])))
    x1, y1 = np.sin(w1) + d * np.cos(w1), -np.cos(w1) + d * np.sin(w1)
    xg, yg = np.sin(w1) + rho * np.sin(w2), -np.cos(w1) - rho * np.cos(w2)
    vell = np.max(np.abs(plega(np.arctan2(xg - x1, -(yg - y1)) - v2)))
    mira(e14 < 1e-9, f"Y14 geometria pura, bracos plegats, marca a tan 4 graus de costat: error {e14:.1e} rad "
         f"(< 1e-9; la versio d'abans del 25/10, {math.degrees(vell):.3f} graus a theta2)")
    return not problemes


def autoprova_dues_parts() -> bool:
    """Y11-Y12 de documentacio/pla-correccio-biaix.md («L'opcio 1»), escrits abans del codi."""
    import contextlib
    import io
    import tempfile
    from pendol import amollada, tracker
    from pendol.fisica import Parametres

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    print("\nAutoprova de la correccio en dues parts (Y11-Y12 de documentacio/pla-correccio-biaix.md)")
    p = Parametres.des_de_json(ARREL / "parametres.json")
    FPS, QUIET, MOU, REPOS = 240.0, 0.5, 3.0, 2.0
    rho = p.r_gomet / p.L1
    n_cua = int(round(1.0 * FPS))
    px = p.L1 / 200.0                                      # 1 px, amb L1 = 200 px
    for graus in (45.0, 150.0):
        t, th1, th2 = amollada.tir_fals_amb_repos(p, np.radians([graus, graus, 0.0, 0.0]), QUIET, MOU, FPS)
        nr = int(round(REPOS * FPS))
        t = np.concatenate([t, t[-1] + np.arange(1, nr + 1) / FPS])
        th1 = np.concatenate([th1, np.zeros(nr)])
        th2 = np.concatenate([th2, np.zeros(nr)])
        ref = None
        for nom, d_px, s_px in (("sense cap error", (0.0, 0.0), 0.0),
                                ("pivot (0, 6) px per sota", (0.0, 6.0), 0.0),
                                ("pivot (2, 4) px i la marca 3 px de costat", (2.0, 4.0), 3.0)):
            # L'eix de debo a l'origen; el pivot marcat a -d (o sigui, centre - pivot = d); la marca, s de costat.
            Pm = -np.array(d_px) * px
            s = s_px * px
            jx, jy = p.L1 * np.sin(th1), -p.L1 * np.cos(th1)
            ax, ay = jx + s * np.cos(th1), jy + s * np.sin(th1)
            gx, gy = jx + p.r_gomet * np.sin(th2), jy - p.r_gomet * np.cos(th2)
            # La vertical de la conversio es la recta pivot marcat -> extrem al repos: es gira la imatge.
            gr = np.array([gx[-n_cua:].mean(), gy[-n_cua:].mean()]) - Pm
            beta = math.atan2(gr[0], -gr[1])
            cb, sb = math.cos(-beta), math.sin(-beta)
            gira = lambda x, y: ((x - Pm[0]) * cb - (y - Pm[1]) * sb, (x - Pm[0]) * sb + (y - Pm[1]) * cb)   # noqa: E731
            x1, y1 = gira(ax, ay)
            x2, y2 = gira(gx, gy)
            dc = gira(0.0, 0.0)
            d = (dc[0] / p.L1, dc[1] / p.L1)
            rho_g = float(np.hypot(*gr)) / p.L1
            r_marca = math.hypot(p.L1, s) / p.L1          # la marca gira a aquesta distancia del centre
            with tempfile.TemporaryDirectory() as dd:
                cru, net = Path(dd) / "cru.csv", Path(dd) / "net.csv"
                amollada._escriu_cru(cru, t, x1, y1, x2, y2)
                with contextlib.redirect_stdout(io.StringIO()):
                    tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False)
                taula, _ = dades.carrega(net)
            m1, m2 = taula["theta1"].to_numpy(), taula["theta2"].to_numpy()
            c1, c2, _ = corregeix_dues_parts(m1, m2, d, rho_g, rho, n_cua, r=r_marca)
            if nom == "sense cap error":
                ref = (m1, m2)
                mou_res = max(np.max(np.abs(c1 - m1)), np.max(np.abs(c2 - m2)))
                mira(mou_res < 1e-9, f"Y12 {graus:g}: sense cap error, la correccio mou {mou_res:.1e} rad (< 1e-9)")
                continue
            ini = t <= QUIET - 0.05
            mou = (t >= QUIET + 0.2) & (t <= QUIET + MOU - 0.2)
            e_ini = math.degrees(max(abs(float(np.mean(c1[ini] - th1[ini]))), abs(float(np.mean(c2[ini] - th2[ini])))))
            # 24/10: en moviment es compara amb la conversio de la mateixa trajectoria SENSE cap error, no amb
            # la veritat crua: el filtre de la conversio ja s'aparta 0,12 / 0,29 graus als 150 a velocitat alta
            # (la primera versio hi va fallar per aixo). El criteri no canvia; vegeu el pla.
            e_mou = math.degrees(max(np.max(np.abs(c1[mou] - ref[0][mou])), np.max(np.abs(c2[mou] - ref[1][mou]))))
            abans = math.degrees(max(abs(float(np.mean(m1[ini] - th1[ini]))), abs(float(np.mean(m2[ini] - th2[ini])))))
            mira(e_ini < 0.05 and e_mou < 0.05,
                 f"Y11 {graus:g}, {nom}: condicions inicials a {e_ini:.4f} graus (sense corregir {abans:.3f}), "
                 f"moviment a {e_mou:.4f} graus (< 0,05)")
    return not problemes


def autoprova() -> int:
    """troba_desfasament ha de recuperar un retard CONEGUT.

    Per que existeix. L'alineacio temporal es l'unica cosa d'aquest fitxer que
    pot fer mentir totes les altres: un fotograma de diferencia entre el t=0
    del video i el de la simulacio es disfressa de divergencia. I fins ara no
    s'havia comprovat mai que funcionis -no te sentit provar-ho amb dades
    reals, perque alli no se sap la resposta-.

    La prova: s'agafa un tir simulat de debo, se'n fa una copia DESPLACADA un
    retard conegut, se li posa el soroll de seguiment que s'ha mesurat de
    debo als videos (0,30 graus a theta1) i es mira si el recupera. El llindar
    es UN FOTOGRAMA a 240 fps, que es la precisio maxima que te sentit
    demanar-li.
    """
    import sys as _sys
    from pendol.fisica import Parametres
    from pendol.simulador import simula

    FPS, SOROLL_GRAUS = 240.0, 0.30
    TOLERANCIA = 1.0 / FPS  # un fotograma

    p = Parametres.des_de_json("parametres.json")
    t_s, y = simula(p, np.radians([150.0, 150.0, 0.0, 0.0]), 8.0, fps=FPS)
    th_s = y[0]

    print("Autoprova de l'alineacio temporal")
    print(f"   tir simulat de 8 s a {FPS:.0f} fps, soroll de {SOROLL_GRAUS} graus "
          f"(el mesurat als videos)")
    print(f"   tolerancia: {1000*TOLERANCIA:.2f} ms = un fotograma\n")
    print(f"   {'retard posat':>14} {'recuperat':>12} {'error':>10}")

    rng = np.random.default_rng(0)
    # El tram central, perque desplacar-lo no se'n surti per cap punta.
    dins = (t_s >= 1.0) & (t_s <= 6.0)
    t_r = t_s[dins]
    ok = True
    for s0 in (-0.200, -0.0375, 0.0, 0.0375, 0.200):
        # La corba "real" es la simulada mirada amb un rellotge desplacat:
        # real(t) = sim(t + s0). troba_desfasament ha de tornar s0.
        th_r = np.interp(t_r + s0, t_s, th_s)
        th_r = th_r + np.radians(rng.normal(0.0, SOROLL_GRAUS, th_r.shape))
        trobat, _ = troba_desfasament(t_r, th_r, t_s, th_s)
        error = abs(trobat - s0)
        aquest = error <= TOLERANCIA
        ok &= aquest
        print(f"   {1000*s0:>11.1f} ms {1000*trobat:>9.1f} ms {1000*error:>7.1f} ms"
              f"   {'ok' if aquest else '<<< NO EL TROBA'}")

    ok_y = autoprova_y()
    ok_c = autoprova_correccio()
    ok_d = autoprova_dues_parts()
    print()
    if not ok_c:
        print("ATENCIO -> la correccio del repos no compleix Y9-Y10. Un criteri que falla NO es relaxa.")
        return 1
    if not ok_d:
        print("ATENCIO -> la correccio en dues parts no compleix Y11-Y12. Un criteri que falla NO es relaxa.")
        return 1
    print("OK -> la correccio del repos compleix Y9-Y10 i Y14, i la de dues parts, Y11-Y12.")
    if not ok_y:
        print("ATENCIO -> la Y no compleix els seus criteris (Y1-Y7). Un criteri que falla NO es")
        print("           relaxa: s'atura i s'explica.")
        return 1
    print("OK -> la Y compleix Y1-Y7 (la reproduccio dels bessons, Y8, la fa")
    print("      esborranys/_comprova_y_comparar.py).")
    if ok:
        print("OK -> l'alineacio recupera el retard dins d'un fotograma.")
        print("     Limit declarat: aixo prova el metode amb una corba que ES la")
        print("     simulacio. Amb un pendol real hi ha a mes error de model, i")
        print("     aixo aquesta prova no ho pot veure.")
        return 0
    print("ATENCIO -> l'alineacio no recupera el retard. Mira la finestra i el")
    print("           rang de cerca de troba_desfasament abans de fiar-te de cap")
    print("           horitzo, que sortiria escurcat per un error d'alineacio.")
    return 1


def _percentils_feix(taules, t_obj, columna, desfasament):
    """Interpola tot el feix sobre la base temporal real i en treu percentils."""
    apilat = np.array(
        [np.interp(t_obj + desfasament, tb["t"].to_numpy(), tb[columna].to_numpy()) for tb in taules]
    )
    return (
        np.percentile(apilat, 2.5, axis=0),
        np.percentile(apilat, 97.5, axis=0),
        np.min(apilat, axis=0),
        np.max(apilat, axis=0),
    )


def main(argv=None) -> int:
    import sys
    if "--autoprova" in (sys.argv[1:] if argv is None else list(argv)):
        return autoprova()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova que l'alineacio temporal recupera un retard conegut")
    ap.add_argument("--real", required=True, help="CSV net del Tracker (sortida de pendol/tracker.py)")
    grup = ap.add_mutually_exclusive_group(required=True)
    grup.add_argument("--sim", help="CSV d'una simulacio")
    grup.add_argument("--feix", help="directori amb un feix de simulacions")
    ap.add_argument("--nom", default="comparacio", help="prefix dels fitxers de figura")
    ap.add_argument("--figures", default="figures", help="directori de sortida de les figures")
    ap.add_argument("--alinea", choices=("auto", "cap"), default="auto", help="ajust del desfasament inicial")
    ap.add_argument("--finestra-alineacio", type=float, default=2.0, help="segons usats per alinear")
    ap.add_argument("--llindar", type=float, default=0.5,
                    help="divergencia (rad) que marca la perdua de predictibilitat (0.5 rad ~ 29 graus)")
    ap.add_argument("--persistencia", type=float, default=0.2,
                    help="temps seguit (s) que la realitat ha d'estar fora del feix perque compti")
    args = ap.parse_args(argv)
    pany()   # abans d'obrir res

    estil.aplica()

    real, meta_real = dades.carrega(args.real)
    if args.feix:
        feix = dades.carrega_conjunt(args.feix)
        sim = feix[0]  # el membre 0 es sempre el nominal (veure simulador.py)
        print(f"Feix de {len(feix)} simulacions carregat de {args.feix}")
    else:
        feix, sim = None, dades.carrega(args.sim)[0]

    t_r = real["t"].to_numpy()
    t_s = sim["t"].to_numpy()

    # --- Alineacio temporal ---------------------------------------------
    if args.alinea == "auto":
        desf, err = troba_desfasament(
            t_r, real["theta1"].to_numpy(), t_s, sim["theta1"].to_numpy(), args.finestra_alineacio
        )
        print(f"Desfasament ajustat: {desf*1000:+.1f} ms  (RMSE inicial {np.degrees(err):.2f} graus)")
    else:
        desf = 0.0

    # Ens quedem nomes amb el tram on totes dues corbes existeixen.
    valid = (t_r + desf >= t_s[0]) & (t_r + desf <= t_s[-1])
    t = t_r[valid]
    if len(t) < 10:
        print("ERROR: no hi ha prou solapament temporal. Revisa --alinea o les durades.")
        return 1

    th1_r, th2_r = real["theta1"].to_numpy()[valid], real["theta2"].to_numpy()[valid]
    th1_s = np.interp(t + desf, t_s, sim["theta1"].to_numpy())
    th2_s = np.interp(t + desf, t_s, sim["theta2"].to_numpy())

    # --- Metriques -------------------------------------------------------
    d1, d2 = dif_angular(th1_s, th1_r), dif_angular(th2_s, th2_r)
    divergencia = np.hypot(d1, d2)

    passa = np.flatnonzero(divergencia > args.llindar)
    horitzo = float(t[passa[0]] - t[0]) if len(passa) else None

    dins_2s = t <= t[0] + 2.0
    rmse_2s = np.degrees(np.sqrt(np.mean(d1[dins_2s] ** 2 + d2[dins_2s] ** 2)))

    # Bandes del feix, calculades un sol cop: serveixen per a la figura i
    # tambe per a la metrica de sota.
    bandes = {}
    if feix:
        for col in ("theta1", "theta2"):
            bandes[col] = _percentils_feix(feix, t, col, desf)

    # Metrica clau: quant triga la realitat a sortir del feix. A diferencia
    # de --llindar (un numero triat a ma), aquesta surt de la TEVA incertesa
    # de mesura, i per tant es el que de debo pots afirmar del teu muntatge.
    #
    # Ha de ser PERSISTENT: alli on totes les trajectories del feix es
    # creuen, la banda es pinca fins a amplada zero i el soroll de mesura et
    # deixa fora un instant sense que aixo signifiqui res. Nomes compta si
    # la realitat es queda fora una estona seguida.
    temps_dins, fraccio_dins = None, None
    if feix:
        dins = np.ones_like(t, dtype=bool)
        for col, th_r in (("theta1", th1_r), ("theta2", th2_r)):
            _, _, mn, mx = bandes[col]
            dins &= (th_r >= mn) & (th_r <= mx)
        fraccio_dins = float(dins.mean())

        n_min = max(1, int(round(args.persistencia / float(np.median(np.diff(t))))))
        seguits = np.convolve((~dins).astype(int), np.ones(n_min, dtype=int), mode="valid")
        surt = np.flatnonzero(seguits == n_min)
        temps_dins = float(t[surt[0]] - t[0]) if len(surt) else None

    resultats = {
        "real": args.real,
        "simulacio": args.sim or args.feix,
        "membres_feix": len(feix) if feix else 1,
        "desfasament_s": desf,
        "durada_comparada_s": float(t[-1] - t[0]),
        "rmse_primers_2s_graus": float(rmse_2s),
        "llindar_rad": args.llindar,
        "horitzo_predictibilitat_s": horitzo,
        "temps_dins_del_feix_s": temps_dins,
        "fraccio_de_temps_dins_del_feix": fraccio_dins,
        "persistencia_s": args.persistencia,
    }

    print("\n" + "=" * 62)
    print("RESULTATS")
    print("=" * 62)
    print(f"Tram comparat            : {t[-1]-t[0]:.2f} s")
    print(f"RMSE dels primers 2 s    : {rmse_2s:.2f} graus")
    if horitzo is not None:
        print(f"HORITZO DE PREDICTIBILITAT: {horitzo:.2f} s")
        print(f"  (temps fins que la divergencia supera {args.llindar} rad = {np.degrees(args.llindar):.0f} graus)")
    else:
        print(f"La divergencia no arriba mai a {args.llindar} rad en aquest tram.")
    if feix:
        if temps_dins is not None:
            print(f"La realitat surt del feix simulat als {temps_dins:.2f} s")
        else:
            print("La realitat es queda DINS el feix simulat tot el tram.")
        print(f"Fraccio de temps dins del feix: {fraccio_dins:.1%}")

    # --- Figura 1: trajectories -----------------------------------------
    fig, eixos = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    mans, etiqs = [], []
    for eix, (th_r, th_s, col, etiq) in zip(
        eixos,
        [(th1_r, th1_s, "theta1", r"$\theta_1$"), (th2_r, th2_s, "theta2", r"$\theta_2$")],
    ):
        if feix:
            p_baix, p_alt, mn, mx = bandes[col]
            b_mm = eix.fill_between(t, np.degrees(mn), np.degrees(mx), color=estil.FEIX,
                                    alpha=0.35, linewidth=0)
            b_95 = eix.fill_between(t, np.degrees(p_baix), np.degrees(p_alt), color=estil.FEIX,
                                    alpha=0.75, linewidth=0)
        (l_sim,) = eix.plot(t, np.degrees(th_s), color=estil.SIMULACIO, linestyle=estil.TRAC_SIMULACIO)
        (l_real,) = eix.plot(t, np.degrees(th_r), color=estil.REAL, linestyle=estil.TRAC_REAL)
        eix.set_ylabel(f"{etiq}  (graus)")

        # Els limits verticals els manen les DUES CORBES, no el feix. Quan el
        # segon angle es descabdella, el feix arriba a milers de graus i, si
        # el deixessim manar, aixafaria la figura fins a fer-la illegible.
        # Que el feix desbordi el panell no es cap perdua d'informacio: vol
        # dir que ja no en sabem res, que es justament el missatge.
        graus_r, graus_s = np.degrees(th_r), np.degrees(th_s)
        baix, alt = min(graus_r.min(), graus_s.min()), max(graus_r.max(), graus_s.max())
        marge = 0.10 * (alt - baix)
        eix.set_ylim(baix - marge, alt + marge)

        if not mans:
            if feix:
                mans += [b_mm, b_95]
                etiqs += ["feix simulat (min-max)", "feix simulat (95 %)"]
            mans += [l_sim, l_real]
            etiqs += ["simulacio (nominal)", "pendol real (Tracker)"]

    if horitzo is not None:
        for eix in eixos:
            eix.axvline(t[0] + horitzo, color=estil.TINTA_SUAU, linewidth=1.0, linestyle=":")
        eixos[0].annotate(
            f"horitzo {horitzo:.1f} s",
            xy=(t[0] + horitzo, eixos[0].get_ylim()[1]),
            xytext=(4, -12), textcoords="offset points",
            color=estil.TINTA_2, fontsize=9,
        )

    eixos[1].set_xlabel("temps (s)")
    eixos[0].set_title(f"Pendol real vs simulacio  —  {args.nom}", loc="left", pad=10)
    fig.tight_layout()
    # Llegenda sota la figura: mai tapa les dades, passi el que passi.
    fig.legend(mans, etiqs, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncols=len(mans))
    estil.desa(fig, f"{args.nom}_trajectories", args.figures)

    # --- Figura 2: divergencia -------------------------------------------
    fig2, eix = plt.subplots(figsize=(9, 4.2))
    eix.semilogy(t - t[0], np.maximum(divergencia, 1e-6), color=estil.SIMULACIO)
    eix.axhline(args.llindar, color=estil.TINTA_SUAU, linewidth=1.0, linestyle="--")
    eix.annotate(f"llindar {np.degrees(args.llindar):.0f}$\\degree$",
                 xy=(0, args.llindar), xytext=(4, 4), textcoords="offset points",
                 color=estil.TINTA_2, fontsize=9)
    if horitzo is not None:
        eix.axvline(horitzo, color=estil.TINTA_SUAU, linewidth=1.0, linestyle=":")
        eix.annotate(f"{horitzo:.1f} s", xy=(horitzo, eix.get_ylim()[0]), xytext=(4, 10),
                     textcoords="offset points", color=estil.TINTA_2, fontsize=9)
    eix.set_xlabel("temps des de l'inici (s)")
    eix.set_ylabel("divergencia  $\\sqrt{\\Delta\\theta_1^2+\\Delta\\theta_2^2}$  (rad)")
    eix.set_title("Separacio entre el pendol real i la simulacio", loc="left", pad=10)
    fig2.tight_layout()
    estil.desa(fig2, f"{args.nom}_divergencia", args.figures)

    # --- Resultats numerics a disc ---------------------------------------
    cami_res = Path(args.figures) / f"{args.nom}_resultats.json"
    cami_res.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetriques desades a {cami_res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
