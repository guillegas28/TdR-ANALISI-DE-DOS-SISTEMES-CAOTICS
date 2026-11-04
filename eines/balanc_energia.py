"""Balanc d'energia de les preses caotiques: qui s'emporta l'energia.

El pendol s'amolla del repos i acaba gairebe al repos. Aixo fa que l'energia
que s'ha dissipat es pugui saber SENSE DERIVAR RES: es la diferencia d'energia
potencial entre el principi i el final. Es la mesura mes robusta que dona un
video, perque la posicio es el que el Tracker mesura de debo i la velocitat ja
es una derivada amb el soroll que aixo comporta.

Despres es compara amb el que poden explicar els rodaments, amb els coeficients
que va donar eines/friccio_pendol_doble.py a partir de dues preses de decaiment
que NO comparteixen res amb aquestes. Si en falta, en falta per algun mecanisme
que aquelles preses no podien veure.

    energia alliberada  -  la que expliquen els rodaments  =  el forat

I el forat, si es aire, ha de complir una cosa comprovable: UN SOL coeficient
l'ha de tancar a totes les preses alhora, que van de 200 a 360 mJ. Amb dues
preses aixo es un ajust; amb set, es una prediccio.

MODEL. Pendol doble FISIC: barres amb massa repartida, no masses puntuals.

    A = m1*d1 + (m_cargols + m2)*L1        B = m2*d2
    V(th1,th2) = -g * [ A*cos(th1) + B*cos(th2) ]
    T = 0.5*I1*w1^2 + 0.5*I2*w2^2 + m2*L1*d2*w1*w2*cos(th2-th1)

L'aire va contra la velocitat ABSOLUTA de cada tros de barra, no contra la
rotacio relativa del rodament: no sap on son les articulacions. Per a la barra
1, que nomes gira, l'integral es tanca; per a la 2, que gira i alhora se la
passeja el primer brac, s'ha d'avaluar numericament.

Us. Les preses es passen amb comodins per grup i NO amb dades/net/*.csv:
alli hi ha tambe el magnetic_3072.csv, que es de l'altre sistema i no te
angles, i el balanc hi peta.

    .venv\\Scripts\\python eines\\balanc_energia.py dades/net/90_*.csv dades/net/150_*.csv
    .venv\\Scripts\\python eines\\balanc_energia.py <csv...> --json esborranys/balanc.json
    .venv\\Scripts\\python eines\\balanc_energia.py --autoprova
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import amollada  # noqa: E402
from pendol.fisica import Friccio, Parametres  # noqa: E402

# Nomes per a les preses que no porten l'amollada a les metadades, convertides
# abans del 24/09/2026: el criteri vell, sobre el detector de pixels.
QUIET_VELL_S = 0.05

CUA_S = 2.0        # s de final que es promitgen per l'estat final
PUNTS_BARRA = 40   # trossos en que es talla cada barra per l'integral de l'aire


# ------------------------------------------------------------- els parametres


class Muntatge:
    """Les constants del pendol real. Una capa prima sobre pendol/fisica.py.

    Fins al 22/09/2026 aquesta classe es calculava les inercies i el potencial
    pel seu compte, amb les mateixes formules que fisica.py pero escrites dues
    vegades. Ara les demana, i aixo no es estetica: es l'unica manera de poder
    dir que EL SIMULADOR I EL BALANC SON EL MATEIX PENDOL, que es justament el
    que fa que comparar la simulacio amb el video signifiqui alguna cosa.

    Els noms dels atributs es queden com eren perque la resta del fitxer no
    s'hagi de tocar. L'`acoblament` es la K de fisica.py.
    """

    def __init__(self, p: Parametres, fr: Friccio):
        self.parametres, self.friccio = p, fr

        self.g = p.g
        self.L1 = p.L1
        self.m1, self.d1, self.R1 = p.m1, p.d1, p.R1
        self.ms, self.rs = p.ms, p.rs
        self.m2, self.d2, self.R2 = p.m2, p.d2, p.R2
        self.L2_extrem = p.L_extrem

        # Potencial: V = -g*(A*cos th1 + B*cos th2)
        self.A, self.B = p.A, p.B

        # Inercies. La barra 1 amb els cargols i la massa de la 2 penjada al
        # pivot 2; la barra 2 sobre el seu propi pivot.
        self.I1, self.I2 = p.I1, p.I2
        self.acoblament = p.K

        self.tau_f1, self.b1 = fr.tau_f1, fr.b1
        self.tau_f2, self.b2 = fr.tau_f2, fr.b2

        # Fins on arriba cada barra, per a l'integral de l'aire.
        self.barra1_de, self.barra1_a = p.barra1_de, p.barra1_a
        self.barra2_de, self.barra2_a = p.barra2_de, p.barra2_a


def sostres_dels_decaiments() -> dict:
    """La velocitat mes alta que van veure les preses de decaiment de cada rodament.

    De dades/friccio_doble/Friccio_1.*.json (rodament 1: w1) i Friccio_2.*.json
    (rodament 2, amb la barra 1 lligada: w2 - w1 = w2), el maxim de
    energia_sencera.w_max_rad_s. Buit si no hi ha els fitxers.
    """
    sostres = {}
    for quin, nom in (("1", "rodament 1 (w1)"), ("2", "rodament 2 (w2-w1)")):
        w = [json.loads(c.read_text(encoding="utf-8"))["energia_sencera"]["w_max_rad_s"]
             for c in sorted((ARREL / "dades" / "friccio_doble").glob(f"Friccio_{quin}.*.json"))]
        if w:
            sostres[nom] = float(max(w))
    return sostres


def carrega(cami=None) -> Muntatge:
    cami = Path(cami) if cami else ARREL / "parametres.json"
    return Muntatge(Parametres.des_de_json(cami), Friccio.des_de_json(cami))


# -------------------------------------------------------------- les energies


def potencial(th1, th2, p: Muntatge):
    return -p.g * (p.A * np.cos(th1) + p.B * np.cos(th2))


def cinetica(th1, th2, w1, w2, p: Muntatge):
    return (0.5 * p.I1 * w1**2 + 0.5 * p.I2 * w2**2
            + p.acoblament * w1 * w2 * np.cos(th2 - th1))


def energia_rodaments(t, w1, w2, p: Muntatge, factor_carrega_2=None):
    """El que dissipen els dos eixos, integrant la potencia.

    El segon rodament frega contra w2 - w1 i no contra w2: va contra el PRIMER
    BRAC, que es qui li fa de suport.

    Amb factor_carrega_2 (un vector, N/(m2*g) instant a instant), el fregament
    SEC del segon eix creix amb la carrega. Vegeu carrega_del_rodament_2: es una
    hipotesi que es prova amb --centrifuga, no el model per defecte.
    """
    rel = w2 - w1
    sec1 = p.tau_f1 * np.abs(w1)
    vis1 = p.b1 * w1**2
    sec2 = p.tau_f2 * np.abs(rel)
    if factor_carrega_2 is not None:
        sec2 = sec2 * factor_carrega_2
    vis2 = p.b2 * rel**2
    return {
        "sec_1": float(np.trapezoid(sec1, t)),
        "viscos_1": float(np.trapezoid(vis1, t)),
        "sec_2": float(np.trapezoid(sec2, t)),
        "viscos_2": float(np.trapezoid(vis2, t)),
        "total": float(np.trapezoid(sec1 + vis1 + sec2 + vis2, t)),
    }


def carrega_del_rodament_2(th1, th2, w1, w2, p: Muntatge):
    """Quant pesa, en cada instant, el que aguanta el segon rodament.

    HIPOTESI A PROVAR, no un fet establert. El fregament sec va com mu*N*r, o
    sigui que si la carrega N puja, tau_f2 puja amb ella. I a una presa caotica
    la carrega NO es el pes: la barra 2 gira sobre el seu pivot i alhora se la
    passeja el primer brac, i totes dues coses demanen forca centripeta.

        N = m2 * | a_cg + g |      amb a_cg = -L1*w1^2*u1 - d2*w2^2*u2

    Nomes s'hi posen els termes CENTRIPETS: els tangencials demanarien derivar
    les velocitats, que al video es soroll. A velocitat zero N surt m2*g, que es
    el pes, i el factor val 1: el model es redueix al d'ara sense discontinuitat.

    Si aixo es el que passa, tau_f2 creix com w^2 i la seva potencia com w^3 --
    LA MATEIXA SIGNATURA QUE L'AIRE-- i el balanc tal com esta plantejat no els
    pot distingir. Per aixo es una opcio i no el model per defecte.
    """
    ax = -p.L1 * w1**2 * np.sin(th1) - p.d2 * w2**2 * np.sin(th2)
    ay = p.L1 * w1**2 * np.cos(th1) + p.d2 * w2**2 * np.cos(th2)
    return p.m2 * np.hypot(ax, ay + p.g)


def integral_aire(t, th1, th2, w1, w2, p: Muntatge):
    """L'integral K tal que l'energia que s'emporta l'aire val C * K.

    C es el coeficient per unitat de llargada: dF = C * v^2 * ds, amb
    C = 0.5 * rho * Cd * amplada. Aqui no se suposa cap valor: es calcula K i
    despres es resol quina C tanca el balanc.

    Barra 1, que nomes gira, te forma tancada:
        P = C * |w1|^3 * (a^4 + b^4) / 4
    Barra 2 no en te cap, perque cada tros seu va a una velocitat que depen de
    w1, de w2 i de l'angle entre bracos alhora. S'integra numericament.
    """
    a, b = abs(p.barra1_de), abs(p.barra1_a)
    k1 = (a**4 + b**4) / 4.0
    pot1 = k1 * np.abs(w1) ** 3

    u = np.linspace(p.barra2_de, p.barra2_a, PUNTS_BARRA)
    du = u[1] - u[0]
    # v = L1*w1*(cos th1, sin th1) + u*w2*(cos th2, sin th2)
    c1, s1 = np.cos(th1)[:, None], np.sin(th1)[:, None]
    c2, s2 = np.cos(th2)[:, None], np.sin(th2)[:, None]
    vx = p.L1 * w1[:, None] * c1 + u[None, :] * w2[:, None] * c2
    vy = p.L1 * w1[:, None] * s1 + u[None, :] * w2[:, None] * s2
    v = np.hypot(vx, vy)
    pot2 = np.sum(v**3, axis=1) * du

    return {
        "K": float(np.trapezoid(pot1 + pot2, t)),
        "K_barra1": float(np.trapezoid(pot1, t)),
        "K_barra2": float(np.trapezoid(pot2, t)),
    }


def integral_aire_exacta(t, th1, th2, w1, w2, p: Muntatge):
    """La mateixa K, pero amb la quadratura del SIMULADOR (--aire-exacte).

    integral_aire suma v^3 a PUNTS_BARRA punts de la barra 2, extrems inclosos,
    i multiplica per l'interval entre punts: aixo dona un pes de 40/39 al tram i
    compta sencers els dos extrems, i la punta de la barra 2 es on v^3 es mes
    gran. eines/comprova_energia.py ho va mesurar el 27/09/2026: un 5-6 % de
    mes. Aqui la barra 2 va per Gauss-Legendre, amb fisica.potencia_dissipada,
    que es literalment el que integra el simulador.

    NO es el cami per defecte: la C publicada surt de l'altra, i canviar-la es
    una decisio, no una correccio que es fa de passada.
    """
    from pendol.fisica import Dissipacio, _k_barra1, potencia_dissipada

    nomes_aire = Dissipacio(tau_f1=0.0, b1=0.0, tau_f2=0.0, b2=0.0, C=1.0, eps=1.0)
    total = potencia_dissipada((th1, th2, w1, w2), p.parametres, nomes_aire)
    pot1 = _k_barra1(p.parametres) * np.abs(w1) ** 3
    return {
        "K": float(np.trapezoid(total, t)),
        "K_barra1": float(np.trapezoid(pot1, t)),
        "K_barra2": float(np.trapezoid(total - pot1, t)),
        "quadratura": "Gauss-Legendre, la del simulador",
    }


# ------------------------------------------------------------------ el balanc


def balanc(cami_csv, p: Muntatge, cua_s=CUA_S, centrifuga=False, aire_exacte=False):
    import pandas as pd

    d = pd.read_csv(cami_csv)
    t = d["t"].values
    th1, th2 = d["theta1"].values, d["theta2"].values
    w1, w2 = d["omega1"].values, d["omega2"].values
    fps = 1.0 / np.median(np.diff(t))
    n = max(10, int(round(cua_s * fps)))

    E = potencial(th1, th2, p) + cinetica(th1, th2, w1, w2, p)
    E0 = float(E[:max(3, n // 8)].mean())     # el principi: poc, que sigui abans d'amollar
    Ef = float(E[-n:].mean())                 # el final: la cua sencera
    alliberada = E0 - Ef

    fc = None
    if centrifuga:
        fc = carrega_del_rodament_2(th1, th2, w1, w2, p) / (p.m2 * p.g)
    rod = energia_rodaments(t, w1, w2, p, factor_carrega_2=fc)
    aire = (integral_aire_exacta if aire_exacte else integral_aire)(t, th1, th2, w1, w2, p)
    forat = alliberada - rod["total"]

    # L'energia total ha de baixar sempre: tots els mecanismes dissipen. Si puja,
    # o els parametres son dolents o hi ha un error de signe.
    #
    # Els primers segons van a part. Alli el seguidor perd la marca del pivot
    # mentre s'amolla el pendol -al 90_2 se'n va 113 px- i l'energia calculada
    # fa un bot que no es fisic. Barrejar-ho amb la resta convertiria l'avis en
    # soroll: passa a totes les preses i no vol dir que els parametres siguin
    # dolents. Es diu a part i es mesura, que son deces de mJ sobre centenars.
    suau = np.convolve(E, np.ones(n) / n, mode="valid")
    dif = np.diff(suau)
    tall = int(round(3.0 * fps))
    puja = float(np.max(dif[tall:], initial=0.0)) if len(dif) > tall else 0.0
    puja_transitori = float(np.max(dif[:tall], initial=0.0))

    meta = {}
    cami_json = Path(cami_csv).with_suffix(".json")
    if cami_json.exists():
        meta = json.loads(cami_json.read_text(encoding="utf-8"))
    geo = meta.get("geometria", {})

    return {
        "presa": Path(cami_csv).stem,
        "durada_s": float(t[-1]),
        "angles_inicials_graus": [float(np.degrees(th1[0])), float(np.degrees(th2[0]))],
        "quiet_al_principi_s": geo.get("quiet_al_principi_s"),
        # Si arrenca del repos ho diu l'amollada (None si la presa no la porta).
        "arrenca_del_repos": amollada.arrenca_del_repos(meta),
        "motiu_sense_amollada": (meta.get("amollada") or {}).get("motiu"),
        "alliberada_J": alliberada,
        "rodaments": rod,
        "carrega_2": None if fc is None else {
            "factor_mitja": float(fc.mean()), "factor_max": float(fc.max())},
        "velocitats": {
            "w1_rms": float(np.sqrt((w1**2).mean())),
            "w1_p90": float(np.percentile(np.abs(w1), 90)),
            "w1_max": float(np.abs(w1).max()),
            "rel_rms": float(np.sqrt(((w2 - w1)**2).mean())),
            "rel_p90": float(np.percentile(np.abs(w2 - w1), 90)),
            "rel_max": float(np.abs(w2 - w1).max()),
        },
        "fraccio_rodaments": rod["total"] / alliberada if alliberada else float("nan"),
        "forat_J": forat,
        "K_aire": aire,
        "C_que_tanca": forat / aire["K"] if aire["K"] else float("nan"),
        "puja_energia_J_per_mostra": puja,
        "puja_al_transitori_J": puja_transitori,
    }


# ---------------------------------------------------------------- autoprova


def autoprova() -> int:
    """Un pendol fisic SIMPLE amb friccio coneguda: el balanc l'ha de tancar.

    No cal el pendol doble per provar la comptabilitat. S'integra un cos rigid
    sobre un eix fix amb fregament sec mes viscos, i es comprova que la caiguda
    d'energia potencial coincideix amb la integral de la potencia dissipada. Si
    aixo no quadra, res del que hi ha a sobre no val.
    """
    I, Mgd, tau_f, b = 2.971e-3, 0.19075, 2.10e-4, 8.34e-5
    print("Pendol fisic simple, amollat de 60 graus i integrat fins que s'atura.")
    print(f"  I = {I:.4e} kg m^2   M*g*d = {Mgd:.5f} N m")
    print(f"  tau_f = {tau_f:.2e} N m   b = {b:.2e} N m s/rad\n")

    def f(_, y):
        th, w = y
        # tanh regularitza el signe del fregament sec: sense aixo l'integrador
        # pica contra la discontinuitat a w = 0 i no acaba mai.
        return [w, (-Mgd * math.sin(th) - tau_f * math.tanh(w / 1e-4) - b * w) / I]

    th0 = math.radians(60.0)
    s = solve_ivp(f, (0, 400), [th0, 0.0], max_step=0.002, rtol=1e-10, atol=1e-12,
                  dense_output=False)
    t, th, w = s.t, s.y[0], s.y[1]

    baixada = Mgd * (math.cos(th[-1]) - math.cos(th0)) - 0.5 * I * w[-1] ** 2
    dissipada = np.trapezoid(tau_f * np.abs(w) + b * w**2, t)
    error = abs(dissipada - baixada) / baixada

    print(f"  energia que ha baixat  : {baixada*1000:9.4f} mJ")
    print(f"  potencia dissipada     : {dissipada*1000:9.4f} mJ")
    print(f"  error relatiu          : {error:9.2e}")
    print(f"  amplitud final         : {math.degrees(abs(th[-1])):9.3f} graus\n")

    if error < 5e-3:
        print("OK -> la comptabilitat d'energia tanca a millor del 0,5 %.")
        print("     Limit conegut: aixo prova la INTEGRACIO, no els parametres del")
        print("     muntatge ni el model d'aire, que no es poden provar sols.")
        return 0
    print("ATENCIO -> el balanc no tanca. Hi ha un error de signe o d'unitats.")
    return 1


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("entrades", nargs="*", help="CSV canonics (accepta comodins)")
    ap.add_argument("--parametres", help="un parametres.json alternatiu")
    ap.add_argument("--cua", type=float, default=CUA_S,
                    help=f"segons de final que es promitgen (per defecte {CUA_S})")
    ap.add_argument("--totes", action="store_true",
                    help="inclou les preses que no arrenquen del repos (l'energia "
                         "inicial hi surt de la velocitat mesurada, mes sorollosa)")
    ap.add_argument("--centrifuga", action="store_true",
                    help="HIPOTESI: el fregament sec del segon eix creix amb la "
                         "carrega, i a alta velocitat la carrega es centripeta i no "
                         "el pes. Te la mateixa signatura que l'aire; aixo mira si "
                         "en tapa el forat. Vegeu carrega_del_rodament_2")
    ap.add_argument("--aire-exacte", action="store_true",
                    help="la K de l'aire amb la quadratura del simulador (Gauss) en "
                         "lloc de la suma de 40 punts, que la sobreestima un 5-6 %%. "
                         "NO es el cami per defecte: la C publicada surt de l'altre")
    ap.add_argument("--json", help="on desar els numeros")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.entrades:
        ap.error("digue'm quins CSV he de mirar")

    camins = sorted({c for patro in args.entrades for c in glob.glob(patro)})
    if not camins:
        raise SystemExit(f"Cap fitxer no coincideix amb {args.entrades}")

    p = carrega(args.parametres)
    print(f"Muntatge: A = {p.A:.5f} kg m   B = {p.B:.6f} kg m   "
          f"I1 = {p.I1:.4e}   I2 = {p.I2:.4e}\n")

    # QUI ENTRA. Des del 24/09/2026 ho decideix l'amollada desada a les
    # metadades de cada presa (amollada.arrenca_del_repos): si no s'hi troba,
    # la presa no arrenca del repos i l'energia inicial no es coneix sense
    # derivar la velocitat. Abans ho decidia un llindar sobre el detector de
    # pixels, que es disparava amb una sola sacsejada del seguidor; aquell
    # criteri nomes queda per a les preses que encara no porten l'amollada.
    #
    # I DES DEL 26/09/2026 una presa pot estar marcada «fora del balanc» a
    # dades/retalls.json, amb el motiu (la 45_7: un fotograma trencat en ple
    # moviment). La marca arriba a les metadades per la conversio i aqui no
    # entra MAI, ni amb --totes: no es un criteri de repos, es una presa dolenta.
    fets, fora, criteri_vell, descartades = [], [], [], []
    for c in camins:
        cami_meta = Path(c).with_suffix(".json")
        meta_c = json.loads(cami_meta.read_text(encoding="utf-8")) if cami_meta.exists() else {}
        if meta_c.get("fora_del_balanc"):
            descartades.append((Path(c).stem, meta_c["fora_del_balanc"]))
            continue
        r = balanc(c, p, args.cua, centrifuga=args.centrifuga,
                   aire_exacte=args.aire_exacte)
        repos = r["arrenca_del_repos"]
        motiu = r["motiu_sense_amollada"]
        if repos is None:
            q = r["quiet_al_principi_s"]
            repos = not (q is not None and q < QUIET_VELL_S)
            motiu = f"detector de pixels, {q:.2f} s de quietud" if q is not None else None
            criteri_vell.append(r["presa"])
        if not repos and not args.totes:
            fora.append((r["presa"], motiu))
            continue
        fets.append(r)

    print(f"{'presa':<8} {'alliberada':>11} {'rodaments':>11} {'%':>6} {'forat':>10} {'C':>9}")
    print("-" * 60)
    for r in fets:
        print(f"{r['presa']:<8} {r['alliberada_J']*1000:9.1f} mJ "
              f"{r['rodaments']['total']*1000:9.1f} mJ {100*r['fraccio_rodaments']:5.1f} "
              f"{r['forat_J']*1000:8.1f} mJ {r['C_que_tanca']:9.5f}")

    # El desglossament per mecanisme. Els quatre primers surten d'INTEGRAR la
    # potencia de cada terme al llarg de la trajectoria; el cinque NO es mesura:
    # es el forat que queda, i se li atribueix a l'aire. Per aixo va separat amb
    # una barra a la capcalera -les quatre primeres columnes son un calcul i
    # l'ultima es un residu-, i per aixo aquesta taula no pot dir mai que l'aire
    # estigui mesurat. Nomes diu quant hauria d'explicar.
    print()
    print(f"{'presa':<8} {'sec 1':>8} {'visc 1':>8} {'sec 2':>8} {'visc 2':>8} "
          f"{'| aire':>9}   (% de l'alliberada)")
    print("-" * 60)
    for r in fets:
        e, rd = r["alliberada_J"], r["rodaments"]
        print(f"{r['presa']:<8} "
              f"{100*rd['sec_1']/e:7.1f}% {100*rd['viscos_1']/e:7.1f}% "
              f"{100*rd['sec_2']/e:7.1f}% {100*rd['viscos_2']/e:7.1f}% "
              f"|{100*r['forat_J']/e:7.1f}%")
    parts = {
        "sec 1": np.array([r["rodaments"]["sec_1"] / r["alliberada_J"] for r in fets]),
        "viscos 1": np.array([r["rodaments"]["viscos_1"] / r["alliberada_J"] for r in fets]),
        "sec 2": np.array([r["rodaments"]["sec_2"] / r["alliberada_J"] for r in fets]),
        "viscos 2": np.array([r["rodaments"]["viscos_2"] / r["alliberada_J"] for r in fets]),
        "aire": np.array([r["forat_J"] / r["alliberada_J"] for r in fets]),
    }
    print("-" * 60)
    print(f"{'mitjana':<8} " + " ".join(f"{100*v.mean():7.1f}%" for v in parts.values()))
    # Qui mana s'ha de mirar per MECANISME, no per columna: el sec i el viscos
    # hi surten dos cops -un per rodament- i l'aire nomes un, o sigui que
    # comparar les cinc columnes entre elles faria guanyar l'aire tot sol.
    classes = {
        "SEC": parts["sec 1"] + parts["sec 2"],
        "VISCOS": parts["viscos 1"] + parts["viscos 2"],
        "AIRE": parts["aire"],
    }
    print("\nper mecanisme: " + "   ".join(
        f"{k} {100*v.mean():.1f} %" for k, v in classes.items()))
    ordre = sorted(classes, key=lambda k: -classes[k].mean())
    dalt, segon = classes[ordre[0]].mean(), classes[ordre[1]].mean()
    if (dalt - segon) / dalt < 0.10:
        print(f"   CAP no mana: {ordre[0]} i {ordre[1]} van a menys d'un 10 % l'un "
              f"de l'altre.")
        print("   En aquest regim els tres mecanismes son del mateix ordre, i "
              "cap model")
        print("   que en deixi un fora no pot descriure el pendol.")
    else:
        print(f"   mana el {ordre[0]}, amb el {100*dalt:.1f} % de l'energia "
              f"alliberada de mitjana.")

    if fora:
        print("\nFora, perque no s'hi troba l'amollada -no arrenquen del repos- i "
              "l'energia inicial no es coneix:")
        for n, motiu in fora:
            print(f"  {n}: {motiu}")
        print("  (--totes les inclou, traient l'energia inicial de la velocitat)")
    if descartades:
        print("\nFora del balanc per decisio escrita a dades/retalls.json:")
        for n, motiu in descartades:
            print(f"  {n}: {motiu}")
    if criteri_vell:
        print(f"\nSense l'amollada a les metadades (convertides abans del 24/09/2026), "
              f"classificades amb el detector de pixels vell ({1000*QUIET_VELL_S:.0f} ms): "
              + ", ".join(criteri_vell))
        print("  Torna-les a convertir amb eines/converteix_preses.py.")

    C = np.array([r["C_que_tanca"] for r in fets])
    frac = np.array([r["fraccio_rodaments"] for r in fets])
    print()
    print(f"rodaments: expliquen del {100*frac.min():.0f} % al {100*frac.max():.0f} % "
          f"(mitjana {100*frac.mean():.0f} %)")
    if len(C) > 1:
        disp = C.std(ddof=1) / abs(C.mean())
        print(f"aire: C = {C.mean():.5f} +- {C.std(ddof=1):.5f}  "
              f"(dispersio {100*disp:.0f} % sobre {len(C)} preses)")
        # De parametres.json i no escrit a ma: fins al 29/09 hi deia «0,0099 per a
        # una barra plana d'1,5 cm», que era d'abans de mesurar el gruix.
        print(f"      valor de llibre (parametres.json, gruix de {100*p.friccio.gruix:.1f} cm): "
              f"{p.friccio.C_llibre:.4f}")
        print()
        if disp < 0.25:
            print("OK -> un sol coeficient tanca totes les preses: es compatible amb")
            print("     un unic mecanisme que va com v^3, i l'aire ho es.")
        elif disp < 0.50:
            print("DUBTOS -> la dispersio es massa gran per dir que un sol mecanisme")
            print("          les expliqui totes, i massa petita per descartar-ho.")
        else:
            print("ATENCIO -> el coeficient no es el mateix a cada presa. El forat NO")
            print("           s'explica amb un sol mecanisme que vagi com v^3.")

    puja = max(r["puja_energia_J_per_mostra"] for r in fets)
    trans = max(r["puja_al_transitori_J"] for r in fets)
    mitjana = float(np.mean([r["alliberada_J"] for r in fets]))
    # Fins on val l'extrapolacio. Els coeficients de friccio es van mesurar en
    # dues preses de decaiment i s'apliquen aqui: la pregunta que ningu no havia
    # fet es si aquestes preses es mouen dins d'aquell rang de velocitats o fora.
    #
    # El sostre de cada rodament es la velocitat mes alta que van arribar a
    # veure les seves preses de decaiment, amb l'ajust per energia sobre el
    # registre sencer (dades/friccio_doble/*.json, energia_sencera). Fins al
    # 29/09 era 55 graus * 2*pi/T0 escrit a ma -7,7 i 9,2 rad/s-, que era el
    # sostre del metode vell dels cims, i la conclusio de sota tambe estava
    # escrita a ma. Ara totes dues coses surten dels fitxers.
    SOSTRE = sostres_dels_decaiments()
    v = {k: np.array([r["velocitats"][k] for r in fets])
         for k in ("w1_rms", "w1_p90", "w1_max", "rel_rms", "rel_p90", "rel_max")}
    print("\nfins on val l'extrapolacio (rad/s):")
    print(f"   {'':<20} {'decaiment':>10} {'| rms':>8} {'p90':>8} {'max':>8}")
    for nom, pre in (("rodament 1 (w1)", "w1"), ("rodament 2 (w2-w1)", "rel")):
        if nom not in SOSTRE:
            print(f"   {nom:<20} {'?':>10}   (no hi ha preses de decaiment a dades/friccio_doble/)")
            continue
        print(f"   {nom:<20} {SOSTRE[nom]:10.1f} |{v[pre+'_rms'].mean():7.1f} "
              f"{v[pre+'_p90'].mean():8.1f} {v[pre+'_max'].max():8.1f}")
    for nom, pre in (("rodament 1 (w1)", "w1"), ("rodament 2 (w2-w1)", "rel")):
        if nom in SOSTRE:
            fora = 100 * np.mean(v[pre + "_p90"] > SOSTRE[nom])
            print(f"   {nom}: el percentil 90 passa del sostre al {fora:.0f} % de les preses"
                  + ("" if fora else " -> el decaiment cobreix el seu regim"))
    print("   Tot el que un terme es deixi per sobre del seu sostre cau al forat.")

    if args.centrifuga:
        fm = np.array([r["carrega_2"]["factor_mitja"] for r in fets])
        fx = np.array([r["carrega_2"]["factor_max"] for r in fets])
        print(f"\ncentrifuga: la carrega del rodament 2 val de mitjana "
              f"{fm.mean():.2f} vegades el pes")
        print(f"            (per presa, de {fm.min():.2f} a {fm.max():.2f}; "
              f"puntes de fins a {fx.max():.1f})")
        print("            NOMES compara-ho amb la correguda sense --centrifuga:")
        print("            si la dispersio de la C baixa clarament, la hipotesi")
        print("            guanya; si no es mou, no es pot distingir de l'aire.")

    print(f"\ncoherencia: passats els 3 primers segons l'energia total no puja mai mes")
    print(f"            de {puja*1000:.3f} mJ, que es el que ha de passar en un sistema")
    print(f"            que nomes dissipa. Als 3 primers puja fins a {trans*1000:.2f} mJ")
    print(f"            ({100*trans/mitjana:.2f} % del total): alli el seguidor perd la")
    print(f"            marca del pivot mentre s'amolla, i no es cap error de model.")
    # El llindar es RELATIU. Un absolut no vol dir res: amb un soroll angular
    # de 0,3 graus, l'energia potencial ja balla g*A*dtheta = 0,8 mJ tota sola,
    # o sigui que una pujada de deces de mJ es soroll de mesura i prou.
    if puja > 0.005 * mitjana:
        print(f"ATENCIO -> la pujada de fora del transitori es el "
              f"{100*puja/mitjana:.1f} % de l'energia alliberada, massa per ser soroll.")
        print("           Mira els parametres o el signe d'algun terme.")

    if args.json:
        desti = Path(args.json)
        desti.parent.mkdir(parents=True, exist_ok=True)
        desti.write_text(json.dumps(fets, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nNumeros desats a {desti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
