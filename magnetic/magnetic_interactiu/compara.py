"""El JavaScript diu el mateix que el Python? La prova que decideix tota la resta.

PER QUE EXISTEIX AQUEST FITXER
------------------------------
La pagina interactiva ha de poder recalcular el mapa dins del navegador, i aixo
vol dir tenir el nucli escrit dues vegades: en Python (magnetic/motor.py, el que
mana i el que es valida) i en JavaScript (motor.js, la copia). Dues
implementacions de la mateixa fisica es la manera classica de tenir, al cap d'un
mes, dos models lleugerament diferents sense saber-ho.

Amb Numba aixo no passava: alla no hi ha dues versions del nucli, n'hi ha una de
sola amb el compilador engegat o apagat (vegeu magnetic/compila.py). Aqui si que
n'hi ha dues de debo, i per aixo la comparacio ha de ser MES exigent, no menys.

L'ordre es: primer aixo, i nomes despres la pagina. Si el motor de JavaScript no
passa aquesta prova, tot el que es construeixi a sobre no val res.

LES QUATRE PROVES
-----------------
0. delta^(n+1) tota sola. La sospitosa numero u, mesurada abans de res.
1. Trajectoria a curt termini: l'estat despres d'1, 10, 100 i 1.000 passos de
   RK4 des de cinc punts. Aqui s'exigeix coincidencia al bit o a un parell
   d'ulp: es el que demostra que les equacions son les mateixes.
2. Desenllacos d'una graella d'uns 2.000 punts, SEPARATS EN DOS: dins de conca
   (ha de coincidir el 100 %) i frontera (hi haura diferencies, i es reporta
   quantes). Es miren les dues meitats del desenllac, el desti I el temps de
   caiguda, perque un temps diferent dins d'una conca voldria dir que els dos
   comptadors de passos no van sincronitzats.
3. Punts de repos i estabilitat del centre per a diverses parelles (n, K).

PER QUE LA PROVA 2 VA SEPARADA EN DOS -- I QUE VOL DIR EL NUMERO DE LA FRONTERA
------------------------------------------------------------------------------
El Python calcula delta^(n+1) amb el pow de la biblioteca de C i el JavaScript
amb el Math.pow de V8. Cap de les dues implementacions no esta obligada a
arrodonir igual, o sigui que una diferencia a l'ultim bit es INEVITABLE. I en un
sistema caotic, l'ultim bit acaba canviant el desti d'alguns punts.

Esperar que surti perfecte a tot arreu es no haver entes el problema. El que si
que s'ha d'exigir es que les diferencies quedin NOMES a la frontera: si en
canvia cap de dins d'una conca, les dues implementacions no son la mateixa
fisica i s'ha d'anar a buscar l'error.

El numero de la frontera es exactament el mateix fenomen que magnetic/converge.py
ja mesura amb dt/2 (0,02 % dins i 13,27 % a la frontera, a la mateixa finestra
de +-7 cm). Per aixo la finestra per defecte d'aqui es aquella: els dos numeros
s'han de poder posar de costat a la memoria.

    .venv\\Scripts\\python.exe -m magnetic.magnetic_interactiu.compara

Cal node instal.lat (winget install OpenJS.NodeJS.LTS). No toca el .venv.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

from magnetic import compila
from magnetic.analisi import (
    compta_capses,
    dimensio_per_box_counting,
    mascara_de_frontera,
    pendents_locals,
    percentatge_de_frontera,
)
from magnetic.fisica import (Parametres, avis_inventats, factor_critic,
                             punts_dequilibri, sobreescriu)
from magnetic.mapa import calcula
from magnetic.motor import (
    DT_PER_DEFECTE, Criteri, classifica, destins, graella_nucli, pas_rk4,
)

ARREL = Path(__file__).resolve().parent
PROVA_JS = ARREL / "prova.js"
PROVA_PAGINA_JS = ARREL / "prova_pagina.js"
PAGINA_HTML = ARREL / "mapa-interactiu.html"

# La finestra de la prova 4 es la del zoom2: conques i frontera barrejades, que
# es on una diferencia de calcul es veuria. Des del 28/10 es el zoom2 NOU: el
# vell (0,47-1,97 x 1,99-3,49), triat sobre el mapa de n = 3, amb la llei
# mesurada cau a la pols i nomes 235 dels 22.801 punts eren de dins d'una conca.
FINESTRA_PAGINA_CM = (1.3057, 2.8057, 0.3030, 1.8030)

# Quant es pot apartar l'estat despres de N passos de RK4. El pla en deia "al
# bit, o a un parell d'ulp": aquest es aquell parell.
LLINDAR_ULP = 2

# Quant es poden apartar els punts de repos. La biseccio del JavaScript i el
# fsolve del Python son dos algorismes diferents buscant la mateixa arrel, o
# sigui que aqui NO es pot demanar coincidencia al bit; el que es demana es que
# la diferencia sigui irrellevant al costat del radi de captura, que es de
# 8 mm. Un nanometre ja ho es un mili de vegades.
LLINDAR_REPOS_M = 1e-9


# ---------------------------------------------------------------------------
# Comparar dos doubles: ulps
# ---------------------------------------------------------------------------
def _ordinal(x: float) -> int:
    """Els bits d'un double llegits com a enter, arreglats perque siguin ordenats.

    Els doubles consecutius tenen patrons de bits consecutius, i per aixo restar
    els dos enters diu QUANTS DOUBLES hi ha entremig: aixo es un ulp. Es la
    manera honesta de dir "gairebe igual" quan els numeros van de 1e-4 a 1e+2,
    perque no depen de l'escala com si que en depen una diferencia absoluta.

    L'arreglada del signe es perque els negatius, llegits com a enters, van a
    l'inreves.
    """
    (s,) = struct.unpack("<q", struct.pack("<d", float(x)))
    return s if s >= 0 else -(s & 0x7FFFFFFFFFFFFFFF)


def dif_ulps(a: float, b: float) -> int:
    return abs(_ordinal(a) - _ordinal(b))


def ulps_a_escala(py: list, js: list) -> list:
    """Quant s'aparten dos estats, en ulps de la PROPIA ESCALA de l'estat.

    PER QUE NO N'HI HA PROU AMB ELS ULPS DE CADA COMPONENT. Un dels punts de
    prova surt de (1 cm, 0), que es damunt d'un eix de simetria del muntatge:
    amb tres imants a 0, 120 i 240 graus, l'eix de les x es un mirall. La y
    d'aquella trajectoria val zero EXACTAMENT en aritmetica de debo, i el que
    surt del calcul es nomes escuma d'arrodoniment: 1,68e-18.

    Comparar els ulps de dues escumes dona numeros astronomics -- 1,4e13 ulp --
    perque a prop del zero els doubles son densissims, mentre que la diferencia
    absoluta es de 3e-21 m, o sigui res. El numero gros no vol dir que els dos
    programes discrepin: vol dir que se'ls esta preguntant quants doubles hi ha
    entre dos zeros.

    La mesura que si que vol dir alguna cosa es aquesta: la diferencia entre les
    dues components, dividida per la mida de l'ultim bit de L'ESTAT SENCER. La
    posicio es mesura contra la posicio mes gran i la velocitat contra la
    velocitat mes gran, perque son magnituds diferents i no es poden barrejar.
    Aixi "menys d'1 ulp" vol dir el que sembla que vol dir: que les dues
    trajectories son la mateixa fins a l'ultim bit que l'estat pot representar.

    Els ulps component a component es continuen desant al JSON, que per a les
    components que no son cap zero de simetria son la mesura mes fina.
    """
    escala_pos = max(abs(py[0]), abs(py[1]), abs(js[0]), abs(js[1]))
    escala_vel = max(abs(py[2]), abs(py[3]), abs(js[2]), abs(js[3]))
    fora = []
    for i in range(4):
        escala = escala_pos if i < 2 else escala_vel
        pas = math.ulp(escala) if escala > 0.0 else math.ulp(1.0)
        fora.append(abs(py[i] - js[i]) / pas)
    return fora


# ---------------------------------------------------------------------------
# Parlar amb node
# ---------------------------------------------------------------------------
def busca_node(cami=None) -> str:
    """On es el node. El winget el deixa al PATH, pero no fins que no obris una
    consola nova, i per aixo tambe es mira on l'instal.la."""
    if cami:
        return cami
    trobat = shutil.which("node")
    if trobat:
        return trobat
    for c in (r"C:\Program Files\nodejs\node.exe", r"C:\Program Files (x86)\nodejs\node.exe"):
        if Path(c).exists():
            return c
    raise SystemExit(
        "No s'ha trobat node.\n"
        "   Instal.la'l amb:  winget install OpenJS.NodeJS.LTS\n"
        "   o digues on es:   --node C:\\ruta\\a\\node.exe\n"
        "   No toca el .venv ni res del projecte."
    )


def crida_node(node: str, feines: list, potencia_rapida: bool = False) -> list:
    """Passa les feines a prova.js i torna els resultats.

    L'error estandard del node NO es captura a posta: aixi les linies de
    progres surten en directe en comptes de totes de cop al final.
    """
    entrada = json.dumps({"potencia_rapida": potencia_rapida, "feines": feines})
    r = subprocess.run(
        [node, str(PROVA_JS)], input=entrada, stdout=subprocess.PIPE,
        text=True, encoding="utf-8",
    )
    if r.returncode != 0:
        raise RuntimeError(f"node ha acabat amb el codi {r.returncode}. Mira l'error de sobre.")
    return json.loads(r.stdout)["resultats"]


def _parametres_json(p: Parametres) -> dict:
    """Els parametres tal com els vol prova.js. Els noms son els del Python."""
    return {
        "L": p.L, "g": p.g, "b": p.b, "n": p.n, "d0": p.d0,
        "im_x": [float(v) for v in p.im_x],
        "im_y": [float(v) for v in p.im_y],
        "im_K": [float(v) for v in p.im_K],
    }


# ---------------------------------------------------------------------------
# Prova 0 -- delta^(n+1) tota sola
# ---------------------------------------------------------------------------
# El rang va de 1,5 cm (la massa gairebe a sobre d'un imant, comptant que
# sempre queda d0 = 2 cm d'alcada) a 30 cm (la vora de tot el que el fil
# permet). Son les distancies que apareixen de debo al mapa.
BASES_POTENCIA = np.linspace(0.015, 0.30, 20001).tolist()


def prova_potencia(node: str, p: Parametres) -> dict:
    exponent = p.n + 1.0
    python = [d ** exponent for d in BASES_POTENCIA]

    resultat = {}
    for etiqueta, rapid in (("Math.pow", False), ("multiplicacions", True)):
        js = crida_node(
            node, [{"tipus": "potencia", "bases": BASES_POTENCIA, "exponent": exponent}], rapid
        )[0]["valors"]
        difs = [dif_ulps(a, b) for a, b in zip(python, js) if a != b]
        resultat[etiqueta] = {
            "diferents": len(difs),
            "total": len(python),
            "percent": 100.0 * len(difs) / len(python),
            "pitjor_ulp": max(difs) if difs else 0,
        }
    return resultat


# ---------------------------------------------------------------------------
# Prova 1 -- trajectoria a curt termini
# ---------------------------------------------------------------------------
# Cinc punts triats per cobrir els regims diferents del mapa, no a l'atzar: si
# nomes es provessin punts tranquils, la prova passaria sense voler dir res.
PUNTS_TRAJECTORIA = (
    ("a prop del centre", 0.010, 0.000),
    ("zona de conques", 0.030, 0.020),
    ("arran d'un imant", 0.045, 0.005),
    ("vora de la pols", -0.060, 0.040),
    ("dins de la pols", 0.090, -0.050),
)
PASSOS_TRAJECTORIA = (1, 10, 100, 1000)

# L'escala per buscar ON NEIX la diferencia. Va de l'1 fins passat el llindar.
PASSOS_NAIXEMENT = (1, 2, 3, 5, 10, 25, 50, 100, 150, 200, 300, 400, 500, 700, 1000)

# EL CRITERI, I PER QUE NO ES UN LLINDAR D'ULPS.
#
# Fins al 2026-09-29 la prova 1 exigia menys de 2 ulp despres de 1.000 passos, i
# el CLAUDE.md tenia escrit que donava "0,00 ulp, identic bit a bit". Va deixar
# de ser cert en MESURAR la b: amb la inventada (0,15) la massa de "vora de la
# pols" ja s'estava morint als 2 s; amb la de debo (0,039) encara vaga per la
# pols, que es on un sol ulp del Math.pow es multiplica sol. Mesurat:
#
#     b = 0,150 -> 0,00 ulp     b = 0,100 -> 0,00     b = 0,060 -> 0,00
#     b = 0,039 -> 4,00 ulp   <- la mesurada
#
# O sigui que aquell llindar no mesurava el codi, mesurava la friccio. El que de
# debo distingeix "el mateix codi" de "codi diferent" es QUAN NEIX la
# diferencia: una aritmetica diferent es veu al PRIMER PAS, i el caos triga
# centenars.
#
# ON ES POSA LA RATLLA, I PER QUE A 100. Amb --escombra-b es veu a quin pas neix
# la diferencia a quatre friccions, i NO TE CAP ORDRE:
#
#     b = 0,150 -> neix al pas  100      b = 0,100 -> 1.000
#     b = 0,060 -> neix al pas  500      b = 0,039 -> 700   <- la mesurada
#
# No es que amb menys friccio neixi abans: depen de per on passa la
# trajectoria, que es el que vol dir caos. Per tant la ratlla NO es pot posar
# arran del que dona avui, perque el dia que es mesurin n i K la dinamica
# canviara i el numero es tornara a moure.
#
# Amb 100 passen les tres friccions que es fan servir o s'han considerat, i
# nomes hi falla la 0,150, que es la INVENTADA i ja no s'usa. Es va considerar
# 500 -passa nomes amb la mesurada- i 50 -passa totes quatre-; es tria 100 com
# a terme mig, I QUEDA DECLARAT que a b = 0,150 fallaria.
#
# El marge cap a l'altra banda es el que importa: el control positiu salta al
# pas 1-3, o sigui trenta vegades abans. Un error de debo no s'escapa.
PASSOS_IDENTICS = 100

# QUAN ES DIU QUE LA DIFERENCIA HA NASCUT. Fins al 26/10/2026 n'hi havia prou amb
# qualsevol diferencia (> 0). Amb dt = 0,0005 s va fallar al punt "a prop del
# centre", que surt de damunt de l'eix de simetria: la seva y i la seva vy valen
# zero de debo, i el que hi queda es escuma d'arrodoniment (y ~ 7,5e-20 m). Alla
# els dos motors diferien des del pas 100 en 0,001 ulp de l'escala de l'estat,
# mentre la x i la vx eren identiques bit a bit fins al pas 1.000, igual que els
# altres quatre punts (esborranys/_compara_neix_26-10.py).
#
# Es exactament el cas per al qual existeix ulps_a_escala(), pero el > 0 el
# tornava a fer comptar. Des del 26/10, per decisio d'en Guillem i DESPRES DE
# FALLAR, la diferencia neix quan arriba a 1 ulp de l'escala de l'estat: "menys
# d'1 ulp" vol dir identics fins a l'ultim bit que l'estat pot representar.
# El preu, mesurat: el control positiu se'n adona al primer pas amb dos punts
# de cinc en comptes de tres.
ULP_NAIXEMENT = 1.0

# El nombre de passos dins del qual el control positiu s'ha de detectar. Amb una
# K moguda un sol ulp -la discrepancia mes petita que pot existir- tres dels cinc
# punts ja discrepaven al primer pas i el del centre ho feia per 4 ulp (amb el
# criteri > 0 i els parametres d'abans). Amb el llindar d'1 ulp i els del 26/10,
# en son dos: el del centre (2 ulp) i el d'arran d'un imant (1 ulp).
PASSOS_CONTROL = 5


def _passos_python(p: Parametres, x0: float, y0: float, dt: float, quants: int) -> list:
    x, y, vx, vy = x0, y0, 0.0, 0.0
    for _ in range(quants):
        x, y, vx, vy = pas_rk4(x, y, vx, vy, dt, p.L, p.g, p.b, p.n, p.d0,
                               p.im_x, p.im_y, p.im_K)
    return [float(x), float(y), float(vx), float(vy)]


def prova_trajectoria(node: str, p: Parametres, dt: float) -> list:
    feines, esperat = [], []
    for etiqueta, x0, y0 in PUNTS_TRAJECTORIA:
        for quants in PASSOS_TRAJECTORIA:
            feines.append({
                "tipus": "passos", "parametres": _parametres_json(p),
                "estat": [x0, y0, 0.0, 0.0], "dt": dt, "passos": quants,
            })
            esperat.append((etiqueta, x0, y0, quants, _passos_python(p, x0, y0, dt, quants)))

    resultats = crida_node(node, feines)

    files = []
    for (etiqueta, x0, y0, quants, py), js in zip(esperat, resultats):
        escalats = ulps_a_escala(py, js["estat"])
        files.append({
            "punt": etiqueta, "x0": x0, "y0": y0, "passos": quants,
            "pitjor_ulp": max(escalats),
            "ulps_a_escala": escalats,
            "ulps_component_a_component": [dif_ulps(a, b) for a, b in zip(py, js["estat"])],
            "python": py, "js": js["estat"],
        })
    return files


def _naixements(node: str, p: Parametres, dt: float, p_node: Parametres = None) -> dict:
    """A quin pas neix la diferencia, a cada punt. None vol dir "a cap".

    p_node, si s'hi posa, son uns parametres DIFERENTS per al costat del node:
    es com es fabrica el control positiu sense tocar cap codi.
    """
    dels_altres = p if p_node is None else p_node
    feines, esperat = [], []
    for etiqueta, x0, y0 in PUNTS_TRAJECTORIA:
        for quants in PASSOS_NAIXEMENT:
            feines.append({
                "tipus": "passos", "parametres": _parametres_json(dels_altres),
                "estat": [x0, y0, 0.0, 0.0], "dt": dt, "passos": quants,
            })
            esperat.append((etiqueta, x0, y0, quants))

    resultats = crida_node(node, feines)
    neix = {etiqueta: None for etiqueta, _, _ in PUNTS_TRAJECTORIA}
    for (etiqueta, x0, y0, quants), js in zip(esperat, resultats):
        if neix[etiqueta] is not None:
            continue
        py = _passos_python(p, x0, y0, dt, quants)
        if max(ulps_a_escala(py, js["estat"])) >= ULP_NAIXEMENT:
            neix[etiqueta] = quants
    return neix


FRICCIONS_ESCOMBRAT = (0.150, 0.100, 0.060, 0.039)


def escombra_de_friccio(node: str, p: Parametres, dt: float) -> int:
    """Per que la prova 1 va deixar de ser un llindar d'ulps. S'hi arriba amb
    --escombra-b.

    NO ES UN LUXE, i no s'ha de treure encara que sembli que no prova res: es
    l'unica manera de tornar a treure els numeros que hi ha escrits al
    magnetic/CLAUDE.md, i la norma del projecte diu que un resultat que no es
    pot tornar a calcular no es un resultat.

    El que ensenya: amb la b inventada (0,15) els dos motors donen 0,00 ulp als
    1.000 passos, i amb la mesurada (0,039) en donen 4,00. El codi es el mateix
    a totes quatre files. O sigui que aquell "llindar de 2 ulp" no mesurava si
    les dues implementacions feien la mateixa aritmetica: mesurava quanta
    friccio hi havia. Per aixo ara el criteri es a quin pas NEIX la diferencia.
    """
    print("-" * 72)
    print("El llindar d'ulps mesurava la FRICCIO, no el codi")
    print("-" * 72)
    print("\n   El mateix codi a totes les files; nomes canvia la b.\n")
    print(f"   {'b':>8} {'ulp a 1.000 passos':>20} {'neix al pas':>13}   que es")
    for b in FRICCIONS_ESCOMBRAT:
        q, _ = sobreescriu(p, b=b)
        files = prova_trajectoria(node, q, dt)
        pitjor = max(f["pitjor_ulp"] for f in files)
        neix = _naixements(node, q, dt)
        primer = min((v for v in neix.values() if v is not None), default=None)
        quin = "la MESURADA" if abs(b - p.b) < 1e-12 else "inventada" if b > 0.14 else ""
        print(f"   {b:>8.3f} {pitjor:>20.2f} {(str(primer) if primer else 'mai'):>13}"
              f"   {quin}")
    print("\n   Amb la b inventada la massa de «vora de la pols» ja s'estava morint")
    print("   als 2 s; amb la de debo encara hi vaga, que es on un sol ulp del")
    print("   Math.pow es multiplica sol. El codi no s'ha mogut.")
    return 0


def control_de_la_prova_1(node: str, p: Parametres, dt: float) -> dict:
    """Comprova que la prova 1 NO ES CEGA, i es torna a fer cada vegada.

    Es mou la K UN SOL ULP nomes al costat del node -- la discrepancia mes petita
    que pot existir entre les dues implementacions -- i s'exigeix que la prova
    se n'adoni de seguida. Si no se n'adona, la prova no val res i el que diu
    "OK" no vol dir res.

    No es fa encenent el cami rapid de motor.js, que seria el que semblaria: NO
    SERVEIX. acceleracioNucli crida Math.pow directament i no passa per
    l'embolcall, a posta, o sigui que encendre'l no canvia ni un bit de cap
    trajectoria. Esta comprovat: dona exactament els mateixos numeros.
    """
    p_mogut = replace(
        p, im_K=np.array([math.nextafter(float(k), 1.0) for k in p.im_K]))
    neix = _naixements(node, p, dt, p_node=p_mogut)
    detectats = [q for q in neix.values() if q is not None and q <= PASSOS_CONTROL]
    return {
        "k_del_python": float(p.im_K[0]),
        "k_del_node": float(p_mogut.im_K[0]),
        "neix": neix,
        "punts_que_ho_detecten": len(detectats),
        "dins_de_passos": PASSOS_CONTROL,
        "passa": len(detectats) >= 1,
    }


# ---------------------------------------------------------------------------
# Prova 2 -- desenllacos d'una graella
# ---------------------------------------------------------------------------
def prova_desenllacos(node, p, finestra, resolucio, criteri, dt) -> dict:
    """La prova de debo. Es la que decideix si el mapa del navegador es de fiar.

    Les taules de destins que se li passen al JavaScript son LES DEL PYTHON, a
    posta. Aixi l'unica cosa que es compara es la integracio; si el JavaScript
    calcules tambe els seus punts de repos, una diferencia no es sabria d'on ve.
    Que els sapiga calcular es la prova 3.
    """
    print("   Python (el que mana):")
    x0, y0, imant_py, temps_py = calcula(p, finestra, resolucio, criteri, dt, cada=resolucio)

    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    feina = {
        "tipus": "desenllac", "parametres": _parametres_json(p),
        "x0": [float(v) for v in x0], "y0": [float(v) for v in y0],
        "dest_x": [float(v) for v in dest_x], "dest_y": [float(v) for v in dest_y],
        "dest_codi": [int(v) for v in dest_codi],
        "dest_radi2": [float(v) for v in dest_radi2],
        "dt": dt,
        "passos": int(round(criteri.t_max / dt)),
        "passos_confirma": int(round(criteri.t_confirma / dt)),
        "passos_confirma_centre": int(round(criteri.t_confirma_centre / dt)),
    }

    print("   JavaScript (la copia):")
    t0 = time.perf_counter()
    res = crida_node(node, [feina])[0]
    segons_js = time.perf_counter() - t0

    imant_js = np.array(res["codi"], dtype=np.int64).reshape(resolucio, resolucio)
    temps_js = np.array(res["temps"], dtype=float).reshape(resolucio, resolucio)

    # "Dins" i "frontera" es decideixen amb el mapa del PYTHON, que es el que es
    # publicaria. Es el mateix criteri que magnetic/converge.py.
    frontera = mascara_de_frontera(imant_py)
    dins = ~frontera
    canvia = imant_py != imant_js
    n_dins = int(np.count_nonzero(dins))
    n_front = int(np.count_nonzero(frontera))

    # El temps de caiguda tambe s'ha de separar en dins i frontera, pel mateix
    # motiu que el desti. Es multiple de dt, o sigui que on la trajectoria es la
    # mateixa el temps ha de sortir EXACTE; una diferencia de dins d'una conca
    # voldria dir que els dos comptadors de passos no van sincronitzats, i aixo
    # si que seria un error de traduccio.
    #
    # A la frontera, en canvi, en pot diferir, i s'ha vist per que: el punt de
    # (-7, 0) cm es de la corona de pols, triga 21 s a assentar-se i per tant te
    # temps de separar-se; quan finalment entra al cercle de captura hi entra a
    # 70 um per pas, i el JavaScript hi arriba quatre passos mes tard. El desti
    # continua sent el mateix. Es el mateix fenomen de la frontera, en el temps
    # en comptes de en el color, i per aixo es reporta en comptes de fallar.
    igual = ~canvia
    temps_difereix = temps_py != temps_js
    temps_dins = int(np.count_nonzero(igual & dins & temps_difereix))
    temps_frontera = int(np.count_nonzero(igual & frontera & temps_difereix))

    # La llista dels punts que discrepen, per poder-los anar a mirar d'un en un.
    # Es desa sempre al JSON; amb --explica, a mes, es diu que els passa.
    discrepants = []
    for j, i in zip(*np.nonzero(canvia | (igual & temps_difereix))):
        discrepants.append({
            "fila": int(j), "columna": int(i),
            "x0_cm": float(x0[i]) * 100.0, "y0_cm": float(y0[j]) * 100.0,
            "de_frontera": bool(frontera[j, i]),
            "desti_python": int(imant_py[j, i]), "desti_js": int(imant_js[j, i]),
            "temps_python_s": float(temps_py[j, i]), "temps_js_s": float(temps_js[j, i]),
        })

    return {
        "pixels": int(imant_py.size),
        "n_dins": n_dins,
        "n_frontera": n_front,
        "canvien_total": int(np.count_nonzero(canvia)),
        "percent_total": 100.0 * float(np.count_nonzero(canvia)) / imant_py.size,
        "percent_dins": 100.0 * float(np.count_nonzero(canvia & dins)) / max(1, n_dins),
        "percent_frontera": 100.0 * float(np.count_nonzero(canvia & frontera)) / max(1, n_front),
        "temps_difereix_dins": temps_dins,
        "temps_difereix_frontera": temps_frontera,
        "grisos_python": int(np.count_nonzero(imant_py == -1)),
        "grisos_js": int(np.count_nonzero(imant_js == -1)),
        "segons_js": segons_js,
        "discrepants": discrepants,
    }


def explica_discrepancia(p, criteri, dt, punt, dest) -> None:
    """Que li passa, a un punt que discrepa. S'engega amb --explica.

    NO ES UN LUXE. Un numero com "el temps de caiguda difereix en 4 passos" no
    diu res tot sol: podria ser un error de traduccio o podria ser el caos, i
    son coses ben diferents. Aixo ho decideix mirant a quina velocitat la massa
    creua el cercle de captura al moment d'entrar-hi.

    Si hi entra de gairebe, fregant la vora, quatre passos de diferencia son
    fraccions de mil.limetre i la culpa es del caos. Si hi entres de ple, un
    comptador que balla es un error.
    """
    dest_x, dest_y, dest_codi, dest_radi2 = dest
    x, y, vx, vy = punt["x0_cm"] / 100.0, punt["y0_cm"] / 100.0, 0.0, 0.0
    t_min = min(punt["temps_python_s"], punt["temps_js_s"])
    t_max = max(punt["temps_python_s"], punt["temps_js_s"])
    k_ini = max(1, int(round(t_min / dt)) - 2)
    k_fi = int(round(t_max / dt)) + 2

    print(f"\n   punt ({punt['x0_cm']:+.4f}, {punt['y0_cm']:+.4f}) cm, "
          f"{'de FRONTERA' if punt['de_frontera'] else 'de DINS de conca'}")
    print(f"   desti: Python {punt['desti_python']}, JavaScript {punt['desti_js']}   |   "
          f"temps: {punt['temps_python_s']:.3f} s contra {punt['temps_js_s']:.3f} s "
          f"({abs(t_max - t_min) / dt:.0f} passos)")

    anterior = None
    for k in range(1, k_fi + 1):
        x, y, vx, vy = pas_rk4(x, y, vx, vy, dt, p.L, p.g, p.b, p.n, p.d0,
                               p.im_x, p.im_y, p.im_K)
        if k < k_ini:
            continue
        d = float(np.min(np.hypot(x - dest_x, y - dest_y)))
        radi = float(np.sqrt(np.min(dest_radi2)))
        codi = int(classifica(x, y, dest_x, dest_y, dest_codi, dest_radi2))
        ritme = "" if anterior is None else f"  {(anterior - d) * 1e6:+8.1f} um/pas"
        print(f"      k={k:6d}  t={k * dt:8.3f} s  codi={codi:3d}  "
              f"al repos mes proper {d * 1000:8.4f} mm  "
              f"(radi {radi * 1000:.2f} mm){ritme}")
        anterior = d


# ---------------------------------------------------------------------------
# Prova 3 -- punts de repos i estabilitat del centre
# ---------------------------------------------------------------------------
# Les parelles cobreixen els tres casos que el mapa pot tenir: centre atractor,
# centre just al limit i centre inestable; i tambe una K prou fluixa perque
# l'imant no arribi a retenir la massa, que es el cas en que un desti DESAPAREIX
# del mapa. Son els casos que les banderes --n i --K de magnetic.mapa permeten
# mirar de debo, i d'aqui van sortir les dues comprovacions del CLAUDE.md.
#
# Els casos de n = 3 son de la llei INVENTADA (fins al 26/10, n = 3 i
# K = 1e-4 eren "els valors de serie"). Es queden perque cobreixen tots els
# casos, pero ja no es diuen de serie. Els de la llei mesurada els afegeix
# casos_repos(), perque depenen de parametres.json.
CASOS_REPOS = (
    ("n = 3, K = 1e-4", 3.0, 1.0e-4),
    ("n = 3, K a la meitat", 3.0, 0.5e-4),
    # Just per sobre del critic: 1,419e-4 des del 25/10 (1,286e-4 abans, amb 1,29e-4 aqui).
    ("n = 3, K al critic", 3.0, 1.42e-4),
    ("n = 3, K sobre el critic", 3.0, 2.0e-4),
    ("n = 2, com un monopol", 2.0, 1.0e-4),
    ("n = 4, com un dipol", 4.0, 1.0e-4),
    ("n = 3, K massa fluixa", 3.0, 0.05e-4),
)


def casos_repos(p: Parametres) -> tuple:
    """CASOS_REPOS i, des del 28/10, tres casos amb la llei MESURADA: la de
    parametres.json, i la K a x0,97 i a x1,03 de la critica (les mateixes que
    els mapes K_sota_critica i K_sobre_critica). Ha de sortir: el centre
    atractor, inestable just per sobre, i el repos a 4,477 cm."""
    K = float(p.im_K[0])
    critica = factor_critic(p) * K
    return CASOS_REPOS + (
        ("parametres.json", p.n, K),
        ("x0,97 de la K critica", p.n, 0.97 * critica),
        ("x1,03 de la K critica", p.n, 1.03 * critica),
    )


def prova_repos(node: str, p: Parametres) -> list:
    feines, casos = [], []
    for etiqueta, n, K in casos_repos(p):
        pc = replace(p, n=n, im_K=np.full(len(p.im_K), K))
        casos.append((etiqueta, n, K, pc))
        feines.append({"tipus": "repos", "parametres": _parametres_json(pc)})

    resultats = crida_node(node, feines)

    files = []
    for (etiqueta, n, K, pc), js in zip(casos, resultats):
        # Amb una K massa fluixa no hi ha cap punt de repos i el Python peta a
        # posta. No es cap error: es el cas en que aquell desti no existeix.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                eq_x, eq_y, estable_py = punts_dequilibri(pc)
                r_py = math.hypot(float(eq_x[0]), float(eq_y[0]))
            except RuntimeError:
                r_py, estable_py = None, None
        if estable_py is None:
            from magnetic.fisica import rigidesa_al_centre
            jota, _ = rigidesa_al_centre(pc)
            estable_py = bool(np.all(np.linalg.eigvalsh(jota) < 0))

        q = js["punts"][0]
        r_js = None if q is None else q["r"]
        fc_py = factor_critic(pc)
        fc_js = math.inf if js["factor_critic"] is None else js["factor_critic"]

        files.append({
            "cas": etiqueta, "n": n, "K": K,
            "r_python_m": r_py, "r_js_m": r_js,
            "dif_repos_m": None if (r_py is None or r_js is None) else abs(r_py - r_js),
            "hi_ha_repos_python": r_py is not None,
            "hi_ha_repos_js": r_js is not None,
            "residu_js": None if q is None else q["residu"],
            "centre_estable_python": bool(estable_py),
            "centre_estable_js": bool(js["centre_estable"]),
            "factor_critic_python": fc_py,
            "factor_critic_js": fc_js,
            "dif_factor_relativa": (
                0.0 if (math.isinf(fc_py) and math.isinf(fc_js))
                else abs(fc_js - fc_py) / abs(fc_py)
            ),
        })
    return files


# ---------------------------------------------------------------------------
# Prova 5 -- frontera i dimensio fractal (analisi.js contra analisi.py)
# ---------------------------------------------------------------------------
# PER QUE HI ES. La pagina ensenya el % de frontera i la dimensio DE LA FINESTRA
# QUE S'ESTA MIRANT, i aixo el Python no ho pot saber per endavant: s'ha de
# calcular al navegador. O sigui que aquestes dues operacions passen a estar
# escrites dues vegades, i val la mateixa regla que per al motor: la copia s'ha
# de guanyar el dret a fer-se servir.
#
# AQUI SI QUE ES POT EXIGIR COINCIDENCIA EXACTA, i no com a la prova 2. Els
# comptes de pixels i de capses son SENCERS: no hi ha cap arrodoniment que els
# pugui moure. L'unic numero de coma flotant es el pendent de la recta, i
# s'ajusta sobre els mateixos comptes sencers.
#
# LA MATRIU ARRIBA FETA DES DEL PYTHON, a posta: aixi l'unica cosa que es compara
# es el compte, i no s'hi barreja cap diferencia d'integracio de la prova 2.
CASOS_ANALISI = (
    # (etiqueta, finestra en cm, costat en pixels)
    ("vista general", (-12.0, 12.0, -12.0, 12.0), 64),
    # El zoom2 nou (28/10): conques i frontera. El vell era pols.
    ("conques i frontera", (1.3057, 2.8057, 0.3030, 1.8030), 128),
    # 64 i 128 son divisibles per totes les capses, i no posen a prova la regio
    # comuna del 31/10. 151 no ho es (la regio es de 144), com els nivells de la
    # pagina.
    ("conques, 151 punts", (1.3057, 2.8057, 0.3030, 1.8030), 151),
)

LLINDAR_DIMENSIO = 1e-12


def prova_analisi(node: str, p: Parametres, criteri: Criteri, dt: float) -> list:
    """Els dos numeros, calculats per les dues implementacions sobre la MATEIXA
    matriu. Inclou el cas degenerat, que es on es comporten diferent a posta."""
    matrius = []
    for etiqueta, finestra_cm, costat in CASOS_ANALISI:
        finestra = [v / 100.0 for v in finestra_cm]
        _, _, imant, _ = calcula(p, finestra, costat, criteri, dt, cada=costat)
        matrius.append((etiqueta, np.asarray(imant, dtype=np.int64)))

    # Tot d'un color: no hi ha frontera, i llavors no hi ha prou escales per
    # ajustar cap recta. El Python peta a posta amb un missatge que ho explica i
    # el JavaScript torna null. Les dues coses volen dir el mateix, i val mes
    # tenir-ho comprovat que descobrir-ho el dia que passi a la pagina.
    matrius.append(("tot d'un color", np.ones((32, 32), dtype=np.int64)))

    feines = [{"tipus": "analisi", "codis": [int(v) for v in m.ravel()],
               "files": int(m.shape[0]), "columnes": int(m.shape[1])}
              for _, m in matrius]
    resultats = crida_node(node, feines)

    files = []
    for (etiqueta, m), js in zip(matrius, resultats):
        mascara = mascara_de_frontera(m)
        pc_py = percentatge_de_frontera(m)
        try:
            dim_py, costats_py, comptes_py = dimensio_per_box_counting(mascara)
            locals_py = pendents_locals(costats_py, comptes_py)
            interval_py = [float(locals_py.min()), float(locals_py.max())]
            costats_py = [int(v) for v in costats_py]
            comptes_py = [int(v) for v in comptes_py]
        except RuntimeError:
            dim_py, interval_py, costats_py, comptes_py = None, None, [], []

        dim_js = js["dimensio"]
        interval_js = js["interval"]
        if dim_py is None or dim_js is None:
            dif_dim = 0.0 if (dim_py is None and dim_js is None) else float("inf")
        else:
            # La dimensio i els dos extrems de l'interval: la pitjor de les tres
            # diferencies.
            dif_dim = max(abs(dim_js - dim_py), abs(interval_js[0] - interval_py[0]),
                          abs(interval_js[1] - interval_py[1]))

        files.append({
            "cas": etiqueta,
            "costat": int(m.shape[0]),
            "pixels_frontera_python": int(np.count_nonzero(mascara)),
            "pixels_frontera_js": js["pixels_de_frontera"],
            "percentatge_python": pc_py,
            "percentatge_js": js["percentatge_de_frontera"],
            "percentatge_identic": pc_py == js["percentatge_de_frontera"],
            "dimensio_python": dim_py,
            "dimensio_js": dim_js,
            "interval_python": interval_py,
            "interval_js": interval_js,
            "dif_dimensio": dif_dim,
            "capses_python": comptes_py,
            "capses_js": js["comptes"],
            "capses_identiques": comptes_py == list(js["comptes"])
                                 and costats_py == list(js["costats"]),
        })
    return files


# ---------------------------------------------------------------------------
# Prova 4 -- la pagina sencera (opcional: --pagina)
# ---------------------------------------------------------------------------
def prova_pagina(node: str, p: Parametres, criteri: Criteri, dt: float, finestra_m,
                 mode: str = "fil") -> dict:
    """El mapa que pinta mapa-interactiu.html contra el que en diu el Python.

    Les tres primeres proves diuen que el MOTOR es bo. Aquesta diu que la
    PAGINA n'treu el mapa bo, que no es el mateix: entremig hi ha la reticula de
    tres nivells, el repartiment de files i la pintura, on la y del mapa va cap
    amunt i la del llenc cap avall.

    Les coordenades es fabriquen com les fabrica pagina.js -- una sola reticula
    de 601 punts recorreguda a salts de 4 -- i NO amb np.linspace, que donaria
    uns altres ultims bits i faria comparar dues coses lleugerament diferents.
    """
    if not PAGINA_HTML.exists():
        raise SystemExit(
            f"No hi ha {PAGINA_HTML}.\n"
            "   Fes-la primer:  .venv\\Scripts\\python.exe "
            "-m magnetic.magnetic_interactiu.construeix"
        )

    ordre = [node, str(PROVA_PAGINA_JS), str(PAGINA_HTML), json.dumps(list(finestra_m))]
    if mode == "fils":
        ordre.append("--fils")
    elif mode.startswith("vista:"):
        ordre += ["--vista", mode.split(":")[1]]
    r = subprocess.run(ordre, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"prova_pagina.js ha acabat amb el codi {r.returncode}.")
    sortida = json.loads(r.stdout)
    pagina = np.array(sortida["codis"], dtype=np.int64)

    # La finestra la torna l'arnes: amb --vista es la que porta incrustada la
    # pagina, que no te per que ser la que se li ha demanat.
    x_min, x_max, y_min, y_max = sortida["finestra"]
    fi, salt = 600, sortida["salt"]
    pas = (x_max - x_min) / fi
    ks = range(0, fi + 1, salt)
    x0 = np.array([x_min + k * pas for k in ks])
    y0 = np.array([y_min + k * pas for k in ks])

    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    imant = np.empty((len(y0), len(x0)), dtype=np.int64)
    temps = np.empty((len(y0), len(x0)), dtype=np.float64)
    graella_nucli(x0, y0, p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K,
                  dest_x, dest_y, dest_codi, dest_radi2, dt,
                  int(round(criteri.t_max / dt)), int(round(criteri.t_confirma / dt)),
                  int(round(criteri.t_confirma_centre / dt)),
                  imant, temps)

    frontera = mascara_de_frontera(imant)
    dins = ~frontera
    canvia = imant != pagina
    n_dins = int(np.count_nonzero(dins))
    n_front = int(np.count_nonzero(frontera))
    return {
        "mode": mode,
        "finestra_cm": [v * 100 for v in sortida["finestra"]],
        "punts": int(pagina.size),
        "costat": int(pagina.shape[0]),
        "n_dins": n_dins,
        "n_frontera": n_front,
        "percent_dins": 100.0 * float(np.count_nonzero(canvia & dins)) / max(1, n_dins),
        "percent_frontera": 100.0 * float(np.count_nonzero(canvia & frontera)) / max(1, n_front),
        "percent_total": 100.0 * float(np.count_nonzero(canvia)) / pagina.size,
        # -99 vol dir que del llenc n'ha sortit un color que no es de la paleta:
        # seria un error de pintura, no de calcul, i s'ha de veure a part.
        "colors_desconeguts": int(np.count_nonzero(pagina == -99)),
        "segons": sortida["segons"],
        "com_va": sortida["com_va"],
        "autotest": sortida["autotest"],
        "estat_calcul": sortida.get("estat_calcul"),
    }


def vistes_de_la_pagina() -> list:
    """Les etiquetes de les vistes que porta incrustades la pagina, en ordre.

    Es llegeixen de l'HTML (construeix.py hi escriu VISTES en una sola linia) i
    no de construeix.py: el que s'ha de provar es el fitxer que s'obrira.
    """
    for linia in PAGINA_HTML.read_text(encoding="utf-8").splitlines():
        if linia.startswith("var VISTES = "):
            return [v["etiqueta"] for v in json.loads(linia[len("var VISTES = "):].rstrip(";"))]
    return []


# ---------------------------------------------------------------------------
# El programa
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--node", default=None, help="cami del node.exe, si no es al PATH")
    ap.add_argument("--finestra", type=float, nargs=4, default=[-7, 7, -7, 7],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="limits en CENTIMETRES")
    ap.add_argument("--resolucio", type=int, default=45,
                    help="pixels per costat de la prova 2 (45 -> 2025 punts)")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE, help="pas d'integracio (s)")
    ap.add_argument("--temps", type=float, default=None, help="temps maxim (s)")
    ap.add_argument("--sortida", default="magnetic/figures/comparacio_js.json",
                    help="on van els numeros")
    ap.add_argument("--pagina", action="store_true",
                    help="passa tambe la prova 4: la pagina sencera dins de node")
    ap.add_argument("--explica", action="store_true",
                    help="mira d'un en un els punts que discrepen a la prova 2")
    ap.add_argument("--explica-quants", type=int, default=3,
                    help="quants punts s'expliquen com a maxim")
    ap.add_argument("--escombra-b", action="store_true",
                    help="nomes la prova 1, a diverses friccions: ensenya que el "
                         "llindar d'ulps que hi havia mesurava la b i no el codi")
    args = ap.parse_args(argv)

    node = busca_node(args.node)
    p = Parametres.des_de_json()
    d = Criteri()
    criteri = Criteri(t_max=d.t_max if args.temps is None else args.temps)
    finestra = [v / 100.0 for v in args.finestra]

    print("=" * 72)
    print(" magnetic.magnetic_interactiu.compara -- el JavaScript diu el mateix?")
    print("=" * 72)
    avis = avis_inventats(p)
    if avis:
        print(avis)
    print(f"   Python : {sys.version.split()[0]}, {compila.com_va()}")
    print(f"   node   : {node}")
    print("   El Python es el que mana. El JavaScript es la copia que s'ha de")
    print("   guanyar el dret a fer-se servir.\n")

    if args.escombra_b:
        return escombra_de_friccio(node, p, args.dt)

    tot_be = True

    # -- 0 -----------------------------------------------------------------
    print("-" * 72)
    print("0. delta^(n+1) tota sola, en 20.001 distancies d'1,5 a 30 cm")
    print("   Es l'operacio d'on surten totes les diferencies. Val mes tenir-la")
    print("   mesurada que deduir-la de la resta.\n")
    pot = prova_potencia(node, p)
    print(f"   {'com ho fa el JavaScript':>28} {'difereixen':>12} {'':>4} {'pitjor':>8}")
    for etiqueta, r in pot.items():
        print(f"   {etiqueta:>28} {r['diferents']:>7}/{r['total']:<5} "
              f"{r['percent']:5.1f} % {r['pitjor_ulp']:>4} ulp")
    print()
    print("   El motor fa SEMPRE la primera fila: acceleracioNucli crida Math.pow")
    print("   directament. La segona esta aqui nomes per tenir mesurat que costaria")
    print("   el cami rapid, i per aixo el nucli no el fa servir. Treure l'embolcall")
    print("   del bucle, a mes, va fer l'acceleracio un 10,5 % mes rapida.")

    # -- 1 -----------------------------------------------------------------
    print("\n" + "-" * 72)
    print("1. Trajectoria a curt termini: l'estat despres de N passos de RK4")
    print("   Sense cap classificacio pel mig. Es el que demostra que les")
    print("   equacions son les mateixes.")
    print(f"   CRITERI: durant els primers {PASSOS_IDENTICS} passos els dos motors no poden")
    print(f"   diferir ni {ULP_NAIXEMENT:.0f} ulp de l'escala de l'estat: identics fins a l'ultim bit")
    print("   que l'estat pot representar. NO es un llindar d'ulps als 1.000 passos:")
    print("   aquell mesura la friccio i no el codi. Vegeu PASSOS_IDENTICS i ULP_NAIXEMENT.")
    print("   Els ulps son de l'ESCALA DE L'ESTAT, no de cada component: vegeu")
    print("   ulps_a_escala() per que una component que val zero per simetria no")
    print("   es pot comparar component a component.\n")
    files = prova_trajectoria(node, p, args.dt)
    print(f"   {'punt de sortida':>20} {'(cm)':>14} " +
          "".join(f"{str(q) + ' passos':>13}" for q in PASSOS_TRAJECTORIA))
    for etiqueta, x0, y0 in PUNTS_TRAJECTORIA:
        seves = [f for f in files if f["punt"] == etiqueta]
        pitjors = "".join(f"{f['pitjor_ulp']:>9.2f} ulp" for f in seves)
        print(f"   {etiqueta:>20} {f'({x0*100:.1f}, {y0*100:.1f})':>14} {pitjors}")
    pitjor_ulp = max(f["pitjor_ulp"] for f in files)

    # El control positiu, i es torna a fer CADA VEGADA: si la prova fos cega,
    # tota la resta d'aquest programa no voldria dir res.
    ctrl = control_de_la_prova_1(node, p, args.dt)
    neix = _naixements(node, p, args.dt)
    print(f"\n   {'punt de sortida':>20}   {'de serie':>12} {'CONTROL (K +1 ulp)':>20}")
    print(f"   {'':>20}   {'la diferencia neix al pas':>33}")
    for etiqueta, _, _ in PUNTS_TRAJECTORIA:
        a, b = neix[etiqueta], ctrl["neix"][etiqueta]
        print(f"   {etiqueta:>20}   {(str(a) if a else 'mai'):>12} "
              f"{(str(b) if b else 'mai'):>20}")

    primer = min((q for q in neix.values() if q is not None), default=None)
    ok1 = (primer is None or primer > PASSOS_IDENTICS) and ctrl["passa"]
    tot_be = tot_be and ok1

    print(f"\n   control: amb la K moguda un ulp se n'adonen "
          f"{ctrl['punts_que_ho_detecten']} de {len(PUNTS_TRAJECTORIA)} punts dins "
          f"de {ctrl['dins_de_passos']} passos   "
          f"{'OK' if ctrl['passa'] else 'LA PROVA ES CEGA'}")
    print(f"   de serie: identics fins al pas "
          f"{(primer - 1) if primer else PASSOS_NAIXEMENT[-1]}, i se'n demanen "
          f"{PASSOS_IDENTICS}   (pitjor ulp a 1.000: {pitjor_ulp:.2f})"
          f"   {'OK' if ok1 else 'MALAMENT'}")

    if not ctrl["passa"]:
        print("\n   -> LA PROVA ES CEGA: amb els dos costats discrepant a posta no")
        print("      se n'adona. Res del que digui aquest programa no vol dir res")
        print("      fins que aixo estigui arreglat.")
    elif not ok1 and primer is not None and primer > 10:
        print(f"\n   -> COMPTE: aixo te cara de CAOS, no d'error. La diferencia neix")
        print(f"      al pas {primer} i una aritmetica diferent es veuria al primer o")
        print("      gairebe -- el control de dalt ho demostra. El que segurament")
        print(f"      s'ha quedat curt es el criteri dels {PASSOS_IDENTICS} passos, no el codi;")
        print("      pero aixo es una DECISIO, i es pren mirant-ho, no abaixant")
        print("      el numero perque falla.")
    elif not ok1:
        print("   -> les dues implementacions NO fan la mateixa aritmetica.")
        print("      Mira l'ordre de les operacions de motor.js abans que res:")
        print("      en coma flotant la suma no es associativa.")

    # -- 2 -----------------------------------------------------------------
    print("\n" + "-" * 72)
    print(f"2. Desenllacos: graella de {args.resolucio}x{args.resolucio} = "
          f"{args.resolucio**2} punts, finestra {args.finestra} cm")
    print("   Dins de conca ha de coincidir el 100 %. A la frontera n'hi haura")
    print("   de diferents, i el numero interessant es quants.\n")
    des = prova_desenllacos(node, p, finestra, args.resolucio, criteri, args.dt)
    print(f"\n   {'':>22} {'punts':>10} {'canvien de desti':>18} {'temps diferent':>16}")
    print(f"   {'DINS de les conques':>22} {des['n_dins']:>10} "
          f"{des['percent_dins']:>16.2f} % {des['temps_difereix_dins']:>16}")
    print(f"   {'a la FRONTERA':>22} {des['n_frontera']:>10} "
          f"{des['percent_frontera']:>16.2f} % {des['temps_difereix_frontera']:>16}")
    print(f"   {'tota la graella':>22} {des['pixels']:>10} {des['percent_total']:>16.2f} %")
    print(f"\n   grisos (no convergits): Python {des['grisos_python']}, "
          f"JavaScript {des['grisos_js']}")
    print(f"   el node hi ha trigat {des['segons_js']:.1f} s")

    ok2 = des["percent_dins"] == 0.0 and des["temps_difereix_dins"] == 0
    tot_be = tot_be and ok2
    print()
    if ok2:
        print("   OK -> dins de les conques no en canvia CAP, ni el desti ni el temps.")
        print(f"        A la frontera canvien de desti el {des['percent_frontera']:.2f} % i tenen un")
        print(f"        temps diferent {des['temps_difereix_frontera']} punts, i AIXO NO ES CAP DEFECTE:")
        print("        son els punts on el desenllac no esta determinat de manera")
        print("        robusta. Es el mateix fenomen que magnetic.converge ja mesura")
        print("        amb dt/2 a la mateixa finestra.")
    else:
        if des["percent_dins"] != 0.0:
            print(f"   MALAMENT -> dins de les conques en canvia el {des['percent_dins']:.2f} %.")
            print("        Aixo no ho pot explicar l'arrodoniment de pow: dins d'una conca")
            print("        el desti es robust. Hi ha un error de traduccio a motor.js.")
        if des["temps_difereix_dins"]:
            print(f"   MALAMENT -> {des['temps_difereix_dins']} punts de DINS d'una conca amb el mateix")
            print("        desti i un temps de caiguda diferent. Dins d'una conca la")
            print("        trajectoria es robusta, o sigui que els dos comptadors de passos")
            print("        no van iguals: mira candidat/seguits/kEntrada de desenllacNucli.")

    if des["discrepants"]:
        print(f"\n   Punts que discrepen: {len(des['discrepants'])}. "
              f"Son tots al JSON de sortida.")
        if args.explica:
            print("   Que els passa, pas a pas (nomes Python; el JavaScript hi entra")
            print("   al pas que digui la seva columna de temps):")
            dest = destins(p, criteri)[:4]
            for punt in des["discrepants"][: args.explica_quants]:
                explica_discrepancia(p, criteri, args.dt, punt, dest)
        else:
            print("   Amb --explica es veu, d'un en un, a quina velocitat la massa")
            print("   creua el cercle de captura: es el que distingeix el caos d'un error.")

    # -- 3 -----------------------------------------------------------------
    print("\n" + "-" * 72)
    print("3. Punts de repos i estabilitat del centre")
    print("   El Python els treu amb el fsolve de scipy i el JavaScript per")
    print("   biseccio pel radi. Son dos algorismes diferents: aqui no es pot")
    print(f"   demanar el bit, sino que la diferencia sigui < {LLINDAR_REPOS_M:.0e} m,")
    print("   irrellevant al costat dels 8 mm del radi de captura.\n")
    rep = prova_repos(node, p)
    print(f"   {'cas':>24} {'n':>5} {'K':>10} {'r py (cm)':>11} {'dif (m)':>10} "
          f"{'centre':>10} {'x critic':>10}")
    ok3 = True
    for f in rep:
        # L'estabilitat del centre i el factor critic es comproven SEMPRE, tambe
        # quan no hi ha punt de repos: son dues coses independents, i el cas de
        # la K fluixa (l'imant no rete la massa pero el centre continua atraient)
        # es precisament el que ho ensenya.
        be_centre = (f["centre_estable_python"] == f["centre_estable_js"]
                     and f["dif_factor_relativa"] < 1e-9)
        centre = "atractor" if f["centre_estable_python"] else "inestable"
        if f["centre_estable_python"] != f["centre_estable_js"]:
            centre = "NO COINC."

        if not f["hi_ha_repos_python"] or not f["hi_ha_repos_js"]:
            coincideix = f["hi_ha_repos_python"] == f["hi_ha_repos_js"]
            be = coincideix and be_centre
            ok3 = ok3 and be
            nota = "cap punt de repos" if coincideix else "NOMES UN N'HI TROBA"
            print(f"   {f['cas']:>24} {f['n']:>5.2f} {f['K']:>10.2e} {nota:>22} "
                  f"{centre:>10} {f['factor_critic_python']:>10.3f}"
                  f"{'' if be else '   <-- MIRA-HO'}")
            continue

        be = f["dif_repos_m"] < LLINDAR_REPOS_M and be_centre
        ok3 = ok3 and be
        print(f"   {f['cas']:>24} {f['n']:>5.2f} {f['K']:>10.2e} "
              f"{f['r_python_m']*100:>11.4f} {f['dif_repos_m']:>10.1e} "
              f"{centre:>10} {f['factor_critic_python']:>10.3f}"
              f"{'' if be else '   <-- MIRA-HO'}")
    tot_be = tot_be and ok3
    residu = max((f["residu_js"] for f in rep if f["residu_js"] is not None), default=0.0)
    print(f"\n   residu mes gros del JavaScript: {residu:.1e} m/s2")
    print("   (l'acceleracio que queda de debo al punt trobat. Amb les tres K")
    print("    iguals ha de ser del nivell del soroll de la coma flotant: es la")
    print("    comprovacio que la biseccio pel radi era licita.)")
    print(f"\n   {'OK' if ok3 else 'MALAMENT'}")

    # -- 5 -----------------------------------------------------------------
    print("\n" + "-" * 72)
    print("5. Frontera i dimensio fractal: analisi.js contra analisi.py")
    print("   La pagina ensenya aquests dos numeros de la finestra que estas")
    print("   mirant, i aixo s'ha de calcular al navegador. Aqui si que es pot")
    print("   exigir el numero EXACTE i no com a la prova 2: els comptes de")
    print("   pixels i de capses son sencers, no hi ha cap arrodoniment que els")
    print("   pugui moure.\n")
    ana = prova_analisi(node, p, criteri, args.dt)
    print(f"   {'cas':>22} {'costat':>7} {'frontera py':>12} {'frontera js':>12} "
          f"{'capses':>8} {'dimensio py':>12} {'dif dim':>10}")
    ok5 = True
    for f in ana:
        be = (f["percentatge_identic"] and f["capses_identiques"]
              and f["dif_dimensio"] <= LLINDAR_DIMENSIO)
        ok5 = ok5 and be
        dim = "cap" if f["dimensio_python"] is None else f"{f['dimensio_python']:.6f}"
        print(f"   {f['cas']:>22} {f['costat']:>7} "
              f"{f['percentatge_python']:>11.4f}% {f['percentatge_js']:>11.4f}% "
              f"{'iguals' if f['capses_identiques'] else 'DIFEREIXEN':>8} "
              f"{dim:>12} {f['dif_dimensio']:>10.1e}"
              f"{'' if be else '   <-- MIRA-HO'}")
    tot_be = tot_be and ok5
    print()
    if ok5:
        print("   OK -> els dos numeros surten identics sobre la mateixa matriu, i")
        print("        els comptes de capses tambe, escala per escala. El cas de")
        print("        'tot d'un color' comprova que les dues bandes es rendeixen")
        print("        igual quan no hi ha frontera: el Python peta a posta i el")
        print("        JavaScript torna null, i tots dos volen dir el mateix.")
    else:
        print("   MALAMENT -> analisi.js no compta com analisi.py. Mira primer la")
        print("        regio comuna (regio_comuna / regioComuna): totes les escales")
        print("        es compten damunt del mateix rectangle central, i si una banda")
        print("        el talla diferent, els numeros no poden coincidir.")

    # -- 4 -----------------------------------------------------------------
    pag = None
    if args.pagina:
        print("\n" + "-" * 72)
        print("4. La pagina sencera, feta correr dins de node sense navegador")
        print("   Les tres proves de sobre diuen que el MOTOR es bo. Aquesta diu")
        print("   que la PAGINA en treu el mapa bo: la reticula de tres nivells, el")
        print("   repartiment de files i la pintura. El resultat es llegeix DEL")
        print("   LLENC, no de cap variable de dins, perque hi entri la pintura.\n")
        print("   Es passa pels DOS camins que la pagina pot agafar, perque un")
        print("   navegador que obre un fitxer local pot bloquejar els treballadors,")
        print("   i a mes es comprova CADA VISTA que la pagina porta INCRUSTADA: la")
        print("   general es el primer que es veu en obrir-la, i cap no passa per cap")
        print("   calcul. Com que son el mateix calcul del Python a la mateixa")
        print("   reticula, n'ha de coincidir TOT, frontera inclosa.\n")
        finestra_pag = [v / 100.0 for v in FINESTRA_PAGINA_CM]
        print(f"   finestra de calcul {list(FINESTRA_PAGINA_CM)} cm, "
              "la del zoom2 (el nou, des del 28/10)")
        incrustades = vistes_de_la_pagina()
        print(f"   vistes incrustades: {len(incrustades)} ({', '.join(incrustades)})")
        pag, ok4 = {}, True
        casos = [("calculant, al fil principal", "fil"),
                 ("calculant, amb treballadors", "fils")]
        casos += [(f"la vista incrustada {e}", f"vista:{i}") for i, e in enumerate(incrustades)]
        for etiqueta, mode in casos:
            r = prova_pagina(node, p, criteri, args.dt, finestra_pag, mode)
            pag[etiqueta] = r
            be = (r["percent_dins"] == 0.0 and r["colors_desconeguts"] == 0
                  and r["autotest"].startswith("OK"))
            if mode.startswith("vista:"):
                # Una vista incrustada es el calcul del Python: ha de coincidir
                # tota, i la pagina l'ha d'ensenyar a 601, que es com esta feta.
                be = (be and r["percent_total"] == 0.0
                      and (r["estat_calcul"] or "").startswith("Ja calculada a 601"))
            ok4 = ok4 and be
            finestra_txt = " ".join(f"{v:.2f}" for v in r["finestra_cm"])
            print(f"\n   -- {etiqueta} --  {r['costat']}x{r['costat']} punts, "
                  f"{r['segons']:.0f} s, finestra {finestra_txt} cm")
            print(f"   {'DINS de les conques':>22} {r['n_dins']:>8} "
                  f"difereixen {r['percent_dins']:>7.2f} %")
            print(f"   {'a la FRONTERA':>22} {r['n_frontera']:>8} "
                  f"difereixen {r['percent_frontera']:>7.2f} %")
            print(f"   {'tot el mapa':>22} {r['punts']:>8} "
                  f"difereixen {r['percent_total']:>7.2f} %"
                  f"{'' if be else '   <-- MIRA-HO'}")
            if r["colors_desconeguts"]:
                print(f"   {r['colors_desconeguts']} punts amb un color que no es de la "
                      "paleta: aixo es un error de PINTURA.")
            if mode.startswith("vista:"):
                print(f"   {'la pagina diu':>22} {r['estat_calcul']}")

        # Els dos camins de CALCUL han de donar exactament el mateix: nomes canvia
        # on es crida desenllacNucli. Si no, el repartiment de files s'esta
        # menjant o duplicant alguna cosa.
        dos = [pag["calculant, al fil principal"], pag["calculant, amb treballadors"]]
        iguals = dos[0]["percent_total"] == dos[1]["percent_total"]
        ok4 = ok4 and iguals
        tot_be = tot_be and ok4
        print()
        if ok4 and all(r["percent_total"] == 0.0 for r in pag.values()):
            print("   OK -> la pagina pinta EXACTAMENT el mateix mapa que el Python pels")
            print("        dos camins de calcul, frontera inclosa, i les vistes que porta")
            print("        incrustades les col.loca be. La reticula, el repartiment de")
            print("        files i la pintura no hi afegeixen res.")
        elif ok4:
            print("   OK -> dins de les conques no en canvia cap pels dos camins.")
        elif not iguals:
            print("   MALAMENT -> els dos camins NO donen el mateix. El repartiment de")
            print("        files s'esta menjant o duplicant feina.")
        else:
            print("   MALAMENT -> la pagina NO pinta el mapa que toca. Com que el motor")
            print("        ja ha passat les proves 1 a 3, l'error es de la reticula, del")
            print("        repartiment de files o de la pintura, no de la fisica.")

    # -- resum -------------------------------------------------------------
    sortida = Path(args.sortida)
    sortida.parent.mkdir(parents=True, exist_ok=True)
    sortida.write_text(json.dumps({
        "_que_es": "Comparacio del nucli de JavaScript (motor.js) amb el de Python.",
        "_el_que_mana": "El Python. El JavaScript es una copia i ha de passar aixo.",
        "node": node,
        "python": sys.version.split()[0],
        "calcul_python": compila.com_va(),
        "finestra_cm": args.finestra,
        "resolucio": args.resolucio,
        "dt": args.dt,
        "criteri": criteri.com_a_dict(),
        "parametres": p.com_a_dict(),
        "prova_0_potencia": pot,
        "prova_1_trajectoria": files,
        "prova_2_desenllacos": des,
        "prova_3_repos": rep,
        "prova_5_analisi": ana,
        "prova_4_pagina": pag,
        "llindar_ulp": LLINDAR_ULP,
        "ulp_naixement": ULP_NAIXEMENT,
        "llindar_repos_m": LLINDAR_REPOS_M,
        "llindar_dimensio": LLINDAR_DIMENSIO,
        "tot_be": tot_be,
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    if tot_be:
        print("OK -> el motor de JavaScript es pot fer servir per a la pagina.")
        print("      Les uniques diferencies son a la frontera de les conques, que es")
        print("      on han de ser, i estan mesurades.")
    else:
        print("MALAMENT -> el motor de JavaScript NO es pot fer servir encara.")
        print("      Mira les linies marcades. No construeixis la pagina a sobre.")
    print(f"      Els numeros, a {sortida}")
    return 0 if tot_be else 1


if __name__ == "__main__":
    raise SystemExit(main())
