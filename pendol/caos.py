r"""Les eines del caos: l'exponent de Lyapunov i l'energia del pendol.

Aquest modul es PRIM a posta, com pendol/mesures.py: no importa matplotlib, no
llegeix cap fitxer i no sap d'on venen les dades. Viu aqui i no dins de cap
eina perque el que calcula ha de donar EL MATEIX NUMERO a l'eina, a la figura
i, algun dia, a la finestra interactiva.

Pla: documentacio/pla-analisi-caos.md. Els criteris de l'autoprova (B1-B5 i E1-E2)
son els d'aquell pla, fixats el 29/09/2026 abans d'executar res. Les tres
correccions del mateix dia (l'interval de reescalat, els trossos fixos
d'integracio i com es mesura l'E2) hi son escrites amb el seu motiu.

QUE HI HA AQUI

    benettin          l'exponent mes gran, amb dues trajectories i reescalat
    classifica        caotica, regular o sense decidir, amb criteris escrits
    benettin_pendol   Benettin sobre el pendol doble de pendol/fisica.py
    energia_relativa  l'energia comptada des del repos (E = 0 penjant quiet)
    energia_volteig_2, energia_volteig_1
    condicions_de_repos  la familia (th1, th2, 0, 0) a una energia donada
    seccio_pendol     la seccio de Poincare exacta, amb esdeveniments
    creuaments_interpolats  la mateixa seccio en dades mostrejades (video)

EL METODE DE BENETTIN, i per que aquest i no un altre
-----------------------------------------------------
Dues trajectories: la de referencia x i una de pertorbada y a distancia d0.
Cada INTERVAL DE REESCALAT es mesura d = |y - x|, se n'acumula ln(d/d0) i es
torna a posar y a distancia d0 de x, SENSE canviar-ne la direccio. Com que es
reescala abans que d creixi gaire, la separacio no arriba mai a saturar i es
pot integrar tant com calgui:

    lambda = suma ln(d_i / d0) / (N * interval)   (nomes despres del transitori)

COMPTE AMB EL NOM. Als llibres aquest interval es diu tau. AQUI NO, a posta: al
projecte tau_f1 i tau_f2 son el fregament sec dels rodaments, que es fisica
mesurada al banc. L'interval de reescalat no es cap propietat del pendol: es
cada quant el programa fa la comprovacio, i la lambda no en depen (B5).

No es fan servir les equacions variacionals, que son el cami de llibre: demanen
el jacobia del camp, i el del pendol hauria d'anar a pendol/fisica.py, que no es
d'aquest modul. Amb dues trajectories nomes cal saber integrar, i s'integra amb
pendol/simulador.py: EL MATEIX pendol que dona tots els altres numeros.

LA TOLERANCIA ES LA ESTRICTA I NO ES NEGOCIA. Amb d0 = 1e-8, la rapida
(rtol = 1e-7) hi posaria un error mes gran que la pertorbacio mateixa, i el que
es mesuraria seria l'integrador.

    .venv\\Scripts\\python -m pendol.caos --autoprova
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

# ------------------------------------------------------------- les tries
#
# Totes son tries, i van aqui dalt amb el seu motiu perque es puguin discutir.

D0 = 1e-8
# La pertorbacio. Ha de ser prou petita perque en un interval la separacio es
# quedi al regim lineal (d0 * e^(lambda*interval) << 1), i prou gran perque
# l'error de la tolerancia estricta (rtol 1e-13 sobre estats d'ordre 1) hi
# quedi cinc ordres per sota. B5 comprova que amb 1e-7 surt el mateix.

INTERVAL_REESCALAT_S = 0.05
RAO_MAXIMA_PER_PAS = 10.0
# Cada quant es reescala. documentacio/lyapunov-laboratori.html demana que d/d0
# no passi de 10 en un interval: si passa, l'interval es massa llarg i s'ha
# d'escurcar. HISTORIA, que consti: el pla deia 0,2 s, i el 29/09/2026 la
# primera autoprova va donar d/d0 = 18,3 a 150/150. Mesurat despres
# (esborranys/_diag_caos_tau.txt), la d/d0 mes gran d'un interval val
#     150/150:  18,3 amb 0,2 s   12,8 amb 0,1 s   4,9 amb 0,05 s
#     179/179:  29,8 amb 0,2 s    9,5 amb 0,1 s   3,4 amb 0,05 s
# i per aixo 0,05 s, que es el que el document prescriu. El criteri de 10 NO es
# va tocar.

PAS_INTEGRACIO_S = 0.05
# Els trossos en que s'integra el pendol, SEMPRE aquests, faci el que faci
# l'interval de reescalat (que n'ha de ser un multiple).
# PER QUE (29/09/2026). Cada vegada que l'integrador s'atura i torna a arrencar,
# el seu error minuscul (~1e-13) cau en un lloc una mica diferent, i en un
# sistema caotic aixo creix fins a canviar la trajectoria sencera: a 150 graus,
# en uns 11 s. Si el pendol de referencia es reiniciava a cada interval, dos
# calculs amb intervals diferents seguien DOS MOVIMENTS DIFERENTS, i la prova
# que havia de dir que la lambda no depen de l'interval comparava en realitat
# dues orbites (va passar amb un 1,9 % per casualitat; el diagnostic en va donar
# fins a un 12 %). Amb trossos fixos, la referencia es la mateixa bit a bit per
# a qualsevol interval, i B5 compara el que diu que compara.

TRANSITORI_S = 5.0
# Els primers intervals serveixen perque la pertorbacio s'alinii amb la direccio
# mes inestable, i depenen de la direccio inicial triada, no de la fisica. No es
# sumen. Amb lambda d'ordre 1 s^-1, cinc segons son cinc temps d'e-plegament.

CONVERGENCIA_RELATIVA = 0.01
# |lambda(T) - lambda(0,75 T)| / lambda(T) < 1 %: el criteri d'aturada de
# documentacio/llindar-del-caos.html, seccio 03. DES DEL 29/09/2026 NOMES ES UN
# DIAGNOSTIC (el camp convergida_1pc de classifica), no decideix res: vegeu la
# nota del criteri de caotica, a sota.

LAMBDA_REGULAR = 0.05          # s^-1
RAO_DECAIMENT_REGULAR = 0.5
LAMBDA_ZERO_NUMERIC = 1e-3     # s^-1
# El criteri de «regular» (B3 del pla). Una orbita regular no convergeix mai
# amb el criteri relatiu: la seva lambda de temps finit baixa com ln(T)/T, i
# entre 0,75 T i T sempre canvia un 25 %. Es regular si, alhora,
#   |lambda(T)| < LAMBDA_REGULAR, i
#   |lambda(T)| <= RAO_DECAIMENT_REGULAR * |lambda(T/4)| + LAMBDA_ZERO_NUMERIC,
# o sigui que es petita I que va baixant. Per a ln(T)/T, lambda(T)/lambda(T/4)
# val ~0,3. El LAMBDA_ZERO_NUMERIC es per al cas en que la separacio ni tan sols
# creix (una rotacio pura): alli les dues lambdes son zero i el quocient no vol
# dir res.
#
# «CAOTICA»: no es regular i |lambda(T)| >= LAMBDA_REGULAR. La resta (no regular
# pero amb lambda petita) es SENSE DECIDIR, i no s'amaga dins de cap de les dues.
#
# CRITERI CANVIAT EL 29/09/2026, DESPRES DE MIRAR, i aprovat per en Guillem.
# Fins llavors «caotica» demanava a mes el criteri de l'1 % en aquell instant.
# La mesura de esborranys/_diag_convergencia.txt (12 orbites, 3000 s) va
# ensenyar que les 9 caotiques el complien entre 100 i 300 s i despres en
# TORNAVEN A SORTIR: l'orbita s'enganxa estones prop de les zones regulars i la
# seva lambda baixa (una de 150 graus va de 2,62 a 1,72 entre 1000 i 2000 s).
# Amb l'1 %, la meitat de les caotiques sortien «sense decidir» per atzar. En
# canvi regular contra caotica no va dubtar mai: 0,002-0,02 contra 1,7-2,7.


# ------------------------------------------------------------- Benettin

def benettin(
    avanca: Callable,
    distancia: Callable,
    x0,
    direccio,
    temps_s: float,
    *,
    d0: float = D0,
    interval: float = INTERVAL_REESCALAT_S,
    transitori_s: float = TRANSITORI_S,
    diferencia: Callable | None = None,
    continua: dict | None = None,
) -> dict:
    r"""L'exponent de Lyapunov mes gran, pel metode de Benettin.

    avanca(x, dt)      l'estat x integrat dt segons
    distancia(a, b)    la norma de la separacio, la MATEIXA a tot arreu
    direccio           cap on es posa la pertorbacio inicial (s'escala a d0)
    diferencia(y, x)   opcional: y - x tal com s'ha de reescalar. Per al pendol
                       plega les voltes senceres dels angles; per defecte, y - x
    continua           el resultat d'una crida anterior: la continua fins a
                       temps_s en comptes de tornar a comencar

    Retorna un diccionari amb la lambda final, la lambda acumulada contra el
    temps (nomes despres del transitori), la d/d0 mes gran d'un interval i la
    referencia a cada interval, que serveix per mirar-hi la deriva d'energia.
    """
    resta = diferencia or (lambda a, b: a - b)
    if continua is None:
        x = np.array(x0, dtype=float)
        v = np.asarray(direccio, dtype=float)
        escala = distancia(x, x + v)
        if not escala > 0:
            raise ValueError("la direccio de la pertorbacio te norma zero")
        y = x + v * (d0 / escala)
        logs: list[float] = []
        referencies = [x.copy()]
    else:
        x, y = continua["_x"].copy(), continua["_y"].copy()
        logs = list(continua["_logs"])
        referencies = list(continua["referencies"])
        d0, interval, transitori_s = continua["d0"], continua["interval_s"], continua["transitori_s"]

    passos = int(round(temps_s / interval))
    n_tr = int(round(transitori_s / interval))
    if passos <= n_tr:
        raise ValueError(f"amb {temps_s} s i {transitori_s} s de transitori no queda res per sumar")

    for _ in range(len(logs), passos):
        x = np.asarray(avanca(x, interval), dtype=float)
        y = np.asarray(avanca(y, interval), dtype=float)
        d = float(distancia(x, y))
        if not (d > 0 and math.isfinite(d)):
            raise RuntimeError(f"la separacio ha sortit {d} a l'interval {len(logs) + 1}")
        logs.append(math.log(d / d0))
        # EL REESCALAT: la mateixa direccio, a distancia d0 de la referencia.
        y = x + resta(y, x) * (d0 / d)
        referencies.append(x.copy())

    l = np.asarray(logs)
    suma = np.cumsum(l[n_tr:])
    k = np.arange(1, len(suma) + 1)
    return {
        "lambda": float(suma[-1] / (k[-1] * interval)),
        "t_s": (n_tr + k) * interval,             # instant de cada lambda acumulada
        "lambda_acumulada": suma / (k * interval),
        "rao_max_per_pas": float(np.exp(l.max())),
        "temps_s": passos * interval,
        "d0": d0, "interval_s": interval, "transitori_s": transitori_s,
        "referencies": np.asarray(referencies),
        "_x": x, "_y": y, "_logs": l,
    }


def lambda_a(res: dict, T: float) -> float:
    """La lambda acumulada fins a l'instant T (l'interval mes proper per sota)."""
    i = int(np.searchsorted(res["t_s"], T + 1e-9 * res["interval_s"], side="right")) - 1
    if i < 0:
        return float("nan")
    return float(res["lambda_acumulada"][i])


def classifica(res: dict) -> dict:
    """Caotica, regular o sense decidir, amb els criteris del capdamunt.

    Tot es mira a la lambda acumulada, que ja no inclou el transitori.
    """
    T = res["temps_s"]
    lT, l75, l25 = res["lambda"], lambda_a(res, 0.75 * T), lambda_a(res, 0.25 * T)
    canvi = abs(lT - l75) / abs(lT) if lT != 0 else math.inf
    regular = (abs(lT) < LAMBDA_REGULAR
               and math.isfinite(l25)
               and abs(lT) <= RAO_DECAIMENT_REGULAR * abs(l25) + LAMBDA_ZERO_NUMERIC)
    if regular:
        mena = "regular"
    elif abs(lT) >= LAMBDA_REGULAR:
        mena = "caotica"
    else:
        mena = "sense_decidir"
    return {"mena": mena, "lambda_T": lT, "lambda_075T": l75, "lambda_025T": l25,
            "canvi_relatiu_ultim_quart": canvi,
            "convergida_1pc": bool(canvi < CONVERGENCIA_RELATIVA)}


# ------------------------------------------------------- el pendol doble

def energia_relativa(estat, p) -> np.ndarray | float:
    """L'energia comptada des del repos: zero amb els dos bracos penjant quiets.

    fisica.energia la compta amb V = 0 als bracos horitzontals, que es un zero
    tan bo com qualsevol per conservar-la, pero no per comparar-la amb els
    llindars de volteig. Aqui es resta l'energia del repos, calculada per la
    MATEIXA funcio: no s'escriu cap formula de l'energia per segona vegada.
    """
    from pendol.fisica import energia

    return energia(estat, p) - energia(np.zeros(4), p)


def energia_volteig_2(p) -> float:
    """L'energia per posar el segon brac cap per amunt amb el primer quiet: 2 g B.

    COMPTE: documentacio/llindar-del-caos.html l'escriu 2*m2*g*d2, que es la
    mateixa cosa amb el vocabulari del model puntual. Aqui B = m2*d2 surt de
    fisica.Parametres, i l'autoprova comprova que es la diferencia d'energia
    entre (0, pi) i el repos.
    """
    return 2.0 * p.g * p.B


def energia_volteig_1(p) -> float:
    """L'energia per posar el primer brac cap per amunt amb el segon penjant: 2 g A.

    No es la (m1*d1 + m2*L1)*2g del document del llindar, que era del model
    puntual: al fisic A = m1*d1 + ms*rs + m2*L1, amb els cargols.
    """
    return 2.0 * p.g * p.A


def _energia_al_raig(r, c, s, p):
    """L'energia des del repos a (r c, r s), amb omega = 0. Per a la biseccio."""
    return p.g * p.A * (1.0 - np.cos(r * c)) + p.g * p.B * (1.0 - np.cos(r * s))


def corba_de_repos(E_rel: float, p, finor: int = 20000):
    r"""La mitja corba E_rel = constant del pla (th1, th2), amb omega = 0.

    Torna (th1, th2, arc): els punts, un per raig, i la longitud de corba
    acumulada fins a cadascun.

    COM ES RECORRE. Dins del quadrat [-pi, pi]^2, que cobreix tot el tor,
    l'energia creix MONOTONAMENT al llarg de qualsevol raig que surti de
    l'origen (dE/dr = gA c sin(rc) + gB s sin(rs) >= 0 mentre |rc|, |rs| <= pi),
    o sigui que cada raig talla la corba com a molt un cop i n'hi ha prou amb
    una biseccio per raig. Si el raig surt del quadrat abans d'arribar a
    l'energia, en aquella direccio no hi ha corba.

    NOMES MITJA CORBA. El pendol es simetric: (th1, th2) i (-th1, -th2) son el
    mateix moviment mirat a l'espill, amb la mateixa lambda. Els raigs van de
    phi = 0 a phi < pi.

    ELS TROSSOS SEPARATS. Per sobre de E_volteig,2 la corba toca les vores del
    quadrat i queda partida. La longitud es compta dins de cada tros, i el
    salt d'un tros a l'altre no hi suma.

    UN ANGLE VIU A E = E_volteig,2. Aquella corba passa exactament pel punt de
    sella (0, pi) -el segon brac just cap per amunt- i hi fa un vertex. No es
    cap error: es el punt on la corba canvia de forma, i els 45 graus hi son a
    tocar (E/E_v2 ~ 0,97).
    """
    phi = np.linspace(0.0, np.pi, finor, endpoint=False)
    c, s = np.cos(phi), np.sin(phi)
    rmax = np.pi / np.maximum(np.abs(c), np.abs(s))

    # _energia_al_raig NO es una segona formula de l'energia a qui fiar-se: es
    # energia_relativa amb omega = 0, escrita en vector per fer la biseccio de
    # molts raigs alhora. L'autoprova (E2) comprova cada punt que en surt amb
    # energia_relativa, que passa per fisica.energia.
    valid = _energia_al_raig(rmax, c, s, p) >= E_rel
    if not valid.any():
        raise ValueError(f"cap configuracio de repos te {E_rel:.4g} J")
    lo, hi = np.zeros(valid.sum()), rmax[valid]
    for _ in range(80):
        mig = 0.5 * (lo + hi)
        massa = _energia_al_raig(mig, c[valid], s[valid], p) >= E_rel
        hi = np.where(massa, mig, hi)
        lo = np.where(massa, lo, mig)
    r = 0.5 * (lo + hi)
    th1, th2 = r * c[valid], r * s[valid]

    idx = np.nonzero(valid)[0]
    passos = np.hypot(np.diff(th1), np.diff(th2))
    passos[np.diff(idx) != 1] = 0.0          # dos raigs no consecutius: un altre tros
    return th1, th2, np.concatenate([[0.0], np.cumsum(passos)])


def condicions_de_repos(E_rel: float, p, n: int, finor: int = 20000) -> np.ndarray:
    """n condicions inicials (th1, th2, 0, 0) amb energia E_rel, des del repos.

    Es la familia que es fa al laboratori: amollar del repos, amb tota
    l'energia potencial. Es mostreja A DISTANCIES IGUALS AL LLARG DE LA CORBA
    (corba_de_repos), no a angles phi iguals, perque cap tros no hi pesi mes
    que un altre.
    """
    th1, th2, arc = corba_de_repos(E_rel, p, finor)
    objectius = (np.arange(n) + 0.5) * arc[-1] / n
    triats = np.clip(np.searchsorted(arc, objectius), 0, len(arc) - 1)
    return np.column_stack([th1[triats], th2[triats], np.zeros(n), np.zeros(n)])


def benettin_pendol(p, estat0, temps_s: float, *, omega0: float | None = None,
                    d0: float = D0, interval: float = INTERVAL_REESCALAT_S,
                    transitori_s: float = TRANSITORI_S,
                    llavor: int = 0, dis=None, continua: dict | None = None) -> dict:
    """Benettin sobre el pendol doble, amb simulador.simula i la tolerancia estricta.

    La distancia es mesures.distancia_fases amb omega0 (per defecte, la del
    muntatge): la mateixa norma que fara servir el metode de parelles. La
    direccio inicial es aleatoria i uniforme a l'espai adimensional, amb llavor.

    S'integra SEMPRE a trossos de PAS_INTEGRACIO_S, faci el que faci
    l'interval: vegeu el capdamunt del modul. Despres de cada tros, cada
    trajectoria es recentra per separat (es treuen les voltes senceres dels seus
    angles), perque en 500 s de rotacio els angles desembolicats arriben a
    milers de radians i restar-ne dos es menjaria les xifres de la d0. La fisica
    no ho nota: fisica.py nomes en fa servir el sinus i el cosinus. Per aixo la
    diferencia que es reescala plega les voltes, igual que distancia_fases.

    Amb `dis`, el pendol perd energia i el que surt es una lambda de temps
    finit, DE PROVA mentre els coeficients no estiguin decidits.

    COMPTE AMB EL TEMPS QUAN HI HA FRICCIO. Sense friccio s'integra molt de
    temps (1000 s al pas 2) perque la lambda d'una orbita s'estabilitzi. Amb
    friccio aixo no te cap sentit: el pendol perd energia, s'atura, i aturat
    ja no hi ha res a mesurar. La lambda asimptotica d'un pendol que frega es
    negativa (tot cau al repos). L'unica que val es la de TEMPS FINIT durant el
    transitori caotic, mentre encara te energia per sobre del llindar, i la
    finestra s'ha de declarar en segons.
    """
    from pendol.fisica import energia
    from pendol.mesures import dif_angular, distancia_fases, omega_zero
    from pendol.simulador import ATOL, RTOL, simula

    w0 = omega_zero(p) if omega0 is None else float(omega0)
    trossos = interval / PAS_INTEGRACIO_S
    if abs(trossos - round(trossos)) > 1e-9 or round(trossos) < 1:
        raise ValueError(f"l'interval ({interval} s) ha de ser un multiple de "
                         f"PAS_INTEGRACIO_S ({PAS_INTEGRACIO_S} s)")

    def avanca(x, dt):
        for _ in range(int(round(dt / PAS_INTEGRACIO_S))):
            _, y = simula(p, x, t_eval=[0.0, PAS_INTEGRACIO_S], rtol=RTOL, atol=ATOL, dis=dis)
            x = y[:, -1].copy()
            x[:2] -= 2 * np.pi * np.round(x[:2] / (2 * np.pi))
        return x

    def distancia(a, b):
        return distancia_fases(a, b, w0)

    def diferencia(yy, xx):
        return np.array([dif_angular(yy[0], xx[0]), dif_angular(yy[1], xx[1]),
                         yy[2] - xx[2], yy[3] - xx[3]])

    if continua is None:
        z = np.random.default_rng(llavor).standard_normal(4)
        direccio = z * np.array([1.0, 1.0, w0, w0])   # uniforme a l'espai adimensional
    else:
        direccio = None
    res = benettin(avanca, distancia, estat0, direccio, temps_s, d0=d0, interval=interval,
                   transitori_s=transitori_s, diferencia=diferencia, continua=continua)

    # La deriva d'energia de la referencia, que es l'equivalent numeric del
    # terra de soroll: si deriva, part de l'«estructura» pot ser numerica.
    E = energia(res["referencies"].T, p)
    E_rel0 = float(energia_relativa(res["referencies"][0], p))
    res["omega0"] = w0
    res["pas_integracio_s"] = PAS_INTEGRACIO_S
    res["deriva_energia_J"] = float(np.max(np.abs(E - E[0])))
    res["deriva_energia_relativa"] = (res["deriva_energia_J"] / E_rel0
                                      if E_rel0 > 0 else float("nan"))
    res["energia_relativa_J"] = E_rel0
    res["amb_friccio"] = dis is not None
    return res


# ------------------------------------------------------ la seccio de Poincare
#
# LA CONDICIO, i per que no es «theta2 = 0». Els angles es desen DESEMBOLICATS:
# despres de la primera volta del segon brac, theta2 ja no torna a passar mai
# per zero, i una condicio literal perdria tots els creuaments d'alla endavant.
# La condicio de debo es geometrica: el segon brac passa per la vertical de sota
# (sin th2 = 0 amb cos th2 > 0) girant en sentit positiu (omega2 > 0). Sense la
# condicio de signe se superposen els creuaments dels dos sentits i l'estructura
# queda reflectida sobre ella mateixa (documentacio/tallar-espai-fases.html).
#
# DUES MANERES DE TROBAR ELS CREUAMENTS, per a dues menes de dades:
#   seccio_amb_esdeveniments  a la simulacio: l'integrador troba l'instant
#                             exacte per cerca d'arrels. Cap error de mostreig.
#   creuaments_interpolats    a dades mostrejades (el video, 240 fps):
#                             interpolacio lineal entre els dos fotogrames del
#                             creuament. MAI el fotograma mes proper: a 15 rad/s
#                             i 240 fps, mig fotograma son 1,8 graus, l'escala
#                             mateixa de l'estructura que es busca.

def seccio_amb_esdeveniments(camp, x0, temps_s: float, *, i_angle: int, i_velocitat: int,
                             args=(), rtol: float = 1e-12, atol: float = 1e-12,
                             metode: str = "DOP853", pas_maxim: float | None = None) -> dict:
    """Els estats exactes on sin(x[i_angle]) = 0, cos > 0 i x[i_velocitat] > 0.

    Un sol solve_ivp de 0 a temps_s, sense reiniciar. L'esdeveniment es
    sin(angle) creixent, que passa tant per 0 amb velocitat positiva com per pi
    amb velocitat negativa; es queden nomes els de cos > 0.

    ES NEGA A RESPONDRE SI POT HAVER PERDUT CREUAMENTS (29/09/2026). L'integrador
    nomes veu un creuament si l'angle canvia de signe entre el principi i el
    final d'un pas. Si en un pas l'angle es mou mitja volta (pi) o mes, hi
    poden cabre dos zeros del sinus i no se'n veuria cap. Per sota de pi es
    impossible, perque entre dos zeros del sinus hi ha exactament pi. La prova
    P2 ho va destapar: un rotor, que l'integrador travessa a passos de segons,
    tornava «0 creuaments» sense queixar-se. Ara, si passa, s'atura amb error i
    cal donar `pas_maxim`. Al pendol, amb la tolerancia estricta, els passos
    son de mil.lisegons i la comprovacio no hi canvia res: nomes vigila.
    """
    from scipy.integrate import solve_ivp

    def creua(t, x, *a):
        return math.sin(x[i_angle])
    creua.direction = 1.0

    # Sense t_eval, a posta: aixi sol.y porta l'estat al final de CADA pas i es
    # pot comprovar quant s'ha mogut l'angle en cadascun. La trajectoria no en
    # depen: solve_ivp tria els passos sense mirar t_eval.
    extra = {} if pas_maxim is None else {"max_step": float(pas_maxim)}
    sol = solve_ivp(camp, (0.0, float(temps_s)), np.asarray(x0, dtype=float), args=args,
                    method=metode, rtol=rtol, atol=atol, events=creua, **extra)
    if not sol.success:
        raise RuntimeError(f"la integracio ha fallat: {sol.message}")
    salts = np.abs(np.diff(sol.y[i_angle]))
    salt_maxim = float(salts.max()) if salts.size else 0.0
    if salt_maxim >= np.pi:
        raise RuntimeError(
            f"en un pas de l'integrador l'angle s'ha mogut {salt_maxim:.2f} rad (mitja volta o "
            "mes): s'hi podrien haver amagat creuaments. Dona un pas_maxim mes curt."
        )
    t, y = sol.t_events[0], sol.y_events[0]
    bons = (np.cos(y[:, i_angle]) > 0) & (y[:, i_velocitat] > 0) if len(t) else np.zeros(0, bool)
    return {"t_s": t[bons], "estats": y[bons], "final": sol.y[:, -1],
            "salt_maxim_rad": salt_maxim}


def seccio_pendol(p, estat0, temps_s: float, dis=None) -> dict:
    """La seccio del pendol doble: th2 per la vertical de sota amb omega2 > 0.

    Integra fisica.derivades amb EL MATEIX metode i la MATEIXA tolerancia
    estricta que pendol/simulador.py: simula() no admet esdeveniments, i per
    aixo aqui es crida solve_ivp directament, pero amb la fisica i els numeros
    del simulador, no amb uns de propis.

    Torna, a mes dels estats, (th1 plegat, omega1), que es el que es dibuixa,
    i l'energia a cada creuament: sense friccio ha de ser la inicial (si
    deriva, part de l'«estructura» pot ser numerica), i amb friccio es el que
    acoloreix els punts.
    """
    from pendol.fisica import derivades, energia
    from pendol.mesures import dif_angular
    from pendol.simulador import ATOL, METODE, RTOL

    r = seccio_amb_esdeveniments(derivades, estat0, temps_s, i_angle=1, i_velocitat=3,
                                 args=(p,) if dis is None else (p, dis),
                                 rtol=RTOL, atol=ATOL, metode=METODE)
    e = r["estats"]
    r["theta1_plegat"] = dif_angular(e[:, 0], 0.0) if len(e) else np.zeros(0)
    r["omega1"] = e[:, 2] if len(e) else np.zeros(0)
    r["energia_J"] = energia(e.T, p) if len(e) else np.zeros(0)
    r["energia_inicial_J"] = float(energia(np.asarray(estat0, dtype=float), p))
    r["amb_friccio"] = dis is not None
    return r


def condicions_a_la_seccio(E_rel: float, p, n: int, llavor: int = 0) -> np.ndarray:
    r"""n estats a la seccio (th2 = 0, omega2 > 0) amb energia E_rel, des del repos.

    Per dibuixar una seccio de Poincare, les orbites no han de sortir del repos
    (aquelles cauen totes a la mateixa corba), sino de punts repartits PER LA
    SECCIO MATEIXA, perque se'n vegin les illes i el mar. Es trien (th1, omega1)
    a l'atzar dins d'una capsa, i omega2 surt de l'energia:

        1/2 I2 w2^2 + K c w1 w2 + [1/2 I1 w1^2 + gA (1 - cos th1) - E] = 0,
        c = cos(th2 - th1) = cos(th1)

    agafant l'arrel POSITIVA (omega2 > 0, la condicio de signe de la seccio). Si
    no n'hi ha cap de positiva, aquell (th1, omega1) no es accessible i es
    descarta. Aquesta energia s'escriu aqui a ma: l'autoprova de
    eines/poincare.py (Q1) comprova cada estat amb fisica.energia, a 1e-12.
    """
    rng = np.random.default_rng(llavor)
    I1, I2, K = p.I1, p.I2, p.K
    # Una capsa que segur que conte tota la regio accessible: amb omega2 lliure,
    # l'energia cinetica minima per a un omega1 es 1/2 (I1 - K^2/I2) omega1^2.
    w1_max = math.sqrt(2.0 * E_rel / (I1 - K * K / I2))
    fora = []
    while len(fora) < n:
        th1 = rng.uniform(-np.pi, np.pi)
        w1 = rng.uniform(-w1_max, w1_max)
        c = math.cos(th1)
        a, b = 0.5 * I2, K * c * w1
        cc = 0.5 * I1 * w1 * w1 + p.g * p.A * (1.0 - c) - E_rel
        disc = b * b - 4.0 * a * cc
        if disc < 0:
            continue
        w2 = (-b + math.sqrt(disc)) / (2.0 * a)
        if w2 > 0:
            fora.append([th1, 0.0, w1, w2])
    return np.array(fora)


def creuaments_interpolats(t, estats, *, i_angle: int = 1, i_velocitat: int = 3) -> dict:
    """Els creuaments de la seccio en dades MOSTREJADES, per interpolacio lineal.

    `estats` es (n_components, N), com surt de simula() o de
    mesures.estats_de_taula(). L'angle es plega a [-pi, pi): un creuament es un
    pas de negatiu a no negatiu SENSE el salt de +pi a -pi (que es la vertical
    de dalt, no la de sota). Entre els fotogrames i i i+1:

        f = a[i] / (a[i] - a[i+1])        x* = x[i] + f * (x[i+1] - x[i])

    per a TOTS els components, desembolicats tal com venen (interpolar un
    angle plegat a traves del salt donaria disbarats). La velocitat
    interpolada ha de ser positiva: amb soroll, un fals creuament en sentit
    contrari hi pot fer semblar un pas.
    """
    from pendol.mesures import dif_angular

    t = np.asarray(t, dtype=float)
    x = np.asarray(estats, dtype=float)
    a = dif_angular(x[i_angle], 0.0)
    i = np.nonzero((a[:-1] < 0) & (a[1:] >= 0) & (np.abs(a[1:] - a[:-1]) < np.pi))[0]
    f = a[i] / (a[i] - a[i + 1])
    tt = t[i] + f * (t[i + 1] - t[i])
    xx = x[:, i] + f * (x[:, i + 1] - x[:, i])
    bons = xx[i_velocitat] > 0
    return {"t_s": tt[bons], "estats": xx[:, bons].T}


# ------------------------------------------------ el metode de parelles (A)
#
# L'estimador de l'experiment 4: <ln(d/d0)> sobre parelles de tirs, i el
# pendent del seu tros recte. Es l'unic que es pot fer amb el pendol real,
# perque no s'hi pot fer cap reescalat com a Benettin.
#
# LA REGLA PER LLEGIR LA LAMBDA: la «regla 2», aprovada per en Guillem el
# 03/10/2026, sense haver mirat cap presa real (documentacio/pla-analisi-caos.md,
# «1b»). Decideix quina lambda surt, i per aixo cada numero va aqui amb el seu
# motiu.
#
# HISTORIA, que consti. La primera regla (aprovada el 30/09) buscava un REPLA:
# el tram on el pendent local no es movia mes d'un +-20 %. Va fallar A1 i A2, i
# el diagnostic (esborranys/_diag_repla.txt) va ensenyar per que: el pendol
# s'estira A BATZEGADES a cada oscil.lacio (pendent local de -4 a +9 s^-1 en 12
# tirs simulats que surten del mateix lloc), i com que tots els tirs fan el
# mateix cami, la mitjana de les parelles no ho allisa. Aquell repla no existeix
# ni amb dades perfectes. La regla 2 no el busca: ajusta UNA recta sobre tota
# la zona util, que fa la mitjana de les batzegades.

FACTOR_TERRA = 3.0
FACTOR_SOSTRE = 10.0
# La zona util: <ln d> entre 3 vegades el terra de soroll i 10 vegades per
# sota de la saturacio. Amb d = 3 vegades el soroll, el soroll (que es suma al
# quadrat) hi afegeix un 5 %: sqrt(9 + 1) = 3,16. Per dalt, 10 i no 3: una
# saturacio logistica ja doblega la corba un 33 % a un terc del sostre, i un
# 10 % a una desena part.

CREIXEMENT_MINIM_LN = 1.0
RAO_TEMPS_ZONA = 2.0
# Nomes es dona lambda si, dins de la zona, la separacio creix com a minim un
# factor e (1 en ln d), i si la zona acaba com a minim al DOBLE del temps en
# que comenca (comptat des de l'amollada). La segona separa el caos de l'ordre:
# si la separacio creix LINEALMENT, ln d va com ln t, i un tram de t a 2t hi
# creix nomes ln 2 = 0,69, menys del factor e que demana la primera. Totes dues
# juntes rebutgen la separacio lineal (A4).

FINESTRA_PENDENT_S = 0.1
# Nomes per dibuixar el pendent local (pendent_local): ja no decideix res.

SATURACIO_DES_DE_S = 5.0
# El sostre es la d tipica de dues trajectories que ja no tenen res a veure:
# la mediana de d a partir de 5 s.

REMOSTREJOS = 1000
# L'error: 1000 remostrejos DE TIRS, no de parelles (les 66 parelles surten de
# 12 tirs i no son 66 mesures independents), tornant a buscar la zona a cada
# un. Aixi la incertesa de la finestra tambe hi entra.


def matriu_ln_d(estats, omega0: float) -> np.ndarray:
    """ln d entre cada parella de tirs: una matriu (N, N, M), simetrica.

    `estats` es (N_tirs, 4, M): tots els tirs sobre la MATEIXA graella de
    temps, comptada des de l'amollada. La distancia es mesures.distancia_fases,
    la mateixa norma a tot arreu. La diagonal (un tir amb ell mateix) es -inf i
    no es fa servir mai.
    """
    from pendol.mesures import distancia_fases

    x = np.asarray(estats, dtype=float)
    n, _, m = x.shape
    L = np.full((n, n, m), -np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            d = distancia_fases(x[i], x[j], omega0)
            if np.any(d <= 0):
                raise ValueError(f"els tirs {i} i {j} coincideixen exactament en algun instant")
            L[i, j] = L[j, i] = np.log(d)
    return L


def mitjana_log(ln_d, ln_d0):
    """<ln(d/d0)> i <ln d> sobre les parelles, a cada instant.

    La mitjana es dels LOGARITMES, mai el logaritme de la mitjana: amb
    creixement exponencial la d es aproximadament log-normal, i ln<d> la
    dominarien les poques parelles que mes s'han separat.
    """
    ln_d = np.asarray(ln_d, dtype=float)
    ln_d0 = np.asarray(ln_d0, dtype=float)
    return np.mean(ln_d - ln_d0[:, None], axis=0), np.mean(ln_d, axis=0)


def pendent_local(t, y, finestra_s: float = FINESTRA_PENDENT_S) -> np.ndarray:
    """El pendent per minims quadrats en una finestra centrada. nan a les vores.

    Demana una graella de temps regular (la comuna dels tirs).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0):
        raise ValueError("pendent_local demana una graella de temps regular")
    mig = max(1, int(round(0.5 * finestra_s / dt)))
    k = np.arange(-mig, mig + 1, dtype=float)
    s = np.full(len(y), np.nan)
    if len(y) > 2 * mig:
        s[mig:-mig] = np.correlate(y, k, mode="valid") / (dt * np.sum(k * k))
    return s


def ajusta_zona(t, m_norm, m_ln_d, terra: float, sostre: float, *,
                factor_terra: float = FACTOR_TERRA, factor_sostre: float = FACTOR_SOSTRE,
                creixement_minim: float = CREIXEMENT_MINIM_LN,
                rao_temps: float = RAO_TEMPS_ZONA) -> dict:
    """La lambda de <ln(d/d0)> per la regla 2 (aprovada el 03/10/2026).

    `t` es compta des de l'amollada (el zero importa: vegeu RAO_TEMPS_ZONA).

    La zona util es on <ln d> es entre ln(factor_terra x terra) i
    ln(sostre / factor_sostre). Si la corba hi entra i en surt mes d'un cop, es
    pren el tram continu MES LLARG en temps (si n'hi ha dos d'iguals, el que
    comenca abans). La lambda es el pendent de la recta ajustada a tot aquell
    tram, sense buscar-hi cap repla. Nomes es dona si el tram hi creix com a
    minim `creixement_minim` en ln d i acaba com a minim a `rao_temps` vegades
    el temps en que comenca.
    """
    t = np.asarray(t, dtype=float)
    m_ln_d = np.asarray(m_ln_d, dtype=float)
    m_norm = np.asarray(m_norm, dtype=float)
    util = ((m_ln_d >= math.log(factor_terra * terra))
            & (m_ln_d <= math.log(sostre / factor_sostre)) & np.isfinite(m_norm))
    base = {"ok": False, "terra": terra, "sostre": sostre}
    if not util.any():
        return {**base, "motiu": "la d no passa mai per la zona util "
                                 f"({factor_terra:g} x terra a sostre / {factor_sostre:g})"}
    idx = np.nonzero(util)[0]
    trams = np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1)
    tram = max(trams, key=lambda v: (t[v[-1]] - t[v[0]], -v[0]))
    i, j = int(tram[0]), int(tram[-1])
    creix = float(m_ln_d[j] - m_ln_d[i])
    base.update(t_inici_s=float(t[i]), t_fi_s=float(t[j]), durada_s=float(t[j] - t[i]),
                creixement_ln=creix, n_punts=int(j - i + 1))
    if j - i < 2:
        return {**base, "motiu": "la zona util te menys de tres punts"}
    tt, yy = t[i:j + 1], m_norm[i:j + 1]
    pendent, ordenada = np.polyfit(tt, yy, 1)
    residu = yy - (pendent * tt + ordenada)
    r2 = 1.0 - float(np.sum(residu ** 2)) / float(np.sum((yy - yy.mean()) ** 2))
    base.update(pendent=float(pendent), r2=r2)
    if creix < creixement_minim:
        return {**base, "motiu": f"dins de la zona la separacio nomes creix {creix:.2f} en ln d "
                                 f"(cal {creixement_minim:g}, un factor e)"}
    if t[j] < rao_temps * t[i]:
        return {**base, "motiu": f"la zona va de {t[i]:.3f} a {t[j]:.3f} s: no arriba al "
                                 f"doble del temps en que comenca, i no es pot distingir d'una "
                                 "separacio lineal"}
    return {**base, "ok": True, "motiu": None, "lambda": float(pendent)}


def lambda_de_corbes(t, ln_d, ln_d0, terra: float, sostre: float, **opcions) -> dict:
    """La lambda d'un conjunt de corbes ln d(t), una per parella."""
    m_norm, m_ln = mitjana_log(ln_d, ln_d0)
    r = ajusta_zona(t, m_norm, m_ln, terra, sostre, **opcions)
    r["n_parelles"] = int(np.asarray(ln_d).shape[0])
    r["corba"] = m_norm
    return r


def sostre_de_saturacio(t, ln_d, des_de_s: float = SATURACIO_DES_DE_S) -> float:
    """La mediana de d a partir de `des_de_s`, sobre totes les parelles."""
    t = np.asarray(t, dtype=float)
    tard = t >= des_de_s
    if not tard.any():
        raise ValueError(f"els tirs no arriben a {des_de_s:g} s: dona el sostre de fora")
    return float(np.median(np.exp(np.asarray(ln_d)[:, tard])))


def _parelles_de(tirs):
    """Les parelles d'una llista d'indexs de tirs, SENSE les d'un tir amb ell mateix.

    En un remostreig amb reemplacament un tir pot sortir dues vegades; aquella
    parella no es cap mesura (d = 0) i no hi entra.
    """
    return [(tirs[a], tirs[b]) for a in range(len(tirs)) for b in range(a + 1, len(tirs))
            if tirs[a] != tirs[b]]


def lambda_de_tirs(t, L, L0, terra: float, sostre: float, tirs=None, **opcions) -> dict:
    """La lambda d'un grup de tirs, amb la matriu de matriu_ln_d.

    `L0` (N, N) es ln d0 de cada parella: la separacio a l'amollada. Per
    defecte es la del primer instant; a les preses reals s'ha de donar la dels
    angles de repos mesurats, que no porta el soroll d'un sol fotograma.
    """
    n = L.shape[0]
    tirs = list(range(n)) if tirs is None else list(tirs)
    parelles = _parelles_de(tirs)
    if len(parelles) < 1:
        return {"ok": False, "motiu": "menys de dues preses diferents", "n_parelles": 0}
    ln_d = np.array([L[i, j] for i, j in parelles])
    ln_d0 = np.array([L0[i, j] for i, j in parelles])
    return lambda_de_corbes(t, ln_d, ln_d0, terra, sostre, **opcions)


def bootstrap_per_tirs(t, L, L0, terra: float, sostre: float, n: int = REMOSTREJOS,
                       llavor: int = 0, **opcions) -> dict:
    """L'error de la lambda remostrejant TIRS, i tornant a buscar la zona cada vegada.

    El sostre es el de la mostra sencera i no es torna a calcular: es una
    propietat del sistema (quant es poden separar dues trajectories), no de
    quins tirs s'han agafat.
    """
    rng = np.random.default_rng(llavor)
    N = L.shape[0]
    lambdes, sense = [], 0
    for _ in range(n):
        r = lambda_de_tirs(t, L, L0, terra, sostre, tirs=rng.integers(0, N, N).tolist(),
                           **opcions)
        if r["ok"]:
            lambdes.append(r["lambda"])
        else:
            sense += 1
    v = np.array(lambdes)
    return {
        "remostrejos": n,
        "sense_lambda": sense,
        "desviacio": float(v.std(ddof=1)) if len(v) > 1 else None,
        "percentils_16_84": [float(x) for x in np.percentile(v, [16, 84])] if len(v) else None,
        "percentils_2_5_97_5": [float(x) for x in np.percentile(v, [2.5, 97.5])] if len(v) else None,
    }


# --------------------------------------------------------------- l'autoprova

def _avanca_ivp(camp, rtol=1e-12, atol=1e-12):
    """Un avanca() per a un camp qualsevol, amb DOP853."""
    from scipy.integrate import solve_ivp

    def avanca(x, dt):
        sol = solve_ivp(camp, (0.0, dt), x, method="DOP853", rtol=rtol, atol=atol)
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y[:, -1]
    return avanca


def _prova_E(mira, p, w0, rng):
    """E1-E2: l'energia des del repos, els llindars i la familia de repos."""
    # --- E1. Els llindars de volteig, contra fisica.energia --------------
    print("E1. Els llindars de volteig, per dos camins")
    Ev2, Ev1 = energia_volteig_2(p), energia_volteig_1(p)
    e2 = float(energia_relativa(np.array([0.0, np.pi, 0.0, 0.0]), p))
    e1 = float(energia_relativa(np.array([np.pi, 0.0, 0.0, 0.0]), p))
    print(f"      E_volteig,2 = 2gB = {1000 * Ev2:.4f} mJ;  E_volteig,1 = 2gA = {1000 * Ev1:.4f} mJ")
    mira(abs(e2 / Ev2 - 1) < 1e-12 and abs(e1 / Ev1 - 1) < 1e-12,
         "2gB i 2gA son les diferencies d'energia de fisica.energia entre (0, pi), (pi, 0) i el repos")

    # --- E2. La familia de repos ------------------------------------------
    # La separacio es mesura AL LLARG DE LA CORBA, que es el que diu el criteri.
    # Fins al 29/09 es mesurava en linia recta (la corda), i a E = E_volteig,2
    # la corba fa un vertex al punt de sella (0, pi): alli la corda es molt mes
    # curta que l'arc i la prova fallava sense que hi hagues cap error. L'arc
    # es mesura sobre una corba deu vegades mes fina que la que fa servir
    # condicions_de_repos, o sigui que no es el mateix calcul repetit.
    print("\nE2. Les condicions inicials des del repos")
    pitjor, fora, desigual = 0.0, False, 0.0
    for rao in (0.3, 0.97, 1.0, 3.3, 6.2):
        E = rao * Ev2
        ci = condicions_de_repos(E, p, 40)
        err = np.abs(energia_relativa(ci.T, p) - E) / E
        pitjor = max(pitjor, float(err.max()))
        fora |= bool(np.any(np.abs(ci[:, :2]) > np.pi + 1e-12) or np.any(ci[:, 2:] != 0))
        f1, f2, arc = corba_de_repos(E, p, finor=200000)
        on = [int(np.argmin(np.hypot(f1 - a, f2 - b))) for a, b in ci[:, :2]]
        sep = np.diff(arc[on])
        desigual = max(desigual, float(np.max(np.abs(sep / (arc[-1] / 40) - 1))))
    mira(pitjor < 1e-12, f"totes tenen l'energia demanada (pitjor error relatiu {pitjor:.1e})")
    mira(not fora, "totes son dins del quadrat [-pi, pi]^2 i amb omega = 0 exacte")
    mira(desigual < 0.05, f"repartides a distancies iguals al llarg de la corba "
                          f"(desviacio maxima {100 * desigual:.2f} %)")



def _prova_B(mira, p, w0, rng):
    """B1-B5: Benettin contra lambdes conegudes, i sobre el pendol."""
    import time

    from scipy.linalg import expm

    from pendol.mesures import dif_angular

    # --- B1. La sella lineal: lambda = k exacta ---------------------------
    print("\nB1. Sella lineal x'' = k^2 x, k = 2: lambda = 2 exacta (criteri < 1 %)")
    k = 2.0
    M = np.array([[0.0, 1.0], [k * k, 0.0]])
    mapa = {}

    def avanca_sella(x, dt):
        if dt not in mapa:
            mapa[dt] = expm(M * dt)
        return mapa[dt] @ x

    # La referencia es el punt fix mateix (l'origen): aixi no creix i la resta
    # y - x no perd xifres. La lambda d'un punt fix es el seu valor propi.
    r = benettin(avanca_sella, lambda a, b: float(np.linalg.norm(a - b)), np.zeros(2),
                 rng.standard_normal(2), 20.0, interval=0.2, transitori_s=2.0)
    mira(abs(r["lambda"] / k - 1) < 0.01, f"lambda = {r['lambda']:.6f} (error "
                                          f"{100 * abs(r['lambda'] / k - 1):.4f} %)")

    # --- B2. Lorenz: 0,9056 ------------------------------------------------
    print("\nB2. Lorenz (10, 28, 8/3): lambda1 = 0,9056 de la bibliografia (criteri < 3 %)")

    def lorenz(_, v):
        x, y, z = v
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]

    av = _avanca_ivp(lorenz)
    x0 = av(np.array([1.0, 1.0, 1.0]), 50.0)          # ja dins de l'atractor
    ara = time.perf_counter()
    r = benettin(av, lambda a, b: float(np.linalg.norm(a - b)), x0, rng.standard_normal(3),
                 1000.0, interval=0.1, transitori_s=20.0)
    err = abs(r["lambda"] / 0.9056 - 1)
    mira(err < 0.03, f"lambda1 = {r['lambda']:.4f} en T = 1000 (error {100 * err:.2f} %, "
                     f"d/d0 maxima per interval {r['rao_max_per_pas']:.1f}, "
                     f"{time.perf_counter() - ara:.0f} s)")

    # --- B3. Dos sistemes integrables -------------------------------------
    print("\nB3. Integrables: |lambda(500 s)| < 0,05 s^-1 i baixant com ln T / T")

    def oscil(_, v):
        return [v[1], -4.0 * v[0]]           # frequencia 2 rad/s

    # La norma amb omega0 = 1, que NO es la frequencia de l'oscil.lador: amb la
    # seva, el flux seria una rotacio pura i la separacio ni tan sols oscil.laria.
    r = benettin(_avanca_ivp(oscil), lambda a, b: float(np.hypot(a[0] - b[0], a[1] - b[1])),
                 np.array([1.0, 0.0]), rng.standard_normal(2), 500.0, interval=0.2)
    c = classifica(r)
    mira(c["mena"] == "regular", f"oscil.lador harmonic: lambda(T) = {c['lambda_T']:+.2e}, "
                                 f"lambda(T/4) = {c['lambda_025T']:+.2e} -> {c['mena']}")

    def simple(_, v):
        return [v[1], -w0 * w0 * np.sin(v[0])]

    def dist_simple(a, b):
        return float(np.hypot(dif_angular(a[0], b[0]), (a[1] - b[1]) / w0))

    r = benettin(_avanca_ivp(simple), dist_simple, np.array([np.radians(120.0), 0.0]),
                 rng.standard_normal(2), 500.0, interval=0.2)
    c = classifica(r)
    mira(c["mena"] == "regular", f"pendol simple a 120 graus, omega0 = {w0:.2f}: "
                                 f"lambda(T) = {c['lambda_T']:.4f}, lambda(T/4) = "
                                 f"{c['lambda_025T']:.4f} -> {c['mena']}")

    # --- B4. El pendol doble quasiperiodic ---------------------------------
    print("\nB4. Pendol doble de fisica.py a 1 / 1 graus (quasiperiodic): com B3")
    ara = time.perf_counter()
    r = benettin_pendol(p, np.radians([1.0, 1.0, 0.0, 0.0]), 500.0, llavor=1)
    c = classifica(r)
    mira(c["mena"] == "regular", f"lambda(T) = {c['lambda_T']:.4f}, lambda(T/4) = "
                                 f"{c['lambda_025T']:.4f} -> {c['mena']} "
                                 f"({time.perf_counter() - ara:.0f} s)")
    print(f"      deriva d'energia {r['deriva_energia_J']:.1e} J "
          f"({r['deriva_energia_relativa']:.1e} de l'energia des del repos)")

    # --- B5. La lambda es del sistema, no de la norma ni de l'interval -------
    # Totes les variants integren a trossos de PAS_INTEGRACIO_S, o sigui que la
    # referencia es EXACTAMENT la mateixa orbita: el que es compara es el
    # calcul, no dues orbites diferents.
    print("\nB5. Pendol doble a 150 / 150 graus: la lambda no es mou mes d'un 3 %")
    estat = np.radians([150.0, 150.0, 0.0, 0.0])
    T5 = 100.0
    ara = time.perf_counter()
    base = benettin_pendol(p, estat, T5, llavor=5)
    print(f"      base (omega0 = {w0:.3f}, d0 = 1e-8, interval = {INTERVAL_REESCALAT_S} s): "
          f"lambda = {base['lambda']:.4f} s^-1, d/d0 maxima per interval "
          f"{base['rao_max_per_pas']:.2f}, deriva d'energia "
          f"{base['deriva_energia_relativa']:.1e} ({time.perf_counter() - ara:.0f} s)")
    mira(base["rao_max_per_pas"] < RAO_MAXIMA_PER_PAS,
         f"d/d0 no passa de {RAO_MAXIMA_PER_PAS:g} en cap interval")
    variants = {
        "omega0 x3": dict(omega0=3 * w0),
        "d0 = 1e-7": dict(d0=1e-7),
        "interval 0,1": dict(interval=0.1),
    }
    for nom, kw in variants.items():
        r = benettin_pendol(p, estat, T5, llavor=5, **kw)
        # La referencia es desa a cada interval: amb un interval doble, una de
        # cada dues de la base. Ha de ser la mateixa BIT A BIT.
        cada = int(round(r["interval_s"] / base["interval_s"]))
        mateixa = np.array_equal(r["referencies"], base["referencies"][::cada])
        rel = abs(r["lambda"] / base["lambda"] - 1)
        mira(rel < 0.03, f"{nom:<12}: lambda = {r['lambda']:.4f} ({100 * rel:.2f} %)"
                         f"{'' if mateixa else '  <<< la referencia NO es la mateixa orbita'}")
        if not mateixa:
            mira(False, f"{nom}: la referencia no ha seguit la mateixa orbita que la base")


def _prova_P(mira, p, w0, rng):
    """P1-P4: la seccio de Poincare (part 1c, criteris del 29/09)."""
    from pendol.mesures import dif_angular

    # --- P1-P4. La seccio de Poincare (part 1c, criteris del 29/09) ----------
    print("\nP1. Dos oscil.ladors desacoblats incommensurables: a la seccio, E1 constant a 1e-9")

    def dos_oscil(_, v):
        q1, p1_, q2, p2 = v
        return [p1_, -q1, p2, -2.0 * q2]        # frequencies 1 i sqrt(2)

    x0 = np.array([0.7, 0.0, 1.0, 0.3])          # |q2| < pi/2: nomes creua per q2 = 0
    s = seccio_amb_esdeveniments(dos_oscil, x0, 2000.0, i_angle=2, i_velocitat=3)
    E1 = 0.5 * (s["estats"][:, 1] ** 2 + s["estats"][:, 0] ** 2)
    E1_0 = 0.5 * (x0[1] ** 2 + x0[0] ** 2)
    dev = float(np.max(np.abs(E1 / E1_0 - 1)))
    mira(dev < 1e-9 and len(E1) > 100,
         f"{len(E1)} punts, E1 constant a {dev:.1e}")

    print("\nP2. Rotor lliure amb l'angle desembolicat: exactament floor(w T / 2 pi) creuaments")
    w_r, th0, T_r = 3.7, 0.1, 100.0

    def rotor(_, v):
        return [v[1], 0.0]

    # Primer, sense pas maxim: l'integrador hi fa passos de segons i s'hi
    # amagarien creuaments. S'ha de NEGAR a respondre, no donar-ne menys.
    try:
        seccio_amb_esdeveniments(rotor, [th0, w_r], T_r, i_angle=0, i_velocitat=1)
        es_nega = False
    except RuntimeError:
        es_nega = True
    mira(es_nega, "sense pas maxim es nega a respondre, en lloc de tornar-ne menys en silenci")
    # I amb un pas maxim que no deixa passar de mitja volta (3,7 x 0,5 = 1,85 rad):
    s = seccio_amb_esdeveniments(rotor, [th0, w_r], T_r, i_angle=0, i_velocitat=1,
                                 pas_maxim=0.5)
    n_esperat = int(math.floor((th0 + w_r * T_r) / (2 * np.pi)))
    t_esperat = (2 * np.pi * np.arange(1, n_esperat + 1) - th0) / w_r
    ok_t = len(s["t_s"]) == n_esperat and np.allclose(s["t_s"], t_esperat, atol=1e-9, rtol=0)
    mira(len(s["t_s"]) == n_esperat and ok_t,
         f"{len(s['t_s'])} creuaments (n'hi ha d'haver {n_esperat}), als instants exactes a 1e-9 s")

    print("\nP3. Creuaments interpolats d'una simulacio a 240 fps, contra els esdeveniments")
    from pendol.simulador import simula

    estat = np.radians([150.0, 150.0, 0.0, 0.0])
    ref = seccio_pendol(p, estat, 20.0)
    t240 = np.arange(0.0, 20.0, 1.0 / 240.0)
    _, y240 = simula(p, estat, t_eval=t240)
    interp = creuaments_interpolats(t240, y240)
    parells = [(int(np.argmin(np.abs(ref["t_s"] - tc))), k) for k, tc in enumerate(interp["t_s"])]
    errs = np.degrees(np.abs(dif_angular(interp["estats"][[k for _, k in parells], 0],
                                         ref["estats"][[j for j, _ in parells], 0])))
    mira(abs(len(interp["t_s"]) - len(ref["t_s"])) <= 1,
         f"{len(interp['t_s'])} creuaments interpolats i {len(ref['t_s'])} d'esdeveniments")
    mira(float(errs.mean()) < 0.5, f"error mitja de theta1 {errs.mean():.4f} graus "
                                   f"(maxim {errs.max():.4f}), sense soroll")

    print("\nP4. Els punts de la seccio conservativa compleixen la condicio i tenen l'energia inicial")
    s = seccio_pendol(p, estat, 200.0)
    e = s["estats"]
    condicio = bool(np.all(np.abs(np.sin(e[:, 1])) < 1e-9) and np.all(np.cos(e[:, 1]) > 0)
                    and np.all(e[:, 3] > 0))
    E_rel0 = float(energia_relativa(estat, p))
    dev = float(np.max(np.abs(s["energia_J"] - s["energia_inicial_J"])) / E_rel0)
    mira(condicio, f"{len(e)} punts en 200 s, tots amb sin th2 = 0, cos th2 > 0 i omega2 > 0 "
                   f"(el pas mes gran hi mou th2 {s['salt_maxim_rad']:.3f} rad)")
    mira(dev < 1e-9, f"energia constant a {dev:.1e} de l'energia des del repos")


# ------------------------------------------- l'autoprova del metode de parelles
#
# Totes les corbes d'aqui son FABRICADES: cap presa real. El soroll es suma al
# quadrat a la d (independent de la separacio), i la saturacio es logistica.

_T_PAR = np.arange(0.0, 6.0, 1.0 / 240.0)      # la graella de les preses, fins a 6 s
_TERRA_PAR = 0.035                             # ~2 graus, com el dels 90 i 150


def _shots_fabricats(llavor: int, N: int = 12, lam: float = 2.5, sd_tir: float = 0.3,
                     sd_parella: float = 0.2, d0_mig: float = _TERRA_PAR, sd_d0: float = 0.5,
                     sostre_log: float = 2.0, soroll: bool = True, lineal: bool = False):
    """N tirs fabricats: la matriu (N, N, M) de ln d, i la de ln d0.

    Cada tir porta el seu propi desplacament del ritme (sd_tir): dues parelles
    que comparteixen tir s'assemblen, com a les preses de debo. Amb lineal=True
    la separacio creix com d0 (1 + c t), que es el que fa una orbita regular.
    """
    rng = np.random.default_rng(llavor)
    a = rng.normal(0.0, sd_tir, N)
    M = len(_T_PAR)
    L = np.full((N, N, M), -np.inf)
    L0 = np.full((N, N), -np.inf)
    for i in range(N):
        for j in range(i + 1, N):
            d0 = d0_mig * math.exp(rng.normal(0.0, sd_d0))
            if lineal:
                d = d0 * (1.0 + rng.uniform(1.0, 4.0) * _T_PAR)
            else:
                ritme = lam + 0.5 * (a[i] + a[j]) + rng.normal(0.0, sd_parella)
                e = np.exp(ritme * _T_PAR)
                d = d0 * e / (1.0 + d0 * (e - 1.0) / sostre_log)
            if soroll:
                d = np.sqrt(d * d + _TERRA_PAR ** 2 * (0.5 + 0.5 * rng.normal(size=M) ** 2))
            L[i, j] = L[j, i] = np.log(d)
            L0[i, j] = L0[j, i] = math.log(d0)
    return L, L0


def _cobertura_una(llavor: int) -> tuple:
    """Una repeticio de A3: la lambda i el seu error de bootstrap. Va en un proces."""
    L, L0 = _shots_fabricats(llavor, soroll=False, sd_d0=0.5, d0_mig=0.01,
                             sostre_log=1e9)
    # Exponencial pura, sense saturacio ni terra: la lambda de la poblacio es
    # 2,5 exacta i l'estimador no hi posa biaix. Aixi A3 prova NOMES l'error.
    rng = np.random.default_rng(llavor + 1)
    L = L + rng.normal(0.0, 0.05, L.shape)
    t = _T_PAR[_T_PAR <= 1.8]
    L = L[:, :, :len(t)]
    r = lambda_de_tirs(t, L, L0, 0.01, 1.64)
    b = bootstrap_per_tirs(t, L, L0, 0.01, 1.64, n=200, llavor=llavor + 2)
    return (r.get("lambda"), b["desviacio"], b["sense_lambda"])


def _prova_A(mira, p, w0, rng):
    """A1-A5: el metode de parelles, sobre corbes fabricades i sobre Lorenz."""
    import multiprocessing
    import time

    from scipy.integrate import solve_ivp

    t = _T_PAR

    # A1 es parteix en dos des del 03/10 (aprovat per en Guillem amb la regla 2).
    # La A1 del 29/09 (la lambda a < 5 % amb el terra real i saturacio) no la
    # podia passar CAP regla: el pendent de debo d'aquelles corbes no arriba
    # mai a 2,5 dins de la zona util (esborranys/_diag_repla.txt). Ara:
    #   A1a prova el CODI, amb molt d'espai entre el terra i la saturacio;
    #   A1b MESURA el biaix amb l'espai real, i el publica sense criteri.
    rng1 = np.random.default_rng(101)
    P = 66
    d0 = _TERRA_PAR * np.exp(rng1.normal(0.0, 0.5, P))
    e = np.exp(2.5 * t)
    soroll = 0.5 + 0.5 * rng1.normal(size=(P, len(t))) ** 2

    print("\nA1a. Exponencial pura amb molt d'espai (terra 1e-5, sense saturacio): la lambda a < 1 %")
    d = np.sqrt((d0[:, None] * e) ** 2 + 1e-10 * soroll)
    ln_d1, ln_d0 = np.log(d), np.log(d0)
    r = lambda_de_corbes(t, ln_d1, ln_d0, 1e-5, sostre_de_saturacio(t, ln_d1))
    if r["ok"]:
        err = abs(r["lambda"] / 2.5 - 1)
        mira(err < 0.01, f"lambda = {r['lambda']:.4f} (error {100 * err:.3f} %), zona de "
                         f"{r['t_inici_s']:.3f} a {r['t_fi_s']:.3f} s, R2 = {r['r2']:.5f}")
    else:
        mira(False, f"no dona lambda: {r['motiu']}")

    print("\nA1b. Amb el terra de 2 graus i saturacio logistica: el BIAIX, mesurat (sense criteri)")
    d = d0[:, None] * e / (1.0 + d0[:, None] * (e - 1.0) / 2.0)
    d = np.sqrt(d * d + _TERRA_PAR ** 2 * soroll)
    ln_d, ln_d0 = np.log(d), np.log(d0)
    sostre = sostre_de_saturacio(t, ln_d)
    rb = lambda_de_corbes(t, ln_d, ln_d0, _TERRA_PAR, sostre)
    if rb["ok"]:
        print(f"      lambda = {rb['lambda']:.4f}: biaix {100 * (rb['lambda'] / 2.5 - 1):+.1f} %, "
              f"zona de {rb['t_inici_s']:.3f} a {rb['t_fi_s']:.3f} s")
    else:
        print(f"      sense lambda: {rb['motiu']}. Amb un terra de 2 graus aquesta corba no "
              "deixa prou espai")

    # La mitjana dels logaritmes, contra el logaritme de la mitjana: parelles
    # amb ritmes diferents i sense soroll. <ln(d/d0)> ha de donar la mitjana
    # dels ritmes; ln<d> la dominen les que van mes de pressa.
    ritmes = np.random.default_rng(102).uniform(1.5, 3.5, P)
    ln_d = np.log(0.01) + ritmes[:, None] * t[None, :]
    m_norm, _ = mitjana_log(ln_d, np.full(P, np.log(0.01)))
    fin = (t >= 0.5) & (t <= 1.5)
    p_log = np.polyfit(t[fin], m_norm[fin], 1)[0]
    p_mitj = np.polyfit(t[fin], np.log(np.mean(np.exp(ln_d), axis=0))[fin], 1)[0]
    mira(abs(p_log / ritmes.mean() - 1) < 1e-9 and abs(p_mitj / ritmes.mean() - 1) > 0.05,
         f"<ln d> dona la mitjana dels ritmes ({p_log:.4f} = {ritmes.mean():.4f}); "
         f"ln<d> dona {p_mitj:.4f}, un {100 * (p_mitj / ritmes.mean() - 1):+.0f} %")

    print("\nA5. El temps en mil.lisegons: la lambda es divideix exactament per 1000")
    # Sobre les corbes d'A1a. La regla 2 no te cap parametre en segons: la
    # zona es en unitats de d, i la rao de temps no en te.
    rms = lambda_de_corbes(t * 1000.0, ln_d1, ln_d0, 1e-5, sostre_de_saturacio(t, ln_d1))
    ok5 = (r["ok"] and rms["ok"] and rms["n_punts"] == r["n_punts"]
           and abs(rms["lambda"] * 1000.0 / r["lambda"] - 1) < 1e-9)
    mira(ok5, f"lambda en ms^-1 x 1000 = {rms.get('lambda', float('nan')) * 1000:.6f}, "
              f"en s^-1 = {r.get('lambda', float('nan')):.6f}, i la mateixa zona")

    print("\nA2. Lorenz (10, 28, 8/3), 200 parelles de punts de l'atractor: 0,906 a < 10 %")

    def lorenz2(_, v):
        out = []
        for k in (0, 3):
            x, y, z = v[k:k + 3]
            out += [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]
        return out

    def lorenz(_, v):
        x, y, z = v
        return [10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z]

    ara = time.perf_counter()
    base = solve_ivp(lorenz, (0.0, 250.0), [1.0, 1.0, 1.0], method="DOP853", rtol=1e-10,
                     atol=1e-10, t_eval=np.arange(50.0, 250.0, 1.0)).y.T
    rngL = np.random.default_rng(103)
    tL = np.arange(0.0, 30.0, 0.01)
    d0L = 1e-6
    ln_dL = []
    for x0 in base:
        u = rngL.standard_normal(3)
        y0 = x0 + d0L * u / np.linalg.norm(u)
        s = solve_ivp(lorenz2, (0.0, tL[-1]), np.concatenate([x0, y0]), method="DOP853",
                      rtol=1e-10, atol=1e-12, t_eval=tL)
        ln_dL.append(np.log(np.linalg.norm(s.y[:3] - s.y[3:], axis=0)))
    ln_dL = np.array(ln_dL)
    # El sostre, a partir de 25 unitats: el SATURACIO_DES_DE_S = 5 s es del
    # pendol, i a Lorenz, amb d0 = 1e-6, a 5 unitats encara no ha saturat res.
    sostreL = sostre_de_saturacio(tL, ln_dL, des_de_s=25.0)
    rL = lambda_de_corbes(tL, ln_dL, np.full(len(ln_dL), np.log(d0L)), d0L, sostreL)
    if rL["ok"]:
        err = abs(rL["lambda"] / 0.9056 - 1)
        mira(err < 0.10, f"lambda = {rL['lambda']:.4f} (error {100 * err:.1f} %), zona de "
                         f"{rL['t_inici_s']:.2f} a {rL['t_fi_s']:.2f}, R2 = {rL['r2']:.4f} "
                         f"({time.perf_counter() - ara:.0f} s)")
    else:
        mira(False, f"no dona lambda a Lorenz: {rL['motiu']}")

    print("\nA4. Separacio LINEAL (una orbita regular): cap lambda, o lambda compatible amb 0 a 2 sigma")
    for nom, kw in (("d0 com el terra (com 90 i 150)", dict(d0_mig=_TERRA_PAR)),
                    ("d0 = 4 x terra (com els 45)", dict(d0_mig=4 * _TERRA_PAR, sd_d0=0.2))):
        L, L0 = _shots_fabricats(104, lineal=True, **kw)
        sostre4 = sostre_de_saturacio(t, np.array([L[i, j] for i, j in _parelles_de(list(range(12)))]))
        r4 = lambda_de_tirs(t, L, L0, _TERRA_PAR, sostre4)
        if not r4["ok"]:
            mira(True, f"{nom}: no dona lambda ({r4['motiu']})")
        else:
            b4 = bootstrap_per_tirs(t, L, L0, _TERRA_PAR, sostre4, n=200, llavor=105)
            compatible = b4["desviacio"] is not None and abs(r4["lambda"]) <= 2 * b4["desviacio"]
            mira(compatible, f"{nom}: HI DONA LAMBDA, zona de {r4['t_inici_s']:.3f} a "
                             f"{r4['t_fi_s']:.3f} s, lambda = {r4['lambda']:.3f} +- "
                             f"{b4['desviacio']:.3f} (creix {r4['creixement_ln']:.2f} en ln d)")

    print("\nA3. Cobertura del bootstrap per tirs: 200 repeticions, la lambda certa dins d'1 sigma"
          " entre el 55 i el 80 %")
    ara = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(6) as pool:
        res = pool.map(_cobertura_una, [1000 + k for k in range(200)])
    bons = [(l, s) for l, s, _ in res if l is not None and s is not None]
    dins = sum(1 for l, s in bons if abs(l - 2.5) <= s)
    frac = dins / len(bons) if bons else 0.0
    mira(len(bons) == 200 and 0.55 <= frac <= 0.80,
         f"{dins} de {len(bons)} dins d'1 sigma ({100 * frac:.0f} %); sense lambda en "
         f"{sum(n for _, _, n in res)} remostrejos de {200 * 200} "
         f"({time.perf_counter() - ara:.0f} s)")


def autoprova(parts: str = "EBPA") -> int:
    """Els criteris de documentacio/pla-analisi-caos.md (29/09/2026).

    E1-E2 l'energia i la familia de repos    B1-B5 Benettin
    P1-P4 la seccio de Poincare              A1-A5 el metode de parelles
    """
    from pendol.fisica import Parametres
    from pendol.mesures import omega_zero

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json("parametres.json")
    w0 = omega_zero(p)
    rng = np.random.default_rng(29)
    print(f"Autoprova de pendol/caos.py, parts {parts} (criteris fixats el 29/09/2026)\n",
          flush=True)
    for lletra, prova in (("E", _prova_E), ("B", _prova_B), ("P", _prova_P), ("A", _prova_A)):
        if lletra in parts:
            prova(mira, p, w0, rng)

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print(f"OK (parts {parts}).")
    print("     Limits declarats:")
    print("     - B5 compara la MATEIXA orbita amb normes i intervals diferents: prova el")
    print("       calcul, no que 100 s hagin convergit. Dues orbites diferents de la")
    print("       mateixa energia encara difereixen un ~10 % a 100 s.")
    print("     - El criteri de regular es una tria (LAMBDA_REGULAR, RAO_DECAIMENT_REGULAR):")
    print("       una orbita molt feblement caotica pot sortir-ne sense decidir.")
    print("     - El metode de parelles s'ha provat amb corbes FABRICADES i amb Lorenz.")
    print("       A3 fa 200 remostrejos per repeticio, no 1000, per temps de calcul.")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoprova", action="store_true", help="passa l'autoprova i surt")
    ap.add_argument("--part", default="EBPA",
                    help="quines parts: E (energia), B (Benettin), P (Poincare), "
                         "A (parelles). Per defecte, totes")
    args = ap.parse_args(argv)
    if not args.autoprova:
        ap.error("aquest modul es una biblioteca: de moment nomes te --autoprova")
    return autoprova(args.part.upper())


if __name__ == "__main__":
    raise SystemExit(main())
