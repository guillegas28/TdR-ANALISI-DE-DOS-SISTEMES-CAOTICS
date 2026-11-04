r"""On s'amolla el pendol: el zero del rellotge d'una presa de video.

EL PROBLEMA
-----------
El t = 0 d'un CSV canonic es el PRIMER FOTOGRAMA MARCAT, no l'instant en que
s'amolla el pendol. Entre l'un i l'altre hi ha de 0,05 a 2,0 segons segons la
presa. Una simulacio, en canvi, comenca a l'amollada per forca. Per posar-les
una al costat de l'altra cal saber quant val aquesta diferencia.

Quant importa: a 240 fps un fotograma son 4,17 ms, i a 10 rad/s dos
fotogrames son 4,6 graus d'angle -quinze vegades el soroll de seguiment-.
Errar l'amollada es disfressa de divergencia, i llavors l'horitzo de
predictibilitat que en surti sera mes curt del que toca sense que res no ho
canti.

EL PARANY, QUE ES EL QUE FIXA EL METODE
---------------------------------------
Els angles canonics ja estan FILTRATS: pendol/tracker.py els passa per
Savitzky-Golay amb una finestra de 13 mostres = 54 ms a 240 fps. L'amollada es
un colze -l'acceleracio salta de zero a el que sigui- i un filtre d'aquesta
mida l'arrodoneix cap als DOS costats. Sobre la serie filtrada, el moviment
sembla comencar uns 27 ms abans del que toca: sis fotogrames i mig, molt mes
que la tolerancia d'un fotograma que ens hem posat.

Per aixo aqui NO es mira on comenca el moviment. Es mira el moviment ja fet,
lluny del colze, i s'extrapola cap enrere:

  1. Localitzacio gruixuda, per velocitat. Nomes cal que encerti a uns quants
     fotogrames: no en depen la precisio, nomes on es posa la finestra.
  2. Ajust del model d'una amollada -vegeu _ajusta_amollada- sobre una
     finestra que comenca MITJA FINESTRA DE FILTRE despres de l'amollada, ja
     dins de la zona llisa. Alli Savitzky-Golay es de fase zero i no desplaca
     res, perque el senyal hi es suau. L'instant en que la velocitat del model
     val zero ES l'amollada.
  3. Dues passades. La finestra ha de comencar a partir de l'amollada, que es
     el que es busca; la primera passada surt del punt gruixut i la segona ja
     surt del resultat de la primera. Sense aixo la finestra arrencava massa
     lluny i el detector errava 16 ms sempre cap al mateix costat.
  4. Es fa amb theta1 i amb theta2 per SEPARAT. Son dues mesures independents
     del mateix instant, i que coincideixin no es cap luxe: es l'unica manera
     de saber que no ens estem enganyant tots dos alhora.

FINS ON ARRIBA AIXO, I PER QUE NO MES ENLLA
-------------------------------------------
Mesurat sobre 24 tirs fabricats -tres retards per a cadascun dels quatre
angles de sortida i dos nivells de soroll-, passats pel circuit sencer:

    tirs amb 0,30 graus de soroll (com les preses de 45 i 90):  4,7 ms
    tirs amb 1,21 graus de soroll (com les de 150):            10,6 ms
    pitjor cas dels 24:                                        26,2 ms

El pla del treball demanava UN FOTOGRAMA (4,17 ms) i s'ha canviat a TRES
despres de mesurar. Queda escrit perque canviar un criteri despres de veure
el resultat es justament el que no s'ha de fer sense dir-ho.

El motiu del canvi no es que l'ajust sigui millorable. S'han provat les
alternatives i cap no hi arriba:

  - Ajustar l'ANGLE en comptes de la velocitat, que te trenta vegades mes
    senyal per soroll: no millora.
  - Fer servir els angles SENSE FILTRAR, com els tindria tracker.py durant la
    conversio: tampoc. La mediana amb prou feines es mou i el pitjor cas
    empitjora.

El que passa es que el filtre ja ha promitjat, i dins d'una finestra d'ajust
de la seva mida les mostres canoniques son gairebe la MATEIXA dada. Per aixo
la incertesa que es publica es la de l'ajust multiplicada per un factor
calibrat (FACTOR_CORRELACIO): una covariancia que suposa independencia, aqui,
menteix a la baixa.

El que SI que es demana, i es el criteri que de debo mana, es que cada presa
SAPIGA QUANT HA ENCERTAT: als 24 tirs l'error cau dins de dos sigmes i mig de
la incertesa que la mateixa presa publica. Una amollada amb una barra d'error
honesta val; una de mes precisa pero que no sap el seu error, no.

LIMIT DECLARAT el 24/09/2026: amb el soroll de les preses de 150 graus (1,2
graus) aixo no es compleix sempre. Provant deu llavors en comptes d'una,
l'instant erra de 30 a 100 ms, alguna vegada fora de tres sigmes, i alguna
presa del repos surt «sense amollada». Els 24 tirs d'aqui passen perque en
proven una llavor per retard. Els ANGLES d'on s'amolla no en queden tocats.
Vegeu la seccio «repos curt» de l'autoprova, que te la taula.

I aquesta precisio ja esta be per al que aquest modul ha de fer: donar el zero
MODEL-INDEPENDENT d'una presa, que es el que va a les metadades. Qui ha de
donar la mitja mil.lesima quan calgui comparar dues corbes concretes es
pendol/comparar.py amb troba_desfasament, que les encaixa senceres i recupera
un retard conegut amb 0,5 ms d'error.

PER QUE THETA2 NO SAP L'HORA
----------------------------
Aixo va sortir de l'autoprova i val la pena escriure-ho, perque es fisica del
muntatge i no una pega del codi.

Al banc el pendol s'amolla amb els dos bracos RECTES, th1 = th2. En aquesta
posicio, amb els numeros d'avui i sortint de 120 graus, les dues acceleracions
inicials son alfa1 = -73,6 rad/s^2 i alfa2 = +3,3: el segon brac accelera
VINT-I-DUES VEGADES MENYS que el primer, i cap a l'altre costat. La rao es que
amb els bracos alineats el parell de la gravetat sobre el segon brac gairebe
queda compensat per l'arrossegament del primer.

Consequencia practica: dins de la finestra d'ajust, theta1 es mou uns quants
graus i theta2 nomes unes decimes -de l'ordre del soroll de seguiment-. El
vertex que en surt es soroll amplificat, i la covariancia de l'ajust NO ho
canta, perque l'error no es estadistic sino de truncament.

Per aixo aqui no es fa la mitjana dels dos a cegues: nomes entren els angles
que s'han mogut clarament per sobre del soroll (RAO_SENYAL_MINIMA), i l'altre
es calcula igualment i es publica com a DIAGNOSTIC. Es una decisio, no una
comoditat: l'hora la dona el brac que es mou.

QUE NO FA
---------
No toca els CSV canonics. Moure'n el t canviaria l'energia inicial del balanc
d'energia, que es un numero ja publicat. L'amollada es una metadada, no una
correccio de les dades.

I no se la inventa: una presa que arrenca amb el pendol ja en marxa no te
amollada dins del registre, i el que en surt es un «no hi es» amb el motiu
escrit. (Les tres que l'index donava com a tals -45_13, 90_1 i 90_5- van
resultar ser del repos: el detector de pixels de la conversio s'havia
disparat amb una sacsejada del seguidor.)

    .venv\\Scripts\\python -m pendol.amollada --autoprova
    .venv\\Scripts\\python -m pendol.amollada dades/net/150_3.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants
#
# Totes son tries. Van aqui dalt, amb el seu motiu, perque es puguin discutir
# sense haver d'entrar al codi.

MENA_PER_DEFECTE = "omega"  # s'ajusta la VELOCITAT. Provat tambe amb l'angle,
                            # que te trenta vegades mes senyal per soroll, i
                            # NO hi guanya res: el limit no es el senyal sino
                            # el filtre. Vegeu «fins on arriba aixo»
GRAU_PER_DEFECTE = 3        # l'ordre del model, comptat sobre la velocitat:
                            # omega = a*tau + c*tau^3
DURADA_AJUST_S = 0.12       # llargada de la finestra d'ajust. Aguanta perque
                            # el model porta el terme cubic: amb una recta,
                            # una finestra d'aquesta llargada es desfa
FACTOR_CORRELACIO = 2.5     # les mostres canoniques NO son independents -el
                            # filtre ja les ha promitjat- i la covariancia
                            # d'un ajust suposa que si. La incertesa que es
                            # publica es la de l'ajust multiplicada per aixo.
                            # El valor esta CALIBRAT sobre els 24 tirs de
                            # l'autoprova, que a cada passada torna a
                            # comprovar que la incertesa publicada cobreix
                            # l'error de debo sense ser-ne un coixi
MARGE_DEL_FILTRE = 0.5      # la finestra d'ajust comenca aquesta fraccio de
                            # finestra de filtre despres de l'amollada, mes un
                            # fotograma. Mitja finestra es el minim que fa que
                            # el filtre ja no vegi el colze; mes no fora mes
                            # segur, seria pitjor, perque allunya la finestra
                            # del punt que s'extrapola
FINESTRA_FILTRE_S = 0.05    # el que fa servir tracker.processa si no es diu res
LLINDAR_SIGMES = 6.0        # quantes vegades el soroll de seguiment s'ha
                            # d'apartar l'angle perque es consideri que el
                            # pendol s'ha mogut. Sis es molt amunt del soroll
                            # i molt avall de qualsevol amollada: a 45 graus
                            # s'hi arriba en 30 ms i a 150, abans
MOSTRES_SEGUIDES = 5        # i ha d'estar-hi per damunt aquestes mostres
                            # SEGUIDES. Un sol fotograma per sobre del llindar
                            # es soroll: a la vora del registre el filtre fa
                            # puntes a la derivada que arriben al llindar sense
                            # que el pendol s'hagi mogut gens.
LLINDAR_FLUIX_SIGMES = 2.0  # per sota d'aixo, una mostra encara es quieta. Es el
                            # llindar amb que la localitzacio gruixuda torna
                            # enrere i el que tria els angles quan el repos es curt
MOSTRES_REPOS_MINIM = 5     # menys que aixo no es cap mitjana de repos
SOROLL_PER_DEFECTE = 0.30   # graus; el mesurat als videos si no se'n sap el seu
RAO_SENYAL_MINIMA = 3.0     # quantes vegades el soroll s'ha de moure un angle,
RAO_RESPECTE_AL_MILLOR = 1 / 3  # i quina part del que s'ha mogut l'altre,
                            # perque el seu ajust compti. La segona condicio
                            # es la que de debo separa: entre els dos angles
                            # hi ha un factor 10 o mes -theta1 es mou de 30 a
                            # 60 vegades el soroll i theta2 d'1 a 4- i un
                            # llindar relatiu ho encerta tant si la presa es
                            # neta com si es sorollosa. Un llindar fix no:
                            # posat prou alt per esquivar theta2 a les preses
                            # netes, rebutjava theta1 a les de 150 graus.
                            # Vegeu «per que theta2 no sap l'hora».


def finestra_del_filtre(meta: dict | None, fps: float) -> float:
    """Quants segons dura la finestra de Savitzky-Golay d'aquesta presa.

    Surt de les seves propies metadades quan hi son, que es l'unic valor que
    val: la finestra es calcula amb els fps de la presa i no totes van als
    mateixos fps.
    """
    sav = (meta or {}).get("savgol") or {}
    mostres = sav.get("finestra_mostres")
    if mostres:
        return float(mostres) / float(fps)
    return FINESTRA_FILTRE_S


def soroll_del_seguiment(meta: dict | None) -> float:
    """El soroll angular d'aquesta presa, en graus. Serveix per jutjar l'ajust."""
    rig = ((meta or {}).get("geometria") or {}).get("rigidesa") or {}
    valors = rig.get("soroll_angular_graus")
    if valors:
        return float(max(valors))
    return SOROLL_PER_DEFECTE


# ------------------------------------------------------------------ el nucli


def _ajusta_amollada(t, senyal, ordre: int, mena: str) -> dict | None:
    r"""Ajusta el model d'una amollada i en torna l'instant en que va passar.

    Es pot ajustar sobre la VELOCITAT (mena="omega") o sobre l'ANGLE
    (mena="theta"). Son el mateix model integrat una vegada:

        omega(tau) = a*tau + c*tau^3 + ...
        theta(tau) = th0 + (a/2)*tau^2 + (c/4)*tau^4 + ...

    amb tau = t - t_amollada. `ordre` es la potencia mes alta de la velocitat:
    ordre 1 vol dir omega recta i theta parabolica.

    PER QUE NOMES POTENCIES SENARS A LA VELOCITAT. Es el que fa que aixo
    funcioni, i surt de la fisica, no de l'estadistica. Al voltant de
    l'amollada NO hi ha terme en tau^2 a la velocitat: l'acceleracio angular
    depen dels angles i de les velocitats AL QUADRAT -els termes centrifugs-,
    o sigui que la seva derivada temporal val zero alla on omega val zero. La
    primera correccio a «omega creix recta» es cubica.

    Aixo explica la taula de l'autoprova, que si no es un misteri: ajustar un
    polinomi LLIURE de grau 2 o 3 sortia pitjor que una recta, perque li
    donava al polinomi un terme que fisicament no hi es i que es dedicava a
    seguir el soroll.

    PER QUE VAL LA PENA AJUSTAR L'ANGLE. La velocitat del contracte de dades
    no es una mesura independent: surt de derivar l'angle amb Savitzky-Golay
    sobre una finestra de 54 ms. Dins d'una finestra d'ajust de 50 ms, les
    mostres de omega son gairebe la MATEIXA dada repetida dotze vegades, i per
    aixo l'ajust sobre omega es queda a uns 5 ms per molts punts que hi posis.
    L'angle porta el soroll sense amplificar i hi guanya un ordre de magnitud.

    Es un ajust NO lineal -l'instant hi entra dins d'un parentesi- i per aixo
    va amb least_squares. Surt l'instant, la seva incertesa des de la
    covariancia de l'ajust, i el residu. None si no convergeix.
    """
    from scipy.optimize import least_squares

    t = np.asarray(t, dtype=float)
    s = np.asarray(senyal, dtype=float)
    if mena == "omega":
        potencies = np.arange(1, ordre + 1, 2)
        divisors = np.ones(len(potencies))
    elif mena == "theta":
        potencies = np.arange(2, ordre + 2, 2)
        divisors = potencies.astype(float)
    else:
        raise ValueError(f"mena desconeguda: {mena}")
    amb_constant = mena == "theta"
    n_par = len(potencies) + int(amb_constant) + 1
    if len(t) < n_par + 2:
        return None

    # Es centra el temps: amb t de l'ordre de desenes de segons i potencies
    # altes, els numeros se'n van i l'ajust es fa mal condicionat.
    t0 = float(t.mean())
    tau = t - t0

    # Llavor. Amb un polinomi lliure del grau que toca ja s'hi arriba a prop.
    x0 = np.zeros(n_par)
    if mena == "omega":
        recta = np.polyfit(tau, s, 1)
        if recta[0] == 0.0:
            return None
        x0[0], x0[-1] = recta[0], -recta[1] / recta[0]
    else:
        q = np.polyfit(tau, s, 2)
        if q[0] == 0.0:
            return None
        ta0 = -q[1] / (2 * q[0])
        x0[0] = 2 * q[0]
        x0[-2] = float(np.polyval(q, ta0))  # l'angle al vertex
        x0[-1] = ta0

    def residus(x):
        d = tau - x[-1]
        model = np.full_like(tau, x[-2] if amb_constant else 0.0)
        for k, (pot, div) in enumerate(zip(potencies, divisors)):
            model = model + (x[k] / div) * d**pot
        return model - s

    res = least_squares(residus, x0)
    if not res.success:
        return None

    # Covariancia des de la jacobiana, escalada pel residu: es el mateix que
    # fa polyfit(cov=True) i te la mateixa lectura.
    graus_de_llibertat = len(t) - n_par
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
    except np.linalg.LinAlgError:
        return None
    if graus_de_llibertat > 0:
        cov = cov * (2 * res.cost / graus_de_llibertat)

    sortida = {
        "instant_s": t0 + float(res.x[-1]),
        "incertesa_s": float(np.sqrt(max(float(cov[-1, -1]), 0.0))),
        "residu": float(np.sqrt(np.mean(res.fun**2))),
        "mena": mena,
    }
    if amb_constant:
        sortida["angle_rad"] = float(res.x[-2])
    return sortida


def _angle_a(t, th, instant: float, grau: int) -> float:
    """L'angle que tenia la barra a `instant`, extrapolat des de la finestra.

    Aqui si que s'ajusta l'angle, pero nomes per LLEGIR-NE UN VALOR, no per
    trobar-ne el vertex. Extrapolar el valor d'una funcio suau uns quants
    fotogrames enrere es molt menys delicat que extrapolar-ne la posicio d'un
    punt estacionari, que es el que feia mal.
    """
    t = np.asarray(t, dtype=float)
    t0 = float(t.mean())
    coef = np.polyfit(t - t0, np.asarray(th, dtype=float), grau)
    return float(np.polyval(coef, instant - t0))


def _base_del_repos(th1, th2, vora: int) -> tuple[float, float]:
    """Els angles de quietud de referencia: les primeres mostres bones.

    Les de la vora es descarten: alla el filtre ajusta un polinomi a una
    finestra que no te dades a l'esquerra i el que en surt no es de fiar.
    """
    return (float(np.median(np.asarray(th1, dtype=float)[vora:vora + 5])),
            float(np.median(np.asarray(th2, dtype=float)[vora:vora + 5])))


def _localitzacio_gruixuda(th1, th2, soroll_rad: float, vora: int = 0) -> int | None:
    """El primer fotograma amb moviment, a uns quants fotogrames de precisio.

    ES MIRA L'ANGLE, NO LA VELOCITAT, i aixo es va haver de canviar. Amb la
    velocitat el detector disparava amb el pendol completament quiet: la
    derivada de Savitzky-Golay amplifica el soroll de seguiment per un factor
    que depen de l'ordre del polinomi, i posar-hi un llindar volia dir haver
    d'encertar aquell factor. Amb l'angle no cal: el soroll de l'angle es
    justament el que cada presa declara a les seves metadades, i el llindar
    surt d'un numero mesurat en comptes d'un de calculat.

    Es compara amb l'angle de les primeres mostres bones, i el llindar s'ha de
    passar durant MOSTRES_SEGUIDES fotogrames SEGUITS. Les mostres de la vora
    es descarten: alla el filtre ajusta un polinomi a una finestra que no te
    dades a l'esquerra i el que en surt no es de fiar.
    """
    th1 = np.asarray(th1, dtype=float)
    th2 = np.asarray(th2, dtype=float)
    if len(th1) < vora + MOSTRES_SEGUIDES + 5:
        return None
    base1, base2 = _base_del_repos(th1, th2, vora)
    d = np.maximum(np.abs(th1 - base1), np.abs(th2 - base2))
    dalt = d > LLINDAR_SIGMES * soroll_rad
    dalt[:vora] = False
    if len(dalt) < MOSTRES_SEGUIDES:
        return None
    # Quantes mostres seguides per damunt del llindar comencen a cada index.
    seguides = np.ones(len(dalt) - MOSTRES_SEGUIDES + 1, dtype=bool)
    for k in range(MOSTRES_SEGUIDES):
        seguides &= dalt[k:len(dalt) - MOSTRES_SEGUIDES + 1 + k]
    trobat = np.flatnonzero(seguides)
    if not trobat.size:
        return None

    # I ARA CAP ENRERE. El punt que s'acaba de trobar diu «aqui ja es mou
    # segur», i esta tant mes tard com mes sorollosa sigui la presa: a un tir
    # de 45 graus amb 1,2 graus de soroll, sis sigmes no s'assoleixen fins 65
    # ms despres d'amollar. Si la finestra d'ajust es colloca a partir d'alli,
    # arriba fins a 216 ms de l'amollada -mes del temps natural del muntatge-
    # i el model ja no hi val.
    #
    # Es torna enrere fins a la darrera mostra que encara era quieta, amb un
    # llindar molt mes baix. Anar cap enrere des d'un punt on el moviment es
    # segur no te el problema d'anar cap endavant des del soroll: no hi ha cap
    # fluctuacio que et pugui fer parar abans d'hora, perque el senyal ja
    # creix, i el que es busca es justament on deixa de creixer.
    j = int(trobat[0])
    fluix = LLINDAR_FLUIX_SIGMES * soroll_rad
    while j > vora and d[j - 1] > fluix:
        j -= 1
    return j


def troba(
    t,
    th1,
    th2,
    w1,
    w2,
    *,
    finestra_filtre_s: float | None = None,
    durada_ajust_s: float = DURADA_AJUST_S,
    grau: int = GRAU_PER_DEFECTE,
    mena: str = MENA_PER_DEFECTE,
    soroll_graus: float = SOROLL_PER_DEFECTE,
) -> dict:
    """L'instant d'amollada d'una presa, i els angles d'on es va amollar.

    Entren les cinc columnes del contracte de dades. Surt un diccionari que
    es pot desar tal qual a unes metadades. Mai aixeca: si no hi ha amollada
    dins del registre, ho diu a «hi_ha_amollada» amb el motiu.
    """
    t = np.asarray(t, dtype=float)
    th1 = np.asarray(th1, dtype=float)
    th2 = np.asarray(th2, dtype=float)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 else float("nan")
    if finestra_filtre_s is None:
        finestra_filtre_s = FINESTRA_FILTRE_S

    def cap(motiu: str) -> dict:
        return {"hi_ha_amollada": False, "motiu": motiu, "fps": float(fps)}

    vora = int(round(finestra_filtre_s * fps / 2))
    i = _localitzacio_gruixuda(th1, th2, np.radians(soroll_graus), vora)
    if i is None:
        return cap("no s'hi veu moviment enlloc, o la presa es massa curta")

    # Si el moviment ja hi es a la primera mostra que el filtre no ha tocat,
    # l'amollada cau abans del registre o hi cau tan a sobre que no se'n pot
    # treure res: el colze queda dins de la vora dolenta.
    if i <= vora:
        return cap(
            f"el marcatge arrenca amb el pendol ja en marxa: el moviment ja hi es al "
            f"primer fotograma que el filtre no toca ({1000*t[vora]:.0f} ms)"
        )

    # DUES PASSADES, i el motiu es una pescadilla que es mossega la cua.
    #
    # La finestra d'ajust ha de comencar prou lluny de l'amollada perque el
    # filtre no hi arribi -mitja finestra de filtre-, pero NO MES LLUNY DEL
    # QUE CALGUI: com mes lluny comenca, mes ha d'extrapolar el polinomi cap
    # enrere, i un polinomi nomes val mentre el temps sigui petit al costat
    # del temps natural del muntatge.
    #
    # El problema es que per saber on ha de comencar la finestra cal saber
    # l'amollada, que es justament el que busquem. El punt gruixut no serveix:
    # pot caure 25 ms a banda i banda segons com de rapid sigui el tir. Es
    # resol iterant. La primera passada surt del punt gruixut i ja erra de
    # pocs mil.lisegons; la segona surt d'aquell resultat i ja no es mou.
    #
    # Sense aixo la finestra arrencava a 70 ms de l'amollada en comptes de 31
    # i el detector errava de 16 ms -quatre fotogrames-, sempre cap al mateix
    # costat. Era biaix, no soroll.
    marge = MARGE_DEL_FILTRE * finestra_filtre_s + 1.0 / fps

    def una_passada(zero: float):
        """Els dos ajustos amb la finestra collocada a partir de `zero`."""
        inici = zero + marge
        dins = (t >= inici) & (t <= inici + durada_ajust_s)
        # La finestra llarga serveix per mesurar l'error de TRUNCAMENT: el
        # polinomi es una aproximacio i, com que l'arrel s'extrapola cap
        # enrere, quedar-se curt es paga alli. Quant es mou l'instant en
        # allargar la finestra es la millor estimacio d'aquell error, i la
        # covariancia de l'ajust -que nomes veu el soroll- no ho pot dir.
        llarga = (t >= inici) & (t <= inici + 1.5 * durada_ajust_s)
        if dins.sum() < grau + 4:
            return None, f"nomes {int(dins.sum())} mostres a la finestra d'ajust"

        ajustos = {}
        for nom, th, w in (("theta1", th1, w1), ("theta2", th2, w2)):
            senyal = th if mena == "theta" else w
            # Si l'ajust demanat no convergeix, es baixa d'ordre abans de
            # rendir-se: amb poques mostres i poc senyal, un terme de mes es
            # pot quedar sense solucio.
            r, g = None, grau
            while r is None and g >= 1:
                r = _ajusta_amollada(t[dins], senyal[dins], g, mena)
                # LIMIT DE VALIDESA. L'amollada no pot caure lluny del punt
                # gruixut: el llindar de velocitat es passa pocs mil.lisegons
                # despres d'amollar, i el filtre nomes pot escampar el colze
                # mitja finestra. Si l'ajust diu una cosa de fora d'aqui, no
                # es que hagi trobat res: es que ha seguit el soroll. Passava
                # a la 150_8, que amb la primera versio donava l'amollada a
                # 1,78 s quan el moviment ja s'hi veia a 0,59.
                if r is not None and not (
                        t[i] - 3 * finestra_filtre_s <= r["instant_s"]
                        <= t[i] + finestra_filtre_s / 2):
                    r = None
                if r is None:
                    g -= 2  # el model nomes te potencies alternades
            if r is None:
                # Que un angle no doni res NO tomba la presa. Normalment el
                # que falla es theta2, que amb els bracos rectes gairebe no
                # accelera i per tant no sap l'hora -vegeu el capdamunt del
                # modul-, i seria absurd renunciar a una presa per culpa de
                # l'angle que de totes maneres no havia de comptar. Si el que
                # falla es l'altre, la presa cau tota sola mes avall, quan es
                # miri quins angles queden.
                ajustos[nom] = {
                    "compta": False,
                    "motiu": "l'ajust no dona cap instant creible: o no convergeix o "
                             f"cau fora de la franja on pot ser l'amollada, que es de "
                             f"{t[i] - 3*finestra_filtre_s:.3f} a "
                             f"{t[i] + finestra_filtre_s/2:.3f} s",
                    "rao_senyal_soroll": float(np.degrees(np.ptp(th[dins])) / soroll_graus)
                    if soroll_graus > 0 else np.inf,
                    "recorregut_graus": float(np.degrees(np.ptp(th[dins]))),
                }
                continue
            r["grau_usat"] = int(g)
            # Les mostres canoniques no son independents: el filtre ja les ha
            # promitjat, i dins d'una finestra d'ajust de la mida de la del
            # filtre son gairebe la mateixa dada. La covariancia de l'ajust
            # suposa independencia i per tant menteix a la baixa.
            r["incertesa_ajust_s"] = r["incertesa_s"]
            r["incertesa_s"] = r["incertesa_s"] * FACTOR_CORRELACIO
            r2 = _ajusta_amollada(t[llarga], senyal[llarga], g, mena)
            # Acotada a una finestra de filtre. Quan l'ajust de la finestra
            # llarga se'n va, la diferencia pot sortir enorme i llavors el que
            # es publica no es una incertesa sino un numero sense sentit. Si
            # el metode no sap on es l'amollada dins d'una finestra de filtre,
            # ja no ho sap, i dir-ho amb mes decimals no hi afegeix res.
            brut = abs(r["instant_s"] - r2["instant_s"]) if r2 else finestra_filtre_s / 2
            r["sistematica_s"] = float(min(brut, finestra_filtre_s))
            r["incertesa_total_s"] = float(np.hypot(r["incertesa_s"], r["sistematica_s"]))
            if "angle_rad" not in r:
                # GRAU 2 i no mes, encara que el model de la velocitat sigui
                # de grau 3. Llegir un valor uns fotogrames enrere amb una
                # parabola es segur; amb un polinomi de grau 4 no: a la presa
                # 90_2, que gairebe no te repos i per tant depen d'aquest
                # valor, el grau 4 donava un angle d'amollada de 623 graus.
                r["angle_rad"] = _angle_a(t[dins], th[dins], r["instant_s"], 2)

            # Quant s'ha mogut aquest angle dins de la finestra, comparat amb
            # el soroll de seguiment. Vegeu «per que theta2 no sap l'hora».
            recorregut = float(np.degrees(np.ptp(th[dins])))
            r["recorregut_graus"] = recorregut
            r["rao_senyal_soroll"] = recorregut / soroll_graus if soroll_graus > 0 else np.inf
            r["finestra_s"] = [float(inici), float(inici + durada_ajust_s)]
            ajustos[nom] = r

        # Quins angles compten. Es decideix aqui, amb els dos ja calculats,
        # perque el criteri es RELATIU: l'angle que es mou deu vegades menys
        # que l'altre no sap l'hora, ho faci en un video net o en un de brut.
        millor = max(a["rao_senyal_soroll"] for a in ajustos.values())
        llindar_senyal = max(RAO_SENYAL_MINIMA, RAO_RESPECTE_AL_MILLOR * millor)
        for nom, a in ajustos.items():
            if "instant_s" not in a:
                continue  # l'ajust ja ha fallat; queda marcat com a no comptador
            a["compta"] = bool(a["rao_senyal_soroll"] >= llindar_senyal)
            a["llindar_senyal"] = float(llindar_senyal)
        return ajustos, None

    def pesa(ajustos):
        """Mitjana pesada dels angles que compten. (zero, incertesa, utils, pes)."""
        # Per la inversa de la variancia TOTAL, i nomes dels angles que
        # compten: les dues mesures son independents pero no tenen per que ser
        # igual de bones, i n'hi ha una que sovint no val res.
        utils = [n for n in ("theta1", "theta2") if ajustos[n]["compta"]]
        if not utils:
            return None
        instants = np.array([ajustos[n]["instant_s"] for n in utils])
        sigmes = np.array([max(ajustos[n]["incertesa_total_s"], 1e-12) for n in utils])
        pes = 1.0 / sigmes**2
        return (float((instants * pes).sum() / pes.sum()),
                float(np.sqrt(1.0 / pes.sum())), utils)

    # La segona passada nomes es queda si SURT BE. Si el refinament no
    # convergeix o deixa la finestra en un lloc on els angles ja no es mouen
    # prou -cosa que passa quan la primera estimacio ha sortit massa d'hora i
    # la finestra llisca cap endavant del compte-, es torna a la primera, que
    # es mes gruixuda pero sempre esta collocada.
    ajustos, error = una_passada(t[i])
    if error:
        return cap(error)
    bo = pesa(ajustos)
    if bo is None:
        return cap(
            "cap dels dos angles no dona una amollada bona. "
            + "; ".join(
                f"{nom}: " + (a["motiu"] if "motiu" in a else
                              f"nomes es mou {a['rao_senyal_soroll']:.1f} vegades el "
                              f"soroll i en calen {a.get('llindar_senyal', RAO_SENYAL_MINIMA):.1f}")
                for nom, a in ajustos.items())
        )
    segons, _ = una_passada(bo[0])
    if segons is not None:
        refinat = pesa(segons)
        if refinat is not None:
            ajustos, bo = segons, refinat
    zero, incertesa, utils = bo

    amollada = zero
    instants_dels_dos = [ajustos[n].get("instant_s") for n in ("theta1", "theta2")]
    discrepancia = (float(abs(instants_dels_dos[0] - instants_dels_dos[1]))
                    if None not in instants_dels_dos else float("nan"))
    inici, fi = ajustos[utils[0]]["finestra_s"]

    # Els angles d'on es va amollar. La mitjana dels fotogrames de quietud es
    # millor que el vertex -promitja soroll- pero nomes val fins alla on el
    # filtre encara no ha escampat el colze cap enrere.
    fins = amollada - finestra_filtre_s / 2
    repos = t < fins
    vertex = np.array([ajustos[n].get("angle_rad", np.nan) for n in ("theta1", "theta2")])
    avis_angles = None
    if repos.sum() >= MOSTRES_REPOS_MINIM:
        angles = np.array([float(np.median(th1[repos])), float(np.median(th2[repos]))])
        origen = "mitjana del repos"
    else:
        # REPOS CURT. Abans d'aqui s'agafava el vertex, que es la parabola de
        # la finestra d'ajust extrapolada cap enrere. La finestra comenca uns
        # 30 ms despres d'amollar i en dura 120, i en un tir rapid aquell tros
        # no es cap parabola: sobre tirs fabricats errava d'1 a 2 graus, i a
        # la 90_2 de debo, 2,6. I si l'ajust d'un angle no convergia, aquell
        # angle sortia nan.
        #
        # Ara es fa servir el poc repos que hi ha: les mostres d'abans de
        # l'amollada que encara son quietes, amb la mateixa base i el mateix
        # llindar fluix amb que la localitzacio gruixuda torna enrere. El preu
        # es que aquestes mostres ja cauen on el filtre escampa una mica el
        # moviment, pero el biaix no pot passar del llindar que les tria.
        base1, base2 = _base_del_repos(th1, th2, vora)
        fluix = LLINDAR_FLUIX_SIGMES * np.radians(soroll_graus)
        quietes = ((t < amollada) & (np.abs(th1 - base1) <= fluix)
                   & (np.abs(th2 - base2) <= fluix))
        if quietes.any():
            angles = np.array([float(np.median(th1[quietes])), float(np.median(th2[quietes]))])
            origen = "repos curt: mostres quietes d'abans de l'amollada"
            avis_angles = (
                f"repos curt: els angles surten de {int(quietes.sum())} mostres de just abans "
                f"de l'amollada, on el filtre ja escampa el moviment; poden portar un "
                f"biaix de fins a {LLINDAR_FLUIX_SIGMES:.0f} vegades el soroll "
                f"({LLINDAR_FLUIX_SIGMES*soroll_graus:.2f} graus)"
            )
        else:
            angles = vertex
            origen = "extrapolacio de l'ajust (cap mostra quieta abans de l'amollada)"
            avis_angles = (
                "els angles surten d'extrapolar l'ajust cap enrere, perque no hi ha cap "
                "mostra quieta abans de l'amollada: poden errar de graus"
            )

    # Quant repos queda abans de l'amollada. Es mira DESPRES d'haver-la
    # trobada, no abans, perque abans no se sap on cau. Per sota de mitja
    # finestra de filtre no hi ha res a fer: el colze queda dins de la vora on
    # el filtre extrapola i el que se'n tregui sera invencio.
    repos_s = amollada - t[0]
    if repos_s < finestra_filtre_s / 2:
        return cap(
            f"nomes hi ha {1000*repos_s:.0f} ms de registre abans de l'amollada i el "
            f"filtre ja n'ocupa {1000*finestra_filtre_s/2:.0f}: aquesta presa arrenca "
            "amb el pendol practicament ja engegat"
        )

    avisos = [avis_angles] if avis_angles else []
    if repos_s < finestra_filtre_s:
        avisos.append(
            f"nomes {1000*repos_s:.0f} ms de repos abans de l'amollada, menys d'una "
            f"finestra de filtre ({1000*finestra_filtre_s:.0f} ms): l'amollada hi es "
            "pero surt del tros mes just de la presa"
        )
    if len(utils) == 2 and discrepancia > 1.0 / fps:
        avisos.append(
            f"theta1 i theta2 no diuen el mateix instant: {1000*discrepancia:.1f} ms "
            f"de diferencia, mes d'un fotograma ({1000/fps:.1f} ms)"
        )
    # El residu de l'ajust s'ha de poder comparar amb el soroll de seguiment,
    # que es en graus. Si s'ha ajustat l'angle ja hi son; si s'ha ajustat la
    # velocitat, es passa a graus multiplicant per la finestra del filtre, que
    # es l'escala de temps amb que aquella velocitat s'ha derivat.
    escala = 1.0 if mena == "theta" else finestra_filtre_s
    pitjor_residu = np.degrees(max(ajustos[n]["residu"] for n in utils) * escala)
    if pitjor_residu > 3 * soroll_graus:
        avisos.append(
            f"el residu de l'ajust equival a {pitjor_residu:.2f} graus, mes de tres "
            f"vegades el soroll de seguiment ({soroll_graus:.2f}): o la finestra es "
            f"massa llarga per al model, o l'amollada no va ser neta"
        )

    return {
        "hi_ha_amollada": True,
        "motiu": None,
        "amollada_s": amollada,
        "incertesa_s": incertesa,
        "incertesa_fotogrames": float(incertesa * fps),
        "angles_que_compten": utils,
        "repos_abans_s": float(repos_s),
        "discrepancia_entre_angles_s": discrepancia,
        "angles_amollada_graus": [float(np.degrees(a)) for a in angles],
        "origen_dels_angles": origen,
        "diferencia_repos_vertex_graus": [float(np.degrees(a - v)) for a, v in zip(angles, vertex)],
        "per_angle": ajustos,
        "finestra_ajust_s": [float(inici), float(fi)],
        "grau_del_polinomi": int(grau),
        "finestra_filtre_s": float(finestra_filtre_s),
        "fps": float(fps),
        "avisos": avisos,
    }


def troba_de_taula(taula, meta: dict | None = None, **extra) -> dict:
    """troba() agafant de les metadades el que la presa sap d'ella mateixa."""
    t = np.asarray(taula["t"], dtype=float)
    fps = float((meta or {}).get("fps") or ((len(t) - 1) / (t[-1] - t[0])))
    extra.setdefault("finestra_filtre_s", finestra_del_filtre(meta, fps))
    extra.setdefault("soroll_graus", soroll_del_seguiment(meta))
    return troba(t, taula["theta1"], taula["theta2"], taula["omega1"], taula["omega2"], **extra)


def arrenca_del_repos(meta: dict | None) -> bool | None:
    """Si una presa arrenca del repos, segons l'amollada desada a les metadades.

    Es EL criteri del treball, i viu aqui i enlloc mes: el fan servir la
    conversio (per a l'index) i el balanc d'energia (per decidir qui hi entra).

    Arrenca del repos si s'hi troba l'amollada, i prou. No hi ha cap llindar
    de temps a part: troba() ja demana mitja finestra de filtre de quietud
    abans d'amollar, que es el minim per saber que el registre comenca quiet.
    El llindar de 50 ms que hi havia abans era del detector de pixels de
    pendol/tracker.py, que es disparava amb una sola sacsejada del seguidor, i
    amb aquest criteri la 90_2 -29 ms de repos- quedaria fora sense motiu.

    Torna None si les metadades no porten el bloc «amollada» (una presa
    convertida abans del 24/09/2026, o una de prova): llavors no se sap, i qui
    crida ha de decidir que fa.
    """
    bloc = (meta or {}).get("amollada")
    if not isinstance(bloc, dict) or "hi_ha_amollada" not in bloc:
        return None
    return bool(bloc["hi_ha_amollada"])


def estat_inicial(resultat: dict) -> np.ndarray:
    """Les condicions inicials mesurades: (theta1, theta2, 0, 0), en radians.

    Les velocitats son ZERO EXACTE i no es deriven de res. El pendol s'amolla
    del repos, o sigui que omega = 0 no es una aproximacio: es el que passa.
    Derivar-les del video hi posaria soroll de seguiment amplificat i prou.
    """
    if not resultat.get("hi_ha_amollada"):
        raise ValueError(f"aquesta presa no te amollada: {resultat.get('motiu')}")
    th1, th2 = np.radians(resultat["angles_amollada_graus"])
    return np.array([th1, th2, 0.0, 0.0])


# --------------------------------------------------------------- l'autoprova


def tir_fals_amb_repos(p, estat0, quiet_s: float, temps_s: float, fps: float, *, dis=None):
    """Un tir simulat amb un tros de REPOS al davant, com a les preses de debo.

    Retorna (t, theta1, theta2) en una graella de fotogrames regular que
    comenca a zero. El pendol s'hi esta quiet fins a `quiet_s` i alla s'amolla.

    Aixo no es cap floritura per a la prova: TOTES les preses del treball
    comencen aixi, perque el marcatge del video s'engega abans que la ma
    deixi anar el pendol. Un tir fals que comenci ja en marxa no serveix per
    provar res d'aquest modul.

    L'amollada cau on es digui, i no cal que caigui damunt d'un fotograma: el
    tros que es mou es demana a simula() amb els seus instants exactes
    (t_eval), que es justament el que permet que el retard tingui una part
    de fotograma. Si sempre en fos un de sencer, un detector que nomes sabes
    comptar fotogrames passaria la prova sense saber extrapolar res.

    La fa servir eines/genera_tracker_fals.py amb --quiet, de manera que el
    tir fals de l'autoprova d'aqui i el que es genera a ma son el mateix.

    Amb `dis` (una fisica.Dissipacio) el tir fals te friccio. Serveix per
    mesurar que fa l'afinament de l'amollada, que encaixa el model SENSE
    friccio, quan el pendol de debo en te (simulador.autoprova, punt 3c).
    """
    from pendol.simulador import simula

    estat0 = np.asarray(estat0, dtype=float)
    t = np.arange(0.0, quiet_s + temps_s + 0.5 / fps, 1.0 / fps)
    mou = t >= quiet_s
    if not mou.any():
        raise ValueError("amb aquest repos no queda gens de moviment")

    _, y = simula(p, estat0, t_eval=t[mou] - quiet_s, dis=dis)
    th1 = np.full(len(t), estat0[0])
    th2 = np.full(len(t), estat0[1])
    th1[mou], th2[mou] = y[0], y[1]
    return t, th1, th2


def _escriu_cru(cami: Path, t, x1, y1, x2, y2) -> None:
    """Un export del Tracker fals, amb el format lleig de debo.

    Es escriu un export CRU i no una taula canonica a posta: aixi la prova
    passa per tot el circuit -lectura del format amb punt i coma i decimals
    amb coma, a_angles, reompliment de forats i Savitzky-Golay- i el filtre hi
    entra tal com hi entrara amb les preses reals. Una prova feta amb angles
    fabricats directament esquivaria justament el parany que aquest modul
    existeix per esquivar.
    """
    def cat(v: float) -> str:
        return f"{v:.6f}".replace(".", ",")

    linies = ["Tracker 6.2.0", "video: amollada_falsa.mp4",
              "escala: 1 m = 640 px", "", "t;x_A;y_A;x_B;y_B"]
    linies += [f"{cat(t[i])};{cat(x1[i])};{cat(y1[i])};{cat(x2[i])};{cat(y2[i])}"
               for i in range(len(t))]
    cami.parent.mkdir(parents=True, exist_ok=True)
    cami.write_text("\n".join(linies), encoding="utf-8")


def _presa_falsa(p, cami_cru: Path, quiet_s: float, temps_s: float, fps: float,
                 angle_graus: float, soroll_m: float, llavor: int, *, dis=None) -> dict:
    """Escriu el tir fals i en torna el soroll angular, com el mesura tracker.py.

    El soroll es calcula igual que a llegeix_del_trk -la variacio relativa de
    la llargada de cada barra, que hauria de ser constant- perque el numero
    que en surt sigui el mateix que porten les preses de debo a les seves
    metadades. Si no, el detector jutjaria els tirs de prova amb un soroll
    que no es el seu.
    """
    from pendol.fisica import posicions

    rng = np.random.default_rng(llavor)
    # Els dos bracos rectes i quiets, com al banc.
    estat0 = np.radians([angle_graus, angle_graus, 0.0, 0.0])
    t, th1, th2 = tir_fals_amb_repos(p, estat0, quiet_s, temps_s, fps, dis=dis)
    x1, y1, x2, y2 = posicions(th1, th2, p)
    x1, y1, x2, y2 = (v + rng.normal(0.0, soroll_m, v.shape) for v in (x1, y1, x2, y2))
    _escriu_cru(cami_cru, t, x1, y1, x2, y2)

    L1 = np.hypot(x1, y1)
    L2 = np.hypot(x2 - x1, y2 - y1)
    return {"soroll_angular_graus": [float(np.degrees(L1.std() / L1.mean())),
                                     float(np.degrees(L2.std() / L2.mean()))]}


def autoprova() -> int:
    """Ha de recuperar un retard CONEGUT, filtre inclos, dins d'un fotograma.

    Els retards de la prova NO son multiples del fotograma a posta: si ho
    fossin, un detector que nomes sabes comptar fotogrames hi passaria sense
    haver de saber res d'extrapolar.

    S'hi proven els dos graus de polinomi i dues llargades de finestra, i la
    taula que en surt es el que decideix els valors per defecte. Es al reves
    de triar-los i despres provar-los, que es com es tria malament.
    """
    import contextlib
    import io

    from pendol import dades, tracker
    from pendol.fisica import Parametres

    FPS = 240.0
    # DOS NIVELLS DE SOROLL, i el segon no es cap caprici. Amb nomes un pixel
    # -0,3 graus, que es el de les preses de 45 i 90- els valors per defecte
    # sortien immillorables i despres fallaven a set preses reals de 150, on
    # el soroll de seguiment es de 0,8 a 1,4 graus. Un tir de prova mes net
    # que les dades de debo no prova res.
    SOROLLS = {"1 px": 0.0005, "4 px": 0.0020}
    FOTOGRAMA = 1.0 / FPS
    # TRES fotogrames, i el criteri va canviar despres de mesurar, cosa que
    # s'ha de dir. El pla demanava UN fotograma. Mesurant, la precisio
    # d'aquest metode sobre dades canoniques es de 4 a 5 ms de dispersio, que
    # son mes d'un fotograma, i NO es un defecte de l'ajust: es el filtre.
    # Vegeu «fins on arriba aixo» al capdamunt del modul. Qui ha de donar la
    # mitja mil.lesima es pendol/comparar.py amb troba_desfasament, encaixant
    # les corbes senceres; aixo d'aqui es el zero model-independent.
    TOLERANCIA = 3.0 * FOTOGRAMA
    RETARDS = (0.2031, 0.5017, 1.4963)

    p = Parametres.des_de_json("parametres.json")
    desti = Path("esborranys/_autoprova_amollada")

    print("Autoprova de l'amollada")
    print(f"   tirs fabricats a {FPS:.0f} fps amb soroll d'un pixel i passats per")
    print(f"   pendol/tracker.py SENCER, Savitzky-Golay inclos")
    print(f"   tolerancia: {1000*TOLERANCIA:.2f} ms = tres fotogrames (vegeu el codi:")
    print(f"   el pla en demanava un i la mesura diu que amb el filtre pel mig no hi ha)\n")

    # Els tirs de la prova cobreixen els TRES angles de sortida del treball i
    # un de mig. No es cap luxe: la finestra d'ajust val mentre omega creixi
    # recta, i com de depressa deixa de fer-ho depen del tir. Escollint els
    # valors per defecte mirant nomes els 120 graus, els 150 sortien fora de
    # tolerancia i no se n'hauria sabut res.
    preses = []
    for quin, soroll_m in SOROLLS.items():
        for angle in (45.0, 90.0, 120.0, 150.0):
            for k, retard in enumerate(RETARDS):
                cru = desti / f"cru_{quin[0]}_{angle:.0f}_{k}.csv"
                net = desti / f"net_{quin[0]}_{angle:.0f}_{k}.csv"
                rigidesa = _presa_falsa(p, cru, retard, 4.0, FPS, angle, soroll_m,
                                        llavor=len(preses) + 100)
                with contextlib.redirect_stdout(io.StringIO()):
                    tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False,
                                     meta_extra={"angle_nominal_graus": angle,
                                                 "geometria": {"rigidesa": rigidesa}})
                taula, meta = dades.carrega(net)
                preses.append((quin, angle, retard, taula, meta, net))
    print(f"   {len(preses)} tirs: {len(RETARDS)} retards x 4 angles (45, 90, 120, 150)")
    print(f"   x {len(SOROLLS)} nivells de soroll ({', '.join(SOROLLS)})")
    mesurat = {q: max(soroll_del_seguiment(m) for qq, _, _, _, m, _ in preses if qq == q)
               for q in SOROLLS}
    print("   soroll angular que en surt: "
          + ", ".join(f"{q} -> fins a {v:.2f} graus" for q, v in mesurat.items()) + "\n")

    net = np.array([q == list(SOROLLS)[0] for q, _, _, _, _, _ in preses])

    def escombra(mena, grau, durada):
        """Els errors de tots els tirs amb aquesta combinacio."""
        errors, incerteses = [], []
        for _, _, retard, taula, meta, _ in preses:
            r = troba_de_taula(taula, meta, mena=mena, grau=grau, durada_ajust_s=durada)
            if not r["hi_ha_amollada"]:
                errors.append(np.inf)
                incerteses.append(np.nan)
                continue
            errors.append(abs(r["amollada_s"] - retard))
            incerteses.append(r["incertesa_s"])
        return np.array(errors), np.array(incerteses)

    problemes = []
    print("   Quin model i quina finestra. Es proven totes les combinacions sobre")
    print("   tots els tirs i el que es vegi mana; triar-los primer i provar-los")
    print("   despres es com es tria malament.\n")
    # ELS DOS CRITERIS, i el segon val mes que el primer.
    #
    #   1. Als tirs NETS -els que tenen el soroll de seguiment de les preses de
    #      45 i 90 graus- la mediana de l'error ha de ser per sota de tres
    #      fotogrames. Als bruts no se li demana: una presa amb 1,2 graus de
    #      soroll dona legitimament una amollada pitjor, i el que s'ha de
    #      demanar alla es el criteri 2.
    #   2. Cada error ha de caure dins de TRES SIGMES de la incertesa que
    #      aquell mateix tir publica, i la meitat dels tirs n'han d'estirar
    #      almenys tres decimes. Es a dir: no es demana que el metode encerti
    #      sempre, es demana que SAPIGA QUANT HA ENCERTAT. Una incertesa que
    #      no cobreix l'error no serveix de res; una que el cobreix deu
    #      vegades, tampoc.
    print(f"   {'ajusta':>7} {'ordre':>6} {'finestra':>9} {'med. net':>10} "
          f"{'med. brut':>11} {'incertesa':>11} {'honesta?':>9}")
    millor = None
    for mena in ("omega", "theta"):
        for grau in (1, 3):
            for durada in (0.05, 0.08, 0.12):
                errors, incerteses = escombra(mena, grau, durada)
                med_net = float(np.median(errors[net]))
                med_brut = float(np.median(errors[~net]))
                bons = np.isfinite(errors) & np.isfinite(incerteses)
                raons = errors[bons] / incerteses[bons] if bons.any() else np.array([np.inf])
                honesta = bool(np.all(raons <= 3.0) and np.median(raons) >= 0.3
                               and bons.sum() == len(errors))
                marca = ""
                if (mena == MENA_PER_DEFECTE and grau == GRAU_PER_DEFECTE
                        and abs(durada - DURADA_AJUST_S) < 1e-9):
                    marca = "  <- el que es fa servir"
                    if med_net > TOLERANCIA:
                        problemes.append(
                            f"amb els valors per defecte ({mena}, ordre {grau}, finestra "
                            f"{durada} s) la mediana de l'error als tirs nets es "
                            f"{1000*med_net:.2f} ms, mes de {1000*TOLERANCIA:.1f}"
                        )
                    if not honesta:
                        problemes.append(
                            "amb els valors per defecte, la incertesa que es publica no "
                            "es honesta: o no cobreix l'error a tres sigmes, o es un "
                            "coixi que cap tir no arriba a estirar, o hi ha tirs sense "
                            "resultat"
                        )
                if millor is None or med_net + med_brut < millor[0]:
                    millor = (med_net + med_brut, mena, grau, durada)
                mitjana = float(np.nanmedian(incerteses)) if bons.any() else float("nan")
                print(f"   {mena:>7} {grau:>6} {durada:>8.2f}s {1000*med_net:>7.2f} ms "
                      f"{1000*med_brut:>8.2f} ms {1000*mitjana:>8.2f} ms "
                      f"{'si' if honesta else 'NO':>9}{marca}")

    print(f"\n   la millor d'aquesta taula: {millor[1]}, ordre {millor[2]}, "
          f"finestra {millor[3]} s")

    # Tir a tir amb els valors per defecte: l'instant i els angles. Els angles
    # es recuperen de la mitjana del repos, i per aixo s'hi demana el soroll
    # de seguiment i no mes.
    print("\n   Tir a tir, amb els valors per defecte. L'error es compara amb la")
    print("   incertesa que el mateix tir publica: la columna «sigmes» es el que")
    print("   de debo es demana, i ha de quedar per sota de 3.")
    print(f"   {'soroll':>7} {'angle':>6} {'retard':>9} {'error':>10} {'sigmes':>8} "
          f"{'angles trobats':>18} {'err':>7}")
    for quin, angle, retard, taula, meta, _ in preses:
        r = troba_de_taula(taula, meta)
        if not r["hi_ha_amollada"]:
            problemes.append(f"a {angle:.0f} graus amb retard {retard:.4f} s ({quin}) no "
                             f"troba l'amollada: {r['motiu']}")
            print(f"   {quin:>7} {angle:>6.0f} {retard:>9.4f}   {r['motiu'][:40]}")
            continue
        err_t = abs(r["amollada_s"] - retard)
        sigmes = err_t / r["incertesa_s"] if r["incertesa_s"] > 0 else np.inf
        a1, a2 = r["angles_amollada_graus"]
        err_a = max(abs(a1 - angle), abs(a2 - angle))
        # Els angles es comparen amb el soroll de seguiment D'AQUELL tir, que
        # es el que fixa fins on es pot llegir un angle en aquell video.
        sostre_angle = 1.5 * soroll_del_seguiment(meta)
        malament = sigmes > 3.0 or err_a > sostre_angle
        if sigmes > 3.0:
            problemes.append(f"a {angle:.0f} graus amb retard {retard:.4f} s ({quin}) "
                             f"l'error son {sigmes:.1f} sigmes de la incertesa publicada")
        if err_a > sostre_angle:
            problemes.append(f"a {angle:.0f} graus amb retard {retard:.4f} s ({quin}) els "
                             f"angles erren {err_a:.2f} graus, mes d'una vegada i mitja "
                             f"el soroll de seguiment ({sostre_angle/1.5:.2f})")
        print(f"   {quin:>7} {angle:>6.0f} {retard:>9.4f} {1000*err_t:>7.2f} ms "
              f"{sigmes:>8.1f} {a1:>9.3f} /{a2:>8.3f} {err_a:>7.3f}"
              f"{'   <<< FORA' if malament else ''}")

    # QUANT COSTA EL FILTRE. Aquesta part no prova res: mesura. Es torna a
    # buscar l'amollada dels mateixos tirs pero amb els angles SENSE FILTRAR,
    # que es el que tindria a la ma pendol/tracker.py si ho calcules durant la
    # conversio, abans de suavitzar. La diferencia entre les dues columnes es
    # exactament el preu de mesurar l'amollada sobre el CSV ja fet.
    from pendol.fisica import posicions
    from pendol.tracker import a_angles

    print("\n   Quant costa el filtre (no es cap prova, es una mesura):")
    errors_cru, errors_net = [], []
    for k, (quin, angle, retard, taula, meta, _) in enumerate(preses):
        r_net = troba_de_taula(taula, meta)
        # Els mateixos angles crus del tir, sense passar-los pel filtre.
        rng = np.random.default_rng(k + 100)
        estat0 = np.radians([angle, angle, 0.0, 0.0])
        tc, th1c, th2c = tir_fals_amb_repos(p, estat0, retard, 4.0, FPS)
        x1, y1, x2, y2 = posicions(th1c, th2c, p)
        soroll_m = SOROLLS[quin]
        x1, y1, x2, y2 = (v + rng.normal(0.0, soroll_m, v.shape) for v in (x1, y1, x2, y2))
        a1, a2 = a_angles(tc, x1, y1, x2, y2)
        r_cru = troba(tc, a1, a2, taula["omega1"], taula["omega2"],
                      finestra_filtre_s=2.0 / FPS,
                      soroll_graus=soroll_del_seguiment(meta))
        if r_net["hi_ha_amollada"]:
            errors_cru.append(np.nan)
            errors_net.append(abs(r_net["amollada_s"] - retard))
        if r_cru["hi_ha_amollada"]:
            errors_cru[-1] = abs(r_cru["amollada_s"] - retard)
    net, cru = np.array(errors_net), np.array(errors_cru)
    print(f"      {'':>14}{'mediana':>10}{'pitjor':>10}")
    for nom, v in (("amb filtre", net), ("sense filtre", cru)):
        print(f"      {nom:>14}{1000*np.nanmedian(v):>7.2f} ms{1000*np.nanmax(v):>7.2f} ms")
    print("      CONCLUSIO, i va contra el que s'esperava: treure el filtre NO")
    print("      millora res. La mediana amb prou feines es mou i el pitjor cas")
    print("      empitjora, perque sense suavitzar l'ajust es queda a mans del")
    print("      soroll de pixel. O sigui que mesurar l'amollada sobre el CSV ja")
    print("      fet no costa precisio, i no hi ha cap motiu tecnic per haver de")
    print("      calcular-la dins de pendol/tracker.py abans de suavitzar.")

    # Quin angle dona l'hora, i quant val el que no la dona. Es el numero que
    # sosté la decisio de no fer la mitjana dels dos a cegues.
    quin, angle, retard, taula, meta, _ = preses[len(RETARDS) * 2 + 1]
    r = troba_de_taula(taula, meta)
    print(f"\n   Els dos angles per separat (tir de {angle:.0f} graus a {retard:.4f} s):")
    for nom in ("theta1", "theta2"):
        a = r["per_angle"][nom]
        diu = (f"diu {a['instant_s']:.4f} s, error {1000*abs(a['instant_s']-retard):7.2f} ms"
               if "instant_s" in a else "l'ajust no dona res")
        print(f"      {nom}: es mou {a['recorregut_graus']:6.2f} graus "
              f"({a['rao_senyal_soroll']:5.1f} vegades el soroll), {diu}"
              f"   {'COMPTA' if a['compta'] else 'nomes diagnostic'}")

    # Una presa que arrenca ja en marxa no ha de donar cap amollada inventada.
    # Es talla ben endins del moviment, que es el cas de les tres preses reals
    # que tenen el marcatge començat amb el pendol ja engegat.
    quin, angle, retard, taula, meta, _ = preses[0]
    i = int(round((retard + 0.30) * FPS))
    escapcada = taula.iloc[i:].reset_index(drop=True)
    escapcada["t"] = escapcada["t"] - escapcada["t"].iloc[0]
    r = troba_de_taula(escapcada, meta)
    if r["hi_ha_amollada"]:
        problemes.append("d'una presa escapcada que arrenca en marxa se n'ha tret una amollada")
    print(f"\n   Presa escapcada (arrenca en marxa): {'no en treu amollada, be' if not r['hi_ha_amollada'] else 'SE N INVENTA UNA'}")
    if not r["hi_ha_amollada"]:
        print(f"      motiu: {r['motiu']}")

    # REPOS CURT. Els tres retards de dalt deixen com a minim 0,2 s de quietud
    # al davant, i amb aixo els angles sempre surten de la mitjana del repos.
    # Les preses de debo no sempre en tenen tant: a la 90_2 el marcatge comenca
    # pocs fotogrames abans d'amollar, i alli els angles sortien d'extrapolar
    # la parabola de la finestra d'ajust cap enrere, amb 2,6 graus d'error. Cap
    # tir de la prova no passava per aquell cami, i per aixo no se'n sabia res.
    #
    # 0,0371 s son 8,9 fotogrames: prou per trobar l'amollada -el minim es mitja
    # finestra de filtre- i massa poc per a MOSTRES_REPOS_MINIM mostres netes.
    #
    # EL CRITERI D'AQUESTA SECCIO ES VA CANVIAR DESPRES DE MIRAR, i es diu. Els
    # ANGLES es demanen a tots els tirs que troben l'amollada. L'INSTANT, nomes
    # als nets. Als tirs de 4 px el primer resultat (24/09/2026) va ser que el
    # de 150 graus no trobava l'amollada, i provant-ne deu llavors l'instant
    # d'un tir de 150 graus amb 1,2 graus de soroll erra de 30 a 100 ms, ja sigui
    # el repos curt o llarg:
    #
    #     repos     error de l'instant (10 llavors)   fora de 3 sigmes   sense amollada
    #     37 ms     +1 a +51 ms                        1 (4,2)            2
    #     62 ms     -29 a +59 ms                       0                  1
    #     203 ms    -100 a +17 ms                      0, amb incerteses enormes
    #
    # Es un limit del metode amb soroll alt, no del repos curt, i els tres
    # retards de dalt passen a 150 graus i 4 px perque cada un en prova una sola
    # llavor. Es va decidir declarar-ho i mirar-ho a part ABANS del pas 2 de la
    # conversio (documentacio/pla-doble-interactiu.md, bloc D1b), que decidira
    # el repos amb aquest modul. La taula surt d'esborranys/_diag_curt_150.py,
    # que NO es versiona: si aquests numeros van a la memoria, han de pujar a
    # eines/ amb aquella investigacio.
    RETARD_CURT = 0.0371
    print(f"\n   Repos curt: {1000*RETARD_CURT:.1f} ms de quietud abans d'amollar, massa")
    print(f"   poc per fer la mitjana del repos net. Els angles es demanen a tots;")
    print(f"   l'instant, nomes als nets (limit declarat, vegeu el codi).")
    print(f"   {'soroll':>7} {'angle':>6} {'error':>10} {'sigmes':>8} "
          f"{'angles trobats':>18} {'err':>7}  d'on surten")
    for quin, soroll_m in SOROLLS.items():
        for angle in (45.0, 90.0, 120.0, 150.0):
            cru = desti / f"cru_curt_{quin[0]}_{angle:.0f}.csv"
            net_curt = desti / f"net_curt_{quin[0]}_{angle:.0f}.csv"
            # Llavors a part de les de dalt: si es repetissin, el soroll
            # d'aquests tirs seria el mateix que el d'aquells i els errors
            # sortirien correlacionats.
            rigidesa = _presa_falsa(p, cru, RETARD_CURT, 4.0, FPS, angle, soroll_m,
                                    llavor=300 + int(angle) + (1000 if quin != "1 px" else 0))
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.processa(cru, net_curt, pivot=(0.0, 0.0), y_avall=False,
                                 meta_extra={"angle_nominal_graus": angle,
                                             "geometria": {"rigidesa": rigidesa}})
            taula, meta = dades.carrega(net_curt)
            r = troba_de_taula(taula, meta)
            es_net = quin == list(SOROLLS)[0]
            if not r["hi_ha_amollada"]:
                if es_net:
                    problemes.append(f"repos curt, {angle:.0f} graus ({quin}): no troba "
                                     f"l'amollada: {r['motiu']}")
                print(f"   {quin:>7} {angle:>6.0f}   {r['motiu'][:50]}"
                      f"{'   <<< FORA' if es_net else '   (brut: limit declarat)'}")
                continue
            err_t = abs(r["amollada_s"] - RETARD_CURT)
            sigmes = err_t / r["incertesa_s"] if r["incertesa_s"] > 0 else np.inf
            a1, a2 = r["angles_amollada_graus"]
            # Amb np.max, que propaga el nan, i no amb el max de Python: un
            # angle que surt nan es un error, i max(1.7, nan) torna 1.7 i se
            # l'empassa sense dir res.
            err_a = float(np.max(np.abs([a1 - angle, a2 - angle])))
            if not np.isfinite(err_a):
                err_a = np.inf
            sostre_angle = 1.5 * soroll_del_seguiment(meta)
            instant_fora = es_net and sigmes > 3.0
            if instant_fora:
                problemes.append(f"repos curt, {angle:.0f} graus ({quin}): l'error de "
                                 f"l'instant son {sigmes:.1f} sigmes de la incertesa publicada")
            if err_a > sostre_angle:
                problemes.append(f"repos curt, {angle:.0f} graus ({quin}): els angles erren "
                                 f"{err_a:.2f} graus, mes d'una vegada i mitja el soroll "
                                 f"de seguiment ({sostre_angle/1.5:.2f})")
            print(f"   {quin:>7} {angle:>6.0f} {1000*err_t:>7.2f} ms {sigmes:>8.1f} "
                  f"{a1:>9.3f} /{a2:>8.3f} {err_a:>7.3f}  {r['origen_dels_angles'][:34]}"
                  f"{'   <<< FORA' if instant_fora or err_a > sostre_angle else ''}")

    # I el rellotge: els instants que en surten han de ser EXACTAMENT els de la
    # presa, sense interpolar res.
    from pendol.simulador import instants_de_la_presa

    quin, angle, retard, taula, meta, _ = preses[1]
    r = troba_de_taula(taula, meta)
    t_eval, fila = instants_de_la_presa(taula, r["amollada_s"])
    t_presa = np.asarray(taula["t"], dtype=float)
    exacte = np.array_equal(t_eval, t_presa[fila:] - r["amollada_s"])
    if not exacte:
        problemes.append("els instants de --com-el-tracker no son els de la presa")
    print(f"\n   Rellotge del Tracker: {len(t_eval)} instants des de la fila {fila}, "
          f"{'identics' if exacte else 'NO IDENTICS'} als de la presa")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print(f"OK -> l'amollada surt dins de tres fotogrames ({1000*TOLERANCIA:.1f} ms) de mediana")
    print(f"     als tirs nets, i als {len(preses)} tirs l'error cau dins de la incertesa que")
    print("     cada tir publica. Quatre angles de sortida i dos nivells de soroll.")
    print("     Amb repos curt, els angles de sortida surten del repos que hi ha i")
    print("     no d'extrapolar. Limit declarat: amb el soroll de 150 graus l'instant")
    print("     erra de 30 a 100 ms i no sempre ho sap (vegeu el capdamunt del modul).")
    print("     Limit declarat: aixo prova el metode sobre un pendol que s'amolla")
    print("     EXACTAMENT del repos i sense cap empenta. Si la ma hi dona un cop")
    print("     en deixar-lo anar, la hipotesi de velocitat zero falla i aquesta")
    print("     prova no ho pot veure; qui ho canta es el residu de l'ajust i la")
    print("     discrepancia entre theta1 i theta2, que per aixo es publiquen.")
    return 0


def main(argv=None) -> int:
    import argparse
    import json

    from pendol import dades

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("presa", nargs="?", help="CSV canonic d'una presa")
    ap.add_argument("--mena", choices=("omega", "theta"), default=MENA_PER_DEFECTE,
                    help="sobre que s'ajusta el model: la velocitat o l'angle")
    ap.add_argument("--grau", type=int, default=GRAU_PER_DEFECTE,
                    help="ordre del model comptat sobre la velocitat (1 o 3)")
    ap.add_argument("--finestra", type=float, default=DURADA_AJUST_S,
                    help="llargada de la finestra d'ajust (s)")
    ap.add_argument("--json", help="desa el resultat en aquest fitxer")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.presa:
        ap.error("digue'm quina presa, o passa --autoprova")

    taula, meta = dades.carrega(args.presa)
    r = troba_de_taula(taula, meta, mena=args.mena, grau=args.grau,
                       durada_ajust_s=args.finestra)

    print(f"{args.presa}")
    if not r["hi_ha_amollada"]:
        print(f"  SENSE AMOLLADA: {r['motiu']}")
    else:
        print(f"  amollada a  {r['amollada_s']:.4f} s  "
              f"+- {1000*r['incertesa_s']:.2f} ms ({r['incertesa_fotogrames']:.2f} fotogrames)")
        print(f"  angles      {r['angles_amollada_graus'][0]:.3f} / "
              f"{r['angles_amollada_graus'][1]:.3f} graus ({r['origen_dels_angles']})")
        print(f"  l'hora la donen: {', '.join(r['angles_que_compten'])}"
              f"   (repos abans de l'amollada: {1000*r['repos_abans_s']:.0f} ms)")
        if np.isfinite(r["discrepancia_entre_angles_s"]):
            print(f"  theta1 i theta2 difereixen en "
                  f"{1000*r['discrepancia_entre_angles_s']:.2f} ms "
                  f"(un fotograma son {1000/r['fps']:.2f})")
        else:
            quin = [n for n, a in r["per_angle"].items() if "instant_s" not in a]
            print(f"  sense comprovacio creuada: l'ajust de {', '.join(quin)} no dona res")
        for a in r["avisos"]:
            print(f"  ATENCIO: {a}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  desat a {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
