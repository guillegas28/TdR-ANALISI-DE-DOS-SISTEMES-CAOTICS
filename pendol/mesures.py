r"""Les mesures que es fan sobre una trajectoria, vingui d'on vingui.

Aquest modul es PRIM a posta: no importa matplotlib, no llegeix cap fitxer i
no sap d'on venen les dades. Aixi el poden fer servir pendol/comparar.py, les
eines de eines/ i la finestra interactiva sense arrossegar-hi res al darrere.

QUE HI HA AQUI I PER QUE HI ES

    dif_angular      la diferencia de dos angles, plegada a [-pi, pi]
    omega_zero       l'escala de velocitat angular del muntatge (rad/s)
    distancia_fases  la separacio de dues trajectories a l'espai de fases
    voltes           quantes voltes senceres escombra un angle

Les dues ultimes son el fonament de l'experiment 4 -l'exponent de Lyapunov
mesurat per parelles de tirs reals- i tambe de la lectura en directe de la
finestra interactiva. Viuen aqui i no dins de cap eina perque han de donar EL
MATEIX NUMERO a tot arreu: si la finestra ensenya una separacio i la figura de
la memoria n'ensenya una altra, no hi ha manera de saber quina de les dues
val.

    .venv\\Scripts\\python -m pendol.mesures --autoprova
"""

from __future__ import annotations

import numpy as np

# L'ordre del contracte de dades, sense la columna t. Vegeu pendol/dades.py.
COLUMNES_ESTAT = ("theta1", "theta2", "omega1", "omega2")


def estats_de_taula(taula) -> np.ndarray:
    """D'una taula canonica a la matriu (4, N) amb que treballa aquest modul.

    Es exactament la forma que retorna pendol.simulador.simula, de manera que
    una presa del Tracker i una simulacio entren per la mateixa porta i cap
    funcio d'aqui no ha de saber d'on ve el que li donen.
    """
    return np.asarray([np.asarray(taula[c], dtype=float) for c in COLUMNES_ESTAT])


def dif_angular(a, b):
    """Diferencia d'angles plegada a [-pi, pi].

    Cal plegar-la: dos angles de 359 i 1 graus difereixen 2 graus, no 358.
    Sense aixo, la corba de divergencia fa salts artificials de 2*pi.
    """
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def omega_zero(p) -> float:
    r"""L'escala de velocitat angular del muntatge, en rad/s.

    Serveix per adimensionalitzar les velocitats a distancia_fases(). Sense
    ella sumariem radians amb rad/s, que no es cap norma: el numero que en
    sortiria dependria de si el temps es mesura en segons o en minuts.

    QUINA ESCALA, I PER QUE AQUESTA. Es la frequencia de petites oscil.lacions
    del COS COMPOST: les dues barres lligades rectes penjant del pivot de
    dalt. Amb l'articulacio bloquejada th1 = th2 i tot gira junt, o sigui que

        I_total = I1 + I2 + 2*K        (el 2*K ve del terme creuat)
        M*d     = A + B
        L_eq    = I_total / (A + B)    longitud equivalent del pendol
        omega0  = sqrt(g / L_eq) = sqrt( g*(A + B) / I_total )

    Amb el muntatge d'avui dona 8,06 rad/s. El periode que es mesura al banc
    per a aquesta mateixa configuracio es 0,7833 s, que son 8,02 rad/s: mig
    per cent. No es cap validacio nova -es el mateix desacord del -0,89 %
    entre el I del model i el del periode que ja comprova
    eines/comprova_energia.py- pero val la pena saber que aquesta constant no
    surt d'enlloc: surt d'un experiment que es va fer de debo.

    COMPTE AMB LA FORMULA VELLA. documentacio/lyapunov-laboratori.html diu
    sqrt(g/L1) = 9,04 rad/s. Aquella era del model de MASSES PUNTUALS, on L1
    era la llargada del brac. Al pendol fisic L1 es la distancia entre pivots
    i no es cap longitud de pendol: la formula ja no hi val.

    Es una TRIA, no una llei. Qualsevol altra escala del mateix ordre donaria
    una distancia igual de valida; el que no val es no dir-la. Per aixo el
    numero usat ha d'anar a les metadades de tota figura que en depengui.
    """
    I_total = p.I1 + p.I2 + 2 * p.K
    return float(np.sqrt(p.g * (p.A + p.B) / I_total))


def distancia_fases(a, b, omega0: float):
    r"""Separacio de dues trajectories a l'espai de fases. Adimensional.

        d = sqrt( dth1^2 + dth2^2 + (dw1/omega0)^2 + (dw2/omega0)^2 )

    `a` i `b` son matrius (4, N) -o vectors (4,)- amb l'ordre del contracte:
    theta1, theta2, omega1, omega2. Retorna un vector de N distancies (o un
    escalar si li dones dos estats sols).

    TRES DECISIONS, i totes tres mouen el numero, o sigui que van escrites:

    1. Les diferencies d'angle es PLEGUEN (dif_angular). Dues trajectories que
       nomes es diferencien en una volta sencera son al mateix lloc de
       l'espai de fases, i sense plegar sortirien separadissimes.

    2. Els quatre components pesen IGUAL. Un cop adimensionalitzades les
       velocitats no hi ha cap motiu fisic per pesar mes els angles, i
       qualsevol altre repartiment seria una tria amagada dins d'un numero.

    3. Es ADIMENSIONAL. Si mesurressis el temps en mil.lisegons, les dues
       velocitats i omega0 es dividirien per mil totes tres i la d no es
       mouria. Aixo es el que permet comparar la d d'un tir de 45 graus amb
       la d'un de 150, i ho comprova l'autoprova.

    El numero surt en radians d'angle equivalent: si dues trajectories
    difereixen nomes en theta1, la d ES aquella diferencia en radians.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"formes diferents: {a.shape} i {b.shape}")
    if a.shape[0] != 4:
        raise ValueError(
            f"esperava 4 components (theta1, theta2, omega1, omega2) i n'hi ha {a.shape[0]}. "
            "D'una taula canonica en surten amb estats_de_taula()."
        )
    dth1 = dif_angular(a[0], b[0])
    dth2 = dif_angular(a[1], b[1])
    dw1 = (a[2] - b[2]) / omega0
    dw2 = (a[3] - b[3]) / omega0
    return np.sqrt(dth1**2 + dth2**2 + dw1**2 + dw2**2)


def voltes(theta) -> float:
    r"""Quantes voltes senceres escombra un angle: el RECORREGUT, no el cami.

        voltes = ( max(theta) - min(theta) ) / (2*pi)

    COMPTE, que son dues coses diferents. Un angle que va i ve cent vegades
    entre 0 i 237 graus dona 0,66 voltes, no 66: el que es mesura es fins on
    arriba, no quant camina. Es a posta, perque la pregunta que respon es
    «el segon brac passa mai per dalt?», i per a aixo compta l'abast.

    L'angle ha d'estar DESEMBOLICAT (np.unwrap), cosa que el contracte de
    dades del projecte ja garanteix. Sobre angles plegats a [-180, 180] el
    resultat no pot passar mai d'1 volta i no voldria dir res.

    PER QUE AIXO IMPORTA. Es el criteri de caos del treball que NO DEPEN DE
    CAP MODEL ni de cap ajust: nomes de comptar. Els numeros publicats a
    documentacio/limitacions-preses-i-friccio.html, seccio 02, son 0,66 als
    45 graus -el segon brac no bolca mai-, entre 3,6 i 15,6 als 90 i entre
    6,8 i 21,0 als 150 (fins al 24/09/2026 hi deia 8,9-11,9, que eren dues
    preses de dotze). Fins al 23/09 sortien de esborranys/revisa_doble.py,
    que es un esborrany; ara surten d'aqui.
    """
    th = np.asarray(theta, dtype=float)
    return float((th.max() - th.min()) / (2 * np.pi))


def voltes_fins_ara(theta) -> np.ndarray:
    """Les voltes que porta un angle a cada instant: voltes(theta[:i+1]).

    Es el comptador que ensenya pendol/interactiu.py mentre el pendol es mou.
    Mateixa definicio que voltes() -el recorregut, no el cami-, calculada amb
    el maxim i el minim acumulats, o sigui que mai no baixa i l'ultim valor es
    exactament el de voltes().
    """
    th = np.asarray(theta, dtype=float)
    return (np.maximum.accumulate(th) - np.minimum.accumulate(th)) / (2 * np.pi)


# --------------------------------------------------------------- l'autoprova


def autoprova() -> int:
    """Comprova les propietats que han de valer SEMPRE, no uns valors concrets.

    Una autoprova que compares numeros contra una taula escrita a ma nomes
    diria que el codi no ha canviat. El que es comprova aqui son identitats:
    que dues trajectories iguals disten zero, que plegar funciona, que la
    norma es adimensional de debo i que omega_zero surt del muntatge i no
    d'un numero copiat.
    """
    from pendol.fisica import Parametres

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 2.0, (4, 500))

    print("Autoprova de pendol/mesures.py\n")
    print("1. distancia_fases")
    w0 = 8.0
    mira(np.allclose(distancia_fases(a, a, w0), 0.0),
         "dues trajectories identiques disten zero")

    b = a.copy()
    b[0] += 2 * np.pi
    b[1] -= 4 * np.pi
    mira(np.allclose(distancia_fases(a, b, w0), 0.0),
         "una volta sencera de mes en un angle no es cap separacio")

    # Adimensional: mesurar el temps en mil.lisegons divideix per mil les dues
    # velocitats I l'escala omega0. La d no s'ha de moure ni un pel.
    c = a.copy()
    c[0] += 0.01
    d_s = distancia_fases(a, c, w0)
    a_ms, c_ms = a.copy(), c.copy()
    a_ms[2:] /= 1000.0
    c_ms[2:] /= 1000.0
    d_ms = distancia_fases(a_ms, c_ms, w0 / 1000.0)
    mira(np.allclose(d_s, d_ms),
         "la norma no canvia si el temps es mesura en mil.lisegons")

    # Si nomes difereixen en theta1, la d ES aquella diferencia en radians.
    mira(np.allclose(d_s, 0.01),
         "amb nomes 0,01 rad de diferencia a theta1, la d val 0,01")

    # I cada component ha de pesar igual: 0,01 rad a theta2 o 0,01*omega0
    # rad/s a omega1 han de donar exactament la mateixa distancia.
    for i, quant in ((1, 0.01), (2, 0.01 * w0), (3, 0.01 * w0)):
        e = a.copy()
        e[i] += quant
        if not np.allclose(distancia_fases(a, e, w0), 0.01):
            problemes.append(f"el component {COLUMNES_ESTAT[i]} no pesa igual que els altres")
    mira(all("no pesa igual" not in p for p in problemes),
         "els quatre components pesen igual un cop adimensionalitzats")

    print("\n2. voltes")
    th = np.linspace(0.0, 3.25 * 2 * np.pi, 4000)
    mira(abs(voltes(th) - 3.25) < 1e-9, "una rotacio de 3,25 voltes en dona 3,25")

    # Cent oscil.lacions d'amplitud 237 graus: el RECORREGUT es 237 graus.
    tt = np.linspace(0.0, 100.0, 40000)
    th = np.radians(237.0 / 2) * np.sin(2 * np.pi * tt)
    mira(abs(voltes(th) - 237.0 / 360.0) < 1e-6,
         "cent anades i vingudes de 237 graus donen 0,66 voltes, no 66")

    acum = voltes_fins_ara(th)
    mira(acum[-1] == voltes(th) and np.all(np.diff(acum) >= 0.0),
         "voltes_fins_ara mai no baixa i acaba exactament en voltes()")

    # Les identitats de dalt proven que el codi fa el que diu. Aixo d'aqui
    # prova una altra cosa, i val mes: que la DEFINICIO sigui la mateixa que
    # va donar els numeros que ja son escrits als documents. Es fa nomes si
    # les preses hi son, perque els CSV de dades/net/ no van al repositori.
    print("\n2b. voltes sobre les preses de debo")
    from pathlib import Path

    arrel = Path("dades/net")
    grups: dict[str, list[tuple[str, float]]] = {}
    for cami in sorted(arrel.glob("*.csv")) if arrel.exists() else []:
        nom = cami.stem
        if nom in ("magnetic_3072", "prova") or "_" not in nom:
            continue
        try:
            import pandas as pd

            taula = pd.read_csv(cami)
            grups.setdefault(nom.split("_")[0], []).append((nom, voltes(taula["theta2"])))
        except Exception as e:  # noqa: BLE001
            problemes.append(f"no he pogut llegir {cami.name}: {e}")

    if not grups:
        print("      no hi ha preses a dades/net/: aquesta part no s'ha pogut fer.")
        print("      Es refan amb eines/converteix_preses.py. No ha fallat res.")
    else:
        for g in sorted(grups, key=int):
            vals = [v for _, v in grups[g]]
            print(f"      {g:>4} graus: {min(vals):.2f} a {max(vals):.2f} voltes "
                  f"({len(vals)} preses)")
        # Els 90 graus son la comprovacio de debo: el document de limitacions
        # hi diu «entre 3,6 i 15,6», i aixo ha de sortir clavat.
        if "90" in grups:
            vals = [v for _, v in grups["90"]]
            clava = abs(min(vals) - 3.6) < 0.05 and abs(max(vals) - 15.6) < 0.05
            mira(clava, "als 90 graus surt el 3,6-15,6 que ja hi ha publicat")
        # Els 150 van ser els que ho van destapar: fins al 24/09/2026 el
        # document hi deia 8,9-11,9, que eren la 150_11 i la 150_12 i prou.
        # Corregit a 6,8-21,0 sobre les dotze. Si algun dia torna a sortir un
        # altre rang, el document s'ha de tornar a mirar.
        if "150" in grups and len(grups["150"]) == 12:
            vals = [v for _, v in grups["150"]]
            clava = abs(min(vals) - 6.8) < 0.05 and abs(max(vals) - 21.0) < 0.05
            mira(clava, "als 150 graus surt el 6,8-21,0 que hi ha publicat des del 24/09")

    print("\n3. omega_zero")
    p = Parametres.des_de_json("parametres.json")
    w0_p = omega_zero(p)
    print(f"      omega0 del muntatge = {w0_p:.4f} rad/s "
          f"(periode equivalent {2*np.pi/w0_p:.4f} s)")

    # Ha d'escalar com sqrt(g): es l'unica cosa dimensional que hi entra.
    from dataclasses import replace
    mira(abs(omega_zero(replace(p, g=4 * p.g)) - 2 * w0_p) < 1e-12,
         "amb quatre vegades la gravetat, omega0 es dobla")

    # I la inercia que hi ha a sota s'ha de poder calcular per un altre cami:
    # amb els bracos rectes tot es un sol cos rigid, i la barra 2 hi entra
    # sencera a distancia L1 + d2 del pivot de dalt. Que aixo doni el mateix
    # que I1 + I2 + 2K es el que prova que el terme creuat porta el seu 2.
    I_del_model = p.I1 + p.I2 + 2 * p.K
    I_directe = ((1 / 12) * p.m1 * p.R1**2 + p.m1 * p.d1**2
                 + p.ms * p.rs**2
                 + (1 / 12) * p.m2 * p.R2**2 + p.m2 * (p.L1 + p.d2) ** 2)
    mira(abs(I_del_model / I_directe - 1.0) < 1e-12,
         "I1 + I2 + 2K es la inercia del cos rigid amb els bracos rectes")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> les mesures compleixen les propietats que les fan comparables "
          "entre tirs.")
    print("     Limit declarat: aixo prova les IDENTITATS, no que omega0 sigui")
    print("     la millor escala possible. Que sigui una tria es per que el seu")
    print("     valor ha d'anar a les metadades de tota figura.")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova les propietats de les mesures i surt")
    args = ap.parse_args(argv)
    if not args.autoprova:
        ap.error("aquest modul es una biblioteca: de moment nomes te --autoprova")
    return autoprova()


if __name__ == "__main__":
    raise SystemExit(main())
