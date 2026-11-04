"""Fisica del pendol magnetic: equacions del moviment, alcada i energia.

Aquest modul NO integra res: nomes diu quina acceleracio te la massa a cada
punt. L'integrador es a magnetic/motor.py i la validacio a magnetic/valida.py.

No importa res de pendol/, a posta: el pendol doble i el magnetic son sistemes
diferents i s'han de poder entendre per separat.

EL MODEL
--------
Amb l'estat [x, y, vx, vy] al pla horitzontal dels imants i

    s       = sqrt(L^2 - x^2 - y^2)             del pivot a la massa, en vertical
    d       = d0 + L - s                        alcada de la massa sobre el pla
    delta_i = sqrt((x-xi)^2 + (y-yi)^2 + d^2)   distancia 3D a l'imant i

les equacions son

    x'' = -(g + fz)*x/s - b*x' + SUM_i K_i*(xi - x)/delta_i^(n+1)
    y'' = -(g + fz)*y/s - b*y' + SUM_i K_i*(yi - y)/delta_i^(n+1)

    fz  = SUM_i K_i*d/delta_i^(n+1)             l'estirada CAP AVALL dels imants

TOT VA PER UNITAT DE MASSA. La massa m NO surt a les equacions: el mapa de
conques depen de g/L, b, K, n, R i d0, i de res mes. Si canvies la massa del
pendol sense tocar res mes, el fractal no canvia. La m nomes serveix per
convertir una forca mesurada al laboratori en la constant K.

D'on surt cada terme:

  * -g*x/s es la component horitzontal de la tensio del fil. NO es
    l'aproximacio de petits angles -g*x/L. La s ja s'ha de calcular per a
    l'alcada, o sigui que la versio bona surt de franc (la taula 4 de la
    comprovacio d'aquest modul diu quant s'equivoca l'altra).
  * d creix quan la massa se separa del centre, perque en desplacar-se PUJA, i
    llavors els imants l'estiren menys. Es l'efecte esferic, i es el que fa que
    aquest sigui el model del muntatge real i no un model de joguina.
  * La component horitzontal dels imants va directa. Dividir per delta un cop
    mes que la llei K/delta^n dona el vector unitari, i d'aqui l'exponent n+1.
  * LA VERTICAL TAMBE COMPTA, i es el que diu fz (corregit el 25/10/2026). La
    tensio del fil no s'empassa tota la forca vertical, nomes la component al
    llarg del fil: si els imants estiren la massa cap avall, el balanc vertical
    demana una tensio mes gran, i la seva component horitzontal tambe ho es.
    L'imant fa de gravetat de mes, g -> g + fz, pel mateix cami que dona el
    -g*x/s. Fins al 25/10 aquest terme NO hi era, i el model no conservava
    l'energia amb els imants encesos; ara es exactament
        x'' = -dV/dx - b*x'      V = g*(L - s) + SUM_i U_i(delta_i)
    amb U_i = -K_i/((n-1)*delta_i^(n-1)), i energia() la conserva. Ho va fer
    veure magnetic/imant_rigid.py: amb el terme, una sola llei central explica
    les dues tandes de la llei de forca.

Comprovacio d'aquest modul, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.fisica
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from magnetic.compila import njit

CAMI_PER_DEFECTE = Path(__file__).resolve().parent / "parametres.json"


# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class Parametres:
    """El muntatge. Els imants van en vectors paral.lels, numerats des de 1."""

    L: float
    R: float
    d0: float
    m: float
    g: float
    b: float
    n: float
    im_x: np.ndarray
    im_y: np.ndarray
    im_K: np.ndarray
    noms: tuple = ()
    inventats: tuple = ()

    @classmethod
    def des_de_json(cls, cami=CAMI_PER_DEFECTE) -> "Parametres":
        d = json.loads(Path(cami).read_text(encoding="utf-8"))
        imants = d["imants"]
        R = float(d["R"])
        angles = np.radians([float(i["angle_graus"]) for i in imants])
        return cls(
            L=float(d["L"]),
            R=R,
            d0=float(d["d0"]),
            m=float(d["m"]),
            g=float(d["g"]),
            b=float(d["b"]),
            n=float(d["n"]),
            im_x=R * np.cos(angles),
            im_y=R * np.sin(angles),
            im_K=np.array([float(i["K"]) for i in imants], dtype=float),
            noms=tuple(str(i["nom"]) for i in imants),
            inventats=tuple(d.get("_valors_inventats", ())),
        )

    def sense_imants(self) -> "Parametres":
        """El mateix muntatge amb els imants retirats: un pendol simple."""
        return replace(self, im_K=np.zeros_like(self.im_K))

    def sense_friccio(self) -> "Parametres":
        return replace(self, b=0.0)

    def com_a_dict(self) -> dict:
        """Per als JSON de metadades. La marca d'inventat hi va enganxada."""
        return {
            "L": self.L,
            "R": self.R,
            "d0": self.d0,
            "m": self.m,
            "g": self.g,
            "b": self.b,
            "n": self.n,
            "imants": [
                {"nom": nm, "x": float(x), "y": float(y), "K": float(k)}
                for nm, x, y, k in zip(self.noms, self.im_x, self.im_y, self.im_K)
            ],
            "valors_inventats": list(self.inventats),
            "_avis": (
                "Els valors llistats a valors_inventats NO son mesures: se'ls ha "
                "donat un numero nomes per poder provar el codi."
            )
            if self.inventats
            else "Tots els parametres son mesures.",
        }


def avis_inventats(p: Parametres) -> str:
    """El rerol que ha de sortir a la consola abans de calcular res."""
    if not p.inventats:
        return ""
    return (
        "!! AVIS: hi ha VALORS INVENTATS en us: "
        + ", ".join(p.inventats)
        + "\n   No son mesures del teu muntatge. El que surti d'aqui sera correcte com\n"
        "   a calcul, pero no descriu el teu pendol fins que estiguin mesurats.\n"
        "   Com es mesuren: documentacio/calibrar-el-magnetic.html"
    )


# ---------------------------------------------------------------------------
# Provar altres valors sense tocar parametres.json
# ---------------------------------------------------------------------------
# PER QUE EXISTEIX. La b, la n i la K son els tres valors inventats, i mirar que
# passaria amb uns altres es una pregunta constant. Fins ara l'unica manera era
# editar parametres.json, calcular, i recordar-se de tornar-l'hi a posar -- amb
# el risc evident de deixar-lo tocat sense adonar-se'n i que tots els mapes
# seguents fossin d'un altre muntatge sense que ho digues res.
#
# PER QUE BANDERES I NO UN SEGON parametres.json. Un fitxer a part hauria de
# duplicar L, R, d0, m i g, i el dia que se'n mesuri un quedaria mentint. Una
# bandera no pot envellir. I la traca no es perd: com_a_dict() ja desa TOTS els
# parametres dins del JSON de cada graella, o sigui que la figura continua sabent
# dir d'on surt.
def sobreescriu(p: "Parametres", b=None, n=None, K=None) -> tuple["Parametres", str]:
    """Canvia b, n o les tres K. Retorna (parametres, avis); l'avis es "" si res.

    UN SOL --K POSA EL MATEIX VALOR ALS TRES IMANTS. No es una simplificacio per
    estalviar banderes: amb les tres K iguals el muntatge te la simetria de 120
    graus, i es aquesta simetria la que fa que els punts de repos es puguin
    buscar pel radi en comptes d'amb un fsolve en dues dimensions (vegeu
    punts_dequilibri() aqui i puntsDeRepos() a magnetic_interactiu/motor.js).
    El dia que es mesurin tres K diferents, van a parametres.json.

    L'avis es MES FORT que el de valors inventats, i ha de ser-ho: amb un valor
    sobreescrit ja no s'esta ni als valors provisionals del fitxer.
    """
    canvis, camps = {}, {}
    if b is not None:
        canvis["b"] = camps["b"] = float(b)
    if n is not None:
        canvis["n"] = camps["n"] = float(n)
    if K is not None:
        canvis["K"] = float(K)
        camps["im_K"] = np.full(len(p.im_K), float(K))
    if not canvis:
        return p, ""

    nou = replace(p, **camps)
    llista = ", ".join(f"{clau} = {valor:g}" for clau, valor in canvis.items())
    avis = (
        "!! MUNTATGE ARBITRARI: s'han sobreescrit valors des de la linia d'ordres:\n"
        f"   {llista}\n"
        "   Aixo NO son ni els valors provisionals de parametres.json. El que surti\n"
        "   d'aqui no descriu cap pendol: serveix per veure QUE PASSARIA, i prou.\n"
        "   Els valors de debo son a magnetic/parametres.json i no s'han tocat."
    )
    if K is not None:
        avis += "\n   La --K s'ha posat igual als tres imants (simetria de 120 graus)."
    return nou, avis


def afegeix_banderes_de_parametres(ap) -> None:
    """Les tres banderes --b/--n/--K, escrites un sol cop per a tots els mòduls.

    Va aqui i no a cada main() perque el text d'ajuda ha de dir el mateix a tot
    arreu: si a mapa.py digues una cosa i a construeix.py una altra, algun dia
    voldrien dir coses diferents.
    """
    ap.add_argument("--b", type=float, default=None,
                    help="friccio per unitat de massa (s^-1), en lloc de la de parametres.json")
    ap.add_argument("--n", type=float, default=None,
                    help="exponent de la llei de forca, en lloc del de parametres.json")
    ap.add_argument("--K", type=float, default=None,
                    help="constant dels TRES imants (m^(n+1)/s^2), en lloc de la de parametres.json")


# ---------------------------------------------------------------------------
# Les equacions
# ---------------------------------------------------------------------------
@njit(cache=True)
def acceleracio_nucli(x, y, vx, vy, L, g, b, n, d0, im_x, im_y, im_K):
    """(ax, ay) a partir d'escalars i vectors plans.

    Escrita amb un bucle i floats a posta, no vectoritzada: aixi Numba la pot
    compilar tal com esta, sense duplicar ni una linia de fisica. El @njit
    ve de magnetic.compila i es pot apagar amb MAGNETIC_SENSE_NUMBA=1.
    La versio comoda, que treballa amb l'estat i els Parametres, es
    acceleracio().
    """
    s = math.sqrt(L * L - x * x - y * y)
    d = d0 + L - s

    # Primer els imants, perque la seva estirada vertical entra a la gravetat.
    mx = 0.0
    my = 0.0
    gv = g
    for i in range(im_x.shape[0]):
        dx = im_x[i] - x
        dy = im_y[i] - y
        delta = math.sqrt(dx * dx + dy * dy + d * d)
        forca = im_K[i] / delta ** (n + 1.0)
        mx += forca * dx
        my += forca * dy
        gv += forca * d

    ax = -gv * x / s - b * vx + mx
    ay = -gv * y / s - b * vy + my

    return ax, ay


def acceleracio(estat, p: Parametres):
    """(ax, ay) a partir de l'estat [x, y, vx, vy] i els Parametres."""
    x, y, vx, vy = (float(v) for v in estat)
    return acceleracio_nucli(x, y, vx, vy, p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K)


def alcada(x, y, p: Parametres):
    """Retorna (s, d): del pivot a la massa, i de la massa al pla dels imants."""
    s = np.sqrt(p.L**2 - np.asarray(x, dtype=float) ** 2 - np.asarray(y, dtype=float) ** 2)
    return s, p.d0 + p.L - s


def dins_de_labast(x, y, p: Parametres):
    """El fil te llargada fixa: els punts amb x^2+y^2 >= L^2 NO EXISTEIXEN.

    S'han de rebutjar abans d'integrar-los, o l'arrel de s es torna imaginaria.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * x + y * y < p.L**2


def energia(estat, p: Parametres) -> float:
    """Energia mecanica PER UNITAT DE MASSA (m^2/s^2).

        E = 1/2*(vx^2 + vy^2) + g*(L - s) + SUM_i U_i

        U_i = -K_i/((n-1)*delta_i^(n-1))      (K_i*ln(delta_i) si n = 1)

    El segon terme es g per l'alcada real de la massa sobre el punt mes baix, i
    el tercer, el pou de cada imant a la distancia on es de debo la massa.

    ES EXACTAMENT CONSTANT sense friccio, AMB ELS IMANTS ENCESOS I APAGATS: es
    la quantitat que valida magnetic/valida.py (proves 3 i 3b).

    FINS AL 25/10/2026 NO HO ERA amb els imants encesos: el model nomes es
    quedava la component horitzontal de l'imant, i la vertical feia treball
    quan la massa pujava sense que ningu el comptes. Era un error del model, no
    una aproximacio; vegeu acceleracio_nucli i el docstring d'aquest modul.
    """
    x, y, vx, vy = (float(v) for v in estat)
    s = math.sqrt(p.L * p.L - x * x - y * y)
    d = p.d0 + p.L - s
    E = 0.5 * (vx * vx + vy * vy) + p.g * (p.L - s)
    for xi, yi, K in zip(p.im_x, p.im_y, p.im_K):
        delta = math.sqrt((xi - x) ** 2 + (yi - y) ** 2 + d * d)
        if p.n == 1.0:
            E += K * math.log(delta)
        else:
            E += -K / ((p.n - 1.0) * delta ** (p.n - 1.0))
    return E


def periode_petites_oscillacions(p: Parametres) -> float:
    """T0 = 2*pi*sqrt(L/g). El numero de control del CLAUDE.md: 1.07 s."""
    return 2.0 * math.pi * math.sqrt(p.L / p.g)


# ---------------------------------------------------------------------------
# El centre: atractor o no?
# ---------------------------------------------------------------------------
def rigidesa_al_centre(p: Parametres):
    """Linealitzacio de l'acceleracio a l'origen. Retorna (J, M).

    A prop de l'origen i sense friccio, (ax, ay) = J @ (x, y) amb

        J = -(g/L)*I + M
        M = SUM_i K_i*[ -(1 + d0/L)*I/D^(n+1) + (n+1)*pi*pi^T/D^(n+3) ]

    i D_i = sqrt(R_i^2 + d0^2), la distancia de la massa en repos a l'imant i.
    La gravetat estira cap ENDINS i els imants, sumats, cap ENFORA.

    El d0/L es l'estirada vertical dels imants (25/10/2026): al centre fa de
    gravetat de mes, -(g + F0)*x/L amb F0 = SUM_i K_i*d0/D_i^(n+1), i es fica a
    M perque M sigui TOT el que fan els imants. Aixi factor_critic no canvia de
    forma: les K escalen F0 igual que la resta de M.

    El centre es un ATRACTOR si els dos valors propis de J son negatius.

    Aixo no es cap detall d'implementacio: que el centre atregui o no depen de
    la forca dels imants, i es un dels resultats del treball. Si guanyen els
    imants, la regio negra del mapa no existeix, i que existeixi ja diu alguna
    cosa del muntatge.
    """
    M = np.zeros((2, 2))
    for i in range(len(p.im_K)):
        pi = np.array([p.im_x[i], p.im_y[i]])
        D = math.sqrt(pi[0] ** 2 + pi[1] ** 2 + p.d0**2)
        M += p.im_K[i] * (
            -np.eye(2) * (1.0 + p.d0 / p.L) / D ** (p.n + 1.0)
            + (p.n + 1.0) * np.outer(pi, pi) / D ** (p.n + 3.0)
        )
    return -(p.g / p.L) * np.eye(2) + M, M


# Acceleracio maxima que pot quedar en un punt perque compti com a repos, en
# m/s2. Les del muntatge son d'1 a 10 m/s2: aixo es molt per sobre del soroll de
# la coma flotant (1e-14) i molt per sota d'un fals equilibri (6e-3).
RESIDU_REPOS = 1e-9


def punts_dequilibri(p: Parametres):
    """On reposa la massa de debo. Retorna (eq_x, eq_y, centre_estable).

    LA MASSA NO REPOSA A SOBRE DE L'IMANT. Encara que hi sigui a tocar, la
    gravetat continua estirant cap al centre, i el punt de repos queda uns
    quants mil.limetres CAP ENDINS de l'imant, alla on les dues forces
    s'igualen. Amb els valors d'ara son 9,34 mm: l'imant es a 5,00 cm del
    centre i la massa s'atura a 4,07 cm. (Eren 5,85 mm abans del 25/10/2026,
    quan l'estirada vertical dels imants encara no era al model.)

    Aixo no es cap curiositat. El criteri d'aturada del mapa mesura distancies
    A AQUESTS PUNTS i no als imants; fer-ho als imants classifica malament les
    trajectories que hi passen a prop sense quedar-s'hi.

    L'ultim element de eq_x/eq_y es SEMPRE el centre. Si el centre es
    inestable (els imants guanyen la gravetat), centre_estable es fals i
    aquell desti simplement no existeix al mapa: no hi haura regio negra.

    Es resol amb scipy, o sigui que aixo NO es pot compilar amb numba. Tant
    se val: es calcula un cop, abans de la graella, i al bucle dels pixels
    nomes hi entren els numeros que en surten.
    """
    from scipy.optimize import fsolve

    def sense_velocitat(u):
        return np.array(acceleracio([u[0], u[1], 0.0, 0.0], p))

    eq_x, eq_y = [], []
    for i in range(len(p.im_K)):
        if p.im_K[i] == 0.0:
            # Imant apagat: no hi ha cap punt de repos seu. Es marca amb NaN i
            # destins() se'l salta. Passa amb Parametres.sense_imants().
            eq_x.append(math.nan)
            eq_y.append(math.nan)
            continue
        llavor = [p.im_x[i] * 0.88, p.im_y[i] * 0.88]
        sol = fsolve(sense_velocitat, llavor)
        # EL RESIDU, NO NOMES LA DISTANCIA (25/10/2026). Si l'imant no rete la
        # massa, fsolve no convergeix pero retorna igualment el darrer punt, i
        # podia caure a prop de l'imant: amb K = 5e-5 es quedava a 2,94 cm amb
        # 6e-3 m/s2 d'acceleracio i passava per punt de repos. El JavaScript,
        # que busca un canvi de signe, ja deia que no n'hi havia.
        if (math.hypot(sol[0] - p.im_x[i], sol[1] - p.im_y[i]) > p.R / 2
                or float(np.max(np.abs(sense_velocitat(sol)))) > RESIDU_REPOS):
            raise RuntimeError(
                f"No s'ha trobat cap punt de repos a prop de l'imant {p.noms[i]}. "
                "Amb aquesta K l'imant no arriba a retenir la massa."
            )
        eq_x.append(float(sol[0]))
        eq_y.append(float(sol[1]))

    # El centre es exactament (0, 0) i fsolve s'hi encalla avisant que «no fa
    # progres»; l'avis no vol dir res, i el que mana es el residu de sota.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = fsolve(sense_velocitat, [1e-6, 1e-6])
    if float(np.max(np.abs(sense_velocitat(sol)))) > RESIDU_REPOS:
        raise RuntimeError("El centre no ha sortit com a punt d'equilibri: no hauria de passar mai.")
    eq_x.append(float(sol[0]))
    eq_y.append(float(sol[1]))

    jota, _ = rigidesa_al_centre(p)
    estable = bool(np.all(np.linalg.eigvalsh(jota) < 0))

    return np.array(eq_x), np.array(eq_y), estable


def periode_local(p: Parametres, x: float, y: float) -> float:
    """Periode de les oscil.lacions petites al voltant d'un punt d'equilibri.

    Serveix per dimensionar la finestra de confirmacio del criteri d'aturada:
    per dir que una massa s'ha quedat en un lloc, cal mirar-la-hi durant mes
    d'una oscil.lacio local. Es calcula per diferencies finites.
    """
    h = 1e-6
    jota = np.zeros((2, 2))
    for j in range(2):
        du = np.zeros(2)
        du[j] = h
        mes = np.array(acceleracio([x + du[0], y + du[1], 0.0, 0.0], p))
        menys = np.array(acceleracio([x - du[0], y - du[1], 0.0, 0.0], p))
        jota[:, j] = (mes - menys) / (2 * h)
    vp = np.linalg.eigvals(jota).real
    return math.inf if np.max(vp) >= 0 else 2 * math.pi / math.sqrt(-np.min(vp))


def factor_critic(p: Parametres) -> float:
    """Per quant es podrien multiplicar TOTES les K abans que el centre deixes
    de ser atractor. Menys d'1 vol dir que ja no ho es."""
    _, M = rigidesa_al_centre(p)
    mu = float(np.max(np.linalg.eigvalsh(M)))
    return math.inf if mu <= 0.0 else (p.g / p.L) / mu


# ---------------------------------------------------------------------------
# Comprovacio del modul: python -m magnetic.fisica
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    # Les banderes son OPCIONALS i sense elles aquesta ordre fa exactament el que
    # feia: es la comprovacio del modul i ha de continuar dient OK. Amb elles
    # passa a servir tambe per preguntar "amb aquesta K, el centre encara es
    # atractor?" en un segon, en comptes d'haver de calcular un mapa sencer.
    ap = argparse.ArgumentParser(
        description="Comprovacio del modul d'equacions del pendol magnetic."
    )
    afegeix_banderes_de_parametres(ap)
    args = ap.parse_args(argv)

    p = Parametres.des_de_json()
    p, avis_sobreescrit = sobreescriu(p, b=args.b, n=args.n, K=args.K)

    print("=" * 72)
    print(" magnetic.fisica -- comprovacio del modul d'equacions")
    print("=" * 72)
    if avis_sobreescrit:
        print(avis_sobreescrit)
    avis = avis_inventats(p)
    if avis:
        print(avis)
    print()

    print("Muntatge")
    print(f"  L = {p.L:.4f} m      R = {p.R:.4f} m      d0 = {p.d0:.4f} m")
    print(f"  g = {p.g:.3f} m/s2   b = {p.b:.3f} s-1    n  = {p.n:.2f}")
    for nm, x, y, k in zip(p.noms, p.im_x, p.im_y, p.im_K):
        print(f"  imant {nm}: ({x:+.4f}, {y:+.4f}) m   K = {k:.3e} m^(n+1)/s2")

    w0 = math.sqrt(p.g / p.L)
    T0 = periode_petites_oscillacions(p)
    print("\nNumeros de control (el CLAUDE.md diu 5.86 rad/s i 1.07 s)")
    print(f"  omega0 = sqrt(g/L)   = {w0:.4f} rad/s")
    print(f"  T0     = 2*pi/omega0 = {T0:.4f} s")

    tot_be = True

    # -- 1 -----------------------------------------------------------------
    print("\n1. L'origen es un punt d'equilibri")
    a = acceleracio([0.0, 0.0, 0.0, 0.0], p)
    ok = max(abs(a[0]), abs(a[1])) < 1e-12
    tot_be = tot_be and ok
    print(f"   acceleracio a (0,0) amb els imants encesos: ({a[0]:+.2e}, {a[1]:+.2e}) m/s2")
    print(f"   els tres imants s'han d'anul.lar entre ells             {'OK' if ok else 'MALAMENT'}")

    # -- 2 -----------------------------------------------------------------
    print("\n2. Simetria de 120 graus")
    if len(np.unique(np.round(p.im_K, 15))) != 1:
        print("   les K no son iguals: la simetria no s'ha de complir. No es comprova.")
    else:
        pitjor = 0.0
        c, sn = math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)
        rot = np.array([[c, -sn], [sn, c]])
        for punt in ((0.03, 0.0), (-0.02, 0.05), (0.07, -0.04)):
            a1 = np.array(acceleracio([punt[0], punt[1], 0.0, 0.0], p))
            q = rot @ np.array(punt)
            a2 = np.array(acceleracio([q[0], q[1], 0.0, 0.0], p))
            pitjor = max(pitjor, float(np.max(np.abs(rot @ a1 - a2))))
        ok = pitjor < 1e-12
        tot_be = tot_be and ok
        print("   girar el punt 120 graus ha de girar l'acceleracio 120 graus")
        print(f"   pitjor desviacio: {pitjor:.2e} m/s2                       {'OK' if ok else 'MALAMENT'}")

    # -- 3 -----------------------------------------------------------------
    print("\n3. Signe de la gravetat")
    ps = p.sense_imants()
    ax, _ = acceleracio([0.03, 0.0, 0.0, 0.0], ps)
    ok = ax < 0
    tot_be = tot_be and ok
    print(f"   a (0.03, 0) amb els imants apagats: ax = {ax:+.4f} m/s2")
    print(f"   ha de ser NEGATIVA: la gravetat torna la massa al centre   {'OK' if ok else 'MALAMENT'}")

    # -- 4 -----------------------------------------------------------------
    print("\n4. Gravetat exacta contra l'aproximacio de petits angles")
    print("   r (cm)   -g*r/s (m/s2)   -g*r/L (m/s2)   s'equivoca")
    for r in (0.01, 0.05, 0.10, 0.15):
        s = math.sqrt(p.L**2 - r**2)
        exacta, aprox = p.g * r / s, p.g * r / p.L
        print(f"   {r*100:6.1f}   {exacta:13.5f}   {aprox:13.5f}   {100*(exacta/aprox-1):8.2f} %")
    print("   Per aixo el model fa servir la s i no la L.")
    print("   (Una versio antiga del CLAUDE.md deia 'a 5 cm, un 0,1%'. El numero bo")
    print("    es el d'aquesta taula, que surt de les equacions mateixes.)")

    # -- 5 -----------------------------------------------------------------
    print("\n5. El centre: es atractor?")
    jota, _ = rigidesa_al_centre(p)
    vp = np.linalg.eigvalsh(jota)
    lam = factor_critic(p)
    print(f"   gravetat, cap endins        : {-p.g/p.L:+10.2f} s-2")
    print(f"   imants sumats, cap enfora   : {vp[1] + p.g/p.L:+10.2f} s-2")
    print(f"   valors propis de la suma    : {vp[0]:+10.2f}, {vp[1]:+10.2f} s-2")
    if bool(np.all(vp < 0)):
        print("   -> EL CENTRE ES ATRACTOR: la regio negra del mapa existira.")
        print(f"      Periode d'aquest pou: {2*math.pi/math.sqrt(-vp[1]):.2f} s")
        print(f"      Les K podrien creixer x{lam:.3f} abans que deixes de ser-ho.")
    else:
        print("   -> EL CENTRE ES INESTABLE: no hi haura regio negra al mapa.")
        print(f"      Caldria dividir les K per {1/lam:.3f} perque n'hi hagues.")

    # -- 6 -----------------------------------------------------------------
    print("\n6. Els imants contra la gravetat (magnitud de l'acceleracio)")
    print("   punt                          gravetat    imants   qui mana")
    proves = (
        ("a 1.2 cm de l'imant 1", p.R - 0.012, 0.0),
        ("a mig cami del centre", p.R / 2, 0.0),
        ("a 12 cm, vora del mapa", 0.12, 0.0),
    )
    for etiqueta, x, y in proves:
        r = math.hypot(x, y)
        s = math.sqrt(p.L**2 - r**2)
        grav = p.g * r / s
        ax_t, ay_t = acceleracio([x, y, 0.0, 0.0], p)
        ax_g, ay_g = acceleracio([x, y, 0.0, 0.0], p.sense_imants())
        imants = math.hypot(ax_t - ax_g, ay_t - ay_g)
        print(f"   {etiqueta:26} {grav:9.3f} {imants:9.3f}   {'imants' if imants > grav else 'gravetat'}")

    print("\n" + "-" * 72)
    if tot_be:
        print("OK -> el modul d'equacions es coherent.")
    else:
        print("MALAMENT -> mira les linies marcades.")
    print("     Encara no s'ha integrat res: aixo nomes diu que les equacions")
    print("     tenen el signe, les unitats i les simetries que toquen.")
    return 0 if tot_be else 1


if __name__ == "__main__":
    raise SystemExit(main())
