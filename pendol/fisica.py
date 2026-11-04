"""Model del pendol doble FISIC: dues barres amb la massa repartida.

NO es un pendol de masses puntuals. No es una tria d'elegancia, es el que hi ha
al banc: la barra 1 pesa 67 g, els cargols que uneixen els dos bracos 46 i la
barra 2, 58. No hi ha cap massa puntual enlloc, i el model puntual no seria una
aproximacio una mica tosca d'aquest muntatge sino un altre muntatge -la seva
inercia sobre el pivot de dalt es 6,3 vegades la de debo-.

CONVENIS (valen a TOT el projecte; si els canvies aqui, canvia'ls a tot arreu)
---------------------------------------------------------------------------
  * theta1, theta2 es mesuren des de la VERTICAL INFERIOR, sempre en RADIANS.
  * Positiu = sentit antihorari, vist des de la posicio de la camera.
  * theta2 es ABSOLUT (respecte la vertical), NO relatiu al primer brac.
  * Origen de coordenades al pivot fix, eix y cap AMUNT.
  * Estat = [theta1, theta2, omega1, omega2], sempre en unitats SI.

EL VOCABULARI, que amb barres no es pot escurcar
------------------------------------------------
Amb barres, «la longitud del brac» tota sola no vol dir res: cal dir si es fins
al centre de gravetat o fins al pivot seguent.

    m1, m2      massa de cada BARRA
    d1, d2      del pivot al CENTRE DE GRAVETAT de la seva barra
    R1, R2      llargada de cada barra
    ms, rs      els cargols que uneixen els dos bracos, com a massa puntual,
                i a quina distancia del pivot 1 son
    L1          del pivot 1 al PIVOT 2, que no es cap distancia a un centre
                de gravetat
    L_extrem    del pivot 2 a la punta de la barra 2
    r_gomet     del pivot 2 al gomet, que es la marca que segueix el Tracker

L'ENERGIA
---------
    I1 = (1/12)*m1*R1^2 + m1*d1^2 + ms*rs^2 + m2*L1^2
    I2 = (1/12)*m2*R2^2 + m2*d2^2
    K  = m2*L1*d2                                   (el terme creuat)
    A  = m1*d1 + ms*rs + m2*L1        B = m2*d2

    T = 1/2*I1*w1^2 + 1/2*I2*w2^2 + K*w1*w2*cos(th2 - th1)
    V = -g * ( A*cos th1 + B*cos th2 )

El terme creuat de la cinetica es el que no existeix al model puntual i el que
fa que les dues barres no es puguin tractar per separat.

LES EQUACIONS, amb c = cos(th2 - th1) i s = sin(th2 - th1)
----------------------------------------------------------
    I1*a1 + K*c*a2 = +K*s*w2^2 - g*A*sin(th1) + Q1
    K*c*a1 + I2*a2 = -K*s*w1^2 - g*B*sin(th2) + Q2

Regla per no perdre mai els signes: el terme centrifug de cada equacio porta el
QUADRAT DE L'ALTRA velocitat i el signe CONTRARI al de l'altra equacio.
Verificat amb SymPy a eines/deriva_equacions.py, que ho torna a derivar del
lagrangia i ho compara amb aquest fitxer en 500 estats aleatoris.

det(M) = I1*I2 - K^2*c^2, i NO POT SER ZERO MAI, per un motiu mes fort que al
model puntual: I1 conte el terme m2*L1^2 i I2 conte m2*d2^2, o sigui que
I1*I2 >= (m2*L1*d2)^2 = K^2 per construccio, i la desigualtat es ESTRICTA
perque les barres pesen. Amb els numeros del banc K^2/(I1*I2) = 0,21: el
determinant no baixa mai del 79 % del seu maxim. Al model puntual, en canvi,
el determinant s'anul.lava quan els bracos s'alineaven si la primera massa era
menyspreable.

TRES COSES QUE ES CONFONEN I MOUEN ELS NUMEROS SENSE AVISAR
-----------------------------------------------------------
1. AQUESTA I1 NO ES LA I1 DE `_inercies` a parametres.json. La d'aqui val
   1,92e-3 i es el coeficient de 1/2*w1^2: la barra 1, mes els cargols, mes
   TOTA la massa de la barra 2 posada al pivot 2. La de `_inercies` val
   2,971e-3 i son LES DUES BARRES LLIGADES RECTES girant com un sol cos sobre
   el pivot de dalt, mesurada pel periode de la presa de decaiment; la fa
   servir eines/friccio_pendol_doble.py. Son dues coses diferents amb el
   mateix nom i confondre-les mou els numeros un 55 %.
   Es lliguen: amb l'articulacio bloquejada i els bracos rectes, th1 = th2 i
   tot gira junt amb I = I1 + I2 + 2*K = 2,937e-3, que es el que
   eines/comprova_energia.py compara contra el periode mesurat.

2. rs I L1 VALEN TOTS DOS 0,120 m, I NO SON EL MATEIX. rs es on cauen els caps
   dels cargols i L1 es del pivot 1 al pivot 2; coincideixen perque els
   cargols son justament al pivot 2. Per aixo A s'escriu amb ms*rs i no amb
   (ms + m2)*L1, encara que avui donin el mateix numero: si algun dia es
   remesuren els cargols -i fins al 9/09/2026 els 12,78 cm eren una hipotesi
   viva-, I1 s'actualitzaria (usa rs) i A no (usa L1), i no petaria res.
   L'energia es continuaria conservant perfectament perque el lagrangia seria
   coherent. Simplement seria un altre pendol.

3. posicions() FA SERVIR r_gomet, no d2 ni L_extrem. d2 (5,1 cm) es el centre
   de gravetat i mana la DINAMICA; r_gomet (13,9 cm) es on hi ha la marca que
   segueix el Tracker i es l'unic punt que existeix alhora al video i a la
   simulacio. Amb d2, l'animacio sortiria amb una barra 2 de 5 cm -fluida,
   versemblant i falsa- i cap numero no ho cacaria.

LA FRICCIO: NOMES SI ES DEMANA (pas 3, 27/09/2026)
-------------------------------------------------
Sense dir res, Q1 i Q2 son zero i el pendol es conservatiu, exactament com
abans del pas 3: les mateixes operacions, el mateix resultat bit a bit. Amb
una Dissipacio, hi entren els cinc termes decidits:

    rodament 1:  -tau_f1*tanh(w1/eps) - b1*w1                    sobre th1
    rodament 2:  tau_r = -tau_f2*tanh(dw/eps) - b2*dw,  dw = w2 - w1
                 +tau_r sobre th2 i -tau_r sobre th1: frega contra el PRIMER
                 BRAC, que li fa de suport, i no contra el terra
    aire:        dF = -C*|v|*v*ds sobre la velocitat ABSOLUTA de cada tros de
                 barra, i Q_i = Int dF . dr/dth_i. No sap on son les
                 articulacions.

ELS COEFICIENTS NO ESTAN DECIDITS, i per aixo la C i la eps no tenen cap valor
per defecte: s'han de donar a ma cada vegada. Els dels rodaments surten de
parametres.json, que es el que fa servir el balanc. Una simulacio amb friccio
voldra dir alguna cosa el dia que en Guillem decideixi els coeficients; fins
llavors, el codi es a punt i prou.

Els parells i la potencia dissipada es calculen per DOS CAMINS: parells() amb
el jacobia de cada tros de barra, i potencia_dissipada() directament amb
|v|^3. eines/comprova_energia.py comprova que l'energia baixa exactament el
que diu la integral de la segona, o sigui que un error de signe o de brac de
palanca als parells no pot passar.

Els coeficients de friccio viuen a la classe Friccio, SEPARATS dels
parametres, i no per ordre sino per seguretat: al parametres.json hi ha dos
`b1` -un d'inventat que valia 2,0e-4 i el mesurat, 8,2e-5- i dos `b2`, amb
factors 2,44 i 31 entre ells. Si compartissin objecte amb el mateix nom,
eines/balanc_energia.py podria acabar fent servir el numero equivocat sense
que ho cantes res, i els seus resultats ja son escrits a la memoria del
treball.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parametres:
    """La geometria i les masses del muntatge, mesurades. Tot en SI.

    Es el que veuen derivades(), energia() i posicions(). No hi ha cap
    coeficient de friccio: vegeu Friccio i la nota del capdamunt del modul.
    """

    m1: float        # massa de la barra 1 (kg)
    d1: float        # pivot 1 -> centre de gravetat de la barra 1 (m)
    R1: float        # llargada de la barra 1 (m)
    ms: float        # massa dels cargols que uneixen els dos bracos (kg)
    rs: float        # pivot 1 -> els cargols (m). Avui igual que L1, i no es
                     # el mateix: vegeu la nota 2 del capdamunt
    m2: float        # massa de la barra 2 (kg)
    d2: float        # pivot 2 -> centre de gravetat de la barra 2 (m)
    R2: float        # llargada de la barra 2 (m)
    L1: float        # pivot 1 -> PIVOT 2 (m)
    L_extrem: float  # pivot 2 -> punta de la barra 2 (m)
    r_gomet: float   # pivot 2 -> el gomet que segueix el Tracker (m)
    g: float = 9.802  # gravetat a Vic (m/s^2); mana el parametres.json

    # --- el que en surt, i que es el que entra a les equacions -------------

    @property
    def I1(self) -> float:
        """Inercia del PRIMER BRAC sobre el pivot 1.

        La barra 1, mes els cargols com a massa puntual, mes tota la massa de
        la barra 2 posada al pivot 2. NO es la I1 de `_inercies`: vegeu la
        nota 1 del capdamunt del modul.
        """
        return ((1 / 12) * self.m1 * self.R1**2 + self.m1 * self.d1**2
                + self.ms * self.rs**2 + self.m2 * self.L1**2)

    @property
    def I2(self) -> float:
        """Inercia de la barra 2 sobre el SEU propi pivot."""
        return (1 / 12) * self.m2 * self.R2**2 + self.m2 * self.d2**2

    @property
    def K(self) -> float:
        """El terme creuat de la cinetica. No existeix al model puntual."""
        return self.m2 * self.L1 * self.d2

    @property
    def A(self) -> float:
        """Coeficient de cos(th1) al potencial. Vegeu la nota 2: va amb rs."""
        return self.m1 * self.d1 + self.ms * self.rs + self.m2 * self.L1

    @property
    def B(self) -> float:
        """Coeficient de cos(th2) al potencial."""
        return self.m2 * self.d2

    # Fins on arriba cada barra, mesurat des del seu pivot. Ho fa servir
    # l'integral de l'aire de eines/balanc_energia.py: la barra 1 se suposa
    # uniforme -el seu centre geometric cau al de gravetat, cosa comprovada
    # amb el calibre el 9/09/2026- i de la 2 se'n sap l'extrem i la llargada.
    @property
    def barra1_de(self) -> float:
        return self.d1 - self.R1 / 2

    @property
    def barra1_a(self) -> float:
        return self.d1 + self.R1 / 2

    @property
    def barra2_de(self) -> float:
        return self.L_extrem - self.R2

    @property
    def barra2_a(self) -> float:
        return self.L_extrem

    @classmethod
    def des_de_json(cls, cami: str | Path = "parametres.json") -> "Parametres":
        """Llegeix la seccio "fisic" de parametres.json, que es la mesurada.

        La `g` NO es alli dins: es a dalt de tot del fitxer, i es llegeix
        explicitament. Si hi faltes, s'avisa en comptes d'agafar el valor per
        defecte del dataclass, que coincidiria per casualitat i el dia que es
        toques la de dalt el simulador i el balanc correrien amb gravetats
        diferents.
        """
        cami = Path(cami)
        dades = json.loads(cami.read_text(encoding="utf-8"))
        if "fisic" not in dades:
            raise SystemExit(
                f"{cami} no te la seccio 'fisic' amb el muntatge mesurat.\n"
                "Les claus de dalt eren el model de masses puntuals i es van "
                "treure el 22/09/2026: aquest model ja no les fa servir."
            )
        if "g" not in dades:
            raise SystemExit(
                f"{cami} no diu la gravetat. Ha d'anar a dalt de tot ('g'), "
                "fora de la seccio 'fisic', que es on la llegeix tothom."
            )
        f = dades["fisic"]
        b1, cg, b2 = f["barra1"], f["cargols"], f["barra2"]
        return cls(
            m1=b1["m"], d1=b1["d_cg"], R1=b1["R"],
            ms=cg["m"], rs=cg["r"],
            m2=b2["m"], d2=b2["d_cg"], R2=b2["R"],
            L1=f["L1"], L_extrem=b2["L_extrem"], r_gomet=b2["r_gomet"],
            g=dades["g"],
        )

    def com_a_dict(self) -> dict:
        """El que va a les metadades d'un tir.

        Les inercies hi van EXPLICITAMENT encara que siguin derivades:
        asdict() nomes veu els camps i no les @property, i sense elles un tir
        desat no es pot tornar a reproduir sabent amb quin pendol es va fer.
        """
        d = asdict(self)
        d["_derivats"] = {
            "I1": self.I1, "I2": self.I2, "K": self.K,
            "A": self.A, "B": self.B,
        }
        return d


@dataclass(frozen=True)
class Friccio:
    """Els coeficients dels dos rodaments i de l'aire.

    NO entren a derivades(): vegeu la nota del capdamunt del modul. Existeix
    perque eines/balanc_energia.py els necessita i ha de poder demanar-ho tot
    a un sol lloc, de manera que el simulador i el balanc siguin literalment
    el mateix pendol.
    """

    tau_f1: float   # fregament sec del rodament 1 (N*m)
    b1: float       # viscos del rodament 1 (N*m*s/rad)
    tau_f2: float   # fregament sec del rodament 2 (N*m)
    b2: float       # viscos del rodament 2 (N*m*s/rad)
    C_llibre: float  # resistencia de l'aire per unitat de llargada, de llibre
    amplada: float  # seccio de la barra: DINS del pla d'oscil.lacio (m)
    gruix: float    # seccio de la barra: el que travessa l'aire (m)

    @classmethod
    def des_de_json(cls, cami: str | Path = "parametres.json") -> "Friccio":
        f = json.loads(Path(cami).read_text(encoding="utf-8"))["fisic"]
        fr, aire, sec = f["friccio"], f["aire"], f["seccio"]
        return cls(
            tau_f1=fr["tau_f1"], b1=fr["b1"], tau_f2=fr["tau_f2"], b2=fr["b2"],
            C_llibre=aire["C_llibre"], amplada=sec["amplada"], gruix=sec["gruix"],
        )

    def com_a_dict(self) -> dict:
        return asdict(self)


# Nodes de Gauss-Legendre per a l'aire de la barra 2. L'integrand es suau
# llevat d'un lloc: quan algun tros de la barra gairebe no es mou, |v| hi fa
# colze. Es va comencar amb 16 nodes creient que n'hi havia prou, i
# eines/deriva_equacions.py, contra scipy.quad, ho va desmentir el 27/09:
# 1,4e-5 d'error als estats on el tros mes lent de la barra 2 va a un 0,2-4 %
# del mes rapid, i 1e-16 a tots els altres. Amb 128 dona 2,2e-9, dins del
# criteri d'1e-8 que hi havia escrit abans de mirar, i costava 30 us en lloc
# de 25: la despesa es de numpy, no dels nodes. Tallar la barra on |v| es
# minima costava vuit vegades mes i no hi arribava. Des del 27/09 parells()
# fa servir nomes cos(th2 - th1) (vegeu _nodes_barra2) i en costa 10.
NODES_AIRE = 128
_GL_X, _GL_W = np.polynomial.legendre.leggauss(NODES_AIRE)


@dataclass(frozen=True)
class Dissipacio:
    """El que dissipa energia DINS de la simulacio. Tot en SI.

    Es una classe a part de Friccio a posta. Friccio es el que llegeix el
    balanc de parametres.json, i la C no hi es: nomes hi ha la de llibre.
    Aqui hi ha els sis numeros que fan falta per integrar, i DOS no tenen cap
    valor per defecte, la C i la eps, perque encara no estan decidits.
    Des del 27/10/2026, la friccio FIRMADA es Dissipacio.firmada(angle).
    """

    tau_f1: float   # fregament sec del rodament 1 (N*m)
    b1: float       # viscos del rodament 1 (N*m*s/rad)
    tau_f2: float   # fregament sec del rodament 2 (N*m)
    b2: float       # viscos del rodament 2 (N*m*s/rad)
    C: float        # aire, per unitat de llargada: dF = C*v^2*ds (kg/m)
    eps: float      # regularitzacio del sec, tanh(w/eps) (rad/s)

    @classmethod
    def des_de_friccio(cls, fr: Friccio, C: float, eps: float) -> "Dissipacio":
        """Els rodaments de parametres.json, i la C i la eps donades a ma."""
        if eps <= 0:
            raise ValueError("la eps del tanh ha de ser positiva")
        return cls(tau_f1=fr.tau_f1, b1=fr.b1, tau_f2=fr.tau_f2, b2=fr.b2,
                   C=float(C), eps=float(eps))

    @classmethod
    def firmada(cls, angle_amollada_graus: float, cami: str | Path = "parametres.json",
                simple: bool = False) -> "Dissipacio":
        """LA FRICCIO FIRMADA el 27/10/2026 (parametres.json, fisic.friccio_firmada;
        documentacio/pla-friccio-firmada.md).

        Del doble: el joc que toca segons l'angle d'amollada del primer brac (45,
        o 90 i 150 graus, amb la tolerancia del bloc). La C es una mitjana
        efectiva de tota la presa, de l'amollada al repos, i per a un altre angle
        cap joc no esta validat: s'atura (ValueError). Del simple (simple=True):
        el seu joc, mesurat als decaiments 1.x, fins a l'amplitud maxima que hi
        van arribar.
        """
        bloc = json.loads(Path(cami).read_text(encoding="utf-8"))["fisic"]["friccio_firmada"]
        a = abs(float(angle_amollada_graus))
        if simple:
            j = bloc["jocs"]["simple"]
            if not 0.0 < a <= j["amplitud_maxima_graus"]:
                raise ValueError(f"la friccio firmada del simple val fins a {j['amplitud_maxima_graus']:g} graus "
                                 f"(els decaiments 1.x), no a {a:g}")
        else:
            tol = bloc["tolerancia_angle_graus"]
            jocs = [j for nom, j in bloc["jocs"].items()
                    if nom != "simple" and any(abs(a - g) <= tol for g in j["angles_graus"])]
            if len(jocs) != 1:
                raise ValueError(f"cap joc de la friccio firmada no esta validat per a un doble amollat des de "
                                 f"{a:g} graus: nomes 45, o 90 i 150, +-{tol:g}")
            j = jocs[0]
        return cls(tau_f1=j["tau_f1"], b1=j["b1"], tau_f2=j["tau_f2"], b2=j["b2"],
                   C=float(j["C"]), eps=float(j["eps"]))

    def com_a_dict(self) -> dict:
        return asdict(self)


@functools.lru_cache(maxsize=16)
def _nodes_barra2(de: float, a: float):
    """(u, pesos, pesos*u) dels nodes de Gauss sobre la barra 2, de `de` a `a`."""
    u = 0.5 * (a - de) * _GL_X + 0.5 * (a + de)
    pes = 0.5 * (a - de) * _GL_W
    return u, pes, pes * u


# L'AIRE DE LA BARRA 2 NOMES DEMANA cos(th2 - th1). Cada tros, a u del pivot 2,
# va a v = L1*w1*t1 + u*w2*t2, amb t1 i t2 les tangents unitaries de cada barra,
# i t1.t2 = cos(th2 - th1) = cd. O sigui:
#     |v|^2 = (L1*w1)^2 + (u*w2)^2 + 2*(L1*w1)*(u*w2)*cd
#     v.t1  = L1*w1 + u*w2*cd          (el que empeny th1, amb brac L1)
#     v.t2  = L1*w1*cd + u*w2          (el que empeny th2, amb brac u)
# Es el mateix que projectar sobre els eixos x, y, pero sense cap sinus ni
# cosinus per node: fins al 27/09 es feia aixi i costava el triple.


def _k_barra1(p: Parametres) -> float:


    """Int |r|^3 dr al llarg de la barra 1, que va de barra1_de a barra1_a."""
    a, b = p.barra1_de, p.barra1_a
    # La barra pot sobresortir per sobre del pivot (a < 0): cada tros suma
    # |r|^3, o sigui que els dos costats es tracten per separat.
    if a < 0 < b:
        return (a**4 + b**4) / 4.0
    return abs(b**4 - a**4) / 4.0


def parells(estat, p: Parametres, dis: Dissipacio):
    """Els parells generalitzats (Q1, Q2) dels rodaments i de l'aire (N*m)."""
    t1, t2, w1, w2 = (float(x) for x in estat)

    # Rodament 1, contra el terra.
    Q1 = -dis.tau_f1 * math.tanh(w1 / dis.eps) - dis.b1 * w1

    # Rodament 2, contra el primer brac: el que frena la barra 2 empeny el 1.
    dw = w2 - w1
    tau_r = -dis.tau_f2 * math.tanh(dw / dis.eps) - dis.b2 * dw
    Q1 = Q1 - tau_r
    Q2 = tau_r

    if dis.C:
        # Barra 1: cada tros va a r*w1 perpendicular a la barra, i el seu
        # brac es r. Tancat: -C*w1*|w1|*Int |r|^3 dr.
        Q1 = Q1 - dis.C * w1 * abs(w1) * _k_barra1(p)

        # Barra 2: vegeu la nota de sobre de _nodes_barra2.
        u, pes, pes_u = _nodes_barra2(p.barra2_de, p.barra2_a)
        a, cd = p.L1 * w1, math.cos(t2 - t1)
        ub = u * w2
        # >= 0 per construccio; el maxim nomes treu el -1e-20 de l'arrodoniment
        v = np.sqrt(np.maximum(a * a + ub * ub + 2.0 * a * cd * ub, 0.0))
        Q1 = Q1 - dis.C * p.L1 * float(np.dot(pes * v, a + ub * cd))
        Q2 = Q2 - dis.C * float(np.dot(pes_u * v, a * cd + ub))

    return Q1, Q2


def potencia_dissipada(estat, p: Parametres, dis: Dissipacio):
    """La potencia que es perd (W, sempre >= 0), per un cami DIFERENT de parells().

    No passa per cap jacobia: cada terme es el seu parell per la seva
    velocitat, i l'aire es C*|v|^3 a cada tros de barra. Accepta vectors
    d'estats (forma (4, N)).
    """
    t1, t2, w1, w2 = (np.asarray(x, dtype=float) for x in estat)
    dw = w2 - w1
    P = (dis.tau_f1 * w1 * np.tanh(w1 / dis.eps) + dis.b1 * w1**2
         + dis.tau_f2 * dw * np.tanh(dw / dis.eps) + dis.b2 * dw**2)
    if dis.C:
        P = P + dis.C * np.abs(w1) ** 3 * _k_barra1(p)
        u, pes, _ = _nodes_barra2(p.barra2_de, p.barra2_a)
        a = (p.L1 * w1)[..., None]
        cd = np.cos(t2 - t1)[..., None]
        ub = u * w2[..., None]
        v2 = np.maximum(a * a + ub * ub + 2.0 * a * cd * ub, 0.0)
        P = P + dis.C * np.sum(pes * v2 * np.sqrt(v2), axis=-1)
    return P


def derivades(t: float, estat, p: Parametres, dis: Dissipacio | None = None):
    """Camp vectorial del sistema: retorna d(estat)/dt.

    Signatura compatible amb scipy.integrate.solve_ivp(fun, ...).
    El parametre `t` no s'usa (sistema autonom) pero cal per a solve_ivp.
    Sense `dis`, el pendol es conservatiu: vegeu el capdamunt del modul.
    """
    t1, t2, w1, w2 = estat
    delta = t2 - t1
    s, c = np.sin(delta), np.cos(delta)

    I1, I2, K = p.I1, p.I2, p.K

    # Parells generalitzats. Sense Dissipacio son zero, i les operacions que
    # segueixen son les mateixes que abans del pas 3: la trajectoria surt
    # identica bit a bit.
    if dis is None:
        Q1 = 0.0
        Q2 = 0.0
    else:
        Q1, Q2 = parells(estat, p, dis)

    # M @ alfa = r, amb M = [[I1, K*c], [K*c, I2]].
    r1 = K * s * w2**2 - p.g * p.A * np.sin(t1) + Q1
    r2 = -K * s * w1**2 - p.g * p.B * np.sin(t2) + Q2

    det = I1 * I2 - (K * c) ** 2  # mai zero: vegeu el capdamunt del modul

    alfa1 = (I2 * r1 - K * c * r2) / det
    alfa2 = (-K * c * r1 + I1 * r2) / det

    return [w1, w2, alfa1, alfa2]


def energia(estat, p: Parametres) -> float:
    """Energia mecanica total (J), amb V = 0 als dos bracos horitzontals.

    Sense friccio aquesta quantitat s'ha de conservar. Es la prova de validesa
    numerica del simulador (veure eines/comprova_energia.py).
    """
    t1, t2, w1, w2 = estat
    T = (0.5 * p.I1 * w1**2 + 0.5 * p.I2 * w2**2
         + p.K * w1 * w2 * np.cos(t2 - t1))
    V = -p.g * (p.A * np.cos(t1) + p.B * np.cos(t2))
    return T + V


def posicions(theta1, theta2, p: Parametres):
    """Angles -> coordenades del PIVOT 2 i del GOMET, en metres.

    Compte amb quin radi es fa servir per al segon punt: ha de ser r_gomet,
    que es la marca que segueix el Tracker i l'unic punt que existeix alhora
    al video i a la simulacio. Vegeu la nota 3 del capdamunt del modul.

    Serveix per animar i, sobretot, per fer el cami invers del que fa
    pendol/tracker.py i comprovar que els convenis de signe quadren.
    """
    x1 = p.L1 * np.sin(theta1)
    y1 = -p.L1 * np.cos(theta1)
    x2 = x1 + p.r_gomet * np.sin(theta2)
    y2 = y1 - p.r_gomet * np.cos(theta2)
    return x1, y1, x2, y2


def periode_petites_oscillacions(p: Parametres) -> tuple[float, float]:
    """Periodes dels dos modes normals per a oscil.lacions petites (s).

    Linealitzant sin(x) ~ x i cos(th2 - th1) ~ 1:
        M = [[I1, K], [K, I2]]        S = diag(g*A, g*B)

    Util com a comprovacio d'ordre de magnitud i per triar la finestra del
    filtre de derivacio de pendol/tracker.py, que ha de ser molt mes curta que
    el periode del brac rapid.

    COMPTE, que enganya: amb el muntatge d'ara dona 0,806 s i 0,487 s, i
    aquests NO son els 0,7833 s i 0,6570 s que es mesuren al banc. Aquells dos
    son el COS COMPOST (les dues barres lligades rectes) i la BARRA 2 SOLA,
    que son dos experiments amb l'articulacio bloquejada; aquests son els
    modes del pendol doble articulat. Si no quadren, no es cap error.
    """
    M = np.array([[p.I1, p.K], [p.K, p.I2]])
    S = np.diag([p.g * p.A, p.g * p.B])
    omegues2 = np.linalg.eigvals(np.linalg.solve(M, S))
    periodes = 2 * np.pi / np.sqrt(np.sort(np.real(omegues2)))
    return float(periodes[0]), float(periodes[1])
