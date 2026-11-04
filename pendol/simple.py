r"""El pendol SIMPLE: el control de l'experiment 2 i de l'experiment 4.

ES EL MATEIX MUNTATGE amb l'articulacio bloquejada i els bracos rectes: un
sol cos rigid que gira sobre el pivot de dalt. Es exactament la configuracio
de les preses de decaiment Friccio_1.x. No es la «barra 2 sola» (que seria
I2 i 0,657 s de periode): documentacio/quatre-experiments.html en diu
«primer brac bloquejat», i aquella frase descriu un altre pendol.

Aquest modul es PRIM, com pendol/mesures.py i pendol/caos.py: no importa
matplotlib i no llegeix cap fitxer. Les figures son a eines/control_simple.py.

PER QUE ES EL CONTROL
---------------------
El seu espai de fases es (th, w): DUES dimensions, i el retrat que se'n
dibuixa es el sistema sencer. Dues trajectories no s'hi poden creuar, o sigui
que nomes poden tancar-se o anar a un punt fix: no hi ha lloc per al caos.
Dos tirs des de gairebe el mateix angle se separen LINEALMENT (el periode
depen de l'amplitud i la fase es va desfent a ritme constant), mentre que al
pendol doble se separen EXPONENCIALMENT. Vegeu documentacio/tallar-espai-fases.html,
seccions 01 i 02.

EL MODEL, tot de fisica.Parametres i cap numero de geometria escrit a ma
------------------------------------------------------------------------
Amb th1 = th2 = th i w1 = w2 = w, l'energia de fisica.energia queda

    T = 1/2 * (I1 + I2 + 2*K) * w^2        V = -g * (A + B) * cos(th)

o sigui
    I  = I1 + I2 + 2*K         (2,937e-3 kg m^2)
    Md = A + B                 (massa per distancia al centre de gravetat)
    I * th'' = -g*Md*sin(th) + Q
    omega0 = sqrt(g*Md / I)    EXACTAMENT mesures.omega_zero (8,06 rad/s)

COMPTE AMB LA I. La que es fa servir aqui es la DEL MODEL, I1 + I2 + 2K. La de
`_inercies` a parametres.json (2,971e-3) es la mateixa configuracio mesurada
pel periode, i difereixen un 1,1 %. L'autoprova de eines/balanc_energia.py fa
servir la mesurada, escrita a ma. Son dos numeros per al mateix cos; aqui
mana el model perque el simple ha de ser el doble amb l'articulacio bloquejada.

LA FRICCIO, NOMES SI ES DEMANA, I SEMPRE DE PROVA
-------------------------------------------------
Es fa servir la mateixa fisica.Dissipacio que el pendol doble. Amb
l'articulacio bloquejada w2 - w1 = 0, o sigui que el rodament 2 no hi fa res
i tau_f2 i b2 hi desapareixen sols. Queda

    Q = -tau_f1*tanh(w/eps) - b1*w - C*w*|w|*(k1 + k2)

amb k1 i k2 les integrals de |r|^3 al llarg de cada barra, comptant r des del
pivot 1. L'aire es el MATEIX que el de fisica.parells(): tambe hi tracta les
dues barres com a independents alla on s'encavalquen (de 8,9 a 15 cm), i
l'autoprova (S4) comprova que el parell d'aqui es Q1 + Q2 d'alla.

Els coeficients NO estan decidits: la C i la eps no tenen cap valor per
defecte, i tot el que surti amb friccio porta «DE PROVA» a les metadades.

L'ESTAT AL CONTRACTE DE DADES
-----------------------------
El simple es mou a (th, w), pero tot el projecte compta amb
[theta1, theta2, omega1, omega2]. com_a_doble() el posa com (th, th, w, w), que
es literalment el que es: el doble amb els dos angles iguals. Aixi
mesures.distancia_fases, caos.matriu_ln_d i dades.desa hi funcionen sense
tocar-los. Dos compte:
  * (th, th, w, w) NO es cap solucio de fisica.derivades: sense el parell de
    l'articulacio bloquejada, el doble se'n separaria de seguida.
  * La d de dos simples hi surt multiplicada per sqrt(2), perque cada
    diferencia hi apareix dues vegades. ln(d/d0) no ho nota; ln d si.

    .venv\\Scripts\\python -m pendol.simple --autoprova
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np


def _int_r3(a: float, b: float) -> float:
    """Int |r|^3 dr de a a b. Si el tros passa per sobre del pivot (a < 0 < b),
    cada costat suma, com a fisica._k_barra1."""
    if a < 0 < b:
        return (a**4 + b**4) / 4.0
    return abs(b**4 - a**4) / 4.0


@dataclass(frozen=True)
class Simple:
    """El cos compost rigid. Tot en SI, i tot surt de fisica.Parametres."""

    I: float        # inercia sobre el pivot 1: I1 + I2 + 2K (kg m^2)
    Md: float       # A + B (kg m)
    g: float        # la de parametres.json
    k_aire: float   # Int |r|^3 dr sobre les dues barres, des del pivot 1 (m^4)

    @classmethod
    def des_de_parametres(cls, p) -> "Simple":
        # El mateix ordre d'operacions que mesures.omega_zero: aixi omega0 hi
        # surt identica bit a bit (S1), i no nomes «molt a prop».
        I = p.I1 + p.I2 + 2 * p.K
        Md = p.A + p.B
        # Barra 1: de barra1_de a barra1_a (-3 a 15 cm del pivot 1).
        # Barra 2: amb els bracos rectes, cada tros a u del pivot 2 es a
        # L1 + u del pivot 1, de L1 + barra2_de a L1 + barra2_a.
        k1 = _int_r3(p.barra1_de, p.barra1_a)
        k2 = _int_r3(p.L1 + p.barra2_de, p.L1 + p.barra2_a)
        return cls(I=I, Md=Md, g=p.g, k_aire=k1 + k2)

    @property
    def omega0(self) -> float:
        """sqrt(g*Md/I): la mateixa constant que mesures.omega_zero (S1)."""
        return float(np.sqrt(self.g * self.Md / self.I))

    def com_a_dict(self) -> dict:
        d = asdict(self)
        d["omega0"] = self.omega0
        return d


def parell(th, w, s: Simple, dis) -> float:
    """El parell de friccio sobre el cos (N m). Vegeu el capdamunt del modul.

    tau_f2 i b2 no hi son: amb l'articulacio bloquejada multiplicarien zero.
    """
    Q = -dis.tau_f1 * math.tanh(w / dis.eps) - dis.b1 * w
    if dis.C:
        Q = Q - dis.C * w * abs(w) * s.k_aire
    return Q


def potencia_dissipada(th, w, s: Simple, dis):
    """La potencia que es perd (W, >= 0): -Q*w. Accepta vectors."""
    w = np.asarray(w, dtype=float)
    P = dis.tau_f1 * w * np.tanh(w / dis.eps) + dis.b1 * w**2
    if dis.C:
        P = P + dis.C * np.abs(w) ** 3 * s.k_aire
    return P


def derivades(t: float, estat, s: Simple, dis=None):
    """d(th, w)/dt. Signatura de solve_ivp, com fisica.derivades."""
    th, w = estat
    r = -s.g * s.Md * np.sin(th)
    if dis is not None:
        r = r + parell(th, w, s, dis)
    return [w, r / s.I]


def energia(th, w, s: Simple):
    """Energia mecanica (J), amb el mateix zero que fisica.energia (S4)."""
    return 0.5 * s.I * np.asarray(w) ** 2 - s.g * s.Md * np.cos(th)


def simula(s: Simple, estat_inicial, temps: float | None = None, fps: float = 200.0, *,
           t_eval=None, rtol: float | None = None, atol: float | None = None,
           dis=None) -> tuple[np.ndarray, np.ndarray]:
    """Integra el simple i retorna (t, estats), amb estats de forma (2, N).

    El metode i la tolerancia per defecte son els de pendol/simulador.py
    (DOP853, l'estricta), importats i no copiats.
    """
    from scipy.integrate import solve_ivp

    from pendol import simulador

    rtol = simulador.RTOL if rtol is None else rtol
    atol = simulador.ATOL if atol is None else atol
    if t_eval is None:
        if temps is None:
            raise ValueError("digues o `temps` o `t_eval`")
        t_eval = np.arange(0.0, temps + 0.5 / fps, 1.0 / fps)
    else:
        t_eval = np.asarray(t_eval, dtype=float)
    sol = solve_ivp(derivades, (0.0, float(t_eval[-1])),
                    np.asarray(estat_inicial, dtype=float),
                    args=(s,) if dis is None else (s, dis),
                    method=simulador.METODE, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"La integracio ha fallat: {sol.message}")
    return sol.t, sol.y


def com_a_doble(estats) -> np.ndarray:
    """(2, N) -> (4, N) com (th, th, w, w). Vegeu el capdamunt del modul."""
    th, w = np.asarray(estats, dtype=float)
    return np.vstack([th, th, w, w])


def periode_teoric(theta0: float, omega0: float) -> float:
    """El periode exacte a amplitud theta0 (rad): 4*K(m)/omega0, m = sin^2(theta0/2).

    COMPTE: scipy.special.ellipk(m) vol el PARAMETRE m = k^2, no el modul
    k = sin(theta0/2). Es fa amb ellipkm1(1 - m) = ellipkm1(cos^2(theta0/2)),
    que no perd xifres a prop de 180 graus, on 1 - sin^2 les perdria.
    """
    from scipy.special import ellipkm1

    return 4.0 * float(ellipkm1(math.cos(theta0 / 2.0) ** 2)) / omega0


def metadades(s: Simple, dis=None, firmada: bool = False) -> dict:
    """El que ha d'anar al costat de tot resultat del simple. firmada (28/10): la dis es la friccio firmada."""
    m = {"model": "pendol simple: el doble amb l'articulacio bloquejada i els bracos rectes",
         "simple": s.com_a_dict()}
    if dis is None:
        m["friccio"] = "sense"
    else:
        m["friccio"] = ("FIRMADA el 27/10 (Dissipacio.firmada, el joc del simple)" if firmada
                        else "DE PROVA: els coeficients no estan decidits")
        m["dissipacio"] = dis.com_a_dict()
    return m


# --------------------------------------------------------------- l'autoprova
#
# Criteris fixats a la conversa del 01/10/2026, ABANS d'executar res
# (documentacio/pla-analisi-caos.md, «Pas 4b»). Si un falla, NO es relaxa.

AMPLITUDS_GRAUS = (1.0, 10.0, 45.0, 90.0, 120.0, 150.0, 170.0, 179.0)
TOL_PERIODE = 1e-6       # S2, relatiu
PERIODES_MINIMS = 5      # S2
TEMPS_ENERGIA_S = 200.0  # S3
TOL_ENERGIA = 1e-10      # S3, sobre g*Md
TOL_IDENT_E = 1e-14      # S4, sobre g*Md
TOL_IDENT_Q = 1e-12      # S4, relatiu
TOL_PUJADA = 1e-12       # S5, sobre g*Md
TOL_BALANC = 1e-8        # S5, sobre l'energia inicial des del repos
TEMPS_FRICCIO_S = 900.0  # S5
C_DE_PROVA = 0.00794     # S4 i S5: nomes per a l'autoprova
EPS_DE_PROVA = 1e-3


def autoprova() -> int:
    from dataclasses import replace

    from scipy.integrate import solve_ivp

    from pendol import fisica, simulador
    from pendol.fisica import Dissipacio, Friccio, Parametres
    from pendol.mesures import omega_zero

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json("parametres.json")
    s = Simple.des_de_parametres(p)
    w0 = s.omega0
    gMd = s.g * s.Md
    print("Autoprova de pendol/simple.py (criteris fixats el 01/10/2026)\n")
    print(f"   I = {s.I:.6e} kg m^2   Md = {s.Md:.6e} kg m   g*Md = {gMd:.6f} N m")
    print(f"   k_aire = {s.k_aire:.6e} m^4   omega0 = {w0:.6f} rad/s "
          f"(periode petit {2*math.pi/w0:.6f} s)\n")

    # --- S1 ---------------------------------------------------------------
    print("S1. omega0")
    mira(w0 == omega_zero(p),
         f"omega0 = {w0!r} es bit a bit mesures.omega_zero = {omega_zero(p)!r}")

    # --- S2 ---------------------------------------------------------------
    print("\nS2. El periode contra 4*K(m)/omega0, m = sin^2(theta0/2)")

    def pas_per_zero(t, y, *_):
        return y[0]
    pas_per_zero.direction = 1.0

    for graus in AMPLITUDS_GRAUS:
        th0 = math.radians(graus)
        T_teo = periode_teoric(th0, w0)
        sol = solve_ivp(derivades, (0.0, (PERIODES_MINIMS + 2.5) * T_teo), [th0, 0.0],
                        args=(s,), method=simulador.METODE, rtol=simulador.RTOL,
                        atol=simulador.ATOL, events=pas_per_zero)
        tz = sol.t_events[0]
        T_mes = np.diff(tz)
        if len(T_mes) < PERIODES_MINIMS:
            mira(False, f"{graus:5.1f} graus: nomes {len(T_mes)} periodes")
            continue
        err = float(np.max(np.abs(T_mes / T_teo - 1.0)))
        mira(err < TOL_PERIODE,
             f"{graus:5.1f} graus: T = {T_teo:.9f} s, {len(T_mes)} periodes, "
             f"error maxim {err:.2e} (< {TOL_PERIODE:g})")

    # --- S3 ---------------------------------------------------------------
    print(f"\nS3. Sense friccio l'energia es conserva ({TEMPS_ENERGIA_S:g} s)")
    for graus in AMPLITUDS_GRAUS:
        t, y = simula(s, [math.radians(graus), 0.0], TEMPS_ENERGIA_S)
        E = energia(y[0], y[1], s)
        der = float(np.max(np.abs(E - E[0]))) / gMd
        mira(der < TOL_ENERGIA,
             f"{graus:5.1f} graus: max|E - E0| / (g Md) = {der:.2e} (< {TOL_ENERGIA:g})")

    # --- S4 ---------------------------------------------------------------
    print("\nS4. Les identitats amb el pendol doble, a 1000 estats aleatoris")
    rng = np.random.default_rng(1)
    th = rng.uniform(-math.pi, math.pi, 1000)
    w = rng.uniform(-20.0, 20.0, 1000)
    E_s = energia(th, w, s)
    E_d = fisica.energia([th, th, w, w], p)
    err_E = float(np.max(np.abs(E_s - E_d))) / gMd
    mira(err_E < TOL_IDENT_E,
         f"energia del simple = fisica.energia(th, th, w, w): {err_E:.2e} (< {TOL_IDENT_E:g})")

    # Tots els termes encesos, tambe tau_f2 i b2: han de desapareixer sols.
    dis = Dissipacio.des_de_friccio(Friccio.des_de_json("parametres.json"),
                                    C_DE_PROVA, EPS_DE_PROVA)
    err_Q = 0.0
    for a, b in zip(th, w):
        Qs = parell(a, b, s, dis)
        Q1, Q2 = fisica.parells([a, a, b, b], p, dis)
        err_Q = max(err_Q, abs(Qs - (Q1 + Q2)) / abs(Qs))
    mira(err_Q < TOL_IDENT_Q,
         f"parell del simple = Q1 + Q2 de fisica.parells (tots els termes encesos): "
         f"{err_Q:.2e} (< {TOL_IDENT_Q:g})")

    # --- S5 ---------------------------------------------------------------
    print(f"\nS5. Amb friccio DE PROVA (C = {C_DE_PROVA}, eps = {EPS_DE_PROVA:g}, "
          f"tau_f1 i b1 de parametres.json), des de 150 graus, {TEMPS_FRICCIO_S:g} s")

    def amb_treball(t, y):
        d = derivades(t, y[:2], s, dis)
        return [d[0], d[1], float(potencia_dissipada(y[0], y[1], s, dis))]

    th0 = math.radians(150.0)
    tt = np.arange(0.0, TEMPS_FRICCIO_S + 0.0025, 0.005)
    sol = solve_ivp(amb_treball, (0.0, TEMPS_FRICCIO_S), [th0, 0.0, 0.0],
                    method=simulador.METODE, t_eval=tt, rtol=simulador.RTOL,
                    atol=simulador.ATOL)
    if not sol.success:
        mira(False, f"la integracio amb friccio ha fallat: {sol.message}")
    else:
        E = energia(sol.y[0], sol.y[1], s)
        W = sol.y[2]
        pujada = float(np.max(np.diff(E))) / gMd
        mira(pujada <= TOL_PUJADA,
             f"l'energia no puja mai: increment mes gran {pujada:+.2e} g Md "
             f"(<= {TOL_PUJADA:g})")
        E_rel0 = float(E[0] + gMd)
        bal = float(np.max(np.abs((E[0] - E) - W))) / E_rel0
        mira(bal < TOL_BALANC,
             f"E(0) - E(t) = Int P dt a cada instant: {bal:.2e} (< {TOL_BALANC:g})")
        # Nomes informatiu: quan baixa d'1 grau l'amplitud de cada retorn.
        rets = np.nonzero((sol.y[1][:-1] > 0) != (sol.y[1][1:] > 0))[0]
        amp = np.degrees(np.abs(sol.y[0][rets]))
        sota = np.nonzero(amp < 1.0)[0]
        quan = f"{sol.t[rets[sota[0]]]:.1f} s" if len(sota) else "mai"
        print(f"      (informatiu: l'amplitud de retorn baixa d'1 grau a {quan}; "
              f"al final, {math.degrees(abs(sol.y[0][-1])):.4f} graus i "
              f"w = {sol.y[1][-1]:+.2e} rad/s)")

    # --- S6 ---------------------------------------------------------------
    print("\nS6. Amb tots els coeficients a zero es el conservatiu, bit a bit")
    zero = Dissipacio(tau_f1=0.0, b1=0.0, tau_f2=0.0, b2=0.0, C=0.0, eps=EPS_DE_PROVA)
    _, y_c = simula(s, [th0, 0.0], 20.0)
    _, y_z = simula(s, [th0, 0.0], 20.0, dis=zero)
    mira(np.array_equal(y_c, y_z), "150 graus, 20 s: les dues trajectories son identiques")

    # Que no hi hagi cap numero de geometria escrit a ma: amb un altre muntatge
    # la I i la Md segueixen Parametres, no un valor fix.
    q = replace(p, m1=2 * p.m1, L1=1.1 * p.L1)
    s_q = Simple.des_de_parametres(q)
    mira(s_q.I == q.I1 + q.I2 + 2 * q.K and s_q.Md == q.A + q.B
         and s_q.omega0 == omega_zero(q),
         "amb un altre muntatge, I, Md i omega0 segueixen Parametres")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for x in problemes:
            print(f"  - {x}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> el simple es el doble amb l'articulacio bloquejada, i el seu periode")
    print("     es l'exacte de la integral el.liptica.")
    print("     Limits declarats:")
    print("     - La I es la del model (I1 + I2 + 2K), no la mesurada pel periode.")
    print("     - La friccio de S4 i S5 es DE PROVA: prova el calcul, no els coeficients.")
    print("     - Amb tanh(w/eps), al final el simple s'arrossega fins a th = 0; el de")
    print("       debo s'aturaria dins de +-tau_f1/(g Md), uns 0,06 graus.")
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoprova", action="store_true", help="passa l'autoprova i surt")
    args = ap.parse_args(argv)
    if not args.autoprova:
        ap.error("aquest modul es una biblioteca: de moment nomes te --autoprova")
    return autoprova()


if __name__ == "__main__":
    raise SystemExit(main())
