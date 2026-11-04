r"""Deriva les equacions del pendol doble FISIC amb SymPy i verifica fisica.py.

Fa dues coses, totes dues utils per al treball:

1. DERIVACIO SIMBOLICA. Parteix del lagrangia L = T - V i aplica
   Euler-Lagrange. Escriu les equacions en LaTeX, a punt per enganxar a la
   memoria del TdR. Aixi pots ensenyar d'on surten les equacions en comptes de
   copiar-les d'internet.

2. VERIFICACIO NUMERICA. Avalua les equacions simboliques i les de
   pendol/fisica.py en 500 estats aleatoris i comprova que coincideixen.
   Si algun dia toques fisica.py i t'equivoques en un signe, aixo ho caca.

PER QUE ES CONSTRUEIX DES DE LA CINEMATICA I NO DES DE T I V
------------------------------------------------------------
Seria mes curt copiar aqui la T i la V que hi ha escrites a fisica.py i
derivar-les. No serviria de res: si aquelles expressions tinguessin un error,
la «verificacio» el reproduiria i diria OK.

Per aixo aqui es parteix d'on son els cossos -el centre de gravetat de cada
barra, els cargols- i la T i la V es CONSTRUEIXEN. Si les inercies de
fisica.py estiguessin malament, els dos camins no coincidirien.

LA COMPROVACIO EXTRA, que es de franc
-------------------------------------
Despres de verificar, es mira el LIMIT PUNTUAL: encongint el pendol fisic fins
a tornar-lo el de masses puntuals -els cargols sense massa, cada barra
infinitament prima amb tota la massa a la punta-, les equacions s'han de
reduir a les del model antic. Aixo demostra que el model nou no es una altra
fisica sino el cas general, i que el vell n'era un cas particular.

ELS PARELLS DE FRICCIO (pas 3, 27/09/2026)
-----------------------------------------
El lagrangia es el del pendol sense friccio: la friccio no hi cap, entra com a
parells generalitzats Q_i. La tercera part d'aquest programa els torna a
treure pel TREBALL VIRTUAL, Q_i = Sum F . dr/dth_i, amb els jacobians que
SymPy deriva de les mateixes posicions d'aqui dalt, i els integra al llarg de
les barres amb scipy.quad. Si fisica.parells() s'equivoca en un signe, en la
reaccio del segon rodament sobre el primer brac o en un brac de palanca de
l'aire, aqui peta.

Us:
    .venv\Scripts\python.exe eines\deriva_equacions.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendol.fisica import (Dissipacio, Parametres, derivades, parells,  # noqa: E402
                           potencia_dissipada)


def deriva():
    """El lagrangia i les acceleracions, construits des de la cinematica."""
    t = sp.symbols("t", positive=True)
    m1, d1, R1 = sp.symbols("m_1 d_1 R_1", positive=True)
    ms, rs = sp.symbols("m_s r_s", positive=True)
    m2, d2, R2 = sp.symbols("m_2 d_2 R_2", positive=True)
    L1, g = sp.symbols("L_1 g", positive=True)

    th1 = sp.Function("theta_1")(t)
    th2 = sp.Function("theta_2")(t)
    w1, w2 = sp.diff(th1, t), sp.diff(th2, t)

    # ON SON ELS COSSOS. Origen al pivot 1, y cap amunt: x = r*sin(th),
    # y = -r*cos(th). Son les mateixes formules que pendol/fisica.posicions.
    def penja(r, th):
        return r * sp.sin(th), -r * sp.cos(th)

    x_b1, y_b1 = penja(d1, th1)          # centre de gravetat de la barra 1
    x_s, y_s = penja(rs, th1)            # els cargols, com a massa puntual
    x_p2, y_p2 = penja(L1, th1)          # el pivot 2
    x_b2 = x_p2 + d2 * sp.sin(th2)       # centre de gravetat de la barra 2
    y_b2 = y_p2 - d2 * sp.cos(th2)

    def quadrat_velocitat(x, y):
        return sp.diff(x, t) ** 2 + sp.diff(y, t) ** 2

    # CINETICA: translacio del centre de gravetat + rotacio sobre ell mateix.
    # Les barres son uniformes, o sigui I_cm = (1/12)*m*R^2; els cargols son
    # una massa puntual i no en tenen.
    T = sp.Rational(1, 2) * m1 * quadrat_velocitat(x_b1, y_b1)
    T += sp.Rational(1, 2) * (sp.Rational(1, 12) * m1 * R1**2) * w1**2
    T += sp.Rational(1, 2) * ms * quadrat_velocitat(x_s, y_s)
    T += sp.Rational(1, 2) * m2 * quadrat_velocitat(x_b2, y_b2)
    T += sp.Rational(1, 2) * (sp.Rational(1, 12) * m2 * R2**2) * w2**2

    # POTENCIAL: el pes de cada cos per la seva alcada.
    V = g * (m1 * y_b1 + ms * y_s + m2 * y_b2)

    L = sp.simplify(T - V)

    # Euler-Lagrange sense friccio: d/dt(dL/dq') - dL/dq = 0
    eq1 = sp.simplify(sp.diff(sp.diff(L, w1), t) - sp.diff(L, th1))
    eq2 = sp.simplify(sp.diff(sp.diff(L, w2), t) - sp.diff(L, th2))

    a1, a2 = sp.diff(th1, t, 2), sp.diff(th2, t, 2)
    solucio = sp.solve([eq1, eq2], [a1, a2], dict=True)[0]

    simbols = (th1, th2, w1, w2, m1, d1, R1, ms, rs, m2, d2, R2, L1, g)
    return L, eq1, eq2, solucio[a1], solucio[a2], simbols


def _args(p: Parametres):
    """Els parametres, en l'ordre en que els vol la funcio lambdificada."""
    return (p.m1, p.d1, p.R1, p.ms, p.rs, p.m2, p.d2, p.R2, p.L1, p.g)


def verifica(f_alfa, p: Parametres, quants=500, llavor=0) -> float:
    """Error relatiu maxim entre SymPy i fisica.derivades en estats a l'atzar."""
    rng = np.random.default_rng(llavor)
    pitjor = 0.0
    for _ in range(quants):
        estat = rng.uniform(-np.pi, np.pi, 2).tolist() + rng.uniform(-12, 12, 2).tolist()
        ref = f_alfa(*estat, *_args(p))
        meu = derivades(0.0, estat, p)[2:]
        pitjor = max(pitjor, max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(ref, meu)))
    return pitjor


def parells_simbolics():
    """Els Q_i de la friccio, SIMBOLICS, des de les POSICIONS.

    Sense passar per cap formula de fisica.py. Cada rodament fa un parell sobre
    el seu angle: el 1 sobre th1, el 2 sobre l'angle RELATIU phi = th2 - th1, i
    Q_i = parell * dphi/dth_i. L'aire fa una forca -C*|v|*v per unitat de
    llargada a cada tros de barra, i Q_i = Int F . dr/dth_i, amb els jacobians
    de les posicions.

    Des del 29/09/2026 es la MATEIXA expressio la que va a equacions.tex i la
    que es comprova contra fisica.parells(): el que s'escriu es el que es prova.
    Torna un diccionari amb els simbols, els Q (amb les integrals sense avaluar)
    i, per avaluar-los, la part dels rodaments i els integrands de l'aire.
    """
    th1, th2, w1, w2 = sp.symbols("theta_1 theta_2 omega_1 omega_2", real=True)
    r = sp.Symbol("r", real=True)
    L1, C, eps = sp.symbols("L_1 C varepsilon", positive=True)
    tf1, tf2 = sp.symbols("tau_f1 tau_f2", positive=True)
    b1, b2 = sp.symbols("b_1 b_2", nonnegative=True)
    # d'on a on va cada barra, mesurat des del seu pivot
    lim = {1: sp.symbols("r_{1i} r_{1f}", real=True), 2: sp.symbols("r_{2i} r_{2f}", real=True)}
    q = (th1, th2)
    punts = {
        1: sp.Matrix([r * sp.sin(th1), -r * sp.cos(th1)]),
        2: sp.Matrix([L1 * sp.sin(th1) + r * sp.sin(th2), -L1 * sp.cos(th1) - r * sp.cos(th2)]),
    }
    tau1 = -tf1 * sp.tanh(w1 / eps) - b1 * w1
    tau2 = -tf2 * sp.tanh((w2 - w1) / eps) - b2 * (w2 - w1)
    rodaments = [tau1 * sp.diff(th1, qi) + tau2 * sp.diff(th2 - th1, qi) for qi in q]

    integrands = {1: [], 2: []}
    for barra, pos in punts.items():
        J = [pos.diff(qi) for qi in q]
        v = J[0] * w1 + J[1] * w2              # dr/dt = Sum dr/dth_i * w_i
        modul = sp.sqrt(sp.trigsimp(sp.expand(v.dot(v))))
        for Ji in J:
            integrands[barra].append(-C * modul * sp.trigsimp(sp.expand(v.dot(Ji))))
    # La barra 1 no empeny th2 (el seu jacobia respecte de th2 es zero): la seva
    # integral de Q2 val zero i no s'escriu.
    Q = [rodaments[i] + sum(sp.Integral(integrands[b][i], (r, *lim[b]))
                            for b in (1, 2) if integrands[b][i] != 0)
         for i in range(2)]
    return {"estat": (th1, th2, w1, w2), "r": r, "coef": (L1, tf1, b1, tf2, b2, C, eps),
            "lim": lim, "Q": Q, "rodaments": rodaments, "integrands": integrands}


def parells_pel_treball_virtual(p: Parametres, dis: Dissipacio, simb=None):
    """Avalua els Q de parells_simbolics() en un estat, amb les integrals per quad."""
    simb = simb or parells_simbolics()
    est, r, coef = simb["estat"], simb["r"], simb["coef"]
    f_rod = sp.lambdify((*est, *coef), simb["rodaments"], "math")
    f_int = {b: [sp.lambdify((r, *est, *coef), e, "math") for e in simb["integrands"][b]]
             for b in (1, 2)}
    trams = {1: (p.barra1_de, p.barra1_a), 2: (p.barra2_de, p.barra2_a)}
    valors = (p.L1, dis.tau_f1, dis.b1, dis.tau_f2, dis.b2, dis.C, dis.eps)

    def Q(estat):
        Qs = list(f_rod(*estat, *valors))
        for b, (de, a) in trams.items():
            # A la barra 1, r = 0 es el pivot: |v| hi fa colze.
            talls = [0.0] if b == 1 and de < 0.0 < a else None
            for i in range(2):
                Qs[i] += quad(f_int[b][i], de, a, args=(*estat, *valors), points=talls,
                              epsabs=1e-16, epsrel=1e-12, limit=200)[0]
        return Qs

    return Q


def latex_friccio(simb) -> str:
    """La seccio de equacions.tex amb els parells de friccio."""
    return (
        "\n% --- Afegit el 29/09/2026: els parells generalitzats de friccio (pas 3).\n"
        "% Son els MATEIXOS que eines/deriva_equacions.py comprova contra\n"
        "% pendol/fisica.py. Els coeficients NO estan decidits.\n"
        "%   tau_f = fregament sec (Coulomb, regularitzat amb tanh(w/eps)), b = viscos,\n"
        "%   C = aire per unitat de llargada, sobre la velocitat ABSOLUTA de cada tros.\n"
        "%   r_{1i}..r_{1f}, r_{2i}..r_{2f}: d'on a on va cada barra, des del seu pivot.\n"
        "\\section*{Parells generalitzats de fricci\\'o}\n"
        "\\begin{equation}\n\\frac{d}{dt}\\frac{\\partial L}{\\partial \\dot\\theta_i}"
        " - \\frac{\\partial L}{\\partial \\theta_i} = Q_i\n\\end{equation}\n"
        "\\begin{equation}\nQ_1 = " + sp.latex(simb["Q"][0]) + "\n\\end{equation}\n"
        "\\begin{equation}\nQ_2 = " + sp.latex(simb["Q"][1]) + "\n\\end{equation}\n"
    )


def main() -> int:
    print("Derivant el pendol FISIC amb SymPy (triga uns segons)...\n")
    L, eq1, eq2, alfa1, alfa2, simbols = deriva()

    print("=" * 70)
    print("LAGRANGIA  L = T - V")
    print("=" * 70)
    print(sp.latex(L))
    print("\n" + "=" * 70)
    print("EQUACIONS DE LAGRANGE (= 0)")
    print("=" * 70)
    print(sp.latex(eq1), "= 0\n")
    print(sp.latex(eq2), "= 0")

    simb = parells_simbolics()
    sortida = Path(__file__).resolve().parent.parent / "documentacio" / "equacions.tex"
    sortida.parent.mkdir(exist_ok=True)
    # El que hi havia fins al 29/09 es queda igual byte a byte; la friccio va
    # al final, en una seccio a part.
    sortida.write_text(
        "% Generat per eines/deriva_equacions.py\n"
        "% Pendol doble FISIC: dues barres amb la massa repartida.\n"
        "%   d1, d2 = del pivot al centre de gravetat de la seva barra\n"
        "%   R1, R2 = llargada de cada barra   L1 = del pivot 1 al pivot 2\n"
        "%   m_s, r_s = els cargols que uneixen els bracos, com a massa puntual\n"
        "% Sense friccio: els parells generalitzats son zero.\n"
        "\\section*{Lagrangia}\n\\begin{equation}\nL = " + sp.latex(L) + "\n\\end{equation}\n\n"
        "\\section*{Equacions del moviment}\n\\begin{equation}\n"
        + sp.latex(eq1) + " = 0\n\\end{equation}\n"
        "\\begin{equation}\n" + sp.latex(eq2) + " = 0\n\\end{equation}\n"
        + latex_friccio(simb),
        encoding="utf-8",
    )
    print("\nLaTeX desat a documentacio/equacions.tex")

    f_alfa = sp.lambdify(simbols, [alfa1, alfa2], "numpy")

    # --- Verificacio numerica contra pendol/fisica.py --------------------
    print("\n" + "=" * 70)
    print("VERIFICACIO: SymPy vs pendol/fisica.py")
    print("=" * 70)

    # Parametres INVENTATS a posta, i que no s'assemblen als del banc: la
    # prova no ha de dependre del parametres.json ni acertar per casualitat
    # amb una simetria del muntatge de debo.
    p = Parametres(m1=0.137, d1=0.071, R1=0.203, ms=0.052, rs=0.131,
                   m2=0.211, d2=0.064, R2=0.178, L1=0.144,
                   L_extrem=0.170, r_gomet=0.160, g=9.802)
    pitjor = verifica(f_alfa, p)
    print(f"Error relatiu maxim en 500 estats aleatoris: {pitjor:.3e}")
    ok = pitjor < 1e-10
    print("OK -> fisica.py implementa exactament les equacions derivades."
          if ok else
          "ATENCIO -> fisica.py NO coincideix amb la derivacio. Revisa els signes.")

    # --- El limit puntual ------------------------------------------------
    # Encongint el pendol fisic fins a tornar-lo el de masses puntuals -els
    # cargols sense massa, cada barra infinitament prima amb tota la massa a
    # la punta- les equacions s'han de reduir a les del model antic:
    #   I1 -> (m1+m2)*L1^2   I2 -> m2*L2^2   K -> m2*L1*L2
    # Si hi arriba, el model nou no es una altra fisica: es el cas general.
    print("\n" + "=" * 70)
    print("EL LIMIT PUNTUAL: encongit, ha de donar el model de masses puntuals")
    print("=" * 70)
    eps, Lp1, Lp2, mp1, mp2 = 1e-7, 0.200, 0.200, 0.150, 0.150
    q = Parametres(m1=mp1, d1=Lp1, R1=eps, ms=0.0, rs=Lp1,
                   m2=mp2, d2=Lp2, R2=eps, L1=Lp1,
                   L_extrem=Lp2, r_gomet=Lp2, g=9.802)
    print(f"   I1 = {q.I1:.8e}   contra (m1+m2)*L1^2 = {(mp1+mp2)*Lp1**2:.8e}")
    print(f"   I2 = {q.I2:.8e}   contra m2*L2^2      = {mp2*Lp2**2:.8e}")
    print(f"   K  = {q.K:.8e}   contra m2*L1*L2     = {mp2*Lp1*Lp2:.8e}")

    def puntual(estat):
        """Les acceleracions del model puntual, escrites a part."""
        t1, t2, w1, w2 = estat
        d = t1 - t2
        sin_d, cos_d = np.sin(d), np.cos(d)
        M11, M12, M22 = (mp1 + mp2) * Lp1**2, mp2 * Lp1 * Lp2 * cos_d, mp2 * Lp2**2
        r1 = -mp2 * Lp1 * Lp2 * w2**2 * sin_d - (mp1 + mp2) * 9.802 * Lp1 * np.sin(t1)
        r2 = mp2 * Lp1 * Lp2 * w1**2 * sin_d - mp2 * 9.802 * Lp2 * np.sin(t2)
        det = mp2 * Lp1**2 * Lp2**2 * (mp1 + mp2 * sin_d**2)
        return [(M22 * r1 - M12 * r2) / det, (-M12 * r1 + M11 * r2) / det]

    rng = np.random.default_rng(1)
    pitjor_lim = 0.0
    for _ in range(400):
        estat = rng.uniform(-np.pi, np.pi, 2).tolist() + rng.uniform(-12, 12, 2).tolist()
        a, b = puntual(estat), derivades(0.0, estat, q)[2:]
        pitjor_lim = max(pitjor_lim, max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a, b)))
    print(f"\n   Error relatiu maxim en 400 estats aleatoris: {pitjor_lim:.3e}")
    ok_lim = pitjor_lim < 1e-6
    print("   OK -> el model puntual es un cas particular d'aquest."
          if ok_lim else
          "   ATENCIO -> el limit puntual NO surt. Un dels dos models te un error.")

    # --- Els parells de friccio, pel treball virtual ----------------------
    print("\n" + "=" * 70)
    print("ELS PARELLS DE FRICCIO: fisica.parells() contra el treball virtual")
    print("=" * 70)
    # Coeficients INVENTATS, com els parametres: la prova no ha de dependre
    # del que es decideixi. La eps es gran a posta, perque el tanh no estigui
    # saturat i se'n provi la forma i no nomes el signe.
    dis = Dissipacio(tau_f1=3.1e-4, b1=5.3e-5, tau_f2=2.2e-4, b2=7.7e-6,
                     C=0.0113, eps=0.37)
    Q_ref = parells_pel_treball_virtual(p, dis, simb)   # les del LaTeX
    rng = np.random.default_rng(2)
    pitjor_Q, pitjor_P = 0.0, 0.0
    for _ in range(300):
        estat = rng.uniform(-np.pi, np.pi, 2).tolist() + rng.uniform(-12, 12, 2).tolist()
        a, b = Q_ref(estat), parells(estat, p, dis)
        escala = max(abs(a[0]), abs(a[1]))
        pitjor_Q = max(pitjor_Q, max(abs(x - y) for x, y in zip(a, b)) / escala)
        # I els dos camins de fisica.py entre ells: la potencia que es perd ha
        # de ser exactament -Q.w, que es el que comprova_energia.py fa servir.
        P = float(potencia_dissipada(estat, p, dis))
        Pw = -(b[0] * estat[2] + b[1] * estat[3])
        pitjor_P = max(pitjor_P, abs(P - Pw) / abs(Pw))
    print(f"   Q: error relatiu maxim en 300 estats aleatoris:  {pitjor_Q:.3e}")
    print(f"   potencia_dissipada contra -Q.w:                  {pitjor_P:.3e}")
    ok_Q = pitjor_Q < 1e-8 and pitjor_P < 1e-10
    print("   OK -> els parells son els del treball virtual: signes, reaccio del\n"
          "         segon rodament sobre el primer brac i bracos de palanca de l'aire."
          if ok_Q else
          "   ATENCIO -> fisica.parells() NO coincideix amb el treball virtual.")

    print()
    return 0 if (ok and ok_lim and ok_Q) else 1


if __name__ == "__main__":
    raise SystemExit(main())
