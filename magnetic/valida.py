"""Validacio numerica del pendol magnetic. La porta que ha de passar tot.

Equival a eines/comprova_energia.py del pendol doble, i te la mateixa rao de
ser: demostrar que el simulador es correcte ABANS de treure'n cap conclusio.
Sense aquesta prova, qualsevol lector pot dir que l'estructura fina del mapa
es error numeric, i tindra rao.

Amb els imants apagats i sense friccio, aquest sistema es un pendol simple:
sabem quant ha de valer el periode i sabem que l'energia s'ha de conservar.
Es barat i enxampa els errors de signe i d'unitats abans que contaminin cap
mapa.

Cinc proves:

  1. Periode a amplitud petita contra 2*pi*sqrt(L/g).   -> unitats i signes
  2. Periode contra una QUADRATURA D'ENERGIA exacta.     -> l'integrador
  3. Deriva de l'energia en 100 s.                       -> el pas dt
  3b. El mateix, amb els imants ENCESOS (des del 25/10).  -> el terme vertical
  4. El mateix calcul amb dt i amb dt/2.                 -> convergencia
  5. Caiguda de l'amplitud amb friccio.                  -> les unitats de b

La prova 2 mereix una explicacio. No comparem contra la formula del pendol
simple, que aquest model NO compleix exactament, sino contra una integral que
surt de la quantitat conservada del model mateix:

    E = 1/2*v^2 + g*(L-s)  constant  =>  T = 4*Integral d'una funcio suau

Aixo dona el periode EXACTE del model, a qualsevol amplitud, sense integrar
les equacions del moviment. Es un oracle independent de l'integrador: si RK4
i la quadratura coincideixen a set xifres, els dos calculen el mateix sistema.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.valida
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk

from magnetic.fisica import (
    Parametres,
    avis_inventats,
    energia,
    periode_petites_oscillacions,
)
from magnetic.motor import DT_PER_DEFECTE, integra

DURADA_PERIODE = 20.0  # s, prou per a una vintena d'oscil.lacions
DURADA_ENERGIA = 100.0  # s


# ---------------------------------------------------------------------------
# Oracles: periodes calculats SENSE integrar les equacions
# ---------------------------------------------------------------------------
def periode_exacte_del_model(a: float, p: Parametres) -> float:
    """Periode d'oscil.lacio pel pla, des del repos a x = a. Exacte per al model.

    De la conservacio de E = 1/2*x'^2 + g*(L-s) surt

        T = 4 * Integral_0^a dx / sqrt(2*g*(s(x) - s(a)))

    que te una singularitat integrable al punt de retorn. El canvi x = a*sin(u)
    la elimina del tot i deixa un integrand suau:

        T = 4 * Integral_0^(pi/2) sqrt( (sqrt(L^2 - a^2 sin^2 u) + sqrt(L^2-a^2)) / (2g) ) du

    Quan a -> 0 l'integrand val sqrt(L/g) i surt 2*pi*sqrt(L/g), com ha de ser.
    """
    L, g = p.L, p.g
    sa = math.sqrt(L * L - a * a)

    def integrand(u):
        return math.sqrt((math.sqrt(L * L - a * a * math.sin(u) ** 2) + sa) / (2.0 * g))

    valor, _ = quad(integrand, 0.0, math.pi / 2, epsabs=1e-13, epsrel=1e-13)
    return 4.0 * valor


def periode_pendol_esferic_real(a: float, p: Parametres) -> float:
    """Periode del pendol simple de debo (integral el.liptica completa).

    NO es el del nostre model: serveix per mesurar quant se n'aparta, que es
    el preu de l'aproximacio declarada al CLAUDE.md (ignorar la dinamica
    vertical del lligam).
    """
    theta0 = math.asin(a / p.L)
    return 4.0 * math.sqrt(p.L / p.g) * float(ellipk(math.sin(theta0 / 2.0) ** 2))


# ---------------------------------------------------------------------------
# Mesures sobre una trajectoria integrada
# ---------------------------------------------------------------------------
def periode_per_creuaments(t, x) -> float:
    """Periode mesurat pels passos per zero, interpolats linealment.

    La interpolacio lineal aqui es molt mes bona del que sembla: al pas per
    zero l'acceleracio tambe val zero, o sigui que x(t) no te curvatura i la
    recta hi encaixa a tercer ordre.
    """
    signes = np.signbit(x)
    i = np.nonzero(signes[1:] != signes[:-1])[0]
    if len(i) < 3:
        raise RuntimeError("Trajectoria massa curta: no hi ha prou passos per zero.")
    t0 = t[i] + (t[i + 1] - t[i]) * (-x[i]) / (x[i + 1] - x[i])
    return 2.0 * (t0[-1] - t0[0]) / (len(t0) - 1)


def amplituds_successives(t, x, vx):
    """(temps, amplitud) a cada punt de retorn, on la velocitat canvia de signe."""
    signes = np.signbit(vx)
    i = np.nonzero(signes[1:] != signes[:-1])[0]
    return t[i], np.abs(x[i])


def _oscil_la(p: Parametres, a: float, durada: float, dt: float):
    """Deixa anar la massa des de (a, 0) en repos. Retorna (t, estats)."""
    return integra(p, [a, 0.0, 0.0, 0.0], durada, dt)


# ---------------------------------------------------------------------------
def main() -> int:
    p_real = Parametres.des_de_json()
    p = p_real.sense_imants().sense_friccio()  # pendol simple: el cas que sabem resoldre

    print("=" * 72)
    print(" magnetic.valida -- validacio numerica de l'integrador")
    print("=" * 72)
    avis = avis_inventats(p_real)
    if avis:
        print(avis)
        print("   (b, n i K no intervenen a les proves 1-4: aqui els imants estan")
        print("    apagats. A la prova 5 hi entra nomes la b.)")
    print()

    T0 = periode_petites_oscillacions(p)
    print(f"Muntatge: L = {p.L:.4f} m, g = {p.g:.3f} m/s2, dt = {DT_PER_DEFECTE} s")
    print(f"Periode teoric d'oscil.lacions petites: T0 = {T0:.5f} s")
    print("Imants APAGATS i friccio ZERO a les proves 1-4.")

    tot_be = True

    # -- 1 -----------------------------------------------------------------
    print("\n1. Periode a amplitud petita contra 2*pi*sqrt(L/g)")
    a = 0.005
    t, y = _oscil_la(p, a, DURADA_PERIODE, DT_PER_DEFECTE)
    T_mes = periode_per_creuaments(t, y[0])
    err = abs(T_mes - T0) / T0
    ok = err < 1e-3
    tot_be = tot_be and ok
    print(f"   amplitud {a*1000:.0f} mm  ->  T = {T_mes:.6f} s   (teoric {T0:.6f} s)")
    print(f"   error relatiu {err:.2e}, llindar 1e-3                       {'OK' if ok else 'MALAMENT'}")
    print("   Aixo es el numero de control del CLAUDE.md: 1.07 s.")

    # -- 2 -----------------------------------------------------------------
    print("\n2. Periode contra la quadratura d'energia (oracle independent)")
    print("   amplitud    RK4 (s)    quadratura (s)   error relatiu")
    pitjor = 0.0
    for a in (0.01, 0.05, 0.10, 0.15):
        t, y = _oscil_la(p, a, DURADA_PERIODE, DT_PER_DEFECTE)
        T_mes = periode_per_creuaments(t, y[0])
        T_exa = periode_exacte_del_model(a, p)
        e = abs(T_mes - T_exa) / T_exa
        pitjor = max(pitjor, e)
        print(f"   {a*100:6.1f} cm  {T_mes:9.6f}   {T_exa:14.6f}   {e:13.2e}")
    ok = pitjor < 1e-5
    tot_be = tot_be and ok
    print(f"   pitjor error {pitjor:.2e}, llindar 1e-5                     {'OK' if ok else 'MALAMENT'}")
    print("   L'integrador i la quadratura resolen el mateix sistema.")

    # -- 3 -----------------------------------------------------------------
    print(f"\n3. Conservacio de l'energia en {DURADA_ENERGIA:.0f} s")
    print("   amplitud    E0 (m2/s2)    deriva maxima    relativa")
    pitjor = 0.0
    for a in (0.01, 0.05, 0.12):
        t, y = _oscil_la(p, a, DURADA_ENERGIA, DT_PER_DEFECTE)
        E = np.array([energia(y[:, k], p) for k in range(0, y.shape[1], 10)])
        deriva = float(np.max(np.abs(E - E[0])))
        rel = deriva / E[0]
        pitjor = max(pitjor, rel)
        print(f"   {a*100:6.1f} cm  {E[0]:12.6f}   {deriva:14.3e}   {rel:9.2e}")
    ok = pitjor < 1e-6
    tot_be = tot_be and ok
    print(f"   pitjor {pitjor:.2e}, llindar 1e-6                           {'OK' if ok else 'MALAMENT'}")

    # -- 3b ----------------------------------------------------------------
    # Des del 25/10/2026 l'estirada vertical dels imants es al model, i amb
    # ella l'energia es conserva TAMBE AMB ELS IMANTS ENCESOS. Abans no es
    # podia provar. El criteri es el de la prova 3, escrit abans de mirar, pero
    # dividint per l'energia cinetica maxima del tir i no per E0: amb els
    # imants, E0 pot valer gairebe zero i la relativa no voldria dir res.
    pi_ = p_real.sense_friccio()
    print(f"\n3b. Conservacio de l'energia en {DURADA_ENERGIA:.0f} s, AMB ELS IMANTS ENCESOS")
    print("   sortida (cm)      Ecin max (m2/s2)   deriva maxima   relativa")
    pitjor = 0.0
    for x0, y0 in ((0.03, 0.0), (-0.02, 0.05), (0.08, -0.06)):
        t, y = integra(pi_, [x0, y0, 0.0, 0.0], DURADA_ENERGIA, DT_PER_DEFECTE)
        idx = range(0, y.shape[1], 10)
        E = np.array([energia(y[:, k], pi_) for k in idx])
        ecin = float(np.max(0.5 * (y[2] ** 2 + y[3] ** 2)))
        deriva = float(np.max(np.abs(E - E[0])))
        rel = deriva / ecin
        pitjor = max(pitjor, rel)
        print(f"   ({x0*100:+5.1f}, {y0*100:+5.1f})    {ecin:14.6f}   {deriva:14.3e}   {rel:9.2e}")
    ok = pitjor < 1e-6
    tot_be = tot_be and ok
    print(f"   pitjor {pitjor:.2e}, llindar 1e-6                           {'OK' if ok else 'MALAMENT'}")

    # -- 4 -----------------------------------------------------------------
    print("\n4. El pas dt: el mateix calcul amb dt i amb dt/2")
    a = 0.12
    t1, y1 = _oscil_la(p, a, DURADA_PERIODE, DT_PER_DEFECTE)
    t2, y2 = _oscil_la(p, a, DURADA_PERIODE, DT_PER_DEFECTE / 2)
    T1 = periode_per_creuaments(t1, y1[0])
    T2 = periode_per_creuaments(t2, y2[0])
    canvi = abs(T1 - T2) / T2
    ok = canvi < 1e-6
    tot_be = tot_be and ok
    print(f"   dt = {DT_PER_DEFECTE:.5f} s  ->  T = {T1:.8f} s")
    print(f"   dt = {DT_PER_DEFECTE/2:.5f} s  ->  T = {T2:.8f} s")
    print(f"   canvi relatiu {canvi:.2e}, llindar 1e-6                     {'OK' if ok else 'MALAMENT'}")
    print("   Aixo NOMES valida el dt lluny dels imants. Prop d'un imant la")
    print("   forca es dispara, i alla la comprovacio s'haura de refer sobre")
    print("   el mapa mateix (bloc 10).")

    # -- 5 -----------------------------------------------------------------
    print("\n5. Les unitats de la friccio (imants apagats, friccio ENCESA)")
    pf = p_real.sense_imants()
    t, y = _oscil_la(pf, 0.05, 30.0, DT_PER_DEFECTE)
    tp, ap = amplituds_successives(t, y[0], y[2])
    tall = ap > 0.005  # nomes on encara hi ha amplitud de debo
    pendent, _ = np.polyfit(tp[tall], np.log(ap[tall]), 1)
    b_mesurada = -2.0 * pendent
    err = abs(b_mesurada - pf.b) / pf.b
    ok = err < 0.05
    tot_be = tot_be and ok
    print(f"   b del fitxer de parametres : {pf.b:.4f} s-1")
    print(f"   b llegida de la caiguda    : {b_mesurada:.4f} s-1   (error {err*100:.2f} %)")
    print(f"   llindar 5 %                                              {'OK' if ok else 'MALAMENT'}")
    print(f"   zeta = b/(2*omega0) = {pf.b/(2*math.sqrt(pf.g/pf.L)):.4f}: molt lluny de")
    print("   l'amortiment critic, com ha de ser. Si algu hagues interpretat la b")
    print(f"   com a coeficient de forca, el ritme seria b/m = {pf.b/pf.m:.0f} s-1 i el")
    print("   pendol moriria sense fer ni una oscil.lacio.")

    # -- informatiu --------------------------------------------------------
    print("\n   --- Informatiu, no es cap prova: quant s'aparta el model ---")
    print("   Del pendol esferic de debo. El model ignora la dinamica vertical")
    print("   del lligam, i aixo li treu inercia: surt mes rigid del compte.")
    print("   amplitud    model (s)   pendol real (s)   diferencia")
    for a in (0.01, 0.05, 0.10, 0.12, 0.15):
        Tm = periode_exacte_del_model(a, p)
        Tr = periode_pendol_esferic_real(a, p)
        print(f"   {a*100:6.1f} cm  {Tm:10.5f}   {Tr:15.5f}   {100*(Tm/Tr-1):8.2f} %")
    print("   A amplitud petita coincideixen. A la vora del mapa (12 cm) el model")
    print("   va un 4 % mes rapid. Es un limit DECLARAT del model, no un error,")
    print("   pero val la pena que aquest numero surti a la memoria.")

    print("\n" + "-" * 72)
    if tot_be:
        print("OK -> l'integrador es correcte. Es pot passar a la part B.")
        print("     Signes, unitats, periode, energia i pas dt, tots comprovats")
        print("     contra resultats que no depenen de l'integrador.")
    else:
        print("ATENCIO -> alguna prova ha fallat. No facis cap mapa fins que")
        print("     estigui resolt: un mapa fet amb l'integrador malament no")
        print("     s'assembla gens a un mapa fet be, pero tampoc no ho sembla.")
    return 0 if tot_be else 1


if __name__ == "__main__":
    raise SystemExit(main())
