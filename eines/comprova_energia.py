r"""Validacio del simulador: conservacio d'energia i inercies contra el banc.

Son DUES proves diferents i fan falta totes dues.

1. CONSERVACIO. Sense friccio el pendol doble es un sistema conservatiu:
   l'energia mecanica ha de ser CONSTANT. Qualsevol deriva es error del
   integrador. Aixo es una seccio curta pero important del TdR: demostra que
   el simulador es correcte ABANS de treure'n conclusions sobre el caos. Sense
   aquesta prova, un lector pot dir que la divergencia que mesures es error
   numeric.

2. LES INERCIES CONTRA EL PERIODE. I aquesta hi es perque la primera NO POT
   CACAR UN MUNTATGE EQUIVOCAT. Si energia() i derivades() fan servir la
   mateixa I1 dolenta, l'energia es conserva a 1e-12 i la prova diu OK
   igualment: seria un pendol integrat perfectament, pero un altre pendol. I
   n'hi ha un d'equivocat a tocar, dins del mateix parametres.json:
   `_inercies.I1` = 2,971e-3 es EL COS COMPOST i no la I1 del model, que val
   1,92e-3 -un 55 % de diferencia-.

   La xarxa es el periode. Amb l'articulacio bloquejada i els bracos rectes
   les dues barres giren com un sol cos amb I = I1 + I2 + 2*K, i la barra 2
   sola gira amb I2; totes dues coses estan MESURADES a les preses de
   decaiment, i els seus periodes son a dades/friccio_doble/*.json, que es
   versionen. Si el model i el banc discrepen mes del que ja esta declarat,
   alguna massa o alguna distancia esta mal posada.

   Es llegeixen del JSON i no s'escriuen a ma a posta: aixi la prova no pot
   quedar desfasada el dia que es tornin a analitzar les preses. De cada
   configuracio se'n pren la MEDIANA i no la mitjana, perque a la Friccio_2.3
   un sol cim fals li mou el T0 un 2,2 % (vegeu la seccio 12 del document de
   deures, que ho te apuntat com a feina pendent a l'eina de friccio).

3. AMB FRICCIO (pas 3, 27/09/2026), L'ENERGIA HA DE BAIXAR EXACTAMENT EL QUE
   ES DISSIPA. S'integra amb el treball dissipat com a cinquena variable
   d'estat, calculat amb fisica.potencia_dissipada() -que no passa pels
   parells-, i E(t) + W(t) ha de quedar a E0 millor que el 0,5 % de W a tota la
   trajectoria. Si els parells de derivades() tinguessin un signe o un brac de
   palanca malament, l'energia no baixaria el que diu la potencia.
   ELS COEFICIENTS D'AQUESTA PROVA NO ESTAN ADOPTATS: son els rodaments de
   parametres.json i una C i una eps de prova. La prova mira la comptabilitat,
   no els numeros.

   I dues MESURES, que no son criteris: quant canvia el pendol amb la eps del
   tanh, que encara s'ha de triar, i quant se separa de l'exacta la quadratura
   de l'aire que fa servir eines/balanc_energia.py.

Us:
    .venv\Scripts\python.exe eines\comprova_energia.py
    .venv\Scripts\python.exe eines\comprova_energia.py --sense-mesures   (nomes els criteris)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
from pendol.fisica import (Dissipacio, Friccio, Parametres, derivades,  # noqa: E402
                           energia, periode_petites_oscillacions, potencia_dissipada)
from pendol.simulador import ATOL, METODE, RTOL, simula  # noqa: E402

DURADA = 100.0   # s
LLINDAR = 1e-9   # 1 part en 10^9: molt per sota de qualsevol incertesa experimental

# Fins on poden discrepar el model i el periode mesurat. El desacord ja
# declarat a parametres.json es de l'1,1 % al cos compost i del 2,8 % a la
# barra 2 -la barra 2 no es simetrica i el seu I calculat va per sota-, o
# sigui que el 5 % deixa marge sense deixar passar un error de bult. No es un
# ajust: es el desacord que ja esta escrit, arrodonit cap amunt.
LLINDAR_INERCIA = 0.05

PRESES = ARREL / "dades" / "friccio_doble"

# La part 3. VALORS DE PROVA, NO ADOPTATS: la C es la del balanc de 15 preses
# i la eps, una del mig de l'escombrat. Una simulacio amb friccio costa unes
# deu vegades mes que una sense, i per aixo la durada es mes curta.
C_DE_PROVA = 0.00794
EPS_DE_PROVA = 1e-3
DURADA_FRICCIO = 20.0
LLINDAR_BALANC = 5e-3        # el 0,5 % de l'energia dissipada
EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4)


def periodes_del_banc() -> dict[str, float]:
    """La mediana del T0 de cada configuracio, de dades/friccio_doble/*.json.

    Friccio_1.x son les dues barres lligades sobre el pivot de dalt;
    Friccio_2.x, la barra 2 sola sobre el seu. Retorna {} si no hi ha les
    preses, i llavors la comprovacio se salta en comptes de fallar.
    """
    grups: dict[str, list[float]] = {"lligat": [], "barra2": []}
    for cami in sorted(PRESES.glob("Friccio_*.json")):
        quin = "lligat" if cami.stem.startswith("Friccio_1") else "barra2"
        grups[quin].append(json.loads(cami.read_text(encoding="utf-8"))["T0_s"])
    return {k: float(np.median(v)) for k, v in grups.items() if v}


def comprova_inercies(p: Parametres) -> tuple[bool, bool]:
    """El model contra el periode mesurat. Retorna (s'ha pogut fer, passa)."""
    T = periodes_del_banc()
    if not T:
        print(f"No hi ha les preses de decaiment a {PRESES.relative_to(ARREL)}:")
        print("   no puc comparar les inercies amb el banc. NO es un error.")
        return False, True

    # Cada configuracio, amb la seva inercia del model i el seu M*g*d.
    casos = [
        ("les dues barres lligades", "lligat",
         p.I1 + p.I2 + 2 * p.K, p.g * (p.A + p.B)),
        ("la barra 2 sola        ", "barra2",
         p.I2, p.g * p.B),
    ]
    print(f"{'muntatge':<24} | {'T mesurat':>10} | {'I del model':>12} | "
          f"{'I del periode':>13} | {'difereixen':>10}")
    print("-" * 82)
    passa = True
    for nom, clau, I_model, Mgd in casos:
        if clau not in T:
            continue
        I_periode = Mgd * T[clau] ** 2 / (4 * math.pi**2)
        dif = I_model / I_periode - 1.0
        ok = abs(dif) < LLINDAR_INERCIA
        passa &= ok
        print(f"{nom:<24} | {T[clau]:>9.4f} s | {I_model:>12.4e} | "
              f"{I_periode:>13.4e} | {100*dif:>+9.2f} % {'OK' if ok else '!!'}")
    return True, passa


def comprova_conservacio(p: Parametres) -> tuple[bool, float]:
    """Integra sense friccio i mira que l'energia no es mogui."""
    print(f"{'angle inicial':>14} | {'E (J)':>12} | {'deriva max (J)':>15} | {'relativa':>10}")
    print("-" * 62)

    # Escala d'energia del sistema, per no dividir per un E0 proper a zero.
    # Es el M*g*d del conjunt: l'energia que costa posar-lo tot horitzontal.
    escala = p.g * (p.A + p.B)
    pitjor = 0.0
    for graus in (10.0, 90.0, 150.0, 179.0):
        estat0 = np.radians([graus, graus, 0.0, 0.0])
        t, y = simula(p, estat0, DURADA, fps=100.0)

        E = np.array([energia(y[:, i], p) for i in range(y.shape[1])])
        deriva = float(np.max(np.abs(E - E[0])))
        relativa = deriva / escala
        pitjor = max(pitjor, relativa)

        marca = "OK" if relativa < LLINDAR else "!!"
        print(f"{graus:>11.0f} gr | {E[0]:>12.6f} | {deriva:>15.3e} | "
              f"{relativa:>9.2e} {marca}")
    return pitjor < LLINDAR, pitjor


def simula_amb_treball(p: Parametres, dis: Dissipacio, estat0, durada, fps):
    """Integra amb friccio i, alhora, W(t) = Int potencia_dissipada dt.

    El treball va com una cinquena variable d'estat, o sigui que l'integra el
    mateix integrador amb la mateixa tolerancia, i no una regla del trapezi
    sobre la sortida, que a cada inversio de sentit s'equivocaria.
    """
    from scipy.integrate import solve_ivp

    def f(t, y):
        return [*derivades(t, y[:4], p, dis), float(potencia_dissipada(y[:4], p, dis))]

    t_eval = np.arange(0.0, durada + 0.5 / fps, 1.0 / fps)
    s = solve_ivp(f, (0.0, float(t_eval[-1])), [*estat0, 0.0], method=METODE,
                  t_eval=t_eval, rtol=RTOL, atol=ATOL)
    if not s.success:
        raise RuntimeError(s.message)
    return s.t, s.y


def comprova_balanc_de_potencia(p: Parametres, dis: Dissipacio) -> bool:
    """E(t) + W(t) = E0, a millor del 0,5 % de W, a tota la trajectoria."""
    print(f"{'angle inicial':>14} | {'dissipada (mJ)':>14} | {'desquadrament max':>18} | "
          f"{'del dissipat':>12}")
    print("-" * 70)
    passa = True
    for graus in (10.0, 90.0, 150.0, 179.0):
        t, y = simula_amb_treball(p, dis, np.radians([graus, graus, 0.0, 0.0]),
                                  DURADA_FRICCIO, 100.0)
        E = energia(y[:4], p)
        W = y[4]
        desquadrament = float(np.max(np.abs(E + W - E[0])))
        relatiu = desquadrament / W[-1]
        ok = relatiu < LLINDAR_BALANC
        passa &= ok
        print(f"{graus:>11.0f} gr | {1000 * W[-1]:>14.4f} | {desquadrament:>16.2e} J | "
              f"{relatiu:>11.2e} {'OK' if ok else '!!'}")
    return passa


def mesura_epsilon(p: Parametres, fr: Friccio) -> None:
    """Quant canvia el pendol amb la eps del tanh. No es un criteri: la tria."""
    import time

    print(f"{'angle':>6} {'eps (rad/s)':>12} | {'dissipada 20 s':>14} | "
          f"{'vs la mes petita, a 1 s':>24} | {'a 5 s':>9} | {'costa':>7}")
    print("-" * 88)
    for graus in (90.0, 150.0):
        estat0 = np.radians([graus, graus, 0.0, 0.0])
        sortides = {}
        for eps in EPSILONS:
            dis = Dissipacio.des_de_friccio(fr, C_DE_PROVA, eps)
            ara = time.perf_counter()
            t, y = simula(p, estat0, DURADA_FRICCIO, fps=100.0, dis=dis)
            sortides[eps] = (t, y, time.perf_counter() - ara)
        _, y_ref, _ = sortides[min(EPSILONS)]
        for eps in EPSILONS:
            t, y, triga = sortides[eps]
            dissipada = energia(y[:, 0], p) - energia(y[:, -1], p)

            def separacio(segons):
                i = int(round(segons * 100.0))
                return math.degrees(float(np.max(np.abs(y[:2, i] - y_ref[:2, i]))))

            print(f"{graus:>5.0f}o {eps:>12.0e} | {1000 * dissipada:>11.3f} mJ | "
                  f"{separacio(1.0):>21.2e} gr | {separacio(5.0):>6.2f} gr | "
                  f"{triga:>5.1f} s")
        print()


def mesura_quadratura_del_balanc(p: Parametres) -> None:
    """La integral de l'aire de balanc_energia.py contra l'exacta.

    balanc_energia.integral_aire suma v^3 a 40 punts de la barra 2, extrems
    inclosos, i multiplica per l'interval entre punts: aixo dona un pes de 40/39
    al tram. Aqui es compara amb la integral que fa el model sobre una
    trajectoria simulada, mostrejada a 240 fps com una presa. No es corregeix
    res: la C publicada surt d'alla, i si s'hi mou, es decideix a part.
    """
    sys.path.insert(0, str(ARREL / "eines"))
    import balanc_energia

    muntatge = balanc_energia.carrega(ARREL / "parametres.json")
    nomes_aire = Dissipacio(tau_f1=0.0, b1=0.0, tau_f2=0.0, b2=0.0, C=1.0, eps=1.0)
    print(f"{'angle':>6} | {'K del balanc':>13} | {'K exacta':>13} | {'balanc/exacta':>13}"
          f" | {'nomes barra 2':>13}")
    print("-" * 72)
    for graus in (90.0, 150.0):
        t, y = simula(p, np.radians([graus, graus, 0.0, 0.0]), 10.0, fps=240.0)
        K_bal = balanc_energia.integral_aire(t, *y, muntatge)
        K_exacta = float(np.trapezoid(potencia_dissipada(y, p, nomes_aire), t))
        k1 = (abs(p.barra1_de) ** 4 + abs(p.barra1_a) ** 4) / 4.0
        K1 = float(np.trapezoid(k1 * np.abs(y[2]) ** 3, t))
        print(f"{graus:>5.0f}o | {K_bal['K']:>13.5e} | {K_exacta:>13.5e} | "
              f"{K_bal['K'] / K_exacta:>13.4f} | "
              f"{K_bal['K_barra2'] / (K_exacta - K1):>13.4f}")


def main() -> int:
    mesures = "--sense-mesures" not in sys.argv[1:]
    p = Parametres.des_de_json(ARREL / "parametres.json")

    print("=" * 82)
    print(" 1 · LES INERCIES DEL MODEL CONTRA EL PERIODE MESURAT AL BANC")
    print("=" * 82)
    print(f"Model: I1 = {p.I1:.4e}   I2 = {p.I2:.4e}   K = {p.K:.4e}   "
          f"A = {p.A:.5f}   B = {p.B:.6f}\n")
    s_ha_fet, inercies_ok = comprova_inercies(p)
    if s_ha_fet:
        print()
        if inercies_ok:
            print("OK -> el muntatge del model dona els periodes que es mesuren al banc.")
            print("     Compte: aixo NO diu que les equacions siguin bones, diu que les")
            print("     masses i les distancies ho son. Les equacions les prova la part 2")
            print("     i, sobretot, eines/deriva_equacions.py.")
        else:
            print(f"ATENCIO -> el model i el banc discrepen mes del {100*LLINDAR_INERCIA:.0f} %.")
            print("           Mira les masses i les distancies de la seccio 'fisic' de")
            print("           parametres.json abans de fiar-te de cap simulacio.")

    T1, T2 = periode_petites_oscillacions(p)
    print("\n" + "=" * 82)
    print(" 2 · CONSERVACIO DE L'ENERGIA, SENSE FRICCIO")
    print("=" * 82)
    print(f"Periodes dels modes normals (oscil.lacions petites): {T1:.3f} s i {T2:.3f} s")
    print("   (son els del pendol ARTICULAT, i no els de la taula de dalt, que son")
    print("    amb l'articulacio bloquejada. Que no coincideixin es correcte.)")
    print(f"Integrant {DURADA:.0f} s sense friccio...\n")
    conserva_ok, pitjor = comprova_conservacio(p)

    print()
    if conserva_ok:
        exponent = int(-np.floor(np.log10(pitjor)))
        print(f"OK -> l'energia es conserva a millor d'1 part en 10^{exponent} durant {DURADA:.0f} s.")
        print("     L'error numeric queda molt per sota de qualsevol incertesa experimental,")
        print("     aixi que la divergencia que mesuris sera fisica i no numerica.")
    else:
        print("ATENCIO -> hi ha deriva d'energia. Baixa RTOL/ATOL a pendol/simulador.py.")

    fr = Friccio.des_de_json(ARREL / "parametres.json")
    dis = Dissipacio.des_de_friccio(fr, C_DE_PROVA, EPS_DE_PROVA)
    print("\n" + "=" * 82)
    print(" 3 · AMB FRICCIO: L'ENERGIA BAIXA EXACTAMENT EL QUE ES DISSIPA")
    print("=" * 82)
    print(f"Valors de PROVA, no adoptats: rodaments de parametres.json, "
          f"C = {C_DE_PROVA}, eps = {EPS_DE_PROVA:.0e} rad/s")
    print(f"Integrant {DURADA_FRICCIO:.0f} s amb friccio i amb el treball dissipat al costat...\n")
    balanc_ok = comprova_balanc_de_potencia(p, dis)
    print()
    if balanc_ok:
        print(f"OK -> E + W es constant a millor del {100 * LLINDAR_BALANC:.1f} % del que es "
              "dissipa: els parells\n     de derivades() son els que diu la potencia.")
    else:
        print("ATENCIO -> l'energia no baixa el que diu la potencia dissipada.")

    if mesures:
        print("\n" + "=" * 82)
        print(" MESURA (no es criteri): LA eps DEL tanh, que s'ha de triar")
        print("=" * 82)
        print("La separacio es el maxim de |dth1|, |dth2| contra la eps mes petita.\n")
        mesura_epsilon(p, fr)
        print("=" * 82)
        print(" MESURA (no es criteri): LA QUADRATURA DE L'AIRE DE balanc_energia.py")
        print("=" * 82)
        mesura_quadratura_del_balanc(p)
        print("\nSi 'balanc/exacta' no es 1, la C que tanca el balanc en surt dividida")
        print("per aquest factor. No es corregeix aqui: es decideix.")

    print()
    if conserva_ok and inercies_ok and balanc_ok:
        print("OK")
        return 0
    print("NO OK: mira quina de les tres parts ha fallat.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
