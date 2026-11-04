"""Mesura la b del pendol magnetic pel decrement logaritmic.

Amb els imants RETIRATS el muntatge es un pendol simple, i el ritme amb que
se li mor l'amplitud dona la b que despres decideix quin aspecte te el mapa de
conques. Aquest programa llegeix una presa de calibratge (vegeu
magnetic/tracker.py) i n'extreu quatre numeros, cadascun per un cami diferent:

    b   del pendent de ln|A| contra t. L'equacio es x'' = ... - b*x', o sigui
        que l'amplitud cau com exp(-b*t/2) i el pendent val -b/2.
    T   de l'ajust dels instants dels extrems contra el seu numero d'ordre.
    L   de dues maneres INDEPENDENTS: g*T^2/(4*pi^2) i el radi de la
        circumferencia que ajusta el nuvol de punts. Si no coincideixen, o
        l'escala del video o l'origen del Tracker no son on es pensa.

PER QUE DE x I NO DE L'ANGLE
----------------------------
Perque el model del magnetic es cartesia: l'esmorteiment hi va contra x', no
contra theta'. A 30 graus d'amplitud les dues maneres difereixen un 6,9%, i
mesurar-ho d'una manera per posar-ho en una equacio escrita de l'altra es un
error que no dona cap error.

LES DUES COMPROVACIONS QUE NO ES PODEN SALTAR
---------------------------------------------
1. ln|A| contra t ha de sortir una RECTA. Si es corba, l'esmorteiment no es
   viscos -- l'aire hi posa una part quadratica, i la suspensio tambe pot
   dependre de l'amplitud -- i llavors el que es reporta es una b EFECTIVA per
   a aquell rang d'amplituds, i s'ha de dir a la memoria. Per aixo s'ajusta DUES
   vegades: tot el decaiment i nomes el tram que fara servir el mapa.
2. r, la distancia al pivot, ha de ser PLANA. Si baixa o balla, o la camera no
   era perpendicular, o la massa feia el.lipses, o l'autotracker va saltar.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.decrement dades/net/magnetic_3072.csv

I sense cap dada, per comprovar que l'eina mateixa no menteix:

    .venv\\Scripts\\python.exe -m magnetic.decrement --autoprova

AQUEST PROGRAMA NO TOCA parametres.json. La b mesurada s'hi posa a ma i a
posta: treure-la de _valors_inventats vol dir que tots els mapes de la memoria
s'han de tornar a fer, i aixo es fa un cop i sabent que es fa.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from magnetic import dades, estil
from magnetic.geometria import ajusta_circumferencia, inclinacio_de_l_arc
from magnetic.fisica import Parametres

TRAM_PER_DEFECTE = (15.0, 10.0)  # cm: el rang d'amplituds que fara servir el mapa


# ---------------------------------------------------------------------------
# Els extrems
# ---------------------------------------------------------------------------
def periode_aproximat(t, x) -> float:
    """El periode dominant, de la transformada. Nomes per saber com de lluny
    han d'estar els extrems abans de buscar-los."""
    dt = float(np.median(np.diff(t)))
    senyal = x - np.mean(x)
    espectre = np.abs(np.fft.rfft(senyal * np.hanning(len(senyal))))
    espectre[0] = 0.0
    freq = np.fft.rfftfreq(len(senyal), dt)[int(np.argmax(espectre))]
    if freq <= 0:
        raise SystemExit("No he sabut trobar cap oscil.lacio en aquestes dades.")
    return 1.0 / freq


def soroll_estimat(x) -> float:
    """La desviacio del soroll de posicio, de la segona diferencia del senyal.

    La segona diferencia mata qualsevol cosa que sigui llisa i deixa passar el
    soroll multiplicat per sqrt(6). Del pendol nomes hi queda w^2*A*dt^2, que a
    240 fps es una desena part del soroll d'un pixel i per tant no molesta.
    """
    d2 = np.diff(np.asarray(x, dtype=float), 2)
    return float(np.std(d2) / math.sqrt(6.0))


def extrems(t, x, periode_est=None, ample=None):
    """Els maxims I els minims de x, ajustant una parabola a cada cim.

    Es fan servir els dos costats a posta: son el doble de punts per al mateix
    ajust, i a mes qualsevol asimetria entre els dos costats -- que voldria dir
    que l'origen no es al mig -- es veu de seguida a la figura.

    PER QUE UNA PARABOLA AJUSTADA I NO LA MOSTRA MES ALTA. Llegir el maxim
    d'una sola mostra no dona el maxim de la corba: dona el maxim del SOROLL
    que hi ha a sobre. I aquest error no es simetric, sempre tira cap amunt, o
    sigui que no se'n va fent mitjanes: infla ln|A| i mes com mes petita es
    l'amplitud, que es exactament el final del decaiment. Aixo aplana la recta
    i la b surt PETITA.

    Amb l'amplitud final a 1,7 mm sobre 0,62 mm de soroll, llegir la mostra
    pelada dona una b un 2,7% PETITA. Ajustant una parabola a les mostres del
    voltant del cim -- unes quaranta a 240 fps -- el soroll del vertex queda
    dividit per l'arrel de quaranta i el biaix baixa al 0,1%, sense canviar de
    metode: el que es mesura continua sent el maxim de la corba.

    Els dos numeros no son cap record: els torna a mesurar --autoprova cada
    vegada, i si algun dia deixessin de ser certs es veuria alla.

    ample=0 llegeix la mostra pelada, que es nomes per a aquella comparacio.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    dt = float(np.median(np.diff(t)))
    if periode_est is None:
        periode_est = periode_aproximat(t, x)
    separacio = max(int(0.6 * periode_est / dt), 1)
    # Nomes el tros de dalt del cim: mes enlla la corba ja no es una parabola i
    # ajustar-n'hi una tornaria a esbiaixar, ara per l'altre costat.
    if ample is None:
        ample = max(int(0.08 * periode_est / dt), 1)

    t_k, a_k, signe = [], [], []
    for s in (+1, -1):
        idx, _ = find_peaks(s * x, distance=separacio)
        for i in idx:
            if ample == 0:  # la mostra pelada, nomes per a la comparacio
                t_k.append(t[i])
                a_k.append(x[i])
                signe.append(s)
                continue
            i0, i1 = max(i - ample, 0), min(i + ample + 1, len(x))
            if i1 - i0 < 3:
                continue
            u = t[i0:i1] - t[i]
            c2, c1, c0 = np.polyfit(u, s * x[i0:i1], 2)
            if c2 >= 0:  # no es cap cim: soroll pur
                continue
            vertex = -0.5 * c1 / c2
            if abs(vertex) > ample * dt:  # el cim cau fora de la finestra
                continue
            t_k.append(t[i] + vertex)
            a_k.append(s * (c0 - 0.25 * c1**2 / c2))
            signe.append(s)

    ordre = np.argsort(t_k)
    return (np.asarray(t_k)[ordre], np.asarray(a_k)[ordre], np.asarray(signe)[ordre])


# ---------------------------------------------------------------------------
# Els ajustos
# ---------------------------------------------------------------------------
def ajusta_recta(u, v):
    """Recta per minims quadrats. Retorna (pendent, ordenada, sigma_pendent, R2)."""
    if len(u) < 3:
        raise SystemExit(f"Nomes hi ha {len(u)} punts: no n'hi ha prou per ajustar res.")
    coef, cov = np.polyfit(u, v, 1, cov=True)
    pendent, ordenada = float(coef[0]), float(coef[1])
    residu = v - (pendent * u + ordenada)
    variacio = float(np.sum((v - np.mean(v)) ** 2))
    r2 = 1.0 - float(np.sum(residu**2)) / variacio if variacio > 0 else float("nan")
    return pendent, ordenada, float(math.sqrt(cov[0, 0])), r2


def friccio(t_k, a_k):
    """La b del decaiment de les amplituds. Retorna (b, sigma_b, R2, ordenada)."""
    bons = np.abs(a_k) > 0
    pendent, ordenada, sigma, r2 = ajusta_recta(t_k[bons], np.log(np.abs(a_k[bons])))
    return -2.0 * pendent, 2.0 * sigma, r2, ordenada


def periode(t_k, periode_est=None):
    """El periode, de l'instant de cada extrem contra el seu numero d'ordre.

    Extrems consecutius van a MITJA oscil.lacio l'un de l'altre, d'aqui el 2.
    Es fa amb un ajust i no amb la mitjana de les diferencies perque l'ajust fa
    servir tota la durada de la presa: amb vuitanta oscil.lacions, l'error de
    l'ultim instant queda dividit per vuitanta.

    EL NUMERO D'ORDRE NO ES LA POSICIO A LA LLISTA. Si al final del decaiment
    algun cim s'ha descartat -- perque alla nomes hi havia soroll --, numerar
    els extrems d'un en un els ajuntaria i el periode sortiria llarg. Es
    calcula a quin numero de mig periode cau cada instant, i es refina un cop.
    Ho va enxampar --autoprova: amb 108 extrems en comptes de 110 el periode
    sortia un 0,6% llarg, que amb L = g*T^2/4pi^2 son 3,5 mm de longitud.
    """
    t_k = np.asarray(t_k, dtype=float)
    if len(t_k) < 3:
        raise SystemExit(f"Nomes {len(t_k)} extrems: no es pot mesurar cap periode.")
    # El punt de partida es la MEDIANA de les distancies entre extrems, no
    # l'estimacio de la transformada: amb 40 s de senyal la transformada te una
    # resolucio de l'1,4%, prou per numerar malament algun extrem del final i
    # deixar el periode un 2% curt. La mediana, en canvi, es bona fins al
    # fotograma. periode_est nomes fa d'ultim recurs.
    mig = float(np.median(np.diff(t_k)))
    if not np.isfinite(mig) or mig <= 0:
        mig = periode_est / 2.0
    k = k_ant = None
    for _ in range(6):
        k = np.round((t_k - t_k[0]) / mig)
        if k_ant is not None and np.array_equal(k, k_ant):
            break  # la numeracio ja no es mou: hem convergit
        k_ant = k
        mig, _, sigma, r2 = ajusta_recta(k, t_k)
    return 2.0 * mig, 2.0 * sigma, r2


def longitud_del_periode(T, g):
    return g * T**2 / (4.0 * math.pi**2)


# ---------------------------------------------------------------------------
# L'analisi sencera
# ---------------------------------------------------------------------------
def analitza(t, x, y, g=9.802, tram=TRAM_PER_DEFECTE, ample_cim=None):
    """Tots els numeros. Cap efecte secundari: ni figures, ni fitxers."""
    xc, yc, radi, rms = ajusta_circumferencia(x, y)
    r = np.hypot(x, y)
    periode_est = periode_aproximat(t, x)
    t_k, a_k, signe = extrems(t, x, periode_est, ample=ample_cim)

    # L'AMPLITUD ES MESURA DES DE L'EQUILIBRI, no des del zero de coordenades.
    # Si el zero esta desplacat uns mil.limetres, els maxims surten tots una
    # mica grossos i els minims tots una mica petits, i a ln|A| els extrems es
    # parteixen en DUES families paral.leles que es separen a mesura que
    # l'amplitud minva. Es veu de seguida al panell dels residus, i infla el
    # residu de l'ajust sense que la b n'estigui malament del tot.
    #
    # L'equilibri es el punt mig entre cada extrem i el seguent: aixi el
    # desplacament se'n va exactament, i de propina queda mesurat.
    equilibri = float(np.mean((a_k[:-1] + a_k[1:]) / 2.0)) if len(a_k) > 1 else 0.0
    a_k = a_k - equilibri

    T, sigma_T, r2_T = periode(t_k, periode_est)
    esperats = int(round(2.0 * (t_k[-1] - t_k[0]) / T)) + 1
    b_tot, sigma_b_tot, r2_tot, ordenada = friccio(t_k, a_k)

    alt, baix = max(tram) / 100.0, min(tram) / 100.0
    dins = (np.abs(a_k) <= alt) & (np.abs(a_k) >= baix)
    if dins.sum() >= 3:
        b_tram, sigma_b_tram, r2_tram, _ = friccio(t_k[dins], a_k[dins])
    else:
        b_tram = sigma_b_tram = r2_tram = float("nan")

    return {
        "n_mostres": int(len(t)),
        "durada_s": float(t[-1] - t[0]),
        "n_extrems": int(len(t_k)),
        "equilibri_m": equilibri,
        "n_extrems_esperats": esperats,
        "amplitud_inicial_m": float(abs(a_k[0])) if len(a_k) else float("nan"),
        "amplitud_final_m": float(abs(a_k[-1])) if len(a_k) else float("nan"),
        "periode_s": T,
        "sigma_periode_s": sigma_T,
        "r2_periode": r2_T,
        "L_del_periode_m": longitud_del_periode(T, g),
        "circumferencia": {
            "centre_m": [xc, yc],
            "radi_m": radi,
            "residu_rms_m": rms,
            "inclinacio_de_l_arc_graus": inclinacio_de_l_arc(x, y, xc, yc),
        },
        "r": {
            "mitjana_m": float(np.mean(r)),
            "desviacio_m": float(np.std(r)),
            "min_m": float(np.min(r)),
            "max_m": float(np.max(r)),
            "deriva_m_per_s": float(np.polyfit(t, r, 1)[0]),
        },
        "y_recorregut_m": float(np.max(y) - np.min(y)),
        "b_tot": {"b": b_tot, "sigma": sigma_b_tot, "r2": r2_tot, "n": int(len(t_k))},
        "b_tram": {
            "b": b_tram, "sigma": sigma_b_tram, "r2": r2_tram,
            "n": int(dins.sum()), "tram_cm": list(tram),
        },
        "_extrems": (t_k, a_k, signe, dins, ordenada, b_tot),
    }


def escriu(res: dict, p: Parametres) -> None:
    """El resultat per pantalla, amb el que se'n pot creure i el que no."""
    c = res["circumferencia"]
    r = res["r"]
    cm = 100.0

    print("\nLa geometria (d'on surt la L de debo)")
    print(f"   centre ajustat        ({c['centre_m'][0] * cm:+.2f}, "
          f"{c['centre_m'][1] * cm:+.2f}) cm   <- ha de ser (0, 0) si l'origen es al pivot")
    print(f"   radi ajustat          {c['radi_m'] * cm:.2f} cm")
    print(f"   residu de l'ajust     {c['residu_rms_m'] * 1000:.2f} mm")
    print(f"   inclinacio de l'arc   {c['inclinacio_de_l_arc_graus']:+.3f} graus   "
          "<- ha de ser ~0")
    if abs(c["inclinacio_de_l_arc_graus"]) > 1.0:
        print("   !! l'arc no es simetric respecte de la vertical. Si aquest numero")
        print("      s'assembla al DOBLE de l'angle del .trk, el signe amb que s'ha")
        print("      aplicat es l'altre: torna a passar magnetic.tracker amb")
        print("      --angle-signe -  i compara.")

    print(f"\n   r al llarg de la presa  {r['mitjana_m'] * cm:.2f} +- "
          f"{r['desviacio_m'] * cm:.2f} cm   (de {r['min_m'] * cm:.2f} a {r['max_m'] * cm:.2f})")
    print(f"   deriva de r             {r['deriva_m_per_s'] * 1000:+.4f} mm/s   <- ha de ser ~0")
    print(f"   recorregut vertical     {res['y_recorregut_m'] * cm:.2f} cm")
    print(f"   equilibri de x          {res['equilibri_m'] * 1000:+.2f} mm   "
          "<- des d'aqui es mesuren les amplituds")

    if res["n_extrems"] < res["n_extrems_esperats"]:
        falten = res["n_extrems_esperats"] - res["n_extrems"]
        print("")
        print(f"   {falten} extrems descartats de {res['n_extrems_esperats']}: "
              "alla el cim no es distingia del soroll.")


    print(f"\nEl rellotge del pendol")
    print(f"   periode          T = {res['periode_s']:.5f} +- {res['sigma_periode_s']:.5f} s"
          f"   (R2 = {res['r2_periode']:.8f})")
    print(f"   longitud del periode  L = {res['L_del_periode_m'] * cm:.2f} cm")
    print(f"   longitud del radi     L = {c['radi_m'] * cm:.2f} cm")
    print(f"   longitud del regle    L = {p.L * cm:.2f} cm  (parametres.json)")
    tres = [res["L_del_periode_m"], c["radi_m"], p.L]
    escampall = (max(tres) - min(tres)) / min(tres)
    print(f"   les tres difereixen un {escampall:.1%}")
    if escampall > 0.03:
        print("   !! tres camins cap a la mateixa longitud i no coincideixen. El del")
        print("      PERIODE es el mes fiable dels tres: no depen ni de l'escala del")
        print("      video ni d'on has posat l'origen ni del regle. Si el del radi se")
        print("      n'aparta, sospita de l'escala (la vara de calibracio no era al pla")
        print("      del pendol); si el que se n'aparta es el del regle, es que el fil")
        print("      no comenca ni acaba on el vas mesurar.")

    print(f"\nLa friccio")
    bt, bp = res["b_tot"], res["b_tram"]
    print(f"   tot el decaiment    b = {bt['b']:.5f} +- {bt['sigma']:.5f} s-1   "
          f"(R2 = {bt['r2']:.5f}, {bt['n']} extrems)")
    if math.isnan(bp["b"]):
        print(f"   tram {bp['tram_cm'][0]:.0f}->{bp['tram_cm'][1]:.0f} cm: nomes "
              f"{bp['n']} extrems, no n'hi ha prou per ajustar-hi res")
    else:
        print(f"   tram {bp['tram_cm'][0]:.0f}->{bp['tram_cm'][1]:.0f} cm     "
              f"b = {bp['b']:.5f} +- {bp['sigma']:.5f} s-1   "
              f"(R2 = {bp['r2']:.5f}, {bp['n']} extrems)")
        dif = abs(bp["b"] - bt["b"]) / bt["b"]
        print(f"   difereixen un {dif:.1%}")
        if dif > 0.10:
            print("   !! els dos ajustos NO diuen el mateix: l'esmorteiment no es")
            print("      viscos pur, i b depen de l'amplitud. No ho diu de qui es la")
            print("      culpa: l'aire hi posa una part quadratica, pero en aquesta presa")
            print("      n'explica com a molt un terc (parametres.json, _b_no_es_l_aire).")
            print("      Una b sola s'ha de donar amb la dispersio entre trams com a")
            print("      incertesa, dient de quin rang d'amplituds surt; la decisio presa")
            print("      es a parametres.json (_b_per_que_0.007_i_no_0.0003).")
        else:
            print("   els dos ajustos coincideixen: l'esmorteiment es viscos en aquest rang.")

    print(f"\n   b que hi ha ara a parametres.json: {p.b:.5f} s-1"
          + ("  (INVENTADA)" if "b" in p.inventats else ""))
    print(f"   b mesurada:                        {bt['b']:.5f} s-1")
    print("\n   Aquest programa NO ha tocat parametres.json. Posar-hi la b mesurada")
    print("   vol dir refer tots els mapes de la memoria: es fa a ma i sabent-ho.")


# ---------------------------------------------------------------------------
# La figura
# ---------------------------------------------------------------------------
def dibuixa(res: dict, t, x, y, nom: str, directori: str, mostra: bool) -> None:
    t_k, a_k, signe, dins, ordenada, b_tot = res["_extrems"]
    cm = 100.0
    estil.aplica()
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.4))

    # 1. x(t) amb els extrems trobats
    ax = axs[0, 0]
    ax.plot(t, x * cm, lw=0.7, color=estil.TINTA, alpha=0.8)
    ax.plot(t_k, a_k * cm, "o", ms=3, mfc="none", mew=0.9, color=estil.IMANTS[1])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("x (cm)")
    ax.set_title(f"x(t) i els {len(t_k)} extrems trobats", loc="left")

    # 2. ln|A| contra t, amb els dos ajustos
    ax = axs[0, 1]
    ln = np.log(np.abs(a_k))
    ax.plot(t_k, ln, "o", ms=3.2, mfc="none", mew=0.9, color=estil.TINTA_2,
            label="tots els extrems")
    ax.plot(t_k, ordenada - 0.5 * b_tot * t_k, "-", lw=1.4, color=estil.IMANTS[0],
            label=f"tot: b = {res['b_tot']['b']:.4f}")
    if dins.sum() >= 3 and not math.isnan(res["b_tram"]["b"]):
        ax.plot(t_k[dins], ln[dins], "o", ms=3.2, color=estil.IMANTS[1],
                label=f"tram {res['b_tram']['tram_cm'][0]:.0f}-"
                      f"{res['b_tram']['tram_cm'][1]:.0f} cm")
        pend = -0.5 * res["b_tram"]["b"]
        orde = np.mean(ln[dins]) - pend * np.mean(t_k[dins])
        ax.plot(t_k[dins], orde + pend * t_k[dins], "--", lw=1.4, color=estil.IMANTS[1],
                label=f"tram: b = {res['b_tram']['b']:.4f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("ln |A|   (A en m)")
    ax.set_title("el decrement: ha de ser una RECTA", loc="left")
    ax.legend(loc="upper right")

    # 3. els residus de la recta: aqui es veu si es corba
    ax = axs[1, 0]
    ax.axhline(0, lw=0.8, color=estil.EIX)
    ax.plot(t_k, ln - (ordenada - 0.5 * b_tot * t_k), "o", ms=3, mfc="none",
            mew=0.9, color=estil.TINTA_2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("residu de ln |A|")
    ax.set_title("residus: si dibuixen una U, l'esmorteiment no es viscos", loc="left")

    # 4. r(t): el control de qualitat del seguiment
    ax = axs[1, 1]
    r = np.hypot(x, y)
    ax.plot(t, r * cm, lw=0.7, color=estil.TINTA, alpha=0.8)
    ax.axhline(res["circumferencia"]["radi_m"] * cm, ls=(0, (4, 3)), lw=1.1,
               color=estil.IMANTS[0], label=f"radi ajustat {res['circumferencia']['radi_m']*cm:.2f} cm")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("r (cm)")
    ax.set_title("r ha de ser PLANA (camera perpendicular, res d'el.lipses)", loc="left")
    ax.legend(loc="best")

    fig.suptitle(
        f"Pendol magnetic, friccio de {nom}:  "
        f"b = {res['b_tot']['b']:.4f} s-1,  T = {res['periode_s']:.4f} s,  "
        f"L = {res['L_del_periode_m']*cm:.2f} cm",
        fontsize=11,
    )
    fig.tight_layout()
    if mostra:
        plt.show()
    else:
        estil.desa(fig, nom, directori)


# ---------------------------------------------------------------------------
# L'autoprova: comprovar que l'eina no menteix, sense cap dada
# ---------------------------------------------------------------------------
def autoprova() -> int:
    """Fabrica una oscil.lacio de b i T CONEGUDES i mira si les recupera.

    El senyal es una cosinusoide esmorteida amb soroll d'un pixel a sobre, que
    es el que dona el video. No preten ser un pendol de debo: preten ser un cas
    on la resposta bona es sap, que es l'unica manera de saber si el que
    respon el programa vol dir alguna cosa.
    """
    print("=" * 72)
    print(" magnetic.decrement --autoprova")
    print("=" * 72)

    fps, durada = 239.688, 60.0
    b_bona, T_bona, A0, L = 0.15, 1.08996, 0.15, 0.295
    soroll_m = 0.000618  # un pixel del video de debo

    t = np.arange(0.0, durada, 1.0 / fps)
    rng = np.random.default_rng(20260824)
    x = A0 * np.exp(-0.5 * b_bona * t) * np.cos(2 * math.pi * t / T_bona)
    x = x + rng.uniform(-0.5, 0.5, size=len(t)) * soroll_m
    y = -np.sqrt(np.maximum(L**2 - x**2, 0.0)) + rng.uniform(-0.5, 0.5, size=len(t)) * soroll_m

    print(f"\n   posat:  b = {b_bona:.5f} s-1   T = {T_bona:.5f} s   "
          f"A0 = {A0 * 100:.1f} cm   L = {L * 100:.1f} cm")
    print(f"   {len(t)} mostres a {fps:.3f} fps, amb {soroll_m * 1000:.3f} mm de soroll")

    res = analitza(t, x, y)
    b, T = res["b_tot"]["b"], res["periode_s"]
    radi = res["circumferencia"]["radi_m"]
    print(f"   trobat: b = {b:.5f} s-1   T = {T:.5f} s   "
          f"radi = {radi * 100:.2f} cm   ({res['n_extrems']} extrems)")

    # La comparacio que justifica ajustar una parabola al cim en comptes de
    # llegir la mostra mes alta. Es torna a mesurar cada vegada: aixi el numero
    # que hi ha escrit al docstring de extrems() no pot envellir sense que es
    # noti. Vegeu-hi el per que.
    pelat = analitza(t, x, y, ample_cim=0)["b_tot"]["b"]
    print("")
    print(f"   soroll estimat del senyal: {soroll_estimat(x) * 1000:.4f} mm"
          f"   (posat: {soroll_m * 1000 / math.sqrt(12):.4f} mm de desviacio)")
    print(f"   b amb la parabola ajustada al cim: {b:.5f}  "
          f"({(b - b_bona) / b_bona:+.2%})")
    print(f"   b llegint la mostra mes alta:      {pelat:.5f}  "
          f"({(pelat - b_bona) / b_bona:+.2%})   <- el biaix del soroll")

    proves = [
        ("b", abs(b - b_bona) / b_bona, 0.005),
        ("T", abs(T - T_bona) / T_bona, 0.001),
        ("radi", abs(radi - L) / L, 0.01),
        ("centre", math.hypot(*res["circumferencia"]["centre_m"]) / L, 0.01),
        ("inclinacio", abs(res["circumferencia"]["inclinacio_de_l_arc_graus"]) / 90.0, 0.01),
    ]
    print(f"\n   {'que':>12}  {'error':>10}  {'llindar':>8}")
    tot_be = True
    for nom, error, llindar in proves:
        be = error <= llindar
        tot_be &= be
        print(f"   {nom:>12}  {error:>9.4%}  {llindar:>7.1%}   {'OK' if be else 'MALAMENT'}")

    print("\n   " + ("OK -- l'eina recupera el que se li ha posat."
                     if tot_be else
                     "MALAMENT -- l'eina no recupera el que se li ha posat."))
    return 0 if tot_be else 1


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("entrada", nargs="?", help="CSV de calibratge (de magnetic.tracker)")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova l'eina amb un senyal de b i T conegudes")
    ap.add_argument("--tram", type=float, nargs=2, default=list(TRAM_PER_DEFECTE),
                    metavar=("ALT", "BAIX"),
                    help="rang d'amplituds del segon ajust, en CM (per defecte 15 10)")
    ap.add_argument("--nom", default=None, help="nom base de la figura i del JSON")
    ap.add_argument("--figures", default="magnetic/figures", help="on va la figura")
    ap.add_argument("--mostra", action="store_true", help="obre una finestra en comptes de desar")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.entrada:
        ap.error("digue'm quin CSV he de llegir, o passa --autoprova")

    p = Parametres.des_de_json()
    print("=" * 72)
    print(" magnetic.decrement -- la friccio, pel decrement logaritmic")
    print("=" * 72)

    taula, meta = dades.carrega_calibratge(args.entrada)
    t = taula["t"].to_numpy()
    x = taula["x"].to_numpy()
    y = taula["y"].to_numpy()
    print(f"\n{args.entrada}: {len(t)} mostres, {t[-1] - t[0]:.2f} s")
    if meta.get("que_es"):
        print(f"   {meta['que_es']}")
    if meta.get("fotogrames_interpolats"):
        print(f"   !! {meta['fotogrames_interpolats']} fotogrames son interpolats")

    res = analitza(t, x, y, g=p.g, tram=tuple(args.tram))
    escriu(res, p)

    nom = args.nom or Path(args.entrada).stem
    dibuixa(res, t, x, y, nom, args.figures, args.mostra)

    if not args.mostra:
        sortida = Path(args.figures) / f"{nom}.json"
        publicable = {k: v for k, v in res.items() if not k.startswith("_")}
        publicable["origen"] = "magnetic.decrement"
        publicable["entrada"] = str(args.entrada)
        publicable["metadades_de_l_entrada"] = meta
        publicable["parametres_en_us"] = p.com_a_dict()
        sortida.write_text(json.dumps(publicable, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"   numeros: {sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
