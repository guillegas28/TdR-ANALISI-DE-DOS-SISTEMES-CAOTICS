"""Analisi d'una graella: percentatge de frontera i dimensio fractal.

Aixo es el que converteix un dibuix bonic en un RESULTAT. El mapa ensenya que
la frontera es enrevessada; aquests dos numeros diuen quant.

PERCENTATGE DE FRONTERA
    Fraccio de pixels que tenen algun vei d'un altre color. DEPEN DE LA
    RESOLUCIO, i per aixo no es pot escriure mai sol: sempre amb la resolucio
    al costat. En una frontera normal, doblar la resolucio hauria de baixar
    aquesta fraccio a la meitat; en una de fractal, baixa molt menys, i aquesta
    diferencia es precisament la dimensio.

    NO es calcula amb scipy.ndimage.generic_filter. Aquella funcio crida una
    funcio de Python un cop per pixel: amb un milio de pixels son minuts, i
    contradiu que el criteri de tot aquest modul sigui el rendiment. Aqui es
    compara la matriu amb les seves vuit veines desplacades amb numpy, i son
    mil.lisegons.

DIMENSIO FRACTAL PER BOX-COUNTING
    Es cobreix la frontera amb quadrats de costat s i es compten els que en
    contenen algun tros, N(s). Si la frontera fos una corba normal, N creixeria
    com 1/s: doblar la finor doblaria el compte, i la dimensio seria 1. Si fos
    una taca, creixeria com 1/s^2 i la dimensio seria 2.

    En una frontera fractal surt un numero d'entremig, NO ENTER. Es el mateix
    numero que explica per que ampliant sempre hi torna a haver estructura.

    TOTES LES ESCALES ES COMPTEN DAMUNT DE LA MATEIXA AREA (31/10/2026). Fins
    aleshores cada escala retallava les files i columnes que no omplien una
    capsa sencera, i com que 600 no es divisible per 16, 32 ni 64, a les
    escales grosses es comptava damunt de 592 i 576 pixels en comptes de 600.
    Aixo feia que N baixes mes de pressa del que tocava: UN QUADRAT PLE, QUE TE
    DIMENSIO 2 EXACTA, DONAVA 2,022, i el mapa general en donava 2,006. Ara es
    compta tot damunt d'un quadrat central amb el costat multiple de la capsa
    mes gran (regio_comuna): 576 pixels als mapes de 600 i de 601.

    I L'ERROR ES L'INTERVAL DELS PENDENTS LOCALS, el de cada parell d'escales
    consecutives (pendents_locals). Diu com canvia la dimensio segons l'escala,
    que es el que de debo limita el numero: els residus de l'ajust son
    sistematics, no aleatoris. Amb la regio comuna hi ha dues coses exactes:
      - cap pendent local no pot passar de 2, perque una capsa de costat 2s
        conte com a molt quatre capses de costat s: N(2s) >= N(s)/4;
      - amb les escales espaiades igual en log, el pendent de l'ajust es una
        mitjana ponderada dels pendents locals, o sigui que LA DIMENSIO SEMPRE
        CAU DINS DEL SEU INTERVAL I NO POT PASSAR DE 2.
    Ho comprova --autoprova.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.analisi magnetic/dades/general.npz
    .venv\\Scripts\\python.exe -m magnetic.analisi --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import dades, estil

VUIT_VEINS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def mascara_de_frontera(imant) -> np.ndarray:
    """True als pixels que tenen algun vei d'un altre color.

    Es fa amb vuit comparacions de matrius desplacades. Cada desplacament
    compara nomes la part que se solapa, i aixi les vores de la imatge queden
    ben tractades sense cap cas especial.
    """
    m = np.asarray(imant)
    frontera = np.zeros(m.shape, dtype=bool)
    for dj, di in VUIT_VEINS:
        a = m[max(0, dj):m.shape[0] + min(0, dj), max(0, di):m.shape[1] + min(0, di)]
        b = m[max(0, -dj):m.shape[0] + min(0, -dj), max(0, -di):m.shape[1] + min(0, -di)]
        diferents = a != b
        frontera[max(0, dj):m.shape[0] + min(0, dj),
                 max(0, di):m.shape[1] + min(0, di)] |= diferents
    return frontera


def percentatge_de_frontera(imant) -> float:
    return 100.0 * float(np.count_nonzero(mascara_de_frontera(imant))) / imant.size


def compta_capses(mascara, costat: int) -> int:
    """Quantes capses de costat x costat contenen algun pixel de la mascara.

    Es fa reordenant la matriu en blocs amb reshape i preguntant si n'hi ha cap
    de cert a cada bloc. Sense bucles de Python.

    Si el costat no divideix la matriu, la franja que sobra es llenca.
    dimensio_per_box_counting no la hi deixa arribar mai: li passa la regio
    comuna, que totes les capses divideixen.
    """
    h, w = mascara.shape
    hh = (h // costat) * costat
    ww = (w // costat) * costat
    if hh == 0 or ww == 0:
        return 0
    tallada = mascara[:hh, :ww]
    blocs = tallada.reshape(hh // costat, costat, ww // costat, costat)
    return int(np.count_nonzero(blocs.any(axis=(1, 3))))


def escales(forma, costat_min=1, costat_max=None) -> list:
    """Els costats de capsa: costat_min, el doble, el doble... fins a costat_max,
    i mai mes grans que la matriu."""
    if costat_max is None:
        costat_max = max(4, min(forma) // 8)
    fora, s = [], costat_min
    while s <= costat_max and s <= min(forma):
        fora.append(s)
        s *= 2
    return fora


def regio_comuna(forma, costats):
    """(j0, i0, hh, ww): el rectangle central, amb els costats multiples de la
    capsa mes gran, on es compten TOTES les escales.

    Com que els costats van doblant, un multiple de la capsa mes gran ho es de
    totes, i cap escala no retalla res. Es centrat perque el que es deixa fora
    -- com a molt una capsa grossa menys un pixel -- es reparteixi entre les
    quatre vores en comptes de caure tot a dues.
    """
    gran = max(costats)
    h, w = forma
    hh = (h // gran) * gran
    ww = (w // gran) * gran
    return (h - hh) // 2, (w - ww) // 2, hh, ww


def dimensio_per_box_counting(mascara, costat_min=1, costat_max=None):
    """(dimensio, costats, comptes). Pendent de log N contra log (1/s), amb totes
    les escales comptades damunt de la regio comuna."""
    mascara = np.asarray(mascara)
    totes = escales(mascara.shape, costat_min, costat_max)
    if len(totes) < 3:
        raise RuntimeError(
            "No hi ha prou escales per ajustar cap dimensio. La graella es massa "
            "petita: fes-la de 200x200 com a minim."
        )
    j0, i0, hh, ww = regio_comuna(mascara.shape, totes)
    regio = mascara[j0:j0 + hh, i0:i0 + ww]
    costats, comptes = [], []
    for s in totes:
        n = compta_capses(regio, s)
        if n > 0:
            costats.append(s)
            comptes.append(n)
    if len(costats) < 3:
        raise RuntimeError(
            "No hi ha prou escales per ajustar cap dimensio. La graella es massa "
            "petita: fes-la de 200x200 com a minim."
        )
    costats = np.array(costats, dtype=float)
    comptes = np.array(comptes, dtype=float)
    pendent, _ = np.polyfit(np.log(1.0 / costats), np.log(comptes), 1)
    return float(pendent), costats, comptes


def pendents_locals(costats, comptes) -> np.ndarray:
    """El pendent de cada parell d'escales consecutives. El minim i el maxim son
    l'interval de la dimensio."""
    x = np.log(1.0 / np.asarray(costats, dtype=float))
    y = np.log(np.asarray(comptes, dtype=float))
    return np.diff(y) / np.diff(x)


def _figura(costats, comptes, dimensio, interval, pixel_m):
    """El grafic del box-counting, fet per poder-lo ENTENDRE sense el codi.

    ABANS NO S'ENTENIA, i val la pena tenir escrit per que, perque son tres
    errors independents i cap era el calcul:

      1. Els eixos eren log10(costat) i log10(capses), o sigui NUMEROS SENSE
         UNITATS. "-0.4" no vol dir res per a ningu. Ara els eixos van en escala
         logaritmica pero amb els VALORS de debo: mil.limetres i capses.
      2. No deia enlloc que es una capsa. Ho diuen els eixos («quantes capses
         toquen la frontera») i el peu de figura.
      3. El pendent sortia NEGATIU i la dimensio POSITIVA, sense dir que una es
         menys l'altra. Ara ho diu la llegenda.

    SENSE TITOL NI PARAGRAF DE SOTA (02/11/2026): la figura del mapa general va a
    la memoria, i el titol i l'explicacio de que es una capsa van al peu. La
    llegenda porta la coma decimal.

    I s'hi afegeixen DUES RECTES DE REFERENCIA, que son el que converteix un
    numero solt en una mesura: si la frontera fos una corba llisa el pendent
    seria -1, i si fos una taca que omple l'area seria -2. Veure entre quines
    dues coses cau la recta mesurada es tot el resultat.
    """
    fig, eix = plt.subplots(figsize=(6.8, 5.0))
    s_mm = np.asarray(costats, dtype=float) * pixel_m * 1000.0
    n = np.asarray(comptes, dtype=float)

    # Les dues referencies, ancorades al punt del mig perque la mesurada quedi
    # entremig i es puguin comparar els PENDENTS, que es del que va.
    mig = len(s_mm) // 2
    for d, estil_linia, etiqueta in (
        (1.0, (0, (5, 4)), "una corba llisa donaria pendent −1  (D = 1)"),
        (2.0, (0, (1, 3)), "una taca que omple l'àrea, pendent −2  (D = 2)"),
    ):
        eix.plot(s_mm, n[mig] * (s_mm / s_mm[mig]) ** (-d), ls=estil_linia,
                 lw=1.5, color=estil.EIX, label=etiqueta, zorder=4)

    # LA MESURA ES UNA BANDA AMPLA I TRANSLUCIDA, no una linia. Amb D = 1,998
    # la recta mesurada cau EXACTAMENT damunt de la referencia de D = 2, i
    # dibuixada com a linia l'amagava: semblava que la referencia no hi fos.
    # Com a banda, la referencia se li veu a traves i es llegeix el que de debo
    # passa -- que la mesura i el D = 2 son la mateixa cosa.
    coef = np.polyfit(np.log10(s_mm), np.log10(n), 1)
    eix.plot(s_mm, 10.0 ** np.polyval(coef, np.log10(s_mm)), "-", lw=7,
             color=estil.IMANTS[1], alpha=0.35, solid_capstyle="round", zorder=2,
             label=estil.coma(f"mesurat: D = −pendent = {dimensio:.2f}, "
                              f"de {interval[0]:.2f} a {interval[1]:.2f} segons l'escala"))
    eix.plot(s_mm, n, "o", ms=6, color=estil.TINTA, zorder=5,
             label="capses que toquen la frontera")

    eix.set_xscale("log")
    eix.set_yscale("log")
    eix.set_xlabel("costat de la capsa, s (mm)")
    eix.set_ylabel("quantes capses toquen la frontera, N(s)")
    eix.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# L'autoprova: casos on la dimensio se sap, sense cap mapa
# ---------------------------------------------------------------------------
def _dimensio_vella(mascara, costat_min=1, costat_max=None):
    """L'algorisme d'abans del 31/10, que retallava a cada escala. Nomes hi es
    per comprovar que on no retallava res (graelles divisibles per totes les
    capses) el nou dona exactament el mateix."""
    if costat_max is None:
        costat_max = max(4, min(mascara.shape) // 8)
    costats, comptes = [], []
    s = costat_min
    while s <= costat_max:
        n = compta_capses(mascara, s)
        if n > 0:
            costats.append(s)
            comptes.append(n)
        s *= 2
    costats = np.array(costats, dtype=float)
    comptes = np.array(comptes, dtype=float)
    pendent, _ = np.polyfit(np.log(1.0 / costats), np.log(comptes), 1)
    return float(pendent), costats, comptes


def autoprova() -> int:
    """Els criteris del pla del 31/10 (documentacio/pla-dimensio-box-counting.md),
    escrits abans de corregir res:

      A1  un quadrat ple dona D = 2 a 1e-12, i l'interval [2, 2]
      A2  una recta horitzontal d'un pixel dona D = 1 a 1e-12
      A3  on no hi havia res a retallar, els comptes i la D son els d'abans bit a bit
      A4  la D sempre cau dins del seu interval, i no passa mai de 2
    """
    print("=" * 72)
    print(" magnetic.analisi --autoprova")
    print("=" * 72)
    tol = 1e-12
    proves = []

    def interval_de(costats, comptes):
        p = pendents_locals(costats, comptes)
        return float(p.min()), float(p.max())

    # A1 -- el quadrat ple. Amb el retall d'abans, el de 600 donava 2,022.
    for costat in (600, 601, 200, 151, 512):
        m = np.ones((costat, costat), dtype=bool)
        d, cs, ns = dimensio_per_box_counting(m)
        lo, hi = interval_de(cs, ns)
        vell = _dimensio_vella(m)[0]
        error = max(abs(d - 2.0), abs(lo - 2.0), abs(hi - 2.0))
        proves.append((f"A1 ple {costat}", error, f"D = {d:.12f} (abans {vell:.4f})"))

    # A2 -- una recta horitzontal d'un pixel que travessa tota la graella
    for costat in (600, 601):
        m = np.zeros((costat, costat), dtype=bool)
        m[costat // 2, :] = True
        d, cs, ns = dimensio_per_box_counting(m)
        proves.append((f"A2 recta {costat}", abs(d - 1.0), f"D = {d:.12f}"))

    # A3 -- graelles que totes les capses divideixen: 512 (fins a 64, com la
    # Koch de figures_metodologia), 128 i 64 (com els casos de la prova 5 de
    # compara). Mascares a l'atzar, amb la llavor fixada.
    rng = np.random.default_rng(20261031)
    for costat, maxim in ((512, 64), (128, None), (64, None)):
        m = rng.random((costat, costat)) < 0.3
        nou = dimensio_per_box_counting(m, 1, maxim)
        vell = _dimensio_vella(m, 1, maxim)
        iguals = (nou[0] == vell[0] and np.array_equal(nou[1], vell[1])
                  and np.array_equal(nou[2], vell[2]))
        proves.append((f"A3 {costat}", 0.0 if iguals else 1.0,
                       "bit a bit" if iguals else f"{nou[0]!r} contra {vell[0]!r}"))

    # A4 -- mascares de densitats diferents a graelles que no es divideixen,
    # i taques: la D dins de l'interval i mai per sobre de 2
    pitjor = 0.0
    for costat in (600, 601, 200, 151):
        for densitat in (0.02, 0.3, 0.9, 0.999):
            m = rng.random((costat, costat)) < densitat
            d, cs, ns = dimensio_per_box_counting(m)
            lo, hi = interval_de(cs, ns)
            fora = max(lo - d, d - hi, d - 2.0, hi - 2.0, 0.0)
            pitjor = max(pitjor, fora)
    proves.append(("A4 dins i <= 2", pitjor, "16 mascares a l'atzar"))

    print(f"\n   {'prova':>16}  {'error':>10}  {'llindar':>8}")
    tot_be = True
    for nom, error, nota in proves:
        be = error <= tol
        tot_be &= be
        print(f"   {nom:>16}  {error:>10.1e}  {tol:>8.0e}   "
              f"{'OK' if be else 'MALAMENT'}   {nota}")

    print("\n   " + ("OK -- el box-counting dona la dimensio que toca als casos coneguts."
                     if tot_be else
                     "MALAMENT -- el box-counting no dona el que toca."))
    return 0 if tot_be else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("graella", nargs="?", help="fitxer .npz d'una graella")
    ap.add_argument("--figures", default="magnetic/figures", help="on va la figura")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova el box-counting amb casos de dimensio coneguda")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.graella:
        ap.error("digue'm quina graella he de llegir, o passa --autoprova")

    x0, y0, imant, temps, meta = dades.carrega_graella(args.graella)
    nom = Path(args.graella).with_suffix("").name
    res = imant.shape[0]
    pixel = meta.get("pixel_m", float(x0[1] - x0[0]))

    print("=" * 72)
    print(f" magnetic.analisi -- {nom}")
    print("=" * 72)
    inventats = meta.get("parametres", {}).get("valors_inventats", [])
    if inventats:
        print("!! Aquesta graella s'ha fet amb VALORS INVENTATS: " + ", ".join(inventats))
    f = meta.get("finestra_m", {})
    if f:
        print(f"   finestra {f['x_min']*100:.1f} a {f['x_max']*100:.1f} cm x "
              f"{f['y_min']*100:.1f} a {f['y_max']*100:.1f} cm")
    print(f"   {res}x{res} pixels, un pixel = {pixel*1000:.4f} mm\n")

    gris = 100.0 * float(np.count_nonzero(imant == -1)) / imant.size
    if gris > 0.5:
        print(f"   ATENCIO: {gris:.2f} % de pixels grisos. Aquesta graella no esta")
        print("   acabada i els numeros de sota surten tocats.\n")

    mascara = mascara_de_frontera(imant)
    pct = 100.0 * float(np.count_nonzero(mascara)) / imant.size
    print(f"   PERCENTATGE DE FRONTERA: {pct:.2f} %   a {res}x{res} pixels")
    print("   (aquest numero no vol dir res sense la resolucio al costat: com mes")
    print("    fina la graella, mes petita la fraccio)\n")

    dimensio, costats, comptes = dimensio_per_box_counting(mascara)
    j0, i0, hh, ww = regio_comuna(mascara.shape, escales(mascara.shape))
    locals_ = pendents_locals(costats, comptes)
    interval = (float(locals_.min()), float(locals_.max()))
    print(f"   capses comptades damunt dels {hh}x{ww} pixels del centre")
    print(f"   {'costat (px)':>12} {'costat (mm)':>12} {'capses':>10} {'pendent local':>14}")
    for k, (s, n) in enumerate(zip(costats, comptes)):
        local = f"{locals_[k - 1]:>14.3f}" if k else ""
        print(f"   {int(s):>12} {s*pixel*1000:>12.4f} {int(n):>10}{local}")
    print(f"\n   DIMENSIO FRACTAL (box-counting): D = {dimensio:.2f}, "
          f"de {interval[0]:.2f} a {interval[1]:.2f} segons l'escala")

    if dimensio <= 1.02:
        print("   D ~ 1: la frontera es una corba normal, no fractal.")
    elif dimensio >= 1.95:
        print("   D ~ 2: la frontera omple el pla. No es una frontera, es pols:")
        print("   en aquesta finestra el desti no esta determinat gairebe enlloc.")
    else:
        print("   Un numero entre 1 i 2 i no enter: la frontera no es ni una corba")
        print("   ni una taca. Es el que fa que ampliant hi torni a haver estructura.")

    estil.aplica()
    fig = _figura(costats, comptes, dimensio, interval, pixel)
    estil.peu_de_valors_inventats(fig, inventats)
    estil.desa(fig, f"{nom}_boxcounting", args.figures)

    sortida = Path(args.figures) / f"{nom}_metriques.json"
    sortida.write_text(json.dumps({
        "graella": nom,
        "resolucio": res,
        "pixel_m": pixel,
        "percentatge_de_frontera": round(pct, 4),
        "dimensio_box_counting": round(dimensio, 4),
        "dimensio_interval": [round(v, 4) for v in interval],
        "pendents_locals": [round(float(v), 4) for v in locals_],
        "regio_comptada_px": [hh, ww],
        "_regio": "totes les escales es compten damunt del rectangle central amb els "
                  "costats multiples de la capsa mes gran (des del 31/10/2026)",
        "costats_px": [int(s) for s in costats],
        "capses": [int(n) for n in comptes],
        "pixels_grisos_percent": round(gris, 4),
        "valors_inventats": inventats,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metriques: {sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
