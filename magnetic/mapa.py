"""Generador del mapa de conques d'atraccio. El resultat de debo del treball.

Per a cada punt de sortida d'una graella es deixa anar la massa EN REPOS i
s'apunta en quin imant acaba i quants segons hi ha trigat. El mapa de qui guanya
a cada punt es el fractal.

EL MATEIX CODI, COMPILAT O NO. El nucli va compilat amb Numba i paral.lelitzat
per files. Posant MAGNETIC_SENSE_NUMBA=1 a l'entorn, el mateix programa es
calcula amb Python pelat, unes trenta vegades mes a poc a poc, i les dues
matrius han de sortir IDENTIQUES PIXEL A PIXEL. No son dos programes que es
comparen: es un sol programa amb el compilador engegat o apagat. Vegeu
magnetic/compila.py.

La finestra es variable (--finestra x_min x_max y_min y_max) perque es pugui
ampliar un requadre i tornar-lo a calcular amb mes resolucio. Aixo es
l'autosemblanca, i es la prova visual que la frontera no te gruix.

Es desen SEMPRE la matriu (.npz), el contracte de dades (.csv) i les metadades
(.json), no nomes el PNG. D'un PNG comprimit no es pot recuperar de manera
fiable quin imant hi havia a cada pixel.

Exemples, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.mapa --resolucio 120 --nom general

    .venv\\Scripts\\python.exe -m magnetic.mapa --finestra -2 2 -2 2 ^
        --resolucio 240 --nom zoom1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import compila, dades, estil
from magnetic.fisica import (
    Parametres,
    afegeix_banderes_de_parametres,
    avis_inventats,
    dins_de_labast,
    sobreescriu,
)
from magnetic.motor import (
    DT_PER_DEFECTE,
    NO_CLASSIFICAT,
    Criteri,
    desenllac,
    destins,
    graella_nucli,
    integra_fins_a_classificar,
)


def _comprova_els_dos_camins(p, criteri, dest, dt):
    """desenllac() i integra_fins_a_classificar() han de dir el mateix.

    Son dos bucles diferents -- un guarda la trajectoria i l'altre no -- i per
    tant es poden desincronitzar amb el temps sense que ningu se n'adoni. Es
    comprova aqui, que es barat, abans de gastar hores de calcul.
    """
    dest_x, dest_y, dest_codi, dest_radi2, _ = dest
    for x, y in ((0.06, 0.06), (-0.07, 0.03), (0.02, -0.01), (0.10, 0.0)):
        c1, t1 = desenllac(p, x, y, criteri, dest_x, dest_y, dest_codi, dest_radi2, dt)
        c2, t2, _, _ = integra_fins_a_classificar(p, [x, y, 0.0, 0.0], criteri, dt)
        if c1 != c2 or abs(t1 - t2) > 1e-9:
            raise RuntimeError(
                f"Els dos camins no coincideixen a ({x}, {y}): "
                f"desenllac diu {c1}@{t1:.3f} i integra_fins_a_classificar {c2}@{t2:.3f}."
            )


def calcula(p, finestra, resolucio, criteri, dt, cada=1):
    """La graella. Retorna (x0, y0, imant, temps_caiguda).

    imant es una matriu (resolucio, resolucio) amb la fila 0 a y_min. temps
    en segons.
    """
    x_min, x_max, y_min, y_max = finestra
    x0 = np.linspace(x_min, x_max, resolucio)
    y0 = np.linspace(y_min, y_max, resolucio)

    # El fil te llargada fixa: els punts de fora no existeixen i no s'integren.
    xx, yy = np.meshgrid(x0, y0)
    if not np.all(dins_de_labast(xx, yy, p)):
        r = float(np.max(np.hypot(xx, yy)))
        raise ValueError(
            f"La finestra arriba a {r*100:.1f} cm del centre i el fil nomes fa "
            f"{p.L*100:.1f} cm: aquells punts NO EXISTEIXEN. Fes la finestra mes petita."
        )

    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    imant = np.empty((resolucio, resolucio), dtype=np.int64)
    temps = np.empty((resolucio, resolucio), dtype=np.float64)

    passos = int(round(criteri.t_max / dt))
    passos_confirma = int(round(criteri.t_confirma / dt))
    passos_confirma_centre = int(round(criteri.t_confirma_centre / dt))

    # Es calcula per blocs de files nomes per poder dir com va. El resultat es
    # el mateix es faci d'un cop o de mil: cada pixel es independent.
    t_inici = time.perf_counter()
    for a in range(0, resolucio, cada):
        z = min(a + cada, resolucio)
        graella_nucli(x0, y0[a:z], p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K,
                      dest_x, dest_y, dest_codi, dest_radi2,
                      dt, passos, passos_confirma, passos_confirma_centre,
                      imant[a:z], temps[a:z])
        fet = z / resolucio
        passat = time.perf_counter() - t_inici
        queda = passat / fet - passat
        print(f"   fila {z:5d}/{resolucio}  ({100*fet:5.1f} %)  "
              f"passats {passat/60:5.1f} min, queden {queda/60:5.1f} min", flush=True)

    return x0, y0, imant, temps


def resum(imant) -> dict:
    """Quants pixels de cada desti. El gris SEMPRE s'ha d'imprimir."""
    total = imant.size
    comptes = {}
    for codi in (-1, 0, 1, 2, 3):
        n = int(np.count_nonzero(imant == codi))
        if n or codi in (-1, 0):
            comptes[codi] = {"pixels": n, "percent": round(100 * n / total, 3)}
    return comptes


def _figura_conques(x0, y0, imant, p, titol):
    cmap, norm = estil.mapa_de_colors()
    fig, eix = plt.subplots(figsize=(6.4, 6.4))
    eix.imshow(
        imant, cmap=cmap, norm=norm, interpolation="nearest", origin="lower",
        extent=[x0[0] * 100, x0[-1] * 100, y0[0] * 100, y0[-1] * 100],
    )
    for nm, ix, iy in zip(p.noms, p.im_x, p.im_y):
        if x0[0] <= ix <= x0[-1] and y0[0] <= iy <= y0[-1]:
            eix.plot(ix * 100, iy * 100, "o", ms=5, mfc="none",
                     mec=estil.SUPERFICIE, mew=1.4)
            eix.annotate(nm, (ix * 100, iy * 100), textcoords="offset points",
                         xytext=(7, 5), fontsize=9, color=estil.SUPERFICIE)
    eix.set_xlabel("x de sortida (cm)")
    eix.set_ylabel("y de sortida (cm)")
    eix.set_title(titol)
    eix.grid(False)
    return fig


def _figura_temps(x0, y0, imant, temps, titol):
    """El temps de caiguda. La frontera fractal s'hi veu encara millor."""
    dades_temps = np.ma.masked_where(imant == NO_CLASSIFICAT, temps)
    fig, eix = plt.subplots(figsize=(6.9, 6.4))
    im = eix.imshow(
        dades_temps, cmap="magma_r", interpolation="nearest", origin="lower",
        extent=[x0[0] * 100, x0[-1] * 100, y0[0] * 100, y0[-1] * 100],
    )
    barra = fig.colorbar(im, ax=eix, fraction=0.046, pad=0.04)
    barra.set_label("segons fins a caure")
    eix.set_xlabel("x de sortida (cm)")
    eix.set_ylabel("y de sortida (cm)")
    eix.set_title(titol)
    eix.grid(False)
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--finestra", type=float, nargs=4, default=[-12, 12, -12, 12],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="limits en CENTIMETRES (per defecte -12 12 -12 12)")
    ap.add_argument("--resolucio", type=int, default=120, help="pixels per costat")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE, help="pas d'integracio (s)")
    ap.add_argument("--temps", type=float, default=None, help="temps maxim (s)")
    ap.add_argument("--r-captura", type=float, default=None, help="radi de confinament (CM)")
    ap.add_argument("--t-confirma", type=float, default=None,
                    help="temps de confinament als IMANTS (S); el del centre es "
                         "Criteri.t_confirma_centre")
    ap.add_argument("--nom", default="mapa", help="nom base dels fitxers")
    ap.add_argument("--dades", default="magnetic/dades", help="on van .npz/.csv/.json")
    ap.add_argument("--figures", default="magnetic/figures", help="on van les figures")
    afegeix_banderes_de_parametres(ap)
    args = ap.parse_args(argv)

    p = Parametres.des_de_json()
    p, avis_sobreescrit = sobreescriu(p, b=args.b, n=args.n, K=args.K)
    d = Criteri()
    criteri = Criteri(
        r_captura=d.r_captura if args.r_captura is None else args.r_captura / 100.0,
        r_centre=d.r_centre if args.r_captura is None else args.r_captura / 100.0,
        t_confirma=d.t_confirma if args.t_confirma is None else args.t_confirma,
        t_max=d.t_max if args.temps is None else args.temps,
    )
    finestra = [v / 100.0 for v in args.finestra]

    print("=" * 72)
    print(" magnetic.mapa -- conques d'atraccio")
    print("=" * 72)
    if avis_sobreescrit:
        print(avis_sobreescrit)
    avis = avis_inventats(p)
    if avis:
        print(avis)
    print("!! " + criteri.avis())

    dest = destins(p, criteri)
    _comprova_els_dos_camins(p, criteri, dest, args.dt)
    print("   els dos camins de calcul coincideixen                        OK")
    if not dest[4]:
        print("   EL CENTRE ES INESTABLE: aquest mapa no tindra regio negra.")

    ample = (finestra[1] - finestra[0]) / args.resolucio
    print(f"\n   finestra {args.finestra} cm, {args.resolucio}x{args.resolucio} pixels")
    print(f"   un pixel = {ample*1000:.3f} mm     dt = {args.dt} s")
    print(f"   {args.resolucio**2} trajectories per calcular\n")

    t0 = time.perf_counter()
    x0, y0, imant, temps = calcula(p, finestra, args.resolucio, criteri, args.dt,
                                   cada=max(1, args.resolucio // 20))
    durada = time.perf_counter() - t0

    comptes = resum(imant)
    print(f"\n   calculat en {durada/60:.1f} min")
    print(f"\n   {'desti':>14} {'pixels':>9} {'part':>8}")
    noms = {-1: "NO CONVERGIT", 0: "el centre"}
    for codi, v in sorted(comptes.items()):
        nom = noms.get(codi, f"imant {codi}")
        print(f"   {nom:>14} {v['pixels']:>9} {v['percent']:>7.2f} %")

    # EL TEMPS DE CAIGUDA MES LLARG, que es el germa del percentatge de gris i
    # diu una cosa que el gris no diu: quant MARGE queda. Amb 0 % de gris el
    # mapa esta acabat, pero si la trajectoria mes lenta triga 198 s d'un t_max
    # de 200, el marge es de dos segons i el proxim mapa pot quedar clapat.
    # Nomes compta els pixels que han convergit: els grisos hi desen el t_max,
    # que no es cap temps de caiguda sino el rellotge que s'ha acabat.
    convergits = temps[imant != -1]
    if convergits.size:
        print(f"\n   temps de caiguda: mediana {np.median(convergits):.1f} s, "
              f"el mes llarg {convergits.max():.1f} s "
              f"(el maxim son {criteri.t_max:.0f} s, "
              f"o sigui un marge del {100*(1 - convergits.max()/criteri.t_max):.0f} %)")

    gris = comptes[-1]["percent"]
    if gris > 0.5:
        print(f"\n   ATENCIO: {gris:.2f} % de pixels grisos. El temps maxim de "
              f"{criteri.t_max:.0f} s es massa curt\n   i el mapa NO esta acabat.")
    else:
        print(f"\n   El gris es {gris:.2f} %: practicament inexistent, com ha de ser.")

    meta = {
        "origen": "magnetic.mapa",
        "calcul": compila.com_va(),
        "finestra_m": {"x_min": finestra[0], "x_max": finestra[1],
                       "y_min": finestra[2], "y_max": finestra[3]},
        "resolucio": args.resolucio,
        "pixel_m": ample,
        "integrador": {"metode": "RK4 pas fix", "dt_s": args.dt},
        "criteri": criteri.com_a_dict(),
        "parametres": p.com_a_dict(),
        # p.com_a_dict() ja desa els valors que s'han fet servir de debo; aixo
        # diu, a mes, quins NO venien de parametres.json. Sense aixo, d'aqui a
        # mesos un mapa fet amb --b 0.8 semblaria el del muntatge de sempre.
        "_sobreescrits": {clau: valor for clau, valor in
                          (("b", args.b), ("n", args.n), ("K", args.K))
                          if valor is not None} or None,
        "punts_de_repos_m": [{"codi": int(c), "x": float(ex), "y": float(ey)}
                             for ex, ey, c in zip(dest[0], dest[1], dest[2])],
        "resultat": {str(k): v for k, v in comptes.items()},
        "durada_s": round(durada, 1),
    }
    cami = dades.desa_graella(Path(args.dades) / args.nom, x0, y0, imant, temps, meta)
    print(f"\n   dades: {cami}  (+ .csv i .json)")

    estil.aplica()
    marca = f"{args.finestra[0]:g} a {args.finestra[1]:g} cm, {args.resolucio}x{args.resolucio} px"
    f1 = _figura_conques(x0, y0, imant, p, f"Conques d'atraccio\n{marca}")
    estil.peu_de_valors_inventats(f1, p.inventats)
    estil.desa(f1, args.nom, args.figures)
    f2 = _figura_temps(x0, y0, imant, temps, f"Temps de caiguda\n{marca}")
    estil.peu_de_valors_inventats(f2, p.inventats)
    estil.desa(f2, f"{args.nom}_temps", args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
