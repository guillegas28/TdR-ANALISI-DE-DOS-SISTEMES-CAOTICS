"""Un sol llancament del pendol magnetic: la integra, la desa i la dibuixa.

Aquest programa NO fa mapes. Serveix per mirar-se trajectories concretes una
per una, que es l'unica manera honesta de decidir el criteri d'aturada: primer
veus que fa la massa, i despres comproves que el codi en diu el mateix que has
vist. Un criteri triat sense mirar cap trajectoria inventa fronteres que no
existeixen.

Es dibuixa un requadre per llancament, amb els tres imants, les CIRCUMFERENCIES
DEL CRITERI en discontinu i el titol dient com ha acabat. Aixi el dibuix i la
classificacio es poden contrastar d'un cop d'ull.

Les distancies de la linia d'ordres van en CENTIMETRES, i les dels eixos de les
figures tambe. Dins del codi, SI: tot metres.

Exemples, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.trajectoria --des-de 10 0

    .venv\\Scripts\\python.exe -m magnetic.trajectoria ^
        --des-de 10 0 --des-de 0 10 --des-de -7 -7 --nom sis_tirs

Comprovacio de sanitat (sense imants i sense friccio ha de sortir una
el.lipse tancada, perque es un oscil.lador harmonic en dues dimensions):

    .venv\\Scripts\\python.exe -m magnetic.trajectoria --des-de 2 0 ^
        --velocitat 0 8 --sense-imants --sense-friccio --temps 5 --nom ellipse
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import dades, estil
from magnetic.fisica import Parametres, avis_inventats
from magnetic.motor import (
    DT_PER_DEFECTE,
    Criteri,
    destins,
    integra_fins_a_classificar,
    nom_del_desti,
)

LLETRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _dibuixa_un(ax, p, criteri, dest, t, y, etiqueta, titol, limit_cm):
    """Un requadre: imants, punts de repos, rodones del criteri i trajectoria."""
    cm = 100.0
    dest_x, dest_y, dest_codi, dest_radi2, _ = dest

    # Els imants, amb el seu numero. Sense color: la identitat la dona el
    # numero, perque els colors son una decisio del bloc 6.
    for nm, ix, iy in zip(p.noms, p.im_x, p.im_y):
        ax.plot(ix * cm, iy * cm, "o", ms=6, color=estil.TINTA)
        ax.annotate(nm, (ix * cm, iy * cm), textcoords="offset points",
                    xytext=(7, 5), fontsize=9, color=estil.TINTA)

    # Les rodones del criteri van al voltant dels PUNTS DE REPOS, que queden
    # cap endins dels imants: alla es on la massa s'atura de debo.
    for ex, ey, radi2 in zip(dest_x, dest_y, dest_radi2):
        ax.add_patch(
            plt.Circle((ex * cm, ey * cm), math.sqrt(radi2) * cm, fill=False,
                       ls=(0, (3, 2)), lw=0.9, color=estil.TINTA_SUAU)
        )
        ax.plot(ex * cm, ey * cm, "x", ms=5, mew=1.1, color=estil.TINTA_2)

    ax.plot(y[0] * cm, y[1] * cm, "-", lw=0.9, color=estil.TINTA, alpha=0.85)
    ax.plot(y[0, 0] * cm, y[1, 0] * cm, "s", ms=6, mfc=estil.SUPERFICIE,
            mec=estil.TINTA, mew=1.2)
    ax.annotate(etiqueta, (y[0, 0] * cm, y[1, 0] * cm), textcoords="offset points",
                xytext=(8, -12), fontsize=10, fontweight="bold", color=estil.TINTA)
    ax.plot(y[0, -1] * cm, y[1, -1] * cm, "o", ms=5, color=estil.TINTA)

    ax.set_aspect("equal")
    ax.set_xlim(-limit_cm, limit_cm)
    ax.set_ylim(-limit_cm, limit_cm)
    ax.set_title(titol, fontsize=10)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--des-de", type=float, nargs=2, action="append", metavar=("X", "Y"),
                    required=True, help="punt de sortida en CENTIMETRES; repetible")
    ap.add_argument("--velocitat", type=float, nargs=2, default=[0.0, 0.0],
                    metavar=("VX", "VY"), help="velocitat inicial en CM/S (per defecte, repos)")
    ap.add_argument("--sense-imants", action="store_true", help="totes les K a zero")
    ap.add_argument("--sense-friccio", action="store_true", help="b a zero")
    ap.add_argument("--temps", type=float, default=None, help="temps maxim (s)")
    # Els tres llindars, per poder fer la prova de sensibilitat del criteri:
    # si canviant-los canvia el DESTI d'algun punt, el criteri no es de fiar.
    ap.add_argument("--r-captura", type=float, default=None, help="radi de captura (CM)")
    ap.add_argument("--r-centre", type=float, default=None, help="radi del centre (CM)")
    ap.add_argument("--t-confirma", type=float, default=None, help="temps de confinament (S)")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE, help="pas d'integracio (s)")
    ap.add_argument("--nom", default="tir", help="nom base dels fitxers de sortida")
    ap.add_argument("--dades", default="magnetic/dades", help="on van els CSV")
    ap.add_argument("--figures", default="magnetic/figures", help="on van les figures")
    ap.add_argument("--mostra", action="store_true", help="obre una finestra en comptes de desar")
    args = ap.parse_args(argv)

    p = Parametres.des_de_json()
    if args.sense_imants:
        p = p.sense_imants()
    if args.sense_friccio:
        p = p.sense_friccio()
    per_defecte = Criteri()
    criteri = Criteri(
        r_captura=per_defecte.r_captura if args.r_captura is None else args.r_captura / 100.0,
        r_centre=per_defecte.r_centre if args.r_centre is None else args.r_centre / 100.0,
        t_confirma=per_defecte.t_confirma if args.t_confirma is None else args.t_confirma,
        t_max=per_defecte.t_max if args.temps is None else args.temps,
    )

    print("=" * 72)
    print(" magnetic.trajectoria -- llancaments d'un en un")
    print("=" * 72)
    avis = avis_inventats(Parametres.des_de_json())
    if avis:
        print(avis)
    print("!! " + criteri.avis())
    if args.sense_imants or args.sense_friccio:
        apagat = [n for n, v in (("imants", args.sense_imants), ("friccio", args.sense_friccio)) if v]
        print("!! MUNTATGE ALTERAT: " + " i ".join(apagat) + " fora. No es el pendol de debo.")
    print()

    v0 = (args.velocitat[0] / 100.0, args.velocitat[1] / 100.0)
    resultats = []

    print(f"   {'':2} {'sortida (cm)':>16}  {'desti':>14}  {'temps (s)':>9}  {'passos':>8}")
    print("   " + "-" * 56)
    for i, (xc, yc) in enumerate(args.des_de):
        estat0 = [xc / 100.0, yc / 100.0, v0[0], v0[1]]
        codi, temps, t, y = integra_fins_a_classificar(p, estat0, criteri, args.dt)
        desti = nom_del_desti(codi, p.noms)
        lletra = LLETRES[i % len(LLETRES)]
        resultats.append((lletra, xc, yc, codi, temps, t, y))
        print(f"   {lletra:2} {f'({xc:+.2f}, {yc:+.2f})':>16}  {desti:>14}  {temps:9.3f}  {len(t):8d}")

        cami = Path(args.dades) / f"{args.nom}_{lletra}.csv"
        dades.desa_trajectoria(
            cami, t, y[0], y[1], y[2], y[3],
            meta={
                "origen": "magnetic.trajectoria",
                "sortida_cm": [xc, yc],
                "velocitat_inicial_cm_s": list(args.velocitat),
                "desti": {"codi": int(codi), "nom": desti, "temps_s": temps},
                "integrador": {"metode": "RK4 pas fix", "dt_s": args.dt},
                "criteri": criteri.com_a_dict(),
                "parametres": p.com_a_dict(),
                "muntatge_alterat": {
                    "sense_imants": bool(args.sense_imants),
                    "sense_friccio": bool(args.sense_friccio),
                },
            },
        )
    print(f"\n   {len(resultats)} CSV + JSON a {args.dades}/")

    # --- la figura --------------------------------------------------------
    estil.aplica()
    n = len(resultats)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.3 * nrows), squeeze=False)

    limit = max(float(np.max(np.abs(y[:2]))) for *_, y in resultats)
    limit_cm = max(limit * 100.0 * 1.12, p.R * 100.0 * 1.6)

    dest = destins(p, criteri)
    for eix, (lletra, xc, yc, codi, temps, t, y) in zip(axs.flat, resultats):
        titol = f"{lletra}: ({xc:+.1f}, {yc:+.1f}) cm  ->  {nom_del_desti(codi, p.noms)}, {temps:.2f} s"
        _dibuixa_un(eix, p, criteri, dest, t, y, lletra, titol, limit_cm)
    for eix in axs.flat[n:]:
        eix.axis("off")

    fig.suptitle(
        "Pendol magnetic: trajectories i criteri d'aturada\n"
        "punt ple = imant;  x = punt de repos, uns mm cap endins;  "
        "rodona = radi de confinament",
        fontsize=11,
    )
    fig.tight_layout()
    estil.peu_de_valors_inventats(fig, Parametres.des_de_json().inventats)

    if args.mostra:
        plt.show()
    else:
        estil.desa(fig, args.nom, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
