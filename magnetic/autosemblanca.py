"""La figura estrella: el mateix mapa a quatre escales.

Cada requadre es una ampliacio del requadre marcat a l'anterior. Si al final de
la serie hi continua havent estructura -- filaments, capes, mes filaments a
dins -- vol dir que la frontera entre conques NO TE GRUIX: per molt que
l'ampliis, mai no arribes a una ratlla neta que separi dos colors.

Aixo es l'autosemblanca, i es la prova visual del que la dimensio fractal de
magnetic/analisi.py diu amb un numero.

Es llegeixen graelles ja calculades; aquest modul no calcula res. Primer fes
els mapes amb magnetic.mapa i despres ajunta'ls aqui.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.autosemblanca ^
        magnetic/dades/mapa_general.npz magnetic/dades/zoom1.npz ^
        magnetic/dades/zoom2.npz magnetic/dades/zoom3.npz
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import dades, estil


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("graelles", nargs="+", help="fitxers .npz, de mes ample a mes estret")
    ap.add_argument("--nom", default="autosemblanca", help="nom de la figura")
    ap.add_argument("--figures", default="magnetic/figures", help="on va la figura")
    args = ap.parse_args(argv)

    mapes = []
    for cami in args.graelles:
        x0, y0, imant, temps, meta = dades.carrega_graella(cami)
        mapes.append({"x0": x0, "y0": y0, "imant": imant, "meta": meta,
                      "nom": Path(cami).with_suffix("").name})

    amples = [m["x0"][-1] - m["x0"][0] for m in mapes]
    if any(b >= a for a, b in zip(amples[:-1], amples[1:])) is False:
        pass  # ja van de gran a petit

    estil.aplica()
    cmap, norm = estil.mapa_de_colors()
    n = len(mapes)
    fig, eixos = plt.subplots(1, n, figsize=(4.1 * n, 4.3))
    if n == 1:
        eixos = [eixos]

    print(f"{'mapa':>16} {'ample (cm)':>12} {'pixel (mm)':>12} {'ampliacio':>11}")
    for k, (eix, m) in enumerate(zip(eixos, mapes)):
        x0, y0, imant = m["x0"], m["y0"], m["imant"]
        ample = (x0[-1] - x0[0]) * 100
        pixel = (x0[1] - x0[0]) * 1000
        ampliacio = amples[0] / (x0[-1] - x0[0])
        print(f"{m['nom']:>16} {ample:>12.4f} {pixel:>12.5f} {ampliacio:>10.0f}x")

        eix.imshow(imant, cmap=cmap, norm=norm, interpolation="nearest", origin="lower",
                   extent=[x0[0] * 100, x0[-1] * 100, y0[0] * 100, y0[-1] * 100])
        # Els imants que cauen dins del requadre, com a mapa._figura_conques. Les
        # posicions surten de les metadades de cada mapa, no de parametres.json:
        # la figura ha de dir on eren els imants del calcul que ensenya.
        for im in m["meta"].get("parametres", {}).get("imants", []):
            if x0[0] <= im["x"] <= x0[-1] and y0[0] <= im["y"] <= y0[-1]:
                eix.plot(im["x"] * 100, im["y"] * 100, "o", ms=5, mfc="none",
                         mec=estil.SUPERFICIE, mew=1.4)
                eix.annotate(im["nom"], (im["x"] * 100, im["y"] * 100),
                             textcoords="offset points", xytext=(7, 5), fontsize=9,
                             color=estil.SUPERFICIE)
        # Una lletra per panell, perque el peu de la memoria els pugui citar.
        # Sense cap numero de figura: el posa el peu (02/11).
        lletra = f"({chr(ord('a') + k)})"
        quin = "mapa sencer" if k == 0 else f"×{ampliacio:.0f}"
        eix.set_title(estil.coma(f"{lletra} {quin}, {ample:.4g} cm de costat"),
                      fontsize=10, loc="left", fontweight="normal")
        eix.set_xlabel("x (cm)")
        if k == 0:
            eix.set_ylabel("y (cm)")
        eix.grid(False)
        eix.tick_params(labelsize=8)

        # El requadre del panell seguent, dibuixat a sobre d'aquest.
        if k + 1 < n:
            s = mapes[k + 1]
            eix.add_patch(plt.Rectangle(
                (s["x0"][0] * 100, s["y0"][0] * 100),
                (s["x0"][-1] - s["x0"][0]) * 100, (s["y0"][-1] - s["y0"][0]) * 100,
                fill=False, lw=1.4, ec=estil.SUPERFICIE))
            eix.add_patch(plt.Rectangle(
                (s["x0"][0] * 100, s["y0"][0] * 100),
                (s["x0"][-1] - s["x0"][0]) * 100, (s["y0"][-1] - s["y0"][0]) * 100,
                fill=False, lw=0.7, ec=estil.TINTA))

    # SENSE TITOL GRAN (02/11): el titol i l'explicacio -- que el requadre de
    # cada panell es el que s'amplia al seguent -- van al peu de la memoria.
    # Fins aleshores hi havia un suptitle de dues linies, sense accents.
    fig.tight_layout()
    inventats = mapes[0]["meta"].get("parametres", {}).get("valors_inventats", [])
    estil.peu_de_valors_inventats(fig, inventats)
    estil.desa(fig, args.nom, args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
