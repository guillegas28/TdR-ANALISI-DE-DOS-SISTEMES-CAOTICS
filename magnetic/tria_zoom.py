"""Les finestres dels zooms (x4, x16, x64): on es fa cada ampliacio.

No calcula cap trajectoria: llegeix una graella ja feta (el mapa pare) i hi
busca requadres de costat 1/4 del pare que caiguin damunt de la frontera. Es
l'eina que va proposar els zooms del 27/10: els vells es van triar damunt del
fractal de n = 3 i amb la llei mesurada podien no caure-hi.

LA REGLA (proposada el 27/10; les finestres les va triar en Guillem):
  - el requadre es quadrat, de costat 1/4 del pare, i cau sencer a dins;
  - hi surten els tres imants, cadascun amb un 5 % o mes dels pixels;
  - la frontera (analisi.mascara_de_frontera, al pixel del pare) hi ocupa
    entre el 15 i el 60 %: ni una sola ratlla neta ni pols;
  - el centre del requadre cau entre 0 i 60 graus: el mapa te la simetria dels
    tres imants i del mirall, i fora d'aqui tot seria una copia girada;
  - un candidat per franja de frontera (15-30, 30-45, 45-60 %): el que reparteix
    mes igual els tres imants (la mes gran de les tres parts mes petites).
  Ordenar per "mes frontera" a seques porta a la pols del sostre (provat amb el
  mapa vell: els quatre primers, al 59-60 %, i tres eren el mateix girat).

COM VAN SORTIR ELS TRES ZOOMS (27/10):
  - zoom1, el de sempre (x -2,1 a 3,9, y -1,5 a 4,5 cm): al mapa general
    compleix la regla (50,8 % de frontera), i es va quedar;
  - zoom2, el candidat B sobre el zoom1 (42,3 % de frontera, al pixel del pare);
  - zoom3, el candidat B sobre el zoom2 (44,2 %).
  El zoom2 vell (x 0,47 a 1,97, y 1,99 a 3,49 cm) queia a la pols: 99,1 % de
  frontera al pixel del zoom1, i NO compleix la regla.

Puja d'esborranys/_tria_zoom_27-10.py el 28/10 amb la sortida identica.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.tria_zoom magnetic/dades/mapa_general.npz ^
        --nom zoom1 --vell -2.1 3.9 -1.5 4.5
    .venv\\Scripts\\python.exe -m magnetic.tria_zoom magnetic/dades/zoom1.npz ^
        --nom zoom2 --vell 0.47 1.97 1.99 3.49
    .venv\\Scripts\\python.exe -m magnetic.tria_zoom magnetic/dades/zoom2.npz --nom zoom3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import dades, estil
from magnetic.analisi import mascara_de_frontera

IMANTS = (1, 2, 3)
MIN_IMANT = 0.05
FRONTERA = (0.15, 0.60)
FRANGES = ((0.15, 0.30), (0.30, 0.45), (0.45, 0.60))
SECTOR_GRAUS = (0.0, 60.0)


def integral(m):
    """Imatge integral amb una fila i una columna de zeros al davant."""
    s = np.zeros((m.shape[0] + 1, m.shape[1] + 1))
    s[1:, 1:] = np.cumsum(np.cumsum(m, axis=0), axis=1)
    return s


def suma(s, j, i, w):
    return s[j + w, i + w] - s[j, i + w] - s[j + w, i] + s[j, i]


def descriu(imant, front, j, i, w):
    tros = imant[j:j + w, i:i + w]
    fr = float(front[j:j + w, i:i + w].mean())
    parts = {c: float(np.mean(tros == c)) for c in IMANTS}
    return fr, parts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pare", help="graella .npz del mapa pare")
    ap.add_argument("--nom", required=True, help="nom del zoom que es proposa")
    ap.add_argument("--vell", type=float, nargs=4, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="una finestra (cm) per mesurar-la al mapa pare amb la regla")
    ap.add_argument("--sortida", default="magnetic/figures", help="on va la figura")
    args = ap.parse_args(argv)

    x0, y0, imant, _, _ = dades.carrega_graella(args.pare)
    n = imant.shape[0]
    pix = x0[1] - x0[0]
    costat = (x0[-1] - x0[0]) / 4
    w = int(round(costat / pix))
    pas = max(1, n // 60)

    front = mascara_de_frontera(imant)
    s_front = integral(front.astype(float))
    s_imant = {c: integral((imant == c).astype(float)) for c in IMANTS}

    files = []
    for j in range(0, n - w + 1, pas):
        for i in range(0, n - w + 1, pas):
            xc = x0[i] + costat / 2
            yc = y0[j] + costat / 2
            angle = np.degrees(np.arctan2(yc, xc))
            if not SECTOR_GRAUS[0] <= angle <= SECTOR_GRAUS[1]:
                continue
            fr = suma(s_front, j, i, w) / w**2
            parts = [suma(s_imant[c], j, i, w) / w**2 for c in IMANTS]
            if FRONTERA[0] <= fr <= FRONTERA[1] and min(parts) >= MIN_IMANT:
                files.append((fr, min(parts), j, i))

    triats = []
    for a, b in FRANGES:
        dins = [f for f in files if a <= f[0] < b or (b == FRONTERA[1] and f[0] == b)]
        if dins:
            fr, _, j, i = max(dins, key=lambda f: f[1])
            triats.append((fr, j, i))

    print(f"pare {Path(args.pare).name}: {n}x{n}, pixel {pix*1000:.4f} mm, "
          f"costat del zoom {costat*100:.4f} cm ({w} px del pare)")
    print(f"requadres que compleixen la regla: {len(files)}")
    print(f"{'':>3} {'x (cm)':>21} {'y (cm)':>21} {'frontera':>9} "
          f"{'imant 1':>8} {'imant 2':>8} {'imant 3':>8}")
    finestres = []
    for k, (fr, j, i) in enumerate(triats):
        xa, ya = x0[i] * 100, y0[j] * 100
        fin = (round(xa, 4), round(xa + costat * 100, 4), round(ya, 4), round(ya + costat * 100, 4))
        finestres.append(fin)
        _, parts = descriu(imant, front, j, i, w)
        print(f"{chr(65+k):>3} {fin[0]:>9.4f} .. {fin[1]:>8.4f} {fin[2]:>9.4f} .. {fin[3]:>8.4f} "
              f"{100*fr:>8.1f}% " + " ".join(f"{100*parts[c]:>7.1f}%" for c in IMANTS))

    if args.vell:
        xa, xb, ya, yb = (v / 100 for v in args.vell)
        i = int(round((xa - x0[0]) / pix))
        j = int(round((ya - y0[0]) / pix))
        wv = int(round((xb - xa) / pix))
        fr, parts = descriu(imant, front, j, i, wv)
        compleix = FRONTERA[0] <= fr <= FRONTERA[1] and min(parts.values()) >= MIN_IMANT
        print(f"vell {args.vell[0]:>8.4f} .. {args.vell[1]:>8.4f} {args.vell[2]:>9.4f} .. "
              f"{args.vell[3]:>8.4f} {100*fr:>8.1f}% "
              + " ".join(f"{100*parts[c]:>7.1f}%" for c in IMANTS)
              + ("   compleix la regla" if compleix else "   NO compleix la regla"))

    estil.aplica()
    cmap, norm = estil.mapa_de_colors()
    ncol = 1 + len(finestres)
    fig, eixos = plt.subplots(1, ncol, figsize=(4.0 * ncol, 4.4))
    ext = [x0[0] * 100, x0[-1] * 100, y0[0] * 100, y0[-1] * 100]
    eixos[0].imshow(imant, cmap=cmap, norm=norm, interpolation="nearest",
                    origin="lower", extent=ext)
    eixos[0].set_title(f"{Path(args.pare).stem}: candidats per al {args.nom}", fontsize=10)
    for k, fin in enumerate(finestres):
        eixos[0].add_patch(plt.Rectangle((fin[0], fin[2]), fin[1] - fin[0], fin[3] - fin[2],
                                         fill=False, lw=1.4, ec="black"))
        eixos[0].text(fin[0], fin[3], f" {chr(65+k)}", va="bottom", fontsize=10,
                      fontweight="bold")
        fr, j, i = triats[k]
        eixos[k + 1].imshow(imant[j:j + w, i:i + w], cmap=cmap, norm=norm,
                            interpolation="nearest", origin="lower",
                            extent=[fin[0], fin[1], fin[2], fin[3]])
        eixos[k + 1].set_title(f"{chr(65+k)}: frontera {100*fr:.0f} %\n"
                               f"(al pixel del pare, {pix*1000:.3f} mm)", fontsize=9)
    if args.vell:
        eixos[0].add_patch(plt.Rectangle((args.vell[0], args.vell[2]),
                                         args.vell[1] - args.vell[0], args.vell[3] - args.vell[2],
                                         fill=False, lw=1.2, ec="black", ls="--"))
        eixos[0].text(args.vell[1], args.vell[2], "vell ", ha="right", va="bottom", fontsize=8)
    for e in eixos:
        e.grid(False)
        e.tick_params(labelsize=7)
    sortida = Path(args.sortida)
    sortida.mkdir(parents=True, exist_ok=True)
    cami = sortida / f"{args.nom}_candidats.png"
    fig.savefig(cami, dpi=130, bbox_inches="tight")
    print(f"figura: {cami}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
