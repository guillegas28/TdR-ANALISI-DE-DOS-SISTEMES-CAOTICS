r"""El seguiment a 60 fps, es de debo mes imprecis que a 240?

Es la mateixa barra 2 i el mateix experiment gravat de les dues maneres, o
sigui que es pot comparar.

QUIN ESTIMADOR I PER QUE. No el de la diferencia quarta dels angles: mata els
polinomis fins a cubic, pero el que queda del moviment real va com (w*dt)^4, i
dt es QUATRE vegades mes gran a 60 fps -256 vegades mes contaminacio-, o sigui
que carregaria contra els 60 fps una cosa que no es soroll.

L'estimador net es el RESIDU RADIAL: la barra es rigida, la marca gira sobre
una circumferencia i la distancia al centre ha de ser constant. Tot el que se
n'aparta es soroll de seguiment, i NO depen de dt ni del moviment.

Es normalitza pel radi (px/R) perque les dues gravacions no tenen el mateix
enquadrament, i es reparteix per velocitat angular perque ja sabem que el
soroll creix amb la velocitat.

Limit conegut, i va a favor dels 60 fps NO: el residu radial es la direccio en
que la marca es localitza PITJOR (problema de l'obertura, la barra es llarga i
prima). El soroll angular, que es el que entra als resultats, es forca mes
petit. Aixo mesura la cota alta, no el que arriba als coeficients.

Era un esborrany (9/09/2026) i va pujar a eines/ el 29/09. La sortida es la mateixa
que la de l'esborrany, comprovat en pujar-lo. S'executa des de l'arrel:

    .venv\Scripts\python.exe eines\soroll_60_contra_240.py
"""

import math
import sys

import numpy as np

from pathlib import Path

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
from friccio_pendol_doble import ajusta_circumferencia  # noqa: E402
from pendol import trk as T  # noqa: E402

PRESES = [
    ("IMG_3075  (240 fps, agost)", "dades/video/tracker/Fricció_doble_2.trk"),
    ("Friccio_2.1  (60 fps)", "dades/video/tracker/fricció_doble/Friccio_2.1.trk"),
    ("Friccio_2.2  (60 fps)", "dades/video/tracker/fricció_doble/Friccio_2.2.trk"),
    ("Friccio_2.3  (60 fps)", "dades/video/tracker/fricció_doble/Friccio_2.3.trk"),
]

TALLS = [(0.0, 2.0), (2.0, 6.0), (6.0, 12.0), (12.0, 30.0)]   # rad/s


def mesura(cami):
    d = T.llegeix_trk(cami)
    (k, X, Y), = d["pistes"].values()
    tall = np.where(np.diff(k) > 1)[0]
    tria = max(np.split(np.arange(len(k)), tall + 1), key=len)
    k, X, Y = k[tria], X[tria], Y[tria]

    xc, yc, R, _ = ajusta_circumferencia(X, Y)
    x, y = X - xc, Y - yc
    r = np.hypot(x, y)
    th = np.unwrap(np.arctan2(y, x))
    dt = d["delta_t_s"]
    # velocitat nomes per repartir per trams: una diferencia central hi va be
    w = np.abs(np.gradient(th, dt))

    return {"R": R, "dt": dt, "fps": 1.0 / dt, "r": r, "w": w, "n": len(k)}


def main():
    print("Residu radial del seguiment: la barra es rigida, o sigui que tot")
    print("el que se n'aparta es soroll.\n")
    print(f"  {'presa':<28}{'fps':>6}{'R (px)':>9}{'n':>7}   "
          + "".join(f"{f'{a:.0f}-{b:.0f} rad/s':>13}" for a, b in TALLS))
    print(f"  {'':<28}{'':>6}{'':>9}{'':>7}   "
          + "".join(f"{'px':>13}" for _ in TALLS))

    fila_norm = []
    for nom, cami in PRESES:
        m = mesura(cami)
        cel = []
        for a, b in TALLS:
            sel = (m["w"] >= a) & (m["w"] < b)
            # el radi decau? no: la barra es rigida. Es treu la mitjana del
            # tram i es mira la dispersio del que queda.
            cel.append(float(np.std(m["r"][sel])) if sel.sum() > 30 else math.nan)
        print(f"  {nom:<28}{m['fps']:>6.0f}{m['R']:>9.1f}{m['n']:>7}   "
              + "".join(f"{c:>13.2f}" if c == c else f"{'-':>13}" for c in cel))
        fila_norm.append((nom, m["R"], cel))

    print("\n  el mateix en TANT PER MIL DEL RADI, que es el que es pot comparar")
    print("  entre dos enquadraments diferents:\n")
    print(f"  {'presa':<28}   "
          + "".join(f"{f'{a:.0f}-{b:.0f} rad/s':>13}" for a, b in TALLS))
    for nom, R, cel in fila_norm:
        print(f"  {nom:<28}   "
              + "".join(f"{1000 * c / R:>13.2f}" if c == c else f"{'-':>13}"
                        for c in cel))

    print("\n  i quant val aixo en GRAUS d'angle, si tot el soroll radial es")
    print("  traduis a angle (que no ho fa: es la cota mes alta possible):\n")
    print(f"  {'presa':<28}   "
          + "".join(f"{f'{a:.0f}-{b:.0f} rad/s':>13}" for a, b in TALLS))
    for nom, R, cel in fila_norm:
        print(f"  {nom:<28}   "
              + "".join(f"{math.degrees(c / R):>13.3f}" if c == c else f"{'-':>13}"
                        for c in cel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
