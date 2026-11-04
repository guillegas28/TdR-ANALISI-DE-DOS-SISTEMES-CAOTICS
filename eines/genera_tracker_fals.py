"""Genera un fals export del Tracker per provar tot el circuit sense video.

Simula un tir, el converteix a posicions (x, y) com les que et donaria el
Tracker, hi afegeix soroll de pixel i algun fotograma perdut, i ho escriu
amb el format mes lleig possible (separador punt i coma, decimals amb coma,
linies de metadades al davant) per comprovar que pendol/tracker.py se'n surt.

IMPORTANT: fixa't que el tir "real" fals surt d'un angle LLEUGERAMENT
diferent del nominal (--desviacio). Aixo imita el fet que a la vida real no
pots deixar anar el pendol exactament als 120 graus. Es el que fa que la
comparacio tingui sentit.

Us:
    .venv\\Scripts\\python eines\\genera_tracker_fals.py --theta1 120 \\
        --sortida dades/cru/prova.csv

Amb --quiet el tir surt amb el pendol QUIET un tros abans d'amollar-lo, que es
com son totes les preses de debo. Es el que cal per provar el circuit sencer
fins a pendol/amollada.py, que ha de trobar aquell instant:

    .venv\\Scripts\\python eines\\genera_tracker_fals.py --theta1 120 --quiet 0.8 \\
        --sortida esborranys/prova_amollada.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pendol import amollada  # noqa: E402
from pendol.fisica import Parametres, posicions  # noqa: E402
from pendol.simulador import simula  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parametres", default="parametres.json")
    ap.add_argument("--theta1", type=float, default=120.0, help="angle nominal (graus)")
    ap.add_argument("--theta2", type=float, default=0.0)
    ap.add_argument("--desviacio", type=float, default=0.7,
                    help="quant s'aparta el tir 'real' del nominal (graus)")
    ap.add_argument("--temps", type=float, default=15.0)
    ap.add_argument("--fps", type=float, default=240.0)
    ap.add_argument("--soroll", type=float, default=0.0005, help="soroll de posicio (m), ~1 pixel")
    ap.add_argument("--perduts", type=int, default=6, help="fotogrames en que el Tracker perd el punt")
    ap.add_argument("--friccio", type=float, default=0.0008,
                    help="b1 i b2 si parametres.json encara els te a zero (N*m*s/rad)")
    ap.add_argument("--llavor", type=int, default=7)
    ap.add_argument("--quiet", type=float, default=0.0,
                    help="segons de pendol QUIET abans d'amollar-lo. Les preses de "
                         "debo comencen aixi -el marcatge del video s'engega abans "
                         "que la ma deixi anar el pendol- i entre l'un i l'altre hi "
                         "ha de 0,05 a 2 s. Sense aixo, el tir fals comenca ja en "
                         "marxa i no serveix per provar pendol/amollada.py")
    ap.add_argument("--sortida", required=True)
    args = ap.parse_args(argv)

    p = Parametres.des_de_json(args.parametres)
    # El model d'ara no te friccio (vegeu pendol/fisica.py), o sigui que el tir
    # fals surt conservatiu. Quan el pas 3 hi posi els cinc termes, --friccio
    # tornara a voler dir alguna cosa.
    if args.friccio:
        print("AVIS: --friccio no fa res mentre el model no tingui termes "
              "dissipatius. El tir sortira sense friccio.")

    rng = np.random.default_rng(args.llavor)
    estat0 = np.radians([args.theta1 + args.desviacio, args.theta2, 0.0, 0.0])
    if args.quiet > 0:
        # Una sola implementacio del tir amb repos, la de pendol/amollada.py,
        # que es qui l'ha de saber trobar. Si algun dia canvia, canvia per als
        # dos costats alhora.
        t, th1, th2 = amollada.tir_fals_amb_repos(p, estat0, args.quiet, args.temps, args.fps)
    else:
        t, y = simula(p, estat0, args.temps, args.fps)
        th1, th2 = y[0], y[1]

    x1, y1, x2, y2 = posicions(th1, th2, p)
    for v in (x1, y1, x2, y2):
        v += rng.normal(0.0, args.soroll, v.shape)

    # Fotogrames on el Tracker "perd" el segon punt.
    perduts = rng.choice(np.arange(10, len(t) - 10), size=args.perduts, replace=False)

    cami = Path(args.sortida)
    cami.parent.mkdir(parents=True, exist_ok=True)
    linies = [
        "Tracker 6.2.0",
        f"video: prova_{args.theta1:.0f}graus.mp4",
        f"escala: 1 m = 640 px;  fps = {args.fps:.1f}",
        "",
        "t;x_A;y_A;x_B;y_B",
    ]

    def cat(v: float) -> str:
        return f"{v:.6f}".replace(".", ",")

    for i in range(len(t)):
        if i in perduts:
            linies.append(f"{cat(t[i])};{cat(x1[i])};{cat(y1[i])};;")
        else:
            linies.append(f"{cat(t[i])};{cat(x1[i])};{cat(y1[i])};{cat(x2[i])};{cat(y2[i])}")

    cami.write_text("\n".join(linies), encoding="utf-8")
    print(f"Escrit {cami}  ({len(t)} fotogrames a {args.fps:.0f} fps)")
    print(f"  angle real del tir: {args.theta1 + args.desviacio:.2f} graus (nominal {args.theta1:.0f})")
    print(f"  muntatge: I1 = {p.I1:.4e}, I2 = {p.I2:.4e}, sense friccio")
    print(f"  separador ';', decimals amb coma, {args.perduts} fotogrames perduts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
