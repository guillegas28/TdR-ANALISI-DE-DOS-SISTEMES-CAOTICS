"""Animacio del pendol doble: converteix una trajectoria canonica en moviment.

Per que serveix
---------------
Les figures de pendol/comparar.py son la prova quantitativa, pero no es veuen.
Aquest modul dona la part que s'enten d'un cop d'ull: el pendol movent-se de
debo. Com que la simulacio i el Tracker comparteixen el format canonic, el
mateix codi serveix per veure la simulacio i per veure el TEU pendol real
reconstruit a partir del video.

El cas que de debo explica el treball es el de DOS fitxers alhora: es dibuixen
els dos pendols superposats, real i simulat, i es veu com van junts uns segons
i despres se separen. Aixo es el caos, sense haver de llegir cap grafica.

Us
--
Un sol tir, en una finestra:

    .venv\\Scripts\\python -m pendol.anima dades/sim/feix_120graus/tir0000.csv

Real i simulat superposats, desats en un GIF:

    .venv\\Scripts\\python -m pendol.anima dades/net/tir01.csv \\
        dades/sim/feix_120graus/tir0000.csv --sortida figures/tir01_animacio.gif

A mig gas, per veure be el moment en que se separen:

    .venv\\Scripts\\python -m pendol.anima ... --velocitat 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from pendol import dades, estil
from pendol.fisica import Parametres, posicions


def _carrega_tir(cami, per_defecte: Parametres | None):
    """Llegeix un CSV canonic i resol amb quins parametres s'ha de dibuixar.

    Els CSV de simulacio ja porten els parametres al seu JSON de metadades, o
    sigui que es dibuixen amb les longituds amb que es van generar. Els del
    Tracker no en porten (surten d'un video, no d'un model), i per a aquests cal
    el fitxer de parametres.
    """
    taula, meta = dades.carrega(cami)
    if "parametres" in meta:
        # com_a_dict() hi afegeix "_derivats" (les inercies amb que es va
        # generar el tir), que no son camps del dataclass: es treuen abans de
        # reconstruir-lo. I si les metadades son del model puntual d'abans del
        # 22/09/2026, hi haura camps que ja no existeixen; val mes dir-ho que
        # petar amb un TypeError.
        camps = {k: v for k, v in meta["parametres"].items() if not k.startswith("_")}
        try:
            p = Parametres(**camps)
        except TypeError as e:
            raise SystemExit(
                f"{cami} es d'abans que el model passes a pendol fisic "
                f"(22/09/2026) i les seves metadades ja no encaixen: {e}\n"
                "Torna a generar el tir amb pendol/simulador.py."
            ) from e
    elif per_defecte is not None:
        p = per_defecte
    else:
        raise SystemExit(
            f"{Path(cami).name} no porta parametres a les metadades i no he trobat "
            "el fitxer de parametres. Passa'l amb --parametres."
        )

    # El taronja es sempre el pendol real i el blau la simulacio, com a totes
    # les figures del treball (veure pendol/estil.py).
    real = meta.get("origen") == "tracker"
    return {
        "t": taula["t"].to_numpy(),
        "theta1": taula["theta1"].to_numpy(),
        "theta2": taula["theta2"].to_numpy(),
        "p": p,
        "color": estil.REAL if real else estil.SIMULACIO,
        "trac": estil.TRAC_REAL if real else estil.TRAC_SIMULACIO,
        "etiqueta": "pendol real (Tracker)" if real else "simulacio",
    }


def prepara(tirs, fps: float, velocitat: float, temps: float | None):
    """Remostreja tots els tirs sobre la base temporal de l'animacio.

    Els CSV estan a 200 mostres per segon i una animacio en va a 25: cal
    submostrejar. Interpolar es segur perque els angles es guarden
    desembolicats (veure el contracte de dades a pendol/dades.py); si estiguessin
    plegats a [-180, 180] cada volta completa donaria un salt fals.
    """
    durada = min(float(tir["t"][-1]) for tir in tirs)
    if temps is not None:
        durada = min(durada, temps)

    # velocitat = segons simulats per segon de reproduccio.
    t_anim = np.arange(0.0, durada + 1e-9, velocitat / fps)
    for tir in tirs:
        th1 = np.interp(t_anim, tir["t"], tir["theta1"])
        th2 = np.interp(t_anim, tir["t"], tir["theta2"])
        tir["x1"], tir["y1"], tir["x2"], tir["y2"] = posicions(th1, th2, tir["p"])
    return t_anim


def _segments_traca(x, y, k, n_traca, color):
    """Trossos de la traca de la segona massa, amb l'opacitat que es va apagant."""
    i0 = max(0, k - n_traca)
    punts = np.column_stack([x[i0 : k + 1], y[i0 : k + 1]])
    if len(punts) < 2:
        return np.empty((0, 2, 2)), np.empty((0, 4))
    segments = np.stack([punts[:-1], punts[1:]], axis=1)
    colors = np.tile(to_rgba(color), (len(segments), 1))
    colors[:, 3] = np.linspace(0.0, 0.55, len(segments))
    return segments, colors


def construeix(tirs, t_anim, fps: float, velocitat: float, traca: float):
    """Munta la figura i retorna (figura, funcio de fotograma, nombre de fotogrames)."""
    # Fins on pot arribar el gomet: del pivot 1 al pivot 2, i d'alli al gomet.
    abast = max(tir["p"].L1 + tir["p"].r_gomet for tir in tirs) * 1.08

    fig, eix = plt.subplots(figsize=(5.4, 5.4))
    eix.set_aspect("equal")
    eix.set_xlim(-abast, abast)
    eix.set_ylim(-abast, abast)
    eix.set_xlabel("x (m)")
    eix.set_ylabel("y (m)")

    # Pivot fix: el punt de referencia de tot el sistema.
    eix.plot([0], [0], marker="o", markersize=5, color=estil.TINTA_2, zorder=5)

    n_traca = max(1, int(round(traca * fps / velocitat)))
    for tir in tirs:
        tir["traca_art"] = LineCollection([], linewidths=1.3, zorder=2)
        eix.add_collection(tir["traca_art"])
        (tir["brac_art"],) = eix.plot(
            [], [], color=tir["color"], linestyle=tir["trac"], linewidth=1.9,
            marker="o", markersize=7, markerfacecolor=tir["color"],
            markeredgecolor="none", zorder=3, label=tir["etiqueta"],
        )

    rellotge = eix.text(
        0.03, 0.96, "", transform=eix.transAxes, va="top",
        color=estil.TINTA_2, fontsize=10,
    )
    if len(tirs) > 1:
        eix.legend(loc="upper right", fontsize=9)

    def fotograma(k):
        artistes = [rellotge]
        for tir in tirs:
            tir["brac_art"].set_data(
                [0.0, tir["x1"][k], tir["x2"][k]], [0.0, tir["y1"][k], tir["y2"][k]]
            )
            segments, colors = _segments_traca(tir["x2"], tir["y2"], k, n_traca, tir["color"])
            tir["traca_art"].set_segments(segments)
            if len(segments):
                tir["traca_art"].set_color(colors)
            artistes += [tir["brac_art"], tir["traca_art"]]
        rellotge.set_text(f"t = {t_anim[k]:5.2f} s")
        return artistes

    return fig, fotograma, len(t_anim)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fitxers", nargs="+", help="un o dos CSV en format canonic")
    ap.add_argument("--parametres", default="parametres.json",
                    help="JSON de parametres, per als tirs que no en portin (els del Tracker)")
    ap.add_argument("--sortida", help="desa un GIF en comptes d'obrir una finestra")
    ap.add_argument("--fps", type=float, default=25.0, help="fotogrames per segon de l'animacio")
    ap.add_argument("--velocitat", type=float, default=1.0,
                    help="segons simulats per segon de reproduccio (1 = temps real)")
    ap.add_argument("--traca", type=float, default=2.0,
                    help="segons de rastre que deixa la segona massa (0 = cap)")
    ap.add_argument("--temps", type=float, help="retalla l'animacio a aquests segons")
    args = ap.parse_args(argv)

    if len(args.fitxers) > 2:
        print("ERROR: com a molt dos fitxers. Amb mes, la figura deixa de dir res.")
        return 1

    estil.aplica()

    cami_p = Path(args.parametres)
    per_defecte = Parametres.des_de_json(cami_p) if cami_p.exists() else None
    tirs = [_carrega_tir(f, per_defecte) for f in args.fitxers]

    t_anim = prepara(tirs, args.fps, args.velocitat, args.temps)
    fig, fotograma, n = construeix(tirs, t_anim, args.fps, args.velocitat, max(args.traca, 0.0))
    print(f"{n} fotogrames, {t_anim[-1]:.2f} s simulats a {args.velocitat}x")

    animacio = FuncAnimation(
        fig, fotograma, frames=n, interval=1000.0 / args.fps,
        blit=True, cache_frame_data=False,
    )

    if args.sortida:
        sortida = Path(args.sortida)
        sortida.parent.mkdir(parents=True, exist_ok=True)
        animacio.save(sortida, writer=PillowWriter(fps=args.fps), dpi=90)
        plt.close(fig)
        print(f"Desat: {sortida}  ({sortida.stat().st_size / 1e6:.1f} MB)")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
