r"""Seccions de Poincare del pendol doble: simulacio CONSERVATIVA, sense friccio.

Per que existeix. L'espai de fases del pendol doble te quatre dimensions i no
es pot dibuixar. A energia fixa en queden tres, i la seccio en talla una mes:
cada vegada que el segon brac passa per la vertical de sota girant en sentit
positiu (th2 = 0, omega2 > 0) s'apunta (th1, omega1). El que queda es un dibuix
de 2D que es una representacio FIDEL de la dinamica a aquella energia: corbes
tancades on el moviment es regular, i un nuvol de punts (el mar caotic) on no
ho es (documentacio/tallar-espai-fases.html).

Pla: documentacio/pla-analisi-caos.md, pas 4, nomes la part conservativa. Tot
el calcul es de pendol/caos.py (condicions_a_la_seccio, seccio_pendol), que te
la seva autoprova (P1-P4). Aquesta eina reparteix les orbites, en desa els
punts, comprova l'energia i dibuixa.

QUE NO FA, encara:
  - la seccio AMB FRICCIO: no hi ha coeficients decidits;
  - el pendol simple, el control: falta decidir on va el seu model;
  - la seccio del pendol REAL: toca dades reals, i va despres de preguntar-ho.
    Posar-la al costat d'una de simulada, a mes, es comparar simulacio i
    realitat, i aixo va despres de firmar X, Y i Z.

LES ORBITES SURTEN DE LA SECCIO, NO DEL REPOS. Les orbites amollades del repos
a una energia donada caurien totes al mateix tros; repartint-les per la seccio
se'n veuen les illes i el mar.

LES ENERGIES, per defecte, son les medianes de l'energia inicial mesurada de
cada grup (45, 90 i 150 graus), llegides NOMES de dades/net/preses.json amb la
mateixa funcio que eines/lyapunov.py.

Us:
    .venv\\Scripts\\python eines\\poincare.py --figures figures
    .venv\\Scripts\\python eines\\poincare.py --energies 0.94,3.44 --orbites 10 --temps 100
    .venv\\Scripts\\python eines\\poincare.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol.caos import (condicions_a_la_seccio, energia_relativa,  # noqa: E402
                         energia_volteig_2, seccio_pendol)
from pendol.fisica import Parametres  # noqa: E402
from pendol.simulador import ATOL, METODE, RTOL  # noqa: E402

import peus  # noqa: E402

INDEX = ARREL / "dades" / "net" / "preses.json"
ORBITES = 30
TEMPS_S = 300.0
LLAVOR = 20261001


def llavor_de(rao: float) -> int:
    return LLAVOR + zlib.crc32(f"{rao:g}".encode("utf-8"))


def una_orbita(args) -> dict:
    """Una orbita: els seus creuaments, en un proces a part."""
    estat, temps = args
    p = Parametres.des_de_json(ARREL / "parametres.json")
    s = seccio_pendol(p, np.asarray(estat, dtype=float), temps)
    E_rel0 = float(energia_relativa(np.asarray(estat, dtype=float), p))
    deriva = (float(np.max(np.abs(s["energia_J"] - s["energia_inicial_J"]))) / E_rel0
              if len(s["energia_J"]) else 0.0)
    return {"theta1": s["theta1_plegat"], "omega1": s["omega1"], "deriva": deriva,
            "salt": s["salt_maxim_rad"], "estats": s["estats"]}


def calcula_energia(p, rao: float, orbites: int, temps: float, processos: int) -> dict:
    E = rao * energia_volteig_2(p)
    ci = condicions_a_la_seccio(E, p, orbites, llavor=llavor_de(rao))
    feines = [(c.tolist(), temps) for c in ci]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            res = pool.map(una_orbita, feines)
    else:
        res = [una_orbita(f) for f in feines]
    return {"rao": rao, "E_J": E, "condicions": ci, "orbites": res}


def desa_i_dibuixa(r: dict, temps: float, figures: Path) -> dict:
    import matplotlib.pyplot as plt

    from pendol import estil

    th = np.concatenate([o["theta1"] for o in r["orbites"]])
    w = np.concatenate([o["omega1"] for o in r["orbites"]])
    nom = f"poincare_E{r['rao']:g}".replace(".", "_")
    metriques = {
        "eina": "eines/poincare.py",
        "que_es": "seccio de Poincare de la simulacio CONSERVATIVA, sense friccio: "
                  "th2 = 0 (vertical de sota) amb omega2 > 0; es registra (th1 plegat, omega1)",
        "energia_rao_E_volteig_2": r["rao"],
        "energia_J": r["E_J"],
        "orbites": len(r["orbites"]),
        "temps_per_orbita_s": temps,
        "punts": int(len(th)),
        "punts_per_orbita": [int(len(o["theta1"])) for o in r["orbites"]],
        "deriva_energia_relativa_max": max(o["deriva"] for o in r["orbites"]),
        "pas_mes_gran_de_th2_rad": max(o["salt"] for o in r["orbites"]),
        "llavor": llavor_de(r["rao"]),
        "integrador": {"metode": METODE, "rtol": RTOL, "atol": ATOL},
        "condicions_inicials": r["condicions"].tolist(),
    }
    figures.mkdir(parents=True, exist_ok=True)
    (figures / f"{nom}.json").write_text(json.dumps(metriques, indent=1, ensure_ascii=False)
                                         + "\n", encoding="utf-8")
    np.savez_compressed(figures / f"{nom}.npz", theta1=th, omega1=w)

    estil.aplica()
    fig, eix = plt.subplots(figsize=(7.2, 5.6))
    eix.scatter(np.degrees(th), w, s=0.6, color=estil.SIMULACIO, alpha=0.55, lw=0,
                rasterized=True)
    eix.set_xlim(-180, 180)
    eix.set_xticks(range(-180, 181, 60))
    eix.set_xlabel(r"$\theta_1$  (graus, plegat)")
    eix.set_ylabel(r"$\omega_1$  (rad/s)")
    eix.set_title(f"Secció de Poincaré a E = {r['rao']:g}".replace(".", ",") + " E_volteig,2 — simulació "
                  "conservativa", loc="left", pad=10)
    fig.tight_layout()
    peus.peu(fig, 0.01, -0.01,
             f"{len(r['orbites'])} òrbites des de punts de la secció, {temps:g} s cadascuna, "
             f"{len(th)} punts. Condició: θ2 = 0 amb ω2 > 0. Deriva d'energia màxima "
             f"{metriques['deriva_energia_relativa_max']:.0e}.",
             fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    peus.desa(fig, nom, figures)
    return metriques


# --------------------------------------------------------------- l'autoprova

def autoprova() -> int:
    """Q1-Q3, fixats el 01/10/2026 abans d'executar-la.

    Q1  els punts de sortida: l'energia demanada a 1e-12 (amb fisica.energia),
        th2 = 0 i omega2 > 0, a tres energies
    Q2  una orbita de 50 s a cada energia: tots els creuaments compleixen la
        condicio i conserven l'energia a 1e-9
    Q3  dues passades amb la mateixa llavor donen els mateixos punts, bit a bit
    """
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json(ARREL / "parametres.json")
    Ev2 = energia_volteig_2(p)
    raons = (0.94, 3.44, 6.23)
    print("Autoprova de eines/poincare.py (criteris fixats el 01/10/2026)\n")

    print("Q1. Els punts de sortida son a la seccio i a l'energia demanada")
    for rao in raons:
        ci = condicions_a_la_seccio(rao * Ev2, p, 200, llavor=llavor_de(rao))
        err = float(np.max(np.abs(energia_relativa(ci.T, p) / (rao * Ev2) - 1)))
        ok = err < 1e-12 and np.all(ci[:, 1] == 0.0) and np.all(ci[:, 3] > 0)
        mira(ok, f"E = {rao} E_v2: 200 punts, error d'energia {err:.1e}, th2 = 0, omega2 > 0")

    print("\nQ2. Una orbita de 50 s a cada energia: la condicio i l'energia")
    for rao in raons:
        ci = condicions_a_la_seccio(rao * Ev2, p, 1, llavor=llavor_de(rao))
        o = una_orbita((ci[0].tolist(), 50.0))
        e = o["estats"]
        condicio = bool(len(e) > 0 and np.all(np.abs(np.sin(e[:, 1])) < 1e-9)
                        and np.all(np.cos(e[:, 1]) > 0) and np.all(e[:, 3] > 0))
        mira(condicio and o["deriva"] < 1e-9,
             f"E = {rao} E_v2: {len(e)} creuaments, deriva d'energia {o['deriva']:.1e}")

    print("\nQ3. Dues passades amb la mateixa llavor donen el mateix, bit a bit")
    a = calcula_energia(p, 3.44, 2, 20.0, 1)
    b = calcula_energia(p, 3.44, 2, 20.0, 1)
    iguals = all(np.array_equal(x["theta1"], y["theta1"]) and np.array_equal(x["omega1"], y["omega1"])
                 for x, y in zip(a["orbites"], b["orbites"]))
    mira(iguals, "els mateixos punts de sortida i els mateixos creuaments")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> els punts de sortida son a la seccio i a l'energia, l'energia es conserva")
    print("     a cada creuament, i el calcul es reproduible. La seccio mateixa (condicio,")
    print("     voltes desembolicades, interpolacio) la prova pendol/caos.py, P1-P4.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--energies", help="E/E_volteig,2 separades per comes (per defecte, la "
                                       "mediana de l'energia inicial de cada grup)")
    ap.add_argument("--orbites", type=int, default=ORBITES)
    ap.add_argument("--temps", type=float, default=TEMPS_S, help="segons per orbita")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--processos", type=int, default=1, help="com a molt 6")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a <figures>/memoria/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if not 1 <= args.processos <= 6:
        ap.error("--processos ha d'anar d'1 a 6")
    if args.memoria:
        peus.memoria()
        args.figures = str(Path(args.figures) / "memoria")

    p = Parametres.des_de_json(ARREL / "parametres.json")
    if args.energies:
        raons = [float(x) for x in args.energies.split(",")]
    else:
        from lyapunov import energies_dels_grups

        grups = energies_dels_grups(INDEX, p)
        raons = [round(v["mediana"], 2) for v in grups.values()]
        print("Energies dels grups (mediana, E/E_v2): "
              + ", ".join(f"{g} graus {round(v['mediana'], 2)}" for g, v in grups.items()))
    for rao in raons:
        ara = time.perf_counter()
        r = calcula_energia(p, rao, args.orbites, args.temps, args.processos)
        m = desa_i_dibuixa(r, args.temps, Path(args.figures))
        print(f"E = {rao:g} E_v2: {m['punts']} punts de {m['orbites']} orbites, deriva "
              f"d'energia {m['deriva_energia_relativa_max']:.1e} "
              f"({time.perf_counter() - ara:.0f} s)", flush=True)
    peus.escriu(args.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
