"""L'area de la regio negra en funcio de la friccio.

Amb friccio hi ha QUATRE destins i no tres: una trajectoria que perd l'energia
abans d'entrar a l'orbita d'un imant s'atura al centre, que es l'equilibri
estable de la gravetat. Al mapa aixo es el negre.

La corba de l'area negra en funcio de b surt del mateix codi del mapa sense
escriure'n de nou: nomes cal tornar a calcular la graella canviant la b.

LA HISTORIA D'AQUESTA CORBA, PERQUE JA S'HI HAN ESCRIT DUES FORMES FALSES:

  1. Abans de mesurar-ho hi deia que era CREIXENT (com mes friccio, mes aviat es
     mor la massa). La mesura ho va desmentir.
  2. Amb la llei de forca INVENTADA (n = 3, K = 1e-4) la corba tenia forma d'U:
     baixava fins a un minim (cap a b = 0,9 a +-12 cm i cap a b = 0,6 a +-6 cm)
     i tornava a pujar. Aqui s'hi va escriure aixo, amb una hipotesi de dos
     mecanismes que mai no es va comprovar.
  3. Amb la llei MESURADA (n = 6,87, K = 1,959e-10; 27-28/10) ja no hi ha cap
     minim pel mig (escombrada de 0,02 a 2, temps maxim 500 s, 0,00 % de gris):
     - a +-12 cm (160x160), gairebe plana fins a b = 0,1 (1,22-1,45 %) i despres
       puja fins a l'11,56 % a b = 2. El minim es a la b mes baixa. No es pot
       dir que "creixi sempre": entre 0,039 i 0,05 fa un pas enrere de 0,04
       punts (10 pixels de 25.600);
     - a +-6 cm (200x200), puja del 3,32 % fins al 9,87 % a b = 1,3 i baixa al
       8,97 % a b = 2.
     La frontera baixa sempre, a totes dues finestres.

Per aixo els titols de la figura no diuen cap forma: els numeros es treuen de
les dades. I la forma depen de la llei de forca, o sigui que no es una propietat
del pendol magnetic en general sino d'aquest muntatge.

Es un resultat del treball, no un detall d'implementacio, i te una gracia
afegida: es un experiment que NOMES es pot fer amb la simulacio. Al laboratori
no pots canviar la friccio a voluntat sense canviar mitja altra cosa.

El mateix calcul dona de regal una segona corba: quants punts queden a la
frontera, es a dir quina part del mapa es impredictible. Es la que explica per
que la corona de fora surt com surt.

Us, des de l'ARREL del projecte (les dues de la memoria, 28/10):

    .venv\\Scripts\\python.exe -m magnetic.regio_negra --temps 500
    .venv\\Scripts\\python.exe -m magnetic.regio_negra --finestra -6 6 -6 6 ^
        --resolucio 200 --temps 500 --nom regio_negra_nucli

Amb la b mes baixa (0,02) l'amplitud cau a 1/e en 100 s: per aixo el temps
maxim de 500 s. Pujar-lo no canvia cap pixel que ja convergia: els nou punts
de b >= 0,05 surten identics als de 200 s.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetic import compila, estil
from magnetic.analisi import mascara_de_frontera
from magnetic.fisica import Parametres, avis_inventats
from magnetic.mapa import calcula
from magnetic.motor import DT_PER_DEFECTE, Criteri

FRICCIONS = (0.02, 0.03, 0.039, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.90, 1.30, 2.00)


def figura(files, p0, args):
    """Les dues corbes.

    ATENCIO ALS TITOLS D'AQUESTA FIGURA. Ja s'hi han escrit dues formes que les
    dades van desmentir: "creix amb la friccio" (abans de mesurar-ho) i "baixa,
    toca fons i torna a pujar" (la U de la llei inventada, n = 3). Per aixo els
    titols no diuen cap forma: el minim, el maxim i els extrems es treuen de les
    dades, i no poden tornar a quedar desaparellats.

    L'eix de la b es logaritmic: l'escombrada va de 0,02 a 2, dues decades, i
    en lineal els punts de friccio baixa (la mesurada inclosa) quedaven apilats.
    """
    ordre = np.argsort([f["b"] for f in files])
    b = np.array([files[i]["b"] for i in ordre])
    negre = np.array([files[i]["negre"] for i in ordre])
    front = np.array([files[i]["frontera"] for i in ordre])
    kmin, kmax = int(np.argmin(negre)), int(np.argmax(negre))

    estil.aplica()
    fig, eixos = plt.subplots(1, 2, figsize=(10.4, 4.4))

    eixos[0].plot(b, negre, "o-", color=estil.TINTA, ms=5)
    # El minim i el maxim, encerclats, i els seus valors al titol: escrits al
    # costat del punt trepitjaven la corba.
    for k in (kmin, kmax):
        eixos[0].plot(b[k], negre[k], "o", ms=11, mfc="none", mew=1.6,
                      color=estil.IMANTS[0])
    eixos[0].set_ylabel("part del mapa que acaba al centre (%)")
    eixos[0].set_title(f"Part del mapa que acaba al centre:\n"
                       f"mínim {negre[kmin]:.2f} % a b = {b[kmin]:g}, "
                       f"màxim {negre[kmax]:.2f} % a b = {b[kmax]:g}")

    eixos[1].plot(b, front, "s-", color=estil.IMANTS[1], ms=5)
    eixos[1].set_ylabel("part del mapa que es frontera (%)")
    eixos[1].set_title(f"Part del mapa que es frontera:\n"
                       f"{front[0]:.0f} % a b = {b[0]:g}, {front[-1]:.0f} % a b = {b[-1]:g}")

    etiqueta = "inventada" if "b" in p0.inventats else "mesurada"
    for eix, valors in zip(eixos, (negre, front)):
        eix.set_xscale("log")
        eix.set_xlabel("friccio b (s$^{-1}$, escala logaritmica)")
        # L'anotacio de la b va on NO hi ha dades: a dalt si la corba passa per
        # la meitat de baix a l'altura de la b, i a baix si passa per la de dalt.
        eix.axvline(p0.b, color=estil.TINTA_SUAU, lw=1, ls=(0, (4, 3)))
        baix, dalt = eix.get_ylim()
        a_la_b = np.interp(np.log(p0.b), np.log(b), valors)
        a_dalt = a_la_b < (baix + dalt) / 2
        eix.annotate(f"b = {p0.b:g}\n({etiqueta})", (p0.b, dalt if a_dalt else baix),
                     textcoords="offset points", xytext=(6, -8 if a_dalt else 8),
                     va="top" if a_dalt else "bottom", fontsize=8,
                     color=estil.TINTA_SUAU)

    fig.suptitle(f"Finestra {args.finestra[0]:g} a {args.finestra[1]:g} cm, "
                 f"{args.resolucio}x{args.resolucio} px, "
                 f"temps maxim {args.temps:g} s", fontsize=10)
    fig.tight_layout()
    estil.peu_de_valors_inventats(fig, p0.inventats)
    estil.desa(fig, args.nom, args.figures)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--finestra", type=float, nargs=4, default=[-12, 12, -12, 12],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="limits en CENTIMETRES")
    ap.add_argument("--resolucio", type=int, default=160, help="pixels per costat")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE, help="pas d'integracio (s)")
    ap.add_argument("--temps", type=float, default=90.0,
                    help="temps maxim (s). Amb friccio baixa cal que sigui llarg")
    ap.add_argument("--friccions", type=float, nargs="+", default=list(FRICCIONS),
                    help="valors de b a provar (s^-1)")
    ap.add_argument("--nom", default="regio_negra", help="nom base dels fitxers")
    ap.add_argument("--figures", default="magnetic/figures", help="on van els resultats")
    # Refer NOMES el dibuix, a partir del JSON que ja hi ha. Els numeros no
    # canvien -- son els mateixos que ja estan desats -- i estalvia tornar a
    # calcular nou mapes de 160x160 amb t_max de 120 s, que son un quart d'hora,
    # nomes per canviar un titol.
    ap.add_argument("--refes-figura", action="store_true",
                    help="no calculis res: redibuixa a partir del JSON ja desat")
    args = ap.parse_args(argv)

    p0 = Parametres.des_de_json()
    criteri = Criteri(t_max=args.temps)
    finestra = [v / 100.0 for v in args.finestra]

    print("=" * 72)
    print(" magnetic.regio_negra -- l'area negra en funcio de la friccio")
    print("=" * 72)

    if args.refes_figura:
        cami = Path(args.figures) / f"{args.nom}.json"
        if not cami.exists():
            print(f"   No hi ha {cami}: no hi ha res per redibuixar.")
            print("   Executa-ho sense --refes-figura per calcular-ho.")
            return 1
        desat = json.loads(cami.read_text(encoding="utf-8"))
        # La finestra i la resolucio han de sortir del JSON i NO de les banderes:
        # la figura ha de dir de quin calcul es, no del que s'hagi escrit ara.
        args.finestra = desat["finestra_cm"]
        args.resolucio = desat["resolucio"]
        args.temps = desat["temps_maxim_s"]
        print(f"   Redibuixant a partir de {cami}")
        print(f"   finestra {args.finestra} cm, {args.resolucio}x{args.resolucio} px, "
              f"{len(desat['files'])} valors de b")
        print("   NO es recalcula res: els numeros son els que ja hi havia.")
        figura(desat["files"], p0, args)
        return 0

    avis = avis_inventats(p0)
    if avis:
        print(avis)
        print("   La b es un dels inventats, i aquesta corba es justament la que diu")
        print("   quant importa encertar-la.")
    print(f"   finestra {args.finestra} cm, {args.resolucio}x{args.resolucio} pixels")
    print(f"   {len(args.friccions)} mapes, temps maxim {args.temps:.0f} s")
    print(f"   calcul: {compila.com_va()}\n")

    print(f"   {'b (s-1)':>9} {'zeta':>7} {'negre %':>9} {'gris %':>8} {'frontera %':>11}")
    files = []
    for b in args.friccions:
        p = replace(p0, b=float(b))
        _, _, imant, _ = calcula(p, finestra, args.resolucio, criteri, args.dt,
                                 cada=args.resolucio)
        total = imant.size
        negre = 100.0 * float(np.count_nonzero(imant == 0)) / total
        gris = 100.0 * float(np.count_nonzero(imant == -1)) / total
        front = 100.0 * float(np.count_nonzero(mascara_de_frontera(imant))) / total
        zeta = b / (2 * np.sqrt(p0.g / p0.L))
        files.append({"b": float(b), "zeta": float(zeta), "negre": negre,
                      "gris": gris, "frontera": front})
        marca = "  <- gris massa alt" if gris > 0.5 else ""
        print(f"   {b:>9.2f} {zeta:>7.3f} {negre:>8.2f} % {gris:>7.2f} % "
              f"{front:>10.2f} %{marca}")

    figura(files, p0, args)

    sortida = Path(args.figures) / f"{args.nom}.json"
    sortida.write_text(json.dumps({
        "finestra_cm": args.finestra,
        "resolucio": args.resolucio,
        "dt_s": args.dt,
        "temps_maxim_s": args.temps,
        "files": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}
                  for f in files],
        "b_del_fitxer_de_parametres": p0.b,
        "valors_inventats": list(p0.inventats),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  dades: {sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
