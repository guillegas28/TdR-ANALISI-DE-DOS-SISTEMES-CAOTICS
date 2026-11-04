"""El control que s'ha de passar abans de publicar cap mapa.

RK4 es de pas fix. Prop d'un imant la forca es dispara i la trajectoria canvia
molt de pressa, i amb un dt massa gran hi apareix estructura fina que sembla
fractal i es NUMERICA. Sense aquesta comprovacio, un lector te tot el dret a
dir que el fractal que ensenyes te'l va fer l'integrador.

La prova: calcular el mateix tros dues vegades, amb dt i amb dt/2, i dir quin
percentatge de pixels canvia de color.

I sobretot, SEPARANT-HO EN DOS:

  * pixels de DINS d'una conca (tots els vuit veins del mateix color).
    Aqui no n'hauria de canviar CAP. Si en canvien, el dt es massa gran i el
    mapa no es de fiar enlloc.

  * pixels de FRONTERA. Aqui en canviaran, i no es cap defecte: son els punts
    on el desti no esta determinat de manera robusta, que es exactament el
    fenomen que el treball vol ensenyar. El numero interessant es quants.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.converge --finestra -7 7 -7 7 --resolucio 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from magnetic import compila, estil
from magnetic.analisi import mascara_de_frontera
from magnetic.fisica import (
    Parametres,
    afegeix_banderes_de_parametres,
    avis_inventats,
    sobreescriu,
)
from magnetic.mapa import calcula
from magnetic.motor import DT_PER_DEFECTE, Criteri


def compara(imant_a, imant_b):
    """Retorna (percent_total, percent_dins, percent_frontera, n_dins, n_frontera).

    "Dins" i "frontera" es decideixen amb el mapa de referencia (el del dt gran),
    que es el que es publicaria.
    """
    frontera = mascara_de_frontera(imant_a)
    dins = ~frontera
    canvia = imant_a != imant_b
    n_dins = int(np.count_nonzero(dins))
    n_front = int(np.count_nonzero(frontera))
    return (
        100.0 * float(np.count_nonzero(canvia)) / imant_a.size,
        100.0 * float(np.count_nonzero(canvia & dins)) / max(1, n_dins),
        100.0 * float(np.count_nonzero(canvia & frontera)) / max(1, n_front),
        n_dins,
        n_front,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--finestra", type=float, nargs=4, default=[-7, 7, -7, 7],
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="limits en CENTIMETRES")
    ap.add_argument("--resolucio", type=int, default=200, help="pixels per costat")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE, help="el dt a comprovar (s)")
    ap.add_argument("--temps", type=float, default=None, help="temps maxim (s)")
    ap.add_argument("--nom", default="convergencia", help="nom del fitxer de resultats")
    ap.add_argument("--figures", default="magnetic/figures", help="on va el resultat")
    # Les mateixes --b/--n/--K que mapa.py (25/10/2026): un dt que val per a una
    # llei pot no valer per a una altra amb el camp proper mes dur.
    afegeix_banderes_de_parametres(ap)
    args = ap.parse_args(argv)

    p = Parametres.des_de_json()
    p, avis_sobreescrit = sobreescriu(p, b=args.b, n=args.n, K=args.K)
    d = Criteri()
    criteri = Criteri(t_max=d.t_max if args.temps is None else args.temps)
    finestra = [v / 100.0 for v in args.finestra]

    print("=" * 72)
    print(" magnetic.converge -- el mapa depen del pas d'integracio?")
    print("=" * 72)
    if avis_sobreescrit:
        print(avis_sobreescrit)
    avis = avis_inventats(p)
    if avis:
        print(avis)
    print(f"   finestra {args.finestra} cm, {args.resolucio}x{args.resolucio} pixels")
    print(f"   calcul: {compila.com_va()}\n")

    cada = max(1, args.resolucio // 4)
    print(f"   amb dt = {args.dt} s")
    _, _, imant_a, _ = calcula(p, finestra, args.resolucio, criteri, args.dt, cada)
    print(f"\n   amb dt = {args.dt/2} s")
    _, _, imant_b, _ = calcula(p, finestra, args.resolucio, criteri, args.dt / 2, cada)

    total, dins, front, n_dins, n_front = compara(imant_a, imant_b)

    print(f"\n   {'':>22} {'pixels':>10} {'canvien':>10}")
    print(f"   {'DINS de les conques':>22} {n_dins:>10} {dins:>9.2f} %")
    print(f"   {'a la FRONTERA':>22} {n_front:>10} {front:>9.2f} %")
    print(f"   {'tot el mapa':>22} {imant_a.size:>10} {total:>9.2f} %")

    llindar_dins = 0.5
    ok = dins <= llindar_dins
    print()
    if ok:
        print(f"   OK -> dins de les conques nomes canvia el {dins:.2f} % (llindar "
              f"{llindar_dins} %).")
        print("        L'estructura que ensenya el mapa NO se la inventa l'integrador:")
        print("        afinant el pas, les conques es queden on son.")
    else:
        print(f"   ATENCIO -> dins de les conques canvia el {dins:.2f} %, per sobre del")
        print(f"        llindar de {llindar_dins} %. El dt de {args.dt} s es massa gran per")
        print("        a aquesta finestra i el mapa no es publicable tal com esta.")
    print(f"\n   A la frontera en canvia el {front:.2f} %, i AIXO NO ES CAP DEFECTE:")
    print("   son els punts on el desti no esta determinat de manera robusta.")
    print("   Es el fenomen que el treball vol ensenyar, mesurat.")

    sortida = Path(args.figures) / f"{args.nom}.json"
    sortida.parent.mkdir(parents=True, exist_ok=True)
    sortida.write_text(json.dumps({
        "finestra_cm": args.finestra,
        "resolucio": args.resolucio,
        "dt_s": args.dt,
        "dt_comparat_s": args.dt / 2,
        "temps_maxim_s": criteri.t_max,
        "canvien_percent": {"dins_de_conca": round(dins, 4),
                            "frontera": round(front, 4),
                            "tot": round(total, 4)},
        "pixels": {"dins_de_conca": n_dins, "frontera": n_front, "tot": int(imant_a.size)},
        "veredicte": "OK" if ok else "ATENCIO",
        "valors_inventats": list(p.inventats),
        "_sobreescrits": {clau: valor for clau, valor in
                          (("b", args.b), ("n", args.n), ("K", args.K))
                          if valor is not None} or None,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  resultats: {sortida}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
