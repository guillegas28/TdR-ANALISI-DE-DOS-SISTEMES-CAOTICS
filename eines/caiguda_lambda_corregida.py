r"""Per que la lambda* dels 150 graus baixa amb el repos final a (0, 0) (NOMES PRESES REALS).

Pla: documentacio/pla-analisi-caos.md, «24/10 · La λ amb el repòs final a (0, 0)» i «24/10 (tarda)».
Era esborranys/_diag_corregit_r2.py; puja el 24/10 amb la sortida identica, perque el que diu pot
anar a la memoria.

QUE MIRA. Per a les preses dels 150 graus que entren al resultat 2 (lyapunov_real), abans i despres
de la correccio (lyapunov_real.correccio_repos): la separacio inicial d0 de cada parella (dels angles
d'amollada), el temps T(D*) en que arriba a D*, ln(D*/d0) i la lambda de cada parella. I els angles
d'amollada de cada presa abans i despres, amb la c del desplacament de la marca.

El que en surt: la correccio separa les condicions inicials (la d0 mediana passa d'1,85 a 3,33
graus), i per aixo la lambda* baixa. I amb el tope a una marca fixa amb que es col·locava el pendol
(en Guillem, 24/10), els angles corregits es parteixen en dos grups just on canvia la c: el biaix de
l'amollada potser no es el del repos final.

25/10: la correccio d'un sol pas (el repos final a (0, 0)) es un MODEL DESCARTAT el 24/10: l'eix de gir de
debo es per sobre del pivot de la conversio (documentacio/pla-correccio-biaix.md). Aquesta eina queda perque
ensenya per que es va descartar; la lambda* que val es la publicada.

Cap simulacio. De dades/ nomes obre el JSON del pas 0b, les preses de dades/net/ i els .trk dels 150.

    .venv\\Scripts\\python eines\\caiguda_lambda_corregida.py --json dades/divergencia/caiguda_lambda_corregida_2026-10-24.json
    .venv\\Scripts\\python eines\\caiguda_lambda_corregida.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import lyapunov_real as L  # noqa: E402

PAS0B = ARREL / "dades" / "divergencia" / "lyapunov_real_pas0b_2026-10-17.json"
GRUP = "150"


def _resum(d0, T, Ds):
    lam = np.log(Ds / d0) / T
    return {"d0_mediana_graus": math.degrees(np.median(d0)),
            "d0_p16_graus": math.degrees(np.percentile(d0, 16)),
            "d0_p84_graus": math.degrees(np.percentile(d0, 84)),
            "T_mediana_s": float(np.median(T)),
            "ln_D_d0_mediana": float(np.median(np.log(Ds / d0))),
            "lambda_mediana": float(np.median(lam))}


def diagnostic(escriu=print) -> dict:
    """Tot el diagnostic. Escriu les mateixes linies que l'esborrany del 24/10 i torna els numeros."""
    pas02 = json.loads(PAS0B.read_text(encoding="utf-8"))
    R = pas02["resultat2"][GRUP]
    noms = R["preses"]
    N = len(noms)
    Ds = R["D_estrella_rad"]
    pr = {p["nom"]: p for p in L.preses_reals() if p["nom"] in noms}
    cor = {n: L.correccio_repos(pr[n]) for n in noms}
    tc = np.arange(0.0, L.T_MAX_S + 0.5 * L.DT, L.DT)
    par = [(a, b) for a in range(N) for b in range(a + 1, N)]
    out = {"grup": GRUP, "D_estrella_graus": math.degrees(Ds), "parelles": len(par)}
    for nom, rep, est in (("abans", lambda n: pr[n]["repos_rad"], lambda n: pr[n]["estats"]),
                          ("despres", lambda n: cor[n]["repos_rad"], lambda n: cor[n]["estats"])):
        d0 = np.array([float(L.dist_ang(rep(noms[a]), rep(noms[b]))) for a, b in par])
        X = [L.a_la_graella(pr[n]["t"], est(n), 0.0, tc) for n in noms]
        d = np.array([L.dist_ang(X[a], X[b]) for a, b in par])
        T = L.creuaments(tc, d, [Ds])[:, 0]
        r = out[nom] = _resum(d0, T, Ds)
        escriu(f"{nom}: d0 mediana {r['d0_mediana_graus']:.2f} gr (p16-p84 {r['d0_p16_graus']:.2f}-"
               f"{r['d0_p84_graus']:.2f}); T(D*) mediana {r['T_mediana_s']:.3f} s; ln(D*/d0) mediana "
               f"{r['ln_D_d0_mediana']:.2f}; lambda mediana {r['lambda_mediana']:.3f}")
    escriu("angles d'amollada (graus) abans -> despres, i la c:")
    out["per_presa"] = {}
    for n in noms:
        a, b = np.degrees(pr[n]["repos_rad"]), np.degrees(cor[n]["repos_rad"])
        out["per_presa"][n] = {"amollada_abans_graus": [float(a[0]), float(a[1])],
                               "amollada_despres_graus": [float(b[0]), float(b[1])], "c_px": float(cor[n]["c_px"])}
        escriu(f"  {n:7} th1 {a[0]:7.2f} -> {b[0]:7.2f}   th2 {a[1]:7.2f} -> {b[1]:7.2f}   c {cor[n]['c_px']:+6.2f} px")
    return out


# ------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """C2: reproduible bit a bit, i de dades/ nomes obre el JSON del pas 0b, dades/net/ i els .trk dels 150
    (pla, 24/10; la C2b, reescrita el 25/10 despres de fallar)."""
    global _OBERTS
    sys.addaudithook(_registra)
    print("Autoprova de eines/caiguda_lambda_corregida.py (pla, 24/10). NOMES PRESES REALS.\n")
    _OBERTS = []
    a = diagnostic(lambda *_: None)
    oberts, _OBERTS = _OBERTS, None
    b = diagnostic(lambda *_: None)
    problemes = []
    ok = json.dumps(a) == json.dumps(b)
    print(f"   {'ok ' if ok else 'NO '} C2a: dues passades identiques")
    if not ok:
        problemes.append("C2a")
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    # C2b reescrita el 25/10 DESPRES de fallar, per decisio d'en Guillem: la correccio torna a llegir les
    # posicions del .trk de les preses dels 150 (el criteri del 24/10 les havia oblidat). Vegeu el pla.
    trk = [x for x in dins if x.startswith("video/tracker/150/") and x.endswith(".trk")]
    fora = [x for x in dins if x != "divergencia/lyapunov_real_pas0b_2026-10-17.json" and not x.startswith("net/")
            and x not in trk]
    ok = not fora and "divergencia/lyapunov_real_pas0b_2026-10-17.json" in dins
    print(f"   {'ok ' if ok else 'NO '} C2b: de dades/ nomes obre el pas 0b, {sum(x.startswith('net/') for x in dins)} "
          f"fitxers de net/ i {len(trk)} .trk dels 150{'; a mes: ' + ', '.join(fora) if fora else ''}")
    if not ok:
        problemes.append("C2b")
    print()
    if problemes:
        print("AUTOPROVA FALLADA: " + ", ".join(problemes) + ". Un criteri que falla NO es relaxa.")
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar els numeros")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    r = diagnostic()
    if args.json:
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        r = {"eina": "eines/caiguda_lambda_corregida.py", "pla": "documentacio/pla-analisi-caos.md (24/10)",
             "que_es": "per que la lambda* dels 150 baixa amb el repos final a (0, 0): NOMES PRESES REALS, cap simulacio",
             **r}
        cami.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
