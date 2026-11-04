r"""El biaix de seleccio de la primera versio dels veins (NOMES SIMULACIO).

Pla: documentacio/pla-analisi-caos.md, «Resultat del pas 1, la validacio (17/10)» i «23/10 · El
diagnostic del biaix de seleccio». Era esborranys/_diag_veins_90.py; puja el 23/10 amb la sortida
identica, perque el que diu va a la memoria.

QUE MIRA. Amb els tirs simulats des dels angles de repos de les preses de 90 i de 150 graus (sense
soroll; sense friccio i amb la DE PROVA), quines parelles troba el metode dels veins de la primera
versio (lyapunov_real.avalua_veins, amb la configuracio del pas 0): si son del mateix tir, a quina
distancia en temps des de les amollades respectives, i com es la corba <ln(d/d0)>. Als 90 graus, el
99 % de les parelles son dos tirs AL MATEIX INSTANT: tirs que van sortir junts i encara son a prop.
Son els que se separen a poc a poc, i per aixo la lambda sortia 0,68 on Benettin dona 2,38.

De dades/ nomes obre el JSON del pas 0 (els angles de repos i la configuracio, cap trajectoria real).

    .venv\\Scripts\\python eines\\seleccio_dels_veins.py --json dades/divergencia/seleccio_veins_2026-10-23.json
    .venv\\Scripts\\python eines\\seleccio_dels_veins.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import lyapunov_real as L  # noqa: E402
from pendol import caos  # noqa: E402
from pendol.mesures import omega_zero  # noqa: E402

PAS0 = ARREL / "dades" / "divergencia" / "lyapunov_real_pas0_2026-10-17.json"
INSTANTS_S = (0, .1, .2, .3, .5, .8, 1.0, 1.5)
MATEIX_INSTANT_S = 0.1


def diagnostic(escriu=print) -> dict:
    """Tot el diagnostic. Escriu les mateixes linies que l'esborrany del 17/10 i torna els numeros."""
    p = L._parametres()
    w0 = omega_zero(p)
    Ev2 = caos.energia_volteig_2(p)
    pas0 = json.loads(PAS0.read_text(encoding="utf-8"))
    out = {}
    for g in ("90", "150"):
        G = pas0["per_grup"][g]
        x0s = [np.array([*np.radians(pas0["per_presa"][n]["angles_repos_graus"]), 0, 0]) for n in G["preses"]]
        cfg = L._veins_cfg(g, pas0)
        for fr in (False, True):
            Y = [L._sim_doble((x, 6.0, fr)) for x in x0s]
            obs = [L.observa(y, x, None, G["rho"], None) for y, x in zip(Y, x0s)]
            obs = [(t[t >= 0], X[:, t >= 0]) for t, X in obs]
            Q, V, E = L._veins_entrada(obs, p, Ev2, w0, L.FRANGES_E[g])
            r = L.avalua_veins(Q, V, E, cfg, n_boot=0)
            par = r["_parelles"]
            mateix = np.mean([a == c for a, _, c, _ in par])
            dt = np.array([abs(b - d) * L.DT for a, b, c, d in par])
            tref = np.array([b * L.DT for a, b, c, d in par])
            m = r["_lnr_mitjana"]
            escriu(f"{g} graus, friccio {'DE PROVA' if fr else 'cap'}: lambda {r['lambda']:.2f}, finestra {r['finestra_s']}, "
                   f"{len(par)} parelles; mateix tir {100*mateix:.0f} %; |t_a - t_b| mediana {np.median(dt):.2f} s "
                   f"(< 0,1 s: {100*np.mean(dt < MATEIX_INSTANT_S):.0f} %); t de referencia mediana {np.median(tref):.2f} s")
            corba = [float(m[int(round(x / L.DT))]) for x in INSTANTS_S]
            escriu("    <ln(d/d0)> a 0, 0,1, 0,2, 0,3, 0,5, 0,8, 1,0, 1,5 s:",
                   " ".join(f"{v:+.2f}" for v in corba))
            d = {"friccio": "DE PROVA" if fr else "cap", "lambda": r["lambda"], "finestra_s": r["finestra_s"],
                 "parelles": len(par), "fraccio_mateix_tir": float(mateix),
                 "dt_amollades_mediana_s": float(np.median(dt)),
                 "fraccio_al_mateix_instant": float(np.mean(dt < MATEIX_INSTANT_S)),
                 "t_referencia_mediana_s": float(np.median(tref)),
                 "ln_d_d0_mitjana": dict(zip([f"{x:g}" for x in INSTANTS_S], corba)), "subconjunts": {}}
            sel = dt < MATEIX_INSTANT_S
            if sel.any() and (~sel).any():
                for nom, s in (("parelles 'al mateix instant'", sel), ("la resta", ~sel)):
                    D = L.segueix(Q, [par[k] for k in np.nonzero(s)[0]], periodic=cfg["periodic"], sep=cfg["sep"],
                                  H=cfg["H"])
                    lr = np.log(D) - np.log(D[:, :1])
                    mm = lr.mean(0)
                    escriu(f"    {nom} ({s.sum()}): <ln(d/d0)> a 0,3 / 0,8 / 1,5 s = "
                           f"{mm[72]:+.2f} / {mm[192]:+.2f} / {mm[360]:+.2f}")
                    d["subconjunts"][nom] = {"parelles": int(s.sum()),
                                             "ln_d_d0_mitjana": {"0.3": float(mm[72]), "0.8": float(mm[192]),
                                                                 "1.5": float(mm[360])}}
            out[f"{g}_{'friccio_de_prova' if fr else 'sense_friccio'}"] = d
    return out


# ------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """D2: reproduible bit a bit, i de dades/ nomes obre el JSON del pas 0 (pla, 23/10)."""
    global _OBERTS
    sys.addaudithook(_registra)
    print("Autoprova de eines/seleccio_dels_veins.py (pla, 23/10). NOMES SIMULACIO.\n")
    _OBERTS = []
    a = diagnostic(lambda *_: None)
    oberts, _OBERTS = _OBERTS, None
    b = diagnostic(lambda *_: None)
    problemes = []
    ok = json.dumps(a) == json.dumps(b)
    print(f"   {'ok ' if ok else 'NO '} D2a: dues passades identiques")
    if not ok:
        problemes.append("D2a")
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve().relative_to(arrel_dades)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel_dades)})
    ok = dins == ["divergencia/lyapunov_real_pas0_2026-10-17.json"]
    print(f"   {'ok ' if ok else 'NO '} D2b: de dades/ nomes obre {', '.join(dins) or 'res'}")
    if not ok:
        problemes.append("D2b")
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
        r = {"eina": "eines/seleccio_dels_veins.py", "pla": "documentacio/pla-analisi-caos.md (17/10 i 23/10)",
             "que_es": "el biaix de seleccio de la primera versio dels veins: NOMES SIMULACIO, sense soroll", **r}
        cami.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
