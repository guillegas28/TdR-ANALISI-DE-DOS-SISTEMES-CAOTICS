r"""Les dues C de la friccio FIRMADA (27/10/2026) i el joc del pendol simple (documentacio/pla-friccio-firmada.md).

Puja d'esborranys/_c_proposta.py (que llegia esborranys/_viscos_carrega.json): el balanc de les 35 preses amb
eines/balanc_energia.balanc i els coeficients firmats (tau_f1 = 1,84e-4 i tau_f2 = 1,36e-4 N m, els dels decaiments; b1 = b2 = 0;
sense terme de carrega) i, a cada presa, la C que tanca el balanc amb l'aire exacte: C = (alliberada - rodaments) / K. Per grup:
la mitjana, la desviacio (ddof = 1) i la dispersio (desviacio / mitjana), com la C publicada.

I el pendol simple: el doble amb l'articulacio bloquejada, que es el muntatge de les preses de decaiment 1.x. L'aire hi es mesurat
com un parell del cos sencer, k*w*|w| (la mitjana de les tres 1.x, de dades/friccio_doble/); en un cos rigid que gira, el parell de
l'aire repartit de pendol/fisica.py es exactament proporcional a w*|w|, o sigui que la k es passa a la C equivalent: la C que, amb
fisica.parells i theta1 = theta2, w1 = w2, dona el parell total k*w*|w|.

    .venv\Scripts\python.exe eines\c_friccio_firmada.py               # escriu dades/balanc/c_friccio_firmada_2026-10-27.json
    .venv\Scripts\python.exe eines\c_friccio_firmada.py --autoprova   # F3 i F5 del pla
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import balanc_energia as BE  # noqa: E402
from pendol.fisica import Dissipacio, Friccio, Parametres, parells  # noqa: E402

TAU_F1, TAU_F2 = 1.84e-4, 1.36e-4        # els firmats (C2), els dels decaiments arrodonits com a la proposta
PARAMETRES = ARREL / "parametres.json"
DECAIMENTS = ARREL / "dades" / "friccio_doble"
SORTIDA = ARREL / "dades" / "balanc" / "c_friccio_firmada_2026-10-27.json"
GRUPS = (("90+150", ("90", "150")), ("45", ("45",)), ("90", ("90",)), ("150", ("150",)))


def preses() -> tuple[list[Path], set]:
    """Les preses del balanc, com l'esborrany: les 35 de dades/net/, sense les fora_del_balanc de dades/retalls.json."""
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    fora = {n for n, v in retalls.items() if isinstance(v, dict) and v.get("fora_del_balanc")}
    noms = [c for g in ("45", "90", "150") for c in sorted((ARREL / "dades" / "net").glob(f"{g}_*.csv"))
            if c.stem not in fora]
    return noms, fora


def muntatge() -> "BE.Muntatge":
    fr = Friccio.des_de_json(PARAMETRES)
    p = Parametres.des_de_json(PARAMETRES)
    return BE.Muntatge(p, Friccio(tau_f1=TAU_F1, b1=0.0, tau_f2=TAU_F2, b2=0.0, C_llibre=fr.C_llibre,
                                  amplada=fr.amplada, gruix=fr.gruix))


def balanc_de_la_presa(cami: Path, M) -> dict:
    a = BE.balanc(cami, M)
    x = BE.balanc(cami, M, aire_exacte=True)
    return {"presa": a["presa"], "grup": cami.stem.split("_")[0], "alliberada": a["alliberada_J"],
            "R0": a["rodaments"]["total"], "K_exacta": x["K_aire"]["K"]}


def resum(P: list[dict]) -> dict:
    fora = {}
    for nom, gs in GRUPS:
        Q = [q for q in P if q["grup"] in gs]
        rod = np.array([q["R0"] / q["alliberada"] for q in Q])
        C = np.array([(q["alliberada"] - q["R0"]) / q["K_exacta"] for q in Q])
        fora[nom] = {"preses": len(Q), "rodaments_mitjana": float(rod.mean()), "rodaments_max": float(rod.max()),
                     "C_mitjana": float(C.mean()), "C_desviacio": float(C.std(ddof=1)),
                     "dispersio": float(C.std(ddof=1) / C.mean())}
    return fora


def k_del_simple() -> dict:
    """La k de l'aire del cos lligat: la mitjana de les tres preses 1.x (a_aerodinamic * la inercia, com taula_friccio_nova)."""
    ks = {}
    for n in ("1.1", "1.2", "1.3"):
        d = json.loads((DECAIMENTS / f"Friccio_{n}.json").read_text(encoding="utf-8"))
        ks[n] = float(d["energia_sencera"]["a_aerodinamic_rad-1"] * d["inercia_kg_m2"])
    v = np.array(list(ks.values()))
    return {"per_presa": ks, "mitjana": float(v.mean()), "dispersio": float(v.std(ddof=1) / v.mean())}


def parell_aire_del_cos_lligat(p: Parametres, C: float, w: float) -> float:
    """El parell total de l'aire sobre el cos lligat (theta1 = theta2 = 0, w1 = w2 = w), amb fisica.parells."""
    Q1, Q2 = parells((0.0, 0.0, w, w), p, Dissipacio(tau_f1=0.0, b1=0.0, tau_f2=0.0, b2=0.0, C=C, eps=1e-3))
    return Q1 + Q2


def C_equivalent(p: Parametres, k: float) -> float:
    """La C que fa que l'aire del model sobre el cos lligat sigui -k*w*|w|."""
    return k / -parell_aire_del_cos_lligat(p, 1.0, 1.0)


def main_calcula() -> int:
    camins, fora = preses()
    print(f"{len(camins)} preses (fora del balanc: {sorted(fora)}); tau_f1 = {TAU_F1}, tau_f2 = {TAU_F2}, b = 0, aire exacte")
    M = muntatge()
    P = [balanc_de_la_presa(c, M) for c in camins]
    R = resum(P)
    for nom, d in R.items():
        print(f"   {nom:7} ({d['preses']:2} preses): rodaments {100 * d['rodaments_mitjana']:5.1f} % (max "
              f"{100 * d['rodaments_max']:5.1f} %); C = {d['C_mitjana']:.5f} +- {d['C_desviacio']:.5f} "
              f"(dispersio {100 * d['dispersio']:.1f} %)")
    p = Parametres.des_de_json(PARAMETRES)
    ks = k_del_simple()
    Cs = C_equivalent(p, ks["mitjana"])
    print(f"   simple: k = {ks['mitjana']:.5e} N m s^2 (dispersio {100 * ks['dispersio']:.1f} %) -> C equivalent = {Cs:.6f}")
    SORTIDA.parent.mkdir(parents=True, exist_ok=True)
    SORTIDA.write_text(json.dumps({
        "_que_es": "Les C de la friccio firmada el 27/10/2026 (documentacio/pla-friccio-firmada.md), d'eines/c_friccio_firmada.py.",
        "coeficients": {"tau_f1": TAU_F1, "tau_f2": TAU_F2, "b1": 0.0, "b2": 0.0, "terme_de_carrega": None,
                        "aire": "exacte (balanc_energia.balanc(aire_exacte=True))"},
        "preses": P, "per_grup": R,
        "simple": {"k_aire_N_m_s2": ks, "C_equivalent": Cs,
                   "_com": "la C que, amb fisica.parells sobre el cos lligat (theta1 = theta2, w1 = w2), dona -k*w*|w|"}},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {SORTIDA.relative_to(ARREL)}")
    return 0


def autoprova() -> int:
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova d'eines/c_friccio_firmada.py (F3 i F5 de documentacio/pla-friccio-firmada.md)")
    p = Parametres.des_de_json(PARAMETRES)
    bloc = json.loads(PARAMETRES.read_text(encoding="utf-8"))["fisic"]["friccio_firmada"]
    # F3: el joc que toca segons l'angle d'amollada, i fora dels angles validats s'atura.
    ok, detall = True, []
    for angle, joc in ((45, "doble_45"), (36, "doble_45"), (54, "doble_45"), (90, "doble_90_150"), (99, "doble_90_150"),
                       (150, "doble_90_150"), (141, "doble_90_150"), (-90, "doble_90_150")):
        d = Dissipacio.firmada(angle, PARAMETRES)
        j = bloc["jocs"][joc]
        ok &= (d.tau_f1, d.tau_f2, d.b1, d.b2, d.C, d.eps) == (j["tau_f1"], j["tau_f2"], j["b1"], j["b2"], j["C"], j["eps"])
        detall.append(f"{angle}->{d.C:g}")
    atura = []
    for angle in (120, 10, 170, 65):
        try:
            Dissipacio.firmada(angle, PARAMETRES)
        except ValueError:
            atura.append(angle)
    mira(ok and atura == [120, 10, 170, 65], f"F3 el joc segons l'angle: {', '.join(detall)}; s'atura a {atura} (120, 10, 170, 65)")
    # F5: el joc del simple: el parell de l'aire del cos lligat amb la C equivalent es k*w^2.
    js = bloc["jocs"]["simple"]
    k = js["k_aire_N_m_s2"]
    rel = max(abs(-parell_aire_del_cos_lligat(p, js["C"], w) / (k * w * w) - 1.0) for w in (1.0, 5.0, 15.0))
    ds = Dissipacio.firmada(120, PARAMETRES, simple=True)
    mira(rel < 1e-9 and ds.C == js["C"] and ds.tau_f1 == js["tau_f1"],
         f"F5 el simple: el parell de l'aire del cos lligat amb la C equivalent ({js['C']:.6f}) es k*w^2 a w = 1, 5 i 15 rad/s: "
         f"{rel:.1e} relatiu (< 1e-9)")
    try:
        Dissipacio.firmada(170, PARAMETRES, simple=True)
        simple_atura = False
    except ValueError:
        simple_atura = True
    mira(simple_atura, "F5 el simple s'atura per sobre de l'amplitud maxima dels decaiments (170 graus)")
    # La coherencia: el bloc de parametres.json diu el que va calcular l'eina.
    if SORTIDA.exists():
        R = json.loads(SORTIDA.read_text(encoding="utf-8"))
        mira(bloc["jocs"]["doble_90_150"]["C"] == round(R["per_grup"]["90+150"]["C_mitjana"], 5)
             and bloc["jocs"]["doble_45"]["C"] == round(R["per_grup"]["45"]["C_mitjana"], 5)
             and js["C"] == R["simple"]["C_equivalent"],
             "el bloc friccio_firmada de parametres.json diu les C del JSON (arrodonides a 5 decimals les del doble, com la firma)")
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    return autoprova() if args.autoprova else main_calcula()


if __name__ == "__main__":
    raise SystemExit(main())
