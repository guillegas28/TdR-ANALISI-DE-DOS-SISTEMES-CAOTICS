"""Compara dos balancos d'energia: que s'ha mogut, a quines preses i per que.

Per que existeix. El bloc D1b de documentacio/pla-doble-interactiu.md canvia
la conversio de les preses en quatre passos, i desa el balanc d'energia
despres de cadascun. Si la C es mou, cal saber si es mou perque han entrat
preses noves o perque les que ja hi eren han canviat. Mirar dues mitjanes no
ho diu: una presa nova i una de corregida poden moure la C igual.

Entren dos JSON de eines/balanc_energia.py --json, que son la llista de
preses que aquell balanc va fer servir amb els numeros de cadascuna. Surt:

  - presa a presa: IGUAL, CANVIADA (i que s'hi ha mogut), NOVA o TRETA;
  - la C -mitjana, desviacio i dispersio- i el repartiment sec/viscos/aire,
    sobre les preses COMUNES als dos balancos i sobre totes.

IGUAL vol dir igual a totes les xifres, sense cap tolerancia. El balanc es
determinista i el JSON desa els nombres sencers, o sigui que dues conversions
que donen el mateix CSV han de donar exactament els mateixos numeros. Si una
presa surt CANVIADA per una diferencia de l'ordre de 1e-15, no es cap
arrodoniment que s'hagi d'amagar: vol dir que alguna cosa de l'entrada ha
canviat, i la columna de la diferencia ho ensenya.

El repartiment es calcula igual que a eines/balanc_energia.py: cada terme
dividit per l'energia alliberada de la seva presa, i la mitjana entre preses.
L'aire no es cap mesura, es el forat que queda.

Us:
    .venv\\Scripts\\python eines\\compara_balancos.py abans.json despres.json
    .venv\\Scripts\\python eines\\compara_balancos.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Que es mira de cada presa per dir si ha canviat, i com es diu a la taula.
# Els angles van en graus i la resta en unitats SI, tal com els desa el balanc.
CAMPS = (
    ("alliberada_J", "energia alliberada"),
    ("rodaments.total", "rodaments"),
    ("forat_J", "forat"),
    ("C_que_tanca", "C"),
    ("angles_inicials_graus", "angles inicials"),
    ("durada_s", "durada"),
    ("quiet_al_principi_s", "repos inicial"),
)


def carrega(cami) -> dict:
    """Un JSON de balanc_energia.py --json, indexat pel nom de la presa."""
    llista = json.loads(Path(cami).read_text(encoding="utf-8"))
    return {r["presa"]: r for r in llista}


def _valor(r: dict, camp: str):
    """Un camp d'una presa. Els camps amb punt van a dins d'un diccionari."""
    for tros in camp.split("."):
        r = r[tros]
    return r


def que_ha_canviat(a: dict, b: dict) -> list[tuple[str, str]]:
    """Els camps que no son iguals entre dues versions de la mateixa presa.

    Torna (nom, com ha canviat) per a cadascun. Buit vol dir IGUAL.
    """
    canvis = []
    for camp, nom in CAMPS:
        va, vb = _valor(a, camp), _valor(b, camp)
        if va == vb:
            continue
        if camp == "angles_inicials_graus":
            d = np.asarray(vb, dtype=float) - np.asarray(va, dtype=float)
            canvis.append((nom, f"{d[0]:+.3f} / {d[1]:+.3f} graus"))
        elif va is None or vb is None:
            canvis.append((nom, f"{va} -> {vb}"))
        elif va == 0:
            canvis.append((nom, f"{va:.4g} -> {vb:.4g}"))
        else:
            canvis.append((nom, f"{100 * (vb - va) / abs(va):+.3g} %"))
    return canvis


def resum(preses: list[dict]) -> dict:
    """La C i el repartiment d'un conjunt de preses, com els dona el balanc."""
    if not preses:
        return {"n": 0}
    C = np.array([r["C_que_tanca"] for r in preses])
    e = np.array([r["alliberada_J"] for r in preses])
    rod = [r["rodaments"] for r in preses]
    sec = np.array([x["sec_1"] + x["sec_2"] for x in rod]) / e
    vis = np.array([x["viscos_1"] + x["viscos_2"] for x in rod]) / e
    aire = np.array([r["forat_J"] for r in preses]) / e
    sortida = {
        "n": len(preses),
        "C": float(C.mean()),
        "sec": float(sec.mean()),
        "viscos": float(vis.mean()),
        "aire": float(aire.mean()),
    }
    if len(C) > 1:
        sortida["sigma_C"] = float(C.std(ddof=1))
        sortida["dispersio"] = float(C.std(ddof=1) / abs(C.mean()))
    return sortida


def compara(abans: dict, despres: dict) -> dict:
    """Presa a presa i en conjunt. Els dos arguments, com els torna carrega()."""
    comunes = sorted(set(abans) & set(despres))
    estat = {}
    for nom in sorted(set(abans) | set(despres)):
        if nom not in despres:
            estat[nom] = ("treta", [])
        elif nom not in abans:
            estat[nom] = ("nova", [])
        else:
            canvis = que_ha_canviat(abans[nom], despres[nom])
            estat[nom] = ("canviada" if canvis else "igual", canvis)
    return {
        "estat": estat,
        "comunes": comunes,
        "comunes_abans": resum([abans[n] for n in comunes]),
        "comunes_despres": resum([despres[n] for n in comunes]),
        "totes_abans": resum(list(abans.values())),
        "totes_despres": resum(list(despres.values())),
    }


def _clau(nom: str):
    """Ordena 150_9 abans que 150_10."""
    grup, _, num = nom.partition("_")
    return (int(grup), int(num)) if grup.isdigit() and num.isdigit() else (10**9, nom)


def imprimeix(res: dict, abans: dict, despres: dict, nom_a: str, nom_b: str) -> None:
    print(f"abans:   {nom_a} ({len(abans)} preses)")
    print(f"despres: {nom_b} ({len(despres)} preses)\n")

    print(f"{'presa':<8} {'estat':<9} {'C abans':>9} {'C despres':>10}   que s'hi ha mogut")
    print("-" * 78)
    for nom in sorted(res["estat"], key=_clau):
        estat, canvis = res["estat"][nom]
        ca = f"{abans[nom]['C_que_tanca']:.5f}" if nom in abans else "-"
        cb = f"{despres[nom]['C_que_tanca']:.5f}" if nom in despres else "-"
        marca = estat.upper() if estat != "igual" else estat
        detall = "; ".join(f"{n} {c}" for n, c in canvis)
        print(f"{nom:<8} {marca:<9} {ca:>9} {cb:>10}   {detall}")

    plural = {"igual": "iguals", "canviada": "canviades", "nova": "noves", "treta": "tretes"}
    comptes = {}
    for estat, _ in res["estat"].values():
        comptes[estat] = comptes.get(estat, 0) + 1
    print("\n" + ", ".join(f"{v} {k if v == 1 else plural[k]}"
                           for k, v in sorted(comptes.items())))

    print(f"\n{'':<17} {'preses':>6} {'C':>9} {'sigma':>9} {'disp.':>7} "
          f"{'sec':>7} {'viscos':>7} {'aire':>7}")
    print("-" * 78)
    for clau, nom in (("comunes_abans", "comunes, abans"), ("comunes_despres", "comunes, despres"),
                      ("totes_abans", "totes, abans"), ("totes_despres", "totes, despres")):
        s = res[clau]
        if not s["n"]:
            print(f"{nom:<17} {0:>6}")
            continue
        sig = f"{s['sigma_C']:.5f}" if "sigma_C" in s else "-"
        dis = f"{100*s['dispersio']:.2f}%" if "dispersio" in s else "-"
        print(f"{nom:<17} {s['n']:>6} {s['C']:>9.5f} {sig:>9} {dis:>7} "
              f"{100*s['sec']:>6.1f}% {100*s['viscos']:>6.1f}% {100*s['aire']:>6.1f}%")

    a, b = res["comunes_abans"], res["comunes_despres"]
    print()
    if not res["comunes"]:
        print("Cap presa comuna: no hi ha res a comparar de la mateixa manera.")
    elif a == b:
        print(f"Les {a['n']} preses comunes donen EXACTAMENT el mateix: la C no s'ha")
        print("mogut per cap canvi de les que ja hi eren.")
    else:
        print(f"La C de les {a['n']} preses comunes passa de {a['C']:.5f} a {b['C']:.5f} "
              f"({100*(b['C']-a['C'])/abs(a['C']):+.2f} %).")
        print("Aixo ho mouen les preses CANVIADES; les noves i les tretes no hi entren.")


# ----------------------------------------------------------------- l'autoprova


def _presa_falsa(nom: str, rng) -> dict:
    """Una entrada de balanc inventada, amb la forma de les de debo."""
    e = float(rng.uniform(0.19, 0.37))
    parts = rng.uniform(0.05, 0.3, 4) * e
    rod = {"sec_1": float(parts[0]), "viscos_1": float(parts[1]),
           "sec_2": float(parts[2]), "viscos_2": float(parts[3]),
           "total": float(parts.sum())}
    forat = e - rod["total"]
    K = float(rng.uniform(10.0, 20.0))
    return {
        "presa": nom,
        "durada_s": float(rng.uniform(40, 90)),
        "angles_inicials_graus": [float(rng.uniform(85, 155)), float(rng.uniform(85, 155))],
        "quiet_al_principi_s": float(rng.uniform(0.1, 2.0)),
        "alliberada_J": e,
        "rodaments": rod,
        "forat_J": forat,
        "C_que_tanca": forat / K,
    }


def autoprova() -> int:
    """Una llista contra ella mateixa, i contra una versio manipulada.

    Passa pel circuit sencer: els JSON s'escriuen i es tornen a llegir, com
    els de debo. Les C esperades es calculen a ma, a part, i no amb resum().
    """
    rng = np.random.default_rng(7)
    desti = Path("esborranys/_autoprova_compara_balancos")
    desti.mkdir(parents=True, exist_ok=True)
    noms = ["90_2", "90_3", "150_1", "150_9", "150_10"]
    abans = [_presa_falsa(n, rng) for n in noms]
    (desti / "abans.json").write_text(json.dumps(abans, indent=2), encoding="utf-8")

    problemes = []
    print("Autoprova de la comparacio de balancos\n")

    # 1. Contra ella mateixa: tot igual i les dues C exactament iguals.
    a = carrega(desti / "abans.json")
    r = compara(a, carrega(desti / "abans.json"))
    if any(e != "igual" for e, _ in r["estat"].values()):
        problemes.append("una llista comparada amb ella mateixa no surt tota igual")
    if r["comunes_abans"] != r["comunes_despres"] or r["totes_abans"] != r["totes_despres"]:
        problemes.append("una llista comparada amb ella mateixa canvia la C")
    print(f"   1. contra ella mateixa: {sum(e == 'igual' for e, _ in r['estat'].values())} "
          f"de {len(noms)} iguals")

    # 2. Manipulada: 150_1 canviada (forat i C), 90_3 treta, 90_11 nova, i un
    #    canvi de 1e-15 relatiu a 150_9, que ha de sortir canviada i no igual.
    despres = json.loads(json.dumps(abans))
    per_nom = {x["presa"]: x for x in despres}
    per_nom["150_1"]["forat_J"] *= 1.10
    per_nom["150_1"]["C_que_tanca"] *= 1.10
    per_nom["150_9"]["alliberada_J"] *= 1 + 1e-15
    despres = [x for x in despres if x["presa"] != "90_3"]
    despres.append(_presa_falsa("90_11", rng))
    (desti / "despres.json").write_text(json.dumps(despres, indent=2), encoding="utf-8")

    b = carrega(desti / "despres.json")
    r = compara(a, b)
    esperat = {"90_2": "igual", "90_3": "treta", "150_1": "canviada", "150_9": "canviada",
               "150_10": "igual", "90_11": "nova"}
    trobat = {n: e for n, (e, _) in r["estat"].items()}
    if trobat != esperat:
        problemes.append(f"classificacio equivocada: {trobat} en comptes de {esperat}")
    canvis_150_1 = {n for n, _ in r["estat"]["150_1"][1]}
    if canvis_150_1 != {"forat", "C"}:
        problemes.append(f"a la 150_1 han canviat el forat i la C, i diu {sorted(canvis_150_1)}")

    # La C de les comunes, a ma: 90_2, 150_1, 150_9 i 150_10.
    comunes = ["90_2", "150_1", "150_9", "150_10"]
    C_a = np.mean([a[n]["C_que_tanca"] for n in comunes])
    C_b = np.mean([b[n]["C_que_tanca"] for n in comunes])
    if sorted(r["comunes"]) != sorted(comunes):
        problemes.append(f"les comunes haurien de ser {comunes} i son {r['comunes']}")
    if not (np.isclose(r["comunes_abans"]["C"], C_a, rtol=1e-14, atol=0)
            and np.isclose(r["comunes_despres"]["C"], C_b, rtol=1e-14, atol=0)):
        problemes.append("la C de les comunes no es la que surt a ma")
    print(f"   2. manipulada: {trobat}")
    print(f"      C de les comunes a ma {C_a:.6f} -> {C_b:.6f}; l'eina "
          f"{r['comunes_abans']['C']:.6f} -> {r['comunes_despres']['C']:.6f}")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> distingeix igual, canviada, nova i treta, veu un canvi d'1e-15, i la")
    print("     C de les comunes es la que surt a ma. Limit: aixo prova la")
    print("     comptabilitat, no diu PER QUE una presa ha canviat; aixo s'ha de")
    print("     buscar a la conversio.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("abans", nargs="?", help="JSON de balanc_energia.py --json")
    ap.add_argument("despres", nargs="?", help="JSON de balanc_energia.py --json")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not (args.abans and args.despres):
        ap.error("digue'm els dos JSON, o passa --autoprova")

    abans, despres = carrega(args.abans), carrega(args.despres)
    imprimeix(compara(abans, despres), abans, despres, args.abans, args.despres)
    return 0


if __name__ == "__main__":
    sys.exit(main())
