r"""La creu del Tracker en moviment (27/10, documentacio/pla-correccio-biaix.md, «La dissetena tanda», passos 1 i 2).

Puja d'esborranys/_creu_moviment.py el 29/10 (documentacio/pla-figura-i-eines-experiment1.md, pas 3.2), amb la sortida igual.
Dos canvis, perque una eina no llegeix res d'esborranys/: la llista de fotogrames (decidida abans de clicar) es ara
dades/versemblant/creu_moviment_llista.json, copiada tal com era; i la M0 (la creu del repos inicial calculada aqui contra la de
esborranys/_creu_correccio.json) ja no es fa: va passar el 27/10 (0,0e+00 px), i la pujada es va comprovar amb la sortida.

Els clics d'en Guillem son a copies dels .trk, a dades/video/tracker/analisi_moviment/<presa>.trk (i <presa>_repeticio.trk, la
segona sessio), amb les pistes velles esborrades i tres de noves: pivot (un clic), rodament i extrem (els cinc fotogrames de
la llista). El .trk desa el clic en pixels de la imatge i amb el fotograma, o sigui que no passa
pels eixos ni per cap taula arrodonida. Els originals no s'obren per escriure mai. Es desa un registre dels clics a
dades/versemblant/clics_moviment.json.

A cada fotograma clicat, la creu (la posicio crua del .trk original en aquell fotograma) menys el clic, de costat de la barra
(el rodament respecte de pivot -> clic del rodament; l'extrem respecte de clic del rodament -> clic de l'extrem, com
esborranys/_creu_correccio.py), i Delta = aquesta menys la del repos inicial. Informatiu: el mateix al llarg de la barra.

Criteris de l'instrument (si un falla, s'atura sense ensenyar cap Delta):
    C0  cada copia es la mateixa presa (el mateix video i el mateix pas de temps que l'original), i te exactament tres pistes
        amb marques: pivot (o fix) amb 1, i rodament i extrem als cinc fotogrames de la llista; cap altra pista amb marques
    F1  el pivot clicat a < 8 px de la mediana del fix, i els gomets a < 15 px de la seva creu al mateix fotograma
    J   nomes obre els fitxers dits al pla
Decisio del pilot (escrita abans): si l'rms de Delta de costat de les 20 mesures passa d'1 px, s'atura (la dotzena en un sol
sentit). Pas 2: la sigma de la tanda, sigma_D^2 = rms(Delta)^2 - sigma_cm^2 - sigma_cr^2, per grup i per gomet.

    .venv\Scripts\python.exe eines\creu_moviment.py
    .venv\Scripts\python.exe eines\creu_moviment.py --carpeta <altra>   (prova: no desa res)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
import correccio_amb_clics as cac  # noqa: E402
from pendol import trk  # noqa: E402

COPIES = ARREL / "dades" / "video" / "tracker" / "analisi_moviment"
LLISTA = ARREL / "dades" / "versemblant" / "creu_moviment_llista.json"
CLICS_MOV = ARREL / "dades" / "versemblant" / "clics_moviment.json"
SORTIDA = ARREL / "dades" / "versemblant" / "creu_moviment.json"
PILOT = ("45_11", "90_6")
LLINDAR_PILOT_PX = 1.0
SIGMA_CR_PX = 0.35          # el clic al repos (pas B, 26/10): cada gomet s'hi va clicar un sol cop


def descomp(creu, clic, origen) -> tuple[float, float]:
    """(de costat, al llarg) de creu - clic, respecte de la recta origen -> clic. La de _creu_correccio.py."""
    u = (clic - origen) / np.linalg.norm(clic - origen)
    e = creu - clic
    return float(e @ np.array([u[1], -u[0]])), float(e @ u)


def pistes_de_la_copia(cami: Path) -> list[tuple[str, dict]]:
    """Les pistes de massa puntual amb alguna marca: [(nom, {fotograma: (x_px, y_px)})]. Com trk.llegeix_trk, pero sense
    saltar les de menys de 10 marques (les dels clics en tenen 5 i 1)."""
    p = trk._fills(ET.parse(cami).getroot())
    fora = []
    tracks = p.get("tracks")
    for item in ([] if tracks is None else tracks.findall("property[@name='item']")):
        obj = trk._dins(item)
        if obj is None or not obj.get("class", "").endswith("PointMass"):
            continue
        props = trk._fills(obj)
        fd = props.get("framedata")
        marques = {}
        for prop in ([] if fd is None else fd.findall("property")):
            marc = trk._dins(prop)
            if marc is None:
                continue
            m = trk._fills(marc)
            x, y = trk._num(m, "x"), trk._num(m, "y")
            if x is not None and y is not None:
                marques[int(prop.get("name").strip("[]"))] = (x, y)
        if marques:
            fora.append((trk._text(props, "name", "?"), marques))
    return fora


def llegeix_copies(carpeta: Path, esperats: dict) -> tuple[dict, list]:
    """Els clics de les copies, en el format de clics_creu.json pero en pixels, i els problemes del C0."""
    clics, prob = {}, []
    for cami in sorted(carpeta.glob("*.trk")):
        n = cami.stem.removesuffix("_repeticio")
        ses = "repeticio" if cami.stem.endswith("_repeticio") else "clics"
        if n not in esperats:
            prob.append(f"{cami.name}: no es cap presa de la llista")
            continue
        orig = trk.llegeix_trk(ARREL / "dades" / "video" / "tracker" / n.split("_")[0] / f"{n}.trk")
        copia = trk.llegeix_trk(cami)
        if copia.get("video") != orig.get("video") or copia["delta_t_s"] != orig["delta_t_s"]:
            prob.append(f"{cami.name}: no es la mateixa presa (video {copia.get('video')!r}, pas {copia['delta_t_s']} s)")
            continue
        qui = {}
        for nom, marques in pistes_de_la_copia(cami):
            b = nom.lower()
            q = ("pivot" if "pivot" in b or "fix" in b else "rodament" if "rodament" in b
                 else "extrem" if "extrem" in b else None)
            if q is None or q in qui:
                prob.append(f"{cami.name}: la pista {nom!r} té {len(marques)} marques de més (s'ha d'esborrar)")
                continue
            qui[q] = marques
        falten = [q for q in ("pivot", "rodament", "extrem") if q not in qui]
        if falten:
            prob.append(f"{cami.name}: falta la pista {', '.join(falten)}")
        if "pivot" in qui and len(qui["pivot"]) != 1:
            prob.append(f"{cami.name}: el pivot té {len(qui['pivot'])} marques, i n'ha de tenir 1")
        for q in ("rodament", "extrem"):
            k = sorted(qui.get(q, {}))
            if q in qui and k != esperats[n]:
                on = k if len(k) <= 10 else f"{len(k)} marques, del {k[0]} al {k[-1]} (pista vella?)"
                prob.append(f"{cami.name}: {q} als fotogrames {on}, i havien de ser {esperats[n]}")
        llista = [{"que": q, "fotograma": int(k), "x": float(xy[0]), "y": float(xy[1])}
                  for q in ("pivot", "rodament", "extrem") for k, xy in sorted(qui.get(q, {}).items())]
        clics.setdefault(n, {})[ses] = llista if ses == "clics" else {"clics": llista}
    for n, d in clics.items():
        if "clics" not in d:
            prob.append(f"{n}: hi ha la repetició però no la primera sessió")
    return clics, prob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--carpeta", type=Path, default=COPIES, help="d'on es llegeixen les copies (si no es la de debo, "
                                                                 "no es desa res)")
    args = ap.parse_args()
    prova = args.carpeta.resolve() != COPIES.resolve()
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    nous = json.loads(cac.CLICS_NOUS.read_text(encoding="utf-8"))["preses"]
    vells = json.loads(cac.CLICS_VELLS.read_text(encoding="utf-8"))["preses"]
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    llista = json.loads(LLISTA.read_text(encoding="utf-8"))
    esperats = {n: x["fotogrames"] for qui in ("pilot", "pas2") for n, x in llista[qui].items()}
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("La creu en moviment (pla-correccio-biaix.md, «La dissetena tanda»)"
          + (f"\n(PROVA: les copies de {args.carpeta}, i no es desa res)" if prova else ""))
    lect, ini = {}, {}
    for n in cac.preses_clicades():
        x = cac.llegeix_presa(n, nous, vells, retalls)
        t, A, G = x["t"], x["A"], x["G"]
        w = (t >= max(0.0, x["ta"] - 0.4)) & (t <= x["ta"] - 0.02)
        ra = descomp(A[w].mean(0), x["clics"]["Ai"], x["P"])
        rg = descomp(G[w].mean(0), x["clics"]["Gi"], x["clics"]["Ai"])
        ini[n] = {"rodament": ra, "extrem": rg}
        lect[n] = x
    if not (args.carpeta.is_dir() and any(args.carpeta.glob("*.trk"))):
        mirant[0] = False
        print(f"\n(no hi ha cap copia a {args.carpeta})")
        return 1

    mov, c0 = llegeix_copies(args.carpeta, esperats)
    mira(not c0, f"C0 les copies son les preses i porten nomes els clics: {len(mov)} preses"
                 + ("" if not c0 else "\n        " + "\n        ".join(c0)))
    if c0:
        mirant[0] = False
        print("\nUn criteri de l'instrument falla: no s'ensenya cap Delta.")
        return 1

    fora, f1 = {}, []
    for n, d in mov.items():
        x = lect[n]
        sessions = [("1", d["clics"])] + ([("2", d["repeticio"]["clics"])] if "repeticio" in d else [])
        fora[n] = {"grup": n.split("_")[0], "sessions": {}}
        for s, clics in sessions:
            px = lambda c: np.array([c["x"], c["y"]])   # noqa: E731
            piv = [px(c) for c in clics if c["que"] == "pivot"]
            rod = {c["fotograma"]: px(c) for c in clics if c["que"] == "rodament"}
            ext = {c["fotograma"]: px(c) for c in clics if c["que"] == "extrem"}
            dp = float(np.hypot(*(piv[0] - x["P"])))
            f1.append((n, s, "pivot", dp, dp < 8.0))
            files = []
            for fr in sorted(rod):
                i = np.nonzero(x["k"] == fr)[0]
                if not len(i):
                    f1.append((n, s, f"fotograma {fr}", math.inf, False))
                    continue
                i = int(i[0])
                da, dg = float(np.hypot(*(x["A"][i] - rod[fr]))), float(np.hypot(*(x["G"][i] - ext[fr])))
                f1.append((n, s, f"fotograma {fr}", max(da, dg), da < 15.0 and dg < 15.0))
                ca = descomp(x["A"][i], rod[fr], x["P"])
                cg = descomp(x["G"][i], ext[fr], rod[fr])
                files.append({"fotograma": fr, "t_des_de_amollada_s": float(x["t"][i] - x["ta"]),
                              "creu_rodament_px": ca, "creu_extrem_px": cg,
                              "delta_costat_px": [ca[0] - ini[n]["rodament"][0], cg[0] - ini[n]["extrem"][0]],
                              "delta_llarg_px": [ca[1] - ini[n]["rodament"][1], cg[1] - ini[n]["extrem"][1]]})
            fora[n]["sessions"][s] = files
    dolents = [f for f in f1 if not f[4]]
    mira(not dolents, f"F1 els clics a prop (pivot < 8 px, gomets < 15 px de la creu): {len(f1) - len(dolents)} de {len(f1)}"
                      + ("" if not dolents else f": {dolents[:6]}"))
    mirant[0] = False
    permesos_dir = [ARREL / "dades" / "video" / "tracker", ARREL / "dades" / "net", args.carpeta.resolve()]
    permesos = {LLISTA.resolve(), cac.CLICS_NOUS.resolve(), cac.CLICS_VELLS.resolve(),
                (ARREL / "dades" / "retalls.json").resolve(), cac.comparar.CENTRES.resolve(),
                (ARREL / "parametres.json").resolve()}
    altres = sorted({str(Path(o).resolve()) for o in oberts
                     if ARREL in Path(o).resolve().parents and (ARREL / ".venv") not in Path(o).resolve().parents
                     and Path(o).suffix not in (".py", ".pyc", ".pth") and Path(o).resolve() not in permesos
                     and not any(dd in Path(o).resolve().parents for dd in permesos_dir)})
    mira(not altres, f"J nomes obre els fitxers dits: {altres or 'cap altre'}")
    if problemes:
        if not prova:
            SORTIDA.write_text(json.dumps({"problemes": problemes}, indent=1, ensure_ascii=False), encoding="utf-8")
        print("\nUn criteri de l'instrument falla: no s'ensenya cap Delta.")
        return 1

    print(f"\n{'presa':7}{'ses.':>5}{'fotog.':>8}{'t (s)':>8}   Delta de costat (px): rodament, extrem   "
          "| al llarg (informatiu)")
    for n, d in fora.items():
        for s, files in d["sessions"].items():
            for r in files:
                print(f"{n:7}{s:>5}{r['fotograma']:>8}{r['t_des_de_amollada_s']:8.3f}   {r['delta_costat_px'][0]:+8.2f} "
                      f"{r['delta_costat_px'][1]:+8.2f}   | {r['delta_llarg_px'][0]:+7.2f} {r['delta_llarg_px'][1]:+7.2f}")
    rms = lambda v: float(np.sqrt(np.mean(np.square(v)))) if len(v) else math.nan   # noqa: E731
    pil = [v for n in PILOT if n in fora for r in fora[n]["sessions"]["1"] for v in r["delta_costat_px"]]
    resum = {"pilot": {"n": len(pil), "rms_px": rms(pil), "llindar_px": LLINDAR_PILOT_PX,
                       "segueix": bool(len(pil) == 20 and rms(pil) <= LLINDAR_PILOT_PX)}}
    print(f"\nEL PILOT: rms de Delta de costat de {len(pil)} mesures = {resum['pilot']['rms_px']:.2f} px "
          f"(llindar {LLINDAR_PILOT_PX} px): {'segueix al pas 2' if resum['pilot']['segueix'] else 'S ATURA (pla B)'}"
          + ("" if len(pil) == 20 else " -- falten mesures del pilot"))
    # Pas 2: la precisio del clic en moviment (les dues sessions d'una presa) i la sigma per grup i per gomet.
    rep = [n for n, d in fora.items() if "2" in d["sessions"]]
    dif = []
    for n in rep:
        a = {r["fotograma"]: r for r in fora[n]["sessions"]["1"]}
        for r in fora[n]["sessions"]["2"]:
            if r["fotograma"] in a:
                dif += [r["creu_rodament_px"][0] - a[r["fotograma"]]["creu_rodament_px"][0],
                        r["creu_extrem_px"][0] - a[r["fotograma"]]["creu_extrem_px"][0]]
    s_cm = rms(dif) / math.sqrt(2.0) if dif else None
    resum["sigma_clic_moviment_px"] = s_cm
    resum["repetides"] = rep
    if s_cm is not None:
        print(f"La precisio del clic en moviment ({', '.join(rep)}, {len(dif)} diferencies): sigma_cm = {s_cm:.2f} px")
        for g in ("45", "90"):
            r_ = {}
            for k, q in ((0, "A"), (1, "G")):
                v = [r["delta_costat_px"][k] for n, d in fora.items() if d["grup"] == g for r in d["sessions"]["1"]]
                r_[f"rms_delta_{q}_px"] = rms(v)
                r_[f"n_{q}"] = len(v)
                r_[f"sigma_{q}_px"] = math.sqrt(max(rms(v) ** 2 - s_cm ** 2 - SIGMA_CR_PX ** 2, 0.0))
            resum[g] = r_
            print(f"   {g} graus: rms Delta rodament {r_['rms_delta_A_px']:.2f} px, extrem {r_['rms_delta_G_px']:.2f} px "
                  f"({r_['n_A']} mesures) -> sigma_Delta de la tanda {r_['sigma_A_px']:.2f} / {r_['sigma_G_px']:.2f} px")
    if prova:
        print("(prova: no es desa res)")
        return 0
    CLICS_MOV.write_text(json.dumps({"_que_es": "Els clics de la creu en moviment (pla-correccio-biaix.md, «La dissetena "
                                                "tanda»), llegits de les copies dels .trk. Registre: la font son les copies.",
                                     "font": str(COPIES.relative_to(ARREL)).replace("\\", "/"),
                                     "unitats": "pixels de la imatge, com els desa el .trk", "preses": mov},
                                    indent=1, ensure_ascii=False), encoding="utf-8")
    SORTIDA.write_text(json.dumps({"_que_es": "La creu en moviment (pla-correccio-biaix.md, «La dissetena tanda»; "
                                              "eines/creu_moviment.py).",
                                   "creu_inicial_px": ini, "preses": fora, "resum": resum}, indent=1,
                                  ensure_ascii=False), encoding="utf-8")
    print(f"-> {CLICS_MOV.relative_to(ARREL)}, {SORTIDA.relative_to(ARREL)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
