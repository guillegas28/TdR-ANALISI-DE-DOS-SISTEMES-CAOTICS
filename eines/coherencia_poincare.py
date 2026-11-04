r"""La coherencia del Poincare real: C1, C2 i el repos final de cada presa. NOMES PRESES REALS, cap simulacio.

Pla: documentacio/pla-analisi-caos.md, «25/10 · La coherencia del Poincare real (C1 i C2)», «El filtre de
creuaments» i «Tres feines» (el 3). Puja d'esborranys/_coherencia_poincare.py i _coherencia_poincare_diag.py amb
la sortida identica (CQ1, comprovada fora de l'eina). Es la font dels numeros de les declaracions per a la
memoria (seccio 5 de documentacio/esborrany-memoria-experiment2-real.md).

  C1  a cada tir, Spearman(t, E/E_v2) dels seus creuaments <= -0,9 (amb menys de 3 creuaments, no hi arriba):
      l'energia baixa, perque el pendol frega i ningu no li'n dona. No passa per construccio.
  C2  cap creuament als ultims cua_s de cada presa (el repos de la conversio). Amb el filtre, passa per
      construccio de la seva regla 1.

Tres versions dels punts:
  sense_filtre  els creuaments crus de caos.creuaments_interpolats, com abans del 25/10 (on la C2 va fallar)
  amb_filtre    els de figures_reals.calcula_poincare (el filtre de creuaments)
  corregit      els mateixos, amb els angles de comparar.corregeix_presa

I el repos final de cada presa (sense corregir): theta2 als ultims cua_s (mitjana, desviacio, pic a pic) i |omega2|
maxima, i als cua_s d'abans; quantes fan <= 0,5 graus de pic a pic, la h del filtre.

    .venv\Scripts\python.exe eines\coherencia_poincare.py
    .venv\Scripts\python.exe eines\coherencia_poincare.py --autoprova
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import figures_reals as fr  # noqa: E402
from pendol import caos  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import dif_angular  # noqa: E402

PUJADA = 0.05   # informatiu: les pujades d'E/E_v2 entre creuaments seguits que es llisten
SPEARMAN = -0.9


def punts_crus(preses, p_mod) -> dict:
    """Els creuaments sense cap filtre, tal com els treia figures_reals.calcula_poincare abans del 25/10."""
    Ev2 = caos.energia_volteig_2(p_mod)
    r = {}
    for g in fr.GRUPS:
        punts = []
        for p in preses:
            if p["grup"] != g:
                continue
            mou = p["t"] >= 0.0
            c = caos.creuaments_interpolats(p["t"][mou], p["estats"][:, mou])
            if not len(c["t_s"]):
                continue
            E = np.atleast_1d(caos.energia_relativa(c["estats"].T, p_mod)) / Ev2
            for k in range(len(c["t_s"])):
                punts.append((p["nom"], float(c["t_s"][k]), float(np.degrees(dif_angular(c["estats"][k, 0], 0.0))),
                              float(c["estats"][k, 2]), float(E[k])))
        r[g] = punts
    return r


def coherencia(preses, poi) -> dict:
    """La C1 i la C2 sobre uns punts: les dades de cada tir i el veredicte."""
    tirs, falla1, falla2 = {}, [], []
    for g in fr.GRUPS:
        for p in (x for x in preses if x["grup"] == g):
            n = p["nom"]
            final = float(p["t"][-1])
            q = [x for x in poi[g] if x[0] == n]
            t = np.array([x[1] for x in q])
            E = np.array([x[4] for x in q])
            assert np.all(np.diff(t) > 0)
            rs = float(spearmanr(t, E).statistic) if len(q) >= 3 else None
            dE = np.diff(E)
            puj = dE[dE > PUJADA]
            cua = int(np.sum(t > final - p["cua_s"]))
            if not (rs is not None and rs <= SPEARMAN):
                falla1.append(n)
            if cua:
                falla2.append(n)
            tirs[n] = {"grup": g, "creuaments": len(q), "spearman": rs, "pujades": len(puj),
                       "pujada_max": float(puj.max()) if len(puj) else None,
                       "ultim_abans_del_final_s": final - float(t[-1]) if len(q) else None,
                       "E_min": float(E.min()) if len(q) else None, "a_la_cua": cua}
    return {"C1": {"passa": not falla1, "no_hi_arriben": falla1},
            "C2": {"passa": not falla2, "amb_creuaments_a_la_cua": falla2,
                   "creuaments_a_la_cua": sum(d["a_la_cua"] for d in tirs.values())},
            "tirs": tirs}


def repos_final(preses) -> dict:
    """theta2 i |omega2| als ultims cua_s de cada presa i als cua_s d'abans; quantes fan <= h de pic a pic."""
    h = math.degrees(fr.H_PASSADA)
    per_presa = {}
    for p in preses:
        t, X = p["t"], p["estats"]
        fi = t[-1]
        th2 = np.degrees(dif_angular(X[1], 0.0))
        w2 = X[3]
        cua = t > fi - p["cua_s"]
        abans = (t > fi - 2 * p["cua_s"]) & (t <= fi - p["cua_s"])
        per_presa[p["nom"]] = {"mitjana_graus": float(th2[cua].mean()), "desviacio_graus": float(th2[cua].std()),
                               "pic_a_pic_graus": float(np.ptp(th2[cua])), "omega2_max": float(np.abs(w2[cua]).max()),
                               "abans_pic_a_pic_graus": float(np.ptp(th2[abans])),
                               "abans_omega2_max": float(np.abs(w2[abans]).max())}
    quietes = {n: v for n, v in per_presa.items() if v["pic_a_pic_graus"] <= h}
    altres = {n: v["pic_a_pic_graus"] for n, v in per_presa.items() if n not in quietes}
    return {"h_graus": h, "per_presa": per_presa, "quietes": len(quietes),
            "pic_a_pic_maxim_de_les_quietes_graus": max(v["pic_a_pic_graus"] for v in quietes.values()),
            "les_altres_pic_a_pic_graus": altres}


def calcula() -> dict:
    p_mod = Parametres.des_de_json(ARREL / "parametres.json")
    preses = fr.preses_reals()
    pc = fr.preses_reals(corregit=True)
    return {"sense_filtre": coherencia(preses, punts_crus(preses, p_mod)),
            "amb_filtre": coherencia(preses, fr.calcula_poincare(preses, p_mod)),
            "corregit": coherencia(pc, fr.calcula_poincare(pc, p_mod)),
            "repos_final": repos_final(preses)}


def imprimeix(r):
    for versio in ("sense_filtre", "amb_filtre", "corregit"):
        v = r[versio]
        print(f"\n{versio}: per tir, creuaments, Spearman(t, E), pujades > {PUJADA:g} E_v2 (la mes gran), ultim "
              f"creuament abans del final, E/E_v2 minima, creuaments als ultims cua_s")
        for n, d in v["tirs"].items():
            rs = d["spearman"]
            print(f"   {n:7} {d['creuaments']:4}  {rs if rs is None else format(rs, '+.4f'):>8}  "
                  f"{d['pujades']:3} ({'-' if d['pujada_max'] is None else format(d['pujada_max'], '.3f')})  "
                  f"{d['ultim_abans_del_final_s']:7.2f} s  {d['E_min']:.4f}  {d['a_la_cua']}")
        print(f"   C1: {len(v['tirs']) - len(v['C1']['no_hi_arriben'])} de {len(v['tirs'])} -> "
              f"{'ok' if v['C1']['passa'] else 'FALLA'}; C2: {v['C2']['creuaments_a_la_cua']} creuaments a "
              f"{len(v['C2']['amb_creuaments_a_la_cua'])} tirs -> {'ok' if v['C2']['passa'] else 'FALLA'}")
    rf = r["repos_final"]
    print(f"\nRepos final (sense corregir): {rf['quietes']} de {len(rf['per_presa'])} preses fan <= "
          f"{rf['h_graus']:g} graus de pic a pic (la mes gran, {rf['pic_a_pic_maxim_de_les_quietes_graus']:.3f}); "
          "les altres: " + ", ".join(f"{n} {v:.3f}" for n, v in rf["les_altres_pic_a_pic_graus"].items()))


# --------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """CQ2 (pla, 25/10, «Tres feines»): reproduible bit a bit; de dades/ nomes obre les preses reals,
    centres_cercle.json i (des del pas D, 28/10, reescrita abans de correr) correccio_clics.json, i res de dades/sim/."""
    global _OBERTS
    problemes = []

    def mira(c, que):
        if not c:
            problemes.append(que)
        print(f"   {'ok ' if c else 'NO '} {que}", flush=True)

    print("Autoprova de eines/coherencia_poincare.py (CQ2)\n")
    sys.addaudithook(_registra)
    _OBERTS = []
    a = json.dumps(calcula(), sort_keys=True)
    oberts, _OBERTS = _OBERTS, None
    b = json.dumps(calcula(), sort_keys=True)
    mira(a == b, f"reproduible: empremta {hashlib.sha256(a.encode()).hexdigest()[:16]} les dues vegades")
    arrel_dades = (ARREL / "dades").resolve()
    dins = {str(Path(x).resolve()) for x in oberts if Path(x).resolve().is_relative_to(arrel_dades)}
    esperats = {str(fr.INDEX.resolve()), str(fr.CENTRES.resolve()), str(fr.CORRECCIO_CLICS.resolve())}
    for f in json.loads(fr.INDEX.read_text(encoding="utf-8"))["preses"]:
        if not f.get("fora_del_balanc"):
            esperats |= {str((fr.NET / f"{f['nom']}.csv").resolve()), str((fr.NET / f"{f['nom']}.json").resolve())}
    sim = [x for x in dins if "sim" in Path(x).relative_to(arrel_dades).parts]
    mira(dins == esperats and not sim, f"{len(dins)} fitxers oberts a dades/, {len(esperats)} esperats; de dades/sim/: "
                                       f"{len(sim)}" + ("" if dins == esperats else
                                                        f"; de mes: {sorted(dins - esperats)[:3]}; "
                                                        f"falten: {sorted(esperats - dins)[:3]}"))
    print()
    if problemes:
        print("AUTOPROVA FALLADA:\n" + "\n".join(f"  - {q}" for q in problemes))
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> reproduible, i nomes llegeix preses reals.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default="figures/coherencia_poincare.json")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    r = calcula()
    imprimeix(r)
    cami = Path(args.json)
    cami.parent.mkdir(parents=True, exist_ok=True)
    cami.write_text(json.dumps({"que_es": "la coherencia del Poincare real (C1, C2) i el repos final: NOMES PRESES "
                                          "REALS, cap simulacio",
                                "eina": "eines/coherencia_poincare.py",
                                "pla": "documentacio/pla-analisi-caos.md (25/10)", **r},
                               indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
