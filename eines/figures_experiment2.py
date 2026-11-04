r"""Les figures de l'EXPERIMENT 2 que posen la simulacio al costat del pendol real (28/10).

SON COMPARACIONS NO FIRMADES. Les dues proves firmades (la Y i la Z) ja s'han fet, i els seus veredictes no es toquen
(documentacio/repartiment-28-10.md). La friccio, NOMES Dissipacio.firmada: cap coeficient llegit a ma.

  cinc_corbes       figures/cinc_corbes_150_3: el retrat (theta1 plegat, omega1) de la 150_3 real (corregida), del doble
                    simulat amb la friccio firmada i sense, des de les mateixes condicions corregides, i del simple amb
                    la seva friccio firmada i sense, des de theta = theta1 de la mateixa condicio. Cinc panells.
  poincare_friccio  figures/poincare_friccio: la seccio de Poincare del model amb la friccio firmada (una simulacio per
                    presa, des de les seves condicions corregides i tan llarga com la presa) a dalt, i la real
                    (corregida, amb les funcions de figures_reals) a sota, amb la mateixa escala d'E/E_v2. Els
                    creuaments del model passen pel mateix tall i el mateix filtre que els reals, sense soroll.

Les condicions inicials corregides surten de dades/divergencia/lyapunov_real_dues_parts_2026-10-27.json
(per_presa.amollada_despres_graus): les mateixes de la Z. Pla i criteris (CC1-CC4, PF1-PF4), escrits abans del codi:
documentacio/pla-analisi-caos.md, «28/10 · La feina del repartiment», 3 i 4.

    .venv\\Scripts\\python eines\\figures_experiment2.py cinc_corbes
    .venv\\Scripts\\python eines\\figures_experiment2.py poincare_friccio --processos 2
    .venv\\Scripts\\python eines\\figures_experiment2.py --autoprova
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import figures_reals as fr  # noqa: E402
import peus  # noqa: E402
from pendol import caos, simple  # noqa: E402
from pendol.fisica import Dissipacio, Parametres  # noqa: E402
from pendol.mesures import dif_angular  # noqa: E402
from pendol.simulador import simula  # noqa: E402

DUES_PARTS = ARREL / "dades" / "divergencia" / "lyapunov_real_dues_parts_2026-10-27.json"
PRESA_CINC = "150_3"
TOL_PUJADA = 1e-9          # CC2 i PF2: l'energia pot pujar, com a molt, aixo per l'energia inicial (des del repos)
GRUPS = fr.GRUPS


def _p() -> Parametres:
    return Parametres.des_de_json(ARREL / "parametres.json")


def _coma(x: float, fmt: str = ".2f") -> str:
    return format(x, fmt).replace(".", ",")


def condicions_corregides() -> dict:
    """{nom: (theta1, theta2, 0, 0)} amb els angles de sortida corregits, els de la Z (dues_parts del 27/10)."""
    pp = json.loads(DUES_PARTS.read_text(encoding="utf-8"))["per_presa"]
    return {n: np.r_[np.radians(v["amollada_despres_graus"]), 0.0, 0.0] for n, v in pp.items()}


def graella(pr) -> np.ndarray:
    """Els instants de la presa des de l'amollada, amb el 0 al davant: la simulacio hi surt exactament de la condicio."""
    tg = pr["t"][pr["t"] >= 0.0]
    return tg if tg[0] == 0.0 else np.r_[0.0, tg]


def puja(E) -> float:
    """El que l'energia puja de mes entre dues mostres seguides, per l'energia inicial (CC2, PF2)."""
    E = np.asarray(E, dtype=float)
    return float(max(0.0, np.max(np.diff(E))) / abs(E[0]))


# ------------------------------------------------------------- les cinc corbes

def dades_cinc(nom: str = PRESA_CINC) -> dict:
    """Les cinc corbes de la presa `nom`. Nomes obre l'index, la presa, els fitxers de la correccio i dues_parts."""
    p = _p()
    pr = fr.preses_reals(corregit=True, noms={nom})[0]
    mou = pr["t"] >= 0.0
    x0 = condicions_corregides()[nom]
    tg = graella(pr)
    dis_d = Dissipacio.firmada(float(pr["grup"]))
    _, y_f = simula(p, x0, t_eval=tg, dis=dis_d)
    _, y_c = simula(p, x0, t_eval=tg)
    s = simple.Simple.des_de_parametres(p)
    th0 = float(x0[0])
    dis_s = Dissipacio.firmada(math.degrees(abs(th0)), simple=True)
    _, z_f = simple.simula(s, [th0, 0.0], t_eval=tg, dis=dis_s)
    _, z_c = simple.simula(s, [th0, 0.0], t_eval=tg)
    return {"nom": nom, "grup": pr["grup"], "x0": x0, "t": tg, "t_real": pr["t"][mou], "real": pr["estats"][:, mou],
            "doble_friccio": y_f, "doble_sense": y_c, "simple_friccio": z_f, "simple_sense": z_c,
            "dis_doble": dis_d, "dis_simple": dis_s, "rho": pr["rho"]}


def amplitud_final_simple(d) -> float:
    """L'amplitud (graus) del simple amb friccio als ultims 3 s: el radi del forat que deixa a la figura."""
    fi = d["t"] >= d["t"][-1] - 3.0
    return float(np.degrees(np.max(np.abs(d["simple_friccio"][0][fi]))))


def figura_cinc(d):
    import matplotlib.pyplot as plt
    from pendol import estil

    estil.aplica()
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True, sharey=True, layout="constrained")
    panells = [
        (ax[0, 0], d["real"][0], d["real"][2], estil.REAL, f"Real: la {d['nom']} (angles corregits)"),
        (ax[0, 1], d["doble_friccio"][0], d["doble_friccio"][2], estil.SIMULACIO, "Doble simulat, amb la fricció firmada"),
        (ax[1, 1], d["doble_sense"][0], d["doble_sense"][2], estil.SIMULACIO, "Doble simulat, sense fricció"),
        (ax[0, 2], d["simple_friccio"][0], d["simple_friccio"][1], estil.SIMULACIO, "Simple simulat, amb la seva fricció firmada"),
        (ax[1, 2], d["simple_sense"][0], d["simple_sense"][1], estil.SIMULACIO, "Simple simulat, sense fricció"),
    ]
    linies = []
    for a, th, w, color, titol in panells:
        x, y = fr._amb_talls(np.degrees(dif_angular(th, 0.0)), w)
        (l,) = a.plot(x, y, color=color, lw=0.35, rasterized=True)
        linies.append(l)
        a.set_title(titol, loc="left", fontsize=10)
        a.set_xlim(-180, 180)
        a.set_xticks(range(-180, 181, 90))
    for a in ax[1, :]:
        a.set_xlabel("θ1 plegat (graus) · al simple, θ")
    for a in ax[:, 0]:
        a.set_ylabel("ω1 (rad/s) · al simple, ω")
    ax[1, 0].axis("off")
    ax[1, 0].text(0.0, 1.0,
                  "Les columnes: el pèndol real, el doble\nsimulat i el simple simulat.\n"
                  "Les files: amb fricció (a dalt) i\nsense (a sota).\n\n"
                  f"Totes surten de la mateixa condició\n(θ1, θ2) = ({_coma(math.degrees(d['x0'][0]), '.1f')}°, "
                  f"{_coma(math.degrees(d['x0'][1]), '.1f')}°), en repòs;\nel simple, de θ = θ1.\n\n"
                  f"{_coma(float(d['t'][-1]), '.1f')} s, des de l'amollada.\n\n"
                  f"El forat del simple amb fricció: al final,\nencara oscil·la amb "
                  f"{_coma(amplitud_final_simple(d), '.0f')}°. Dissipa\nmés a poc a poc que el doble.",
                  transform=ax[1, 0].transAxes, va="top", fontsize=9.5, color=estil.TINTA_2)
    peus.peu(fig, 0.0, -0.01,
             "COMPARACIÓ NO FIRMADA (les proves firmades, la Y i la Z, ja estan fetes i no es toquen). Real: els angles "
             "de comparar.corregeix_presa; les velocitats, les de la conversió més la derivada del que mou la correcció. "
             "Als 150° la correcció no està validada. Simulat: DOP853 amb la tolerància estricta, als mateixos instants que "
             "la presa, des de les condicions corregides (les de la Z). La fricció és la firmada el 27/10 "
             "(Dissipacio.firmada: la del doble per a 90° i 150°, i la del simple, de les preses 1.x). El doble té un espai "
             "de fases de 4 dimensions i aquí se'n veu una ombra de 2: els creuaments de les seves corbes són aparents. "
             "El simple en té 2, i les seves corbes no es poden creuar.",
             fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    return fig, linies


def metriques_cinc(d) -> dict:
    p = _p()
    s = simple.Simple.des_de_parametres(p)
    Ef = caos.energia_relativa(d["doble_friccio"], p)
    Ec = caos.energia_relativa(d["doble_sense"], p)
    E0 = simple.energia(0.0, 0.0, s)          # el simple, des del repos, com el doble
    Sf = simple.energia(d["simple_friccio"][0], d["simple_friccio"][1], s) - E0
    Sc = simple.energia(d["simple_sense"][0], d["simple_sense"][1], s) - E0
    return {"figura": "cinc_corbes_" + d["nom"], "eina": "eines/figures_experiment2.py cinc_corbes",
            "que_es": "COMPARACIO NO FIRMADA: la presa real (corregida) i quatre simulacions des de la mateixa condicio",
            "presa": d["nom"], "condicio_graus": np.degrees(d["x0"][:2]).tolist(),
            "condicions_de": str(DUES_PARTS.relative_to(ARREL)).replace("\\", "/") + " (per_presa.amollada_despres_graus)",
            "durada_s": float(d["t"][-1]), "mostres": int(len(d["t"])),
            "friccio_doble": d["dis_doble"].com_a_dict(), "friccio_simple": d["dis_simple"].com_a_dict(),
            "rho": d["rho"],
            "energia_final_sobre_inicial": {"doble_friccio": float(Ef[-1] / Ef[0]), "simple_friccio": float(Sf[-1] / Sf[0])},
            "amplitud_final_simple_graus": amplitud_final_simple(d),
            "deriva_sense_friccio": {"doble": float(np.max(np.abs(Ec - Ec[0])) / abs(Ec[0])),
                                     "simple": float(np.max(np.abs(Sc - Sc[0])) / abs(Sc[0]))}}


def fes_cinc(directori="figures") -> dict:
    from pendol import estil
    import matplotlib.pyplot as plt

    d = dades_cinc()
    fig, _ = figura_cinc(d)
    peus.desa(fig, "cinc_corbes_" + d["nom"], directori)
    plt.close(fig)
    m = metriques_cinc(d)
    cami = Path(directori) / f"cinc_corbes_{d['nom']}.json"
    cami.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  metriques: {cami}")
    return m


# ------------------------------------------------------------- el Poincare amb friccio

def _sim_presa(args):
    """Una presa simulada amb la friccio firmada del seu grup (per al pool)."""
    tg, x0, grup = args
    return simula(_p(), x0, t_eval=tg, dis=Dissipacio.firmada(float(grup)))[1]


def simula_preses(preses, processos: int = 1) -> list[dict]:
    """Les preses del model: la mateixa forma que les reals de figures_reals, perque passin per calcula_poincare."""
    x0s = condicions_corregides()
    tasques = [(graella(pr), x0s[pr["nom"]], pr["grup"]) for pr in preses]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            Y = pool.map(_sim_presa, tasques)
    else:
        Y = [_sim_presa(t) for t in tasques]
    return [{"nom": pr["nom"], "grup": pr["grup"], "t": tg, "estats": y, "cua_s": pr["cua_s"], "x0": x0}
            for pr, (tg, x0, _), y in zip(preses, tasques, Y)]


def resum_grup(punts, noms) -> dict:
    """Per grup: els creuaments de cada presa i l'E/E_v2 del seu ultim creuament (si en te)."""
    per = {n: [] for n in noms}
    for q in punts:
        per[q[0]].append(q)
    n = [len(v) for v in per.values()]
    ultima = [max(v, key=lambda q: q[1])[4] for v in per.values() if v]
    return {"preses": len(noms), "creuaments_per_presa": dict(zip(per, n)),
            "mediana_creuaments": float(np.median(n)),
            "mediana_E_ultim": float(np.median(ultima)) if ultima else None,
            "preses_sense_creuaments": int(sum(1 for x in n if x == 0))}


def dades_poincare(processos: int = 1) -> dict:
    p = _p()
    preses = fr.preses_reals(corregit=True)
    real = fr.calcula_poincare(preses, p)
    sims = simula_preses(preses, processos)
    model = fr.calcula_poincare(sims, p)
    resum = {g: {"model": resum_grup(model[g], [x["nom"] for x in sims if x["grup"] == g]),
                 "real": resum_grup(real[g], [x["nom"] for x in preses if x["grup"] == g])} for g in GRUPS}
    return {"real": real, "model": model, "sims": sims, "preses": preses, "resum": resum}


def figura_poincare(d):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from pendol import estil

    estil.aplica()
    E = [q[4] for k in ("model", "real") for g in GRUPS for q in d[k][g]]
    norm = Normalize(min(E), max(E))
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.8), sharex=True, sharey=True, layout="constrained")
    fila = {}
    for i, (k, mapa, nom_fila) in enumerate((("model", estil.mapa_simulacio(), "Model, fricció firmada"),
                                              ("real", estil.mapa_real(), "Real, angles corregits"))):
        for j, g in enumerate(GRUPS):
            a = ax[i, j]
            pts = sorted(d[k][g], key=lambda q: q[4])
            sc = a.scatter([q[2] for q in pts], [q[3] for q in pts], c=[q[4] for q in pts], cmap=mapa, norm=norm,
                           s=6, linewidths=0, rasterized=True)
            r = d["resum"][g][k]
            a.set_title(f"{nom_fila} · {g}° ({r['preses']} tirs)", loc="left", fontsize=10, pad=15)
            a.text(0.0, 1.01, f"{len(pts)} punts; mediana per tir {_coma(r['mediana_creuaments'], '.0f')}",
                   transform=a.transAxes, ha="left", va="bottom", fontsize=8, color=estil.TINTA_2)
            a.set_xlim(-180, 180)
            a.set_xticks(range(-180, 181, 90))
            fila[k] = sc
        ax[i, 0].set_ylabel("ω1 (rad/s)")
    for a in ax[1, :]:
        a.set_xlabel("θ1 plegat (graus)")
    fig.colorbar(fila["model"], ax=list(ax[0, :]), shrink=0.85, pad=0.02, label="E/E_v2 (model)")
    fig.colorbar(fila["real"], ax=list(ax[1, :]), shrink=0.85, pad=0.02, label="E/E_v2 (real)")
    peus.peu(fig, 0.0, -0.01,
             "COMPARACIÓ NO FIRMADA. Secció de Poincaré: el braç 2 passa per la vertical de sota girant en sentit positiu "
             "(θ2 = 0, ω2 > 0), per interpolació entre mostres, amb el mateix filtre als dos (res al tram de repòs final, i "
             "un sol punt per passada de θ2 de −0,5° a +0,5°). Model: una simulació per presa, des de les seves condicions "
             "corregides (les de la Z), amb la fricció firmada del seu grup (Dissipacio.firmada), als mateixos instants que "
             "la presa i sense soroll de seguiment. Real: els angles de comparar.corregeix_presa; als 150° la correcció no "
             "està validada. El color és l'energia de cada punt, amb la mateixa escala a les dues files. El model està "
             "validat qualitativament i en ordres de magnitud, no al detall.",
             fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    return fig, norm


def metriques_poincare(d) -> dict:
    sims = d["sims"]
    return {"figura": "poincare_friccio", "eina": "eines/figures_experiment2.py poincare_friccio",
            "que_es": "COMPARACIO NO FIRMADA: la seccio de Poincare del model amb la friccio firmada i la real (corregida)",
            "condicions_de": str(DUES_PARTS.relative_to(ARREL)).replace("\\", "/") + " (per_presa.amollada_despres_graus)",
            "friccio": {g: Dissipacio.firmada(float(g)).com_a_dict() for g in GRUPS},
            "tall_i_filtre": "figures_reals.calcula_poincare, el mateix per al model i per al real",
            "escala_E_sobre_Ev2": [min(q[4] for k in ("model", "real") for g in GRUPS for q in d[k][g]),
                                   max(q[4] for k in ("model", "real") for g in GRUPS for q in d[k][g])],
            "resum": d["resum"],
            "punts": {k: {g: len(d[k][g]) for g in GRUPS} for k in ("model", "real")},
            "durada_simulada_s": {x["nom"]: float(x["t"][-1]) for x in sims}}


def fes_poincare(directori="figures", processos=1) -> dict:
    from pendol import estil
    import matplotlib.pyplot as plt

    d = dades_poincare(processos)
    fig, _ = figura_poincare(d)
    peus.desa(fig, "poincare_friccio", directori)
    plt.close(fig)
    m = metriques_poincare(d)
    cami = Path(directori) / "poincare_friccio.json"
    cami.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  metriques: {cami}")
    for g in GRUPS:
        a, b = m["resum"][g]["model"], m["resum"][g]["real"]
        print(f"  {g}: creuaments per tir (mediana) model {a['mediana_creuaments']:g}, real {b['mediana_creuaments']:g}; "
              f"E/E_v2 de l'ultim (mediana) model {a['mediana_E_ultim']}, real {b['mediana_E_ultim']}; "
              f"sense creuaments: model {a['preses_sense_creuaments']}, real {b['preses_sense_creuaments']}")
    return m


# ------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def _de_dades(oberts) -> list[str]:
    arrel = (ARREL / "dades").resolve()
    return sorted({str(Path(x).resolve().relative_to(arrel)).replace("\\", "/") for x in oberts
                   if Path(x).resolve().is_relative_to(arrel)})


def _empremta(obj) -> str:
    h = hashlib.sha256()

    def passa(x):
        if isinstance(x, dict):
            for k in sorted(x):
                h.update(str(k).encode())
                passa(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                passa(v)
        elif isinstance(x, np.ndarray):
            h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, Dissipacio):
            h.update(repr(x).encode())
        else:
            h.update(repr(x).encode())
    passa(obj)
    return h.hexdigest()


def autoprova() -> int:
    """CC1-CC4 i PF1-PF4 del pla (28/10), escrits abans del codi. PF2 i PF3 sobre una presa de cada grup (les tres
    primeres de l'index): fer-les sobre les 35 vol dir simular-les dues vegades."""
    global _OBERTS
    problemes = []

    def mira(c, que):
        if not c:
            problemes.append(que)
        print(f"   {'ok ' if c else 'NO '} {que}", flush=True)

    sys.addaudithook(_registra)
    p = _p()
    print("Autoprova de eines/figures_experiment2.py\n")

    print("Les cinc corbes")
    _OBERTS = []
    d1 = dades_cinc()
    oberts, _OBERTS = _de_dades(_OBERTS), None
    d2 = dades_cinc()
    x0 = d1["x0"]
    mira(all(np.array_equal(d1[k][:, 0], x0) for k in ("doble_friccio", "doble_sense"))
         and all(np.array_equal(d1[k][:, 0], [x0[0], 0.0]) for k in ("simple_friccio", "simple_sense"))
         and d1["t"][0] == 0.0,
         f"CC1 les quatre simulacions surten exactament de la condicio, a t = 0: {np.degrees(x0[:2]).round(4).tolist()} graus")
    s = simple.Simple.des_de_parametres(p)
    pd = puja(caos.energia_relativa(d1["doble_friccio"], p))
    ps = puja(simple.energia(d1["simple_friccio"][0], d1["simple_friccio"][1], s)
              - simple.energia(0.0, 0.0, s))
    mira(pd <= TOL_PUJADA and ps <= TOL_PUJADA,
         f"CC2 amb friccio l'energia no puja mai: doble {pd:.1e}, simple {ps:.1e} (com a molt {TOL_PUJADA:g})")
    mira(_empremta({k: v for k, v in d1.items()}) == _empremta({k: v for k, v in d2.items()}),
         "CC3 dues passades donen el mateix bit a bit")
    permesos = {"net/preses.json", f"net/{PRESA_CINC}.csv", f"net/{PRESA_CINC}.json",
                "versemblant/centres_cercle.json", "versemblant/correccio_clics.json",
                "divergencia/" + DUES_PARTS.name}
    mira(set(oberts) <= permesos and not any(x.startswith("sim/") for x in oberts),
         "CC4 de dades/ nomes obre: " + ", ".join(oberts))

    print("\nEl Poincare amb friccio")
    preses = fr.preses_reals(corregit=True)
    real = fr.calcula_poincare(preses, p)
    real_b = fr.calcula_poincare(fr.preses_reals(corregit=True), p)
    mira(real == real_b, f"PF1 els punts reals son bit a bit els de figures_reals: "
                         f"{', '.join(f'{g}: {len(real[g])}' for g in GRUPS)}")
    una = [next(x for x in preses if x["grup"] == g) for g in GRUPS]
    _OBERTS = []
    a = simula_preses(una)
    oberts_sim, _OBERTS = _de_dades(_OBERTS), None
    b = simula_preses(una)
    pujades = [puja(caos.energia_relativa(x["estats"], p)) for x in a]
    mira(max(pujades) <= TOL_PUJADA and all(np.array_equal(x["estats"][:, 0], x["x0"]) for x in a),
         f"PF2 l'energia del model no puja mai ({', '.join(f'{x['nom']} {v:.1e}' for x, v in zip(a, pujades))}), "
         f"i cada presa surt de la seva condicio")
    mira(all(np.array_equal(x["estats"], y["estats"]) for x, y in zip(a, b))
         and fr.calcula_poincare(a, p) == fr.calcula_poincare(b, p),
         f"PF3 reproduible bit a bit ({', '.join(x['nom'] for x in a)})")
    _OBERTS = []
    fr.preses_reals(corregit=True)
    oberts_real, _OBERTS = _de_dades(_OBERTS), None
    nets = {x for x in oberts_real if x.startswith("net/")}
    mira(not any(x.startswith("sim/") for x in oberts_real + oberts_sim)
         and set(oberts_real) - nets <= {"versemblant/centres_cercle.json", "versemblant/correccio_clics.json"}
         and set(oberts_sim) <= {"divergencia/" + DUES_PARTS.name},
         f"PF4 de dades/ nomes obre les preses ({len(nets)} fitxers de net/), "
         + ", ".join(sorted(set(oberts_real) - nets) + oberts_sim))
    print()
    if problemes:
        print("AUTOPROVA FALLADA:\n" + "\n".join(f"  - {q}" for q in problemes))
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("quines", nargs="*", help="cinc_corbes i/o poincare_friccio")
    ap.add_argument("--processos", type=int, default=1)
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a <figures>/memoria/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.memoria:
        peus.memoria()
        args.figures = str(Path(args.figures) / "memoria")
    if args.processos > 2:
        ap.error("com a molt 2 processos: el processador es comparteix amb el magnetic (repartiment del 28/10)")
    if not args.quines or set(args.quines) - {"cinc_corbes", "poincare_friccio"}:
        ap.error("digues quina figura: cinc_corbes i/o poincare_friccio")
    if "cinc_corbes" in args.quines:
        fes_cinc(args.figures)
    if "poincare_friccio" in args.quines:
        fes_poincare(args.figures, args.processos)
    peus.escriu(args.figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
