r"""El terra de soroll de l'experiment 4, amb la norma sencera, mesurat al repos.

Per que existeix. La lambda dels tirs reals surt de com se separen dues preses
(<ln(d/d0)>), i la d nomes vol dir alguna cosa per sobre del soroll de
seguiment. La taula del 23/09 (limitacions-preses-i-friccio.html, seccio 05)
donava un terra de ~2 graus, pero era:
  - nomes dels ANGLES, i la norma te quatre components: les velocitats surten
    d'una derivada de Savitzky-Golay, que amplifica el soroll;
  - d'abans de corregir la marca de l'articulacio i d'abans de les 10 preses
    noves de 45 graus, marcades molt mes netes.
Amb 2 graus, el metode de parelles no troba cap lambda als 90 ni als 150
(documentacio/pla-analisi-caos.md, 30/09 i 01/10). Aqui es mesura el de debo.

COM. Al repos l'estat de debo es fix (els angles quiets, omega = 0), o sigui
que tot el que s'hi mou es soroll. Per a cada presa, la desviacio tipus de cada
component (th1, th2, omega1/omega0, omega2/omega0) al voltant de la seva
mitjana, i sigma_d = l'arrel de la suma dels quadrats: el soroll d'UNA presa en
la norma de mesures.distancia_fases. El terra d'una PARELLA es
sqrt(sigma_A^2 + sigma_B^2), perque els dos sorolls son independents (T3 ho
comprova sobre la norma de debo).

LA BARRERA. De les preses reals nomes es llegeix el REPOS d'abans d'amollar,
amb la mateixa regla de tall que eines/prepara_versemblant.py (en Guillem ho va
autoritzar el 03/10). Cap mostra de moviment i cap simulacio.

DOS LIMITS, i van a la memoria:
  - en moviment el soroll pot ser mes gran (estela, el seguidor que s'arrossega):
    el repos dona UN MINIM del terra, no el terra;
  - el biaix de theta1 (constant dins de cada presa) no hi surt, perque una
    constant no es desvia de la seva mitjana. Es dona a part, sense sumar.

Us:
    .venv\\Scripts\\python eines\\terra_de_soroll.py
    .venv\\Scripts\\python eines\\terra_de_soroll.py --json dades/divergencia/terra_2026-10-03.json
    .venv\\Scripts\\python eines\\terra_de_soroll.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_coeffs, savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol import dades  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import distancia_fases, omega_zero  # noqa: E402
from prepara_versemblant import SIGMES_AMOLLADA  # noqa: E402  la MATEIXA regla de tall

INDEX = ARREL / "dades" / "net" / "preses.json"
FINESTRES_MINIMES = 2
# Un repos de menys de dues finestres de filtre no es pot fer servir: dins
# d'una finestra les mostres son gairebe la mateixa dada (el filtre les ha
# promitjat), i la desviacio que en sortiria seria massa petita.


def repos_complet(cami_csv, amollada_s: float, incertesa_s: float,
                  finestra_mostres: int, fps: float):
    """(t, estats (4, N)) del REPOS d'una presa real, i res mes.

    ES L'UNICA FUNCIO D'AQUESTA EINA QUE OBRE UN CSV REAL. La regla de tall es
    la de prepara_versemblant.repos_de_la_presa, amb la mateixa constant: fora
    la primera mitja finestra (vora del filtre) i, abans d'amollar, mitja
    finestra mes tres vegades la incertesa de l'amollada. Aquella funcio nomes
    torna els angles; aqui calen tambe les velocitats.
    """
    taula, _ = dades.carrega(cami_csv)
    t = taula["t"].to_numpy()
    mitja = 0.5 * finestra_mostres / fps
    fi = amollada_s - mitja - SIGMES_AMOLLADA * incertesa_s
    sel = (t >= mitja) & (t < fi)
    x = np.array([taula[c].to_numpy()[sel] for c in ("theta1", "theta2", "omega1", "omega2")])
    return t[sel], x


def soroll_de_la_presa(x, omega0: float) -> dict:
    """La desviacio tipus de cada component, adimensional, i sigma_d (radians d'angle).

    Les velocitats es divideixen per omega0, com fa distancia_fases: aixi els
    quatre numeros son comparables i sigma_d es el soroll en la norma de debo.
    """
    x = np.asarray(x, dtype=float)
    s = np.array([x[0].std(ddof=1), x[1].std(ddof=1),
                  x[2].std(ddof=1) / omega0, x[3].std(ddof=1) / omega0])
    return {"components": s, "sigma_d": float(np.sqrt(np.sum(s * s)))}


def soroll_esperat(sigma_fotograma_rad: float, finestra: int, ordre: int, fps: float):
    """El que hauria de donar un soroll blanc d'un fotograma, despres del filtre.

    (angle, velocitat): la suma dels quadrats dels coeficients de
    Savitzky-Golay, de suavitzar i de derivar. Serveix per a T1 i, sobre les
    preses, com a diagnostic: si el mesurat en surt molt, el soroll no es blanc.
    """
    g0 = math.sqrt(float(np.sum(savgol_coeffs(finestra, ordre) ** 2)))
    g1 = math.sqrt(float(np.sum(savgol_coeffs(finestra, ordre, deriv=1, delta=1.0 / fps) ** 2)))
    return sigma_fotograma_rad * g0, sigma_fotograma_rad * g1


def _estad(v) -> dict | None:
    v = np.asarray(v, dtype=float)
    if not len(v):
        return None
    return {"n": int(len(v)), "mediana": float(np.median(v)),
            "min": float(v.min()), "max": float(v.max())}


def mesura(index, p) -> dict:
    """Per presa i per grup. Nomes el repos de cada presa."""
    w0 = omega_zero(p)
    preses = {}
    for f in index:
        if f.get("fora_del_balanc"):
            continue
        meta = json.loads((ARREL / "dades" / "net" / f"{f['nom']}.json").read_text(encoding="utf-8"))
        a = meta.get("amollada") or {}
        if not a.get("hi_ha_amollada"):
            continue
        fps = float(meta["fps"])
        sg = meta["savgol"]
        fin, ordre = int(sg["finestra_mostres"]), int(sg["ordre"])
        t, x = repos_complet(ARREL / "dades" / "net" / f"{f['nom']}.csv", a["amollada_s"],
                             a["incertesa_s"], fin, fps)
        fila = {"grup": f"{f['grup_graus']:.0f}", "mostres": int(len(t)),
                "angles_graus": [float(v) for v in a["angles_amollada_graus"]],
                "curt": bool(len(t) < FINESTRES_MINIMES * fin)}
        if not fila["curt"]:
            s = soroll_de_la_presa(x, w0)
            fila["components_graus"] = [float(np.degrees(c)) for c in s["components"]]
            fila["sigma_d_graus"] = float(np.degrees(s["sigma_d"]))
            # El diagnostic: el que donaria el soroll d'un fotograma de l'index
            # si fos blanc, despres del mateix filtre.
            e = [soroll_esperat(math.radians(sf), fin, ordre, fps) for sf in f["soroll_angular_graus"]]
            fila["esperat_si_blanc_graus"] = [math.degrees(e[0][0]), math.degrees(e[1][0]),
                                              math.degrees(e[0][1] / w0), math.degrees(e[1][1] / w0)]
        preses[f["nom"]] = fila

    grups = {}
    for g in sorted({v["grup"] for v in preses.values()}, key=float):
        noms = [n for n, v in preses.items() if v["grup"] == g and not v["curt"]]
        curts = [n for n, v in preses.items() if v["grup"] == g and v["curt"]]
        terres, d0s, raons = [], [], []
        for i, a in enumerate(noms):
            for b in noms[i + 1:]:
                terra = math.hypot(preses[a]["sigma_d_graus"], preses[b]["sigma_d_graus"])
                ea = np.radians([*preses[a]["angles_graus"], 0.0, 0.0])
                eb = np.radians([*preses[b]["angles_graus"], 0.0, 0.0])
                d0 = math.degrees(float(distancia_fases(ea, eb, w0)))
                terres.append(terra)
                d0s.append(d0)
                raons.append(d0 / terra)
        comp = np.array([preses[n]["components_graus"] for n in noms]) if noms else np.zeros((0, 4))
        esp = np.array([preses[n]["esperat_si_blanc_graus"] for n in noms]) if noms else np.zeros((0, 4))
        raons_a = np.array(raons)
        grups[g] = {
            "preses": noms,
            "curtes_fora": curts,
            "sigma_d_presa_graus": _estad([preses[n]["sigma_d_graus"] for n in noms]),
            "components_mediana_graus": dict(zip(("theta1", "theta2", "omega1/omega0", "omega2/omega0"),
                                                 [float(v) for v in np.median(comp, axis=0)])) if noms else None,
            "esperat_si_blanc_mediana_graus": dict(zip(("theta1", "theta2", "omega1/omega0", "omega2/omega0"),
                                                       [float(v) for v in np.median(esp, axis=0)])) if noms else None,
            "parelles": len(terres),
            "terra_parella_graus": _estad(terres),
            "d0_parella_graus": _estad(d0s),
            "rao_d0_terra": _estad(raons),
            "fraccio_parelles_d0_per_sobre_del_terra": float(np.mean(raons_a > 1)) if len(raons) else None,
            "fraccio_parelles_d0_per_sobre_de_3_terres": float(np.mean(raons_a > 3)) if len(raons) else None,
        }
    return {"omega0_rad_s": w0, "per_presa": preses, "per_grup": grups}


def imprimeix(r) -> None:
    print(f"omega0 = {r['omega0_rad_s']:.3f} rad/s. Graus equivalents (la d en radians, en graus).\n")
    print(f"{'grup':>5} {'n':>3} {'curtes':>6}  {'sigma_d presa':>18}  {'th1':>6} {'th2':>6} "
          f"{'w1/w0':>6} {'w2/w0':>6}  {'terra parella':>18}  {'d0 parella':>8}  {'d0/terra':>8} "
          f"{'>1':>5} {'>3':>5}")
    for g, v in r["per_grup"].items():
        s, tp, d0, ra = v["sigma_d_presa_graus"], v["terra_parella_graus"], v["d0_parella_graus"], v["rao_d0_terra"]
        c = v["components_mediana_graus"] or {}
        print(f"{g:>5} {len(v['preses']):>3} {len(v['curtes_fora']):>6}  "
              f"{s['mediana']:6.3f} ({s['min']:.3f}-{s['max']:.3f})  "
              f"{c.get('theta1', 0):6.3f} {c.get('theta2', 0):6.3f} {c.get('omega1/omega0', 0):6.3f} "
              f"{c.get('omega2/omega0', 0):6.3f}  {tp['mediana']:6.3f} ({tp['min']:.3f}-{tp['max']:.3f})  "
              f"{d0['mediana']:8.3f}  {ra['mediana']:8.2f} "
              f"{100 * v['fraccio_parelles_d0_per_sobre_del_terra']:4.0f}% "
              f"{100 * v['fraccio_parelles_d0_per_sobre_de_3_terres']:4.0f}%")
    print("\nDiagnostic: mediana del que donaria el soroll d'un fotograma de l'index, si fos blanc:")
    for g, v in r["per_grup"].items():
        e = v["esperat_si_blanc_mediana_graus"] or {}
        print(f"{g:>5}  th1 {e.get('theta1', 0):.3f}  th2 {e.get('theta2', 0):.3f}  "
              f"w1/w0 {e.get('omega1/omega0', 0):.3f}  w2/w0 {e.get('omega2/omega0', 0):.3f}")
    curtes = [n for v in r["per_grup"].values() for n in v["curtes_fora"]]
    if curtes:
        print(f"\nRepos massa curt (menys de {FINESTRES_MINIMES} finestres de filtre), fora: "
              + ", ".join(curtes))


# --------------------------------------------------------------- l'autoprova

def autoprova() -> int:
    """T1-T3, fixats el 03/10/2026 abans d'executar-la."""
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    fps, fin, ordre = 240.0, 13, 3
    print("Autoprova de eines/terra_de_soroll.py (criteris fixats el 03/10/2026)\n")

    # T1. Un repos fabricat amb soroll blanc conegut, passat pel MATEIX filtre
    # que la conversio (pendol/tracker.py): angles suavitzats i velocitats per
    # derivada de Savitzky-Golay. 6000 mostres (25 s), perque l'estimacio
    # d'una desviacio amb mostres correlacionades tingui poc error propi.
    print("T1. Un repos fabricat amb soroll blanc: la desviacio de cada component a < 10 %")
    rng = np.random.default_rng(31)
    sig = math.radians(0.3)
    cru = np.radians([150.0, 140.0])[:, None] + rng.normal(0.0, sig, (2, 6000))
    th = savgol_filter(cru, fin, ordre, axis=1)
    w = savgol_filter(cru, fin, ordre, deriv=1, delta=1.0 / fps, axis=1)
    s = soroll_de_la_presa(np.vstack([th, w])[:, 50:-50], w0)
    e_th, e_w = soroll_esperat(sig, fin, ordre, fps)
    esperat = np.array([e_th, e_th, e_w / w0, e_w / w0])
    rel = np.abs(s["components"] / esperat - 1)
    mira(bool(np.all(rel < 0.10)), "mesurat / esperat: " + ", ".join(f"{v:.3f}" for v in s["components"] / esperat))

    # T2. La barrera: sobre una presa fabricada amb repos i moviment, no torna
    # cap mostra de despres del limit, i totes son del repos.
    print("\nT2. Nomes llegeix el repos")
    desti = ARREL / "esborranys" / "_autoprova_terra"
    t = np.arange(0.0, 3.0, 1.0 / fps)
    amollada = 1.0
    angle = np.where(t < amollada, 0.5, 0.5 - 2.0 * (t - amollada) ** 2)
    cami = dades.desa(desti / "fals.csv", t, angle, angle, np.zeros_like(t), np.zeros_like(t),
                      {"origen": "autoprova de terra_de_soroll"})
    tr, xr = repos_complet(cami, amollada, 0.005, fin, fps)
    limit = amollada - 0.5 * fin / fps - SIGMES_AMOLLADA * 0.005
    mira(len(tr) > 0 and tr.max() < limit and np.all(xr[:2] == 0.5),
         f"{len(tr)} mostres, fins a {tr.max():.4f} s (limit {limit:.4f} s), totes del repos")

    # T3. El terra d'una parella: dos repos independents amb el mateix soroll.
    # La rms de distancia_fases mostra a mostra ha de ser sqrt(sA^2 + sB^2).
    print("\nT3. El terra d'una parella es sqrt(sA^2 + sB^2), sobre la norma de debo, a < 10 %")
    rng = np.random.default_rng(32)
    rep = []
    for _ in range(2):
        c = np.radians([150.0, 140.0])[:, None] + rng.normal(0.0, sig, (2, 6000))
        rep.append(np.vstack([savgol_filter(c, fin, ordre, axis=1),
                              savgol_filter(c, fin, ordre, deriv=1, delta=1.0 / fps, axis=1)])[:, 50:-50])
    sA, sB = soroll_de_la_presa(rep[0], w0)["sigma_d"], soroll_de_la_presa(rep[1], w0)["sigma_d"]
    # Els dos repos han de ser al mateix lloc, com dos tirs: es resta la mitjana.
    a = rep[0] - rep[0].mean(axis=1, keepdims=True)
    b = rep[1] - rep[1].mean(axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(distancia_fases(a, b, w0) ** 2)))
    formula = math.hypot(sA, sB)
    mira(abs(rms / formula - 1) < 0.10, f"rms {math.degrees(rms):.4f} graus, formula "
                                        f"{math.degrees(formula):.4f} graus ({100 * (rms / formula - 1):+.1f} %)")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> recupera un soroll conegut darrere del filtre, nomes llegeix el repos, i el")
    print("     terra d'una parella es la suma en quadratura dels dos sorolls.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    p = Parametres.des_de_json(ARREL / "parametres.json")
    index = json.loads(INDEX.read_text(encoding="utf-8"))["preses"]
    r = mesura(index, p)
    imprimeix(r)
    if args.json:
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        sortida = {"eina": "eines/terra_de_soroll.py", "pla": "documentacio/pla-analisi-caos.md",
                   "que_es": "soroll de seguiment al REPOS d'abans d'amollar, en la norma de "
                             "mesures.distancia_fases; es un MINIM del terra en moviment; sense el "
                             "biaix de theta1", **r}
        cami.write_text(json.dumps(sortida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
