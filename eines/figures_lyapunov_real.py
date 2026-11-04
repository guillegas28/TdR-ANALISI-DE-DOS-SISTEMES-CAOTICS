r"""Les figures de la lambda dels tirs reals (pla: documentacio/pla-analisi-caos.md, 17 i 22/10).

Tres figures que NO barregen simulacio i preses reals, i una (la de la Z) que si, perque es la comparacio firmada:
  veins        NOMES REAL. <ln(d/d0)> contra el temps de les parelles de veins de debo a E/E_v2 1,7-3,5, amb
               la banda del bootstrap de tirs (16-84), la finestra i la recta.
  llindar_150  NOMES REAL. La lambda(D) dels 150 graus a 5 distancies, i la lambda* a D*.
  validacio    NOMES SIMULACIO. El metode dels veins contra la lambda de Benettin a les tres
               franges, i el pendol simple de control.
  z            (29/10) LA Z FIRMADA, feta el 27/10: la lambda* real contra les 20 realitzacions del model, i el
               diagnostic D4b (posterior, sense veredicte). NOMES DELS DOS JSON de la Z: no obre cap presa ni simula res.

Les dues reals passen pel pany de eines/lyapunov_real.py (dades/divergencia/lyapunov_real_aprovat.json).
Estil: pendol/estil.py (taronja = real, blau = simulacio; textos en tinta). PNG, PDF i JSON.

Des del 28/10, --corregit (amb veins i llindar_150): les preses corregides (lyapunov_real.preses_corregides, la
correccio que es firma, comparar.corregeix_presa) i el sufix _corregit. Des del 29/10, les sense corregir es
comparen amb la base del 24/10 (la 90_10 reconvertida), no amb la del 17/10.

    .venv\\Scripts\\python eines\\figures_lyapunov_real.py totes
    .venv\\Scripts\\python eines\\figures_lyapunov_real.py --autoprova
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

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import lyapunov_real as lr  # noqa: E402
import peus  # noqa: E402

DIV = ARREL / "dades" / "divergencia"
PAS0 = DIV / "lyapunov_real_pas0_2026-10-17.json"
PAS0B = DIV / "lyapunov_real_pas0b_2026-10-17.json"
# 29/10: la base del 24/10, la que va triar en Guillem quan es va reconvertir la 90_10 (abans, la del 17/10; la R2 dels
# 150 graus es identica a totes dues).
REALS = DIV / "lyapunov_real_reals_2026-10-24.json"
VALIDACIO = DIV / "lyapunov_real_validacio2_2026-10-17.json"
DUES_PARTS = DIV / "lyapunov_real_dues_parts_2026-10-27.json"     # 28/10: els resultats corregits, amb els clics
Z_R2 = DIV / "lyapunov_real_z_r2_2026-10-27.json"                 # 29/10: la Z firmada
Z_DIAGNOSTIC = DIV / "lyapunov_real_z_diagnostic_2026-10-27.json"
TEXT_CORREGIT = (" ANGLES CORREGITS amb comparar.corregeix_presa (l'eix de gir del cercle de cada presa i el repòs final; "
                 "als 45° i als 90°, amb els clics de la creu del 27/10). Als 150° la correcció no està validada.")


def _coma(x, fmt=".2f"):
    return format(x, fmt).replace(".", ",")


def _carrega(c):
    return json.loads(Path(c).read_text(encoding="utf-8"))


# ------------------------------------------------------------- les dades

def dades_veins(n_boot=lr.REMOSTREJOS, corregit=False) -> dict:
    """La corba real dels veins a 1,7-3,5, la banda del bootstrap i la recta. NOMES REAL. Amb corregit (28/10), les
    preses de lyapunov_real.preses_corregides()."""
    if not lr.APROVAT.exists():
        raise SystemExit("Pany tancat: no hi ha l'aprovacio de les preses reals.")
    pas02 = _carrega(PAS0B)
    p = lr._parametres()
    w0 = lr.omega_zero(p)
    Ev2 = lr.caos.energia_volteig_2(p)
    fora = pas02["fora_per_soroll"]
    totes = list(lr.preses_corregides()[0].values()) if corregit else lr.preses_reals()
    pr1 = [pr for pr in totes if pr["nom"] not in fora]
    R = pas02["resultat1"]["1,7-3,5"]
    Q, V, E, T = lr._entrada_r1([(pr["t"], pr["estats"]) for pr in pr1], p, Ev2, w0, lr.FRANGES_R1["1,7-3,5"])
    cfg = lr._cfg_r1(R["banda_rad"], R["sostre_rad"], T)
    par = lr.troba_veins(Q, V, E, periodic=cfg["periodic"], sep=cfg["sep"], vel=cfg["vel"], banda=cfg["banda"],
                         theiler=cfg["theiler"], cada=cfg["cada"], de_max=cfg["de_max"], temps=T,
                         separacio_creuada=lr.SEPARACIO_CREUADA_S)
    D = lr.segueix(Q, par, periodic=cfg["periodic"], sep=cfg["sep"], H=cfg["H"])
    lnd = np.log(D)
    lnr = lnd - lnd[:, :1]
    r = lr.pendent_veins(lnr, lnd, np.ones(len(par)), cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])
    rng = np.random.default_rng(lr.LLAVOR)
    pt = [(a, c) for a, _, c, _ in par]
    corbes, lam = [], []
    for _ in range(n_boot):
        w = lr.pesos_bootstrap(pt, rng.integers(0, len(Q), len(Q)), len(Q))
        if w.sum() <= 0:
            continue
        corbes.append((w @ lnr) / w.sum())
        lam.append(lr.pendent_veins(lnr, lnd, w, cfg["h0"], cfg["dt"], cfg["ln_sostre_3"])["lambda"])
    corbes = np.array(corbes)
    t = np.arange(cfg["H"] + 1) * cfg["dt"]
    m = lnr.mean(axis=0)
    a, b = int(round(r["finestra_s"][0] / cfg["dt"])), int(round(r["finestra_s"][1] / cfg["dt"]))
    pend, ordenada = np.polyfit(t[a:b + 1], m[a:b + 1], 1)
    return {"t_s": t.tolist(), "mitjana": m.tolist(), "p16": np.percentile(corbes, 16, axis=0).tolist(),
            "p84": np.percentile(corbes, 84, axis=0).tolist(), "finestra_s": r["finestra_s"],
            "lambda": float(pend), "ordenada": float(ordenada), "sigma": float(np.std(lam, ddof=1)),
            "parelles": len(par), "tirs": len({x[0] for x in par} | {x[2] for x in par})}


def dades_llindar(corregit=False) -> dict:
    """La lambda(D) dels 150 graus, del JSON del pas 3 (o, corregit, del de dues_parts del 27/10). NOMES REAL (ja
    calculat darrere del pany)."""
    r = _carrega(DUES_PARTS if corregit else REALS)["resultat2"]["150"]
    dg = r["diagnostic_lambda_D"]
    return {"D_graus": dg["D_graus"], "lambda_D": dg["lambda_D"], "D_estrella_graus": r["D_estrella_graus"],
            "lambda": r["lambda"], "sigma_estadistica": r["bootstrap"]["sigma_D"][0],
            "sigma_alineacio": r["mc_alineacio"]["desviacio"], "parelles": r["parelles"]}


def dades_z() -> dict:
    """La Z firmada (27/10) i el diagnostic D4b, NOMES DELS DOS JSON: cap presa, cap simulacio."""
    z, dg = _carrega(Z_R2), _carrega(Z_DIAGNOSTIC)
    return {"real": {k: z["real"][k] for k in ("lambda", "sigma")},
            "model": {"lambda": z["simulat"]["lambda"], "dispersio": z["simulat"]["dispersio"],
                      "realitzacions": z["simulat"]["lambda_realitzacions"]},
            "D4b": {"lambda": dg["D4b"]["lambda"], "dispersio": dg["D4b"]["dispersio"],
                    "realitzacions": dg["D4b"]["lambda_realitzacions"]},
            "diferencia_sigmes": dg["referencia"]["diferencia_sigmes"], "veredicte": z["veredicte"],
            "sigmes": z["regla"]["sigmes"], "preses": len(z["preses"]), "parelles": z["simulat"]["parelles"]}


def dades_validacio() -> dict:
    """El metode contra Benettin a les tres franges. NOMES SIMULACIO."""
    v = _carrega(VALIDACIO)["resultat1"]
    out = {}
    for nom, b in v.items():
        if not b.get("avaluable"):
            continue
        sor = [d["lambda"] for d in b["amb_soroll"]]
        sim = [s["lambda"] for s in b["simple_amb_soroll"]]
        out[nom] = {"benettin": b["benettin_mitjana"], "ideal": b["ideal"]["lambda"],
                    "soroll_mediana": float(np.median(sor)), "soroll_min": min(sor), "soroll_max": max(sor),
                    "simple_mediana": float(np.median(sim)), "passa": bool(b["passa"]),
                    "fraccio_caotiques": b["fraccio_caotiques"]}
    return out


# ------------------------------------------------------------- les figures

def _peu(fig, text):
    from pendol import estil
    peus.peu(fig, 0.01, -0.01, text, fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)


def dibuixa_veins(d, directori, corregit=False):
    import matplotlib.pyplot as plt
    from pendol import estil
    t = np.array(d["t_s"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axvspan(*d["finestra_s"], color=estil.GRAELLA, alpha=0.6, lw=0, label="finestra de l'ajust")
    ax.fill_between(t, d["p16"], d["p84"], color=estil.REAL, alpha=0.22, lw=0,
                    label="bootstrap de tirs (16–84 %)")
    ax.plot(t, d["mitjana"], color=estil.REAL, ls=estil.TRAC_REAL, label="preses reals, ⟨ln(d/d₀)⟩")
    tf = np.array(d["finestra_s"])
    ax.plot(tf, d["ordenada"] + d["lambda"] * tf, color=estil.TINTA, lw=1.2, ls=":",
            label=f"recta: λ = {_coma(d['lambda'])} ± {_coma(d['sigma'])} s⁻¹")
    ax.set_xlabel("temps des de la trobada dels dos veïns (s)")
    ax.set_ylabel("⟨ln(d/d₀)⟩")
    ax.set_title("Veïns de debò a les preses reals, E/E_v2 = 1,7–3,5")
    ax.set_xlim(0, t[-1])
    ax.legend(loc="upper left")
    _peu(fig, f"{d['parelles']} parelles d'estats de {d['tirs']} tirs de 90° i 150°; d en angles. "
              "Només preses reals: cap simulació. eines/lyapunov_real.py, pla del 17/10."
              + (TEXT_CORREGIT if corregit else ""))
    peus.desa(fig, "lyapunov_real_veins" + ("_corregit" if corregit else ""), directori)


def dibuixa_llindar(d, directori, corregit=False):
    import matplotlib.pyplot as plt
    from pendol import estil
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(d["D_graus"], d["lambda_D"], color=estil.REAL, ls=estil.TRAC_REAL, marker="o", ms=6,
            label="λ(D), diagnòstic sense veredicte de forma")
    s = math.hypot(d["sigma_estadistica"], d["sigma_alineacio"])
    ax.errorbar([d["D_estrella_graus"]], [d["lambda"]], yerr=[s], color=estil.TINTA, marker="s", ms=6,
                capsize=3, lw=1.2, ls="none",
                label=f"λ* a D* = {_coma(d['D_estrella_graus'], '.0f')}°: {_coma(d['lambda'])} ± {_coma(s)} s⁻¹")
    ax.set_ylim(0, max(d["lambda_D"]) * 1.35)
    ax.set_xlabel("distància de separació D (graus, en angles)")
    ax.set_ylabel("ln(D/d₀) / T(D)  (s⁻¹)")
    ax.set_title("Ritme de separació del primer segon, preses reals de 150°")
    ax.legend(loc="lower right")
    _peu(fig, f"{d['parelles']} parelles de tirs alineats a l'amollada del vídeo; ± estadística i alineació en "
              "quadratura. És l'exponent de temps finit del primer segon; amb aquest rang de D no prova la "
              "forma exponencial. Només preses reals." + (TEXT_CORREGIT if corregit else ""))
    peus.desa(fig, "lyapunov_real_llindar_150" + ("_corregit" if corregit else ""), directori)


def dibuixa_z(d, directori):
    import matplotlib.pyplot as plt
    from pendol import estil
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    files = (("real", "pèndol real\n(10 tirs de 150°)"), ("model", "model, des de les\nmateixes condicions"),
             ("D4b", "diagnòstic D4b\n(posterior)"))
    rng = np.random.default_rng(0)     # nomes la dispersio vertical dels punts, perque no es tapin
    for y, (k, _) in zip((2, 1, 0), files):
        if k == "real":
            ax.errorbar([d["real"]["lambda"]], [y], xerr=[d["real"]["sigma"]], color=estil.REAL, marker="o", ms=8,
                        capsize=4, lw=1.6, ls="none")
            ax.text(d["real"]["lambda"], y + 0.22, f"{_coma(d['real']['lambda'])} ± {_coma(d['real']['sigma'])}",
                    ha="center", va="bottom", fontsize=9, color=estil.TINTA)
            continue
        v = d[k]
        buit = k == "D4b"
        ax.scatter(v["realitzacions"], y + rng.uniform(-0.12, 0.12, len(v["realitzacions"])), s=22,
                   facecolors="none" if buit else estil.FEIX, edgecolors=estil.SIMULACIO, lw=0.9, zorder=2)
        ax.errorbar([v["lambda"]], [y], xerr=[v["dispersio"]], color=estil.SIMULACIO, marker="D", ms=7,
                    mfc="none" if buit else estil.SIMULACIO, capsize=4, lw=1.6, ls="none", zorder=3)
        ax.text(v["lambda"], y + 0.22, f"{_coma(v['lambda'])} ± {_coma(v['dispersio'])}", ha="center", va="bottom",
                fontsize=9, color=estil.TINTA)
    ax.set_yticks([2, 1, 0], [f[1] for f in files])
    ax.set_ylim(-0.5, 2.6)
    ax.set_xlabel("λ* dels 150° (s⁻¹): el ritme de separació del primer segon")
    ax.set_title(f"La Z: el model se separa més de pressa que el pèndol real "
                 f"({_coma(d['diferencia_sigmes']).replace('-', '−')}σ)", loc="left")
    _peu(fig, f"LA Z FIRMADA (27/10): «{d['veredicte']}». El real, ± bootstrap de tirs i alineació; el model, "
              f"{len(d['model']['realitzacions'])} realitzacions (punts) des de les condicions corregides més l'error "
              f"de la creu, amb la mateixa amollada, soroll i mètode ({d['parelles']} parelles), ± la dispersió entre "
              f"realitzacions (opció 2). Passa si la diferència és de menys de {_coma(d['sigmes'], 'g')}σ. "
              "Sense fricció. El D4b es va fer DESPRÉS del resultat i no té veredicte: les condicions de sortida "
              "contretes cap a la mitjana (empirical Bayes) expliquen prop de la meitat de la diferència. "
              "eines/lyapunov_real.py z_r2 i z_diagnostic.")
    peus.desa(fig, "lyapunov_real_z", directori)


def dibuixa_validacio(d, directori):
    import matplotlib.pyplot as plt
    from pendol import estil
    noms = list(d)
    x = np.arange(len(noms))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bn = [d[n]["benettin"] for n in noms]
    ax.scatter(x - 0.18, bn, s=60, facecolors="none", edgecolors=estil.SIMULACIO, lw=1.5, zorder=3,
               label="resposta coneguda (Benettin, 1000 s)")
    ax.scatter(x, [d[n]["ideal"] for n in noms], s=45, color=estil.SIMULACIO, zorder=3,
               label="el mètode, sense soroll")
    med = np.array([d[n]["soroll_mediana"] for n in noms])
    err = [med - [d[n]["soroll_min"] for n in noms], [d[n]["soroll_max"] for n in noms] - med]
    ax.errorbar(x + 0.18, med, yerr=err, color=estil.SIMULACIO, marker="D", ms=5, ls="none", capsize=3,
                lw=1.2, mfc=estil.FEIX, label="el mètode, amb el soroll de les preses (5)")
    ax.scatter(x + 0.36, [d[n]["simple_mediana"] for n in noms], s=45, marker="x", color=estil.TINTA_SUAU,
               zorder=3, label="pèndol simple de control, amb soroll")
    for i, n in enumerate(noms):
        ax.text(x[i], -0.55, "passa" if d[n]["passa"] else "no passa", ha="center", va="top", fontsize=9,
                color=estil.TINTA if d[n]["passa"] else estil.TINTA_SUAU)
    ax.set_xticks(x, [f"E/E_v2 {n}" for n in noms])
    ax.set_ylim(-0.9, max(max(bn), max(d[n]["soroll_max"] for n in noms)) * 1.15)
    ax.axhline(0, color=estil.EIX, lw=0.8)
    ax.set_ylabel("λ (s⁻¹)")
    ax.set_title("Validació del mètode dels veïns (NOMÉS SIMULACIÓ)")
    ax.legend(loc="upper left", fontsize=8)
    _peu(fig, "12 òrbites del model a les energies de cada franja, amb la quantitat de dades de les preses. "
              "Cap presa real. A 0,8–1,1 les òrbites són regulars; a 3,5–6,3 no hi ha prou rang.")
    peus.desa(fig, "lyapunov_real_validacio", directori)


def _empremta(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def fes(quines, directori="figures", corregit=False) -> dict:
    from pendol import estil
    estil.aplica()
    metr = {}
    sx = "_corregit" if corregit else ""
    if "veins" in quines:
        d = dades_veins(corregit=corregit)
        dibuixa_veins(d, directori, corregit)
        metr["veins" + sx] = {k: v for k, v in d.items() if k not in ("t_s", "mitjana", "p16", "p84")}
    if "llindar_150" in quines:
        d = dades_llindar(corregit)
        dibuixa_llindar(d, directori, corregit)
        metr["llindar_150" + sx] = d
    if "validacio" in quines:
        d = dades_validacio()
        dibuixa_validacio(d, directori)
        metr["validacio"] = d
    if "z" in quines:
        d = dades_z()
        dibuixa_z(d, directori)
        metr["z"] = d
    cami = Path(directori) / "lyapunov_real_metriques.json"
    antic = json.loads(cami.read_text(encoding="utf-8")) if cami.exists() else {}
    antic.update(metr)
    cami.write_text(json.dumps(antic, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  metriques: {cami}")
    return metr


# ------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """FL1: les dades dibuixades son reproduibles bit a bit. FL2: la de validacio no obre res de
    dades/net/ (cap presa real), i les reals no obren res de dades/sim/."""
    global _OBERTS
    problemes = []

    def mira(c, que):
        if not c:
            problemes.append(que)
        print(f"   {'ok ' if c else 'NO '} {que}", flush=True)

    sys.addaudithook(_registra)
    print("Autoprova de eines/figures_lyapunov_real.py\n")
    _OBERTS = []
    a = dades_validacio()
    ob_val, _OBERTS = _OBERTS, []
    b = dades_validacio()
    mira(_empremta(a) == _empremta(b), "FL1 validacio: dues passades identiques")
    mira(not any("net" in Path(x).parts for x in ob_val), "FL2 validacio: no obre cap presa real")
    _OBERTS = []
    a = dades_veins(n_boot=50)
    ob_real, _OBERTS = _OBERTS, None
    b = dades_veins(n_boot=50)
    mira(_empremta(a) == _empremta(b), "FL1 veins: dues passades identiques")
    mira(not any("sim" in Path(x).parts for x in ob_real), "FL2 veins: no obre res de dades/sim/")
    mira(_empremta(dades_llindar()) == _empremta(dades_llindar()), "FL1 llindar_150: dues passades identiques")
    # 28/10: les corregides (pla, «28/10 · La feina del repartiment», 1)
    _OBERTS = []
    a = dades_veins(n_boot=50, corregit=True)
    ob_c, _OBERTS = _OBERTS, None
    b = dades_veins(n_boot=50, corregit=True)
    mira(_empremta(a) == _empremta(b), "FL1 veins corregit: dues passades identiques")
    mira(not any("sim" in Path(x).parts for x in ob_c), "FL2 veins corregit: no obre res de dades/sim/")
    mira(_empremta(dades_llindar(True)) == _empremta(dades_llindar(True)), "FL1 llindar_150 corregit: dues passades identiques")
    j = _carrega(DUES_PARTS)
    c = dades_veins(corregit=True)
    r1 = j["resultat1"]["1,7-3,5"]
    mira(abs(c["lambda"] / r1["lambda"] - 1) <= 1e-12 and c["sigma"] == r1["bootstrap"]["sigma"]
         and c["parelles"] == r1["parelles"],
         f"FL3 veins corregit: lambda {c['lambda']!r} contra {r1['lambda']!r}, sigma {c['sigma']!r} contra "
         f"{r1['bootstrap']['sigma']!r}, {c['parelles']} parelles contra {r1['parelles']}")
    ll = dades_llindar(True)
    mira(ll["lambda"] == j["resultat2"]["150"]["lambda"], f"FL3 llindar_150 corregit: la lambda* es la del JSON ({ll['lambda']!r})")
    # 29/10 (pla, «29/10 · Sis feines», F4 i F2), escrits abans del codi
    b = _carrega(REALS)
    c = dades_veins()
    r1 = b["resultat1"]["1,7-3,5"]
    mira(abs(c["lambda"] / r1["lambda"] - 1) <= 1e-12 and c["sigma"] == r1["bootstrap"]["sigma"]
         and c["parelles"] == r1["parelles"] == 912,
         f"FL4 veins sense corregir: lambda {c['lambda']!r} contra {r1['lambda']!r} (24/10), sigma {c['sigma']!r} contra "
         f"{r1['bootstrap']['sigma']!r}, {c['parelles']} parelles contra {r1['parelles']}")
    mira(dades_llindar()["lambda"] == b["resultat2"]["150"]["lambda"],
         f"FL5 llindar_150 sense corregir: la lambda* es la del JSON del 24/10 ({b['resultat2']['150']['lambda']!r})")
    _OBERTS = []
    a = dades_z()
    ob_z, _OBERTS = _OBERTS, None
    z, dg = _carrega(Z_R2), _carrega(Z_DIAGNOSTIC)
    iguals = (a["real"]["lambda"] == z["real"]["lambda"] and a["real"]["sigma"] == z["real"]["sigma"]
              and a["model"]["lambda"] == z["simulat"]["lambda"] and a["model"]["dispersio"] == z["simulat"]["dispersio"]
              and a["model"]["realitzacions"] == z["simulat"]["lambda_realitzacions"]
              and a["D4b"]["realitzacions"] == dg["D4b"]["lambda_realitzacions"] and a["D4b"]["lambda"] == dg["D4b"]["lambda"]
              and a["model"]["realitzacions"] == dg["referencia"]["lambda_realitzacions"])
    mira(iguals, "FZ1 z: cada numero dibuixat es el dels dos JSON de la Z, bit a bit")
    arrel_dades = (ARREL / "dades").resolve()
    dins = {Path(x).resolve() for x in ob_z if Path(x).resolve().is_relative_to(arrel_dades)}
    mira(dins == {Z_R2.resolve(), Z_DIAGNOSTIC.resolve()},
         f"FZ2 z: de dades/ nomes obre els dos JSON de la Z ({sorted(p.name for p in dins)})")
    mira(_empremta(a) == _empremta(dades_z()), "FZ3 z: dues passades identiques")
    print()
    if problemes:
        print("AUTOPROVA FALLADA:\n" + "\n".join(f"  - {q}" for q in problemes))
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("quines", nargs="*", default=[], help="veins, llindar_150, validacio, z o totes")
    ap.add_argument("--autoprova", action="store_true")
    ap.add_argument("--corregit", action="store_true", help="veins i llindar_150 amb les preses corregides (28/10)")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a figures/memoria/ (el peu, a peus.json; 2/11)")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    q = set(args.quines)
    if not q or "totes" in q:
        q = {"veins", "llindar_150", "validacio", "z"}
    if args.corregit:
        q -= {"validacio", "z"}     # la validacio, nomes simulacio; la Z, ja es amb les condicions corregides
    directori = "figures/memoria" if args.memoria else "figures"
    if args.memoria:
        peus.memoria()
    fes(q, directori, corregit=args.corregit)
    peus.escriu(directori)
    return 0


if __name__ == "__main__":
    sys.exit(main())
