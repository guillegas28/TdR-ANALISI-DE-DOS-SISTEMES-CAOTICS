r"""La corba lambda(E) del pendol doble: Benettin sobre la simulacio conservativa.

Per que existeix. «L'exponent de Lyapunov del meu pendol» no es un numero: a
energia baixa el pendol es regular (lambda ~ 0) i a energia alta es caotic, i
a la mateixa energia hi conviuen illes regulars i un mar caotic
(documentacio/llindar-del-caos.html). Aqui es mesura, a cada energia, la lambda
de moltes condicions inicials de la familia que es fa al laboratori -amollar
del repos, (th1, th2, 0, 0)- i se'n treu la mitjana de les caotiques, la
dispersio i la fraccio regular. Es la referencia que despres ha de validar el
metode de parelles (pas 3) i situar els tres grups d'angle.

Pla: documentacio/pla-analisi-caos.md, pas 2. Tot el calcul es de
pendol/caos.py (benettin_pendol, classifica, condicions_de_repos), que te la
seva autoprova (B1-B5, E1-E2). Aquesta eina nomes reparteix la feina, la desa,
la resumeix i la dibuixa.

NOMES SENSE FRICCIO, i no hi ha cap opcio per posar-n'hi. Cada orbita s'integra
1000 s perque la seva lambda s'estabilitzi. Amb friccio aixo no te cap sentit:
el pendol perd energia, s'atura, i aturat ja no hi ha res a mesurar (la lambda
asimptotica d'un pendol que frega es negativa). La lambda amb friccio es de
temps finit durant el transitori i demana una altra eina (vegeu el pla).

LA BARRERA. De les preses reals nomes es llegeix dades/net/preses.json, per
situar l'energia inicial de cada grup a l'eix. Cap CSV, i cap lambda real: posar
punts reals damunt d'aquesta corba es comparar simulacio i realitat, i aixo va
despres de firmar X, Y i Z.

ES POT ATURAR I REPRENDRE. Cada orbita s'afegeix a <sortida>/orbites.jsonl en
acabar. Tornant a cridar l'eina amb el mateix directori, salta les fetes. El
<sortida>/config.json guarda N, T i la resta: amb uns altres, l'eina s'atura,
perque les condicions inicials cauen a llocs diferents de la corba.

Us:
    .venv\\Scripts\\python eines\\lyapunov.py --sortida dades/lyapunov/tanda --figures figures
    .venv\\Scripts\\python eines\\lyapunov.py --energies 0.97,3.3,6.2 --ci 4 --sortida esborranys/_lyapunov_prova
    .venv\\Scripts\\python eines\\lyapunov.py --sortida esborranys/_lyapunov_prova --nomes-resum
    .venv\\Scripts\\python eines\\lyapunov.py --autoprova
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol.caos import (D0, INTERVAL_REESCALAT_S, LAMBDA_REGULAR,  # noqa: E402
                         PAS_INTEGRACIO_S, RAO_DECAIMENT_REGULAR, TRANSITORI_S,
                         benettin_pendol, classifica, condicions_de_repos,
                         energia_relativa, energia_volteig_1, energia_volteig_2, lambda_a)
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import omega_zero  # noqa: E402
from pendol.simulador import ATOL, RTOL  # noqa: E402

import peus  # noqa: E402

INDEX = ARREL / "dades" / "net" / "preses.json"

# La graella per defecte, fixada el 29/09/2026 abans de calcular res: densa al
# genoll (0,8-1,3, on cauen els 45 graus) i amb les energies aproximades dels
# grups de 90 (3,3) i 150 (6,2). En unitats de E_volteig,2.
ENERGIES_PER_DEFECTE = (0.5, 0.8, 0.9, 0.97, 1.03, 1.1, 1.3, 1.7, 2.5, 3.3, 4.5, 6.2)
CI_PER_DEFECTE = 40
TEMPS_S = 1000.0         # nomes sense friccio: vegeu el capdamunt
LLAVOR = 20260929
SUBMOSTREIG_S = 10.0     # cada quant es desa la lambda acumulada, per a figures


# ------------------------------------------------------------ la feina

def clau(rao: float, i: int) -> str:
    """El nom d'una orbita: l'energia (en E/E_v2) i la seva posicio a la corba."""
    return f"{rao:g}|{i}"


def llavor_de(rao: float, i: int) -> int:
    """Pel nom, no per l'ordre: afegir energies no canvia les orbites fetes."""
    return LLAVOR + zlib.crc32(clau(rao, i).encode("utf-8"))


def configuracio(p, n: int, temps: float) -> dict:
    """El que ha de coincidir per poder continuar un directori."""
    return {
        "ci_per_energia": n,
        "temps_s": temps,
        "interval_reescalat_s": INTERVAL_REESCALAT_S,
        "pas_integracio_s": PAS_INTEGRACIO_S,
        "d0": D0,
        "transitori_s": TRANSITORI_S,
        "omega0_rad_s": omega_zero(p),
        "tolerancia": {"rtol": RTOL, "atol": ATOL},
        "llavor": LLAVOR,
        "llavor_per_orbita": "LLAVOR + crc32('<E/E_v2>|<index>')",
        "friccio": None,
        "classificacio": {
            "regular": f"|lambda(T)| < {LAMBDA_REGULAR} i |lambda(T)| <= "
                       f"{RAO_DECAIMENT_REGULAR}*|lambda(T/4)| + 1e-3",
            "caotica": f"no regular i |lambda(T)| >= {LAMBDA_REGULAR} (canviat el 29/09/2026, "
                       "vegeu pendol/caos.py)",
            "convergida_1pc": "nomes diagnostic",
        },
        "parametres": p.com_a_dict(),
    }


def _normal(v):
    """Tal com quedara al JSON, per comparar configuracions sense sorpreses."""
    return json.loads(json.dumps(_json_net(v)))


def prepara_directori(directori: Path, config: dict) -> None:
    """Crea el directori, o comprova que el que hi ha es de la mateixa tanda."""
    directori.mkdir(parents=True, exist_ok=True)
    cami = directori / "config.json"
    if cami.exists():
        vella = json.loads(cami.read_text(encoding="utf-8"))
        nova = _normal(config)
        diferents = sorted(k for k in set(vella) | set(nova) if vella.get(k) != nova.get(k))
        if diferents:
            raise SystemExit(
                f"{directori} es d'una altra tanda: hi canvien {', '.join(diferents)}.\n"
                "Amb una altra N les condicions inicials cauen a llocs diferents de la corba, "
                "i amb una altra T les lambdes no son comparables. Fes servir un altre directori."
            )
    else:
        cami.write_text(json.dumps(_json_net(config), indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def llegeix_fetes(directori: Path) -> dict:
    """Les orbites ja desades. Una linia a mitges (una interrupcio) es descarta."""
    cami = directori / "orbites.jsonl"
    fetes = {}
    if not cami.exists():
        return fetes
    for n, linia in enumerate(cami.read_text(encoding="utf-8").splitlines(), 1):
        if not linia.strip():
            continue
        try:
            r = json.loads(linia)
        except json.JSONDecodeError:
            print(f"   ATENCIO: la linia {n} de {cami.name} esta a mitges i es descarta; "
                  "aquella orbita es tornara a calcular")
            continue
        fetes[clau(r["energia_rao"], r["index"])] = r
    return fetes


def feina_orbita(args) -> dict:
    """Una orbita: Benettin i la classificacio. Va en un proces a part."""
    rao, i, E_J, estat, temps = args
    p = Parametres.des_de_json(ARREL / "parametres.json")
    ara = time.perf_counter()
    res = benettin_pendol(p, np.asarray(estat, dtype=float), temps, llavor=llavor_de(rao, i))
    c = classifica(res)
    punts = np.arange(SUBMOSTREIG_S, temps + 1e-9, SUBMOSTREIG_S)
    punts = punts[punts > TRANSITORI_S]
    return {
        "energia_rao": rao,
        "energia_J": E_J,
        "index": i,
        "angles_graus": [float(np.degrees(estat[0])), float(np.degrees(estat[1]))],
        "llavor": llavor_de(rao, i),
        "mena": c["mena"],
        "lambda_T": c["lambda_T"],
        "lambda_075T": c["lambda_075T"],
        "lambda_025T": c["lambda_025T"],
        "canvi_relatiu_ultim_quart": c["canvi_relatiu_ultim_quart"],
        "convergida_1pc": c["convergida_1pc"],
        "rao_max_per_pas": res["rao_max_per_pas"],
        "deriva_energia_J": res["deriva_energia_J"],
        "deriva_energia_relativa": res["deriva_energia_relativa"],
        "segons_de_calcul": time.perf_counter() - ara,
        "t_acumulada_s": [float(t) for t in punts],
        "lambda_acumulada": [lambda_a(res, float(t)) for t in punts],
    }


def feines_pendents(p, raons, n: int, temps: float, fetes: dict) -> list:
    Ev2 = energia_volteig_2(p)
    feines = []
    for rao in raons:
        E = rao * Ev2
        ci = condicions_de_repos(E, p, n)
        for i in range(n):
            if clau(rao, i) not in fetes:
                feines.append((rao, i, E, ci[i].tolist(), temps))
    return feines


def calcula(directori: Path, raons, n: int, temps: float, processos: int) -> int:
    """Calcula les orbites que falten i les desa una a una. Torna quantes n'ha fet."""
    p = Parametres.des_de_json(ARREL / "parametres.json")
    prepara_directori(directori, configuracio(p, n, temps))
    fetes = llegeix_fetes(directori)
    feines = feines_pendents(p, raons, n, temps, fetes)
    total = len(raons) * n
    print(f"{total} orbites demanades ({len(raons)} energies x {n}), {total - len(feines)} ja "
          f"fetes, {len(feines)} per fer, de {temps:g} s, amb {processos} processos", flush=True)
    if not feines:
        return 0
    t0 = time.perf_counter()
    cami = directori / "orbites.jsonl"
    with multiprocessing.get_context("spawn").Pool(processos) as pool, \
            cami.open("a", encoding="utf-8") as f:
        for k, r in enumerate(pool.imap_unordered(feina_orbita, feines), 1):
            f.write(json.dumps(_json_net(r), ensure_ascii=False) + "\n")
            f.flush()
            print(f"   {k:>4}/{len(feines)}  E/Ev2 {r['energia_rao']:<5g} ci {r['index']:<3} "
                  f"{r['mena']:<14} lambda {r['lambda_T']:.4f}   "
                  f"{time.perf_counter() - t0:6.0f} s", flush=True)
    return len(feines)


# ------------------------------------------------------------- el resum

def resum_per_energia(registres) -> list:
    """Per energia: quantes de cada mena, i la lambda de les caotiques i de totes."""
    per_rao: dict = {}
    for r in registres:
        per_rao.setdefault(r["energia_rao"], []).append(r)
    fora = []
    for rao in sorted(per_rao):
        rs = sorted(per_rao[rao], key=lambda r: r["index"])
        menes = [r["mena"] for r in rs]
        caot = np.array([r["lambda_T"] for r in rs if r["mena"] == "caotica"])
        totes = np.array([r["lambda_T"] for r in rs])
        fila = {
            "energia_rao": rao,
            "energia_J": rs[0]["energia_J"],
            "n": len(rs),
            "n_regular": menes.count("regular"),
            "n_caotica": menes.count("caotica"),
            "n_sense_decidir": menes.count("sense_decidir"),
            "fraccio_regular": menes.count("regular") / len(rs),
            "caotiques": None,
            "totes": {"mitjana": float(totes.mean()),
                      "desviacio": float(totes.std(ddof=1)) if len(totes) > 1 else None},
            "diagnostic": {
                "rao_max_per_pas": max(r["rao_max_per_pas"] for r in rs),
                "deriva_energia_relativa_max": max(r["deriva_energia_relativa"] for r in rs),
                "caotiques_convergides_1pc": sum(1 for r in rs if r["mena"] == "caotica"
                                                 and r["convergida_1pc"]),
            },
        }
        if len(caot):
            desv = float(caot.std(ddof=1)) if len(caot) > 1 else None
            fila["caotiques"] = {
                "mitjana": float(caot.mean()),
                "error_de_la_mitjana": desv / math.sqrt(len(caot)) if desv is not None else None,
                "desviacio": desv,
                "mediana": float(np.median(caot)),
                "min": float(caot.min()),
                "max": float(caot.max()),
            }
        fora.append(fila)
    return fora


def energies_dels_grups(cami_index: Path, p) -> dict:
    """L'energia inicial de cada presa, en E/E_v2, agrupada per angle.

    LLEGEIX NOMES L'INDEX: els angles de sortida mesurats (angles_amollada_graus)
    ja hi son. No obre cap CSV ni cap JSON de presa, i les preses marcades fora
    del balanc (la 45_7) no hi entren. Serveix per situar els grups a l'eix de
    les energies, i prou.
    """
    index = json.loads(Path(cami_index).read_text(encoding="utf-8"))["preses"]
    Ev2 = energia_volteig_2(p)
    grups: dict = {}
    for f in index:
        if f.get("fora_del_balanc") or not f.get("angles_amollada_graus"):
            continue
        th1, th2 = np.radians(f["angles_amollada_graus"])
        E = float(energia_relativa(np.array([th1, th2, 0.0, 0.0]), p))
        grups.setdefault(f"{f['grup_graus']:.0f}", []).append((f["nom"], E / Ev2))
    return {
        g: {"n": len(v), "min": min(e for _, e in v), "max": max(e for _, e in v),
            "mediana": float(np.median([e for _, e in v])),
            "per_presa": {nom: e for nom, e in v}}
        for g, v in sorted(grups.items(), key=lambda kv: float(kv[0]))
    }


def _json_net(v):
    """inf i nan -> None, i arrays -> llistes, perque el JSON sigui valid."""
    if isinstance(v, dict):
        return {str(k): _json_net(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_net(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_json_net(x) for x in v.tolist()]
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def resumeix(directori: Path, figures: Path, cami_index: Path = INDEX) -> dict:
    """Del que hi ha desat, el resum i la figura. No calcula cap orbita."""
    p = Parametres.des_de_json(ARREL / "parametres.json")
    cami_config = directori / "config.json"
    if not cami_config.exists():
        raise SystemExit(f"{directori} no te config.json: aqui no s'ha calculat res")
    config = json.loads(cami_config.read_text(encoding="utf-8"))
    registres = list(llegeix_fetes(directori).values())
    if not registres:
        raise SystemExit(f"{directori} encara no te cap orbita desada")
    files = resum_per_energia(registres)
    w0 = config["omega0_rad_s"]
    for fila in files:
        if fila["caotiques"]:
            fila["caotiques"]["mitjana_sobre_omega0"] = fila["caotiques"]["mitjana"] / w0
    grups = energies_dels_grups(cami_index, p) if Path(cami_index).exists() else None
    sortida = {
        "eina": "eines/lyapunov.py",
        "pla": "documentacio/pla-analisi-caos.md",
        "que_es": "lambda(E) de la simulacio CONSERVATIVA, sense friccio; familia de repos "
                  "(th1, th2, 0, 0). Cap dada real, llevat de l'energia inicial dels grups",
        "energia_volteig_2_J": energia_volteig_2(p),
        "energia_volteig_1_J": energia_volteig_1(p),
        "config": config,
        "orbites": len(registres),
        "per_energia": files,
        "grups_energia_inicial_rao": grups,
    }
    figures.mkdir(parents=True, exist_ok=True)
    (figures / "lambda_E.json").write_text(
        json.dumps(_json_net(sortida), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    imprimeix(files, grups)
    dibuixa(registres, files, grups, config, energia_volteig_1(p) / energia_volteig_2(p),
            figures)
    return sortida


def imprimeix(files, grups) -> None:
    print(f"\n{'E/E_v2':>7} {'n':>3} {'reg':>4} {'caot':>5} {'s.d.':>5} {'f.reg':>6}   "
          f"{'lambda caotiques (s^-1)':<28} {'d/d0 max':>8} {'deriva E':>9}")
    for f in files:
        c = f["caotiques"]
        if c is None:
            txt = "-"
        elif c["error_de_la_mitjana"] is None:
            txt = f"{c['mitjana']:.3f} (una sola)"
        else:
            txt = f"{c['mitjana']:.3f} +- {c['error_de_la_mitjana']:.3f} (desv. {c['desviacio']:.3f})"
        d = f["diagnostic"]
        print(f"{f['energia_rao']:>7g} {f['n']:>3} {f['n_regular']:>4} {f['n_caotica']:>5} "
              f"{f['n_sense_decidir']:>5} {f['fraccio_regular']:>6.2f}   {txt:<28} "
              f"{d['rao_max_per_pas']:>8.2f} {d['deriva_energia_relativa_max']:>9.1e}")
    if grups:
        print("\nEnergia inicial dels grups (E/E_v2), nomes de preses.json:")
        for g, v in grups.items():
            print(f"   {g:>4} graus: {v['min']:.2f} a {v['max']:.2f}, mediana {v['mediana']:.2f} "
                  f"({v['n']} preses)")


# ------------------------------------------------------------- la figura

def dibuixa(registres, files, grups, config, Ev1_sobre_Ev2: float, figures: Path) -> None:
    """lambda contra E/E_v2, i la fraccio regular. Un sol color: es tot simulacio.

    UN SOL EIX VERTICAL per panell, a posta: lambda/omega0 es la mateixa dada
    dividida per una constant, i un segon eix faria llegir dues mesures on
    n'hi ha una. omega0 va al rotul i el valor adimensional, al JSON.
    """
    import matplotlib.pyplot as plt

    from pendol import estil

    estil.aplica()
    w0 = config["omega0_rad_s"]
    fig, (dalt, baix) = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1.3]})

    # Les franges dels grups: l'energia inicial mesurada de cada un, en tinta
    # neutra. Sense cap lambda real: nomes on comencen.
    if grups:
        for g, v in grups.items():
            for eix in (dalt, baix):
                eix.axvspan(v["min"], v["max"], color=estil.GRAELLA, alpha=0.9, lw=0, zorder=0)
            # Per sobre del panell: a dins, el rotul tapava les orbites de dalt.
            dalt.annotate(f"{g}°", xy=(0.5 * (v["min"] + v["max"]), 1.0),
                          xycoords=("data", "axes fraction"), xytext=(0, 3),
                          textcoords="offset points", ha="center", va="bottom",
                          color=estil.TINTA_2, fontsize=9)

    for x, nom in ((1.0, r"$E_{volteig,2}$"), (Ev1_sobre_Ev2, r"$E_{volteig,1}$")):
        for eix in (dalt, baix):
            eix.axvline(x, color=estil.TINTA_SUAU, lw=0.9, ls=":", zorder=1)
        baix.annotate(nom, xy=(x, 0.0), xycoords=("data", "axes fraction"),
                      xytext=(3, 3), textcoords="offset points", color=estil.TINTA_2,
                      fontsize=8)

    # Les orbites una a una: caotiques plenes, regulars buides, sense decidir
    # amb una creu. La forma diu la mena; el color es sempre el de la simulacio.
    vora = dict(edgecolors=estil.SIMULACIO, linewidths=0.9)
    formes = {"caotica": dict(marker="o", facecolors=estil.SIMULACIO, alpha=0.35, **vora),
              "regular": dict(marker="o", facecolors="none", alpha=0.8, **vora),
              # La creu no te farciment ni vora: nomes color. Amb vora, matplotlib avisa.
              "sense_decidir": dict(marker="x", color=estil.SIMULACIO, linewidths=0.9,
                                    alpha=0.8)}
    etiquetes = {"caotica": "òrbita caòtica", "regular": "òrbita regular",
                 "sense_decidir": "sense decidir"}
    for mena, estil_m in formes.items():
        rs = [r for r in registres if r["mena"] == mena]
        if not rs:
            continue
        dalt.scatter([r["energia_rao"] for r in rs], [r["lambda_T"] for r in rs], s=22,
                     zorder=2, label=etiquetes[mena], **estil_m)

    amb = [f for f in files if f["caotiques"]]
    if amb:
        x = [f["energia_rao"] for f in amb]
        y = [f["caotiques"]["mitjana"] for f in amb]
        e = [f["caotiques"]["error_de_la_mitjana"] or 0.0 for f in amb]
        dalt.errorbar(x, y, yerr=e, color=estil.SIMULACIO, lw=1.8, capsize=3, zorder=3,
                      label="mitjana de les caòtiques ± error")

    dalt.set_ylabel(r"$\lambda$  (s$^{-1}$)" + f"\n" + rf"($\lambda/\omega_0$ amb $\omega_0$ = {w0:.2f} rad/s: al JSON)".replace(".", ","),
                    fontsize=9)
    dalt.set_ylim(bottom=-0.1)
    dalt.set_title("Exponent de Lyapunov contra l'energia — simulació conservativa, sense "
                   "fricció", loc="left", pad=22)

    baix.plot([f["energia_rao"] for f in files], [f["fraccio_regular"] for f in files],
              color=estil.SIMULACIO, marker="o", ms=5, lw=1.7)
    baix.set_ylim(-0.05, 1.05)
    baix.set_ylabel("fracció regular")
    baix.set_xlabel(r"energia des del repòs,  $E / E_{volteig,2}$")

    n = config["ci_per_energia"]
    peu = (f"{n} condicions inicials per energia, amollades del repòs; Benettin durant "
           f"{config['temps_s']:g} s, tolerància estricta. Franges grises: energia inicial "
           "mesurada de cada grup (només preses.json).")
    fig.tight_layout()
    peus.peu(fig, 0.01, -0.02, peu, fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    handles, labels = dalt.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06),
               ncols=len(labels))
    peus.desa(fig, "lambda_E", figures)


# --------------------------------------------------------------- l'autoprova

def autoprova() -> int:
    """L1-L5 de documentacio/pla-analisi-caos.md: l'eina no perd ni barreja res.

    La fisica ja la proven B1-B5 i E1-E2 de pendol/caos.py. Aqui es prova la
    llibreteria: reprendre, repartir en processos, resumir i llegir els grups.
    """
    import builtins
    import io
    import shutil

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    base = ARREL / "esborranys" / "_autoprova_lyapunov"
    for sub in ("una", "dues"):
        if (base / sub).exists():
            shutil.rmtree(base / sub)
    raons, n, T = (0.97, 6.2), 2, 20.0
    lambdes = ("lambda_T", "lambda_075T", "lambda_025T", "lambda_acumulada", "mena")

    print("Autoprova de eines/lyapunov.py (criteris fixats el 29/09/2026)\n")
    print("L1. Reprendre: dues crides han de donar el mateix, bit a bit, que una")
    calcula(base / "una", raons, n, T, processos=2)
    calcula(base / "dues", raons[:1], n, T, processos=2)
    calcula(base / "dues", raons[1:], n, T, processos=2)
    una, dues = llegeix_fetes(base / "una"), llegeix_fetes(base / "dues")
    iguals = set(una) == set(dues) and all(una[k][c] == dues[k][c] for k in una for c in lambdes)
    mira(iguals and len(una) == len(raons) * n,
         f"{len(una)} orbites, les mateixes lambdes en una crida i en dues")
    fetes = calcula(base / "dues", raons, n, T, processos=2)
    mira(fetes == 0, "una tercera crida no calcula res")

    print("\nL2. Un proces del pool dona el mateix que el proces principal")
    p = Parametres.des_de_json(ARREL / "parametres.json")
    feina = feines_pendents(p, raons[1:], n, T, {})[0]
    # Es compara tal com queda DESAT, que es el que l'eina fa servir. Fins al
    # 29/09 es comparava el diccionari fresc amb el llegit del fitxer, i un
    # camp nan (la lambda a T/4, que amb T = 20 s cau al final del transitori)
    # al fitxer es None: None != nan, i nan != nan. El diagnostic
    # (esborranys/_diag_lyapunov_l2.py) va ensenyar que el calcul era identic
    # bit a bit; el criteri no ha canviat.
    aqui = _normal(feina_orbita(feina))
    mira(all(aqui[c] == una[clau(feina[0], feina[1])][c] for c in lambdes),
         f"l'orbita {clau(feina[0], feina[1])}: lambda {aqui['lambda_T']:.6f} als dos llocs, bit a bit")

    print("\nL3. El resum, sobre registres fabricats")
    fals = [{"energia_rao": 1.0, "energia_J": 0.05, "index": i, "mena": m, "lambda_T": l,
             "rao_max_per_pas": 2.0, "deriva_energia_relativa": 1e-12, "convergida_1pc": True}
            for i, (m, l) in enumerate([("regular", 0.01), ("regular", 0.02), ("caotica", 2.0),
                                        ("caotica", 3.0), ("sense_decidir", 0.03)])]
    f = resum_per_energia(fals)[0]
    c = f["caotiques"]
    esperat = (f["n"] == 5 and f["n_regular"] == 2 and f["n_caotica"] == 2
               and f["n_sense_decidir"] == 1 and math.isclose(f["fraccio_regular"], 0.4)
               and math.isclose(c["mitjana"], 2.5) and math.isclose(c["desviacio"], math.sqrt(0.5))
               and math.isclose(c["error_de_la_mitjana"], 0.5)
               and math.isclose(f["totes"]["mitjana"], 5.06 / 5))
    mira(esperat, "5 orbites (2 regulars, 2 caotiques de 2 i 3, 1 sense decidir): fraccio "
                  "0,4, mitjana 2,5 +- 0,5, mitjana de totes 1,012")

    print("\nL4. Els grups es llegeixen de l'index i de res mes")
    index_fals = base / "preses_fals.json"
    index_fals.write_text(json.dumps({"preses": [
        {"nom": "g45", "grup_graus": 45.0, "angles_amollada_graus": [45.0, 45.0],
         "fitxer_trk": "no/existeix/g45.trk"},
        {"nom": "g90", "grup_graus": 90.0, "angles_amollada_graus": [90.0, 90.0],
         "fitxer_trk": "no/existeix/g90.trk"},
        {"nom": "fora", "grup_graus": 90.0, "angles_amollada_graus": [170.0, 170.0],
         "fora_del_balanc": True},
    ]}), encoding="utf-8")
    oberts = []
    obre_b, obre_io = builtins.open, io.open

    def espia(fitxer, *a, **k):
        oberts.append(str(fitxer))
        return obre_b(fitxer, *a, **k)

    builtins.open, io.open = espia, espia
    try:
        g = energies_dels_grups(index_fals, p)
    finally:
        builtins.open, io.open = obre_b, obre_io
    # A 90/90 tota l'energia es g(A + B) (un calcul independent del de caos.py),
    # i en unitats de E_v2 = 2gB queda (A + B) / (2B).
    esperat_90 = (p.A + p.B) / (2 * p.B)
    mira(set(g) == {"45", "90"} and g["90"]["n"] == 1
         and math.isclose(g["90"]["max"], esperat_90, rel_tol=1e-12),
         f"dos grups, sense la presa fora del balanc, i 90/90 a {esperat_90:.4f} E_v2")
    mira(all(Path(o).resolve() == index_fals.resolve() for o in oberts) and oberts,
         f"nomes s'ha obert l'index ({len(oberts)} fitxer)")

    print("\nL5. Continuar amb una altra N o una altra T s'atura")
    for que, cfg in (("N = 3", configuracio(p, 3, T)), ("T = 30 s", configuracio(p, n, 30.0))):
        try:
            prepara_directori(base / "una", cfg)
            aturat = False
        except SystemExit:
            aturat = True
        mira(aturat, f"amb {que} s'atura")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> l'eina repren sense perdre ni repetir res, dona el mateix en paral.lel que")
    print("     sola, resumeix be i dels grups nomes llegeix l'index.")
    print("     La fisica (Benettin, energia, condicions inicials) la prova pendol/caos.py.")
    return 0


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sortida", help="directori de la tanda (orbites.jsonl i config.json)")
    ap.add_argument("--figures", help="on va la figura i el JSON de metriques "
                                      "(per defecte, el directori de la tanda)")
    ap.add_argument("--energies", help="E/E_volteig,2 separades per comes (per defecte, les "
                    f"{len(ENERGIES_PER_DEFECTE)} de la graella)")
    ap.add_argument("--ci", type=int, default=CI_PER_DEFECTE,
                    help=f"condicions inicials per energia (per defecte {CI_PER_DEFECTE})")
    ap.add_argument("--temps", type=float, default=TEMPS_S,
                    help=f"segons de Benettin per orbita (per defecte {TEMPS_S:g})")
    ap.add_argument("--processos", type=int, default=6, help="com a molt 6")
    ap.add_argument("--nomes-resum", action="store_true",
                    help="no calcula res: resumeix i dibuixa el que ja hi ha")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a <figures>/memoria/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.sortida:
        ap.error("falta --sortida")
    if not 1 <= args.processos <= 6:
        ap.error("--processos ha d'anar d'1 a 6: l'ordinador te 12 fils i se'n fan servir la meitat")
    directori = Path(args.sortida)
    figures = Path(args.figures) if args.figures else directori
    if args.memoria:
        if not args.figures:
            ap.error("--memoria va amb --figures")
        peus.memoria()
        figures = figures / "memoria"

    if not args.nomes_resum:
        raons = (tuple(float(x) for x in args.energies.split(","))
                 if args.energies else ENERGIES_PER_DEFECTE)
        calcula(directori, raons, args.ci, args.temps, args.processos)
    resumeix(directori, figures)
    if args.figures:          # el peus.json, mai a la carpeta de la tanda (dades/lyapunov/)
        peus.escriu(figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
