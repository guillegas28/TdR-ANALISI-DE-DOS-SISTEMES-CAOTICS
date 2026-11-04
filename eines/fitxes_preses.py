r"""Una fitxa per presa: com evoluciona cada tir real. NOMES PRESES REALS, cap simulacio.

Pla: documentacio/pla-analisi-caos.md, «28/10 · Una fitxa per presa» (criteris FF1-FF7, escrits abans del codi). Les
figures de grup (eines/figures_reals.py) no es toquen: aqui hi ha les mateixes dades, una presa per figura, perque
amb totes juntes nomes es poden comparar i no se'n veu l'evolucio (en Guillem, 28/10).

  fila 1     theta1(t) en graus; theta2(t) en voltes, desembolicat; E/E_v2(t), amb els punts del Poincare al seu
             instant
  fila 2     el Poincare de la presa (theta2 = 0 amb omega2 > 0; el tall i el filtre de figures_reals.calcula_poincare),
             pintat per E/E_v2 amb la mateixa escala que poincare_real_energia_corregit; i, a les 10 preses de 150 graus
             de la lambda*, com se separa la presa de les altres: les seves parelles del R2, pel mateix cami que
             lyapunov_real.py dues_parts. NO es cap lambda de la presa: un tir sol no en te
  files 3-4  l'espai de fases del brac 1 i del brac 2 a quatre trams, amb la mateixa escala a cada fila

Els angles, sempre corregits (comparar.corregeix_presa); la 45_7 fora, com a les figures de grup. El panell de la
lambda* es darrere del pany d'aprovat.json, com la lambda* mateixa.

    .venv\Scripts\python.exe eines\fitxes_preses.py
    .venv\Scripts\python.exe eines\fitxes_preses.py --preses 45_1 150_3
    .venv\Scripts\python.exe eines\fitxes_preses.py --autoprova
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

import figures_reals as fr  # noqa: E402
import lyapunov_real as lr  # noqa: E402
import peus  # noqa: E402
from pendol import caos  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import voltes  # noqa: E402

DIRECTORI = ARREL / "figures" / "per_presa"
TRAMS_S = ((0.0, 3.0), (3.0, 10.0), (10.0, 30.0), (30.0, math.inf))   # els dos primers, els de figures_reals
DES_DE_S = -0.2          # la fila del temps comenca com els espaguetis
REPOS_S = (-0.4, -0.02)  # el repos d'abans d'amollar: la finestra de lyapunov_real.preses_corregides


# ------------------------------------------------------------- les dades

def dades_lambda() -> dict:
    """Les parelles del R2 dels 150 graus amb la correccio de dues_parts. Es el cos de lyapunov_real.r2_de_preses
    sense el bootstrap: la FF6 comprova que en torna a sortir la lambda* del JSON del 27/10 bit a bit."""
    aprovat = json.loads(lr.APROVAT.read_text(encoding="utf-8")) if lr.APROVAT.exists() else {}
    if "150" not in aprovat.get("resultat2", []):
        raise SystemExit("Pany tancat: la lambda* dels 150 graus no esta aprovada (dades/divergencia/"
                         "lyapunov_real_aprovat.json). No s'obre cap presa.")
    R = json.loads(lr.PAS0B_JSON.read_text(encoding="utf-8"))["resultat2"]["150"]
    ref = json.loads(lr.DUES_PARTS_2710_JSON.read_text(encoding="utf-8"))["resultat2"]["150"]
    pc, _, _ = lr.preses_corregides()
    noms, Dstar = R["preses"], R["D_estrella_rad"]
    N = len(noms)
    tc = np.arange(0.0, lr.T_MAX_S + 0.5 * lr.DT, lr.DT)
    rep = [pc[n]["repos_rad"] for n in noms]
    par_all = [(a, b) for a in range(N) for b in range(a + 1, N)]
    d0 = np.array([float(lr.dist_ang(rep[a], rep[b])) for a, b in par_all])
    sel = [k for k in range(len(par_all)) if d0[k] <= Dstar / math.e]
    par = [par_all[k] for k in sel]
    X = [lr.a_la_graella(pc[n]["t"], pc[n]["estats"], 0.0, tc) for n in noms]
    d = np.array([lr.dist_ang(X[a], X[b]) for a, b in par])
    T = lr.creuaments(tc, d, [Dstar])[:, 0]
    return {"preses": list(noms), "fora": list(R["fora"]), "D_estrella_rad": Dstar, "t": tc,
            "parelles": [(noms[a], noms[b]) for a, b in par], "d0": d0[sel], "d": d, "T": T,
            "lambda_grup": ref["lambda"], "parelles_grup": ref["parelles"]}


def lambda_de(d0, T, Dstar) -> float:
    """La lambda* d'unes parelles, com avalua_llindars: la mediana de ln(D*/d0)/T (les censurades, amb la fita)."""
    lam, fita, cens = lr.lambdes_parelles(np.asarray(T, dtype=float)[:, None], d0, [Dstar], lr.T_MAX_S)
    return float(lr.agrega(lam, fita, cens)[0][0])


def calcula_fitxa(p, p_mod, Ev2, lam) -> dict:
    t, X = p["t"], p["estats"]
    f1 = t >= DES_DE_S
    mou = t >= 0.0
    rep = (t >= REPOS_S[0]) & (t <= REPOS_S[1])
    E = np.atleast_1d(caos.energia_relativa(X[:, f1], p_mod)) / Ev2
    sobre = np.nonzero(E[t[f1] >= 0.0] >= 1.0)[0]
    trams = []
    for a, b in TRAMS_S:
        s = (t >= a) & (t <= b)
        trams.append({"de_s": a, "fins_s": b, "fotogrames": np.nonzero(s)[0],
                      "th1": fr._plega_graus(X[0][s]), "w1": X[2][s], "th2": fr._plega_graus(X[1][s]), "w2": X[3][s]})
    f = {"nom": p["nom"], "grup": p["grup"], "durada_s": float(t[-1]), "incertesa_zero_s": p["incertesa_zero_s"],
         "t": t[f1], "th1_graus": np.degrees(X[0][f1]), "th2_voltes": X[1][f1] / (2 * math.pi), "E": E,
         "poincare": fr.calcula_poincare([p], p_mod)[p["grup"]], "trams": trams,
         "E_repos": float(np.mean(caos.energia_relativa(X[:, rep], p_mod)) / Ev2) if rep.any() else None,
         "ultim_E_sobre_1_s": float(t[f1][t[f1] >= 0.0][sobre[-1]]) if len(sobre) else None,
         "voltes_theta2": float(voltes(X[1][mou])), "lambda": None}
    if p["nom"] in lam["preses"]:
        f["lambda"] = [{"altra": b if a == p["nom"] else a, "parella": [a, b], "d0": float(lam["d0"][k]),
                        "d": lam["d"][k], "T": float(lam["T"][k])}
                       for k, (a, b) in enumerate(lam["parelles"]) if p["nom"] in (a, b)]
    return f


def calcula(noms=None) -> dict:
    p_mod = Parametres.des_de_json(ARREL / "parametres.json")
    Ev2 = caos.energia_volteig_2(p_mod)
    preses = fr.preses_reals(corregit=True)
    poi = fr.calcula_poincare(preses, p_mod)
    lam = dades_lambda()
    fitxes = {p["nom"]: calcula_fitxa(p, p_mod, Ev2, lam) for p in preses if noms is None or p["nom"] in noms}
    return {"preses": preses, "poincare": poi, "escala": fr.escala_energia(poi), "lambda": lam, "fitxes": fitxes}


# ------------------------------------------------------------- les figures

def _text_lambda_absent(f, lam) -> str:
    if f["grup"] != "150":
        return (f"Als {f['grup']}° la λ* no es mesura: les parelles no hi tenen prou rang (el pas 0 i la validació del "
                "17/10). La λ dels veïns barreja tirs de 90° i 150°, i no és de cap presa.")
    if f["nom"] in lam["fora"]:
        return (f"La {f['nom']} és fora de la λ*: el seu instant d'amollada té una incertesa de més de "
                f"{lr.SIGMA_AMOLLADA_MAX_S * 1000:.0f} ms (criteri del 17/10).")
    return f"Cap parella de la {f['nom']} no entra a la λ* (d₀ > D*/e)."


def dibuixa_lambda(ax, f, lam):
    from pendol import estil

    if not f["lambda"]:
        ax.set_axis_off()
        ax.text(0.5, 0.5, textwrap.fill(_text_lambda_absent(f, lam), 60), transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=estil.TINTA_SUAU)
        ax.set_title("Com se separa dels altres tirs", loc="left")
        return
    t, Dg = lam["t"], math.degrees(lam["D_estrella_rad"])
    for q in f["lambda"]:
        ax.semilogy(t, np.degrees(q["d"]), color=estil.REAL, lw=0.9, alpha=0.6)
        if np.isfinite(q["T"]):
            ax.plot(q["T"], Dg, "o", ms=4, color=estil.TINTA, zorder=3)
    d0m = math.degrees(float(np.median([q["d0"] for q in f["lambda"]])))
    tt = np.linspace(0.0, math.log(Dg / d0m) / lam["lambda_grup"], 50)
    ax.semilogy(tt, d0m * np.exp(lam["lambda_grup"] * tt), color=estil.TINTA, lw=1.2, ls=":",
                label=f"el ritme del grup, λ* = {fr._coma(lam['lambda_grup'])} s⁻¹ (des de la mediana de les d₀)")
    ax.axhline(Dg, color=estil.TINTA_SUAU, lw=0.9, ls="--", label=f"D* = {Dg:.0f}°")
    cens = sum(not np.isfinite(q["T"]) for q in f["lambda"])
    ax.set_xlim(0.0, lr.T_MAX_S)
    ax.set_xlabel("t des de l'amollada (s)")
    ax.set_ylabel("d entre els dos tirs (graus, en angles)")
    ax.set_title(f"Com se separa dels altres tirs de 150° ({len(f['lambda'])} parelles)", loc="left", pad=28)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.0, 1.01, "cada línia, una parella amb aquesta presa; el punt, quan arriba a D*"
            + (f"; {cens} no hi arriben en {lr.T_MAX_S:g} s" if cens else "")
            + f".\nNo és cap λ d'aquesta presa: la λ* és la mediana de les {lam['parelles_grup']} parelles del grup.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=estil.TINTA_2)


def dibuixa_fitxa(f, preses, escala, lam, directori):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from pendol import estil

    estil.aplica()
    mapa, norm = estil.mapa_real(), Normalize(*escala)
    # Una graella per fila, cadascuna amb el seu espai entre panells (el constrained layout s'hi ensorrava).
    fig = plt.figure(figsize=(16.0, 18.0))
    files = fig.add_gridspec(4, 1, height_ratios=(1.0, 1.25, 1.0, 1.0), hspace=0.42,
                             left=0.06, right=0.98, top=0.95, bottom=0.11)
    g0 = files[0].subgridspec(1, 3, wspace=0.28)
    g1 = files[1].subgridspec(1, 2, wspace=0.22)
    a_th1, a_th2, a_E = (fig.add_subplot(g0[0, k]) for k in range(3))
    a_poi, a_lam = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])

    for ax, y, nom in ((a_th1, f["th1_graus"], "θ1 (graus)"), (a_th2, f["th2_voltes"], "θ2 (voltes, desembolicat)")):
        ax.plot(f["t"], y, color=estil.REAL, lw=0.6)
        ax.set_ylabel(nom)
    a_E.plot(f["t"], f["E"], color=estil.REAL, lw=0.5, alpha=0.7)
    pts = sorted(f["poincare"], key=lambda q: q[4])
    if pts:
        a_E.scatter([q[1] for q in pts], [q[4] for q in pts], c=[q[4] for q in pts], cmap=mapa, norm=norm, s=12,
                    linewidths=0.4, edgecolors=estil.SUPERFICIE, zorder=3, label="els punts del Poincaré")
    a_E.axhline(1.0, color=estil.TINTA_SUAU, lw=0.9, ls="--", label="E_v2: per sota, el braç 2 ja no pot voltar")
    a_E.set_ylabel("E/E_v2")
    a_E.legend(loc="upper right", fontsize=8)
    for ax, tit in ((a_th1, "θ1 al llarg del tir"), (a_th2, "θ2 al llarg del tir"), (a_E, "L'energia")):
        ax.set_title(tit, loc="left")
        ax.set_xlabel("t des de l'amollada (s)")
        ax.axvline(0.0, color=estil.EIX, lw=0.8, zorder=0)

    if pts:
        sc = a_poi.scatter([q[2] for q in pts], [q[3] for q in pts], c=[q[4] for q in pts], cmap=mapa, norm=norm,
                           s=14, linewidths=0)
        fig.colorbar(sc, ax=a_poi, shrink=0.9, pad=0.02, label="E/E_v2")
    a_poi.set_xlim(-180, 180)
    a_poi.set_xticks(range(-180, 181, 90))
    a_poi.set_xlabel("θ1 plegat (graus)")
    a_poi.set_ylabel("ω1 (rad/s)")
    a_poi.set_title(f"Secció de Poincaré ({len(pts)} punts)", loc="left", pad=28)
    a_poi.text(0.0, 1.01, "el braç 2 passa per la vertical de sota girant en sentit positiu (θ2 = 0, ω2 > 0);\n"
               "el color, l'energia de cada punt, amb la mateixa escala a les 35 fitxes",
               transform=a_poi.transAxes, ha="left", va="bottom", fontsize=7.5, color=estil.TINTA_2)
    dibuixa_lambda(a_lam, f, lam)

    for fila, (th, w, nom) in enumerate((("th1", "w1", "1"), ("th2", "w2", "2"))):
        gf = files[2 + fila].subgridspec(1, 4, wspace=0.14)
        eixos = [fig.add_subplot(gf[0, k]) for k in range(4)]
        for ax in eixos[1:]:
            ax.sharey(eixos[0])
            ax.tick_params(labelleft=False)
        for ax, tr in zip(eixos, f["trams"]):
            x, y = fr._amb_talls(tr[th], tr[w])
            ax.plot(x, y, color=estil.REAL, lw=0.5, alpha=0.6, rasterized=True)
            fins = "al final" if math.isinf(tr["fins_s"]) else f"a {tr['fins_s']:g} s"
            ax.set_title(f"Braç {nom} · de {tr['de_s']:g} s {fins}", loc="left", fontsize=10)
            ax.set_xlim(-180, 180)
            ax.set_xticks(range(-180, 181, 90))
            ax.set_xlabel(f"θ{nom} plegat (graus)")
        eixos[0].set_ylabel(f"ω{nom} (rad/s)")

    Er = "—" if f["E_repos"] is None else fr._coma(f["E_repos"])
    fig.suptitle(f"Presa {f['nom']} · amollada des de {f['grup']}° · {f['durada_s']:.0f} s des de l'amollada · "
                 f"E/E_v2 a l'amollada {Er} · θ2 escombra {fr._coma(f['voltes_theta2'], '.1f')} voltes",
                 x=0.06, y=0.975, ha="left", fontweight="bold")
    peus.peu(fig, 0.06, 0.065, textwrap.fill(
             f"DADES REALS, cap simulació. Una sola presa; les mateixes dades són a les figures de grup "
             f"(figures/fases_reals_{f['grup']}_corregit, poincare_real_corregit, poincare_real_energia_corregit). El "
             f"zero és l'amollada del vídeo (incertesa {f['incertesa_zero_s'] * 1000:.0f} ms). Les dues files de baix són "
             "l'espai de fases partit en trams de temps, amb la mateixa escala a cada fila: s'hi veu com s'encongeix "
             "l'òrbita. Cada panell és l'OMBRA d'un espai de quatre dimensions: que la corba es creui no vol dir que es "
             "creui de debò. L'energia es calcula amb el model sobre l'estat mesurat; les velocitats porten el soroll "
             "de seguiment, i per això la corba de l'energia és sorollosa. " + fr._text_filtre(preses)
             + fr._text_correccio(preses), 230),
             fontsize=8, color=estil.TINTA_2, ha="left", va="top")
    peus.desa(fig, f"fitxa_{f['nom']}_corregit", directori)


# ------------------------------------------------------------- el JSON

_empremta = fr._empremta


def resum(r) -> dict:
    lam = r["lambda"]
    per = {}
    for n, f in r["fitxes"].items():
        per[n] = {"grup": f["grup"], "durada_s": f["durada_s"], "E_sobre_Ev2_a_l_amollada": f["E_repos"],
                  "ultim_instant_amb_E_sobre_Ev2_ge_1_s": f["ultim_E_sobre_1_s"], "voltes_theta2": f["voltes_theta2"],
                  "poincare": [{"t_s": q[1], "theta1_graus": q[2], "omega1": q[3], "E_sobre_Ev2": q[4]}
                               for q in f["poincare"]],
                  "lambda_parelles": None if f["lambda"] is None else
                  [{"altra": q["altra"], "d0_graus": math.degrees(q["d0"]),
                    "T_s": q["T"] if np.isfinite(q["T"]) else None} for q in f["lambda"]],
                  "empremta": _empremta(f)}
    fitxers = (fr.CENTRES, fr.CORRECCIO_CLICS)
    return {"que_es": "una fitxa per presa: NOMES PRESES REALS, cap simulacio",
            "eina": "eines/fitxes_preses.py", "pla": "documentacio/pla-analisi-caos.md («28/10 · Una fitxa per presa»)",
            "fora": "45_7 (fora_del_balanc), com a les figures de grup",
            "trams_s": [[a, None if math.isinf(b) else b] for a, b in TRAMS_S],
            "repos_s": list(REPOS_S), "escala_E_sobre_Ev2": list(r["escala"]),
            "lambda": {"de": "lyapunov_real.py dues_parts (dades/divergencia/lyapunov_real_dues_parts_2026-10-27.json)",
                       "D_estrella_graus": math.degrees(lam["D_estrella_rad"]), "lambda_grup": lam["lambda_grup"],
                       "preses": lam["preses"], "fora": lam["fora"],
                       "avis": "cap lambda per presa: la lambda* es la mediana de les parelles del grup"},
            "correccio": {"funcio": "pendol.comparar.corregeix_presa",
                          "fitxers_sha256": {c.relative_to(ARREL).as_posix(): hashlib.sha256(c.read_bytes()).hexdigest()
                                             for c in fitxers},
                          "grups_amb_clics": fr._grups_amb_clics(), "validada_als_150": False,
                          "definitiva": fr.DEFINITIVA,
                          "espera": None if fr.DEFINITIVA else "que en Guillem les doni per definitives"},
            "preses": per}


# ------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """FF1-FF6 (pla, «28/10 · Una fitxa per presa»), escrits abans del codi. La FF7 es mirar-ne quatre."""
    global _OBERTS
    problemes = []

    def mira(c, que):
        if not c:
            problemes.append(que)
        print(f"   {'ok ' if c else 'NO '} {que}", flush=True)

    print("Autoprova de eines/fitxes_preses.py (FF1-FF6)\n")
    sys.addaudithook(_registra)
    _OBERTS = []
    r = calcula()
    oberts, _OBERTS = _OBERTS, None
    preses, F, lam = r["preses"], r["fitxes"], r["lambda"]

    fases = fr.calcula_fases(preses)
    mal1 = []
    for p in preses:
        f = F[p["nom"]]
        grup = [q for q in r["poincare"][p["grup"]] if q[0] == p["nom"]]
        if f["poincare"] != grup:
            mal1.append(f"{p['nom']} (Poincare)")
        for k, (a, b) in enumerate(TRAMS_S[:2]):
            g = next(x for x in fases[p["grup"]][f"{a:g}-{b:g}"] if x["nom"] == p["nom"])
            if not all(np.array_equal(f["trams"][k][c], g[c]) for c in ("th1", "w1", "th2", "w2")):
                mal1.append(f"{p['nom']} (tram {a:g}-{b:g})")
    mira(not mal1, f"FF1: el Poincare (amb la presa sola) i els trams de 0-3 i 3-10 s de les {len(F)} fitxes son els de "
                   f"les figures de grup bit a bit" + (f"; no: {mal1[:4]}" if mal1 else ""))

    # Els punts del panell de l'energia es dibuixen de f["poincare"] mateix (q[1], q[4]): la FF2 comprova que la t i
    # l'E de cada punt son les del Poincare de grup, i que la t cau dins del tir dibuixat.
    mal2 = []
    for p in preses:
        f = F[p["nom"]]
        grup = [(q[1], q[4]) for q in r["poincare"][p["grup"]] if q[0] == p["nom"]]
        dins = all(f["t"][0] <= q[1] <= f["t"][-1] for q in f["poincare"])
        if [(q[1], q[4]) for q in f["poincare"]] != grup or not dins:
            mal2.append(p["nom"])
    mira(not mal2, "FF2: els punts del panell de l'energia tenen la t i l'E dels del Poincare, i cauen dins del tir"
                   + (f"; no: {mal2}" if mal2 else ""))

    mal3 = []
    for p in preses:
        tots = np.nonzero(p["t"] >= 0.0)[0]
        unio = np.unique(np.concatenate([tr["fotogrames"] for tr in F[p["nom"]]["trams"]]))
        if not np.array_equal(unio, tots):
            mal3.append(p["nom"])
    mira(not mal3, "FF3: els quatre trams cobreixen tots els fotogrames des de l'amollada" + (f"; no: {mal3}" if mal3 else ""))

    r2 = calcula()
    e1 = {n: _empremta(f) for n, f in F.items()}
    e2 = {n: _empremta(f) for n, f in r2["fitxes"].items()}
    mira(e1 == e2, f"FF4: dues passades, la mateixa empremta a les {len(e1)} fitxes")

    arrel_dades = (ARREL / "dades").resolve()
    dins = {str(Path(x).resolve()) for x in oberts if Path(x).resolve().is_relative_to(arrel_dades)}
    esperats = {str(fr.INDEX.resolve()), str(fr.CENTRES.resolve()), str(fr.CORRECCIO_CLICS.resolve()),
                str(lr.PAS0B_JSON.resolve()), str(lr.APROVAT.resolve()), str(lr.DUES_PARTS_2710_JSON.resolve())}
    for x in json.loads(fr.INDEX.read_text(encoding="utf-8"))["preses"]:
        if not x.get("fora_del_balanc"):
            esperats |= {str((fr.NET / f"{x['nom']}.csv").resolve()), str((fr.NET / f"{x['nom']}.json").resolve())}
    sim = [x for x in dins if "sim" in Path(x).relative_to(arrel_dades).parts]
    mira(dins == esperats and not sim,
         f"FF5: {len(dins)} fitxers oberts a dades/, {len(esperats)} esperats; de dades/sim/: {len(sim)}"
         + ("" if dins == esperats else f"; de mes: {sorted(dins - esperats)[:3]}; falten: {sorted(esperats - dins)[:3]}"))

    vistes = {}
    for n, f in F.items():
        for q in f["lambda"] or []:
            vistes.setdefault(tuple(q["parella"]), []).append((n, q["d0"], q["T"]))
    dues = all(sorted(x[0] for x in v) == sorted(k) for k, v in vistes.items())
    iguals = all(len({(x[1], x[2] if np.isfinite(x[2]) else None) for x in v}) == 1 for v in vistes.values())
    claus = list(vistes)
    lam_fit = lambda_de(np.array([vistes[k][0][1] for k in claus]),
                        np.array([vistes[k][0][2] for k in claus]), lam["D_estrella_rad"])
    mira(lam_fit == lam["lambda_grup"] and len(claus) == lam["parelles_grup"] == 45 and dues and iguals,
         f"FF6: amb les {len(claus)} parelles de les fitxes, lambda* = {lam_fit!r} (el JSON del 27/10: "
         f"{lam['lambda_grup']!r}); cada parella, a les fitxes de les seves dues preses: {dues and iguals}")
    print()
    if problemes:
        print("AUTOPROVA FALLADA:\n" + "\n".join(f"  - {q}" for q in problemes))
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> FF1-FF6. La FF7 es mirar a ull la fitxa de la 45_1, la 90_6, la 150_3 i la 150_4.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preses", nargs="+", help="nomes aquestes fitxes (per defecte, totes)")
    ap.add_argument("--figures", default=str(DIRECTORI))
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a figures/memoria/per_presa/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.memoria:
        peus.memoria()
        args.figures = str(Path(args.figures).parent / "memoria" / Path(args.figures).name)
    r = calcula(args.preses)
    for f in r["fitxes"].values():
        dibuixa_fitxa(f, [p for p in r["preses"] if p["nom"] == f["nom"]], r["escala"], r["lambda"], args.figures)
    if args.preses is None:
        cami = Path(args.figures) / "fitxes_corregit.json"
        cami.write_text(json.dumps(resum(r), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    peus.escriu(Path(args.figures).parent)     # el mateix peus.json que les figures de grup
    return 0


if __name__ == "__main__":
    sys.exit(main())
