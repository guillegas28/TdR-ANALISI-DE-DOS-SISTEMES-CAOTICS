r"""Les figures del pendol SIMPLE, el control de l'experiment 2 (i del 4).

NOMES SIMULACIO. No llegeix res de dades/ i no posa res real al costat de res
(regla 1 de documentacio/pla-analisi-caos.md): l'autoprova ho comprova.

    retrat   figures/simple_retrat: el retrat de fases del simple. Sense friccio,
             orbites tancades i la separatriu; amb la friccio FIRMADA, l'espiral
             cap al repos (tots els cicles en blau clar i un de cada 10 en fosc,
             la variant 3 que va triar en Guillem el 25/10) i l'amplitud de
             cada retorn.
    divergencia  figures/divergencia_simple_doble: 12 tirs SIMULATS del simple i 12
             del doble des de gairebe el mateix angle, i <ln(d/d0)> de les 66
             parelles de cadascun. NOMES LES CORBES: es una figura qualitativa
             del control i no se'n treu cap lambda. Sense friccio.

La friccio (28/10) es la FIRMADA el 27/10: Dissipacio.firmada(150, simple=True), el joc del
simple mesurat a les preses de decaiment 1.x (parametres.json, fisic.friccio_firmada). Ja no
es donen la C i la eps a ma (abans, «DE PROVA», amb --aire-C i --epsilon).

    .venv\\Scripts\\python eines\\control_simple.py retrat
    .venv\\Scripts\\python eines\\control_simple.py divergencia
    .venv\\Scripts\\python eines\\control_simple.py --autoprova

Model i criteris: pendol/simple.py. Pla: documentacio/pla-analisi-caos.md, «Pas 4b».
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import peus  # noqa: E402
from pendol import simple  # noqa: E402
from pendol.fisica import Dissipacio, Parametres  # noqa: E402

# ------------------------------------------------------------- les tries

AMPLITUDS_RETRAT_GRAUS = (20.0, 45.0, 90.0, 120.0, 150.0, 170.0)
# Les orbites conservatives. Els 150 son l'angle de l'espiral i de la figura de
# divergencia; els 170 ensenyen la deformacio a prop de la separatriu.

AMOLLADA_FRICCIO_GRAUS = 150.0
FPS_RETRAT = 1000.0
# A 1000 fps el retorn es troba al mostreig amb un error relatiu de l'amplitud
# de l'ordre de (omega0*dt)^2 / 8 ~ 1e-5: invisible a la figura.

ATURADA_GRAUS = 1.0
TEMPS_MAXIM_S = 900.0
# L'espiral es dibuixa fins al primer retorn per sota d'1 grau, i mai mes enlla
# de 900 s. Per sota d'1 grau el que mana es la regularitzacio del sec (tanh),
# que es del model i no del muntatge.

CADA_CICLES = 10
# L'espiral sencera fa una taca al centre (les voltes hi son a un quart de grau)
# i moire a les diagonals. Es dibuixa tota en blau clar i, a sobre, un cicle de
# cada 10 en fosc, perque es pugui llegir la forma (variant 3, 25/10).

AMOLLADA_DIVERGENCIA_GRAUS = 150.0
DISPERSIO_GRAUS = 0.3
TIRS = 12
LLAVOR = 20261001
DURADA_DIVERGENCIA_S = 10.0
FPS_DIVERGENCIA = 240.0
# Els dos pendols surten dels bracos rectes al MATEIX angle: l'unica diferencia
# es si l'articulacio esta bloquejada. La dispersio es la +-0,3 graus del «feix
# contra realitat» del CLAUDE.md. Al doble es sorteja a cada angle per separat;
# el simple, que nomes en te un, fa servir el sorteig del theta1, o sigui que el
# tir i del simple i el tir i del doble surten amb el mateix theta1. 240 fps
# perque es la graella de les preses i la del metode de parelles.
INSTANTS_DE_LECTURA_S = (0.5, 1.0, 2.0, 4.0, 8.0)
# Nomes per llegir-los al JSON. NO es cap ajust.


def dissipacio_firmada(cami: str = "parametres.json") -> Dissipacio:
    """La friccio FIRMADA del simple (27/10), per a l'amplitud d'on s'amolla l'espiral. NOMES aixo:
    els coeficients no es llegeixen a ma (documentacio/repartiment-28-10.md)."""
    return Dissipacio.firmada(AMOLLADA_FRICCIO_GRAUS, cami, simple=True)


# ------------------------------------------------------------- el retrat

def _retorns(t, y):
    """Els instants i les amplituds (rad) on w canvia de signe.

    L'instant s'interpola linealment entre les dues mostres: al retorn w passa
    per zero amb pendent -omega0^2*sin(th) != 0, o sigui que l'error es de
    segon ordre en dt (~1e-7 s a 1000 fps). L'amplitud es la mostra de |th|
    mes gran, amb l'error de (omega0*dt)^2 / 8 que diu FPS_RETRAT.
    """
    w = y[1]
    i = np.nonzero((w[:-1] > 0) != (w[1:] > 0))[0]
    tz = t[i] - w[i] * (t[i + 1] - t[i]) / (w[i + 1] - w[i])
    j = np.where(np.abs(y[0][i + 1]) > np.abs(y[0][i]), i + 1, i)
    return tz, np.abs(y[0][j])


def calcula_retrat(s: simple.Simple, dis: Dissipacio) -> dict:
    """Tot el que dibuixa la figura, sense dibuixar res."""
    w0 = s.omega0
    orbites = []
    for graus in AMPLITUDS_RETRAT_GRAUS:
        th0 = math.radians(graus)
        T = simple.periode_teoric(th0, w0)
        # Tres periodes: el primer es el que es dibuixa, i els tres serveixen
        # per mesurar el periode i la deriva d'energia.
        t, y = simple.simula(s, [th0, 0.0], 3.0 * T, fps=FPS_RETRAT)
        E = simple.energia(y[0], y[1], s)
        # Periode: el primer retorn al punt de sortida (th maxim, w = 0) es
        # interpola linealment sobre els passos per zero de w.
        tr, _ = _retorns(t, y)
        orbites.append({
            "amplitud_graus": graus,
            "t": t, "y": y,
            "periode_teoric_s": T,
            "periode_mostreig_s": float(np.mean(np.diff(tr[1::2]))) if len(tr) >= 4 else None,
            "deriva_energia_gMd": float(np.max(np.abs(E - E[0])) / (s.g * s.Md)),
        })

    th0 = math.radians(AMOLLADA_FRICCIO_GRAUS)
    t, y = simple.simula(s, [th0, 0.0], TEMPS_MAXIM_S, fps=FPS_RETRAT, dis=dis)
    tr, ar = _retorns(t, y)
    sota = np.nonzero(ar < math.radians(ATURADA_GRAUS))[0]
    if len(sota):
        fi = int(np.searchsorted(t, tr[sota[0]])) + 1
        t_aturada = float(tr[sota[0]])
    else:
        fi, t_aturada = len(t), None
    E = simple.energia(y[0][:fi], y[1][:fi], s)
    return {
        "orbites": orbites,
        "espiral": {"t": t[:fi], "y": y[:, :fi], "t_retorns": tr[tr <= t[fi - 1]],
                    "amplituds": ar[tr <= t[fi - 1]], "t_sota_1_grau_s": t_aturada,
                    "energia_inicial_des_del_repos_J": float(E[0] + s.g * s.Md),
                    "energia_final_des_del_repos_J": float(E[-1] + s.g * s.Md),
                    "energia_puja_mai": bool(np.all(np.diff(E) <= 0.0))},
    }


def cicles_destacats(e: dict, cada: int = CADA_CICLES) -> list[tuple[int, int]]:
    """Els cicles de l'espiral que es dibuixen en fosc, com a trossos [i:j] de e["y"].

    Un cicle va d'un extrem de la dreta (w = 0 amb th > 0) al seguent: el primer surt
    de l'amollada (t = 0) i els altres, dels retorns de la dreta. Cada extrem es la
    primera mostra despres del retorn, i el tros inclou la de l'extrem final. Es
    destaquen els cicles 0, cada, 2*cada... que acaben dins de l'espiral.
    """
    t, th = e["t"], e["y"][0]
    k = np.searchsorted(t, e["t_retorns"])
    inicis = [0] + [int(x) for x in k if th[x] > 0.0]
    return [(inicis[n], inicis[n + 1] + 1) for n in range(0, len(inicis) - 1, cada)]


def figura_retrat(s: simple.Simple, dis: Dissipacio, r: dict):
    """La figura, sense desar-la; torna (fig, les linies de l'espiral, els trossos)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from pendol import estil

    estil.aplica()
    w0 = s.omega0
    fig = plt.figure(figsize=(11.0, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0), hspace=0.38, wspace=0.22)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1], sharex=a, sharey=a)
    c = fig.add_subplot(gs[1, :])

    # --- A: conservatiu
    th = np.linspace(-math.pi, math.pi, 721)
    sep = 2.0 * w0 * np.cos(th / 2.0)
    for signe in (1.0, -1.0):
        a.plot(np.degrees(th), signe * sep, color=estil.TINTA_SUAU, lw=1.0, ls=":",
               label="separatriu (analítica)" if signe > 0 else None)
    for k, o in enumerate(r["orbites"]):
        T = o["periode_teoric_s"]
        n = int(np.searchsorted(o["t"], T)) + 2
        # Continu: el discontinu vol dir «simulacio» nomes al costat de dades reals.
        a.plot(np.degrees(o["y"][0][:n]), o["y"][1][:n], color=estil.SIMULACIO,
               ls="-", lw=1.3, label="simulació sense fricció" if k == 0 else None)
        # Alternades a sobre i a sota de l'eix: 150 i 170 son a 20 graus.
        a.annotate(f"{o['amplitud_graus']:g}°", (o["amplitud_graus"], 0.0),
                   xytext=(3, 4 if k % 2 == 0 else -12), textcoords="offset points",
                   fontsize=8, color=estil.TINTA_2)
    a.set_title("Sense fricció: òrbites tancades", loc="left")
    a.set_xlabel("θ (graus)")
    a.set_ylabel("ω (rad/s)")
    a.set_xlim(-185, 185)
    a.set_ylim(-19.0, 23.0)          # el marge de dalt es per a la llegenda
    a.set_xticks(range(-180, 181, 60))
    a.legend(loc="upper left", fontsize=8, ncol=2)

    # --- B: amb la friccio firmada
    e = r["espiral"]
    (fons,) = b.plot(np.degrees(e["y"][0]), e["y"][1], color=estil.FEIX, lw=0.35,
                     rasterized=True)
    trossos = cicles_destacats(e)
    cicles = [b.plot(np.degrees(e["y"][0][i:j]), e["y"][1][i:j], color=estil.SIMULACIO,
                     lw=0.9)[0] for i, j in trossos]
    b.legend(handles=[Line2D([], [], color=estil.FEIX, lw=1.5, label="tots els cicles"),
                      Line2D([], [], color=estil.SIMULACIO, lw=1.5,
                             label=f"un cicle de cada {CADA_CICLES}")],
             loc="upper right", fontsize=7.5, handlelength=1.2, handletextpad=0.5,
             borderaxespad=0.3)   # estreta: amb la de per defecte toca el text dels coeficients
    b.plot([AMOLLADA_FRICCIO_GRAUS], [0.0], marker="o", ms=5, color=estil.TINTA_2, ls="")
    b.annotate(f"amollat\nde {AMOLLADA_FRICCIO_GRAUS:g}°", (AMOLLADA_FRICCIO_GRAUS, 0.0),
               xytext=(5, -26), textcoords="offset points", ha="left", fontsize=8,
               color=estil.TINTA_2)
    b.set_title("Amb la fricció firmada: espiral cap al repòs", loc="left")
    b.set_xlabel("θ (graus)")
    b.tick_params(labelleft=False)
    b.text(0.02, 0.98, _text_dissipacio(dis), transform=b.transAxes, va="top",
           fontsize=7.5, color=estil.TINTA_2)

    # --- C: l'amplitud de cada retorn
    c.plot(e["t_retorns"], np.degrees(e["amplituds"]), ls="", marker="o", ms=2.6,
           color=estil.SIMULACIO)
    c.set_title("Amb la fricció firmada: amplitud de cada retorn", loc="left")
    c.set_xlabel("t (s)")
    c.set_ylabel("amplitud (graus)")
    c.set_ylim(bottom=0)
    if e["t_sota_1_grau_s"] is not None:
        c.annotate(f"per sota d'{ATURADA_GRAUS:g}° a {_coma(e['t_sota_1_grau_s'], '.1f')} s",
                   (e["t_sota_1_grau_s"], ATURADA_GRAUS), xytext=(-6, 14),
                   textcoords="offset points", ha="right", fontsize=8, color=estil.TINTA_2)

    peu = (f"Pèndol simple SIMULAT: el doble amb l'articulació bloquejada i els braços "
           f"rectes, I = I1 + I2 + 2K = {_coma(s.I, '.4e')} kg·m², ω0 = {_coma(w0, '.3f')} rad/s "
           f"(mesures.omega_zero). Cap dada real. DOP853, tolerància estricta. "
           f"Espiral: de fons, tots els cicles en blau clar; a sobre, en fosc, un cicle de "
           f"cada {CADA_CICLES} començant per l'amollada. Les voltes fosques no són òrbites "
           f"tancades: entre l'una i l'altra el pèndol en fa {CADA_CICLES - 1} més. "
           f"La fricció és la firmada el 27/10, la del simple: la mesurada a les preses de "
           f"decaiment 1.x (Dissipacio.firmada(150, simple=True)).")
    peus.peu(fig, 0.01, 0.005, peu, fontsize=8, color=estil.TINTA_2, ha="left", va="top",
             wrap=True)
    return fig, {"fons": fons, "cicles": cicles}, trossos


def marges_de_la_llegenda(fig, b, th, w) -> tuple[float, float]:
    """La distancia (punts tipografics) de la llegenda del panell B al text dels
    coeficients i a l'espiral. 0 vol dir que s'hi toquen o s'hi solapen."""
    fig.canvas.draw()
    caixa = b.get_legend().get_window_extent(fig.canvas.get_renderer())
    text = [x for x in b.texts if x.get_text().startswith("τ_f1")][0].get_window_extent()
    pt = 72.0 / fig.dpi
    dx = max(text.x0 - caixa.x1, caixa.x0 - text.x1, 0.0)
    dy = max(text.y0 - caixa.y1, caixa.y0 - text.y1, 0.0)
    px = b.transData.transform(np.column_stack([np.degrees(th), w]))
    ex = np.maximum(np.maximum(caixa.x0 - px[:, 0], px[:, 0] - caixa.x1), 0.0)
    ey = np.maximum(np.maximum(caixa.y0 - px[:, 1], px[:, 1] - caixa.y1), 0.0)
    return math.hypot(dx, dy) * pt, float(np.min(np.hypot(ex, ey))) * pt


def dibuixa_retrat(s: simple.Simple, dis: Dissipacio, r: dict, base: str,
                   directori: str = "figures") -> list[tuple[int, int]]:
    from pendol import estil

    fig, _, trossos = figura_retrat(s, dis, r)
    peus.desa(fig, base, directori)
    return trossos


def _coma(x: float, fmt: str) -> str:
    """Un numero amb coma decimal, com s'escriu en catala."""
    return format(x, fmt).replace(".", ",")


def _text_dissipacio(dis: Dissipacio) -> str:
    return (f"τ_f1 = {_coma(dis.tau_f1, '.3e')} N·m   b1 = {_coma(dis.b1, 'g')}\n"
            f"C = {_coma(dis.C, '.4g')} kg/m   ε = {_coma(dis.eps, 'g')} rad/s\n"
            f"fricció firmada (27/10)")


def metriques_retrat(s: simple.Simple, dis: Dissipacio, r: dict) -> dict:
    e = r["espiral"]
    trossos = cicles_destacats(e)
    return {
        "figura": "simple_retrat",
        "que_es": "retrat de fases del pendol simple SIMULAT; cap dada real",
        **simple.metadades(s, dis, firmada=True),
        "friccio_nomes_als_panells": "B i C (el A es sense friccio)",
        "orbites_sense_friccio": [
            {k: v for k, v in o.items() if k not in ("t", "y")} for o in r["orbites"]],
        "espiral": {
            "etiqueta": "amb la friccio FIRMADA el 27/10 (Dissipacio.firmada(150, simple=True))",
            "amollat_graus": AMOLLADA_FRICCIO_GRAUS,
            "t_sota_1_grau_s": e["t_sota_1_grau_s"],
            "retorns": int(len(e["t_retorns"])),
            "energia_inicial_des_del_repos_J": e["energia_inicial_des_del_repos_J"],
            "energia_final_des_del_repos_J": e["energia_final_des_del_repos_J"],
            "energia_puja_mai": e["energia_puja_mai"],
            "t_retorns_s": [float(x) for x in e["t_retorns"]],
            "amplituds_retorn_graus": [float(x) for x in np.degrees(e["amplituds"])],
            "voltes_destacades": {
                "que_es": "els cicles dibuixats en fosc: un de cada N, comencant per "
                          "l'amollada; de fons, tots en blau clar",
                "cada_cicles": CADA_CICLES,
                "destacats": len(trossos),
                "t_inici_s": [float(e["t"][i]) for i, _ in trossos],
                "t_final_s": [float(e["t"][j - 1]) for _, j in trossos],
            },
        },
        "fps": FPS_RETRAT,
        "tolerancia": "estricta (pendol/simulador.py: RTOL, ATOL, DOP853)",
    }


def fes_retrat(args) -> int:
    p = Parametres.des_de_json(args.parametres)
    s = simple.Simple.des_de_parametres(p)
    dis = dissipacio_firmada(args.parametres)
    print(f"Retrat del simple: omega0 = {s.omega0:.4f} rad/s; friccio FIRMADA "
          f"tau_f1 = {dis.tau_f1:.3e}, b1 = {dis.b1:.3e}, C = {dis.C:g}, eps = {dis.eps:g}")
    r = calcula_retrat(s, dis)
    dibuixa_retrat(s, dis, r, "simple_retrat", args.figures)
    m = metriques_retrat(s, dis, r)
    cami = Path(args.figures) / "simple_retrat.json"
    cami.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  metriques: {cami}")
    for o in m["orbites_sense_friccio"]:
        print(f"  {o['amplitud_graus']:5.0f} graus: T teoric {o['periode_teoric_s']:.6f} s, "
              f"del mostreig {o['periode_mostreig_s']:.6f} s, deriva {o['deriva_energia_gMd']:.1e}")
    e = m["espiral"]
    print(f"  espiral amb la friccio firmada: {e['retorns']} retorns, per sota d'1 grau a "
          f"{e['t_sota_1_grau_s']} s, l'energia no puja mai: {e['energia_puja_mai']}")
    print(f"  en fosc: {e['voltes_destacades']['destacats']} cicles, un de cada "
          f"{CADA_CICLES} comencant per l'amollada")
    return 0


# ------------------------------------------------------------- la divergencia

def calcula_divergencia(p, s: simple.Simple) -> dict:
    """Els 12 + 12 tirs i <ln(d/d0)> de les seves parelles, sense dibuixar res.

    La distancia es la de tot arreu (caos.matriu_ln_d -> mesures.distancia_fases)
    amb la mateixa omega0 per als dos pendols, que es justament la del simple. El
    simple hi entra com (th, th, w, w): la seva d surt per sqrt(2), cosa que
    ln(d/d0) no nota (pendol/simple.py, capdamunt).
    """
    from pendol import caos, fisica, simulador

    w0 = s.omega0
    rng = np.random.default_rng(LLAVOR)
    desv = np.radians(DISPERSIO_GRAUS) * rng.standard_normal((TIRS, 2))
    th0 = math.radians(AMOLLADA_DIVERGENCIA_GRAUS)
    t = np.arange(0.0, DURADA_DIVERGENCIA_S + 0.5 / FPS_DIVERGENCIA, 1.0 / FPS_DIVERGENCIA)
    gMd = s.g * s.Md

    doble, simples, der_d, der_s = [], [], [], []
    for e1, e2 in desv:
        x0 = [th0 + e1, th0 + e2, 0.0, 0.0]
        _, y = simulador.simula(p, x0, t_eval=t)
        doble.append(y)
        E = fisica.energia(y, p)
        der_d.append(float(np.max(np.abs(E - E[0])) / gMd))

        _, z = simple.simula(s, [th0 + e1, 0.0], t_eval=t)
        simples.append(simple.com_a_doble(z))
        E = simple.energia(z[0], z[1], s)
        der_s.append(float(np.max(np.abs(E - E[0])) / gMd))

    sortida = {"t": t, "desviacions_rad": desv, "omega0": w0,
               "E_rel_nominal_doble": float(caos.energia_relativa([th0, th0, 0.0, 0.0], p)
                                            / caos.energia_volteig_2(p))}
    for nom, tirs, der in (("doble", doble, der_d), ("simple", simples, der_s)):
        L = caos.matriu_ln_d(np.array(tirs), w0)
        i, j = np.triu_indices(TIRS, 1)
        ln_d = L[i, j]                       # (66, M)
        m_norm, m_ln = caos.mitjana_log(ln_d, ln_d[:, 0])
        sortida[nom] = {"ln_d": ln_d, "mitjana_ln_d_d0": m_norm, "mitjana_ln_d": m_ln,
                        "deriva_energia_gMd": der, "estats": np.array(tirs)}
    return sortida


def dibuixa_divergencia(r: dict, base: str, directori: str = "figures") -> None:
    import matplotlib.pyplot as plt

    from pendol import estil

    estil.aplica()
    t = r["t"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.9))
    series = (
        ("doble", "pèndol doble", estil.SIMULACIO, estil.FEIX, "-"),
        ("simple", "pèndol simple (articulació bloquejada)", estil.TINTA_2, estil.EIX,
         (0, (1.2, 1.4))),
    )
    for eix, escala in ((a, "lineal"), (b, "log")):
        sel = slice(None) if escala == "lineal" else slice(1, None)   # log: sense t = 0
        for clau, nom, color, color_fons, trac in series:
            d = r[clau]
            ln_rel = d["ln_d"] - d["ln_d"][:, :1]
            for fila in ln_rel:
                eix.plot(t[sel], fila[sel], color=color_fons, lw=0.4, alpha=0.55,
                         zorder=1, rasterized=True)
            eix.plot(t[sel], d["mitjana_ln_d_d0"][sel], color=color, ls=trac, lw=2.0,
                     zorder=3, label=f"{nom}: mitjana de les 66 parelles")
        eix.axhline(0.0, color=estil.EIX, lw=0.8, zorder=0)
        eix.set_ylabel("⟨ln(d / d₀)⟩")
    a.set_xlabel("t des de l'amollada (s)")
    a.set_title("Temps lineal: el doble puja recte fins que satura", loc="left")
    b.set_xscale("log")
    b.set_xlabel("t des de l'amollada (s), escala logarítmica")
    b.set_title("Temps logarítmic: l'envolupant del simple puja recta", loc="left")
    a.legend(loc="upper left", fontsize=8)
    peu = (f"SIMULACIÓ, cap dada real, sense fricció. {TIRS} tirs de cada pèndol amollats dels "
           f"braços rectes a {AMOLLADA_DIVERGENCIA_GRAUS:g}° + N(0; {_coma(DISPERSIO_GRAUS, 'g')}°) "
           f"a cada angle (llavor {LLAVOR}), {FPS_DIVERGENCIA:g} fps, tolerància estricta. "
           f"d = mesures.distancia_fases amb ω0 = {_coma(r['omega0'], '.3f')} rad/s per als dos; "
           f"el simple com a (θ, θ, ω, ω). Línies fines: cada parella; gruixudes: ⟨ln(d/d₀)⟩. "
           f"Les 66 parelles del simple gairebé coincideixen (d/d₀ no depèn de quina parella) "
           f"i queden amagades sota la mitjana. "
           f"Figura qualitativa del control: no se'n treu cap λ.")
    peus.peu(fig, 0.01, -0.01, peu, fontsize=8, color=estil.TINTA_2, ha="left", va="top", wrap=True)
    fig.tight_layout()
    peus.desa(fig, base, directori)


def metriques_divergencia(r: dict) -> dict:
    t = r["t"]
    lectura = {}
    for clau in ("doble", "simple"):
        d = r[clau]
        lectura[clau] = {
            "mitjana_ln_d_d0_a": {f"{x:g} s": float(d["mitjana_ln_d_d0"][int(round(x * FPS_DIVERGENCIA))])
                                  for x in INSTANTS_DE_LECTURA_S},
            "d0_rad": {"minim": float(np.exp(d["ln_d"][:, 0].min())),
                       "mediana": float(np.exp(np.median(d["ln_d"][:, 0]))),
                       "maxim": float(np.exp(d["ln_d"][:, 0].max()))},
            "deriva_energia_gMd_maxima": float(max(d["deriva_energia_gMd"])),
        }
    return {
        "figura": "divergencia_simple_doble",
        "que_es": "<ln(d/d0)> de 66 parelles de tirs SIMULATS de cada pendol; cap dada real",
        "cap_lambda": "no se'n treu cap lambda: es una figura qualitativa del control",
        "friccio": "sense",
        "amollada_graus": AMOLLADA_DIVERGENCIA_GRAUS,
        "dispersio_graus": DISPERSIO_GRAUS,
        "tirs": TIRS, "parelles": TIRS * (TIRS - 1) // 2, "llavor": LLAVOR,
        "durada_s": DURADA_DIVERGENCIA_S, "fps": FPS_DIVERGENCIA,
        "omega0": r["omega0"],
        "norma": "mesures.distancia_fases (via caos.matriu_ln_d); el simple com (th, th, w, w)",
        "E_sobre_E_volteig_2_del_doble_nominal": r["E_rel_nominal_doble"],
        "desviacions_graus": np.degrees(r["desviacions_rad"]).tolist(),
        "tolerancia": "estricta (pendol/simulador.py)",
        "lectura_sense_ajust": lectura,
        "corbes": "figures/divergencia_simple_doble.npz",
        "t_final_s": float(t[-1]),
    }


def fes_divergencia(args) -> int:
    p = Parametres.des_de_json(args.parametres)
    s = simple.Simple.des_de_parametres(p)
    print(f"Divergencia: {TIRS} + {TIRS} tirs SIMULATS des de {AMOLLADA_DIVERGENCIA_GRAUS:g} graus "
          f"+- {DISPERSIO_GRAUS:g}, {DURADA_DIVERGENCIA_S:g} s a {FPS_DIVERGENCIA:g} fps")
    r = calcula_divergencia(p, s)
    dibuixa_divergencia(r, "divergencia_simple_doble", args.figures)
    m = metriques_divergencia(r)
    d = Path(args.figures)
    (d / "divergencia_simple_doble.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(d / "divergencia_simple_doble.npz", t=r["t"],
                        doble_ln_d=r["doble"]["ln_d"], simple_ln_d=r["simple"]["ln_d"],
                        doble_mitjana=r["doble"]["mitjana_ln_d_d0"],
                        simple_mitjana=r["simple"]["mitjana_ln_d_d0"])
    print(f"  metriques: {d / 'divergencia_simple_doble.json'}  i  .npz")
    print(f"  E/E_v2 del doble nominal: {m['E_sobre_E_volteig_2_del_doble_nominal']:.3f}")
    for clau, l in m["lectura_sense_ajust"].items():
        vals = "  ".join(f"{k}: {v:5.2f}" for k, v in l["mitjana_ln_d_d0_a"].items())
        print(f"  {clau:6s} <ln(d/d0)>  {vals}   (deriva d'energia {l['deriva_energia_gMd_maxima']:.1e})")
    return 0


# --------------------------------------------------------------- l'autoprova

def autoprova() -> int:
    """C1-C3 del pla (01/10/2026) i C4 (25/10), fixats abans d'executar res."""
    import builtins
    import io

    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    print("Autoprova de eines/control_simple.py\n")
    p = Parametres.des_de_json("parametres.json")
    s = simple.Simple.des_de_parametres(p)
    dis = dissipacio_firmada()

    # C2: es vigila TOT el que s'obre mentre es calcula.
    oberts = []
    obre_b, obre_io = builtins.open, io.open

    def espia(f, *a, **k):
        oberts.append(str(f))
        return obre_b(f, *a, **k)

    builtins.open = io.open = espia
    try:
        r1 = calcula_retrat(s, dis)
        v1 = calcula_divergencia(p, s)
    finally:
        builtins.open, io.open = obre_b, obre_io
    r2 = calcula_retrat(s, dis)
    v2 = calcula_divergencia(p, s)

    print("C1. Dues passades donen el mateix bit a bit")
    iguals = all(np.array_equal(o1["y"], o2["y"]) for o1, o2 in zip(r1["orbites"], r2["orbites"]))
    iguals = iguals and np.array_equal(r1["espiral"]["y"], r2["espiral"]["y"])
    mira(iguals, "retrat: orbites i espiral identiques")
    mira(all(np.array_equal(v1[k]["estats"], v2[k]["estats"])
             and np.array_equal(v1[k]["mitjana_ln_d_d0"], v2[k]["mitjana_ln_d_d0"])
             for k in ("doble", "simple")),
         "divergencia: els 24 tirs i les dues corbes identics")

    print("\nC2. No obre res de dades/")
    de_dades = [f for f in oberts if "dades" in Path(f).parts]
    mira(not de_dades, f"fitxers oberts durant el calcul: {len(oberts)}, de dades/: {de_dades}")

    # C3 reescrita el 28/10 ABANS de correr (pla, «28/10 · La feina del repartiment», 2): abans era «sense C i
    # eps no corre, i diu DE PROVA»; ara la friccio es la firmada.
    print("\nC3. La friccio de l'espiral es exactament la firmada del simple, i ho diu")
    mira(dis == Dissipacio.firmada(AMOLLADA_FRICCIO_GRAUS, simple=True),
         f"es Dissipacio.firmada({AMOLLADA_FRICCIO_GRAUS:g}, simple=True): {dis}")
    m = metriques_retrat(s, dis, r1)
    mira("FIRMADA" in m["friccio"] and "FIRMADA" in m["espiral"]["etiqueta"],
         "les metriques del retrat diuen FIRMADA a la friccio i a l'espiral")
    mira(bool(r1["espiral"]["energia_puja_mai"]), "a l'espiral, l'energia no puja mai")

    # C4 (SR2 del pla, 25/10): es comprova amb els canvis de signe de w, sense passar
    # pels retorns interpolats que fa servir cicles_destacats.
    print(f"\nC4. Els cicles foscos de l'espiral: un de cada {CADA_CICLES}, trossos exactes")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e = r1["espiral"]
    th, w = e["y"]
    canvi = np.nonzero((w[:-1] > 0) != (w[1:] > 0))[0] + 1   # la mostra just despres
    trossos = cicles_destacats(e)
    mira(bool(trossos) and trossos[0][0] == 0, "el primer cicle fosc surt de l'amollada (mostra 0)")
    bons = all(th[i] > 0 and th[j - 1] > 0 and len(dins := canvi[(canvi > i) & (canvi < j)]) == 2
               and dins[-1] == j - 1 for i, j in trossos)
    mira(bons, "cadascun va d'un extrem de la dreta al seguent: dos retorns, el segon a la "
               "seva ultima mostra")
    inicis = [i for i, _ in trossos]
    salts = [int(np.count_nonzero((canvi > a) & (canvi <= b))) for a, b in zip(inicis, inicis[1:])]
    mira(bool(salts) and all(x == 2 * CADA_CICLES for x in salts),
         f"entre dos cicles foscos seguits n'hi ha {CADA_CICLES}")
    complets = int(np.count_nonzero(th[canvi] > 0))
    mira(len(trossos) == len(range(0, complets, CADA_CICLES))
         and all(j <= len(e["t"]) for _, j in trossos),
         f"{len(trossos)} foscos de {complets} cicles complets, tots dins de l'espiral dibuixada")
    fig, linies, de_la_figura = figura_retrat(s, dis, r1)
    fx, fy = linies["fons"].get_data(orig=True)
    mira(np.array_equal(fx, np.degrees(th)) and np.array_equal(fy, w),
         "el fons de la figura es l'espiral sencera, bit a bit")
    mira(de_la_figura == trossos and len(linies["cicles"]) == len(trossos)
         and all(np.array_equal(l.get_xdata(orig=True), np.degrees(th[i:j]))
                 and np.array_equal(l.get_ydata(orig=True), w[i:j])
                 for l, (i, j) in zip(linies["cicles"], trossos)),
         "cada cicle fosc de la figura es el seu tros de l'espiral, bit a bit (no recalculat)")
    # Afegit el 25/10 despres que la SR4 trobes la llegenda tocant el text (pla, «25/10 ·
    # L'espiral»): la llegenda queda a MARGE_PT o mes del text dels coeficients i de l'espiral.
    MARGE_PT = 5.0
    sep_text, sep_espiral = marges_de_la_llegenda(fig, linies["fons"].axes, th, w)
    mira(sep_text >= MARGE_PT and sep_espiral >= MARGE_PT,
         f"la llegenda queda a {sep_text:.1f} pt del text dels coeficients i a {sep_espiral:.1f} pt "
         f"de l'espiral (com a minim {MARGE_PT:g})")
    peu = " ".join(x.get_text() for x in fig.texts)
    plt.close(fig)
    mira(f"un cicle de cada {CADA_CICLES}" in peu and "firmada" in peu,
         "el peu diu «un cicle de cada 10» i «firmada»")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for x in problemes:
            print(f"  - {x}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> l'eina es reproduible, no toca res real, la friccio es la firmada, i els "
          "cicles foscos de l'espiral en son trossos exactes.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ordre", nargs="?", choices=("retrat", "divergencia"))
    ap.add_argument("--parametres", default="parametres.json")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--memoria", action="store_true",
                    help="sense el peu tecnic dins de la imatge, a <figures>/memoria/ (el peu, a peus.json; 2/11)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.ordre is None:
        ap.error("digues quina figura: retrat o divergencia")
    if args.memoria:
        peus.memoria()
        args.figures = str(Path(args.figures) / "memoria")
    sortida = fes_divergencia(args) if args.ordre == "divergencia" else fes_retrat(args)
    peus.escriu(args.figures)
    return sortida


if __name__ == "__main__":
    raise SystemExit(main())
