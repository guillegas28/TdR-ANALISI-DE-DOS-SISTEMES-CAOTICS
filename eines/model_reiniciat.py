r"""El model reiniciat despres d'amollar (29/10/2026). ANALISI POSTERIOR, NO FIRMADA: no canvia cap veredicte.

La pregunta: un cop la placa ja no toca el pendol, el model el segueix tant com el seu propi caos permet? Pla, criteris i lectura
(escrits abans de calcular): documentacio/pla-placa-reinici-comes.md, punt 2.

Per a cada presa de l'experiment 1 (45 i 90 graus), amb la serie corregida i el zero afinat (experiment1.presa_corregida):
  - a t0 = 0,10 s despres del zero (0,05 i 0,20, de sensibilitat), l'estat (theta i omega) amb el Savitzky-Golay de la conversio
    (13 mostres, ordre 3) a la primera mostra amb tau >= t0;
  - la simulacio des d'aquell estat, amb els parametres nominals i sense friccio (com el feix firmat), fins a 3 s despres;
  - T_D: el primer instant en que max(|dtheta1|, |dtheta2|) passa de D i s'hi queda 0,2 s (experiment1.temps_separacio), amb
    D = 10 graus (la principal), 5 i 20;
  - la referencia, els 200 bessons: els membres del feix firmat de la presa amb el seu soroll (els de cap_on), cadascun passat
    exactament pel mateix cami. El numero es el percentil de la T_D real entre els bessons, i per grup, la mediana amb l'interval
    del 95 % per bootstrap (experiment1.resum_cap_on);
  - la friccio (informacio): quant triga la simulacio amb la friccio firmada a separar-se D de la de sense, des del mateix estat.

Darrere del pany de firma.json.

    .venv\Scripts\python.exe eines\model_reiniciat.py --autoprova
    .venv\Scripts\python.exe eines\model_reiniciat.py [--processos 3]
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_coeffs

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
import experiment1 as e1  # noqa: E402
import prepara_versemblant as pv  # noqa: E402
from pendol.fisica import Dissipacio, Parametres  # noqa: E402
from pendol.simulador import simula  # noqa: E402

SORTIDA = ARREL / "dades" / "experiment1" / "model_reiniciat_2026-10-29.json"
CAP_ON = ARREL / "dades" / "experiment1" / "cap_on_2026-10-27.json"
FIGURES = ARREL / "figures"
T0_S = (0.10, 0.05, 0.20)                # el principal, i dos de sensibilitat (fixats abans de calcular)
T_DESPRES_S = 3.0
T_FEIX_S = max(T0_S) + T_DESPRES_S + 0.05
D_GRAUS = e1.D_CAP_ON_GRAUS              # (10, 5, 20): els de cap_on
FINESTRA_SG, ORDRE_SG = 13, 3            # els de la conversio


def clau(x: float) -> str:
    return f"{x:g}"


# ------------------------------------------------------------ el reinici d'una serie (el mateix per a la presa i els bessons)

def estat_sg(a1, a2, i0: int, dt: float) -> np.ndarray:
    """theta i omega a la mostra i0, amb el Savitzky-Golay (derivada 0 i 1) sobre les mostres i0 - 6 ... i0 + 6."""
    h = FINESTRA_SG // 2
    c0 = savgol_coeffs(FINESTRA_SG, ORDRE_SG, deriv=0, use="dot")
    c1 = savgol_coeffs(FINESTRA_SG, ORDRE_SG, deriv=1, delta=dt, use="dot")
    w1, w2 = np.asarray(a1[i0 - h:i0 + h + 1], float), np.asarray(a2[i0 - h:i0 + h + 1], float)
    return np.array([c0 @ w1, c0 @ w2, c1 @ w1, c1 @ w2])


def separa(ts_rel, a1, a2, b1, b2) -> dict:
    """La T_D de cada D entre (a1, a2) i (b1, b2), sobre ts_rel."""
    d = np.maximum(np.abs(np.asarray(a1) - b1), np.abs(np.asarray(a2) - b2))
    return {clau(D): float(e1.temps_separacio(ts_rel, d, math.radians(D))[0]) for D in D_GRAUS}


def reinicia(p: Parametres, ts, a1, a2, i0: int, dt: float, dis=None):
    """L'estat a i0, la simulacio des d'alli fins a T_DESPRES_S, i la T_D contra la mateixa serie."""
    fi = int(np.searchsorted(ts, ts[i0] + T_DESPRES_S, side="right"))
    ts_rel = ts[i0:fi] - ts[i0]
    estat = estat_sg(a1, a2, i0, dt)
    _, y = simula(p, estat, t_eval=ts_rel, dis=dis)
    return estat, ts_rel, y, separa(ts_rel, a1[i0:fi], a2[i0:fi], y[0], y[1])


def analitza(q: dict, tg, A1, A2, N, p: Parametres, grup: str, n_bessons: int | None = None) -> dict:
    """q: la presa (t, th1, th2, zero), com experiment1.presa_corregida. Torna, per a cada t0, la T_D real, la dels bessons, el
    percentil i la comprovacio de la friccio."""
    tau = np.asarray(q["t"], float) - q["zero"]
    sel = (tau >= 0.0) & (tau <= T_FEIX_S)
    ts = tau[sel]
    r1, r2 = np.asarray(q["th1"])[sel], np.asarray(q["th2"])[sel]
    dt = float(np.median(np.diff(ts)))
    nb = len(A1) - 1 if n_bessons is None else n_bessons
    M1 = np.array([np.interp(ts, tg, A1[i]) for i in range(1, nb + 1)]) + N[0][1:nb + 1, :len(ts)]
    M2 = np.array([np.interp(ts, tg, A2[i]) for i in range(1, nb + 1)]) + N[1][1:nb + 1, :len(ts)]
    fora = {}
    for t0 in T0_S:
        i0 = int(np.nonzero(ts >= t0)[0][0])
        estat, ts_rel, y, t_real = reinicia(p, ts, r1, r2, i0, dt)
        _, yf = simula(p, estat, t_eval=ts_rel, dis=Dissipacio.firmada(float(grup)))
        t_bes = [reinicia(p, ts, M1[i], M2[i], i0, dt)[3] for i in range(nb)]
        fora[clau(t0)] = {
            "i0": i0, "t_reinici_s": float(ts[i0]),
            "estat_real": {"theta_graus": [math.degrees(estat[0]), math.degrees(estat[1])], "omega_rad_s": [float(estat[2]),
                                                                                                          float(estat[3])]},
            "friccio_t_D_s": separa(ts_rel, y[0], y[1], yf[0], yf[1]),
            "per_D": {k: {"t_real_s": t_real[k], "percentil": e1.percentil(t_real[k], [b[k] for b in t_bes]),
                          "t_bessons_mediana_s": float(np.median([b[k] for b in t_bes])),
                          "bessons_que_no_se_separen": int(sum(not math.isfinite(b[k]) for b in t_bes)),
                          "t_bessons": [b[k] for b in t_bes]} for k in t_real}}
    return fora


_OBERTS: list[str] = []
_GANXO = [False]


def _vigila() -> None:
    """La J: un sol ganxo per proces, que apunta tot el que s'obre (tambe als fills del Pool)."""
    if not _GANXO[0]:
        sys.addaudithook(lambda ev, a: _OBERTS.append(str(a[0])) if ev == "open" and isinstance(a[0], str) else None)
        _GANXO[0] = True


def una_presa(nom: str) -> dict:
    """Una presa real. Nomes es crida des de l'ordre, despres del pany."""
    _vigila()
    n0 = len(_OBERTS)
    t00 = time.time()
    grup = nom.split("_")[0]
    p = Parametres.des_de_json(e1.PARAMETRES)
    q = e1.presa_corregida(nom, p)
    tg, A1, A2 = e1.feix(p, q["estat0"], nom, grup, t_fins=T_FEIX_S)
    N = e1.soroll_de_la_presa(p, nom, q["meta"], T_FEIX_S)
    fora = {"nom": nom, "grup": grup, "zero_s": q["zero"], "per_t0": analitza(q, tg, A1, A2, N, p, grup)}
    for v in fora["per_t0"].values():           # les 200 T_D de cada besso no es desen: el JSON es versiona
        for w in v["per_D"].values():
            del w["t_bessons"]
    ob = _OBERTS[n0:]
    fora["oberts"] = {"net": sorted({Path(o).stem for o in ob if "net" in Path(o).parts}),
                      "sim": sorted({o for o in ob if "sim" in Path(o).parts})}
    print(f"   {nom}: {time.time() - t00:.0f} s", flush=True)
    return fora


# ------------------------------------------------------------ la lectura (escrita abans: pla-placa-reinici-comes.md)

def lectura(resum: dict) -> str:
    lo, hi = resum["interval_95"]
    return ("abans que els bessons" if hi < 0.5 else "despres que els bessons" if lo > 0.5 else "compatible amb els bessons")


def contrast(per_D_reinici: dict, per_D_sensibilitat: list[dict]) -> str:
    """La lectura del grup, amb t0 = 0,10 s, i si t0 = 0,05 i 0,20 diuen el mateix."""
    lect = {k: v["lectura"] for k, v in per_D_reinici.items()}
    if any(s[k]["lectura"] != lect[k] for s in per_D_sensibilitat for k in lect):
        return "depen de quan es reinicia: no se'n treu cap conclusio"
    if all(v == "compatible amb els bessons" for v in lect.values()):
        return "compatible a les tres D: indicacio que, un cop amollat, el model segueix el pendol"
    if any(lect[clau(D)] == "abans que els bessons" for D in (5.0, 20.0)):
        return "abans a 5 o a 20 graus: el model difereix del pendol tambe despres de l'amollada"
    return "cap de les lectures escrites: " + ", ".join(f"D = {k}: {v}" for k, v in lect.items())


def ordre(firma: dict | None, processos: int) -> int:
    x = float(firma["X"]) if firma and isinstance(firma.get("X"), (int, float)) else math.nan
    e1.pany(firma, x, e1.GRUPS_Y)
    noms = [n for g in e1.GRUPS_Y for n in e1.preses_del_grup(g)]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(processos) as pool:
            res = pool.map(una_presa, noms)
    else:
        res = [una_presa(n) for n in noms]
    capon = json.loads(CAP_ON.read_text(encoding="utf-8"))["per_grup"]
    estranyes = sorted({o for r in res for o in r["oberts"]["net"] if o != r["nom"]})
    sim = sorted({o for r in res for o in r["oberts"]["sim"]})
    fora = {"_que_es": "ANALISI POSTERIOR, NO FIRMADA (eines/model_reiniciat.py; documentacio/pla-placa-reinici-comes.md, punt 2): "
                       "el model reiniciat des de l'estat mesurat despres d'amollar, contra 200 bessons que passen pel mateix cami.",
            "t0_s": T0_S, "t_despres_s": T_DESPRES_S, "D_graus": D_GRAUS,
            "J": {"cada_presa_nomes_obre_la_seva": not estranyes, "altres_de_dades_net": estranyes, "dades_sim": sim,
                  "passa": not estranyes and not sim},
            "per_grup": {}}
    for g in e1.GRUPS_Y:
        rg = [r for r in res if r["grup"] == g]
        pg = {"per_tir": {r["nom"]: r for r in rg}, "per_t0": {}, "cap_on": {}}
        for t0 in T0_S:
            pg["per_t0"][clau(t0)] = {}
            for D in D_GRAUS:
                s = e1.resum_cap_on([r["per_t0"][clau(t0)]["per_D"][clau(D)]["percentil"] for r in rg])
                s["lectura"] = lectura(s)
                fr = [r["per_t0"][clau(t0)]["friccio_t_D_s"][clau(D)] for r in rg]
                tr = [r["per_t0"][clau(t0)]["per_D"][clau(D)]["t_real_s"] for r in rg]
                s["friccio_t_D_mediana_s"], s["t_real_mediana_s"] = float(np.median(fr)), float(np.median(tr))
                pg["per_t0"][clau(t0)][clau(D)] = s
        for D in D_GRAUS:
            c = capon[g]["per_D"][clau(D)]
            pg["cap_on"][clau(D)] = {"mediana": c["mediana"], "interval_95": c["interval_95"], "lectura": c["lectura"]}
        pg["contrast"] = contrast(pg["per_t0"][clau(T0_S[0])], [pg["per_t0"][clau(t)] for t in T0_S[1:]])
        fora["per_grup"][g] = pg
        for t0 in T0_S:
            for D in D_GRAUS:
                s = pg["per_t0"][clau(t0)][clau(D)]
                print(f"{g} graus, t0 = {t0:.2f} s, D = {D:>4g}: mediana del percentil {s['mediana']:.2f} (95 %: "
                      f"{s['interval_95'][0]:.2f}-{s['interval_95'][1]:.2f}) -> {s['lectura']}; T_D real (mediana) "
                      f"{s['t_real_mediana_s']:.2f} s, friccio {s['friccio_t_D_mediana_s']:.2f} s")
        for D in D_GRAUS:
            c = pg["cap_on"][clau(D)]
            print(f"{g} graus, des de l'amollada (cap_on), D = {D:>4g}: {c['mediana']:.2f} ({c['interval_95'][0]:.2f}-"
                  f"{c['interval_95'][1]:.2f}) -> {c['lectura']}")
        print(f"{g} graus: {pg['contrast'].upper()}")
    print(f"J: cada presa nomes obre la seva de dades/net/ ({estranyes or 'cap altra'}) i res de dades/sim/ ({sim or 'res'}): "
          f"{'PASSA' if fora['J']['passa'] else 'FALLA'}")
    SORTIDA.parent.mkdir(parents=True, exist_ok=True)
    SORTIDA.write_text(json.dumps(pv._json_net(fora), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {SORTIDA.relative_to(ARREL)}")
    dibuixa(fora)
    return 0


def dibuixa(fora: dict) -> None:
    """El percentil de cada tir des de l'amollada (cap_on) i reiniciat a t0 = 0,10 s, per grup i per D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pendol import estil
    estil.aplica()
    capon = json.loads(CAP_ON.read_text(encoding="utf-8"))["per_grup"]
    Ds = (5.0, 10.0, 20.0)
    fig, eixos = plt.subplots(len(e1.GRUPS_Y), len(Ds), figsize=(12, 6.4), sharey=True)
    for r, g in enumerate(e1.GRUPS_Y):
        pg = fora["per_grup"][g]
        noms = sorted(pg["per_tir"], key=lambda n: int(n.split("_")[1]))
        x = np.arange(len(noms))
        for c, D in enumerate(Ds):
            ax = eixos[r, c]
            k = clau(D)
            ab = [capon[g]["per_tir"][n][k]["percentil"] for n in noms]
            re = [pg["per_tir"][n]["per_t0"][clau(T0_S[0])]["per_D"][k]["percentil"] for n in noms]
            for xi, a, b in zip(x, ab, re):
                ax.plot([xi, xi], [a, b], color=estil.EIX, lw=0.8, zorder=1)
            ax.plot(x, ab, "o", mfc="none", mec=estil.TINTA_SUAU, ms=5, label="des de l'amollada", zorder=2)
            ax.plot(x, re, "o", color=estil.REAL, ms=5, label="reiniciat a 0,10 s", zorder=3)
            sa, sr = pg["cap_on"][k], pg["per_t0"][clau(T0_S[0])][k]
            ax.axhline(0.5, color=estil.TINTA_SUAU, lw=0.7)
            ax.axhspan(*sa["interval_95"], color=estil.TINTA_SUAU, alpha=0.10, lw=0)
            ax.axhspan(*sr["interval_95"], color=estil.REAL, alpha=0.15, lw=0)
            ax.set_xticks(x, [n.split("_")[1] for n in noms], fontsize=7.5)
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(f"{g}°, D = {D:g}°", loc="left", fontsize=10)
            if r == len(e1.GRUPS_Y) - 1:
                ax.set_xlabel("presa")
        eixos[r, 0].set_ylabel("percentil entre els bessons")
    eixos[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    estil.desa(fig, "experiment1_model_reiniciat", directori=FIGURES)


# ------------------------------------------------------------ l'autoprova (RE1-RE4; la J, a l'execucio real)

def autoprova() -> int:
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova d'eines/model_reiniciat.py (criteris de documentacio/pla-placa-reinici-comes.md, punt 2; sense cap presa real)")
    p = Parametres.des_de_json(e1.PARAMETRES)
    estat0 = np.array([math.radians(90.0), math.radians(90.0), 0.0, 0.0])
    dt = 1.0 / 240.0
    ts = np.arange(0.0, 3.5, dt)
    _, y = simula(p, estat0, t_eval=ts)

    # RE1: l'estat del Savitzky-Golay a t0 = 0,10 s contra l'exacte, sense soroll.
    i0 = int(np.nonzero(ts >= 0.10)[0][0])
    e = estat_sg(y[0], y[1], i0, dt)
    dth = math.degrees(max(abs(e[0] - y[0, i0]), abs(e[1] - y[1, i0])))
    dom = max(abs(e[2] - y[2, i0]), abs(e[3] - y[3, i0]))
    mira(dth < 0.01 and dom < 0.05, f"RE1 l'estat a t0 = 0,10 s sense soroll: |dtheta| {dth:.1e} graus (< 0,01), |domega| "
                                    f"{dom:.1e} rad/s (< 0,05)")

    # RE2: des de l'estat exacte, el reinici reprodueix la trajectoria.
    fi = int(np.searchsorted(ts, ts[i0] + T_DESPRES_S, side="right"))
    _, z = simula(p, y[:, i0], t_eval=ts[i0:fi] - ts[i0])
    err = float(np.max(np.abs(z[:2] - y[:2, i0:fi])))
    mira(err < 1e-6, f"RE2 el reinici des de l'estat exacte: max |dtheta| {err:.1e} rad en {T_DESPRES_S:g} s (< 1e-6)")

    # RE3: la serie mesurada d'un besso, pel cami de la presa real (amb repos abans del zero), dona la mateixa T_D que pel dels
    # bessons. Un feix i un soroll fabricats (els d'experiment1 amb parametres de prova; cap fitxer de dades/).
    nb = 10
    tg, A1, A2 = e1.feix(p, np.array([math.radians(91.0), math.radians(90.0), 0.0, 0.0]), "90_1", "90", t_fins=T_FEIX_S)
    N = e1.soroll(p, "90_1", 13, [0.3, 0.6], [0.5, 0.7], [0.97, 0.98], 1.0 / dt, T_FEIX_S)
    tsr = np.arange(0.0, T_FEIX_S, dt)
    b1 = np.interp(tsr, tg, A1[1]) + N[0][1, :len(tsr)]
    b2 = np.interp(tsr, tg, A2[1]) + N[1][1, :len(tsr)]
    rep = np.arange(-60, 0) * dt
    q = {"t": np.concatenate([rep, tsr]), "th1": np.concatenate([np.full(60, b1[0]), b1]),
         "th2": np.concatenate([np.full(60, b2[0]), b2]), "zero": 0.0}
    res = analitza(q, tg, A1, A2, N, p, "90", n_bessons=nb)
    iguals = all(res[k]["per_D"][d]["t_real_s"] == res[k]["per_D"][d]["t_bessons"][0] for k in res for d in res[k]["per_D"])
    mira(iguals, "RE3 el besso 1 pel cami de la presa real i pel dels bessons: la mateixa T_D a les tres t0 i les tres D ("
                 + "; ".join(f"t0 {k}: {res[k]['per_D']['10']['t_real_s']:.4f} s" for k in res) + ", D = 10)")

    # RE4: reproduible.
    res2 = analitza(q, tg, A1, A2, N, p, "90", n_bessons=nb)
    mira(json.dumps(res, sort_keys=True) == json.dumps(res2, sort_keys=True), "RE4 dues passades del cas fabricat: bit a bit")
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    ap.add_argument("--processos", type=int, default=1, help="com a molt 3")
    a = ap.parse_args(argv)
    if a.autoprova:
        return autoprova()
    return ordre(e1.llegeix_firma(), min(max(a.processos, 1), 3))


if __name__ == "__main__":
    raise SystemExit(main())
