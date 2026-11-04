r"""La prova de la placa contra l'articulacio (29/10/2026). ANALISI POSTERIOR, cap veredicte: va a la discussio com a indicacio.

Puja el 29/10 (documentacio/pla-placa-reinici-comes.md, punt 1) d'esborranys/_angle_entre_barres.py, amb les mateixes operacions, i
s'hi afegeixen els 150 graus (el punt 3). Els criteris dels 45 i els 90: documentacio/pla-figura-i-eines-experiment1.md, pas 4.

phi = theta2 - theta1, fotograma a fotograma i SENSE el Savitzky-Golay, amb el centre automatic (eines/centre_automatic.py) i el
Tracker al costat, de ta - 0,4 s a ta + 0,12 s. Els angles, pel cami de la conversio (el pivot, la delta, la vertical, a_angles),
sense la correccio de corregeix_presa. Es dibuixa phi' = s*(phi - phi_repos) contra dth1 = s*(theta1_repos - theta1), amb s el signe
de theta1 (dth1 > 0 quan cau; el model dona phi' > 0). Al costat, el model des de les condicions inicials, sense friccio i amb la
firmada. sigma: la desviacio de phi al repos [ta - 0,4; ta - 0,05].

Classificacio (al primer fotograma amb dth1 >= 1 grau, i als d'abans des de ta - 0,05): «placa» si phi' < -3 sigma en algun
fotograma d'abans; «encallada» si |phi'| < 3 sigma quan el model ja dona mes de 3 sigma; «model» si |phi' - phi'_model| < 3 sigma;
«cap». Si en compleix mes d'una: placa > encallada > model. El grup s'inclina per una classe si en son >= 2/3 de les preses.

Els 150 graus: les condicions inicials del model, de comparar.corregeix_presa (no hi ha clics), i el centre automatic nomes si la
presa passa M (k = 0, el minim estricte de la distancia al Tracker) i Q (cap fotograma sense centre, >= 99 % a 13 px del Tracker);
si no, «sense dades». Als 45 i als 90, M i Q ja es van comprovar a l'experiment 1 amb el centre automatic.

A mes: la tira de fotogrames de la placa per presa, i el parell que caldria a l'articulacio per mantenir les barres rectes.
Necessita els videos (dades/video/conv/), que no son al repositori, i l'ffmpeg.

    .venv\Scripts\python.exe eines\angle_entre_barres.py --autoprova
    .venv\Scripts\python.exe eines\angle_entre_barres.py [--placa]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
import centre_automatic as ca  # noqa: E402
import experiment1 as e1  # noqa: E402
from pendol import comparar, dades, fisica, tracker, trk  # noqa: E402
from pendol.fisica import Dissipacio, Parametres  # noqa: E402
from pendol.simulador import simula  # noqa: E402

SORTIDA = ARREL / "dades" / "experiment1" / "angle_entre_barres_2026-10-29.json"
FIGURES = ARREL / "figures"
GRUPS = ("45", "90", "150")
N_150 = 12
FINESTRA_S = (-0.4, 0.12)
REPOS_S = (-0.4, -0.05)
D1_GRAUS, NSIGMA = 1.0, 3.0
SENSE_AUTOMATIC = {"45_11"}              # no te gomet al rodament: nomes el Tracker
CLASSES = ("placa", "encallada", "model", "cap", "sense dades")


def angles_crus(k, A, G, Pf, meta):
    """Els angles de la conversio, sense el Savitzky-Golay (ni el remostreig): als instants del .trk."""
    geo = meta["geometria"]
    P = np.broadcast_to(np.median(Pf, axis=0), (len(k), 2))
    A = np.asarray(A, float)
    if geo["marca_articulacio"].get("corregida"):
        A = tracker.corregeix_marca(P, A, geo["marca_articulacio"]["delta_px"])
    J, Gg = tracker.adreca_i_corregeix(P, A, np.asarray(G, float), geo, False)
    t = (np.asarray(k) - k[0]) * float(meta["delta_t_s"])
    x1, y1, x2, y2 = tracker._reomple_forats(t, J[:, 0], J[:, 1], Gg[:, 0], Gg[:, 1])
    th1, th2 = tracker.a_angles(t, x1, y1, x2, y2, (0.0, 0.0), True, bool(meta["signe_invertit"]))
    return t, np.asarray(th1), np.asarray(th2)


def corba_model(p, estat0, dis=None):
    """phi' contra dth1 del model, en graus, des del repos (amb s, el signe de theta1)."""
    tg = np.arange(0.0, 0.2, 1e-4)
    _, y = simula(p, estat0, t_eval=tg, dis=dis)
    s = math.copysign(1.0, estat0[0])
    d1 = np.degrees(s * (estat0[0] - y[0]))
    ph = np.degrees(s * ((y[1] - y[0]) - (estat0[1] - estat0[0])))
    return d1, ph


def llegeix(nom: str, retalls: dict):
    g = nom.split("_")[0]
    d = trk.retalla(trk.llegeix_trk(ARREL / "dades" / "video" / "tracker" / g / f"{nom}.trk"),
                    retalls.get(nom, {}).get("des_del_fotograma"))
    k, A, G, _noms, (_kf, Pf) = trk.tres_pistes(d)
    _taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{nom}.csv")
    return np.asarray(k), np.asarray(A, float), np.asarray(G, float), Pf, meta


def classifica(t, th1, th2, ta, model):
    rep = (t >= ta + REPOS_S[0]) & (t <= ta + REPOS_S[1])
    phi = th2 - th1
    phi_r, sig, th1_r = float(np.mean(phi[rep])), float(np.std(phi[rep], ddof=1)), float(np.mean(th1[rep]))
    s = math.copysign(1.0, th1_r)
    d1 = np.degrees(s * (th1_r - th1))
    ph = np.degrees(s * (phi - phi_r))
    sig = math.degrees(sig)
    i0 = int(np.nonzero(t >= ta + REPOS_S[1])[0][0])
    j = np.nonzero((np.arange(len(t)) >= i0) & (d1 >= D1_GRAUS))[0]
    if not len(j):
        return {"problema": "theta1 no cau 1 grau a la finestra"}
    j1 = int(j[0])
    md1, mph = model
    ph_m = float(np.interp(d1[j1], md1, mph))
    placa = bool(np.any(ph[i0:j1] < -NSIGMA * sig))
    encallada = bool(abs(ph[j1]) < NSIGMA * sig and ph_m > NSIGMA * sig)
    segueix = bool(abs(ph[j1] - ph_m) < NSIGMA * sig)
    classe = "placa" if placa else "encallada" if encallada else "model" if segueix else "cap"
    fin = (t >= ta + REPOS_S[1]) & (t <= ta + FINESTRA_S[1])
    return {"sigma_graus": sig, "dth1_j1_graus": float(d1[j1]), "t_j1_menys_ta_s": float(t[j1] - ta),
            "phi_j1_graus": float(ph[j1]), "phi_model_j1_graus": ph_m, "min_phi_abans_graus": float(np.min(ph[i0:j1 + 1])),
            "condicions": {"placa": placa, "encallada": encallada, "model": segueix}, "classe": classe,
            "corba": {"dth1": d1[fin].tolist(), "phi": ph[fin].tolist()}}


def condicions_corregides(nom: str) -> np.ndarray:
    """Als 150 graus, que no tenen clics: la mitjana de [ta - 0,4; ta - 0,02] dels angles de comparar.corregeix_presa (com
    experiment1.presa_corregida, sense la comprovacio dels clics)."""
    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{nom}.csv")
    th1, th2, _info = comparar.corregeix_presa(taula, meta, nom)
    t = taula["t"].to_numpy(dtype=float)
    ta = float(meta["amollada"]["amollada_s"])
    ini = (t >= max(0.0, ta - 0.4)) & (t <= ta - 0.02)
    return np.array([float(np.mean(th1[ini])), float(np.mean(th2[ini])), 0.0, 0.0])


def mapatge(nom: str, k, POS: dict, t, ta: float) -> dict:
    """M (com centre_automatic.una_presa): la distancia mediana del centre automatic al Tracker, als dos gomets, a ta + 0,25 i
    + 0,45 s, amb el video desplacat de -3 a +3 fotogrames. Passa si k = 0 es el minim estricte (la M relaxada)."""
    llavor = lambda Q, i: Q[i] if np.all(np.isfinite(Q[i])) else np.array(   # noqa: E731
        [np.interp(i, np.nonzero(np.isfinite(Q[:, c]))[0], Q[np.isfinite(Q[:, c]), c]) for c in range(2)])
    iM = [int(np.argmin(np.abs(t - (ta + s)))) for s in ca.M_S]
    feines = {}
    for i in iM:
        for dk in ca.OFFSETS:
            for gm in POS:
                feines.setdefault(int(k[i]) + dk, []).append(((i, dk, gm), llavor(POS[gm], i)))
    cmp_ = ca.centres(ca.video_de(nom), feines)
    med = {dk: float(np.median([99.0 if not np.all(np.isfinite(cmp_[(i, dk, gm)])) else
                                float(np.hypot(*(np.array(cmp_[(i, dk, gm)]) - llavor(POS[gm], i)))) for i in iM for gm in POS]))
           for dk in ca.OFFSETS}
    ordre = sorted(med, key=med.get)
    return {"k": ordre[0], "mediana_px": med, "passa": ordre[0] == 0 and med[ordre[0]] < med[ordre[1]]}


def una_presa(nom: str, p: Parametres, retalls: dict, clics: dict) -> dict:
    g = nom.split("_")[0]
    k, A, G, Pf, meta = llegeix(nom, retalls)
    ta = float(meta["amollada"]["amollada_s"])
    t, th1, th2 = angles_crus(k, A, G, Pf, meta)
    if g == "150":
        estat0 = condicions_corregides(nom)
    else:
        c = clics[nom]["amollada_amb_clics_graus"]
        estat0 = np.array([math.radians(c[0]), math.radians(c[1]), 0.0, 0.0])
    models = {"sense": corba_model(p, estat0), "firmada": corba_model(p, estat0, Dissipacio.firmada(float(g)))}
    fora = {"nom": nom, "grup": g, "model": {q: {"dth1": m[0][::20].tolist(), "phi": m[1][::20].tolist()} for q, m in models.items()},
            "tracker": classifica(t, th1, th2, ta, models["sense"])}
    if g == "150":
        fora["condicions_inicials_graus"] = [math.degrees(estat0[0]), math.degrees(estat0[1])]
    if nom in SENSE_AUTOMATIC:
        return fora
    if g == "150":
        fora["M"] = mapatge(nom, k, {"rodament": A, "extrem": G}, t, ta)
        if not fora["M"]["passa"]:
            fora["automatic"] = {"problema": "no passa M (el mapatge dels fotogrames)"}
            return fora
    iW = np.nonzero((t >= ta + FINESTRA_S[0] - 0.01) & (t <= ta + FINESTRA_S[1] + 0.01))[0]
    feines = {}
    for gm, Q in (("rodament", A), ("extrem", G)):
        for i in iW:
            if np.all(np.isfinite(Q[i])):
                feines.setdefault(int(k[i]), []).append(((int(i), gm), Q[i]))
    cw = ca.centres(ca.video_de(nom), feines)
    A2, G2 = A.copy(), G.copy()
    for (i, gm), xy in cw.items():
        (A2 if gm == "rodament" else G2)[i] = xy
    if g == "150":
        dist = np.array([float(np.hypot(*(np.array(xy) - (A if gm == "rodament" else G)[i]))) if np.all(np.isfinite(xy))
                         else math.inf for (i, gm), xy in cw.items()])
        fora["Q"] = {"centres": int(len(dist)), "sense_centre": int(np.sum(~np.isfinite(dist))),
                     "fraccio_a_13px": float(np.mean(dist <= ca.Q_PX)),
                     "mediana_px": float(np.median(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else math.nan}
        fora["Q"]["passa"] = fora["Q"]["sense_centre"] == 0 and fora["Q"]["fraccio_a_13px"] >= ca.Q_FRACCIO
        if not fora["Q"]["passa"]:
            fora["automatic"] = {"problema": "no passa Q (el seguiment del centre automatic)"}
            return fora
    t, n1, n2 = angles_crus(k, A2, G2, Pf, meta)
    fora["automatic"] = classifica(t, n1, n2, ta, models["sense"])
    return fora


def parell_articulacio(p: Parametres, angles=(45.0, 90.0, 150.0)) -> dict:
    """El parell de la barra 1 sobre la 2 que mantindria les barres rectes en amollar (theta1 = theta2 = theta0, omega = 0), i les
    acceleracions de l'amollada lliure."""
    fora = {}
    for th0 in angles:
        s = math.sin(math.radians(th0))
        a_r = -p.g * (p.A + p.B) * s / (p.I1 + p.I2 + 2 * p.K)
        tau = (p.I2 + p.K) * a_r + p.g * p.B * s
        M = np.array([[p.I1, p.K], [p.K, p.I2]])
        a1, a2 = np.linalg.solve(M, [-p.g * p.A * s, -p.g * p.B * s])
        tf2 = Dissipacio.firmada(th0).tau_f2
        fora[f"{th0:g}"] = {"alfa_rigid_rad_s2": a_r, "parell_N_m": tau, "tau_f2_firmat_N_m": tf2, "vegades_tau_f2": abs(tau) / tf2,
                            "alfa1_lliure": float(a1), "alfa2_lliure": float(a2)}
    return fora


def parell_per_fraccio(p: Parametres, th0: float, r: float) -> float:
    """Afegit DESPRES DE VEURE ELS 150 GRAUS (pla-placa-reinici-comes.md): el parell constant a l'articulacio (sobre la barra 2, i
    el contrari sobre la 1) amb que, en amollar de theta1 = theta2 = th0 amb omega = 0, phi' / dth1 es r vegades el de l'amollada
    lliure. Les acceleracions son lineals en el parell: alfa = a + b tau. r = 0 es el parell que les mante rectes; r = 1, cap."""
    s = math.sin(math.radians(th0))
    M = np.array([[p.I1, p.K], [p.K, p.I2]])
    a = np.linalg.solve(M, [-p.g * p.A * s, -p.g * p.B * s])
    b = np.linalg.solve(M, [-1.0, 1.0])
    R0 = (a[1] - a[0]) / (-a[0])
    return float((-r * R0 * a[0] - a[1] + a[0]) / ((b[1] - b[0]) + r * R0 * b[0]))


def frenada(res: list, p: Parametres) -> dict:
    """Per grup: la mediana de phi' / phi'_model al primer fotograma amb dth1 >= 1 grau (les preses classificades), i el parell
    que la donaria, en vegades el tau_f2 firmat."""
    fora = {}
    for g in GRUPS:
        xs = [r["automatic" if "automatic" in r else "tracker"] for r in res if r["grup"] == g]
        q = [x["phi_j1_graus"] / x["phi_model_j1_graus"] for x in xs if "classe" in x]
        rm = float(np.median(q))
        tau = parell_per_fraccio(p, float(g), rm)
        fora[g] = {"n": len(q), "fraccio_mediana": rm, "fraccio_min_max": [float(min(q)), float(max(q))], "parell_N_m": tau,
                   "vegades_tau_f2": abs(tau) / Dissipacio.firmada(float(g)).tau_f2}
    return fora


def preses(g: str) -> list[str]:
    if g in e1.GRUPS_Y:
        return e1.preses_del_grup(g)
    noms = json.loads(e1.NUMEROS.read_text(encoding="utf-8"))["B_condicions_inicials"]["per_presa"]
    fora = sorted((n for n in noms if n.split("_")[0] == g), key=lambda n: int(n.split("_")[1]))
    if len(fora) != N_150:
        raise SystemExit(f"Al grup {g} hi ha {len(fora)} preses, i se n'esperaven {N_150}.")
    return fora


def tira_placa(nom: str, retalls: dict, zero_s: float, instants_ms=(-8, 0, 8, 17, 25, 33, 42, 50)):
    k, A, G, Pf, meta = llegeix(nom, retalls)
    dt = float(meta["delta_t_s"])
    n0 = int(k[0]) + int(round(zero_s / dt))
    ns = [n0 + int(round(ms / 1000.0 / dt)) for ms in instants_ms]
    i = [int(np.argmin(np.abs(k - n))) for n in ns]
    P = np.median(Pf, axis=0)
    pts = np.vstack([P[None, :], A[i], G[i]])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    x0, x1 = int(max(0, pts[:, 0].min() - 60)), int(min(ca.W, pts[:, 0].max() + 60))
    y0, y1 = int(max(0, pts[:, 1].min() - 60)), int(min(ca.H, pts[:, 1].max() + 160))
    trams = "+".join(f"eq(n\\,{n})" for n in sorted(set(ns)))
    talls = {}
    with ca._ffmpeg(ca.VIDEOS_PILOT.get(nom, ca.video_de(nom)), trams) as proc:   # la 45_11 es diu _no_gomet
        for n in sorted(set(ns)):
            buf = proc.stdout.read(ca.W * ca.H * 3)
            img = np.frombuffer(buf, np.uint8).reshape(ca.H, ca.W, 3)
            talls[n] = img[y0:y1, x0:x1].copy()
    return [(ms, talls[n]) for ms, n in zip(instants_ms, ns)]


def ordre_placa(retalls: dict) -> int:
    """Les tires de fotogrames: als 45 i als 90, des del zero afinat de la Y firmada; als 150, des de l'amollada del video."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = json.loads(e1.Y_FIRMADA.read_text(encoding="utf-8"))["per_grup"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    for g in GRUPS:
        if g in y:
            zeros = {n: y[g]["per_tir"][n]["zero_s"] for n in y[g]["per_tir"]}
        else:
            zeros = {n: float(dades.carrega(ARREL / "dades" / "net" / f"{n}.csv")[1]["amollada"]["amollada_s"]) for n in preses(g)}
        noms = sorted(zeros, key=lambda n: int(n.split("_")[1]))
        tires = {n: tira_placa(n, retalls, zeros[n]) for n in noms}
        fig, eixos = plt.subplots(len(noms), 8, figsize=(8 * 2.6, len(noms) * 1.5), squeeze=False)
        for r, n in enumerate(noms):
            for c, (ms, im) in enumerate(tires[n]):
                ax = eixos[r, c]
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{n}  {ms:+d} ms", fontsize=7)
        fig.tight_layout()
        cami = FIGURES / f"experiment1_placa_{g}.png"
        fig.savefig(cami, dpi=90)
        plt.close(fig)
        print(f"-> {cami.relative_to(ARREL)}")
    return 0


def dibuixa(res: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pendol import estil
    estil.aplica()
    fig, eixos = plt.subplots(1, len(GRUPS), figsize=(13.5, 4.4), sharey=True)
    for ax, g in zip(eixos, GRUPS):
        rg = [r for r in res if r["grup"] == g]
        xs = [r["automatic" if "automatic" in r else "tracker"] for r in rg]
        sig = np.median([x["sigma_graus"] for x in xs if "sigma_graus" in x])
        ax.axhspan(-NSIGMA * sig, NSIGMA * sig, color=estil.GRAELLA, lw=0, zorder=0)
        for x in xs:
            if "corba" in x:
                ax.plot(x["corba"]["dth1"], x["corba"]["phi"], ".-", color=estil.REAL, ms=2.5, lw=0.7, alpha=0.6)
        m = rg[0]["model"]
        ax.plot(m["sense"]["dth1"], m["sense"]["phi"], color=estil.SIMULACIO, ls=estil.TRAC_SIMULACIO, lw=1.6,
                label=f"model ({rg[0]['nom']}), sense fricció")
        ax.plot(m["firmada"]["dth1"], m["firmada"]["phi"], color=estil.SIMULACIO, ls=":", lw=1.4, label="model, fricció firmada")
        ax.plot([], [], ".-", color=estil.REAL, label="tirs (centre automàtic, sense filtre)")
        ax.axvline(D1_GRAUS, color=estil.TINTA_SUAU, lw=0.7)
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-3, 7)
        ax.set_title(f"{g}°", loc="left", fontsize=10)
        ax.set_xlabel("Δθ1, el que ha caigut θ1 (°)")
    eixos[0].set_ylabel("φ − φ_repòs, amb el signe del model (°)")
    eixos[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    estil.desa(fig, "experiment1_angle_entre_barres", directori=FIGURES)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    ap.add_argument("--placa", action="store_true", help="nomes les tires de fotogrames de la placa")
    a = ap.parse_args(argv)
    if a.autoprova:
        return autoprova()
    e1.pany(e1.llegeix_firma(), 5.0, e1.GRUPS_Y)
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    if a.placa:
        return ordre_placa(retalls)
    p = Parametres.des_de_json(e1.PARAMETRES)
    clics = json.loads(comparar.CLICS.read_text(encoding="utf-8"))["preses"]
    res = []
    for g in GRUPS:
        for n in preses(g):
            r = una_presa(n, p, retalls, clics)
            res.append(r)
            for qui in ("automatic", "tracker"):
                if qui in r and "classe" in r[qui]:
                    x = r[qui]
                    print(f"  {n:6} {qui:9} sigma {x['sigma_graus']:.3f}; a dth1 = {x['dth1_j1_graus']:.2f} graus "
                          f"(ta {x['t_j1_menys_ta_s'] * 1000:+.0f} ms): phi' {x['phi_j1_graus']:+.2f}, model "
                          f"{x['phi_model_j1_graus']:+.2f}; minim abans {x['min_phi_abans_graus']:+.2f} -> {x['classe'].upper()}",
                          flush=True)
                elif qui in r:
                    print(f"  {n:6} {qui:9} {r[qui]['problema']}", flush=True)
    resum = {}
    for g in GRUPS:
        cl = [r["automatic" if "automatic" in r else "tracker"].get("classe", "sense dades") for r in res if r["grup"] == g]
        cl_t = [r["tracker"].get("classe", "sense dades") for r in res if r["grup"] == g]
        compte = {c: cl.count(c) for c in CLASSES}
        maj = [c for c, v in compte.items() if v >= math.ceil(2 * len(cl) / 3)]
        resum[g] = {"centre_automatic": compte, "tracker": {c: cl_t.count(c) for c in CLASSES},
                    "lectura": maj[0] if maj else "no es pot separar", "n": len(cl)}
        print(f"{g} graus: {compte} (Tracker: {resum[g]['tracker']}) -> {resum[g]['lectura']}")
    tq = parell_articulacio(p)
    for th0, v in tq.items():
        print(f"parell per mantenir-les rectes a {th0} graus: {v['parell_N_m']:.2e} N m = {v['vegades_tau_f2']:.0f} vegades tau_f2 "
              f"firmat; alfa lliure {v['alfa1_lliure']:.1f} / {v['alfa2_lliure']:.1f}, rigid {v['alfa_rigid_rad_s2']:.1f} rad/s2")
    fr = frenada(res, p)
    for g, v in fr.items():
        print(f"{g} graus: phi' / model = {v['fraccio_mediana']:.2f} (mediana de {v['n']}; {v['fraccio_min_max'][0]:.2f}-"
              f"{v['fraccio_min_max'][1]:.2f}); el parell que la donaria, {v['parell_N_m']:.2e} N m = {v['vegades_tau_f2']:.0f} "
              f"vegades tau_f2 firmat")
    SORTIDA.parent.mkdir(parents=True, exist_ok=True)
    SORTIDA.write_text(json.dumps({"_que_es": "La prova de la placa contra l'articulacio (ANALISI POSTERIOR, cap veredicte; "
                                              "eines/angle_entre_barres.py, documentacio/pla-placa-reinici-comes.md).",
                                   "per_presa": res, "resum": resum, "parell": tq,
                                   "frenada": {"_que_es": "afegit despres de veure els 150 graus", **fr}},
                                  indent=1, ensure_ascii=False),
                       encoding="utf-8")
    print(f"-> {SORTIDA.relative_to(ARREL)}")
    dibuixa(res)
    return 0


# ------------------------------------------------------------ l'autoprova (AB1-AB3 i J; la AB0, a l'esborrany)

def autoprova() -> int:
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova d'eines/angle_entre_barres.py (criteris de documentacio/pla-placa-reinici-comes.md; la AB0, a "
          "esborranys/_comprova_angle_entre_barres.py)")

    # AB1: el classificador sobre series fabricades. Repos de 0,5 s amb un soroll determinista de desviacio sigma (mai mes enlla
    # de 1,5 sigma), i des de ta una caiguda de theta1 de 6 graus en 0,1 s. El model, phi' = 1,045 dth1.
    dt, ta, sig = 1.0 / 240.0, 0.5, 0.05
    t = np.arange(0.0, 0.7, dt)
    n = np.arange(len(t))
    soroll = math.radians(sig) * math.sqrt(2.0) * np.sin(2.0 * math.pi * n / 7.0)
    model = (np.linspace(0.0, 6.0, 601), 1.045 * np.linspace(0.0, 6.0, 601))
    cau = np.where(t > ta, 0.5 * (2.0 * 6.0 / 0.1**2) * (t - ta) ** 2, 0.0)          # dth1 en graus

    def serie(phi_de_d1, s):
        th1r, phir = s * math.radians(90.0), math.radians(-0.3)
        th1 = th1r - s * np.radians(cau)
        th2 = th1 + phir + s * np.radians(phi_de_d1(cau)) + soroll
        return classifica(t, th1, th2, ta, model)

    casos = {"model": (lambda d: 1.045 * d, 1.0), "encallada": (lambda d: 0.0 * d, 1.0),
             "placa": (lambda d: np.where((d > 0.2) & (d < 0.8), -10.0 * sig, 1.045 * d), -1.0),
             "cap": (lambda d: 3.0 * 1.045 * d, 1.0)}
    surt = {k: serie(f, s).get("classe") for k, (f, s) in casos.items()}
    mira(all(surt[k] == k for k in casos), f"AB1 el classificador: {surt}")

    # AB2: el parell i les acceleracions, contra fisica.derivades.
    p = Parametres.des_de_json(e1.PARAMETRES)
    tq = parell_articulacio(p)
    pitjor_lliure, pitjor_rigid = 0.0, 0.0
    for th0, v in tq.items():
        th = math.radians(float(th0))
        d = fisica.derivades(0.0, np.array([th, th, 0.0, 0.0]), p)
        pitjor_lliure = max(pitjor_lliure, abs(v["alfa1_lliure"] - d[2]) / abs(d[2]), abs(v["alfa2_lliure"] - d[3]) / abs(d[3]))
        s, a, tau = math.sin(th), v["alfa_rigid_rad_s2"], v["parell_N_m"]
        r1 = (p.I1 + p.K) * a + p.g * p.A * s + tau
        r2 = (p.K + p.I2) * a + p.g * p.B * s - tau
        pitjor_rigid = max(pitjor_rigid, abs(r1) / (p.g * p.A * abs(s)), abs(r2) / (p.g * p.A * abs(s)))
    mira(pitjor_lliure < 1e-12 and pitjor_rigid < 1e-12,
         f"AB2 les acceleracions lliures contra fisica.derivades: {pitjor_lliure:.1e} relatiu; el rigid amb el parell, residu "
         f"{pitjor_rigid:.1e} (a 45, 90 i 150 graus)")

    # AB4 (afegida despres de veure els 150 graus): el parell per a una fraccio r. Amb r = 1, cap; amb r = 0, el que les mante
    # rectes; i amb r = 0,5, les acceleracions que dona compleixen phi' / dth1 = 0,5 del lliure.
    pitjor = 0.0
    for th0, v in tq.items():
        th = float(th0)
        t1, t0_, th_ = parell_per_fraccio(p, th, 1.0), parell_per_fraccio(p, th, 0.0), parell_per_fraccio(p, th, 0.5)
        s = math.sin(math.radians(th))
        M = np.array([[p.I1, p.K], [p.K, p.I2]])
        al = np.linalg.solve(M, [-p.g * p.A * s - th_, -p.g * p.B * s + th_])
        R0 = (v["alfa2_lliure"] - v["alfa1_lliure"]) / (-v["alfa1_lliure"])
        pitjor = max(pitjor, abs(t1) / abs(v["parell_N_m"]), abs(t0_ - v["parell_N_m"]) / abs(v["parell_N_m"]),
                     abs((al[1] - al[0]) / (-al[0]) - 0.5 * R0) / R0)
    mira(pitjor < 1e-12, f"AB4 el parell per a una fraccio: r = 1 -> cap, r = 0 -> el rigid, r = 0,5 -> la meitat (pitjor {pitjor:.1e})")

    # AB3: els angles sense filtre, passats pel Savitzky-Golay, son els de centre_automatic.angles (90_6).
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    k, A, G, Pf, meta = llegeix("90_6", retalls)
    _t, c1, c2 = angles_crus(k, A, G, Pf, meta)
    sg = meta["savgol"]
    r1, r2 = ca.angles(k, A, G, Pf, meta)
    dif = (max(float(np.max(np.abs(savgol_filter(c1, sg["finestra_mostres"], sg["ordre"]) - r1))),
               float(np.max(np.abs(savgol_filter(c2, sg["finestra_mostres"], sg["ordre"]) - r2))))
           if len(c1) == len(r1) else math.inf)
    mira(dif < 1e-12, f"AB3 a la 90_6, Savitzky-Golay dels angles sense filtre contra centre_automatic.angles: {dif:.1e} rad")

    mirant[0] = False
    dolents = [o for o in oberts if ("net" in Path(o).parts and Path(o).stem != "90_6") or "sim" in Path(o).parts]
    mira(not dolents, f"J de dades/net/ nomes la 90_6, i res de dades/sim/: {dolents or 'cap altre'}")
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


if __name__ == "__main__":
    raise SystemExit(main())
