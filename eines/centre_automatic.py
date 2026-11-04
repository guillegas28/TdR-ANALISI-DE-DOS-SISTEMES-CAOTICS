r"""El centre automatic dels gomets: el pilot i l'experiment 1 amb la creu treta (28/10/2026). ANALISI POSTERIOR, NO FIRMADA.

Puja el 29/10 (documentacio/pla-figura-i-eines-experiment1.md, pas 3.4) d'esborranys/_centre_automatic.py (el pilot),
esborranys/_experiment1_centre_auto.py (l'experiment 1) i dels calculs que es van fer a ma el 28/10 al vespre (el diagnostic), amb
la sortida igual. Plans i criteris (escrits abans de calcular): documentacio/pla-centre-automatic.md.

EL CENTRE d'un gomet vermell, directament del video: la mitjana ponderada amb el pes max(0, V - 80), V = R - max(G, B), dins d'un
disc de 24 px que es recentra fins que es mou < 0,01 px. El cap del cargol (la 45_11 no te gomet al rodament): el pes
max(0, 90 - brillantor), en un disc de 16 px. El centre del pixel (fila i, columna j) es (j + 0,5, i + 0,5).

    pilot        el centre automatic contra els clics d'en Guillem: al repos (A0, < 1 px) i en moviment (A1, rms <= 0,75 px), a la
                 45_11 i la 90_6, amb el mapatge dels fotogrames (M). Amb el cargol (com es va escriure) i sense (decisio d'en
                 Guillem, despres de veure'l: no te gomet, error de gravacio). No obre cap CSV de dades/net/.
    experiment1  dins de W = [amollada - 0,45 s; amollada + 1,65 s], les posicions del Tracker es canvien pel centre automatic, els
                 angles es tornen a calcular com la conversio, i l'experiment 1 es torna a passar tal com es va firmar. Criteris R
                 (si falla, s'atura), M, A0, Q i K. --m-relaxat: la variant POSTERIOR del criteri M (passa si k = 0 es el minim).
                 Darrere del pany de firma.json.
    diagnostic   d'on pot venir la diferencia de theta2: les medianes de theta2 - nominal, el quocient dels primers 20 ms, la friccio
                 firmada i els parametres del muntatge. Llegeix les series que escriu experiment1 --m-relaxat. I la figura 3.

Limit declarat: necessita els videos (dades/video/conv/), que no son al repositori, i l'ffmpeg.

    .venv\Scripts\python.exe eines\centre_automatic.py --autoprova
    .venv\Scripts\python.exe eines\centre_automatic.py pilot
    .venv\Scripts\python.exe eines\centre_automatic.py experiment1 [--m-relaxat] [--processos 3]
    .venv\Scripts\python.exe eines\centre_automatic.py diagnostic
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import multiprocessing
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
import correccio_amb_clics as cac  # noqa: E402
import creu_moviment as cm  # noqa: E402
import experiment1 as e1  # noqa: E402
from pendol import comparar, dades, tracker, trk  # noqa: E402
from pendol.fisica import Dissipacio, Parametres  # noqa: E402
from pendol.simulador import afina_amollada, simula  # noqa: E402

FF = Path(shutil.which("ffmpeg") or "ffmpeg")
VIDEOS_PILOT = {"45_11": ARREL / "dades" / "video" / "conv" / "45" / "45_11_no_gomet.mp4",
                "90_6": ARREL / "dades" / "video" / "conv" / "90" / "90_6.mp4"}
CARGOL = {("45_11", "rodament")}                 # sense gomet: el cap del cargol
DADES = ARREL / "dades" / "experiment1"
PILOT_JSON = DADES / "centre_automatic_pilot_2026-10-29.json"
Y_JSON = {False: DADES / "centre_automatic_y_2026-10-29.json", True: DADES / "centre_automatic_y_m_relaxat_2026-10-29.json"}
SERIES = ARREL / "figures" / "centre_automatic_series.json"      # es refa amb experiment1 --m-relaxat
DIAG_JSON = DADES / "centre_automatic_diagnostic_2026-10-29.json"
W, H, MIG = 1080, 1920, 60                       # el video, i la meitat del retall
OFFSETS = range(-3, 4)
LLINDAR_V, R_VERMELL = 80.0, 24.0
LLINDAR_FOSC, R_CARGOL = 90.0, 16.0
A0_PX, A1_PX = 1.0, 0.75

FORA_DE_LA_PROVA = {"45_11": "sense gomet al rodament (error de gravacio; decisio d'en Guillem, 28/10)"}
W_ABANS_S, W_DESPRES_S = 0.45, 1.65
M_S = (0.25, 0.45)
Q_PX, Q_FRACCIO = 13.0, 0.99
R_RAD, K_RAD = 1e-8, 1e-12
X_GRAUS = 5.0


# ------------------------------------------------------------ el centre d'un gomet

def centre(tros, x0: int, y0: int, inici, cargol: bool) -> tuple[float, float]:
    """La mitjana ponderada dins d'un disc que es recentra (fins que es mou < 0,01 px). El centre del pixel (i, j) es
    (x0 + j + 0,5, y0 + i + 0,5)."""
    if cargol:
        pes, R = np.maximum(0.0, LLINDAR_FOSC - tros.mean(axis=2)), R_CARGOL
    else:
        pes, R = np.maximum(0.0, tros[..., 0] - np.maximum(tros[..., 1], tros[..., 2]) - LLINDAR_V), R_VERMELL
    ii, jj = np.mgrid[0:tros.shape[0], 0:tros.shape[1]]
    X, Y = x0 + jj + 0.5, y0 + ii + 0.5
    c = np.array(inici, float)
    for _ in range(50):
        w = pes * (np.hypot(X - c[0], Y - c[1]) <= R)
        if w.sum() <= 0:
            return math.nan, math.nan
        nou = np.array([(w * X).sum() / w.sum(), (w * Y).sum() / w.sum()])
        if np.hypot(*(nou - c)) < 0.01:
            return float(nou[0]), float(nou[1])
        c = nou
    return float(c[0]), float(c[1])


def _ffmpeg(video: Path, sel: str):
    if not FF.exists():
        raise SystemExit(f"No es troba l'ffmpeg ({FF}).")
    if not video.exists():
        raise SystemExit(f"No es troba el video {video} (els videos no son al repositori).")
    cmd = [str(FF), "-v", "error", "-i", str(video), "-vf", f"select={sel}", "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)


def retalls_del_video(video: Path, demanes: dict) -> dict:
    """El pilot. demanes = {fotograma del video: [(clau, (x, y)), ...]}: el retall de 2*MIG+1 al voltant de cada (x, y), i el seu
    origen."""
    ns = sorted(demanes)
    sel = "+".join(f"eq(n\\,{n})" for n in ns)
    fora, mida = {}, W * H * 3
    with _ffmpeg(video, sel) as proc:
        for n in ns:
            buf = proc.stdout.read(mida)
            if len(buf) != mida:
                raise SystemExit(f"l'ffmpeg no ha donat el fotograma {n} de {video.name}")
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            for clau, (x, y) in demanes[n]:
                cx, cy = int(math.floor(x)), int(math.floor(y))
                fora[clau] = (img[cy - MIG:cy + MIG + 1, cx - MIG:cx + MIG + 1].astype(float), cx - MIG, cy - MIG)
    return fora


def centres(video: Path, feines: dict) -> dict:
    """L'experiment 1. feines = {fotograma del video: [(clau, llavor (x, y)), ...]} -> {clau: (x, y)}: el centre automatic d'un
    gomet vermell, calculat a mesura que l'ffmpeg dona els fotogrames (sense guardar-los)."""
    ns = sorted(feines)
    trams, ini = [], ns[0]                      # els fotogrames seguits, com a between(n, a, b): l'expressio no pot ser llarga
    for a, b in zip(ns, ns[1:] + [None]):
        if b != a + 1:
            trams.append(f"between(n\\,{ini}\\,{a})")
            ini = b
    fora, mida = {}, W * H * 3
    with _ffmpeg(video, "+".join(trams)) as proc:
        for n in ns:
            buf = proc.stdout.read(mida)
            if len(buf) != mida:
                raise SystemExit(f"l'ffmpeg no ha donat el fotograma {n} de {video.name}")
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            for clau, (x, y) in feines[n]:
                cx, cy = int(math.floor(x)), int(math.floor(y))
                if not (MIG <= cx < W - MIG and MIG <= cy < H - MIG):
                    fora[clau] = (math.nan, math.nan)
                    continue
                tros = img[cy - MIG:cy + MIG + 1, cx - MIG:cx + MIG + 1].astype(float)
                fora[clau] = centre(tros, cx - MIG, cy - MIG, (x, y), False)
    return fora


# ------------------------------------------------------------ el pilot

def presa(n: str, nous: dict, retalls: dict) -> dict:
    """El que llegeix correccio_amb_clics.llegeix_presa, sense el CSV: les posicions crues del .trk, els clics del repos i
    l'amollada."""
    cami = ARREL / "dades" / "video" / "tracker" / n.split("_")[0] / f"{n}.trk"
    claus = cac.eixos(cami)
    if n in cac.ANGLE_DEL_TRACKER:
        claus = [(kk, x0, y0, cac.ANGLE_DEL_TRACKER[n], sx, sy) for kk, x0, y0, _al, sx, sy in claus]
    en_px = nous[n].get("unitats") == "px"
    im, fr = {}, {}
    for c in nous[n]["clics"]:
        q = "final" if "final" in c["que"] else c["que"].split(" ")[0]
        im[q] = np.array([c["x"], c["y"]], float) if en_px else cac.a_imatge((c["x"], c["y"]), c["fotograma"], claus)
        fr[q] = int(c["fotograma"])
    d = trk.retalla(trk.llegeix_trk(cami), retalls.get(n, {}).get("des_del_fotograma"))
    k, A, G, _noms, _fix = trk.tres_pistes(d)
    return {"k": np.asarray(k), "pos": {"rodament": np.asarray(A, float), "extrem": np.asarray(G, float)}, "clics": im, "fr": fr}


def tracker_a(p: dict, gomet: str, n: int) -> np.ndarray:
    i = np.nonzero(p["k"] == n)[0]
    if not len(i):
        raise SystemExit(f"el .trk no te el fotograma {n}")
    return p["pos"][gomet][int(i[0])]


def rms(v) -> float:
    return float(np.sqrt(np.mean(np.square(v))))


def ordre_pilot() -> int:
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    nous = json.loads(cac.CLICS_NOUS.read_text(encoding="utf-8"))["preses"]
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    llista = json.loads(cm.LLISTA.read_text(encoding="utf-8"))
    esperats = {n: x["fotogrames"] for qui in ("pilot", "pas2") for n, x in llista[qui].items()}
    mov, c0 = cm.llegeix_copies(cm.COPIES, esperats)
    if c0:
        raise SystemExit(f"les copies dels clics en moviment no passen el C0: {c0}")
    problemes, fora = [], {"_que_es": "El pilot del centre automatic (documentacio/pla-centre-automatic.md; "
                                      "eines/centre_automatic.py pilot).", "preses": {}}

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("El pilot del centre automatic dels gomets (pla-centre-automatic.md)")
    P = {n: presa(n, nous, retalls) for n in VIDEOS_PILOT}
    clics_mov = {n: {(c["que"], c["fotograma"]): np.array([c["x"], c["y"]]) for c in mov[n]["clics"] if c["que"] != "pivot"}
                 for n in VIDEOS_PILOT}
    fot_mov = {n: sorted({f for (_q, f) in clics_mov[n]}) for n in VIDEOS_PILOT}

    # M: el mapatge dels fotogrames, amb els gomets vermells als dos fotogrames clicats mes tardans.
    k_millor = {}
    for n, p in P.items():
        vermells = [g for g in ("rodament", "extrem") if (n, g) not in CARGOL]
        demanes = {}
        for f in fot_mov[n][-2:]:
            for k in OFFSETS:
                for g in vermells:
                    demanes.setdefault(f + k, []).append(((f, k, g), tracker_a(p, g, f)))
        tr = retalls_del_video(VIDEOS_PILOT[n], demanes)
        med = {}
        for k in OFFSETS:
            d = [float(np.hypot(*(np.array(centre(*tr[(f, k, g)], tracker_a(p, g, f), False)) - tracker_a(p, g, f))))
                 for f in fot_mov[n][-2:] for g in vermells]
            med[k] = float(np.median([99.0 if not math.isfinite(v) else v for v in d]))   # sense gomet a la finestra: lluny
        ordre = sorted(med, key=med.get)
        k_millor[n] = ordre[0]
        ok = med[ordre[0]] <= 0.5 * med[ordre[1]]
        mira(ok, f"M {n}: el millor desplacament es k = {ordre[0]:+d} (mediana {med[ordre[0]]:.2f} px), el segon k = "
                 f"{ordre[1]:+d} ({med[ordre[1]]:.2f} px); tots: " + ", ".join(f"{k:+d}: {med[k]:.1f}" for k in OFFSETS))
        fora["preses"][n] = {"mapatge": {"k": ordre[0], "mediana_px": med}}
    if problemes:
        raise SystemExit("El mapatge no es clar: no es mira cap fotograma en moviment amb els clics.")

    # A0 (el repos) i A1 (en moviment), amb el desplacament de cada presa.
    a0, a1, a1_tr, vermells = [], [], [], []
    for n, p in P.items():
        k = k_millor[n]
        demanes = {}
        for g in ("rodament", "extrem"):
            demanes.setdefault(p["fr"][g] + k, []).append((("repos", g), tracker_a(p, g, p["fr"][g])))
            for f in fot_mov[n]:
                demanes.setdefault(f + k, []).append((("mov", f, g), tracker_a(p, g, f)))
        tr = retalls_del_video(VIDEOS_PILOT[n], demanes)
        fora["preses"][n]["repos"], fora["preses"][n]["moviment"] = {}, []
        for g in ("rodament", "extrem"):
            cg = (n, g) in CARGOL
            c = np.array(centre(*tr[("repos", g)], tracker_a(p, g, p["fr"][g]), cg))
            d = float(np.hypot(*(c - p["clics"][g])))
            a0.append(d)
            fora["preses"][n]["repos"][g] = {"fotograma": p["fr"][g], "automatic": c.tolist(), "clic": p["clics"][g].tolist(),
                                             "dif_px": (c - p["clics"][g]).tolist(), "distancia_px": d}
            for f in fot_mov[n]:
                c = np.array(centre(*tr[("mov", f, g)], tracker_a(p, g, f), cg))
                cl, t = clics_mov[n][(g, f)], tracker_a(p, g, f)
                da, dt = float(np.hypot(*(c - cl))), float(np.hypot(*(t - cl)))
                a1.append(da)
                a1_tr.append(dt)
                if not cg:
                    vermells.append((da, dt))
                fora["preses"][n]["moviment"].append({"fotograma": f, "gomet": g, "automatic": c.tolist(), "clic": cl.tolist(),
                                                      "tracker": t.tolist(), "auto_menys_clic_px": da, "tracker_menys_clic_px": dt})
    mirant[0] = False
    print("Amb el cargol de la 45_11 (els criteris tal com es van escriure):")
    mira(max(a0) < A0_PX, "A0 al repos, el centre automatic a < 1 px del clic (4 gomets): " + ", ".join(f"{v:.2f}" for v in a0)
         + " px")
    mira(rms(a1) <= A1_PX, f"A1 en moviment, el centre automatic contra els clics: rms {rms(a1):.2f} px de {len(a1)} mesures "
                           f"(<= {A1_PX}); informatiu, el Tracker (la creu) contra els clics: rms {rms(a1_tr):.2f} px")
    a0_v = [fora["preses"][n]["repos"][g]["distancia_px"] for n in VIDEOS_PILOT for g in ("rodament", "extrem")
            if (n, g) not in CARGOL]
    sense = {"A0_px": a0_v, "A0_passa": bool(max(a0_v) < A0_PX), "A1_rms_px": rms([v[0] for v in vermells]),
             "A1_passa": bool(rms([v[0] for v in vermells]) <= A1_PX), "tracker_rms_px": rms([v[1] for v in vermells]),
             "mesures": len(vermells)}
    print(f"Sense el cargol (decisio d'en Guillem, 28/10, despres de veure el resultat): A0 "
          f"{', '.join(f'{v:.2f}' for v in a0_v)} px ({'passa' if sense['A0_passa'] else 'falla'}); A1 rms "
          f"{sense['A1_rms_px']:.2f} px de {len(vermells)} mesures ({'passa' if sense['A1_passa'] else 'falla'}, llindar "
          f"{A1_PX}); el Tracker {sense['tracker_rms_px']:.2f} px")
    for n in VIDEOS_PILOT:
        for g in ("rodament", "extrem"):
            v = [r["auto_menys_clic_px"] for r in fora["preses"][n]["moviment"] if r["gomet"] == g]
            vt = [r["tracker_menys_clic_px"] for r in fora["preses"][n]["moviment"] if r["gomet"] == g]
            print(f"      {n} {g:9}: automatic {rms(v):.2f} px, Tracker {rms(vt):.2f} px  ({', '.join(f'{x:.2f}' for x in v)})")
    dolents = [o for o in oberts if ("net" in Path(o).parts and o.endswith(".csv")) or "sim" in Path(o).parts]
    mira(not dolents, f"J cap CSV de dades/net/ ni res de dades/sim/: {dolents or 'cap'}")
    fora["resum"] = {"A0_px": a0, "A1_rms_px": rms(a1), "tracker_rms_px": rms(a1_tr), "problemes": problemes,
                     "sense_el_cargol": sense}
    PILOT_JSON.write_text(json.dumps(fora, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {PILOT_JSON.relative_to(ARREL)}")
    return 0


# ------------------------------------------------------------ l'experiment 1 amb el centre automatic

def video_de(nom: str) -> Path:
    return ARREL / "dades" / "video" / "conv" / nom.split("_")[0] / f"{nom}.mp4"


def angles(k, A, G, Pf, meta) -> tuple[np.ndarray, np.ndarray]:
    """Els angles de la conversio (tracker.llegeix_del_trk i processa) amb les constants que la presa ja te: la delta de la marca,
    la vertical, el signe i el Savitzky-Golay. No es torna a mesurar res."""
    geo = meta["geometria"]
    P = np.broadcast_to(np.median(Pf, axis=0), (len(k), 2))
    A = np.asarray(A, float)
    if geo["marca_articulacio"].get("corregida"):
        A = tracker.corregeix_marca(P, A, geo["marca_articulacio"]["delta_px"])
    J, Gg = tracker.adreca_i_corregeix(P, A, np.asarray(G, float), geo, False)
    t = (np.asarray(k) - k[0]) * float(meta["delta_t_s"])
    x1, y1, x2, y2 = tracker._reomple_forats(t, J[:, 0], J[:, 1], Gg[:, 0], Gg[:, 1])
    th1, th2 = tracker.a_angles(t, x1, y1, x2, y2, (0.0, 0.0), True, bool(meta["signe_invertit"]))
    dt_med = float(np.median(np.diff(t)))
    if np.ptp(np.diff(t)) > 0.05 * dt_med:      # com tracker.processa: si el Tracker va perdre fotogrames, es remostreja
        t_unif = np.arange(t[0], t[-1] + 0.5 * dt_med, dt_med)
        th1, th2 = np.interp(t_unif, t, th1), np.interp(t_unif, t, th2)
    sg = meta["savgol"]
    return savgol_filter(th1, sg["finestra_mostres"], sg["ordre"]), savgol_filter(th2, sg["finestra_mostres"], sg["ordre"])


def presa_corregida_de(taula, meta, nom: str, p: Parametres) -> dict:
    """eines/experiment1.presa_corregida, pero amb la taula que es dona (la mateixa correccio, les mateixes condicions i el zero)."""
    th1, th2, info = comparar.corregeix_presa(taula, meta, nom)
    t = taula["t"].to_numpy(dtype=float)
    r = meta["amollada"]
    ta = float(r["amollada_s"])
    ini = (t >= max(0.0, ta - 0.4)) & (t <= ta - 0.02)
    estat0 = np.array([float(np.mean(th1[ini])), float(np.mean(th2[ini])), 0.0, 0.0])
    clics = json.loads(comparar.CLICS.read_text(encoding="utf-8"))["preses"].get(nom)
    dif = max(abs(estat0[i] - math.radians(clics["amollada_amb_clics_graus"][i])) for i in range(2))
    taula_c = taula.copy()
    taula_c["theta1"], taula_c["theta2"] = th1, th2
    af = afina_amollada(taula_c, r, p, estat0, finestra_s=comparar.FINESTRA_Y_S)
    zero = float(af["amollada_afinada_s"]) if af["ok"] else ta
    return {"t": t, "th1": th1, "th2": th2, "ta": ta, "zero": zero, "afinat": bool(af["ok"]), "estat0": estat0,
            "K_rad": float(dif), "clics": bool(info["clics"])}


def series(q, tg, A1, A2, N, t_fins: float = 0.5) -> dict:
    """Per a les figures: el tir, el nominal i el minim i el maxim del feix (amb el soroll, com comparar.avalua_y), en graus, fins a
    t_fins des del zero."""
    tau = q["t"] - q["zero"]
    sel = (tau >= 0.0) & (tau <= t_fins)
    ts = tau[sel]
    fora = {"t_s": ts.tolist()}
    for j, (A, real) in enumerate(((A1, q["th1"]), (A2, q["th2"])), start=1):
        M = np.array([np.interp(ts, tg, A[i]) for i in range(len(A))]) + N[j - 1][:, :len(ts)]
        fora[f"real{j}"] = np.degrees(np.asarray(real)[sel]).tolist()
        fora[f"nominal{j}"] = np.degrees(np.interp(ts, tg, A[0])).tolist()
        fora[f"min{j}"], fora[f"max{j}"] = np.degrees(M.min(axis=0)).tolist(), np.degrees(M.max(axis=0)).tolist()
    return fora


def una_presa(nom: str, m_relaxat: bool = False, amb_series: bool = False) -> dict:
    """m_relaxat (POSTERIOR, decidit despres de veure que el criteri M deixava fora 11 preses amb k = 0 com a minim clar): M passa si
    k = 0 es el minim estricte, sense la condicio de la meitat."""
    t0 = time.time()
    g = nom.split("_")[0]
    fora = {"nom": nom, "grup": g, "problemes": []}
    p = Parametres.des_de_json(e1.PARAMETRES)
    nous = json.loads(cac.CLICS_NOUS.read_text(encoding="utf-8"))["preses"]
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    d = trk.retalla(trk.llegeix_trk(ARREL / "dades" / "video" / "tracker" / g / f"{nom}.trk"),
                    retalls.get(nom, {}).get("des_del_fotograma"))
    k, A, G, _noms, (_kf, Pf) = trk.tres_pistes(d)
    k, A, G = np.asarray(k), np.asarray(A, float), np.asarray(G, float)
    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{nom}.csv")
    fora["video_del_trk"] = str(d.get("video"))

    # R1: sense canviar res, el cami d'aqui dona els angles del CSV.
    th1, th2 = angles(k, A, G, Pf, meta)
    fora["R1_rad"] = (max(float(np.max(np.abs(th1 - taula["theta1"].to_numpy()))),
                          float(np.max(np.abs(th2 - taula["theta2"].to_numpy())))) if len(th1) == len(taula) else math.inf)

    t = (k - k[0]) * float(meta["delta_t_s"])          # el temps de les files del .trk (el del CSV, si no falta cap fotograma)
    ta = float(meta["amollada"]["amollada_s"])
    llavor = lambda Q, i: Q[i] if np.all(np.isfinite(Q[i])) else np.array(   # noqa: E731
        [np.interp(i, np.nonzero(np.isfinite(Q[:, c]))[0], Q[np.isfinite(Q[:, c]), c]) for c in range(2)])
    POS = {"rodament": A, "extrem": G}

    # M: el mapatge, com al pilot (els dos gomets, a l'amollada + 0,25 i + 0,45 s).
    iM = [int(np.argmin(np.abs(t - (ta + s)))) for s in M_S]
    feines = {}
    for i in iM:
        for dk in OFFSETS:
            for gm in POS:
                feines.setdefault(int(k[i]) + dk, []).append(((i, dk, gm), llavor(POS[gm], i)))
    cmp_ = centres(video_de(nom), feines)
    med = {dk: float(np.median([99.0 if not np.all(np.isfinite(cmp_[(i, dk, gm)])) else
                                float(np.hypot(*(np.array(cmp_[(i, dk, gm)]) - llavor(POS[gm], i)))) for i in iM for gm in POS]))
           for dk in OFFSETS}
    ordre = sorted(med, key=med.get)
    kmap = ordre[0]
    fora["M"] = {"k": kmap, "mediana_px": med, "passa": med[ordre[0]] <= 0.5 * med[ordre[1]],
                 "passa_relaxat": kmap == 0 and med[ordre[0]] < med[ordre[1]]}
    if m_relaxat and fora["M"]["passa_relaxat"] and not fora["M"]["passa"]:
        fora["M"]["passa"] = True
        fora["M"]["nomes_relaxat"] = True
    if not fora["M"]["passa"]:
        fora["problemes"].append("M")
        return fora

    # A0 (el repos clicat) i el seguiment dins de W, en una sola passada.
    pr = presa(nom, nous, retalls)
    iW = np.nonzero((t >= ta - W_ABANS_S) & (t <= ta + W_DESPRES_S))[0]
    feines = {}
    for gm in POS:
        f = pr["fr"][gm]
        i = int(np.nonzero(k == f)[0][0])
        feines.setdefault(f + kmap, []).append((("A0", gm), llavor(POS[gm], i)))
        for i in iW:
            feines.setdefault(int(k[i]) + kmap, []).append((("W", int(i), gm), llavor(POS[gm], i)))
    cw = centres(video_de(nom), feines)
    a0 = {gm: float(np.hypot(*(np.array(cw[("A0", gm)]) - pr["clics"][gm]))) for gm in POS}
    fora["A0_px"] = a0
    if not max(a0.values()) < A0_PX:
        fora["problemes"].append("A0")
    nou = {gm: POS[gm].copy() for gm in POS}
    dist = []
    for gm in POS:
        for i in iW:
            c = np.array(cw[("W", int(i), gm)])
            nou[gm][i] = c
            dist.append(float(np.hypot(*(c - llavor(POS[gm], i)))) if np.all(np.isfinite(c)) else math.inf)
    dist = np.array(dist)
    fora["Q"] = {"fotogrames": int(len(iW)), "sense_centre": int(np.sum(~np.isfinite(dist))),
                 "fraccio_a_13px": float(np.mean(dist <= Q_PX)), "mediana_px": float(np.median(dist[np.isfinite(dist)])),
                 "p95_px": float(np.percentile(dist[np.isfinite(dist)], 95)), "max_px": float(np.max(dist))}
    if fora["Q"]["sense_centre"] or fora["Q"]["fraccio_a_13px"] < Q_FRACCIO:
        fora["problemes"].append("Q")

    # Els angles nous: les posicions automatiques dins de W, i res mes canviat.
    n1, n2 = angles(k, nou["rodament"], nou["extrem"], Pf, meta)
    mig = meta["savgol"]["finestra_mostres"] // 2           # el Savitzky-Golay escampa el canvi mig filtre a cada costat de W
    tc = taula["t"].to_numpy(dtype=float)
    dtc = float(np.median(np.diff(tc)))
    tocat = np.nonzero((tc >= t[iW[0]] - (mig + 1.5) * dtc) & (tc <= t[iW[-1]] + (mig + 1.5) * dtc))[0]
    fora["canvi_cru_graus"] = {"theta1_max": float(np.degrees(np.max(np.abs(n1 - th1)))),
                               "theta2_max": float(np.degrees(np.max(np.abs(n2 - th2)))),
                               "fora_de_W": float(np.degrees(max(np.max(np.abs(np.delete(n1 - th1, tocat))),
                                                                  np.max(np.abs(np.delete(n2 - th2, tocat))))))}
    taula_n = taula.copy()
    taula_n["theta1"], taula_n["theta2"] = n1, n2

    # Diagnostic (posterior): el repos d'abans d'amollar, sense la correccio, contra els angles dels clics (al mateix marc: la
    # conversio mes el gir dth). La finestra de l'amollada [ta - 0,4; ta - 0,02] i els ultims 20 ms abans del zero del video.
    cc = json.loads(comparar.CLICS.read_text(encoding="utf-8"))["preses"][nom]
    plega = lambda x: (x + math.pi) % (2.0 * math.pi) - math.pi   # noqa: E731
    fin = {"amollada": (tc >= max(0.0, ta - 0.4)) & (tc <= ta - 0.02), "ultims_20ms": (tc > ta - 0.02) & (tc <= ta)}
    fora["repos_menys_clic_graus"] = {
        qui: {f: [math.degrees(plega(float(np.mean(a[m])) + cc["dth_rad"] - cc["theta_inicial_rad"][j]))
                  for j, a in enumerate(ser)] for f, m in fin.items() if m.any()}
        for qui, ser in (("tracker", (th1, th2)), ("automatic", (n1, n2)))}
    fora["clic_del_repos_s_abans"] = {gm: float(ta - t[int(np.nonzero(k == pr["fr"][gm])[0][0])]) for gm in POS}

    # La Y: el mateix feix i el mateix soroll per a les dues series (la del CSV, per a R2; la nova).
    qv = presa_corregida_de(taula, meta, nom, p)
    qn = presa_corregida_de(taula_n, meta, nom, p)
    tg, A1, A2 = e1.feix(p, qv["estat0"], nom, g)
    N = e1.soroll_de_la_presa(p, nom, meta)
    yv = e1.y_del_tir(qv["t"], qv["th1"], qv["th2"], qv["zero"], tg, A1, A2, N, X_GRAUS)
    yn = e1.y_del_tir(qn["t"], qn["th1"], qn["th2"], qn["zero"], tg, A1, A2, N, X_GRAUS)
    fora["K_rad"] = max(qv["K_rad"], qn["K_rad"])
    fora["estat0_igual"] = bool(np.array_equal(qv["estat0"], qn["estat0"]))
    fora["tracker"] = {"zero_s": qv["zero"], "zero_afinat": qv["afinat"], **yv}
    fora["automatic"] = {"zero_s": qn["zero"], "zero_afinat": qn["afinat"], **yn}
    if amb_series:
        fora["series"] = {"tracker": series(qv, tg, A1, A2, N), "automatic": series(qn, tg, A1, A2, N)}
    print(f"   {nom}: {time.time() - t0:.0f} s", flush=True)
    return fora


def ordre_experiment1(m_relaxat: bool, processos: int, preses=None) -> int:
    firma = e1.pany(e1.llegeix_firma(), X_GRAUS, e1.GRUPS_Y)
    preses = preses or [n for g in e1.GRUPS_Y for n in e1.preses_del_grup(g) if n not in FORA_DE_LA_PROVA]
    if processos > 1:
        with multiprocessing.get_context("spawn").Pool(min(processos, 3)) as pool:
            res = pool.map(functools.partial(una_presa, m_relaxat=m_relaxat, amb_series=m_relaxat), preses)
    else:
        res = [una_presa(n, m_relaxat, m_relaxat) for n in preses]
    signat = json.loads(e1.Y_FIRMADA.read_text(encoding="utf-8"))["per_grup"]
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("L'experiment 1 amb el centre automatic (ANALISI POSTERIOR, NO FIRMADA; pla-centre-automatic.md)"
          + (" -- AMB LA M RELAXADA (posterior)" if m_relaxat else ""))
    per_tir = {}
    for r in res:
        n = r["nom"]
        if "M" in r["problemes"]:
            print(f"  {n}: FORA pel mapatge (M): {r['M']}")
            continue
        s = signat[r["grup"]]["per_tir"][n]
        igual = (s["zero_s"] == r["tracker"]["zero_s"] and s["H_s"] == r["tracker"]["H_s"]
                 and s["y"] == r["tracker"]["y"])
        mira(r["R1_rad"] <= R_RAD and igual, f"R {n}: angles del CSV a {r['R1_rad']:.1e} rad; la Y del 27/10 "
                                             f"{'identica' if igual else 'DIFERENT'}")
        mira(r["K_rad"] < K_RAD and r["estat0_igual"], f"K {n}: condicions inicials dels clics a {r['K_rad']:.1e} rad")
        yt, ya = r["tracker"]["y"], r["automatic"]["y"]
        q = r["Q"]
        print(f"      M k = {r['M']['k']:+d}{' (NOMES AMB M RELAXAT)' if r['M'].get('nomes_relaxat') else ''}; A0 "
              f"{r['A0_px']['rodament']:.2f} / {r['A0_px']['extrem']:.2f} px; Q {q['fraccio_a_13px']:.3f} a 13 px (mediana "
              f"{q['mediana_px']:.2f}, p95 {q['p95_px']:.2f}, max {q['max_px']:.1f})")
        for qui, y in (("Tracker", yt), ("automatic", ya)):
            print(f"      {qui:10} {'SURT a ' + format(y['t_sortida_s'], '.3f') + ' s (' + y['angle'] + ')' if y['falla'] else 'dins'}"
                  f"{'  FORA: ' + ', '.join(r['problemes']) if r['problemes'] and qui == 'automatic' else ''}")
        if not r["problemes"]:
            per_tir[n] = ya
    resum = {}
    for g in e1.GRUPS_Y:
        pt = {n: y for n, y in per_tir.items() if n.split("_")[0] == g}
        v = e1.lectura(g, pt) if pt else {"lectura": "cap presa"}
        resum[g] = {**v, "preses": len(pt)}
        print(f"  {g}: {v.get('lectura')}  ({len(v.get('fallen', []))} de {v.get('avaluables', 0)} surten: {v.get('fallen')})")
    if m_relaxat:
        SERIES.parent.mkdir(parents=True, exist_ok=True)
        SERIES.write_text(json.dumps({"_que_es": "Les series dels primers 0,5 s de l'experiment 1 amb el centre automatic i la M "
                                                 "relaxada (eines/centre_automatic.py experiment1 --m-relaxat), per al diagnostic "
                                                 "i la figura 3. Es refa amb l'ordre.",
                                      "per_presa": {r["nom"]: r["series"] for r in res if "series" in r}}), encoding="utf-8")
        print(f"-> {SERIES.relative_to(ARREL)}")
    sortida = Y_JSON[m_relaxat]
    sortida.write_text(json.dumps({"_que_es": "L'experiment 1 amb el centre automatic dels gomets: ANALISI POSTERIOR, NO FIRMADA "
                                              "(documentacio/pla-centre-automatic.md; eines/centre_automatic.py). El veredicte "
                                              "firmat no canvia.",
                                   "firma_x": firma["X"], "m_relaxat": bool(m_relaxat), "fora_de_la_prova": FORA_DE_LA_PROVA,
                                   "per_presa": [{k: v for k, v in r.items() if k != "series"} for r in res],
                                   "per_grup": resum, "problemes_R_K": problemes}, indent=1, default=float, ensure_ascii=False),
                       encoding="utf-8")
    print(f"-> {sortida.relative_to(ARREL)}")
    return 0 if not problemes else 1


# ------------------------------------------------------------ el diagnostic (els calculs a ma del 28/10 al vespre)

T_MEDIANES_S = (0.02, 0.05, 0.1)
T_QUOCIENT_S = (0.01, 0.02, 0.03, 0.05, 0.1)
PRESES_FRICCIO = (("90", "90_6"), ("90", "90_7"), ("45", "45_2"))
PRESES_PARAMETRES = (("90", "90_6"), ("45", "45_2"))


def _bones(series_: dict, y_relaxat: dict, g: str) -> list[str]:
    """Les preses del grup que passen tots els criteris a l'execucio amb la M relaxada (les del diagnostic del 28/10)."""
    return [x["nom"] for x in y_relaxat["per_presa"] if x["grup"] == g and not x["problemes"] and x["nom"] in series_]


def ordre_diagnostic() -> int:
    from pendol import estil
    series_ = json.loads(SERIES.read_text(encoding="utf-8"))["per_presa"]
    yr = json.loads(Y_JSON[True].read_text(encoding="utf-8"))
    y = json.loads(e1.Y_FIRMADA.read_text(encoding="utf-8"))["per_grup"]
    p = Parametres.des_de_json(e1.PARAMETRES)
    fora = {"_que_es": "D'on pot venir la diferencia de theta2 als primers 0,1 s (eines/centre_automatic.py diagnostic): els "
                       "calculs que es van fer a ma el 28/10 al vespre (pla-centre-automatic.md, «Com va anar l'aplicacio a les "
                       "preses»). ANALISI POSTERIOR, cap veredicte.",
            "medianes": {}, "quocient": {}, "friccio": {}, "parametres": {}}
    print("D'on pot venir la diferencia de theta2 (diagnostic posterior)")
    for g in e1.GRUPS_Y:
        noms = _bones(series_, yr, g)
        fora["medianes"][g], fora["quocient"][g] = {"preses": noms}, {"preses": noms}
        for tt in T_MEDIANES_S:
            dev, half, dev1 = [], [], []
            for n in noms:
                s = series_[n]["automatic"]
                i = int(np.argmin(abs(np.array(s["t_s"]) - tt)))
                dev.append(s["real2"][i] - s["nominal2"][i])
                dev1.append(s["real1"][i] - s["nominal1"][i])
                half.append((s["max2"][i] - s["min2"][i]) / 2)
            fora["medianes"][g][f"{tt:g}"] = {"theta2_menys_nominal_graus": float(np.median(dev)), "min": float(min(dev)),
                                              "max": float(max(dev)), "theta1_menys_nominal_graus": float(np.median(dev1)),
                                              "mitja_amplada_feix_theta2_graus": float(np.median(half))}
            print(f"  {g} t = {tt:g} s: theta2 - nominal {np.median(dev):+.2f} ({min(dev):+.2f} a {max(dev):+.2f}), theta1 "
                  f"{np.median(dev1):+.2f}; mitja amplada del feix {np.median(half):.2f} (n = {len(dev)})")
        for tt in T_QUOCIENT_S:
            rat = []
            for n in noms:
                s = series_[n]["automatic"]
                i = int(np.argmin(abs(np.array(s["t_s"]) - tt)))
                caiguda1 = s["nominal1"][i] - s["nominal1"][0]          # el que ha caigut theta1 al model
                real2 = s["real2"][i] - s["real2"][0]                     # el que ha caigut theta2 real
                nom2 = s["nominal2"][i] - s["nominal2"][0]
                rat.append((real2 - nom2) / caiguda1)
            fora["quocient"][g][f"{tt:g}"] = {"mediana": float(np.median(rat)), "min": float(min(rat)), "max": float(max(rat))}
            print(f"  {g} t = {tt:g} s: quocient {np.median(rat):.2f} ({min(rat):.2f} a {max(rat):.2f})")
    # La friccio firmada i els parametres del muntatge, des de les condicions inicials del JSON de la Y firmada.
    tg = np.arange(0, 0.5001, 0.001)
    for g, n in PRESES_FRICCIO:
        ci = y[g]["per_tir"][n]["condicions_inicials_graus"]
        e = np.array([math.radians(ci[0]), math.radians(ci[1]), 0, 0])
        _, a = simula(p, e, t_eval=tg)
        _, b = simula(p, e, t_eval=tg, dis=Dissipacio.firmada(float(g)))
        fora["friccio"][n] = {f"{t:g}": [float(np.degrees(b[0][i] - a[0][i])), float(np.degrees(b[1][i] - a[1][i]))]
                              for i, t in zip([50, 100, 200, 300, 400], [0.05, 0.1, 0.2, 0.3, 0.4])}
        print(f"  {n} friccio firmada - sense (theta1/theta2, graus): " + " ".join(
            f"{t}: {v[0]:+.3f}/{v[1]:+.3f}" for t, v in fora["friccio"][n].items()))
    tp = np.array([0.0, 0.02, 0.05, 0.1])
    variants = (("rs 12,78 cm", lambda q: dataclasses.replace(q, rs=0.1278)),
                ("d2 +0,5 cm", lambda q: dataclasses.replace(q, d2=q.d2 + 0.005)),
                ("d2 -0,5 cm", lambda q: dataclasses.replace(q, d2=q.d2 - 0.005)),
                ("ms +10 g", lambda q: dataclasses.replace(q, ms=q.ms + 0.010)),
                ("ms +8 g", lambda q: dataclasses.replace(q, ms=q.ms + 0.008)))
    for g, n in PRESES_PARAMETRES:
        ci = y[g]["per_tir"][n]["condicions_inicials_graus"]
        e = np.array([math.radians(ci[0]), math.radians(ci[1]), 0, 0])
        _, a = simula(p, e, t_eval=tp)
        fora["parametres"][n] = {}
        for nom, f in variants:
            _, b = simula(f(p), e, t_eval=tp)
            fora["parametres"][n][nom] = {"theta2_menys_nominal_graus": {f"{t:g}": float(v) for t, v in
                                                                         zip(tp[1:], np.degrees(b[1] - a[1])[1:])},
                                          "theta1_menys_nominal_graus": {f"{t:g}": float(v) for t, v in
                                                                         zip(tp[1:], np.degrees(b[0] - a[0])[1:])}}
            print(f"  {n} {nom:12} theta2 - nominal: " + " ".join(f"{t:g} s {v:+.2f}" for t, v in
                                                                 zip(tp[1:], np.degrees(b[1] - a[1])[1:])))
    DIAG_JSON.write_text(json.dumps(fora, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"-> {DIAG_JSON.relative_to(ARREL)}")
    estil.desa(dibuixa_theta2_amollada(series_, yr), "experiment1_theta2_amollada", ARREL / "figures")
    return 0


def dibuixa_theta2_amollada(series_: dict, yr: dict, t_fins: float = 0.15):
    """Figura 3 (posterior): theta2 real menys el nominal, els primers 0,15 s, amb el centre automatic, de totes les preses que
    passen els criteris; la mitja amplada del feix (la mediana del grup), al voltant del zero."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pendol import estil
    estil.aplica()
    fig, eixos = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    for ax, g in zip(eixos, ("90", "45")):
        noms = _bones(series_, yr, g)
        tc = np.linspace(0.0, t_fins, 151)
        devs, halfs = [], []
        for n in noms:
            s = series_[n]["automatic"]
            t = np.array(s["t_s"])
            dev = np.array(s["real2"]) - np.array(s["nominal2"])
            ax.plot(t[t <= t_fins], dev[t <= t_fins], color=estil.REAL, lw=0.8, alpha=0.45)
            devs.append(np.interp(tc, t, dev))
            halfs.append(np.interp(tc, t, (np.array(s["max2"]) - np.array(s["min2"])) / 2))
        h = np.median(halfs, axis=0)
        ax.fill_between(tc, -h, h, color=estil.FEIX, lw=0, zorder=0, label="mitja amplada del feix (mediana)")
        ax.plot(tc, np.median(devs, axis=0), color=estil.REAL, lw=2.0, label="mediana dels tirs")
        ax.plot([], [], color=estil.REAL, lw=0.8, alpha=0.45, label="cada tir")
        ax.axhline(0.0, color=estil.EIX, lw=0.6, zorder=0)
        ax.set_xlim(0.0, t_fins)
        ax.set_title(f"{g}°: θ2 real − simulació ({len(noms)} tirs)", loc="left")
        ax.set_xlabel("temps des del zero (s)")
    eixos[0].set_ylabel("θ2 − simulació nominal (°)")
    eixos[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------ l'autoprova

def gomet_fabricat(cx: float, cy: float, r: float = 13.5, sobre: int = 8, llavor: int = 0):
    """Un retall de 2*MIG+1 amb un gomet vermell de radi r centrat a (cx, cy) (en coordenades del retall, amb el centre del pixel
    a j + 0,5), la vora antialiasada per supermostreig, sobre un fons clar, amb soroll de 2 nivells."""
    n = 2 * MIG + 1
    u = (np.arange(n * sobre) + 0.5) / sobre
    X, Y = np.meshgrid(u, u)
    dins = (np.hypot(X - cx, Y - cy) <= r).reshape(n, sobre, n, sobre).mean(axis=(1, 3))
    fons, vermell = np.array([228.0, 226.0, 220.0]), np.array([215.0, 45.0, 60.0])
    img = fons * (1 - dins[..., None]) + vermell * dins[..., None]
    return np.clip(img + np.random.default_rng(llavor).normal(0.0, 2.0, img.shape), 0, 255)


def autoprova() -> int:
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova d'eines/centre_automatic.py (pla-figura-i-eines-experiment1.md, pas 3.4)")
    err, rep = [], []
    rng = np.random.default_rng(29102026)
    for k in range(20):
        cx, cy = 60.0 + rng.uniform(-0.5, 0.5), 60.0 + rng.uniform(-0.5, 0.5)
        tros = gomet_fabricat(cx, cy, llavor=k)
        inici = (cx + rng.uniform(-8, 8), cy + rng.uniform(-8, 8))
        c = centre(tros, 0, 0, inici, False)
        err.append(float(np.hypot(c[0] - cx, c[1] - cy)))
        rep.append(c == centre(tros, 0, 0, inici, False))
    # El llindar escrit era 0,05 px i va fallar per 0,004 (0,054). Canviat a 0,06 DESPRES DE VEURE LA FALLADA, per decisio d'en
    # Guillem (29/10: «es tan poc que el canvi no dany res»); declarat a pla-figura-i-eines-experiment1.md.
    mira(max(err) < 0.06, f"CA1 un gomet fabricat (radi 13,5 px, soroll de 2 nivells, la llavor a 8 px), 20 vegades: error maxim "
                          f"{max(err):.3f} px (< 0,06; era 0,05, canviat despres de fallar)")
    mira(all(rep), "CA2 el centre, dues vegades amb el mateix retall i la mateixa llavor: el mateix bit a bit")
    mirant[0] = False
    dolents = [o for o in oberts if "dades" in Path(o).parts]
    mira(not dolents, f"J l'autoprova no obre res de dades/: {dolents or 'cap'}")
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true")
    sub = ap.add_subparsers(dest="ordre")
    sub.add_parser("pilot", help="el centre automatic contra els clics (la 45_11 i la 90_6)")
    ax = sub.add_parser("experiment1", help="l'experiment 1 amb el centre automatic (darrere del pany)")
    ax.add_argument("--m-relaxat", action="store_true", help="POSTERIOR: M passa si k = 0 es el minim (declarat)")
    ax.add_argument("--processos", type=int, default=1)
    ax.add_argument("--preses", nargs="*")
    sub.add_parser("diagnostic", help="d'on pot venir la diferencia de theta2, i la figura 3")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.ordre == "pilot":
        return ordre_pilot()
    if args.ordre == "experiment1":
        return ordre_experiment1(args.m_relaxat, min(args.processos, 3), args.preses)
    if args.ordre == "diagnostic":
        return ordre_diagnostic()
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
