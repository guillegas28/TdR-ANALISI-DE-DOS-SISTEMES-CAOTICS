"""La comparacio del mapa amb el pendol de debo: la quadricula i les dues cel.les.

Es l'ultim tros del magnetic. La simulacio es va tancar el 28/10/2026 (la n i
la K, definitives), i aixo la compara amb el banc. El pla, amb els criteris
escrits ABANS de tirar, es documentacio/pla-quadricula-real-magnetic.md
(31/10/2026). Res d'aqui no toca el model: el llegeix.

QUE FA
    prepara   abans de tirar. Classifica cada cel.la de la quadricula com a
              SEGURA o DE FRONTERA segons el model, tria les dues cel.les del
              mini-experiment, i fa la plantilla per imprimir a mida real, el
              full de tirs en un ordre a l'atzar i el CSV buit on s'apunten.
    compara   despres de tirar. Llegeix el CSV i les comprovacions del muntatge
              i diu els criteris. Es pot passar amb la feina a mitges: el que
              no s'ha tirat no compta.

LA QUADRICULA
    Un quadrat de 9,75 x 9,75 cm centrat a l'origen (de -4,875 a 4,875 cm), amb
    cel.les de 7,5 mm: 13 x 13 = 169. La del mig cau damunt de l'origen. Hi ha
    la regio negra del centre, els tres punts de repos (a 4,48 cm) amb les seves
    conques i les fronteres entre elles. Les columnes C01-C13 van d'esquerra a
    dreta (x creixent) i les files F01-F13 de dalt a baix (y decreixent), com es
    llegeix un full.

    El 31/10 era un quadrat de 6 x 6 cm amb cel.les de 5 mm, de l'origen cap a
    l'imant 1. Es va canviar l'1/11, abans de cap tir, en veure que les 42
    cel.les segures eren totes de l'imant 1 o del centre: no hauria dit res de
    les conques dels imants 2 i 3 (decisio d'en Guillem). Cada cel.la es tira UN cop, des del seu centre, en un ordre
    a l'atzar amb la llavor fixada: si la tarda s'acaba abans, el que s'ha tirat
    continua sent una mostra a l'atzar.

SEGURA I DE FRONTERA
    Una cel.la es SEGURA si, en un disc de 1,5 mm de radi al voltant del seu
    centre, el model dona sempre el mateix desti, i aixo amb la b mesurada i
    amb els dos extrems de la seva incertesa (0,030 i 0,050). Els 1,5 mm son
    0,5 mm de posar la massa (una estimacio, no cap mesura) mes 1 mm de
    col.locar el paper (la tolerancia de les comprovacions E2 i E3). El disc es
    mostreja amb una subgraella de 21 x 21 punts. Si no es segura es DE
    FRONTERA, i el model en dona la fraccio del disc que va a cada desti.

ELS CRITERIS (escrits abans de tirar; no es canvien despres de veure'ls)
    M1  coincidencia a les segures: destins reals iguals als del model
    M2  el mateix a les de frontera
    C1  M1 >= 80 %
    C2  els desacords es concentren a la frontera: desacord a la frontera mes
        gran que a les segures, Fisher d'una cua, p < 0,05
    C3  sense veredicte: M2 contra la que esperaria el model amb el disc de r
    Avaluable si E1-E4 passen i s'han tirat >= 30 cel.les de cada classe.
    Les dues cel.les, 10 tirs cadascuna:
    D1  la de conca neta: >= 9 de 10 al desti del model
    D2  la de la pols: el desti mes frequent, com a molt 7 de 10

LES DADES
    magnetic/dades/quadricula_classificacio.json   la classificacio (abans)
    magnetic/dades/quadricula_real.csv             els tirs, escrits a ma:
                                                   ordre,cella,desti,repeticio,nota
    magnetic/dades/quadricula_real.json            qui, quan i les comprovacions
    Els tres es versionen. desti: 1, 2 o 3 (imant), 0 (centre), -1 (no s'atura
    en 200 s). Un tir fallit es torna a fer i es diu a la nota.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.quadricula prepara
    .venv\\Scripts\\python.exe -m magnetic.quadricula plantilla   # nomes la plantilla i el full
    .venv\\Scripts\\python.exe -m magnetic.quadricula compara
    .venv\\Scripts\\python.exe -m magnetic.quadricula dues_celles  # els dos punts on es va tirar (3/11)
    .venv\\Scripts\\python.exe -m magnetic.quadricula --autoprova
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle
from scipy.stats import binom, fisher_exact

from magnetic import compila, estil
from magnetic.fisica import CAMI_PER_DEFECTE, Parametres, dins_de_labast, punts_dequilibri, sobreescriu
from magnetic.motor import DT_PER_DEFECTE, Criteri, destins, graella_nucli

# ---------------------------------------------------------------------------
# Les decisions del pla (31/10/2026)
# ---------------------------------------------------------------------------
ZONA_M = (-0.04875, 0.04875, -0.04875, 0.04875)   # x_min, x_max, y_min, y_max
CELLA_M = 0.0075                                    # 13 x 13 cel.les
R_DISC_M = 0.0015                        # 0,5 mm de posar la massa + 1 mm del paper
PUNTS_DISC = 21                          # subgraella del disc, per costat
SENSIBILITAT_R_M = (0.0005, 0.003)       # nomes es diu quantes en serien segures
PUNTS_SENSIBILITAT = 15
B_EXTREMS = (0.030, 0.050)               # la incertesa mesurada de la b
LLAVOR_ORDRE = 20261031

# Les dues cel.les
NETA_DISTANCIES_M = (0.010, 0.015, 0.020)    # del punt de repos de l'imant 1
NETA_ANGLES_GRAUS = tuple(range(0, 360, 30))
NETA_FINESTRA_M = 0.006                      # es mira fins a 6 mm del candidat
NETA_PUNTS = 41
NETA_RADI_MINIM_M = 0.003
POLS_RADI_M = 0.09                           # del centre
POLS_ANGLES_GRAUS = tuple(range(0, 360, 5))
POLS_PMAX_MAXIM = 0.5
TIRS_CELLA = 10

# Els criteris
LLINDAR_M1 = 0.80
P_FISHER = 0.05
MIN_CELLES = 30
NETA_MINIM = 9          # de 10
POLS_MAXIM = 7          # de 10

# Les comprovacions del muntatge abans de tirar
TOL_E1_MM = 0.5         # la regla impresa, 100 mm
TOL_E2_MM = 1.0         # cada imant dins del seu cercle
DIST_IMANTS_MM = 50.0 * math.sqrt(3.0)
TOL_E2_DIST_MM = 1.0
TOL_E3_MM = 1.0         # la massa en repos al centre, sobre la creu
D0_CM, TOL_E4_CM = 2.0, 0.1

DADES = Path("magnetic/dades")
FIGURES = Path("magnetic/figures")
CLASSIFICACIO = DADES / "quadricula_classificacio.json"
TIRS_CSV = DADES / "quadricula_real.csv"
TIRS_JSON = DADES / "quadricula_real.json"
CODIS_VALIDS = (-1, 0, 1, 2, 3)


# ---------------------------------------------------------------------------
# La quadricula i els discs
# ---------------------------------------------------------------------------
def celles() -> list:
    """Les 169 cel.les, amb el centre en metres."""
    x_min, x_max, y_min, y_max = ZONA_M
    n_col = int(round((x_max - x_min) / CELLA_M))
    n_fil = int(round((y_max - y_min) / CELLA_M))
    fora = []
    for f in range(n_fil):
        y = y_max - (f + 0.5) * CELLA_M
        for c in range(n_col):
            x = x_min + (c + 0.5) * CELLA_M
            fora.append({"cella": f"F{f + 1:02d}C{c + 1:02d}", "fila": f + 1,
                         "columna": c + 1, "x_m": x, "y_m": y})
    return fora


def subgraella(r: float, punts: int):
    """Els desplacaments d'un costat i la mascara del disc de radi r."""
    off = np.linspace(-r, r, punts)
    dx, dy = np.meshgrid(off, off)
    return off, dx * dx + dy * dy <= r * r * (1.0 + 1e-12)


def destins_als_discs(centres, r, punts, desti_de_graella):
    """Per a cada centre (x, y): (destins dels punts del disc, desti del centre).

    desti_de_graella(xs, ys) torna una matriu (len(ys), len(xs)) de codis. Els
    centres amb la mateixa y es calculen d'una tirada: les seves subgraelles
    comparteixen les ys, i n'hi ha prou de posar totes les xs en fila.
    """
    off, mascara = subgraella(r, punts)
    mig = punts // 2
    fora = [None] * len(centres)
    per_y = {}
    for k, (x, y) in enumerate(centres):
        per_y.setdefault(y, []).append(k)
    for y, ks in per_y.items():
        xs = np.concatenate([centres[k][0] + off for k in ks])
        m = desti_de_graella(xs, y + off)
        for i, k in enumerate(ks):
            bloc = m[:, i * punts:(i + 1) * punts]
            fora[k] = (bloc[mascara], int(bloc[mig, mig]))
    return fora


def graella_del_model(p: Parametres, criteri: Criteri, dt: float):
    """desti_de_graella amb el model: el mateix nucli que fa els mapes."""
    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    passos = int(round(criteri.t_max / dt))
    passos_confirma = int(round(criteri.t_confirma / dt))
    passos_confirma_centre = int(round(criteri.t_confirma_centre / dt))

    def calcula(xs, ys):
        xs = np.ascontiguousarray(xs, dtype=np.float64)
        ys = np.ascontiguousarray(ys, dtype=np.float64)
        if not np.all(dins_de_labast(*np.meshgrid(xs, ys), p)):
            raise ValueError("Hi ha punts mes enlla del que arriba el fil.")
        imant = np.empty((ys.size, xs.size), dtype=np.int64)
        temps = np.empty((ys.size, xs.size), dtype=np.float64)
        graella_nucli(xs, ys, p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K,
                      dest_x, dest_y, dest_codi, dest_radi2,
                      dt, passos, passos_confirma, passos_confirma_centre, imant, temps)
        return imant

    return calcula


def fraccions(codis) -> dict:
    valors, comptes = np.unique(np.asarray(codis), return_counts=True)
    return {int(v): float(c) / float(len(codis)) for v, c in zip(valors, comptes)}


def classifica(centres, r, punts, funcions) -> list:
    """Segura o de frontera, per a cada centre.

    funcions: les desti_de_graella de cada b, la PRIMERA la de la b mesurada.
    Segura vol dir que tots els punts del disc donen el mateix desti amb totes
    les b. El desti del model es el del centre amb la b mesurada.
    """
    per_b = [destins_als_discs(centres, r, punts, f) for f in funcions]
    fora = []
    for k in range(len(centres)):
        disc, centre = per_b[0][k]
        tots = np.concatenate([per_b[i][k][0] for i in range(len(funcions))])
        fora.append({
            "classe": "segura" if np.all(tots == centre) else "frontera",
            "desti_model": centre,
            "fraccions": fraccions(disc),
        })
    return fora


def radi_net(m, off, centre_codi) -> float:
    """El radi del disc mes gran al voltant del centre on tot dona centre_codi."""
    dx, dy = np.meshgrid(off, off)
    d = np.hypot(dx, dy)
    altres = d[m != centre_codi]
    return float(altres.min()) if altres.size else float(off.max())


# ---------------------------------------------------------------------------
# prepara
# ---------------------------------------------------------------------------
def _funcions_del_model(p, criteri, dt):
    fs = [graella_del_model(p, criteri, dt)]
    for b in B_EXTREMS:
        q, _ = sobreescriu(p, b=b)
        fs.append(graella_del_model(q, criteri, dt))
    return fs


def tria_neta(p, funcions):
    """La cel.la de conca neta: a 1-2 cm del punt de repos de l'imant 1, la que
    dona el mateix desti en el disc mes gran (amb les tres b), d'almenys 3 mm.
    Empat: la mes a prop de 2 cm, i despres l'angle mes petit."""
    eq_x, eq_y, _ = punts_dequilibri(p)
    off = np.linspace(-NETA_FINESTRA_M, NETA_FINESTRA_M, NETA_PUNTS)
    mig = NETA_PUNTS // 2
    millor = None
    for d in NETA_DISTANCIES_M:
        for a in NETA_ANGLES_GRAUS:
            x = eq_x[0] + d * math.cos(math.radians(a))
            y = eq_y[0] + d * math.sin(math.radians(a))
            radis, codis = [], []
            for f in funcions:
                m = f(x + off, y + off)
                codis.append(int(m[mig, mig]))
                radis.append(radi_net(m, off, codis[0]))
            r = min(radis) if len(set(codis)) == 1 else 0.0
            clau = (r, -abs(d - 0.020), -a)
            if millor is None or clau > millor[0]:
                millor = (clau, {"x_m": x, "y_m": y, "distancia_al_repos_m": d,
                                 "angle_graus": a, "desti_model": codis[0], "radi_net_m": r})
    neta = millor[1]
    if neta["radi_net_m"] < NETA_RADI_MINIM_M:
        raise RuntimeError("Cap candidat de conca neta no arriba als 3 mm: torna-ho a plantejar.")
    neta["probabilitat_de_passar_D1"] = 1.0  # el model hi dona sempre el mateix desti
    return neta


def tria_pols(funcio):
    """La cel.la de la pols: a 9 cm del centre, la que reparteix el disc de r mes
    igual entre els destins (la p mes gran, la mes petita), amb cap p > 0,5.
    Empat: l'angle mes petit."""
    centres = [(POLS_RADI_M * math.cos(math.radians(a)), POLS_RADI_M * math.sin(math.radians(a)))
               for a in POLS_ANGLES_GRAUS]
    discs = destins_als_discs(centres, R_DISC_M, PUNTS_DISC, funcio)
    millor = None
    for a, (x, y), (disc, centre) in zip(POLS_ANGLES_GRAUS, centres, discs):
        fr = fraccions(disc)
        clau = (-max(fr.values()), -a)
        if millor is None or clau > millor[0]:
            millor = (clau, {"x_m": x, "y_m": y, "angle_graus": a,
                             "desti_model": centre, "fraccions": fr})
    pols = millor[1]
    if max(pols["fraccions"].values()) > POLS_PMAX_MAXIM:
        raise RuntimeError("Cap candidat de la pols no te la p mes gran <= 0,5.")
    # Com que amb 10 tirs dos destins no poden passar alhora de 7, la
    # probabilitat de fallar D2 es la suma de la de cada desti.
    p_falla = sum(float(binom.sf(POLS_MAXIM, TIRS_CELLA, q)) for q in pols["fraccions"].values())
    pols["probabilitat_de_fallar_D2"] = p_falla
    return pols


def _sha256(cami) -> str:
    return hashlib.sha256(Path(cami).read_bytes()).hexdigest()


def _hi_ha_tirs(cami: Path) -> bool:
    if not cami.exists():
        return False
    with cami.open(encoding="utf-8", newline="") as f:
        return any((fila.get("desti") or "").strip() for fila in csv.DictReader(f))


def prepara(fils: int) -> int:
    if compila.HI_HA_NUMBA:
        import numba
        numba.set_num_threads(min(fils, numba.config.NUMBA_NUM_THREADS))
    if _hi_ha_tirs(TIRS_CSV):
        raise SystemExit(f"{TIRS_CSV} ja te tirs apuntats: no el sobreescric. "
                         "Si de debo vols tornar a preparar, mou-lo abans.")

    p = Parametres.des_de_json()
    criteri, dt = Criteri(), DT_PER_DEFECTE
    print("=" * 72)
    print(" magnetic.quadricula prepara -- abans de tirar")
    print("=" * 72)
    print(f"   {compila.com_va()}, {fils} fils com a molt")
    t0 = time.perf_counter()
    funcions = _funcions_del_model(p, criteri, dt)

    llista = celles()
    centres = [(c["x_m"], c["y_m"]) for c in llista]
    print(f"   {len(llista)} cel.les, discs de {R_DISC_M * 1000:.1f} mm amb la b = "
          f"{p.b} i {B_EXTREMS[0]} / {B_EXTREMS[1]}...", flush=True)
    for c, r in zip(llista, classifica(centres, R_DISC_M, PUNTS_DISC, funcions)):
        c.update(r)
    print(f"   ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    sensibilitat = {}
    for r_s in SENSIBILITAT_R_M:
        res = classifica(centres, r_s, PUNTS_SENSIBILITAT, funcions[:1])
        sensibilitat[f"{r_s * 1000:g}_mm"] = sum(x["classe"] == "segura" for x in res)
    print(f"   sensibilitat: {sensibilitat} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    neta = tria_neta(p, funcions)
    pols = tria_pols(funcions[0])
    print(f"   les dues cel.les triades ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)

    ordre = [llista[i]["cella"] for i in np.random.default_rng(LLAVOR_ORDRE).permutation(len(llista))]
    n_seg = sum(c["classe"] == "segura" for c in llista)
    n_fro = len(llista) - n_seg
    eq_x, eq_y, _ = punts_dequilibri(p)

    sortida = {
        "_que_es": ("La classificacio de les cel.les de la quadricula, feta ABANS de tirar "
                    "(documentacio/pla-quadricula-real-magnetic.md). La fa magnetic.quadricula prepara."),
        "data": time.strftime("%Y-%m-%d %H:%M"),
        "parametres": p.com_a_dict(),
        "parametres_sha256": _sha256(CAMI_PER_DEFECTE),
        "criteri": criteri.com_a_dict(),
        "dt_s": dt,
        "zona_m": list(ZONA_M),
        "cella_m": CELLA_M,
        "disc": {"radi_m": R_DISC_M, "punts_per_costat": PUNTS_DISC,
                 "b_extrems": list(B_EXTREMS),
                 "_per_que": "0,5 mm de posar la massa (estimacio) + 1 mm del paper (E2, E3)"},
        "segures": n_seg,
        "de_frontera": n_fro,
        "segures_amb_altres_radis": sensibilitat,
        "punts_de_repos_m": [[float(eq_x[i]), float(eq_y[i])] for i in range(len(eq_x))],
        "celles": llista,
        "neta": neta,
        "pols": pols,
        "ordre": ordre,
        "llavor_ordre": LLAVOR_ORDRE,
    }
    DADES.mkdir(parents=True, exist_ok=True)
    CLASSIFICACIO.write_text(json.dumps(sortida, indent=1, ensure_ascii=False), encoding="utf-8")

    with TIRS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ordre", "cella", "desti", "repeticio", "nota"])
        k = 1
        for nom in ("neta", "pols"):
            for rep in range(1, TIRS_CELLA + 1):
                w.writerow([k, nom, "", rep, ""])
                k += 1
        for nom in ordre:
            w.writerow([k, nom, "", 1, ""])
            k += 1
    if not TIRS_JSON.exists():
        TIRS_JSON.write_text(json.dumps(PLANTILLA_COMPROVACIONS, indent=2, ensure_ascii=False),
                             encoding="utf-8")

    estil.aplica()
    fig = fes_plantilla(p, neta, pols)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "quadricula_plantilla.pdf")   # SENSE bbox tight: mida real
    plt.close(fig)
    fes_full_de_tirs(FIGURES / "quadricula_full_de_tirs.pdf")

    print(f"\n   segures: {n_seg}   de frontera: {n_fro}   (de {len(llista)})")
    if n_seg < MIN_CELLES or n_fro < MIN_CELLES:
        print(f"   ATENCIO: menys de {MIN_CELLES} d'alguna classe. El pla diu que s'ha de")
        print("   tornar a plantejar la zona abans de tirar.")
    print(f"   neta: ({neta['x_m'] * 100:.2f}, {neta['y_m'] * 100:.2f}) cm, net fins a "
          f"{neta['radi_net_m'] * 1000:.1f} mm")
    print(f"   pols: ({pols['x_m'] * 100:.2f}, {pols['y_m'] * 100:.2f}) cm, probabilitat de "
          f"fallar D2 amb el model: {pols['probabilitat_de_fallar_D2']:.3f}")
    print(f"\n   {CLASSIFICACIO}\n   {TIRS_CSV}  (buit)\n   {TIRS_JSON}")
    print(f"   {FIGURES / 'quadricula_plantilla.pdf'}  <- imprimeix-la a mida real, 100 %")
    print(f"   {FIGURES / 'quadricula_full_de_tirs.pdf'}")
    print(f"   {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


PLANTILLA_COMPROVACIONS = {
    "_que_es": ("Qui, quan i les comprovacions del muntatge abans de tirar. S'omple a ma. "
                "Tolerancies: E1 100 +- 0,5 mm; E2 cada imant a <= 1 mm del seu cercle i les "
                "distancies 86,6 +- 1 mm; E3 la massa en repos al centre a <= 1 mm de la creu; "
                "E4 la d0 2,0 +- 0,1 cm."),
    "data": None,
    "qui": None,
    "imant_fisic_a_cada_cercle": {"1": None, "2": None, "3": None},
    "E1_regla_mm": None,
    "E2_desviacio_imants_mm": {"1": None, "2": None, "3": None},
    "E2_distancies_mm": {"1-2": None, "2-3": None, "3-1": None},
    "E3_origen_mm": None,
    "E4_d0_cm": None,
    "observacio_repos_mm": {"1": None, "2": None, "3": None},
    "notes": "",
}


# ---------------------------------------------------------------------------
# La plantilla, a mida real, i el full de tirs
# ---------------------------------------------------------------------------
A4_CM = (21.0, 29.7)


def fes_plantilla(p, neta, pols):
    """Un A4 on 1 unitat de dades es 1 cm de paper, amb l'origen al mig.

    NO hi surt cap prediccio del model: qui tira no ha de saber que s'espera.
    """
    amp, alt = A4_CM
    fig = plt.figure(figsize=(amp / 2.54, alt / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-amp / 2, amp / 2)
    ax.set_ylim(-alt / 2, alt / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    cm = 100.0
    gris = estil.TINTA_SUAU

    # La quadricula
    x_min, x_max, y_min, y_max = (v * cm for v in ZONA_M)
    pas = CELLA_M * cm
    for k in range(int(round((x_max - x_min) / pas)) + 1):
        ax.plot([x_min + k * pas] * 2, [y_min, y_max], color=gris, lw=0.35)
    for k in range(int(round((y_max - y_min) / pas)) + 1):
        ax.plot([x_min, x_max], [y_min + k * pas] * 2, color=gris, lw=0.35)
    for c in range(int(round((x_max - x_min) / pas))):
        ax.text(x_min + (c + 0.5) * pas, y_max + 0.15, f"{c + 1:02d}", ha="center",
                va="bottom", fontsize=4.5, color=estil.TINTA)
    # Els numeros de fila, prou a l'esquerra perque no els tapi la creu de
    # l'origen, que es a la vora esquerra de la quadricula.
    for f in range(int(round((y_max - y_min) / pas))):
        ax.text(x_min - 0.55, y_max - (f + 0.5) * pas, f"{f + 1:02d}", ha="right",
                va="center", fontsize=4.5, color=estil.TINTA)
    ax.text((x_min + x_max) / 2, y_max + 0.55, "columna (C)", ha="center", fontsize=6)
    ax.text(x_min - 1.2, (y_min + y_max) / 2, "fila (F)", ha="center", va="center",
            rotation=90, fontsize=6)

    # Els imants, a mida real (5 mm de radi), i els punts de repos
    eq_x, eq_y, _ = punts_dequilibri(p)
    for i in range(len(p.im_x)):
        x, y = p.im_x[i] * cm, p.im_y[i] * cm
        ax.add_patch(Circle((x, y), 0.5, fill=False, lw=0.8, color=estil.TINTA))
        rr = math.hypot(x, y)
        ax.text(x * (rr + 1.4) / rr, y * (rr + 1.4) / rr, str(i + 1), ha="center",
                va="center", fontsize=9)
        ax.plot([eq_x[i] * cm - 0.12, eq_x[i] * cm + 0.12], [eq_y[i] * cm] * 2, color=estil.TINTA, lw=0.5)
        ax.plot([eq_x[i] * cm] * 2, [eq_y[i] * cm - 0.12, eq_y[i] * cm + 0.12], color=estil.TINTA, lw=0.5)

    # L'origen i els eixos. La creu va a l'origen, al mig de la quadricula; les
    # fletxes, fora, perque no tapin cap cel.la.
    ax.plot([-0.4, 0.4], [0, 0], color=estil.TINTA, lw=0.9)
    ax.plot([0, 0], [-0.4, 0.4], color=estil.TINTA, lw=0.9)
    ax.text(-0.15, -0.15, "O", ha="right", va="top", fontsize=6)
    ax.annotate("", xy=(2.5, y_min - 0.6), xytext=(0.0, y_min - 0.6),
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(2.65, y_min - 0.6, "x, cap a l'imant 1", va="center", fontsize=6)
    ax.annotate("", xy=(x_min - 2.0, 1.5), xytext=(x_min - 2.0, -1.0),
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(x_min - 2.0, 1.6, "y", ha="center", va="bottom", fontsize=6)

    # Les dues cel.les
    for nom, c in (("N", neta), ("P", pols)):
        x, y = c["x_m"] * cm, c["y_m"] * cm
        ax.add_patch(Circle((x, y), 0.25, fill=False, lw=0.7, color=estil.TINTA))
        ax.plot([x - 0.08, x + 0.08], [y, y], color=estil.TINTA, lw=0.5)
        ax.plot([x, x], [y - 0.08, y + 0.08], color=estil.TINTA, lw=0.5)
        ax.text(x + 0.35, y + 0.35, nom, fontsize=8, fontweight="bold")

    # La regla de 10 cm
    y_r = -12.0
    ax.plot([-5, 5], [y_r, y_r], color=estil.TINTA, lw=0.8)
    for k in range(11):
        ax.plot([-5 + k] * 2, [y_r, y_r + (0.35 if k % 5 == 0 else 0.2)], color=estil.TINTA, lw=0.6)
    ax.text(0, y_r - 0.35, "10 cm. Comprova-ho amb un regle: 100 ± 0,5 mm (E1). "
            "Imprimeix a mida real (100 %), sense ajustar a la pàgina.",
            ha="center", va="top", fontsize=6.5)

    ax.text(0, 13.6, "Quadrícula del pèndol magnètic — plantilla", ha="center", fontsize=11,
            fontweight="bold")
    ax.text(0, 12.9,
            "Els cercles 1, 2 i 3 van damunt dels imants, i l'imant del cercle 1 és l'«1» (E2). "
            "La creu O és l'origen:\nla massa en repòs al centre hi ha de quedar a ≤ 1 mm (E3). "
            "Les creuetes petites són els punts de repòs del model. N i P: les dues cel·les.",
            ha="center", va="top", fontsize=6.5)
    ax.text(0, -14.2, "magnetic.quadricula prepara · documentacio/pla-quadricula-real-magnetic.md",
            ha="center", fontsize=5.5, color=gris)
    return fig


def fes_nomes_quadricula():
    """Un A4 a mida real NOMES amb la quadricula, els numeros de fila i de
    columna i una regla de 10 cm (02/11, en Guillem: «només vull quadrícula, la
    resta ja ho faig jo»). Sense cercles, sense N i P i sense textos: la
    col.locacio la fa ell. La quadricula, centrada a la pagina."""
    amp, alt = A4_CM
    fig = plt.figure(figsize=(amp / 2.54, alt / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-amp / 2, amp / 2)
    ax.set_ylim(-alt / 2, alt / 2)
    ax.set_aspect("equal")
    ax.axis("off")
    cm = 100.0
    x_min, x_max, y_min, y_max = (v * cm for v in ZONA_M)
    pas = CELLA_M * cm
    n_col = int(round((x_max - x_min) / pas))
    n_fil = int(round((y_max - y_min) / pas))
    for k in range(n_col + 1):
        ax.plot([x_min + k * pas] * 2, [y_min, y_max], color=estil.TINTA, lw=0.4)
    for k in range(n_fil + 1):
        ax.plot([x_min, x_max], [y_min + k * pas] * 2, color=estil.TINTA, lw=0.4)
    for c in range(n_col):
        ax.text(x_min + (c + 0.5) * pas, y_max + 0.15, f"{c + 1:02d}", ha="center",
                va="bottom", fontsize=6, color=estil.TINTA)
    for f in range(n_fil):
        ax.text(x_min - 0.15, y_max - (f + 0.5) * pas, f"{f + 1:02d}", ha="right",
                va="center", fontsize=6, color=estil.TINTA)
    # La regla de 10 cm, per comprovar que s'ha impres a mida real (E1)
    y_r = y_min - 2.0
    ax.plot([-5, 5], [y_r, y_r], color=estil.TINTA, lw=0.8)
    for k in range(11):
        ax.plot([-5 + k] * 2, [y_r, y_r + (0.35 if k % 5 == 0 else 0.2)], color=estil.TINTA, lw=0.6)
    ax.text(5.2, y_r, "10 cm", va="center", fontsize=7, color=estil.TINTA)
    return fig


def fes_full_de_tirs(cami: Path) -> None:
    """L'ordre dels tirs, amb un buit per apuntar-hi el desti."""
    files = []
    with TIRS_CSV.open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            files.append(fila)
    per_pagina = 56
    with PdfPages(cami) as pdf:
        for p0 in range(0, len(files), per_pagina):
            fig = plt.figure(figsize=(A4_CM[0] / 2.54, A4_CM[1] / 2.54))
            fig.text(0.08, 0.96, "Full de tirs — quadrícula del pèndol magnètic", fontsize=11,
                     fontweight="bold")
            fig.text(0.08, 0.94, "destí: 1, 2, 3 (imant), 0 (centre), −1 (no s'atura en 200 s). "
                     "Un tir fallit es torna a fer i es diu a la nota.", fontsize=7)
            for k, fila in enumerate(files[p0:p0 + per_pagina]):
                y = 0.91 - k * 0.0155
                fig.text(0.08, y, fila["ordre"], fontsize=8, ha="right")
                fig.text(0.11, y, f"{fila['cella']}  (rep. {fila['repeticio']})", fontsize=8)
                fig.text(0.40, y, "destí: ______", fontsize=8)
                fig.text(0.58, y, "nota: ______________________________", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)


# ---------------------------------------------------------------------------
# compara
# ---------------------------------------------------------------------------
def llegeix_tirs(cami: Path, noms_valids) -> list:
    """Les files del CSV que tenen desti. Peta amb el numero de linia si n'hi
    ha cap de mal escrita: es mes facil arreglar-ho que no pas descobrir-ho a
    la figura."""
    tirs, vistos = [], set()
    with cami.open(encoding="utf-8", newline="") as f:
        lector = csv.DictReader(f)
        if lector.fieldnames != ["ordre", "cella", "desti", "repeticio", "nota"]:
            raise ValueError(f"{cami}: la capcalera ha de ser ordre,cella,desti,repeticio,nota")
        for linia, fila in enumerate(lector, start=2):
            desti = (fila["desti"] or "").strip()
            if not desti:
                continue
            try:
                codi = int(desti)
            except ValueError:
                raise ValueError(f"{cami}, linia {linia}: el desti '{desti}' no es un numero")
            if codi not in CODIS_VALIDS:
                raise ValueError(f"{cami}, linia {linia}: el desti {codi} no existeix")
            cella = fila["cella"].strip()
            if cella not in noms_valids:
                raise ValueError(f"{cami}, linia {linia}: la cel.la '{cella}' no existeix")
            clau = (cella, int(fila["repeticio"]))
            if clau in vistos:
                raise ValueError(f"{cami}, linia {linia}: {cella} rep. {clau[1]} hi es dues vegades")
            vistos.add(clau)
            tirs.append({"cella": cella, "desti": codi, "repeticio": clau[1]})
    return tirs


def wilson(encerts: int, n: int, z: float = 1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = encerts / n
    mig = (p + z * z / (2 * n)) / (1 + z * z / n)
    amp = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (mig - amp, mig + amp)


def comprova_muntatge(c: dict) -> list:
    """Els problemes de les comprovacions E1-E4 (llista buida si tot passa)."""
    mal = []

    def falta(v):
        return v is None or (isinstance(v, dict) and any(x is None for x in v.values()))

    if falta(c.get("E1_regla_mm")) or abs(c["E1_regla_mm"] - 100.0) > TOL_E1_MM:
        mal.append(f"E1: la regla ha de fer 100 +- {TOL_E1_MM} mm ({c.get('E1_regla_mm')})")
    des, dis = c.get("E2_desviacio_imants_mm"), c.get("E2_distancies_mm")
    if falta(des) or falta(dis) or max(des.values()) > TOL_E2_MM or \
            max(abs(v - DIST_IMANTS_MM) for v in dis.values()) > TOL_E2_DIST_MM:
        mal.append(f"E2: imants a <= {TOL_E2_MM} mm i distancies de {DIST_IMANTS_MM:.1f} +- "
                   f"{TOL_E2_DIST_MM} mm ({des}, {dis})")
    if falta(c.get("E3_origen_mm")) or c["E3_origen_mm"] > TOL_E3_MM:
        mal.append(f"E3: l'origen a <= {TOL_E3_MM} mm ({c.get('E3_origen_mm')})")
    if falta(c.get("E4_d0_cm")) or abs(c["E4_d0_cm"] - D0_CM) > TOL_E4_CM:
        mal.append(f"E4: la d0 ha de fer {D0_CM} +- {TOL_E4_CM} cm ({c.get('E4_d0_cm')})")
    return mal


def avalua(classificacio: dict, tirs: list, comprovacions: dict) -> dict:
    """Els criteris, a partir de la classificacio i dels tirs. Sense fitxers,
    perque l'autoprova la pugui cridar amb dades fabricades."""
    per_nom = {c["cella"]: c for c in classificacio["celles"]}
    res = {"muntatge": comprova_muntatge(comprovacions)}

    # La quadricula
    files = []
    for t in tirs:
        if t["cella"] in per_nom:
            c = per_nom[t["cella"]]
            files.append({"cella": t["cella"], "classe": c["classe"], "real": t["desti"],
                          "model": c["desti_model"],
                          "p_model": c["fraccions"].get(str(c["desti_model"]),
                                                        c["fraccions"].get(c["desti_model"], 0.0))})
    for classe in ("segura", "frontera"):
        de = [f for f in files if f["classe"] == classe]
        enc = sum(f["real"] == f["model"] for f in de)
        res[classe] = {"tirades": len(de), "coincidencies": enc,
                       "fraccio": enc / len(de) if de else None,
                       "wilson_95": list(wilson(enc, len(de)))}
    seg, fro = res["segura"], res["frontera"]
    res["avaluable"] = (not res["muntatge"] and seg["tirades"] >= MIN_CELLES
                        and fro["tirades"] >= MIN_CELLES)
    # Una decisio presa DESPRES de tirar (la del 3/11: E1-E4 fetes a ull). No toca
    # cap numero ni el criteri d'abans: nomes s'hi afegeix la segona lectura.
    decisio = comprovacions.get("decisio_avaluable")
    res["avaluable_per_decisio"] = bool(decisio) and seg["tirades"] >= MIN_CELLES \
        and fro["tirades"] >= MIN_CELLES
    res["decisio_avaluable"] = decisio
    res["C1"] = None if seg["fraccio"] is None else seg["fraccio"] >= LLINDAR_M1
    if seg["tirades"] and fro["tirades"]:
        taula = [[fro["tirades"] - fro["coincidencies"], fro["coincidencies"]],
                 [seg["tirades"] - seg["coincidencies"], seg["coincidencies"]]]
        p = float(fisher_exact(taula, alternative="greater")[1])
        res["C2"] = {"p": p, "passa": p < P_FISHER, "taula_desacord_acord": taula}
    else:
        res["C2"] = None
    de_fro = [f for f in files if f["classe"] == "frontera"]
    res["C3"] = {"m2_esperada_pel_model": (sum(f["p_model"] for f in de_fro) / len(de_fro)
                                           if de_fro else None)}
    res["files"] = files

    # Les dues cel.les
    for nom in ("neta", "pols"):
        de = [t["desti"] for t in tirs if t["cella"] == nom]
        comptes = {str(k): de.count(k) for k in sorted(set(de))}
        cel = classificacio[nom]
        entrada = {"tirs": len(de), "comptes": comptes, "desti_model": cel["desti_model"]}
        if len(de) < TIRS_CELLA:
            entrada["passa"] = None
        elif nom == "neta":
            entrada["passa"] = de.count(cel["desti_model"]) >= NETA_MINIM
        else:
            entrada["passa"] = max(de.count(k) for k in set(de)) <= POLS_MAXIM
        res[nom] = entrada
    return res


def plantilla(nomes_quadricula: bool = False) -> int:
    """Torna a fer la plantilla i el full de tirs a partir de la classificacio
    ja desada, sense calcular res. No toca ni la classificacio ni el CSV. Amb
    nomes_quadricula, nomes el full de la quadricula sola."""
    estil.aplica()
    if nomes_quadricula:
        fig = fes_nomes_quadricula()
        fig.savefig(FIGURES / "quadricula_nomes_quadricula.pdf")   # SENSE bbox tight
        plt.close(fig)
        print(f"   {FIGURES / 'quadricula_nomes_quadricula.pdf'}")
        return 0
    classificacio = json.loads(CLASSIFICACIO.read_text(encoding="utf-8"))
    fig = fes_plantilla(Parametres.des_de_json(), classificacio["neta"], classificacio["pols"])
    fig.savefig(FIGURES / "quadricula_plantilla.pdf")   # SENSE bbox tight: mida real
    plt.close(fig)
    fes_full_de_tirs(FIGURES / "quadricula_full_de_tirs.pdf")
    print(f"   {FIGURES / 'quadricula_plantilla.pdf'}\n   {FIGURES / 'quadricula_full_de_tirs.pdf'}")
    return 0


def _text_si(v):
    return "pendent" if v is None else ("PASSA" if v else "NO PASSA")


def compara() -> int:
    classificacio = json.loads(CLASSIFICACIO.read_text(encoding="utf-8"))
    comprovacions = json.loads(TIRS_JSON.read_text(encoding="utf-8"))
    noms = {c["cella"] for c in classificacio["celles"]} | {"neta", "pols"}
    tirs = llegeix_tirs(TIRS_CSV, noms)
    res = avalua(classificacio, tirs, comprovacions)

    print("=" * 72)
    print(" magnetic.quadricula compara -- el mapa contra el pendol")
    print("=" * 72)
    print(f"   {len(tirs)} tirs apuntats")
    if res["muntatge"]:
        print("   LES COMPROVACIONS DEL MUNTATGE NO PASSEN (o falten):")
        for m in res["muntatge"]:
            print(f"     {m}")
    for nom, d in (("neta", "D1"), ("pols", "D2")):
        e = res[nom]
        print(f"   {d} {nom}: {e['tirs']} tirs, {e['comptes']} (model: {e['desti_model']})  "
              f"-> {_text_si(e['passa'])}")
    for classe in ("segura", "frontera"):
        e = res[classe]
        fr = "—" if e["fraccio"] is None else f"{100 * e['fraccio']:.1f} %"
        print(f"   {classe:>9}: {e['coincidencies']} de {e['tirades']} coincideixen ({fr}, "
              f"Wilson 95 % {100 * e['wilson_95'][0]:.0f}-{100 * e['wilson_95'][1]:.0f} %)")
    print(f"   C1 (M1 >= {100 * LLINDAR_M1:.0f} %): {_text_si(res['C1'])}")
    if res["C2"]:
        print(f"   C2 (desacords a la frontera, Fisher p < {P_FISHER}): p = {res['C2']['p']:.2g}  "
              f"-> {_text_si(res['C2']['passa'])}")
    m2e = res["C3"]["m2_esperada_pel_model"]
    if m2e is not None:
        print(f"   C3 (sense veredicte): la M2 que esperaria el model, {100 * m2e:.1f} %")
    print(f"   avaluable pel criteri escrit abans de tirar: {'si' if res['avaluable'] else 'no'}")
    if res["decisio_avaluable"]:
        d = res["decisio_avaluable"]
        print(f"   avaluable per decisio posterior ({d.get('qui')}, {d.get('data')}, DESPRES de tirar): "
              f"{'si' if res['avaluable_per_decisio'] else 'no'}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "quadricula_resultats.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    estil.aplica()
    estil.desa(_figura_comparacio(classificacio, tirs), "quadricula_comparacio")
    return 0


# ---------------------------------------------------------------------------
# Les dues cel.les, tal com es van tirar (3/11)
# ---------------------------------------------------------------------------
DUES_CELLES_CSV = DADES / "quadricula_dues_celles.csv"


def avalua_dues_celles(punts) -> list:
    """Les regles D1 i D2 de sempre, sobre els punts on es va tirar de debo.

    D1 (neta): >= 9 de 10 al desti del model; s'aplica si la cel.la es segura.
    D2 (pols): el desti mes frequent, com a molt 7 de 10; s'aplica si el model
    hi reparteix el disc amb cap desti per sobre de 0,5.
    Sense fitxers, perque l'autoprova la pugui cridar amb dades fabricades.
    """
    fora = []
    for p in punts:
        d, pr = p["destins"], p["prediccio"]
        fr = {int(k): v for k, v in pr["fraccions"].items()}
        e = {"punt": p["punt"], "tipus": p["tipus"], "x_m": p.get("x_m"), "y_m": p.get("y_m"),
             "tirs": len(d), "comptes": {str(k): d.count(k) for k in sorted(set(d))},
             "prediccio": {"classe": pr["classe"], "desti_model": pr["desti_model"],
                           "fraccions": {str(k): v for k, v in fr.items()}},
             "prediccio_abans_de_tirar": p["prediccio_abans_de_tirar"]}
        if p["tipus"] == "neta":
            e["regla"] = f"D1: >= {NETA_MINIM} de {TIRS_CELLA} al desti del model"
            e["aplicable"] = pr["classe"] == "segura"
            e["passa"] = None if len(d) < TIRS_CELLA else d.count(pr["desti_model"]) >= NETA_MINIM
        else:
            e["regla"] = f"D2: el desti mes frequent, com a molt {POLS_MAXIM} de {TIRS_CELLA}"
            e["aplicable"] = max(fr.values()) <= POLS_PMAX_MAXIM
            e["probabilitat_de_fallar_amb_el_model"] = sum(
                float(binom.sf(POLS_MAXIM, TIRS_CELLA, q)) for q in fr.values())
            e["passa"] = None if len(d) < TIRS_CELLA else max(d.count(k) for k in set(d)) <= POLS_MAXIM
        fora.append(e)
    return fora


def dues_celles(fils: int) -> int:
    """Les dues cel.les tal com es van tirar el 3/11, que NO van ser N i P: la
    neta, a F11C06 (per error), i la de la pols, a (12; 0,5) cm (triada per en
    Guillem). La prediccio d'un punt que es una cel.la de la quadricula es la de
    la classificacio, feta abans de tirar; la de qualsevol altre punt es calcula
    ara, DESPRES de tirar, amb el mateix disc i les tres b, i la sortida ho diu."""
    if compila.HI_HA_NUMBA:
        import numba
        numba.set_num_threads(min(fils, numba.config.NUMBA_NUM_THREADS))
    classificacio = json.loads(CLASSIFICACIO.read_text(encoding="utf-8"))
    per_punt = {}
    with DUES_CELLES_CSV.open(encoding="utf-8", newline="") as f:
        for linia, fila in enumerate(csv.DictReader(f), start=2):
            codi = int(fila["desti"])
            if codi not in CODIS_VALIDS:
                raise ValueError(f"{DUES_CELLES_CSV}, linia {linia}: el desti {codi} no existeix")
            q = per_punt.setdefault(fila["punt"], {
                "punt": fila["punt"], "tipus": fila["tipus"], "destins": [],
                "x_m": float(fila["x_cm"]) / 100.0, "y_m": float(fila["y_cm"]) / 100.0})
            q["destins"].append(codi)

    funcions = None
    for q in per_punt.values():
        cella = next((c for c in classificacio["celles"]
                      if abs(c["x_m"] - q["x_m"]) < 1e-9 and abs(c["y_m"] - q["y_m"]) < 1e-9), None)
        if cella is not None:
            q["prediccio"] = {k: cella[k] for k in ("classe", "desti_model", "fraccions")}
            q["prediccio_abans_de_tirar"] = True
            q["punt_de_la_classificacio"] = cella["cella"]
        else:
            if funcions is None:
                funcions = _funcions_del_model(Parametres.des_de_json(), Criteri(), DT_PER_DEFECTE)
            q["prediccio"] = classifica([(q["x_m"], q["y_m"])], R_DISC_M, PUNTS_DISC, funcions)[0]
            q["prediccio_abans_de_tirar"] = False

    res = avalua_dues_celles(list(per_punt.values()))
    print("=" * 72)
    print(" magnetic.quadricula dues_celles -- els dos punts on es va tirar de debo")
    print("=" * 72)
    for e in res:
        quan = "ABANS de tirar (classificacio)" if e["prediccio_abans_de_tirar"] else "DESPRES de tirar"
        print(f"   {e['punt']} ({e['tipus']}, {e['x_m'] * 100:g} ; {e['y_m'] * 100:g} cm): "
              f"{e['tirs']} tirs {e['comptes']}")
        print(f"      model ({quan}): {e['prediccio']['classe']}, desti {e['prediccio']['desti_model']}, "
              f"disc {e['prediccio']['fraccions']}")
        extra = (f", probabilitat de fallar-la amb el model {e['probabilitat_de_fallar_amb_el_model']:.3f}"
                 if "probabilitat_de_fallar_amb_el_model" in e else "")
        print(f"      {e['regla']}: {_text_si(e['passa'])} (aplicable: {'si' if e['aplicable'] else 'no'}{extra})")
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "quadricula_dues_celles.json").write_text(
        json.dumps({"_que_es": ("Les dues cel.les tal com es van tirar el 3/11/2026: no van ser N i P. "
                                "Vegeu documentacio/pla-quadricula-real-magnetic.md."),
                    "punts": res}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n   {FIGURES / 'quadricula_dues_celles.json'}")
    return 0


def _figura_comparacio(classificacio, tirs):
    """El model i el pendol, cel.la a cel.la, amb els desacords marcats."""
    real = {t["cella"]: t["desti"] for t in tirs if t["repeticio"] == 1}
    fig, eixos = plt.subplots(1, 2, figsize=(9.6, 5.0))
    cm, mida = 100.0, CELLA_M * 100.0
    for eix, titol, quin in ((eixos[0], "El model", "model"), (eixos[1], "El pèndol", "real")):
        for c in classificacio["celles"]:
            x0, y0 = c["x_m"] * cm - mida / 2, c["y_m"] * cm - mida / 2
            codi = c["desti_model"] if quin == "model" else real.get(c["cella"])
            if codi is not None:
                eix.add_patch(Rectangle((x0, y0), mida, mida, color=estil.color_del_desti(codi), lw=0))
            if quin == "model" and c["classe"] == "frontera":
                eix.plot(c["x_m"] * cm, c["y_m"] * cm, ".", color="white", ms=2.5)
            if quin == "real" and codi is not None and codi != c["desti_model"]:
                eix.plot(c["x_m"] * cm, c["y_m"] * cm, "x", color="white", ms=4, mew=1.0)
        x_min, x_max, y_min, y_max = (v * cm for v in ZONA_M)
        eix.set_xlim(x_min, x_max)
        eix.set_ylim(y_min, y_max)
        eix.set_aspect("equal")
        eix.set_title(titol)
        eix.set_xlabel("x (cm)")
        eix.grid(False)
    eixos[0].set_ylabel("y (cm)")
    fig.text(0.5, 0.01, "Esquerra: el desti del model al centre de cada cel·la; el punt blanc marca "
             "les de frontera. Dreta: on va acabar la massa; la creu marca els desacords.",
             ha="center", fontsize=8, color=estil.TINTA_2)
    return fig


# ---------------------------------------------------------------------------
# L'autoprova: casos fabricats, on la resposta es sap
# ---------------------------------------------------------------------------
def autoprova() -> int:
    """Q1 la classificacio d'un mapa fabricat (una recta) es la que toca, i la
    fraccio del disc a cada banda es la de l'area d'un segment circular
    Q2 si una b mou la frontera, la cel.la del costat passa a ser de frontera
    Q3 les metriques: tot igual dona el 100 %, i k desacords coneguts donen el
    que toca, amb el p de Fisher de scipy
    Q4 les dues cel.les: 9 de 10 passa i 8 no; 7 de 10 passa i 8 no
    Q5 el CSV mal escrit es rebutja
    Q6 la plantilla es a mida real: 1 unitat de dades = 1 cm de paper
    Q7 reproduible: la classificacio i l'ordre surten iguals dues vegades
    """
    print("=" * 72)
    print(" magnetic.quadricula --autoprova")
    print("=" * 72)
    proves = []
    r, punts = R_DISC_M, PUNTS_DISC

    # Q1 -- una frontera recta a x = xf: desti 1 a l'esquerra i 2 a la dreta.
    # Els centres es desplacen mig pas de la subgraella, perque la recta caigui
    # entre dues columnes de punts i no damunt d'una: si no, aquella columna
    # sencera aniria a una banda i la fraccio s'apartaria un 3 % per construccio.
    xf = 0.02
    mig_pas = r / (punts - 1) / 2

    def recta(despl=0.0):
        return lambda xs, ys: np.where(np.meshgrid(xs, ys)[0] < xf + despl, 1, 2)

    distancies = tuple(d + mig_pas for d in (0.0, 0.5 * r, 0.9 * r, 1.2 * r, 3.0 * r))
    centres = [(xf + d, 0.0) for d in distancies]
    res = classifica(centres, r, punts, [recta()] * 3)
    pitjor_fr, classes_be = 0.0, True
    for d, c in zip(distancies, res):
        segura = d > r
        classes_be &= (c["classe"] == "segura") == segura
        a = math.acos(min(d / r, 1.0))
        exacta = (r * r * a - d * math.sqrt(max(r * r - d * d, 0.0))) / (math.pi * r * r)
        pitjor_fr = max(pitjor_fr, abs(c["fraccions"].get(1, 0.0) - exacta))
    proves.append(("Q1 classes", classes_be, "segura nomes si d > r"))
    proves.append(("Q1 fraccions", pitjor_fr <= 0.03, f"pitjor {pitjor_fr:.3f} contra 0,03"))

    # Q2 -- la mateixa recta, pero amb una de les b que la mou 2 mm
    c = classifica([(xf + 1.2 * r + mig_pas, 0.0)], r, punts, [recta(), recta(0.002), recta()])[0]
    proves.append(("Q2 b que mou", c["classe"] == "frontera", c["classe"]))

    # Q3 -- les metriques amb 40 segures i 40 de frontera fabricades
    cl = {"celles": [{"cella": f"S{k}", "classe": "segura", "desti_model": 1,
                      "fraccions": {1: 1.0}} for k in range(40)]
          + [{"cella": f"R{k}", "classe": "frontera", "desti_model": 2,
              "fraccions": {2: 0.6, 3: 0.4}} for k in range(40)],
          "neta": {"desti_model": 1}, "pols": {"desti_model": 2}}
    bo = {"E1_regla_mm": 100.0, "E2_desviacio_imants_mm": {"1": 0.2, "2": 0.3, "3": 0.1},
          "E2_distancies_mm": {"1-2": 86.6, "2-3": 86.8, "3-1": 86.4}, "E3_origen_mm": 0.5,
          "E4_d0_cm": 2.0}
    iguals = [{"cella": x["cella"], "desti": x["desti_model"], "repeticio": 1} for x in cl["celles"]]
    a = avalua(cl, iguals, bo)
    proves.append(("Q3 tot igual", a["segura"]["fraccio"] == 1.0 and a["frontera"]["fraccio"] == 1.0
                   and a["avaluable"], "100 % i 100 %"))
    k_fro, k_seg = 15, 2
    canviats = [dict(t) for t in iguals]
    for t in canviats[40:40 + k_fro]:
        t["desti"] = 3
    for t in canviats[:k_seg]:
        t["desti"] = 2
    a = avalua(cl, canviats, bo)
    p_bo = float(fisher_exact([[k_fro, 40 - k_fro], [k_seg, 40 - k_seg]], alternative="greater")[1])
    proves.append(("Q3 desacords", a["segura"]["coincidencies"] == 38
                   and a["frontera"]["coincidencies"] == 25 and a["C1"] is True
                   and abs(a["C2"]["p"] - p_bo) < 1e-15 and a["C2"]["passa"],
                   f"38/40, 25/40, p = {a['C2']['p']:.2g}"))
    proves.append(("Q3 C3", abs(a["C3"]["m2_esperada_pel_model"] - 0.6) < 1e-12, "0,6"))
    mal = dict(bo, E3_origen_mm=1.5)
    proves.append(("Q3 E3 fora", not avalua(cl, iguals, mal)["avaluable"], "no avaluable"))
    # La decisio posterior no mou cap numero: nomes afegeix la segona lectura
    amb = avalua(cl, canviats, dict(mal, decisio_avaluable={"qui": "prova", "data": "-"}))
    sense = avalua(cl, canviats, mal)
    proves.append(("Q3 decisio", amb["avaluable_per_decisio"] and not amb["avaluable"]
                   and all(amb[k] == sense[k] for k in ("segura", "frontera", "C1", "C2", "C3")),
                   "nomes canvia la lectura"))

    # Q4 -- les dues cel.les
    def cella(nom, destins_):
        return [{"cella": nom, "desti": d, "repeticio": k + 1} for k, d in enumerate(destins_)]

    def passa(nom, destins_):
        return avalua(cl, cella(nom, destins_), bo)[nom]["passa"]

    proves.append(("Q4 neta", passa("neta", [1] * 9 + [2]) and not passa("neta", [1] * 8 + [2, 3]),
                   "9/10 si, 8/10 no"))
    proves.append(("Q4 pols", passa("pols", [1] * 7 + [2, 3, 0]) and not passa("pols", [1] * 8 + [2, 3]),
                   "7/10 si, 8/10 no"))

    # Q8 -- les dues cel.les tal com es van tirar (3/11): les mateixes regles
    neta = {"classe": "segura", "desti_model": 3, "fraccions": {"3": 1.0}}
    pols = {"classe": "frontera", "desti_model": 1, "fraccions": {"1": 0.34, "2": 0.33, "3": 0.33}}
    pols_desigual = {"classe": "frontera", "desti_model": 1, "fraccions": {"1": 0.6, "2": 0.4}}

    def dc(tipus, pr, destins_, abans=True):
        return avalua_dues_celles([{"punt": "p", "tipus": tipus, "destins": destins_,
                                    "prediccio": pr, "prediccio_abans_de_tirar": abans}])[0]

    q8 = (dc("neta", neta, [3] * 9 + [1])["passa"] and not dc("neta", neta, [3] * 8 + [1, 1])["passa"]
          and dc("pols", pols, [3] * 4 + [2] * 3 + [0] * 3)["passa"]
          and not dc("pols", pols, [1] * 8 + [2, 3])["passa"]
          and dc("pols", pols, [1] * 7 + [2, 3, 0])["aplicable"]
          and not dc("pols", pols_desigual, [1] * 7 + [2, 3, 0])["aplicable"]
          and dc("pols", pols, [1] * 10, abans=False)["prediccio_abans_de_tirar"] is False
          and abs(dc("pols", pols, [1] * 10)["probabilitat_de_fallar_amb_el_model"]
                  - sum(float(binom.sf(POLS_MAXIM, TIRS_CELLA, q)) for q in (0.34, 0.33, 0.33))) < 1e-15)
    proves.append(("Q8 dues cel.les", q8, "D1 i D2 com a Q4, aplicable i abans/despres"))

    # Q5 -- el CSV mal escrit
    prova_csv = Path("esborranys") / "_quadricula_autoprova.csv"
    prova_csv.parent.mkdir(exist_ok=True)
    rebutjats = 0
    for contingut in ("ordre,cella,desti,repeticio,nota\n1,S0,7,1,\n",
                      "ordre,cella,desti,repeticio,nota\n1,Z9,1,1,\n",
                      "ordre,cella,desti,repeticio,nota\n1,S0,1,1,\n2,S0,2,1,\n",
                      "ordre,cella,desti,nota\n1,S0,1,\n"):
        prova_csv.write_text(contingut, encoding="utf-8")
        try:
            llegeix_tirs(prova_csv, {"S0", "neta", "pols"})
        except ValueError:
            rebutjats += 1
    prova_csv.write_text("ordre,cella,desti,repeticio,nota\n1,S0,1,1,\n2,S0,,2,\n", encoding="utf-8")
    llegits = llegeix_tirs(prova_csv, {"S0"})
    prova_csv.unlink()
    proves.append(("Q5 CSV", rebutjats == 4 and len(llegits) == 1, f"{rebutjats} de 4 rebutjats"))

    # Q6 -- la plantilla, a mida real
    p = Parametres.des_de_json()
    fig = fes_plantilla(p, {"x_m": 0.06, "y_m": 0.0}, {"x_m": -0.09, "y_m": 0.0})
    ax = fig.axes[0]
    w, h = fig.get_size_inches() * 2.54
    escala = (w / np.ptp(ax.get_xlim()), h / np.ptp(ax.get_ylim()))
    plt.close(fig)
    proves.append(("Q6 mida real", max(abs(e - 1.0) for e in escala) < 1e-12,
                   f"{escala[0]:.12f} i {escala[1]:.12f} cm per unitat"))

    # Q7 -- reproduible
    dues = [classifica(centres, r, punts, [recta()] * 3) for _ in range(2)]
    ordres = [list(np.random.default_rng(LLAVOR_ORDRE).permutation(len(celles()))) for _ in range(2)]
    proves.append(("Q7 reproduible", dues[0] == dues[1] and ordres[0] == ordres[1], "igual"))

    print(f"\n   {'prova':>16}")
    tot_be = True
    for nom, be, nota in proves:
        tot_be &= bool(be)
        print(f"   {nom:>16}  {'OK' if be else 'MALAMENT'}   {nota}")
    print("\n   " + ("OK -- l'eina classifica, compta i dibuixa el que toca."
                     if tot_be else "MALAMENT -- mira les files de sobre."))
    return 0 if tot_be else 1


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feina", nargs="?", choices=("prepara", "plantilla", "compara", "dues_celles"))
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova l'eina amb casos fabricats")
    ap.add_argument("--fils", type=int, default=6, help="fils de Numba, com a molt (per defecte 6)")
    ap.add_argument("--nomes-quadricula", action="store_true",
                    help="amb 'plantilla': nomes la quadricula, els numeros i la regla")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    if args.feina == "prepara":
        return prepara(args.fils)
    if args.feina == "plantilla":
        return plantilla(args.nomes_quadricula)
    if args.feina == "compara":
        return compara()
    if args.feina == "dues_celles":
        return dues_celles(args.fils)
    ap.error("digue'm 'prepara', 'plantilla', 'compara' o 'dues_celles', o passa --autoprova")


if __name__ == "__main__":
    raise SystemExit(main())
