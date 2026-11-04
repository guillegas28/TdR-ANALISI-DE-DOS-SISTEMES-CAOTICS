r"""Figures de la METODOLOGIA del pendol doble: esquemes de metode, no resultats.

Cap numero de resultat (lambda, C, percentatges) no surt en aquestes figures, i
cap presa real no hi surt al costat d'una simulacio. Si una figura porta una
corba, es il.lustrativa i el peu ho ha de dir. Les del pendol magnetic NO van
aqui: aquell sistema no importa res de pendol/ i tindra el seu propi fitxer.

  esquema_doble  les dues barres A ESCALA, amb les mides de parametres.json
                 (seccio "fisic", la mesurada al banc): pivots, centres de
                 gravetat, les cotes d1, L1, d2 i r_g, els angles theta1 i
                 theta2 des de la vertical inferior i positius en sentit
                 antihorari, la prolongacio de la barra 1 (theta2 NO es mesura
                 des d'alla) i les tres marques del Tracker.
                 La columna del suport s'inspira en el model de Tinkercad
                 (documentacio/*.stl), pero cap mida no en surt: el model te
                 els forats de dalt a 1,6 cm de la punta, i el banc a 3,0.

  cami_dades     el cami del video al format canonic, en caixes: mobil,
                 ffmpeg, comprovacio (amb la tornada «es refa»), Tracker,
                 lectura del .trk (amb els retalls que hi entren i el rebuig
                 que en surt), angles, Savitzky-Golay i format canonic (amb
                 l'amollada, que va nomes al JSON). El filtre i els fps es
                 llegeixen de les metadades de les preses reals, i el llindar
                 de la marca de pendol/tracker.py: res no s'escriu a ma.

  friccio_sec    el fregament sec: el signe contra el tanh(w/eps), amb la
                 franja |w| < eps. Eixos en w/eps i parell/tau_f, perque la
                 eps NO esta decidida i la figura no en pot donar cap valor.
                 La corba suavitzada surt de fisica.parells().

  tres_trams     <ln(d/d0)> contra t: transitori, zona recta i saturacio, amb
                 el terra de soroll. La corba es INVENTADA i sense numeros,
                 pero la zona util i la recta les calcula caos.ajusta_zona(),
                 la regla 2 mateixa.

  angles_desembolicats
                 theta2(t) de la 150_3, UNA PRESA REAL SOLA: plegat a
                 [-180, 180] (amb els salts de 360 de punts) i desembolicat.
                 Els 3 primers segons des de l'amollada del video.

  derivada_vs_filtre
                 omega1 de la 150_3 (la mateixa presa, SOLA): restant
                 fotogrames contra Savitzky-Golay, i a sota la diferencia. Els
                 angles crus surten del .trk (soroll_en_moviment.posicions_del_trk
                 i cadena), i R0 comprova que donen el CSV abans de dibuixar.
  assajos_friccio
                 els dos assajos de friccio, a escala i a la mateixa escala:
                 A, les dues barres rectes, l'articulacio bloquejada amb
                 cinta, girant sobre el pivot 1
                 (rodament 1); B, la barra 1 lligada amb cinta al suport en
                 vertical i nomes la barra 2 oscil.lant (rodament 2).

  feix           NOMES SIMULACIO, sense friccio i sense cap presa real: 30 tirs
                 de simulador.simula des de 90 graus amb una incertesa petita,
                 l'envolupant (la zona de la «prova del model») i l'horitzo on
                 el feix passa de X. X i l'horitzo NO estan firmats: la X es de
                 dibuix i els eixos no porten numeros.

  poincare       NOMES SIMULACIO, sense friccio: una orbita regular des del
                 repos, (a) projectada a (theta1, omega1) i (b) tallada per
                 theta2 = 0 amb omega2 > 0 (caos.seccio_pendol). Sense numeros.

LES TRIES (03/10/2026, en Guillem):
  - mides del banc, i el model 3D nomes com a referencia visual;
  - cotes nomes amb el simbol; els valors van AL PEU, no a la figura (hi va
    haver un requadre i es va treure el mateix 03/10). El programa els
    imprimeix, llegits de parametres.json, per copiar-los al peu;
  - el suport sencer, fins a baix de tot, sense tallar;
  - les tres marques del Tracker, la prolongacio de la barra 1 i una fletxa
    de sentit positiu; sense eixos x-y;
  - colors neutres: el blau i el taronja d'estil.py volen dir simulacio i
    real a tota la memoria, i un esquema no es cap de les dues coses.

LA COMPROVACIO: la posicio del pivot 2 i del gomet dibuixats es compara amb
fisica.posicions(), la funcio que fa servir la simulacio. Si no coincideixen,
el programa s'atura: l'esquema no pot contradir la convencio del codi.

Us:
    .venv\Scripts\python.exe eines\figures_metodologia.py esquema_doble
    .venv\Scripts\python.exe eines\figures_metodologia.py totes

Surt a figures/metodologia/<nom>.png i .pdf.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Wedge  # noqa: E402

from pendol import estil  # noqa: E402
from pendol.fisica import Parametres, posicions  # noqa: E402

SORTIDA = ARREL / "figures" / "metodologia"

# Tintes de l'esquema: totes neutres (vegeu el capdamunt)
FONS_BARRA1 = "#d6d5cd"
FONS_BARRA2 = "#efeee9"
FONS_SUPORT = "#f3f2ee"

# La pose: NOMES IL.LUSTRATIVA. Cap etiqueta no en diu el valor.
THETA1_GRAUS = 35.0
THETA2_GRAUS = 80.0

AMPLADA_BARRA_CM = 4.0  # es torna a llegir de parametres.json ("seccio") a esquema_doble

# Amplada de la figura IMPRESA (cm): mitja pagina, perque al costat hi vagi
# text. Es dibuixa ja a aquesta mida perque la lletra surti a 7-10 punts de
# debo; si s'encongis un PNG mes gran a la memoria, el text no es llegiria.
MIDA_CM = 9.0

# Limits del dibuix de l'esquema del doble (cm). El suport baixa fins a YMIN.
XMIN, XMAX, YMIN, YMAX = -8.2, 24.0, -19.8, 6.0


def _cm(x_m: float) -> str:
    """Metres -> centimetres amb coma decimal i una xifra, com a la memoria."""
    return f"{x_m * 100:.1f}".replace(".", ",")


# ---------------------------------------------------------------- utilitats

def _dir(theta: float) -> np.ndarray:
    """Vector unitari de la barra: des del pivot cap a la punta (y amunt)."""
    return np.array([math.sin(theta), -math.cos(theta)])


def _normal(theta: float) -> np.ndarray:
    """Perpendicular a la barra, girada +90 graus (antihorari) des de _dir."""
    return np.array([math.cos(theta), math.sin(theta)])


def _barra(ax, pivot, theta, de, a, fons, z):
    """Rectangle d'una barra, de `de` a `a` cm al llarg del seu eix."""
    u, n = _dir(theta), _normal(theta)
    h = AMPLADA_BARRA_CM / 2
    cantons = [pivot + de * u + h * n, pivot + a * u + h * n,
               pivot + a * u - h * n, pivot + de * u - h * n]
    ax.add_patch(Polygon(cantons, closed=True, facecolor=fons,
                         edgecolor=estil.TINTA, linewidth=0.9, zorder=z))


def _centre_de_gravetat(ax, centre, r=0.55, z=6):
    """El simbol de quarts alterns, que no es pot confondre amb cap marca."""
    for k in range(4):
        ax.add_patch(Wedge(centre, r, 90 * k, 90 * (k + 1),
                           facecolor=estil.TINTA if k % 2 == 0 else "white",
                           edgecolor=estil.TINTA, linewidth=0.8, zorder=z))


def _eix(ax, centre, z=5):
    """L'eix d'un pivot: un cercle buit (la mida es simbolica)."""
    ax.add_patch(Circle(centre, 0.55, facecolor="white", edgecolor=estil.TINTA,
                        linewidth=1.1, zorder=z))


def _arc_amb_fletxa(ax, centre, r, de_graus, a_graus, z=8, lw=1.0, color=None):
    """Arc de `de` a `a` (graus polars) amb la punta de fletxa al final."""
    color = color or estil.TINTA
    t = np.radians(np.linspace(de_graus, a_graus, 80))
    x = centre[0] + r * np.cos(t)
    y = centre[1] + r * np.sin(t)
    ax.plot(x[:-3], y[:-3], color=color, lw=lw, zorder=z, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((x[-6], y[-6]), (x[-1], y[-1]),
                                 arrowstyle="-|>", mutation_scale=8,
                                 color=color, lw=lw, zorder=z))


def _cota(ax, a, b, n, offset, text, z=7):
    """Cota paral.lela a la barra, del punt a al b, desplacada `offset` cm.

    Les linies d'extensio comencen just fora de la barra (mitja amplada mes un
    marge) i passen una mica la linia de cota.
    """
    inici = AMPLADA_BARRA_CM / 2 + 0.35
    for p in (a, b):
        q0, q1 = p + inici * n, p + (offset + 0.35) * n
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=estil.TINTA_2, lw=0.6, zorder=z)
    pa, pb = a + offset * n, b + offset * n
    ax.add_patch(FancyArrowPatch(pa, pb, arrowstyle="<|-|>", mutation_scale=6,
                                 color=estil.TINTA_2, lw=0.8, zorder=z,
                                 shrinkA=0, shrinkB=0))
    m = (pa + pb) / 2 + 0.75 * n
    ax.text(m[0], m[1], text, ha="center", va="center", fontsize=8.5,
            color=estil.TINTA, zorder=z + 1,
            bbox=dict(boxstyle="round,pad=0.12", fc=estil.SUPERFICIE, ec="none"))


# ------------------------------------------------------------ esquema_doble

def esquema_doble() -> None:
    global AMPLADA_BARRA_CM
    p = Parametres.des_de_json(ARREL / "parametres.json")
    seccio = json.loads((ARREL / "parametres.json").read_text(encoding="utf-8"))["fisic"]["seccio"]
    AMPLADA_BARRA_CM = seccio["amplada"] * 100
    cm = 100.0
    L1, d1, d2, rg = p.L1 * cm, p.d1 * cm, p.d2 * cm, p.r_gomet * cm
    t1, t2 = math.radians(THETA1_GRAUS), math.radians(THETA2_GRAUS)
    u1, u2 = _dir(t1), _dir(t2)
    n1, n2 = _normal(t1), _normal(t2)

    P1 = np.zeros(2)
    P2 = P1 + L1 * u1
    G1 = P1 + d1 * u1
    G2 = P2 + d2 * u2
    GOMET = P2 + rg * u2

    # La comprovacio: el dibuix i la simulacio han de dir el mateix
    x1, y1, x2, y2 = posicions(t1, t2, p)
    if not (np.allclose(P2, [x1 * cm, y1 * cm], atol=1e-9)
            and np.allclose(GOMET, [x2 * cm, y2 * cm], atol=1e-9)):
        raise SystemExit("L'esquema no quadra amb fisica.posicions(): "
                         "algun signe de la convencio esta girat.")

    estil.aplica()
    fig, ax = plt.subplots(figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * (YMAX - YMIN) / (XMAX - XMIN)))
    ax.set_aspect("equal")
    ax.axis("off")

    # El suport: la columna sencera, fins a baix de tot (referencia visual del
    # model 3D; l'amplada es simbolica, no es cap mida del banc)
    ample, dalt = 6.0, 4.2
    contorn = [(-ample / 2, dalt), (ample / 2, dalt), (ample / 2, YMIN), (-ample / 2, YMIN)]
    ax.add_patch(Polygon(contorn, closed=True, facecolor=FONS_SUPORT,
                         edgecolor=estil.EIX, hatch="///", linewidth=0.9, zorder=1))
    ax.text(0, dalt + 0.6, "suport", ha="center", va="bottom",
            fontsize=7, color=estil.TINTA_2)

    # Les barres, a escala: la 1 sobresurt 3 cm per sobre del pivot 1, i la 2
    # 3,1 cm per sobre del pivot 2 (barra*_de surt negatiu)
    _barra(ax, P1, t1, p.barra1_de * cm, p.barra1_a * cm, FONS_BARRA1, z=2)
    _barra(ax, P2, t2, p.barra2_de * cm, p.barra2_a * cm, FONS_BARRA2, z=3)

    # Verticals de referencia i prolongacio de la barra 1. La de dalt es curta
    # perque no arribi a la cota d1, que va a l'esquerra de la barra 1.
    for P, llarg in ((P1, 6.0), (P2, 8.6)):
        ax.plot([P[0], P[0]], [P[1], P[1] - llarg], color=estil.TINTA_2,
                lw=0.9, ls=(0, (5, 3)), zorder=7)
    q = P2 + 7.5 * u1
    ax.plot([P2[0], q[0]], [P2[1], q[1]], color=estil.TINTA_SUAU,
            lw=1.0, ls=(0, (1, 2.2)), zorder=7)

    # Els angles: des de la vertical inferior (-90 graus polars), antihoraris.
    # Van per sobre de les barres: amb 4 cm d'amplada, a sota no es veurien.
    r1, r2 = 4.4, 7.0
    _arc_amb_fletxa(ax, P1, r1, -90, -90 + THETA1_GRAUS)
    _arc_amb_fletxa(ax, P2, r2, -90, -90 + THETA2_GRAUS)
    caixa = dict(boxstyle="round,pad=0.1", fc=estil.SUPERFICIE, ec="none")
    a = math.radians(-90 + THETA1_GRAUS / 2)
    ax.text(P1[0] + (r1 + 1.3) * math.cos(a), P1[1] + (r1 + 1.3) * math.sin(a),
            r"$\theta_1$", ha="center", va="center", fontsize=10, zorder=9, bbox=caixa)
    # theta2, entre la vertical i la prolongacio de la barra 1: aixi es veu
    # que l'arc comenca a la vertical i no a la barra 1
    a = math.radians(-90 + THETA1_GRAUS * 0.5)
    ax.text(P2[0] + (r2 + 1.3) * math.cos(a), P2[1] + (r2 + 1.3) * math.sin(a),
            r"$\theta_2$", ha="center", va="center", fontsize=10, zorder=9, bbox=caixa)

    # Pivots, centres de gravetat i marques del Tracker
    _eix(ax, P1)
    _eix(ax, P2)
    _centre_de_gravetat(ax, G1)
    _centre_de_gravetat(ax, G2)
    marques = dict(markeredgecolor=estil.TINTA, markeredgewidth=0.9, zorder=10,
                   linestyle="none")
    ax.plot(*P1, marker="s", markersize=3.8, markerfacecolor="white", **marques)
    ax.plot(*P2, marker="^", markersize=4.6, markerfacecolor="white", **marques)
    ax.plot(*GOMET, marker="o", markersize=4.8, markerfacecolor=estil.TINTA, **marques)

    # Cotes: la barra 1 a l'esquerra (-n1) i la 2 per sobre (+n2), perque els
    # angles, que sempre queden al costat de la vertical, no les trepitgin
    _cota(ax, P1, G1, -n1, 3.6, r"$d_1$")
    _cota(ax, P1, P2, -n1, 5.4, r"$L_1$")
    _cota(ax, P2, G2, n2, 3.4, r"$d_2$")
    _cota(ax, P2, GOMET, n2, 5.3, r"$r_g$")

    # Noms
    def nom(xy, txt, dx, dy, ha="left"):
        ax.annotate(txt, xy=xy, xytext=(xy[0] + dx, xy[1] + dy), fontsize=7.5,
                    color=estil.TINTA, ha=ha, va="center", zorder=11,
                    bbox=dict(boxstyle="round,pad=0.15", fc=estil.SUPERFICIE, ec="none"),
                    arrowprops=dict(arrowstyle="-", color=estil.TINTA_SUAU,
                                    lw=0.6, shrinkA=1, shrinkB=4))
    nom(P1, "pivot 1", 6.0, 1.6)
    nom(P2, "pivot 2\n(articulació)", -6.4, -6.6, ha="center")
    for P, th, s, de_costat, txt in ((P1, t1, 3.0, 0.9, "barra 1"),
                                     (P2, t2, 9.6, 0.0, "barra 2")):
        c = P + s * _dir(th) + de_costat * _normal(th)
        ax.text(c[0], c[1], txt, rotation=math.degrees(th) - 90, rotation_mode="anchor",
                ha="center", va="center", fontsize=7, color=estil.TINTA_2, zorder=4)

    # Sentit positiu: a dalt a l'esquerra, al costat del suport
    cf = np.array([-5.7, 3.6])
    _arc_amb_fletxa(ax, cf, 1.2, -40, 230)
    ax.text(*cf, "+", ha="center", va="center", fontsize=9, color=estil.TINTA)
    ax.text(cf[0], cf[1] - 1.9, "sentit\npositiu", ha="center", va="top",
            fontsize=7, color=estil.TINTA_2, linespacing=1.1)

    # Llegenda de les marques del Tracker, dibuixada a ma, a dalt a la dreta
    lx, ly = 11.8, 4.6
    ax.text(lx, ly + 1.45, "Marques del Tracker", fontsize=7.5, fontweight="bold",
            color=estil.TINTA, va="center")
    for k, (mk, fc, txt) in enumerate((("s", "white", "pivot 1"),
                                       ("^", "white", "articulació"),
                                       ("o", estil.TINTA, "gomet de la barra 2"))):
        y = ly - 1.3 * k
        ax.plot(lx + 0.4, y, marker=mk, markersize=4.6 if mk != "s" else 3.8,
                markerfacecolor=fc, **marques)
        ax.text(lx + 1.3, y, txt, fontsize=7, color=estil.TINTA_2, va="center")
    # El centre de gravetat NO es cap marca del Tracker: va separat
    _centre_de_gravetat(ax, (lx + 0.4, ly - 4.7), r=0.42)
    ax.text(lx + 1.3, ly - 4.7, "centre de gravetat", fontsize=7,
            color=estil.TINTA_2, va="center")

    # Les mides mesurades NO van a la figura (en Guillem, 03/10): van al peu.
    # S'imprimeixen aqui, llegides de parametres.json, per copiar-les-hi.
    print("  per al peu: "
          f"d1 = {_cm(p.d1)} cm, L1 = {_cm(p.L1)} cm, d2 = {_cm(p.d2)} cm, "
          f"r_g = {_cm(p.r_gomet)} cm; barres {_cm(p.R1)} × "
          f"{_cm(seccio['amplada'])} × {_cm(seccio['gruix'])} cm")

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    estil.desa(fig, "esquema_doble", directori=SORTIDA)


# --------------------------------------------------------------- cami_dades

def _parametres_del_cami() -> dict:
    """Els numeros que surten a la figura 3, llegits i no escrits a ma.

    El filtre i la freqüència, de les metadades de les preses REALS de
    dades/net/ (noms com 45_3.json; ni prova ni magnetic); si no tots diuen el
    mateix filtre, la figura no pot dir-ne un i s'atura. El llindar de la
    marca, de pendol/tracker.py.
    """
    import re
    import statistics

    from pendol import tracker

    filtres, fps = set(), []
    for cami in sorted((ARREL / "dades" / "net").glob("*.json")):
        if not re.fullmatch(r"\d+_\d+", cami.stem):
            continue
        m = json.loads(cami.read_text(encoding="utf-8"))
        filtres.add((m["savgol"]["finestra_mostres"], m["savgol"]["ordre"]))
        fps.append(m["fps"])
    if len(filtres) != 1:
        raise SystemExit(f"Les preses no fan servir totes el mateix filtre: {filtres}")
    finestra, ordre = filtres.pop()
    return {"finestra": finestra, "ordre": ordre, "n_preses": len(fps),
            "fps": round(statistics.median(fps)),
            "marca_mm": tracker.MARCA_A_LEIX_MM}


def cami_dades() -> None:
    from matplotlib.patches import FancyBboxPatch

    q = _parametres_del_cami()
    marca = f"{q['marca_mm']:g}".replace(".", ",")
    print(f"  del codi: Savitzky-Golay de {q['finestra']} fotogrames, ordre {q['ordre']} "
          f"({q['n_preses']} preses); ~{q['fps']} fps; marca a l'eix si |d| > {marca} mm")

    # (titol, detall) de cada caixa del cami principal, i la franja on va
    caixes = [
        # MINIMALISTA (en Guillem, 03/10): titol i UNA linia
        ("Mòbil", f"HEVC · ≈{q['fps']} fps · freqüència variable", "vídeo"),
        ("ffmpeg", "H.264 · freqüència constant", "vídeo"),
        ("Comprovació", "cap fotograma perdut ni duplicat", "vídeo"),
        ("Tracker", "marques fix · rodament · extrem", "Tracker"),
        ("Lectura del .trk", f"articulació a l'eix si |δ| > {marca} mm", "programa"),
        ("Angles", "des de la vertical, desembolicats", "programa"),
        ("Savitzky-Golay", f"{q['finestra']} fotogrames, ordre {q['ordre']} → θ i ω", "programa"),
        ("CSV canònic", "t, θ₁, θ₂, ω₁, ω₂", "programa"),
    ]

    X0, X1 = 1.0, 6.9          # caixes del cami principal
    S0, S1 = 7.2, 10.0         # caixes del costat
    LINIA, PAD, BUIT = 0.36, 0.24, 0.36
    alts = [PAD * 2 + LINIA * (1 + det.count("\n") + 1) for _, det, _ in caixes]
    tops, y = [], 0.0
    for h in alts:
        tops.append(y)
        y -= h + BUIT
    ymin = y + BUIT - 0.35

    estil.aplica()
    ample = MIDA_CM / 2.54
    fig, ax = plt.subplots(figsize=(ample, ample * (0.35 - ymin) / 10.2))
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(ymin, 0.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Les tres etapes (quina eina fa cada pas): una linia fina i el nom
    for nom in ("vídeo", "Tracker", "programa"):
        idx = [i for i, c in enumerate(caixes) if c[2] == nom]
        dalt = tops[idx[0]] - 0.05
        baix = tops[idx[-1]] - alts[idx[-1]] + 0.05
        ax.plot([0.72, 0.72], [baix, dalt], color=estil.EIX, lw=1.0,
                solid_capstyle="butt")
        ax.text(0.42, (dalt + baix) / 2, nom, rotation=90, ha="center", va="center",
                fontsize=6.5, color=estil.TINTA_SUAU)

    def caixa(x0, x1, top, h, titol, detall, auxiliar=False):
        ax.add_patch(FancyBboxPatch((x0, top - h), x1 - x0, h,
                                    boxstyle="round,pad=0,rounding_size=0.15",
                                    facecolor="white", zorder=2,
                                    edgecolor=estil.TINTA_SUAU if auxiliar else estil.TINTA,
                                    lw=0.7 if auxiliar else 0.9,
                                    ls=(0, (3, 2)) if auxiliar else "-"))
        xc = (x0 + x1) / 2
        ax.text(xc, top - PAD, titol, ha="center", va="top", fontsize=6.9,
                fontweight="bold", color=estil.TINTA, zorder=3)
        ax.text(xc, top - PAD - LINIA * 1.1, detall, ha="center", va="top", fontsize=5.9,
                color=estil.TINTA_2, linespacing=1.25, zorder=3)

    def fletxa(a, b, z=4, **kw):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                     color=estil.TINTA_2, lw=0.8, zorder=z,
                                     shrinkA=0, shrinkB=0, **kw))

    xc = (X0 + X1) / 2
    for i, ((titol, detall, _), top, h) in enumerate(zip(caixes, tops, alts)):
        caixa(X0, X1, top, h, titol, detall)
        if i:
            fletxa((xc, tops[i - 1] - alts[i - 1]), (xc, top))

    def mig(i):
        return tops[i] - alts[i] / 2

    # La tornada: si la comprovacio falla, es refa la conversio
    xr = X1 + 0.22
    ax.plot([X1, xr, xr], [mig(2), mig(2), mig(1)], color=estil.TINTA_2, lw=0.8, zorder=4)
    fletxa((xr, mig(1)), (X1, mig(1)))
    ax.text(xr + 0.15, (mig(1) + mig(2)) / 2, "si falla", ha="left",
            va="center", fontsize=6.3, color=estil.TINTA_2, style="italic")

    # Els retalls (entren a la lectura) i el rebuig (en surt)
    # Fletxes RECTES, directes (en Guillem, 03/10), i dues linies de detall
    h_aux = PAD * 2 + LINIA * 3
    sc = (S0 + S1) / 2
    caixa(S0, S1, tops[3] + 0.1, h_aux, "retalls.json", "on comença\ncada presa",
          auxiliar=True)
    fletxa((sc, tops[3] + 0.1 - h_aux), (X1, tops[4] - 0.25))
    caixa(S0, S1, tops[5] - 0.1, h_aux, "rebuig", "si al final no\nqueda quieta",
          auxiliar=True)
    fletxa((X1, tops[4] - alts[4] + 0.25), (sc, tops[5] - 0.1))

    # L'amollada: es mesura sobre el CSV i s'escriu NOMES al JSON
    caixa(S0, S1, mig(7) + h_aux / 2, h_aux, "amollada", "zero del rellotge,\nnomés al JSON",
          auxiliar=True)
    fletxa((S0, mig(7)), (X1, mig(7)))

    estil.desa(fig, "cami_dades", directori=SORTIDA)


# ------------------------------------------------------------- friccio_sec

def friccio_sec() -> None:
    """El fregament sec: el signe (discontinu a w = 0) contra el tanh(w/eps).

    IL.LUSTRATIVA: la eps del treball NO esta decidida. Per aixo els eixos
    van en w/eps i en parell/tau_f, i la figura no porta cap valor de cap
    dels dos. La corba suavitzada no es dibuixa amb un tanh escrit aqui: la
    calcula fisica.parells(), el parell que fa servir el simulador, amb un
    sol terme ences (tau_f1 = 1, eps = 1, i la resta a zero).
    """
    from pendol.fisica import Dissipacio, parells

    p = Parametres.des_de_json(ARREL / "parametres.json")
    x = np.linspace(-4, 4, 801)  # w/eps
    nomes_sec = Dissipacio(tau_f1=1.0, b1=0.0, tau_f2=0.0, b2=0.0, C=0.0, eps=1.0)
    # w1 = w2 = w: el rodament 2 no gira (dw = 0) i nomes hi queda el sec del 1
    suau = np.array([parells([0.0, 0.0, w, w], p, nomes_sec)[0] for w in x])
    if not np.allclose(suau, -np.tanh(x), atol=1e-15):
        raise SystemExit("fisica.parells() no dona -tau_f*tanh(w/eps): la figura "
                         "no dibuixaria el que fa el simulador.")

    estil.aplica()
    fig, ax = plt.subplots(figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.62))

    # La franja |w| < eps, on els dos models no s'assemblen
    ax.axvspan(-1, 1, color="#ecebe5", lw=0, zorder=0)
    ax.text(0, 1.32, r"$|\omega| < \varepsilon$", ha="center", va="center",
            fontsize=7.5, color=estil.TINTA_2)

    # El signe: dues branques planes i el salt a w = 0 (punts)
    ax.plot(x[x < 0], np.ones((x < 0).sum()), color=estil.TINTA, lw=1.6,
            label=r"signe:  $-\tau_f\,\mathrm{sgn}(\omega)$", zorder=3)
    ax.plot(x[x > 0], -np.ones((x > 0).sum()), color=estil.TINTA, lw=1.6, zorder=3)
    ax.plot([0, 0], [-1, 1], color=estil.TINTA, lw=0.8, ls=(0, (1, 1.5)), zorder=3)
    # El suavitzat, discontinu: es llegeix igual en blanc i negre
    ax.plot(x, suau, color=estil.TINTA_2, lw=1.6, ls=(0, (4, 2)),
            label=r"suavitzat:  $-\tau_f\,\tanh(\omega/\varepsilon)$", zorder=4)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([-4, -2, -1, 0, 1, 2, 4])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"$\omega\,/\,\varepsilon$", fontsize=8)
    ax.set_ylabel(r"parell $/\,\tau_f$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", fontsize=6.8, handlelength=2.6)
    estil.desa(fig, "friccio_sec", directori=SORTIDA)


# --------------------------------------------------------------- tres_trams

def tres_trams() -> None:
    """Els tres trams de <ln(d/d0)>: transitori, zona recta i saturacio.

    IL.LUSTRATIVA: la corba es INVENTADA (un inici suau, creixement
    exponencial, saturacio logistica i un terra de soroll sumat en
    quadratura), sense cap numero als eixos. Pero la zona i la recta NO es
    dibuixen a ull: les calcula caos.ajusta_zona(), la regla 2 mateixa
    (aprovada el 03/10), amb els seus factors (3 x terra, sostre / 10, un
    factor e, el doble del temps). Si la regla canvia, la figura la segueix.
    """
    from pendol import caos

    # La corba d'exemple (unitats arbitraries; res d'aixo no es cap mesura)
    t = np.linspace(0.0, 10.0, 2001)
    lam, t0, k = 1.0, 1.4, 2.5
    terra, d0 = 1.0, 2.0
    sostre = d0 * math.exp(7.0)
    s = np.log1p(np.exp(k * (t - t0))) / k           # inici suau: el transitori
    e = np.exp(lam * s)
    neta = d0 * e / (1.0 + d0 * (e - 1.0) / sostre)  # saturacio logistica
    d = np.sqrt(neta**2 + terra**2)                  # el terra, en quadratura
    ln_d = np.log(d)
    y = ln_d - math.log(d0)                          # <ln(d/d0)>

    r = caos.ajusta_zona(t, y, ln_d, terra, sostre)
    if not r["ok"]:
        raise SystemExit(f"La corba d'exemple no passa la regla 2: {r['motiu']}")
    ti, tf = r["t_inici_s"], r["t_fi_s"]
    a = r["pendent"]
    b = float(np.mean(y[(t >= ti) & (t <= tf)]) - a * np.mean(t[(t >= ti) & (t <= tf)]))

    estil.aplica()
    fig, ax = plt.subplots(figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.66))

    nivells = [(math.log(terra / d0), "terra"),
               (math.log(caos.FACTOR_TERRA * terra / d0),
                f"{caos.FACTOR_TERRA:g} × terra"),
               (math.log(sostre / caos.FACTOR_SOSTRE / d0),
                f"sostre / {caos.FACTOR_SOSTRE:g}"),
               (math.log(sostre / d0), "sostre")]
    # La zona util, en gris; per sota del terra, ratllat (alla no es veu res)
    ax.axhspan(nivells[1][0], nivells[2][0], color="#ecebe5", lw=0, zorder=0)
    ax.axhspan(-1.6, nivells[0][0], facecolor="none", edgecolor=estil.EIX, hatch="////",
               lw=0, zorder=0)
    for yv, nom in nivells:
        ax.axhline(yv, color=estil.TINTA_SUAU, lw=0.7, ls=(0, (1, 2)), zorder=1)
        ax.text(10.15, yv, nom, ha="left", va="center", fontsize=6.8, color=estil.TINTA_2,
                clip_on=False)

    # Els limits dels tres trams
    for tv in (ti, tf):
        ax.axvline(tv, color=estil.EIX, lw=0.7, zorder=1)
    ytop = nivells[3][0] + 0.75
    for x0, x1, nom in ((0.0, ti, "transitori"), (ti, tf, "zona recta"), (tf, 10.0, "saturació")):
        ax.text((x0 + x1) / 2, ytop, nom, ha="center", va="center", fontsize=7,
                color=estil.TINTA_2, fontweight="bold")

    # La corba, i la recta de la regla 2 sobre la zona (una mica allargada)
    ax.plot(t, y, color=estil.TINTA, lw=1.5, zorder=3)
    # Dins de la zona la recta i la corba coincideixen; allargada, es veu on
    # la corba se n'aparta (el transitori i la saturacio)
    tl = np.array([ti - 1.0, tf + 1.3])
    ax.plot(tl, a * tl + b, color=estil.TINTA_2, lw=1.2, ls=(0, (4, 2)), zorder=4)
    tm = (ti + tf) / 2
    ax.text(tm + 0.25, a * tm + b - 0.35, r"pendent $=\lambda$", ha="left", va="top",
            fontsize=7.5, color=estil.TINTA, zorder=5,
            bbox=dict(boxstyle="round,pad=0.12", fc="#ecebe5", ec="none"))

    ax.set_xlim(0, 10)
    ax.set_ylim(-1.6, ytop + 0.5)
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.grid(False)
    ax.set_xlabel(r"$t$, des de l'amollada", fontsize=8)
    ax.set_ylabel(r"$\langle \ln(d/d_0) \rangle$", fontsize=8)
    ax.tick_params(labelsize=7)
    estil.desa(fig, "tres_trams", directori=SORTIDA)


# ------------------------------------------------------ angles_desembolicats

PRESA_ANGLES = "150_3"   # presa real d'exemple, SOLA (sense cap simulacio)
FINESTRA_ANGLES_S = (0.0, 3.0)  # des de l'amollada del video; en 6 s els
                                # salts quedaven massa apilats


def angles_desembolicats() -> None:
    """theta2(t) d'una presa real: plegat a [-180, 180] contra desembolicat.

    UNA PRESA REAL SOLA, per il.lustrar un pas del processament (permes: cap
    simulacio al costat). El zero es l'amollada del VIDEO del JSON, mai
    l'afinada. El CSV canonic ja es desembolicat (contracte de dades): el
    programa ho comprova, i el plegat es fabrica aqui a partir d'ell.
    """
    from pendol import dades

    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{PRESA_ANGLES}.csv")
    t = taula["t"].to_numpy() - meta["amollada"]["amollada_s"]
    th2 = taula["theta2"].to_numpy()
    if np.max(np.abs(np.diff(th2))) > math.pi:
        raise SystemExit(f"La {PRESA_ANGLES} no ve desembolicada: el contracte diu que ho ha de ser.")
    dins = (t >= FINESTRA_ANGLES_S[0]) & (t <= FINESTRA_ANGLES_S[1])
    t, des = t[dins], np.degrees(th2[dins])
    ple = (des + 180.0) % 360.0 - 180.0
    salts = np.nonzero(np.abs(np.diff(ple)) > 180.0)[0]
    print(f"  {PRESA_ANGLES}: {len(salts)} salts de 360 graus de {FINESTRA_ANGLES_S[0]:g} "
          f"a {FINESTRA_ANGLES_S[1]:g} s; theta2 va de {des.min():.0f} a {des.max():.0f} graus")

    estil.aplica()
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True,
                                 figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.85),
                                 gridspec_kw=dict(height_ratios=[1, 1.35], hspace=0.12))

    # Plegat: es talla la linia a cada salt i el salt es marca de punts
    y = ple.copy().astype(float)
    tt = t.copy()
    y = np.insert(y, salts + 1, np.nan)
    tt = np.insert(tt, salts + 1, np.nan)
    a1.plot(tt, y, color=estil.REAL, lw=1.1, ls=estil.TRAC_REAL)
    for k in salts:
        a1.plot([t[k], t[k + 1]], [ple[k], ple[k + 1]], color=estil.TINTA_SUAU,
                lw=0.6, ls=(0, (1, 1.5)))
    a1.set_ylim(-200, 200)
    a1.set_yticks([-180, 0, 180])
    a1.set_yticklabels(["−180°", "0°", "180°"])
    a1.set_ylabel(r"$\theta_2$ plegat", fontsize=8)

    # Desembolicat: continu
    a2.plot(t, des, color=estil.REAL, lw=1.1, ls=estil.TRAC_REAL)
    marques = np.arange(360 * math.floor(des.min() / 360), des.max() + 360, 360)
    a2.set_yticks(marques)
    a2.set_yticklabels([f"{m:.0f}°".replace("-", "−") for m in marques])
    a2.set_ylabel(r"$\theta_2$ desembolicat", fontsize=8)
    a2.set_xlabel("t (s), des de l'amollada", fontsize=8)
    for a in (a1, a2):
        a.tick_params(labelsize=7)
    a2.text(0.02, 0.05, f"presa {PRESA_ANGLES}", transform=a2.transAxes,
            ha="left", va="bottom", fontsize=6.8, color=estil.TINTA_2)
    estil.desa(fig, "angles_desembolicats", directori=SORTIDA)


# ------------------------------------------------------- derivada_vs_filtre

FINESTRA_DERIVADA_S = (0.85, 1.15)  # des de l'amollada: un tros curt, perque
                                    # es vegi cada fotograma. El soroll de la
                                    # derivada crua es nomes un 1-4 % del
                                    # senyal (mesurat el 03/10), i per aixo
                                    # hi ha el panell de la diferencia


def derivada_vs_filtre() -> None:
    """omega1 d'una presa real: restant fotogrames contra Savitzky-Golay.

    UNA PRESA REAL SOLA (la de la figura 6), cap simulacio. El CSV ja desa els
    angles FILTRATS, o sigui que la derivada crua no en pot sortir: els angles
    sense filtrar es tornen a llegir del .trk amb les funcions de
    eines/soroll_en_moviment.py (posicions_del_trk i cadena), i la seva R0
    comprova que, passats pel filtre de la conversio, donen el CSV. Si no, la
    figura no es fa.
    """
    import contextlib
    import io

    from scipy.signal import savgol_filter

    sys.path.insert(0, str(ARREL / "eines"))
    import soroll_en_moviment as sem
    from pendol import dades

    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{PRESA_ANGLES}.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        t_fot, J, G, corregida = sem.posicions_del_trk(meta)
    c = sem.cadena(t_fot, J, G, bool(meta.get("signe_invertit", False)))
    r0 = sem.comprova_r0(c, meta, taula, corregida)
    if not r0["ok"]:
        raise SystemExit(f"Els angles del .trk no reprodueixen el CSV de la {PRESA_ANGLES}: {r0}")
    fin, ordre = int(meta["savgol"]["finestra_mostres"]), int(meta["savgol"]["ordre"])
    fps = c["fps"]
    th1 = c["th"][0]
    w_filtre = savgol_filter(th1, fin, ordre, deriv=1, delta=1.0 / fps)  # = omega1 del CSV
    w_cru = np.diff(th1) * fps                                          # restant fotogrames
    t = c["t"] - meta["amollada"]["amollada_s"]
    t_mig = (t[:-1] + t[1:]) / 2
    sel = (t >= FINESTRA_DERIVADA_S[0]) & (t <= FINESTRA_DERIVADA_S[1])
    sel_c = (t_mig >= FINESTRA_DERIVADA_S[0]) & (t_mig <= FINESTRA_DERIVADA_S[1])
    print(f"  {PRESA_ANGLES}: R0 ok; {sel_c.sum()} fotogrames; desviacio tipica de la "
          f"diferencia cru - filtrat: {np.std(w_cru[sel_c] - np.interp(t_mig[sel_c], t, w_filtre)):.2f} rad/s")

    estil.aplica()
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,
                                  figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.85),
                                  gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.12))
    ax.plot(t_mig[sel_c], w_cru[sel_c], color=estil.TINTA_SUAU, lw=0.6, marker="o",
            markersize=2.0, label="restant fotogrames")
    ax.plot(t[sel], w_filtre[sel], color=estil.REAL, lw=1.8, ls=estil.TRAC_REAL,
            label=f"Savitzky-Golay ({fin} fotogrames, ordre {ordre})")
    ax.set_ylabel(r"$\omega_1$ (rad/s)", fontsize=8)
    ax.legend(loc="lower left", fontsize=6.6)
    ax.text(0.98, 0.95, f"presa {PRESA_ANGLES}", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.8, color=estil.TINTA_2)

    # El que el filtre treu: la diferencia, fotograma a fotograma
    dif = w_cru[sel_c] - np.interp(t_mig[sel_c], t, w_filtre)
    ax2.axhline(0, color=estil.EIX, lw=0.8)
    ax2.plot(t_mig[sel_c], dif, color=estil.TINTA_SUAU, lw=0.6, marker="o", markersize=2.0)
    ax2.set_ylabel("diferència\n(rad/s)", fontsize=8)
    ax2.set_xlabel("t (s), des de l'amollada", fontsize=8)
    ax2.set_xlim(*FINESTRA_DERIVADA_S)
    for a in (ax, ax2):
        a.tick_params(labelsize=7)
    estil.desa(fig, "derivada_vs_filtre", directori=SORTIDA)


# ---------------------------------------------------------- assajos_friccio

ANGLE_ASSAJOS_GRAUS = 25.0  # nomes il.lustratiu


def assajos_friccio() -> None:
    """Els dos assajos de friccio del doble, a escala i a la MATEIXA escala.

    A, el conjunt lligat (presa Friccio_doble_1): les dues barres en linia
       recta, amb l'articulacio bloquejada amb cinta a la zona del rodament,
       girant juntes sobre el pivot 1 -> el rodament 1.
    B, la barra 2 sola (presa Friccio_doble_2): la barra 1 HI ES, pero lligada
       amb cinta al suport en vertical perque no es mogui, i nomes oscil.la la
       barra 2 sobre el pivot 2 -> el rodament 2 (en Guillem, 03/10; una
       primera versio la dibuixava sense la barra 1, i no era aixi).
    El muntatge descartat (treure la barra 2 per mesurar el rodament 1) no hi
    surt, a posta: el motiu va al text.
    """
    global AMPLADA_BARRA_CM
    p = Parametres.des_de_json(ARREL / "parametres.json")
    seccio = json.loads((ARREL / "parametres.json").read_text(encoding="utf-8"))["fisic"]["seccio"]
    AMPLADA_BARRA_CM = seccio["amplada"] * 100
    cm = 100.0
    th = math.radians(ANGLE_ASSAJOS_GRAUS)
    u = _dir(th)

    estil.aplica()
    ymin, ymax = -(p.L1 + p.L_extrem) * cm - 2.5, 7.0
    xa_lims, xb_lims = (-6.5, 15.5), (-6.5, 14.5)
    amples = [xa_lims[1] - xa_lims[0], xb_lims[1] - xb_lims[0]]
    alt_cm = MIDA_CM * (ymax - ymin) / sum(amples)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(MIDA_CM / 2.54, alt_cm / 2.54),
                                   gridspec_kw=dict(width_ratios=amples, wspace=0.0))

    def suport(ax):
        ax.add_patch(Polygon([(-3, 4.2), (3, 4.2), (3, ymin), (-3, ymin)], closed=True,
                             facecolor=FONS_SUPORT, edgecolor=estil.EIX, hatch="///",
                             linewidth=0.9, zorder=1))

    def angle(ax, P, r):
        ax.plot([P[0], P[0]], [P[1], P[1] - r - 1.5], color=estil.TINTA_2, lw=0.8,
                ls=(0, (5, 3)), zorder=7)
        _arc_amb_fletxa(ax, P, r, -90, -90 + ANGLE_ASSAJOS_GRAUS)
        a = math.radians(-90 + ANGLE_ASSAJOS_GRAUS / 2)
        ax.text(P[0] + (r + 1.6) * math.cos(a), P[1] + (r + 1.6) * math.sin(a), r"$\theta$",
                ha="center", va="center", fontsize=9, zorder=9,
                bbox=dict(boxstyle="round,pad=0.1", fc=estil.SUPERFICIE, ec="none"))

    def rodament(ax, P, nom, dx, dy=2.6, ha="center"):
        # el rodament que es mesura: un anell gruixut al voltant de l'eix
        ax.add_patch(Circle(P, 0.95, facecolor="none", edgecolor=estil.TINTA, lw=2.0,
                            zorder=8))
        ax.annotate(nom, xy=P, xytext=(P[0] + dx, P[1] + dy), fontsize=7.2,
                    ha=ha, va="bottom", color=estil.TINTA, zorder=11,
                    bbox=dict(boxstyle="round,pad=0.1", fc=estil.SUPERFICIE, ec="none"),
                    arrowprops=dict(arrowstyle="-", color=estil.TINTA_SUAU, lw=0.6,
                                    shrinkA=1, shrinkB=6))

    # --- A: el conjunt lligat, recte, sobre el pivot 1
    P1 = np.zeros(2)
    P2 = P1 + p.L1 * cm * u
    suport(axa)
    _barra(axa, P1, th, p.barra1_de * cm, p.barra1_a * cm, FONS_BARRA1, z=2)
    _barra(axa, P2, th, p.barra2_de * cm, p.barra2_a * cm, FONS_BARRA2, z=3)
    _eix(axa, P1)
    _eix(axa, P2)
    # l'articulacio, bloquejada amb CINTA a la zona del rodament (en Guillem,
    # 03/10; abans hi deia «cargolades», i no era aixi): dues tires que
    # travessen les dues barres, una a cada banda del pivot 2
    n = _normal(th)
    for s in (-1.6, 1.6):
        c0 = P2 + s * u
        ax_ = AMPLADA_BARRA_CM / 2 + 0.7
        axa.add_patch(Polygon([c0 - ax_ * n - 0.45 * u, c0 + ax_ * n - 0.45 * u,
                               c0 + ax_ * n + 0.45 * u, c0 - ax_ * n + 0.45 * u],
                              closed=True, facecolor="#d6d5cd", edgecolor=estil.TINTA_2,
                              lw=0.6, zorder=6))
    axa.annotate("cinta", xy=P2 + 1.6 * u + (AMPLADA_BARRA_CM / 2 + 0.5) * n,
                 xytext=(P2[0] + 5.5, P2[1] + 0.5), fontsize=6.8, ha="left", va="center",
                 color=estil.TINTA_2,
                 arrowprops=dict(arrowstyle="-", color=estil.TINTA_SUAU, lw=0.6,
                                 shrinkA=1, shrinkB=2))
    angle(axa, P1, 5.0)
    rodament(axa, P1, "rodament 1", 6.5)
    axa.set_title("A · conjunt lligat", fontsize=8, loc="left")

    # --- B: la barra 1 vertical, lligada amb cinta al suport; nomes oscil.la
    # la barra 2 sobre el pivot 2
    suport(axb)
    Q2 = np.array([0.0, -p.L1 * cm])
    _barra(axb, np.zeros(2), 0.0, p.barra1_de * cm, p.barra1_a * cm, FONS_BARRA1, z=2)
    _barra(axb, Q2, th, p.barra2_de * cm, p.barra2_a * cm, FONS_BARRA2, z=3)
    _eix(axb, np.zeros(2))
    _eix(axb, Q2)
    # la cinta: dues tires que travessen la barra 1 i la columna
    for yc in (-2.3, -7.0):
        axb.add_patch(Polygon([(-3.4, yc - 0.55), (3.4, yc + 0.25), (3.4, yc + 1.15),
                               (-3.4, yc + 0.35)], closed=True, facecolor="#d6d5cd",
                              edgecolor=estil.TINTA_2, lw=0.6, zorder=4))
    # l'etiqueta «cinta» nomes va al panell A (en Guillem, 03/10)
    angle(axb, Q2, 5.0)
    rodament(axb, Q2, "rodament 2", 5.0, dy=-4.5, ha="left")
    axb.set_title("B · només la barra 2", fontsize=8, loc="left")

    for ax, (x0, x1) in ((axa, xa_lims), (axb, xb_lims)):
        ax.set_xlim(x0, x1)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
    estil.desa(fig, "assajos_friccio", directori=SORTIDA)


# --------------------------------------------------------------------- feix

FEIX_NOMINAL_GRAUS = 90.0   # theta1 = theta2, amollat del repos
FEIX_SIGMA_GRAUS = 0.3      # nomes il.lustratiu
FEIX_MEMBRES = 30
FEIX_TEMPS_S = 2.5
FEIX_LLAVOR = 0
FEIX_X_DIBUIX_GRAUS = 35.0  # NOMES PER DIBUIXAR: la X del treball NO esta
                            # firmada, i per aixo la figura no porta numeros


def feix() -> None:
    """El feix: NOMES SIMULACIO, sense friccio, sense cap presa real.

    30 tirs del simulador de debo (simulador.simula), del nominal i de
    pertorbacions normals com fa simulador.py --conjunt (el tir 0 es el
    nominal). L'envolupant (min-max) es la zona de la «prova del model»;
    l'horitzo, on el feix s'eixampla mes que X. La X es il.lustrativa i els
    eixos no porten numeros: ni X ni l'horitzo no estan firmats.
    """
    from pendol.simulador import simula

    p = Parametres.des_de_json(ARREL / "parametres.json")
    rng = np.random.default_rng(FEIX_LLAVOR)
    sigma = math.radians(FEIX_SIGMA_GRAUS)
    e0 = np.array([math.radians(FEIX_NOMINAL_GRAUS)] * 2 + [0.0, 0.0])
    th1 = []
    for i in range(FEIX_MEMBRES):
        e = e0.copy()
        if i > 0:
            e[0] += rng.normal(0.0, sigma)
            e[1] += rng.normal(0.0, sigma)
        t, y = simula(p, e, FEIX_TEMPS_S, fps=200.0)
        th1.append(np.degrees(y[0]))
    th1 = np.array(th1)
    baix, dalt = th1.min(axis=0), th1.max(axis=0)
    obert = np.nonzero(dalt - baix > FEIX_X_DIBUIX_GRAUS)[0]
    if not len(obert):
        raise SystemExit("El feix no s'obre prou dins del temps dibuixat.")
    th = t[obert[0]]

    estil.aplica()
    fig, ax = plt.subplots(figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.62))
    ax.fill_between(t, baix, dalt, color=estil.FEIX, alpha=0.35, lw=0, zorder=1)
    for k in range(1, FEIX_MEMBRES):
        ax.plot(t, th1[k], color=estil.FEIX, lw=0.6, zorder=2)
    ax.plot(t, th1[0], color=estil.SIMULACIO, lw=1.3, ls=estil.TRAC_SIMULACIO, zorder=3,
            label="tir nominal")
    ax.plot([], [], color=estil.FEIX, lw=0.8, label="feix de simulacions")

    # L'horitzo: on l'amplada del feix passa de X
    ax.axvline(th, color=estil.TINTA_2, lw=0.8, ls=(0, (1, 2)), zorder=4)
    ax.add_patch(FancyArrowPatch((th, baix[obert[0]]), (th, dalt[obert[0]]),
                                 arrowstyle="<|-|>", mutation_scale=6, color=estil.TINTA,
                                 lw=0.8, shrinkA=0, shrinkB=0, zorder=5))
    ax.text(th - 0.04, (baix[obert[0]] + dalt[obert[0]]) / 2, r"$X$", ha="right",
            va="center", fontsize=8, zorder=6,
            bbox=dict(boxstyle="round,pad=0.1", fc=estil.SUPERFICIE, ec="none"))
    ax.text(th + 0.04, 0.97, "horitzó", transform=ax.get_xaxis_transform(), ha="left",
            va="top", fontsize=7, color=estil.TINTA_2)

    ax.set_xlim(0, FEIX_TEMPS_S)
    ax.set_xticks([0])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel(r"$t$, des de l'amollada", fontsize=8)
    ax.set_ylabel(r"$\theta_1$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", fontsize=6.6)
    estil.desa(fig, "feix", directori=SORTIDA)


# ---------------------------------------------------------------- poincare

POINCARE_REPOS_GRAUS = (40.0, 10.0)  # (theta1, theta2) del repos: una orbita
                                     # REGULAR, perque es vegi que guanya el tall
POINCARE_TEMPS_S = 400.0             # per a la seccio (n'hi ha prou punts)
POINCARE_PROJ_S = 40.0               # per a la projeccio (prou per fer madeixa)


def poincare() -> None:
    """Projeccio de l'espai de fases contra seccio de Poincare, de UNA orbita.

    NOMES SIMULACIO, sense friccio. La projeccio surt de simulador.simula i la
    seccio de caos.seccio_pendol (theta2 = 0 amb omega2 > 0), les funcions del
    treball. L'orbita es regular a posta. Els eixos no porten numeros: es un
    esquema del metode, no la seccio dels resultats (eines/poincare.py).
    """
    from pendol import caos
    from pendol.mesures import dif_angular
    from pendol.simulador import simula

    p = Parametres.des_de_json(ARREL / "parametres.json")
    e0 = np.array([math.radians(POINCARE_REPOS_GRAUS[0]), math.radians(POINCARE_REPOS_GRAUS[1]),
                   0.0, 0.0])
    t, y = simula(p, e0, POINCARE_PROJ_S, fps=200.0)
    s = caos.seccio_pendol(p, e0, POINCARE_TEMPS_S)
    th_s, w_s = s["theta1_plegat"], s["omega1"]
    deriva = float(np.max(np.abs(s["energia_J"] / s["energia_inicial_J"] - 1)))
    print(f"  {len(th_s)} creuaments en {POINCARE_TEMPS_S:g} s; deriva d'energia {deriva:.1e}")
    if len(th_s) < 50 or deriva > 1e-6:
        raise SystemExit("La seccio no te prou punts o l'energia no es conserva.")

    estil.aplica()
    fig, (a1, a2) = plt.subplots(1, 2, sharex=True, sharey=True,
                                 figsize=(MIDA_CM / 2.54, MIDA_CM / 2.54 * 0.55),
                                 gridspec_kw=dict(wspace=0.12))
    a1.plot(np.degrees(dif_angular(y[0], 0.0)), y[2], color=estil.SIMULACIO, lw=0.4)
    a2.plot(np.degrees(th_s), w_s, "o", color=estil.SIMULACIO, markersize=1.6)
    a1.set_title("(a) Projecció", fontsize=7.5, loc="left", fontweight="normal")
    a2.set_title("(b) Secció de Poincaré", fontsize=7.5, loc="left", fontweight="normal")
    for a in (a1, a2):
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        a.set_xlabel(r"$\theta_1$", fontsize=8)
    a1.set_ylabel(r"$\omega_1$", fontsize=8)
    estil.desa(fig, "poincare", directori=SORTIDA)


FIGURES = {"esquema_doble": esquema_doble, "cami_dades": cami_dades,
           "friccio_sec": friccio_sec, "tres_trams": tres_trams,
           "angles_desembolicats": angles_desembolicats,
           "derivada_vs_filtre": derivada_vs_filtre,
           "assajos_friccio": assajos_friccio, "feix": feix, "poincare": poincare}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figura", choices=[*FIGURES, "totes"])
    a = ap.parse_args()
    for nom, f in FIGURES.items():
        if a.figura in (nom, "totes"):
            f()


if __name__ == "__main__":
    main()
