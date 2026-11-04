r"""Figures de la METODOLOGIA del pendol magnetic: esquemes compactes, el just.

Esquemes de metode, no resultats: cap mapa ni cap numero de resultat. No
importa res de pendol/ (vegeu magnetic/CLAUDE.md): nomes magnetic.estil i
magnetic.fisica.

  dalt    vista de dalt: els tres imants numerats, encastats a la placa de
          suport, sobre la circumferencia de radi R, i els eixos x, y amb
          l'origen al punt de repos. Les posicions surten de
          fisica.Parametres, no s'escriuen a ma.
  costat  vista de costat, NOMES LA MEITAT DE L'IMANT 1: el punt de gir, el
          fil L sencer, la massa en repos (discontinua) a d0 de la cara dels
          imants i la massa desviada, que segueix l'arc i PUJA fins a d.
          NO ES A ESCALA (variant D de esborranys/_variants_2b.py, triada per
          en Guillem el 03/10): les mides son del dibuix, pero l'alcada d la
          calcula fisica.alcada() amb aquestes mides i es comprova que el fil
          dibuixat no s'allarga. Abans hi va haver una versio a escala (massa
          alta), una amb el fil tallat (no: si es veu on comenca el fil, no hi
          pot haver un tall pel mig) i una amb el triangle de Delta (no calia).

  aturada el criteri d'aturada, en dos panells l'un sota l'altre (vertical,
          5 cm d'ample), ampliats a prop de l'imant 1:
          una trajectoria que passa pel cercle i en surt (no compta) i una
          que s'hi queda (capturada). El radi i el temps es llegeixen de
          motor.Criteri; la distancia imant -> punt de repos es QUALITATIVA
          (depen de K i n, inventades), i les trajectories, dibuixades a ma.

  box_counting
          el box-counting il.lustrat amb una CORBA DE KOCH (no amb cap mapa:
          seria un resultat): la corba coberta amb capses de tres mides, i
          log N contra log(1/s). Les capses i la recta les fa magnetic.analisi,
          les funcions de debo. Cap valor de D a la figura. Vertical, 6,5 cm.

  conques com es fa el mapa: una quadricula d'11 x 11 que cobreix els imants;
          cada cel.la es un punt de sortida, amb una fletxa cap al seu desti i
          el color d'aquell desti. Zona obvia al voltant de cada imant, el
          centre, i la pols, on el desti NO es l'imant mes proper. Els destins
          surten d'una regla il.lustrativa, no de la fisica. Cap mapa calculat.

LES TRIES (03/10/2026, en Guillem):
  - dues figures separades, compactes, amb el just;
  - imants amb els colors dels mapes (estil.IMANTS) i el numero;
  - d0 des del punt de repos de la massa (el punt que la representa al
    model) fins a la cara de dalt dels imants;
  - cap requadre de mides: els valors van al peu o al text;
  - la vista de costat NO a escala, nomes la meitat de l'imant, 4,5 cm d'ample.

Us, des de l'ARREL del projecte:
    .venv\Scripts\python.exe -m magnetic.figures_metodologia dalt|costat|totes

Surt a magnetic/figures/metodologia/esquema_<nom>.png i .pdf.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                Rectangle)

from magnetic import estil, fisica  # noqa: E402

ARREL = Path(__file__).resolve().parent.parent
SORTIDA = ARREL / "magnetic" / "figures" / "metodologia"

# Mida IMPRESA (cm): es dibuixa ja a la mida final perque la lletra surti a
# 7-9 punts de debo. Si s'encongis un PNG mes gran, no es llegiria.
AMPLADA_CM = 6.5

DISC_IMANT_CM = 1.0   # el disc de 10 x 3 mm, a mida real (com a la pagina)
GRUIX_IMANT_CM = 0.3
RADI_MASSA_CM = 0.45  # simbolic: el model la tracta com un punt

FONS_PLACA = "#efeee9"   # la placa de suport, neutra
FONS_ENCAIX = "#d6d5cd"  # l'encaix de cada imant, una mica mes fosc

# La vista de costat NO ES A ESCALA (en Guillem, 03/10): a escala, amb un fil
# de 29,5 cm i una d0 de 2, la figura surt massa alta o la pujada no es veu.
# Mides PROPIES DEL DIBUIX, en les unitats del dibuix: no son cap mesura i
# no surten a la figura. La geometria si que es la del model (vegeu costat()).
L_DIBUIX = 9.0
D0_DIBUIX = 1.2
ANGLE_DIBUIX_GRAUS = 33.0   # nomes il.lustratiu
X_IMANT_DIBUIX = 3.2        # l'imant 1, prou abans de la massa desviada
                            # perque la Delta no es confongui amb la d
AMPLADA_COSTAT_CM = 4.5


def _figura(xmin, xmax, ymin, ymax, amplada_cm=AMPLADA_CM):
    estil.aplica()
    ample = amplada_cm / 2.54
    fig, ax = plt.subplots(figsize=(ample, ample * (ymax - ymin) / (xmax - xmin)))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    return fig, ax


def _fletxa(ax, a, b, estil_fletxa="-|>", lw=0.8, color=None, z=6, escala=7):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=estil_fletxa, mutation_scale=escala,
                                 color=color or estil.TINTA_2, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def _text(ax, x, y, s, mida=8.5, fons=None, **kw):
    """Text amb un fons llis perque es llegeixi damunt de qualsevol linia.

    `fons` es el color del fons (per defecte, el de la pagina); bbox=None el
    treu del tot.
    """
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("color", estil.TINTA)
    kw.setdefault("bbox", dict(boxstyle="round,pad=0.1", fc=fons or estil.SUPERFICIE,
                               ec="none"))
    ax.text(x, y, s, fontsize=mida, zorder=9, **kw)


# ------------------------------------------------------------------- dalt

def dalt() -> None:
    p = fisica.Parametres.des_de_json()
    R = p.R * 100
    xs, ys = p.im_x * 100, p.im_y * 100

    # Comprovacio: tres imants sobre la circumferencia de radi R i a 120 graus
    # l'un de l'altre, que es el que diu el text. Si algu canvia els angles de
    # parametres.json, la figura no ho ha d'amagar.
    angles = np.degrees(np.arctan2(ys, xs)) % 360
    if not (len(xs) == 3 and np.allclose(np.hypot(xs, ys), R)
            and np.allclose(np.sort(angles), [0, 120, 240], atol=1e-9)):
        raise SystemExit("Els imants de magnetic/parametres.json no son a 0, 120 i 240 "
                         "graus sobre el radi R: l'esquema no els pot dibuixar com a tals.")

    lim = R + 2.6
    fig, ax = _figura(-lim, lim, -lim, lim)

    # La placa de suport on van encastats els imants (referencia visual del
    # muntatge; la mida es simbolica, no es cap mesura)
    costat_placa = 2 * (R + 1.9)
    ax.add_patch(FancyBboxPatch((-costat_placa / 2, -costat_placa / 2), costat_placa,
                                costat_placa, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=FONS_PLACA, edgecolor=estil.EIX, lw=1.0, zorder=0))
    _text(ax, costat_placa / 2 - 0.5, -costat_placa / 2 + 0.55, "suport", mida=6.5,
          ha="right", va="bottom", color=estil.TINTA_SUAU,
          bbox=None)

    # La circumferencia de radi R
    t = np.linspace(0, 2 * np.pi, 300)
    ax.plot(R * np.cos(t), R * np.sin(t), color=estil.TINTA_SUAU, lw=0.8,
            ls=(0, (4, 3)), zorder=2)

    # Eixos x, y amb l'origen al punt de repos
    llarg = R + 1.5  # dins de la placa (que fa R + 1,9 de mig costat)
    _fletxa(ax, (-llarg, 0), (llarg, 0), lw=0.7, color=estil.TINTA_2, z=1)
    _fletxa(ax, (0, -llarg), (0, llarg), lw=0.7, color=estil.TINTA_2, z=1)
    _text(ax, llarg, -0.7, r"$x$", ha="right", fons=FONS_PLACA)
    _text(ax, 0.6, llarg, r"$y$", ha="left", va="top", fons=FONS_PLACA)
    ax.plot(0, 0, marker="o", markersize=3.2, markerfacecolor="white",
            markeredgecolor=estil.TINTA, markeredgewidth=0.9, zorder=8)
    _text(ax, -0.55, -0.65, "O", mida=7.5, ha="right", va="top", color=estil.TINTA_2,
          fons=FONS_PLACA)

    # R, cap a l'imant 2 (cap a l'1 coincidiria amb l'eix x)
    _fletxa(ax, (0, 0), (xs[1] * (1 - 0.55 / R), ys[1] * (1 - 0.55 / R)),
            estil_fletxa="-|>", lw=0.8, color=estil.TINTA, z=5)
    m = np.array([xs[1], ys[1]]) / 2
    nrm = np.array([-ys[1], xs[1]]) / R
    _text(ax, *(m + 0.7 * nrm), r"$R$", fons=FONS_PLACA)

    # Els imants: disc a mida real, color dels mapes, numero a fora
    for k, (x, y) in enumerate(zip(xs, ys)):
        # l'encaix a la placa, perque es vegi que l'imant hi va ficat
        ax.add_patch(Circle((x, y), DISC_IMANT_CM / 2 + 0.22, facecolor=FONS_ENCAIX,
                            edgecolor=estil.EIX, lw=0.7, zorder=6))
        ax.add_patch(Circle((x, y), DISC_IMANT_CM / 2, facecolor=estil.IMANTS[k],
                            edgecolor=estil.TINTA, lw=0.8, zorder=7))
        # el numero, girat 30 graus respecte del radi: aixi el de l'imant 1
        # no cau damunt de l'eix x
        a = math.atan2(y, x) + math.radians(30)
        _text(ax, x + 1.15 * math.cos(a), y + 1.15 * math.sin(a), p.noms[k],
              mida=8.5, fontweight="bold", fons=FONS_PLACA)

    estil.desa(fig, "esquema_dalt", directori=str(SORTIDA))


# ----------------------------------------------------------------- costat

def costat() -> None:
    p = fisica.Parametres.des_de_json()
    if abs(float(p.im_y[0])) > 1e-12:
        raise SystemExit("L'imant 1 no es a y = 0: la vista de costat no el conte.")

    # NO A ESCALA: les mides son les del dibuix (vegeu el capdamunt). Pero la
    # geometria es la del model: l'alcada de la massa desviada la calcula
    # fisica.alcada() amb aquestes mides, i el fil dibuixat fa L_DIBUIX.
    pd = replace(p, L=L_DIBUIX / 100, d0=D0_DIBUIX / 100)
    th = math.radians(ANGLE_DIBUIX_GRAUS)
    xm = L_DIBUIX * math.sin(th)
    d = float(fisica.alcada(xm / 100, 0.0, pd)[1]) * 100
    P = np.array([0.0, D0_DIBUIX + L_DIBUIX])
    repos = np.array([0.0, D0_DIBUIX])
    despl = np.array([xm, d])
    assert abs(np.linalg.norm(despl - P) - L_DIBUIX) < 1e-9  # el fil no s'allarga

    # Nomes la meitat de l'imant 1: la de l'esquerra no es mostra, i aixi no
    # cal decidir on es dibuixen el 2 i el 3 (en Guillem, 03/10)
    xa, xb = -0.9, xm + 2.1
    fig, ax = _figura(xa, xb, -1.7, P[1] + 0.9, amplada_cm=AMPLADA_COSTAT_CM)

    # El pla dels imants (la cara de dalt es z = 0), el suport i l'imant 1
    ax.add_patch(Rectangle((xa, -1.2), xb - xa, 1.2 - GRUIX_IMANT_CM, facecolor="white",
                           edgecolor=estil.EIX, hatch="////", lw=0.0, zorder=1))
    ax.plot([xa, xb], [-GRUIX_IMANT_CM, -GRUIX_IMANT_CM], color=estil.TINTA_SUAU,
            lw=0.8, zorder=2)
    ax.add_patch(Rectangle((X_IMANT_DIBUIX - DISC_IMANT_CM / 2, -GRUIX_IMANT_CM),
                           DISC_IMANT_CM, GRUIX_IMANT_CM, facecolor=estil.IMANTS[0],
                           edgecolor=estil.TINTA, lw=0.8, zorder=3))
    _text(ax, X_IMANT_DIBUIX, -0.85, f"imant {p.noms[0]}", mida=7, color=estil.TINTA_2,
          va="top")

    # El punt de gir
    ax.plot([xa, 1.4], [P[1], P[1]], color=estil.TINTA, lw=1.0, zorder=5)
    ax.add_patch(Rectangle((xa, P[1]), 1.4 - xa, 0.5, facecolor="white",
                           edgecolor=estil.EIX, hatch="////", lw=0.0, zorder=4))

    # El cami de la massa (l'arc), nomes cap a l'imant
    a = np.radians(np.linspace(0, ANGLE_DIBUIX_GRAUS + 8, 100))
    ax.plot(P[0] + L_DIBUIX * np.sin(a), P[1] - L_DIBUIX * np.cos(a),
            color=estil.TINTA_SUAU, lw=0.8, ls=(0, (1, 2)), zorder=2)

    # Els dos fils: en repos (discontinu) i desviat (continu)
    ax.plot(*zip(P, repos), color=estil.TINTA_SUAU, lw=0.9, ls=(0, (3, 2)), zorder=3)
    ax.plot(*zip(P, despl), color=estil.TINTA, lw=0.9, zorder=3)
    ax.add_patch(Circle(repos, RADI_MASSA_CM, facecolor="white", edgecolor=estil.TINTA_2,
                        lw=0.9, ls=(0, (2, 1.5)), zorder=6))
    ax.add_patch(Circle(despl, RADI_MASSA_CM, facecolor=estil.TINTA_2,
                        edgecolor=estil.TINTA, lw=0.9, zorder=6))

    # Les cotes: L al costat del fil en repos, d0 i d
    _text(ax, 0.75, (P[1] + D0_DIBUIX) / 2 + 1.0, r"$L$", mida=9)
    _fletxa(ax, (0.85, D0_DIBUIX), (0.85, 0.0), estil_fletxa="<|-|>", lw=0.7, escala=5)
    _text(ax, 1.25, D0_DIBUIX / 2, r"$d_0$", mida=9, ha="left")
    _fletxa(ax, (xm + 0.9, d), (xm + 0.9, 0.0), estil_fletxa="<|-|>", lw=0.7, escala=5)
    _text(ax, xm + 1.25, d / 2, r"$d$", mida=9, ha="left")

    # Delta, la distancia de la massa a l'imant (en 3D), DISCRETA: una linia
    # fina i clara (en Guillem, 03/10). Va de la cara de l'imant a la massa.
    q = np.array([X_IMANT_DIBUIX, 0.0])
    u = (despl - q) / np.linalg.norm(despl - q)
    fi = despl - RADI_MASSA_CM * u
    ax.plot([q[0], fi[0]], [q[1], fi[1]], color=estil.TINTA_SUAU, lw=0.7, zorder=4)
    m = (q + fi) / 2 + 0.45 * np.array([-u[1], u[0]])
    _text(ax, *m, r"$\Delta$", mida=8, color=estil.TINTA_2)

    estil.desa(fig, "esquema_costat", directori=str(SORTIDA))


# ---------------------------------------------------------------- aturada

DELTA_REPOS_DIBUIX_CM = 0.65  # QUALITATIU: l'imant -> el punt de repos. El de
                              # debo (9,34 mm avui; 5,85 abans del 25/10) depen de K i n, que son
                              # INVENTADES, i per aixo no se'n fa servir el valor


def aturada() -> None:
    """El criteri d'aturada, vist de dalt i ampliat a prop de l'imant 1.

    El radi del cercle i el temps de confinament es LLEGEIXEN de
    motor.Criteri (no s'escriuen a ma). La distancia de l'imant al punt de
    repos es QUALITATIVA (vegeu DELTA_REPOS_DIBUIX_CM). Les dues trajectories
    son dibuixades a ma, il.lustratives: una passa pel cercle i en surt (no
    compta), l'altra s'hi queda (capturada). No s'ha simulat res.
    """
    from magnetic import motor

    cr = motor.Criteri()
    p = fisica.Parametres.des_de_json()
    R = p.R * 100
    rc = cr.r_captura * 100
    imant = np.array([R, 0.0])
    repos = imant - np.array([DELTA_REPOS_DIBUIX_CM, 0.0])
    print(f"  del codi: cercle de {cr.r_captura*100:g} cm, {cr.t_confirma:g} s seguits")

    # Dos panells: la mateixa escena, amb una trajectoria a cada un
    # VERTICAL (en Guillem, 04/10): els dos panells l'un sota l'altre
    x0, x1, y0, y1 = R - 2.5, R + 0.9, -1.55, 1.55
    estil.aplica()
    ample = 5.0 / 2.54
    fig, axs = plt.subplots(2, 1, figsize=(ample, 2 * ample * (y1 - y0) / (x1 - x0) + 0.75),
                            gridspec_kw=dict(hspace=0.32))

    def escena(ax, titol):
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=FONS_PLACA, lw=0,
                               zorder=0))
        ax.set_title(titol, fontsize=7, loc="left", fontweight="normal",
                     color=estil.TINTA, linespacing=1.15)
        # l'imant, encastat com a la 2a
        ax.add_patch(Circle(imant, DISC_IMANT_CM / 2 + 0.11, facecolor=FONS_ENCAIX,
                            edgecolor=estil.EIX, lw=0.7, zorder=3))
        ax.add_patch(Circle(imant, DISC_IMANT_CM / 2, facecolor=estil.IMANTS[0],
                            edgecolor=estil.TINTA, lw=0.8, zorder=4))
        # el punt de repos i el cercle de captura, des d'alla
        ax.plot(*repos, marker="+", markersize=6, mew=1.1, color=estil.TINTA, zorder=8)
        t = np.linspace(0, 2 * np.pi, 300)
        ax.plot(repos[0] + rc * np.cos(t), repos[1] + rc * np.sin(t), color=estil.TINTA,
                lw=0.9, ls=(0, (4, 2)), zorder=5)

    a1, a2 = axs
    # Titols formals (en Guillem, 03/10)
    escena(a1, "(a) Trajectòria que travessa\nla zona de captura")
    escena(a2, f"(b) Trajectòria confinada\ndurant {cr.t_confirma:g} s".replace(".", ","))

    # Les etiquetes, nomes al primer panell
    _text(a1, imant[0], imant[1] + 0.95, f"imant {p.noms[0]}", mida=6.5, fons=FONS_PLACA)
    a1.annotate("punt de repòs", xy=repos, xytext=(x0 + 0.1, y0 + 0.3), fontsize=6.5,
                ha="left", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.1", fc=FONS_PLACA, ec="none"),
                arrowprops=dict(arrowstyle="-", color=estil.TINTA_SUAU, lw=0.6,
                                shrinkA=1, shrinkB=3))
    ang = math.radians(135)
    fi = repos + rc * np.array([math.cos(ang), math.sin(ang)])
    _fletxa(a1, repos, fi, lw=0.7, escala=5, color=estil.TINTA)
    _text(a1, fi[0] - 0.05, fi[1] + 0.22, f"{cr.r_captura*100:g} cm".replace(".", ","),
          mida=6.5, fons=FONS_PLACA, ha="right")
    _fletxa(a1, (x0 + 0.75, y1 - 0.22), (x0 + 0.1, y1 - 0.22), lw=0.7, escala=5)
    _text(a1, x0 + 0.82, y1 - 0.22, "direcció del centre", mida=6, ha="left",
          color=estil.TINTA_2, fons=FONS_PLACA)

    # Les trajectories, DIBUIXADES A MA (il.lustratives)
    # 1. creua el cercle i en surt
    s = np.linspace(0, 1, 200)
    A0, A2 = np.array([x0 + 0.15, -1.0]), np.array([x1 - 0.1, 1.45])
    M = repos + np.array([-0.15, -0.45])
    pas = ((1 - s)**2)[:, None] * A0 + (2 * s * (1 - s))[:, None] * (2 * M - (A0 + A2) / 2) \
        + (s**2)[:, None] * A2
    a1.plot(pas[:, 0], pas[:, 1], color=estil.TINTA, lw=1.0, zorder=6)
    # 2. una espiral que es tanca sobre el punt de repos
    u = np.linspace(0, 1, 600)
    r = 1.25 * np.exp(-3.4 * u)
    an = math.radians(200) + 3 * 2 * np.pi * u
    esp = repos + np.column_stack([r * np.cos(an), 0.85 * r * np.sin(an)])
    a2.plot(esp[:, 0], esp[:, 1], color=estil.TINTA, lw=1.0, zorder=6)

    estil.desa(fig, "esquema_aturada", directori=str(SORTIDA))


# ----------------------------------------------------------- box_counting

KOCH_NIVELL = 6
KOCH_PIXELS = 512
COSTATS_DIBUIX = (64, 32, 16)  # les mides de capsa que es dibuixen (pixels):
                               # minim tres (en Guillem, 03/10)
AMPLADA_BOX_CM = 6.5           # VERTICAL i estreta (en Guillem, 03/10): les
                               # quadricules a dalt i el grafic, petit, a sota


def _koch(nivell: int) -> np.ndarray:
    """Els vertexs de la corba de Koch entre (0, 0) i (1, 0)."""
    p = np.array([[0.0, 0.0], [1.0, 0.0]])
    rot = np.array([[0.5, -math.sqrt(3) / 2], [math.sqrt(3) / 2, 0.5]])
    for _ in range(nivell):
        nous = [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            d = (b - a) / 3
            c = a + d
            nous += [c, c + rot @ d, a + 2 * d, b]
        p = np.array(nous)
    return p


def box_counting() -> None:
    """El box-counting, IL.LUSTRAT AMB UNA CORBA DE KOCH, no amb cap mapa.

    Tres mides de capsa (COSTATS_DIBUIX), en vertical i a 6,5 cm d'ample
    (AMPLADA_BOX_CM): les quadricules a dalt i el grafic log-log a sota.

    Una frontera del mapa de debo seria un resultat; la de Koch es un exemple
    de llibre. Pero el recompte de capses i la recta els fa magnetic.analisi
    (compta_capses i dimensio_per_box_counting), les MATEIXES funcions que
    mesuren la dimensio de les fronteres reals. La figura no en dona cap valor.
    """
    from matplotlib.patches import Rectangle as Rect

    from magnetic import analisi

    # La corba, rasteritzada en una mascara (y cap avall, com una imatge)
    v = _koch(KOCH_NIVELL)
    n = KOCH_PIXELS
    marge = 6
    xy = v * (n - 2 * marge - 1) + np.array([marge, 0.0])
    xy[:, 1] = n - 1 - marge - xy[:, 1]
    mascara = np.zeros((n, n), dtype=bool)
    for a, b in zip(xy[:-1], xy[1:]):
        m = int(np.ceil(np.hypot(*(b - a)) * 2)) + 1
        pts = np.round(a + np.linspace(0, 1, m)[:, None] * (b - a)).astype(int)
        mascara[pts[:, 1], pts[:, 0]] = True
    D, costats, comptes = analisi.dimensio_per_box_counting(mascara, 1, 64)
    print(f"  Koch, nivell {KOCH_NIVELL}: D = {D:.3f} per box-counting "
          f"(la teorica es log 4 / log 3 = {math.log(4) / math.log(3):.3f}); NO va a la figura")

    estil.aplica()
    nf = len(COSTATS_DIBUIX)
    fig = plt.figure(figsize=(AMPLADA_BOX_CM / 2.54, (1.75 * nf + 4.6) / 2.54))
    gs = fig.add_gridspec(nf + 1, 2, height_ratios=[1] * nf + [2.3],
                          width_ratios=[0.12, 1], hspace=0.25, wspace=0.0)
    # Nomes la franja de la mascara on hi ha la corba
    files = np.nonzero(mascara.any(axis=1))[0]
    ya = max(0, (files.min() // COSTATS_DIBUIX[0]) * COSTATS_DIBUIX[0])
    yb = min(n, ((files.max() // COSTATS_DIBUIX[0]) + 1) * COSTATS_DIBUIX[0])
    for k, s in enumerate(COSTATS_DIBUIX):
        ax = fig.add_subplot(gs[k, 1])
        blocs = mascara[: (n // s) * s, : (n // s) * s].reshape(n // s, s, n // s, s).any(axis=(1, 3))
        for i, j in zip(*np.nonzero(blocs)):
            ax.add_patch(Rect((j * s, i * s), s, s, facecolor="#e1e0d9",
                              edgecolor=estil.TINTA_2, lw=0.5, zorder=1))
        ax.plot(xy[:, 0], xy[:, 1], color=estil.TINTA, lw=0.6, zorder=2)
        ax.set_xlim(0, n)
        ax.set_ylim(yb, ya)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(-8, (ya + yb) / 2, f"$s_{k + 1}$", ha="right", va="center", fontsize=7.5)

    # log N contra log(1/s), a sota i mes petit: els punts, la recta, i les
    # mides dibuixades, marcades
    ax = fig.add_subplot(gs[nf, 1])
    x, y = np.log(1.0 / costats), np.log(comptes)
    a, b = np.polyfit(x, y, 1)
    assert abs(a - D) < 1e-12  # la recta es la de dimensio_per_box_counting
    ax.plot(x, a * x + b, color=estil.TINTA_2, lw=1.0, ls=(0, (4, 2)), zorder=2)
    ax.plot(x, y, "o", color=estil.TINTA, markersize=3.2, zorder=3)
    for k, s in enumerate(COSTATS_DIBUIX):
        i = int(np.nonzero(costats == s)[0][0])
        ax.plot(x[i], y[i], "o", markersize=7, markerfacecolor="none",
                markeredgecolor=estil.TINTA, mew=0.9, zorder=4)
        ax.text(x[i] + 0.25, y[i] - 0.35, f"$s_{k + 1}$", fontsize=7.5, ha="left", va="top")
    xm = x.mean()
    ax.text(xm + 0.5, a * xm + b - 0.5, r"pendent $= D$", fontsize=7, ha="left",
            va="top")
    ax.set_xlabel(r"$\log(1/s)$", fontsize=7.5)
    ax.set_ylabel(r"$\log N(s)$", fontsize=7.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    estil.desa(fig, "esquema_box_counting", directori=str(SORTIDA))


# -------------------------------------------------------------- conques

# La quadricula de l'esquema (el mapa de debo en te 600 x 600). Cobreix tambe
# els imants (en Guillem, 04/10): a prop de cada imant, i una mica mes enlla,
# la ZONA OBVIA (tot hi acaba); al mig, el centre; i la resta, la POLS, on el
# desti es barreja i NO es l'imant mes proper. Variant B de
# esborranys/_variants_conques.py.
CONQUES_CEL_LES = 11
CONQUES_MIG_COSTAT = 8.25  # cm: la quadricula va de -8,25 a 8,25 (els imants a R)
CONQUES_FINESTRA = 8.6     # cm: el tros de placa que es veu
CONQUES_R_OBVIA = 2.7      # cm al voltant de cada punt de repos: la zona obvia
CONQUES_R_CENTRE = 1.6     # cm al voltant de l'origen: acaba al centre
CONQUES_LLAVOR = 7         # per a la pols


def _destins_il_lustratius(centres, desti_xy) -> list:
    """Els destins de l'ESQUEMA. NO surten de la fisica: una regla i prou.

    Dins de la zona obvia d'un imant, aquell imant; a prop de l'origen, el
    centre; a la resta (la pols), un imant a l'atzar amb una llavor fixa, de
    manera que sovint NO es el mes proper, que es el que l'esquema vol ensenyar.
    """
    rng = np.random.default_rng(CONQUES_LLAVOR)
    out = []
    for ctr in centres:
        if np.linalg.norm(ctr) < CONQUES_R_CENTRE:
            out.append(0)
            continue
        d_im = {k: np.linalg.norm(ctr - desti_xy[k]) for k in (1, 2, 3)}
        k_prop = min(d_im, key=d_im.get)
        out.append(k_prop if d_im[k_prop] < CONQUES_R_OBVIA else int(rng.integers(1, 4)))
    return out


def conques() -> None:
    """Com es fa el mapa de conques: cada pixel es un punt de sortida, i pren el
    color del desti on acaba la seva trajectoria (la fletxa hi apunta).

    ESQUEMA: no es cap mapa calculat (seria un resultat, i a mes amb K i n
    inventades). Els destins surten d'una regla il.lustrativa
    (_destins_il_lustratius), no de la fisica. Els imants i els colors surten
    de Parametres i d'estil, els mateixos que als mapes.
    """
    p = fisica.Parametres.des_de_json()
    R = p.R * 100
    m = CONQUES_MIG_COSTAT
    nc = CONQUES_CEL_LES
    c = 2 * m / nc
    w = CONQUES_FINESTRA
    imants = np.column_stack([p.im_x, p.im_y]) * 100
    # cap on apunta cada fletxa: el punt de repos de cada desti (QUALITATIU)
    desti_xy = {k + 1: imants[k] * (1 - DELTA_REPOS_DIBUIX_CM / R) for k in range(len(imants))}
    desti_xy[0] = np.zeros(2)

    estil.aplica()
    ample = 9.0 / 2.54
    fig, ax = plt.subplots(figsize=(ample, ample))
    ax.set_xlim(-w, w)
    ax.set_ylim(-w, w)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((-w, -w), 2 * w, 2 * w, facecolor=FONS_PLACA, lw=0, zorder=0))

    centres = [np.array([-m + (col + 0.5) * c, m - (f + 0.5) * c])
               for f in range(nc) for col in range(nc)]
    for ctr, d in zip(centres, _destins_il_lustratius(centres, desti_xy)):
        ax.add_patch(Rectangle(ctr - c / 2, c, c, facecolor=estil.color_del_desti(d),
                               alpha=0.38, lw=0, zorder=2))
        v = desti_xy[d] - ctr
        if np.linalg.norm(v) > 0.2 * c:
            _fletxa(ax, ctr - 0.3 * c * v / np.linalg.norm(v),
                    ctr + 0.33 * c * v / np.linalg.norm(v), lw=0.6, escala=5,
                    color=estil.TINTA, z=6)
        else:
            ax.plot(*ctr, "o", markersize=2.2, color=estil.TINTA, zorder=6)
    for k in range(nc + 1):
        ax.plot([-m + k * c] * 2, [-m, m], color=estil.TINTA_SUAU, lw=0.4, zorder=3)
        ax.plot([-m, m], [-m + k * c] * 2, color=estil.TINTA_SUAU, lw=0.4, zorder=3)

    # Els imants, fora de la quadricula, amb el numero
    for k, (x, y) in enumerate(imants):
        ax.add_patch(Circle((x, y), DISC_IMANT_CM / 2 + 0.12, facecolor=FONS_ENCAIX,
                            edgecolor=estil.EIX, lw=0.7, zorder=4))
        ax.add_patch(Circle((x, y), DISC_IMANT_CM / 2, facecolor=estil.IMANTS[k],
                            edgecolor=estil.TINTA, lw=0.8, zorder=5))
        u = np.array([x, y]) / R
        _text(ax, *(np.array([x, y]) + 1.05 * u), p.noms[k], mida=8, fontweight="bold",
              fons=FONS_PLACA)

    estil.desa(fig, "esquema_conques", directori=str(SORTIDA))


FIGURES = {"dalt": dalt, "costat": costat, "aturada": aturada,
           "box_counting": box_counting, "conques": conques}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("figura", choices=[*FIGURES, "totes"])
    a = ap.parse_args()
    for nom, f in FIGURES.items():
        if a.figura in (nom, "totes"):
            f()


if __name__ == "__main__":
    main()
