"""Estil i colors de les figures del pendol magnetic.

Escrit de zero i NO importat de pendol/estil.py, encara que les tintes neutres
siguin les mateixes. Es una duplicacio deliberada, decidida al CLAUDE.md: si
algun dia es canvia un color, s'ha de canviar als dos llocs a ma. El preu de
tenir dos sistemes que s'entenen per separat.

PER QUE ELS COLORS NO ES PODEN TRIAR "PERQUE FAN BONIC"
-------------------------------------------------------
Al pendol doble la identitat d'una serie mai no depen nomes del color: cada
corba te tambe un trac diferent, i aixi la figura es llegeix igual fotocopiada
en blanc i negre. En un mapa de pixels NO HI HA TRACOS. Una conca es una taca,
i si dues taques tenen el mateix clar-fosc, en blanc i negre el fractal
desapareix.

Per tant la feina l'ha de fer la LLUMINOSITAT. Els cinc destins possibles
s'han repartit al llarg de tota l'escala de grisos, ben separats:

    centre (negre)   gris  11
    imant 1 (blau)   gris  75
    imant 2 (taronja) gris 123
    imant 3 (or)     gris 177
    no convergit     gris 225

Els tons son les dues families del pendol doble -- el blau i el taronja -- mes
un or per al tercer imant. Els blaus i els taronges s'han enfosquit o aclarit
respecte dels del pendol doble NOMES per separar-ne el clar-fosc; la familia
de color es la mateixa i la memoria continua llegint-se com una sola cosa.

De regal, aquesta separacio tambe resol el daltonisme: per a qui no distingeix
el taronja de l'or, els queda la diferencia de clar-fosc, que es de 54 nivells.

LA COMA DECIMAL (02/11/2026)
----------------------------
aplica() posa la coma decimal als numeros dels eixos lineals, com fa
pendol/estil.py des del 29/10 (copiat, no importat). Els textos de dins de les
figures (titols, llegendes) la porten amb coma().

Comprovacio, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.estil

Fa una tira de prova amb els cinc colors DUES vegades, en color i en el gris
que en sortiria en fotocopiar-la, i una franja de ratlles d'un pixel per
comprovar que tambe s'hi distingeixen. Imprimeix-la en blanc i negre: si els
distingeixes, els colors son bons. I comprova que un eix dibuixat despres
d'aplica() surt amb la coma.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

# Superficie i tintes. Els mateixos valors que al pendol doble.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_SUAU = "#898781"
GRAELLA = "#e1e0d9"
EIX = "#c3c2b7"

# Els cinc destins. La clau es el codi que fa servir magnetic.motor.
NO_CONVERGIT = "#e4e3dc"  # gris clar: NO es cap resultat, es un avis
CENTRE = "#0b0b0b"  # negre: si que es un resultat fisic
IMANTS = ("#1c4a8f", "#c85a28", "#d9a941")  # blau fosc, taronja, or

# Ordre dels codis a la paleta: -1, 0, 1, 2, 3
CODIS = (-1, 0, 1, 2, 3)


def color_del_desti(codi: int) -> str:
    if codi == -1:
        return NO_CONVERGIT
    if codi == 0:
        return CENTRE
    if 1 <= codi <= len(IMANTS):
        return IMANTS[codi - 1]
    raise ValueError(
        f"No hi ha color per al desti {codi}. La paleta te {len(IMANTS)} imants; "
        "si n'afegeixes un quart, cal triar-li un color amb un clar-fosc que no "
        "xoqui amb cap dels altres (mira magnetic.estil)."
    )


def mapa_de_colors():
    """(ListedColormap, BoundaryNorm) per pintar la matriu de destins.

    Es fa amb BoundaryNorm i no amb un colormap continu perque els valors son
    CINC ETIQUETES, no una escala: entre l'imant 1 i l'imant 2 no hi ha res.
    Un colormap continu inventaria colors intermedis a les vores i la frontera
    fractal sortiria difuminada, que es precisament el que no ha de passar.
    """
    cmap = ListedColormap([color_del_desti(c) for c in CODIS])
    norm = BoundaryNorm([c - 0.5 for c in CODIS] + [CODIS[-1] + 0.5], cmap.N)
    return cmap, norm


# ---------------------------------------------------------------------------
# Lluminositat
# ---------------------------------------------------------------------------
def lluminancia(color: str) -> float:
    """Lluminancia relativa (0 negre, 1 blanc), formula de la WCAG."""
    canals = []
    for c in to_rgb(color):
        canals.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * canals[0] + 0.7152 * canals[1] + 0.0722 * canals[2]


def gris(color: str) -> int:
    """El nivell de gris (0-255) que sortiria en fotocopiar aquest color."""
    y = lluminancia(color)
    v = y / 12.92 if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055
    return int(round(255 * v))


def _color_gris(color: str) -> str:
    g = gris(color) / 255.0
    return mpl.colors.to_hex((g, g, g))


# ---------------------------------------------------------------------------
# La coma decimal (02/11/2026)
# ---------------------------------------------------------------------------
# La memoria va en catala, i els numeros dels eixos han de dur coma. Es el mateix
# que fa pendol/estil.py des del 29/10, ESCRIT AQUI DE NOU i no importat, per la
# mateixa duplicacio deliberada que els colors. Sense el locale: no depen de la
# configuracio de Windows i no posa separador de milers.
def coma(text: str) -> str:
    """El punt decimal d'un text, canviat per la coma. Dins de text matematic
    ($...$) la coma va entre claus, perque no hi afegeixi espai."""
    return text.replace(".", "{,}" if "$" in text else ",")


class FormatadorComa(mticker.ScalarFormatter):
    """El ScalarFormatter de matplotlib amb la coma decimal."""

    def __call__(self, x, pos=None):
        return coma(super().__call__(x, pos))

    def get_offset(self):
        return coma(super().get_offset())

    def format_data_short(self, value):
        return coma(super().format_data_short(value))


_LINEAL_ORIGINAL = mscale.LinearScale.set_default_locators_and_formatters


def _lineal_amb_coma(self, axis) -> None:
    _LINEAL_ORIGINAL(self, axis)
    axis.set_major_formatter(FormatadorComa())


def aplica() -> None:
    """Aplica l'estil global. Crida-la un cop, abans de dibuixar res. Els eixos
    lineals que es facin despres porten la coma decimal."""
    mscale.LinearScale.set_default_locators_and_formatters = _lineal_amb_coma
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": TINTA,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": TINTA,
            "axes.labelsize": 10,
            "axes.labelcolor": TINTA_2,
            "axes.edgecolor": EIX,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRAELLA,
            "grid.linewidth": 0.7,
            "xtick.color": TINTA_SUAU,
            "ytick.color": TINTA_SUAU,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.2,
            "figure.dpi": 110,
        }
    )


def peu_de_valors_inventats(fig, inventats) -> None:
    """Escriu l'avis a sota de la figura.

    Es el quart nivell de marcatge dels valors inventats. Els altres tres son
    el JSON de parametres, el rertol de la consola i el camp valors_inventats
    de les metadades. Aquest es el que sobreviu quan algu retalla la figura i
    l'enganxa a la memoria sense el text del voltant.
    """
    if not inventats:
        return
    fig.text(
        0.5,
        -0.015,
        "Parametres provisionals inventats (" + ", ".join(inventats) + ") -- no son mesures",
        ha="center",
        va="top",
        fontsize=8,
        color=TINTA_SUAU,
    )


def desa(fig, base: str, directori: str = "magnetic/figures") -> None:
    """Desa en PNG (per mirar) i PDF (vectorial, per imprimir)."""
    d = Path(directori)
    d.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 200), ("pdf", None)):
        fig.savefig(d / f"{base}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"  figura: {d / base}.png  i  .pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comprovacio: python -m magnetic.estil
# ---------------------------------------------------------------------------
SEPARACIO_MINIMA = 40  # nivells de gris entre dos destins qualssevol


def main() -> int:
    noms = ["no convergit", "el centre", "imant 1", "imant 2", "imant 3"]
    colors = [color_del_desti(c) for c in CODIS]

    print("=" * 68)
    print(" magnetic.estil -- els colors dels cinc destins")
    print("=" * 68)
    print(f"\n{'desti':>14} {'codi':>5}  {'color':>9}  {'gris':>5}")
    for nom, codi, col in zip(noms, CODIS, colors):
        print(f"{nom:>14} {codi:>5}  {col:>9}  {gris(col):>5}")

    print("\nSeparacio de clar-fosc entre destins veins (ordenats per gris):")
    ordenats = sorted(zip(colors, noms), key=lambda u: gris(u[0]))
    pitjor = 255
    for (c1, n1), (c2, n2) in zip(ordenats[:-1], ordenats[1:]):
        d = gris(c2) - gris(c1)
        pitjor = min(pitjor, d)
        print(f"   {n1:>14} -> {n2:<14} {d:4d} nivells")
    ok = pitjor >= SEPARACIO_MINIMA
    print(f"\n   la mes justa: {pitjor} nivells (llindar {SEPARACIO_MINIMA})   "
          f"{'OK' if ok else 'MASSA JUSTA'}")

    # La coma decimal: el text que surt d'un eix lineal dibuixat despres
    # d'aplica(). El formatador sol no es pot provar: sense cap eix al darrere,
    # el ScalarFormatter de matplotlib torna un text buit (la primera versio
    # d'aquesta prova ho feia, i va fallar per aixo, el 02/11).
    aplica()
    fig_c, eix_c = plt.subplots()
    eix_c.set_xlim(0.0, 0.375)
    fig_c.canvas.draw()
    textos = [t.get_text() for t in eix_c.get_xticklabels() if t.get_text()]
    plt.close(fig_c)
    be_coma = (coma("1.5 cm") == "1,5 cm" and coma("$10.5$") == "$10{,}5$"
               and all("." not in t for t in textos) and any("," in t for t in textos))
    ok = ok and be_coma
    print(f"   la coma als eixos: {' '.join(textos)}   {'OK' if be_coma else 'MALAMENT'}")

    # --- tira de prova -----------------------------------------------------
    aplica()
    fig, axs = plt.subplots(3, 1, figsize=(9, 5.2), height_ratios=[1, 1, 1.1])

    for eix, tira, titol in (
        (axs[0], colors, "en color"),
        (axs[1], [_color_gris(c) for c in colors], "el mateix, fotocopiat en blanc i negre"),
    ):
        for i, (col, nom) in enumerate(zip(tira, noms)):
            eix.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=col, edgecolor=EIX, lw=0.5))
            eix.text(i + 0.5, -0.22, nom, ha="center", va="top", fontsize=9, color=TINTA_2)
        eix.set_xlim(0, len(tira))
        eix.set_ylim(0, 1)
        eix.set_title(titol, fontsize=10, loc="left")
        eix.axis("off")

    # Ratlles d'un pixel: la frontera fractal acaba tenint aquest gruix.
    cmap, norm = mapa_de_colors()
    rng = np.random.default_rng(0)
    m = rng.choice([1, 2, 3], size=(60, 300))
    m[:, ::7] = 0  # ratlles negres d'un pixel
    axs[2].imshow(m, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
    axs[2].set_title(
        "fronteres d'UN pixel d'ample (les ratlles negres son el centre)", fontsize=10, loc="left"
    )
    axs[2].axis("off")

    fig.suptitle("Pendol magnetic: paleta dels destins", fontsize=11)
    fig.tight_layout()
    desa(fig, "paleta")

    print("\n   Imprimeix figures/paleta.png en BLANC I NEGRE. Si distingeixes")
    print("   els cinc, la paleta es bona; si no, s'han de tornar a separar.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
