"""Estil unic per a totes les figures del treball.

Que totes les grafiques del TdR facin joc no es cosmetica: fa que el lector
aprengui una sola vegada que el TARONJA es el pendol real i el BLAU la
simulacio, i despres ho llegeixi automaticament a totes les figures.

Decisions:
  * Nomes DOS colors de serie. Son la parella 1-2 d'una paleta validada per
    a daltonisme (deuteranopia/protanopia/tritanopia).
  * A mes del color, cada serie te un TRAC diferent (continu / discontinu).
    La identitat mai depen nomes del color: aixi el treball es llegeix igual
    imprès en blanc i negre o fotocopiat.
  * Els textos van en tinta neutra, mai del color de la serie.
  * Graella i eixos discrets: les dades han de dominar la figura.
  * La coma decimal als eixos (29/10): en catala s'escriu 0,5 i no 0.5.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.ticker as mticker

# Superficie i tintes
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_SUAU = "#898781"
GRAELLA = "#e1e0d9"
EIX = "#c3c2b7"

# Series (parella validada; no n'afegeixis una tercera sense revisar-ho)
SIMULACIO = "#2a78d6"  # blau
REAL = "#eb6834"  # taronja
FEIX = "#9ec5f4"  # blau clar: el feix es la MATEIXA entitat que la simulacio

TRAC_SIMULACIO = (0, (5, 2))  # discontinu
TRAC_REAL = "-"  # continu

# Rampa sequencial de les DADES REALS (25/10): una magnitud (l'energia, per exemple) sobre
# punts reals. Un sol to, el de REAL, clar -> fosc, fet en OKLCH (to 40-41 graus). No es
# cap color de serie: el blau ja vol dir simulacio. El pas mes clar fa 2,05:1 sobre la
# SUPERFICIE, i la guia de visualitzacio la valida en mode --ordinal
# (esborranys/_p2_rampa_real.txt; documentacio/pla-analisi-caos.md, «25/10 · El Poincare
# real pintat per energia»).
RAMPA_REAL = ("#fd9974", "#f37d51", "#e36431", "#c75528", "#ac471e", "#913a15", "#782c0c")

# Rampa sequencial de la SIMULACIO (28/10): la germana de RAMPA_REAL, per pintar la mateixa magnitud sobre punts
# simulats (el Poincare amb friccio, al costat del real). Un sol to, el de SIMULACIO (el pas del mig ES SIMULACIO), en
# OKLCH. El pas mes clar fa 2,06:1 sobre la SUPERFICIE, i la guia de visualitzacio la valida en mode --ordinal
# (esborranys/_rampa_simulacio.py i .txt, amb els dos intents que van fallar; documentacio/pla-analisi-caos.md,
# «28/10 · La feina del repartiment»).
RAMPA_SIMULACIO = ("#7db5ff", "#56a0ff", "#3f8cec", "#2a78d6", "#1165c1", "#0053a6", "#004287")


class FormatadorComa(mticker.ScalarFormatter):
    """El ScalarFormatter de matplotlib amb la coma decimal (29/10; documentacio/pla-placa-reinici-comes.md, punt 4). Canvia el punt
    per la coma al text que surt dels metodes publics, sense el `locale`: no depen del sistema i no posa separador de milers. Dins
    de text matematic ($...$) la coma va entre claus, perque no hi afegeixi espai."""

    @staticmethod
    def _amb_coma(s: str) -> str:
        return s.replace(".", "{,}" if "$" in s else ",")

    def __call__(self, x, pos=None):
        return self._amb_coma(super().__call__(x, pos))

    def get_offset(self):
        return self._amb_coma(super().get_offset())

    def format_data_short(self, value):
        return self._amb_coma(super().format_data_short(value))


_LINEAL_ORIGINAL = mscale.LinearScale.set_default_locators_and_formatters


def _lineal_amb_coma(self, axis) -> None:
    _LINEAL_ORIGINAL(self, axis)
    axis.set_major_formatter(FormatadorComa())


def aplica() -> None:
    """Aplica l'estil global. Crida-la un cop, abans de dibuixar res. Els eixos lineals que es facin despres (colorbars incloses)
    porten la coma decimal."""
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
            "lines.linewidth": 1.7,
            "figure.dpi": 110,
        }
    )


def mapa_real():
    """RAMPA_REAL com a mapa de color de matplotlib (interpolat entre els passos)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rampa_real", RAMPA_REAL)


def mapa_simulacio():
    """RAMPA_SIMULACIO com a mapa de color de matplotlib (interpolat entre els passos)."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("rampa_simulacio", RAMPA_SIMULACIO)


def desa(fig, base, directori="figures") -> None:
    """Desa la figura en PNG (per mirar) i PDF (vectorial, per imprimir)."""
    from pathlib import Path

    d = Path(directori)
    d.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 200), ("pdf", None)):
        fig.savefig(d / f"{base}.{ext}", dpi=dpi, bbox_inches="tight")
    print(f"  figura: {d / base}.png  i  .pdf")
    plt.close(fig)


def autoprova() -> int:
    """CO1-CO6 de documentacio/pla-placa-reinici-comes.md, punt 4."""
    mpl.use("Agg")
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    def textos(ax, eix="x"):
        ax.figure.canvas.draw()
        return [t.get_text() for t in (ax.get_xticklabels() if eix == "x" else ax.get_yticklabels()) if t.get_text()]

    print("Autoprova de pendol/estil.py (les comes; criteris de documentacio/pla-placa-reinici-comes.md, punt 4)")
    aplica()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [-1.0, 0.0])
    x, y = textos(ax), textos(ax, "y")
    mira("0,2" in x and not any("." in s for s in x), f"CO1 un eix lineal de 0 a 1: {x}")
    # A la primera versio s'esperava «−0,5», que no es cap marca d'aquest eix (van de 0,2 en 0,2), i la CO4 feia servir 1e-4, que
    # encara no passa a notacio cientifica (ho fa per sota d'1e-5). Errors de Claude en escriure la prova, corregits el 29/10 i
    # declarats a pla-placa-reinici-comes.md.
    mira("\N{MINUS SIGN}0,2" in y and not any("." in s for s in y), f"CO2 els negatius, amb el signe menys: {y}")
    sm = ax.scatter([0, 1], [0, 1], c=[0.0, 1.0])
    cb = fig.colorbar(sm)
    c = textos(cb.ax, "y")
    mira("0,2" in c and not any("." in s for s in c), f"CO3 la colorbar: {c}")
    plt.close(fig)
    fig, (a1, a2) = plt.subplots(1, 2)
    a1.plot([0.0, 1.0], [0.0, 1.5e-7])
    a2.plot([1000.1, 1000.5], [0.0, 1.0])
    y1, x2 = textos(a1, "y"), textos(a2)
    o1, o2 = a1.yaxis.get_offset_text().get_text(), a2.xaxis.get_offset_text().get_text()
    mira(o1 != "" and o2 != "" and "." not in o1 + o2 and not any("." in s for s in y1 + x2),
         f"CO4 la notacio cientifica ({o1!r}, {y1}) i el desplacament ({o2!r}, {x2})")
    plt.close(fig)
    aplica()
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1000.0])
    x, y = textos(ax), textos(ax, "y")
    mira(isinstance(ax.xaxis.get_major_formatter(), FormatadorComa) and "0,2" in x and "." not in "".join(x),
         f"CO5 aplica() dues vegades: {x}")
    mira("1000" in y and not any("," in s or "." in s for s in y), f"CO6 els enters no canvien: {y}")
    plt.close(fig)
    print("\nTOT PASSA" if not problemes else f"\nFALLA: {problemes}")
    return 0 if not problemes else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(autoprova() if "--autoprova" in sys.argv[1:] else 0)
