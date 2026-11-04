"""Mesura la n i la K dels imants fent servir el pendol com a dinamometre.

Son les dues ultimes entrades INVENTADES de magnetic/parametres.json. Amb un
sol imant posat a una distancia coneguda, la massa s'atura desviada cap a ell;
alla l'estirada horitzontal de l'imant es exactament la que el fil fa per
tornar-la al centre (amb la gravetat i amb l'estirada vertical de l'imant
mateix: vegeu mes avall), i mesurant el desviament a diverses distancies en
surt la llei de potencia sencera:

    ln(F/m) = ln K - n * ln(delta)

o sigui una RECTA en log-log, de pendent -n i ordenada ln K. Les dues constants
del mateix experiment i del mateix grafic.

PER QUE AIXO NO ES POT FER A MA
-------------------------------
Es va fer a ma, i el 7 de setembre del 2026 van circular dues taules de sis
valors del mateix document, semblants i diferents: amb una el pendent surt
-2,85 i amb l'altra -1,96. Cap de les dues no venia amb cap ordre que la
tornes a treure. Aquest programa existeix perque el numero que anira a la
memoria tingui un cami que es pugui repetir.

LA K ES PER UNITAT DE MASSA, I AIXO NO ES UN DETALL
---------------------------------------------------
A magnetic/fisica.py la magnitud de l'acceleracio que fa un imant es K/delta^n,
no una forca: la m no surt a les equacions del mapa. Per tant l'ajust va sobre
ln(F/m) i no sobre ln(F). Ajustar la forca en newtons donaria la mateixa n i
una K equivocada per un factor 500, i el mapa no en diria res.

L'ALCADA VARIABLE NO ES CAP CORRECCIO AFEGIDA
---------------------------------------------
La massa penja d'un fil de llargada fixa, o sigui que en desviar-se PUJA:

    h(x) = h0 + L - sqrt(L^2 - x^2)

Aquest h es EXACTAMENT el d = d0 + L - s de fisica.acceleracio_nucli. Ignorar-lo
no es aproximar, es calibrar un model diferent del que despres s'integra --i
llavors la n que en surt no es la n del simulador--. Quant se n'aparta ho mesura
l'autoprova cada vegada, i --alcada-fixa ho ensenya sobre les dades de debo.

L'ESTIRADA VERTICAL DE L'IMANT TAMBE (25/10/2026)
-------------------------------------------------
La massa va obligada per l'esfera del fil, i la tensio nomes s'empassa la
component RADIAL de les forces. L'imant l'estira cap avall, el fil s'ha de
tibar mes, i la seva component horitzontal creix: l'imant fa de gravetat de
mes. L'equilibri es

    g*x/s + f*(h/delta)*(x/s) = f*a/delta    ->    f = (g*x/s)*delta/(a - h*x/s)

i no f = (g*x/s)*delta/a, que es el que hi havia fins al 25/10. Es el mateix
terme que es va afegir a fisica.acceleracio_nucli el mateix dia. Pesa sobretot a
la tanda 1, on la massa queda gairebe sota l'imant i l'estirada vertical es
molt mes gran que l'horitzontal: sense aquest terme, les dues tandes no
quadraven amb cap llei. Ho va fer veure magnetic/imant_rigid.py. --alcada-fixa
continua sense el terme: sense pujada no hi ha pendent, i no n'hi pot haver.

LES DUES BARRES D'ERROR
-----------------------
Es calculen les dues i s'imprimeixen una al costat de l'altra:

  minims quadrats  la classica, amb la sigma de cada ln F propagada des de la
                   sigma del desviament. Es de primer ordre.
  Monte Carlo      es remostreja cada lectura dins de la seva sigma i es torna
                   a ajustar. Les sistematiques (L, h0, sep) es sortegen UN SOL
                   COP PER TANDA, perque afecten tots els seus punts alhora.

ES REPORTA LA DE MONTE CARLO, i el motiu es que la classica no pot valer aqui:
de la mesura nomes se'n veu la component horitzontal de la forca, i recuperar-ne
la total vol dir multiplicar per delta/(a - h*x/s), que a la tanda 1 es fa molt
gran (el programa l'imprimeix). Amb una amplificacio aixi la relacio ja no es
lineal, i una propagacio de primer ordre no descriu res.

COMPTE AMB EL QUE AIXO NO VOL DIR. Es facil suposar que la classica sortira
mes ESTRETA del que toca; sobre la tanda 1, amb el balanc vell, va sortir al
reves (la de Monte Carlo era x0,87 la classica). El programa no ho dona per
suposat: imprimeix la rao entre les dues i diu el que hi troba, sigui quin
sigui el sentit.

ELS PUNTS QUE NO ES PODEN CONVERTIR (25/10/2026)
-----------------------------------------------
Si l'imant captura la massa, la deixa gairebe a sobre seu, alla on el fil
apunta a l'imant. Alla una forca central queda gairebe al llarg del fil i la
tensio se l'empassa: on es para la massa ho decideix el fil, no la llei, i
l'amplificacio de dalt es dispara. Si la lectura cau a l'altra banda d'aquell
punt (a <= h*x/s), la conversio a forca NO EXISTEIX. Aquests punts queden fora
de la recta i el programa ho diu en gros; no es tria cap llindar. Els que hi
queden a tocar entren amb el pes que els toca, que es gairebe zero, i per aixo
la sigma de lectura que falta es una sigma EN MIL.LIMETRES comuna a la tanda,
no una de ln F (vegeu analitza). En mil.limetres de desviament, els de fora es
continuen comparant amb cada llei.

I quan un punt te repeticions pero poques (menys de MIN_LECTURES_PROPIES), la
seva sigma tampoc no es la seva: es la de lectura comuna de la tanda, amb totes
les repeticions agrupades (vegeu punts_de_la_taula). Les tandes que el JSON
marca a tandes_fora_de_l_ajust no entren a l'ajust, i es diu.

Us, des de l'ARREL del projecte:

    .venv\\Scripts\\python.exe -m magnetic.llei_de_forca magnetic/dades/llei_de_forca.csv

I sense cap dada, per comprovar que l'eina mateixa no menteix:

    .venv\\Scripts\\python.exe -m magnetic.llei_de_forca --autoprova

AQUEST PROGRAMA NO TOCA parametres.json, igual que magnetic.decrement. Canviar
la n i la K obliga a refer tots els mapes de la memoria, i aixo es fa un cop, a
ma i sabent que es fa.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, fsolve

from magnetic import dades, estil, fisica
from magnetic.fisica import Parametres

SORTIDES_PER_DEFECTE = 10000
LLAVOR_PER_DEFECTE = 20260907
# Per sota d'aquestes lectures, la sigma d'un punt es la de lectura comuna de la
# seva tanda (vegeu punts_de_la_taula).
MIN_LECTURES_PROPIES = 3
# La distancia critica es resol amb un fsolve per sorteig, que costa. Amb 2000
# n'hi ha de sobres per a un percentil 16-84: la seva incertesa d'ostreig es del
# 2% de l'amplada de la forquilla.
SORTIDES_CRITICA = 2000


# ---------------------------------------------------------------------------
# De (sep, x) als numeros derivats
# ---------------------------------------------------------------------------
def magnituds(sep, x, h0, L, g, alcada_fixa: bool = False, estricte: bool = True):
    """Les columnes del full del banc, calculades.

    sep  distancia horitzontal a la qual s'ha posat l'imant
    x    desviament de la massa cap a ell
    h0   alcada del centre de la massa sobre el de l'imant, en repos

    Tot en SI. Retorna un diccionari amb a, s_fil, h, delta, f_hor, f,
    denominador i amplificacio, on les dues forces son PER UNITAT DE MASSA
    (m/s^2). L'amplificacio es delta/denominador: per quant es multiplica la
    forca horitzontal per tenir la total.

    alcada_fixa=True es l'ajust DOLENT, el que ignora que la massa puja. Hi es
    per poder ensenyar quant se n'aparta, no per fer-lo servir.

    estricte=False no s'atura on la conversio no existeix (denominador <= 0):
    hi posa f = nan. Ho fan servir separa_convertibles() i el Monte Carlo.
    """
    sep = np.asarray(sep, dtype=float)
    x = np.asarray(x, dtype=float)
    h0 = np.asarray(h0, dtype=float)
    L = np.asarray(L, dtype=float)

    if np.any(x >= sep):
        raise ValueError(
            "Hi ha un desviament mes gran que la distancia de l'imant (x >= sep): "
            "la massa hauria passat de llarg. Mira les unitats de la taula."
        )
    if np.any(x >= L):
        raise ValueError("Hi ha un desviament mes gran que el fil (x >= L).")

    s_fil = np.sqrt(L**2 - x**2)
    h = h0 if alcada_fixa else h0 + L - s_fil
    a = sep - x
    delta = np.sqrt(a * a + h * h)
    f_hor = g * x / s_fil
    # Amb l'estirada vertical de l'imant (vegeu el docstring del modul). Sense
    # pujada, l'ajust dolent tampoc no la porta.
    denominador = a if alcada_fixa else a - h * x / s_fil
    if estricte and np.any(denominador <= 0):
        raise ValueError(
            "Hi ha un desviament que cap forca central no pot sostenir "
            "(a <= h*x/s): l'estirada vertical de l'imant ja tornaria la "
            "massa al centre mes del que l'horitzontal l'acosta."
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        amplificacio = np.where(denominador > 0, delta / denominador, np.nan)
    f = f_hor * amplificacio
    return {"a": a, "s_fil": s_fil, "h": h, "delta": delta, "f_hor": f_hor, "f": f,
            "denominador": denominador, "amplificacio": amplificacio}


# ---------------------------------------------------------------------------
# Els punts de mesura
# ---------------------------------------------------------------------------
@dataclass
class Punt:
    """Un punt de mesura: una distancia de l'imant, amb totes les seves lectures."""

    tanda: int
    sep: float
    h0: float
    L: float
    x: float
    lectures: np.ndarray
    sigma_x: float | None
    sigma_de: str
    costats: tuple = field(default_factory=tuple)
    # Els dos costats per separat, quan n'hi ha dos. Vegeu desnivell_dels_costats().
    x_per_costat: dict = field(default_factory=dict)
    sigma_repeticions: float | None = None
    desnivell: float | None = None


def punts_de_la_taula(taula, L_per_defecte: float, sigma_lectura: float | None):
    """Agrupa les lectures per (tanda, sep) i en treu el desviament i la seva sigma.

    LA MITJANA ES FA EN x, NO EN LA FORCA. La cancel.lacio de l'error de zero
    es una propietat de x: mesurant pels dos costats les lectures son x+e i
    x-e, i la seva mitjana es x. Fent la mitjana de les forces, que hi depenen
    de manera no lineal, l'error de zero NO se n'aniria.

    LA SIGMA D'UN PUNT AMB POQUES LECTURES ES LA DE LA SEVA TANDA (25/10). Amb
    dues lectures, la dispersio d'un punt te un sol grau de llibertat i no es
    fiable: dues lectures que coincideixen per casualitat donarien una sigma
    gairebe zero, i aquell punt s'enduria la recta. Com que dins d'una tanda
    es llegeix igual a tot arreu, als punts amb menys de MIN_LECTURES_PROPIES
    lectures se'ls posa la sigma de lectura COMUNA de la tanda, la variancia
    agrupada de tots els seus punts amb repeticions:

        sigma^2 = suma((n_i - 1) * s_i^2) / suma(n_i - 1)

    dividida per sqrt(n) del punt. Els punts amb prou lectures es queden amb la
    seva: a la tanda 2, la dispersio creix amb x per la diferencia entre els
    dos costats, i agrupar-la l'amagaria.
    """
    punts = []
    for (tanda, sep_cm), grup in taula.groupby(["tanda", "sep_cm"], sort=True):
        h0s = grup["h0_cm"].unique()
        if len(h0s) != 1:
            raise ValueError(
                f"La tanda {tanda} a sep = {sep_cm} cm te dues alcades diferents "
                f"({h0s}). L'alcada es del muntatge, no de la lectura."
            )
        Ls = grup["L_cm"].dropna().unique()
        if len(Ls) > 1:
            raise ValueError(f"La tanda {tanda} a sep = {sep_cm} cm te dues L diferents ({Ls}).")
        L = float(Ls[0]) / 100.0 if len(Ls) == 1 else L_per_defecte

        sep = float(sep_cm) / 100.0
        valor = grup["valor_mm"].to_numpy(dtype=float) / 1000.0
        es_a = (grup["mesura"] == "a").to_numpy()
        lectures = np.where(es_a, sep - valor, valor)
        costat = grup["costat"].astype(str).to_numpy()

        # Els dos costats per separat. La mitjana de tot continua sent la x que
        # es fa servir -- es el que cancel.la l'error de zero -- pero saber que
        # dona cada costat permet distingir DUES coses que la dispersio de les
        # sis lectures barreja: el pols de la lectura i el fet que l'imant no
        # sigui a la mateixa distancia pels dos costats.
        per_costat = {c: lectures[costat == c] for c in sorted(set(costat))}
        x = float(np.mean(lectures))

        sigma_rep, desnivell = None, None
        if all(len(v) >= 2 for v in per_costat.values()):
            # Variancia comuna DINS de cada costat: nomes la repetibilitat.
            gl = sum(len(v) - 1 for v in per_costat.values())
            var = sum(np.var(v, ddof=1) * (len(v) - 1) for v in per_costat.values()) / gl
            sigma_rep = float(math.sqrt(var / len(lectures)))
        if len(per_costat) == 2:
            (a_, b_) = (float(np.mean(v)) for v in per_costat.values())
            desnivell = abs(b_ - a_) / 2.0

        if len(lectures) >= 2:
            sigma = float(np.std(lectures, ddof=1) / math.sqrt(len(lectures)))
            origen = f"dispersio de {len(lectures)} lectures"
        elif sigma_lectura is not None:
            sigma = float(sigma_lectura)
            origen = "sigma_lectura_mm del JSON"
        else:
            sigma = None
            origen = "PENDENT -- dels residus de l'ajust"

        punts.append(
            Punt(
                tanda=int(tanda),
                sep=sep,
                h0=float(h0s[0]) / 100.0,
                L=L,
                x=x,
                lectures=lectures,
                sigma_x=sigma,
                sigma_de=origen,
                costats=tuple(sorted(set(costat))),
                x_per_costat={c: float(np.mean(v)) for c, v in per_costat.items()},
                sigma_repeticions=sigma_rep,
                desnivell=desnivell,
            )
        )

    for t in sorted({q.tanda for q in punts}):
        seus = [q for q in punts if q.tanda == t]
        amb_rep = [q for q in seus if len(q.lectures) >= 2]
        graus = sum(len(q.lectures) - 1 for q in amb_rep)
        if graus == 0 or all(len(q.lectures) >= MIN_LECTURES_PROPIES for q in seus):
            continue
        var = sum(np.var(q.lectures, ddof=1) * (len(q.lectures) - 1) for q in amb_rep) / graus
        comuna = float(math.sqrt(var))
        for q in seus:
            if len(q.lectures) < MIN_LECTURES_PROPIES:
                q.sigma_x = comuna / math.sqrt(len(q.lectures))
                q.sigma_de = (f"lectura comuna de la tanda {t}: {comuna*1000:.3f} mm "
                              f"(variancia agrupada, {graus} graus de llibertat)")

    punts.sort(key=lambda p: (p.tanda, -p.sep))
    return punts


def vectors(punts):
    """Els quatre vectors que fan falta a tot arreu: sep, x, h0, L."""
    return (
        np.array([p.sep for p in punts]),
        np.array([p.x for p in punts]),
        np.array([p.h0 for p in punts]),
        np.array([p.L for p in punts]),
    )


def separa_convertibles(punts, g, alcada_fixa=False):
    """Els punts que es poden passar a forca, i els que no. Retorna (dins, fora).

    NO ES TRIA CAP LLINDAR. Un punt queda fora nomes si la conversio no existeix:
    la massa es llegeix mes enlla d'on el fil apunta a l'imant (a <= h*x/s), i
    alla cap forca central no la pot sostenir. Passa quan l'imant l'ha capturada
    i ha quedat gairebe a sobre: on es para ho decideix el fil, no la llei, i un
    error de lectura de dues decimes de mil.limetre ja la fa passar a l'altra
    banda. Els que hi queden a tocar entren a la recta amb el pes que els toca,
    que es gairebe zero perque l'amplificacio s'hi dispara.

    Els de fora continuen sent mesures bones de desviament: afegeix_comprovacions
    els compara en mil.limetres com els altres.
    """
    sep, x, h0, L = vectors(punts)
    m = magnituds(sep, x, h0, L, g, alcada_fixa, estricte=False)
    dins = [q for q, d in zip(punts, m["denominador"]) if d > 0]
    fora = [{"punt": q, "denominador_m": float(d)}
            for q, d in zip(punts, m["denominador"]) if d <= 0]
    return dins, fora


def sensibilitat_lnf(punts, g, alcada_fixa=False):
    """d(ln F)/dx a cada punt, per diferencies centrades.

    Es el factor que converteix una sigma de desviament en una sigma de log de
    forca. NO es analitic a posta: aixi no pot quedar desaparellat de
    magnituds() el dia que algu hi toqui una linia.
    """
    sep, x, h0, L = vectors(punts)
    e = 1e-8
    mes = np.log(magnituds(sep, x + e, h0, L, g, alcada_fixa)["f"])
    menys = np.log(magnituds(sep, x - e, h0, L, g, alcada_fixa)["f"])
    return (mes - menys) / (2 * e)


# ---------------------------------------------------------------------------
# La recta
# ---------------------------------------------------------------------------
def recta(lx, ly, sigma_y=None):
    """Ajusta ly = ordenada + pendent*lx. Retorna n = -pendent, lnK i la covariancia."""
    lx = np.asarray(lx, dtype=float)
    ly = np.asarray(ly, dtype=float)
    w = np.ones_like(lx) if sigma_y is None else 1.0 / np.asarray(sigma_y, dtype=float) ** 2

    S = w.sum()
    Sx = (w * lx).sum()
    Sy = (w * ly).sum()
    Sxx = (w * lx * lx).sum()
    Sxy = (w * lx * ly).sum()
    den = S * Sxx - Sx * Sx
    if den <= 0:
        raise ValueError("Tots els punts son a la mateixa distancia: no hi ha cap recta.")

    pendent = (S * Sxy - Sx * Sy) / den
    ordenada = (Sxx * Sy - Sx * Sxy) / den
    var_p = S / den
    var_o = Sxx / den
    cov_po = -Sx / den

    residus = ly - (ordenada + pendent * lx)
    if sigma_y is None and len(lx) > 2:
        # Sense sigmes declarades, l'escala de l'error surt dels residus.
        s2 = (residus**2).sum() / (len(lx) - 2)
        var_p, var_o, cov_po = var_p * s2, var_o * s2, cov_po * s2

    mitjana = (w * ly).sum() / S
    ss_tot = (w * (ly - mitjana) ** 2).sum()
    ss_res = (w * residus**2).sum()

    return {
        "n": -pendent,
        "lnK": ordenada,
        "K": math.exp(ordenada),
        "sigma_n": math.sqrt(var_p),
        "sigma_lnK": math.sqrt(var_o),
        "correlacio": -cov_po / math.sqrt(var_p * var_o),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "residus": residus,
        "n_punts": len(lx),
    }


def _recta_vectoritzada(lx, ly, w):
    """La mateixa recta, per a moltes files alhora. lx, ly: (M, N); w: (N,) o (M, N)."""
    w = np.broadcast_to(w, lx.shape)
    S = w.sum(axis=1)
    Sx = (w * lx).sum(axis=1)
    Sy = (w * ly).sum(axis=1)
    Sxx = (w * lx * lx).sum(axis=1)
    Sxy = (w * lx * ly).sum(axis=1)
    den = S * Sxx - Sx * Sx
    pendent = (S * Sxy - Sx * Sy) / den
    ordenada = (Sxx * Sy - Sx * Sxy) / den
    return -pendent, ordenada


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo(punts, g, sistematiques, sortides, llavor, alcada_fixa=False):
    """Remostreja les mesures i torna a ajustar. Retorna el nuvol de (n, lnK).

    LES SISTEMATIQUES ES SORTEGEN UN COP PER TANDA. Si el regle de posar
    l'imant esta desplacat, ho esta per a tots els punts d'aquella tanda alhora:
    sortejar-lo punt a punt el convertiria en soroll aleatori, que es cancel.la
    fent mitjanes, i la barra d'error sortiria mes estreta del que toca.
    """
    rng = np.random.default_rng(llavor)
    sep0, x0, h00, L0 = vectors(punts)
    sigma_x = np.array([p.sigma_x for p in punts], dtype=float)

    tandes = sorted({p.tanda for p in punts})
    quina = np.array([tandes.index(p.tanda) for p in punts])

    s_sep = sistematiques.get("sep") or 0.0
    s_h0 = sistematiques.get("h0") or 0.0
    s_L = sistematiques.get("L") or 0.0

    x = x0 + rng.normal(0.0, 1.0, size=(sortides, len(punts))) * sigma_x
    sep = sep0 + rng.normal(0.0, s_sep, size=(sortides, len(tandes)))[:, quina]
    h0 = h00 + rng.normal(0.0, s_h0, size=(sortides, len(tandes)))[:, quina]
    L = L0 + rng.normal(0.0, s_L, size=(sortides, 1))

    bo = np.all((x > 0) & (x < sep) & (x < L), axis=1)
    if not bo.any():
        raise ValueError("Cap sorteig no dona una geometria possible. Mira les sigmes.")

    m = magnituds(sep[bo], x[bo], h0[bo], L[bo], g, alcada_fixa, estricte=False)
    pes = 1.0 / sigma_de_lnf(punts, g, sigma_x, alcada_fixa) ** 2
    # LA MATEIXA REGLA QUE A L'AJUST CENTRAL, SORTEIG A SORTEIG: un punt que en
    # un sorteig cau on la conversio no existeix queda fora de la recta d'aquell
    # sorteig (pes zero). Nomes els passa als que ja eren a tocar, que pesen poc.
    convertible = np.isfinite(m["f"])
    lf = np.log(np.where(convertible, m["f"], 1.0))
    n, lnK = _recta_vectoritzada(np.log(m["delta"]), lf, np.where(convertible, pes, 0.0))
    amb_recta = convertible.sum(axis=1) >= 2
    return {
        "n": n[amb_recta],
        "lnK": lnK[amb_recta],
        "descartats": int((~bo).sum() + (~amb_recta).sum()),
        "sortides": int(amb_recta.sum()),
        "punts_sense_conversio": int((~convertible[amb_recta]).sum()),
        "llavor": int(llavor),
    }


def sigma_de_lnf(punts, g, sigma_x, alcada_fixa=False):
    """La sigma de ln F de cada punt, propagada des de la del desviament."""
    return np.abs(sensibilitat_lnf(punts, g, alcada_fixa)) * np.asarray(sigma_x, dtype=float)


def percentils(mostra):
    """Mediana i les dues cues del 68%, que es el que vol dir '+-1 sigma'."""
    p16, p50, p84 = np.percentile(mostra, [15.865, 50.0, 84.135])
    return {"p16": float(p16), "p50": float(p50), "p84": float(p84),
            "sigma": float((p84 - p16) / 2.0)}


# ---------------------------------------------------------------------------
# La distancia critica: la comprovacio que l'ajust no pot fer trampa
# ---------------------------------------------------------------------------
def _ax(x, sep, n, K, h0, L, g):
    """L'acceleracio horitzontal amb UN sol imant, vectoritzada.

    Es la mateixa expressio que fisica.acceleracio_nucli amb y = 0, v = 0 i un
    imant a (sep, 0). Aqui esta escrita a part perque cal avaluar-la milers de
    vegades sobre vectors i la de fisica es escalar (va compilada amb Numba).
    QUE DIUEN EL MATEIX es comprova de debo, no es una promesa: ho fa
    comprova_contra_fisica(), que l'autoprova crida cada vegada.
    """
    s_fil = np.sqrt(L * L - x * x)
    d = h0 + L - s_fil
    a = sep - x
    delta = np.sqrt(a * a + d * d)
    forca = K / delta ** (n + 1.0)
    # L'estirada vertical de l'imant fa de gravetat de mes (25/10/2026), en el
    # mateix ordre d'operacions que fisica.acceleracio_nucli.
    return -(g + forca * d) * x / s_fil + forca * a


def comprova_contra_fisica(n, K, h0, L, g, sep=0.05):
    """La maxima diferencia entre _ax i fisica.acceleracio_nucli. Ha de ser ~0."""
    im_x = np.array([sep])
    im_y = np.array([0.0])
    im_K = np.array([K])
    pitjor = 0.0
    for x in np.linspace(0.0, sep * 0.98, 25):
        meu = float(_ax(np.array([x]), sep, n, K, h0, L, g)[0])
        seu = fisica.acceleracio_nucli(float(x), 0.0, 0.0, 0.0, L, g, 0.0, n, h0,
                                       im_x, im_y, im_K)[0]
        pitjor = max(pitjor, abs(meu - seu))
    return pitjor


def equilibris(sep, n, K, h0, L, g, punts_graella=4000):
    """Els x on l'acceleracio horitzontal s'anul.la, de fora cap a dins.

    COM VA DE DEBO, que no es el que sembla. Movent l'imant de lluny cap a prop
    el nombre d'equilibris fa 1 -> 3 -> 1:

      imant molt lluny   NOMES el de prop del centre. Sota l'imant no n'hi ha
                         cap, perque per anar-hi la massa hauria de pujar tant
                         que l'imant ja no l'arriba a retenir.
      banda del mig      TRES: el de prop del centre (estable), un d'inestable,
                         i el de sota l'imant (estable). Son els dos que la
                         tanda 1 va veure a sep = 7 cm.
      imant a prop       nomes el de sota l'imant: el de prop del centre s'ha
                         fos amb l'inestable i ha desaparegut. AIXO ES EL SALT.

    O sigui que hi ha DOS plecs, no un, i el que es veu al banc es el de sota.
    """
    sostre = min(sep, L) * 0.999999
    x = np.linspace(0.0, sostre, punts_graella)
    f = _ax(x, sep, n, K, h0, L, g)
    canvis = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    arrels = []
    for i in canvis:
        x0, x1 = x[i], x[i + 1]
        f0, f1 = f[i], f[i + 1]
        arrels.append(float(x0 - f0 * (x1 - x0) / (f1 - f0)))
    return arrels


def plecs(n, K, h0, L, g, punts_graella=8000):
    """ELS DOS plecs, en metres. Retorna {'salt': (sep, x), 'desenganxada': (sep, x)}.

    Hi ha una banda de distancies amb TRES equilibris, i te dues vores. Les dues
    es veuen al banc i no son la mateixa cosa:

      salt          la vora de BAIX. Acostant l'imant, l'equilibri de prop del
                    centre desapareix i la massa se'n va de cop cap a l'imant.
      desenganxada  la vora de DALT. Amb la massa ja enganxada sota l'imant,
                    allunyant-lo arriba un punt que aquell equilibri desapareix
                    i la massa torna al centre de cop.

    ENTRE LES DUES VORES EL SISTEMA ES BIESTABLE, i on es posi la massa depen
    de com hi hagi arribat. Aixo no es un detall: una mesura feta en aquella
    banda NO diu on es cap dels dos plecs, nomes diu on era la massa. Confondre
    «alla la massa estava enganxada» amb «alla es on s'enganxa» fa sortir una
    distancia critica que no ho es.
    """
    escombrat = np.linspace(2e-3, 0.9 * L, 600)
    quants = [len(equilibris(s, n, K, h0, L, g, punts_graella)) for s in escombrat]
    dins = [i for i, q in enumerate(quants) if q >= 3]
    if not dins or dins[0] == 0 or dins[-1] == len(escombrat) - 1:
        return {"salt": (math.nan, math.nan), "desenganxada": (math.nan, math.nan)}

    def afina(i_fora, i_dins, quines):
        baix, alt = escombrat[i_fora], escombrat[i_dins]
        for _ in range(50):
            mig = 0.5 * (baix + alt)
            if len(equilibris(mig, n, K, h0, L, g, punts_graella)) >= 3:
                alt = mig
            else:
                baix = mig
        arrels = equilibris(alt, n, K, h0, L, g, punts_graella)
        x0 = 0.5 * (arrels[quines[0]] + arrels[quines[1]])
        sep_c, x_c = _afina_plec(x0, alt, n, K, h0, L, g)
        return (sep_c, x_c)

    return {
        # A baix es fonen l'equilibri de prop del centre (0) i l'inestable (1).
        "salt": afina(dins[0] - 1, dins[0], (0, 1)),
        # A dalt, l'inestable (1) i el de sota l'imant (2).
        "desenganxada": afina(dins[-1] + 1, dins[-1], (1, 2)),
    }


def _afina_plec(x0, sep0, n, K, h0, L, g):
    """El plec exacte des d'una llavor: on l'acceleracio i la seva derivada
    valen zero alhora."""

    def sistema(u):
        x, sep = float(u[0]), float(u[1])
        if not (0 < x < min(sep, L)):
            return [1e6, 1e6]
        e = 1e-9
        f0 = float(_ax(np.array([x]), sep, n, K, h0, L, g)[0])
        fp = float(
            (_ax(np.array([x + e]), sep, n, K, h0, L, g)[0]
             - _ax(np.array([x - e]), sep, n, K, h0, L, g)[0]) / (2 * e)
        )
        return [f0, fp]

    sol, _, be, _ = fsolve(sistema, [x0, sep0], full_output=True)
    if be != 1 or not (0 < sol[0] < min(sol[1], L)):
        return math.nan, math.nan
    return float(sol[1]), float(sol[0])


def distancia_critica(n, K, h0, L, g, llavor=None):
    """El sep on desapareix l'equilibri de prop del centre. Retorna (sep_c, x_c).

    Es resol el PLEC: l'acceleracio i la seva derivada valen zero alhora, que es
    on les dues arrels es fonen. La bisseccio nomes serveix per trobar per on
    comencar; qui dona el numero es el fsolve.

    NOMES DONA EL PLEC DE BAIX. L'altre, el de la desenganxada, el dona plecs().
    """
    if llavor is None:
        # La banda dels tres equilibris no arriba ni a sep = 0 ni a sep gran, o
        # sigui que no es pot bisseccionar des dels extrems: primer s'ha de
        # TROBAR la banda escombrant, i despres afinar-ne la vora de sota.
        escombrat = np.linspace(2e-3, 0.9 * L, 400)
        quants = [len(equilibris(s, n, K, h0, L, g)) for s in escombrat]
        dins = [i for i, q in enumerate(quants) if q >= 3]
        if not dins or dins[0] == 0:
            return math.nan, math.nan
        baix, alt = escombrat[dins[0] - 1], escombrat[dins[0]]
        for _ in range(50):
            mig = 0.5 * (baix + alt)
            if len(equilibris(mig, n, K, h0, L, g)) >= 3:
                alt = mig
            else:
                baix = mig
        arrels = equilibris(alt, n, K, h0, L, g)
        llavor = (0.5 * (arrels[0] + arrels[1]), alt)

    def plec(u):
        x, sep = float(u[0]), float(u[1])
        if not (0 < x < min(sep, L)):
            return [1e6, 1e6]
        e = 1e-9
        f0 = float(_ax(np.array([x]), sep, n, K, h0, L, g)[0])
        fp = float(
            (_ax(np.array([x + e]), sep, n, K, h0, L, g)[0]
             - _ax(np.array([x - e]), sep, n, K, h0, L, g)[0]) / (2 * e)
        )
        return [f0, fp]

    sol, _, be, _ = fsolve(plec, [llavor[0], llavor[1]], full_output=True)
    if be != 1 or not (0 < sol[0] < min(sol[1], L)):
        return math.nan, math.nan
    sep_c = float(sol[1])
    # Que sigui el plec DE SOTA, i no el de dalt: just per sobre hi ha d'haver
    # els tres equilibris i just per sota, un de sol. Sense aquesta comprovacio,
    # amb una llavor dolenta el fsolve podria caure a l'altre plec i el numero
    # semblaria bo.
    e = max(1e-5, 1e-4 * sep_c)
    if not (len(equilibris(sep_c + e, n, K, h0, L, g)) >= 3
            and len(equilibris(sep_c - e, n, K, h0, L, g)) < 3):
        return math.nan, math.nan
    return sep_c, float(sol[0])


# ---------------------------------------------------------------------------
# L'analisi sencera
# ---------------------------------------------------------------------------
def analitza(punts, p: Parametres, sistematiques=None, sortides=SORTIDES_PER_DEFECTE,
             llavor=LLAVOR_PER_DEFECTE, alcada_fixa=False):
    """De la llista de punts a tots els numeros. No dibuixa ni imprimeix res.

    Els punts on la conversio a forca no existeix queden fora de la recta i van
    a res["fora_de_la_recta"] (vegeu separa_convertibles).
    """
    sistematiques = sistematiques or {}
    g = p.g

    punts, fora = separa_convertibles(punts, g, alcada_fixa)
    sep, x, h0, L = vectors(punts)
    m = magnituds(sep, x, h0, L, g, alcada_fixa)
    ld = np.log(m["delta"])
    lf = np.log(m["f"])

    # Les sigmes que falten: dels residus d'un primer ajust sense pesos.
    falten = [i for i, q in enumerate(punts) if q.sigma_x is None]
    heretada = None
    if falten:
        # LA DISPERSIO ES DE LA SEVA PROPIA TANDA, NO DE TOTES.
        # Si les tandes no cauen sobre la mateixa recta, els residus de l'ajust
        # conjunt no son error de mesura: son l'error del MODEL. Fer-los servir
        # de sigma inflaria la barra de cada tanda just amb allo que despres es
        # vol posar a prova, i la comparacio entre tandes es convertiria en una
        # que no pot sortir malament mai. La sigma d'un punt ha de sortir de la
        # dispersio dels seus VEINS, no de la discrepancia amb els d'un altre
        # regim.
        #
        # I ES UNA SIGMA DE LECTURA, EN MIL.LIMETRES, COMUNA A LA TANDA (25/10).
        # El que es llegeix al video es el desviament, i el seu error es el mateix
        # a tots els punts; el de ln F, en canvi, es aquell multiplicat per la
        # sensibilitat, que amb el balanc exacte va de 1 a centenars d'un punt a
        # l'altre. Fins al 25/10 es deduia una dispersio comuna de ln F, que dona
        # el mateix pes a tots els punts: el que la massa ha quedat a sobre de
        # l'imant, on una decima de mil.limetre canvia la forca un factor 3, hi
        # pesava com els altres i s'enduia la recta. Ara: recta ponderada amb
        # sigma(ln F) = sensibilitat * sigma_x (no depen del valor de sigma_x), i
        # la sigma_x es la que fa chi2 = graus de llibertat.
        sensibilitat = np.abs(sensibilitat_lnf(punts, g, alcada_fixa))
        comunes = {}
        for t in sorted({punts[i].tanda for i in falten}):
            seus = [i for i, q in enumerate(punts) if q.tanda == t]
            if len(seus) >= 3:
                j = np.array(seus)
                d_de = f"la tanda {t} sola ({len(seus)} punts)"
            else:
                j = np.arange(len(punts))
                d_de = f"totes les tandes -- la {t} no te prou punts per si sola"
            residus = recta(ld[j], lf[j], sensibilitat[j])["residus"]
            graus = max(len(j) - 2, 1)
            comunes[t] = (float(np.sqrt(np.sum((residus / sensibilitat[j]) ** 2) / graus)), d_de)
        for i in falten:
            punts[i].sigma_x = comunes[punts[i].tanda][0]
        heretada = {
            "sigma_x_comuna_m": {str(t): v[0] for t, v in comunes.items()},
            "d_on_surt": {str(t): v[1] for t, v in comunes.items()},
            "punts": [punts[i].sep for i in falten],
            "tandes": sorted({punts[i].tanda for i in falten}),
            "sigma_x_m": [punts[i].sigma_x for i in falten],
        }

    sigma_x = np.array([q.sigma_x for q in punts])
    sigma_lf = sigma_de_lnf(punts, g, sigma_x, alcada_fixa)

    grups = {"totes": list(range(len(punts)))}
    quines = sorted({q.tanda for q in punts})
    if len(quines) > 1:  # amb una sola tanda, "totes" i "tanda 1" serien la mateixa
        for t in quines:
            idx = [i for i, q in enumerate(punts) if q.tanda == t]
            if len(idx) >= 3:
                grups[f"tanda {t}"] = idx

    ajustos = {}
    for nom, idx in grups.items():
        i = np.array(idx)
        quadrats = recta(ld[i], lf[i], sigma_lf[i])
        mc = monte_carlo([punts[k] for k in idx], g, sistematiques, sortides,
                         llavor, alcada_fixa)
        # La distancia de referencia: alla on n i lnK deixen d'anar lligades, i
        # per tant on la forca esta mesurada de debo en comptes d'extrapolada.
        d_ref = float(np.exp(np.mean(ld[i])))
        ajustos[nom] = {
            "quadrats": {k: v for k, v in quadrats.items() if k != "residus"},
            "residus": quadrats["residus"].tolist(),
            "monte_carlo": {
                "n": percentils(mc["n"]),
                "K": percentils(np.exp(mc["lnK"])),
                "lnK": percentils(mc["lnK"]),
                "forquilla_K": float(np.exp(np.percentile(mc["lnK"], 84.135)
                                            - np.percentile(mc["lnK"], 15.865))),
                "correlacio_n_lnK": float(np.corrcoef(mc["n"], mc["lnK"])[0, 1]),
                "descartats": mc["descartats"],
                "sortides": mc["sortides"],
                "punts_sense_conversio": mc["punts_sense_conversio"],
            },
            "referencia": {
                "delta_m": d_ref,
                "F_per_massa": percentils(np.exp(mc["lnK"] - mc["n"] * math.log(d_ref))),
            },
            "_nuvol": (mc["n"], mc["lnK"]),
            "_index": idx,
        }

    res = {
        "punts": [
            {
                "tanda": q.tanda,
                "sep_m": q.sep,
                "h0_m": q.h0,
                "L_m": q.L,
                "x_m": q.x,
                "lectures_m": q.lectures.tolist(),
                "sigma_x_m": q.sigma_x,
                "sigma_de": q.sigma_de,
                "costats": list(q.costats),
                "x_per_costat_m": q.x_per_costat,
                "sigma_repeticions_m": q.sigma_repeticions,
                "desnivell_dels_costats_m": q.desnivell,
                "a_m": float(m["a"][i]),
                "h_m": float(m["h"][i]),
                "delta_m": float(m["delta"][i]),
                "F_hor_per_massa": float(m["f_hor"][i]),
                "F_per_massa": float(m["f"][i]),
                "amplificacio": float(m["amplificacio"][i]),
                "sigma_lnF": float(sigma_lf[i]),
            }
            for i, q in enumerate(punts)
        ],
        "fora_de_la_recta": [
            {
                "tanda": e["punt"].tanda,
                "sep_m": e["punt"].sep,
                "h0_m": e["punt"].h0,
                "L_m": e["punt"].L,
                "x_m": e["punt"].x,
                "lectures_m": e["punt"].lectures.tolist(),
                "denominador_m": e["denominador_m"],
                "motiu": "la conversio a forca no existeix: la massa es llegeix mes "
                         "enlla d'on el fil apunta a l'imant (a <= h*x/s)",
            }
            for e in fora
        ],
        "ajustos": ajustos,
        "sigma_heretada": heretada,
        "alcada_fixa": bool(alcada_fixa),
        "recorregut_de_delta": float(m["delta"].max() / m["delta"].min()),
        "amplificacio_maxima": float(np.max(m["amplificacio"])),
        "g_m_s2": float(g),
    }

    # La comparacio entre tandes, si n'hi ha mes d'una.
    noms = [k for k in ajustos if k.startswith("tanda")]
    if len(noms) == 2:
        a, b = (ajustos[k] for k in noms)
        na, nb = a["monte_carlo"]["n"], b["monte_carlo"]["n"]
        sigma = math.hypot(na["sigma"], nb["sigma"])
        # I a una distancia comuna, perque n i K van correlacionades i comparar
        # nomes les n pot dir que discrepen dues rectes que se superposen.
        d_com = math.sqrt(a["referencia"]["delta_m"] * b["referencia"]["delta_m"])
        fa = np.exp(a["_nuvol"][1] - a["_nuvol"][0] * math.log(d_com))
        fb = np.exp(b["_nuvol"][1] - b["_nuvol"][0] * math.log(d_com))
        pa, pb = percentils(np.log(fa)), percentils(np.log(fb))
        res["comparacio"] = {
            "tandes": noms,
            "n": [na["p50"], nb["p50"]],
            "diferencia_sigmes": abs(na["p50"] - nb["p50"]) / sigma if sigma > 0 else float("inf"),
            "delta_comuna_m": d_com,
            "F_a_la_delta_comuna": [math.exp(pa["p50"]), math.exp(pb["p50"])],
            "diferencia_F_sigmes": (abs(pa["p50"] - pb["p50"])
                                    / math.hypot(pa["sigma"], pb["sigma"])),
        }

    return res


def afegeix_comprovacions(res, punts, p: Parametres, sortides=SORTIDES_CRITICA):
    """Les dues comprovacions que no surten de l'ajust: el salt i el mapa."""
    millor = res["ajustos"]["totes"]
    n = millor["monte_carlo"]["n"]["p50"]
    K = millor["monte_carlo"]["K"]["p50"]

    # 1. La distancia critica, per cada geometria I amb cada llei ajustada.
    #
    # PER QUE AMB CADA LLEI I NO NOMES AMB LA CONJUNTA. El salt passa a prop de
    # l'imant, o sigui que el prediu la llei del regim DE PROP. Si les tandes no
    # cauen sobre la mateixa recta, la conjunta es un compromis que no descriu
    # cap dels dos extrems, i comparar-la amb el que es va veure al banc no
    # comprovaria res. Fent-ho amb totes, la comparacio passa a dir QUINA llei
    # descriu el que va passar, que es molt mes util que un si o un no.
    critiques = {}
    for tanda in sorted({q.tanda for q in punts}):
        dels = [q for q in punts if q.tanda == tanda]
        h0, L = dels[0].h0, dels[0].L
        per_llei = {}
        for nom, aj in res["ajustos"].items():
            nn0 = aj["monte_carlo"]["n"]["p50"]
            KK0 = aj["monte_carlo"]["K"]["p50"]
            sep_c, x_c = distancia_critica(nn0, KK0, h0, L, p.g)
            mostra = []
            if not math.isnan(sep_c):
                ns, lnKs = aj["_nuvol"]
                pas = max(1, len(ns) // sortides)
                for nn, lk in zip(ns[::pas], lnKs[::pas]):
                    s, _ = distancia_critica(float(nn), float(math.exp(lk)), h0, L, p.g,
                                             llavor=(x_c, sep_c))
                    if not math.isnan(s):
                        mostra.append(s)
            els_dos = plecs(nn0, KK0, h0, L, p.g)
            per_llei[nom] = {
                "n": nn0,
                "K": KK0,
                "sep_critic_m": sep_c,
                "x_al_plec_m": x_c,
                "forquilla": percentils(mostra) if len(mostra) > 10 else None,
                "sep_del_salt_m": els_dos["salt"][0],
                "sep_de_la_desenganxada_m": els_dos["desenganxada"][0],
            }
        critiques[str(tanda)] = {"h0_m": h0, "L_m": L, "per_llei": per_llei}
    res["distancia_critica"] = critiques

    # 1b. Cada llei contra L'OBSERVABLE CRU, en mil.limetres de desviament.
    #
    # PER QUE AIXO NO ES REDUNDANT AMB ELS RESIDUS DE L'AJUST. Prop de l'imant
    # la forca creix moltissim amb la posicio, i per tant la posicio d'equilibri
    # es MOLT POC SENSIBLE a la llei: dues lleis que difereixen en un factor 3 a
    # la mateixa distancia poden posar la massa gairebe al mateix lloc. En
    # logaritmes de forca aquella diferencia sembla enorme; en mil.limetres de
    # desviament -- que es l'unic que es va mesurar de debo, i de qui es coneix
    # la precisio -- pot ser un no-res.
    #
    # Comparar nomes en log-log fa semblar que la tanda 1 fixa l'exponent, i no
    # el fixa: el que fa es fixar molt be on es posa la massa.
    #
    # Hi entren TAMBE els punts que han quedat fora de la recta: el desviament
    # es una mesura tan bona com les altres, nomes es la conversio a forca que
    # no hi existeix.
    fora = {(e["tanda"], e["sep_m"]) for e in res.get("fora_de_la_recta", [])}
    en_x = {}
    for nom, aj in res["ajustos"].items():
        nn0 = aj["monte_carlo"]["n"]["p50"]
        KK0 = aj["monte_carlo"]["K"]["p50"]
        files = []
        for q in punts:
            arrels = equilibris(q.sep, nn0, KK0, q.h0, q.L, p.g)
            prediu = min(arrels, key=lambda r: abs(r - q.x)) if arrels else math.nan
            files.append({"tanda": q.tanda, "sep_m": q.sep, "x_m": q.x,
                          "x_previst_m": prediu, "residu_m": prediu - q.x,
                          "a_la_recta": (q.tanda, q.sep) not in fora})
        bons = [f["residu_m"] for f in files if not math.isnan(f["residu_m"])]
        en_x[nom] = {
            "n": nn0, "K": KK0, "files": files,
            "rms_m": float(np.sqrt(np.mean(np.square(bons)))) if bons else math.nan,
        }
    res["contra_l_observable_cru"] = en_x

    # 2. Que li fa al mapa: el centre continua sent atractor?
    p2, avis = fisica.sobreescriu(p, n=n, K=K)
    consequencia = {
        "n": n,
        "K": K,
        "avis": avis,
        "factor_critic": float(fisica.factor_critic(p2)),
    }
    try:
        eq_x, eq_y, estable = fisica.punts_dequilibri(p2)
        consequencia["centre_estable"] = bool(estable)
        consequencia["radi_dels_punts_de_repos_m"] = float(
            np.hypot(eq_x[0], eq_y[0])
        )
        consequencia["periode_local_s"] = float(fisica.periode_local(p2, eq_x[0], eq_y[0]))
    except RuntimeError as e:
        consequencia["centre_estable"] = None
        consequencia["problema"] = str(e)
    res["consequencia_per_al_mapa"] = consequencia
    return res


# ---------------------------------------------------------------------------
# El que surt per pantalla
# ---------------------------------------------------------------------------
def escriu(res, p: Parametres) -> None:
    mm, cm = 1000.0, 100.0

    print("\n-- LES COLUMNES DEL FULL DEL BANC, CALCULADES " + "-" * 27)
    print("   (F es PER UNITAT DE MASSA; entre parentesi, en newtons amb m ="
          f" {p.m * 1000:.1f} g)\n")
    print(f"   {'t':>2} {'sep':>7} {'x':>8} {'a':>8} {'h':>8} {'d':>8} "
          f"{'F_hor/m':>9} {'F/m':>9} {'ampl':>7}")
    print(f"   {'':>2} {'(cm)':>7} {'(mm)':>8} {'(mm)':>8} {'(mm)':>8} {'(mm)':>8} "
          f"{'(m/s2)':>9} {'(m/s2)':>9} {'':>7}")
    for q in res["punts"]:
        print(f"   {q['tanda']:>2} {q['sep_m']*cm:>7.2f} {q['x_m']*mm:>8.3f} "
              f"{q['a_m']*mm:>8.3f} {q['h_m']*mm:>8.3f} {q['delta_m']*mm:>8.3f} "
              f"{q['F_hor_per_massa']:>9.4f} {q['F_per_massa']:>9.4f} "
              f"{q['amplificacio']:>7.2f}")
    print("\n   ampl = d/(a - h*x/s): per quant es multiplica la F horitzontal per")
    print("   tenir la total" + (" (amb --alcada-fixa, d/a)." if res["alcada_fixa"] else "."))
    print(f"   la d recorre un factor {res['recorregut_de_delta']:.3f}"
          f"   i l'amplificacio arriba a x{res['amplificacio_maxima']:.1f}")
    if res["amplificacio_maxima"] > 5:
        print("   !! Amb una amplificacio aixi, del video nomes se'n veu una part")
        print("      petita de la forca, i la propagacio lineal d'errors no hi val.")

    if res["fora_de_la_recta"]:
        print("\n   " + "!" * 70)
        print("   !! PUNTS FORA DE LA RECTA: aqui la conversio a forca NO EXISTEIX")
        print("   " + "!" * 70)
        print(f"   {'t':>2} {'sep':>7} {'x':>8} {'a - h*x/s':>10}")
        print(f"   {'':>2} {'(cm)':>7} {'(mm)':>8} {'(mm)':>10}")
        for q in res["fora_de_la_recta"]:
            print(f"   {q['tanda']:>2} {q['sep_m']*cm:>7.2f} {q['x_m']*mm:>8.3f} "
                  f"{q['denominador_m']*mm:>10.3f}")
        print("   La massa es llegeix mes enlla d'on el fil apunta a l'imant, i alla")
        print("   cap forca central no la pot sostenir. Vol dir que l'imant l'ha")
        print("   capturada i ha quedat gairebe a sobre: on es para ho decideix el fil,")
        print("   no la llei, i dues decimes de mil.limetre de lectura ja la passen a")
        print("   l'altra banda. NO es cap llindar triat: on la conversio existeix, el")
        print("   punt entra amb el pes que li toca. Mes avall, en mil.limetres de")
        print("   desviament, aquests punts SI que es comparen amb cada llei.")

    dos_costats = [q for q in res["punts"] if q["desnivell_dels_costats_m"] is not None]
    if dos_costats:
        print("\n-- ELS DOS COSTATS, I QUE HI HA DINS DE LA SIGMA " + "-" * 24)
        print("   Mesurant pels dos costats, un error de ZERO de l'eix entra com a")
        print("   x+e i x-e i la mitjana el cancel.la. Pero llavors el desnivell que")
        print("   queda entre els dos costats JA NO es aquell error: es una altra cosa,")
        print("   i val la pena mirar-la abans de dir-ne incertesa.\n")
        print(f"   {'t':>2} {'sep':>7} {'costat +':>9} {'costat -':>9} {'desnivell':>10} "
              f"{'%de x':>7} {'s(6 lect)':>10} {'s(repet)':>9} {'quant es':>9}")
        print(f"   {'':>2} {'(cm)':>7} {'(mm)':>9} {'(mm)':>9} {'(mm)':>10} "
              f"{'':>7} {'(mm)':>10} {'(mm)':>9} {'desnivell':>9}")
        for q in dos_costats:
            c = q["x_per_costat_m"]
            mes = c.get("+", float("nan")) * mm
            menys = c.get("-", float("nan")) * mm
            s6 = q["sigma_x_m"] * mm
            sr = q["sigma_repeticions_m"]
            sr_txt = f"{sr*mm:9.4f}" if sr is not None else f"{'--':>9}"
            part = f"{math.sqrt(max(s6**2 - (sr*mm)**2, 0.0))/s6:9.0%}" if sr else f"{'--':>9}"
            print(f"   {q['tanda']:>2} {q['sep_m']*cm:>7.2f} {mes:>9.4f} {menys:>9.4f} "
                  f"{q['desnivell_dels_costats_m']*mm:>10.4f} "
                  f"{q['desnivell_dels_costats_m']/q['x_m']:>7.1%} {s6:>10.4f} {sr_txt} {part}")
        proporcions = [q["desnivell_dels_costats_m"] / q["x_m"] for q in dos_costats]
        absoluts = [q["desnivell_dels_costats_m"] for q in dos_costats]
        disp_rel = float(np.std(proporcions) / np.mean(proporcions))
        disp_abs = float(np.std(absoluts) / np.mean(absoluts))
        print(f"\n   El desnivell, en mil.limetres, varia un {disp_abs:.0%} entre punts;")
        print(f"   en tant per cent de la x, un {disp_rel:.0%}.")
        if disp_rel < disp_abs:
            print("   O sigui que es PROPORCIONAL, no constant: NO es un error de zero")
            print("   (aquell seria els mateixos mil.limetres a tot arreu). Es mes aviat")
            print("   que l'imant no era a la mateixa distancia pels dos costats.")
            print("   La mitjana ho corregeix a primer ordre, pero no exactament.")
        else:
            print("   O sigui que es CONSTANT en mil.limetres: te la cara d'un error de")
            print("   zero de l'eix, i fer la mitjana dels dos costats el cancel.la.")
        print("\n   LA QUE ES FA SERVIR ES LA DE LES 6 LECTURES, la de la columna")
        print("   's(6 lect)': es mes ampla perque hi inclou el desnivell, i mentre no")
        print("   se sapiga d'on surt, tractar-lo com a incertesa es el que toca.")

    comunes = {}
    for q in res["punts"]:
        if str(q["sigma_de"]).startswith("lectura comuna"):
            comunes.setdefault(q["tanda"], (q["sigma_de"], []))[1].append(q["sep_m"])
    if comunes:
        print("\n-- LA SIGMA DE LECTURA COMUNA " + "-" * 43)
        print(f"   Els punts amb menys de {MIN_LECTURES_PROPIES} lectures no fan servir la seva")
        print("   propia dispersio, que amb un o dos graus de llibertat no es fiable,")
        print("   sino la de la seva tanda, amb totes les repeticions agrupades:")
        for t, (text, seps) in sorted(comunes.items()):
            print(f"      tanda {t}: {text}")
            print("         punts (sep, cm): " + ", ".join(f"{s*cm:.1f}" for s in seps))

    if res["sigma_heretada"]:
        h = res["sigma_heretada"]
        print("\n   !! ATENCIO: hi ha punts SENSE REPETICIONS i sense sigma declarada.")
        print("      La seva sigma s'ha deduit dels residus de l'ajust, o sigui que")
        print("      NO es una mesura independent de l'error: surt de suposar que el")
        print("      model es bo. Amb repeticions, ho seria.")
        for t, s in h["sigma_x_comuna_m"].items():
            print(f"      tanda {t}: sigma de lectura = {s*mm:.3f} mm, comuna a tots els")
            print(f"         punts, de {h['d_on_surt'][t]}")
        print("      Punts afectats (sep, cm): "
              + ", ".join(f"{s*cm:.1f}" for s in h["punts"]))

    print("\n-- L'AJUST  ln(F/m) = ln K - n*ln(delta) " + "-" * 33)
    if res["alcada_fixa"]:
        print("   !! --alcada-fixa: aquest ajust IGNORA que la massa puja en desviar-se.")
        print("      Es l'ajust DOLENT, hi es per ensenyar-lo, no per fer-lo servir.")
    for nom, aj in res["ajustos"].items():
        q, mc = aj["quadrats"], aj["monte_carlo"]
        print(f"\n   {nom}  ({q['n_punts']} punts, R2 = {q['r2']:.4f})")
        print(f"      minims quadrats   n = {q['n']:.3f} +- {q['sigma_n']:.3f}"
              f"    K = {q['K']:.3e}  (ln K +- {q['sigma_lnK']:.3f})")
        print(f"      Monte Carlo       n = {mc['n']['p50']:.3f} +- {mc['n']['sigma']:.3f}"
              f"    K = {mc['K']['p50']:.3e}"
              f"  [{mc['K']['p16']:.2e}, {mc['K']['p84']:.2e}]")
        eixample = mc["n"]["sigma"] / q["sigma_n"] if q["sigma_n"] > 0 else float("nan")
        print(f"      la de Monte Carlo es x{eixample:.2f} la classica"
              f"    correlacio n/lnK = {mc['correlacio_n_lnK']:+.3f}")
        print(f"      forquilla de K al 68%: un factor {mc['forquilla_K']:.1f}")
        if mc["punts_sense_conversio"]:
            print(f"      en {mc['punts_sense_conversio']} casos (punt x sorteig) la conversio"
                  f" no existia i el punt va quedar fora d'aquell sorteig")
        ref = aj["referencia"]
        print(f"      a delta = {ref['delta_m']*mm:.2f} mm (on n i K es descorrelacionen):")
        print(f"         F/m = {ref['F_per_massa']['p50']:.4f} +- "
              f"{ref['F_per_massa']['sigma']:.4f} m/s2   <- el numero que de debo esta mesurat")

    q = res["ajustos"]["totes"]["quadrats"]
    mc = res["ajustos"]["totes"]["monte_carlo"]
    rao = mc["n"]["sigma"] / q["sigma_n"] if q["sigma_n"] > 0 else float("nan")
    print("\n   PER QUE ES REPORTA LA DE MONTE CARLO: la classica propaga a PRIMER")
    print("   ORDRE, i aqui la forca total surt de multiplicar la horitzontal per")
    print(f"   l'amplificacio, que arriba a {res['amplificacio_maxima']:.1f}. Amb aquesta")
    print("   amplificacio la relacio entre el que es llegeix i el que es vol ja no")
    print("   es lineal, i llavors la formula de la propagacio no descriu res.")
    print(f"   Mesurat aqui: la barra de Monte Carlo es x{rao:.2f} la classica.")
    if rao < 0.95:
        print("   Es a dir, MES ESTRETA. No es cap contradiccio amb el de sobre: el que")
        print("   diu es que la linealitzacio, a mes de no valer, aqui s'equivoca cap")
        print("   amunt. El que no es pot fer es donar per bona la classica sense mirar.")
    elif rao > 1.05:
        print("   Es a dir, MES AMPLA: la classica es queda curta, que es el cas que")
        print("   es temia. La que s'ha de reportar es la de Monte Carlo.")
    else:
        print("   O sigui que aqui les dues coincideixen i la linealitzacio hi valia.")
        print("   Val la pena dir-ho a la memoria: no sempre passa, i s'ha comprovat.")

    if "comparacio" in res:
        c = res["comparacio"]
        print("\n-- LES DUES TANDES, CAUEN SOBRE LA MATEIXA RECTA? " + "-" * 24)
        print(f"   n = {c['n'][0]:.3f} i {c['n'][1]:.3f}   ->  "
              f"difereixen {c['diferencia_sigmes']:.1f} sigma")
        print(f"   a delta = {c['delta_comuna_m']*mm:.2f} mm (comuna): "
              f"F/m = {c['F_a_la_delta_comuna'][0]:.4f} i "
              f"{c['F_a_la_delta_comuna'][1]:.4f}  ->  "
              f"{c['diferencia_F_sigmes']:.1f} sigma")
        print("   La segona linia es la que mana: n i K van correlacionades, i dues")
        print("   rectes que se superposen poden tenir n's que semblin discrepar.")

    if "contra_l_observable_cru" in res and len(res["contra_l_observable_cru"]) > 1:
        en_x = res["contra_l_observable_cru"]
        print("\n-- CADA LLEI CONTRA EL QUE ES VA MESURAR DE DEBO " + "-" * 24)
        print("   Els residus de l'ajust son en LOGARITMES DE FORCA, i alla una")
        print("   diferencia d'un factor 3 sembla enorme. Pero prop de l'imant la")
        print("   forca creix moltissim amb la posicio, i per tant on es posa la massa")
        print("   depen molt poc de la llei. Aixo es el mateix, en mil.limetres de")
        print("   desviament, que es l'unica cosa que es va llegir.\n")
        noms = list(en_x)
        print(f"   {'t':>2} {'sep':>6} {'mesurat':>9} " +
              "".join(f"{('«'+n+'»'):>16}" for n in noms))
        print(f"   {'':>2} {'(cm)':>6} {'(mm)':>9} " +
              "".join(f"{'previst  residu':>16}" for _ in noms))
        for i, q in enumerate(en_x[noms[0]]["files"]):
            linia = f"   {q['tanda']:>2} {q['sep_m']*cm:>6.2f} {q['x_m']*mm:>9.3f} "
            for n_ in noms:
                f = en_x[n_]["files"][i]
                linia += f"{f['x_previst_m']*mm:>8.2f} {f['residu_m']*mm:>+7.2f}"
            print(linia + ("" if q.get("a_la_recta", True) else "   * fora de la recta"))
        print(f"\n   {'rms del residu (mm)':>19} " +
              "".join(f"{en_x[n_]['rms_m']*mm:>16.3f}" for n_ in noms))
        if any(not q.get("a_la_recta", True) for q in en_x[noms[0]]["files"]):
            print("   (l'rms compta tots els punts, tambe els de fora de la recta)")
        millor_x = min(noms, key=lambda n_: en_x[n_]["rms_m"])
        print(f"\n   La que reprodueix millor el que es va mesurar es «{millor_x}».")
        print("   COMPARA AQUESTS MIL.LIMETRES AMB LA TEVA PRECISIO DE LECTURA: si el")
        print("   residu d'una llei hi cap a dins, aquelles dades NO la descarten, per")
        print("   molt que en log-log sembli que si.")

    if "distancia_critica" in res:
        print("\n-- ELS DOS PLECS: LA COMPROVACIO INDEPENDENT " + "-" * 29)
        print("   Hi ha una banda de distancies amb TRES equilibris, i te DUES vores:")
        print("      SALT           acostant l'imant, la massa se'n va de cop cap a ell")
        print("      DESENGANXADA   allunyant-lo, la massa ja enganxada torna al centre")
        print("   Entre les dues el sistema es BIESTABLE i on es posa la massa depen de")
        print("   com hi hagi arribat. L'AJUST NO SAP RES DE CAP DE LES DUES.")
        for tanda, c in res["distancia_critica"].items():
            print(f"\n   Geometria de la tanda {tanda} (h0 = {c['h0_m']*cm:.1f} cm, "
                  f"L = {c['L_m']*cm:.1f} cm):")
            print(f"      {'llei':>12} {'n':>6}   {'salta a':>9} {'es desenganxa a':>17}")
            for nom, v in c["per_llei"].items():
                s = v["sep_del_salt_m"]
                d = v["sep_de_la_desenganxada_m"]
                txt_s = f"{s*cm:8.2f} cm" if not math.isnan(s) else f"{'--':>11}"
                txt_d = f"{d*cm:14.2f} cm" if not math.isnan(d) else f"{'--':>17}"
                print(f"      {('«'+nom+'»'):>12} {v['n']:>6.2f}   {txt_s} {txt_d}")
        print("\n   COMPARA-HO AMB EL QUE VAS VEURE, I MIRA QUINA DE LES DUES COSES ERA.")
        print("   «Mes enlla de tants cm la gravetat guanya» es la DESENGANXADA, no el")
        print("   salt. Prendre-ho pel salt fa sortir una distancia critica que no ho")
        print("   es, i llavors sembla que confirmi una llei quan en confirma una altra.")

    if "consequencia_per_al_mapa" in res:
        c = res["consequencia_per_al_mapa"]
        print("\n-- QUE LI FA AL MAPA " + "-" * 52)
        print(f"   Amb n = {c['n']:.3f} i K = {c['K']:.3e} als tres imants:")
        if c.get("centre_estable") is None:
            print(f"   {c.get('problema', '')}")
        else:
            if c["centre_estable"]:
                print(f"   el centre ES atractor: la regio negra del mapa existeix,")
                print(f"   i les K encara podrien creixer x{c['factor_critic']:.3f} "
                      "abans que deixes de ser-ho.")
            else:
                print("   el centre ES INESTABLE: amb aquests valors NO hi ha regio negra.")
                print(f"   Caldria dividir les K per {1/c['factor_critic']:.3f} perque n'hi hagues.")
            print(f"   punts de repos a r = {c['radi_dels_punts_de_repos_m']*cm:.2f} cm "
                  f"(els imants son a {p.R*cm:.2f}), periode local "
                  f"{c['periode_local_s']:.3f} s")
        print("\n   AIXO NO ES CAP MAPA: es el que en prediu l'algebra. El mapa de debo")
        print("   demana canviar parametres.json i tornar a executar-ho tot, i aquest")
        print("   programa NO el toca.")


# ---------------------------------------------------------------------------
# La figura
# ---------------------------------------------------------------------------
def _eix_log_llegible(ax, valors, quants=5):
    """Escala log amb els numeros escrits tal com son, no com a 2.1x10^1.

    Amb un rang de nomes un factor 1,4 les marques automatiques del log cauen
    totes a les menors i matplotlib les escriu en notacio cientifica: surten
    vuit rotuls encavalcats que no es poden llegir. Aqui es posen unes quantes
    marques triades i s'escriuen en clar.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.set_xscale("log")
    baix, alt = float(np.min(valors)), float(np.max(valors))
    # Espaiades en LOG, no en lineal: en un eix log, unes marques equiespaiades
    # s'apinyen totes a la dreta i deixen buida l'esquerra.
    marques = np.unique(np.round(np.geomspace(baix, alt, quants), 1))
    ax.xaxis.set_major_locator(FixedLocator(marques))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _eix_y_log_llegible(ax, valors, quants=5):
    """El mateix per a l'eix vertical."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ax.set_yscale("log")
    baix, alt = float(np.min(valors)), float(np.max(valors))
    marques = np.geomspace(baix, alt, quants)
    ax.yaxis.set_major_locator(FixedLocator(marques))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{v:.3g}" if v >= 0.1 else f"{v:.2g}")
    )
    ax.yaxis.set_minor_formatter(NullFormatter())


def dibuixa(res, nom, directori, mostra=False) -> None:
    estil.aplica()
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.4))
    mm = 1000.0

    tandes = sorted({q["tanda"] for q in res["punts"]})
    color = {t: estil.IMANTS[i % len(estil.IMANTS)] for i, t in enumerate(tandes)}
    ld = np.array([math.log(q["delta_m"]) for q in res["punts"]])
    lf = np.array([math.log(q["F_per_massa"]) for q in res["punts"]])

    # 1. la recta log-log
    ax = axs[0, 0]
    aj = res["ajustos"]["totes"]
    graella = np.linspace(ld.min() - 0.06, ld.max() + 0.06, 200)
    ns, lnKs = aj["_nuvol"]
    pas = max(1, len(ns) // 600)
    banda = np.array([lnKs[::pas][i] - ns[::pas][i] * graella for i in range(len(ns[::pas]))])
    ax.fill_between(np.exp(graella) * mm, np.exp(np.percentile(banda, 15.865, axis=0)),
                    np.exp(np.percentile(banda, 84.135, axis=0)),
                    color=estil.TINTA_SUAU, alpha=0.22, lw=0,
                    label="Monte Carlo, 68 %")
    for t in tandes:
        i = [k for k, q in enumerate(res["punts"]) if q["tanda"] == t]
        ax.plot(np.exp(ld[i]) * mm, np.exp(lf[i]), "o", ms=6, color=color[t],
                mfc="none", mew=1.6, label=f"tanda {t}")
        clau = f"tanda {t}"
        if clau in res["ajustos"]:
            q = res["ajustos"][clau]["monte_carlo"]
            ax.plot(np.exp(graella) * mm,
                    np.exp(q["lnK"]["p50"] - q["n"]["p50"] * graella),
                    "--", lw=1.1, color=color[t], alpha=0.8)
    ax.plot(np.exp(graella) * mm,
            np.exp(aj["monte_carlo"]["lnK"]["p50"] - aj["monte_carlo"]["n"]["p50"] * graella),
            "-", lw=1.6, color=estil.TINTA,
            label=f"n = {aj['monte_carlo']['n']['p50']:.2f} "
                  f"+- {aj['monte_carlo']['n']['sigma']:.2f}")
    _eix_log_llegible(ax, np.exp(graella) * mm)
    _eix_y_log_llegible(ax, np.exp(lf))
    ax.set_xlabel("delta, distancia massa-imant (mm)")
    ax.set_ylabel("F / m  (m/s2)")
    ax.set_title("la llei de forca: ha de ser una RECTA en log-log", loc="left")
    ax.legend(loc="best")

    # 2. els residus
    ax = axs[0, 1]
    ax.axhline(0, lw=0.8, color=estil.EIX)
    residus, sigmes = [], []
    for t in tandes:
        i = [k for k, q in enumerate(res["punts"]) if q["tanda"] == t]
        r = lf[i] - (aj["monte_carlo"]["lnK"]["p50"]
                     - aj["monte_carlo"]["n"]["p50"] * ld[i])
        s = np.array([res["punts"][k]["sigma_lnF"] for k in i])
        residus += list(r)
        sigmes += list(s)
        ax.errorbar(np.exp(ld[i]) * mm, r, yerr=s, fmt="o", ms=5, mfc="none",
                    mew=1.5, color=color[t], elinewidth=1.0, capsize=2.5,
                    label=f"tanda {t}")
    # L'eix el fixen els PUNTS, no les barres: un punt a tocar de la captura
    # porta una barra de desenes en ln F, i deixaria tots els altres aixafats a
    # zero. Les barres que no hi caben es retallen, i es diu.
    tipiques = [s for s in sigmes if s <= 2.0 * np.median(sigmes)]
    sostre = 1.3 * max(float(np.max(np.abs(residus))), max(tipiques, default=0.0)) + 0.05
    ax.set_ylim(-sostre, sostre)
    retallades = [s for r_, s in zip(residus, sigmes) if abs(r_) + s > sostre]
    if retallades:
        ax.text(0.03, 0.04, f"barres retallades: la mes llarga, +- {max(retallades):.1f}",
                transform=ax.transAxes, fontsize=8.5, color=estil.TINTA_2, va="bottom")
    _eix_log_llegible(ax, np.exp(ld) * mm)
    ax.set_xlabel("delta (mm)")
    ax.set_ylabel("residu de ln(F/m)")
    ax.set_title("residus: si dibuixen una corba, la llei de potencia no val", loc="left")
    ax.legend(loc="best")

    # 3. l'observable cru: el que es va mesurar de debo
    ax = axs[1, 0]
    for t in tandes:
        dels = [q for q in res["punts"] if q["tanda"] == t]
        ax.plot([q["sep_m"] * 100 for q in dels], [q["x_m"] * mm for q in dels],
                "o", ms=6, mfc="none", mew=1.6, color=color[t], label=f"tanda {t}")
        fora = [q for q in res["fora_de_la_recta"] if q["tanda"] == t]
        if fora:
            ax.plot([q["sep_m"] * 100 for q in fora], [q["x_m"] * mm for q in fora],
                    "x", ms=7, mew=1.6, color=color[t],
                    label=f"tanda {t}, fora de la recta")
        # LA LLEI DE LA SEVA PROPIA TANDA, no la conjunta. Aquest panell serveix
        # per veure si el model reprodueix el que es va mesurar, i si les tandes
        # son de regims diferents la conjunta no descriu cap de les dues: hi
        # dibuixaria una corba que no passa pels punts de cap dels dos costats.
        clau = f"tanda {t}" if f"tanda {t}" in res["ajustos"] else "totes"
        mc = res["ajustos"][clau]["monte_carlo"]
        n50, K50 = mc["n"]["p50"], mc["K"]["p50"]
        h0, L = dels[0]["h0_m"], dels[0]["L_m"]
        seps = np.linspace(min(q["sep_m"] for q in dels) * 0.9,
                           max(q["sep_m"] for q in dels) * 1.1, 200)
        corba, anterior = [], None
        for s in seps:
            arrels = equilibris(s, n50, K50, h0, L, res["g_m_s2"])
            if not arrels:
                corba.append(math.nan)
                anterior = None
                continue
            # Se segueix la MATEIXA branca, per continuitat des del punt mesurat.
            # Si la branca desapareix, la corba es TALLA en comptes de saltar a
            # l'altra: un salt vertical dibuixat sembla una corba i no ho es.
            if anterior is None:
                anterior = min(dels, key=lambda q: abs(q["sep_m"] - s))["x_m"]
            tria = min(arrels, key=lambda r: abs(r - anterior))
            if abs(tria - anterior) > 0.25 * max(anterior, 1e-3):
                corba.append(math.nan)
                anterior = None
            else:
                corba.append(tria)
                anterior = tria
        ax.plot(seps * 100, np.array(corba) * mm, "-", lw=1.3, color=color[t],
                alpha=0.75, label=f"model de la tanda {t}" if len(tandes) > 1 else None)
    ax.set_xlabel("sep, on s'ha posat l'imant (cm)")
    ax.set_ylabel("x, desviament de la massa (mm)")
    ax.set_title("l'observable cru, amb el model de cada tanda a sobre", loc="left")
    ax.legend(loc="best")

    # 4. el nuvol de Monte Carlo
    ax = axs[1, 1]
    ax.plot(ns, np.exp(lnKs), ".", ms=1.4, alpha=0.25, color=estil.IMANTS[0])
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("K  (m^(n+1)/s2, per unitat de massa)")
    ax.set_title(f"n i K van lligades: correlacio "
                 f"{aj['monte_carlo']['correlacio_n_lnK']:+.2f}", loc="left")
    ax.text(0.03, 0.04,
            f"forquilla de K al 68 %: factor {aj['monte_carlo']['forquilla_K']:.1f}\n"
            "les unitats de K depenen de la n:\nsola no vol dir res",
            transform=ax.transAxes, fontsize=8.5, color=estil.TINTA_2, va="bottom")

    mc = aj["monte_carlo"]
    fig.suptitle(
        f"Pendol magnetic, llei de forca dels imants:  "
        f"n = {mc['n']['p50']:.3f} +- {mc['n']['sigma']:.3f},  "
        f"K = {mc['K']['p50']:.3e} m^(n+1)/s2"
        + ("   [ALCADA FIXA: ajust dolent]" if res["alcada_fixa"] else ""),
        fontsize=11,
    )
    fig.tight_layout()
    if mostra:
        plt.show()
    else:
        estil.desa(fig, nom, directori)


# ---------------------------------------------------------------------------
# L'autoprova
# ---------------------------------------------------------------------------
N_BONA, K_BONA = 3.2, 1.5e-4
SOROLL_BO = 0.0002  # 0,2 mm de lectura


def _mesures_fabricades(seps, h0, L, g, n, K, quin, rng, repeticions, soroll):
    """Fabrica lectures d'un muntatge del qual ja se sap la resposta."""
    files = []
    for sep in seps:
        arrels = equilibris(sep, n, K, h0, L, g)
        if not arrels:
            continue
        x = arrels[0] if quin == "exterior" else arrels[-1]
        for r in range(repeticions):
            for costat in ("+", "-"):
                files.append((sep, h0, costat, x + rng.normal(0.0, soroll), r + 1))
    return files


def autoprova(sortides=4000, llavor=LLAVOR_PER_DEFECTE) -> int:
    """Recupera una n i una K conegudes d'unes mesures fabricades amb soroll."""
    import pandas as pd

    print("=" * 74)
    print(" magnetic.llei_de_forca --autoprova")
    print("=" * 74)

    p = Parametres.des_de_json()
    L, g = p.L, p.g
    rng = np.random.default_rng(llavor)

    print(f"\n   posat:  n = {N_BONA}   K = {K_BONA:.4e} m^(n+1)/s2"
          f"   L = {L*100:.1f} cm   g = {g}")
    print(f"   soroll de lectura {SOROLL_BO*1000:.1f} mm, 3 repeticions, els dos costats")

    # Que _ax i fisica.acceleracio_nucli diguin el mateix NO es una promesa.
    pitjor = max(comprova_contra_fisica(N_BONA, K_BONA, 0.02, L, g, sep=s)
                 for s in (0.02, 0.05, 0.10))
    print(f"\n   _ax contra fisica.acceleracio_nucli: difereixen com a molt "
          f"{pitjor:.2e} m/s2")

    files = []
    # Tanda A: com la tanda 1 de debo -- l'imant a prop, i es mesura l'equilibri
    # de SOTA l'imant, el que queda despres del salt.
    files += [(1,) + f for f in _mesures_fabricades(
        [0.07, 0.06, 0.05, 0.04, 0.03, 0.02], 0.02, L, g, N_BONA, K_BONA,
        "interior", rng, 3, SOROLL_BO)]
    # Tanda B: com la tanda 2 -- l'imant mes avall i mes lluny, cap salt, i es
    # mesura l'equilibri de prop del centre.
    files += [(2,) + f for f in _mesures_fabricades(
        [0.10, 0.085, 0.07, 0.055, 0.04], 0.03, L, g, N_BONA, K_BONA,
        "exterior", rng, 3, SOROLL_BO)]
    # I un punt que cap forca central no pot sostenir: una sola lectura 0,1 mm
    # mes enlla d'on el fil apunta a l'imant, com la tanda 1 de debo a 3 i 4 cm.
    # Ha de quedar fora de la recta sense fer caure res ni moure la n.
    SEP_IMPOSSIBLE, H0_IMPOSSIBLE = 0.025, 0.02

    def _passat(x):
        s = math.sqrt(L * L - x * x)
        return SEP_IMPOSSIBLE - x - (H0_IMPOSSIBLE + L - s) * x / s + 1e-4

    x_impossible = brentq(_passat, 0.0, SEP_IMPOSSIBLE * 0.999999)
    files.append((1, SEP_IMPOSSIBLE, H0_IMPOSSIBLE, "+", x_impossible, 1))

    taula = pd.DataFrame(
        [{"tanda": t, "sep_cm": sep * 100, "h0_cm": h0 * 100, "costat": c,
          "mesura": "x", "valor_mm": x * 1000, "repeticio": r, "L_cm": np.nan}
         for t, sep, h0, c, x, r in files]
    )
    punts = punts_de_la_taula(taula, L, None)
    print(f"   {len(taula)} lectures a {len(punts)} distancies, de dues tandes")

    res = analitza(punts, p, sistematiques={}, sortides=sortides, llavor=llavor)
    afegeix_comprovacions(res, punts, p, sortides=400)

    mc = res["ajustos"]["totes"]["monte_carlo"]
    n, K = mc["n"]["p50"], mc["K"]["p50"]
    ref = res["ajustos"]["totes"]["referencia"]
    f_bona = K_BONA / ref["delta_m"] ** N_BONA
    f_troba = ref["F_per_massa"]["p50"]
    print(f"\n   trobat: n = {n:.4f} +- {mc['n']['sigma']:.4f}   "
          f"K = {K:.4e}   ({mc['sortides']} sorteigs)")
    print(f"   a delta = {ref['delta_m']*1000:.2f} mm: F/m = {f_troba:.5f} "
          f"contra {f_bona:.5f} posat")

    # El biaix d'ignorar l'alcada variable. Es torna a mesurar CADA VEGADA, aixi
    # el numero que hi ha escrit al docstring no pot envellir sense que es noti.
    punts_2 = punts_de_la_taula(taula, L, None)
    dolent = analitza(punts_2, p, sortides=500, llavor=llavor, alcada_fixa=True)
    n_dolent = dolent["ajustos"]["totes"]["monte_carlo"]["n"]["p50"]
    print(f"\n   n ignorant que la massa puja: {n_dolent:.3f}  "
          f"({(n_dolent - N_BONA)/N_BONA:+.1%})   <- l'ajust DOLENT")
    print(f"   n amb l'alcada variable:      {n:.3f}  ({(n - N_BONA)/N_BONA:+.1%})")
    print("   Per aixo l'alcada no es opcional: no es una correccio, es el model.")

    # La distancia critica, contra la que dona directament la n i la K posades.
    sep_bo, _ = distancia_critica(N_BONA, K_BONA, 0.02, L, g)
    sep_trobat = res["distancia_critica"]["1"]["per_llei"]["totes"]["sep_critic_m"]
    print(f"\n   distancia critica (h0 = 2 cm): {sep_trobat*100:.3f} cm trobada "
          f"contra {sep_bo*100:.3f} cm posada")

    fora = res["fora_de_la_recta"]
    nomes_l_impossible = (len(fora) == 1 and abs(fora[0]["sep_m"] - SEP_IMPOSSIBLE) < 1e-12
                          and abs(fora[0]["denominador_m"] + 1e-4) < 1e-9)
    print(f"\n   fora de la recta: {len(fora)} punt(s)"
          + "".join(f", sep = {q['sep_m']*100:.1f} cm (a - h*x/s = "
                    f"{q['denominador_m']*1000:+.3f} mm)" for q in fora)
          + "   <- ha de ser nomes el fabricat a 2,5 cm")

    # La sigma de lectura comuna (25/10). CRITERI, ESCRIT ABANS: amb 8 punts de
    # 2 lectures i un soroll conegut de 0,2 mm, la sigma agrupada (8 graus de
    # llibertat) ha de caure a menys d'un 40 % del soroll posat (uns dos cops
    # el que pot variar amb 8 graus); cada punt hi ha de tenir exactament
    # sigma/sqrt(2); i els punts amb 6 lectures de les tandes de sobre s'han de
    # quedar amb la seva.
    rng_c = np.random.default_rng(llavor + 1)
    files_c = [{"tanda": 9, "sep_cm": 4.0 + k, "h0_cm": 3.0, "costat": "+", "mesura": "x",
                "valor_mm": 1.0 + k + rng_c.normal(0.0, SOROLL_BO * 1000), "repeticio": r,
                "L_cm": np.nan}
               for k in range(8) for r in (1, 2)]
    punts_c = punts_de_la_taula(pd.DataFrame(files_c), L, None)
    comuna = punts_c[0].sigma_x * math.sqrt(2)
    iguals = all(abs(q.sigma_x - comuna / math.sqrt(2)) < 1e-15 for q in punts_c)
    propies = all(str(q.sigma_de).startswith("dispersio") for q in punts if len(q.lectures) >= 3)
    print(f"\n   sigma de lectura comuna: {comuna*1000:.3f} mm contra {SOROLL_BO*1000:.3f} posat"
          f" (8 punts de 2 lectures); la mateixa a tots: {'si' if iguals else 'NO'};"
          f" els de 6 lectures, la seva: {'si' if propies else 'NO'}")

    proves = [
        ("_ax = fisica", pitjor / 9.8, 1e-12),
        ("sigma comuna", abs(comuna - SOROLL_BO) / SOROLL_BO if iguals and propies else 1.0, 0.40),
        ("n", abs(n - N_BONA) / N_BONA, 0.03),
        ("F a delta_ref", abs(f_troba - f_bona) / f_bona, 0.03),
        ("distancia critica", abs(sep_trobat - sep_bo) / sep_bo, 0.02),
        ("n de les dues tandes", res["comparacio"]["diferencia_sigmes"] / 10.0, 0.30),
        ("punt impossible, fora", 0.0 if nomes_l_impossible else 1.0, 0.0),
    ]
    print(f"\n   {'que':>22}  {'error':>11}  {'llindar':>8}")
    tot_be = True
    for nom, error, llindar in proves:
        be = error <= llindar
        tot_be &= be
        print(f"   {nom:>22}  {error:>10.4%}  {llindar:>7.1%}   {'OK' if be else 'MALAMENT'}")

    print("\n   " + ("OK -- l'eina recupera el que se li ha posat."
                     if tot_be else
                     "MALAMENT -- l'eina no recupera el que se li ha posat."))
    return 0 if tot_be else 1


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("entrada", nargs="?", help="CSV de mesures de forca")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova l'eina amb una n i una K conegudes")
    ap.add_argument("--tanda", default="totes",
                    help="quina tanda s'ajusta: un numero, o 'totes' (per defecte)")
    ap.add_argument("--sortides", type=int, default=SORTIDES_PER_DEFECTE,
                    help=f"remostreigs de Monte Carlo (per defecte {SORTIDES_PER_DEFECTE})")
    ap.add_argument("--llavor", type=int, default=LLAVOR_PER_DEFECTE,
                    help="llavor aleatoria, perque el resultat es pugui repetir")
    ap.add_argument("--alcada-fixa", action="store_true",
                    help="ajust DOLENT que ignora que la massa puja; nomes per ensenyar-lo")
    ap.add_argument("--nom", default=None, help="nom base de la figura i del JSON")
    ap.add_argument("--figures", default="magnetic/figures", help="on va la figura")
    ap.add_argument("--mostra", action="store_true",
                    help="obre una finestra en comptes de desar")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova(llavor=args.llavor)
    if not args.entrada:
        ap.error("digue'm quin CSV he de llegir, o passa --autoprova")

    p = Parametres.des_de_json()
    print("=" * 74)
    print(" magnetic.llei_de_forca -- la n i la K, amb el pendol de dinamometre")
    print("=" * 74)
    avis = fisica.avis_inventats(p)
    if avis:
        print(avis)

    taula, meta = dades.carrega_mesures_de_forca(args.entrada)
    # Les tandes que el JSON deixa fora de l'ajust, amb el motiu. Les lectures
    # es queden a la taula: treure-les d'alla seria perdre-les.
    fora = meta.get("tandes_fora_de_l_ajust") or {}
    if fora:
        print("\n   " + "!" * 70)
        print("   !! TANDES FORA DE L'AJUST (tandes_fora_de_l_ajust del JSON)")
        print("   " + "!" * 70)
        for t, motiu in fora.items():
            print(f"   tanda {t}: {motiu}")
        taula = taula[~taula["tanda"].astype(str).isin([str(t) for t in fora])]
    if args.tanda != "totes":
        taula = taula[taula["tanda"] == int(args.tanda)]
        if taula.empty:
            raise SystemExit(f"No hi ha cap fila de la tanda {args.tanda} a {args.entrada}.")

    sigma_lectura = meta.get("sigma_lectura_mm")
    sistematiques = {
        "sep": (meta.get("sigma_sep_mm") or 0.0) / 1000.0,
        "h0": (meta.get("sigma_h0_mm") or 0.0) / 1000.0,
        "L": (meta.get("sigma_L_mm") or 0.0) / 1000.0,
    }
    punts = punts_de_la_taula(taula, p.L, None if sigma_lectura is None
                              else sigma_lectura / 1000.0)

    print(f"\n{args.entrada}: {len(taula)} lectures, {len(punts)} distancies, "
          f"tandes {sorted({q.tanda for q in punts})}")
    buides = [k for k in ("sigma_lectura_mm", "sigma_sep_mm", "sigma_h0_mm", "sigma_L_mm")
              if meta.get(k) is None]
    if buides:
        print("   !! SIGMES NO DECLARADES al JSON: " + ", ".join(buides))
        print("      No me les invento. Les sistematiques que falten NO es sortegen,")
        print("      o sigui que les barres d'error de sota son un MINIM, no la")
        print("      incertesa sencera. Omple-les a "
              + str(dades.cami_meta(args.entrada)) + ".")

    res = analitza(punts, p, sistematiques, args.sortides, args.llavor, args.alcada_fixa)
    afegeix_comprovacions(res, punts, p)
    escriu(res, p)

    nom = args.nom or Path(args.entrada).stem
    dibuixa(res, nom, args.figures, args.mostra)

    if not args.mostra:
        publicable = {k: v for k, v in res.items() if not k.startswith("_")}
        publicable["ajustos"] = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in res["ajustos"].items()
        }
        publicable["origen"] = "magnetic.llei_de_forca"
        publicable["entrada"] = str(args.entrada)
        publicable["metadades_de_l_entrada"] = meta
        publicable["parametres_en_us"] = p.com_a_dict()
        publicable["monte_carlo"] = {"sortides": args.sortides, "llavor": args.llavor}
        sortida = Path(args.figures) / f"{nom}.json"
        sortida.parent.mkdir(parents=True, exist_ok=True)
        sortida.write_text(json.dumps(publicable, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        print(f"   numeros: {sortida}")

    print("\n   Aquest programa NO ha tocat magnetic/parametres.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
