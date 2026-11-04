"""Contracte de dades del pendol magnetic.

Independent del de pendol/dades.py, i amb columnes diferents: aquest sistema
es mou pel PLA i no treballa amb angles, o sigui que el format t,theta1,...
d'alla no hi voldria dir res.

Hi ha QUATRE formats, i cap no es variant de cap altre.

TRAJECTORIA d'un sol llancament, per mirar casos concrets:

    t,x,y,vx,vy         segons, metres, metres/segon

GRAELLA DE CONQUES, que es el resultat de debo:

    x0,y0,imant,temps_caiguda

CALIBRATGE: una presa de video del muntatge real, per mesurar-ne la b:

    t,x,y               segons, metres, metres

MESURES DE FORCA: la taula del banc de treball, per mesurar-ne la n i la K:

    tanda,sep_cm,h0_cm,costat,mesura,valor_mm,repeticio

    Aquesta es l'UNICA que s'escriu a ma. No la genera cap programa: es la
    taula del quadern copiada al projecte, i existeix precisament perque els
    numeros mesurats no visquin nomes en un full o en una conversa.

Tots porten al costat un JSON amb el MATEIX NOM i les metadades. Sense
aquest JSON, d'aqui a tres mesos no sabras d'on surt cada fitxer ni amb quins
parametres es va fer, i una figura de la qual no pots dir d'on surt no serveix
per a res en un treball.

Als JSON hi va sempre el camp valors_inventats, que arriba de
Parametres.com_a_dict(). Es el tercer nivell de marcatge dels valors que no son
mesures: aixi la marca viatja ENGANXADA a la dada i no en un paper al costat.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNES_TRAJECTORIA = ["t", "x", "y", "vx", "vy"]


def cami_meta(cami_csv: str | Path) -> Path:
    return Path(cami_csv).with_suffix(".json")


def desa_trajectoria(cami_csv, t, x, y, vx, vy, meta: dict | None = None) -> Path:
    """Escriu una trajectoria en el format canonic + el JSON de metadades."""
    cami_csv = Path(cami_csv)
    cami_csv.parent.mkdir(parents=True, exist_ok=True)

    taula = pd.DataFrame(
        {
            "t": np.asarray(t, dtype=float),
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
            "vx": np.asarray(vx, dtype=float),
            "vy": np.asarray(vy, dtype=float),
        }
    )
    # Punt decimal i prou xifres per no perdre informacio al re-llegir.
    taula.to_csv(cami_csv, index=False, float_format="%.10g")

    if meta is not None:
        cami_meta(cami_csv).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return cami_csv


def carrega_trajectoria(cami_csv):
    """Llegeix una trajectoria canonica. Retorna (taula, metadades)."""
    cami_csv = Path(cami_csv)
    taula = pd.read_csv(cami_csv)

    falten = [c for c in COLUMNES_TRAJECTORIA if c not in taula.columns]
    if falten:
        raise ValueError(
            f"{cami_csv.name} no segueix el contracte de dades del pendol magnetic: "
            f"hi falten {falten}.\nColumnes trobades: {list(taula.columns)}"
        )

    meta_p = cami_meta(cami_csv)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return taula[COLUMNES_TRAJECTORIA].copy(), meta


# ---------------------------------------------------------------------------
# La graella de conques: el resultat de debo
# ---------------------------------------------------------------------------
COLUMNES_GRAELLA = ["x0", "y0", "imant", "temps_caiguda"]


def desa_graella(cami_base, x0, y0, imant, temps, meta: dict) -> Path:
    """Desa una graella en TRES fitxers amb el mateix nom.

        .npz   les matrius. AQUEST ES EL BO.
        .csv   el contracte x0,y0,imant,temps_caiguda, una fila per pixel
        .json  les metadades

    Per que el .npz i no nomes el PNG: d'un PNG comprimit no es pot recuperar
    de manera fiable quin imant hi havia a cada pixel, i sense aixo no es poden
    tornar a calcular ni el percentatge de frontera ni la dimensio fractal
    sense refer hores de mapa. I el dia que es llegeixi la foto de la
    quadricula pintada a ma, caldra tenir contra que comparar.

    imant va en int8 (nomes hi ha valors de -1 a 3) i temps en float32: amb un
    mapa de 600x600 la diferencia es de megabytes.
    """
    cami_base = Path(cami_base).with_suffix("")
    cami_base.parent.mkdir(parents=True, exist_ok=True)

    imant = np.asarray(imant, dtype=np.int8)
    temps = np.asarray(temps, dtype=np.float32)
    np.savez_compressed(
        cami_base.with_suffix(".npz"),
        x0=np.asarray(x0, dtype=float),
        y0=np.asarray(y0, dtype=float),
        imant=imant,
        temps_caiguda=temps,
    )

    xx, yy = np.meshgrid(np.asarray(x0, dtype=float), np.asarray(y0, dtype=float))
    pd.DataFrame(
        {
            "x0": xx.ravel(),
            "y0": yy.ravel(),
            "imant": imant.ravel(),
            "temps_caiguda": temps.ravel(),
        }
    ).to_csv(cami_base.with_suffix(".csv"), index=False, float_format="%.8g")

    cami_base.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return cami_base.with_suffix(".npz")


def carrega_graella(cami):
    """Llegeix una graella del .npz. Retorna (x0, y0, imant, temps, metadades)."""
    cami = Path(cami).with_suffix(".npz")
    d = np.load(cami)
    meta_p = cami.with_suffix(".json")
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return d["x0"], d["y0"], d["imant"], d["temps_caiguda"], meta


# ---------------------------------------------------------------------------
# El calibratge: el que surt d'un video del muntatge real
# ---------------------------------------------------------------------------
# PER QUE UN TERCER CONTRACTE I NO EL DE TRAJECTORIA. Els dos tenen columnes
# que es diuen x i y, i volen dir coses DIFERENTS:
#
#   trajectoria   x, y son el PLA HORITZONTAL dels imants, amb l'origen a la
#                 projeccio del punt de repos. Es el que integra el motor.
#   calibratge    x, y son un PLA VERTICAL vist de costat per la camera, amb
#                 l'origen al PIVOT: x horitzontal i y cap amunt. Es el que
#                 veu el Tracker quan es mesura la friccio amb els imants
#                 retirats, que llavors es un pendol simple.
#
# Posar-hi les mateixes lletres volent dir una altra cosa es exactament com es
# fabriquen els errors que no donen cap error: el fitxer s'obriria, les
# columnes hi serien, i el numero que en sortiria seria d'un altre experiment.
# Per aixo son dues funcions i no una bandera dins de desa_trajectoria().
#
# NO S'HI DESA LA r. El Tracker si que la sap exportar i val la pena demanar-la
# per mirar-la alla mateix, pero aqui es DERIVADA de x i de y, i una columna
# derivada desada es una columna que un dia quedara desaparellada de les seves
# fonts. La calcula magnetic.decrement, que es qui la fa servir.
COLUMNES_CALIBRATGE = ["t", "x", "y"]


def desa_calibratge(cami_csv, t, x, y, meta: dict | None = None) -> Path:
    """Escriu una presa de calibratge + el JSON de metadades.

    Al JSON hi ha d'anar SEMPRE de quin video ve i amb quina configuracio del
    Tracker: escala, origen i angle. Sense aixo, d'aqui a tres mesos una
    columna de metres no es pot tornar a lligar amb res.
    """
    cami_csv = Path(cami_csv)
    cami_csv.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "t": np.asarray(t, dtype=float),
            "x": np.asarray(x, dtype=float),
            "y": np.asarray(y, dtype=float),
        }
    ).to_csv(cami_csv, index=False, float_format="%.10g")

    if meta is not None:
        cami_meta(cami_csv).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return cami_csv


def carrega_calibratge(cami_csv):
    """Llegeix una presa de calibratge. Retorna (taula, metadades)."""
    cami_csv = Path(cami_csv)
    taula = pd.read_csv(cami_csv)

    falten = [c for c in COLUMNES_CALIBRATGE if c not in taula.columns]
    if falten:
        raise ValueError(
            f"{cami_csv.name} no segueix el contracte de calibratge del pendol "
            f"magnetic: hi falten {falten}.\nColumnes trobades: {list(taula.columns)}\n"
            "Si es un export del Tracker sense passar pel netejador, passa-l'hi:\n"
            "  .venv\\Scripts\\python.exe -m magnetic.tracker <cru.csv> --sortida <net.csv>"
        )

    meta_p = cami_meta(cami_csv)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return taula[COLUMNES_CALIBRATGE].copy(), meta


# ---------------------------------------------------------------------------
# Les mesures de forca: la taula del banc, per a la n i la K
# ---------------------------------------------------------------------------
# PER QUE UN QUART CONTRACTE. Els altres tres son sortides de programes; aquest
# es una ENTRADA escrita a ma, i te un problema que els altres no tenen: la
# columna valor_mm no vol dir sempre el mateix. A la tanda 1 l'eix del Tracker
# es va posar a l'imant i el que es va llegir va ser la separacio a; a partir
# de la tanda 2, amb l'eix al punt de repos, el que es llegeix es el desviament
# x. Son dos numeros diferents que caben a la mateixa casella.
#
# Per aixo hi ha la columna `mesura`, i per aixo la conversio x = sep - a la fa
# el codi. Una taula on la meitat de les files ja estiguessin convertides a ma
# tindria l'aspecte de ser homogenia i no ho seria, i aquesta es exactament la
# mena d'error que no dona cap error.
#
# LES ALTRES DUES COLUMNES QUE NO SON DECORACIO:
#   costat     de quin costat era l'imant. Mesurant pels dos costats, l'error
#              de zero de l'eix entra amb signes contraris (les lectures son
#              x+e i x-e) i la mitjana el cancel.la. Amb un sol costat, no.
#   repeticio  les repeticions del mateix punt. La seva dispersio ES la barra
#              d'error del punt; sense repeticions no n'hi ha cap de propia.
#
# L_cm ES OPCIONAL i nomes cal si una tanda es va fer amb un altre fil. Per
# defecte s'agafa la L de parametres.json, que es on viu la del muntatge: no
# es duplica aqui, per la mateixa rao per la qual no hi ha un segon fitxer de
# parametres.
COLUMNES_MESURES_FORCA = ["tanda", "sep_cm", "h0_cm", "costat", "mesura", "valor_mm"]
MESURES_VALIDES = ("a", "x")


def carrega_mesures_de_forca(cami_csv):
    """Llegeix la taula de mesures de forca. Retorna (taula, metadades).

    NO fa cap fisica: nomes comprova que el fitxer digui el que ha de dir. La
    conversio a x, a delta i a forca es de magnetic.llei_de_forca.
    """
    cami_csv = Path(cami_csv)
    taula = pd.read_csv(cami_csv)

    falten = [c for c in COLUMNES_MESURES_FORCA if c not in taula.columns]
    if falten:
        raise ValueError(
            f"{cami_csv.name} no segueix el contracte de mesures de forca del "
            f"pendol magnetic: hi falten {falten}.\n"
            f"Columnes trobades: {list(taula.columns)}\n"
            f"Han de ser: {','.join(COLUMNES_MESURES_FORCA)}[,repeticio][,L_cm]"
        )

    if "repeticio" not in taula.columns:
        taula = taula.assign(repeticio=1)
    if "L_cm" not in taula.columns:
        taula = taula.assign(L_cm=np.nan)

    dolentes = sorted(set(taula["mesura"]) - set(MESURES_VALIDES))
    if dolentes:
        raise ValueError(
            f"{cami_csv.name}: la columna `mesura` nomes pot dir {MESURES_VALIDES} "
            f"-- 'a' si es va mesurar la separacio (eix a l'imant) i 'x' si es va "
            f"mesurar el desviament (eix al punt de repos). Hi he trobat {dolentes}."
        )

    dolents = sorted(set(taula["costat"].astype(str)) - {"+", "-"})
    if dolents:
        raise ValueError(
            f"{cami_csv.name}: la columna `costat` nomes pot dir '+' o '-'. "
            f"Hi he trobat {dolents}."
        )

    if (taula["valor_mm"] <= 0).any():
        raise ValueError(
            f"{cami_csv.name}: valor_mm es sempre una MAGNITUD POSITIVA, tant si "
            "l'imant era al costat + com al -. El signe el porta la columna "
            "`costat`, no el numero."
        )

    meta_p = cami_meta(cami_csv)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return taula, meta
