"""Llegeix un projecte .trk del Tracker: les tres pistes del pendol doble.

El .trk es XML de l'Open Source Physics. Cada cosa es un <object> amb
<property> a dins, i les posicions van a un "framedata" que es un vector
ESCAMPAT: els noms de les propietats son [0], [1], [225]... i aquell numero ES
el numero de fotograma. D'alli en surten alhora les posicions i l'instant.

AIXO ES UNA COPIA de la lectura d'XML de magnetic/tracker.py, i es a posta.
El CLAUDE.md diu que el pendol doble i el magnetic no comparteixen codi: cada
sistema s'ha de defensar sol, i tocar el magnetic no ha de poder trencar el
doble. Son unes seixanta linies que gairebe no canviaran mai. Si algun dia es
decideix unificar-ho, s'ha de canviar la regla del CLAUDE.md primer.

Que NO cal d'aqui: l'escala. Un angle no te unitats, o sigui que tot el
problema de les vares de calibracio del Tracker no toca res del pendol doble.
Nomes calen les posicions en pixels i el rellotge.

Us:
    from pendol.trk import llegeix_trk
    d = llegeix_trk("dades/video/tracker/90/90_1.trk")
    d["pistes"]["fix"]        -> (fotogrames, x_px, y_px)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Com es diuen les tres marques als .trk d'aquest projecte. Es compara en
# minuscules perque als del 150 van amb majuscula inicial i als del 90 no.
NOMS_ESPERATS = ("fix", "rodament", "extrem")


def _fills(node) -> dict:
    """Els <property> fills directes d'un node, indexats pel nom."""
    return {p.get("name"): p for p in node.findall("property")}


def _dins(prop):
    """L'<object> que hi ha dins d'un <property type='object'>."""
    return None if prop is None else prop.find("object")


def _num(props: dict, nom: str, per_defecte=None):
    p = props.get(nom)
    if p is None or p.text is None:
        return per_defecte
    return float(p.text)


def _text(props: dict, nom: str, per_defecte=None):
    p = props.get(nom)
    if p is None or p.text is None:
        return per_defecte
    return p.text


def _primer_framedata(props: dict):
    """El framedata[0] d'un objecte del Tracker, ja obert."""
    fd = props.get("framedata")
    if fd is None:
        return None
    return _dins(fd.find("property[@name='[0]']"))


def llegeix_trk(cami: str | Path) -> dict:
    """Tot el que cal d'un .trk: pistes, rellotge i sistema de coordenades.

    Torna un diccionari amb:
      pistes         {nom: (fotogrames, x_px, y_px)}, ordenades per fotograma
      delta_t_s      el pas de temps del Tracker, en segons
      fps            1/delta_t_s
      origen_px      on es l'origen que vas clicar (per informar, no es fa servir)
      angle_graus    l'angle amb que es va adrecar l'eix al Tracker
      video          la ruta del video que porta el .trk a dins
    """
    cami = Path(cami)
    arrel = ET.parse(cami).getroot()
    p = _fills(arrel)
    d: dict = {"fitxer": str(cami)}

    # El rellotge. delta_t va en MIL.LISEGONS.
    control = _dins(p.get("clipcontrol"))
    dt_ms = _num(_fills(control), "delta_t") if control is not None else None
    if not dt_ms:
        raise SystemExit(
            f"{cami} no diu el pas de temps (clipcontrol/delta_t).\n"
            "Obre'l al Tracker, comprova els Ajustos del clip i torna a desar."
        )
    d["delta_t_s"] = dt_ms / 1000.0
    d["fps"] = 1000.0 / dt_ms

    clip = _dins(p.get("videoclip"))
    if clip is not None:
        pc = _fills(clip)
        d["fotogrames"] = _num(pc, "video_framecount")
        d["pas"] = _num(pc, "stepsize")
        video = _dins(pc.get("video"))
        if video is not None:
            d["video"] = _text(_fills(video), "path")

    coords = _dins(p.get("coords"))
    if coords is not None:
        marc = _primer_framedata(_fills(coords))
        if marc is not None:
            c = _fills(marc)
            d["origen_px"] = [_num(c, "xorigin"), _num(c, "yorigin")]
            d["angle_graus"] = math.degrees(_num(c, "angle", 0.0))

    # Les pistes.
    pistes: dict[str, tuple] = {}
    tracks = p.get("tracks")
    if tracks is not None:
        for item in tracks.findall("property[@name='item']"):
            obj = _dins(item)
            if obj is None or not obj.get("class", "").endswith("PointMass"):
                continue
            props = _fills(obj)
            fd = props.get("framedata")
            ks, xs, ys = [], [], []
            if fd is not None:
                for prop in fd.findall("property"):
                    marc = _dins(prop)
                    if marc is None:
                        continue
                    m = _fills(marc)
                    x, y = _num(m, "x"), _num(m, "y")
                    if x is None or y is None:
                        continue
                    ks.append(int(prop.get("name").strip("[]")))
                    xs.append(x)
                    ys.append(y)
            if len(ks) < 10:
                continue
            ordre = np.argsort(ks)
            pistes[_text(props, "name", "?")] = (
                np.array(ks)[ordre], np.array(xs)[ordre], np.array(ys)[ordre]
            )
    d["pistes"] = pistes
    return d


def retalla(d: dict, des_del_fotograma: int | None) -> dict:
    """El mateix .trk llegit, sense els fotogrames anteriors a des_del_fotograma.

    Per a les preses on el principi del marcatge no es repos de debo: a la
    45_3 i la 45_6 del 25/09/2026 la barra de baix es mou 0,4-0,6 graus durant
    el repos i amollada.py ho pren per l'amollada. Vegeu dades/retalls.json,
    que es on es decideix, presa a presa i amb el motiu.

    Es retallen totes les pistes MENYS EL FIX: del fix nomes se'n fa la mediana
    i la cua (vegeu tres_pistes), i els seus fotogrames del principi no toquen
    res. No modifica d: en torna una copia. Amb None, torna d tal qual.
    """
    if des_del_fotograma is None:
        return d
    pistes = {}
    for nom, (k, x, y) in d["pistes"].items():
        if "fix" in nom.lower():
            pistes[nom] = (k, x, y)
            continue
        sel = k >= des_del_fotograma
        pistes[nom] = (k[sel], x[sel], y[sel])
    return {**d, "pistes": pistes}


def tres_pistes(d: dict):
    """Les tres marques del pendol doble, i la base de temps que en surt.

    Torna (fotogrames, articulacio, extrem, noms, fix), on l'articulacio i
    l'extrem son matrius de dues columnes (x, y) en pixels sobre aquells
    fotogrames, i fix es la parella (fotogrames_del_fix, posicions_del_fix) amb
    ELS SEUS PROPIS fotogrames, que no tenen per que ser els mateixos.

    LA BASE DE TEMPS SURT DE RODAMENT I EXTREM, NO DE LES TRES. Abans es feia
    la interseccio de les tres pistes, i aixo llencava presa per una marca que
    despres no es fa servir fotograma a fotograma: del fix nomes se'n treu la
    MEDIANA com a origen del pivot, la cua per a la vertical i el diagnostic de
    tremolor (vegeu tracker.llegeix_del_trk). Els seus fotogrames del principi
    no els mira ningu.

    I costava car. A les preses caotiques del 30/08/2026 la ma que aguantava el
    suport tapava el gomet del pivot una estona despres de comencar a marcar, o
    sigui que el fix arrencava tard: 496 fotogrames al 150_4, 169 al 150_10, 127
    al 150_6... Amb la interseccio, aquelles preses perdien l'amollada i
    quedaven com si el marcatge hagues comencat amb el pendol ja en marxa. Eren
    vuit, i eren exactament les vuit que el balanc d'energia deixava fora.

    Pel final passa el mateix: al 90_11 el fix s'acaba al fotograma 10515 i les
    altres dues arriben a 17104, o sigui que la interseccio en llencava 27
    segons -justament els del final, on el pendol s'atura- i despres la presa es
    rebutjava per no aturar-se.

    Qui es qui es decideix pel NOM, no per la posicio: fiar-se de quina marca
    esta mes amunt funciona al repos i falla si el pendol acaba doblegat cap
    enrere. Els noms es comparen en minuscules.
    """
    pistes = d["pistes"]
    if len(pistes) < 3:
        raise SystemExit(
            f"{d['fitxer']} nomes te {len(pistes)} pistes amb marques desades "
            f"({', '.join(pistes) or 'cap'}).\n"
            "El pendol doble en vol tres: el pivot fix, l'articulacio i l'extrem."
        )

    per_nom = {n.lower(): n for n in pistes}
    tria = []
    for esperat in NOMS_ESPERATS:
        coincideixen = [k for k in per_nom if esperat in k]
        if len(coincideixen) != 1:
            raise SystemExit(
                f"A {d['fitxer']} no se quina pista es {esperat!r}. N'hi ha: "
                + ", ".join(repr(n) for n in pistes)
                + f".\nEls noms han de contenir {', '.join(NOMS_ESPERATS)}."
            )
        tria.append(per_nom[coincideixen[0]])

    nom_fix, nom_rod, nom_ext = tria

    # Nomes les dues que es fan servir a cada instant manen la base de temps.
    comuns = None
    for nom in (nom_rod, nom_ext):
        k = pistes[nom][0]
        comuns = set(k) if comuns is None else comuns & set(k)
    k_com = np.array(sorted(comuns))
    if len(k_com) < 100:
        raise SystemExit(
            f"A {d['fitxer']} nomes hi ha {len(k_com)} fotogrames amb "
            f"{nom_rod!r} i {nom_ext!r} alhora. Falta seguiment."
        )

    fora = []
    for nom in (nom_rod, nom_ext):
        k, x, y = pistes[nom]
        sel = np.isin(k, k_com)
        fora.append(np.column_stack([x[sel], y[sel]]))

    k_fix, x_fix, y_fix = pistes[nom_fix]
    if len(k_fix) < 10:
        raise SystemExit(
            f"A {d['fitxer']} la pista {nom_fix!r} nomes te {len(k_fix)} "
            "fotogrames. D'aqui no en surt ni l'origen del pivot."
        )
    fix = (np.asarray(k_fix), np.column_stack([x_fix, y_fix]))

    return (k_com, *fora, tria, fix)
