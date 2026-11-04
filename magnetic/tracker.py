"""Del Tracker al format de calibratge, amb les guardes posades.

Hi ha DOS camins d'entrada, i el bo es el primer:

    .trk   el projecte del Tracker. Les posicions hi son en PIXELS amb tota la
           precisio del double, i al costat hi ha l'origen, l'angle i l'escala
           per convertir-les. No cal exportar res.
    .csv   un export del Tracker, per si algun dia nomes es te aixo. Passa per
           l'endevinador de dialectes i per la guarda de l'arrodoniment.


Aquest modul es nomes per a les preses de CALIBRATGE del pendol magnetic: el
video del muntatge real amb els imants RETIRATS, que llavors es un pendol
simple i del seu decaiment en surt la b. La camera el mira de costat, o sigui
que x es horitzontal i y es vertical, i l'origen va al PIVOT. No te res a
veure amb el pla horitzontal dels imants; vegeu magnetic/dades.py.

QUE HI FA UN import DE pendol/
------------------------------
El CLAUDE.md de magnetic/ diu que aquest sistema no importa res de pendol/, i
que si alguna cosa sembla reaprofitable s'ha de proposar. Es va proposar i es
va acceptar per a UNA funcio: pendol.tracker.llegeix_cru.

El que fa es endevinar el dialecte d'un export del Tracker -- separador
tabulador, punt i coma o coma, decimals amb punt o amb COMA (el Tracker
d'aquesta maquina va en castella), linies de metadades al davant, tres
codificacions possibles, i els fotogrames on el Tracker va perdre el punt, que
han de quedar en NaN i no fer petar la lectura. No es fisica, ni el simulador,
ni el contracte de dades, ni l'estil de figures: es un endevinador de dialectes
de CSV. Duplicar-lo voldria dir dos llocs on arreglar-ho el dia que un export
sorprengui, i el dia que sorprengui sera un de sol dels dos el que fallara.

LES QUATRE GUARDES, QUE SON LA RAO DE SER D'AQUEST FITXER
--------------------------------------------------------
Un pd.read_csv tambe llegiria el fitxer. El que no faria es aturar-se:

  1. ELS FPS. Es l'unic error del circuit que produeix dades d'aspecte
     impecable: si el video declara 240 fps i de debo en fa 30, la corba surt
     igual de maca i tots els temps -- i per tant la b i el periode -- estan
     multiplicats per vuit. Es compara el que diu la columna t amb el que
     esperes i s'atura si difereixen mes d'un 2%.
  2. ELS FORATS. Si l'autotracker va perdre la marca vint fotogrames seguits,
     interpolar-los es dibuixar una recta per sobre d'una oscil.lacio. Els
     forats curts s'omplen; els llargs aturen el programa i s'han de decidir a
     ma (o amb --talla).
  3. ELS FOTOGRAMES DUPLICATS. Una posicio exactament repetida MENTRE EL PENDOL
     ES MOU es la firma d'un fotograma clonat en convertir el video. A prop
     dels extrems, en canvi, repetir-se es normal i no vol dir res.
  4. LA QUANTITZACIO. El Tracker exporta el que la TAULA ensenya. Si la taula
     va a tres decimals, un metre queda partit en mil.limetres i el pas de
     quantitzacio (1 mm) es mes gros que un pixel (0,62 mm): s'ha llencat
     resolucio que ja es tenia. Es detecta i es diu com arreglar-ho.

Us, des de l'ARREL del projecte. El cami bo:

    .venv\\Scripts\\python.exe -m magnetic.tracker dades/video/tracker/IMG_3072.trk ^
        --sortida dades/net/magnetic_3072.csv

I el cami del CSV, que necessita que li diguis on es el projecte:

    .venv\\Scripts\\python.exe -m magnetic.tracker dades/cru/magnetic_3072.csv ^
        --sortida dades/net/magnetic_3072.csv --fps-esperat 239.69 ^
        --trk dades/video/tracker/IMG_3072.trk

El .trk fa falta en tots dos casos: d'alla en surten l'escala, l'origen,
l'angle de correccio i les vares de calibracio, i tot aixo va al JSON. Sense
aixo, d'aqui a tres mesos la columna de metres no es pot tornar a lligar amb
res.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from magnetic import dades
from magnetic.geometria import adreca_amb_la_vertical

# L'unica excepcio a la independencia de pendol/. Vegeu el docstring.
from pendol.tracker import llegeix_cru

# Noms de columna que sap reconeixer, en catala, castella i angles.
NOMS = {
    "t": ("t", "temps", "tiempo", "time"),
    "x": ("x",),
    "y": ("y",),
}
TOLERANCIA_FPS = 0.02  # 2%
FORAT_MAXIM = 3  # fotogrames seguits sense marca que s'accepta interpolar


def fps_lletra(fps: float) -> str:
    return f"{fps:.2f} fps"


# ---------------------------------------------------------------------------
# El projecte del Tracker: la configuracio que el CSV no porta
# ---------------------------------------------------------------------------
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
    zero = fd.find("property[@name='[0]']")
    return _dins(zero)


def llegeix_configuracio_trk(cami: str | Path) -> dict:
    """Llegeix un .trk del Tracker i en treu la configuracio, no les dades.

    El .trk es XML de l'Open Source Physics: cada cosa es un <object> amb
    <property> a dins. Aqui nomes se'n vol el que el CSV exportat NO porta i
    que sense ell la columna de metres no vol dir res: l'escala, on es
    l'origen, l'angle amb que s'ha adrecat la camera i les vares de
    calibracio.

    De propina compta quantes posicions de massa hi ha DESADES al fitxer. Un
    .trk amb la configuracio feta i cap marca vol dir que el seguiment nomes
    es a la memoria del Tracker i encara no ha baixat al disc.
    """
    cami = Path(cami)
    arrel = ET.parse(cami).getroot()
    p = _fills(arrel)
    info: dict = {"fitxer": str(cami)}

    # El rellotge. delta_t va en MIL.LISEGONS.
    control = _dins(p.get("clipcontrol"))
    if control is not None:
        dt_ms = _num(_fills(control), "delta_t")
        if dt_ms:
            info["delta_t_ms"] = dt_ms
            info["fps_declarats"] = 1000.0 / dt_ms

    # El video i el retall.
    clip = _dins(p.get("videoclip"))
    if clip is not None:
        pc = _fills(clip)
        info["fotogrames"] = _num(pc, "video_framecount")
        info["fotograma_inicial"] = _num(pc, "startframe")
        info["pas"] = _num(pc, "stepsize")
        video = _dins(pc.get("video"))
        if video is not None:
            info["video"] = _text(_fills(video), "path")

    # El sistema de coordenades: origen, angle i escala.
    coords = _dins(p.get("coords"))
    if coords is not None:
        marc = _primer_framedata(_fills(coords))
        if marc is not None:
            c = _fills(marc)
            info["origen_px"] = [_num(c, "xorigin"), _num(c, "yorigin")]
            angle = _num(c, "angle", 0.0)
            info["angle_rad"] = angle
            info["angle_graus"] = math.degrees(angle)
            info["escala_px_per_m"] = _num(c, "xscale")
            info["escala_y_px_per_m"] = _num(c, "yscale")

    # Les pistes: vares de calibracio i masses.
    vares, marques = [], 0
    tracks = p.get("tracks")
    if tracks is not None:
        for item in tracks.findall("property[@name='item']"):
            obj = _dins(item)
            if obj is None:
                continue
            classe = obj.get("class", "")
            props = _fills(obj)
            if classe.endswith("TapeMeasure"):
                marc = _primer_framedata(props)
                llargada_px = None
                if marc is not None:
                    m = _fills(marc)
                    llargada_px = math.hypot(
                        _num(m, "x2", 0.0) - _num(m, "x1", 0.0),
                        _num(m, "y2", 0.0) - _num(m, "y1", 0.0),
                    )
                mons = props.get("worldlengths")
                llargada_m = None
                if mons is not None:
                    zero = mons.find("property[@name='[0]']")
                    if zero is not None and zero.text:
                        llargada_m = float(zero.text)
                vara = {
                    "nom": _text(props, "name"),
                    "longitud_m": llargada_m,
                    "longitud_px": llargada_px,
                }
                if llargada_m:
                    vara["px_per_m"] = llargada_px / llargada_m
                vares.append(vara)
            elif classe.endswith("PointMass"):
                fd = props.get("framedata")
                if fd is not None:
                    marques += len(fd.findall("property"))
    info["vares"] = vares
    info["marques_desades"] = marques
    return info


def escriu_configuracio(info: dict) -> None:
    """Ensenya la configuracio per pantalla, amb el que se'n pot comprovar."""
    print(f"   video          {info.get('video', '?')}")
    if "fps_declarats" in info:
        print(f"   fps declarats  {info['fps_declarats']:.3f}   "
              f"({info.get('fotogrames', 0):.0f} fotogrames)")
    if "origen_px" in info:
        ox, oy = info["origen_px"]
        print(f"   origen         ({ox:.2f}, {oy:.2f}) px")
    if "angle_graus" in info:
        print(f"   angle          {info['angle_graus']:+.2f} graus (camera adrecada)")
    if info.get("escala_px_per_m"):
        e = info["escala_px_per_m"]
        print(f"   escala         {e:.3f} px/m   ->  1 px = {1000.0 / e:.3f} mm")

    for v in info.get("vares", []):
        if v.get("px_per_m"):
            print(f"   vara {v['nom']!r}: {v['longitud_m'] * 100:.1f} cm = "
                  f"{v['longitud_px']:.2f} px  ->  {v['px_per_m']:.1f} px/m")
    # Dues vares que coincideixen NO demostren que l'escala sigui bona: el
    # Tracker calcula la longitud de la segona amb l'escala que ja te posada,
    # i per tant coincidiran sempre EXCEPTE si li has escrit el numero mesurat
    # a ma. Si les has mesurades totes dues amb el regle i coincideixen, alla
    # si que vol dir que no hi ha perspectiva; si no, no vol dir res.
    px = [v["px_per_m"] for v in info.get("vares", []) if v.get("px_per_m")]
    if len(px) >= 2:
        dif = (max(px) - min(px)) / min(px)
        if dif < 1e-6:
            print("   !! les dues vares donen la MATEIXA escala fins a l'ultim decimal.")
            print("      Aixo passa sol si la longitud de la segona no la vas escriure")
            print("      tu: el Tracker l'hi posa amb l'escala que ja tenia. Nomes es")
            print("      una comprovacio si vas mesurar les dues amb el regle.")
        else:
            print(f"   les dues vares difereixen un {dif:.2%} en l'escala")

    if info.get("marques_desades", 0) <= 1:
        print(f"   !! el .trk nomes te {info.get('marques_desades', 0)} posicio(ns) de massa")
        print("      desades: el seguiment encara no ha baixat al disc.")


# ---------------------------------------------------------------------------
# Llegir la trajectoria del .trk, sense passar per cap export
# ---------------------------------------------------------------------------
# PER QUE ES MILLOR QUE EXPORTAR. El CSV que escriu el Tracker es el que ENSENYA
# LA TAULA: ja arrodonit al nombre de decimals que hi tinguis posat, amb els
# decimals amb coma si el programa va en castella, i amb el separador que hagis
# triat al dialeg. El .trk, en canvi, desa les posicions en PIXELS amb tota la
# precisio del double, i desa al costat l'origen, l'angle i l'escala per
# convertir-les. Llegint d'aqui no hi ha cap dialeg on equivocar-se i no es
# perd ni un decimal.
#
# El preu es que la conversio de pixels a metres la fem nosaltres, i el conveni
# de l'angle del Tracker no esta documentat enlloc que es pugui citar. Per aixo
# s'aplica amb un signe explicit i DESPRES es mesura: magnetic.decrement calcula
# de quin angle es de debo l'eix de simetria de l'arc, i si el signe fos l'altre
# el numero sortiria el doble de gros i es veuria de seguida.
def llegeix_trajectoria_trk(cami: str | Path, nom_massa=None):
    """Les posicions d'una massa, en PIXELS de la imatge.

    Retorna (numeros de fotograma, x_px, y_px, nom). El framedata del Tracker
    es un vector escampat: els noms de les propietats son [0], [1], [225]... i
    aquell numero ES el fotograma, o sigui que d'aqui en surten alhora les
    posicions i a quin instant van.
    """
    arrel = ET.parse(Path(cami)).getroot()
    tracks = _fills(arrel).get("tracks")
    masses = []
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
                    px, py = _num(m, "x"), _num(m, "y")
                    if px is None or py is None:
                        continue
                    ks.append(int(prop.get("name").strip("[]")))
                    xs.append(px)
                    ys.append(py)
            masses.append((_text(props, "name", "?"), np.array(ks, dtype=int),
                           np.array(xs), np.array(ys)))

    if not masses:
        raise SystemExit(f"A {cami} no hi ha cap massa (PointMass).")
    if nom_massa is not None:
        tria = [m for m in masses if m[0] == nom_massa]
        if not tria:
            raise SystemExit(
                f"No hi ha cap massa que es digui {nom_massa!r}. N'hi ha: "
                + ", ".join(repr(m[0]) for m in masses)
            )
        nom, k, x, y = tria[0]
    else:
        nom, k, x, y = max(masses, key=lambda m: len(m[1]))
        if len(masses) > 1:
            print(f"   hi ha {len(masses)} masses; agafo {nom!r}, que es la que en te mes "
                  f"({len(k)}). Tria'n una altra amb --massa.")

    if len(k) < 10:
        raise SystemExit(
            f"La massa {nom!r} nomes te {len(k)} posicions desades al .trk.\n"
            "El seguiment encara es a la memoria del Tracker: ves-hi i fes\n"
            "Archivo > Guardar abans de tornar a executar aixo."
        )
    ordre = np.argsort(k)
    return k[ordre], x[ordre], y[ordre], nom


def a_coordenades_mon(x_px, y_px, info: dict, signe: int = +1):
    """De pixels de la imatge a metres, amb l'origen al pivot i la y CAP AMUNT.

    Tres passos, i el segon es l'unic discutible:

      1. Restar l'origen. La y de la imatge creix cap AVALL i la del mon cap
         amunt, o sigui que la vertical canvia de signe aqui.
      2. Girar l'angle amb que es va adrecar la camera. El conveni de signe del
         Tracker no esta documentat de manera citable: s'aplica amb `signe` i
         despres magnetic.decrement mesura l'eix de simetria de l'arc, que ha de
         sortir vertical. Si surt inclinat el doble de l'angle, el signe es
         l'altre i es capgira amb --angle-signe.
      3. Dividir per l'escala, que va en pixels per metre.
    """
    ox, oy = info["origen_px"]
    escala = info["escala_px_per_m"]
    ang = signe * info.get("angle_rad", 0.0)
    u = np.asarray(x_px, dtype=float) - ox
    v = -(np.asarray(y_px, dtype=float) - oy)  # y del mon cap amunt
    c, s = math.cos(ang), math.sin(ang)
    return (u * c + v * s) / escala, (-u * s + v * c) / escala


# ---------------------------------------------------------------------------
# Les columnes
# ---------------------------------------------------------------------------
def tria_columnes(matriu: np.ndarray, capcalera: list[str], columnes: str | None):
    """Selecciona (t, x, y). Primer pel nom de la columna, si el Tracker l'ha posat."""
    if columnes:
        idx = [int(i) for i in columnes.split(",")]
        if len(idx) != 3:
            raise ValueError("--columnes vol 3 indexs: t,x,y")
        print(f"   columnes (a ma):  t={idx[0]} x={idx[1]} y={idx[2]}")
        return matriu[:, idx]

    net = [c.strip().strip('"').split("(")[0].strip().lower() for c in capcalera]
    idx = [next((i for i, c in enumerate(net) if c in noms), None) for noms in
           (NOMS["t"], NOMS["x"], NOMS["y"])]

    if None in idx:
        if matriu.shape[1] < 3:
            raise ValueError(
                f"Nomes hi ha {matriu.shape[1]} columnes i en calen tres (t, x, y).\n"
                "Torna a exportar del Tracker marcant x i y (la t sempre hi es)."
            )
        idx = [0, 1, 2]
        print(f"   columnes (per posicio): t=0 x=1 y=2   capcalera: {capcalera}")
        print("   !! no he sabut llegir els noms de les columnes. Comprova que")
        print("      l'ordre sigui t, x, y; si no, passa'l amb --columnes.")
    else:
        print(f"   columnes (pel nom): t={idx[0]} x={idx[1]} y={idx[2]}")
    return matriu[:, idx]


# ---------------------------------------------------------------------------
# Les guardes
# ---------------------------------------------------------------------------
def mesura_fps(t: np.ndarray) -> tuple[float, float]:
    """Els fps de debo, robustos a fotogrames que faltin a l'export.

    NO ES 1/mediana(dt), i aixo no es cap detall. El Tracker escriu la columna
    t arrodonida als decimals que ensenya la taula: a 239,688 fps el pas de
    debo es 4,1721 ms i a la columna hi surten salts de 4 i de 5 ms barrejats.
    La MEDIANA d'aixo es 4 ms, o sigui 250 fps, un 4% de mes -- prou per fer
    saltar la guarda dels fps amb dades perfectament bones.

    La mitjana dels salts que no son forats, en canvi, es exacta: sumar-los
    torna a donar la durada sencera i els arrodoniments es cancel.len pel cami.
    D'aqui surt el primer dt; despres s'assigna a cada fila el seu NUMERO de
    fotograma i s'ajusta t contra aquell numero, cosa que fa servir tota la
    durada i sobreviu a les files que faltin.
    """
    salts = np.diff(t)
    med = float(np.median(salts))
    dt = float(np.mean(salts[salts <= 1.5 * med]))
    for _ in range(2):
        k = np.round((t - t[0]) / dt)
        dt = float(np.polyfit(k, t, 1)[0])
    return 1.0 / dt, dt


def _talls(marca: np.ndarray) -> list[tuple[int, int]]:
    """Els trams seguits de True dins d'un vector booleà, com a (inici, fi)."""
    trams, inici = [], None
    for i, v in enumerate(marca):
        if v and inici is None:
            inici = i
        elif not v and inici is not None:
            trams.append((inici, i))
            inici = None
    if inici is not None:
        trams.append((inici, len(marca)))
    return trams


def pas_de_quantitzacio(v: np.ndarray) -> float:
    """El salt mes petit entre dos valors diferents de la columna.

    Amb dades continues surt un numero minuscul; amb dades arrodonides a la
    taula del Tracker surt exactament el pas de l'arrodoniment.
    """
    u = np.unique(v[np.isfinite(v)])
    return float(np.min(np.diff(u))) if len(u) > 1 else 0.0


def compta_duplicats(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> tuple[int, int]:
    """(duplicats en total, duplicats mentre el pendol es mou).

    El segon numero es el que importa. Als extrems la massa s'atura de debo i
    dues posicions iguals hi son normals; al mig del recorregut, no.
    """
    iguals = (np.diff(x) == 0) & (np.diff(y) == 0)
    if not iguals.any():
        return 0, 0
    with np.errstate(invalid="ignore"):
        v = np.abs(np.gradient(x, t))
    rapid = v[:-1] > 0.1 * np.nanmax(v)
    return int(iguals.sum()), int((iguals & rapid).sum())


# ---------------------------------------------------------------------------
def processa(
    cami_entrada,
    cami_sortida,
    fps_esperat=None,
    columnes=None,
    cami_trk=None,
    forat_maxim=FORAT_MAXIM,
    talla=False,
    signe_angle=+1,
    vertical_de_les_dades=False,
    massa=None,
    meta_extra=None,
):
    print("=" * 72)
    print(" magnetic.tracker -- export del Tracker -> format de calibratge")
    print("=" * 72)

    entrada = Path(cami_entrada)
    del_trk = entrada.suffix.lower() == ".trk"
    if del_trk and cami_trk is None:
        cami_trk = entrada

    info_trk = {}
    if cami_trk:
        print(f"\nConfiguracio de {cami_trk}")
        info_trk = llegeix_configuracio_trk(cami_trk)
        escriu_configuracio(info_trk)
        if fps_esperat is None and "fps_declarats" in info_trk:
            fps_esperat = info_trk["fps_declarats"]
            print(f"   --fps-esperat surt del .trk: {fps_esperat:.3f}")

    if del_trk:
        # --- del projecte del Tracker, sense cap export ---------------------
        print(f"\nLlegint les posicions de {entrada}")
        for clau in ("origen_px", "escala_px_per_m", "delta_t_ms"):
            if not info_trk.get(clau):
                raise SystemExit(
                    f"Al .trk hi falta {clau}: no s'hi ha acabat de configurar el\n"
                    "sistema de coordenades. Cal l'origen, l'escala i la frequencia."
                )
        k, x_px, y_px, nom = llegeix_trajectoria_trk(entrada, massa)
        capcalera = [f"del .trk, massa {nom!r}", "x_px", "y_px"]
        print(f"   massa {nom!r}: {len(k)} posicions, fotogrames {k[0]} a {k[-1]}")
        print(f"   angle aplicat amb signe {signe_angle:+d}  "
              f"({signe_angle * info_trk['angle_graus']:+.3f} graus)")
        x, y = a_coordenades_mon(x_px, y_px, info_trk, signe_angle)
        dt = info_trk["delta_t_ms"] / 1000.0
        # El pas del clip. Seguir un video de 37.000 fotogrames sencer omple la
        # memoria del Tracker; posant-hi un pas de 2 o de 4 se'n marca la
        # meitat o la quarta part i n'hi ha igualment de sobres. Aqui es
        # dedueix de les marques mateixes i no de la propietat stepsize del
        # .trk, perque el que mana es a quins fotogrames hi ha marca de debo.
        k0 = int(k[0])
        # El pas surt del MAXIM COMU DIVISOR de les distancies, no de la
        # mediana. Amb la mediana, un fitxer on una part es va seguir de dos en
        # dos i una altra d'un en un donaria pas 2, i les marques senars
        # cauirien totes a sobre de les parelles: es perdrien sense dir res.
        # El divisor comu no pot ajuntar mai dues marques.
        pas = int(np.gcd.reduce(np.diff(k))) if len(k) > 1 else 1
        pas = max(pas, 1)
        if pas > 1:
            print(f"   les marques van de {pas} en {pas} fotogrames: "
                  f"{fps_lletra(1.0 / (dt * pas))} efectius")
        dt = dt * pas
        k = np.round((k - k0) / pas).astype(int)
        fps = 1.0 / dt

        # La guarda dels fps aqui NO es pot passar de debo, i val mes dir-ho
        # que fer veure que si: en aquest cami el temps no ve de cap columna
        # mesurada, ve del delta_t que DECLARA el mateix .trk. Comparar-lo amb
        # --fps-esperat nomes comprova que el Tracker i tu penseu el mateix, no
        # que cap dels dos tingui rao. Qui ho comprova de debo es la conversio
        # del video, que ja es va verificar fotograma a fotograma.
        print(f"\nEl rellotge")
        print(f"   fps declarats al .trk: {fps:.3f}   (dt = {dt * 1000:.4f} ms)")
        if fps_esperat and abs(fps - fps_esperat) / fps_esperat > TOLERANCIA_FPS:
            raise SystemExit(
                f"\nATURAT: el .trk declara {fps:.2f} fps i tu n'esperaves "
                f"{fps_esperat:.2f}.\nMira el Clip Settings del Tracker: alla hi ha "
                "d'anar la frequencia que\ndeclara el fitxer de video, no un 240 rodo."
            )
        print("   !! per aquest cami el temps NO es una mesura: surt del delta_t que")
        print("      declara el .trk. Qui garanteix que sigui bo es la conversio del")
        print("      video, que es va verificar fotograma a fotograma.")
    else:
        # --- d'un CSV exportat ----------------------------------------------
        print(f"\nLlegint {entrada}")
        matriu, capcalera = llegeix_cru(entrada)
        t, x, y = tria_columnes(matriu, capcalera, columnes).T

        bons = np.isfinite(t)
        if bons.sum() < 10:
            raise ValueError("Menys de deu files amb temps valid: aixo no es un export de dades.")
        t, x, y = t[bons], x[bons], y[bons]
        t = t - t[0]

        # --- guarda 1: els fps ---------------------------------------------
        fps, dt = mesura_fps(t)
        print(f"\nEl rellotge")
        print(f"   fps mesurats a la columna t: {fps:.3f}   (dt = {dt * 1000:.4f} ms)")
        if fps_esperat:
            desviacio = abs(fps - fps_esperat) / fps_esperat
            print(f"   fps esperats:                {fps_esperat:.3f}   "
                  f"-> difereixen un {desviacio:.3%}")
            if desviacio > TOLERANCIA_FPS:
                raise SystemExit(
                    f"\nATURAT: els fps no quadren ({fps:.2f} mesurats contra "
                    f"{fps_esperat:.2f} esperats, {desviacio:.1%}).\n"
                    "Aquest es l'unic error del circuit que produeix dades d'aspecte\n"
                    "impecable: la corba sortiria igual de maca i la b i el periode\n"
                    "estarien tots escalats. Mira, per aquest ordre:\n"
                    "  - que el video convertit no dupliqui fotogrames (-r amb la\n"
                    "    frequencia que DECLARA el fitxer, mai un 240 automatic);\n"
                    "  - que al Clip Settings del Tracker hi hagi la frequencia bona;\n"
                    "  - que no hagis exportat amb un pas de fotograma diferent d'1.\n"
                    "Si estas segur que els fps mesurats son els bons, torna-ho a\n"
                    "executar amb --fps-esperat i el numero de debo."
                )
        else:
            print("   (sense --fps-esperat ni --trk: la guarda dels fps NO s'ha passat)")
        k = np.round(t / dt).astype(int)
        k0, nom = 0, None

    # --- graella uniforme de fotogrames ------------------------------------
    # Es posa tot sobre la reixa de fotogrames abans de mirar els forats: aixi
    # una fila que falta a l'export, una casella buida i un fotograma que
    # l'autotracker no va marcar son el mateix problema i es compten una sola
    # vegada.
    n = int(k[-1]) + 1
    T = np.arange(n) * dt
    X = np.full(n, np.nan)
    Y = np.full(n, np.nan)
    if len(np.unique(k)) != len(k):
        raise SystemExit(
            "Dues marques han caigut al mateix fotograma de la reixa. Aixo "
            "vol dir que el pas del clip s'ha calculat malament i s'estaria "
            "perdent feina sense dir-ho. No segueixo."
        )
    X[k] = x
    Y[k] = y
    buit = ~(np.isfinite(X) & np.isfinite(Y))
    print(f"   {len(k)} posicions -> {n} fotogrames de reixa, {int(buit.sum())} sense marca")

    # --- guarda 2: els forats ----------------------------------------------
    # Primer s'omplen els CURTS i despres es mira si en queda cap de llarg.
    # L'ordre importa: fent-ho al reves, --talla partiria la presa a cada
    # fotograma perdut encara que fos un de sol, i et quedaries amb un tros
    # qualsevol en comptes d'amb la presa sencera.
    curts = [(i, f) for i, f in _talls(buit) if f - i <= forat_maxim]
    if curts:
        n_curts = sum(f - i for i, f in curts)
        print(f"   {n_curts} fotogrames en {len(curts)} forats de fins a "
              f"{forat_maxim}: s'interpolen")
        for i, f in curts:
            buit[i:f] = False
    trams = _talls(buit)
    if trams:
        llarg = max(f - i for i, f in trams)
        print(f"\nEls forats")
        print(f"   {len(trams)} trams sense marca, el mes llarg de {llarg} fotogrames")
        for i, f in sorted(trams, key=lambda u: u[1] - u[0], reverse=True)[:5]:
            print(f"      fotogrames {i}-{f - 1}  ({f - i})  a t = {T[i]:.3f} s")
        if llarg > forat_maxim and not talla:
            raise SystemExit(
                f"\nATURAT: hi ha un forat de {llarg} fotogrames i el maxim acceptat\n"
                f"es {forat_maxim}. Interpolar-lo seria dibuixar una recta per sobre\n"
                "d'una oscil.lacio, i despres ningu no sabria que aquell tros es\n"
                "inventat. Tries tu:\n"
                "  - tornar al Tracker i marcar aquell tram (a ma si cal: quan la\n"
                "    marca passa per una zona fosca no hi ha res a fer);\n"
                "  - o executar amb --talla, que es queda nomes amb el tram seguit\n"
                "    mes llarg i diu quant ha llencat;\n"
                "  - o pujar --forat-maxim si saps que aquell forat es curt de debo."
            )

    if talla:
        plens = _talls(~buit)
        i, f = max(plens, key=lambda u: u[1] - u[0])
        llencat = n - (f - i)
        print(f"   --talla: em quedo amb els fotogrames {i}-{f - 1} "
              f"({f - i} de {n}, en llenco {llencat} = {llencat / n:.1%})")
        T, X, Y, buit = T[i:f] - T[i], X[i:f], Y[i:f], buit[i:f]
        n = f - i
        k0 += i
        trams = _talls(buit)

    per_omplir = ~(np.isfinite(X) & np.isfinite(Y))
    if per_omplir.any():
        print(f"   {int(per_omplir.sum())} fotogrames interpolats linealment")
        te = ~per_omplir
        X[per_omplir] = np.interp(T[per_omplir], T[te], X[te])
        Y[per_omplir] = np.interp(T[per_omplir], T[te], Y[te])
    buit = per_omplir

    # --- adrecar la vertical amb el pendol mateix --------------------------
    gir = 0.0
    if vertical_de_les_dades:
        X, Y, gir = adreca_amb_la_vertical(X, Y)
        print(f"""
La vertical""")
        print(f"   l'arc estava inclinat {gir:+.3f} graus i l'he adrecat")
        if info_trk.get("angle_graus") is not None:
            total = signe_angle * info_trk["angle_graus"] - gir
            print(f"   el .trk ja hi aplicava {signe_angle * info_trk['angle_graus']:+.3f} graus, "
                  f"o sigui que la camera anava tombada {total:+.3f}")
        print("   (un pendol lliure oscil.la simetric respecte de la vertical: es")
        print("    la millor referencia del quadre, millor que cap paret)")

    # --- guarda 3: fotogrames duplicats ------------------------------------
    total, movent = compta_duplicats(X, Y, T)
    print(f"\nFotogrames duplicats")
    print(f"   posicions repetides: {total} en total, {movent} amb el pendol movent-se")
    if movent > 0:
        print("   !! una posicio repetida MENTRE la massa es mou es la firma d'un")
        print("      fotograma clonat en convertir el video. Comprova la conversio:")
        print("      amb -fps_mode cfr, un -r mes alt que la frequencia real en duplica.")

    # --- guarda 4: quantitzacio --------------------------------------------
    pas = max(pas_de_quantitzacio(X), pas_de_quantitzacio(Y))
    escala = info_trk.get("escala_px_per_m")
    print(f"\nResolucio")
    print(f"   salt mes petit entre dos valors: {pas * 1000:.4f} mm")
    if escala:
        px_mm = 1000.0 / escala
        print(f"   un pixel:                        {px_mm:.4f} mm")
        if pas > 0.5 * px_mm / 1000.0:
            if del_trk:
                print("   !! les posicions del .trk ja venen arrodonides, cosa que no")
                print("      hauria de passar: alla hi van en doubles. Mira-t'ho.")
            else:
                print("   !! l'export esta ARRODONIT per sota de la resolucio del video.")
                print("      El Tracker exporta el que la TAULA ensenya: clic dret a la")
                print("      taula > Formato de numero, puja-ho a cinc o sis decimals i")
                print("      torna a exportar. No es cap detall: aixo es precisio que el")
                print("      video ja tenia i que s'esta llencant.")
                print("      O millor: passa-li el .trk directament a aquest programa i")
                print("      no cal cap export.")

    # --- desa ---------------------------------------------------------------
    meta = {
        "origen": "magnetic.tracker",
        "fitxer_cru": str(cami_entrada),
        "font": "projecte .trk (posicions en pixels)" if del_trk else "CSV exportat",
        "capcalera_original": capcalera,
        "massa": nom,
        "primer_fotograma_del_video": int(k0),
        "signe_angle": int(signe_angle) if del_trk else None,
        "gir_per_adrecar_la_vertical_graus": float(gir),
        "pla": "vertical",
        "origen_coordenades": "pivot",
        "que_es": (
            "Presa de calibratge del pendol magnetic amb els imants RETIRATS: "
            "un pendol simple vist de costat. x horitzontal, y vertical, origen "
            "al pivot. NO son coordenades del pla dels imants."
        ),
        "fps_mesurats": float(fps),
        "fps_esperats": float(fps_esperat) if fps_esperat else None,
        "durada_s": float(T[-1]),
        "n_mostres": int(n),
        "fotogrames_interpolats": int(buit.sum()),
        "forats": [{"inici": int(i), "fi": int(f - 1), "fotogrames": int(f - i)}
                   for i, f in trams],
        "retallat": bool(talla),
        "duplicats": {"total": total, "movent_se": movent},
        "pas_de_quantitzacio_m": pas,
        "tracker": info_trk,
        **(meta_extra or {}),
    }
    dades.desa_calibratge(cami_sortida, T, X, Y, meta)

    r = np.hypot(X, Y)
    print(f"\nDesat: {cami_sortida}")
    print(f"   {n} mostres, {T[-1]:.2f} s a {fps:.2f} fps")
    print(f"   x va de {X.min() * 100:+.2f} a {X.max() * 100:+.2f} cm")
    print(f"   r = {r.mean() * 100:.2f} +- {r.std() * 100:.2f} cm  "
          f"(ha de ser plana: es la distancia al pivot)")
    print("\n   Seguent pas:")
    print(f"   .venv\\Scripts\\python.exe -m magnetic.decrement {cami_sortida}")
    return cami_sortida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("entrada", help="projecte .trk del Tracker (recomanat) o CSV exportat")
    ap.add_argument("--sortida", required=True, help="CSV de calibratge (format canonic)")
    ap.add_argument("--fps-esperat", type=float, default=None,
                    help="els fps que ha de tenir el video; si hi ha .trk, en surt sol")
    ap.add_argument("--trk", default=None,
                    help="projecte del Tracker per a la configuracio, si l'entrada es un CSV")
    ap.add_argument("--massa", default=None,
                    help="quina massa del .trk (per defecte, la que te mes posicions)")
    ap.add_argument("--angle-signe", choices=("+", "-"), default="+",
                    help="signe amb que s'aplica l'angle de la camera; magnetic.decrement "
                         "mesura si l'has encertat")
    ap.add_argument("--vertical-de-les-dades", action="store_true",
                    help="gira les coordenades fins que l'arc del pendol quedi simetric "
                         "respecte de la vertical, que es el que diu la gravetat")
    ap.add_argument("--columnes", help="indexs de t,x,y si l'auto-deteccio falla, p.ex. 0,1,2")
    ap.add_argument("--forat-maxim", type=int, default=FORAT_MAXIM,
                    help=f"fotogrames seguits sense marca que s'accepta interpolar (per defecte {FORAT_MAXIM})")
    ap.add_argument("--talla", action="store_true",
                    help="queda't nomes amb el tram seguit mes llarg en comptes d'aturar-te")
    ap.add_argument("--video", default=None, help="de quin video ve, per a metadades")
    args = ap.parse_args(argv)

    processa(
        args.entrada,
        args.sortida,
        fps_esperat=args.fps_esperat,
        columnes=args.columnes,
        cami_trk=args.trk,
        forat_maxim=args.forat_maxim,
        talla=args.talla,
        signe_angle=+1 if args.angle_signe == "+" else -1,
        vertical_de_les_dades=args.vertical_de_les_dades,
        massa=args.massa,
        meta_extra={"video": args.video} if args.video else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
