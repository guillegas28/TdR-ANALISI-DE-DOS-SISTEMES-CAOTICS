"""Incrusta-ho tot i escriu la pagina: mapa-interactiu.html.

QUE FA
------
Agafa pagina.html, pagina.css, pagina.js i motor.js, hi posa les constants que
surten de magnetic/parametres.json i de magnetic/estil.py, i ho escriu tot dins
d'UN SOL FITXER que s'obre amb doble clic. Sense servidor, sense connexio i
sense res de fora.

PER QUE UN SOL FITXER I NO QUATRE
---------------------------------
Obrint un fitxer local, el navegador tracta cada fitxer com si vingues d'un
origen desconegut i no deixa que l'un en carregui un altre. Amb els fitxers
separats, la pagina no arrencaria. Incrustar-ho tot ho resol i, de propina, fa
que el que ensenyes sigui una cosa sola que es pot copiar en un llapis.

El fitxer que en surt es REGENERABLE amb una ordre, i per tant va al
.gitignore. Les fonts, al repositori.

L'AUTOTEST QUE S'HI INCRUSTA
----------------------------
compara.py demostra que el motor de JavaScript diu el mateix que el Python EN
AQUESTA MAQUINA. La pagina, en canvi, l'obrira algu altre, amb un altre
navegador i un altre pow. Per aixo s'hi incrusta una taula de punts de
referencia calculats amb el Python, i la pagina els recalcula en obrir-se.

Aquests punts es trien amb dues condicions, i cap de les dues es cosmetica:

  1. Han de ser de DINS d'una conca: el desti no els pot canviar movent-los una
     desena de mil.limetre. A la frontera, demanar coincidencia exacta seria
     demanar una cosa que ni tan sols el Python compleix amb ell mateix quan es
     canvia el dt.
  2. Han de passar per node ABANS d'entrar a la pagina, amb els punts de repos
     que calcula el JavaScript pel seu compte -- que es el que fara la pagina.
     Un autotest que no s'ha vist passar mai no val res.

La segona condicio vol dir que per construir la pagina cal node. No es una
molestia, es la disciplina que el CLAUDE.md ja demana: la copia s'ha de guanyar
el dret a fer-se servir. Amb --sense-node es pot saltar, i llavors ho diu.

    .venv\\Scripts\\python.exe -m magnetic.magnetic_interactiu.construeix
    ii magnetic\\magnetic_interactiu\\mapa-interactiu.html
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
import zlib
from datetime import date
from pathlib import Path

import numpy as np

from magnetic import estil
from magnetic.fisica import (
    Parametres,
    afegeix_banderes_de_parametres,
    avis_inventats,
    sobreescriu,
)
from magnetic.magnetic_interactiu.compara import (
    _parametres_json, busca_node, crida_node,
)
from magnetic.motor import (
    DT_PER_DEFECTE, Criteri, desenllac, destins, graella_nucli,
)

ARREL = Path(__file__).resolve().parent

# La finestra amb que s'obre la pagina: la mateixa del mapa general.
FINESTRA_INICIAL_CM = (-12.0, 12.0, -12.0, 12.0)

# ELS NIVELLS HAN DE SER ELS MATEIXOS QUE A pagina.js. Si algun dia se'n canvia
# un, s'ha de canviar als dos llocs: aqui es fa el mapa incrustat i alla es
# llegeix, i si no encaixen la pagina obriria amb el mapa mal col.locat.
FI = 600
NIVELLS = ((76, 8), (151, 4), (301, 2), (601, 1))

# Les vistes van incrustades al nivell MES FI, i la pagina obre amb la general:
# es la que es mira mes i la que es veu millor. Calcular-les aqui es UNA vegada,
# quan es construeix la pagina, i despres son instantanies per sempre i per a
# tothom.
NIVELL_INCRUSTAT = 3                      # 601 x 601

# Amb quina resolucio arrenca la pagina: 301, decidit per en Guillem el 27/10.
# Amb la llei mesurada, una finestra de la frontera triga molt mes que amb la
# inventada, i a 601 al navegador podia ser mitja hora amb el processador al
# 80 %. Les vistes incrustades s'ensenyen igualment a 601: ja estan fetes.
OBJECTIU_PER_DEFECTE = 2

# LES VISTES QUE LA PAGINA PORTA JA CALCULADES (28/10). Abans nomes hi anava la
# general; ara tambe les tres ampliacions de la figura d'autosemblanca, perque
# recalcular-les al navegador a 601 era inviable i son les que s'ensenyen.
# La finestra de les ampliacions es LLEGEIX del JSON del zoom, no s'escriu aqui:
# aixi el boto x16 de la pagina i el panell x16 de la figura son el mateix lloc.
VISTES_INCRUSTADES = (
    # (nom, etiqueta del boto, graella d'on surt la finestra; None = --finestra)
    ("general", "General", None),
    ("zoom1", "×4", "zoom1"),
    ("zoom2", "×16", "zoom2"),
    ("zoom3", "×64", "zoom3"),
)

# On es desen les vistes calculades, per no tornar-les a calcular si no ha
# canviat res. Van a magnetic/dades, que es al .gitignore: es refan amb aquesta
# mateixa ordre.
CAU_VISTES = Path("magnetic/dades")


# ---------------------------------------------------------------------------
# Les vistes que la pagina porta incrustades
# ---------------------------------------------------------------------------
def calcula_reticula(p: Parametres, criteri: Criteri, dt: float, finestra_m, nivell: int):
    """Una vista, calculada AMB EL PYTHON per incrustar-la a la pagina.

    PER QUE. Sense aixo la pagina s'obre en gris i es passa deu segons calculant
    abans d'ensenyar res, que es la primera impressio que fa i la pitjor. Amb
    aixo obre amb el mapa posat i nomes calcula quan toques alguna cosa.

    NO ES CAP DRECERA NI CAP IMATGE ESTIRADA: son els mateixos desenllacos que
    la pagina calcularia, punt per punt, fets pel nucli que mana. De fet es una
    garantia de mes, perque el que veus en obrir es literalment el mapa validat.

    Les coordenades es fabriquen com les fabrica pagina.js -- una sola reticula
    de 601 punts recorreguda a salts -- i NO amb np.linspace, que donaria uns
    altres ultims bits i faria que el mapa incrustat i el que la pagina calcula
    a sobre no fossin del mateix lloc. Per aixo les vistes de la pagina no son
    els mapes de la memoria (600 punts amb linspace) i es calculen a part.

    Torna els codis i el temps de caiguda COMPTAT EN PASSOS (vegeu codifica()).
    """
    punts, salt = NIVELLS[nivell]
    x_min, x_max, y_min, y_max = finestra_m
    pas = (x_max - x_min) / FI
    ks = range(0, FI + 1, salt)
    x0 = np.array([x_min + k * pas for k in ks])
    y0 = np.array([y_min + k * pas for k in ks])
    assert len(x0) == punts, f"la reticula no encaixa: {len(x0)} punts i n'esperava {punts}"

    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    imant = np.empty((punts, punts), dtype=np.int64)
    temps = np.empty((punts, punts), dtype=np.float64)
    graella_nucli(x0, y0, p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K,
                  dest_x, dest_y, dest_codi, dest_radi2, dt,
                  int(round(criteri.t_max / dt)), int(round(criteri.t_confirma / dt)),
                  int(round(criteri.t_confirma_centre / dt)),
                  imant, temps)
    return imant, np.rint(temps / dt).astype(np.int64)


def clau_de_vista(p: Parametres, criteri: Criteri, dt: float, finestra_m, nivell: int) -> str:
    """Tot el que decideix una vista. Si la clau desada no es aquesta, es recalcula."""
    return json.dumps({
        "parametres": _parametres_json(p),
        "criteri": criteri.com_a_dict(),
        "dt": dt,
        "finestra_m": [float(v) for v in finestra_m],
        "fi": FI,
        "nivell": nivell,
    }, sort_keys=True)


def vista_desada(nom: str, p: Parametres, criteri: Criteri, dt: float, finestra_m,
                 nivell: int, refes: bool = False):
    """La vista de la cau si s'hi va fer amb el mateix de tot; si no, calculada i desada.

    Torna (codis, passos, d'on_surt). Una segona construccio sense canvis no
    recalcula res: es el que la fa trigar segons en comptes d'una hora.
    """
    cami = CAU_VISTES / f"pagina_{nom}.npz"
    clau = clau_de_vista(p, criteri, dt, finestra_m, nivell)
    if cami.exists() and not refes:
        with np.load(cami, allow_pickle=False) as z:
            if str(z["clau"]) == clau:
                return z["codis"].astype(np.int64), z["passos"].astype(np.int64), "de la cau"
    codis, passos = calcula_reticula(p, criteri, dt, finestra_m, nivell)
    cami.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cami, codis=codis.astype(np.int8), passos=passos,
                        clau=np.array(clau))
    return codis, passos, "calculada"


def codifica(nom: str, etiqueta: str, finestra_m, nivell: int, codis, passos) -> dict:
    """Una vista en el format que llegeix pagina.js.

    COMPRIMIDES (28/10): zlib aqui i DecompressionStream("deflate") al navegador,
    que es el mateix format. Sense comprimir, cada vista ocupava 2,4 MB en base64
    i quatre farien una pagina de 10 MB.
    """
    punts, salt = NIVELLS[nivell]
    assert codis.shape == (punts, punts) and passos.shape == (punts, punts)

    # EL TEMPS NO ES DESA COM A DECIMAL, SINO COMPTANT PASSOS. El temps de
    # caiguda sempre es k*dt amb k sencer, i desar-hi el k es EXACTE: la pagina
    # refa k*dt amb els mateixos dos doubles i li surt el mateix numero bit a
    # bit. Un float32 hi perdria xifres i el temps de la pagina no seria el del
    # Python.
    #
    # L'AMPLADA LA DECIDEIXEN LES DADES, no una suposicio. Aixo va canviar el
    # 2026-09-29, amb el t_max de 60 a 200 s: amb 60 s el k mes gros possible
    # era 30.000 i aqui hi havia escrit que "cap en dos bytes" com si fos una
    # propietat del format. No ho era, era una propietat d'aquell t_max. Amb
    # 200 s el sostre puja a 100.000 i, mesurat a +-12 cm, la trajectoria mes
    # lenta triga 150,6 s = 75.320 passos: no hi cabria.
    #
    # Es mira el maxim DE DEBO i no el t_max, perque amb 0 % de gris ningu no hi
    # arriba: a 120x120 la mediana son 86,8 s i nomes un 0,5 % dels pixels passa
    # dels 131 s. Aixi la pagina nomes creix quan cal, i el dia que algu posi un
    # t_max de 500 s no petara ni mentira: ocupara una mica mes.
    amplada = 2 if passos.max() <= 65535 else 4
    if passos.max() > 4294967295:
        raise RuntimeError(
            f"El temps maxim son {passos.max()} passos i ni en quatre bytes n'hi "
            "caben (4.294.967.295, que son 99 dies). Aixo no es un limit del "
            "format: si hi has arribat, mira el t_max.")

    # Fila per fila de baix a dalt i columna per columna, el mateix ordre amb
    # que ho llegeix pagina.js. Els codis van de -1 a 3: hi caben en un byte.
    crus = (codis.astype("<i1").tobytes(),
            passos.astype("<u2" if amplada == 2 else "<u4").tobytes())
    comprimits = [zlib.compress(c, 9) for c in crus]
    # Que es desfaci exactament: si no, la pagina ensenyaria un altre mapa.
    assert all(zlib.decompress(z) == c for z, c in zip(comprimits, crus))
    return {
        "nom": nom,
        "etiqueta": etiqueta,
        "nivell": nivell,
        "salt": salt,
        "punts": punts,
        "finestra": [float(v) for v in finestra_m],
        "codis": base64.b64encode(comprimits[0]).decode("ascii"),
        "passos": base64.b64encode(comprimits[1]).decode("ascii"),
        # Quants bytes ocupa cada compte de passos. La pagina el LLEGEIX d'aqui;
        # si el suposes, el dia que canvii el t_max el mapa incrustat surt
        # escombraries sense que res no ho digui.
        "bytes_del_temps": amplada,
        "grisos": int(np.count_nonzero(codis == -1)),
    }


def finestra_de_vista(graella, finestra_general_m):
    """La finestra d'una vista, en metres: la de --finestra o la del JSON del zoom."""
    if graella is None:
        return list(finestra_general_m)
    cami = GRAELLES / f"{graella}.json"
    if not cami.exists():
        return None
    f = json.loads(cami.read_text(encoding="utf-8"))["finestra_m"]
    return [f["x_min"], f["x_max"], f["y_min"], f["y_max"]]

# Quant s'ha de moure un punt de l'autotest sense que li canvii el desti perque
# el considerem de dins de conca. Una desena de mil.limetre es mil vegades el
# pixel mes fi que la pagina ensenya a la vista general.
MARGE_AUTOTEST_M = 1e-4
QUANTS_AUTOTEST = 24


# ---------------------------------------------------------------------------
# Els punts de l'autotest
# ---------------------------------------------------------------------------
def candidats():
    """Punts escampats per tot el mapa, no nomes d'una zona.

    Es fa amb una espiral perque surtin repartits en radi i en angle alhora:
    si tots fossin de prop del centre, l'autotest no diria res dels imants.
    """
    fora = []
    for k in range(120):
        r = 0.005 + 0.075 * (k / 119.0)
        ang = k * 2.399963  # angle auri: reparteix be sense repetir direccions
        fora.append((r * math.cos(ang), r * math.sin(ang)))
    return fora


def tria_punts(p: Parametres, criteri: Criteri, dt: float):
    """Els punts de dins de conca, amb el desti i el temps que en diu el Python."""
    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    h = MARGE_AUTOTEST_M
    tria = []
    for x, y in candidats():
        codi, temps = desenllac(p, x, y, criteri, dest_x, dest_y, dest_codi, dest_radi2, dt)
        if codi < 0:
            continue  # no convergit: no serveix de referencia
        estable = True
        for dx, dy in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            c2, _ = desenllac(p, x + dx, y + dy, criteri,
                              dest_x, dest_y, dest_codi, dest_radi2, dt)
            if c2 != codi:
                estable = False
                break
        if not estable:
            continue
        tria.append({"x": float(x), "y": float(y), "codi": int(codi), "temps": float(temps)})
        if len(tria) >= QUANTS_AUTOTEST:
            break
    return tria


def confirma_amb_node(node, p, criteri, dt, punts):
    """Que el JavaScript hi doni el mateix, amb ELS SEUS punts de repos.

    Es demana primer a node que calculi els punts de repos pel seu compte
    (biseccio pel radi) i despres se li fa fer els desenllacos amb aquells, que
    es exactament el que fara la pagina. Retorna els punts que coincideixen del
    tot i la llista dels que no.
    """
    pj = _parametres_json(p)
    repos = crida_node(node, [{"tipus": "repos", "parametres": pj}])[0]

    dx, dy, codi, radi2 = [], [], [], []
    for i, q in enumerate(repos["punts"][:-1]):
        if q is None:
            continue
        dx.append(q["x"]); dy.append(q["y"])
        codi.append(i + 1); radi2.append(criteri.r_captura ** 2)
    if repos["centre_estable"]:
        c = repos["punts"][-1]
        dx.append(c["x"]); dy.append(c["y"])
        codi.append(0); radi2.append(criteri.r_centre ** 2)

    base = {
        "tipus": "desenllac", "parametres": pj,
        "dest_x": dx, "dest_y": dy, "dest_codi": codi, "dest_radi2": radi2,
        "dt": dt,
        "passos": int(round(criteri.t_max / dt)),
        "passos_confirma": int(round(criteri.t_confirma / dt)),
        "passos_confirma_centre": int(round(criteri.t_confirma_centre / dt)),
    }
    feines = [dict(base, x0=[q["x"]], y0=[q["y"]]) for q in punts]
    resultats = crida_node(node, feines)

    bons, dolents = [], []
    for q, r in zip(punts, resultats):
        if r["codi"][0] == q["codi"] and r["temps"][0] == q["temps"]:
            bons.append(q)
        else:
            dolents.append((q, int(r["codi"][0]), float(r["temps"][0])))
    return bons, dolents


# ---------------------------------------------------------------------------
# La galeria: els numeros i les figures ja calculades
# ---------------------------------------------------------------------------
# NOMES ES LLEGEIX. Aqui no es recalcula cap figura ni cap numero: es van fer
# amb magnetic.mapa, magnetic.analisi, magnetic.autosemblanca i
# magnetic.regio_negra, i el que fa la pagina es ensenyar-los. Els numeros
# surten dels JSON i NO s'escriuen mai a ma.
#
# DOS PARANYS QUE HI HA, I QUE NO ES VEUEN FINS QUE PASSEN
#
# 1. magnetic/figures/ i magnetic/dades/ SON AL .gitignore. Qui cloni el
#    repositori no te ni figures ni JSON. Per aixo, si en falta un, aixo no
#    peta: se'l salta, ho diu amb l'ordre que el regenera, i la pagina marca
#    que aquell tros falta. Mateix criteri que --sense-mapa.
#
# 2. CADA FIGURA ES D'UN MUNTATGE. Construint amb --b 0.8, el mapa en viu seria
#    d'un muntatge i les figures d'un altre, i res no ho diria. Els JSON de les
#    graelles porten els parametres amb que es van fer, o sigui que es poden
#    comparar amb els d'ara i marcar les que no coincideixin. Es el parany del
#    bloc I3 en un lloc que encara no tenia cap defensa.
FIGURES = Path("magnetic/figures")
GRAELLES = Path("magnetic/dades")

# Quines graelles van a la taula de referencia, i com s'etiqueten. L'ordre es el
# de la memoria: la vista general primer i les ampliacions a sota.
REFERENCIA = (
    ("mapa_general", "×1"),
    ("zoom1", "×4"),
    ("zoom2", "×16"),
    ("zoom3", "×64"),
    ("friccio_13", "b = 1,3"),
)


# ELS PEUS DE FIGURA AMB NUMEROS ES FAN AMB LES DADES, no s'escriuen a ma. El
# 27/10 la pagina encara deia "forma d'U", "b inventada" i "la pols desapareix a
# b = 0,8, dimensio de 1,998 a 1,748": tot de la llei inventada, i cap d'aquells
# numeros no sortia de cap fitxer. Una funcio que no troba el seu JSON peta amb
# FileNotFoundError, i figures_galeria() ho apunta a "falten".
def _coma(v: float, xifres: int) -> str:
    return f"{v:.{xifres}f}".replace(".", ",")


def _de(v: float, xifres: int) -> str:
    """"del 30 %" pero "de l'1,32 %" i "de l'11,56 %": s'apostrofa davant d'u i d'onze."""
    t = _coma(v, xifres)
    return ("de l'" if t.split(",")[0] in ("1", "11") else "del ") + t


def _a(v: float, xifres: int) -> str:
    t = _coma(v, xifres)
    return ("a l'" if t.split(",")[0] in ("1", "11") else "al ") + t


def _llegeix(cami: Path) -> dict:
    return json.loads(cami.read_text(encoding="utf-8"))


def _peu_regio_negra() -> str:
    """La forma de la corba la diuen les dades. Amb n = 3 era una U -- un minim
    pel mig --, i aqui hi deia aixo. Ara nomes es diu "sense cap minim pel mig"
    si el minim cau de debo a la b mes baixa de l'escombrada. "Creix sempre" no
    es pot dir: a +-12 cm hi ha un pas enrere de 0,04 punts entre 0,039 i 0,05."""
    d = _llegeix(FIGURES / "regio_negra.json")
    files = sorted(d["files"], key=lambda f: f["b"])
    b = [f["b"] for f in files]
    negre = [f["negre"] for f in files]
    front = [f["frontera"] for f in files]
    kmin, kmax = negre.index(min(negre)), negre.index(max(negre))
    sense_u = kmin == 0

    def xifra(v):
        return _coma(v, 3).rstrip("0").rstrip(",")

    forma = (
        f"va {_de(negre[kmin], 2)} % a la b més baixa {_a(negre[kmax], 2)} % a b = {xifra(b[kmax])}, "
        f"sense cap mínim pel mig" if sense_u else
        f"té un mínim a b = {xifra(b[kmin])} ({_coma(negre[kmin], 2)} %) i el màxim a "
        f"b = {xifra(b[kmax])} ({_coma(negre[kmax], 2)} %)")
    abans = ("Amb la llei de força inventada (n = 3) aquesta corba tenia forma d'U, amb "
             "un mínim pel mig; amb la mesurada, no." if sense_u else
             "Amb la llei de força inventada (n = 3) aquesta corba tenia forma d'U.")
    return (
        f"Una escombrada de {len(b)} mapes fent variar la fricció, de b = {xifra(b[0])} "
        f"a {xifra(b[-1])} s⁻¹ (la mesurada és {xifra(d['b_del_fitxer_de_parametres'])}), "
        f"a ±{abs(d['finestra_cm'][0]):g} cm i {d['resolucio']} × {d['resolucio']} píxels. "
        f"La part del mapa que acaba al centre {forma}. La frontera passa "
        f"{_de(front[0], 0)} % {_a(front[-1], 0)} %. {abans}")


def _peu_friccio_gran() -> str:
    general = _llegeix(FIGURES / "mapa_general_metriques.json")
    gran = _llegeix(FIGURES / "friccio_13_metriques.json")
    b_gran = _llegeix(GRAELLES / "friccio_13.json")["_sobreescrits"]["b"]
    b_mesurada = _llegeix(GRAELLES / "mapa_general.json")["parametres"]["b"]
    return (
        f"Mateixa finestra que el mapa general, amb una fricció "
        f"{b_gran / b_mesurada:.0f} vegades la mesurada (b = {_coma(b_gran, 1)} s⁻¹ en "
        f"comptes de {_coma(b_mesurada, 3)}). No és cap mesura: és un valor triat "
        f"perquè s'hi vegi el mapa sense pols. La pols de fora es converteix en "
        f"bandes netes: a {gran['resolucio']} × {gran['resolucio']} píxels, la "
        f"frontera passa {_de(general['percentatge_de_frontera'], 2)} % "
        f"{_a(gran['percentatge_de_frontera'], 2)} % i la dimensió, de "
        f"{_coma(general['dimensio_box_counting'], 2)} a "
        f"{_coma(gran['dimensio_box_counting'], 2)}.")

# Les figures que la pagina NO sap fer sola. El mapa general, els tres zooms i
# el temps de caiguda no hi son a posta: la pagina ja els fa en viu i
# ampliables, i un PNG fix del mateix mapa seria 1,9 MB per ensenyar-ne una
# versio pitjor -- a mes de contradir el que la pagina defensa, que ampliar
# RECALCULA i no estira cap imatge.
FIGURES_GALERIA = (
    {
        "fitxer": "autosemblanca.png",
        "titol": "Autosemblança",
        "peu": "La mateixa estructura a ×1, ×4, ×16 i ×64. És la prova visual que "
               "la frontera no té gruix: per molt que hi ampliïs, hi torna a "
               "aparèixer detall. La pàgina no pot ensenyar quatre escales alhora.",
        "ordre": "-m magnetic.autosemblanca",
        "graella": "mapa_general",
    },
    {
        "fitxer": "regio_negra.png",
        "titol": "L'àrea negra en funció de la fricció",
        # Fins al 27/10 deia "Te forma d'U", que era de la llei inventada.
        "peu": _peu_regio_negra,
        "ordre": "-m magnetic.regio_negra",
        "graella": None,          # aquesta figura fa variar la b a posta
    },
    {
        "fitxer": "mapa_general_boxcounting.png",
        "titol": "D'on surt la dimensió fractal",
        "peu": "L'ajust log-log del box-counting: quantes capses de costat s "
               "calen per cobrir la frontera. El pendent és la dimensió. Sense "
               "aquesta figura, el número del costat no té procedència.",
        "ordre": "-m magnetic.analisi magnetic/dades/mapa_general.npz",
        "graella": "mapa_general",
    },
    {
        "fitxer": "friccio_13.png",
        "titol": "El mateix mapa amb b = 1,3",
        # Fins al 27/10 era b = 0,8. Amb la llei mesurada, a 0,8 encara hi
        # queda gra a fora; en Guillem va triar 1,3 el 28/10 mirant-los de costat.
        "peu": _peu_friccio_gran,
        "ordre": "-m magnetic.mapa --b 1.3 --resolucio 600 --nom friccio_13",
        "graella": "friccio_13",
    },
)


def _muntatge(dades: dict) -> dict:
    """Els tres valors que decideixen el mapa, tal com els va desar la graella."""
    q = dades.get("parametres", {})
    return {
        "b": q.get("b"),
        "n": q.get("n"),
        "K": [i["K"] for i in q.get("imants", [])],
        "sobreescrits": dades.get("_sobreescrits"),
    }


def _mateix_muntatge(m: dict, p: Parametres) -> bool:
    return (m["b"] == p.b and m["n"] == p.n
            and m["K"] == [float(v) for v in p.im_K])


def _llegeix_json(cami: Path, falten: list, ordre: str):
    if not cami.exists():
        falten.append(f"{cami}  ->  {ordre}")
        return None
    return json.loads(cami.read_text(encoding="utf-8"))


def referencia(p: Parametres, falten: list) -> list:
    """La taula de % de frontera i dimensio, llegida dels JSON."""
    fora = []
    for nom, etiqueta in REFERENCIA:
        ordre = f"-m magnetic.mapa --nom {nom}  (i despres -m magnetic.analisi)"
        met = _llegeix_json(FIGURES / f"{nom}_metriques.json", falten, ordre)
        gra = _llegeix_json(GRAELLES / f"{nom}.json", falten, ordre)
        if met is None or gra is None:
            continue
        f = gra["finestra_m"]
        m = _muntatge(gra)
        # L'ordre que la torna a calcular es DEDUEIX del JSON, no s'escriu a ma:
        # aixi no pot quedar desaparellada de la figura que descriu.
        cm = [f["x_min"] * 100, f["x_max"] * 100, f["y_min"] * 100, f["y_max"] * 100]
        banderes = "".join(f" --{k} {v:g}" for k, v in (m["sobreescrits"] or {}).items())
        fora.append({
            "etiqueta": etiqueta,
            "graella": nom,
            "finestra_cm": cm,
            "resolucio": met["resolucio"],
            "pixel_m": met["pixel_m"],
            "frontera": met["percentatge_de_frontera"],
            "dimensio": met["dimensio_box_counting"],
            # Des del 31/10, l'interval dels pendents locals. Un JSON de mes
            # abans no el porta, i la pagina llavors ensenya la dimensio sola.
            "interval": met.get("dimensio_interval"),
            "grisos": met["pixels_grisos_percent"],
            "muntatge": m,
            "mateix_muntatge": _mateix_muntatge(m, p),
            "ordre": (f"-m magnetic.mapa{banderes} --finestra "
                      + " ".join(f"{v:g}" for v in cm)
                      + f" --resolucio {met['resolucio']} --nom {nom}"),
        })
    return fora


def figures_galeria(p: Parametres, falten: list) -> list:
    """Les figures en base64, amb la marca de quin muntatge son."""
    fora = []
    for q in FIGURES_GALERIA:
        cami = FIGURES / q["fitxer"]
        if not cami.exists():
            falten.append(f"{cami}  ->  {q['ordre']}")
            continue
        peu = q["peu"]
        if callable(peu):
            try:
                peu = peu()
            except FileNotFoundError as e:
                falten.append(f"{e.filename}  ->  els numeros del peu de {q['fitxer']}")
                continue
        mateix, muntatge = None, None
        if q["graella"]:
            gra = GRAELLES / f"{q['graella']}.json"
            if gra.exists():
                muntatge = _muntatge(json.loads(gra.read_text(encoding="utf-8")))
                mateix = _mateix_muntatge(muntatge, p)
        fora.append({
            "fitxer": q["fitxer"],
            "titol": q["titol"],
            "peu": peu,
            "ordre": q["ordre"],
            "muntatge": muntatge,
            "mateix_muntatge": mateix,
            "kb": round(cami.stat().st_size / 1024),
            "dades": "data:image/png;base64,"
                     + base64.b64encode(cami.read_bytes()).decode("ascii"),
        })
    return fora


def galeria(p: Parametres):
    falten = []
    fora = {
        "referencia": referencia(p, falten),
        "figures": figures_galeria(p, falten),
        "falten": falten,
    }
    return fora


# ---------------------------------------------------------------------------
# Les constants que s'incrusten
# ---------------------------------------------------------------------------
def magma_256():
    """La paleta del temps de caiguda, la mateixa que fa servir magnetic.mapa.

    S'agafa de matplotlib i no es copia a ma: la figura del PDF i la de la
    pagina han de sortir del mateix lloc o algun dia no s'assemblaran.
    """
    import matplotlib

    cmap = matplotlib.colormaps["magma"]
    return [[int(round(255 * c)) for c in cmap(i / 255.0)[:3]] for i in range(256)]


def constants(p: Parametres, criteri: Criteri, dt: float, autotest, finestra_m,
              vistes, galeria_dades):
    return "\n".join([
        # Una vista per linia de text: compara.py les llegeix d'aqui per saber
        # quantes n'ha de comprovar.
        "var VISTES = " + json.dumps(vistes) + ";",
        "var GALERIA = " + json.dumps(galeria_dades) + ";",
        "var OBJECTIU_PER_DEFECTE = " + str(OBJECTIU_PER_DEFECTE) + ";",
        # R, m i inventats NO son al _parametres_json: aquell es el que menja
        # el motor, i el motor no en te cap necessitat -- la massa no surt a
        # les equacions i la R ja va dins de im_x/im_y. Hi son perque el bloc
        # de parametres de la pagina els ha d'ENSENYAR, i sobretot perque
        # l'asterisc de valor inventat ha de sortir de la llista de debo: el
        # dia que b, n o K es mesurin i surtin de _valors_inventats, la marca
        # ha de desapareixer sola de la pagina, com ja fa l'avis de dalt.
        "var PARAMETRES = " + json.dumps(
            dict(_parametres_json(p), noms=list(p.noms),
                 R=p.R, m=p.m, inventats=list(p.inventats))) + ";",
        "var CRITERI = " + json.dumps(criteri.com_a_dict()) + ";",
        "var DT = " + repr(dt) + ";",
        "var FINESTRA_INICIAL = " + json.dumps(list(finestra_m)) + ";",
        "var COLORS = " + json.dumps({
            "no_convergit": estil.NO_CONVERGIT,
            "centre": estil.CENTRE,
            "imants": list(estil.IMANTS),
            "graella": estil.GRAELLA,
        }) + ";",
        "var MAGMA = " + json.dumps(magma_256()) + ";",
        "var AUTOTEST = " + json.dumps(autotest) + ";",
        # El motor va dins d'una cadena a mes d'anar com a <script>: els
        # treballadors web es fabriquen amb un Blob i necessiten el codi font.
        "var FONT_MOTOR = " + json.dumps((ARREL / "motor.js").read_text(encoding="utf-8")) + ";",
    ])


def omple(text: str, forats: dict) -> str:
    for clau, valor in forats.items():
        text = text.replace("{{" + clau + "}}", str(valor))
    if "{{" in text:
        sobrant = text[text.index("{{"): text.index("{{") + 40]
        raise RuntimeError(f"Ha quedat un forat sense omplir a la plantilla: {sobrant!r}")
    return text


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sortida", default=str(ARREL / "mapa-interactiu.html"))
    ap.add_argument("--finestra", type=float, nargs=4, default=list(FINESTRA_INICIAL_CM),
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="en CENTIMETRES")
    ap.add_argument("--dt", type=float, default=DT_PER_DEFECTE)
    ap.add_argument("--node", default=None, help="cami del node.exe, si no es al PATH")
    ap.add_argument("--sense-node", action="store_true",
                    help="no confirmis l'autotest amb node (es desaconsella)")
    ap.add_argument("--sense-mapa", action="store_true",
                    help="no incrustis cap vista (mes rapid, pero la pagina "
                         "s'obrira en gris i s'haura de calcular)")
    ap.add_argument("--refes-vistes", action="store_true",
                    help="torna a calcular les vistes encara que n'hi hagi de desades "
                         "amb els mateixos parametres")
    ap.add_argument("--nivell-incrustat", type=int, default=NIVELL_INCRUSTAT,
                    choices=range(len(NIVELLS)),
                    help=f"a quin nivell s'incrusten les vistes: "
                         + ", ".join(f"{i}={n[0]}" for i, n in enumerate(NIVELLS))
                         + f" (per defecte {NIVELL_INCRUSTAT}). Baixar-lo fa la pagina "
                           "mes petita i la construccio mes curta, pero llavors les "
                           "vistes s'han d'acabar de calcular al navegador.")
    ap.add_argument("--temps", type=float, default=None,
                    help="temps maxim d'integracio (s). Per defecte, el del "
                         "criteri (200 s). No te sostre: el temps incrustat "
                         "s'amplia sol de dos a quatre bytes si cal")
    afegeix_banderes_de_parametres(ap)
    args = ap.parse_args(argv)

    p = Parametres.des_de_json()
    p, avis_sobreescrit = sobreescriu(p, b=args.b, n=args.n, K=args.K)
    # El temps maxim el posa el criteri (200 s des del 2026-09-29) i la bandera
    # nomes hi es per provar. Amb la b inventada de 0,15 les trajectories es
    # morien en 13 s i els 60 d'abans sobraven; amb la mesurada, 0,039,
    # l'amplitud cau a 1/e en 51 s i 60 no arriben ni de bon tros: el mapa
    # sortiria clapat de gris, que NO es cap resultat sino l'avis que el temps
    # s'ha quedat curt. Aqui NO hi ha cap sostre: el que n'hi havia era dels dos
    # bytes del temps incrustat, i ara l'amplada la decideixen les dades.
    criteri = Criteri(t_max=args.temps) if args.temps else Criteri()
    finestra_m = [v / 100.0 for v in args.finestra]

    print("=" * 72)
    print(" magnetic.magnetic_interactiu.construeix -- la pagina en un fitxer")
    print("=" * 72)
    if avis_sobreescrit:
        print(avis_sobreescrit)
    avis = avis_inventats(p)
    if avis:
        print(avis)
    print()

    print("1. Triant els punts de l'autotest (de dins de conca)...")
    punts = tria_punts(p, criteri, args.dt)
    print(f"   {len(punts)} punts, amb el desti estable movent-los "
          f"{MARGE_AUTOTEST_M*1000:.1f} mm en les quatre direccions")

    if args.sense_node:
        print("\n2. node: SALTAT (--sense-node).")
        print("   L'autotest que s'incrusta NO s'ha vist passar mai. Aixo va contra")
        print("   la regla del CLAUDE.md: la copia s'ha de guanyar el dret a fer-se")
        print("   servir. Fes-ho servir nomes per mirar el disseny de la pagina.")
    else:
        node = busca_node(args.node)
        print(f"\n2. Confirmant-los amb node ({node})...")
        bons, dolents = confirma_amb_node(node, p, criteri, args.dt, punts)
        for q, c, t in dolents:
            print(f"   FORA ({q['x']*100:+.3f}, {q['y']*100:+.3f}) cm: "
                  f"Python diu {q['codi']}@{q['temps']:.3f} s i node {c}@{t:.3f} s")
        if not bons:
            print("\n   MALAMENT -> cap punt no coincideix. El motor de JavaScript no")
            print("   fa el mateix que el Python. Passa compara abans de continuar.")
            return 1
        if dolents:
            print(f"   {len(dolents)} punts descartats; en queden {len(bons)}.")
        else:
            print(f"   Els {len(bons)} coincideixen exactament, desti i temps.")
        punts = bons

    vistes = []
    if args.sense_mapa:
        print("\n3. Vistes incrustades: SALTADES (--sense-mapa). La pagina s'obrira en gris.")
    else:
        punts_nivell = NIVELLS[args.nivell_incrustat][0]
        print(f"\n3. Les vistes per incrustar ({punts_nivell}x{punts_nivell})...")
        print("   Es la part llarga de construir la pagina, i es fa UN COP: es desen a")
        print(f"   {CAU_VISTES}/pagina_*.npz i es reaprofiten si no ha canviat res.")
        for nom, etiqueta, graella in VISTES_INCRUSTADES:
            fin = finestra_de_vista(graella, finestra_m)
            if fin is None:
                print(f"   {etiqueta:>8}: FALTA {GRAELLES / (graella + '.json')} "
                      f"(-m magnetic.mapa --nom {graella}); la pagina no la portara")
                continue
            t0 = time.time()
            codis, passos, origen = vista_desada(nom, p, criteri, args.dt, fin,
                                                 args.nivell_incrustat, args.refes_vistes)
            v = codifica(nom, etiqueta, fin, args.nivell_incrustat, codis, passos)
            vistes.append(v)
            print(f"   {etiqueta:>8}: x {fin[0]*100:.4f} a {fin[1]*100:.4f} cm, "
                  f"y {fin[2]*100:.4f} a {fin[3]*100:.4f} cm  {origen} "
                  f"({(time.time() - t0)/60:.1f} min), {v['grisos']} grisos, "
                  f"{(len(v['codis']) + len(v['passos']))/1024:.0f} kB")
            if v["grisos"]:
                print("            ATENCIO: hi ha punts sense convergir.")

    print("\n4. Llegint la galeria (numeros i figures ja calculats)...")
    gal = galeria(p)
    for r in gal["referencia"]:
        marca = "" if r["mateix_muntatge"] else "   <-- ALTRE MUNTATGE"
        print(f"   {r['etiqueta']:>8}  frontera {r['frontera']:7.4f} %  "
              f"dimensio {r['dimensio']:.4f}  ({r['graella']}){marca}")
    for q in gal["figures"]:
        marca = "" if q["mateix_muntatge"] is not False else "   <-- ALTRE MUNTATGE"
        print(f"   figura    {q['fitxer']:<32} {q['kb']:>5} kB{marca}")
    if gal["falten"]:
        print("\n   FALTEN, i la pagina ho dira:")
        for q in gal["falten"]:
            print(f"      {q}")
        print("   (magnetic/figures i magnetic/dades son al .gitignore: en un")
        print("    repositori acabat de clonar no hi son fins que es tornen a fer.)")

    print("\n5. Escrivint la pagina...")
    css = omple((ARREL / "pagina.css").read_text(encoding="utf-8"), {
        "SUPERFICIE": estil.SUPERFICIE, "TINTA": estil.TINTA,
        "TINTA_2": estil.TINTA_2, "TINTA_SUAU": estil.TINTA_SUAU,
        "GRAELLA": estil.GRAELLA, "EIX": estil.EIX,
    })
    # La caixa de dalt de la pagina. Si s'han sobreescrit valors, aquell avis va
    # PRIMER: es el mes fort dels dos, perque vol dir que no s'esta ni als valors
    # provisionals del fitxer.
    avis_pagina = "\n\n".join(q for q in (avis_sobreescrit, avis) if q)
    cos = omple((ARREL / "pagina.html").read_text(encoding="utf-8"), {
        "AVIS_INVENTATS": avis_pagina or "Tots els paràmetres són mesures.",
        "CLASSE_AVIS": "arbitrari" if avis_sobreescrit else "",
        "DATA": date.today().isoformat(),
        "DT": args.dt,
        "R_CAPTURA": f"{criteri.r_captura*100:.1f}",
        "T_CONFIRMA": f"{criteri.t_confirma:.2f}",
        "T_CONFIRMA_CENTRE": f"{criteri.t_confirma_centre:.2f}",
        "T_MAX": f"{criteri.t_max:.0f}",
    })

    html = "\n".join([
        "<!doctype html>",
        '<html lang="ca">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Mapa de conques del pèndol magnètic</title>",
        "<style>\n" + css + "\n</style>",
        "</head>",
        "<body>",
        cos,
        "<script>\n" + (ARREL / "motor.js").read_text(encoding="utf-8") + "\n</script>",
        # analisi.js va a part de motor.js i NO entra als treballadors: els
        # treballadors calculen desenllacos i prou. La frontera i la dimensio es
        # compten un sol cop al fil principal, sobre la matriu ja feta.
        "<script>\n" + (ARREL / "analisi.js").read_text(encoding="utf-8") + "\n</script>",
        "<script>\n" + constants(p, criteri, args.dt, punts, finestra_m, vistes, gal)
        + "\n</script>",
        "<script>\n" + (ARREL / "pagina.js").read_text(encoding="utf-8") + "\n</script>",
        "</body>",
        "</html>",
    ])

    sortida = Path(args.sortida)
    sortida.parent.mkdir(parents=True, exist_ok=True)
    sortida.write_text(html, encoding="utf-8")

    print(f"   {sortida}  ({len(html.encode('utf-8'))/1024:.0f} kB)")
    print("\n" + "=" * 72)
    print("OK -> obre-la amb doble clic. No necessita servidor ni connexio.")
    print(f"   ii {sortida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
