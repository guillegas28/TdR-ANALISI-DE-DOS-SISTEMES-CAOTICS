"""La finestra interactiva del pendol doble: veure'l, clonar-lo i veure com se separen.

Es el bloc D2 de documentacio/pla-doble-interactiu.md. Serveix per FER ENTENDRE
el pendol -al tribunal, i a qui l'estudia- i per trobar errors que cap prova no
veu: una barra que fa la llargada que no toca, un signe girat, un rellotge que
no quadra. NO DEMOSTRA RES. Els numeros de la memoria surten de les ordres de
Python de pendol/ i eines/, i la finestra ho diu al peu.

QUE HI HA A SOTA
----------------
Res de nou. Tot el que es veu surt de funcions que ja existeixen i ja estan
provades: fisica.derivades, energia i posicions, simulador.simula amb les
seves tolerancies, i mesures.distancia_fases, omega_zero, dif_angular i
voltes. Amb la friccio, a mes, fisica.Dissipacio.firmada. La finestra nomes ho dibuixa: el que es veu moure's es literalment el
que dona els numeros. Per aixo aquest fitxer te dues parts separades a posta,
el calcul (escena), que es prova sol, i la finestra, que nomes el dibuixa.

ELS DOS PENDOLS
---------------
L'original, en el blau de la simulacio, i el CLON, en el blau clar del feix.
No s'hi afegeix cap color: el clon es un membre del feix, la mateixa entitat
que la simulacio, i el taronja es nomes per al pendol real. El clon pot ser:

  - «clon d'angle»: el mateix pendol amb l'angle inicial mogut el que es
    digui. Per defecte 0,3 graus a theta1, que es el soroll de seguiment: la
    incertesa amb que es coneix d'on s'amolla un pendol de debo.
  - «dues tolerancies»: la MATEIXA condicio inicial, calculada amb la
    tolerancia rapida i amb l'estricta. Comencen iguals fins a la desena xifra
    i acaben separant-se: el pendol es determinista pero no es pot calcular
    indefinidament, ni tan sols sense cap soroll.

LES PRESES REALS
----------------
En aquesta finestra, cap. Entren pel D3 (pendol/interactiu_d3.py, el boto
«Importa»), que passa pel PANY (pot_importar: res del Tracker sense
dades/versemblant/firma.json) i, des del 2/11, les corregeix amb
comparar.corregeix_presa abans de dibuixar-les. El pany es obert des del 27/10,
quan es van firmar X, Y i Z. El que s'hi veu NO es cap comparacio firmada: el
veredicte de l'experiment 1 es el d'eines/experiment1.py.

LA FRICCIO
----------
Des del 2/11, la FIRMADA (Dissipacio.firmada: parametres.json, fisic.friccio_firmada),
si es demana. Nomes als angles validats (45, o 90 i 150 graus, +-10) i amb els
bracos rectes (|theta2 - theta1| fins a la mateixa tolerancia, decidit per en
Guillem el 2/11): la friccio es va mesurar amb tirs amollats aixi. Fora d'aqui no
es calcula. documentacio/pla-interactiu-al-dia.md.

Us:
    .venv\\Scripts\\python -m pendol.interactiu
    .venv\\Scripts\\python -m pendol.interactiu --theta1 120 --theta2 120 --temps 20
    .venv\\Scripts\\python -m pendol.interactiu --theta1 90 --theta2 90 --friccio --temps 60
    .venv\\Scripts\\python -m pendol.interactiu --autoprova
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pendol import estil, mesures
from pendol.fisica import Dissipacio, Parametres, energia, posicions
from pendol.simulador import TOLERANCIES, simula

ARREL = Path(__file__).resolve().parent.parent

FPS_CALCUL = 200.0    # mostres per segon de les taules que es dibuixen
FPS_PANTALLA = 25.0   # fotogrames per segon de l'animacio
TRACA_S = 1.5         # quants segons de traca del gomet es veuen
SOROLL_GRAUS = 0.3    # el soroll de seguiment: el clon per defecte, i la ratlla
                      # de referencia del grafic de la distancia
MODES = {"angle": "clon d'angle", "tolerancies": "dues tolerancies"}
PER_DEFECTE = {"theta1": 150.0, "theta2": 150.0, "d_theta1": SOROLL_GRAUS,
               "d_theta2": 0.0, "temps": 30.0, "mode": "angle", "friccio": False}


def parametres() -> Parametres:
    """El muntatge mesurat, de parametres.json."""
    return Parametres.des_de_json(ARREL / "parametres.json")


def bloc_friccio() -> dict:
    """El bloc fisic.friccio_firmada de parametres.json, nomes per llegir-ne la
    tolerancia i ensenyar-lo. La friccio de debo la dona Dissipacio.firmada."""
    return json.loads((ARREL / "parametres.json").read_text(encoding="utf-8"))["fisic"]["friccio_firmada"]


def friccio_firmada(theta1: float, theta2: float) -> Dissipacio:
    """La friccio FIRMADA per a un pendol amollat de (theta1, theta2) graus, o ValueError.

    El joc el tria Dissipacio.firmada pel theta1, i fora dels angles validats
    s'atura amb el seu missatge. A mes, decidit per en Guillem el 2/11: amb els
    bracos rectes, perque la friccio es va mesurar amb tirs amollats aixi.
    """
    tol = float(bloc_friccio()["tolerancia_angle_graus"])
    plec = abs(float(np.degrees(mesures.dif_angular(np.radians(theta2), np.radians(theta1)))))
    if plec > tol:
        raise ValueError(_coma(f"la friccio firmada es va mesurar amb els bracos rectes, i aqui "
                               f"theta2 - theta1 = {plec:.1f} graus (nomes val fins a {tol:g})"))
    return Dissipacio.firmada(theta1, cami=ARREL / "parametres.json")


def _coma(text: str) -> str:
    """El text amb la coma decimal, com els eixos (pendol/estil.py)."""
    return text.replace(".", ",")


# ------------------------------------------------------------------ el calcul


def escena(p: Parametres, theta1: float, theta2: float, d_theta1: float = SOROLL_GRAUS,
           d_theta2: float = 0.0, temps: float = 30.0, mode: str = "angle",
           fps: float = FPS_CALCUL, friccio: bool = False) -> dict:
    """Els dos pendols, ja calculats, i tot el que la finestra n'ensenya.

    Angles en GRAUS, que son el que s'escriu a la finestra; tot el que torna es
    en radians i SI, com a la resta del codi, excepte les separacions, que ja
    surten en graus perque son per llegir.

    Amb `friccio`, tots dos pendols porten la friccio firmada del joc de l'original
    (friccio_firmada); si no hi es validada, ValueError i no es calcula res.

    No hi ha cap estat de finestra: es pot cridar i provar sense Tk.
    """
    if mode not in MODES:
        raise ValueError(f"mode desconegut: {mode}")
    dis = friccio_firmada(theta1, theta2) if friccio else None
    estat_a = np.radians([theta1, theta2, 0.0, 0.0])
    if mode == "angle":
        estat_b = np.radians([theta1 + d_theta1, theta2 + d_theta2, 0.0, 0.0])
        tol_a = tol_b = "rapida"
    else:
        estat_b = estat_a.copy()
        tol_a, tol_b = "rapida", "estricta"

    def corre(estat, tol):
        rtol, atol = TOLERANCIES[tol]
        return simula(p, estat, temps=temps, fps=fps, rtol=rtol, atol=atol, dis=dis)

    t, ya = corre(estat_a, tol_a)
    _, yb = corre(estat_b, tol_b)
    w0 = mesures.omega_zero(p)
    return {
        "t": t,
        "a": ya,
        "b": yb,
        "energia_a": energia(ya, p),
        "energia_b": energia(yb, p),
        "voltes_a": mesures.voltes_fins_ara(ya[1]),
        "voltes_b": mesures.voltes_fins_ara(yb[1]),
        "sep_theta1_graus": np.degrees(mesures.dif_angular(yb[0], ya[0])),
        "sep_theta2_graus": np.degrees(mesures.dif_angular(yb[1], ya[1])),
        "d": mesures.distancia_fases(ya, yb, w0),
        "omega0": w0,
        "posicions_a": posicions(ya[0], ya[1], p),
        "posicions_b": posicions(yb[0], yb[1], p),
        "tolerancies": (tol_a, tol_b),
        "dissipacio": dis,
        "entrada": {"theta1": theta1, "theta2": theta2, "d_theta1": d_theta1,
                    "d_theta2": d_theta2, "temps": temps, "mode": mode, "fps": fps,
                    "friccio": bool(friccio)},
    }


def _plegat_graus(theta):
    """L'angle plegat a (-180, 180] graus, que es com es llegeix «on es ara»."""
    return np.degrees(mesures.dif_angular(theta, 0.0))


def _amb_forats(graus):
    """Angle plegat per dibuixar: un forat on salta de +180 a -180, no una ratlla."""
    g = np.array(graus, dtype=float)
    g[1:][np.abs(np.diff(g)) > 180.0] = np.nan
    return g


def lectura(e: dict, k: int) -> dict:
    """Els textos de la lectura en directe a l'instant k de l'escena, amb la coma decimal.

    L'energia perduda, sense friccio, es nomes la deriva numerica (milionesimes de
    mJ) i va en notacio cientifica; amb la friccio, son desenes de mJ, en decimals.
    """
    format_perduda = "8.1e" if e.get("dissipacio") is None else "8.1f"

    def pendol(y, E, voltes):
        return {
            "theta1": _coma(f"{_plegat_graus(y[0, k]):8.2f} graus"),
            "theta2": _coma(f"{_plegat_graus(y[1, k]):8.2f} graus"),
            "omega1": _coma(f"{y[2, k]:8.2f} rad/s"),
            "omega2": _coma(f"{y[3, k]:8.2f} rad/s"),
            "energia": _coma(f"{1000 * E[k]:8.3f} mJ"),
            "perduda": _coma(f"{1000 * (E[0] - E[k]):{format_perduda}} mJ"),
            "voltes": _coma(f"{voltes[k]:8.2f}"),
        }
    return {
        "a": pendol(e["a"], e["energia_a"], e["voltes_a"]),
        "b": pendol(e["b"], e["energia_b"], e["voltes_b"]),
        "sep_theta1": _coma(f"{e['sep_theta1_graus'][k]:8.2f} graus"),
        "sep_theta2": _coma(f"{e['sep_theta2_graus'][k]:8.2f} graus"),
        "d": _coma(f"{e['d'][k]:8.2e}"),
        "t": _coma(f"t = {e['t'][k]:6.2f} s"),
    }


# ------------------------------------------ importar i exportar (bloc D3, part 1)
#
# Fet el 29/09/2026 SENSE tocar la finestra: el calcul, provat sol, com escena().
# Els botons que ho fan servir son al D3 (pendol/interactiu_d3.py, des del 25/10).
#
# EL PANY. Importar una presa del Tracker i veure-la al costat de la seva
# simulacio ja es mirar la comparacio de l'experiment 1, i aixo no es podia fer
# abans de firmar X, Y i Z (deure 4). Per aixo pot_importar() no deixa passar
# res que vingui del Tracker fins que existeixi el fitxer de la firma, amb els
# tres numeros. Es conservador a posta: tambe tanca la «prova» sintetica de
# eines/genera_tracker_fals.py, que entra pel mateix cami que les de debo i
# porta el mateix «origen». La firma es va escriure el 27/10: des d'aleshores el
# pany deixa passar les preses del Tracker, i el D3 les corregeix (2/11).

FIRMA = ARREL / "dades" / "versemblant" / "firma.json"


def exporta_escena(e: dict, cami, p: Parametres | None = None) -> tuple[Path, Path]:
    """Els dos pendols de la pantalla, en format canonic: `cami` i `<cami>_clon`.

    Surten tal com son a la pantalla, AMB LA SEVA TOLERANCIA ESCRITA: la finestra
    calcula amb la rapida (i, en mode «dues tolerancies», el clon amb
    l'estricta), i un tir exportat ha de dir amb quina es va fer. Si se'n vol un
    per als numeros, es l'ordre del simulador, que va sempre per l'estricta.

    Sense friccio el pendol no s'atura mai, o sigui que l'exportacio no te final
    natural: el limit es el temps de l'escena, i es declara. I passat l'horitzo
    de prediccio, una trajectoria caotica concreta ja no vol dir res. Amb la
    friccio firmada, la `dissipacio` va al JSON, i el final continua sent el
    temps de l'escena.
    """
    from pendol import dades
    from pendol.simulador import METODE

    p = p or parametres()
    cami = Path(cami)
    cami_b = cami.with_name(f"{cami.stem}_clon{cami.suffix}")
    dis = e.get("dissipacio")
    comu = {
        "origen": "simulacio",
        "tipus": "finestra_interactiva",
        "parametres": p.com_a_dict(),
        "dissipacio": dis.com_a_dict() if dis is not None else None,
        "omega0_rad_s": float(e["omega0"]),
        "entrada_de_la_finestra": e["entrada"],
        "limit_de_temps_s": float(e["t"][-1]),
        "_limit_de_temps": ("sense friccio el pendol no s'atura: el final es el temps de "
                            "l'escena, i passat l'horitzo de prediccio la trajectoria "
                            "concreta ja no vol dir res") if dis is None else
                           ("el final es el temps de l'escena, i passat l'horitzo de "
                            "prediccio la trajectoria concreta ja no vol dir res"),
    }
    if dis is not None:
        comu["_dissipacio"] = ("la friccio firmada el 27/10/2026 (Dissipacio.firmada: parametres.json, "
                               "fisic.friccio_firmada), el joc de l'angle d'amollada de l'original")
    for y, tol, desti, quin in ((e["a"], e["tolerancies"][0], cami, "original"),
                                (e["b"], e["tolerancies"][1], cami_b, "clon")):
        rtol, atol = TOLERANCIES[tol]
        dades.desa(desti, e["t"], y[0], y[1], y[2], y[3], {
            **comu,
            "pendol": quin,
            "integrador": {"metode": METODE, "rtol": rtol, "atol": atol, "tolerancia": tol},
            "estat_inicial_rad": [float(v) for v in y[:, 0]],
            "energia_inicial_J": float(energia(y[:, 0], p)),
        })
    return cami, cami_b


def pot_importar(cami, firma=FIRMA) -> tuple[bool, str]:
    """(si es pot importar, per que). Vegeu EL PANY, a sobre."""
    import json

    from pendol import dades

    _, meta = dades.carrega(cami)
    origen = meta.get("origen")
    if origen == "simulacio":
        return True, "es una simulacio"
    firma = Path(firma)
    if not firma.exists():
        return False, (f"ve de «{origen}» i X, Y i Z encara no estan firmats "
                       f"(no hi ha {firma.name}): importar-la seria mirar la comparacio")
    f = json.loads(firma.read_text(encoding="utf-8"))
    falten = [k for k in ("X", "Y", "Z") if f.get(k) is None]
    if falten:
        return False, f"la firma no diu {', '.join(falten)}"
    return True, f"X, Y i Z firmats ({f.get('data', 'sense data')})"


def llegenda(meta: dict, cami=None) -> dict:
    """D'on surt un fitxer importat, TOT del seu JSON: no es calcula res.

    Aixi qualsevol captura de pantalla porta la seva procedencia a sobre.
    """
    if meta.get("origen") == "simulacio":
        est = meta.get("estat_inicial_rad") or [float("nan")] * 4
        integ = meta.get("integrador", {})
        return {
            "fitxer": str(cami) if cami else None,
            "origen": "simulacio",
            "angles_inicials_graus": [float(np.degrees(est[0])), float(np.degrees(est[1]))],
            "tolerancia": integ.get("tolerancia"),
            "friccio": meta.get("dissipacio") is not None,
            "durada_s": meta.get("limit_de_temps_s"),
        }
    a = meta.get("amollada") or {}
    rig = (meta.get("geometria") or {}).get("rigidesa") or {}
    return {
        "fitxer": str(cami) if cami else meta.get("fitxer_trk") or meta.get("fitxer_cru"),
        "origen": meta.get("origen"),
        "grup_nominal_graus": meta.get("angle_nominal_graus"),
        "angles_inicials_mesurats_graus": a.get("angles_amollada_graus"),
        "amollada_s": a.get("amollada_s"),
        "soroll_de_seguiment_graus": rig.get("soroll_angular_graus"),
        "fps": meta.get("fps"),
        "durada_s": meta.get("durada_s"),
    }


# ---------------------------------------------------------------- la finestra


class _Blit:
    """Redibuixa nomes el que es mou, sobre un fons desat.

    Sense aixo, a cada fotograma matplotlib tornaria a dibuixar eixos, graella
    i corbes senceres, i l'animacio aniria a batzegades.
    """

    def __init__(self, canvas, artistes):
        self.canvas = canvas
        self.artistes = artistes
        self.fons = None
        for a in artistes:
            a.set_animated(True)
        canvas.mpl_connect("draw_event", self._desa_fons)

    def _desa_fons(self, _event):
        self.fons = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._pinta()

    def _pinta(self):
        for a in self.artistes:
            self.canvas.figure.draw_artist(a)

    def actualitza(self):
        if self.fons is None:
            self.canvas.draw()
            return
        self.canvas.restore_region(self.fons)
        self._pinta()
        self.canvas.blit(self.canvas.figure.bbox)


class Finestra:
    """La finestra. Nomes dibuixa el que torna escena()."""

    def __init__(self, arrel_tk, p: Parametres, valors: dict | None = None):
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk, self.ttk = tk, ttk
        self.arrel = arrel_tk
        self.p = p
        self.e = None
        self.t = 0.0
        self.jugant = False
        self._movent_barra = False
        self._rellotge = None
        valors = {**PER_DEFECTE, **(valors or {})}

        estil.aplica()
        arrel_tk.title("Pendol doble - finestra interactiva")
        arrel_tk.configure(background=estil.SUPERFICIE)
        arrel_tk.protocol("WM_DELETE_WINDOW", self.tanca)

        pestanyes = ttk.Notebook(arrel_tk)
        pestanyes.pack(fill="both", expand=True)
        full = ttk.Frame(pestanyes, padding=6)
        full_par = ttk.Frame(pestanyes, padding=10)
        pestanyes.add(full, text="Pendol")
        pestanyes.add(full_par, text="Parametres")

        # --- controls ------------------------------------------------------
        ctl = ttk.Frame(full)
        ctl.pack(fill="x")
        self.var = {}
        for nom, text, amplada in (("theta1", "theta1 (graus)", 7), ("theta2", "theta2 (graus)", 7),
                                   ("d_theta1", "clon: +theta1", 6), ("d_theta2", "+theta2", 6),
                                   ("temps", "durada (s)", 5)):
            ttk.Label(ctl, text=text).pack(side="left", padx=(8, 2))
            self.var[nom] = tk.StringVar(value=_coma(f"{valors[nom]:g}"))
            ttk.Entry(ctl, textvariable=self.var[nom], width=amplada).pack(side="left")
        self.var_mode = tk.StringVar(value=valors["mode"])
        for clau, text in MODES.items():
            ttk.Radiobutton(ctl, text=text, value=clau, variable=self.var_mode).pack(
                side="left", padx=(10, 0))
        ttk.Button(ctl, text="Calcula", command=self.calcula).pack(side="left", padx=10)
        self.var_friccio = tk.BooleanVar(value=bool(valors["friccio"]))
        ttk.Checkbutton(ctl, text="amb la friccio firmada",
                        variable=self.var_friccio).pack(side="left", padx=6)

        # --- les dues figures ----------------------------------------------
        figures = ttk.Frame(full)
        figures.pack(fill="both", expand=True, pady=4)
        self.fig_p = Figure(figsize=(4.6, 4.6), dpi=90)
        self.eix_p = self.fig_p.add_subplot(111)
        self.fig_g = Figure(figsize=(6.4, 4.6), dpi=90)
        self.eix_th, self.eix_d, self.eix_e = self.fig_g.subplots(3, 1, sharex=True)
        self.canvas_p = FigureCanvasTkAgg(self.fig_p, master=figures)
        self.canvas_p.get_tk_widget().pack(side="left", fill="both", expand=True)
        self.canvas_g = FigureCanvasTkAgg(self.fig_g, master=figures)
        self.canvas_g.get_tk_widget().pack(side="left", fill="both", expand=True)
        self._munta_pendol()
        self._munta_grafics()

        # --- reproduccio ----------------------------------------------------
        rep = ttk.Frame(full)
        rep.pack(fill="x")
        self.boto_juga = ttk.Button(rep, text="Engega", width=8, command=self.alterna)
        self.boto_juga.pack(side="left")
        ttk.Label(rep, text="velocitat").pack(side="left", padx=(10, 2))
        self.var_velocitat = tk.StringVar(value="0,5")
        ttk.Combobox(rep, textvariable=self.var_velocitat, width=5, state="readonly",
                     values=("0,1", "0,25", "0,5", "1", "2")).pack(side="left")
        self.barra = ttk.Scale(rep, from_=0.0, to=1.0, orient="horizontal",
                               command=self._barra_moguda)
        self.barra.pack(side="left", fill="x", expand=True, padx=10)
        self.etiqueta_t = ttk.Label(rep, text="t = 0,00 s", width=12)
        self.etiqueta_t.pack(side="left")

        # --- lectura en directe --------------------------------------------
        lec = ttk.Frame(full)
        lec.pack(fill="x", pady=4)
        mono = ("Consolas", 10)
        self.etiquetes = {}
        files = (("theta1", "theta1"), ("theta2", "theta2"), ("omega1", "omega1"),
                 ("omega2", "omega2"), ("energia", "energia"), ("perduda", "energia perduda"),
                 ("voltes", "voltes de theta2"))
        self.titol_a = ttk.Label(lec, text="original", font=mono, foreground=estil.SIMULACIO)
        self.titol_a.grid(row=0, column=1, sticky="w", padx=8)
        self.titol_b = ttk.Label(lec, text="clon", font=mono, foreground=estil.FEIX)
        self.titol_b.grid(row=0, column=2, sticky="w", padx=8)
        for i, (clau, text) in enumerate(files, start=1):
            ttk.Label(lec, text=text, font=mono).grid(row=i, column=0, sticky="w")
            for col, quin in ((1, "a"), (2, "b")):
                self.etiquetes[f"{quin}_{clau}"] = ttk.Label(lec, text="", font=mono)
                self.etiquetes[f"{quin}_{clau}"].grid(row=i, column=col, sticky="w", padx=8)
        ttk.Label(lec, text="entre els dos", font=mono).grid(row=0, column=4, sticky="w", padx=(30, 0))
        for i, (clau, text) in enumerate((("sep_theta1", "separacio theta1"),
                                          ("sep_theta2", "separacio theta2"),
                                          ("d", "d (espai de fases)")), start=1):
            ttk.Label(lec, text=text, font=mono).grid(row=i, column=3, sticky="w", padx=(30, 0))
            self.etiquetes[clau] = ttk.Label(lec, text="", font=mono)
            self.etiquetes[clau].grid(row=i, column=4, sticky="w")

        # --- estat i peu -----------------------------------------------------
        self.etiqueta_estat = ttk.Label(full, text="", foreground=estil.REAL)
        self.etiqueta_estat.pack(fill="x")
        self.peu = ttk.Label(full, text="", foreground=estil.TINTA_2, wraplength=1000)
        self.peu.pack(fill="x", pady=(4, 0))

        self._munta_parametres(full_par)
        self.calcula()
        self._rellotge = arrel_tk.after(int(1000 / FPS_PANTALLA), self._tic)

    # --- muntatge de les figures ---------------------------------------------

    def _munta_pendol(self):
        eix, p = self.eix_p, self.p
        abast = (p.L1 + p.r_gomet) * 1.08
        eix.set_aspect("equal")
        eix.set_xlim(-abast, abast)
        eix.set_ylim(-abast, abast)
        eix.set_xlabel("x (m)")
        eix.set_ylabel("y (m)")
        eix.plot([0], [0], marker="o", markersize=5, color=estil.TINTA_2, zorder=5)
        (self.traca_a,) = eix.plot([], [], color=estil.SIMULACIO, alpha=0.35, linewidth=1.2)
        (self.traca_b,) = eix.plot([], [], color=estil.FEIX, alpha=0.6, linewidth=1.2)
        estil_brac = {"linewidth": 2.2, "marker": "o", "markersize": 7, "markeredgecolor": "none"}
        (self.brac_b,) = eix.plot([], [], color=estil.FEIX, markerfacecolor=estil.FEIX,
                                  label="clon", **estil_brac)
        (self.brac_a,) = eix.plot([], [], color=estil.SIMULACIO, linestyle=estil.TRAC_SIMULACIO,
                                  markerfacecolor=estil.SIMULACIO, label="original", **estil_brac)
        self.rellotge = eix.text(0.03, 0.96, "", transform=eix.transAxes, va="top",
                                 color=estil.TINTA_2)
        self.fig_p.tight_layout()
        self.blit_p = _Blit(self.canvas_p, [self.traca_a, self.traca_b, self.brac_b,
                                            self.brac_a, self.rellotge])

    def _munta_grafics(self):
        self.eix_th.set_ylabel("angle (graus)")
        self.eix_th.set_ylim(-190, 190)
        self.eix_th.set_yticks([-180, -90, 0, 90, 180])
        self.eix_d.set_ylabel("d")
        self.eix_d.set_yscale("log")
        self.eix_e.set_ylabel("energia\nperduda (mJ)")
        self.eix_e.set_xlabel("t (s)")
        self.cursors = [ax.axvline(0.0, color=estil.TINTA_SUAU, linewidth=0.9)
                        for ax in (self.eix_th, self.eix_d, self.eix_e)]
        self.fig_g.tight_layout()
        self.blit_g = _Blit(self.canvas_g, self.cursors)

    def _munta_parametres(self, full):
        p = self.p
        files = [
            ("massa de la barra 1", "m1", p.m1, "kg"), ("pivot 1 -> CG barra 1", "d1", p.d1, "m"),
            ("llargada de la barra 1", "R1", p.R1, "m"), ("massa dels cargols", "ms", p.ms, "kg"),
            ("pivot 1 -> cargols", "rs", p.rs, "m"), ("massa de la barra 2", "m2", p.m2, "kg"),
            ("pivot 2 -> CG barra 2", "d2", p.d2, "m"), ("llargada de la barra 2", "R2", p.R2, "m"),
            ("pivot 1 -> PIVOT 2", "L1", p.L1, "m"), ("pivot 2 -> gomet", "r_gomet", p.r_gomet, "m"),
            ("gravetat", "g", p.g, "m/s^2"),
            ("", "", None, ""),
            ("inercia del primer brac (pivot 1)", "I1", p.I1, "kg m^2"),
            ("inercia de la barra 2 (pivot 2)", "I2", p.I2, "kg m^2"),
            ("terme creuat de la cinetica", "K", p.K, "kg m^2"),
            ("coeficient de cos(theta1) a V", "A", p.A, "kg m"),
            ("coeficient de cos(theta2) a V", "B", p.B, "kg m"),
            ("escala de velocitats de d", "omega0", mesures.omega_zero(p), "rad/s"),
        ]
        mono = ("Consolas", 10)
        self.ttk.Label(full, text="De parametres.json, seccio «fisic»: el muntatge mesurat. "
                                  "Nomes es mostra, no es pot canviar des d'aqui.").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        for i, (text, simbol, valor, unitat) in enumerate(files, start=1):
            if valor is None:
                continue
            self.ttk.Label(full, text=text, font=mono).grid(row=i, column=0, sticky="w")
            self.ttk.Label(full, text=simbol, font=mono).grid(row=i, column=1, sticky="w", padx=12)
            self.ttk.Label(full, text=_coma(f"{valor:.5g}"), font=mono).grid(row=i, column=2, sticky="e")
            self.ttk.Label(full, text=unitat, font=mono).grid(row=i, column=3, sticky="w", padx=6)
        fila = len(files) + 2
        for nom, (rtol, atol) in TOLERANCIES.items():
            self.ttk.Label(full, text=_coma(f"tolerancia {nom}: rtol {rtol:g}, atol {atol:g}"),
                           font=mono).grid(row=fila, column=0, columnspan=4, sticky="w")
            fila += 1
        # La friccio firmada (27/10), tal com la llegeix Dissipacio.firmada. El
        # joc del simple no hi surt: aquesta finestra nomes fa el doble.
        bloc = bloc_friccio()
        fila += 1
        tol = _coma(f"{bloc['tolerancia_angle_graus']:g}")
        self.ttk.Label(full, text=(
            f"Friccio firmada el {bloc['data']} (parametres.json, fisic.friccio_firmada): el joc es tria per "
            f"l'angle d'amollada, +-{tol} graus, i amb els bracos rectes."),
            ).grid(row=fila, column=0, columnspan=4, sticky="w", pady=(8, 2))
        fila += 1
        for nom, j in bloc["jocs"].items():
            if nom == "simple":
                continue
            angles = " i ".join(f"{a:g}" for a in j["angles_graus"])
            self.ttk.Label(full, font=mono, text=_coma(
                f"doble, {angles} graus: tau_f1 {j['tau_f1']:.3g}, tau_f2 {j['tau_f2']:.3g} N m, "
                f"b1 {j['b1']:g}, b2 {j['b2']:g}, C {j['C']:g} kg/m, eps {j['eps']:g} rad/s")).grid(
                row=fila, column=0, columnspan=4, sticky="w")
            fila += 1

    # --- calcul i dibuix -----------------------------------------------------

    def _llegeix_entrada(self) -> dict:
        valors = {}
        for nom, var in self.var.items():
            text = var.get().strip().replace(",", ".")
            try:
                valors[nom] = float(text)
            except ValueError:
                raise ValueError(f"«{var.get()}» no es un nombre ({nom})") from None
        if not 1.0 <= valors["temps"] <= 120.0:
            raise ValueError("la durada ha d'anar d'1 a 120 s")
        valors["mode"] = self.var_mode.get()
        valors["friccio"] = bool(self.var_friccio.get())
        return valors

    def calcula(self):
        """Torna a calcular l'escena amb el que hi ha escrit a dalt."""
        try:
            v = self._llegeix_entrada()
        except ValueError as err:
            self.etiqueta_estat.configure(text=f"No es pot calcular: {err}")
            return
        self.etiqueta_estat.configure(text="calculant...")
        self.arrel.configure(cursor="watch")
        self.arrel.update_idletasks()
        try:
            self.e = escena(self.p, v["theta1"], v["theta2"], v["d_theta1"], v["d_theta2"],
                            v["temps"], v["mode"], friccio=v["friccio"])
        except ValueError as err:
            # La friccio firmada no hi es validada: es queda l'escena d'abans.
            self.etiqueta_estat.configure(text=f"No es pot calcular: {err}")
            return
        finally:
            self.arrel.configure(cursor="")
        self.etiqueta_estat.configure(text="")
        self.t = 0.0
        self.jugant = False
        self.boto_juga.configure(text="Engega")
        self.barra.configure(to=float(self.e["t"][-1]))
        self._redibuixa_grafics()
        self._mostra(0.0)

    def _redibuixa_grafics(self):
        e = self.e
        t = e["t"]
        for ax in (self.eix_th, self.eix_d, self.eix_e):
            for linia in list(ax.lines):
                if linia not in self.cursors:
                    linia.remove()
        for quin, color, estil_linia in (("a", estil.SIMULACIO, "-"), ("b", estil.FEIX, "-")):
            y = e[quin]
            self.eix_th.plot(t, _amb_forats(_plegat_graus(y[0])), color=color,
                             linewidth=1.0, linestyle=estil_linia)
            self.eix_th.plot(t, _amb_forats(_plegat_graus(y[1])), color=color,
                             linewidth=1.0, linestyle=":")
            E = e[f"energia_{quin}"]
            self.eix_e.plot(t, 1000 * (E[0] - E), color=color, linewidth=1.0)
        for txt in list(self.eix_d.texts):
            txt.remove()
        d = np.where(e["d"] > 0, e["d"], np.nan)
        self.eix_d.plot(t, d, color=estil.TINTA, linewidth=1.0)
        self.eix_d.axhline(np.radians(SOROLL_GRAUS), color=estil.REAL, linestyle="--",
                           linewidth=0.8)
        self.eix_d.text(0.99, np.radians(SOROLL_GRAUS), _coma(f" {SOROLL_GRAUS:g} graus: el soroll de seguiment"),
                        transform=self.eix_d.get_yaxis_transform(), ha="right", va="bottom",
                        fontsize=8, color=estil.REAL)
        if np.isfinite(d).any():
            self.eix_d.set_ylim(max(np.nanmin(d), 1e-16) / 3, max(np.nanmax(d), 1.0) * 3)
        for txt in list(self.eix_e.texts) + list(self.eix_th.texts):
            txt.remove()
        dis = e.get("dissipacio")
        self.eix_e.text(0.01, 0.95, "sense friccio: nomes deriva numerica" if dis is None
                        else "amb la friccio firmada: l'energia que es dissipa",
                        transform=self.eix_e.transAxes, va="top", fontsize=8, color=estil.TINTA_SUAU)
        self.eix_th.text(0.01, 0.95, "continu theta1, puntejat theta2",
                         transform=self.eix_th.transAxes, va="top", fontsize=8, color=estil.TINTA_SUAU)
        self.eix_e.relim()
        self.eix_e.autoscale_view()
        self.eix_e.set_xlim(0.0, float(t[-1]))
        # Els titols dels pendols diuen que es cada un, amb les seves tolerancies.
        tol_a, tol_b = e["tolerancies"]
        ent = e["entrada"]
        if ent["mode"] == "angle":
            d1, d2 = _coma(f"{ent['d_theta1']:g}"), _coma(f"{ent['d_theta2']:g}")
            clon = f"clon (+{d1} / +{d2} graus, tol. {tol_b})"
        else:
            clon = f"clon (mateix inici, tol. {tol_b})"
        self.titol_a.configure(text=f"original (tol. {tol_a})")
        self.titol_b.configure(text=clon)
        self.brac_b.set_label(clon)
        self.brac_a.set_label(f"original (tol. {tol_a})")
        self.eix_p.legend(loc="upper right", fontsize=8)
        friccio = ("sense friccio" if dis is None
                   else f"amb la friccio firmada (Dissipacio.firmada, C = {_coma(f'{dis.C:g}')} kg/m)")
        self.peu.configure(text=(
            "Els numeros de la memoria surten de les ordres de Python (pendol/ i eines/), no "
            f"d'aquesta finestra. Calculat amb la tolerancia {tol_a}"
            + (f" (i {tol_b} al clon)" if tol_b != tol_a else "")
            + f", {friccio}, a {ent['fps']:g} mostres per segon."))
        self.canvas_g.draw()
        self.canvas_p.draw()

    def _index(self, t: float) -> int:
        return int(min(max(round(t * self.e["entrada"]["fps"]), 0), len(self.e["t"]) - 1))

    def _mostra(self, t: float):
        """Posa-ho tot a l'instant t: pendols, cursors, barra i lectura."""
        e = self.e
        k = self._index(t)
        k0 = max(0, k - int(TRACA_S * e["entrada"]["fps"]))
        for quin, brac, traca in (("a", self.brac_a, self.traca_a), ("b", self.brac_b, self.traca_b)):
            x1, y1, x2, y2 = e[f"posicions_{quin}"]
            brac.set_data([0.0, x1[k], x2[k]], [0.0, y1[k], y2[k]])
            traca.set_data(x2[k0:k + 1], y2[k0:k + 1])
        lec = lectura(e, k)
        self.rellotge.set_text(lec["t"])
        for c in self.cursors:
            c.set_xdata([e["t"][k], e["t"][k]])
        self.blit_p.actualitza()
        self.blit_g.actualitza()
        for quin in ("a", "b"):
            for clau, text in lec[quin].items():
                self.etiquetes[f"{quin}_{clau}"].configure(text=text)
        for clau in ("sep_theta1", "sep_theta2", "d"):
            self.etiquetes[clau].configure(text=lec[clau])
        self.etiqueta_t.configure(text=lec["t"])
        self._movent_barra = True
        self.barra.set(float(e["t"][k]))
        self._movent_barra = False

    # --- reproduccio ---------------------------------------------------------

    def alterna(self):
        if self.jugant:
            self.jugant = False
        else:
            if self.t >= float(self.e["t"][-1]):
                self.t = 0.0
            self.jugant = True
        self.boto_juga.configure(text="Atura" if self.jugant else "Engega")

    def avanca(self):
        """Un fotograma endavant, si s'esta reproduint."""
        if not self.jugant or self.e is None:
            return
        self.t += float(self.var_velocitat.get().replace(",", ".")) / FPS_PANTALLA
        fi = float(self.e["t"][-1])
        if self.t >= fi:
            self.t = fi
            self.jugant = False
            self.boto_juga.configure(text="Engega")
        self._mostra(self.t)

    def ves_a(self, t: float):
        self.t = float(t)
        self._mostra(self.t)

    def _barra_moguda(self, valor):
        if self._movent_barra or self.e is None:
            return
        self.t = float(valor)
        self._mostra(self.t)

    def _tic(self):
        self.avanca()
        self._rellotge = self.arrel.after(int(1000 / FPS_PANTALLA), self._tic)

    def tanca(self):
        if self._rellotge is not None:
            self.arrel.after_cancel(self._rellotge)
            self._rellotge = None
        self.arrel.destroy()


# ---------------------------------------------------------------- l'autoprova


def autoprova() -> int:
    """El calcul, amb criteris escrits abans de mirar, i la finestra de debo.

    Criteris del pla del bloc D2 (25/09/2026):
      1. Clon d'angle amb pertorbacio zero: d(t) EXACTAMENT zero tota l'estona.
      2. Clon de 0,3 graus a theta1: d(0) = 0,3 graus en radians, a 1e-12,
         perque les velocitats inicials son zero.
      3. Amb la tolerancia rapida, l'energia no deriva mes d'un 1e-4 de
         g*(A+B) en 30 s.
      4. Dues tolerancies a 150 graus: d(0) = 0 i d < 1e-6 el primer segon
         (comencen junts), i d > 1e-3 als 30 s (s'han separat).
      5. La finestra s'obre, avanca 100 fotogrames, salta a la meitat, canvia
         de mode, torna a calcular i es tanca sense cap error, i la lectura
         ensenya el theta1 de la taula a l'instant on es. Si no hi ha pantalla,
         surt com a «no feta», no com a fallada.
    """
    p = parametres()
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    print("Autoprova de la finestra interactiva\n")
    print("1-3. El calcul, 30 s a 150 graus")
    e0 = escena(p, 150.0, 150.0, 0.0, 0.0, 30.0, "angle")
    mira(bool(np.all(e0["d"] == 0.0)), "clon amb pertorbacio zero: d(t) exactament zero")

    e1 = escena(p, 150.0, 150.0, SOROLL_GRAUS, 0.0, 30.0, "angle")
    mira(abs(e1["d"][0] - np.radians(SOROLL_GRAUS)) < 1e-12,
         f"clon de {SOROLL_GRAUS:g} graus: d(0) = {e1['d'][0]:.12f} rad")
    escala = p.g * (p.A + p.B)
    deriva = max(np.abs(e1["energia_a"] - e1["energia_a"][0]).max(),
                 np.abs(e1["energia_b"] - e1["energia_b"][0]).max())
    mira(deriva < 1e-4 * escala,
         f"l'energia deriva {1000*deriva:.2e} mJ en 30 s (sostre {1000*1e-4*escala:.2e} mJ)")

    print("\n4. Dues tolerancies a 150 graus")
    e2 = escena(p, 150.0, 150.0, 0.0, 0.0, 30.0, "tolerancies")
    primer = e2["t"] <= 1.0
    mira(e2["d"][0] == 0.0 and e2["d"][primer].max() < 1e-6,
         f"comencen junts: d(0) = 0 i d < 1e-6 el primer segon ({e2['d'][primer].max():.1e})")
    mira(e2["d"][-1] > 1e-3, f"als 30 s s'han separat: d = {e2['d'][-1]:.2e}")

    print("\n5. La finestra de debo")
    try:
        import tkinter as tk
        arrel = tk.Tk()
    except Exception as err:  # noqa: BLE001  sense pantalla
        print(f"      no feta: no s'ha pogut obrir cap finestra ({err})")
        arrel = None
    if arrel is not None:
        errors = []
        arrel.report_callback_exception = lambda tipus, valor, tb: errors.append(f"{tipus.__name__}: {valor}")
        try:
            f = Finestra(arrel, p, {"temps": 5.0})
            arrel.update()
            f.alterna()
            abans = time.perf_counter()
            for _ in range(100):
                f.avanca()
                arrel.update()
            per_fotograma = (time.perf_counter() - abans) / 100
            # S'atura abans de saltar: si no, el rellotge de la finestra hi
            # afegeix un fotograma mentre s'actualitza. I es compara amb
            # l'instant que ensenya DE DEBO, que es el que es vol provar.
            f.alterna()
            meitat = float(f.e["t"][-1]) / 2
            f.ves_a(meitat)
            arrel.update()
            k = f._index(f.t)
            esperat = lectura(f.e, k)["a"]["theta1"]
            mostrat = f.etiquetes["a_theta1"].cget("text")
            mira(mostrat == esperat, f"a la meitat la lectura diu theta1 = {mostrat.strip()}, "
                                     f"el de la taula")
            # B1 (2/11, documentacio/pla-interactiu-al-dia.md): amb la coma, i el numero de la taula.
            taula_k = float(f"{_plegat_graus(f.e['a'][0, k]):.2f}")
            mira("," in mostrat and float(mostrat.split()[0].replace(",", ".")) == taula_k,
                 f"B1: la lectura escriu la coma, i es el theta1 de la taula ({taula_k:.2f})")
            f.var_mode.set("tolerancies")
            f.calcula()
            arrel.update()
            mira(f.e["tolerancies"] == ("rapida", "estricta"), "canvia de mode i torna a calcular")
            f.tanca()
            print(f"      (informatiu: {1000*per_fotograma:.0f} ms per fotograma, i n'hi ha "
                  f"{1000/FPS_PANTALLA:.0f} de disponibles)")
        except Exception as err:  # noqa: BLE001
            errors.append(f"{type(err).__name__}: {err}")
            try:
                arrel.destroy()
            except Exception:  # noqa: BLE001
                pass
        mira(not errors, "la finestra s'obre, es mou i es tanca sense cap error"
             + (f": {errors}" if errors else ""))

    # 6 (29/09/2026, bloc D3 part 1). Criteris del pla, escrits abans de mirar
    # (documentacio/pla-segona-taula.md, punt E).
    import json
    import sys

    from pendol import dades

    print("\n6. Exportar, el pany i la llegenda (D3, sense finestra)")
    desti = ARREL / "esborranys" / "_autoprova_interactiu"
    desti.mkdir(parents=True, exist_ok=True)
    e = escena(p, 120.0, 120.0, SOROLL_GRAUS, 0.0, 5.0, "angle")
    cami_a, cami_b = exporta_escena(e, desti / "exportat.csv", p)
    # «A 1e-9»: el CSV canonic te deu xifres significatives, o sigui que es
    # llegeix relatiu al valor (i absolut per sota d'1).
    pitjor = 0.0
    for cami, y in ((cami_a, e["a"]), (cami_b, e["b"])):
        taula, _ = dades.carrega(cami)
        for col, v in zip(("t", "theta1", "theta2", "omega1", "omega2"), (e["t"], *y)):
            w = taula[col].to_numpy()
            pitjor = max(pitjor, float(np.max(np.abs(w - v) / np.maximum(1.0, np.abs(v)))))
    mira(pitjor < 1e-9, f"exportar i tornar a llegir: difereix {pitjor:.1e} (relatiu)")

    sys.path.insert(0, str(ARREL / "eines"))
    try:
        import balanc_energia
        balanc_energia.balanc(cami_a, balanc_energia.carrega(ARREL / "parametres.json"))
        entra = True
    except Exception as err:  # noqa: BLE001
        entra = False
        print(f"      {type(err).__name__}: {err}")
    mira(entra, "el CSV exportat entra a balanc_energia.balanc sense cap error")

    sense_firma = desti / "no_hi_es.json"
    firma_prova = desti / "firma_de_prova.json"
    firma_prova.write_text(json.dumps({"X": 2.0, "Y": 0.5, "Z": 0.2, "data": "prova"}),
                           encoding="utf-8")
    real = ARREL / "dades" / "net" / "150_3.csv"
    if real.exists():
        tancat, _ = pot_importar(real, sense_firma)
        obert_amb_firma, _ = pot_importar(real, firma_prova)
        mira(not tancat, "el pany TANCA una presa del Tracker sense firma")
        mira(obert_amb_firma, "amb una firma (de prova, a esborranys/) la deixa passar")
    else:
        print("      no feta la del Tracker: no hi ha dades/net/150_3.csv")
    obert_sim, _ = pot_importar(cami_a, sense_firma)
    mira(obert_sim, "el pany deixa passar una simulacio")

    ll = llegenda(dades.carrega(cami_a)[1], cami_a)
    buits = [k for k, v in ll.items() if v is None]
    mira(not buits, "la llegenda d'una simulacio exportada te tots els camps"
         + (f" (falten {buits})" if buits else ""))

    # 7 (2/11/2026). La friccio firmada. Criteris A2-A6 de
    # documentacio/pla-interactiu-al-dia.md, escrits abans del codi (l'A1 es
    # que tot el de sobre continua passant).
    print("\n7. La friccio firmada (150 graus, 30 s)")
    dis = Dissipacio.firmada(150.0, cami=ARREL / "parametres.json")
    ef = escena(p, 150.0, 150.0, SOROLL_GRAUS, 0.0, 30.0, "angle", friccio=True)
    rtol, atol = TOLERANCIES["rapida"]
    _, ya = simula(p, np.radians([150.0, 150.0, 0.0, 0.0]), temps=30.0, fps=FPS_CALCUL,
                   rtol=rtol, atol=atol, dis=dis)
    _, yb = simula(p, np.radians([150.0 + SOROLL_GRAUS, 150.0, 0.0, 0.0]), temps=30.0,
                   fps=FPS_CALCUL, rtol=rtol, atol=atol, dis=dis)
    mira(ef["dissipacio"] == dis and np.array_equal(ef["a"], ya) and np.array_equal(ef["b"], yb),
         "A2: amb la friccio, cada pendol es bit a bit simula(..., dis=Dissipacio.firmada(theta1))")

    pujada = max(float(np.max(E - np.minimum.accumulate(E)))
                 for E in (ef["energia_a"], ef["energia_b"]))
    mira(pujada < 1e-4 * escala,
         f"A3: l'energia no puja: com a molt {1000*pujada:.1e} mJ per sobre del seu minim "
         f"(sostre {1000*1e-4*escala:.2e} mJ)")
    print(f"      (informatiu: l'energia de l'original passa de {1000*ef['energia_a'][0]:.1f} mJ "
          f"a {1000*ef['energia_a'][-1]:.1f} mJ en 30 s)")

    def missatge(fer):
        try:
            fer()
        except ValueError as err:
            return str(err)
        return None

    esperat_120 = missatge(lambda: Dissipacio.firmada(120.0, cami=ARREL / "parametres.json"))
    rebut_120 = missatge(lambda: escena(p, 120.0, 120.0, SOROLL_GRAUS, 0.0, 5.0, "angle", friccio=True))
    mira(esperat_120 is not None and rebut_120 == esperat_120,
         f"A4: a 120 graus, escena s'atura amb el missatge de Dissipacio.firmada («{rebut_120}»)")
    plegat = missatge(lambda: escena(p, 150.0, 120.0, SOROLL_GRAUS, 0.0, 5.0, "angle", friccio=True))
    mira(plegat is not None and "bracos rectes" in plegat,
         f"A4b: amb theta2 a 30 graus de theta1, no es calcula («{plegat}»)")

    cami_f, _ = exporta_escena(ef, desti / "exportat_friccio.csv", p)
    meta_f = dades.carrega(cami_f)[1]
    mira(meta_f.get("dissipacio") == dis.com_a_dict() and llegenda(meta_f, cami_f)["friccio"] is True,
         "A5: exportat amb la friccio, el JSON porta la dissipacio firmada i la llegenda ho diu")

    try:
        import tkinter as tk
        arrel = tk.Tk()
    except Exception as err:  # noqa: BLE001  sense pantalla
        print(f"      A6 no feta: no s'ha pogut obrir cap finestra ({err})")
        arrel = None
    if arrel is not None:
        errors = []
        arrel.report_callback_exception = lambda tipus, valor, tb: errors.append(f"{tipus.__name__}: {valor}")
        try:
            f = Finestra(arrel, p, {"temps": 5.0, "friccio": True})
            arrel.update()
            mira(f.e["dissipacio"] == dis, "A6: amb la casella marcada, la finestra calcula amb la firmada")
            f.alterna()
            for _ in range(50):
                f.avanca()
                arrel.update()
            f.alterna()
            abans = f.e
            f.var["theta1"].set("120")
            f.var["theta2"].set("120")
            f.calcula()
            arrel.update()
            estat = f.etiqueta_estat.cget("text")
            mira(f.e is abans and esperat_120 in estat,
                 "A4: a la finestra, la linia d'estat ho diu i es queda l'escena d'abans")
            f.tanca()
        except Exception as err:  # noqa: BLE001
            errors.append(f"{type(err).__name__}: {err}")
            try:
                arrel.destroy()
            except Exception:  # noqa: BLE001
                pass
        mira(not errors, "A6: amb la friccio, la finestra s'obre, calcula, avanca i es tanca sense cap error"
             + (f": {errors}" if errors else ""))

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> el que la finestra ensenya surt de les funcions provades del model, i")
    print("     la finestra s'obre, es mou i es tanca. Limit: que es VEGI be -la")
    print("     llargada de les barres, que arrenquin junts- no ho pot dir cap prova.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--theta1", type=float, default=PER_DEFECTE["theta1"], help="graus")
    ap.add_argument("--theta2", type=float, default=PER_DEFECTE["theta2"], help="graus")
    ap.add_argument("--d-theta1", type=float, default=PER_DEFECTE["d_theta1"],
                    help="pertorbacio del clon a theta1 (graus)")
    ap.add_argument("--d-theta2", type=float, default=PER_DEFECTE["d_theta2"],
                    help="pertorbacio del clon a theta2 (graus)")
    ap.add_argument("--temps", type=float, default=PER_DEFECTE["temps"], help="segons")
    ap.add_argument("--mode", choices=tuple(MODES), default=PER_DEFECTE["mode"])
    ap.add_argument("--friccio", action="store_true", help="amb la friccio firmada")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()

    import tkinter as tk

    arrel = tk.Tk()
    Finestra(arrel, parametres(), {"theta1": args.theta1, "theta2": args.theta2,
                                   "d_theta1": args.d_theta1, "d_theta2": args.d_theta2,
                                   "temps": args.temps, "mode": args.mode,
                                   "friccio": args.friccio})
    arrel.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
