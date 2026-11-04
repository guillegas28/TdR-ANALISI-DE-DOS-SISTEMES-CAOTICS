"""Els botons d'importar i exportar de la finestra interactiva (bloc D3, part 2).

Es l'encarrec C2 (documentacio/encarrec-company-C2.md). No toca
pendol/interactiu.py: FinestraD3 hereta de Finestra i hi afegeix dos botons.

QUE HI HA A SOTA
----------------
Res de nou, com a la finestra. Exportar es exporta_escena(); importar passa
primer pel PANY (pot_importar) i la procedencia la diu llegenda(). Les tres
funcions son d'interactiu.py, ja provades. Aqui nomes s'hi afegeix:

  - importat(cami, firma): el calcul, sense finestra i provat sol, com escena().
    Passa pel pany, llegeix el fitxer amb dades.carrega() i posa el rellotge:
    una simulacio, tal com es (t des de 0); una presa real, amb
    t - meta["amollada"]["amollada_s"], perque el zero sigui l'amollada.
  - FinestraD3: els botons, i importa(cami) i exporta(cami), que es poden
    cridar sense dialeg. Els dialegs nomes les criden.

EL PANY
-------
No es toca. Sense dades/versemblant/firma.json amb X, Y i Z, una presa del
Tracker no entra: importar-la seria mirar la comparacio de l'experiment 1.
La firma es va escriure el 27/10, i des d'aleshores les preses hi entren.

LES PRESES REALS, CORREGIDES (2/11)
-----------------------------------
Una presa real es dibuixa amb els angles de comparar.corregeix_presa, l'unica
manera de corregir-la (amb els clics als 45 i als 90 graus, en dues parts als
150). Una que no es pot corregir (no es a dades/versemblant/centres_cercle.json)
no entra: sense corregir no s'ha de fer servir. El zero continua sent l'amollada
del video. El que s'hi veu NO es cap comparacio firmada.
documentacio/pla-interactiu-al-dia.md.

L'IMPORTAT ES MOU (2/11, la D')
-------------------------------
Cada fitxer importat es dibuixa tambe al dibuix del pendol, a l'instant que
ensenya el rellotge: la mostra mes propera (mostra_de), amb la seva traca. Una
presa real, en taronja i amb els angles corregits; una simulacio, en gris. Es
dibuixa amb les LLARGADES DEL BANC (fisica.posicions, amb L1 i r_gomet de
parametres.json), no amb els pixels del video. La simulacio de la finestra surt
dels angles que s'escriuen a dalt, no dels de la presa: si no son els mateixos,
se separen des del primer instant. El boto «Treu importats» els treu tots, per
poder-ne importar uns altres.

L'autoprova prova el pany amb una presa FABRICADA i una firma DE PROVA, totes
dues en un directori temporal, i la correccio amb tres preses de debo (nomes
les llegeix).

Us:
    .venv\\Scripts\\python -m pendol.interactiu_d3
    .venv\\Scripts\\python -m pendol.interactiu_d3 --theta1 120 --theta2 120 --temps 20
    .venv\\Scripts\\python -m pendol.interactiu_d3 --autoprova
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pendol import comparar, dades, estil
from pendol.fisica import posicions
from pendol.interactiu import (ARREL, FIRMA, MODES, PER_DEFECTE, SOROLL_GRAUS, TRACA_S, Finestra,
                               _amb_forats, _coma, _plegat_graus, escena, exporta_escena, llegenda,
                               parametres, pot_importar)

# ------------------------------------------------------------------ el calcul


def importat(cami, firma=FIRMA) -> tuple[dict | None, str]:
    """(el que es dibuixa d'un fitxer, per que), o (None, per que no).

    Primer el pany: si no deixa, no es llegeix res mes. El temps que torna,
    `t_dibuix`, es el que va a l'eix: el de la taula en una simulacio, i des de
    l'amollada en una presa real. Una presa real sense `amollada` al JSON no es
    dibuixa: no hi ha d'on treure el zero, i inventar-lo seria pitjor.

    Una presa real torna els angles de comparar.corregeix_presa (2/11), i
    `correccio` diu quina («amb clics» o «en dues parts»). Si no es pot corregir,
    no entra. Les omegues son les del CSV, sense corregir: no es dibuixen.
    """
    cami = Path(cami)
    pot, motiu = pot_importar(cami, firma)
    if not pot:
        return None, motiu
    taula, meta = dades.carrega(cami)
    t = taula["t"].to_numpy()
    theta1, theta2 = taula["theta1"].to_numpy(), taula["theta2"].to_numpy()
    if meta.get("origen") == "simulacio":
        desplacament = 0.0
        correccio = None
    else:
        amollada = (meta.get("amollada") or {}).get("amollada_s")
        if amollada is None:
            return None, "es una presa real sense amollada_s al JSON: no se sap on posar el zero"
        nom = cami.stem
        if nom not in json.loads(comparar.CENTRES.read_text(encoding="utf-8"))["preses"]:
            return None, (f"es una presa real que no es pot corregir ({nom} no es a "
                          f"{comparar.CENTRES.name}), i sense corregir no es fa servir")
        theta1, theta2, info = comparar.corregeix_presa(taula, meta, nom)
        correccio = "amb clics" if info["clics"] else "en dues parts"
        desplacament = float(amollada)
        motiu = f"{motiu}; corregida {correccio}"
    return {
        "cami": cami,
        "meta": meta,
        "real": meta.get("origen") != "simulacio",
        "t": t,
        "t_dibuix": t - desplacament,
        "desplacament_s": desplacament,
        "theta1": theta1,
        "theta2": theta2,
        "omega1": taula["omega1"].to_numpy(),
        "omega2": taula["omega2"].to_numpy(),
        "correccio": correccio,
        "llegenda": llegenda(meta, cami),
    }, motiu


def mostra_de(imp: dict, t: float) -> int:
    """La mostra d'un importat que es dibuixa a l'instant t de la finestra: la mes
    propera a t sobre el seu t_dibuix (si n'hi ha dues a la mateixa distancia, la
    primera). Despres del final, l'ultima: el pendol hi queda quiet."""
    td = imp["t_dibuix"]
    k = int(np.searchsorted(td, t))
    if k <= 0:
        return 0
    if k >= len(td):
        return len(td) - 1
    return k if td[k] - t < t - td[k - 1] else k - 1


def _text_llegenda(ll: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in ll.items())


# ---------------------------------------------------------------- la finestra


class FinestraD3(Finestra):
    """La finestra del D2, amb els botons «Exporta» i «Importa»."""

    def __init__(self, arrel_tk, p, valors: dict | None = None):
        # Abans del super: Finestra.__init__ ja crida calcula(), i aquest crida
        # _redibuixa_grafics(), que aqui vol la llista.
        self.importats = []
        super().__init__(arrel_tk, p, valors)
        tk, ttk = self.tk, self.ttk

        # Els botons van a la fila dels controls, al costat de «Calcula».
        # Finestra no la desa com a atribut: es busca pel boto.
        ctl = self._fila_de_controls()
        ttk.Button(ctl, text="Exporta", command=self._dialeg_exporta).pack(side="left", padx=(10, 2))
        ttk.Button(ctl, text="Importa", command=self._dialeg_importa).pack(side="left", padx=2)
        ttk.Button(ctl, text="Treu importats", command=self.treu_importats).pack(side="left", padx=2)

        # La llegenda dels fitxers importats, sota el peu.
        self.quadre_llegenda = tk.Text(ctl.master, height=6, wrap="none", font=("Consolas", 9),
                                       background=estil.SUPERFICIE, foreground=estil.TINTA_2,
                                       relief="flat")
        self.quadre_llegenda.pack(fill="x", pady=(4, 0))
        self._escriu_llegenda()

    def _fila_de_controls(self):
        pendents = [self.arrel]
        while pendents:
            w = pendents.pop()
            if isinstance(w, self.ttk.Button) and w.cget("text") == "Calcula":
                return w.master
            pendents.extend(w.winfo_children())
        raise RuntimeError("no trobo el boto «Calcula» de la Finestra")

    # --- sense dialeg --------------------------------------------------------

    def exporta(self, cami) -> tuple[Path, Path]:
        """L'escena de la pantalla, a `cami` i `<cami>_clon`."""
        return exporta_escena(self.e, cami, self.p)

    def importa(self, cami, firma=FIRMA) -> tuple[bool, str]:
        """Superposa theta1 i theta2 d'un fitxer, si el pany ho deixa."""
        imp, motiu = importat(cami, firma)
        if imp is None:
            self.etiqueta_estat.configure(text=f"No s'importa {Path(cami).name}: {motiu}")
            return False, motiu
        imp["posicions"] = posicions(imp["theta1"], imp["theta2"], self.p)
        self._munta_importat(imp)
        self.importats.append(imp)
        self.etiqueta_estat.configure(text=f"Importat {Path(cami).name} ({motiu})")
        self._redibuixa_grafics()
        self._mostra(self.t)
        self._escriu_llegenda()
        return True, motiu

    def treu_importats(self) -> int:
        """Treu tots els fitxers importats (corbes, pendols i llegendes), per poder-ne
        importar uns altres. L'original i el clon no es toquen. Torna quants n'ha tret."""
        n = len(self.importats)
        for imp in self.importats:
            for a in (imp["brac"], imp["traca"]):
                self.blit_p.artistes.remove(a)
                a.remove()
        self.importats = []
        if n == 0:
            self.etiqueta_estat.configure(text="No hi ha cap fitxer importat")
        else:
            self.etiqueta_estat.configure(text="Tret 1 fitxer importat" if n == 1
                                          else f"Trets {n} fitxers importats")
        if self.e is not None:
            self._redibuixa_grafics()
            self._mostra(self.t)
        self._escriu_llegenda()
        return n

    def _munta_importat(self, imp):
        """El brac i la traca de l'importat al dibuix del pendol, dins del blit."""
        if imp["real"]:
            color, trac = estil.REAL, estil.TRAC_REAL
            etiqueta = f"{imp['cami'].name} (corregida)"    # curta: si no, la llegenda tapa el rellotge
        else:
            color, trac = estil.TINTA_SUAU, "-"
            etiqueta = imp["cami"].name
        (traca,) = self.eix_p.plot([], [], color=color, alpha=0.5, linewidth=1.2)
        (brac,) = self.eix_p.plot([], [], color=color, linestyle=trac, linewidth=2.2, marker="o",
                                  markersize=6, markerfacecolor=color, markeredgecolor="none",
                                  label=etiqueta)
        for a in (traca, brac):
            a.set_animated(True)
            self.blit_p.artistes.append(a)
        imp["brac"], imp["traca"] = brac, traca

    def _mostra(self, t: float):
        # Els importats es posen a l'instant que ensenya el rellotge, abans que la
        # Finestra ho pinti tot amb el blit.
        if self.e is not None:
            tk = float(self.e["t"][self._index(t)])
            for imp in getattr(self, "importats", []):
                x1, y1, x2, y2 = imp["posicions"]
                k, k0 = mostra_de(imp, tk), mostra_de(imp, tk - TRACA_S)
                imp["brac"].set_data([0.0, x1[k], x2[k]], [0.0, y1[k], y2[k]])
                imp["traca"].set_data(x2[k0:k + 1], y2[k0:k + 1])
        super()._mostra(t)

    # --- els dialegs ---------------------------------------------------------

    def _dialeg_exporta(self):
        from tkinter import filedialog, messagebox

        cami = filedialog.asksaveasfilename(parent=self.arrel, title="Exporta l'escena",
                                            defaultextension=".csv",
                                            filetypes=[("CSV canonic", "*.csv")])
        if not cami:
            return
        a, b = self.exporta(cami)
        messagebox.showinfo("Exporta", f"Desat:\n{a}\n{b}", parent=self.arrel)

    def _dialeg_importa(self):
        from tkinter import filedialog, messagebox

        cami = filedialog.askopenfilename(parent=self.arrel, title="Importa una trajectoria",
                                          filetypes=[("CSV canonic", "*.csv")])
        if not cami:
            return
        try:
            ok, motiu = self.importa(cami)
        except Exception as err:  # noqa: BLE001  un CSV que no segueix el contracte
            messagebox.showerror("Importa", f"{type(err).__name__}: {err}", parent=self.arrel)
            return
        if not ok:
            messagebox.showwarning("Importa", f"No s'importa:\n{motiu}", parent=self.arrel)

    # --- el dibuix -----------------------------------------------------------

    def _redibuixa_grafics(self):
        # Finestra treu totes les linies dels eixos i torna a dibuixar l'escena:
        # els importats es tornen a posar a sobre cada vegada.
        super()._redibuixa_grafics()
        importats = getattr(self, "importats", [])
        for imp in importats:
            color = estil.REAL if imp["real"] else estil.SIMULACIO
            trac = estil.TRAC_REAL if imp["real"] else estil.TRAC_SIMULACIO
            desp = _coma(f"{imp['desplacament_s']:g}")
            nom = imp["cami"].name + (f" (corregida {imp['correccio']}, t - {desp} s)"
                                      if imp["real"] else "")
            self.eix_th.plot(imp["t_dibuix"], _amb_forats(_plegat_graus(imp["theta1"])),
                             color=color, linewidth=1.6, linestyle=trac,
                             label=f"{nom}: theta1")
            self.eix_th.plot(imp["t_dibuix"], _amb_forats(_plegat_graus(imp["theta2"])),
                             color=color, linewidth=1.6, linestyle=":", alpha=0.8,
                             label=f"{nom}: theta2")
        if importats:
            self.eix_th.legend(loc="lower right", fontsize=7)
        elif self.eix_th.get_legend() is not None:
            self.eix_th.get_legend().remove()
        self.canvas_g.draw()

    def _escriu_llegenda(self):
        q = self.quadre_llegenda
        q.configure(state="normal")
        q.delete("1.0", "end")
        if not self.importats:
            q.insert("end", "Cap fitxer importat. «Importa» passa pel pany (dades/versemblant/firma.json) i "
                            "corregeix les preses del Tracker amb comparar.corregeix_presa; les que no es "
                            "poden corregir no entren. No es cap comparacio firmada.")
        for imp in self.importats:
            correccio = (f"  correccio: comparar.corregeix_presa, {imp['correccio']}; el pendol, dibuixat amb "
                         f"les llargades del banc (la resta, del JSON de la conversio, sense corregir)\n"
                         if imp["correccio"] else "")
            q.insert("end", f"{imp['cami'].name}\n{correccio}{_text_llegenda(imp['llegenda'])}\n")
        q.configure(state="disabled")


# ---------------------------------------------------------------- l'autoprova


def _foto_de_dades() -> tuple[str, dict]:
    """El `git status` de dades/ i, a mes, mida i data de cada fitxer.

    La segona part hi es perque els CSV de dades/net/ son al .gitignore: el
    git status no veuria que s'han tocat.
    """
    import subprocess

    estat = subprocess.run(["git", "status", "--porcelain", "--ignored", "--", "dades/"],
                           cwd=ARREL, capture_output=True, text=True, check=True).stdout
    fitxers = {str(f.relative_to(ARREL)): (f.stat().st_size, f.stat().st_mtime_ns)
               for f in (ARREL / "dades").rglob("*") if f.is_file()}
    return estat, fitxers


def _arrodonit(v) -> np.ndarray:
    """El que surt d'escriure amb dades.desa (%.10g) i tornar a llegir."""
    return np.array([float(f"{x:.10g}") for x in np.asarray(v, dtype=float)])


def autoprova() -> int:
    """Criteris I1-I6 de documentacio/encarrec-company-C2.md, escrits abans del codi.

    I1 concretat ABANS d'executar, per decisio d'en Guillem: comparacio EXACTA
    (np.array_equal) contra l'escena arrodonida com l'arrodoneix dades.desa.

    2/11, les preses reals corregides (C1-C3 de documentacio/pla-interactiu-al-dia.md,
    escrits abans del codi). La I2 canvia, decidit abans de correr-la: la presa
    fabricada prova el pany i que una presa sense correccio no entra (C2); el zero a
    l'amollada es prova a la C1, amb preses de debo. Per la mateixa rao, la I3 i la
    part de finestra de la I2 passen a fer servir la 150_3 en comptes de la fabricada.
    """
    import subprocess
    import sys
    import tempfile
    import time

    p = parametres()
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}")

    print("Autoprova dels botons d'importar i exportar (D3)\n")
    abans = _foto_de_dades()

    with tempfile.TemporaryDirectory(prefix="autoprova_d3_") as tmp:
        tmp = Path(tmp)
        sense_firma = tmp / "no_hi_es" / "firma.json"    # no existeix mai
        firma_prova = tmp / "firma_de_prova.json"
        firma_prova.write_text(json.dumps({"X": 3.0, "Y": 2, "Z": 0.2, "data": "prova"}),
                               encoding="utf-8")

        # --- I1 ------------------------------------------------------------
        print("I1. Exportar i tornar a importar")
        e = escena(p, 120.0, 120.0, SOROLL_GRAUS, 0.0, 5.0, "angle")
        cami_a, cami_b = exporta_escena(e, tmp / "escena.csv", p)
        iguals = True
        for cami, y in ((cami_a, e["a"]), (cami_b, e["b"])):
            imp, motiu = importat(cami, sense_firma)
            if imp is None:
                iguals = False
                print(f"      {cami.name} no s'importa: {motiu}")
                continue
            for col, v in zip(("t", "theta1", "theta2", "omega1", "omega2"), (e["t"], *y)):
                if not np.array_equal(imp[col], _arrodonit(v)):
                    iguals = False
                    print(f"      {cami.name}, {col}: difereix fins a "
                          f"{np.max(np.abs(imp[col] - _arrodonit(v))):.1e}")
            iguals &= np.array_equal(imp["t_dibuix"], imp["t"])
        mira(iguals, "original i clon: els cinc arrays, np.array_equal amb l'escena arrodonida "
                     "a %.10g, i una simulacio es dibuixa amb el seu t")

        # --- I2 ------------------------------------------------------------
        print("\nI2. Una presa del Tracker FABRICADA (al temporal): el pany i la correccio")
        amollada_s = 0.6125
        t = np.arange(0.0, 3.0, 1 / 240)
        th = np.where(t < amollada_s, 1.0, np.cos(3.0 * (t - amollada_s)))
        presa = dades.desa(tmp / "presa_fabricada.csv", t, th, 1.1 * th, 0 * t, 0 * t, {
            "origen": "tracker",
            "angle_nominal_graus": 60,
            "fps": 240.0,
            "durada_s": float(t[-1]),
            "amollada": {"hi_ha_amollada": True, "amollada_s": amollada_s,
                         "angles_amollada_graus": [57.3, 63.0]},
            "geometria": {"rigidesa": {"soroll_angular_graus": 0.3}},
        })
        tancat, motiu_tancat = importat(presa, sense_firma)
        mira(tancat is None, f"sense firma, el pany la refusa ({motiu_tancat})")
        sense_corr, motiu_sense = importat(presa, firma_prova)
        mira(sense_corr is None and "no es pot corregir" in motiu_sense,
             f"C2: amb la firma de prova, no es pot corregir i no entra ({motiu_sense})")

        # --- C1 i C3 (2/11): tres preses de debo, nomes llegides --------------
        print("\nC1 i C3. Preses reals, corregides")
        correccions = {"45_6": "amb clics", "90_11": "amb clics", "150_3": "en dues parts"}
        reals = {nom: ARREL / "dades" / "net" / f"{nom}.csv" for nom in correccions}
        hi_son = all(c.exists() for c in reals.values())
        if not hi_son:
            print("      no feta: falten preses a dades/net/ (es refan amb eines/converteix_preses.py)")
        for nom, cami in (reals.items() if hi_son else ()):
            imp, motiu = importat(cami, firma_prova)
            taula, meta = dades.carrega(cami)
            th1, th2, _ = comparar.corregeix_presa(taula, meta, nom)
            amollada = float(meta["amollada"]["amollada_s"])
            mira(imp is not None and np.array_equal(imp["theta1"], th1) and np.array_equal(imp["theta2"], th2)
                 and np.array_equal(imp["t_dibuix"], taula["t"].to_numpy() - amollada),
                 f"C1: {nom}, els angles de corregeix_presa (np.array_equal) i t - {amollada:g} s")
            mira(imp is not None and imp["correccio"] == correccions[nom],
                 f"C3: {nom}, corregida {imp['correccio'] if imp else '-'} ({correccions[nom]})")

        # --- I3 ------------------------------------------------------------
        print("\nI3. La llegenda")
        for cami, firma in ((cami_a, sense_firma),) + (((reals["150_3"], firma_prova),) if hi_son else ()):
            imp, _ = importat(cami, firma)
            esperada = llegenda(dades.carrega(cami)[1], cami)
            mira(imp is not None and list(imp["llegenda"]) == list(esperada)
                 and imp["llegenda"] == esperada,
                 f"{cami.name}: totes les claus de llegenda() ({len(esperada)})")

        # --- I4 (i la part de finestra d'I2 i I3) ----------------------------
        print("\nI4. La finestra de debo")
        try:
            import tkinter as tk
            arrel = tk.Tk()
        except Exception as err:  # noqa: BLE001  sense pantalla
            print(f"      no feta: no s'ha pogut obrir cap finestra ({err})")
            arrel = None
        if arrel is not None:
            errors = []
            arrel.report_callback_exception = lambda tipus, valor, tb: errors.append(
                f"{tipus.__name__}: {valor}")
            try:
                f = FinestraD3(arrel, p, {"temps": 5.0})
                arrel.update()
                linies = len(f.eix_th.lines)
                exportats = f.exporta(tmp / "de_la_finestra.csv")
                ok, _ = f.importa(exportats[0], sense_firma)
                arrel.update()
                mira(ok and len(f.eix_th.lines) == linies + 2,
                     "importa l'escena que acaba d'exportar: dues linies mes a la figura dels angles")
                text = f.quadre_llegenda.get("1.0", "end")
                mira(all(f"  {k}:" in text for k in f.importats[0]["llegenda"]),
                     "I3: el quadre de text ensenya totes les claus de la llegenda")
                etiquetes = [t.get_text() for t in f.eix_th.get_legend().get_texts()]
                mira(any(exportats[0].name in t for t in etiquetes),
                     "te la seva entrada a la llegenda de la figura")
                # M2 (2/11, la D'): la simulacio importada es mou damunt de l'original.
                f.ves_a(float(f.e["t"][-1]) / 2)
                arrel.update()
                xa, ya = (np.asarray(v, dtype=float) for v in f.brac_a.get_data())
                xi, yi = (np.asarray(v, dtype=float) for v in f.importats[0]["brac"].get_data())
                dif = float(max(np.max(np.abs(xa - xi)), np.max(np.abs(ya - yi))))
                mira(dif <= 1e-9, f"M2: la simulacio importada es mou damunt de l'original "
                                  f"({dif:.1e} m a la meitat)")

                n = len(f.eix_th.lines)
                ok_tancat, _ = f.importa(presa, sense_firma)
                arrel.update()
                mira(not ok_tancat and len(f.eix_th.lines) == n and len(f.importats) == 1,
                     "I2: sense firma, la presa fabricada no dibuixa res")
                ok_sense, _ = f.importa(presa, firma_prova)
                arrel.update()
                mira(not ok_sense and len(f.eix_th.lines) == n and len(f.importats) == 1,
                     "C2: amb la firma de prova, la fabricada (no es pot corregir) no dibuixa res")
                if hi_son:
                    real = reals["150_3"]
                    ok_obert, _ = f.importa(real, firma_prova)
                    arrel.update()
                    taula_r, meta_r = dades.carrega(real)
                    th1_r, th2_r, _ = comparar.corregeix_presa(taula_r, meta_r, "150_3")
                    linia = f.eix_th.lines[-1]
                    mira(ok_obert
                         and np.array_equal(linia.get_xdata(), taula_r["t"].to_numpy()
                                            - float(meta_r["amollada"]["amollada_s"]))
                         and np.array_equal(linia.get_ydata(), _amb_forats(_plegat_graus(th2_r)),
                                            equal_nan=True),
                         "C1: a la finestra, la 150_3 es dibuixa corregida i desplacada per l'amollada")
                    text = f.quadre_llegenda.get("1.0", "end")
                    mira("correccio: comparar.corregeix_presa, en dues parts" in text,
                         "C3: el quadre de la llegenda diu la correccio de la 150_3")
                    # M1 i M3 (2/11, la D'): el brac de la presa, a l'instant que ensenya el
                    # rellotge, son les posicions dels angles corregits a la mostra mes propera.
                    imp_r = f.importats[-1]
                    f.ves_a(1.0)
                    arrel.update()
                    tk = float(f.e["t"][f._index(f.t)])
                    td = imp_r["t_dibuix"]
                    kk = int(np.argmin(np.abs(td - tk)))
                    x1, y1, x2, y2 = posicions(th1_r, th2_r, p)
                    xb, yb = (np.asarray(v, dtype=float) for v in imp_r["brac"].get_data())
                    mitja = 0.5 * float(np.median(np.diff(td)))
                    mira(abs(td[kk] - tk) <= mitja and np.array_equal(xb, [0.0, x1[kk], x2[kk]])
                         and np.array_equal(yb, [0.0, y1[kk], y2[kk]]),
                         f"M1: a t = {tk:g} s, el brac de la 150_3 es posicions() dels angles corregits "
                         f"a la mostra mes propera ({1000*abs(td[kk] - tk):.2f} ms)")
                    mira(imp_r["brac"].get_color() == estil.REAL and "corregida" in imp_r["brac"].get_label(),
                         f"M3: la presa real va en taronja i l'etiqueta diu «{imp_r['brac'].get_label()}»")

                f.var_mode.set("tolerancies")
                f.calcula()
                arrel.update()
                importats = 4 if hi_son else 2
                mira(len(f.eix_th.lines) == linies + importats,
                     "en tornar a calcular, els importats hi continuen")

                # T1 i T2 (2/11): «Treu importats», i tornar-ne a importar un.
                artistes = [a for imp in f.importats for a in (imp["brac"], imp["traca"])]
                tret = f.treu_importats()
                arrel.update()
                etiquetes_p = [t.get_text() for t in f.eix_p.get_legend().get_texts()]
                text = f.quadre_llegenda.get("1.0", "end")
                mira(tret == importats // 2 and not f.importats and len(f.eix_th.lines) == linies
                     and not any(a in f.eix_p.lines or a in f.blit_p.artistes for a in artistes)
                     and f.eix_th.get_legend() is None and not any(".csv" in t for t in etiquetes_p)
                     and text.startswith("Cap fitxer importat"),
                     f"T1: «Treu importats» en treu {tret} i la finestra queda com abans d'importar")
                if hi_son:
                    ok_45, _ = f.importa(reals["45_6"], firma_prova)
                    arrel.update()
                    imp45 = f.importats[-1]
                    f.ves_a(1.0)
                    arrel.update()
                    tk = float(f.e["t"][f._index(f.t)])
                    td = imp45["t_dibuix"]
                    kk = int(np.argmin(np.abs(td - tk)))
                    th1_45, th2_45, _ = comparar.corregeix_presa(*dades.carrega(reals["45_6"]), "45_6")
                    x1, y1, x2, y2 = posicions(th1_45, th2_45, p)
                    xb, yb = (np.asarray(v, dtype=float) for v in imp45["brac"].get_data())
                    mira(ok_45 and len(f.eix_th.lines) == linies + 2
                         and abs(td[kk] - tk) <= 0.5 * float(np.median(np.diff(td)))
                         and np.array_equal(xb, [0.0, x1[kk], x2[kk]]) and np.array_equal(yb, [0.0, y1[kk], y2[kk]]),
                         "T2: despres de treure'ls, la 45_6 s'importa i el seu pendol es mou com diu la M1")
                else:
                    print("      T2 no feta: falten preses a dades/net/")
                f.alterna()
                for _ in range(50):    # M4 (2/11): 50 fotogrames, amb els importats movent-se
                    f.avanca()
                    arrel.update()
                f.alterna()
                f.tanca()
            except Exception as err:  # noqa: BLE001
                errors.append(f"{type(err).__name__}: {err}")
                try:
                    arrel.destroy()
                except Exception:  # noqa: BLE001
                    pass
            mira(not errors, "la finestra s'obre, importa, exporta i es tanca sense cap error"
                 + (f": {errors}" if errors else ""))

    # --- I5 ----------------------------------------------------------------
    print("\nI5. dades/ no s'ha tocat")
    despres = _foto_de_dades()
    mira(despres[0] == abans[0], "el git status de dades/ es el mateix")
    mira(despres[1] == abans[1], f"els {len(abans[1])} fitxers de dades/, mateixa mida i data")

    # --- I6 ----------------------------------------------------------------
    print("\nI6. L'autoprova d'interactiu.py")
    inici = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", "pendol.interactiu", "--autoprova"],
                       cwd=ARREL, capture_output=True, text=True)
    ultima = (r.stdout.strip().splitlines() or [""])
    resum = next((l for l in ultima if l.startswith(("OK", "AUTOPROVA FALLADA"))), ultima[-1])
    mira(r.returncode == 0, f"python -m pendol.interactiu --autoprova: codi {r.returncode}, "
                            f"«{resum.strip()}» ({time.perf_counter() - inici:.0f} s)")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    if _foto_de_dades() != abans:
        problemes.append("I5 (despres d'I6): dades/ ha canviat")
        print("   NO  despres d'I6, dades/ ha canviat")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> exportar i importar tornen el mateix, el pany tanca les preses reals sense")
    print("     firma, les preses reals entren corregides (i les que no es poden corregir, no),")
    print("     la llegenda es completa i la finestra s'obre i es tanca. Limit: que es VEGI be")
    print("     ho ha de mirar en Guillem.")
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
    FinestraD3(arrel, parametres(), {"theta1": args.theta1, "theta2": args.theta2,
                                     "d_theta1": args.d_theta1, "d_theta2": args.d_theta2,
                                     "temps": args.temps, "mode": args.mode,
                                     "friccio": args.friccio})
    arrel.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
