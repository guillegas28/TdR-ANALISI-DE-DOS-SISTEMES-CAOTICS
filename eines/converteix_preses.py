"""Converteix TOTES les preses caotiques del .trk al format canonic, de cop.

Per que existeix aquesta eina. Els CSV de les preses es van fer un a un, a ma,
i van anar a parar a esborranys/. Aixo vol dir que els numeros del balanc
d'energia que ja son escrits a la memoria del treball sortien d'uns fitxers que
el repositori NO podia tornar a generar. Amb aquesta eina, si; i la comprovacio
de debo no es que l'eina funcioni, es que eines/balanc_energia.py doni sobre la
seva sortida els MATEIXOS numeros que ja hi ha escrits.

No fa cap fisica nova: nomes recorre els .trk i crida pendol.tracker.processa(),
que es qui resol la vertical, el pivot i els angles. Aixi pendol/tracker.py
continua sent l'eina d'UNA presa i aixo n'es una capa a sobre.

QUINS .trk AGAFA. Nomes els de dades/video/tracker/<angle>/, amb <angle> un
numero: 45/, 90/ i 150/. Els .trk que pengen directament de tracker/
-Friccio_doble_1.1, Friccio_doble_2, IMG_3072, doble_90 de prova- queden fora a
posta: son preses de decaiment i de prova, tenen el seu circuit propi, i barrejar
-les aqui seria dir que son tirs caotics quan no ho son.

UNA PRESA REBUTJADA NO ES UN ERROR. Si la cua d'una presa no s'assossega, sense
repos no hi ha vertical i tracker.py la rebutja sol. El lot ho recull, li escriu
el motiu al seu .log, la marca a l'index i continua amb les altres. El codi de
sortida es 0: rebutjar-la amb el motiu escrit ES el funcionament correcte.

Del 06/09/2026 aquest cas no es dona a cap de les 16 preses. Abans si: la 90_11
es rebutjava per no aturar-se, i resulta que el que passava es que la
interseccio de les tres pistes li escapcava els 27 segons finals, que son
justament els que s'aturava. Vegeu pendol/trk.py, tres_pistes.

QUE EN SURT, a dades/net/:

    <nom>.csv     el format canonic (t,theta1,theta2,omega1,omega2)
    <nom>.json    les metadades, amb l'angle nominal que diu la carpeta
    <nom>.log     tot el que la conversio va escriure per pantalla
    preses.json   l'index de tot el lot

L'index no porta cap marca de temps a posta: dues passades sobre els mateixos
.trk han de donar un fitxer identic, i aixi Git nomes el veu canviar quan hi ha
alguna cosa que ha canviat de debo.

Us:
    .venv\\Scripts\\python eines\\converteix_preses.py
    .venv\\Scripts\\python eines\\converteix_preses.py --refes
    .venv\\Scripts\\python eines\\converteix_preses.py --nomes 150_3 90_1
    .venv\\Scripts\\python eines\\converteix_preses.py --sortida esborranys/proves
    .venv\\Scripts\\python eines\\converteix_preses.py --autoprova

Codi de sortida 1 nomes si cap presa peta per una rao que no sigui el rebuig
documentat, o si l'autoprova falla.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
from pathlib import Path

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import amollada, dades, tracker  # noqa: E402

TRACKER = "dades/video/tracker"
DESTI = "dades/net"

# ELS RETALLS, presa a presa: on comenca una presa quan el principi del
# marcatge no es repos de debo. Es decideix en aquest fitxer versionat, amb el
# motiu, i no tocant els .trk (vegeu pendol/trk.py, retalla). No va a
# dades/video/tracker/ perque alli el .gitignore nomes deixa passar .trk.
RETALLS = "dades/retalls.json"


def carrega_retalls(cami) -> dict:
    """{nom: {"des_del_fotograma": N, "motiu": ...}}. Les claus amb _ son comentaris."""
    cami = Path(cami)
    if not cami.exists():
        return {}
    tot = json.loads(cami.read_text(encoding="utf-8"))
    return {nom: v for nom, v in tot.items() if not nom.startswith("_")}

# SI UNA PRESA ARRENCA DEL REPOS no es decideix aqui des del 24/09/2026. Fins
# llavors hi havia un llindar de 0,05 s sobre la quietud que donava el
# detector de pixels de pendol/tracker.py, i aquell detector es disparava amb
# una sola sacsejada del seguidor: la 90_5, la 150_8 i la 45_13 van sortir en
# marxa sense ser-hi. Ara ho decideix l'amollada, amb un criteri que viu en un
# sol lloc -amollada.arrenca_del_repos- i que fa servir tambe el balanc
# d'energia. La quietud del detector de pixels continua a l'index, com a
# diagnostic.


# ------------------------------------------------------------- trobar les preses


def _clau(cami: Path) -> tuple:
    """Ordena 150_9 abans que 150_10, que es el que un espera i el sort no fa."""
    m = re.match(r"^(\d+)_(\d+)$", cami.stem)
    if m:
        return (int(m.group(1)), int(m.group(2)), cami.stem)
    return (10**9, 0, cami.stem)


def troba_preses(arrel: Path) -> list[tuple[Path, float]]:
    """Els .trk de les carpetes amb nom numeric. Torna (cami, angle nominal)."""
    trobades = []
    if not arrel.is_dir():
        return trobades
    for carpeta in sorted(arrel.iterdir()):
        if not carpeta.is_dir() or not carpeta.name.isdigit():
            continue
        angle = float(carpeta.name)
        for trk in sorted(carpeta.glob("*.trk"), key=_clau):
            trobades.append((trk, angle))
    return sorted(trobades, key=lambda p: _clau(p[0]))


# ----------------------------------------------------------------- la conversio


def _fitxa(nom, angle, cami_trk, estat, meta=None, motiu=None) -> dict:
    """Una fila de l'index: el que se sap d'aquesta presa, vingui d'on vingui."""
    fila = {
        "nom": nom,
        "grup_graus": angle,
        "fitxer_trk": cami_trk,
        "estat": estat,
    }
    if motiu:
        fila["motiu"] = motiu
    if not meta:
        return fila

    geo = meta.get("geometria", {})
    rig = geo.get("rigidesa", {})
    quiet = geo.get("quiet_al_principi_s")
    amo = meta.get("amollada") or {}
    fila.update({
        "durada_s": meta.get("durada_s"),
        "n_mostres": meta.get("n_mostres"),
        "fps": meta.get("fps"),
        "angles_inicials_graus": meta.get("angles_inicials_graus"),
        "vertical_graus": geo.get("gir_graus"),
        "estabilitat_graus": geo.get("estabilitat_graus"),
        "residu_px": geo.get("residu_px"),
        "articulacio_delta_px": geo.get("delta_px"),
        "soroll_angular_graus": rig.get("soroll_angular_graus"),
        # La quietud del detector de pixels: DIAGNOSTIC, ja no classifica res.
        "quiet_al_principi_s": quiet,
        "velocitat_inicial_px_s": geo.get("velocitat_inicial_px_s"),
        # Aquesta es la clau que separa les preses que serveixen per al balanc
        # d'energia (l'energia alliberada surt de la diferencia d'altura, sense
        # derivar res) de les que arrenquen ja en marxa i necessitarien el terme
        # cinetic. Deixar-ho escrit val mes que saber-ho. La decideix l'amollada.
        "arrenca_del_repos": bool(amollada.arrenca_del_repos(meta)),
    })
    # Nomes si n'hi ha, perque l'index de les preses sense retall no canvii.
    if meta.get("retall_des_del_fotograma") is not None:
        fila["retall_des_del_fotograma"] = meta["retall_des_del_fotograma"]
    if meta.get("fora_del_balanc"):
        fila["fora_del_balanc"] = meta["fora_del_balanc"]
    # On era la marca de l'articulacio respecte de l'eix del pivot 2, i si la
    # conversio l'hi ha tornada (vegeu tracker.mesura_marca_articulacio).
    marca = geo.get("marca_articulacio") or {}
    if marca.get("mesurada"):
        fila.update({
            "marca_delta_mm": marca.get("delta_mm"),
            "marca_error_theta2_graus": marca["error_theta2_maxim_graus"],
            "marca_a_leix": marca.get("a_leix"),
            "marca_corregida": bool(marca.get("corregida")),
            "rao_gomet_L1": marca["rao_gomet_L1"],
            "rao_gomet_L1_a_leix": marca["rao_gomet_L1_a_leix"],
        })
    if amo.get("hi_ha_amollada"):
        fila.update({
            "amollada_s": amo["amollada_s"],
            "amollada_incertesa_s": amo["incertesa_s"],
            "repos_abans_s": amo["repos_abans_s"],
            "angles_amollada_graus": amo["angles_amollada_graus"],
        })
    else:
        fila["motiu_sense_amollada"] = amo.get(
            "motiu", "les metadades no porten el bloc de l'amollada")
    return fila


def converteix(cami_trk: Path, angle: float, desti: Path, cua: float, refes: bool,
               retall: dict | None = None) -> dict:
    """Una presa. No aixeca mai: els problemes tornen dins de la fitxa."""
    nom = cami_trk.stem
    rel_trk = cami_trk.as_posix()
    cami_csv = desti / f"{nom}.csv"
    cami_json = dades.cami_meta(cami_csv)
    cami_log = desti / f"{nom}.log"
    des_del = (retall or {}).get("des_del_fotograma")
    fora_balanc = (retall or {}).get("fora_del_balanc")

    if not refes and cami_csv.exists() and cami_json.exists() \
            and cami_csv.stat().st_mtime >= cami_trk.stat().st_mtime:
        meta = json.loads(cami_json.read_text(encoding="utf-8"))
        # Si el retall o la marca han canviat, la presa s'ha de refer encara
        # que el .trk no.
        if meta.get("retall_des_del_fotograma") == des_del and meta.get("fora_del_balanc") == fora_balanc:
            fila = _fitxa(nom, angle, rel_trk, "convertida", meta)
            fila["saltada"] = True
            return fila
    extra = {"angle_nominal_graus": angle}
    if des_del is not None:
        extra["retall_motiu"] = retall.get("motiu")
    if fora_balanc:
        # La presa es converteix igual, pero eines/balanc_energia.py no la fa servir.
        extra["fora_del_balanc"] = fora_balanc

    desti.mkdir(parents=True, exist_ok=True)
    pantalla = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(pantalla):
            tracker.processa(
                rel_trk,
                cami_csv,
                meta_extra=extra,
                cua_s=cua,
                des_del_fotograma=des_del,
            )
    except SystemExit as e:
        # El rebuig previst: tracker.py diu que sense repos no hi ha vertical.
        error = str(e)
    except Exception as e:  # noqa: BLE001  qualsevol altra cosa tampoc no ha de fer caure el lot
        error = f"{type(e).__name__}: {e}"

    text = pantalla.getvalue()
    if error:
        text = f"{error}\n{text}" if text else f"{error}\n"
    cami_log.write_text(text, encoding="utf-8")

    if error:
        # Si hi havia sortida d'una passada anterior, fora: dades/net/ no ha de
        # contenir una presa que avui es rebutja.
        for vell in (cami_csv, cami_json):
            if vell.exists():
                vell.unlink()
                print(f"  (esborrat el {vell.name} d'una passada anterior)")
        return _fitxa(nom, angle, rel_trk, "rebutjada", motiu=error.strip())

    meta = json.loads(cami_json.read_text(encoding="utf-8"))
    return _fitxa(nom, angle, rel_trk, "convertida", meta)


def executa_lot(arrel: Path, desti: Path, cua: float, refes: bool,
                nomes: list[str] | None = None, retalls: dict | None = None) -> list[dict]:
    preses = troba_preses(arrel)
    if nomes:
        vols = set(nomes)
        preses = [p for p in preses if p[0].stem in vols]
    retalls = retalls or {}
    files = []
    for cami_trk, angle in preses:
        fila = converteix(cami_trk, angle, desti, cua, refes, retalls.get(cami_trk.stem))
        marca = {"convertida": "ok", "rebutjada": "REBUTJADA"}[fila["estat"]]
        if fila.get("saltada"):
            marca = "ja hi era"
        print(f"  {fila['nom']:<8} {marca}")
        files.append(fila)
    return files


# -------------------------------------------------------------------- la taula


def imprimeix(files: list[dict]) -> None:
    if not files:
        print("Cap presa.")
        return

    print(f"\n{'presa':<8} {'grup':>5} {'durada':>8} {'th1,th2 inicials':>18} "
          f"{'vertical':>9} {'soroll':>13}  {'arrencada':<15}marca")
    print("-" * 100)
    for f in files:
        if f["estat"] != "convertida":
            print(f"{f['nom']:<8} {f['grup_graus']:>4.0f}  {'REBUTJADA':>8}  "
                  f"{f.get('motiu', '').splitlines()[0][:50]}")
            continue
        a1, a2 = f["angles_inicials_graus"]
        s1, s2 = f["soroll_angular_graus"]
        if f.get("marca_delta_mm") is None:
            marca = "-"
        else:
            veredicte = {True: "a l'eix", False: "FORA", None: "no se sap"}[f["marca_a_leix"]]
            marca = f"{f['marca_delta_mm']:+.1f} mm {veredicte}"
            if f.get("marca_corregida"):
                marca += " -> corregida"
        print(f"{f['nom']:<8} {f['grup_graus']:>4.0f}  {f['durada_s']:>6.1f} s  "
              f"{a1:>7.1f} {a2:>7.1f}  {f['vertical_graus']:>+8.3f}  "
              f"{s1:>5.2f} {s2:>5.2f}   "
              f"{'del repos' if f['arrenca_del_repos'] else 'SENSE AMOLLADA':<15}{marca}")

    bones = [f for f in files if f["estat"] == "convertida"]
    repos = [f for f in bones if f["arrenca_del_repos"]]
    dolentes = len(files) - len(bones)
    print(f"\n{len(bones)} convertides, {dolentes} "
          f"{'rebutjada' if dolentes == 1 else 'rebutjades'}.")
    sense = [f for f in bones if not f["arrenca_del_repos"]]
    print(f"{len(repos)} arrenquen del repos i serveixen per al balanc d'energia"
          + (f"; {len(sense)} no, perque no s'hi troba l'amollada:" if sense
             else ": a totes s'hi troba l'amollada."))
    for f in sense:
        print(f"  {f['nom']}: {f.get('motiu_sense_amollada', '')}")


# ----------------------------------------------------------------- l'autoprova


def autoprova() -> int:
    """Dues preses de debo a un lloc a part, i una entrada dolenta inventada.

    La 90_11 hi es a posta i no per casualitat: es la presa que la interseccio
    de les tres pistes escapcava 27 segons -els del final, on el pendol s'atura-
    i que per aixo es rebutjava per no aturar-se. Si algun dia algu torna a fer
    la interseccio amb el fix inclos, la seva durada cau de 62 s a 35 i aquesta
    prova ho canta. Vegeu trk.tres_pistes.
    """
    desti = Path("esborranys/_autoprova_preses")
    arrel = Path(TRACKER)
    volem = ["90_1", "90_11"]
    hi_son = {c.stem for c, _ in troba_preses(arrel)}
    if not set(volem) <= hi_son:
        print(f"No hi ha {volem} a {arrel}: no puc provar res. "
              "L'autoprova no ha fallat, no s'ha pogut fer.")
        return 0

    print(f"Convertint {volem} cap a {desti}...")
    files = executa_lot(arrel, desti, tracker.CUA_PER_DEFECTE, refes=True, nomes=volem)

    problemes = []
    per_nom = {f["nom"]: f for f in files}
    if len(files) != 2:
        problemes.append(f"esperava 2 preses i n'han sortit {len(files)}")

    for nom, durada_min in (("90_1", 40.0), ("90_11", 60.0)):
        fila = per_nom.get(nom, {})
        if fila.get("estat") != "convertida":
            problemes.append(f"la {nom} havia de convertir-se: {fila.get('motiu', 'no hi es')}")
            continue
        try:
            taula, meta = dades.carrega(desti / f"{nom}.csv")
        except Exception as e:  # noqa: BLE001
            problemes.append(f"la {nom} no es pot rellegir: {e}")
            continue
        if list(taula.columns) != dades.COLUMNES:
            problemes.append(f"{nom}: columnes fora de contracte, {list(taula.columns)}")
        if meta.get("angle_nominal_graus") != 90.0:
            problemes.append(f"{nom}: angle nominal {meta.get('angle_nominal_graus')}, esperava 90")
        if meta.get("durada_s", 0) < durada_min:
            problemes.append(
                f"{nom}: nomes {meta.get('durada_s'):.1f} s i n'hi ha d'haver mes de "
                f"{durada_min:.0f}. Alguna pista esta escapcant la presa."
            )
        if not (desti / f"{nom}.log").exists():
            problemes.append(f"falta el log de {nom}")

        # L'AMOLLADA: hi ha de ser, ha de dir que la presa arrenca del repos, i
        # el que hi ha desat ha de ser EXACTAMENT el que surt de tornar-la a
        # mesurar sobre el CSV, que es la promesa de pendol/tracker.py. Es
        # compara el text JSON i no els diccionaris perque un nan no es igual
        # a ell mateix.
        if "amollada" not in meta:
            problemes.append(f"{nom}: les metadades no porten el bloc de l'amollada")
            continue
        if not fila.get("arrenca_del_repos"):
            problemes.append(f"{nom}: surt sense amollada ({fila.get('motiu_sense_amollada')}) "
                             "i les dues preses de la prova arrenquen del repos")
        marca = meta.get("geometria", {}).get("marca_articulacio") or {}
        if not marca.get("mesurada") or "marca_delta_mm" not in fila:
            problemes.append(f"{nom}: falta la mesura de la marca de l'articulacio "
                             f"({marca.get('motiu', 'no hi es')})")
        # La 90_1 te la marca uns 5 mm fora de l'eix: s'ha de corregir, i un
        # cop corregida, tornada a mesurar, ha de quedar a l'eix.
        elif nom == "90_1" and not (marca.get("corregida")
                                    and abs(marca.get("delta_despres_px", 99.0)) < 0.3):
            problemes.append(f"90_1: la marca no queda corregida a l'eix "
                             f"({marca.get('motiu_sense_correccio')}, "
                             f"delta despres {marca.get('delta_despres_px')})")
        refeta = amollada.troba_de_taula(taula, meta)
        if json.dumps(refeta, sort_keys=True) != json.dumps(meta["amollada"], sort_keys=True):
            problemes.append(f"{nom}: l'amollada desada no es la que dona el CSV rellegit")
        else:
            print(f"  {nom}: amollada a {meta['amollada']['amollada_s']:.4f} s, identica "
                  f"a la que es torna a mesurar sobre el CSV")

    # I que una entrada dolenta no faci caure el lot: es recull, es documenta i
    # es continua. Amb un fitxer inventat, que no embruta cap carpeta de dades.
    fila = converteix(Path("dades/video/tracker/90/aquesta_no_existeix.trk"),
                      90.0, desti, tracker.CUA_PER_DEFECTE, refes=True)
    if fila["estat"] != "rebutjada":
        problemes.append("un .trk inexistent s'havia de recollir com a rebutjat")
    if not (desti / "aquesta_no_existeix.log").exists():
        problemes.append("d'una entrada dolenta n'ha de quedar el log amb el motiu")

    if problemes:
        print("\nAUTOPROVA FALLADA:")
        for p in problemes:
            print(f"  - {p}")
        return 1
    print("\nOK: les dues preses es converteixen senceres i es poden rellegir, porten "
          "l'amollada\n    que dona el seu propi CSV, i una entrada dolenta queda "
          "documentada sense fer caure el lot.")
    return 0


# ----------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trk", default=TRACKER, help=f"arrel dels .trk (per defecte {TRACKER})")
    ap.add_argument("--sortida", default=DESTI, help=f"on van els CSV (per defecte {DESTI})")
    ap.add_argument("--cua", type=float, default=tracker.CUA_PER_DEFECTE,
                    help="segons de final de video que es promitgen per treure la vertical")
    ap.add_argument("--refes", action="store_true",
                    help="torna a fer-les totes encara que el CSV ja sigui mes nou que el .trk")
    ap.add_argument("--nomes", nargs="+", metavar="NOM", help="nomes aquestes preses, p.ex. 150_3")
    ap.add_argument("--retalls", default=RETALLS,
                    help=f"on comenca cada presa que no comenca al principi (per defecte {RETALLS})")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    # Els camins de dins de les metadades han de sortir relatius al projecte i
    # iguals des d'on sigui que es cridi l'eina, i tracker.processa() desa el
    # cami tal com l'hi donem.
    os.chdir(ARREL)

    if args.autoprova:
        return autoprova()

    arrel, desti = Path(args.trk), Path(args.sortida)
    preses = troba_preses(arrel)
    if not preses:
        raise SystemExit(
            f"No he trobat cap .trk a {arrel}/<angle>/. Recorda que nomes miro les "
            "carpetes amb nom numeric (45, 90, 150)."
        )

    retalls = carrega_retalls(args.retalls)
    print(f"{len(preses)} preses a {arrel}, cap a {desti}"
          + (f" (retalls a {args.retalls}: {', '.join(retalls)})" if retalls else "") + ":")
    files = executa_lot(arrel, desti, args.cua, args.refes, args.nomes, retalls)
    imprimeix(files)

    if args.nomes:
        # L'index descriu el lot SENCER. Reescriure'l amb dues preses seria
        # dir que el conjunt son dues preses, i no ho es.
        print("\n(amb --nomes no toco preses.json: descriu el lot sencer)")
        return 0

    index = desti / "preses.json"
    index.write_text(
        json.dumps({"ordre": "eines/converteix_preses.py", "cua_s": args.cua,
                    "preses": [{k: v for k, v in f.items() if k != "saltada"} for f in files]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nIndex a {index.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
