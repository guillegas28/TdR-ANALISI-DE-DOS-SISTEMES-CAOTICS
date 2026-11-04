"""Verifica que una conversio de video no ha perdut ni duplicat fotogrames.

Els videos del pendol doble passen de l'HEVC de l'iPhone a H.264 amb ffmpeg
abans d'entrar al Tracker, imposant frequencia constant amb la frequencia que
declara el fitxer (documentacio/convertir-el-video.html). Funciona quasi
sempre, i te una trampa que va costar de trobar:

  Un fitxer pot portar fotogrames de PREESCALFAMENT amb marca de temps
  negativa, que una llista d'edicio del contenidor marca com a no
  reproduibles. L'ffprobe els compta per fer la mitjana de frequencia i li
  surt mes baixa del que toca. Amb aquell -r massa baix, l'ffmpeg va
  DESCARTANT fotogrames repartits per tot el clip i no ho diu enlloc: les
  marques de temps queden impecables i el pendol fa un salt doble cada dos
  segons. Al 45_1 del 30/08/2026 eren 31 fotogrames a 120 fps ocupant els
  0,254 s d'abans del clip, i van costar 31 fotogrames de dins.

La comprovacio de debo no es comptar fotogrames: es fer que l'ffmpeg s'acusi
ell mateix. Es torna a aplicar a l'ORIGINAL la mateixa decisio de temps que va
prendre la conversio -amb la frequencia que declara el fitxer ja convertit- i
es llegeixen els comptadors dup= i drop=. Amb "-f null -" descodifica pero no
codifica: uns 25 s per fitxer en comptes de 8 minuts.

Per que no val comptar i prou: els originals de l'iPhone declaren a nb_frames
un fotograma MES del que descodifiquen (la llista d'edicio en talla un al
final). O sigui que una diferencia de -1 no vol dir res, i una de 0 tampoc no
garanteix res: 30 descartats i 29 duplicats tambe sumen -1.

Veredictes:
    OK      el convertit te els fotogrames que toca i no se n'ha descartat cap
    OK*     igual, pero amb fotogrames duplicats; es diu on son
    REFER   s'han descartat fotogrames, o el compte no quadra amb la recepta

Us:
    .venv\\Scripts\\python eines\\verifica_conversio.py dades/video/doble/45_nou dades/video/conv/45
    .venv\\Scripts\\python eines\\verifica_conversio.py <orig> <conv> --rapid
    .venv\\Scripts\\python eines\\verifica_conversio.py <orig> <conv> --estricte
    .venv\\Scripts\\python eines\\verifica_conversio.py <orig> <conv> --json esborranys/v45.json
    .venv\\Scripts\\python eines\\verifica_conversio.py --autoprova

Codi de sortida 1 si hi ha cap REFER, o si l'autoprova falla.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

FFPROBE = "ffprobe"
FFMPEG = "ffmpeg"

EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv")

# Per sota d'aixo la frequencia declarada i la real es consideren la mateixa.
# El 45_1 fallava per 1,7e-3, o sigui amb tres vegades de marge.
TOLERANCIA_FREQUENCIA = 5e-4

# Quants paquets es llegeixen del principi per buscar preescalfament.
PAQUETS_DE_CAP = 200


# ---------------------------------------------------------------- utilitats


def executa(ordre: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ordre, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )


def cal_ffmpeg() -> None:
    for nom in (FFPROBE, FFMPEG):
        if shutil.which(nom) is None:
            raise SystemExit(
                f"No trobo {nom} al PATH. Instal.la'l amb:  winget install Gyan.FFmpeg"
            )


def fraccio(text: str | None) -> Fraction | None:
    """'21336000/88937' -> Fraction. Torna None si no diu res util."""
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    try:
        if int(den) == 0:
            return None
        return Fraction(int(num), int(den))
    except ValueError:
        return None


def sonda(cami: Path) -> dict:
    """Metadades del primer flux de video. Instantani: no llegeix el fitxer."""
    ordre = [
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,r_frame_rate,time_base,width,height,"
        "codec_name,duration:format=duration",
        "-of", "json", str(cami),
    ]
    r = executa(ordre)
    if r.returncode != 0:
        return {"error": r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "ffprobe ha fallat"}
    dades = json.loads(r.stdout)
    if not dades.get("streams"):
        return {"error": "no hi ha cap flux de video"}
    s = dades["streams"][0]
    durada = s.get("duration") or dades.get("format", {}).get("duration")
    base = fraccio(s.get("time_base"))
    return {
        "nb_frames": int(s["nb_frames"]) if str(s.get("nb_frames", "")).isdigit() else None,
        "frequencia": fraccio(s.get("avg_frame_rate")),
        "base_de_temps": base,
        "durada": float(durada) if durada else None,
        "mida": f"{s.get('width')}x{s.get('height')}",
        "codec": s.get("codec_name"),
    }


def marques_de_temps(cami: Path, interval: str | None = None) -> list[int]:
    """Marques de temps dels paquets, ORDENADES.

    Venen en ordre de descodificacio, no de presentacio: amb fotogrames B
    surten desordenades i s'han d'ordenar sempre.
    """
    ordre = [FFPROBE, "-v", "error", "-select_streams", "v:0"]
    if interval:
        ordre += ["-read_intervals", interval]
    ordre += ["-show_entries", "packet=pts", "-of", "csv=p=0", str(cami)]
    r = executa(ordre)
    valors = []
    for linia in r.stdout.splitlines():
        text = linia.strip().rstrip(",")
        try:
            valors.append(int(text))
        except ValueError:
            continue
    return sorted(valors)


def preescalfament(cami: Path) -> int:
    """Quants fotogrames porta el fitxer ABANS del principi del clip."""
    caps = marques_de_temps(cami, f"%+#{PAQUETS_DE_CAP}")
    negatius = sum(1 for p in caps if p < 0)
    if negatius == len(caps) and caps:
        # N'hi ha mes dels que hem llegit: cal mirar-s'ho tot.
        negatius = sum(1 for p in marques_de_temps(cami) if p < 0)
    return negatius


def frequencia_honesta(cami: Path, info: dict) -> Fraction | None:
    """La frequencia real, ignorant el preescalfament i el retall de la cua.

    Es la que hauria de dir l'ffprobe. Llegeix tots els paquets (no
    descodifica): uns 10 s per fitxer.
    """
    base = info.get("base_de_temps")
    if not base:
        return None
    pts = [p for p in marques_de_temps(cami) if p >= 0]
    if len(pts) < 3:
        return None
    salts = [b - a for a, b in zip(pts, pts[1:])]
    tipic = statistics.median(salts)
    # L'ultim fotograma sovint el talla la llista d'edicio i deixa un salt
    # doble: si hi es, no compta per fer la mitjana.
    if salts[-1] > 1.5 * tipic:
        pts = pts[:-1]
    if len(pts) < 2 or pts[-1] == pts[0]:
        return None
    return Fraction(len(pts) - 1, 1) / (Fraction(pts[-1] - pts[0], 1) * base)


def forats(cami: Path) -> list[tuple[int, int, float]]:
    """On el mobil va perdre fotogrames gravant: (index, salt, segons)."""
    info = sonda(cami)
    base = info.get("base_de_temps")
    pts = [p for p in marques_de_temps(cami) if p >= 0]
    if len(pts) < 3 or not base:
        return []
    salts = [b - a for a, b in zip(pts, pts[1:])]
    tipic = statistics.median(salts)
    trobats = []
    for i, salt in enumerate(salts, start=1):
        if salt > 1.5 * tipic:
            trobats.append((i, salt, float(pts[i - 1] * base)))
    return trobats


def compta(cami: Path, mode: str, frequencia: Fraction | None = None) -> dict:
    """Passa l'original per l'ffmpeg sense codificar i llegeix els comptadors.

    mode 'cfr' reprodueix la decisio de temps de la conversio; mode
    'passthrough' copia els fotogrames un a un i dona la veritat de referencia:
    quants fotogrames presenta el fitxer.
    """
    ordre = [FFMPEG, "-hide_banner", "-nostats", "-v", "info", "-i", str(cami),
             "-an", "-map", "0:v:0", "-fps_mode", mode]
    if mode == "cfr":
        if frequencia is None:
            return {"error": "cal una frequencia per al mode cfr"}
        ordre += ["-r", f"{frequencia.numerator}/{frequencia.denominator}"]
    ordre += ["-f", "null", "-"]
    r = executa(ordre)
    text = (r.stderr or "").replace("\r", "\n")
    ultima = None
    for linia in text.splitlines():
        if linia.startswith("frame="):
            ultima = linia
    if ultima is None:
        return {"error": "l'ffmpeg no ha dit quants fotogrames ha fet"}
    n = int(re.search(r"frame=\s*(\d+)", ultima).group(1))
    dup = re.search(r"dup=\s*(\d+)", ultima)
    drop = re.search(r"drop=\s*(\d+)", ultima)
    return {
        "fotogrames": n,
        "duplicats": int(dup.group(1)) if dup else 0,
        "descartats": int(drop.group(1)) if drop else 0,
    }


# ------------------------------------------------------------- comprovacio


def revisa(original: Path, convertit: Path, rapid: bool, estricte: bool) -> dict:
    """Compara una parella i decideix el veredicte."""
    res = {"nom": original.stem, "original": str(original), "convertit": str(convertit)}

    o = sonda(original)
    c = sonda(convertit)
    if "error" in o or "error" in c:
        res["veredicte"] = "REFER"
        res["motiu"] = o.get("error") or c.get("error")
        return res

    res["mida"] = c["mida"]
    res["codec"] = c["codec"]
    res["nb_frames_original"] = o["nb_frames"]
    res["fotogrames_convertit"] = c["nb_frames"]
    res["frequencia_declarada"] = float(o["frequencia"]) if o["frequencia"] else None
    res["frequencia_del_convertit"] = float(c["frequencia"]) if c["frequencia"] else None

    res["preescalfament"] = preescalfament(original)

    # La frequencia honesta costa una lectura sencera del fitxer (uns 10 s) i
    # nomes fa falta quan hi ha motiu: si hi ha preescalfament -perque llavors
    # la declarada segur que menteix- o si al final el veredicte es dolent i
    # cal escriure la recepta que l'arregla.
    def calcula_honesta() -> None:
        if "frequencia_honesta" in res:
            return
        honesta = frequencia_honesta(original, o)
        res["frequencia_honesta"] = float(honesta) if honesta else None
        if honesta:
            res["recepta"] = f"{honesta.numerator}/{honesta.denominator}"
            if o["frequencia"]:
                res["error_de_frequencia"] = abs(
                    float(o["frequencia"] - honesta) / float(honesta)
                )

    if res["preescalfament"]:
        calcula_honesta()

    if rapid:
        # Triatge sense descodificar. NO ES CONCLOENT i el veredicte ho diu.
        calcula_honesta()  # aqui si que cal sempre: es l'unic senyal que tenim
        error = res.get("error_de_frequencia")
        presentats = (o["nb_frames"] or 0) - res["preescalfament"]
        diferencia = (c["nb_frames"] or 0) - presentats
        res["diferencia"] = diferencia
        sospitos = (
            res["preescalfament"] > 0
            or (error is not None and error > TOLERANCIA_FREQUENCIA)
            or abs(diferencia) > 2
        )
        res["veredicte"] = "MIRAR" if sospitos else "sembla ok"
        return res

    # La comprovacio de debo: reproduir la decisio de temps de la conversio.
    frequencia = c["frequencia"] or o["frequencia"]
    passada = compta(original, "cfr", frequencia)
    if "error" in passada:
        res["veredicte"] = "REFER"
        res["motiu"] = passada["error"]
        return res
    res.update(passada)

    if estricte:
        # Contrast 1: quants en presenta l'original, copiant-los un a un.
        res["presentats"] = compta(original, "passthrough").get("fotogrames")
        # Contrast 2: el convertit, descodificat de dalt a baix. Sense aixo
        # ens fiem de la seva capcalera, i un fitxer truncat te la capcalera
        # bona i no es descodifica sencer.
        res["descodificats"] = compta(convertit, "passthrough").get("fotogrames")
        if res["descodificats"] != c["nb_frames"]:
            res["veredicte"] = "REFER"
            res["motiu"] = (
                f"el convertit diu {c['nb_frames']} a la capcalera pero nomes en "
                f"descodifica {res['descodificats']}: esta trencat"
            )
            return res

    if passada["descartats"] > 0:
        res["veredicte"] = "REFER"
        calcula_honesta()  # cal la recepta que l'arregla
        res["motiu"] = (
            f"{passada['descartats']} fotogrames descartats de dins del clip"
        )
    elif c["nb_frames"] != passada["fotogrames"]:
        res["veredicte"] = "REFER"
        calcula_honesta()
        res["motiu"] = (
            f"el convertit en te {c['nb_frames']} i aquesta recepta en fa "
            f"{passada['fotogrames']}: no es va convertir aixi"
        )
    elif passada["duplicats"] > 0:
        res["veredicte"] = "OK*"
        trobats = forats(original)
        total = (o["nb_frames"] or 0) - res["preescalfament"]
        res["forats"] = [
            {"index": i, "segons": round(t, 3), "a_la_cua": i >= total - 2}
            for i, _, t in trobats
        ]
        a_la_cua = all(f["a_la_cua"] for f in res["forats"]) if res["forats"] else False
        res["motiu"] = (
            f"{passada['duplicats']} duplicat(s), "
            + ("a l'ultim fotograma: inofensiu" if a_la_cua else "DINS del clip: mira-t'ho")
        )
    else:
        res["veredicte"] = "OK"

    return res


def parelles(dir_orig: Path, dir_conv: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Aparella per nom. Busca dins de subcarpetes: aixi troba els de bons/."""
    originals = sorted(
        f for f in dir_orig.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS
    )
    convertits: dict[str, list[Path]] = {}
    for f in sorted(dir_conv.rglob("*")):
        if f.is_file() and f.suffix.lower() in EXTENSIONS:
            convertits.setdefault(f.stem, []).append(f)

    trobades, orfes = [], []
    for o in originals:
        if o.stem in convertits:
            for c in convertits[o.stem]:
                trobades.append((o, c))
        else:
            orfes.append(o)
    return trobades, orfes


# ------------------------------------------------------------------ sortida


def relatiu(cami: Path) -> str:
    try:
        return str(cami.relative_to(Path.cwd()))
    except ValueError:
        return str(cami)


def imprimeix(resultats: list[dict], orfes: list[Path], rapid: bool) -> None:
    print(f"{'fitxer':<18} {'convertit':<22} {'orig':>7} {'conv':>7} "
          f"{'dup':>4} {'desc':>5}  veredicte")
    print("-" * 88)
    for r in resultats:
        carpeta = Path(r["convertit"]).parent.name
        print(
            f"{r['nom']:<18} {carpeta[:22]:<22} "
            f"{str(r.get('nb_frames_original', '?')):>7} "
            f"{str(r.get('fotogrames_convertit', '?')):>7} "
            f"{str(r.get('duplicats', '-')):>4} "
            f"{str(r.get('descartats', '-')):>5}  {r['veredicte']}"
        )
        if r.get("motiu"):
            print(f"{'':<18} -> {r['motiu']}")
        if r.get("presentats") is not None:
            print(f"{'':<18} -> passthrough: l'original en presenta {r['presentats']}, "
                  f"i el convertit en descodifica {r.get('descodificats')}")
        if r.get("preescalfament"):
            declarada, honesta = r.get("frequencia_declarada"), r.get("frequencia_honesta")
            detall = ""
            if declarada and honesta:
                detall = (f": l'ffprobe declara {declarada:.4f} fps i en son "
                          f"{honesta:.4f}")
            print(f"{'':<18} -> {r['preescalfament']} fotogrames de preescalfament{detall}")

    if orfes:
        print()
        print("Sense conversio:")
        for o in orfes:
            print(f"  {o.name}")

    dolents = [r for r in resultats if r["veredicte"] == "REFER"]
    if dolents:
        print()
        print("Per arreglar-los, la frequencia surt dels fotogrames que el fitxer")
        print("presenta de debo, no de la que declara:")
        for r in dolents:
            if not r.get("recepta"):
                continue
            print()
            print(f'  ffmpeg -y -i "{relatiu(Path(r["original"]))}" `')
            print("      -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p `")
            print(f'      -fps_mode cfr -r {r["recepta"]} -an "<carpeta nova>/{r["nom"]}.mp4"')

    print()
    if rapid:
        print("ATENCIO -> aixo era el triatge rapid i NO ES CONCLOENT: nomes mira")
        print("           metadades. Passa-ho sense --rapid abans de fiar-te'n.")
    elif dolents:
        print(f"ATENCIO -> {len(dolents)} conversio(ns) s'han de refer.")
    else:
        print(f"OK -> les {len(resultats)} conversions son fidels a l'original.")


# ---------------------------------------------------------------- autoprova


def autoprova() -> int:
    """Fabrica un clip, el converteix be i malament, i mira si ho encerta."""
    arrel = Path(__file__).resolve().parent.parent / "esborranys" / "_autoprova_conversio"
    arrel.mkdir(parents=True, exist_ok=True)
    patro = arrel / "patro.mp4"
    bo = arrel / "bo.mp4"
    dolent = arrel / "dolent.mp4"
    codec = ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an"]
    try:
        print("Fabricant un clip de 300 fotogrames a 100 fps...")
        executa([FFMPEG, "-y", "-hide_banner", "-v", "error",
                 "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=100",
                 "-frames:v", "300", *codec, str(patro)])
        print("Convertint-lo BE (-r 100) i MALAMENT (-r 95)...")
        for desti, fps in ((bo, "100"), (dolent, "95")):
            executa([FFMPEG, "-y", "-hide_banner", "-v", "error", "-i", str(patro),
                     *codec, "-fps_mode", "cfr", "-r", fps, str(desti)])
        if not (patro.exists() and bo.exists() and dolent.exists()):
            print("ATENCIO -> l'ffmpeg no ha pogut fabricar el clip de prova.")
            return 1

        print()
        ok = True
        r_bo = revisa(patro, bo, rapid=False, estricte=False)
        print(f"  conversio bona    -> {r_bo['veredicte']:<6} "
              f"descartats={r_bo.get('descartats')}")
        if r_bo["veredicte"] != "OK" or r_bo.get("descartats") != 0:
            print("     !! hauria de dir OK amb 0 descartats")
            ok = False

        r_dolent = revisa(patro, dolent, rapid=False, estricte=False)
        print(f"  conversio dolenta -> {r_dolent['veredicte']:<6} "
              f"descartats={r_dolent.get('descartats')}")
        if r_dolent["veredicte"] != "REFER" or not r_dolent.get("descartats"):
            print("     !! hauria de dir REFER amb descartats > 0")
            ok = False

        print()
        if ok:
            print("OK -> l'eina distingeix una conversio fidel d'una que perd fotogrames.")
            print("     Limit conegut: l'autoprova NO cobreix el preescalfament amb")
            print("     marca de temps negativa, que no es pot fabricar amb l'ffmpeg.")
            return 0
        print("ATENCIO -> l'autoprova ha fallat.")
        return 1
    finally:
        shutil.rmtree(arrel, ignore_errors=True)


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("originals", nargs="?", help="carpeta amb els videos de la camera")
    ap.add_argument("convertits", nargs="?", help="carpeta amb els .mp4 convertits")
    ap.add_argument("--rapid", action="store_true",
                    help="nomes metadades, sense descodificar; triatge NO concloent")
    ap.add_argument("--estricte", action="store_true",
                    help="descodifica tambe l'original i el convertit sencers (el triple de temps)")
    ap.add_argument("--processos", type=int, default=3,
                    help="quants fitxers alhora (per defecte 3)")
    ap.add_argument("--json", help="on desar els numeros")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova que l'eina detecta el que ha de detectar")
    args = ap.parse_args(argv)

    cal_ffmpeg()

    if args.autoprova:
        return autoprova()

    if not args.originals or not args.convertits:
        ap.error("calen les dues carpetes: originals i convertits")

    dir_orig, dir_conv = Path(args.originals), Path(args.convertits)
    for d in (dir_orig, dir_conv):
        if not d.is_dir():
            raise SystemExit(f"No hi ha cap carpeta a {d}")

    trobades, orfes = parelles(dir_orig, dir_conv)
    if not trobades:
        raise SystemExit(f"Cap fitxer de {dir_orig} no te parella a {dir_conv}")

    quant = "metadades" if args.rapid else f"~25 s cadascun, de {args.processos} en {args.processos}"
    print(f"Comprovant {len(trobades)} parelles ({quant})...\n")

    with ThreadPoolExecutor(max_workers=max(1, args.processos)) as executor:
        resultats = list(executor.map(
            lambda p: revisa(p[0], p[1], args.rapid, args.estricte), trobades
        ))

    imprimeix(resultats, orfes, args.rapid)

    if args.json:
        desti = Path(args.json)
        desti.parent.mkdir(parents=True, exist_ok=True)
        desti.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Numeros desats a {relatiu(desti)}")

    return 1 if any(r["veredicte"] == "REFER" for r in resultats) else 0


if __name__ == "__main__":
    raise SystemExit(main())
