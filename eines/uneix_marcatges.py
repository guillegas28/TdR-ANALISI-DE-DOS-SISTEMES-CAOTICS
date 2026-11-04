"""Uneix dos marcatges d'una presa: el nou fins a un fotograma, el vell d'alli endavant.

Per que existeix. El 25/09/2026 en Guillem va afegir al Tracker el repos
d'abans d'amollar a les 10 preses noves de 45. L'autotracker no es va aturar al
primer fotograma que ja hi era i en va tornar a marcar un bloc seguit. A la
majoria el marcatge nou es tan bo com el vell, pero no a totes:

  - 45_2 i 45_10: el seguiment de l'articulacio es degrada en ple moviment;
  - 45_7: una marca de l'articulacio salta 341 px al fotograma 2119;
  - 45_5 i 45_8: el gomet fa un salt de 4-5 px on s'acaba el tram remarcat.

La primera provatura (esborranys/_empelta_principi.py) posava el marcatge vell
DES DEL SEU PRIMER FOTOGRAMA, i va fallar: els 4-5 primers fotogrames del
marcatge vell tenen un transitori d'arrencada, i l'empelt hi posava un salt de
fins a 0,8 graus de theta1 just a l'amollada. Vegeu
documentacio/pla-45-amb-principi.md.

Que fa, i el criteri, fixat abans de mirar (pla del 26/09/2026):
  - A les pistes rodament i extrem, el marcatge d'ara fins al fotograma j-1 i
    el del commit vell des del j. El fix no es toca: nomes se'n fa la mediana.
  - j es tria entre el 6e fotograma vell (el transitori ja ha passat) i
    l'ultim del tram remarcat, alli on la suma de distancies entre els dos
    marcatges, a les dues pistes, es mes petita.
  - S'accepta nomes si:
      1. a j, cada pista coincideix a 1 px o menys;
      2. el pas de cada pista a la unio no passa de la mediana dels pasos
         veins mes 1 px;
      3. en angles crus, el pas de theta1 i de theta2 a la unio no passa de la
         mediana dels pasos veins mes 3 sigmes del soroll de la presa.
    Si no, la presa NO es toca.

Com s'escriu: sobre el TEXT del .trk, reordenant blocs de linies que ja
existeixen (el Tracker llegeix amb un lector propi i no es pot reescriure amb
un parser d'XML). Abans de substituir res, el resultat es comprova: fora dels
dos blocs framedata el fitxer es identic; es XML ben format; i tornat a llegir
dona exactament el marcatge d'ara abans de j i el vell des de j.

Sense --escriu nomes diu que faria.

Us:
    .venv\\Scripts\\python eines\\uneix_marcatges.py 45_2 45_5 45_7 45_8 45_10
    .venv\\Scripts\\python eines\\uneix_marcatges.py 45_2 45_5 45_7 45_8 45_10 --escriu
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import tracker, trk  # noqa: E402

COMMIT_VELL = "efdc5c2"
PISTES = ("rodament", "extrem")      # el fix no es toca
MARGE_TRANSITORI = 5                 # j >= primer fotograma vell + 5 (el 6e)
COINCIDENCIA_PX = 1.0
MARGE_PAS_PX = 1.0
SIGMES_ANGLE = 3.0
VEINS = 5

# L'estructura d'una pista al .trk, tal com l'escriu el Tracker 6.
INICI_PISTA = re.compile(r'^ {8}<object class="org\.opensourcephysics\.cabrillo\.tracker\.PointMass">')
FI_PISTA = re.compile(r"^ {8}</object>\s*$")
NOM = re.compile(r'^ {12}<property name="name" type="string">(.*)</property>\s*$')
INICI_FD = re.compile(r'^ {12}<property name="framedata" type="array" '
                      r'class="\[Lorg\.opensourcephysics\.cabrillo\.tracker\.PointMass\$FrameData;">')
FI_FD = re.compile(r"^ {12}</property>\s*$")
INICI_ENTRADA = re.compile(r'^ {16}<property name="\[(\d+)\]" type="object">')
FI_ENTRADA = re.compile(r"^ {16}</property>\s*$")
XY = re.compile(r'<property name="([xy])" type="double">([^<]*)</property>')


# ------------------------------------------------------------ el text del .trk

def blocs_framedata(linies: list[str]) -> dict:
    """{nom de la pista: (inici, fi, {fotograma: [linies]})}. fi es exclusiu."""
    fora = {}
    i = 0
    while i < len(linies):
        if not INICI_PISTA.match(linies[i]):
            i += 1
            continue
        nom = None
        j = i + 1
        while j < len(linies) and not FI_PISTA.match(linies[j]):
            m = NOM.match(linies[j])
            if m and nom is None:
                nom = m.group(1)
            if INICI_FD.match(linies[j]):
                inici = j
                entrades: dict[int, list[str]] = {}
                k = j + 1
                while not FI_FD.match(linies[k]):
                    m = INICI_ENTRADA.match(linies[k])
                    if not m:
                        raise SystemExit(f"linia {k + 1}: esperava una entrada de framedata i hi ha "
                                         f"{linies[k].strip()[:80]!r}")
                    fi_e = k + 1
                    while not FI_ENTRADA.match(linies[fi_e]):
                        fi_e += 1
                    entrades[int(m.group(1))] = linies[k:fi_e + 1]
                    k = fi_e + 1
                fora[nom] = (inici, k + 1, entrades)
                j = k
            j += 1
        i = j + 1
    return fora


def tria_pistes(blocs: dict, fitxer: str) -> dict:
    """fix, rodament, extrem -> nom de la pista, amb el criteri de pendol/trk.py."""
    per_nom = {n.lower(): n for n in blocs}
    fora = {}
    for esperat in trk.NOMS_ESPERATS:
        c = [k for k in per_nom if esperat in k]
        if len(c) != 1:
            raise SystemExit(f"{fitxer}: no se quina pista es {esperat!r} ({', '.join(blocs)})")
        fora[esperat] = per_nom[c[0]]
    return fora


def posicio(entrada: list[str]) -> np.ndarray:
    """La (x, y) d'una entrada de framedata."""
    v = dict(XY.findall("".join(entrada)))
    return np.array([float(v["x"]), float(v["y"])])


def fora_dels_blocs(linies: list[str]) -> list[str]:
    """El fitxer sense els framedata de rodament i extrem: el que no s'ha de moure."""
    blocs = blocs_framedata(linies)
    pistes = tria_pistes(blocs, "?")
    trams = sorted(blocs[pistes[p]][:2] for p in PISTES)
    fora, anterior = [], 0
    for i, f in trams:
        fora.extend(linies[anterior:i])
        anterior = f
    fora.extend(linies[anterior:])
    return fora


# ------------------------------------------------------------------- la unio

def tria_la_unio(ara: dict, vell: dict) -> tuple[int, int, int, dict]:
    """(k0 vell, ultim remarcat, j, distancies a j per pista)."""
    k0 = max(min(vell[p]) for p in PISTES)
    # Per POSICIO, no pel text: el Git desa el .trk amb LF i al disc va amb
    # CRLF, i comparant linies tot sortiria remarcat.
    remarcats = [k for p in PISTES for k in vell[p]
                 if k in ara[p] and np.linalg.norm(posicio(ara[p][k]) - posicio(vell[p][k])) > 1e-6]
    if not remarcats:
        raise SystemExit("no hi ha cap fotograma remarcat: no cal unir res")
    kfi = max(remarcats)
    millor = None
    for j in range(k0 + MARGE_TRANSITORI, kfi + 1):
        if not all(j in ara[p] and j in vell[p] for p in PISTES):
            continue
        d = {p: float(np.linalg.norm(posicio(ara[p][j]) - posicio(vell[p][j]))) for p in PISTES}
        s = sum(d.values())
        if millor is None or s < millor[0]:
            millor = (s, j, d)
    if millor is None:
        raise SystemExit("no hi ha cap fotograma candidat a la unio")
    return k0, kfi, millor[1], millor[2]


def uneix(linies_ara, linies_vell, fitxer):
    blocs_ara, blocs_vell = blocs_framedata(linies_ara), blocs_framedata(linies_vell)
    p_ara, p_vell = tria_pistes(blocs_ara, fitxer), tria_pistes(blocs_vell, fitxer + " (vell)")
    ara = {p: blocs_ara[p_ara[p]][2] for p in PISTES}
    vell = {p: blocs_vell[p_vell[p]][2] for p in PISTES}
    k0, kfi, j, dist = tria_la_unio(ara, vell)

    final = "\r\n" if linies_ara[0].endswith("\r\n") else "\n"
    substitucions, nou_per_pista = [], {}
    for p in PISTES:
        inici, fi, _ = blocs_ara[p_ara[p]]
        nou = {k: v for k, v in ara[p].items() if k < j}
        for k, v in vell[p].items():
            if k >= j:
                nou[k] = [l.rstrip("\r\n") + final for l in v]
        nou_per_pista[p] = nou
        cos = [l for k in sorted(nou) for l in nou[k]]
        substitucions.append((inici, fi, [linies_ara[inici], *cos, linies_ara[fi - 1]]))
    sortida = list(linies_ara)
    for inici, fi, noves in sorted(substitucions, key=lambda s: s[0], reverse=True):
        sortida[inici:fi] = noves
    return sortida, {"k0": k0, "kfi": kfi, "j": j, "dist": dist}, nou_per_pista


def pasos(nou: dict, j: int):
    """(pas a la unio j-1 -> j, mediana dels pasos veins)."""
    pos = {k: posicio(v) for k, v in nou.items() if j - VEINS - 1 <= k <= j + VEINS}
    def pas(k):
        return float(np.linalg.norm(pos[k] - pos[k - 1])) if k in pos and k - 1 in pos else np.nan
    veins = [pas(k) for k in list(range(j - VEINS, j)) + list(range(j + 1, j + VEINS + 1))]
    return pas(j), float(np.nanmedian(veins))


def pasos_angulars(cami: Path, j: int):
    """Pas de theta1 i theta2 crus a la unio, mediana dels veins, i el soroll de la presa."""
    with contextlib.redirect_stdout(io.StringIO()):
        t, x1, y1, x2, y2, meta = tracker.llegeix_del_trk(cami)
    k, *_ = trk.tres_pistes(trk.llegeix_trk(cami))
    th1, th2 = tracker.a_angles(t, x1, y1, x2, y2, (0.0, 0.0), True, False)
    idx = {int(kk): i for i, kk in enumerate(k)}
    soroll = meta["geometria"]["rigidesa"]["soroll_angular_graus"]
    fora = []
    for th, s in ((np.degrees(th1), soroll[0]), (np.degrees(th2), soroll[1])):
        def pas(kk):
            return abs(th[idx[kk]] - th[idx[kk - 1]]) if kk in idx and kk - 1 in idx else np.nan
        veins = [pas(kk) for kk in list(range(j - VEINS, j)) + list(range(j + 1, j + VEINS + 1))]
        fora.append((float(pas(j)), float(np.nanmedian(veins)), float(s)))
    return fora


def processa(nom: str, commit: str, escriu: bool) -> bool:
    cami = ARREL / "dades" / "video" / "tracker" / nom.split("_")[0] / f"{nom}.trk"
    rel = cami.relative_to(ARREL).as_posix()
    vell_bytes = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ARREL,
                                capture_output=True, check=True).stdout
    linies_ara = cami.read_bytes().decode("utf-8").splitlines(keepends=True)
    linies_vell = vell_bytes.decode("utf-8").splitlines(keepends=True)

    linies_nou, info, nou_per_pista = uneix(linies_ara, linies_vell, str(cami))
    j = info["j"]
    temporal = ARREL / "esborranys" / "_unio" / f"{nom}.trk"
    temporal.parent.mkdir(parents=True, exist_ok=True)
    temporal.write_bytes("".join(linies_nou).encode("utf-8"))

    print(f"\n{nom}: primer fotograma vell {info['k0']}, tram remarcat fins al {info['kfi']}, "
          f"unio al {j} ({j - info['k0']} despres del primer vell)")
    problemes = []
    for p in PISTES:
        d = info["dist"][p]
        pas_j, med = pasos(nou_per_pista[p], j)
        print(f"  {p:<9} coincidencia a la unio {d:.2f} px   pas a la unio {pas_j:.2f} px (veins {med:.2f})")
        if d > COINCIDENCIA_PX:
            problemes.append(f"{p}: a la unio els dos marcatges difereixen {d:.2f} px")
        if pas_j > med + MARGE_PAS_PX:
            problemes.append(f"{p}: pas de {pas_j:.2f} px a la unio, veins {med:.2f}")

    # Comprovacions del fitxer
    if fora_dels_blocs(linies_ara) != fora_dels_blocs(linies_nou):
        problemes.append("fora dels framedata el fitxer ha canviat")
    try:
        ET.parse(temporal)
    except ET.ParseError as e:
        problemes.append(f"no es XML ben format: {e}")
    if not problemes:
        d = trk.llegeix_trk(temporal)
        per = {n.lower(): n for n in d["pistes"]}
        for p in PISTES:
            k, x, y = d["pistes"][per[[q for q in per if p in q][0]]]
            esperat = nou_per_pista[p]
            ks = sorted(esperat)
            xy = np.array([posicio(esperat[kk]) for kk in ks])
            if not (np.array_equal(k, np.array(ks)) and np.array_equal(x, xy[:, 0])
                    and np.array_equal(y, xy[:, 1])):
                problemes.append(f"{p}: tornat a llegir no dona el marcatge esperat")
        for nom_a, (pas_j, med, s) in zip(("theta1", "theta2"), pasos_angulars(temporal, j)):
            print(f"  {nom_a}    pas a la unio {pas_j:.3f} graus (veins {med:.3f}, 3 sigmes {SIGMES_ANGLE * s:.3f})")
            if pas_j > med + SIGMES_ANGLE * s:
                problemes.append(f"{nom_a}: pas de {pas_j:.3f} graus a la unio")

    if problemes:
        print("  NO ES TOCA:")
        for q in problemes:
            print(f"    - {q}")
        temporal.unlink()
        return False
    if escriu:
        os.replace(temporal, cami)
        print(f"  comprovat i escrit: {rel}")
    else:
        temporal.unlink()
        print("  comprovat; no s'ha escrit res (falta --escriu)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("preses", nargs="+", help="p.ex. 45_2 45_10")
    ap.add_argument("--vell", default=COMMIT_VELL, help=f"el commit del marcatge vell (per defecte {COMMIT_VELL})")
    ap.add_argument("--escriu", action="store_true", help="escriu el .trk unit si passa les comprovacions")
    args = ap.parse_args(argv)
    bones = [processa(n, args.vell, args.escriu) for n in args.preses]
    return 0 if all(bones) else 1


if __name__ == "__main__":
    sys.exit(main())
