r"""Que mesura la regla 2? L'exponent de TEMPS FINIT (FTLE) de la seva finestra. NOMES SIMULACIO.

Puja d'esborranys/_ftle_finestra.py el 06/10/2026, amb la sortida identica (criteri
P1 de documentacio/pla-analisi-caos.md, «06/10 · Feina sense intervencio»).

LA PREGUNTA. La regla 2 (caos.ajusta_zona), sense soroll, dona als 90 graus un 22-32 %
mes que Benettin, i als 45 graus lambda ~ 1,5 on Benettin diu que 9 de 12 orbites son
regulars. Llegeix el primer segon. El que llegeix es la FTLE d'aquella finestra?

COM. Els 12 tirs de eines/regla2_amb_soroll.py (mateixes condicions inicials). Per a
cada tir, 8 direccions a l'atzar al pla (theta1, theta2) amb omega = 0 i delta = 1e-8
a la norma de mesures.distancia_fases; x i x + delta integrats amb simulador.simula
(tolerancia estricta) fins a 2 s, sense reescalar. <ln(d/d0)> sobre tirs i
direccions, i el seu pendent a la finestra [t_inici, t_fi] de la zona de la regla 2
sense soroll (llegida de dades/divergencia/regla2_amb_soroll_2026-10-03.json).

EL RESULTAT (03/10): la regla 2 coincideix amb la FTLE de la seva finestra a -4/+1 %.
No s'equivoca: mesura el ritme de separacio del PRIMER SEGON, que no es la lambda
asimptotica. Als 45 graus aquell ritme es real (~1,4 s^-1) encara que l'orbita sigui
regular a la llarga.

Criteris (escrits al pla el 03/10, abans del codi):
  F0 (--autoprova)  delta 1e-8 i 1e-7 coincideixen a < 1e-3 en ln, i es reproduible;
  F1 (l'execucio)   als 90 i 150 graus, |regla 2 / FTLE de la finestra - 1| <= 20 %.

Us:
    .venv\\Scripts\\python eines\\ftle_finestra.py --autoprova
    .venv\\Scripts\\python eines\\ftle_finestra.py --json dades/divergencia/ftle_finestra_2026-10-06.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))

from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import distancia_fases, omega_zero  # noqa: E402
from pendol.simulador import simula  # noqa: E402
import regla2_amb_soroll as rg  # noqa: E402

REGLA2_JSON = ARREL / "dades" / "divergencia" / "regla2_amb_soroll_2026-10-03.json"
DIRECCIONS = 8
DELTA = 1e-8
LLAVOR = 20261007
FPS = 240.0


def corbes(p, x0s, t, delta, llavor, tirs=None):
    """(tirs x direccions, M): ln(d/d0) de cada perturbacio infinitesimal."""
    w0 = omega_zero(p)
    rng = np.random.default_rng(llavor)
    fi = rng.uniform(0.0, 2 * np.pi, (len(x0s), DIRECCIONS))
    sortida = []
    for i, x0 in enumerate(x0s):
        if tirs is not None and i not in tirs:
            continue
        _, ya = simula(p, x0, t_eval=t)
        for k in range(DIRECCIONS):
            dx = delta * np.array([math.cos(fi[i, k]), math.sin(fi[i, k]), 0.0, 0.0])
            _, yb = simula(p, x0 + dx, t_eval=t)
            d0 = float(distancia_fases(x0, x0 + dx, w0))
            sortida.append(np.log(distancia_fases(ya, yb, w0) / d0))
    return np.array(sortida)


def pendent(t, y, a, b):
    sel = (t >= a) & (t <= b)
    return float(np.polyfit(t[sel], y[sel], 1)[0])


def _mira(problemes):
    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)
    return mira


def _tanca(problemes) -> int:
    print()
    if problemes:
        print(f"FALLEN {len(problemes)} criteris:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK")
    return 0


def autoprova() -> int:
    """F0: el codi. Linealitat (1e-8 contra 1e-7) i reproduible, sobre 2 tirs dels 150."""
    problemes = []
    mira = _mira(problemes)
    p = Parametres.des_de_json(ARREL / "parametres.json")
    print("Autoprova de eines/ftle_finestra.py (F0, escrit al pla el 03/10). NOMES SIMULACIO.\n")
    t = np.arange(0.0, 2.0 + 0.5 / FPS, 1.0 / FPS)
    print("F0. delta 1e-8 i 1e-7 coincideixen a < 1e-3 en ln, i es reproduible bit a bit")
    x0s = rg.condicions_inicials(150.0, 1.0, rg.llavor_de(150.0, 1.0))
    a = corbes(p, x0s, t, DELTA, LLAVOR, tirs={0, 1})
    b = corbes(p, x0s, t, 10 * DELTA, LLAVOR, tirs={0, 1})
    a2 = corbes(p, x0s, t, DELTA, LLAVOR, tirs={0, 1})
    dif = float(np.max(np.abs(a - b)))
    mira(dif < 1e-3, f"diferencia maxima en ln(d/d0) fins a 2 s: {dif:.1e} (creixement maxim {a.max():.2f})")
    mira(np.array_equal(a, a2), "dues passades identiques")
    return _tanca(problemes)


def calcula(json_sortida: Path | None) -> int:
    """F1: la regla 2 contra la FTLE de la mateixa finestra, als tres grups i dues dispersions."""
    p = Parametres.des_de_json(ARREL / "parametres.json")
    ref = json.loads(REGLA2_JSON.read_text(encoding="utf-8"))
    problemes = []
    mira = _mira(problemes)
    print("La FTLE a la finestra de la regla 2 (F1, escrit al pla el 03/10). NOMES SIMULACIO.\n")
    t = np.arange(0.0, 2.0 + 0.5 / FPS, 1.0 / FPS)
    print("F1. Als 90 i 150 graus, |regla 2 / FTLE de la finestra - 1| <= 20 %")
    resultats = {}
    for g in rg.GRUPS:
        k = f"{g:.0f}"
        lb = ref["grups"][k].get("benettin", {}).get("lambda_mediana_caotiques")
        for disp in rg.DISPERSIONS_GRAUS:
            s = ref["grups"][k]["dispersions"][f"{disp:g}"]["sense_soroll"]
            x0s = rg.condicions_inicials(g, disp, rg.llavor_de(g, disp))
            C = corbes(p, x0s, t, DELTA, LLAVOR + int(g) + int(10 * disp))
            m = C.mean(axis=0)
            ti, tf = s["t_inici_s"], s["t_fi_s"]
            ftle = pendent(t, m, ti, tf)
            per_tir = C.reshape(len(x0s), DIRECCIONS, -1).mean(axis=1)
            ftle_tirs = [pendent(t, c, ti, tf) for c in per_tir]
            l2 = s.get("lambda")
            resultats[f"{k}/{disp:g}"] = {"finestra_s": [ti, tf], "regla2": l2, "ftle_finestra": ftle,
                                          "ftle_per_tir": ftle_tirs, "benettin": lb,
                                          "ln_d_d0_mitjana_a_t": {f"{x:g}": float(m[int(round(x * FPS))])
                                                                  for x in (0.25, 0.5, 1.0, 1.5, 2.0)}}
            txt = (f"{k} graus, dispersio {disp:g}: finestra {ti:.2f}-{tf:.2f} s; regla 2 "
                   f"{l2 if l2 is None else round(l2, 3)}, FTLE {ftle:.3f} (per tir {min(ftle_tirs):.2f}-"
                   f"{max(ftle_tirs):.2f}), Benettin {lb if lb is None else round(lb, 3)}")
            if k == "45":
                print(f"   (info) {txt}")
                continue
            ok = l2 is not None and abs(l2 / ftle - 1) <= 0.20
            mira(ok, txt + (f"  -> {100 * (l2 / ftle - 1):+.0f} %" if l2 else ""))
    if json_sortida:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(
            {"eina": "eines/ftle_finestra.py", "pla": "documentacio/pla-analisi-caos.md",
             "que_es": "NOMES SIMULACIO, sense friccio: la regla 2 sense soroll contra la FTLE infinitesimal "
                       "de la mateixa finestra, 12 tirs x 8 direccions per grup i dispersio",
             "resultats": resultats}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {json_sortida}")
    return _tanca(problemes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    return calcula(Path(args.json) if args.json else None)


if __name__ == "__main__":
    sys.exit(main())
