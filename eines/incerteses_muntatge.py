r"""Quant pesen a la simulacio les incerteses del muntatge.

Pla: documentacio/pla-incerteses-i-feix.md, punt 2 (01/10/2026).

El CLAUDE.md te una llista d'incerteses del muntatge que «van a la memoria com a
tals»: on cauen els caps dels cargols, la barra 2 asimetrica (el seu I va per
sota del que dona el periode), el desacord del pendol sencer amb el periode del
banc, i la L1. Fins ara ningu no havia mesurat si pesen mes o menys que les
altres dues coses que limiten la prediccio: la incertesa de les condicions
inicials (l'amplada del feix) i la friccio.

Per a cada variant del muntatge, una sola cosa canviada:
  - els periodes dels dos modes i les energies de volteig, en % de canvi;
  - L'HORITZO DEL MODEL: el pendol nominal i la variant surten de la MATEIXA
    condicio inicial (les 35 de dades/versemblant/), i es pren el primer instant
    en que |theta2 nominal - theta2 variant| passa de X. Es quant temps segueix
    un pendol perfectament conegut d'on surt, pero amb el muntatge una mica
    diferent.

NOMES SIMULACIO I SENSE FRICCIO. De les preses reals nomes es llegeixen els
angles de sortida del JSON de dades/versemblant/; cap CSV.

    .venv\Scripts\python.exe eines\incerteses_muntatge.py --autoprova
    .venv\Scripts\python.exe eines\incerteses_muntatge.py --json esborranys/incerteses.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import caos  # noqa: E402
from pendol.fisica import Parametres, periode_petites_oscillacions  # noqa: E402
from pendol.simulador import ATOL, RTOL, simula  # noqa: E402

NUMEROS = ARREL / "dades" / "versemblant" / "numeros_2026-09-26.json"
FRICCIO = ARREL / "esborranys" / "_horitzo_friccio" / "banc.json"
X_GRAUS = (2.0, 3.0, 5.0)
TEMPS_S = 6.0
FPS = 240.0


def variants(p: Parametres) -> dict:
    """Les incerteses documentades al CLAUDE.md, cadascuna sola."""
    # La barra 2 com diu el seu periode: I2 un 2,98 % mes gran, allargant R2,
    # com fa el punt 3b de l'autoprova del simulador.
    R2_periode = p.R2 * math.sqrt(1 + 0.0298 * p.I2 / ((1 / 12) * p.m2 * p.R2**2))
    return {
        "cargols a 12,78 cm (I1 +4,6 %)": replace(p, rs=0.1278),
        "barra 2 com el periode (I2 +2,98 %)": replace(p, R2=R2_periode),
        "tot el pendol com el banc (-0,89 %)": replace(p, g=p.g * (1 - 0.0089)),
        "L1 +1 mm": replace(p, L1=p.L1 + 0.001),
        "L1 -1 mm": replace(p, L1=p.L1 - 0.001),
    }


def canvis(p: Parametres, q: Parametres) -> dict:
    """Els periodes dels modes i les energies de volteig de q respecte de p, en %."""
    Tp, Tq = periode_petites_oscillacions(p), periode_petites_oscillacions(q)
    return {
        "periode_mode_lent_pc": 100 * (Tq[0] / Tp[0] - 1),
        "periode_mode_rapid_pc": 100 * (Tq[1] / Tp[1] - 1),
        "E_volteig_1_pc": 100 * (caos.energia_volteig_1(q) / caos.energia_volteig_1(p) - 1),
        "E_volteig_2_pc": 100 * (caos.energia_volteig_2(q) / caos.energia_volteig_2(p) - 1),
        "I1_pc": 100 * (q.I1 / p.I1 - 1),
        "I2_pc": 100 * (q.I2 / p.I2 - 1),
    }


def horitzo_del_model(p, q, angles_graus, xs=X_GRAUS, temps=TEMPS_S) -> dict:
    """El primer instant en que |theta2(p) - theta2(q)| > X, des de la mateixa CI."""
    estat0 = np.radians([*angles_graus, 0.0, 0.0])
    t, yp = simula(p, estat0, temps, FPS, rtol=RTOL, atol=ATOL)
    _, yq = simula(q, estat0, temps, FPS, rtol=RTOL, atol=ATOL)
    dif = np.abs(yq[1] - yp[1])
    fora = {}
    for x in xs:
        k = np.nonzero(dif > math.radians(x))[0]
        fora[x] = float(t[k[0]]) if len(k) else math.inf
    return fora


def grup_de(nom: str) -> str:
    return nom.split("_")[0]


def _mediana(v):
    return float(np.median(v)) if v else math.nan


def _s(v: float) -> str:
    return f"> {TEMPS_S:g}" if math.isinf(v) else f"{v:.2f}"


def autoprova() -> int:
    p = Parametres.des_de_json(ARREL / "parametres.json")
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova de les incerteses del muntatge (criteris escrits abans de mirar)\n")
    igual = replace(p)
    c = canvis(p, igual)
    h = horitzo_del_model(p, igual, (150.0, 150.0))
    mira(all(v == 0.0 for v in c.values()) and all(math.isinf(v) for v in h.values()),
         "una variant identica al nominal: canvis de 0,0 % i no se'n separa mai")

    g = replace(p, g=p.g * (1 - 0.0089))
    esperat = 100 * (1 / math.sqrt(1 - 0.0089) - 1)
    cg = canvis(p, g)
    err = max(abs(cg["periode_mode_lent_pc"] - esperat), abs(cg["periode_mode_rapid_pc"] - esperat))
    mira(err < 1e-4, f"g -0,89 %: els dos periodes +{esperat:.4f} % (difereixen {err:.1e} punts)")

    hq = horitzo_del_model(p, replace(p, rs=0.1278), (150.0, 150.0))
    mira(hq[2.0] <= hq[3.0] <= hq[5.0],
         f"amb una X mes gran no arriba abans: {_s(hq[2.0])} <= {_s(hq[3.0])} <= {_s(hq[5.0])} s")

    print()
    if problemes:
        print("AUTOPROVA FALLADA")
        return 1
    print("OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="on desar els numeros")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()

    p = Parametres.des_de_json(ARREL / "parametres.json")
    num = json.loads(NUMEROS.read_text(encoding="utf-8"))
    ci = {n: c["angles_graus"] for n, c in num["B_condicions_inicials"]["per_presa"].items()}
    grups = sorted({grup_de(n) for n in ci}, key=float)
    vs = variants(p)

    print("Les incerteses del muntatge a la simulacio (sense friccio)")
    print(f"{len(ci)} condicions inicials de {NUMEROS.relative_to(ARREL)}\n")
    print("1. Que canvia cada variant (%)")
    print(f"   {'variant':<38} {'I1':>7} {'I2':>7} {'T lent':>7} {'T rapid':>8} {'E_v1':>7} {'E_v2':>7}")
    sortida = {"variants": {}, "per_grup": {}}
    for nom, q in vs.items():
        c = canvis(p, q)
        sortida["variants"][nom] = {"canvis": c}
        print(f"   {nom:<38} {c['I1_pc']:>+7.2f} {c['I2_pc']:>+7.2f} {c['periode_mode_lent_pc']:>+7.3f} "
              f"{c['periode_mode_rapid_pc']:>+8.3f} {c['E_volteig_1_pc']:>+7.2f} {c['E_volteig_2_pc']:>+7.2f}")

    print(f"\n2. L'horitzo del model: quan |dtheta2| passa de X, des de la mateixa condicio inicial")
    print(f"   (mediana per grup, i entre claudators el rang; {TEMPS_S:g} s simulats)")
    hor = {nom: {n: horitzo_del_model(p, q, a) for n, a in ci.items()} for nom, q in vs.items()}
    for nom in vs:
        sortida["variants"][nom]["horitzo_per_presa"] = {
            n: {str(x): (None if math.isinf(v) else v) for x, v in h.items()} for n, h in hor[nom].items()}

    D = num["D_horitzo_amplada_del_feix"]["per_grup"]
    fr = json.loads(FRICCIO.read_text(encoding="utf-8"))["D_horitzo_amplada_del_feix"]["per_grup"] \
        if FRICCIO.exists() else None
    for x in X_GRAUS:
        print(f"\n   X = {x:g} graus")
        print(f"   {'':<38}" + "".join(f"{g + ' graus':>20}" for g in grups))
        for nom in vs:
            cel = []
            for g in grups:
                v = [hor[nom][n][x] for n in ci if grup_de(n) == g]
                cel.append(f"{_s(_mediana(v))} [{_s(min(v))}-{_s(max(v))}]")
                sortida["per_grup"].setdefault(g, {}).setdefault(str(x), {})[nom] = {
                    "mediana_s": None if math.isinf(_mediana(v)) else _mediana(v),
                    "min_s": None if math.isinf(min(v)) else min(v),
                    "max_s": None if math.isinf(max(v)) else max(v)}
            print(f"   {nom:<38}" + "".join(f"{c:>20}" for c in cel))
        for amp in ("0.03", "0.1", "0.3"):
            cel = []
            for g in grups:
                m = D[g][amp][str(int(x))]["minmax_theta2_s"]["mediana"]
                cel.append(_s(math.inf if m is None else m))
            print(f"   {'feix de ' + amp + ' graus (condicions inicials)':<38}" + "".join(f"{c:>20}" for c in cel))
        if fr:
            cel = []
            for g in grups:
                m = fr[g]["0.1"][str(int(x))]["minmax_theta2_s"]["mediana"]
                cel.append(_s(math.inf if m is None else m))
            print(f"   {'feix de 0.1 graus AMB friccio de prova':<38}" + "".join(f"{c:>20}" for c in cel))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(sortida, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nDesat: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
