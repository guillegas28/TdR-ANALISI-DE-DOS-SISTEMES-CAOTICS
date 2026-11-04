r"""El centre del cercle de cada presa: l'eix de gir de debo vist per la camera (25/10/2026).

Puja d'esborranys/_prova_cercle.py (24/10) amb el mateix ajust i la mateixa sortida (criteri T6a de
documentacio/pla-correccio-biaix.md, «La T6 del cercle»). A cada presa s'ajusta un cercle a la trajectoria
crua de la marca de l'articulacio, nomes en moviment (de l'amollada + 0,1 s fins a 2,5 s abans del final).
Delta = centre - pivot de la conversio (la mediana de la pista fix), en px de la imatge (y avall).

La sortida, dades/versemblant/centres_cercle.json (versionada), porta per a cada presa el que necessita la
part A de comparar.corregeix_dues_parts, al marc de la conversio (x dreta, y amunt, en unitats de L1):
    d       (centre - pivot) / L1
    rho_g   |extrem - pivot| / L1 al repos final
    r       el radi del cercle / L1
La correccio d'una presa real es fa SEMPRE amb comparar.corregeix_presa, que llegeix aquest fitxer.

Nomes llegeix els .trk, dades/retalls.json i les metadades de dades/net/; cap simulacio (llevat de la C0).

    .venv\Scripts\python.exe eines\centre_del_cercle.py              # les 35 preses -> el JSON
    .venv\Scripts\python.exe eines\centre_del_cercle.py --autoprova  # la C0, amb preses fabricades
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
from pendol import dades, trk  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402

SORTIDA = ARREL / "dades" / "versemblant" / "centres_cercle.json"
CUA_S = 2.0                  # el repos final, com la conversio (meta["geometria"]["cua_s"])
FORA_DEL_FINAL_S = 2.5       # el cercle no fa servir el repos final
DES_DE_AMOLLADA_S = 0.1
BLOC_S = 1.0                 # bootstrap per blocs
N_BOOT = 200


# ------------------------------------------------------------ l'ajust

def ajusta_cercle(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa (lineal) i despres geometric. Torna (centre, radi)."""
    x, y = X[:, 0], X[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    c0 = np.array([-D / 2, -E / 2])
    r0 = math.sqrt(max(c0 @ c0 - F, 1e-9))
    res = least_squares(lambda q: np.hypot(x - q[0], y - q[1]) - q[2], [c0[0], c0[1], r0])
    return res.x[:2], float(res.x[2])


def angle_entre(v, a) -> float:
    """L'angle de a respecte de v, en graus."""
    return math.degrees(math.atan2(v[0] * a[1] - v[1] * a[0], v[0] * a[0] + v[1] * a[1]))


def analitza(t, A, G, P, ta, fps) -> dict:
    """El cercle d'una presa (posicions en px) i els numeros de la part A de la correccio."""
    mou = (t >= ta + DES_DE_AMOLLADA_S) & (t <= t[-1] - FORA_DEL_FINAL_S)
    C, R = ajusta_cercle(A[mou])
    d = C - P
    # Bootstrap per blocs d'1 s sobre el tram en moviment.
    idx = np.nonzero(mou)[0]
    nb = max(2, int(round(BLOC_S * fps)))
    blocs = [idx[i:i + nb] for i in range(0, len(idx), nb)]
    rng = np.random.default_rng(24102026)
    cs = []
    for _ in range(N_BOOT):
        tria = rng.integers(0, len(blocs), len(blocs))
        cs.append(ajusta_cercle(A[np.concatenate([blocs[j] for j in tria])])[0])
    sig = np.std(np.array(cs), axis=0, ddof=1)
    n = int(round(CUA_S * fps))
    a, g = A[-n:].mean(0), G[-n:].mean(0)
    v = g - P
    u = v / np.linalg.norm(v)
    lat = np.array([-u[1], u[0]])
    L1 = float(np.linalg.norm(a - P))
    return {"delta_px": d.tolist(), "sigma_px": sig.tolist(), "radi_px": R, "L1_px": L1,
            "delta_lateral_graus": math.degrees(math.atan2(float(d @ lat), L1)),
            "delta_vertical_graus": math.degrees(math.atan2(float(d @ u), L1)),
            "r1_graus": angle_entre(v, a - P),
            "r1_Hp_graus": angle_entre(g - C, a - C),
            "r1_Hm_graus": angle_entre(v, a - d - P),
            # la part A, al marc de la conversio (x dreta, y amunt), en unitats de L1
            "part_a": {"d": [float(d @ np.array([u[1], -u[0]])) / L1, -float(d @ u) / L1],
                       "rho_g": float(np.linalg.norm(v)) / L1, "r": R / L1}}


def llegeix(nom: str, retalls: dict):
    g = nom.split("_")[0]
    d = trk.retalla(trk.llegeix_trk(ARREL / "dades" / "video" / "tracker" / g / f"{nom}.trk"),
                    retalls.get(nom, {}).get("des_del_fotograma"))
    k, A, G, _noms, (_kf, Pf) = trk.tres_pistes(d)
    t = (k - k[0]) * d["delta_t_s"]
    return t, np.asarray(A, float), np.asarray(G, float), np.median(Pf, axis=0), 1.0 / d["delta_t_s"]


def noms_de_les_preses() -> list[str]:
    return sorted([f.stem for f in (ARREL / "dades" / "net").glob("*_*.json")
                   if f.stem.split("_")[0] in ("45", "90", "150") and f.stem != "45_7"],
                  key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))


# ------------------------------------------------------------ C0: l'instrument

def c0() -> list[str]:
    """La C0 de la prova del cercle (24/10), amb preses fabricades."""
    from pendol.simulador import simula
    p = Parametres.des_de_json(ARREL / "parametres.json")
    fps, s = 240.0, 200.0 / p.L1                          # px/m
    gir = math.radians(0.3)
    cg, sg = math.cos(gir), math.sin(gir)
    P0 = np.array([500.0, 300.0])

    def a_imatge(x, y):                                     # fisic (y amunt) -> imatge (y avall), girada
        return np.column_stack([P0[0] + s * (cg * x + sg * y), P0[1] + s * (sg * x - cg * y)])

    mal = []
    rng = np.random.default_rng(7)
    for graus in (150.0, 45.0):
        t, y = simula(p, np.radians([graus, graus, 0.0, 0.0]), 12.0, fps=fps)
        nr = int(round(3.0 * fps))
        t = np.concatenate([t, t[-1] + np.arange(1, nr + 1) / fps])
        th1 = np.concatenate([y[0], np.zeros(nr)])
        th2 = np.concatenate([y[1], np.zeros(nr)])
        xj, yj = p.L1 * np.sin(th1), -p.L1 * np.cos(th1)
        xg, yg = xj + p.r_gomet * np.sin(th2), yj - p.r_gomet * np.cos(th2)
        Aj, Gt = a_imatge(xj, yj), a_imatge(xg, yg)
        soroll = lambda X: X + rng.normal(0.0, 1.0, X.shape)    # noqa: E731
        ta = 0.0
        # (a) el pivot marcat (+3, -2) px fora de l'eix
        r = analitza(t, soroll(Aj), soroll(Gt), P0 + np.array([3.0, -2.0]), ta, fps)
        e = np.array(r["delta_px"]) - np.array([-3.0, 2.0])
        # 24/10: el C0a deia |e| < 0,3 px en total i va fallar als 45 (0,58 px, tot vertical: un arc de +-45
        # graus no fixa el centre amunt-avall). Primer canvi: la vertical dins de 2 sigma del bootstrap; va
        # tornar a fallar (el bootstrap la subestima, tambe als 150). Segon canvi, tots dos DESPRES de fallar i
        # per decisio d'en Guillem: de costat (x, la camera nomes es gira 0,3 graus) < 0,3 px i el repos
        # corregit < 0,05 graus. La vertical, nomes informativa (la seva precisio amb el soroll real la mesura
        # esborranys/_precisio_centre.py). Vegeu documentacio/pla-correccio-biaix.md.
        ok_a = abs(e[0]) < 0.3 and abs(r["r1_Hp_graus"]) < 0.05
        if not ok_a:
            mal.append(f"C0a {graus:g}: Delta a ({e[0]:+.2f}, {e[1]:+.2f}) px de la veritat (sigma vertical "
                       f"{r['sigma_px'][1]:.2f}), r1 amb Hp {r['r1_Hp_graus']:+.3f} graus")
        print(f"   C0a {graus:g}: pivot fora de l'eix -> Delta a ({e[0]:+.3f}, {e[1]:+.3f}) px (sigma "
              f"{r['sigma_px'][0]:.3f}, {r['sigma_px'][1]:.3f}), r1 {r['r1_graus']:+.3f} -> Hp "
              f"{r['r1_Hp_graus']:+.4f} graus")
        # (b) la marca desplacada (+3, +1) px, fixa a la imatge
        r = analitza(t, soroll(Aj + np.array([3.0, 1.0])), soroll(Gt), P0, ta, fps)
        if abs(r["r1_Hm_graus"]) >= 0.05:
            mal.append(f"C0b {graus:g}: r1 amb Hm {r['r1_Hm_graus']:+.3f} graus")
        print(f"   C0b {graus:g}: marca fixa a la imatge -> r1 {r['r1_graus']:+.3f} -> Hm {r['r1_Hm_graus']:+.4f} graus")
        # (c) la marca 3 px de costat, enganxada a la barra 1
        lat = np.column_stack([np.cos(th1), np.sin(th1)]) * (3.0 / s)
        Ac = a_imatge(xj + lat[:, 0], yj + lat[:, 1])
        r = analitza(t, soroll(Ac), soroll(Gt), P0, ta, fps)
        dd = float(np.hypot(*r["delta_px"]))
        if dd >= 0.3:
            mal.append(f"C0c {graus:g}: |Delta| = {dd:.2f} px amb la marca enganxada")
        print(f"   C0c {graus:g}: marca enganxada a la barra -> |Delta| {dd:.3f} px, r1 {r['r1_graus']:+.3f} graus")
    return mal


# ------------------------------------------------------------ les preses reals

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autoprova", action="store_true", help="la C0, amb preses fabricades")
    args = ap.parse_args(argv)
    print("C0, l'instrument (preses fabricades)")
    mal = c0()
    if mal:
        print("\nC0 FALLA: " + "; ".join(mal) + ("" if args.autoprova else "\nNo es mira cap presa real."))
        return 1
    print("   C0 passa")
    if args.autoprova:
        print("OK -> l'ajust troba el pivot fora de l'eix de costat a < 0,3 px, i el repos que en surt a < 0,05 graus.")
        return 0
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    fora = {}
    print(f"\n{'presa':7}{'Dx px':>7}{'Dy px':>7}{'sig':>5}{'lat':>7}{'vert':>7}   (graus sobre L1)")
    for n in noms_de_les_preses():
        _, meta = dades.carrega(ARREL / "dades" / "net" / f"{n}.csv")
        t, A, G, P, fps = llegeix(n, retalls)
        r = analitza(t, A, G, P, meta["amollada"]["amollada_s"], fps)
        fora[n] = r
        print(f"{n:7}{r['delta_px'][0]:+7.2f}{r['delta_px'][1]:+7.2f}{max(r['sigma_px']):5.2f}"
              f"{r['delta_lateral_graus']:+7.2f}{r['delta_vertical_graus']:+7.2f}")
    SORTIDA.write_text(json.dumps({
        "eina": "eines/centre_del_cercle.py", "pla": "documentacio/pla-correccio-biaix.md (24-25/10)",
        "que_es": "el centre del cercle de la marca de l'articulacio (l'eix de debo) de cada presa, i la part A "
                  "de comparar.corregeix_dues_parts al marc de la conversio (d, rho_g, r en unitats de L1)",
        "preses": fora}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {SORTIDA.relative_to(ARREL)} ({len(fora)} preses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
