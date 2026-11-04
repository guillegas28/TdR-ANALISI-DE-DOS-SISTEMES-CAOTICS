r"""Es mou el pivot de debo durant les preses? NOMES PRESES REALS (autoritzat el 03/10).

Puja d'esborranys/_diag_pivot.py el 06/10/2026, amb la sortida identica (criteri P1
de documentacio/pla-analisi-caos.md, «06/10 · Feina sense intervencio»).

LA IDEA. La conversio (tracker.llegeix_del_trk) fa servir la MEDIANA de la pista
`fix` com a pivot, constant. Si el pivot de debo es desplaca p, la llargada mesurada
de la barra 1 es

    L1 = |J - P_mediana| ~ L1_de_debo + p . u1        (u1: direccio de la barra 1)

Si la pista `fix` es el moviment de debo (m) mes soroll de seguiment (eta):
    var(L1 amb la mediana) - var(L1 fotograma a fotograma) = var(m_par) - var(eta_par)
o sigui, si el pivot fotograma a fotograma TREU variancia, el pivot es mou mes que el
soroll de la seva pista. I sense saber eta: sigma(m_par) = sqrt(cov(dL1, p_par)).
El pendent de dL1 contra p_par es nomes un diagnostic: la dilucio de qualsevol
regressio amb soroll a l'eix x el fa baixar (opcio A d'en Guillem, 03/10).

EL RESULTAT (03/10): el pivot NO es mou, o gairebe gens: com a molt 0,05 graus de
theta1 en mediana, contra l'1,7 graus que varia L1 als 150. L'error en moviment es
del seguiment de les marques que es mouen.

Criteris de l'autoprova (pla, 03/10; V1 canviada amb l'opcio A despres de fallar):
  V1  un pivot fabricat que es mou de debo: R2 > 0,9 i el pivot fotograma a fotograma
      treu > 90 % de la variancia de L1;
  V2  un pivot quiet amb soroll a les pistes: |pendent| < 0,1 i R2 < 0,05;
  V3  sobre una presa real, de dades/ nomes obre el .trk i el JSON.

Us:
    .venv\\Scripts\\python eines\\moviment_del_pivot.py --autoprova
    .venv\\Scripts\\python eines\\moviment_del_pivot.py --json dades/divergencia/pivot_2026-10-06.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import tracker, trk  # noqa: E402

NET = ARREL / "dades" / "net"
INDEX = NET / "preses.json"
DES_DE_S, FINS_A_S = 0.1, 10.0
PERDUT_PX = 10.0   # el fix a mes d'aixo de la mediana: el seguidor l'ha perdut


def posicions(meta):
    """(t, t_fix, J, P_fix, P_mediana, n) als fotogrames amb articulacio i fix. Obre el .trk i prou."""
    d = trk.retalla(trk.llegeix_trk(ARREL / meta["fitxer_trk"]), meta.get("retall_des_del_fotograma"))
    k, A, G, _, (k_fix, P) = trk.tres_pistes(d)
    P_med = np.median(P, axis=0)               # com a llegeix_del_trk
    ma = meta["geometria"]["marca_articulacio"]
    J = tracker.corregeix_marca(P_med, A, ma["delta_px"]) if ma.get("corregida") else A
    t = (k - k[0]) * d["delta_t_s"]
    _, ia, ib = np.intersect1d(k, k_fix, return_indices=True)
    return t, t[ia], J[ia], P[ib], P_med, len(k)


def analitza(J, P, P_med) -> dict:
    p = P - P_med
    bons = np.hypot(p[:, 0], p[:, 1]) <= PERDUT_PX
    u = J - P_med
    L = np.hypot(u[:, 0], u[:, 1])
    u1 = u / L[:, None]
    n1 = np.column_stack([-u1[:, 1], u1[:, 0]])
    par = np.sum(p * u1, axis=1)
    perp = np.sum(p * n1, axis=1)
    dL = L - np.median(L[bons])
    X = np.column_stack([np.ones(bons.sum()), par[bons]])
    coef = np.linalg.lstsq(X, dL[bons], rcond=None)[0]
    res = dL[bons] - X @ coef
    r2 = 1.0 - res.var() / dL[bons].var()
    Lf = np.hypot(*(J - P).T)
    reduccio = 1.0 - Lf[bons].var() / L[bons].var()
    Lm = float(np.median(L[bons]))
    # El moviment de debo del pivot, sense saber el soroll de la pista fix:
    # dL1 = m_par + soroll de J i p_par = m_par + soroll del fix, independents,
    # o sigui cov(dL1, p_par) = var(m_par).
    cov = float(np.cov(dL[bons], par[bons])[0, 1])
    sigma_m = math.sqrt(max(cov, 0.0))
    return {"fotogrames": int(len(L)), "perduts": int((~bons).sum()),
            "sigma_L1_px": float(L[bons].std()), "sigma_p_paral_px": float(par[bons].std()),
            "sigma_p_perp_px": float(perp[bons].std()), "pendent": float(coef[1]), "r2": float(r2),
            "reduccio_variancia_L1": float(reduccio),
            "cov_dL_ppar_px2": cov, "sigma_moviment_px": sigma_m,
            "error_theta1_moviment_graus": math.degrees(sigma_m / Lm),
            "error_theta1_graus": math.degrees(float(perp[bons].std()) / Lm),
            "sigma_L1_graus": math.degrees(float(L[bons].std()) / Lm)}


def mesura_presa(nom: str) -> dict | None:
    meta = json.loads((NET / f"{nom}.json").read_text(encoding="utf-8"))
    a = meta.get("amollada") or {}
    if not a.get("hi_ha_amollada"):
        return None
    t_tot, t, J, P, P_med, n_tot = posicions(meta)
    z = float(a["amollada_s"])
    sel = (t >= z + DES_DE_S) & (t <= z + FINS_A_S)
    n_fin = int(((t_tot >= z + DES_DE_S) & (t_tot <= z + FINS_A_S)).sum())
    if sel.sum() < 100:
        return {"cobertura": float(sel.sum() / max(n_fin, 1)), "motiu": "el fix no cobreix la finestra"}
    r = analitza(J[sel], P[sel], P_med)
    r["cobertura"] = float(sel.sum() / n_fin)
    return r


# --------------------------------------------------------------- l'autoprova

_OBERTS = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    global _OBERTS
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    from pendol.fisica import Parametres
    from pendol.simulador import simula
    p_mod = Parametres.des_de_json(ARREL / "parametres.json")
    t, y = simula(p_mod, np.radians([150.0, 150.0, 0.0, 0.0]), temps=10.0, fps=240.0)
    th1 = y[0]
    rng = np.random.default_rng(20261006)
    n = len(t)
    P0 = np.array([600.0, 200.0])
    Jrel = np.column_stack([195.0 * np.sin(th1), 195.0 * np.cos(th1)])
    print("Autoprova de eines/moviment_del_pivot.py (V1-V3, escrits al pla el 03/10)\n")

    print("V1. Un pivot que es mou de debo (3 px, 2-8 Hz, deriva), seguit amb 0,3 px de soroll:")
    print("    R2 > 0,9 i el pivot fotograma a fotograma treu > 90 % de la variancia. El pendent")
    print("    ja no es criteri (opcio A d'en Guillem, 03/10, despres de veure'l fallar a 0,943)")
    w = 2 * np.pi
    pv = np.column_stack([1.5 * np.sin(w * 2.3 * t) + 1.0 * np.sin(w * 5.1 * t + 1) + 0.8 * np.sin(w * 7.7 * t + 2) + 0.1 * t,
                          1.2 * np.sin(w * 3.1 * t + 0.5) + 1.1 * np.sin(w * 6.3 * t + 2.5) + 0.6 * np.sin(w * 2.0 * t) - 0.1 * t])
    Ptrue = P0 + pv
    fix = Ptrue + rng.normal(0.0, 0.3, (n, 2))
    r = analitza(Ptrue + Jrel, fix, np.median(fix, axis=0))
    mira(r["r2"] > 0.9 and r["reduccio_variancia_L1"] > 0.9,
         f"R2 {r['r2']:.3f}, treu el {100 * r['reduccio_variancia_L1']:.1f} %; diagnostic: pendent "
         f"{r['pendent']:.3f}, moviment estimat {r['sigma_moviment_px']:.2f} px (de debo "
         f"{np.std(np.sum((pv - np.median(pv, axis=0)) * ((Jrel) / 195.0), axis=1)):.2f} px al llarg)")

    print("\nV2. Un pivot quiet; 1 px de soroll a la pista fix i 1 px a l'articulacio: |pendent| < 0,1, R2 < 0,05")
    fix = P0 + rng.normal(0.0, 1.0, (n, 2))
    r = analitza(P0 + Jrel + rng.normal(0.0, 1.0, (n, 2)), fix, np.median(fix, axis=0))
    mira(abs(r["pendent"]) < 0.1 and r["r2"] < 0.05,
         f"pendent {r['pendent']:+.3f}, R2 {r['r2']:.4f}, el pivot fotograma a fotograma 'treu' el "
         f"{100 * r['reduccio_variancia_L1']:.0f} % (negatiu: hi afegeix soroll)")

    print("\nV3. Sobre una presa real (150_3), de dades/ nomes obre el .trk i el JSON")
    sys.addaudithook(_registra)
    _OBERTS = []
    mesura_presa("150_3")
    oberts, _OBERTS = _OBERTS, None
    arrel = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve()) for x in oberts if Path(x).resolve().is_relative_to(arrel)})
    esperats = sorted(str(x.resolve()) for x in (ARREL / "dades/video/tracker/150/150_3.trk", NET / "150_3.json"))
    mira(dins == esperats, "obre: " + ", ".join(str(Path(x).relative_to(arrel)) for x in dins))
    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK")
    return 0


def principal(json_sortida: Path | None = None) -> int:
    index = json.loads(INDEX.read_text(encoding="utf-8"))["preses"]
    files = {}
    print(f"Finestra {DES_DE_S}-{FINS_A_S} s des de l'amollada; fix a > {PERDUT_PX:g} px de la mediana = perdut.")
    print("pendent +1: el fix segueix un pivot que es mou de debo; 0: el fix es soroll.\n")
    print("treu > 0: el pivot es mou mes que el soroll de la seva pista. s_mov = sqrt(cov(dL1, p_par)):")
    print("el moviment de debo al llarg de la barra, i 'err mov' l'error de theta1 que implicaria.\n")
    print(f"{'presa':>7} {'cobert':>6} {'perduts':>7} {'sL1 px':>7} {'sp_par':>6} {'sp_perp':>7} "
          f"{'pendent':>7} {'R2':>5} {'treu':>6} {'s_mov px':>8} {'err mov':>7} {'sL1 graus':>9}")
    for f in index:
        if f.get("fora_del_balanc"):
            continue
        r = mesura_presa(f["nom"])
        if r is None:
            continue
        if "motiu" in r:
            print(f"{f['nom']:>7} {100 * r['cobertura']:5.0f}%  {r['motiu']}")
            continue
        files[f["nom"]] = (f"{f['grup_graus']:.0f}", r)
        print(f"{f['nom']:>7} {100 * r['cobertura']:5.0f}% {r['perduts']:7d} {r['sigma_L1_px']:7.2f} "
              f"{r['sigma_p_paral_px']:6.2f} {r['sigma_p_perp_px']:7.2f} {r['pendent']:+7.3f} {r['r2']:5.2f} "
              f"{100 * r['reduccio_variancia_L1']:5.0f}% {r['sigma_moviment_px']:8.2f} "
              f"{r['error_theta1_moviment_graus']:7.3f} {r['sigma_L1_graus']:9.3f}")
    print()
    grups = {}
    for g in ("45", "90", "150"):
        rs = [r for gg, r in files.values() if gg == g]
        m = lambda k: float(np.median([r[k] for r in rs]))  # noqa: E731
        grups[g] = {k: m(k) for k in ("pendent", "r2", "reduccio_variancia_L1", "sigma_p_paral_px",
                                      "sigma_p_perp_px", "sigma_moviment_px", "error_theta1_moviment_graus",
                                      "sigma_L1_graus")}
        grups[g]["preses"] = len(rs)
        grups[g]["positives"] = int(sum(r["reduccio_variancia_L1"] > 0 for r in rs))
        print(f"Grup {g} ({len(rs)}): pendent {m('pendent'):+.3f}, R2 {m('r2'):.2f}, el pivot fotograma a "
              f"fotograma treu el {100 * m('reduccio_variancia_L1'):.0f} % (positiu a "
              f"{sum(r['reduccio_variancia_L1'] > 0 for r in rs)} de {len(rs)}); pista fix: p_par "
              f"{m('sigma_p_paral_px'):.2f} px, p_perp {m('sigma_p_perp_px'):.2f} px; moviment de debo "
              f"{m('sigma_moviment_px'):.2f} px = {m('error_theta1_moviment_graus'):.3f} graus de theta1 "
              f"(sigma L1 {m('sigma_L1_graus'):.3f} graus)")
    if json_sortida:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(
            {"eina": "eines/moviment_del_pivot.py", "pla": "documentacio/pla-analisi-caos.md",
             "que_es": "es mou el pivot? pista fix fotograma a fotograma contra la mediana, 0,1-10 s despres "
                       "de l'amollada; nomes preses reals",
             "per_presa": {n: {"grup": g, **r} for n, (g, r) in files.items()}, "per_grup": grups},
            indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {json_sortida}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    return principal(Path(args.json) if args.json else None)


if __name__ == "__main__":
    sys.exit(main())
