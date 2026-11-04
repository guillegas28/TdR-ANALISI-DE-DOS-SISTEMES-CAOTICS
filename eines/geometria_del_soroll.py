r"""Quanta part del soroll en moviment es geometria, sense limit d'harmonics. NOMES PRESES REALS.

Puja d'esborranys/_diag_geometria_completa.py el 06/10/2026, amb la sortida identica
(criteri P1 de documentacio/pla-analisi-caos.md, «06/10 · Feina sense intervencio»).

EL RESULTAT (03/10): la geometria fixa (que s'anul.laria entre tirs) explica fora de
mostra el 2-9 % de L1 i el 19-62 % de L2, i el terra NO baixa: 1,38 / 4,26 / 9,29
graus per parella (45 / 90 / 150), amb un metode que el sobreestima ~12 % (G3, G4).

eines/soroll_en_moviment.py nomes treu la geometria fins al segon harmonic (D1).
Aqui: e1 com a funcio QUALSEVOL de theta1 (L1 nomes depen de theta1: tot el que
en depengui es geometria), i e2 com a suma d'una funcio de theta2 - theta1 i
una de theta2. FORA DE MOSTRA: blocs d'1 s des de l'amollada, alternats; la
geometria s'ajusta als parells i s'avalua als senars, i a l'inreves. El soroll
lent no pot fer d'explicacio fora de mostra; la geometria, que es la mateixa a
tots els blocs, si.

HISTORIA (03/10). La primera versio feia servir Fourier fins a l'harmonic 10.
Va passar G1-G3 i va EXPLOTAR sobre les preses de 45 i 90 graus: alla theta1
nomes cobreix uns quants graus, i 21 termes ajustats en un rang curt
extrapolen disbarats. Ara son MITJANES PER FRANGES de 5 graus (a L2, un model
additiu per backfitting), i una franja amb menys de MIN_MOSTRES a l'altra
meitat no es corregeix: el que no es pot predir compta com a soroll. G4 (sobre
un pendol de 45 graus) es va escriure al pla abans de canviar-ho.

Criteris G1-G4 a documentacio/pla-analisi-caos.md («Cinc feines sense
intervencio»), escrits abans d'escriure i de canviar el codi.

    .venv\\Scripts\\python eines\\geometria_del_soroll.py --autoprova
    .venv\\Scripts\\python eines\\geometria_del_soroll.py --json dades/divergencia/geometria_2026-10-06.json
"""
from __future__ import annotations

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
import soroll_en_moviment as sm  # noqa: E402

FRANJA = math.radians(5.0)
N_FRANGES = 72
MIN_MOSTRES = 24     # 0,1 s de mostres a l'altra meitat, com a minim
ITERACIONS = 10      # backfitting de L2
BLOC_S = 1.0         # blocs alternats


def franges(x):
    return (np.floor(((np.asarray(x) + np.pi) % (2 * np.pi)) / FRANJA).astype(int)) % N_FRANGES


def geometria_fora_de_mostra(c, amollada_s):
    """La e menys la geometria predita fora de mostra (des de l'amollada)."""
    t = c["t"]
    th1, th2 = c["th"]
    mou = t >= amollada_s
    bloc = np.floor((t - amollada_s) / BLOC_S).astype(int) % 2
    variables = ([franges(th1)], [franges(th2 - th1), franges(th2)])
    resid = c["e"].copy()
    for i, idxs in enumerate(variables):
        e = c["e"][i]
        pred = np.zeros_like(e)
        for b in (0, 1):
            ajust = mou & (bloc == b)
            aval = mou & (bloc != b)
            y = e[ajust] - e[ajust].mean()
            ix = [v[ajust] for v in idxs]
            comptes = [np.bincount(v, minlength=N_FRANGES) for v in ix]
            efectes = [np.zeros(N_FRANGES) for _ in ix]
            for _ in range(ITERACIONS if len(ix) > 1 else 1):
                for k in range(len(ix)):
                    parcial = y - sum(efectes[m][ix[m]] for m in range(len(ix)) if m != k)
                    s = np.bincount(ix[k], weights=parcial, minlength=N_FRANGES)
                    efectes[k] = np.where(comptes[k] >= MIN_MOSTRES, s / np.maximum(comptes[k], 1), 0.0)
            pred[aval] = sum(efectes[k][idxs[k][aval]] for k in range(len(ix)))
        resid[i] = e - pred
    return resid


def mesura(c, amollada_s, w0, fin, ordre, fins_a=sm.FINS_A_S):
    t, fps = c["t"], c["fps"]
    sel = (t >= amollada_s + sm.DES_DE_S) & (t <= amollada_s + fins_a)
    r = geometria_fora_de_mostra(c, amollada_s)
    frac = [float(1.0 - (r[i][sel] - r[i][sel].mean()).var() / (c["e"][i][sel] - c["e"][i][sel].mean()).var())
            for i in range(2)]
    s, d = sm.filtra(c["e"], fin, ordre, fps)
    comp, sd = sm.components(s, d, sel, w0)
    sr, dr = sm.filtra(r, fin, ordre, fps)
    comp_g, sd_g = sm.components(sr, dr, sel, w0)
    return {"fraccio_fora_de_mostra": frac, "sigma_d_total": sd, "sigma_d_sense_geometria": sd_g,
            "components_sense_geometria": comp_g, "components_total": comp}


def _dif(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _prova(mira, nom, angle, bony_graus, p, w0, llavor):
    """G1-G3 (o G4) sobre un pendol fabricat amollat d'`angle` graus."""
    fps, fin, ordre = 240.0, 13, 3
    from pendol.simulador import simula
    t, y = simula(p, np.radians([angle, angle, 0.0, 0.0]), temps=70.0, fps=fps)
    th1, th2 = y[0], y[1]
    J0, G0 = sm._posicions(th1, th2)
    n = len(t)
    rng = np.random.default_rng(llavor)
    rng_g = np.random.default_rng(7)
    g1 = sum(rng_g.uniform(-1, 1) * 0.004 / k * np.cos(k * th1 + rng_g.uniform(0, 6.3)) for k in range(1, 7))
    g1 = g1 + 0.006 * np.exp(-0.5 * (_dif(th1, math.radians(bony_graus)) / math.radians(15.0)) ** 2)
    g2 = sum(rng_g.uniform(-1, 1) * 0.004 / k * np.cos(k * (th2 - th1) + rng_g.uniform(0, 6.3)) for k in range(1, 7))
    g2 = g2 + sum(rng_g.uniform(-1, 1) * 0.003 / k * np.cos(k * th2 + rng_g.uniform(0, 6.3)) for k in range(1, 7))

    def amb_llargades(f1, f2):
        J = J0 * (1.0 + f1)[:, None]
        u2 = (G0 - J0) / 227.0
        return J, J + 227.0 * (1.0 + f2)[:, None] * u2

    def ar1(rho, sigma):
        z = rng.normal(0.0, sigma, n)
        x = np.empty(n)
        x[0] = z[0]
        for k in range(1, n):
            x[k] = rho * x[k - 1] + math.sqrt(1 - rho * rho) * z[k]
        return x

    print(f"   (pendol fabricat {angle:g}/{angle:g} graus, 70 s: theta1 de "
          f"{math.degrees(th1.min()):.0f} a {math.degrees(th1.max()):.0f} graus)")
    m = mesura(sm.cadena(t, *amb_llargades(g1, g2)), 0.0, w0, fin, ordre)
    f = m["fraccio_fora_de_mostra"]
    mira(min(f) > 0.90, f"{nom} geometria sola: explica {100 * f[0]:.2f} % (L1) i {100 * f[1]:.2f} % (L2)")
    s1, s2 = ar1(0.99, 0.012), ar1(0.99, 0.012)
    m = mesura(sm.cadena(t, *amb_llargades(s1, s2)), 0.0, w0, fin, ordre)
    f = m["fraccio_fora_de_mostra"]
    sd_soroll = m["sigma_d_total"]
    if nom == "G1-G3":
        mira(max(f) < 0.05, f"G2 soroll lent sol (rho 0,99): explica {100 * f[0]:.2f} % (L1) i "
                            f"{100 * f[1]:.2f} % (L2)")
    m = mesura(sm.cadena(t, *amb_llargades(g1 + s1, g2 + s2)), 0.0, w0, fin, ordre)
    rel = m["sigma_d_sense_geometria"] / sd_soroll - 1
    mira(abs(rel) < 0.15, f"{'G3' if nom == 'G1-G3' else 'G4'} geometria + soroll: sense geometria "
                          f"{math.degrees(m['sigma_d_sense_geometria']):.3f} graus, soroll sol "
                          f"{math.degrees(sd_soroll):.3f} ({100 * rel:+.1f} %); total "
                          f"{math.degrees(m['sigma_d_total']):.3f}")


def autoprova() -> int:
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    print("Autoprova de eines/geometria_del_soroll.py (G1-G4, escrits al pla el 03/10)")
    print(f"Franges de {math.degrees(FRANJA):g} graus, minim {MIN_MOSTRES} mostres a l'altra meitat; "
          "finestra 0,1-10 s\n")
    print("G1-G3, a 150 graus (theta1 fa voltes): geometria > 90 %, soroll sol < 5 %, sigma a < 15 %")
    _prova(mira, "G1-G3", 150.0, 40.0, p, w0, 20261004)
    print("\nG4, a 45 graus (theta1 en un rang curt): geometria > 90 %, sigma a < 15 %")
    _prova(mira, "G4", 45.0, 20.0, p, w0, 20261005)
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
    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    index = json.loads(sm.INDEX.read_text(encoding="utf-8"))["preses"]
    d1 = json.loads((ARREL / "dades/divergencia/soroll_moviment_2026-10-03.json").read_text(encoding="utf-8"))
    preses = {}
    print(f"Franges de {math.degrees(FRANJA):g} graus, fora de mostra amb blocs alternats de {BLOC_S:g} s.")
    print("sigma_d en graus; 'D1' es la xifra sense geometria d'eines/soroll_en_moviment.py (2 harmonics)\n")
    print(f"{'presa':>7}  {'fora mostra L1':>14} {'L2':>6}   {'total':>6} {'D1':>6} {'completa':>8}")
    for fila in index:
        if fila.get("fora_del_balanc"):
            continue
        meta = json.loads((sm.NET / f"{fila['nom']}.json").read_text(encoding="utf-8"))
        a = meta.get("amollada") or {}
        if not a.get("hi_ha_amollada"):
            continue
        t_fot, J, G, _ = sm.posicions_del_trk(meta)
        c = sm.cadena(t_fot, J, G, bool(meta.get("signe_invertit", False)))
        sg = meta["savgol"]
        m = mesura(c, float(a["amollada_s"]), w0, int(sg["finestra_mostres"]), int(sg["ordre"]))
        ref = d1["per_presa"][fila["nom"]]
        preses[fila["nom"]] = {"grup": f"{fila['grup_graus']:.0f}", "angles": a["angles_amollada_graus"],
                               "sd": math.degrees(m["sigma_d_sense_geometria"]),
                               "frac": m["fraccio_fora_de_mostra"],
                               "comp": [math.degrees(x) for x in m["components_sense_geometria"]]}
        print(f"{fila['nom']:>7}  {100 * m['fraccio_fora_de_mostra'][0]:13.0f}% "
              f"{100 * m['fraccio_fora_de_mostra'][1]:5.0f}%   {math.degrees(m['sigma_d_total']):6.3f} "
              f"{ref['sense_geometria']['sigma_d_graus']:6.3f} {math.degrees(m['sigma_d_sense_geometria']):8.3f}")
    print()
    grups = {}
    for g in ("45", "90", "150"):
        noms = [n for n, v in preses.items() if v["grup"] == g]
        terres, raons = [], []
        for i, x in enumerate(noms):
            for y in noms[i + 1:]:
                tr = math.hypot(preses[x]["sd"], preses[y]["sd"])
                d0 = math.degrees(float(distancia_fases(np.radians([*preses[x]["angles"], 0, 0]),
                                                        np.radians([*preses[y]["angles"], 0, 0]), w0)))
                terres.append(tr)
                raons.append(d0 / tr)
        comp = np.median([preses[n]["comp"] for n in noms], axis=0)
        frac = np.median([preses[n]["frac"] for n in noms], axis=0)
        ref = d1["per_grup"][g]
        print(f"Grup {g}: fora de mostra L1 {100 * frac[0]:.0f} %, L2 {100 * frac[1]:.0f} %;  components "
              f"(th1 th2 w1 w2) {comp[0]:.3f} {comp[1]:.3f} {comp[2]:.3f} {comp[3]:.3f}")
        print(f"   terra parella: total {ref['terra_parella_total_graus']['mediana']:.3f}, D1 "
              f"{ref['terra_parella_sense_geometria_graus']['mediana']:.3f}, completa "
              f"{np.median(terres):.3f} ({min(terres):.3f}-{max(terres):.3f});  d0/terra completa "
              f"{np.median(raons):.2f} ({min(raons):.2f}-{max(raons):.2f}), >1: "
              f"{100 * np.mean(np.array(raons) > 1):.0f} %, >3: {100 * np.mean(np.array(raons) > 3):.0f} %")
        grups[g] = {"preses": noms, "fraccio_fora_de_mostra_mediana": [float(x) for x in frac],
                    "components_sense_geometria_mediana_graus": [float(x) for x in comp],
                    "terra_parella_graus": {"mediana": float(np.median(terres)), "min": float(min(terres)),
                                            "max": float(max(terres))},
                    "rao_d0_terra": {"mediana": float(np.median(raons)), "min": float(min(raons)),
                                     "max": float(max(raons))},
                    "fraccio_d0_per_sobre_del_terra": float(np.mean(np.array(raons) > 1)),
                    "fraccio_d0_per_sobre_de_3_terres": float(np.mean(np.array(raons) > 3))}
    if json_sortida:
        json_sortida.parent.mkdir(parents=True, exist_ok=True)
        json_sortida.write_text(json.dumps(
            {"eina": "eines/geometria_del_soroll.py", "pla": "documentacio/pla-analisi-caos.md",
             "que_es": "el soroll en moviment sense la geometria fixa (franges de 5 graus, fora de mostra amb "
                       "blocs alternats d'1 s); el metode el sobreestima ~12 %; nomes preses reals",
             "per_presa": preses, "per_grup": grups}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {json_sortida}")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    return principal(Path(args.json) if args.json else None)


if __name__ == "__main__":
    sys.exit(main())
