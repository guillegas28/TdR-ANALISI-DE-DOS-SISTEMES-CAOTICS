r"""La Z del deure 4: el ritme de separacio real contra el simulat, a la MATEIXA finestra.

DECIDIT PER EN GUILLEM EL 04/10, NO FIRMAT (documentacio/pla-analisi-caos.md, «On som»):
la Z es la FTLE de la mateixa finestra -el pendent de <ln(d/d0)> amb la regla 2-, real
contra simulada des de les condicions inicials mesurades de cada tir, per grup.
    passa         si les dues lambdes son compatibles a 2 sigma del bootstrap per tirs;
    no avaluable  si l'error de qualsevol de les dues passa del 20 % de la seva lambda;
    no passa      altrament.
Substitueix la sigma(t) de la seccio 11 de tancar-el-treball.html.

PER QUE LA MATEIXA FINESTRA. La regla 2 mesura l'exponent de TEMPS FINIT de la seva
finestra (esborranys/_ftle_finestra.py: -4 % a +1 % sense soroll), no la lambda
asimptotica de lambda(E). I la FTLE canvia molt d'una finestra a l'altra, perque el
pendol s'estira a batzegades (esborranys/_regla2_amb_d0.py). Comparar finestres
diferents, o contra lambda(E), faria fallar un model perfecte.

EL COSTAT REAL ESTA TANCAT AMB PANY. Comparar amb preses reals demana la firma de X, Y i
Z (dades/versemblant/firma.json), i a mes l'alineacio real-real, que es una decisio
pendent del pas 5. costat_real() es nega a funcionar sense firma, abans d'obrir res.

26/10: LA Z DE DEBO ES A eines/lyapunov_real.py. La regla 2 no dona cap lambda real, o sigui
que la Z es fa amb els resultats que si que n'han donat, amb el mateix metode a cada costat:
    z_r2     la lambda* dels 150 graus (el R2, l'unic avaluable: un 13 % d'error);
    z_veins  la lambda dels veins (el R1): un 41 % d'error, «no avaluable» per construccio.
Totes dues fan servir veredicte() d'aqui i el mateix pany. Pla: «26/10 · La Z de la λ*».

Criteris Z1-Z6 a documentacio/pla-analisi-caos.md («C i la preparacio de B»), escrits
el 04/10 abans d'escriure aixo.

Us:
    .venv\\Scripts\\python eines\\criteri_z.py --autoprova
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
sys.path.insert(0, str(ARREL / "eines"))

from pendol import caos  # noqa: E402

FIRMA = ARREL / "dades" / "versemblant" / "firma.json"
SIGMES = 2.0          # compatibles a 2 sigma (decidit el 04/10)
ERROR_MAXIM = 0.20    # error relatiu maxim per poder avaluar (decidit el 04/10)
REMOSTREJOS = caos.REMOSTREJOS


def corba(L, L0, tirs=None):
    """<ln(d/d0)> sobre les parelles dels tirs donats (amb repeticio, sense parelles d'un tir amb ell)."""
    n = L.shape[0]
    tirs = list(range(n)) if tirs is None else list(tirs)
    par = caos._parelles_de(tirs)
    ln_d = np.array([L[i, j] for i, j in par])
    ln_d0 = np.array([L0[i, j] for i, j in par])
    return caos.mitjana_log(ln_d, ln_d0)[0]


def lambda_finestra(t, L, L0, ti: float, tf: float, tirs=None) -> float:
    """El pendent de <ln(d/d0)> a [ti, tf], amb el mateix ajust que caos.ajusta_zona."""
    t = np.asarray(t, dtype=float)
    sel = (t >= ti) & (t <= tf)
    return float(np.polyfit(t[sel], corba(L, L0, tirs)[sel], 1)[0])


def bootstrap_finestra(t, L, L0, ti: float, tf: float, n: int = REMOSTREJOS, llavor: int = 0) -> float:
    """La desviacio de la lambda remostrejant TIRS, amb la finestra fixa."""
    rng = np.random.default_rng(llavor)
    N = L.shape[0]
    v = []
    for _ in range(n):
        tirs = rng.integers(0, N, N).tolist()
        if len(set(tirs)) < 2:
            continue
        v.append(lambda_finestra(t, L, L0, ti, tf, tirs))
    return float(np.std(v, ddof=1))


def veredicte(l_real: float, s_real: float, l_sim: float, s_sim: float) -> dict:
    """La regla de la Z, decidida el 04/10."""
    rel_r = s_real / abs(l_real) if l_real else math.inf
    rel_s = s_sim / abs(l_sim) if l_sim else math.inf
    sigma = math.hypot(s_real, s_sim)
    base = {"error_relatiu_real": rel_r, "error_relatiu_sim": rel_s,
            "diferencia": l_real - l_sim, "sigma_combinada": sigma}
    if rel_r > ERROR_MAXIM or rel_s > ERROR_MAXIM:
        return {**base, "veredicte": "no avaluable"}
    return {**base, "veredicte": "passa" if abs(l_real - l_sim) <= SIGMES * sigma else "no passa"}


def costat_simulat(p, x0s, t, ti: float, tf: float, w0: float, n: int = REMOSTREJOS, llavor: int = 0) -> dict:
    """lambda_sim i el seu error: els tirs simulats des de x0s, filtrats com la conversio."""
    import regla2_amb_soroll as rg
    from pendol.mesures import distancia_fases
    n_repos = int(round(rg.REPOS_S * rg.FPS))
    y = rg.simula_tirs(p, x0s, t)
    est = np.array([rg.estats_amb_soroll(yy, None, 0.0, None, n_repos) for yy in y])
    L = caos.matriu_ln_d(est, w0)
    L0 = np.full(L.shape[:2], -np.inf)
    for i in range(len(x0s)):
        for j in range(i + 1, len(x0s)):
            L0[i, j] = L0[j, i] = math.log(float(distancia_fases(x0s[i], x0s[j], w0)))
    return {"lambda": lambda_finestra(t, L, L0, ti, tf),
            "error": bootstrap_finestra(t, L, L0, ti, tf, n, llavor), "L": L, "L0": L0}


def costat_real(*_args, **_kwargs):
    """EL PANY. Sense la firma de X, Y i Z no es compara res real, i no s'obre res."""
    if not FIRMA.exists():
        raise SystemExit(f"Pany tancat: no hi ha {FIRMA.relative_to(ARREL)}. La Z compara simulacio i "
                         "realitat, i aixo no es fa abans de firmar X, Y i Z.")
    raise NotImplementedError("L'alineacio real-real (pas 5) esta pendent de decidir per en Guillem.")


# --------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """Z1-Z6, escrits al pla el 04/10/2026 abans d'escriure el codi (la Z5, refeta el 29/10)."""
    global _OBERTS, FIRMA
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    t = caos._T_PAR
    ti, tf = 0.2, 1.5
    nb = 300
    print("Autoprova de eines/criteri_z.py (criteris escrits al pla el 04/10/2026)\n")
    print(f"Corbes fabricades de caos._shots_fabricats, exponencial pura; finestra {ti}-{tf} s; "
          f"bootstrap de {nb} remostrejos\n")

    def un_cas(llavor, lam_a, lam_b, sd_tir=0.3):
        La, L0a = caos._shots_fabricats(llavor, lam=lam_a, sd_tir=sd_tir, soroll=False,
                                        d0_mig=0.01, sostre_log=1e9)
        Lb, L0b = caos._shots_fabricats(llavor + 5000, lam=lam_b, sd_tir=sd_tir, soroll=False,
                                        d0_mig=0.01, sostre_log=1e9)
        la, lb = lambda_finestra(t, La, L0a, ti, tf), lambda_finestra(t, Lb, L0b, ti, tf)
        sa = bootstrap_finestra(t, La, L0a, ti, tf, nb, llavor + 1)
        sb = bootstrap_finestra(t, Lb, L0b, ti, tf, nb, llavor + 2)
        return {"l_a": la, "l_b": lb, **veredicte(la, sa, lb, sb)}

    def tanda(nom, lam_a, lam_b, sd_tir=0.3):
        res = [un_cas(3000 + s, lam_a, lam_b, sd_tir) for s in range(20)]
        comptes = {v: sum(r["veredicte"] == v for r in res) for v in ("passa", "no passa", "no avaluable")}
        err = np.median([r["error_relatiu_real"] for r in res])
        print(f"   {nom}: {comptes}; error relatiu mitja {100 * err:.0f} %")
        return comptes, res

    print("Z1. La mateixa lambda (2,5 i 2,5): passa a >= 17 de 20")
    c, res1 = tanda("2,5 contra 2,5", 2.5, 2.5)
    mira(c["passa"] >= 17, f"passa a {c['passa']} de 20")

    print("\nZ2. Lambdes diferents (2,5 i 3,5): no passa a >= 18 de 20")
    c, _ = tanda("2,5 contra 3,5", 2.5, 3.5)
    mira(c["no passa"] >= 18, f"no passa a {c['no passa']} de 20")

    # Z3 CANVIADA el 04/10, DESPRES de veure-la fallar, per decisio d'en Guillem (opcio b).
    # La primera versio demanava «cap passa» amb una dispersio de 2,0, que deixa l'error
    # just al llindar: a 3 de 20 casos l'error va quedar per sota del 20 % amb les dues
    # lambdes iguals de debo, i «passa» era el veredicte correcte. El que la regla ha de
    # garantir es aixo: cap cas amb error > 20 % pot passar. Minim 5 casos per sobre del
    # llindar perque la prova no sigui buida (tambe afegit despres de mirar).
    print("\nZ3. Errors grans (dispersio entre tirs de 2,0): cap cas amb error > 20 % surt 'passa'")
    print("    (i n'hi ha d'haver com a minim 5, perque la prova no sigui buida)")
    c, res3 = tanda("2,5 contra 2,5, sd_tir 2,0", 2.5, 2.5, sd_tir=2.0)
    grans = [r for r in res3 if max(r["error_relatiu_real"], r["error_relatiu_sim"]) > ERROR_MAXIM]
    infractors = [r for r in grans if r["veredicte"] == "passa"]
    mira(len(grans) >= 5 and not infractors,
         f"{len(grans)} casos amb error > 20 %, dels quals {len(infractors)} surten 'passa'; "
         f"dels {20 - len(grans)} amb error <= 20 %, {sum(r['veredicte'] == 'passa' for r in res3 if r not in grans)} passen")

    print("\nZ4. Sobre el doble simulat (150 graus, 1,0 graus), la finestra fixa a la zona de la regla 2")
    print("    dona la mateixa lambda que la regla 2 (< 1e-12)")
    from pendol.fisica import Parametres
    from pendol.mesures import omega_zero
    import regla2_amb_soroll as rg
    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    tt = np.arange(0.0, rg.DURADA_S + 0.5 / rg.FPS, 1.0 / rg.FPS)
    x0s = rg.condicions_inicials(150.0, 1.0, rg.llavor_de(150.0, 1.0))
    s = costat_simulat(p, x0s, tt, 0.0, 0.74, w0, n=50)
    par = caos._parelles_de(list(range(len(x0s))))
    ln_d = np.array([s["L"][i, j] for i, j in par])
    ln_d0 = np.array([s["L0"][i, j] for i, j in par])
    r2 = caos.lambda_de_corbes(tt, ln_d, ln_d0, 1e-9, caos.sostre_de_saturacio(tt, ln_d))
    lf = lambda_finestra(tt, s["L"], s["L0"], r2["t_inici_s"], r2["t_fi_s"])
    mira(r2["ok"] and abs(lf - r2["lambda"]) < 1e-12,
         f"regla 2 {r2.get('lambda', float('nan')):.12f}, finestra fixa {lf:.12f} "
         f"(zona {r2.get('t_inici_s', float('nan')):.3f}-{r2.get('t_fi_s', float('nan')):.3f} s); "
         f"error del bootstrap {s['error']:.3f}")

    # Z5 CANVIADA el 29/10, DESPRES de veure-la fallar, per decisio d'en Guillem (opcio a).
    # La del 04/10 demanava que firma.json no existis, i des de la firma del 27/10 no es
    # pot complir. Ara el pany s'apunta a una firma de prova que no existeix, com la ZR6'
    # de lyapunov_real.py: prova el mateix pany sense dependre de la firma de debo.
    print("\nZ5. Sense firma (una de prova que no existeix), el costat real es nega a funcionar i no obre res de dades/net/")
    desat, prova = FIRMA, ARREL / "esborranys" / "_firma_que_no_existeix_z5.json"
    abans = prova.exists()
    sys.addaudithook(_registra)
    _OBERTS = []
    try:
        FIRMA = prova
        costat_real()
        negat = False
        motiu = "ha funcionat"
    except SystemExit as e:
        negat = True
        motiu = str(e)[:80]
    except Exception as e:  # noqa: BLE001
        negat, motiu = False, f"{type(e).__name__}: {e}"
    finally:
        FIRMA = desat
    oberts, _OBERTS = _OBERTS, None
    net = [x for x in oberts if "net" in Path(x).parts]
    mira(not abans and not prova.exists() and FIRMA == desat and negat and not net,
         f"la firma de prova {'EXISTIA' if abans else 'no existeix'}; {motiu}...; fitxers de dades/net oberts: {len(net)}")

    print("\nZ6. Reproduible bit a bit")
    mira(json.dumps(un_cas(3000, 2.5, 2.5)) == json.dumps(res1[0]), "el primer cas de Z1, refet, es identic")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> la regla de la Z aprova el que es igual, suspen el que es diferent, no s'avalua amb")
    print("     errors grans, coincideix amb la regla 2, i el costat real esta tancat sense firma.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    print("Aqui nomes hi ha la regla (veredicte) i --autoprova. La Z amb preses reals es a "
          "eines/lyapunov_real.py: z_r2 (la lambda* dels 150 graus) i z_veins, darrere del pany de la firma.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
