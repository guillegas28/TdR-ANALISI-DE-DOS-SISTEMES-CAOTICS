r"""La regla 2 amb el soroll de les preses: es pot treure alguna lambda? NOMES SIMULACIO.

Per que existeix. El 03/10 es va mesurar el soroll de seguiment en moviment
(eines/soroll_en_moviment.py): el terra d'una parella es d'1,3 / 3,6 / 6,9
graus als 45 / 90 / 150, i la d0 de les parelles reals, d'1,5-1,8. Abans de
fer servir mai la regla 2 (caos.ajusta_zona) sobre tirs reals, cal saber si,
AMB AQUEST SOROLL, el metode pot donar cap lambda, i de quant s'equivoca. Aixo
es pot respondre sense tocar cap tir real.

COM.
  - 12 tirs SIMULATS per grup, sense friccio, amollats dels bracos rectes a 45,
    90 i 150 graus + N(0; dispersio) a cada angle (0,3 o 1,0 graus).
  - Davant de cada tir, 0,5 s de repos (el pendol hi es quiet de debo): aixi
    el filtre no te vora a l'amollada, com a les preses.
  - Soroll fabricat LENT: AR(1) a cada angle CRU, amb la rho i la proporcio
    theta1:theta2 mesurades al grup, escalat perque la sigma_d per tir sigui la
    que es demana. Despres, el Savitzky-Golay (13, 3) de la conversio dona els
    angles i les velocitats, com a tracker.processa.
  - Escombrat: sigma_d per tir de 0,03 a 10 graus, i els tres valors mesurats
    (repos, sense geometria, total). 5 realitzacions de soroll per cas.
  - La regla 2 amb terra = sqrt(2) x sigma_d (el terra d'una parella), el sostre
    de caos.sostre_de_saturacio i la d0 de les condicions inicials exactes, com
    a les preses (la dels angles de repos).
  - La lambda de referencia: Benettin de 1000 s sobre les 12 condicions
    inicials d'1,0 graus de cada grup (caos.benettin_pendol, classifica).

QUE ENTRA DE LES PRESES REALS: nomes dos JSON de soroll (dades/divergencia/).
Cap trajectoria real, i cap numero d'aqui no es posa al costat de cap presa.

Criteris S1-S4 escrits a documentacio/pla-analisi-caos.md («Cinc feines sense
intervencio») el 03/10, abans d'escriure aixo.

Us:
    .venv\\Scripts\\python eines\\regla2_amb_soroll.py --autoprova
    .venv\\Scripts\\python eines\\regla2_amb_soroll.py --json dades/divergencia/regla2_amb_soroll_2026-10-03.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, savgol_filter

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from pendol import caos  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
from pendol.mesures import omega_zero  # noqa: E402

SOROLL_JSON = ARREL / "dades" / "divergencia" / "soroll_moviment_2026-10-03.json"
TERRA_JSON = ARREL / "dades" / "divergencia" / "terra_2026-10-03.json"

GRUPS = (45.0, 90.0, 150.0)
DISPERSIONS_GRAUS = (0.3, 1.0)
TIRS = 12
DURADA_S = 10.0
FPS = 240.0
REPOS_S = 0.5
FINESTRA, ORDRE = 13, 3          # el Savitzky-Golay de la conversio
ESCOMBRAT_GRAUS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
REALITZACIONS = 5
BOOTSTRAP = 200
LLAVOR = 20261003
TEMPS_BENETTIN_S = 1000.0
PROCESSOS = 6


# ------------------------------------------------------------- els tirs

def condicions_inicials(grup: float, dispersio: float, llavor: int) -> np.ndarray:
    """(TIRS, 4): bracos rectes a `grup` graus + N(0; dispersio) a cada angle."""
    rng = np.random.default_rng(llavor)
    x = []
    for _ in range(TIRS):
        th = np.radians(grup + rng.normal(0.0, dispersio, 2))
        x.append([th[0], th[1], 0.0, 0.0])
    return np.array(x)


def simula_tirs(p, x0s, t) -> np.ndarray:
    """(TIRS, 4, M) sense friccio, sobre la graella t (des de l'amollada)."""
    from pendol.simulador import simula
    return np.array([simula(p, x0, t_eval=t)[1] for x0 in x0s])


def llavor_de(*parts) -> int:
    """Una llavor per cas, estable i sense hash de Python (que canvia entre execucions)."""
    v = LLAVOR
    for x in parts:
        v = (v * 1000003 + int(round(float(x) * 1000))) % (2 ** 63)
    return v


def ar1(rng, rho: float, n: int) -> np.ndarray:
    """AR(1) estacionari de desviacio 1."""
    z = rng.normal(0.0, 1.0, n)
    zi = [rho * rng.normal(0.0, 1.0)]
    return lfilter([math.sqrt(1.0 - rho * rho)], [1.0, -rho], z, zi=zi)[0]


def com_la_conversio(th_cru, fps: float = FPS):
    """(angles, velocitats) d'uns angles crus, amb el filtre de tracker.processa."""
    th = np.vstack([savgol_filter(r, FINESTRA, ORDRE) for r in th_cru])
    w = np.vstack([savgol_filter(r, FINESTRA, ORDRE, deriv=1, delta=1.0 / fps) for r in th_cru])
    return np.vstack([th, w])


def components_soroll(estats_soroll, w0: float) -> np.ndarray:
    """La sigma de cada component (radians d'angle equivalent) d'un soroll ja filtrat."""
    x = np.asarray(estats_soroll)
    return np.array([x[0].std(ddof=1), x[1].std(ddof=1), x[2].std(ddof=1) / w0, x[3].std(ddof=1) / w0])


class Soroll:
    """El soroll d'un grup: rho i proporcio mesurades, i l'escala per a una sigma_d."""

    def __init__(self, rho, proporcio, w0: float):
        self.rho = [float(r) for r in rho]
        self.proporcio = np.asarray(proporcio, dtype=float) / float(np.max(proporcio))
        self.w0 = w0
        # La sigma_d que dona una amplitud unitat, mesurada sobre 200 s
        # (deterministic: llavor fixa). Despres s'escala linealment.
        rng = np.random.default_rng(LLAVOR + 17)
        n = int(200.0 * FPS)
        cru = np.vstack([self.proporcio[k] * ar1(rng, self.rho[k], n) for k in range(2)])
        c = components_soroll(com_la_conversio(cru)[:, 50:-50], w0)
        self.sigma_d_unitat = float(np.sqrt(np.sum(c * c)))

    def cru(self, rng, n: int, sigma_d: float) -> np.ndarray:
        """(2, n) soroll cru d'angle perque, filtrat, doni sigma_d (radians)."""
        a = sigma_d / self.sigma_d_unitat
        return np.vstack([a * self.proporcio[k] * ar1(rng, self.rho[k], n) for k in range(2)])


def estats_amb_soroll(y, soroll: Soroll | None, sigma_d: float, rng, n_repos: int) -> np.ndarray:
    """Un tir com el donaria la conversio: repos + moviment, soroll cru, filtre; des de t = 0."""
    th = np.hstack([np.repeat(y[:2, :1], n_repos, axis=1), y[:2]])
    if soroll is not None and sigma_d > 0:
        th = th + soroll.cru(rng, th.shape[1], sigma_d)
    return com_la_conversio(th)[:, n_repos:]


def regla2(t, estats, x0s, w0: float, terra: float) -> dict:
    """La regla 2 sobre els 12 tirs, amb la d0 de les condicions inicials exactes."""
    L = caos.matriu_ln_d(estats, w0)
    L0 = np.full((TIRS, TIRS), -np.inf)
    from pendol.mesures import distancia_fases
    for i in range(TIRS):
        for j in range(i + 1, TIRS):
            L0[i, j] = L0[j, i] = math.log(float(distancia_fases(x0s[i], x0s[j], w0)))
    par = caos._parelles_de(list(range(TIRS)))
    sostre = caos.sostre_de_saturacio(t, np.array([L[i, j] for i, j in par]))
    r = caos.lambda_de_tirs(t, L, L0, terra, sostre)
    r.pop("corba", None)
    if r["ok"]:
        b = caos.bootstrap_per_tirs(t, L, L0, terra, sostre, n=BOOTSTRAP, llavor=LLAVOR)
        r["bootstrap"] = {"desviacio": b["desviacio"], "sense_lambda": b["sense_lambda"],
                          "remostrejos": b["remostrejos"]}
    r["d0_mediana"] = float(np.exp(np.median([L0[i, j] for i, j in par])))
    return r


# ------------------------------------------------------------- el soroll mesurat

def caracter_mesurat() -> dict:
    """Dels dos JSON de soroll: rho, proporcio i les tres sigma_d per tir, per grup.

    ES L'UNICA FUNCIO QUE OBRE ALGUNA COSA DE dades/.
    """
    mov = json.loads(SOROLL_JSON.read_text(encoding="utf-8"))
    rep = json.loads(TERRA_JSON.read_text(encoding="utf-8"))
    sortida = {}
    for g in GRUPS:
        k = f"{g:.0f}"
        pg = mov["per_grup"][k]
        crus = np.median([v["soroll_cru_per_fotograma_graus"] for v in mov["per_presa"].values()
                          if v["grup"] == k and v["r0"]["ok"]], axis=0)
        sortida[k] = {
            "rho": pg["lag1_mediana"],
            "proporcio_th1_th2": [float(x) for x in crus],
            "sigma_d_graus": {
                "repos": rep["per_grup"][k]["sigma_d_presa_graus"]["mediana"],
                "sense_geometria": pg["sigma_d_presa_sense_geometria_graus"]["mediana"],
                "total": pg["sigma_d_presa_total_graus"]["mediana"],
            },
        }
    return sortida


# ------------------------------------------------------------- Benettin

def _feina_benettin(args) -> dict:
    grup, i, x0 = args
    p = Parametres.des_de_json(ARREL / "parametres.json")
    ara = time.perf_counter()
    res = caos.benettin_pendol(p, np.asarray(x0, dtype=float), TEMPS_BENETTIN_S,
                               llavor=llavor_de(grup, i, 7))
    c = caos.classifica(res)
    return {"grup": grup, "tir": i, "mena": c["mena"], "lambda": c["lambda_T"],
            "E_sobre_Ev2": float(caos.energia_relativa(np.asarray(x0), p) / caos.energia_volteig_2(p)),
            "deriva_energia_relativa": res["deriva_energia_relativa"],
            "segons": time.perf_counter() - ara}


# ------------------------------------------------------------- la tanda

def tanda(p, caracter: dict, benettin: bool = True) -> dict:
    w0 = omega_zero(p)
    t = np.arange(0.0, DURADA_S + 0.5 / FPS, 1.0 / FPS)
    n_repos = int(round(REPOS_S * FPS))
    sortida = {"omega0_rad_s": w0, "caracter_del_soroll": caracter, "grups": {}}
    feines = []
    for g in GRUPS:
        k = f"{g:.0f}"
        soroll = Soroll(caracter[k]["rho"], caracter[k]["proporcio_th1_th2"], w0)
        casos = [("escombrat", s) for s in ESCOMBRAT_GRAUS] + \
                [(nom, v) for nom, v in caracter[k]["sigma_d_graus"].items()]
        bloc = {"sigma_d_unitat_graus": math.degrees(soroll.sigma_d_unitat), "dispersions": {}}
        for disp in DISPERSIONS_GRAUS:
            x0s = condicions_inicials(g, disp, llavor_de(g, disp))
            y = simula_tirs(p, x0s, t)
            if disp == 1.0:
                feines += [(g, i, x0s[i].tolist()) for i in range(TIRS)]
            nets = np.array([estats_amb_soroll(yy, None, 0.0, None, n_repos) for yy in y])
            res = {"sense_soroll": regla2(t, nets, x0s, w0, 1e-9)}
            for nom, sd in casos:
                files = []
                for r in range(REALITZACIONS):
                    rng = np.random.default_rng(llavor_de(g, disp, sd, r))
                    est = np.array([estats_amb_soroll(yy, soroll, math.radians(sd), rng, n_repos) for yy in y])
                    files.append(regla2(t, est, x0s, w0, math.sqrt(2.0) * math.radians(sd)))
                res[f"{nom}:{sd:g}"] = {"cas": nom, "sigma_d_tir_graus": sd,
                                        "terra_parella_graus": math.sqrt(2.0) * sd, "realitzacions": files}
            bloc["dispersions"][f"{disp:g}"] = res
            print(f"  {k} graus, dispersio {disp:g}: fet", flush=True)
        sortida["grups"][k] = bloc
    if benettin:
        print(f"  Benettin: {len(feines)} orbites de {TEMPS_BENETTIN_S:g} s, {PROCESSOS} processos...", flush=True)
        with multiprocessing.get_context("spawn").Pool(PROCESSOS) as pool:
            orbites = pool.map(_feina_benettin, feines)
        for g in GRUPS:
            k = f"{g:.0f}"
            o = [x for x in orbites if x["grup"] == g]
            caot = [x["lambda"] for x in o if x["mena"] == "caotica"]
            sortida["grups"][k]["benettin"] = {
                "orbites": o,
                "caotiques": len(caot), "regulars": sum(x["mena"] == "regular" for x in o),
                "lambda_mediana_caotiques": float(np.median(caot)) if caot else None,
            }
    return sortida


def imprimeix(r: dict) -> None:
    print(f"\nomega0 = {r['omega0_rad_s']:.3f} rad/s. NOMES SIMULACIO, sense friccio.")
    for k, bloc in r["grups"].items():
        c = r["caracter_del_soroll"][k]
        b = bloc.get("benettin") or {}
        lb = b.get("lambda_mediana_caotiques")
        print(f"\n== {k} graus.  rho {c['rho'][0]:.2f}/{c['rho'][1]:.2f};  Benettin (1000 s, 12 orbites "
              f"d'1,0 graus): {b.get('caotiques', '-')} caotiques, {b.get('regulars', '-')} regulars, "
              f"lambda mediana {lb if lb is None else round(lb, 3)} s^-1")
        for disp, res in bloc["dispersions"].items():
            s = res["sense_soroll"]
            print(f"   dispersio {disp} graus (d0 mediana {math.degrees(s['d0_mediana']):.2f} graus)")
            txt = f"lambda {s['lambda']:.3f}" if s["ok"] else f"cap: {s['motiu']}"
            print(f"      {'sense soroll':<26} {txt}")
            for clau, v in res.items():
                if clau == "sense_soroll":
                    continue
                fs = v["realitzacions"]
                lam = [f["lambda"] for f in fs if f["ok"]]
                nom = f"{v['cas']} {v['sigma_d_tir_graus']:.3g} graus/tir"
                if lam:
                    biaix = f", biaix {100 * (np.median(lam) / lb - 1):+.0f} %" if lb else ""
                    zona = [f for f in fs if f["ok"]][0]
                    print(f"      {nom:<26} lambda a {len(lam)}/{len(fs)}: mediana {np.median(lam):.3f}"
                          f"{biaix}  (zona {zona['t_inici_s']:.2f}-{zona['t_fi_s']:.2f} s, creix "
                          f"{zona['creixement_ln']:.2f})")
                else:
                    print(f"      {nom:<26} cap lambda a {len(fs)}/{len(fs)}: {fs[0]['motiu']}")


# --------------------------------------------------------------- l'autoprova

_OBERTS: list | None = None


def _registra(event, args):
    if _OBERTS is not None and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        _OBERTS.append(os.fsdecode(args[0]))


def autoprova() -> int:
    """S1-S4, escrits al pla el 03/10/2026 abans d'escriure el codi."""
    global _OBERTS
    problemes = []

    def mira(condicio, que):
        if not condicio:
            problemes.append(que)
        print(f"   {'ok ' if condicio else 'NO '} {que}", flush=True)

    p = Parametres.des_de_json(ARREL / "parametres.json")
    w0 = omega_zero(p)
    print("Autoprova de eines/regla2_amb_soroll.py (criteris escrits al pla el 03/10/2026)\n")

    sys.addaudithook(_registra)
    _OBERTS = []
    caracter = caracter_mesurat()
    oberts, _OBERTS = _OBERTS, None

    # S1. El soroll fabricat: sigma_d i rho.
    print("S1. El soroll fabricat dona la sigma_d demanada a < 10 % (mediana de 12 tirs) i la rho a +-0,02")
    n = int(round((DURADA_S + REPOS_S) * FPS)) + 1
    for k in ("45", "90", "150"):
        s = Soroll(caracter[k]["rho"], caracter[k]["proporcio_th1_th2"], w0)
        for sd in (1.0, caracter[k]["sigma_d_graus"]["total"]):
            rng = np.random.default_rng(llavor_de(999, float(k), sd))
            mesures, rhos = [], []
            for _ in range(TIRS):
                cru = s.cru(rng, n, math.radians(sd))
                c = components_soroll(com_la_conversio(cru)[:, 50:-50], w0)
                mesures.append(math.degrees(math.sqrt(float(np.sum(c * c)))))
                rhos.append([float(np.corrcoef(x[:-1], x[1:])[0, 1]) for x in cru])
            rel = np.median(mesures) / sd - 1
            drho = np.median(rhos, axis=0) - np.array(caracter[k]["rho"])
            mira(abs(rel) < 0.10 and np.all(np.abs(drho) < 0.02),
                 f"{k} graus, {sd:.3g} graus/tir: sigma_d {100 * rel:+.1f} %, rho "
                 f"{np.median(rhos, axis=0)[0]:.3f}/{np.median(rhos, axis=0)[1]:.3f} (demanada "
                 f"{caracter[k]['rho'][0]:.3f}/{caracter[k]['rho'][1]:.3f})")

    # S2. El diagnostic de l'01/10, amb les seves mateixes entrades.
    print("\nS2. Les entrades de l'01/10 (llavor = grup, 1,0 graus, 6 s, terra de 2 graus, sense soroll")
    print("    ni filtre) donen el mateix que esborranys/_diag_regla_v2.txt")
    esperat = {90.0: (0.60, 1.11), 150.0: (0.56, 1.00)}
    t6 = caos._T_PAR
    for g, (a, b) in esperat.items():
        x0s = condicions_inicials(g, 1.0, int(g))
        y = simula_tirs(p, x0s, t6)
        L = caos.matriu_ln_d(y, w0)
        L0 = L[:, :, 0].copy()
        par = caos._parelles_de(list(range(TIRS)))
        sostre = caos.sostre_de_saturacio(t6, np.array([L[i, j] for i, j in par]))
        r = caos.lambda_de_tirs(t6, L, L0, caos._TERRA_PAR, sostre)
        ok = (not r["ok"]) and round(r.get("t_inici_s", -1), 2) == a and round(r.get("t_fi_s", -1), 2) == b
        mira(ok, f"{g:.0f} graus: {'cap lambda' if not r['ok'] else 'LAMBDA ' + str(r['lambda'])}, zona "
                 f"{r.get('t_inici_s', float('nan')):.2f}-{r.get('t_fi_s', float('nan')):.2f} s "
                 f"(esperat {a:.2f}-{b:.2f})")

    # S3 i S4 sobre un cas petit: 90 graus, 1,0 graus.
    t = np.arange(0.0, DURADA_S + 0.5 / FPS, 1.0 / FPS)
    n_repos = int(round(REPOS_S * FPS))
    x0s = condicions_inicials(90.0, 1.0, llavor_de(90.0, 1.0))
    y = simula_tirs(p, x0s, t)
    s90 = Soroll(caracter["90"]["rho"], caracter["90"]["proporcio_th1_th2"], w0)

    def cas(llavor, sd):
        rng = np.random.default_rng(llavor)
        est = np.array([estats_amb_soroll(yy, s90, math.radians(sd), rng, n_repos) for yy in y])
        return json.dumps(regla2(t, est, x0s, w0, max(math.sqrt(2.0) * math.radians(sd), 1e-9)))

    print("\nS3. Amb el soroll a zero, dues llavors de soroll donen el mateix bit a bit")
    mira(cas(1, 0.0) == cas(2, 0.0), "90 graus, 1,0 graus de dispersio: identics")
    print("\nS4. Reproduible bit a bit, i de dades/ nomes obre els dos JSON de soroll")
    mira(cas(3, 1.0) == cas(3, 1.0), "90 graus amb 1 grau de soroll per tir: dues passades identiques")
    arrel_dades = (ARREL / "dades").resolve()
    dins = sorted({str(Path(x).resolve()) for x in oberts if Path(x).resolve().is_relative_to(arrel_dades)})
    mira(dins == sorted(str(x.resolve()) for x in (SOROLL_JSON, TERRA_JSON)),
         "de dades/ obre: " + ", ".join(str(Path(x).relative_to(arrel_dades)) for x in dins))
    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        print("\nUn criteri que falla NO es relaxa: s'atura i s'explica.")
        return 1
    print("OK -> el soroll fabricat te la sigma i la rho demanades, el cami de la regla 2 es el de")
    print("     l'01/10, el soroll zero no depen de la llavor, es reproduible i nomes llegeix el soroll.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="on desar el resultat")
    ap.add_argument("--sense-benettin", action="store_true", help="sense la referencia (per provar)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    p = Parametres.des_de_json(ARREL / "parametres.json")
    r = tanda(p, caracter_mesurat(), benettin=not args.sense_benettin)
    imprimeix(r)
    if args.json:
        cami = Path(args.json)
        cami.parent.mkdir(parents=True, exist_ok=True)
        r = {"eina": "eines/regla2_amb_soroll.py", "pla": "documentacio/pla-analisi-caos.md",
             "que_es": "NOMES SIMULACIO, sense friccio: la regla 2 sobre 12 tirs simulats per grup amb "
                       "soroll lent fabricat amb el caracter del mesurat; cap trajectoria real",
             **r}
        cami.write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nDesat: {cami}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
