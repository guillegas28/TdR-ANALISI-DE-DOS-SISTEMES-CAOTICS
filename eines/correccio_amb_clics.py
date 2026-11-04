r"""La correccio amb clics (27/10/2026): les constants que fa servir comparar.corregeix_presa a les preses clicades.

Puja d'esborranys/_creu_correccio.py (26/10). La creu del Tracker no es al centre dels gomets, i es mou entre el repos
inicial i el final; en Guillem en va clicar el centre a les 23 preses de 45 i 90 graus (dades/versemblant/clics_creu.json,
mes l'extrem i el pivot del repos final de clics_cargol.json). Per a cada presa, aquesta eina calcula:
    dth_rad              el gir de la vertical: pivot (mediana del fix) -> clic de l'extrem al repos final, en lloc de la creu
    theta_inicial_rad    els angles dels clics del repos inicial, com la conversio (la marca amb la mateixa correccio al
                         llarg de la barra, tracker.corregeix_marca), respecte de la vertical del clic
    theta_final_rad      el mateix, amb els clics del repos final
    part_a               la part A de la correccio en dues parts, refeta amb la cua posada als clics
i les escriu a dades/versemblant/correccio_clics.json (versionat). comparar.posa_clics les aplica. Pla:
documentacio/pla-correccio-biaix.md, «La correccio amb clics i la catorzena tanda» i «27/10: l'endavant».

Nomes llegeix: els dos JSON de clics, els .trk, dades/retalls.json, dades/net/<presa>.json i .csv, centres_cercle.json i
parametres.json. Cap simulacio.

    .venv\Scripts\python.exe eines\correccio_amb_clics.py              # les preses clicades -> el JSON
    .venv\Scripts\python.exe eines\correccio_amb_clics.py --autoprova  # K0, K1, K2, K3, F1 i J
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))
sys.path.insert(0, str(ARREL / "eines"))
from pendol import comparar, dades, tracker, trk  # noqa: E402
from pendol.fisica import Parametres  # noqa: E402
import centre_del_cercle as cc  # noqa: E402

CLICS_NOUS = ARREL / "dades" / "versemblant" / "clics_creu.json"
CLICS_VELLS = ARREL / "dades" / "versemblant" / "clics_cargol.json"
SORTIDA = comparar.CLICS
# A la 90_11 el .trk diu que els eixos estan girats -6,05 graus, pero el Tracker en mostra -0,1 i els clics li donen la
# rao (pla, «El pas 0, refet»). S'hi fa servir l'angle que mostra el Tracker.
ANGLE_DEL_TRACKER = {"90_11": math.radians(-0.1)}


# ------------------------------------------------------------ dels clics a la imatge

def eixos(cami: Path) -> list[tuple]:
    """Els fotogrames clau del sistema de coordenades del .trk: (fotograma, x0, y0, angle, xescala, yescala)."""
    coords = trk._dins(trk._fills(ET.parse(cami).getroot()).get("coords"))
    fd = trk._fills(coords)["framedata"]
    fora = []
    for prop in fd.findall("property"):
        c = trk._fills(prop.find("object"))
        fora.append((int(prop.get("name").strip("[]")), trk._num(c, "xorigin"), trk._num(c, "yorigin"),
                     trk._num(c, "angle", 0.0), trk._num(c, "xscale"), trk._num(c, "yscale")))
    return sorted(fora)


def a_imatge(q, f: int, claus: list[tuple]) -> np.ndarray:
    """Un clic (unitats del Tracker, y amunt) -> pixels d'imatge (y avall), amb el fotograma clau vigent a f."""
    _k, x0, y0, al, sx, sy = [c for c in claus if c[0] <= f][-1]
    x, y = q
    return np.array([x0 + sx * x * math.cos(al) - sy * y * math.sin(al),
                     y0 - sx * x * math.sin(al) - sy * y * math.cos(al)])


def fotogrames(text: str) -> list[int]:
    n = [int(v) for v in re.findall(r"\d+", str(text))]
    return list(range(n[0], n[-1] + 1)) if len(n) == 2 else n * 3


def a_amollada(c, t, ta) -> float:
    """Els angles de sortida: la mitjana de [ta - 0,4; ta - 0,02] (la de la T7a), en graus."""
    ini = (t >= max(0.0, ta - 0.4)) & (t <= ta - 0.02)
    return math.degrees(float(np.mean(c[ini])))


# ------------------------------------------------------------ el metode

def phi(v) -> float:
    """L'angle d'un vector de la imatge (y avall) des de la vertical d'avall: el de tracker._inclinacio."""
    return math.atan2(float(v[0]), float(v[1]))


def angles_conversio(P, A, G, gir: float, delta: float | None, s: float) -> tuple[float, float]:
    """Els angles (theta1, theta2) d'unes posicions al repos, com la conversio: la marca moguda delta px al llarg de la
    barra (tracker.corregeix_marca), i theta = s*(phi - gir)."""
    J = tracker.corregeix_marca(P, np.asarray(A, float)[None, :], delta)[0] if delta is not None else np.asarray(A, float)
    return s * (phi(J - P) - gir), s * (phi(np.asarray(G, float) - J) - gir)


def constants(t, A, G, P, ta, fps, gir_x, delta, s, clics: dict) -> dict:
    """Les constants de la correccio amb clics d'una presa. clics: Ai, Gi (repos inicial), Af, Gf (repos final), en px."""
    gir_c = phi(clics["Gf"] - P)
    ti = angles_conversio(P, clics["Ai"], clics["Gi"], gir_c, delta, s)
    tf = angles_conversio(P, clics["Af"], clics["Gf"], gir_c, delta, s)
    n = int(round(cc.CUA_S * fps))
    Am, Gm = A.copy(), G.copy()
    Am[-n:], Gm[-n:] = clics["Af"], clics["Gf"]
    return {"dth_rad": s * (gir_x - gir_c), "theta_inicial_rad": [ti[0], ti[1]], "theta_final_rad": [tf[0], tf[1]],
            "part_a": cc.analitza(t, Am, Gm, P, ta, fps)["part_a"]}


def aplica(th1, th2, tt, ta, n_cua, rho, c: dict):
    """La correccio amb clics d'una serie: comparar.posa_clics i la correccio en dues parts amb la part A dels clics."""
    a1, a2 = comparar.posa_clics(th1, th2, tt, ta, n_cua, c)
    pa = c["part_a"]
    return comparar.corregeix_dues_parts(a1, a2, pa["d"], pa["rho_g"], rho, n_cua, pa["r"])[:2]


# ------------------------------------------------------------ les preses reals

def preses_clicades() -> list[str]:
    nous = json.loads(CLICS_NOUS.read_text(encoding="utf-8"))["preses"]
    return sorted([n for n in nous if len(nous[n].get("clics", [])) == 4],
                  key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))


def llegeix_presa(n: str, nous: dict, vells: dict, retalls: dict) -> dict:
    """Tot el que cal d'una presa clicada: els clics en px, les posicions crues del .trk i la serie de la conversio."""
    g = n.split("_")[0]
    cami = ARREL / "dades" / "video" / "tracker" / g / f"{n}.trk"
    claus = eixos(cami)
    if n in ANGLE_DEL_TRACKER:
        claus = [(kk, x0, y0, ANGLE_DEL_TRACKER[n], sx, sy) for kk, x0, y0, _al, sx, sy in claus]
    en_px = nous[n].get("unitats") == "px"
    im, fr = {}, {}
    for c in nous[n]["clics"]:
        q = "final" if "final" in c["que"] else c["que"].split(" ")[0]
        im[q] = np.array([c["x"], c["y"]], float) if en_px else a_imatge((c["x"], c["y"]), c["fotograma"], claus)
        fr[q] = c["fotograma"]
    fs = fotogrames(vells[n]["fotogrames"])
    d = trk.retalla(trk.llegeix_trk(cami), retalls.get(n, {}).get("des_del_fotograma"))
    k, A, G, _noms, (_kf, Pf) = trk.tres_pistes(d)
    taula, meta = dades.carrega(ARREL / "dades" / "net" / f"{n}.csv")
    geo = meta["geometria"]
    ma = geo["marca_articulacio"]
    A = np.asarray(A, float)
    return {"k": k, "A": A, "G": np.asarray(G, float), "P": np.median(Pf, axis=0), "fps": 1.0 / d["delta_t_s"],
            "t": (k - k[0]) * d["delta_t_s"], "taula": taula, "meta": meta, "fr": fr, "cua_s": float(geo["cua_s"]),
            "delta": float(ma["delta_px"]) if ma.get("corregida") else None,
            "s": -1.0 if meta["signe_invertit"] else 1.0, "gir_x": math.radians(float(geo["gir_graus"])),
            "ta": float(meta["amollada"]["amollada_s"]),
            "clics": {"Ai": im["rodament"], "Gi": im["extrem"], "Af": im["final"],
                      "Gf": a_imatge(vells[n]["gomet"], fs[-1], claus)},
            "pivot_clic": im["pivot"]}


def calcula() -> tuple[dict, dict]:
    nous = json.loads(CLICS_NOUS.read_text(encoding="utf-8"))["preses"]
    vells = json.loads(CLICS_VELLS.read_text(encoding="utf-8"))["preses"]
    retalls = json.loads((ARREL / "dades" / "retalls.json").read_text(encoding="utf-8"))
    p = Parametres.des_de_json(ARREL / "parametres.json")
    rho = p.r_gomet / p.L1
    fora, lectures = {}, {}
    for n in preses_clicades():
        x = llegeix_presa(n, nous, vells, retalls)
        c = constants(x["t"], x["A"], x["G"], x["P"], x["ta"], x["fps"], x["gir_x"], x["delta"], x["s"], x["clics"])
        tt = x["taula"]["t"].to_numpy()
        n_cua = max(10, int(round(x["cua_s"] / float(np.median(np.diff(tt))))))
        n1, n2 = aplica(x["taula"]["theta1"].to_numpy(), x["taula"]["theta2"].to_numpy(), tt, x["ta"], n_cua, rho, c)
        c["amollada_amb_clics_graus"] = [a_amollada(n1, tt, x["ta"]), a_amollada(n2, tt, x["ta"])]
        if n in ANGLE_DEL_TRACKER:
            c["angle_eixos_tracker_graus"] = math.degrees(ANGLE_DEL_TRACKER[n])
        fora[n] = c
        lectures[n] = x
    return fora, lectures


def escriu(fora: dict) -> None:
    SORTIDA.write_text(json.dumps({
        "_que_es": ("Les constants de la correccio amb clics de cada presa clicada (27/10/2026), les que aplica "
                    "comparar.corregeix_presa (posa_clics i la part A d'aqui). Ho escriu eines/correccio_amb_clics.py "
                    "amb els clics de clics_creu.json i clics_cargol.json. Angles en radians, al marc de la conversio."),
        "preses": fora}, indent=1, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------ l'autoprova

def presa_fabricada(creu: bool):
    """Geometria pura (px d'imatge, y avall): l'eix de debo C, el pivot de la conversio P fora de l'eix, la camera inclinada,
    la marca del rodament de costat (fixa a la barra 1) i delta mes enlla de l'articulacio, i, si creu, la creu desplacada
    fixa a cada barra, diferent al repos inicial, en moviment i als ultims 2,5 s (fora de la finestra del cercle)."""
    p = Parametres.des_de_json(ARREL / "parametres.json")
    rho = p.r_gomet / p.L1
    fps, ta, T = 240.0, 1.0, 12.0
    t = np.arange(0.0, T, 1.0 / fps)
    C, P, tau, L1, sig, delta = np.array([500.0, 300.0]), np.array([501.5, 304.0]), math.radians(0.5), 200.0, 0.02, 7.0
    th10, th20 = math.radians(45.0), math.radians(44.0)
    m = (t > ta) & (t < T - 3.0)
    tm = t - ta
    th1 = np.where(t <= ta, th10, np.where(m, th10 * np.cos(2 * np.pi * 0.9 * tm) * np.exp(-tm / 3), 0.0))
    th2 = np.where(t <= ta, th20, np.where(m, 2.5 * th20 * np.cos(2 * np.pi * 1.3 * tm + 0.5) * np.exp(-tm / 3), 0.0))
    e = lambda a: np.column_stack([np.sin(a + tau), np.cos(a + tau)])     # noqa: E731
    pp = lambda a: np.column_stack([np.cos(a + tau), -np.sin(a + tau)])   # noqa: E731
    J = C + L1 * e(th1)
    Av = J + delta * e(th1) + sig * L1 * pp(th1)
    Gv = J + rho * L1 * e(th2)
    A, G = Av.copy(), Gv.copy()
    if creu:
        oA = {"i": (2.0, -1.0), "m": (1.0, 0.5), "f": (-1.5, 1.0)}
        oG = {"i": (-1.5, 2.0), "m": (0.5, -1.0), "f": (2.5, 0.5)}
        for tram, w in (("i", t <= ta), ("m", (t > ta) & (t < T - cc.FORA_DEL_FINAL_S)),
                        ("f", t >= T - cc.FORA_DEL_FINAL_S)):
            A[w] += oA[tram][0] * pp(th1[w]) + oA[tram][1] * e(th1[w])
            G[w] += oG[tram][0] * pp(th2[w]) + oG[tram][1] * e(th2[w])
    n_cua = max(10, int(round(2.0 * fps)))
    gir_x = phi(G[-n_cua:].mean(0) - P)
    Jc = tracker.corregeix_marca(P, A, delta)
    a1 = np.unwrap(np.array([phi(q - P) - gir_x for q in Jc]))
    a2 = np.unwrap(np.array([phi(g - q) - gir_x for q, g in zip(Jc, G)]))
    return dict(t=t, A=A, G=G, P=P, th1=a1, th2=a2, ta=ta, fps=fps, gir_x=gir_x, delta=delta, s=1.0, rho=rho,
                n_cua=n_cua, clics={"Ai": Av[0], "Gi": Gv[0], "Af": Av[-1], "Gf": Gv[-1]})


def sortida_creu(f):
    pa = cc.analitza(f["t"], f["A"], f["G"], f["P"], f["ta"], f["fps"])["part_a"]
    n1, n2, _ = comparar.corregeix_dues_parts(f["th1"], f["th2"], pa["d"], pa["rho_g"], f["rho"], f["n_cua"], pa["r"])
    return a_amollada(n1, f["t"], f["ta"]), a_amollada(n2, f["t"], f["ta"])


def sortida_clics(f, clics):
    c = constants(f["t"], f["A"], f["G"], f["P"], f["ta"], f["fps"], f["gir_x"], f["delta"], f["s"], clics)
    n1, n2 = aplica(f["th1"], f["th2"], f["t"], f["ta"], f["n_cua"], f["rho"], c)
    return a_amollada(n1, f["t"], f["ta"]), a_amollada(n2, f["t"], f["ta"])


def autoprova() -> int:
    oberts, mirant = [], [True]
    sys.addaudithook(lambda ev, a: oberts.append(str(a[0])) if mirant[0] and ev == "open"
                     and isinstance(a[0], str) else None)
    problemes = []

    def mira(cond, que):
        print(f"   {'ok ' if cond else 'NO '} {que}")
        if not cond:
            problemes.append(que)

    print("Autoprova de la correccio amb clics (pla-correccio-biaix.md, «La correccio amb clics», K0-K2, i la D3b)")
    sense, amb = presa_fabricada(False), presa_fabricada(True)
    ref = sortida_creu(sense)
    ck, cr = sortida_clics(amb, amb["clics"]), sortida_creu(amb)
    e1 = max(abs(ck[0] - ref[0]), abs(ck[1] - ref[1]))
    e2 = max(abs(cr[0] - ref[0]), abs(cr[1] - ref[1]))
    mira(e1 < 0.01, f"K1 en una presa fabricada, els clics al lloc bo = la presa sense creu: {e1:.5f} graus (< 0,01)")
    mira(e2 > 0.1, f"K2 la creu sola s'hi aparta {e2:.3f} graus (> 0,1)")
    fora, lect = calcula()
    p = Parametres.des_de_json(ARREL / "parametres.json")
    rho = p.r_gomet / p.L1
    k0, k3, f1 = 0.0, 0.0, []
    for n, x in lect.items():
        tt = x["taula"]["t"].to_numpy()
        n_cua = max(10, int(round(x["cua_s"] / float(np.median(np.diff(tt))))))
        th1, th2 = x["taula"]["theta1"].to_numpy(), x["taula"]["theta2"].to_numpy()
        # K0: els «clics» posats a la creu (la mitjana de les posicions crues a cada finestra) = la correccio d'abans.
        t, A, G = x["t"], x["A"], x["G"]
        ini = (t >= max(0.0, x["ta"] - 0.4)) & (t <= x["ta"] - 0.02)
        ncua = max(10, int(round(x["cua_s"] * x["fps"])))
        a_creu = {"Ai": A[ini].mean(0), "Gi": G[ini].mean(0), "Af": A[-ncua:].mean(0), "Gf": G[-ncua:].mean(0)}
        c0 = constants(t, A, G, x["P"], x["ta"], x["fps"], x["gir_x"], x["delta"], x["s"], a_creu)
        q1, q2 = aplica(th1, th2, tt, x["ta"], n_cua, rho, c0)
        v1, v2, _ = comparar.corregeix_presa(x["taula"], x["meta"], n, amb_clics=False)
        k0 = max(k0, abs(a_amollada(q1, tt, x["ta"]) - a_amollada(v1, tt, x["ta"])),
                 abs(a_amollada(q2, tt, x["ta"]) - a_amollada(v2, tt, x["ta"])))
        # K3 (la D3b del pla): comparar.corregeix_presa, amb el JSON del disc, dona els angles d'aqui.
        w1, w2, info = comparar.corregeix_presa(x["taula"], x["meta"], n)
        k3 = max(k3, abs(math.radians(a_amollada(w1, tt, x["ta"]) - fora[n]["amollada_amb_clics_graus"][0])),
                 abs(math.radians(a_amollada(w2, tt, x["ta"]) - fora[n]["amollada_amb_clics_graus"][1])))
        if not info["clics"]:
            k3 = math.inf
        # F1: els clics nous, a prop de la seva creu al mateix fotograma.
        dist = [float(np.hypot(*(x["pivot_clic"] - x["P"])))]
        for q, X, cl in (("rodament", A, "Ai"), ("extrem", G, "Gi"), ("final", A, "Af")):
            i = np.nonzero(x["k"] == x["fr"][q])[0]
            dist.append(float(np.hypot(*(X[i[0]] - x["clics"][cl]))) if len(i) else math.inf)
        f1.append(dist[0] < 8.0 and all(dd < 15.0 for dd in dist[1:]))
    mira(k0 < 0.005, f"K0 amb els clics a la creu = la correccio d'abans, {len(lect)} preses: {k0:.5f} graus (< 0,005)")
    mira(k3 < 1e-12, f"K3 (D3b) corregeix_presa amb el JSON = els angles d'aqui: {k3:.1e} rad (< 1e-12); "
                     "si falla, cal tornar a escriure el JSON")
    mira(all(f1), f"F1 els clics, a prop de la seva creu (pivot < 8 px, marques < 15 px): {sum(f1)} de {len(f1)}")
    mirant[0] = False
    permesos_dir = [ARREL / "dades" / "video" / "tracker", ARREL / "dades" / "net"]
    permesos = {CLICS_NOUS.resolve(), CLICS_VELLS.resolve(), (ARREL / "dades" / "retalls.json").resolve(),
                comparar.CENTRES.resolve(), SORTIDA.resolve(), (ARREL / "parametres.json").resolve()}
    altres = sorted({str(Path(o).resolve()) for o in oberts
                     if ARREL in Path(o).resolve().parents and (ARREL / ".venv") not in Path(o).resolve().parents
                     and Path(o).suffix not in (".py", ".pyc", ".pth") and Path(o).resolve() not in permesos
                     and not any(dd in Path(o).resolve().parents for dd in permesos_dir)})
    mira(not altres, f"J nomes obre els fitxers dits: {altres or 'cap altre'}")
    print(f"\n{'TOT PASSA' if not problemes else 'FALLA: ' + '; '.join(problemes)}")
    return 0 if not problemes else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)
    if args.autoprova:
        return autoprova()
    fora, _ = calcula()
    escriu(fora)
    print(f"{'presa':7}{'theta1':>10}{'theta2':>10}   (angles d'amollada amb clics, graus)")
    for n, c in fora.items():
        print(f"{n:7}" + "".join(f"{a:+10.3f}" for a in c["amollada_amb_clics_graus"]))
    print(f"-> {SORTIDA.relative_to(ARREL)} ({len(fora)} preses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
