r"""Els sis JSON de friccio_pendol_doble.py, en una taula comparable.

Es el pas que fins ara no es podia fer: cada coeficient sortia d'UNA presa, o
sigui que la seva incertesa era la de l'ajust i prou. Amb tres repeticions de
cada cos hi ha dues incerteses diferents i s'han de mirar totes dues:

  - la de l'AJUST, que diu com de be queda la corba dins d'una presa
  - la DISPERSIO entre repeticions, que diu si l'experiment es repetible

Si la segona es molt mes gran que la primera, el que varia no es el soroll:
es el muntatge o la presa.

Conversio a SI: el codi treballa amb s (rad/s2), v (1/s) i a (1/rad), i
tau_f = s*I, b = v*I, k = a*I. Es llegeix de energia_sencera, que es el
registre complet.

Era un esborrany (9/09/2026) i va pujar a eines/ el 29/09, perque la taula va a la
memoria. La sortida es la mateixa que la de l'esborrany, comprovat en pujar-lo.

    .venv\Scripts\python.exe eines\taula_friccio_nova.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# Des de l'arrel del projecte, i no des d'on s'executi l'ordre.
CARPETA = Path(__file__).resolve().parent.parent / "dades" / "friccio_doble"

COSSOS = {
    "conjunt lligat (rodament 1)": (["1.1", "1.2", "1.3"], "1"),
    "barra 2 sola (rodament 2)":   (["2.1", "2.2", "2.3"], "2"),
}

# El que hi ha publicat, per comparar-hi. Dos jocs que NO coincideixen.
PUBLICAT = {
    "1": {"parametres.json (cims, 8-55o)": {"tau_f": 2.10e-4, "b": 8.34e-5},
          "seccio 08 (energia)":           {"tau_f": 2.101e-4, "b": 8.20e-5}},
    "2": {"parametres.json (cims, 8-55o)": {"tau_f": 1.54e-4, "b": 9.59e-6},
          "seccio 08 (energia)":           {"tau_f": 1.608e-4, "b": 6.46e-6}},
}


def llegeix(nom):
    d = json.loads((CARPETA / f"Friccio_{nom}.json").read_text(encoding="utf-8"))
    e, I = d["energia_sencera"], d["inercia_kg_m2"]
    return {
        "nom": nom, "T0": d["T0_s"], "L_eq": d["I_partit_per_M_d_m"],
        "fps": d["fps"], "finestra_s": d.get("finestra_savgol_s"),
        "A_max": e["amplitud_graus"][1], "w_max": e["w_max_rad_s"],
        "tau_f": e["s_sec_rad_s-2"] * I,
        "b":     e["v_viscos_s-1"] * I,
        "k":     e["a_aerodinamic_rad-1"] * I,
        "tau_f_err": e["errors"][0] * I,
        "b_err":     e["errors"][1] * I,
        "k_err":     e["errors"][2] * I,
        "residu": e["residu_relatiu"],
    }


def main():
    for titol, (noms, clau) in COSSOS.items():
        p = [llegeix(n) for n in noms]
        print("=" * 78)
        print(f" {titol}")
        print("=" * 78)
        print(f"\n  {'presa':<8}{'T0 (s)':>9}{'A max':>8}{'w max':>8}"
              f"{'tau_f (N m)':>14}{'b (N m s)':>13}{'k (N m s2)':>13}")
        for r in p:
            print(f"  {r['nom']:<8}{r['T0']:>9.4f}{r['A_max']:>7.0f}o"
                  f"{r['w_max']:>8.1f}{r['tau_f']:>14.4e}"
                  f"{r['b']:>13.2e}{r['k']:>13.4e}")

        print()
        for et, cl, cle in (("tau_f", "tau_f", "tau_f_err"),
                            ("b", "b", "b_err"), ("k", "k", "k_err")):
            v = np.array([r[cl] for r in p])
            aj = float(np.mean([r[cle] for r in p]))
            m, s = float(v.mean()), float(v.std(ddof=1))
            rel = f"{100 * s / m:.1f} %" if m > 1e-12 else "-"
            print(f"  {et:<7} mitjana {m:.4e}   dispersio {s:.2e} ({rel})"
                  f"   error d'ajust {aj:.2e}")

        # Contra el que hi ha publicat. La sigma que val es la mes gran de les
        # dues: si la dispersio entre preses supera l'error d'ajust, mana ella.
        print()
        for etiqueta, vals in PUBLICAT[clau].items():
            print(f"  contra {etiqueta}:")
            for et in ("tau_f", "b"):
                v = np.array([r[et] for r in p])
                m, s = float(v.mean()), float(v.std(ddof=1))
                aj = float(np.mean([r[et + "_err"] for r in p]))
                sig = max(s, aj)
                d = abs(m - vals[et]) / sig if sig > 0 else float("inf")
                marca = "ok" if d < 2 else ("!!" if d > 3 else "..")
                print(f"     {et:<7} nou {m:.4e}  publicat {vals[et]:.4e}"
                      f"   {d:5.1f} sigma  {marca}")
        print()

    # El zero del viscos: quin sostre hi posen les preses?
    print("=" * 78)
    print(" El terme VISCOS: fins on el poden acotar les sis preses")
    print("=" * 78)
    for titol, (noms, clau) in COSSOS.items():
        p = [llegeix(n) for n in noms]
        b = np.array([r["b"] for r in p])
        err = np.array([r["b_err"] for r in p])
        # amb 3 preses independents, el sostre a 2 sigma de la mitjana
        sostre = float(b.mean() + 2 * err.mean() / np.sqrt(len(p)))
        print(f"\n  {titol}")
        print(f"     b mesurada:  {'  '.join(f'{x:.2e}' for x in b)}")
        print(f"     sostre a 2 sigma:  b < {sostre:.2e} N m s/rad")
        for et, v in PUBLICAT[clau].items():
            estat = "QUEDA FORA" if v["b"] > sostre else "hi cap"
            print(f"     el b de {et:<30} = {v['b']:.2e}  ->  {estat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
