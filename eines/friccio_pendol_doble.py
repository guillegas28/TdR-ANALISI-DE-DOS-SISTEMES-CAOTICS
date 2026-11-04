r"""Els tres mecanismes de friccio de cada rodament del pendol doble.

D'una presa de decaiment lliure (el pendol deixat anar i filmat fins que
s'atura) en surten quatre coses: el periode a amplitud zero, i els tres
coeficients d'esmorteiment separats.

    dA/dt = -( c0 + c1*A + c2*A^2 )        A = amplitud ANGULAR, rad
               |     |       |
               |     |       +-- AERODINAMIC  tau = -k*w|w|   c2 = 8k/(3*I*T)
               |     +---------- VISCOS       tau = -b*w      c1 = b/(2I) = sigma
               +---------------- SEC/COULOMB  tau = -tau_f    c0 = 4*tau_f/(M*g*d*T)

Surt de l'energia perduda per cicle de cada mecanisme. Els coeficients es
forcen a >= 0: cap dels tres no pot donar energia.

TRES COSES QUE AQUEST PROGRAMA VA HAVER D'APRENDRE, i que no es veuen a ull:

1. NO es pot ajustar derivant. La caiguda d'amplitud entre dos extrems
   consecutius es mes petita que el soroll d'un sol cim (mig pixel a 23 cm
   son 0.12 graus; la caiguda per mig cicle, 0.24). Derivar dona R2 = 0.006.
   S'integra l'EDO i s'ajusta A(t) sencera: R2 = 0.9999.

2. El sostre dels 55 graus NO es dels cims plans, com es va creure molt de
   temps: es de l'arcsinus. A = arcsin(|x-eq|/radi) no pot passar de 90 per
   construccio, i pitjor: per sobre de 90 la projeccio x deixa de creixer amb
   l'angle -val radi*sin(th), o sigui que fa el maxim JUSTAMENT a 90 i despres
   torna enrere-, i el detector marca el pas per la horitzontal en comptes del
   punt de retorn. Per aixo passar de x a atan2 no ho arreglava: el problema
   no era el sistema de coordenades sino mesurar amb un sinus.
   Consequencia mesurada: la presa 2 no es va amollar de 84 graus sino de
   141.5. Qui vulgui el registre sencer, que faci servir --energia.

3. El periode depen de l'amplitud (a 90 graus es un 23% mes llarg que a zero).
   El que va a I = M*g*d*T^2/4pi^2 es T a AMPLITUD ZERO, i s'hi arriba
   mesurant T a 12-25 graus i traient-ne la correccio 1 + theta^2/16.

LIMIT DECLARAT, i es el mes important: els tres regressors (1, A, A^2) son
gairebe degenerats en un rang d'amplituds finit. Les correlacions entre
coeficients son de -0.99. La caiguda TOTAL queda mesurada molt be; el
REPARTIMENT entre els tres mecanismes, molt pitjor del que diuen les barres
d'error formals. Vegeu documentacio/friccio-del-pendol-doble.html.

PENDENT D'ARQUITECTURA: importa el cercador d'extrems i l'ajust de
circumferencia de magnetic/, que segons el CLAUDE.md no comparteix codi amb
pendol/. Son trossos generics (trobar cims, ajustar una circumferencia) i no
fisica de cap dels dos sistemes, pero la decisio de moure'ls a un lloc comu
esta per prendre.

Us, des de l'ARREL del projecte. L'entrada bona es el .trk; el CSV hi es
perque les preses velles hi van sortir (vegeu llegeix_entrada):

    .venv\Scripts\python.exe eines\friccio_pendol_doble.py <presa>.trk --inercia 2.971e-3
    .venv\Scripts\python.exe eines\friccio_pendol_doble.py <presa>.trk --inclinacio
    .venv\Scripts\python.exe eines\friccio_pendol_doble.py <presa>.csv --sostres
    .venv\Scripts\python.exe eines\friccio_pendol_doble.py <presa>.trk --aire 0.00794 --cos lligat --inercia 2.971e-3
    .venv\Scripts\python.exe eines\friccio_pendol_doble.py --autoprova

Amb --inercia en surten tau_f i b en SI. La I NO es la mateixa a cada presa:
vegeu _inercies a parametres.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid as sci_cumulative_trapezoid
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# L'arrel al davant, com fan les altres eines de eines/: sense aixo l'ordre que
# hi ha escrita al capdamunt d'aquest fitxer no arrenca, perque en executar un
# script Python hi posa la carpeta del script i no el directori de treball, i
# llavors "magnetic" no es troba.
ARREL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARREL))

from magnetic.decrement import extrems, periode_aproximat  # noqa: E402
from magnetic.geometria import ajusta_circumferencia  # noqa: E402

G_VIC = 9.802  # gravetat a Vic; ha de coincidir amb la g de parametres.json
LLINDAR_ALT = 55.0   # graus: per sobre, el cim es massa pla (vegeu nota 2)
LLINDAR_BAIX = 8.0   # graus: per sota, l'amplitud toca el soroll del seguiment

# Finestra de Savitzky-Golay per a la derivada, EN TEMPS. Es va triar amb les
# preses de 240 fps, on son 13 mostres. A les de 60 fps (setembre del 2026) en
# surten 3, i el codi les puja a 5, que es el minim que admet un polinomi de
# grau 3: quatre graus de llibertat per a cinc punts, o sigui que gairebe no
# suavitza. Per a aquelles preses cal passar --finestra 0.12.
#
# Importa perque l'energia cinetica va amb omega^2 i el soroll d'omega hi entra
# com un BIAIX positiu, no com una dispersio que s'esborri promitjant; i com que
# el soroll creix amb la velocitat, el biaix es mes gros quan el pendol va
# rapid, que es exactament la forma que te la dissipacio que es vol mesurar.
FINESTRA_SAVGOL_S = 0.05


def envolupant(t, x, radi):
    """(instant, amplitud angular) de cada extrem, amb l'equilibri tret.

    L'equilibri es el punt mig entre cada extrem i el seguent: aixi no depen
    d'on hagis posat l'origen del Tracker.
    """
    tk, ak, _ = extrems(t, x, periode_aproximat(t, x))
    eq = 0.5 * (ak[:-1] + ak[1:])
    A = np.arcsin(np.clip(np.abs(ak[:-1] - eq) / radi, 0.0, 1.0))
    return tk[:-1], A, tk


def periode_a_amplitud_zero(tk, A, graus_lo=12.0, graus_hi=25.0):
    """T0 = T(amplitud petita) / (1 + theta^2/16). Retorna (T0, T mesurat, n)."""
    T_local = 2.0 * np.diff(tk)
    ang = A[:len(T_local)]
    m = (ang >= math.radians(graus_lo)) & (ang < math.radians(graus_hi))
    if m.sum() < 5:
        raise SystemExit(
            f"Nomes {m.sum()} extrems entre {graus_lo} i {graus_hi} graus: no hi ha\n"
            "prou tram d'amplitud petita per treure el periode. Grava mes estona."
        )
    T_mes = float(T_local[m].mean())
    theta = float(ang[m].mean())
    return T_mes / (1.0 + theta ** 2 / 16.0), T_mes, int(m.sum())


def finestra_neta(tt, A):
    """Talla el tram d'amplitud gran, amb els extrems espuris (vegeu nota 2)."""
    ultim_alt = np.max(np.flatnonzero(A >= math.radians(LLINDAR_ALT)), initial=-1)
    b = np.zeros_like(A, dtype=bool)
    b[ultim_alt + 1:] = True
    b &= (A > math.radians(LLINDAR_BAIX)) & (A < math.radians(LLINDAR_ALT))
    return b


def ajusta_tres_mecanismes(tt, A, c2_fixat=None):
    """Integra l'EDO i ajusta (c0, c1, c2). Retorna (coefs, errors, R2, rms, corr).

    Amb c2_fixat, el terme aerodinamic NO s'ajusta: es queda al valor donat i
    nomes es reajusten c0, c1 i l'amplitud inicial. Serveix per preguntar-li a
    les dades «i si l'aire valgues AIXO?», que es el que fa --aire.
    """
    lliure = c2_fixat is None

    def desplega(p):
        return (p[0], p[1], p[2], p[3]) if lliure else (p[0], p[1], c2_fixat, p[2])

    def model(p):
        c0, c1, c2, a0 = desplega(p)
        s = solve_ivp(lambda _, a: -(c0 + c1 * a[0] + c2 * a[0] ** 2),
                      (tt[0], tt[-1]), [a0], t_eval=tt, rtol=1e-9, atol=1e-12)
        return s.y[0] - A

    p0 = [0.01, 0.01, 0.01, A[0]] if lliure else [0.01, 0.01, A[0]]
    sol = least_squares(model, p0, bounds=([0.0] * len(p0), [np.inf] * len(p0)))
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * (2 * sol.cost / (len(A) - len(p0)))
    dia = np.sqrt(np.diag(cov))
    corr = cov / np.outer(dia, dia)
    r2 = 1.0 - sol.fun.var() / A.var()
    rms = math.degrees(math.sqrt(float(np.mean(sol.fun ** 2))))
    if lliure:
        return sol.x[:3], dia[:3], float(r2), rms, corr[:3, :3]
    c = np.array([sol.x[0], sol.x[1], c2_fixat])
    e = np.array([dia[0], dia[1], 0.0])
    return c, e, float(r2), rms, corr[:2, :2]


def llegeix_entrada(cami, radi=None):
    """Del .trk o d'un CSV a (t, x, y, radi, unitat, notes).

    EL .trk ES EL CAMI BO, i el CSV hi es perque les preses velles hi van sortir.
    Tres coses el fan millor, i la primera no es cosmetica:

    1. EL RADI. Amb el CSV surt de hypot(x,y).mean(), o sigui mesurat des d'on
       es va clicar l'origen al Tracker. A la presa 2 aixo dona 12,69 cm i la
       circumferencia ajustada en dona 12,49: un 1,6 % que entra a TOTES les
       amplituds, perque A = arcsin(|x-eq|/radi). Del .trk el radi surt de
       l'ajust de circumferencia, que no depen d'on siguin els eixos.
    2. LA INCLINACIO DE LA CAMERA es mesura i es treu, en comptes de confiar que
       l'angle de l'eix del Tracker estigui posat be. Vegeu inclinacio_del_trk.
    3. ELS TEMPS son els dels fotogrames de debo. Es fa servir el TRAM CONTINU
       MES LLARG i no s'inventa cap posicio: si el seguidor va perdre la marca,
       aquell tros no hi es, no s'interpola.

    Amb el .trk tot va en PIXELS, i tant se val: les amplituds son angulars i el
    periode es temps. L'escala del video no entra enlloc.
    """
    cami = str(cami)
    if not cami.lower().endswith(".trk"):
        d = pd.read_csv(cami, comment="#")
        t, x, y = (d[c].to_numpy() for c in ("t", "x", "y"))
        xc, yc, radi_ajustat, residu = ajusta_circumferencia(x, y)
        if radi is None:
            radi = float(np.hypot(x, y).mean())
        return t, x, y, radi, radi_ajustat, residu, "m", {"origen": "csv"}

    from pendol import trk as _trk

    d = _trk.llegeix_trk(cami)
    pistes = d["pistes"]
    if len(pistes) != 1:
        raise SystemExit(
            f"{cami} te {len(pistes)} pistes ({', '.join(pistes)}). Una presa de "
            "decaiment n'ha de tenir una de sola."
        )
    (nom_pista, (k, X, Y)), = pistes.items()

    # El tram continu mes llarg. Els forats del seguiment son reals i no
    # s'omplen: al .trk de la friccio 2 n'hi ha un de 7 s, i cau abans de la
    # finestra de l'ajust, o sigui que no costa res deixar-lo fora.
    talls = np.where(np.diff(k) > 1)[0]
    trams = np.split(np.arange(len(k)), talls + 1)
    tria = max(trams, key=len)
    k, X, Y = k[tria], X[tria], Y[tria]

    xc, yc, radi_ajustat, residu = ajusta_circumferencia(X, Y)
    x, y = X - xc, Y - yc
    t = (k - k.min()) * d["delta_t_s"]

    # Adrecar: girar fins que l'equilibri caigui recte avall.
    th = np.arctan2(y, x)
    cua = t > t.max() - 3.0
    eq = math.atan2(float(np.sin(th[cua]).mean()), float(np.cos(th[cua]).mean()))
    gir = math.pi / 2 - eq
    c_, s_ = math.cos(gir), math.sin(gir)
    x, y = x * c_ - y * s_, x * s_ + y * c_

    if radi is None:
        radi = float(radi_ajustat)
    return t, x, y, radi, radi_ajustat, residu, "px", {
        "origen": "trk",
        "pista": nom_pista,
        "trams_continus": len(trams),
        "punts_del_tram_triat": int(len(tria)),
        "inclinacio_de_la_camera_graus": math.degrees(eq) - 90.0,
        "angle_de_l_eix_al_trk_graus": d.get("angle_graus"),
    }


def analitza(cami, nom=None, radi=None, inercia=None):
    t, x, y, radi, radi_ajustat, residu, unitat, notes = llegeix_entrada(cami, radi)

    tt, A, tk = envolupant(t, x, radi)
    T0, T_mes, n_T = periode_a_amplitud_zero(tk, A)
    L_eq = G_VIC * T0 ** 2 / (4.0 * math.pi ** 2)

    b = finestra_neta(tt, A)
    tt_n, A_n = tt[b] - tt[b][0], A[b]
    c, err, r2, rms, corr = ajusta_tres_mecanismes(tt_n, A_n)

    res = {
        "entrada": str(cami),
        "unitat": unitat,
        "lectura": notes,
        "radi_de_la_marca": radi,
        "circumferencia": {"radi": radi_ajustat, "residu_rms": residu},
        "periode": {"T0_s": T0, "T_mesurat_12_25_graus_s": T_mes, "n_extrems": n_T},
        "I_partit_per_M_d_m": L_eq,
        "finestra_graus": [LLINDAR_BAIX, LLINDAR_ALT],
        "n_extrems_ajustats": int(b.sum()),
        "durada_ajustada_s": float(tt_n[-1]),
        "r2": r2, "residu_graus": rms,
        "c0_sec": {"valor": c[0], "error": err[0]},
        "c1_viscos": {"valor": c[1], "error": err[1]},
        "c2_aerodinamic": {"valor": c[2], "error": err[2]},
        "correlacions": corr.tolist(),
        "sense_massa": {
            "tau_f_partit_per_Mgd_graus": math.degrees(c[0] * T0 / 4.0),
            "sigma_s-1": c[1],
            "k_partit_per_I_rad-1": 3.0 * T0 * c[2] / 8.0,
        },
    }

    # Els coeficients en SI. Fins avui aquesta multiplicacio es feia a MA i
    # nomes en quedava el resultat a parametres.json, que es justament el que el
    # CLAUDE.md no vol: un numero de la memoria que no es pot tornar a calcular.
    #
    # tau_f = (c0*T0/4) * M*g*d      i      b = 2*I*sigma,   amb sigma = c1
    # i M*d = I / (I/(M*d)), que surt del periode.
    #
    # La I s'ha de donar de fora perque NO es la mateixa a cada presa: a la
    # presa 1 son les dues barres lligades sobre el pivot de dalt i a la 2 la
    # barra 2 sobre el seu. Fer que l'eina l'endevini seria demanar problemes.
    if inercia:
        Md = inercia / L_eq
        res["si"] = {
            "inercia_kg_m2": float(inercia),
            "M_per_d_kg_m": float(Md),
            "tau_f_Nm": float(c[0] * T0 / 4.0 * Md * G_VIC),
            "b_Nm_s_per_rad": float(2.0 * inercia * c[1]),
        }

    escriu(res)
    if nom:
        Path(nom).write_text(json.dumps(res, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        print(f"\n   numeros: {nom}")
    return res


def equilibri(cami):
    """On cau l'equilibri del pendol dins del marc del CSV, i que hi fa.

    El pendol penja de la gravetat: al marc bo, l'equilibri ha de caure recte
    avall, a -90 graus del centre de la circumferencia. Si el marc esta girat
    -perque la camera anava inclinada i l'angle de l'eix del Tracker no ho
    compensa- l'equilibri se'n va, i com que l'amplitud es mesura amb la
    PROJECCIO x, totes les amplituds queden multiplicades per cos(desviament).

    Es una comprovacio que val la pena tenir a ma: diu si el marc es bo amb un
    numero, en comptes de suposar-ho.
    """
    d = pd.read_csv(cami, comment="#")
    t, x, y = (d[c].to_numpy() for c in ("t", "x", "y"))
    xc, yc, radi, _ = ajusta_circumferencia(x, y)
    th = np.degrees(np.arctan2(y - yc, x - xc))
    # La cua: alli l'amplitud ja es petita i el promig ES l'equilibri.
    cua = t > t[-1] - 3.0
    eq = float(th[cua].mean())
    desviament = eq + 90.0
    return {
        "entrada": str(cami),
        "centre_m": [float(xc), float(yc)],
        "radi_m": float(radi),
        "equilibri_graus": eq,
        "desviament_de_la_vertical_graus": desviament,
        "factor_sobre_les_amplituds": math.cos(math.radians(desviament)),
    }


def inclinacio_del_trk(cami):
    """Quant esta inclinada la camera, dels PIXELS EN CRU i de res mes.

    No depen de l'origen, ni de l'angle de l'eix, ni de l'escala que hi hagi
    posats al Tracker: nomes de les marques i del fet que un pendol penja de la
    gravetat. S'ajusta la circumferencia que descriu la marca i es mira cap on
    apunta l'equilibri. Als pixels, avall es +y, o sigui que l'equilibri ha de
    caure a +90 graus exactes.

    Serveix per decidir que hi va, a l'angle de l'eix del Tracker. A les dues
    preses de friccio hi havia escrits -42,53 i -63,03 graus i la camera nomes
    va torta 0,86 i 3,25: aquells angles eren espuris i val mes deixar-ho a
    zero, que l'error que aixo comet es d'una part per mil.
    """
    from pendol import trk as _trk  # nomes aqui: la resta de l'eina va amb CSV

    d = _trk.llegeix_trk(cami)
    pistes = d["pistes"]
    if len(pistes) != 1:
        raise SystemExit(
            f"{cami} te {len(pistes)} pistes ({', '.join(pistes)}). Aixo espera una "
            "presa de decaiment, que en te una de sola."
        )
    (nom, (k, X, Y)), = pistes.items()
    xc, yc, radi, residu = ajusta_circumferencia(X, Y)
    th = np.arctan2(Y - yc, X - xc)
    t = (k - k.min()) * d["delta_t_s"]
    cua = t > t.max() - 3.0
    # Promig d'angles: per components, que la mitjana directa no val a prop de +-180.
    eq = math.degrees(math.atan2(float(np.sin(th[cua]).mean()),
                                 float(np.cos(th[cua]).mean())))
    incl = eq - 90.0
    return {
        "entrada": str(cami),
        "pista": nom,
        "centre_px": [float(xc), float(yc)],
        "radi_px": float(radi),
        "residu_px": float(residu),
        "equilibri_graus": eq,
        "inclinacio_de_la_camera_graus": incl,
        "factor_sobre_les_amplituds": math.cos(math.radians(incl)),
        "angle_de_l_eix_al_trk_graus": d.get("angle_graus"),
    }


def escriu_inclinacio(r):
    print("=" * 74)
    print(f" Inclinacio de la camera a {r['entrada']}")
    print("=" * 74)
    print(f"\n   pista {r['pista']!r}, circumferencia de radi {r['radi_px']:.1f} px "
          f"(residu {r['residu_px']:.2f} px)")
    print(f"   equilibri a {r['equilibri_graus']:+.2f} graus, i avall son +90")
    print(f"   -> la camera esta inclinada {r['inclinacio_de_la_camera_graus']:+.2f} graus")
    print(f"      les amplituds queden multiplicades per "
          f"{r['factor_sobre_les_amplituds']:.4f}")
    a = r["angle_de_l_eix_al_trk_graus"]
    if a is not None:
        print(f"\n   angle de l'eix escrit al .trk: {a:+.3f} graus")
        if abs(a - r["inclinacio_de_la_camera_graus"]) > 5.0 and abs(a) > 5.0:
            print("      NO es la inclinacio de la camera. Es un angle espuri, de")
            print("      quan s'escriu 0 a una vara que no esta horitzontal. Val mes")
            print("      deixar-lo a zero que posar-hi un numero inventat.")


def integral_de_cub(p, q):
    """int |s|^3 ds d'un tros de barra que va de s=p a s=q respecte del pivot.

    Es el factor geometric del parell d'aire d'una barra que NOMES gira:
    dF = C*v^2*ds amb v = s*w dona tau = C*w|w| * int |s|^3 ds. Va com s^4, o
    sigui que el que mana es l'extrem de fora i prou.
    """
    p, q = min(p, q), max(p, q)
    if p < 0 < q:
        return (abs(p) ** 4 + q ** 4) / 4.0
    return abs(q ** 4 - p ** 4) / 4.0


def geometria_del_cos(quin, fis):
    """(factor geometric en m^4, com esta fet) del cos que oscil.la a la presa.

    Les dues preses de decaiment NO mouen el mateix cos:
      lligat  -> presa 1: les dues barres cargolades, girant sobre el pivot 1.
      barra2  -> presa 2: la barra 2 sola, girant sobre el SEU pivot.
    """
    R1, L1 = fis["barra1"]["R"], fis["L1"]
    R2, Lx = fis["barra2"]["R"], fis["barra2"]["L_extrem"]
    dalt2 = R2 - Lx                      # el tros de barra 2 que sobresurt per sobre
    if quin == "barra2":
        return integral_de_cub(-dalt2, Lx), [
            ("barra 2 sobre el seu pivot", -dalt2, Lx)]
    # La barra 1 porta els dos forats separats L1 en una barra de R1: aixo els
    # posa per forca a (R1-L1)/2 de cada extrem. Vegeu _barra1_es_simetrica.
    vora = (R1 - L1) / 2.0
    trams = [("barra 1 sobre el pivot 1", -vora, L1 + vora),
             ("barra 2, penjant en linia", L1 - dalt2, L1 + Lx)]
    return sum(integral_de_cub(a, b) for _, a, b in trams), trams


def contra_l_aire(cami, C, quin, inercia, radi=None):
    """El c2 que aquesta presa HAURIA de tenir si l'aire fos el del balanc.

    La pregunta es una comprovacio creuada entre dos experiments que no s'havien
    comparat mai: el balanc d'energia de les preses caotiques resol una C que
    tanca el forat, i aquesta presa de decaiment mou EL MATEIX COS. Si aquella C
    es aire de debo, aqui s'ha de veure.

        tau_aire = C * (int |s|^3 ds) * w|w| = k*w|w|      i      c2 = 8k/(3*I*T0)

    Despres es reajusta la presa amb el c2 FIXAT al valor predit, per veure quant
    es degrada el residu: es la manera de preguntar-li a les dades si aquell aire
    hi cap o no.
    """
    fis = json.loads((ARREL / "parametres.json").read_text(encoding="utf-8"))["fisic"]
    G, trams = geometria_del_cos(quin, fis)

    # El radi de la marca pot anar en pixels i tant se val: l'amplitud A es un
    # arcsinus d'una RAO i surt en radians igual. Les longituds del parell d'aire
    # no venen del video sino del banc, per parametres.json.
    t, x, y, radi, _, _, unitat, notes = llegeix_entrada(cami, radi)
    tt, A, tk = envolupant(t, x, radi)
    T0, _, _ = periode_a_amplitud_zero(tk, A)
    b = finestra_neta(tt, A)
    tt_n, A_n = tt[b] - tt[b][0], A[b]

    c, err, _, rms, _ = ajusta_tres_mecanismes(tt_n, A_n)
    k = C * G
    c2_predit = 8.0 * k / (3.0 * inercia * T0)
    sig = (c2_predit - c[2]) / err[2] if err[2] else float("inf")

    cf, errf, _, rmsf, _ = ajusta_tres_mecanismes(tt_n, A_n, c2_fixat=c2_predit)

    return {
        "entrada": str(cami), "cos": quin, "C_del_balanc": C,
        "trams": trams, "factor_geometric_m4": G,
        "k_Nm_s2_per_rad2": k, "T0_s": T0, "inercia_kg_m2": inercia,
        "c2_predit": c2_predit,
        "c2_ajustat": {"valor": c[2], "error": err[2]},
        "distancia_sigmes": sig,
        "lliure": {"c0": c[0], "c1": c[1], "c2": c[2], "residu_graus": rms},
        "amb_c2_forcat": {"c0": cf[0], "c1": cf[1], "c2": cf[2], "residu_graus": rmsf},
        "el_residu_es_degrada": rmsf / rms,
    }


def escriu_aire(r):
    print("=" * 74)
    print(f" L'aire del balanc contra el decaiment de {r['entrada']}")
    print("=" * 74)
    print(f"\n   cos que oscil.la: {r['cos']}")
    for nom, a, b in r["trams"]:
        print(f"      {nom:<28} de {100*a:+6.1f} a {100*b:+6.1f} cm del pivot")
    print(f"   factor geometric int|s|^3 ds = {r['factor_geometric_m4']:.4e} m4")
    print(f"   amb C = {r['C_del_balanc']:.5f} del balanc  ->  k = "
          f"{r['k_Nm_s2_per_rad2']:.3e} N m s2/rad2")
    print(f"   i amb T0 = {r['T0_s']:.4f} s, I = {r['inercia_kg_m2']:.4e} kg m2:")
    print(f"\n   c2 que HAURIA de tenir : {r['c2_predit']:.5f}")
    print(f"   c2 que hi ha mesurat   : {r['c2_ajustat']['valor']:.5f} +- "
          f"{r['c2_ajustat']['error']:.5f}")
    print(f"   distancia              : {r['distancia_sigmes']:.1f} sigma")

    ll, fo = r["lliure"], r["amb_c2_forcat"]
    print("\n   i si s'hi forca aquell aire i es reajusta la resta:")
    print(f"      {'':<12} {'c0 SEC':>10} {'c1 VISCOS':>11} {'c2 AIRE':>10} {'residu':>9}")
    print(f"      {'ajust lliure':<12} {ll['c0']:10.5f} {ll['c1']:11.5f} "
          f"{ll['c2']:10.5f} {ll['residu_graus']:8.3f}o")
    print(f"      {'c2 forcat':<12} {fo['c0']:10.5f} {fo['c1']:11.5f} "
          f"{fo['c2']:10.5f} {fo['residu_graus']:8.3f}o")
    print(f"\n   el residu es degrada x{r['el_residu_es_degrada']:.2f}")
    if r["distancia_sigmes"] > 3.0:
        print("\n   !! LES DUES MESURES NO QUADREN. O aquella C no es aire, o l'aire")
        print("      no va com v^2 en tot aquest rang, o hi ha un tercer mecanisme")
        print("      que nomes surt a les preses caotiques. Vegeu la seccio 09 de")
        print("      documentacio/friccio-del-pendol-doble.html.")
    else:
        print("\n   les dues mesures son compatibles: la C del balanc SI que es pot")
        print("   llegir com a aire sobre les barres.")


# ------------------------------------------------- l'ajust per energia


def angle_i_velocitat(t, x, y, finestra_s=FINESTRA_SAVGOL_S, ordre=3):
    """(theta des de la vertical inferior, omega) de la marca, sense cap cim.

    llegeix_entrada ja ha centrat la marca al centre de la circumferencia i ha
    girat el marc perque l'equilibri caigui a +90 graus, o sigui que l'angle del
    pendol es atan2(y,x) - pi/2. Es desembolica, com mana el CLAUDE.md.

    La derivada, per Savitzky-Golay amb els MATEIXOS parametres que
    pendol/tracker.py: ajusta un polinomi local i en deriva el polinomi, en
    comptes d'amplificar el soroll punt a punt.
    """
    from scipy.signal import savgol_filter

    fps = (len(t) - 1) / (t[-1] - t[0])
    dt = 1.0 / fps
    n = int(round(finestra_s * fps))
    n = max(ordre + 2, n + (n % 2 == 0))
    if n >= len(t):
        n = len(t) - 1 - (len(t) % 2 == 0)
    th = np.unwrap(np.arctan2(y, x)) - math.pi / 2.0
    w = savgol_filter(th, n, ordre, deriv=1, delta=dt)
    return th, w, fps, n


def ajusta_per_energia(t, th, w, L_eq, sostre_graus=None, terra_graus=LLINDAR_BAIX):
    """Els tres mecanismes, de la caiguda d'ENERGIA fotograma a fotograma.

    PER QUE, si ja hi ha ajusta_tres_mecanismes. Perque aquell necessita trobar
    els extrems, i per sobre d'uns 55 graus el cim es tan pla que el detector
    se'ls inventa: aixo obliga a llencar el tram rapid, que es justament el que
    s'assembla al pendol caotic. Aqui no hi ha cap cim per trobar.

    Per unitat d'inercia, i amb M*g*d/I = g/L_eq, que el periode ja dona:

        e = (g/L_eq)*(1 - cos th) + 0.5*w^2
        de/dt = -( s*|w| + v*w^2 + a*|w|^3 )      s = tau_f/I, v = b/I, a = k/I

    NO es deriva e, que seria tornar a amplificar soroll: s'integra el membre de
    la dreta i s'ajusta e(t) sencera, que deixa el problema LINEAL en els tres
    coeficients:

        e(t) = e0 - s*Int|w| - v*Int w^2 - a*Int|w|^3

    L'amplitud, per posar-hi finestra, tambe surt de l'energia i no de cap cim:
    al punt de retorn w = 0, o sigui A(t) = arccos(1 - e*L_eq/g).

    LIMIT DECLARAT: les barres d'error formals d'aquest ajust surten optimistes,
    perque les mostres consecutives NO son independents -el residu es una corba
    que deriva, no soroll blanc-. S'inflen per l'arrel de N/N_efectiu, amb
    N_efectiu = el nombre de mitjos cicles, que es el que l'altre metode fa
    servir de debo. Es una corrreccio declarada, no una mesura de la correlacio.
    """
    from scipy.optimize import lsq_linear

    e = (G_VIC / L_eq) * (1.0 - np.cos(th)) + 0.5 * w ** 2
    A = np.arccos(np.clip(1.0 - e * L_eq / G_VIC, -1.0, 1.0))

    b = A > math.radians(terra_graus)
    if sostre_graus is not None:
        # Igual que finestra_neta: es talla TOT el que hi ha abans de l'ultim
        # cop que es passa pel sostre, no nomes els punts de sobre.
        alt = np.max(np.flatnonzero(A >= math.radians(sostre_graus)), initial=-1)
        b[:alt + 1] = False
    if b.sum() < 100:
        raise SystemExit(f"Nomes {b.sum()} punts a la finestra: no n'hi ha prou.")
    # El tram que es fa servir ha de ser CONTINU: si el pendol torna a pujar per
    # sobre del terra despres d'haver-lo passat, no hi ha res mes a mesurar.
    i0 = int(np.flatnonzero(b)[0])
    i1 = i0 + int(np.argmin(b[i0:])) if not b[i0:].all() else len(b)
    sl = slice(i0, i1)

    tt, ee, ww = t[sl], e[sl], np.abs(w[sl])
    M = np.column_stack([
        np.ones_like(tt),
        -sci_cumulative_trapezoid(ww, tt, initial=0.0),
        -sci_cumulative_trapezoid(ww ** 2, tt, initial=0.0),
        -sci_cumulative_trapezoid(ww ** 3, tt, initial=0.0),
    ])
    sol = lsq_linear(M, ee, bounds=([-np.inf, 0, 0, 0], np.inf))
    res = M @ sol.x - ee
    gl = max(1, len(ee) - 4)
    cov = np.linalg.pinv(M.T @ M) * float(res @ res) / gl
    err = np.sqrt(np.diag(cov))[1:]

    # Les mostres no son independents: vegeu el limit declarat de la docstring.
    T0 = 2.0 * math.pi * math.sqrt(L_eq / G_VIC)
    n_efectiu = max(2.0, 2.0 * (tt[-1] - tt[0]) / T0)
    err = err * math.sqrt(len(ee) / n_efectiu)

    return {
        "s_sec_rad_s-2": float(sol.x[1]), "v_viscos_s-1": float(sol.x[2]),
        "a_aerodinamic_rad-1": float(sol.x[3]),
        "errors": [float(x) for x in err],
        "n_punts": int(len(ee)), "durada_s": float(tt[-1] - tt[0]),
        "amplitud_graus": [math.degrees(A[sl].min()), math.degrees(A[sl].max())],
        "w_max_rad_s": float(np.abs(w[sl]).max()),
        "residu_relatiu": float(np.sqrt(float(res @ res) / len(ee)) / ee.mean()),
        "n_efectiu": float(n_efectiu),
    }


def per_energia(cami, radi=None, inercia=None, sostre=None,
                finestra_s=FINESTRA_SAVGOL_S):
    """L'ajust per energia d'una presa, i la comparacio amb el d'extrems.

    La comparacio NO es decorativa: es la prova que diu si el metode nou entra o
    no. A la MATEIXA finestra 8-55 graus els dos han de donar el mateix; si no
    ho fan, el metode nou es dolent i no es fa servir.

    finestra_s es la finestra de Savitzky-Golay EN TEMPS, i importa quins fps te
    la presa: vegeu FINESTRA_SAVGOL_S.
    """
    t, x, y, radi, _, _, unitat, notes = llegeix_entrada(cami, radi)
    th, w, fps, finestra = angle_i_velocitat(t, x, y, finestra_s=finestra_s)

    tt, A, tk = envolupant(t, x, radi)
    T0, _, _ = periode_a_amplitud_zero(tk, A)
    L_eq = G_VIC * T0 ** 2 / (4.0 * math.pi ** 2)

    b = finestra_neta(tt, A)
    c, cerr, _, rms_ex, _ = ajusta_tres_mecanismes(tt[b] - tt[b][0], A[b])
    # Els coeficients de l'ajust d'extrems, passats a les mateixes unitats.
    extrems_en_energia = (c[0] * G_VIC * T0 / (4.0 * L_eq),
                          2.0 * c[1], 3.0 * T0 * c[2] / 8.0)
    err_en_energia = (cerr[0] * G_VIC * T0 / (4.0 * L_eq),
                      2.0 * cerr[1], 3.0 * T0 * cerr[2] / 8.0)

    return {
        "entrada": str(cami), "unitat": unitat, "lectura": notes,
        "T0_s": T0, "I_partit_per_M_d_m": L_eq, "fps": fps,
        "finestra_savgol_mostres": finestra,
        "finestra_savgol_s": finestra_s,
        "extrems": {"valors": list(extrems_en_energia), "errors": list(err_en_energia),
                    "residu_graus": rms_ex, "n_extrems": int(b.sum())},
        "energia_mateixa_finestra": ajusta_per_energia(t, th, w, L_eq,
                                                       sostre_graus=LLINDAR_ALT),
        "energia_sencera": ajusta_per_energia(t, th, w, L_eq, sostre_graus=sostre),
        "inercia_kg_m2": inercia,
    }


def escriu_energia(r):
    ex, mf, se = r["extrems"], r["energia_mateixa_finestra"], r["energia_sencera"]
    print("=" * 74)
    print(f" Friccio per ENERGIA (sense extrems) de {r['entrada']}")
    print("=" * 74)
    print(f"\n   T0 = {r['T0_s']:.4f} s   I/(M d) = {100*r['I_partit_per_M_d_m']:.2f} cm"
          f"   {r['fps']:.1f} fps   Savitzky-Golay de {r['finestra_savgol_mostres']} mostres"
          f" ({1000 * r['finestra_savgol_mostres'] / r['fps']:.0f} ms)")

    def fila(et, v, e, extra=""):
        print(f"   {et:<26} {v[0]:9.5f} {v[1]:9.5f} {v[2]:9.5f}   {extra}")

    print(f"\n   {'':<26} {'s SEC':>9} {'v VISCOS':>9} {'a AIRE':>9}")
    print(f"   {'':<26} {'rad/s2':>9} {'s-1':>9} {'rad-1':>9}")
    fila("per extrems, 8-55o", ex["valors"], ex["errors"],
         f"{ex['n_extrems']} extrems")
    fila("  +- ", ex["errors"], ex["errors"])
    fila("per energia, 8-55o", [mf["s_sec_rad_s-2"], mf["v_viscos_s-1"],
                                mf["a_aerodinamic_rad-1"]], mf["errors"],
         f"{mf['n_punts']} punts, w fins a {mf['w_max_rad_s']:.1f} rad/s")
    fila("  +- ", mf["errors"], mf["errors"])

    print("\n   LA PROVA: els dos han de coincidir a la mateixa finestra")
    noms = ("s SEC ", "v VISC", "a AIRE")
    tot_ok = True
    for i, nom in enumerate(noms):
        a, b_ = ex["valors"][i], [mf["s_sec_rad_s-2"], mf["v_viscos_s-1"],
                                  mf["a_aerodinamic_rad-1"]][i]
        sig = math.hypot(ex["errors"][i], mf["errors"][i])
        d = abs(a - b_) / sig if sig else float("inf")
        ok = d < 3.0
        tot_ok &= ok
        print(f"      {nom}: {a:9.5f} contra {b_:9.5f}   {d:5.1f} sigma"
              f"   {'ok' if ok else '<<< NO COINCIDEIXEN'}")

    print(f"\n   i ara el registre SENCER, que es el que es guanya:")
    fila("per energia, tot", [se["s_sec_rad_s-2"], se["v_viscos_s-1"],
                             se["a_aerodinamic_rad-1"]], se["errors"],
         f"{se['n_punts']} punts")
    fila("  +- ", se["errors"], se["errors"])
    print(f"      amplituds de {se['amplitud_graus'][0]:.1f} a "
          f"{se['amplitud_graus'][1]:.1f} graus, {se['durada_s']:.1f} s")
    print(f"      w arriba a {se['w_max_rad_s']:.1f} rad/s, contra els "
          f"{mf['w_max_rad_s']:.1f} de la finestra de 55o")

    if r.get("inercia_kg_m2"):
        I = r["inercia_kg_m2"]
        print(f"\n   en SI amb I = {I:.4e} kg m2:")
        print(f"      tau_f = {se['s_sec_rad_s-2'] * I:.4e} N m")
        print(f"      b     = {se['v_viscos_s-1'] * I:.4e} N m s/rad")
        print(f"      k     = {se['a_aerodinamic_rad-1'] * I:.4e} N m s2/rad2")

    if not tot_ok:
        print("\n   !! ELS DOS METODES NO COINCIDEIXEN a la finestra comuna. El")
        print("      metode per energia NO s'ha de fer servir mentre aixo no")
        print("      s'expliqui: no val que doni un resultat mes maco.")


def sostres(cami, llista=(55, 45, 35, 30, 25), radi=None):
    """Que passaria si la presa s'hagues amollat de mes avall.

    NO cal tornar a gravar per saber-ho: un tir amollat de 84 graus JA PASSA per
    30, o sigui que el tram de sota de 30 es exactament el que hauria donat una
    presa amollada d'alli. Refent l'ajust amb cada sostre en surt la resposta.

    I la resposta es que amollar mes avall EMPITJORA el repartiment entre els
    tres mecanismes. No es cap sorpresa un cop dit: els tres regressors (1, A,
    A^2) tenen correlacions de -0.99 i el que els separa es el RANG d'amplituds.
    Escurcar-lo per dalt es el pitjor que se li pot fer.

    Aixo desmenteix el consell que hi havia escrit a documentacio/ de refer la
    presa 2 amollant des de 30 graus. Vegeu documentacio/friccio-del-pendol-doble.html.
    """
    d = pd.read_csv(cami, comment="#")
    t, x, y = (d[c].to_numpy() for c in ("t", "x", "y"))
    if radi is None:
        radi = float(np.hypot(x, y).mean())
    tt, A, tk = envolupant(t, x, radi)
    base = finestra_neta(tt, A)

    files = []
    for sostre in llista:
        m = base & (A < math.radians(sostre))
        if m.sum() < 10:
            files.append({"sostre_graus": sostre, "n_extrems": int(m.sum()),
                          "prou": False})
            continue
        tt_n, A_n = tt[m] - tt[m][0], A[m]
        c, err, r2, rms, _ = ajusta_tres_mecanismes(tt_n, A_n)
        files.append({
            "sostre_graus": sostre,
            "n_extrems": int(m.sum()),
            "durada_s": float(tt_n[-1]),
            "prou": True,
            "c0_sec": {"valor": c[0], "error": err[0]},
            "c1_viscos": {"valor": c[1], "error": err[1]},
            "c2_aerodinamic": {"valor": c[2], "error": err[2]},
            "sigma_c1": float(c[1] / err[1]) if err[1] else float("nan"),
            "r2": r2,
        })
    return {"entrada": str(cami), "radi_m": radi, "sostres": files}


def escriu_sostres(r):
    print("=" * 74)
    print(f" Que passaria si {r['entrada']} s'hagues amollat de mes avall")
    print("=" * 74)
    print("\nEl tir ja passa per tots aquests angles, o sigui que el tram de sota de")
    print("cadascun ES el que hauria donat una presa amollada d'alli. No cal gravar.\n")
    print(f"   {'amollat de':>11} {'extrems':>8} {'durada':>8}   "
          f"{'c1 VISCOS':>21} {'sigma':>6}")
    print("   " + "-" * 60)
    for f in r["sostres"]:
        if not f["prou"]:
            print(f"   {f['sostre_graus']:>8} deg   nomes {f['n_extrems']} extrems")
            continue
        v, e = f["c1_viscos"]["valor"], f["c1_viscos"]["error"]
        print(f"   {f['sostre_graus']:>8} deg {f['n_extrems']:8} {f['durada_s']:7.1f}s   "
              f"{v:.5f} +- {e:.5f} {f['sigma_c1']:6.1f}")
    bons = [f for f in r["sostres"] if f["prou"]]
    if len(bons) >= 2:
        alt, baix = bons[0], bons[-1]
        pitjor = baix["c1_viscos"]["error"] / alt["c1_viscos"]["error"]
        print(f"\n   De {alt['sostre_graus']} a {baix['sostre_graus']} graus, la incertesa "
              f"de c1 es multiplica per {pitjor:.1f}.")
        print("   Amollar mes avall NO arregla el viscos: l'empitjora. El que separa")
        print("   els tres mecanismes es el RANG d'amplituds, i escurcar-lo per dalt")
        print("   es el pitjor que se li pot fer.")


def escriu_equilibri(r):
    d = r["desviament_de_la_vertical_graus"]
    print(f"\nL'equilibri, dins del marc del CSV")
    print(f"   centre del cercle     ({100*r['centre_m'][0]:+.2f}, "
          f"{100*r['centre_m'][1]:+.2f}) cm, radi {100*r['radi_m']:.2f} cm")
    print(f"   equilibri             {r['equilibri_graus']:+.2f} graus "
          f"(recte avall serien -90)")
    print(f"   desviament            {d:+.2f} graus  ->  les amplituds queden "
          f"multiplicades per {r['factor_sobre_les_amplituds']:.4f}")
    if abs(d) > 3.0:
        print("      ATENCIO: el marc esta girat de debo. La camera anava inclinada")
        print("      i l'angle de l'eix del Tracker no ho compensa, i aixo esbiaixa")
        print("      TOTES les amplituds cap avall.")


def escriu(r):
    c = (r["c0_sec"], r["c1_viscos"], r["c2_aerodinamic"])
    print("=" * 74)
    print(f" {r['entrada']}")
    print("=" * 74)
    px = r.get("unitat") == "px"
    fac, u, ures = (1.0, "px", "px") if px else (100.0, "cm", "mm")
    fres = 1.0 if px else 1000.0
    print("\nLa geometria")
    n = r.get("lectura", {})
    if n.get("origen") == "trk":
        print(f"   del .trk, pista {n['pista']!r}: {n['punts_del_tram_triat']} punts "
              f"del tram continu mes llarg (n'hi ha {n['trams_continus']})")
        print(f"   camera inclinada      {n['inclinacio_de_la_camera_graus']:+.2f} graus, "
              f"i s'ha adrecat")
    print(f"   radi de la marca      {r['radi_de_la_marca'] * fac:.2f} {u}")
    print(f"   circumferencia        radi {r['circumferencia']['radi'] * fac:.2f} {u}, "
          f"residu {r['circumferencia']['residu_rms'] * fres:.2f} {ures}")
    print("\nEl rellotge")
    print(f"   T mesurat 12-25 graus {r['periode']['T_mesurat_12_25_graus_s']:.4f} s "
          f"({r['periode']['n_extrems']} extrems)")
    print(f"   T0 a amplitud zero    {r['periode']['T0_s']:.4f} s")
    print(f"   ==> I/(M*d)           {r['I_partit_per_M_d_m'] * 100:.2f} cm")
    print(f"\nEls tres mecanismes   ({r['n_extrems_ajustats']} extrems, "
          f"{r['durada_ajustada_s']:.1f} s, {r['finestra_graus'][0]:.0f}-"
          f"{r['finestra_graus'][1]:.0f} graus)")
    print(f"   R2 = {r['r2']:.5f}   residu {r['residu_graus']:.3f} graus")
    for et, v in zip(("c0  SEC        ", "c1  VISCOS     ", "c2  AERODINAMIC"), c):
        sig = v["valor"] / v["error"] if v["error"] else 0.0
        marca = "   <- NO es detecta" if sig < 2.0 else ""
        print(f"   {et} = {v['valor']:.5f} +- {v['error']:.5f}  ({sig:4.1f} sigma){marca}")
    sm = r["sense_massa"]
    print("\n   sense massa (el coeficient dimensional surt amb la balanca):")
    print(f"      sec         tau_f/(M g d) = {sm['tau_f_partit_per_Mgd_graus']:.4f} graus")
    print(f"      viscos      sigma         = {sm['sigma_s-1']:.5f} s-1")
    print(f"      aerodinamic k/I           = {sm['k_partit_per_I_rad-1']:.5f} rad-1")
    print("\n   !! els tres regressors son gairebe degenerats: correlacio de "
          f"{r['correlacions'][1][2]:+.3f} entre")
    print("      viscos i aerodinamic. La caiguda TOTAL esta ben mesurada; el")
    print("      REPARTIMENT entre mecanismes, molt pitjor del que diuen les barres.")
    if "si" in r:
        s = r["si"]
        print(f"\n   amb I = {s['inercia_kg_m2']:.4e} kg m2  ->  M*d = "
              f"{s['M_per_d_kg_m']:.5f} kg m")
        print(f"      tau_f = {s['tau_f_Nm']:.4e} N m       (fregament sec)")
        print(f"      b     = {s['b_Nm_s_per_rad']:.4e} N m s/rad (viscos)")

    print("\n   part de la caiguda de cada mecanisme, per amplitud:")
    print(f"      {'amplitud':>9} {'SEC':>9} {'VISCOS':>9} {'AIRE':>9}")
    for g in (60, 45, 30, 20, 10, 5):
        a = math.radians(g)
        p = np.array([c[0]["valor"], c[1]["valor"] * a, c[2]["valor"] * a ** 2])
        s = p.sum()
        print(f"      {g:7d}o {100 * p[0] / s:8.1f} {100 * p[1] / s:9.1f} "
              f"{100 * p[2] / s:9.1f}")


def autoprova():
    """Genera un decaiment de coeficients CONEGUTS i comprova que els recupera."""
    c_ver = (0.006, 0.014, 0.010)
    T, radi = 0.80, 0.2313
    t = np.arange(0.0, 90.0, 1.0 / 239.688)
    sol = solve_ivp(lambda _, a: -(c_ver[0] + c_ver[1] * a[0] + c_ver[2] * a[0] ** 2),
                    (t[0], t[-1]), [math.radians(85.0)], t_eval=t,
                    rtol=1e-10, atol=1e-13)
    A = np.clip(sol.y[0], 0.0, None)
    rng = np.random.default_rng(0)
    x = radi * np.sin(A * np.cos(2 * np.pi * t / T)) + rng.normal(0.0, 0.00053, t.size)
    tt, Am, tk = envolupant(t, x, radi)
    T0, _, _ = periode_a_amplitud_zero(tk, Am)
    b = finestra_neta(tt, Am)
    c, err, r2, rms, _ = ajusta_tres_mecanismes(tt[b] - tt[b][0], Am[b])

    print("Autoprova -- decaiment sintetic de coeficients coneguts")
    print(f"   soroll afegit: 1 pixel (0.53 mm) a {radi * 100:.1f} cm")
    print(f"   T0     verdader {T:.4f}   recuperat {T0:.4f}   error {abs(T0 / T - 1):.2%}")
    ok = abs(T0 / T - 1) < 0.01
    for et, v, e, ver in zip(("c0 SEC", "c1 VIS", "c2 AER"), c, err, c_ver):
        dif = abs(v - ver) / ver
        ok &= dif < 0.35
        print(f"   {et} verdader {ver:.5f}   recuperat {v:.5f} +- {e:.5f}   "
              f"error {dif:6.1%}")
    print(f"   R2 = {r2:.5f}, residu {rms:.3f} graus")

    # El cami de --aire: ajustar amb el c2 FIXAT. Es codi que sosté un numero de
    # la memoria, o sigui que ha de tenir la seva prova. Dues coses:
    #   - fixant el c2 VERDADER, els altres dos han de sortir on toca
    #   - fixant-ne un de dolent, el residu s'ha de degradar; si no ho fes, la
    #     comparacio de --aire no voldria dir res
    tt_n, A_n = tt[b] - tt[b][0], Am[b]
    cb, _, _, rms_b, _ = ajusta_tres_mecanismes(tt_n, A_n, c2_fixat=c_ver[2])
    cd, _, _, rms_d, _ = ajusta_tres_mecanismes(tt_n, A_n, c2_fixat=4.0 * c_ver[2])
    print("\n   ajust amb el c2 FIXAT (el cami de --aire)")
    for et, v, ver in zip(("c0 SEC", "c1 VIS"), cb, c_ver):
        dif = abs(v - ver) / ver
        ok &= dif < 0.35
        print(f"      c2 al valor bo:  {et} {v:.5f} contra {ver:.5f}, "
              f"error {dif:5.1%}")
    print(f"      residu amb el c2 bo {rms_b:.3f}o, amb un c2 x4 {rms_d:.3f}o "
          f"(x{rms_d / rms_b:.1f})")
    ok &= rms_d > 2.0 * rms_b

    ok &= autoprova_energia()

    print("\n" + ("OK" if ok else "MALAMENT: l'eina no recupera el que sap que hi ha"))
    return 0 if ok else 1


def autoprova_energia():
    """Un pendol de debo amb els tres mecanismes: l'ajust per energia els recupera?

    Aixo NO es el mateix que l'altra autoprova. Alli es genera una envolupant
    d'amplituds que ja compleix l'EDO del decaiment; aqui s'integra l'equacio
    del PENDOL sencera, amb els tres parells, i despres se'n fan els pixels que
    veuria la camera. Es una prova d'extrem a extrem: si el conveni d'angles,
    la derivada de Savitzky-Golay o el balanc d'energia estan malament, peta.

        th'' = -(g/L)*sin(th) - s*signe(th') - v*th' - a*th'*|th'|

    S'amolla de 140 graus expres: es el regim que el metode vell no pot veure,
    perque arcsin(|x-eq|/radi) satura a 90.
    """
    from scipy.signal import savgol_filter  # noqa: F401  (el fa servir angle_i_velocitat)

    s_ver, v_ver, a_ver = 0.55, 0.020, 0.0064
    L, radi, fps = 0.1067, 237.0, 239.688
    t = np.arange(0.0, 30.0, 1.0 / fps)
    wn2 = G_VIC / L

    def f(_, y):
        th, w = y
        return [w, -wn2 * math.sin(th) - s_ver * np.tanh(w / 1e-4)
                - v_ver * w - a_ver * w * abs(w)]

    sol = solve_ivp(f, (t[0], t[-1]), [math.radians(140.0), 0.0], t_eval=t,
                    rtol=1e-10, atol=1e-12)
    th_ver = sol.y[0]

    # Els pixels que veuria la camera, amb el conveni de llegeix_entrada:
    # l'equilibri a +90 graus, o sigui th = atan2(y,x) - pi/2.
    rng = np.random.default_rng(0)
    x = radi * np.cos(th_ver + math.pi / 2) + rng.normal(0.0, 0.5, t.size)
    y = radi * np.sin(th_ver + math.pi / 2) + rng.normal(0.0, 0.5, t.size)

    th, w, _, n = angle_i_velocitat(t, x, y)
    r = ajusta_per_energia(t, th, w, L)

    print("\nAutoprova de l'ajust per ENERGIA -- pendol integrat de debo")
    print(f"   amollat de 140 graus, soroll d'1/2 pixel, finestra SG de {n} mostres")
    print(f"   {'':<12} {'verdader':>10} {'recuperat':>11} {'error':>8}")
    ok = True
    parells = (("s SEC", s_ver, r["s_sec_rad_s-2"], r["errors"][0]),
               ("v VISCOS", v_ver, r["v_viscos_s-1"], r["errors"][1]),
               ("a AIRE", a_ver, r["a_aerodinamic_rad-1"], r["errors"][2]))
    for et, ver, val, err in parells:
        dif = abs(val - ver) / ver
        ok &= dif < 0.30
        print(f"   {et:<12} {ver:10.5f} {val:11.5f} {dif:7.1%}")
    print(f"   amplituds de {r['amplitud_graus'][0]:.0f} a "
          f"{r['amplitud_graus'][1]:.0f} graus, w fins a {r['w_max_rad_s']:.1f} rad/s")
    if not ok:
        print("   MALAMENT: l'ajust per energia no recupera el que sap que hi ha")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("entrada", nargs="?",
                    help="el .trk de la presa de decaiment (recomanat) o un CSV")
    ap.add_argument("--nom", help="on desar el JSON amb els numeros")
    ap.add_argument("--radi", type=float,
                    help="radi de la marca (per defecte, de les dades)")
    ap.add_argument("--inercia", type=float, metavar="I",
                    help="moment d'inercia del brac sobre el seu pivot, en kg m2. "
                         "Si es dona, en surten tau_f i b en SI. NO es la mateixa a "
                         "cada presa: vegeu _inercies a parametres.json")
    ap.add_argument("--sostres", action="store_true",
                    help="que passaria si la presa s'hagues amollat de mes avall: "
                         "refa l'ajust amb sostres de 55, 45, 35, 30 i 25 graus")
    ap.add_argument("--inclinacio", action="store_true",
                    help="quant esta inclinada la camera, dels pixels en cru d'un .trk")
    ap.add_argument("--aire", type=float, metavar="C",
                    help="comprovacio creuada: donada la C del balanc d'energia "
                         "(eines/balanc_energia.py), quin terme aerodinamic hauria "
                         "de veure aquesta presa. Demana --cos i --inercia")
    ap.add_argument("--energia", action="store_true",
                    help="ajusta la friccio per la caiguda d'ENERGIA fotograma a "
                         "fotograma en comptes de per l'envolupant d'extrems. No "
                         "necessita cap cim, o sigui que aprofita el registre "
                         "sencer i no el tram de sota de 55 graus")
    ap.add_argument("--finestra", type=float, metavar="S",
                    default=FINESTRA_SAVGOL_S,
                    help="amb --energia, finestra de Savitzky-Golay EN SEGONS "
                         f"(per defecte {FINESTRA_SAVGOL_S}). A 240 fps son 13 "
                         "mostres; a 60 fps nomes 5, que gairebe no suavitzen. "
                         "Per a les preses de 60 fps: --finestra 0.12")
    ap.add_argument("--sostre", type=float, metavar="G",
                    help="amb --energia, sostre d'amplitud en graus (per defecte, "
                         "cap: el registre sencer)")
    ap.add_argument("--cos", choices=("lligat", "barra2"), default="lligat",
                    help="quin cos oscil.la: 'lligat' les dues barres cargolades "
                         "(presa 1), 'barra2' la barra 2 sola (presa 2)")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova l'eina amb un decaiment de coeficients coneguts")
    a = ap.parse_args(argv)
    if a.autoprova:
        return autoprova()
    if not a.entrada:
        ap.error("cal una entrada, o --autoprova")

    if a.inclinacio:
        r = inclinacio_del_trk(a.entrada)
        escriu_inclinacio(r)
        if a.nom:
            Path(a.nom).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"\n   numeros: {a.nom}")
        return 0

    if a.energia:
        r = per_energia(a.entrada, radi=a.radi, inercia=a.inercia, sostre=a.sostre,
                        finestra_s=a.finestra)
        escriu_energia(r)
        if a.nom:
            Path(a.nom).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"\n   numeros: {a.nom}")
        return 0

    if a.aire is not None:
        if not a.inercia:
            ap.error("--aire necessita tambe --inercia: el c2 predit va dividit "
                     "per la I del cos que oscil.la")
        r = contra_l_aire(a.entrada, a.aire, a.cos, a.inercia, radi=a.radi)
        escriu_aire(r)
        if a.nom:
            Path(a.nom).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"\n   numeros: {a.nom}")
        return 0

    if a.sostres:
        r = sostres(a.entrada, radi=a.radi)
        r["equilibri"] = equilibri(a.entrada)
        escriu_sostres(r)
        escriu_equilibri(r["equilibri"])
        if a.nom:
            Path(a.nom).write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
            print(f"\n   numeros: {a.nom}")
        return 0

    analitza(a.entrada, a.nom, a.radi, a.inercia)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
