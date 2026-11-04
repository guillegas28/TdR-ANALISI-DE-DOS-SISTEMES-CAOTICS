r"""Simulador del pendol doble FISIC. Integra i exporta en format canonic.

Dues barres amb la massa repartida. Per defecte SENSE FRICCIO: vegeu
pendol/fisica.py. Amb --friccio hi entren els cinc termes dissipatius del pas 3,
pero la C de l'aire i la eps del tanh s'han de donar a ma, perque encara no
estan decidides:

    .venv\Scripts\python -m pendol.simulador --theta1 120 --temps 20 \
        --friccio --aire-C 0.00794 --epsilon 1e-3 --sortida dades/sim/tir01_friccio.csv

Exemples
--------
Un sol tir de 20 s deixat anar des de 120 graus:

    .venv\\Scripts\\python -m pendol.simulador --theta1 120 --temps 20 \\
        --sortida dades/sim/tir01.csv

Un FEIX de 200 tirs amb la incertesa real de l'angle de sortida (+-1 grau).
Aquesta es la manera honesta de comparar amb la realitat: no pots exigir que
una sola trajectoria simulada segueixi la real, perque no saps l'angle inicial
amb precisio infinita. El que si pots exigir es que la real caigui DINS el feix.

    .venv\\Scripts\\python -m pendol.simulador --theta1 120 --temps 20 \\
        --conjunt 200 --incertesa 1.0 --sortida dades/sim/feix_120graus/

EL RELLOTGE D'UNA PRESA. Amb --com-el-tracker la simulacio dona EXACTAMENT els
mateixos instants que un CSV del Tracker, comptats des de l'amollada d'aquella
presa, i surt de les seves condicions inicials mesurades si no en dius cap
altra. Llavors la comparacio es punt a punt i no cal interpolar res -ni,
sobretot, tocar les dades mesurades-:

    .venv\\Scripts\\python -m pendol.simulador --com-el-tracker dades/net/150_3.csv \\
        --sortida dades/sim/150_3_simulat.csv

    .venv\\Scripts\\python -m pendol.simulador --autoprova
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pendol import dades
from pendol.fisica import Dissipacio, Friccio, Parametres, derivades, energia

# Tolerancies molt estretes: en un sistema caotic l'error del integrador
# creix exponencialment igual que qualsevol altra pertorbacio. Amb aquests
# valors, l'error numeric queda molt per sota de la incertesa experimental,
# de manera que la divergencia que mesuris sera FISICA i no numerica.
#
# Triades mesurant (100 s, sortida des de 179 graus, sense friccio):
#   rtol=1e-11 -> deriva d'energia 1.6e-09 en 2.2 s de calcul
#   rtol=1e-13 -> deriva d'energia 1.2e-11 en 3.3 s de calcul   <-- escollida
#   Radau      -> 3.6e-12 pero 121 s de calcul: no compensa
RTOL = 1e-13
ATOL = 1e-15
METODE = "DOP853"

# DOS CAMINS DE CALCUL, i cal saber per que n'hi ha dos.
#
# Integrar 15 s amb la tolerancia estricta costa 735 ms, que per a una
# finestra que ha de dibuixar el pendol movent-se es nota. Mesurat el
# 23/09/2026 (15 s des de 150 graus, sense friccio):
#
#     rtol      costa    deriva d'energia   equival a
#     1e-13     735 ms   1,7e-12            ~1e-10 graus
#     1e-9      298 ms   5,7e-9             ~4e-7 graus
#     1e-7      168 ms   3,3e-6             ~2e-4 graus
#     1e-5      100 ms   5,0e-4             ~0,03 graus
#
# El soroll de seguiment dels videos es de 0,3 graus, o sigui que la rapida
# deixa l'error numeric 1500 vegades per sota del que la camera pot veure.
#
# AIXO NO VOL DIR QUE LES DUES DONIN LA MATEIXA TRAJECTORIA. En un sistema
# caotic, que l'energia es conservi be no garanteix res sobre el cami: les
# mateixes condicions inicials integrades amb tolerancies diferents se
# separen al cap d'uns segons, i aixo es fisica, no un error. Per aixo
# L'EXPORTACIO VA SEMPRE PER L'ESTRICTA i tot el que es dibuixi ha de dir
# amb quina tolerancia s'ha calculat.
TOLERANCIES = {
    "estricta": (RTOL, ATOL),
    "rapida": (1e-7, 1e-9),
}


def simula(
    p: Parametres,
    estat_inicial,
    temps: float | None = None,
    fps: float = 200.0,
    *,
    t_eval=None,
    rtol: float = RTOL,
    atol: float = ATOL,
    dis: Dissipacio | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integra el sistema i retorna (t, estats) mostrejats a `fps` Hz.

    estats te forma (4, N): files theta1, theta2, omega1, omega2.

    Amb `t_eval` la sortida surt EXACTAMENT en aquells instants en comptes de
    sobre una graella regular, i llavors `temps` i `fps` no es miren. Serveix
    per donar els mateixos instants que un CSV del Tracker i poder comparar
    punt a punt sense interpolar res.

    La precisio de la integracio NO depen de la graella de sortida: solve_ivp
    tria els seus propis passos i despres interpola el polinomi del metode
    (DOP853 en te un de grau 7) alli on li demanis. Demanar-li menys punts no
    fa la resposta mes dolenta, ni demanar-n'hi mes la fa millor.

    Amb `dis`, el pendol perd energia pels rodaments i per l'aire. Sense, es
    conservatiu i la crida a l'integrador es LA MATEIXA que abans del pas 3.
    """
    if t_eval is None:
        if temps is None:
            raise ValueError("digues o `temps` o `t_eval`")
        t_eval = np.arange(0.0, temps + 0.5 / fps, 1.0 / fps)
    else:
        t_eval = np.asarray(t_eval, dtype=float)
        if t_eval.ndim != 1 or len(t_eval) < 2:
            raise ValueError("t_eval ha de ser una llista d'instants")
        if t_eval[0] < 0.0:
            raise ValueError("t_eval no pot comencar abans de 0: l'estat inicial hi es")
        if np.any(np.diff(t_eval) <= 0):
            raise ValueError("t_eval ha d'anar de menys a mes")

    sol = solve_ivp(
        derivades,
        (0.0, float(t_eval[-1])),
        np.asarray(estat_inicial, dtype=float),
        args=(p,) if dis is None else (p, dis),
        method=METODE,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"La integracio ha fallat: {sol.message}")
    return sol.t, sol.y


def instants_de_la_presa(taula, amollada_s: float, temps_max: float | None = None):
    """Els instants d'una presa del Tracker, comptats des de l'amollada.

    Retorna (t_eval, fila_inicial): els instants que s'han de donar a simula()
    i des de quina fila de la presa comencen. Per construccio

        t_eval[i] == taula["t"][i + fila_inicial] - amollada_s

    EXACTAMENT, sense cap interpolacio. Es la peca que permet comparar una
    simulacio amb una presa punt a punt: el que es mou es la simulacio, que es
    pot calcular on es vulgui, i les dades mesurades no es toquen mai.

    UN MATIS, per no dir mes del que hi ha: aquesta igualtat es exacta EN
    MEMORIA, i solve_ivp torna els instants que se li demanen sense tocar-los.
    Quan la simulacio es desa, pendol/dades.py escriu el CSV amb deu xifres
    significatives i alla s'hi perd l'ultim bit: rellegint el fitxer, els
    instants coincideixen amb els de la presa a menys d'un nanosegon. Es
    setze ordres de magnitud per sota d'un fotograma i no toca cap resultat,
    pero «identics» i «identics a un nanosegon» no son la mateixa frase.
    """
    t = np.asarray(taula["t"], dtype=float)
    fila = int(np.searchsorted(t, amollada_s, side="left"))
    if fila >= len(t) - 1:
        raise SystemExit(
            f"l'amollada cau a {amollada_s:.3f} s i la presa acaba a {t[-1]:.3f} s: "
            "no queda res per simular"
        )
    t_eval = t[fila:] - amollada_s
    if temps_max is not None:
        t_eval = t_eval[t_eval <= temps_max]
        if len(t_eval) < 2:
            raise SystemExit(f"amb --temps {temps_max} no queden prou instants")
    return t_eval, fila


# L'AFINAMENT DE L'AMOLLADA
#
# pendol/amollada.py dona l'instant d'amollada nomes amb el video: uns 5 ms
# d'error a les preses netes i uns 10 a les de 150 graus. Per a la finestra
# visual i per a lambda entre preses reals n'hi ha prou. Per comparar una
# simulacio amb una presa PUNT A PUNT, no: a 5-10 rad/s, 5 ms son 1,5-3 graus,
# de la mida del soroll de seguiment, i semblarien divergencia sense ser-ho.
#
# L'afinament encaixa la simulacio amb la presa durant els primers dos decims
# i mig de moviment, amb comparar.troba_desfasament, que recupera un retard
# conegut amb mig mil.lisegon.
#
# EL PREU, que s'ha de dir: l'amollada afinada JA NO ES INDEPENDENT DEL MODEL.
# Es «l'instant que fa que aquest model i aquest video coincideixin al
# principi». Si el model te un error de periode -i en te un: les inercies
# donen el periode mesurat al -0,89 %-, aquell error hi queda absorbit. Per
# aixo els dos numeros es desen per separat, i l'afinat nomes es fa servir
# per a la comparacio simulacio-realitat. I per aixo la finestra es curta: el
# desacord de periode creix amb el temps, i en 0,25 s el que pot absorbir es
# de l'ordre del mil.lisegon.
#
# UNA CONSEQUENCIA per a qui llegeixi les figures: en els primers 0,25 s la
# simulacio i la presa coincideixen PERQUE S'HAN FET COINCIDIR. La divergencia
# que vingui despres es la que compta; la d'aquell tros no vol dir res.
FINESTRA_AFINAMENT_S = 0.25   # quant de moviment es fa coincidir
PAS_AFINAMENT_S = 0.0002      # pas de la cerca: una vintena part de fotograma
CERCA_MAXIMA_S = 0.10         # mai no es busca mes lluny d'aixo


def afina_amollada(taula, r: dict, p: Parametres, estat0,
                   finestra_s: float = FINESTRA_AFINAMENT_S,
                   dis: Dissipacio | None = None) -> dict:
    """Afina l'amollada del video encaixant-hi la simulacio.

    `r` es el resultat de amollada.troba_de_taula i `estat0` les condicions
    inicials amb que es simulara. Retorna un diccionari que es pot desar tal
    qual a les metadades. Mai aixeca: si l'afinament no surt, ho diu a «ok» i
    qui crida es queda amb l'amollada del video.

    `dis` ha de ser la mateixa dissipacio amb que despres s'exportara: el
    model que afina i el que es compara han de ser el mateix.
    """
    from pendol.comparar import troba_desfasament

    t = np.asarray(taula["t"], dtype=float)
    angles = {"theta1": np.asarray(taula["theta1"], dtype=float),
              "theta2": np.asarray(taula["theta2"], dtype=float)}
    ta, sigma, fps = r["amollada_s"], r["incertesa_s"], r["fps"]

    # Es busca a quatre sigmes de l'amollada del video, i mai a menys de tres
    # fotogrames ni a mes de CERCA_MAXIMA_S. Si el millor encaix cau al limit
    # de la cerca, no es cap resultat: es que el minim era mes enlla, i
    # llavors no es fa servir.
    cerca = float(min(max(4 * sigma, 3.0 / fps), CERCA_MAXIMA_S))

    # La presa es compara a partir de mitja finestra de filtre despres de
    # l'amollada. Abans, el filtre de la conversio ha arrodonit el colze i la
    # presa no es la trajectoria de debo, mentre que la simulacio si.
    marge = 0.5 * r["finestra_filtre_s"] + 1.0 / fps
    tau = t - ta
    dins = (tau >= marge) & (tau <= marge + finestra_s)
    if dins.sum() < 20:
        return {"ok": False, "motiu": f"nomes {int(dins.sum())} mostres per afinar"}

    # La simulacio, en una graella densa que cobreix tot el que la cerca pot
    # demanar. A 0,25 ms de pas, interpolar-la linealment hi erra menys d'una
    # milionesima de radian.
    t_sim = np.arange(0.0, marge + finestra_s + cerca + 0.01, 2.5e-4)
    _, y = simula(p, estat0, t_eval=t_sim, dis=dis)

    per_angle = {}
    for i, (nom, th) in enumerate(angles.items()):
        s, err = troba_desfasament(tau[dins], th[dins], t_sim, y[i],
                                   finestra=finestra_s + 1.0, maxim=cerca,
                                   pas=PAS_AFINAMENT_S)
        per_angle[nom] = {
            "desplacament_s": float(s),
            "amollada_s": float(ta - s),
            "rms_graus": float(np.degrees(err)),
            "al_limit": bool(abs(s) >= cerca - 1.5 * PAS_AFINAMENT_S),
        }

    # Mana theta1, pel mateix motiu que a amollada.py: amollat amb els bracos
    # rectes, es el brac que es mou. theta2 es calcula igualment i es publica
    # com a comprovacio creuada.
    a = per_angle["theta1"]
    sortida = {
        "ok": bool(not a["al_limit"] and np.isfinite(a["rms_graus"])),
        "amollada_del_video_s": float(ta),
        "amollada_afinada_s": a["amollada_s"],
        "desplacament_s": a["desplacament_s"],
        "rms_de_l_encaix_graus": a["rms_graus"],
        "discrepancia_theta2_s": float(abs(per_angle["theta2"]["amollada_s"] - a["amollada_s"])),
        "finestra_s": [float(marge), float(marge + finestra_s)],
        "cerca_s": cerca,
        "per_angle": per_angle,
        "depen_del_model": True,
    }
    if a["al_limit"]:
        sortida["motiu"] = (f"el millor encaix cau al limit de la cerca ({1000*cerca:.0f} ms): "
                            "el minim de debo es mes enlla i no es fa servir")
    return sortida


def _desa_tir(cami, t, y, p, meta_extra, tolerancia="estricta", dis=None):
    rtol, atol = TOLERANCIES[tolerancia]
    meta = {
        "origen": "simulacio",
        "parametres": p.com_a_dict(),
        # Sempre present, tambe quan es None: un tir sense friccio ho ha de
        # dir, i no deixar-ho endevinar per l'absencia d'un camp.
        "dissipacio": None if dis is None else {
            **dis.com_a_dict(),
            "_estat": "coeficients NO decidits: la C i la eps s'han donat a ma",
        },
        # La tolerancia hi va amb nom i amb numeros: dues trajectories
        # calculades amb tolerancies diferents no son la mateixa trajectoria, i
        # d'aqui a uns mesos aquest camp sera l'unica manera de saber-ho.
        "integrador": {"metode": METODE, "rtol": rtol, "atol": atol,
                       "tolerancia": tolerancia},
        "estat_inicial_rad": [float(v) for v in y[:, 0]],
        "energia_inicial_J": float(energia(y[:, 0], p)),
        "energia_final_J": float(energia(y[:, -1], p)),
        **meta_extra,
    }
    dades.desa(cami, t, y[0], y[1], y[2], y[3], meta)


def _membre_del_feix(feina) -> int:
    """Simula i desa un membre del feix. Es una funcio de modul perque la
    pugui cridar un proces fill (--processos)."""
    (i, estat_i, p, temps, fps, t_eval, rtol, atol, dis, cami, meta, tolerancia) = feina
    t, y = simula(p, estat_i, temps, fps, t_eval=t_eval, rtol=rtol, atol=atol, dis=dis)
    _desa_tir(Path(cami), t, y, p, meta, tolerancia, dis=dis)
    return i


def autoprova() -> int:
    """Les dues coses noves del simulador: el rellotge de fora i les tolerancies.

    La primera es una identitat i es comprova exactament, sense marge: si
    demanes uns instants, han de sortir aquells i no uns de semblants. La
    segona es una mesura, i el que es comprova es que cada cami estigui a
    l'ordre de magnitud que se li ha promes a qui el faci servir.
    """
    import time

    problemes = []
    p = Parametres.des_de_json("parametres.json")

    print("Autoprova del simulador\n")
    print("1. El rellotge de fora (t_eval)")

    # Una graella lletja a posta: irregular, que no comenca a zero i amb els
    # instants d'un video de 240 fps, que no son numeros rodons.
    rng = np.random.default_rng(11)
    t_eval = np.sort(rng.uniform(0.3, 6.0, 500))
    t_eval = np.unique(np.concatenate([t_eval, np.arange(0, 1200) / 239.9002309468676]))
    t, y = simula(p, np.radians([150.0, 150.0, 0.0, 0.0]), t_eval=t_eval)
    exacte = np.array_equal(t, t_eval)
    if not exacte:
        problemes.append("els instants de sortida no son els que s'han demanat")
    print(f"   {'ok ' if exacte else 'NO '} {len(t_eval)} instants irregulars tornen "
          f"identics bit a bit")

    # I la trajectoria no pot dependre de la graella: els punts que son a les
    # dues graelles han de donar el mateix estat.
    t2, y2 = simula(p, np.radians([150.0, 150.0, 0.0, 0.0]), t_eval=t_eval[::7])
    dif = float(np.abs(y[:, ::7] - y2).max())
    ok = dif < 1e-9
    if not ok:
        problemes.append(f"demanar menys punts canvia la trajectoria en {dif:.2e}")
    print(f"   {'ok ' if ok else 'NO '} demanar-ne un de cada set dona el mateix "
          f"(difereixen {dif:.1e})")

    # I quant se'n perd en desar-ho. El CSV canonic s'escriu amb deu xifres
    # significatives, o sigui que la igualtat exacta no sobreviu al fitxer.
    # Aixo no es cap problema -parlem de nanosegons contra fotogrames de 4 ms-
    # pero s'ha de mesurar per poder-ho dir amb un numero.
    from pendol import dades

    cami = Path("esborranys/_autoprova_simulador/rellotge.csv")
    _desa_tir(cami, t, y, p, {"tipus": "autoprova"})
    tornat, _ = dades.carrega(cami)
    perdut = float(np.abs(tornat["t"].to_numpy() - t_eval).max())
    ok = perdut < 1e-6
    if not ok:
        problemes.append(f"desar i rellegir mou els instants {perdut:.1e} s")
    print(f"   {'ok ' if ok else 'NO '} desar-ho i rellegir-ho els mou "
          f"{perdut:.1e} s (un fotograma son 4.2e-03)")

    print("\n2. Els dos camins de calcul")
    estat0 = np.radians([150.0, 150.0, 0.0, 0.0])
    sostres = {"estricta": 1e-9, "rapida": 1e-4}
    print(f"   {'cami':>10} {'rtol':>8} {'costa':>9} {'deriva E':>11} {'sostre':>9}")
    for nom, (rtol, atol) in TOLERANCIES.items():
        ara = time.perf_counter()
        t, y = simula(p, estat0, 15.0, 200.0, rtol=rtol, atol=atol)
        triga = time.perf_counter() - ara
        deriva = abs(energia(y[:, -1], p) - energia(y[:, 0], p))
        ok = deriva < sostres[nom]
        if not ok:
            problemes.append(f"la tolerancia {nom} deriva {deriva:.2e} J, per sobre "
                             f"del sostre {sostres[nom]:.0e}")
        print(f"   {nom:>10} {rtol:>8.0e} {1000*triga:>7.0f} ms {deriva:>11.2e} "
              f"{sostres[nom]:>9.0e} {'ok' if ok else '<<< FORA'}")

    # 3. L'afinament de l'amollada, sobre tirs fabricats que passen pel
    # circuit sencer -posicions amb soroll, lectura del format lleig i
    # Savitzky-Golay-, amb els dos nivells de soroll de les preses de debo.
    # Aqui el criteri SI que torna a ser el d'un fotograma, que era el del pla.
    import contextlib
    import io

    from pendol import amollada, tracker

    print("\n3. L'afinament de l'amollada (el criteri: un fotograma de mediana)")
    FPS, RETARD = 240.0, 0.4137
    desti = Path("esborranys/_autoprova_simulador")
    print(f"   {'soroll':>7} {'angle':>6} {'del video':>11} {'afinada':>10} {'encaix':>9}")
    err_video, err_afinada = [], []
    k = 0
    for quin, soroll_m in (("1 px", 0.0005), ("4 px", 0.0020)):
        for angle in (45.0, 90.0, 150.0):
            k += 1
            cru, net = desti / f"cru_{k}.csv", desti / f"net_{k}.csv"
            rig = amollada._presa_falsa(p, cru, RETARD, 3.0, FPS, angle, soroll_m, llavor=500 + k)
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False,
                                 meta_extra={"geometria": {"rigidesa": rig}})
            taula, meta = dades.carrega(net)
            r = amollada.troba_de_taula(taula, meta)
            if not r["hi_ha_amollada"]:
                problemes.append(f"{quin} a {angle:.0f} graus: sense amollada ({r['motiu']})")
                continue
            af = afina_amollada(taula, r, p, amollada.estat_inicial(r))
            if not af["ok"]:
                problemes.append(f"{quin} a {angle:.0f} graus: l'afinament no surt ({af['motiu']})")
                continue
            ev, ea = abs(r["amollada_s"] - RETARD), abs(af["amollada_afinada_s"] - RETARD)
            err_video.append(ev)
            err_afinada.append(ea)
            print(f"   {quin:>7} {angle:>6.0f} {1000*ev:>8.2f} ms {1000*ea:>7.2f} ms "
                  f"{af['rms_de_l_encaix_graus']:>7.2f} gr")
    if err_afinada:
        med_v, med_a = float(np.median(err_video)), float(np.median(err_afinada))
        print(f"   mediana: {1000*med_v:.2f} ms del video, {1000*med_a:.2f} ms afinada "
              f"(un fotograma son {1000/FPS:.2f})")
        if med_a > 1.0 / FPS:
            problemes.append(f"l'afinament deixa {1000*med_a:.2f} ms de mediana, mes d'un fotograma")
        if med_a >= med_v:
            problemes.append("l'afinament no millora l'amollada del video")

    # 3b. EL PREU DE DEPENDRE DEL MODEL. Els tirs de dalt estan fabricats amb
    # el mateix model que despres els afina, o sigui que alli el model es
    # perfecte i l'afinament no pot sortir esbiaixat. A la vida real no ho es:
    # les inercies del model donen el periode del banc al -0,89 %. Aqui es
    # fabriquen tirs amb un pendol que discrepa del model EXACTAMENT aixi -una
    # g un 0,89 % mes petita fa el mateix al periode que una inercia un 0,89 %
    # mes gran- i s'afinen amb el model de sempre. L'error que en surt es el
    # biaix que el model hi posa. No es un criteri: es una mesura, i el numero
    # va a la memoria al costat de l'afinament.
    from dataclasses import replace

    # Dos desacords, perque son de dues menes. El del cos compost (-0,89 %)
    # fa anar tot el pendol una mica mes a poc a poc, i una g un 0,89 % mes
    # petita fa exactament aixo. El de la barra 2 sola (-2,98 %) es mes gros i
    # no es uniforme: nomes la barra 2 es mes inercial que al model, i aixo
    # canvia la proporcio entre els dos modes. S'hi arriba allargant R2 un
    # 2,9 %, que puja I2 un 2,98 % sense tocar res mes de la dinamica.
    fabricats = {
        "tot el pendol -0,89 %": replace(p, g=p.g * (1 - 0.0089)),
        "barra 2 sola -2,98 %": replace(p, R2=p.R2 * np.sqrt(
            1 + 0.0298 * p.I2 / ((1 / 12) * p.m2 * p.R2**2))),
    }
    print("\n3b. El que hi posa el model: tirs d'un pendol que discrepa del model")
    print("    com el banc en discrepa, afinats amb el model")
    biaixos = []
    for etiqueta, p_banc in fabricats.items():
        for angle in (45.0, 90.0, 150.0):
            k += 1
            cru, net = desti / f"cru_{k}.csv", desti / f"net_{k}.csv"
            rig = amollada._presa_falsa(p_banc, cru, RETARD, 3.0, FPS, angle, 0.0005,
                                        llavor=500 + k)
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False,
                                 meta_extra={"geometria": {"rigidesa": rig}})
            taula, meta = dades.carrega(net)
            r = amollada.troba_de_taula(taula, meta)
            if not r["hi_ha_amollada"]:
                continue
            af = afina_amollada(taula, r, p, amollada.estat_inicial(r))
            if af["ok"]:
                biaix = af["amollada_afinada_s"] - RETARD
                biaixos.append(abs(biaix))
                print(f"   {etiqueta:<22} {angle:>4.0f} graus: afinada {1000*biaix:+.2f} ms, "
                      f"encaix {af['rms_de_l_encaix_graus']:.2f} graus")
    if biaixos:
        print(f"   el model hi posa com a molt {1000*max(biaixos):.2f} ms "
              f"({max(biaixos)*FPS:.2f} fotogrames)")

    # 3c (29/09/2026). EL PREU DE LA FRICCIO QUE EL MODEL NO TE. Tirs fabricats
    # AMB friccio i afinats amb el model SENSE, que es el que fa --com-el-tracker
    # mentre els coeficients no estiguin decidits. Coeficients de PROVA: els
    # rodaments de parametres.json amb C = 0,00794 i eps = 1e-3, i el mateix
    # sense viscos. Tampoc no es un criteri: es una mesura, com la 3b.
    from dataclasses import replace as _replace

    joc = Dissipacio.des_de_friccio(Friccio.des_de_json("parametres.json"), 0.00794, 1e-3)
    jocs = {"rodaments + aire": joc, "sense viscos": _replace(joc, b1=0.0, b2=0.0)}
    print("\n3c. El que hi posa la friccio: tirs AMB friccio de prova, afinats SENSE")
    biaixos_f = []
    for etiqueta, dis_f in jocs.items():
        for angle in (45.0, 90.0, 150.0):
            k += 1
            cru, net = desti / f"cru_{k}.csv", desti / f"net_{k}.csv"
            rig = amollada._presa_falsa(p, cru, RETARD, 3.0, FPS, angle, 0.0005,
                                        llavor=500 + k, dis=dis_f)
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.processa(cru, net, pivot=(0.0, 0.0), y_avall=False,
                                 meta_extra={"geometria": {"rigidesa": rig}})
            taula, meta = dades.carrega(net)
            r = amollada.troba_de_taula(taula, meta)
            if not r["hi_ha_amollada"]:
                print(f"   {etiqueta:<18} {angle:>4.0f} graus: sense amollada ({r['motiu']})")
                continue
            af = afina_amollada(taula, r, p, amollada.estat_inicial(r))
            if af["ok"]:
                biaix = af["amollada_afinada_s"] - RETARD
                biaixos_f.append(abs(biaix))
                print(f"   {etiqueta:<18} {angle:>4.0f} graus: afinada {1000*biaix:+.2f} ms, "
                      f"del video {1000*(r['amollada_s'] - RETARD):+.2f} ms, "
                      f"encaix {af['rms_de_l_encaix_graus']:.2f} graus")
            else:
                print(f"   {etiqueta:<18} {angle:>4.0f} graus: no s'afina ({af['motiu']})")
    if biaixos_f:
        print(f"   la friccio que el model no te hi posa com a molt {1000*max(biaixos_f):.2f} ms "
              f"({max(biaixos_f)*FPS:.2f} fotogrames)")

    # 4. LA FRICCIO (pas 3). Dues propietats que no depenen de quins
    # coeficients es decideixin: amb tots a zero ha de ser el pendol
    # conservatiu, i amb friccio l'energia no pot pujar mai. La comptabilitat
    # exacta -que baixa el que es dissipa- es a eines/comprova_energia.py.
    print("\n4. La friccio (coeficients de prova: no n'hi ha cap de decidit)")
    estat0 = np.radians([150.0, 150.0, 0.0, 0.0])
    _, y_sense = simula(p, estat0, 5.0, 200.0)
    zero = Dissipacio(tau_f1=0.0, b1=0.0, tau_f2=0.0, b2=0.0, C=0.0, eps=1e-3)
    _, y_zero = simula(p, estat0, 5.0, 200.0, dis=zero)
    dif = float(np.abs(y_zero - y_sense).max())
    ok = dif == 0.0
    if not ok:
        problemes.append(f"amb tots els coeficients a zero difereix {dif:.1e} de sense friccio")
    print(f"   {'ok ' if ok else 'NO '} amb tots els coeficients a zero dona el mateix que "
          f"sense friccio (difereix {dif:.1e})")

    dis = Dissipacio.des_de_friccio(Friccio.des_de_json("parametres.json"), 0.00794, 1e-3)
    _, y = simula(p, estat0, 10.0, 200.0, dis=dis)
    E = energia(y, p)
    pujada = float(np.max(np.diff(E)))
    ok = pujada < 1e-9
    if not ok:
        problemes.append(f"amb friccio l'energia puja {pujada:.1e} J d'una mostra a la seguent")
    print(f"   {'ok ' if ok else 'NO '} amb friccio l'energia no puja mai (la pujada mes "
          f"gran entre mostres, {pujada:.1e} J; ha baixat {1000 * (E[0] - E[-1]):.1f} mJ "
          f"en 10 s)")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> la graella de sortida no toca la fisica, els dos camins compleixen i")
    print("     l'afinament deixa l'amollada per sota d'un fotograma.")
    print("     Limits declarats:")
    print("     - que l'energia es conservi NO vol dir que les dues tolerancies donin")
    print("       la mateixa trajectoria. En un sistema caotic se separen, i per aixo")
    print("       l'exportacio va sempre per l'estricta.")
    print("     - l'afinament s'ha provat amb els desacords d'inercia que es coneixen,")
    print("       pero NO amb tirs fabricats amb friccio i afinats sense (o al reves).")
    print("       En 0,25 s el pendol real gairebe no perd energia, pero aixo aquesta")
    print("       prova no ho mesura.")
    print("     - la friccio te el codi i les proves, pero NO te coeficients decidits:")
    print("       la C i la eps es donen a ma i cap tir amb friccio val per a la memoria.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parametres", default="parametres.json", help="JSON amb els parametres fisics")
    ap.add_argument("--theta1", type=float, help="angle inicial 1 (GRAUS)")
    ap.add_argument("--theta2", type=float, default=0.0, help="angle inicial 2 (GRAUS)")
    ap.add_argument("--omega1", type=float, default=0.0, help="velocitat angular inicial 1 (GRAUS/s)")
    ap.add_argument("--omega2", type=float, default=0.0, help="velocitat angular inicial 2 (GRAUS/s)")
    ap.add_argument("--temps", type=float, default=None,
                    help="durada simulada (s). Per defecte 20; amb --com-el-tracker, "
                         "tota la presa si no es diu res")
    ap.add_argument("--fps", type=float, default=200.0, help="mostres per segon a la sortida")
    ap.add_argument("--sortida", help="fitxer .csv, o directori si uses --conjunt")
    ap.add_argument("--conjunt", type=int, default=0, help="nombre de tirs del feix (0 = un sol tir)")
    ap.add_argument(
        "--incertesa",
        type=float,
        default=1.0,
        help="desviacio tipus dels angles inicials del feix (GRAUS). "
        "Posa-hi la teva incertesa real de mesura de l'angle de sortida.",
    )
    ap.add_argument("--llavor", type=int, default=0, help="llavor aleatoria (reproductibilitat)")
    ap.add_argument("--processos", type=int, default=1,
                    help="amb --conjunt, quants processos simulen alhora (per defecte 1). "
                         "El feix surt identic byte a byte amb qualsevol nombre: els estats "
                         "inicials es generen tots abans de repartir-los")
    ap.add_argument("--com-el-tracker", metavar="CSV",
                    help="dona exactament els mateixos instants que aquesta presa, "
                         "comptats des de la seva amollada, i pren-ne les condicions "
                         "inicials mesurades si no en dius cap")
    ap.add_argument("--tolerancia", choices=sorted(TOLERANCIES), default="estricta",
                    help="estricta per exportar (per defecte), rapida per mirar")
    ap.add_argument("--sense-afinar", action="store_true",
                    help="amb --com-el-tracker, fes servir l'amollada del video tal com "
                         "surt, sense afinar-la encaixant-hi la simulacio. Es el zero "
                         "independent del model, i uns 5-10 ms menys precis")
    ap.add_argument("--friccio", action="store_true",
                    help="afegeix els cinc termes dissipatius: els rodaments de "
                         "parametres.json, i la C i la eps que es donin a ma")
    ap.add_argument("--aire-C", type=float, metavar="C",
                    help="amb --friccio, OBLIGATORI: el coeficient de l'aire per unitat "
                         "de llargada (kg/m). No hi ha cap valor per defecte perque "
                         "encara no esta decidit")
    ap.add_argument("--epsilon", type=float, metavar="EPS",
                    help="amb --friccio, OBLIGATORI: la eps de tanh(w/eps) del sec (rad/s)")
    ap.add_argument("--autoprova", action="store_true")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not args.sortida:
        ap.error("falta --sortida")

    p = Parametres.des_de_json(args.parametres)
    rtol, atol = TOLERANCIES[args.tolerancia]

    dis = None
    if args.friccio:
        if args.aire_C is None or args.epsilon is None:
            ap.error("amb --friccio cal dir --aire-C i --epsilon: no estan decidides i "
                     "no hi ha cap valor per defecte")
        dis = Dissipacio.des_de_friccio(Friccio.des_de_json(args.parametres),
                                        args.aire_C, args.epsilon)
        print(f"AMB FRICCIO: tau_f1 {dis.tau_f1:.3e}, b1 {dis.b1:.3e}, tau_f2 "
              f"{dis.tau_f2:.3e}, b2 {dis.b2:.3e}, C {dis.C:.5f}, eps {dis.eps:.0e}")
        print("  Coeficients NO decidits: el resultat no val per a la memoria.")
    elif args.aire_C is not None or args.epsilon is not None:
        ap.error("--aire-C i --epsilon nomes tenen sentit amb --friccio")

    # --- El rellotge: propi o el d'una presa del Tracker -----------------
    t_eval, estat_mesurat, meta_rellotge = None, None, {}
    if args.com_el_tracker:
        from pendol import amollada

        taula, meta_presa = dades.carrega(args.com_el_tracker)
        r = amollada.troba_de_taula(taula, meta_presa)
        if not r["hi_ha_amollada"]:
            raise SystemExit(
                f"{args.com_el_tracker}: no s'hi pot llegir l'amollada i per tant no hi ha "
                f"cap zero comu.\n  Motiu: {r['motiu']}\n"
                "  Aquesta presa no es pot comparar amb una simulacio que surt del repos."
            )
        estat_mesurat = amollada.estat_inicial(r)
        print(f"Amollada del video a {args.com_el_tracker}: {r['amollada_s']:.4f} s "
              f"(+-{1000*r['incertesa_s']:.1f} ms)")
        for a in r["avisos"]:
            print(f"  ATENCIO: {a}")

    # Els graus nomes existeixen a la interficie. A partir d'aqui, radians.
    if args.theta1 is not None:
        estat0 = np.radians([args.theta1, args.theta2, args.omega1, args.omega2])
        nominal = [args.theta1, args.theta2]
    elif estat_mesurat is not None:
        estat0 = estat_mesurat
        nominal = [float(np.degrees(estat0[0])), float(np.degrees(estat0[1]))]
        print(f"Condicions inicials mesurades de la presa: "
              f"{nominal[0]:.3f} / {nominal[1]:.3f} graus, amb omega = 0 exacte")
    else:
        ap.error("falta --theta1 (o --com-el-tracker, que el treu de la presa)")

    # --- El zero del rellotge, afinat o no ------------------------------
    # Va despres de les condicions inicials perque l'afinament simula, i ho
    # ha de fer amb les mateixes condicions que despres s'exportaran.
    if args.com_el_tracker:
        zero, origen, afinament = r["amollada_s"], "video", None
        if not args.sense_afinar:
            afinament = afina_amollada(taula, r, p, estat0, dis=dis)
            if afinament["ok"]:
                zero, origen = afinament["amollada_afinada_s"], "afinada amb el model"
                print(f"Afinada encaixant-hi la simulacio: {zero:.4f} s "
                      f"({1000*afinament['desplacament_s']:+.1f} ms, encaix de "
                      f"{afinament['rms_de_l_encaix_graus']:.2f} graus rms; "
                      f"theta2 hi discrepa {1000*afinament['discrepancia_theta2_s']:.1f} ms)")
                if abs(afinament["desplacament_s"]) > 3 * r["incertesa_s"]:
                    # Pot fallar qualsevol dels dos, i l'autoprova ho ha vist
                    # passar pel costat del video: en un tir fabricat amb
                    # fotogrames perduts, la incertesa del video es quedava
                    # curta i l'afinament encertava a 0,2 ms.
                    print(f"  ATENCIO: l'afinament mou l'amollada mes de tres vegades la "
                          f"incertesa del video. O aquella incertesa es queda curta, o el "
                          f"model hi posa mes del compte; mira l'encaix abans de fiar-t'hi.")
            else:
                print(f"  No s'ha pogut afinar ({afinament['motiu']}); "
                      f"es fa servir l'amollada del video.")
        t_eval, fila = instants_de_la_presa(taula, zero, args.temps)
        meta_rellotge = {
            "rellotge": {
                "presa": str(args.com_el_tracker),
                # La que s'ha fet servir, i d'on surt. Amb aquests tres numeros
                # qualsevol pot refer la correspondencia:
                #   t_sim[i] = t_presa[i + fila] - amollada_s
                "amollada_s": float(zero),
                "origen_de_l_amollada": origen,
                "fila_inicial_de_la_presa": fila,
                "instants": "els de la presa, sense interpolar (a 1 ns pel CSV)",
                # I les dues per separat, perque NO son el mateix numero: la
                # del video no depen de cap model i l'afinada si.
                "amollada_del_video_s": r["amollada_s"],
                "incertesa_amollada_del_video_s": r["incertesa_s"],
                "afinament": afinament,
            }
        }
        print(f"Rellotge: {len(t_eval)} instants de la presa des de la fila {fila}")

    temps = 20.0 if args.temps is None else args.temps

    if args.conjunt <= 0:
        t, y = simula(p, estat0, temps, args.fps, t_eval=t_eval, rtol=rtol, atol=atol,
                      dis=dis)
        _desa_tir(args.sortida, t, y, p, {"tipus": "tir_unic", **meta_rellotge},
                  args.tolerancia, dis=dis)
        variacio = energia(y[:, -1], p) - energia(y[:, 0], p)
        print(f"Desat: {args.sortida}  ({len(t)} mostres, {t[-1]:.3f} s, "
              f"tolerancia {args.tolerancia})")
        if dis is None:
            print(f"Sense friccio -> variacio d'energia: {abs(variacio):.3e} J "
                  f"(hauria de ser ~0)")
        else:
            print(f"Amb friccio -> la friccio s'ha emportat {-1000 * variacio:.3f} mJ")
        return 0

    # --- Feix de trajectories -------------------------------------------
    directori = Path(args.sortida)
    directori.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.llavor)
    sigma = np.radians(args.incertesa)

    # Els estats inicials, TOTS primer i en el mateix ordre que quan es feien
    # dins del bucle: aixi el feix es el mateix, es faci amb un proces o amb sis.
    feines = []
    for i in range(args.conjunt):
        estat_i = estat0.copy()
        if i > 0:  # el tir 0 es sempre el nominal, sense pertorbar
            estat_i[0] += rng.normal(0.0, sigma)
            estat_i[1] += rng.normal(0.0, sigma)
        feines.append((i, estat_i, p, temps, args.fps, t_eval, rtol, atol, dis,
                       str(directori / f"tir{i:04d}.csv"),
                       {"tipus": "membre_de_feix", "index": i, "nominal_graus": nominal,
                        "incertesa_graus": args.incertesa, "llavor": args.llavor,
                        **meta_rellotge},
                       args.tolerancia))

    if args.processos <= 1:
        for feina in feines:
            _membre_del_feix(feina)
    else:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(args.processos) as pool:
            for _ in pool.imap_unordered(_membre_del_feix, feines):
                pass

    print(f"Feix de {args.conjunt} trajectories desat a {directori}/")
    print(f"Angles nominals: {nominal[0]:.3f} / {nominal[1]:.3f} graus, "
          f"sigma = {args.incertesa} graus, tolerancia {args.tolerancia}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
