"""Neteja les exportacions del Tracker i les passa al format canonic.

El Tracker no et dona angles: et dona POSICIONS (x, y) de cada massa,
fotograma a fotograma. Aquest modul fa tota la feina bruta:

  1. Llegeix el CSV encara que porti linies de metadades al davant,
     encara que el separador sigui tabulador o punt i coma, i encara que
     els decimals siguin amb COMA (exportacio en configuracio catalana).
  2. Passa (x, y) a angles amb atan2, respecte de la vertical inferior.
  3. DESEMBOLICA els angles (np.unwrap). Sense aixo, quan theta passa de
     179 a -179 graus la grafica fa un salt vertical fals i qualsevol
     calcul de divergencia o de Lyapunov surt disparat. Es l'error numero
     u en aquest tipus de treball.
  4. Reomple fotogrames perduts i remostreja a una base temporal uniforme.
  5. Deriva els angles amb un filtre de Savitzky-Golay per obtenir omega.
     Fer (theta[i+1]-theta[i])/dt directament sobre dades de video dona
     soroll pur: el video te un error de mig pixel per fotograma que, en
     dividir per dt = 1/240 s, s'amplifica per 240.
  6. Mesura l'AMOLLADA sobre el CSV ja desat (pendol/amollada.py) i la deixa
     a les metadades, al bloc «amollada». Es el zero del rellotge de la presa
     i, des del 24/09/2026, el que decideix si arrenca del repos.
  7. Mesura on es la marca de l'articulacio respecte del centre del cargol
     del pivot 2 i, si n'es fora mes d'1 mm, la hi torna abans de calcular
     theta2 (des del 25/09/2026). Vegeu mesura_marca_articulacio.

Menja dues coses:

  UN .trk (recomanat). Ha de portar tres pistes: "fix" al pivot de dalt,
  "rodament" a l'articulacio i "extrem" al gomet. D'aqui en surt tot -les
  posicions, el rellotge i la geometria- i no cal exportar res a ma:

    .venv\\Scripts\\python -m pendol.tracker dades/video/tracker/90/90_1.trk \\
        --sortida dades/net/90_1.csv --angle-nominal 90

  UN CSV exportat del Tracker, amb dues masses i el pivot com a constant. Es
  el cami vell i es queda igual:

    .venv\\Scripts\\python -m pendol.tracker dades/cru/tir01.csv \\
        --sortida dades/net/tir01.csv --pivot 0.031 0.402

La diferencia gran del .trk no es la comoditat, es LA VERTICAL. Als setze
.trk del 30/08/2026 l'angle de l'eix escrit al fitxer es +0,000 graus a
quinze, o sigui que ningu no havia corregit la inclinacio de la camera, que
varia entre preses (dispersio 0,47 graus, rang -0,37 a +1,14). El truc del
pendol magnetic -adrecar-lo amb la simetria de l'oscil.lacio- no serveix per
a un pendol caotic. Vegeu geometria_del_repos().
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from pendol import amollada, dades, trk

_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _es_num(tros: str) -> bool:
    return bool(_NUM.match(tros.strip().strip('"')))


def _detecta_format(linies: list[str]) -> tuple[str, bool]:
    """Endevina (separador, decimals_amb_coma) mirant les ultimes linies.

    L'ordre de prioritat NO es negociable: es queda amb el PRIMER candidat
    que dona un nombre constant de camps, no amb el que en dona mes. Si no,
    un fitxer amb separador ';' i decimals amb coma (exportacio catalana)
    es detecta com a separat per comes, perque cada numero n'aporta una.
    """
    cua = [ln for ln in linies[-20:] if ln.strip()]
    if not cua:
        raise ValueError("El fitxer no te cap linia amb contingut")

    for sep in ("\t", ";", ",", " "):
        comptes = [ln.count(sep) for ln in cua]
        if min(comptes) >= 1 and len(set(comptes)) == 1:
            # Decimals amb coma: nomes es pot donar si el separador NO es la coma.
            coma_decimal = sep != "," and any(re.search(r"\d,\d", ln) for ln in cua)
            return sep, coma_decimal

    raise ValueError(
        "No he sabut trobar el separador de columnes. Obre el CSV amb un editor "
        "de text i mira si les files tenen totes el mateix nombre de camps."
    )


def llegeix_cru(cami: str | Path) -> tuple[np.ndarray, list[str]]:
    """Llegeix un export del Tracker. Retorna (matriu de floats, capcalera)."""
    cami = Path(cami)
    for codificacio in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = cami.read_text(encoding=codificacio)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"No he pogut descodificar {cami}")

    linies = text.splitlines()
    sep, coma_decimal = _detecta_format(linies)

    def trosseja(ln: str) -> list[str]:
        return [t.strip().strip('"') for t in ln.split(sep)] if sep != " " else ln.split()

    # Primera linia on almenys dos camps son numeros: alli comencen les dades.
    inici = None
    for i, ln in enumerate(linies):
        trossos = trosseja(ln)
        if len(trossos) >= 2 and sum(_es_num(t) for t in trossos) >= 2:
            inici = i
            break
    if inici is None:
        raise ValueError(f"No he trobat cap fila de dades numeriques a {cami.name}")

    ncols = len(trosseja(linies[inici]))
    capcalera = []
    if inici > 0:
        anterior = trosseja(linies[inici - 1])
        if len(anterior) == ncols:
            capcalera = anterior
    if not capcalera:
        capcalera = [f"col{j}" for j in range(ncols)]

    files = []
    for ln in linies[inici:]:
        if not ln.strip():
            continue
        trossos = trosseja(ln)
        if len(trossos) != ncols:
            continue
        fila = []
        for t in trossos:
            t = t.replace(",", ".") if coma_decimal else t
            try:
                fila.append(float(t))
            except ValueError:
                fila.append(np.nan)  # fotograma en que el Tracker va perdre el punt
        files.append(fila)

    matriu = np.array(files, dtype=float)
    print(f"  separador: {sep!r}   decimals amb coma: {coma_decimal}   files: {len(matriu)}   columnes: {ncols}")
    print(f"  capcalera detectada: {capcalera}")
    return matriu, capcalera


def tria_columnes(matriu: np.ndarray, columnes: str | None) -> np.ndarray:
    """Selecciona (t, x1, y1, x2, y2) de la matriu crua."""
    if columnes:
        idx = [int(i) for i in columnes.split(",")]
        if len(idx) != 5:
            raise ValueError("--columnes vol 5 indexs: t,x1,y1,x2,y2")
        return matriu[:, idx]

    n = matriu.shape[1]
    if n == 5:  # t, x1, y1, x2, y2
        idx = [0, 1, 2, 3, 4]
    elif n == 6:  # t, x1, y1, t(repetit), x2, y2
        idx = [0, 1, 2, 4, 5]
    else:
        raise ValueError(
            f"He trobat {n} columnes i no se com interpretar-les.\n"
            "Digue-m'ho amb --columnes t,x1,y1,x2,y2 (indexs comencant a 0)."
        )
    print(f"  columnes usades (auto): t={idx[0]} x1={idx[1]} y1={idx[2]} x2={idx[3]} y2={idx[4]}")
    return matriu[:, idx]


def a_angles(t, x1, y1, x2, y2, pivot=(0.0, 0.0), y_avall=False, inverteix=False):
    """Passa posicions a angles desembolicats, amb el conveni de fisica.py.

    theta = atan2(dx, -dy) mesura l'angle des de la vertical INFERIOR,
    positiu en sentit antihorari, exactament com x = L*sin(theta),
    y = -L*cos(theta) a pendol/fisica.py.
    """
    xp, yp = pivot
    dx1, dy1 = x1 - xp, y1 - yp
    dx2, dy2 = x2 - x1, y2 - y1
    if y_avall:  # el video tenia l'eix y cap avall
        dy1, dy2 = -dy1, -dy2

    th1 = np.unwrap(np.arctan2(dx1, -dy1))
    th2 = np.unwrap(np.arctan2(dx2, -dy2))
    if inverteix:  # camera en mirall respecte del conveni
        th1, th2 = -th1, -th2
    return th1, th2


# ---------------------------------------------------------------------------
# La geometria del .trk: la vertical i la marca de l'articulacio
# ---------------------------------------------------------------------------
# Un pendol doble caotic no oscil.la simetric respecte de la vertical, o sigui
# que el truc del magnetic (adrecar-lo amb l'eix de simetria de l'arc) no hi
# val. L'unic plom que dona es la POSICIO FINAL: al repos el conjunt penja
# vertical, i llavors la recta pivot->extrem ES la vertical.
#
# Les preses es tallen a ull quan el moviment deixa de veure's, o sigui que cap
# no arriba al repos de debo. No importa: el PROMIG d'una oscil.lacio petita es
# l'equilibri encara que cap fotograma no hi estigui. Comprovat als 16 .trk del
# 30/08/2026: promitjant finestres d'1, 2, 5 i 10 s la inclinacio no es mou mes
# de 0,11 graus (mediana; pitjor 0,39).
#
# El que en surt NO es "la inclinacio de la camera" a seques. Si el gomet no
# esta clavat a l'eix de la barra, el seu desviament s'absorbeix aqui dins i no
# es pot separar de la camera amb aquestes dades. Aixo es correcte per al que
# volem -fa que el repos doni theta = 0- pero deixa un error residual quan el
# segon brac gira, de l'ordre de (desviament del gomet)/L2.

CUA_PER_DEFECTE = 2.0      # s de video que es promitgen per treure la vertical
LLINDAR_ESTABILITAT = 0.5  # graus entre les finestres d'1 i 2 s
LLINDAR_RESIDU = 50.0      # px de recorregut del gomet a l'ultim segon


def _mitjana_de_cua(punts, n):
    return punts[-n:].mean(axis=0) if n < len(punts) else punts.mean(axis=0)


def _inclinacio(P, A, G, n):
    """L'angle de pivot->extrem a la cua, mesurat des de la vertical d'avall.

    En pixels d'imatge, o sigui amb la y cap AVALL: penjar recte es (0, +).
    No depen d'on s'ha clicat l'articulacio, que es tot el motiu de fer-ho
    amb l'extrem i no amb el primer brac.
    """
    p, g = _mitjana_de_cua(P, n), _mitjana_de_cua(G, n)
    return math.atan2(g[0] - p[0], g[1] - p[1])


def geometria_del_repos(P, A, G, fps, cua_s=CUA_PER_DEFECTE) -> dict:
    """Vertical, desviament de l'articulacio i si la cua s'ha assossegat."""
    n = max(10, int(round(cua_s * fps)))
    gir = _inclinacio(P, A, G, n)

    # Estabilitat: la mateixa mesura amb la meitat de cua. Si no coincideix,
    # es que encara es movia i el promig no vol dir res.
    estabilitat = abs(gir - _inclinacio(P, A, G, max(10, n // 2)))

    # Quant es mou encara el gomet a l'ultim segon.
    m = max(10, int(round(fps)))
    ultim = G[-m:]
    residu = float(math.hypot(np.ptp(ultim[:, 0]), np.ptp(ultim[:, 1])))

    # El desviament de la marca de l'articulacio: la seva distancia
    # perpendicular a la recta pivot-extrem, un cop adrecada la vertical.
    c, s = math.cos(gir), math.sin(gir)
    p, a = _mitjana_de_cua(P, n), _mitjana_de_cua(A, n)
    u, v = a[0] - p[0], a[1] - p[1]
    delta = u * c - v * s          # component x un cop girat: la vertical ja es y

    return {
        "gir_graus": math.degrees(gir),
        "estabilitat_graus": math.degrees(estabilitat),
        "residu_px": residu,
        "delta_px": float(delta),
        "L1_px": float(np.hypot(u, v)),
        "L2_px": float(np.hypot(*(_mitjana_de_cua(G, n) - a))),
        "cua_s": float(cua_s),
    }


def adreca_i_corregeix(P, A, G, geo: dict, corregeix_articulacio=False):
    """Gira a la vertical i, si es demana, treu el desviament de l'articulacio.

    Torna (articulacio, extrem) en pixels, relatius al pivot i ja girats.

    ATENCIO AMB corregeix_articulacio: per defecte es FALS a posta, i no es
    una precaucio sino un resultat. La idea era que la marca del rodament esta
    clicada fora de l'eix una quantitat fixa, i que per tant nomes calia
    desplacar-la. Les dades del 30/08/2026 ho desmenteixen:

      - al repos el desviament aparent entre bracos es de -7,9 graus (150_3)
        i a la sortida, del mateix tir, es de +2,5. CANVIA DE SIGNE. Un
        desviament fix respecte de la barra no pot fer aixo: giraria amb la
        barra i es veuria igual a les dues posicions.
      - aplicant-la, la dispersio de l'angle de sortida dins d'un grup
        EMPITJORA: 1,31 -> 2,11 graus al 150 i 1,46 -> 2,16 al 90.
      - i la correlacio del desviament residual amb delta puja de +0,38 a
        +0,91. Si la correccio fos bona, hauria de baixar a zero.

    O sigui que hi ha alguna cosa mes que un desplacament de la marca, i
    corregir amb un model equivocat es pitjor que no corregir. Delta es mesura
    i es desa igualment, que es informacio bona; el que no es fa es fer-hi res.
    Per resoldre-ho cal mirar el video: on son exactament les tres marques
    respecte dels eixos de les barres i del gomet.

    Si es fa en positions i no en angles es perque, quan algun dia se sapiga
    quin es el model bo, la correccio de theta2 NO sera una constant: val
    delta/L2 amb els bracos alineats i zero quan fan 90 graus.
    """
    gir = math.radians(geo["gir_graus"])
    c, s = math.cos(gir), math.sin(gir)

    def gira(Q):
        u, v = Q[:, 0] - P[:, 0], Q[:, 1] - P[:, 1]
        return np.column_stack([u * c - v * s, u * s + v * c])

    A_g, G_g = gira(A), gira(G)
    if not corregeix_articulacio:
        return A_g, G_g

    # J = A - delta * n1, amb n1 perpendicular al primer brac. n1 depen de J,
    # o sigui que s'itera: amb delta/L1 de l'ordre de 4 graus, dues voltes ja
    # deixen l'error molt per sota del soroll de mesura.
    J = A_g.copy()
    for _ in range(3):
        L = np.hypot(J[:, 0], J[:, 1])
        L = np.where(L == 0.0, 1.0, L)
        ux, uy = J[:, 0] / L, J[:, 1] / L
        J = A_g - geo["delta_px"] * np.column_stack([uy, -ux])
    return J, G_g


# ---------------------------------------------------------------------------
# La marca de l'articulacio: es a l'eix del pivot 2?
# ---------------------------------------------------------------------------
# Trobat el 24/09/2026: a les preses velles de 45 i 90 graus la marca
# «rodament» queda uns 4,4 mm MES ENLLA de l'eix del pivot 2, al llarg de la
# barra 1. Aixo posa a theta2 un error que es nul amb els bracos alineats i
# maxim quan fan angle recte, de fins a 1,8 graus.
#
# NO ES LA CORRECCIO REFUTADA DE MES AMUNT (adreca_i_corregeix). Aquella movia
# la marca DE COSTAT, amb un valor mesurat en un sol repos. Aquesta mira el
# desplacament AL LLARG de la barra 1, i el mesura amb TOT el moviment: la
# barra 2 es rigida, o sigui que el gomet ha d'estar a distancia constant del
# pivot 2 de debo. Si la marca no hi es, la distancia marca -> gomet balla
# quan el pendol es plega:
#
#     |G - A|^2 = r^2 + delta^2 - 2*r*delta*cos(theta2 - theta1)
#
# ES MESURA SEMPRE, I ES CORREGEIX QUAN LA MARCA ES FORA DE L'EIX. El
# 25/09/2026 es va decidir primer tornar a marcar a ma la pista de
# l'articulacio (opcio A), i el mateix dia es va tornar a l'opcio B, que es
# aquesta: eren 17-23 preses i massa temps. Remarcar continua valent per una
# altra cosa -les preses noves de 45 graus, marcades al centre del cargol,
# tenen un soroll de theta1 cinc vegades mes petit, i aixo la correccio no ho
# dona-, i no xoca amb aixo: una presa remarcada surt a l'eix i la correccio
# ja no la toca.
#
# Nomes es corregeix el desplacament AL LLARG de la barra 1. El de costat es
# mesura com a diagnostic, pero no es corregeix: es el que es va refutar.

MARCA_A_LEIX_MM = 1.0  # |delta| per sota d'aixo, la marca es a l'eix. 1 mm posa
                       # uns 0,4 graus a theta2, del mateix ordre que el soroll
                       # de theta2 de les preses ben marcades (0,22-0,30 graus)


def _muntatge():
    """La L1 i el pivot 2 -> gomet de parametres.json, o None si no es pot llegir.

    Nomes serveixen per passar delta a mm i per saber quina rao gomet / L1 dona
    el regle: la mesura en si es en pixels i no depen de cap escala.
    """
    try:
        from pendol.fisica import Parametres
        p = Parametres.des_de_json(Path(__file__).resolve().parent.parent / "parametres.json")
        return p.L1, p.r_gomet
    except (Exception, SystemExit):  # noqa: BLE001  sense parametres, es mesura igual en px
        return None


def mesura_marca_articulacio(P, A, G, muntatge=None) -> dict:
    """On es la marca de l'articulacio respecte de l'eix del pivot 2.

    P, A, G: posicions en pixels (N, 2) del pivot fix, la marca de
    l'articulacio i el gomet. P pot ser una sola posicio.

    El model: el pivot 2 de debo es J = A - delta*u1 - eps*n1, amb u1 la
    direccio pivot -> marca i n1 la perpendicular. Es busquen delta, eps i la
    distancia r fins al gomet que fan |G - J| tan constant com es pugui, sobre
    tots els fotogrames. delta > 0 vol dir que la marca queda mes enlla del
    pivot 2, cap a la punta de la barra 1.

    eps, el desplacament de costat, es mesura NOMES COM A DIAGNOSTIC: es el que
    va refutar el 30/08/2026 i no se'n treu cap veredicte.

    La incertesa es la formal de l'ajust. Les mostres son fotogrames de
    seguiment, no gaire correlacionades entre si, pero no del tot
    independents: la de debo pot ser una mica mes gran.

    Mai aixeca: si no es pot mesurar, torna «mesurada»: False amb el motiu.
    """
    from scipy.optimize import least_squares

    A = np.asarray(A, dtype=float)
    G = np.asarray(G, dtype=float)
    P = np.broadcast_to(np.asarray(P, dtype=float), A.shape)
    u = A - P
    L = np.hypot(u[:, 0], u[:, 1])
    bons = np.isfinite(L) & (L > 0) & np.all(np.isfinite(G), axis=1)
    if bons.sum() < 50:
        return {"mesurada": False, "motiu": f"nomes {int(bons.sum())} fotogrames amb les tres marques"}
    u, L, A, G = u[bons], L[bons], A[bons], G[bons]
    u1 = u / L[:, None]
    n1 = np.column_stack([-u1[:, 1], u1[:, 0]])
    v = G - A
    Lv = np.hypot(v[:, 0], v[:, 1])

    # Punt de partida: la regressio de esborranys/_diag_articulacio.py, amb
    # l'angle entre bracos calculat des de la marca tal com es. Es aproximada
    # -aquell angle porta justament l'error que es busca-, i per aixo despres
    # es refina sense aproximar res.
    c = np.sum(u1 * v, axis=1) / Lv
    s = np.sum(n1 * v, axis=1) / Lv
    coef = np.linalg.lstsq(np.column_stack([np.ones_like(c), c, s]), Lv**2, rcond=None)[0]
    r0 = float(np.sqrt(max(coef[0], 1.0)))

    def residu(x):
        d, e, r = x
        J = A - d * u1 - e * n1
        return np.hypot(*(G - J).T) - r

    res = least_squares(residu, [-coef[1] / (2 * r0), -coef[2] / (2 * r0), r0])
    if not res.success:
        return {"mesurada": False, "motiu": f"l'ajust no convergeix: {res.message}"}
    delta, eps, r = (float(x) for x in res.x)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * (2 * res.cost / max(len(L) - 3, 1))
        sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        sig = np.array([np.inf, np.inf, np.inf])

    L1_px = float(np.median(L)) - delta          # del pivot 1 al pivot 2 de debo
    sortida = {
        "mesurada": True,
        "motiu": None,
        "delta_px": delta,
        "incertesa_delta_px": float(sig[0]),
        "de_costat_px": eps,
        "incertesa_de_costat_px": float(sig[1]),
        "gomet_px": r,
        "L1_px": L1_px,
        # L'error que posa a theta2: l'angle amb que es veu delta des del
        # gomet, maxim quan els bracos fan angle recte.
        "error_theta2_maxim_graus": float(math.degrees(math.asin(min(abs(delta) / r, 1.0)))),
        "rao_gomet_L1": float(np.median(Lv) / np.median(L)),
        "rao_gomet_L1_a_leix": r / L1_px,
        "vibracio_px": float(Lv.std()),
        "vibracio_a_leix_px": float(np.std(res.fun)),
        "fotogrames": int(len(L)),
        "criteri_mm": MARCA_A_LEIX_MM,
    }
    if muntatge is None:
        muntatge = _muntatge()
    if muntatge is None:
        sortida.update({"delta_mm": None, "a_leix": None,
                        "motiu": "sense parametres.json no es pot passar a mm"})
        return sortida
    L1_m, r_gomet_m = muntatge
    mm_per_px = 1000.0 * L1_m / L1_px
    sortida["delta_mm"] = delta * mm_per_px
    sortida["incertesa_delta_mm"] = float(sig[0]) * mm_per_px
    sortida["rao_del_regle"] = r_gomet_m / L1_m
    # Si la presa no plega prou els bracos, delta no es pot mesurar amb la
    # precisio que demana el criteri, i llavors no es diu ni que si ni que no.
    if sortida["incertesa_delta_mm"] > MARCA_A_LEIX_MM:
        sortida["a_leix"] = None
        sortida["motiu"] = (f"no es pot dir: la incertesa de delta es de "
                            f"{sortida['incertesa_delta_mm']:.1f} mm, mes que el criteri")
    else:
        sortida["a_leix"] = bool(abs(sortida["delta_mm"]) <= MARCA_A_LEIX_MM)
    return sortida


def cal_corregir(marca: dict) -> tuple[bool, str | None]:
    """Si la marca s'ha de moure a l'eix, i si no, per que.

    Es corregeix quan la mesura diu que es FORA de l'eix (|delta| mes gran que
    MARCA_A_LEIX_MM i mesurable), i prou. Per sota d'aquell milimetre l'error
    de theta2 queda per sota del soroll i la presa es deixa tal com es va
    marcar: aixi una presa ben marcada, o remarcada, surt identica.

    Es va pensar en una segona condicio -corregir nomes si la rao gomet / L1
    s'acosta a la del regle- i es va descartar abans de programar-la: la rao
    te un 2-3 % de soroll d'una presa a l'altra (les preses noves ben marcades
    donen de 1,156 a 1,181), o sigui que com a condicio no discrimina res.
    """
    if not marca.get("mesurada"):
        return False, f"no s'ha pogut mesurar: {marca.get('motiu')}"
    if marca.get("a_leix") is None:
        return False, f"no es pot dir si es a l'eix: {marca.get('motiu')}"
    if marca["a_leix"]:
        return False, f"ja es a l'eix (|delta| <= {MARCA_A_LEIX_MM:g} mm)"
    return True, None


def corregeix_marca(P, A, delta_px: float) -> np.ndarray:
    """La marca de l'articulacio moguda delta px cap al pivot 1, al llarg de la barra 1.

    J = A - delta*u1, amb u1 la direccio pivot -> marca. Com que J cau sobre la
    mateixa recta que A, theta1 no canvia; el que canvia es theta2, que ara es
    mesura des del centre del cargol.
    """
    A = np.asarray(A, dtype=float)
    P = np.broadcast_to(np.asarray(P, dtype=float), A.shape)
    u = A - P
    L = np.hypot(u[:, 0], u[:, 1])
    L = np.where(L == 0.0, 1.0, L)
    return A - delta_px * u / L[:, None]


def llegeix_del_trk(cami, cua_s=CUA_PER_DEFECTE, corregeix_articulacio=False,
                    corregeix_marca_articulacio=True, des_del_fotograma=None):
    """Del .trk a (t, x_articulacio, y_articulacio, x_extrem, y_extrem, meta).

    Les posicions surten relatives al pivot i ja adrecades, o sigui que despres
    a_angles() s'ha de cridar amb pivot=(0,0) i y_avall=True.

    Amb des_del_fotograma, la presa comenca en aquell fotograma (trk.retalla).
    """
    d = trk.retalla(trk.llegeix_trk(cami), des_del_fotograma)
    k, A, G, noms, (k_fix, P) = trk.tres_pistes(d)
    fps = d["fps"]
    t = (k - k[0]) * d["delta_t_s"]

    # El pivot hauria de ser quiet. Es fa servir una posicio constant, no la de
    # cada fotograma: el seguiment del punt fix es soroll, i posar-lo a theta1
    # seria fabricar-lo.
    #
    # I es la MEDIANA, no la mitjana. Al 90_2 del 30/08/2026 la marca del pivot
    # se'n va 113 px durant els dos primers segons -el seguidor la perd mentre
    # s'amolla el pendol- i torna al seu lloc: la mitjana se n'anava darrere i
    # contaminava tota la presa. La mediana ni se n'assabenta.
    #
    # El fix porta ELS SEUS fotogrames, que poden no ser els de la presa (vegeu
    # trk.tres_pistes). Com que d'aqui nomes en surt una constant, tant se val
    # quins siguin: el que compta es que n'hi hagi prou i saber quins cobreix.
    centre = np.median(P, axis=0)
    desviament = np.hypot(*(P - centre).T)
    t_fix = (k_fix - k[0]) * d["delta_t_s"]
    m = max(10, int(round(fps)))
    tremolor = {
        "max_px": float(desviament.max()),
        "quan_s": float(t_fix[int(desviament.argmax())]),
        "primer_segon_px": float(desviament[:m].max()),
        "ultim_segon_px": float(desviament[-m:].max()),
        "cobreix_de_s": float(t_fix[0]),
        "cobreix_fins_s": float(t_fix[-1]),
    }
    P_fix = np.broadcast_to(centre, (len(k), 2))

    # La vertical surt de la cua, i la cua es del fix nomes com a constant. Pero
    # si el fix no arriba fins alla, la constant ve d'un altre tros del video i
    # aixo s'ha de dir: si la camera va derivar entremig, l'origen del pivot no
    # es el que tocaria. Al 90_11 el fix s'acaba 27 s abans que la presa.
    fix_cobreix_la_cua = t_fix[-1] >= t[-1] - cua_s
    fix_cobreix_el_principi = t_fix[0] <= t[0] + 0.5

    # LA MARCA DE L'ARTICULACIO, abans de res mes: si es fora de l'eix, la
    # vertical, la rigidesa i els angles s'han de calcular des del centre del
    # cargol, no des d'on es va clicar. Vegeu mesura_marca_articulacio.
    marca = mesura_marca_articulacio(P_fix, A, G)
    corregir, motiu = cal_corregir(marca)
    if not corregeix_marca_articulacio:
        corregir, motiu = False, "desactivada (--sense-corregir-marca)"
    marca["corregida"] = bool(corregir)
    marca["motiu_sense_correccio"] = motiu
    if corregir:
        A = corregeix_marca(P_fix, A, marca["delta_px"])
        # La prova de dins: mesurada sobre la marca ja moguda, delta ha de
        # sortir zero. Si no, alguna cosa del calcul no quadra.
        despres = mesura_marca_articulacio(P_fix, A, G)
        marca["delta_despres_px"] = despres.get("delta_px")
        marca["delta_despres_mm"] = despres.get("delta_mm")

    geo = geometria_del_repos(P_fix, A, G, fps, cua_s)
    geo["tremolor_del_pivot"] = tremolor
    geo["fix_cobreix_la_cua"] = bool(fix_cobreix_la_cua)
    geo["fix_cobreix_el_principi"] = bool(fix_cobreix_el_principi)

    # Comenca el marcatge amb el pendol quiet? Si no, la presa no diu de quin
    # angle es va amollar, per mes que la carpeta es digui 90 o 150: el
    # seguiment ja arrenca amb el tir engegat. Al 150_6 el gomet va a 8257 px/s
    # al primer fotograma.
    velocitat = np.hypot(*np.diff(G, axis=0).T) * fps
    quiets = int(np.argmax(velocitat > 0.02 * velocitat.max())) if len(velocitat) else 0
    # DIAGNOSTIC i prou des del 24/09/2026: qui decideix si la presa arrenca
    # del repos es l'amollada (vegeu processa i amollada.arrenca_del_repos).
    geo["quiet_al_principi_s"] = float(quiets / fps)
    geo["velocitat_inicial_px_s"] = float(velocitat[0]) if len(velocitat) else 0.0

    problemes = []
    if geo["estabilitat_graus"] > LLINDAR_ESTABILITAT:
        problemes.append(
            f"la vertical canvia {geo['estabilitat_graus']:.2f} graus segons "
            f"quanta cua es promitgi (limit {LLINDAR_ESTABILITAT})"
        )
    if geo["residu_px"] > LLINDAR_RESIDU:
        problemes.append(
            f"el gomet encara recorre {geo['residu_px']:.0f} px a l'ultim segon "
            f"(limit {LLINDAR_RESIDU:.0f})"
        )
    if problemes:
        raise SystemExit(
            f"{cami}: la cua no s'ha assossegat i sense repos no hi ha vertical.\n"
            + "".join(f"  - {p}\n" for p in problemes)
            + "  Aquesta presa no es pot adrecar amb ella mateixa, i heretar la\n"
            "  vertical d'una altra nomes valdria si fos de la mateixa gravacio\n"
            "  seguida, sense haver tocat la camera."
        )

    # Les barres son rigides: en pixels, L1 i L2 haurien de ser constants. Si
    # no ho son, es soroll de seguiment i marca el sostre de precisio dels
    # angles: una desviacio de s pixels a la marca son atan(s/L) graus.
    L1 = np.hypot(*(A - P_fix).T)
    L2 = np.hypot(*(G - A).T)
    geo["rigidesa"] = {
        "L1_sigma_px": float(L1.std()), "L1_sigma_relatiu": float(L1.std() / L1.mean()),
        "L2_sigma_px": float(L2.std()), "L2_sigma_relatiu": float(L2.std() / L2.mean()),
        "soroll_angular_graus": [float(math.degrees(L1.std() / L1.mean())),
                                 float(math.degrees(L2.std() / L2.mean()))],
    }

    # A la rigidesa de dalt, A ja es la marca corregida, si ho ha estat: el
    # soroll de theta2 surt de |G - J|, que es el que de debo ha de ser
    # constant. Una part del «soroll» d'abans era l'error de la marca.
    geo["marca_articulacio"] = marca

    J, G_g = adreca_i_corregeix(P_fix, A, G, geo, corregeix_articulacio)
    meta = {
        "fitxer_trk": str(cami),
        "video_del_trk": d.get("video"),
        "pistes": noms,
        "delta_t_s": d["delta_t_s"],
        "angle_de_l_eix_al_trk_graus": d.get("angle_graus"),
        "articulacio_corregida": bool(corregeix_articulacio),
        "geometria": geo,
    }
    # Nomes si n'hi ha: una presa sense retall ha de sortir byte a byte igual.
    if des_del_fotograma is not None:
        meta["retall_des_del_fotograma"] = int(des_del_fotograma)
    print(f"  tres pistes: {', '.join(noms)}   {len(t)} fotogrames a {fps:.2f} fps")
    if des_del_fotograma is not None:
        print(f"  retall      la presa comenca al fotograma {int(des_del_fotograma)} (el fix no es retalla)")
    print(f"  vertical    {geo['gir_graus']:+.3f} graus "
          f"(estable a {geo['estabilitat_graus']:.3f}, residu {geo['residu_px']:.1f} px)")
    r = geo["rigidesa"]
    print(f"  articulacio {geo['delta_px']:+.1f} px de costat, al repos "
          f"= {math.degrees(math.atan2(geo['delta_px'], geo['L1_px'])):+.2f} graus de theta1"
          + ("  (CORREGIT)" if corregeix_articulacio else "  (mesurat, NO corregit)"))
    ma = geo["marca_articulacio"]
    if not ma["mesurada"]:
        print(f"  marca       no es pot mesurar: {ma['motiu']}")
    else:
        mm = f" = {ma['delta_mm']:+.1f} mm" if ma.get("delta_mm") is not None else ""
        veredicte = {True: "a l'eix", False: "NO ES A L'EIX", None: "no es pot dir"}[ma["a_leix"]]
        print(f"  marca       {ma['delta_px']:+.1f} px{mm} al llarg de la barra 1 (+ = mes enlla "
              f"del pivot 2): fins a {ma['error_theta2_maxim_graus']:.2f} graus a theta2 "
              f"-> {veredicte} (criteri {MARCA_A_LEIX_MM:.0f} mm)")
        print(f"              rao marca-gomet / L1 {ma['rao_gomet_L1']:.3f}, amb la marca a l'eix "
              f"{ma['rao_gomet_L1_a_leix']:.3f}; vibracio {ma['vibracio_px']:.2f} -> "
              f"{ma['vibracio_a_leix_px']:.2f} px")
        if ma["corregida"]:
            print(f"              CORREGIDA: moguda {ma['delta_px']:+.2f} px cap al centre del "
                  f"cargol; tornada a mesurar, queda a {ma['delta_despres_px']:+.3f} px")
        else:
            print(f"              no es corregeix: {ma['motiu_sense_correccio']}")
    print(f"  soroll      les barres varien {r['L1_sigma_px']:.1f} i {r['L2_sigma_px']:.1f} px "
          f"= {r['soroll_angular_graus'][0]:.2f} i {r['soroll_angular_graus'][1]:.2f} graus d'angle")
    print(f"  pivot       es mou {tremolor['max_px']:.1f} px com a molt "
          f"(primer segon {tremolor['primer_segon_px']:.1f}, ultim {tremolor['ultim_segon_px']:.2f})")
    print(f"  el fix      cobreix de {tremolor['cobreix_de_s']:+.2f} a "
          f"{tremolor['cobreix_fins_s']:.2f} s de la presa, que va de 0.00 a {t[-1]:.2f}")
    if not fix_cobreix_la_cua:
        print(f"     ATENCIO: el fix no arriba a la cua, li falten "
              f"{t[-1] - tremolor['cobreix_fins_s']:.1f} s. L'origen del pivot surt")
        print(f"     d'un altre tros del video: val si la camera no s'ha mogut,")
        print(f"     i el tremolor de {tremolor['max_px']:.1f} px es el que ho diu.")
    if tremolor["max_px"] > 5.0:
        print(f"     ATENCIO: el punt fix no es quiet. Si es al principi, es que el")
        print(f"     seguidor el perd mentre s'amolla, i aquells fotogrames no valen.")
    # Aqui hi havia un avis de «el marcatge arrenca amb el pendol ja en
    # marxa» si el detector de pixels donava menys de 0,1 s de quietud. Es va
    # treure el 24/09/2026: aquell detector es dispara amb una sola sacsejada
    # del seguidor i l'avis sortia a preses amollades del repos. Si una presa
    # arrenca del repos ho diu ara l'amollada, a processa(), que la mesura
    # sobre els angles amb el soroll de la presa.
    return t, J[:, 0], J[:, 1], G_g[:, 0], G_g[:, 1], meta


def _reomple_forats(t, *senyals):
    """Interpola linealment els fotogrames que el Tracker va perdre (NaN)."""
    sortida = []
    for i, s in enumerate(senyals):
        buits = np.isnan(s)
        if buits.all():
            raise ValueError(
                f"La columna {i + 1} de posicions no te cap valor numeric.\n"
                "Segurament les columnes s'han identificat malament: passa-les "
                "a ma amb --columnes t,x1,y1,x2,y2 (indexs comencant a 0)."
            )
        if buits.any():
            if buits.mean() > 0.2:
                print(f"  AVIS GREU: s'ha perdut el {buits.mean():.0%} dels fotogrames d'una columna")
            print(f"  {buits.sum()} fotogrames perduts, interpolats linealment")
            s = s.copy()
            s[buits] = np.interp(t[buits], t[~buits], s[~buits])
        sortida.append(s)
    return sortida


def processa(
    cami_entrada,
    cami_sortida,
    pivot=(0.0, 0.0),
    columnes=None,
    finestra_s=0.05,
    ordre=3,
    y_avall=False,
    inverteix=False,
    meta_extra=None,
    cua_s=CUA_PER_DEFECTE,
    corregeix_articulacio=False,
    corregeix_marca=True,
    des_del_fotograma=None,
):
    print(f"Llegint {cami_entrada}")
    meta_extra = dict(meta_extra or {})

    if str(cami_entrada).lower().endswith(".trk"):
        # Del projecte del Tracker: tres pistes, i la geometria resolta amb la
        # cua del video. Surt ja relatiu al pivot, girat a la vertical i amb el
        # desviament de l'articulacio tret.
        t, x1, y1, x2, y2, meta_trk = llegeix_del_trk(
            cami_entrada, cua_s, corregeix_articulacio, corregeix_marca, des_del_fotograma)
        meta_extra.update(meta_trk)
        capcalera = meta_trk["pistes"]
        pivot, y_avall = (0.0, 0.0), True
    elif des_del_fotograma is not None:
        raise SystemExit("--des-del-fotograma nomes val amb un .trk")
    else:
        matriu, capcalera = llegeix_cru(cami_entrada)
        t, x1, y1, x2, y2 = tria_columnes(matriu, columnes).T

    # Temps relatiu al primer fotograma valid.
    bons = ~np.isnan(t)
    t, x1, y1, x2, y2 = (v[bons] for v in (t, x1, y1, x2, y2))
    t = t - t[0]
    x1, y1, x2, y2 = _reomple_forats(t, x1, y1, x2, y2)

    th1, th2 = a_angles(t, x1, y1, x2, y2, pivot, y_avall, inverteix)

    # Base temporal uniforme (el Tracker sol donar-la ja uniforme, pero si
    # has retallat fotogrames o el video te fps variable, cal arreglar-ho).
    # La tolerancia es RELATIVA: el Tracker escriu els temps arrodonits a
    # unes poques xifres i aixo fa variar dt en una part per milio, cosa que
    # no es cap irregularitat real.
    dt_med = float(np.median(np.diff(t)))
    if np.ptp(np.diff(t)) > 0.05 * dt_med:
        fps = 1.0 / dt_med
        print(f"  base temporal irregular -> remostrejo a {fps:.2f} fps")
        t_unif = np.arange(t[0], t[-1] + 0.5 / fps, 1.0 / fps)
        th1 = np.interp(t_unif, t, th1)  # ja desembolicats: interpolar es segur
        th2 = np.interp(t_unif, t, th2)
        t = t_unif
    else:
        # Uniforme: estimo els fps amb la durada total, que compensa
        # l'arrodoniment de l'export molt millor que la mediana de dt.
        fps = (len(t) - 1) / (t[-1] - t[0])
    dt = 1.0 / fps

    # Derivada suavitzada: Savitzky-Golay ajusta un polinomi local i en
    # deriva el polinomi, en comptes d'amplificar el soroll pixel a pixel.
    finestra = int(round(finestra_s * fps))
    finestra = max(ordre + 2, finestra + (finestra % 2 == 0))  # senar i prou gran
    if finestra >= len(t):
        finestra = len(t) - 1 - (len(t) % 2 == 0)
    print(f"  fps = {fps:.2f}   finestra Savitzky-Golay = {finestra} mostres ({finestra*dt*1000:.0f} ms)")

    th1_s = savgol_filter(th1, finestra, ordre)
    th2_s = savgol_filter(th2, finestra, ordre)
    w1 = savgol_filter(th1, finestra, ordre, deriv=1, delta=dt)
    w2 = savgol_filter(th2, finestra, ordre, deriv=1, delta=dt)

    meta = {
        "origen": "tracker",
        "fitxer_cru": str(cami_entrada),
        "capcalera_original": capcalera,
        "fps": float(fps),
        "durada_s": float(t[-1]),
        "n_mostres": int(len(t)),
        "pivot": list(pivot),
        "y_avall": bool(y_avall),
        "signe_invertit": bool(inverteix),
        "savgol": {"finestra_mostres": int(finestra), "ordre": int(ordre)},
        "angles_inicials_graus": [float(np.degrees(th1_s[0])), float(np.degrees(th2_s[0]))],
        **(meta_extra or {}),
    }
    dades.desa(cami_sortida, t, th1_s, th2_s, w1, w2, meta)

    # L'AMOLLADA, el zero del rellotge d'aquesta presa, i el que decideix si
    # arrenca del repos. Es mesura sobre el CSV JA DESAT I RELLEGIT, no sobre
    # les series de la memoria: el CSV arrodoneix a deu xifres, i aixi el que
    # queda a les metadades es exactament el que dona
    #     python -m pendol.amollada <presa>.csv
    # sense cap diferencia a l'ultima xifra. troba() no aixeca mai: si no hi
    # ha amollada, el bloc ho diu amb el motiu.
    taula, _ = dades.carrega(cami_sortida)
    meta["amollada"] = amollada.troba_de_taula(taula, meta)
    dades.desa_meta(cami_sortida, meta)

    print(f"\nDesat: {cami_sortida}")
    r = meta["amollada"]
    if r["hi_ha_amollada"]:
        print(f"  amollada a {r['amollada_s']:.4f} s +- {1000*r['incertesa_s']:.1f} ms, "
              f"amb {1000*r['repos_abans_s']:.0f} ms de repos abans: arrenca del repos")
        for a in r["avisos"]:
            print(f"     ATENCIO: {a}")
    else:
        print(f"  SENSE AMOLLADA, no arrenca del repos: {r['motiu']}")
    print(f"  durada {t[-1]:.2f} s, {len(t)} mostres")
    print(f"  angles inicials: {np.degrees(th1_s[0]):.2f} / {np.degrees(th2_s[0]):.2f} graus")
    print(f"  recorregut de theta1: {np.degrees(th1_s.min()):.0f} a {np.degrees(th1_s.max()):.0f} graus")
    return cami_sortida


def autoprova() -> int:
    """La mesura de la marca ha de recuperar un desplacament CONEGUT.

    Tirs integrats amb el model fisic de debo, passats a pixels amb l'escala
    de les preses (L1 ~ 195 px), amb la marca de l'articulacio desplacada el
    que es digui i 1 px de soroll de seguiment a la marca i al gomet. Els
    criteris es van escriure abans de mirar cap resultat (pla del pas 3 de
    D1b, 25/09/2026):

      - delta, recuperada a menys de 0,3 px;
      - el desplacament de costat, a menys de 0,5 px;
      - amb la marca a l'eix, el veredicte ha de ser «a l'eix».

    I la correccio (pla de l'opcio B, 25/09/2026):

      - amb delta = 7 px, l'error rms de theta2 ha de baixar cinc vegades com
        a minim. ES MIRA SOBRE LES POSICIONS SENSE SOROLL, i es va decidir
        abans d'executar-la: el soroll d'1 px ja posa uns 0,36 graus rms a
        theta2, i amb soroll cap correccio no podria dividir per cinc un error
        que val 1,3 graus rms. El que s'ha de provar es que treu l'error de la
        marca, no el soroll. La versio amb soroll s'imprimeix com a informacio;
      - amb delta = 0, no es corregeix;
      - un cop corregida, la delta que es torna a mesurar queda per sota de
        0,3 px.

    Els tirs de 45 graus hi son a posta: alla els bracos es pleguen poc, i es
    on la mesura te menys palanca.
    """

    def theta2_rms(G_, J_, th2):
        """L'error rms de l'angle de J_ -> G_ contra la veritat, en graus."""
        v = G_ - J_
        d = np.arctan2(v[:, 0], v[:, 1]) - th2
        return float(np.degrees(np.sqrt(np.mean(((d + np.pi) % (2 * np.pi) - np.pi) ** 2))))

    from pendol.fisica import Parametres
    from pendol.simulador import simula

    p = Parametres.des_de_json(Path(__file__).resolve().parent.parent / "parametres.json")
    fps, px_per_m = 240.0, 1625.0
    P = np.array([400.0, 150.0])        # el pivot fix, en pixels d'imatge (y cap avall)
    casos = [  # (angle de sortida, delta px, de costat px)
        (90.0, 7.0, 0.0),
        (45.0, 7.0, 0.0),
        (90.0, 0.0, 0.0),
        (90.0, 7.0, 3.0),
    ]
    problemes = []
    print("Autoprova de la mesura de la marca de l'articulacio")
    print(f"   tirs de 20 s a {fps:.0f} fps integrats amb pendol/fisica.py, a {px_per_m:.0f} px/m,")
    print("   amb 1 px de soroll a la marca i al gomet\n")
    print(f"   {'angle':>6} {'delta':>7} {'costat':>7} {'->':>3} {'delta':>7} {'costat':>7} "
          f"{'mm':>6} {'veredicte':>12}")
    correccions = []
    for k, (angle, delta, eps) in enumerate(casos):
        rng = np.random.default_rng(700 + k)
        _, y = simula(p, np.radians([angle, angle, 0.0, 0.0]), temps=20.0, fps=fps,
                      rtol=1e-7, atol=1e-9)
        th1, th2 = y[0], y[1]
        u1 = np.column_stack([np.sin(th1), np.cos(th1)])     # y cap avall: penjar es (0, +)
        n1 = np.column_stack([-u1[:, 1], u1[:, 0]])          # el mateix conveni que la mesura
        J = P + p.L1 * px_per_m * u1
        G = J + p.r_gomet * px_per_m * np.column_stack([np.sin(th2), np.cos(th2)])
        A_net = J + delta * u1 + eps * n1
        G_net = G
        A = A_net + rng.normal(0.0, 1.0, A_net.shape)
        G = G_net + rng.normal(0.0, 1.0, G_net.shape)
        m = mesura_marca_articulacio(P, A, G, muntatge=(p.L1, p.r_gomet))
        if not m["mesurada"]:
            problemes.append(f"{angle:.0f} graus, delta {delta}: no es pot mesurar ({m['motiu']})")
            continue
        veredicte = {True: "a l'eix", False: "fora", None: "no es sap"}[m["a_leix"]]
        print(f"   {angle:>6.0f} {delta:>7.2f} {eps:>7.2f} {'':>3} {m['delta_px']:>7.2f} "
              f"{m['de_costat_px']:>7.2f} {m['delta_mm']:>6.2f} {veredicte:>12}")
        if abs(m["delta_px"] - delta) > 0.3:
            problemes.append(f"{angle:.0f} graus: delta {m['delta_px']:.2f} px en comptes de {delta}")
        if abs(m["de_costat_px"] - eps) > 0.5:
            problemes.append(f"{angle:.0f} graus: de costat {m['de_costat_px']:.2f} px en comptes de {eps}")
        if delta == 0.0 and m["a_leix"] is not True:
            problemes.append(f"{angle:.0f} graus: amb la marca a l'eix el veredicte es «{veredicte}»")
        correccions.append((angle, delta, eps, m, A_net, G_net, A, G, th2))

    print("\n   La correccio: error rms de theta2 contra la veritat, en graus")
    print(f"   {'angle':>6} {'delta':>7} {'costat':>7} {'corregeix':>10} {'sense soroll':>21} "
          f"{'amb soroll':>21} {'delta despres':>14}")
    for angle, delta, eps, m, A_net, G_net, A, G, th2 in correccions:
        corregir, motiu = cal_corregir(m)
        if delta == 0.0:
            if corregir:
                problemes.append(f"{angle:.0f} graus amb la marca a l'eix: la corregeix igualment")
            print(f"   {angle:>6.0f} {delta:>7.2f} {eps:>7.2f} {'no':>10}   ({motiu})")
            continue
        if not corregir:
            problemes.append(f"{angle:.0f} graus, delta {delta}: no la corregeix ({motiu})")
            continue
        net_abans = theta2_rms(G_net, A_net, th2)
        net_despres = theta2_rms(G_net, corregeix_marca(P, A_net, m["delta_px"]), th2)
        brut_abans = theta2_rms(G, A, th2)
        A_corr = corregeix_marca(P, A, m["delta_px"])
        brut_despres = theta2_rms(G, A_corr, th2)
        m2 = mesura_marca_articulacio(P, A_corr, G, muntatge=(p.L1, p.r_gomet))
        print(f"   {angle:>6.0f} {delta:>7.2f} {eps:>7.2f} {'si':>10} {net_abans:>9.3f} -> {net_despres:<8.4f}"
              f" {brut_abans:>9.3f} -> {brut_despres:<8.3f} {m2['delta_px']:>+13.3f}")
        # Amb desplacament de costat, la correccio al llarg de la barra no el
        # treu -a posta- i l'error que queda es el d'aquell; per aixo el
        # criteri de les cinc vegades nomes es demana amb el de costat a zero.
        if eps == 0.0 and not net_despres <= net_abans / 5:
            problemes.append(f"{angle:.0f} graus: l'error de theta2 passa de {net_abans:.3f} a "
                             f"{net_despres:.3f} graus, no baixa cinc vegades")
        if abs(m2["delta_px"]) > 0.3:
            problemes.append(f"{angle:.0f} graus: un cop corregida, la delta que queda es "
                             f"{m2['delta_px']:.3f} px")

    # EL RETALL (pla del 26/09/2026, criteris escrits abans de mirar): treu
    # exactament els fotogrames anteriors de les pistes que no son el fix,
    # deixa el fix intacte, no toca el diccionari d'entrada, i amb None no fa res.
    k = np.arange(100)
    fals = {"pistes": {nom: (k.copy(), k * 1.0, k * 2.0) for nom in ("Fix", "rodament", "extrem")}}
    r = trk.retalla(fals, 40)
    bo = (all(np.array_equal(r["pistes"][n][0], np.arange(40, 100)) for n in ("rodament", "extrem"))
          and np.array_equal(r["pistes"]["rodament"][1], np.arange(40, 100) * 1.0)
          and np.array_equal(r["pistes"]["Fix"][0], k)
          and len(fals["pistes"]["rodament"][0]) == 100
          and trk.retalla(fals, None) is fals)
    print(f"  retall: des del 40 queden {len(r['pistes']['rodament'][0])} fotogrames a rodament i extrem, "
          f"{len(r['pistes']['Fix'][0])} al fix  {'OK' if bo else 'FALLA'}")
    if not bo:
        problemes.append("el retall no treu exactament els fotogrames anteriors, o toca el fix")

    print()
    if problemes:
        print("AUTOPROVA FALLADA:")
        for q in problemes:
            print(f"  - {q}")
        return 1
    print("OK -> recupera el desplacament de la marca a menys de 0,3 px, tambe a 45")
    print("     graus, amb la marca a l'eix diu que hi es i no la toca, i quan en es")
    print("     fora, la correccio treu l'error de theta2 que hi posava. Limit: suposa que les")
    print("     barres son rigides i que el soroll de seguiment no depen de la")
    print("     posicio; no pot veure un gomet que no sigui sobre l'eix de la barra 2,")
    print("     perque aquest tambe queda a distancia constant del pivot 2.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada", nargs="?", help="projecte .trk del Tracker (tres pistes) o CSV exportat")
    ap.add_argument("--sortida", help="CSV net de sortida (format canonic)")
    ap.add_argument("--cua", type=float, default=CUA_PER_DEFECTE,
                    help="segons de final de video que es promitgen per treure la "
                         f"vertical, nomes amb .trk (per defecte {CUA_PER_DEFECTE})")
    ap.add_argument("--pivot", nargs=2, type=float, default=[0.0, 0.0], metavar=("X", "Y"),
                    help="coordenades del pivot fix (per defecte 0 0: origen del Tracker al pivot)")
    ap.add_argument("--columnes", help="indexs de t,x1,y1,x2,y2 si l'auto-deteccio falla, p.ex. 0,1,2,4,5")
    ap.add_argument("--finestra", type=float, default=0.05, help="finestra del filtre de derivacio (s)")
    ap.add_argument("--ordre", type=int, default=3, help="ordre del polinomi de Savitzky-Golay")
    ap.add_argument("--y-avall", action="store_true", help="el video tenia l'eix y cap avall")
    ap.add_argument("--inverteix-signe", action="store_true", help="camera en mirall: capgira el signe dels angles")
    ap.add_argument("--corregeix-articulacio", action="store_true",
                    help="desplaca la marca del rodament als seus delta px. DESACTIVAT "
                         "a posta: les dades del 30/08/2026 refuten el model i "
                         "aplicar-ho empitjora la coherencia. Vegeu adreca_i_corregeix()")
    ap.add_argument("--angle-nominal", type=float, help="angle de sortida que volies (graus), per a metadades")
    ap.add_argument("--sense-corregir-marca", action="store_true",
                    help="no moguis la marca de l'articulacio a l'eix encara que en sigui fora "
                         "(per refer les conversions d'abans del 25/09/2026)")
    ap.add_argument("--des-del-fotograma", type=int, metavar="N",
                    help="la presa comenca al fotograma N del .trk (el fix no es retalla). "
                         "Per a les preses on el principi del marcatge no es repos de debo; "
                         "les del lot son a dades/retalls.json")
    ap.add_argument("--autoprova", action="store_true",
                    help="comprova que la mesura i la correccio de la marca de l'articulacio "
                         "recuperen un desplacament conegut, i que el retall treu el que toca")
    args = ap.parse_args(argv)

    if args.autoprova:
        return autoprova()
    if not (args.entrada and args.sortida):
        ap.error("digue'm l'entrada i --sortida, o passa --autoprova")

    extra = {"angle_nominal_graus": args.angle_nominal} if args.angle_nominal is not None else {}
    processa(
        args.entrada,
        args.sortida,
        pivot=tuple(args.pivot),
        columnes=args.columnes,
        finestra_s=args.finestra,
        ordre=args.ordre,
        y_avall=args.y_avall,
        inverteix=args.inverteix_signe,
        meta_extra=extra,
        cua_s=args.cua,
        corregeix_articulacio=args.corregeix_articulacio,
        corregeix_marca=not args.sense_corregir_marca,
        des_del_fotograma=args.des_del_fotograma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
