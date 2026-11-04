"""Integrador del pendol magnetic: RK4 de pas fix.

PER QUE RK4 I NO UN INTEGRADOR ADAPTATIU com el DOP853 del pendol doble.
Alla s'integra una trajectoria i es vol la maxima precisio; aqui se n'hauran
d'integrar un milio, una per pixel del mapa. Un pas fix es previsible, no fa
cap crida a Python dins del bucle i es pot compilar sencer amb numba (bloc 8).

El preu d'aquesta decisio es que la validesa del pas dt s'ha de DEMOSTRAR en
comptes de deixar-la en mans de l'integrador. Es el que fa magnetic/valida.py,
i mes endavant el recalcul d'un tros del mapa amb dt/2.

De moment aquest modul sap integrar UNA trajectoria i dir com acaba. La
graella (bloc 7) encara no hi es.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from magnetic.compila import njit, prange
from magnetic.fisica import (
    Parametres,
    acceleracio_nucli,
    dins_de_labast,
    punts_dequilibri,
)

# Pas per defecte. Amb T0 = 1.07 s son 2143 passos per oscil.lacio.
# No es un valor sagrat: magnetic/valida.py el compara amb dt/2 i diu quant
# canvia el resultat.
#
# La historia, tota per decisio d'en Guillem:
#   0,002  fins al 25/10/2026.
#   0,001  des del 25/10. Amb la n mesurada (6,71) la forca creix molt de pressa
#          a tocar de l'imant, i amb 0,002 la prova 3b de valida.py (l'energia
#          amb els imants encesos) donava 2,4e-5 contra un llindar de 1e-6
#          escrit abans; amb 0,001 donava 7,5e-7.
#   0,0005 des del 26/10. Amb L = 0,285 i n = 6,87, 0,001 ja no passava la 3b:
#          1,03e-6. Amb 0,0008 dona 3,37e-7 i amb 0,0005, 3,22e-8; en Guillem
#          va triar 0,0005. Els mapes costen el doble que amb 0,001.
# Els numeros de la 3b surten d'esborranys/_energia_dt_25-10.py.
DT_PER_DEFECTE = 0.0005


@njit(cache=True)
def pas_rk4(x, y, vx, vy, dt, L, g, b, n, d0, im_x, im_y, im_K):
    """Un pas de Runge-Kutta de 4t ordre. Retorna (x, y, vx, vy).

    El sistema es de segon ordre, o sigui que la derivada de l'estat es
    (vx, vy, ax, ay). Escrit amb escalars perque numba el pugui compilar.
    """
    a1x, a1y = acceleracio_nucli(x, y, vx, vy, L, g, b, n, d0, im_x, im_y, im_K)
    k1x, k1y = vx, vy

    x2, y2 = x + 0.5 * dt * k1x, y + 0.5 * dt * k1y
    vx2, vy2 = vx + 0.5 * dt * a1x, vy + 0.5 * dt * a1y
    a2x, a2y = acceleracio_nucli(x2, y2, vx2, vy2, L, g, b, n, d0, im_x, im_y, im_K)

    x3, y3 = x + 0.5 * dt * vx2, y + 0.5 * dt * vy2
    vx3, vy3 = vx + 0.5 * dt * a2x, vy + 0.5 * dt * a2y
    a3x, a3y = acceleracio_nucli(x3, y3, vx3, vy3, L, g, b, n, d0, im_x, im_y, im_K)

    x4, y4 = x + dt * vx3, y + dt * vy3
    vx4, vy4 = vx + dt * a3x, vy + dt * a3y
    a4x, a4y = acceleracio_nucli(x4, y4, vx4, vy4, L, g, b, n, d0, im_x, im_y, im_K)

    sis = dt / 6.0
    return (
        x + sis * (k1x + 2.0 * vx2 + 2.0 * vx3 + vx4),
        y + sis * (k1y + 2.0 * vy2 + 2.0 * vy3 + vy4),
        vx + sis * (a1x + 2.0 * a2x + 2.0 * a3x + a4x),
        vy + sis * (a1y + 2.0 * a2y + 2.0 * a3y + a4y),
    )


def integra(p: Parametres, estat0, temps: float, dt: float = DT_PER_DEFECTE):
    """Integra una trajectoria sencera. Retorna (t, estats) amb estats (4, N).

    Guarda TOTS els passos: aquesta funcio es per mirar-se una trajectoria,
    no per fer mapes. La del mapa nomes en guardara el desenllac.
    """
    x, y, vx, vy = (float(v) for v in estat0)
    if not bool(dins_de_labast(x, y, p)):
        raise ValueError(
            f"El punt ({x:.4f}, {y:.4f}) es a {math.hypot(x, y):.4f} m del centre i el fil "
            f"nomes fa {p.L:.4f} m: aquest punt NO EXISTEIX. El pendol no hi pot arribar."
        )

    passos = int(round(temps / dt))
    t = np.arange(passos + 1, dtype=float) * dt
    estats = np.empty((4, passos + 1), dtype=float)
    estats[:, 0] = (x, y, vx, vy)

    L, g, b, n, d0 = p.L, p.g, p.b, p.n, p.d0
    im_x, im_y, im_K = p.im_x, p.im_y, p.im_K

    for k in range(passos):
        x, y, vx, vy = pas_rk4(x, y, vx, vy, dt, L, g, b, n, d0, im_x, im_y, im_K)
        if x * x + y * y >= L * L:
            raise RuntimeError(
                f"A t = {t[k+1]:.3f} s la massa ha sortit de l'abast del fil "
                f"(r = {math.hypot(x, y):.4f} m > L = {L:.4f} m). O el dt es massa gran, "
                "o el punt de sortida tenia massa energia."
            )
        estats[:, k + 1] = (x, y, vx, vy)

    return t, estats


# ---------------------------------------------------------------------------
# El criteri d'aturada
# ---------------------------------------------------------------------------
# Els quatre destins, tal com els fixa el CLAUDE.md. El negre i el gris NO es
# poden barrejar: el negre es un resultat fisic i el gris es un avis que el
# calcul s'ha quedat curt.
CENTRE = 0
NO_CLASSIFICAT = -1


@dataclass(frozen=True)
class Criteri:
    """Quan es considera que una trajectoria ja te desenllac.

    ELS VALORS SON PROVISIONALS I TRIATS A MA, mirant trajectories i comprovant
    que el codi diu el mateix que es veu. Van en un objecte propi i s'escriuen
    a totes les metadades perque es puguin tornar a discutir.

    EL CRITERI ES DE CONFINAMENT, NO DE VELOCITAT. Es va provar primer amb un
    llindar de velocitat -- "a prop d'un imant i anant a poc a poc" -- i era
    DOLENT: encertava 7 desenllacos de 12. El motiu es que una massa que passa
    per sobre d'un imant sense quedar-s'hi tambe hi va lenta un instant, al
    tombant de la corba. Anar lent es momentani; quedar-se, no.

    Per aixo el que es demana es que la massa NO SURTI d'una rodona petita
    durant t_confirma segons seguits. Una passada no ho pot fer; una captura,
    si. Amb aquest canvi passa a encertar-los tots.

    r_captura   Radi de la rodona, mesurat des del PUNT DE REPOS, que NO es la
                posicio de l'imant: amb els valors d'ara la massa reposa 9,34 mm
                cap endins de l'imant (magnetic.fisica.punts_dequilibri; eren
                5,85 abans del 25/10, sense l'estirada vertical dels imants).
    r_centre    El mateix per al centre, que es el quart desti.
    t_confirma  Quant de temps seguit s'hi ha de quedar, a un IMANT. El periode
                local a prop d'un imant es de 0,348 s amb els valors d'ara
                (0,345 fins al 25/10), o sigui que 0,5 s es mes d'una
                oscil.lacio sencera: una passada no hi arriba.
    t_confirma_centre
                El mateix per al CENTRE, i NO pot ser el mateix numero (25/10).
                El pou del centre es molt pla: el seu periode local es de 2,0 s
                amb els valors d'ara, i 0,5 s n'es una quarta part. Una massa
                que el travessa a poc a poc (2-3 cm/s) s'esta mes de 0,5 s dins
                de la rodona i despres acaba en un imant. Amb 0,5 s, aixo passava
                a 20 de 200 llancaments amb n = 3 (esborranys/
                _criteri_aturada_200_25-10.py); amb 2,5 s, a cap, ni amb n = 3 ni
                amb la n provisional de 6,688. LIMIT: a prop de la K critica el
                pou s'aplana encara mes i el seu periode creix sense limit; alla
                2,5 s poden no bastar.
    t_max       Si a aquest temps encara no s'ha decidit, es GRIS.

                SON 200 s, i es una DECISIO presa el 2026-09-29 (abans eren 60).
                Amb la b mesurada -0,039 s^-1, una quarta part de la inventada-
                les trajectories duren molt mes: a +-12 cm, 60 s deixen un
                81,12 % de pixels grisos i el mapa no es pot ni calcular. Amb
                150 i amb 200 en queda un 0,00 %.

                PUJAR-LO NO POT CANVIAR CAP PIXEL QUE JA CONVERGIA. Mira el
                bucle de desenllac_nucli(): el t_max nomes hi es el limit del
                for, i la trajectoria en surt amb un return en classificar-se.
                Un pixel que es decideix al segon 12 dona el mateix desti i el
                mateix temps, bit a bit, amb 60 o amb 200. I el cost tambe:
                nomes el paguen els que no convergeixen, i per aixo de 150
                endavant es gratis.

                D'ON SURT EL 200 I NO EL 150. Alla fora els imants gairebe no
                hi son i el que decideix quant triga la massa es NOMES la b:
                l'amplitud cau com exp(-b*t/2), o sigui que baixar de 17 cm -un
                raco del quadrat de +-12- als 0,8 cm del criteri son 157 s.
                El 200 es aixo mes marge. Fixa't que NO depen de la n ni de la
                K, que encara son inventades: depen de la b, que esta mesurada.
    """

    r_captura: float = 0.008
    r_centre: float = 0.008
    t_confirma: float = 0.5
    t_max: float = 200.0
    t_confirma_centre: float = 2.5

    def com_a_dict(self) -> dict:
        return {
            "tipus": "confinament",
            "r_captura_m": self.r_captura,
            "r_centre_m": self.r_centre,
            "t_confirma_s": self.t_confirma,
            "t_confirma_centre_s": self.t_confirma_centre,
            "t_max_s": self.t_max,
            "_mesurat_des_de": "els punts de repos, NO les posicions dels imants",
            "_estat": "PROVISIONALS: triats mirant trajectories, no mesurats.",
        }

    def avis(self) -> str:
        return (
            f"Criteri d'aturada PROVISIONAL, per confinament: no sortir de "
            f"{self.r_captura*100:.1f} cm del punt de repos\n"
            f"   durant {self.t_confirma:.2f} s seguits ({self.t_confirma_centre:.2f} s al centre). "
            f"Temps maxim {self.t_max:.0f} s."
        )


def destins(p: Parametres, criteri: Criteri):
    """Prepara les taules de destins per al bucle: (dest_x, dest_y, codi, radi2).

    Es calcula UN COP, abans de la graella. A dins del bucle dels pixels nomes
    hi entren aquests quatre vectors de numeros, que numba si que sap compilar.

    Si el centre es inestable no hi entra: aquell desti no existeix, i al mapa
    no hi haura regio negra. Es un resultat, no un cas particular del codi.
    """
    eq_x, eq_y, centre_estable = punts_dequilibri(p)
    n = len(p.im_K)
    dx, dy, codi, radi2 = [], [], [], []
    for i in range(n):
        if not math.isfinite(eq_x[i]):
            continue  # imant apagat
        dx.append(eq_x[i])
        dy.append(eq_y[i])
        codi.append(i + 1)  # els imants es numeren des de 1
        radi2.append(criteri.r_captura**2)
    if centre_estable:
        dx.append(eq_x[n])
        dy.append(eq_y[n])
        codi.append(CENTRE)
        radi2.append(criteri.r_centre**2)
    return (
        np.array(dx),
        np.array(dy),
        np.array(codi, dtype=np.int64),
        np.array(radi2),
        centre_estable,
    )


@njit(cache=True)
def classifica(x, y, dest_x, dest_y, dest_codi, dest_radi2):
    """Dins de quina rodona de desti es l'estat, o NO_CLASSIFICAT.

    Nomes mira la POSICIO. Que la massa s'hi hagi quedat o nomes hi passi ho
    decideix el temps que hi aguanta, no aquesta funcio.
    """
    for i in range(dest_x.shape[0]):
        ex = x - dest_x[i]
        ey = y - dest_y[i]
        if ex * ex + ey * ey < dest_radi2[i]:
            return dest_codi[i]
    return NO_CLASSIFICAT


def integra_fins_a_classificar(p: Parametres, estat0, criteri: Criteri, dt: float = DT_PER_DEFECTE):
    """Integra fins que la trajectoria te desenllac. Retorna (codi, temps, t, estats).

    SORTIDA ANTICIPADA: es para en el moment que queda classificada. Es el que
    fara viable el mapa; sense aixo cada pixel costaria el temps maxim sencer.

    El temps que es retorna es el de l'ENTRADA a la rodona definitiva, no el de
    la confirmacio: els t_confirma segons de mirar-la-hi no formen part del
    temps de caiguda.

    Aquesta versio GUARDA tota la trajectoria, perque es per mirar-se un
    llancament. La del mapa (bloc 7) nomes es quedara el desenllac.
    """
    x, y, vx, vy = (float(v) for v in estat0)
    if not bool(dins_de_labast(x, y, p)):
        raise ValueError(
            f"El punt ({x:.4f}, {y:.4f}) es a {math.hypot(x, y):.4f} m del centre i el fil "
            f"nomes fa {p.L:.4f} m: aquest punt NO EXISTEIX."
        )

    dest_x, dest_y, dest_codi, dest_radi2, _ = destins(p, criteri)
    passos = int(round(criteri.t_max / dt))
    passos_confirma = int(round(criteri.t_confirma / dt))
    passos_confirma_centre = int(round(criteri.t_confirma_centre / dt))

    t = np.arange(passos + 1, dtype=float) * dt
    estats = np.empty((4, passos + 1), dtype=float)
    estats[:, 0] = (x, y, vx, vy)

    L, g, b, n, d0 = p.L, p.g, p.b, p.n, p.d0
    im_x, im_y, im_K = p.im_x, p.im_y, p.im_K

    candidat = NO_CLASSIFICAT
    seguits = 0
    k_entrada = 0

    for k in range(passos):
        x, y, vx, vy = pas_rk4(x, y, vx, vy, dt, L, g, b, n, d0, im_x, im_y, im_K)
        if x * x + y * y >= L * L:
            raise RuntimeError(
                f"A t = {t[k+1]:.3f} s la massa ha sortit de l'abast del fil "
                f"(r = {math.hypot(x, y):.4f} m > L = {L:.4f} m)."
            )
        estats[:, k + 1] = (x, y, vx, vy)

        codi = classifica(x, y, dest_x, dest_y, dest_codi, dest_radi2)
        if codi == NO_CLASSIFICAT or codi != candidat:
            candidat = codi
            seguits = 1 if codi != NO_CLASSIFICAT else 0
            k_entrada = k + 1
        else:
            seguits += 1
            if seguits >= (passos_confirma_centre if codi == CENTRE else passos_confirma):
                return int(codi), float(t[k_entrada]), t[: k + 2], estats[:, : k + 2]

    return NO_CLASSIFICAT, float(criteri.t_max), t, estats


def nom_del_desti(codi, noms) -> str:
    """Text llegible del codi de desti."""
    if codi == NO_CLASSIFICAT:
        return "NO CONVERGIT"
    if codi == CENTRE:
        return "el centre"
    return f"imant {noms[codi - 1]}" if noms else f"imant {codi}"


def desenllac(p: Parametres, x0: float, y0: float, criteri: Criteri,
              dest_x, dest_y, dest_codi, dest_radi2, dt: float = DT_PER_DEFECTE):
    """Versio comoda de desenllac_nucli, amb Parametres i Criteri."""
    return desenllac_nucli(
        x0, y0, p.L, p.g, p.b, p.n, p.d0, p.im_x, p.im_y, p.im_K,
        dest_x, dest_y, dest_codi, dest_radi2, dt,
        int(round(criteri.t_max / dt)), int(round(criteri.t_confirma / dt)),
        int(round(criteri.t_confirma_centre / dt)))


# ---------------------------------------------------------------------------
# El nucli del mapa
# ---------------------------------------------------------------------------
@njit(cache=True)
def desenllac_nucli(x0, y0, L, g, b, n, d0, im_x, im_y, im_K,
                    dest_x, dest_y, dest_codi, dest_radi2,
                    dt, passos, passos_confirma, passos_confirma_centre):
    """Com acaba una trajectoria, nomes amb numeros. Retorna (codi, temps).

    passos_confirma val per als imants i passos_confirma_centre per al centre:
    el pou del centre es molt mes pla i s'hi ha de mirar mes estona (Criteri).

    Aquesta es la funcio que es crida un cop per pixel del mapa, i per aixo no
    hi entra ni un objecte de Python: nomes escalars i vectors. Es la condicio
    per poder-la compilar. La versio comoda, amb Parametres i Criteri, es
    desenllac().

    SORTIDA ANTICIPADA: es para al moment que la trajectoria queda classificada.
    Es el que fa viable el mapa; sense aixo, cada pixel costaria el temps maxim.
    """
    x, y, vx, vy = x0, y0, 0.0, 0.0
    candidat = NO_CLASSIFICAT
    seguits = 0
    k_entrada = 0

    for k in range(passos):
        x, y, vx, vy = pas_rk4(x, y, vx, vy, dt, L, g, b, n, d0, im_x, im_y, im_K)
        if x * x + y * y >= L * L:
            return NO_CLASSIFICAT, passos * dt

        codi = classifica(x, y, dest_x, dest_y, dest_codi, dest_radi2)
        if codi == NO_CLASSIFICAT or codi != candidat:
            candidat = codi
            seguits = 1 if codi != NO_CLASSIFICAT else 0
            k_entrada = k + 1
        else:
            seguits += 1
            if seguits >= (passos_confirma_centre if codi == CENTRE else passos_confirma):
                return codi, k_entrada * dt

    return NO_CLASSIFICAT, passos * dt


@njit(cache=True, parallel=True)
def graella_nucli(x0, y0, L, g, b, n, d0, im_x, im_y, im_K,
                  dest_x, dest_y, dest_codi, dest_radi2,
                  dt, passos, passos_confirma, passos_confirma_centre, imant, temps):
    """Omple les matrius imant i temps, una fila per fil.

    Cada pixel escriu NOMES a la seva cel.la, o sigui que no hi ha cap conflicte
    entre fils i el prange es segur. Tampoc no hi ha cap suma acumulada, que es
    l'altra cosa que faria que el resultat depengues de l'ordre.
    """
    for j in prange(y0.shape[0]):
        for i in range(x0.shape[0]):
            codi, tc = desenllac_nucli(
                x0[i], y0[j], L, g, b, n, d0, im_x, im_y, im_K,
                dest_x, dest_y, dest_codi, dest_radi2, dt, passos, passos_confirma,
                passos_confirma_centre)
            imant[j, i] = codi
            temps[j, i] = tc
