"""Geometria d'un arc de pendol vist de costat: on es el pivot i on es la vertical.

Son quatre linies de numpy, pero viuen en un modul propi perque les fan servir
els DOS costats del circuit i en direccions contraries: magnetic/tracker.py les
necessita per CORREGIR les coordenades en importar-les, i magnetic/decrement.py
per COMPROVAR-les despres. Si visquessin a decrement.py, l'importador hauria
d'importar l'analitzador, que es al reves de com ha d'anar.
"""

from __future__ import annotations

import math

import numpy as np


def ajusta_circumferencia(x, y):
    """Centre i radi de la circumferencia que millor passa pels punts.

    Es l'ajust de Kasa: minimitzant el residu de x^2+y^2 - 2*a*x - 2*b*y - c el
    problema es LINEAL en (a, b, c), o sigui que surt d'un sol lstsq i no d'una
    iteracio que pugui no convergir. Amb un arc de mig centenar de graus n'hi ha
    de sobres.

    Retorna (xc, yc, radi, residu rms). El centre diu si l'origen del Tracker es
    al pivot de debo, i el radi diu quina longitud veu la camera, sense dependre
    d'on el vas posar.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    xc, yc, c = sol
    radi = math.sqrt(max(c + xc**2 + yc**2, 0.0))
    residu = np.hypot(x - xc, y - yc) - radi
    return float(xc), float(yc), float(radi), float(np.sqrt(np.mean(residu**2)))


def inclinacio_de_l_arc(x, y, xc, yc) -> float:
    """Quant s'aparta de la vertical l'eix de simetria de l'arc, en graus.

    EL PENDOL ES EL MILLOR PLOM DEL QUADRE. Un pendol lliure oscil.la simetric
    respecte de la vertical -- ho diu la gravetat, no cap conveni -- o sigui que
    la direccio MITJANA de la massa vista des del centre de l'arc APUNTA AVALL.
    Si no ho fa, la vertical de les coordenades no es la vertical de debo.

    Aixo val mes que qualsevol referencia del quadre: una paret pot no ser
    aplomada, una taula pot no ser de nivell i l'ull no distingeix cinc graus
    en una imatge tombada. La massa penjada, si.
    """
    ux, uy = np.asarray(x) - xc, np.asarray(y) - yc
    norma = np.hypot(ux, uy)
    bons = norma > 0
    ux, uy = ux[bons] / norma[bons], uy[bons] / norma[bons]
    return math.degrees(math.atan2(float(np.mean(ux)), -float(np.mean(uy))))


def gira(x, y, graus: float):
    """Gira el nuvol de punts. Positiu porta el +x cap avall, cap al -y."""
    a = math.radians(graus)
    c, s = math.cos(a), math.sin(a)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * c + y * s, -x * s + y * c


def adreca_amb_la_vertical(x, y):
    """Gira les coordenades fins que l'arc quedi simetric respecte de la vertical.

    Retorna (x girada, y girada, graus girats). El gir que fa es exactament
    menys la inclinacio mesurada, o sigui que despres inclinacio_de_l_arc()
    n'ha de tornar zero -- i per aixo aquesta comprovacio deixa de valer per a
    aquestes dades: el que continua valent, i es independent, es que r sigui
    plana i que el residu de la circumferencia sigui petit.
    """
    xc, yc, _, _ = ajusta_circumferencia(x, y)
    graus = inclinacio_de_l_arc(x, y, xc, yc)
    nx, ny = gira(x, y, graus)
    return nx, ny, graus
