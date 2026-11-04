"""Interruptor de compilacio: el mateix codi, compilat o no.

EL PROBLEMA QUE RESOL
---------------------
El mapa s'ha de poder calcular de dues maneres: a poc a poc amb Python, que es
lent pero transparent, i compilat amb Numba, que es rapid. I les dues maneres
han de donar EXACTAMENT el mateix resultat, pixel a pixel.

La manera facil de fer-ho seria escriure el nucli dues vegades, una per a cada
cami. Es tambe la manera de tenir, al cap d'un mes, dos models lleugerament
diferents sense saber-ho.

Per aixo aqui no hi ha cap duplicat. Les funcions del nucli s'escriuen UNA
vegada, decorades amb el @njit d'aquest modul, i aquest decorador es:

  * el @njit de debo de Numba, normalment
  * un decorador que no fa res, si poses MAGNETIC_SENSE_NUMBA=1 a l'entorn

O sigui que la comparacio del bloc 8 no compara dos programes: compara el
MATEIX programa compilat i sense compilar. Si els dos mapes surten identics,
l'unic que ha canviat es el temps de calcul.

    # el cami lent, transparent
    $env:MAGNETIC_SENSE_NUMBA=1
    .venv\\Scripts\\python.exe -m magnetic.mapa --resolucio 120 --nom lent
    Remove-Item Env:MAGNETIC_SENSE_NUMBA

    # el cami rapid
    .venv\\Scripts\\python.exe -m magnetic.mapa --resolucio 120 --nom rapid

IMPORTANT: enlloc no s'hi posa fastmath=True. Amb fastmath el compilador es
pren la llibertat de reordenar les operacions de coma flotant, i llavors els
dos camins ja NO donen bit a bit el mateix. En un sistema caotic, una
diferencia al bit 53 acaba canviant de color un pixel. La comprovacio val mes
que el poc que s'hi guanyaria.
"""

from __future__ import annotations

import os

SENSE_NUMBA = os.environ.get("MAGNETIC_SENSE_NUMBA", "").strip() not in ("", "0")

HI_HA_NUMBA = False
if not SENSE_NUMBA:
    try:
        from numba import njit, prange  # noqa: F401

        HI_HA_NUMBA = True
    except ImportError:
        pass

if not HI_HA_NUMBA:

    def njit(*args, **kwargs):
        """@njit que no fa res: deixa la funcio tal com esta, en Python."""
        if args and callable(args[0]):
            return args[0]
        return lambda funcio: funcio

    prange = range


def com_va() -> str:
    if SENSE_NUMBA:
        return "Python pelat (MAGNETIC_SENSE_NUMBA esta posat): lent i transparent"
    if HI_HA_NUMBA:
        return "compilat amb Numba"
    return "Python pelat (Numba no esta instal.lada): lent"
