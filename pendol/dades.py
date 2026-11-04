"""Contracte de dades del projecte.

TOTA taula de trajectoria (vingui del simulador o del Tracker) te
exactament aquestes columnes i aquestes unitats:

    t        segons, comencant a 0
    theta1   radians, desembolicats (unwrapped), des de la vertical inferior
    theta2   radians, desembolicats, ABSOLUTS (no relatius al primer brac)
    omega1   rad/s
    omega2   rad/s

Al costat de cada CSV hi ha un JSON amb el mateix nom que guarda les
metadades (origen, parametres fisics, fps, angle nominal de sortida...).
Sense aquest JSON, d'aqui a tres mesos no sabras quin tir es quin.

Gracies a aquest contracte, pendol/comparar.py no ha de saber d'on ve
cada taula: simulacio i realitat entren pel mateix carrer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNES = ["t", "theta1", "theta2", "omega1", "omega2"]


def cami_meta(cami_csv: str | Path) -> Path:
    return Path(cami_csv).with_suffix(".json")


def desa(cami_csv: str | Path, t, theta1, theta2, omega1, omega2, meta: dict | None = None) -> Path:
    """Escriu una trajectoria en el format canonic + el JSON de metadades."""
    cami_csv = Path(cami_csv)
    cami_csv.parent.mkdir(parents=True, exist_ok=True)

    taula = pd.DataFrame(
        {
            "t": np.asarray(t, dtype=float),
            "theta1": np.asarray(theta1, dtype=float),
            "theta2": np.asarray(theta2, dtype=float),
            "omega1": np.asarray(omega1, dtype=float),
            "omega2": np.asarray(omega2, dtype=float),
        }
    )
    # Punt decimal i prou decimals per no perdre informacio al re-llegir.
    taula.to_csv(cami_csv, index=False, float_format="%.10g")

    if meta is not None:
        desa_meta(cami_csv, meta)
    return cami_csv


def desa_meta(cami_csv: str | Path, meta: dict) -> Path:
    """Escriu (o reescriu) nomes el JSON de metadades d'una trajectoria.

    Existeix perque pendol/tracker.py mesura l'amollada sobre el CSV ja desat
    i rellegit, i despres ha d'afegir-la a les metadades sense tornar a
    escriure la taula.
    """
    cami = cami_meta(cami_csv)
    cami.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return cami


def carrega(cami_csv: str | Path) -> tuple[pd.DataFrame, dict]:
    """Llegeix una trajectoria canonica. Retorna (taula, metadades)."""
    cami_csv = Path(cami_csv)
    taula = pd.read_csv(cami_csv)

    falten = [c for c in COLUMNES if c not in taula.columns]
    if falten:
        raise ValueError(
            f"{cami_csv.name} no segueix el contracte de dades: hi falten {falten}.\n"
            f"Columnes trobades: {list(taula.columns)}\n"
            "Si ve del Tracker, passa'l primer per pendol/tracker.py."
        )

    meta_p = cami_meta(cami_csv)
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return taula[COLUMNES].copy(), meta


def carrega_conjunt(directori: str | Path) -> list[pd.DataFrame]:
    """Llegeix totes les trajectories d'un directori (un feix de simulacions)."""
    directori = Path(directori)
    camins = sorted(directori.glob("*.csv"))
    if not camins:
        raise FileNotFoundError(f"No hi ha cap CSV a {directori}")
    return [carrega(c)[0] for c in camins]
