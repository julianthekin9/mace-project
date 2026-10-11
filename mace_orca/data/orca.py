"""Разбор ORCA-логов (оптимизация геометрии / single point с градиентом) в кадры ASE.

Каждый кадр = (координаты, энергия, градиент), взятые из ОДНОГО и того же шага.
Берутся ВСЕ шаги оптимизации (промежуточные и финальный), у которых есть полный
набор: координаты -> FINAL SINGLE POINT ENERGY -> CARTESIAN GRADIENT.

Единицы на выходе: энергия в эВ, силы в эВ/Å (сила = −градиент).
Заряд и мультиплетность читаются из лога; если их найти нельзя, файл пропускается
(никаких значений по умолчанию).
"""
import os
import re
from collections import Counter

import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree

from .keys import CHARGE_KEY, CYCLE_KEY, ENERGY_KEY, FILE_KEY, FORCES_KEY, MULT_KEY

GRAD_TO_FORCE = -Hartree / Bohr  # Eh/Bohr (градиент) -> эВ/Å (сила)

FLOAT = r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
RE_COORD = re.compile(rf"^\s*([A-Z][a-z]?)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
RE_GRAD = re.compile(rf"^\s*\d+\s+([A-Z][a-z]?)\s*:\s*({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
RE_ENERGY = re.compile(rf"^FINAL SINGLE POINT ENERGY\s+({FLOAT})")
RE_CHARGE = re.compile(r"Total Charge\s+Charge\s+\.\.\.\.\s+(-?\d+)")
RE_MULT = re.compile(r"Multiplicity\s+Mult\s+\.\.\.\.\s+(\d+)")
RE_NEL = re.compile(r"Number of Electrons\s+NEL\s+\.\.\.\.\s+(\d+)")
# служебные файлы, которые ORCA сама создаёт рядом с основным расчётом
# (например Name_atom53.out: подбор атомной плотности для I, ROHF/def2-SVP)
RE_AUX_NAME = re.compile(r"_atom\d+\.out$")


def is_aux_file(filename):
    return bool(RE_AUX_NAME.search(filename))


def read_block(lines, i, rx):
    """Читает блок строк, подходящих под rx, после заголовка на строке i."""
    j = i + 1
    skipped = 0
    while j < len(lines) and skipped < 3 and (
        not lines[j].strip() or set(lines[j].strip()) <= {"-"}
    ):
        j += 1
        skipped += 1
    rows = []
    while j < len(lines):
        m = rx.match(lines[j])
        if not m:
            break
        rows.append(m.groups())
        j += 1
    return rows, j


def read_charge_mult(lines):
    """(charge, mult) из заголовка лога. Бросает ValueError, если их нет или они
    несовместимы с числом электронов."""
    charge = mult = nel = None
    for ln in lines:
        if charge is None and (m := RE_CHARGE.search(ln)):
            charge = int(m.group(1))
        if mult is None and (m := RE_MULT.search(ln)):
            mult = int(m.group(1))
        if nel is None and (m := RE_NEL.search(ln)):
            nel = int(m.group(1))
        if charge is not None and mult is not None and nel is not None:
            break
    if charge is None or mult is None:
        raise ValueError("не найдены Total Charge / Multiplicity")
    if nel is not None and ((nel - (mult - 1)) % 2 != 0 or mult - 1 > nel):
        raise ValueError(f"несовместимы число электронов ({nel}) и мультиплетность ({mult})")
    return charge, mult


def parse_orca_file(path):
    """Возвращает (frames, info). Бросает ValueError, если файл непригоден."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    if any("input to generate an atomic fitting density" in ln for ln in lines[:400]):
        raise ValueError("служебный расчёт ORCA (atomic fitting density)")

    charge, mult = read_charge_mult(lines)

    coords = None  # (номер строки, символы, xyz)
    energy = None  # Eh, обязательно после coords
    scf_bad = False
    frames = []
    skipped = Counter()

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if s == "CARTESIAN COORDINATES (ANGSTROEM)":
            rows, j = read_block(lines, i, RE_COORD)
            if rows:
                coords = (
                    i,
                    [r[0] for r in rows],
                    np.array([[float(x) for x in r[1:]] for r in rows]),
                )
                energy = None
                scf_bad = False
            i = max(j, i + 1)
            continue

        m = RE_ENERGY.match(s)
        if m:
            if coords is not None:
                energy = float(m.group(1))
            i += 1
            continue

        if "SCF NOT CONVERGED" in s:
            scf_bad = True
            i += 1
            continue

        if s == "CARTESIAN GRADIENT":
            rows, j = read_block(lines, i, RE_GRAD)
            if rows:
                if coords is None or energy is None:
                    skipped["нет координат или энергии перед градиентом"] += 1
                elif scf_bad:
                    skipped["SCF не сошёлся"] += 1
                elif [r[0] for r in rows] != coords[1]:
                    skipped["атомы градиента не совпали с координатами"] += 1
                else:
                    grad = np.array([[float(x) for x in r[1:]] for r in rows])
                    pos = coords[2]
                    e_ev = energy * Hartree
                    dup = frames and np.allclose(frames[-1].positions, pos, atol=1e-6) \
                        and abs(frames[-1].info[ENERGY_KEY] - e_ev) < 1e-9
                    if dup:
                        skipped["дубликат предыдущего кадра"] += 1
                    else:
                        a = Atoms(symbols=coords[1], positions=pos)
                        a.info[ENERGY_KEY] = e_ev
                        a.arrays[FORCES_KEY] = grad * GRAD_TO_FORCE
                        a.info[CHARGE_KEY] = charge
                        a.info[MULT_KEY] = mult
                        a.info[FILE_KEY] = os.path.basename(path)
                        a.info[CYCLE_KEY] = len(frames)
                        frames.append(a)
                coords = None
                energy = None
            i = max(j, i + 1)
            continue

        i += 1

    return frames, {"charge": charge, "mult": mult, "skipped": skipped}
