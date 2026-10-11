"""Разбиение датасета на train/valid/test по молекулам, а не по кадрам.

Все шаги одной ORCA-оптимизации (один .out = одна молекула) попадают в одну
выборку: иначе соседние почти одинаковые геометрии оказываются и в train, и в
test, и метрики завышены. Внутри каждого 3d-металла молекулы делятся в заданных
долях, чтобы каждый металл был представлен во всех выборках.
"""
import random
from collections import Counter, defaultdict

from .keys import CYCLE_KEY, FILE_KEY, METALS_3D

SUBSETS = ("train", "valid", "test")


def metal_of(atoms):
    """Метка страты: 3d-металлы кадра через '+', или 'none'."""
    present = set(atoms.get_chemical_symbols())
    metals = [m for m in METALS_3D if m in present]
    return "+".join(metals) or "none"


def group_frames(frames):
    """{orca_file: [кадры по возрастанию cycle]}. Кадры без orca_file — ошибка:
    без ключа молекулы честный сплит невозможен."""
    groups = defaultdict(list)
    for i, a in enumerate(frames):
        if FILE_KEY not in a.info:
            raise ValueError(f"кадр {i}: нет info['{FILE_KEY}'] — пересоберите датасет парсером")
        groups[a.info[FILE_KEY]].append(a)
    for fr in groups.values():
        fr.sort(key=lambda a: a.info.get(CYCLE_KEY, 0))
    return dict(groups)


def split_groups(groups, valid_frac, test_frac, seed):
    """{subset: [ключи молекул]}. Стратификация по металлу; в каждой страте из
    трёх и более молекул valid и test получают хотя бы по одной."""
    strata = defaultdict(list)
    for key in sorted(groups):
        strata[metal_of(groups[key][0])].append(key)

    rng = random.Random(seed)
    parts = {s: [] for s in SUBSETS}
    for stratum in sorted(strata):
        keys = strata[stratum]
        rng.shuffle(keys)
        n = len(keys)
        n_valid = round(valid_frac * n)
        n_test = round(test_frac * n)
        if n >= 3:
            n_valid, n_test = max(1, n_valid), max(1, n_test)
        parts["valid"] += keys[:n_valid]
        parts["test"] += keys[n_valid:n_valid + n_test]
        parts["train"] += keys[n_valid + n_test:]
    return {s: sorted(k) for s, k in parts.items()}


def check_disjoint(parts):
    """Бросает ValueError, если одна молекула попала в несколько выборок."""
    seen = Counter(k for keys in parts.values() for k in keys)
    shared = [k for k, c in seen.items() if c > 1]
    if shared:
        raise ValueError(f"молекулы в нескольких выборках: {shared[:5]}")


def summary(groups, parts):
    """Строки {subset, metal, molecules, frames} для печати и манифеста."""
    rows = []
    for subset in SUBSETS:
        by_metal = defaultdict(lambda: [0, 0])
        for key in parts[subset]:
            m = metal_of(groups[key][0])
            by_metal[m][0] += 1
            by_metal[m][1] += len(groups[key])
        for m in sorted(by_metal):
            rows.append({"subset": subset, "metal": m,
                         "molecules": by_metal[m][0], "frames": by_metal[m][1]})
    return rows
