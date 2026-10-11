import pytest
from ase import Atoms

from mace_orca.data.keys import CYCLE_KEY, FILE_KEY
from mace_orca.data.split import check_disjoint, group_frames, metal_of, split_groups


def make_frames():
    frames = []
    for metal in ("Co", "Fe", "Zn"):
        for mol in range(10):
            for cycle in (2, 0, 1):  # порядок шагов перемешан намеренно
                a = Atoms(f"{metal}O", positions=[[0, 0, 0], [0, 0, 2]])
                a.info[FILE_KEY] = f"{metal}_{mol}.out"
                a.info[CYCLE_KEY] = cycle
                frames.append(a)
    return frames


def test_split_is_by_molecule_and_stratified():
    groups = group_frames(make_frames())
    assert [a.info[CYCLE_KEY] for a in groups["Co_0.out"]] == [0, 1, 2]
    parts = split_groups(groups, 0.1, 0.1, seed=0)
    check_disjoint(parts)
    assert sorted(k for p in parts.values() for k in p) == sorted(groups)
    for subset in ("valid", "test"):
        assert {metal_of(groups[k][0]) for k in parts[subset]} == {"Co", "Fe", "Zn"}


def test_split_is_deterministic():
    groups = group_frames(make_frames())
    assert split_groups(groups, 0.1, 0.1, seed=1) == split_groups(groups, 0.1, 0.1, seed=1)


def test_frames_without_molecule_key_rejected():
    with pytest.raises(ValueError, match=FILE_KEY):
        group_frames([Atoms("H")])


def test_disjoint_check():
    with pytest.raises(ValueError):
        check_disjoint({"train": ["a"], "valid": ["a"], "test": []})
