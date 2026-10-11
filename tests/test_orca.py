import numpy as np
import pytest
from ase.units import Bohr, Hartree

from mace_orca.data.keys import CHARGE_KEY, CYCLE_KEY, ENERGY_KEY, FILE_KEY, FORCES_KEY, MULT_KEY
from mace_orca.data.orca import is_aux_file, parse_orca_file

HEADER = """
 Total Charge           Charge          ....    1
 Multiplicity           Mult            ....    2
 Number of Electrons    NEL             ....   35
"""


def step(x, energy, gx):
    return f"""
---------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  Cu     {x:.6f}    0.000000    0.000000
  O      2.000000    0.000000    0.000000

FINAL SINGLE POINT ENERGY      {energy:.9f}

------------------
CARTESIAN GRADIENT
------------------

   1   Cu  :   {gx:.9f}    0.000000000    0.000000000
   2   O   :   {-gx:.9f}    0.000000000    0.000000000

Difference to translation invariance:
"""


def write_log(tmp_path, body, name="Cu_test_1.out"):
    p = tmp_path / name
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_units_sign_charge_mult(tmp_path):
    p = write_log(tmp_path, step(0.0, -100.5, 0.01) + step(0.1, -100.6, 0.002))
    frames, meta = parse_orca_file(p)
    assert len(frames) == 2
    a = frames[0]
    assert a.get_chemical_symbols() == ["Cu", "O"]
    assert a.info[ENERGY_KEY] == pytest.approx(-100.5 * Hartree)
    # сила = −градиент, Eh/Bohr -> эВ/Å
    assert a.arrays[FORCES_KEY][0, 0] == pytest.approx(-0.01 * Hartree / Bohr)
    assert np.allclose(a.arrays[FORCES_KEY].sum(axis=0), 0)
    assert (a.info[CHARGE_KEY], a.info[MULT_KEY]) == (1, 2)
    assert a.info[FILE_KEY] == "Cu_test_1.out"
    assert [f.info[CYCLE_KEY] for f in frames] == [0, 1]


def test_skips_unconverged_scf_and_duplicates(tmp_path):
    bad = step(0.2, -100.7, 0.003).replace("FINAL SINGLE", "SCF NOT CONVERGED\nFINAL SINGLE")
    body = step(0.0, -100.5, 0.01) + step(0.0, -100.5, 0.01) + bad
    frames, meta = parse_orca_file(write_log(tmp_path, body))
    assert len(frames) == 1
    assert meta["skipped"]["дубликат предыдущего кадра"] == 1
    assert meta["skipped"]["SCF не сошёлся"] == 1


def test_inconsistent_multiplicity_rejected(tmp_path):
    p = tmp_path / "x.out"
    p.write_text(HEADER.replace("....    2", "....    1") + step(0.0, -1.0, 0.0), encoding="utf-8")
    with pytest.raises(ValueError, match="несовместимы"):
        parse_orca_file(p)


def test_aux_file_name():
    assert is_aux_file("Co_CSD_code_5_atom53.out")
    assert not is_aux_file("Co_CSD_code_5.out")
