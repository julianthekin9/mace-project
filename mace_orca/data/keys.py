"""Ключи extxyz, общие для парсера, сплита, оценки и конфигов mace_run_train."""

ENERGY_KEY = "REF_energy"  # info, эВ
FORCES_KEY = "REF_forces"  # arrays, эВ/Å (сила = −градиент ORCA)
CHARGE_KEY = "charge"  # info, полный заряд
MULT_KEY = "multiplicity"  # info, мультиплетность 2S+1 (MACE total_spin)
FILE_KEY = "orca_file"  # info, имя исходного .out — ключ молекулы при сплите
CYCLE_KEY = "cycle"  # info, номер шага оптимизации внутри файла

# 3d-металлы в порядке таблицы: по ним стратифицируется сплит
METALS_3D = ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")

# соответствие для MACECalculator(info_keys=...): имена MACE -> наши ключи
CALC_INFO_KEYS = {"total_charge": CHARGE_KEY, "total_spin": MULT_KEY}
