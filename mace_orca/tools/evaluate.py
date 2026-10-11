"""Ошибки модели по кадрам: энергия на атом и силы, против эталона ORCA.

Метрики сил считаются по всем компонентам (как в логах MACE), энергия — на атом.
"""
import numpy as np

from mace_orca.data.keys import CHARGE_KEY, CYCLE_KEY, ENERGY_KEY, FILE_KEY, FORCES_KEY, MULT_KEY
from mace_orca.data.split import metal_of


def frame_errors(frames, calc):
    """Список словарей с ошибками для каждого кадра (эВ -> меВ)."""
    rows = []
    for a in frames:
        ref_e = a.info[ENERGY_KEY]
        ref_f = a.arrays[FORCES_KEY]
        b = a.copy()
        b.calc = calc
        pred_e = b.get_potential_energy()
        df = b.get_forces() - ref_f
        rows.append({
            FILE_KEY: a.info.get(FILE_KEY, ""),
            CYCLE_KEY: a.info.get(CYCLE_KEY, -1),
            "metal": metal_of(a),
            "n_atoms": len(a),
            CHARGE_KEY: a.info.get(CHARGE_KEY),
            MULT_KEY: a.info.get(MULT_KEY),
            "e_ref_eV": ref_e,
            "e_pred_eV": pred_e,
            "de_per_atom_meV": 1000 * (pred_e - ref_e) / len(a),
            "f_mae_meV_A": 1000 * np.abs(df).mean(),
            "f_rmse_meV_A": 1000 * np.sqrt((df ** 2).mean()),
            "f_maxerr_meV_A": 1000 * np.linalg.norm(df, axis=1).max(),
            "fmax_ref_eV_A": np.linalg.norm(ref_f, axis=1).max(),
            "_df": df,
        })
    return rows


def aggregate(rows):
    """MAE/RMSE энергии на атом и сил по всем компонентам для набора кадров."""
    de = np.array([r["de_per_atom_meV"] for r in rows])
    df = np.concatenate([r["_df"].ravel() for r in rows]) * 1000
    return {
        "frames": len(rows),
        "mae_e_meV_atom": np.abs(de).mean(),
        "rmse_e_meV_atom": np.sqrt((de ** 2).mean()),
        "mae_f_meV_A": np.abs(df).mean(),
        "rmse_f_meV_A": np.sqrt((df ** 2).mean()),
    }
