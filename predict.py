"""
Предсказание энергии обученной моделью MACE.

Использование:
    python predict.py --model models/mace_energy_model.model --input data/test.xyz
"""
import argparse

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="путь к обученной .model")
    ap.add_argument("--input", required=True, help="extxyz со структурами для предсказания")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    calc = MACECalculator(model_paths=args.model, device=args.device)

    frames = read(args.input, index=":")
    pred, true = [], []
    for atoms in frames:
        has_ref = atoms.calc is not None
        e_true = atoms.get_potential_energy() if has_ref else None

        atoms.calc = calc
        e_pred = atoms.get_potential_energy()

        pred.append(e_pred)
        if has_ref:
            true.append(e_true)

    pred = np.array(pred)
    print(f"Предсказано структур: {len(pred)}")

    if true:
        true = np.array(true)
        mae = np.abs(pred - true).mean()
        rmse = np.sqrt(((pred - true) ** 2).mean())
        per_atom_mae = np.abs(pred - true) / np.array([len(a) for a in frames])
        print(f"MAE  = {mae:.4f} eV   ({mae * 1000:.2f} meV)")
        print(f"RMSE = {rmse:.4f} eV")
        print(f"MAE/atom = {per_atom_mae.mean() * 1000:.2f} meV/atom")
    else:
        for i, e in enumerate(pred):
            print(f"структура {i}: E_pred = {e:.6f} eV")


if __name__ == "__main__":
    main()
