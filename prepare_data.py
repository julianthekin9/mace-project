"""
    python prepare_data.py --input dataset.extxyz --outdir data \
        --valid-frac 0.1 --test-frac 0.1 --seed 42
"""
import argparse
import random
from pathlib import Path

from ase.io import read, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="путь к исходному .extxyz")
    ap.add_argument("--outdir", default="data", help="куда сохранить train/valid/test")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    frames = read(args.input, index=":")
    print(f"Прочитано структур: {len(frames)}")

    # sanity-check: у каждой структуры должна быть энергия
    bad = [i for i, a in enumerate(frames) if a.calc is None]
    if bad:
        raise ValueError(f"У {len(bad)} структур нет energy/forces (calc=None), "
                          f"например индексы {bad[:5]}")

    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})
    print(f"Химические элементы в датасете ({len(elements)}): {elements}")

    random.seed(args.seed)
    idx = list(range(len(frames)))
    random.shuffle(idx)

    n_valid = int(len(idx) * args.valid_frac)
    n_test = int(len(idx) * args.test_frac)
    valid_idx = set(idx[:n_valid])
    test_idx = set(idx[n_valid:n_valid + n_test])

    train, valid, test = [], [], []
    for i, a in enumerate(frames):
        if i in valid_idx:
            valid.append(a)
        elif i in test_idx:
            test.append(a)
        else:
            train.append(a)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write(outdir / "train.xyz", train)
    write(outdir / "valid.xyz", valid)
    write(outdir / "test.xyz", test)

    print(f"train: {len(train)}  valid: {len(valid)}  test: {len(test)}")
    print(f"Сохранено в {outdir.resolve()}")
    print("\nСтроку для конфига MACE (E0s считаются автоматически при E0s: average, "
          "но список элементов пригодится для проверки):")
    print(elements)


if __name__ == "__main__":
    main()
