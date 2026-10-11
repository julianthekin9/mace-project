"""mace-orca-parse: папка ORCA-логов -> один extxyz со всеми шагами оптимизаций.

    mace-orca-parse --input orca_data --out data/dataset.extxyz [--max-force 10]

Разбиение на train/valid/test — отдельной командой mace-orca-split.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
from ase.io import write

from mace_orca.data.keys import CHARGE_KEY, FORCES_KEY, MULT_KEY
from mace_orca.data.orca import is_aux_file, parse_orca_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="orca_data", help="папка с *.out")
    ap.add_argument("--out", default="data/dataset.extxyz")
    ap.add_argument("--max-force", type=float, default=0.0,
                    help="отбрасывать кадры с max|F| выше порога, эВ/Å (0 = не фильтровать)")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.input):
        sys.exit(f"Папка {args.input} не найдена")

    by_file = {}
    n_bad, n_aux = 0, 0
    for fn in sorted(os.listdir(args.input)):
        if not fn.endswith(".out"):
            continue
        if is_aux_file(fn):
            n_aux += 1
            continue
        try:
            frames, meta = parse_orca_file(os.path.join(args.input, fn))
        except ValueError as e:
            print(f"[пропуск] {fn}: {e}")
            n_bad += 1
            continue
        if args.max_force > 0:
            keep = [a for a in frames if np.abs(a.arrays[FORCES_KEY]).max() <= args.max_force]
            if len(keep) != len(frames):
                meta["skipped"]["max|F| выше порога"] += len(frames) - len(keep)
            frames = keep
        if not frames:
            print(f"[пусто] {fn}: нет полных кадров {dict(meta['skipped'])}")
            n_bad += 1
            continue
        by_file[fn] = frames
        extra = f" (пропущено: {dict(meta['skipped'])})" if meta["skipped"] else ""
        print(f"[ok] {fn}: {len(frames)} кадров, charge={meta['charge']}, mult={meta['mult']}{extra}")

    all_frames = [a for fr in by_file.values() for a in fr]
    print("-" * 60)
    print(f"файлов с данными: {len(by_file)}, без данных/с ошибкой: {n_bad}, служебных _atomNN: {n_aux}")
    print(f"всего кадров: {len(all_frames)}")
    if not all_frames:
        sys.exit("Нечего сохранять")

    fmax = np.array([np.abs(a.arrays[FORCES_KEY]).max() for a in all_frames])
    print("max|F| по кадрам, эВ/Å: медиана %.3f, 95%% %.3f, максимум %.3f"
          % (np.median(fmax), np.percentile(fmax, 95), fmax.max()))
    print("charge:", dict(Counter(a.info[CHARGE_KEY] for a in all_frames)))
    print("multiplicity:", dict(Counter(a.info[MULT_KEY] for a in all_frames)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write(args.out, all_frames, format="extxyz")
    print(f"записано: {args.out}")


if __name__ == "__main__":
    main()
