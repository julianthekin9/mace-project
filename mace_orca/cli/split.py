"""mace-orca-split: extxyz -> train/valid/test.xyz с разбиением по молекулам.

    mace-orca-split --dataset data/dataset.extxyz --out-dir data

Пишет {out-dir}/{train,valid,test}.xyz и {out-dir}/split.json — список молекул
каждой выборки и сводку по металлам (манифест хранится в git, xyz — нет).
"""
import argparse
import json
from pathlib import Path

from ase.io import read, write

from mace_orca.data.split import SUBSETS, check_disjoint, group_frames, split_groups, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="data/dataset.extxyz")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    groups = group_frames(read(args.dataset, index=":"))
    parts = split_groups(groups, args.valid_frac, args.test_frac, args.seed)
    check_disjoint(parts)
    rows = summary(groups, parts)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        frames = [a for key in parts[subset] for a in groups[key]]
        write(out / f"{subset}.xyz", frames, format="extxyz")
        print(f"{subset}: {len(parts[subset])} молекул, {len(frames)} кадров")
    for r in rows:
        print(f"  {r['subset']:5s} {r['metal']:6s} молекул {r['molecules']:3d}, кадров {r['frames']:5d}")

    manifest = {
        "dataset": Path(args.dataset).name,
        "seed": args.seed,
        "valid_frac": args.valid_frac,
        "test_frac": args.test_frac,
        "summary": rows,
        **parts,
    }
    with open(out / "split.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"манифест: {out / 'split.json'}")


if __name__ == "__main__":
    main()
