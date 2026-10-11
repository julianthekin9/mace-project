"""mace-orca-eval: ошибки модели на extxyz — сводка, разбивка по металлам и CSV по кадрам.

    mace-orca-eval --model models/model.model --input data/test.xyz --out-csv results/test_frames.csv

Заряд и мультиплетность передаются в модель из info['charge'] / info['multiplicity'].
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from ase.io import read

from mace_orca.data.keys import CALC_INFO_KEYS
from mace_orca.tools.evaluate import aggregate, frame_errors


def fmt(m):
    return (f"кадров {m['frames']:5d} | E: MAE {m['mae_e_meV_atom']:7.2f}, RMSE {m['rmse_e_meV_atom']:7.2f} меВ/атом"
            f" | F: MAE {m['mae_f_meV_A']:7.2f}, RMSE {m['rmse_f_meV_A']:7.2f} меВ/Å")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="обученная .model")
    ap.add_argument("--input", required=True, help="extxyz с REF_energy / REF_forces")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--default-dtype", default="float64")
    ap.add_argument("--out-csv", default=None, help="ошибки по каждому кадру")
    ap.add_argument("--worst", type=int, default=10, help="сколько худших кадров по RMSE сил напечатать")
    args = ap.parse_args(argv)

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=args.model, device=args.device,
                          default_dtype=args.default_dtype, info_keys=CALC_INFO_KEYS)

    rows = frame_errors(read(args.input, index=":"), calc)
    print("всего:   ", fmt(aggregate(rows)))
    by_metal = defaultdict(list)
    for r in rows:
        by_metal[r["metal"]].append(r)
    for metal in sorted(by_metal):
        print(f"{metal:8s} ", fmt(aggregate(by_metal[metal])))

    print(f"\nхудшие {args.worst} кадров по RMSE сил:")
    for r in sorted(rows, key=lambda r: -r["f_rmse_meV_A"])[:args.worst]:
        print(f"  {r['orca_file']} cycle={r['cycle']}: F RMSE {r['f_rmse_meV_A']:.1f} меВ/Å, "
              f"max|ΔF| {r['f_maxerr_meV_A']:.1f} меВ/Å, ΔE {r['de_per_atom_meV']:.2f} меВ/атом, "
              f"max|F_ref| {r['fmax_ref_eV_A']:.2f} эВ/Å")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        fields = [k for k in rows[0] if not k.startswith("_")]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
