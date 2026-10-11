"""mace-orca-optimize: оптимизация геометрий MACE-моделью по всему датасету и сравнение с
оптимизацией, уже проведённой в исходных данных (ORCA).

Логика сравнения:
  1. Кадры датасета группируются по молекуле (по умолчанию — по info["orca_file"],
     который проставляет mace-orca-parse; можно указать другой ключ).
  2. Внутри группы кадры сортируются по info["cycle"] (тоже проставляется парсером),
     если его нет — сохраняется порядок из файла.
  3. Стартовая геометрия для ML-оптимизации — первый кадр группы (начало ORCA-
     оптимизации), опорная («истинная») оптимизированная геометрия — последний
     кадр группы (то, чем ORCA-оптимизация закончилась).
  4. ML-оптимизация запускается из стартовой геометрии, результат сравнивается
     с опорной: RMSD геометрии (после выравнивания Кабша), разница по энергии,
     число шагов, сошлась ли оптимизация.

Имена info-полей (заряд, спин, энергия, силы в датасете) и имена, под которыми
их должен видеть калькулятор MACE, задаются раздельно через аргументы — они не
обязаны совпадать.

Примеры:
    # прогнать по всему датасету, ключи по умолчанию (как у mace-orca-parse)
    mace-orca-optimize --model models_ft/mace_omol_finetuned.model \\
        --dataset data_ft/test.xyz --output-csv opt_report.csv

    # свои имена ключей в датасете и/или ожидаемые калькулятором
    mace-orca-optimize --model model.pt --dataset test.xyz \\
        --dataset-charge-key charge --dataset-spin-key multiplicity \\
        --calc-charge-key charge --calc-spin-key spin \\
        --group-key orca_file --output-csv opt_report.csv --save-dir optimized/

    # одна структура, без группировки (старое поведение)
    mace-orca-optimize --model model.pt --dataset molecule.xyz \\
        --index 0 --charge 0 --spin 1 --output optimized.xyz
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

from mace_orca.tools.optimize import (
    OPTIMIZERS, get_charge_spin, kabsch_align, load_groups, model_from_checkpoint, optimize_one,
    rmsd, set_calc_info,
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                     help="путь к готовой обученной MACE .model (взаимоисключимо с --checkpoint)")
    ap.add_argument("--checkpoint", default=None,
                     help="путь к чекпоинту обучения (checkpoints_dir/*.pt, только веса) - "
                          "если указан, модель будет собрана из него перед запуском; "
                          "требует --template-model")
    ap.add_argument("--template-model", default=None,
                     help="файл .model с ТОЙ ЖЕ архитектурой, что и чекпоинт "
                          "(обычно исходная foundation-модель, например foundation-omol.model, "
                          "или ранее экспортированная .model из того же прогона обучения) "
                          "- нужен только вместе с --checkpoint")
    ap.add_argument("--save-converted-model", default=None,
                     help="куда сохранить модель, собранную из --checkpoint "
                          "(по умолчанию - рядом с чекпоинтом, с суффиксом .model)")
    ap.add_argument("--dataset", required=True, help="extxyz с одной или несколькими структурами")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--optimizer", choices=list(OPTIMIZERS), default="gpmin")
    ap.add_argument("--fmax", type=float, default=0.02, help="эВ/Å")
    ap.add_argument("--max-steps", type=int, default=200)

    # режим по всему датасету
    ap.add_argument("--group-key", default="orca_file",
                     help="info-ключ, по которому кадры группируются в 'одну молекулу'")
    ap.add_argument("--cycle-key", default="cycle",
                     help="info-ключ порядкового номера шага внутри группы (для сортировки)")
    ap.add_argument("--output-csv", default=None,
                     help="сводная таблица по всем молекулам (если не задан --index)")
    ap.add_argument("--save-dir", default=None,
                     help="куда сохранить оптимизированные структуры каждой группы (опционально)")

    # режим одной структуры (совместимость со старым скриптом)
    ap.add_argument("--index", type=int, default=None,
                     help="если задан - обработать только один кадр по номеру, без группировки")
    ap.add_argument("--output", default=None, help="файл для --index режима")

    # имена ключей заряда/спина/энергии/сил В ДАТАСЕТЕ
    ap.add_argument("--dataset-charge-key", default="charge")
    ap.add_argument("--dataset-spin-key", default="multiplicity")
    ap.add_argument("--dataset-energy-key", default="REF_energy")
    ap.add_argument("--dataset-forces-key", default="REF_forces")

    # имена ключей, под которыми заряд/спин должен видеть КАЛЬКУЛЯТОР MACE
    # !! проверьте под свою версию mace.calculators.MACECalculator, если не уверены !!
    ap.add_argument("--calc-charge-key", default="charge")
    ap.add_argument("--calc-spin-key", default="spin")

    ap.add_argument("--charge", type=int, default=None, help="переопределить заряд для всех структур")
    ap.add_argument("--spin", type=int, default=None, help="переопределить спин для всех структур")
    args = ap.parse_args(argv)

    if bool(args.model) == bool(args.checkpoint):
        sys.exit("Укажите ровно один из флагов: --model (готовая модель) "
                 "или --checkpoint (собрать модель из чекпоинта)")

    if args.checkpoint:
        if not args.template_model:
            sys.exit("--checkpoint требует --template-model (файл с той же архитектурой, "
                     "например исходную foundation-модель)")
        save_to = Path(args.save_converted_model) if args.save_converted_model \
            else Path(args.checkpoint).with_suffix(".model")
        model_path = model_from_checkpoint(
            args.checkpoint, args.template_model, save_to, args.device)
    else:
        model_path = args.model

    from mace.calculators import MACECalculator
    calc = MACECalculator(model_paths=model_path, device=args.device)
    optimizer_cls = OPTIMIZERS[args.optimizer]

    # ---- режим одной структуры ---------------------------------------- #
    if args.index is not None:
        if not args.output:
            sys.exit("--output обязателен в режиме --index")
        atoms = read(args.dataset, index=args.index)
        charge, spin = get_charge_spin(
            atoms, args.dataset_charge_key, args.dataset_spin_key, args.charge, args.spin)
        set_calc_info(atoms, charge, spin, args.calc_charge_key, args.calc_spin_key)
        print(f"charge={charge}, spin={spin}")
        res = optimize_one(atoms, calc, optimizer_cls, args.fmax, args.max_steps, "opt.log")
        print(f"Начальная энергия:  {res['e_start']:.6f} эВ, макс. сила: {res['fmax_start']:.4f} эВ/А")
        print(f"Итоговая энергия:   {res['e_end']:.6f} эВ, макс. сила: {res['fmax_end']:.4f} эВ/А")
        print(f"Шагов: {res['n_steps']}, сошлась: {res['converged']}")
        write(args.output, res["atoms"])
        print(f"Сохранено в {args.output}")
        return

    # ---- режим всего датасета, сравнение с ORCA-оптимизацией ---------- #
    groups = load_groups(args.dataset, args.group_key, args.cycle_key)
    print(f"Групп (молекул) в датасете: {len(groups)}")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, frames in groups.items():
        if len(frames) < 2:
            print(f"[пропуск] {name}: в группе меньше 2 кадров, сравнивать не с чем")
            continue

        start, ref_final = frames[0], frames[-1]
        try:
            charge, spin = get_charge_spin(
                start, args.dataset_charge_key, args.dataset_spin_key, args.charge, args.spin)
        except ValueError as e:
            print(f"[пропуск] {name}: {e}")
            continue

        set_calc_info(start, charge, spin, args.calc_charge_key, args.calc_spin_key)

        try:
            res = optimize_one(
                start, calc, optimizer_cls, args.fmax, args.max_steps,
                logfile=str(save_dir / f"{name}.optlog") if save_dir else None,
            )
        except Exception as e:
            print(f"[ошибка] {name}: оптимизация упала: {e}")
            continue

        ml_pos = res["atoms"].get_positions()
        ref_pos = ref_final.get_positions()
        if len(ml_pos) != len(ref_pos):
            print(f"[пропуск] {name}: число атомов не совпадает с опорной структурой")
            continue

        aligned = kabsch_align(ml_pos, ref_pos)
        geom_rmsd = rmsd(aligned, ref_pos)

        ref_energy = ref_final.info.get(args.dataset_energy_key)
        e_diff = (res["e_end"] - ref_energy) if ref_energy is not None else None

        row = {
            "group": name,
            "n_atoms": len(start),
            "charge": charge, "spin": spin,
            "n_ref_frames": len(frames),
            "n_opt_steps": res["n_steps"],
            "converged": res["converged"],
            "rmsd_vs_orca_final_A": geom_rmsd,
            "fmax_end_eV_A": res["fmax_end"],
            "e_ml_end_eV": res["e_end"],
            "e_ref_final_eV": ref_energy,
            "e_diff_eV": e_diff,
        }
        rows.append(row)
        print(f"[ok] {name}: {len(start)} атомов, RMSD={geom_rmsd:.4f} А, "
              f"шагов={res['n_steps']}, сошлась={res['converged']}, "
              f"dE={e_diff if e_diff is None else f'{e_diff:.4f}'} эВ")

        if save_dir:
            write(save_dir / f"{name}_ml_opt.xyz", res["atoms"])

    if not rows:
        sys.exit("Не удалось обработать ни одной группы")

    rmsds = np.array([r["rmsd_vs_orca_final_A"] for r in rows])
    conv = np.array([r["converged"] for r in rows])
    print("-" * 60)
    print(f"Обработано молекул: {len(rows)}")
    print(f"RMSD к ORCA-финалу, А: медиана {np.median(rmsds):.4f}, "
          f"среднее {rmsds.mean():.4f}, максимум {rmsds.max():.4f}")
    print(f"Доля сошедшихся оптимизаций: {conv.mean() * 100:.1f}%")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Таблица сохранена в {args.output_csv}")


if __name__ == "__main__":
    main()
