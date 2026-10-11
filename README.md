# mace-orca

Дообучение foundation-модели MACE-OMOL-0 на шагах геометрических оптимизаций ORCA для комплексов 3d-металлов (Cr–Zn). Цель — ML-потенциал (энергия и силы), который заменяет ORCA при оптимизации геометрии.

## Установка

```bash
pip install -e ".[dev,report]"
```

## Пайплайн

```bash
mace-orca-parse --input orca_data --out data/dataset.extxyz      # ORCA .out -> extxyz (эВ, эВ/Å, F = −градиент)
mace-orca-split --dataset data/dataset.extxyz --out-dir data     # train/valid/test по молекулам + data/split.json
mace_run_train --config configs/ft_molsplit_ef1-100.yaml         # обучение (на Colab: docs/COLAB.md)
mace-orca-eval --model <run>.model --input data/test.xyz --out-csv test_frames.csv
mace-orca-optimize --model <run>.model --dataset data/test.xyz --output-csv opt_report.csv --save-dir optimized/
pytest tests/
```

## Структура

| Путь | Что там |
|---|---|
| `mace_orca/data/` | `keys.py` (ключи extxyz), `orca.py` (парсер ORCA), `split.py` (сплит по молекулам) |
| `mace_orca/tools/` | `evaluate.py` (ошибки по кадрам), `optimize.py` (ML-оптимизация и сравнение с ORCA) |
| `mace_orca/cli/` | точки входа `mace-orca-*` |
| `configs/` | конфиги `mace_run_train` |
| `data/` | `dataset.extxyz` и `split.json` в git; `train/valid/test.xyz` генерируются |
| `report/` | графики и таблицы к докладу (`make_report.py`) |
| `docs/COLAB.md` | работа с Colab |
| `orca_data/`, `models/`, `checkpoints/`, `runs/` | данные и артефакты, не в git |
