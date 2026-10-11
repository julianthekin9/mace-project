# CLAUDE.md — проект `mace`

## Что делаем

Дообучаем (fine-tuning) foundation-модель **MACE-OMOL-0** (`foundation-omol.model`) на структурах с **3d-металлами**, чтобы модель предсказывала **энергию и силы** для разных положений атомов.

Конечная цель — **оптимизация геометрии через градиенты ML-потенциала** как более быстрая альтернатива геометрической оптимизации в **ORCA**.

Общение по проекту — на русском. Комментарии в коде — на русском. Подписи на графиках для доклада — на украинском.

## Структура

Пакет `mace_orca` (по образцу `syscall-anomaly-detection`, своя модель не пишется — только данные, оценка и применение вокруг `mace_run_train`).

- `mace_orca/data/keys.py` — ключи extxyz и список 3d-металлов; `orca.py` — парсер ORCA (`parse_orca_file`); `split.py` — сплит по молекулам со стратификацией по металлу.
- `mace_orca/tools/evaluate.py` — ошибки по кадрам/агрегаты; `optimize.py` — ML-оптимизация и сравнение с финальной геометрией ORCA.
- `mace_orca/cli/` — точки входа `mace-orca-parse`, `mace-orca-split`, `mace-orca-eval`, `mace-orca-optimize` (`pyproject.toml`).
- `configs/*.yaml` — конфиги `mace_run_train` (ключи = флаги установленной версии MACE). Только ASCII: configargparse читает их в кодировке локали, кириллица на Windows (cp1251) ломает разбор.
- `data/dataset.extxyz` + `data/split.json` — в git; `data/{train,valid,test}.xyz` генерирует `mace-orca-split` (в `.gitignore`).
- `report/` — материалы к докладу (`make_report.py`, исходники в `report/raw/`).
- `tests/` — парсер (единицы, знак сил, заряд/мультиплетность, пропуски) и сплит (без утечки, стратификация).
- `orca_data/`, `models/`, `checkpoints/`, `runs/` — данные и артефакты, не в git.

## Пайплайн

```bash
pip install -e ".[dev,report]"
mace-orca-parse --input orca_data --out data/dataset.extxyz
mace-orca-split --dataset data/dataset.extxyz --out-dir data
mace_run_train --config configs/ft_molsplit_ef1-100.yaml     # на Colab — docs/COLAB.md
mace-orca-eval --model <run>.model --input data/test.xyz --out-csv test_frames.csv
mace-orca-optimize --model <run>.model --dataset data/test.xyz --output-csv opt_report.csv
pytest tests/
```

## Формат extxyz

| Что | Ключ |
|---|---|
| Энергия | `REF_energy` (в `info`) |
| Силы | `REF_forces` (в `arrays`) |
| Заряд | `charge` (в `info`) |
| Спин (мультиплетность) | `multiplicity` (в `info`) |
| Молекула (ключ сплита) | `orca_file` (в `info`) |
| Шаг оптимизации | `cycle` (в `info`) |

Флаги `mace_run_train`: `energy_key: REF_energy`, `forces_key: REF_forces`, `total_charge_key: charge`, `total_spin_key: multiplicity`. MACE понимает `total_spin` как мультиплетность 2S+1. `MACECalculator` по умолчанию читает `info["charge"]` / `info["spin"]`, поэтому в оценке передаётся `info_keys=CALC_INFO_KEYS`.

**Единицы:** MACE ожидает эВ и эВ/Å. ORCA выдаёт Hartree и Hartree/Bohr, а в `.out` печатается *градиент*, т.е. силы = −градиент. Проверяй, что парсер делает конвертацию и смену знака.

## Сплит

Только по молекулам (`orca_file`), стратификация по 3d-металлу: `mace-orca-split`, seed 42 → 238/30/30 молекул, 8072/1165/993 кадров. Старый сплит по кадрам (до 2026-10) давал утечку: почти каждая молекула была во всех трёх выборках. Метрики донавчаний 1 и 2 на нём завышены, а их модели уже видели молекулы нового test — сравнивать на новом сплите можно только модели, обученные с нуля от `foundation-omol.model`.

## Конфиги

- `configs/ft_molsplit_ef5-10.yaml` / `ft_molsplit_ef1-100.yaml` — с нуля от MACE-OMOL-0 на новом сплите; отличаются только весами E:F. ef5-10 — база (веса донавчання 1), ef1-100 — упор на силы.
- `configs/ft2_continue.yaml` — история: донавчання 2.
- Донавчання 1 (по логу): `lr` 1e-4, `batch_size` 8, E:F = 5:10, `weight_decay` 0, EMA 0.999, SWA с 90-й эпохи (`lr` 1e-5), 100 эпох, `E0s: estimated`. Батчи 32 и 16 не запускались.
- Донавчання 2: старт с эпохи 89 донавчання 1, 30 эпох, `lr` 1e-5, `weight_decay` 1e-5, E:F = 5:10, без SWA.

## Окружение

- Python 3.13, PyTorch, mace-torch 0.3.16 (локально в `.venv`)
- `cuequivariance` **не установлен** — не включать `--enable_cueq`
- `mace_run_train --help` в консоли Windows падает на cp1251: запускать с `PYTHONIOENCODING=utf-8`.

## Известные проблемы

- `mace_run_train` падает с `unrecognized argument --swa_start=90`. Такого флага нет: в MACE это `--swa` + `--start_swa=90` (в новых версиях SWA переименован во второй этап — `--stage_two` / `--start_stage_two`). Сверяйся с `mace_run_train --help` установленной версии, а не с памятью.

## Подключение к Colab

Подробно — `docs/COLAB.md`. Коротко: Colab CLI в WSL (`wsl bash -l`), сессия `mace`, `ssh` не работает — `colab console -s mace` внутри tmux `mace` в WSL (поднимать самому), файлы — `colab ls` / `colab download`. VM и `drivemount` — делает пользователь; ссылки/коды авторизации показать ему, токены не вводить.

Drive: `/content/drive/MyDrive/MACE`. Старые прогоны: `results/`, `logs/`, `checkpoints/`, `models/` — донавчання 1; `*_2_ft/` — донавчання 2; `optimized/` — ML-оптимизации. Новые — `runs/<name>`. Ничего в `/content/drive` не удалять и не перезаписывать; `/content` сбрасывается при перезапуске рантайма.

## Правила для работы

- Не трогай исходные `orca_data/*.out` — это первичные данные.
- Перед изменением флагов обучения проверяй `mace_run_train --help` текущей версии MACE.
- После изменений в парсере — `pytest tests/` и проверить на одном логе: число атомов, единицы, знак сил, заряд и мультиплетность.
