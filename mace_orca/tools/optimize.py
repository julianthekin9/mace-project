"""Оптимизация геометрии MACE-моделью и сравнение с финальной геометрией ORCA."""
from collections import defaultdict

import numpy as np
import torch
from ase.io import read
from ase.optimize import BFGS, FIRE, GPMin

OPTIMIZERS = {"gpmin": GPMin, "bfgs": BFGS, "fire": FIRE}

# ключи, под которыми разные версии MACE могут хранить веса модели внутри
# файла чекпоинта (checkpoints_dir/*.pt) - пробуем по очереди
_CKPT_STATE_DICT_KEYS = ("model", "model_state_dict", "state_dict")


def model_from_checkpoint(checkpoint_path, template_model_path, save_to, device):
    """Собирает рабочую .model из чекпоинта (только веса) + модели-шаблона
    (та же архитектура, например исходная foundation-модель или ранее
    экспортированная .model из того же прогона обучения).

    Идея взята из официального обсуждения ACEsuit/mace #564: чекпоинт не
    содержит архитектуру, поэтому веса из него грузятся в уже готовый объект
    модели той же архитектуры, и результат сохраняется целиком как новая
    .model.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(
            f"{checkpoint_path} не похож на чекпоинт (это уже готовая модель?) "
            "- передайте его через --model, а не --checkpoint"
        )

    state_dict = None
    used_key = None
    for key in _CKPT_STATE_DICT_KEYS:
        if key in ckpt:
            state_dict = ckpt[key]
            used_key = key
            break
    if state_dict is None:
        raise SystemExit(
            f"В чекпоинте {checkpoint_path} не найден ни один из ожидаемых ключей "
            f"{_CKPT_STATE_DICT_KEYS}; есть ключи: {list(ckpt.keys())}. "
            "Формат чекпоинта отличается от ожидаемого - откройте issue/структуру вручную."
        )
    # на случай, если под ключом лежит не сам state_dict, а объект модели
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()

    template = torch.load(template_model_path, map_location=device, weights_only=False)
    if not hasattr(template, "load_state_dict"):
        raise SystemExit(
            f"{template_model_path}, похоже, тоже не готовая модель "
            "(нет load_state_dict) - нужен .model файл с той же архитектурой"
        )

    missing, unexpected = template.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[внимание] при загрузке весов из чекпоинта (ключ '{used_key}') "
            f"в шаблон {template_model_path}:\n"
            f"  отсутствуют в чекпоинте: {list(missing)}\n"
            f"  лишние в чекпоинте: {list(unexpected)}\n"
            "  если списки не пустые, шаблон архитектурно не совпадает с "
            "чекпоинтом - результат может быть неверным."
        )

    template.to(device)
    torch.save(template, save_to)
    print(f"Модель собрана из чекпоинта и сохранена в {save_to} (эпоха/шаг в ckpt: "
          f"{ckpt.get('epoch', '?')})")
    return str(save_to)


# --------------------------------------------------------------------------- #
# геометрия
# --------------------------------------------------------------------------- #
def kabsch_align(mobile, target):
    """Возвращает mobile, совмещённый с target оптимальным поворотом+сдвигом
    (Kabsch). Оба массива — (N, 3), порядок атомов должен совпадать."""
    mc = mobile - mobile.mean(axis=0)
    tc = target - target.mean(axis=0)
    h = mc.T @ tc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return (r @ mc.T).T + target.mean(axis=0)


def rmsd(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


# --------------------------------------------------------------------------- #
# датасет
# --------------------------------------------------------------------------- #
def load_groups(path, group_key, cycle_key):
    """Читает extxyz и группирует кадры по info[group_key].
    Если group_key нет ни у одного кадра — каждый кадр своя группа (по индексу)."""
    frames = read(path, index=":")
    if not frames:
        raise SystemExit(f"В {path} не найдено ни одной структуры")

    groups = defaultdict(list)
    have_key = any(group_key in a.info for a in frames)
    for i, a in enumerate(frames):
        key = a.info.get(group_key, f"__frame_{i}") if have_key else f"__frame_{i}"
        groups[key].append(a)

    for key, fr in groups.items():
        if all(cycle_key in a.info for a in fr):
            fr.sort(key=lambda a: a.info[cycle_key])
    return groups


def set_calc_info(atoms, charge, spin, calc_charge_key, calc_spin_key):
    atoms.info[calc_charge_key] = charge
    atoms.info[calc_spin_key] = spin


def get_charge_spin(atoms, charge_key, spin_key, charge_arg, spin_arg):
    charge = charge_arg if charge_arg is not None else atoms.info.get(charge_key)
    spin = spin_arg if spin_arg is not None else atoms.info.get(spin_key)
    if charge is None or spin is None:
        raise ValueError(
            f"не найден заряд/спин (искал info['{charge_key}']/info['{spin_key}'], "
            f"есть ключи: {list(atoms.info)}); задайте --charge/--spin явно"
        )
    return charge, spin


# --------------------------------------------------------------------------- #
# оптимизация одной структуры
# --------------------------------------------------------------------------- #
def optimize_one(atoms, calc, optimizer_cls, fmax, max_steps, logfile):
    atoms = atoms.copy()
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    f0 = float(np.abs(atoms.get_forces()).max())

    opt = optimizer_cls(atoms, logfile=logfile)
    converged = opt.run(fmax=fmax, steps=max_steps)
    n_steps = opt.get_number_of_steps()

    e1 = atoms.get_potential_energy()
    f1 = float(np.abs(atoms.get_forces()).max())
    return {
        "atoms": atoms,
        "e_start": e0, "fmax_start": f0,
        "e_end": e1, "fmax_end": f1,
        "n_steps": n_steps, "converged": bool(converged),
    }
