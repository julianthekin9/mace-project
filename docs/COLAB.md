# Расчёты на Colab

Colab CLI (`colab`) установлен в WSL (Ubuntu, пользователь `julian`). Из Windows все команды идут через `wsl bash -l` (login shell, иначе `colab: command not found`). Сессия называется `mace`.

## Сессия
- VM создаёт и останавливает пользователь, потому что это тратит compute units: `colab new -s mace --gpu T4`, `colab drivemount -s mace`, `colab stop -s mace`.
- `drivemount` интерактивный (вход в Google). Ссылку или код показать пользователю; токены и пароли не вводить.
- Статус: `wsl bash -lc "colab sessions && colab usage"`, `wsl bash -lc "colab status -s mace"`.

## Как выполнять команды на VM
`ssh colab-mace` не работает. Консоль VM запускается внутри tmux в WSL (сокет по умолчанию):

```bash
wsl bash -lc "tmux new -d -s mace && tmux send-keys -t mace:0 'colab console -s mace' Enter"
wsl bash -lc "tmux send-keys -t mace:0 'nvidia-smi; ls /content/drive/MyDrive/MACE' Enter; sleep 3; tmux capture-pane -p -J -t mace:0 -S -200"
```

Файлы удобнее смотреть и забирать через CLI, чем читать вывод консоли:

```bash
wsl bash -l -s <<'EOF'
colab ls -s mace /content/drive/MyDrive/MACE/runs
colab download -s mace /content/drive/MyDrive/MACE/runs/<run>/logs/<run>_run-42.log ./report/raw/<run>_run-42.log
EOF
```

## Где что лежит
- код: `/content/repo` (git-клон, `pip install -e .`). Код на VM только из git: сначала push с Windows, потом `git pull` на VM;
- `foundation-omol.model`: `/content/drive/MyDrive/MACE/foundation-omol.model`;
- результаты новых запусков: `/content/drive/MyDrive/MACE/runs/<name>` (`--work_dir`). Только они переживают перезапуск рантайма: `/content` при этом сбрасывается;
- старые прогоны: `MACE/{results,logs,checkpoints,models}` (донавчання 1) и `MACE/*_2_ft` (донавчання 2). Ничего в `/content/drive` не удалять и не перезаписывать.

## Обучение
Подготовка VM (в консоли `mace:0`):

```bash
test -d /content/repo || git clone -q https://github.com/julianthekin9/mace-project.git /content/repo
cd /content/repo && git pull -q && pip install -q -e . && git rev-parse HEAD
mace-orca-split    # data/{train,valid,test}.xyz из data/dataset.extxyz, сплит по молекулам
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

Запуск: в фоне через `nohup`, чтобы обрыв консоли не убил процесс. `cuequivariance` не включать.

```bash
RUN=ft_molsplit_ef1-100; D=/content/drive/MyDrive/MACE; mkdir -p $D/runs/$RUN
nohup mace_run_train --config configs/$RUN.yaml --foundation_model $D/foundation-omol.model \
  --work_dir $D/runs/$RUN >> $D/runs/$RUN/nohup.log 2>&1 &
```

- Продолжить после перезапуска VM: та же команда плюс `--restart_latest`.
- Прогресс (не чаще раза в несколько минут): `tail -n 5 $D/runs/$RUN/logs/${RUN}_run-42.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`.
- Не запускать два обучения на одной GPU. Когда обучение закончилось, сказать пользователю, чтобы он остановил VM.

Оценка по кадрам и металлам:

```bash
mace-orca-eval --model $D/runs/$RUN/${RUN}.model --input data/test.xyz --out-csv $D/runs/$RUN/test_frames.csv
```
