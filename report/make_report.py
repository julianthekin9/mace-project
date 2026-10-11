"""Графіки й таблиці для доповіді: донавчання MACE-OMOL-0 (етап 1 + SWA) і продовження (донавчання 2).

Вхід: report/raw/*_train.txt і *.log (скачані з Google Drive, MACE/results*, MACE/logs*).
Вихід: report/figures/*.png|pdf, report/tables/*.csv|md.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
FIG = ROOT / "figures"
TAB = ROOT / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

RUN1 = RAW / "mace_omol_finetuned_run-42_train.txt"
RUN2 = RAW / "mace_omol_finetuned_2_run-42_train.txt"
LOG1 = RAW / "mace_omol_finetuned_run-42.log"
LOG2 = RAW / "mace_omol_finetuned_2_run-42.log"

SWA_START = 90  # перша епоха етапу 2 (SWA) у донавчанні 1
FT2_FROM = 89  # донавчання 2 стартує з чекпойнта епохи 89 донавчання 1
FT2_OFFSET = FT2_FROM + 1  # епоха e донавчання 2 -> FT2_OFFSET + e на спільній осі

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # етап 1, SWA, донавчання 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LBL_STAGE1 = "Донавчання 1 (етап 1)"
LBL_SWA = "Донавчання 1, етап 2 (SWA)"
LBL_FT2 = "Донавчання 2 (продовження)"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

METRICS = ["mae_e_per_atom", "rmse_e_per_atom", "mae_f", "rmse_f", "rel_rmse_f", "loss"]


def read_segments(path):
    """Розбиває train.txt на запуски (кожен починається з eval з epoch=None).

    Повертає список сегментів: {"init": dict|None, "eval": {epoch: dict}, "train": {epoch: [loss]}}.
    """
    segs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["mode"] == "eval" and r["epoch"] is None:
            if segs and not segs[-1]["eval"] and not segs[-1]["train"]:
                segs[-1]["init"] = r  # повторний initial без навчання (впалий запуск)
            else:
                segs.append({"init": r, "eval": {}, "train": {}})
        elif r["mode"] == "eval":
            segs[-1]["eval"][r["epoch"]] = r
        elif r["mode"] == "opt":
            segs[-1]["train"].setdefault(r["epoch"], []).append(r["loss"])
    return [s for s in segs if s["eval"]]


def to_frame(evals, train, offset=0):
    rows = []
    for ep, r in sorted(evals.items()):
        row = {"epoch": ep + offset, **{m: r[m] for m in METRICS}}
        losses = train.get(ep)
        row["train_loss"] = sum(losses) / len(losses) if losses else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    for m in ["mae_e_per_atom", "rmse_e_per_atom", "mae_f", "rmse_f"]:
        df[m] *= 1000  # еВ -> меВ
    return df


# --- донавчання 1: сегменти A (невдалий старт, E0 без корекції), B (0–23), C (рестарт 23–99)
segs1 = read_segments(RUN1)
assert len(segs1) == 3, [len(s["eval"]) for s in segs1]
segA, segB, segC = segs1
ev1 = {**segB["eval"], **segC["eval"]}  # C перезаписує епоху 23 після рестарту
tr1 = {**segB["train"]}
for ep, losses in segC["train"].items():
    tr1[ep] = losses[-1023:] if len(losses) > 1023 else losses
df1 = to_frame(ev1, tr1)
init1 = to_frame({-1: segB["init"]}, {})  # фундаментальна модель (E0 скориговані) до навчання
stage1 = df1[df1.epoch < SWA_START]
swa = df1[df1.epoch >= SWA_START - 1]  # з'єднуємо з епохою 89
dfA = to_frame(segA["eval"], segA["train"])
initA = to_frame({-1: segA["init"]}, {})

# --- донавчання 2
segs2 = read_segments(RUN2)
assert len(segs2) == 1
df2 = to_frame(segs2[0]["eval"], segs2[0]["train"], offset=FT2_OFFSET)
ft2 = pd.concat([stage1[stage1.epoch == FT2_FROM], df2])  # лінія з точки старту

full = pd.concat([init1.assign(етап="до донавчання"),
                  stage1.assign(етап="донавчання 1, етап 1"),
                  df1[df1.epoch >= SWA_START].assign(етап="донавчання 1, SWA"),
                  df2.assign(етап="донавчання 2")])
full.to_csv(TAB / "metrics_per_epoch.csv", index=False, encoding="utf-8-sig", float_format="%.5g")


# ------------------------------------------------------------------ графіки
def plain_log(ax):
    """Логарифмічна вісь Y з підписами звичайними числами (1, 2, 5, 10, 20, ...)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def decorate(ax, ylabel, logy):
    ax.set_xlabel("Епоха (спільна шкала)")
    ax.set_ylabel(ylabel)
    if logy:
        plain_log(ax)
    ax.axvline(FT2_FROM + 0.5, color=INK2, ls="--", lw=1.2)
    ax.set_xlim(-2, FT2_OFFSET + 30)


def vline_label(ax, y_frac=0.97):
    ax.text(FT2_FROM + 1.5, y_frac, "старт донавчання 2\nі SWA (з епохи 89)", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=10, color=INK2)


def curves(ax, col, with_init=True):
    s = pd.concat([init1, stage1]) if with_init else stage1
    ax.plot(s.epoch, s[col], color=C1, label=LBL_STAGE1)
    ax.plot(ft2.epoch, ft2[col], color=C3, label=LBL_FT2, zorder=3)
    ax.plot(swa.epoch, swa[col], color=C2, ls=(0, (4, 2)), label=LBL_SWA, zorder=4)
    if with_init:
        ax.plot(init1.epoch, init1[col], "o", ms=7, color=C1, mec="white", mew=1.5)


def two_panel(cols, ylabels, title, fname, logy=True, xlim=None, with_init=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax, col, yl in zip(axes, cols, ylabels):
        curves(ax, col, with_init)
        decorate(ax, yl, logy)
        if xlim:
            ax.set_xlim(*xlim)
        vline_label(ax)
    axes[0].set_title(title[0])
    axes[1].set_title(title[1])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend([h[0], h[2], h[1]], [l[0], l[2], l[1]], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(FIG / f"{fname}.png")
    fig.savefig(FIG / f"{fname}.pdf")
    plt.close(fig)


two_panel(["mae_e_per_atom", "mae_f"], ["MAE енергії, меВ/атом", "MAE сил, меВ/Å"],
          ["Похибка енергії (валідаційна вибірка)", "Похибка сил (валідаційна вибірка)"],
          "fig1_mae_valid")
two_panel(["rmse_e_per_atom", "rmse_f"], ["RMSE енергії, меВ/атом", "RMSE сил, меВ/Å"],
          ["Похибка енергії (валідаційна вибірка)", "Похибка сил (валідаційна вибірка)"],
          "fig2_rmse_valid")
ZOOM_FROM = 60
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
for ax, col, yl in zip(axes.flat, ["mae_e_per_atom", "rmse_e_per_atom", "mae_f", "rmse_f"],
                       ["MAE енергії, меВ/атом", "RMSE енергії, меВ/атом", "MAE сил, меВ/Å", "RMSE сил, меВ/Å"]):
    curves(ax, col, with_init=False)
    decorate(ax, yl, False)
    ax.set_xlim(ZOOM_FROM - 1, FT2_OFFSET + 30)
    vis = pd.concat([stage1[stage1.epoch >= ZOOM_FROM], swa, ft2])[col]
    pad = (vis.max() - vis.min()) * 0.12
    ax.set_ylim(vis.min() - pad, vis.max() + pad * 2.5)
    vline_label(ax)
    end = ft2.iloc[-1]
    ax.annotate(f"{end[col]:.2f}", (end.epoch, end[col]), xytext=(4, 6), textcoords="offset points",
                fontsize=10, color=INK)
    ax.annotate(f"{stage1.iloc[-1][col]:.2f}", (FT2_FROM, stage1.iloc[-1][col]), xytext=(-6, 8),
                textcoords="offset points", ha="right", fontsize=10, color=INK)
axes[0, 0].set_title("Енергія, MAE")
axes[0, 1].set_title("Енергія, RMSE")
axes[1, 0].set_title("Сили, MAE")
axes[1, 1].set_title("Сили, RMSE")
fig.suptitle(f"Фінальна фаза навчання (епохи {ZOOM_FROM}–{FT2_OFFSET + 29}), валідаційна вибірка",
             fontsize=15, fontweight="bold")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend([h[0], h[2], h[1]], [l[0], l[2], l[1]], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout()
fig.savefig(FIG / "fig4_zoom_final_phase.png")
fig.savefig(FIG / "fig4_zoom_final_phase.pdf")
plt.close(fig)

# --- функція втрат: навчальна (середнє за епоху) і валідаційна
fig, ax = plt.subplots(figsize=(10, 5.5))
for part, c, lbl in [(stage1, C1, LBL_STAGE1), (swa, C2, LBL_SWA), (ft2, C3, LBL_FT2)]:
    ls = (0, (4, 2)) if c == C2 else "-"
    ax.plot(part.epoch, part.loss, color=c, ls=ls, label=f"{lbl}: валідація")
    ax.plot(part.epoch, part.train_loss, color=c, ls=ls, lw=1.2, alpha=0.55, label=f"{lbl}: навчання")
decorate(ax, "Функція втрат (зважена, E:F = 5:10)", True)
vline_label(ax)
ax.set_title("Функція втрат на навчальній і валідаційній вибірках")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=10, ncol=3)
fig.savefig(FIG / "fig3_loss.png")
fig.savefig(FIG / "fig3_loss.pdf")
plt.close(fig)

# --- додатковий: вплив корекції E0 (невдалий старт A vs B)
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
b = pd.concat([init1, df1[df1.epoch <= 11]])
a = pd.concat([initA, dfA])
for ax, col, yl, t in [(axes[0], "rmse_e_per_atom", "RMSE енергії, меВ/атом", "Енергія"),
                       (axes[1], "rmse_f", "RMSE сил, меВ/Å", "Сили")]:
    ax.plot(a.epoch, a[col], color=C2, marker="o", ms=5, label="E0 фундаментальної моделі (без корекції)")
    ax.plot(b.epoch, b[col], color=C1, marker="o", ms=5, label="E0, скориговані під навчальні дані")
    plain_log(ax)
    ax.set_xlabel("Епоха (−1 — до навчання)")
    ax.set_ylabel(yl)
    ax.set_title(f"{t}: вплив оцінки атомних енергій E0")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
fig.savefig(FIG / "fig6_e0_effect.png")
fig.savefig(FIG / "fig6_e0_effect.pdf")
plt.close(fig)


# ------------------------------------------------------------------ таблиці
def error_tables(log):
    """Таблиці Error-table з логу MACE: список {model, set, rmse_e, rmse_f, rel_f}."""
    out, model = [], None
    for line in open(log, encoding="utf-8"):
        if "Saving model to" in line:
            model = "SWA" if "stagetwo" in line else "stage1"
        m = re.match(r"\|\s*(\w+?)_?Default(?:_Default)?\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", line)
        m = m or re.match(r"\|\s*(Default)_Default\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)", line)
        if m:
            name = {"train": "навчальна", "valid": "валідаційна", "Default": "тестова"}[m.group(1)]
            out.append({"model": model, "set": name, "rmse_e": float(m.group(2)),
                        "rmse_f": float(m.group(3)), "rel_f": float(m.group(4))})
    return out


rows = []
names = {("1", "stage1"): "Донавчання 1, етап 1 (епоха 89)",
         ("1", "SWA"): "Донавчання 1, SWA (епоха 99)",
         ("2", "stage1"): "Донавчання 2 (епоха 30 / 119 на спільній шкалі)"}
for run, log in [("1", LOG1), ("2", LOG2)]:
    for r in error_tables(log):
        rows.append({"Модель": names[(run, r["model"])], "Вибірка": r["set"],
                     "RMSE E, меВ/атом": r["rmse_e"], "RMSE F, меВ/Å": r["rmse_f"],
                     "Відносна RMSE F, %": r["rel_f"]})
final = pd.DataFrame(rows)
order = {"навчальна": 0, "валідаційна": 1, "тестова": 2}
final = final.sort_values("Вибірка", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
final.to_csv(TAB / "table_final_errors.csv", index=False, encoding="utf-8-sig")

# ключові точки на валідаційній вибірці
key = [(-1, "Фундаментальна модель (E0 скориговані), до навчання")]
key += [(e, f"Донавчання 1, епоха {e}") for e in (0, 10, 25, 50, 75, 89)]
key += [(99, "Донавчання 1, SWA, епоха 99")]
key += [(FT2_OFFSET + e, f"Донавчання 2, епоха {e + 1} ({FT2_OFFSET + e})") for e in (0, 9, 19, 29)]
src = pd.concat([init1, df1, df2]).drop_duplicates("epoch", keep="first")
# epochs 90–99 є і в SWA, і в донавчанні 2 — беремо явно
prog = []
for ep, name in key:
    d = swa if name.startswith("Донавчання 1, SWA") else (df2 if name.startswith("Донавчання 2") else src)
    r = d[d.epoch == ep].iloc[0]
    prog.append({"Точка": name, "MAE E, меВ/атом": r.mae_e_per_atom, "RMSE E, меВ/атом": r.rmse_e_per_atom,
                 "MAE F, меВ/Å": r.mae_f, "RMSE F, меВ/Å": r.rmse_f})
prog = pd.DataFrame(prog)
prog.to_csv(TAB / "table_valid_progress.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

# зміни між моделями
base = prog.iloc[0]
s1 = prog[prog["Точка"] == "Донавчання 1, епоха 89"].iloc[0]
f2 = prog.iloc[-1]
delta = pd.DataFrame([
    {"Порівняння": "Фундаментальна → донавчання 1 (епоха 89)",
     **{f"Δ {c.split(',')[0]}, %": 100 * (s1[c] - base[c]) / base[c] for c in prog.columns[1:]}},
    {"Порівняння": "Донавчання 1 (епоха 89) → донавчання 2 (епоха 30)",
     **{f"Δ {c.split(',')[0]}, %": 100 * (f2[c] - s1[c]) / s1[c] for c in prog.columns[1:]}},
])
delta.to_csv(TAB / "table_relative_change.csv", index=False, encoding="utf-8-sig", float_format="%.1f")

hp = pd.DataFrame([
    ("Початкова модель", "MACE-OMOL-0 (foundation-omol.model)", "модель донавчання 1, епоха 89"),
    ("Кількість епох", "90 (етап 1) + 10 (SWA)", "30"),
    ("Розмір батчу", "8", "8"),
    ("Швидкість навчання", "1·10⁻⁴; SWA: 1·10⁻⁵", "1·10⁻⁵"),
    ("Weight decay", "0", "1·10⁻⁵"),
    ("Ваги функції втрат E : F", "5 : 10", "5 : 10"),
    ("EMA (decay)", "так (0.999)", "так (0.999)"),
    ("Планувальник", "ReduceLROnPlateau (за замовчуванням)", "ReduceLROnPlateau, patience 10"),
    ("Атомні енергії E0", "оцінені за фундаментальною моделлю", "оцінені за фундаментальною моделлю"),
    ("Точність обчислень", "float64", "float64"),
    ("Тривалість епохи", "≈ 7.7 хв", "≈ 7.8 хв"),
], columns=["Параметр", "Донавчання 1", "Донавчання 2"])
hp.to_csv(TAB / "table_hyperparams.csv", index=False, encoding="utf-8-sig")

ds = pd.DataFrame([("Навчальна", 8184), ("Валідаційна", 1023), ("Тестова", 1023)],
                  columns=["Вибірка", "Кількість конфігурацій"])
ds.to_csv(TAB / "table_dataset.csv", index=False, encoding="utf-8-sig")

with open(TAB / "tables.md", "w", encoding="utf-8") as f:
    for title, df, fmt in [
        ("Таблиця 1. Склад даних (кроки оптимізації геометрії ORCA, 19 елементів, 3d-метали)", ds, None),
        ("Таблиця 2. Параметри донавчання", hp, None),
        ("Таблиця 3. Похибки на валідаційній вибірці в ключових точках навчання", prog, ".2f"),
        ("Таблиця 4. Відносна зміна похибок на валідаційній вибірці", delta, ".1f"),
        ("Таблиця 5. Підсумкові похибки моделей (з логів MACE)", final, ".2f"),
    ]:
        cell = lambda v: format(v, fmt) if fmt and isinstance(v, float) else str(v)
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
        f.write(f"### {title}\n\n" + "\n".join(lines) + "\n\n")

print("OK:", *sorted(p.name for p in FIG.iterdir()), *sorted(p.name for p in TAB.iterdir()))
