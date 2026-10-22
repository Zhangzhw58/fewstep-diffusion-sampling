from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def charts() -> None:
    fid_rows = read_csv(METRICS / "fid.csv")
    timing_rows = read_csv(METRICS / "timing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for sched, label, marker in [("default", "Default linspace", "o"),
                                  ("logsnr_power", "Uniform log-SNR (selected)", "s")]:
        vals = sorted((r for r in fid_rows if r["schedule"] == sched), key=lambda r: int(r["n_steps"]))
        axes[0].plot([int(r["n_steps"]) for r in vals], [float(r["fid_clean"]) for r in vals],
                     marker=marker, linewidth=2, label=label)
        vals = sorted((r for r in timing_rows if r["schedule"] == sched), key=lambda r: int(r["n_steps"]))
        axes[1].plot([int(r["n_steps"]) for r in vals], [float(r["seconds_per_image"]) * 1000 for r in vals],
                     marker=marker, linewidth=2, label=label)
    axes[0].set(title="Clean-FID (lower is better)", xlabel="NFE", ylabel="FID")
    axes[1].set(title="Sampling latency", xlabel="NFE", ylabel="Milliseconds per image")
    for ax in axes:
        ax.set_xticks([5, 10, 20])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("CIFAR-10 · DPM-Solver++ · 5,000 samples per setting")
    fig.savefig(OUT / "metrics" / "dpmpp_results.png", dpi=180)
    plt.close(fig)


def contact_sheet() -> None:
    configs = [(5, "default"), (5, "logsnr_power"), (10, "default"),
               (10, "logsnr_power"), (20, "default"), (20, "logsnr_power")]
    scale, n_rows = 5, 8
    cell = 32 * scale
    left = 190
    top = 42
    sheet = Image.new("RGB", (left + len(configs) * cell, top + n_rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (n, schedule) in enumerate(configs):
        label = f"{n} steps\n" + ("default" if schedule == "default" else "log-SNR")
        draw.multiline_text((left + j * cell + 3, 2), label, fill="black", spacing=1)
        folder = OUT / "samples" / f"dpmpp_n{n}_{schedule}"
        for i in range(n_rows):
            seed = 1000 + i
            im = Image.open(folder / f"seed_{seed:05d}.png").convert("RGB").resize((cell, cell), Image.Resampling.NEAREST)
            sheet.paste(im, (left + j * cell, top + i * cell))
            if j == 0:
                draw.text((4, top + i * cell + 4), f"seed {seed}", fill="black")
    sheet.save(OUT / "samples" / "contact_sheet_48.png")


def summary() -> None:
    fid_rows = read_csv(METRICS / "fid.csv")
    timing_rows = read_csv(METRICS / "timing.csv")
    selected = json.loads((METRICS / "selected_gamma.json").read_text(encoding="utf-8"))
    lines = [
        "# DPM-Solver++ CIFAR-10 实验结果小结",
        "",
        "## 实验设置",
        "",
        "- 模型：`google/ddpm-cifar10-32`，固定 revision `267b167dc01f0e4e61923ea244e8b988f84deb80`，32×32 无条件 CIFAR-10 生成。",
        "- 设备：NVIDIA GeForce RTX 3070 Laptop GPU（8 GB）；batch size 32。",
        "- Solver：二阶 DPM-Solver++；NFE 为 5、10、20。",
        "- 配对设置：所有对比共用相同的初始噪声 seed。时间步选择只用 16 个验证 seed；生成评估使用另 5,000 个 seed（1000–5999）。",
        "- 自定义策略：在 log-SNR 空间对时间点做 power warp，在 gamma ∈ {0.65, 0.8, 1.0, 1.2, 1.4} 中以验证集相对 100 步 DPM-Solver++ 输出的像素 MSE 选择 gamma。该代理目标不等于 FID，也不保证测试指标必然改善。",
        "- FID：clean-fid 的 CIFAR-10 train 32×32 参考统计；每个配置恰好 5,000 张生成图。这里与参考统计比较，不是拿生成图与 CIFAR-10 test 图片逐张比较。",
        "",
        "## 结果",
        "",
        "| NFE | 时间步 | gamma | Clean-FID ↓ | ms/张 ↓ |",
        "|---:|---|---:|---:|---:|",
    ]
    for f in sorted(fid_rows, key=lambda r: (int(r["n_steps"]), r["schedule"])):
        t = next(r for r in timing_rows if r["n_steps"] == f["n_steps"] and r["schedule"] == f["schedule"])
        gamma = selected[f["n_steps"]]["gamma"] if f["schedule"] == "logsnr_power" else "—"
        name = "默认 linspace" if f["schedule"] == "default" else "log-SNR power"
        lines.append(f"| {f['n_steps']} | {name} | {gamma} | {float(f['fid_clean']):.2f} | {float(t['seconds_per_image'])*1000:.2f} |")
    lines += [
        "",
        "## 解读与边界",
        "",
        "- 验证集在 5/10/20 NFE 都选出 gamma=1.0（均匀 log-SNR 网格）。测试 FID 分别由默认时间步的 "
        + ", ".join(f"{int(n)}步 {float(next(r['fid_clean'] for r in fid_rows if r['n_steps']==str(n) and r['schedule']=='default')):.2f}" for n in [5,10,20])
        + " 降至 log-SNR 策略的 "
        + ", ".join(f"{int(n)}步 {float(next(r['fid_clean'] for r in fid_rows if r['n_steps']==str(n) and r['schedule']=='logsnr_power')):.2f}" for n in [5,10,20])
        + "。",
        "- 本机单张采样约 8.75 / 17.45 / 34.88 ms（5/10/20 NFE），自定义网格约 8.77 / 17.49 / 35.02 ms。计时含批次采样和图片生成保存前推理，不含 FID 特征提取；硬件、驱动和软件栈会影响数值。",
        "- 结果支持“在本设置下，均匀 log-SNR 网格相较 Diffusers 默认 linspace 提高了 FID”的观察；它不是普遍结论。FID 有有限样本误差，正式报告建议对结果作谨慎表述，并在条件允许时增加重复随机种子/样本数。",
        "- 由于 Windows 路径 glob 大小写不敏感，计算脚本先显式列出并核验 5,000 个唯一文件，再提取特征，避免对 `.png` 和 `.PNG` 模式重复计数。",
        "- UniPC 未参与这份 DPM++ 对比实验。只对交接框架执行了单样本 5 步 UniPC 冒烟检查，确认接口能运行；该样本不计入实验指标。",
        "",
        "## 关键产物",
        "",
        "- 全部指标：`outputs/metrics/fid.csv`、`timing.csv`、`validation_search.csv`、`selected_gamma.json`。",
        "- 图表：`outputs/metrics/dpmpp_results.png`；样本拼图：`outputs/samples/contact_sheet_48.png`。",
        "- 可复现实验脚本：`scripts/run_experiment.py`；共享模型框架：`shared_framework/`。",
    ]
    (OUT / "metrics" / "experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    charts()
    contact_sheet()
    summary()
    print("Created result chart, sample contact sheet, and experiment summary.")
