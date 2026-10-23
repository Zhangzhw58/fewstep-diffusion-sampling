# Few-Step Diffusion Sampling: DPM-Solver++ vs UniPC

少步扩散采样中求解器与时间步分配策略的比较研究（模式识别课程小组作业）。

- **固定**：预训练 `google/ddpm-cifar10-32`（像素空间 DDPM，32×32，无条件，预测 ε），版本 `267b167dc01f0e4e61923ea244e8b988f84deb80`，权重全程不动。
- **只改**：求解器（DPM-Solver++、UniPC，均为二阶）、步数 NFE ∈ {5, 10, 20}、时间步网格（默认 linspace，或 log-SNR 幂次网格 γ）。
- **协议**：γ 只在 16 个验证种子上挑选（以 100 步 DPM-Solver++ 为参考算像素 MSE），最终在 5,000 个留出测试种子（1000–5999）上用 clean-FID 和单张耗时评估。参数统一写在 `configs/experiment.json`。

## 目录与分工

| 路径 | 作用 | 负责人 | 修改规则 |
|---|---|---|---|
| `shared_framework/sampler.py` | 共用推理接口：`load_pipeline`、`make_scheduler`、`logsnr_power_timesteps`、`set_custom_timesteps`（UniPC 自定义网格，zzw 贡献）、`sample_batch` | 组长 | 共用代码，**改之前先在群里说**；改完跑 `scripts/check_custom_timesteps.py` |
| `shared_framework/README.md`、`requirements.txt` | 框架说明与最小依赖 | 组长 | |
| `shared_framework/checkpoints/` | 模型权重（**不进仓库**） | — | 用 `scripts/download_model.py` 下载 |
| `configs/experiment.json` | 实验协议：NFE、验证种子、γ 候选、测试种子、批大小 | 组长 | 改它等于改实验协议，**须全组同意**，并在提交说明里写明原因 |
| `requirements-project.txt` | 实验所需依赖（diffusers 0.31.0、clean-fid 0.1.35） | 组长 | |
| `scripts/run_experiment.py` | DPM-Solver++ 实验：smoke → validate → generate → fid | 组长 | |
| `scripts/run_unipc.py` | UniPC 实验，流程与上面完全相同，只换求解器；输出文件带 `_unipc` 后缀 | zzw | 与 `run_experiment.py` 保持同一协议 |
| `scripts/make_deliverables.py` | 生成结果图、样本拼图和小结（目前只画 DPM-Solver++） | 组长 | |
| `scripts/smoke.py`、`smoke_custom.py` | 早期冒烟测试 | 组长 | |
| `scripts/download_model.py` | 下载固定版本的模型 | zzw | |
| `scripts/check_custom_timesteps.py` | 自定义时间网格的正确性检验（18 项） | zzw | 改 `sampler.py` 后必须通过 |
| `results/rtx3070_laptop_dpmpp/` | 组长在 RTX 3070 笔记本上的 DPM-Solver++ 结果 | 组长 | 只追加，不改已有结果 |
| `results/a100_cn4/` | 两个求解器在同一张 A100 上的结果与对比表 | zzw | 只追加，不改已有结果 |
| `outputs/` | 运行时生成的样图、日志、中间指标（**不进仓库**） | — | 需要保留的结果复制到 `results/<机器>_<内容>/` |
| `paper/`（待建） | 英文 CVPR 双栏报告 | 全组 | |

## 快速开始

```bash
git clone https://github.com/Zhangzhw58/fewstep-diffusion-sampling.git
cd fewstep-diffusion-sampling
# 先按本机 CUDA 安装 PyTorch，再装依赖
pip install -r requirements-project.txt
python scripts/download_model.py              # 模型 → shared_framework/checkpoints/
python scripts/check_custom_timesteps.py      # 可选：确认时间网格功能正常
```

连不上 Hugging Face 时先设置 `HF_ENDPOINT=https://hf-mirror.com`（PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。clean-fid 第一次运行会自动下载 CIFAR-10 参考统计和 Inception 权重。

## 运行实验

在仓库根目录执行（Windows 把 `python` 换成 `.\.venv\Scripts\python.exe`）：

```bash
python scripts/run_experiment.py --phase all   # DPM-Solver++
python scripts/run_unipc.py --phase all        # UniPC
python scripts/make_deliverables.py            # DPM-Solver++ 的图表和小结
```

`--phase` 可选 `smoke`、`validate`、`generate`、`fid`、`all`，可以分阶段运行、断点续跑。单张 A100 上每个求解器全部阶段约 15 分钟（含 FID）。

## 主要结果（A100，clean-FID ↓，5,000 张）

| NFE | DPM++ 默认 | DPM++ log-SNR | UniPC 默认 | UniPC log-SNR |
|---:|---:|---:|---:|---:|
| 5 | 58.64 | 56.94 (γ=1.0) | **54.58** | 60.22 (γ=0.8) |
| 10 | 31.68 | 23.89 (γ=1.0) | 31.07 | **22.86** (γ=0.8) |
| 20 | 24.08 | 20.04 (γ=1.0) | 23.88 | **19.62** (γ=1.0) |

耗时、讨论和注意事项见 `results/a100_cn4/README.md`。DPM++ 在 A100 和 RTX 3070 上的 FID 相差不超过 0.08，结果可以复现。

## 协作约定

- **分支**：`main` 保持可运行的稳定版本；各人在自己的分支开发（如 `unipc`、`dpmpp`、`paper`），完成后提 Pull Request 合并到 `main`。
- **提交前**：`git status` 确认没有误加大文件；改了 `sampler.py` 就先跑 `scripts/check_custom_timesteps.py`。
- **结果**：新结果放在 `results/<机器>_<内容>/`，附一个 README 说明硬件、软件版本和运行命令；不要覆盖别人已有的结果。
- **不提交**：模型权重、生成的图片、`outputs/`、虚拟环境、压缩包、密钥（已写入 `.gitignore`）。
