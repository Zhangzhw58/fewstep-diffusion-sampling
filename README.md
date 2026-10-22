# CIFAR-10 少步扩散采样：DPM-Solver++ 与 UniPC 对比

模式识别研究生课程小组作业。

- **模型固定**：`google/ddpm-cifar10-32`（像素空间 DDPM，32×32，无条件，预测噪声 ε），版本 `267b167dc01f0e4e61923ea244e8b988f84deb80`，权重全程不动。
- **只改变**：采样器（DPM-Solver++、UniPC）、步数 NFE、时间步网格（log-SNR 幂次参数 γ）。

## 目录

```
├── sampler.py                    # 共用推理函数（组长维护）
├── requirements.txt
├── checkpoints/
│   └── google-ddpm-cifar10-32/   # 模型（不进仓库，用 scripts/download_model.py 下载）
├── scripts/
│   ├── download_model.py         # 下载固定版本的模型
│   ├── smoke_test.py             # 求解器 × NFE → 样图 + FID
│   └── check_custom_timesteps.py # 自定义时间网格的正确性检验
├── fid/                          # FID 所需文件（不进仓库，见下文）
└── outputs/                      # 样图和结果（不进仓库）
```

模型文件（143 MB）超过 GitHub 单文件 100 MB 的上限，FID 文件也很大，都不提交，各人按下面的步骤自行下载；`.gitignore` 已排除这些目录。

## `sampler.py` 提供的函数

| 函数 | 作用 |
|---|---|
| `load_pipeline(model_dir, device)` | 从本地加载模型，不联网 |
| `make_scheduler(pipe, "dpmpp" / "unipc", solver_order=2)` | 按模型原始的 DDPM 噪声配置建 DPM-Solver++ 或 UniPC 调度器 |
| `logsnr_power_timesteps(scheduler, n_steps, gamma)` | log-SNR 均匀网格经幂次 γ 变换后的时间步（γ=1 即 log-SNR 均匀） |
| `set_custom_timesteps(scheduler, timesteps, device)` | 把调度器换成自定义时间网格，DPM-Solver++ 和 UniPC 通用 |
| `sample_batch(pipe, scheduler, seeds, n_steps, gamma=None, device)` | 按种子批量生成，返回 PIL 图像；`gamma=None` 用默认 linspace 网格 |

```python
from sampler import load_pipeline, make_scheduler, sample_batch
pipe = load_pipeline("checkpoints/google-ddpm-cifar10-32", "cuda")
scheduler = make_scheduler(pipe, "unipc", solver_order=2)
images = sample_batch(pipe, scheduler, seeds=[1, 2, 3], n_steps=10, gamma=1.0, device="cuda")
```

每次调用 `sample_batch` 都会重置调度器的时间网格；并行或互相独立的任务请各自新建调度器。

## 环境

组内统一用 **diffusers 0.31.0**，不同版本的调度器细节不同，结果会对不上。

```bash
conda create -n cifar_sampler python=3.10 -y
conda activate cifar_sampler
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # 按自己机器的 CUDA 版本选择
pip install -r requirements.txt "huggingface_hub<1.0"
```

已验证的组合：PyTorch 2.9.0+cu128（组长）；PyTorch 2.5.1+cu121（cn5）。

## 准备文件（每人一次）

```bash
python scripts/download_model.py     # 模型，固定版本 267b167
# 连不上 Hugging Face：export HF_ENDPOINT=https://hf-mirror.com
#（Windows PowerShell：$env:HF_ENDPOINT="https://hf-mirror.com"）

# FID 所需文件（只出样图时可跳过）
git clone https://github.com/NVlabs/edm.git fid/edm
curl -L -o fid/inception-2015-12-05.pkl https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
curl -L -o fid/cifar10-32x32.npz https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```

FID 使用 EDM 官方实现和参考统计量（与 EDM、DPM-Solver 等论文口径一致）。全组必须用同一套文件，数字才能互相比较。

## 运行

```bash
python scripts/check_custom_timesteps.py                      # 先确认时间网格功能正确
python scripts/smoke_test.py --nfe 5 10 --num 5000            # 4 种配置 × 2 种 NFE
python scripts/smoke_test.py --configs dpmpp unipc --nfe 5 --num 0   # 只出样图，几秒钟
```

| `smoke_test.py` 参数 | 默认值 | 含义 |
|---|---|---|
| `--configs` | `dpmpp unipc dpmpp:1.0 unipc:1.0` | `求解器` 用默认 linspace 网格；`求解器:γ` 用 log-SNR 幂次网格 |
| `--nfe` | `5 10` | 步数，可写多个 |
| `--num` | `5000` | 每个配置生成多少张图算 FID；`0` 只出样图；正式结果用 `50000` |
| `--batch` | `500` | 每批张数，显存不够就调小 |

输出：
- `outputs/samples/*.png`：8×8 样图，不同配置下同一位置是同一个随机种子；
- `outputs/metrics/smoke.jsonl`：每个配置一行，记录实际时间步、每张图耗时、FID 和状态。

单张 A100 上，默认设置约 3 分钟跑完，显存约 19 GB。

## 分工

| 部分 | 负责人 | 分支 |
|---|---|---|
| 共用框架 `sampler.py`、DPM-Solver++ 实验 | 组长 | `main` / `dpmpp` |
| UniPC 实验；数据与评测（FID 流程、结果汇总） | zzw | `unipc` / `eval` |
| 其他 | （补充） | |

协作约定：
- 各自在自己的分支上开发，改好后提 Pull Request 合并到 `main`；
- **修改 `sampler.py` 等共用代码要先在群里说一声**；
- 不提交模型、FID 文件和大量样图；结果表和少量代表性样图可以提交。

## 正确性检验（2026-10-22，diffusers 0.31.0）

`scripts/check_custom_timesteps.py` 共 18 项，全部通过：
- 把 UniPC 自己的默认网格当成自定义网格传入，生成的图与默认采样逐像素一致（5 步、10 步）；DPM-Solver++ 同样一致；
- UniPC 的 γ = 0.5、1、2 网格可正常运行，实际时间步与请求的一致；
- 加 `--reference` 与原始版本对比：DPM-Solver++（默认、γ=1、γ=2）和 UniPC 默认网格的输出逐像素一致。

## 冒烟测试结果（2026-10-22，diffusers 0.31.0，FID-5k，仅用于比较配置）

| 求解器 | 时间网格 | NFE=5 | NFE=10 |
|---|---|---|---|
| DPM-Solver++ | linspace | 56.06 | 28.73 |
| UniPC | linspace | 52.11 | 28.08 |
| DPM-Solver++ | γ=1 | 55.55 | 21.79 |
| UniPC | γ=1 | 50.18 | 20.21 |

前 6 组与 diffusers 0.33.1 下测得的结果完全相同。

## 注意事项

1. **不能直接和论文表格对比**：DPM-Solver、UniPC 论文用的是 ScoreSDE 的 CIFAR-10 模型，我们用的是 diffusers 的 DDPM 模型。
2. **对照组放三行**：γ=1（log-SNR 均匀）本身就是 DPM-Solver 作者推荐的设置，结果表应同时列 linspace、γ=1、最优 γ。
3. **FID-5k 比 FID-50k 系统性偏高**，只用于调参；报告里的正式数字用 50k，并写明样本数、参考统计量和实现。
4. **验证集与测试集分开**：γ 只在验证种子上挑选，最终结果用另一批没参与挑选的测试种子。
