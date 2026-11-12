# 新增消融实验（2026-11）

本目录保存新增的两组 DPM-Solver++ 消融，不覆盖原始基线结果。

## 实验目的

1. 扩大 log-SNR 时间步 gamma 的验证规模，观察原先 16 个验证 seed 得到的 gamma=1.0 是否稳定。
2. 固定 10 NFE，对比 DPM-Solver++ 一阶与二阶在默认时间步及验证选定 log-SNR 时间步下的表现。

## 固定条件

- 模型：`google/ddpm-cifar10-32`，revision `267b167dc01f0e4e61923ea244e8b988f84deb80`。
- 硬件：NVIDIA GeForce RTX 3070 Laptop GPU（8 GB）。
- NFE：10。
- gamma 验证集：128 个 seed（20000–20127）；以 100 步二阶 DPM-Solver++ 输出为参考，使用像素 MSE 选择 gamma。
- 测试集：独立 1,000 个 seed（10000–10999）；同一 seed 用于各配置，FID 参考 clean-fid CIFAR-10 train 32x32 统计。
- 测试 gamma：验证选出的 gamma=0.8；额外包含 gamma=1.2 作为敏感度探索值。gamma=1.2 不是验证集选出的方案，不能当作确认性结果。

## 主要结果

### Gamma 验证与敏感度

| 时间步 | gamma | 验证 MSE（对 100 步参考） |
|---|---:|---:|
| 默认 | — | **0.000803** |
| log-SNR | 0.65 | 0.006676 |
| log-SNR | 0.8 | 0.001258 |
| log-SNR | 1.0 | 0.001277 |
| log-SNR | 1.2 | 0.001595 |
| log-SNR | 1.4 | 0.001690 |

按验证 MSE 选择的 gamma 是 0.8，但默认时间步的验证 MSE 更低。

### 一阶与二阶（1,000 张测试样本）

| Solver 阶数 | 时间步 | Clean-FID ↓ | ms/张（约） |
|---:|---|---:|---:|
| 2 | 默认 | 56.15 | 19.98 |
| 2 | gamma=0.8 | 48.45 | 21.35 |
| 1 | 默认 | 69.58 | 19.32 |
| 1 | gamma=0.8 | 71.72 | 18.94 |
| 2 | gamma=1.2（探索） | 46.93 | 18.85 |

在该设置下，二阶的 FID 低于一阶；gamma=0.8 在二阶时降低 FID，在一阶时没有改善。gamma=1.2 测试 FID 更低，但它并非由验证标准选出，只作为探索值展示。

## 限制与报告建议

- 新增 FID 每配置 1,000 张，原始基线每配置 5,000 张，因此两者的 FID 数值不可直接比较。仅在本目录内相同 1,000-seed 配置之间作初步比较。
- MSE 参考目标与 FID 评价目标不同，二者排序不一致；不要把验证 MSE 改善等同于分布质量普遍改善。
- gamma=1.2 是额外敏感度探索，不能以测试 FID 反向挑选后再报告为验证选出的改进。
- 建议报告表述为“在本模型、10 NFE 和当前样本规模下观察到的结果”，避免普遍性结论。若时间允许，可为关键配置增加独立 seed 重复和更大测试样本数。

## 文件清单

- `../scripts/run_ablation_nov2026.py`：完整新增实验脚本。
- `metrics/gamma_validation.csv`：gamma 验证 MSE。
- `metrics/fid.csv`：测试 FID（每配置 1,000 张）。
- `metrics/timing.csv`：生成耗时。
- `samples/validation_ref_order2_default/`：验证用 10 步默认时间步样本。
- `samples/validation_gamma_*/`：各 gamma 的验证生成样本。
- `samples/order1_*`、`samples/order2_*`、`samples/gamma*_sensitivity/`：独立测试 seed 生成图像。

## 复跑

从项目根目录执行：

```powershell
.\.venv\Scripts\python.exe .\scripts\run_ablation_nov2026.py
```

需要 CUDA PyTorch、Diffusers、clean-fid 与本地模型权重。结果写到 `outputs/ablation_2026_11/`。

## 入库说明

仓库里只放了本说明和 `metrics/`。`samples/`（约 6,800 张 PNG）按仓库约定不入库；`scripts/run_ablation_nov2026.py` 待组长提交。
