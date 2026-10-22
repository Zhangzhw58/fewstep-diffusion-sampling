# DPM-Solver++ CIFAR-10 实验结果小结

## 实验设置

- 模型：`google/ddpm-cifar10-32`，固定 revision `267b167dc01f0e4e61923ea244e8b988f84deb80`，32×32 无条件 CIFAR-10 生成。
- 设备：NVIDIA GeForce RTX 3070 Laptop GPU（8 GB）；batch size 32。
- Solver：二阶 DPM-Solver++；NFE 为 5、10、20。
- 配对设置：所有对比共用相同的初始噪声 seed。时间步选择只用 16 个验证 seed；生成评估使用另 5,000 个 seed（1000–5999）。
- 自定义策略：在 log-SNR 空间对时间点做 power warp，在 gamma ∈ {0.65, 0.8, 1.0, 1.2, 1.4} 中以验证集相对 100 步 DPM-Solver++ 输出的像素 MSE 选择 gamma。该代理目标不等于 FID，也不保证测试指标必然改善。
- FID：clean-fid 的 CIFAR-10 train 32×32 参考统计；每个配置恰好 5,000 张生成图。这里与参考统计比较，不是拿生成图与 CIFAR-10 test 图片逐张比较。

## 结果

| NFE | 时间步 | gamma | Clean-FID ↓ | ms/张 ↓ |
|---:|---|---:|---:|---:|
| 5 | 默认 linspace | — | 58.60 | 8.75 |
| 5 | log-SNR power | 1.0 | 56.97 | 8.77 |
| 10 | 默认 linspace | — | 31.60 | 17.45 |
| 10 | log-SNR power | 1.0 | 23.87 | 17.49 |
| 20 | 默认 linspace | — | 24.02 | 34.88 |
| 20 | log-SNR power | 1.0 | 20.05 | 35.02 |

## 解读与边界

- 验证集在 5/10/20 NFE 都选出 gamma=1.0（均匀 log-SNR 网格）。测试 FID 分别由默认时间步的 5步 58.60, 10步 31.60, 20步 24.02 降至 log-SNR 策略的 5步 56.97, 10步 23.87, 20步 20.05。
- 本机单张采样约 8.75 / 17.45 / 34.88 ms（5/10/20 NFE），自定义网格约 8.77 / 17.49 / 35.02 ms。计时含批次采样和图片生成保存前推理，不含 FID 特征提取；硬件、驱动和软件栈会影响数值。
- 结果支持“在本设置下，均匀 log-SNR 网格相较 Diffusers 默认 linspace 提高了 FID”的观察；它不是普遍结论。FID 有有限样本误差，正式报告建议对结果作谨慎表述，并在条件允许时增加重复随机种子/样本数。
- 由于 Windows 路径 glob 大小写不敏感，计算脚本先显式列出并核验 5,000 个唯一文件，再提取特征，避免对 `.png` 和 `.PNG` 模式重复计数。
- UniPC 未参与这份 DPM++ 对比实验。只对交接框架执行了单样本 5 步 UniPC 冒烟检查，确认接口能运行；该样本不计入实验指标。

## 关键产物

- 全部指标：`outputs/metrics/fid.csv`、`timing.csv`、`validation_search.csv`、`selected_gamma.json`。
- 图表：`outputs/metrics/dpmpp_results.png`；样本拼图：`outputs/samples/contact_sheet_48.png`。
- 可复现实验脚本：`scripts/run_experiment.py`；共享模型框架：`shared_framework/`。
