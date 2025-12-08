# YOLO 分割训练流程

简洁的步骤和常用命令，帮助把原始图像/掩膜转换为 YOLOv8 segmentation 所需的数据，并完成训练、可视化与推理。

**快速开始**
- 准备 Python 3.10+ 虚拟环境。
- 安装依赖：

```bash
pip install ultralytics opencv-python pillow tqdm matplotlib
```

**目录概览**
- `insert_images/`：原始图像（支持 `.tif`），不入库
- `insert_masks/`：原始掩膜（支持 `.tif/.tiff/.png/.gif`），不入库
- `images/`：由 `insert_images/` 转换的 `.jpg`
- `labels/`：YOLO segmentation 格式的 `.txt` 标签
- `dataset/`：按 `train/val/test` 拆分的 `images` + `labels`
- `runs/segment/`：训练输出（日志、`results.csv`、权重等）
- `rectus_femoris_seg.yaml`：数据集配置

## 使用 Notebook (`main.ipynb`) 的流程（推荐）
按顺序运行 notebook 中的 Step 单元：

1. convert_tif_folder_to_jpg: 将 `insert_images/*.tif` 转为 `images/*.jpg`。
2. build_labels_from_masks: 将掩膜转换为 YOLO segmentation 标签写入 `labels/`。
3. split_dataset: 按比例把 `images` + `labels` 拷贝到 `dataset/train|val|test`。
4. write_data_yaml: 生成 `rectus_femoris_seg.yaml`（模型训练使用）。
5. 在 notebook 中调用 `model.train(...)` 训练分割模型（或使用命令行）。
6. 可视化训练结果并用最新权重在测试集上做推理。

## 重要函数与说明
- `mask_tif_to_yolo_txt(mask_path, txt_path, class_id=0)`：把单张掩膜转换为 YOLO seg 文本（每行为 class_id 后跟归一化的点坐标）。
- `build_labels_from_masks(mask_dir, label_dir, class_id=0)`：批量处理掩膜并写入标签文件。
- `split_dataset(images_dir, labels_dir, dataset_dir, train_ratio, val_ratio)`：按比例拆分并复制数据到 `dataset/`。
- `write_data_yaml(dataset_dir, out_path, class_names)`：生成 YOLOv8 数据配置文件（`path, train, val, test, names`）。

## 常用命令（CLI）

训练（示例）：
```bash
yolo train model=yolo11n-seg.pt data=rectus_femoris_seg.yaml imgsz=640 epochs=50 batch=4 device=0
```

推理（示例）：
```bash
yolo predict model=runs/segment/train*/weights/best.pt source=dataset/test/images imgsz=640 conf=0.25
```

查看训练曲线（Notebook 中也有可视化单元）：

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")
runs_dir = Path("runs/segment")
csv_path = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime) / "results.csv"
df = pd.read_csv(csv_path)

# 绘制 loss
loss_cols = [c for c in df.columns if "loss" in c]
if loss_cols:
  df.plot(x="epoch", y=loss_cols, figsize=(10,4), title="训练/验证损失")
  plt.grid(True); plt.show()

# 绘制 metrics
metric_cols = [c for c in df.columns if c.startswith("metrics/")]
if metric_cols:
  df.plot(x="epoch", y=metric_cols, figsize=(10,4), title="验证指标")
  plt.grid(True); plt.show()
```

## 注意事项与建议
- 确保每张图片对应的 `labels/<stem>.txt` 存在，否则该图片会在 `split_dataset` 时被忽略。
- 掩膜中应使用二值前景（非 0 为前景），脚本会提取外轮廓用于生成多边形点坐标。
- 当数据集很小时（如仅几十张），训练指标可能不稳定，建议增加数据或使用数据增强。

## 示例：快速操作流程

1. 放入原始文件：把 `.tif` 图像放 `insert_images/`，掩膜放 `insert_masks/`。
2. 在 `main.ipynb` 依次运行 Steps 1-4：转换、生成标签、拆分、写 `rectus_femoris_seg.yaml`。
3. 训练：在 notebook 运行 Step 5 或使用上方 CLI 命令。
4. 推理与可视化：运行 notebook Step 6（或使用 `yolo predict`）。
