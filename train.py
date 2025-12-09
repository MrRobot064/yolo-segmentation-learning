# %%
# 环境 & 设备检查
import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {DEVICE}")

# %%
# 训练 YOLO 分割模型
model = YOLO("yolo11n-seg.pt")

results = model.train(
    data="coco8-seg.yaml",
    imgsz=320,
    epochs=5,
    batch=4,
    workers=0,
    device=DEVICE,
)

print("训练完成，权重保存在 runs/segment/train*/weights/ 下。")

# %%

# 仅在 VSCode Interactive / Jupyter 环境使用
%matplotlib inline

# 可视化训练结果（loss / metrics）
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8")

ROOT = Path(".").resolve()
runs_dir = ROOT / "runs" / "segment"

if not runs_dir.exists():
    raise FileNotFoundError("没有找到 runs/segment/ 目录。")

# 最近一次训练 run
last_run = max(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
print("检测到最新 run：", last_run)

csv_path = last_run / "results.csv"
print("加载训练日志 CSV 文件：", csv_path)

if not csv_path.exists():
    raise FileNotFoundError("results.csv 不存在，可能训练未正常结束。")

df = pd.read_csv(csv_path)
print("\nCSV 列名：")
print(df.columns.tolist())

epochs = df["epoch"]

# Loss 曲线
loss_cols = [c for c in df.columns if "loss" in c]
if loss_cols:
    plt.figure(figsize=(12, 5))
    for col in loss_cols:
        plt.plot(epochs, df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练 / 验证 Loss 曲线")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("没有找到 loss 相关列。")

# Metrics 曲线
metric_cols = [c for c in df.columns if c.startswith("metrics/")]
if metric_cols:
    plt.figure(figsize=(12, 5))
    for col in metric_cols:
        plt.plot(epochs, df[col], label=col)
    plt.xlabel("Epoch")
    plt.ylabel("Metrics")
    plt.title("验证集 Metrics 曲线（BBox + Mask）")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("没有找到 metrics 相关列。")

# %%
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(".").resolve()

# 1. 训练结束后，从 model 拿 best 权重路径
best_ckpt = model.ckpt_path  # Ultralytics 会在训练结束时写好这个属性
print("使用权重:", best_ckpt)

pred_model = YOLO(str(best_ckpt))

# 2. 用 coco8-seg 的 val 集做预测
#   数据集默认会下载到 datasets/coco8-seg/
DATASET_ROOT = Path("datasets") / "coco8-seg"
val_dir = DATASET_ROOT / "images" / "val"  # 直接给文件夹，YOLO 会自动遍历里面所有图片

pred_model.predict(
    source=str(val_dir),   # 这里用 source=目录
    imgsz=320,
    device=DEVICE,
    save=True,             # 保存预测结果
    project=ROOT / "runs" / "segment",
    name="predict_coco8",  # 输出目录：runs/segment/predict_coco8/
    exist_ok=True,
)

print("预测完成，查看目录:", ROOT / "runs" / "segment" / "predict_coco8")

# %%
