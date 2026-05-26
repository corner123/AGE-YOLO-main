import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

MODEL_CONFIG = "ultralytics/cfg/models/yolo11n.yaml"
DATA_CONFIG = "dataset/gc10_data.yaml"
PROJECT_DIR = "runs/YOLOv11n-GC10"
EXPERIMENT_NAME = "baseline"
PAPER_SEEDS = (42, 2025, 1024, 666, 777)


def train_one(seed=42, name=None, workers=4, device=0):
    model = YOLO(MODEL_CONFIG, task="detect")
    model.train(
        data=DATA_CONFIG,
        epochs=300,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=50,
        cos_lr=False,
        close_mosaic=10,
        mosaic=1.0,
        fliplr=0.0,
        flipud=0.0,
        scale=0.1,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        pretrained=False,
        deterministic=True,
        cache=False,
        workers=workers,
        amp=True,
        project=PROJECT_DIR,
        name=name or EXPERIMENT_NAME,
        device=device,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11n baseline on GC10-DET with the paper protocol.")
    parser.add_argument("--seed", type=int, default=42, help="Single-run seed. Ignored when --all-seeds is set.")
    parser.add_argument("--all-seeds", action="store_true", help="Run the five seeds reported in the paper.")
    parser.add_argument("--name", default=EXPERIMENT_NAME, help="Run name for a single seed.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers. 4 is safer for 16 GB RAM.")
    parser.add_argument("--device", default=0, help="CUDA device id, e.g. 0.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_seeds:
        for seed in PAPER_SEEDS:
            train_one(seed=seed, name=f"{EXPERIMENT_NAME}-seed{seed}", workers=args.workers, device=args.device)
    else:
        train_one(seed=args.seed, name=args.name, workers=args.workers, device=args.device)


if __name__ == "__main__":
    main()
