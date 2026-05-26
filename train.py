import argparse
import warnings

warnings.filterwarnings("ignore")

from ultralytics import YOLO

# Paper protocol: Table 2 hyperparameters + Section 4.2 augmentation.
MODEL_CONFIG = "cfg/models/age_yolo.yaml"
DATA_CONFIG = "dataset/NEU-DET-paper/steel_defect.yaml"
PROJECT_DIR = "runs/AGE-YOLO-NEU"
EXPERIMENT_NAME = "Official-Release-PaperSplit"
PAPER_SEEDS = (42, 2025, 1024, 666, 777)


def train_one(seed=42, name=None, workers=4, device=0, data=DATA_CONFIG, epochs=300, batch=16, patience=50):
    run_name = name or EXPERIMENT_NAME
    print(f"Training AGE-YOLO | data={data} | seed={seed} | name={run_name}")
    model = YOLO(MODEL_CONFIG, task="detect")
    model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        patience=patience,
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
        name=run_name,
        device=device,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train AGE-YOLO on NEU-DET with the paper protocol.")
    parser.add_argument("--seed", type=int, default=42, help="Single-run seed. Ignored when --all-seeds is set.")
    parser.add_argument("--all-seeds", action="store_true", help="Run the five seeds reported in the paper.")
    parser.add_argument("--name", default=EXPERIMENT_NAME, help="Run name for a single seed.")
    parser.add_argument("--data", default=DATA_CONFIG, help="Dataset yaml. Defaults to the paper-testlist NEU split.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--patience", type=int, default=50, help="Early-stopping patience. Paper default is 50.")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers. 4 is safer for 16 GB RAM.")
    parser.add_argument("--device", default=0, help="CUDA device id, e.g. 0.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.all_seeds:
        for seed in PAPER_SEEDS:
            train_one(
                seed=seed,
                name=f"{args.name}-seed{seed}",
                workers=args.workers,
                device=args.device,
                data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                patience=args.patience,
            )
    else:
        train_one(
            seed=args.seed,
            name=args.name,
            workers=args.workers,
            device=args.device,
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
        )


if __name__ == "__main__":
    main()
