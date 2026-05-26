"""
RT-DETR training script for GC10-DET.

The AGE-YOLO repository contains a modified local ultralytics package. RT-DETR
is trained with the official pip-installed ultralytics package instead.
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CONFIG = os.path.join(PROJECT_ROOT, "dataset", "gc10_data.yaml")
PROJECT_DIR = os.path.join(PROJECT_ROOT, "runs", "RT-DETR-GC10")
EXPERIMENT_NAME = "rtdetr-l-gc10"


def use_official_ultralytics():
    project_root = os.path.normcase(os.path.abspath(PROJECT_ROOT))
    sys.path[:] = [
        p for p in sys.path
        if p and os.path.normcase(os.path.abspath(p)) != project_root
    ]

    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            module_file = getattr(sys.modules[name], "__file__", "") or ""
            if os.path.normcase(os.path.abspath(module_file)).startswith(project_root):
                del sys.modules[name]


use_official_ultralytics()

from ultralytics import RTDETR  # noqa: E402
import ultralytics  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train official RT-DETR on GC10-DET.")
    parser.add_argument("--model", default="rtdetr-l.pt", help="RT-DETR model, e.g. rtdetr-l.pt or rtdetr-l.yaml.")
    parser.add_argument("--from-scratch", action="store_true", help="Use YAML architecture instead of pretrained .pt weights.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16, help="A100 40 GB fits batch 16; use --paper-batch for paper-style batch 8.")
    parser.add_argument("--paper-batch", action="store_true", help="Use batch 8, matching the RT-DETR exception noted in the paper.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4, help="Conservative for 16 GB system RAM.")
    parser.add_argument("--device", default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=EXPERIMENT_NAME)
    parser.add_argument("--fraction", type=float, default=1.0, help="Use less than 1.0 only for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Only check imports, model loading, and config.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation during training smoke tests.")

    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine LR instead of the paper-style linear decay.")

    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--hsv-h", type=float, default=0.01)
    parser.add_argument("--hsv-s", type=float, default=0.2)
    parser.add_argument("--hsv-v", type=float, default=0.2)
    return parser.parse_args()


def resolve_model_name(args):
    if not args.from_scratch:
        return args.model
    stem, ext = os.path.splitext(args.model)
    return stem + ".yaml" if ext.lower() == ".pt" else args.model


def main():
    args = parse_args()
    model_name = resolve_model_name(args)
    batch = 8 if args.paper_batch else args.batch
    pretrained = not args.from_scratch and model_name.lower().endswith(".pt")

    print(f"Using official ultralytics {ultralytics.__version__}: {ultralytics.__file__}")
    print(f"Model: {model_name} | pretrained={pretrained} | batch={batch}")
    print(f"Data: {DATA_CONFIG}")

    model = RTDETR(model_name)
    if args.dry_run:
        print("Dry run OK: RT-DETR model and official ultralytics loaded.")
        return

    model.train(
        data=DATA_CONFIG,
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=0.9,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        mosaic=args.mosaic,
        fliplr=args.fliplr,
        flipud=args.flipud,
        scale=args.scale,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mixup=0.0,
        copy_paste=0.0,
        deterministic=False,
        workers=args.workers,
        cache=False,
        amp=True,
        val=not args.no_val,
        fraction=args.fraction,
        project=PROJECT_DIR,
        name=args.name,
        device=args.device,
        seed=args.seed,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    main()
