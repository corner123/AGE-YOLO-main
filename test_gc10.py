import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

from ultralytics import YOLO


DEFAULT_DATA = "dataset/gc10_data.yaml"
DEFAULT_PROJECT = Path("runs/AGE-YOLO-GC10")


def find_latest_best(project=DEFAULT_PROJECT):
    candidates = sorted(
        project.glob("*/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_weights(args):
    if args.weights:
        return Path(args.weights)

    if args.run:
        return DEFAULT_PROJECT / args.run / "weights" / "best.pt"

    weights = find_latest_best()
    if weights is None:
        raise FileNotFoundError(
            "No GC10 weights found under runs/AGE-YOLO-GC10/*/weights/best.pt. "
            "Pass --weights explicitly."
        )
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AGE-YOLO on the GC10-DET split.")
    parser.add_argument("--weights", default=None, help="Path to model weights. Defaults to latest GC10 best.pt.")
    parser.add_argument("--run", default=None, help="Run folder under runs/AGE-YOLO-GC10, e.g. Official-Release4.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="GC10 dataset yaml.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default=0, help="CUDA device id, e.g. 0.")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for stable Windows validation output.")
    parser.add_argument("--plots", action="store_true", help="Save confusion matrix and PR/F1 curves.")
    parser.add_argument("--name", default=None, help="Evaluation output name under runs/AGE-YOLO-GC10.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights = resolve_weights(args)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    eval_name = args.name or f"eval-{args.split}"
    print(f"weights: {weights}")
    print(f"data: {args.data}")
    print(f"split: {args.split}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=args.plots,
        project=str(DEFAULT_PROJECT),
        name=eval_name,
    )

    print(f"mAP@50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")


if __name__ == "__main__":
    main()
