import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from ultralytics import YOLO


DEFAULT_WEIGHTS = "runs/AGE-YOLO-NEU/Official-Release-PaperSplit-seed42/weights/best.pt"
DEFAULT_DATA = "dataset/NEU-DET-paper/steel_defect.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AGE-YOLO on the NEU paper split.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to model weights.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset yaml.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split to evaluate.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default=0, help="CUDA device id, e.g. 0.")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for stable Windows validation output.")
    parser.add_argument("--plots", action="store_true", help="Save confusion matrix and PR/F1 curves.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"weights: {args.weights}")
    print(f"data: {args.data}")
    print(f"split: {args.split}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        plots=args.plots,
        project="runs/AGE-YOLO-NEU",
        name=f"eval-{args.split}",
    )

    print(f"mAP@50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"mAP@50-95: {metrics.results_dict['metrics/mAP50-95(B)']:.4f}")


if __name__ == "__main__":
    main()
