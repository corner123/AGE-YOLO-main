from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained AGE-YOLO weights
    model = YOLO('weights/pretrained_gc10.pt')

    # Evaluate on the test split defined in dataset/gc10.yaml
    metrics = model.val(
        data='dataset/gc10.yaml',
        split='test',
        imgsz=640,
        device=0
    )

    # Output key evaluation metrics
    print(f"Test mAP@50: {metrics.results_dict['metrics/mAP50(B)']:.4f}")

