from ultralytics import YOLO

if __name__ == '__main__':
    # Load trained AGE-YOLO weights
    model = YOLO('weights/pretrained_neu.pt')

    # Run inference on sample images
    model.predict(
        source='assets/samples',
        conf=0.25,
        imgsz=640,
        save=True,
        project='assets',
        name='results',
        exist_ok=True
    )

    print('✅ Inference complete.')
    print('📂 Visual results saved to: assets/results')
