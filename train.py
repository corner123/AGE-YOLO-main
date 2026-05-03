import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# ===================== Configuration =====================
MODEL_CONFIG = 'ultralytics/cfg/models/age_yolo.yaml'
DATA_CONFIG = 'dataset/gc10_data.yaml'
PROJECT_DIR = 'runs/AGE-YOLO-GC10'
EXPERIMENT_NAME = 'Official-Release'
# =========================================================

def main():
    model = YOLO(MODEL_CONFIG, task='detect')

    model.train(
        data=DATA_CONFIG,
        epochs=300,
        batch=16,
        imgsz=640,
        optimizer='SGD',
        lr0=0.01,
        patience=50,

        # Mild online augmentation (offline augmentation already applied)
        mosaic=1.0,
        fliplr=0.0,
        flipud=0.0,
        scale=0.1,
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,

        pretrained=False,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        device=0,
        seed=42
    )

if __name__ == '__main__':
    main()
