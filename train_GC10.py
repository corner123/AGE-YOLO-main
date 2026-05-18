import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

# ===================== Configuration =====================
# Paper: GC10-DET as primary benchmark (Table 2 + Section 4.2)
MODEL_CONFIG = 'cfg/models/age_yolo_gc10.yaml'
DATA_CONFIG = 'dataset/gc10_data.yaml'
PROJECT_DIR = 'runs/AGE-YOLO-GC10'
EXPERIMENT_NAME = 'Official-Release'
# =========================================================

def main():
    model = YOLO(MODEL_CONFIG, task='detect')

    model.train(
        data=DATA_CONFIG,
        epochs=300,           # Table 2: Maximum epochs
        batch=16,             # Table 2: Batch size
        imgsz=640,            # Table 2: Image resolution
        optimizer='SGD',      # Table 2: Optimizer
        lr0=0.01,             # Table 2: Initial learning rate
        patience=50,          # Table 2: Early stopping patience

        # Mild online augmentation (offline x4 already applied)
        # Paper Section 4.2: GC10-DET uses x4 offline augmentation
        mosaic=1.0,
        fliplr=0.0,           # Offline already did horizontal flip
        flipud=0.0,
        scale=0.1,            # Mild jitter only, offline did [0.8,1.2]
        hsv_h=0.01,
        hsv_s=0.2,            # Match offline HSV jitter factor
        hsv_v=0.2,            # Match offline HSV jitter factor

        pretrained=False,
        project=PROJECT_DIR,
        name=EXPERIMENT_NAME,
        device=0,
        seed=42
    )

if __name__ == '__main__':
    main()
