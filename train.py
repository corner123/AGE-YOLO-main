import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO

# ===================== Configuration =====================
# Paper: Table 2 hyperparameters + Section 4.2 augmentation
MODEL_CONFIG = 'cfg/models/age_yolo.yaml'
DATA_CONFIG = 'dataset/NEU-DET/steel_defect.yaml'
PROJECT_DIR = 'runs/AGE-YOLO-NEU'
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

        # Mild online augmentation (offline x5 already applied)
        # Paper Section 4.2: offline aug does flip/scale/HSV; online only keeps mosaic
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
