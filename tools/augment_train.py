import os
import cv2
import yaml
import random
import numpy as np
from tqdm import tqdm
import argparse


class AGE_YOLO_Augmentor:
    """Scaling + Horizontal Flip + Mild HSV Jitter"""

    def __init__(self, flip_p=0.5, scale_range=(0.8, 1.2), hsv_jitter=0.2):
        self.flip_p = flip_p
        self.scale_range = scale_range
        self.hsv_jitter = hsv_jitter

    def __call__(self, img, labels):
        # Random scaling
        s = random.uniform(*self.scale_range)
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * s), int(h * s)))

        # Horizontal flip
        if random.random() < self.flip_p:
            img = img[:, ::-1].copy()
            if labels.size > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        # HSV jitter
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(1 - self.hsv_jitter, 1 + self.hsv_jitter)
        hsv[:, :, 2] *= random.uniform(1 - self.hsv_jitter, 1 + self.hsv_jitter)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img, labels


def run_augment(yaml_path, multiplier):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    root = cfg['path']
    img_dir = os.path.join(root, cfg['train'])
    lbl_dir = img_dir.replace('images', 'labels')

    assert os.path.exists(img_dir), f"Train images not found: {img_dir}"

    augmentor = AGE_YOLO_Augmentor()

    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.jpg', '.png')) and '_aug' not in f
    ]

    for fname in tqdm(img_files, desc=f"Offline Augmentation x{multiplier}"):
        base = os.path.splitext(fname)[0]
        img = cv2.imread(os.path.join(img_dir, fname))

        lbl_path = os.path.join(lbl_dir, base + '.txt')
        if os.path.exists(lbl_path):
            labels = np.loadtxt(lbl_path).reshape(-1, 5)
        else:
            labels = np.zeros((0, 5), dtype=np.float32)

        for i in range(1, multiplier):
            aug_img, aug_lbl = augmentor(img.copy(), labels.copy())
            out_base = f"{base}_aug{i}"

            cv2.imwrite(os.path.join(img_dir, out_base + '.jpg'), aug_img)

            if aug_lbl.size > 0:
                with open(os.path.join(lbl_dir, out_base + '.txt'), 'w') as f:
                    for l in aug_lbl:
                        f.write(f"{int(l[0])} {' '.join(f'{x:.6f}' for x in l[1:])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default='dataset/data.yaml')
    parser.add_argument('--mult', type=int, default=4)
    args = parser.parse_args()

    run_augment(args.yaml, args.mult)
