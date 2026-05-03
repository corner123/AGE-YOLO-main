import os
import random
import shutil
from tqdm import tqdm


def split_dataset(image_dir, label_dir, output_dir, ratio=(0.8, 0.1, 0.1), seed=42):
    """
    Split dataset into train, val, and test sets.
    1. Group files by basename.
    2. Shuffle and split according to the ratio.
    3. Copy images and labels to subdirectories.
    """
    random.seed(seed)

    if not os.path.exists(image_dir):
        return

    # Scan for image files
    image_files = []
    for root, _, files in os.walk(image_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, f), image_dir)
                image_files.append(rel_path)

    # Group by basename to handle image pairs
    groups = {}
    for f in image_files:
        base = os.path.splitext(os.path.basename(f))[0]
        if base not in groups:
            groups[base] = []
        groups[base].append(f)

    basenames = list(groups.keys())
    random.shuffle(basenames)

    # Calculate split points
    n = len(basenames)
    idx_val = int(n * ratio[0])
    idx_test = int(n * (ratio[0] + ratio[1]))

    splits = {
        'train': basenames[:idx_val],
        'val': basenames[idx_val:idx_test],
        'test': basenames[idx_test:]
    }

    # Move files to split folders
    for s, names in splits.items():
        img_out = os.path.join(output_dir, 'images', s)
        lbl_out = os.path.join(output_dir, 'labels', s)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for name in tqdm(names, desc=f"Creating {s} set"):
            for img_rel in groups[name]:
                shutil.copy(os.path.join(image_dir, img_rel),
                            os.path.join(img_out, os.path.basename(img_rel)))
                src_lbl = os.path.join(label_dir, name + '.txt')
                if os.path.exists(src_lbl):
                    shutil.copy(src_lbl, os.path.join(lbl_out, name + '.txt'))

    # Save filenames of the test set
    with open(os.path.join(output_dir, "testlist.txt"), "w") as f:
        for name in splits['test']:
            f.write(f"{name}\n")


if __name__ == "__main__":
    split_dataset(
        image_dir='./dataset/raw/GC10-DET/images',
        label_dir='./dataset/raw/GC10-DET/labels',
        output_dir='./dataset/processed/GC10'
    )