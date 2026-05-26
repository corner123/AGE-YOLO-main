import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from augment_train import AGE_YOLO_Augmentor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def resolve_path(value, base=PROJECT_ROOT):
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def stem_from_list_line(line):
    name = line.strip().replace("\\", "/").split("/")[-1]
    return Path(name).stem


def collect_originals(dataset_root):
    originals = {}
    for split_dir in sorted((dataset_root / "images").glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        for image_path in sorted(split_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTS or "_aug" in image_path.stem:
                continue
            label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
            if image_path.stem in originals:
                raise ValueError(f"Duplicate original image stem: {image_path.stem}")
            originals[image_path.stem] = (image_path, label_path)
    return originals


def read_test_stems(test_list):
    stems = []
    for line in test_list.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            stems.append(stem_from_list_line(line))
    seen = set()
    unique = []
    for stem in stems:
        if stem not in seen:
            unique.append(stem)
            seen.add(stem)
    return unique


def copy_split(stems, originals, output_root, split):
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for stem in tqdm(stems, desc=f"Copy {split}"):
        image_path, label_path = originals[stem]
        shutil.copy2(image_path, image_dir / image_path.name)
        labels = load_labels(label_path)
        write_labels(label_dir / f"{stem}.txt", deduplicate_labels(labels))


def load_labels(label_path):
    text = label_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return np.zeros((0, 5), dtype=np.float32)
    labels = np.loadtxt(str(label_path), dtype=np.float32)
    return labels.reshape(-1, 5)


def deduplicate_labels(labels):
    if labels.size == 0:
        return labels
    _, indices = np.unique(labels, axis=0, return_index=True)
    return labels[np.sort(indices)]


def write_labels(label_path, labels):
    with label_path.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(f"{int(row[0])} {' '.join(f'{x:.6f}' for x in row[1:])}\n")


def augment_train(output_root, multiplier, seed):
    if multiplier <= 1:
        return
    random.seed(seed)
    np.random.seed(seed)

    image_dir = output_root / "images" / "train"
    label_dir = output_root / "labels" / "train"
    augmentor = AGE_YOLO_Augmentor()
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and "_aug" not in p.stem)

    for image_path in tqdm(images, desc=f"Augment train x{multiplier}"):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        labels = load_labels(label_dir / f"{image_path.stem}.txt")

        for index in range(1, multiplier):
            aug_image, aug_labels = augmentor(image.copy(), labels.copy())
            out_stem = f"{image_path.stem}_aug{index}"
            cv2.imwrite(str(image_dir / f"{out_stem}{image_path.suffix.lower()}"), aug_image)
            write_labels(label_dir / f"{out_stem}.txt", aug_labels)


def write_yaml(output_root, source_yaml):
    names = DEFAULT_NAMES
    if source_yaml.exists():
        source_cfg = yaml.safe_load(source_yaml.read_text(encoding="utf-8", errors="replace")) or {}
        names = source_cfg.get("names", names)
    cfg = {
        "path": output_root.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    with (output_root / "steel_defect.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def write_list(output_root, name, stems):
    (output_root / name).write_text("\n".join(stems) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the NEU split using the paper/testlist protocol.")
    parser.add_argument("--dataset-root", default="dataset/NEU-DET", help="Existing NEU dataset root.")
    parser.add_argument("--test-list", default="dataset/testlist_neu.txt", help="Paper test list.")
    parser.add_argument("--output-root", default="dataset/NEU-DET-paper", help="New output dataset root.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for validation split and offline augmentation.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction of total original images.")
    parser.add_argument("--augment-mult", type=int, default=5, help="Total train multiplier; 5 means original + 4 augments.")
    parser.add_argument("--force", action="store_true", help="Replace output root if it already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = resolve_path(args.dataset_root)
    test_list = resolve_path(args.test_list)
    output_root = resolve_path(args.output_root)

    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not test_list.exists():
        raise FileNotFoundError(test_list)
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"{output_root} already exists. Use --force to rebuild it.")
        shutil.rmtree(output_root)

    originals = collect_originals(dataset_root)
    test_stems = read_test_stems(test_list)
    missing = sorted(set(test_stems) - set(originals))
    if missing:
        raise ValueError(f"Test list contains {len(missing)} missing images, e.g. {missing[:10]}")

    remaining = sorted(set(originals) - set(test_stems))
    rng = random.Random(args.seed)
    rng.shuffle(remaining)
    val_count = round(len(originals) * args.val_ratio)
    val_stems = sorted(remaining[:val_count])
    train_stems = sorted(remaining[val_count:])
    test_stems = sorted(test_stems)

    copy_split(train_stems, originals, output_root, "train")
    copy_split(val_stems, originals, output_root, "val")
    copy_split(test_stems, originals, output_root, "test")
    augment_train(output_root, args.augment_mult, args.seed)

    write_yaml(output_root, dataset_root / "steel_defect.yaml")
    write_list(output_root, "trainlist.txt", train_stems)
    write_list(output_root, "vallist.txt", val_stems)
    write_list(output_root, "testlist.txt", test_stems)

    train_total = sum(1 for p in (output_root / "images" / "train").iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"Prepared: {output_root}")
    print(f"Original split: train={len(train_stems)}, val={len(val_stems)}, test={len(test_stems)}")
    print(f"Train images after augmentation: {train_total}")


if __name__ == "__main__":
    main()
