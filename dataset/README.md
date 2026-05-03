# Dataset Organization and Usage

This directory documents how datasets are organized and used in the AGE-YOLO project.
Two alternative dataset usage schemes are supported:

1. Preparing datasets from the original raw sources.
2. Directly using the processed datasets provided on Hugging Face.

Both schemes are compatible with the provided training and evaluation code.


------------------------------------------------------------
1. Directory Structure Overview
------------------------------------------------------------

The recommended dataset directory structure is:

```text
dataset/
├── raw/
│   ├── GC10/
│   ├── NEU/
│   └── HRIPCB/
│
├── processed/
│   ├── GC10/
│   ├── NEU/
│   └── HRIPCB/
│
├── gc10_data.yaml
├── neu_data.yaml
├── hripcb_data.yaml
│
└── testlist_gc10.txt   (optional, raw-data preparation only)
```

------------------------------------------------------------
2. Using Original Raw Datasets
------------------------------------------------------------

This scheme is intended for users who wish to reproduce the complete data
preparation pipeline starting from the original datasets.


2.1 Original Dataset Sources
----------------------------

The original datasets can be obtained from the following links:

GC10-DET:
https://www.kaggle.com/datasets/alex000kim/gc10det

NEU-DET:
https://universe.roboflow.com/park-sung-ho/neu-det-object-detection

HRIPCB:
https://www.kaggle.com/datasets/akhatova/pcb-defects


2.2 Raw Dataset Placement
-------------------------

After downloading the original datasets, organize the files as follows:

dataset/raw/DATASET_NAME/
├── images/        # all original images
└── annotations/   # original annotations (XML or other formats)

Notes:

- GC10-DET and HRIPCB datasets are provided in XML format and require conversion
  to YOLO format.
- NEU-DET is already provided in YOLO format with an official train/val/test split.
- For datasets with irregular original structures (e.g., GC10-DET), users only
  need to manually collect all images and annotation files into the directories above.


2.3 Special Note on HRIPCB Rotation Images
------------------------------------------

The original HRIPCB dataset provides an additional `rotation/` directory created
by the dataset authors.

Important clarifications:

- The `rotation/` directory contains only rotated images.
- These rotation images do NOT include corresponding annotation files.
- Image filenames are identical to the original images.

If using the original HRIPCB dataset:

1. First split the original (non-rotated) images into train / val / test sets.
2. For the training set only:
   - Select rotation images with the same base filenames as training images.
   - Rename the rotation images to avoid filename conflicts.
   - Generate corresponding annotations by copying the original XML files
     and converting them using the provided script.
3. Place the generated rotation samples into the training set.
4. Apply further data augmentation to the training set if needed.

Rotation images are NOT required and are NOT included in the processed datasets
provided by this project.


2.4 Annotation Conversion (GC10-DET and HRIPCB)
-----------------------------------------------

XML annotations must be converted to YOLO format using:

tools/xml2yolo.py

After conversion, labels should be placed under:

dataset/raw/DATASET_NAME/labels/


2.5 Dataset Splitting
---------------------

After preparing images and labels, datasets can be split into
train / val / test subsets using:

tools/split_data.py

Optional test list:

- A custom test list file (e.g., testlist_gc10.txt) may be placed under:
  dataset/
- Each line contains one image name without file extension.
- This test list is ONLY used during raw data preparation.
- It is NOT part of Ultralytics YAML configuration.


2.6 Data Augmentation
---------------------

Data augmentation is applied to the training set only.

Augmentation is performed using:

tools/augment_train.py

Validation and test sets must remain unchanged.

------------------------------------------------------------
3. Using Processed Datasets from Hugging Face (Recommended)
------------------------------------------------------------

Processed datasets are provided on Hugging Face:

https://huggingface.co/datasets/YYYHHHH993/AGE-YOLO


3.1 Processed Dataset Description
---------------------------------

Each processed dataset:

- Is already split into train / val / test subsets.
- Is fully converted into YOLO format.
- Does NOT include any offline data augmentation.

The directory structure is:

```text
dataset/processed/DATASET_NAME/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

3.2 Usage Notes
---------------

- Processed datasets can be directly used for evaluation.
- For training, users may optionally apply augmentation to the training set
  using the provided augmentation script.
- No further preprocessing is required for testing.


------------------------------------------------------------
4. Dataset Configuration Files
------------------------------------------------------------

Each dataset has a corresponding YAML configuration file located under:

dataset/

Examples:

- gc10.yaml
- neu.yaml
- hripcb.yaml

These YAML files specify dataset paths and class names only.
They do NOT control data augmentation or custom test lists.


------------------------------------------------------------
5. Dataset Processing Scripts
------------------------------------------------------------

All dataset processing scripts are located under:

tools/

Directory structure:
```text
tools/
├── xml2yolo.py
├── split_data.py
└── augment_train.py
```
Script descriptions:

xml2yolo.py:
- Converts XML annotations to YOLO format.
- Used for GC10-DET and HRIPCB original datasets.

split_data.py:
- Splits images and labels into train / val / test subsets.
- Optionally follows a custom test list during raw data preparation.

augment_train.py:
- Applies data augmentation to the training set only.
- Validation and test sets remain unchanged.
