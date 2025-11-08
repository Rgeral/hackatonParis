# Data Augmentation for YOLO — Quick reference

This file documents the main helpers in `src/augmentation/utils.py` and `src/augmentation/augmentation.py`.

## YOLO label format
Each label line should be: `class_id x_center y_center width height` (all values normalized to [0,1]).

## Main helpers (what and why)

- `utils.load_yolo_labels(label_path) -> (bboxes, class_labels)`
  - What: Read a YOLO `.txt` file and return `bboxes` = [[x,y,w,h], ...] and `class_labels`.
  - Why: Centralizes parsing and validation of label files to ensure consistent, normalized coordinates.

- `utils.save_yolo_labels(label_path, bboxes, class_labels)`
  - What: Write YOLO-format labels to `label_path` (creates parent folder if needed).
  - Why: Keeps writing logic in one place (fixed format, precision) to avoid mistakes when saving augmented labels.

- `utils.load_image(image_path) -> image | None`
  - What: Load an image using OpenCV and return a BGR ndarray or `None` if not found.
  - Why: Encapsulates loading and simple validation so callers can handle missing files cleanly.

- `utils.load_image_and_labels(base_name, img_dir='Datasets/img', label_dir='Datasets/labels') -> (image, bboxes, class_labels)`
  - What: Convenience helper that loads `{img_dir}/{base_name}.jpg` and `{label_dir}/{base_name}.txt` together.
  - Why: Reduces boilerplate in the augmentation flow (path building and basic checks).

- `augmentation.apply_augmentation(image, bboxes, class_labels, transform_type) -> (img, bboxes, classes)`
  - What: Apply an Albumentations pipeline to image and bboxes and return the augmented image and updated bboxes.
  - Why: Centralizes augmentation parameters and bbox handling (so behavior is consistent and testable).

Supported transforms include: `horizontal_flip`, `vertical_flip`, `rotation`, `scale`, `perspective`, `blur`, `noise`, `brightness`, `weather`.

## Run the augmentation script
From the project root:

```bash
python3 src/augmentation/augmentation.py
```

The script will:
- Scan `Datasets/img/*.jpg` for original images (it ignores files that already have augmentation suffixes),
- Apply the configured transforms to each original image,
- Write augmented images to `Datasets/img/` and labels to `Datasets/labels/` using the suffix `_transformname`.

## How bboxes outside the image are handled
- The augmentation pipelines use `A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)`.
- `min_visibility=0.3` means a bbox is kept only if at least 30% of its area remains visible after transform.
- `clip=True` constrains coordinates to [0,1]. If all bboxes are dropped for a transform, that augmented image is skipped.

Adjust `min_visibility` (lower to keep more partial boxes, raise to be stricter) or reduce transform strength if many augmentations are skipped.

## Quick notes / next steps
- Use `visualize.py` (recommended) to draw bboxes on augmented images before training.
- Optionally add CLI flags to `augmentation.py` (e.g. `--min-visibility`, `--transforms`) to make runs configurable without editing code.

