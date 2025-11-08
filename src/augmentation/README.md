# Augmentation — quick reference

This module provides a small pipeline to create an augmented training set for YOLO (drone detection).

Key points
- Target folder: the script copies your split (e.g. `Datasets/splited_dataset/train`) into
  `Datasets/augmented_train_dataset/` and creates the structure:
  - `Datasets/augmented_train_dataset/images/`
  - `Datasets/augmented_train_dataset/labels/`
  - `Datasets/augmented_train_dataset/classes.txt` (contains the single class: `Drone`).
- Negative images (no bboxes): they are now augmented too. Empty `.txt` label files are created for augmented negatives.
- Augmentation summary: the script prints a single concise summary like
  `Done: <N> augmented images generated (<M> from negatives); <K> skips`.

YOLO label format
Each label line is: `class_id x_center y_center width height` (normalized to [0,1]).

Helpers (short)
- `utils.load_yolo_labels(path)`: parse `.txt` -> (bboxes, class_ids)
- `utils.save_yolo_labels(path, bboxes, class_ids)`: write labels (creates parent folder)
- `utils.load_image(path)`: read image with OpenCV or return `None`
- `utils.load_image_and_labels(base_name, img_dir, label_dir)`: returns `(image, bboxes, class_labels)`
- `augmentation.apply_augmentation(image, bboxes, class_labels, transform_type)`: applies a named pipeline and returns `(image, bboxes, class_labels)`

Transforms
- Geometric (bbox-aware): horizontal_flip, rotation, scale (Affine), perspective
- Photometric: blur, gauss noise, brightness
- Notes: vertical flip was removed; scale uses `Affine`; GaussNoise params fixed. Geometric transforms use
  `A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True)`.

Behavior notes
- For positive images (with boxes): if a transform removes all boxes (visibility < min_visibility), that augmented image is skipped.
- For negatives (no boxes): all transforms are applied and empty label files are written.

Run
From project root:

```bash
python3 src/augmentation/augmentation.py
```

Customizations
- Edit `src/augmentation/augmentation.py` to change the transforms list, `min_visibility` or output folder.
- Consider adding CLI flags (e.g. `--min-visibility`, `--transforms`) if you want runtime configuration.

Tips
- Use a small test run first to verify label handling.
- Use a visualize helper to render bboxes on augmented images before training.

