# Data Augmentation for YOLO

## What is YOLO format?

Each label file (`.txt`) contains one line per object:
```
class_id x_center y_center width height
```

Example:
```
0 0.20078125 0.51015625 0.021875 0.0390625
```

Where:
- `class_id = 0` (0 = drone detected)
- `x_center = 0.20078125` (center at 20% from left)
- `y_center = 0.51015625` (center at 51% from top)
- `width = 0.021875` (width = 2.2% of image)
- `height = 0.0390625` (height = 3.9% of image)

## Normalized coordinates

Coordinates are **normalized** (divided by image size) to get values between 0 and 1.

**Why?** So they work for any image size.

**Example:**
```
Image: 1024 × 768 pixels
Drone center: x=205 pixels, y=391 pixels

Normalized:
x_center = 205 / 1024 = 0.20
y_center = 391 / 768 = 0.51
```

## Why recalculate coordinates during augmentation?

When we transform an image (flip, rotate, zoom), the bounding box coordinates change!

**Example with horizontal flip:**
```
Original                    Flipped
┌──────────────┐           ┌──────────────┐
│  🚁          │           │          🚁  │
└──────────────┘           └──────────────┘

x_center: 0.20             x_center: 0.80
(20% from left)            (80% from left)
```

We can't just copy the original labels - we need to recalculate them.

## Available functions

### 1. `load_yolo_labels(label_path)`

Reads a YOLO label file and returns two lists.

**Arguments:**
- `label_path`: Path to the `.txt` file (e.g., `"dataset/labels/image.txt"`)

**Returns:**
- `bboxes`: List of bounding box coordinates `[[x, y, w, h], ...]`
- `class_labels`: List of class IDs `[0, 1, ...]`

**What it does:**
1. Opens the file
2. Reads line by line
3. Parses each line into 5 values
4. Validates coordinates (must be between 0 and 1)
5. Returns two separate lists

**Example:**
```python
bboxes, classes = load_yolo_labels("dataset/labels/image_1.txt")
# bboxes = [[0.20, 0.51, 0.02, 0.04]]
# classes = [0]
```

**Edge case:** If file is empty or doesn't exist, returns `[], []`

---

### 2. `save_yolo_labels(label_path, bboxes, class_labels)`

Writes bboxes and classes to a new YOLO label file.

**Arguments:**
- `label_path`: Path where to save the file (e.g., `"output/labels/image_aug.txt"`)
- `bboxes`: List of coordinates `[[x, y, w, h], ...]`
- `class_labels`: List of class IDs `[0, 1, ...]`

**What it does:**
1. Creates parent directory if needed
2. Opens file in write mode (overwrites if exists!)
3. Formats each line as: `"class_id x y w h"`
4. Writes to file with 8 decimal precision

**Example:**
```python
bboxes = [[0.5, 0.5, 0.2, 0.3]]
classes = [0]
save_yolo_labels("output/labels/image_aug.txt", bboxes, classes)
# Creates file with: "0 0.50000000 0.50000000 0.20000000 0.30000000"
```

**Warning:** This function OVERWRITES existing files!

---

### 3. `load_image(image_path)`

Loads an image using OpenCV.

**Arguments:**
- `image_path`: Path to the image (e.g., `"dataset/img/image.jpg"`)

**Returns:**
- `image`: Numpy array with shape `(height, width, 3)` in BGR format
- `None` if loading failed

**What it does:**
1. Checks if file exists
2. Loads image with `cv2.imread()`
3. Validates the image loaded successfully
4. Prints dimensions
5. Returns the image as numpy array

**Example:**
```python
image = load_image("dataset/img/image_1.jpg")
# image.shape = (640, 640, 3)
# Format: BGR (Blue, Green, Red)
```

**Note:** OpenCV uses BGR format, not RGB!

---

### 4. `load_image_and_labels(base_name, img_dir="dataset/img", label_dir="dataset/labels")`

Loads both image and labels together using just the base filename.

**Arguments:**
- `base_name`: Filename without extension or directory (e.g., `"image_1"`)
- `img_dir`: Directory containing images (default: `"dataset/img"`)
- `label_dir`: Directory containing labels (default: `"dataset/labels"`)

**Returns:**
- `image`: The loaded image
- `bboxes`: List of bounding boxes
- `class_labels`: List of class IDs

**What it does:**
1. Constructs full paths: `{img_dir}/{base_name}.jpg` and `{label_dir}/{base_name}.txt`
2. Calls `load_image()` and `load_yolo_labels()`
3. Validates everything loaded correctly
4. Returns all three together

**Why use this?**
- Simpler code when working with image-label pairs
- Ensures image and labels match
- Only need to specify filename once

**Example:**
```python
# Instead of:
image = load_image("dataset/img/image_1.jpg")
bboxes, classes = load_yolo_labels("dataset/labels/image_1.txt")

# Just do:
image, bboxes, classes = load_image_and_labels("image_1")
```

**Edge cases:**
- Empty label file → Returns `image, [], []`
- Missing label file → Returns `image, [], []`
- Missing image → Returns `None, [], []`

---

## Images without objects

It's **perfectly normal** to have images with empty label files!

```
dataset/labels/image_no_drone.txt
(empty file)
```

This represents a **negative example** - teaching the model what is NOT a drone.

A good dataset should have:
- 70% images WITH drones
- 30% images WITHOUT drones (to avoid false positives)

Our functions handle empty files gracefully:
```python
bboxes, classes = load_yolo_labels("empty_file.txt")
# Returns: [], []  (no error!)
```

---

## Usage example

```python
from augmentation import load_image_and_labels, save_yolo_labels

# Load existing data
image, bboxes, classes = load_image_and_labels("image_1")

# After augmentation (coordinates are adjusted automatically by albumentations)
new_image = transformed_image  # From albumentations
new_bboxes = [[0.80, 0.51, 0.02, 0.04]]  # Example

# Save to new file
save_yolo_labels("dataset_augmented/labels/image_1_aug.txt", new_bboxes, classes)
```

---

## Test the functions

```bash
python3 augmentation.py
```

This will run 4 tests:
1. Load a YOLO label file
2. Save and reload to verify format
3. Load an image and show dimensions
4. Load image + labels together (combined function)

