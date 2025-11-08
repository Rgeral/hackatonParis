import os
import cv2


def load_yolo_labels(label_path):
    """Load YOLO format labels from file."""
    bboxes = []
    class_labels = []

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                    0 <= width <= 1 and 0 <= height <= 1):
                continue
            
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
    
    return bboxes, class_labels


def save_yolo_labels(label_path, bboxes, class_labels):
    """Save YOLO format labels to file."""
    parent_dir = os.path.dirname(label_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
   
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            line = f"{class_id} {bbox[0]:.8f} {bbox[1]:.8f} {bbox[2]:.8f} {bbox[3]:.8f}\n"
            f.write(line)


def load_image(image_path):
    """Load image from file."""
    if not os.path.exists(image_path):
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    return image


def load_image_and_labels(base_name, img_dir="Datasets/img", label_dir="Datasets/labels"):
    """Load image and corresponding labels."""
    image_path = f"{img_dir}/{base_name}.jpg"
    label_path = f"{label_dir}/{base_name}.txt"
    
    image = load_image(image_path)
    bboxes, class_labels = load_yolo_labels(label_path)
    
    if image is None or len(bboxes) == 0:
        return None, [], []
    
    return image, bboxes, class_labels
