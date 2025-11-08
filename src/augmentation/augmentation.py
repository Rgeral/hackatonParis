import cv2
import albumentations as A
from utils import load_image_and_labels, save_yolo_labels


def apply_augmentation(image, bboxes, class_labels, transform_type="horizontal_flip"):
    """Apply augmentation to image and bounding boxes."""
    transforms_dict = {
        "horizontal_flip": A.Compose([
            A.HorizontalFlip(p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "vertical_flip": A.Compose([
            A.VerticalFlip(p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "rotation": A.Compose([
            A.Rotate(limit=15, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "scale": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=0, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "perspective": A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "blur": A.Compose([
            A.Blur(blur_limit=7, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "noise": A.Compose([
            A.GaussNoise(var_limit=(1.0, 5.0), p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "brightness": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "weather": A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    }
    
    if transform_type not in transforms_dict:
        return image, bboxes, class_labels
    
    transform = transforms_dict[transform_type]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], class_labels



if __name__ == "__main__":
    transformations_to_test = [
        "horizontal_flip",
        "rotation",
        "scale",
        "perspective",
        "blur",
        "noise",
        "brightness",
    ]
    
    base_name = "image_1_20250902_005339_jpg.rf.7ed2497be4411029e3cc85b4e05dfa40"
    
    image, bboxes, class_labels = load_image_and_labels(base_name)
    
    if image is None or len(bboxes) == 0:
        print("❌ Impossible de charger l'image ou les labels")
        exit(1)
    
    print(f"✅ Image chargée avec {len(bboxes)} bbox(es)")
    
    for transform_name in transformations_to_test:
        aug_image, aug_bboxes, aug_class_labels = apply_augmentation(
            image, bboxes, class_labels, transform_type=transform_name
        )
        
        output_image = f"Datasets/img/{base_name}_{transform_name}.jpg"
        output_label = f"Datasets/labels/{base_name}_{transform_name}.txt"
        
        cv2.imwrite(output_image, aug_image)
        save_yolo_labels(output_label, aug_bboxes, aug_class_labels)
        
        print(f"✅ {transform_name}: {len(aug_bboxes)} bbox(es) sauvegardées")
    
    print(f"\n🎉 {len(transformations_to_test)} transformations terminées")
