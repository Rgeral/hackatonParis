import cv2
import albumentations as A
from utils import load_image_and_labels, save_yolo_labels


def apply_augmentation(image, bboxes, class_labels, transform_type="horizontal_flip"):
    bbox_params = A.BboxParams(
        format='yolo', 
        label_fields=['class_labels'],
        min_visibility=0.3,
        clip=True
    )
    
    transforms_dict = {
        "horizontal_flip": A.Compose([
            A.HorizontalFlip(p=1.0)
        ], bbox_params=bbox_params),
        
        "vertical_flip": A.Compose([
            A.VerticalFlip(p=1.0)
        ], bbox_params=bbox_params),
        
        "rotation": A.Compose([
            A.Rotate(limit=15, p=1.0)
        ], bbox_params=bbox_params),
        
        "scale": A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=0, p=1.0)
        ], bbox_params=bbox_params),
        
        "perspective": A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=1.0)
        ], bbox_params=bbox_params),
        
        "blur": A.Compose([
            A.Blur(blur_limit=7, p=1.0)
        ], bbox_params=bbox_params),
        
        "noise": A.Compose([
            A.GaussNoise(var_limit=(1.0, 5.0), p=1.0)
        ], bbox_params=bbox_params),
        
        "brightness": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)
        ], bbox_params=bbox_params),
        
        "weather": A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0)
        ], bbox_params=bbox_params),
    }
    
    if transform_type not in transforms_dict:
        return image, bboxes, class_labels
    
    transform = transforms_dict[transform_type]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], class_labels



if __name__ == "__main__":
    import os
    import glob
    
    transformations_to_test = [
        "horizontal_flip",
        "rotation",
        "scale",
        "perspective",
        "blur",
        "noise",
        "brightness",
    ]
    
    all_images = glob.glob("Datasets/img/*.jpg")
    transform_suffixes = ["_blur", "_brightness", "_horizontal_flip", "_vertical_flip", 
                         "_rotation", "_scale", "_perspective", "_noise", "_weather"]
    
    original_images = [img for img in all_images 
                      if not any(suffix in img for suffix in transform_suffixes)]
    
    base_names = [os.path.splitext(os.path.basename(img))[0] for img in original_images]
    
    print(f"🚀 Traitement de {len(base_names)} images avec {len(transformations_to_test)} transformations")
    print(f"📊 Total d'augmentations à générer : {len(base_names) * len(transformations_to_test)}\n")
    
    total_generated = 0
    total_skipped = 0
    
    for idx, base_name in enumerate(base_names, 1):
        print(f"\n[{idx}/{len(base_names)}] 📂 {base_name[:50]}...")
        
        image, bboxes, class_labels = load_image_and_labels(base_name)
        
        if image is None or len(bboxes) == 0:
            print(f"    ⚠️  Ignorée (image ou labels manquants)")
            continue
        
        print(f"    ✅ Chargée avec {len(bboxes)} bbox(es)")
        
        for transform_name in transformations_to_test:
            aug_image, aug_bboxes, aug_class_labels = apply_augmentation(
                image, bboxes, class_labels, transform_type=transform_name
            )
            
            if len(aug_bboxes) == 0:
                print(f"    ⚠️  {transform_name}: drone hors image (ignoré)")
                total_skipped += 1
                continue
            
            output_image = f"Datasets/img/{base_name}_{transform_name}.jpg"
            output_label = f"Datasets/labels/{base_name}_{transform_name}.txt"
            
            cv2.imwrite(output_image, aug_image)
            save_yolo_labels(output_label, aug_bboxes, aug_class_labels)
            
            total_generated += 1
        
        print(f"    ✅ {len(transformations_to_test)} transformations appliquées")
    
    print(f"\n{'='*70}")
    print(f"🎉 Terminé !")
    print(f"✅ {total_generated} images augmentées générées")
    if total_skipped > 0:
        print(f"⚠️  {total_skipped} augmentations ignorées (drone hors image)")
    print(f"📁 Images : Datasets/img/")
    print(f"📁 Labels : Datasets/labels/")
    print(f"{'='*70}")
