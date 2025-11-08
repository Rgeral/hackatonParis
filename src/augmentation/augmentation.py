import cv2
import albumentations as A
from utils import load_image_and_labels, save_yolo_labels


def apply_augmentation(image, bboxes, class_labels, transform_type="horizontal_flip"):
    # bbox params used only for geometric transforms
    bbox_params = A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
        clip=True,
    )

    # geometric transforms (need bbox processing)
    geometric = {
        "horizontal_flip": A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params),
        "vertical_flip": A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbox_params),
        "rotation": A.Compose([A.Rotate(limit=15, p=1.0)], bbox_params=bbox_params),
        # ShiftScaleRotate suggests using Affine in newer albumentations
        "scale": A.Compose([
            A.Affine(scale=(0.7, 1.3), translate_percent=0.1, rotate=0, p=1.0)
        ], bbox_params=bbox_params),
        "perspective": A.Compose([A.Perspective(scale=(0.05, 0.1), p=1.0)], bbox_params=bbox_params),
    }

    # photometric / non-geometric transforms (no bbox processing needed)
    photometric = {
        "blur": A.Compose([A.Blur(blur_limit=7, p=1.0)]),
        "noise": A.Compose([A.GaussNoise(p=1.0)]),
        "brightness": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)]),
    }

    transforms_dict = {**geometric, **photometric}
    
    if transform_type not in transforms_dict:
        return image, bboxes, class_labels
    
    transform = transforms_dict[transform_type]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    return transformed['image'], transformed['bboxes'], class_labels



if __name__ == "__main__":
    import os
    import glob
    import shutil
    
    transformations_to_test = [
        "horizontal_flip",
        "rotation",
        "scale",
        "perspective",
        "blur",
        "noise",
        "brightness",
    ]
    
    source_dir = "Datasets/splited_dataset/train"
    target_dir = "Datasets/augmented_train_dataset"
    
    # Step 1: Create directory structure
    if os.path.exists(target_dir):
        print(f"Cleaning existing {target_dir}...")
        shutil.rmtree(target_dir)
    
    print(f"Creating {target_dir} structure...")
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/labels", exist_ok=True)
    
    # Step 2: Create classes.txt file
    with open(f"{target_dir}/classes.txt", 'w') as f:
        f.write("Drone\n")
    print(f"Created classes.txt with 'Drone'")
    
    # Step 3: Copy images and labels
    print(f"Copying {source_dir} to {target_dir}...")
    for img_file in glob.glob(f"{source_dir}/images/*"):
        shutil.copy2(img_file, f"{target_dir}/images/")
    for label_file in glob.glob(f"{source_dir}/labels/*"):
        shutil.copy2(label_file, f"{target_dir}/labels/")
    print(f"Copy complete.")
    
    all_images = glob.glob(f"{target_dir}/images/*.jpg")
    transform_suffixes = ["_blur", "_brightness", "_horizontal_flip", "_vertical_flip", 
                         "_rotation", "_scale", "_perspective", "_noise"]
    
    original_images = [img for img in all_images 
                      if not any(suffix in img for suffix in transform_suffixes)]
    
    base_names = [os.path.splitext(os.path.basename(img))[0] for img in original_images]
    
    total_generated = 0
    total_skipped = 0
    images_ignored = 0
    
    # Apply augmentations
    img_dir = f"{target_dir}/images"
    label_dir = f"{target_dir}/labels"
    
    for base_name in base_names:
        image, bboxes, class_labels = load_image_and_labels(base_name, img_dir=img_dir, label_dir=label_dir)

        if image is None or len(bboxes) == 0:
            images_ignored += 1
            continue

        for transform_name in transformations_to_test:
            aug_image, aug_bboxes, aug_class_labels = apply_augmentation(
                image, bboxes, class_labels, transform_type=transform_name
            )

            if len(aug_bboxes) == 0:
                total_skipped += 1
                continue

            output_image = f"{img_dir}/{base_name}_{transform_name}.jpg"
            output_label = f"{label_dir}/{base_name}_{transform_name}.txt"

            cv2.imwrite(output_image, aug_image)
            save_yolo_labels(output_label, aug_bboxes, aug_class_labels)

            total_generated += 1
    
    # concise summary
    print(f"Done: {total_generated} augmented images generated; {total_skipped} skips; {images_ignored} ignored images")
