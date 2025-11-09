import cv2
import albumentations as A
from tqdm import tqdm
from utils import load_image_and_labels, save_yolo_labels
import multiprocessing as mp
from functools import partial
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()


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


def process_single_image(base_name, img_dir, label_dir, transformations_to_test):
    """
    Process a single image with all its transformations.
    Returns: (generated_count, skipped_count, negative_count)
    """
    image, bboxes, class_labels = load_image_and_labels(base_name, img_dir=img_dir, label_dir=label_dir)
    
    if image is None:
        return 0, 0, 0
    
    is_negative = len(bboxes) == 0
    generated = 0
    skipped = 0
    negatives = 0
    
    for transform_name in transformations_to_test:
        aug_image, aug_bboxes, aug_class_labels = apply_augmentation(
            image, bboxes, class_labels, transform_type=transform_name
        )
        
        # For negatives, always save (no bbox check needed)
        # For positives, skip if all bboxes are lost
        if not is_negative and len(aug_bboxes) == 0:
            skipped += 1
            continue
        
        output_image = f"{img_dir}/{base_name}_{transform_name}.jpg"
        output_label = f"{label_dir}/{base_name}_{transform_name}.txt"
        
        cv2.imwrite(output_image, aug_image)
        save_yolo_labels(output_label, aug_bboxes, aug_class_labels)
        
        generated += 1
        if is_negative:
            negatives += 1
    
    return generated, skipped, negatives



if __name__ == "__main__":
    import glob
    import shutil
    
    # Transformations les plus utiles pour la détection de drones
    # Réduites de 7 à 4 pour optimiser le temps et la pertinence
    transformations_to_test = [
        "horizontal_flip",    # Très utile : symétrie naturelle
        "rotation",          # Très utile : drones à différents angles
        "brightness",        # Très utile : conditions d'éclairage variables
        "blur",             # Utile : simule mouvement ou mauvaise qualité caméra
    ]
    
    # Charger les chemins depuis les variables d'environnement
    source_dir = os.getenv('AUG_SOURCE_DIR', 'Datasets/splited_dataset/train')
    target_dir = os.getenv('AUG_TARGET_DIR', 'Datasets/augmented_train_dataset')
    
    # Create directory structure
    if os.path.exists(target_dir):
        print(f"Cleaning existing {target_dir}...")
        shutil.rmtree(target_dir)
    
    print(f"Creating {target_dir} structure...")
    os.makedirs(f"{target_dir}/images", exist_ok=True)
    os.makedirs(f"{target_dir}/labels", exist_ok=True)
    
    # Create classes.txt file
    with open(f"{target_dir}/classes.txt", 'w') as f:
        f.write("Drone\n")
    print(f"Created classes.txt with 'Drone'")
    
    # Copy images and labels
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
    images_without_labels = 0
    
    # Apply augmentations
    img_dir = f"{target_dir}/images"
    label_dir = f"{target_dir}/labels"
    
    print(f"\nAugmentation des images du dossier train...")
    print(f"Nombre d'images originales à augmenter: {len(base_names)}")
    print(f"Nombre de transformations par image: {len(transformations_to_test)}")
    print(f"Total attendu: {len(base_names) * len(transformations_to_test)} images augmentées\n")
    
    # Detect available CPU cores
    num_cores = mp.cpu_count()
    use_multiprocessing = num_cores > 1
    
    if use_multiprocessing:
        # Use 75% of cores to avoid overloading the system
        num_workers = max(1, int(num_cores * 0.75))
        print(f"🚀 Mode multiprocessing activé: utilisation de {num_workers}/{num_cores} cœurs CPU\n")
    else:
        print(f"⚙️  Mode séquentiel (1 cœur détecté)\n")
    
    try:
        if use_multiprocessing:
            # Multiprocessing mode with progress bar
            process_func = partial(process_single_image, 
                                  img_dir=img_dir, 
                                  label_dir=label_dir, 
                                  transformations_to_test=transformations_to_test)
            
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, base_names),
                    total=len(base_names),
                    desc="🔄 Progression",
                    unit="img"
                ))
            
            # Aggregate results
            for generated, skipped, negatives in results:
                total_generated += generated
                total_skipped += skipped
                images_without_labels += negatives
        
        else:
            # Sequential fallback mode
            for base_name in tqdm(base_names, desc="🔄 Progression", unit="img"):
                generated, skipped, negatives = process_single_image(
                    base_name, img_dir, label_dir, transformations_to_test
                )
                total_generated += generated
                total_skipped += skipped
                images_without_labels += negatives
    
    except Exception as e:
        print(f"\n⚠️  Erreur multiprocessing, basculement en mode séquentiel: {e}")
        # Fallback to sequential processing
        total_generated = 0
        total_skipped = 0
        images_without_labels = 0
        
        for base_name in tqdm(base_names, desc="🔄 Progression (mode secours)", unit="img"):
            generated, skipped, negatives = process_single_image(
                base_name, img_dir, label_dir, transformations_to_test
            )
            total_generated += generated
            total_skipped += skipped
            images_without_labels += negatives
    
    print(f"\n✅ Terminé: {total_generated} images augmentées générées ({images_without_labels} depuis négatives); {total_skipped} ignorées")
