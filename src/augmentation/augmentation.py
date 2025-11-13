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
    
    transformations_to_test = [
        "horizontal_flip",    # Very useful: natural symmetry
        "rotation",          # Very useful: drones at different angles
        "brightness",        # Very useful: variable lighting conditions
        "blur",             # Useful: simulates motion or low camera quality
    ]
    
    # Load paths from environment variables
    source_dir = os.getenv('AUG_SOURCE_DIR', 'Datasets/splited_dataset/train')
    target_dir = os.getenv('AUG_TARGET_DIR', 'Datasets/augmented_train_dataset')
    
    # Subdirectory names (configurable)
    images_subdir = os.getenv('AUG_IMAGES_SUBDIR', 'images')
    labels_subdir = os.getenv('AUG_LABELS_SUBDIR', 'labels')
    
    # Create directory structure
    if os.path.exists(target_dir):
        print(f"Cleaning existing {target_dir}...")
        shutil.rmtree(target_dir)
    
    print(f"Creating {target_dir} structure...")
    os.makedirs(f"{target_dir}/{images_subdir}", exist_ok=True)
    os.makedirs(f"{target_dir}/{labels_subdir}", exist_ok=True)
    
    # Create classes.txt file
    with open(f"{target_dir}/classes.txt", 'w') as f:
        f.write("Drone\n")
    print(f"Created classes.txt with 'Drone'")
    
    # Copy images and labels
    print(f"Copying {source_dir} to {target_dir}...")
    for img_file in glob.glob(f"{source_dir}/{images_subdir}/*"):
        shutil.copy2(img_file, f"{target_dir}/{images_subdir}/")
    for label_file in glob.glob(f"{source_dir}/{labels_subdir}/*"):
        shutil.copy2(label_file, f"{target_dir}/{labels_subdir}/")
    print(f"Copy complete.")
    
    all_images = glob.glob(f"{target_dir}/{images_subdir}/*.jpg")
    transform_suffixes = ["_blur", "_brightness", "_horizontal_flip", "_vertical_flip", 
                         "_rotation", "_scale", "_perspective", "_noise"]
    
    original_images = [img for img in all_images 
                      if not any(suffix in img for suffix in transform_suffixes)]
    
    base_names = [os.path.splitext(os.path.basename(img))[0] for img in original_images]
    
    total_generated = 0
    total_skipped = 0
    images_without_labels = 0
    
    # Apply augmentations
    img_dir = f"{target_dir}/{images_subdir}"
    label_dir = f"{target_dir}/{labels_subdir}"
    
    print(f"\nAugmenting images from train folder...")
    print(f"Original images to augment: {len(base_names)}")
    print(f"Transformations per image: {len(transformations_to_test)}")
    print(f"Expected total: {len(base_names) * len(transformations_to_test)} augmented images\n")
    
    # ========================================
    # Multiprocessing configuration
    # ========================================
    
    # Load multiprocessing settings from environment
    multiprocessing_mode = os.getenv('AUG_MULTIPROCESSING', 'auto')
    cpu_usage_percent = float(os.getenv('AUG_CPU_USAGE', '0.75'))
    workers_config = os.getenv('AUG_WORKERS', 'auto')
    
    # Detect available CPU cores
    num_cores = mp.cpu_count()
    
    # Determine if multiprocessing should be used
    if multiprocessing_mode == 'auto':
        use_multiprocessing = num_cores > 1
    elif multiprocessing_mode == 'enabled':
        use_multiprocessing = True
    else:  # 'disabled'
        use_multiprocessing = False
    
    if use_multiprocessing:
        # Smart worker count calculation based on CPU cores
        if workers_config == 'auto':
            if num_cores <= 2:
                num_workers = 1  # Single core for small systems
            elif num_cores <= 4:
                num_workers = num_cores - 1  # Leave 1 core free
            else:
                num_workers = max(1, int(num_cores * cpu_usage_percent))  # Use percentage
        else:
            num_workers = max(1, int(workers_config))
        
        # Calculate optimal chunksize for better load balancing
        if len(base_names) < 100:
            chunksize = 1  # No chunking for small datasets
        else:
            chunksize = max(1, len(base_names) // (num_workers * 4))
        
        print(f"🚀 Multiprocessing enabled: {num_workers}/{num_cores} CPU cores")
        print(f"   Strategy: {cpu_usage_percent*100:.0f}% CPU usage, chunksize={chunksize}\n")
    else:
        num_workers = 1
        chunksize = 1
        print(f"⚙️  Sequential mode (single core)\n")
    
    try:
        if use_multiprocessing:
            # Multiprocessing mode with progress bar and optimized chunking
            process_func = partial(process_single_image, 
                                  img_dir=img_dir, 
                                  label_dir=label_dir, 
                                  transformations_to_test=transformations_to_test)
            
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_func, base_names, chunksize=chunksize),
                    total=len(base_names),
                    desc="🔄 Progress",
                    unit="img"
                ))
            
            # Aggregate results
            for generated, skipped, negatives in results:
                total_generated += generated
                total_skipped += skipped
                images_without_labels += negatives
        
        else:
            # Sequential fallback mode
            for base_name in tqdm(base_names, desc="🔄 Progress", unit="img"):
                generated, skipped, negatives = process_single_image(
                    base_name, img_dir, label_dir, transformations_to_test
                )
                total_generated += generated
                total_skipped += skipped
                images_without_labels += negatives
    
    except Exception as e:
        print(f"\n⚠️  Multiprocessing error, falling back to sequential mode: {e}")
        # Fallback to sequential processing
        total_generated = 0
        total_skipped = 0
        images_without_labels = 0
        
        for base_name in tqdm(base_names, desc="🔄 Progress (fallback)", unit="img"):
            generated, skipped, negatives = process_single_image(
                base_name, img_dir, label_dir, transformations_to_test
            )
            total_generated += generated
            total_skipped += skipped
            images_without_labels += negatives
    
    print(f"\n✅ Completed: {total_generated} augmented images generated ({images_without_labels} from negatives); {total_skipped} skipped")
