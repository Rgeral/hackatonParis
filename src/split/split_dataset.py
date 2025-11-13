"""
Split dataset en train/val/test pour entraînement YOLO
- Train: 70% (avec augmentations futures)
- Val: 20% (sans augmentation)
- Test: 10% (sans augmentation)

Usage:
    python src/split/split_dataset.py
"""

import os
import random
import shutil
from pathlib import Path
from typing import Tuple, List

import yaml
from tqdm import tqdm
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()


def create_directory_structure(output_dir: Path) -> dict:
    """
    Crée l'arborescence train/val/test avec images et labels
    
    Returns:
        dict: Dictionnaire des chemins créés
    """
    splits = ["train", "val", "test"]
    paths = {}
    
    for split in splits:
        split_dir = output_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        paths[split] = {
            "images": images_dir,
            "labels": labels_dir
        }
    
    return paths


def get_image_label_pairs(
    images_dir: Path,
    labels_dir: Path
) -> List[Tuple[Path, Path]]:
    """
    Récupère toutes les paires (image, label) valides
    
    Args:
        images_dir: Dossier contenant les images
        labels_dir: Dossier contenant les labels
    
    Returns:
        Liste de tuples (image_path, label_path)
    """
    pairs = []
    
    # Trouver toutes les images
    image_extensions = [".jpg", ".jpeg", ".png"]
    
    for ext in image_extensions:
        for img_path in images_dir.glob(f"*{ext}"):
            # Chercher le label correspondant
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if label_path.exists():
                pairs.append((img_path, label_path))
            else:
                print(f"⚠️  Label manquant pour {img_path.name}")
    
    return pairs


def split_dataset(
    pairs: List[Tuple[Path, Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
) -> dict:
    """
    Split les paires en train/val/test
    
    Args:
        pairs: Liste de tuples (image, label)
        train_ratio: Proportion pour train
        val_ratio: Proportion pour val
        test_ratio: Proportion pour test
        seed: Seed pour reproductibilité
    
    Returns:
        dict: {"train": [...], "val": [...], "test": [...]}
    """
    # Vérifier que les ratios font 100%
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Les ratios doivent sommer à 1.0"
    
    # Mélanger aléatoirement avec seed fixe
    random.seed(seed)
    pairs_shuffled = pairs.copy()
    random.shuffle(pairs_shuffled)
    
    # Calculer les indices de split
    total = len(pairs_shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    splits = {
        "train": pairs_shuffled[:train_end],
        "val": pairs_shuffled[train_end:val_end],
        "test": pairs_shuffled[val_end:]
    }
    
    return splits


def copy_files(
    splits: dict,
    output_paths: dict
) -> dict:
    """
    Copie les fichiers dans les bons dossiers
    
    Args:
        splits: Dictionnaire des splits
        output_paths: Dictionnaire des chemins de sortie
    
    Returns:
        dict: Statistiques de copie
    """
    stats = {}
    
    for split_name, pairs in splits.items():
        print(f"\n📦 Copie {split_name.upper()}...")
        
        images_copied = 0
        labels_copied = 0
        
        for img_path, label_path in tqdm(pairs, desc=split_name):
            # Copier image
            dst_img = output_paths[split_name]["images"] / img_path.name
            shutil.copy2(img_path, dst_img)
            images_copied += 1
            
            # Copier label
            dst_label = output_paths[split_name]["labels"] / label_path.name
            shutil.copy2(label_path, dst_label)
            labels_copied += 1
        
        stats[split_name] = {
            "images": images_copied,
            "labels": labels_copied
        }
    
    return stats


def create_yaml_config(
    output_dir: Path,
    stats: dict,
    num_classes: int = 1,
    class_names: List[str] = None
) -> None:
    """
    Crée le fichier data.yaml pour YOLO
    
    Args:
        output_dir: Dossier racine du dataset
        stats: Statistiques du split
        num_classes: Nombre de classes
        class_names: Noms des classes
    """
    if class_names is None:
        class_names = ["drone"]
    
    config = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names
    }
    
    yaml_path = output_dir / "data.yaml"
    
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Fichier YAML créé : {yaml_path}")


def print_statistics(stats: dict, total_pairs: int) -> None:
    """
    Affiche les statistiques du split
    
    Args:
        stats: Statistiques de copie
        total_pairs: Nombre total de paires
    """
    print("\n" + "="*60)
    print("📊 STATISTIQUES DU SPLIT")
    print("="*60)
    
    for split_name, split_stats in stats.items():
        percentage = (split_stats["images"] / total_pairs) * 100
        print(f"\n{split_name.upper()}:")
        print(f"  Images: {split_stats['images']}")
        print(f"  Labels: {split_stats['labels']}")
        print(f"  Pourcentage: {percentage:.1f}%")
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_pairs} paires (image + label)")
    print("="*60)


def main():
    """
    Fonction principale
    """
    print("="*60)
    print("🔀 SPLIT DATASET TRAIN/VAL/TEST")
    print("="*60)
    
    # Configuration - Charger depuis les variables d'environnement
    positives_images_dir = Path(os.getenv('SPLIT_POSITIVE_IMAGES_DIR', 'Datasets/dataset/images'))
    positives_labels_dir = Path(os.getenv('SPLIT_POSITIVE_LABELS_DIR', 'Datasets/dataset/labels'))
    
    negatives_images_dir = Path(os.getenv('SPLIT_NEGATIVE_IMAGES_DIR', 'Datasets/dataset/negatives_images'))
    negatives_labels_dir = Path(os.getenv('SPLIT_NEGATIVE_LABELS_DIR', 'Datasets/dataset/negatives_labels'))
    
    output_dir = Path(os.getenv('SPLIT_OUTPUT_DIR', 'Datasets/splited_dataset'))
    
    # Vérifier existence des dossiers
    for dir_path in [positives_images_dir, positives_labels_dir,
                     negatives_images_dir, negatives_labels_dir]:
        if not dir_path.exists():
            print(f"❌ Dossier manquant : {dir_path}")
            return
    
    print(f"\n📂 Images positives : {positives_images_dir}")
    print(f"📂 Labels positifs : {positives_labels_dir}")
    print(f"📂 Images négatives : {negatives_images_dir}")
    print(f"📂 Labels négatifs : {negatives_labels_dir}")
    print(f"📂 Sortie : {output_dir}")
    
    # Récupérer les paires positives
    print("\n🔍 Récupération paires positives...")
    positive_pairs = get_image_label_pairs(
        positives_images_dir,
        positives_labels_dir
    )
    print(f"✅ {len(positive_pairs)} paires positives trouvées")
    
    # Récupérer les paires négatives
    print("\n🔍 Récupération paires négatives...")
    negative_pairs = get_image_label_pairs(
        negatives_images_dir,
        negatives_labels_dir
    )
    print(f"✅ {len(negative_pairs)} paires négatives trouvées")
    
    # Combiner toutes les paires
    all_pairs = positive_pairs + negative_pairs
    print(f"\n📊 TOTAL : {len(all_pairs)} paires")
    print(f"   - Positives : {len(positive_pairs)} ({len(positive_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"   - Négatives : {len(negative_pairs)} ({len(negative_pairs)/len(all_pairs)*100:.1f}%)")
    
    # Créer structure de dossiers
    print("\n📁 Création structure de dossiers...")
    output_paths = create_directory_structure(output_dir)
    print("✅ Structure créée")
    
    # Split dataset
    print("\n🔀 Split du dataset (seed=42 pour reproductibilité)...")
    splits = split_dataset(
        all_pairs,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    # Copier les fichiers
    stats = copy_files(splits, output_paths)
    
    # Créer fichier YAML
    create_yaml_config(
        output_dir,
        stats,
        num_classes=1,
        class_names=["drone"]
    )
    
    # Afficher statistiques
    print_statistics(stats, len(all_pairs))
    
    print("\n✅ Split terminé avec succès !")
    print(f"📂 Dataset disponible dans : {output_dir.absolute()}")


if __name__ == "__main__":
    main()
