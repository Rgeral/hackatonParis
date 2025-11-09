"""
Script pour préparer le dataset complet en combinant :
- Le train set augmenté (Datasets/augmented_train_dataset/)
- Le val set splitté (Datasets/splited_dataset/val/)
- Le test set splitté (Datasets/splited_dataset/test/)

Usage: python src/dataset/prepare_dataset.py
"""

from logging import config
import os
import shutil
from pathlib import Path

from polars import List
import yaml

# Couleurs pour l'affichage terminal
BLUE = '\033[0;34m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color


def prepare_dataset():
    """Prépare le dataset complet en copiant les données depuis les sources."""
    
    print(f"{BLUE}📂 Préparation du dataset complet...{NC}")
    
    # Définition des chemins source et destination
    sources = {
        'train': 'Datasets/augmented_train_dataset/',
        'val': 'Datasets/splited_dataset/val/',
        'test': 'Datasets/splited_dataset/test/'
    }
    
    destination_base = 'Datasets/final_dataset'

    # Vérifier que les sources existent
    print(f"{YELLOW}Vérification des sources...{NC}")
    for split_name, source_path in sources.items():
        if not os.path.exists(source_path):
            print(f"{RED}❌ Erreur: Le dossier source '{source_path}' n'existe pas !{NC}")
            print(f"{YELLOW}💡 Assurez-vous d'avoir exécuté 'make split' et 'make augmentation' avant.{NC}")
            return False
        else:
            # Compter les fichiers
            images_count = len(list(Path(source_path).rglob('*.jpg'))) + len(list(Path(source_path).rglob('*.png')))
            labels_count = len(list(Path(source_path).rglob('*.txt')))
            print(f"  ✓ {split_name}: {images_count} images, {labels_count} labels")

    print()

    # Créer les dossiers de destination
    print(f"{BLUE}Création des dossiers de destination...{NC}")
    for split_name in sources.keys():
        dest_path = os.path.join(destination_base, split_name)
        os.makedirs(dest_path, exist_ok=True)
        print(f"  ✓ {dest_path}")

    print()

    # Copier les données
    print(f"{BLUE}Copie des données...{NC}")
    for split_name, source_path in sources.items():
        dest_path = os.path.join(destination_base, split_name)
        
        print(f"{YELLOW}  Copie de {split_name}...{NC}")
        
        # Supprimer le contenu existant si présent
        if os.path.exists(dest_path):
            for item in os.listdir(dest_path):
                item_path = os.path.join(dest_path, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
        
        # Copier récursivement le contenu (pas le dossier parent)
        if os.path.isdir(source_path):
            for item in os.listdir(source_path):
                source_item = os.path.join(source_path, item)
                dest_item = os.path.join(dest_path, item)
                
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(source_item, dest_item)
        
        # Compter les fichiers copiés
        images_count = len(list(Path(dest_path).rglob('*.jpg'))) + len(list(Path(dest_path).rglob('*.png')))
        labels_count = len(list(Path(dest_path).rglob('*.txt')))
        print(f"  ✓ {split_name}: {images_count} images, {labels_count} labels copiés")
    
    print()
    print(f"{GREEN}✅ Dataset complet prêt dans {destination_base}/ !{NC}")
    
    # Résumé final
    print()
    print(f"{BLUE}📊 Résumé du dataset final:{NC}")
    for split_name in sources.keys():
        dest_path = os.path.join(destination_base, split_name)
        images_count = len(list(Path(dest_path).rglob('*.jpg'))) + len(list(Path(dest_path).rglob('*.png')))
        labels_count = len(list(Path(dest_path).rglob('*.txt')))
        print(f"  - {split_name}: {images_count} images, {labels_count} labels")
    
    return True


def create_yaml_config():
	"""
#	Crée le fichier data.yaml pour YOLO

	Args:
		output_dir: Dossier racine du dataset
		num_classes: Nombre de classes
		class_names: Noms des classes
	"""

	config = {
	"path": "Datasets/final_dataset",
	"train": "train/images",
	"val": "val/images",
	"test": "test/images",
	"nc": 1,
	"names": ["drone"]
	}

	yaml_path = "data.yaml"

	with open(yaml_path, "w") as f:
		yaml.dump(config, f, default_flow_style=False)

	print(f"\n✅ Fichier YAML créé : {yaml_path}")

def main():
    """Point d'entrée principal."""
    success = prepare_dataset()
    if not success:
        exit(1)
    create_yaml_config()

if __name__ == "__main__":
    main()
