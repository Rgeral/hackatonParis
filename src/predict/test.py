"""
Test & Évaluation du modèle YOLO sur le Test Set
================================================

Ce script évalue la performance du modèle sur le dataset TEST (10% jamais vu).
Contrairement à predict.py qui fait de l'inférence visuelle, ce script calcule
les métriques précises : Precision, Recall, mAP@0.5, mAP@0.5:0.95.

Usage:
    python src/predict/test.py

Configuration:
    - Modifier le fichier .env pour changer les chemins
    - Par défaut utilise: Datasets/splited_dataset/test/
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from ultralytics import YOLO


def load_config():
    """
    Charge la configuration depuis le fichier .env
    
    Returns:
        dict: Configuration avec tous les paramètres
    """
    # Charger les variables d'environnement
    load_dotenv()
    
    config = {
        'model_path': os.getenv('MODEL_PATH', 'runs/detect/train/weights/best.pt'),
        'data_yaml': os.getenv('DATA_YAML_PATH', 'data.yaml'),
        'test_images': os.getenv('TEST_IMAGES_PATH', 'Datasets/splited_dataset/test/images'),
        'test_labels': os.getenv('TEST_LABELS_PATH', 'Datasets/splited_dataset/test/labels'),
        'device': os.getenv('DEVICE', '0'),
        'conf_threshold': float(os.getenv('CONF_THRESHOLD', '0.25')),
        'iou_threshold': float(os.getenv('IOU_THRESHOLD', '0.45')),
        'save_json': os.getenv('SAVE_JSON', 'false').lower() == 'true',
    }
    
    return config


def validate_paths(config):
    """
    Vérifie que tous les chemins nécessaires existent
    
    Args:
        config: Dictionnaire de configuration
        
    Returns:
        bool: True si tous les chemins sont valides
    """
    print("🔍 Vérification des chemins...")
    
    paths_to_check = [
        ('Modèle', config['model_path']),
        ('Data YAML', config['data_yaml']),
        ('Images test', config['test_images']),
        ('Labels test', config['test_labels']),
    ]
    
    all_valid = True
    
    for name, path in paths_to_check:
        if Path(path).exists():
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name} introuvable: {path}")
            all_valid = False
    
    return all_valid


def print_config(config):
    """
    Affiche la configuration utilisée
    
    Args:
        config: Dictionnaire de configuration
    """
    print("\n" + "="*70)
    print("⚙️  CONFIGURATION")
    print("="*70)
    print(f"Modèle          : {config['model_path']}")
    print(f"Data YAML       : {config['data_yaml']}")
    print(f"Images test     : {config['test_images']}")
    print(f"Labels test     : {config['test_labels']}")
    print(f"Device          : {config['device']}")
    print(f"Conf Threshold  : {config['conf_threshold']}")
    print(f"IoU Threshold   : {config['iou_threshold']}")
    print(f"Save JSON       : {config['save_json']}")
    print("="*70 + "\n")


def count_test_images(test_images_path):
    """
    Compte le nombre d'images dans le test set
    
    Args:
        test_images_path: Chemin vers les images de test
        
    Returns:
        int: Nombre d'images
    """
    test_path = Path(test_images_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    count = sum(
        len(list(test_path.glob(f'*{ext}'))) 
        for ext in image_extensions
    )
    
    return count


def run_evaluation(config):
    """
    Lance l'évaluation du modèle sur le test set
    
    Args:
        config: Dictionnaire de configuration
        
    Returns:
        metrics: Objet contenant les métriques
    """
    print("📊 Chargement du modèle...")
    model = YOLO(config['model_path'])
    
    # Compter les images de test
    num_test_images = count_test_images(config['test_images'])
    print(f"📸 Nombre d'images dans le test set: {num_test_images}")
    
    if num_test_images == 0:
        print("❌ Aucune image trouvée dans le test set!")
        return None
    
    print("\n🚀 Lancement de l'évaluation...")
    print("⏳ Cette opération peut prendre plusieurs minutes...\n")
    
    # Évaluation avec model.val()
    # split='test' indique d'utiliser le dossier test du data.yaml
    metrics = model.val(
        data=config['data_yaml'],
        split='test',
        device=config['device'],
        conf=config['conf_threshold'],
        iou=config['iou_threshold'],
        save_json=config['save_json'],
        plots=True,  # Génère les graphiques
    )
    
    return metrics


def print_metrics(metrics):
    """
    Affiche les métriques de manière formatée
    
    Args:
        metrics: Objet contenant les métriques YOLO
    """
    if metrics is None:
        return
    
    print("\n" + "="*70)
    print("📊 RÉSULTATS DE L'ÉVALUATION SUR LE TEST SET")
    print("="*70)
    
    # Métriques Box (détection)
    box_metrics = metrics.box
    
    print("\n🎯 Métriques de Détection:")
    print(f"  • Precision       : {box_metrics.mp:.4f} ({box_metrics.mp*100:.2f}%)")
    print(f"  • Recall          : {box_metrics.mr:.4f} ({box_metrics.mr*100:.2f}%)")
    print(f"  • mAP@0.5         : {box_metrics.map50:.4f} ({box_metrics.map50*100:.2f}%)")
    print(f"  • mAP@0.5:0.95    : {box_metrics.map:.4f} ({box_metrics.map*100:.2f}%)")
    
    # Interprétation
    print("\n📈 Interprétation:")
    
    if box_metrics.map50 >= 0.9:
        print("  🌟 Excellent ! Le modèle détecte très bien les drones.")
    elif box_metrics.map50 >= 0.7:
        print("  ✅ Bon ! Performance satisfaisante.")
    elif box_metrics.map50 >= 0.5:
        print("  ⚠️  Acceptable. Objectif hackathon atteint (>50%).")
    else:
        print("  ❌ Faible. Nécessite plus d'entraînement ou de données.")
    
    # Objectif hackathon
    print(f"\n🎯 Objectif Hackathon (mAP@0.5 > 50%): ", end="")
    if box_metrics.map50 >= 0.5:
        print("✅ ATTEINT !")
    else:
        print(f"❌ Non atteint ({box_metrics.map50*100:.2f}%)")
    
    print("\n" + "="*70)
    
    # Conseils d'amélioration
    if box_metrics.map50 < 0.7:
        print("\n💡 Suggestions d'amélioration:")
        if box_metrics.mp < 0.7:
            print("  • Precision faible → Réduire les faux positifs")
            print("    - Augmenter conf_threshold dans .env")
            print("    - Ajouter plus d'images négatives")
        if box_metrics.mr < 0.7:
            print("  • Recall faible → Détecter plus de drones")
            print("    - Augmenter le nombre d'époques")
            print("    - Améliorer l'augmentation de données")
        print("  • Entraîner plus longtemps (epochs)")
        print("  • Essayer un modèle plus grand (yolov8m.pt, yolov8l.pt)")
        print("="*70)


def save_summary(config, metrics, output_file='test_results.txt'):
    """
    Sauvegarde un résumé des résultats dans un fichier
    
    Args:
        config: Configuration utilisée
        metrics: Métriques obtenues
        output_file: Nom du fichier de sortie
    """
    if metrics is None:
        return
    
    box_metrics = metrics.box
    
    output_path = Path(output_file)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("RÉSULTATS DE L'ÉVALUATION - TEST SET\n")
        f.write("="*70 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Modèle: {config['model_path']}\n")
        f.write(f"  Dataset: {config['test_images']}\n")
        f.write(f"  Device: {config['device']}\n")
        f.write(f"  Conf Threshold: {config['conf_threshold']}\n")
        f.write(f"  IoU Threshold: {config['iou_threshold']}\n\n")
        
        f.write("Métriques:\n")
        f.write(f"  Precision: {box_metrics.mp:.4f} ({box_metrics.mp*100:.2f}%)\n")
        f.write(f"  Recall: {box_metrics.mr:.4f} ({box_metrics.mr*100:.2f}%)\n")
        f.write(f"  mAP@0.5: {box_metrics.map50:.4f} ({box_metrics.map50*100:.2f}%)\n")
        f.write(f"  mAP@0.5:0.95: {box_metrics.map:.4f} ({box_metrics.map*100:.2f}%)\n\n")
        
        f.write("Objectif Hackathon (>50%): ")
        if box_metrics.map50 >= 0.5:
            f.write("✅ ATTEINT\n")
        else:
            f.write("❌ Non atteint\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"\n💾 Résumé sauvegardé dans: {output_path.absolute()}")


def main():
    """
    Fonction principale
    """
    print("\n" + "="*70)
    print("🧪 TEST & ÉVALUATION - DATASET TEST (jamais vu)")
    print("="*70 + "\n")
    
    # Charger configuration
    config = load_config()
    
    # Afficher configuration
    print_config(config)
    
    # Valider les chemins
    if not validate_paths(config):
        print("\n❌ Erreur: Certains chemins sont invalides.")
        print("💡 Vérifiez votre fichier .env et les chemins des fichiers.")
        return
    
    # Lancer évaluation
    try:
        metrics = run_evaluation(config)
        
        # Afficher métriques
        print_metrics(metrics)
        
        # Sauvegarder résumé
        save_summary(config, metrics)
        
        print("\n✅ Évaluation terminée avec succès !")
        print("📁 Les graphiques sont sauvegardés dans: runs/detect/val/")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'évaluation: {e}")
        print("💡 Vérifiez que le modèle est bien entraîné et que les chemins sont corrects.")
        raise


if __name__ == "__main__":
    main()
