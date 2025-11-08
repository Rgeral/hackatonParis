import os
import cv2  # OpenCV pour manipuler les images
import albumentations as A  # Pour les augmentations avec gestion des bboxes


def load_yolo_labels(label_path):
    bboxes = []           # Contiendra les coordonnées des boîtes
    class_labels = []     # Contiendra les IDs des classes

    
    # Ouvrir et lire le fichier ligne par ligne
    with open(label_path, 'r') as f:
        for line in f:
            # Enlever les espaces/retours à la ligne au début et fin
            line = line.strip()
            
            # Ignorer les lignes vides
            if not line:
                continue
            
            # Séparer la ligne en morceaux (split par espace)
            # "0 0.5 0.5 0.2 0.3" → ["0", "0.5", "0.5", "0.2", "0.3"]
            parts = line.split()
            
            # Vérifier qu'on a bien 5 valeurs
            if len(parts) != 5:
                print(f"⚠️  Ligne ignorée (format incorrect) : {line}")
                continue
            
            # Convertir les valeurs texte en nombres
            class_id = int(parts[0])      # class_id est un entier
            x_center = float(parts[1])    # coordonnées sont des décimaux
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Vérification de sécurité : les coordonnées doivent être entre 0 et 1
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                    0 <= width <= 1 and 0 <= height <= 1):
                print(f"⚠️  Coordonnées hors limites (ignorée) : {line}")
                continue
            
            # Ajouter aux listes
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(class_id)
    
    print(f"✅ Chargé {len(bboxes)} bounding box(es) depuis {os.path.basename(label_path)}")
    return bboxes, class_labels


def save_yolo_labels(label_path, bboxes, class_labels):
    """
    Sauvegarde les annotations au format YOLO dans un fichier .txt.
    
    C'est l'INVERSE de load_yolo_labels : on prend des données Python
    et on les écrit dans un fichier texte.
    
    Args:
        label_path: Chemin où créer le fichier (ex: "output/labels/image_aug.txt")
        bboxes: Liste de listes [[x_center, y_center, width, height], ...]
        class_labels: Liste des class_id [0, 1, 2, ...]
        
    Exemple :
        Si on a :
            bboxes = [[0.5, 0.5, 0.2, 0.3], [0.8, 0.2, 0.1, 0.15]]
            class_labels = [0, 1]
        
        Le fichier contiendra :
            0 0.50000000 0.50000000 0.20000000 0.30000000
            1 0.80000000 0.20000000 0.10000000 0.15000000
    """
    # Créer le dossier parent si nécessaire
    # Ex: si label_path = "output/labels/image.txt", crée "output/labels/"
    parent_dir = os.path.dirname(label_path)
    if parent_dir:  # Seulement si le chemin contient un dossier
        os.makedirs(parent_dir, exist_ok=True)
    
    # Ouvrir le fichier en mode écriture ('w' = write)
    # Si le fichier existe déjà, il sera écrasé !
    with open(label_path, 'w') as f:
        # Parcourir chaque bbox avec sa classe correspondante
        # zip() associe les éléments : (bbox[0] avec class[0], bbox[1] avec class[1], etc.)
        for bbox, class_id in zip(bboxes, class_labels):
            # Formater la ligne : "class_id x y w h"
            # .8f = 8 chiffres après la virgule pour la précision
            line = f"{class_id} {bbox[0]:.8f} {bbox[1]:.8f} {bbox[2]:.8f} {bbox[3]:.8f}\n"
            
            # Écrire la ligne dans le fichier
            f.write(line)
    
    print(f"✅ Sauvegardé {len(bboxes)} bounding box(es) dans {os.path.basename(label_path)}")


def load_image(image_path):
    # Vérifier que le fichier existe
    if not os.path.exists(image_path):
        print(f"❌ Image introuvable : {image_path}")
        return None
    
    # Charger l'image avec OpenCV
    image = cv2.imread(image_path)
    
    # Vérifier que le chargement a réussi
    if image is None:
        print(f"❌ Erreur lors du chargement de : {image_path}")
        return None
    
    # Récupérer les dimensions (hauteur, largeur, canaux)
    height, width = image.shape[:2]
    
    print(f"✅ Image chargée : {os.path.basename(image_path)}")
    print(f"   Dimensions : {width} × {height} pixels")
    
    return image


def load_image_and_labels(base_name, img_dir="Datasets/img", label_dir="Datasets/labels"):
    # Construire les chemins complets
    image_path = f"{img_dir}/{base_name}.jpg"
    label_path = f"{label_dir}/{base_name}.txt"
    
    # Charger l'image
    image = load_image(image_path)
    
    # Charger les labels
    bboxes, class_labels = load_yolo_labels(label_path)
    
    # Vérifier que tout est OK
    if image is None:
        print(f"⚠️  Image non chargée pour {base_name}")
        return None, [], []
    
    if len(bboxes) == 0:
        print(f"⚠️  Aucune bbox trouvée pour {base_name}")
    
    return image, bboxes, class_labels


def apply_augmentation(image, bboxes, class_labels, transform_type="horizontal_flip"):
    """
    Applique une transformation à l'image et ajuste automatiquement les bboxes
    
    Args:
        image: Image numpy array (BGR format OpenCV)
        bboxes: Liste de bboxes au format YOLO [[x, y, w, h], ...]
        class_labels: Liste des class_id [0, 1, ...]
        transform_type: Type de transformation à appliquer
        
    Returns:
        augmented_image, augmented_bboxes, class_labels
    """
    
    # Dictionnaire des transformations disponibles
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
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=0, p=1.0)  # Zoom + déplacement
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "perspective": A.Compose([
            A.Perspective(scale=(0.05, 0.1), p=1.0)  # Transformation projective/3D
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "blur": A.Compose([
            A.Blur(blur_limit=7, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "noise": A.Compose([
            A.GaussNoise(var_limit=(1.0, 5.0), p=1.0)  # Bruit ultra léger
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "brightness": A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1.0)  # ±20%
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
        
        "weather": A.Compose([
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
    }
    
    # Vérifier que la transformation existe
    if transform_type not in transforms_dict:
        print(f"❌ Transformation '{transform_type}' inconnue !")
        print(f"   Disponibles : {list(transforms_dict.keys())}")
        return image, bboxes, class_labels
    
    # Appliquer la transformation
    transform = transforms_dict[transform_type]
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    
    augmented_image = transformed['image']
    augmented_bboxes = transformed['bboxes']
    
    print(f"✨ Transformation '{transform_type}' appliquée")
    print(f"   Bboxes : {len(bboxes)} → {len(augmented_bboxes)}")
    
    return augmented_image, augmented_bboxes, class_labels


# ============================================================
# SECTION DE TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 TEST DES AUGMENTATIONS")
    print("=" * 70)
    
    # Transformations à tester
    transformations_to_test = [
        ("horizontal_flip", "📐 Flip horizontal"),
        ("rotation", "🔄 Rotation (±15°)"),
        ("scale", "🔍 Zoom caméra"),
        ("perspective", "📐 Perspective"),
        ("blur", "💨 Flou gaussien"),
        ("noise", "📡 Bruit ultra léger (1-5)"),
        ("brightness", "💡 Luminosité ±20%"),
    ]
    
    print(f"\n🎨 {len(transformations_to_test)} transformations à tester\n")
    
    # Charger l'image de test
    base_name = "image_1_20250901_230254_jpg.rf.148f935c5d8de1ed1d797f649b8a6b1d"
    print(f"📂 Chargement : {base_name}\n")
    
    image, bboxes, class_labels = load_image_and_labels(base_name)
    
    if image is None or len(bboxes) == 0:
        print("\n❌ Impossible de continuer (image ou labels manquants)")
        exit(1)
    
    print(f"\n📊 Labels originaux :")
    for i, (bbox, cls) in enumerate(zip(bboxes, class_labels)):
        print(f"   Bbox {i+1}: classe={cls}, x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}")
    
    # Tester chaque transformation
    for transform_name, transform_description in transformations_to_test:
        print("\n" + "=" * 70)
        print(f"{transform_description}")
        print("=" * 70)
        
        # Appliquer la transformation
        aug_image, aug_bboxes, aug_class_labels = apply_augmentation(
            image, bboxes, class_labels, transform_type=transform_name
        )
        
        # Afficher les coordonnées transformées
        print(f"\n📊 Labels après transformation :")
        for i, (bbox, cls) in enumerate(zip(aug_bboxes, aug_class_labels)):
            print(f"   Bbox {i+1}: classe={cls}, x={bbox[0]:.3f}, y={bbox[1]:.3f}, w={bbox[2]:.3f}, h={bbox[3]:.3f}")
        
        # Sauvegarder avec nom original + suffixe transformation
        # Ex: image_1_20250901_230254_jpg.rf.148f935c5d8de1ed1d797f649b8a6b1d_horizontal_flip.jpg
        output_image = f"Datasets/img/{base_name}_{transform_name}.jpg"
        output_label = f"Datasets/labels/{base_name}_{transform_name}.txt"
        
        cv2.imwrite(output_image, aug_image)
        save_yolo_labels(output_label, aug_bboxes, aug_class_labels)
        
        print(f"💾 Sauvegardé : {output_image}")
    
    print("\n" + "=" * 70)
    print("🎉 TESTS TERMINÉS !")
    print("=" * 70)
    print(f"📁 Résultats dans Datasets/img/ et Datasets/labels/\n")

