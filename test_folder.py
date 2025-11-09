from ultralytics import YOLO
import glob
from pathlib import Path

def test_folder(folder_path: str):
    """
    Teste le modèle sur toutes les images d'un dossier
    
    Args:
        folder_path: Chemin vers le dossier contenant les images
    """
    # Charger le meilleur modèle
    model = YOLO('runs/detect/train/weights/best.pt')
    
    # Prédire sur toutes les images
    results = model.predict(
        source=folder_path,
        save=True,           # Sauvegarde les images avec détections
        conf=0.25,           # Seuil de confiance
        device=0,            # GPU
        save_txt=True        # Sauvegarde aussi les labels au format YOLO
    )
    
    # Statistiques
    total_images = len(results)
    images_with_drones = 0
    total_detections = 0
    
    for result in results:
        num_detections = len(result.boxes)
        total_detections += num_detections
        if num_detections > 0:
            images_with_drones += 1
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DES DÉTECTIONS")
    print("="*60)
    print(f"Total images testées: {total_images}")
    print(f"Images avec drones: {images_with_drones}")
    print(f"Images sans drones: {total_images - images_with_drones}")
    print(f"Total détections: {total_detections}")
    print(f"Moyenne par image: {total_detections/total_images:.2f}")
    print("="*60)
    print(f"\n✅ Résultats sauvegardés dans: runs/detect/predict/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_folder.py <chemin_dossier>")
        print("Exemple: python test_folder.py mes_photos/")
        sys.exit(1)
    
    test_folder(sys.argv[1])
