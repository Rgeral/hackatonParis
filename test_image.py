from ultralytics import YOLO
from pathlib import Path

def test_single_image(image_path: str):
    """
    Teste le modèle sur une seule image
    
    Args:
        image_path: Chemin vers l'image à tester
    """
    # Charger le meilleur modèle
    model = YOLO('runs/detect/train/weights/best.pt')
    
    # Prédire
    results = model.predict(
        source=image_path,
        save=True,           # Sauvegarde l'image avec les détections
        conf=0.25,           # Seuil de confiance (ajustable)
        device=0             # GPU
    )
    
    # Afficher les résultats
    for result in results:
        print(f"\n📷 Image: {Path(image_path).name}")
        print(f"🔍 Nombre de détections: {len(result.boxes)}")
        
        for i, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = result.names[cls]
            
            print(f"  Détection #{i+1}:")
            print(f"    - Classe: {class_name}")
            print(f"    - Confiance: {conf:.2%}")
    
    print(f"\n✅ Image annotée sauvegardée dans: runs/detect/predict/")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_image.py <chemin_image>")
        print("Exemple: python test_image.py mon_drone.jpg")
        sys.exit(1)
    
    test_single_image(sys.argv[1])
