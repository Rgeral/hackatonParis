from ultralytics import YOLO
import glob

def main():
    # Entraînement du modèle
    # model = YOLO('yolov8s.pt')
    # model.train(
    #     data='data.yaml',
    #     epochs=60,
    #     imgsz=640,
    #     batch=16,
    #     device=0
    # )

    # Prédiction sur les images de validation avec le meilleur modèle entraîné
    best_model = YOLO('runs/detect/train3/weights/best.pt')
    results = best_model.predict(
        source='data/val/images',
        save=True,         # sauvegarde les images annotées
        device=0
    )
    print(f"Prédiction terminée. {len(results)} images traitées.")

    # Affichage des images prédites (adapté pour script Python)
    print("\nImages prédites sauvegardées dans runs/detect/predict/")
    predicted_images = glob.glob('runs/detect/predict/*.jpg')[:10]
    for img_path in predicted_images:
        print(f"  - {img_path}")
    print(f"\nTotal: {len(predicted_images)} images affichées (limite: 10 premières)\n")

if __name__ == "__main__":
    main()
# !pip install -r requirements.txt
# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16


