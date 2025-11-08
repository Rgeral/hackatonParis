from ultralytics import YOLO
import glob

def main():

	# Prédiction sur les images de validation avec le meilleur modèle entraîné
	best_model = YOLO('runs/detect/train3/weights/best.pt')
	results = best_model.predict(
		source='Datasets/dataset/val/images',
		save=True,		# sauvegarde les images annotées
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

# !yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=Datasets/dataset/val/images save=True
