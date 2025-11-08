import glob

from ultralytics import YOLO


def main():

	# Validation sur les images de test avec le meilleur modèle entraîné
	best_model = YOLO('runs/detect/train/weights/best.pt')
	
	# Utiliser val() au lieu de predict() pour obtenir les métriques
	metrics = best_model.val(
		data='data.yaml',
		split='test',  # Utiliser le split test au lieu de val
		device=0
	)
	
	print("\n" + "="*60)
	print("📊 MÉTRIQUES SUR LE DATASET TEST")
	print("="*60)
	print(f"Précision (Precision): {metrics.box.mp:.4f}")
	print(f"Rappel (Recall): {metrics.box.mr:.4f}")
	print(f"mAP@0.5: {metrics.box.map50:.4f}")
	print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
	print("="*60)


if __name__ == "__main__":
	main()

# !yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=Datasets/dataset/val/images save=True
