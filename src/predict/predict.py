import glob

from ultralytics import YOLO


def main():

	# Validation sur les images de test avec le meilleur modèle entraîné
	best_model = YOLO('runs/detect/train/weights/best.pt')
	result = best_model.predict(
		source='Datasets/splited_dataset/val/images',
		device=0,
		save=True
	)

	# Affichage des metriques
	for i, res in enumerate(result):
		print(f"\nImage {i+1}: {res.orig_shape} -> {res.shape}")
		print(f"Nombre d'objets détectés: {len(res.boxes)}")
		for box in res.boxes:
			print(f"  Classe: {int(box.cls)}, Confiance: {box.conf:.2f}, Boîte: {box.xyxy.tolist()}")
	# Utiliser val() au lieu de predict() pour obtenir les métriques
	# metrics = best_model.val(
	# 	data='data.yaml',
	# 	split='test',  # Utiliser le split test au lieu de val
	# 	device=0
	# )
	
	# print("\n" + "="*60)
	# print("📊 MÉTRIQUES SUR LE DATASET TEST")
	# print("="*60)
	# print(f"Précision (Precision): {metrics.box.mp:.4f}")
	# print(f"Rappel (Recall): {metrics.box.mr:.4f}")
	# print(f"mAP@0.5: {metrics.box.map50:.4f}")
	# print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
	# print("="*60)


if __name__ == "__main__":
	main()

# !yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=Datasets/dataset/val/images save=True
