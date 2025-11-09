from ultralytics import YOLO
import glob

def main():

	# Prédiction sur les images de validation avec le meilleur modèle entraîné
	best_model = YOLO('runs/detect/train/weights/best.pt')
	results = best_model.predict(
		source='Datasets/dataset/val/images',
		save=True,		# sauvegarde les images annotées
		device=0
	)
	print(f"Prédiction terminée. {len(results)} images traitées.")

	# # Affichage des images prédites (adapté pour script Python)
	# print("\nImages prédites sauvegardées dans runs/detect/predict/")
	# predicted_images = glob.glob('runs/detect/predict/*.jpg')[:10]
	# for img_path in predicted_images:
	# 	print(f"  - {img_path}")
	# print(f"\nTotal: {len(predicted_images)} images affichées (limite: 10 premières)\n")

	#affichage des metriques
	for i, result in enumerate(results):
		print(f"\nImage {i+1}: {result.path}")
		if result.boxes is not None and len(result.boxes) > 0:
			print(f"  Nombre de détections: {len(result.boxes)}")
			for j, box in enumerate(result.boxes):
				cls = int(box.cls[0])
				conf = float(box.conf[0])
				class_name = result.names[cls]
				print(f"    Détection {j+1}: {class_name} (confiance: {conf:.2f})")
		else:
			print("  Aucune détection")
	
	# Pour obtenir les métriques (précision, rappel, mAP), utilisez model.val()
	print("\n=== Validation pour obtenir les métriques ===")
	metrics = best_model.val(data='data.yaml')
	print(f"Précision (P): {metrics.box.p[0]:.4f}")
	print(f"Rappel (R): {metrics.box.r[0]:.4f}")
	print(f"mAP@0.5: {metrics.box.map50:.4f}")
	print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
if __name__ == "__main__":
	main()

# !yolo task=detect mode=predict model=runs/detect/train3/weights/best.pt source=Datasets/dataset/val/images save=True
