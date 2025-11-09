from ultralytics import YOLO

def main():
	# Entraînement du modèle avec optimisations GPU
	model = YOLO('yolov8s.pt')
	model.train(
		data='data.yaml',
		epochs=5,
		imgsz=640,
		batch=16,           # Bon pour 8GB VRAM
		device=0,           # GPU NVIDIA
		workers=8,          # Multi-threading pour le data loading
		cache='disk',       # Cache sur disque (plus rapide que pas de cache, sans utiliser 99GB RAM)
		amp=True,           # Mixed precision (FP16) pour accélération GPU
		patience=50,        # Early stopping si pas d'amélioration
		plots=True,         # Génère des graphiques de résultats
		save_period=5,      # Sauvegarde checkpoints tous les 5 epochs
	)

if __name__ == "__main__":
	main()

# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16
