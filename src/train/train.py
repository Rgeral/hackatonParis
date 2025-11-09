import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

def main():
	# Entraînement du modèle avec optimisations GPU
	# Tous les paramètres sont configurables via .env
	model_name = os.getenv('TRAIN_MODEL', 'yolov8s.pt')
	model = YOLO(model_name)
	
	model.train(
		data=os.getenv('TRAIN_DATA_YAML', 'data.yaml'),
		epochs=int(os.getenv('TRAIN_EPOCHS', '5')),
		imgsz=int(os.getenv('TRAIN_IMGSZ', '640')),
		batch=int(os.getenv('TRAIN_BATCH', '16')),           # Bon pour 8GB VRAM
		device=os.getenv('TRAIN_DEVICE', '0'),               # GPU NVIDIA (0, 1, etc.) ou 'cpu'
		workers=int(os.getenv('TRAIN_WORKERS', '8')),        # Multi-threading pour le data loading
		cache=os.getenv('TRAIN_CACHE', 'disk'),              # Cache sur disque (plus rapide que pas de cache)
		amp=os.getenv('TRAIN_AMP', 'true').lower() == 'true',  # Mixed precision (FP16) pour accélération GPU
		patience=int(os.getenv('TRAIN_PATIENCE', '50')),     # Early stopping si pas d'amélioration
		plots=True,                                           # Génère des graphiques de résultats
		save_period=int(os.getenv('TRAIN_SAVE_PERIOD', '5')),  # Sauvegarde checkpoints tous les N epochs
	)

if __name__ == "__main__":
	main()

# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16
