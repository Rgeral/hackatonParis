import os
import glob
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_optimal_predict_device():
	"""
	Auto-detect best device for prediction
	
	Returns:
		str: Device string ('0', 'cpu', '0,1', etc.)
	"""
	device_config = os.getenv('PREDICT_DEVICE', 'auto')
	
	# If user specified a device, use it
	if device_config != 'auto':
		return device_config
	
	# Auto-detection
	try:
		import torch
		
		if torch.cuda.is_available():
			gpu_name = torch.cuda.get_device_name(0)
			print(f"✅ GPU detected: {gpu_name}")
			return '0'
		else:
			print("⚠️  No GPU detected, using CPU")
			return 'cpu'
	except ImportError:
		print("⚠️  PyTorch not available, using CPU")
		return 'cpu'


def main():
	# Prediction on test images with the best trained model
	model_path = os.getenv('PREDICT_MODEL_PATH', 'runs/detect/train/weights/best.pt')
	best_model = YOLO(model_path)
	
	# Auto-detect device
	device = get_optimal_predict_device()
	
	# Get current working directory to ensure results are saved in the right place
	project_dir = os.getcwd()
	
	result = best_model.predict(
		source=os.getenv('PREDICT_SOURCE', 'Datasets/splited_dataset/val/images'),
		device=device,
		conf=float(os.getenv('PREDICT_CONF', '0.25')),
		save=True,
		project=project_dir,
		name='runs/detect/predict'
	)

	# Display metrics
	for i, res in enumerate(result):
		print(f"\nImage {i+1}: Original shape: {res.orig_shape}")
		if res.boxes is not None:
			print(f"Detected objects: {len(res.boxes)}")
			for box in res.boxes:
				cls_id = int(box.cls.item())
				confidence = float(box.conf.item())
				bbox = box.xyxy.tolist()[0]  # Get first (and only) box coordinates
				print(f"  Class: {cls_id}, Confidence: {confidence:.2f}, Box: {bbox}")
		else:
			print("No objects detected")
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
