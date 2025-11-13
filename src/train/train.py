import os
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


def get_optimal_train_device():
	"""
	Auto-detect best device for training
	
	Returns:
		str: Device string ('0', 'cpu', '0,1', etc.)
	"""
	device_config = os.getenv('TRAIN_DEVICE', 'auto')
	
	# If user specified a device, use it
	if device_config != 'auto':
		return device_config
	
	# Auto-detection
	try:
		import torch
		
		if torch.cuda.is_available():
			num_gpus = torch.cuda.device_count()
			gpu_name = torch.cuda.get_device_name(0)
			print(f"✅ GPU detected for training: {gpu_name}")
			
			if num_gpus > 1:
				print(f"   {num_gpus} GPUs available, using all")
				return ','.join(str(i) for i in range(num_gpus))
			return '0'
		else:
			print("⚠️  No GPU detected, using CPU for training")
			return 'cpu'
	except ImportError:
		print("⚠️  PyTorch not available, using CPU")
		return 'cpu'


def main():
	# Model training with optimizations
	# All parameters are configurable via .env
	model_name = os.getenv('TRAIN_MODEL', 'yolov8s.pt')
	model = YOLO(model_name)
	
	# Auto-detect device
	device = get_optimal_train_device()
	
	# Get current working directory to ensure results are saved in the right place
	project_dir = os.getcwd()
	
	model.train(
		data=os.getenv('TRAIN_DATA_YAML', 'data.yaml'),
		epochs=int(os.getenv('TRAIN_EPOCHS', '5')),
		imgsz=int(os.getenv('TRAIN_IMGSZ', '640')),
		batch=int(os.getenv('TRAIN_BATCH', '16')),           # Good for 8GB VRAM
		device=device,                                        # GPU (0, 1, etc.) or 'cpu'
		workers=int(os.getenv('TRAIN_WORKERS', '8')),        # Multi-threading for data loading
		cache=os.getenv('TRAIN_CACHE', 'disk'),              # Disk cache (faster than no cache)
		amp=os.getenv('TRAIN_AMP', 'true').lower() == 'true',  # Mixed precision (FP16) for GPU acceleration
		patience=int(os.getenv('TRAIN_PATIENCE', '50')),     # Early stopping if no improvement
		plots=True,                                           # Generate result graphs
		save_period=int(os.getenv('TRAIN_SAVE_PERIOD', '5')),  # Save checkpoints every N epochs
		project=project_dir,                                  # Force save in current directory
		name='runs/detect/train'                             # Explicit path for results
	)

if __name__ == "__main__":
	main()

# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16
