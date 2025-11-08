from ultralytics import YOLO

def main():
	# Entraînement du modèle
	model = YOLO('yolov8s.pt')
	model.train(
		data='data.yaml',
		epochs=5,
		imgsz=640,
		batch=16,
		device='cpu'
	)

if __name__ == "__main__":
	main()

# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16
