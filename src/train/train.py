from ultralytics import YOLO

def main():
	# Entraînement du modèle
	model = YOLO('yolov8s.pt')
	model.train(
		data='data.yaml',
		epochs=2,
		imgsz=640,
		batch=16,
		device=0
	)

if __name__ == "__main__":
	main()

# !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=60 imgsz=640 #batch=16
