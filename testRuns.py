from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 pre-trained model
model.train(data="C:/Users/user/Desktop/Real-time Object Detection/data.yaml", epochs=50, imgsz=640)

