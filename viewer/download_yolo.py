from ultralytics import YOLO

# Example: load YOLOv8n (nano)
model = YOLO("yolov8x-seg.pt")

model.export(format="onnx")  
