from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Run detection AND save output
results = model("https://ultralytics.com/images/bus.jpg", save=True)

results[0].show()