import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("STEP 1: IMAGE DETECTION")

img = cv2.imread("bus.jpg")
results = model(img)
annotated = results[0].plot()

cv2.imshow("Image Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("STEP 2: CAMERA DETECTION")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Camera Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()