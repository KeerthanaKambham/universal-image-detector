import cv2
from ultralytics import YOLO
import os

# --------------------------------------
# Advanced Universal Image Detector
# --------------------------------------

print("Loading High Accuracy YOLOv8x model...")
model = YOLO("yolov8x.pt")   # Highest accuracy pretrained model

print("Model loaded successfully!\n")

# Get image path
image_path = input("Enter full image path: ").strip()

if not os.path.exists(image_path):
    print("Error: Image file not found.")
    exit()

print("\nRunning detection...")

# Run detection with confidence threshold
results = model(image_path, conf=0.5)  # Ignore weak detections below 50%

result = results[0]
annotated_image = result.plot()

# Show detected image
cv2.imshow("Advanced Universal Detection", annotated_image)

print("\nDetected Objects:")
print("=" * 50)

if result.boxes is not None and len(result.boxes) > 0:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        print(f"Object     : {label}")
        print(f"Confidence : {confidence:.2f}")
        print("-" * 50)
else:
    print("No objects detected.")

# Save output image
output_path = "advanced_detected_output.jpg"
cv2.imwrite(output_path, annotated_image)

print("\nDetection completed successfully!")
print(f"Output image saved as: {output_path}")

cv2.waitKey(0)
cv2.destroyAllWindows()
