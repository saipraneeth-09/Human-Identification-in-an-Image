# Install the necessary libraries first
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained on COCO dataset)
model = YOLO("yolov8n.pt")  # You can use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read the first frame to set as background
ret, background_frame = cap.read()

if not ret:
    print("Error: Failed to capture initial frame.")
    cap.release()
    exit()

# Convert background to grayscale for easier comparison
background_gray = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

detections = []  # List to store detection info

while True:
    # Read current frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess current frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Compute the absolute difference between current frame and background
    frame_diff = cv2.absdiff(background_gray, gray_frame)
    thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours (changes)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If significant change detected, re-run detection
    change_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 5000:  # Adjust sensitivity (bigger area = more motion)
            change_detected = True
            break

    if change_detected:
        print("Change detected! Running detection...")
        results = model.predict(frame, stream=True)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = box.conf[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                label = f"{class_name} {conf:.2f}"

                detections.append((x1, y1, x2, y2, label))

        # Update background frame to avoid detecting the same motion again
        background_gray = gray_frame.copy()

        # Print detected object names
        detected_classes = list(set([label.split()[0] for _, _, _, _, label in detections]))
        print("Detected Objects:", ", ".join(detected_classes))

    # Draw detections
    for (x1, y1, x2, y2, label) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Smart Human and Object Detection', frame)

    # Break when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
