import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression
import pytesseract

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
]

# Set input dimensions (must be multiple of 32)
newW, newH = 640, 640  # increased resolution for better detection
min_confidence = 0.6   # increased confidence threshold

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Failed to grab frame")
        break

    orig = frame.copy()
    (H, W) = frame.shape[:2]
    rW, rH = W / float(newW), H / float(newH)

    # Resize and create blob
    image = cv2.resize(frame, (newW, newH))
    blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    fps = 1.0 / (end - start)

    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []

    # Extract bounding boxes
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0, x1, x2, x3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX, offsetY = x * 4.0, y * 4.0
            angle = angles[x]
            cos, sin = np.cos(angle), np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            endX = int(offsetX + cos * x1[x] + sin * x2[x])
            endY = int(offsetY - sin * x1[x] + cos * x2[x])
            startX, startY = int(endX - w), int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    overlay = orig.copy()

    for (startX, startY, endX, endY) in boxes:
        startX = max(0, int(startX * rW))
        startY = max(0, int(startY * rH))
        endX = min(orig.shape[1], int(endX * rW))
        endY = min(orig.shape[0], int(endY * rH))

        # Draw semi-transparent box
        cv2.rectangle(overlay, (startX, startY), (endX, endY), (0, 255, 0), -1)

        # Preprocess ROI for better OCR accuracy
        roi = orig[startY:endY, startX:endX]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # OCR config: OEM 3 (LSTM + legacy), PSM 6 (single uniform block)
        custom_config = r'--oem 3 --psm 6'

        text = pytesseract.image_to_string(roi_thresh, config=custom_config).strip()
        text = text.replace('\n', ' ').strip()

        # Add text label if not empty
        if text:
            cv2.putText(orig, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    # Blend overlay with original image
    cv2.addWeighted(overlay, 0.25, orig, 0.75, 0, orig)

    # FPS Counter
    cv2.putText(orig, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live Text Detection - Press 'q' to Exit", orig)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
