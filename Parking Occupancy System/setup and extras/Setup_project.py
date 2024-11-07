import cv2
import json
import numpy as np

# Function to load points from JSON file
def load_points(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    points = [np.array(shape['points'], dtype=np.int32) for shape in data['shapes']]
    return points

# Load points from file
json_path = 'C:/Users/ziadt/OneDrive - Queen Mary, University of London/semister 2/MSc Project - diss/Project Implementation/Setup Implenetation/Coorinates_for_Setup.json'  # Update this path
points = load_points(json_path)
print(points)

# Setup camera and window
cap = cv2.VideoCapture(1)  # Adjust the device index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width to 1920 pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set the height to 1080 pixels

cv2.namedWindow('Camera Calibration')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw all points
    for point in points:
        for p in point:
            cv2.circle(frame, tuple(p), 5, (0, 255, 0), -1)  # Green circle at each point

    frame_resized = cv2.resize(frame, (960, 540))  # Resize to half of 1920x1080
    cv2.imshow('Camera Calibration', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
