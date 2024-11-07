"""
Date: 2024-07-01
Author: Ziad Tamim
Discription:
### Inputs:
- **Video Feed:** The script captures real-time video input from a webcam or processes a pre-recorded video file, focusing on a parking lot with multiple parking slots.
- **Parking Slot Coordinates:** A JSON file is loaded at the start, containing the coordinates for each parking slot within the video frame. These coordinates are crucial for cropping and isolating each parking slot for further analysis.

### Processing:
- **Model Loading:** The script loads the QCustom model specifically optimized for edge device inference, allowing for efficient real-time parking slot occupancy detection.
- **Image Cropping:** The `crop_to_square` function is employed to extract each parking slot from the video frame, transforming the cropped section into a square image (150x150 pixels) suitable for model input.
- **Prediction with History:** The script uses the `predict_image` function to normalize the cropped images and perform inference using the loaded model. The predictions are stored in a history buffer (deque) for each slot, and the most common prediction is used to determine the final occupancy status.
- **Overlaying Results:** The script overlays the occupancy status ("Occupied" or "Free") directly onto the video frame, providing real-time visual feedback on the parking slots' status.

### Outputs:
- **Annotated Video Feed:** The script displays the video feed with annotations indicating the occupancy status of each parking slot. This output is intended for live monitoring and visualization.
- **Parking Data Transmission to Centralized Server:** The occupancy data for each parking slot is formatted into a JSON structure and sent to a centralized server via an HTTP POST request. The `send_parking_data` function handles this transmission, ensuring the data is correctly serialized and successfully delivered.

### Use of Webcam:
- **Live Video Feed:** The script captures and processes live video input from a connected webcam. This functionality makes the script ideal for dynamic parking management applications, where real-time monitoring is critical.

### Centralized Server Integration:
- **Data Centralization:** The script integrates with a centralized parking management system by sending occupancy data to a central server. This allows the system to aggregate and analyze data from multiple cameras or locations, providing a comprehensive overview of parking availability across a wider area.

This script is designed for smart parking systems that require real-time detection, centralized data processing, and remote monitoring. It is optimized to run on edge devices like the Raspberry Pi, offering efficient performance while supporting integration with broader centralized management systems.
"""

import json
import cv2
import numpy as np
import requests
from collections import deque, Counter
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load tflite model
interpreter = tf.lite.Interpreter(model_path='/home/ziad_t/Desktop/Projects/MSc Project/Testing inference/Znet 150 x 150/MyData/Custom_150_D2_V1/Qstudent_D2_V1.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load JSON data from file to get the coordinates
#with open('C:/Users/ziadt/Desktop/Projects/MSc Project implimintation/Detection Algorithem Real-time/Parking videos/Olypic park/Olympic_coordinates.json', 'r') as file:
with open('/home/ziad_t/Desktop/Projects/MSc Project/Improving-Model-Efficiency-for-Parking-Occupancy-Detection-on-Edge-Devices-main/Detection Algorithem Real-time/Slot_coordinates.json', 'r') as file:
    data = json.load(file)

slots = {f"Slot {idx + 1}": {'coordinates': shape['points'], 'history': deque(maxlen=10), 'occupied': None} # slot dictionary with slot id, coordinates, history and occupied status
         for idx, shape in enumerate(data['shapes'])}

def crop_to_square(image, vertices, size=(150, 150)): # crop the slot image
    pts_src = np.array(vertices, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(pts_src)
    dimension = max(w, h) # get the max dimension between width and height
    pts_dst = np.array([
        [0, 0],
        [dimension - 1, 0],
        [dimension - 1, dimension - 1],
        [0, dimension - 1]
    ], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)  # get the perspective transform matrix
    transformed_image = cv2.warpPerspective(image, matrix, (dimension, dimension)) # warp the image to the new perspective
    resized_image = cv2.resize(transformed_image, size) # resize the image based on the input size of the model
    return resized_image

def predict_image(image):
    # Assuming your model expects images normalized to [0, 1]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
    image = image.astype('float32')
    image = np.expand_dims(image, axis=0) # Add batch dimension
    interpreter.set_tensor(input_details[0]['index'], image) # Set the input tensor
    interpreter.invoke() # Perform inference

    # Get the output
    prediction = interpreter.get_tensor(output_details[0]['index']) # Get the output tensor value

    if prediction > 0.5: # Thresholding the prediction
        return 1
    else:
        return 0

def send_parking_data(data):
    url = 'http://172.20.10.10:5000/update_parking' # Server URL
    headers = {'Content-Type': 'application/json'} # Request headers
    
    # Convert data to ensure all types are JSON serializable
    def convert_data(data): # Convert data to JSON serializable format
        if isinstance(data, dict):
            return {k: convert_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_data(i) for i in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    filtered_data = {slot_id: {'occupied': slot_info['occupied']} for slot_id, slot_info in data.items()} # Filter the data to send
    serializable_data = convert_data(filtered_data) # Convert the data to JSON serializable format
    response = requests.post(url, headers=headers, json=serializable_data) # Send the data to the server

    if response.status_code == 200: # Check if the data was sent successfully
        print("Data sent successfully")
    else:
        print("Failed to send data")

# Start the webcam

# video path for testing
# path = 'C:/Users/ziadt/Desktop/Projects/MSc Project implimintation/Detection Algorithem Real-time/Parking videos/Olypic park/IMG_8573.avi'
# path = 'C:/Users/ziadt/Desktop/Projects/MSc Project implimintation/Detection Algorithem Real-time/Parking videos/Protype/Prototype Parking test.avi'
# cap = cv2.VideoCapture(path)

# webcam for testing
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # Set the codec for the video
#cap = cv2.VideoCapture('C:/Users/ziadt/Desktop/Projects/My Dataset/parking_lot_video.avi')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # Set the frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # Set the frame height

if not cap.isOpened():
    raise IOError("Cannot open video")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read or failed to fetch the frame.")
            break

        # Process each parking slot
        for slot_id, slot_info in slots.items():
            slot_image = crop_to_square(frame, slot_info['coordinates'])  # Crop the slot image
            prediction = predict_image(slot_image)  # Get prediction for the slot
            slot_info['history'].append(prediction)  # Append the prediction to history

            # Determine the most common prediction in the history
            if len(slot_info['history']) == slot_info['history'].maxlen:
                most_common_prediction = Counter(slot_info['history']).most_common(1)[0][0]
                slot_info['occupied'] = most_common_prediction
            else:
                slot_info['occupied'] = None  # Not enough history yet

            # Overlay the prediction on the main frame if there is enough history
            if slot_info['occupied'] is not None:
                text = f"Occupied" if slot_info['occupied'] == 1 else "Free"
                cv2.putText(frame, f"{slot_id}: {text}", (int(slot_info['coordinates'][0][0]), int(slot_info['coordinates'][0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if slot_info['occupied'] == 1 else (0, 0, 255), 2)

        # Display the main frame with predictions
        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("Parking Slots", frame)

        # Send the parking data to the server
        send_parking_data(slots)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

# # ########################################## use script below for keras model like alexnet ##################################
# import json
# import cv2
# import numpy as np
# import requests
# from tensorflow.keras.models import load_model
# import tensorflow as tf
# import requests

# # Load the model
# model = load_model('/home/ziad_t/Desktop/Projects/MSc Project/Testing inference/Alexnet with 150 x150/MyData/Alexnet_150D2_V1/AlexNet_my_dataset_v1_.h5')

# # Load JSON data from file to get the coordinates
# with open('/home/ziad_t/Desktop/Projects/MSc Project/Improving-Model-Efficiency-for-Parking-Occupancy-Detection-on-Edge-Devices-main/Detection Algorithem Real-time/Slot_coordinates.json', 'r') as file:
#     data = json.load(file)

# slots = {f"Slot {idx + 1}": {'coordinates': shape['points'], 'occupied': None}
#          for idx, shape in enumerate(data['shapes'])}

# def crop_to_square(image, vertices, size=(150, 150)):
#     pts_src = np.array(vertices, dtype=np.float32)
#     x, y, w, h = cv2.boundingRect(pts_src)
#     dimension = max(w, h)
#     pts_dst = np.array([
#         [0, 0],
#         [dimension - 1, 0],
#         [dimension - 1, dimension - 1],
#         [0, dimension - 1]
#     ], dtype=np.float32)
    
#     matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
#     transformed_image = cv2.warpPerspective(image, matrix, (dimension, dimension))
#     resized_image = cv2.resize(transformed_image, size)
#     return resized_image

# def predict_image(image):
#     # Assuming your model expects images normalized to [0, 1]
#     #image = image.astype('float32') / 255.0
#     image = np.expand_dims(image, axis=0)
#     prediction = model.predict(image)
#     print(prediction)
#     result = np.argmax(prediction)
#     print(result)
#     return result  # Return the class with the highest probability


# def send_parking_data(data):
#     url = 'http://172.20.10.10:5000/update_parking'
#     headers = {'Content-Type': 'application/json'}
    
#     # Convert data to ensure all types are JSON serializable
#     def convert_data(data):
#         if isinstance(data, dict):
#             return {k: convert_data(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             return [convert_data(i) for i in data]
#         elif isinstance(data, np.integer):
#             return int(data)
#         elif isinstance(data, np.floating):
#             return float(data)
#         elif isinstance(data, np.ndarray):
#             return data.tolist()
#         else:
#             return data
     
#     filtered_data = {slot_id: {'occupied': slot_info['occupied']} for slot_id, slot_info in data.items()}
#     serializable_data = convert_data(filtered_data)
#     response = requests.post(url, headers=headers, json=serializable_data)
    
#     if response.status_code == 200:
#         print("Data sent successfully")
#     else:
#         print("Failed to send data")

# # Start the webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# #cap = cv2.VideoCapture('C:/Users/ziadt/Desktop/Projects/My Dataset/parking_lot_video.avi')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# if not cap.isOpened():
#     raise IOError("Cannot open video")

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("No more frames to read or failed to fetch the frame.")
#             break

#         # Process each parking slot
#         for slot_id, slot_info in slots.items():
#             slot_image = crop_to_square(frame, slot_info['coordinates'])
#             prediction = predict_image(slot_image)  # Get prediction for the slot
#             slot_info['occupied'] = prediction

#             # Display the cropped image (for debugging)
#             cv2.imshow(slot_id, slot_image)

#             # Overlay the prediction on the main frame
#             text = f"Occupied" if prediction == 1 else "Free"
#             cv2.putText(frame, f"{slot_id}: {text}", (int(slot_info['coordinates'][0][0]), int(slot_info['coordinates'][0][1])),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if prediction == 1 else (0, 0, 255), 2)
#         # Display the main frame with predictions
#         frame = cv2.resize(frame, (960, 540))
#         cv2.imshow("Parking Slots", frame)
#         print(slots)

#         # Send the parking data to the server
#         send_parking_data(slots)

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#             break
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
