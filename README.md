# Enhancing Computational Efficiency for Parking Occupancy Detection on Edge Devices
![Screenshot 2024-10-09 231151](https://github.com/user-attachments/assets/49e31819-f5e0-4c00-8eb8-787a4f490005)


## Project Overview
This project addresses the challenge of parking occupancy detection in urban environments by leveraging deep learning models optimized for edge devices. The system is built using a custom deep learning model called QCustom, based on MobileNetV1 architecture, designed specifically for real-time parking classification on resource-constrained devices like the Raspberry Pi. By implementing model compression techniques, this project achieves high accuracy and fast inference speeds, making it suitable for practical deployment in smart parking systems.

## Project Objectives
- Develop an edge-based deep learning model for real-time parking occupancy classification.
- Optimize the model using compression techniques (knowledge distillation and quantization).
- Test the system's performance on both a prototype parking lot and real-world video data.

## Datasets
The project utilizes three datasets for training and testing the parking occupancy classification model:

1. CNRPark-EXT: A comprehensive dataset with 144,965 labeled images of parking spaces captured under various conditions.
2. Prototype Parking Lot Dataset: Created using a controlled setup with toy cars and a parking mat to simulate real-world conditions. This dataset helps refine the model for edge-based implementation.
3. Hyderabad Dataset: An open-air parking lot dataset that includes night-time images, allowing the model to handle diverse lighting scenarios.

| Dataset               | Occupied Samples | Empty Samples | Total Samples |
|-----------------------|------------------|---------------|---------------|
| CNRPark-EXT           | 79,307          | 65,658        | 144,965       |
| Prototype Dataset     | 119             | 292           | 411           |
| Hyderabad Dataset     | 3,150           | 3,372         | 6,522         |


## Implementation
### Model Architecture
The QCustom model is based on the MobileNetV1 architecture but customized to reduce complexity while maintaining high accuracy. Key features of the model:

<div align="center">
    <img src="https://github.com/user-attachments/assets/4c6222df-6303-4a39-8323-126e0bb50f8e" alt="Screenshot 2024-08-22 162142" width="300"/>
</div>

- Depthwise Separable Convolutions: Used to create a lightweight architecture that performs spatial filtering and feature combining with minimal computational overhead.
- Compression Techniques: Knowledge distillation and quantization reduce model size, enabling deployment on resource-limited devices.
<div align="center">
    <img src="https://github.com/user-attachments/assets/6111320d-32ae-4aaa-bb0c-b9dbf8366891" alt="Screenshot 2024-08-22 170356" width="300"/>
</div>

### Code Structure
- Data Preprocessing: Analyse and organization of dataset images to create labeled data for training.
- Model Training: Custom model is trained using TensorFlow with knowledge distillation from a larger teacher (ResNet50) model and quantized using TensorFlow Lite.
- Occupancy Detection Algorithm: Processes real-time video frames on a Raspberry Pi to detect parking occupancy in designated slots.
- Data Transmission: Sends occupancy data from the Raspberry Pi to a centralized system for data aggregation and visualization.
- User Interface: Hosts a local web server using Flask, allowing users to view real-time parking lot occupancy status.

### System Design
The system is composed of three layers:

1. Sensor Layer: Captures video frames and processes occupancy data on the edge device.
2. Centralized System Layer: Aggregates data from multiple sensors.
3. User Interface Layer: Displays real-time parking occupancy status through a web interface.

![Screenshot 2024-08-11 153500](https://github.com/user-attachments/assets/4b8754ea-fd80-45fa-afbd-a00ac2902b0a)

## Results 
### Model Accuracy Across Datasets

| Model         | CNRPark_EXT (DS1) | Collected Data (DS2) | Hyderabad Dataset (DS3) |
|---------------|--------------------|-----------------------|--------------------------|
| AlexNet 227x227 | 0.9843           | 1.000                | 0.9985                   |
| AlexNet 150x150 | 0.9851           | 1.000                | 0.9985                   |
| MobileNetV3    | 0.9906            | 1.000                | 0.999                    |
| MobileNetV1    | 0.9969            | 0.5625               | 0.9992                   |
| Custom         | 0.9892            | 1.000                | 0.9824                   |
| QCustom        | 0.9869            | 1.000                | 0.9821                   |

### Model Inference Speeds on a Single Slot (ms)

| Model          | PC Inference (ms)  | Raspberry Pi Inference (ms) |
|----------------|---------------------|------------------------------|
| AlexNet 227x227| 60.62 / 64.67      | 189.16 / 197.04             |
| AlexNet 150x150| 55.63 / 56.51      | 152.95 / 153.47             |
| MobileNetV3    | 52.99 / 62.36      | 143.3 / 151.02              |
| MobileNetV1    | 55.99 / 56.84      | 151.6 / 156.5               |
| Custom         | 52.08 / 53.73      | 123.88 / 125.57             |
| QCustom        | 0.21 / 0.21        | 0.74 / 0.77                 |

![Screenshot 2024-08-07 214801](https://github.com/user-attachments/assets/5078d090-24a8-4426-b2c0-330566ff3c8b)

### Prototype Testing
Using the prototype parking lot setup, the QCustom model achieved 100% accuracy in detecting occupied and empty spaces with an inference speed of 30 fps. The system has a 1-2 second latency for updating the user interface, suitable for practical applications.

### Real-World Testing
Simulating real-world conditions using a video feed, the QCustom model demonstrated a 97.44% accuracy with consistent detection even under partial occlusions and varying lighting conditions. The high accuracy and fast inference time highlight the model’s robustness and efficiency in real-world scenarios.

## Getting Started
1. Clone the repository and install dependencies.
2. Prepare the Dataset: Place the datasets in the data/ directory and ensure the format aligns with the project specifications.
3. Deploy on Raspberry Pi: Upload the quantized model to the Raspberry Pi and run the occupancy_detection.py script.
4. Launch the User Interface: Use a computer as a centrilsed server and connect to the flask server.
