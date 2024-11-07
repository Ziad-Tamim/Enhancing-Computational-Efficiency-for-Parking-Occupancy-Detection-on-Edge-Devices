"""
Date: 2024-07-01
Autor: Ziad Tamim
Desctiption:
### Centralized Layer in the Parking Occupancy System

This Flask application functions as the centralized layer in a parking occupancy detection system. It is designed to aggregate and display parking slot status data sent from edge devices, such as Raspberry Pi units, which are responsible for detecting the occupancy of parking slots in real-time.

### Key Components:

- **In-Memory Storage:**
  - The application maintains a global dictionary `slots_status` to store the latest occupancy status of each parking slot. This dictionary is updated each time new data is received from the edge devices.

- **API Endpoints:**
  - **/update_parking (POST):** 
    - This endpoint accepts JSON data from the edge devices. The data includes the occupancy status of each parking slot, which is then stored in the `slots_status` dictionary. This endpoint is critical for keeping the centralized system up to date with real-time parking slot information.
  
  - **/status (GET):** 
    - Provides the current occupancy status of all parking slots in JSON format. This endpoint allows the frontend to retrieve the latest data for display purposes.
  
  - **/ (GET):**
    - Serves an HTML page that visually represents the occupancy status of the parking lot. The page displays each parking slot's status (Occupied or Free) and updates dynamically based on data fetched from the `/status` endpoint.

### How It Works:

1. **Data Collection:** 
   - The system collects parking slot occupancy data from multiple edge devices across different areas of the parking lot. These devices run detection algorithms and send the results to the `/update_parking` endpoint as JSON data.

2. **Data Aggregation and Display:**
   - The received data is stored centrally and can be accessed through the `/status` endpoint or viewed on a web interface. The web interface, rendered by the root endpoint `/`, shows a visual representation of the parking lot with the status of each slot. The frontend periodically refreshes this status using data fetched from the `/status` endpoint, ensuring that the display is always current.

3. **Integration with Edge Devices:**
   - The application is designed to integrate seamlessly with edge devices deployed in the parking lot. These devices handle the on-site detection, and the centralized layer aggregates the results, providing a comprehensive view of the entire parking facility's status.

This centralized layer plays a crucial role in the overall parking occupancy detection system, enabling real-time monitoring and management of parking spaces through a centralized interface.
"""

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# In-memory storage for parking slots status
slots_status = {}

@app.route('/update_parking', methods=['POST'])
def update_parking():
    global slots_status
    data = request.json
    slots_status = data
    print(f"Updated parking data: {slots_status}")
    return jsonify({"status": "success"}), 200

@app.route('/')
def index():
    return render_template('index.html', slots_status=slots_status)

@app.route('/status', methods=['GET'])
def status():
    return jsonify(slots_status)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
