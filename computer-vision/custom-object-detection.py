import cv2
import requests
import json
from datetime import datetime
import time

# Replace with your Azure Vision AI endpoint and key
vision_endpoint = "https://eastus.api.cognitive.microsoft.com/computervision/imageanalysis:analyze"
vision_key = "xxxxx"
custom_model_name = "xxxxx"

# Initialize the video capture object
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read a frame from the video capture object
        ret, frame = cap.read()

        # Convert the frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)

        # Prepare the headers for the request
        headers = {
            'Content-Type': 'application/octet-stream',
            'Ocp-Apim-Subscription-Key': vision_key,
        }

        # Prepare the parameters for the request
        params = {
            # 'visualFeatures': 'Objects',
            'model-name': custom_model_name,
            'api-version': '2023-02-01-preview'
        }
        
        # Record the start time
        start_time = time.time()

        # Send the POST request to the Vision API
        response = requests.post(vision_endpoint, headers=headers, params=params, data=jpeg.tobytes())

        # Record the end time
        end_time = time.time()

        # Print out the time taken
        print(datetime.now().strftime("%H:%M:%S") + f" - Time taken: {end_time - start_time}")

        # Parse the response
        results = response.json()

        # Print the results
        # print(json.dumps(results, indent=4))
        # Navigate through the JSON structure
        objects = results.get('customModelResult', {}).get('objectsResult', {}).get('values', [])

        # print the count of objects detected
        print(f"Objects detected: {len(objects)}")

        for obj in objects:
            tags = obj.get('tags', [])
            for tag in tags:
                name = tag.get('name')
                confidence = tag.get('confidence')
                print(f"Name: {name}, Confidence: {confidence}")

        # Wait for 5 seconds
        cv2.waitKey(5000)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object
    cap.release()