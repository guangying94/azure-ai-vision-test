import cv2
import requests
import json
from datetime import datetime
import time

# Replace with your Azure Vision AI endpoint and key
vision_endpoint = "https://eastus.api.cognitive.microsoft.com/face/v1.0/detect"
vision_key = "xxxxx"

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
        # reference: https://westus.dev.cognitive.microsoft.com/docs/services/563879b61984550e40cbbe8d/operations/563879b61984550f30395236
        params = {
            # 'visualFeatures': 'Objects',
            # 'returnFaceId': 'true',
            'returnFaceLandmarks': 'true',
            'returnFaceAttributes': 'headPose,mask',
            ## 'recognitionModel': 'recognition_04',
            ## 'returnRecognitionModel': 'false',
            'detectionModel': 'detection_03',
            'faceIdTimeToLive': '60'
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
        print(f"Faces detected: {len(results)}")
        for face in results:
            face_attributes = face.get('faceAttributes', {})
            head_pose = face_attributes.get('headPose', {})
            print(f"Head Pose: [Pitch] {head_pose.get('pitch')}, [Roll] {head_pose.get('roll')}, [Yaw] {head_pose.get('yaw')}")

        # Wait for 5 seconds
        cv2.waitKey(5000)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object
    cap.release()