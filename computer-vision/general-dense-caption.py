import cv2
import requests
import json
from datetime import datetime
import time

# Replace with your Azure Vision AI endpoint and key
vision_endpoint = "https://eastus.api.cognitive.microsoft.com/computervision/imageanalysis:analyze"
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
        # reference: https://eastus.dev.cognitive.microsoft.com/docs/services/Cognitive_Services_Unified_Vision_API_2023-10-01/operations/61d65934cd35050c20f73ab6
        params = {
            'features': 'caption,denseCaptions,people,objects',
            'model-name': 'latest',
            'language': 'en',
            'api-version': '2023-10-01'
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
        print(json.dumps(results, indent=4))

        # Print out general captions
        captions = results.get('captionResult', {}).get('text', [])
        caption_confidence = results.get('captionResult', {}).get('confidence', [])
        print(f"Caption: {captions}, Confidence: {caption_confidence}")

        # Print out dense captions
        dense_captions = results.get('denseCaptionsResult', {}).get('values', [])
        print(f"Suggested Dense Captions: {len(dense_captions)}")
        for caption in dense_captions:
            caption_text = caption.get('text', [])
            caption_confidence = caption.get('confidence', [])
            print(f"Dense Caption: {caption_text}, Confidence: {caption_confidence}")

        # Print out people result
        detected_people = results.get('peopleResult', {}).get('values', [])
        print(f"Detected People: {len(detected_people)}")
        for people in detected_people:
            people_confidence = people.get('confidence', [])
            print(f"People Confidence: {people_confidence}")
        
        # Print out objects result
        detected_objects = results.get('objectsResult', {}).get('values', [])
        print(f"Detected Objects: {len(detected_objects)}")
        for obj in detected_objects:
            tags = obj.get('tags', [])
            for tag in tags:
                obj_name = tag.get('name')
                obj_confidence = tag.get('confidence')
                print(f"Object Name: {obj_name}, Confidence: {obj_confidence}")



        # Wait for 5 seconds
        cv2.waitKey(5000)

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Release the video capture object
    cap.release()