from flask import Flask, request, jsonify
import cv2
import base64
import tempfile
from openai import OpenAI

app = Flask(__name__)

# Initialize the OpenAI client with your API key
openai_client = OpenAI(api_key='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')

@app.route('/process-video', methods=['POST'])
def process_video():
    # Check if 'video' key is in the files
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video_file = request.files['video']

    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        video_file.save(temp_video.name)
        temp_video.close()

        # Open the video file
        video = cv2.VideoCapture(temp_video.name)

        # Check if video is opened successfully
        if not video.isOpened():
            return jsonify({"error": "Failed to open video file"}), 400

        # Process the first frame of the video
        success, frame = video.read()
        if not success:
            return jsonify({"error": "Failed to read video frame"}), 400

        # Convert the frame to base64 for processing
        _, buffer = cv2.imencode('.jpg', frame)
        base64_frame = base64.b64encode(buffer).decode('utf-8')

        # Create the payload for OpenAI
        prompt_messages = [
            {
                "role": "user",
                "content": [
                    "description",
                    {"image": base64_frame, "resize": 768},
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": prompt_messages,
            "max_tokens": 200,
        }

        # Call the OpenAI API
        result = openai_client.chat.completions.create(**params)
        response_text = result.choices[0].message.content

        # Return the response
        return jsonify({"description": response_text})

if __name__ == '__main__':
    app.run(debug=True)
