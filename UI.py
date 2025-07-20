import gradio as gr
import numpy as np
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
import cohere
import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Load the model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Load weights into the new model
loaded_model.load_weights("fer.h5")
print("Loaded model from disk")

# Setting image resizing parameters
WIDTH = 48
HEIGHT = 48
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize global variables
max_emotion_percentage = 0
transcript = ""

# Define function to process video with emotion detection
def detect_emotions(video_file):
    global max_emotion_percentage, transcript, angry, disgust, fear, happy, sad, surprise, neutral, max_emotion_index

    cap = cv2.VideoCapture(video_file)
    result_frames = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    angry = disgust = fear = happy = sad = surprise = neutral = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 10)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (WIDTH, HEIGHT)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Predicting the emotion
            yhat = loaded_model.predict(cropped_img)
            emotion = labels[int(np.argmax(yhat))]

            if emotion == 'Angry':
                angry += 1
            elif emotion == 'Disgust':
                disgust += 1
            elif emotion == 'Fear':
                fear += 1
            elif emotion == 'Happy':
                happy += 1
            elif emotion == 'Sad':
                sad += 1
            elif emotion == 'Surprise':
                surprise += 1
            elif emotion == 'Neutral':
                neutral += 1

            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        emotion_counts = [angry, sad, surprise, disgust, happy, neutral, fear]
        total = sum(emotion_counts)
        max_emotion_index = emotion_counts.index(max(emotion_counts))
        max_emotion_percentage = max(emotion_counts) * 100 / total if total > 0 else 0
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    API_KEY = "your_api_key"
    try:
        deepgram = DeepgramClient(API_KEY)
        with open(video_file, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]

    except Exception as e:
        print(f"Exception: {e}")

    return 'output.mp4'

def display_text():
    global max_emotion_percentage
    
    if max_emotion_index == 0:
        emotion = "Angry"
    elif max_emotion_index == 1:
        emotion = "Sad"
    elif max_emotion_index == 2:
        emotion = "Surprise"
    elif max_emotion_index == 3:
        emotion = "Disgust"
    elif max_emotion_index == 4:
        emotion = "Happy"
    elif max_emotion_index == 5:
        emotion = "Neutral"
    elif max_emotion_index == 6:
        emotion = "Fear"
        
    return f"Max Emotion is {emotion} and Percentage is : {max_emotion_percentage:.2f}%"

def graph():
    global angry, sad, surprise, disgust, happy, neutral, fear
    
    x = np.array(["Angry", "Sad", "Surprise", "Disgust", "Happy", "Neutral", "Fear"])
    y = np.array([angry, sad, surprise, disgust, happy, neutral, fear])
    fig = plt.figure()
    plt.bar(x, y)
    return fig

def get_transcript():
    return transcript

def final_report():
    co = cohere.Client(api_key="your_api_key")
    message = (
        f"Summarize how this person particularly felt about saying the following transcript: {transcript} "
        f"and also by taking into account the emotion scores obtained from the facial sentiment analysis "
        f"for each of the seven emotions:- Angry: {angry}, Sad: {sad}, "
        f"Disgust: {disgust}, Neutral: {neutral}, Happy: {happy}, "
        f"Fear: {fear}, Surprise: {surprise}. "
        "By combining the emotion scores obtained through visual analysis and the audio transcript, make sure you catch "
        "the exact emotion in each sentence he/she speaks."
    )

    stream = co.chat_stream(message=message)
    generated_text = ""
    for event in stream:
        if event.event_type == "text-generation":
            generated_text += event.text

    return generated_text


    
    
with gr.Blocks() as demo:
    

    gr.Markdown("# Video Emotion Analysis Tool")
    gr.Markdown("Upload a video to analyze emotions, display overall emotion, visualize graphs, transcribe audio, and provide a combined review.")
    
    with gr.Row():
        with gr.Column(scale=1):
            graph_output = gr.Plot(label="Emotion Graphs")
            graph_button = gr.Button("Display Emotion Graphs")
            graph_button.click(graph, inputs=None, outputs=graph_output)
            
            text_output = gr.Textbox(label="Overall Emotion from Video")
            text_button = gr.Button("Display Overall Emotion")
            text_button.click(display_text, inputs=None, outputs=text_output)
            
        
        with gr.Column(scale=2):
            video_output = gr.Video(label="Emotion Detection Output", elem_id="output-video")
        
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload a video")
            video_input.upload(detect_emotions, inputs=video_input, outputs=video_output)
    
    gr.Markdown("___")  # Horizontal line for margin

    with gr.Row():
        with gr.Column():
            transcript_output = gr.Textbox(label="Audio Transcript")
            transcript_button = gr.Button("Display Audio Transcript")
            transcript_button.click(get_transcript, inputs=None, outputs=transcript_output)

    with gr.Row():
        with gr.Column():
            report_output = gr.Textbox(label="Combined Model Review")
            report_button = gr.Button("Generate Final Report")
            report_button.click(final_report, inputs=None, outputs=report_output)
            

demo.launch(share=True)
