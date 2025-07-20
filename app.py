import gradio as gr
import numpy as np
import pandas as pd
import cv2
from keras.models import model_from_json
import matplotlib.pyplot as plt
import cohere
import os
import pymysql
from sqlalchemy import create_engine
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


# Connect to MySQL
db = pymysql.connect(host='localhost',
                     user='root',
                     password='password',
                     database='mydatabase',
                     charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)



# Define function to insert transcript into MySQL
def insert_into_table():
    global transcript
    try:
        with db.cursor() as cursor:
            sql = "INSERT INTO Summary(Transcript,Report,Max_emotion) VALUES (%s,%s,%s)"
            cursor.execute(sql, (transcript,Final_report,emotion))
        db.commit()
        print("Transcript,report and Max_emotion inserted into MySQL.")
    except Exception as e:
        print(f"Error inserting transcript into MySQL: {e}")
        db.rollback()
        
       
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

    API_KEY = "559369807055d769c818eeff7ad8fb07de313054"
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
    global emotion
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
    co = cohere.Client(api_key="ssKgGMQGAM6wBhRQYTaZhFPTtpPVUM6Xws4tRv3C")
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
    global Final_report
    Final_report = generated_text
    insert_into_table()
    return generated_text

#Define a function which displays the recent data

def recent_data():
    try:
        engine = create_engine('mysql+pymysql://root:password@localhost/mydatabase')
        query = "SELECT * FROM Summary "
        df = pd.read_sql(query, engine)
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {e}"

def display_table():
    df, error = recent_data()
    if error is None and not df.empty:
        return df
    else:
        return error

def recent_particular_data(personid):
    try:
        engine = create_engine('mysql+pymysql://root:password@localhost/mydatabase')
        query = "SELECT * FROM Summary WHERE person_id = %(person_id)s"
        df = pd.read_sql(query, engine,params = {'person_id':personid})
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Error fetching data: {e}"


def display_particular_report(personid):
    df, error = recent_particular_data(int(personid))
    if error is None and not df.empty:
        return df
    else:
        return error

                
# Define the Gradio interface for emotion analysis
def create_emotion_analysis_interface():
    with gr.Blocks() as main_interface:
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
                video_output = gr.Video(label="Emotion Detection Output")
            
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

        with gr.Row():
            recents_button = gr.Button("Recents")
            recents_button.click(lambda: "recents", inputs=None, outputs=None) # Navigate to recents interface

    return main_interface

# Define the Gradio interface for recents
def create_recents_interface():
    with gr.Blocks() as recents_interface:
        gr.Markdown("# Recents Interface")
        

        with gr.Column():
            recents_output = gr.DataFrame(label="Recent Reports")
            recents_button = gr.Button("Display Recent Reports")
            recents_button.click(fn = display_table, inputs=None, outputs=recents_output)

        with gr.Row():
            back_button = gr.Button("Back")
            back_button.click(lambda: "main", inputs=None, outputs=None) # Navigate to main interface
 
    return recents_interface

# Define the Gradio interface for searching
def create_search_interface():
    with gr.Blocks() as search_interface:
        gr.Markdown("# Search Interface")
        

        with gr.Column():
            person_id_input = gr.Textbox(label = "Enter your id" , placeholder ="Enter a valid Person ID",type = "text" )
            person_id_report = gr.DataFrame(label="Search Reports")
            search_button = gr.Button("Display Search Reports")
            search_button.click(fn = display_particular_report, inputs=person_id_input, outputs=person_id_report)

        with gr.Row():
            back_button = gr.Button("Back")
            back_button.click(lambda: "main", inputs=None, outputs=None) # Navigate to main interface
 
    return search_interface
    
# Define the main function to create the Gradio application
def create_app():
    main_interface = create_emotion_analysis_interface()
    recents_interface = create_recents_interface()
    search_interface = create_search_interface()
    # Define a dictionary to store the interfaces
    interfaces = {
        "main": main_interface,
        "recents": recents_interface,
        "search" : search_interface
    }

    # Create a Gradio Tab component to switch between interfaces
    with gr.Blocks() as app:
        gr.Markdown("# Emotion Detection Application")
        with gr.Tabs() as tabs:
            for name, interface in interfaces.items():
                with gr.TabItem(name):
                    interface.render()

    return app
    
iface = create_app()
iface.launch(share=True, server_port=7860)

