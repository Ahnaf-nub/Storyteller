import streamlit as st
import cv2
import google.generativeai as genai
from deepface import DeepFace as df
from PIL import Image
import numpy as np
import io

# Configure Google Generative AI
GOOGLE_API_KEY = st.secrets['API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Load the face cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Function to detect face and emotion
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    dominant_emotion = "No face detected"
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        try:
            analyze = df.analyze(frame, actions=['emotion'])
            dominant_emotion = analyze[0]['dominant_emotion']
        except:
            pass
        
    
    return frame, dominant_emotion

st.markdown("<h1 style='text-align: center;'>Storytellercv</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter word limit</p>", unsafe_allow_html=True)
word_limit = st.number_input("", min_value=30, value=100)
if st.button("Start"):
        success, frame = webrtc_streamer(key="example")
        
        if success:
            frame, dominant_emotion = detect_face(frame)
            st.image(frame, channels="BGR")
            st.write("Emotion: ", dominant_emotion)
            
            if dominant_emotion != "No face detected":
                response = model.generate_content(f"Write a story within {word_limit} words. The story should be Focused on emotional wellbeing and support. My emotion: {dominant_emotion}. For example, if I'm angry, it should make me happy.")
                output_text = response.text
                st.write("Story: ", output_text)
        

cv2.destroyAllWindows()
