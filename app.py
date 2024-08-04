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

# Path to Haarcascade XML file
face_cascade_path = os.path.join("haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(face_cascade_path)

# Function to detect face and emotion
def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    dominant_emotion = "No face detected"
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        try:
            face_image = frame[y:y+h, x:x+w]
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_image_rgb)
            analyze = df.analyze(pil_image, actions=['emotion'])
            dominant_emotion = analyze[0]['dominant_emotion']
        except Exception as e:
            st.write(f"Error analyzing emotion: {e}")
        
    return frame, dominant_emotion

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Storytellercv</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter word limit</p>", unsafe_allow_html=True)
word_limit = st.number_input("", min_value=30, value=100)

# JavaScript to trigger camera input
start_button = st.button("Start")

if start_button:
    # Display the camera input widget
    uploaded_image = st.camera_input("Take a picture", key="camera_input")
    
    if uploaded_image is not None:
        # Convert image to OpenCV format
        image = Image.open(uploaded_image)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect face and emotion
        frame, dominant_emotion = detect_face(frame)
        
        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb)
        st.write("Emotion: ", dominant_emotion)
        
        if dominant_emotion != "No face detected":
            try:
                response = model.generate_content(f"Write a story within {word_limit} words. The story should be focused on emotional wellbeing and support. My emotion: {dominant_emotion}. For example, if I'm angry, it should make me happy.")
                output_text = response.text
                st.write("Story: ", output_text)
            except Exception as e:
                st.write(f"Error generating story: {e}")
