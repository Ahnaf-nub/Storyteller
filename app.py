import cv2
import streamlit as st
import google.generativeai as genai
from deepface import DeepFace as df

# Initialize Generative AI model
GOOGLE_API_KEY = ""
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
        
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame, dominant_emotion

# Streamlit UI
st.title("Real-Time Face Detection")

if st.button("Open Camera"):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        st.write("Error: Could not open camera.")
    else:
        success, frame = cap.read()
        
        if success:
            frame, dominant_emotion = detect_face(frame)
            st.image(frame, channels="BGR")
            st.write("Emotion: ", dominant_emotion)
            
            if dominant_emotion != "No face detected":
                response = model.generate_content(f"Write a story within 100 words. The story should improve my emotion: {dominant_emotion}. For example, if I'm angry, it should make me happy.")
                output_text = response.text
                st.write("Story: ", output_text)
        
        cap.release()

cv2.destroyAllWindows()
