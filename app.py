import streamlit as st
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration, webrtc_streamer, WebRtcMode
import cv2
import google.generativeai as genai
from deepface import DeepFace as df
import os

# Initialize Generative AI model
GOOGLE_API_KEY = st.secrets['API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

face_cascade_path = os.path.join("haarcascade_frontalface_default.xml")
faceCascade = cv2.CascadeClassifier(face_cascade_path)

class EmotionDetector(VideoProcessorBase):
    def __init__(self):
        self.dominant_emotion = "No face detected"
        self.frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, self.dominant_emotion = self.detect_face(img)
        return img

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        dominant_emotion = "No face detected"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            try:
                analyze = df.analyze(frame, actions=['emotion'])
                dominant_emotion = analyze[0]['dominant_emotion']
            except Exception as e:
                pass
        
        return frame, dominant_emotion

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Storytellercv</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter word limit</p>", unsafe_allow_html=True)
word_limit = st.number_input("", min_value=30, value=100)

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=EmotionDetector,
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_processor:
    emotion_detector = webrtc_ctx.video_processor
    dominant_emotion = emotion_detector.dominant_emotion
    
    if st.button("Generate Story"):
        if dominant_emotion != "No face detected":
            response = model.generate_content(f"Write a story within {word_limit} words. The story should be focused on emotional wellbeing and support. My emotion: {dominant_emotion}. For example, if I'm angry, it should make me happy.")
            output_text = response.text
            st.write("Emotion: ", dominant_emotion)
            st.write("Story: ", output_text)
        else:
            st.write("No face detected")

