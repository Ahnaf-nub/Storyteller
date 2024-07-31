from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import google.generativeai as genai
import cv2
from deepface import DeepFace as df
import time
from fastapi.responses import StreamingResponse
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
app = FastAPI()

GOOGLE_API_KEY = "AIzaSyCly-89E1Cr6N9jy88_PWQjuMJsaxD66cA"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        try:
            analyze = df.analyze(frame, actions = ['emotion'])
            cv2.putText(frame, analyze[0]['dominant_emotion'],  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        except:
            cv2.putText(frame, "No face detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_with_faces = detect_face(frame)
            _, buffer = cv2.imencode('.jpg', frame_with_faces)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output_text": ""})


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post("/upload", response_class=HTMLResponse)
async def generate_story(request: Request, topic: str = Form(...), genre: str = Form(...), word_limit: int = Form(...)):
    try:
        response = model.generate_content(f"Write a story about {topic} in {genre} within {word_limit} words.")
        output_text = response.text
        return templates.TemplateResponse("index.html", {"request": request, "output_text": output_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
