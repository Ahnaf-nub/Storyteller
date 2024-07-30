from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import google.generativeai as genai
app = FastAPI()

GOOGLE_API_KEY = "AIzaSyCly-89E1Cr6N9jy88_PWQjuMJsaxD66cA"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "output_text": ""})

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
