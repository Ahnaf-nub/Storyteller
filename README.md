# Storytellercv
Storytellercv is a real-time facial emotion detection and story generation application. Using Streamlit, OpenCV, and DeepFace for emotion analysis, this app captures video from your webcam, detects your emotion, and generates a personalized story to improve your emotional wellbeing using Google Generative AI model Gemini. You can also give the word limit default is 100.

## Features
- Real-time emotion detection from webcam feed.
- Generates personalized stories based on detected emotions.
- Integration with Google Generative AI model Gemini for story generation.

## Usage
- Clone the repository:
```
git clone https://github.com/Ahnaf-nub/Storyteller.git
cd Storyteller
```
- Install the dependencies `pip install requirements.txt`
- Run the File using `streamlit run app.py`
- Enter the desired word limit for the generated story.
- Click on the "Start" button to begin the webcam feed.
- The app will detect your dominant emotion in real-time and display it on the screen.
- Click on the "Generate Story" button to get a personalized story based on your detected emotion.
**N.B:**
  There is a Text version too in `Storyteller/Storytellertext/` where user has to provide only topic, word limit and genre and it will generate story for you.
