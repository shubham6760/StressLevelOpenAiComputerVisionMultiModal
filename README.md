Medical Chatbot with Real-Time Stress Level Prediction

The medical chatbot application is designed to provide users with medical information and advice while simultaneously analyzing their stress levels in real-time using computer vision technology. The application integrates OpenAI's GPT-3 or GPT-4 for generating responses and DeepFace for emotion detection from video streams.

Key Features
User Interface:

Home Page: Introductory page with application overview and features.
Chat Interface: A text box for user queries, real-time video stream, display of chatbot responses, and audio playback of responses.
Medical Question Handling:

Text Input: Users can input medical questions through a text box.
Chatbot Responses: The chatbot generates responses using OpenAI's GPT-3 or GPT-4 models.
Audio Playback: Responses are converted to speech using gTTS and played back to the user.
Real-Time Stress Level Analysis:

Video Stream: Captures the user's face in real-time.
Emotion Detection: Uses DeepFace to analyze facial expressions and detect emotions.
Stress Level Calculation: Calculates a stress level score based on detected emotions like anger, fear, and sadness.
Stress Management Advice: Provides advice based on the detected stress level.
Stress Level Display:

Current Stress Level: Real-time display of the user's stress level.
Historical Stress Level Graph: A graph showing stress levels over time (mock data for demonstration).
Technical Requirements
Operating System: Windows 10 or later, macOS, or a Linux distribution (Ubuntu recommended).
Python Environment: Python 3.8 or later.
Libraries and Packages
Core Libraries:
openai: For generating chatbot responses using OpenAI's GPT-3/GPT-4.
streamlit: For building the web interface.
streamlit-webrtc: For handling real-time video and audio streams.
gtts: For converting text responses to speech.
numpy: For numerical operations.
opencv-python-headless: For computer vision tasks.
face-recognition: For detecting faces in video streams.
deepface: For facial expression analysis and emotion detection.
python-dotenv: For managing environment variables.
