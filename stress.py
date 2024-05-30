import streamlit as st
from dotenv import load_dotenv
import os
import openai
import cv2
import numpy as np
from io import BytesIO
from gtts import gTTS
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Developer Information
developer_info = "Developed by Shubham Raj. Contact at sr6760.sr@gmail.com"

# Handle user input using OpenAI GPT
def handle_userinput(user_question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_question,
        max_tokens=150
    )
    chat_history = [{"content": response.choices[0].text.strip()}]
    
    # Save chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.extend(chat_history)
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        st.write(message['content'])
        
    # Convert the last response to speech
    response_text = chat_history[-1]['content'] if chat_history else ""
    tts = gTTS(response_text)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)

    # Play the audio response
    st.audio(audio_file, format='audio/mp3')

# Analyze stress level from facial expressions using DeepFace
def analyze_stress_level_from_video(frame):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use DeepFace to analyze the frame
    analysis = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
    
    emotion = analysis['dominant_emotion']
    emotion_scores = analysis['emotion']
    
    # Example stress-related emotions
    stress_emotions = ['angry', 'fear', 'sad']
    
    # Calculate a stress level score
    stress_level = sum(emotion_scores[emotion] for emotion in stress_emotions)
    
    return stress_level, emotion

# Custom audio processor for handling audio streams
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = np.array([], dtype=np.int16)
    
    def recv(self, frame):
        audio = frame.to_ndarray()
        self.audio_buffer = np.concatenate((self.audio_buffer, audio), axis=None)
        return frame

def main():
    st.set_page_config(page_title="Medical Chatbot", page_icon="💊")

    # Display the logo
    st.image("logo.png", width=150)  # Update the path and size as needed

    st.sidebar.markdown(developer_info)

    st.title("Medical Chatbot 💉")

    st.write("Ask a medical question or talk to the chatbot:")

    user_question = st.text_input("")

    # Display the video stream
    col1, col2 = st.columns([1, 3])
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="audio-video-stream",
            mode=WebRtcMode.SENDRECV,
            audio_processor_factory=AudioProcessor,
            media_stream_constraints={"audio": True, "video": True},
            async_processing=True,
        )

    with col2:
        if webrtc_ctx.state.playing:
            st.write("Current Stress Level:")
            if webrtc_ctx.video_receiver:
                video_frame = webrtc_ctx.video_receiver.get_frame()
                if video_frame is not None:
                    stress_level, dominant_emotion = analyze_stress_level_from_video(video_frame.to_ndarray())
                    st.write(f"Stress Level: {stress_level}")
                    st.write(f"Dominant Emotion: {dominant_emotion}")

                    # Advice to reduce stress
                    st.write("Advice to Reduce Stress:")
                    st.write("Take a deep breath, go for a walk, and listen to calming music.")

                    # Graph to show stress level over time (mocked for now)
                    st.write("Stress Level Over Time:")
                    x = np.arange(1, 11)
                    y = np.random.randint(1, 10, size=10)
                    fig, ax = plt.subplots()
                    ax.plot(x, y)
                    st.pyplot(fig)
        else:
            st.write("How are you doing today?")

    if user_question:
        handle_userinput(user_question)
    
    if 'stress_level' in st.session_state:
        st.write("Current Stress Level:")
        st.write(st.session_state['stress_level'])

if __name__ == '__main__':
    main()
