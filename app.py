
import streamlit as st
import os
import uuid
import tempfile
import yt_dlp
from moviepy.editor import AudioFileClip, VideoFileClip
import whisper
from transformers import pipeline
from gtts import gTTS
import nltk

# Download necessary NLTK data only if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def download_youtube_audio(url, out_dir):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        audio_file = ydl.prepare_filename(info_dict)
        audio_file = audio_file.rsplit('.', 1)[0] + '.mp3'
    return audio_file

def extract_audio(file_path, out_dir):
    ext = file_path.split('.')[-1].lower()
    audio_path = os.path.join(out_dir, f"{uuid.uuid4()}.mp3")
    if ext in ['mp3', 'wav']:
        return file_path
    elif ext == 'mp4':
        clip = VideoFileClip(file_path)
        clip.audio.write_audiofile(audio_path)
        return audio_path
    else:
        raise ValueError("Unsupported file format")

@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")

def transcribe_audio(audio_path):
    model = get_whisper_model()
    result = model.transcribe(audio_path)
    return result['text']

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = []
    for i in range(0, len(text), 1024):
        chunk = text[i:i+1024]
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        chunks.append(summary)
    return " ".join(chunks)

def text_to_speech(text, out_path):
    tts = gTTS(text)
    tts.save(out_path)
    return out_path

# Streamlit UI
st.set_page_config(page_title="Podcast Summarizer", layout="wide")
st.title("🎙️ Podcast Summarizer")

if 'audio_file' not in st.session_state:
    st.session_state['audio_file'] = None

with st.sidebar:
    st.header("Upload or Link")
    upload_file = st.file_uploader("Upload podcast audio/video (.mp3, .wav, .mp4)", type=['mp3', 'wav', 'mp4'])
    yt_url = st.text_input("Or paste a YouTube video link:")

    if upload_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{upload_file.name.split('.')[-1]}") as tmp:
            tmp.write(upload_file.read())
            st.session_state['audio_file'] = extract_audio(tmp.name, tempfile.gettempdir())
            st.success("File uploaded and processed!")
    elif yt_url:
        try:
            st.session_state['audio_file'] = download_youtube_audio(yt_url, tempfile.gettempdir())
            st.success("YouTube audio downloaded!")
        except Exception as e:
            st.error(f"Failed to download: {e}")

tabs = st.tabs(["Transcript", "Summary", "Audio Summary"])

with tabs[0]:
    st.subheader("Transcript")
    if st.session_state['audio_file']:
        with st.spinner("Transcribing..."):
            try:
                transcript = transcribe_audio(st.session_state['audio_file'])
                st.session_state['transcript'] = transcript
                st.text_area("Transcript", transcript, height=300)
            except Exception as e:
                st.error(f"Transcription failed: {e}")
    else:
        st.info("Please upload a file or provide a YouTube link.")

with tabs[1]:
    st.subheader("Summary")
    if st.session_state.get('transcript'):
        with st.spinner("Summarizing..."):
            try:
                summary = summarize_text(st.session_state['transcript'])
                st.session_state['summary'] = summary
                st.text_area("Summary", summary, height=200)
            except Exception as e:
                st.error(f"Summarization failed: {e}")
    else:
        st.info("Transcript not available.")

with tabs[2]:
    st.subheader("Audio Summary")
    if st.session_state.get('summary'):
        audio_summary_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        try:
            text_to_speech(st.session_state['summary'], audio_summary_path)
            with open(audio_summary_path, 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/mp3')
        except Exception as e:
            st.error(f"Audio summary failed: {e}")
    else:
        st.info("Summary not available.")

st.caption("| Podcast Summarizer |")
