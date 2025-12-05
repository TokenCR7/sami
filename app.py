import streamlit as st
import whisper
import os
import tempfile
from datetime import timedelta

# Page Setup
st.set_page_config(page_title="TurboScribe Pro", page_icon="💎", layout="centered")

# Custom CSS to hide "Manage App", Footer, Header, and Menu
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {visibility: hidden;}
        [data-testid="stStatusWidget"] {visibility: hidden;}
        
        .stButton>button {
            width: 100%;
            background-color: #2e7af9;
            color: white;
            font-size: 20px;
            border-radius: 8px;
            padding: 12px;
            border: none;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #1a5cce;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💎 TurboScribe Pro")
st.caption("High Accuracy AI Transcription (100% Precise)")

def format_timestamp(seconds: float):
    delta = timedelta(seconds=seconds)
    total_seconds = int(delta.total_seconds())
    ms = int((delta.total_seconds() - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

def segments_to_srt(segments):
    srt_output = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_output.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_output)

@st.cache_resource
def load_model():
    return whisper.load_model("small")

uploaded_file = st.file_uploader("📂 Upload Audio/Video File", type=["mp4", "mp3", "wav", "mkv", "mov", "m4a", "avi"])

if uploaded_file:
    # Play Feature
    file_type = uploaded_file.name.split('.')[-1].lower()
    if file_type in ['mp4', 'mkv', 'mov', 'avi']:
        st.video(uploaded_file)
    else:
        st.audio(uploaded_file)

    if st.button("⚡ Start High Accuracy Extraction"):
        with st.spinner('Processing with High Precision... (Please wait)'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                model = load_model()
                
                # High Accuracy Settings
                result = model.transcribe(
                    path, 
                    fp16=False, 
                    beam_size=5, 
                    temperature=0
                )
                
                st.success("✅ Extraction Complete!")
                
                st.text_area("Extracted Text:", result["text"], height=200)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇️ Download SRT", segments_to_srt(result["segments"]), file_name="subtitles.srt")
                with col2:
                    st.download_button("⬇️ Download TXT", result["text"], file_name="transcript.txt")
                
                os.remove(path)
            except Exception as e:
                st.error(f"Error: {e}")
