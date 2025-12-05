import streamlit as st
import whisper
import os
import tempfile
from datetime import timedelta

st.set_page_config(page_title="TurboScribe AI", page_icon="🚀", layout="centered")
st.markdown("""<style>
    .stButton>button{width:100%; background-color:#00C853; color:white; font-size:18px; border-radius:10px;}
    .reportview-container {background: #f0f2f6}
</style>""", unsafe_allow_html=True)

st.title("🚀 TurboScribe AI")
st.caption("Professional AI Transcription - Unlimited & Fast")

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
    return whisper.load_model("tiny")

uploaded_file = st.file_uploader("📂 Upload Audio/Video (MP3, MP4, WAV)", type=["mp4", "mp3", "wav", "mkv"])

if uploaded_file:
    st.audio(uploaded_file)
    if st.button("⚡ Extract Subtitles Now"):
        with st.spinner('⚡ Processing... Please wait a moment.'):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix="."+uploaded_file.name.split(".")[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    path = tmp.name
                
                model = load_model()
                result = model.transcribe(path, fp16=False)
                
                st.success("✅ Completed! Download your files below:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("⬇️ Download SRT (Subtitles)", segments_to_srt(result["segments"]), file_name="subtitles.srt")
                with col2:
                    st.download_button("⬇️ Download TXT (Text)", result["text"], file_name="transcript.txt")
                
                os.remove(path)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
