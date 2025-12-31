import streamlit as st
import torch
import librosa
import numpy as np
import soundfile as sf
import os
import sys

# Point to source folder
sys.path.append(os.path.join(os.getcwd(), 'src'))
try:
    from model import TinyUNet, get_spectrogram
except ImportError:
    st.error("Missing source files in src/")

st.set_page_config(page_title="Audio Restore", layout="centered")

# Minimal CSS for a clean "Human" look
st.markdown("""
    <style>
    .main { background-color: #fafafa; }
    h1 { font-family: sans-serif; font-weight: 800; color: #111; }
    .stAudio { margin-top: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("Audio Restore")
st.write("Remove surface noise from historical recordings using deep learning.")

file = st.file_uploader("Drop a noisy audio file here", type=["mp3", "wav", "flac"])

if file:
    with open("input.wav", "wb") as f:
        f.write(file.getbuffer())
    
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Noisy Original")
        st.audio("input.wav")

    if st.button("Clean Audio"):
        if not os.path.exists("models/restorer.pth"):
            st.warning("No trained model found. Run src/model.py first.")
        else:
            with st.spinner("Cleaning..."):
                net = TinyUNet()
                net.load_state_dict(torch.load("models/restorer.pth"))
                net.eval()

                # Process
                spec = get_spectrogram("input.wav")
                with torch.no_grad():
                    out = net(torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)).squeeze().numpy()

                # Reconstruct
                y_out = librosa.griffinlim(librosa.db_to_amplitude(out))
                sf.write("cleaned.wav", y_out, 22050)

                with col2:
                    st.caption("Restored Version")
                    st.audio("cleaned.wav")
