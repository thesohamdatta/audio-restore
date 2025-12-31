import os
import librosa
import numpy as np

def create_spectrogram(audio_path, output_path):
    # 1. Load the audio (first 30 seconds only to keep it fast)
    y, sr = librosa.load(audio_path, duration=30)

    # 2. Convert to a Spectrogram (STFT)
    spectrogram = librosa.stft(y)
    
    # 3. Convert to Decibels (dB)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    # 4. Save as a NumPy array (.npy)
    np.save(output_path, spectrogram_db)

def process_folders():
    # Make sure we have a place to save the 'images'
    os.makedirs("processed/noisy", exist_ok=True)
    os.makedirs("processed/clean", exist_ok=True)

    for subset in ["noisy", "clean"]:
        input_folder = f"data/{subset}"
        if not os.path.exists(input_folder): continue
        
        output_folder = f"processed/{subset}"
        
        for file in os.listdir(input_folder):
            if file.endswith(".flac"):
                print(f"Creating spectrogram for: {file}")
                create_spectrogram(
                    os.path.join(input_folder, file),
                    os.path.join(output_folder, file.replace(".flac", ".npy"))
                )

if __name__ == "__main__":
    process_folders()
