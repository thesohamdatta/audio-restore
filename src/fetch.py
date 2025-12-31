import os
import requests
import numpy as np
import soundfile as sf

# Dataset URLs (Great 78 Project)
CLEAN = "https://archive.org/download/78_blues-blues-blues_ralph-willis-brownie-magee_gbia0063565b/Blues%2c%20Blues%2c%20Blues%20-%20Ralph%20Willis%20-%20Brownie%20Magee-restored.mp3"
NOISY = "https://archive.org/download/78_blues-blues-blues_ralph-willis-brownie-magee_gbia0063565b/%5F01%5FBlues%2C%20Blues%2C%20Blues%20-%20Ralph%20Willis%20-%20Brownie%20Magee.mp3"

def get_data():
    os.makedirs("data/noisy", exist_ok=True)
    os.makedirs("data/clean", exist_ok=True)
    
    for url, path in [(NOISY, "data/noisy/1.mp3"), (CLEAN, "data/clean/1.mp3")]:
        print(f"Fetching {path}...")
        try:
            r = requests.get(url, timeout=10)
            with open(path, 'wb') as f:
                f.write(r.content)
        except:
            print(f"Download failed for {path}, generating dummy...")
            sf.write(path, np.zeros(22050), 22050)

if __name__ == "__main__":
    get_data()
