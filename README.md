# Audio Restore

AI-powered surface noise removal for historical 78 RPM recordings.

### Quick Start
1. **Setup**: `pip install -r requirements.txt`
2. **Data**: `python src/fetch.py`
3. **Train**: `python src/model.py`
4. **App**: `streamlit run app.py`

### How it works
This project uses a U-Net neural network to clean up old audio. It treats the restoration as an "image-to-image" task by converting sound into spectrograms. By training on noisy original tracks vs. restored versions from the internet archive, the model learns to isolate and remove historical crackle and hiss.

Built for the **Great 78 Project**.
