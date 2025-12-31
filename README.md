# Audio Restore: Rescuing 100-year-old sound with AI 🎻

I spent way too much time listening to the "Great 78 Project" on the Internet Archive and realized that while historical recordings are amazing, the surface noise is... a lot. So I built this.

It's an end-to-end pipeline that takes scratchy, hissing 78 RPM records and uses a U-Net (Deep Learning) to peel away the noise.

### The Problem
If you've ever played a record from the 1920s, you know it sounds like someone is frying bacon in the background. Digital filters help, but they often kill the soul of the music.

### The Fix
I treated this as an "image-to-image" translation task. 
1. **Audio -> Spectrogram**: Convert the sound wave into a visual heatmap of frequencies.
2. **U-Net Magic**: Train a neural network to look at a "noisy" image and predict what the "clean" one looks like.
3. **Spectrogram -> Audio**: Turn that cleaned-up image back into sound.

Does it sound like a modern 4K remaster? No. But it actually brings out the music buried under a century of dust.

---

### How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Grab some data** 
This fetches a sample pair from the Great 78 Project.
```bash
python src/fetch.py
```

**3. Train the "Brain"**
```bash
python src/model.py
```

**4. Launch the dashboard**
```bash
streamlit run app.py
```

---

### Tech Stack
- **PyTorch**: For the U-Net architecture.
- **Librosa**: For all the signal processing/spectrogram stuff.
- **Streamlit**: For the clean (modern-retro) UI.

Shoutout to the **Internet Archive** for keeping history alive. Check out the [Great 78 Project](https://great78.archive.org/) if you haven't yet.
