import os
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Tools for audio-to-image conversion
def get_spectrogram(path):
    y, sr = librosa.load(path, duration=5)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    if S_db.shape[1] < 128:
        S_db = np.pad(S_db, ((0, 0), (0, 128 - S_db.shape[1])))
    return S_db[:128, :128]

class AudioData(Dataset):
    def __init__(self, n_dir, c_dir):
        self.n = sorted([f for f in os.listdir(n_dir) if f.endswith(('.mp3', '.wav'))])
        self.c = sorted([f for f in os.listdir(c_dir) if f.endswith(('.mp3', '.wav'))])
        self.n_dir, self.c_dir = n_dir, c_dir

    def __len__(self): return len(self.n)

    def __getitem__(self, i):
        noisy = get_spectrogram(os.path.join(self.n_dir, self.n[i]))
        clean = get_spectrogram(os.path.join(self.c_dir, self.c[i]))
        return torch.FloatTensor(noisy).unsqueeze(0), torch.FloatTensor(clean).unsqueeze(0)

# Simple cleaning network
class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.up = nn.Sequential(nn.ConvTranspose2d(16, 1, 2, stride=2))

    def forward(self, x): return self.up(self.down(x))

def train():
    os.makedirs("models", exist_ok=True)
    if not os.path.exists("data/noisy"): return
    
    loader = DataLoader(AudioData("data/noisy", "data/clean"), batch_size=1)
    net = TinyUNet()
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    crit = nn.MSELoss()

    print("Training...")
    for epoch in range(10):
        for noisy, clean in loader:
            out = net(noisy)
            loss = crit(out, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    torch.save(net.state_dict(), "models/restorer.pth")
    print("Done.")

if __name__ == "__main__":
    train()
