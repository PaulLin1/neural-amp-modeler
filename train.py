# %%
# import dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import math
from scipy.io import wavfile
from scipy.signal import butter, filtfilt
from scipy.io.wavfile import write
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# %%
def preprocess_audio(path, target_rms=0.1):
    sr, waveform = wavfile.read(path)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    waveform -= np.mean(waveform)
    
    # Normalize amplitude and prevent clipping
    waveform /= (np.max(np.abs(waveform)) + 1e-7)
    waveform *= 0.99
    
    # High-pass filter to remove DC / subsonic
    b, a = butter(1, 20 / (sr / 2), btype='highpass')
    waveform = filtfilt(b, a, waveform)
    
    # RMS normalization
    def rms(x): return np.sqrt(np.mean(x**2))
    waveform *= target_rms / (rms(waveform) + 1e-9)
    
    return sr, waveform

# %%
class AmpDataset(Dataset):
    def __init__(self, data, labels, seq_len):
        if data.ndim == 1:
            data = data.unsqueeze(0)
            labels = labels.unsqueeze(0)

        self.data = data
        self.labels = labels
        self.seq_len = seq_len

        self.num_chunks = data.shape[-1] // seq_len

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[..., start:end]
        y = self.labels[..., start:end]
        return x, y

# %%
sr, clean_quant = preprocess_audio("scale_clean.wav")
sr, amp_quant   = preprocess_audio("scale_amp.wav")

clean_quant = torch.from_numpy(clean_quant.copy()).to(torch.float32)
amp_quant = torch.from_numpy(amp_quant.copy()).to(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# q = mu_law_decode(amp_quant)
# write("outpu11.wav", sr, (clean_quant * 32767).astype(np.int16))

dataset = AmpDataset(clean_quant, amp_quant, 1024)
dl = DataLoader(dataset, 100)

# %%
class CausalLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.channels = channels

        # Double output channels for gated activation
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size, dilation=dilation)

        # 1x1 conv for residual
        self.res_conv = nn.Conv1d(channels, channels, 1)
        # 1x1 conv for skip
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        # pad for causality
        x_padded = F.pad(x, (self.left_pad, 0))
        conv_out = self.conv(x_padded)

        # split for gated activation
        out = torch.tanh(conv_out[:, :self.channels, :]) * torch.sigmoid(conv_out[:, self.channels:, :])

        # skip and residual
        skip = self.skip_conv(out)
        res = x + 0.1 * self.res_conv(out)

        return res, skip


class WaveNet(nn.Module):
    def __init__(self, in_channels=1, channels=16, kernel_size=3, dilations=None):
        super().__init__()
        self.dilations = dilations if dilations is not None else [2 ** i for i in range(10)]
        self.input_proj = nn.Conv1d(in_channels, channels, 1)

        self.blocks = nn.ModuleList([
            CausalLayer(channels, kernel_size, d)
            for d in self.dilations
        ])

        self.output_proj = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        x = self.input_proj(x)
        skip_connections = []

        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # sum all skip connections and apply final projection
        x = sum(skip_connections)
        x = self.output_proj(x)
        return x

# %%
model = WaveNet().to(device)
model = torch.compile(model)
opt = torch.optim.Adam(model.parameters(), lr=0.004)
criterion = nn.MSELoss()

# Training loop
num_epochs = 20000

for epoch in range(num_epochs):
    model.train()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()
        y_pred = model(x)   
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1} / {num_epochs}, Loss: {loss.item():.6f}")
    
torch.save(model.state_dict(), "model.pth")