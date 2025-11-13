import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import math
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CausalConv1d(nn.Module):
    """1D causal convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad, dilation=dilation
        )
        
    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # remove causal padding
        return out

class ResidualBlock(nn.Module):
    """Residual block with gated activation."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        
    def forward(self, x):
        out = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        skip = self.skip_conv(out)
        res = self.residual_conv(out) + x
        return res, skip

class WaveNet(nn.Module):
    def __init__(self, in_channels=1, channels=16, kernel_size=3, num_blocks=1, dilations=None):
        super().__init__()
        self.causal_in = CausalConv1d(in_channels, channels, kernel_size=1)
        self.dilations = dilations if dilations is not None else [2 ** i for i in range(10)]
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, kernel_size, d) 
            for _ in range(num_blocks) 
            for d in self.dilations
        ])
        self.relu = nn.ReLU()
        self.out1 = nn.Conv1d(channels, channels, 1)
        self.out2 = nn.Conv1d(channels, 1, 1)

    def forward(self, x):
        x = self.causal_in(x)
        skip_connections = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_connections = skip_connections + skip if isinstance(skip_connections, torch.Tensor) else skip
        out = self.relu(skip_connections)
        out = self.relu(self.out1(out))
        out = self.out2(out)
        return out

    @property
    def receptive_field(self):
        rf = 1
        for d in self.dilations:
            rf += (3 - 1) * d
        return rf

class AmpDatasetVectorized(torch.utils.data.Dataset):
    """
    Fully vectorized dataset:
    - Feed entire waveform to model at once
    - No Python slicing loops
    - Output is trimmed to match receptive field
    """
    def __init__(self, clean_wave, amp_wave, model: WaveNet):
        self.x = torch.tensor(clean_wave, dtype=torch.float32).unsqueeze(0)  # (1, L)
        self.y = torch.tensor(amp_wave, dtype=torch.float32).unsqueeze(0)    # (1, L)
        self.rf = model.receptive_field
        assert self.x.shape[-1] >= self.rf, "Waveform too short for receptive field"

    def __len__(self):
        return 1  # Entire waveform in one pass

    def __getitem__(self, idx):
        # Slice outputs to ignore initial zeros from causal padding
        x_input = self.x
        y_target = self.y[:, self.rf - 1:]  # trim first rf-1 samples
        return x_input, y_target

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

sr, clean_quant = preprocess_audio("scale_clean.wav")
sr, amp_quant   = preprocess_audio("scale_amp.wav")

model = WaveNet(
    in_channels=1,
    channels=16,
    kernel_size=3,
    num_blocks=1,
).to(device)

dataset = AmpDatasetVectorized(clean_quant, amp_quant, model)
dl = DataLoader(dataset, batch_size=1, shuffle=False)

opt = torch.optim.Adam(model.parameters(), lr=0.004)
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.993)
criterion = nn.L1Loss()

# Training loop
num_epochs = 20000
for epoch in range(num_epochs):
    model.train()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        opt.zero_grad()
        y_pred = model(x)[:, :, model.receptive_field - 1:]
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1} / {num_epochs}, Loss: {loss.item():.6f}")

torch.save(model, "model.pkl")

