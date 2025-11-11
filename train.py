# import dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


# mu-law for compression
# just a bunch of math that uses log compression to remove harsh sounds
# might not use dont want to compress dynamics
MU = 255

def mu_law_encode(x, mu=MU):
    x = np.clip(x, -1.0, 1.0)
    mag = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    encoded = ((np.sign(x) * mag) + 1) / 2 * mu
    return np.round(encoded).astype(np.int64)

def mu_law_decode(encoded, mu=MU):
    x = (encoded.astype(np.float32) / mu) * 2 - 1
    sign = np.sign(x)
    mag = (1 / mu) * ((1 + mu) ** np.abs(x) - 1)
    return sign * mag



class CausalConv1d(nn.Module):
    """A simple causal 1D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad != 0 else out


class ResidualBlock(nn.Module):
    """A single non-gated residual block."""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        res = self.residual(out) + x[:, :, -out.shape[2]:]  # Align temporal dims
        skip = self.skip(out)
        return res, skip


class SimpleWaveNet(nn.Module):
    """A minimal WaveNet implementation (no gating, no conditioning)."""
    def __init__(self,
                 in_channels=1,
                 residual_channels=64,
                 skip_channels=128,
                 kernel_size=2,
                 num_layers=10):
        super().__init__()
        self.input_conv = CausalConv1d(in_channels, residual_channels, 1)

        dilations = [2 ** i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            ResidualBlock(residual_channels, kernel_size, d) for d in dilations
        ])

        self.skip_conv1 = nn.Conv1d(residual_channels, skip_channels, 1)
        self.skip_conv2 = nn.Conv1d(skip_channels, in_channels, 1)

    def forward(self, x):
        """
        x: (B, 1, L)
        returns: (B, 1, L_out)
        """
        x = self.input_conv(x)
        skip_total = 0

        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip_total + skip if isinstance(skip_total, torch.Tensor) else skip

        out = F.relu(self.skip_conv1(F.relu(skip_total)))
        out = self.skip_conv2(out)
        return out


class AmpDataset(torch.utils.data.Dataset):
    def __init__(self, clean_wave, amp_wave, seq_len):
        assert len(clean_wave) == len(amp_wave), "Input and target lengths must match"
        self.x = torch.tensor(clean_wave, dtype=torch.float)
        self.y = torch.tensor(amp_wave, dtype=torch.float)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.x) - self.seq_len
    
    def __getitem__(self, idx):
        return self.x[idx:idx + self.seq_len], self.y[idx + 1:idx + self.seq_len + 1]
    
    # process input stuff
from scipy.io import wavfile

# Process clean
sr, waveform = wavfile.read("samples/clean/double stop.wav")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

waveform = waveform.astype(np.float32)
waveform /= np.abs(waveform).max()
clean_quant = mu_law_encode(waveform)

# Process amped
sr, waveform = wavfile.read("samples/real/double stop.wav")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

waveform = waveform.astype(np.float32)
waveform /= np.abs(waveform).max()
amp_quant = mu_law_encode(waveform)

clean_tensor = torch.tensor(clean_quant, dtype=torch.float)
amp_tensor   = torch.tensor(amp_quant, dtype=torch.float)

dataset = AmpDataset(clean_quant, amp_quant, seq_len=1024)
# dl = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
dl = DataLoader(dataset, batch_size=4096, pin_memory=True, num_workers=8, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleWaveNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.004)
criterion = nn.CrossEntropyLoss()

model = torch.compile(model)
for i in range(50):
    print(i)
    for epoch, (x, y) in enumerate(dl):
        # if epoch % 5000 == 0:
        print(f'{epoch} / {len(dl)}')
        x, y = x.to(device), y.to(device)
        cond = x  # clean input (or use different clean signal tensor)
        logits = model(x.unsqueeze(1))
        # match sequence length
        logits = logits[:, :, -y.shape[1]:]
        loss = criterion(logits.squeeze(1), y)

        opt.zero_grad()
        loss.backward()
        opt.step()

clean_tensor = torch.tensor(clean_quant, dtype=torch.long)
amp_tensor   = torch.tensor(amp_quant, dtype=torch.long)

with torch.no_grad():
    out_logits = model(clean_tensor.unsqueeze(1).to(device))
    pred = torch.argmax(out_logits, dim=1).cpu().numpy().flatten()
    audio = mu_law_decode(pred)
    audio = audio.astype(np.float32)
    # audio /= np.abs(audio).max() + 1e-8

# z = mu_law_decode(audio)
wavfile.write("amp_output_13.wav", sr, (audio * 32767).astype(np.int16))
print("Saved to amp_output.wav")
