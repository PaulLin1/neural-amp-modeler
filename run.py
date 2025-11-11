# import dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
import math

# mu-law for compression
# just a bunch of math that uses log compression to remove harsh sounds
# might not use dont want to compress dynamics
MU = 255
def mu_law_encode(x, mu=MU):
    if isinstance(x, torch.Tensor):
        sign = torch.sign(x)
        mag = torch.log1p(mu * x.abs()) / math.log1p(mu)
        return ((sign * mag + 1) / 2 * mu).long()
    else:
        x = np.clip(x, -1, 1)
        mag = np.log1p(mu * np.abs(x)) / np.log1p(mu)
        encoded = np.sign(x) * mag
        return ((encoded + 1) / 2 * mu).astype(np.int64)


def mu_law_decode(encoded, mu=MU):
    if isinstance(encoded, torch.Tensor):
        enc = encoded.float()
        x = 2 * (enc / mu) - 1
        sign = torch.sign(x)
        mag = (1 / mu) * ((1 + mu) ** x.abs() - 1)
        return sign * mag
    else:
        x = 2 * (encoded.astype(np.float32) / mu) - 1
        sign = np.sign(x)
        mag = (1 / mu) * ((1 + mu) ** np.abs(x) - 1)
        return sign * mag
    

# Causal Conv1d wrapper
# Just a conv layer thats causal meaning that it can see into the future
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        # x: (batch, channel, timesteps)
        out = self.conv(x)
        if self.pad:
            return out[:, :, :-self.pad]  # remove future context
        return out

# resid block. you dont need to understand this.
# basically this is what makes wavenet a wave
class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.filter_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(residual_channels, residual_channels, kernel_size, dilation)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        # x: (batch, res, timesteps)
        f = self.filter_conv(x)
        g = self.gate_conv(x)
        # gated activation unit
        out = torch.tanh(f) * torch.sigmoid(g)
        skip = self.skip_conv(out)
        res = self.res_conv(out)
        res = res + x  # residual connection
        return res, skip # return pair of the output and skip conn

class ConditionalWaveNet(nn.Module):
    def __init__(self, n_quantize=256, residual_channels=32, skip_channels=64,
                 kernel_size=2, dilations=None):
        super().__init__()
        if dilations is None:
            # this is the wavenet patter youll see in the imgaes
            # like th wave image
            dilations = [1, 2, 4, 8]

        self.n_quantize = n_quantize
        self.embedding = nn.Embedding(n_quantize, residual_channels)
        self.causal_in = CausalConv1d(residual_channels, residual_channels, kernel_size=1)

        # conditioning pathway (for clean input)
        self.condition_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(residual_channels, skip_channels, kernel_size, d)
            for d in dilations
        ])

        self.relu = nn.ReLU()
        self.post1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.post2 = nn.Conv1d(skip_channels, n_quantize, kernel_size=1)

    def forward(self, x, cond):
        # x: (target, input, sequence) (amp’d)
        # cond: conditioning clean sequence (same length)

        x = self.embedding(x).permute(0, 2, 1).contiguous()
        cond = self.embedding(cond).permute(0, 2, 1).contiguous()

        x = self.causal_in(x)
        cond = self.condition_conv(cond)

        skip_sum = 0
        for block in self.res_blocks:
            # inject conditioning
            x, skip = block(x + cond)
            skip_sum = skip_sum + skip if isinstance(skip_sum, torch.Tensor) else skip

        out = self.relu(skip_sum)
        out = self.post1(out)
        out = self.relu(out)
        out = self.post2(out)
        return out

# This is how i load in my wave fiels
class AmpDataset(torch.utils.data.Dataset):
    def __init__(self, clean_wave, amp_wave, seq_len):
        assert len(clean_wave) == len(amp_wave), "Input and target lengths must match"
        self.x = torch.tensor(clean_wave, dtype=torch.long)
        self.y = torch.tensor(amp_wave, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.x) - self.seq_len

    def __getitem__(self, idx):
        x = self.x[idx:idx + self.seq_len]
        y = self.y[idx + 1:idx + self.seq_len + 1]
        return x, y
    
# process input stuff
from scipy.io import wavfile

# Process clean
sr, waveform = wavfile.read("samples_new/untitled_2025-11-04 16-58-28_Insert 1 - Part_1.wav")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

waveform = waveform.astype(np.float32)
waveform /= np.abs(waveform).max()
clean_quant = mu_law_encode(waveform)

# Process amped
sr, waveform = wavfile.read("samples_new/Untitled project - _5 - Part_1.wav")
if waveform.ndim > 1:
    waveform = waveform.mean(axis=1)

waveform = waveform.astype(np.float32)
waveform /= np.abs(waveform).max()
amp_quant = mu_law_encode(waveform)


dataset = AmpDataset(clean_quant, amp_quant, seq_len=512)
dl = DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalWaveNet().to(device)
model = torch.compile(model)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


for a in range(50):
    print(a)
    for epoch, (x, y) in enumerate(dl):
        if epoch % 100:
            print(f'{epoch} / {len(dl)}')
        x, y = x.to(device), y.to(device)
        cond = x  # clean input (or use different clean signal tensor)
        logits = model(x, cond)
        # match sequence length
        logits = logits[:, :, -y.shape[1]:]
        loss = criterion(logits.permute(0, 2, 1).reshape(-1, model.n_quantize), y.reshape(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

from scipy.io import wavfile

clean_tensor = torch.tensor(clean_quant, dtype=torch.long)
amp_tensor   = torch.tensor(amp_quant, dtype=torch.long)

with torch.no_grad():
    out_logits = model(clean_tensor.unsqueeze(0).to(device), clean_tensor.unsqueeze(0).to(device))
    pred = torch.argmax(out_logits, dim=1).cpu().numpy().flatten()
    audio = mu_law_decode(pred / 255.0)
    audio = audio.astype(np.float32)
    audio /= np.abs(audio).max() + 1e-8

wavfile.write("amp_output2.wav", sr, (audio * 32767).astype(np.int16))