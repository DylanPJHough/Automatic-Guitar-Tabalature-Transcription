"""
Guitar Tablature Predictor — Standalone Inference Script
Predicts string (1-6) and fret (0-22) from a .wav file.

Usage:
    python predict.py path/to/note.wav
    python predict.py path/to/note.wav --no-tta
    python predict.py path/to/folder/  --batch

Requires: checkpoints_v5/best_v5.pt (trained model weights)
"""

import sys
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as AF


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

N_STRINGS = 6
N_FRETS = 23  # 0-22
SR = 16000
CLIP_DUR = 2.0
CLIP_LEN = int(SR * CLIP_DUR)
TTA_RUNS = 9

N_MELS = 128
N_FFT = 2048
HOP = 256
N_CH = 5
ATTACK_LEN = int(SR * 0.15)

OPEN_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
STR_NAMES = ["E2 (low E)", "A2", "D3", "G3", "B3", "E4 (high e)"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHECKPOINT_PATH = Path("checkpoints_v5/best_v5.pt")

# ── Position maps ──
POS_LIST = []
POS_TO_IDX = {}
for s in range(N_STRINGS):
    for f in range(N_FRETS):
        POS_TO_IDX[(s, f)] = len(POS_LIST)
        POS_LIST.append((s, f))
N_POS = len(POS_LIST)


# ═══════════════════════════════════════════════════════════
#  Model Definition
# ═══════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, ch, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.GELU(), nn.Dropout2d(drop),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class GuitarNetV5(nn.Module):
    def __init__(self, n_pos=138, n_str=6, n_fret=23, in_ch=5, drop=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            ResBlock(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            ResBlock(64), ResBlock(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            ResBlock(128), ResBlock(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.GELU(),
            ResBlock(256), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.flatten = nn.Flatten()
        self.shared = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.GELU(), nn.Dropout(drop),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(drop / 2),
            nn.Linear(256, n_pos),
        )
        self.str_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(drop / 3),
            nn.Linear(128, n_str),
        )
        self.fret_head = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(drop / 3),
            nn.Linear(128, n_fret),
        )

    def forward(self, x):
        emb = self.shared(self.flatten(self.backbone(x)))
        return self.pos_head(emb), self.str_head(emb), self.fret_head(emb)


# ═══════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════

_mel = T.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
_db = T.AmplitudeToDB(top_db=80)
_mfcc = T.MFCC(sample_rate=SR, n_mfcc=20,
               melkwargs={'n_fft': N_FFT, 'hop_length': HOP, 'n_mels': N_MELS})


def _rsz(t2d, h, w):
    return F.interpolate(t2d[None, None], (h, w), mode='bilinear',
                         align_corners=False)[0, 0]


def pad_or_trim(x, n):
    if x.shape[-1] >= n:
        return x[..., :n]
    return F.pad(x, (0, n - x.shape[-1]))


def extract_features(x):
    """Extract 5-channel features from waveform tensor (1, samples)."""
    mel = _db(_mel(x))
    Tf = mel.shape[-1]
    harm = F.avg_pool2d(mel[None], (1, 9), 1, (0, 4))[0]
    delta = F.pad(mel[..., 1:] - mel[..., :-1], (1, 0))
    mfcc_r = _rsz(_mfcc(x).squeeze(0), N_MELS, Tf)[None]
    att = _db(_mel(x[..., :ATTACK_LEN]))
    att_r = _rsz(att.squeeze(0), N_MELS, Tf)[None]

    feat = torch.cat([mel, harm, delta, mfcc_r, att_r], 0)  # (5, 128, T)
    for c in range(N_CH):
        ch = feat[c]
        feat[c] = (ch - ch.mean()) / (ch.std() + 1e-6)
    return feat


def augment_tta(x):
    """Light augmentation for test-time augmentation."""
    if random.random() < 0.5:
        x = x * random.uniform(0.8, 1.2)
    if random.random() < 0.3:
        snr = random.uniform(30, 50)
        n = torch.randn_like(x)
        sp = x.pow(2).mean().clamp(min=1e-9)
        np_ = n.pow(2).mean().clamp(min=1e-9)
        x = x + n * torch.sqrt(sp / (np_ * 10 ** (snr / 10)))
    if random.random() < 0.3:
        s = random.randint(-CLIP_LEN // 16, CLIP_LEN // 16)
        if s > 0:
            x = torch.cat([torch.zeros(1, s), x[:, :-s]], 1)
        elif s < 0:
            x = torch.cat([x[:, -s:], torch.zeros(1, -s)], 1)
    return x


# ═══════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════

def _get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU for inference.")
        return torch.device("cuda")
    
    print("Using CPU for inference.")
    return torch.device("cpu")


_device = _get_device()
_model = None  # lazy loaded


def _load_model(checkpoint_path=None):
    global _model
    if _model is not None:
        return _model

    cp = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    if not cp.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{cp}'.\n"
            f"Train the model first using the v6 notebook, or provide the correct path."
        )

    _model = GuitarNetV5(n_pos=N_POS, in_ch=N_CH).to(_device)
    state = torch.load(cp, map_location=_device, weights_only=True)
    # Handle both formats: raw state_dict or {'state': state_dict}
    if isinstance(state, dict) and 'state' in state:
        _model.load_state_dict(state['state'])
    else:
        _model.load_state_dict(state)
    _model.eval()
    return _model


# ═══════════════════════════════════════════════════════════
#  Predict Function
# ═══════════════════════════════════════════════════════════

def predict(wav_path, tta=True, model=None):
    """
    Predict string and fret from a .wav file.

    Args:
        wav_path:        Path to a .wav recording of a single guitar note.
        tta:             Use test-time augmentation (9 augmented passes).
                         Slightly slower but more robust.
        checkpoint_path: Path to model checkpoint (default: checkpoints_v5/best_v5.pt)

    Returns:
        dict with keys:
            'string'     : int 1-6 (1=low E, 6=high e)
            'fret'       : int 0-22
            'string_name': str e.g. "E2 (low E)"
            'note'       : str e.g. "A"
            'midi'       : int MIDI note number
            'confidence' : float 0-1
            'top3'       : list of (string, fret, confidence) tuples
    """
    

    # Load and preprocess audio
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    wav = pad_or_trim(wav, CLIP_LEN)
    pk = wav.abs().max()
    if pk > 1e-6:
        wav = wav / pk

    # Run inference (with optional TTA)
    runs = TTA_RUNS if tta else 1
    logsum = None

    with torch.no_grad():
        for i in range(runs):
            x = wav.clone() if i == 0 else augment_tta(wav.clone())
            feat = extract_features(x).unsqueeze(0).to(_device)
            with torch.amp.autocast('cuda', enabled=_device.type == 'cuda'):
                p_log, _, _ = model(feat)
            logsum = p_log if logsum is None else logsum + p_log

    # Decode prediction
    probs = F.softmax(logsum / runs, 1).squeeze(0)
    pi = probs.argmax().item()
    s, f = POS_LIST[pi]
    conf = probs[pi].item()
    midi = OPEN_MIDI[s] + f
    note = NOTE_NAMES[midi % 12]

    # Top 3 alternatives
    top3_vals, top3_idx = probs.topk(3)
    top3 = []
    for v, idx in zip(top3_vals, top3_idx):
        ts, tf = POS_LIST[idx.item()]
        top3.append((ts + 1, tf, v.item()))

    return {
        'string': s + 1,
        'fret': f,
        'string_name': STR_NAMES[s],
        'note': note,
        'midi': midi,
        'confidence': conf,
        'top3': top3,
    }

def predict2(wav_path, tta=True, model=None):
    # Load and preprocess audio
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = AF.resample(wav, sr, SR)
    wav = pad_or_trim(wav, CLIP_LEN)
    pk = wav.abs().max()
    if pk > 1e-6:
        wav = wav / pk

    # Build all augmented versions at once, then extract features in one batch
    runs = TTA_RUNS if tta else 1
    wavs = [wav.clone() if i == 0 else augment_tta(wav.clone()) for i in range(runs)]
    feats = torch.stack([extract_features(w) for w in wavs], dim=0).to(_device)

    # Single batched forward pass instead of 9 sequential ones
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=_device.type == 'cuda'):
            p_log, _, _ = model(feats)          # (runs, 138)
    logsum = p_log.sum(0, keepdim=True)          # (1, 138)

    # Decode prediction
    probs = F.softmax(logsum / runs, 1).squeeze(0)
    pi = probs.argmax().item()
    s, f = POS_LIST[pi]
    conf = probs[pi].item()
    midi = OPEN_MIDI[s] + f
    note = NOTE_NAMES[midi % 12]

    top3_vals, top3_idx = probs.topk(3)
    top3 = [(POS_LIST[idx.item()][0] + 1, POS_LIST[idx.item()][1], v.item())
            for v, idx in zip(top3_vals, top3_idx)]

    return {
        'string': s + 1,
        'fret': f,
        'string_name': STR_NAMES[s],
        'note': note,
        'midi': midi,
        'confidence': conf,
        'top3': top3,
    }
def predict_and_print(wav_path, tta=True, model=None):
    """Predict and print results to console."""
    result = predict(wav_path, tta=tta, model=model)
    tta_tag = f" (TTA×{TTA_RUNS})" if tta else ""
    print(f"  String : {result['string_name']} (string {result['string']})")
    print(f"  Fret   : {result['fret']}")
    print(f"  Note   : {result['note']} (MIDI {result['midi']})")
    print(f"  Conf   : {result['confidence']:.1%}{tta_tag}")
    alts = ", ".join(f"s{s}f{f}({c:.0%})" for s, f, c in result['top3'])
    print(f"  Top 3  : {alts}")
    return result

def get_string_fret_from_wav(wav_path, model=None):
    """Predict and return string and fret as a tuple (string_name, fret)."""
    result = predict(wav_path, tta=True, model=model)
    return result['string_name'], result['fret']

def get_model(checkpoint_path=None):
    """Load and return the model (for external use)."""
    return _load_model(checkpoint_path)

# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════



