# Deep Learning for Guitar Tablature Transcription from Monophonic Audio
 
A deep learning system that listens to a guitar being played and generates tablature in real time. The system captures individual notes through gate-based onset detection, predicts the string (1–6) and fret (0–22) using a trained CNN, and displays the results as interactive tablature in a browser-based interface.
 
**Author:** Dylan Hough  
**Supervisor:** Dr Saif Alzubi  
**Module:** ECM3401 Individual Project, University of Exeter  
 
---
 
## Overview
 
Guitar tablature specifies *where* on the neck a note is played (which string, which fret), not just *what pitch*. This is a harder problem than pitch detection because the same note can be played at multiple positions. For example, A2 can be played on string 1 fret 5, string 2 fret 0, or string 3 fret 7.
 
This project tackles this with a CNN that classifies audio into one of **138 joint (string × fret) positions**, forcing same-pitch alternatives to compete directly in a single softmax. The model achieves **99.9% accuracy** on held-out data from the same recording session and **75.0%** on separately-recorded test notes.
 
---
 
## Project Structure
 
```
noteCapture/
├── main.py                  # Entry point — starts HTTP server + browser UI
├── guitar_tab.html          # Browser-based tablature interface
├── dissertation/
│   ├── __init__.py
│   ├── predict.py           # Model definition + inference (predict function)
│   └── guitar_capture.py    # Live audio capture with gate-based onset detection
├── guitar_trainer.ipynb     # Model training notebook (run on GPU machine)
├── checkpoints_v5/
│   └── best_v5.pt           # Trained model weights
├── sessions/                # Session JSON files (created at runtime)
├── recordings/              # Captured WAV files (created at runtime)
├── data/                    # Training data (not included — see Dataset section)
│   ├── string_1_fret_0/
│   │   ├── 001.wav
│   │   └── ...
│   └── string_6_fret_22/
├── test_notes/              # Separately recorded test notes
│   ├── 10.wav               # string 1, fret 0
│   ├── 37.wav               # string 3, fret 7
│   └── ...
├── Requirements.txt
└── README.md
```
 
---
 
## Setup
 
### Prerequisites
 
- Python 3.10+
- audio interface for live capture
- The trained model checkpoint (`checkpoints_v5/best_v5.pt`)
### Installation
 
```bash
# Clone or download the project
cd noteCapture
 
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# or: venv\Scripts\activate     # Windows
 
# Install dependencies
pip install -r Requirements.txt
```
 
#### Platform-specific notes
 
**Windows (with NVIDIA GPU):**
```bash
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```
 
**macOS (CPU/MPS):**
```bash
pip install torch torchaudio
```
 
**PyAudio** (required for live audio capture):
- **macOS:** `brew install portaudio && pip install pyaudio`
- **Windows:** `pip install pyaudio`
- **Linux:** `sudo apt install portaudio19-dev && pip install pyaudio`
---
 
## Usage
 
### Live Tablature (Full System)
 
```bash
python main.py
```
 
This starts a local HTTP server and opens the browser interface. From the UI you can:
 
1. **Select an audio input device** (your guitar interface)
2. **Create or select a session** to group notes
3. **Click Start** — the system calibrates the noise floor (~2 seconds of silence), then listens
4. Play single notes on your guitar — each detected note appears as tablature in the browser
The system uses gate-based onset detection: it measures your room's noise floor during calibration, then triggers recording when the audio level exceeds the threshold. Each captured note is saved as a WAV file, run through the model, and the predicted string/fret is appended to the session and displayed.
 
### Standalone Prediction
 
To predict string and fret from a WAV file without the live system:
 
```python
from dissertation.predict import get_model, get_string_fret_from_wav
 
model = get_model("checkpoints_v5/best_v5.pt")
string, fret = get_string_fret_from_wav("test_notes/37.wav", model=model)
print(f"String: {string}, Fret: {fret}")
# Output: String: D3, Fret: 7
```
 
---
 
## Model
 
### Architecture
 
The model is a residual CNN with a **joint 138-class position head** instead of separate string and fret predictions, it classifies directly into one of 138 (string × fret) positions. This forces same-pitch alternatives to compete in a single softmax.
 
```
Input: 5 × 128 × T feature tensor
  ↓
Conv2d(5→32) + ResBlock + MaxPool
Conv2d(32→64) + 2×ResBlock + MaxPool
Conv2d(64→128) + 2×ResBlock + MaxPool
Conv2d(128→256) + ResBlock + AdaptiveAvgPool
  ↓
FC 4096→512
  ↓
Primary: Joint Position Head → 138 classes
Auxiliary: String Head → 6 classes  (regularisation only)
Auxiliary: Fret Head → 23 classes   (regularisation only)
```
 
### Features (5 channels)
 
| Channel | Feature | Purpose |
|---------|---------|---------|
| 0 | Mel spectrogram (dB) | Overall energy and pitch |
| 1 | Harmonic blend | Sustained pitch emphasis |
| 2 | Delta mel | Onset transient shape |
| 3 | MFCC | Timbral fingerprint (wound vs plain string) |
| 4 | Attack window (first 150ms) | String-specific pluck character |
 
### Results
 
| Metric | Accuracy |
|--------|----------|
| Held-out test set (same session) | 99.9% |
| Separately-recorded test notes (48 notes) | 75.0% |
 
All remaining errors on the separate test set are **same-pitch string confusions** — the model identifies the correct pitch every time but occasionally selects the wrong string position. This is a fundamental acoustic ambiguity where adjacent strings at the same pitch produce near-identical spectral content.
 
---
 
## Training
 
Training requires a CUDA GPU and the dataset (see below). Open `guitar_trainer.ipynb` in Jupyter and run all cells. Training takes approximately 1 hour on an RTX 2060.
 
### Dataset
 
The training data consists of **11,802 WAV recordings** organised as:
 
```
data/
├── string_1_fret_0/     # 70-100 .wav files
├── string_1_fret_1/
├── ...
└── string_6_fret_22/
```
 
Each file is a single note recorded on the specified string and fret. The dataset is not included in this repository due to size.
 
contact me through Dylanpjhough@gmail.com about aquiring the dataset
---
 
## Technologies
 
| Component | Technology |
|-----------|-----------|
| Deep learning | PyTorch, torchaudio |
| Audio capture | PyAudio, gate-based onset detection |
| Web interface | HTML/CSS/JavaScript, local HTTP server |
| Training hardware | NVIDIA RTX 2060 (CUDA) |
 
---
 
## Acknowledgements
 
Supervised by Dr Saif Alzubi, University of Exeter.