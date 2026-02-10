# CoolASL - Real-Time ASL Classifier/Translator**


### Roadmap

- **Phase 1**** (current): Static alphabet recognition (A-Z)
- **Phase 2**: Dynamic word-level gesture recognition and translation using WLASL dataset
- **Phase 3**: Interactive ASL practice tool with LLM-powered tutor

Real-time American Sign Language alphabet classifier using hand landmark detection. Recognizes A-Z hand signs from a webcam feed with high accuracy.

## How It Works

```
Webcam → MediaPipe Hand Landmarks → Normalize → LandmarkClassifier MLP → Prediction Overlay
```

MediaPipe detects 21 hand landmarks (x, y, z) per frame. These 63 coordinates are normalized relative to the wrist and scaled by hand size, then classified by a lightweight MLP.

### Why Landmarks Instead of Raw Images?

The project started with a ResNet18 CNN trained on the ASL-HG dataset. It hit 99%+ validation accuracy but performed poorly on real webcam input due to domain gap. The training images have clean, consistent backgrounds while webcam feeds vary wildly in lighting, background, and skin tone.

Switching to landmark-based classification eliminated this problem entirely. Landmarks are invariant to background, lighting, and appearance since only hand geometry matters. The LandmarkClassifier is also significantly faster and smaller (222KB vs 45MB).

## Quick Start

**Prerequisites:** Python 3.10+, a webcam

```bash
# Clone and setup
git clone https://github.com/speakerchef/CoolASL.git
cd CoolASL
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**(MUST!) Download the MediaPipe hand landmarker model** from [Google's MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place `hand_landmarker.task` in the `models/` directory.

**(Optional: Use only if retraining) Download the dataset** from [ASL-HG on Mendeley](https://data.mendeley.com/datasets/j4y5w2c8w9/1) and extract to `data/asl_processed/` with `train/` and `test/` subdirectories containing folders A-Z.

```bash
# Enter source dir
cd src

# Run inference on camera feed
python inference.py
```

**If you would like to re-train:**
```bash
# Enter source dir
cd src

# Load data (this will take a while)
python dataloader.py

# Train
python train.py

# Run inference on camera feed
python inference.py
```

Press `q` to quit the webcam feed.

## Project Structure

```
CoolASL/
├── configs/
│   ├── config.yaml          # Hyperparameters and paths
│   └── cfg_importer.py      # Config loader
├── src/
│   ├── model.py             # LandmarkClassifier MLP + ResNet18 (deprecated)
│   ├── dataloader.py        # Landmark extraction, normalization, data loading
│   ├── train.py             # Training loops with early stopping
│   └── inference.py         # Real-time webcam inference pipeline
├── models/                  # Trained weights (download from Releases)
├── data/                    # Dataset (download separately)
├── logs/                    # Training metrics
├── notebooks/               # Visualization notebooks
└── requirements.txt
```

## Technical Details

**LandmarkClassifier architecture:** 63 → 256 → 128 → 26 with BatchNorm, ReLU, and Dropout.

**Landmark normalization:** Coordinates are centered on the wrist, scaled by max hand span.

**Dynamic bounding box:** The inference pipeline computes bounding boxes using the max span between the wrist and all five fingertips, with exponential decay scaling to adapt to varying hand distances from the camera.

**Dataset:** ASL-HG from Mendeley. 36,000 images across 10 signers with varied lighting and skin tones. Numbers (0-9) were excluded from training because several share hand positions with letters.

## Configuration

Edit `configs/config.yaml` to adjust:

```yaml
MODEL_MODE: 'lmc'      # 'lmc' for LandmarkClassifier, 'resnet18' for ResNet
batch_size: 256
num_epochs: 100
lr: 0.001
```


