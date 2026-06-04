# PiCar — Autonomous RC Car

Behavioural cloning pipeline for a SunFounder PiCar-V (Raspberry Pi 4) trained to drive autonomously around a track.

## Project Phases

### Phase 1 — Kaggle Challenge
A Kaggle competition to predict steering angle and speed from dashcam images using a CNN with MobileNetV3 backbone, trained on ~14k labelled images.

**Best private leaderboard MSE: 0.01342 (3rd/6)**

### Phase 2 — Live Testing
Live on-track testing across 12 scenarios: lane keeping, stopping for pedestrians/objects, and responding to traffic signs across T-junction, oval, and figure-of-eight tracks. The live model was rebuilt from the Kaggle model with several key changes:

- **Angle-only output** — speed fixed at default; stopping handled by a rule-based layer
- **Rule-based obstacle detection** — HSV floor segmentation in a centre-lane ROI; only fires when driving straight to avoid cornering false positives
- **Improved training** — WeightedRandomSampler upweights corner frames 5x; horizontal flip inverts the angle label to balance left/right distribution
- **Self-collected data** — ~3k additional images with per-source bottom cropping to account for a lower camera position, verified with Grad-CAM

## Repository Structure

```
PiCar/
├── kaggle_model/                  # Phase 1: Kaggle submission
│   ├── data.py                   
│   ├── main.py                    # Training script
│   ├── model.py                   # Inference wrapper (deployed on Pi)
│   ├── picarnet.py                # Model architecture
│   ├── seed.py                    
│   ├── train.py                   
│   └── outputs/training_curves/   
│
├── live_model/                    # Phase 2: Live testing
│   ├── data.py                    
│   ├── gradcam.py                 # Grad-CAM visualisation script
│   ├── main.py                    # Training script (from scratch)
│   ├── main_fine_tune.py          # Training script (fine-tune from checkpoint)
│   ├── model.py                   # Inference wrapper with obstacle detection
│   ├── picarnet.py                # Model architecture (switchable via ARCH constant)
│   ├── prepare_collected_data.py  # Merges self-collected data with original training data
│   └── train.py                   
│
├── requirements.in
└── README.md
```


## Training

Install dependencies: `pip install -r requirements.in`

Training data is not included (Kaggle competition data). Place it at `data/` relative to the repo root — see `main.py` for the expected directory structure.

```bash
python live_model/main.py            # train from scratch
python live_model/main_fine_tune.py  # fine-tune from checkpoint
```

## Deployment

Update the model filename in `model.py` before deploying:
```python
torch.load(os.path.join(model_dir, 'your_model.pth'), map_location='cpu')
```
Then copy `model.py` and the `.pth` checkpoint to the Pi 
