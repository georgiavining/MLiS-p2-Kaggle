# PiCar — Autonomous RC Car

Behavioural cloning pipeline for a SunFounder PiCar-V (Raspberry Pi 4) trained to drive autonomously around a track.

## Project Phases

### Phase 1 — Kaggle Challenge
The first phase was a Kaggle competition to predict steering angle and speed from dashcam images. The pipeline used an EfficientNet-B0/MobileNetV3 ensemble with test-time augmentation, trained on ~14k labelled images.

**Best private leaderboard MSE: 0.01342 (3rd/6)**

### Phase 2 — Live Testing
The second phase was live on-track testing across 12 scenarios: lane keeping, stopping for pedestrians/objects, and responding to traffic signs (on t-junction, oval and figure-8 tracks). The live model was rebuilt from the Kaggle model with several key changes:

- **Angle-only output** — speed is fixed at a default value; stopping is handled by a rule-based layer rather than the neural network
- **Rule-based obstacle detection** — HSV-based floor segmentation in a centre-lane ROI triggers a speed=0 override when an object is detected in the path
- **Improved training** — corner frame oversampling, horizontal flip augmentation with label inversion, and crop tuning guided by Grad-CAM
- **Self-collected data** — additional ~3k images across oval and figure-8 scenarios, with per-source bottom cropping to handle a different camera position


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

### Setup
```bash
pip install -r requirements.in
```

### Data
Not included (Kaggle competition data). Expected at `data/` relative to repo root

## Deployment

In 'model.py', update the model filename in `Model.__init__` to match your checkpoint:
```python
torch.load(os.path.join(model_dir, 'your_model.pth'), map_location='cpu')
```

Copy `model.py` and the `.pth` checkpoint to the Pi

