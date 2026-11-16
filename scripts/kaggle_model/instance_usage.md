# Model format

PyTorch checkpoints (torch.save dicts) and NumPy arrays. Load a checkpoint with torch.load(path, weights_only=False)['model'] into ResGRUUNet(20) from study.py.

# Training data

Hourly WRF-Chem fields over India for April, July, October and December 2016, from the ANRF AISEHack 2026 Phase 2 Theme 2 challenge data.

# Model inputs

(batch, 20, 10, 140, 124) and the last observed log(1 + PM2.5) frame (batch, 140, 124). README.md lists the 20 channels.

# Model outputs

(batch, 16, 140, 124) in log(1 + PM2.5); the forecast is max(exp(output) - 1, 0). predict.py writes (218, 140, 124, 16).

# Model usage

Attach this model and the challenge data (Kaggle provides it to the challenge's participants), then run predict.py. It averages the five networks and multiplies lead hour h by horizon_scales[h]; --no-cal skips the factors. Without the data, the networks still load and run on any input of the shapes above.

# Fine-tuning

Initialize ResGRUUNet(20) from any *_best.pt and continue training with study.py.
