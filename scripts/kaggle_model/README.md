# ResGRU-UNet - PM2.5 forecasting over India

Five trained ResGRU-UNet networks (seeds 42, 1337, 1, 2 and 3), their 16 calibration factors and the normalization statistics. They are the best model of the study "Pipeline Over Backbone: A Controlled Study of Deep Networks for PM2.5 Forecasting over India".

- Paper and code: https://github.com/Vetri-78640/pm2.5-backbone-study-pipeline-over-backbone

## Task

Forecast PM2.5 for the next 16 hours from the last 10 hours, at every cell of a 140 x 124 grid (about 25 km; 7.06 to 38.00 N, 67.79 to 97.93 E). The data are hourly WRF-Chem simulation fields released for the ANRF AISEHack 2026, Phase 2, Theme 2 challenge (IIT Delhi) on Kaggle.

## Files

| File | Contents |
|:--|:--|
| `resgru_s42_phaseA_best.pt`, `resgru_s1337_phaseA_best.pt`, `resgru_s1_phaseA_best.pt`, `resgru_s2_phaseA_best.pt`, `resgru_s3_phaseA_best.pt` | weight-averaged (EMA) weights at the best validation epoch; keys `model`, `metrics`, `epoch`, `backbone` |
| `horizon_scales.npy` | 16 factors (float32), one per lead hour, fitted on validation for the five-seed mean |
| `stats.npz` | normalization statistics used in training |
| `ep_mask_APRIL_16.npz`, `ep_mask_JULY_16.npz`, `ep_mask_OCT_16.npz`, `ep_mask_DEC_16.npz` | episode masks and STL residual spread of the training months; the spread is an input channel |
| `study.py` | the shared pipeline and network code of the study |
| `predict.py` | builds the test windows, averages the five networks, applies the factors, writes `preds.npy` |

## Architecture

7,122,113 parameters per network.

- **Input:** (batch, 20 channels, 10 hours, 140, 124). The 10 hours are stacked as 200 channels.
- **Encoder:** residual convolution blocks (two 3 x 3 convolutions, GroupNorm, GELU, channel dropout 0.1) with 64, 128 and 256 channels, two max-pools and one average-pool to a 256-channel bottleneck at 18 x 16.
- **Lead time:** a convolutional GRU (256 channels, 3 x 3, GroupNorm on the gates) steps 16 times from the bottleneck state, one step per lead hour.
- **Decoder:** at each step, a 1 x 1 projection and a U-Net decoder (128, 64, 32 channels) with skips from the three encoder levels, then a zero-initialized output head.
- **Output:** the last observed log(1 + PM2.5) plus the predicted change for each lead hour. The forecast is max(exp(output) - 1, 0).

The architecture is ported from the original ResGRU-UNet notebook (kept in the repository under `reference/originals`), unchanged except for the 20-channel input; the port reproduces the original's output for the same weights. All weights here were trained from scratch in the study pipeline.

## Inputs (20 channels)

- **Fields (9):** PM2.5, q2, t2, u10, v10, pblh, psfc, swdown, rain
- **Physics (5):** wind speed, ventilation (PBLH x wind speed), advection, PM2.5 tendency, diurnal anomaly
- **Time (2):** hour sin and cos (estimated from swdown and t2 for test windows, which carry no timestamps)
- **Static (4):** latitude, longitude, PM2.5 climatology, STL residual spread

## Training

- **Data:** April, July, October and December 2016. Validation is the last 18% of each month after a 12-hour gap (214 windows); training uses 2,257 windows.
- **Loss:** 0.40 Huber in log space + 0.25 global SMAPE + 0.15 episode SMAPE + 0.20 (1 - episode correlation). In the Huber term episode cells weigh 6 times more; in the Huber and global SMAPE terms lead hours weigh 0.7 to 1.3.
- **Optimization:** AdamW (learning rate 2.5e-4, weight decay 1e-4), one-cycle schedule, 16 epochs, batch 4, mixed precision, gradient clipping at 1.0, EMA 0.999.
- **Cost:** about 51 minutes per seed on one NVIDIA T4. Inference takes 25.8 ms per window and 0.23 GB of peak memory at batch 4 (one network).

## Results

Score: the mean of (1 - global SMAPE / 2), (1 + episode correlation) / 2 and (1 - episode SMAPE / 2), with equal weights on validation (the server's weights are not disclosed). Test: the challenge's hidden 2017 test set (218 windows), scored by its evaluation server after the deadline, so not ranked.

| Forecast | Val score | Test (private) | Test (public) |
|:--|--:|--:|--:|
| seed 42 | 0.9158 | 0.8798 | 0.8924 |
| seed 1337 | 0.9152 | 0.8788 | 0.8930 |
| seed 1 | 0.9169 | 0.8796 | 0.8956 |
| seed 2 | 0.9178 | 0.8805 | 0.8951 |
| seed 3 | 0.9158 | 0.8810 | 0.8957 |
| five-seed mean | 0.9193 | 0.8838 | 0.8975 |
| five-seed mean + shipped factors | 0.9215 (in sample) | 0.8914 | 0.9056 |

- On validation, the five-seed mean has MAE 6.32, RMSE 18.04 and R² 0.892 (MAE and RMSE in µg/m³).
- The shipped factors are fitted on all four validation months, so their validation numbers are in sample: score 0.9215, MAE 7.13, RMSE 19.94, R² 0.868. With each month's factors fitted on the other three, as the paper reports, the score is 0.9210 (MAE 7.12, RMSE 19.71, R² 0.871).
- Calibration raises the score but worsens MAE, RMSE and R². Use the uncalibrated mean (`predict.py --no-cal`) when absolute or squared error matters.

## Usage

`predict.py` needs this model and the challenge data attached on Kaggle; Kaggle provides the data to the challenge's participants. It writes `preds.npy` with shape (218, 140, 124, 16): window, row, column, lead hour. A check run on Kaggle gave output identical to the submitted forecast that scored 0.8914 (difference 0), with and without the factors.

Without the data, the networks still load and run on any input of shape (batch, 20, 10, 140, 124) with a last frame of shape (batch, 140, 124).

## Limits

- Trained and tested on simulation fields, not station observations.
- Validation windows are the tails of the training months; the test set is from 2017.
- The hour of day is estimated for test windows.

## Licence

MIT, as the code repository. The challenge data are not included and are subject to the challenge's own terms.
