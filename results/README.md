# Results

Run records downloaded from the Kaggle notebooks in [`notebooks/kaggle/`](../notebooks/kaggle/), and the tables built from them. Validation means the same 214 windows for every model; test means the held-out Kaggle set, which reports only the benchmark score. Each Kaggle run's full log is `notebooks/kaggle/<stage>/run_log.txt`.

## Files

| Path | Contents |
|:--|:--|
| `resgru/`, `fmgwaclu/`, `convlstm/`, `patchafno/` | round one, seed 42: `summary.json` (best epoch, scores, timing) and `train_log.csv` (one row per epoch) |
| `final/` | final round: ResGRU-UNet seed 1337, seed ensembles and calibration |
| `seeds_resgru/`, `seeds_afnov4/` | seeds 1 to 3 of each finalist |
| `analysis/analysis.json` | five-seed statistics, block bootstrap, breakdowns by lead hour, month and concentration |
| `ablations_a/`, `ablations_b/` | the four single-change variants (H1 to H4) |
| `regression/regression_metrics.json` | MAE, RMSE, R² and benchmark score for every model on validation |
| `baselines/baselines.json` | persistence and seasonal naive on validation |
| `ridge/ridge.json` | ridge regression: penalty search, weights, validation metrics |
| `hybrid/hybrid.json` | the gradient-boosted correction: out-of-fold scores per month, the rule and its verdict |
| `maps/maps.json`, `maps/maps.npz` | per-cell error maps (five-seed means), as 140 x 124 arrays |
| `test_scores.csv` | every test score of the study, with the submission description |
| `earlier_iterations.csv` | test score at each development iteration |

`maps.npz` holds per-cell averages over the validation windows, including the mean PM2.5 per cell. It is a summary, not the competition data.

## Round one

Seed 42, same pipeline, no ensemble or calibration.

| Network | Parameters | Validation | Test | Against AFNO V4 |
|:--|--:|--:|--:|:--|
| ResGRU-UNet | 7.12M | 0.9158 | 0.8798 | better |
| AFNO V4 | 6.98M | 0.9110 | 0.8762 | reference |
| FMG-WACLU | 3.45M | 0.9099 | 0.8710 | tie |
| patch AFNO | 5.71M | 0.9031 | 0.8530 | worse |
| ConvLSTM | 19.0M | 0.8945 | 0.8452 | worse |

A network is better than AFNO V4 only if it leads by more than 0.003 on validation and also scores higher on test; a smaller validation gap is a tie. The rule was fixed before training.

## Two and five seeds

With two seeds, ensembles and calibration, ResGRU-UNet leads 0.9175 against 0.9131 on validation and 0.8898 against 0.8833 on test.

| | AFNO V4 | ResGRU-UNet | Gap |
|:--|--:|--:|--:|
| Single seeds, validation, mean | 0.91146 | 0.91630 | +0.00484 |
| standard deviation | 0.00059 | 0.00104 | |
| Five-seed mean, validation | 0.91454 | 0.91933 | +0.00479 |
| + calibration (one month out) | 0.91492 | 0.92097 | +0.00605 |
| Single seeds, test, mean | 0.87540 | 0.87994 | +0.00454 |
| standard deviation | 0.00130 | 0.00085 | |
| Five-seed mean, test | 0.8793 | 0.8838 | +0.0045 |
| + calibration, test | 0.8848 | 0.8914 | +0.0066 |

The 95% block-bootstrap interval of the five-seed validation gap is [+0.00383, +0.00578] (blocks of 13 consecutive windows within each month, 2,000 resamples). Every ResGRU-UNet seed beats every AFNO V4 seed on validation and on test (25 of 25 pairs).

## Regression metrics on validation

MAE and RMSE in µg/m³, over all 5.9 x 10⁷ values. Skill is the RMSE reduction against persistence. Test labels are hidden, so these metrics exist on validation only.

| Model | MAE | RMSE | R² | Skill | Score |
|:--|--:|--:|--:|--:|--:|
| Persistence | 10.44 | 26.46 | 0.767 | 0.0% | 0.8515 |
| Seasonal naive (24 h) | 11.53 | 26.80 | 0.761 | -1.3% | 0.7907 |
| Ridge regression | 8.51 | 21.48 | 0.847 | 18.8% | 0.8637 |
| patch AFNO, seed 42 | 6.85 | 19.02 | 0.880 | 28.1% | 0.9031 |
| ConvLSTM, seed 42 | 7.05 | 18.93 | 0.881 | 28.5% | 0.8945 |
| FMG-WACLU, seed 42 | 7.14 | 19.32 | 0.876 | 27.0% | 0.9099 |
| AFNO V4, seed 42 | 7.31 | 20.17 | 0.865 | 23.8% | 0.9110 |
| ResGRU-UNet, seed 42 | 6.57 | 18.57 | 0.885 | 29.8% | 0.9158 |
| AFNO V4, five-seed mean | 6.99 | 19.45 | 0.874 | 26.5% | 0.9145 |
| + calibration | 7.76 | 20.90 | 0.855 | 21.0% | 0.9149 |
| ResGRU-UNet, five-seed mean | 6.32 | 18.04 | 0.892 | 31.8% | 0.9193 |
| + calibration | 7.12 | 19.71 | 0.871 | 25.5% | 0.9210 |
| + LightGBM correction (hybrid, out of fold) | 6.42 | 17.75 | 0.895 | 32.9% | 0.9051 |

Calibration raises the benchmark score but worsens MAE, RMSE and R². On the test set, ridge regression scores 0.7951 and persistence 0.7910.

## Hybrid

A LightGBM model predicts the error of the uncalibrated ResGRU-UNet five-seed ensemble in log(1 + y). The rule, fixed before the run: fit on the validation windows of three months, predict the fourth, and call the hybrid better only if its pooled score is higher and it wins in at least three of four months.

| Held-out month | Ensemble | Hybrid | Change |
|:--|--:|--:|--:|
| April | 0.9162 | 0.8999 | -0.0162 |
| July | 0.9091 | 0.8940 | -0.0151 |
| October | 0.9271 | 0.9203 | -0.0068 |
| December | 0.9249 | 0.9061 | -0.0188 |
| Pooled | 0.9193 | 0.9051 | -0.0142 |

Verdict: no gain. RMSE and R² improve slightly, but episode SMAPE rises from 0.235 to 0.305. The hybrid was not submitted to the test set.

## Ablations

One change to one finalist, seed 42. d is the validation change from the family's five-seed mean, d/G its share of the 0.00484 gap. A change counts only if it moves validation by more than max(2 sd, 0.002).

| Hypothesis | Change | d | d/G | Test |
|:--|:--|--:|--:|--:|
| H1 | AFNO V4, skips fuse all 10 hours | +0.00113 | +0.23 | 0.8748 |
| H2 | AFNO V4, zero-padded FFT | -0.00084 | -0.17 | 0.8745 |
| H3 | AFNO V4, no soft threshold | -0.00052 | -0.11 | 0.8758 |
| H4 | ResGRU-UNet, ConvGRU replaced | -0.00174 | -0.36 | 0.8798 |

None passes the threshold at one seed.

## Pipeline against backbone

The same network in its own earlier pipeline and in the V4 pipeline (test, seed 42, no ensemble or calibration).

| Network | Own pipeline | V4 pipeline | Gain |
|:--|--:|--:|--:|
| ResGRU-UNet | 0.8506 | 0.8798 | +0.0292 |
| FMG-WACLU | 0.8465 | 0.8710 | +0.0245 |
| patch AFNO | 0.8218 | 0.8530 | +0.0312 |

At the same setting the gap between ResGRU-UNet and AFNO V4 is +0.0036. AFNO V4's own test gain over V3 (+0.0224) splits into +0.0147 from the per-seed changes, +0.0024 from the two-seed ensemble and +0.0054 from calibration.

## Development iterations

Test score at each iteration ([`earlier_iterations.csv`](earlier_iterations.csv)). Iterations 1 to 4 were chosen on a random validation split, which leaked; iteration 5 introduced the temporal hold-out.

| Iteration | Model | Test |
|:--|:--|--:|
| 1 | patch AFNO, random split | 0.8218 |
| 2 | ResGRU-UNet | 0.8506 |
| 3 | ResGRU-UNet, fine-tuned on episode peaks | 0.8594 |
| 4 | ResGRU-UNet, quantile blends | 0.8648 |
| 5 | AFNO with a SimVP translator, temporal hold-out | 0.8561 |
| 6 | AFNO V3 | 0.8609 |
| 7 | AFNO V4 | 0.8833 |
| 8 | ResGRU-UNet in the V4 pipeline, five seeds, calibration | 0.8914 |
