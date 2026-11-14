# Pipeline before backbone

**16-hour PM2.5 forecasting over India with Fourier neural operators and convolutional-recurrent networks**

Vetriselvan R

Paper: [`paper/paper.pdf`](paper/paper.pdf)

## Abstract

We forecast PM2.5 over all of India 16 hours ahead, on a 140 x 124 grid of about 25 km cells, from 10 hours of WRF-Chem simulation fields, and ask how much of a deep forecaster's skill comes from its network and how much from the pipeline around it.

Over seven development iterations we built a physics-informed pipeline:
- ventilation, advection and tendency channels;
- a log-space persistence residual;
- a leak-free temporal validation;
- a loss matched to the evaluation metric;
- seed ensembles with per-hour calibration.

Five networks were then trained inside this one pipeline, with decision rules fixed before each stage and 1, 2 and 5 seeds. A U-Net with a convolutional GRU over lead time (ResGRU-UNet) beats an adaptive Fourier neural operator hybrid (AFNO V4) on all 25 seed pairs and at every lead hour.

Moving the same network into the improved pipeline gained 6 to 8 times more than swapping the network. Four single-change ablations do not explain the gap between the two networks. A gradient-boosted correction of the ensemble lowers RMSE but also the benchmark score, so it is reported and not adopted.

## Results

Validation, 214 windows, MAE and RMSE in µg/m³. The score is the benchmark's composite of global SMAPE, episode SMAPE and episode correlation (higher is better).

| Model | MAE | RMSE | R² | Score |
|:--|--:|--:|--:|--:|
| Persistence | 10.44 | 26.46 | 0.767 | 0.8515 |
| Seasonal naive (24 h) | 11.53 | 26.80 | 0.761 | 0.7907 |
| Ridge regression | 8.51 | 21.48 | 0.847 | 0.8637 |
| AFNO V4, five-seed mean | 6.99 | 19.45 | 0.874 | 0.9145 |
| **ResGRU-UNet, five-seed mean** | **6.32** | **18.04** | **0.892** | **0.9193** |
| ResGRU-UNet + LightGBM correction (out of fold) | 6.42 | 17.75 | 0.895 | 0.9051 |

- **Lower error:** ResGRU-UNet has 31.8% lower RMSE than persistence.
- **Held-out test:** the test set reports only the benchmark score. With five seeds and calibration, ResGRU-UNet scores 0.8914 against 0.8848 for AFNO V4.
- **Seeds:** every ResGRU-UNet seed beats every AFNO V4 seed on validation and on test. The five-seed validation gap is +0.00479, with a 95% block-bootstrap interval of [+0.00383, +0.00578].
- **Pipeline against network:** moving ResGRU-UNet into the improved pipeline raised its test score by +0.0292. At the same setting, swapping AFNO V4 for ResGRU-UNet gains +0.0036.

All tables: [`results/README.md`](results/README.md).

![Time-averaged validation maps: mean PM2.5, episode frequency, MAE of persistence and of ResGRU-UNet, the MAE difference to AFNO V4, and R² over time](paper/figures/fig_maps.png)

*Time-averaged validation maps. (a) Mean PM2.5. (b) Share of hours that are episodes. (c, d) MAE of persistence and of the ResGRU-UNet five-seed mean. (e) MAE difference, ResGRU-UNet minus AFNO V4 (blue: ResGRU-UNet lower). (f) R² of each cell's time series, ResGRU-UNet.*

## Method

**Data.** WRF-Chem fields released for the ANRF AISEHack 2026, Phase 2, Theme 2 challenge (IIT Delhi) on Kaggle:
- grid: 140 x 124 cells, about 25 km each, over India;
- period: April, July, October and December 2016, hourly;
- each window: 10 input hours to 16 forecast hours;
- test: 218 windows with hidden targets.

The data are simulations, not station measurements.

**Pipeline**, the same for every network:
- **Inputs:** 20 channels per cell and hour: 9 fields, 4 physics channels (wind speed, ventilation, advection, tendency), the daily cycle and 4 static maps.
- **Output:** every network predicts the change of log(1 + PM2.5) from the last input hour, with a zero-initialized last layer, so an untrained network equals persistence.
- **Split:** validation is the last 18% of each month after a 12-hour gap, because windows one hour apart share 96% of their hours and a random split leaks.
- **Loss:** combines a log-space Huber term with the three terms of the benchmark score, weighted toward later lead hours.

**Networks:**
- AFNO V4: a U-Net encoder with adaptive Fourier neural operator blocks;
- ResGRU-UNet: a U-Net with a ConvGRU over lead time;
- FMG-WACLU, ConvLSTM and patch AFNO.

Each is ported from its original notebook, and the port is checked to give exactly the original's output for the same weights.

**Study:**
- **Rule fixed before training:** a network is better than AFNO V4 only if it leads by more than 0.003 on validation and also scores higher on test.
- **Three stages:** five networks on one seed; the two finalists with two seeds and the full recipe; then five seeds per finalist with a block bootstrap.
- **Ablations:** four single-change ablations, with predictions written before the runs.
- **Hybrid:** a LightGBM correction of the ensemble, tested by leaving one month out.

## Repository layout

```
src/                 shared pipeline and all networks (study.py), stage sources, local loader
scripts/             builders for each Kaggle notebook, the ablation verdict, Kaggle tools
tests/               config test, port check (local_check.py), dry run of every stage
notebooks/           six local notebooks with outputs
notebooks/kaggle/    the Kaggle notebooks exactly as they ran, each with its run log
configs/             study settings (study.yaml), checked against the code by a test
results/             run records from Kaggle and the result tables
reference/           the AFNO V4 training script and the original notebooks of the ported networks
paper/               paper.tex, paper.pdf and figures
data/                download steps (data/README.md); no data included
```

## Reproduce

**Install**

```bash
git clone https://github.com/Vetri-78640/pm2.5-backbone-study-pipeline-over-backbone.git
cd pm2.5-backbone-study-pipeline-over-backbone
pip install -e ".[dev]"            # or: pip install -r requirements.txt
```

**Data.** Accept the competition rules on Kaggle, then follow [`data/README.md`](data/README.md). The data are not part of this repository.

**Tests**

```bash
pytest                                          # config against the code, seconds
pytest -m dryrun                                # every stage on synthetic data and the port check, CPU, about 35 min
python tests/local_check.py reference/originals # port check only
```

**Notebooks.** Each local notebook imports the code exactly as it ran on Kaggle (through `src/local.py`) and retrains nothing.
- Notebooks 01, 02, 03 and 05 need the downloaded data; 04 and 06 read only `results/`.
- All run on a laptop CPU; `03_models` takes about 3.5 minutes.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_and_eda.ipynb
```

| Notebook | Contents |
|:--|:--|
| `01_data_and_eda` | files, grid, PM2.5 in space and time, predictability, transport |
| `02_preprocessing` | physics channels, normalization, episode masks, leak-free split, hour estimation |
| `03_models` | the networks, a forward pass on a real window, the port check |
| `04_training` | recipe, loss, regularization, training curves |
| `05_evaluation` | score, baselines recomputed from raw data, MAE / RMSE / R², calibration |
| `06_analysis_and_ablations` | decision rules, the three stages, maps, ablations |

**Kaggle stages.** Each stage in `notebooks/kaggle/` is built from `src/` by a script in `scripts/` and ran on Kaggle; the training stages used one T4 GPU. To run a stage again:

```bash
uvx --python 3.12 --from kaggle==2.2.5 --with six kaggle kernels push -p notebooks/kaggle/<stage> --accelerator NvidiaTeslaT4
```

| Stage | Builder |
|:--|:--|
| round one: `resgru`, `fmgwaclu`, `convlstm`, `patchafno` | `build_notebooks.py` |
| final round: `final` | `build_final.py` |
| seeds 1 to 3: `seeds_resgru`, `seeds_afnov4` | `build_seeds.py` |
| five-seed analysis: `analysis` | `build_analysis.py` |
| ablations: `ablations_a`, `ablations_b` | `build_ablations.py` |
| regression metrics, baselines, ridge, maps, hybrid | `build_regression.py`, `build_baselines.py`, `build_ridge.py`, `build_maps.py`, `build_hybrid.py` |
| persistence on the test set: `persistence` | none |

Notes on the Kaggle stages:
- **Builders overwrite.** The builders write into `notebooks/kaggle/` and would overwrite the notebooks that ran, so run them in a copy.
- **Rebuilt notebooks.** They differ from the committed ones in two ways. The round-one notebooks hard-coded seed 42, where `study.py` now reads a top-level `SEED` (three lines). And two comment lines in `src/` were reworded after the runs.
- **Inputs.** Each notebook expects the competition data and the AFNO V4 run output (`imvetri0/afno-v4`) attached on Kaggle.

## Citation

```bibtex
@misc{vetriselvan2026pipeline,
  author       = {{Vetriselvan R}},
  title        = {Pipeline before backbone: 16-hour {PM2.5} forecasting over {India} with {Fourier} neural operators and convolutional-recurrent networks},
  year         = {2026},
  howpublished = {\url{https://github.com/Vetri-78640/pm2.5-backbone-study-pipeline-over-backbone}}
}
```

## Licence

The code, results and paper in this repository are released under the [MIT licence](LICENSE).

The competition data are not included and are not covered by this licence; they are subject to the challenge's own terms on Kaggle.
