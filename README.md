# Tail-GAN Extensions for Crypto Tail-Risk Simulation

This repository builds on the original **Tail-GAN** codebase introduced in *Tail-GAN: Learning to Simulate Tail Risk Scenarios*. The original methodology was developed by **Rama Cont, Mihai Cucuringu, Renyuan Xu, and Chao Zhang**.

This version extends the original project with:
- a crypto-data construction pipeline based on Binance market data,
- an out-of-sample evaluation script,
- new benchmark trading strategies,
- support for a correlation-regularized Tail-GAN variant (`TailGAN_Corr`),
- and plotting utilities to compare several trained models jointly.

The goal of this repository is therefore twofold:
1. reproduce the main Tail-GAN framework,
2. extend it to a new experimental setting focused on crypto-assets and enriched benchmark constraints.

## Acknowledgment

This project is based on the original Tail-GAN framework proposed by:
- **Rama Cont**
- **Mihai Cucuringu**
- **Renyuan Xu**
- **Chao Zhang**

All credit for the original Tail-GAN methodology goes to the original authors.

## Project Structure

```text
TailGAN/
├── data/                           # raw and processed datasets
├── gen_synthetic.py                # synthetic return paths generation
├── gen_static_port.py              # static portfolio matrices generation
├── gen_thresholds.py               # thresholds for benchmark strategies
├── build_tailgan_crypto_dataset.py # builds crypto datasets from Binance data
├── Dataset.py                      # PyTorch dataset helpers
├── Transform.py                    # price/return to PnL utilities
├── util.py                         # utility functions
├── TailGAN.py                      # Tail-GAN training script
├── WGAN.py                         # Wasserstein-GAN baseline
├── GOM.py                          # Generative-Only Model baseline
├── NewStrategies.py                # additional benchmark trading strategies
├── Evaluation.py                   # standard evaluation
├── Evaluation_OOS.py               # out-of-sample evaluation
├── Rejection_rate.py               # coverage and score tests
├── EigenPort.py                    # eigen-portfolio construction
├── Plot_Training.py                # training curves for one model
├── Plot_multiple_trainings.py      # compares training curves across models
├── Plot_Quantile_PnL.py            # quantile / VaR / PnL plots for one model
├── Plot_quantile_pnl_multiple.py   # compares quantile / PnL plots across models
├── Plot_Corr_Auto.py               # correlation / autocorrelation diagnostics
└── README.md
'''
## Main Additions Compared to the Original Repository

### `build_tailgan_crypto_dataset.py`
Builds Tail-GAN-compatible datasets from Binance market data.

This script allows the creation of crypto datasets with configurable:
- asset universe,
- sampling frequency,
- time period,
- rolling-window structure,
- preprocessing choices.

It makes it possible to move from the original market setting to a crypto-asset setting while preserving the data format required by the Tail-GAN pipeline.

### `Evaluation_OOS.py`
Performs out-of-sample evaluation of trained models.

This script was added to assess how well the generated scenarios generalize beyond the training period, which is particularly important in the crypto setting.

### `NewStrategies.py`
Implements additional benchmark trading strategies beyond the original ones.

These new strategies enrich the set of portfolio constraints used during training and provide a broader view of tail-risk quality.

### `Plot_multiple_trainings.py`
Plots and compares training curves from several models on the same figure.

This is useful to visually compare convergence behavior across variants such as:
- TailGAN,
- TailGAN_Static,
- TailGAN_Corr,
- versions with new strategies.

### `Plot_quantile_pnl_multiple.py`
Plots and compares quantile/PnL diagnostics for several models jointly.

This makes model comparison more direct and helps assess which version best reproduces tail behavior.

## Additional Code Modifications

### Integration of the new benchmark strategies
The newly implemented strategies were propagated through the project pipeline.

In practice, these strategy extensions were integrated into the relevant generation, training, evaluation, and plotting scripts so that they can be used in the same way as the original benchmark strategies.

### Modification of `TailGAN.py` for `TailGAN_Corr`
The main Tail-GAN training script was modified to support a correlation-regularized variant, referred to as `TailGAN_Corr`.

This extension adds an additional mechanism aimed at improving the reproduction of dependence structures across assets.

## Usage

A standard workflow is the following:

1. Build or prepare the dataset
   - For synthetic experiments: run `gen_synthetic.py`
   - For crypto experiments: run `build_tailgan_crypto_dataset.py`

2. Prepare benchmark strategy inputs
   - Run `gen_static_port.py`
   - Run `gen_thresholds.py`

3. Train the model
   - Run `TailGAN.py` for the standard Tail-GAN model
   - Optionally use the modified version supporting `TailGAN_Corr`

4. Evaluate the model
   - Run `Evaluation.py` for standard evaluation
   - Run `Evaluation_OOS.py` for out-of-sample evaluation

5. Plot and compare results
   - Use `Plot_Training.py` for a single training run
   - Use `Plot_multiple_trainings.py` to compare several training runs
   - Use `Plot_Quantile_PnL.py` for a single model
   - Use `Plot_quantile_pnl_multiple.py` to compare several models
   - Use `Plot_Corr_Auto.py` for dependence diagnostics

## Data

This repository can be used with:
- synthetic data generated internally by the project,
- crypto datasets constructed from Binance market data through `build_tailgan_crypto_dataset.py`.

The original paper also includes an application based on LOBSTER data. LOBSTER data is not distributed in this repository and must be obtained separately by users if needed.


## Citation

If you use the original Tail-GAN methodology, please cite the original paper:

```bibtex
@article{cont2025tail,
  title={Tail-gan: Learning to simulate tail risk scenarios},
  author={Cont, Rama and Cucuringu, Mihai and Xu, Renyuan and Zhang, Chao},
  journal={Management Science},
  year={2025},
  publisher={INFORMS}
}
