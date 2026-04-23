"""
Plot the quantiles of PnLs of benchmark strategies for multiple versions.
"""

import argparse
import os
import random
import sys
from os.path import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

# Prevent TailGAN.py from consuming this script's CLI arguments
argv_backup = sys.argv.copy()
sys.argv = [sys.argv[0]]
from TailGAN import *
sys.argv = argv_backup

from Dataset import Dataset_IS
from NewStrategies import Momentum, Breakout, VolTarget


your_path = '/Users/jcognon/Tail-GAN'
plot_path = join(your_path, 'Plots')
gen_data_path = join(your_path, 'Gens')
sample_number = 1000
result_data_path = join(your_path, f'Results_S{sample_number}')

DISPLAY_NAMES = {
    "Crypto10_TailGAN_base_1500": "TailGAN",
    "Crypto10_TailGAN_base_1500_corr1": "TailGAN_Corr",
    "Crypto10_TailGAN_newstrats_1500": "TailGAN_newstrats",
    "Crypto10_TailGAN_Static": "TailGAN_Static",
}

STOCK_NAMES = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'LTC', 'TRX', 'LINK']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help="Versions to compare"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="Crypto10_quantile_pnl_compare.pdf",
        help="Output PDF filename"
    )
    return parser.parse_args()


def override_opt_for_crypto10(opt):
    opt.n_epochs = 1500
    opt.len = 2326
    opt.data_name = 'Crypto10_Binance_1h_2024_2026_step6'
    opt.tickers = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'LTC', 'TRX', 'LINK']
    opt.strategies = ['Port', 'MR', 'TF', 'MOM', 'BO', 'VT']
    opt.n_trans = 50
    opt.Cap = 10
    opt.WH = 10
    opt.ratios = [1.0, 1.0]
    opt.thresholds_pct = [[31, 69]]
    opt.static_way = 'LShort'
    return opt


def compute_pnl_spc(R, opt):
    R_tensor = Tensor(R)

    prices_l = Inc2Price(R_tensor)
    port_prices_l = StaticPort(prices_l, opt.n_trans, opt.static_way, insample=True)

    pnl_bh = BuyHold(prices_l, opt.Cap)
    pnl_l = [pnl_bh]

    for strategy in opt.strategies:
        if strategy == 'Port':
            pnl_bh_port = BuyHold(port_prices_l, opt.Cap)
            pnl_l.append(pnl_bh_port)

        elif strategy == 'MR':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name, opt.tickers, strategy, percentile_l, opt.len, opt.WH
                )
                pnl_mr = MeanRev(
                    prices_l, opt.Cap, opt.WH,
                    LR=opt.ratios[0], SR=opt.ratios[1],
                    ST=thresholds_array[:, -1], LT=thresholds_array[:, -2]
                )
                pnl_l.append(pnl_mr)

        elif strategy == 'TF':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name, opt.tickers, strategy, percentile_l, opt.len, opt.WH
                )
                pnl_tf = TrendFollow(
                    prices_l, opt.Cap, opt.WH,
                    LR=opt.ratios[0], SR=opt.ratios[1],
                    ST=thresholds_array[:, 0], LT=thresholds_array[:, 1]
                )
                pnl_l.append(pnl_tf)

        elif strategy == 'MOM':
            pnl_mom = Momentum(
                prices_l, opt.Cap, opt.WH,
                LR=opt.ratios[0], SR=opt.ratios[1],
                upper_pct=opt.thresholds_pct[0][1],
                lower_pct=opt.thresholds_pct[0][0],
            )
            pnl_l.append(pnl_mom)

        elif strategy == 'BO':
            pnl_bo = Breakout(
                prices_l, opt.Cap, opt.WH,
                LR=opt.ratios[0], SR=opt.ratios[1],
            )
            pnl_l.append(pnl_bo)

        elif strategy == 'VT':
            pnl_vt = VolTarget(
                prices_l, opt.Cap, opt.WH,
                target_vol=0.10,
            )
            pnl_l.append(pnl_vt)

    pnl = torch.cat(pnl_l, dim=1)
    return pnl.cpu().numpy()


def load_real_data(opt):
    dataset = Dataset_IS(
        tickers=opt.tickers,
        data_path=join(your_path, "gan_data", opt.data_name),
        length=opt.len
    )
    real_r = np.array([d.detach().numpy() for d in dataset.samples])
    n_sample = min(10000, real_r.shape[0])
    sample_idx = random.sample(range(real_r.shape[0]), n_sample)
    sample_real = real_r[sample_idx, :, :]
    return sample_real

def load_fake_data_for_version(version, opt):
    candidate_dirs = sorted([
        join(gen_data_path, d)
        for d in os.listdir(gen_data_path)
        if isdir(join(gen_data_path, d)) and d.startswith(f'gen_data_{version}')
    ])

    print("version =", version)
    print("candidate_dirs =", candidate_dirs)

    if len(candidate_dirs) == 0:
        print(f"[ERROR] No folder found for version {version}")
        return None

    fake_l = []

    for folder in candidate_dirs:
        n_found_in_folder = 0

        for epoch in range(1, opt.n_epochs + 1):
            fake_file = join(folder, f'Fake_id0_E{epoch}.npy')
            if isfile(fake_file):
                tmp_fake = np.load(fake_file)
                fake_l.append(tmp_fake)
                n_found_in_folder += 1

        print(f"[INFO] {basename(folder)}: n_fake_files_found = {n_found_in_folder}")

    if len(fake_l) == 0:
        print(f"[ERROR] No fake files found for version {version}")
        return None

    fake_r = np.concatenate(fake_l, axis=0)
    n_sample_fake = min(10000, fake_r.shape[0])
    sample_idx = random.sample(range(fake_r.shape[0]), n_sample_fake)
    sample_fake_r = fake_r[sample_idx, :, :]
    return sample_fake_r



def load_data_multiple(versions, opt):
    R_dic = {'Market Data': load_real_data(opt)}

    for version in versions:
        fake_data = load_fake_data_for_version(version, opt)
        if fake_data is not None:
            display_name = DISPLAY_NAMES.get(version, version)
            R_dic[display_name] = fake_data

    return R_dic


def split_pnl_blocks(PNL, opt):
    num_stocks = len(opt.tickers)
    blocks = {}
    idx = 0

    blocks['Stk'] = PNL[:, idx:idx + num_stocks]
    idx += num_stocks

    for strategy in opt.strategies:
        if strategy == 'Port':
            blocks['Port'] = PNL[:, idx:idx + opt.n_trans]
            idx += opt.n_trans

        elif strategy in ['MR', 'TF']:
            width = num_stocks * len(opt.thresholds_pct)
            blocks[strategy] = PNL[:, idx:idx + width]
            idx += width

        elif strategy in ['MOM', 'BO', 'VT']:
            blocks[strategy] = PNL[:, idx:idx + num_stocks]
            idx += num_stocks

    return blocks


def compute_var_tables(PNL_dic, opt, alpha=0.05):
    strategy_keys = ['Stk', 'Port', 'MR', 'TF', 'MOM', 'BO', 'VT']

    for strat in strategy_keys:
        rows = {}

        for k in PNL_dic.keys():
            PNL = PNL_dic[k]
            blocks = split_pnl_blocks(PNL, opt)

            if strat not in blocks:
                continue

            block = blocks[strat]
            vals = []

            for i in range(block.shape[1]):
                data = np.sort(block[:, i])
                size = block.shape[0]
                idx = min(int(alpha * size), size - 1)
                vals.append(np.round(data[idx], 3))

            rows[k] = vals

        if len(rows) == 0:
            continue

        df = pd.DataFrame(rows)

        if strat == 'Port':
            df.index = [f'Trans-{i+1}' for i in range(df.shape[0])]
        else:
            df.index = opt.tickers[:df.shape[0]]

        df.to_csv(join(plot_path, f'{strat}_VaR_Synthetic_multiple.csv'))


def plot_rank(PNL_dic, opt, output_name):
    strategy_keys = ['Stk', 'MR', 'TF', 'MOM', 'BO', 'VT']
    num_stocks = len(opt.tickers)
    n_cols = len(strategy_keys)

    pdf_name = join(plot_path, output_name)

    col_titles = {
        'Stk': 'Static buy-and-hold',
        'MR': 'Mean-reversion',
        'TF': 'Trend-following',
        'MOM': 'Momentum',
        'BO': 'Breakout',
        'VT': 'Vol-targeting',
    }

    with PdfPages(pdf_name) as pdf:
        fig, axes = plt.subplots(num_stocks, n_cols, figsize=(5 * n_cols, 4 * num_stocks), sharex=True)

        if n_cols == 1:
            axes = np.array(axes).reshape(num_stocks, 1)

        for j, strat in enumerate(strategy_keys):
            axes[0, j].set_title(col_titles[strat], fontsize=16)

        for i, row in enumerate(STOCK_NAMES[:num_stocks]):
            axes[i, 0].set_ylabel(row + '\n' + r'$\alpha$-quantile (log scale)', rotation=90, fontsize=16)

        for j in range(n_cols):
            axes[-1, j].set_xlabel(r'$\alpha$ (log scale)', fontsize=16)

        for i in range(num_stocks):
            for model_name, PNL in PNL_dic.items():
                blocks = split_pnl_blocks(PNL, opt)

                for j, strat in enumerate(strategy_keys):
                    if strat not in blocks:
                        continue

                    block = blocks[strat]
                    if i >= block.shape[1]:
                        continue

                    data = block[:, i]
                    size = len(data)
                    x = np.cumsum(np.ones(size)) / size

                    axes[i, j].grid(True)
                    axes[i, j].plot(x, np.sort(data), linewidth=2.5, label=model_name)
                    axes[i, j].set_yscale('symlog')
                    axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    axes[i, j].set_xscale('log')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=16)
        pdf.savefig()
        plt.close()


if __name__ == '__main__':
    args = parse_args()

    os.makedirs(plot_path, exist_ok=True)

    opt = override_opt_for_crypto10(opt)

    print(opt)
    print("Versions compared:")
    for v in args.versions:
        print(" -", DISPLAY_NAMES.get(v, v))

    R_dic = load_data_multiple(args.versions, opt)

    PNL_dic = {}
    for k in R_dic:
        PNL_dic[k] = compute_pnl_spc(R_dic[k], opt)

    compute_var_tables(PNL_dic, opt, alpha=0.05)
    plot_rank(PNL_dic, opt, args.output_name)
