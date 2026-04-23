"""
Estimate the thresholds for producing trading signals based on training data
"""

import numpy as np
import pandas as pd

import os
from os.path import *

import torch
from torch import nn

from Transform import Tensor, Inc2Price, movingaverage

your_path = '/Users/jcognon/Tail-GAN'
parent_data_path = join(your_path, 'gan_data')


def gen_thresholds(data_name, tickers, strategy, percentile_l, length, WH):
    thresholds_data_folder = join(your_path, 'Thresholds', data_name)
    os.makedirs(thresholds_data_folder, exist_ok=True)

    if 'MR' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'MR' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_Port*MR_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' not in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '_TF_%s.npy' % '_'.join(map(str, percentile_l)))
    elif 'TF' in strategy and 'Port' in strategy:
        thresholds_path = join(thresholds_data_folder, '_'.join(tickers) + '__Port*TF_%s.npy' % '_'.join(map(str, percentile_l)))
    else:
        pass

    if isfile(thresholds_path):
        thresholds_array_stocks = np.load(thresholds_path)
    else:
        data_path = join(parent_data_path, data_name)
        data_l = []
        # files = os.listdir(data_path)
        # files.sort()
        # for item in range(length):
        #     file_path = join(data_path, files[item])
        #     tmp_data = pd.read_csv(file_path)[tickers].values.T
        #     data_l.append(tmp_data)

        files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        files.sort()

        length = min(length, len(files))

        for item in range(length):
            file_path = join(data_path, files[item])
            tmp_data = pd.read_csv(file_path)[tickers].values.T
            data_l.append(tmp_data)

        data = np.stack(data_l)
        data = Tensor(data)

        prices_l = Inc2Price(data)



        prices_l = Inc2Price(data)

        thresholds_array_list = []
        for stk in range(data.shape[1]):
            if 'MR' in strategy:
                prices_stk = prices_l[:, stk, :]  # (batch, T+1)
                prices_ma_stk = torch.mean(prices_stk[:, :WH + 1], dim=1, keepdim=True)  # (batch, 1)

                zscores_MR = (prices_stk - prices_ma_stk) / 0.01
                zscores_MR = zscores_MR.cpu().detach().numpy().reshape(-1)

                thresholds_array = np.array([np.percentile(zscores_MR, i) for i in percentile_l])
                thresholds_array_list.append(thresholds_array)

            elif 'TF' in strategy:
                prices_ma_stk = movingaverage(prices_l[:, stk:stk+1, :], WH).squeeze(1)
                prices_ma2_stk = movingaverage(prices_l[:, stk:stk+1, :], WH * 2).squeeze(1)

                zscores_TF = (prices_ma_stk - prices_ma2_stk) / 0.01
                zscores_TF = zscores_TF.cpu().detach().numpy().reshape(-1)

                thresholds_array = np.array([np.percentile(zscores_TF, i) for i in percentile_l])
                thresholds_array_list.append(thresholds_array)

            else:
                pass

        thresholds_array_stocks = np.stack(thresholds_array_list)
        np.save(thresholds_path, thresholds_array_stocks)
    return thresholds_array_stocks

if __name__ == "__main__":
    data_name = "Crypto5_Binance_1m_2025Q4_step10"
    tickers = ["BTC", "ETH", "BNB", "XRP", "SOL"]

    for s in ["MR", "TF"]:
        gen_thresholds(
            data_name=data_name,
            tickers=tickers,
            strategy=s,
            percentile_l=[31, 69],
            length=3563,
            WH=10
        )

# if __name__ == "__main__":
#     import json

#     with open("gan_data/Crypto10_Binance_1h_2024_2026_step6/metadata.json") as f:
#         md = json.load(f)

#     data_name = md["data_name"]
#     tickers = md["repo_usage"]["tailgan_tickers"]

#     for s in ["MR", "TF"]:
#         gen_thresholds(
#             data_name=data_name,
#             tickers=tickers,
#             strategy=s,
#             percentile_l=[31, 69],
#             length=100,
#             WH=10
#         )