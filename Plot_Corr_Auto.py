"""
Compute and plot correlation and autocorrelation statistics for the generated data and ground truth.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("ggplot")

import numpy as np
import pandas as pd
import os
from os.path import join, isfile
import random
import seaborn as sns
from statsmodels.tsa.stattools import acf

from TailGAN import opt, this_version
from Dataset import Dataset_IS

sample_number = 1000

your_path = '/Users/jcognon/Tail-GAN'
plot_path = join(your_path, 'Plots')
#gen_data_path = join(your_path, 'Gens')
# gen_data_path = join(your_path, 'Gens', 'gen_data_{this_version}')
gen_data_path = join(your_path, 'Gens', f'gen_data_{this_version}')
result_data_path = join(your_path, f'Results_S{sample_number}')

fontsize = 24
nlags = 10

os.makedirs(plot_path, exist_ok=True)
os.makedirs(result_data_path, exist_ok=True)
os.makedirs(join(result_data_path, this_version), exist_ok=True)


def Load_Data(opt, sample_number, model_index=1):
    dataset = Dataset_IS(
        tickers=opt.tickers,
        data_path=join(your_path, "gan_data", opt.data_name),
        length=opt.len
    )
    real = np.array([d.detach().numpy() for d in dataset.samples])

    if not os.path.isdir(gen_data_path):
        raise FileNotFoundError(f"Dossier introuvable : {gen_data_path}")

    files = os.listdir(gen_data_path)

    model_files = []
    for f in files:
        if f.startswith(f'Fake_id{model_index}_E') and f.endswith('.npy'):
            try:
                epoch = int(f.split('_E')[1][:-4])
                model_files.append((epoch, f))
            except Exception:
                pass

    if len(model_files) == 0:
        raise FileNotFoundError(
            f"Aucun fichier Fake_id{model_index}_E*.npy trouvé dans {gen_data_path}"
        )

    model_files.sort(key=lambda x: x[0])

    epoches_l = []
    fake_l = []

    for epoch, fname in model_files:
        tmp_fake = np.load(join(gen_data_path, fname))

        n_available = tmp_fake.shape[0]
        n_take = min(sample_number, n_available)

        if n_take < n_available:
            sample_idx = random.sample(range(n_available), n_take)
            tmp_fake = tmp_fake[sample_idx, :]

        fake_l.append(tmp_fake)
        epoches_l.append(epoch)

    return real, fake_l, epoches_l


# def Mean_Corr(data):
#     pnl = np.sum(data, axis=2)
#     corr = np.corrcoef(pnl.T)
#     return corr

def Mean_Corr(data):
    # data: (n_samples, n_assets, n_time)
    x = data.transpose(0, 2, 1).reshape(-1, data.shape[1])  # (n_samples*n_time, n_assets)
    corr = np.corrcoef(x, rowvar=False)
    return corr


def Mean_AutoCorr(data, nlags):
    autocorr_l = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            series = data[i, j, :]
            try:
                autoc = acf(series, nlags=nlags, fft=False)
            except Exception:
                autoc = np.full(nlags + 1, np.nan)
            autocorr_l.append(autoc)

    autocorr_l = np.array(autocorr_l).reshape(data.shape[0], data.shape[1], nlags + 1)
    autocorr_mean = np.nanmean(autocorr_l, axis=0)
    return autocorr_mean


def Other_Real_Stats(opt, real):
    sub_corr_path = f'{opt.data_name}_Stats_Corr.csv'
    sub_auto_path = f'{opt.data_name}_Stats_Auto.csv'

    corr_file = join(result_data_path, sub_corr_path)
    auto_file = join(result_data_path, sub_auto_path)

    if isfile(corr_file) and isfile(auto_file):
        print('GroundTruth Estimates Exist!')
        real_corr_df = pd.read_csv(corr_file, index_col=0)
        real_autocorr_df = pd.read_csv(auto_file, index_col=0)
    else:
        print('Creating GroundTruth Estimates ......')

        real_corr_mean = Mean_Corr(real)
        real_corr_df = pd.DataFrame(
            np.round(real_corr_mean, 6),
            columns=opt.tickers,
            index=opt.tickers
        )
        real_corr_df.to_csv(corr_file)

        real_autocorr = Mean_AutoCorr(real, nlags)
        real_autocorr_df = pd.DataFrame(
            np.round(real_autocorr, 6),
            index=opt.tickers,
            columns=[f'AC-{i}' for i in range(nlags + 1)]
        )
        real_autocorr_df.to_csv(auto_file)

    return real_corr_df, real_autocorr_df


def Other_Fake_Stats(opt, fake_l, epoches_l, inc):
    corr_l = []
    autocorr_l = []

    selected_fake = fake_l[::inc]
    selected_epochs = epoches_l[::inc]

    if len(selected_fake) == 0:
        raise ValueError("Aucune fake data sélectionnée. Vérifie les epochs générés.")

    for fake in selected_fake:
        fake_corr = Mean_Corr(fake)
        corr_l.append(fake_corr.reshape(-1))

    index_l = [f'E{epoch}' for epoch in selected_epochs]

    corr_df = pd.DataFrame(np.array(corr_l), index=index_l)
    corr_save_path = join(result_data_path, this_version, 'Stats_Corr.csv')
    corr_df.to_csv(corr_save_path)

    for i, fake in enumerate(selected_fake):
        print('  ' * 4 + f'Epoch block {i + 1}/{len(selected_fake)}')
        fake_autocorr = Mean_AutoCorr(fake, nlags=nlags)
        autocorr_l.append(fake_autocorr.reshape(-1))

    autocorr_df = pd.DataFrame(np.array(autocorr_l), index=index_l)
    auto_save_path = join(result_data_path, this_version, 'Stats_Auto.csv')
    autocorr_df.to_csv(auto_save_path)

    return corr_l, autocorr_l, index_l


def CorrAuto_Error(opt, real, fake_l, epoches_l):
    inc = 10

    real_corr_df, real_autocorr_df = Other_Real_Stats(opt, real)
    corr_l, autocorr_l, index_l = Other_Fake_Stats(opt, fake_l, epoches_l, inc)

    corr_error_l = []
    autocorr_error_l = []
    err_index_l = []

    real_corr_flat = real_corr_df.values.reshape(-1)
    real_autocorr_flat = real_autocorr_df.values.reshape(-1)

    for i in range(len(corr_l)):
        corr_error_l.append(np.sum(np.abs(real_corr_flat - corr_l[i])))
        autocorr_error_l.append(np.sum(np.abs(real_autocorr_flat - autocorr_l[i])))
        err_index_l.append(index_l[i])

    df = pd.DataFrame(
        np.round(np.array([corr_error_l, autocorr_error_l]).T, 6),
        index=err_index_l,
        columns=['Corr', 'AutoCorr']
    )

    save_path = join(result_data_path, this_version, 'Stats_CorrAuto_Error.csv')
    df.to_csv(save_path)
    print(df)

    return df


def Corr_Plot(corr_df, name):
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    f, ax = plt.subplots(1, 1, figsize=(12, 10))

    sns.heatmap(
        corr_df,
        vmin=-1,
        vmax=1,
        ax=ax,
        cmap=cmap,
        cbar=True,
        square=True,
        linewidths=0.5,
        xticklabels=True,
        yticklabels=True
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'Correlation - {name}', fontsize=fontsize, color="black")

    f.tight_layout()
    f.savefig(join(plot_path, f'Corr_{name}.pdf'))
    plt.close(f)


def AutoCorr_Plot(auto_df, name):
    f, ax = plt.subplots(1, 1, figsize=(12, 8))

    plot_df = auto_df.copy()
    plot_df.reset_index(drop=True, inplace=True)

    ax.plot(plot_df, linewidth=2.5)
    ax.set_title(f'Autocorrelation - {name}', fontsize=fontsize, color="black")
    ax.set_xlim([0, nlags])
    ax.set_ylim([-0.25, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel('Lags', fontsize=18)
    ax.set_ylabel('Autocorrelation', fontsize=18)

    plt.grid(True)
    plt.legend(plot_df.columns.values, fontsize=12, loc='best')
    f.tight_layout()
    f.savefig(join(plot_path, f'AutoCorr_{name}.pdf'))
    plt.close(f)


if __name__ == "__main__":
    real, fake_l, epoches_l = Load_Data(opt, sample_number, model_index=0)

    CorrAuto_Error(opt, real, fake_l, epoches_l)

    sub_corr_path = f'{opt.data_name}_Stats_Corr.csv'
    sub_auto_path = f'{opt.data_name}_Stats_Auto.csv'

    real_corr_df = pd.read_csv(join(result_data_path, sub_corr_path), index_col=0)
    real_autocorr_df = pd.read_csv(join(result_data_path, sub_auto_path), index_col=0)

    corr_save_path = join(result_data_path, this_version, 'Stats_Corr.csv')
    corr_l = pd.read_csv(corr_save_path, index_col=0)

    if corr_l.shape[0] == 0:
        raise ValueError("Stats_Corr.csv est vide.")

    corr_vec = corr_l.mean(axis=0).values
    # corr_vec = corr_l.iloc[-1].values
    # target_epoch = 'E300'   # ou 'E400'
    # if target_epoch not in corr_l.index:
    #     raise ValueError(f"{target_epoch} absent de Stats_Corr.csv. Index dispo: {list(corr_l.index)}")
    # corr_vec = corr_l.loc[target_epoch].values
    expected_corr_size = opt.n_rows * opt.n_rows

    if corr_vec.size != expected_corr_size:
        raise ValueError(
            f"Taille incohérente pour la corrélation : {corr_vec.size} au lieu de {expected_corr_size}"
        )

    corr_df = corr_vec.reshape(opt.n_rows, opt.n_rows)
    corr_df = pd.DataFrame(corr_df, index=opt.tickers, columns=opt.tickers)

    print(np.sum(np.abs(real_corr_df.values.reshape(-1) - corr_df.values.reshape(-1))))
    Corr_Plot(corr_df, name='TailGAN')
    Corr_Plot(real_corr_df, name='GT')

    auto_save_path = join(result_data_path, this_version, 'Stats_Auto.csv')
    auto_l = pd.read_csv(auto_save_path, index_col=0)

    if auto_l.shape[0] == 0:
        raise ValueError("Stats_Auto.csv est vide.")

    # auto_vec = auto_l.mean(axis=0).values
    auto_vec = auto_l.iloc[-1].values
    # target_epoch = 'E300'   # ou 'E400'
    # if target_epoch not in auto_l.index:
    #     raise ValueError(f"{target_epoch} absent de Stats_Auto.csv. Index dispo: {list(auto_l.index)}")
    # auto_vec = auto_l.loc[target_epoch].values
    expected_auto_size = opt.n_rows * (nlags + 1)

    if auto_vec.size != expected_auto_size:
        raise ValueError(
            f"Taille incohérente pour l'autocorrélation : {auto_vec.size} au lieu de {expected_auto_size}"
        )

    auto_df = auto_vec.reshape(opt.n_rows, nlags + 1)
    auto_df = pd.DataFrame(
        auto_df,
        index=opt.tickers,
        columns=[f'AC-{i}' for i in range(nlags + 1)]
    )

    AutoCorr_Plot(auto_df.T, name='TailGAN')
    AutoCorr_Plot(real_autocorr_df.T, name='GT')
