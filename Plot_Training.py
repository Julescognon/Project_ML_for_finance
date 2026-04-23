# """
# Plot the performance of TailGAN during training.
# """

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import os
# from os.path import *
# from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# import openpyxl
# from openpyxl import load_workbook
# import random

# your_path = '/Users/jcognon/Tail-GAN'
# fontsize = 40
# sample_number = 1000


# def Name_Keyword(keyword1, keyword2):
#     if keyword1 == '':
#         s1 = 'All'
#     elif keyword1 == 'Stk':
#         s1 = 'Stocks'
#     elif keyword1 == 'Trans':
#         s1 = 'Static'
#     elif keyword1 == 'MR':
#         s1 = 'MR'
#     elif keyword1 == 'TF':
#         s1 = 'TF'
#     else:
#         s1 = keyword1

#     s2 = keyword2

#     return s1, s2


# def VaR_ES_Stats(category_dic, keyword1, keyword2, path):
#     s1, s2 = Name_Keyword(keyword1, keyword2)

#     for k in category_dic:
#         save_path_l = category_dic[k]
#         screen_save_path_l = save_path_l 
#         fake_RE_mean_l = []
#         if len(save_path_l) > 0:
#             for save_path in screen_save_path_l:
#                 df = pd.read_csv(f"{your_path}/Results_S{sample_number}/Sample_Error_RE.csv", index_col=0)

#                 orig_columns = df.columns

#                 columns = [i for i in orig_columns if keyword1 in i]
#                 columns = [i for i in columns if keyword2 in i]
#                 fake_RE = df[columns].mean(1)

#                 epoches = [int(i.split('-')[0][1:]) for i in fake_RE.index[2:]]
#                 fake_RE.index = ['Sample-RE-Mean'] + ['Sample-RE-Std'] +  epoches

#                 fake_RE_mean_l.append(fake_RE)

#             fake_RE_mean_all = pd.concat(fake_RE_mean_l, axis=1)
#             os.makedirs(join(path, 'Stats'), exist_ok=True)
#             fake_RE_mean_all.to_csv(join(path, 'Stats', '_'.join([k, s1, s2]) + 'IS.csv'))


# def VaR_ES_Plot(category_dic, color_dic, pdf, keyword1, keyword2, window):
#     s1, s2 = Name_Keyword(keyword1, keyword2)
#     markers_l = ['v', '^', 'o']
#     f, ax1 = plt.subplots(1, 1, figsize=(18, 12))
#     for ik, k in enumerate(category_dic.keys()):
#         save_path_l = category_dic[k]
#         color = color_dic[k]
#         if len(save_path_l) > 0:
#             fake_RE_mean_all = pd.read_csv(join(path, 'Stats', '_'.join([k, s1, s2]) + 'IS.csv'), index_col=0)

#             sample_RE = fake_RE_mean_all.loc['Sample-RE-Mean'].mean()
#             sample_RE_Std = fake_RE_mean_all.loc['Sample-RE-Std'].mean()

#             fake_RE = fake_RE_mean_all[2:]
#             print(" * " * 8 + k + " * " * 8 )
#             print("  " * 6 + " + " * 2 + "Sample" + " + " * 2 )
#             print(fake_RE_mean_all[:2].mean(1))
#             print("  " * 6 + " + " * 16)

#             fake_RE.index = fake_RE.index.astype(int)

#             subdf_group = fake_RE.groupby(fake_RE.index // window)
#             fake_RE_cong_all = subdf_group.mean()
#             fake_RE_cong_all.index *= window
#             fake_RE_cong_all.index += window

#             fake_RE_cong_mean = fake_RE_cong_all.mean(1)
#             fake_RE_cong_std = fake_RE_cong_all.std(1)
#             print(np.nanmin(fake_RE_cong_mean))
#             print(np.nanmin(fake_RE_cong_std))

#             y1 = fake_RE_cong_mean[int(100 / window): int(10000 / window)]

#             y2 = y1 - fake_RE_cong_std[int(100 / window): int(10000 / window)]
#             y3 = y1 + fake_RE_cong_std[int(100 / window): int(10000 / window)]

#             ax1.plot(y1.index, y1, color=color, label=k, linewidth=6, linestyle='-', marker=markers_l[ik], markersize=15)
#             ax1.fill_between(y1.index, y2, y3, color=color, alpha=0.5)

#     plt.axhline(y=sample_RE, color='grey', linewidth=6.6)
#     plt.axhline(y=sample_RE+sample_RE_Std, color='grey', linewidth=2, linestyle='dashed')
#     plt.grid()

#     ax1.set_ylabel('In-Sample Error', fontsize=30)
#     # ax1.set_ylabel(s_inout + ' Error across ' + s1, fontsize=fontsize)
#     ax1.set_xlabel('Epochs', fontsize=30)
#     ax1.set_ylim([0, 1.2])
#     ax1.legend(fontsize=30)
#     plt.xticks(fontsize=30)
#     plt.yticks(fontsize=30)
#     pdf.savefig()  # saves the current figure into a pdf page
#     plt.close()


# def Stats(path, category_dic, keyword1_l, keyword2_l):
#     for keyword1 in keyword1_l:
#         for keyword2 in keyword2_l:
#             VaR_ES_Stats(category_dic, keyword1, keyword2, path)


# def DataFrame_window(fake_RE_mean_all, window):
#     sample_RE = fake_RE_mean_all[:1]
#     fake_RE = fake_RE_mean_all[2:]
#     fake_RE.index = fake_RE.index.astype(int)

#     subdf_group = fake_RE.groupby(fake_RE.index // window)
#     fake_RE_cong_all = subdf_group.mean()
#     fake_RE_cong_all.index *= window
#     return sample_RE, fake_RE_cong_all


# def Plots(path, category_dic, color_dic, window, keyword1_l, keyword2_l):
#     os.makedirs(join(path, 'Plots'), exist_ok=True)
#     pdf_name = join(path, 'Plots', 'Tails_IS_Simple.pdf')

#     with PdfPages(pdf_name) as pdf:
#         for keyword1 in keyword1_l:
#             for keyword2 in keyword2_l:
#                 VaR_ES_Plot(category_dic, color_dic, pdf, keyword1, keyword2, window)


# if __name__ == "__main__":
#     keyword1_l = ['']#, 'Stk', 'MR', 'TF']
#     keyword2_l = ['0.05']
#     window = 100

#     path = join(your_path, 'Tails_IS_Simple')

#     exps_l = os.listdir(path)
#     exps_l.sort()
#     save_path_p0_l = [join(path, exp) for exp in exps_l if ('P0' in exp) & isdir(join(path, exp))]
#     save_path_p50_l = [join(path, exp) for exp in exps_l if ('P50' in exp and 'MR' not in exp ) & isdir(join(path, exp))]
#     save_path_p50_mrtf_R1_l = [join(path, exp) for exp in exps_l if ('P50' in exp and 'MR' in exp and 'TF' in exp) & isdir(join(path, exp))]

#     category_dic = {'Tail-GAN-Raw': save_path_p0_l,
#                     'Tail-GAN-Static': save_path_p50_l,
#                     'Tail-GAN': save_path_p50_mrtf_R1_l,
#                     }
#     color_dic = {'Tail-GAN-Raw': sns.color_palette("Set2")[0],
#                  'Tail-GAN-Static': sns.color_palette("Set2")[2],
#                  'Tail-GAN': sns.color_palette("Set2")[1],
#                  }

#     Stats(path, category_dic, keyword1_l, keyword2_l)
#     Plots(path, category_dic, color_dic, window, keyword1_l, keyword2_l)


"""
Plot the performance of TailGAN during training for the current dataset/run.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
from os.path import *
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import openpyxl
from openpyxl import load_workbook

from TailGAN import opt, this_version

your_path = '/Users/jcognon/Tail-GAN'
fontsize = 40
sample_number = 1000


def Name_Keyword(keyword1, keyword2):
    if keyword1 == '':
        s1 = 'All'
    elif keyword1 == 'Stk':
        s1 = 'Stocks'
    elif keyword1 == 'Trans':
        s1 = 'Static'
    elif keyword1 == 'MR':
        s1 = 'MR'
    elif keyword1 == 'TF':
        s1 = 'TF'
    else:
        s1 = keyword1

    s2 = keyword2

    return s1, s2


def _filter_columns(df, keyword1, keyword2):
    columns = df.columns.to_list()
    if keyword1 != '':
        columns = [c for c in columns if keyword1 in c]
    if keyword2 != '':
        columns = [c for c in columns if keyword2 in c]
    return columns


def VaR_ES_Stats(category_dic, keyword1, keyword2, path):
    s1, s2 = Name_Keyword(keyword1, keyword2)

    sample_error_path = join(path, 'Sample_Error_RE.csv')
    if not isfile(sample_error_path):
        raise FileNotFoundError(f"Missing sample error file: {sample_error_path}")

    sample_df = pd.read_csv(sample_error_path, index_col=0)
    sample_columns = _filter_columns(sample_df, keyword1, keyword2)
    if len(sample_columns) == 0:
        return

    sample_mean = float(sample_df.loc['Sample-RE-Mean', sample_columns].mean())
    sample_std = float(sample_df.loc['Sample-RE-Std', sample_columns].mean())

    for k in category_dic:
        save_path_l = category_dic[k]
        fake_RE_mean_l = []

        if len(save_path_l) > 0:
            for save_path in save_path_l:
                mean_path = join(save_path, 'Mean_OOS_RE_Mean.csv')
                if not isfile(mean_path):
                    continue

                df = pd.read_csv(mean_path, index_col=0)
                columns = _filter_columns(df, keyword1, keyword2)
                if len(columns) == 0:
                    continue

                fake_RE = df[columns].mean(1)
                fake_RE.index = pd.Index([int(i) for i in fake_RE.index], name='Epoch')
                fake_RE.name = basename(save_path)
                fake_RE_mean_l.append(fake_RE)

            if len(fake_RE_mean_l) == 0:
                continue

            fake_RE_mean_all = pd.concat(fake_RE_mean_l, axis=1)

            sample_rows = pd.DataFrame(
                [np.repeat(sample_mean, fake_RE_mean_all.shape[1]),
                 np.repeat(sample_std, fake_RE_mean_all.shape[1])],
                index=['Sample-RE-Mean', 'Sample-RE-Std'],
                columns=fake_RE_mean_all.columns
            )

            fake_RE_mean_all = pd.concat([sample_rows, fake_RE_mean_all], axis=0)

            os.makedirs(join(path, 'Stats'), exist_ok=True)
            fake_RE_mean_all.to_csv(join(path, 'Stats', '_'.join([k, s1, s2]) + 'IS.csv'))


def VaR_ES_Plot(category_dic, color_dic, pdf, keyword1, keyword2, window):
    s1, s2 = Name_Keyword(keyword1, keyword2)
    markers_l = ['v', '^', 'o', 's', 'D', 'P', 'X']
    f, ax1 = plt.subplots(1, 1, figsize=(18, 12))

    sample_RE = None
    sample_RE_Std = None

    for ik, k in enumerate(category_dic.keys()):
        stats_path = join(path, 'Stats', '_'.join([k, s1, s2]) + 'IS.csv')
        if not isfile(stats_path):
            continue

        fake_RE_mean_all = pd.read_csv(stats_path, index_col=0)

        if 'Sample-RE-Mean' not in fake_RE_mean_all.index or 'Sample-RE-Std' not in fake_RE_mean_all.index:
            continue

        sample_RE = float(fake_RE_mean_all.loc['Sample-RE-Mean'].mean())
        sample_RE_Std = float(fake_RE_mean_all.loc['Sample-RE-Std'].mean())

        fake_RE = fake_RE_mean_all.drop(index=['Sample-RE-Mean', 'Sample-RE-Std'], errors='ignore').copy()
        if fake_RE.empty:
            continue

        fake_RE.index = fake_RE.index.astype(int)
        fake_RE = fake_RE.sort_index()

        if window > 1:
            subdf_group = fake_RE.groupby(fake_RE.index // window)
            fake_RE_cong_all = subdf_group.mean()
            fake_RE_cong_all.index = fake_RE_cong_all.index * window
        else:
            fake_RE_cong_all = fake_RE.copy()

        fake_RE_cong_mean = fake_RE_cong_all.mean(1)
        fake_RE_cong_std = fake_RE_cong_all.std(1).fillna(0.0)

        y1 = fake_RE_cong_mean
        y2 = y1 - fake_RE_cong_std
        y3 = y1 + fake_RE_cong_std

        color = color_dic[k]
        marker = markers_l[ik % len(markers_l)]
        ax1.plot(y1.index, y1.values, color=color, label=k, linewidth=4, linestyle='-', marker=marker, markersize=10)
        ax1.fill_between(y1.index, y2.values, y3.values, color=color, alpha=0.25)

    if sample_RE is not None:
        plt.axhline(y=sample_RE, color='grey', linewidth=4)
        plt.axhline(y=sample_RE + sample_RE_Std, color='grey', linewidth=2, linestyle='dashed')
        lower = max(0.0, sample_RE - sample_RE_Std)
        plt.axhline(y=lower, color='grey', linewidth=2, linestyle='dashed')

    plt.grid()

    ax1.set_ylabel('In-Sample Relative Error', fontsize=30)
    ax1.set_xlabel('Epochs', fontsize=30)
    ax1.set_title(f'{s1} - alpha {s2}', fontsize=28)
    ax1.set_ylim(0, 2)
    ax1.legend(fontsize=22)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    pdf.savefig()
    plt.close()


def Stats(path, category_dic, keyword1_l, keyword2_l):
    for keyword1 in keyword1_l:
        for keyword2 in keyword2_l:
            VaR_ES_Stats(category_dic, keyword1, keyword2, path)


def DataFrame_window(fake_RE_mean_all, window):
    sample_RE = fake_RE_mean_all[:1]
    fake_RE = fake_RE_mean_all[2:]
    fake_RE.index = fake_RE.index.astype(int)

    subdf_group = fake_RE.groupby(fake_RE.index // window)
    fake_RE_cong_all = subdf_group.mean()
    fake_RE_cong_all.index *= window
    return sample_RE, fake_RE_cong_all


def Plots(path, category_dic, color_dic, window, keyword1_l, keyword2_l):
    os.makedirs(join(path, 'Plots'), exist_ok=True)
    pdf_name = join(path, 'Plots', f'{this_version}_Training.pdf')

    with PdfPages(pdf_name) as pdf:
        for keyword1 in keyword1_l:
            for keyword2 in keyword2_l:
                VaR_ES_Plot(category_dic, color_dic, pdf, keyword1, keyword2, window)


if __name__ == "__main__":
    # keyword1_l = ['', 'Stk', 'Trans', 'MR', 'TF']
    keyword1_l = ['', 'Stk', 'Trans', 'MR', 'TF', 'MOM', 'BO', 'VT']
    keyword2_l = [f'{alpha:.2f}' for alpha in opt.alphas]
    window = 1

    path = join(your_path, f'Results_S{sample_number}')
    os.makedirs(path, exist_ok=True)

    # folders = [
    #     join(path, f)
    #     for f in os.listdir(path)
    #     if isdir(join(path, f)) and f.startswith(f'{this_version}_Model_')
    # ]
    folders = [
    join(path, f)
    for f in os.listdir(path)
    if isdir(join(path, f)) and f.startswith(f'{this_version}_') and '_Model_' in f
    ]
    print("this_version =", this_version)
    print("folders =", folders)
    folders.sort()

    category_dic = {
        'Tail-GAN': folders,
    }
    color_dic = {
        'Tail-GAN': sns.color_palette("Set2")[1],
    }

    Stats(path, category_dic, keyword1_l, keyword2_l)
    Plots(path, category_dic, color_dic, window, keyword1_l, keyword2_l)