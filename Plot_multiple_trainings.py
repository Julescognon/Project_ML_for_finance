"""
Plot the performance of multiple TailGAN versions during training
on the same graphs.
"""

import argparse
import os
from os.path import join, isdir, isfile, basename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import pandas as pd
import seaborn as sns


your_path = '/Users/jcognon/Tail-GAN'
fontsize = 40
sample_number = 1000
DISPLAY_NAMES = {
    "Crypto10_TailGAN_base_1500": "TailGAN",
    "Crypto10_TailGAN_base_1500_corr1": "TailGAN_Corr",
    "Crypto10_TailGAN_newstrats_1500": "TailGAN_newstrats",
    "Crypto10_TailGAN_Static": "TailGAN_Static",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versions",
        nargs="+",
        required=True,
        help="List of versions to compare"
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.05],
        help="Tail levels to plot, e.g. --alphas 0.05 0.01"
    )
    parser.add_argument(
        "--keywords1",
        nargs="+",
        default=["", "Stk", "Trans", "MR", "TF", "MOM", "BO", "VT"],
        help="Strategy filters to plot"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="Epoch averaging window"
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=0.0,
        help="Lower y-axis bound"
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=2.0,
        help="Upper y-axis bound"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="Multiple_Trainings.pdf",
        help="Output PDF filename"
    )
    return parser.parse_args()


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


def get_version_folders(results_path, version):
    folders = [
        join(results_path, f)
        for f in os.listdir(results_path)
        if isdir(join(results_path, f)) and f.startswith(f"{version}_") and "_Model_" in f
    ]
    folders.sort()
    return folders


def build_stats_for_version(results_path, version, keyword1, keyword2):
    sample_error_path = join(results_path, 'Sample_Error_RE.csv')
    if not isfile(sample_error_path):
        raise FileNotFoundError(f"Missing sample error file: {sample_error_path}")

    sample_df = pd.read_csv(sample_error_path, index_col=0)
    sample_columns = _filter_columns(sample_df, keyword1, keyword2)
    if len(sample_columns) == 0:
        return None

    sample_mean = float(sample_df.loc['Sample-RE-Mean', sample_columns].mean())
    sample_std = float(sample_df.loc['Sample-RE-Std', sample_columns].mean())

    folders = get_version_folders(results_path, version)
    if len(folders) == 0:
        return None

    fake_RE_mean_l = []

    for folder in folders:
        mean_path = join(folder, 'Mean_OOS_RE_Mean.csv')
        if not isfile(mean_path):
            continue

        df = pd.read_csv(mean_path, index_col=0)
        columns = _filter_columns(df, keyword1, keyword2)
        if len(columns) == 0:
            continue

        fake_RE = df[columns].mean(1)
        fake_RE.index = pd.Index([int(i) for i in fake_RE.index], name='Epoch')
        fake_RE.name = basename(folder)
        fake_RE_mean_l.append(fake_RE)

    if len(fake_RE_mean_l) == 0:
        return None

    fake_RE_mean_all = pd.concat(fake_RE_mean_l, axis=1)

    sample_rows = pd.DataFrame(
        [
            np.repeat(sample_mean, fake_RE_mean_all.shape[1]),
            np.repeat(sample_std, fake_RE_mean_all.shape[1])
        ],
        index=['Sample-RE-Mean', 'Sample-RE-Std'],
        columns=fake_RE_mean_all.columns
    )

    fake_RE_mean_all = pd.concat([sample_rows, fake_RE_mean_all], axis=0)
    return fake_RE_mean_all


def smooth_dataframe(fake_RE, window):
    fake_RE = fake_RE.copy()
    fake_RE.index = fake_RE.index.astype(int)
    fake_RE = fake_RE.sort_index()

    if window > 1:
        subdf_group = fake_RE.groupby(fake_RE.index // window)
        fake_RE_cong_all = subdf_group.mean()
        fake_RE_cong_all.index = fake_RE_cong_all.index * window
        return fake_RE_cong_all

    return fake_RE


def plot_one_page(results_path, versions, color_dic, pdf, keyword1, keyword2, window, ymin, ymax):
    s1, s2 = Name_Keyword(keyword1, keyword2)

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    markers_l = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

    sample_RE = None
    sample_RE_std = None
    plotted_anything = False

    for iv, version in enumerate(versions):
        stats_df = build_stats_for_version(results_path, version, keyword1, keyword2)
        if stats_df is None:
            continue

        if 'Sample-RE-Mean' in stats_df.index and 'Sample-RE-Std' in stats_df.index:
            sample_RE = float(stats_df.loc['Sample-RE-Mean'].mean())
            sample_RE_std = float(stats_df.loc['Sample-RE-Std'].mean())

        fake_RE = stats_df.drop(index=['Sample-RE-Mean', 'Sample-RE-Std'], errors='ignore').copy()
        if fake_RE.empty:
            continue

        fake_RE = smooth_dataframe(fake_RE, window)

        fake_RE_mean = fake_RE.mean(1)
        fake_RE_std = fake_RE.std(1).fillna(0.0)

        y1 = fake_RE_mean
        y2 = y1 - fake_RE_std
        y3 = y1 + fake_RE_std

        ax.plot(
            y1.index,
            y1.values,
            # label=version,
            label=DISPLAY_NAMES.get(version, version),
            linewidth=4,
            marker=markers_l[iv % len(markers_l)],
            markersize=8,
            color=color_dic[version]
        )
        ax.fill_between(
            y1.index,
            y2.values,
            y3.values,
            alpha=0.20,
            color=color_dic[version]
        )

        plotted_anything = True

    if not plotted_anything:
        plt.close()
        return

    if sample_RE is not None and sample_RE_std is not None:
        ax.axhline(y=sample_RE, color='grey', linewidth=4, label='Sample mean')
        ax.axhline(y=sample_RE + sample_RE_std, color='grey', linewidth=2, linestyle='dashed')
        ax.axhline(y=max(0.0, sample_RE - sample_RE_std), color='grey', linewidth=2, linestyle='dashed')

    ax.set_ylabel('In-Sample Relative Error', fontsize=30)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.set_title(f'{s1} - alpha {s2}', fontsize=28)
    ax.set_ylim(ymin, ymax)
    ax.grid()
    ax.legend(fontsize=18)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    pdf.savefig()
    plt.close()


def main():
    args = parse_args()

    results_path = join(your_path, f"Results_S{sample_number}")
    os.makedirs(join(results_path, 'Plots'), exist_ok=True)

    keyword1_l = args.keywords1
    keyword2_l = [f"{alpha:.2f}" for alpha in args.alphas]

    palette = sns.color_palette("Set2", n_colors=max(len(args.versions), 3))
    color_dic = {
        version: palette[i]
        for i, version in enumerate(args.versions)
    }

    output_path = join(results_path, 'Plots', args.output_name)

    print("Comparing versions:")
    for v in args.versions:
        print(" -", v)
        print("   folders:", get_version_folders(results_path, v))

    with PdfPages(output_path) as pdf:
        for keyword1 in keyword1_l:
            for keyword2 in keyword2_l:
                plot_one_page(
                    results_path=results_path,
                    versions=args.versions,
                    color_dic=color_dic,
                    pdf=pdf,
                    keyword1=keyword1,
                    keyword2=keyword2,
                    window=args.window,
                    ymin=args.ymin,
                    ymax=args.ymax
                )

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()