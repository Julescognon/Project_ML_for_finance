"""
Plot the quantiles of PnLs of benchmark strategies
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use("ggplot")

import numpy as np
import pandas as pd
import seaborn as sns
#sns.set()
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

from TailGAN import *
from Dataset import Dataset_IS

from NewStrategies import Momentum, Breakout, VolTarget
from matplotlib.ticker import FormatStrFormatter

# stock_names =  ['Gaussian', r'AR(1) with $\phi_1=0.5$', r'AR(1) with $\phi_2=-0.12$', r'GARCH(1,1) with $t(5)$', r'GARCH(1,1) with $t(10)$']
stock_names = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'LTC', 'TRX', 'LINK']

your_path = '/Users/jcognon/Tail-GAN'  # Replace with your actual path
plot_path = join(your_path, 'Plots')
gen_data_path = join(your_path, 'Gens/')
# result_data_path = join(your_path, 'Results/')
sample_number = 1000
result_data_path = join(your_path, f'Results_S{sample_number}')

def Compute_PNL_Spc(R, opt):
    R_Tensor = Tensor(R)

    # Price
    prices_l = Inc2Price(R_Tensor)
    port_prices_l = StaticPort(prices_l, opt.n_trans, opt.static_way, insample=True)

    PNL_BH = BuyHold(prices_l, opt.Cap)
    PNL_l = [PNL_BH]

    for strategy in opt.strategies:
        if strategy == 'Port':
            PNL_BHPort = BuyHold(port_prices_l, opt.Cap)
            PNL_l.append(PNL_BHPort)

        elif strategy == 'MR':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name, opt.tickers, strategy, percentile_l, opt.len, opt.WH
                )
                PNL_MR = MeanRev(
                    prices_l, opt.Cap, opt.WH,
                    LR=opt.ratios[0], SR=opt.ratios[1],
                    ST=thresholds_array[:, -1], LT=thresholds_array[:, -2]
                )
                PNL_l.append(PNL_MR)

        elif strategy == 'TF':
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    opt.data_name, opt.tickers, strategy, percentile_l, opt.len, opt.WH
                )
                PNL_TF = TrendFollow(
                    prices_l, opt.Cap, opt.WH,
                    LR=opt.ratios[0], SR=opt.ratios[1],
                    ST=thresholds_array[:, 0], LT=thresholds_array[:, 1]
                )
                PNL_l.append(PNL_TF)

        elif strategy == 'MOM':
            PNL_MOM = Momentum(
                prices_l, opt.Cap, opt.WH,
                LR=opt.ratios[0], SR=opt.ratios[1],
                upper_pct=opt.thresholds_pct[0][1],
                lower_pct=opt.thresholds_pct[0][0],
            )
            PNL_l.append(PNL_MOM)

        elif strategy == 'BO':
            PNL_BO = Breakout(
                prices_l, opt.Cap, opt.WH,
                LR=opt.ratios[0], SR=opt.ratios[1],
            )
            PNL_l.append(PNL_BO)

        elif strategy == 'VT':
            PNL_VT = VolTarget(
                prices_l, opt.Cap, opt.WH,
                target_vol=0.10,
            )
            PNL_l.append(PNL_VT)

        else:
            pass

    PNL = torch.cat(PNL_l, dim=1)
    return PNL.cpu().numpy()

# def Compute_PNL_Spc(R, opt):
#     R_Tensor = Tensor(R)
#     # Price
#     prices_l = Inc2Price(R_Tensor)

#     PNL_BH = BuyHold(prices_l, opt.Cap)
#     PNL_l = [PNL_BH]

#     thresholds_pct = [[31, 69]]
#     for strategy in ['MR', 'TF']:
#         if strategy == 'MR':
#             for percentile_l in thresholds_pct:
#                 thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 1000, opt.WH)
#                 PNL_MR = MeanRev(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
#                                  ST=thresholds_array[:, -1], LT=thresholds_array[:, -2])
#                 PNL_l.append(PNL_MR)
#         elif strategy == 'TF':
#             for percentile_l in thresholds_pct:
#                 thresholds_array = gen_thresholds(opt.data_name, opt.tickers, strategy, percentile_l, 1000, opt.WH)
#                 PNL_TF = TrendFollow(prices_l, opt.Cap, opt.WH, LR=opt.ratios[0], SR=opt.ratios[1],
#                                      ST=thresholds_array[:, 0], LT=thresholds_array[:, 1])
#                 PNL_l.append(PNL_TF)
#         else:
#             pass

#     PNL = torch.cat(PNL_l, dim=1)
#     return PNL.cpu().numpy()


def Load_Data(opt):
    # Realistic Data
    dataset = Dataset_IS(tickers=opt.tickers, data_path=join(your_path, "gan_data", opt.data_name), length=opt.len)
    real_r = np.array([d.detach().numpy() for d in dataset.samples])
    # sample_idx = random.sample(range(real_r.shape[0]), 10000)
    n_sample = min(10000, real_r.shape[0])
    sample_idx = random.sample(range(real_r.shape[0]), n_sample)
    sample_real = real_r[sample_idx, :, :]
    R_dic = {'Real': sample_real}

    # Fake Data
    exps_l = os.listdir(result_data_path)
    exps_l.sort()

    # target_runs = [exp for exp in exps_l if exp.startswith(this_version)]
    # category_dic = {
    #     'Tail-GAN': target_runs,
    # }
    target_runs = [
        exp for exp in exps_l
        if exp.startswith(this_version) and '_Model_' in exp
    ]
    category_dic = {
        'Tail-GAN': target_runs,
    }

    print("this_version =", this_version)
    print("target_runs =", target_runs)

    for k in category_dic:
        if len(category_dic[k]) == 0:
            continue

        run_version = category_dic[k][0]
        # gen_data_path_spc = join(gen_data_path, 'gen_data_' + run_version)
        gen_data_path_spc = join(gen_data_path, f'gen_data_{this_version}')
        fake_l = []

        print("run_version =", run_version)
        print("gen_data_path_spc =", gen_data_path_spc)

        for epoch in range(1, opt.n_epochs + 1):
            fake_file = join(gen_data_path_spc, f'Fake_id0_E{epoch}.npy')
            if isfile(fake_file):
                tmp_fake = np.load(fake_file)
                fake_l.append(tmp_fake)

        print("n_fake_files_found =", len(fake_l))

        if len(fake_l) == 0:
            continue

        fake_r = np.concatenate(fake_l)
        n_sample_fake = min(10000, fake_r.shape[0])
        sample_idx = random.sample(range(fake_r.shape[0]), n_sample_fake)
        sample_fake_r = fake_r[sample_idx, :, :]
        R_dic[k] = sample_fake_r

    return R_dic

def Split_PNL_Blocks(PNL, opt):
    num_stocks = len(opt.tickers)
    blocks = {}
    idx = 0

    # Buy-and-hold sur actifs
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

def VaR(PNL_dic, alpha):
    strategy_keys = ['Stk', 'MR', 'TF', 'MOM', 'BO', 'VT']

    for strat in strategy_keys:
        rows = {}

        for j, k in enumerate(PNL_dic.keys()):
            PNL = PNL_dic[k]
            blocks = Split_PNL_Blocks(PNL, opt)

            if strat not in blocks:
                continue

            block = blocks[strat]

            # pour Port, ce n’est pas lié aux tickers
            if strat == 'Port':
                names = [f'Trans-{i+1}' for i in range(block.shape[1])]
            else:
                names = opt.tickers[:block.shape[1]]

            vals = []
            for i in range(block.shape[1]):
                data = np.sort(block[:, i])
                size = block.shape[0]
                vals.append(np.round(data[int(alpha * size)], 3))

            rows[k] = vals

        if len(rows) == 0:
            continue

        df = pd.DataFrame(rows)

        if strat == 'Port':
            df.index = [f'Trans-{i+1}' for i in range(df.shape[0])]
        else:
            df.index = opt.tickers[:df.shape[0]]

        df.to_csv(join(plot_path, f'{strat}_VaR_Synthetic.csv'))

# def VaR(PNL_dic, alpha):
#     num_stocks = len(opt.tickers)
#     stk_var_dic = {}
#     mr_var_dic = {}
#     tf_var_dic = {}
#     for i in range(num_stocks):
#         stk_var_dic[opt.tickers[i]] = []
#         mr_var_dic[opt.tickers[i]] = []
#         tf_var_dic[opt.tickers[i]] = []

#         for j, k in enumerate(PNL_dic.keys()):
#             PNL = PNL_dic[k]
#             size = PNL.shape[0]
#             data1_stock = np.sort(PNL[:, i])
#             data1_mr = np.sort(PNL[:, num_stocks + i])
#             data1_tf = np.sort(PNL[:, 2 * num_stocks + i])
#             stk_var_dic[opt.tickers[i]].append(np.round(data1_stock[int(alpha*size)], 3))
#             mr_var_dic[opt.tickers[i]].append(np.round(data1_mr[int(alpha*size)], 3))
#             tf_var_dic[opt.tickers[i]].append(np.round(data1_tf[int(alpha*size)], 3))

#     stk_var_df = pd.DataFrame(stk_var_dic).T
#     stk_var_df.columns = PNL_dic.keys()
#     mr_var_df = pd.DataFrame(mr_var_dic).T
#     mr_var_df.columns = PNL_dic.keys()
#     tf_var_df = pd.DataFrame(tf_var_dic).T
#     tf_var_df.columns = PNL_dic.keys()

#     stk_var_df.to_csv(join(plot_path, 'Stock_VaR_Synthetic.csv'))
#     mr_var_df.to_csv(join(plot_path, 'MR_VaR_Synthetic.csv'))
#     tf_var_df.to_csv(join(plot_path, 'TF_VaR_Synthetic.csv'))

def Plot_Rank(PNL_dic, opt):
    strategy_keys = ['Stk', 'MR', 'TF', 'MOM', 'BO', 'VT']
    available_strats = []

    sample_blocks = Split_PNL_Blocks(next(iter(PNL_dic.values())), opt)
    for strat in strategy_keys:
        if strat in sample_blocks:
            available_strats.append(strat)

    num_stocks = len(opt.tickers)
    n_cols = len(available_strats)

    pdf_name = join(plot_path, 'Tails_IS_Synthetic.pdf')

    with PdfPages(pdf_name) as pdf:
        fig, axes = plt.subplots(num_stocks, n_cols, figsize=(5 * n_cols, 4 * num_stocks), sharex=True)

        if n_cols == 1:
            axes = np.array(axes).reshape(num_stocks, 1)

        col_titles = {
            'Stk': 'Static buy-and-hold',
            'MR': 'Mean-reversion',
            'TF': 'Trend-following',
            'MOM': 'Momentum',
            'BO': 'Breakout',
            'VT': 'Vol-targeting',
        }

        for j, strat in enumerate(available_strats):
            axes[0, j].set_title(col_titles[strat], fontsize=16)

        for i, row in enumerate(stock_names[:num_stocks]):
            axes[i, 0].set_ylabel(row + '\n' + r'$\alpha$-quantile (log scale)', rotation=90, fontsize=16)

        for j in range(n_cols):
            axes[-1, j].set_xlabel(r'$\alpha$ (log scale)', fontsize=16)

        for i in range(num_stocks):
            for k in PNL_dic.keys():
                PNL = PNL_dic[k]
                blocks = Split_PNL_Blocks(PNL, opt)

                for j, strat in enumerate(available_strats):
                    block = blocks[strat]

                    if i >= block.shape[1]:
                        continue

                    data = block[:, i]
                    size = len(data)
                    x = np.cumsum(np.ones(size)) / size

                    axes[i, j].grid(True)
                    axes[i, j].plot(x, np.sort(data), linewidth=3, label=k)
                    axes[i, j].set_yscale('symlog')
                    axes[i, j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    axes[i, j].set_xscale('log')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), fontsize=16)
        pdf.savefig()
        plt.close()

# def Plot_Rank(PNL_dic, opt):
#     pdf_name = join(plot_path, 'Tails_IS_Synthetic.pdf')

#     with PdfPages(pdf_name) as pdf:
#         num_stocks = len(opt.tickers)
#         fig, axes = plt.subplots(num_stocks, 3, figsize=(20, 24), sharex=True)

#         cols = ['Static buy-and-hold', 'Mean-reversion', 'Trend-following']
#         for ax, col in zip(axes[0], cols):
#             ax.set_title(col, fontsize=16)

#         rows = stock_names
#         for ax, row in zip(axes[:, 0], rows):
#             ax.set_ylabel(row+ '\n' + r'$\alpha$-quantile (log scale)', rotation=90, fontsize=16)

#         columns = [r'$\alpha$ (log scale)'] * 3
#         for ax, row in zip(axes[-1, :], columns):
#             ax.set_xlabel(row, rotation=0, fontsize=16)

#         for i in range(num_stocks):
#             for j, k in enumerate(PNL_dic.keys()):
#                 PNL = PNL_dic[k]
#                 size = PNL.shape[0]
#                 data1_stock = PNL[:, i]
#                 data1_mr = PNL[:, num_stocks+i]
#                 data1_tf = PNL[:, 2*num_stocks+i]

#                 x = np.cumsum(np.ones(size)) / size
#                 axes[i, 0].grid(True)
#                 axes[i, 0].plot(x, np.sort(data1_stock), linewidth=3, label=k)
#                 axes[i, 0].set_yscale('symlog')
#                 axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#                 axes[i, 1].grid(True)
#                 axes[i, 1].plot(x, np.sort(data1_mr), linewidth=3, label=k)
#                 axes[i, 1].set_yscale('symlog')
#                 axes[i, 1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#                 axes[i, 2].grid(True)
#                 axes[i, 2].plot(x, np.sort(data1_tf), linewidth=3, label=k)
#                 axes[i, 2].set_yscale('symlog')
#                 axes[i, 2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#             plt.xscale('log')

#         handles, labels = axes[0, 0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.51, -0.00), fontsize=16)
#         pdf.savefig()  # saves the current figure into a pdf page
#         plt.close()


if __name__ == '__main__':
    R_dic = Load_Data(opt)
    
    PNL_dic = {}
    
    for k in R_dic:
        PNL = Compute_PNL_Spc(R_dic[k], opt)
        if k == 'Real':
            k_name = 'Market Data'
        else:
            k_name = k
        PNL_dic[k_name] = PNL

    VaR(PNL_dic, alpha=0.05)
    Plot_Rank(PNL_dic, opt)