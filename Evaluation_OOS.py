import argparse
import os
import random
import sys
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch

# -----------------------------
# Custom arguments for OOS eval
# -----------------------------
custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument("--oos_data_name", type=str, required=True,
                           help="Folder under gan_data/ containing the real OOS csv files.")
custom_parser.add_argument("--oos_len", type=int, default=None,
                           help="Number of OOS samples to load. Default: all csv files in oos_data_name.")
custom_parser.add_argument("--sample_number", type=int, default=1000,
                           help="Number of generated / sampled scenarios used to estimate VaR/ES.")
custom_parser.add_argument("--n_bootstrap", type=int, default=16,
                           help="Number of bootstrap repetitions for Oracle and fake models.")
custom_parser.add_argument("--epoch_block", type=int, default=10,
                           help="Number of consecutive Fake_id*_E*.npy files concatenated together before evaluation.")
custom_parser.add_argument("--strategy_source", choices=["is", "oos"], default="is",
                           help="Use IS or OOS strategy parameters (thresholds/static portfolios) during evaluation.")
custom_parser.add_argument("--model_indices", type=str, default="",
                           help="Comma-separated model indices to evaluate. Default: Screen_Ensemble().")
custom_args, remaining_args = custom_parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

from TailGAN import opt, this_version, gen_data_path, Screen_Ensemble, Tensor, your_path  # noqa: E402
from Transform import Inc2Price, StaticPort, BuyHold, MeanRev, TrendFollow  # noqa: E402
from gen_thresholds import gen_thresholds  # noqa: E402

np.random.seed(1)
random.seed(1)
torch.manual_seed(1)

SAMPLE_NUMBER = custom_args.sample_number
RESULTS_ROOT = join(your_path, f"Results_OOS_S{SAMPLE_NUMBER}")
os.makedirs(RESULTS_ROOT, exist_ok=True)


def load_folder_data(data_name, tickers, max_len=None):
    data_path = join(your_path, "gan_data", data_name)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")

    files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
    if len(files) == 0:
        raise RuntimeError(f"No csv files found in {data_path}")

    if max_len is not None:
        files = files[:max_len]

    samples = []
    for f in files:
        f_path = join(data_path, f)
        arr = pd.read_csv(f_path)[tickers].values.T
        samples.append(arr)

    data = np.stack(samples, axis=0)
    return data, files


def get_strategy_reference():
    if custom_args.strategy_source == "is":
        return {
            "data_name": opt.data_name,
            "length": opt.len,
            "insample": True,
        }

    oos_data, _ = load_folder_data(custom_args.oos_data_name, opt.tickers, custom_args.oos_len)
    return {
        "data_name": custom_args.oos_data_name,
        "length": oos_data.shape[0],
        "insample": False,
    }


STRATEGY_REF = get_strategy_reference()


def empirical_stats(data_np):
    ep_stats_l = []
    for alpha in opt.alphas:
        var_l = np.percentile(data_np, alpha * 100, axis=0)
        var_l_reshaped = var_l.reshape(1, *var_l.shape)
        if alpha < 0.5:
            tmp_data = data_np * (data_np <= var_l_reshaped)
            tmp_data[tmp_data == 0.0] = np.nan
            es = np.nanmean(tmp_data, axis=0)
        else:
            tmp_data = data_np * (data_np >= var_l_reshaped)
            tmp_data[tmp_data == 0.0] = np.nan
            es = np.nanmean(tmp_data, axis=0)
        ep_stats_l.extend(np.stack([var_l, es]))
    return np.round(np.array(ep_stats_l).T, 6)



def compute_pnl_np(R_np):
    R_tensor = Tensor(R_np)
    prices_l = Inc2Price(R_tensor)
    port_prices_l = StaticPort(prices_l, opt.n_trans, opt.static_way, insample=STRATEGY_REF["insample"])

    pnl_bh = BuyHold(prices_l, opt.Cap)
    pnl_l = [pnl_bh]
    columns = [f"Stk-{i + 1}" for i in range(pnl_bh.shape[1])]

    for strategy in opt.strategies:
        if strategy == "Port":
            pnl_bh_port = BuyHold(port_prices_l, opt.Cap)
            pnl_l.append(pnl_bh_port)
            columns.extend([f"Trans-{i + 1}" for i in range(pnl_bh_port.shape[1])])
        elif strategy == "MR":
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    STRATEGY_REF["data_name"],
                    opt.tickers,
                    strategy,
                    percentile_l,
                    STRATEGY_REF["length"],
                    opt.WH,
                )
                pnl_mr = MeanRev(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thresholds_array[:, -1],
                    LT=thresholds_array[:, -2],
                )
                pnl_l.append(pnl_mr)
                columns.extend([f"MR-{i + 1}" for i in range(pnl_mr.shape[1])])
        elif strategy == "TF":
            for percentile_l in opt.thresholds_pct:
                thresholds_array = gen_thresholds(
                    STRATEGY_REF["data_name"],
                    opt.tickers,
                    strategy,
                    percentile_l,
                    STRATEGY_REF["length"],
                    opt.WH,
                )
                pnl_tf = TrendFollow(
                    prices_l,
                    opt.Cap,
                    opt.WH,
                    LR=opt.ratios[0],
                    SR=opt.ratios[1],
                    ST=thresholds_array[:, 0],
                    LT=thresholds_array[:, 1],
                )
                pnl_l.append(pnl_tf)
                columns.extend([f"TF-{i + 1}" for i in range(pnl_tf.shape[1])])

    pnl = torch.cat(pnl_l, dim=1)
    return pnl.detach().cpu().numpy(), columns



def safe_relative_error(ref_vec, est_vec, eps=1e-8):
    denom = np.maximum(np.abs(ref_vec), eps)
    return np.abs(ref_vec - est_vec) / denom



def build_columns(pnl_columns):
    stat_cols = []
    for alpha in opt.alphas:
        stat_cols.extend([f"VaR_{alpha:.2f}", f"ES_{alpha:.2f}"])
    return [f"{pnl_col}_{stat_col}" for pnl_col in pnl_columns for stat_col in stat_cols]


def get_ground_truth_oos():
    cache_dir = join(RESULTS_ROOT, "GroundTruth")
    os.makedirs(cache_dir, exist_ok=True)
    cache_name = "_".join([
        custom_args.oos_data_name,
        f"strategy_{custom_args.strategy_source}",
        "_".join(opt.strategies),
        f"P{opt.n_trans}",
        f"Cap{opt.Cap}",
        f"WH{opt.WH}",
        "Q" + "+".join([str(a) for a in opt.alphas]),
        "R" + "+".join([str(a) for a in opt.ratios]),
        "T" + "+".join(["_".join(map(str, i)) for i in opt.thresholds_pct]),
        "ground_truth.csv",
    ])
    cache_path = join(cache_dir, cache_name)

    if isfile(cache_path):
        df = pd.read_csv(cache_path, index_col=0)

        meta_cols = ["num_oos_samples", "first_file", "last_file"]
        stat_cols = [c for c in df.columns if c not in meta_cols]

        stats_vec = df.loc["Real_OOS", stat_cols].astype(float).values
        return stats_vec, stat_cols, df

    real_oos, used_files = load_folder_data(custom_args.oos_data_name, opt.tickers, custom_args.oos_len)
    pnl_oos, pnl_cols = compute_pnl_np(real_oos)
    stats_vec = empirical_stats(pnl_oos).reshape(-1)
    final_cols = build_columns(pnl_cols)

    df = pd.DataFrame([np.round(stats_vec, 6)], index=["Real_OOS"], columns=final_cols)
    df.insert(0, "num_oos_samples", real_oos.shape[0])
    df.insert(1, "first_file", used_files[0])
    df.insert(2, "last_file", used_files[-1])
    df.to_csv(cache_path)
    return stats_vec, final_cols, df



def compute_oracle(real_oos, real_stats_vec, final_cols):
    oracle_errs = []
    for _ in range(custom_args.n_bootstrap):
        sample_idx = np.random.choice(real_oos.shape[0], SAMPLE_NUMBER, replace=True)
        sample_real = real_oos[sample_idx, :, :]
        sample_pnl, _ = compute_pnl_np(sample_real)
        sample_vec = empirical_stats(sample_pnl).reshape(-1)
        oracle_errs.append(safe_relative_error(real_stats_vec, sample_vec))

    oracle_errs = np.stack(oracle_errs, axis=1)
    df = pd.DataFrame(
        np.round(np.vstack([oracle_errs.mean(axis=1), oracle_errs.std(axis=1)]), 6),
        index=["Oracle_RE_Mean", "Oracle_RE_Std"],
        columns=final_cols,
    )
    df.to_csv(join(RESULTS_ROOT, f"Oracle_{custom_args.oos_data_name}.csv"))
    return df



def load_fake_data_grouped(model_index):
    files = os.listdir(gen_data_path)
    epochs = [
        int(s.split("_E")[1][:-4])
        for s in files
        if s.startswith("Fake") and s.endswith(".npy") and f"id{model_index}" in s
    ]
    epochs = sorted(epochs)
    if len(epochs) == 0:
        raise RuntimeError(f"No Fake_id{model_index}_E*.npy found in {gen_data_path}")

    fake_groups = []
    epoch_labels = []

    step = max(1, custom_args.epoch_block)
    for start in range(0, len(epochs), step):
        block_epochs = epochs[start:start + step]
        block_arrays = [np.load(join(gen_data_path, f"Fake_id{model_index}_E{ep}.npy")) for ep in block_epochs]
        fake_groups.append(np.concatenate(block_arrays, axis=0))
        epoch_labels.append(block_epochs[0])

    return fake_groups, epoch_labels



def evaluate_model_against_oos(model_index, real_stats_vec, final_cols):
    fake_groups, epoch_labels = load_fake_data_grouped(model_index)
    mean_err_l = []
    std_err_l = []

    for epoch_label, fake in zip(epoch_labels, fake_groups):
        bootstrap_errs = []
        print(f"Model {model_index} | Epoch block starting at {epoch_label}")
        for _ in range(custom_args.n_bootstrap):
            sample_idx = np.random.choice(fake.shape[0], SAMPLE_NUMBER, replace=True)
            sample_fake = fake[sample_idx, :, :]
            fake_pnl, _ = compute_pnl_np(sample_fake)
            fake_vec = empirical_stats(fake_pnl).reshape(-1)
            bootstrap_errs.append(safe_relative_error(real_stats_vec, fake_vec))

        bootstrap_errs = np.stack(bootstrap_errs, axis=1)
        mean_err_l.append(bootstrap_errs.mean(axis=1))
        std_err_l.append(bootstrap_errs.std(axis=1))

    mean_df = pd.DataFrame(np.round(np.column_stack(mean_err_l).T, 6), index=epoch_labels, columns=final_cols)
    std_df = pd.DataFrame(np.round(np.column_stack(std_err_l).T, 6), index=epoch_labels, columns=final_cols)

    save_dir = join(RESULTS_ROOT, f"{this_version}_OOS_{custom_args.oos_data_name}_Model_{model_index}")
    os.makedirs(save_dir, exist_ok=True)
    mean_df.to_csv(join(save_dir, "Mean_OOS_RE_Mean.csv"))
    std_df.to_csv(join(save_dir, "Std_OOS_RE.csv"))

    avg_mean = mean_df.mean(axis=1)
    avg_std = std_df.mean(axis=1)
    best_epoch = int(avg_mean.idxmin())
    result = {
        "model_index": model_index,
        "best_epoch": best_epoch,
        "best_mean_re": float(avg_mean.loc[best_epoch]),
        "best_std_re": float(avg_std.loc[best_epoch]),
        "mean_df": mean_df,
        "std_df": std_df,
    }
    return result



def parse_model_indices():
    if custom_args.model_indices.strip() != "":
        return [int(x.strip()) for x in custom_args.model_indices.split(",") if x.strip() != ""]
    return Screen_Ensemble(thres_perc=50)



def save_summary(summary_rows, oracle_df):
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df["best_mean_pct"] = 100.0 * summary_df["best_mean_re"]
        summary_df["best_std_pct"] = 100.0 * summary_df["best_std_re"]

    oracle_mean = float(oracle_df.mean(axis=1).loc["Oracle_RE_Mean"])
    oracle_std = float(oracle_df.mean(axis=1).loc["Oracle_RE_Std"])

    oracle_row = pd.DataFrame([
        {
            "model_index": "Oracle",
            "best_epoch": "-",
            "best_mean_re": oracle_mean,
            "best_std_re": oracle_std,
            "best_mean_pct": 100.0 * oracle_mean,
            "best_std_pct": 100.0 * oracle_std,
        }
    ])

    final_df = pd.concat([oracle_row, summary_df], ignore_index=True)
    save_path = join(RESULTS_ROOT, f"Summary_{this_version}_vs_{custom_args.oos_data_name}.csv")
    final_df.to_csv(save_path, index=False)
    return final_df, save_path



def main():
    print("=" * 80)
    print("OOS evaluation")
    print("=" * 80)
    print(f"Train dataset       : {opt.data_name}")
    print(f"OOS dataset         : {custom_args.oos_data_name}")
    print(f"Strategy source     : {custom_args.strategy_source}")
    print(f"Fake data path      : {gen_data_path}")
    print(f"Results root        : {RESULTS_ROOT}")
    print(f"Sample number       : {SAMPLE_NUMBER}")
    print(f"Bootstrap repeats   : {custom_args.n_bootstrap}")
    print(f"Epoch block         : {custom_args.epoch_block}")
    print("=" * 80)

    real_oos, _ = load_folder_data(custom_args.oos_data_name, opt.tickers, custom_args.oos_len)
    real_stats_vec, final_cols, gt_df = get_ground_truth_oos()
    print("Ground truth OOS ready.")

    oracle_df = compute_oracle(real_oos, real_stats_vec, final_cols)
    print("Oracle summary:")
    print((100.0 * oracle_df.mean(axis=1)).round(4))

    model_indices = parse_model_indices()
    print(f"Models to evaluate: {model_indices}")

    summary_rows = []
    for model_index in model_indices:
        result = evaluate_model_against_oos(model_index, real_stats_vec, final_cols)
        summary_rows.append({
            "model_index": result["model_index"],
            "best_epoch": result["best_epoch"],
            "best_mean_re": result["best_mean_re"],
            "best_std_re": result["best_std_re"],
        })
        print(
            f"Model {model_index} best epoch {result['best_epoch']} | "
            f"Mean RE = {100.0 * result['best_mean_re']:.3f}% | "
            f"Std RE = {100.0 * result['best_std_re']:.3f}%"
        )

    final_df, save_path = save_summary(summary_rows, oracle_df)

    print("\nFinal summary:")
    display_cols = [c for c in ["model_index", "best_epoch", "best_mean_pct", "best_std_pct"] if c in final_df.columns]
    print(final_df[display_cols].round(4))
    print(f"\nSaved summary to: {save_path}")


if __name__ == "__main__":
    main()
