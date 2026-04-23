#!/usr/bin/env python3
"""
Build a Tail-GAN-compatible crypto dataset from free public Binance Spot data.

What this script produces
-------------------------
A folder of CSV files ready for Tail-GAN's Dataset_IS / Dataset_OOS loaders:

    gan_data/Crypto10_Binance_1h/
        000001.csv
        000002.csv
        ...

Each CSV contains:
- rows   = timestamps inside one window
- cols   = assets / tickers
- values = log returns

This matches Tail-GAN's loader, which reads a folder of CSV files, selects the
requested ticker columns, and transposes each CSV into shape (n_assets, n_cols).

Default design choices
----------------------
- 10 major spot crypto pairs quoted in USDT
- hourly data from Binance public API (no API key)
- log-returns
- rolling windows of length 100
- step = 1 by default, because real crypto data would otherwise yield too few
  windows for Tail-GAN when using non-overlapping blocks
- chronological file order with zero-padded filenames, so Tail-GAN's simple
  lexical sort preserves time order

Typical usage
-------------
python build_tailgan_crypto_dataset.py \
    --repo_root /Users/you/Desktop/Tail-GAN \
    --data_name Crypto10_Binance_1h \
    --start 2021-01-01 \
    --end 2026-01-01 \
    --interval 1h \
    --window 100 \
    --step 1 \
    --train_ratio 0.8

Then train Tail-GAN with something like:
python TailGAN.py \
    --data_name Crypto10_Binance_1h \
    --tickers "['BTC','ETH','BNB','XRP','ADA','DOGE','SOL','LTC','TRX','LINK']" \
    --n_rows 10 \
    --n_cols 100 \
    --len <recommended_train_len>

Notes
-----
- Tail-GAN's argparse declarations for `type=list` are awkward on the CLI.
  If needed, hard-code the tickers directly in TailGAN.py or pass them from an
  environment you already know works in your setup.
- The script also writes a metadata JSON file with the recommended training
  length (`recommended_train_len`) so the first N samples can be used as IS and
  the remainder as OOS by the repo's dataset split logic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
DEFAULT_INTERVAL = "1h"
# DEFAULT_TICKERS = [
#     ("BTC", "BTCUSDT"),
#     ("ETH", "ETHUSDT"),
#     ("BNB", "BNBUSDT"),
#     ("XRP", "XRPUSDT"),
#     ("ADA", "ADAUSDT"),
#     ("DOGE", "DOGEUSDT"),
#     ("SOL", "SOLUSDT"),
#     ("LTC", "LTCUSDT"),
#     ("TRX", "TRXUSDT"),
#     ("LINK", "LINKUSDT"),
# ]
DEFAULT_TICKERS = [
    ("BTC", "BTCUSDT"),
    ("ETH", "ETHUSDT"),
    ("BNB", "BNBUSDT"),
    ("XRP", "XRPUSDT"),
    ("SOL", "SOLUSDT"),
]
INTERVAL_TO_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


@dataclass(frozen=True)
class SymbolSpec:
    name: str
    binance_symbol: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Tail-GAN-ready crypto dataset.")
    parser.add_argument(
        "--repo_root",
        type=str,
        required=True,
        help="Path to the Tail-GAN repository root. The script writes into <repo_root>/gan_data/<data_name>/.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="Crypto10_Binance_1h",
        help="Folder name created under gan_data/.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2021-01-01",
        help="Inclusive UTC start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2026-01-01",
        help="Exclusive UTC end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        choices=sorted(INTERVAL_TO_MS.keys()),
        help="Binance kline interval.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=100,
        help="Number of timestamps per sample window. Tail-GAN defaults to 100.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Window step. Use 1 for many overlapping samples, or set =window for non-overlapping blocks.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of samples reserved for training / IS. The remainder stays in the same folder and can be used as OOS.",
    )
    parser.add_argument(
        "--pause_seconds",
        type=float,
        default=0.10,
        help="Small pause between Binance requests.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=30.0,
        help="HTTP timeout per request.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing CSV files in the target dataset folder.",
    )
    return parser.parse_args()


def date_to_millis(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_existing_csvs(path: str) -> None:
    for filename in os.listdir(path):
        if filename.endswith(".csv"):
            os.remove(os.path.join(path, filename))


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_seconds: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    step_ms = INTERVAL_TO_MS[interval]
    url = f"{BASE_URL}{KLINES_PATH}"
    session = requests.Session()

    rows: List[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        response = session.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        batch = response.json()

        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Binance response for {symbol}: {batch}")
        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_start = last_open_time + step_ms

        if next_start <= current_start:
            raise RuntimeError(
                f"Pagination did not advance for {symbol}. current_start={current_start}, next_start={next_start}"
            )

        current_start = next_start
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if not rows:
        raise RuntimeError(f"No kline data returned for {symbol} in the requested range.")

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["open_time", "close"]].dropna().drop_duplicates(subset=["open_time"]).sort_values("open_time")
    df = df[(df["open_time"] >= pd.to_datetime(start_ms, unit="ms", utc=True)) & (df["open_time"] < pd.to_datetime(end_ms, unit="ms", utc=True))]
    return df.reset_index(drop=True)


def build_price_panel(
    symbols: Iterable[SymbolSpec],
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_seconds: float,
    timeout_seconds: float,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}

    for spec in symbols:
        print(f"Downloading {spec.binance_symbol} ...")
        df = fetch_klines(
            symbol=spec.binance_symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            pause_seconds=pause_seconds,
            timeout_seconds=timeout_seconds,
        )
        series_map[spec.name] = df.set_index("open_time")["close"].rename(spec.name)

    price_df = pd.concat(series_map.values(), axis=1, join="inner").sort_index()
    price_df = price_df.dropna(axis=0, how="any")

    if price_df.empty:
        raise RuntimeError("The aligned price panel is empty after inner-joining all assets.")

    return price_df


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    returns_df = np.log(price_df).diff().dropna()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if returns_df.empty:
        raise RuntimeError("The returns dataframe is empty after log-differencing.")
    return returns_df


def iter_windows(df: pd.DataFrame, window: int, step: int) -> Iterable[Tuple[int, pd.DataFrame]]:
    if window <= 0:
        raise ValueError("window must be > 0")
    if step <= 0:
        raise ValueError("step must be > 0")
    if len(df) < window:
        raise ValueError(f"Not enough rows ({len(df)}) for window={window}.")

    sample_idx = 0
    for start in range(0, len(df) - window + 1, step):
        end = start + window
        yield sample_idx, df.iloc[start:end].copy()
        sample_idx += 1


def write_windows(
    returns_df: pd.DataFrame,
    output_dir: str,
    window: int,
    step: int,
) -> int:
    total = 0
    max_samples = ((len(returns_df) - window) // step) + 1
    pad = max(6, len(str(max_samples)))

    for sample_idx, sample_df in iter_windows(returns_df, window=window, step=step):
        filename = f"{sample_idx + 1:0{pad}d}.csv"
        path = os.path.join(output_dir, filename)
        sample_df.to_csv(path, index=False)
        total += 1

    return total


def save_metadata(
    output_dir: str,
    data_name: str,
    symbols: List[SymbolSpec],
    interval: str,
    start: str,
    end: str,
    window: int,
    step: int,
    price_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    total_samples: int,
    train_ratio: float,
) -> None:
    recommended_train_len = max(1, int(math.floor(total_samples * train_ratio)))
    metadata = {
        "data_name": data_name,
        "source": "Binance Spot public klines",
        "interval": interval,
        "start": start,
        "end": end,
        "assets": [spec.name for spec in symbols],
        "binance_symbols": [spec.binance_symbol for spec in symbols],
        "n_rows": len(symbols),
        "n_cols": window,
        "step": step,
        "num_price_rows_aligned": int(len(price_df)),
        "num_return_rows": int(len(returns_df)),
        "num_samples_total": int(total_samples),
        "recommended_train_len": recommended_train_len,
        "recommended_oos_len": int(total_samples - recommended_train_len),
        "repo_usage": {
            "tailgan_data_name": data_name,
            "tailgan_tickers": [spec.name for spec in symbols],
            "tailgan_n_rows": len(symbols),
            "tailgan_n_cols": window,
            "tailgan_len": recommended_train_len,
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()

    if not (0 < args.train_ratio < 1):
        raise ValueError("train_ratio must be strictly between 0 and 1")

    start_ms = date_to_millis(args.start)
    end_ms = date_to_millis(args.end)
    if end_ms <= start_ms:
        raise ValueError("end must be strictly after start")

    symbols = [SymbolSpec(name=name, binance_symbol=symbol) for name, symbol in DEFAULT_TICKERS]

    gan_data_root = os.path.join(args.repo_root, "gan_data")
    output_dir = os.path.join(gan_data_root, args.data_name)
    ensure_dir(gan_data_root)
    ensure_dir(output_dir)

    existing_csvs = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
    if existing_csvs and not args.force:
        raise FileExistsError(
            f"{output_dir} already contains CSV files. Re-run with --force to overwrite them."
        )
    if existing_csvs and args.force:
        clean_existing_csvs(output_dir)

    print("Building aligned price panel...")
    price_df = build_price_panel(
        symbols=symbols,
        interval=args.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        pause_seconds=args.pause_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Aligned price rows: {len(price_df)}")

    returns_df = compute_log_returns(price_df)
    print(f"Return rows: {len(returns_df)}")

    total_samples = write_windows(
        returns_df=returns_df,
        output_dir=output_dir,
        window=args.window,
        step=args.step,
    )
    print(f"Saved {total_samples} samples to {output_dir}")

    save_metadata(
        output_dir=output_dir,
        data_name=args.data_name,
        symbols=symbols,
        interval=args.interval,
        start=args.start,
        end=args.end,
        window=args.window,
        step=args.step,
        price_df=price_df,
        returns_df=returns_df,
        total_samples=total_samples,
        train_ratio=args.train_ratio,
    )

    recommended_train_len = max(1, int(math.floor(total_samples * args.train_ratio)))
    print("\nRecommended Tail-GAN settings")
    print("---------------------------")
    print(f"data_name = {args.data_name}")
    print(f"tickers   = {[spec.name for spec in symbols]}")
    print(f"n_rows    = {len(symbols)}")
    print(f"n_cols    = {args.window}")
    print(f"len       = {recommended_train_len}  # first samples used for training / IS")
    print(f"OOS count = {total_samples - recommended_train_len}  # remaining samples")
    print(f"Metadata  = {os.path.join(output_dir, 'metadata.json')}")


if __name__ == "__main__":
    main()
