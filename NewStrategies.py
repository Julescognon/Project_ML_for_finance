"""
NewStrategies.py — Additional benchmark trading strategies for Tail-GAN
=========================================================================

This module extends the original Tail-GAN benchmark strategy set (BuyHold, MeanReversion,
TrendFollowing) with three new dynamic strategies that probe different dimensions of the
data-generating process:

    1. **Momentum (MOM)**
       Exploits serial correlation in cumulative returns. Computes the cumulative return
       over a rolling lookback window; goes long when the signal exceeds an upper threshold,
       short when it falls below a lower threshold. This differs from the existing
       TrendFollowing (which uses a dual-MA crossover) and from MeanReversion (which
       trades *against* deviations from a MA).

       Financial rationale: captures continuation effects (positive autocorrelation of
       returns at the window scale), which is a well-documented stylized fact in both
       equity and intraday markets (Jegadeesh & Titman 1993, Moskowitz et al. 2012).

    2. **Breakout (BO)**
       Goes long (resp. short) when the price crosses above (resp. below) its rolling
       maximum (resp. minimum) over the lookback window. Exit is triggered when the price
       reverts inside the channel. This is a classical channel-breakout (Donchian) strategy.

       Financial rationale: probes the distribution of price extremes, directly testing
       whether generated scenarios reproduce the correct frequency and magnitude of new
       highs/lows — a tail-relevant property that BH and MR do not capture.

    3. **Volatility-Targeting (VT)**
       A buy-and-hold strategy where the position size is dynamically scaled by the inverse
       of realized volatility estimated over a rolling window, targeting a constant
       annualized volatility exposure. No thresholds needed.

       Financial rationale: tests whether generated scenarios reproduce the heteroskedastic
       dynamics (volatility clustering, leverage effect). A generator that fails to
       reproduce GARCH-type features will produce incorrect PnL distributions under this
       strategy.

All strategies follow the same interface as the original Transform.py functions:
    Input:  prices_l (batch × n_assets × T+1 tensor of price levels)
    Output: sum_PNL  (batch × n_assets tensor of cumulative PnL)

They operate entirely in PyTorch and are compatible with the Tail-GAN training loop
(gradients flow through the differentiable NeuralSort in the discriminator, not through
the strategy itself, so non-differentiable operations like argmax are acceptable here,
as in the original MR/TF implementations).

Course: Machine Learning for Finance (J.-D. Fermanian, H. Pham)
"""

import numpy as np
import torch
from torch import nn

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
BoolTensor = torch.cuda.BoolTensor if cuda else torch.BoolTensor


# =============================================================================
# 1. MOMENTUM STRATEGY (MOM)
# =============================================================================

def Momentum(prices_l, Cap, WH, LR=1.0, SR=1.0, upper_pct=69, lower_pct=31):
    """
    Time-series momentum strategy based on rolling cumulative returns.

    For each asset and each scenario, at time t we compute:
        mom_signal(t) = price(t) / price(t - WH) - 1      (rolling return over WH periods)

    Trading rule (per asset, per scenario):
        - Go long  (+Cap * LR) when mom_signal crosses *above* the upper threshold
        - Go short (-Cap * SR) when mom_signal crosses *below* the lower threshold
        - Flatten position at the end of the path

    The thresholds are defined as percentiles of the momentum signal computed on the
    first `WH+1` periods (warm-up phase), calibrated in-sample. This mirrors the
    threshold calibration logic of MR/TF in the original codebase.

    Parameters
    ----------
    prices_l : Tensor, shape (batch, n_assets, T+1)
        Price paths starting at 1.
    Cap : float
        Maximum investment capital per asset.
    WH : int
        Lookback window for momentum computation.
    LR, SR : float
        Long ratio and short ratio (scaling of Cap).
    upper_pct, lower_pct : int
        Percentiles for upper/lower thresholds (applied to the momentum signal).

    Returns
    -------
    sum_PNL_MOM : Tensor, shape (batch, n_assets)
        Cumulative PnL of the momentum strategy for each asset and scenario.
    """
    batch, n_assets, T_plus_1 = prices_l.shape

    # --- Compute momentum signal: rolling return over WH periods ---
    # mom_signal[:, :, t] = prices_l[:, :, t] / prices_l[:, :, t-WH] - 1  for t >= WH
    # For t < WH, we set the signal to 0 (no trading in warm-up phase)
    mom_signal = torch.zeros_like(prices_l)
    if T_plus_1 > WH:
        # Avoid division by zero: add small epsilon
        mom_signal[:, :, WH:] = (
            prices_l[:, :, WH:] / (prices_l[:, :, :T_plus_1 - WH] + 1e-8) - 1.0
        )

    # --- Calibrate thresholds from the warm-up window ---
    # Use the first 2*WH+1 time steps for calibration (or all available)
    cal_end = min(2 * WH + 1, T_plus_1)
    cal_signal = mom_signal[:, :, WH:cal_end].reshape(-1).cpu().detach().numpy()
    if len(cal_signal) > 0 and np.std(cal_signal) > 1e-10:
        upper_th = float(np.percentile(cal_signal, upper_pct))
        lower_th = float(np.percentile(cal_signal, lower_pct))
    else:
        upper_th = 0.01
        lower_th = -0.01

    # --- Build positions ---
    # Flatten to (batch * n_assets, T+1) for vectorized processing
    signal_flat = mom_signal.view(batch * n_assets, -1)
    prices_flat = prices_l.view(batch * n_assets, -1)

    # Position: +1 when signal > upper_th, -1 when signal < lower_th, 0 otherwise
    position = torch.zeros_like(signal_flat)
    position[signal_flat > upper_th] = Cap * LR
    position[signal_flat < lower_th] = -Cap * SR
    # No position in warm-up phase
    position[:, :WH] = 0.0
    # Flatten at end
    position[:, -1] = 0.0

    # --- Compute PnL ---
    pnl = position[:, :-1] * (prices_flat[:, 1:] - prices_flat[:, :-1])
    pnl = pnl.reshape(batch, n_assets, -1)
    sum_PNL_MOM = torch.sum(pnl, dim=2)

    return sum_PNL_MOM


# =============================================================================
# 2. BREAKOUT (BO)
# =============================================================================

def Breakout(prices_l, Cap, WH, LR=1.0, SR=1.0):
    """
    Channel breakout (Donchian) strategy.

    At each time step t (for t >= WH), compute:
        upper_channel(t) = max(price[t-WH : t])
        lower_channel(t) = min(price[t-WH : t])

    Trading rule:
        - Go long  (+Cap * LR) when price(t) >= upper_channel(t)  (new high)
        - Go short (-Cap * SR) when price(t) <= lower_channel(t)  (new low)
        - Flatten when price reverts to mid-channel
        - Flatten at end

    This strategy directly probes the tail/extreme-value properties of generated paths:
    if the generator fails to reproduce the correct distribution of running maxima and
    minima, the PnL distribution under Breakout will be misspecified.

    Parameters
    ----------
    prices_l : Tensor, shape (batch, n_assets, T+1)
    Cap, WH, LR, SR : as in Momentum

    Returns
    -------
    sum_PNL_BO : Tensor, shape (batch, n_assets)
    """
    batch, n_assets, T_plus_1 = prices_l.shape
    prices_flat = prices_l.view(batch * n_assets, -1)  # (B*M, T+1)

    position = torch.zeros_like(prices_flat)

    for t in range(WH, T_plus_1):
        window = prices_flat[:, t - WH:t]  # (B*M, WH), excludes current price
        upper_ch = window.max(dim=1, keepdim=True)[0]   # (B*M, 1)
        lower_ch = window.min(dim=1, keepdim=True)[0]   # (B*M, 1)
        mid_ch = (upper_ch + lower_ch) / 2.0
        current_price = prices_flat[:, t:t + 1]         # (B*M, 1)

        # Breakout long
        long_signal = (current_price >= upper_ch).float()
        # Breakout short
        short_signal = (current_price <= lower_ch).float()
        # Neutral (revert to mid)
        neutral = ((current_price > lower_ch) & (current_price < upper_ch)).float()

        # If both long and short are zero and neutral is 1 → flatten
        # Otherwise keep breakout direction
        new_pos = Cap * LR * long_signal - Cap * SR * short_signal
        # If in neutral zone, carry previous position but decay toward 0
        # Simple rule: if neutral, set position to 0 (aggressive exit)
        position[:, t:t + 1] = new_pos * (1.0 - neutral) + 0.0 * neutral

    # Flatten at end
    position[:, -1] = 0.0

    # PnL computation
    pnl = position[:, :-1] * (prices_flat[:, 1:] - prices_flat[:, :-1])
    pnl = pnl.reshape(batch, n_assets, -1)
    sum_PNL_BO = torch.sum(pnl, dim=2)

    return sum_PNL_BO


# =============================================================================
# 3. VOLATILITY-TARGETING STRATEGY (VT)
# =============================================================================

def VolTarget(prices_l, Cap, WH, target_vol=0.10):
    """
    Volatility-targeting buy-and-hold strategy.

    At each time step t (for t >= WH), the position is:
        position(t) = Cap * (target_vol / realized_vol(t))

    where realized_vol(t) is the sample standard deviation of increments
    over [t-WH, t], annualized by sqrt(252 * T) (or equivalently, just the
    raw std since we work with the synthetic scale).

    The position is clipped to [-2*Cap, 2*Cap] to avoid extreme leverage.

    Financial rationale: a generator that fails to reproduce GARCH-type
    heteroskedastic dynamics will produce incorrect position sizing under
    this strategy, leading to a different PnL tail distribution.

    Parameters
    ----------
    prices_l : Tensor, shape (batch, n_assets, T+1)
    Cap : float
        Base capital.
    WH : int
        Lookback window for volatility estimation.
    target_vol : float
        Target volatility level (in the scale of the increments).

    Returns
    -------
    sum_PNL_VT : Tensor, shape (batch, n_assets)
    """
    batch, n_assets, T_plus_1 = prices_l.shape

    # Compute increments: ΔP(t) = P(t) - P(t-1)
    increments = prices_l[:, :, 1:] - prices_l[:, :, :-1]  # (batch, n_assets, T)
    T = increments.shape[2]

    # Flatten for processing
    inc_flat = increments.view(batch * n_assets, T)
    prices_flat = prices_l.view(batch * n_assets, T_plus_1)

    position = torch.zeros(batch * n_assets, T_plus_1).type(Tensor)

    for t in range(WH, T_plus_1):
        # Rolling realized volatility over [t-WH, t) on increments
        window_inc = inc_flat[:, max(0, t - WH):t]  # (B*M, WH) or less
        real_vol = torch.std(window_inc, dim=1, unbiased=True) + 1e-8  # (B*M,)

        # Scale position by inverse vol
        scale = target_vol / real_vol
        # Clip to avoid extreme leverage
        scale = torch.clamp(scale, min=0.0, max=2.0)

        # Long position (buy-and-hold direction), sized by vol-targeting
        position[:, t] = Cap * scale

    # No position during warm-up
    position[:, :WH] = 0.0
    # Flatten at end
    position[:, -1] = 0.0

    # PnL
    pnl = position[:, :-1] * (prices_flat[:, 1:] - prices_flat[:, :-1])
    pnl = pnl.reshape(batch, n_assets, -1)
    sum_PNL_VT = torch.sum(pnl, dim=2)

    return sum_PNL_VT


# =============================================================================
# STRATEGY REGISTRY — for clean integration with Compute_PNL
# =============================================================================

STRATEGY_REGISTRY = {
    'MOM': {
        'fn': Momentum,
        'description': 'Time-series momentum (rolling cumulative return signal)',
        'needs_thresholds': False,
    },
    'BO': {
        'fn': Breakout,
        'description': 'Channel breakout / Donchian strategy (rolling max/min)',
        'needs_thresholds': False,
    },
    'VT': {
        'fn': VolTarget,
        'description': 'Volatility-targeting buy-and-hold',
        'needs_thresholds': False,
    },
}
