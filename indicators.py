"""技术指标计算模块：MA / MACD / KDJ / RSI / BOLL
全部基于 pandas 实现，避免安装 TA-Lib 的繁琐过程"""
import pandas as pd


def add_ma(df: pd.DataFrame, windows=(5, 10, 20, 60)) -> pd.DataFrame:
    """添加均线列 MA5/MA10/MA20/MA60，基于收盘价"""
    for w in windows:
        df[f"MA{w}"] = df["收盘"].rolling(w).mean()
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """添加 MACD：DIF（快线差）、DEA（信号线）、MACD柱（国内惯例 = 2*(DIF-DEA)）"""
    ema_fast = df["收盘"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["收盘"].ewm(span=slow, adjust=False).mean()
    df["DIF"] = ema_fast - ema_slow
    df["DEA"] = df["DIF"].ewm(span=signal, adjust=False).mean()
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])
    return df


def add_kdj(df: pd.DataFrame, n=9) -> pd.DataFrame:
    """添加 KDJ（参数 9,3,3）：RSV -> K -> D -> J"""
    low_n = df["最低"].rolling(n).min()
    high_n = df["最高"].rolling(n).max()
    rsv = (df["收盘"] - low_n) / (high_n - low_n) * 100
    df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()
    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df


def add_rsi(df: pd.DataFrame, windows=(6, 12, 24)) -> pd.DataFrame:
    """添加 RSI 相对强弱指标：RSI6/RSI12/RSI24"""
    delta = df["收盘"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    for w in windows:
        avg_gain = gain.ewm(alpha=1 / w, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / w, adjust=False).mean()
        df[f"RSI{w}"] = 100 * avg_gain / (avg_gain + avg_loss)
    return df


def add_boll(df: pd.DataFrame, n=20, k=2) -> pd.DataFrame:
    """添加布林带：中轨(MA20)、上轨(+2倍标准差)、下轨(-2倍标准差)"""
    mid = df["收盘"].rolling(n).mean()
    std = df["收盘"].rolling(n).std()
    df["BOLL_MID"] = mid
    df["BOLL_UP"] = mid + k * std
    df["BOLL_LOW"] = mid - k * std
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算全部指标，返回新DataFrame"""
    df = df.copy()
    df = add_ma(df)
    df = add_macd(df)
    df = add_kdj(df)
    df = add_rsi(df)
    df = add_boll(df)
    return df
