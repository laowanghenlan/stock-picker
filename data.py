"""数据获取模块：通过 akshare 获取 A 股行情数据，并用 streamlit 缓存减少重复请求"""
import datetime as dt
import re
import time
from pathlib import Path

import akshare as ak
import pandas as pd
import streamlit as st

# 全市场快照的磁盘缓存文件：进程重启后仍可直接读盘，避免每次启动都联网拉取
_SPOT_FILE = Path(__file__).parent / "spot_cache.csv"


def _read_spot_cache() -> pd.DataFrame | None:
    """读取本地快照缓存，不存在或损坏则返回 None"""
    if not _SPOT_FILE.exists():
        return None
    try:
        df = pd.read_csv(_SPOT_FILE, dtype={"代码": str})
        return df if len(df) > 100 else None
    except Exception:
        return None


def _fetch_spot() -> pd.DataFrame:
    """联网拉取全市场快照（约1分钟）并写入本地缓存"""
    df = ak.stock_zh_a_spot_em()
    try:
        df.to_csv(_SPOT_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass                           # 写盘失败不影响本次使用
    return df


def spot_cache_time() -> str:
    """返回本地缓存的数据时间，用于界面展示"""
    if not _SPOT_FILE.exists():
        return "无缓存"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(_SPOT_FILE.stat().st_mtime))


def _needs_auto_refresh() -> bool:
    """是否需要自动更新行情：
    规则——工作日开盘(9:30)之后，若缓存还是开盘前生成的，则自动刷新一次；
    其余时间（盘中已刷新过、收盘后、周末）直接用缓存，保证秒开。
    """
    if not _SPOT_FILE.exists():
        return True
    mtime = dt.datetime.fromtimestamp(_SPOT_FILE.stat().st_mtime)
    now = dt.datetime.now()
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return now.weekday() < 5 and now >= open_time and mtime < open_time


def get_spot_impl() -> pd.DataFrame:
    """获取全部 A 股实时行情快照（无缓存装饰器，供裸运行脚本直接调用）。
    策略：
      1. 有缓存且无需更新 → 读盘返回（毫秒级）
      2. 需要更新（工作日开盘后首次打开）→ 联网刷新；若联网失败则降级用旧缓存
      3. 完全没有缓存 → 联网拉取
    """
    cached = _read_spot_cache()
    if cached is not None and not _needs_auto_refresh():
        return cached
    if cached is not None:
        try:
            return _fetch_spot()
        except Exception:
            return cached              # 联网失败不阻塞，继续用旧快照
    return _fetch_spot()


# 缓存包装：web UI 一律调 get_spot；GitHub Actions 等裸运行环境请调 get_spot_impl
get_spot = st.cache_data(show_spinner="正在更新今日行情（每天首次打开需等待约1分钟，后续秒开）...")(get_spot_impl)


def refresh_spot() -> pd.DataFrame:
    """强制联网刷新快照并写盘，供界面「刷新行情」按钮调用"""
    df = _fetch_spot()
    get_spot.clear()                   # 只清行情缓存，不影响K线缓存
    return df


def get_history_impl(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    获取单只股票日K历史数据（前复权，无缓存装饰器，供裸运行脚本直接调用）
    symbol: 6位股票代码，如 "600519"；start/end: "YYYYMMDD"
    返回列：日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌幅、涨跌额、换手率
    """
    symbol = re.sub(r"\D", "", str(symbol))[-6:]  # 兼容带 sh/sz 前缀的代码
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                            start_date=start, end_date=end, adjust="qfq")
    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期").reset_index(drop=True)


# 缓存包装：web UI 一律调 get_history；GitHub Actions 等裸运行环境请调 get_history_impl
get_history = st.cache_data(ttl=3600, show_spinner="正在下载历史K线数据...")(get_history_impl)


def find_code(keyword: str, spot: pd.DataFrame) -> pd.DataFrame:
    """按代码或名称关键字模糊搜索，最多返回20条"""
    kw = keyword.strip()
    if not kw:
        return spot.head(0)
    mask = (spot["代码"].astype(str).str.contains(kw)
            | spot["名称"].astype(str).str.contains(kw))
    return spot[mask].head(20)
