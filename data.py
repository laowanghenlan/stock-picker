"""数据获取模块：通过 akshare 获取 A 股行情数据，并用 streamlit 缓存减少重复请求"""
import datetime as dt
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import akshare as ak
import pandas as pd
import streamlit as st

# 全市场快照的磁盘缓存文件：进程重启后仍可直接读盘，避免每次启动都联网拉取
_SPOT_FILE = Path(__file__).parent / "spot_cache.csv"
_SPOT_TIMEOUT = 100        # 全市场快照的硬超时（秒）。慢网正常约60~80秒，超过说明被限流
_spot_fail_at = 0.0        # 上次拉取失败的时刻：10分钟内不再重试，避免每次点击都卡满超时


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
    """联网拉取全市场快照（约1分钟）并写入本地缓存。
    akshare 内部请求没有超时，云端 IP 被东财限流时会无限挂起（页面一直转圈），
    这里用线程加硬超时，超时抛 TimeoutError 交给上层降级用缓存。"""
    global _spot_fail_at
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ak.stock_zh_a_spot_em)
    try:
        df = fut.result(timeout=_SPOT_TIMEOUT)
    except FuturesTimeout:
        _spot_fail_at = time.time()
        raise TimeoutError(f"行情接口 {_SPOT_TIMEOUT} 秒无响应，可能被限流或网络不通")
    except Exception:
        _spot_fail_at = time.time()
        raise
    finally:
        ex.shutdown(wait=False)       # 不等待卡住的线程，立即把控制权交回页面
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


def spot_fresh_enough(max_age_sec: int = 1800) -> bool:
    """快照缓存是否在 max_age_sec 秒内刷新过（避免每次点选股都强制全量重拉）"""
    try:
        return _SPOT_FILE.exists() and (time.time() - _SPOT_FILE.stat().st_mtime) < max_age_sec
    except Exception:
        return False


def spot_recently_failed(cooldown: int = 600) -> bool:
    """距上次拉取失败是否还在冷却期内（避免每次点击都白等100秒超时）"""
    return _spot_fail_at > 0 and (time.time() - _spot_fail_at) < cooldown


def _needs_auto_refresh() -> bool:
    """是否需要自动更新行情：
    规则——工作日开盘(9:30)之后，若缓存还是开盘前生成的，则自动刷新一次；
    其余时间（盘中已刷新过、收盘后、周末）直接用缓存，保证秒开。
    """
    if not _SPOT_FILE.exists():
        return True
    if time.time() - _spot_fail_at < 600:      # 刚失败过，别立刻再卡100秒
        return False
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


_IND_FILE = Path(__file__).parent / "industry_cache.csv"
_IND_TTL_DAYS = 30          # 行业归属很少变，缓存30天


def _read_ind_cache() -> dict:
    """读取代码→行业缓存，返回 {代码: 行业}；文件不存在或损坏返回空字典"""
    if not _IND_FILE.exists():
        return {}
    try:
        df = pd.read_csv(_IND_FILE, dtype={"代码": str})
        return dict(zip(df["代码"].str.zfill(6), df["行业"].astype(str)))
    except Exception:
        return {}


def _fetch_industry(code: str) -> str:
    """抓取单只股票的行业（东财个股信息），失败返回空串"""
    try:
        info = ak.stock_individual_info_em(symbol=code)
        hit = info[info["item"] == "行业"]
        return str(hit.iloc[0]["value"]).strip() if len(hit) else ""
    except Exception:
        return ""


def industry_map_impl(codes) -> dict:
    """返回 {代码: 行业}。优先读磁盘缓存，只对缺失的代码联网补抓（并发5路）。
    首次可能耗时几十秒，之后走缓存秒回；抓不到的代码不在结果里（上层按未分类处理）。"""
    cache = _read_ind_cache()
    if _IND_FILE.exists():
        try:
            stale = (time.time() - _IND_FILE.stat().st_mtime) > _IND_TTL_DAYS * 86400
        except Exception:
            stale = False
        if stale:
            cache = {}                      # 过期整体重抓
    codes = [str(c).zfill(6) for c in codes]
    todo = [c for c in codes if c not in cache]
    if todo:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=5) as pool:
            for c, ind in zip(todo, pool.map(_fetch_industry, todo)):
                if ind:
                    cache[c] = ind
        try:
            pd.DataFrame({"代码": list(cache), "行业": list(cache.values())}).to_csv(
                _IND_FILE, index=False, encoding="utf-8-sig")
        except Exception:
            pass                            # 写盘失败不影响本次使用
    return {c: cache[c] for c in codes if c in cache}


def find_code(keyword: str, spot: pd.DataFrame) -> pd.DataFrame:
    """按代码或名称关键字模糊搜索，最多返回20条"""
    kw = keyword.strip()
    if not kw:
        return spot.head(0)
    mask = (spot["代码"].astype(str).str.contains(kw)
            | spot["名称"].astype(str).str.contains(kw))
    return spot[mask].head(20)
