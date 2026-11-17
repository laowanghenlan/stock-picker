# -*- coding: utf-8 -*-
"""数据获取模块（v2：底层已换成 data_pipeline 多源 + DuckDB 数仓）。

对外接口完全不变 —— app.py / screen_service.py / daily_push.py 一行不用改。
底层新增：
  data_pipeline.DataPipeline —— 多源抽象 + 健康度路由 + 对账 + 熔断降级
  warehouse.Warehouse        —— DuckDB 内嵌列存（可选 coexist，CSV 仍保留）

三大痛点解决：
  1. 东财限流 → 腾讯/新浪/东财 三路并行，健康度加权选源，熔断冷却
  2. Baostock 常连不上 → 东财F10 作为第二基本面源，Baostock 挂了自动切
  3. K线历史拉不到 → 东财/腾讯qfq/腾讯不复权/新浪 四级降级链

旧实现仍保留在文件底部，用作 pipeline 不可用时的手动 fallback。
"""
from __future__ import annotations
import datetime as dt
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# -----------------------------------------------------------------------
# Pipeline + Warehouse 初始化（惰性加载，首次调用才 import，不阻塞启动）
# -----------------------------------------------------------------------
_PIPELINE: Optional[object] = None
_PIPELINE_LOCK = __import__("threading").Lock()
_PIPELINE_ENABLED = True   # 设 False 强制回退到旧实现


def _get_pipeline():
    """线程安全获取全局 DataPipeline 单例。"""
    global _PIPELINE
    if _PIPELINE is not None or not _PIPELINE_ENABLED:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            try:
                from data_pipeline import DataPipeline, get_pipeline
                from warehouse import Warehouse
                _PIPELINE = DataPipeline()
                # 不自动连 DuckDB（Streamlit Cloud 偶发磁盘权限问题时降级到 CSV 继续跑）
            except Exception:
                _PIPELINE = None
    return _PIPELINE


# -----------------------------------------------------------------------
# 行情快照（全市场）
# -----------------------------------------------------------------------
import streamlit as st

_SPOT_FILE = Path(__file__).parent / "spot_cache.csv"
_CODE_FILE = Path(__file__).parent / "code_list.csv"
_spot_fail_at = 0.0


def spot_cache_time() -> str:
    """返回本地缓存的数据时间，用于界面展示"""
    if not _SPOT_FILE.exists():
        return "无缓存"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(_SPOT_FILE.stat().st_mtime))


def spot_fresh_enough(max_age_sec: int = 1800) -> bool:
    try:
        return _SPOT_FILE.exists() and (time.time() - _SPOT_FILE.stat().st_mtime) < max_age_sec
    except Exception:
        return False


def spot_recently_failed(cooldown: int = 600) -> bool:
    """距上次拉取失败是否还在冷却期内"""
    return _spot_fail_at > 0 and (time.time() - _spot_fail_at) < cooldown


def _read_spot_cache() -> pd.DataFrame | None:
    if not _SPOT_FILE.exists():
        return None
    try:
        df = pd.read_csv(_SPOT_FILE, dtype={"代码": str})
        return df if len(df) > 100 else None
    except Exception:
        return None


def _write_spot_cache(df: pd.DataFrame) -> None:
    try:
        df.to_csv(_SPOT_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _needs_auto_refresh() -> bool:
    if not _SPOT_FILE.exists():
        return True
    if time.time() - _spot_fail_at < 600:
        return False
    mtime = dt.datetime.fromtimestamp(_SPOT_FILE.stat().st_mtime)
    now = dt.datetime.now()
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return now.weekday() < 5 and now >= open_time and mtime < open_time


def get_spot_impl() -> pd.DataFrame:
    """获取全部 A 股实时行情快照（pipeline 优先，失败回退旧实现）。"""
    global _spot_fail_at
    cached = _read_spot_cache()

    # Pipeline 路径：健康度加权选源 + 对账 + 自动写 CSV
    pl = _get_pipeline()
    if pl is not None:
        try:
            df = pl.get_spot()
            if df is not None and len(df) > 100:
                _write_spot_cache(df)
                _spot_fail_at = 0.0
                return df
        except Exception:
            _spot_fail_at = time.time()

    # 降级：读旧 CSV 缓存 / 或用旧实现拉
    if cached is not None:
        if spot_fresh_enough() or not _needs_auto_refresh():
            return cached
        # 缓存过期了但 pipeline 也挂了 → 尝试旧实现
        df = _fetch_spot_legacy()
        if df is not None and len(df) > 100:
            _write_spot_cache(df)
            return df
        return cached  # 最终兜底：旧缓存

    # 完全没有缓存 → 旧实现拉
    df = _fetch_spot_legacy()
    if df is not None and len(df) > 100:
        _write_spot_cache(df)
        return df
    raise RuntimeError("无法获取行情快照（所有源均失败且无本地缓存）")


# 缓存包装：web UI 一律调 get_spot；GitHub Actions 等裸运行环境请调 get_spot_impl
get_spot = st.cache_data(show_spinner="正在更新今日行情（每天首次打开需等待约1分钟，后续秒开）...")(get_spot_impl)


def refresh_spot() -> pd.DataFrame:
    """强制联网刷新快照并写盘。"""
    global _spot_fail_at
    pl = _get_pipeline()
    if pl is not None:
        try:
            df = pl.get_spot()
            _write_spot_cache(df)
            get_spot.clear()
            _spot_fail_at = 0.0
            return df
        except Exception:
            _spot_fail_at = time.time()

    df = _fetch_spot_legacy()
    if df is not None and len(df) > 100:
        _write_spot_cache(df)
        get_spot.clear()
        _spot_fail_at = 0.0
        return df
    raise RuntimeError("强制刷新失败")


# -----------------------------------------------------------------------
# 历史K线
# -----------------------------------------------------------------------
def get_history_impl(symbol: str, start: str, end: str) -> pd.DataFrame:
    """获取单只股票日K历史数据（前复权）。"""
    symbol = re.sub(r"\D", "", str(symbol))[-6:]

    pl = _get_pipeline()
    if pl is not None:
        try:
            df = pl.get_history(symbol, start, end, full=True, prefer_qfq=True)
            if df is not None and len(df) > 0:
                return df.sort_values("日期").reset_index(drop=True)
        except Exception:
            pass

    # 降级到旧实现
    return _get_history_legacy(symbol, start, end)


get_history = st.cache_data(ttl=3600, show_spinner="正在下载历史K线数据...")(get_history_impl)


def get_history_tx(symbol: str, start: str, end: str,
                   full: bool = True, fq: str = "qfq") -> pd.DataFrame:
    """腾讯日K（pipeline 包装，不复权时自动走 prefer_qfq=False）。"""
    symbol = re.sub(r"\D", "", str(symbol))[-6:]
    pl = _get_pipeline()
    if pl is not None:
        try:
            df = pl.get_history(symbol, start, end, full=full, prefer_qfq=(fq == "qfq"))
            if df is not None and len(df) > 0:
                return df.sort_values("日期").reset_index(drop=True)
        except Exception:
            pass
    return _get_history_tx_legacy(symbol, start, end, full, fq)


# -----------------------------------------------------------------------
# 行业映射 / 舆情 / Tushare（保持原实现，稳定可靠）
# -----------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
import requests

_INDUSTRY_FILE = Path(__file__).parent / "industry_cache.csv"


def _read_ind_cache() -> dict:
    if not _INDUSTRY_FILE.exists():
        return {}
    try:
        df = pd.read_csv(_INDUSTRY_FILE, dtype={"代码": str})
        return dict(zip(df["代码"].str.zfill(6), df["行业"].astype(str)))
    except Exception:
        return {}


def industry_map_impl(codes) -> dict:
    """返回 {代码: 行业}。读磁盘缓存 + 缺失的联网补抓。"""
    cache = _read_ind_cache()
    codes = [str(c).zfill(6) for c in codes]
    todo = [c for c in codes if c not in cache]
    if todo:
        try:
            import akshare as ak
            def _fetch(code):
                try:
                    info = ak.stock_individual_info_em(symbol=code)
                    hit = info[info["item"] == "行业"]
                    return str(hit.iloc[0]["value"]).strip() if len(hit) else ""
                except Exception:
                    return ""
            with ThreadPoolExecutor(max_workers=5) as pool:
                for c, ind in zip(todo, pool.map(_fetch, todo)):
                    if ind:
                        cache[c] = ind
            pd.DataFrame({"代码": list(cache), "行业": list(cache.values())}).to_csv(
                _INDUSTRY_FILE, index=False, encoding="utf-8-sig")
        except Exception:
            pass
    return {c: cache[c] for c in codes if c in cache}


# -----------------------------------------------------------------------
# 巨潮公告排雷（保持原实现，稳定可靠）
# -----------------------------------------------------------------------
_CNINFO_FILE = Path(__file__).parent / "cninfo_stock.csv"
_CNINFO_KW = ("立案", "处罚", "警示函", "监管函", "问询函", "违规", "退市",
              "减持", "诉讼", "仲裁", "冻结", "停牌", "风险提示")


def _cninfo_org_map() -> dict:
    if _CNINFO_FILE.exists() and (time.time() - _CNINFO_FILE.stat().st_mtime) < 30 * 86400:
        try:
            df = pd.read_csv(_CNINFO_FILE, dtype={"代码": str})
            return dict(zip(df["代码"].str.zfill(6), df["orgId"].astype(str)))
        except Exception:
            pass
    try:
        r = requests.get("https://www.cninfo.com.cn/new/data/szse_stock.json",
                         headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"},
                         timeout=20)
        m = {str(x["code"]).zfill(6): str(x["orgId"]) for x in (r.json().get("stockList") or [])}
        pd.DataFrame({"代码": list(m), "orgId": list(m.values())}).to_csv(
            _CNINFO_FILE, index=False, encoding="utf-8-sig")
        return m
    except Exception:
        return {}


def cninfo_risk_impl(code: str, days: int = 7) -> list | None:
    code = re.sub(r"\D", "", str(code))[-6:]
    org = _cninfo_org_map().get(code)
    if not org:
        return []
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    payload = {"pageNum": 1, "pageSize": 50, "column": "szse", "tabName": "fulltext",
               "stock": f"{code},{org}", "seDate": f"{start}~{end}", "isHLtitle": "true"}
    try:
        r = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query",
                          data=payload, timeout=12,
                          headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
        hits = []
        for a in (r.json().get("announcements") or []):
            title = str(a.get("announcementTitle", ""))
            if any(k in title for k in _CNINFO_KW):
                hits.append({"标题": title, "日期": "", "来源": "巨潮公告"})
        return hits
    except Exception:
        return None


# -----------------------------------------------------------------------
# Tushare Pro 可选增强（保持原实现）
# -----------------------------------------------------------------------
def tushare_enabled() -> bool:
    import os
    return bool(os.environ.get("TUSHARE_TOKEN", "").strip())


def tushare_daily_basic(trade_date: str | None = None) -> pd.DataFrame | None:
    import os
    if not tushare_enabled():
        return None
    try:
        import tushare as ts
        pro = ts.pro_api(os.environ["TUSHARE_TOKEN"].strip())
        if not trade_date:
            trade_date = dt.date.today().strftime("%Y%m%d")
        df = pro.daily_basic(trade_date=trade_date,
                             fields="ts_code,trade_date,pe_ttm,pb,turnover_rate,circ_mv,total_mv")
        if df is None or len(df) == 0:
            return None
        return pd.DataFrame({
            "代码": df["ts_code"].str[:6],
            "市盈率TTM_ts": pd.to_numeric(df["pe_ttm"], errors="coerce"),
            "市净率_ts": pd.to_numeric(df["pb"], errors="coerce"),
            "换手率_ts": pd.to_numeric(df["turnover_rate"], errors="coerce"),
            "流通市值_ts": pd.to_numeric(df["circ_mv"], errors="coerce") * 1e4,
            "总市值_ts": pd.to_numeric(df["total_mv"], errors="coerce") * 1e4,
        })
    except Exception:
        return None


# -----------------------------------------------------------------------
# 代码搜索
# -----------------------------------------------------------------------
def find_code(keyword: str, spot: pd.DataFrame) -> pd.DataFrame:
    kw = keyword.strip()
    if not kw:
        return spot.head(0)
    mask = (spot["代码"].astype(str).str.contains(kw)
            | spot["名称"].astype(str).str.contains(kw))
    return spot[mask].head(20)


# ===============================================================
# 旧实现（pipeline 不可用时的手动 fallback，保留但不再主导）
# ===============================================================
def _fetch_spot_legacy() -> pd.DataFrame | None:
    """旧实现快照拉取（腾讯优先 + 东财兜底）。"""
    try:
        from data_pipeline import TencentSource, EastMoneySource
        for src in [TencentSource(), EastMoneySource()]:
            try:
                df = src.fetch_spot()
                if df is not None and len(df) > 100:
                    return df
            except Exception:
                continue
    except Exception:
        pass
    # 兜底：akshare 东财直接调
    try:
        import akshare as ak
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FT
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(ak.stock_zh_a_spot_em)
        try:
            df = fut.result(timeout=100)
            if df is not None and len(df) > 100:
                df["代码"] = df["代码"].astype(str).str.zfill(6)
                return df
        except FT:
            pass
        finally:
            ex.shutdown(wait=False)
    except Exception:
        pass
    return None


def _get_history_legacy(symbol: str, start: str, end: str) -> pd.DataFrame:
    """旧实现 K线（东财 → 腾讯 qfq → 新浪 三级降级）。"""
    try:
        from data_pipeline import EastMoneySource, TencentSource, SinaSource
        for src, kwargs in [
            (EastMoneySource(), {}),
            (TencentSource(), {}),
            (SinaSource(), {}),
        ]:
            try:
                df = src.fetch_history(symbol, start, end, full=True)
                if df is not None and len(df) > 30:
                    return df.sort_values("日期").reset_index(drop=True)
            except Exception:
                continue
    except Exception:
        pass
    # 最终兜底：akshare 直接调
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is not None and len(df) > 0:
            df["日期"] = pd.to_datetime(df["日期"])
            return df.sort_values("日期").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


def _get_history_tx_legacy(symbol: str, start: str, end: str,
                           full: bool, fq: str) -> pd.DataFrame:
    """旧实现腾讯 K线。"""
    try:
        from data_pipeline import TencentSource
        df = TencentSource().fetch_history(symbol, start, end, full=full)
        if df is not None and len(df) > 0:
            return df.sort_values("日期").reset_index(drop=True)
    except Exception:
        pass
    import requests
    pfx = "sh" if symbol.startswith(("6", "9")) else "sz"
    sd = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 else start
    ed = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 else end
    u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
         f"param={pfx}{symbol},day,{sd},{ed},800,{fq}")
    try:
        j = requests.get(u, timeout=12, headers={"User-Agent": "Mozilla/5.0",
                                                   "Referer": "https://gu.qq.com/"}).json()
        d = j.get("data", {}).get(pfx + symbol, {})
        k = d.get("qfqday") or d.get("day") or []
        if not k:
            return pd.DataFrame()
        data = {"日期": pd.to_datetime([r[0] for r in k]),
                "收盘": [float(r[2]) for r in k]}
        if full:
            data.update({"开盘": [float(r[1]) for r in k],
                         "最高": [float(r[3]) for r in k],
                         "最低": [float(r[4]) for r in k],
                         "成交量": [float(r[5]) for r in k]})
        return pd.DataFrame(data).sort_values("日期").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()
