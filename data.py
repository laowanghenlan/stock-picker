"""数据获取模块：通过 akshare 获取 A 股行情数据，并用 streamlit 缓存减少重复请求"""
import datetime as dt
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import akshare as ak
import pandas as pd
import requests
import streamlit as st

# 全市场快照的磁盘缓存文件：进程重启后仍可直接读盘，避免每次启动都联网拉取
_SPOT_FILE = Path(__file__).parent / "spot_cache.csv"
_CODE_FILE = Path(__file__).parent / "code_list.csv"   # 全A代码表：腾讯备用源的代码来源（随仓库提交）
_SPOT_TIMEOUT = 100        # 东财全市场快照的硬超时（秒）。慢网正常约60~80秒，超过说明被限流
_TX_BATCH = 600            # 腾讯批量接口单次请求的股票数
_TX_TIMEOUT = 15           # 腾讯单批请求超时（秒）
_TX_WORKERS = 6            # 腾讯分片并发数
_spot_fail_at = 0.0        # 所有数据源都失败的时刻：10分钟内不再重试，避免每次点击都卡满超时
_em_fail_at = 0.0          # 东财失败时刻：30分钟内直接走腾讯，不再白等100秒


def _read_spot_cache() -> pd.DataFrame | None:
    """读取本地快照缓存，不存在或损坏则返回 None"""
    if not _SPOT_FILE.exists():
        return None
    try:
        df = pd.read_csv(_SPOT_FILE, dtype={"代码": str})
        return df if len(df) > 100 else None
    except Exception:
        return None


def _write_spot_cache(df: pd.DataFrame) -> None:
    """快照落盘，供下次启动秒开；写盘失败不影响本次使用"""
    try:
        df.to_csv(_SPOT_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def _all_codes() -> list[str]:
    """全A代码表：优先用本地快照缓存里的代码（最新），没有则读仓库里的 code_list.csv。
    北交所（4/8/920 开头）小资金买不了，不请求，省一批流量。"""
    codes: list[str] = []
    cached = _read_spot_cache()
    if cached is not None and "代码" in cached.columns:
        codes = cached["代码"].astype(str).str.zfill(6).tolist()
    elif _CODE_FILE.exists():
        try:
            codes = pd.read_csv(_CODE_FILE, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
        except Exception:
            codes = []
    return [c for c in dict.fromkeys(codes) if not c.startswith(("4", "8", "920"))]


def _fetch_spot_em() -> pd.DataFrame:
    """东财全市场快照（一次请求拿全市场，约60~80秒）。
    akshare 内部请求没有超时，云端 IP 被东财限流时会无限挂起（页面一直转圈），
    这里用线程加硬超时，超时后由上层换腾讯备用源。"""
    global _em_fail_at
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(ak.stock_zh_a_spot_em)
    try:
        return fut.result(timeout=_SPOT_TIMEOUT)
    except FuturesTimeout:
        _em_fail_at = time.time()
        raise TimeoutError(f"东财行情接口 {_SPOT_TIMEOUT} 秒无响应，可能被限流或网络不通")
    except Exception:
        _em_fail_at = time.time()
        raise
    finally:
        ex.shutdown(wait=False)       # 不等待卡住的线程，立即把控制权交回页面


def _fetch_spot_tx() -> pd.DataFrame:
    """腾讯行情批量接口兜底：每批600只、6路并发，通常5秒内拿全市场。
    字段位置（~分隔）：1名称 2代码 3现价 4昨收 5今开 31涨跌额 32涨跌幅 33最高 34最低
    36成交量(手) 37成交额(万元) 38换手率 39市盈率TTM 43振幅 44流通市值(亿) 45总市值(亿)
    46市净率 49量比 52市盈率(动) 53市盈率(静)"""
    codes = _all_codes()
    if not codes:
        raise RuntimeError("没有可用的股票代码表（code_list.csv 缺失）")
    batches = [codes[i:i + _TX_BATCH] for i in range(0, len(codes), _TX_BATCH)]

    def fetch(batch: list[str]) -> list[str]:
        q = ",".join(("sh" if c.startswith(("6", "9")) else "sz") + c for c in batch)
        r = requests.get(f"https://qt.gtimg.cn/q={q}", timeout=_TX_TIMEOUT,
                         headers={"Referer": "https://gu.qq.com/"})
        r.encoding = "gbk"
        return r.text.split(";")

    def num(f: list[str], i: int) -> float:
        try:
            return float(f[i])
        except Exception:
            return float("nan")

    rows = []
    with ThreadPoolExecutor(max_workers=_TX_WORKERS) as ex:
        for lines in ex.map(fetch, batches):
            for ln in lines:
                f = ln.split("~")
                if len(f) < 54:
                    continue
                pe_dyn = num(f, 52)
                if pe_dyn != pe_dyn:          # 停牌/无动态PE时退回TTM
                    pe_dyn = num(f, 39)
                rows.append({
                    "代码": f[2].zfill(6), "名称": f[1], "最新价": num(f, 3),
                    "涨跌幅": num(f, 32), "涨跌额": num(f, 31),
                    "成交量": num(f, 36), "成交额": num(f, 37) * 1e4,
                    "振幅": num(f, 43), "最高": num(f, 33), "最低": num(f, 34),
                    "今开": num(f, 5), "昨收": num(f, 4),
                    "换手率": num(f, 38), "量比": num(f, 49),
                    "市盈率-动态": pe_dyn, "市净率": num(f, 46),
                    "总市值": num(f, 45) * 1e8, "流通市值": num(f, 44) * 1e8,
                })
    if len(rows) < 100:
        raise RuntimeError(f"腾讯备用源只返回 {len(rows)} 条数据，判定为异常")
    return pd.DataFrame(rows)


def _fetch_spot() -> pd.DataFrame:
    """拉全市场快照：东财优先（一次拿全），失败或超时自动降级腾讯批量接口。
    东财30分钟内翻过车就直接走腾讯，避免每次点击都白等100秒。"""
    global _spot_fail_at
    if time.time() - _em_fail_at >= 1800:
        try:
            df = _fetch_spot_em()
            _write_spot_cache(df)
            return df
        except Exception:
            pass
    try:
        df = _fetch_spot_tx()
        _write_spot_cache(df)
        return df
    except Exception:
        _spot_fail_at = time.time()       # 两个源都挂了才记录，用于界面10分钟冷却提示
        raise


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


def get_history_tx(symbol: str, start: str, end: str,
                   full: bool = True, fq: str = "qfq") -> pd.DataFrame:
    """腾讯日K兜底：支持前复权(qfq)/不复权(fq="")，实测单次可返 600+ 根。
    行格式 [日期, 开, 收, 高, 低, 成交量(手)]；full=False 只回日期+收盘（复盘用）。
    失败返回空 DataFrame，由上层决定继续降级。"""
    symbol = re.sub(r"\D", "", str(symbol))[-6:]
    if not symbol:
        return pd.DataFrame()
    pfx = "sh" if symbol.startswith(("6", "9")) else "sz"
    sd = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 else start
    ed = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 else end
    u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
         f"param={pfx}{symbol},day,{sd},{ed},800,{fq}")
    try:
        j = requests.get(u, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0",
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


# ---------------- 巨潮资讯（法定信披）公告排雷 ----------------
_CNINFO_FILE = Path(__file__).parent / "cninfo_stock.csv"   # 代码→orgId 映射缓存
_CNINFO_TTL_DAYS = 30
_CNINFO_KW = ("立案", "处罚", "警示函", "监管函", "问询函", "违规", "退市",
              "减持", "诉讼", "仲裁", "冻结", "停牌", "风险提示")


def _cninfo_org_map() -> dict:
    """代码→巨潮 orgId 映射。官网 stockList 一次给全市场（约6000只），缓存30天。"""
    if _CNINFO_FILE.exists():
        try:
            if time.time() - _CNINFO_FILE.stat().st_mtime < _CNINFO_TTL_DAYS * 86400:
                df = pd.read_csv(_CNINFO_FILE, dtype={"代码": str})
                return dict(zip(df["代码"].str.zfill(6), df["orgId"].astype(str)))
        except Exception:
            pass
    try:
        r = requests.get("https://www.cninfo.com.cn/new/data/szse_stock.json",
                         headers={"User-Agent": "Mozilla/5.0",
                                  "Referer": "https://www.cninfo.com.cn/"}, timeout=20)
        lst = r.json().get("stockList") or []
        m = {str(x["code"]).zfill(6): str(x["orgId"]) for x in lst}
        if m:
            pd.DataFrame({"代码": list(m), "orgId": list(m.values())}).to_csv(
                _CNINFO_FILE, index=False, encoding="utf-8-sig")
        return m
    except Exception:
        return {}


def cninfo_risk_impl(code: str, days: int = 7) -> list | None:
    """巨潮公告排雷：法定信披里的立案/处罚/减持/诉讼等，比新闻更硬。
    返回 [{"标题", "日期", "来源"}]；接口失败返回 None（上层降级处理，不当作无雷）。"""
    code = re.sub(r"\D", "", str(code))[-6:]
    org = _cninfo_org_map().get(code)
    if not org:
        return []                       # 非A股或映射缺失：无公告可查，不视为接口失败
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    payload = {"pageNum": 1, "pageSize": 50, "column": "szse", "tabName": "fulltext",
               "stock": f"{code},{org}", "seDate": f"{start}~{end}", "isHLtitle": "true"}
    try:
        r = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query",
                          data=payload, timeout=12,
                          headers={"User-Agent": "Mozilla/5.0",
                                   "Referer": "https://www.cninfo.com.cn/"})
        out = []
        for a in (r.json().get("announcements") or []):
            title = str(a.get("announcementTitle", ""))
            if any(k in title for k in _CNINFO_KW):
                ts = a.get("announcementTime")
                day = dt.datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d") if ts else ""
                out.append({"标题": title, "日期": day, "来源": "巨潮公告"})
        return out
    except Exception:
        return None


# ---------------- Tushare Pro 可选增强（需自行配置 token） ----------------
def tushare_enabled() -> bool:
    """是否配置了 TUSHARE_TOKEN。未配置时所有增强自动关闭，不影响现有流程。"""
    import os
    return bool(os.environ.get("TUSHARE_TOKEN", "").strip())


def tushare_daily_basic(trade_date: str | None = None) -> pd.DataFrame | None:
    """Tushare daily_basic：PE(TTM)/PB/换手率/流通市值/总市值，一次拿全市场。
    用于东财快照缺字段时补齐估值。未配置 token 或调用失败返回 None。"""
    if not tushare_enabled():
        return None
    try:
        import os
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
            "流通市值_ts": pd.to_numeric(df["circ_mv"], errors="coerce") * 1e4,   # 万元→元
            "总市值_ts": pd.to_numeric(df["total_mv"], errors="coerce") * 1e4,
        })
    except Exception:
        return None


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
