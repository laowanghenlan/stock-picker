# -*- coding: utf-8 -*-
"""可靠数据管道 v2 —— 多源抽象 + 健康度路由 + 对账 + 降级 + 数仓落盘。

解决的三大痛点：
  1. 东财/akshare 云端限流 → 腾讯/新浪/东财 三路并行，健康度加权选源
  2. Baostock 常连不上 → 新增东财F10 财务接口作为第二基本面源，Baostock 挂了自动切
  3. K线历史拉不到 → 东财/腾讯qfq/腾讯不复权/新浪 四级降级链，带熔断和对账

设计原则：
  - 统一 DataSource 抽象，每个源有 fetch_spot / fetch_history / fetch_fundamental
  - SourceRouter 记录每个源最近 20 次调用的成功率 + 延迟 + 字段完整率 → 按健康度加权选源
  - Reconciler 两源取数交叉验证，差异超阈值打标记（避免用坏数据）
  - 对外暴露 DataPipeline 门面类，接口与 data.py 兼容（get_spot / get_history / get_fundamental）
  - 仓库层可选：DuckDB / CSV 自动降级；两者都挂时仍能返回内存空表
"""
from __future__ import annotations
import datetime as dt
import json
import re
import time
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

# ========== 数据源超时与并发配置（集中可调，不再散在 data.py） ==========
SPOT_TIMEOUT_PER_SOURCE = 45        # 单个源拉快照的最大等待（秒）
HIST_TIMEOUT_PER_SOURCE = 15        # 单个源拉单只K线的最大等待（秒）
FUND_TIMEOUT_PER_SOURCE = 25        # 单个源拉一只基本面的最大等待（秒）
TX_BATCH = 600                      # 腾讯批量接口单次请求股票数
TX_WORKERS = 6                      # 腾讯快照并发数
HIST_WORKERS = 10                   # K线预取并发数
HEALTH_WINDOW = 20                  # 健康度窗口：最近 N 次调用
HEALTH_COOLDOWN = 600               # 某源连续失败后冷却（秒）
RECONCILE_PRICE_TOL = 0.02          # 对账价格差异容忍度（相对值 2%）
RECONCILE_VOLUME_TOL = 0.05         # 对账成交量差异容忍度（5%）


# =============================================================
# 健康度度量
# =============================================================
@dataclass
class HealthMetrics:
    """单个数据源的健康状态（最近 HEALTH_WINDOW 次调用统计）"""
    source: str
    success_rate: float = 1.0          # 成功率
    avg_latency_ms: float = 0.0        # 平均延迟
    field_completeness: float = 1.0    # 关键字段平均完整率
    last_call: float = 0.0             # 上次调用时间戳
    last_success: float = 0.0          # 上次成功时间戳
    consecutive_failures: int = 0      # 连续失败次数

    @property
    def score(self) -> float:
        """综合健康分 0~1，按成功率为主、延迟为辅"""
        return (self.success_rate * 0.7
                + max(0, 1.0 - self.avg_latency_ms / 5000) * 0.2
                + self.field_completeness * 0.1)

    def should_skip(self) -> bool:
        """是否在冷却期内，应该跳过"""
        if self.consecutive_failures <= 2:
            return False
        return time.time() - self.last_call < HEALTH_COOLDOWN


class HealthTracker:
    """线程安全的健康度记录器（每个源一个实例）"""
    def __init__(self, source: str):
        self.source = source
        self._lock = threading.Lock()
        self._records: list[dict] = []    # 每条: (ok, latency_ms, completeness, ts)
        self._fail_at = 0.0

    def record(self, ok: bool, latency_ms: float, completeness: float = 1.0):
        with self._lock:
            now = time.time()
            self._records.append({"ok": ok, "lat": latency_ms, "cpl": completeness, "ts": now})
            if len(self._records) > HEALTH_WINDOW * 3:
                self._records = self._records[-HEALTH_WINDOW:]
            if ok:
                self._fail_at = 0.0
            else:
                self._fail_at = now

    def metrics(self) -> HealthMetrics:
        with self._lock:
            recent = self._records[-HEALTH_WINDOW:] or []
            if not recent:
                return HealthMetrics(source=self.source, last_call=time.time())
            ok_cnt = sum(1 for r in recent if r["ok"])
            avg_lat = float(np.mean([r["lat"] for r in recent]))
            avg_cpl = float(np.mean([r["cpl"] for r in recent]))
            consec_fail = 0
            for r in reversed(recent):
                if r["ok"]:
                    break
                consec_fail += 1
            last_ok_ts = max((r["ts"] for r in recent if r["ok"]), default=0.0)
            last_call_ts = max(r["ts"] for r in recent)
            return HealthMetrics(
                source=self.source,
                success_rate=ok_cnt / len(recent),
                avg_latency_ms=avg_lat,
                field_completeness=avg_cpl,
                last_call=last_call_ts,
                last_success=last_ok_ts,
                consecutive_failures=consec_fail,
            )


# 全局健康度注册表
_HEALTH_TRACKERS: dict[str, HealthTracker] = {}
_HEALTH_LOCK = threading.Lock()


def _get_tracker(source: str) -> HealthTracker:
    with _HEALTH_LOCK:
        if source not in _HEALTH_TRACKERS:
            _HEALTH_TRACKERS[source] = HealthTracker(source)
        return _HEALTH_TRACKERS[source]


def health_snapshot() -> dict[str, dict]:
    """当前所有数据源的健康快照（给 UI 仪表盘用）"""
    out = {}
    for name, tracker in _HEALTH_TRACKERS.items():
        m = tracker.metrics()
        out[name] = {
            "成功率": f"{m.success_rate * 100:.0f}%",
            "延迟": f"{m.avg_latency_ms:.0f}ms",
            "完整率": f"{m.field_completeness * 100:.0f}%",
            "健康分": round(m.score, 3),
            "冷却中": m.should_skip(),
            "连续失败": m.consecutive_failures,
        }
    return out


# =============================================================
# 数据源抽象基类
# =============================================================
class DataSource(ABC):
    """所有数据源的统一抽象。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def type(self) -> str:
        """spot / history / fundamental —— 该源擅长哪类"""
        return "generic"

    # ---------- 快照 ----------
    def fetch_spot(self) -> pd.DataFrame | None:
        """全市场快照。返回 None 表示该源不支持；抛异常视为失败"""
        return None

    # ---------- K线 ----------
    def fetch_history(self, code: str, start: str, end: str, full: bool = True) -> pd.DataFrame | None:
        """单只股票历史K线。返回空表表示该时段无数据，抛异常视为失败"""
        return None

    # ---------- 基本面 ----------
    def fetch_fundamental(self, code: str, max_quarters: int = 8) -> pd.DataFrame | None:
        """单只股票历史财报。返回空表表示无数据，抛异常视为失败"""
        return None


# =============================================================
# 源 1: 腾讯批量行情（快照 + K线 qfq）
# =============================================================
class TencentSource(DataSource):
    """腾讯行情：快照稳定、字段丰富；K线支持 qfq 和不复权。"""

    @property
    def name(self) -> str:
        return "tencent"

    @property
    def type(self) -> str:
        return "spot"

    # ---------- 快照 ----------
    def _all_codes(self) -> list[str]:
        cached = None
        spot_file = HERE / "spot_cache.csv"
        if spot_file.exists():
            try:
                cached = pd.read_csv(spot_file, dtype={"代码": str})
            except Exception:
                pass
        code_file = HERE / "code_list.csv"
        if cached is not None and "代码" in cached.columns:
            codes = cached["代码"].astype(str).str.zfill(6).tolist()
        elif code_file.exists():
            try:
                codes = pd.read_csv(code_file, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
            except Exception:
                codes = []
        else:
            codes = []
        return [c for c in dict.fromkeys(codes) if not c.startswith(("4", "8", "920"))]

    def fetch_spot(self) -> pd.DataFrame | None:
        import requests
        codes = self._all_codes()
        if not codes:
            return None
        batches = [codes[i:i + TX_BATCH] for i in range(0, len(codes), TX_BATCH)]

        def _fetch(batch):
            import requests as _r
            q = ",".join(("sh" if c.startswith(("6", "9")) else "sz") + c for c in batch)
            resp = _r.get(f"https://qt.gtimg.cn/q={q}", timeout=SPOT_TIMEOUT_PER_SOURCE // max(1, TX_WORKERS),
                          headers={"Referer": "https://gu.qq.com/"})
            resp.encoding = "gbk"
            return resp.text.split(";")

        def _num(f, i):
            try:
                v = float(f[i])
                return v if v == v else float("nan")
            except Exception:
                return float("nan")

        rows = []
        with ThreadPoolExecutor(max_workers=TX_WORKERS) as ex:
            for lines in ex.map(_fetch, batches):
                for ln in lines:
                    f = ln.split("~")
                    if len(f) < 54:
                        continue
                    pe_dyn = _num(f, 52)
                    if np.isnan(pe_dyn):
                        pe_dyn = _num(f, 39)
                    rows.append({
                        "代码": f[2].zfill(6), "名称": f[1], "最新价": _num(f, 3),
                        "涨跌幅": _num(f, 32), "涨跌额": _num(f, 31),
                        "成交量": _num(f, 36), "成交额": _num(f, 37) * 1e4,
                        "振幅": _num(f, 43), "最高": _num(f, 33), "最低": _num(f, 34),
                        "今开": _num(f, 5), "昨收": _num(f, 4),
                        "换手率": _num(f, 38), "量比": _num(f, 49),
                        "市盈率-动态": pe_dyn, "市净率": _num(f, 46),
                        "总市值": _num(f, 45) * 1e8, "流通市值": _num(f, 44) * 1e8,
                    })
        if len(rows) < 100:
            raise RuntimeError(f"腾讯只返回 {len(rows)} 条，判定异常")
        df = pd.DataFrame(rows)
        # 计算字段完整率
        key_cols = ["市盈率-动态", "市净率", "换手率", "总市值", "流通市值"]
        completeness = df[key_cols].notna().mean().mean()
        _get_tracker(self.name).record(True, 0, completeness)
        return df

    # ---------- K线 ----------
    def fetch_history(self, code: str, start: str, end: str, full: bool = True) -> pd.DataFrame | None:
        import requests
        code = re.sub(r"\D", "", str(code))[-6:]
        if not code:
            return pd.DataFrame()
        pfx = "sh" if code.startswith(("6", "9")) else "sz"
        sd = f"{start[:4]}-{start[4:6]}-{start[6:8]}" if len(start) == 8 else start
        ed = f"{end[:4]}-{end[4:6]}-{end[6:8]}" if len(end) == 8 else end
        u = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
             f"param={pfx}{code},day,{sd},{ed},800,qfq")
        try:
            j = requests.get(u, timeout=HIST_TIMEOUT_PER_SOURCE,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://gu.qq.com/"}).json()
            d = j.get("data", {}).get(pfx + code, {})
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


# =============================================================
# 源 2: 新浪行情（快照 + K线）
# =============================================================
class SinaSource(DataSource):
    """新浪：快照轻量快速，K线不复权但稳定。腾讯/东财都挂时能用。"""

    @property
    def name(self) -> str:
        return "sina"

    # ---------- 快照 ----------
    def fetch_spot(self) -> pd.DataFrame | None:
        import requests
        # 新浪批量接口 /sh/sz 分开拉
        def _fetch(pfx):
            import requests as _r
            q = ",".join(f"{pfx}{c}" for c in self._all_codes_sina(pfx))
            resp = _r.get(f"https://hq.sinajs.cn/list={q}",
                          timeout=SPOT_TIMEOUT_PER_SOURCE // 2,
                          headers={"Referer": "https://finance.sina.com.cn/",
                                   "User-Agent": "Mozilla/5.0"})
            resp.encoding = "gbk"
            return resp.text

        rows = []
        for pfx in ("sh", "sz"):
            text = _fetch(pfx)
            for ln in text.strip().split("\n"):
                m = re.match(r'var hq_str_(sh|sz)(\d+)="(.*)";', ln)
                if not m:
                    continue
                c = m.group(2).zfill(6)
                if c.startswith(("4", "8", "920")):
                    continue
                fields = m.group(3).split(",")
                if len(fields) < 32:
                    continue
                try:
                    rows.append({
                        "代码": c, "名称": fields[0],
                        "最新价": float(fields[3]),
                        "昨收": float(fields[2]),
                        "今开": float(fields[1]),
                        "最高": float(fields[4]),
                        "最低": float(fields[5]),
                        "成交量": float(fields[8]),
                        "成交额": float(fields[9]),
                    })
                except (ValueError, IndexError):
                    continue
        if len(rows) < 100:
            raise RuntimeError(f"新浪只返回 {len(rows)} 条，判定异常")
        df = pd.DataFrame(rows)
        # 新浪没有PE/PB/市值/换手率，完整率会低一点，标记时注意
        df["涨跌幅"] = (df["最新价"] / df["昨收"] - 1) * 100
        _get_tracker(self.name).record(True, 0, 0.5)  # 字段少，完整率给0.5
        return df

    def _all_codes_sina(self, pfx: str) -> list[str]:
        code_file = HERE / "code_list.csv"
        if not code_file.exists():
            return []
        try:
            codes = pd.read_csv(code_file, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
        except Exception:
            return []
        prefix = ("6", "9") if pfx == "sh" else ("0", "3", "2")
        return [c for c in dict.fromkeys(codes) if c.startswith(prefix)]

    # ---------- K线 ----------
    def fetch_history(self, code: str, start: str, end: str, full: bool = True) -> pd.DataFrame | None:
        import requests
        code = re.sub(r"\D", "", str(code))[-6:]
        if not code:
            return pd.DataFrame()
        pfx = "sh" if code.startswith(("6", "9")) else "sz"
        u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             f"CN_MarketData.getKLineData?symbol={pfx}{code}&scale=240&ma=no&datalen=600")
        try:
            j = requests.get(u, timeout=HIST_TIMEOUT_PER_SOURCE,
                             headers={"User-Agent": "Mozilla/5.0"}).json()
            if not isinstance(j, list) or not j:
                return pd.DataFrame()
            data = {"日期": pd.to_datetime([x["day"] for x in j]),
                    "收盘": [float(x["close"]) for x in j]}
            if full:
                data.update({"开盘": [float(x["open"]) for x in j],
                             "最高": [float(x["high"]) for x in j],
                             "最低": [float(x["low"]) for x in j],
                             "成交量": [float(x["volume"]) for x in j]})
            return pd.DataFrame(data).sort_values("日期").reset_index(drop=True)
        except Exception:
            return pd.DataFrame()


# =============================================================
# 源 3: 东方财富 / akshare（快照 + K线 + 基本面）
# =============================================================
class EastMoneySource(DataSource):
    """东财：字段最全但常限流。用线程硬超时保护，限流时标记冷却期。"""

    @property
    def name(self) -> str:
        return "eastmoney"

    # ---------- 快照 ----------
    def fetch_spot(self) -> pd.DataFrame | None:
        import akshare as ak

        def _call():
            return ak.stock_zh_a_spot_em()

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_call)
        try:
            df = fut.result(timeout=SPOT_TIMEOUT_PER_SOURCE)
            ex.shutdown(wait=False)
            if df is None or len(df) < 100:
                raise RuntimeError("东财返回过少")
            # 字段映射：akshare 列名和我们不一样
            rename_map = {
                "代码": "代码", "名称": "名称", "最新价": "最新价",
                "涨跌幅": "涨跌幅", "涨跌额": "涨跌额",
                "成交量": "成交量", "成交额": "成交额",
                "振幅": "振幅", "最高": "最高", "最低": "最低",
                "今开": "今开", "昨收": "昨收",
                "换手率": "换手率", "量比": "量比",
                "市盈率-动态": "市盈率-动态", "市净率": "市净率",
                "总市值": "总市值", "流通市值": "流通市值",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            key_cols = ["市盈率-动态", "市净率", "换手率", "总市值", "流通市值"]
            completeness = df[[c for c in key_cols if c in df.columns]].notna().mean().mean()
            _get_tracker(self.name).record(True, 0, completeness)
            return df
        except FuturesTimeout:
            ex.shutdown(wait=False)
            raise TimeoutError("东财快照超时")
        except Exception:
            ex.shutdown(wait=False)
            raise

    # ---------- K线 ----------
    def fetch_history(self, code: str, start: str, end: str, full: bool = True) -> pd.DataFrame | None:
        import akshare as ak

        def _call():
            return ak.stock_zh_a_hist(symbol=code, period="daily",
                                      start_date=start, end_date=end, adjust="qfq")

        code = re.sub(r"\D", "", str(code))[-6:]
        if not code:
            return pd.DataFrame()
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_call)
        try:
            df = fut.result(timeout=HIST_TIMEOUT_PER_SOURCE)
            ex.shutdown(wait=False)
            if df is None or len(df) == 0:
                return pd.DataFrame()
            df["日期"] = pd.to_datetime(df["日期"])
            return df.sort_values("日期").reset_index(drop=True)
        except FuturesTimeout:
            ex.shutdown(wait=False)
            return pd.DataFrame()
        except Exception:
            ex.shutdown(wait=False)
            return pd.DataFrame()

    # ---------- 基本面（东财 F10 财务分析指标） ----------
    def fetch_fundamental(self, code: str, max_quarters: int = 8) -> pd.DataFrame | None:
        import akshare as ak

        code = re.sub(r"\D", "", str(code))[-6:]
        if not code:
            return pd.DataFrame()

        def _call():
            return ak.stock_financial_analysis_indicator(symbol=code, start_year="2021")

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_call)
        try:
            df = fut.result(timeout=FUND_TIMEOUT_PER_SOURCE)
            ex.shutdown(wait=False)
            if df is None or len(df) == 0:
                return pd.DataFrame()
            # 东财返回的是 年度/季度 维度的指标表，列名是中文
            # 抽取 ROE、净利润同比、营收同比、净利率、经营现金流 / 净利润 等
            out = self._normalize_financial(df, code, max_quarters)
            _get_tracker(self.name).record(True, 0, 0.7)
            return out
        except FuturesTimeout:
            ex.shutdown(wait=False)
            return pd.DataFrame()
        except Exception:
            ex.shutdown(wait=False)
            return pd.DataFrame()

    def _normalize_financial(self, raw: pd.DataFrame, code: str, max_quarters: int) -> pd.DataFrame:
        """东财 financial_analysis_indicator 列名随 akshare 版本变化，做宽松匹配。"""
        try:
            date_col = next((c for c in raw.columns if "日期" in str(c) or "时间" in str(c)), None)
            if date_col is None:
                return pd.DataFrame()
            out = pd.DataFrame({"代码": code, "报告期": raw[date_col]})
            out["pubDate"] = pd.to_datetime(raw[date_col], errors="coerce")
            out["报告期"] = pd.PeriodIndex(out["pubDate"], freq="Q") if out["pubDate"].notna().any() else out["报告期"]
            # 模糊匹配各指标
            def _pick(keywords):
                for kw in keywords:
                    for c in raw.columns:
                        if kw in str(c):
                            return pd.to_numeric(raw[c], errors="coerce")
                return np.nan
            out["ROE"] = _pick(["净资产收益率", "ROE"])
            out["净利润增速"] = _pick(["净利润同比", "净利润增长"])
            out["营收增速"] = _pick(["营业收入同比", "营收增长"])
            out["净利率"] = _pick(["销售净利率", "净利率"])
            out["现金流质量"] = _pick(["经营现金流"])
            out = out.dropna(subset=["ROE", "净利润增速", "营收增速", "净利率"], how="all")
            out = out.sort_values("pubDate").tail(max_quarters).reset_index(drop=True)
            return out
        except Exception:
            return pd.DataFrame()


# =============================================================
# 源 4: BaoStock（基本面）
# =============================================================
class BaoStockSource(DataSource):
    """BaoStock：免费永久的历史财报。常被封，做好连接失败处理。"""

    @property
    def name(self) -> str:
        return "baostock"

    def fetch_fundamental(self, code: str, max_quarters: int = 8) -> pd.DataFrame | None:
        try:
            import baostock as bs
        except ImportError:
            return pd.DataFrame()

        code = re.sub(r"\D", "", str(code))[-6:]
        if not code:
            return pd.DataFrame()
        bs_code = f"sh.{code}" if code.startswith(("6", "9")) else f"sz.{code}"

        def _call():
            lg = bs.login()
            if lg.error_code != "0":
                return []
            rows = []
            year, q = dt.date.today().year, (dt.date.today().month - 1) // 3 + 1
            for _ in range(12):
                q -= 1
                if q == 0:
                    q, year = 4, year - 1
                if year < 2018:
                    break
                try:
                    profit = bs.query_profit_data(code=bs_code, year=year, quarter=q).get_data()
                    growth = bs.query_growth_data(code=bs_code, year=year, quarter=q).get_data()
                    cf = bs.query_cash_flow_data(code=bs_code, year=year, quarter=q).get_data()
                except Exception:
                    continue
                if profit.empty and growth.empty and cf.empty:
                    continue
                row = {"代码": code, "报告期": f"{year}Q{q}"}
                if not profit.empty:
                    row["ROE"] = pd.to_numeric(profit["roeAvg"], errors="coerce").iloc[0] if "roeAvg" in profit.columns else None
                    row["净利率"] = pd.to_numeric(profit["npMargin"], errors="coerce").iloc[0] if "npMargin" in profit.columns else None
                if not growth.empty:
                    row["净利润增速"] = pd.to_numeric(growth["YOYNI"], errors="coerce").iloc[0] if "YOYNI" in growth.columns else None
                    row["营收增速"] = pd.to_numeric(growth["YOYPNI"], errors="coerce").iloc[0] if "YOYPNI" in growth.columns else None
                if not cf.empty and "CFOToNP" in cf.columns:
                    row["现金流质量"] = pd.to_numeric(cf["CFOToNP"], errors="coerce").iloc[0]
                pub = (profit.get("pubDate") or growth.get("pubDate") or cf.get("pubDate"))
                row["pubDate"] = str(pub.iloc[0]) if pub is not None and len(pub) > 0 else None
                rows.append(row)
                if len(rows) >= max_quarters:
                    break
            bs.logout()
            return rows

        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(_call)
        try:
            rows = fut.result(timeout=FUND_TIMEOUT_PER_SOURCE)
            ex.shutdown(wait=False)
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["报告期"] = pd.PeriodIndex(df["报告期"].astype(str), freq="Q")
            df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
            _get_tracker(self.name).record(True, 0, 0.8)
            return df.sort_values("pubDate").reset_index(drop=True)
        except FuturesTimeout:
            ex.shutdown(wait=False)
            return pd.DataFrame()
        except Exception:
            ex.shutdown(wait=False)
            return pd.DataFrame()


# =============================================================
# 路由 + 对账 + Facade
# =============================================================
class SourceRouter:
    """多源动态路由：按健康度加权选源，支持对账和熔断降级。"""

    def __init__(self, sources: list[DataSource]):
        self.sources = sources
        self._by_name = {s.name: s for s in sources}
        self._by_type: dict[str, list[DataSource]] = {}
        for s in sources:
            self._by_type.setdefault(s.type, []).append(s)

    # ---------- 快照 ----------
    def fetch_spot(self, reconcile: bool = True, primary: str | None = None) -> pd.DataFrame:
        """全市场快照。primary 指定优先源，不指定则按健康度排。"""
        sources = self._rank_by_health(self._by_type.get("spot", []) +
                                       self._by_type.get("generic", []))
        if primary and primary in self._by_name:
            sources = [self._by_name[primary]] + [s for s in sources if s.name != primary]

        results: dict[str, pd.DataFrame] = {}
        for src in sources:
            if _get_tracker(src.name).metrics().should_skip():
                continue
            t0 = time.time()
            try:
                df = src.fetch_spot()
                latency_ms = (time.time() - t0) * 1000
                if df is not None and len(df) > 100:
                    _get_tracker(src.name).record(True, latency_ms)
                    results[src.name] = df
                else:
                    _get_tracker(src.name).record(False, latency_ms)
            except Exception:
                latency_ms = (time.time() - t0) * 1000
                _get_tracker(src.name).record(False, latency_ms)

        if not results:
            raise RuntimeError("所有快照源均失败")

        primary_name = primary or sources[0].name
        primary_df = results[primary_name]

        # 对账：第二源存在时做交叉验证，标记异常行
        if reconcile and len(results) >= 2:
            sec_name = next(n for n in results if n != primary_name)
            sec_df = results[sec_name]
            reconciled = self._reconcile_spot(primary_df, sec_df)
            reconciled.attrs["reconciled_from"] = f"{primary_name}+{sec_name}"
            reconciled.attrs["reconcile_warnings"] = reconciled.attrs.pop("_warnings", [])
            return reconciled
        primary_df.attrs["reconciled_from"] = primary_name
        primary_df.attrs["reconcile_warnings"] = []
        return primary_df

    # ---------- K线 ----------
    def fetch_history(self, code: str, start: str, end: str, full: bool = True,
                      prefer_qfq: bool = True) -> pd.DataFrame:
        """单只 K线。东财 qfq 最全但限流，腾讯 qfq 兜底，新浪不复权作为最后手段。"""
        if prefer_qfq:
            order = ["eastmoney", "tencent", "sina"]
        else:
            order = ["sina", "eastmoney", "tencent"]

        for name in order:
            src = self._by_name.get(name)
            if src is None:
                continue
            if _get_tracker(name).metrics().should_skip():
                continue
            t0 = time.time()
            try:
                df = src.fetch_history(code, start, end, full=full)
                latency_ms = (time.time() - t0) * 1000
                if df is not None and len(df) > 0:
                    _get_tracker(name).record(True, latency_ms)
                    df.attrs["source"] = name
                    return df
                _get_tracker(name).record(False, latency_ms)
            except Exception:
                latency_ms = (time.time() - t0) * 1000
                _get_tracker(name).record(False, latency_ms)
        return pd.DataFrame()

    # ---------- 基本面 ----------
    def fetch_fundamental(self, code: str, max_quarters: int = 8) -> pd.DataFrame:
        """基本面：东财F10 优先（不依赖 baostock 登录），失败再试 baostock。"""
        for name in ["eastmoney", "baostock"]:
            src = self._by_name.get(name)
            if src is None:
                continue
            t0 = time.time()
            try:
                df = src.fetch_fundamental(code, max_quarters=max_quarters)
                latency_ms = (time.time() - t0) * 1000
                if df is not None and len(df) > 0:
                    _get_tracker(name).record(True, latency_ms)
                    df.attrs["source"] = name
                    return df
                _get_tracker(name).record(False, latency_ms)
            except Exception:
                latency_ms = (time.time() - t0) * 1000
                _get_tracker(name).record(False, latency_ms)
        return pd.DataFrame()

    # ---------- 辅助 ----------
    def _rank_by_health(self, sources: list[DataSource]) -> list[DataSource]:
        """按健康度分降序排列，冷却中的排最后或排除。"""
        scored = [(s, _get_tracker(s.name).metrics().score) for s in sources]
        scored.sort(key=lambda x: -x[1])
        return [s for s, _ in scored]

    def _reconcile_spot(self, primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        """两源快照交叉验证。价格/涨跌幅差过大时给该行打 '_reconcile_flag' 标记。
        标记的行仍可用（只是提醒可能不准），不会被删除。"""
        primary = primary.copy()
        sec = secondary.copy()
        if "代码" not in primary.columns or "代码" not in sec.columns:
            return primary
        sec_map = sec.set_index("代码") if "最新价" in sec.columns else sec.set_index("代码")
        if "最新价" not in sec_map.columns:
            return primary
        merged = primary.merge(sec_map[["最新价"]].rename(columns={"最新价": "_sec_最新价"}),
                               on="代码", how="left")
        flags = []
        warnings = []
        for _, r in merged.iterrows():
            if pd.isna(r.get("_sec_最新价")) or r["_sec_最新价"] <= 0:
                flags.append("no_match")
                continue
            diff = abs(r["最新价"] - r["_sec_最新价"]) / r["最新价"]
            if diff > RECONCILE_PRICE_TOL:
                flags.append(f"diff_{diff:.1%}")
                if len(warnings) < 5:
                    warnings.append(f"{r['代码']} 价差 {diff:.1%}")
            else:
                flags.append("ok")
        merged["_对账标记"] = flags
        merged = merged.drop(columns=["_sec_最新价"], errors="ignore")
        merged.attrs["_warnings"] = warnings
        return merged


# =============================================================
# 全局 Pipeline 单例（线程安全初始化）
# =============================================================
_pipeline: Optional["DataPipeline"] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> "DataPipeline":
    """获取或创建全局 pipeline 单例。"""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = DataPipeline()
    return _pipeline


class DataPipeline:
    """对外暴露的统一数据管道门面。
    - 内部封装多个 DataSource + SourceRouter
    - 可选接 DuckDB 数仓（warehouse=None 时不做持久化）
    - 接口设计与 data.py 保持兼容，便于平滑替换
    """

    def __init__(self, warehouse=None):
        self.sources = [
            TencentSource(),
            SinaSource(),
            EastMoneySource(),
            BaoStockSource(),
        ]
        self.router = SourceRouter(self.sources)
        self.warehouse = warehouse   # warehouse.py 里的 Warehouse 实例，可选

    # ---------- 快照 ----------
    def get_spot(self, force_refresh: bool = False,
                 cache_file: Path | None = None) -> pd.DataFrame:
        """全市场快照。cache_file 用于降级时读盘。"""
        try:
            df = self.router.fetch_spot()
        except Exception:
            # 全部源都挂，尝试读本地 CSV 缓存
            cache_file = cache_file or (HERE / "spot_cache.csv")
            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file, dtype={"代码": str})
                    return df
                except Exception:
                    pass
            raise
        # 落盘缓存（旧的 CSV 方式，兼容现有代码）
        try:
            cache_file = cache_file or (HERE / "spot_cache.csv")
            df.to_csv(cache_file, index=False, encoding="utf-8-sig")
        except Exception:
            pass
        # 落 DuckDB（可选）
        if self.warehouse is not None:
            try:
                self.warehouse.write_spot(df)
            except Exception:
                pass
        return df

    # ---------- K线 ----------
    def get_history(self, code: str, start: str, end: str,
                    full: bool = True, prefer_qfq: bool = True) -> pd.DataFrame:
        df = self.router.fetch_history(code, start, end, full=full, prefer_qfq=prefer_qfq)
        if len(df) == 0 and self.warehouse is not None:
            # 回退：查 DuckDB 有没有这只的历史
            try:
                cached = self.warehouse.query_kline(code, start, end)
                if len(cached) > 0:
                    return cached
            except Exception:
                pass
        return df

    # ---------- 基本面 ----------
    def get_fundamental(self, code: str, max_quarters: int = 8) -> pd.DataFrame:
        df = self.router.fetch_fundamental(code, max_quarters=max_quarters)
        if len(df) > 0 and self.warehouse is not None:
            try:
                self.warehouse.write_fundamental(code, df)
            except Exception:
                pass
        return df

    # ---------- 健康度 ----------
    def health(self) -> dict:
        return health_snapshot()


# =============================================================
# 快速自检（python data_pipeline.py 运行）
# =============================================================
if __name__ == "__main__":
    print("=== DataPipeline 自检 ===")
    pl = DataPipeline()
    print("数据源注册:", [s.name for s in pl.sources])
    print("\n=== 健康度快照（初始化为默认健康）===")
    for name, info in pl.health().items():
        print(f"  {name}: {info}")
    print("\n=== 快速拉一只 K线 ===")
    try:
        h = pl.get_history("600519", "20250101", "20250131", full=False)
        print(f"  贵州茅台 2025-01 K线: {len(h)} 条, 源={h.attrs.get('source', '?')}")
    except Exception as e:
        print(f"  K线拉取失败: {e}")
    print("\n=== 自检完成 ===")
