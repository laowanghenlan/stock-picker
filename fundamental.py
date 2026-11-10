# -*- coding: utf-8 -*-
"""
Baostock 历史财报数据层（免费、永久、无次数限制）。
用途：补全 akshare 的缺口——akshare 只能拿当前快照估值，
     baostock 能拿到历史季度财报的真实 ROE/净利润增速/营收增速/净利率，
     消除因子计算的前视偏差。

设计：磁盘 CSV 缓存（一个季度一更新，不频繁请求）
      → IC 检验每日只在因子值缺失时才请求
      → 接口挂了自动降级（因子值留 NaN，IC 检验自动跳过）
"""
from __future__ import annotations
import os, json, time, logging
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

HERE = Path(__file__).parent
CACHE_DIR = HERE / "fundamental_cache"
CACHE_DIR.mkdir(exist_ok=True)
FUNDAMENTAL_CSV = CACHE_DIR / "fundamental_{quarter}.csv"  # 按季度缓存
_LAST_LOGIN = 0  # 控制频率避免爆 baostock 服务器

logger = logging.getLogger("fundamental")

# baostock 代码格式转换：A 股 6 位代码 → "sh.600000" / "sz.000001"
def _to_bs_code(code: str) -> str | None:
    try:
        code = str(code).zfill(6)
        if code.startswith(("6", "9")):
            return f"sh.{code}"
        if code.startswith(("0", "3", "2")):
            return f"sz.{code}"
    except Exception:
        pass
    return None


# ========== 会话管理（避免高频 login/logout） ==========
def _bs_login() -> bool:
    """每 30 秒内只 login 一次，避免被 baostock 服务器限流"""
    global _LAST_LOGIN
    import baostock as bs
    now = time.time()
    if now - _LAST_LOGIN < 30:
        return True
    lg = bs.login()
    if lg.error_code == "0":
        _LAST_LOGIN = now
        return True
    logger.warning("baostock login 失败: %s", lg.error_msg)
    return False


def _bs_logout():
    import baostock as bs
    bs.logout()


# ========== 缓存命中检查 ==========
def _current_quarter() -> tuple[int, int]:
    """返回 (year, quarter)，例如 (2024, 3)"""
    today = date.today()
    q = (today.month - 1) // 3 + 1
    return today.year, q


def _cache_path(year: int, q: int) -> Path:
    return CACHE_DIR / f"fundamental_{year}Q{q}.csv"


def _cache_hit(year: int, q: int) -> pd.DataFrame | None:
    p = _cache_path(year, q)
    if p.exists():
        try:
            return pd.read_csv(p, dtype={"代码": str})
        except Exception:
            return None
    return None


def _cache_save(df: pd.DataFrame, year: int, q: int) -> None:
    try:
        df.to_csv(_cache_path(year, q), index=False)
    except Exception:
        pass


# ========== 单股历史财报缓存（当日有效：财报按季度发布，日内不可能变） ==========
_HIST_DIR = CACHE_DIR / "hist"
_HIST_DIR.mkdir(exist_ok=True)


def _hist_cache_path(code: str) -> Path:
    return _HIST_DIR / f"hist_{code}.csv"


def _hist_cache_read(code: str) -> pd.DataFrame | None:
    """读取单股历史财报缓存，仅当天内有效；不存在/过期/损坏返回 None"""
    p = _hist_cache_path(code)
    try:
        if p.exists() and date.fromtimestamp(p.stat().st_mtime) == date.today():
            df = pd.read_csv(p, dtype={"代码": str})
            if len(df) == 0:
                return None
            # 报告期还原成季度索引（注意：不能先转成日期，"2025Q4" 会被误读为 2025 年 4 月）
            df["报告期"] = pd.PeriodIndex(df["报告期"].astype(str).tolist(), freq="Q")
            df["pubDate"] = pd.to_datetime(df["pubDate"], errors="coerce")
            return df
    except Exception:
        pass
    return None


def _hist_cache_save(code: str, df: pd.DataFrame) -> None:
    try:
        df.to_csv(_hist_cache_path(code), index=False)
    except Exception:
        pass


# ========== 核心：拿一只股票最近 4 个季度财报 ==========
def fetch_fundamental(code: str, max_history_quarters: int = 8) -> pd.DataFrame:
    """
    从 baostock 拉一只股票的历史财报（最近 max_history_quarters 个季度）。
    返回 DataFrame 列：代码, 报告期, pubDate, ROE, 净利润增速, 净利率, 营收增速
    失败返回空 DataFrame。当日磁盘缓存命中时直接返回（财报日内不会变）。
    """
    code = str(code).zfill(6)
    cached = _hist_cache_read(code)
    if cached is not None:
        return cached.tail(max_history_quarters).reset_index(drop=True)
    bs_code = _to_bs_code(code)
    if not bs_code:
        return pd.DataFrame()
    try:
        import baostock as bs
    except ImportError:
        return pd.DataFrame()

    if not _bs_login():
        return pd.DataFrame()

    year, q = _current_quarter()
    rows = []
    # 往前最多扫 12 个季度（3 年），足够因子窗口用
    for _ in range(12):
        q -= 1
        if q == 0:
            q = 4
            year -= 1
        if year < 2018:
            break
        try:
            # 拿 ROE 和净利率
            rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=q)
            profit = rs_profit.get_data()
            rs_growth = bs.query_growth_data(code=bs_code, year=year, quarter=q)
            growth = rs_growth.get_data()
        except Exception:
            continue

        if profit.empty and growth.empty:
            continue

        row = {"代码": code, "报告期": f"{year}Q{q}"}
        # ROE/净利率：baostock 返回的已是小数比例（0.049 即 4.9%），不能再除以 100
        if not profit.empty:
            row["ROE"] = pd.to_numeric(profit["roeAvg"], errors="coerce").iloc[0] if "roeAvg" in profit.columns else None
            row["净利率"] = pd.to_numeric(profit["npMargin"], errors="coerce").iloc[0] if "npMargin" in profit.columns else None
            row["EPS_TTM"] = pd.to_numeric(profit["epsTTM"], errors="coerce").iloc[0] if "epsTTM" in profit.columns else None
        # 增速：同样已是小数比例
        if not growth.empty:
            row["净利润增速"] = pd.to_numeric(growth["YOYNI"], errors="coerce").iloc[0] if "YOYNI" in growth.columns else None
            row["营收增速"] = pd.to_numeric(growth["YOYPNI"], errors="coerce").iloc[0] if "YOYPNI" in growth.columns else None

        # pubDate（发布日期很重要——因子只能用发布日之后的行情，避免前视偏差）
        pub = profit.get("pubDate", growth.get("pubDate"))
        row["pubDate"] = str(pub.iloc[0]) if pub is not None and len(pub) > 0 else None
        rows.append(row)

        if len(rows) >= max_history_quarters:
            break

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    # 报告期统一转成季度索引（注意：不能先转成日期，"2025Q4" 会被误读为 2025 年 4 月）
    out["报告期"] = pd.PeriodIndex(out["报告期"].astype(str).tolist(), freq="Q")
    out["pubDate"] = pd.to_datetime(out["pubDate"], errors="coerce")
    out = out.sort_values("pubDate").reset_index(drop=True)
    _hist_cache_save(code, out)                     # 写当日缓存（空结果不缓存，避免污染）
    return out


# ========== 批量拉全市场（缓存驱动，不会频繁请求） ==========
def fetch_fundamental_batch(codes: list[str], year: int | None = None,
                            q: int | None = None) -> pd.DataFrame:
    """
    批量拿多只股票的基本面。优先读磁盘缓存，缓存 miss 才请求 baostock。
    单季度一次性拉下来保存，后续直接读。
    """
    if year is None or q is None:
        year, q = _current_quarter()

    # 先查缓存
    cached = _cache_hit(year, q)
    if cached is not None:
        existing = set(cached["代码"].astype(str))
        need = [c for c in codes if str(c).zfill(6) not in existing]
        if not need:
            return cached

    # 没缓存，或缓存不全 → 先建缓存
    all_codes = list({str(c).zfill(6) for c in codes})
    try:
        import baostock as bs
    except ImportError:
        return cached if cached is not None else pd.DataFrame()

    if not _bs_login():
        return cached if cached is not None else pd.DataFrame()

    results = []
    for i, code in enumerate(all_codes):
        bs_code = _to_bs_code(code)
        if not bs_code:
            continue
        try:
            rs_profit = bs.query_profit_data(code=bs_code, year=year, quarter=q)
            profit = rs_profit.get_data()
            rs_growth = bs.query_growth_data(code=bs_code, year=year, quarter=q)
            growth = rs_growth.get_data()
        except Exception:
            continue

        row = {"代码": code, "报告期": f"{year}Q{q}"}
        if not profit.empty:
            row["ROE"] = pd.to_numeric(profit["roeAvg"], errors="coerce").iloc[0] if "roeAvg" in profit.columns else None
            row["净利率"] = pd.to_numeric(profit["npMargin"], errors="coerce").iloc[0] if "npMargin" in profit.columns else None
        if not growth.empty:
            row["净利润增速"] = pd.to_numeric(growth["YOYNI"], errors="coerce").iloc[0] if "YOYNI" in growth.columns else None
            row["营收增速"] = pd.to_numeric(growth["YOYPNI"], errors="coerce").iloc[0] if "YOYPNI" in growth.columns else None
        results.append(row)

        if (i + 1) % 50 == 0:
            time.sleep(0.5)  # 礼貌请求

    if results:
        df_new = pd.DataFrame(results)
        # 合并到缓存
        if cached is not None:
            df_new = pd.concat([cached, df_new]).drop_duplicates("代码", keep="last").reset_index(drop=True)
        _cache_save(df_new, year, q)
        return df_new

    return cached if cached is not None else pd.DataFrame()


# ========== 快速测试入口 ==========
if __name__ == "__main__":
    print("=== 单票测试：sh.600000 ===")
    df = fetch_fundamental("600000")
    print(df)
    print("\n=== 批量测试：3 只 ===")
    df2 = fetch_fundamental_batch(["600000", "000001", "601318"])
    print(df2)
    _bs_logout()
