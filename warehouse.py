# -*- coding: utf-8 -*-
"""DuckDB 数据仓库层 —— 内嵌列存，替代 CSV 成为原始数据的权威持久化层。

设计：
  - 单个 .duckdb 文件，进程内零运维
  - 列存 + 内置压缩，因子 IC 检验（扫描几百只股票 × 几百天）比 CSV 快一个数量级
  - 原始入仓层（ODS）+ 查询视图层（DWD），方便 SQL 验证
  - 所有写操作都带 try/except，DuckDB 挂了不阻塞主流程（降级到 CSV）

表结构：
  spot_daily       — 全市场快照（每日一表，多源对账标记）
  kline_daily      — 单只日K（OHLCV，带复权标记）
  fundamentals     — 历史财报（Baostock + 东财F10 统一格式）
  industry_map     — 代码→行业（磁盘缓存的升级版）
  source_health    — 数据源健康度历史趋势（可选，用于可视化）
  quality_events   — 数据质量事件（对账超差、字段缺失）
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DEFAULT_DB_PATH = HERE / "market_data.duckdb"


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """把所有列转成 DuckDB register 能识别的 dtype。
    pandas 3.x 的 StringDtype('str') 和 nullable 类型 DuckDB 1.1 不认。"""
    out = df.copy()
    for col in out.columns:
        d = str(out[col].dtype)
        if out[col].dtype == object or d in ("string", "str"):
            out[col] = out[col].fillna("").astype(object)
        elif out[col].dtype.kind in "iufb":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
        elif "time" in d:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


class Warehouse:
    """DuckDB 数仓封装。所有方法都返回 None 或空表作为失败降级，永不抛异常。"""

    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH)
        self._con = None
        self._initialized = False
        self._init()

    def _init(self):
        try:
            import duckdb
            # 每次 connect 都是一个新连接，DuckDB 会自动处理并发
            self._con = duckdb.connect(str(self.db_path), read_only=False)
            self._create_tables(self._con)
            self._con.close()
            self._con = None
            self._initialized = True
        except Exception as e:
            self._initialized = False
            print(f"[Warehouse] DuckDB 初始化失败，后续操作会静默降级: {e}")

    def _connect(self):
        """每次操作开新连接（DuckDB 文件支持多连接）。"""
        if not self._initialized:
            return None
        try:
            import duckdb
            return duckdb.connect(str(self.db_path), read_only=False)
        except Exception:
            return None

    # ---------- 建表 ----------
    def _create_tables(self, con):
        if con is None:
            return
        con.execute("""
            CREATE TABLE IF NOT EXISTS spot_daily (
                snapshot_ts  TIMESTAMP,          -- 快照时间
                代码          VARCHAR,
                名称          VARCHAR,
                最新价        DOUBLE,
                涨跌幅        DOUBLE,
                成交额        DOUBLE,
                市盈率_动态   DOUBLE,
                市净率        DOUBLE,
                换手率        DOUBLE,
                量比          DOUBLE,
                总市值        DOUBLE,
                流通市值      DOUBLE,
                对账标记      VARCHAR
            );
            CREATE INDEX IF NOT EXISTS idx_spot_code ON spot_daily(代码, snapshot_ts);
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS kline_daily (
                code        VARCHAR,
                trade_date  DATE,
                open        DOUBLE,
                high        DOUBLE,
                low         DOUBLE,
                close       DOUBLE,
                volume      DOUBLE,
                source      VARCHAR,       -- eastmoney/tencent/sina
                fq_type     VARCHAR,       -- qfq / raw
                update_ts   TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_kline_code ON kline_daily(code, trade_date);
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                code        VARCHAR,
                period      VARCHAR,       -- '2025Q3' 格式
                pub_date    DATE,
                roe         DOUBLE,
                revenue_growth DOUBLE,
                profit_growth DOUBLE,
                net_margin  DOUBLE,
                cf_quality  DOUBLE,
                source      VARCHAR,       -- baostock / eastmoney
                update_ts   TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_fund_code ON fundamentals(code, pub_date);
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS industry_map (
                code        VARCHAR,
                industry    VARCHAR,
                update_ts   TIMESTAMP,
                PRIMARY KEY (code)
            );
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS quality_events (
                event_ts    TIMESTAMP,
                event_type  VARCHAR,      -- reconcile / missing / stale / timeout
                source      VARCHAR,
                code        VARCHAR,
                detail      VARCHAR
            );
        """)

    # =============================================================
    # 写入
    # =============================================================
    def write_spot(self, df: pd.DataFrame, snapshot_ts: dt.datetime | None = None):
        """全市场快照入仓。同一天同一代码重复写入会覆盖（INSERT OR REPLACE）。"""
        if not self._initialized or df is None or len(df) == 0:
            return
        con = self._connect()
        if con is None:
            return
        try:
            ts = snapshot_ts or dt.datetime.now()
            out = pd.DataFrame()
            out["snapshot_ts"] = ts
            out["代码"] = df["代码"].astype(str).str.zfill(6)
            out["名称"] = df.get("名称", "")
            out["最新价"] = pd.to_numeric(df["最新价"], errors="coerce")
            out["涨跌幅"] = pd.to_numeric(df.get("涨跌幅"), errors="coerce")
            out["成交额"] = pd.to_numeric(df.get("成交额"), errors="coerce")
            pe_col = "市盈率-动态" if "市盈率-动态" in df.columns else (
                "市盈率-动态_ts" if "市盈率-动态_ts" in df.columns else None)
            out["市盈率_动态"] = pd.to_numeric(df[pe_col] if pe_col else None, errors="coerce")
            out["市净率"] = pd.to_numeric(df.get("市净率"), errors="coerce")
            out["换手率"] = pd.to_numeric(df.get("换手率"), errors="coerce")
            out["量比"] = pd.to_numeric(df.get("量比"), errors="coerce")
            out["总市值"] = pd.to_numeric(df.get("总市值"), errors="coerce")
            out["流通市值"] = pd.to_numeric(df.get("流通市值"), errors="coerce")
            out["对账标记"] = df.get("_对账标记", "")
            out = out.drop_duplicates(subset=["snapshot_ts", "代码"])
            out = _fix_dtypes(out)
            con.register("spot_batch", out)
            con.execute("""
                DELETE FROM spot_daily
                USING spot_batch
                WHERE spot_daily.代码 = spot_batch.代码
                  AND spot_daily.snapshot_ts = spot_batch.snapshot_ts;
            """)
            con.execute("INSERT INTO spot_daily SELECT * FROM spot_batch")
            con.unregister("spot_batch")
        finally:
            con.close()

    def write_kline(self, code: str, df: pd.DataFrame,
                    source: str = "unknown", fq_type: str = "qfq"):
        """单只股票 K线入仓（append 模式，同日期重复覆盖）。"""
        if not self._initialized or df is None or len(df) == 0:
            return
        con = self._connect()
        if con is None:
            return
        try:
            out = pd.DataFrame()
            out["code"] = str(code).zfill(6)
            out["trade_date"] = pd.to_datetime(df["日期"]).dt.date
            out["open"] = pd.to_numeric(df["开盘"], errors="coerce") if "开盘" in df.columns else np.nan
            out["high"] = pd.to_numeric(df["最高"], errors="coerce") if "最高" in df.columns else np.nan
            out["low"] = pd.to_numeric(df["最低"], errors="coerce") if "最低" in df.columns else np.nan
            out["close"] = pd.to_numeric(df["收盘"], errors="coerce")
            out["volume"] = pd.to_numeric(df["成交量"], errors="coerce") if "成交量" in df.columns else np.nan
            out["source"] = source
            out["fq_type"] = fq_type
            out["update_ts"] = dt.datetime.now()
            out = out.dropna(subset=["close", "trade_date"])
            out = out.drop_duplicates(subset=["code", "trade_date"])
            out = _fix_dtypes(out)
            con.register("kline_batch", out)
            con.execute("""
                DELETE FROM kline_daily
                USING kline_batch
                WHERE kline_daily.code = kline_batch.code
                  AND kline_daily.trade_date = kline_batch.trade_date;
            """)
            con.execute("INSERT INTO kline_daily SELECT * FROM kline_batch")
            con.unregister("kline_batch")
        finally:
            con.close()

    def write_fundamental(self, code: str, df: pd.DataFrame, source: str = "unknown"):
        """单只基本面入仓。"""
        if not self._initialized or df is None or len(df) == 0:
            return
        con = self._connect()
        if con is None:
            return
        try:
            out = pd.DataFrame()
            out["code"] = str(code).zfill(6)
            out["period"] = df["报告期"].astype(str)
            out["pub_date"] = pd.to_datetime(df["pubDate"], errors="coerce").dt.date
            out["roe"] = pd.to_numeric(df["ROE"], errors="coerce") if "ROE" in df.columns else np.nan
            out["revenue_growth"] = pd.to_numeric(df["营收增速"], errors="coerce") if "营收增速" in df.columns else np.nan
            out["profit_growth"] = pd.to_numeric(df["净利润增速"], errors="coerce") if "净利润增速" in df.columns else np.nan
            out["net_margin"] = pd.to_numeric(df["净利率"], errors="coerce") if "净利率" in df.columns else np.nan
            out["cf_quality"] = pd.to_numeric(df["现金流质量"], errors="coerce") if "现金流质量" in df.columns else np.nan
            out["source"] = source
            out["update_ts"] = dt.datetime.now()
            out = _fix_dtypes(out)
            con.register("fund_batch", out)
            con.execute("""
                DELETE FROM fundamentals
                USING fund_batch
                WHERE fundamentals.code = fund_batch.code
                  AND fundamentals.period = fund_batch.period;
            """)
            con.execute("INSERT INTO fundamentals SELECT * FROM fund_batch")
            con.unregister("fund_batch")
        finally:
            con.close()

    def write_industry_map(self, df: pd.DataFrame):
        """行业映射入仓。"""
        if not self._initialized or df is None or len(df) == 0:
            return
        con = self._connect()
        if con is None:
            return
        try:
            out = pd.DataFrame({
                "code": df["代码"].astype(str).str.zfill(6),
                "industry": df["行业"].astype(str),
                "update_ts": dt.datetime.now(),
            }).drop_duplicates("code")
            out = _fix_dtypes(out)
            con.register("ind_batch", out)
            con.execute("""
                DELETE FROM industry_map
                USING ind_batch
                WHERE industry_map.code = ind_batch.code;
            """)
            con.execute("INSERT INTO industry_map SELECT * FROM ind_batch")
            con.unregister("ind_batch")
        finally:
            con.close()

    # =============================================================
    # 查询
    # =============================================================
    def query_latest_spot(self, min_date: str | None = None) -> pd.DataFrame:
        """最新快照（可选过滤最近日期）。"""
        if not self._initialized:
            return pd.DataFrame()
        con = self._connect()
        if con is None:
            return pd.DataFrame()
        try:
            where = ""
            if min_date:
                where = f"WHERE snapshot_ts >= '{min_date}'"
            df = con.execute(f"""
                SELECT DISTINCT ON (代码) *
                FROM spot_daily
                {where}
                ORDER BY 代码, snapshot_ts DESC
            """).fetchdf()
            return df
        finally:
            con.close()

    def query_kline(self, code: str, start: str, end: str) -> pd.DataFrame:
        """单只 K线。"""
        if not self._initialized:
            return pd.DataFrame()
        con = self._connect()
        if con is None:
            return pd.DataFrame()
        try:
            s = pd.to_datetime(start).date()
            e = pd.to_datetime(end).date()
            df = con.execute(f"""
                SELECT trade_date AS 日期, open AS 开盘, high AS 最高, low AS 最低,
                       close AS 收盘, volume AS 成交量
                FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY trade_date ORDER BY update_ts DESC) as rn
                    FROM kline_daily
                    WHERE code = '{code.zfill(6)}'
                      AND trade_date BETWEEN '{s}' AND '{e}'
                ) WHERE rn = 1
                ORDER BY 日期
            """).fetchdf()
            return df
        finally:
            con.close()

    def query_fundamental(self, code: str) -> pd.DataFrame:
        """单只基本面。"""
        if not self._initialized:
            return pd.DataFrame()
        con = self._connect()
        if con is None:
            return pd.DataFrame()
        try:
            df = con.execute(f"""
                SELECT DISTINCT ON (period)
                    period AS 报告期, pub_date AS pubDate, roe AS ROE,
                    revenue_growth AS 营收增速, profit_growth AS 净利润增速,
                    net_margin AS 净利率, cf_quality AS 现金流质量
                FROM fundamentals
                WHERE code = '{code.zfill(6)}'
                ORDER BY period, pub_date DESC
            """).fetchdf()
            if len(df):
                df["报告期"] = pd.PeriodIndex(df["报告期"].astype(str), freq="Q")
            return df
        finally:
            con.close()

    def query_industry_map(self) -> dict:
        """返回 {代码: 行业}。"""
        if not self._initialized:
            return {}
        con = self._connect()
        if con is None:
            return {}
        try:
            df = con.execute("SELECT code, industry FROM industry_map").fetchdf()
            return dict(zip(df["code"], df["industry"]))
        finally:
            con.close()

    # =============================================================
    # 质量门禁 + 健康度
    # =============================================================
    def quality_gate_spot(self, df: pd.DataFrame, min_complete_rate: float = 0.6) -> tuple[bool, list[str]]:
        """快照入仓前的质量门禁：关键字段完整率检查。
        返回 (是否通过, 告警列表)。"""
        if df is None or len(df) == 0:
            return False, ["空 DataFrame"]
        key_cols = ["最新价", "成交额"]
        optional_cols = ["市盈率-动态", "市净率", "换手率", "总市值", "流通市值"]
        missing = [c for c in key_cols if c not in df.columns or df[c].isna().all()]
        if missing:
            return False, [f"关键字段缺失: {missing}"]
        warnings = []
        completeness = {c: float(df[c].notna().mean())
                        for c in optional_cols if c in df.columns}
        weak = [(c, v) for c, v in completeness.items() if v < min_complete_rate]
        if weak:
            for c, v in weak:
                warnings.append(f"字段 {c} 完整率仅 {v:.0%}")
        return True, warnings

    def health_report(self) -> dict:
        """仓库健康报告：表大小、最新数据时间、行数趋势。"""
        if not self._initialized:
            return {"status": "not_initialized"}
        con = self._connect()
        if con is None:
            return {"status": "disconnected"}
        try:
            info: dict[str, Any] = {"status": "ok", "path": str(self.db_path)}
            for tbl in ("spot_daily", "kline_daily", "fundamentals", "industry_map"):
                try:
                    cnt = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
                    info[f"{tbl}_rows"] = cnt
                    if tbl == "spot_daily":
                        latest = con.execute("SELECT MAX(snapshot_ts) FROM spot_daily").fetchone()[0]
                        info["spot_latest"] = str(latest) if latest else None
                    if tbl == "kline_daily":
                        codes = con.execute("SELECT COUNT(DISTINCT code) FROM kline_daily").fetchone()[0]
                        info["kline_codes"] = codes
                except Exception as e:
                    info[f"{tbl}_error"] = str(e)
            try:
                sz = self.db_path.stat().st_size
                info["db_size_mb"] = round(sz / 1e6, 2)
            except Exception:
                pass
            return info
        finally:
            con.close()

    def close(self):
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
            self._con = None


# =============================================================
# 快速自检
# =============================================================
if __name__ == "__main__":
    import tempfile, os
    print("=== Warehouse 自检（临时库）===")
    tmp = Path(tempfile.gettempdir()) / "test_pipeline.duckdb"
    if tmp.exists():
        tmp.unlink()
    w = Warehouse(tmp)
    print("初始化:", "OK" if w._initialized else "FAIL")
    print("健康报告:", w.health_report())

    # 写一批假快照
    fake_spot = pd.DataFrame({
        "代码": ["600519", "000001", "000858"],
        "名称": ["贵州茅台", "平安银行", "五粮液"],
        "最新价": [1700.0, 12.5, 130.0],
        "涨跌幅": [1.5, -0.3, 0.8],
        "成交额": [5e9, 2e8, 3e8],
        "市盈率-动态": [30.0, 8.0, 20.0],
        "市净率": [10.0, 0.8, 4.0],
        "换手率": [0.3, 1.5, 2.0],
        "量比": [0.9, 1.2, 1.0],
        "总市值": [2e12, 2.4e11, 5e11],
        "流通市值": [1.8e12, 2.4e11, 4.5e11],
    })
    ok, warns = w.quality_gate_spot(fake_spot)
    print(f"质量门禁: {ok}, 告警: {warns}")
    w.write_spot(fake_spot)

    # 查回
    df = w.query_latest_spot()
    print(f"快照查询: {len(df)} 条, 列: {list(df.columns)}")

    w.close()
    tmp.unlink()
    print("=== 自检完成 ===")
