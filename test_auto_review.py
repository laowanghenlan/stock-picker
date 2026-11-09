# -*- coding: utf-8 -*-
"""AppTest 无头测试：验证 app.py 打开页面即自动复盘。

覆盖三种场景（akshare 全部打桩为假数据，不联网）：
  A. 推送记录.csv 不存在  → 安静显示"暂无历史推荐"，无异常
  B. 推送记录.csv 有数据  → 自动复盘板块出现，汇总数字与明细标签逐一核对
  C. 恢复真实推送记录.csv → 打桩无后续行情，标 ⏳持有中(当日新推)，不计入胜率
运行：python test_auto_review.py
"""
import datetime as dt
import shutil
from pathlib import Path
from unittest import mock

import akshare
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
APP = ROOT / "app.py"
CSV = ROOT / "推送记录.csv"
BAK = ROOT / "推送记录.csv.bak"

# ---------------- 假数据：5 只票覆盖全部 5 种结果标签 ----------------
REC_DATE = {"000001": "2026-10-26", "000002": "2026-10-26",
            "000003": "2026-10-27", "000004": "2026-10-27",
            "000005": dt.date.today().strftime("%Y-%m-%d")}
FAKE_CLOSES = {
    # 10.00 推荐，第1日触止损-5%，之后20日横在9.00 → 🛑+挽回(9.40-9.00)*100=40元
    "000001": [9.40, 9.30, 9.20, 9.10, 9.00] + [9.00] * 15,
    # 5.00 推荐，第2日触止盈+10% → 🎯
    "000002": [5.20, 5.56] + [5.45] * 18,
    # 4.00 推荐，20日未触发（区间3.80~4.40），到期收4.20 → 🏁到期盈利
    "000003": [4.04, 4.10, 3.95, 4.12] + [4.20] * 16,
    # 8.00 推荐，20日未触发（区间7.60~8.80），到期收7.80 → 🏁到期亏损
    "000004": [7.95, 7.90, 7.85, 7.88] + [7.80] * 16,
    # 今日推荐，无后续交易日 → ⏳持有中(当日新推)
    "000005": [],
}
SHARES = {"000001": 100, "000002": 200, "000003": 100, "000004": 100, "000005": 100}
PRICE0 = {"000001": 10.00, "000002": 5.00, "000003": 4.00, "000004": 8.00, "000005": 3.00}


def fake_hist(symbol=None, period=None, start_date=None, end_date=None, adjust=None):
    closes = FAKE_CLOSES.get(symbol, [])
    base = pd.Timestamp(REC_DATE.get(symbol, dt.date.today().strftime("%Y-%m-%d")))
    dates = pd.bdate_range(base + pd.Timedelta(days=1), periods=len(closes))
    return pd.DataFrame({"日期": dates.strftime("%Y-%m-%d"), "收盘": closes})


def fake_index(symbol=None, period=None, start_date=None, end_date=None):
    # 带日期列：既供大盘温度（收盘不低于MA60判回暖），也供复盘基准对比（恒定4000 → 基准涨跌为0）
    days = pd.bdate_range(end=dt.date.today(), periods=120)
    return pd.DataFrame({"日期": days.strftime("%Y-%m-%d"), "收盘": [4000.0] * 120})


def run_app() -> AppTest:
    st.cache_data.clear()                      # 防止跨场景缓存污染
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert len(at.exception) == 0, f"页面抛异常: {[e.value for e in at.exception]}"
    return at


def metric(at: AppTest, label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return m.value
    raise AssertionError(f"找不到指标 {label}，现有: {[m.label for m in at.metric]}")


def review_table(at: AppTest) -> pd.DataFrame:
    """页面上的 st.dataframe 里找出复盘明细表（含「结果」列的那张）"""
    for d in at.dataframe:
        if "结果" in d.value.columns:
            return d.value
    raise AssertionError(f"找不到复盘明细表，现有: {[list(d.value.columns) for d in at.dataframe]}")


def scenario_a():
    assert not CSV.exists()
    at = run_app()
    titles = [e.label for e in at.expander]
    assert any("历史推荐复盘" in t for t in titles), f"复盘板块未出现: {titles}"
    caps = [c.value for c in at.caption]
    assert "暂无历史推荐，有推荐后会自动复盘。" in caps, f"空态文案未出现: {caps}"
    print("场景A（无推送记录）：通过 —— 复盘板块出现，显示空态文案，无异常")


def scenario_b():
    rows = [{"日期": REC_DATE[c], "代码": c, "名称": f"测试{c[-1]}",
             "现价": PRICE0[c], "买入股数": SHARES[c], "大盘标签": "【回暖期】"}
            for c in ["000001", "000002", "000003", "000004", "000005"]]
    pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig")
    at = run_app()

    # 与 app.py 完全相同的口径手工重算期望值
    pnls = [round((9.40 / 10.00 - 1) * 100, 2), round((5.56 / 5.00 - 1) * 100, 2),
            round((4.20 / 4.00 - 1) * 100, 2), round((7.80 / 8.00 - 1) * 100, 2)]
    amounts = [round((9.40 - 10.00) * 100, 1), round((5.56 - 5.00) * 200, 1),
               round((4.20 - 4.00) * 100, 1), round((7.80 - 8.00) * 100, 1)]
    assert metric(at, "推荐总数") == "5 条"
    assert metric(at, "胜率") == "50%（2/4）", metric(at, "胜率")
    assert metric(at, "平均每笔盈亏") == f"{sum(pnls) / len(pnls):+.2f}%"
    assert metric(at, "累计盈亏") == f"{sum(amounts):+.0f} 元"

    tbl = review_table(at)
    assert len(tbl) == 5
    got = set(tbl["结果"])
    want = {"🎯止盈", "🛑止损保命", "🏁到期盈利", "🏁到期亏损", "⏳持有中(当日新推)"}
    assert got == want, f"结果标签不符: {got}"

    infos = [i.value for i in at.info]
    assert any("有赚有亏" in t and "止损纪律避免了约 40 元额外亏损" in t for t in infos), \
        f"自动结论不符: {infos}"
    assert any("同期沪深300平均" in t for t in infos), f"缺少基准对比结论: {infos}"
    print("场景B（5条假推荐）：通过 —— 汇总数字/明细标签/自动结论全部正确")


def scenario_c():
    shutil.copy2(BAK, CSV)                     # 恢复真实推送记录
    n_real = len(pd.read_csv(CSV))
    at = run_app()
    assert metric(at, "推荐总数") == f"{n_real} 条"
    assert metric(at, "胜率") == "—"
    infos = [i.value for i in at.info]
    assert any("持有期" in t for t in infos), f"结论不符: {infos}"
    tbl = review_table(at)
    assert set(tbl["结果"]) == {"⏳持有中(当日新推)"}
    print("场景C（真实推送记录）：通过 —— 打桩无后续行情，全部标 ⏳持有中，不计入胜率")


if __name__ == "__main__":
    shutil.copy2(CSV, BAK)
    CSV.unlink()
    try:
        with mock.patch.object(akshare, "stock_zh_a_hist", fake_hist), \
             mock.patch.object(akshare, "index_zh_a_hist", fake_index):
            scenario_a()
            scenario_b()
            scenario_c()
    finally:
        shutil.copy2(BAK, CSV)                 # 无论成败都恢复真实文件
        BAK.unlink()
    print("\n全部场景通过 ✅")
