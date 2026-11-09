"""每日量化选股脚本：多因子打分 + 历史回测验证
运行方式：.venv\\Scripts\\python.exe screen_today.py
流程：全市场快照 -> 初筛 -> 逐只下载K线算因子 -> MA5/20策略回测 -> 综合得分排序
"""
import datetime as dt
import warnings

import akshare as ak
import pandas as pd

from indicators import add_all
from strategy import generate_signals, backtest

warnings.filterwarnings("ignore")

TOP_N_DETAIL = 80   # 进入深度分析（下载K线）的候选数量
FINAL_N = 10        # 最终清单数量


def load_spot_cached() -> pd.DataFrame:
    """下载全市场快照；30分钟内有本地缓存则直接复用，避免重复等待80秒"""
    import glob, os, time
    cache = f"spot_cache_{dt.date.today()}.csv"
    if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 1800:
        print("      （使用30分钟内的本地缓存）")
        return pd.read_csv(cache, dtype={"代码": str})
    df = ak.stock_zh_a_spot_em()
    for old in glob.glob("spot_cache_*.csv"):   # 清理旧缓存
        os.remove(old)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def prefilter(spot: pd.DataFrame) -> pd.DataFrame:
    """第一层：风险与流动性硬过滤（纯本地计算，秒出）
    注意：技术面条件不放在这里，全部交给后续打分制，避免过度筛选"""
    df = spot.copy()
    df = df[~df["名称"].astype(str).str.contains("ST|退", na=False)]           # 剔除ST/退市股
    df = df[~df["代码"].astype(str).str.startswith(("4", "8", "92", "688"))]   # 剔除北交所/科创板
    df = df[df["最新价"].between(3, 10)]                                       # 小资金限定：只买10元以下
    df = df[df["涨跌幅"].between(-9.8, 19.5)]                                  # 剔除涨跌停附近（买不进/风险大）
    df = df[df["换手率"].between(2, 20)]                                       # 有基本活跃度
    df = df[df["量比"] >= 1.2]                                                 # 今日有所放量
    df = df[(df["市盈率-动态"] > 0) & (df["市盈率-动态"] <= 80)]               # 盈利且估值不离谱
    df = df[df["总市值"].between(50e8, 3000e8)]                                # 中大盘股，抗操纵
    df = df[df["成交额"] >= 1.5e8]                                             # 流动性充足
    if "60日涨跌幅" in df.columns:
        df = df[df["60日涨跌幅"].between(0, 80)]                               # 中期趋势向上且未严重透支
    return df


def factor_check(hist: pd.DataFrame):
    """第二层：单只股票趋势硬条件（基础上升趋势）+ 计算打分因子
    硬条件只保留3条：收盘>MA20、MA5>MA10、MACD多头；其余改为加分项"""
    if len(hist) < 120:
        return None
    h = add_all(hist)
    last = h.iloc[-1]
    close = h["收盘"]
    trend_ok = last["收盘"] > last["MA20"] and last["MA5"] > last["MA10"]
    macd_ok = last["DIF"] > last["DEA"]
    if not (trend_ok and macd_ok):
        return None
    return {
        "mom20": close.iloc[-1] / close.iloc[-21] - 1,                          # 20日动量
        "vol20": close.pct_change().tail(20).std(),                             # 20日波动率
        "rsi6": last["RSI6"],
        "rsi_ok": 50 <= last["RSI6"] <= 78,                                     # 强势但未超买
        "full_bull": last["收盘"] > last["MA5"] > last["MA10"] > last["MA20"]
                     and last["收盘"] > last["MA60"],                           # 完全多头排列（加分项）
    }


def quick_backtest(hist: pd.DataFrame) -> dict:
    """第三层：MA5/20金叉死叉策略近一年回测，返回原始数值（胜率/单笔均收益/交易数/总收益）"""
    h = add_all(hist)
    sig = generate_signals(h, "ma_cross", {"fast": 5, "slow": 20})
    result, trades, _ = backtest(sig, 100000)
    rets = []
    for i in range(0, len(trades) - 1, 2):
        buy, sell = trades.iloc[i], trades.iloc[i + 1]      # DataFrame须按位置取行
        if buy["方向"] == "买入" and sell["方向"] == "卖出":
            rets.append(sell["价格"] / buy["价格"] - 1)
    wins = sum(r > 0 for r in rets)
    return {"win_rate": wins / len(rets) if rets else None,
            "avg_trade": sum(rets) / len(rets) if rets else None,
            "n_trades": len(rets),
            "bt_total": float(result["equity"].iloc[-1]) / 100000 - 1}


def main():
    end = dt.date.today()
    start = end - dt.timedelta(days=400)
    print(f"分析日期: {end}（信号基于最近一个交易日收盘数据）\n")

    print("[1/4] 下载全市场行情快照（首次约80秒）...")
    spot = load_spot_cached()
    print(f"      全市场 {len(spot)} 只")

    print("[2/4] 初筛（流动性+基本面+动量）...")
    cand = prefilter(spot).copy()
    mom_col = "60日涨跌幅" if "60日涨跌幅" in cand.columns else "涨跌幅"
    # 预打分：量比排名 + 中期动量排名，取前 TOP_N_DETAIL 进入深度分析
    cand["_s"] = cand["量比"].rank(pct=True) * 0.5 + cand[mom_col].rank(pct=True) * 0.5
    cand = cand.sort_values("_s", ascending=False).head(TOP_N_DETAIL)
    print(f"      {len(cand)} 只进入深度分析")

    print("[3/4] 逐只下载K线，因子检验 + 策略回测...")
    import time
    rows, fails = [], 0
    for i, (_, r) in enumerate(cand.iterrows(), 1):
        code = str(r["代码"]).zfill(6)[-6:]
        hist = None
        for attempt in range(2):                    # 失败重试1次（东财限流时有效）
            try:
                hist = ak.stock_zh_a_hist(symbol=code, period="daily",
                                          start_date=start.strftime("%Y%m%d"),
                                          end_date=end.strftime("%Y%m%d"), adjust="qfq")
                break
            except Exception:
                if attempt == 0:
                    time.sleep(2)                   # 被限流时等2秒再试
        if hist is None or len(hist) == 0:
            fails += 1
            continue
        time.sleep(0.3)                             # 控制请求频率，防限流
        hist["日期"] = pd.to_datetime(hist["日期"])
        try:
            f = factor_check(hist)
            if f is None:
                continue
            b = quick_backtest(hist)
            rows.append({"代码": code, "名称": r["名称"], "最新价": r["最新价"],
                         "涨跌幅": r["涨跌幅"], "换手率": r["换手率"], "量比": r["量比"],
                         "总市值亿": round(r["总市值"] / 1e8), **f, **b})
        except Exception:
            fails += 1
            continue
        if i % 10 == 0:
            print(f"      进度 {i}/{len(cand)}，通过 {len(rows)} 只，失败 {fails} 只")
    print(f"      深度分析完成：通过 {len(rows)} 只，趋势不达标 {len(cand) - len(rows) - fails} 只，数据失败 {fails} 只")

    if not rows:
        print("\n今日没有股票通过全部量化条件，建议空仓观望。")
        return

    print("[4/4] 综合打分排序...")
    df = pd.DataFrame(rows)
    # 综合得分（百分位加权）：回测胜率20 + 单笔均收益20 + 动量15 + 量比15 + 完全多头15 + RSI区间10 + 低波动5
    df["得分"] = (df["win_rate"].fillna(0).rank(pct=True) * 20
                 + df["avg_trade"].fillna(0).rank(pct=True) * 20
                 + df["mom20"].rank(pct=True) * 15
                 + df["量比"].rank(pct=True) * 15
                 + df["full_bull"].astype(int) * 15
                 + df["rsi_ok"].astype(int) * 10
                 + (1 - df["vol20"].rank(pct=True)) * 5)
    df = df.sort_values("得分", ascending=False).head(FINAL_N).reset_index(drop=True)

    # 输出明细表
    show = df[["代码", "名称", "最新价", "涨跌幅", "换手率", "量比", "总市值亿",
               "mom20", "full_bull", "win_rate", "avg_trade", "n_trades", "bt_total", "得分"]].copy()
    for c, fmt in {"mom20": "{:+.1%}", "win_rate": "{:.0%}", "avg_trade": "{:+.1%}",
                   "bt_total": "{:+.0%}", "得分": "{:.1f}"}.items():
        show[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "-")
    show["full_bull"] = df["full_bull"].map({True: "✓", False: "-"})
    # 2000元本金可买股数（按200股算，佣金最低5元摊薄到0.25%以内）
    show["买200股需"] = (df["最新价"] * 200).round(0).astype(int)
    show["买300股需"] = (df["最新价"] * 300).round(0).astype(int)
    show.columns = ["代码", "名称", "最新价", "今日涨跌", "换手率%", "量比", "总市值亿",
                    "20日动量", "完全多头", "回测胜率", "单笔均收益", "年交易次数", "回测年收益",
                    "综合得分", "买200股需元", "买300股需元"]
    print("\n========== 今日低价股清单（10元以下） ==========")
    print(show.to_string(index=False))

    out = f"../低价股清单_{end}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")

    # 组合期望估算（基于回测统计）
    valid = df.dropna(subset=["win_rate", "avg_trade"])
    if len(valid) > 0:
        exp_trade = valid["avg_trade"].mean()
        avg_trades = valid["n_trades"].mean()
        print(f"\n[组合回测统计] 平均胜率 {valid['win_rate'].mean():.0%}，"
          f"平均单笔收益 {exp_trade:+.2%}，平均年交易 {avg_trades:.1f} 次/只")


if __name__ == "__main__":
    main()
