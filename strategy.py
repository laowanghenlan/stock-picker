"""策略与回测模块
v2 升级：
1) 回测计入真实交易成本：佣金(最低5元)、印花税(卖出)、过户费、滑点
2) 支持仓位管理模式：跌破MA20减半仓 + 硬止损
3) 支持样本内/样本外分割验证，用于识别过拟合
"""
import pandas as pd


def cross_up(short: pd.Series, long: pd.Series) -> pd.Series:
    """金叉：今天 short>long 且昨天 short<=long"""
    return (short > long) & (short.shift(1) <= long.shift(1))


def cross_down(short: pd.Series, long: pd.Series) -> pd.Series:
    """死叉：今天 short<long 且昨天 short>=long"""
    return (short < long) & (short.shift(1) >= long.shift(1))


def generate_signals(df: pd.DataFrame, strategy: str, params: dict,
                     cooldown: int = 0, trend_filter: bool = False,
                     min_hold_days: int = 0) -> pd.DataFrame:
    """
    生成交易信号列 signal：1=买入，-1=卖出，0=无操作
    strategy: "ma_cross" 双均线交叉 | "macd" MACD金叉死叉
    降费过滤器（小资金专用）：
      cooldown       卖出后至少等待 N 个交易日才允许再买，避免反复进出
      trend_filter   仅在收盘价站上 MA60（中期趋势向上）时才允许买入
      min_hold_days  买入后至少持有 N 个交易日才允许卖出，避免隔日来回
    """
    df = df.copy()
    if strategy == "ma_cross":
        fast, slow = params["fast"], params["slow"]
        df["_fast"] = df["收盘"].rolling(fast).mean()
        df["_slow"] = df["收盘"].rolling(slow).mean()
        buy = cross_up(df["_fast"], df["_slow"])
        sell = cross_down(df["_fast"], df["_slow"])
    elif strategy == "macd":
        buy = cross_up(df["DIF"], df["DEA"])
        sell = cross_down(df["DIF"], df["DEA"])
    else:
        raise ValueError(f"未知策略: {strategy}")

    buy = buy.fillna(False).to_numpy()
    sell = sell.fillna(False).to_numpy()
    if trend_filter:                                    # 中期趋势过滤：只做多头趋势
        ma60 = df["收盘"].rolling(60).mean().to_numpy()
        close = df["收盘"].to_numpy()
        buy = buy & (close > ma60)

    # 逐个交易日推进，应用冷却期与最短持有期
    sig = [0] * len(df)
    last_buy = -10 ** 9
    last_sell = -10 ** 9
    for i in range(len(df)):
        if buy[i] and (i - last_sell) >= cooldown:
            sig[i] = 1
            last_buy = i
        elif sell[i] and (i - last_buy) >= min_hold_days:
            sig[i] = -1
            last_sell = i
    df["signal"] = sig
    return df


def backtest(df: pd.DataFrame, initial_cash: float = 100000,
             commission_rate: float = 0.00025, min_commission: float = 5.0,
             stamp_tax: float = 0.0005, transfer_fee: float = 0.00001,
             slippage: float = 0.001,
             position_mode: str = "all_in", stop_loss: float | None = None,
             build_fraction: float = 0.7):
    """
    回测（计入交易成本）
    参数：
      commission_rate 佣金费率，默认万2.5（0.025%），单笔最低 min_commission 元
      stamp_tax       印花税，仅卖出收取，默认 0.05%
      transfer_fee    过户费，双向收取，默认 0.001%
      slippage        滑点，买入价上浮/卖出价下浮，默认单边 0.1%
      position_mode   "all_in" 信号全仓 | "risk_control" 建仓70% + 跌破MA20减半 + 硬止损
      stop_loss       风控模式下的硬止损比例（如 0.08 表示 -8% 清仓）
      build_fraction  风控模式首次建仓资金占比
    返回 (含净值曲线的df, 交易记录DataFrame, 统计指标dict)
    """
    df = df.copy().reset_index(drop=True)
    ma20 = df["收盘"].rolling(20).mean()

    cash = float(initial_cash)
    shares = 0
    avg_cost = 0.0          # 当前持仓平均买入价（含滑点）
    halved = False          # 本轮持仓是否已执行过"跌破MA20减半"
    trades, equity = [], []
    total_fee = 0.0

    def _buy(price, day, budget):
        """按预算买入，返回 (占用现金, 股数, 手续费, 成交价)"""
        bp = price * (1 + slippage)
        sh = int(budget / (bp * (1 + commission_rate + transfer_fee)) // 100) * 100
        while sh > 0:
            amount = sh * bp
            fee = max(amount * commission_rate, min_commission) + amount * transfer_fee
            if amount + fee <= budget:
                break
            sh -= 100
        if sh <= 0:
            return 0.0, 0, 0.0, bp
        amount = sh * bp
        fee = max(amount * commission_rate, min_commission) + amount * transfer_fee
        trades.append({"日期": day, "方向": "买入", "价格": round(bp, 3),
                       "数量": sh, "费用": round(fee, 2), "成本价": round(bp, 3)})
        return amount + fee, sh, fee, bp

    def _sell(price, day, qty, cost_basis):
        """卖出 qty 股，返回 (到手现金, 手续费, 成交价)"""
        sp = price * (1 - slippage)
        amount = qty * sp
        fee = max(amount * commission_rate, min_commission) + amount * (stamp_tax + transfer_fee)
        trades.append({"日期": day, "方向": "卖出", "价格": round(sp, 3),
                       "数量": qty, "费用": round(fee, 2), "成本价": round(cost_basis, 3)})
        return amount - fee, fee, sp

    for i, row in df.iterrows():
        price = float(row["收盘"])
        day = row["日期"]
        sig = int(row.get("signal", 0))
        ma = ma20.iloc[i]

        # ---------- 风控模式：先处理止损与减仓 ----------
        if position_mode == "risk_control" and shares > 0:
            hit_stop = (stop_loss is not None and avg_cost > 0
                        and price <= avg_cost * (1 - stop_loss))
            if hit_stop:                                    # 硬止损：清仓
                proceeds, fee, _ = _sell(price, day, shares, avg_cost)
                cash += proceeds
                total_fee += fee
                shares, avg_cost, halved = 0, 0.0, False
            elif not pd.isna(ma) and price < ma and not halved:   # 跌破MA20：减半仓
                qty = (shares // 2 // 100) * 100
                if qty > 0:
                    proceeds, fee, _ = _sell(price, day, qty, avg_cost)
                    cash += proceeds
                    total_fee += fee
                    shares -= qty
                halved = True
            elif not pd.isna(ma) and price > ma:
                halved = False                              # 重新站上MA20，允许再次减仓

        # ---------- 信号交易 ----------
        if sig == 1 and shares == 0:
            budget = cash if position_mode == "all_in" else cash * build_fraction
            spent, qty, fee, bp = _buy(price, day, budget)
            if qty > 0:
                cash -= spent
                total_fee += fee
                shares, avg_cost, halved = qty, bp, False
        elif sig == -1 and shares > 0:
            proceeds, fee, _ = _sell(price, day, shares, avg_cost)
            cash += proceeds
            total_fee += fee
            shares, avg_cost, halved = 0, 0.0, False

        equity.append(cash + shares * price)                # 每日总资产

    result = df.copy()
    result["equity"] = equity
    # 买入持有基准：期初全仓买入不动（不计成本，仅作对照）
    result["hold_equity"] = df["收盘"] / df["收盘"].iloc[0] * initial_cash

    total_return = equity[-1] / initial_cash - 1
    hold_return = result["hold_equity"].iloc[-1] / initial_cash - 1
    eq = pd.Series(equity)
    days = max((result["日期"].iloc[-1] - result["日期"].iloc[0]).days, 1)
    annual = (1 + total_return) ** (365 / days) - 1
    max_dd = (eq / eq.cummax() - 1).min()

    trades_df = pd.DataFrame(trades)
    # 胜率：每笔卖出是否高于其对应持仓成本
    sells = trades_df[trades_df["方向"] == "卖出"] if not trades_df.empty else trades_df
    if not sells.empty:
        wins = int((sells["价格"] > sells["成本价"]).sum())
        win_rate = f"{wins / len(sells):.2%}"
        n_round_trips = len(sells)
    else:
        win_rate, n_round_trips = "-", 0

    stats = {
        "总收益率": f"{total_return:.2%}",
        "年化收益率": f"{annual:.2%}",
        "最大回撤": f"{max_dd:.2%}",
        "买入持有收益": f"{hold_return:.2%}",
        "交易次数": len(trades_df),
        "完整卖出笔数": n_round_trips,
        "胜率": win_rate,
        "累计手续费": f"{total_fee:.0f}元",
        "成本占本金": f"{total_fee / initial_cash:.2%}",
    }
    return result, trades_df, stats


def split_validation(df: pd.DataFrame, strategy: str, params: dict,
                     split: float = 0.7, initial_cash: float = 100000,
                     signal_kwargs: dict | None = None, **bt_kwargs):
    """
    样本内/样本外分割验证（防过拟合）
    做法：先在完整数据上计算指标与信号（rolling/ewm 均为因果计算，不会用到未来数据），
          再按时间切成前 split 与后 (1-split) 两段分别回测。
    signal_kwargs: 传给 generate_signals 的降费过滤器参数
    返回 (对比表DataFrame, 过拟合提示str或None)
    """
    from indicators import add_all
    data = df.copy().reset_index(drop=True)
    if len(data) < 150:
        return pd.DataFrame(), "数据太短，无法做样本外验证（至少需要150个交易日）"
    sig_df = generate_signals(add_all(data), strategy, params, **(signal_kwargs or {}))
    n = int(len(sig_df) * split)

    rows = []
    for label, part in [("样本内", sig_df.iloc[:n]), ("样本外", sig_df.iloc[n:])]:
        part = part.reset_index(drop=True)
        if len(part) < 30:
            rows.append({"区间": label, "天数": len(part), "总收益": "-",
                         "年化": "-", "最大回撤": "-", "交易次数": 0, "胜率": "-"})
            continue
        _, trades, st = backtest(part, initial_cash, **bt_kwargs)
        rows.append({"区间": label, "天数": len(part),
                     "总收益": st["总收益率"], "年化": st["年化收益率"],
                     "最大回撤": st["最大回撤"],
                     "交易次数": st["交易次数"], "胜率": st["胜率"]})

    table = pd.DataFrame(rows)
    warn = None
    try:
        in_ret = float(str(table.loc[0, "年化"]).rstrip("%"))
        out_ret = float(str(table.loc[1, "年化"]).rstrip("%"))
        if in_ret <= 0 and out_ret <= 0:
            warn = (f"⚠️ 策略无效：样本内 {in_ret:.1f}%、样本外 {out_ret:.1f}%，"
                    f"两段均为负收益，该参数组合不建议实盘使用")
        elif in_ret > 0 and out_ret < in_ret * 0.5:
            warn = (f"⚠️ 过拟合预警：样本内年化 {in_ret:.1f}%，样本外仅 {out_ret:.1f}%，"
                    f"衰减超过一半，策略大概率只是拟合了历史")
        elif out_ret < 0 <= in_ret:
            warn = (f"⚠️ 策略在样本外失效：样本内 {in_ret:.1f}% → 样本外 {out_ret:.1f}%，"
                    f"建议放弃该参数组合")
    except (ValueError, TypeError):
        pass
    return table, warn


# ---------------- 概率校准 & 涨跌停约束（v1.17） ----------------
def calibrate_win_rate(win_rate: float | None, n_samples: int,
                       prior: float = 0.5, strength: float = 20.0) -> float:
    """贝叶斯收缩：把回测胜率向先验 50% 拉，样本越少拉得越狠。
        p = (胜率×n + 先验×strength) / (n + strength)
    n=0 → 先验；n=strength → 先验与回测各占一半；n 很大 → 接近原胜率。
    用于凯利仓位与"盈利概率"展示，避免 10 笔以内的小样本把仓位带偏。"""
    prior = min(max(float(prior), 0.0), 1.0)
    try:
        n = max(int(n_samples), 0)
    except (TypeError, ValueError):
        n = 0
    if win_rate is None or win_rate != win_rate:      # None / NaN
        return prior
    wr = min(max(float(win_rate), 0.0), 1.0)
    k = max(float(strength), 1e-9)
    return (wr * n + prior * k) / (n + k)


def price_limit_pct(code: str) -> float:
    """涨跌停幅度：创业板(30)/科创板(68) 20%，北交所 30%，其余主板 10%（ST 用实际涨跌幅判断）。"""
    c = str(code).zfill(6)
    if c.startswith(("30", "68")):
        return 0.20
    if c.startswith(("4", "8", "920")):
        return 0.30
    return 0.10


def at_limit_up(pct_chg: float, code: str, tol: float = 0.002) -> bool:
    """当日涨幅是否已触及涨停（涨停买不进）。pct_chg 为百分数，如 9.98。"""
    if pct_chg is None or pct_chg != pct_chg:
        return False
    return float(pct_chg) >= price_limit_pct(code) * 100 - tol * 100


def at_limit_down(pct_chg: float, code: str, tol: float = 0.002) -> bool:
    """当日跌幅是否已触及跌停（跌停卖不出）。"""
    if pct_chg is None or pct_chg != pct_chg:
        return False
    return float(pct_chg) <= -price_limit_pct(code) * 100 + tol * 100
