"""L1 多因子层：动量/波动率/换手 + pooled IC 检验。

做法：在候选股的2年日线历史上，每10个交易日采样一次，
记录当时因子值与未来5日收益，做截面Spearman秩相关（pooled IC）。
|IC|>=0.03 才算有效因子；权重=带符号IC归一化——方向由数据决定
（IC为正则因子值越大越好，为负则自动变成反转因子）。
全部无效时返回空权重，上层评分自动退回回测原版（安全回退）。
"""
import numpy as np
import pandas as pd

FACTOR_COLS = ["动量20", "波动率20", "换手20"]
IC_MIN = 0.03          # 有效性门槛（量化通行经验值）
SAMPLE_EVERY = 10      # 每隔10个交易日采一个样本
FWD_DAYS = 5           # 检验因子对未来5日收益的预测力
WARMUP = 60            # 前60根K线不参与采样（保证因子窗口完整）


def _fvals(close, turn):
    """在某一截面上计算三个因子原始值（方向不预设，交给IC符号决定）。"""
    if len(close) < 21:
        return dict.fromkeys(FACTOR_COLS, np.nan)
    mom = close.iloc[-1] / close.iloc[-21] - 1
    vol = close.pct_change().tail(20).std()
    to = float(turn.tail(20).mean()) if turn is not None else np.nan
    return {"动量20": mom, "波动率20": vol, "换手20": to}


def latest_factors(h: pd.DataFrame) -> dict:
    """最新一个交易日的因子值（用于给候选股打分）。"""
    turn = h["换手率"] if "换手率" in h.columns else None
    return _fvals(h["收盘"], turn)


def collect_obs(code: str, h: pd.DataFrame) -> list:
    """在一只股的历史上滚动采样（因子值, 未来5日收益）观测对。"""
    close = h["收盘"]
    turn = h["换手率"] if "换手率" in h.columns else None
    obs = []
    for t in range(WARMUP, len(h) - FWD_DAYS, SAMPLE_EVERY):
        fv = _fvals(close.iloc[:t + 1], turn.iloc[:t + 1] if turn is not None else None)
        fv["未来5日收益"] = close.iloc[t + FWD_DAYS] / close.iloc[t] - 1
        obs.append(fv)
    return obs


def pooled_ic(obs: pd.DataFrame) -> dict:
    """pooled Spearman IC：因子秩与未来收益秩的相关性，样本不足记0。"""
    ic = {}
    for f in FACTOR_COLS:
        d = obs[[f, "未来5日收益"]].dropna()
        ic[f] = float(d[f].rank().corr(d["未来5日收益"].rank())) if len(d) >= 30 else 0.0
    return ic


def ic_weights(ic: dict) -> dict:
    """|IC|<门槛的因子剔除；剩余按带符号IC的绝对值归一化。"""
    valid = {f: v for f, v in ic.items() if abs(v) >= IC_MIN}
    s = sum(abs(v) for v in valid.values())
    return {f: v / s for f, v in valid.items()} if s > 0 else {}
