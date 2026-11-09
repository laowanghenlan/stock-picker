"""新旧评分 A/B 样本外对比（预测型正规军流程）：
样本内（历史前半段）估因子IC和权重 -> 样本外（后半段）逐采样日做横截面打分，
比较 旧评分(收益/胜率/趋势) vs 新评分(0.6旧+0.4因子) 对未来5日收益的预测力(pooled Spearman IC)。
新评分跑赢才保留因子层，跑输应回滚。
用法：python validate_ab.py"""
import datetime as dt
import pandas as pd

import app
import factors
from strategy import generate_signals, backtest

print("A/B样本外验证：抓取候选股历史（前60只控制耗时）……")
hists = {}
end_d = dt.date.today()
start_d = end_d - dt.timedelta(days=app.HISTORY_DAYS)
for _, r in app.screen_impl().head(60).iterrows():
    try:
        h = app.get_history_impl(r["代码"], start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
        if h is not None and len(h) >= 200:
            hists[r["代码"]] = h.reset_index(drop=True)
    except Exception:
        pass
print(f"可用股票 {len(hists)} 只")

# ---- 样本内（前半段）：估因子权重，模拟线上每天的IC自检 ----
obs_in = []
for code, h in hists.items():
    mid = len(h) // 2
    turn = h["换手率"] if "换手率" in h.columns else None
    for t in range(factors.WARMUP, mid, factors.SAMPLE_EVERY):
        fv = factors._fvals(h["收盘"].iloc[:t + 1], turn.iloc[:t + 1] if turn is not None else None)
        fv["未来5日收益"] = h["收盘"].iloc[t + factors.FWD_DAYS] / h["收盘"].iloc[t] - 1
        obs_in.append(fv)
ic_in = factors.pooled_ic(pd.DataFrame(obs_in))
w = factors.ic_weights(ic_in)
print("样本内因子IC：" + "，".join(f"{k}={v:+.3f}" for k, v in ic_in.items()))
print("样本内权重：" + ("，".join(f"{k}={v:+.2f}" for k, v in w.items()) if w else "空（因子全无效）"))

if not w:
    print("\n样本内因子全部无效：线上代码本就会自动回退回测原版，因子层形同虚设，无需A/B。")
    raise SystemExit(0)

# ---- 样本外（后半段）：每个采样日做横截面新旧打分，并记录未来5日收益 ----
def _f(x):
    """百分数字符串安全转float；回测无成交时胜率是'-'，转NaN后续统一剔除。"""
    x = str(x).rstrip("%")
    return float(x) if x not in ("-", "") else float("nan")


rows = []
for code, h in hists.items():
    mid = len(h) // 2
    turn_all = h["换手率"] if "换手率" in h.columns else None
    for t in range(mid, len(h) - factors.FWD_DAYS, factors.SAMPLE_EVERY):
        hh = h.iloc[:t + 1]
        try:
            sig = generate_signals(hh, "ma_cross", {"fast": 5, "slow": 20},
                                   cooldown=20, min_hold_days=5)
            _, _, bs = backtest(sig, initial_cash=2200)
            last = sig.iloc[-1]
            rows.append({
                "t": t, "采样日": t, "代码": code,
                "扣费总收益": _f(bs["总收益率"]),
                "胜率": _f(bs["胜率"]),
                "趋势向上": bool(last["收盘"] > last["_slow"] and last["_fast"] > last["_slow"]),
                **factors._fvals(hh["收盘"], turn_all.iloc[:t + 1] if turn_all is not None else None),
                "未来5日收益": h["收盘"].iloc[t + factors.FWD_DAYS] / h["收盘"].iloc[t] - 1,
            })
        except Exception:
            continue


def _score(g):
    """单个采样日的横截面百分位打分（与线上 pick_top3 同口径）。"""
    g = g.copy()
    s_ret = g["扣费总收益"].rank(pct=True) * 100
    s_win = g["胜率"].rank(pct=True) * 100
    s_trd = g["趋势向上"].map({True: 100.0, False: 0.0})
    g["旧评分"] = s_ret * 0.5 + s_win * 0.2 + s_trd * 0.3
    fs = sum(g[c].rank(pct=True).fillna(0.5) * wt for c, wt in w.items()) * 100
    g["新评分"] = g["旧评分"] * 0.6 + fs * 0.4
    return g


df = pd.DataFrame(rows)
if len(df) == 0:
    print("样本外没有任何完成回测的观测，无法A/B。")
    raise SystemExit(1)
df = df.dropna(subset=["胜率", "扣费总收益"])   # 剔除该截面无成交的股票（与线上口径一致）
df = df.groupby("t", group_keys=False).apply(_score)
oos = df.dropna(subset=["未来5日收益"])
if len(oos) < 30:
    print(f"样本外有效观测仅 {len(oos)} 条，不足以判定。")
    raise SystemExit(1)
ic_old = oos["旧评分"].rank().corr(oos["未来5日收益"].rank())
ic_new = oos["新评分"].rank().corr(oos["未来5日收益"].rank())
print(f"\n样本外观测 {len(oos)} 条（{oos['采样日'].nunique()} 个采样日，每日横截面打分）")
print(f"旧评分 样本外IC = {ic_old:+.4f}")
print(f"新评分 样本外IC = {ic_new:+.4f}")
print("判定：" + ("新评分跑赢，因子层保留" if ic_new > ic_old
                   else "新评分跑输或不增，应回滚因子层"))
