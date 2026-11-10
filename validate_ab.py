"""新旧评分样本外A/B对决（预测型正规军流程）：
前半段历史估因子权重 → 后半段逐采样日横截面打分 →
比较 旧评分(回测收益/胜率/趋势) 与 新评分(0.6旧+0.4行业中性因子) 对未来5日收益的预测力(IC)。
新评分不跑赢就该回滚因子层。
用法：python validate_ab.py"""
import datetime as dt

import numpy as np
import pandas as pd

import app
import factors
from strategy import backtest, generate_signals

MAX_STOCKS = 60
STEP = 10                      # 样本外每10个交易日取一个截面（控制回测耗时）

cands = app.screen_impl().nlargest(MAX_STOCKS, "成交额")
ind_map = app.industry_map_impl(cands["代码"].astype(str).tolist())
end_d = dt.date.today()
start_d = end_d - dt.timedelta(days=app.HISTORY_DAYS)

hists, meta = {}, {}
for _, r in cands.iterrows():
    code = str(r["代码"]).zfill(6)
    try:
        h = app.get_history_impl(code, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
        if h is None or len(h) < 300:
            continue
        hists[code] = h.reset_index(drop=True)
        pe, pb = app._num(r.get("市盈率-动态")), app._num(r.get("市净率"))
        meta[code] = {"行业": ind_map.get(code, "未分类"),
                      "PE": pe if pe and pe > 0 else None,
                      "PB": pb if pb and pb > 0 else None}
    except Exception:
        continue
print(f"可用股票 {len(hists)} 只")

# ---------- 样本内：前半段估权重 ----------
panel_in = factors.neutralize(factors.build_panel(
    {c: h.iloc[:len(h) // 2] for c, h in hists.items()}, meta))
ics_in = factors.ic_series(panel_in)
ic_in, ir_in = factors.ic_summary(ics_in)
w = factors.ic_weights(ic_in)
print("样本内IC：" + "，".join(f"{k}={v:+.3f}(ICIR{ir_in.get(k, 0):+.2f})" for k, v in ic_in.items()))
print("样本内权重：" + ("，".join(f"{k}={v:+.2f}" for k, v in w.items()) if w else "空（因子全无效）"))
if not w:
    print("\n样本内因子全部无效：线上会自动回退原版，无需A/B。")
    raise SystemExit(0)

# ---------- 样本外：后半段逐采样日打分 ----------
rows = []
for code, h in hists.items():
    turn = h["换手率"] if "换手率" in h.columns else None
    for t in range(len(h) // 2, len(h) - factors.FWD_DAYS, STEP):
        hh = h.iloc[:t + 1]
        try:
            sig = generate_signals(hh, "ma_cross", {"fast": 5, "slow": 20},
                                   cooldown=20, min_hold_days=5)
            _, _, bs = backtest(sig, initial_cash=2200)
            win = str(bs["胜率"]).rstrip("%")
            if win in ("-", ""):
                continue                      # 该截面无成交，与线上口径一致
            last = sig.iloc[-1]
            s = factors.factor_series(hh, meta[code]["PE"], meta[code]["PB"]).iloc[-1]
            rows.append({
                "日期": pd.Timestamp(h["日期"].iloc[t]), "代码": code,
                "行业": meta[code]["行业"],
                "扣费总收益": float(str(bs["总收益率"]).rstrip("%")),
                "胜率": float(win),
                "趋势向上": bool(last["收盘"] > last["_slow"] and last["_fast"] > last["_slow"]),
                **{f: s[f] for f in factors.FACTOR_COLS},
                "未来5日收益": h["收盘"].iloc[t + factors.FWD_DAYS] / h["收盘"].iloc[t] - 1,
            })
        except Exception:
            continue

df = pd.DataFrame(rows)
if len(df) == 0:
    print("样本外无有效观测，无法判定。")
    raise SystemExit(1)


def _score(g):
    g = g.copy()
    old = (g["扣费总收益"].rank(pct=True) * 100 * 0.5
           + g["胜率"].rank(pct=True) * 100 * 0.2
           + g["趋势向上"].map({True: 100.0, False: 0.0}) * 0.3)
    g["旧评分"] = old
    fs = sum(factors.neutral_score(g[c], g["行业"]).fillna(0.5) * wt for c, wt in w.items())
    g["新评分"] = old * 0.6 + fs * 100 * 0.4
    return g


df = pd.concat([_score(g) for _, g in df.groupby("日期")])   # 逐截面打分（保留日期列）
oos = df.dropna(subset=["未来5日收益"])
if len(oos) < 30:
    print(f"样本外有效观测仅 {len(oos)} 条，不足以判定。")
    raise SystemExit(1)

ic_old = oos["旧评分"].rank().corr(oos["未来5日收益"].rank())
ic_new = oos["新评分"].rank().corr(oos["未来5日收益"].rank())
print(f"\n样本外 {len(oos)} 条观测（{oos['日期'].nunique()} 个截面）")
print(f"旧评分 IC = {ic_old:+.4f}")
print(f"新评分 IC = {ic_new:+.4f}")
thr = 1.96 / np.sqrt(len(oos))
print(f"统计显著门槛 |IC| ≈ {thr:.4f}")
print("判定：" + ("新评分跑赢，因子层保留" if ic_new > ic_old else "新评分跑输，应回滚因子层"))
print("（若两者均未过显著门槛，说明样本量不足，结论只能算方向性参考）")
