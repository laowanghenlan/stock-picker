# -*- coding: utf-8 -*-
"""
趋势过滤验证（结论：无效，不上线；版本保持 v1.59）：
检验"剔除过去60日跌幅过深的票"能否改善次日(T+1)收益。
口径与实盘一致：T日收盘买、T+1收盘卖，扣 0.40% 往返成本。
样本：wf_cache 全A日线（372只，约27.8万条有效配对）。

结论（勿重复踩坑）：
- 单调性反向：跌>30%组次日净收益 -0.157%（全组最好），跌10~20%组 -0.343%（最差）。
  不是"越跌越差"而是 U 型，深跌股存在超跌反弹，剔除它们反而丢掉相对更好的样本。
- 剔除跌超20%后整体均值 -0.289% → -0.289%，零改善（甚至 -0.007% 略变差）。
- 稳定性崩塌：深跌组相对优势 前半段 -0.179%/日 → 后半段 -0.011%/日 → 2026年 -0.003%/日，
  是 2023~2025 特定行情产物，今年已归零。RankIC=-0.0119≈0。
- 台账19笔小样本曾显示"剔除更好"，是噪声，不可作为上线依据。
- 用户已确认：接受结论，不上线。策略代码零改动。

样本偏差说明：wf_cache 只含当前存活股票，有幸存者偏差；但由于结论本身是
"深跌不比其余差"，纳入退市股只会让"剔除深跌"看起来更没价值，不改变否决方向。
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

CACHE = "wf_cache"
COST = 0.40  # 往返成本 %
LOOKBACK = 60

frames = []
for p in glob.glob(os.path.join(CACHE, "hist_*.csv")):
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        continue
    code = os.path.basename(p).replace("hist_", "").replace(".csv", "")
    df = df[["日期", "收盘"]].copy()
    df["收盘"] = pd.to_numeric(df["收盘"], errors="coerce")
    df = df.dropna().sort_values("日期").reset_index(drop=True)
    if len(df) < LOOKBACK + 2:
        continue
    df["代码"] = code
    df["mom60"] = df["收盘"] / df["收盘"].shift(LOOKBACK) - 1
    df["fwd1"] = df["收盘"].shift(-1) / df["收盘"] - 1
    df["fwd1_net"] = df["fwd1"] * 100 - COST
    frames.append(df[["日期", "代码", "收盘", "mom60", "fwd1_net"]])

panel = pd.concat(frames, ignore_index=True)
panel = panel.dropna(subset=["mom60", "fwd1_net"])
print(f"全样本有效配对: {len(panel):,} 条, {panel['代码'].nunique()} 只, "
      f"{panel['日期'].min()} ~ {panel['日期'].max()}")
print()

def row(label, sub):
    v = sub["fwd1_net"].values
    if len(v) == 0:
        print(f"{label:>16}: 无样本")
        return
    t, p = stats.ttest_1samp(v, 0)
    win = 100 * (v > 0).mean()
    print(f"{label:>16}: n={len(v):>7,}  均值{v.mean():+7.3f}%  "
          f"中位{np.median(v):+7.3f}%  胜率{win:5.1f}%  t={t:7.2f} p={p:.4g}")

print("=== 按过去60日动量分组的次日净收益（T+1，扣0.4%成本）===")
bins = [-np.inf, -0.30, -0.20, -0.10, 0, 0.10, 0.30, np.inf]
labels = ["跌>30%", "跌20~30%", "跌10~20%", "跌0~10%", "涨0~10%", "涨10~30%", "涨>30%"]
panel["grp"] = pd.cut(panel["mom60"], bins=bins, labels=labels)
for g in labels:
    row(g, panel[panel["grp"] == g])
print()

print("=== 关键二分：深跌(<=-20%) vs 其余 ===")
deep = panel[panel["mom60"] <= -0.20]
rest = panel[panel["mom60"] > -0.20]
row("深跌<=-20%", deep)
row("其余", rest)
t, p = stats.ttest_ind(deep["fwd1_net"], rest["fwd1_net"], equal_var=False)
print(f"  两组差异检验: t={t:.2f} p={p:.4g}  "
      f"深跌组比其余差 {(rest['fwd1_net'].mean()-deep['fwd1_net'].mean()):.3f}%/日")
print()

print("===  correlations / Rank IC（mom60 vs 次日净收益）===")
corr = panel["mom60"].corr(panel["fwd1_net"])
rank_ic = panel["mom60"].corr(panel["fwd1_net"], method="spearman")
print(f"  Pearson={corr:.4f}  RankIC={rank_ic:.4f}")
print()

print("=== 阈值扫描：剔除 mom60 < -X% 后，整体次日净收益改善量 ===")
base = panel["fwd1_net"].mean() * 100
for x in [5, 10, 15, 20, 25, 30]:
    kept = panel[panel["mom60"] > -x / 100]
    print(f"  剔除跌超{x:>2}%: 保留{len(kept):>7,}条({100*len(kept)/len(panel):5.1f}%)  "
          f"均值{kept['fwd1_net'].mean():+7.3f}%  改善{(kept['fwd1_net'].mean()-panel['fwd1_net'].mean()):+6.3f}%/日")
print()

print("=== 前后半段稳定性（深跌组 vs 其余的差，样本外衰减检验）===")
dates = sorted(panel["日期"].unique())
mid = dates[len(dates)//2]
for seg, sub in [("前半段", panel[panel["日期"] <= mid]), ("后半段", panel[panel["日期"] > mid])]:
    d = sub[sub["mom60"] <= -0.20]["fwd1_net"]
    r = sub[sub["mom60"] > -0.20]["fwd1_net"]
    if len(d) and len(r):
        print(f"  {seg}({sub['日期'].min()}~{sub['日期'].max()}): "
              f"深跌{d.mean():+.3f}% 其余{r.mean():+.3f}% "
              f"差{(r.mean()-d.mean()):+.3f}%/日  n深={len(d):,}")
print()

print("=== 年度稳定性 ===")
panel["年"] = panel["日期"].str[:4]
for y, sub in panel.groupby("年"):
    d = sub[sub["mom60"] <= -0.20]["fwd1_net"]
    r = sub[sub["mom60"] > -0.20]["fwd1_net"]
    if len(d) > 100 and len(r) > 100:
        print(f"  {y}: 深跌{d.mean():+.3f}% 其余{r.mean():+.3f}% "
              f"差{(r.mean()-d.mean()):+.3f}%/日")
