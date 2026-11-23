# -*- coding: utf-8 -*-
"""
行业硬分散验证（取长补短：对标 AlphaSift 的 8 行业桶 max_same_bucket=1）。
问题：给每行业最多选1只，能否改善组合？
方法：
1) 全样本验证同行业股票日收益相关性是否显著高于跨行业（相关性高→同涨同跌→集中有风险）。
2) 模拟"每天从候选里选N只"的组合：随机选 vs 行业硬分散（每行业最多1只），
   比较组合 T+1 收益的波动/最大回撤。
行业映射：industry_full.json（东财 f100，372只全覆盖）。
数据源：wf_cache 日线（有幸存者偏差，但行业相关性结论方向稳健）。
"""
import json
import glob
import os
import itertools
import numpy as np
import pandas as pd
from scipy import stats

IND = json.load(open("industry_full.json", encoding="utf-8"))

# 读所有有行业的股票日收益，按日期对齐
series = {}
for p in glob.glob("wf_cache/hist_*.csv"):
    c = os.path.basename(p).replace("hist_", "").replace(".csv", "")
    if c not in IND:
        continue
    df = pd.read_csv(p, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    s = pd.to_numeric(df.set_index("日期")["收盘"], errors="coerce").pct_change()
    series[c] = s
panel = pd.DataFrame(series)
print(f"有效股票(有行业+有序列): {panel.shape[1]} 只, {len(panel)} 个交易日")

# ===== 1. 同行业 vs 跨行业 日收益相关性 =====
same, cross = [], []
for c1, c2 in itertools.combinations(panel.columns, 2):
    sub = panel[[c1, c2]].dropna()
    if len(sub) < 120:
        continue
    corr = sub[c1].corr(sub[c2])
    if not np.isfinite(corr):
        continue
    (same if IND[c1] == IND[c2] else cross).append(corr)
same, cross = np.array(same), np.array(cross)
print("\n=== 1. 日收益相关性（同行业 vs 跨行业）===")
print(f"同行业对 n={len(same)}  均值={same.mean():.3f} 中位={np.median(same):.3f}")
print(f"跨行业对 n={len(cross)}  均值={cross.mean():.3f} 中位={np.median(cross):.3f}")
print(f"差值={same.mean()-cross.mean():+.3f}")
t, p = stats.ttest_ind(same, cross, equal_var=False)
print(f"Welch t={t:.2f} p={p:.4g}  {'显著' if p < 0.05 else '不显著'}")

# ===== 2. 组合模拟：随机选N只 vs 行业硬分散 =====
# 每天从当天有数据的股票里模拟选 N=5 只，持有1日（T+1）。
# 随机组：完全随机；分散组：每行业最多1只。各重复 500 次取分布。
rets = panel
valid_days = rets.dropna(thresh=50).index
rng = np.random.default_rng(42)
N = 5
REPEAT = 500
rand_port, div_port = [], []
for d in valid_days:
    row = rets.loc[d].dropna()
    if len(row) < N + 5:
        continue
    # 随机组
    for _ in range(REPEAT // 10):
        pick = rng.choice(row.index, N, replace=False)
        rand_port.append(row[pick].mean())
    # 分散组：先打乱，每个行业只取第一只，直到够N只
    codes = list(row.index)
    rng.shuffle(codes)
    seen, div = set(), []
    for c in codes:
        ind = IND.get(c, c)
        if ind in seen:
            continue
        seen.add(ind)
        div.append(c)
        if len(div) == N:
            break
    if len(div) == N:
        div_port.append(row[div].mean())

rand_port, div_port = np.array(rand_port), np.array(div_port)
print(f"\n=== 2. 组合模拟（每日选{N}只，T+1，各{REPEAT}次）===")
for name, v in [("随机选", rand_port), ("行业硬分散", div_port)]:
    print(f"{name}: n={len(v)} 日均{v.mean():+.4%} 日波动{v.std():.4%} "
          f"P5={np.percentile(v,5):+.3%} P95={np.percentile(v,95):+.3%}")
print(f"波动下降: {(1-div_port.std()/rand_port.std())*100:.1f}%  "
      f"日均收益差: {(div_port.mean()-rand_port.mean())*100:+.4f}%/日")
t, p = stats.ttest_ind(rand_port, div_port, equal_var=False)
print(f"收益差异 Welch t={t:.2f} p={p:.4g}（不显著=分散不改收益，只降波动）")

# ===== 3. 结论自检 =====
print("\n=== 3. 判定 ===")
corr_gap = same.mean() - cross.mean()
vol_drop = (1 - div_port.std() / rand_port.std()) * 100
ret_diff_bp = (div_port.mean() - rand_port.mean()) * 10000
print(f"同行业相关高 {corr_gap:.3f} ({'是' if corr_gap > 0.1 else '否'})；"
      f"分散降波动 {vol_drop:.1f}% ({'是' if vol_drop > 5 else '否'})；"
      f"收益影响 {ret_diff_bp:+.1f}bp/日")
