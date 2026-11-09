"""离线验证脚本：对当前候选池做因子 pooled IC 检验。
打印各因子IC与启用权重；全部无效时评分自动回退原版（代码已内置，无需人工处理）。
用法：python validate_factors.py"""
import datetime as dt
import pandas as pd

import app
import factors

cands = app.screen_impl()
print(f"初筛候选 {len(cands)} 只，抓取历史并做IC检验（取前60只控制耗时）……")

obs, used = [], 0
end_d = dt.date.today()
start_d = end_d - dt.timedelta(days=app.HISTORY_DAYS)
for _, r in cands.head(60).iterrows():
    try:
        h = app.get_history_impl(r["代码"], start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
        if h is None or len(h) < 120:
            continue
        obs += factors.collect_obs(r["代码"], h)
        used += 1
    except Exception:
        continue

df = pd.DataFrame(obs)
ic = factors.pooled_ic(df)
w = factors.ic_weights(ic)
print(f"\n样本：{used} 只股票，每{factors.SAMPLE_EVERY}个交易日采样，共 {len(df)} 个观测")
for k, v in ic.items():
    if k in w:
        print(f"  {k}: IC={v:+.3f} 启用 权重={w[k]:+.2f}")
    else:
        print(f"  {k}: IC={v:+.3f} 无效（|IC|<{factors.IC_MIN}），已剔除")
print("\n结论：" + ("因子层生效，新评分 = 0.6x回测分 + 0.4x因子分" if w
                   else "全部因子无效，评分保持回测原版，不推送代码改动"))
