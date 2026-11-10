"""离线验证：多因子层（动量/波动率/换手/估值 EP·BP）行业中性化的IC检验。
输出各因子近20日IC均值、ICIR、样本截面数与启用权重。
用法：python validate_factors.py"""
import datetime as dt

import pandas as pd

import app
import factors

MAX_STOCKS = 120        # 横截面越大，行业中性化越有意义

spot = app.get_spot_impl()
spot["代码"] = spot["代码"].astype(str).str.zfill(6)
cands = app.screen_impl().nlargest(MAX_STOCKS, "成交额")
print(f"初筛候选 {len(cands)} 只，开始抓历史与行业（首次较慢）……")

ind_map = app.industry_map_impl(cands["代码"].tolist())
end_d = dt.date.today()
start_d = end_d - dt.timedelta(days=app.HISTORY_DAYS)

hists, meta = {}, {}
for _, r in cands.iterrows():
    code = str(r["代码"]).zfill(6)
    try:
        h = app.get_history_impl(code, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
        if h is None or len(h) < 200:
            continue
        hists[code] = h
        pe, pb = app._num(r.get("市盈率-动态")), app._num(r.get("市净率"))
        meta[code] = {"行业": ind_map.get(code, "未分类"),
                      "PE": pe if pe and pe > 0 else None,
                      "PB": pb if pb and pb > 0 else None}
    except Exception:
        continue

panel = factors.neutralize(factors.build_panel(hists, meta))
ics = factors.ic_series(panel)
ic, ir = factors.ic_summary(ics)
w = factors.ic_weights(ic)

n_days = len(ics)
print(f"\n样本：{len(hists)} 只股票，{n_days} 个交易日截面，{len(panel)} 条观测")
print(f"行业覆盖：{meta and len({m['行业'] for m in meta.values()})} 个行业")
for f in factors.FACTOR_COLS:
    v, s = ic.get(f), ir.get(f, 0.0)
    if v is None:
        print(f"  {f}: 无有效样本")
    elif f in w:
        print(f"  {f}: IC={v:+.3f} ICIR={s:+.2f} 启用 权重={w[f]:+.2f}")
    else:
        print(f"  {f}: IC={v:+.3f} ICIR={s:+.2f} 无效（|IC|<{factors.IC_MIN}）已剔除")
print("\n结论：" + ("因子层生效" if w else "全部因子无效，评分保持回测原版"))
