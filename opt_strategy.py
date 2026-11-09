"""策略降费寻优：在5只低价股上网格搜索低交易频率参数
评估标准：扣费后收益 + 手续费占比 + 样本外表现
"""
import itertools
import akshare as ak
import pandas as pd
from indicators import add_all
from strategy import generate_signals, backtest

STOCKS = [("002697", "红旗连锁"), ("002081", "金螳螂"), ("600368", "五洲交通"),
          ("600163", "中闽能源"), ("002419", "天虹股份")]
CASH = 2000

# 成本按中信实际：佣金万2.5最低5元、印花税0.05%、滑点0.1%
COST = dict(commission_rate=0.00025, min_commission=5.0, stamp_tax=0.0005,
            transfer_fee=0.00001, slippage=0.001, position_mode="all_in")

data = {}
for code, nm in STOCKS:
    h = ak.stock_zh_a_hist(symbol=code, period="daily",
                           start_date="20240101", end_date="20261109", adjust="qfq")
    h["日期"] = pd.to_datetime(h["日期"])
    data[code] = add_all(h.sort_values("日期").reset_index(drop=True))
print(f"已加载 {len(data)} 只股票，样本约 {len(next(iter(data.values())))} 个交易日\n")

grid = list(itertools.product([(5, 20), (10, 30), (20, 60)], [0, 10, 20], [False, True], [0, 5]))
rows = []

for (fast, slow), cd, tf, mh in grid:
    in_rets, out_rets, fees, ntr = [], [], [], []
    for code, df in data.items():
        n = int(len(df) * 0.7)
        for part, bucket in [(df.iloc[:n], in_rets), (df.iloc[n:], out_rets)]:
            p = part.reset_index(drop=True)
            sig = generate_signals(p, "ma_cross", {"fast": fast, "slow": slow},
                                   cooldown=cd, trend_filter=tf, min_hold_days=mh)
            _, tr, st = backtest(sig, CASH, **COST)
            ret = float(st["总收益率"].rstrip("%")) / 100
            days = max((p["日期"].iloc[-1] - p["日期"].iloc[0]).days, 1)
            ann = (1 + ret) ** (365 / days) - 1
            bucket.append(ann)
            if bucket is out_rets:
                fees.append(float(st["累计手续费"].rstrip("元")))
                ntr.append(st["交易次数"])
    rows.append({"均线": f"MA{fast}/{slow}", "冷却期": cd, "趋势过滤": "是" if tf else "否",
                 "最短持有": mh,
                 "样本内年化": sum(in_rets) / len(in_rets),
                 "样本外年化": sum(out_rets) / len(out_rets),
                 "平均手续费": sum(fees) / len(fees),
                 "手续费占本金": sum(fees) / len(fees) / CASH,
                 "平均交易次数": sum(ntr) / len(ntr)})

res = pd.DataFrame(rows).sort_values("样本外年化", ascending=False).reset_index(drop=True)
pd.set_option("display.unicode.east_asian_width", True)
show = res.head(12).copy()
for c, fmt in {"样本内年化": "{:+.2%}", "样本外年化": "{:+.2%}", "手续费占本金": "{:.2%}",
               "平均手续费": "{:.0f}元", "平均交易次数": "{:.1f}"}.items():
    show[c] = show[c].map(lambda x: fmt.format(x))
print("===== 按样本外年化排序 TOP12（2000元本金，扣全部费用）=====")
print(show.to_string(index=False))

# 同时给出基线对比
base = res[(res["均线"] == "MA5/20") & (res["冷却期"] == 0)
           & (res["趋势过滤"] == "否") & (res["最短持有"] == 0)]
if not base.empty:
    b = base.iloc[0]
    print(f"\n基线 MA5/20 无过滤：样本外 {b['样本外年化']:+.2%}，"
          f"手续费 {b['平均手续费']:.0f}元({b['手续费占本金']:.2%})，交易 {b['平均交易次数']:.1f}次")
res.to_csv("../策略降费寻优结果.csv", index=False, encoding="utf-8-sig")
print("\n完整36组结果已保存: ../策略降费寻优结果.csv")
