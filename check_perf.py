"""核对选股清单表现：清单价（11-09下午13:56盘中） vs 今日收盘价"""
import time
import pandas as pd
import akshare as ak

CSV = "../周一选股清单_2026-11-09.csv"
df = pd.read_csv(CSV, dtype={"代码": str})

rows = []
for _, r in df.iterrows():
    code = str(r["代码"]).zfill(6)[-6:]
    try:
        h = ak.stock_zh_a_hist(symbol=code, period="daily",
                               start_date="20261105", end_date="20261109", adjust="qfq")
        h["日期"] = pd.to_datetime(h["日期"])
        d = h[h["日期"] == "2026-11-09"]
        if d.empty:
            rows.append({"代码": code, "名称": r["名称"], "清单价": r["最新价"],
                         "收盘价": None, "变动%": None})
        else:
            close = float(d.iloc[-1]["收盘"])
            rows.append({"代码": code, "名称": str(r["名称"]).replace(" ", ""),
                         "清单价": float(r["最新价"]), "收盘价": close,
                         "变动%": close / float(r["最新价"]) - 1})
    except Exception as e:
        rows.append({"代码": code, "名称": r["名称"], "清单价": r["最新价"],
                     "收盘价": None, "变动%": None})
    time.sleep(0.4)

out = pd.DataFrame(rows)
pd.set_option("display.unicode.east_asian_width", True)
print("===== 清单价(下午13:56盘中) → 今日收盘 =====\n")
for _, r in out.iterrows():
    if r["收盘价"] is None:
        print(f'{r["代码"]} {r["名称"]}: 无数据')
    else:
        pnl = (r["收盘价"] - r["清单价"]) * 100
        print(f'{r["代码"]} {r["名称"]:<6} 清单 {r["清单价"]:>7.2f} -> 收盘 {r["收盘价"]:>7.2f} '
              f'({r["变动%"]:>+6.2%})  1手浮盈亏 {pnl:>+7.1f}元')

valid = out.dropna(subset=["变动%"])
print(f"\n平均变动: {valid['变动%'].mean():+.2%}")
print(f"上涨 {int((valid['变动%']>0).sum())} 只 / 下跌 {int((valid['变动%']<0).sum())} 只 / 总数 {len(valid)}")
out.to_csv("../清单表现核对_2026-11-09.csv", index=False, encoding="utf-8-sig")
