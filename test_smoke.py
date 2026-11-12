"""端到端冒烟测试：跑完整选股链路，抓所有异常与列缺失。
不依赖 Streamlit UI，纯逻辑函数调用，5分钟跑完。"""
import sys, traceback, pandas as pd
sys.path.insert(0, r"D:\文档\股票")

errors = []
def check(name, fn):
    try:
        fn()
        print(f"  ✅ {name}")
    except Exception as e:
        errors.append((name, traceback.format_exc()))
        print(f"  ❌ {name} → {e}")

# ---------- 1. import all ----------
print("① import 全部模块 ...")
check("import indicators", lambda: __import__("indicators"))
check("import strategy", lambda: __import__("strategy"))
check("import factors", lambda: __import__("factors"))
check("import data", lambda: __import__("data"))
import app
check("import app", lambda: None)

# verify_impl 内部引用了模块顶层的 capital（Streamlit 页面的 number_input 变量）
# 裸跑测试时注入一下默认值
app.capital = 2200.0

# ---------- 2. 数据加载 ----------
print("\n② 行情拉取 ...")
spot = app.get_spot_impl()
print(f"  行情：{len(spot)} 条")
assert "代码" in spot.columns, f"缺列：代码，实际 {list(spot.columns)}"

# ---------- 3. verify（核心选股链路）----------
print("\n③ verify 选股主流程（取前3只加速验证）...")
# 先验证 verify_impl 的输出列是否完整
cands = spot[(spot["最新价"] >= app.MIN_PRICE) & (spot["最新价"] <= 10.4)].head(3).copy()
res = app.verify_impl(cands, progress=lambda x, y: None)
print(f"  verify 输出：{len(res)} 条，列={list(res.columns)}")

expected_cols = ["代码", "扣费总收益", "胜率", "平均盈利率", "平均亏损率",
                 "趋势向上", "有雷", "动量20", "波动率20", "换手20", "MACD柱", "KDJ_J",
                 "EP", "BP", "行业"]
for c in expected_cols:
    if c not in res.columns:
        errors.append(("verify列缺失", f"缺 {c}"))
        print(f"  ⚠️ 缺列：{c}")

# ---------- 4. pick_top3 ----------
print("\n④ pick_top3 打分 ...")
if len(res) > 0:
    ranked = app.pick_top3(res, 5)
    print(f"  pick_top3 输出：{len(ranked)} 条，列={list(ranked.columns)}")
    check("ranked有总分列", lambda: ranked["总分"] is not None)
    check("ranked有扣费总收益列", lambda: ranked["扣费总收益"] is not None)

    # ---------- 5. best_plan（之前炸的函数）----------
    print("\n⑤ best_plan 只数优化 ...")
    plan, n = app.best_plan(ranked, 2200.0)
    print(f"  best_plan：最优 {n} 只")
    if len(ranked) == 0:
        # 3只样本可能全部扣费亏损/有雷被过滤，空仓是合法结果，不是bug
        assert plan is None and n == 0, f"空候选应返回(None,0)，实际 {type(plan)} / {n}"
        print("  候选全部被过滤（扣费亏损或有雷），空仓结果合法，跳过 allocate")
    else:
        assert len(plan) == n, f"只数不匹配 {len(plan)} vs {n}"
        check("plan 投入含佣金", lambda: "投入(含佣金)" in plan.columns)

        # ---------- 6. allocate ----------
        print("\n⑥ allocate 分配本金 ...")
        p2, left = app.allocate(ranked.head(3), 2200.0)
        print(f"  allocate：{len(p2)} 只，剩余 {left} 元")
        check("allocate 有买入股数列", lambda: "买入股数" in p2.columns)

# ---------- 7. 复盘 ----------
print("\n⑦ 复盘 ...")
rv = app.review_impl()
print(f"  review：{len(rv)} 条")
if len(rv) > 0:
    for c in ["结果", "盈亏%"]:
        if c not in rv.columns:
            errors.append(("review列缺失", c))
    app.review_conclusion(rv)

# ---------- 8. 因子 ----------
print("\n⑧ 因子自检 ...")
import factors
# 用 verify_impl 已算出的 attrs
icw = res.attrs.get("ic_weights") or {}
print(f"  ic_weights：{icw}")
check("ic_weights dict", lambda: isinstance(icw, dict))

# ---------- 汇总 ----------
print("\n" + "=" * 50)
if errors:
    print(f"❌ 共 {len(errors)} 个错误：")
    for n, tb in errors:
        print(f"\n--- {n} ---")
        print(tb)
    sys.exit(1)
else:
    print("✅ 全部通过，无错误")
