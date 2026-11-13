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
check("import ml_model", lambda: __import__("ml_model"))
import app
check("import app", lambda: None)

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
    for c in ["结果", "盈亏%", "股数", "盈亏额", "_盈亏额"]:
        if c not in rv.columns:
            errors.append(("review列缺失", c))
    app.review_conclusion(rv)

# 用固定复盘样本核对每笔金额、股数、批次只数和邮件摘要，不依赖历史台账内容
import daily_push

def review_fields():
    day0 = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    day1 = pd.Timestamp.today().strftime("%Y-%m-%d")
    rows = []
    for code, day, shares, p0, p1, done, result in [
        ("000001", day0, 100, 10.0, 11.0, True, "🏁到期盈利"),
        ("000002", day0, 200, 10.0, 9.0, True, "🛑止损保命"),
        ("000003", day1, 100, 10.0, 10.5, False, "⏳持有中(第1天)"),
    ]:
        rows.append({"日期": day, "推送时间": "09:00:00", "_batch": day + " 09:00:00",
                     "代码": code, "名称": code, "推荐价": p0, "现价": p1, "股数": shares,
                     "盈亏%": (p1 / p0 - 1) * 100, "盈亏额": (p1 - p0) * shares,
                     "_盈亏额": (p1 - p0) * shares, "_done": done, "结果": result,
                     "同期沪深300%": 0.0})
    sample = pd.DataFrame(rows)
    r = daily_push.build_review(sample)
    assert r["win_n"] == 1 and r["win_amt"] == 100
    assert r["loss_n"] == 1 and r["loss_amt"] == -200
    assert r["pnl"] == -100
    assert r["holding"] == 1 and r["holding_pnl"] == 50
    assert r["total"] == 3 and r["batches"] == 2 and r["avg_per_batch"] == 1.5
    assert r["rows"][0]["股数"] == 100
    lb = daily_push.build_last_batch(sample)
    assert len(lb["rows"]) == 2 and lb["rows"][0]["股数"] == 100
    text = "\n".join(daily_push._last_text_lines(lb))
    html = daily_push._last_html(lb)
    assert "100 股" in text and "+100 元" in text and "-200 元" in text
    assert "100 股" in html and "+100 元" in html and "-200 元" in html

check("复盘金额/股数/批次与邮件明细", review_fields)

# ---------- 8. 因子 ----------
print("\n⑧ 因子自检 ...")
import factors
# 用 verify_impl 已算出的 attrs
icw = res.attrs.get("ic_weights") or {}
print(f"  ic_weights：{icw}")
check("ic_weights dict", lambda: isinstance(icw, dict))

# ---------- 9. 机器学习门控与评分 ----------
print("\n⑨ 机器学习门控与评分 ...")
import ml_model

def ml_gates():
    if ml_model.available() is None:
        print("  未安装LightGBM/XGBoost，按安全降级跳过")
        return
    import numpy as np

    def one(seed, signal):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2025-01-01", periods=140)
        codes = [f"{i:06d}" for i in range(30)]
        df = pd.MultiIndex.from_product([dates, codes], names=["日期", "代码"]).to_frame(index=False)
        for f in factors.ALL_FACTOR_COLS:
            df[f + "_n"] = rng.uniform(-0.5, 0.5, len(df))
        label = (0.04 * df["动量20_n"] + 0.02 * df["MACD柱_n"]) if signal else 0.0
        df["未来5日收益"] = label + rng.normal(0, 0.003 if signal else 0.02, len(df))
        today = df[df["日期"] == dates[-1]].drop(columns=["日期", "未来5日收益"]).copy()
        r = ml_model.train_predict(df, today)
        assert r is not None and r["oof_pass"] == signal, r
        return r["oof_ic"]

    ic_signal, ic_noise = one(42, True), one(123, False)
    print(f"  有信号IC {ic_signal:+.3f} 放行；纯噪声IC {ic_noise:+.3f} 拦截")


def ml_weighting():
    sample = pd.DataFrame({"代码": ["A", "B"], "扣费总收益": [0.2, 0.1],
                           "胜率": [60.0, 50.0], "趋势向上": [True, True],
                           "有雷": [False, False], "ML分": [0.0, 1.0]})
    ranked = app.pick_top3(sample, 2).set_index("代码")
    assert abs(ranked.loc["A", "总分"] - 80.0) < 1e-9
    assert abs(ranked.loc["B", "总分"] - 72.0) < 1e-9

check("LightGBM/XGBoost信号与噪声门控", ml_gates)
check("ML分按配置权重混入总分", ml_weighting)

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
