"""T+1 超短线精筛器（v1.29）
=================================================
定位：web（app.py）负责宽候选池粗筛 + 导出 CSV；本脚本负责二次精筛。
输入：web 导出的「粗筛候选池_*.csv」（至少含 代码 列，其余因子列越多越好）。
核心：含退市全A面板(1592只/136万条)训练的 LightGBM，预测次日收益截面排序；
      样本外(2025-07起)实测 top3 次日净超额 +1.558%/期、t=7.57、跑赢率68%、IC=+0.073。
口径：今收买、明收卖，已扣双边费0.15%；仅为统计优势，不构成投资建议。

用法：
  python refine_pool.py                       # 自动找最新导出的候选池 CSV
  python refine_pool.py 候选池.csv            # 指定输入
  python refine_pool.py 候选池.csv --topn 5   # 指定精筛只数
输出：精筛清单_YYYYMMDD_HHMM.csv（代码/名称/精筛分/排序/排名依据/快照字段）
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("C:/Users/Admin/WorkBuddy/2026-11-18-08-05-50")
PANEL_CACHE = WORKSPACE / "t1_panel_cache.csv"      # 训练面板（工作区，含退市全A）
FEATS = ["动量20", "反转5", "反转1", "波动20", "规模代理", "Amihud", "MACD柱", "KDJ_J"]
SPLIT = pd.Timestamp("2025-07-01")
MODEL_FILE = WORKSPACE / "refine_model.txt"          # 训练好的LGB模型(缓存,放工作区英文路径避中文写入问题)


def log(*a): print(*a, flush=True)


def find_input(arg):
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else HERE / arg
    cands = sorted(HERE.glob("粗筛候选池_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands += sorted(Path.cwd().glob("粗筛候选池_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def train_or_load():
    """加载缓存模型；无缓存则用含退市全A面板训练T+1 LightGBM并缓存。"""
    from lightgbm import LGBMRegressor, Booster
    if MODEL_FILE.exists():
        try:
            return Booster(model_file=str(MODEL_FILE))
        except Exception:
            pass
    if not PANEL_CACHE.exists():
        log(f"!! 训练面板缺失：{PANEL_CACHE}")
        log("   请先把含退市全A面板缓存放回该路径，或改用 --rule 规则精筛。")
        return None
    log(f"训练面板：{PANEL_CACHE.name}（首次训练约1-2分钟，之后读缓存）")
    p = pd.read_csv(PANEL_CACHE, parse_dates=["日期"])
    p = p.dropna(subset=FEATS + ["次日收益"]).copy()
    p = p[p["次日收益"].between(-0.11, 0.11)]
    p["y"] = p.groupby("日期")["次日收益"].rank(pct=True)
    tr = p[p["日期"] < SPLIT]
    lgb = LGBMRegressor(n_estimators=150, learning_rate=0.04, num_leaves=15,
                        min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
                        random_state=42, verbosity=-1, n_jobs=4)
    lgb.fit(tr[FEATS], tr["y"])
    lgb.booster_.save_model(str(MODEL_FILE))
    log(f"模型已缓存：{MODEL_FILE.name}")
    return lgb.booster_


def rule_score(df):
    """无模型时的降级规则：T+1验证 KDJ_J(70%)+动量20(30%)。"""
    return (df["KDJ_J"].rank(pct=True) * 0.7 + df["动量20"].rank(pct=True) * 0.3) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="候选池CSV路径（默认找最新的粗筛候选池_*.csv）")
    ap.add_argument("--topn", type=int, default=5, help="输出精筛只数（默认5）")
    ap.add_argument("--rule", action="store_true", help="强制用规则精筛（不加载ML模型）")
    a = ap.parse_args()

    inp = find_input(a.input)
    if inp is None or not inp.exists():
        log("!! 没找到候选池 CSV。请先在 web 点「导出粗筛候选池」下载，把文件放到项目目录后重试。")
        sys.exit(1)
    df = pd.read_csv(inp, dtype={"代码": str})
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    log(f"读入候选池 {len(df)} 只：{inp.name}")

    # 从面板补当日因子（导出CSV里可能已有因子列，优先用面板的最新截面保证口径一致）
    model = None if a.rule else train_or_load()
    if model is not None:
        panel = pd.read_csv(PANEL_CACHE, parse_dates=["日期"], usecols=["代码", "日期"] + FEATS) \
            if PANEL_CACHE.exists() else None
        if panel is not None:
            # 面板代码是 sh.600000 / sz.000001 格式，统一成6位数字
            panel["代码"] = panel["代码"].astype(str).str[-6:].str.zfill(6)
            latest = panel.sort_values("日期").groupby("代码").tail(1)
            fmap = latest.set_index("代码")[FEATS]
            for f in FEATS:
                df[f] = df["代码"].map(fmap[f])   # 始终用面板当日因子覆盖，保证与训练口径一致
    have = [f for f in FEATS if f in df.columns and df[f].notna().any()]
    if len(have) < 2:
        log("!! 候选池缺少技术因子列，无法精筛。请用 v1.29+ 导出的候选池 CSV。")
        sys.exit(1)

    if model is not None:
        X = df[FEATS].copy()
        for f in FEATS:
            if f not in X.columns:
                X[f] = np.nan
        df["精筛分"] = model.predict(X)
        basis = "LightGBM(T+1,含退市全A样本外+1.56%/期)"
    else:
        for f in ("KDJ_J", "动量20"):
            if f not in df.columns:
                df[f] = np.nan
        df["精筛分"] = rule_score(df)
        basis = "规则KDJ_J70%+动量20(样本外+1.48%/期)"

    df = df.sort_values("精筛分", ascending=False).reset_index(drop=True)
    df.insert(0, "精筛排名", range(1, len(df) + 1))
    out = df.head(a.topn)

    keep = ["精筛排名", "代码", "名称", "精筛分", "最新价", "现价", "今日涨跌", "隔夜涨跌%",
            "换手率", "量比", "振幅", "市盈率", "市净率", "总市值"] + FEATS
    keep = [c for c in keep if c in out.columns]
    show = out[keep].copy()
    for c in show.columns:
        if show[c].dtype.kind in "fc":
            show[c] = show[c].round(4)

    log(f"\n===== T+1 精筛 TOP {a.topn}（{basis}）=====")
    log(show.to_string(index=False))

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    outp = HERE / f"精筛清单_{ts}.csv"
    show.to_csv(outp, index=False, encoding="utf-8-sig")
    log(f"\n已保存：{outp.name}")
    log("提示：T+1=今收买明收卖，统计优势不等于必赚；涨停买不进/停牌/滑点会让实盘低于回测。")


if __name__ == "__main__":
    main()
