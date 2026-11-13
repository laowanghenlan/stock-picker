"""L2 机器学习层：LightGBM（优先）/ XGBoost（备用）学习因子 → 未来5日收益的非线性关系。

定位：IC 动态定权之外的"加分项"，不替代原有趋势/回测/排雷纪律。
- 训练面板复用 factors.build_panel（已剔除每只股票最后 FWD_DAYS 根，标签不伸入预测日）
- 时间顺序切分 + FWD_DAYS 天 embargo，验证段只用于报告样本外 IC
- 样本外 IC 低于门槛（默认0.02）→ 返回结果但上层不给分，避免模型失灵时拖累选股
- 未安装 lightgbm/xgboost 时 available() 返回 None，整个流程自动跳过（安全降级）
"""
import numpy as np
import pandas as pd

import factors

MIN_ROWS = 600          # 训练样本最少行数（候选池太小时不训练，防过拟合）
MIN_DATES = 40          # 最少交易日数
EMBARGO = factors.FWD_DAYS   # 训练/验证之间留5日，防止5日收益标签跨段重叠
OOF_IC_MIN = 0.02       # 样本外IC低于此值不给分
LABEL_CLIP = 0.5        # 标签裁剪到±50%，避免单次暴涨暴跌主导训练


def available() -> str | None:
    """返回可用的后端：lightgbm / xgboost / None"""
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except Exception:
        pass
    try:
        import xgboost  # noqa: F401
        return "xgboost"
    except Exception:
        return None


def _make_model(backend: str, seed: int = 42):
    """小容量模型：候选池只有几十只股票，复杂模型必然过拟合"""
    if backend == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=0.5,
                             random_state=seed, n_jobs=2, verbose=-1)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4,
                        min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                        reg_lambda=5.0, reg_alpha=0.5, random_state=seed,
                        n_jobs=2, verbosity=0)


def fit_model(npanel: pd.DataFrame, feature_cols: list[str] | None = None) -> dict | None:
    """只用给定历史面板训练；返回的模型可预测后续多个截面，供滚动样本外回测复用。"""
    backend = available()
    if backend is None or len(npanel) == 0:
        return None
    feats = feature_cols or [f + "_n" for f in factors.ALL_FACTOR_COLS if f + "_n" in npanel.columns]
    feats = [f for f in dict.fromkeys(feats)
             if f in npanel.columns and npanel[f].notna().mean() > 0.3]
    if len(feats) < 3:
        return None
    df = npanel[["日期", "代码", "未来5日收益"] + feats].dropna(subset=["未来5日收益"]).copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    dates = np.array(sorted(df["日期"].unique()))
    if len(df) < MIN_ROWS or len(dates) < MIN_DATES:
        return None

    # 保留因子列名，训练与预测使用同一特征顺序，也避免 sklearn 特征名警告
    X = df[feats].astype(float).fillna(0.0)
    y = df["未来5日收益"].astype(float).clip(-LABEL_CLIP, LABEL_CLIP)

    # 样本外验证：最后20%的交易日，训练/验证之间留 embargo
    split = int(len(dates) * 0.8)
    tr_d, va_d = dates[:max(1, split - EMBARGO)], dates[split:]
    if len(tr_d) < MIN_DATES * 0.5 or len(va_d) < 5:
        return None
    tr = df["日期"].isin(tr_d).values
    va = df["日期"].isin(va_d).values
    model = _make_model(backend)
    model.fit(X[tr], y[tr])
    pv = model.predict(X[va])
    chk = df.loc[va, ["日期"]].copy()
    chk["p"], chk["y"] = pv, y[va]
    ics = []
    for _, g in chk.groupby("日期"):
        if len(g) >= factors.MIN_XSEC:
            ics.append(g["p"].rank().corr(g["y"].rank()))
    oof_ic = float(np.nanmean(ics)) if ics else float("nan")

    # 最终模型只用传入的全部历史重训，不接触后续测试期
    model.fit(X, y)
    imp = sorted(zip(feats, np.asarray(model.feature_importances_, dtype=float)),
                 key=lambda x: -x[1])[:5]
    return {"backend": backend, "_model": model,
            "oof_ic": oof_ic, "oof_pass": bool(oof_ic == oof_ic and oof_ic >= OOF_IC_MIN),
            "n_rows": int(len(df)), "n_dates": int(len(dates)),
            "features": feats, "importance": imp}


def predict_scores(fitted: dict, today: pd.DataFrame) -> pd.Series:
    """预测一个或多个截面，保留原行索引；逐日百分位由调用方按各自截面计算。"""
    X = today[fitted["features"]].astype(float).fillna(0.0)
    return pd.Series(fitted["_model"].predict(X), index=today.index, dtype=float)


def train_predict(npanel: pd.DataFrame, today: pd.DataFrame,
                  feature_cols: list[str] | None = None) -> dict | None:
    """训练并预测当日截面；样本外IC不达标时仍返回诊断，但上层不给分。"""
    if len(today) == 0:
        return None
    feats = feature_cols or [f + "_n" for f in factors.ALL_FACTOR_COLS
                             if f + "_n" in npanel.columns and f + "_n" in today.columns]
    fitted = fit_model(npanel, feats)
    if fitted is None:
        return None
    scores = predict_scores(fitted, today).rank(pct=True)
    return {**{k: v for k, v in fitted.items() if k != "_model"},
            "scores": dict(zip(today["代码"].astype(str), scores.round(4)))}
