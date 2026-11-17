"""L2 机器学习层：LightGBM（优先）/ XGBoost（备用）学习因子 → 未来N日相对强弱的非线性关系。

定位：IC 动态定权之外的"加分项"，不替代原有趋势/回测/排雷纪律。
- 训练面板复用 factors.build_panel（已剔除每只股票最后 FWD_DAYS 根，标签不伸入预测日）
- 标签优先用"超额排名"（同日横截面百分位，剥离大盘beta）；面板没有该列时退回原始5日收益
- 时间顺序切分 + FWD_DAYS 天 embargo，验证段只用于报告样本外 IC
- 样本外 IC 低于门槛（默认0.02）→ 返回结果但上层不给分，避免模型失灵时拖累选股
- 未安装 lightgbm/xgboost 时 available() 返回 None，整个流程自动跳过（安全降级）
"""
import numpy as np
import pandas as pd
from pathlib import Path

import factors

MIN_ROWS = 600          # 训练样本最少行数（候选池太小时不训练，防过拟合）
MIN_DATES = 40          # 最少交易日数
EMBARGO = factors.FWD_DAYS   # 训练/验证之间留5日，防止5日收益标签跨段重叠
OOF_IC_MIN = 0.02       # 样本外IC低于此值不给分
LABEL_CLIP = 0.5        # 标签裁剪到±50%，避免单次暴涨暴跌主导训练

MODEL_DIR = Path(__file__).parent / "models"   # 预训练模型落盘目录（随仓库分发，云端免训练）


def _label_of(npanel: pd.DataFrame) -> tuple[str, float]:
    """选训练标签：优先横截面超额排名（剥离beta），缺失时退回原始5日收益。
    返回 (列名, 裁剪上下限)。超额排名本身就在 ±0.5 内，裁剪不改变数值。"""
    if factors.EXCESS_RANK_COL in npanel.columns and \
            npanel[factors.EXCESS_RANK_COL].notna().mean() > 0.5:
        return factors.EXCESS_RANK_COL, 0.5
    return factors.LABEL_COL, LABEL_CLIP


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
    label_col, clip = _label_of(npanel)
    if label_col not in npanel.columns:
        return None
    df = npanel[["日期", "代码", label_col] + feats].dropna(subset=[label_col]).copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期")
    dates = np.array(sorted(df["日期"].unique()))
    if len(df) < MIN_ROWS or len(dates) < MIN_DATES:
        return None

    # 保留因子列名，训练与预测使用同一特征顺序，也避免 sklearn 特征名警告
    X = df[feats].astype(float).fillna(0.0)
    y = df[label_col].astype(float).clip(-clip, clip)

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
    return {"backend": backend, "_model": model, "label": label_col,
            "oof_ic": oof_ic, "oof_pass": bool(oof_ic == oof_ic and oof_ic >= OOF_IC_MIN),
            "n_rows": int(len(df)), "n_dates": int(len(dates)),
            "features": feats, "importance": imp}


def predict_scores(fitted: dict, today: pd.DataFrame) -> pd.Series:
    """预测一个或多个截面，保留原行索引；逐日百分位由调用方按各自截面计算。"""
    X = today[fitted["features"]].astype(float).fillna(0.0)
    return pd.Series(fitted["_model"].predict(X), index=today.index, dtype=float)


def save_model(fitted: dict, name: str = "lgb") -> Path:
    """把训练好的模型落盘到 models/，供云端直接加载推理（免训练）。"""
    if fitted is None:
        raise ValueError("fitted is None")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    fitted["_pretrained"] = True
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(fitted, path, compress=3)
    return path


def load_model(name: str = "lgb") -> dict | None:
    """加载预训练模型；文件不存在或加载失败返回 None。"""
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        return None
    try:
        import joblib
        fitted = joblib.load(path)
        # 验证特征列齐全
        if "features" not in fitted or "_model" not in fitted:
            return None
        return fitted
    except Exception:
        return None


def train_predict(npanel: pd.DataFrame, today: pd.DataFrame,
                  feature_cols: list[str] | None = None,
                  use_pretrained: bool = True) -> dict | None:
    """训练并预测当日截面；样本外IC不达标时仍返回诊断，但上层不给分。

    use_pretrained=True 时优先加载 models/lgb.joblib（本地预训练的全市场模型），
    避免云端因候选股样本太少导致 IC 为负。模型不存在则现场训练。
    """
    if len(today) == 0:
        return None
    feats = feature_cols or [f + "_n" for f in factors.ALL_FACTOR_COLS
                             if f + "_n" in npanel.columns and f + "_n" in today.columns]

    fitted = None
    if use_pretrained:
        fitted = load_model("lgb")
        # 预训练模型的特征列必须与 today 对齐，否则退回现场训练
        if fitted is not None:
            missing = [f for f in fitted["features"] if f not in today.columns]
            if missing:
                fitted = None

    if fitted is None:
        fitted = fit_model(npanel, feats)
        if fitted is None:
            return None

    scores = predict_scores(fitted, today).rank(pct=True)
    return {**{k: v for k, v in fitted.items() if k != "_model"},
            "scores": dict(zip(today["代码"].astype(str), scores.round(4))),
            "pretrained": use_pretrained and fitted.get("_pretrained", False)}
