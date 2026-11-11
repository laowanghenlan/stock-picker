# -*- coding: utf-8 -*-
"""仓位优化模块（v1.8 新增）。

两层设计：
  1. 优先用 PyPortfolioOpt 的 EfficientFrontier 做 max_sharpe 均值-方差优化。
  2. 如果 PyPortfolioOpt 不可用（Windows 上 cvxpy 与 numpy 二进制不兼容时常见），
     自动降级为纯 numpy 解析解（max Sharpe 的闭式解 + 归一化 + 非负裁剪）。

置信度折扣：
  实盘样本不足时（n_done < 置信度阈值），把优化权重向等额权重靠拢：
    final = confidence * opt_weights + (1 - confidence) * equal_weights
  样本越少，越接近等额分配，避免小样本下 optimizer 过度自信。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

_CFG_PATH = Path(__file__).parent / "strategy_config.json"


def _cfg() -> dict:
    try:
        import json
        return json.load(open(_CFG_PATH, encoding="utf-8"))
    except Exception:
        return {}


# ---------- PyPortfolioOpt 可选导入 ----------
try:
    from pypfopt import EfficientFrontier, expected_returns, risk_models
    _HAS_PPF = True
except Exception:
    EfficientFrontier = expected_returns = risk_models = None
    _HAS_PPF = False


def _confidence(n_done: int, threshold: int) -> float:
    """置信度 = min(1.0, n_done / threshold)。样本不足时向等额权重靠拢。"""
    if threshold <= 0:
        return 1.0
    return max(0.0, min(1.0, n_done / threshold))


def _blend_with_equal(weights: dict, confidence: float) -> dict:
    """把优化权重与等额权重按置信度混合。"""
    if confidence >= 1.0:
        return weights
    n = len(weights)
    if n == 0:
        return weights
    eq = 1.0 / n
    return {k: confidence * v + (1 - confidence) * eq for k, v in weights.items()}


def _numpy_max_sharpe(mu: np.ndarray, cov: np.ndarray,
                      weight_bounds: tuple = (0.01, 0.5)) -> dict | None:
    """纯 numpy 的 max Sharpe 闭式解 + 非负约束 + 归一化。

    无约束 max Sharpe 解析解：w ∝ Σ^(-1) (μ - rf)。
    加非负约束用迭代裁剪法（对 1~20 只股票足够稳定）。
    """
    n = len(mu)
    if n < 2:
        return None
    try:
        cov_inv = np.linalg.pinv(cov)
    except Exception:
        return None
    rf = 0.02 / 252  # 日无风险利率
    excess = mu - rf
    w = cov_inv @ excess
    # 非负裁剪 + 归一化
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    # 应用单票上下限
    lo, hi = weight_bounds
    w = np.clip(w, lo, hi)
    w = w / w.sum()
    return w


def optimize_weights(codes: list[str], price_history: dict[str, pd.DataFrame],
                     expected_returns: dict[str, float],
                     n_done: int = 0) -> tuple[dict | None, str]:
    """计算最优仓位权重。

    参数：
      codes: 股票代码列表
      price_history: {代码: 历史日线 DataFrame}（需含「收盘」「日期」列）
      expected_returns: {代码: 预期收益率}（来自回测的扣费总收益%，转小数）
      n_done: 已结算实盘笔数，用于置信度折扣

    返回：(权重dict {代码: 0~1}, 使用的方法名)
          失败时返回 (None, reason)
    """
    if len(codes) < 2:
        return None, "股票数<2，无需优化"

    cfg = _cfg()
    mv_cfg = cfg.get("仓位管理", {}).get("均值方差优化", {})
    confidence_threshold = int(mv_cfg.get("置信度阈值", 30))
    confidence = _confidence(n_done, confidence_threshold)

    # 构建价格矩阵（对齐日期）
    closes = {}
    for c in codes:
        h = price_history.get(c)
        if h is None or len(h) < 60:
            continue
        s = h.sort_values("日期")["收盘"].astype(float)
        s.index = pd.to_datetime(h.sort_values("日期")["日期"])
        closes[c] = s
    if len(closes) < 2:
        return None, "有效历史数据不足2只"

    prices = pd.DataFrame(closes).dropna(how="any")
    if len(prices) < 60:
        return None, f"对齐后历史仅{len(prices)}天，不足60"

    # 预期收益向量（对齐到 prices 的列）
    use_codes = list(prices.columns)
    mu = np.array([float(expected_returns.get(c, 0.0)) for c in use_codes])
    # 预期收益转成"每笔交易"口径（回测总收益%/100），做相对排序即可，优化器会归一化
    mu = np.clip(mu, 0, None)  # 亏损预期的票不给正权重
    if mu.sum() <= 0:
        return None, "所有候选预期收益≤0，回退等额"

    # 协方差矩阵（日收益率）
    returns = prices.pct_change().dropna()
    if len(returns) < 30:
        return None, "收益率样本不足"
    cov = returns.cov().values

    # ===== 优先 PyPortfolioOpt =====
    if _HAS_PPF:
        try:
            mu_s = pd.Series(mu, index=use_codes)
            cov_df = pd.DataFrame(cov, index=use_codes, columns=use_codes)
            ef = EfficientFrontier(mu_s, cov_df, weight_bounds=(0.01, 0.5))
            raw = ef.max_sharpe(risk_free_rate=0.02 / 252)
            w = ef.clean_weights()
            w = {k: float(v) for k, v in w.items() if v > 0}
            if not w:
                raise ValueError("PPF 返回空权重")
            w = _blend_with_equal(w, confidence)
            return w, f"PyPortfolioOpt(max_sharpe) · 置信度{confidence:.0%}"
        except Exception:
            pass  # 降级到 numpy

    # ===== numpy 兜底 =====
    w_arr = _numpy_max_sharpe(mu, cov, weight_bounds=(0.01, 0.5))
    if w_arr is None:
        return None, "numpy 优化失败，回退等额"
    w = {use_codes[i]: float(w_arr[i]) for i in range(len(use_codes)) if w_arr[i] > 0}
    w = _blend_with_equal(w, confidence)
    return w, f"numpy(max_sharpe闭式解) · 置信度{confidence:.0%}"


def available() -> bool:
    """PyPortfolioOpt 是否可用（供 UI 展示用）。"""
    return _HAS_PPF


def apply_risk_constraints(weights: dict, industries: dict,
                           returns: pd.DataFrame | None = None,
                           max_name: float = 0.35, max_ind: float = 0.50,
                           vol_target: float = 0.30) -> tuple[dict, dict]:
    """组合层风控（v1.17）：单票上限 → 行业上限 → 组合波动率目标。

    参数：
      weights:    {代码: 权重}，来自均值方差 / 凯利 / 等额
      industries: {代码: 行业}
      returns:    各票日收益率矩阵（列=代码），用于估计组合波动率；None 则跳过波动率目标
      max_name:   单票最大权重（0 或 None 关闭）
      max_ind:    单行业最大权重（0 或 None 关闭）
      vol_target: 组合年化波动率目标，超过则整体等比降杠杆（留现金）

    返回 (约束后的权重dict, 诊断dict)。
    约束只会减仓不会加仓，权重和可能 < 1（差额即留存现金）。
    """
    w = {k: max(float(v), 0.0) for k, v in (weights or {}).items() if float(v) > 0}
    info = {"单票裁剪": False, "行业裁剪": False, "波动率缩放": 1.0, "组合年化波动": None}
    if not w:
        return {}, info
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}          # 先归一到满仓口径

    # 1. 单票上限
    if max_name and max_name > 0:
        for k in list(w):
            if w[k] > max_name:
                w[k] = max_name
                info["单票裁剪"] = True

    # 2. 行业上限（同行业合计超限则组内等比缩小）
    if max_ind and max_ind > 0:
        groups: dict[str, list[str]] = {}
        for k in w:
            groups.setdefault(industries.get(k) or "未分类", []).append(k)
        for _, members in groups.items():
            s = sum(w[k] for k in members)
            if s > max_ind and s > 0:
                scale = max_ind / s
                for k in members:
                    w[k] *= scale
                info["行业裁剪"] = True

    # 3. 组合波动率目标（年化）：只降杠杆，不加杠杆
    if returns is not None and vol_target and vol_target > 0:
        cols = [c for c in w if c in returns.columns]
        if len(cols) >= 2 and len(returns) >= 30:
            try:
                cov = returns[cols].cov().values
                wv = np.array([w[c] for c in cols])
                vol = float(np.sqrt(max(wv @ cov @ wv, 0.0) * 252))
                info["组合年化波动"] = round(vol, 4)
                if vol > vol_target and vol > 1e-9:
                    scale = vol_target / vol
                    for k in w:
                        w[k] *= scale
                    info["波动率缩放"] = round(scale, 3)
            except Exception:
                pass

    if sum(w.values()) > 1.0:                          # 双保险：绝不加杠杆
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
    return w, info
