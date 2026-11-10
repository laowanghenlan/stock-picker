"""因子自学习层：复盘反哺权重 + 因子衰减监控。

两件事：
1. 复盘反哺（blend_ic_weights）
   - 每次复盘出了实盘盈亏，计算"实盘IC"（因子值 vs 实际未来收益的rank corr）
   - 与历史IC按 7:3 加权融合（历史更重、实盘轻但只反映近期真实表现）
   - 融合后的权重 = 原来的 ic_weights() 流程，但 ic 输入换成 blended

2. 因子衰减监控（factor_health）
   - 把每日IC序列存磁盘（FACTOR_HIST_FILE = 因子历史IC.csv）
   - 滚动检查：连续 DECAY_DAYS(30) 天 |IC| < IC_MIN → 标记"衰减"
   - 衰减因子从权重里临时剔除（保留数据，IC恢复自动归队）
   - 返回健康报告：{factor: {"ic": float, "ic_ir": float, "decay_days": int, "healthy": bool}}

两个模块都无IO重依赖（复盘反哺用推送记录.csv、因子衰减用verify_impl已经算出的ic_df），
失败不抛异常、返回空dict/空DataFrame让上层回退原版权重。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# 与 factors.py 对齐（import 会失败时 fallback 内置值）
try:
    from factors import IC_MIN, FACTOR_COLS, FWD_DAYS
except Exception:
    IC_MIN = 0.03
    FACTOR_COLS = ["动量20", "波动率20", "换手20", "EP", "BP"]
    FWD_DAYS = 5

DECAY_DAYS = 30       # 连续 N 天 |IC| < IC_MIN → 判为衰减
MIN_ACTUAL_SAMPLES = 10   # 实盘IC至少要这么多推荐记录才算有效

# ===== 动态融合比例：根据实盘样本量自动调整历史 vs 实盘权重 =====
def blend_ratios(n_actual: int) -> tuple[float, float]:
    """n_actual 是参与复盘反哺的实笔数。
       < 30 笔: 历史 70% + 实盘 30%（保守，实盘样本小容易抖）
     30~60 笔: 历史 50% + 实盘 50%（平衡）
       > 60 笔: 历史 30% + 实盘 70%（信任实盘、历史只做锚点）
       < MIN_ACTUAL_SAMPLES: (1.0, 0.0) 实盘样本不足，完全用历史
    """
    if n_actual < MIN_ACTUAL_SAMPLES:
        return 1.0, 0.0
    if n_actual < 30:
        return 0.7, 0.3
    if n_actual < 60:
        return 0.5, 0.5
    return 0.3, 0.7


# ===== 策略熔断（实盘风控的最后一道防线） =====
CB_LEVELS = {0: "正常", 1: "降半仓（连亏5笔）", 2: "停策略（连亏10笔）",
             3: "降半仓（累计回撤超15%）", 4: "停策略（累计回撤超25%）"}


def circuit_breaker(review_df: pd.DataFrame) -> tuple[int, str, dict]:
    """返回 (熔断等级 0~4, 原因说明, 诊断dict)。完全不抛异常。"""
    diag = {"连亏笔数": 0, "累计盈亏%": 0.0, "累计回撤%": 0.0}
    if review_df is None or len(review_df) == 0:
        return 0, "无复盘数据，熔断正常", diag
    done = review_df[review_df.get("_done",
                                   (review_df["结果"].str.startswith("🎯") |
                                    review_df["结果"].str.startswith("🛑") |
                                    review_df["结果"].str.startswith("🏁"))
                                   if "结果" in review_df.columns else False)]
    if len(done) < 5:
        return 0, f"仅 {len(done)} 笔已出结果，熔断正常", diag

    # 连亏计数（按日期升序，从最新往前找连续亏损）
    by_date = done.sort_values("日期")
    consec_loss = 0
    for pnl in reversed(by_date["盈亏%"].values):
        if float(pnl) < 0:
            consec_loss += 1
        else:
            break
    diag["连亏笔数"] = consec_loss

    # 累计盈亏 + 最大回撤（简单算：累计盈亏从0开始逐日叠加，找峰值到谷底最大跌幅）
    cum = (1 + by_date["盈亏%"] / 100).cumprod()
    peak = cum.cummax()
    dd = ((cum - peak) / peak).min() * 100  # 负数，数值越小回撤越大
    cum_ret = (cum.iloc[-1] - 1) * 100
    diag["累计盈亏%"] = round(cum_ret, 2)
    diag["累计回撤%"] = round(dd, 2)

    if consec_loss >= 10:
        return 2, f"连亏 {consec_loss} 笔，触发停策略（建议手动检查）", diag
    if consec_loss >= 5:
        return 1, f"连亏 {consec_loss} 笔，触发降半仓", diag
    if dd <= -25:
        return 4, f"累计回撤 {dd:+.1f}%，触发停策略（建议手动检查）", diag
    if dd <= -15:
        return 3, f"累计回撤 {dd:+.1f}%，触发降半仓", diag
    return 0, f"正常（连亏{consec_loss}笔，累计{cum_ret:+.1f}%，最大回撤{dd:+.1f}%）", diag




# 因子历史IC存储路径（与推送记录.csv同目录，GitHub Actions每天自动commit回）
BASE_DIR = Path(__file__).parent
FACTOR_HIST_FILE = BASE_DIR / "因子历史IC.csv"


# =============================================================
# 1. 复盘反哺权重
# =============================================================

def actual_ic_from_review(review_df: pd.DataFrame, picks_with_factors: pd.DataFrame) -> dict:
    """用复盘记录反推实盘IC：因子值 vs 推荐后的实际盈亏%。

    review_df 来自 review_impl()，必须有列：日期、代码、盈亏%、_done
    picks_with_factors 是对应那天 verify_impl 产生的 DataFrame，
      至少包含 代码 + 所有 FACTOR_COLS 因子列（行业中性化后的值）

    返回 {factor_name: rank_corr(float)}，样本不足返回 {}。
    """
    if review_df is None or len(review_df) == 0:
        return {}
    if picks_with_factors is None or len(picks_with_factors) == 0:
        return {}

    # 只看已出结果的（排除持有中）
    done = review_df[review_df.get("_done", review_df["结果"].str.startswith("🎯") |
                                   review_df["结果"].str.startswith("🛑") |
                                   review_df["结果"].str.startswith("🏁"))]
    if len(done) < MIN_ACTUAL_SAMPLES:
        return {}

    # 用代码 + 日期 匹配：推送记录.csv 里的 推荐日期 那天 verify_impl 才算了因子
    # 简化：只按代码匹配（不区分哪天的因子快照），因为推送记录只取 20 天内、因子值相对稳定
    merged = done.merge(picks_with_factors[["代码"] + [c for c in FACTOR_COLS if c in picks_with_factors.columns]],
                        on="代码", how="left")
    merged = merged.dropna(subset=["盈亏%"] + [c for c in FACTOR_COLS if c in merged.columns])
    if len(merged) < MIN_ACTUAL_SAMPLES:
        return {}

    out = {}
    ret = merged["盈亏%"].rank()
    for f in FACTOR_COLS:
        if f not in merged.columns:
            continue
        v = merged[f].rank()
        if v.std() < 1e-9:
            continue
        corr = float(v.corr(ret))
        if not np.isnan(corr):
            out[f] = corr
    return out


def blended_ic_weights(historical_ic: dict, actual_ic: dict, n_actual: int = 0) -> tuple[dict, dict, tuple[float, float]]:
    """融合历史IC和实盘IC，返回 ({factor: blended_ic}, {factor: weight}, (hist_ratio, actual_ratio))。

    融合比例根据 n_actual（实笔数）自动动态调整（见 blend_ratios）。
    """
    hist_r, act_r = blend_ratios(n_actual)
    blended = {}
    all_factors = set(list(historical_ic.keys()) + list(actual_ic.keys()))
    for f in all_factors:
        h = historical_ic.get(f, 0.0)
        a = actual_ic.get(f, 0.0)
        blended[f] = hist_r * h + act_r * a

    # 剔除 + 归一化（与 factors.ic_weights 口径一致）
    valid = {f: v for f, v in blended.items() if abs(v) >= IC_MIN}
    s = sum(abs(v) for v in valid.values())
    weights = {f: v / s for f, v in valid.items()} if s > 0 else {}
    return blended, weights, (hist_r, act_r)


# =============================================================
# 2. 因子衰减监控
# =============================================================

def append_factor_ic_history(ic_row: dict, date_str: str) -> None:
    """把一天的 IC 结果追加到因子历史CSV（原子写）。"""
    if not ic_row:
        return
    row = {"日期": date_str}
    row.update({f: v for f, v in ic_row.items() if not isinstance(v, (dict, list))})
    try:
        new_df = pd.DataFrame([row])
        if FACTOR_HIST_FILE.exists():
            old = pd.read_csv(FACTOR_HIST_FILE, dtype={"日期": str})
            df = pd.concat([old, new_df], ignore_index=True)
        else:
            df = new_df
        # 去重（同一天重算只留最新）
        df = df.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")
        df.to_csv(FACTOR_HIST_FILE, index=False)
    except Exception:
        pass   # 衰减监控失败不影响主流程


def factor_health(ic_row: dict | None = None) -> pd.DataFrame:
    """读因子历史CSV，返回衰减诊断表（empty时返回空DataFrame）。

    列：因子 | 最新IC | ICIR(20日) | 连续失效天数 | 健康状态
    健康状态：健康 / 衰减(连续>=DECAY_DAYS天|IC|<IC_MIN) / 新上线(历史<DECAY_DAYS)
    """
    if not FACTOR_HIST_FILE.exists():
        return pd.DataFrame()
    try:
        hist = pd.read_csv(FACTOR_HIST_FILE, dtype={"日期": str})
    except Exception:
        return pd.DataFrame()

    # 补一行当日IC（如果有、且没存过）
    if ic_row:
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        row = {"日期": today}
        row.update({f: v for f, v in ic_row.items() if not isinstance(v, (dict, list))})
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        hist = hist.drop_duplicates(subset=["日期"], keep="last").sort_values("日期")

    factor_cols = [c for c in FACTOR_COLS if c in hist.columns]
    if not factor_cols:
        return pd.DataFrame()

    rows = []
    for f in factor_cols:
        s = hist[f].dropna()
        if len(s) == 0:
            rows.append({"因子": f, "最新IC": np.nan, "ICIR(20日)": np.nan,
                         "连续失效天数": 0, "健康状态": "新上线"})
            continue
        latest_ic = float(s.iloc[-1])
        # ICIR 20日
        tail = s.tail(20)
        sd = tail.std()
        icir = float(tail.mean() / sd) if sd and not np.isnan(sd) and sd > 0 else 0.0
        # 连续失效天数（从最新往前数）
        consecutive_fail = 0
        for val in reversed(s.values):
            if abs(float(val)) < IC_MIN:
                consecutive_fail += 1
            else:
                break
        status = "衰减" if consecutive_fail >= DECAY_DAYS else "健康" if consecutive_fail == 0 else "新上线"
        rows.append({"因子": f, "最新IC": round(latest_ic, 4), "ICIR(20日)": round(icir, 3),
                     "连续失效天数": consecutive_fail, "健康状态": status})
    return pd.DataFrame(rows)


def filter_decayed_factors(ic_weights: dict, health_df: pd.DataFrame) -> dict:
    """把衰减因子从权重里临时剔除。"""
    if health_df is None or len(health_df) == 0:
        return ic_weights
    decayed = set(health_df.loc[health_df["健康状态"] == "衰减", "因子"].tolist())
    filtered = {f: w for f, w in ic_weights.items() if f not in decayed}
    s = sum(abs(w) for w in filtered.values())
    return {f: w / s for f, w in filtered.items()} if s > 0 else {}
