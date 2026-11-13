"""L1 多因子层 v3：动量/5日反转/波动率/换手率及其变化/MACD/KDJ/估值(EP、BP)/财务质量 + 行业中性化 + 相关性去冗余 + 20日IC均值配权。

流程（全部离线可复现）：
1. 每只候选股按日线向量化算出因子时间序列（不逐点重算，速度快）
2. 同一交易日内，各因子在【同行业内部】转百分位（行业中性化）；
   行业成分不足3只时退回全截面百分位
3. 逐交易日计算横截面 Spearman IC（因子 vs 未来5日收益）→ IC时间序列
4. 相关性去冗余：同一交易日截面 |相关| >= CORR_MAX(0.7) 的因子只留 |IC| 更强的
5. 权重 = 最近 IC_AVG_N(20) 个交易日 IC 的均值；|均值| < IC_MIN(0.03) 的因子剔除，
   剩余按 |IC| 归一化。符号由数据决定（IC为负自动变成反转因子）
6. 全部无效 → 返回空权重，上层评分自动回退原版（安全兜底）

已知局限：EP/BP 取当前快照估值，非历史时点估值（免费数据拿不到时点PE/PB），
所以估值因子的IC只反映"当前便宜股在历史上是否更抗跌"，不宜过度解读。
"""
import numpy as np
import pandas as pd
from pathlib import Path

from indicators import add_kdj, add_macd

# ========== 自动进化配置（每天 evolution.py 可能改这个文件） ==========
_CFG_PATH = Path(__file__).parent / "strategy_config.json"
def _cfg() -> dict:
    if _CFG_PATH.exists():
        try:
            return json.load(open(_CFG_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}

import json  # noqa: E402

FACTOR_COLS = ["动量20", "反转5", "波动率20", "换手20", "换手变化", "MACD柱", "KDJ_J", "EP", "BP"]
# 基本面因子（需要 Baostock 历史财报数据，避免前视偏差）
# IC 检验自动决定要不要进评分；Baostock 挂了自动降级（留 NaN，因子跳过）
FUND_FACTORS = ["ROE", "净利润增速", "营收增速", "净利率", "现金流质量"]
ALL_FACTOR_COLS = FACTOR_COLS + FUND_FACTORS
# 以下参数优先读 strategy_config.json，读不到用默认值（v1.4 上线时的硬编码）
IC_MIN = float(_cfg().get("因子", {}).get("IC门槛", 0.03))
IC_AVG_N = 20        # 用最近20个交易日的截面IC均值定权（降权重抖动）
FWD_DAYS = 5         # 检验因子对未来5日收益的预测力
WARMUP = 60          # 前60根K线不参与采样（保证因子窗口完整）
MIN_XSEC = 15        # 单个交易日截面至少15只股票才算IC
MIN_IND = 3          # 同行业至少3只才做组内中性化，否则退回全截面
DECAY_DAYS = int(_cfg().get("因子", {}).get("衰减天数", 30))
CORR_MAX = float(_cfg().get("因子", {}).get("相关性阈值", 0.70))   # 两因子截面相关超过它即剔除冗余的那个
# 每期最多用几个因子定权：样本外验证显示因子普遍弱且每期轮换，
# 保留过多噪声因子会稀释有效信号（2因子累计超额 +149pp vs 不限制 +83pp）
MAX_FACTORS = int(_cfg().get("因子", {}).get("最多因子数", 2))


def factor_series(h: pd.DataFrame, pe: float = None, pb: float = None,
                  fundamental: pd.DataFrame = None) -> pd.DataFrame:
    """把一只股票的日线转成因子时间序列（向量化，含未来5日收益标签）。
    fundamental: baostock 财报 DataFrame（列：pubDate, ROE, 净利润增速, 营收增速, 净利率, 现金流质量）
                 按 pubDate 向后填充——因子只用发布日之后的基本面数据，消除前视偏差。"""
    h = h.sort_values("日期").reset_index(drop=True).copy()
    h = add_macd(h)
    if {"最高", "最低"}.issubset(h.columns):
        h = add_kdj(h)
    else:
        h["J"] = np.nan
    close = h["收盘"]
    out = pd.DataFrame({"日期": pd.to_datetime(h["日期"])})
    out["动量20"] = close / close.shift(20) - 1
    # 5日反转：过去5日跌幅越大值越高（方向由IC符号最终决定）
    out["反转5"] = -(close / close.shift(5) - 1)
    out["波动率20"] = close.pct_change().rolling(20).std()
    turn = h["换手率"] if "换手率" in h.columns else None
    out["换手20"] = turn.rolling(20).mean() if turn is not None else np.nan
    # 换手率变化：近5日均换手 / 近20日均换手 - 1，>0 表示近期交投升温
    if turn is not None:
        out["换手变化"] = turn.rolling(5).mean() / (turn.rolling(20).mean() + 1e-12) - 1
    else:
        out["换手变化"] = np.nan
    # MACD柱除以价格消除量纲；KDJ_J保留原始J值，方向均由样本外IC决定
    out["MACD柱"] = (h["MACD"] / close.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
    out["KDJ_J"] = h["J"].replace([np.inf, -np.inf], np.nan)
    # 估值因子用倒数：越大=越便宜，方向与IC符号一致
    out["EP"] = 1.0 / pe if (pe and pe > 0) else np.nan
    out["BP"] = 1.0 / pb if (pb and pb > 0) else np.nan
    # 基本面因子：用 merge_asof 按 pubDate 往后填
    if fundamental is not None and len(fundamental) > 0:
        try:
            out = out.sort_values("日期").reset_index(drop=True)
            fund_cols = [c for c in FUND_FACTORS if c in fundamental.columns] + ["pubDate"]
            f = fundamental[fund_cols].sort_values("pubDate").drop_duplicates("pubDate")
            # 统一 dtype：pandas 3.1 要求 merge_asof 两边 datetime 精度完全一致
            out["日期"] = pd.to_datetime(out["日期"]).astype("datetime64[ns]")
            f["pubDate"] = pd.to_datetime(f["pubDate"]).astype("datetime64[ns]")
            # merge_asof：out 的日期 >= f 的 pubDate，取最近的那条
            merged = pd.merge_asof(out, f, left_on="日期", right_on="pubDate",
                                   direction="backward")
            for fc in FUND_FACTORS:
                if fc in merged.columns:
                    out[fc] = merged[fc]
        except Exception:
            for fc in FUND_FACTORS:
                out[fc] = np.nan
    else:
        for fc in FUND_FACTORS:
            out[fc] = np.nan
    out["未来5日收益"] = close.shift(-FWD_DAYS) / close - 1
    return out


def latest_factors(h: pd.DataFrame, pe: float = None, pb: float = None,
                    fundamental: pd.DataFrame = None) -> dict:
    """最新一个交易日的因子值（给候选股打分用）。"""
    s = factor_series(h, pe, pb, fundamental=fundamental)[ALL_FACTOR_COLS]
    last = s.ffill().iloc[-1] if len(s) else pd.Series(dtype=float)
    return {c: float(last.get(c, np.nan)) for c in ALL_FACTOR_COLS}


def build_panel(hists: dict, meta: dict, fundamentals: dict = None) -> pd.DataFrame:
    """把多只股票的因子序列拼成面板（长表），供逐日横截面IC计算。
    hists: {代码: 历史DataFrame}
    meta: {代码: {"行业":..., "PE":..., "PB":...}}
    fundamentals: {代码: baostock财报DataFrame}（可选，默认 None，基本面因子全为 NaN 自动跳过）"""
    fundamentals = fundamentals or {}
    frames = []
    for code, h in hists.items():
        m = meta.get(code) or {}
        fund = fundamentals.get(code)
        s = factor_series(h, m.get("PE"), m.get("PB"), fundamental=fund)
        s = s.iloc[WARMUP:len(s) - FWD_DAYS]
        if len(s) == 0:
            continue
        s["代码"] = code
        s["行业"] = m.get("行业") or "未分类"
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def neutralize(panel: pd.DataFrame, factor_cols: list[str] | None = None) -> pd.DataFrame:
    """行业中性化：同交易日内每个因子在行业内部转百分位（-0.5~+0.5）。
    factor_cols: 指定要中性化的因子列表，默认 ALL_FACTOR_COLS 里有数据的。
    行业成分不足 MIN_IND 只时，退回该交易日全截面百分位。"""
    if len(panel) == 0:
        return panel
    df = panel.copy()
    ind = df["行业"].fillna("未分类")
    cols = factor_cols or [f for f in ALL_FACTOR_COLS if f in df.columns]
    for f in cols:
        if f not in df.columns:
            continue
        in_ind = df.groupby([df["日期"], ind])[f].rank(pct=True)
        cnt = df.groupby([df["日期"], ind])[f].transform("count")
        whole = df.groupby(df["日期"])[f].rank(pct=True)
        df[f + "_n"] = np.where(cnt >= MIN_IND, in_ind, whole) - 0.5
    return df


def ic_series(npanel: pd.DataFrame) -> pd.DataFrame:
    """逐交易日的横截面IC序列：index=日期，columns=因子。"""
    rows = {}
    for d, g in npanel.groupby("日期"):
        if len(g) < MIN_XSEC:
            continue
        row = {}
        for f in ALL_FACTOR_COLS:
            c = f + "_n"
            if c not in g.columns:
                continue
            d2 = g[[c, "未来5日收益"]].dropna()
            # 基本面因子可能当天全空（不是每个季度都有财报），跳过
            if len(d2) < MIN_XSEC * 0.5:
                continue
            row[f] = (d2[c].rank().corr(d2["未来5日收益"].rank())
                      if len(d2) >= MIN_XSEC else np.nan)
        if row:
            rows[d] = row
    return pd.DataFrame(rows).T.sort_index() if rows else pd.DataFrame()


def ic_summary(ics: pd.DataFrame) -> tuple:
    """最近IC_AVG_N个交易日的IC均值与ICIR(均值/标准差)，ICIR越高越稳定。"""
    if len(ics) == 0:
        return {}, {}
    tail = ics.tail(IC_AVG_N)
    ic = {f: float(tail[f].mean()) for f in tail.columns if tail[f].notna().any()}
    ir = {}
    for f in tail.columns:
        sd = tail[f].std()
        ir[f] = float(tail[f].mean() / sd) if sd and not np.isnan(sd) and sd > 0 else 0.0
    return ic, ir


def cap_weights(w: dict, n: int | None = None) -> dict:
    """只保留 |权重| 最大的前 n 个因子并重新归一化（n 默认取 MAX_FACTORS）。
    因子普遍弱且每期轮换时，多保留因子等于把噪声一起加权，会稀释有效信号。"""
    limit = MAX_FACTORS if n is None else int(n)
    if limit <= 0 or len(w) <= limit:
        return dict(w)
    top = sorted(w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:limit]
    s = sum(abs(v) for _, v in top)
    return {f: v / s for f, v in top} if s > 0 else {}


def ic_weights(ic: dict, keep: set | None = None) -> dict:
    """|IC|<门槛剔除；剩余按 |IC| 归一化，保留符号（负IC=反转因子）。
    keep: 相关性去冗余后允许保留的因子集合（None=不剔除）。
    最后截断到 MAX_FACTORS 个，避免弱因子噪声稀释信号。
    注：曾试过按 ICIR 排序选因子，样本外未通过（见 validate 记录），故仍按 |IC|。"""
    valid = {f: v for f, v in ic.items() if abs(v) >= IC_MIN}
    if keep is not None:
        valid = {f: v for f, v in valid.items() if f in keep}
    s = sum(abs(v) for v in valid.values())
    w = {f: v / s for f, v in valid.items()} if s > 0 else {}
    return cap_weights(w)


def prune_correlated(npanel: pd.DataFrame, ic: dict,
                     threshold: float | None = None,
                     lookback: int = IC_AVG_N) -> set:
    """相关性去冗余：同一交易日的横截面暴露高度相关的两个因子只留一个。
    按 |IC| 从大到小贪心保留（IC 更强的优先），|相关| >= threshold 的后来者剔除。
    相关性用最近 lookback 个交易日的截面相关矩阵均值，反映当前而非远古的因子结构。
    返回保留因子名集合。"""
    thr = CORR_MAX if threshold is None else threshold
    cand = [f for f in ic if f + "_n" in npanel.columns and not pd.isna(ic[f])]
    if len(cand) < 2:
        return set(cand)
    dates = sorted(npanel["日期"].unique())[-lookback:]
    mats = []
    for d in dates:
        g = npanel[npanel["日期"] == d]
        sub = g[[f + "_n" for f in cand]].dropna()
        if len(sub) < MIN_XSEC:
            continue
        mats.append(sub.corr())
    if not mats:
        return set(cand)
    avg = sum(mats) / len(mats)
    kept: list[str] = []
    for f in sorted(cand, key=lambda x: abs(ic[x]), reverse=True):
        if any(abs(avg.loc[f, k]) >= thr for k in kept
               if f in avg.index and k in avg.columns):
            continue
        kept.append(f)
    return set(kept)


def neutral_score(values: pd.Series, industries: pd.Series) -> pd.Series:
    """打分用：单个交易日的行业中性百分位（与IC检验同口径）。"""
    in_ind = values.groupby(industries.fillna("未分类")).rank(pct=True)
    cnt = values.groupby(industries.fillna("未分类")).transform("count")
    whole = values.rank(pct=True)
    return pd.Series(np.where(cnt >= MIN_IND, in_ind, whole), index=values.index)
