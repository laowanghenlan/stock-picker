"""L1 多因子层 v2：动量/波动率/换手/估值(EP、BP) + 行业中性化 + 20日IC均值配权。

流程（全部离线可复现）：
1. 每只候选股按日线向量化算出因子时间序列（不逐点重算，速度快）
2. 同一交易日内，各因子在【同行业内部】转百分位（行业中性化）；
   行业成分不足3只时退回全截面百分位
3. 逐交易日计算横截面 Spearman IC（因子 vs 未来5日收益）→ IC时间序列
4. 权重 = 最近 IC_AVG_N(20) 个交易日 IC 的均值；|均值| < IC_MIN(0.03) 的因子剔除，
   剩余按 |IC| 归一化。符号由数据决定（IC为负自动变成反转因子）
5. 全部无效 → 返回空权重，上层评分自动回退原版（安全兜底）

已知局限：EP/BP 取当前快照估值，非历史时点估值（免费数据拿不到时点PE/PB），
所以估值因子的IC只反映"当前便宜股在历史上是否更抗跌"，不宜过度解读。
"""
import numpy as np
import pandas as pd

FACTOR_COLS = ["动量20", "波动率20", "换手20", "EP", "BP"]
IC_MIN = 0.03        # 20日IC均值的有效性门槛
IC_AVG_N = 20        # 用最近20个交易日的截面IC均值定权（降权重抖动）
FWD_DAYS = 5         # 检验因子对未来5日收益的预测力
WARMUP = 60          # 前60根K线不参与采样（保证因子窗口完整）
MIN_XSEC = 15        # 单个交易日截面至少15只股票才算IC
MIN_IND = 3          # 同行业至少3只才做组内中性化，否则退回全截面


def factor_series(h: pd.DataFrame, pe: float = None, pb: float = None) -> pd.DataFrame:
    """把一只股票的日线转成因子时间序列（向量化，含未来5日收益标签）。"""
    close = h["收盘"]
    out = pd.DataFrame({"日期": pd.to_datetime(h["日期"])})
    out["动量20"] = close / close.shift(20) - 1
    out["波动率20"] = close.pct_change().rolling(20).std()
    out["换手20"] = (h["换手率"].rolling(20).mean() if "换手率" in h.columns
                     else np.nan)
    # 估值因子用倒数：越大=越便宜，方向与IC符号一致
    out["EP"] = 1.0 / pe if (pe and pe > 0) else np.nan
    out["BP"] = 1.0 / pb if (pb and pb > 0) else np.nan
    out["未来5日收益"] = close.shift(-FWD_DAYS) / close - 1
    return out


def latest_factors(h: pd.DataFrame, pe: float = None, pb: float = None) -> dict:
    """最新一个交易日的因子值（给候选股打分用）。"""
    s = factor_series(h, pe, pb)[FACTOR_COLS]
    last = s.ffill().iloc[-1] if len(s) else pd.Series(dtype=float)
    return {c: float(last.get(c, np.nan)) for c in FACTOR_COLS}


def build_panel(hists: dict, meta: dict) -> pd.DataFrame:
    """把多只股票的因子序列拼成面板（长表），供逐日横截面IC计算。
    hists: {代码: 历史DataFrame}；meta: {代码: {"行业":..., "PE":..., "PB":...}}"""
    frames = []
    for code, h in hists.items():
        m = meta.get(code) or {}
        s = factor_series(h, m.get("PE"), m.get("PB"))
        s = s.iloc[WARMUP:len(s) - FWD_DAYS]
        if len(s) == 0:
            continue
        s["代码"] = code
        s["行业"] = m.get("行业") or "未分类"
        frames.append(s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def neutralize(panel: pd.DataFrame) -> pd.DataFrame:
    """行业中性化：同交易日内每个因子在行业内部转百分位（-0.5~+0.5）。
    行业成分不足MIN_IND只时，退回该交易日全截面百分位，避免样本被大量丢弃。"""
    if len(panel) == 0:
        return panel
    df = panel.copy()
    ind = df["行业"].fillna("未分类")
    for f in FACTOR_COLS:
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
        for f in FACTOR_COLS:
            c = f + "_n"
            if c not in g.columns:
                continue
            d2 = g[[c, "未来5日收益"]].dropna()
            # Spearman = 秩上的皮尔逊相关，避免依赖 scipy
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


def ic_weights(ic: dict) -> dict:
    """|IC|<门槛剔除；剩余按 |IC| 归一化，保留符号（负IC=反转因子）。"""
    valid = {f: v for f, v in ic.items() if abs(v) >= IC_MIN}
    s = sum(abs(v) for v in valid.values())
    return {f: v / s for f, v in valid.items()} if s > 0 else {}


def neutral_score(values: pd.Series, industries: pd.Series) -> pd.Series:
    """打分用：单个交易日的行业中性百分位（与IC检验同口径）。"""
    in_ind = values.groupby(industries.fillna("未分类")).rank(pct=True)
    cnt = values.groupby(industries.fillna("未分类")).transform("count")
    whole = values.rank(pct=True)
    return pd.Series(np.where(cnt >= MIN_IND, in_ind, whole), index=values.index)
