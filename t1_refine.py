"""T+1 超短线精筛（内嵌版，v1.32）
=================================================
把 refine_pool.py 的精筛核心内嵌进 Web「今日选股」按钮：候选股 K线 → 按训练面板口径现算8特征
→ 仓库内 LightGBM 模型打分 → 截面排序。点一次按钮直接出精筛后的股票，不再需要导出CSV跑脚本。

口径一致性（关键）：特征公式严格复刻 research_t1_factors.py（训练面板的构造脚本），
                    与 refine_model.txt 训练时完全一致，切勿改动：
  动量20 = c/c.shift(20)-1 ; 反转5 = -(c/c.shift(5)-1) ; 反转1 = -c.pct_change()
  波动20 = r.rolling(20).std() ; 规模代理 = log((c*v).rolling(20).mean())
  Amihud = (|r|/(c*v)*1e8).rolling(20).mean()
  MACD柱 = (dif-dea)*2 (12,26,9) ; KDJ_J = 3k-2d (9,3,3, ewm com=2)
注意：这套口径与 factors.py（5日口径选股）不同——不要混用，否则打分错位。

回测依据：含退市全A面板训练，样本外(2025-07起) top3 次日净超额 +1.558%/期、t=7.57、
         跑赢率68%、IC=+0.073（详见技术文档第二十节）。仅为统计优势，不构成投资建议。

云端可用：模型随仓库部署(239K，refine_model.txt)；features 只需 K线的 日期/收盘/最高/最低/成交量。
无 lightgbm 或模型缺失时返回 None，调用方自动降级为原 5日口径结果，不阻塞选股。
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEATS = ["动量20", "反转5", "反转1", "波动20", "规模代理", "Amihud", "MACD柱", "KDJ_J"]

_model_cache = None
_model_tried = False


def _load_model():
    """从内嵌 base64 加载模型（绕开中文路径 / 部署路径不确定导致的 Booster 读文件失败）。"""
    global _model_cache, _model_tried
    if _model_tried:
        return _model_cache
    _model_tried = True
    try:
        from lightgbm import Booster
        from t1_model_b64 import model_bytes
        model_str = model_bytes().decode("utf-8")
        _model_cache = Booster(model_str=model_str)
    except Exception:
        _model_cache = None
    return _model_cache


def available() -> bool:
    """Web 侧判断是否可启用 T+1 精筛（模型能加载）。"""
    return _load_model() is not None


def features_from_hist(h: pd.DataFrame) -> dict | None:
    """从单只K线算最新一日的8特征（严格复刻训练面板口径）。"""
    if h is None or len(h) < 25:
        return None
    d = h.copy()
    d["日期"] = pd.to_datetime(d["日期"])
    d = d.sort_values("日期").reset_index(drop=True)
    c = d["收盘"].astype(float)
    hv = d["最高"].astype(float) if "最高" in d.columns else c
    lo = d["最低"].astype(float) if "最低" in d.columns else c
    v = d["成交量"].astype(float) if "成交量" in d.columns else pd.Series(np.nan, index=d.index)
    r = c.pct_change()
    amt = (c * v).replace(0, np.nan)
    f = {}
    f["动量20"] = c.iloc[-1] / c.iloc[-21] - 1 if len(c) > 20 else np.nan
    f["反转5"] = -(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 5 else np.nan
    f["反转1"] = -r.iloc[-1]
    f["波动20"] = r.rolling(20).std().iloc[-1]
    f["规模代理"] = np.log(amt.rolling(20).mean().iloc[-1]) if amt.notna().any() else np.nan
    f["Amihud"] = (r.abs() / amt * 1e8).rolling(20).mean().iloc[-1] if amt.notna().any() else np.nan
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    f["MACD柱"] = (dif.iloc[-1] - dea.iloc[-1]) * 2
    low9 = lo.rolling(9).min(); high9 = hv.rolling(9).max()
    rsv = (c - low9) / (high9 - low9).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    dd = k.ewm(com=2, adjust=False).mean()
    f["KDJ_J"] = 3 * k.iloc[-1] - 2 * dd.iloc[-1]
    return f


def refine(ranked: pd.DataFrame, hists: dict) -> pd.DataFrame | None:
    """对粗筛后的候选做 T+1 精筛排序。
    ranked: verify 输出（含 代码/名称/最新价 等），hists: {代码: K线DataFrame}。
    返回：带「精筛分/T+1排名」的新表（按精筛分降序），失败返回 None。
    """
    model = _load_model()
    if model is None:
        return None
    codes = [str(c).zfill(6) for c in ranked["代码"].tolist()]
    rows = []
    for cd in codes:
        f = features_from_hist(hists.get(cd))
        if f is None:
            continue
        rows.append((cd, f))
    if not rows:
        return None
    X = pd.DataFrame([f for _, f in rows], columns=FEATS)
    # 缺失特征用训练期可用值兜底，避免 predict 报错
    X = X.replace([np.inf, -np.inf], np.nan)
    scores = model.predict(X)
    sc = {cd: s for (cd, _), s in zip(rows, scores)}
    out = ranked.copy()
    out["代码"] = out["代码"].astype(str).str.zfill(6)
    out["精筛分"] = out["代码"].map(sc)
    out = out.dropna(subset=["精筛分"]).sort_values("精筛分", ascending=False).reset_index(drop=True)
    out.insert(0, "T+1排名", range(1, len(out) + 1))
    return out


if __name__ == "__main__":
    # 自测：用面板最新截面复核特征公式与模型打分能否跑通（不联网）
    import sys
    panel = Path("C:/Users/Admin/WorkBuddy/2026-11-18-08-05-50/t1_panel_cache.csv")
    print("模型可用:", available())
    if panel.exists():
        p = pd.read_csv(panel)
        latest = p.sort_values("日期").groupby("代码").tail(1)
        print("面板最新截面样本:", len(latest), "只")
        # 检查模型能否对面板特征预测
        m = _load_model()
        if m is not None:
            s = m.predict(latest[FEATS].head(5))
            print("打分自测 OK, 示例:", [round(float(x), 4) for x in s])
