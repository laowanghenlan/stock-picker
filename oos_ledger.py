# -*- coding: utf-8 -*-
"""实盘 OOS 台账：每次推送 = 一次不可复现的样本外预测，落盘后回填实际 5/10 日收益，
滚动计算真实 IC，用来和回测的样本外 IC 对照（验证"回测里的优势在实盘还剩多少"）。

与「推送记录.csv + 复盘」的区别：
  复盘是交易口径（止盈+10%/止损-5% 判定，算胜率与盈亏金额）
  台账是预测口径（只记当时评分与价格，事后只比对 5/10 日后收盘收益，算 IC）

用法：
    python oos_ledger.py --update    # 回填到期样本并打印 IC 汇总
    python oos_ledger.py             # 只看已有台账
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LEDGER = HERE / "实盘台账.csv"
COLS = ["日期", "推送时间", "代码", "名称", "评分", "推荐价",
        "实际5日收益%", "实际10日收益%", "回填日"]
MIN_XSEC = 5                    # 同一批次至少5只才算横截面IC（和 factors.MIN_XSEC 同思路）
_SINA = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         "CN_MarketData.getKLineData?symbol={sym}&scale=240&ma=no&datalen=60")


def _close_series(code: str) -> pd.Series:
    """新浪日线收盘（不复权，和台账里记录的实盘成交价同口径）。失败返回空 Series。"""
    import requests
    n = str(code).zfill(6)
    pfx = "sh" if n.startswith(("6", "9")) else "sz"
    try:
        j = requests.get(_SINA.format(sym=pfx + n), timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"}).json()
        s = pd.Series([float(x["close"]) for x in j],
                      index=pd.to_datetime([x["day"] for x in j]))
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def _read() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(LEDGER, dtype={"代码": str})
    except Exception:
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[COLS]


def record(date_str: str, time_str: str, picks: list[dict]) -> None:
    """记录一批推送的预测（代码/名称/评分/推荐价）。同日期+时间+代码只留最后一次。"""
    if not picks:
        return
    rows = [{"日期": date_str, "推送时间": time_str, "代码": str(p["代码"]).zfill(6),
             "名称": p.get("名称", ""), "评分": p.get("评分"), "推荐价": p.get("现价"),
             "实际5日收益%": np.nan, "实际10日收益%": np.nan, "回填日": ""} for p in picks]
    new = pd.DataFrame(rows)
    old = _read()
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    out = out.drop_duplicates(subset=["日期", "推送时间", "代码"], keep="last")
    out = out.sort_values(["日期", "推送时间", "代码"], kind="stable")
    out.to_csv(LEDGER, index=False, encoding="utf-8-sig")


def update() -> int:
    """回填已满 5/10 个交易日的样本，返回本次新回填行数。"""
    df = _read()
    if len(df) == 0:
        return 0
    today = dt.date.today()
    cache: dict[str, pd.Series] = {}
    filled = 0
    for i, r in df.iterrows():
        need5 = pd.isna(r["实际5日收益%"])
        need10 = pd.isna(r["实际10日收益%"])
        if not (need5 or need10):
            continue
        try:
            rec = pd.Timestamp(str(r["日期"]))
            p0 = float(r["推荐价"])
        except Exception:
            continue
        if p0 <= 0 or (today - rec.date()).days < 7:      # 5个交易日≈7自然日
            continue
        code = str(r["代码"]).zfill(6)
        if code not in cache:
            cache[code] = _close_series(code)
        s = cache[code]
        if len(s) == 0:
            continue
        fut = s[s.index > rec]
        if need5 and len(fut) >= 5:
            df.at[i, "实际5日收益%"] = round((float(fut.iloc[4]) / p0 - 1) * 100, 2)
            filled += 1
        if need10 and len(fut) >= 10:
            df.at[i, "实际10日收益%"] = round((float(fut.iloc[9]) / p0 - 1) * 100, 2)
            df.at[i, "回填日"] = today.strftime("%Y-%m-%d")
            filled += 1
    if filled:
        df.to_csv(LEDGER, index=False, encoding="utf-8-sig")
    return filled


def ic_stats() -> tuple[pd.DataFrame, str]:
    """按推送批次算横截面 Spearman IC（评分 vs 实际5日收益），返回 (明细, 汇总文本)。"""
    df = _read()
    rows = []
    if len(df):
        for (d, t), g in df.dropna(subset=["实际5日收益%", "评分"]).groupby(["日期", "推送时间"]):
            if len(g) < MIN_XSEC:
                continue
            ic = g["评分"].rank().corr(g["实际5日收益%"].rank())
            if ic == ic:
                rows.append({"日期": d, "批次": t, "样本": len(g), "IC": round(float(ic), 4)})
    det = pd.DataFrame(rows)
    if len(det) == 0:
        return det, "实盘台账：样本不足（同一批次至少5只、且满5个交易日才算一次IC），先攒数据。"
    mean_ic = det["IC"].mean()
    sd = det["IC"].std()
    icir = float(mean_ic / sd) if sd and sd > 0 else 0.0
    done = df.dropna(subset=["实际5日收益%"])
    txt = (f"实盘台账：{len(df)} 条预测，已回填5日 {len(done)} 条 | "
           f"5日平均收益 {done['实际5日收益%'].mean():+.2f}% | "
           f"胜率 {(done['实际5日收益%'] > 0).mean() * 100:.0f}%\n"
           f"实盘 IC：{mean_ic:+.4f}（{len(det)} 批次）ICIR {icir:+.2f} | "
           f"IC为正 {(det['IC'] > 0).sum()}/{len(det)}")
    return det, txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="回填到期的5/10日收益")
    a = ap.parse_args()
    if a.update:
        n = update()
        print(f"本次回填 {n} 个字段")
    det, txt = ic_stats()
    print(txt)
    if len(det):
        print(det.to_string(index=False))
