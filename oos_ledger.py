# -*- coding: utf-8 -*-
"""实盘 OOS 台账：每次推送 = 一次不可复现的样本外预测，落盘后回填实际 T+1/5/10 日收益，
滚动计算真实 IC，用来和回测的样本外 IC 对照（验证"回测里的优势在实盘还剩多少"）。

与「推送记录.csv + 复盘」的区别：
  复盘是交易口径（止盈+10%/止损-5% 判定，算胜率与盈亏金额）
  台账是预测口径（只记当时评分与价格，事后只比对 T+1/5/10 日后收盘收益，算 IC）

v1.54 口径变更（用户的真实打法=今收买、明收卖）：
  - 去重键 日期+推送时间+代码 → 日期+代码：一天点多次选股，同一只股票只留一条，
    不按点击次数重复计数；保留当天最后一次名单，且已回填的行优先保留（不丢历史回填）。
  - 新增 T+1 收益：买入价=推荐日收盘价（不是点击那一刻的现价，消除盘中点击时刻影响），
    卖出价=下一交易日收盘价；毛收益=卖出/买入-1，净收益=毛收益-COST_PCT。
  - T+1 是隔夜打法口径；5/10 日仍是预测口径，两者并存互不覆盖。

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
        "T+1买入日", "T+1买入价", "实际T+1收益%", "T+1净收益%",
        "实际5日收益%", "实际10日收益%", "回填日"]
# T+1 往返成本百分比：佣金万2.5双边(小额触底5元按0.1%估上限)+印花税0.05%卖出
# +过户费0.001%双边+双边滑点各0.1%。保守取 0.40%，宁可少算收益不可高估。
T1_COST_PCT = 0.40
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
    # 整列为空时 pandas 3.1 读成 float64，写入日期字符串会报 LossySetitemError。
    # 回填日是文本列，强制转字符串（NaN→空串），否则第一次回填必崩。
    df["回填日"] = df["回填日"].fillna("").astype(str).replace("nan", "")
    df["T+1买入日"] = df["T+1买入日"].fillna("").astype(str).replace("nan", "")
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    return df[COLS]


def _last_update_stamp() -> Path:
    """回填节流标记：一天最多真正联网回填一次，避免云端每次页面刷新都拉60只股票日线。"""
    return HERE / ".oos_last_update"


def update_due(force: bool = False) -> int:
    """带节流壳：同一自然日已回填过就跳过。返回本次回填字段数。"""
    stamp = _last_update_stamp()
    if not force and stamp.exists():
        try:
            if stamp.read_text(encoding="utf-8").strip() == dt.date.today().isoformat():
                return 0
        except Exception:
            pass
    n = update()
    try:
        stamp.write_text(dt.date.today().isoformat(), encoding="utf-8")
    except Exception:
        pass
    return n


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    """v1.54 统一去重：同日期+代码只留一条。

    合并优先级（关键）：保留当天最后一次名单的时间/价格；但若某行已回填收益，
    不能被另一次的空值覆盖——否则每点一次就把已兑现的绩效抹掉。
    做法：按「已回填」升序 + 推送时间升序排，keep="last" → 优先已回填、同级取最新。
    """
    if df.empty:
        return df
    out = df.copy()
    out["_filled"] = out[["实际5日收益%", "实际10日收益%", "实际T+1收益%"]].notna().any(axis=1).astype(int)
    out = out.sort_values(["日期", "代码", "_filled", "推送时间"], kind="stable")
    out = out.drop_duplicates(subset=["日期", "代码"], keep="last").drop(columns=["_filled"])
    return out.sort_values(["日期", "代码"], kind="stable")


def record(date_str: str, time_str: str, picks: list[dict]) -> None:
    """记录一批推送的预测。v1.54：同日期+代码只留一条（一天点多次不重复计数）。"""
    if not picks:
        return
    rows = [{"日期": date_str, "推送时间": time_str, "代码": str(p["代码"]).zfill(6),
             "名称": p.get("名称", ""), "评分": p.get("评分"), "推荐价": p.get("现价"),
             "T+1买入日": "", "T+1买入价": np.nan,
             "实际T+1收益%": np.nan, "T+1净收益%": np.nan,
             "实际5日收益%": np.nan, "实际10日收益%": np.nan, "回填日": ""} for p in picks]
    new = pd.DataFrame(rows)
    old = _read()
    out = pd.concat([old, new], ignore_index=True) if len(old) else new
    _dedup(out).to_csv(LEDGER, index=False, encoding="utf-8-sig")


def update() -> int:
    """回填已满 T+1/5/10 个交易日的样本，返回本次新回填行数。"""
    df = _read()
    if len(df) == 0:
        return 0
    # 旧格式残留（同日同代码多批次，如迁移前的 11-18 两批）先收敛到新口径。
    # 只在真有重复时重写，避免每次 update 都触发一次 CSV 写盘。
    deduped = _dedup(df)
    if len(deduped) < len(df):
        df = deduped.reset_index(drop=True)
        df.to_csv(LEDGER, index=False, encoding="utf-8-sig")
    today = dt.date.today()
    cache: dict[str, pd.Series] = {}
    filled = 0
    for i, r in df.iterrows():
        need1 = pd.isna(r.get("实际T+1收益%", np.nan))
        need5 = pd.isna(r["实际5日收益%"])
        need10 = pd.isna(r["实际10日收益%"])
        if not (need1 or need5 or need10):
            continue
        try:
            rec = pd.Timestamp(str(r["日期"]))
        except Exception:
            continue
        try:
            p0 = float(r["推荐价"])
        except Exception:
            p0 = float("nan")
        code = str(r["代码"]).zfill(6)
        if code not in cache:
            cache[code] = _close_series(code)
        s = cache[code]
        if len(s) == 0:
            continue
        touched = False
        # ---- v1.54 T+1 隔夜口径：买入价=推荐日(或之后最近交易日)收盘，卖出价=下一交易日收盘 ----
        # 用收盘价而非点击时现价：消除"一天点多次、点击时刻不同"造成的基准价漂移。
        # 周末/节假日推送没有当日K线，顺延到之后第一个交易日收盘买入（贴近真实操作）。
        if need1:
            buy = s[s.index >= rec.normalize()]
            if len(buy) >= 1:
                buy_date = buy.index[0]
                buy_px = float(buy.iloc[0])
                after = s[s.index > buy_date]
                if buy_px > 0 and len(after) >= 1:
                    sell_px = float(after.iloc[0])
                    gross = (sell_px / buy_px - 1) * 100
                    df.at[i, "T+1买入日"] = buy_date.strftime("%Y-%m-%d")
                    df.at[i, "T+1买入价"] = round(buy_px, 2)
                    df.at[i, "实际T+1收益%"] = round(gross, 2)
                    df.at[i, "T+1净收益%"] = round(gross - T1_COST_PCT, 2)
                    filled += 1
                    touched = True
        # ---- 5/10 日预测口径不变：基准仍是当时的推荐价（点击现价） ----
        # 不再用自然日门槛（7天会误卡：周三推送→下周三才够，实际5个交易日只要到下周二）。
        # 直接看"推荐日之后有几个交易日"，够几个填几个，日期不足自然填不上。
        if (need5 or need10) and p0 == p0 and p0 > 0:
            fut = s[s.index > rec]
            if need5 and len(fut) >= 5:
                df.at[i, "实际5日收益%"] = round((float(fut.iloc[4]) / p0 - 1) * 100, 2)
                filled += 1
                touched = True
            if need10 and len(fut) >= 10:
                df.at[i, "实际10日收益%"] = round((float(fut.iloc[9]) / p0 - 1) * 100, 2)
                filled += 1
                touched = True
        if touched:                       # T+1/5日/10日任一新填，就记回填日
            df.at[i, "回填日"] = today.strftime("%Y-%m-%d")
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
    # T+1 隔夜口径汇总（用户真实打法：今收买、明收卖，已扣往返成本）
    t1 = df.dropna(subset=["实际T+1收益%"])
    if len(t1):
        t1_txt = (f"T+1隔夜(扣{T1_COST_PCT:.2f}%成本)：{len(t1)} 只 | "
                  f"平均净收益 {t1['T+1净收益%'].mean():+.2f}% | "
                  f"净胜率 {(t1['T+1净收益%'] > 0).mean() * 100:.0f}%\n")
    else:
        t1_txt = "T+1隔夜：暂无回填样本（推荐日收盘+下一交易日收盘都到齐后才填）\n"
    if len(det) == 0:
        return det, t1_txt + "实盘台账：样本不足（同一批次至少5只、且满5个交易日才算一次IC），先攒数据。"
    mean_ic = det["IC"].mean()
    sd = det["IC"].std()
    icir = float(mean_ic / sd) if sd and sd > 0 else 0.0
    done = df.dropna(subset=["实际5日收益%"])
    txt = (t1_txt +
           f"实盘台账：{len(df)} 条预测，已回填5日 {len(done)} 条 | "
           f"5日平均收益 {done['实际5日收益%'].mean():+.2f}% | "
           f"胜率 {(done['实际5日收益%'] > 0).mean() * 100:.0f}%\n"
           f"实盘 IC：{mean_ic:+.4f}（{len(det)} 批次）ICIR {icir:+.2f} | "
           f"IC为正 {(det['IC'] > 0).sum()}/{len(det)}")
    return det, txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="回填到期的T+1/5/10日收益（一天最多一次）")
    ap.add_argument("--force", action="store_true", help="忽略当天节流，强制回填")
    a = ap.parse_args()
    if a.update or a.force:
        n = update_due(force=a.force)
        print(f"本次回填 {n} 个字段")
    det, txt = ic_stats()
    print(txt)
    if len(det):
        print(det.to_string(index=False))
