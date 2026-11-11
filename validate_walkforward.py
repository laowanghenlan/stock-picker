"""滚动样本外验证（walk-forward）：严格检验多因子评分是否真有预测力。

规则（v1.17）：
  - 训练 250 个交易日 → 测试 60 个交易日，按 60 日向前滚动，每期重新定权（不偷看测试期）
  - 训练期最后 5 个交易日不参与定权（它们的 5 日标签会伸进测试期，避免前视）
  - 因子：动量20 / 反转5 / 波动率20 / 换手20 / 换手变化 / EP / BP / ROE / 净利润增速 /
          营收增速 / 净利率 / 现金流质量
  - 去冗余：训练期内截面相关 >= 0.70 的因子只留 |IC| 更强的
  - 测试期每 5 个交易日调仓一次，等权持有前 5 只
  - 真实成交约束：涨停当天不买、跌停当天不卖（顺延到可卖日）
  - 成本：双边滑点 0.1% + 佣金万2.5 + 卖出印花税 0.05% + 过户费（指数口径，忽略最低5元）
  - 每期记录训练/测试 IC，落盘两个 CSV

用法：
    python validate_walkforward.py                # 用本地缓存（首次自动下载并缓存到 wf_cache/）
    python validate_walkforward.py --max 120      # 扩大股票池
    python validate_walkforward.py --fund 40      # 额外抓 40 只的 Baostock 财报，验证财务因子
    python validate_walkforward.py --refresh      # 忽略本地缓存重新下载

输出：
    滚动样本外验证.csv      每期的组合/基准收益、超额、IC、涨停跌停触发
    因子IC_样本外逐期.csv   每期每因子的训练IC、是否入选、测试IC
"""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import factors
from strategy import at_limit_down as _ald
from strategy import at_limit_up as _alu

FWD = factors.FWD_DAYS

HERE = Path(__file__).parent
CACHE = HERE / "wf_cache"
CACHE.mkdir(exist_ok=True)

TRAIN, TEST, STEP = 250, 60, 60
TOP_N = 5
REBAL = 5
SLIP = 0.001           # 单边滑点 0.1%
BUY_FEE = 0.00025 + 0.00001                 # 佣金 + 过户费
SELL_FEE = 0.00025 + 0.0005 + 0.00001       # 佣金 + 印花税 + 过户费
MAX_STOCKS = 100
LOOKBACK_DAYS = 1150   # 自然日 ≈ 780 个交易日，够 250 训练 + 8 期测试


def _download(code: str, start: str, end: str, refresh: bool) -> pd.DataFrame | None:
    """新浪日线（1023 根、无限流）为主，东财作兜底。返回 日期/收盘/成交量(股)。"""
    f = CACHE / f"hist_{code}.csv"
    if f.exists() and not refresh:
        try:
            h = pd.read_csv(f, dtype={"代码": str})
            if len(h) > 300:
                h["日期"] = pd.to_datetime(h["日期"])
                return h
        except Exception:
            pass
    n = str(code).zfill(6)
    pfx = "sh" if n.startswith(("6", "9")) else "sz"
    try:
        import requests
        u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             f"CN_MarketData.getKLineData?symbol={pfx}{n}&scale=240&ma=no&datalen=1200")
        j = requests.get(u, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).json()
        if isinstance(j, list) and len(j) > 300:
            h = pd.DataFrame(j).rename(columns={"day": "日期", "close": "收盘",
                                                "volume": "成交量"})
            h["日期"] = pd.to_datetime(h["日期"])
            for c in ["收盘", "成交量"]:
                h[c] = pd.to_numeric(h[c], errors="coerce")
            h = h.dropna(subset=["收盘"]).sort_values("日期").reset_index(drop=True)
            h = h[h["日期"] >= pd.to_datetime(f"{start[:4]}-{start[4:6]}-{start[6:]}")]
            if len(h) < 300:
                return None
            h[["日期", "收盘", "成交量"]].to_csv(f, index=False, encoding="utf-8-sig")
            return h[["日期", "收盘", "成交量"]]
    except Exception:
        pass
    # 东财兜底（可能被限流，只试一次）
    try:
        import akshare as ak
        h = ak.stock_zh_a_hist(symbol=n, period="daily", start_date=start,
                               end_date=end, adjust="qfq")
        if h is None or len(h) < 300:
            return None
        h["日期"] = pd.to_datetime(h["日期"])
        h = h.sort_values("日期").reset_index(drop=True)
        out = h[["日期", "收盘", "成交量"]]
        out.to_csv(f, index=False, encoding="utf-8-sig")
        return out
    except Exception:
        return None


def _board_ok(code: str) -> bool:
    """与 app.screen_impl 一致：剔除创业板/科创板/北交所。"""
    return not code.startswith(("300", "301", "688", "8", "4", "920"))


def _num(v):
    try:
        f = float(v)
        return f if f == f else None
    except Exception:
        return None


def _load_universe(max_stocks: int, refresh: bool) -> tuple[dict, dict]:
    """初筛候选 + 行业 + 历史K线。返回 (hists, meta)。
    直接读本地行情快照并按 app.screen_impl 的硬条件过滤（不查龙虎榜、不导入 app，
    这样离线跑不会被复盘/指数接口卡住）。"""
    spot_file = HERE / "spot_cache.csv"
    if not spot_file.exists():
        raise RuntimeError("缺少 spot_cache.csv：请先在网页端点一次刷新行情，或直接运行 app")
    spot = pd.read_csv(spot_file, dtype={"代码": str}).copy()
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    capital, n_picks = 2200.0, 2
    min_price = 2.0
    max_price = max(min_price + 0.5, min(20.0, capital / (n_picks * 100) * 0.95))
    name = spot["名称"].astype(str)
    mask = (
        spot["代码"].map(_board_ok)
        & ~name.str.contains("ST|退", na=False)
        & spot["最新价"].between(min_price, max_price)
        & (spot["成交额"] > 5e7)
        & spot["市盈率-动态"].between(0, 80)
        & spot["总市值"].between(3e9, 2e11)
        & spot["涨跌幅"].between(-9, 7)
        & spot["量比"].between(0.5, 10)
    )
    cands = spot[mask].nlargest(max_stocks, "成交额")
    codes = cands["代码"].tolist()
    print(f"初筛候选 {len(codes)} 只（价格 {min_price}~{max_price:.2f} 元，主板），"
          f"开始下载/读取历史K线……")
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=LOOKBACK_DAYS)
    # 行业映射：离线验证不额外抓（东财逐股接口慢且易卡），统一退回全截面百分位，
    # 不影响因子 IC 的方向与显著性判断；实盘链路里仍按真实行业中性化。
    ind_map: dict = {}

    def one(cd):
        return cd, _download(cd, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"), refresh)

    hists, meta = {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for k, (cd, h) in enumerate(pool.map(one, codes), 1):
            if h is None or len(h) < TRAIN + TEST:
                continue
            row = cands[cands["代码"] == cd].iloc[0]
            price = _num(row.get("最新价")) or 1.0
            fcap = _num(row.get("流通市值")) or 0.0
            fs = fcap / price if price > 0 else 0.0           # 流通股本（近似，用当前值）
            h = h.sort_values("日期").reset_index(drop=True)
            h["涨跌幅"] = h["收盘"].pct_change() * 100
            # 新浪日线成交量单位是股，用当前流通股本近似折算换手率（%）
            h["换手率"] = (h["成交量"] / fs * 100) if fs > 0 else np.nan
            hists[cd] = h
            pe, pb = _num(row.get("市盈率-动态")), _num(row.get("市净率"))
            meta[cd] = {"行业": ind_map.get(cd, "未分类"),
                        "PE": pe if pe and pe > 0 else None,
                        "PB": pb if pb and pb > 0 else None}
            if k % 20 == 0:
                print(f"  已就绪 {len(hists)}/{len(codes)}（{k} 次请求）")
    return hists, meta


def _fundamentals(codes: list[str], n: int) -> dict:
    """可选：抓前 n 只的 Baostock 历史财报（当日缓存，重复运行很快）。"""
    if n <= 0:
        return {}
    import fundamental as fd
    out = {}
    for i, c in enumerate(codes[:n], 1):
        try:
            df = fd.fetch_fundamental(c)
            if df is not None and len(df):
                out[c] = df
        except Exception:
            pass
        if i % 5 == 0:
            print(f"  财报 {i}/{min(n, len(codes))}（成功 {len(out)}）")
    return out


def _bars(hists: dict, meta: dict) -> pd.DataFrame:
    frames = []
    for code, h in hists.items():
        d = h[["日期", "收盘", "涨跌幅"]].copy()
        d["代码"] = code
        d["行业"] = (meta.get(code) or {}).get("行业", "未分类")
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ic_of(npanel: pd.DataFrame, dates: list, col: str) -> float:
    """指定日期区间内，某因子中性化暴露与未来5日收益的横截面IC均值。"""
    g = npanel[npanel["日期"].isin(dates)]
    vals = []
    for _, day in g.groupby("日期"):
        sub = day[[col, "未来5日收益"]].dropna()
        if len(sub) < factors.MIN_XSEC:
            continue
        c = sub[col].rank().corr(sub["未来5日收益"].rank())
        if c == c:
            vals.append(float(c))
    return float(np.mean(vals)) if vals else float("nan")


def _composite_ic(npanel: pd.DataFrame, dates: list, weights: dict) -> float:
    """组合评分（各因子中性化百分位 × 带符号权重）与未来5日收益的横截面IC均值。
    这才是真正决定选股方向的指标；单因子IC均值不会体现权重符号。"""
    g = npanel[npanel["日期"].isin(dates)]
    vals = []
    for _, day in g.groupby("日期"):
        score = pd.Series(0.0, index=day.index)
        used = False
        for f, wf in weights.items():
            col = f + "_n"
            if col in day.columns:
                score = score.add(day[col].fillna(0.0) * wf, fill_value=0.0)
                used = True
        if not used:
            continue
        sub = pd.DataFrame({"s": score, "r": day["未来5日收益"]}).dropna()
        if len(sub) < factors.MIN_XSEC:
            continue
        c = sub["s"].rank().corr(sub["r"].rank())
        if c == c:
            vals.append(float(c))
    return float(np.mean(vals)) if vals else float("nan")


def _period_ic(npanel: pd.DataFrame, dates: list, factors_use: list[str]) -> dict:
    out = {}
    for f in factors_use:
        col = f + "_n"
        if col in npanel.columns:
            out[f] = _ic_of(npanel, dates, col)
    return out


def run(max_stocks: int = MAX_STOCKS, refresh: bool = False, n_fund: int = 0,
        rebal: int = REBAL):
    global REBAL
    REBAL = rebal
    hists, meta = _load_universe(max_stocks, refresh)
    if len(hists) < 30:
        print("可用股票不足 30 只，终止。")
        return
    print(f"股票池 {len(hists)} 只，准备财报因子……")
    funds = _fundamentals(list(hists), n_fund)
    if funds:
        print(f"财报因子覆盖 {len(funds)} 只")

    panel = factors.build_panel(hists, meta, fundamentals=funds)
    npanel = factors.neutralize(panel)
    bars = _bars(hists, meta)
    bar_idx = bars.set_index(["日期", "代码"]).sort_index()
    dates = sorted(npanel["日期"].unique())
    n_win = max(0, (len(dates) - TRAIN) // STEP)
    print(f"交易日 {len(dates)} 天，可滚动 {n_win} 期（训练{TRAIN}/测试{TEST}）")

    rows, ic_rows = [], []
    for k in range(n_win):
        i0 = k * STEP
        train_all = dates[max(0, i0): i0 + TRAIN]
        train_lbl = train_all[:-FWD]            # 最后5天标签伸进测试期，不参与定权
        test_dates = dates[i0 + TRAIN: i0 + TRAIN + TEST]
        if len(test_dates) < TEST:
            break

        # ---------- 训练：逐因子IC → 去冗余 → 定权 ----------
        ic_tr = _period_ic(npanel, train_lbl, factors.ALL_FACTOR_COLS)
        keep = factors.prune_correlated(npanel[npanel["日期"].isin(train_lbl)], ic_tr)
        w = factors.ic_weights(ic_tr, keep=keep)
        if not w:
            # 全部无效也要留一期记录，说明这一期没有可用因子
            rows.append({"期数": k + 1, "训练结束": str(train_all[-1])[:10],
                         "测试区间": f"{str(test_dates[0])[:10]}~{str(test_dates[-1])[:10]}",
                         "入选因子": "无", "训练IC": 0.0, "测试IC": 0.0,
                         "毛收益%": 0.0, "组合收益%": 0.0, "成本拖累%": 0.0,
                         "基准收益%": 0.0, "超额%": 0.0,
                         "最大回撤%": 0.0, "平均持仓": 0, "涨停跳过": 0, "跌停顺延": 0})
            continue

        # ---------- 测试：每 REBAL 日调仓，等权TOP_N，含涨跌停/滑点/费用 ----------
        daily_ret, periods, gross_periods, buys_skip, sells_delay, holdings = [], [], [], 0, 0, []
        j = 0
        while j + REBAL + 1 < len(test_dates):
            sig_day = test_dates[j]                    # 用收盘数据打分
            buy_day = test_dates[j + 1]                # 次日收盘成交
            sell_day = test_dates[j + 1 + REBAL]       # 持有5个交易日后卖出

            # 评分：各因子中性化百分位 × 权重
            day_panel = npanel[npanel["日期"] == sig_day].set_index("代码")
            score = pd.Series(0.0, index=day_panel.index)
            for f, wf in w.items():
                col = f + "_n"
                if col in day_panel.columns:
                    score = score.add(day_panel[col].fillna(0.0) * wf, fill_value=0.0)
            score = score.sort_values(ascending=False)

            picked, skipped = [], 0
            for code in score.index:
                if len(picked) >= TOP_N:
                    break
                try:
                    b = bar_idx.loc[(buy_day, code)]
                except KeyError:
                    continue
                if _alu(float(b["涨跌幅"]), code):
                    skipped += 1                       # 涨停买不进
                    continue
                picked.append((code, float(b["收盘"])))
            buys_skip += skipped
            if not picked:
                j += REBAL
                continue

            # 收益：买入价含滑点，卖出价含滑点，扣除双边费用（同时记录毛收益，量化成本拖累）
            rets, gross_rets, delayed = [], [], 0
            for code, buy_px in picked:
                sd, extra = sell_day, 0
                while extra < 3:
                    try:
                        sbar = bar_idx.loc[(sd, code)]
                    except KeyError:
                        sbar = None
                    if sbar is not None and not _ald(float(sbar["涨跌幅"]), code):
                        break
                    delayed += 1
                    extra += 1
                    ix = test_dates.index(sd) + 1
                    if ix >= len(test_dates):
                        break
                    sd = test_dates[ix]
                try:
                    sell_px = float(bar_idx.loc[(sd, code)]["收盘"])
                except KeyError:
                    continue
                gross = sell_px * (1 - SLIP) / (buy_px * (1 + SLIP)) - 1
                gross_rets.append(gross)
                rets.append(gross - BUY_FEE - SELL_FEE)
            if not rets:
                j += REBAL
                continue
            r = float(np.mean(rets))
            periods.append(r)
            gross_periods.append(float(np.mean(gross_rets)))
            holdings.append(len(rets))
            sells_delay += delayed
            # 调仓期内的日度净值（用于最大回撤）
            seg = test_dates[j + 1: j + 1 + REBAL + 1]
            for a, b_day in zip(seg[:-1], seg[1:]):
                ch = []
                for code, _ in picked:
                    try:
                        ch.append(float(bar_idx.loc[(b_day, code)]["收盘"]) /
                                  float(bar_idx.loc[(a, code)]["收盘"]) - 1)
                    except KeyError:
                        continue
                if ch:
                    daily_ret.append(float(np.mean(ch)))
            j += REBAL

        # ---------- 本期统计 ----------
        test_lbl = test_dates[:-FWD]
        ic_te = _period_ic(npanel, test_lbl, list(w.keys()))
        # 综合评分IC（带权重符号），样本外真正要看的指标
        ic_test = _composite_ic(npanel, test_lbl, w)
        ic_train = _composite_ic(npanel, train_lbl, w)
        if ic_test != ic_test:
            ic_test = 0.0
        if ic_train != ic_train:
            ic_train = 0.0

        port = float(np.prod([1 + x for x in periods]) - 1) if periods else 0.0
        gross_port = float(np.prod([1 + x for x in gross_periods]) - 1) if gross_periods else 0.0
        eq = np.cumprod([1 + x for x in daily_ret]) if daily_ret else np.array([1.0])
        mdd = float((pd.Series(eq) / pd.Series(eq).cummax() - 1).min())

        # 基准：全池等权 buy&hold（测试期首尾收盘）
        first, last = test_dates[0], test_dates[-1]
        b0 = bars[bars["日期"] == first].set_index("代码")["收盘"]
        b1 = bars[bars["日期"] == last].set_index("代码")["收盘"]
        common = b0.index.intersection(b1.index)
        bench = float((b1[common] / b0[common] - 1).mean()) if len(common) else 0.0

        rows.append({
            "期数": k + 1,
            "训练结束": str(train_all[-1])[:10],
            "测试区间": f"{str(test_dates[0])[:10]}~{str(test_dates[-1])[:10]}",
            "入选因子": "、".join(w.keys()),
            "训练IC": round(ic_train, 4),
            "测试IC": round(ic_test, 4),
            "毛收益%": round(gross_port * 100, 2),
            "组合收益%": round(port * 100, 2),
            "成本拖累%": round((gross_port - port) * 100, 2),
            "基准收益%": round(bench * 100, 2),
            "超额%": round((port - bench) * 100, 2),
            "最大回撤%": round(mdd * 100, 2),
            "平均持仓": int(np.mean(holdings)) if holdings else 0,
            "涨停跳过": buys_skip,
            "跌停顺延": sells_delay,
        })
        for f, v in ic_tr.items():
            ic_rows.append({"期数": k + 1, "因子": f, "训练IC": round(v, 4) if v == v else None,
                            "入选": f in w, "测试IC": round(ic_te.get(f, np.nan), 4)
                            if ic_te.get(f, np.nan) == ic_te.get(f, np.nan) else None})
        print(f"  第{k + 1}期 {rows[-1]['测试区间']}：组合 {port:+.2%} / 基准 {bench:+.2%} / "
              f"测试IC {ic_test:+.3f} / 因子 {'、'.join(w.keys())}")

    res = pd.DataFrame(rows)
    res.to_csv(HERE / "滚动样本外验证.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ic_rows).to_csv(HERE / "因子IC_样本外逐期.csv", index=False, encoding="utf-8-sig")

    # ---------- 汇总 ----------
    print("\n" + "=" * 70)
    if len(res) == 0:
        print("没有可评估的期次。")
        return
    ics = res["测试IC"].astype(float)
    mean_ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() and ics.std() > 0 else 0.0
    pos = int((ics > 0).sum())
    port_cum = float(np.prod(1 + res["组合收益%"] / 100) - 1)
    gross_cum = float(np.prod(1 + res["毛收益%"] / 100) - 1)
    bench_cum = float(np.prod(1 + res["基准收益%"] / 100) - 1)
    print(f"期数 {len(res)} | 测试IC均值 {mean_ic:+.4f} | ICIR {icir:+.2f} | "
          f"IC为正 {pos}/{len(res)} ({pos / len(res):.0%})")
    print(f"毛收益累计 {gross_cum:+.2%} | 扣费累计 {port_cum:+.2%} | "
          f"成本拖累 {(gross_cum - port_cum) * 100:+.2f}pp")
    print(f"基准累计 {bench_cum:+.2%} | 平均每期超额(扣费) {res['超额%'].mean():+.2f}% | "
          f"平均最大回撤 {res['最大回撤%'].mean():.2f}%")
    print(f"成交约束触发：涨停跳过 {int(res['涨停跳过'].sum())} 次，跌停顺延 {int(res['跌停顺延'].sum())} 次")
    stable = mean_ic > 0 and pos / len(res) >= 0.6 and icir >= 0.3
    if not stable:
        print("\n结论：样本外 IC 不稳定 ❌ 当前因子还不足以支撑机器学习排序模型，先别上 LightGBM")
    elif res["超额%"].mean() <= 0:
        print("\n结论：样本外 IC 稳定为正（有预测力），但扣费后跑不赢等权基准 ❌"
              "\n      优势被换手成本吃掉——应先拉长持有期/降低换手，再谈 LightGBM")
    else:
        print("\n结论：样本外 IC 稳定为正且扣费后仍有超额 ✅ 可以考虑 LightGBM 排序模型")
    print("明细已写入 滚动样本外验证.csv / 因子IC_样本外逐期.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=MAX_STOCKS)
    ap.add_argument("--fund", type=int, default=0, help="抓取多少只股票的Baostock财报")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rebal", type=int, default=REBAL, help="调仓间隔（交易日）")
    a = ap.parse_args()
    run(max_stocks=a.max, refresh=a.refresh, n_fund=a.fund, rebal=a.rebal)
