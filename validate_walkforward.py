"""滚动样本外验证（walk-forward）：严格检验多因子评分是否真有预测力。

规则（v1.18）：
  - 训练 250 个交易日 → 测试 60 个交易日，按 60 日向前滚动，每期重新定权（不偷看测试期）
  - 训练期最后 5 个交易日不参与定权（它们的 5 日标签会伸进测试期，避免前视）
  - 因子：动量20 / 反转5 / 波动率20 / 换手20 / 换手变化 / MACD柱 / KDJ_J / EP / BP / ROE /
          净利润增速 / 营收增速 / 净利率 / 现金流质量（财务因子按 pubDate 严格对齐，只用已公布数据）
  - 去冗余：训练期内截面相关 >= 0.70 的因子只留 |IC| 更强的
  - 两种调仓规则：
      fixed：每 REBAL 个交易日全换仓（旧规则，成本高）
      hold ：条件换仓——最短持有 MIN_HOLD 日，跌出 EXIT_RANK 名或满 MAX_HOLD 日才卖，
             空出的仓位按评分补足 TOP_N（降换手，省成本）
  - 真实成交约束：涨停当天不买、跌停当天不卖（顺延到可卖日）
  - 成本：双边滑点 0.1% + 佣金万2.5 + 卖出印花税 0.05% + 过户费（指数口径，忽略最低5元）
  - 股票池：lowprice（默认，与实盘一致的 2~10.45 元主板小资金池）/ all（全 A，仅流动性约束）

用法：
    python validate_walkforward.py                       # 旧规则：每5日全换仓（本地缓存）
    python validate_walkforward.py --rule hold           # 降换手：条件换仓
    python validate_walkforward.py --rule hold --min-hold 10 --max-hold 20 --exit-rank 15
    python validate_walkforward.py --pool all --max 200  # 扩池到全 A（去掉低价小盘偏好）
    python validate_walkforward.py --industry            # 联网补齐行业（Baostock），按真实行业中性化
    python validate_walkforward.py --lookback 1350       # 拉长回看窗口，可跑 8 期以上
    python validate_walkforward.py --fund 20             # 额外抓 20 只的 Baostock 财报，验证财务因子
    python validate_walkforward.py --refresh             # 忽略本地缓存重新下载
    python validate_walkforward.py --ml                  # 增加ML对照组，独立输出、不覆盖原结果

输出：
    滚动样本外验证.csv      每期的组合/基准收益、超额、IC、换手、平均持有天数、涨跌停触发
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
MIN_HOLD, MAX_HOLD = 10, 20      # 条件换仓：最短/最长持有交易日
EXIT_RANK = 15                   # 评分跌出前 N 名才卖（TOP_N 的 3 倍，容忍短期排名波动）
SLIP = 0.001           # 单边滑点 0.1%
BUY_FEE = 0.00025 + 0.00001                 # 佣金 + 过户费
SELL_FEE = 0.00025 + 0.0005 + 0.00001       # 佣金 + 印花税 + 过户费
MAX_STOCKS = 100
LOOKBACK_DAYS = 1150   # 自然日 ≈ 780 个交易日，够 250 训练 + 8 期测试


def _industry_map(codes: list[str]) -> dict:
    """代码→证监会行业。用 Baostock 全市场行业表（一次查询，不逐股请求）。"""
    cache: dict = {}
    try:
        import baostock as bs
        if bs.login().error_code == "0":
            rs = bs.query_stock_industry()
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()      # [updateDate, code, name, industry, class]
                cd = str(row[1]).split(".")[-1].zfill(6)
                ind = str(row[3]).strip()
                if ind:
                    cache[cd] = ind
            bs.logout()
    except Exception as e:
        print(f"Baostock 行业查询失败：{e}")
    return {c: cache[c] for c in codes if c in cache}


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    """CSV 偶发被占用（杀软/索引），重试几次；仍失败则退到系统临时目录，避免整轮白跑。"""
    import tempfile
    import time
    for i in range(10):
        try:
            df.to_csv(path, index=False, encoding="utf-8-sig")
            return
        except PermissionError:
            if i < 9:
                time.sleep(2)
                continue
            alt = Path(tempfile.gettempdir()) / path.name
            df.to_csv(alt, index=False, encoding="utf-8-sig")
            print(f"⚠ {path} 被占用，结果已改写到 {alt}")


def _download(code: str, start: str, end: str, refresh: bool) -> pd.DataFrame | None:
    """新浪日线为主，东财兜底。统一返回 日期/收盘/最高/最低/成交量，供MACD/KDJ使用。"""
    f = CACHE / f"hist_{code}.csv"
    required = {"日期", "收盘", "最高", "最低", "成交量"}
    if f.exists() and not refresh:
        try:
            h = pd.read_csv(f, dtype={"代码": str})
            if len(h) > 300 and required.issubset(h.columns):
                h["日期"] = pd.to_datetime(h["日期"])
                # 个别旧缓存覆盖更早区间，也必须按本次起点裁剪，避免拉长公共滚动窗口
                h = h[h["日期"] >= pd.to_datetime(f"{start[:4]}-{start[4:6]}-{start[6:]}")]
                if len(h) > 300:
                    return h.reset_index(drop=True)
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
                                                "high": "最高", "low": "最低", "volume": "成交量"})
            h["日期"] = pd.to_datetime(h["日期"])
            for c in ["收盘", "最高", "最低", "成交量"]:
                h[c] = pd.to_numeric(h[c], errors="coerce")
            h = h.dropna(subset=["收盘", "最高", "最低"]).sort_values("日期").reset_index(drop=True)
            h = h[h["日期"] >= pd.to_datetime(f"{start[:4]}-{start[4:6]}-{start[6:]}")]
            if len(h) < 300:
                return None
            out = h[["日期", "收盘", "最高", "最低", "成交量"]]
            out.to_csv(f, index=False, encoding="utf-8-sig")
            return out
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
        out = h[["日期", "收盘", "最高", "最低", "成交量"]]
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


def _load_universe(max_stocks: int, refresh: bool, pool: str = "lowprice",
                   use_industry: bool = False,
                   lookback_days: int = LOOKBACK_DAYS) -> tuple[dict, dict]:
    """初筛候选 + 行业 + 历史K线。返回 (hists, meta)。
    pool="lowprice"：与 app.screen_impl 一致的小资金低价池（默认，可比历史结果）
    pool="all"    ：全 A（含创业板/科创板），只保留流动性约束，去掉低价小盘偏好
    直接读本地行情快照（不查龙虎榜、不导入 app，这样离线跑不会被复盘/指数接口卡住）。"""
    spot_file = HERE / "spot_cache.csv"
    if not spot_file.exists():
        raise RuntimeError("缺少 spot_cache.csv：请先在网页端点一次刷新行情，或直接运行 app")
    spot = pd.read_csv(spot_file, dtype={"代码": str}).copy()
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    name = spot["名称"].astype(str)
    if pool == "all":
        # 全 A：不限制价格/板块，只要求「能买能卖」——成交额和市值够大
        mask = (
            ~name.str.contains("ST|退", na=False)
            & (spot["成交额"] > 1e8)                   # 日成交额 > 1 亿，流动性约束
            & (spot["总市值"] > 5e9)                   # 市值 > 50 亿，避开壳资源
            & spot["最新价"].between(1.0, 300.0)
        )
        tag = "全A（成交额>1亿、市值>50亿）"
    else:
        capital, n_picks = 2200.0, 2
        min_price = 2.0
        max_price = max(min_price + 0.5, min(20.0, capital / (n_picks * 100) * 0.95))
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
        tag = f"价格 {min_price}~{max_price:.2f} 元，主板"
    cands = spot[mask].nlargest(max_stocks, "成交额")
    codes = cands["代码"].tolist()
    print(f"初筛候选 {len(codes)} 只（{tag}），开始下载/读取历史K线……")
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=lookback_days)
    # 行业映射：默认不联网（东财逐股接口慢），统一退回全截面百分位；
    # use_industry=True 时用 Baostock 全市场行业表（一次查询）
    ind_map: dict = {}
    if use_industry:
        ind_map = _industry_map(codes)
        print(f"行业映射覆盖 {len(ind_map)}/{len(codes)} 只")

    def one(cd):
        return cd, _download(cd, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"), refresh)

    hists, meta = {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool_:
        for k, (cd, h) in enumerate(pool_.map(one, codes), 1):
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
    """可选：抓前 n 只的 Baostock 历史财报（当日缓存，重复运行很快）。
    多取 16 个季度（约 4 年），让财务因子覆盖大部分回测区间；pubDate 由 factor_series 严格对齐。"""
    if n <= 0:
        return {}
    import fundamental as fd
    out = {}
    for i, c in enumerate(codes[:n], 1):
        try:
            df = fd.fetch_fundamental(c, max_history_quarters=16)
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


def _score_series(npanel: pd.DataFrame, dates: list, w: dict) -> dict:
    """每日评分（代码 -> 分数，降序），供两种调仓规则共用。"""
    out = {}
    for d in dates:
        dp = npanel[npanel["日期"] == d].set_index("代码")
        s = pd.Series(0.0, index=dp.index)
        for f, wf in w.items():
            col = f + "_n"
            if col in dp.columns:
                s = s.add(dp[col].fillna(0.0) * wf, fill_value=0.0)
        out[d] = s.sort_values(ascending=False)
    return out


def _score_dict_ic(npanel: pd.DataFrame, scores: dict, dates: list) -> float:
    """逐日评分与未来5日收益的截面 Spearman IC。"""
    vals = []
    for d in dates:
        s = scores.get(d)
        if s is None:
            continue
        day = npanel[npanel["日期"] == d].set_index("代码")
        sub = pd.DataFrame({"s": s, "r": day["未来5日收益"]}).dropna()
        if len(sub) < factors.MIN_XSEC:
            continue
        ic = sub["s"].rank().corr(sub["r"].rank())
        if ic == ic:
            vals.append(float(ic))
    return float(np.mean(vals)) if vals else float("nan")


def _ml_score_series(npanel: pd.DataFrame, train_dates: list, test_dates: list,
                     base_scores: dict, weight: float) -> tuple[dict | None, dict]:
    """训练窗拟合一次并预测整个测试窗；训练验证不达标时返回 None，绝不用测试标签训练。"""
    import ml_model

    train = npanel[npanel["日期"].isin(train_dates)]
    fitted = ml_model.fit_model(train)
    if fitted is None:
        return None, {"backend": ml_model.available(), "oof_ic": float("nan")}
    meta = {k: v for k, v in fitted.items() if k != "_model"}
    if not fitted["oof_pass"]:
        return None, meta
    out = {}
    for d in test_dates:
        day = npanel[npanel["日期"] == d]
        if len(day) == 0:
            out[d] = base_scores[d]
            continue
        codes = day["代码"].astype(str)
        ml_s = pd.Series(ml_model.predict_scores(fitted, day).to_numpy(), index=codes)
        base = base_scores[d].reindex(codes)
        # 原因子分与ML分都转为当日截面百分位后再混合，避免量纲不一致
        blend = (1 - weight) * base.rank(pct=True) + weight * ml_s.rank(pct=True)
        out[d] = blend.sort_values(ascending=False)
    return out, meta


def _metrics(navs: list, gross_navs: list) -> dict:
    nav = navs[-1] if navs else 1.0
    gross = gross_navs[-1] if gross_navs else 1.0
    eq = pd.Series(navs) if navs else pd.Series([1.0])
    mdd = float((eq / eq.cummax() - 1).min())
    return {"net": float(nav - 1), "gross": float(gross - 1), "mdd": mdd}


def _sim_fixed(test_dates, scores, bar_idx, top_n=TOP_N, rebal=REBAL, cost=True):
    """旧规则：每 rebal 日整仓换一次（信号日收盘打分，次日收盘成交）。
    返回净值序列 + 毛收益序列（cost=False 时同一条路径，用于量化成本拖累）。"""
    slip = SLIP if cost else 0.0
    bfee = BUY_FEE if cost else 0.0
    sfee = SELL_FEE if cost else 0.0
    cash, pos, navs = 1.0, {}, []
    buys_skip = sells_delay = turn = 0
    j = 0
    pending_sell: list = []
    while j < len(test_dates):
        d = test_dates[j]
        # 到点全卖
        if j > 0:
            for code in list(pos):
                p = _px(bar_idx, d, code)
                if p is None:
                    continue
                if _ald(float(bar_idx.loc[(d, code)]["涨跌幅"]), code):
                    pending_sell.append(code)
                    sells_delay += 1
                    continue
                cash += pos.pop(code) * p * (1 - slip) * (1 - sfee)
            for code in list(pending_sell):          # 跌停顺延的，能卖就补卖
                p = _px(bar_idx, d, code)
                if p is None or code not in pos:
                    continue
                if _ald(float(bar_idx.loc[(d, code)]["涨跌幅"]), code):
                    continue
                cash += pos.pop(code) * p * (1 - slip) * (1 - sfee)
                pending_sell.remove(code)
        # 按评分买入 top_n（涨停跳过）
        s = scores.get(test_dates[j - 1], pd.Series(dtype=float)) if j > 0 else pd.Series(dtype=float)
        bought = 0
        for code in s.index:
            if bought >= top_n:
                break
            p = _px(bar_idx, d, code)
            if p is None:
                continue
            if _alu(float(bar_idx.loc[(d, code)]["涨跌幅"]), code):
                buys_skip += 1
                continue
            amount = min(cash, cash / max(1, top_n - bought))
            if amount <= 1e-9:
                continue
            cash -= amount + amount * bfee
            pos[code] = amount / (p * (1 + slip))
            bought += 1
            turn += 1
        navs.append(cash + sum(sh * (_px(bar_idx, d, c) or 0) for c, sh in pos.items()))
        j += rebal
    # 期末清仓
    last = test_dates[-1]
    for code in list(pos):
        p = _px(bar_idx, last, code) or 0.0
        cash += pos.pop(code) * p * (1 - slip) * (1 - sfee)
    if navs:
        navs[-1] = cash
    return navs, buys_skip, sells_delay, turn


def _px(bar_idx, d, code):
    try:
        return float(bar_idx.loc[(d, code)]["收盘"])
    except KeyError:
        return None


def _sim_hold(test_dates, scores, bar_idx, top_n=TOP_N, min_hold=MIN_HOLD,
              max_hold=MAX_HOLD, exit_rank=EXIT_RANK, cost=True):
    """条件换仓（降换手）：最短持有 min_hold 日，评分跌出前 exit_rank 名或满 max_hold 日才卖；
    空出的仓位按当日评分补足 top_n。信号日收盘打分，次日收盘成交。"""
    slip = SLIP if cost else 0.0
    bfee = BUY_FEE if cost else 0.0
    sfee = SELL_FEE if cost else 0.0
    cash, pos, buy_i = 1.0, {}, {}
    navs, hold_len = [], []
    buys_skip = sells_delay = turn = 0
    pend_sell, pend_buy = [], []
    for i, d in enumerate(test_dates):
        # (a) 执行昨日挂单（今日收盘价）
        still = []
        for code in pend_sell:
            p = _px(bar_idx, d, code)
            if p is None or code not in pos:
                continue
            if _ald(float(bar_idx.loc[(d, code)]["涨跌幅"]), code):
                still.append(code)
                sells_delay += 1
                continue
            cash += pos.pop(code) * p * (1 - slip) * (1 - sfee)
            hold_len.append(i - buy_i.pop(code, i))
            turn += 1
        pend_sell = still
        for code in pend_buy:
            if len(pos) >= top_n:
                break
            if code in pos:
                continue
            p = _px(bar_idx, d, code)
            if p is None:
                continue
            if _alu(float(bar_idx.loc[(d, code)]["涨跌幅"]), code):
                buys_skip += 1
                continue
            nav_now = cash + sum(sh * (_px(bar_idx, d, c) or 0) for c, sh in pos.items())
            amount = min(cash, nav_now / top_n)
            if amount <= 1e-9:
                break
            fee = amount * bfee
            if amount + fee > cash:
                amount = cash / (1 + bfee)
                fee = amount * bfee
            cash -= amount + fee
            pos[code] = amount / (p * (1 + slip))
            buy_i[code] = i - 1                    # 昨日信号、今日成交 → 当日算持有1天
            turn += 1
        pend_buy = []
        navs.append(cash + sum(sh * (_px(bar_idx, d, c) or 0) for c, sh in pos.items()))
        if i + 1 >= len(test_dates):
            break
        # (b) 生成明日订单
        s = scores[d]
        new_sell = []
        for code in list(pos):
            held = i - buy_i.get(code, i)
            if held >= max_hold:
                new_sell.append(code)
            elif held >= min_hold:
                rank = int(s.index.get_loc(code)) if code in s.index else 10 ** 6
                if rank >= exit_rank:
                    new_sell.append(code)
        pend_sell = new_sell
        keep = set(pos) - set(new_sell)
        need = max(0, top_n - len(keep))
        slots = need * 4 + 5
        pend_buy = [c for c in s.index if c not in keep][:slots]
    # 期末清仓
    last = test_dates[-1]
    for code in list(pos):
        p = _px(bar_idx, last, code) or 0.0
        cash += pos.pop(code) * p * (1 - slip) * (1 - sfee)
        hold_len.append(len(test_dates) - 1 - buy_i.get(code, 0))
    if navs:
        navs[-1] = cash
    return navs, buys_skip, sells_delay, turn, (float(np.mean(hold_len)) if hold_len else 0.0)


def run(max_stocks: int = MAX_STOCKS, refresh: bool = False, n_fund: int = 0,
        rebal: int = REBAL, rule: str = "fixed", min_hold: int = MIN_HOLD,
        max_hold: int = MAX_HOLD, exit_rank: int = EXIT_RANK, pool: str = "lowprice",
        ml: bool = False, ml_weight: float = 0.20,
        use_industry: bool = False, lookback_days: int = LOOKBACK_DAYS):
    ml_weight = max(0.0, min(0.5, float(ml_weight)))
    hists, meta = _load_universe(max_stocks, refresh, pool, use_industry, lookback_days)
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
    print(f"交易日 {len(dates)} 天，可滚动 {n_win} 期（训练{TRAIN}/测试{TEST}，规则 {rule}，"
          f"行业 {'开' if use_industry else '关'}）")

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
            rows.append({"期数": k + 1, "训练结束": str(train_all[-1])[:10],
                         "测试区间": f"{str(test_dates[0])[:10]}~{str(test_dates[-1])[:10]}",
                         "入选因子": "无", "训练IC": 0.0, "测试IC": 0.0,
                         "毛收益%": 0.0, "组合收益%": 0.0, "成本拖累%": 0.0,
                         "基准收益%": 0.0, "超额%": 0.0, "最大回撤%": 0.0,
                         "换手次数": 0, "平均持有天数": 0,
                         "平均持仓": 0, "涨停跳过": 0, "跌停顺延": 0})
            continue

        scores = _score_series(npanel, test_dates, w)
        ml_scores, ml_meta = None, {}
        if ml:
            ml_scores, ml_meta = _ml_score_series(npanel, train_lbl, test_dates, scores, ml_weight)

        # ---------- 测试：含涨跌停/滑点/费用 ----------
        if rule == "hold":
            navs, bskip, sdelay, turn, avg_hold = _sim_hold(
                test_dates, scores, bar_idx, TOP_N, min_hold, max_hold, exit_rank, cost=True)
            gnavs, *_ = _sim_hold(
                test_dates, scores, bar_idx, TOP_N, min_hold, max_hold, exit_rank, cost=False)
            m = _metrics(navs, gnavs)
        else:
            navs, bskip, sdelay, turn = _sim_fixed(test_dates, scores, bar_idx, TOP_N, rebal, cost=True)
            gnavs, *_ = _sim_fixed(test_dates, scores, bar_idx, TOP_N, rebal, cost=False)
            m = _metrics(navs, gnavs)
            avg_hold = float(rebal)

        ml_m, ml_turn, ml_avg_hold = None, None, None
        if ml_scores is not None:
            if rule == "hold":
                mnavs, mbskip, msdelay, ml_turn, ml_avg_hold = _sim_hold(
                    test_dates, ml_scores, bar_idx, TOP_N, min_hold, max_hold, exit_rank, cost=True)
                mgnavs, *_ = _sim_hold(
                    test_dates, ml_scores, bar_idx, TOP_N, min_hold, max_hold, exit_rank, cost=False)
            else:
                mnavs, mbskip, msdelay, ml_turn = _sim_fixed(
                    test_dates, ml_scores, bar_idx, TOP_N, rebal, cost=True)
                mgnavs, *_ = _sim_fixed(
                    test_dates, ml_scores, bar_idx, TOP_N, rebal, cost=False)
                ml_avg_hold = float(rebal)
            ml_m = _metrics(mnavs, mgnavs)

        # ---------- 本期统计 ----------
        test_lbl = test_dates[:-FWD]
        ic_te = _period_ic(npanel, test_lbl, list(w.keys()))
        ic_test = _composite_ic(npanel, test_lbl, w)
        ic_train = _composite_ic(npanel, train_lbl, w)
        ic_ml = _score_dict_ic(npanel, ml_scores, test_lbl) if ml_scores is not None else float("nan")
        if ic_test != ic_test:
            ic_test = 0.0
        if ic_train != ic_train:
            ic_train = 0.0

        # 基准：全池等权 buy&hold（测试期首尾收盘）
        first, last = test_dates[0], test_dates[-1]
        b0 = bars[bars["日期"] == first].set_index("代码")["收盘"]
        b1 = bars[bars["日期"] == last].set_index("代码")["收盘"]
        common = b0.index.intersection(b1.index)
        r_all = (b1[common] / b0[common] - 1) if len(common) else pd.Series(dtype=float)
        # 基准用中位数：全A池里偶尔有连续涨停的妖股，均值会被单只股票拉爆、失去可比性
        bench = float(r_all.median()) if len(r_all) else 0.0
        bench_mean = float(r_all.mean()) if len(r_all) else 0.0

        rows.append({
            "期数": k + 1,
            "训练结束": str(train_all[-1])[:10],
            "测试区间": f"{str(test_dates[0])[:10]}~{str(test_dates[-1])[:10]}",
            "入选因子": "、".join(w.keys()),
            "训练IC": round(ic_train, 4),
            "测试IC": round(ic_test, 4),
            "毛收益%": round(m["gross"] * 100, 2),
            "组合收益%": round(m["net"] * 100, 2),
            "成本拖累%": round((m["gross"] - m["net"]) * 100, 2),
            "基准收益%": round(bench_mean * 100, 2),     # 全池等权买入持有（随机选股的期望值）
            "基准中位%": round(bench * 100, 2),          # 池内中位个股（避免单只妖股拉偏）
            "超额%": round((m["net"] - bench_mean) * 100, 2),
            "最大回撤%": round(m["mdd"] * 100, 2),
            "换手次数": turn,
            "平均持有天数": round(avg_hold, 1),
            "平均持仓": TOP_N,
            "涨停跳过": bskip,
            "跌停顺延": sdelay,
        })
        for f, v in ic_tr.items():
            ic_rows.append({"期数": k + 1, "因子": f, "训练IC": round(v, 4) if v == v else None,
                            "入选": f in w, "测试IC": round(ic_te.get(f, np.nan), 4)
                            if ic_te.get(f, np.nan) == ic_te.get(f, np.nan) else None})
        if ml:
            ml_oof = ml_meta.get("oof_ic", float("nan"))
            rows[-1].update({
                "ML后端": ml_meta.get("backend") or "无",
                "ML训练验证IC": round(ml_oof, 4) if ml_oof == ml_oof else None,
                "ML参与打分": ml_scores is not None,
                "ML测试IC": round(ic_ml, 4) if ic_ml == ic_ml else None,
                "ML毛收益%": round(ml_m["gross"] * 100, 2) if ml_m else None,
                "ML组合收益%": round(ml_m["net"] * 100, 2) if ml_m else None,
                "ML成本拖累%": round((ml_m["gross"] - ml_m["net"]) * 100, 2) if ml_m else None,
                "ML超额%": round((ml_m["net"] - bench_mean) * 100, 2) if ml_m else None,
                "ML最大回撤%": round(ml_m["mdd"] * 100, 2) if ml_m else None,
                "ML换手次数": ml_turn,
                "ML平均持有天数": round(ml_avg_hold, 1) if ml_avg_hold is not None else None,
            })
        line = (f"  第{k + 1}期 {rows[-1]['测试区间']}：组合 {m['net']:+.2%} / 基准 {bench:+.2%} / "
                f"测试IC {ic_test:+.3f} / 换手 {turn} 次 / 持有 {avg_hold:.0f} 天")
        if ml:
            line += (f" / ML {ml_m['net']:+.2%}、IC {ic_ml:+.3f}" if ml_m
                     else f" / ML未参与（训练验证IC {ml_meta.get('oof_ic', float('nan')):+.3f}）")
        print(line)

    res = pd.DataFrame(rows)
    tag = f"{rule}_t{rebal}" if rule == "fixed" else f"hold_{min_hold}_{max_hold}_{exit_rank}"
    if pool != "lowprice":
        tag += f"_{pool}"
    if use_industry:
        tag += "_ind"
    if ml:
        tag += f"_ml{round(ml_weight * 100)}"
    ic_file = (f"因子IC_样本外逐期_{tag}.csv" if (ml or use_industry)
               else "因子IC_样本外逐期.csv")
    _safe_to_csv(res, HERE / f"滚动样本外验证_{tag}.csv")
    _safe_to_csv(pd.DataFrame(ic_rows), HERE / ic_file)
    if (not ml and rule == "fixed" and rebal == REBAL and pool == "lowprice"
            and n_fund == 0 and not use_industry):
        _safe_to_csv(res, HERE / "滚动样本外验证.csv")  # 默认口径保持旧文件名

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
    bench_med_cum = float(np.prod(1 + res["基准中位%"] / 100) - 1)
    print(f"期数 {len(res)} | 测试IC均值 {mean_ic:+.4f} | ICIR {icir:+.2f} | "
          f"IC为正 {pos}/{len(res)} ({pos / len(res):.0%})")
    print(f"毛收益累计 {gross_cum:+.2%} | 扣费累计 {port_cum:+.2%} | "
          f"成本拖累 {(gross_cum - port_cum) * 100:+.2f}pp")
    print(f"基准累计(等权) {bench_cum:+.2%} | 基准累计(中位) {bench_med_cum:+.2%} | "
          f"平均每期超额(扣费) {res['超额%'].mean():+.2f}% | "
          f"平均最大回撤 {res['最大回撤%'].mean():.2f}%")
    print(f"平均每期换手 {res['换手次数'].mean():.1f} 次 | 平均持有 {res['平均持有天数'].mean():.1f} 天")
    print(f"成交约束触发：涨停跳过 {int(res['涨停跳过'].sum())} 次，跌停顺延 {int(res['跌停顺延'].sum())} 次")
    if ml:
        participated = res["ML参与打分"].fillna(False) if "ML参与打分" in res.columns else pd.Series(dtype=bool)
        mr = res[participated] if len(participated) else res.iloc[0:0]
        if len(mr):
            mics = pd.to_numeric(mr["ML测试IC"], errors="coerce").dropna()
            ml_ic = float(mics.mean()) if len(mics) else float("nan")
            ml_icir = float(mics.mean() / mics.std()) if len(mics) > 1 and mics.std() > 0 else 0.0
            ml_cum = float(np.prod(1 + mr["ML组合收益%"] / 100) - 1)
            base_same = float(np.prod(1 + mr["组合收益%"] / 100) - 1)
            print(f"ML对照：参与 {len(mr)}/{len(res)} 期 | 测试IC均值 {ml_ic:+.4f} | ICIR {ml_icir:+.2f}")
            print(f"ML扣费累计 {ml_cum:+.2%} | 同期原策略 {base_same:+.2%} | "
                  f"ML平均超额 {mr['ML超额%'].mean():+.2f}% | ML平均最大回撤 {mr['ML最大回撤%'].mean():.2f}%")
        else:
            print("ML对照：所有期次均未通过训练窗样本外IC门槛，未参与打分。")
    stable = mean_ic > 0 and pos / len(res) >= 0.6 and icir >= 0.3
    if not stable:
        print("\n结论：样本外 IC 不稳定 ❌ 当前因子还不足以支撑机器学习排序模型，先别上 LightGBM")
    elif res["超额%"].mean() <= 0:
        print("\n结论：样本外 IC 稳定为正（有预测力），但扣费后跑不赢等权基准 ❌"
              "\n      优势被换手成本吃掉——应先拉长持有期/降低换手，再谈 LightGBM")
    else:
        print("\n结论：样本外 IC 稳定为正且扣费后仍有超额 ✅ 可以考虑 LightGBM 排序模型")
    print(f"明细已写入 滚动样本外验证_{tag}.csv / {ic_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=MAX_STOCKS)
    ap.add_argument("--fund", type=int, default=0, help="抓取多少只股票的Baostock财报")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rebal", type=int, default=REBAL, help="fixed 规则的调仓间隔（交易日）")
    ap.add_argument("--rule", choices=["fixed", "hold"], default="fixed",
                    help="fixed=定期全换仓（旧规则）；hold=条件换仓（降换手）")
    ap.add_argument("--min-hold", type=int, default=MIN_HOLD, help="hold 规则最短持有交易日")
    ap.add_argument("--max-hold", type=int, default=MAX_HOLD, help="hold 规则最长持有交易日")
    ap.add_argument("--exit-rank", type=int, default=EXIT_RANK, help="hold 规则跌出前N名才卖")
    ap.add_argument("--pool", choices=["lowprice", "all"], default="lowprice",
                    help="lowprice=实盘同款低价主板池；all=全A扩池")
    ap.add_argument("--ml", action="store_true", help="增加LightGBM/XGBoost对照组，不覆盖原策略结果")
    ap.add_argument("--ml-weight", type=float, default=0.20, help="ML分占混合评分比例，默认0.20，上限0.50")
    ap.add_argument("--industry", action="store_true",
                    help="联网补齐行业映射（默认关闭，全部按未分类=全截面中性）")
    ap.add_argument("--lookback", type=int, default=LOOKBACK_DAYS,
                    help="回看自然日数，默认1150（约7期）；1350可跑8期以上")
    a = ap.parse_args()
    run(max_stocks=a.max, refresh=a.refresh, n_fund=a.fund, rebal=a.rebal,
        rule=a.rule, min_hold=a.min_hold, max_hold=a.max_hold,
        exit_rank=a.exit_rank, pool=a.pool, ml=a.ml, ml_weight=a.ml_weight,
        use_industry=a.industry, lookback_days=a.lookback)
