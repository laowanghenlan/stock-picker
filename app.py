"""每日一键选股 · 极简版
只要点一个按钮，后台自动完成：
全市场初筛（价格/市值/成交/剔除风险股）→ 逐只历史回测（含手续费+降费过滤器）→
综合打分（扣费收益+胜率+趋势）→ 输出3只 + 按页面输入本金分摊的买入方案。

启动方式：streamlit run app.py
"""
import datetime as dt
from pathlib import Path

import akshare as ak
import pandas as pd
import streamlit as st

from data import get_history, get_spot, refresh_spot, spot_cache_time
from strategy import backtest, generate_signals

# 利空关键词：近7天新闻标题命中任一即视为"有雷"，直接排除
NEG_KEYWORDS = ("减持", "立案", "退市", "处罚", "问询", "警示", "冻结", "失信",
                "违规", "诉讼", "仲裁", "预亏", "暴雷", "停牌", "质押", "担保逾期")


@st.cache_data(ttl=3600, show_spinner=False)
def news_risk(code: str) -> list:
    """拉取该股最近新闻标题，返回近7天内命中利空关键词的标题列表。空列表=无雷。"""
    try:
        news = ak.stock_news_em(symbol=code)
    except Exception:
        return []
    if news is None or len(news) == 0:
        return []
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    hits = []
    for _, r in news.iterrows():
        try:
            t = pd.Timestamp(str(r["发布时间"]))
        except Exception:
            continue
        if t < cutoff:
            break                                # 新闻按时间倒序，超过7天即止
        title = str(r["新闻标题"])
        if any(k in title for k in NEG_KEYWORDS):
            hits.append(title)
    return hits


@st.cache_data(ttl=3600, show_spinner="正在检查大盘温度...")
def market_temperature() -> tuple:
    """大盘温度：沪深300最新收盘 vs 60日均线。
    返回 (最新收盘, MA60, 是否低温期)。低温期=收盘在MA60下方，系统性下跌段，应空仓。"""
    try:
        end_d = dt.date.today()
        idx = ak.index_zh_a_hist(symbol="000300", period="daily",
                                 start_date=(end_d - dt.timedelta(days=200)).strftime("%Y%m%d"),
                                 end_date=end_d.strftime("%Y%m%d"))
        close = float(idx.iloc[-1]["收盘"])
        ma60 = float(idx["收盘"].tail(60).mean())
        return close, ma60, close < ma60
    except Exception:
        return 0.0, 0.0, False                     # 接口失败时不阻塞选股


@st.cache_data(ttl=43200, show_spinner=False)
def lhb_codes() -> set:
    """近一月上过龙虎榜的股票代码集合（游资爆炒区，小资金回避）。接口失败返回空集。"""
    try:
        lhb = ak.stock_lhb_stock_statistic_em(symbol="近一月")
        return set(lhb["代码"].astype(str).str.zfill(6))
    except Exception:
        return set()


HOLDINGS_FILE = Path(__file__).parent / "持仓.csv"


def load_holdings() -> pd.DataFrame:
    """读取本地持仓记录（代码/成本价/股数），没有则返回空白模板"""
    if HOLDINGS_FILE.exists():
        try:
            return pd.read_csv(HOLDINGS_FILE, dtype={"代码": str})
        except Exception:
            pass
    return pd.DataFrame({"代码": pd.Series(dtype=str),
                         "成本价": pd.Series(dtype=float),
                         "股数": pd.Series(dtype=int)})


def check_holdings(hd: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    """持仓体检：现价、盈亏%、止损/止盈/利空三种警报"""
    spot = spot.copy()
    spot["代码"] = spot["代码"].astype(str).str.zfill(6)
    px = spot.set_index("代码")["最新价"]
    rows = []
    for _, h in hd.iterrows():
        code = str(h["代码"]).strip().zfill(6)
        try:
            cost = float(h["成本价"])
        except (TypeError, ValueError):
            continue
        if code not in px.index or cost <= 0:
            continue
        now = float(px[code])
        pnl = (now / cost - 1) * 100
        if now <= cost * (1 - STOP_LOSS_PCT):
            act = "🔴 止损卖出"
        elif now >= cost * (1 + TAKE_PROFIT_PCT):
            act = "🟢 到止盈，考虑卖"
        elif news_risk(code):
            act = "⚠️ 有利空新闻，检查"
        else:
            act = "✅ 继续持有"
        rows.append({"代码": code, "成本价": cost, "现价": now,
                     "浮动盈亏%": round(pnl, 2), "操作": act})
    return pd.DataFrame(rows)

# ---------------- 固定参数（本金由页面输入，其余固化在后台） ----------------
MIN_PRICE = 2.0         # 股价下限：避开1元股退市风险区
CANDIDATES = 15         # 初筛后进入回测验证的候选数量
PICKS = 3               # 最终输出股票数
STOP_LOSS_PCT = 0.05    # 止损线：跌5%必须卖
TAKE_PROFIT_PCT = 0.10  # 止盈参考：涨10%考虑落袋
HISTORY_DAYS = 500      # 回测用的历史天数

st.set_page_config(page_title="每日选股", page_icon="📈", layout="centered")
st.title("📈 每日选股")

# ---------------- 大盘温度横幅（低温期提示空仓） ----------------
idx_close, idx_ma60, market_cold = market_temperature()
if market_cold:
    st.error(f"❄️ 大盘低温期：沪深300（{idx_close:.0f}）在60日线（{idx_ma60:.0f}）下方，"
             f"系统性下跌段。默认建议空仓，执意选股请勾选下方开关。")
    force_run = st.checkbox("我知道风险，仍要选股")
else:
    st.success(f"☀️ 大盘温度正常：沪深300（{idx_close:.0f}）位于60日线上方")
    force_run = True

# ---------------- 本金输入（灵活可调，改完点按钮重新选股） ----------------
capital = st.number_input("💰 本金（元）", min_value=1000, max_value=100000,
                          value=2200, step=100)
# 股价上限：保证 PICKS 只各买100股也买得起；本金越大上限越宽，最高20元（保持低价股策略）
MAX_PRICE = max(MIN_PRICE + 0.5, min(20.0, capital / (PICKS * 100) * 0.95))

# ---------------- 我的持仓（每天自动体检：止损/止盈/利空） ----------------
with st.expander("💼 我的持仓（填入代码和成本价，点选股时自动体检，自动保存）", expanded=True):
    holdings = st.data_editor(load_holdings(), num_rows="dynamic",
                              use_container_width=True, key="holdings_editor",
                              column_config={"代码": st.column_config.TextColumn("代码(6位)"),
                                             "成本价": st.column_config.NumberColumn("成本价", format="%.2f"),
                                             "股数": st.column_config.NumberColumn("股数", format="%d")})

# ---------------- 侧边栏：数据状态与手动刷新 ----------------
with st.sidebar:
    st.caption(f"行情快照时间：{spot_cache_time()}")
    if st.button("🔄 刷新行情数据", use_container_width=True):
        with st.spinner("正在联网刷新全市场行情，约1分钟..."):
            refresh_spot()
        st.rerun()
    st.caption("平时不用点：每个交易日开盘后首次打开会自动更新。")

# ---------------- 选股流水线（全部后台执行） ----------------
def _board_ok(code: str) -> bool:
    """剔除小资金买不了或风险过高的板块：
    300/301创业板(需10万门槛)、688科创板(50万门槛)、8/4/920北交所(50万门槛)"""
    return not code.startswith(("300", "301", "688", "8", "4", "920"))


def screen() -> pd.DataFrame:
    """第一步：从全市场快照里做硬性初筛，只留下安全、买得起、流动性够的股票"""
    df = get_spot()
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    lhb = lhb_codes()                                    # 龙虎榜名单（近一月）
    name = df["名称"].astype(str)
    mask = (
        df["代码"].map(_board_ok)
        & ~df["代码"].isin(lhb)                           # 剔除近一月上过龙虎榜的游资爆炒股
        & ~name.str.contains("ST|退", na=False)          # 剔除ST/退市股
        & df["最新价"].between(MIN_PRICE, MAX_PRICE)      # 价格区间：2200元买3只也买得起
        & (df["成交额"] > 5e7)                            # 日成交额>5000万，保证能顺利卖出
        & df["市盈率-动态"].between(0, 80)                # 剔除亏损股和泡沫股
        & df["总市值"].between(3e9, 2e11)                 # 30亿~2000亿，避开微盘操纵股和超级大盘股
        & df["涨跌幅"].between(-9, 7)                     # 剔除已涨停追不进去的、和跌停的
        & df["量比"].between(0.5, 10)                     # 量能正常，剔除异常放量庄股
    )
    return df[mask]


def _trade_stats(sig: pd.DataFrame) -> tuple[float, float]:
    """从信号列配对买卖，估算历史平均每笔盈利率/亏损率（已粗略扣0.75%双边成本）。
    返回 (平均盈利率%, 平均亏损率%)，无样本时返回 (0, 0)。"""
    s = sig["signal"].values
    close = sig["收盘"].values
    rets, entry = [], None
    for i in range(len(s)):
        if s[i] == 1 and entry is None:
            entry = close[i]
        elif s[i] == -1 and entry is not None and entry > 0:
            rets.append(close[i] / entry - 1 - 0.0075)
            entry = None
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    avg_win = sum(wins) / len(wins) * 100 if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) * 100 if losses else 0.0
    return avg_win, avg_loss


def verify(cands: pd.DataFrame, progress) -> pd.DataFrame:
    """第二步：对初筛候选逐只跑 MA5/20 金叉策略回测。
    回测已含：佣金万2.5(最低5元)+印花税+滑点、冷却期20天、最短持有5天。
    用扣费后的真实收益打分，赚不回手续费的股票直接被淘汰。"""
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=HISTORY_DAYS)
    rows = []
    total = len(cands)
    for i, (_, r) in enumerate(cands.iterrows()):
        code = r["代码"]
        try:
            h = get_history(code, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
            if h is None or len(h) < 120:
                continue
            sig = generate_signals(h, "ma_cross", {"fast": 5, "slow": 20},
                                   cooldown=20, min_hold_days=5)
            _, _, bs = backtest(sig, initial_cash=float(capital))
            fee = float(bs["累计手续费"].rstrip("元"))
            avg_win, avg_loss = _trade_stats(sig)
            risk_hits = news_risk(code)          # 舆情排雷：近7天利空新闻
            last = sig.iloc[-1]
            # 技术面硬条件：当前价在20日线上，且5日线在20日线上（趋势仍向上）
            # _fast/_slow 是信号表里的 MA5/MA20 列
            trend_ok = bool(last["收盘"] > last["_slow"] and last["_fast"] > last["_slow"])
            rows.append({
                "代码": code, "名称": r["名称"], "最新价": float(r["最新价"]),
                "今日涨跌": float(r["涨跌幅"]), "量比": float(r["量比"]),
                "扣费总收益": float(bs["总收益率"].rstrip("%")),
                "胜率": float(bs["胜率"].rstrip("%")),
                "最大回撤": float(bs["最大回撤"].rstrip("%")),
                "交易次数": int(bs["交易次数"]), "手续费": fee,
                "平均盈利率": avg_win, "平均亏损率": avg_loss,
                "趋势向上": trend_ok,
                "有雷": len(risk_hits) > 0, "利空标题": "；".join(risk_hits[:2]),
            })
        except Exception:
            continue                                    # 单只失败不影响整体
        progress((i + 1) / total, f"回测验证中 {i + 1}/{total}：{r['名称']}")
    return pd.DataFrame(rows)


def pick_top3(res: pd.DataFrame) -> pd.DataFrame:
    """第三步：综合打分。
    扣费收益占大头，胜率其次，趋势向上是必须条件（不满足扣大分）。"""
    res = res.copy()
    # 百分位打分（0~100），避免不同量纲直接相加
    res["收益分"] = res["扣费总收益"].rank(pct=True) * 100
    res["胜率分"] = res["胜率"].rank(pct=True) * 100
    res["趋势分"] = res["趋势向上"].map({True: 100.0, False: 0.0})
    res["总分"] = res["收益分"] * 0.5 + res["胜率分"] * 0.2 + res["趋势分"] * 0.3
    res = res.sort_values("总分", ascending=False)
    # 只保留扣费后还能赚钱、且近7天无利空新闻的股票
    res = res[(res["扣费总收益"] > 0) & (~res["有雷"])]
    return res.head(PICKS)


def allocate(picks: pd.DataFrame, budget0: float) -> pd.DataFrame:
    """第四步：把本金尽量买满。
    均摊目标 = 剩余预算 / 剩余只数；每只按目标买100股的整数倍，
    买完后剩余的钱自动顺延给下一只，最后一只尽量把预算花完。"""
    plan, budget = [], budget0
    n = len(picks)
    for i, (_, r) in enumerate(picks.iterrows()):
        price = r["最新价"]
        target = budget / (n - i)                         # 均摊到当前这只的目标金额
        shares = int(target / price / 100) * 100          # 100股整数倍
        if shares == 0:
            shares = 100                                  # 至少买1手，否则这笔钱就浪费了
        # 若按此股数买不起，则逐手递减直到买得起
        while shares > 0:
            cost = round(price * shares, 2)
            fee = max(5.0, cost * 0.00025)
            if cost + fee <= budget:
                break
            shares -= 100
        if shares <= 0:
            continue                                      # 连1手都买不起，跳过
        invest = round(cost + fee, 2)
        win_prob = r["胜率"] / 100
        est_win = round(invest * r["平均盈利率"] / 100, 1)
        est_loss = round(invest * r["平均亏损率"] / 100, 1)
        plan.append({
            "代码": r["代码"], "名称": r["名称"],
            "现价": price, "买入股数": shares,
            "投入(含佣金)": invest,
            "盈利概率": round(r["胜率"], 0),
            "预计盈利(元)": est_win,
            "预计亏损(元)": -est_loss,
            "止损价(跌破必卖)": round(price * (1 - STOP_LOSS_PCT), 2),
            "止盈参考(到价考虑卖)": round(price * (1 + TAKE_PROFIT_PCT), 2),
            "_win_prob": win_prob,
            "_est_win": est_win, "_est_loss": est_loss,
        })
        budget -= invest
    df = pd.DataFrame(plan)
    leftover = round(budget, 2)
    return df, leftover


# ---------------- 页面：一个按钮 + 一张结果表 ----------------
st.caption(f"本金 {capital:.0f} 元 · 全部分摊买入 {PICKS} 只 · 股价上限 {MAX_PRICE:.1f} 元 · 已内置手续费/止损/回测验证")
run = st.button("🔍 今日选股", type="primary", use_container_width=True)

if run:
    if market_cold and not force_run:
        st.error("大盘低温期，已拦截选股。请勾选上方「我知道风险，仍要选股」后再点。")
        st.stop()
    # 保存持仓并体检（在选股之前，持有期的风险优先于买新股）
    try:
        holdings.dropna(subset=["代码", "成本价"]).to_csv(HOLDINGS_FILE, index=False)
    except PermissionError:
        st.caption("⚠️ 持仓未能保存到本地（系统文件夹保护），本次体检仍生效，但下次需重新填写。")
    hd_check = check_holdings(holdings, get_spot())
    if len(hd_check) > 0:
        st.markdown("**💼 持仓体检**")
        st.dataframe(hd_check, width="stretch", hide_index=True,
                     column_config={"浮动盈亏%": st.column_config.NumberColumn(format="%+.2f%%")})
        if (hd_check["操作"] != "✅ 继续持有").any():
            st.warning("有持仓需要处理，先处理持仓再考虑买新股。")
        st.markdown("---")

    bar = st.progress(0.0, text="正在全市场初筛...")
    cands = screen()
    if len(cands) == 0:
        st.warning("今天全市场没有满足基本条件的股票，建议空仓等待。")
        st.stop()
    cands = cands.nlargest(CANDIDATES, "成交额")        # 优先验证成交活跃的
    bar.progress(0.05, text=f"初筛通过 {len(cands)} 只，开始逐只回测（含手续费）...")
    res = verify(cands, lambda p, t: bar.progress(0.05 + 0.9 * p, text=t))
    bar.progress(1.0, text="完成")
    bar.empty()

    if len(res) == 0:
        st.warning("候选股历史数据不足，无法验证，今天不建议买入。")
        st.stop()
    # 舆情排雷通报：哪些因近7天利空新闻被排除
    mined = res[res["有雷"]]
    if len(mined) > 0:
        st.error(f"⚠️ 排雷：{len(mined)} 只候选股近7天有利空新闻，已自动排除——")
        for _, m in mined.iterrows():
            st.markdown(f"- **{m['名称']}（{m['代码']}）**：{m['利空标题']}")
    picks = pick_top3(res)
    if len(picks) == 0:
        st.warning("今天没有股票通过回测+排雷验证。空仓也是一种操作，明天再来。")
        st.stop()

    plan, leftover = allocate(picks, float(capital))

    # ---- 汇总指标：总投入 / 期望收益 / 整体盈利概率 ----
    total_in = plan["投入(含佣金)"].sum()
    # 单笔期望 = 胜率×预计盈利 − (1−胜率)×预计亏损；组合期望为各笔之和
    expect = float((plan["_win_prob"] * plan["_est_win"]
                    - (1 - plan["_win_prob"]) * -plan["预计亏损(元)"]).sum())
    # 组合盈利概率：假设各股独立，至少一只盈利的概率
    p_any = 1 - float((1 - plan["_win_prob"]).prod())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总投入", f"{total_in:.0f} 元")
    m2.metric("期望收益(单笔均值)", f"{expect:+.1f} 元")
    m3.metric("至少一只盈利概率", f"{p_any * 100:.0f}%")
    m4.metric("剩余现金", f"{leftover:.0f} 元")

    st.success(f"今日选出 {len(plan)} 只，方案如下：")
    show = plan.drop(columns=["_win_prob", "_est_win", "_est_loss"])
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"盈利概率": st.column_config.NumberColumn(format="%.0f%%"),
                                "预计盈利(元)": st.column_config.NumberColumn(format="%+.1f"),
                                "预计亏损(元)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("预计盈利/亏损 = 投入 × 该股历史平均每笔盈利率/亏损率（已扣交易成本）；"
               "盈利概率 = 历史回测胜率。均为历史统计推算，不代表未来，不构成投资建议。")

    st.markdown("**照做就行：**")
    for _, p in plan.iterrows():
        st.markdown(
            f"- **{p['名称']}（{p['代码']}）**：挂单 **{p['现价']}** 元附近买 **{p['买入股数']} 股**；"
            f"跌破 **{p['止损价(跌破必卖)']}** 元无条件卖出；涨到 **{p['止盈参考(到价考虑卖)']}** 元附近可卖。"
        )
    st.caption("规则只有两条：止损必须执行；单只持有超过20个交易日仍不涨，也卖出。"
               "本结果为历史数据回测推演，不构成投资建议，盈亏自负。")
