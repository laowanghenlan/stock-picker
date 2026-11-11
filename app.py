"""每日一键选股 · 极简版
只要点一个按钮，后台自动完成：
全市场初筛（价格/市值/成交/剔除风险股）→ 逐只历史回测（含手续费+降费过滤器）→
综合打分（扣费收益+胜率+趋势）→ 按本金自动决定只数（本金÷1000）+ 分摊买入方案。

启动方式：streamlit run app.py
"""
import base64
import datetime as dt
import json
import os
import sys
from pathlib import Path

# Streamlit 以 __main__ 运行本脚本：把 "app" 别名指向本模块自身，
# 防止 import daily_push 时其模块级 import app 把本脚本再执行一遍（组件重复渲染报错）
if __name__ == "__main__":
    sys.modules["app"] = sys.modules["__main__"]

import akshare as ak
import pandas as pd
import requests
import streamlit as st

from data import (get_history, get_history_impl, get_spot, get_spot_impl,
                  industry_map_impl, refresh_spot, spot_cache_time, spot_fresh_enough,
                  spot_recently_failed)
from strategy import backtest, generate_signals

# 利空关键词：近7天新闻标题命中任一即视为"有雷"，直接排除
NEG_KEYWORDS = ("减持", "立案", "退市", "处罚", "问询", "警示", "冻结", "失信",
                "违规", "诉讼", "仲裁", "预亏", "暴雷", "停牌", "质押", "担保逾期")


def news_risk_impl(code: str) -> list:
    """拉取该股最近新闻标题，返回近7天内命中利空关键词的标题列表。空列表=无雷。
    （无缓存装饰器，供裸运行脚本直接调用）"""
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


# 缓存包装：web UI 一律调 news_risk；GitHub Actions 等裸运行环境请调 news_risk_impl
news_risk = st.cache_data(ttl=3600, show_spinner=False)(news_risk_impl)


def market_temperature_impl() -> tuple:
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


# 缓存包装：web UI 一律调 market_temperature；裸运行环境请调 market_temperature_impl
market_temperature = st.cache_data(ttl=3600, show_spinner="正在检查大盘温度...")(market_temperature_impl)


def lhb_codes_impl() -> set:
    """近一月上过龙虎榜的股票代码集合（游资爆炒区，小资金回避）。接口失败返回空集。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    try:
        lhb = ak.stock_lhb_stock_statistic_em(symbol="近一月")
        return set(lhb["代码"].astype(str).str.zfill(6))
    except Exception:
        return set()


# 缓存包装：web UI 一律调 lhb_codes；GitHub Actions 等裸运行环境请调 lhb_codes_impl
lhb_codes = st.cache_data(ttl=43200, show_spinner=False)(lhb_codes_impl)


def send_email_with_results(plan, mined, capital, leftover, cold, idx_close, idx_ma60):
    """用网页端已算好的选股结果直接发邮件，不重跑选股流程（v1.8 新增）。
    邮箱配置支持两种来源：环境变量（本地/Actions）或 st.secrets（Streamlit Cloud 后台 Secrets）。
    渲染/发送复用 daily_push；daily_push 模块级要读 QQ_EMAIL/QQ_AUTH_CODE，
    故先注入 os.environ 再延迟 import，整体 try/except，失败绝不影响页面展示。
    返回 (是否成功, 提示信息)。"""
    try:
        email = os.environ.get("QQ_EMAIL") or st.secrets.get("QQ_EMAIL", "")
        auth = os.environ.get("QQ_AUTH_CODE") or st.secrets.get("QQ_AUTH_CODE", "")
    except Exception:
        email, auth = os.environ.get("QQ_EMAIL", ""), os.environ.get("QQ_AUTH_CODE", "")
    if not (email and auth):
        return False, "未配置 QQ_EMAIL / QQ_AUTH_CODE（Streamlit Cloud 需在应用 Settings → Secrets 里添加），跳过发邮件"
    try:
        os.environ.setdefault("QQ_EMAIL", email)
        os.environ.setdefault("QQ_AUTH_CODE", auth)
        import daily_push
        d = {
            # 北京时间（Streamlit Cloud 运行在 UTC，凌晨点击时 UTC 还是前一天）
            "date": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%Y-%m-%d"),
            "tag": "低温期" if cold else "回暖期",
            "cold": bool(cold),
            "idx_close": float(idx_close),
            "idx_ma60": float(idx_ma60),
            "capital": float(capital),
            "mined": [{"名称": m["名称"], "代码": m["代码"], "标题": m["利空标题"]}
                      for _, m in mined.iterrows()],
            "plan": [{
                "名称": p["名称"], "代码": p["代码"], "现价": float(p["现价"]),
                "股数": int(p["买入股数"]), "投入": float(p["投入(含佣金)"]),
                "仓位占比": float(p["投入(含佣金)"]) / float(capital) * 100 if float(capital) else 0.0,
                "盈利概率": float(p["盈利概率"]),
                "止损": float(p["止损价(买入价×0.95)"]),
                "止盈": float(p["止盈参考(买入价×1.10)"]),
                "预计盈利": float(p["预计盈利(元)"]),
                "预计亏损": float(p["预计亏损(元)"]),
                "盈亏比": (float(p["预计盈利(元)"]) / abs(float(p["预计亏损(元)"]))
                           if float(p["预计亏损(元)"]) else 0.0),
            } for _, p in plan.iterrows()],
            "n_picks": len(plan),
            "invest": float(plan["投入(含佣金)"].sum()),
            "left": float(leftover),
            "exp_win": 0.0, "exp_loss": 0.0, "exp_pnl": 0.0, "max_drawdown": 0.0,
            "review": None,
            "last": None,
            "note": "",
        }
        # 组合层：按胜率加权的预期收益 / 全止盈最好 / 全止损最坏（与定时邮件同口径）
        pts = d["plan"]
        if pts:
            d["exp_win"] = sum(x["预计盈利"] for x in pts)
            d["exp_loss"] = sum(x["预计亏损"] for x in pts)
            d["exp_pnl"] = sum(x["盈利概率"] / 100 * x["预计盈利"]
                               + (1 - x["盈利概率"] / 100) * x["预计亏损"] for x in pts)
        # 复盘摘要：复用 app 的复盘逻辑，失败不影响发邮件
        try:
            rv = review_impl()
            d["last"] = daily_push.build_last_batch(rv)      # 上一次推送现在赚了还是亏了
            if rv is not None and len(rv) > 0:
                done = rv[rv["_done"]]
                if len(done) > 0:
                    wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
                    r = {"done": int(len(done)), "winrate": float(wins / len(done) * 100),
                         "pnl": float(rv["_盈亏额"].sum()), "bench": None, "excess": None,
                         "rows": [], "conclusion": ""}
                    if "同期沪深300%" in done.columns:
                        d2 = done.dropna(subset=["同期沪深300%"])
                        if len(d2) > 0:
                            r["bench"] = float(d2["同期沪深300%"].mean())
                            r["excess"] = float((d2["盈亏%"] - d2["同期沪深300%"]).mean())
                    for _, r2 in rv.head(5).iterrows():
                        r["rows"].append({"日期": str(r2["日期"]), "名称": r2["名称"],
                                          "推荐价": float(r2["推荐价"]), "现价": float(r2["现价"]),
                                          "结果": r2["结果"], "盈亏%": float(r2["盈亏%"]),
                                          "盈亏额": float(r2["_盈亏额"])})
                    r["conclusion"] = review_conclusion(rv)
                    d["review"] = r
        except Exception:
            pass
        # 留档供周末复盘（与 daily_push._collect 相同的记录格式）
        try:
            rec = plan[["代码", "名称", "现价", "买入股数"]].copy()
            rec.insert(0, "日期", d["date"])
            rec.insert(1, "推送时间", daily_push._now_hm())   # 同一天多次推送靠它分批
            rec["大盘标签"] = d["tag"]
            daily_push._append_log(rec)                       # 统一整表重写，避免追加时列错位
        except Exception:
            pass
        daily_push.send(daily_push._render_text(d), daily_push._render_html(d))
        return True, f"已发送到 {os.environ['QQ_EMAIL']}"
    except Exception as e:
        return False, f"发邮件失败：{type(e).__name__}: {e}"


HOLDINGS_FILE = Path(__file__).parent / "持仓.csv"

# ---------------- 推荐设置（本金默认值，存仓库文件，网页与每日邮件共用） ----------------
SETTINGS_FILE = Path(__file__).parent / "推荐设置.json"
# 字段含义：capital = 本金（元）。JSON 不支持注释，含义写在这里
DEFAULT_SETTINGS = {"capital": 2200}


def load_settings() -> dict:
    """读取本地仓库里的 推荐设置.json，返回 dict；文件缺失/损坏时返回默认值 {"capital": 2200}"""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {"capital": float(data.get("capital", DEFAULT_SETTINGS["capital"]))}
    except Exception:
        return dict(DEFAULT_SETTINGS)


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
        elif news_risk_impl(code):               # 内部调用走 _impl，避免嵌套缓存
            act = "⚠️ 有利空新闻，检查"
        else:
            act = "✅ 继续持有"
        rows.append({"代码": code, "成本价": cost, "现价": now,
                     "浮动盈亏%": round(pnl, 2), "操作": act})
    return pd.DataFrame(rows)

def auto_pick_count(capital: float) -> int:
    """估算只数（用于股价上限计算）：本金÷1000，最少1只，不设上限。
    最终只数由 best_plan 按'收益-手续费最大化'自动定。"""
    return max(1, int(capital // 1000))


# ========== 预写死功能（v1.8 全开，置信度折扣在调用处应用） ==========
def _correlation_filter(ranked: pd.DataFrame, spot: pd.DataFrame,
                         threshold: float = 0.8, n_done: int = 0) -> pd.DataFrame:
    """相关性过滤：候选股间相关性 > threshold 的只留排名更高的一只。
    注：用日线收盘价 Spearman 相关（免费数据能拿到）。
    v1.8 置信度折扣：n_done < 置信度阈值 时，threshold 向上抬（更宽松，少删票），
    避免小样本下相关系数不稳定而误删好票。
      实际阈值 = threshold + (1 - confidence) * (0.95 - threshold)
      confidence=0 → 阈值 0.95（几乎不删）；confidence=1 → 原 threshold。"""
    conf = _confidence(n_done, _CORR_CONF_TH)
    actual_threshold = threshold + (1 - conf) * (0.95 - threshold)
    try:
        codes = ranked["代码"].head(20).tolist()
        close_mat = pd.DataFrame()
        for c in codes:
            h = spot[spot["代码"] == c].sort_values("交易日")
            col = h.set_index("交易日")["收盘"].rename(c)
            if len(col) >= 60:
                close_mat[c] = col
        if close_mat.shape[1] < 2:
            return ranked
        corr = close_mat.corr().abs()
        keep = set(ranked["代码"].head(1).tolist())
        for c in ranked["代码"].iloc[1:].head(19):
            if c not in corr.columns:
                keep.add(c); continue
            if any(corr.loc[c, k] >= actual_threshold for k in keep if k in corr.columns):
                continue  # 和已保留的高度相关，跳过
            keep.add(c)
        return ranked[ranked["代码"].isin(keep)].reset_index(drop=True)
    except Exception:
        return ranked


def _kelly_position(ranked: pd.DataFrame, capital: float,
                   min_wr: float = 0.55, max_ratio: float = 0.15,
                   n_done: int = 0) -> pd.Series:
    """凯利公式仓位分配。f = (bp - q)/b，f 过大时按 max_ratio 封顶。
    b=平均赔率（平均盈亏%的绝对值比），p=胜率，q=1-p。
    v1.8 置信度折扣：n_done < 置信度阈值 时，f 乘以 confidence，
    样本越少仓位越保守，避免小样本胜率噪声导致过度集中。
      confidence=0 → f 几乎为 0（接近等额）；confidence=1 → 原凯利。"""
    conf = _confidence(n_done, _KELLY_CONF_TH)
    try:
        if "胜率" not in ranked.columns or "扣费总收益" not in ranked.columns:
            return pd.Series(1.0, index=ranked.index)
        p = float(ranked["胜率"].clip(0, 1).mean())
        avg_gain = ranked[ranked["扣费总收益"] > 0]["扣费总收益"].mean()
        avg_loss = abs(ranked[ranked["扣费总收益"] < 0]["扣费总收益"].mean()) + 1e-9
        b = avg_gain / avg_loss
        if p < min_wr or b <= 1e-6:
            return pd.Series(1.0, index=ranked.index)  # 胜率不够降级等额
        f = (b * p - (1 - p)) / b
        f = max(0.01, min(max_ratio, f)) * conf  # 置信度折扣
        # 各票按预期收益加权分 f 的预算
        weights = ranked["扣费总收益"].clip(lower=0).abs()
        if weights.sum() < 1e-6:
            return pd.Series(1.0, index=ranked.index)
        alloc = (weights / weights.sum()) * f
        # 归一化让总比例等于 f（或等额，取大的那个）
        total = alloc.sum()
        if total < 0.05:
            return pd.Series(1.0, index=ranked.index)
        return alloc
    except Exception:
        return pd.Series(1.0, index=ranked.index)


def best_plan(ranked: pd.DataFrame, capital: float):
    """只数不写死：枚举 1~min(候选数,20)，选'预期总收益−买卖全部摩擦成本'最大的方案。
    预期收益 = 各票扣费回测收益% × 实际买入金额；摩擦含最低5元佣金
    （本金越小固定佣金越吃亏，最优只数自然越少；候选越多收益越高，只数自然越多）。
    v1.8: 从 ranked.attrs 取 n_done / hists，传给 allocate 做均值方差优化。
    返回 (买入方案, 最优只数)。"""
    ret_map = ranked.set_index("代码")["扣费总收益"]
    n_done = int(ranked.attrs.get("n_done", 0) or 0)
    hists = ranked.attrs.get("hists") or {}
    best_df, best_n, best_net = None, 0, -1e18
    for n in range(1, min(len(ranked), 20) + 1):
        plan, _left = allocate(ranked.head(n), capital, n_done=n_done, hists=hists)
        amt = plan["买入股数"] * plan["现价"]
        gross = float((amt * plan["代码"].map(ret_map) / 100).sum())
        buy_fee = float((plan["投入(含佣金)"] - amt).sum())
        sell_fee = float((5.0 + amt * 0.00051).sum())   # 卖出：最低佣金5元+印花税0.05%+过户费
        net = gross - buy_fee - sell_fee
        if net > best_net:
            best_df, best_n, best_net = plan, n, net
    return best_df, best_n


# ---------------- 固定参数（本金由页面输入，其余固化在后台） ----------------
MIN_PRICE = 2.0         # 股价下限：避开1元股退市风险区
CANDIDATES = 15         # 初筛后进入回测验证的候选数量
PICKS = 3               # 最终输出股票数（兜底默认值，实际只数由 auto_pick_count(本金) 决定）
HISTORY_DAYS = 500      # 回测用的历史天数

# ========== 自动进化配置（evolution.py 可能改 strategy_config.json） ==========
_CFG_PATH = Path(__file__).parent / "strategy_config.json"
def _cfg() -> dict:
    if _CFG_PATH.exists():
        try:
            return json.load(open(_CFG_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}

import json  # noqa: E402

# 候选池大小：只改 strategy_config.json 的「候选池」即可扩容，不用改代码
_cfg_pool = _cfg().get("候选池", {})
CANDIDATES = int(_cfg_pool.get("网页端", CANDIDATES))
CANDIDATES_PUSH = int(_cfg_pool.get("邮件推送", CANDIDATES))

_risk = _cfg().get("风控", {})
STOP_LOSS_PCT = float(_risk.get("止损%", 0.05))
TAKE_PROFIT_PCT = float(_risk.get("止盈%", 0.10))
STOP_LOSS_PCT = STOP_LOSS_PCT  # 用 config 的覆盖
STOP_LOSS_PCT = STOP_LOSS_PCT  # 保持变量名不变（下游引用）
# 熔断阈值
_cfg_fus = _risk
FUSE_LOSS_STOP = int(_cfg_fus.get("熔断连亏停", 10))
FUSE_LOSS_HALF = int(_cfg_fus.get("熔断连亏降", 5))
FUSE_DD_STOP = float(_cfg_fus.get("熔断回撤停", 0.25))
FUSE_DD_HALF = float(_cfg_fus.get("熔断回撤降", 0.15))

# 仓位管理开关（v1.8 全开 + 置信度折扣：功能从第一天启用，效果按实笔数打折）
_cfg_pos = _cfg().get("仓位管理", {})
_USE_CORR_FILTER = bool(_cfg_pos.get("相关性过滤", {}).get("启用", False))
_CORR_THRESHOLD = float(_cfg_pos.get("相关性过滤", {}).get("阈值", 0.80))
_CORR_CONF_TH = int(_cfg_pos.get("相关性过滤", {}).get("置信度阈值", 30))
_USE_KELLY = bool(_cfg_pos.get("凯利公式", {}).get("启用", False))
_KELLY_MIN_WR = float(_cfg_pos.get("凯利公式", {}).get("最小胜率", 0.55))
_KELLY_MAX_RATIO = float(_cfg_pos.get("凯利公式", {}).get("最大仓位比例", 0.15))
_KELLY_CONF_TH = int(_cfg_pos.get("凯利公式", {}).get("置信度阈值", 30))
_USE_MV_OPT = bool(_cfg_pos.get("均值方差优化", {}).get("启用", False))
_MV_CONF_TH = int(_cfg_pos.get("均值方差优化", {}).get("置信度阈值", 30))


def _confidence(n_done: int, threshold: int) -> float:
    """置信度 = min(1.0, n_done / threshold)。样本不足时效果向保守方向靠拢。"""
    if threshold <= 0:
        return 1.0
    return max(0.0, min(1.0, n_done / threshold))

# ---------------- 历史推荐复盘（打开页面自动执行，无需点击） ----------------
REVIEW_FILE = Path(__file__).parent / "推送记录.csv"   # GitHub Actions 每日推送留档
REVIEW_N = 10            # 默认复盘最近10条推荐（低温期可能多天无推荐，不限于7天）
REVIEW_DAYS = 20         # 推荐后最长跟踪20个交易日（与策略持有期一致），触发止盈/止损即出结果


def _index_series(start: pd.Timestamp, end: dt.date):
    """沪深300日线（日期->收盘），供复盘做基准对比；接口失败返回 None，不影响复盘本身。"""
    try:
        idx = ak.index_zh_a_hist(symbol="000300", period="daily",
                                 start_date=(start - dt.timedelta(days=10)).strftime("%Y%m%d"),
                                 end_date=end.strftime("%Y%m%d"))
        s = pd.Series(idx["收盘"].astype(float).values, index=pd.to_datetime(idx["日期"]))
        return s.sort_index()
    except Exception:
        return None


def review_impl() -> pd.DataFrame:
    """复盘最近 REVIEW_N 条推送记录：
    推荐日后最长跟踪 REVIEW_DAYS 个交易日（与策略持有期一致），逐日判定谁先触发
    止盈(+10%)/止损(-5%)；满20日未触发按到期收盘定盈亏（🏁）；
    未满20日且未触发标 ⏳持有中，不计入胜率。同时给出同期沪深300涨跌幅作基准。
    推送记录.csv 不存在或为空时返回空表，安静跳过。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    if not REVIEW_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(REVIEW_FILE, dtype={"代码": str})
    except Exception:
        return pd.DataFrame()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.dropna(subset=["日期", "代码", "现价"]).tail(REVIEW_N)
    # 同一天可能推送多次（手动重发/冗余触发），用「推送时间」区分批次；旧记录没有该列则留空
    if "推送时间" not in df.columns:
        df["推送时间"] = ""
    end_d = dt.date.today()
    idx_s = _index_series(pd.Timestamp(str(df["日期"].min())), end_d)   # 基准指数，只拉一次
    rows = []
    for _, r in df.iterrows():
        try:
            rec_date = pd.Timestamp(str(r["日期"]).strip())
            price0 = float(r["现价"])
            shares = int(float(r["买入股数"]))
        except Exception:
            continue
        if price0 <= 0:
            continue
        code = str(r["代码"]).strip().zfill(6)[-6:]
        try:
            h = get_history_impl(code, rec_date.strftime("%Y%m%d"),
                                 end_d.strftime("%Y%m%d"))
        except Exception:
            h = None
        future = pd.Series(dtype=float)
        fdates = pd.Series(dtype="datetime64[ns]")
        if h is not None and len(h) > 0:
            fh = h[pd.to_datetime(h["日期"]) > rec_date]
            future = fh["收盘"].astype(float).reset_index(drop=True)
            fdates = pd.to_datetime(fh["日期"]).reset_index(drop=True)
        stop_px, take_px = price0 * (1 - STOP_LOSS_PCT), price0 * (1 + TAKE_PROFIT_PCT)
        label, judge_px, judge_i = None, None, None
        n = min(len(future), REVIEW_DAYS)          # 推荐日后已过去的交易日数（最多跟踪20天）
        for i in range(n):                          # 逐日判定，谁先触发算谁
            c = float(future.iloc[i])
            if c >= take_px:
                label, judge_px, judge_i = "🎯止盈", c, i
                break
            if c <= stop_px:
                label, judge_px, judge_i = "🛑止损保命", c, i
                break
        if label is None:
            if n >= REVIEW_DAYS:
                judge_i = REVIEW_DAYS - 1
                judge_px = float(future.iloc[judge_i])
                label = "🏁到期盈利" if judge_px >= price0 else "🏁到期亏损"
            else:
                judge_i = n - 1
                judge_px = float(future.iloc[n - 1]) if n > 0 else price0
                label = f"⏳持有中(第{n}天)" if n > 0 else "⏳持有中(当日新推)"
        judge_date = (fdates.iloc[judge_i]
                      if judge_i is not None and judge_i >= 0 else rec_date)
        # 同期沪深300涨跌幅：推荐日（或之前最近交易日）收盘 -> 判定日收盘
        bench = None
        if idx_s is not None:
            try:
                b0 = idx_s.loc[:rec_date].iloc[-1]
                b1 = idx_s.loc[:judge_date].iloc[-1]
                bench = round((b1 / b0 - 1) * 100, 2)
            except Exception:
                bench = None
        row = {"日期": rec_date.strftime("%Y-%m-%d"),
               "推送时间": ("" if pd.isna(r.get("推送时间")) else str(r.get("推送时间")).strip()),
               "代码": code,
               "名称": str(r.get("名称", "")).strip(),
               "推荐价": round(price0, 2), "现价": round(judge_px, 2),
               "盈亏%": round((judge_px / price0 - 1) * 100, 2), "结果": label,
               "同期沪深300%": bench,
               "_盈亏额": round((judge_px - price0) * shares, 1),
               "_done": not label.startswith("⏳"), "_saved": 0.0}
        # 止损挽回金额：若止损后继续跌到满期，估算纪律避免了多少额外亏损
        if label == "🛑止损保命" and len(future) >= REVIEW_DAYS:
            day_end = float(future.iloc[REVIEW_DAYS - 1])
            row["_saved"] = round(max(0.0, judge_px - day_end) * shares, 1)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["_batch"] = out["日期"] + " " + out["推送时间"]      # 批次键：日期+推送时间
    # 最新批次在前：先按日期倒序，同一天再按推送时间倒序（旧记录无时间，排最后）
    return out.sort_values(["日期", "推送时间"], ascending=[False, False]).reset_index(drop=True)


# 缓存包装：web UI 一律调 review_performance；ttl=6小时，同一天内重复打开秒开
review_performance = st.cache_data(ttl=21600, show_spinner="正在自动复盘最近推荐表现...")(review_impl)


def review_conclusion(rv: pd.DataFrame) -> str:
    """根据复盘结果自动生成一句大白话结论"""
    done = rv[rv["_done"]]
    if len(done) == 0:
        return "推荐都还在20日持有期内，暂无胜负结果，耐心等复盘。"
    wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
    wr = wins / len(done) * 100
    avg = done["盈亏%"].mean()
    if wr >= 60 and avg > 0:
        text = f"近 {len(done)} 笔已出结果推荐胜率 {wr:.0f}%、平均每笔 {avg:+.1f}%，整体赚钱，继续保持纪律。"
    elif wr < 40 or avg < 0:
        text = (f"近 {len(done)} 笔已出结果推荐胜率 {wr:.0f}%、平均每笔 {avg:+.1f}%，整体偏弱，"
                f"建议降低单笔仓位，等大盘回暖再加大投入。")
    else:
        text = (f"近 {len(done)} 笔已出结果推荐胜率 {wr:.0f}%、平均每笔 {avg:+.1f}%，"
                f"有赚有亏，严格执行止损止盈即可。")
    # 基准对比：同期沪深300平均涨跌幅与超额收益（跑赢大盘才算真赚）
    if "同期沪深300%" in done.columns:
        d2 = done.dropna(subset=["同期沪深300%"])
        if len(d2) > 0:
            bm = d2["同期沪深300%"].mean()
            ex = (d2["盈亏%"] - d2["同期沪深300%"]).mean()
            text += f" 同期沪深300平均 {bm:+.1f}%，平均超额 {ex:+.1f}%。"
    saved = rv["_saved"].sum()
    if saved > 0:
        text += f" 止损纪律避免了约 {saved:.0f} 元额外亏损。"
    return text

APP_VERSION = "v1.16"       # v1.16 空仓日邮件也带「上一次推送表现」（此前 note 分支提前返回会漏掉）

st.set_page_config(page_title="每日选股", page_icon="📈", layout="centered")
st.title("📈 每日选股")
st.caption(APP_VERSION)

# ---------------- 大盘温度横幅（低温期提示空仓） ----------------
idx_close, idx_ma60, market_cold = market_temperature()
if market_cold:
    st.error(f"❄️ 大盘低温期：沪深300（{idx_close:.0f}）在60日线（{idx_ma60:.0f}）下方，"
             f"系统性下跌段。默认建议空仓，执意选股请勾选下方开关。")
    force_run = st.checkbox("我知道风险，仍要选股")
else:
    st.success(f"☀️ 大盘温度正常：沪深300（{idx_close:.0f}）位于60日线上方")
    force_run = True

# ---------------- 本金输入在侧边栏（见下方 sidebar 区块），此处仅保留说明 ----------------
# capital / n_picks / MAX_PRICE 均在侧边栏代码里赋值，本文件后面的选股逻辑直接使用

# ---------------- 我的持仓（自动读本地持仓.csv，每天选股时自动体检） ----------------
with st.expander("💼 持仓体检（自动读持仓.csv，选股时自动执行）", expanded=True):
    holdings = load_holdings()
    if len(holdings) == 0 or holdings.iloc[0].isna().all():
        st.info("📭 暂无持仓。有持仓想自动体检？编辑同目录下的 持仓.csv（列：代码,成本价,股数），下次选股自动识别。")
        holdings = pd.DataFrame(columns=["代码", "成本价", "股数"])
    else:
        st.caption(f"已加载 {len(holdings)} 只持仓，选股时自动体检止损/止盈/利空。")

# ---------------- 侧边栏：只保留本金输入（行情/复盘/因子/熔断全部自动） ----------------
with st.sidebar:
    capital = st.number_input("💰 本金（元）", min_value=1000, max_value=100000,
                              value=int(load_settings()["capital"]), step=100,
                              help="本次修改仅当前页面生效；永久修改请到 GitHub 仓库编辑 推荐设置.json")
    n_picks = auto_pick_count(capital)
    st.caption(f"只数由本金+预期收益自动计算（约 {n_picks} 只起，单笔≥1000元）")
    st.caption(f"行情快照时间：{spot_cache_time()}")

# 股价上限：保证各只买100股也买得起；本金越大上限越宽，最高20元（保持低价股策略）
MAX_PRICE = max(MIN_PRICE + 0.5, min(20.0, capital / (n_picks * 100) * 0.95))

# ---------------- 选股流水线（全部后台执行） ----------------
def _board_ok(code: str) -> bool:
    """剔除小资金买不了或风险过高的板块：
    300/301创业板(需10万门槛)、688科创板(50万门槛)、8/4/920北交所(50万门槛)"""
    return not code.startswith(("300", "301", "688", "8", "4", "920"))


def screen_impl() -> pd.DataFrame:
    """第一步：从全市场快照里做硬性初筛，只留下安全、买得起、流动性够的股票。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    df = get_spot_impl()                                 # 内部调用走 _impl，避免嵌套缓存
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    lhb = lhb_codes_impl()                               # 龙虎榜名单（近一月）
    name = df["名称"].astype(str)
    mask = (
        df["代码"].map(_board_ok)
        & ~df["代码"].isin(lhb)                           # 剔除近一月上过龙虎榜的游资爆炒股
        & ~name.str.contains("ST|退", na=False)          # 剔除ST/退市股
        & df["最新价"].between(MIN_PRICE, MAX_PRICE)      # 价格区间：2200元买2只也买得起
        & (df["成交额"] > 5e7)                            # 日成交额>5000万，保证能顺利卖出
        & df["市盈率-动态"].between(0, 80)                # 剔除亏损股和泡沫股
        & df["总市值"].between(3e9, 2e11)                 # 30亿~2000亿，避开微盘操纵股和超级大盘股
        & df["涨跌幅"].between(-9, 7)                     # 剔除已涨停追不进去的、和跌停的
        & df["量比"].between(0.5, 10)                     # 量能正常，剔除异常放量庄股
    )
    return df[mask]


# web UI 沿用原名 screen；GitHub Actions 等裸运行环境请调 screen_impl
screen = screen_impl


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


import factors  # L1 多因子层：动量/波动率/换手/估值 + 行业中性 + 20日IC均值配权


def _num(v):
    """安全转float：None/NaN/非数字一律返回 None（估值因子缺数据时不参与）"""
    try:
        f = float(v)
        return f if f == f else None
    except Exception:
        return None


def verify_impl(cands: pd.DataFrame, progress) -> pd.DataFrame:
    """第二步：对初筛候选逐只跑 MA5/20 金叉策略回测。
    回测已含：佣金万2.5(最低5元)+印花税+滑点、冷却期20天、最短持有5天。
    用扣费后的真实收益打分，赚不回手续费的股票直接被淘汰。
    （无缓存装饰器，供裸运行脚本直接调用；progress 为进度回调，可传 lambda p, t: None）"""
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=HISTORY_DAYS)
    rows, hists, meta = [], {}, {}          # hists/meta：供因子面板与行业中性化使用
    fundamentals = {}                         # {代码: baostock财报DataFrame}，给基本面因子用
    total = len(cands)
    # ===== 逐票拉候选股基本面（Baostock 历史财报，免费，消除估值因子前视偏差）=====
    try:
        import fundamental as fd
        for fi, cd in enumerate(cands["代码"].astype(str).str.zfill(6).tolist(), 1):
            progress(0.3 * fi / total, f"拉取财报 {fi}/{total}（baostock 串行，今日首次较慢）")
            try:
                fundamentals[cd] = fd.fetch_fundamental(cd)
            except Exception:
                continue                            # 单只失败不影响其他票
    except Exception:
        fundamentals = {}  # Baostock 整体不可用，降级：基本面因子全 NaN，自动跳过
    # ===== K线+舆情 8 路并发预取（纯 HTTP 线程安全；baostock 单连接不能并发，保持串行）=====
    codes_all = cands["代码"].astype(str).str.zfill(6).tolist()
    hists_pre, news_pre = {}, {}
    try:
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_one(cd):
            try:
                h = get_history_impl(cd, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
            except Exception:
                h = None
            try:
                n = news_risk_impl(cd)
            except Exception:
                n = None
            return cd, h, n

        with ThreadPoolExecutor(max_workers=8) as pool:
            for done, (cd, h, n) in enumerate(pool.map(_fetch_one, codes_all), 1):
                hists_pre[cd], news_pre[cd] = h, n
                progress(0.3 + 0.35 * done / total, f"并发下载K线+舆情 {done}/{total}")
    except Exception:
        hists_pre, news_pre = {}, {}        # 预取整体失败：回落到循环内串行拉取
    try:
        ind_map = industry_map_impl(cands["代码"].astype(str).tolist())   # 行业映射（磁盘缓存）
    except Exception:
        ind_map = {}
    for i, (_, r) in enumerate(cands.iterrows()):
        code = str(r["代码"]).zfill(6)
        try:
            if code in hists_pre:                        # 并发预取结果（None 表示拉取失败）
                h = hists_pre[code]
            else:                                        # 预取未覆盖（异常回落）才串行补拉
                h = get_history_impl(code, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"))
            if h is None or len(h) < 120:
                continue
            pe, pb = _num(r.get("市盈率-动态")), _num(r.get("市净率"))
            if pe is not None and pe <= 0:
                pe = None
            if pb is not None and pb <= 0:
                pb = None
            fv = factors.latest_factors(h, pe, pb,
                                        fundamental=fundamentals.get(code))   # 加基本面因子
            hists[code] = h
            meta[code] = {"行业": ind_map.get(code, "未分类"), "PE": pe, "PB": pb}
            sig = generate_signals(h, "ma_cross", {"fast": 5, "slow": 20},
                                   cooldown=20, min_hold_days=5)
            _, _, bs = backtest(sig, initial_cash=float(capital))
            fee = float(bs["累计手续费"].rstrip("元"))
            avg_win, avg_loss = _trade_stats(sig)
            risk_hits = news_pre[code] if code in news_pre else news_risk_impl(code)  # 舆情排雷（优先用并发预取结果）
            if risk_hits is None:
                continue                            # 舆情拉取失败：保持原行为，跳过该票
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
                "动量20": fv["动量20"], "波动率20": fv["波动率20"], "换手20": fv["换手20"],
                "EP": fv["EP"], "BP": fv["BP"], "行业": meta[code]["行业"],
                "趋势向上": trend_ok,
                "有雷": len(risk_hits) > 0, "利空标题": "；".join(risk_hits[:2]),
            })
        except Exception:
            continue                                    # 单只失败不影响整体
        progress(0.65 + 0.35 * (i + 1) / total, f"回测验证中 {i + 1}/{total}：{r['名称']}")
    res = pd.DataFrame(rows)
    # 因子IC检验：行业中性化 → 逐交易日横截面IC → 最近20日均值定权
    try:
        npanel = factors.neutralize(factors.build_panel(hists, meta, fundamentals=fundamentals))
        ics = factors.ic_series(npanel)
        ic, ir = factors.ic_summary(ics)
        res.attrs["ic"], res.attrs["icir"] = ic, ir
        res.attrs["ic_n"] = int(len(ics))               # 参与检验的交易日数（置信度参考）
        # 因子衰减监控：把今天的IC存进历史CSV（每天append一次，供后续趋势判断）
        try:
            import factor_learning
            factor_learning.append_factor_ic_history(ic, pd.Timestamp.now().strftime("%Y-%m-%d"))
        except Exception:
            pass
    except Exception:
        res.attrs["ic"], res.attrs["icir"], res.attrs["ic_n"] = {}, {}, 0
    # 基础权重（历史IC）
    base_icw = factors.ic_weights(res.attrs.get("ic") or {})

    # ========== 复盘反哺权重：用实盘盈亏融合历史IC（比例根据实笔数动态调整） ==========
    blended_icw, actual_ic_used, blend_ratios_used = {}, {}, (1.0, 0.0)
    cb_level, cb_msg, cb_diag = 0, "无复盘数据", {}
    n_done = 0  # v1.8: 已结算实盘笔数，用于置信度折扣
    try:
        import factor_learning
        rv = review_impl()
        # 策略熔断：实盘风控最后一道防线
        cb_level, cb_msg, cb_diag = factor_learning.circuit_breaker(rv)
        actual_ic = factor_learning.actual_ic_from_review(rv, res)
        n_done = int(cb_diag.get("_n_done", len(rv) if rv is not None else 0))
        n_done = cb_diag.get("_n_done", n_done)   # 占位（circuit_breaker返回值不含n_done）
        # 重新算一遍n_done（避免改circuit_breaker返回签名）
        if rv is not None and len(rv) > 0:
            n_done = int(rv.get("_done", False).sum() if "_done" in rv.columns else len(rv))
        blend_ratios_used = factor_learning.blend_ratios(n_done)
        if actual_ic:
            _, blended_icw, blend_ratios_used = factor_learning.blended_ic_weights(
                {f: v for f, v in (res.attrs.get("ic") or {}).items()},
                actual_ic, n_actual=n_done)
            actual_ic_used = actual_ic
    except Exception:
        pass
    final_icw = blended_icw if blended_icw else base_icw

    # ========== 因子衰减过滤：连续30天|IC|<0.03的因子临时剔除 ==========
    health_df = pd.DataFrame()
    try:
        import factor_learning
        health_df = factor_learning.factor_health(res.attrs.get("ic") or None)
        if len(health_df) > 0:
            final_icw = factor_learning.filter_decayed_factors(final_icw, health_df)
    except Exception:
        pass

    res.attrs["ic_weights"] = final_icw
    res.attrs["ic_weights_base"] = base_icw
    res.attrs["ic_actual"] = actual_ic_used
    res.attrs["blend_ratios"] = blend_ratios_used
    res.attrs["factor_health"] = health_df
    res.attrs["cb_level"], res.attrs["cb_msg"], res.attrs["cb_diag"] = cb_level, cb_msg, cb_diag
    res.attrs["n_done"] = int(n_done)          # v1.8: 置信度折扣用
    res.attrs["hists"] = hists                  # v1.8: 均值方差优化用
    return res


# web UI 沿用原名 verify；GitHub Actions 等裸运行环境请调 verify_impl
verify = verify_impl


def pick_top3(res: pd.DataFrame, n: int = PICKS) -> pd.DataFrame:
    """第三步：综合打分。
    扣费收益占大头，胜率其次，趋势向上是必须条件（不满足扣大分）。"""
    res = res.copy()
    # 百分位打分（0~100），避免不同量纲直接相加
    res["收益分"] = res["扣费总收益"].rank(pct=True) * 100
    res["胜率分"] = res["胜率"].rank(pct=True) * 100
    res["趋势分"] = res["趋势向上"].map({True: 100.0, False: 0.0})
    res["总分"] = res["收益分"] * 0.5 + res["胜率分"] * 0.2 + res["趋势分"] * 0.3
    # L1 多因子层：权重来自IC检验（无效因子已被剔除）；无有效因子时保持原评分不变
    icw = res.attrs.get("ic_weights") or {}
    if icw:
        ind = res["行业"] if "行业" in res.columns else pd.Series("未分类", index=res.index)
        # 因子贡献拆解：每个因子的 (中性化得分 × 权重 × 100) 存进单独列
        contribs = {}
        for c, w in icw.items():
            if c not in res.columns:
                continue
            ns = factors.neutral_score(res[c], ind).fillna(0.5)
            contribs[f"因子_{c}_贡献"] = ns * w * 100
            res[f"因子_{c}_贡献"] = contribs[f"因子_{c}_贡献"]
        fscore = sum(contribs.values())
        res["总分"] = res["总分"] * 0.6 + fscore * 0.4
    res = res.sort_values("总分", ascending=False)
    # 只保留扣费后还能赚钱、且近7天无利空新闻的股票
    res = res[(res["扣费总收益"] > 0) & (~res["有雷"])]
    return res.head(n)


def allocate(picks: pd.DataFrame, budget0: float,
             n_done: int = 0, hists: dict | None = None) -> pd.DataFrame:
    """第四步：把本金尽量买满。
    v1.8 仓位分配优先级：均值方差优化 > 凯利公式 > 等额分配
      1. 均值方差优化（PyPortfolioOpt 或 numpy 闭式解）：按 max_sharpe 给权重
      2. 凯利公式：胜率高/赔率大的票拿更多预算
      3. 等额分配：兜底
    三者都带置信度折扣（样本不足时向等额靠拢）。
    每只按目标买100股的整数倍，买完后剩余的钱自动顺延给下一只，最后一只尽量把预算花完。"""
    cfg_pos = _cfg().get("仓位管理", {}) if "_cfg" in dir() else {}
    hists = hists or {}

    # ===== 1. 均值方差优化（PyPortfolioOpt / numpy 兜底） =====
    opt_weights = None
    if _USE_MV_OPT and len(picks) > 1:
        try:
            import portfolio_optimizer as po
            codes = picks["代码"].astype(str).str.zfill(6).tolist()
            exp_ret = {c: float(picks.loc[picks["代码"].astype(str).str.zfill(6) == c, "扣费总收益"].iloc[0]) / 100
                       for c in codes if len(picks.loc[picks["代码"].astype(str).str.zfill(6) == c]) > 0}
            w, method = po.optimize_weights(codes, hists, exp_ret, n_done=n_done)
            if w:
                # 按 picks 的顺序对齐权重
                opt_weights = [w.get(c, 0.0) for c in codes]
        except Exception:
            opt_weights = None

    # ===== 2. 凯利公式（均值方差失败时降级） =====
    kelly_weights = None
    if opt_weights is None and _USE_KELLY and len(picks) > 1:
        try:
            kw = _kelly_position(picks, budget0,
                                 min_wr=float(cfg_pos.get("凯利公式", {}).get("最小胜率", 0.55)),
                                 max_ratio=float(cfg_pos.get("凯利公式", {}).get("最大仓位比例", 0.15)),
                                 n_done=n_done)
            if not (kw == 1.0).all():
                kelly_weights = kw.values
        except Exception:
            pass

    plan, budget = [], budget0
    n = len(picks)
    for i, (_, r) in enumerate(picks.iterrows()):
        price = r["最新价"]
        if opt_weights is not None:
            # 均值方差权重：按预算 × 权重 / 剩余权重和
            w_remain = float(sum(opt_weights[i:])) + 1e-9
            target = budget * float(opt_weights[i]) / w_remain
        elif kelly_weights is not None:
            # 凯利加权：按预算 × 权重 / 剩余权重和
            w_remain = float(kelly_weights[i:].sum()) + 1e-9
            target = budget * float(kelly_weights[i]) / w_remain
        else:
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
            "佣金(元)": round(fee, 2),
            "投入(含佣金)": invest,
            "盈利概率": round(r["胜率"], 0),
            "预计盈利(元)": est_win,
            "预计亏损(元)": -est_loss,
            "止损价(买入价×0.95)": round(price * (1 - STOP_LOSS_PCT), 2),
            "止盈参考(买入价×1.10)": round(price * (1 + TAKE_PROFIT_PCT), 2),
            "_win_prob": win_prob,
            "_est_win": est_win, "_est_loss": est_loss,
        })
        budget -= invest
    df = pd.DataFrame(plan)
    leftover = round(budget, 2)
    return df, leftover


# ---------------- 历史推荐复盘（打开页面自动执行，无需任何点击） ----------------
rv = review_performance()
with st.expander(f"📊 历史推荐复盘（最近{REVIEW_N}条，自动更新）", expanded=True):
    if len(rv) == 0:
        st.caption("暂无历史推荐，有推荐后会自动复盘。")
    else:
        done = rv[rv["_done"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("推荐总数", f"{len(rv)} 条")
        if len(done) > 0:
            wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
            m2.metric("胜率", f"{wins / len(done) * 100:.0f}%（{wins}/{len(done)}）")
            m3.metric("平均每笔盈亏", f"{done['盈亏%'].mean():+.2f}%")
            m4.metric("累计盈亏", f"{done['_盈亏额'].sum():+.0f} 元")
        else:
            m2.metric("胜率", "—")
            m3.metric("平均每笔盈亏", "—")
            m4.metric("累计盈亏", "—")
        _cols = ["名称", "推荐价", "现价", "盈亏%", "同期沪深300%", "结果"]
        _cfg = {"盈亏%": st.column_config.NumberColumn(format="%+.2f%%"),
                "同期沪深300%": st.column_config.NumberColumn(format="%+.2f%%")}
        # 按推送批次（日期+时间）分组显示，批次之间用红色横线隔开：
        # 既区分不同日期，也区分同一天里第几次推送
        for i, (_, g) in enumerate(rv.groupby("_batch", sort=False)):
            if i:
                st.markdown("<hr style='border:none;border-top:2px solid #ff4b4b;margin:12px 0'>",
                            unsafe_allow_html=True)
            _day, _tm = g["日期"].iloc[0], g["推送时间"].iloc[0]
            _when = f"{_day} {_tm}" if _tm else _day
            st.markdown(f"**📅 {_when} 推送 · {len(g)} 只**")
            st.dataframe(g[_cols], width="stretch", hide_index=True, column_config=_cfg)
        st.info("💡 " + review_conclusion(rv))
        if len(done) < 30:
            st.caption(f"⚠️ 已出结果仅 {len(done)} 笔（<30），统计意义不足，结论别太当真；"
                       "攒到30~60笔后再看胜率才有参考价值。")
        st.caption("判定规则：推荐日后最长跟踪20个交易日，收盘涨超+10%为🎯止盈、跌破-5%为🛑止损；"
                   "满20日未触发按到期收盘定盈亏（🏁）；现价为止盈/止损触发价或最新收盘；"
                   "未满20日且未触发的标 ⏳持有中，不计入胜率。")

# ---------------- 页面：一个按钮 + 一张结果表 ----------------
st.caption(f"本金 {capital:.0f} 元 · 只数由本金+预期收益自动优化 · 股价上限 {MAX_PRICE:.1f} 元 · "
           "已内置手续费（佣金万2.5，单笔最低5元；印花税0.05%卖出收取；含过户费/滑点）")
run = st.button("🔍 今日选股", type="primary", width='stretch')

if run:
    # 选股前自动刷新行情快照（联网失败降级用旧缓存，不阻塞选股）
    _cache_tag = spot_cache_time()
    if spot_recently_failed():          # 刚失败过，10分钟内不再干等100秒
        st.warning(f"行情接口刚拉取失败，10分钟内不再重试，沿用 {_cache_tag} 的缓存继续。")
    else:
        with st.spinner(f"正在获取最新行情（本地缓存：{_cache_tag}，云端首次约1~2分钟）..."):
            try:
                if not spot_fresh_enough(1800):   # 30分钟内刷新过快照就跳过，省约1分钟
                    refresh_spot()
            except Exception as e:
                if _cache_tag == "无缓存":
                    st.error(f"行情接口无响应（东财+腾讯备用源均失败），且本地没有可用缓存，暂时无法选股（{e}）。"
                             "云端刚重启时首次拉取较慢，请过几分钟再点一次。")
                    st.stop()
                st.warning(f"行情刷新失败，沿用 {_cache_tag} 的缓存继续：{e}")
    if market_cold and not force_run:
        st.error("大盘低温期，已拦截选股。请勾选上方「我知道风险，仍要选股」后再点。")
        st.stop()
    # 持仓体检（持仓已在页面加载时自动从 持仓.csv 读取）
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
    ranked = pick_top3(res, min(len(res), 20))   # 评分前20作候选池，只数交给收益优化器
    if len(ranked) == 0:
        st.warning("今天没有股票通过回测+排雷验证。空仓也是一种操作，明天再来。")
        st.stop()
    # 因子检验结果透明展示：哪些因子有效、权重多少、置信度如何，用户看得见
    _icv, _icw = res.attrs.get("ic") or {}, res.attrs.get("ic_weights") or {}
    _icir, _icn = res.attrs.get("icir") or {}, int(res.attrs.get("ic_n", 0) or 0)
    _actual_ic = res.attrs.get("ic_actual") or {}
    _health = res.attrs.get("factor_health")
    # 区分：哪些被衰减过滤掉了
    decayed_set = set(_health.loc[_health["健康状态"] == "衰减", "因子"].tolist()) if _health is not None and len(_health) > 0 else set()
    if _icw:
        blend_tag = "· 复盘反哺融合" if _actual_ic else ""
        st.caption(f"🧬 多因子（行业中性 · 近{factors.IC_AVG_N}日IC均值{blend_tag} · {_icn}个交易日截面）：" + "　".join(
            f"{k} IC={_icv.get(k, 0):+.3f} ICIR={_icir.get(k, 0):+.2f} 权重{_icw[k]:+.2f}"
            for k in _icw))
    else:
        if decayed_set:
            st.caption(f"🧬 因子衰减全灭（{decayed_set}连续≥30天IC失效），评分已退回回测原版。")
        else:
            st.caption(f"🧬 本期因子IC均值全部低于{factors.IC_MIN}门槛（{_icn}个交易日截面），"
                       "评分已自动退回回测原版。")
    if _actual_ic:
        # 复盘反哺的实盘IC（小样本，仅供参考）
        act_tag = "　".join(f"{k}实盘IC={v:+.3f}" for k, v in _actual_ic.items())
        hr, ar = res.attrs.get("blend_ratios", (1.0, 0.0))
        st.caption(f"📊 复盘反哺：历史IC {hr:.0%} + 实盘IC {ar:.0%}（动态比例，实笔数增加自动调权）→ {act_tag}")
    if decayed_set:
        st.caption(f"⛔ 衰减预警：{decayed_set}连续≥30天|IC|<{0.03}，已临时剔除，IC恢复自动归队。")
    if _icn and _icn < 60:
        st.caption(f"⚠️ 因子检验仅有 {_icn} 个交易日截面（<60），权重置信度偏低，仅供方向参考。")

    # ===== 策略熔断状态 =====
    cb_level = int(res.attrs.get("cb_level", 0) or 0)
    cb_msg = res.attrs.get("cb_msg", "") or ""
    cb_diag = res.attrs.get("cb_diag", {}) or {}
    cb_color = {0: "🟢", 1: "🟡", 2: "🔴", 3: "🟡", 4: "🔴"}.get(cb_level, "🟢")
    st.caption(f"{cb_color} 熔断状态：{cb_msg}（连亏{cb_diag.get('连亏笔数',0)}笔 / 累计{cb_diag.get('累计盈亏%',0):+.1f}% / 最大回撤{cb_diag.get('累计回撤%',0):+.1f}%）")
    if cb_level >= 2:
        st.error("🔴 已触发停策略！建议今天不要买入，手动检查复盘结果。")
    elif cb_level >= 1:
        st.warning("🟡 已触发降半仓！建议实际买入金额为页面显示的一半。")

    # ---- 相关性过滤（v1.8 全开，带置信度折扣） ----
    if _USE_CORR_FILTER:
        try:
            _n_done = int(ranked.attrs.get("n_done", 0) or 0)
            ranked = _correlation_filter(ranked, get_spot(),
                                         threshold=_CORR_THRESHOLD, n_done=_n_done)
        except Exception:
            pass
    plan, n_final = best_plan(ranked, float(capital))
    leftover = float(capital) - float(plan["投入(含佣金)"].sum())

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

    # ---- 交易成本明细：买入佣金 + 卖出预计费用（佣金/印花税/过户费） ----
    _amt = float((plan["买入股数"] * plan["现价"]).sum())
    _buy_fee = float((plan["投入(含佣金)"] - plan["买入股数"] * plan["现价"]).sum())
    _sell_fee = float((5.0 + plan["买入股数"] * plan["现价"] * 0.00051).sum())   # 最低佣金5元+印花税0.05%+过户费
    st.caption(f"💰 交易成本：买入佣金 **{_buy_fee:.2f} 元**（费率万2.5，单笔最低5元）· "
               f"卖出预计 **{_sell_fee:.2f} 元**（佣金+印花税0.05%+过户费）· "
               f"一买一卖合计 **{_buy_fee + _sell_fee:.2f} 元**，"
               f"占买入金额 {(_buy_fee + _sell_fee) / _amt * 100:.2f}%（约等于股价要涨这么多才回本）")

    st.success(f"今日选出 {len(plan)} 只，方案如下：")
    show = plan.drop(columns=["_win_prob", "_est_win", "_est_loss"])
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"盈利概率": st.column_config.NumberColumn(format="%.0f%%"),
                                "佣金(元)": st.column_config.NumberColumn(format="%.2f"),
                                "预计盈利(元)": st.column_config.NumberColumn(format="%+.1f"),
                                "预计亏损(元)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("预计盈利/亏损 = 投入 × 该股历史平均每笔盈利率/亏损率（已扣交易成本）；"
               "盈利概率 = 历史回测胜率。均为历史统计推算，不代表未来，不构成投资建议。")

    st.markdown("**照做就行：**")
    for _, p in plan.iterrows():
        st.markdown(
            f"- **{p['名称']}（{p['代码']}）**：挂单 **{p['现价']}** 元附近买 **{p['买入股数']} 股**；"
            f"止损=你的实际买入价×0.95（按现价约 **{p['止损价(买入价×0.95)']}** 元），跌破无条件卖；"
            f"止盈=实际买入价×1.10（按现价约 **{p['止盈参考(买入价×1.10)']}** 元），到价考虑卖。"
        )
    st.caption("规则只有两条：止损必须执行；单只持有超过20个交易日仍不涨，也卖出。"
               "本结果为历史数据回测推演，不构成投资建议，盈亏自负。")

    # ---- 顺手把本次选股结果发邮件（复用已算好的数据，不重跑） ----
    with st.spinner("正在发送邮件..."):
        ok, msg = send_email_with_results(plan, mined, float(capital), leftover,
                                          market_cold, idx_close, idx_ma60)
    if ok:
        st.success(f"📧 {msg}")
    else:
        st.caption(f"📧 邮件未发送：{msg}")


# ---------- 版本号：手动维护，每次改完功能 bump 一次（定义已前移到 st.title 上方，页面顶部直接显示） ----------

