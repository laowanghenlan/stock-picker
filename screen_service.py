"""选股引擎（纯 Python，不依赖 Streamlit）：初筛 → 回测验证 → 因子打分 → 仓位分配 → 复盘。
网页（app.py）与每日邮件（daily_push.py）共用本模块，保证同一口径。"""
import datetime as dt
import json
import time
from pathlib import Path

import akshare as ak
import factors  # L1 多因子层：动量/波动率/换手/估值 + 行业中性 + 20日IC均值配权
import pandas as pd
import requests

from data import (cninfo_risk_impl, get_history_impl, get_history_tx,
                  get_spot_impl, industry_map_impl, tushare_daily_basic)
from strategy import backtest, calibrate_win_rate, generate_signals

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


def market_temperature_impl() -> tuple:
    """大盘温度：沪深300最新收盘 vs 60日均线。
    返回 (最新收盘, MA60, 是否低温期)。低温期=收盘在MA60下方，系统性下跌段，应空仓。
    走 _index_series（东财5秒 + 新浪兜底），东财限流时也不至于启动卡10秒。"""
    try:
        end_d = dt.date.today()
        s = _index_series(pd.Timestamp(end_d - dt.timedelta(days=200)), end_d)
        if s is None or len(s) < 60:
            return 0.0, 0.0, False                 # 接口失败时不阻塞选股
        close = float(s.iloc[-1])
        ma60 = float(s.tail(60).mean())
        return close, ma60, close < ma60
    except Exception:
        return 0.0, 0.0, False


def lhb_codes_impl() -> set:
    """近一月上过龙虎榜的股票代码集合（游资爆炒区，小资金回避）。接口失败返回空集。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    try:
        lhb = ak.stock_lhb_stock_statistic_em(symbol="近一月")
        return set(lhb["代码"].astype(str).str.zfill(6))
    except Exception:
        return set()


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
    v1.17 概率校准：胜率先向 50% 做贝叶斯收缩（样本=完整卖出笔数），
    再进凯利公式，避免"10笔8胜"这类小样本极端值把仓位带偏。
      confidence=0 → f 几乎为 0（接近等额）；confidence=1 → 原凯利。"""
    conf = _confidence(n_done, _KELLY_CONF_TH)
    try:
        if "胜率" not in ranked.columns or "扣费总收益" not in ranked.columns:
            return pd.Series(1.0, index=ranked.index)
        n_col = "卖出笔数" if "卖出笔数" in ranked.columns else None
        probs = []
        for _, r in ranked.iterrows():
            n_s = int(r.get(n_col, 0) or 0) if n_col else int(r.get("交易次数", 0) or 0) // 2
            probs.append(calibrate_win_rate(float(r["胜率"]) / 100, n_s,
                                            prior=0.5, strength=20.0))
        p = float(pd.Series(probs).clip(0, 1).mean())
        avg_gain = ranked[ranked["扣费总收益"] > 0]["扣费总收益"].mean()
        avg_loss = abs(ranked[ranked["扣费总收益"] < 0]["扣费总收益"].mean()) + 1e-9
        b = avg_gain / avg_loss
        if p < min_wr or b <= 1e-6:
            return pd.Series(1.0, index=ranked.index)  # 校准后胜率不够，降级等额
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
IC_GATE = 0.02          # 训练期组合IC低于此值则今日空仓（网页与邮件同一口径）

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
_ml_cfg = _cfg().get("机器学习", {})
ML_ENABLED = bool(_ml_cfg.get("启用", True))            # LightGBM加分项总开关
ML_WEIGHT = max(0.0, min(0.5, float(_ml_cfg.get("权重", 0.20))))   # 占总分比例，上限50%
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
REVIEW_N = 10            # 默认复盘最近10个推送批次（按日期+推送时间整批保留，不切断批次）
REVIEW_DAYS = 20         # 推荐后最长跟踪20个交易日（与策略持有期一致），触发止盈/止损即出结果
REVIEW_CACHE_KEY = "v1.22"   # 版本升级强制刷新，避免旧缓存缺少股数/盈亏额等新字段


def _run_with_timeout(fn, *args, timeout=8, **kwargs):
    """带硬超时的调用：东财/akshare 在云端可能 hang（不抛异常），最多等 timeout 秒。
    超时返回 None，不阻塞页面渲染。线程不 join，避免 hang 住的连接拖死整个进程。"""
    import concurrent.futures as _cf
    ex = _cf.ThreadPoolExecutor(max_workers=1)
    try:
        return ex.submit(fn, *args, **kwargs).result(timeout=timeout)
    except Exception:
        return None
    finally:
        ex.shutdown(wait=False)


def _index_series(start: pd.Timestamp, end: dt.date):
    """沪深300日线（日期->收盘），供复盘做基准对比、大盘温度使用。
    新浪优先（快且稳定），失败再试东财；都失败返回 None，不影响复盘本身。"""
    try:
        u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
             "CN_MarketData.getKLineData?symbol=sh000300&scale=240&ma=no&datalen=120")
        j = requests.get(u, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).json()
        s = pd.Series([float(x["close"]) for x in j],
                      index=pd.to_datetime([x["day"] for x in j]))
        if len(s) >= 20:
            return s.sort_index()
    except Exception:
        pass
    idx = _run_with_timeout(
        ak.index_zh_a_hist, symbol="000300", period="daily",
        start_date=(start - dt.timedelta(days=10)).strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"), timeout=5)
    try:
        if idx is not None and len(idx):
            s = pd.Series(idx["收盘"].astype(float).values, index=pd.to_datetime(idx["日期"]))
            return s.sort_index()
    except Exception:
        pass
    return None


def _sina_history(code: str, datalen: int = 60, full: bool = False) -> pd.DataFrame:
    """新浪日线兜底（不复权，与推送时的现价同口径）。
    东财被限流时 get_history_impl 会直接报错，复盘靠这条路径仍能算出盈亏。
    full=True 返回完整 OHLCV（选股回测/因子要用最高最低价）；默认只回日期+收盘。"""
    n = str(code).strip().zfill(6)[-6:]
    pfx = "sh" if n.startswith(("6", "9")) else "sz"
    u = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
         f"CN_MarketData.getKLineData?symbol={pfx}{n}&scale=240&ma=no&datalen={int(datalen)}")
    try:
        j = requests.get(u, timeout=10, headers={"User-Agent": "Mozilla/5.0"}).json()
        if not isinstance(j, list) or not j:
            return pd.DataFrame()
        data = {"日期": pd.to_datetime([x["day"] for x in j]),
                "收盘": [float(x["close"]) for x in j]}
        if full:
            data.update({"开盘": [float(x["open"]) for x in j],
                         "最高": [float(x["high"]) for x in j],
                         "最低": [float(x["low"]) for x in j],
                         "成交量": [float(x["volume"]) for x in j]})
        return pd.DataFrame(data).sort_values("日期").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


_EM_HIST_FAIL_AT = 0.0     # 东财历史接口失败时刻：30分钟内直接跳过，避免每只票白等5秒


def _em_hist_down() -> bool:
    return _EM_HIST_FAIL_AT > 0 and (time.time() - _EM_HIST_FAIL_AT) < 1800


def _hist_any(code: str, start: str, end: str, try_em: bool = True, full: bool = False) -> pd.DataFrame:
    """取历史日线，按用途分两条降级链：
    full=True（选股回测，要前复权+OHLCV）：东财(5秒) → 腾讯qfq → 新浪不复权。
    full=False（复盘，要不复权现价口径）：新浪 → 东财(5秒) → 腾讯不复权。
    东财单步必须短超时——否则外层超时会把整个函数（含兜底）一起掐断，兜底永远用不上。"""
    global _EM_HIST_FAIL_AT

    def _em():
        global _EM_HIST_FAIL_AT
        if not try_em or _em_hist_down():
            return pd.DataFrame()
        try:
            h = _run_with_timeout(get_history_impl, code, start, end, timeout=5)
            if h is not None and len(h) > 0:
                return h
        except Exception:
            pass
        _EM_HIST_FAIL_AT = time.time()      # 熔断30分钟，后续候选直接走腾讯/新浪
        return pd.DataFrame()

    if full:
        h = _em()
        if len(h) > 0:
            return h
        h = get_history_tx(code, start, end, full=True, fq="qfq")
        if len(h) > 0:
            return h
        return _sina_history(code, datalen=500, full=True)

    h = _sina_history(code)
    if len(h) > 0:
        return h
    h = _em()
    if len(h) > 0:
        return h
    return get_history_tx(code, start, end, full=False, fq="")


def review_impl(_key: str = "") -> pd.DataFrame:
    """复盘最近 REVIEW_N 个推送批次（同一天多次推送按「推送时间」分批，整批保留）：
    _key 仅用于让缓存键随版本变化，避免升级后仍读到旧的错误缓存（业务上不使用）。
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
    df = df.dropna(subset=["日期", "代码", "现价"]).copy()
    # 同一天可能推送多次（手动重发/冗余触发），用「推送时间」区分批次；旧记录没有该列则留空
    if "推送时间" not in df.columns:
        df["推送时间"] = ""
    df["日期"] = df["日期"].astype(str).str.strip()
    df["推送时间"] = df["推送时间"].fillna("").astype(str).str.strip()
    df["代码"] = df["代码"].astype(str).str.strip().str.zfill(6).str[-6:]
    _cols = [c for c in ["日期", "推送时间", "代码", "名称", "现价", "买入股数"] if c in df.columns]
    df = df.drop_duplicates(subset=_cols, keep="last")     # 同一次推送被重复留档 → 去重
    df = df.sort_values("日期", kind="stable")             # 按日期排序，批次顺序才正确
    # 同一只票同一天只算一次（当天重复推送不重复计胜负），保留当天最早那次
    df = df.drop_duplicates(subset=["日期", "代码"], keep="first")
    df["_batch"] = df["日期"] + " " + df["推送时间"]
    # 取最近 REVIEW_N 个批次、整批保留（按行 tail 会把一批切成一半，复盘就缺票）
    _recent = df["_batch"].drop_duplicates().tolist()[-REVIEW_N:]
    df = df[df["_batch"].isin(_recent)].reset_index(drop=True)
    end_d = dt.date.today()
    idx_s = _index_series(pd.Timestamp(str(df["日期"].min())), end_d)   # 基准指数，只拉一次
    # 并行预取所有股票历史：东财 hang 时每只最多 8 秒，6 线程并行，总耗时≈8~10 秒而非串行累加
    import concurrent.futures as _cf
    _hist_cache: dict = {}
    _starts: dict = {}                     # 代码 -> 该票最早的推荐日（同一只票只拉一次历史）
    for _, r in df.iterrows():
        try:
            _c = str(r["代码"]).strip().zfill(6)[-6:]
            _d = pd.Timestamp(str(r["日期"]).strip())
        except Exception:
            continue
        _starts[_c] = min(_d, _starts.get(_c, _d))
    _jobs = list(_starts.items())
    if _jobs:
        _ex = _cf.ThreadPoolExecutor(max_workers=8)
        _futs = {_ex.submit(_run_with_timeout, _hist_any, cd,
                            sd.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"),
                            timeout=12): cd
                 for cd, sd in _jobs}
        for _f in _cf.as_completed(_futs):
            _cd = _futs[_f]
            try:
                _hist_cache[_cd] = _f.result()
            except Exception:
                _hist_cache[_cd] = None
        _ex.shutdown(wait=False)
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
        h = _hist_cache.get(code)
        future = pd.Series(dtype=float)
        fdates = pd.Series(dtype="datetime64[ns]")
        if h is not None and len(h) > 0:
            fh = h[pd.to_datetime(h["日期"]) > rec_date]
            future = fh["收盘"].astype(float).reset_index(drop=True)
            fdates = pd.to_datetime(fh["日期"]).reset_index(drop=True)
        base_px = price0
        if h is not None and len(h) > 0:
            _same = h.loc[pd.to_datetime(h["日期"]) == rec_date, "收盘"]
            if len(_same):
                _q0 = float(_same.iloc[-1])
                # 只有明显除权（偏差>15%）才用当日收盘对齐；盘中推送价与收盘价差几个点属正常波动，不能替换
                if _q0 > 0 and abs(_q0 / price0 - 1) > 0.15:
                    base_px = _q0
        stop_px, take_px = base_px * (1 - STOP_LOSS_PCT), base_px * (1 + TAKE_PROFIT_PCT)
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
                label = "🏁到期盈利" if judge_px >= base_px else "🏁到期亏损"
            else:
                judge_i = n - 1
                judge_px = float(future.iloc[n - 1]) if n > 0 else base_px   # 当日新推：起点=基准，盈亏必为0
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
               "股数": shares,
               "盈亏%": round((judge_px / base_px - 1) * 100, 2), "结果": label,
               "同期沪深300%": bench,
               "盈亏额": round((judge_px - base_px) * shares, 1),
               "_盈亏额": round((judge_px - base_px) * shares, 1),
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


def _board_ok(code: str) -> bool:
    """剔除小资金买不了或风险过高的板块：
    300/301创业板(需10万门槛)、688科创板(50万门槛)、8/4/920北交所(50万门槛)
    沪深300/500 成分股由上层 override_board 放行，不走这里"""
    return not code.startswith(("300", "301", "688", "8", "4", "920"))


def _csi_set() -> set:
    """沪深300 + 中证500 成分股（Baostock 专用接口 query_hs300_stocks/query_zz500_stocks）；失败返回空集。"""
    try:
        import baostock as bs
        out = set()
        if bs.login().error_code != "0":
            return out
        for q in (bs.query_hs300_stocks, bs.query_zz500_stocks):
            rs = q()
            while rs.error_code == "0" and rs.next():
                row = rs.get_row_data()   # [updateDate, code, code_name]
                cd = str(row[1]).split(".")[-1].zfill(6)
                if cd:
                    out.add(cd)
        bs.logout()
        print(f"CSI300/500 成分股 {len(out)} 只")
        return out
    except Exception as e:
        print(f"CSI 成分股获取失败：{e}")
        return set()


def screen_impl(max_price: float | None = None) -> pd.DataFrame:
    """第一步：从全市场快照里做硬性初筛，只留下安全、买得起、流动性够的股票。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    if max_price is None:
        _cap = float(load_settings()["capital"])
        _n = auto_pick_count(_cap)
        max_price = max(MIN_PRICE + 0.5, min(50.0, _cap / (_n * 100) * 0.95))
    df = get_spot_impl()                                 # 内部调用走 _impl，避免嵌套缓存
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    lhb = lhb_codes_impl()                               # 龙虎榜名单（近一月）
    name = df["名称"].astype(str)
    csi = _csi_set()
    board_ok = df["代码"].map(_board_ok) | df["代码"].isin(csi)    # CSI成分股放行创业板/科创板
    price_ok = df["最新价"].between(MIN_PRICE, max_price) | df["代码"].isin(csi)
    mask = (
        board_ok
        & ~df["代码"].isin(lhb)                           # 剔除近一月上过龙虎榜的游资爆炒股
        & ~name.str.contains("ST|退", na=False)          # 剔除ST/退市股
        & price_ok                                       # CSI成分股允许突破MAX_PRICE上限
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


def _num(v):
    """安全转float：None/NaN/非数字一律返回 None（估值因子缺数据时不参与）"""
    try:
        f = float(v)
        return f if f == f else None
    except Exception:
        return None


def news_and_announcements(code: str) -> list | None:
    """舆情排雷 = 东财新闻 + 巨潮公告（法定信披，立案/处罚/减持更权威）。
    两个源都失败才返回 None（上层按"无雷"降级并标注）；任一可用即返回合并去重结果。"""
    try:
        n = news_risk_impl(code)
    except Exception:
        n = None
    try:
        a = cninfo_risk_impl(code, days=7)
    except Exception:
        a = None
    if n is None and a is None:
        return None
    out, seen = [], set()
    for hit in (n or []) + (a or []):
        t = str(hit.get("标题", ""))
        if t and t not in seen:
            seen.add(t)
            out.append(hit)
    return out


def _fetch_one_news(cd):
    try:
        return news_and_announcements(cd)
    except Exception:
        return None


def verify_impl(cands: pd.DataFrame, progress,
                capital: float | None = None) -> pd.DataFrame:
    """第二步：对初筛候选逐只跑 MA5/20 金叉策略回测。
    回测已含：佣金万2.5(最低5元)+印花税+滑点、冷却期20天、最短持有5天。
    用扣费后的真实收益打分，赚不回手续费的股票直接被淘汰。
    （无缓存装饰器，供裸运行脚本直接调用；progress 为进度回调，可传 lambda p, t: None）
    capital: 回测本金，None 时取本地设置（裸运行/邮件端用默认值）。"""
    if capital is None:
        capital = float(load_settings()["capital"])
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=HISTORY_DAYS)
    rows, hists, meta = [], {}, {}          # hists/meta：供因子面板与行业中性化使用
    hist_failed, news_failed = [], []       # 取数失败名单：区分"数据源不可用"与"候选不合格"
    fundamentals = {}                         # {代码: baostock财报DataFrame}，给基本面因子用
    total = len(cands)
    # ===== Tushare Pro 可选增强：配置了 TUSHARE_TOKEN 才启用，补齐快照缺失的 PE/PB/换手率 =====
    try:
        ts_basic = tushare_daily_basic()
        if ts_basic is not None and len(cands):
            cands = cands.copy()
            cands["代码"] = cands["代码"].astype(str).str.zfill(6)
            cands = cands.merge(ts_basic, on="代码", how="left")
            for dst, src in (("市盈率-动态", "市盈率TTM_ts"), ("市净率", "市净率_ts"),
                             ("换手率", "换手率_ts")):
                if dst not in cands.columns:
                    cands[dst] = cands[src]
                else:
                    cands[dst] = pd.to_numeric(cands[dst], errors="coerce").fillna(cands[src])
    except Exception:
        pass                                    # 增强失败不影响原有快照字段
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
                # 东财(熔断)→腾讯qfq→新浪：东财限流时不再整批候选归零
                h = _hist_any(cd, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"), full=True)
            except Exception:
                h = None
            n = _fetch_one_news(cd)
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
                h = _hist_any(code, start_d.strftime("%Y%m%d"), end_d.strftime("%Y%m%d"), full=True)
            if h is None or len(h) < 120:
                hist_failed.append(code)
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
            risk_hits = news_pre[code] if code in news_pre else _fetch_one_news(code)  # 新闻+巨潮公告（优先用并发预取结果）
            if risk_hits is None:                   # 舆情接口失败：降级为"无雷"并记录，不再整票跳过
                risk_hits = []
                news_failed.append(code)
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
                # 概率校准要用真实完整卖出笔数当样本量（不是交易次数）
                "卖出笔数": int(bs["完整卖出笔数"]),
                **{c: fv.get(c) for c in factors.ALL_FACTOR_COLS},
                "行业": meta[code]["行业"],
                "趋势向上": trend_ok,
                "有雷": len(risk_hits) > 0, "利空标题": "；".join(risk_hits[:2]),
            })
        except Exception:
            continue                                    # 单只失败不影响整体
        progress(0.65 + 0.35 * (i + 1) / total, f"回测验证中 {i + 1}/{total}：{r['名称']}")
    res = pd.DataFrame(rows)
    res.attrs["hist_failed"] = hist_failed      # 行情取数失败的候选（数据源问题，非选股结论）
    res.attrs["news_failed"] = news_failed      # 舆情取数失败、按"无雷"降级的候选
    res.attrs["n_cands"] = total
    # 因子IC检验：行业中性化 → 逐交易日横截面IC → 最近20日均值定权
    try:
        npanel = factors.neutralize(factors.build_panel(hists, meta, fundamentals=fundamentals))
        ics = factors.ic_series(npanel)
        ic, ir = factors.ic_summary(ics)
        ic_all = dict(ic)                               # 全量IC（落盘监控用，不剔除冗余）
        # 相关性去冗余：同一天截面高度相关的因子只留IC更强的那个
        keep = factors.prune_correlated(npanel, ic_all)
        ic = {f: v for f, v in ic_all.items() if f in keep}
        ir = {f: v for f, v in ir.items() if f in keep}
        res.attrs["ic"], res.attrs["icir"] = ic, ir
        res.attrs["ic_all"] = ic_all
        res.attrs["ic_keep"] = sorted(keep)
        res.attrs["ic_n"] = int(len(ics))               # 参与检验的交易日数（置信度参考）
        # 因子衰减监控：把今天的IC存进历史CSV（每天append一次，供后续趋势判断）
        try:
            import factor_learning
            factor_learning.append_factor_ic_history(ic_all, pd.Timestamp.now().strftime("%Y-%m-%d"))
        except Exception:
            pass
    except Exception:
        res.attrs["ic"], res.attrs["icir"], res.attrs["ic_n"] = {}, {}, 0
    # ========== L2 机器学习加分项：LightGBM/XGBoost；样本外IC不达标则只记录、不打分 ==========
    res.attrs["ml"] = None
    if ML_ENABLED and len(res) and "npanel" in locals() and len(npanel):
        try:
            import ml_model
            today = res[["代码", "行业"]].copy()
            for f in factors.ALL_FACTOR_COLS:
                if f in res.columns:
                    today[f + "_n"] = factors.neutral_score(res[f], res["行业"]) - 0.5
            ml = ml_model.train_predict(npanel, today)
            if ml:
                res.attrs["ml"] = {k: v for k, v in ml.items() if k != "scores"}
                if ml["oof_pass"]:
                    res["ML分"] = res["代码"].astype(str).map(ml["scores"])
        except Exception as _e:
            res.attrs["ml"] = {"error": str(_e)[:80]}
    # 基础权重（历史IC；因子截断按 |IC|）
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

    # 计算训练期组合IC（逐因子训练IC按权重加权，衡量当前因子池整体信号强弱）
    per_ic = res.attrs.get("ic") or {}
    # ========== 因子衰减过滤：连续30天|IC|<0.03的因子临时剔除 ==========
    health_df = pd.DataFrame()
    try:
        import factor_learning
        health_df = factor_learning.factor_health(res.attrs.get("ic") or None)
        if len(health_df) > 0:
            final_icw = factor_learning.filter_decayed_factors(final_icw, health_df)
    except Exception:
        pass
    # 定权后按 |权重| 截断到 MAX_FACTORS 个（复盘融合可能重新引入多余因子）
    final_icw = factors.cap_weights(final_icw)
    if not final_icw:
        final_icw = factors.cap_weights(base_icw)
    # 组合IC按最终（已截断）权重计算，门控衡量的是实际使用的因子组合
    ic_train_composite = float(
        sum(float(per_ic.get(f, 0.0) or 0.0) * float(w) for f, w in final_icw.items())
    ) if final_icw else 0.0

    res.attrs["ic_weights"] = final_icw
    res.attrs["ic_weights_base"] = base_icw
    res.attrs["ic_train_composite"] = ic_train_composite          # 训练期组合IC（门控用）
    res.attrs["ic_actual"] = actual_ic_used
    res.attrs["blend_ratios"] = blend_ratios_used
    res.attrs["factor_health"] = health_df
    res.attrs["cb_level"], res.attrs["cb_msg"], res.attrs["cb_diag"] = cb_level, cb_msg, cb_diag
    res.attrs["n_done"] = int(n_done)          # v1.8: 置信度折扣用
    res.attrs["hists"] = hists                  # v1.8: 均值方差优化用
    return res


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
    # L2 机器学习加分：ML分为当日截面百分位(0~1)，缺失按中位数0.5处理；未通过样本外IC则无此列
    if ML_WEIGHT > 0 and "ML分" in res.columns and res["ML分"].notna().any():
        res["总分"] = (1 - ML_WEIGHT) * res["总分"] + ML_WEIGHT * res["ML分"].fillna(0.5) * 100
    res = res.sort_values("总分", ascending=False)
    # 只保留扣费后还能赚钱、且近7天无利空新闻的股票
    res = res[(res["扣费总收益"] > 0) & (~res["有雷"])]
    return res.head(n)


def allocate(picks: pd.DataFrame, budget0: float,
             n_done: int = 0, hists: dict | None = None) -> pd.DataFrame:
    """第四步：把本金按权重分配，并做组合层风控。
    v1.8 仓位分配优先级：均值方差优化 > 凯利公式 > 等额分配
      1. 均值方差优化（PyPortfolioOpt 或 numpy 闭式解）：按 max_sharpe 给权重
      2. 凯利公式：校准后的胜率 + 赔率决定总仓位，按预期收益分给各票
      3. 等额分配：兜底
    三者都带置信度折扣（样本不足时向等额靠拢）。
    v1.17 组合层风控：单票上限 / 行业上限 / 组合年化波动率目标，
      超限只减仓不加仓，权重和不足 1 的部分留现金。
    每只按目标买100股的整数倍；目标金额买不起1手时按最低1手买入（小资金例外）。"""
    cfg_pos = _cfg().get("仓位管理", {})
    hists = hists or {}

    codes = picks["代码"].astype(str).str.zfill(6).tolist()
    n = len(picks)
    weights = [1.0 / n] * n                 # 兜底：等额
    risk_info = {}

    # ===== 1. 均值方差优化（PyPortfolioOpt / numpy 兜底） =====
    opt_weights = None
    if _USE_MV_OPT and n > 1:
        try:
            import portfolio_optimizer as po
            exp_ret = {c: float(picks.loc[picks["代码"].astype(str).str.zfill(6) == c, "扣费总收益"].iloc[0]) / 100
                       for c in codes if len(picks.loc[picks["代码"].astype(str).str.zfill(6) == c]) > 0}
            w, method = po.optimize_weights(codes, hists, exp_ret, n_done=n_done)
            if w:
                # 按 picks 的顺序对齐权重
                opt_weights = [w.get(c, 0.0) for c in codes]
                weights = list(opt_weights)
        except Exception:
            opt_weights = None

    # ===== 2. 凯利公式（均值方差失败时降级） =====
    if opt_weights is None and _USE_KELLY and n > 1:
        try:
            kw = _kelly_position(picks, budget0,
                                 min_wr=float(cfg_pos.get("凯利公式", {}).get("最小胜率", 0.55)),
                                 max_ratio=float(cfg_pos.get("凯利公式", {}).get("最大仓位比例", 0.15)),
                                 n_done=n_done)
            if not (kw == 1.0).all():
                weights = [float(x) for x in kw.values]
        except Exception:
            pass

    # ===== 3. 组合层风控：单票上限 → 行业上限 → 组合波动率目标 =====
    # 只减仓不加仓：权重和可能 < 1，差额留现金（波动率过高时自动降杠杆）
    try:
        rk = cfg_pos.get("组合风控", {})
        if rk.get("启用", True):
            import portfolio_optimizer as po
            ind_map = (dict(zip(codes, picks["行业"].astype(str)))
                       if "行业" in picks.columns else {})
            rets = None
            if hists:
                closes = {}
                for c in codes:
                    h = hists.get(c)
                    if h is None or len(h) < 60:
                        continue
                    hs = h.sort_values("日期")
                    closes[c] = hs.set_index(pd.to_datetime(hs["日期"]))["收盘"].astype(float)
                if len(closes) >= 2:
                    rets = pd.DataFrame(closes).dropna().pct_change().dropna()
            w2, risk_info = po.apply_risk_constraints(
                dict(zip(codes, weights)), ind_map, rets,
                max_name=float(rk.get("单票上限", 0.35)),
                max_ind=float(rk.get("行业上限", 0.50)),
                vol_target=float(rk.get("组合年化波动率目标", 0.30)))
            if w2 and sum(w2.values()) > 0:
                weights = [float(w2.get(c, 0.0)) for c in codes]
    except Exception:
        risk_info = {}

    if sum(weights) <= 0:                   # 约束异常兜底，绝不出现空方案
        weights = [1.0 / n] * n

    plan, budget = [], budget0
    for i, (_, r) in enumerate(picks.iterrows()):
        price = r["最新价"]
        target = budget0 * float(weights[i])
        if target < price * 100:                          # 目标金额买不起1手
            if budget < price * 100:
                continue                                  # 现金也不够，跳过
            target = price * 100                          # 小资金例外：按最低1手买入
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
        # 概率校准：回测胜率按完整卖出笔数向 50% 做贝叶斯收缩
        n_tr = int(r.get("卖出笔数", 0) or 0)
        p_cal = calibrate_win_rate(float(r["胜率"]) / 100, n_tr, prior=0.5, strength=20.0)
        est_win = round(invest * r["平均盈利率"] / 100, 1)
        est_loss = round(invest * r["平均亏损率"] / 100, 1)
        plan.append({
            "代码": r["代码"], "名称": r["名称"],
            "现价": price, "买入股数": shares,
            "佣金(元)": round(fee, 2),
            "投入(含佣金)": invest,
            "盈利概率": round(p_cal * 100, 0),
            "预计盈利(元)": est_win,
            "预计亏损(元)": -est_loss,
            "止损价(买入价×0.95)": round(price * (1 - STOP_LOSS_PCT), 2),
            "止盈参考(买入价×1.10)": round(price * (1 + TAKE_PROFIT_PCT), 2),
            "_win_prob": p_cal,
            "_est_win": est_win, "_est_loss": est_loss,
        })
        budget -= invest
    df = pd.DataFrame(plan)
    if len(df) and risk_info:
        df.attrs["risk_info"] = risk_info
    leftover = round(budget, 2)
    return df, leftover
