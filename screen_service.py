"""选股引擎（纯 Python，不依赖 Streamlit）：初筛 → 回测验证 → 因子打分 → 仓位分配 → 复盘。
网页（app.py）与每日邮件（daily_push.py）共用本模块，保证同一口径。"""
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import akshare as ak
import factors  # L1 多因子层：动量/波动率/换手/估值 + 行业中性 + 20日IC均值配权
import numpy as np  # v1.43 修复：_snap() 用 np.nan，此前漏导入导致全候选 NameError 被吞、res 恒为空
import pandas as pd
import requests

from data import (cninfo_risk_impl, get_history_impl, get_history_tx,
                  get_spot_impl, industry_map_impl, tushare_daily_basic)
from strategy import backtest, calibrate_win_rate, generate_signals

# 利空关键词：近7天新闻标题命中任一即视为"有雷"，直接排除
NEG_KEYWORDS = ("减持", "立案", "退市", "处罚", "问询", "警示", "冻结", "失信",
                "违规", "诉讼", "仲裁", "预亏", "暴雷", "停牌", "质押", "担保逾期")


def _em_news_direct(code: str) -> list | None:
    """v1.41：直连东财新闻搜索 JSONP（就是 ak.stock_news_em 的底层接口），
    自己解析、不依赖 akshare 的列名（避免版本漂移）与 8 秒硬超时。
    失败重试 1 次，单次超时 15 秒。返回近7天利空标题；接口不可用返回 None。"""
    import json as _json
    code = re.sub(r"\D", "", str(code))[-6:]
    cb = "jp" + str(int(time.time() * 1000))[-8:]
    param = {
        "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "time", "pageIndex": 1, "pageSize": 30}},
    }
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    txt = None
    for _ in range(2):                              # 失败重试 1 次
        try:
            r = requests.get(
                url,
                params={"cb": cb, "param": _json.dumps(param, ensure_ascii=False,
                                                       separators=(",", ":"))},
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://so.eastmoney.com/"},
                timeout=15)
            if r.status_code == 200 and r.text:
                txt = r.text
                break
        except Exception:
            continue
    if not txt:
        return None
    try:
        s = txt.strip()
        i, j = s.find("("), s.rfind(")")
        if i < 0 or j <= i:
            return None
        obj = _json.loads(s[i + 1:j])
        arts = (obj.get("result") or {}).get("cmsArticleWebOld") or []
    except Exception:
        return None
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
    hits = []
    for a in arts:
        try:
            t = pd.Timestamp(str(a.get("date", "")))
        except Exception:
            continue
        if t < cutoff:
            break                                  # 按时间倒序，超过7天即止
        title = re.sub(r"<[^>]+>", "", str(a.get("title", "")))   # 去掉 <em> 高亮标签
        if any(k in title for k in NEG_KEYWORDS):
            hits.append(title)
    return hits


def news_risk_impl(code: str) -> list | None:
    """近7天利空新闻标题；空列表=无雷，None=接口失败。
    v1.41：直连东财（重试+15s），不再走 akshare stock_news_em。
    上层对 None 降级放行并标注，不再直接剔票。"""
    try:
        return _em_news_direct(code)
    except Exception:
        return None


def reduce_label_impl(codes, lookback_days: int = 60) -> dict:
    """v1.55：候选股"股东减持"标签（**纯展示，不参与选股/打分/排序/排雷**）。
    用东财 RPT_SHARE_HOLDER_INCREASE 一次请求批量拉近 lookback_days 天减持明细。
    v1.55 用户要求窗口 90→60 天；web 与邮件都调用本函数默认参数，改这一处两端同步生效；
    🔴新减持的 30 天分层在 app.py / daily_push.py 展示层，口径不变。
    按代码聚合累计减持占总股本比例(AFTER_CHANGE_RATE)、笔数、最新公告日。
    返回 {6位代码: {"ratio":累计减持%, "n":笔数, "last_date":"YYYY-MM-DD"}}。
    用正确字段 AFTER_CHANGE_RATE（不是报价涨跌幅 CHANGE_RATE）。
    失败/超时降级返回 {}（标签显示"—"），绝不影响选股主流程。"""
    codes = [str(c).zfill(6) for c in codes if c is not None]
    codes = list(dict.fromkeys(codes))       # 去重保序
    if not codes:
        return {}
    # 分批，避免 URL 过长（每批≤50只）
    out: dict = {}
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
    start = (dt.date.today() - dt.timedelta(days=lookback_days)).isoformat()
    for k in range(0, len(codes), 50):
        batch = codes[k:k + 50]
        code_in = ",".join(f'"{c}"' for c in batch)
        filt = (f'(DIRECTION="减持")(NOTICE_DATE>=\'{start}\')'
                f'(SECURITY_CODE in ({code_in}))')
        params = {
            "sortColumns": "NOTICE_DATE", "sortTypes": "-1",
            "pageSize": 500, "pageNumber": 1,
            "reportName": "RPT_SHARE_HOLDER_INCREASE",
            "columns": "ALL", "source": "WEB", "client": "WEB",
            "filter": filt,
        }
        try:
            r = requests.get(url, params=params, headers=headers, timeout=12)
            j = r.json()
            res = j.get("result") or {}
            data = res.get("data") or []
        except Exception:
            continue
        for x in data:
            code = str(x.get("SECURITY_CODE", "")).zfill(6)
            if code not in codes:
                continue
            ratio = x.get("AFTER_CHANGE_RATE")     # 本次减持占总股本%
            try:
                ratio = abs(float(ratio))
            except (TypeError, ValueError):
                ratio = 0.0
            notice = str(x.get("NOTICE_DATE", ""))[:10]
            d = out.setdefault(code, {"ratio": 0.0, "n": 0, "last_date": ""})
            d["ratio"] += ratio
            d["n"] += 1
            if notice > d["last_date"]:
                d["last_date"] = notice
    for code in out:
        out[code]["ratio"] = round(out[code]["ratio"], 3)
    return out


# ---------------- v1.58 题材冲刺（默认关闭、纯展示、高风险） ----------------
# 东财 push2 clist：与 akshare stock_board_concept_* 同源，云端直连无需大智慧 KEY。
_EM_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
# 技术形态/宽基标签不是"题材"（昨日连板/涨停/融资融券/沪股通等），必须滤掉，
# 否则 Top 榜前几名全是这类标签板，根本不是机器人/算力/半导体这种题材。
_FAKE_THEME_KW = (
    "昨日", "最近", "题材股", "高换手", "连板", "涨停", "首板", "多板", "打板",
    "融资融券", "深股通", "沪股通", "机构", "基金", "QFII", "社保", "预盈",
    "送转", "股权激励", "注册制", "次新", "ST股", "创业板", "科创", "AH股", "B股",
    "标普", "富时", "MSCI", "上证", "中证", "破净", "低价", "高价", "超跌", "转融",
    "长江三角", "雄安", "粤港澳", "自贸",
)


def _em_clist(params: dict, timeout: int = 12):
    """东财 clist 通用请求，2 次重试；失败返回 None（绝不抛，题材是纯展示功能）。"""
    for _ in range(2):
        try:
            r = requests.get(_EM_CLIST_URL, params=params,
                             headers={"User-Agent": "Mozilla/5.0",
                                      "Referer": "https://quote.eastmoney.com/"},
                             timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.json()
        except Exception:
            continue
    return None


def _theme_num(v, default=float("nan")) -> float:
    """东财 '-'/None/非数字 → default（接口缺字段时不能让整表崩掉）。"""
    try:
        f = float(v)
        return f if f == f else default
    except (TypeError, ValueError):
        return default


def theme_sprint_impl(top_n: int = 3, per_theme: int = 6) -> pd.DataFrame:
    """v1.58 题材冲刺：当日 Top 概念题材的成分股（**高风险、纯展示、默认关闭**）。

    与主 screen_impl 刻意相反——放宽涨幅(无7%上限,允许涨停)、PE<0(允许亏损)、
    量比>10(允许异常放量)、近月龙虎榜(允许游资票)，因为要找的就是强势题材票。
    但保留基本安全底线：非ST/退市、价格 [MIN_PRICE, MAX_PRICE]（买得起）、成交额>5000万（卖得出）。

    数据源：东财概念板块排名(fs=m:90+t:3) + 各题材成分股(fs=b:BKxxxx)，
    云端直连即可，不依赖大智慧 KEY。失败降级返回空 DataFrame，绝不抛异常影响主选股。
    """
    try:
        rank = _em_clist({"pn": 1, "pz": 120, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                          "fid": "f3", "fs": "m:90+t:3", "fields": "f12,f14,f3"})
        rows = ((rank or {}).get("data") or {}).get("diff") or []
        themes = []
        for x in rows:
            name = str(x.get("f14") or "")
            code = str(x.get("f12") or "")
            if not code or not name or any(k in name for k in _FAKE_THEME_KW):
                continue
            themes.append((code, name, _theme_num(x.get("f3"), 0.0)))
        themes.sort(key=lambda t: t[2], reverse=True)
        themes = themes[:max(1, int(top_n))]
        if not themes:
            return pd.DataFrame()

        out = []
        for bk, tname, tpct in themes:
            data = _em_clist({"pn": 1, "pz": 60, "po": 1, "np": 1, "fltt": 2, "invt": 2,
                              "fid": "f3", "fs": f"b:{bk}",
                              "fields": "f12,f14,f2,f3,f6,f8,f9,f10,f20,f62"})
            cons = ((data or {}).get("data") or {}).get("diff") or []
            cons = sorted(cons, key=lambda x: _theme_num(x.get("f3"), -999), reverse=True)
            n = 0
            for c in cons:
                if n >= per_theme:
                    break
                code = str(c.get("f12") or "").zfill(6)
                name = str(c.get("f14") or "")
                price = _theme_num(c.get("f2"))
                amount = _theme_num(c.get("f6"), 0.0)
                if not code or "ST" in name.upper() or "退" in name:
                    continue
                if not (MIN_PRICE <= price <= MAX_PRICE):     # 买得起
                    continue
                if amount <= 5e7:                              # 卖得出
                    continue
                out.append({
                    "题材": tname, "题材涨幅%": round(tpct, 2),
                    "代码": code, "名称": name,
                    "现价": round(price, 2), "涨幅%": round(_theme_num(c.get("f3")), 2),
                    "换手%": round(_theme_num(c.get("f8")), 2),
                    "量比": round(_theme_num(c.get("f10")), 2),
                    "市盈率": round(_theme_num(c.get("f9")), 1),
                    "市值(亿)": round(_theme_num(c.get("f20")) / 1e8, 1),
                    "主力净流入(万)": round(_theme_num(c.get("f62")) / 1e4, 0),
                })
                n += 1
        return pd.DataFrame(out)
    except Exception:
        return pd.DataFrame()


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
        lhb = _run_with_timeout(ak.stock_lhb_stock_statistic_em, symbol="近一月", timeout=15)
        return set(lhb["代码"].astype(str).str.zfill(6))
    except Exception:
        return set()


HOLDINGS_FILE = Path(__file__).parent / "持仓.csv"

# ---------------- 推荐设置（本金存仓库文件，网页与每日邮件共用，不写死） ----------------
SETTINGS_FILE = Path(__file__).parent / "推荐设置.json"
# 字段含义：capital = 本金（元）。JSON 不支持注释，含义写在这里
# 兜底本金：仅在 推荐设置.json 缺失/损坏 且未设置环境变量 CAPITAL 时使用（通用值，不绑定个人账户）
DEFAULT_CAPITAL = 10000.0


def load_settings() -> dict:
    """本金优先级：推荐设置.json > 环境变量 CAPITAL > DEFAULT_CAPITAL。
    网页输入框与每日邮件读同一份设置，改一处两边同步。"""
    cap = None
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        cap = float(data["capital"])
    except Exception:
        cap = None
    if not cap or cap <= 0:
        try:
            cap = float(os.environ.get("CAPITAL") or DEFAULT_CAPITAL)
        except (TypeError, ValueError):
            cap = DEFAULT_CAPITAL
    return {"capital": cap}


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


# ---------------- T+1 交易成本计算器（v1.49，纯函数无副作用） ----------------
# 口径与 strategy.py backtest / allocate / best_plan 完全一致，避免两处算法漂移
_COST_COMM_RATE = 0.00025      # 佣金 万2.5，双向
_COST_MIN_COMM = 5.0           # 单笔最低佣金（元）——小本金的固定税
_COST_STAMP_RATE = 0.0005      # 印花税 0.05%，仅卖出
_COST_TRANSFER_RATE = 0.00001  # 过户费 0.001%，双向
_COST_SLIPPAGE = 0.001         # 滑点 单边0.1%（回测口径：买上浮/卖下浮）
_COST_MIN_CRIT = _COST_MIN_COMM / _COST_COMM_RATE   # 20000元：单笔低于此佣金按5元封底


def cost_calculator(capital: float, price: float, n_stocks: int = 0) -> dict:
    """T+1 一买一卖成本结构：给定本金/假设股价/持股只数，算费用与回本门槛。
    纯计算、无网络、无 Streamlit 依赖，可离线单测。
      逐笔计算最低佣金（每只票单独 max(5, amt×0.00025)），与 allocate/best_plan 同口径。
      n_stocks<=0 时给"成本视角建议只数上限"=每只尽量≥2万避开最低佣金封底（1~20）。
    返回 dict：crit/suggested_n/rows（多只数对比）/detail（选中只数明细）。"""
    capital = max(float(capital or 0), 0.0)
    price = max(float(price or 0), 0.01)
    crit = _COST_MIN_CRIT

    def _row(n: int) -> dict:
        n = max(int(n), 1)
        per_budget = capital / n
        shares = int(per_budget / price // 100) * 100
        if shares <= 0:
            shares = 100                       # 小资金例外：至少1手（与 allocate 一致）
        # 逐手递减直到"成交额+佣金"买得起（与 allocate 完全同口径）
        while shares > 0:
            _amt_try = shares * price
            _buy_try = max(_amt_try * _COST_COMM_RATE, _COST_MIN_COMM) + _amt_try * _COST_TRANSFER_RATE
            if _amt_try + _buy_try <= per_budget + 1e-9:
                break
            shares -= 100
        if shares <= 0:
            shares = 100                       # 极端兜底：连1手都买不起仍按1手展示
        one_amt = shares * price               # 单只成交金额（不含费）
        # 逐笔佣金：每只票单独触底判断（关键，汇总算会低估最低佣金）
        buy_fee = sum(max(one_amt * _COST_COMM_RATE, _COST_MIN_COMM) + one_amt * _COST_TRANSFER_RATE
                      for _ in range(n))
        sell_fee = sum(max(one_amt * _COST_COMM_RATE, _COST_MIN_COMM)
                       + one_amt * (_COST_STAMP_RATE + _COST_TRANSFER_RATE)
                       for _ in range(n))
        total_amt = one_amt * n
        total_fee = buy_fee + sell_fee
        slip = total_amt * _COST_SLIPPAGE * 2   # 双边滑点
        fee_pct = total_fee / total_amt * 100 if total_amt > 0 else 0.0
        be_pct = (total_fee + slip) / total_amt * 100 if total_amt > 0 else 0.0
        invest = total_amt + buy_fee
        return {
            "只数": n, "每只股数": int(shares), "每只金额": round(one_amt, 2),
            "总买入": round(total_amt, 2), "买入佣金": round(buy_fee, 2),
            "卖出费用": round(sell_fee, 2), "往返合计": round(total_fee, 2),
            "费用占比%": round(fee_pct, 3), "回本涨幅%": round(be_pct, 3),
            "投入(含佣)": round(invest, 2), "剩余现金": round(capital - invest, 2),
            "最低佣金封底": bool(one_amt < crit),
        }

    # 成本视角建议：本金够几只≥2万就最多几只（避开最低佣金惩罚），封顶20、至少1
    suggested = max(1, min(20, int(capital // crit)))
    cand_n = [1, 2, 3, 5, 8, 10, 20]
    rows = [_row(n) for n in cand_n if n <= 20]
    chosen = suggested if (n_stocks or 0) <= 0 else max(1, min(20, int(n_stocks)))
    return {
        "临界本金": crit, "建议只数": suggested, "选中只数": chosen,
        "本金": capital, "假设股价": price,
        "rows": rows, "detail": _row(chosen),
    }


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


def dedup_by_industry(ranked: pd.DataFrame) -> pd.DataFrame:
    """v1.60 行业硬分散：每个行业只保留排名最高的1只（按传入顺序 keep first）。
    只用于 best_plan 构造买入方案，不改页面候选池/打分/排序。异常时原样返回。"""
    try:
        if "行业" not in ranked.columns or len(ranked) == 0:
            return ranked
        _attrs = dict(ranked.attrs)
        d = ranked.copy()
        d["_行业键"] = d["行业"].astype(str)
        out = d.drop_duplicates(subset=["_行业键"], keep="first").drop(columns=["_行业键"])
        for _k, _v in _attrs.items():
            out.attrs[_k] = _v
        return out
    except Exception:
        return ranked


def best_plan(ranked: pd.DataFrame, capital: float):
    """只数不写死：枚举 1~min(候选数,20)，选'预期总收益−买卖全部摩擦成本'最大的方案。
    预期收益 = 各票扣费回测收益% × 实际买入金额；摩擦含最低5元佣金
    （本金越小固定佣金越吃亏，最优只数自然越少；候选越多收益越高，只数自然越多）。
    v1.8: 从 ranked.attrs 取 n_done / hists，传给 allocate 做均值方差优化。
    v1.24: 加空 plan 守卫，allocate 全部候选买不起 1 手时返回空 df → 跳过。
    返回 (买入方案, 最优只数)。"""
    # v1.60 行业硬分散（取长补短·对标 AlphaSift 行业桶上限）：
    # 买入方案里每个行业只保留排名最高的1只。全样本验证同行业日收益相关0.483 vs
    # 跨行业0.215，硬分散使组合日波动降6.5%、尾部P5收窄、收益不变(+1.4bp p=0.83)。
    # 只改进入方案的候选列表，不改 ranked 本身（页面候选池/打分/排序照旧），
    # 且 web 与邮件都走 best_plan，一处改动两端同步。配置 false 可回退。
    try:
        if _cfg().get("仓位管理", {}).get("组合风控", {}).get("每行业最多1只", True):
            ranked = dedup_by_industry(ranked)
    except Exception:
        pass   # 去重任何异常都降级回原行为，绝不因它选不出票
    ret_map = ranked.set_index("代码")["扣费总收益"]
    n_done = int(ranked.attrs.get("n_done", 0) or 0)
    hists = ranked.attrs.get("hists") or {}
    best_df, best_n, best_net = None, 0, -1e18
    for n in range(1, min(len(ranked), 20) + 1):
        plan, _left = allocate(ranked.head(n), capital, n_done=n_done, hists=hists)
        if plan is None or len(plan) == 0 or "买入股数" not in plan.columns:
            continue   # allocate 全部买不起 1 手 → 跳过这个只数
        amt = plan["买入股数"] * plan["现价"]
        gross = float((amt * plan["代码"].map(ret_map) / 100).sum())
        buy_fee = float((plan["投入(含佣金)"] - amt).sum())
        sell_fee = float((5.0 + amt * 0.00051).sum())   # 卖出：最低佣金5元+印花税0.05%+过户费
        net = gross - buy_fee - sell_fee
        if net > best_net:
            best_df, best_n, best_net = plan, n, net
    # 兜底：全部 n 都买不起时，返回空 df + n=0
    if best_df is None:
        return pd.DataFrame(), 0
    return best_df, best_n


# ---------------- 固定参数（本金由页面输入，其余固化在后台） ----------------
MIN_PRICE = 10.0         # 股价下限（初值，后面从 config 覆盖）
MAX_PRICE = 30.0         # 股价上限（初值，后面从 config 覆盖）
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

# 价格区间：改 strategy_config.json 的「价格区间」即可换价格带
_price_cfg = _cfg().get("价格区间", {})
MIN_PRICE = float(_price_cfg.get("下限", MIN_PRICE))
MAX_PRICE = float(_price_cfg.get("上限", MAX_PRICE))

_risk = _cfg().get("风控", {})
_ml_cfg = _cfg().get("机器学习", {})
ML_ENABLED = bool(_ml_cfg.get("启用", True))            # LightGBM加分项总开关
ML_WEIGHT = max(0.0, min(0.5, float(_ml_cfg.get("权重", 0.20))))   # 占总分比例，上限50%
ML_ENSEMBLE_ENABLED = bool(_ml_cfg.get("集成精筛", True))  # Stage3 三模型集成精筛总开关
# v1.38 提速：财报(bastock)串行拉取是最大耗时点(15只×最多20秒=上限5分钟)，
# 而实测基本面因子在训练面板100% NaN、ML只用5个价格因子，拉回来几乎不起作用。
# 默认关闭，fundamentals={} → 基本面因子全NaN自动跳过。要用改 config「观测→拉取财报」=true。
FETCH_FUNDAMENTAL = bool(_cfg().get("观测", {}).get("拉取财报", False))
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
REVIEW_CACHE_KEY = "v1.48"   # 版本升级强制刷新，避免旧缓存缺少股数/盈亏额/当时盈利概率等新字段（xx=v1.47加当时盈利概率列后必须失效旧缓存，否则 g[_cols] KeyError）


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
        # ===== 盈亏额按真实到手净额计算（v1.26 修正）=====
        # 买入成本 = 成交金额 + 佣金(max5) + 过户费；卖出到手 = 成交金额 - 佣金(max5) - 印花税 - 过户费
        # 与页面宣称的"已内置手续费"及选股计划的 allocate 口径一致。
        # 之前用 (判定价-推荐价)*股数 的毛额，低估了每笔约10元固定成本，
        # 会把大量小额盈利误显示为盈利（如+10元实际扣费后为-0.5元），连带胜率虚高。
        buy_amt = base_px * shares
        sell_amt = judge_px * shares
        buy_fee = max(5.0, buy_amt * 0.00025) + buy_amt * 0.00001   # 佣金(最低5元)+过户费
        invest0 = buy_amt + buy_fee
        is_holding = label.startswith("⏳")   # 未满20日、未触发：还没卖出，不计卖出费
        if n == 0:
            # 当日新推：无后续行情，盈亏按0显示（买入尚未产生卖出）
            sell_fee = 0.0
            total_fee = 0.0
            net_pnl = 0.0
        elif is_holding:
            # 浮动盈亏：仅按当前市值算浮盈，不扣卖出端费用（未发生卖出）；买入佣金已计入成本
            sell_fee = 0.0
            total_fee = round(buy_fee, 1)
            net_pnl = round(sell_amt - invest0, 1)    # 浮动：现市值 - 买入总成本
        else:
            sell_fee = max(5.0, sell_amt * 0.00025) + sell_amt * 0.0005 + sell_amt * 0.00001  # 佣金+印花税+过户费
            total_fee = round(buy_fee + sell_fee, 1)
            net_pnl = round(sell_amt - sell_fee - invest0, 1)   # 实际到手净盈亏额
        # 净收益率 = 净额 / 初始投入（含买入费），与"盈亏额"自洽，避免金额与百分比对不上
        net_pct = round(net_pnl / invest0 * 100, 2) if invest0 > 0 else 0.0
        row = {"日期": rec_date.strftime("%Y-%m-%d"),
               "推送时间": ("" if pd.isna(r.get("推送时间")) else str(r.get("推送时间")).strip()),
               "代码": code,
               "名称": str(r.get("名称", "")).strip(),
               "推荐价": round(price0, 2), "现价": round(judge_px, 2),
               "股数": shares,
               "费用(元)": total_fee,
               "涨跌%": round((judge_px / base_px - 1) * 100, 2),   # 纯价格涨跌幅（对止盈止损线）
               "盈亏%": net_pct,                                     # 净收益率（对实际到手金额）
               "结果": label,
               "同期沪深300%": bench,
               # v1.60 超额% = 净收益 − 同期沪深300，纯展示：把"赚亏"正确归因成"选股 vs 大盘"。
               # 基准缺失留空（不编造）；持有中/当日新推也留空（未了结不比较）。
               "超额%": (round(net_pct - bench, 2)
                       if bench is not None and not label.startswith("⏳") else None),
               "盈亏额": net_pnl,
               # v1.46 当时的盈利概率：推荐时留档的校准值；旧批次无此列显示"—"，无法回溯，不编造
               "当时盈利概率": ("—" if pd.isna(r.get("盈利概率")) or r.get("盈利概率") is None
                             else f"{float(r.get('盈利概率')):.0f}%"),
               "_盈亏额": net_pnl,
               "_done": not label.startswith("⏳"), "_saved": 0.0}
        # 止损挽回金额：若止损后继续跌到满期，估算纪律避免了多少额外亏损（净额口径）
        if label == "🛑止损保命" and len(future) >= REVIEW_DAYS:
            day_end = float(future.iloc[REVIEW_DAYS - 1])
            end_amt = day_end * shares
            end_fee = max(5.0, end_amt * 0.00025) + end_amt * 0.0005 + end_amt * 0.00001
            end_net = end_amt - end_fee
            stop_net = sell_amt - sell_fee        # 止损当笔实际到手（上方已算）
            row["_saved"] = round(max(0.0, stop_net - end_net), 1)
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
    # 胜率按"实际到手净额>0"判定（v1.26）：标签盈利但扣费后亏损的笔不算赢，口径与盈亏金额一致
    wins = int((done["_盈亏额"] > 0).sum())
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


def _csi_set_impl() -> set:
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


_BAOSTOCK_TIMED_OUT = False


def _csi_set() -> set:
    """Baostock 无响应时按空成分股集合降级，并阻止后续财报请求再次卡住。"""
    global _BAOSTOCK_TIMED_OUT
    if _BAOSTOCK_TIMED_OUT:
        return set()
    result = _run_with_timeout(_csi_set_impl, timeout=25)
    if result is None:
        _BAOSTOCK_TIMED_OUT = True
        print("CSI 成分股获取超时，本次按空集合降级")
        return set()
    return result


def screen_impl(max_price: float | None = None) -> pd.DataFrame:
    """第一步：从全市场快照里做硬性初筛，只留下安全、买得起、流动性够的股票。
    （无缓存装饰器，供裸运行脚本直接调用）"""
    if max_price is None:
        max_price = MAX_PRICE
    df = get_spot_impl()                                 # 内部调用走 _impl，避免嵌套缓存
    df = df.copy()
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    lhb = lhb_codes_impl()                               # 龙虎榜名单（近一月）
    name = df["名称"].astype(str)
    csi = _csi_set()
    board_ok = df["代码"].map(_board_ok) | df["代码"].isin(csi)    # CSI成分股放行创业板/科创板
    # v1.45：价格带对所有股票严格生效，CSI成分股不再豁免价格上限。
    # 旧逻辑 `| df["代码"].isin(csi)` 让沪深300/中证500成分股无视价格上限，
    # 导致 100~500 元的高价股也进候选池（本地快照实测 230 只>50元、65只>100元），与"8~50元"设定矛盾。
    price_ok = df["最新价"].between(MIN_PRICE, max_price)
    mask = (
        board_ok
        & ~df["代码"].isin(lhb)                           # 剔除近一月上过龙虎榜的游资爆炒股
        & ~name.str.contains("ST|退", na=False)          # 剔除ST/退市股
        & price_ok                                       # 价格严格限制在[下限,上限]，无指数成分豁免
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
    返回去重后的利空标题字符串列表；两个源都失败才返回 None（上层按"无雷"降级并标注）。
    注意：新闻源给的是标题字符串，公告源给的是 {"标题","日期","来源"} 字典，两者都要兼容。"""
    # v1.42 提速：两源并行（原来串行等东财15s再等巨潮12s，现在取两者最大值）。
    # 外层 verify_impl 已有 4 路候选并发，这里每只再开 2 路，总并发约 8，可接受。
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _em():
        try:
            return news_risk_impl(code)
        except Exception:
            return None

    def _cn():
        try:
            return cninfo_risk_impl(code, days=7)
        except Exception:
            return None

    with _TPE(max_workers=2) as _ex:
        _f_em = _ex.submit(_em)
        _f_cn = _ex.submit(_cn)
        n = _f_em.result()
        a = _f_cn.result()
    if n is None and a is None:
        return None
    out = []
    for hit in list(n or []) + list(a or []):
        t = str(hit.get("标题", "")) if isinstance(hit, dict) else str(hit)
        if t and t not in out:
            out.append(t)
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
    global _BAOSTOCK_TIMED_OUT
    if capital is None:
        capital = float(load_settings()["capital"])
    end_d = dt.date.today()
    start_d = end_d - dt.timedelta(days=HISTORY_DAYS)
    rows, hists, meta = [], {}, {}          # hists/meta：供因子面板与行业中性化使用
    hist_failed, news_failed = [], []       # 取数失败名单：区分"数据源不可用"与"候选不合格"
    verify_errors = []                      # v1.43: 每只候选处理异常的记录（诊断用，防静默吞错）
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
    # ===== 逐票拉候选股基本面（Baostock 历史财报）=====
    # v1.38 提速：默认跳过（FETCH_FUNDAMENTAL=False）。财报串行拉取是最大耗时点，
    # 且实测基本面因子对打分几乎无贡献（训练面板全NaN、ML只用价格因子）。
    # 需要时把 config「观测→拉取财报」设为 true 即可恢复原行为。
    if FETCH_FUNDAMENTAL:
        try:
            if _BAOSTOCK_TIMED_OUT:
                raise TimeoutError("Baostock 此前已无响应")
            import fundamental as fd
            fundamental_deadline = time.monotonic() + 300
            for fi, cd in enumerate(cands["代码"].astype(str).str.zfill(6).tolist(), 1):
                if time.monotonic() >= fundamental_deadline:
                    fundamentals = {}
                    progress(0.3, "财报拉取超过5分钟，本次统一按基本面缺失降级")
                    break
                progress(0.3 * fi / total, f"拉取财报 {fi}/{total}（baostock 串行，今日首次较慢）")

                def _fetch_fundamental():
                    try:
                        return fd.fetch_fundamental(cd)
                    except Exception:
                        return pd.DataFrame()

                fundamental = _run_with_timeout(_fetch_fundamental, timeout=20)
                if fundamental is None:
                    _BAOSTOCK_TIMED_OUT = True
                    fundamentals = {}
                    progress(0.3, "财报接口超时，本次统一按基本面缺失降级")
                    break
                fundamentals[cd] = fundamental
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

        # v1.41：8→4 路并发——8 路同时打东财极易触发限流，反而让舆情整批失败
        with ThreadPoolExecutor(max_workers=4) as pool:
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
            news_unverified = False
            if risk_hits is None:
                # v1.41 降级放行：双源都失败=风险未核实，不再直接剔票（避免接口抖动把好票全筛掉）。
                # 保留该票参与后续回测/打分，仅记录名单并在结果里标注，由用户自行二次确认。
                news_unverified = True
                news_failed.append(code)
                risk_hits = []                      # 当作"无已知利空"继续，但标记未核实
            last = sig.iloc[-1]
            # 技术面硬条件：当前价在20日线上，且5日线在20日线上（趋势仍向上）
            # _fast/_slow 是信号表里的 MA5/MA20 列
            trend_ok = bool(last["收盘"] > last["_slow"] and last["_fast"] > last["_slow"])
            # v1.29 当日快照字段：T+1 超短线精筛最需要的开盘/隔夜/换手/量能/市值/估值。
            # 这些在 spot 快照里有，但旧版组装结果时丢弃了，外部精筛模型拿不到；用 r.get 防空。
            def _snap(key):
                try:
                    v = r.get(key, np.nan)
                    return float(v) if v is not None and v == v else np.nan
                except Exception:
                    return np.nan
            rows.append({
                "代码": code, "名称": r["名称"], "最新价": float(r["最新价"]),
                "今日涨跌": float(r["涨跌幅"]), "量比": float(r["量比"]),
                # v1.29 快照字段（供导出给外部 T+1 精筛）
                "今开": _snap("今开"), "昨收": _snap("昨收"),
                "最高": _snap("最高"), "最低": _snap("最低"),
                "振幅": _snap("振幅"), "换手率": _snap("换手率"),
                "成交额": _snap("成交额"), "成交量": _snap("成交量"),
                "市盈率": _snap("市盈率-动态"), "市净率": _snap("市净率"),
                "总市值": _snap("总市值"), "流通市值": _snap("流通市值"),
                "隔夜涨跌%": (round((_snap("今开") / _snap("昨收") - 1) * 100, 2)
                            if _snap("昨收") > 0 else np.nan),
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
                "舆情未核实": news_unverified,     # v1.41：True=接口失败降级放行，风险未经核实
            })
        except Exception as _e:
            # v1.43：不再静默吞异常。单只失败仍跳过，但要记录原因——
            # 此前 _snap 的 np NameError 被这里吞掉，30只全灭却只显示"回测不合格"，极难定位。
            verify_errors.append({"代码": code, "错误": f"{type(_e).__name__}: {str(_e)[:120]}"})
            continue
        progress(0.65 + 0.35 * (i + 1) / total, f"回测验证中 {i + 1}/{total}：{r['名称']}")
    res = pd.DataFrame(rows)
    res.attrs["verify_errors"] = verify_errors    # v1.43: 被跳过的候选及原因（诊断用）
    res.attrs["hist_failed"] = hist_failed      # 行情取数失败的候选（数据源问题，非选股结论）
    res.attrs["news_failed"] = news_failed      # v1.41：舆情双源失败、降级放行（未核实）的候选，仅告警不剔除
    res.attrs["n_cands"] = total
    # 因子IC检验：行业中性化 → 逐交易日横截面IC → 最近20日均值定权
    # 因子IC检验：优先用 wf_cache 全市场大面板（数百只×近千日，IC/ML更稳定），
    # 无缓存时退回候选股面板（云端/首次运行）
    try:
        npanel = factors.build_training_panel()
        if npanel is None or len(npanel) == 0:
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
    # Stage3 精筛复用同一份训练面板，避免重复构建（pick_top3 里通过 attrs 读取）
    res.attrs["npanel"] = npanel if "npanel" in locals() else None
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

    # ========== L3 专业统计门禁（v1.24 新增） ==========
    try:
        import statistics_kit as stkit
        # (a) 回测胜率显著性检验（全样本）
        if len(res) > 0 and "卖出笔数" in res.columns and "胜率" in res.columns:
            total_trades = int(res["卖出笔数"].sum())
            total_wins = int((res["胜率"] / 100 * res["卖出笔数"]).sum())
            win_sig = stkit.win_rate_significance(total_wins, total_trades)
            res.attrs["stats_win_sig"] = win_sig
        # (b) 因子稳定性检验（每个因子的 IC 序列 → ICIR + 衰减率）
        if "ic" in res.attrs and res.attrs["ic"]:
            ics = factors.ic_series(npanel) if "npanel" in locals() else None
            if ics is not None:
                stab_per_factor = {}
                for col in res.attrs["ic"].keys():
                    if col in ics.columns:
                        stab_per_factor[col] = stkit.factor_stability(ics[col])
                res.attrs["stats_factor_stab"] = stab_per_factor
                # 稳定性惩罚：权重 *= min(1.5, icir / max(decay, 0.5))
                # ICIR < 0.5 或衰减率 < 0.5 的因子，权重砍半
                if final_icw and stab_per_factor:
                    stab_penalized = {}
                    for f, w in final_icw.items():
                        s = stab_per_factor.get(f, {})
                        icir_s = s.get("icir", 0.5)
                        decay_s = s.get("decay_ratio", 1.0)
                        stab_factor = min(1.5, max(0.3,
                                                     icir_s / max(0.5, icir_s)
                                                     * min(1.0, max(decay_s, 0.3))))
                        stab_penalized[f] = w * stab_factor
                    # 重新归一化
                    total_w = sum(abs(v) for v in stab_penalized.values()) or 1.0
                    stab_penalized = {f: v / total_w for f, v in stab_penalized.items()}
                    res.attrs["ic_weights"] = stab_penalized
                    final_icw = stab_penalized
        # (c) 历史 VaR + 分布诊断（取第一只票的 hists 做 proxy，或全组均值）
        var_info = {}
        if hists:
            all_returns = []
            for cd, hh in list(hists.items())[:30]:
                try:
                    ret = hh["收盘"].pct_change().dropna()
                    all_returns.append(ret)
                except Exception:
                    continue
            if all_returns:
                combined = pd.concat(all_returns)
                var_info = stkit.historical_var(combined)
                dist_info = stkit.return_distribution(combined)
                res.attrs["stats_var"] = var_info
                res.attrs["stats_dist"] = dist_info
    except Exception:
        pass   # 统计失败不影响主流程

    return res


def ensemble_refine(res: pd.DataFrame) -> pd.DataFrame:
    """Stage3 三模型集成精筛：LightGBM(复用L2分数，不重复训练) + ExtraTrees + MLP
    百分位平均，在可交易候选中保留集成前 50%。
    候选<6只 / 无训练面板 / 模型全失败 / 过滤后为空 → 自动跳过或回退，绝不阻塞主流程。
    诊断存 res.attrs["stage3"]（applied/models_used/time_s/reason/rejected）。"""
    diag: dict = {"applied": False, "reason": ""}
    if not ML_ENSEMBLE_ENABLED:
        diag["reason"] = "配置关闭"
        res.attrs["stage3"] = diag
        return res
    if len(res) < 6:
        diag["reason"] = f"可交易候选仅 {len(res)} 只（<6），百分位无意义，跳过精筛"
        res.attrs["stage3"] = diag
        return res
    npanel = res.attrs.get("npanel")
    if npanel is None or len(npanel) == 0:
        diag["reason"] = "无训练面板，跳过精筛"
        res.attrs["stage3"] = diag
        return res
    try:
        from cascade_runner import DeepRefiner, StageResult
        # L2 已给出 LightGBM 分数则直接复用；未通过样本外 IC 时 lgb 分量弃权（空 dict），
        # 两种情况下都不在 Stage3 重复训练 LightGBM
        if "ML分" in res.columns and res["ML分"].notna().any():
            lgb_scores = {str(c).zfill(6): float(s) for c, s in
                          zip(res["代码"], res["ML分"]) if pd.notna(s)}
        else:
            lgb_scores = {}
        stage2 = StageResult("Stage2-主流程", len(res), res.copy(), [], 0.0)
        out = DeepRefiner().refine(stage2, npanel=npanel, lgb_scores=lgb_scores)
        diag.update(out.diagnostics)
        diag["time_s"] = round(out.stage_time_s, 1)
        if not (diag.get("models_used") or []):
            # 三模型都没参与：把所有模型的具体原因拼出来，不再用"缺依赖"一句话掩盖
            errs = diag.get("model_errors") or {}
            detail = "；".join(f"{k}: {v}" for k, v in errs.items()) or "无诊断信息"
            diag["reason"] = f"三模型均未参与（{detail}），跳过精筛"
            res.attrs["stage3"] = diag
            return res
        if len(out.passed) == 0:
            diag["reason"] = "精筛过滤后为空，回退保留全部候选"
            res.attrs["stage3"] = diag
            return res
        keep = out.passed
        res = res.copy()
        res["集成ML分"] = res["代码"].astype(str).str.zfill(6).map(
            diag.get("code_scores") or {})
        res["集成ML百分位"] = res["代码"].astype(str).str.zfill(6).map(
            dict(zip(keep["代码"].astype(str).str.zfill(6), keep["ML百分位"])))
        keep_codes = set(keep["代码"].astype(str).str.zfill(6))
        before = len(res)
        res = res[res["代码"].astype(str).str.zfill(6).isin(keep_codes)].copy()
        diag["applied"] = True
        diag["reason"] = f"{before} → {len(res)} 只"
        diag["rejected"] = [r.detail for _, r in out.rejected]
        res.attrs["stage3"] = diag
        return res
    except Exception as e:
        diag["reason"] = f"精筛异常回退：{str(e)[:80]}"
        res.attrs["stage3"] = diag
        return res


def pick_top3(res: pd.DataFrame, n: int = PICKS) -> pd.DataFrame:
    """第三步：综合打分。
    扣费收益占大头，胜率其次，趋势向上是必须条件（不满足扣大分）。
    v1.24：引入统计折扣（胜率不显著时对总分打折，避免把随机噪声当信号）。"""
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
    # L3 统计折扣（v1.24）：胜率不显著时对总分打折 0.7~0.9
    win_sig = res.attrs.get("stats_win_sig") or {}
    p_val = win_sig.get("p_value", 1.0)
    if p_val < 0.05:
        discount = 1.0
    elif p_val < 0.15:
        discount = 0.9
    else:
        discount = 0.7
    res["总分"] *= discount
    res.attrs["stats_win_discount"] = discount
    res = res.sort_values("总分", ascending=False)
    # 只保留扣费后还能赚钱、且近7天无利空新闻的股票
    res = res[(res["扣费总收益"] > 0) & (~res["有雷"])]
    # Stage3 三模型集成精筛（v1.24）：在可交易集合上做最后一道 DL 筛选，保留集成前50%
    res = ensemble_refine(res)
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


# ==================== v1.59 复盘区外盘背景（纯展示·事后解释，不参与打分/邮件） ====================
# 定位：外盘是"解释工具"不是"预测工具"。已用126个交易日验证：隔夜纳指→沪深300当日
# 相关系数仅0.199、样本外衰减到0.131、分组单调性断裂、加筛子只改善0.017%/天(噪声级)，
# 故绝不进打分/排序/买入方案/邮件，只在复盘区显示当日外盘环境，帮用户理解"这批为何亏"。

# 东财全球指数 secid：纳斯达克100=100.NDX、标普500=100.SPX、日经225=100.N225、韩国KOSPI=100.KS11
_EM_GLOBAL = {
    "纳指": "100.NDX",
    "标普": "100.SPX",
    "日经": "100.N225",
    "韩股": "100.KS11",
}


def _em_index_kline(secid: str, limit: int = 200) -> dict:
    """拉单个东财全球指数日线，返回 {date: close}。失败返回空 dict（静默降级）。"""
    try:
        r = requests.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={"secid": secid, "klt": 101, "fqt": 1, "lmt": limit,
                    "fields1": "f1,f2,f3", "fields2": "f51,f53", "end": "20500101"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        j = r.json()
        rows = (j.get("data") or {}).get("klines") or []
        out = {}
        for line in rows:
            d, c = line.split(",")[:2]
            try:
                out[d] = float(c)
            except ValueError:
                continue
        return out
    except Exception:
        return {}


def external_context_impl(review_dates: list) -> dict:
    """给复盘区的每个'推荐日'算隔夜/当日外盘背景（纯展示）。

    review_dates: 复盘表里的推荐日期字符串列表（如 ['2026-11-18', ...]）。
    返回: {推荐日: {"纳指": +x.x, "标普": +x.x, "日经": +x.x, "韩股": +x.x,
                   "外盘日": "YYYY-MM-DD"}}；涨跌幅单位%，失败项缺省。
    口径：对每个推荐日 D，取美股/外盘在 D 之前最近一个交易日的收益（=A股D日开盘前
    能看到的外盘表现，即"隔夜外盘"）。东财接口偶发失败则该项留空，不影响其它项。
    """
    result = {}
    try:
        dates = sorted({str(d).strip()[:10] for d in review_dates if d})
    except Exception:
        return {}
    if not dates:
        return {}
    # 一次性拉四个指数的日线，之后本地匹配，避免逐日联网
    series = {name: _em_index_kline(secid) for name, secid in _EM_GLOBAL.items()}
    if not any(series.values()):
        return {}
    for d in dates:
        item = {}
        for name, smap in series.items():
            if not smap:
                continue
            prev = [k for k in smap if k < d]
            if len(prev) < 1:
                continue
            pd_ = prev[-1]
            # 收益 = 前一交易日收盘 / 更早一交易日收盘 - 1
            idx = sorted(smap).index(pd_)
            earlier = sorted(smap)[idx - 1] if idx >= 1 else None
            if earlier is None:
                continue
            try:
                item[name] = round((smap[pd_] / smap[earlier] - 1) * 100, 2)
                item["外盘日"] = pd_
            except (ValueError, ZeroDivisionError):
                continue
        if item:
            result[d] = item
    return result


# ==================== v1.61 标题区累计净值曲线（纯展示·首屏自动加载） ====================
# 用户选"首屏自动加载+诚实展示"：把实盘逐笔盈亏累加成一条资金曲线放标题旁。
# 数据来自 review_impl 的复盘结果（已联网算好的逐笔扣费净额），本函数只做本地聚合，
# 不额外联网。app 端用 _run_with_timeout 包裹 + 失败降级，绝不让曲线拖垮首屏白屏。

def equity_curve_impl(rv) -> dict:
    """把复盘结果累加成累计净值曲线（元），供标题区展示。纯本地计算，无网络。

    参数 rv: review_impl/review_performance 返回的 DataFrame，需含列
             _batch(批次键), 日期, _done, _盈亏额(扣费后净额)。
    返回 dict:
      ok        bool  是否有可画的已结算数据
      curve     list  [(日期标签, 累计盈亏元), ...] 按日期正序（含起点0）
      total     最终累计盈亏(元)
      n         已结算笔数
      win_rate  胜率%(按每笔净额>0)
      avg       平均每笔净盈亏(元)
      note      一句话说明
    口径：按推荐批次(日期+推送时间)推进，每批内已结算笔净额求和，逐批累加；
          ⏳持有中(_done=False)不计入。累计为0轴起步的"逐笔到手净额"曲线。
    任何异常返回 ok=False 的空结构，调用方安静降级。
    """
    empty = {"ok": False, "curve": [], "total": 0.0, "n": 0,
             "win_rate": 0.0, "avg": 0.0, "note": "暂无已结算数据"}
    try:
        import pandas as _pd
        if rv is None or not hasattr(rv, "columns") or len(rv) == 0:
            return empty
        need = {"_batch", "日期", "_done", "_盈亏额"}
        if not need.issubset(rv.columns):
            return empty
        d = rv[rv["_done"]].copy()
        if len(d) == 0:
            return empty
        d["_盈亏额"] = _pd.to_numeric(d["_盈亏额"], errors="coerce").fillna(0.0)
        # 按批次聚合：同一次推送的多笔合成一个点（日期用批次里的日期）
        g = (d.groupby("_batch", sort=False)
               .agg(日期=("日期", "first"), 批盈亏=("_盈亏额", "sum"), 笔数=("_盈亏额", "size"))
               .reset_index())
        g["_dt"] = _pd.to_datetime(g["日期"], errors="coerce")
        g = g.dropna(subset=["_dt"]).sort_values("_dt", kind="stable")
        if len(g) == 0:
            return empty
        cum = g["批盈亏"].cumsum().round(2).tolist()
        # 同日多批加 #序号 后缀，避免重复横轴标签（st.line_chart 按标签画点）
        _seen = {}
        labels = []
        for x in g["日期"].tolist():
            _d = str(x)[:10]
            _seen[_d] = _seen.get(_d, 0) + 1
            labels.append(_d if _seen[_d] == 1 else f"{_d}#{_seen[_d]}")
        curve = [("起点", 0.0)] + list(zip(labels, cum))
        n = int(len(d))
        total = round(float(cum[-1]), 2)
        wins = int((d["_盈亏额"] > 0).sum())
        return {"ok": True, "curve": curve, "total": total, "n": n,
                "win_rate": round(100.0 * wins / n, 1) if n else 0.0,
                "avg": round(float(d["_盈亏额"].mean()), 2) if n else 0.0,
                "note": f"{n} 笔已结算逐笔累计（扣费后）"}
    except Exception:
        return empty
