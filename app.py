"""每日一键选股 · 极简版
只要点一个按钮，后台自动完成：
全市场初筛（价格/市值/成交/剔除风险股）→ 逐只历史回测（含手续费+降费过滤器）→
综合打分（扣费收益+胜率+趋势）→ 按本金自动决定只数（本金÷1000）+ 分摊买入方案。

启动方式：streamlit run app.py
"""
import base64
import datetime as dt
import hmac
import json
import os
import sys
import time
from pathlib import Path

# Streamlit 以 __main__ 运行本脚本：把 "app" 别名指向本模块自身，
# 防止 import daily_push 时其模块级 import app 把本脚本再执行一遍（组件重复渲染报错）
if __name__ == "__main__":
    sys.modules["app"] = sys.modules["__main__"]

import akshare as ak
import pandas as pd
import requests
import streamlit as st

from data import (get_history, get_history_impl, get_spot,
                  industry_map_impl, refresh_spot, spot_cache_time, spot_fresh_enough,
                  spot_recently_failed, get_history_tx, cninfo_risk_impl, tushare_daily_basic)
from strategy import (at_limit_down, at_limit_up, backtest, calibrate_win_rate,
                      generate_signals)

import factors  # L1 多因子层：动量/波动率/换手/估值 + 行业中性 + 可配置窗口IC均值配权(默认60日)
# v1.32: T+1 超短线精筛内嵌（模型已 base64 内嵌，云端零文件路径依赖）
try:
    import t1_refine
except Exception:
    t1_refine = None
# 选股引擎统一放在 screen_service（纯 Python，无 Streamlit 依赖）；
# 这里显式导入，保证 app.xxx 在测试/脚本里沿用原名字
from screen_service import (CANDIDATES, CANDIDATES_PUSH, HISTORY_DAYS, IC_GATE, MAX_PRICE, MIN_PRICE, ML_WEIGHT,
                            REVIEW_CACHE_KEY, REVIEW_N, SETTINGS_FILE, STOP_LOSS_PCT,
                            TAKE_PROFIT_PCT, _CORR_THRESHOLD, _USE_CORR_FILTER,
                            _confidence, _correlation_filter, _kelly_position, _num,
                            _run_with_timeout, allocate, auto_pick_count, best_plan, cost_calculator,
                            check_holdings, lhb_codes_impl, load_holdings, load_settings,
                            market_temperature_impl, news_risk_impl, pick_top3,
                            review_conclusion, review_impl, screen_impl, verify_impl)
# v1.51.1 P0止血：减持标签是纯展示功能，改为软导入。增量部署中间态下若云端
# screen_service 还是旧版(无此函数)，硬导入会抛 ImportError 拖垮整个 app 白屏。
try:
    from screen_service import reduce_label_impl
except Exception:
    reduce_label_impl = None

# 缓存包装：web UI 一律调 news_risk；GitHub Actions 等裸运行环境请调 news_risk_impl
news_risk = st.cache_data(ttl=3600, show_spinner=False)(news_risk_impl)
# v1.51 股东减持标签：缓存1小时（减持公告按日更新，无需每次选股都拉），纯展示不影响选股
reduce_labels = (st.cache_data(ttl=3600, show_spinner=False)(reduce_label_impl)
                 if reduce_label_impl is not None else lambda codes: {})

# 缓存包装：web UI 一律调 market_temperature；裸运行环境请调 market_temperature_impl
market_temperature = st.cache_data(ttl=3600, show_spinner="正在检查大盘温度...")(market_temperature_impl)

# 缓存包装：web UI 一律调 lhb_codes；GitHub Actions 等裸运行环境请调 lhb_codes_impl
lhb_codes = st.cache_data(ttl=43200, show_spinner=False)(lhb_codes_impl)

# 缓存包装：web UI 一律调 review_performance(REVIEW_CACHE_KEY)；30分钟内重复点秒开
review_performance = st.cache_data(ttl=1800, show_spinner="正在自动复盘最近推荐表现...")(review_impl)


# ---------------- 数据源健康仪表盘辅助函数 ----------------
def _pipeline_health():
    """返回各数据源健康度 dict；pipeline 未初始化时返回 {}。"""
    try:
        from data_pipeline import health_snapshot
        return health_snapshot() or {}
    except Exception:
        return {}


def _warehouse_health():
    """返回 DuckDB 数仓健康报告；warehouse 不可用时返回 status 字段标注。"""
    try:
        from warehouse import Warehouse
        wh = Warehouse()
        report = wh.health_report()
        wh.close()
        return report
    except Exception as e:
        return {"status": "error", "detail": str(e)[:80]}


def _probe_pipeline():
    """轻量探测：拉 2 只 K线 + 1 只基本面，触发健康度记录。"""
    try:
        from data_pipeline import DataPipeline
        pl = DataPipeline()
        pl.get_history("000001", "20250101", "20250110", full=False)
        pl.get_history("600519", "20250101", "20250110", full=False)
        pl.get_fundamental("600519", max_quarters=2)
    except Exception:
        pass


# ---------------- 页面加载：不再首屏联网探测（v1.38 提速） ----------------
# 旧版每次新会话都会 _probe_pipeline() 拉2只K线+1只财报，只为给折叠的"数据源健康"面板
# 填首次数据，拖慢首屏。健康度在点「今日选股」时会自动积累，无需首屏预热。
if not st.session_state.get("_health_probed"):
    st.session_state["_health_probed"] = True
    st.session_state["_health_last_update"] = time.time()
    # 实盘台账自动回填（一天最多一次，内部节流。云端/本地打开页面即触发，
    # 这样即使某天没发邮件，只要有人看过页面，到期的5/10日收益也会被补齐）
    try:
        import oos_ledger
        oos_ledger.update_due()
    except Exception:
        pass


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
        # 复盘摘要：统一调 daily_push.build_review，与定时邮件完全同口径
        try:
            rv = review_impl()
            d["last"] = daily_push.build_last_batch(rv)
            d["review"] = daily_push.build_review(rv)
        except Exception:
            pass
        daily_push.send(daily_push._render_text(d), daily_push._render_html(d))
        # SMTP 成功后才留档，避免发送失败留下并未推送的推荐记录。
        try:
            _rec_cols = ["代码", "名称", "现价", "买入股数"]
            if "盈利概率" in plan.columns:                  # v1.46 留档当时的校准盈利概率，供复盘对照实际输赢
                _rec_cols.append("盈利概率")
            rec = plan[_rec_cols].copy()
            if "盈利概率" not in rec.columns:               # 兼容异常路径：保证列存在，旧值留空
                rec["盈利概率"] = float("nan")
            rec.insert(0, "日期", d["date"])
            rec.insert(1, "推送时间", daily_push._now_hm())   # 同一天多次推送靠它分批
            rec["大盘标签"] = d["tag"]
            daily_push._append_log(rec)                       # 统一整表重写，避免追加时列错位
            # 实盘 OOS 台账：网页端推送也要记一份评分+推荐价，否则网页发的推荐
            # 永远进不了台账，真实IC样本量始终是0（与定时邮件口径一致）
            try:
                import oos_ledger
                _score_col = next((c for c in ("总分", "评分", "盈利概率", "综合评分")
                                   if c in plan.columns), None)
                oos_ledger.record(d["date"], rec["推送时间"].iloc[0], [
                    {"代码": p["代码"], "名称": p["名称"],
                     "评分": float(p[_score_col]) if _score_col and pd.notna(p.get(_score_col)) else None,
                     "现价": float(p["现价"])}
                    for _, p in plan.iterrows()])
            except Exception as _oe:
                print("OOS台账记录(网页端)失败(不影响推送):", _oe)
        except Exception:
            pass
        return True, f"已发送到 {os.environ['QQ_EMAIL']}"
    except Exception as e:
        return False, f"发邮件失败：{type(e).__name__}: {e}"


# 网页改本金后提交到仓库，GitHub Actions 的每日邮件才能读到同一个值
GH_REPO = "laowanghenlan/stock-picker"
GH_SETTINGS_PATH = "推荐设置.json"
GH_BRANCH = "master"


def _gh_token() -> str:
    try:
        tok = st.secrets.get("GH_TOKEN", "")
    except Exception:
        tok = ""
    return str(tok or os.environ.get("GH_TOKEN", "") or "")


def save_settings(capital: float) -> tuple[bool, str]:
    """把本金写回 推荐设置.json：先写本地（本地运行立即生效），
    再用 GH_TOKEN 提交到 GitHub，每日邮件才能读到新本金。
    返回 (是否已同步到GitHub, 给用户看的提示)。"""
    text = json.dumps({"capital": round(float(capital), 2)}, ensure_ascii=False, indent=2)
    try:
        SETTINGS_FILE.write_text(text, encoding="utf-8")
    except Exception:
        pass
    token = _gh_token()
    if not token:
        return False, f"本金已改为 {capital:.0f} 元（未配置 GH_TOKEN，邮件推送暂不同步）"
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_SETTINGS_PATH}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, params={"ref": GH_BRANCH}, timeout=15)
        body = {"message": f"网页更新本金为 {capital:.0f} 元",
                "content": base64.b64encode(text.encode("utf-8")).decode(),
                "branch": GH_BRANCH}
        if r.status_code == 200:
            body["sha"] = r.json()["sha"]
        requests.put(url, headers=headers, json=body, timeout=15).raise_for_status()
        return True, f"本金已保存为 {capital:.0f} 元，下次邮件按新本金计算"
    except Exception as e:
        return False, f"本金已在本页生效，但同步到 GitHub 失败：{str(e)[:60]}"


def _gh_upload_file(path: Path, message: str) -> tuple[bool, str]:
    """把一个运行留档文件回传仓库；邮件状态仍由 GitHub 触发器单独判重。"""
    token = _gh_token()
    if not token:
        return False, "未配置 GH_TOKEN"
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{path.name}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        r = requests.get(url, headers=headers, params={"ref": GH_BRANCH}, timeout=15)
        body = {"message": message,
                "content": base64.b64encode(path.read_bytes()).decode(),
                "branch": GH_BRANCH}
        if r.status_code == 200:
            body["sha"] = r.json()["sha"]
        elif r.status_code != 404:
            r.raise_for_status()
        requests.put(url, headers=headers, json=body, timeout=15).raise_for_status()
        return True, "已同步"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:80]}"


def _sync_push_artifacts() -> list[str]:
    """邮件成功后同步复盘留档；同步失败只告警，绝不因此重发邮件。"""
    if not _gh_token():
        return ["未配置 GH_TOKEN，推送留档未回传"]
    errors = []
    for name in ("推送记录.csv", "实盘台账.csv", "因子历史IC.csv"):
        path = Path(__file__).parent / name
        if not path.exists():
            continue
        ok, msg = _gh_upload_file(path, f"Web 推送留档 {dt.date.today():%Y-%m-%d}")
        if not ok:
            errors.append(f"{name}: {msg}")
    return errors


@st.cache_data(ttl=10800, show_spinner=False)
def _load_push_precomputed(cache_day: str, slot: str, capital: float, version: str):
    """预热邮件专用选股结果，供同一时段的不同浏览器会话复用。"""
    if not spot_fresh_enough(1800):
        try:
            refresh_spot()
        except Exception:
            if spot_cache_time() == "无缓存":
                raise
    st.code("PRECOMPUTE_STAGE 0.050 全市场初筛")
    cand = screen_impl()
    if len(cand) == 0:
        return cand, pd.DataFrame()
    cand = cand.nlargest(CANDIDATES_PUSH, "成交额")
    res = verify_impl(
        cand, lambda p, t: st.code(f"PRECOMPUTE_STAGE {p:.3f} {t}"), float(capital))
    return cand, res


def _trigger_secret(name: str) -> str:
    try:
        return str(os.environ.get(name) or st.secrets.get(name, "") or "")
    except Exception:
        return str(os.environ.get(name, "") or "")


def _handle_scheduled_action() -> bool:
    """处理 GitHub 浏览器触发；返回 True 时由调用方停止普通页面渲染。"""
    action = str(st.query_params.get("action", "") or "")
    if not action:
        return False
    if action not in ("warmup", "push"):
        st.code("PUSH_ERROR")
        st.error("未知定时任务 action")
        return True
    token = str(st.query_params.get("token", "") or "")
    expected = _trigger_secret("PUSH_TRIGGER_TOKEN")
    if not expected or not hmac.compare_digest(token, expected):
        st.code("PUSH_ERROR")
        st.error("定时任务令牌无效")
        return True
    slot = str(st.query_params.get("slot", "am") or "am")
    if slot not in ("am", "pm"):
        st.code("PUSH_ERROR")
        st.error("slot 必须是 am 或 pm")
        return True

    marker = "WARMUP" if action == "warmup" else "PUSH"
    try:
        capital = float(load_settings()["capital"])
        cache_day = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=8)).strftime("%Y-%m-%d")
        precomputed = _load_push_precomputed(cache_day, slot, capital, APP_VERSION)
        if action == "warmup":
            st.code("WARMUP_OK")
            st.success(f"{cache_day} {slot} 预热完成：{len(precomputed[0])} 只候选")
            return True

        email = _trigger_secret("QQ_EMAIL")
        auth = _trigger_secret("QQ_AUTH_CODE")
        if not (email and auth):
            raise RuntimeError("Streamlit Secrets 未配置 QQ_EMAIL / QQ_AUTH_CODE")
        os.environ["QQ_EMAIL"] = email
        os.environ["QQ_AUTH_CODE"] = auth
        os.environ["PUSH_SLOT"] = slot
        import daily_push
        daily_push.CAPITAL = capital
        d = daily_push._collect(precomputed=precomputed, archive=False)
        daily_push.send(daily_push._render_text(d), daily_push._render_html(d))
        st.code("PUSH_OK")
        st.success(f"{slot} 时段邮件已发送")
        try:
            daily_push._archive(d)
            errors = _sync_push_artifacts()
            if errors:
                st.code("ARCHIVE_WARN")
                st.warning("；".join(errors))
            else:
                st.code("ARCHIVE_OK")
        except Exception as archive_error:
            st.code("ARCHIVE_WARN")
            st.warning(f"邮件已发送，留档同步失败：{str(archive_error)[:160]}")
    except Exception as e:
        st.code(f"{marker}_ERROR")
        st.error(f"定时任务失败：{type(e).__name__}: {str(e)[:160]}")
    return True


# web UI 沿用原名 screen；GitHub Actions 等裸运行环境请调 screen_impl
screen = screen_impl

# web UI 沿用原名 verify；GitHub Actions 等裸运行环境请调 verify_impl
verify = verify_impl

APP_VERSION = "v1.57"     # v1.57 早班cron提前4小时(北京04:00触发=UTC前一日20:00,0-4;预热03:30=UTC19:30):GitHub免费仓库schedule实测延迟4~5h(近5天早班实际触发11:51~13:35),提前4h让实际到信落08:00~09:00;晚班19:30/20:00不变,周末流水线不变,策略/打分/选股/发信逻辑零改动。前版 v1.56 周末云端流水线(GitHub Actions周六周日09:00):重训模型(用仓库自带training_data/wf_cache_bundle.tar.gz解压→pretrain,IC门控坏模型不覆盖)+跑一次选股写独立周末预测存档.csv(复用daily_push._collect,archive=False不污染推送记录/实盘台账,不联网发邮件,只回填到期台账样本);prediction/train两模式独立降级互不影响;新增weekend_pipeline.py+weekend.yml。策略/打分/排序/选股主流程零改动,不改工作日推送。前版 v1.55 股东减持标签窗口90→60天(用户要求"天数改为60天之内"):screen_service.reduce_label_impl默认lookback_days 90→60,web与邮件都调此函数默认参数故改一处两端同步;🔴新减持30天分层不变(累计比例按60天算,最新公告≤30天标🔴);app caption与daily_push注释文案同步改60天;纯展示口径改动,不改选股/打分/排序/排雷。验证:离线9项断言全过(默认60天filter起始日/60天内聚合/80天前不计/AFTER_CHANGE_RATE字段正确/30天分层/显式覆盖兼容/失败降级—),py_compile过。前版 v1.54 实盘台账改为「股票×交易日」口径:去重键日期+推送时间+代码→日期+代码(一天点多次选股不重复计数,保留最后一次名单且已回填行优先不被抹掉);新增T+1隔夜收益(用户真实打法今收买明收卖)——买入价=推荐日(或之后最近交易日)收盘价(非点击现价,消除盘中多次点击的基准漂移),卖出价=下一交易日收盘,毛收益=卖/买-1,净收益=毛-0.40%往返成本(佣金万2.5双边触底5元+印花0.05%卖出+过户0.001%双边+双边滑点各0.1%的保守估计);新增列T+1买入日/T+1买入价/实际T+1收益%/T+1净收益%,旧表缺列自动补全;周末推送顺延到下一交易日买入;5/10日预测口径不变(基准仍为点击现价);update与record共用_dedup,旧格式同日多批次残留(如迁移前11-18两批)在下次回填时自动收敛到一天一条。纯记账口径改动,不改选股/打分/排序/邮件发信行为(邮件照常每次点都发,按用户选择不动)。验证:离线16项断言全过(同日3次合并/保留最后名单/T+1毛净收益/已回填不被覆盖/二次回填不重复/旧格式自动收敛/周末顺延/旧表兼容),py_compile过。台账展示区仍按v1.37要求隐藏,T+1汇总走本地命令行查看。前版 v1.53 邮件也显示减持信息:daily_push的build_plan_items批量查近90天减持写入每个item(与web同口径🔴新减持分层),文本版+HTML版两处渲染都接上,失败降级—不影响发信。前版 v1.52 减持标签分层标注:累计比例仍按近90天算,最新一笔公告在近30天内则标🔴新减持、否则普通"减持X%";复用reduce_label_impl已返回的last_date加30天cutoff判定,纯展示不改选股/打分/排序/排雷。前版 v1.51.2 P0修复:c4.metric缺value参数(只传label+help,metric要求label+value)抛TypeError白屏——v1.50引入被v1.51 ImportError挡在前面,止血后暴露;改为metric("本次 / 成本最优","N只 / M只")。AST全扫35处metric参数完整。前版 v1.51.1 P0止血:减持标签改软导入try/except,防增量部署中间态(云端screen_service旧版无此函数)抛ImportError拖垮整页白屏;函数缺失时标签显示—。前版 v1.51 结果表加"股东减持"标签(纯展示):按入选股代码查东财RPT_SHARE_HOLDER_INCREASE近90天减持,用正确字段AFTER_CHANGE_RATE(非报价涨跌幅CHANGE_RATE)聚合累计减持占股本%+笔数,显示"减持X.XX%·N笔"/"—";失败降级—不影响选股。缓存1h。顺带修复screen_service漏import re(_em_news_direct用re.sub却从未导入→NameError被吞→舆情排雷一直静默失效,补import re后恢复)。纯展示不改选股/打分/排序/排雷。前版 v1.50 把成本计算器集成进选股结果区(用户要求"点今日选股后在股票只数这加进去"):删除v1.49独立折叠面板,改为在"今日选出N只"前用本次真实plan逐笔算——4个metric(往返费用/回本涨幅门槛含双边滑点/费用占买入/本次只数vs成本最优只数)+触底佣金告警+「不同只数成本对比」折叠表(按本次均价试算供对照不改方案);卖出费用逐笔按max(5,amt×0.00025)+印花税+过户费算(大额单只更准);纯展示不改选股/打分/精筛/风控。沿用 v1.49 新增「T+1交易成本计算器」折叠面板:输入本金/股价/只数,实时算往返总费用(佣金万2.5最低5元+印花税0.05%卖出+过户费0.001%双向)、回本涨幅门槛(含双边滑点0.1%)、费用占比、成本视角最优只数(单笔≥2万避开5元佣金封底)、多只数对比表;纯本地计算screen_service.cost_calculator无副作用,口径与allocate/best_plan一致,不改选股/打分/精筛/风控。沿用 v1.48 P0修复线上KeyError:REVIEW_CACHE_KEY v1.22→v1.48强制st.cache_data旧复盘缓存失效(旧缓存无「当时盈利概率」列,g[_cols]报错整页白屏)+app展示前兜底缺列补—;沿用 v1.47 邮件历史表现也显示盈利概率:复盘明细/上一次推送卡片的文本版+HTML版四段渲染都补上「当时概率」列(v1.46只加了数据字段没接渲染层,邮件看不到);沿用 v1.46 历史复盘加「当时盈利概率」列:推送留档存下当时的校准盈利概率(web端app.py+邮件端daily_push.py两条路径)+复盘表与盈亏%并排对照,旧批次无留档显示—不编造;沿用 v1.45 价格带严格生效:去掉CSI成分股价格上限豁免(旧逻辑让100~500元高价股也进池)+下限5→8元+页面显示完整区间8~50;沿用 v1.44 改名老王每日选股+v1.43 P0修复(漏import numpy致选股恒空)+v1.42 提速四项+v1.41 舆情直连降级放行+v1.40 去持仓体检+v1.39 去资金流对照+v1.38 web提速+v1.37 隐藏台账+v1.36 清v1.24标签+v1.32 T+1精筛+v1.28 全因子+v1.26 复盘净额

st.set_page_config(page_title="老王每日选股", page_icon="📈", layout="centered")
if _handle_scheduled_action():
    st.stop()
st.title("📈 老王每日选股")
st.markdown(f"**当前版本：{APP_VERSION}**")

# ---------------- 大盘温度横幅（低温期提示空仓） ----------------
_idx_result = _run_with_timeout(market_temperature, timeout=20)
idx_close, idx_ma60, market_cold = _idx_result or (0.0, 0.0, False)
if idx_close <= 0 or idx_ma60 <= 0:
    st.warning("大盘温度暂时不可用（行情接口超时），已跳过检查，不影响选股和复盘。")
    force_run = True
elif market_cold:
    st.error(f"❄️ 大盘低温期：沪深300（{idx_close:.0f}）在60日线（{idx_ma60:.0f}）下方，"
             f"系统性下跌段。默认建议空仓，执意选股请勾选下方开关。")
    force_run = st.checkbox("我知道风险，仍要选股")
else:
    st.success(f"☀️ 大盘温度正常：沪深300（{idx_close:.0f}）位于60日线上方")
    force_run = True

# ---------------- 本金输入在侧边栏（见下方 sidebar 区块），此处仅保留说明 ----------------
# capital / n_picks 在侧边栏代码里赋值；MIN_PRICE / MAX_PRICE 由 strategy_config.json 决定

# ---------------- 持仓体检：v1.40 按用户要求从页面隐藏（持仓.csv 与 check_holdings 逻辑保留，本地可查） ----------------

# ---------------- 侧边栏：只保留本金输入（行情/复盘/因子/熔断全部自动） ----------------
with st.sidebar:
    def _on_capital_change():
        """输入框改完自动保存，避免网页和每日邮件各用一套本金"""
        _v = float(st.session_state.get("capital_input", 0) or 0)
        if _v > 0 and abs(_v - float(load_settings()["capital"])) > 1e-9:
            st.session_state["_capital_msg"] = save_settings(_v)

    # 本金不设上下限：想投多少就输多少，只要求大于0（后面要用它算仓位和股价上限）
    capital = st.number_input("💰 本金（元）", min_value=1.0, max_value=None,
                              value=float(load_settings()["capital"]), step=100.0,
                              format="%.0f", key="capital_input",
                              on_change=_on_capital_change,
                              help="输入后自动保存，网页与每日邮件共用同一个本金")
    _cap_msg = st.session_state.pop("_capital_msg", None)
    if _cap_msg:
        (st.success if _cap_msg[0] else st.info)(_cap_msg[1])
    n_picks = auto_pick_count(capital)
    st.caption(f"只数由本金+预期收益自动计算（当前约 {n_picks} 只起）")
    st.caption(f"行情快照时间：{spot_cache_time()}")

    # ---------------- 数据源健康仪表盘 ----------------
    with st.expander("🔧 数据源健康", expanded=False):
        # 不做 st.rerun() 定时刷新：脚本先跑侧边栏再跑按钮，rerun 会中断脚本并吞掉
        # 「今日选股」的点击状态（2026-11-17 实测：点击后页面无反应）。健康度是内存实时值，
        # 任何页面交互都会重跑脚本并读到最新数据，无需主动刷新。
        st.caption("多源动态路由 · 健康度随页面操作自动更新 · 冷却中的源会自动跳过")
        _health = _pipeline_health()
        if not _health:
            st.info("等待首次探测（页面加载时已自动触发）...")
        else:
            # 按健康分排序展示
            _sorted = sorted(_health.items(),
                             key=lambda kv: -_health[kv[0]].get("健康分", 0))
            for _name, _info in _sorted:
                _score = _info.get("健康分", 0)
                if _info.get("冷却中"):
                    _ic, _color = "⏸️", "gray"
                elif _score >= 0.8:
                    _ic, _color = "🟢", "green"
                elif _score >= 0.5:
                    _ic, _color = "🟡", "orange"
                else:
                    _ic, _color = "🔴", "red"
                st.markdown(f"{_ic} **{_name}** "
                            f"<span style='color:{_color}'>健康分 {_score:.2f}</span>"
                            f"　成功率 {_info.get('成功率','—')}　延迟 {_info.get('延迟','—')}"
                            f"　完整率 {_info.get('完整率','—')}"
                            f"　连败 {_info.get('连续失败', 0)}",
                            unsafe_allow_html=True)

            # 仓库状态
            _wh = _warehouse_health()
            if _wh.get("status") == "ok":
                st.divider()
                st.caption("📦 DuckDB 数仓")
                _c1, _c2 = st.columns(2)
                _c1.metric("快照条数", f"{_wh.get('spot_daily_rows', 0):,}")
                _c2.metric("K线条数", f"{_wh.get('kline_daily_rows', 0):,}")
                _c3, _c4 = st.columns(2)
                _c3.metric("股票数", f"{_wh.get('kline_codes', 0):,}")
                _c4.metric("文件大小", f"{_wh.get('db_size_mb', 0)} MB")
                if _wh.get("spot_latest"):
                    st.caption(f"最新快照时间：{_wh['spot_latest']}")
            else:
                st.caption(f"📦 数仓：{_wh.get('status', '未初始化')}（不影响选股，自动降级到 CSV）")

# MAX_PRICE 已从 screen_service 导入，由 strategy_config.json 决定

# ---------------- 历史推荐复盘（按需加载，避免云端启动时网络 hang 导致白屏） ----------------
with st.expander(f"📊 历史推荐复盘（最近{REVIEW_N}批）", expanded=False):
    if not st.button("🔄 加载复盘数据", key="load_review"):
        st.caption("点击上方按钮加载复盘（需联网拉历史行情，约10~20秒）。"
                   "云端启动时不自动执行，避免数据源 hang 住拖垮整个页面。")
    else:
        rv = review_performance(REVIEW_CACHE_KEY)
        if len(rv) > 0 and "当时盈利概率" not in rv.columns:
            rv["当时盈利概率"] = "—"   # 兜底：旧缓存缺列也不崩（正常靠 REVIEW_CACHE_KEY 失效解决）
        if len(rv) == 0:
            st.caption("暂无历史推荐，有推荐后会自动复盘。")
        else:
            done = rv[rv["_done"]]
            holding = rv[~rv["_done"]]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("推荐总数", f"{len(rv)} 条 / {rv['_batch'].nunique()} 批")
            if len(done) > 0:
                wins = int((done["_盈亏额"] > 0).sum())   # 按实际到手净额判胜（扣费后）
                m2.metric("胜率", f"{wins / len(done) * 100:.0f}%（{wins}/{len(done)}）")
                m3.metric("平均每笔盈亏", f"{done['盈亏%'].mean():+.2f}%")
                m4.metric("累计盈亏", f"{done['_盈亏额'].sum():+.0f} 元")
            else:
                m2.metric("胜率", "—")
                m3.metric("平均每笔盈亏", "—")
                m4.metric("累计盈亏", "—")
            # 第二行：赚钱和亏钱各多少笔、各多少元（用户最关心的绝对金额）
            _win, _loss = done[done["盈亏额"] > 0], done[done["盈亏额"] < 0]
            n1, n2, n3, n4 = st.columns(4)
            n1.metric("盈利笔数", f"{len(_win)} 笔")
            n2.metric("盈利金额", f"+{_win['盈亏额'].sum():.0f} 元")
            n3.metric("亏损笔数", f"{len(_loss)} 笔")
            n4.metric("亏损金额", f"{_loss['盈亏额'].sum():.0f} 元")
            st.caption(f"持有中 {len(holding)} 只，浮动盈亏 {holding['盈亏额'].sum():+.0f} 元"
                       f"（未计入胜率与累计盈亏）；平均每批推荐 "
                       f"{len(rv) / max(1, rv['_batch'].nunique()):.1f} 只")
            _cols = ["名称", "推荐价", "现价", "股数", "费用(元)", "涨跌%", "当时盈利概率", "盈亏%", "盈亏额", "同期沪深300%", "结果"]
            _col_fmt = {"费用(元)": st.column_config.NumberColumn(format="%.1f 元"),
                        "涨跌%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "盈亏%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "盈亏额": st.column_config.NumberColumn(format="%+.0f 元"),
                        "同期沪深300%": st.column_config.NumberColumn(format="%+.2f%%")}
            # 按推送批次（日期+时间）分组显示，批次之间用红色横线隔开
            for i, (_, g) in enumerate(rv.groupby("_batch", sort=False)):
                if i:
                    st.markdown("<hr style='border:none;border-top:2px solid #ff4b4b;margin:12px 0'>",
                                unsafe_allow_html=True)
                _day, _tm = g["日期"].iloc[0], g["推送时间"].iloc[0]
                _when = f"{_day} {_tm}" if _tm else _day
                st.markdown(f"**📅 {_when} 推送 · {len(g)} 只 · 本批盈亏 {g['盈亏额'].sum():+.0f} 元**")
                st.dataframe(g[_cols], width="stretch", hide_index=True, column_config=_col_fmt)
            st.info("💡 " + review_conclusion(rv))
            if len(done) < 30:
                st.caption(f"⚠️ 已出结果仅 {len(done)} 笔（<30），统计意义不足，结论别太当真；"
                           "攒到30~60笔后再看胜率才有参考价值。")
            st.caption("判定规则：推荐日后最长跟踪20个交易日，收盘涨超+10%为🎯止盈、跌破-5%为🛑止损；"
                       "满20日未触发按到期收盘定盈亏（🏁）；现价为触发当日/最新收盘价；"
                       "未满20日且未触发的标 ⏳持有中，不计入胜率。")
            st.caption("💰 盈亏额=卖出到手−买入成本，已扣佣金(单笔最低5元)、印花税、过户费"
                       "（每笔约10元固定成本）；费用列可核对。涨跌%是纯价格涨跌，盈亏%是扣费后的净收益率，"
                       "胜率按盈亏额正负统计——小额盈利扣费后转亏的，不计为胜。")
            st.caption("🎯 当时盈利概率=推荐当天按回测胜率做贝叶斯收缩后的校准值，和盈亏%并排看："
                       "说60%盈利的那批，实际赚了还是亏了，一眼能看出概率准不准。"
                       "该字段从 v1.46 起才留档，早期批次显示「—」且无法补算（不编造历史）。")

# ---------------- 实盘 OOS 台账：后台照常记录与自动回填（v1.37 按用户要求从页面隐藏，数据不删） ----------------
# 记录/回填逻辑在上方 _handle_scheduled_action 与页面初始化处（oos_ledger.record/update_due），
# 不再在界面展示台账区。数据仍写入 实盘台账.csv，需要时本地查看即可。

# ---------------- 页面：一个按钮 + 一张结果表 ----------------
st.caption(f"本金 {capital:.0f} 元 · 只数由本金+预期收益自动优化 · 股价 {MIN_PRICE:.0f}~{MAX_PRICE:.0f} 元（严格过滤） · "
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
                # v1.42 提速：30分钟→2小时。T+1选股对快照新鲜度不敏感（收盘后价格固定），
                # 频繁重拉全市场反而每次多花1~2分钟；盘中要强制更新可清 spot_cache.csv。
                if not spot_fresh_enough(7200):
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
    _run_t0 = time.time()                      # v1.24: 漏斗视图耗时基准
    # v1.24 修复：这里必须用带缓存的 get_spot()。get_spot_impl() 是无缓存原始函数，
    # 会无视 30 分钟新鲜度、绕过"刚失败不重试"保护，在点击瞬间重新全市场联网拉一遍，
    # 且发生在进度条之前 → 用户看到的就是"点了一下卡住不动"。
    try:
        _spot_snapshot = get_spot() if spot_fresh_enough() else None
    except Exception:
        _spot_snapshot = None

    bar = st.progress(0.0, text="正在全市场初筛...")
    cands = screen(MAX_PRICE)
    if len(cands) == 0:
        st.warning("今天全市场没有满足基本条件的股票，建议空仓等待。")
        st.stop()
    cands = cands.nlargest(CANDIDATES, "成交额")        # 优先验证成交活跃的
    bar.progress(0.05, text=f"初筛通过 {len(cands)} 只，开始逐只回测（含手续费）...")
    res = verify(cands, lambda p, t: bar.progress(0.05 + 0.9 * p, text=t), float(capital))
    bar.progress(1.0, text="完成")
    bar.empty()

    if len(res) == 0:
        _hf = res.attrs.get("hist_failed", [])
        _nf = res.attrs.get("news_failed", [])
        if _nf and len(_nf) >= len(cands):
            st.warning("舆情数据源暂时不可用，候选股风险无法核实（这不是选股结论）。"
                       "但本次结果为空是行情/回测环节所致，与舆情无关，请稍后重试。")
        elif _hf and len(_hf) >= len(cands):
            st.warning("行情数据源暂时不可用（东财限流、新浪未返回有效数据），本次无法完成验证。"
                       "这不是选股结论，请稍后重试，不建议据此操作。")
        else:
            # v1.43：候选处理异常优先显示真实原因，不再一律甩锅"回测不合格"。
            _ve = res.attrs.get("verify_errors", [])
            if _ve:
                _sample = _ve[0].get("错误", "未知错误")
                st.error(f"⚠️ 候选股处理时程序出错（不是选股结论，也不是股票不好）：{_sample}。"
                         f"共 {len(_ve)} 只受影响，请把这个提示反馈给维护者修复。")
            else:
                st.warning("候选股均未通过验证（历史不足/回测不合格），今天不建议买入。")
        st.stop()
    _nf = res.attrs.get("news_failed", [])
    if _nf:
        st.caption(f"⚠️ 舆情接口暂时取不到数据，{len(_nf)} 只候选的风险未核实，已降级放行（未剔除）："
                   f"{', '.join(_nf[:10])}{' 等' if len(_nf) > 10 else ''}。"
                   "这批票的排雷结论不可用，买入前请自行核对最新公告。")
    # 舆情排雷通报：哪些因近7天利空新闻被排除
    mined = res[res["有雷"]]
    if len(mined) > 0:
        st.error(f"⚠️ 排雷：{len(mined)} 只候选股近7天有利空新闻，已自动排除——")
        for _, m in mined.iterrows():
            st.markdown(f"- **{m['名称']}（{m['代码']}）**：{m['利空标题']}")
    # IC 门控：训练期组合 IC < IC_GATE → 信号弱到噪声级，今日空仓
    _ic_gate = float(res.attrs.get("ic_train_composite") or 0.0)
    if _ic_gate < IC_GATE:
        st.error(f"🚫 因子近期信号偏弱（训练期组合IC={_ic_gate:+.4f} < {IC_GATE}），"
                 "按规则今日空仓不选股。")
        st.caption("这是风控开关：信号弱到噪声级时硬推票，期望收益不如空仓。")
        st.stop()

    # ===== 三级漏斗视图（粗筛→验证→DL精筛 的可解释性）=====
    with st.expander("🔻 三级漏斗可视化", expanded=False):
        try:
            from cascade_runner import CascadeRunner, CoarseScreener
            # Stage1: 复用已缓存的快照做粗筛（绝不在这里发起新的联网拉取）
            if _spot_snapshot is None:
                try:
                    _spot_snapshot = get_spot()
                except Exception:
                    _spot_snapshot = None
            if _spot_snapshot is not None and len(_spot_snapshot) > 100:
                _s1 = CoarseScreener().screen(_spot_snapshot)
                st.markdown("##### 漏斗总览")
                _cols = st.columns(4)
                _cols[0].metric("Stage1 输入", f"{_s1.input_count} 只")
                _cols[1].metric("Stage1 通过", f"{_s1.passed_count} 只",
                                f"通过率 {_s1.pass_rate:.0%}")
                _cols[2].metric("Stage2 通过", f"{len(res)} 只 / 输入 {_s1.passed_count} 只")
                _cols[3].metric("耗时", f"{time.time() - _run_t0:.1f}s")
                st.caption(f"漏斗: {_s1.input_count} → {_s1.passed_count} → {len(res)} → {min(len(res), 20)}")

                # Stage1 拒绝原因 Top
                _rejects_by_cat: dict[str, list] = {}
                for _, r in _s1.rejected:
                    _rejects_by_cat.setdefault(r.category, []).append(r)
                with st.expander(f"📋 Stage1 拒绝原因（{len(_s1.rejected)} 只）", expanded=False):
                    for _cat, _list in sorted(_rejects_by_cat.items(),
                                              key=lambda kv: -len(kv[1])):
                        with st.expander(f"  • {_cat}（{len(_list)} 只）"):
                            for _r in _list[:8]:
                                st.caption(f"{_r.detail}")
                            if len(_list) > 8:
                                st.caption(f"... 还有 {len(_list)-8} 只")

                # Stage2 拒绝原因（从 verify_impl 看 res 的 hists 和 trend_up 字段反推）
                _failed_hists = [cd for cd, h in (res.attrs.get("hists") or {}).items()
                                 if h is None or len(h) < 60]
                _failed_trend = (res["趋势向上"] == False).sum() if "趋势向上" in res.columns else 0
                _failed_fee = ((res["扣费总收益"] or 0) <= 0).sum() if "扣费总收益" in res.columns else 0
                _failed_雷 = res["有雷"].sum() if "有雷" in res.columns else 0
                with st.expander(f"📋 Stage2 拒绝原因（从 verify_impl 反推）", expanded=False):
                    st.caption(f"  • 历史K线不足60根: {len(_failed_hists)} 只")
                    st.caption(f"  • MA金叉不成立: {_failed_trend} 只")
                    st.caption(f"  • 扣费回测收益 ≤ 0: {_failed_fee} 只")
                    st.caption(f"  • 有近期利空公告: {_failed_雷} 只")
                    st.caption("  完整的 Stage2 拒绝列表见 cascade_runner.py FactorVerifier")
            else:
                st.info("需要先有快照数据才能展示漏斗")
        except Exception as _e:
            st.caption(f"漏斗视图加载中...（{_e}）")

    ranked = pick_top3(res, min(len(res), 20))
    if len(ranked) == 0:
        st.warning("今天没有股票通过回测+排雷验证。空仓也是一种操作，明天再来。")
        st.stop()

    # ===== v1.32: T+1 超短线精筛，直接决定最终股票顺序 =====
    # 复用 verify 阶段已取的候选股 K线(res.attrs['hists'])，不额外联网；
    # 按训练面板口径现算8特征 → 内嵌 LightGBM 打分 → 截面排序。
    # 模型样本外(2025-07起) top3 次日净超额+1.56%/期、t=7.57、跑赢率68%（技术文档第二十节）。
    # 失败/无模型时保持原5日口径顺序，选股不中断。
    _t1_applied = False
    _t1_basis = ""
    if t1_refine is not None:
        try:
            if t1_refine.available():
                _hists = ranked.attrs.get("hists") or {}
                _refined = t1_refine.refine(ranked, _hists)
                if _refined is not None and len(_refined) > 0:
                    # 保留 attrs（熔断/IC诊断下游仍要用）
                    _old_attrs = dict(ranked.attrs)
                    ranked = _refined
                    ranked.attrs.update(_old_attrs)
                    _t1_applied = True
                    _t1_basis = "T+1 LightGBM（含退市全A样本外+1.56%/期，t=7.57）"
        except Exception as _e:
            _t1_basis = f"T+1精筛跳过：{str(_e)[:60]}"
    if _t1_applied:
        st.caption(f"🎯 **T+1 精筛已启用**：{' '.join(ranked['代码'].astype(str).str.zfill(6).head(5))} "
                   f"按次日收益模型重排（{_t1_basis}）。今收买、明收卖，统计优势非必赚。")
    elif _t1_basis:
        st.caption(f"ℹ️ T+1精筛未启用（{_t1_basis}），沿用5日口径排序。")

    # Stage3 三模型集成精筛诊断（v1.24 接入主流程：LightGBM复用L2 + ExtraTrees + MLP）
    _s3 = ranked.attrs.get("stage3") or {}
    if _s3:
        _s3_models = "、".join(_s3.get("models_used") or []) or "无可用模型"
        _s3_ic = _s3.get("model_ic") or {}
        _s3_ic_txt = "（" + "，".join(
            f"{k} 样本外IC={v:+.3f}" for k, v in _s3_ic.items() if v == v) + "）" \
            if any(v == v for v in _s3_ic.values()) else ""
        if _s3.get("applied"):
            st.caption(f"🧠 Stage3 三模型精筛（{_s3_models}）：保留 {_s3.get('reason', '')}，"
                       f"耗时 {_s3.get('time_s', '?')}s {_s3_ic_txt}")
            with st.expander(f"🧠 精筛剔除明细（{len(_s3.get('rejected') or [])} 只）", expanded=False):
                for _d in (_s3.get("rejected") or [])[:15]:
                    st.caption(_d)
        else:
            st.caption(f"🧠 Stage3 集成精筛未执行：{_s3.get('reason', '')}")
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

    # ===== 机器学习加分项状态（LightGBM/XGBoost）=====
    _ml = res.attrs.get("ml")
    if _ml:
        if _ml.get("error"):
            st.caption(f"🤖 机器学习未启用：{_ml['error']}")
        else:
            _ic_txt = f"{_ml['oof_ic']:+.3f}" if _ml.get("oof_ic") == _ml.get("oof_ic") else "无效"
            _imp = "、".join(str(k).replace("_n", "") for k, _ in _ml.get("importance", [])[:3])
            st.caption(f"🤖 机器学习（{_ml.get('backend')}）：样本外IC {_ic_txt}"
                       f"（{'达标' if _ml.get('oof_pass') else '未达标，不参与打分'}），"
                       f"训练 {_ml.get('n_rows')} 样本 / {_ml.get('n_dates')} 日，"
                       f"占总分 {ML_WEIGHT:.0%}；模型最看重：{_imp}")

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

    # ===== 专业统计诊断（胜率显著性 / IC稳定性 / VaR / 分布） =====
    with st.expander("📐 专业统计诊断", expanded=False):
        _win_sig = res.attrs.get("stats_win_sig") or {}
        _stab = res.attrs.get("stats_factor_stab") or {}
        _var = res.attrs.get("stats_var") or {}
        _dist = res.attrs.get("stats_dist") or {}
        _discount = res.attrs.get("stats_win_discount", 1.0)

        if not _win_sig and not _stab:
            st.caption("（本次未计算统计诊断，可能是候选太少或因子IC检验跳过）")
        else:
            # (a) 胜率显著性
            c1, c2, c3, c4 = st.columns(4)
            _p = _win_sig.get("p_value", 1.0)
            _wilson = _win_sig.get("wilson_lower", 0)
            _wrate = _win_sig.get("胜率", 0)
            _trades = _win_sig.get("n", 0)
            c1.metric("总交易笔数", f"{_trades} 笔")
            c2.metric("回测胜率", f"{_wrate:.0%}")
            c3.metric("Wilson下界", f"{_wilson:.0%}",
                      help="95% 置信区间下界；>50% 才算真的比随机强")
            c4.metric("p-value", f"{_p:.3f}",
                      delta="显著" if _p < 0.05 else ("接近" if _p < 0.15 else "不显著"),
                      help="二项分布精确检验；<0.05 才算统计上显著")
            st.caption(f"→ {_win_sig.get('结论', '')}")
            if _discount < 1.0:
                st.caption(f"→ ⚠️ 本次因胜率不显著，对总分打了 {_discount:.0%} 折")

            # (b) 因子稳定性
            if _stab:
                st.divider()
                st.caption("🧬 因子稳定性（ICIR + 衰减率）")
                _stab_rows = []
                for _f, _s in sorted(_stab.items(),
                                     key=lambda kv: -(kv[1].get("icir", 0))):
                    _stab_rows.append({
                        "因子": _f,
                        "IC均值": f"{_s.get('ic_mean', 0):+.3f}",
                        "ICIR": f"{_s.get('icir', 0):.2f}",
                        "衰减率": f"{_s.get('decay_ratio', 1):.0%}",
                        "偏度": f"{_s.get('ic_skew', 0):+.2f}",
                        "质量": _s.get("quality", "")[0],  # 星数
                    })
                st.dataframe(pd.DataFrame(_stab_rows), width="stretch", hide_index=True)
                st.caption("ICIR = IC均值/IC标准差（≥1.5 极强，<0.5 不稳定）；衰减率 = 最近20天IC/全样本IC（<50% 因子正在失效）"
                           "；权重已按稳定性自动惩罚（ICIR<0.5 或衰减<50% 的因子权重砍半）")

            # (c) VaR + 分布
            if _var:
                st.divider()
                v1, v2, v3, v4 = st.columns(4)
                v1.metric("VaR(95%)", f"{_var.get('var_95', 0):.1%}",
                          help="95% 概率下单日亏损不会超过这个数（历史分位数，不假设正态）")
                v2.metric("CVaR(95%)", f"{_var.get('cvar_95', 0):.1%}",
                          help="极端情况下的平均日亏损，比 VaR 更保守")
                v3.metric("最大回撤", f"{_var.get('max_drawdown', 0):.1%}")
                v4.metric("年化波动", f"{_var.get('annualized_vol', 0):.1%}")
            if _dist:
                st.caption(f"📊 分布形态：偏度 {_dist.get('skew', 0):+.2f} "
                           f"峰度 {_dist.get('kurt', 0):+.2f} → {_dist.get('tail_type', '')}")

    # ---- 相关性过滤（v1.8 全开，带置信度折扣） ----
    if _USE_CORR_FILTER:
        try:
            _n_done = int(ranked.attrs.get("n_done", 0) or 0)
            ranked = _correlation_filter(ranked, get_spot(),
                                         threshold=_CORR_THRESHOLD, n_done=_n_done)
        except Exception:
            pass
    plan, n_final = best_plan(ranked, float(capital))
    if len(plan) == 0:
        st.warning(f"所有候选股的买入金额都超过了本金 {capital:.0f} 元，"
                   "自动将最低 1 手下限放宽。可以：① 增加本金 ② 等到低价股出现 ③ 手动在持仓.csv 里买入。")
        st.stop()
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

    # ---- 交易成本计算器（v1.50 集成）：用本次真实选出的股票逐笔算，输出费用/回本门槛/成本最优只数 ----
    # 口径与 screen_service.cost_calculator / allocate / best_plan 完全一致，只是把本就扣过的成本明面展示
    _amt = float((plan["买入股数"] * plan["现价"]).sum())
    _buy_fee = float((plan["投入(含佣金)"] - plan["买入股数"] * plan["现价"]).sum())
    # 卖出逐笔（每只单独触底判断，单只≥2万按比例、否则5元封底），与 best_plan 同口径
    _amt_each = (plan["买入股数"] * plan["现价"]).tolist()   # 每只成交额数组
    _sell_fee = float(sum(max(5.0, a * 0.00025) + a * 0.0005 + a * 0.00001 for a in _amt_each))
    _fee_total = _buy_fee + _sell_fee
    _slip = _amt * 0.001 * 2                       # 双边滑点 0.1%×2（回本门槛口径）
    _be_pct = (_fee_total + _slip) / _amt * 100 if _amt > 0 else 0.0
    _fee_pct = _fee_total / _amt * 100 if _amt > 0 else 0.0
    # 用本次真实平均股价试算不同只数的成本，给出"成本视角最优只数"供对照（不改实际方案）
    _avg_price = float((plan["买入股数"] * plan["现价"]).sum() / plan["买入股数"].sum())
    _cc = cost_calculator(float(capital), _avg_price, 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("往返费用", f"{_fee_total:.2f} 元",
              help=f"买入佣金 {_buy_fee:.2f}（万2.5/笔最低5元）+ 卖出 {_sell_fee:.2f}（含印花税0.05%）")
    c2.metric("回本涨幅门槛", f"{_be_pct:.3f}%", help="股价涨过这个比例才不亏手续费+双边滑点")
    c3.metric("费用占买入", f"{_fee_pct:.3f}%")
    c4.metric("本次 / 成本最优", f"{len(plan)} 只 / {_cc['建议只数']} 只",
              help=f"单笔≥{_cc['临界本金']:.0f}元才不被5元最低佣金惩罚；实际只数由预期收益定")
    if _cc["detail"]["最低佣金封底"] and len(plan) > 1:
        st.caption(f"⚠️ 每只均价约 {_avg_price:.1f} 元、平均投入 {_amt/len(plan):.0f} 元 < "
                   f"{_cc['临界本金']:.0f} 元，佣金被5元最低标准顶住、费用占比偏高；"
                   "若只求省手续费，减少只数或增加本金可摊薄（但会牺牲分散度，实际只数由预期收益决定）。")
    with st.expander("💰 不同只数的成本对比（按本次均价试算，仅供对照、不改方案）", expanded=False):
        st.caption("佣金万2.5（单笔最低5元）+ 印花税0.05%卖出 + 过户费0.001%双向；回本门槛另含双边滑点各0.1%。"
                   "纯本地计算，口径与选股扣费一致。")
        st.dataframe(pd.DataFrame(_cc["rows"])[
            ["只数", "每只股数", "每只金额", "往返合计", "费用占比%", "回本涨幅%", "最低佣金封底"]
        ], width="stretch", hide_index=True)

    _headline = "🎯 今日 T+1 精筛最终名单" if _t1_applied else "今日选出"
    st.success(f"{_headline} {len(plan)} 只，方案如下：")
    # v1.41：最终名单里若有舆情未核实的票，显著标注，提醒用户自行核对公告
    if "舆情未核实" in ranked.columns:
        _final_codes = set(plan["代码"].astype(str).str.zfill(6))
        _unver = ranked[ranked["舆情未核实"] == True]  # noqa: E712
        _unver_final = [f"{r['名称']}（{str(r['代码']).zfill(6)}）"
                        for _, r in _unver.iterrows()
                        if str(r["代码"]).zfill(6) in _final_codes]
        if _unver_final:
            st.warning("⚠️ 以下入选股票的舆情接口未取到数据、风险未核实（已降级放行，非利空确认），"
                       "买入前请自行核对最新公告：" + "、".join(_unver_final))
    show = plan.drop(columns=["_win_prob", "_est_win", "_est_loss"])
    if _t1_applied and "精筛分" in ranked.columns:
        # 把 T+1 精筛分/排名按代码补进最终方案表（allocate 可能重排，用 map 对齐）
        _sc = ranked.set_index("代码")["精筛分"].to_dict()
        _rk = ranked.set_index("代码")["T+1排名"].to_dict()
        show = show.copy()
        show["代码"] = show["代码"].astype(str).str.zfill(6)
        show.insert(0, "T+1排名", show["代码"].map(_rk))
        show.insert(1, "精筛分", show["代码"].map(_sc).round(4))
    # 股东减持标签（纯展示）：v1.55起查近60天入选股的股东减持，标在每只票上，不参与选股/排序
    _show_codes = show["代码"].astype(str).str.zfill(6).tolist()
    try:
        _reduce = reduce_labels(_show_codes)
    except Exception:
        _reduce = {}
    # 分层标注：累计比例按近60天算(v1.55从90改60)，但最新一笔公告在近30天内 → 标🔴新减持
    _new_cutoff = (dt.date.today() - dt.timedelta(days=30)).isoformat()
    def _reduce_tag(c):
        d = _reduce.get(str(c).zfill(6))
        if not d or d.get("n", 0) == 0:
            return "—"
        tag = f"减持{d['ratio']:.2f}%·{d['n']}笔"
        # last_date 为 "YYYY-MM-DD"，字典序即时间序；空串"" < cutoff 不会误标
        if str(d.get("last_date", ""))[:10] >= _new_cutoff:
            return f"🔴新减持 {tag}"
        return tag
    show = show.copy()
    show.insert(len(show.columns), "股东减持", [_reduce_tag(c) for c in _show_codes])
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={"盈利概率": st.column_config.NumberColumn(format="%.0f%%"),
                                "佣金(元)": st.column_config.NumberColumn(format="%.2f"),
                                "预计盈利(元)": st.column_config.NumberColumn(format="%+.1f"),
                                "预计亏损(元)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("股东减持标签＝近60天该股减持公告累计占股本比例及笔数；🔴新减持＝最新一笔公告在近30天内"
               "（累计比例按60天算，破折号表示近60天无减持公告或数据未取到），"
               "仅供信息参考、不参与选股与排序；预计盈利/亏损 = 投入 × 该股历史平均每笔盈利率/亏损率（已扣交易成本）；"
               "盈利概率 = 回测胜率经贝叶斯收缩后的校准值（样本少时向50%收敛）。均为历史统计推算，不代表未来，不构成投资建议。")

    # ---- 粗筛候选池导出：全部候选(含技术因子值 + v1.29当日快照字段)导出，供 refine_pool.py 精筛 ----
    with st.expander("📦 导出粗筛候选池（给外部模型精筛用）", expanded=False):
        st.caption(f"本 web 定位=**宽候选池粗筛 + 对照基准**，不是精确预测。这里导出通过回测+排雷的 "
                   f"**{len(ranked)} 只候选**及全部技术因子值，你可拿去外部更专业的模型做二次精筛。"
                   "已纳入当前数据能算出的全部技术因子（动量/反转/波动/MACD/KDJ + P1候选），"
                   "并带上当日快照字段（今开/昨收/隔夜涨跌/振幅/换手率/量比/成交额/市值/PE/PB）供 T+1 精筛。"
                   "注意：快照字段只有当日截面、无历史序列，不能做IC回测，仅供外部模型当前截面打分。")
        # 挑出代码/名称/价格/评分 + v1.29当日快照字段(超短线精筛关键) + 所有因子原始列 + 因子贡献列 + ML分
        # ranked 来自 verify_impl，价格列名为"最新价"（best_plan 才叫"现价"）；两者都兼容。
        _exp_cols = ["代码", "名称", "最新价", "现价", "总分", "行业", "扣费总收益", "胜率", "ML分",
                     "今日涨跌", "今开", "昨收", "最高", "最低", "振幅", "换手率", "量比",
                     "成交额", "成交量", "市盈率", "市净率", "总市值", "流通市值", "隔夜涨跌%"]
        _factor_raw = [f for f in factors.ALL_FACTOR_COLS if f in ranked.columns]
        _factor_contrib = [c for c in ranked.columns if c.startswith("因子_") and c.endswith("_贡献")]
        _export = ranked.reindex(columns=list(dict.fromkeys(
            _exp_cols + _factor_raw + _factor_contrib))).copy()
        for _c in _export.columns:
            if _export[_c].dtype.kind in "fc":
                _export[_c] = _export[_c].round(4)
        _ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
        st.dataframe(_export, width="stretch", hide_index=True)
        st.download_button(
            f"⬇️ 下载候选池 CSV（{len(_export)} 只 × {len(_export.columns)} 列）",
            data=_export.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"粗筛候选池_{_ts}.csv", mime="text/csv")
        st.caption("提示：T+1（今收买、明收卖）请自行按次日收盘价核对真实收益；"
                   "本表总分为5日口径历史统计，超短线仅作对照，不构成买卖建议。")

    st.markdown("**照做就行：**")
    for _, p in plan.iterrows():
        st.markdown(
            f"- **{p['名称']}（{p['代码']}）**：挂单 **{p['现价']}** 元附近买 **{p['买入股数']}** 股；"
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
