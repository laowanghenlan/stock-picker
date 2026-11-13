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

from data import (get_history, get_history_impl, get_spot, get_spot_impl,
                  industry_map_impl, refresh_spot, spot_cache_time, spot_fresh_enough,
                  spot_recently_failed, get_history_tx, cninfo_risk_impl, tushare_daily_basic)
from strategy import (at_limit_down, at_limit_up, backtest, calibrate_win_rate,
                      generate_signals)

import factors  # L1 多因子层：动量/波动率/换手/估值 + 行业中性 + 20日IC均值配权
# 选股引擎统一放在 screen_service（纯 Python，无 Streamlit 依赖）；
# 这里显式导入，保证 app.xxx 在测试/脚本里沿用原名字
from screen_service import (CANDIDATES, HISTORY_DAYS, IC_GATE, MIN_PRICE, ML_WEIGHT,
                            REVIEW_CACHE_KEY, REVIEW_N, SETTINGS_FILE, STOP_LOSS_PCT,
                            TAKE_PROFIT_PCT, _CORR_THRESHOLD, _USE_CORR_FILTER,
                            _confidence, _correlation_filter, _kelly_position, _num,
                            _run_with_timeout, allocate, auto_pick_count, best_plan,
                            check_holdings, lhb_codes_impl, load_holdings, load_settings,
                            market_temperature_impl, news_risk_impl, pick_top3,
                            review_conclusion, review_impl, screen_impl, verify_impl)

# 缓存包装：web UI 一律调 news_risk；GitHub Actions 等裸运行环境请调 news_risk_impl
news_risk = st.cache_data(ttl=3600, show_spinner=False)(news_risk_impl)

# 缓存包装：web UI 一律调 market_temperature；裸运行环境请调 market_temperature_impl
market_temperature = st.cache_data(ttl=3600, show_spinner="正在检查大盘温度...")(market_temperature_impl)

# 缓存包装：web UI 一律调 lhb_codes；GitHub Actions 等裸运行环境请调 lhb_codes_impl
lhb_codes = st.cache_data(ttl=43200, show_spinner=False)(lhb_codes_impl)

# 缓存包装：web UI 一律调 review_performance(REVIEW_CACHE_KEY)；30分钟内重复点秒开
review_performance = st.cache_data(ttl=1800, show_spinner="正在自动复盘最近推荐表现...")(review_impl)


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


# web UI 沿用原名 screen；GitHub Actions 等裸运行环境请调 screen_impl
screen = screen_impl

# web UI 沿用原名 verify；GitHub Actions 等裸运行环境请调 verify_impl
verify = verify_impl

APP_VERSION = "v1.22.2"     # v1.22.2 邮件去掉GitHub改本金入口，网页输入本金自动保存并同步邮件

st.set_page_config(page_title="每日选股", page_icon="📈", layout="centered")
st.title("📈 每日选股")
st.caption(APP_VERSION)

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

# 股价上限：保证各只买100股也买得起；本金越大上限越宽，最高50元
MAX_PRICE = max(MIN_PRICE + 0.5, min(50.0, capital / (n_picks * 100) * 0.95))

# ---------------- 历史推荐复盘（按需加载，避免云端启动时网络 hang 导致白屏） ----------------
with st.expander(f"📊 历史推荐复盘（最近{REVIEW_N}批）", expanded=False):
    if not st.button("🔄 加载复盘数据", key="load_review"):
        st.caption("点击上方按钮加载复盘（需联网拉历史行情，约10~20秒）。"
                   "云端启动时不自动执行，避免数据源 hang 住拖垮整个页面。")
    else:
        rv = review_performance(REVIEW_CACHE_KEY)
        if len(rv) == 0:
            st.caption("暂无历史推荐，有推荐后会自动复盘。")
        else:
            done = rv[rv["_done"]]
            holding = rv[~rv["_done"]]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("推荐总数", f"{len(rv)} 条 / {rv['_batch'].nunique()} 批")
            if len(done) > 0:
                wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
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
            _cols = ["名称", "推荐价", "现价", "股数", "盈亏%", "盈亏额", "同期沪深300%", "结果"]
            _col_fmt = {"盈亏%": st.column_config.NumberColumn(format="%+.2f%%"),
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
        if _hf and len(_hf) >= len(cands):
            st.warning("行情数据源暂时不可用（东财限流、新浪未返回有效数据），本次无法完成验证。"
                       "这不是选股结论，请稍后重试，不建议据此操作。")
        else:
            st.warning("候选股均未通过验证（历史不足/回测不合格），今天不建议买入。")
        st.stop()
    _nf = res.attrs.get("news_failed", [])
    if _nf:
        st.caption(f"⚠️ 舆情接口异常，{len(_nf)} 只候选按“无雷”降级处理，排雷结果仅供参考："
                   f"{', '.join(_nf[:10])}{' 等' if len(_nf) > 10 else ''}")
    # 舆情排雷通报：哪些因近7天利空新闻被排除
    mined = res[res["有雷"]]
    if len(mined) > 0:
        st.error(f"⚠️ 排雷：{len(mined)} 只候选股近7天有利空新闻，已自动排除——")
        for _, m in mined.iterrows():
            st.markdown(f"- **{m['名称']}（{m['代码']}）**：{m['利空标题']}")
    # IC 门控：训练期组合 IC < IC_GATE → 信号弱到噪声级，今日空仓（与邮件推送同一口径）
    _ic_gate = float(res.attrs.get("ic_train_composite") or 0.0)
    if _ic_gate < IC_GATE:
        st.error(f"🚫 因子近期信号偏弱（训练期组合IC={_ic_gate:+.4f} < {IC_GATE}），"
                 "按规则今日空仓不选股。")
        st.caption("这是风控开关：信号弱到噪声级时硬推票，期望收益不如空仓。每日邮件同一口径，今天也会跳过。")
        st.stop()
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
               "盈利概率 = 回测胜率经贝叶斯收缩后的校准值（样本少时向50%收敛）。均为历史统计推算，不代表未来，不构成投资建议。")

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
