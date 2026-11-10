# -*- coding: utf-8 -*-
"""
每日自动推送脚本（由 GitHub Actions 定时触发，周一至周五北京时间 9:00）
流程：全市场筛选 -> 扣费回测 -> 舆情排雷 -> 本金+收益自动定只数 -> 分配本金 -> QQ邮箱推送
"""
import json
import os
import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

import app  # 复用 web 应用的选股逻辑；裸运行无 ScriptRunContext，一律调 *_impl 版本（绕开 st.cache_data）


def _load_capital() -> float:
    """本金读取优先级：仓库 推荐设置.json 的 capital > 环境变量 CAPITAL > 2200。
    GitHub Actions checkout 出来的代码自带 推荐设置.json，直接读本地文件即可；
    网页保存本金后，第二天邮件自动按新本金计算（与 web 端共用同一份设置）。"""
    try:
        settings_file = Path(__file__).parent / "推荐设置.json"   # 字段含义：capital=本金（元）
        return float(json.loads(settings_file.read_text(encoding="utf-8"))["capital"])
    except Exception:
        return float(os.environ.get("CAPITAL", "2200"))


CAPITAL = _load_capital()                             # 本金（元），web 与邮件统一来源
TO_ADDR = os.environ["QQ_EMAIL"]                        # 收件邮箱（与发件邮箱相同）
AUTH_CODE = os.environ["QQ_AUTH_CODE"]                  # QQ邮箱SMTP授权码（存于GitHub Secrets）
LOG_FILE = Path(__file__).parent / "推送记录.csv"       # 推送留档，供周末复盘用


# ---------- 版本号：与 app.py 的 APP_VERSION 保持同步 ----------
APP_VERSION = "v1.2"


def build_email() -> str:
    """跑一遍完整选股流程，生成邮件正文"""
    idx_close, idx_ma60, cold = app.market_temperature_impl()
    tag = "【低温期】" if cold else "【回暖期】"
    lines = [f"大盘标签 {tag}：沪深300 收盘 {idx_close:.0f}，60日均线 {idx_ma60:.0f}", ""]

    if cold:
        lines.append("大盘处于低温期（系统性下跌段），今日建议空仓，以下仅为数据参考，不建议买入。")
        lines.append("")

    cand = app.screen_impl()                   # 初筛（内部经 get_spot_impl 拉全市场行情）
    if len(cand) == 0:
        lines.append("今日初筛无符合条件的股票，空仓等待。")
        return "\n".join(lines)

    cand = cand.nlargest(app.CANDIDATES, "成交额")        # 与web一致的候选策略
    res = app.verify_impl(cand, lambda p, t: None)        # 无界面，进度回调置空
    if len(res) == 0:
        lines.append("候选股历史数据不足，今日不建议买入。")
        return "\n".join(lines)

    mined = res[res["有雷"]]
    if len(mined) > 0:
        lines.append(f"排雷：{len(mined)} 只候选股近7天有利空新闻，已排除：")
        for _, m in mined.iterrows():
            lines.append(f"  - {m['名称']}（{m['代码']}）：{m['利空标题']}")
        lines.append("")

    ranked = app.pick_top3(res, min(len(res), 20))   # 评分前20作候选池，只数交给收益优化器
    if len(ranked) == 0:
        lines.append("今日无股票通过扣费回测与排雷验证，空仓也是一种操作。")
        return "\n".join(lines)

    plan, n_picks = app.best_plan(ranked, CAPITAL)
    lines.append(f"今日推荐：本金 {CAPITAL:.0f} 元，按本金+预期收益自动算出 {n_picks} 只，"
                 f"计划投入 {plan['投入(含佣金)'].sum():.0f} 元，"
                 f"剩余 {CAPITAL - plan['投入(含佣金)'].sum():.0f} 元")
    lines.append("")
    for _, p in plan.iterrows():
        lines.append(
            f"{p['名称']}（{p['代码']}）现价 {p['现价']:.2f} 元 | "
            f"买入 {int(p['买入股数'])} 股 约 {p['投入(含佣金)']:.0f} 元 | "
            f"盈利概率 {p['盈利概率']:.0f}% | "
            f"止损=买入价×0.95（约 {p['止损价(买入价×0.95)']:.2f} 元） | "
            f"止盈=买入价×1.10（约 {p['止盈参考(买入价×1.10)']:.2f} 元）"
        )
    # —— 复盘摘要：直接附在邮件里，连网页都不用点开 ——
    try:
        rv = app.review_impl()                       # 复用 app 的复盘逻辑（只读历史推荐，不影响选股）
        if rv is not None and len(rv) > 0:
            done = rv[rv["_done"]]                   # 只统计已出结果的（止盈/止损/到期）
            if len(done) > 0:
                wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
                total_pnl = rv["_盈亏额"].sum()
                lines += ["", "【复盘】最近推荐表现（自动统计，无需打开网页）",
                          f"已出结果 {len(done)} 笔 | 胜率 {wins / len(done) * 100:.0f}% | "
                          f"累计盈亏 {total_pnl:+.0f} 元"]
                if "同期沪深300%" in done.columns:
                    d2 = done.dropna(subset=["同期沪深300%"])
                    if len(d2) > 0:                # 基准对比：跑赢大盘才算真赚
                        bm = d2["同期沪深300%"].mean()
                        ex = (d2["盈亏%"] - d2["同期沪深300%"]).mean()
                        lines.append(f"同期沪深300平均 {bm:+.1f}% | 平均超额 {ex:+.1f}%")
                for _, r2 in rv.head(5).iterrows():  # 最多列最近5条明细
                    lines.append(f"  {r2['日期']} {r2['名称']} {r2['结果']} {r2['盈亏%']:+.1f}%")
                lines.append("结论：" + app.review_conclusion(rv))
    except Exception as e:                           # 复盘失败绝不影响正常推荐和发邮件
        print("复盘摘要生成失败(不影响推荐):", e)

    lines += ["", "纪律：跌破止损价无条件卖出；持有超20个交易日不涨也卖。",
              "",  # 空行分隔
              "📊 网页版（点开自动看复盘，无需任何操作）：",
              "https://stock-picker-mi5bntxj6x4fzrntgtzvko.streamlit.app/",
              f"💰 改本金（当前 {CAPITAL:.0f} 元；点链接 → 铅笔图标 → 改数字 → 绿色Commit，下次邮件生效）：",
              "https://github.com/laowanghenlan/stock-picker/blob/master/推荐设置.json",
              f"🔖 版本 {APP_VERSION}  ·  仓库：https://github.com/laowanghenlan/stock-picker",
              "（本邮件为量化程序自动生成，仅为历史数据推演，不构成投资建议）"]
    # 留档供周末复盘
    rec = plan[["代码", "名称", "现价", "买入股数"]].copy()
    rec.insert(0, "日期", pd.Timestamp.now().strftime("%Y-%m-%d"))
    rec["大盘标签"] = tag
    rec.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(),
               index=False, encoding="utf-8-sig")
    return "\n".join(lines)


def send(body: str):
    """通过QQ邮箱SMTP发送（授权码从GitHub Secrets注入，不落盘）"""
    _, _, cold = app.market_temperature_impl()
    tag = "低温期" if cold else "回暖期"
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = TO_ADDR
    msg["To"] = TO_ADDR
    msg["Subject"] = Header(f"【{tag}】每日选股 {pd.Timestamp.now().strftime('%m-%d')}", "utf-8")
    with smtplib.SMTP_SSL("smtp.qq.com", 465) as s:
        s.login(TO_ADDR, AUTH_CODE)
        s.sendmail(TO_ADDR, [TO_ADDR], msg.as_string())


if __name__ == "__main__":
    try:
        send(build_email())
        print("推送成功")
    except Exception as e:
        print(f"推送失败: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
