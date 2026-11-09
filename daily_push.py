# -*- coding: utf-8 -*-
"""
每日自动推送脚本（由 GitHub Actions 定时触发，周一至周五北京时间 6:30）
流程：全市场筛选 -> 扣费回测 -> 舆情排雷 -> 取前3只 -> 分配本金 -> QQ邮箱推送
"""
import os
import smtplib
import sys
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

import app  # 复用 web 应用里的全部选股逻辑（裸运行模式下 UI 部分不生效）

CAPITAL = float(os.environ.get("CAPITAL", "2200"))     # 本金，可用环境变量覆盖
TO_ADDR = os.environ["QQ_EMAIL"]                        # 收件邮箱（与发件邮箱相同）
AUTH_CODE = os.environ["QQ_AUTH_CODE"]                  # QQ邮箱SMTP授权码（存于GitHub Secrets）
LOG_FILE = Path(__file__).parent / "推送记录.csv"       # 推送留档，供周末复盘用


def build_email() -> str:
    """跑一遍完整选股流程，生成邮件正文"""
    idx_close, idx_ma60, cold = app.market_temperature()
    tag = "【低温期】" if cold else "【回暖期】"
    lines = [f"大盘标签 {tag}：沪深300 收盘 {idx_close:.0f}，60日均线 {idx_ma60:.0f}", ""]

    if cold:
        lines.append("大盘处于低温期（系统性下跌段），今日建议空仓，以下仅为数据参考，不建议买入。")
        lines.append("")

    spot = app.get_spot()
    cand = app.screen(spot)
    if len(cand) == 0:
        lines.append("今日初筛无符合条件的股票，空仓等待。")
        return "\n".join(lines)

    cand = cand.nlargest(app.TOP_N, "成交额")             # 与web一致的候选策略
    res = app.verify(cand)
    if len(res) == 0:
        lines.append("候选股历史数据不足，今日不建议买入。")
        return "\n".join(lines)

    mined = res[res["有雷"]]
    if len(mined) > 0:
        lines.append(f"排雷：{len(mined)} 只候选股近7天有利空新闻，已排除：")
        for _, m in mined.iterrows():
            lines.append(f"  - {m['名称']}（{m['代码']}）：{m['利空标题']}")
        lines.append("")

    picks = app.pick_top3(res)
    if len(picks) == 0:
        lines.append("今日无股票通过扣费回测与排雷验证，空仓也是一种操作。")
        return "\n".join(lines)

    plan, left = app.allocate(picks, CAPITAL)
    lines.append(f"今日推荐：本金 {CAPITAL:.0f} 元，计划投入 {plan['总花费'].sum():.0f} 元，剩余 {left:.0f} 元")
    lines.append("")
    for _, p in plan.iterrows():
        lines.append(
            f"{p['名称']}（{p['代码']}）现价 {p['现价']:.2f} 元 | "
            f"买入 {int(p['股数'])} 股 约 {p['总花费']:.0f} 元 | "
            f"盈利概率 {p['盈利概率'] * 100:.0f}% | "
            f"止损 {p['止损价']:.2f} 元 | 止盈参考 {p['止盈参考']:.2f} 元"
        )
    lines += ["", "纪律：跌破止损价无条件卖出；持有超20个交易日不涨也卖。",
              "（本邮件为量化程序自动生成，仅为历史数据推演，不构成投资建议）"]
    # 留档供周末复盘
    rec = plan[["代码", "名称", "现价", "股数"]].copy()
    rec.insert(0, "日期", pd.Timestamp.now().strftime("%Y-%m-%d"))
    rec["大盘标签"] = tag
    rec.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(),
               index=False, encoding="utf-8-sig")
    return "\n".join(lines)


def send(body: str):
    """通过QQ邮箱SMTP发送（授权码从GitHub Secrets注入，不落盘）"""
    _, _, cold = app.market_temperature()
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
