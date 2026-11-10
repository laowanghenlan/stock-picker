# -*- coding: utf-8 -*-
"""
每日自动推送脚本（由 GitHub Actions 定时触发，周一至周五北京时间 9:00）
流程：全市场筛选 -> 扣费回测 -> 舆情排雷 -> 本金+收益自动定只数 -> 分配本金 -> QQ邮箱推送
邮件同时输出 HTML（卡片排版）与纯文本（兜底）两个版本
"""
import json
import os
import smtplib
import sys
from email.header import Header
from email.mime.multipart import MIMEMultipart
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
APP_VERSION = "v1.7"

FONT = "-apple-system,'PingFang SC','Microsoft YaHei',Arial,sans-serif"


# ========== 数据收集：跑完整选股流程，产出渲染用的数据结构 ==========
def _collect() -> dict:
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    idx_close, idx_ma60, cold = app.market_temperature_impl()
    d = {
        "date": date_str,
        "tag": "低温期" if cold else "回暖期",
        "cold": cold,
        "idx_close": float(idx_close),
        "idx_ma60": float(idx_ma60),
        "capital": float(CAPITAL),
        "mined": [],          # 排雷清单
        "plan": [],           # 推荐列表（每只一个 dict）
        "n_picks": 0,
        "invest": 0.0,
        "left": 0.0,
        "review": None,       # 复盘摘要
        "note": "",           # 提前结束的单一提示（空仓等）
    }

    cand = app.screen_impl()                   # 初筛（内部经 get_spot_impl 拉全市场行情）
    if len(cand) == 0:
        d["note"] = "今日初筛无符合条件的股票，空仓等待。"
        return d

    cand = cand.nlargest(app.CANDIDATES, "成交额")        # 与web一致的候选策略
    res = app.verify_impl(cand, lambda p, t: None)        # 无界面，进度回调置空
    if len(res) == 0:
        d["note"] = "候选股历史数据不足，今日不建议买入。"
        return d

    mined = res[res["有雷"]]
    if len(mined) > 0:
        d["mined"] = [{"名称": m["名称"], "代码": m["代码"], "标题": m["利空标题"]}
                      for _, m in mined.iterrows()]

    ranked = app.pick_top3(res, min(len(res), 20))   # 评分前20作候选池，只数交给收益优化器
    if len(ranked) == 0:
        d["note"] = "今日无股票通过扣费回测与排雷验证，空仓也是一种操作。"
        return d

    plan, n_picks = app.best_plan(ranked, CAPITAL)
    d["n_picks"] = int(n_picks)
    d["invest"] = float(plan["投入(含佣金)"].sum())
    d["left"] = float(d["capital"] - d["invest"])
    d["plan"] = [{
        "名称": p["名称"], "代码": p["代码"], "现价": float(p["现价"]),
        "股数": int(p["买入股数"]), "投入": float(p["投入(含佣金)"]),
        "盈利概率": float(p["盈利概率"]),
        "止损": float(p["止损价(买入价×0.95)"]),
        "止盈": float(p["止盈参考(买入价×1.10)"]),
    } for _, p in plan.iterrows()]

    # —— 复盘摘要：直接附在邮件里，连网页都不用点开 ——
    try:
        rv = app.review_impl()                       # 复用 app 的复盘逻辑（只读历史推荐，不影响选股）
        if rv is not None and len(rv) > 0:
            done = rv[rv["_done"]]                   # 只统计已出结果的（止盈/止损/到期）
            if len(done) > 0:
                wins = int(done["结果"].isin(["🎯止盈", "🏁到期盈利"]).sum())
                r = {
                    "done": int(len(done)),
                    "winrate": float(wins / len(done) * 100),
                    "pnl": float(rv["_盈亏额"].sum()),
                    "bench": None,
                    "excess": None,
                    "rows": [],
                    "conclusion": "",
                }
                if "同期沪深300%" in done.columns:
                    d2 = done.dropna(subset=["同期沪深300%"])
                    if len(d2) > 0:                # 基准对比：跑赢大盘才算真赚
                        r["bench"] = float(d2["同期沪深300%"].mean())
                        r["excess"] = float((d2["盈亏%"] - d2["同期沪深300%"]).mean())
                for _, r2 in rv.head(5).iterrows():  # 最多列最近5条明细
                    r["rows"].append({
                        "日期": str(r2["日期"]), "名称": r2["名称"],
                        "结果": r2["结果"], "盈亏%": float(r2["盈亏%"]),
                    })
                r["conclusion"] = app.review_conclusion(rv)
                d["review"] = r
    except Exception as e:                           # 复盘失败绝不影响正常推荐和发邮件
        print("复盘摘要生成失败(不影响推荐):", e)

    # 留档供周末复盘
    rec = plan[["代码", "名称", "现价", "买入股数"]].copy()
    rec.insert(0, "日期", date_str)
    rec["大盘标签"] = d["tag"]
    rec.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(),
               index=False, encoding="utf-8-sig")
    return d


# ========== 纯文本版（不支持 HTML 的客户端兜底） ==========
def _render_text(d: dict) -> str:
    lines = [f"大盘标签 {d['tag']}：沪深300 收盘 {d['idx_close']:.0f}，"
             f"60日均线 {d['idx_ma60']:.0f}", ""]

    if d["cold"]:
        lines.append("大盘处于低温期（系统性下跌段），今日建议空仓，以下仅为数据参考，不建议买入。")
        lines.append("")

    if d["note"]:
        lines.append(d["note"])
        return "\n".join(lines)

    if d["mined"]:
        lines.append(f"排雷：{len(d['mined'])} 只候选股近7天有利空新闻，已排除：")
        for m in d["mined"]:
            lines.append(f"  - {m['名称']}（{m['代码']}）：{m['标题']}")
        lines.append("")

    lines.append(f"今日推荐：本金 {d['capital']:.0f} 元，按本金+预期收益自动算出 {d['n_picks']} 只，"
                 f"计划投入 {d['invest']:.0f} 元，剩余 {d['left']:.0f} 元")
    lines.append("")
    for p in d["plan"]:
        lines.append(
            f"{p['名称']}（{p['代码']}）现价 {p['现价']:.2f} 元 | "
            f"买入 {p['股数']} 股 约 {p['投入']:.0f} 元 | "
            f"盈利概率 {p['盈利概率']:.0f}% | "
            f"止损=买入价×0.95（约 {p['止损']:.2f} 元） | "
            f"止盈=买入价×1.10（约 {p['止盈']:.2f} 元）"
        )

    r = d["review"]
    if r:
        lines += ["", "【复盘】最近推荐表现（自动统计，无需打开网页）",
                  f"已出结果 {r['done']} 笔 | 胜率 {r['winrate']:.0f}% | "
                  f"累计盈亏 {r['pnl']:+.0f} 元"]
        if r["bench"] is not None:
            lines.append(f"同期沪深300平均 {r['bench']:+.1f}% | 平均超额 {r['excess']:+.1f}%")
        for x in r["rows"]:
            lines.append(f"  {x['日期']} {x['名称']} {x['结果']} {x['盈亏%']:+.1f}%")
        lines.append("结论：" + r["conclusion"])

    lines += ["", "纪律：跌破止损价无条件卖出；持有超20个交易日不涨也卖。",
              "",
              "📊 网页版（点开自动看复盘，无需任何操作）：",
              "https://stock-picker-mi5bntxj6x4fzrntgtzvko.streamlit.app/",
              f"💰 改本金（当前 {d['capital']:.0f} 元；点链接 → 铅笔图标 → 改数字 → 绿色Commit，下次邮件生效）：",
              "https://github.com/laowanghenlan/stock-picker/blob/master/推荐设置.json",
              f"🔖 版本 {APP_VERSION}  ·  仓库：https://github.com/laowanghenlan/stock-picker",
              "（本邮件为量化程序自动生成，仅为历史数据推演，不构成投资建议）"]
    return "\n".join(lines)


# ========== HTML 版（卡片排版，全内联样式，兼容 QQ 邮箱） ==========
def _render_html(d: dict) -> str:
    # 回暖期蓝色 / 低温期橙色
    head_bg = ("linear-gradient(135deg,#d9822b,#b8690f)" if d["cold"]
               else "linear-gradient(135deg,#1a73e8,#0b57d0)")
    head_sub = ("系统性下跌段，建议空仓观察" if d["cold"] else "趋势向上，可择优参与")

    h = []
    h.append(f"<div style=\"background:#eef1f5;padding:26px 12px;font-family:{FONT};\">")
    h.append("<div style=\"max-width:640px;margin:0 auto;background:#ffffff;border-radius:16px;"
             "overflow:hidden;border:1px solid #e3e8ef;\">")

    # —— 头部 ——
    h.append(f"<div style=\"background:{head_bg};padding:26px 26px 22px;\">")
    h.append(f"<div style=\"color:#ffffff;font-size:12px;letter-spacing:2px;opacity:.88;\">"
             f"每日选股 · {d['date']}</div>")
    h.append(f"<div style=\"color:#ffffff;font-size:26px;font-weight:700;margin-top:8px;"
             f"letter-spacing:1px;\">{d['tag']}</div>")
    h.append(f"<div style=\"color:#ffffff;font-size:13px;opacity:.92;margin-top:8px;\">"
             f"沪深300 收盘 {d['idx_close']:.0f} · 60日均线 {d['idx_ma60']:.0f}</div>")
    h.append(f"<div style=\"color:#ffffff;font-size:12px;opacity:.8;margin-top:4px;\">{head_sub}</div>")
    h.append("</div>")

    h.append("<div style=\"padding:22px 26px 6px;\">")

    # —— 低温期警示 ——
    if d["cold"]:
        h.append("<div style=\"background:#fff5e6;border-left:4px solid #e8a33d;padding:12px 14px;"
                 "border-radius:8px;font-size:13px;color:#8a5a00;line-height:1.7;margin-bottom:18px;\">"
                 "大盘处于低温期（系统性下跌段），今日建议空仓，以下仅为数据参考，不建议买入。</div>")

    # —— 提前结束（空仓等） ——
    if d["note"]:
        h.append("<div style=\"background:#f4f7fb;border:1px solid #dbe5f2;padding:20px;"
                 "border-radius:12px;font-size:14px;color:#33475b;text-align:center;line-height:1.7;\">"
                 f"{d['note']}</div>")
        h.append("</div>")
        h.append(_html_footer(d))
        return _html_wrap("".join(h))

    # —— 排雷 ——
    if d["mined"]:
        h.append("<div style=\"background:#fdf3f2;border:1px solid #ffd8d3;border-radius:12px;"
                 "padding:14px 16px;margin-bottom:18px;\">")
        h.append(f"<div style=\"font-size:13px;font-weight:700;color:#b3261e;margin-bottom:8px;\">"
                 f"⚠ 排雷：{len(d['mined'])} 只候选股近7天有利空，已排除</div>")
        for m in d["mined"]:
            h.append(f"<div style=\"font-size:13px;color:#5f6b7a;line-height:1.8;\">"
                     f"· {m['名称']}（{m['代码']}）{m['标题']}</div>")
        h.append("</div>")

    # —— 概要统计 ——
    h.append("<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
             "style=\"background:#f7f9fc;border-radius:12px;margin-bottom:20px;\"><tr>")
    for label, val in [("本金", f"{d['capital']:.0f}"),
                       ("推荐", f"{d['n_picks']} 只"),
                       ("投入", f"{d['invest']:.0f}"),
                       ("剩余", f"{d['left']:.0f}")]:
        h.append(f"<td align=\"center\" style=\"padding:14px 4px;\">"
                 f"<div style=\"font-size:11.5px;color:#8b949e;margin-bottom:4px;\">{label}</div>"
                 f"<div style=\"font-size:18px;font-weight:700;color:#1f2328;\">{val}</div></td>")
    h.append("</tr></table>")

    # —— 推荐卡片 ——
    for p in d["plan"]:
        h.append("<div style=\"border:1px solid #e6e9ee;border-radius:12px;padding:16px 18px;"
                 "margin-bottom:12px;\">")
        h.append("<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\"><tr>"
                 f"<td style=\"font-size:16px;font-weight:700;color:#1f2328;\">{p['名称']}"
                 f"<span style=\"font-size:12px;color:#8b949e;font-weight:400;\"> {p['代码']}</span></td>"
                 f"<td align=\"right\" style=\"font-size:19px;font-weight:700;color:#1a73e8;\">"
                 f"{p['现价']:.2f}</td></tr></table>")
        h.append("<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
                 "style=\"margin-top:12px;font-size:13px;color:#57606a;\">")
        h.append(f"<tr><td style=\"padding:3px 0;\">买入 <b style=\"color:#1f2328;\">{p['股数']} 股</b></td>"
                 f"<td style=\"padding:3px 0;\">投入 <b style=\"color:#1f2328;\">{p['投入']:.0f} 元</b></td>"
                 f"<td style=\"padding:3px 0;\">盈利概率 <b style=\"color:#1a7f37;\">"
                 f"{p['盈利概率']:.0f}%</b></td></tr>")
        h.append(f"<tr><td style=\"padding:3px 0;color:#d93025;\">止损 {p['止损']:.2f}</td>"
                 f"<td colspan=\"2\" style=\"padding:3px 0;color:#1a7f37;\">止盈 {p['止盈']:.2f}</td></tr>")
        h.append("</table></div>")

    # —— 复盘 ——
    r = d["review"]
    if r:
        h.append("<div style=\"border-top:1px dashed #e6e9ee;margin-top:10px;padding-top:18px;\">")
        h.append("<div style=\"font-size:14px;font-weight:700;color:#1f2328;margin-bottom:12px;\">"
                 "📊 复盘 · 最近推荐表现</div>")
        h.append("<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
                 "style=\"margin-bottom:10px;\"><tr>")
        for label, val, color in [("已出结果", f"{r['done']} 笔", "#1f2328"),
                                  ("胜率", f"{r['winrate']:.0f}%", "#1a7f37"),
                                  ("累计盈亏", f"{r['pnl']:+.0f} 元",
                                   "#1a7f37" if r["pnl"] >= 0 else "#d93025")]:
            h.append(f"<td align=\"center\"><div style=\"font-size:11.5px;color:#8b949e;\">{label}</div>"
                     f"<div style=\"font-size:16px;font-weight:700;color:{color};margin-top:3px;\">"
                     f"{val}</div></td>")
        h.append("</tr></table>")
        if r["bench"] is not None:
            h.append(f"<div style=\"font-size:12.5px;color:#8b949e;text-align:center;margin-bottom:12px;\">"
                     f"同期沪深300平均 {r['bench']:+.1f}% · 平均超额 {r['excess']:+.1f}%</div>")
        if r["rows"]:
            h.append("<table role=\"presentation\" width=\"100%\" cellpadding=\"0\" cellspacing=\"0\" "
                     "style=\"font-size:12.5px;color:#57606a;border-collapse:collapse;\">")
            h.append("<tr style=\"color:#9aa4b1;font-size:11.5px;\">"
                     "<td style=\"padding:6px 0;border-bottom:1px solid #eef1f4;\">日期</td>"
                     "<td style=\"border-bottom:1px solid #eef1f4;\">名称</td>"
                     "<td align=\"right\" style=\"border-bottom:1px solid #eef1f4;\">结果</td>"
                     "<td align=\"right\" style=\"border-bottom:1px solid #eef1f4;\">盈亏</td></tr>")
            for x in r["rows"]:
                c = "#1a7f37" if x["盈亏%"] >= 0 else "#d93025"
                h.append(f"<tr><td style=\"padding:7px 0;border-bottom:1px solid #f4f6f9;\">{x['日期']}</td>"
                         f"<td style=\"border-bottom:1px solid #f4f6f9;\">{x['名称']}</td>"
                         f"<td align=\"right\" style=\"border-bottom:1px solid #f4f6f9;\">{x['结果']}</td>"
                         f"<td align=\"right\" style=\"border-bottom:1px solid #f4f6f9;color:{c};\">"
                         f"{x['盈亏%']:+.1f}%</td></tr>")
            h.append("</table>")
        if r["conclusion"]:
            h.append(f"<div style=\"font-size:12.5px;color:#57606a;margin-top:10px;line-height:1.7;\">"
                     f"结论：{r['conclusion']}</div>")
        h.append("</div>")

    # —— 纪律 ——
    h.append("<div style=\"background:#f7f9fb;border-radius:10px;padding:14px 16px;margin:20px 0 16px;"
             "font-size:12.5px;color:#57606a;line-height:1.8;\">"
             "<b style=\"color:#1f2328;\">交易纪律</b><br>"
             "跌破止损价无条件卖出；持有超20个交易日不涨也卖。</div>")

    # —— 按钮 ——
    h.append("<a href=\"https://stock-picker-mi5bntxj6x4fzrntgtzvko.streamlit.app/\" "
             "style=\"display:block;background:#1a73e8;color:#ffffff;text-align:center;padding:13px;"
             "border-radius:10px;font-size:14px;font-weight:700;text-decoration:none;\">"
             "打开网页版看完整复盘</a>")
    h.append("<a href=\"https://github.com/laowanghenlan/stock-picker/blob/master/推荐设置.json\" "
             "style=\"display:block;background:#ffffff;color:#1a73e8;text-align:center;padding:12px;"
             "border-radius:10px;font-size:13px;font-weight:600;text-decoration:none;"
             "border:1px solid #c9dcf8;margin-top:10px;\">"
             f"修改本金（当前 {d['capital']:.0f} 元）</a>")

    h.append("</div>")
    h.append(_html_footer(d))
    return _html_wrap("".join(h))


def _html_footer(d: dict) -> str:
    return ("<div style=\"padding:18px 26px 24px;border-top:1px solid #eef1f4;\">"
            f"<div style=\"font-size:11.5px;color:#9aa4b1;line-height:1.9;font-family:{FONT};\">"
            f"版本 {APP_VERSION} · "
            "<a href=\"https://github.com/laowanghenlan/stock-picker\" "
            "style=\"color:#9aa4b1;text-decoration:none;\">laowanghenlan/stock-picker</a><br>"
            "本邮件为量化程序自动生成，仅为历史数据推演，不构成投资建议"
            "</div></div>")


def _html_wrap(inner: str) -> str:
    return ("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "</head><body style=\"margin:0;padding:0;background:#eef1f5;\">"
            + inner + "</body></html>")


def build_email() -> tuple:
    """跑一遍完整选股流程，返回 (纯文本正文, HTML正文)"""
    d = _collect()
    return _render_text(d), _render_html(d)


def send(text: str, html: str):
    """通过QQ邮箱SMTP发送（授权码从GitHub Secrets注入，不落盘）"""
    _, _, cold = app.market_temperature_impl()
    tag = "低温期" if cold else "回暖期"
    msg = MIMEMultipart("alternative")
    msg["From"] = TO_ADDR
    msg["To"] = TO_ADDR
    msg["Subject"] = Header(f"【{tag}】每日选股 {pd.Timestamp.now().strftime('%m-%d')}", "utf-8")
    # 顺序不能反：先纯文本后 HTML，客户端会优先显示最后一段
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.qq.com", 465) as s:
        s.login(TO_ADDR, AUTH_CODE)
        s.sendmail(TO_ADDR, [TO_ADDR], msg.as_string())


if __name__ == "__main__":
    try:
        send(*build_email())
        print("推送成功")
    except Exception as e:
        print(f"推送失败: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
