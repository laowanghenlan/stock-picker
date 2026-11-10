# -*- coding: utf-8 -*-
"""
自动进化引擎（v1.8 全开+置信度折扣版）。

v1.8 变更：
  - 所有进化开关（相关性过滤/凯利/因子发现/AB实验）从第一天就启用。
  - evolution.py 不再翻转开关，只负责：
      1. 统计实笔数
      2. 计算各功能的置信度 = min(1.0, n_done / 置信度阈值)
      3. 打印状态报告（进 Actions 日志）
  - 置信度折扣在 app.py 的仓位/因子逻辑里实时应用，不靠开关。

设计原则：
  - 只改 JSON 配置，不改源码。
  - 所有功能全开，效果随样本量线性增长。
"""
from __future__ import annotations
import json, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
CFG_PATH = HERE / "strategy_config.json"
PUSH_PATH = HERE / "推送记录.csv"
FACTOR_HIST = HERE / "因子历史IC.csv"


def _actual_record_count() -> int:
    """实笔数 = 推送记录.csv 的行数（每次邮件推送一行）。"""
    if not PUSH_PATH.exists():
        return 0
    try:
        return sum(1 for _ in open(PUSH_PATH, encoding="utf-8", errors="ignore")) - 1
    except Exception:
        return 0


def _win_rate_approx() -> float:
    """近似用最近30天因子IC均值的正负判断。正IC多=市场风格对=胜率应高。"""
    if not FACTOR_HIST.exists():
        return 0.5
    try:
        df = pd.read_csv(FACTOR_HIST)
        ic_cols = [c for c in df.columns if c != "日期"]
        if not ic_cols:
            return 0.5
        last = df.iloc[-1][ic_cols]
        pos = (last.abs() > 0.03).sum()
        total = len(ic_cols)
        return pos / total if total else 0.5
    except Exception:
        return 0.5


def load_cfg() -> dict:
    return json.load(open(CFG_PATH, encoding="utf-8"))


def _confidence(n_done: int, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return max(0.0, min(1.0, n_done / threshold))


def evolve() -> list[str]:
    """执行一次进化检查，返回状态报告。v1.8 不翻转任何开关。"""
    log: list[str] = []
    cfg = load_cfg()
    n = _actual_record_count()
    wr = _win_rate_approx()

    pos = cfg.get("仓位管理", {})
    fact = cfg.get("因子", {})
    obs = cfg.get("观测", {})

    corr_on = pos.get("相关性过滤", {}).get("启用", False)
    corr_th = int(pos.get("相关性过滤", {}).get("置信度阈值", 30))
    kelly_on = pos.get("凯利公式", {}).get("启用", False)
    kelly_th = int(pos.get("凯利公式", {}).get("置信度阈值", 30))
    mv_on = pos.get("均值方差优化", {}).get("启用", False)
    mv_th = int(pos.get("均值方差优化", {}).get("置信度阈值", 30))
    fdisc_on = fact.get("自动发现", {}).get("启用", False)
    fdisc_th = int(fact.get("自动发现", {}).get("置信度阈值", 60))
    ab_on = obs.get("自动AB实验", {}).get("启用", False)
    ab_th = int(obs.get("自动AB实验", {}).get("置信度阈值", 100))

    header = (f"【进化检查】实盘记录 {n} 笔 · 近似胜率 {wr:.0%} · "
              f"v1.8 全开模式：")
    log.append(header)
    log.append(f"  相关性过滤: {'开' if corr_on else '关'} 置信度 {_confidence(n, corr_th):.0%} (满{n}≥{corr_th})")
    log.append(f"  凯利公式:   {'开' if kelly_on else '关'} 置信度 {_confidence(n, kelly_th):.0%} (满{n}≥{kelly_th})")
    log.append(f"  均值方差:   {'开' if mv_on else '关'} 置信度 {_confidence(n, mv_th):.0%} (满{n}≥{mv_th})")
    log.append(f"  因子发现:   {'开' if fdisc_on else '关'} 置信度 {_confidence(n, fdisc_th):.0%} (满{n}≥{fdisc_th})")
    log.append(f"  AB实验:     {'开' if ab_on else '关'} 置信度 {_confidence(n, ab_th):.0%} (满{n}≥{ab_th})")
    return log


if __name__ == "__main__":
    changes = evolve()
    for c in changes:
        print(c)
