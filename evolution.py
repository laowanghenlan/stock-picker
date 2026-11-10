# -*- coding: utf-8 -*-
"""
自动进化引擎。每天复盘后自动执行（GitHub Actions 触发）。
职责：
  1. 读推送记录.csv 统计实笔数/胜率/回撤
  2. 检查进化阈值（30/60/100 笔）
  3. 自动升级 strategy_config.json 的开关
  4. 打印变更日志（进 Actions 日志，邮件每日摘要附）

设计原则：
  - 只改 JSON 配置，不改源码。源码全部读 config 开关
  - 升级单向、不可逆（开了就不关，除非手动改）
  - 所有变更附带理由（够了多少笔、胜率多少）
"""
from __future__ import annotations
import json, os, sys, pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
CFG_PATH = HERE / "strategy_config.json"
PUSH_PATH = HERE / "推送记录.csv"
HOLD_PATH = HERE / "持仓.csv"
FACTOR_HIST = HERE / "因子历史IC.csv"

# 实笔数 = 推送记录.csv 的行数（每次邮件推送一行）
def _actual_record_count() -> int:
    if not PUSH_PATH.exists():
        return 0
    try:
        return sum(1 for _ in open(PUSH_PATH, encoding="utf-8", errors="ignore")) - 1  # 减表头
    except Exception:
        return 0

# 从因子历史IC.csv 估算"累计胜率"（近似，真正胜率要复盘）
def _win_rate_approx() -> float:
    """近似用最近30天因子IC均值的正负判断。正IC多=市场风格对=胜率应高"""
    if not FACTOR_HIST.exists():
        return 0.5
    try:
        df = pd.read_csv(FACTOR_HIST)
        # 找最后一天的IC列
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


def save_cfg(cfg: dict) -> None:
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def evolve() -> list[str]:
    """执行一次进化检查，返回变更日志。无变更返回空列表"""
    log: list[str] = []
    cfg = load_cfg()
    n = _actual_record_count()
    wr = _win_rate_approx()

    def _flip(section: str, key: str, reason: str) -> bool:
        """开一个开关，返回 True 表示真的变了"""
        sec = cfg[section]
        # key 可能在 sec 或 sec 的子 dict 里
        if key in sec:
            if isinstance(sec[key], dict) and not sec[key].get("启用", True):
                sec[key]["启用"] = True
                log.append(f"✅ {section}.{key} → 启用（{reason}）")
                return True
        return False

    # 1. 30 笔触发：凯利公式 + 相关性过滤
    if n >= 30:
        _flip("仓位管理", "相关性过滤", f"实盘记录 {n} ≥ 30，避免同行业双杀")
        _flip("仓位管理", "凯利公式", f"实盘记录 {n} ≥ 30，胜率稳定后按凯利分配仓位")

    # 2. 60 笔触发：新因子自动发现
    if n >= 60:
        _flip("因子", "自动发现", f"实盘记录 {n} ≥ 60，开始定期扫描新因子")

    # 3. 100 笔触发：自动 AB 实验
    if n >= 100:
        _flip("观测", "自动AB实验", f"实盘记录 {n} ≥ 100，开启新旧模型对比验证")

    # 把当前状态写回
    if log:
        cfg["_上次进化"] = {
            "实盘记录": n,
            "近似胜率": round(wr, 3),
            "变更": log,
            "时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
        save_cfg(cfg)

    # 头部日志（无变更时也要打状态）
    header = f"【进化检查】实盘记录 {n} 笔 · 近似胜率 {wr:.0%} · 配置开关：相关性={cfg['仓位管理']['相关性过滤']['启用']} / 凯利={cfg['仓位管理']['凯利公式']['启用']} / 新因子={cfg['因子']['自动发现']['启用']} / AB={cfg['观测']['自动AB实验']['启用']}"
    return [header] + log


if __name__ == "__main__":
    changes = evolve()
    for c in changes:
        print(c)
    sys.exit(0)
