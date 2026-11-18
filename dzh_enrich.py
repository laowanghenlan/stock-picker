"""大智慧资金流/龙虎榜增强标注层（v1.30）
=================================================
定位：精筛器(refine_pool.py)输出的清单 → 本脚本补上大智慧资金流/龙虎榜作为【人工对照标注】。
      这些字段只有当日/近期截面，未经含退市全A历史面板验证，绝不进打分核心（避免幸存者偏差）。

架构说明（重要）：
  大智慧是 MCP 工具，只有 WorkBuddy 对话里的 agent 能调用；普通 Python 脚本运行时调不到 MCP。
  故采用【半自动】模式：
    1) 对话里 agent 调大智慧 MCP，把原始结果落盘成 dzh_mcp_snapshot.json（见工作区）；
    2) 本脚本读该 JSON + 精筛清单，合并输出增强清单；
    3) 每次快照 append 到资金流历史库，攒够3~6个月后可验证是否值得进打分核心。

用法：
  python dzh_enrich.py                          # 增强工作区最新的精筛清单
  python dzh_enrich.py 精筛清单_xxx.csv          # 指定清单
  python dzh_enrich.py --snapshot dzh_mcp_snapshot.json   # 指定MCP快照
输出：精筛增强清单_YYYYMMDD_HHMM.csv（原列 + 资金流/龙虎榜标注列）
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = Path("C:/Users/Admin/WorkBuddy/2026-11-18-08-05-50")
DEFAULT_SNAPSHOT = WORKSPACE / "dzh_mcp_snapshot.json"
FLOW_HISTORY = WORKSPACE / "dzh_fundflow_history.csv"   # 资金流历史库(逐日append)


def log(*a): print(*a, flush=True)


def to_dz(code):
    """600000 -> SH600000；000001/300xxx -> SZ。"""
    c = str(code).zfill(6)
    return ("SH" if c[0] == "6" else "SZ") + c


def find_pick(arg):
    if arg:
        p = Path(arg)
        return p if p.is_absolute() else HERE / arg
    cands = sorted(HERE.glob("精筛清单_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands += sorted(Path.cwd().glob("精筛清单_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def load_snapshot(path):
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / p
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def append_history(snap):
    """把本次资金流快照 append 到历史库，供将来验证。"""
    ts = snap.get("snapshot_time", "")
    rows = []
    for code, d in (snap.get("capital_inflow_min") or {}).items():
        rows.append({"快照时间": ts, "大智慧代码": code,
                     "主力净流入元": d.get("主力净流入"),
                     "特大单元": d.get("特大单"), "大单元": d.get("大单"),
                     "中单元": d.get("中单"), "小单元": d.get("小单")})
    if not rows:
        return
    new = pd.DataFrame(rows)
    if FLOW_HISTORY.exists():
        old = pd.read_csv(FLOW_HISTORY, dtype={"大智慧代码": str})
        new = pd.concat([old[old["快照时间"] != ts], new], ignore_index=True)
    new.to_csv(FLOW_HISTORY, index=False, encoding="utf-8-sig")
    log(f"资金流历史库已更新：{FLOW_HISTORY.name}（累计 {len(new)} 条）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pick", nargs="?", help="精筛清单CSV（默认找最新的）")
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT), help="大智慧MCP快照JSON")
    a = ap.parse_args()

    pick = find_pick(a.pick)
    if pick is None or not pick.exists():
        log("!! 没找到精筛清单。请先跑 refine_pool.py 生成，或指定路径。")
        sys.exit(1)
    df = pd.read_csv(pick, dtype={"代码": str})
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    log(f"读入精筛清单 {len(df)} 只：{pick.name}")

    snap = load_snapshot(a.snapshot)
    if snap is None:
        log(f"!! 没找到MCP快照：{a.snapshot}")
        log("   需在对话里让 agent 调大智慧 MCP 并把结果落盘成该 JSON。")
        sys.exit(1)
    flow = snap.get("capital_inflow_min") or {}
    lhb = snap.get("lhb_last_month") or {}
    log(f"MCP快照时间：{snap.get('snapshot_time', '?')}，覆盖 {len(flow)} 只资金流 / {len(lhb)} 只龙虎榜")

    # 补列（对照标注，不进打分）
    df["大智慧代码"] = df["代码"].map(to_dz)
    df["主力净流入(万)"] = df["大智慧代码"].map(lambda c: round(flow.get(c, {}).get("主力净流入", np.nan) / 1e4, 1))
    df["特大单净流入(万)"] = df["大智慧代码"].map(lambda c: round(flow.get(c, {}).get("特大单", np.nan) / 1e4, 1))
    df["近一月龙虎榜次数"] = df["大智慧代码"].map(lambda c: lhb.get(c, {}).get("次数", np.nan))
    df["最近龙虎榜日"] = df["大智慧代码"].map(lambda c: lhb.get(c, {}).get("最近上榜日", ""))

    def signal(row):
        v = row.get("主力净流入(万)")
        if not np.isfinite(v):
            return "无资金数据"
        if v > 5000:
            return "主力大幅流入"
        if v > 0:
            return "主力流入"
        if v > -5000:
            return "主力流出"
        return "主力大幅流出"

    df["资金信号"] = df.apply(signal, axis=1)
    # 交叉提示：技术精筛靠前但主力流出 = 需警惕
    rank_col = "精筛排名" if "精筛排名" in df.columns else None
    df["对照提示"] = df.apply(lambda r: ("⚠技术靠前但主力流出" if rank_col and r[rank_col] <= 5
                                        and isinstance(r["主力净流入(万)"], (int, float))
                                        and r["主力净流入(万)"] < 0 else ""), axis=1)

    keep = [c for c in ["精筛排名", "代码", "名称", "精筛分", "最新价", "主力净流入(万)",
                        "特大单净流入(万)", "近一月龙虎榜次数", "最近龙虎榜日", "资金信号", "对照提示"] if c in df.columns]
    show = df[keep].copy()

    log(f"\n===== T+1 精筛增强清单（技术分 + 大智慧资金流对照）=====")
    log(show.to_string(index=False))

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
    outp = HERE / f"精筛增强清单_{ts}.csv"
    show.to_csv(outp, index=False, encoding="utf-8-sig")
    log(f"\n已保存：{outp.name}")
    append_history(snap)
    log("提示：资金流/龙虎榜为当日对照标注，未进打分核心、未经历史验证，仅供参考，不构成投资建议。")


if __name__ == "__main__":
    main()
