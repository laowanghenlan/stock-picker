# -*- coding: utf-8 -*-
"""周末云端流水线（v1.56）：休息日不空转，用仓库自带的历史数据包重新训练模型 + 跑一次预测并存档。

在 GitHub Actions 的周末任务里执行，两种模式分开跑、互不拖累：
    python weekend_pipeline.py predict   # 选股（复用 daily_push._collect）→ 写独立周末存档 + 回填台账
    python weekend_pipeline.py train     # 解压 training_data/wf_cache_bundle.tar.gz → 重训 LGB/ET/MLP

设计要点：
  1. 预测存档写独立的 周末预测存档.csv，**不写实盘台账、不写推送记录.csv**——周末预测基于周五
     静态快照、没有真实交易意义，混入会污染真实 OOS IC 统计。只调 oos_ledger.update_due()
     把到期的历史真实样本补齐（这一步本来就该做，周末正好补）。
  2. 训练数据包随仓库分发（wf_cache 被 gitignore，云端拿不到），解压到 wf_cache 后复用
     pretrain_models.main() 的现成门控（LGB 需 oof_pass、ET/MLP 需 IC≥阈值才覆盖模型），
     IC 不达标自动保留旧模型，绝不把坏模型推上云端。
  3. 全程降级：单个模式失败只记日志、返回码 1，不影响另一模式；workflow 用 `|| true` 兜底。
  4. 不联网发邮件（用户在多选里未勾选），选股本身需要的行情接口照常访问。

用法（本地也可跑训练；predict 需 akshare，本地 venv 缺，交给云端）：
    .venv/Scripts/python.exe weekend_pipeline.py train
"""
import csv
import datetime as dt
import io
import os
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BUNDLE = HERE / "training_data" / "wf_cache_bundle.tar.gz"
ARCHIVE = HERE / "周末预测存档.csv"
STATE = HERE / "周末任务状态.json"
ARCHIVE_COLS = ["日期", "推送时间", "代码", "名称", "现价", "股数", "投入",
                "盈利概率", "仓位占比", "股东减持", "大盘标签", "回测收益", "回测胜率"]


def _beijing_today() -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).strftime("%Y-%m-%d")


def _beijing_hm() -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=8)).strftime("%H:%M")


def _write_state(mode: str, ok: bool, detail: str) -> None:
    """把本次周末任务结果落盘（供 workflow 回传、页面/本地查看）。"""
    try:
        import json
        state = {}
        if STATE.exists():
            try:
                state = json.loads(STATE.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        state[mode] = {
            "日期": _beijing_today(),
            "时间": _beijing_hm(),
            "状态": "ok" if ok else "fail",
            "详情": str(detail)[:500],
        }
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[weekend] 状态写入失败(不影响任务): {e}")


def _unpack_bundle() -> bool:
    """把随仓库分发的训练数据包解压到 wf_cache（覆盖式，保证云端与本地同源）。"""
    if not BUNDLE.exists():
        print(f"[weekend] 训练数据包不存在: {BUNDLE}")
        return False
    t0 = time.time()
    with tarfile.open(BUNDLE, "r:gz") as tf:
        # 安全检查：只解包 wf_cache/ 下的常规文件，拒绝路径穿越
        members = []
        for m in tf.getmembers():
            if not (m.name == "wf_cache" or m.name.startswith("wf_cache/")):
                continue
            if m.isfile() or m.isdir():
                members.append(m)
        try:
            tf.extractall(HERE, members=members, filter="data")   # Python 3.12+/3.14 安全解包
        except TypeError:
            tf.extractall(HERE, members=members)                  # 旧版 Python 无 filter 参数
    n = len(list((HERE / "wf_cache").glob("hist_*.csv")))
    print(f"[weekend] 数据包解压完成: {n} 只K线, 耗时 {time.time()-t0:.1f}s")
    return n > 0


def run_train() -> int:
    """重新训练 LGB/ET/MLP，沿用 pretrain_models 的 IC 门控。"""
    t0 = time.time()
    print("[weekend:train] 开始...")
    try:
        if not _unpack_bundle():
            _write_state("train", False, "训练数据包解压失败")
            return 1
        import pretrain_models
        rc = pretrain_models.main()
        ok = (rc == 0)
        detail = f"训练脚本返回 {rc}, 耗时 {time.time()-t0:.0f}s（IC 未达标的模型自动保留旧版）"
        print(f"[weekend:train] {'完成' if ok else '失败'}: {detail}")
        _write_state("train", ok, detail)
        return 0 if ok else 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_state("train", False, f"{type(e).__name__}: {e}")
        return 1


def run_predict() -> int:
    """周末选股并写独立存档；只回填台账到期样本，不记新预测、不发邮件。"""
    t0 = time.time()
    print("[weekend:predict] 开始...")
    try:
        # daily_push 模块级直接读 os.environ["QQ_EMAIL"]/["QQ_AUTH_CODE"]，缺键会 KeyError。
        # 周末任务不发邮件，这里兜底占位，彻底去掉对 GitHub secrets 的依赖。
        os.environ.setdefault("QQ_EMAIL", "weekend@local")
        os.environ.setdefault("QQ_AUTH_CODE", "weekend-no-send")
        import daily_push
        d = daily_push._collect(archive=False)   # archive=False：不写推送记录、不记实盘台账
        note = d.get("note", "")
        if note:
            # 空仓/信号弱也是一次合法的周末结论，照样存档备查
            print(f"[weekend:predict] 选股结论: {note}")
        items = d.get("plan", [])
        date_str = d.get("date", _beijing_today())
        ptime = _beijing_hm()

        rows = []
        for it in items:
            row = {c: it.get(c, "") for c in ARCHIVE_COLS}
            row["日期"] = date_str
            row["推送时间"] = ptime
            row["大盘标签"] = d.get("tag", "")
            rows.append(row)
        if not rows:
            # 无票也留一条空结论，证明周末流程确实跑过
            rows = [{"日期": date_str, "推送时间": ptime, "代码": "", "名称": "（空仓/无票）",
                     "大盘标签": d.get("tag", ""), "备注": note}]

        write_header = not ARCHIVE.exists() or ARCHIVE.stat().st_size == 0
        with ARCHIVE.open("a", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ARCHIVE_COLS + ["备注"], extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerows(rows)
        print(f"[weekend:predict] 已存档 {len(items)} 只 → {ARCHIVE.name}")

        # 只回填到期的历史真实样本（不记周末预测），让周末也能补齐 T+1/5/10 日收益
        try:
            import oos_ledger
            filled = oos_ledger.update_due()
            print(f"[weekend:predict] 台账回填 {filled} 行")
        except Exception as e:
            print(f"[weekend:predict] 台账回填跳过: {type(e).__name__}: {e}")

        detail = f"存档 {len(items)} 只；{note or '正常'}；耗时 {time.time()-t0:.0f}s"
        _write_state("predict", True, detail)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_state("predict", False, f"{type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "predict":
        sys.exit(run_predict())
    elif mode == "train":
        sys.exit(run_train())
    else:
        print(__doc__)
        print("错误：请指定 predict 或 train")
        sys.exit(2)
