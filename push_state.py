# -*- coding: utf-8 -*-
"""早晚邮件的轻量时段判重，供 Web 触发器与本地兜底脚本共用。"""
import csv
import datetime as _datetime
import json
import os
import time
from pathlib import Path

STATE_FILE = Path(__file__).parent / "推送状态.json"
LOG_FILE = Path(__file__).parent / "推送记录.csv"
# 只有落在计划窗口内的留档才作为判重依据，避免手动补发被误判为定时推送。
_SLOT_WINDOWS = {"am": (6, 12), "pm": (18, 24)}


def _today() -> str:
    """北京时间的今天，避免 UTC 跨日把早晚班次误判到不同日期。"""
    return (_datetime.datetime.now(_datetime.timezone.utc)
            + _datetime.timedelta(hours=8)).strftime("%Y-%m-%d")


def _slot() -> str:
    """PUSH_SLOT=pm 表示晚班，其余情况按早班处理。"""
    return "pm" if os.environ.get("PUSH_SLOT") == "pm" else "am"


def _state_slots() -> set:
    """返回今天已发送的时段集合，兼容旧版只有 last_sent 的状态格式。"""
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if state.get("last_sent") != _today():
        return set()
    slots = state.get("slots")
    return set(slots) if isinstance(slots, list) else {"am"}


def _log_slots() -> set:
    """状态文件写入失败时，用推送记录.csv 中今日计划窗口内的记录兜底判重。"""
    try:
        with LOG_FILE.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("日期", "")).strip() != _today():
                    continue
                stamp = str(row.get("推送时间", "")).strip()
                if ":" not in stamp:
                    continue
                hour = int(stamp.split(":")[0])
                for slot, (start, end) in _SLOT_WINDOWS.items():
                    if start <= hour < end:
                        return {slot}
    except Exception:
        pass
    return set()


def _sent_slots() -> set:
    return _state_slots() or _log_slots()


def _already_sent() -> bool:
    return _slot() in _sent_slots()


def _mark_sent_today() -> None:
    """发送成功后记录当前时段；由 GitHub 工作流回传仓库持久化。"""
    slots = _sent_slots()
    slots.add(_slot())
    payload = json.dumps({"last_sent": _today(), "slots": sorted(slots)},
                         ensure_ascii=False, indent=2)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    for attempt in range(3):
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, STATE_FILE)
            return
        except Exception as e:
            if attempt == 2:
                print("写入推送状态失败(不影响已发出的邮件):", e)
            else:
                time.sleep(1)
