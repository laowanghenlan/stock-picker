# -*- coding: utf-8 -*-
"""早晚邮件的轻量时段判重，供 Web 触发器与本地兜底脚本共用。"""
import datetime as _datetime
import json
import os
from pathlib import Path

STATE_FILE = Path(__file__).parent / "推送状态.json"


def _today() -> str:
    """北京时间的今天，避免 UTC 跨日把早晚班次误判到不同日期。"""
    return (_datetime.datetime.now(_datetime.timezone.utc)
            + _datetime.timedelta(hours=8)).strftime("%Y-%m-%d")


def _slot() -> str:
    """PUSH_SLOT=pm 表示晚班，其余情况按早班处理。"""
    return "pm" if os.environ.get("PUSH_SLOT") == "pm" else "am"


def _sent_slots() -> set:
    """返回今天已发送的时段集合，兼容旧版只有 last_sent 的状态格式。"""
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if state.get("last_sent") != _today():
        return set()
    slots = state.get("slots")
    return set(slots) if isinstance(slots, list) else {"am"}


def _already_sent() -> bool:
    return _slot() in _sent_slots()


def _mark_sent_today() -> None:
    """发送成功后记录当前时段；由 GitHub 工作流回传仓库持久化。"""
    try:
        slots = _sent_slots()
        slots.add(_slot())
        STATE_FILE.write_text(
            json.dumps({"last_sent": _today(), "slots": sorted(slots)},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:
        print("写入推送状态失败(不影响已发出的邮件):", e)
