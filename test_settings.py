# -*- coding: utf-8 -*-
"""AppTest 无头测试：本金默认值的读取与云端保存（akshare/GitHub API 全部打桩，不联网、不用真实 token）。

场景A：推荐设置.json 写入 3300 → 侧边栏本金输入框默认值 = 3300（证明页面读的是文件值）
场景B：点「💾 保存为本金默认值」→ mock 假 token + 假 GitHub API 使 save_settings 返回 True，
      确认出现成功提示、无异常，且 PUT 提交的内容确实是新本金
运行：python test_settings.py
"""
import base64
import json
import shutil
from pathlib import Path
from unittest import mock

import akshare
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parent
APP = ROOT / "app.py"
SETTINGS = ROOT / "推荐设置.json"
BAK = ROOT / "推荐设置.json.bak"

PUT_CALLS = []          # 记录 fake_put 收到的请求体，供断言


class _FakeResp:
    """模拟 requests 响应：raise_for_status 不抛错，json() 返回固定数据"""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def fake_get(url, headers=None, timeout=None):
    assert timeout == 15, "GET 必须带 timeout=15"
    assert "推荐设置.json" in url
    return _FakeResp({"sha": "fake-sha-for-test"})


def fake_put(url, headers=None, timeout=None, json=None):
    assert timeout == 15, "PUT 必须带 timeout=15"
    PUT_CALLS.append(json)
    return _FakeResp({})


def fake_index(symbol=None, period=None, start_date=None, end_date=None):
    return pd.DataFrame({"收盘": [4000.0] * 99 + [4100.0]})   # 收盘>MA60，大盘正常


def fake_hist(symbol=None, period=None, start_date=None, end_date=None, adjust=None):
    return pd.DataFrame({"日期": [], "收盘": []})             # 复盘用：无后续交易日


def run_app() -> AppTest:
    st.cache_data.clear()                      # 防止跨场景缓存污染
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert len(at.exception) == 0, f"页面抛异常: {[e.value for e in at.exception]}"
    return at


def capital_input(at: AppTest):
    for n in at.number_input:
        if "本金" in n.label:
            return n
    raise AssertionError(f"找不到本金输入框，现有: {[n.label for n in at.number_input]}")


def scenario_a():
    SETTINGS.write_text(json.dumps({"capital": 3300}), encoding="utf-8")
    at = run_app()
    assert float(capital_input(at).value) == 3300.0, capital_input(at).value
    print("场景A（设置文件=3300）：通过 —— 本金输入框默认值 = 推荐设置.json 里的 3300")


def scenario_b():
    SETTINGS.write_text(json.dumps({"capital": 2200}), encoding="utf-8")
    PUT_CALLS.clear()
    # 假 token（非真实 token）+ 假 GitHub API，等效于 monkeypatch save_settings 返回 True
    with mock.patch.object(st, "secrets", {"GH_TOKEN": "fake-token-for-test-not-real"}), \
         mock.patch("requests.get", fake_get), \
         mock.patch("requests.put", fake_put):
        at = run_app()
        capital_input(at).set_value(3500)
        at.run()
        assert len(at.exception) == 0, f"页面抛异常: {[e.value for e in at.exception]}"
        btn = None
        for b in at.button:
            if "保存为本金默认值" in b.label:
                btn = b
                break
        assert btn is not None, f"找不到保存按钮，现有: {[b.label for b in at.button]}"
        btn.click()
        at.run()
        assert len(at.exception) == 0, f"页面抛异常: {[e.value for e in at.exception]}"
        succ = [s.value for s in at.success]
        assert any("已保存" in s and "3500" in s for s in succ), f"成功提示未出现: {succ}"
        assert len(PUT_CALLS) == 1, "应恰好调用一次 GitHub PUT"
        body = PUT_CALLS[0]
        assert body["sha"] == "fake-sha-for-test"          # 用了 GET 拿到的 sha
        assert "网页更新本金为 3500 元" in body["message"]
        payload = json.loads(base64.b64decode(body["content"]).decode("utf-8"))
        assert payload["capital"] == 3500.0, payload       # 提交的确实是新本金
    print("场景B（点保存按钮）：通过 —— 成功提示出现，PUT 内容/提交信息正确，无异常")


if __name__ == "__main__":
    shutil.copy2(SETTINGS, BAK)
    try:
        with mock.patch.object(akshare, "stock_zh_a_hist", fake_hist), \
             mock.patch.object(akshare, "index_zh_a_hist", fake_index):
            scenario_a()
            scenario_b()
    finally:
        shutil.copy2(BAK, SETTINGS)              # 无论成败都恢复真实设置文件
        BAK.unlink()
    print("\n全部场景通过 ✅")
