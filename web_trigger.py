"""GitHub Actions 通过真实浏览器唤醒 Streamlit，并触发预热或 Web 邮件推送。"""
import os
import re
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from playwright.sync_api import sync_playwright
from push_state import _already_sent, _mark_sent_today

DEFAULT_WEB_URL = "https://stock-picker-mi5bntxj6x4fzrntgtzvko.streamlit.app/"
WARMUP_CRONS = {"30 23 * * 0-4", "30 11 * * 1-5"}
PM_CRONS = {"30 11 * * 1-5", "0,20,40 12 * * 1-5"}


def _requested_action() -> tuple[str, str]:
    schedule = os.environ.get("SCHEDULE_CRON", "")
    action = os.environ.get("TRIGGER_ACTION", "") or (
        "warmup" if schedule in WARMUP_CRONS else "push")
    slot = os.environ.get("TRIGGER_SLOT", "") or (
        "pm" if schedule in PM_CRONS else "am")
    return action, slot


def _target_url(base: str, action: str, slot: str, token: str) -> str:
    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({"action": action, "slot": slot, "token": token})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _workflow_result(value: str) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as f:
            f.write(f"result={value}\n")


def _app_frames(page) -> list:
    """Streamlit Cloud 把应用正文放在 streamlitApp iframe 里，必须进子框架读取。"""
    frames = [f for f in page.frames
              if "/~/+" in (f.url or "") or f.name == "streamlitApp"]
    return frames or [page.main_frame]


def _body_text(page) -> str:
    texts = []
    for frame in _app_frames(page):
        try:
            t = frame.locator("body").inner_text(timeout=8000)
        except Exception:
            t = ""
        if t and t.strip() != "Manage app":
            texts.append(t)
    return "\n".join(texts)


def _wake_if_needed(page, markers: tuple[str, ...]) -> None:
    """Streamlit Cloud 休眠页需要先点击唤醒，已运行时直接等待结果。"""
    deadline = time.monotonic() + 120
    pattern = re.compile(r"get this app back up|wake.*app|唤醒", re.I)
    while time.monotonic() < deadline:
        text = _body_text(page)
        if any(marker in text for marker in markers):
            return
        for frame in _app_frames(page):
            try:
                button = frame.get_by_role("button", name=pattern)
                if button.count() and button.first.is_visible():
                    button.first.click()
                    return
            except Exception:
                pass
        page.wait_for_timeout(2000)


def _wait_marker(page, markers: tuple[str, ...], timeout_seconds: int) -> tuple[str, str]:
    deadline = time.monotonic() + timeout_seconds
    text, last_stage = "", ""
    while time.monotonic() < deadline:
        text = _body_text(page)
        for marker in markers:
            if marker in text:
                return marker, text
        stages = [line.strip() for line in text.splitlines() if "PRECOMPUTE_STAGE" in line]
        if stages and stages[-1] != last_stage:
            last_stage = stages[-1]
            print(last_stage, flush=True)
        page.wait_for_timeout(2000)
    raise TimeoutError(f"{timeout_seconds} 秒内未观察到 {' / '.join(marker for marker in markers)}；"
                       f"最后阶段：{last_stage or '未收到阶段标记'}")


def _safe_text(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


def run() -> int:
    action, slot = _requested_action()
    if action not in ("warmup", "push") or slot not in ("am", "pm"):
        print("TRIGGER_ACTION 必须是 warmup/push，TRIGGER_SLOT 必须是 am/pm", file=sys.stderr)
        return 2
    os.environ["PUSH_SLOT"] = slot
    force = os.environ.get("PUSH_FORCE") == "1"
    if action == "push" and not force and _already_sent():
        print(f"今日 {slot} 时段已推送过，跳过本次冗余触发")
        _workflow_result("skipped")
        return 0

    token = os.environ.get("PUSH_TRIGGER_TOKEN", "")
    if not token:
        print("缺少 PUSH_TRIGGER_TOKEN", file=sys.stderr)
        return 2
    base_url = os.environ.get("PUSH_WEB_URL", DEFAULT_WEB_URL).strip()
    target = _target_url(base_url, action, slot, token)
    # 预热最多等到准点前 5 分钟，避免失败任务占用并发组、阻塞正式推送。
    timeout_seconds = int(os.environ.get(
        "PUSH_BROWSER_TIMEOUT", "1500" if action == "warmup" else "3600"))
    # 参数/令牌错误统一由 Web 输出 PUSH_ERROR，预热请求也必须立即识别。
    markers = (("WARMUP_OK", "WARMUP_ERROR", "PUSH_ERROR") if action == "warmup"
               else ("PUSH_OK", "PUSH_ERROR"))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(target, wait_until="domcontentloaded", timeout=120000)
                _wake_if_needed(page, markers)
                marker, text = _wait_marker(page, markers, timeout_seconds)
                if marker.endswith("_ERROR"):
                    print("Web 返回失败：\n" + _safe_text(text[-1200:], token), file=sys.stderr)
                    return 1

                if action == "warmup":
                    print(f"Web {slot} 时段预热完成")
                    _workflow_result("warmup")
                    return 0

                # 一旦邮件成功，立即记录状态；后续归档超时绝不触发重发。
                _mark_sent_today()
                if not _already_sent():
                    print("警告：邮件已发送，但推送状态未写入，将以推送记录留档兜底判重",
                          file=sys.stderr)
                _workflow_result("sent")
                print(f"Web {slot} 时段邮件发送成功，已记录推送状态")
                try:
                    archive_marker, _ = _wait_marker(
                        page, ("ARCHIVE_OK", "ARCHIVE_WARN"), min(timeout_seconds, 120))
                    if archive_marker == "ARCHIVE_WARN":
                        print("邮件已发送；Web 报告留档同步告警，不自动重发")
                    else:
                        print("Web 留档同步完成")
                except Exception:
                    print("邮件已发送；归档结果等待结束，不自动重发")
                return 0
            finally:
                browser.close()
    except Exception as e:
        message = _safe_text(str(e), token)
        print(f"浏览器触发失败：{type(e).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(run())
