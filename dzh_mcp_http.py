"""大智慧 MCP · KEY 直连 HTTP 客户端（零依赖，标准库实现）
================================================================
用途：用官方 sk-mcp KEY 直连 `https://mcpali.dzh.com.cn/pub/mcp`，
      让【普通 Python 脚本】也能拉资金流/龙虎榜，不再依赖"对话里 agent 调 MCP 落盘"。

背景：
  - 该端点是标准 MCP over streamable-http；WorkBuddy 的连接器走 OAuth2（token 由守护进程托管，脚本拿不到）。
  - 大智慧官方商城(pay.dzh.com.cn/pay-mall/mcp)提供 sk-mcp-* 专属 KEY，
    在请求头带 `key` 即可【免授权】直连同一个端点。
  - 本客户端实现 MCP 握手(initialize → notifications/initialized → tools/call)，
    并解析 JSON 或 SSE 两种响应格式。

KEY 配置（按优先级）：
  1) 命令行 --key sk-mcp-xxxx
  2) 环境变量 DZH_MCP_KEY
  3) 工作区文件 C:/Users/Admin/WorkBuddy/2026-11-18-08-05-50/dzh_key.json  的 {"key": "sk-mcp-xxxx"}
  ⚠ KEY 是敏感凭证：绝不要写进本仓库（公开 Git），只放上面的工作区文件或环境变量。

用法：
  python dzh_mcp_http.py --check                         # 自检：握手+列工具，不打印KEY
  python dzh_mcp_http.py --snapshot --codes SH600410,SZ000001
  python dzh_mcp_http.py --snapshot --from-pick 精筛清单_xxx.csv
  python dzh_mcp_http.py --raw getAShareLhbData '{"param":{...}}'   # 通用工具调用

输出：
  --snapshot 生成 dzh_mcp_snapshot.json（与 dzh_enrich.py 消费的格式同构），
  可直接接 `python dzh_enrich.py` 完成增强。
"""
import sys, os, json, csv, time, argparse, urllib.request, urllib.error
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ENDPOINT = "https://mcpali.dzh.com.cn/pub/mcp"
WORKSPACE = Path("C:/Users/Admin/WorkBuddy/2026-11-18-08-05-50")
KEY_FILE = WORKSPACE / "dzh_key.json"
SNAPSHOT_OUT = WORKSPACE / "dzh_mcp_snapshot.json"
PROTOCOL_VERSION = "2025-06-18"


def log(*a):
    print(*a, flush=True)


class DzhMcpError(Exception):
    pass


class DzhMcpClient:
    """大智慧 MCP streamable-http 客户端。"""

    def __init__(self, key, endpoint=ENDPOINT, auth="key", timeout=40):
        if not key or not str(key).startswith("sk-mcp-"):
            raise DzhMcpError("KEY 为空或格式不对（应为 sk-mcp- 开头）。")
        self.key = key.strip()
        self.endpoint = endpoint
        self.auth = auth  # 'key' 官方推荐；'bearer' 备用
        self.timeout = timeout
        self.session_id = None
        self._id = 0

    def _next_id(self):
        self._id += 1
        return self._id

    def _headers(self):
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self.auth == "bearer":
            h["Authorization"] = f"Bearer {self.key}"
        else:
            h["key"] = self.key
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        return h

    def _parse_body(self, body: bytes, ctype: str):
        """解析 JSON 或 SSE 响应，返回 JSON-RPC 对象（取最后一条 result/error）。"""
        text = body.decode("utf-8", errors="replace")
        if "text/event-stream" in ctype or text.lstrip().startswith("event:"):
            msgs = []
            for block in text.split("\n\n"):
                data_lines = [ln[5:].lstrip() for ln in block.splitlines() if ln.startswith("data:")]
                if not data_lines:
                    continue
                payload = "\n".join(data_lines).strip()
                if not payload:
                    continue
                try:
                    msgs.append(json.loads(payload))
                except Exception:
                    pass
            if not msgs:
                raise DzhMcpError("SSE 响应里没有可解析的 data 消息。")
            return msgs[-1]
        # 纯 JSON（可能是单行或多行）
        try:
            return json.loads(text)
        except Exception:
            # 有的实现返回多行 JSON-RPC，取最后一行
            for ln in reversed([x for x in text.splitlines() if x.strip()]):
                try:
                    return json.loads(ln)
                except Exception:
                    continue
            raise DzhMcpError("响应不是合法 JSON：\n" + text[:500])

    def _post(self, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.endpoint, data=data, headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                return self._parse_body(resp.read(), ctype)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:300]
            wa = e.headers.get("WWW-Authenticate", "")
            if e.code in (401, 403):
                hint = " KEY 无效/过期，或鉴权方式不对。"
                if self.auth == "key":
                    hint += f" 可试 --auth bearer（服务端 WWW-Authenticate={wa or 'Bearer'}）。"
                raise DzhMcpError(f"鉴权失败 HTTP {e.code}.{hint} 响应:{body}")
            raise DzhMcpError(f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise DzhMcpError(f"网络错误：{e.reason}")

    def initialize(self):
        payload = {
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "stock-picker-dzh-http", "version": "1.0"},
            },
        }
        r = self._post(payload)
        if "error" in r:
            raise DzhMcpError(f"initialize 错误：{r['error']}")
        self._notify_initialized()
        return r.get("result", {})

    def _notify_initialized(self):
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            self._post(payload)  # 通常返回 202 空体；忽略解析失败
        except Exception:
            pass

    def list_tools(self):
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}}
        r = self._post(payload)
        if "error" in r:
            raise DzhMcpError(f"tools/list 错误：{r['error']}")
        return [t.get("name") for t in r.get("result", {}).get("tools", [])]

    def call_tool(self, name, arguments):
        """返回工具的结构化数据：优先解析 content[].text 里的 JSON。"""
        payload = {
            "jsonrpc": "2.0", "id": self._next_id(), "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        r = self._post(payload)
        if "error" in r:
            raise DzhMcpError(f"tools/call({name}) 错误：{r['error']}")
        result = r.get("result", {})
        if result.get("isError"):
            txt = " | ".join(c.get("text", "") for c in result.get("content", []))
            raise DzhMcpError(f"工具返回错误：{txt[:400]}")
        # 结构化内容优先
        if "structuredContent" in result:
            return result["structuredContent"]
        texts = []
        for c in result.get("content", []):
            if c.get("type") == "text" and c.get("text"):
                texts.append(c["text"])
        joined = "\n".join(texts)
        try:
            return json.loads(joined)
        except Exception:
            return joined

    # ---------- 高层业务封装 ----------
    def capital_inflow_min(self, codes):
        """资金流向分时(最新交易日) → {大智慧代码: {...}}"""
        raw = self.call_tool("getAShareLhbData", {
            "param": {"stockCodes": codes, "getCapitalinflowMin": True}
        })
        out = {}
        for code, node in (raw or {}).items():
            try:
                rows = node["getCapitalinflowMin"]["data"]
                d = rows[-1] if rows else {}
                out[code] = {
                    "主力净流入": d.get("ZhuLiJingLiuRuE"),
                    "小单": d.get("XiaoDanJingLiuRuE"),
                    "中单": d.get("ZhongDanJingLiuRuE"),
                    "大单": d.get("DaDanJingLiuRuE"),
                    "特大单": d.get("TeDaDanJingLiuRuE"),
                    "时间": d.get("ShiJianStr"),
                }
            except Exception:
                continue
        return out

    def lhb_last_month(self, codes):
        """近一月龙虎榜明细 → {大智慧代码: {次数, 最近上榜日, 最近涨幅%}}"""
        raw = self.call_tool("getAShareLhbData", {
            "param": {"stockCodes": codes, "getF10LhbGpmxData": {}}
        })
        out = {}
        for code, node in (raw or {}).items():
            try:
                rows = node["getF10LhbGpmxData"]["data"]
                rows = rows or []
                if not rows:
                    out[code] = {"次数": 0, "最近上榜日": "", "最近涨幅%": None}
                    continue
                latest = max(rows, key=lambda r: str(r.get("tradedate", "")))
                out[code] = {
                    "次数": len(rows),
                    "最近上榜日": latest.get("tradedate"),
                    "最近涨幅%": latest.get("pchg"),
                }
            except Exception:
                continue
        return out

    def build_snapshot(self, codes, note=""):
        """批量拉取并生成与 dzh_enrich.py 同构的快照 dict。"""
        codes = list(dict.fromkeys(codes))  # 去重保序
        log(f"[1/2] 拉取资金流分时（{len(codes)} 只）…")
        flow = self.capital_inflow_min(codes)
        log(f"[2/2] 拉取近一月龙虎榜（{len(codes)} 只）…")
        lhb = self.lhb_last_month(codes)
        return {
            "snapshot_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": note or "大智慧MCP KEY直连自动拉取（dzh_mcp_http.py）",
            "source": "dzh-mcp-http-key",
            "capital_inflow_min": flow,
            "lhb_last_month": lhb,
        }


def to_dz(code):
    c = str(code).strip().zfill(6)
    if len(c) != 6 or not c.isdigit():
        return None
    return ("SH" if c[0] == "6" else "SZ") + c


def load_key(arg_key):
    if arg_key:
        return arg_key
    env = os.environ.get("DZH_MCP_KEY")
    if env:
        return env
    if KEY_FILE.exists():
        try:
            obj = json.loads(KEY_FILE.read_text(encoding="utf-8"))
            return obj.get("key")
        except Exception:
            pass
    return None


def codes_from_pick(pick):
    p = Path(pick)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        # 尝试常用位置
        for base in (Path("C:/文档/股票"), WORKSPACE):
            q = base / pick
            if q.exists():
                p = q
                break
    codes = []
    with open(p, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            raw = row.get("代码") or row.get("code") or ""
            dz = to_dz(raw)
            if dz:
                codes.append(dz)
    return codes


def main():
    ap = argparse.ArgumentParser(description="大智慧 MCP KEY 直连客户端")
    ap.add_argument("--key", help="sk-mcp KEY（也可用环境变量/工作区文件）")
    ap.add_argument("--auth", choices=["key", "bearer"], default="key", help="鉴权方式，默认 key（官方）")
    ap.add_argument("--endpoint", default=ENDPOINT)
    ap.add_argument("--check", action="store_true", help="自检：握手+列工具")
    ap.add_argument("--snapshot", action="store_true", help="拉取资金流+龙虎榜生成快照")
    ap.add_argument("--codes", help="逗号分隔代码，如 SH600410,SZ000001")
    ap.add_argument("--from-pick", help="从精筛清单CSV读代码")
    ap.add_argument("--out", default=str(SNAPSHOT_OUT), help="快照输出路径")
    ap.add_argument("--raw", nargs=2, metavar=("TOOL", "ARGS_JSON"), help="通用工具调用，ARGS_JSON为参数对象")
    a = ap.parse_args()

    key = load_key(a.key)
    if not key:
        log("!! 没有 KEY。请三选一：")
        log(f"   1) 命令行 --key sk-mcp-xxxx")
        log("   2) 环境变量 DZH_MCP_KEY=sk-mcp-xxxx")
        log(f"   3) 写文件 {KEY_FILE} 内容 {{\"key\": \"sk-mcp-xxxx\"}}")
        log("   申请地址：https://pay.dzh.com.cn/pay-mall/mcp")
        sys.exit(2)

    cli = DzhMcpClient(key, endpoint=a.endpoint, auth=a.auth)
    try:
        log("正在握手 initialize …")
        info = cli.initialize()
        log(f"握手成功。协议版本：{info.get('protocolVersion', '?')}，session={bool(cli.session_id)}")
    except DzhMcpError as e:
        log(f"!! 握手失败：{e}")
        sys.exit(1)

    if a.check:
        try:
            tools = cli.list_tools()
            log(f"可用工具 {len(tools)} 个：" + ", ".join(tools[:20]))
            log("✅ 自检通过：KEY 有效，脚本可直连大智慧 HTTP 接口。")
        except DzhMcpError as e:
            log(f"!! 列工具失败：{e}")
            sys.exit(1)
        return

    if a.raw:
        tool, args = a.raw
        try:
            args_obj = json.loads(args)
        except Exception as e:
            log(f"!! ARGS_JSON 解析失败：{e}")
            sys.exit(2)
        try:
            data = cli.call_tool(tool, args_obj)
            log(json.dumps(data, ensure_ascii=False, indent=2)[:8000])
        except DzhMcpError as e:
            log(f"!! 调用失败：{e}")
            sys.exit(1)
        return

    if a.snapshot:
        if a.codes:
            codes = [to_dz(x) for x in a.codes.split(",")]
            codes = [c for c in codes if c]
        elif a.from_pick:
            codes = codes_from_pick(a.from_pick)
        else:
            log("!! --snapshot 需给 --codes 或 --from-pick")
            sys.exit(2)
        if not codes:
            log("!! 没有有效代码")
            sys.exit(2)
        log(f"目标 {len(codes)} 只：{','.join(codes[:8])}{'...' if len(codes) > 8 else ''}")
        try:
            snap = cli.build_snapshot(codes)
        except DzhMcpError as e:
            log(f"!! 拉取失败：{e}")
            sys.exit(1)
        out = Path(a.out)
        out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"快照已写入：{out}")
        log(f"  资金流 {len(snap['capital_inflow_min'])} 只 / 龙虎榜 {len(snap['lhb_last_month'])} 只")
        log("下一步：python dzh_enrich.py  （即可自动消费这份快照完成增强）")
        return

    # 没给动作：握手成功后提示
    log("握手已成功。请用 --check / --snapshot / --raw 指定动作。")


if __name__ == "__main__":
    main()
