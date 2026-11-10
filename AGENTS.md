# AGENTS.md — 项目交接说明

> 给接手本项目的 AI 智能体看。读完本文件再动手，能避免重复踩坑。

## 项目是什么

A 股散户选股系统。使用者本金 2200 元，中信证券开户。系统每天自动选股、发邮件提醒、记录推演结果，并根据实盘表现自动进化。

**核心设计原则：只改 `strategy_config.json`，不改源码就能完成策略升级。** 任何把可调参数写死回代码的改动都是倒退。

仓库：https://github.com/laowanghenlan/stock-picker （public，master 分支）
网页版：Streamlit Cloud 自动跟随 master，推送后约 2~3 分钟生效
定时任务：GitHub Actions 每天 UTC 1:00（北京 9:00）跑 `daily_push.py` + `evolution.py`

## 环境搭建（新机器必做）

项目根：`D:\文档\股票`

**venv 必须放 C 盘，D 盘装不了 pip 包。**

```powershell
py -m venv C:\venvs\stock
C:\venvs\stock\Scripts\pip install -r requirements.txt
```

注意：历史 venv 曾放在 `C:\Users\Administrator\Documents\Trae-CN\temp_*\stock_analyzer\.venv`，属于临时目录，会失效，不要依赖。

**本地运行所需环境变量**（GitHub Actions 上已配好，本地才需要）：

| 变量 | 说明 |
|---|---|
| `QQ_EMAIL` | QQ 邮箱地址（收件与发件相同） |
| `QQ_AUTH_CODE` | QQ 邮箱 SMTP 授权码 |
| `CAPITAL` | 本金，默认 2200 |

**接线自检**：

```powershell
C:\venvs\stock\Scripts\python -c "import app, factors, fundamental, evolution; print(app.APP_VERSION)"
```

能打印版本号即成功。

## 文件职责

| 文件 | 职责 |
|---|---|
| `app.py` | 主程序。Streamlit 页面、`verify_impl()` 选股主流程、回测、评分、仓位分配 |
| `factors.py` | 因子计算、中性化、IC/IR 检验、因子权重。含 4 个基本面因子 |
| `fundamental.py` | Baostock 历史财报取数层，带磁盘缓存与自动降级 |
| `evolution.py` | 进化检查。v1.8 起不翻转开关，只报告各功能置信度 |
| `daily_push.py` | 邮件推送。`_collect()` 跑选股，`_render_text()` / `_render_html()` 双版本 |
| `factor_learning.py` | 因子学习与熔断判断 |
| `portfolio_optimizer.py` | v1.8 仓位优化。PyPortfolioOpt 优先，numpy 闭式解兜底，带置信度折扣 |
| `strategy_config.json` | **所有可调参数的唯一入口** |
| `.github/workflows/daily_push.yml` | Actions 定时任务 |
| `推送记录.csv` | 推演记录，进化引擎的数据来源 |
| `因子历史IC.csv` | IC 历史留档 |
| `fundamental_cache/` | Baostock 季度财报缓存 |

## 铁律（违反会出数据错误或部署异常）

1. **调参只改 `strategy_config.json`**：本金、候选池、止盈止损、熔断阈值、凯利/相关性开关、IC 门槛、衰减天数全在里面。
2. **版本号有两处**：`app.py` 和 `daily_push.py` 的 `APP_VERSION`，必须同步。
3. **push 前用 `git pull --no-rebase origin master`**。Actions 每天回传数据文件，rebase 属于改写历史，会被安全策略拦截。
4. **PowerShell 不支持 heredoc**。commit 用单行 `-m`，或先写文件再 `git commit -F`。
5. **Baostock 报告期不能转日期**：`"2025Q4"` 若转成 `"2025-4"` 再解析，会被当成 2025 年 4 月（Q4 变成 Q2）。必须用 `pd.PeriodIndex(..., freq="Q")`。
6. **Baostock 数值已是小数比例**：`roeAvg` 返回 0.049 即 4.9%，`npMargin` / `YOYNI` / `YOYPNI` 同理。**不要再除以 100**。
7. **基本面因子必须按 `pubDate` 对齐**（`pd.merge_asof(direction="backward")`），否则会产生前视偏差。

## 已知的坑

| 现象 | 原因与处理 |
|---|---|
| 基本面因子全 NaN | pandas 3.1 的 `merge_asof` 要求两边 datetime 精度一致，先统一 `astype("datetime64[ns]")` |
| baostock 连接失败 | 家用宽带会封它；代码有 try/except 降级，基本面因子变 NaN，IC 检验自动跳过 |
| 网页端选股要等约 6 分钟 | 候选池 30 只；改 config 的「网页端」为 15 即恢复 |
| 邮件端跑约 12 分钟 | 候选池 60 只，Actions 后台跑，无超时风险 |
| GitHub 443 断连 | 用加速工具，或 `git pull --no-rebase` 后重试 |
| Actions 依赖安装失败 | 看日志是否 `Could not find a version that satisfies the requirement baostock` |

## 日常操作（用户自助，不需要改代码）

| 想做什么 | 改哪里 |
|---|---|
| 改本金 | Streamlit 页面输入框，或 `推荐设置.json` |
| 扩/缩候选池 | `strategy_config.json` → 「候选池」→「网页端」/「邮件推送」 |
| 改止盈止损 | `strategy_config.json` → 「风控」 |
| 改 IC 门槛 | `strategy_config.json` → 「因子」→「IC门槛」（默认 0.03） |

## 自动进化机制

v1.8 起改为「全开 + 置信度折扣」模式：

- **所有功能从第一天就启用**（相关性过滤、凯利公式、均值方差优化、因子自动发现、AB 实验）
- **效果按实盘样本量线性打折**：置信度 = min(1.0, 实笔数 / 置信度阈值)
  - 样本不足时，优化权重向等额靠拢、凯利仓位缩小、相关性阈值抬高、因子发现门槛提高
  - 样本充足后自动恢复全力
- `evolution.py` 不再翻转开关，只打印各功能的置信度状态

| 功能 | 置信度阈值 | 折扣效果 |
|---|---|---|
| 相关性过滤 | 30 | 阈值从 0.95 线性降到 0.80 |
| 凯利公式 | 30 | 仓位 f 从 0 线性升到理论值 |
| 均值方差优化 | 30 | 权重从等额线性升到 max_sharpe 最优解 |
| 因子自动发现 | 60 | IC 门槛从高线性降到 0.03 |
| AB 实验 | 100 | 仅记录，满 100 笔才实际对比 |

**仓位优化优先级**：均值方差（PyPortfolioOpt / numpy 闭式解兜底）→ 凯利公式 → 等额分配。

注意：笔数增长与「已结算笔数」是两回事。每笔要等最多 20 个交易日（止盈/止损/到期）才产生盈亏，胜率类指标需要已结算样本才有意义。

## 因子体系

技术面：动量 20、波动率 20、换手 20、EP、BP
基本面：ROE、净利润增速、营收增速、净利率（来自 Baostock，按 pubDate 对齐）

因子权重由 IC/IR 决定，负 IC 因子自动被当作反转因子使用。IC 绝对值低于门槛会被降权。

## 发布流程

```powershell
cd "D:\文档\股票"
git add <具体文件>
git commit -m "单行说明"
git pull --no-rebase origin master
git push origin master
```

推送后：Streamlit Cloud 自动部署；次日 9:00 Actions 自动跑；也可在 Actions 页面手动 Run workflow 立即验证。

## 当前状态

- 版本：v1.8（全开+置信度折扣；新增 PyPortfolioOpt 均值方差仓位优化）
- 候选池：网页 30 / 邮件 60
- 实盘记录：约 11 笔，所有进化功能全开但置信度较低（仓位/相关性/凯利 37%，因子发现 18%，AB 11%）
- 下一步观察点：4 个基本面因子（尤其 ROE）在实盘的 IC 是否稳定；随实笔数增长，置信度折扣自动解除
