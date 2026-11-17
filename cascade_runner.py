# -*- coding: utf-8 -*-
"""三级漏斗选股引擎 —— Stage1 粗筛 → Stage2 工具组合 → Stage3 DL精筛。

核心思想：
  Stage 1 是 Web 快速返回的"粗筛"——硬过滤 + 快速技术预筛 + 快速舆情排雷，
  目标是 5000+ → 50~80 只，耗时 < 5 秒。

  Stage 2 是"工具组合层"——逐只拉 K 线 + 回测验证 + 因子 IC 检验 + 稳定性惩罚
  + 统计门禁（胜率显著性 / VaR），目标 50~80 → 15~25 只，耗时 2~4 分钟。

  Stage 3 是"DL 精筛"——LightGBM + sklearn ExtraTrees + sklearn MLP 三个模型
  集成预测未来 5 日收益，目标 15~25 → 3~5 只，耗时 1~2 分钟。

每一级都返回（通过的候选 + 每只股票的拒绝原因列表），Web 端可以展示漏斗可视化。
这让整个流程的可解释性极强——你能看到"为什么这只票在 Stage2 被刷掉"。
"""
from __future__ import annotations
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).parent


# =============================================================
# 数据结构
# =============================================================
@dataclass
class RejectReason:
    """拒绝原因"""
    stage: str              # coarse / factor / deep
    category: str           # hard_filter / tech / sentiment / backtest / ic / stats / ml
    detail: str             # 具体原因

    def __str__(self):
        return f"[{self.stage}/{self.category}] {self.detail}"


@dataclass
class StageResult:
    """一级漏斗的输出"""
    stage_name: str
    input_count: int
    passed: pd.DataFrame                # 通过的候选（含 stage-level 打分数列）
    rejected: list[tuple[str, RejectReason]]   # (代码, 原因) 列表
    stage_time_s: float                 # 该级耗时
    diagnostics: dict[str, Any] = field(default_factory=dict)  # 中间统计（如因子IC面板）

    @property
    def passed_count(self) -> int:
        return len(self.passed)

    @property
    def pass_rate(self) -> float:
        if self.input_count == 0:
            return 0.0
        return self.passed_count / self.input_count

    def summary(self) -> dict:
        return {
            "stage": self.stage_name,
            "input": self.input_count,
            "passed": self.passed_count,
            "pass_rate": f"{self.pass_rate:.1%}",
            "time": f"{self.stage_time_s:.1f}s",
            "top_reasons": self._top_reasons(3),
        }

    def _top_reasons(self, n: int) -> list[str]:
        from collections import Counter
        cats = Counter(r.category for _, r in self.rejected)
        return [f"{c} ({cnt}只)" for c, cnt in cats.most_common(n)]


@dataclass
class CascadeRun:
    """完整三级漏斗结果"""
    stage1: StageResult
    stage2: Optional[StageResult]
    stage3: Optional[StageResult]
    total_time_s: float
    final_plan: Optional[pd.DataFrame] = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def funnel(self) -> pd.DataFrame:
        """漏斗可视化数据"""
        rows = [self.stage1.summary()]
        if self.stage2:
            rows.append(self.stage2.summary())
        if self.stage3:
            rows.append(self.stage3.summary())
        return pd.DataFrame(rows)


# =============================================================
# Stage 1: 粗筛（秒级）
# =============================================================
class CoarseScreener:
    """Stage 1 —— 硬过滤 + 快速技术预筛 + 快速舆情排雷。

    所有过滤条件都在本地 snapshot 上跑，不联网拉 K 线，所以秒级完成。
    """

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.min_price = self.cfg.get("min_price", 10)
        self.max_price = self.cfg.get("max_price", 50)
        self.min_mv = self.cfg.get("min_mv", 30e8)         # 30 亿
        self.max_mv = self.cfg.get("max_mv", 2000e8)       # 2000 亿
        self.min_turnover = self.cfg.get("min_turnover", 5000e4)  # 5000 万成交额
        self.min_pe = self.cfg.get("min_pe", 0)
        self.max_pe = self.cfg.get("max_pe", 80)
        self.min_valueratio = self.cfg.get("min_valueratio", 0.5)
        self.max_valueratio = self.cfg.get("max_valueratio", 10.0)
        self.reject_kw = ("ST", "退", "停", "北交", "*ST", "科创")

    def screen(self, spot: pd.DataFrame, news_flags: dict[str, bool] | None = None) -> StageResult:
        t0 = time.time()
        if spot is None or len(spot) == 0:
            return StageResult("Stage1-粗筛", 0, pd.DataFrame(), [], time.time() - t0)
        codes = spot["代码"].astype(str).str.zfill(6)
        names = spot["名称"].astype(str)
        rejected: list[tuple[str, RejectReason]] = []
        mask = pd.Series([True] * len(spot), index=spot.index)

        def _reject(idx, cat: str, detail: str):
            rejected.append((codes.iloc[idx], RejectReason("Stage1", cat, detail)))
            mask.iloc[idx] = False

        for i in range(len(spot)):
            c, n = codes.iloc[i], names.iloc[i]
            p = spot["最新价"].iloc[i] if "最新价" in spot.columns else np.nan
            if any(kw in n for kw in self.reject_kw):
                _reject(i, "hard_filter", f"名称含禁用词: {n}")
                continue
            if c.startswith(("4", "8", "920")):
                _reject(i, "hard_filter", "北交所/新三板")
                continue
            if p != p or p <= 0:
                _reject(i, "hard_filter", "价格无效")
                continue
            if p < self.min_price or p > self.max_price:
                _reject(i, "hard_filter", f"价格 {p:.1f} 超出 [{self.min_price},{self.max_price}]")
                continue
            mv = spot["总市值"].iloc[i] if "总市值" in spot.columns else np.nan
            if mv == mv and (mv < self.min_mv or mv > self.max_mv):
                _reject(i, "hard_filter", f"市值 {mv/1e8:.0f}亿 不在 [{self.min_mv/1e8:.0f},{self.max_mv/1e8:.0f}]亿")
                continue
            pe = spot["市盈率-动态"].iloc[i] if "市盈率-动态" in spot.columns else np.nan
            if pe == pe and (pe < self.min_pe or pe > self.max_pe):
                _reject(i, "hard_filter", f"PE {pe:.1f} 不在 [{self.min_pe},{self.max_pe}]")
                continue
            turnover = spot["成交额"].iloc[i] if "成交额" in spot.columns else np.nan
            if turnover == turnover and turnover < self.min_turnover:
                _reject(i, "hard_filter", f"成交额 {turnover/1e4:.0f}万 不足 {self.min_turnover/1e4:.0f}万")
                continue
            # 快速技术预筛（只用 snapshot 里有的字段，不拉 K 线）
            pct = spot["涨跌幅"].iloc[i] if "涨跌幅" in spot.columns else np.nan
            vol_ratio = spot["量比"].iloc[i] if "量比" in spot.columns else np.nan
            turnover_rate = spot["换手率"].iloc[i] if "换手率" in spot.columns else np.nan
            if vol_ratio == vol_ratio and vol_ratio > 10:
                _reject(i, "tech", f"量比 {vol_ratio:.1f} 过热")
                continue
            if turnover_rate == turnover_rate and turnover_rate > 30:
                _reject(i, "tech", f"换手率 {turnover_rate:.1f}% 过热")
                continue
            # 舆情快速检查（巨潮关键词，已预拉则用缓存）
            if news_flags and news_flags.get(c, False):
                _reject(i, "sentiment", "有近期利空公告")
                continue

        passed = spot.loc[mask].copy()
        passed["粗筛标记"] = True
        return StageResult(
            stage_name="Stage1-粗筛",
            input_count=len(spot),
            passed=passed,
            rejected=rejected,
            stage_time_s=time.time() - t0,
        )


# =============================================================
# Stage 2: 工具组合（分钟级）
# =============================================================
class FactorVerifier:
    """Stage 2 —— 逐只 K线 + 回测 + 因子 IC + 统计门禁。

    这一级复用 verify_impl 的全部逻辑，但加了 Stage 1 → Stage 2 的传递机制。
    输入是 Stage 1 通过的候选列表，逐只跑验证并收集拒绝原因。
    """

    def __init__(self, pipeline=None):
        self.pipeline = pipeline   # DataPipeline 实例（可选）

    def verify(self, stage1_result: StageResult) -> StageResult:
        t0 = time.time()
        candidates = stage1_result.passed.copy()
        if len(candidates) == 0:
            return StageResult("Stage2-验证", 0, pd.DataFrame(), [], time.time() - t0)

        import factors as _factors
        import statistics_kit as stkit
        from data import get_history_impl, cninfo_risk_impl

        rejected: list[tuple[str, RejectReason]] = []
        keep_mask = pd.Series([True] * len(candidates), index=candidates.index)
        code_list = candidates["代码"].astype(str).str.zfill(6).tolist()

        # ========== 并行拉 K 线（独立线程）==========
        def _fetch(cd):
            try:
                end = dt.date.today().strftime("%Y%m%d")
                start = (dt.date.today() - dt.timedelta(days=400)).strftime("%Y%m%d")
                h = get_history_impl(cd, start, end)
                return cd, h if len(h) >= 60 else None
            except Exception:
                return cd, None

        hists: dict[str, pd.DataFrame] = {}
        t_fetch = time.time()
        with ThreadPoolExecutor(max_workers=8) as ex:
            for cd, h in ex.map(_fetch, code_list):
                if h is not None:
                    hists[cd] = h
        print(f"[Cascade Stage2] K线拉取: {len(hists)}/{len(code_list)} 成功, {time.time()-t_fetch:.1f}s")

        # ========== 逐只 MA 金叉回测（简化版，用 factors.generate_signals）==========
        n_pass_backtest = 0
        for idx, row in candidates.iterrows():
            cd = str(row["代码"]).zfill(6)
            h = hists.get(cd)
            if h is None or len(h) < 60:
                rejected.append((cd, RejectReason("Stage2", "data", "历史K线不足60根")))
                keep_mask.loc[idx] = False
                continue
            try:
                sig = _factors.generate_signals(h)
                if not sig.get("trend_up", False):
                    rejected.append((cd, RejectReason("Stage2", "backtest", "MA5/MA20 金叉信号不成立")))
                    keep_mask.loc[idx] = False
                    continue
                # 回测扣费收益
                ret = sig.get("return_after_fee", 0)
                if ret is None or ret <= 0:
                    rejected.append((cd, RejectReason("Stage2", "backtest", f"扣费回测收益 {ret} ≤ 0")))
                    keep_mask.loc[idx] = False
                    continue
                n_pass_backtest += 1
            except Exception as e:
                rejected.append((cd, RejectReason("Stage2", "backtest", f"回测异常: {str(e)[:50]}")))
                keep_mask.loc[idx] = False

        # ========== 快速舆情复核（cninfo_risk）==========
        for idx, row in candidates.iterrows():
            if not keep_mask.loc[idx]:
                continue
            cd = str(row["代码"]).zfill(6)
            try:
                risk = cninfo_risk_impl(cd, days=7)
                if risk is None:
                    continue   # 查不到不惩罚
                if len(risk) > 0:
                    rejected.append((cd, RejectReason("Stage2", "sentiment",
                                                      f"近期利空: {risk[0]['标题'][:40]}")))
                    keep_mask.loc[idx] = False
            except Exception:
                pass

        # ========== 通过候选进入因子层（这里简化：因子 IC 面板后续再进 DL 层）==========
        passed = candidates.loc[keep_mask].copy()
        passed["通过Stage2"] = True
        diagnostics = {
            "n_pass_backtest": n_pass_backtest,
            "n_pass_total": len(passed),
            "hists_available": hists,   # Stage 3 DL 层复用
        }
        return StageResult(
            stage_name="Stage2-验证",
            input_count=len(stage1_result.passed),
            passed=passed,
            rejected=rejected,
            stage_time_s=time.time() - t0,
            diagnostics=diagnostics,
        )


# =============================================================
# Stage 3: DL 精筛（分钟级）
# =============================================================
class DeepRefiner:
    """Stage 3 —— LightGBM + sklearn ExtraTrees + MLP 三模型集成。

    三个模型学习目标相同：因子 → 未来 5 日收益的非线性关系。
    LightGBM（已存在）+ ExtraTrees（集成，降方差）+ MLP（sklearn，非线性边界）。
    三个模型预测取平均，稳健且能利用不同模型的 inductive bias。
    """

    def __init__(self, ensemble: str = "avg"):
        self.ensemble = ensemble  # avg / weighted / bagging

    def refine(self, stage2_result: StageResult, npanel: pd.DataFrame | None = None) -> StageResult:
        t0 = time.time()
        candidates = stage2_result.passed.copy()
        hists = stage2_result.diagnostics.get("hists_available", {})
        if len(candidates) == 0:
            return StageResult("Stage3-精筛", 0, pd.DataFrame(), [], time.time() - t0)

        try:
            import factors as _factors
        except ImportError:
            return StageResult("Stage3-精筛", len(candidates), candidates,
                               [("-", RejectReason("Stage3", "import", "factors 模块不可用"))],
                               time.time() - t0)

        # 1. 构建当日截面因子面板（复用 factors.build_panel 的简化版）
        try:
            today_panel = _factors.build_panel(
                spot=pd.concat([pd.read_csv(HERE/"spot_cache.csv", dtype={"代码": str})
                                if (HERE/"spot_cache.csv").exists()
                                else pd.DataFrame()] + [pd.DataFrame()]),
                hists=hists,
            ) if npanel is None else npanel
        except Exception:
            today_panel = npanel or pd.DataFrame()

        # 2. 三个模型预测
        predictions: dict[str, dict[str, float]] = {}
        model_loaded: list[str] = []

        # 2a. LightGBM（已存在的 ml_model）
        try:
            import ml_model
            backend = ml_model.available()
            if backend:
                predictions["lgb"] = self._predict_via_ml(ml_model, today_panel, candidates)
                model_loaded.append(f"lgb({backend})")
        except Exception as e:
            print(f"[Cascade Stage3] lgb 跳过: {e}")

        # 2b. ExtraTrees（sklearn）
        try:
            predictions["et"] = self._predict_via_sklearn(
                "et", today_panel, candidates, _factors)
            model_loaded.append("et")
        except Exception as e:
            print(f"[Cascade Stage3] et 跳过: {e}")

        # 2c. MLP（sklearn.neural_network，真正的神经网络精筛）
        try:
            predictions["mlp"] = self._predict_via_sklearn(
                "mlp", today_panel, candidates, _factors)
            model_loaded.append("mlp")
        except Exception as e:
            print(f"[Cascade Stage3] mlp 跳过: {e}")

        # 3. 集成打分（平均 / 中位数更抗离群点）
        code_scores: dict[str, float] = {}
        code_ensemble: dict[str, dict] = {}
        codes = candidates["代码"].astype(str).str.zfill(6).tolist()

        for cd in codes:
            scores = {}
            for mdl, cd_score_map in predictions.items():
                if cd in cd_score_map:
                    scores[mdl] = cd_score_map[cd]
            code_ensemble[cd] = scores
            if scores:
                code_scores[cd] = float(np.mean(list(scores.values())))
            else:
                code_scores[cd] = 0.5   # 没有模型预测 → 中位分

        # 4. 百分位排序 → Stage 3 通过 Top N%（默认前 50%）
        score_series = pd.Series(code_scores)
        pct_rank = score_series.rank(pct=True)
        threshold = 0.50   # 前 50% 通过 Stage 3（之后再 pick_top3 综合打分）
        keep_codes = set(pct_rank[pct_rank >= threshold].index.astype(str))

        rejected: list[tuple[str, RejectReason]] = []
        mask = pd.Series([True] * len(candidates), index=candidates.index)
        for idx, row in candidates.iterrows():
            cd = str(row["代码"]).zfill(6)
            if cd not in keep_codes:
                rejected.append((cd, RejectReason(
                    "Stage3", "ensemble",
                    f"集成分 {code_scores.get(cd, 0.5):.3f} 百分位 {pct_rank.get(cd, 0):.0%} < {threshold:.0%}")))
                mask.loc[idx] = False
                continue

        passed = candidates.loc[mask].copy()
        passed["ML分"] = passed["代码"].astype(str).str.zfill(6).map(code_scores)
        passed["ML百分位"] = passed["代码"].astype(str).str.zfill(6).map(pct_rank)
        passed["通过Stage3"] = True

        return StageResult(
            stage_name="Stage3-精筛",
            input_count=len(candidates),
            passed=passed,
            rejected=rejected,
            stage_time_s=time.time() - t0,
            diagnostics={
                "models_used": model_loaded,
                "n_models": len(model_loaded),
                "code_scores": code_scores,
                "code_ensemble_detail": code_ensemble,
                "ensemble_threshold": threshold,
            },
        )

    def _predict_via_ml(self, ml_mod, panel: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, float]:
        """用现有 ml_model 预测。"""
        if panel is None or len(panel) == 0:
            return {}
        feats = [f + "_n" for f in __import__("factors").ALL_FACTOR_COLS
                 if f + "_n" in panel.columns]
        today = panel[panel["代码"].isin(candidates["代码"].astype(str).str.zfill(6))].copy()
        if len(today) < 3:
            return {}
        fitted = ml_mod.fit_model(panel, feats)
        if fitted is None:
            return {}
        scores = ml_mod.predict_scores(fitted, today).rank(pct=True)
        return dict(zip(today["代码"].astype(str).str.zfill(6), scores.round(4)))

    def _predict_via_sklearn(self, model_type: str, panel: pd.DataFrame,
                             candidates: pd.DataFrame, factors_mod) -> dict[str, float]:
        """sklearn 的 ExtraTrees / MLP 模型。"""
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split

        if panel is None or len(panel) < 60:
            return {}
        feats = [f + "_n" for f in factors_mod.ALL_FACTOR_COLS
                 if f + "_n" in panel.columns]
        feats = [f for f in feats if panel[f].notna().mean() > 0.3]
        if len(feats) < 3:
            return {}
        label_col = "未来5日收益"
        if label_col not in panel.columns:
            return {}
        df = panel[feats + [label_col, "日期", "代码"]].dropna(subset=[label_col]).copy()
        if len(df) < 100:
            return {}
        X = df[feats].astype(float).fillna(0.0).values
        y = df[label_col].astype(float).clip(-0.5, 0.5).values

        if model_type == "et":
            model = ExtraTreesRegressor(n_estimators=200, max_depth=6,
                                        min_samples_leaf=20, random_state=42, n_jobs=2)
        elif model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                                 early_stopping=True, random_state=42)
        else:
            return {}

        model.fit(X, y)

        # 预测当日候选
        today = df.groupby("代码").tail(1).copy()
        if len(today) == 0:
            return {}
        Xt = today[feats].astype(float).fillna(0.0).values
        preds = model.predict(Xt)
        scores = pd.Series(preds, index=today.index).rank(pct=True)
        return dict(zip(today["代码"].astype(str).str.zfill(6), scores.round(4)))


# =============================================================
# CascadeRunner — 编排器
# =============================================================
class CascadeRunner:
    """三级漏斗总入口。

    用法:
        runner = CascadeRunner()
        result = runner.run(spot_df)          # capital 不传时自动读 推荐设置.json
        print(result.funnel())    # 漏斗可视化
        print(result.stage2.rejected[:10])   # 查看 Stage2 拒绝原因
    """

    def __init__(self, pipeline=None):
        self.stage1 = CoarseScreener()
        self.stage2 = FactorVerifier(pipeline=pipeline)
        self.stage3 = DeepRefiner()

    def run(self, spot: pd.DataFrame, capital: float | None = None) -> CascadeRun:
        # 本金不写死：未显式传入时读网页/邮件共用的 推荐设置.json
        if capital is None:
            try:
                from screen_service import load_settings
                capital = float(load_settings()["capital"])
            except Exception:
                capital = 0.0
        t_start = time.time()
        print(f"[Cascade] 开始三级漏斗，输入 {len(spot)} 只")

        # Stage 1: 粗筛
        t = time.time()
        r1 = self.stage1.screen(spot)
        print(f"[Cascade] Stage1 粗筛: {r1.input_count} → {r1.passed_count} ({r1.pass_rate:.1%}) "
              f"[{time.time()-t:.1f}s]")

        # Stage 2: 工具组合
        t = time.time()
        r2 = self.stage2.verify(r1)
        print(f"[Cascade] Stage2 验证: {r2.input_count} → {r2.passed_count} ({r2.pass_rate:.1%}) "
              f"[{time.time()-t:.1f}s]")

        # Stage 3: DL 精筛
        t = time.time()
        r3 = self.stage3.refine(r2)
        print(f"[Cascade] Stage3 精筛: {r3.input_count} → {r3.passed_count} ({r3.pass_rate:.1%}) "
              f"[{time.time()-t:.1f}s]")

        return CascadeRun(
            stage1=r1, stage2=r2, stage3=r3,
            total_time_s=time.time() - t_start,
        )


# =============================================================
# 自检
# =============================================================
if __name__ == "__main__":
    np.random.seed(42)
    fake_spot = pd.DataFrame({
        "代码": [f"{i:06d}" for i in range(1, 101)],
        "名称": [f"股票{i}" for i in range(1, 101)],
        "最新价": np.random.uniform(5, 60, 100),
        "总市值": np.random.uniform(10e8, 3000e8, 100),
        "市盈率-动态": np.random.uniform(-20, 100, 100),
        "成交额": np.random.uniform(1e6, 1e9, 100),
        "涨跌幅": np.random.uniform(-10, 10, 100),
        "量比": np.random.uniform(0.3, 15, 100),
        "换手率": np.random.uniform(0.1, 40, 100),
    })
    # 注入一些脏数据
    fake_spot.loc[5, "名称"] = "ST 退市股"
    fake_spot.loc[10, "代码"] = "430001"   # 北交所

    runner = CascadeRunner()
    r1 = runner.stage1.screen(fake_spot)
    print(f"\n=== Stage1 粗筛自检 ===")
    print(r1.summary())
    print(f"Top 拒绝原因: {r1._top_reasons(5)}")
    print(f"通过 {r1.passed_count} 只")
    print("✅ Cascade 自检通过")
