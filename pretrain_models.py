# -*- coding: utf-8 -*-
"""本地预训练脚本：用 wf_cache 全市场数据训练 LGB/ET/MLP，落盘到 models/ 供云端加载。

为什么需要预训练？
  云端 Streamlit Cloud 没有 wf_cache，只有当天候选股（~30只），样本太少导致
  ML 模型样本外 IC 为负（实测 -0.156）。本地有 372 只 × 近千日 = 27 万行数据，
  训练出的模型 IC 稳定在 +0.02~+0.03。把模型随仓库分发，云端只做推理。

用法：
  python pretrain_models.py
产物：
  models/lgb.joblib  models/et.joblib  models/mlp.joblib
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import factors
import ml_model
from cascade_runner import DeepRefiner, save_sklearn_model, MODEL_DIR


def main():
    t0 = time.time()
    print(f"[pretrain] 构建全市场训练面板（wf_cache）...")
    npanel = factors.build_training_panel()
    if npanel is None or len(npanel) == 0:
        print("[pretrain] 错误：wf_cache 为空，无法训练。请先运行数据管道收集K线。")
        return 1
    print(f"[pretrain] 面板规模: {len(npanel):,} 行, "
          f"{npanel['日期'].nunique()} 个交易日")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    refiner = DeepRefiner()

    # ---- LightGBM ----
    print("\n[pretrain] 训练 LightGBM ...")
    fitted = ml_model.fit_model(npanel)
    if fitted is None:
        print("[pretrain] LightGBM 训练失败（样本不足或后端不可用）")
    else:
        ic = fitted.get("oof_ic", float("nan"))
        print(f"[pretrain] LightGBM 样本外IC = {ic:.4f}  "
              f"{'通过' if fitted.get('oof_pass') else '未达标'}")
        if fitted.get("oof_pass"):
            path = ml_model.save_model(fitted, "lgb")
            print(f"[pretrain] 已保存: {path}")
        else:
            print("[pretrain] LightGBM IC 未达标，不保存（模型弃权）")

    # ---- ExtraTrees ----
    print("\n[pretrain] 训练 ExtraTrees ...")
    t = time.time()
    et_model, et_feats, et_ic = refiner.train_sklearn_model("et", npanel, factors)
    print(f"[pretrain] ExtraTrees 样本外IC = {et_ic:.4f}  耗时 {time.time()-t:.1f}s")
    if et_model is not None:
        path = save_sklearn_model(et_model, et_feats, et_ic, "et")
        print(f"[pretrain] 已保存: {path}")
    else:
        print("[pretrain] ExtraTrees IC 未达标，不保存")

    # ---- MLP ----
    print("\n[pretrain] 训练 MLP ...")
    t = time.time()
    mlp_model, mlp_feats, mlp_ic = refiner.train_sklearn_model("mlp", npanel, factors)
    print(f"[pretrain] MLP 样本外IC = {mlp_ic:.4f}  耗时 {time.time()-t:.1f}s")
    if mlp_model is not None:
        path = save_sklearn_model(mlp_model, mlp_feats, mlp_ic, "mlp")
        print(f"[pretrain] 已保存: {path}")
    else:
        print("[pretrain] MLP IC 未达标，不保存")

    # ---- 汇总 ----
    print("\n" + "=" * 50)
    print(f"[pretrain] 完成，总耗时 {time.time()-t0:.1f}s")
    print(f"[pretrain] 模型目录: {MODEL_DIR}")
    for p in sorted(MODEL_DIR.glob("*.joblib")):
        print(f"  {p.name}: {p.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
