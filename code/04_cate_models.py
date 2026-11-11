# -*- coding: utf-8 -*-
"""
任务 B：异质性治疗效应（CATE / ITE）—— Causal Forest 与 R-learner
对应：研究规格基线_v1.2.md 第 4 节；外部验证按 hospitalid 分组。

硬性要求（PI 裁决）
------------------
- GroupKFold(n_splits=5, groups=hospitalid)：按 208 家医院分组交叉拟合，
  【同一医院绝不跨 train/test】，防止跨院数据泄漏。
- 交叉拟合的 CATE 用于 Qini / cARD / C-for-benefit，禁止用训练内预测自证。

实现说明
--------
- 主模型：econml.dml.CausalForestDML（本地已安装）。
- 对照模型：R-learner 以 R-loss 等价形式自实现（Ŷ、ê 由交叉拟合的
  XGBoost 给出，再以 pseudo-outcome 拟合效应模型）；econml 不同版本对
  RLearner 类的命名不一致，自实现可避免 API 猜测。
- 若 econml 不可用，明确降级为 sklearn 的 T-learner 并在返回值中标注，
  【不静默假装跑了 Causal Forest】。
- 观察性数据无真实 ITE：所有评估指标基于观测结局与交叉拟合 CATE 的排序，
  属于 uplift 评估，不是「真实验证」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from model_config import (
    ID_COL, HOSPITAL_COL, TREAT_COL, OUTCOME_COL, STEP_COL,
    N_SPLITS, RANDOM_SEED, BASELINE_COVARIATES, OUTPUT_DIR,
    DOSE_ASSUMED_FLAG, DOSE_SOURCE_COL, EARLY_WINDOW_MIN,
)


def aggregate_to_patient_level(df_long: pd.DataFrame) -> pd.DataFrame:
    """把一人一步的长表聚合为一人一行（CATE 建模的标准输入）。"""
    # PI 2026-11-09 裁决：早期启动窗口由 12h 扩展为 24h（EARLY_WINDOW_MIN）。
    # 用原始时点计算「首次启动 DEX 距 Time Zero 的分钟数」，比依赖固定 12h 标志更灵活。
    spec = {
        "Y": (OUTCOME_COL, "max"),
        HOSPITAL_COL: (HOSPITAL_COL, "first"),
        "_dex_first": ("dex_first_start_offset", "first"),
        "_t0": ("t0_offset", "first"),
    }
    for c in BASELINE_COVARIATES:
        if c in df_long.columns and c not in spec:
            spec[c] = (c, "first")
    out = df_long.groupby(ID_COL, as_index=False).agg(**spec)
    dt = (pd.to_numeric(out["_dex_first"], errors="coerce")
          - pd.to_numeric(out["_t0"], errors="coerce"))
    out["T"] = ((dt.notna()) & (dt <= EARLY_WINDOW_MIN)).astype(int)
    out = out.drop(columns=["_dex_first", "_t0"])
    return out


def _feature_matrix(df: pd.DataFrame):
    cols = [c for c in BASELINE_COVARIATES
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
    return X, cols


def _fit_on_fold(X_tr, y_tr, t_tr, model: str, seed: int):
    """在训练折上拟合一个效应模型，返回 (predict 函数, 实际使用的模型名)。"""
    if model == "causal_forest":
        try:
            from econml.dml import CausalForestDML
            est = CausalForestDML(
                discrete_treatment=True,
                n_estimators=160,      # 必须能被 subforest_size(默认 4) 整除
                subforest_size=4,
                min_samples_leaf=10,
                cv=3,
                random_state=seed,
            )
            est.fit(y_tr, t_tr, X=X_tr)
            return (lambda X: np.asarray(est.const_marginal_effect(X)).ravel()), "CausalForestDML"
        except Exception as e:
            print("  [WARN] CausalForestDML unavailable (%s: %s); fallback T-learner"
                  % (type(e).__name__, str(e)[:120]))
    else:
        try:
            from xgboost import XGBRegressor, XGBClassifier
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=True, random_state=seed)
            yhat = np.zeros(len(y_tr))
            ehat = np.zeros(len(t_tr))
            for tr, te in kf.split(X_tr):
                my = XGBRegressor(n_estimators=120, max_depth=3, random_state=seed,
                                  n_jobs=1, verbosity=0)
                mt = XGBClassifier(n_estimators=120, max_depth=3, random_state=seed,
                                   n_jobs=1, eval_metric="logloss", verbosity=0)
                my.fit(X_tr[tr], y_tr[tr])
                mt.fit(X_tr[tr], t_tr[tr])
                yhat[te] = my.predict(X_tr[te])
                ehat[te] = np.clip(mt.predict_proba(X_tr[te])[:, 1], 0.05, 0.95)
            resid_y, resid_t = y_tr - yhat, t_tr - ehat
            denom = np.maximum(resid_t ** 2, 1e-6)
            pseudo = resid_y / resid_t
            weights = resid_t ** 2
            mr = XGBRegressor(n_estimators=150, max_depth=3, random_state=seed,
                              n_jobs=1, verbosity=0)
            mr.fit(X_tr, pseudo, sample_weight=weights)
            return (lambda X: mr.predict(X)), "R-learner(R-loss, XGBoost)"
        except Exception as e:
            print("  [WARN] R-learner unavailable (%s: %s); fallback T-learner"
                  % (type(e).__name__, str(e)[:120]))

    # 明确标注的降级路径
    from sklearn.ensemble import GradientBoostingRegressor
    m1 = GradientBoostingRegressor(random_state=seed, n_estimators=80)
    m0 = GradientBoostingRegressor(random_state=seed, n_estimators=80)
    if (t_tr == 1).sum() > 2:
        m1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
    if (t_tr == 0).sum() > 2:
        m0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    return (lambda X: m1.predict(X) - m0.predict(X)), "T-learner(GBR, degraded)"


def cross_fitted_cate(df_patient: pd.DataFrame,
                      model: str = "causal_forest",
                      n_splits: int = N_SPLITS,
                      seed: int = RANDOM_SEED) -> dict:
    """
    按 hospitalid 做 GroupKFold 分组交叉拟合，返回折外 CATE。
    """
    X, cols = _feature_matrix(df_patient)
    y = pd.to_numeric(df_patient["Y"], errors="coerce").fillna(0).to_numpy(dtype=float)
    t = pd.to_numeric(df_patient["T"], errors="coerce").fillna(0).to_numpy(dtype=int)
    groups = df_patient[HOSPITAL_COL].to_numpy() if HOSPITAL_COL in df_patient else np.arange(len(y))

    cate = np.full(len(y), np.nan)
    used = []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        predict, name = _fit_on_fold(X[tr], y[tr], t[tr], model, seed + k)
        cate[te] = predict(X[te])
        used.append(name)
        g_tr = set(np.asarray(groups)[tr])
        g_te = set(np.asarray(groups)[te])
        assert g_tr.isdisjoint(g_te), "group leakage detected"
        print("  fold %d: train=%d test=%d hospitals_train=%d hospitals_test=%d model=%s"
              % (k + 1, len(tr), len(te), len(g_tr), len(g_te), name))

    return {
        "cate": cate,
        "y": y, "t": t,
        "hospital": np.asarray(groups),
        "feature_names": cols,
        "X": X,
        "model_used": sorted(set(used)),
        "n_splits": n_splits,
        "group_leakage": False,
    }


def subset_validation_report(df, out_path=None) -> dict:
    """方案 B 校验报告（CATE 侧）：剔除 ml/hr 估算记录后分组交叉拟合是否仍可行。

    输入可为长表或病人级表；缺 is_assumed_mlhr 列时如实标注 blocked。
    """
    import json
    from datetime import datetime
    from pathlib import Path
    from sklearn.model_selection import GroupKFold

    rep = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "方案 B 敏感性分析：剔除 ml/hr 默认 4.0 估算记录后 CATE 子集可行性",
        "spec": "研究规格基线_v1.2.md + PI 2026-11-09 方案 C 裁决",
        "flag_column": DOSE_ASSUMED_FLAG,
    }
    if DOSE_ASSUMED_FLAG not in df.columns:
        rep["status"] = "blocked_missing_flag"
        rep["message"] = (
            "长表缺少 is_assumed_mlhr 列，方案 B 无法精确剔除；"
            "报告不冒充已完成剔除。追加方式见 sql_dose_flag_patch.sql。"
        )
        rep["rows_supplied"] = int(len(df))
    else:
        flag = df[DOSE_ASSUMED_FLAG].fillna(False).astype(bool)
        sub_raw = df[~flag]
        full_p = (aggregate_to_patient_level(df) if "Y" not in df.columns
                  and STEP_COL in df.columns and OUTCOME_COL in df.columns else df)
        sub_p = (aggregate_to_patient_level(sub_raw) if "Y" not in sub_raw.columns
                 and STEP_COL in sub_raw.columns and OUTCOME_COL in sub_raw.columns else sub_raw)

        def brief(d):
            return {
                "patients": int(len(d)),
                "hospitals": int(d[HOSPITAL_COL].nunique()) if HOSPITAL_COL in d.columns else None,
                "events": int(d["Y"].sum()) if "Y" in d.columns else None,
                "treated": int(d["T"].sum()) if "T" in d.columns else None,
                "treatment_prevalence": round(float(d["T"].mean()), 4) if "T" in d.columns else None,
            }

        rep["status"] = "ok"
        rep["full"] = brief(full_p)
        rep["subset_excluding_assumed"] = brief(sub_p)
        rep["removed_source_rows"] = int(flag.sum())
        if DOSE_SOURCE_COL in df.columns:
            rep["concentration_source_counts"] = (
                df[DOSE_SOURCE_COL].fillna("NA").value_counts().to_dict()
            )
        # GroupKFold 在子集上是否仍可做（每个训练折需两类处理都存在）
        if HOSPITAL_COL in sub_p.columns and "T" in sub_p.columns and len(sub_p) >= N_SPLITS:
            gkf = GroupKFold(n_splits=N_SPLITS)
            folds = []
            X = np.zeros((len(sub_p), 1))
            groups = sub_p[HOSPITAL_COL].to_numpy()
            for tr, te in gkf.split(X, groups=groups):
                folds.append({
                    "train_n": int(len(tr)), "test_n": int(len(te)),
                    "train_has_both_classes": bool(len(set(sub_p["T"].iloc[tr])) == 2),
                })
            rep["groupkfold_feasibility"] = {
                "n_splits": N_SPLITS, "folds": folds,
                "all_folds_ok": bool(all(f["train_has_both_classes"] for f in folds)),
            }
        else:
            rep["groupkfold_feasibility"] = {"status": "insufficient_data_or_columns"}

    dst = Path(out_path) if out_path else OUTPUT_DIR / "subset_validation_04.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    rep["report_path"] = str(dst)
    return rep


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--input", default=None, help="patient-level CSV")
    ap.add_argument("--model", default="causal_forest",
                    choices=["causal_forest", "rlearner"])
    ap.add_argument("--output", default=str(OUTPUT_DIR / "cate_oof.csv"))
    ap.add_argument("--subset-report", default=None,
                    help="方案 B 子集校验报告路径；默认 outputs/subset_validation_04.json")
    a = ap.parse_args()
    if a.selfcheck or not a.input:
        from importlib import import_module
        syn = import_module("03b_gformula_py").make_synthetic_long(n_patients=400, n_steps=12)
        df = aggregate_to_patient_level(syn)
        print("synthetic patients:", len(df), "hospitals:", df[HOSPITAL_COL].nunique())
        res = cross_fitted_cate(df, model=a.model, n_splits=5)
        print(json.dumps({
            "model_used": res["model_used"],
            "n_splits": res["n_splits"],
            "cate_nan": int(np.isnan(res["cate"]).sum()),
            "cate_mean": float(np.nanmean(res["cate"])),
            "cate_sd": float(np.nanstd(res["cate"])),
            "group_leakage": res["group_leakage"],
        }, ensure_ascii=False, indent=2))
    else:
        df = pd.read_csv(a.input)
        # 方案 B：在聚合之前，用原始长表生成子集校验报告
        sub_rep = subset_validation_report(df, a.subset_report)
        print("  subset validation (方案 B):", sub_rep["status"],
              "| report:", sub_rep["report_path"])
        # 自动接轨：传入的可能是 SQL 骨架产出的长表（一人多步），
        # 也可能是已聚合的病人级表；长表先聚合，避免命令行直接报错。
        if "Y" not in df.columns or "T" not in df.columns:
            if STEP_COL in df.columns and OUTCOME_COL in df.columns:
                print("  detected LONG table -> aggregate to patient level")
                df = aggregate_to_patient_level(df)
            else:
                raise SystemExit(
                    "输入既非病人级表(需 Y/T 列)，也非长表(需 %s/%s 列)"
                    % (STEP_COL, OUTCOME_COL)
                )
        res = cross_fitted_cate(df, model=a.model)
        out = pd.DataFrame({"cate": res["cate"], "Y": res["y"], "T": res["t"],
                            HOSPITAL_COL: res["hospital"]})
        # 保留特征矩阵，供 05_plots.py 的 SHAP 图使用
        for i, name in enumerate(res["feature_names"]):
            out[name] = res["X"][:, i]
        out.to_csv(a.output, index=False, encoding="utf-8-sig")
        print("written:", a.output, "rows:", len(out),
              "features:", len(res["feature_names"]))
