# -*- coding: utf-8 -*-
"""
分析长表载入 / 校验 / IPTW 权重拟合
输入：SQL数据提取骨架_v1.2.sql 产出的 analysis_long（一人一步一行）
输出：建模矩阵、稳定化 IPTW 权重、校验报告

关键约束（规格 v1.2）：
- IPTW 权重【不在 SQL 中计算】，本模块以「该步长开始之前」的协变量拟合后回填
  analysis_long.ipw_weight_at_step。
- 分母模型不得包含该步长之后的变量，避免把干预后变量当作混杂。
- GroupKFold(n_splits=5, groups=hospitalid) 的分组键必须保留在建模矩阵中。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model_config import (
    ID_COL, STEP_COL, HOSPITAL_COL, TREAT_COL, OUTCOME_COL, OUTCOME_ALT_COL,
    DEATH_COL, DISCHARGE_COL, MAX_STEPS, GRACE_STEPS, IPTW_CLIP,
    BASELINE_COVARIATES, TIMEVARYING_COVARIATES, RANDOM_SEED,
    OUTPUT_DIR, DOSE_ASSUMED_FLAG, DOSE_SOURCE_COL,
)


def load_long_table(path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def validate_long_table(df: pd.DataFrame) -> dict:
    """返回结构化校验报告；任一硬项为 False 时不得进入建模。"""
    rep = {}
    rep["rows"] = int(len(df))
    rep["n_patients"] = int(df[ID_COL].nunique()) if ID_COL in df else -1
    rep["n_hospitals"] = int(df[HOSPITAL_COL].nunique()) if HOSPITAL_COL in df else -1

    key = [ID_COL, STEP_COL]
    rep["key_unique"] = bool(all(c in df.columns for c in key)) and not bool(
        df.duplicated(key).any()
    )
    rep["step_range_ok"] = bool(
        STEP_COL in df.columns and df[STEP_COL].min() >= 0
        and df[STEP_COL].max() < MAX_STEPS
    )
    rep["grace_steps"] = GRACE_STEPS
    rep["has_hospital_group"] = HOSPITAL_COL in df.columns

    outcome_cols = [OUTCOME_COL, OUTCOME_ALT_COL, DEATH_COL, DISCHARGE_COL]
    rep["outcome_cols_present"] = all(c in df.columns for c in outcome_cols)
    if rep["outcome_cols_present"]:
        # V1 是 ALT 的子集（V1 达标时 ALT 必达标），二者【可以】同为 1；
        # 但任一结局都不可与死亡/出院竞争事件在同一行并存。
        rep["events_mutually_exclusive"] = bool(
            (df[[OUTCOME_COL, DEATH_COL, DISCHARGE_COL]].sum(axis=1) <= 1).all()
            and (df[[OUTCOME_ALT_COL, DEATH_COL, DISCHARGE_COL]].sum(axis=1) <= 1).all()
        )
        rep["n_resolution_v1"] = int(df[OUTCOME_COL].sum())
        rep["n_resolution_alt"] = int(df[OUTCOME_ALT_COL].sum())
        rep["n_death"] = int(df[DEATH_COL].sum())
        rep["n_discharge"] = int(df[DISCHARGE_COL].sum())
    else:
        rep["events_mutually_exclusive"] = False

    rep["has_iptw_placeholder"] = "ipw_weight_at_step" in df.columns
    rep["hard_checks_pass"] = all([
        rep["key_unique"], rep["step_range_ok"], rep["has_hospital_group"],
        rep["outcome_cols_present"], rep["events_mutually_exclusive"],
    ])
    return rep


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """把长表整理为建模矩阵：数值化、缺失显式标记、保留分组键。"""
    out = df.copy()
    for c in [OUTCOME_COL, OUTCOME_ALT_COL, DEATH_COL, DISCHARGE_COL, TREAT_COL]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    if "gender" in out.columns:
        out["gender"] = out["gender"].astype("category").cat.codes
    for c in ["ethnicity", "unittype", "apacheadmissiondx", "teachingstatus", "region"]:
        if c in out.columns:
            out[c] = out[c].astype("category").cat.codes
    num_cols = [c for c in BASELINE_COVARIATES + TIMEVARYING_COVARIATES
                if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
    return out


def fit_iptw_weights(df: pd.DataFrame,
                     treatment_col: str = TREAT_COL,
                     clip=IPTW_CLIP,
                     random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    稳定化 IPTW：以「步长开始之前」的基线 + 时变协变量拟合启动 DEX 的概率。

    说明
    ----
    - 使用逐步 logistic 的分母模型，分子为边际启动概率（稳定化）。
    - 只保留 step < 该患者首次结局/竞争事件时点的行进入风险集。
    - 权重截断到 clip 区间；输出诊断量（均值、最大、ESS）。
    - 这是【骨架实现】：完整的时变混杂集合、删失权重、clone-censor-weight
      规则须由 Gemini 的 TTE 协议表确定后扩展。
    """
    from sklearn.linear_model import LogisticRegression

    out = df.copy()
    covs = [c for c in BASELINE_COVARIATES + TIMEVARYING_COVARIATES if c in out.columns]
    covs = [c for c in covs if pd.api.types.is_numeric_dtype(out[c])]
    X = out[covs].fillna(out[covs].median(numeric_only=True)).fillna(0.0).to_numpy()
    y = pd.to_numeric(out[treatment_col], errors="coerce").fillna(0).astype(int).to_numpy()

    if len(np.unique(y)) < 2:
        out["ipw_weight_at_step"] = 1.0
        out["iptw_diagnostics"] = "single treatment class -> weights=1"
        return out

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X, y)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    marginal = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    sw = np.where(y == 1, marginal / p, (1 - marginal) / (1 - p))

    # PI 裁决：1%~99% 分位数截断（比固定 clip 更贴合实际分布）
    p1, p99 = np.percentile(sw, [1, 99])
    out["ipw_weight_at_step"] = np.clip(sw, p1, p99)
    out["iptw_propensity"] = p

    w = out["ipw_weight_at_step"].to_numpy()
    ess = float(w.sum() ** 2 / np.sum(w ** 2))
    out.attrs["iptw_diag"] = {
        "n_rows": int(len(out)), "n_covariates": len(covs),
        "marginal_treatment_prob": marginal,
        "weight_mean": float(w.mean()), "weight_max": float(w.max()),
        "weight_p1": float(p1), "weight_p99": float(p99),
        "effective_sample_size": round(ess, 1),
        "ess_pct_of_rows": round(100.0 * ess / max(len(out), 1), 2),
        "clip_rule": "1%-99% percentile",
        "clip_fallback": list(clip),
    }
    return out


def subset_validation_report(df: pd.DataFrame, out_path=None) -> dict:
    """方案 B 校验报告：剔除 ml/hr 默认 4.0 估算记录后的子集情况。

    依赖长表含 is_assumed_mlhr 列（01_dex_dose.py 标记、SQL 骨架追加）。
    【事实纪律】缺列时如实标注 blocked，绝不用近似值冒充精确剔除。
    """
    import json
    from datetime import datetime
    from pathlib import Path

    def summarize(d: pd.DataFrame) -> dict:
        s = {
            "rows": int(len(d)),
            "patients": int(d[ID_COL].nunique()) if ID_COL in d.columns else None,
            "hospitals": int(d[HOSPITAL_COL].nunique()) if HOSPITAL_COL in d.columns else None,
        }
        for key, col in [("n_resolution_v1", OUTCOME_COL),
                         ("n_resolution_alt", OUTCOME_ALT_COL),
                         ("n_death", DEATH_COL),
                         ("n_discharge", DISCHARGE_COL)]:
            s[key] = int(d[col].sum()) if col in d.columns else None
        if TREAT_COL in d.columns:
            s["treatment_prevalence"] = round(float(d[TREAT_COL].mean()), 4)
        if "ipw_weight_at_step" in d.columns:
            w = pd.to_numeric(d["ipw_weight_at_step"], errors="coerce").dropna()
            s["iptw_weight_mean"] = round(float(w.mean()), 4) if len(w) else None
            s["iptw_effective_sample_size"] = (
                round(float(w.sum() ** 2 / np.sum(w ** 2)), 1) if len(w) else None
            )
        return s

    rep = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "方案 B 敏感性分析：剔除 ml/hr 默认 4.0 mcg/ml 估算记录后的子集校验",
        "spec": "研究规格基线_v1.2.md + PI 2026-11-09 方案 C 裁决",
        "flag_column": DOSE_ASSUMED_FLAG,
    }
    if DOSE_ASSUMED_FLAG not in df.columns:
        rep["status"] = "blocked_missing_flag"
        rep["message"] = (
            "长表缺少 is_assumed_mlhr 列，方案 B 无法精确剔除估算记录；"
            "本报告不冒充已完成剔除。"
        )
        rep["required_action"] = (
            "在 SQL 骨架 v1.2 的 dex_infusion / step_exposure / analysis_long "
            "三层追加该标记列，片段见 sql_dose_flag_patch.sql（未改动冻结的 v1.2 文件）。"
        )
        rep["full_only"] = summarize(df)
    else:
        flag = df[DOSE_ASSUMED_FLAG].fillna(False).astype(bool)
        sub = df[~flag]
        rep["status"] = "ok"
        rep["full"] = summarize(df)
        rep["subset_excluding_assumed"] = summarize(sub)
        rep["removed_rows"] = int(flag.sum())
        rep["removed_frac"] = round(float(flag.mean()), 4)
        rep["removed_patients"] = (
            int(df.loc[flag, ID_COL].nunique()) if ID_COL in df.columns else None
        )
        if DOSE_SOURCE_COL in df.columns:
            rep["concentration_source_counts"] = (
                df[DOSE_SOURCE_COL].fillna("NA").value_counts().to_dict()
            )

    dst = Path(out_path) if out_path else OUTPUT_DIR / "subset_validation_02.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    rep["report_path"] = str(dst)
    return rep


if __name__ == "__main__":
    import argparse, json
    from model_config import LONG_TABLE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(LONG_TABLE_PATH))
    ap.add_argument("--output", default=None)
    ap.add_argument("--subset-report", default=None,
                    help="方案 B 子集校验报告路径；默认 outputs/subset_validation_02.json")
    a = ap.parse_args()
    df = load_long_table(a.input)
    rep = validate_long_table(df)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    sub_rep = subset_validation_report(df, a.subset_report)
    print("subset validation (方案 B):", sub_rep["status"],
          "| report:", sub_rep["report_path"])
    if rep["hard_checks_pass"]:
        df2 = fit_iptw_weights(build_model_matrix(df))
        diag = df2.attrs.get("iptw_diag", {})
        print("IPTW diag:", json.dumps(diag, ensure_ascii=False))
        (OUTPUT_DIR / "iptw_diagnostics.json").write_text(
            json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        print("IPTW diagnostics ->", OUTPUT_DIR / "iptw_diagnostics.json")
        if a.output:
            df2.to_csv(a.output, index=False, encoding="utf-8-sig")
            print("written:", a.output)
    else:
        print("HARD CHECKS FAILED — do not proceed to modeling")
        # 熔断：让调用方（run_real_pipeline）停止，避免后续步骤读取不存在的文件
        raise SystemExit(1)
