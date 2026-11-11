# -*- coding: utf-8 -*-
"""
任务 B（图表）：顶刊标准可视化
对标：cARD 十分位图（Goligher 2023 JAMA）、Uplift Qini 曲线、
      C-for-benefit（Buell 2024 JAMA）、SHAP 特征重要性；
      另加 g-formula 反事实 CIF 曲线（竞争风险，禁 1-KM）。

重要方法学说明
--------------
1. 观察性数据【没有真实 ITE】。Qini / cARD / C-for-benefit 均基于
   「交叉拟合 CATE 排序 + 观测结局」，属于 uplift 评估，不是真实验证；
   绝不可写成「模型预测被真实 ITE 验证」。
2. SHAP 作用于【CATE 的代理回归树】，不是直接解释 Causal Forest 内部结点；
   这是因果森林可解释性的标准近似，图表注释中已标明。
3. 图表文字一律英文，避免中文字体缺失；数字仅为合成数据自检示例时，
   文件名带 SELFCHECK_ 前缀，不得用于论文。

运行：python 05_plots.py --selfcheck
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd

from model_config import (
    OUTPUT_DIR, FIG_DPI, FIG_SIZE, CARD_N_DECILES, QINI_N_QUANTILES,
    MAX_STEPS, GRID_STEP_MIN, HOSPITAL_COL, ID_COL,
)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": FIG_DPI, "savefig.bbox": "tight",
})

# 顶刊交付：位图预览 + 矢量图（SVG / PDF）；由 CLI --formats 覆盖
FORMATS = ("png", "svg", "pdf")


def _save(fig, out_path):
    """同一张图同时输出 PNG 与矢量格式，文件名前缀一致。"""
    p = Path(out_path)
    for fmt in FORMATS:
        fig.savefig(p.with_suffix("." + fmt), format=fmt)


def _clean(y, t, cate):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=int)
    cate = np.asarray(cate, dtype=float)
    m = np.isfinite(cate) & np.isfinite(y)
    return y[m], t[m], cate[m]


def plot_card_deciles(y, t, cate, out_path, n_deciles=CARD_N_DECILES):
    """条件绝对风险差（cARD）十分位图：按预测 CATE 分组后的观测风险差。"""
    y, t, cate = _clean(y, t, cate)
    order = np.argsort(-cate)
    groups = np.array_split(order, n_deciles)
    diffs, errs = [], []
    for g in groups:
        yt, yc = y[g][t[g] == 1], y[g][t[g] == 0]
        p1 = yt.mean() if len(yt) else np.nan
        p0 = yc.mean() if len(yc) else np.nan
        d = p1 - p0
        se = np.sqrt((p1 * (1 - p1) / max(len(yt), 1) +
                      p0 * (1 - p0) / max(len(yc), 1))) if np.isfinite(d) else np.nan
        diffs.append(d)
        errs.append(1.96 * se if np.isfinite(se) else 0.0)
    x = np.arange(1, n_deciles + 1)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x, np.nan_to_num(diffs), yerr=np.nan_to_num(errs), capsize=3,
           color="#4C72B0", edgecolor="black", linewidth=0.6, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("CATE decile (1 = highest predicted benefit)")
    ax.set_ylabel("Observed conditional absolute risk difference")
    ax.set_title("cARD by predicted benefit decile\n(observational uplift, not true ITE)")
    ax.set_xticks(x)
    _save(fig, out_path)
    plt.close(fig)
    return {"deciles": [None if not np.isfinite(d) else float(d) for d in diffs]}


def plot_qini(y, t, cate, out_path, n_points=QINI_N_QUANTILES):
    """Uplift Qini 曲线（简化累积增益实现）。"""
    y, t, cate = _clean(y, t, cate)
    order = np.argsort(-cate)
    y, t = y[order], t[order]
    cum_t = np.cumsum((t == 1) * y)
    cum_c = np.cumsum((t == 0) * y)
    n_t = np.cumsum(t == 1)
    n_c = np.cumsum(t == 0)
    qini = cum_t - cum_c * n_t / np.maximum(n_c, 1)
    frac = np.arange(1, len(y) + 1) / len(y)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(np.r_[0, frac], np.r_[0, qini], color="#C44E52", lw=2, label="Model")
    ax.plot([0, 1], [0, qini[-1]], "--", color="grey", lw=1.2, label="Random")
    ax.set_xlabel("Fraction of population targeted")
    ax.set_ylabel("Cumulative incremental gain (Qini)")
    ax.set_title("Uplift Qini curve\n(cross-fitted CATE; observational, not true ITE)")
    ax.legend(frameon=False)
    _save(fig, out_path)
    plt.close(fig)
    return {"qini_end": float(qini[-1])}


def plot_c_for_benefit(y, t, cate, out_path, n_deciles=CARD_N_DECILES):
    """C-for-benefit：高预测获益组与低预测获益组的观测获益分离度。"""
    y, t, cate = _clean(y, t, cate)
    order = np.argsort(-cate)
    groups = np.array_split(order, n_deciles)
    obs_benefit = []
    for g in groups:
        yt, yc = y[g][t[g] == 1], y[g][t[g] == 0]
        obs_benefit.append(yt.mean() - yc.mean() if len(yt) and len(yc) else np.nan)
    obs_benefit = np.array(obs_benefit, dtype=float)
    # 简化的 C-for-benefit 标量：高获益组观测获益是否高于低获益组
    hi = np.nanmean(obs_benefit[:2]) if np.isfinite(obs_benefit[:2]).any() else np.nan
    lo = np.nanmean(obs_benefit[-2:]) if np.isfinite(obs_benefit[-2:]).any() else np.nan
    c_for_benefit = float(np.mean(hi > lo)) if np.isfinite(hi) and np.isfinite(lo) else np.nan

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(np.arange(1, n_deciles + 1), obs_benefit, "o-", color="#55A868", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("CATE decile (1 = highest predicted benefit)")
    ax.set_ylabel("Observed benefit (treated - control)")
    ax.set_title("C-for-benefit: observed benefit across predicted deciles"
                 "\n(simplified; see Buell 2024 JAMA for full definition)")
    _save(fig, out_path)
    plt.close(fig)
    return {"c_for_benefit_simplified": c_for_benefit,
            "high_group_benefit": None if not np.isfinite(hi) else float(hi),
            "low_group_benefit": None if not np.isfinite(lo) else float(lo)}


def plot_shap_importance(X, cate, feature_names, out_path):
    """SHAP：先拟合 CATE 的代理树，再解释该树（不是直接解释因果森林）。"""
    X = np.asarray(X, dtype=float)
    cate = np.asarray(cate, dtype=float)
    m = np.isfinite(cate)
    X, cate = X[m], cate[m]
    from sklearn.tree import DecisionTreeRegressor
    surrogate = DecisionTreeRegressor(max_depth=4, random_state=0)
    surrogate.fit(X, cate)
    import shap
    explainer = shap.TreeExplainer(surrogate)
    sv = np.asarray(explainer.shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, 0]
    imp = np.abs(sv).mean(axis=0)
    names = list(feature_names)[:len(imp)]
    idx = np.argsort(imp)[::-1][:min(10, len(imp))]
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.barh([names[i] for i in idx][::-1], imp[idx][::-1], color="#8172B2",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("mean |SHAP value| (surrogate tree of CATE)")
    ax.set_title("Feature importance for predicted CATE\nSHAP on CATE surrogate tree, not on forest")
    _save(fig, out_path)
    plt.close(fig)
    return {"top_features": [(names[i], float(imp[i])) for i in idx]}


def plot_counterfactual_cif(res_gformula, out_path):
    """两策略反事实 CIF（竞争风险；禁用 1-KM）。"""
    c1 = np.asarray(res_gformula["cif_early_dex"], dtype=float)
    c0 = np.asarray(res_gformula["cif_no_early_dex"], dtype=float)
    days = np.arange(len(c1)) * GRID_STEP_MIN / 1440.0
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.step(days, c1, where="post", color="#C44E52", lw=2, label="Early DEX (12h)")
    ax.step(days, c0, where="post", color="#4C72B0", lw=2, label="No early DEX")
    ax.set_xlabel("Days from Time Zero")
    ax.set_ylabel("Cumulative incidence of sustained resolution")
    ax.set_title("Counterfactual CIF under competing risks\nParametric G-formula (NOT 1-KM)")
    ax.legend(frameon=False)
    _save(fig, out_path)
    plt.close(fig)
    return {"cif_early_end": float(c1[-1]), "cif_control_end": float(c0[-1])}


def run_all(res_cate: dict, res_gformula: dict | None = None,
            prefix: str = "", outdir: Path = OUTPUT_DIR) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    y, t, cate = res_cate["y"], res_cate["t"], res_cate["cate"]
    out = {}
    out["card"] = plot_card_deciles(y, t, cate, outdir / (prefix + "card_deciles.png"))
    out["qini"] = plot_qini(y, t, cate, outdir / (prefix + "qini_curve.png"))
    out["cforbenefit"] = plot_c_for_benefit(y, t, cate, outdir / (prefix + "c_for_benefit.png"))
    out["shap"] = plot_shap_importance(res_cate["X"], cate,
                                       res_cate["feature_names"],
                                       outdir / (prefix + "shap_importance.png"))
    if res_gformula is not None:
        out["cif"] = plot_counterfactual_cif(res_gformula,
                                             outdir / (prefix + "counterfactual_cif.png"))
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--from-files", action="store_true",
                    help="读取 cate_oof.csv + gformula_results.json 绘制真实数据图")
    ap.add_argument("--cate", default=str(OUTPUT_DIR / "cate_oof.csv"))
    ap.add_argument("--gformula", default=str(OUTPUT_DIR / "gformula_results.json"))
    ap.add_argument("--prefix", default="", help="输出文件名前缀；真实数据留空")
    ap.add_argument("--outdir", default=str(OUTPUT_DIR))
    ap.add_argument("--formats", default="png,svg,pdf",
                    help="输出格式，逗号分隔；矢量图用 svg / pdf")
    a = ap.parse_args()
    FORMATS = tuple(x.strip() for x in a.formats.split(",") if x.strip())

    if a.from_files:
        df = pd.read_csv(a.cate)
        needed = {"cate", "Y", "T"}
        if not needed.issubset(set(df.columns)):
            raise SystemExit("cate_oof.csv 缺少列: %s" % (needed - set(df.columns)))
        non_feature = {"cate", "Y", "T", HOSPITAL_COL, ID_COL}
        feats = [c for c in df.columns
                 if c not in non_feature and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feats].apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
        res_cate = {"y": df["Y"].to_numpy(dtype=float),
                    "t": df["T"].to_numpy(dtype=int),
                    "cate": df["cate"].to_numpy(dtype=float),
                    "X": X, "feature_names": feats}
        gf_path = Path(a.gformula)
        res_gf = json.loads(gf_path.read_text(encoding="utf-8")) if gf_path.exists() else None
        info = run_all(res_cate, res_gf, prefix=a.prefix, outdir=Path(a.outdir))
        print("real-data figures written to", a.outdir,
              "| gformula CIF included:", res_gf is not None,
              "| n_features:", len(feats))
        print(json.dumps({k: v for k, v in info.items() if k != "shap"},
                         ensure_ascii=False, indent=2))
    elif a.selfcheck:
        rng = np.random.default_rng(7)
        n = 600
        y = rng.binomial(1, 0.3, n).astype(float)
        t = rng.binomial(1, 0.45, n)
        cate = 0.15 * rng.normal(size=n) + 0.05 * (y - 0.3)
        X = rng.normal(size=(n, 6))
        names = ["age", "apache", "weight", "HR", "MAP", "sao2"]
        res = run_all({"y": y, "t": t, "cate": cate, "X": X, "feature_names": names},
                      prefix="SELFCHECK_")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        print("SELFCHECK plots: 4 PNG written to", OUTPUT_DIR)
