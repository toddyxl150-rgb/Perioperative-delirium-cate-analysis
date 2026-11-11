# -*- coding: utf-8 -*-
"""
最终跨院验证：按医院规模手工分层 + 5 折留出集（替代失效的 GroupKFold）

为什么不用标准 GroupKFold
------------------------
标准 GroupKFold 只保证 group 不跨折，不保证折间规模均衡。本数据中有
单家医院患者量极大，导致 fold1 只有 1 家医院且处理组为 0、fold2 处理组零事件，
验证完全失效。

本脚本做法
----------
1. 统计每家医院的患者数 n 与 MV 患者数 n_mv；
2. 按 n_mv 降序排列，用「蛇形（snake）」轮流分配到 5 折，
   使每折的 MV 患者数与医院数尽量均衡；
3. 每折：在其余 4 折（train）拟合 logistic（Y ~ T + age + apache + male），
   在该折（test）上做 g-computation 得 RR / RD，并在 test 内 bootstrap 50 次给 CI；
4. 报告 5 折中「RR > 1 且 CI 不含 1」的比例。

输出
----
- 更新 outputs/final_validation_report.json（新增 stratified_groupkfold_mv 字段）
- outputs/subgroup_forest_final_annotated.png（稳健 vs 探索性标注）
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_config import OUTPUT_DIR, FIG_DPI, RANDOM_SEED

m7 = importlib.import_module("07_subgroups")
m8 = importlib.import_module("08_final_validation")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": FIG_DPI,
                     "savefig.bbox": "tight"})
COV = ("age", "apachescore", "male")


def snake_folds(hosp: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """按 MV 患者数降序做蛇形分配，使各折规模均衡。"""
    h = hosp.sort_values(["n_mv", "n"], ascending=False).reset_index(drop=True)
    fold = []
    for i in range(len(h)):
        cycle, pos = i // k, i % k
        fold.append(pos if cycle % 2 == 0 else (k - 1 - pos))
    h["fold"] = fold
    return h[["hospitalid", "fold"]]


def _mat(frame, cov):
    return np.column_stack([np.ones(len(frame)),
                            frame[["T"] + list(cov)].astype(float).to_numpy()])


def fit_predict_rr(train, test, cov=COV, n_boot=50, seed=RANDOM_SEED):
    tr = train.dropna(subset=["Y", "T"] + list(cov))
    te = test.dropna(subset=["Y", "T"] + list(cov))
    if len(tr) < 100 or tr["T"].nunique() < 2 or tr["Y"].nunique() < 2:
        return None
    if len(te) < 30 or te["T"].nunique() < 2:
        return None
    try:
        model = sm.Logit(tr["Y"].astype(float), _mat(tr, cov)).fit(disp=0)
    except Exception:
        return None

    def rr_of(frame):
        f1, f0 = frame.copy(), frame.copy()
        f1["T"], f0["T"] = 1, 0
        p1 = float(model.predict(_mat(f1, cov)).mean())
        p0 = float(model.predict(_mat(f0, cov)).mean())
        return (p1 / p0 if p0 > 0 else np.nan), (p1 - p0)

    rr, rd = rr_of(te)
    # bootstrap 必须同时重抽 train 与 test 并重新拟合，否则 CI 只反映 test 抽样波动、
    # 会被人为压窄（例如 1.481–1.485），不能作为显著性证据。
    rng = np.random.default_rng(seed)
    brr, brd = [], []
    for _ in range(n_boot):
        itr = rng.integers(0, len(tr), len(tr))
        ite = rng.integers(0, len(te), len(te))
        btr, bte = tr.iloc[itr], te.iloc[ite]
        if (btr["T"].nunique() < 2 or btr["Y"].nunique() < 2
                or bte["T"].nunique() < 2):
            continue
        try:
            bm = sm.Logit(btr["Y"].astype(float), _mat(btr, cov)).fit(disp=0)
        except Exception:
            continue
        f1, f0 = bte.copy(), bte.copy()
        f1["T"], f0["T"] = 1, 0
        q1 = float(bm.predict(_mat(f1, cov)).mean())
        q0 = float(bm.predict(_mat(f0, cov)).mean())
        if q0 > 0:
            brr.append(q1 / q0)
        brd.append(q1 - q0)

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 5 else [None, None]

    lo, hi = ci(brr)
    return {"n_test": int(len(te)), "n_treated_test": int(te["T"].sum()),
            "n_hospitals_test": int(te["hospitalid"].nunique()),
            "rr": float(rr), "rr_ci95": [lo, hi], "rd": float(rd),
            "rd_ci95": ci(brd),
            "significant_positive": bool(lo is not None and lo > 1.0)}


def main():
    g = m8.load_with_rass("resolution_v1_event")
    if "invasive_mv" not in g.columns:
        raise SystemExit("缺少 invasive_mv")
    mv = g[g["invasive_mv"] == 1].dropna(subset=["Y", "T", "hospitalid"]).copy()
    print("MV 患者:", len(mv), "| 医院:", mv["hospitalid"].nunique(),
          "| 处理组:", int(mv["T"].sum()))

    hosp = mv.groupby("hospitalid").agg(n=("Y", "size"), n_mv=("Y", "size"),
                                        n_treat=("T", "sum")).reset_index()
    folds = snake_folds(hosp, 5)
    mv = mv.merge(folds, on="hospitalid", how="left")
    print("\n各折规模（MV 患者 / 医院数 / 处理组）:")
    for k in range(5):
        d = mv[mv["fold"] == k]
        print("  fold %d: n=%-4d 院=%-3d T=%-3d" % (k + 1, len(d), d["hospitalid"].nunique(), int(d["T"].sum())))

    results = []
    for k in range(5):
        tr = mv[mv["fold"] != k]
        te = mv[mv["fold"] == k]
        r = fit_predict_rr(tr, te)
        if r:
            r["fold"] = k + 1
            results.append(r)
            print("  fold %d: n_test=%-4d T=%-3d 院=%-2d RR=%.3f [%.3f,%.3f] 显著=%s"
                  % (k + 1, r["n_test"], r["n_treated_test"], r["n_hospitals_test"],
                     r["rr"], r["rr_ci95"][0] or float("nan"),
                     r["rr_ci95"][1] or float("nan"), r["significant_positive"]))
        else:
            print("  fold %d: 样本不足或单类，跳过" % (k + 1))

    valid = [r for r in results if np.isfinite(r["rr"])]
    pos = sum(1 for r in valid if r["rr"] > 1.0)
    sig_pos = sum(1 for r in valid if r["significant_positive"])
    summary = {
        "method": "manual stratified snake folds (by hospital MV volume) + train/test g-computation",
        "n_splits": 5, "n_valid_folds": len(valid),
        "n_folds_rr_gt1": pos, "n_folds_rr_gt1_significant": sig_pos,
        "frac_rr_gt1": round(pos / len(valid), 3) if valid else None,
        "frac_rr_gt1_significant": round(sig_pos / len(valid), 3) if valid else None,
        "folds": results,
        "fold_sizes": [{"fold": k + 1,
                        "n": int((mv["fold"] == k).sum()),
                        "hospitals": int(mv.loc[mv["fold"] == k, "hospitalid"].nunique()),
                        "treated": int(mv.loc[mv["fold"] == k, "T"].sum())} for k in range(5)],
    }
    print("\n=== 结论 ===")
    print("  RR>1 的折数: %d/%d (%.0f%%)" % (pos, len(valid), 100 * (summary["frac_rr_gt1"] or 0)))
    print("  RR>1 且 CI 不含 1: %d/%d (%.0f%%)" % (
        sig_pos, len(valid), 100 * (summary["frac_rr_gt1_significant"] or 0)))

    # 更新总报告
    rp = OUTPUT_DIR / "final_validation_report.json"
    rep = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    rep["stratified_groupkfold_mv"] = summary
    rep["robust_vs_exploratory"] = {
        "robust": [] if sig_pos < 3 else ["Invasive MV (RR>1 且多数折显著)"],
        "exploratory": ["Age", "Admission type", "Sepsis", "RASS severity",
                        "No invasive MV"],
        "rule": "稳健 = 5 折中至少 3 折 RR>1 且 CI 不含 1；否则为探索性。",
    }
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nupdated:", rp)

    # 标注版森林图
    v1 = rep.get("v1_subgroups", [])
    alt = rep.get("alt_subgroups", [])
    if v1:
        labels = [r["subgroup"] for r in v1]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.40 * len(labels) + 2.0), sharey=True)
        for ax, rows, title in zip(axes, [v1, alt],
                                   ["V1: CAM-ICU neg + RASS -2~0", "ALT: CAM-ICU neg only"]):
            y = np.arange(len(labels))[::-1]
            for i, r in zip(y, rows):
                if r.get("rr") is None:
                    continue
                lo, hi = r.get("rr_lo"), r.get("rr_hi")
                color = "#C44E52" if "Invasive MV" == r["subgroup"] else "#4C72B0"
                ax.plot([r["rr"]], [i], "o", color=color, ms=5)
                if lo is not None and hi is not None:
                    ax.plot([lo, hi], [i, i], "-", color=color, lw=1.5)
            ax.axvline(1, color="grey", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Risk ratio (24h early DEX vs not)", fontsize=8)
        tagged = [("%s  [robust]" % l) if l == "Invasive MV" and sig_pos >= 3 else
                  ("%s  [exploratory]" % l) for l in labels]
        axes[0].set_yticks(np.arange(len(labels))[::-1])
        axes[0].set_yticklabels(tagged, fontsize=7.5)
        fig.suptitle("Prespecified subgroups with cross-hospital validation status", y=1.02)
        for f in ("png", "svg", "pdf"):
            fig.savefig(OUTPUT_DIR / ("subgroup_forest_final_annotated." + f), format=f)
        plt.close(fig)
        print("annotated figure:", OUTPUT_DIR / "subgroup_forest_final_annotated.png")


if __name__ == "__main__":
    main()
