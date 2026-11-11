# -*- coding: utf-8 -*-
"""
阶段 3 收官：9 项预设亚组 + 基线 RASS 亚组 + 跨医院 GroupKFold 验证

内容
----
1. 亚组（V1 与 ALT 双结局）：年龄、入院类型、脓毒症、有创机械通气、基线 RASS 躁动程度
2. 跨医院 GroupKFold(5, groups=hospitalid) 留出验证 Invasive MV 亚组的方向复现性
3. 输出：9 项亚组森林图（V1/ALT 并排）、汇总表 JSON/CSV

说明
----
- 基线 RASS 由 08 前置步骤从 nurseCharting 提取（Time Zero 前最近一次），
  仅 RASS ∈ {+1,+2} 与 {+3,+4} 进入该亚组；RASS ≤ 0 者不参与。
- GroupKFold 用【留出折上的粗 RR】判断方向复现，不做模型外推（避免小样本过拟合）。
- 全部为探索性，未做多重比较校正。
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from model_config import OUTPUT_DIR, FIG_DPI

m7 = importlib.import_module("07_subgroups")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": FIG_DPI,
                     "savefig.bbox": "tight"})


def load_with_rass(outcome_col):
    g = m7.load_patient_level(outcome_col=outcome_col)
    rp = OUTPUT_DIR / "baseline_rass.csv"
    if rp.exists():
        r = pd.read_csv(rp)[["patientunitstayid", "rass"]]
        g = g.merge(r, on="patientunitstayid", how="left")
    else:
        g["rass"] = np.nan
    g2 = m7.add_mv_flag(g)
    if g2 is not None:
        g = g2
    return g


def build_specs(g):
    g = g.copy()
    g["age_lt65"] = (g["age"] < 65).astype(int)
    g["age_ge65"] = (g["age"] >= 65).astype(int)
    ut = g["unittype"].astype(str).str.lower()
    g["surgical"] = ut.str.contains("sicu|csru|surg").astype(int)
    g["medical"] = ut.str.contains("micu|med").astype(int)
    dx = g["apacheadmissiondx"].astype(str).str.lower()
    g["sepsis"] = dx.str.contains("sepsis|septic").astype(int)
    g["rass_mild"] = g["rass"].isin([1, 2]).astype(int)
    g["rass_severe"] = g["rass"].isin([3, 4]).astype(int)

    specs = [
        ("Age <65", "age_lt65", 1, None),
        ("Age >=65", "age_ge65", 1, None),
        ("Surgical ICU", "surgical", 1, None),
        ("Medical ICU", "medical", 1, None),
        ("Sepsis", "sepsis", 1, None),
        ("Non-sepsis", "sepsis", 0, None),
    ]
    if "invasive_mv" in g.columns:
        specs += [("Invasive MV", "invasive_mv", 1, None),
                  ("No invasive MV", "invasive_mv", 0, None)]
    # RASS 亚组：交互检验只在 +1~+4 子集内做
    specs += [("RASS +1/+2 (mild)", "rass_mild", 1, ("rass", [1, 2, 3, 4])),
              ("RASS +3/+4 (severe)", "rass_severe", 1, ("rass", [1, 2, 3, 4]))]
    return g, specs


def run_subsets(g, specs, n_boot=150):
    rows = []
    for label, col, val, inter_subset in specs:
        sub = g[g[col] == val]
        eff = m7.gcomp_effect(sub, n_boot=n_boot)
        if inter_subset:
            frame = g[g[inter_subset[0]].isin(inter_subset[1])]
        else:
            frame = g
        p_int = m7.interaction_p(frame, col)
        rows.append({"subgroup": label, "variable": col, "level": val,
                     "n": int(len(sub)), "effect": eff, "p_interaction": p_int})
    return rows


def cv_mv(g, n_splits=5):
    """跨医院 GroupKFold：在留出折上计算 MV 亚组的粗 RR，检验方向复现。"""
    if "invasive_mv" not in g.columns:
        return {"status": "missing_mv"}
    mv = g[g["invasive_mv"] == 1].dropna(subset=["Y", "T", "hospitalid"])
    if len(mv) < 200:
        return {"status": "insufficient"}
    gkf = GroupKFold(n_splits=n_splits)
    folds = []
    for i, (tr, te) in enumerate(gkf.split(mv, groups=mv["hospitalid"].astype(str))):
        d = mv.iloc[te]
        r1 = float(d.loc[d["T"] == 1, "Y"].mean()) if (d["T"] == 1).any() else np.nan
        r0 = float(d.loc[d["T"] == 0, "Y"].mean()) if (d["T"] == 0).any() else np.nan
        folds.append({"fold": i + 1, "n_test": int(len(d)),
                      "n_treated": int(d["T"].sum()),
                      "n_hospitals": int(d["hospitalid"].nunique()),
                      "risk_t1": r1, "risk_t0": r0,
                      "rr_crude": (r1 / r0) if (r0 and np.isfinite(r0) and r0 > 0) else None,
                      "rd_crude": (r1 - r0) if np.isfinite(r1) and np.isfinite(r0) else None})
    valid = [f for f in folds if f["rr_crude"] is not None]
    pos = sum(1 for f in valid if f["rr_crude"] > 1.0)
    return {"n_splits": n_splits, "folds": folds,
            "n_valid_folds": len(valid), "n_folds_rr_gt1": pos,
            "frac_rr_gt1": round(pos / len(valid), 3) if valid else None}


def plot_final(v1_rows, alt_rows, path):
    labels = [r["subgroup"] for r in v1_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.40 * len(labels) + 1.8), sharey=True)
    for ax, rows, title in zip(axes, [v1_rows, alt_rows],
                               ["V1: CAM-ICU neg + RASS -2~0", "ALT: CAM-ICU neg only"]):
        y = np.arange(len(labels))[::-1]
        for i, r in zip(y, rows):
            e = r["effect"]
            if not e:
                continue
            lo, hi = e["rr_ci95"]
            ax.plot([e["rr"]], [i], "o", color="#C44E52", ms=5)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [i, i], "-", color="#C44E52", lw=1.5)
        ax.axvline(1, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Risk ratio (24h early DEX vs not)", fontsize=8)
    axes[0].set_yticks(np.arange(len(labels))[::-1])
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Prespecified subgroups (V1 vs ALT), 24h window — exploratory", y=1.02)
    for f in ("png", "svg", "pdf"):
        fig.savefig(Path(path).with_suffix("." + f), format=f)
    plt.close(fig)


def to_table(rows, outcome):
    out = []
    for r in rows:
        e = r["effect"]
        out.append({
            "outcome": outcome, "subgroup": r["subgroup"], "n": r["n"],
            "n_treated": None if not e else e["n_treated"],
            "rr": None if not e else round(e["rr"], 3),
            "rr_lo": None if not e or e["rr_ci95"][0] is None else round(e["rr_ci95"][0], 3),
            "rr_hi": None if not e or e["rr_ci95"][1] is None else round(e["rr_ci95"][1], 3),
            "rd": None if not e else round(e["rd"], 4),
            "p_interaction": None if r["p_interaction"] is None else round(r["p_interaction"], 4),
        })
    return out


def main():
    print("=== 载入 ===")
    gv = load_with_rass("resolution_v1_event")
    ga = load_with_rass("resolution_alt_event")
    print("V1 n=%d | ALT n=%d | 有基线RASS: %d | MV: %s"
          % (len(gv), len(ga), int(gv["rass"].notna().sum()),
             "yes" if "invasive_mv" in gv.columns else "no"))

    gv, specs = build_specs(gv)
    ga, _ = build_specs(ga)

    print("\n=== V1 亚组 ===")
    v1_rows = run_subsets(gv, specs)
    for r in v1_rows:
        e = r["effect"]
        print("  %-22s n=%-5d %s" % (r["subgroup"], r["n"],
              "RR=%.3f [%.3f,%.3f] P_int=%s" % (e["rr"], e["rr_ci95"][0] or float("nan"),
                                                e["rr_ci95"][1] or float("nan"),
                                                "NA" if r["p_interaction"] is None else round(r["p_interaction"], 3))
              if e else "样本不足"))

    print("\n=== ALT 亚组 ===")
    alt_rows = run_subsets(ga, specs)
    for r in alt_rows:
        e = r["effect"]
        print("  %-22s n=%-5d %s" % (r["subgroup"], r["n"],
              "RR=%.3f [%.3f,%.3f] P_int=%s" % (e["rr"], e["rr_ci95"][0] or float("nan"),
                                                e["rr_ci95"][1] or float("nan"),
                                                "NA" if r["p_interaction"] is None else round(r["p_interaction"], 3))
              if e else "样本不足"))

    print("\n=== 跨医院 GroupKFold（MV 亚组方向复现）===")
    cv = cv_mv(gv)
    if "folds" in cv:
        for f in cv["folds"]:
            print("  fold %d: n=%d T=%d 院=%d RR=%.3f" % (
                f["fold"], f["n_test"], f["n_treated"], f["n_hospitals"],
                f["rr_crude"] if f["rr_crude"] is not None else float("nan")))
        print("  RR>1 的折数: %d/%d (%.0f%%)" % (
            cv["n_folds_rr_gt1"], cv["n_valid_folds"], 100 * (cv["frac_rr_gt1"] or 0)))
    else:
        print("  ", cv)

    plot_final(v1_rows, alt_rows, OUTPUT_DIR / "subgroup_forest_final.png")
    table = to_table(v1_rows, "v1") + to_table(alt_rows, "alt")
    pd.DataFrame(table).to_csv(OUTPUT_DIR / "subgroup_final_table.csv",
                               index=False, encoding="utf-8-sig")
    report = {"n_patients": int(len(gv)), "n_treated": int(gv["T"].sum()),
              "n_with_baseline_rass": int(gv["rass"].notna().sum()),
              "v1_subgroups": to_table(v1_rows, "v1"),
              "alt_subgroups": to_table(alt_rows, "alt"),
              "mv_groupkfold": cv,
              "note": "探索性；未做多重比较校正；GroupKFold 用留出折粗 RR 判方向。"}
    (OUTPUT_DIR / "final_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\nwritten:", OUTPUT_DIR / "final_validation_report.json")


if __name__ == "__main__":
    main()
