# -*- coding: utf-8 -*-
"""
预设临床亚组分析 + Risk-based 风险分层（对标 Serpa Neto 2025 Fig2 / Goligher 2023 Fig3）

方法
----
- 患者级数据：Y=缓解(V1)、T=24h 内启动 DEX（early_initiation_flag / dex_first-t0<=1440）
- 每个亚组内用 logistic 回归做 g-computation：
    logit P(Y=1) = b0 + b1*T + b2*age + b3*apachescore + b4*male
  再对 T=1 / T=0 分别预测并取均值，得到亚组边际 RR / RD（bootstrap 95% CI）
- 交互 P 值：全样本 logit(Y ~ T*subgroup + age + apachescore + male) 的交互项 Wald P
- Risk-based：按 apachescore 十分位分层，重复同样的 g-computation，画 RD 趋势

局限（如实标注）
----------------
- 机械通气：从 respiratoryCare.airwaytype 推断（若该列不可用则标 missing）
- 基线 RASS 躁动程度：长表未含原始 RASS 值，本轮【未纳入】，需重扫 nurseCharting
- 亚组样本较小时 CI 很宽，仅作探索性展示，不做多重比较校正

运行：python 07_subgroups.py
输出：outputs/subgroup_results.json / subgroup_forest.png / risk_decile_effects.png（含 svg/pdf）
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

from model_config import OUTPUT_DIR, EARLY_WINDOW_MIN, FIG_DPI, FIG_SIZE, RANDOM_SEED, EICU_DIR

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": FIG_DPI,
                     "savefig.bbox": "tight"})
FORMATS = ("png", "svg", "pdf")


def _save(fig, path):
    p = Path(path)
    for f in FORMATS:
        fig.savefig(p.with_suffix("." + f), format=f)


def load_patient_level(long_path=None, outcome_col="resolution_v1_event"):
    long_path = long_path or (OUTPUT_DIR / "analysis_long.csv")
    df = pd.read_csv(long_path)
    df["_dex_first"] = pd.to_numeric(df.get("dex_first_start_offset"), errors="coerce")
    df["_t0"] = pd.to_numeric(df.get("t0_offset"), errors="coerce")
    g = df.groupby("patientunitstayid", as_index=False).agg(
        Y=(outcome_col, "max"),
        age=("age", "first"),
        apachescore=("apachescore", "first"),
        gender=("gender", "first"),
        unittype=("unittype", "first"),
        apacheadmissiondx=("apacheadmissiondx", "first"),
        hospitalid=("hospitalid", "first"),
        dex_first=("_dex_first", "first"),
        t0=("_t0", "first"),
    )
    g["T"] = ((g["dex_first"].notna()) & ((g["dex_first"] - g["t0"]) <= EARLY_WINDOW_MIN)).astype(int)
    g["male"] = g["gender"].astype(str).str.lower().str.startswith("m").astype(int)
    g["age"] = pd.to_numeric(g["age"], errors="coerce")
    g["apachescore"] = pd.to_numeric(g["apachescore"], errors="coerce")
    return g.drop(columns=["dex_first", "t0"])


def add_mv_flag(g):
    """从 respiratoryCare.airwaytype 推断有创机械通气；不可用则返回 None。"""
    p = EICU_DIR / "respiratoryCare.csv.gz"
    if not p.exists():
        return None
    try:
        rc = pd.read_csv(p, usecols=lambda c: c.lower() in ("patientunitstayid", "airwaytype"))
        col = [c for c in rc.columns if c.lower() == "airwaytype"]
        if not col:
            return None
        s = rc[col[0]].astype(str).str.lower()
        ids = set(rc.loc[s.str.contains("endotracheal|tracheostomy|oral ett", na=False),
                         "patientunitstayid"])
        g = g.copy()
        g["invasive_mv"] = g["patientunitstayid"].isin(ids).astype(int)
        return g
    except Exception:
        return None


def gcomp_effect(df, covariates=("age", "apachescore", "male"), n_boot=200, seed=RANDOM_SEED):
    """亚组内 g-computation：返回 RR / RD 及 bootstrap CI；样本不足返回 None。"""
    d = df.dropna(subset=["Y", "T"] + list(covariates)).copy()
    if len(d) < 60 or d["T"].nunique() < 2 or d["Y"].nunique() < 2:
        return None
    def _mat(frame):
        """显式构造 [截距, T, 协变量...]，避免 add_constant 行为差异导致维度错位。"""
        return np.column_stack([
            np.ones(len(frame)),
            frame[["T"] + list(covariates)].astype(float).to_numpy(),
        ])

    try:
        m = sm.Logit(d["Y"].astype(float), _mat(d)).fit(disp=0)
    except Exception:
        return None

    def pred(model, frame, t):
        f = frame.copy()
        f["T"] = t
        return float(model.predict(_mat(f)).mean())

    p1, p0 = pred(m, d, 1), pred(m, d, 0)
    rr = p1 / p0 if p0 > 0 else np.nan
    rd = p1 - p0

    rng = np.random.default_rng(seed)
    boot_rr, boot_rd = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        b = d.iloc[idx]
        if b["T"].nunique() < 2 or b["Y"].nunique() < 2:
            continue
        try:
            mb = sm.Logit(b["Y"].astype(float), _mat(b)).fit(disp=0)
            q1, q0 = pred(mb, b, 1), pred(mb, b, 0)
            if q0 > 0:
                boot_rr.append(q1 / q0)
            boot_rd.append(q1 - q0)
        except Exception:
            continue

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 10 else [None, None]

    return {"n": int(len(d)), "n_treated": int(d["T"].sum()),
            "risk_t1": p1, "risk_t0": p0,
            "rr": float(rr), "rr_ci95": ci(boot_rr),
            "rd": float(rd), "rd_ci95": ci(boot_rd)}


def interaction_p(df, subgroup_col):
    d = df.dropna(subset=["Y", "T", subgroup_col, "age", "apachescore", "male"]).copy()
    d = d[d[subgroup_col].notna()]
    if d[subgroup_col].nunique() < 2 or len(d) < 100:
        return None
    X = sm.add_constant(d[["T", subgroup_col, "age", "apachescore", "male"]].astype(float))
    X["T_x_S"] = X["T"] * X[subgroup_col]
    try:
        m = sm.Logit(d["Y"].astype(float), X).fit(disp=0)
        return float(m.pvalues.get("T_x_S", np.nan))
    except Exception:
        return None


def build_subgroups(g):
    out = []
    g = g.copy()
    g["age_ge65"] = (g["age"] >= 65).astype(int)
    g["age_lt65"] = (g["age"] < 65).astype(int)

    ut = g["unittype"].astype(str).str.lower()
    g["surgical"] = ut.str.contains("sicu|csru|surg").astype(int)
    g["medical"] = ut.str.contains("micu|med").astype(int)

    dx = g["apacheadmissiondx"].astype(str).str.lower()
    g["sepsis"] = dx.str.contains("sepsis|septic").astype(int)

    specs = [
        ("Age <65", "age_lt65", 1), ("Age >=65", "age_ge65", 1),
        ("Surgical ICU", "surgical", 1), ("Medical ICU", "medical", 1),
        ("Sepsis", "sepsis", 1), ("Non-sepsis", "sepsis", 0),
    ]
    if "invasive_mv" in g.columns:
        specs += [("Invasive MV", "invasive_mv", 1), ("No invasive MV", "invasive_mv", 0)]

    for label, col, val in specs:
        sub = g[g[col] == val]
        eff = gcomp_effect(sub)
        p_int = interaction_p(g, col)
        out.append({"subgroup": label, "variable": col, "level": val,
                    "n_total": int(len(sub)),
                    "p_interaction": p_int,
                    "effect": eff})
    return g, out


def plot_forest(results, path):
    rows = [r for r in results if r["effect"]]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.4))
    y = np.arange(len(rows))[::-1]
    for i, r in zip(y, rows):
        e = r["effect"]
        lo, hi = e["rr_ci95"]
        ax.plot([e["rr"], e["rr"]], [i, i], "o", color="#C44E52", ms=5)
        if lo is not None:
            ax.plot([lo, hi], [i, i], "-", color="#C44E52", lw=1.6)
    ax.axvline(1, color="grey", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(["%s (n=%d, T=%d)" % (r["subgroup"], r["effect"]["n"],
                                             r["effect"]["n_treated"]) for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("Risk ratio (24h early dexmedetomidine vs not)")
    ax.set_title("Prespecified subgroup analysis\n(g-computation; exploratory, no multiplicity adjustment)")
    _save(fig, path)
    plt.close(fig)


def plot_forest_compare(v1_res, alt_res, path):
    """V1 与 ALT 结局的亚组森林图并排对比。"""
    labels = [r["subgroup"] for r in v1_res]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(labels) + 1.8),
                             sharey=True)
    for ax, res, title in zip(
            axes, [v1_res, alt_res],
            ["V1: CAM-ICU negative + RASS -2~0", "ALT: CAM-ICU negative only"]):
        y = np.arange(len(labels))[::-1]
        for i, r in zip(y, res):
            e = r.get("effect")
            if not e:
                continue
            lo, hi = e["rr_ci95"]
            ax.plot([e["rr"]], [i], "o", color="#C44E52", ms=5)
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [i, i], "-", color="#C44E52", lw=1.6)
        ax.axvline(1, color="grey", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Risk ratio (24h early DEX vs not)", fontsize=8)
    axes[0].set_yticks(np.arange(len(labels))[::-1])
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Prespecified subgroups: V1 vs ALT outcome (exploratory)", y=1.02)
    _save(fig, path)
    plt.close(fig)


def risk_deciles(g, n=10, path=None):
    d = g.dropna(subset=["apachescore", "Y", "T"]).copy()
    if len(d) < 200:
        return []
    d["decile"] = pd.qcut(d["apachescore"].rank(method="first"), n, labels=False) + 1
    rows = []
    for k in range(1, n + 1):
        sub = d[d["decile"] == k]
        eff = gcomp_effect(sub, n_boot=120)
        rows.append({"decile": k, "n": int(len(sub)),
                     "apache_mid": float(sub["apachescore"].median()),
                     "effect": eff})
    if path:
        xs = [r["decile"] for r in rows if r["effect"]]
        rds = [r["effect"]["rd"] for r in rows if r["effect"]]
        los = [r["effect"]["rd_ci95"][0] for r in rows if r["effect"]]
        his = [r["effect"]["rd_ci95"][1] for r in rows if r["effect"]]
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.errorbar(xs, rds, yerr=[np.array(rds) - np.array(los),
                                   np.array(his) - np.array(rds)],
                    fmt="o-", color="#4C72B0", capsize=3, lw=1.8)
        ax.axhline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("APACHE score decile (1 = lowest risk)")
        ax.set_ylabel("Risk difference (24h early DEX vs not)")
        ax.set_title("Risk-based treatment effect across baseline severity deciles\n"
                     "(g-computation; exploratory)")
        _save(fig, path)
        plt.close(fig)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcome", default="v1", choices=["v1", "alt"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()

    if a.compare:
        v1 = json.loads((OUTPUT_DIR / "subgroup_results_v1.json").read_text(encoding="utf-8"))
        alt = json.loads((OUTPUT_DIR / "subgroup_results_alt.json").read_text(encoding="utf-8"))
        plot_forest_compare(v1["subgroups"], alt["subgroups"],
                            OUTPUT_DIR / "subgroup_forest_v1_vs_alt.png")
        print("对比森林图:", OUTPUT_DIR / "subgroup_forest_v1_vs_alt.png")
        return

    ycol = "resolution_v1_event" if a.outcome == "v1" else "resolution_alt_event"
    g = load_patient_level(outcome_col=ycol)
    print("outcome:", ycol, "| patient-level n =", len(g), "| T=1:", int(g["T"].sum()),
          "| event rate:", round(float(g["Y"].mean()), 4))

    g_mv = add_mv_flag(g)
    if g_mv is not None:
        g = g_mv
        print("invasive_mv 可用，比例:", round(float(g["invasive_mv"].mean()), 3))
    else:
        print("invasive_mv 不可用（respiratoryCare.airwaytype 缺失），该亚组跳过")

    g, sub = build_subgroups(g)
    dec = risk_deciles(g, path=OUTPUT_DIR / "risk_decile_effects.png")

    report = {
        "outcome": a.outcome,
        "outcome_col": ycol,
        "n_patients": int(len(g)),
        "n_treated": int(g["T"].sum()),
        "event_rate": round(float(g["Y"].mean()), 4),
        "early_window_min": EARLY_WINDOW_MIN,
        "subgroups": sub,
        "risk_deciles": dec,
        "not_included": ["基线 RASS 躁动程度（长表未含原始 RASS 值，需重扫 nurseCharting）"],
        "note": "探索性分析，未做多重比较校正；亚组样本小时 CI 很宽。",
    }
    (OUTPUT_DIR / ("subgroup_results_%s.json" % a.outcome)).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    plot_forest(sub, OUTPUT_DIR / ("subgroup_forest_%s.png" % a.outcome))
    print("\n=== 亚组结果 ===")
    for r in sub:
        e = r["effect"]
        if e:
            print("  %-18s n=%-5d T=%-4d RR=%.3f [%.3f,%.3f] RD=%+.3f  P_int=%s"
                  % (r["subgroup"], e["n"], e["n_treated"], e["rr"],
                     e["rr_ci95"][0] or float("nan"), e["rr_ci95"][1] or float("nan"),
                     e["rd"], "NA" if r["p_interaction"] is None else round(r["p_interaction"], 3)))
        else:
            print("  %-18s 样本不足，跳过" % r["subgroup"])
    print("\n=== 风险十分位 ===")
    for r in dec:
        e = r["effect"]
        print("  D%-2d n=%-5d APACHE~%.0f %s" % (
            r["decile"], r["n"], r["apache_mid"],
            "RD=%+.3f [%.3f,%.3f]" % (e["rd"], e["rd_ci95"][0], e["rd_ci95"][1]) if e else "样本不足"))
    print("\nwritten:", OUTPUT_DIR / ("subgroup_results_%s.json" % a.outcome))


if __name__ == "__main__":
    main()
