# -*- coding: utf-8 -*-
"""
任务 A（Python 路径）：参数化 G-formula，含竞争风险的反事实 CIF / RR / RD
对应：研究规格基线_v1.2.md 第 3-4 节；禁止 1-KM。

方法
----
离散时间（6h 步长）参数化 g-formula：
1. 用多项 logistic 拟合「该步长发生的事件」∈ {none, resolution, death, discharge}；
   death / discharge 为 competing events，事件发生后该轨迹停止随访。
2. 用二项 logistic 拟合「该步长 DEX 策略归属」的观测（行为）模型，
   供宽限期之后的 natural course 使用。
3. 蒙特卡洛模拟两种反事实干预：
   - 'early_dex'    : 12h 宽限期内强制启动 DEX（A=1）
   - 'no_early_dex' : 12h 宽限期内强制不启动（A=0），之后按自然病程
   宽限期之后两种策略都按行为模型预测 A（对应 v1.1 方案 A「之后按各自自然病程随访」）。
4. 逐步累积 resolution 事件 → 反事实 CIF；终点比较 RR / RD + bootstrap CI。

骨架限制（必须如实说明）
------------------------
- 时变协变量演化采用「基线值 + step 时间趋势」的简化模型，未模拟完整协变量轨迹；
  完整实现须按 TTE 协议表补充各时变协变量的演化模型（g-formula 的标准要求）。
- 当前不含 clone-censor-weight 与删失权重；IPTW 由 02_dataset_and_iptw.py 单独拟合。
- 估计结果是【方法学演示】，真实效应须在 PI 本地跑完 SQL v1.2 长表后重新拟合。

运行：
  python 03b_gformula_py.py --selfcheck
  python 03b_gformula_py.py --input analysis_long.csv --output outputs/gformula_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from model_config import (
    ID_COL, STEP_COL, TREAT_COL, OUTCOME_COL, OUTCOME_ALT_COL,
    DEATH_COL, DISCHARGE_COL,
    BASELINE_COVARIATES, MAX_STEPS, GRACE_STEPS, N_SIMUL, N_BOOTSTRAP,
    RANDOM_SEED, OUTPUT_DIR,
)

EVENT_NONE, EVENT_RESOLUTION, EVENT_DEATH, EVENT_DISCHARGE = 0, 1, 2, 3


def e_value_rr(rr):
    """VanderWeele & Ding (2017) 的 E-value：把 RR 推向 1 所需的最小未测量混杂强度。"""
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return None
    r = float(rr) if rr >= 1 else 1.0 / float(rr)
    return float(r + np.sqrt(r * (r - 1.0)))


def e_value_ci(lo, hi):
    """CI 的 E-value：取距离 1 较近的界；若 CI 跨 1 则为 1（无需任何混杂即可解释）。"""
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if lo <= 1.0 <= hi:
        return 1.0
    bound = lo if abs(np.log(lo)) <= abs(np.log(hi)) else hi
    return e_value_rr(bound)


class ParametricGFormula:
    def __init__(self, random_state=RANDOM_SEED, outcome_col=OUTCOME_COL):
        self.random_state = random_state
        self.outcome_col = outcome_col      # V1 或 ALT（PI 2026-11-09 裁决）
        self.event_model = None
        self.exposure_model = None
        self.baseline_pool = None
        self.cov_cols = []
        self.tv_cols = []
        self.tv_median_by_step = None

    def _event_vector(self, df):
        # 竞争风险优先级：同一步长同时标记时，缓解 > 死亡 > 出院（避免出院覆盖缓解）
        res = df[self.outcome_col].to_numpy() == 1
        dth = df[DEATH_COL].to_numpy() == 1
        dis = df[DISCHARGE_COL].to_numpy() == 1
        y = np.zeros(len(df), dtype=int)
        y[dis] = EVENT_DISCHARGE
        y[dth] = EVENT_DEATH
        y[res] = EVENT_RESOLUTION
        return y

    def _features(self, baseline_row: np.ndarray, step: int, a: int) -> np.ndarray:
        # 基线 + 时变协变量（该步长经验中位数）+ 步长趋势 + 当前策略归属
        s = min(int(step), MAX_STEPS - 1)
        if self.tv_cols:
            tv = self.tv_median_by_step.iloc[s].to_numpy(dtype=float)
        else:
            tv = np.array([], dtype=float)
        return np.concatenate([
            baseline_row, tv,
            np.array([step / float(MAX_STEPS), np.sin(step / 12.0), float(a)], dtype=float),
        ]).reshape(1, -1)

    def fit(self, df: pd.DataFrame):
        self.cov_cols = [c for c in BASELINE_COVARIATES
                         if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        d = df.copy()
        d[self.cov_cols] = d[self.cov_cols].apply(pd.to_numeric, errors="coerce")
        med = d[self.cov_cols].median(numeric_only=True).fillna(0.0)
        d[self.cov_cols] = d[self.cov_cols].fillna(med)

        B = d[self.cov_cols].to_numpy(dtype=float)
        step = pd.to_numeric(d[STEP_COL], errors="coerce").fillna(0).to_numpy(dtype=float)
        A = pd.to_numeric(d[TREAT_COL], errors="coerce").fillna(0).to_numpy(dtype=float)

        # PI 裁决：把时变生命体征纳入事件模型的混杂调整向量。
        # 拟合用观测值；模拟时用各步长的经验中位数近似人群轨迹（g-formula 的简化实现）。
        self.tv_cols = [c for c in ("heartrate", "systemicmean", "sao2", "temperature")
                        if c in d.columns and pd.api.types.is_numeric_dtype(d[c])]
        if self.tv_cols:
            tv_obs = d[[STEP_COL] + self.tv_cols].apply(pd.to_numeric, errors="coerce")
            self.tv_median_by_step = (tv_obs.groupby(STEP_COL)[self.tv_cols].median()
                                      .reindex(range(MAX_STEPS)).ffill().bfill().fillna(0.0))
            TV = tv_obs[self.tv_cols].fillna(
                self.tv_median_by_step.median()).to_numpy(dtype=float)
        else:
            self.tv_median_by_step = pd.DataFrame(index=range(MAX_STEPS))
            TV = np.zeros((len(d), 0))

        X = np.column_stack([
            B, TV,
            step / float(MAX_STEPS),
            np.sin(step / 12.0),
            A,
        ])
        y = self._event_vector(d)

        self.event_model = LogisticRegression(max_iter=2000, random_state=self.random_state)
        self.event_model.fit(X, y)
        self.exposure_model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        self.exposure_model.fit(X, A.astype(int))
        # 蒙特卡洛起点：真实基线协变量的经验分布
        self.baseline_pool = d.drop_duplicates(ID_COL)[self.cov_cols].to_numpy(dtype=float) \
            if ID_COL in d.columns else B
        self.X_dim = X.shape[1]
        return self

    def _natural_a(self, baseline_row, step, rng):
        x = self._features(baseline_row, step, 0)
        p = float(self.exposure_model.predict_proba(x)[0, 1])
        return int(rng.random() < p)

    def simulate(self, intervention: str, n_simul: int = N_SIMUL, seed=None,
                 baseline_pool=None) -> np.ndarray:
        """返回长度 MAX_STEPS+1 的反事实 CIF 曲线。"""
        rng = np.random.default_rng(self.random_state if seed is None else seed)
        pool = self.baseline_pool if baseline_pool is None else baseline_pool
        idx = rng.integers(0, len(pool), size=n_simul)
        baselines = pool[idx]
        alive = np.ones(n_simul, dtype=bool)
        cum = np.zeros(MAX_STEPS + 1, dtype=float)

        classes = self.event_model.classes_
        for step in range(MAX_STEPS):
            live_idx = np.flatnonzero(alive)
            if live_idx.size == 0:
                cum[step + 1:] = cum[step]
                break
            # 向量化：整步一次 predict_proba，避免逐个体调用造成自检过慢
            if step < GRACE_STEPS:
                a_vec = np.full(live_idx.size, 1 if intervention == "early_dex" else 0)
            else:
                x0 = np.vstack([self._features(baselines[i], step, 0) for i in live_idx])
                p_a = self.exposure_model.predict_proba(x0)[:, 1]
                a_vec = (rng.random(live_idx.size) < p_a).astype(int)
            X = np.vstack([self._features(baselines[i], step, a_vec[k])
                           for k, i in enumerate(live_idx)])
            prob = self.event_model.predict_proba(X)
            prob = prob / prob.sum(axis=1, keepdims=True)
            u = rng.random(live_idx.size)
            draw = np.clip((u[:, None] > np.cumsum(prob, axis=1)).sum(axis=1),
                           0, len(classes) - 1)
            ev = classes[draw]
            new_res = int(np.sum(ev == EVENT_RESOLUTION))
            stopped = np.isin(ev, [EVENT_RESOLUTION, EVENT_DEATH, EVENT_DISCHARGE])
            alive[live_idx[stopped]] = False   # 竞争事件后停止随访
            cum[step + 1] = cum[step] + new_res / float(n_simul)
        return cum

    def compare(self, n_simul=N_SIMUL, n_boot=N_BOOTSTRAP):
        cif1 = self.simulate("early_dex", n_simul)
        cif0 = self.simulate("no_early_dex", n_simul)
        r1, r0 = float(cif1[-1]), float(cif0[-1])
        rr = r1 / r0 if r0 > 0 else float("nan")
        rd = r1 - r0

        rng = np.random.default_rng(self.random_state + 1)
        boot_rr, boot_rd = [], []
        small = max(50, n_simul // 5)
        for _ in range(n_boot):
            pool = self.baseline_pool[rng.integers(0, len(self.baseline_pool),
                                                  size=min(len(self.baseline_pool), 400))]
            a = float(self.simulate("early_dex", small, seed=int(rng.integers(1, 10**8)),
                                    baseline_pool=pool)[-1])
            b = float(self.simulate("no_early_dex", small, seed=int(rng.integers(1, 10**8)),
                                    baseline_pool=pool)[-1])
            if b > 0:
                boot_rr.append(a / b)
            boot_rd.append(a - b)

        def ci(v):
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]

        rr_ci = ci(boot_rr)
        rd_ci = ci(boot_rd)
        return {
            "cif_early_dex": cif1.tolist(),
            "cif_no_early_dex": cif0.tolist(),
            "risk_early_dex": r1, "risk_no_early_dex": r0,
            "risk_ratio": rr, "risk_ratio_ci95": rr_ci,
            "risk_difference": rd, "risk_difference_ci95": rd_ci,
            "e_value_rr": e_value_rr(rr),
            "e_value_rr_ci": e_value_ci(rr_ci[0], rr_ci[1]),
            "e_value_rd": e_value_rr(1.0 + rd) if rd > -1 else None,
            "n_simul": n_simul, "n_bootstrap": n_boot,
            "competing_events": ["ICU death", "ICU discharge"],
            "estimator": "parametric g-formula (discrete-time cause-specific hazards)",
            "one_minus_km_used": False,
        }


def make_synthetic_long(n_patients=600, n_steps=24, seed=RANDOM_SEED):
    """仅用于代码自检的合成数据；不得作为研究结果。事件每人只标记一次。"""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        base = {
            "age": rng.uniform(40, 80),
            "gender": rng.integers(0, 2),
            "admissionweight": rng.uniform(55, 100),
            "apachescore": rng.uniform(30, 90),
            "acutephysiologyscore": rng.uniform(20, 70),
            "predictedicumortality": rng.uniform(0.05, 0.6),
        }
        a = int(rng.random() < 0.4 + 0.15 * (base["apachescore"] > 60))
        ev_step = int(rng.integers(0, n_steps))
        p_ev = np.array([0.50, 0.22 + 0.30 * a, 0.08, 0.10], dtype=float)
        ev_type = int(rng.choice([0, 1, 2, 3], p=p_ev / p_ev.sum()))
        for s in range(n_steps):
            hit = (s == ev_step)
            rows.append({
                ID_COL: p, STEP_COL: s, "hospitalid": p % 20,
                TREAT_COL: a,
                OUTCOME_COL: int(hit and ev_type == 1),
                "resolution_alt_event": int(hit and ev_type == 1),
                DEATH_COL: int(hit and ev_type == 2),
                DISCHARGE_COL: int(hit and ev_type == 3),
                **base,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=str(OUTPUT_DIR / "gformula_results.json"))
    ap.add_argument("--n-simul", type=int, default=N_SIMUL,
                    help="蒙特卡洛轨迹数；正式运行用配置默认值，演练可减小")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP,
                    help="bootstrap 次数；正式运行用配置默认值，演练可减小")
    ap.add_argument("--outcome", default="v1", choices=["v1", "alt"],
                    help="v1=resolution_v1_event（含 RASS 达标）；alt=resolution_alt_event（仅 CAM-ICU 阴性）")
    a = ap.parse_args()

    if a.selfcheck or not a.input:
        df = make_synthetic_long()
        print("synthetic rows:", len(df))
        gf = ParametricGFormula().fit(df)
        res = gf.compare(n_simul=300, n_boot=25)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("cif")},
                         ensure_ascii=False, indent=2))
        print("SELFCHECK gformula: CIF computed for both strategies,",
              "1-KM used:", res["one_minus_km_used"])
    else:
        import importlib
        _ds = importlib.import_module("02_dataset_and_iptw")   # 文件名以数字开头，须用 importlib
        df = _ds.load_long_table(a.input)
        print("real long table rows:", len(df), "| outcome:", a.outcome)
        ocol = OUTCOME_COL if a.outcome == "v1" else OUTCOME_ALT_COL
        res = ParametricGFormula(outcome_col=ocol).fit(df).compare(
            n_simul=a.n_simul, n_boot=a.n_boot)
        res["outcome"] = a.outcome
        Path(a.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written:", a.output)
