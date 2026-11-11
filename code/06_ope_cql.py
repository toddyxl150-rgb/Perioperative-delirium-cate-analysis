# -*- coding: utf-8 -*-
"""
CQL 策略学习 + FQE / WIS 离线策略评估（OPE）
对应：研究规格基线_v1.2.md 第 4-5 节（CQL 独立模块、Reward 由 PI 裁决）

状态说明（务必如实转达）
------------------------
- CQL 训练需要 d3rlpy / torch；本地环境 torch【未安装】，因此 CQL 部分
  仅提供【接口模板】，未实跑、未产出策略价值。
- FQE 与 WIS 用 numpy / sklearn 给出【可实跑的简化实现】，用于管线自检；
  正式结果须在 d3rlpy 环境下用完整实现复核。
- Reward 参数由 PI 裁决给出（标注为 Lee 2024 标准），本助手未与原文核对。

Reward（v1.2 锁定）
------------------
  步长内谵妄阳性          -> -1
  终点（ICU 出院或第120步）无谵妄 -> +1
  其余步长                ->  0
  gamma = 0.99
"""

from __future__ import annotations

import numpy as np

from model_config import (
    RL_GAMMA, RL_REWARD_DELIRIUM, RL_REWARD_TERMINAL_NO_DELIRIUM,
    RL_REWARD_STEP, MAX_STEPS,
)


def step_reward(delirium_positive: bool, is_terminal: bool,
                terminal_no_delirium: bool) -> float:
    """按 v1.2 锁定规则计算单步 reward。"""
    if delirium_positive:
        return float(RL_REWARD_DELIRIUM)
    if is_terminal:
        return float(RL_REWARD_TERMINAL_NO_DELIRIUM) if terminal_no_delirium \
            else float(RL_REWARD_STEP)
    return float(RL_REWARD_STEP)


def wis(trajectories, gamma: float = RL_GAMMA) -> dict:
    """
    Weighted Importance Sampling（逐步重要性比）。

    trajectories: list of dict，每条含
        rewards      : list[float]
        behavior_prob: list[float]  行为策略在该步选择实际动作的概率
        eval_prob    : list[float]  待评估策略在该步选择同一动作的概率
    """
    num = 0.0
    den = 0.0
    per_traj = []
    for tr in trajectories:
        rho = 1.0
        v = 0.0
        w = 0.0
        for t, r in enumerate(tr["rewards"]):
            bp = max(float(tr["behavior_prob"][t]), 1e-8)
            ep = max(float(tr["eval_prob"][t]), 1e-8)
            rho *= min(ep / bp, 100.0)          # 截断，防止权重爆炸
            disc = gamma ** t
            v += disc * rho * float(r)
            w += disc * rho
        num += v
        den += w
        per_traj.append(v)
    return {
        "wis_value": float(num / den) if den > 0 else float("nan"),
        "denominator": float(den),
        "n_trajectories": len(trajectories),
        "mean_traj_return": float(np.mean(per_traj)) if per_traj else float("nan"),
    }


def fitted_q_evaluation(trajectories, n_iterations: int = 20) -> dict:
    """
    简化 FQE：用 fitted Bellman iteration 估计待评估策略的 Q 值。
    仅作管线自检；正式实现请用 d3rlpy 的 FQE。
    """
    from sklearn.ensemble import GradientBoostingRegressor
    states, actions, rewards, next_states, next_actions, terminals = [], [], [], [], [], []
    for tr in trajectories:
        s = np.asarray(tr["states"], dtype=float)
        a = np.asarray(tr["actions"], dtype=float)
        r = np.asarray(tr["rewards"], dtype=float)
        for t in range(len(r)):
            states.append(s[t]); actions.append(a[t]); rewards.append(float(r[t]))
            terminals.append(float(t == len(r) - 1))
            next_states.append(s[min(t + 1, len(r) - 1)])
            next_actions.append(a[min(t + 1, len(r) - 1)])
    S = np.asarray(states)
    if S.ndim == 1:
        S = S.reshape(-1, 1)
    A = np.asarray(actions).reshape(-1, 1)
    X = np.hstack([S, A])
    NS = np.asarray(next_states)
    if NS.ndim == 1:
        NS = NS.reshape(-1, 1)
    NA = np.asarray(next_actions).reshape(-1, 1)
    X_next = np.hstack([NS, NA])
    y = np.asarray(rewards, dtype=float)
    term = np.asarray(terminals, dtype=float)

    q = np.zeros(len(y), dtype=float)
    model = None
    for _ in range(n_iterations):
        target = y + RL_GAMMA * (1 - term) * q
        model = GradientBoostingRegressor(n_estimators=40, max_depth=2, random_state=0)
        model.fit(X, target)
        q = model.predict(X_next)
    return {
        "fqe_value": float(model.predict(X).mean()),
        "n_transitions": int(len(y)),
        "n_iterations": n_iterations,
        "note": "simplified fitted-Bellman FQE, NOT d3rlpy implementation",
    }


def train_cql(trajectories, **kwargs):
    """
    CQL 接口模板。本地 torch 未安装 -> 抛出明确错误，不静默降级、不假装训练完成。
    """
    try:
        import torch  # noqa: F401
        import d3rlpy  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "CQL requires torch + d3rlpy; local environment lacks them (%s). "
            "Install and then configure: d3rlpy.algos.CQLConfig(...).create(); "
            "fit(trajectories, n_steps=...). This function is a TEMPLATE, not executed."
            % e
        ) from e
    # from d3rlpy.algos import CQLConfig
    # algo = CQLConfig(**kwargs).create(device="cpu")
    # algo.fit(trajectories, n_steps=100000)
    # return algo
    raise NotImplementedError("Fill in d3rlpy CQLConfig after installing dependencies")


def _synth_trajectories(n=40, horizon=12, seed=0):
    """合成轨迹，仅用于管线自检。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.normal(size=(horizon, 3))
        a = rng.integers(0, 5, size=horizon).astype(float)
        r, bp, ep = [], [], []
        for t in range(horizon):
            positive = bool(rng.random() < 0.2)
            terminal = (t == horizon - 1)
            r.append(step_reward(positive, terminal, terminal_no_delirium=not positive))
            bp.append(0.6)
            ep.append(0.6 if a[t] < 3 else 0.3)
        out.append({"states": s, "actions": a, "rewards": r,
                    "behavior_prob": bp, "eval_prob": ep})
    return out


def selfcheck():
    trs = _synth_trajectories()
    w = wis(trs)
    f = fitted_q_evaluation(trs, n_iterations=5)
    print("  WIS:", {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in w.items()})
    print("  FQE:", {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in f.items() if k != "note"})
    try:
        train_cql(trs)
        cql = "UNEXPECTED: CQL ran"
    except RuntimeError as e:
        cql = "expected (dependency missing): " + str(e)[:90]
    print("  CQL:", cql)
    # reward 规则单测
    assert step_reward(True, False, False) == -1
    assert step_reward(False, True, True) == 1
    assert step_reward(False, False, False) == 0
    assert step_reward(False, True, False) == 0
    print("  reward rule checks OK | gamma =", RL_GAMMA, "| max_steps =", MAX_STEPS)
    return {"wis": w, "fqe": f, "cql": cql, "reward_checks": "OK"}


if __name__ == "__main__":
    selfcheck()
