# -*- coding: utf-8 -*-
"""
任务 C：DEX 剂量单位换算与 15 档离散化 —— 方案 C（PI 2026-11-09 裁决）

输入：eICU infusionDrug 原始记录（drugname, drugrate[, patientweight]）+ patient.admissionweight
输出：mcg/kg/h 连续剂量 + 15 个离散档位（0~1.5，步长 0.1）+ 浓度来源标记

【方案 C = 主分析 A + 敏感性 B】
A（默认主分析）：无显式浓度的 ml/hr 记录按 4.0 mcg/ml 填补换算
      mcg/kg/h = (ml/hr * 4.0) / weight_kg
   来源声明：「Precedex 200mcg/50ml = 4 mcg/ml」由 PI 裁决给出，
   本助手【未联网核实】药品说明书；本地 eICU 中名称含浓度的 14 条记录
   解析值均为 4.0 mcg/ml，与之一致。该 4.0 是【外部假设】，不是数据实测值。
B（敏感性分析）：所有被标记为「估算浓度」的记录由 02 / 04 的子集校验报告剔除后重跑；
   本模块提供 --no-default-concentration 与 --sensitivity 两个开关。

浓度优先级（高 -> 低）：
   1. 显式传入 concentration_mcg_per_ml（敏感性档位 2.0 / 8.0）
   2. 从 drugname 正则解析："4 mcg/ml"、"200 mcg/50ml"、
      "400 Mcg in Dextrose 5% 100 ml"（均 -> 4 mcg/ml）
   3. 默认 4.0 mcg/ml（仅 ml/hr 记录；标记 assumed_default）
   4. 以上皆无 -> None（不猜测）

新增审计列（方案 B 依赖）：
   concentration_used_mcg_per_ml  实际使用的浓度
   dose_concentration_source      explicit_in_name / assumed_default /
                                  not_applicable / unresolved
   is_assumed_mlhr                布尔；True 即方案 B 需剔除的估算记录

运行：
  python 01_dex_dose.py --selfcheck
  python 01_dex_dose.py --input infusionDrug.csv --output dose.csv
  python 01_dex_dose.py --input infusionDrug.csv --sensitivity
  python 01_dex_dose.py --input infusionDrug.csv --no-default-concentration
"""

import argparse
import re
from typing import Optional

import numpy as np
import pandas as pd

from model_config import (
    DOSE_BIN_EDGES, DOSE_BIN_LABELS, DOSE_OVER_RANGE_FLAG,
    DEFAULT_DEX_CONCENTRATION_MCG_PER_ML, SENSITIVITY_CONCENTRATIONS,
    DOSE_SOURCE_EXPLICIT, DOSE_SOURCE_ASSUMED, DOSE_SOURCE_NA,
    DOSE_SOURCE_UNRESOLVED, DOSE_ASSUMED_FLAG, DOSE_SOURCE_COL,
)

# 单位识别顺序：先带 /kg/ 的，再 /min，最后 /hr、ml/hr
_UNIT_PATTERNS = [
    (re.compile(r"mcg\s*/\s*kg\s*/\s*(?:hr|h|hour)", re.I), "mcg/kg/hr"),
    (re.compile(r"mcg\s*/\s*kg\s*/\s*min", re.I), "mcg/kg/min"),
    (re.compile(r"mg\s*/\s*kg\s*/\s*(?:hr|h|hour)", re.I), "mg/kg/hr"),
    (re.compile(r"mg\s*/\s*kg\s*/\s*min", re.I), "mg/kg/min"),
    (re.compile(r"mcg\s*/\s*min", re.I), "mcg/min"),
    (re.compile(r"mcg\s*/\s*(?:hr|h|hour)", re.I), "mcg/hr"),
    (re.compile(r"mg\s*/\s*min", re.I), "mg/min"),
    (re.compile(r"mg\s*/\s*(?:hr|h|hour)", re.I), "mg/hr"),
    (re.compile(r"ml\s*/\s*(?:hr|h|hour)", re.I), "ml/hr"),
]

# 配置浓度解析：直接浓度 / 药量体积 / 药量 in ... 体积
_CONC_PATTERNS = [
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mcg|ug)\s*/\s*ml\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mcg|ug)\s*/\s*(\d+(?:\.\d+)?)\s*ml\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mcg|ug)\s+in\b.*?\b(\d+(?:\.\d+)?)\s*ml\b", re.I),
]


def parse_unit(drugname: str) -> Optional[str]:
    """从 eICU drugname 中识别速率单位；识别不出返回 None。"""
    if not isinstance(drugname, str):
        return None
    for rx, unit in _UNIT_PATTERNS:
        if rx.search(drugname):
            return unit
    return None


def parse_concentration_mcg_per_ml(drugname: str) -> Optional[float]:
    """从 drugname 解析配置浓度（mcg/ml）；解析不出返回 None。"""
    if not isinstance(drugname, str):
        return None
    m = _CONC_PATTERNS[0].search(drugname)
    if m:
        return float(m.group(1))
    for rx in _CONC_PATTERNS[1:]:
        m = rx.search(drugname)
        if m:
            amount, volume = float(m.group(1)), float(m.group(2))
            if volume > 0:
                return amount / volume
    return None


def resolve_concentration(drugname: str,
                          explicit_concentration=None,
                          assume_default: bool = True) -> tuple:
    """返回 (浓度 mcg/ml 或 None, 来源标记)。

    仅对 ml/hr 记录有意义。explicit 参数与名称解析都算「有明确依据」；
    默认 4.0 属于外部假设，单独标记，供方案 B 剔除。
    """
    if explicit_concentration is not None:
        return float(explicit_concentration), DOSE_SOURCE_EXPLICIT
    parsed = parse_concentration_mcg_per_ml(drugname)
    if parsed is not None:
        return parsed, DOSE_SOURCE_EXPLICIT
    if parse_unit(drugname) == "ml/hr" and assume_default:
        return DEFAULT_DEX_CONCENTRATION_MCG_PER_ML, DOSE_SOURCE_ASSUMED
    return None, DOSE_SOURCE_UNRESOLVED


def convert_to_mcg_kg_h(rate, drugname: str, weight_kg,
                        concentration_mcg_per_ml=None,
                        assume_default: bool = True):
    """把单条输注速率换算为 mcg/kg/h。

    assume_default=False 时关闭方案 A 的 4.0 填补，用于方案 B 的口径校验。
    返回 float 或 None（单位未知 / 体重缺失 / 浓度不可用）。
    """
    try:
        r = float(rate)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(r):
        return None
    if r <= 0:
        return 0.0

    unit = parse_unit(drugname)
    if unit is None:
        return None
    try:
        w = float(weight_kg)
    except (TypeError, ValueError):
        w = np.nan
    has_weight = np.isfinite(w) and w > 0

    if unit == "mcg/kg/hr":
        return r
    if unit == "mcg/kg/min":
        return r * 60.0
    if unit == "mg/kg/hr":
        return r * 1000.0
    if unit == "mg/kg/min":
        return r * 1000.0 * 60.0
    if unit == "mcg/min":
        return r * 60.0 / w if has_weight else None
    if unit == "mcg/hr":
        return r / w if has_weight else None
    if unit == "mg/min":
        return r * 1000.0 * 60.0 / w if has_weight else None
    if unit == "mg/hr":
        return r * 1000.0 / w if has_weight else None
    if unit == "ml/hr":
        conc, _ = resolve_concentration(drugname, concentration_mcg_per_ml,
                                        assume_default)
        if conc is None or not has_weight:
            return None
        return r * float(conc) / w
    return None


def discretize_dose(value, over_range_flag: bool = True):
    """归入 15 个区间 [0,0.1) ... [1.4,1.5]；>1.5 归第 15 档并置标记。"""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 0, False
    v = float(value)
    if v <= 0:
        return 0, False
    if v >= DOSE_BIN_EDGES[-1]:
        return DOSE_BIN_LABELS[-1], True
    idx = int(np.searchsorted(DOSE_BIN_EDGES, v, side="right") - 1)
    return DOSE_BIN_LABELS[max(0, min(idx, len(DOSE_BIN_LABELS) - 1))], False


def convert_dataframe(df: pd.DataFrame,
                      rate_col: str = "drugrate",
                      name_col: str = "drugname",
                      weight_col: str = "admissionweight",
                      weight_alt_col: str = "patientweight",
                      concentration_mcg_per_ml=None,
                      assume_default: bool = True) -> pd.DataFrame:
    """批量换算并写出审计列（方案 B 依赖 is_assumed_mlhr）。"""
    out = df.copy()
    weights = []
    for _, row in out.iterrows():
        w = row.get(weight_col, np.nan)
        if not (isinstance(w, (int, float)) and np.isfinite(w) and w > 0):
            w = row.get(weight_alt_col, np.nan)
        weights.append(w)
    out["_weight_used"] = weights

    vals, bins, flags, units, parsed_c, used_c, sources, assumed = (
        [], [], [], [], [], [], [], [])
    for _, row in out.iterrows():
        name = row.get(name_col)
        unit = parse_unit(name)
        conc, source = resolve_concentration(name, concentration_mcg_per_ml,
                                             assume_default)
        if unit != "ml/hr":
            source = DOSE_SOURCE_NA if unit is not None else DOSE_SOURCE_UNRESOLVED
            conc_used = None
        else:
            conc_used = conc
        v = convert_to_mcg_kg_h(row.get(rate_col), name, row.get("_weight_used"),
                                concentration_mcg_per_ml, assume_default)
        b, over = discretize_dose(v)
        vals.append(v); bins.append(b); flags.append(over); units.append(unit)
        parsed_c.append(parse_concentration_mcg_per_ml(name))
        used_c.append(conc_used); sources.append(source)
        assumed.append(bool(unit == "ml/hr" and source == DOSE_SOURCE_ASSUMED))

    out["dex_dose_mcg_kg_h"] = vals
    out["dose_bin"] = bins
    out[DOSE_OVER_RANGE_FLAG] = flags
    out["dose_unit_parsed"] = units
    out["concentration_parsed_mcg_per_ml"] = parsed_c
    out["concentration_used_mcg_per_ml"] = used_c
    out[DOSE_SOURCE_COL] = sources
    out[DOSE_ASSUMED_FLAG] = assumed
    return out.drop(columns=["_weight_used"])


def sensitivity_table(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """方案 A 的敏感性档位：2.0 / 4.0 / 8.0 mcg/ml 三档剂量分布对比。"""
    rows = []
    base = convert_dataframe(df, **kwargs)   # 默认路径：标记哪些记录靠 4.0 估算
    n_assumed = int(base[DOSE_ASSUMED_FLAG].sum())
    for conc in SENSITIVITY_CONCENTRATIONS:
        res = convert_dataframe(df, concentration_mcg_per_ml=conc, **kwargs)
        v = pd.to_numeric(res["dex_dose_mcg_kg_h"], errors="coerce").dropna()
        rows.append({
            "assumed_concentration_mcg_per_ml": conc,
            "n_convertible": int(len(v)),
            "dose_mean": round(float(v.mean()), 4) if len(v) else None,
            "dose_median": round(float(v.median()), 4) if len(v) else None,
            "dose_p95": round(float(v.quantile(0.95)), 4) if len(v) else None,
            "over_range_frac": round(float(res[DOSE_OVER_RANGE_FLAG].mean()), 4),
            "n_assumed_mlhr": n_assumed,   # 受 4.0 估算影响的记录数（方案 B 剔除对象）
        })
    return pd.DataFrame(rows)


def _selfcheck():
    cases = [
        # (rate, drugname, weight, expected, assume_default)
        (0.7, "Dexmedetomidine (mcg/kg/hr)", 70.0, 0.7, True),
        (0.01, "Dexmedetomidine (mcg/kg/min)", 70.0, 0.6, True),
        (70.0, "Dexmedetomidine (mcg/hr)", 70.0, 1.0, True),
        (0.07, "Dexmedetomidine (mg/hr)", 70.0, 1.0, True),
        (70.0 / 60.0, "Dexmedetomidine (mcg/min)", 70.0, 1.0, True),
        (0.0007, "Dexmedetomidine (mg/kg/hr)", 70.0, 0.7, True),
        # 方案 A：无显式浓度 -> 默认 4.0 填补
        (5.0, "Dexmedetomidine (ml/hr)", 70.0, 5.0 * 4 / 70.0, True),
        # 方案 B：关闭默认填补 -> None
        (5.0, "Dexmedetomidine (ml/hr)", 70.0, None, False),
        (5.0, "Dexmedetomidine 4 mcg/ml (ml/hr)", 70.0, 5.0 * 4 / 70.0, True),
        (5.0, "Dexmedetomidine 200 mcg/50ml (ml/hr)", 70.0, 5.0 * 4 / 70.0, True),
        (5.0, "Dexmedetomidine Inj 400 Mcg in Dextrose 5% 100 ml (ml/hr)", 70.0,
         5.0 * 4 / 70.0, True),
        (5.0, "Dexmedetomidine Inj 400 Mcg in Sodium Chloride 0.9% 100 ml (ml/hr)",
         70.0, 5.0 * 4 / 70.0, True),
        (5.0, "Dexmedetomidine", 70.0, None, True),   # 单位未知 -> 不猜测
        (0.0, "Dexmedetomidine (mcg/kg/hr)", 70.0, 0.0, True),
    ]
    ok = 0
    for rate, name, w, exp, assume in cases:
        got = convert_to_mcg_kg_h(rate, name, w, assume_default=assume)
        good = (exp is None and got is None) or (exp is not None and got is not None
                                                 and abs(got - exp) < 1e-6)
        ok += int(good)
        print("  %-46s -> %-10s %s" % (name[:46], str(got), "OK" if good else "FAIL"))

    # 浓度来源标记
    c, s = resolve_concentration("Dexmedetomidine (ml/hr)")
    assert c == 4.0 and s == DOSE_SOURCE_ASSUMED
    c, s = resolve_concentration("Dexmedetomidine 200 mcg/50ml (ml/hr)")
    assert c == 4.0 and s == DOSE_SOURCE_EXPLICIT
    c, s = resolve_concentration("Dexmedetomidine (ml/hr)", assume_default=False)
    assert c is None and s == DOSE_SOURCE_UNRESOLVED
    print("  concentration source flags OK")

    demo = pd.DataFrame({
        "drugname": ["Dexmedetomidine (ml/hr)", "Dexmedetomidine 200 mcg/50ml (ml/hr)",
                     "Dexmedetomidine (mcg/kg/hr)"],
        "drugrate": [5.0, 5.0, 0.7],
        "admissionweight": [70.0, 70.0, 70.0],
    })
    tagged = convert_dataframe(demo)
    assert int(tagged[DOSE_ASSUMED_FLAG].sum()) == 1
    assert tagged.loc[0, "concentration_used_mcg_per_ml"] == 4.0
    assert tagged.loc[2, DOSE_SOURCE_COL] == DOSE_SOURCE_NA
    tab = sensitivity_table(demo)
    assert len(tab) == 3 and tab["dose_mean"].is_monotonic_increasing
    print("  sensitivity table (2/4/8 mcg/ml):")
    print(tab.to_string(index=False, max_cols=6))

    assert discretize_dose(0.05)[0] == 1
    assert discretize_dose(0.15)[0] == 2
    assert discretize_dose(1.45)[0] == 15
    assert discretize_dose(1.60) == (15, True)
    assert discretize_dose(None)[0] == 0
    assert parse_concentration_mcg_per_ml("4 mcg/ml") == 4.0
    assert parse_concentration_mcg_per_ml("Dexmedetomidine (ml/hr)") is None
    print("  dose bins: %d intervals, edges %s" % (len(DOSE_BIN_LABELS), DOSE_BIN_EDGES))
    print("SELFCHECK dose: %d/%d unit conversions passed" % (ok, len(cases)))
    return ok == len(cases)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--input", default=None, help="infusionDrug CSV/CSV.GZ 路径")
    ap.add_argument("--output", default=None)
    ap.add_argument("--concentration", type=float, default=None,
                    help="覆盖默认浓度；敏感性档位可传 2.0 或 8.0")
    ap.add_argument("--no-default-concentration", action="store_true",
                    help="方案 B：关闭 4.0 默认填补")
    ap.add_argument("--sensitivity", action="store_true",
                    help="输出 2.0 / 4.0 / 8.0 三档敏感性对比表")
    args = ap.parse_args()
    if args.selfcheck:
        raise SystemExit(0 if _selfcheck() else 1)
    if args.input:
        df = pd.read_csv(args.input)
        res = convert_dataframe(
            df, concentration_mcg_per_ml=args.concentration,
            assume_default=not args.no_default_concentration,
        )
        dst = args.output or "dex_dose_converted.csv"
        res.to_csv(dst, index=False, encoding="utf-8-sig")
        print("written:", dst, "rows:", len(res),
              "unconverted:", int(res["dex_dose_mcg_kg_h"].isna().sum()),
              "assumed_mlhr:", int(res[DOSE_ASSUMED_FLAG].sum()))
        if args.sensitivity:
            print("\n=== sensitivity (方案 A 浓度假设) ===")
            print(sensitivity_table(df).to_string(index=False))
    else:
        print(__doc__)
