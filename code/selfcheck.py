# -*- coding: utf-8 -*-
"""
阶段 3 建模代码统一自检
=============================================================================
用途：用【合成数据】验证所有脚本可导入、可执行、图表可生成；
      验证 GroupKFold 无跨院泄漏、剂量换算与 Reward 规则正确。

【重要】本脚本产出的所有数字都来自合成数据，
        【不是研究结果】，不得进入 Methods / Results 或任何论文图表。

未执行项（会如实写入报告）：
- 真实 eICU 分析长表：SQL 骨架 v1.2 尚未由 PI 在本地跑出，文件不存在；
- R gfoRmula 包：本地 R 4.6.1 未安装，仅基础 R 版 g-formula 实跑；
- CQL 训练：本地 torch / d3rlpy 未安装，仅接口模板。

运行：python selfcheck.py
输出：outputs/selfcheck_report.json + 5 张 SELFCHECK_*.png
"""

from __future__ import annotations

import importlib
import json
import py_compile
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_config import OUTPUT_DIR, RANDOM_SEED  # noqa: E402

MODULES = [
    "01_dex_dose", "02_dataset_and_iptw", "03b_gformula_py",
    "04_cate_models", "05_plots", "06_ope_cql",
]
RSCRIPT = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
R_FILE = str(HERE / "03_gformula.R")


def section(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68, flush=True)


def main():
    report = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "synthetic_data_only": True,
        "warning": "All numbers below are code self-checks on synthetic data, NOT research results.",
        "checks": {}, "not_executed": [], "limitations": [],
    }

    # ---------------------------------------------------------------- 1 语法
    section("[1/8] Python syntax compile")
    compiled, errors = [], []
    for f in sorted(HERE.glob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True)
            compiled.append(f.name)
        except Exception as e:
            errors.append({"file": f.name, "error": str(e)[:200]})
    report["checks"]["syntax_compile"] = {"passed": compiled, "failed": errors}
    print("  compiled %d files, %d failures" % (len(compiled), len(errors)))
    for e in errors:
        print("   FAIL", e)

    # ---------------------------------------------------------------- 2 模块导入
    section("[2/8] Module import")
    mods, imp_err = {}, []
    for name in MODULES:
        try:
            mods[name] = importlib.import_module(name)
            print("  OK  ", name)
        except Exception as e:
            imp_err.append({"module": name, "error": str(e)[:250]})
            print("  FAIL", name, str(e)[:150])
    report["checks"]["module_import"] = {"failed": imp_err}

    # ---------------------------------------------------------------- 3 剂量换算
    section("[3/8] DEX dose conversion (task C)")
    dose = mods["01_dex_dose"]
    dose_ok = dose._selfcheck()
    report["checks"]["dose_conversion"] = {
        "unit_tests_passed": bool(dose_ok),
        "target_unit": "mcg/kg/h",
        "n_bins": 15, "bin_range": "0 ~ 1.5, step 0.1",
    }

    # ---------------------------------------------------------------- 4 长表校验 + IPTW
    section("[4/8] Long-table validation + IPTW")
    gf_mod = mods["03b_gformula_py"]
    ds = mods["02_dataset_and_iptw"]
    syn_long = gf_mod.make_synthetic_long(n_patients=240, n_steps=20,
                                          seed=RANDOM_SEED)
    val = ds.validate_long_table(syn_long)
    mat = ds.build_model_matrix(syn_long)
    iptw = ds.fit_iptw_weights(mat)
    report["checks"]["long_table_validation"] = val
    report["checks"]["iptw"] = iptw.attrs.get("iptw_diag", {})
    print("  rows=%d patients=%d hospitals=%d hard_checks_pass=%s"
          % (val["rows"], val["n_patients"], val["n_hospitals"], val["hard_checks_pass"]))
    print("  IPTW:", json.dumps(report["checks"]["iptw"], ensure_ascii=False))

    # ---------------------------------------------------------------- 5 g-formula (Python)
    section("[5/8] Parametric G-formula (Python, competing risks)")
    gf = gf_mod.ParametricGFormula().fit(syn_long)
    res_gf = gf.compare(n_simul=240, n_boot=20)
    report["checks"]["gformula_python"] = {
        k: v for k, v in res_gf.items() if not k.startswith("cif")
    }
    print("  RR=%.4f  RD=%.4f  risk_early=%.4f  risk_control=%.4f  1-KM=%s"
          % (res_gf["risk_ratio"], res_gf["risk_difference"],
             res_gf["risk_early_dex"], res_gf["risk_no_early_dex"],
             res_gf["one_minus_km_used"]))

    # ---------------------------------------------------------------- 6 CATE + GroupKFold
    section("[6/8] CATE with GroupKFold(groups=hospitalid)")
    cate_mod = mods["04_cate_models"]
    df_patient = cate_mod.aggregate_to_patient_level(syn_long)
    cate_results = {}
    saved_cate = {}
    for model_name in ["causal_forest", "rlearner"]:
        print("  -- model:", model_name)
        try:
            r = cate_mod.cross_fitted_cate(df_patient, model=model_name, n_splits=5)
            saved_cate[model_name] = r
            cate_results[model_name] = {
                "model_used": r["model_used"],
                "n_splits": r["n_splits"],
                "group_leakage": r["group_leakage"],
                "cate_nan": int(sum(1 for x in r["cate"] if x != x)),
                "cate_mean": float(float(__import__("numpy").nanmean(r["cate"]))),
                "cate_sd": float(float(__import__("numpy").nanstd(r["cate"]))),
            }
        except Exception as e:
            cate_results[model_name] = {"error": "%s: %s" % (type(e).__name__, str(e)[:250])}
            print("   FAIL", cate_results[model_name]["error"])
    report["checks"]["cate_groupkfold"] = cate_results

    # ---------------------------------------------------------------- 7 图表
    section("[7/8] Publication-style figures")
    plots = mods["05_plots"]
    res_cate = saved_cate.get("causal_forest")
    if res_cate is None:   # 仅当上一步失败时才重跑
        res_cate = cate_mod.cross_fitted_cate(df_patient, model="causal_forest", n_splits=5)
    fig_info = plots.run_all(res_cate, res_gformula=res_gf, prefix="SELFCHECK_")
    pngs = sorted(p.name for p in OUTPUT_DIR.glob("SELFCHECK_*.png"))
    vecs = sorted(p.name for p in list(OUTPUT_DIR.glob("SELFCHECK_*.svg"))
                  + list(OUTPUT_DIR.glob("SELFCHECK_*.pdf")))
    report["checks"]["plots"] = {"metrics": fig_info, "files": pngs,
                                 "vector_files": vecs}
    print("  written:", len(pngs), "PNG +", len(vecs), "矢量图(SVG/PDF)")

    # ---------------------------------------------------------------- 8 OPE + R
    section("[8/8] OPE (FQE/WIS/CQL template) + R g-formula")
    ope = mods["06_ope_cql"].selfcheck()
    report["checks"]["ope"] = ope

    try:
        p = subprocess.run([RSCRIPT, R_FILE, "--selfcheck"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300)
        r_out = (p.stdout or "") + (p.stderr or "")
        report["checks"]["r_base_gformula"] = {
            "returncode": p.returncode,
            "output_tail": r_out.strip().splitlines()[-14:],
        }
        print("  Rscript returncode:", p.returncode)
        for line in report["checks"]["r_base_gformula"]["output_tail"]:
            print("   ", line[:150])
    except Exception as e:
        report["checks"]["r_base_gformula"] = {"error": str(e)[:250]}
        print("  R FAILED:", str(e)[:150])

    # ---------------------------------------------------------------- 汇总
    report["not_executed"] = [
        "真实 eICU 分析长表（SQL 骨架 v1.2 未在本地跑出，analysis_long.csv 不存在）",
        "R gfoRmula 包路径（R 4.6.1 未安装 gfoRmula，仅基础 R 版 g-formula 实跑）",
        "CQL 训练（torch / d3rlpy 未安装，仅接口模板）",
        "完整时变协变量演化模型、clone-censor-weight、删失权重（待 TTE 协议表）",
    ]
    report["limitations"] = [
        "所有数字均为合成数据自检，不是研究结论",
        "Qini / cARD / C-for-benefit 基于观测结局的 uplift 评估，非真实 ITE 验证",
        "SHAP 作用于 CATE 代理树，不是直接解释因果森林",
        "DEX ml/hr 单位需配置浓度参数，未确认前返回 None，不猜测",
    ]

    out = OUTPUT_DIR / "selfcheck_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    section("DONE")
    print("  report:", out)
    print("  synthetic data only — NOT research results")
    return report


if __name__ == "__main__":
    main()
