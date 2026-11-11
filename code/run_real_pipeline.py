# -*- coding: utf-8 -*-
"""
真实长表到位后的四步流水线（02 -> 03b -> 04 -> 05）

用法
----
  # 1) 正式运行（真实长表放到 modeling/outputs/analysis_long.csv 后）
  python run_real_pipeline.py

  # 2) 指定长表路径
  python run_real_pipeline.py --long-table "D:/somewhere/analysis_long.csv"

  # 3) 端到端演练（用合成数据，所有输出隔离到 outputs/_PIPELINE_DEMO/，
  #    文件名带 DEMO_ 前缀；【绝不写入正式 outputs 根目录】）
  python run_real_pipeline.py --demo

  # 4) 正式运行但先小规模试算
  python run_real_pipeline.py --n-simul 500 --n-boot 30

步骤
----
  1/4  02_dataset_and_iptw.py  -> 长表校验 + IPTW 权重，输出 analysis_long_iptw.csv
  2/4  03b_gformula_py.py     -> 反事实 CIF / RR / RD，输出 gformula_results.json
  3/4  04_cate_models.py      -> GroupKFold 交叉拟合 CATE，输出 cate_oof.csv
                                 （自动识别长表并聚合为病人级）
  4/4  05_plots.py            -> cARD/Qini/C-for-benefit/SHAP/CIF 五张图

重要
----
- 真实长表尚不存在时，正式模式【直接退出，不运行、不伪造结果】。
- 演练模式产出的数字仍是合成数据，不是研究结论。
- 首次真实结果须由 PI 审核后才能进入 Methods / Results。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from model_config import OUTPUT_DIR, LONG_TABLE_PATH, N_SIMUL, N_BOOTSTRAP  # noqa: E402


def run_step(label, cmd):
    print("\n" + "-" * 68)
    print(label)
    print("-" * 68, flush=True)
    p = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    tail = (p.stdout or "").strip().splitlines()[-12:]
    for line in tail:
        print("  ", line[:180])
    if p.returncode != 0:
        print("  STDERR:", (p.stderr or "")[-1500:])
        raise SystemExit("STEP FAILED (exit=%d): %s" % (p.returncode, label))
    print("  OK", flush=True)
    return p.stdout or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="合成数据端到端演练")
    ap.add_argument("--long-table", default=str(LONG_TABLE_PATH))
    ap.add_argument("--n-simul", type=int, default=N_SIMUL)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--model", default="causal_forest",
                    choices=["causal_forest", "rlearner"])
    a = ap.parse_args()

    if a.demo:
        out_dir = OUTPUT_DIR / "_PIPELINE_DEMO"
        out_dir.mkdir(parents=True, exist_ok=True)
        import importlib
        syn = importlib.import_module("03b_gformula_py").make_synthetic_long(
            n_patients=240, n_steps=20
        )
        long_path = out_dir / "analysis_long_DEMO.csv"
        # 模拟 SQL 补丁已追加 is_assumed_mlhr 标记，用于演练方案 B 的精确剔除路径
        rng = np.random.default_rng(20261109)
        syn["is_assumed_mlhr"] = (rng.random(len(syn)) < 0.30).astype(int)
        syn["dose_concentration_source"] = np.where(
            syn["is_assumed_mlhr"] == 1, "assumed_default", "not_applicable"
        )
        syn.to_csv(long_path, index=False, encoding="utf-8-sig")
        prefix = "DEMO_"
        n_simul, n_boot = 100, 5
        print("DEMO MODE: synthetic long table ->", long_path)
        print("all outputs isolated in", out_dir)
    else:
        out_dir = OUTPUT_DIR
        long_path = Path(a.long_table)
        prefix = ""
        n_simul, n_boot = a.n_simul, a.n_boot
        if not long_path.exists():
            print("=" * 68)
            print("真实长表尚未导入，流水线【未运行】。")
            print("期望路径:", long_path)
            print("请把 SQL 骨架 v1.2 产出的 analysis_long.csv 放入该路径后重跑。")
            print("如需先验证链路，请执行: python run_real_pipeline.py --demo")
            print("=" * 68)
            return
        print("REAL MODE: long table =", long_path)

    weighted = out_dir / "analysis_long_iptw.csv"
    gf_json = out_dir / "gformula_results.json"
    cate_csv = out_dir / "cate_oof.csv"

    py = sys.executable
    run_step("1/4 长表校验 + IPTW 权重 + 方案B子集报告", [
        py, "02_dataset_and_iptw.py", "--input", str(long_path),
        "--output", str(weighted),
        "--subset-report", str(out_dir / "subset_validation_02.json"),
    ])
    run_step("2/4 参数化 G-formula：反事实 CIF / RR / RD", [
        py, "03b_gformula_py.py", "--input", str(weighted),
        "--output", str(gf_json), "--n-simul", str(n_simul), "--n-boot", str(n_boot),
    ])
    run_step("3/4 CATE：GroupKFold(groups=hospitalid) + 方案B子集报告", [
        py, "04_cate_models.py", "--input", str(long_path),
        "--output", str(cate_csv), "--model", a.model,
        "--subset-report", str(out_dir / "subset_validation_04.json"),
    ])
    run_step("4/4 顶刊图表（PNG + SVG/PDF 矢量）", [
        py, "05_plots.py", "--from-files", "--cate", str(cate_csv),
        "--gformula", str(gf_json), "--prefix", prefix,
        "--outdir", str(out_dir), "--formats", "png,svg,pdf",
    ])

    produced = sorted(p.name for p in out_dir.glob("*.png"))
    vector_files = sorted(p.name for p in
                          list(out_dir.glob("*.svg")) + list(out_dir.glob("*.pdf")))
    subset_reports = [str(p) for p in
                      [out_dir / "subset_validation_02.json",
                       out_dir / "subset_validation_04.json"] if p.exists()]
    manifest = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "DEMO (synthetic)" if a.demo else "REAL",
        "long_table": str(long_path),
        "outputs": {
            "weighted_long_table": str(weighted),
            "gformula_results": str(gf_json),
            "cate_oof": str(cate_csv),
            "png_files": produced,
            "vector_files": vector_files,
            "subset_validation_reports": subset_reports,
        },
        "params": {"n_simul": n_simul, "n_boot": n_boot, "cate_model": a.model},
        "warning": "DEMO 输出为合成数据，不是研究结论；REAL 输出须经 PI 审核。",
    }
    mf = out_dir / "pipeline_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("PIPELINE DONE | mode:", manifest["mode"])
    print("  weighted :", weighted.name, weighted.exists())
    print("  gformula :", gf_json.name, gf_json.exists())
    print("  cate     :", cate_csv.name, cate_csv.exists())
    print("  figures  :", len(produced), "PNG +", len(vector_files), "矢量(SVG/PDF)")
    print("  方案B报告:", len(subset_reports), "份")
    print("  manifest :", mf)
    print("=" * 68)


if __name__ == "__main__":
    main()
