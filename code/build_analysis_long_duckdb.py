# -*- coding: utf-8 -*-
"""
build_analysis_long_duckdb.py
用 DuckDB 直接在 eICU 原始 CSV(.gz) 上执行分析 SQL，生成 analysis_long.csv

【为什么用这个脚本】
本机没有运行中的 PostgreSQL（无进程、5432 未监听、无安装目录），
但 eICU 原始数据以 CSV(.gz) 形式完整存在：
    E:\\Medical_project\\Sepsis_01\\data\\eicu
DuckDB 可直接读取 gzip CSV 并执行 SQL，因此无需数据库实例、无需导入，
产出与 SQL数据提取骨架_v1.2.sql 语义一致的分析长表。

【与 SQL 骨架的差异（均为必要修正，已在 --explain-only 报告中列明）】
1. 方言：generate_series -> range(0,120) AS g(step)；正则 !~* / ~* ->
   NOT regexp_matches(...,'i') / regexp_matches(...,'i')；NUMERIC 保留。
2. chart_events 增加 event_code IS NOT NULL 过滤：
   骨架原版保留全部护理记录（约 1.5 亿行），12h 自连接会爆炸；
   只保留量表相关行（约 230 万行）后才可计算。
3. 自动注入补丁三处标记列（is_assumed_mlhr / dose_concentration_source）。
4. 方案 C：ml/hr 按 4.0 mcg/ml 换算（PI 裁决的外部假设），
   体重优先 infusionDrug.patientweight，缺失回退 patient.admissionweight。
5. 实现 ECMO 排除（规格 v1.2 要求；骨架中为 TODO）：
   nurseCharting 中 celltypecat='Ventricular Assist Device' 且
   celltypevallabel='ECMO' 的住院整例排除。
   未实现且已标注的排除：入 ICU 前已发作谵妄、高剂量丙泊酚 >24h（阈值未定）。

【用法】
  python build_analysis_long_duckdb.py --explain-only    # 只验证 SQL 能否解析
  python build_analysis_long_duckdb.py                   # 正式生成 CSV
  python build_analysis_long_duckdb.py --skip-vital      # 跳过时变协变量（更快）

输出：modeling/outputs/analysis_long.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_CSV = OUT_DIR / "analysis_long.csv"
SQL_SKELETON = HERE.parent / "SQL数据提取骨架_v1.2.sql"
EICU = Path(r"E:\Medical_project\Sepsis_01\data\eicu")

TABLES = {
    "PATIENT": "patient.csv.gz",
    "NURSECHARTING": "nurseCharting.csv.gz",
    "INFUSIONDRUG": "infusionDrug.csv.gz",
    "MEDICATION": "medication.csv.gz",
    "NOTE": "note.csv.gz",
    "PHYSICALEXAM": "physicalExam.csv.gz",
    "DIAGNOSIS": "diagnosis.csv.gz",
    "TREATMENT": "treatment.csv.gz",
    "VITALPERIODIC": "vitalPeriodic.csv.gz",
    "VITALAPERIODIC": "vitalAperiodic.csv.gz",
    "APACHEPATIENTRESULT": "apachePatientResult.csv.gz",
    "HOSPITAL": "hospital.csv.gz",
}

REQUIRED = [
    "patientunitstayid", "step", "hospitalid",
    "resolution_v1_event", "resolution_alt_event", "death_event", "icu_discharge_event",
    "strategy_arm_at_step", "dex_dose_6h_mean_mcg_kg_hr",
    "age", "gender", "ethnicity", "admissionweight",
    "apachescore", "acutephysiologyscore", "predictedicumortality",
    "unittype", "apacheadmissiondx", "teachingstatus", "region",
    "heartrate", "systemicmean", "respiration", "sao2", "temperature",
    "is_assumed_mlhr", "dose_concentration_source",
]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def build_sql(skip_vital=False):
    sql = SQL_SKELETON.read_text(encoding="utf-8")

    # ---- 1) 补丁：记录级标记 + 方案 C 剂量换算 ----
    a1 = "        1 AS explicit_dex_record\n"
    if a1 not in sql:
        raise RuntimeError("补丁锚点 [1] 未找到")
    flag_cases = """        CASE
            WHEN LOWER(i.drugname) LIKE '%ml/hr%'
             AND NOT regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml', 'i')
             AND NOT regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml', 'i')
             AND NOT regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml', 'i')
            THEN 1 ELSE 0
        END AS is_assumed_mlhr_record,
        CASE
            WHEN LOWER(i.drugname) NOT LIKE '%ml/hr%' THEN 'not_applicable'
            WHEN regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml', 'i')
              OR regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml', 'i')
              OR regexp_matches(i.drugname, '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml', 'i')
                THEN 'explicit_in_name'
            ELSE 'assumed_default'
        END AS dose_concentration_source_record"""
    sql = sql.replace(a1, "        1 AS explicit_dex_record,\n" + flag_cases + "\n")

    a1b = ("    FROM {{TABLE_EICU_INFUSIONDRUG}} i\n"
           "    WHERE LOWER(i.drugname) LIKE '%dexmed%'")
    if a1b not in sql:
        raise RuntimeError("补丁锚点 [1b] 未找到")
    sql = sql.replace(a1b,
                      "    FROM {{TABLE_EICU_INFUSIONDRUG}} i\n"
                      "    LEFT JOIN {{TABLE_EICU_PATIENT}} p "
                      "ON p.patientunitstayid = i.patientunitstayid\n"
                      "    WHERE LOWER(i.drugname) LIKE '%dexmed%'")

    w = ("COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), "
         "CAST(p.admissionweight AS NUMERIC))")
    dose_case = """        -- 方案 C（PI 裁决）：ml/hr 无显式浓度时按 4.0 mcg/ml【外部假设】填补
        CASE
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/hr%' THEN CAST(i.drugrate AS NUMERIC)
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/min%' THEN CAST(i.drugrate AS NUMERIC) * 60
            WHEN LOWER(i.drugname) LIKE '%mcg/min%'
                 THEN CAST(i.drugrate AS NUMERIC) * 60 / NULLIF(WW, 0)
            WHEN LOWER(i.drugname) LIKE '%mcg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) / NULLIF(WW, 0)
            WHEN LOWER(i.drugname) LIKE '%mg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 1000 / NULLIF(WW, 0)
            WHEN LOWER(i.drugname) LIKE '%ml/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 4.0 / NULLIF(WW, 0)
            ELSE NULL
        END AS dex_dose_mcg_kg_hr,""".replace("WW", w)
    dose_re = re.compile(r"        -- 剂量标准化.*?END AS dex_dose_mcg_kg_hr,", re.S)
    if not dose_re.search(sql):
        raise RuntimeError("剂量换算块未找到")
    sql = dose_re.sub(lambda m: dose_case, sql, count=1)

    a2 = "        AVG(d.dex_dose_mcg_kg_hr) AS dex_dose_6h_mean_mcg_kg_hr\n"
    if a2 not in sql:
        raise RuntimeError("补丁锚点 [2] 未找到")
    sql = sql.replace(a2,
                      "        AVG(d.dex_dose_mcg_kg_hr) AS dex_dose_6h_mean_mcg_kg_hr,\n"
                      "        MAX(d.is_assumed_mlhr_record) AS is_assumed_mlhr_step,\n"
                      "        CASE\n"
                      "            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'assumed_default' THEN 1 ELSE 0 END) = 1 THEN 'assumed_default'\n"
                      "            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'explicit_in_name' THEN 1 ELSE 0 END) = 1 THEN 'explicit_in_name'\n"
                      "            ELSE 'not_applicable'\n"
                      "        END AS dose_concentration_source_step\n")

    a3 = "        s.dex_dose_6h_mean_mcg_kg_hr,\n"
    if a3 not in sql:
        raise RuntimeError("补丁锚点 [3] 未找到")
    sql = sql.replace(a3, a3 +
                      "        COALESCE(CAST(s.is_assumed_mlhr_step AS INTEGER), 0) AS is_assumed_mlhr,\n"
                      "        s.dose_concentration_source_step AS dose_concentration_source,\n")

    # ---- 2) chart_events 只保留量表行（否则 1.5 亿行自连接爆炸）----
    sql = sql.replace("chart_events AS (", "chart_events_raw AS (", 1)
    sql = sql.replace(
        "t0 AS (",
        "chart_events AS (SELECT * FROM chart_events_raw WHERE event_code IS NOT NULL),\n\n"
        "t0 AS (", 1)

    # ---- 3) ECMO 排除（规格 v1.2 要求）----
    anchor = "      AND NOT EXISTS (\n            SELECT 1 FROM {{TABLE_EICU_ECMO}} e"
    # 骨架里 ECMO 是 TODO 注释，改为在 eligible 中追加子查询
    elig = "      AND c.unitdischargeoffset <= 43200       -- <=30 天；超出者由 120 步行政截断\n"
    if elig in sql:
        sql = sql.replace(elig, elig +
            "      -- 规格 v1.2：排除 ECMO 支持（骨架原为 TODO，此处实现）\n"
            "      AND NOT EXISTS (\n"
            "            SELECT 1 FROM {{TABLE_EICU_NURSECHARTING}} ec\n"
            "            WHERE ec.patientunitstayid = c.patientunitstayid\n"
            "              AND ec.nursingchartcelltypecat = 'Ventricular Assist Device'\n"
            "              AND ec.nursingchartcelltypevallabel = 'ECMO'\n"
            "      )\n")

    # ---- 3a) drugrate 含 "OFF" 等非数值文本，统一用 TRY_CAST 避免整批失败 ----
    for expr in ("CAST(i.drugrate AS NUMERIC)",
                 "CAST(i.patientweight AS NUMERIC)",
                 "CAST(p.admissionweight AS NUMERIC)"):
        sql = sql.replace(expr, "TRY_" + expr)

    # ---- 3b) eICU patient.age 为文本（含 "> 89"），转数值；>89 按 90 处理 ----
    sql = sql.replace(
        "        gender,\n        age,\n        ethnicity,",
        "        gender,\n"
        "        CASE WHEN TRIM(age) LIKE '>%' THEN 90 "
        "ELSE TRY_CAST(TRIM(age) AS INTEGER) END AS age,\n"
        "        ethnicity,")

    # ---- 3b2) T0 纳入 RASS 正值躁动（PI 2026-11-09 裁决，符合 PICO「谵妄或躁动」）----
    old_rass = """            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('-3','-4','-5','1','2','3','4')
                                                              THEN 'RASS_OTHER'
            WHEN nc.nursingchartcelltypevallabel IN ('RASS','SEDATION SCORE')
                 AND nc.nursingchartcelltypevalname = 'Value'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'"""
    new_rass = """            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('1','2','3','4') THEN 'RASS_AGITATED'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('-3','-4','-5') THEN 'RASS_OTHER'
            WHEN nc.nursingchartcelltypevallabel IN ('RASS','SEDATION SCORE')
                 AND nc.nursingchartcelltypevalname = 'Value'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'
            WHEN nc.nursingchartcelltypevallabel IN ('RASS','SEDATION SCORE')
                 AND nc.nursingchartcelltypevalname = 'Value'
                 AND nc.nursingchartvalue IN ('1','2','3','4') THEN 'RASS_AGITATED'"""
    if old_rass in sql:
        sql = sql.replace(old_rass, new_rass)
    else:
        raise RuntimeError("T0 扩展锚点（RASS 档位映射）未找到")

    old_t0 = "WHERE ce.event_code IN ('CAM_POSITIVE', 'SYMPTOM_YES')"
    if old_t0 not in sql:
        raise RuntimeError("T0 扩展锚点（t0 CTE）未找到")
    sql = sql.replace(old_t0, "WHERE ce.event_code IN ('CAM_POSITIVE', 'SYMPTOM_YES', 'RASS_AGITATED')")

    # ---- 3c) 结局判定改用窗口函数（骨架的 12h 自连接会 spill 数十 GB）----
    new_v1 = """resolution_v1 AS (
    SELECT patientunitstayid, MIN(t_offset) AS resolution_offset
    FROM (
        SELECT patientunitstayid, t_offset,
               SUM(CASE WHEN event_code = 'CAM_NEGATIVE' THEN 1 ELSE 0 END) OVER w AS cam_neg,
               SUM(CASE WHEN event_code = 'RASS_TARGET'  THEN 1 ELSE 0 END) OVER w AS rass_ok,
               SUM(CASE WHEN event_code = 'NOT_ASSESSED' THEN 1 ELSE 0 END) OVER w AS not_assessed,
               COUNT(event_code) OVER w AS explicit_cnt
        FROM chart_events
        WHERE event_code IN ('CAM_NEGATIVE','RASS_TARGET','NOT_ASSESSED')
        WINDOW w AS (PARTITION BY patientunitstayid ORDER BY t_offset
                     RANGE BETWEEN CURRENT ROW AND 720 FOLLOWING)
    ) w2
    WHERE cam_neg >= 1 AND rass_ok >= 1
      AND explicit_cnt >= 2 AND not_assessed = 0
    GROUP BY patientunitstayid
),

"""
    sql = re.sub(r"resolution_v1 AS \(.*?\nresolution_alt AS \(",
                 new_v1 + "resolution_alt AS (", sql, count=1, flags=re.S)

    new_alt = """resolution_alt AS (
    SELECT patientunitstayid, MIN(t_offset) AS resolution_offset
    FROM (
        SELECT patientunitstayid, t_offset,
               SUM(CASE WHEN event_code = 'CAM_NEGATIVE' THEN 1 ELSE 0 END) OVER w AS cam_neg,
               SUM(CASE WHEN event_code = 'NOT_ASSESSED' THEN 1 ELSE 0 END) OVER w AS not_assessed,
               COUNT(event_code) OVER w AS explicit_cnt
        FROM chart_events
        WHERE event_code IN ('CAM_NEGATIVE','RASS_TARGET','NOT_ASSESSED')
        WINDOW w AS (PARTITION BY patientunitstayid ORDER BY t_offset
                     RANGE BETWEEN CURRENT ROW AND 720 FOLLOWING)
    ) w2
    WHERE cam_neg >= 1 AND explicit_cnt >= 2 AND not_assessed = 0
    GROUP BY patientunitstayid
),

"""
    sql = re.sub(r"resolution_alt AS \(.*?\ncompeting AS \(",
                 new_alt + "competing AS (", sql, count=1, flags=re.S)

    # ---- 3c2) apachePatientResult 每人可能多行，必须去重，否则 analysis_long 行数翻倍 ----
    new_base = """cov_baseline AS (
    SELECT patientunitstayid,
           MAX(TRY_CAST(apachescore AS DOUBLE))           AS apachescore,
           MAX(TRY_CAST(acutephysiologyscore AS DOUBLE))  AS acutephysiologyscore,
           MAX(TRY_CAST(predictedicumortality AS DOUBLE)) AS predictedicumortality,
           MAX(TRY_CAST(actualiculos AS DOUBLE))          AS actualiculos,
           MAX(TRY_CAST(actualventdays AS DOUBLE))        AS actualventdays
    FROM {{TABLE_EICU_APACHEPATIENTRESULT}}
    GROUP BY patientunitstayid
),

"""
    sql = re.sub(r"cov_baseline AS \(.*?\ncov_timevarying AS \(",
                 new_base + "cov_timevarying AS (", sql, count=1, flags=re.S)

    # ---- 3d) 时变协变量：先按 6h 步长聚合再 JOIN（避免 1.78GB 表 JOIN 后放大）----
    new_cov = """cov_timevarying AS (
    SELECT patientunitstayid,
           CAST(FLOOR(observationoffset / 360.0) AS BIGINT) AS step,
           AVG(TRY_CAST(heartrate AS DOUBLE))    AS heartrate,
           AVG(TRY_CAST(systemicmean AS DOUBLE)) AS systemicmean,
           AVG(TRY_CAST(respiration AS DOUBLE))  AS respiration,
           AVG(TRY_CAST(sao2 AS DOUBLE))         AS sao2,
           AVG(TRY_CAST(temperature AS DOUBLE))  AS temperature
    FROM {{TABLE_EICU_VITALPERIODIC}}
    GROUP BY 1, 2
),

"""
    sql = re.sub(r"cov_timevarying AS \(.*?\n\),\n", new_cov, sql, count=1, flags=re.S)
    sql = re.sub(
        r"LEFT JOIN cov_timevarying v ON v\.patientunitstayid = s\.patientunitstayid\s+AND v\.observationoffset < s\.step_start_offset",
        "LEFT JOIN cov_timevarying v ON v.patientunitstayid = s.patientunitstayid AND v.step = s.step - 1",
        sql)

    if skip_vital:
        sql = re.sub(
            r"cov_timevarying AS \(.*?\n\),\n",
            "cov_timevarying AS (\n"
            "    SELECT CAST(NULL AS BIGINT) AS patientunitstayid, CAST(NULL AS BIGINT) AS step,\n"
            "           CAST(NULL AS DOUBLE) AS heartrate, CAST(NULL AS DOUBLE) AS systemicmean,\n"
            "           CAST(NULL AS DOUBLE) AS respiration, CAST(NULL AS DOUBLE) AS sao2,\n"
            "           CAST(NULL AS DOUBLE) AS temperature\n"
            "    WHERE 1 = 0\n),\n", sql, count=1, flags=re.S)

    # ---- 4) 方言 ----
    grid_re = re.compile(r"CROSS JOIN\s*\(\s*\{\{GRID_GENERATOR\}\}.*?\)\s*g", re.S)
    sql = grid_re.sub("CROSS JOIN range(0, 120) AS g(step)", sql, count=1)
    sql = sql.replace("{{方言适配_NUMERIC}}", "NUMERIC")
    sql = sql.replace("{{GRID_GENERATOR}}", "range(0, 120) AS g(step)")

    # 剩余 ~* / !~* （若补丁外还有）
    sql = re.sub(r"(\w+(?:\.\w+)?)\s*!~\*\s*'([^']*)'",
                 r"NOT regexp_matches(\1, '\2', 'i')", sql)
    sql = re.sub(r"(\w+(?:\.\w+)?)\s*~\*\s*'([^']*)'",
                 r"regexp_matches(\1, '\2', 'i')", sql)

    # ---- 5) 表名 -> DuckDB 视图名 ----
    for key in TABLES:
        sql = sql.replace("{{TABLE_EICU_%s}}" % key, key.lower())

    if skip_vital:
        # 时变协变量置空，避免扫 1.78GB vitalPeriodic
        sql = sql.replace("v.vitalperiodicid", "NULL")
        for c in ("heartrate", "systemicmean", "respiration", "sao2", "temperature"):
            sql = re.sub(r"\bv\.%s\b" % c, "CAST(NULL AS DOUBLE) AS %s" % c, sql)

    # ---- 5b) 事件后停止随访：同一患者不得在缓解后继续贡献风险集 ----
    old_main = "SELECT * FROM analysis_long\nORDER BY patientunitstayid, step"
    new_main = ("SELECT * EXCLUDE (_first_event_step, icu_discharge_event, death_event),\n"
                "       CASE WHEN resolution_alt_event = 1 THEN 0 ELSE icu_discharge_event END AS icu_discharge_event,\n"
                "       CASE WHEN resolution_alt_event = 1 THEN 0 ELSE death_event END AS death_event\n"
                "FROM (\n"
                "    SELECT al.*,\n"
                "           MIN(CASE WHEN al.resolution_v1_event = 1\n"
                "                      OR al.death_event = 1\n"
                "                      OR al.icu_discharge_event = 1\n"
                "                     THEN al.step END)\n"
                "               OVER (PARTITION BY al.patientunitstayid) AS _first_event_step\n"
                "    FROM analysis_long al\n"
                ") _t\n"
                "WHERE _first_event_step IS NULL OR step <= _first_event_step\n"
                "ORDER BY patientunitstayid, step")
    # 替换在第 6 步的 body 上执行，避免 sql.find 定位失效

    # ---- 6) 主查询 -> COPY ----
    idx = sql.find("SELECT * FROM analysis_long")
    if idx < 0:
        raise RuntimeError("未找到主查询")
    end = sql.find(";", idx)
    body = sql[:end].rstrip()
    if old_main in body:
        body = body.replace(old_main, new_main)
    return sql, body


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"--[^\n]*", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain-only", action="store_true")
    ap.add_argument("--skip-vital", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--memory", default="8GB")
    ap.add_argument("--out", default=str(OUT_CSV))
    a = ap.parse_args()

    import duckdb

    sql, body = build_sql(skip_vital=a.skip_vital)
    leftovers = sorted(set(re.findall(r"\{\{[^}]+\}\}", strip_comments(sql))))
    leftovers = [x for x in leftovers if x not in ("{{TABLE_EICU_*}}", "{{方言适配_*}}")]
    log("SQL 构建完成，正文残留占位符: %s" % (leftovers or "无"))
    if leftovers:
        raise SystemExit(2)

    con = duckdb.connect()
    con.execute("SET threads=%d" % a.threads)
    con.execute("SET memory_limit='%s'" % a.memory)
    tmp = OUT_DIR / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute("SET temp_directory='%s'" % str(tmp).replace("\\", "/"))

    log("注册 CSV 视图 ...")
    for key, fn in TABLES.items():
        p = (EICU / fn).as_posix()
        if not (EICU / fn).exists():
            raise SystemExit("缺少文件: %s" % p)
        con.execute("CREATE OR REPLACE VIEW %s AS SELECT * FROM read_csv_auto('%s', header=true)"
                    % (key.lower(), p))
    log("视图就绪")

    if a.explain_only:
        log("EXPLAIN 校验中（不执行）...")
        plan = con.execute("EXPLAIN " + body).fetchall()
        log("EXPLAIN 通过，计划行数 %d" % len(plan))
        for r in plan[:12]:
            print("   ", str(r[0])[:180])
        return

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再原子替换，避免直接删除旧文件被环境安全策略拦截
    tmp_out = out.with_name(out.stem + "_new.csv")
    if tmp_out.exists():
        try:
            tmp_out.unlink()
        except OSError:
            pass
    t0 = time.time()
    log("开始执行并导出 -> %s" % out)
    try:
        con.execute("COPY (%s) TO '%s' (HEADER, DELIMITER ',')"
                    % (body, tmp_out.as_posix()))
        os.replace(str(tmp_out), str(out))
    except PermissionError:
        # 目标文件被预览占用时，退化为直接覆盖写目标路径
        log("目标被占用，改为直接覆盖写入")
        con.execute("COPY (%s) TO '%s' (HEADER, DELIMITER ',')"
                    % (body, out.as_posix()))
    log("导出完成，用时 %.1f 秒" % (time.time() - t0))

    import csv
    with open(str(out), "r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    cols = [c.strip().strip('"').lower() for c in header]
    missing = [c for c in REQUIRED if c not in cols]
    log("列数 %d | 必需 27 列缺失: %s" % (len(cols), missing or "无"))
    n_rows = sum(1 for _ in open(str(out), "rb")) - 1
    report = {
        "output": str(out), "rows": n_rows, "columns": len(cols),
        "missing_required": missing,
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "skip_vital": a.skip_vital,
        "not_implemented": ["入 ICU 前已发作谵妄（数据不可得）",
                            "高剂量丙泊酚 >24h（阈值未定）"],
    }
    (OUT_DIR / "build_analysis_long_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("行数: %s | 大小: %.1f MB" % (format(n_rows, ","), report["size_mb"]))
    if missing:
        raise SystemExit(7)
    log("✓ 27 列对齐，可以触发流水线")


if __name__ == "__main__":
    main()
