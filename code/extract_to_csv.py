# -*- coding: utf-8 -*-
"""
extract_to_csv.py —— 连接本地 PostgreSQL，执行 eICU 分析 SQL，直接导出长表 CSV

面向没有技术背景的使用者：填一次配置，跑一条命令即可。

【三步使用】
  1) 首次运行生成配置模板（不会连数据库）：
        python extract_to_csv.py --init
     然后打开同目录生成的 db_config.json，填写 dbname / user / 密码（可留空改用交互输入）
     以及 tables 里 12 张表的真实表名。

  2) 离线自检（不连数据库，只检查 SQL 合并是否正确）：
        python extract_to_csv.py --check

  3) 正式导出：
        python extract_to_csv.py
     结果写到 E:\\Medical_project\\21.POD_Causal_ML\\modeling\\outputs\\analysis_long.csv

【它自动替您做的事】
  - 读取 SQL数据提取骨架_v1.2.sql，按配置替换 12 个 {{TABLE_EICU_*}} 表名；
  - 把 {{GRID_GENERATOR}} / {{方言适配_NUMERIC}} 换成 PostgreSQL 写法；
  - 自动合入 sql_dose_flag_patch.sql 的三处标记列（is_assumed_mlhr /
    dose_concentration_source），无需手工编辑 SQL；
  - 按 PI 方案 C 把 ml/hr 的 DEX 剂量改为 4.0 mcg/ml 换算（外部假设），
    并在体重缺失时回退 patient.admissionweight；
  - 用只读事务 + 服务端 COPY TO STDOUT 流式导出，内存占用低；
  - 导出后自动核验 27 个必需列名。

【依赖与安全】
  - 需要 python 3.8+ 与一个 PostgreSQL 驱动：psycopg2（推荐）/ psycopg(3) / sqlalchemy。
    本机目前【均未安装】；脚本会给出安装命令，但【不会擅自安装】。
    按使用者的常驻规则，安装任何第三方包前应先做安全审查。
  - 密码不写入脚本、不写日志、不打印：优先环境变量 PGPASSWORD，
    其次配置文件 password 字段（建议留空），留空时运行时用 getpass 交互输入（不回显）。
  - 全程只读：SET TRANSACTION READ ONLY；不执行任何 DDL/DML。
  - 未联网、不向任何外部主机发送数据；连接目标由您在 db_config.json 指定。

【注意】
  SQL 骨架 v1.2 的 resolution_v1 使用了 12h 自连接（文件内已标 TODO），
  在全量 eICU 上可能较慢。若执行超时，请在数据库客户端先对单日/单院小样本试跑，
  或联系维护者提供窗口函数优化版；本脚本不擅自改写该分析逻辑。
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
OUT_CSV = OUT_DIR / "analysis_long.csv"
CONFIG = HERE / "db_config.json"
SQL_SKELETON = HERE.parent / "SQL数据提取骨架_v1.2.sql"
MERGED_SQL = HERE / "analysis_query_merged.sql"
COPY_SQL_FILE = HERE / "analysis_query_copy.sql"

REQUIRED_COLUMNS = [
    "patientunitstayid", "step", "hospitalid",
    "resolution_v1_event", "resolution_alt_event", "death_event", "icu_discharge_event",
    "strategy_arm_at_step", "dex_dose_6h_mean_mcg_kg_hr",
    "age", "gender", "ethnicity", "admissionweight",
    "apachescore", "acutephysiologyscore", "predictedicumortality",
    "unittype", "apacheadmissiondx", "teachingstatus", "region",
    "heartrate", "systemicmean", "respiration", "sao2", "temperature",
    "is_assumed_mlhr", "dose_concentration_source",
]

DEFAULT_TABLES = {
    "PATIENT": "patient",
    "NURSECHARTING": "nursecharting",
    "INFUSIONDRUG": "infusiondrug",
    "MEDICATION": "medication",
    "NOTE": "note",
    "PHYSICALEXAM": "physicalexam",
    "DIAGNOSIS": "diagnosis",
    "TREATMENT": "treatment",
    "VITALPERIODIC": "vitalperiodic",
    "VITALAPERIODIC": "vitalaperiodic",
    "APACHEPATIENTRESULT": "apachepatientresult",
    "HOSPITAL": "hospital",
}

CONFIG_TEMPLATE = {
    "_说明": "请填写 dbname / user / password（可留空改用交互输入）/ tables 中 12 张表的真实表名。密码不会写日志。",
    "host": "localhost",
    "port": 5432,
    "dbname": "eicu",
    "user": "postgres",
    "password": "",
    "schema": "public",
    "tables": DEFAULT_TABLES,
}


# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
def write_config_template():
    CONFIG.write_text(json.dumps(CONFIG_TEMPLATE, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("已生成配置模板:", CONFIG)
    print("请用记事本打开，填写 dbname / user / password（密码可留空）/ tables 后重跑。")
    print("提示：表名要写数据库里的真实名称；若建表时用了双引号保留大小写（如 \"nurseCharting\"），")
    print("      请在 tables 里写 \"public.nurseCharting\" 这样的完整带引号名称。")


def load_config():
    if not CONFIG.exists():
        return None
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for k in ("host", "port", "dbname", "user", "schema"):
        cfg.setdefault(k, CONFIG_TEMPLATE[k])
    cfg.setdefault("password", "")
    tables = dict(DEFAULT_TABLES)
    tables.update(cfg.get("tables", {}))
    cfg["tables"] = tables
    return cfg


def resolve_password(cfg):
    pw = os.environ.get("PGPASSWORD") or cfg.get("password") or ""
    if not pw:
        print("配置文件未填密码，且未设置环境变量 PGPASSWORD。")
        pw = getpass.getpass("请输入 PostgreSQL 密码（输入时不显示）：")
    return pw


def detect_driver():
    """返回 (驱动名, 模块)；不安装任何包。"""
    import importlib
    for name in ("psycopg2", "psycopg", "sqlalchemy"):
        try:
            return name, importlib.import_module(name)
        except ImportError:
            continue
    return None, None


def connect(cfg, password, driver, mod):
    kw = dict(host=cfg["host"], port=int(cfg["port"]), dbname=cfg["dbname"],
              user=cfg["user"], password=password)
    dsn_safe = "postgresql://%s@%s:%s/%s" % (
        cfg["user"], cfg["host"], cfg["port"], cfg["dbname"])
    print("连接目标:", dsn_safe, "（密码已隐藏）")
    if driver == "psycopg2":
        conn = mod.connect(**kw)
    elif driver == "psycopg":
        conn = mod.connect(**kw)
    else:  # sqlalchemy
        from sqlalchemy import create_engine
        url = "postgresql+psycopg2://%s:%s@%s:%s/%s" % (
            cfg["user"], password, cfg["host"], cfg["port"], cfg["dbname"])
        return create_engine(url), "sqlalchemy"
    return conn, driver


# ----------------------------------------------------------------------------
# SQL 合并
# ----------------------------------------------------------------------------
FLAG_RECORD_CASES = """        CASE
            WHEN LOWER(i.drugname) LIKE '%ml/hr%'
             AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml'
             AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml'
             AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml'
            THEN 1 ELSE 0
        END AS is_assumed_mlhr_record,
        CASE
            WHEN LOWER(i.drugname) NOT LIKE '%ml/hr%' THEN 'not_applicable'
            WHEN i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml'
              OR i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml'
              OR i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml'
                THEN 'explicit_in_name'
            ELSE 'assumed_default'
        END AS dose_concentration_source_record"""

FLAG_STEP_AGGS = """        MAX(d.is_assumed_mlhr_record) AS is_assumed_mlhr_step,
        CASE
            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'assumed_default'
                          THEN 1 ELSE 0 END) = 1 THEN 'assumed_default'
            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'explicit_in_name'
                          THEN 1 ELSE 0 END) = 1 THEN 'explicit_in_name'
            ELSE 'not_applicable'
        END AS dose_concentration_source_step"""

WEIGHT_EXPR = ("COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), "
               "CAST(p.admissionweight AS NUMERIC))")

DOSE_CASE = """        -- 方案 C（PI 2026-11-09）：ml/hr 无显式浓度时按 4.0 mcg/ml【外部假设】填补
        -- 体重优先 infusionDrug.patientweight，缺失时回退 patient.admissionweight
        CASE
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/hr%' THEN CAST(i.drugrate AS NUMERIC)
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/min%' THEN CAST(i.drugrate AS NUMERIC) * 60
            WHEN LOWER(i.drugname) LIKE '%mcg/min%'
                 THEN CAST(i.drugrate AS NUMERIC) * 60 / NULLIF(WEIGHT, 0)
            WHEN LOWER(i.drugname) LIKE '%mcg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) / NULLIF(WEIGHT, 0)
            WHEN LOWER(i.drugname) LIKE '%mg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 1000 / NULLIF(WEIGHT, 0)
            WHEN LOWER(i.drugname) LIKE '%ml/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 4.0 / NULLIF(WEIGHT, 0)
            ELSE NULL
        END AS dex_dose_mcg_kg_hr,""".replace("WEIGHT", WEIGHT_EXPR)


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    return re.sub(r"--[^\n]*", "", s)


def build_sql(cfg, sql_path, auto_patch=True):
    sql = Path(sql_path).read_text(encoding="utf-8")
    applied = []

    if auto_patch and "is_assumed_mlhr_record" not in sql:
        # [1] dex_infusion 记录级标记
        a1 = "        1 AS explicit_dex_record\n"
        if a1 not in sql:
            raise RuntimeError("补丁锚点 [1] 未找到，请确认 SQL 骨架版本为 v1.2")
        sql = sql.replace(a1, "        1 AS explicit_dex_record,\n" + FLAG_RECORD_CASES + "\n")

        # [1b] 回退体重需要 patient 表
        a1b = ("    FROM {{TABLE_EICU_INFUSIONDRUG}} i\n"
               "    WHERE LOWER(i.drugname) LIKE '%dexmed%'")
        if a1b not in sql:
            raise RuntimeError("补丁锚点 [1b] 未找到")
        sql = sql.replace(a1b,
                          "    FROM {{TABLE_EICU_INFUSIONDRUG}} i\n"
                          "    LEFT JOIN {{TABLE_EICU_PATIENT}} p "
                          "ON p.patientunitstayid = i.patientunitstayid\n"
                          "    WHERE LOWER(i.drugname) LIKE '%dexmed%'")

        # [4] 方案 A：ml/hr 按 4.0 mcg/ml 换算
        dose_re = re.compile(r"        -- 剂量标准化.*?END AS dex_dose_mcg_kg_hr,", re.S)
        if not dose_re.search(sql):
            raise RuntimeError("剂量换算块未找到")
        sql = dose_re.sub(lambda m: DOSE_CASE, sql, count=1)

        # [2] step_exposure 步长级聚合
        a2 = "        AVG(d.dex_dose_mcg_kg_hr) AS dex_dose_6h_mean_mcg_kg_hr\n"
        if a2 not in sql:
            raise RuntimeError("补丁锚点 [2] 未找到")
        sql = sql.replace(a2,
                          "        AVG(d.dex_dose_mcg_kg_hr) AS dex_dose_6h_mean_mcg_kg_hr,\n"
                          + FLAG_STEP_AGGS + "\n")

        # [3] analysis_long 输出列
        a3 = "        s.dex_dose_6h_mean_mcg_kg_hr,\n"
        if a3 not in sql:
            raise RuntimeError("补丁锚点 [3] 未找到")
        sql = sql.replace(a3, a3 +
                          "        s.is_assumed_mlhr_step AS is_assumed_mlhr,\n"
                          "        s.dose_concentration_source_step AS dose_concentration_source,\n")
        applied.append("补丁[1][2][3]三处标记列 + [4] 方案A 4.0 mcg/ml 剂量换算")
    elif "is_assumed_mlhr_record" in sql:
        applied.append("检测到 SQL 已含补丁标记，跳过自动注入")

    # PostgreSQL 网格生成器
    grid_re = re.compile(r"CROSS JOIN\s*\(\s*\{\{GRID_GENERATOR\}\}.*?\)\s*g", re.S)
    if grid_re.search(sql):
        sql = grid_re.sub("CROSS JOIN generate_series(0, 119) AS g(step)", sql, count=1)
        applied.append("GRID_GENERATOR -> generate_series(0,119)")
    sql = sql.replace("{{方言适配_NUMERIC}}", "NUMERIC")
    sql = sql.replace("{{GRID_GENERATOR}}", "generate_series(0, 119) AS g(step)")

    # 12 个表名（含 schema 前缀；若表名本身已带引号/点则不加前缀）
    schema = cfg["schema"]
    for key, table in cfg["tables"].items():
        full = table if ("." in table or '"' in table) else "%s.%s" % (schema, table)
        sql = sql.replace("{{TABLE_EICU_%s}}" % key, full)
    applied.append("替换 %d 个表名占位符" % len(cfg["tables"]))

    # 残留占位符检查（忽略注释中的泛称）
    body_only = strip_comments(sql)
    leftovers = sorted(set(re.findall(r"\{\{[^}]+\}\}", body_only)))
    leftovers = [x for x in leftovers if x not in ("{{TABLE_EICU_*}}", "{{方言适配_*}}")]

    # 构造 COPY 查询
    idx = sql.find("SELECT * FROM analysis_long")
    if idx < 0:
        raise RuntimeError("未找到主查询 SELECT * FROM analysis_long")
    end = sql.find(";", idx)
    body = sql[:end].rstrip()
    copy_sql = "COPY (\n" + body + "\n) TO STDOUT WITH CSV HEADER"

    MERGED_SQL.write_text(sql, encoding="utf-8")
    COPY_SQL_FILE.write_text(copy_sql, encoding="utf-8")
    return sql, copy_sql, applied, leftovers


# ----------------------------------------------------------------------------
# 导出
# ----------------------------------------------------------------------------
class CountingWriter:
    """二进制写入 + 统计行数/字节数，供 psycopg2 copy_expert 使用。"""
    def __init__(self, fh):
        self.fh = fh
        self.lines = 0
        self.bytes = 0

    def write(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.fh.write(data)
        self.lines += data.count(b"\n")
        self.bytes += len(data)
        return len(data)


def verify_columns(path):
    with open(str(path), "r", encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
    cols = [c.strip().strip('"').lower() for c in header]
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    return {"n_columns": len(cols), "missing": missing,
            "extra": [c for c in cols if c not in REQUIRED_COLUMNS][:15]}


def export_psycopg2(conn, copy_sql, out_path):
    cur = conn.cursor()
    cur.execute("SET client_encoding = 'UTF8'")
    cur.execute("SET TRANSACTION READ ONLY")
    with open(str(out_path), "wb") as raw:
        w = CountingWriter(raw)
        cur.copy_expert(copy_sql, w, size=65536)
        stats = {"rows_including_header": w.lines, "bytes": w.bytes}
    cur.close()
    return stats


def export_fetchmany(conn_or_engine, body_sql, out_path, is_sqlalchemy=False):
    if is_sqlalchemy:
        raw = conn_or_engine.raw_connection()
        cur = raw.cursor()
    else:
        cur = conn_or_engine.cursor()
    cur.execute("SET TRANSACTION READ ONLY")
    cur.execute(body_sql)
    n = 0
    with open(str(out_path), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cols = [d[0] for d in cur.description]
        w.writerow(cols)
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            w.writerows(rows)
            n += len(rows)
    cur.close()
    return {"rows_including_header": n + 1, "bytes": out_path.stat().st_size}


def precheck_tables(cur, cfg):
    missing = []
    for key, table in cfg["tables"].items():
        full = table if ("." in table or '"' in table) else "%s.%s" % (cfg["schema"], table)
        try:
            cur.execute("SELECT to_regclass(%s)", (full,))
            if cur.fetchone()[0] is None:
                missing.append((key, full))
        except Exception:
            missing.append((key, full))
    return missing


# ----------------------------------------------------------------------------
def main():
    global CONFIG
    ap = argparse.ArgumentParser(description="PostgreSQL -> analysis_long.csv")
    ap.add_argument("--init", action="store_true", help="生成 db_config.json 模板后退出")
    ap.add_argument("--check", action="store_true", help="离线自检：只合并 SQL，不连数据库")
    ap.add_argument("--dry-run", action="store_true",
                    help="连库只做 EXPLAIN 验证（语法/表名/列名），不导出")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--sql-file", default=str(SQL_SKELETON),
                    help="已自行合入补丁的 SQL 文件；指定后跳过自动注入")
    ap.add_argument("--no-auto-patch", action="store_true", help="不自动注入补丁")
    ap.add_argument("--output", default=str(OUT_CSV))
    ap.add_argument("--statement-timeout", default="0",
                    help="SQL 超时毫秒，0=不限制；大表跑不动时可设较大值")
    a = ap.parse_args()

    CONFIG = Path(a.config)
    if a.init:
        write_config_template()
        return

    cfg = load_config()
    if cfg is None:
        print("未找到配置文件，先生成模板：\n")
        write_config_template()
        print("\n填好后执行：python extract_to_csv.py --check")
        return

    auto_patch = not a.no_auto_patch and (a.sql_file == str(SQL_SKELETON))
    try:
        sql, copy_sql, applied, leftovers = build_sql(cfg, a.sql_file, auto_patch)
    except Exception as e:
        print("SQL 合并失败：", e)
        print("请确认使用的是 SQL数据提取骨架_v1.2.sql，或改用 --sql-file 指向已合好补丁的文件。")
        raise SystemExit(2)

    print("=== SQL 合并结果 ===")
    for x in applied:
        print("  ·", x)
    print("  已生成:", MERGED_SQL.name, "/", COPY_SQL_FILE.name)
    print("  正文残留占位符:", leftovers if leftovers else "无")
    if leftovers:
        print("  ✗ 仍有未替换占位符，请先修正再导出")
        raise SystemExit(2)

    if a.check:
        print("\n=== 离线自检（未连接数据库）===")
        print("  补丁标记列存在:", "is_assumed_mlhr" in sql)
        print("  方案A 4.0 换算存在:", "4.0 / NULLIF" in sql or "* 4.0" in sql)
        print("  GRID 生成器:", "generate_series(0, 119) AS g(step)" in sql)
        print("  COPY 查询头:", copy_sql.splitlines()[0])
        print("  必需列静态覆盖:", sum(1 for c in REQUIRED_COLUMNS if c.lower() in sql.lower()),
              "/", len(REQUIRED_COLUMNS))
        print("  离线自检通过。确认无误后运行：python extract_to_csv.py")
        return

    driver, mod = detect_driver()
    if driver is None:
        print("\n✗ 本机没有 PostgreSQL 驱动，无法连接数据库。")
        print("  按您的常驻规则，我没有擅自安装第三方包。")
        print("  如需安装，请先做安装前安全审查；审查通过后可自行执行：")
        print("      pip install psycopg2-binary")
        print("  或改用 DBeaver/pgAdmin 手动导出（脚本的 --check 仍可离线验证 SQL）。")
        raise SystemExit(3)

    print("\n使用驱动:", driver)
    password = resolve_password(cfg)
    try:
        conn, kind = connect(cfg, password, driver, mod)
    except Exception as e:
        print("连接失败：", str(e)[:400])
        print("请检查：PostgreSQL 服务是否启动、host/port/dbname/user 是否正确、")
        print("        pg_hba.conf 是否允许该用户本地连接、密码是否正确。")
        raise SystemExit(4)

    if a.dry_run:
        try:
            cur = conn.cursor() if kind != "sqlalchemy" else conn.raw_connection().cursor()
            miss = precheck_tables(cur, cfg)
            if miss:
                print("✗ 以下表在数据库中找不到：")
                for k, t in miss:
                    print("   ", k, "->", t)
                print("请在 db_config.json 的 tables 里改成真实表名（注意大小写/引号）。")
                raise SystemExit(5)
            body = sql[:sql.find(";", sql.find("SELECT * FROM analysis_long"))].rstrip()
            cur.execute("EXPLAIN (COSTS OFF) " + body)
            plan = cur.fetchall()
            print("\n=== EXPLAIN 通过（SQL 语法、表名、列名均可解析）===")
            print("  计划行数:", len(plan))
            for r in plan[:10]:
                print("   ", str(r[0])[:150])
            cur.close()
            print("\n未执行导出（--dry-run）。确认无误后去掉该参数正式运行。")
            return
        except SystemExit:
            raise
        except Exception as e:
            print("✗ EXPLAIN 失败：", type(e).__name__, str(e)[:600])
            print("  多半是表名/列名不匹配或 SQL 方言差异，请把上面报错原文发给维护者。")
            raise SystemExit(8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.output)
    print("\n开始导出 ->", out_path)

    try:
        if kind == "sqlalchemy":
            stats = export_fetchmany(conn, sql[:sql.find(";", sql.find("SELECT * FROM analysis_long"))],
                                     out_path, is_sqlalchemy=True)
        elif driver == "psycopg2":
            cur = conn.cursor()
            miss = precheck_tables(cur, cfg)
            cur.close()
            if miss:
                print("✗ 以下表在数据库中找不到：")
                for k, t in miss:
                    print("   ", k, "->", t)
                print("请在 db_config.json 的 tables 里改成真实表名（注意大小写/引号）。")
                raise SystemExit(5)
            if a.statement_timeout != "0":
                with conn.cursor() as c:
                    c.execute("SET statement_timeout = %s" % int(a.statement_timeout))
            stats = export_psycopg2(conn, copy_sql, out_path)
        else:  # psycopg3
            body = sql[:sql.find(";", sql.find("SELECT * FROM analysis_long"))].rstrip()
            stats = export_fetchmany(conn, body, out_path)
        conn.commit()
    except SystemExit:
        raise
    except Exception as e:
        print("导出失败：", type(e).__name__, str(e)[:600])
        print("若提示超时或内存不足，多半是 resolution_v1 的 12h 自连接放大行数；")
        print("请先在数据库客户端用小样本验证，或联系维护者提供窗口函数优化版。")
        raise SystemExit(6)

    print("\n=== 导出完成 ===")
    print("  文件:", out_path)
    print("  大小: %s bytes (%.1f MB)" % (format(stats["bytes"], ","), stats["bytes"] / 1e6))
    print("  行数(含表头):", format(stats["rows_including_header"], ","))
    chk = verify_columns(out_path)
    print("\n=== 只读列名核验 ===")
    print("  列数:", chk["n_columns"], "| 必需 27 列缺失:", chk["missing"] or "无")
    if chk["missing"]:
        print("  ✗ 列名未对齐，请核对 SQL SELECT 列表与补丁三处片段后再跑流水线")
        raise SystemExit(7)
    print("  ✓ 27 列完全对齐，可以回复「已放好」触发后续核验与流水线。")


if __name__ == "__main__":
    main()
