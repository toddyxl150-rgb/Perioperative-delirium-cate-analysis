/* =============================================================================
   sql_dose_flag_patch.sql —— 方案 B 所需的 SQL 补丁片段（v1.0 / 2026-11-09）
   对应：PI 方案 C 裁决（主分析 A 用 4.0 mcg/ml 默认填补，敏感性 B 剔除估算记录）

   【用途】
   01_dex_dose.py 会为每条 DEX 输注记录标记 is_assumed_mlhr。
   但 02_dataset_and_iptw.py / 04_cate_models.py 的方案 B 子集校验报告
   需要该标记出现在【分析长表 analysis_long.csv】中。
   SQL数据提取骨架_v1.2.sql 已冻结，本文件【不修改它】，只给出需追加的片段。

   【用法】PI 在本地替换 12 个 {{TABLE_EICU_*}} 表名时，顺带把下面三处
   追加到 v1.2 骨架的对应 CTE 即可；追加后长表会多出两列：
       is_assumed_mlhr          0/1，方案 B 据此剔除
       dose_concentration_source 字符串，审计用

   方言提示：正则部分给出 PostgreSQL 写法；MySQL 8 用 REGEXP / NOT REGEXP，
   SQL Server 可用 LIKE 组合近似，须按本地库调整。
   ============================================================================= */


/* -----------------------------------------------------------------------------
   [1] dex_infusion CTE 的 SELECT 列表追加两列
   ----------------------------------------------------------------------------- */
/*
   , CASE
        WHEN LOWER(i.drugname) LIKE '%ml/hr%'
         AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml'
         AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml'
         AND i.drugname !~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml'
        THEN 1 ELSE 0
     END AS is_assumed_mlhr_record
   , CASE
        WHEN LOWER(i.drugname) NOT LIKE '%ml/hr%' THEN 'not_applicable'
        WHEN i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*ml'
          OR i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]*/[[:space:]]*[0-9]+[[:space:]]*ml'
          OR i.drugname ~* '[0-9]+[[:space:]]*(mcg|ug)[[:space:]]+in.*[0-9]+[[:space:]]*ml'
             THEN 'explicit_in_name'
        ELSE 'assumed_default'
     END AS dose_concentration_source_record

   说明：与 01_dex_dose.py 的 resolve_concentration() 口径一致 ——
   名称含 4 mcg/ml、200 mcg/50ml、400 Mcg in ... 100 ml 的算「显式浓度」；
   其余 ml/hr 记录按 4.0 mcg/ml 填补，标记为 assumed_default。
*/


/* -----------------------------------------------------------------------------
   [2] step_exposure CTE：把记录级标记聚合到 6h 步长
   ----------------------------------------------------------------------------- */
/*
   , MAX(d.is_assumed_mlhr_record) AS is_assumed_mlhr_step
   , CASE
        WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'explicit_in_name'
                      THEN 1 ELSE 0 END) = 1 THEN 'explicit_in_name'
        WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'assumed_default'
                      THEN 1 ELSE 0 END) = 1 THEN 'assumed_default'
        ELSE 'not_applicable'
     END AS dose_concentration_source_step

   注意：若一个步长内同时存在显式浓度与估算浓度记录，
   保守起见整步标记为 assumed_default（方案 B 会剔除该步长），
   避免把估算剂量混入敏感性分析。若 PI 希望按记录级而非步长级剔除，
   请改为在长表中保留记录级明细并调整 02 的剔除粒度。
*/


/* -----------------------------------------------------------------------------
   [3] analysis_long CTE 的 SELECT 列表追加两列（最终长表字段）
   ----------------------------------------------------------------------------- */
/*
   , s.is_assumed_mlhr_step            AS is_assumed_mlhr
   , s.dose_concentration_source_step  AS dose_concentration_source
*/


/* -----------------------------------------------------------------------------
   [4] 剂量换算本身（方案 A 默认 4.0）在 SQL 里的等价写法，供核对
   ----------------------------------------------------------------------------- */
/*
   dex_dose_6h_mean_mcg_kg_hr =
       AVG(
         CASE
           WHEN LOWER(i.drugname) LIKE '%mcg/kg/hr%' THEN CAST(i.drugrate AS NUMERIC)
           WHEN LOWER(i.drugname) LIKE '%mcg/kg/min%' THEN CAST(i.drugrate AS NUMERIC) * 60
           WHEN LOWER(i.drugname) LIKE '%mcg/min%'  THEN CAST(i.drugrate AS NUMERIC) * 60 / w
           WHEN LOWER(i.drugname) LIKE '%mcg/hr%'   THEN CAST(i.drugrate AS NUMERIC) / w
           WHEN LOWER(i.drugname) LIKE '%mg/hr%'    THEN CAST(i.drugrate AS NUMERIC) * 1000 / w
           WHEN LOWER(i.drugname) LIKE '%ml/hr%'    THEN CAST(i.drugrate AS NUMERIC) * 4.0 / w
           ELSE NULL
         END
       )
   其中 w = patient.admissionweight（SQL 骨架 v1.2 已含该列；
   eICU 实测 admissionweight 非空且 >0 的比例为 184,125/200,859 = 91.7%）。
   ml/hr 的 4.0 是【PI 裁决的外部假设】，不是数据实测值，须在 Methods 声明，
   并报告 2.0 / 4.0 / 8.0 mcg/ml 的敏感性分析。

   ！！注意：若本地未追加本补丁，长表不含 is_assumed_mlhr 列，
   02 / 04 的子集校验报告会如实标记 "blocked_missing_flag"，
   不会用近似值冒充已完成方案 B 剔除。
   ============================================================================= */
