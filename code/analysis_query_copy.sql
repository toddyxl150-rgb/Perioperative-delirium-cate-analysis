COPY (
/* =============================================================================
   SQL 数据提取骨架 v1.2 —— 主分析库 eICU-CRD，真实字段回填版
   项目：右美托咪定预防/治疗 ICU 谵妄与躁动的模拟目标试验与因果机器学习研究
   对应规格：研究规格基线_v1.2.md（已冻结）
   配套映射：eICU量表映射规则_v1.2.md
   前一版本：SQL数据提取骨架_v1.0.sql（全占位符）【保留不覆盖】

   【重要 · 使用前必读】
   1. 本文件【字段名全部为 eICU-CRD 实读核实字段】，可直接对照本地 CSV 使用；
      仅 {{TABLE_EICU_*}} 为「CSV 导入数据库后」的表名占位符，由 PI 填写。
   2. 本文件是【提取骨架】，不是可执行脚本：
      - generate_series(0, 119) AS g(step)、{{方言适配_*}} 需按目标库方言替换；
      - {{TABLE_TIME_UNIT_CONFIRMED}} 是时间单位守卫，确认 offset 单位前不得移除。
   3. 时间换算 6h=360、12h=720、30天=43200 均基于「*_offset 单位为分钟」的
      【推断】，未核实。单位若不同须全局替换。
   4. IPTW 权重【不在 SQL 中计算】：本骨架只产出策略归属、删失标记与权重模型
      所需的时变协变量，ipw_weight_at_step 留 NULL，由 Python 端以「该步长开始
      之前」的变量拟合后回填（见规格 v1.2 第 4 节）。
   5. 本文件不含任何真实数据结果、效应估计或已核实的外部文献结论。
   ============================================================================= */


/* -----------------------------------------------------------------------------
   0. 配置开关（v1.2 锁定）
   ----------------------------------------------------------------------------- */
/*
   STRATEGY              = 'A'       -- 12h 内未启动 DEX + 之后自然病程随访
   OUTCOME_DEFINITION    = 'V1'      -- CAM-ICU 阴性 AND RASS in {0,-1,-2}
                                       -- ALT = 仅 CAM-ICU 阴性（默认同时产出）
   GRACE_PERIOD_MIN      = 720       -- 12h（step 0/1）
   GRID_STEP_MIN         = 360       -- 6h
   FOLLOWUP_MAX_MIN      = 43200     -- 30 天 = 120 步
   MIN_ASSESS_IN_12H     = 2
   MISSING_IS_NEGATIVE   = FALSE     -- 'N/A' 必须编码 NOT_ASSESSED
   SCALE_FILTER          = 'CAM-ICU + RASS only'  -- ICDSC/SAS/MAAS 进敏感性分析
   RL_GAMMA              = 0.99
   RL_REWARD_DELIRIUM    = -1        -- 步长内 Delirium Score='Yes'
   RL_REWARD_TERMINAL_NO_DELIRIUM = +1  -- 终点（ICU 出院或第 120 步）无谵妄
   RL_REWARD_STEP        = 0
*/


/* -----------------------------------------------------------------------------
   1. 表名占位符字典（字段名不再占位）
   ----------------------------------------------------------------------------- */
/*
   public.patient              patient.csv.gz
   public.nursecharting        nurseCharting.csv.gz
   public.infusiondrug         infusionDrug.csv.gz
   public.medication           medication.csv.gz
   public.note                 note.csv.gz
   public.physicalexam         physicalExam.csv.gz
   public.diagnosis            diagnosis.csv.gz
   public.treatment            treatment.csv.gz
   public.vitalperiodic        vitalPeriodic.csv.gz
   public.vitalaperiodic       vitalAperiodic.csv.gz
   public.apachepatientresult  apachePatientResult.csv.gz
   public.hospital             hospital.csv.gz
   generate_series(0, 119) AS g(step)                   生成 0..119 步的方言写法
   {{方言适配_DATEADD}}                时间加函数
   {{TABLE_TIME_UNIT_CONFIRMED}}        时间单位确认守卫（TODO）
*/


/* =============================================================================
   2. 基础队列与入排标准（patient 表，字段已核实）
   ============================================================================= */
WITH cohort_base AS (
    SELECT
        patientunitstayid,
        patienthealthsystemstayid,
        uniquepid,
        hospitalid,
        gender,
        age,
        ethnicity,
        apacheadmissiondx,
        admissionweight,
        hospitaladmitoffset,
        unitdischargeoffset,
        unitdischargestatus,
        hospitaldischargeoffset,
        hospitaldischargestatus,
        unittype,
        unitvisitnumber,
        unitstaytype
    FROM public.patient
),

eligible AS (
    SELECT c.*,
           (c.unitdischargeoffset - 0) AS icu_los_offset   -- 基准为入 ICU（offset=0）
    FROM cohort_base c
    WHERE c.age >= 18
      AND c.unitdischargeoffset >= 1440        -- ICU 住院 >=24h（单位：分钟，待核实）
      AND c.unitdischargeoffset <= 43200       -- <=30 天；超出者由 120 步行政截断
      -- TODO 排除 ECMO：
      --   方案 a) nurseCharting 中 cat='Ventricular Assist Device' AND vallabel='ECMO'
      --   方案 b) physicalExam / treatment 中相关路径
      --   两方案均【待核实】，不得静默删除该排除条件
      -- TODO 排除「入 ICU 前已发作谵妄」：note.notepath 含 'hx delirium tremens'
      --   仅为近似，覆盖不全，须在协议表声明
      -- TODO 排除「初始长期深镇静」：RASS -4/-5 或高剂量丙泊酚 >24h
      --   「高剂量」阈值仍未由 PI 给出，不得猜测数值
      -- TODO 排除「Time Zero 前已启动 DEX」：在 t0 算出后于 analysis_long 过滤
),


/* =============================================================================
   3. 量表事件流归一（nurseCharting，映射见 eICU量表映射规则_v1.2.md）
   ============================================================================= */
chart_events AS (
    SELECT
        nc.patientunitstayid,
        nc.nursingchartoffset                    AS t_offset,
        nc.nursingchartentryoffset               AS t_entry_offset,
        nc.nursingchartcelltypecat               AS cat,
        nc.nursingchartcelltypevallabel          AS vallabel,
        nc.nursingchartcelltypevalname           AS valname,
        nc.nursingchartvalue                     AS val,
        CASE
            WHEN nc.nursingchartcelltypevalname = 'Delirium Scale'
                 AND nc.nursingchartvalue = 'CAM-ICU'        THEN 'SCALE_CAMICU'
            WHEN nc.nursingchartcelltypevalname = 'Delirium Scale'
                 AND nc.nursingchartvalue = 'ICDSC'          THEN 'SCALE_ICDSC'
            WHEN nc.nursingchartcelltypevalname = 'Delirium Score'
                 AND nc.nursingchartvalue = 'No'             THEN 'CAM_NEGATIVE'
            WHEN nc.nursingchartcelltypevalname = 'Delirium Score'
                 AND nc.nursingchartvalue = 'Yes'            THEN 'CAM_POSITIVE'
            WHEN nc.nursingchartcelltypevalname = 'Delirium Score'
                 AND nc.nursingchartvalue = 'N/A'            THEN 'NOT_ASSESSED'
            WHEN nc.nursingchartcelltypevallabel = 'Symptoms of Delirium Present'
                 AND nc.nursingchartvalue = 'Yes'            THEN 'SYMPTOM_YES'
            WHEN nc.nursingchartcelltypevallabel = 'Symptoms of Delirium Present'
                 AND nc.nursingchartvalue = 'No'             THEN 'SYMPTOM_NO'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Scale'
                 AND nc.nursingchartvalue = 'RASS'           THEN 'SCALE_RASS'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Scale'
                 AND nc.nursingchartvalue IN ('SAS','MAAS')  THEN 'SCALE_OTHER_SED'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'
            WHEN nc.nursingchartcelltypevalname = 'Sedation Score'
                 AND nc.nursingchartvalue IN ('-3','-4','-5','1','2','3','4')
                                                              THEN 'RASS_OTHER'
            WHEN nc.nursingchartcelltypevallabel IN ('RASS','SEDATION SCORE')
                 AND nc.nursingchartcelltypevalname = 'Value'
                 AND nc.nursingchartvalue IN ('0','-1','-2') THEN 'RASS_TARGET'
            ELSE NULL
        END AS event_code
    FROM public.nursecharting nc
    -- TODO 守卫：确认 *_offset 单位为「分钟」后才可移除本守卫并启用查询
    WHERE 1 = 1
      -- 注意：nursingchartvalue 是字符串列，不可套用数值区间判定（如 BETWEEN 写法）
),


/* =============================================================================
   4. Time Zero：谵妄 / 躁动首次结构化发作
   ============================================================================= */
t0 AS (
    SELECT
        ce.patientunitstayid,
        MIN(ce.t_offset) AS t0_offset
    FROM chart_events ce
    JOIN eligible e ON e.patientunitstayid = ce.patientunitstayid
    WHERE ce.event_code IN ('CAM_POSITIVE', 'SYMPTOM_YES')
       -- 躁动档（RASS 正值）如需纳入，改 event_code 映射后加入此列表
    GROUP BY ce.patientunitstayid
    -- TODO 敏感性 A6：t0_offset ±360（6h）稳健性检验
),


/* =============================================================================
   5. 6h 网格：120 步，step 0/1 为 12h 宽限期；行政截断
   ============================================================================= */
grid AS (
    SELECT
        t.patientunitstayid,
        t.t0_offset,
        g.step,
        g.step * 360                         AS offset_from_t0,
        (g.step + 1) * 360                   AS offset_to_t0,
        CASE WHEN g.step < 2 THEN 1 ELSE 0 END AS in_grace_period,
        t.t0_offset + g.step * 360           AS step_start_offset,
        t.t0_offset + (g.step + 1) * 360     AS step_end_offset,
        e.unitdischargeoffset                AS admin_censor_offset,
        CASE WHEN e.unitdischargestatus = 'Expired'
                  OR e.hospitaldischargestatus = 'Expired'
             THEN 1 ELSE 0 END               AS death_competing_flag
    FROM t0 t
    JOIN eligible e ON e.patientunitstayid = t.patientunitstayid
    CROSS JOIN generate_series(0, 119) AS g(step)
    WHERE g.step < 120
      AND t.t0_offset + g.step * 360 <= e.unitdischargeoffset   -- 随访止于 ICU 出院
      AND t.t0_offset + g.step * 360 <= t.t0_offset + 43200     -- 30 天行政截断
),


/* =============================================================================
   6. DEX 暴露（时间依赖）
   ============================================================================= */
dex_infusion AS (
    SELECT
        i.patientunitstayid,
        i.infusionoffset             AS dex_start_offset,
        i.drugname,
        i.drugrate,
        i.infusionrate,
        i.drugamount,
        i.patientweight,
        -- 方案 C（PI 2026-11-09）：ml/hr 无显式浓度时按 4.0 mcg/ml【外部假设】填补
        -- 体重优先 infusionDrug.patientweight，缺失时回退 patient.admissionweight
        CASE
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/hr%' THEN CAST(i.drugrate AS NUMERIC)
            WHEN LOWER(i.drugname) LIKE '%mcg/kg/min%' THEN CAST(i.drugrate AS NUMERIC) * 60
            WHEN LOWER(i.drugname) LIKE '%mcg/min%'
                 THEN CAST(i.drugrate AS NUMERIC) * 60 / NULLIF(COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), CAST(p.admissionweight AS NUMERIC)), 0)
            WHEN LOWER(i.drugname) LIKE '%mcg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) / NULLIF(COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), CAST(p.admissionweight AS NUMERIC)), 0)
            WHEN LOWER(i.drugname) LIKE '%mg/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 1000 / NULLIF(COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), CAST(p.admissionweight AS NUMERIC)), 0)
            WHEN LOWER(i.drugname) LIKE '%ml/hr%'
                 THEN CAST(i.drugrate AS NUMERIC) * 4.0 / NULLIF(COALESCE(NULLIF(CAST(i.patientweight AS NUMERIC), 0), CAST(p.admissionweight AS NUMERIC)), 0)
            ELSE NULL
        END AS dex_dose_mcg_kg_hr,
        1 AS explicit_dex_record,
        CASE
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
        END AS dose_concentration_source_record
    FROM public.infusiondrug i
    LEFT JOIN public.patient p ON p.patientunitstayid = i.patientunitstayid
    WHERE LOWER(i.drugname) LIKE '%dexmed%'
       OR LOWER(i.drugname) LIKE '%dexmet%'     -- 覆盖库内拼写变体 Dexmetetomidine
       OR LOWER(i.drugname) LIKE '%precedex%'
),

dex_medication AS (
    SELECT
        m.patientunitstayid,
        m.drugstartoffset  AS dex_start_offset,
        m.drugstopoffset   AS dex_stop_offset,
        m.drugname,
        m.dosage
    FROM public.medication m
    WHERE LOWER(m.drugname) LIKE '%dexmed%'
       OR LOWER(m.drugname) LIKE '%dexmet%'
       OR LOWER(m.drugname) LIKE '%precedex%'
),


/* =============================================================================
   7. 逐步时变策略归属（Caveat 2）+ IPTW 三列
   ============================================================================= */
step_exposure AS (
    SELECT
        g.*,
        MIN(d.dex_start_offset) AS dex_first_start_offset,
        CASE WHEN MIN(d.dex_start_offset) IS NOT NULL
                  AND MIN(d.dex_start_offset) <= g.step_end_offset
             THEN 1 ELSE 0 END AS dex_initiated_by_step_end,
        CASE WHEN MIN(d.dex_start_offset) <= g.t0_offset + 720
             THEN 1 ELSE 0 END AS early_initiation_flag,
        -- v1.2 裁决：未启动 DEX 前保持对照臂；启动 DEX 时点即划入 DEX 臂
        CASE WHEN MIN(d.dex_start_offset) IS NOT NULL
                  AND MIN(d.dex_start_offset) <= g.step_start_offset
             THEN 1 ELSE 0 END AS strategy_arm_at_step,   -- 0=对照臂 1=DEX 臂
        -- 偏离策略 / 终止的删失标记（具体规则待 TTE 协议表）
        CAST(NULL AS NUMERIC) AS censor_at_step,
        -- IPTW 权重【由 Python 端拟合后回填】，SQL 不计算
        CAST(NULL AS NUMERIC) AS ipw_weight_at_step,
        AVG(d.dex_dose_mcg_kg_hr) AS dex_dose_6h_mean_mcg_kg_hr,
        MAX(d.is_assumed_mlhr_record) AS is_assumed_mlhr_step,
        CASE
            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'assumed_default'
                          THEN 1 ELSE 0 END) = 1 THEN 'assumed_default'
            WHEN MAX(CASE WHEN d.dose_concentration_source_record = 'explicit_in_name'
                          THEN 1 ELSE 0 END) = 1 THEN 'explicit_in_name'
            ELSE 'not_applicable'
        END AS dose_concentration_source_step
    FROM grid g
    LEFT JOIN dex_infusion d
      ON d.patientunitstayid = g.patientunitstayid
     AND d.dex_start_offset <= g.step_end_offset
    GROUP BY g.patientunitstayid, g.t0_offset, g.step, g.offset_from_t0,
             g.offset_to_t0, g.in_grace_period, g.step_start_offset,
             g.step_end_offset, g.admin_censor_offset, g.death_competing_flag
),


/* =============================================================================
   8. 主要结局：连续 ≥12h 达标 + ≥2 次明确记录 + N/A 单独编码
      （窗口函数实现见 eICU量表映射规则_v1.2.md 第 4 节；此处为逻辑拼装）
   ============================================================================= */
resolution_v1 AS (
    SELECT patientunitstayid, MIN(res_offset) AS resolution_offset
    FROM (
        SELECT
            a.patientunitstayid,
            a.t_offset AS res_offset,
            SUM(CASE WHEN b.event_code = 'CAM_NEGATIVE'  THEN 1 ELSE 0 END) AS cam_neg,
            SUM(CASE WHEN b.event_code = 'RASS_TARGET'   THEN 1 ELSE 0 END) AS rass_ok,
            SUM(CASE WHEN b.event_code = 'NOT_ASSESSED'  THEN 1 ELSE 0 END) AS not_assessed,
            COUNT(b.event_code)                                            AS explicit_cnt
        FROM chart_events a
        JOIN chart_events b
          ON b.patientunitstayid = a.patientunitstayid
         AND b.t_offset >= a.t_offset
         AND b.t_offset <  a.t_offset + 720
        WHERE a.event_code = 'CAM_NEGATIVE'
          AND b.event_code IS NOT NULL
        GROUP BY a.patientunitstayid, a.t_offset
    ) w
    WHERE cam_neg >= 1
      AND rass_ok >= 1
      AND explicit_cnt >= 2
      AND not_assessed = 0
    GROUP BY patientunitstayid
    -- TODO 实现优化：优先用 RANGE BETWEEN 窗口函数或按 6h 步长聚合，
    --      避免自连接放大行数（当前为逻辑示意）
),

resolution_alt AS (   -- 预设敏感性：仅 CAM-ICU 阴性，不限 RASS（默认同时产出）
    SELECT patientunitstayid, MIN(res_offset) AS resolution_offset
    FROM (
        SELECT a.patientunitstayid, a.t_offset AS res_offset,
               SUM(CASE WHEN b.event_code = 'NOT_ASSESSED' THEN 1 ELSE 0 END) AS not_assessed,
               COUNT(b.event_code) AS explicit_cnt
        FROM chart_events a
        JOIN chart_events b
          ON b.patientunitstayid = a.patientunitstayid
         AND b.t_offset >= a.t_offset AND b.t_offset < a.t_offset + 720
        WHERE a.event_code = 'CAM_NEGATIVE' AND b.event_code IS NOT NULL
        GROUP BY a.patientunitstayid, a.t_offset
    ) w
    WHERE explicit_cnt >= 2 AND not_assessed = 0
    GROUP BY patientunitstayid
),


/* =============================================================================
   9. 竞争事件（Q2 裁决：ICU 出院 / ICU 内死亡）
   ============================================================================= */
competing AS (
    SELECT
        e.patientunitstayid,
        e.unitdischargeoffset      AS icu_discharge_event_offset,
        e.unitdischargestatus,
        e.hospitaldischargestatus,
        CASE WHEN e.unitdischargestatus = 'Expired'
                  OR e.hospitaldischargestatus = 'Expired'
             THEN 1 ELSE 0 END    AS death_event_flag
        -- 注意：二者为 competing events，不是普通删失；
        --       主分析用 Parametric G-formula 产出反事实 CIF，禁止 1-KM
    FROM eligible e
),


/* =============================================================================
   10. 协变量：基线 + 时变（g-formula / IPTW / Causal Forest 共用）
   ============================================================================= */
cov_baseline AS (
    SELECT
        a.patientunitstayid,
        a.apachescore,
        a.acutephysiologyscore,
        a.predictedicumortality,
        a.actualiculos,
        a.actualventdays
    FROM public.apachepatientresult a
),

cov_timevarying AS (
    SELECT
        v.patientunitstayid,
        v.observationoffset,
        v.heartrate,
        v.systemicmean,
        v.systemicsystolic,
        v.respiration,
        v.sao2,
        v.temperature
    FROM public.vitalperiodic v
    -- TODO: 时变协变量必须取【该步长开始之前】的值，
    --       不得混入步长之后的干预后变量（g-formula 时序正确性关键）
),


/* =============================================================================
   11. 分析长表：一人一步一行，供 Parametric G-formula 消费
   ============================================================================= */
analysis_long AS (
    SELECT
        s.patientunitstayid,
        s.step,
        s.t0_offset,
        s.offset_from_t0,
        s.offset_to_t0,
        s.in_grace_period,
        s.step_start_offset,
        s.step_end_offset,
        s.admin_censor_offset,
        s.death_competing_flag,

        -- 暴露与时变策略
        s.dex_first_start_offset,
        s.dex_initiated_by_step_end,
        s.early_initiation_flag,
        s.strategy_arm_at_step,
        s.censor_at_step,
        s.ipw_weight_at_step,        -- TODO Python 端回填
        s.dex_dose_6h_mean_mcg_kg_hr,
        s.is_assumed_mlhr_step AS is_assumed_mlhr,
        s.dose_concentration_source_step AS dose_concentration_source,

        -- 结局（V1 主 / ALT 敏感性，默认同时产出）
        CASE WHEN r1.resolution_offset >= s.step_start_offset
                  AND r1.resolution_offset < s.step_end_offset
             THEN 1 ELSE 0 END AS resolution_v1_event,
        CASE WHEN r2.resolution_offset >= s.step_start_offset
                  AND r2.resolution_offset < s.step_end_offset
             THEN 1 ELSE 0 END AS resolution_alt_event,
        r1.resolution_offset    AS resolution_v1_offset,

        -- 竞争事件
        CASE WHEN c.death_event_flag = 1
                  AND c.icu_discharge_event_offset >= s.step_start_offset
                  AND c.icu_discharge_event_offset <  s.step_end_offset
             THEN 1 ELSE 0 END AS death_event,
        CASE WHEN c.death_event_flag = 0
                  AND c.icu_discharge_event_offset >= s.step_start_offset
                  AND c.icu_discharge_event_offset <  s.step_end_offset
             THEN 1 ELSE 0 END AS icu_discharge_event,

        -- 协变量
        b.apachescore, b.acutephysiologyscore, b.predictedicumortality,
        v.heartrate, v.systemicmean, v.respiration, v.sao2, v.temperature,
        e.gender, e.age, e.ethnicity, e.admissionweight,
        e.hospitalid, e.unittype, e.apacheadmissiondx,
        h.teachingstatus,
        h.region
    FROM step_exposure s
    JOIN eligible e   ON e.patientunitstayid = s.patientunitstayid
    LEFT JOIN resolution_v1  r1 ON r1.patientunitstayid = s.patientunitstayid
    LEFT JOIN resolution_alt r2 ON r2.patientunitstayid = s.patientunitstayid
    LEFT JOIN competing c       ON c.patientunitstayid = s.patientunitstayid
    LEFT JOIN cov_baseline b    ON b.patientunitstayid = s.patientunitstayid
    LEFT JOIN cov_timevarying v ON v.patientunitstayid = s.patientunitstayid
                               AND v.observationoffset < s.step_start_offset
    LEFT JOIN public.hospital h ON h.hospitalid = e.hospitalid
    WHERE NOT EXISTS (                       -- 排除 Time Zero 前已启动 DEX
        SELECT 1 FROM dex_infusion d
        WHERE d.patientunitstayid = s.patientunitstayid
          AND d.dex_start_offset < s.t0_offset
    )
)

SELECT * FROM analysis_long
ORDER BY patientunitstayid, step
) TO STDOUT WITH CSV HEADER