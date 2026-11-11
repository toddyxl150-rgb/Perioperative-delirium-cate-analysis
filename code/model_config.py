# -*- coding: utf-8 -*-
"""
阶段 3 建模统一配置 v1.0
对应：研究规格基线_v1.2.md / SQL数据提取骨架_v1.2.sql
所有脚本从这里读取时间常数、路径、列名与模型参数，禁止在别处硬编码。

【时间单位核实记录 · 2026-11-09】
PI 裁决所有 *_offset 单位为分钟；本助手已用【本地原始发布者文档】独立核实：
  E:\\Medical_project\\Sepsis_01\\data\\eicu\\eicu-code-master.zip
    -> eicu-code-master/website/content/about/eicu.md
       "The time stamp of all events are stored as offsets from the time of
        ICU admission, in minutes. As a result, hospital admission time will
        in general be negative."
    -> website/content/eicutables/nurseCharting.md  (nursingChartOffset: minutes)
    -> website/content/eicutables/patient.md        (unitDischargeOffset: minutes)
    -> website/content/eicutables/infusionDrug.md   (infusionOffset: minutes)
未联网；来源为本地官方代码仓库压缩包内文档，路径可打开。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(r"E:\Medical_project\21.POD_Causal_ML")
MODEL_DIR = BASE_DIR / "modeling"
OUTPUT_DIR = MODEL_DIR / "outputs"
DATA_DIR = Path(r"E:\Medical_project\00.Data")
EICU_DIR = Path(r"E:\Medical_project\Sepsis_01\data\eicu")

# SQL 骨架 v1.2 产出的分析长表（由 PI 在本地跑完后指定路径）
LONG_TABLE_PATH = OUTPUT_DIR / "analysis_long.csv"
GFORMULA_INPUT = OUTPUT_DIR / "gformula_input.csv"

# ---------------------------------------------------------------- 时间常数（已核实为分钟）
GRID_STEP_MIN = 360       # 6 小时步长
GRACE_PERIOD_MIN = 720    # 12 小时宽限期 / 主要结局连续达标窗口

# 早期启动窗口（PI 2026-11-09 裁决：12h -> 24h，用于扩大 CATE 处理组）
EARLY_WINDOW_HOURS = 24
EARLY_WINDOW_MIN = EARLY_WINDOW_HOURS * 60      # 1440 分钟 = 4 个 6h 步长
EARLY_STEPS = EARLY_WINDOW_MIN // GRID_STEP_MIN  # step 0..3
FOLLOWUP_MAX_MIN = 43200  # 30 天随访上限
MAX_STEPS = FOLLOWUP_MAX_MIN // GRID_STEP_MIN      # 120
MIN_EXPLICIT_ASSESSMENTS = 2
# PI 2026-11-09 裁决：g-formula 的干预窗口由 12h 对齐为 24h（step 0..3）
GRACE_STEPS = EARLY_STEPS   # = 4

# ---------------------------------------------------------------- 列名（与 SQL v1.2 对齐）
ID_COL = "patientunitstayid"
STEP_COL = "step"
HOSPITAL_COL = "hospitalid"
TREAT_COL = "strategy_arm_at_step"
DEX_START_COL = "dex_first_start_offset"
DOSE_RAW_COL = "drugrate"
DOSE_NAME_COL = "drugname"
WEIGHT_COL = "admissionweight"       # patient 表；infusionDrug 的 patientweight 为备用
WEIGHT_ALT_COL = "patientweight"
OUTCOME_COL = "resolution_v1_event"
OUTCOME_ALT_COL = "resolution_alt_event"
DEATH_COL = "death_event"
DISCHARGE_COL = "icu_discharge_event"
NOT_ASSESSED_COL = "not_assessed_count"
DEX_DOSE_COL = "dex_dose_6h_mean_mcg_kg_hr"

BASELINE_COVARIATES = [
    "age", "gender", "ethnicity", "admissionweight",
    "apachescore", "acutephysiologyscore", "predictedicumortality",
    "unittype", "apacheadmissiondx", "teachingstatus", "region",
]
TIMEVARYING_COVARIATES = [
    "heartrate", "systemicmean", "respiration", "sao2", "temperature",
    "dex_dose_6h_mean_mcg_kg_hr",
]

# ---------------------------------------------------------------- 剂量换算与离散化（任务 C）
# 目标单位：mcg/kg/h（= mcg/kg/hr）
DOSE_TARGET_UNIT = "mcg/kg/h"
DOSE_BIN_EDGES = [round(i * 0.1, 1) for i in range(16)]   # 0.0 ... 1.5 -> 15 个区间
DOSE_BIN_LABELS = list(range(1, 16))
DOSE_OVER_RANGE_FLAG = "dose_over_range"   # >1.5 mcg/kg/h 归入第 15 档并置标记

# ---------------------------------------------------------------- 浓度方案 C（PI 2026-11-09 裁决）
# 主分析 A：无显式浓度的 ml/hr 记录按 4.0 mcg/ml 填补。
# 【来源声明】「Precedex 200mcg/50ml = 4 mcg/ml」由 PI 裁决给出，本助手【未联网核实】
# 药品说明书；本地 eICU 中名称含浓度的 14 条记录解析值均为 4.0 mcg/ml，与之一致。
# 敏感性分析 B：剔除所有「估算浓度」记录后重跑校验。
DEFAULT_DEX_CONCENTRATION_MCG_PER_ML = 4.0
SENSITIVITY_CONCENTRATIONS = [2.0, 4.0, 8.0]
DOSE_SOURCE_EXPLICIT = "explicit_in_name"
DOSE_SOURCE_ASSUMED = "assumed_default"
DOSE_SOURCE_NA = "not_applicable"
DOSE_SOURCE_UNRESOLVED = "unresolved"
DOSE_ASSUMED_FLAG = "is_assumed_mlhr"      # 方案 B 据此列剔除
DOSE_SOURCE_COL = "dose_concentration_source"

# ---------------------------------------------------------------- 模型参数
RANDOM_SEED = 20261109
N_SPLITS = 5                  # GroupKFold(n_splits=5, groups=hospitalid)
N_SIMUL = 5000               # g-formula 蒙特卡洛轨迹数（自检时自动降为 500）
N_BOOTSTRAP = 200            # RR/RD 置信区间 bootstrap 次数（自检时降为 30）
IPTW_CLIP = (0.05, 20.0)     # 稳定化权重截断
QINI_N_QUANTILES = 10
CARD_N_DECILES = 10

# CQL / OPE（规格 v1.2 第 5 节，Reward 由 PI 裁决，未与 Lee 2024 原文核对）
RL_GAMMA = 0.99
RL_REWARD_DELIRIUM = -1
RL_REWARD_TERMINAL_NO_DELIRIUM = 1
RL_REWARD_STEP = 0

# ---------------------------------------------------------------- 图表
FIG_DPI = 300
FIG_SIZE = (7.2, 4.8)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
