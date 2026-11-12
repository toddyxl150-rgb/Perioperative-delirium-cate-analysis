# 早期右美托咪定靶试验仿真 — 分析代码仓库

**状态**：本仓库已公开，提交 `a8e2613`（28 个文件）；论文中的代码可得性声明已写入此地址。
Zenodo 归档 DOI 待生成后补充。

> 白话：这里是论文全部数字所用的分析代码。底层患者级数据不随本仓库分发，
> 需按下方第四节的条件向 PhysioNet 申请；代码本身可自由获取。

仓库地址：https://github.com/toddyxl150-rgb/Perioperative-delirium-cate-analysis

---

## 一、目录结构

| 路径 | 内容 |
|---|---|
| `code/` | 18 个自定义脚本（Python / R / SQL），本文全部数字由图生成 |
| `results/` | 7 个聚合结果文件（JSON / CSV），用于核对论文中的数字 |
| `SPECIFICATION_v1.3.md` | 规格履历与冻结方案（相当于研究方案 / statistical analysis plan） |

**没有放进来、也不应公开的内容**：
- `analysis_long.csv`（350,260 行患者-步长派生表，87 MB）——受 eICU-CRD 数据使用协议约束，作者无权再分发；
- `baseline_rass.csv`（患者级 RASS 提取结果）——同上，已从本包移除；
- `db_config.json`（数据库连接配置）、三篇参考文献全文（版权）。

数据字典（41 个字段定义）在投稿材料 `06_Supplementary_Information.md` 的 Supplementary Data 1 中。

---

## 二、运行顺序（复现论文结果）

需先取得 eICU-CRD v2.0 的 31 个 `.csv.gz`（见第四节数据访问条件），并把脚本里的本机绝对路径
（如 `E:/Medical_project/Sepsis_01/data/eicu`）改为自己的数据目录。

```bash
# 1) 从压缩原始表直接生成患者-步长分析长表（无需 PostgreSQL）
python code/build_analysis_long_duckdb.py

# 2) 一键流水线：长表校验+IPTW → g-formula CIF/RR/RD → CATE → 五张图
python code/run_real_pipeline.py

# 3) ALT 结局（去掉 RASS 达标要求）
python code/03b_gformula_py.py --outcome alt

# 4) 预设亚组与 V1/ALT 对比森林图
python code/07_subgroups.py --outcome v1
python code/07_subgroups.py --outcome alt
python code/07_subgroups.py --compare

# 5) 按医院规模蛇形分层的 5 折跨院验证（最终采用的结果）
python code/09_final_cv.py

# 6) 可选：合成数据自检（仅验证代码能跑通）
python code/selfcheck.py
```

---

## 三、环境与依赖（版本号为本机实读）

- Python 3.13
- numpy 2.5.4 / pandas 3.1.1 / scikit-learn 1.9.1 / statsmodels 0.15.0
- econml 0.17.0 / matplotlib 3.11.2 / scipy 1.18.1
- DuckDB 2.0.0（直接读取 gzip CSV 构造长表）
- R 4.6.1（g-formula 交叉核对实现）
- 绘图与 CATE 模块另需 shap、xgboost

---

## 四、数据访问条件

底层数据来自 **eICU Collaborative Research Database v2.0**，经 PhysioNet 认证访问，
须完成 credentialed data use agreement（DUA）与相应研究培训。数据库分发方而非作者控制访问；
作者不得再分发患者级数据。**代码可公开，数据不可公开** —— 两者要分开写，这正是主稿
Data availability 与 Code availability 分成两段的原因。

---

## 五、必须如实标注的执行边界（审稿人会核对）

| 脚本 | 状态 |
|---|---|
| `build_analysis_long_duckdb.py`、`02`、`03b`、`07`、`09`、`run_real_pipeline.py` | 已在真实数据上实跑，产出论文中的数字 |
| `03_gformula.R` | 基础 R g-formula 实跑成功；R 包 `gfoRmula` / `grf` 本机未安装，高级路径仅为模板 |
| `04_cate_models.py` | CausalForestDML 与 R-learner 均实跑，但 Qini 为负（12h −32.8、24h −48.8），**异质性分析已放弃，不得作为结论** |
| `06_ope_cql.py` | 仅接口模板；`torch` 未安装，**CQL 训练与 FQE/WIS 未执行**，不得写成已使用的方法 |
| `extract_to_csv.py` | 本机无 PostgreSQL 实例，**未跑通**；保留给有数据库环境的复现者 |
| `selfcheck.py` | 合成数据自检，**其中任何数字都不得引用为研究结果** |

---

## 六、发布状态与待办

| # | 事项 | 状态 |
|---|---|---|
| 1 | 建 GitHub 仓库并推送 `code/`、`results/`、README、规格文件 | ✅ 已完成（提交 `a8e2613`） |
| 2 | Zenodo 归档并取持久 DOI | ⬜ 待完成；取得后在本文件与主稿补入 |
| 3 | 添加开源许可证 LICENSE（如 MIT） | ⬜ 待 PI 决定版权所有者 |
| 4 | 论文 Code availability 回填 | ✅ GitHub 地址已填；DOI 待补 |

## 七、可直接粘贴到 GitHub 仓库描述的英文

> Analysis code for a target trial emulation of early dexmedetomidine for ICU delirium and agitation
> using the eICU Collaborative Research Database. Includes DuckDB-based construction of a 6-hour
> patient-step table, a parametric g-formula with competing risks, inverse-probability weighting,
> prespecified subgroup analyses, and hospital-stratified cross-validation. Patient-level data are
> not distributed; access to the source database requires a credentialed PhysioNet data use agreement.

---

**代码已公开；底层患者级数据受 eICU-CRD 数据使用协议约束，不在本仓库分发。**
