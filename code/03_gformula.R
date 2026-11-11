#!/usr/bin/env Rscript
# =============================================================================
# 任务 A（R 路径）：参数化 G-formula / gfoRmula
# 对应：研究规格基线_v1.2.md
#
# 本文件提供两条路径：
#   [1] gfoRmula 包模板 —— 本地 R 4.6.1【未安装 gfoRmula】，且不同版本 API
#       参数名有差异，故此处为【待核对模板】，不声称已跑通。
#   [2] 基础 R 离散时间 g-formula —— 只用 base/stats，【可实跑自检】：
#       三个 cause-specific hazard (resolution / death / discharge) 的 glm，
#       蒙特卡洛模拟两策略反事实 CIF、RR、RD。
#
# 运行：
#   Rscript 03_gformula.R --selfcheck
#   （正式运行前请先 install.packages("gfoRmula") 并核对 [1] 的 API）
#
# 所有 *_offset 单位已核实为分钟（本地 eicu-code-master.zip 官方文档）。
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
SELFCHECK <- "--selfcheck" %in% args

MAX_STEPS     <- 120L
GRACE_STEPS   <- 2L
GRID_STEP_MIN <- 360L

# -----------------------------------------------------------------------------
# [1] gfoRmula 包模板（需安装后按所装版本核对参数名）
# -----------------------------------------------------------------------------
run_gformula_package <- function(obs_data, out_csv = "gformula_r_output.csv") {
  if (!requireNamespace("gfoRmula", quietly = TRUE)) {
    cat("[1] gfoRmula NOT INSTALLED -> template only, NOT executed.\n")
    cat("    install.packages('gfoRmula') and verify argument names with ?gformula_surv\n")
    return(invisible(NULL))
  }
  # 下列为典型参数结构【模板】，须按所安装版本的帮助文档核对：
  #   id / time / outcome / competing / censor / covnames / basecovs /
  #   histories / hazard_models / intvars / interventions / n_simul
  # 竞争事件：ICU 出院与死亡；干预：12h 内启动 DEX vs 12h 内不启动。
  cat("[1] gfoRmula available. Fill in hazard_models and interventions below.\n")
  # res <- gfoRmula::gformula_surv(
  #   obs_data      = obs_data,
  #   id            = "patientunitstayid",
  #   time          = "step",
  #   outcome       = "resolution_v1_event",
  #   competing     = c("death_event", "icu_discharge_event"),
  #   covnames      = c("age", "apache", "heartrate", "strategy_arm_at_step"),
  #   basecovs      = c("age", "gender", "apache"),
  #   intvars       = "strategy_arm_at_step",
  #   interventions = list(function(data) data.frame(...), function(data) ...),
  #   n_simul       = 5000,
  #   seed          = 20261109
  # )
  # write.csv(res, out_csv)
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# [2] 基础 R 离散时间 g-formula（可实跑）
# -----------------------------------------------------------------------------
make_synthetic <- function(n = 300, n_steps = 60, seed = 1) {
  set.seed(seed)
  rows <- list(); k <- 1
  for (p in seq_len(n)) {
    age <- runif(1, 40, 80); apache <- runif(1, 30, 90)
    trt <- rbinom(1, 1, 0.40)
    ev_step <- sample.int(n_steps, 1)
    ev_type <- sample(0:3, 1, prob = c(0.50, 0.34 + 0.08 * trt, 0.08, 0.08))
    for (s in 0:(n_steps - 1)) {
      hit <- as.integer(s == ev_step)
      rows[[k]] <- data.frame(
        patientunitstayid = p, step = s, hospitalid = p %% 20,
        treat = trt, age = age, apache = apache,
        resolution = as.integer(hit && ev_type == 1),
        death      = as.integer(hit && ev_type == 2),
        discharge  = as.integer(hit && ev_type == 3)
      )
      k <- k + 1
    }
  }
  do.call(rbind, rows)
}

fit_hazards <- function(dat) {
  f <- resolution ~ treat + step + age + apache
  list(
    res  = glm(f,                              family = binomial(), data = dat),
    dth  = glm(death ~ treat + step + age,      family = binomial(), data = dat),
    disc = glm(discharge ~ treat + step + age,  family = binomial(), data = dat)
  )
}

simulate_cif <- function(models, baselines, intervention,
                         n_simul = 300, n_steps = MAX_STEPS, seed = 1) {
  set.seed(seed)
  idx <- sample.int(nrow(baselines), n_simul, replace = TRUE)
  b <- baselines[idx, , drop = FALSE]
  alive <- rep(TRUE, n_simul)
  cum <- numeric(n_steps + 1)
  for (s in 0:(n_steps - 1)) {
    live <- which(alive)
    if (!length(live)) { cum[(s + 2):(n_steps + 1)] <- cum[s + 1]; break }
    a <- if (s < GRACE_STEPS) {
      rep(if (intervention == "early_dex") 1L else 0L, length(live))
    } else {
      rbinom(length(live), 1, 0.40)   # 宽限期后 natural course（骨架）
    }
    nd <- data.frame(treat = a, step = s,
                     age = b$age[live], apache = b$apache[live])
    p_res  <- as.numeric(predict(models$res,  nd, type = "response"))
    p_dth  <- as.numeric(predict(models$dth,  nd, type = "response"))
    p_disc <- as.numeric(predict(models$disc, nd, type = "response"))
    p_tot  <- pmin(p_res + p_dth + p_disc, 0.95)
    scale  <- ifelse(p_tot > 0.95, 0.95 / p_tot, 1)
    p_res <- p_res * scale; p_dth <- p_dth * scale; p_disc <- p_disc * scale
    pm <- cbind(1 - p_res - p_dth - p_disc, p_res, p_dth, p_disc)
    pm <- pm / rowSums(pm)
    cs <- matrix(t(apply(pm, 1, cumsum)), nrow = length(live))
    u  <- runif(length(live))
    ev <- rowSums(u > cs[, 1:3, drop = FALSE])  # 0=none 1=res 2=death 3=discharge
    cum[s + 2] <- cum[s + 1] + sum(ev == 1) / n_simul
    alive[live[ev != 0]] <- FALSE              # 竞争事件后停止随访
  }
  cum
}

compare_strategies <- function(models, baselines, n_simul = 300, seed = 1) {
  c1 <- simulate_cif(models, baselines, "early_dex",    n_simul, seed = seed)
  c0 <- simulate_cif(models, baselines, "no_early_dex", n_simul, seed = seed + 1)
  r1 <- c1[length(c1)]; r0 <- c0[length(c0)]
  list(cif_early = c1, cif_control = c0,
       risk_early = r1, risk_control = r0,
       RR = if (r0 > 0) r1 / r0 else NA_real_, RD = r1 - r0)
}

# -----------------------------------------------------------------------------
if (SELFCHECK) {
  cat("=== [2] base-R parametric G-formula SELFCHECK (synthetic data) ===\n")
  dat <- make_synthetic(n = 300, n_steps = 60, seed = 20261109)
  cat(sprintf("synthetic rows: %d, patients: %d\n",
              nrow(dat), length(unique(dat$patientunitstayid))))
  models <- fit_hazards(dat)
  baselines <- unique(dat[, c("patientunitstayid", "age", "apache")])
  res <- compare_strategies(models, baselines, n_simul = 200, seed = 7)
  cat(sprintf("risk early DEX   : %.4f\n", res$risk_early))
  cat(sprintf("risk no early DEX: %.4f\n", res$risk_control))
  cat(sprintf("RR = %.4f | RD = %.4f\n", res$RR, res$RD))
  cat("competing events: ICU death + ICU discharge | estimator: g-formula (NOT 1-KM)\n")
  cat("=== [1] gfoRmula package check ===\n")
  run_gformula_package(dat)
  cat("SELFCHECK R: base-R g-formula executed; gfoRmula template only.\n")
} else {
  cat("Usage: Rscript 03_gformula.R --selfcheck\n")
}
