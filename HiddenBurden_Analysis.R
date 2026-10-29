#############################################################################
################# U.S. Cocci Incidence Estimates Analysis  ##################
#############################################################################

# This is the public script accompanying the manuscript:
# 'Coccidioidomycosis incidence across reporting and non-reporting U.S. states: an ensemble modeling study'
# In brief, it estimates monthly, county-level cocci incidence across the U.S.
# using an XGBoost model trained on AZ/CA tract-level surveillance data, then applied to all U.S. counties

################################################################################
########################### Set-up #############################################
################################################################################
#### 0. Load libraries and functions #####

library(xgboost) # written for xgboost 1.7.x
library(data.table)
library(mltools)
library(tidyverse)
library(sf)
library(tigris)
library(scales)
library(cowplot)
library(ggpattern)

cores <- 4
data.table::setDTthreads(threads = cores)

# Restricted inputs (surveillance, EHR, canine serology, NNDSS) are randomly shuffled in the public data files
#data_dir <- "data"
data_dir <- "~/Downloads/HiddenBurden_Public/data" # example

out_dir  <- "output"
pred_out <- file.path(out_dir, "Predictions")
dir.create(pred_out, recursive = TRUE, showWarnings = FALSE)

tract_file  <- file.path(data_dir, "PublicData_TractLevel_CAZ_2013on.csv.gz")
county_file <- file.path(data_dir, "PublicData_CountyLevel_2013on.csv.gz")
nndss_file  <- file.path(data_dir, "PublicData_NNDSS_2000_2022.csv.gz")

tweedie_deviance <- function(y, mu, p = 1.2) {y <- pmax(y, 0) ; mu <- pmax(mu, 1e-10)
if (p == 1 || p == 2) stop("p cannot be 1 or 2 for this formula")
dev <- 2 * ( (ifelse(y == 0, 0, y^(2 - p) / ((1 - p) * (2 - p)))) -
               (y * mu^(1 - p) / (1 - p)) + (mu^(2 - p) / (2 - p)) )
return(dev)}

#### 1. Set up training data ####

# Model specification: "Main", "AltSpec_NoSkinTest", or "AltSpec_NoVetEpi" (alt. specs used n_boot = 10)
spec   <- "Main"
#n_boot <- 100 
n_boot <- 10 # note this was 100 in the manuscript, but can be set to 10 here for speed

dataRaw <- fread(tract_file)
dataRaw$GEOID <- as.numeric(dataRaw$GEOID)
dataRaw <- dataRaw %>% filter(YEAR >= 2013) %>% select(-FIPS)
dataRaw <- dataRaw %>% filter(PopEstimate > 0)

# Keep only rows with non-missing enhanced surveillance data
data <- dataRaw %>% filter(!is.na(Enhanced_Inc))

# Missing canine values = no reported cases
data <- data %>% mutate(across(contains("Dog"), ~ ifelse(is.na(.) | is.nan(.), 0, .)))

# Drop tracts with very small populations (<100)
data <- data %>% filter(PopEstimate >= 100)
data <- data %>% select(-"cases")

# Retain a single (30cm) depth for each soil variable
data <- data %>% select(-c("SoilpH_30cm", "SoilpH_10cm", "SoilpH_0cm",
                           "SoilOrgC_0cm", "SoilOrgC_10cm", "SoilWaterPercent_0cm", "SoilWaterPercent_10cm",
                           "SoilTexture_0cm", "SoilTexture_10cm"))

# Coccidioidin reactivity: convert bins to numeric midpoints
data$SkinTest_PercentReactors_Numeric <- str_extract_all(data$SkinTest_PercentReactors, "\\d+") %>%  lapply(\(x) as.numeric(x)) %>%  sapply(\(x) mean(x))
data <- data %>% select(-"SkinTest_PercentReactors")

if (spec == "AltSpec_NoSkinTest") data <- data %>% select(-"SkinTest_PercentReactors_Numeric")
if (spec == "AltSpec_NoVetEpi") data <- data %>% select(-c("Monthly_Dog_Inc_lag_0", "Total_Pos_Dog_Tests_lag_0", "Total_Dog_Tests_lag_0", "EHRInc_wt_downscale"))

fwrite(data, file.path(out_dir, sprintf("FullTrainingAndTestingData_ForComparison_%s.csv", spec)))

#### 2. One-hot encode ####

data$MONTH <- as.factor(data$MONTH)
data$SoilTexture_30cm <- as.factor(data$SoilTexture_30cm)
data$modal_land <- as.factor(data$modal_land)

dataModel <- as.data.frame(one_hot(as.data.table(data)))

features <- dataModel %>% select(-c(Enhanced_Inc, GEOID, PopEstimate))
train_feat_names <- colnames(features)
write.csv(train_feat_names, file.path(out_dir, sprintf("TrainingFeatureNames_%s.csv", spec)))

################################################################################
############## Run model (b = 100), then combine iterations ####################
################################################################################
##### 3. Bootstrap xgboost models ####

unique_geoid <- unique(dataModel$GEOID)
best_params <- fread(file.path(data_dir, "BestParameters_Median_Updated.csv"))

# Down-weight coccidioidin reactivity during column sampling
feat_weights_full <- rep(1, length(train_feat_names))
names(feat_weights_full) <- train_feat_names
feat_weights_full["SkinTest_PercentReactors_Numeric"] <- 0.2

# Number of test rows used for SHAP per bootstrap
N_shap <- 5000

for (b in 1:n_boot) {
  set.seed(b)

  # Random 80/20 split by GEOID
  test_geoid_boot <- sample(unique_geoid, size = round(0.2 * length(unique_geoid)))
  train_df_boot   <- dataModel[!dataModel$GEOID %in% test_geoid_boot, , drop = FALSE]
  test_df_boot    <- dataModel[ dataModel$GEOID %in% test_geoid_boot, , drop = FALSE]

  # Target is incidence (Tweedie handles the log link)
  train_x_boot <- train_df_boot %>% select(-c(Enhanced_Inc, GEOID, PopEstimate)) %>% as.matrix()
  train_y_boot <- as.numeric(train_df_boot$Enhanced_Inc)

  test_x_boot  <- test_df_boot %>% select(-c(Enhanced_Inc, GEOID, PopEstimate)) %>% as.matrix()
  test_y_boot  <- as.numeric(test_df_boot$Enhanced_Inc)

  # Population size as sample weight
  w_train <- train_df_boot$PopEstimate
  w_test  <- test_df_boot$PopEstimate

  feat_w_boot <- feat_weights_full[colnames(train_x_boot)]

  xgb_train_boot <- xgb.DMatrix(train_x_boot, label = train_y_boot, weight = w_train, feature_weights = feat_w_boot)
  xgb_test_boot  <- xgb.DMatrix(test_x_boot,  label = test_y_boot,  weight = w_test,  feature_weights = feat_w_boot)

  mfit_xgb_boot <- xgboost(
    data = xgb_train_boot,
    nrounds = best_params[parameter == "nrounds", value],
    eta = best_params[parameter == "eta", value],
    max_depth = best_params[parameter == "max.depth", value],
    gamma = best_params[parameter == "gamma", value],
    min_child_weight = best_params[parameter == "min_child_weight", value],
    subsample = best_params[parameter == "subsample", value],
    colsample_bytree = best_params[parameter == "colsample_bytree", value],
    objective = "reg:tweedie",
    tree_method = "hist",
    max_bin = 256,
    tweedie_variance_power = 1.3,
    eval_metric = c("tweedie-nloglik@1.3","rmse"),
    nthread = cores)

  xgb.save(mfit_xgb_boot, file.path(out_dir, sprintf("final_xgb_model_bootstrap_%d_%s.json", b, spec)))

  # Predicted incidence -> cases
  preds_rate_b <- predict(mfit_xgb_boot, xgb_test_boot)
  preds_cases_b <- preds_rate_b * (test_df_boot$PopEstimate / 100000)

  imp_b <- xgb.importance(model = mfit_xgb_boot)
  fwrite(imp_b, file.path(out_dir, sprintf("feature_importance_bootstrap_%d_%s.csv", b, spec)))

  # Approximate SHAP on a subsample of the test set
  n_test <- nrow(test_x_boot)
  shap_idx <- if (n_test > N_shap) sample(seq_len(n_test), N_shap) else seq_len(n_test)

  xgb_shap <- xgb.DMatrix(
    test_x_boot[shap_idx, , drop = FALSE],
    label = test_y_boot[shap_idx],
    weight = w_test[shap_idx],
    feature_weights = feat_w_boot)

  shap_mat <- predict(mfit_xgb_boot, xgb_shap, predcontrib = TRUE, approxcontrib = TRUE)
  shap_dt  <- as.data.table(shap_mat)
  feat_cols <- setdiff(colnames(shap_dt), "BIAS")

  shap_imp_b <- data.table(
    Feature = feat_cols,
    mean_abs_shap   = sapply(feat_cols, function(v) mean(abs(shap_dt[[v]]), na.rm = TRUE)),
    median_abs_shap = sapply(feat_cols, function(v) median(abs(shap_dt[[v]]), na.rm = TRUE)),
    mean_shap       = sapply(feat_cols, function(v) mean(shap_dt[[v]], na.rm = TRUE)),
    shap_pos_rate   = sapply(feat_cols, function(v) mean(shap_dt[[v]] > 0, na.rm = TRUE))
  )[order(-mean_abs_shap)]

  fwrite(shap_imp_b, file.path(out_dir, sprintf("shap_importance_bootstrap_%d_%s.csv", b, spec)))

  # Decode month for output
  month_cols <- grep("^MONTH_\\d+$", colnames(test_df_boot), value = TRUE)
  mm <- as.matrix(test_df_boot[, month_cols, drop = FALSE])
  month_num <- max.col(mm, ties.method = "first")
  month_name <- ifelse(is.na(month_num), NA_character_, month.abb[month_num])

  out_dt <- data.table(
    GEOID = test_df_boot$GEOID,
    YEAR = test_df_boot$YEAR,
    MONTH = month_num,
    MONTH_NAME = month_name,
    pred_cases = preds_cases_b,
    actual_inc = test_df_boot$Enhanced_Inc,
    pop = test_df_boot$PopEstimate)
  fwrite(out_dt, file.path(out_dir, sprintf("test_predictions_bootstrap_%d_%s.csv", b, spec)))
}


##### 4. Combine bootstrap predictions and feature importance estimates #######

.get_boot_id <- function(x) as.integer(sub(".*bootstrap_(\\d+)_.*$", "\\1", basename(x)))

pred_files <- list.files(out_dir, pattern = sprintf("^test_predictions_bootstrap_\\d+_%s\\.csv$", spec), full.names = TRUE)

pred_list <- lapply(pred_files, function(f) {
  dt <- fread(f)
  dt[, bootstrap_id := .get_boot_id(f)]
  setnames(dt, "pred_cases", "Pred")
  dt[, .(GEOID, YEAR, MONTH, MONTH_NAME, Pred, bootstrap_id)]})

pred_all <- rbindlist(pred_list, use.names = TRUE, fill = TRUE)
setkey(pred_all, GEOID, YEAR, MONTH, bootstrap_id)

# Summary across bootstrap runs per GEOID–YEAR–MONTH
pred_summary <- pred_all[, .(
  n_runs = .N,
  mean_pred = mean(Pred, na.rm = TRUE),
  sd_pred = sd(Pred, na.rm = TRUE),
  q025 = as.numeric(quantile(Pred, 0.025, na.rm = TRUE)),
  q50  = as.numeric(quantile(Pred, 0.50,  na.rm = TRUE)),
  q975 = as.numeric(quantile(Pred, 0.975, na.rm = TRUE))
), by = .(GEOID, YEAR, MONTH, MONTH_NAME)]

fwrite(pred_all,     file.path(out_dir, sprintf("all_test_predictions_long_%s.csv", spec)))
fwrite(pred_summary, file.path(out_dir, sprintf("test_predictions_summary_by_geoid_year_month_%s.csv", spec)))

# Feature importance (gain)
imp_files <- list.files(out_dir, pattern = sprintf("^feature_importance_bootstrap_\\d+_%s\\.csv$", spec), full.names = TRUE)

imp_list <- lapply(imp_files, function(f) {
  dt <- fread(f)
  dt[, bootstrap_id := .get_boot_id(f)]
  dt[, .(Feature, Gain, Cover, Frequency, bootstrap_id)]})

imp_all <- rbindlist(imp_list, use.names = TRUE, fill = TRUE)
imp_all[, RankByGain := frank(-Gain, ties.method = "average"), by = bootstrap_id]

total_runs <- uniqueN(imp_all$bootstrap_id)

imp_summary <- imp_all[, .(
  mean_Gain = mean(Gain, na.rm = TRUE),
  median_Gain = median(Gain, na.rm = TRUE),
  mean_Cover = mean(Cover, na.rm = TRUE),
  mean_Frequency = mean(Frequency, na.rm = TRUE),
  mean_rank_by_gain = mean(RankByGain, na.rm = TRUE),
  selected_in = uniqueN(bootstrap_id),
  selection_rate = uniqueN(bootstrap_id) / total_runs,
  q025_Gain = as.numeric(quantile(Gain, 0.025, na.rm = TRUE)),
  q975_Gain = as.numeric(quantile(Gain, 0.975, na.rm = TRUE))
), by = Feature][order(mean_rank_by_gain, -mean_Gain)]

fwrite(imp_all,     file.path(out_dir, sprintf("all_feature_importance_long_%s.csv", spec)))
fwrite(imp_summary, file.path(out_dir, sprintf("feature_importance_summary_%s.csv", spec)))

# SHAP importance
shap_files <- list.files(out_dir, pattern = sprintf("^shap_importance_bootstrap_\\d+_%s\\.csv$", spec), full.names = TRUE)
shap_list <- lapply(shap_files, function(f) { dt <- fread(f); dt[, bootstrap_id := .get_boot_id(f)]; dt })

shap_all <- rbindlist(shap_list, use.names = TRUE, fill = TRUE)
total_runs <- uniqueN(shap_all$bootstrap_id)

shap_summary <- shap_all[, .(
  mean_mean_abs_shap   = mean(mean_abs_shap, na.rm = TRUE),
  median_mean_abs_shap = median(mean_abs_shap, na.rm = TRUE),
  q025_mean_abs_shap   = as.numeric(quantile(mean_abs_shap, 0.025, na.rm = TRUE)),
  q975_mean_abs_shap   = as.numeric(quantile(mean_abs_shap, 0.975, na.rm = TRUE)),
  mean_median_abs_shap = mean(median_abs_shap, na.rm = TRUE),
  mean_mean_shap       = mean(mean_shap, na.rm = TRUE),
  mean_shap_pos_rate   = mean(shap_pos_rate, na.rm = TRUE),
  selected_in          = uniqueN(bootstrap_id),
  selection_rate       = uniqueN(bootstrap_id) / total_runs), by = Feature][order(-mean_mean_abs_shap)]

fwrite(shap_all,     file.path(out_dir, sprintf("all_shap_importance_long_%s.csv", spec)))
fwrite(shap_summary, file.path(out_dir, sprintf("shap_importance_summary_%s.csv", spec)))

################################################################################
############## Evaluate model performance in California and Arizona ############
################################################################################
##### 5. Out-of-sample predictions at tract and county scales #####

pred_summary <- fread(file.path(out_dir, "test_predictions_summary_by_geoid_year_month_Main.csv"))
dataCompare  <- fread(file.path(out_dir, "FullTrainingAndTestingData_ForComparison_Main.csv"))

obsv_all <- as.data.table(dataCompare)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                           YEAR = as.integer(YEAR),
                                           MONTH = as.integer(MONTH),
                                           PopEstimate = as.numeric(PopEstimate),
                                           Obsv_cases = as.numeric(Enhanced_Inc) * as.numeric(PopEstimate) / 100000,
                                           Obsv_inc = as.numeric(Enhanced_Inc))]
setkey(obsv_all, GEOID, YEAR, MONTH)

pred_oos <- as.data.table(pred_summary)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                            YEAR = as.integer(YEAR),
                                            MONTH = as.integer(MONTH),
                                            Pred_cases_mean = as.numeric(mean_pred),
                                            Pred_cases_lwr = as.numeric(q025),
                                            Pred_cases_upr = as.numeric(q975),
                                            n_runs = as.integer(n_runs))]
setkey(pred_oos, GEOID, YEAR, MONTH)

comparedf_tract_oos <- merge(pred_oos, obsv_all, by = c("GEOID","YEAR","MONTH"), all.x = TRUE)

ok <- stats::complete.cases(comparedf_tract_oos[, c("Obsv_cases", "Pred_cases_mean")]) & comparedf_tract_oos$n_runs >= 10

annual_tract_oos <- comparedf_tract_oos[ok, .(Pred_Annual_cases = sum(Pred_cases_mean),
                                              Obsv_Annual_cases = sum(Obsv_cases)), by = .(GEOID, YEAR)]

comparedf_county_oos <- comparedf_tract_oos[ok, .(Pred_cases_mean = sum(Pred_cases_mean),
                                                  Obsv_cases = sum(Obsv_cases)),
                                            by = .(FIPS = as.integer(substr(GEOID, 1, 5)), YEAR, MONTH)]

annual_county_oos <- comparedf_county_oos[, .(Pred_Annual_cases = sum(Pred_cases_mean),
                                              Obsv_Annual_cases = sum(Obsv_cases)), by = .(FIPS, YEAR)]


##### 5b. Performance metrics (Supplementary Table S1) and bootstrap interval on bias #####

# Point estimates use the ensemble (mean across models); per-fit metrics are used only for the bias interval
# Per-fit county RMSE/deviance are on partial counties (~20% of tracts held out per fit), so are not reported
ensemble_metrics <- function(pred_summary_file, obsv_file, p_tweedie = 1.3, min_runs = 10) {
  obsv <- fread(obsv_file, select = c("GEOID","YEAR","MONTH","Enhanced_Inc","PopEstimate"),
                showProgress = FALSE)
  obsv <- obsv[, .(GEOID = as.numeric(GEOID), YEAR = as.integer(YEAR), MONTH = as.integer(MONTH),
                   Obsv = as.numeric(Enhanced_Inc) * as.numeric(PopEstimate) / 1e5)]
  setkey(obsv, GEOID, YEAR, MONTH)

  p <- fread(pred_summary_file, select = c("GEOID","YEAR","MONTH","mean_pred","n_runs"),
             showProgress = FALSE)[n_runs >= min_runs]
  p[, `:=`(GEOID = as.numeric(GEOID), YEAR = as.integer(YEAR), MONTH = as.integer(MONTH))]
  setkey(p, GEOID, YEAR, MONTH)

  d <- merge(p, obsv, by = c("GEOID","YEAR","MONTH"))[!is.na(Obsv) & !is.na(mean_pred)]
  d[, FIPS := as.integer(substr(sprintf("%011.0f", GEOID), 1, 5))]

  met <- function(x, lab) data.table(
    scale = lab, n = nrow(x),
    R2 = cor(x$Obsv, x$Pred)^2,
    RMSE = sqrt(mean((x$Obsv - x$Pred)^2)),
    TweedieD = mean(tweedie_deviance(y = x$Obsv, mu = x$Pred, p = p_tweedie)),
    bias_pct = 100 * (sum(x$Pred) - sum(x$Obsv)) / sum(x$Obsv))

  rbindlist(list(
    met(d[, .(Pred = mean_pred, Obsv)], "tract-month"),
    met(d[, .(Pred = sum(mean_pred), Obsv = sum(Obsv)), by = .(GEOID, YEAR)], "tract-year"),
    met(d[, .(Pred = sum(mean_pred), Obsv = sum(Obsv)), by = .(FIPS, YEAR, MONTH)], "county-month"),
    met(d[, .(Pred = sum(mean_pred), Obsv = sum(Obsv)), by = .(FIPS, YEAR)], "county-year")))
}

ensemble_main <- ensemble_metrics(
  file.path(out_dir, "test_predictions_summary_by_geoid_year_month_Main.csv"),
  file.path(out_dir, "FullTrainingAndTestingData_ForComparison_Main.csv"))
print(ensemble_main)
fwrite(ensemble_main, file.path(out_dir, "performance_metrics_ensemble_Main.csv"))

boot_metrics <- function(pred_long_file, obsv_file, p_tweedie = 1.3) {

  obsv <- fread(obsv_file, select = c("GEOID","YEAR","MONTH","Enhanced_Inc","PopEstimate"),
                showProgress = FALSE)
  obsv <- obsv[, .(GEOID = as.numeric(GEOID), YEAR = as.integer(YEAR), MONTH = as.integer(MONTH),
                   Obsv_cases = as.numeric(Enhanced_Inc) * as.numeric(PopEstimate) / 1e5)]
  setkey(obsv, GEOID, YEAR, MONTH)

  pred_long <- fread(pred_long_file, select = c("GEOID","YEAR","MONTH","Pred","bootstrap_id"),
                     showProgress = FALSE)
  pred_long[, `:=`(GEOID = as.numeric(GEOID), YEAR = as.integer(YEAR), MONTH = as.integer(MONTH))]
  setkey(pred_long, GEOID, YEAR, MONTH)

  dt <- merge(pred_long, obsv, by = c("GEOID","YEAR","MONTH"))[!is.na(Obsv_cases) & !is.na(Pred)]
  rm(pred_long); invisible(gc())

  # list() not .(), and nrow(d) not .N -- neither is visible inside a function called from j
  metrics <- function(d) list(
    n        = nrow(d),
    R2       = cor(d$Obsv_cases, d$Pred)^2,
    RMSE     = sqrt(mean((d$Obsv_cases - d$Pred)^2)),
    MAE      = mean(abs(d$Obsv_cases - d$Pred)),
    TweedieD = mean(tweedie_deviance(y = d$Obsv_cases, mu = d$Pred, p = p_tweedie)),
    bias_pct = 100 * (sum(d$Pred) - sum(d$Obsv_cases)) / sum(d$Obsv_cases))

  summarise_boot <- function(bt, label) {
    cols <- setdiff(names(bt), "bootstrap_id")
    rbindlist(lapply(cols, function(v) {
      x <- bt[[v]]
      data.table(scale = label, metric = v, mean = mean(x),
                 lwr = quantile(x, 0.025), upr = quantile(x, 0.975))}))
  }

  # Aggregate within each bootstrap at each scale, then compute metrics
  bt_tm <- dt[, metrics(.SD), by = bootstrap_id, .SDcols = c("Pred","Obsv_cases")]

  ty <- dt[, .(Pred = sum(Pred), Obsv_cases = sum(Obsv_cases)), by = .(bootstrap_id, GEOID, YEAR)]
  bt_ty <- ty[, metrics(.SD), by = bootstrap_id, .SDcols = c("Pred","Obsv_cases")]

  dt[, FIPS := as.integer(substr(sprintf("%011.0f", GEOID), 1, 5))]
  cm <- dt[, .(Pred = sum(Pred), Obsv_cases = sum(Obsv_cases)), by = .(bootstrap_id, FIPS, YEAR, MONTH)]
  bt_cm <- cm[, metrics(.SD), by = bootstrap_id, .SDcols = c("Pred","Obsv_cases")]

  cy <- cm[, .(Pred = sum(Pred), Obsv_cases = sum(Obsv_cases)), by = .(bootstrap_id, FIPS, YEAR)]
  bt_cy <- cy[, metrics(.SD), by = bootstrap_id, .SDcols = c("Pred","Obsv_cases")]

  rbindlist(list(summarise_boot(bt_tm, "tract-month"),  summarise_boot(bt_ty, "tract-year"),
                 summarise_boot(bt_cm, "county-month"), summarise_boot(bt_cy, "county-year")))
}

results_main <- boot_metrics(
  file.path(out_dir, "all_test_predictions_long_Main.csv"),
  file.path(out_dir, "FullTrainingAndTestingData_ForComparison_Main.csv"))

# Bias is identical at every scale; tract-level R2 range = range across individual model fits
print(results_main[metric == "bias_pct"])
print(results_main[metric == "R2" & scale %in% c("tract-month","tract-year")])
fwrite(results_main, file.path(out_dir, "performance_metrics_bootstrapCI_Main.csv"))


##### 5c. Supplementary Figure S1: predicted vs reported cases (4 panels) #######

op <- par(no.readonly = TRUE)
par(mfrow = c(2, 2), mar = c(4.5, 4.5, 2.5, 1), oma = c(0, 0, 3, 0), mgp = c(2.2, 0.7, 0))

add_inset <- function(r2, n) {
  usr <- par("usr")
  x <- usr[1] + 0.96 * (usr[2] - usr[1])
  y <- usr[3] + 0.05 * (usr[4] - usr[3])
  lbl <- bquote(atop(R^2 == .(format(round(r2, 3), nsmall = 3)),
                     N == .(format(n, big.mark = ","))))
  text(x, y, labels = lbl, adj = c(1, 0), cex = 1.2, font = 1)}

## 1) TRACT-MONTH
plot(Pred_cases_mean ~ Obsv_cases, data = comparedf_tract_oos[ok, ],
     xlab = "Reported cases", ylab = "Predicted cases",
     main = "", cex.lab = 1.3, cex.axis = 1.1, pch = 16, col = rgb(0,0,0,0.2))
abline(lm(Pred_cases_mean ~ Obsv_cases, data = comparedf_tract_oos[ok, ]), col = "firebrick2", lwd = 3)
abline(0, 1, lty = 2, col = "#2B76CC", lwd = 3)
mtext("Tract-month", side = 3, line = 0.5, cex = 1.2, font = 1)
add_inset(cor(comparedf_tract_oos$Obsv_cases[ok], comparedf_tract_oos$Pred_cases_mean[ok])^2, sum(ok))

## 2) TRACT-YEAR
plot(Pred_Annual_cases ~ Obsv_Annual_cases, data = annual_tract_oos,
     xlab = "Reported cases", ylab = "Predicted cases",
     main = "", cex.lab = 1.3, cex.axis = 1.1, pch = 16, col = rgb(0,0,0,0.2))
abline(lm(Pred_Annual_cases ~ Obsv_Annual_cases, data = annual_tract_oos), col = "firebrick2", lwd = 3)
abline(0, 1, lty = 2, col = "#2B76CC", lwd = 3)
mtext("Tract-year", side = 3, line = 0.5, cex = 1.2, font = 1)
add_inset(cor(annual_tract_oos$Obsv_Annual_cases, annual_tract_oos$Pred_Annual_cases)^2, nrow(annual_tract_oos))

## 3) COUNTY-MONTH
plot(Pred_cases_mean ~ Obsv_cases, data = comparedf_county_oos,
     ylab = "Predicted cases", xlab = "Reported cases",
     main = "", cex.lab = 1.3, cex.axis = 1.1)
abline(lm(Pred_cases_mean ~ Obsv_cases, data = comparedf_county_oos), col = "firebrick2", lwd = 3)
abline(0, 1, lty = 2, col = "#2B76CC", lwd = 3)
mtext("County-month", side = 3, line = 0.5, cex = 1.2, font = 1)
add_inset(cor(comparedf_county_oos$Obsv_cases, comparedf_county_oos$Pred_cases_mean)^2, nrow(comparedf_county_oos))

## 4) COUNTY-YEAR
plot(Pred_Annual_cases ~ Obsv_Annual_cases, data = annual_county_oos,
     ylab = "Predicted cases", xlab = "Reported cases",
     main = "", cex.lab = 1.3, cex.axis = 1.1)
abline(lm(Pred_Annual_cases ~ Obsv_Annual_cases, data = annual_county_oos), col = "firebrick2", lwd = 3)
abline(0, 1, lty = 2, col = "#2B76CC", lwd = 3)
mtext("County-year", side = 3, line = 0.5, cex = 1.2, font = 1)
add_inset(cor(annual_county_oos$Obsv_Annual_cases, annual_county_oos$Pred_Annual_cases)^2, nrow(annual_county_oos))

mtext("Model performance for Arizona and California tracts", side = 3, outer = TRUE, line = 0.2, cex = 1.4, font = 1)
par(op)


##### 6. Supplementary Figure S3: spatial bias in tract-level predictions ######

pad11 <- function(x) sprintf("%011s", gsub("\\D", "", as.character(x)))

ca2020 <- tigris::tracts("CA", year = 2020, class = "sf", cb = TRUE) %>% mutate(GEOID = pad11(GEOID))
az2020 <- tigris::tracts("AZ", year = 2020, class = "sf", cb = TRUE) %>% mutate(GEOID = pad11(GEOID))
v2020  <- dplyr::bind_rows(ca2020, az2020)

ca_counties <- tigris::counties("CA", year = 2020, class = "sf", cb = TRUE)
az_counties <- tigris::counties("AZ", year = 2020, class = "sf", cb = TRUE)
countiesCAZ <- dplyr::bind_rows(ca_counties, az_counties)

statesCAZ <- tigris::states(year = 2020, class = "sf", cb = TRUE) %>%
  dplyr::filter(STUSPS %in% c("CA", "AZ"))

# Total bias by tract (summed across all years/months)
bias_total <- comparedf_tract_oos[ok, ] %>%
  mutate(GEOID = pad11(GEOID)) %>%
  group_by(GEOID) %>%
  summarise(
    Obsv_total = sum(Obsv_cases, na.rm = TRUE),
    Pred_total = sum(Pred_cases_mean, na.rm = TRUE),
    .groups = "drop") %>%
  mutate(pct_over = case_when(
    is.na(Obsv_total) | is.na(Pred_total) ~ NA_real_,
    Obsv_total == 0 & Pred_total == 0 ~ 0,
    Obsv_total == 0 & Pred_total > 0 ~ NA_real_,
    TRUE ~ 100 * ((Pred_total / Obsv_total) - 1)))

v2020_bias <- v2020 %>% left_join(bias_total, by = "GEOID")

# Cap extreme values for visualization
cap <- 200
v2020_bias <- v2020_bias %>%
  mutate(pct_over_plot = pmax(pmin(pct_over, cap), -cap))

ggplot(v2020_bias) +
  geom_sf(aes(fill = pct_over_plot), color = NA) +
  geom_sf(data = countiesCAZ, fill = NA, color = "gray80", size = 0.2) +
  geom_sf(data = statesCAZ,   fill = NA, color = "black",  size = 0.4) +
  scale_fill_gradient2(low = alpha("#2166ac", 0.7), mid = "white", high = alpha("#b2182b", 0.7), midpoint = 0,
                       limits = c(-cap, cap),oob = scales::squish, na.value = "gray97",
                       name = "% over / under\nprediction") +
  theme_void() +
  labs(title = "\nPercent over- vs under-prediction by census tract") +
  guides(fill = guide_colorbar(barheight = unit(3.3, "cm"), barwidth  = unit(0.7, "cm"))) +
  theme(plot.title = element_text(size = 15, hjust = 0.5),
        legend.position = c(0.9, 0.75),
        legend.title  = element_text(size = 12),
        legend.text   = element_text(size = 10.5),
        legend.background = element_rect(fill = alpha("white", 0.6), color = NA))

##### 6b. Supplementary Figure S4: predictors of over- and underprediction #######

bias_labels <- bias_total %>%
  mutate(BiasLabel = case_when(
    pct_over >= 50 ~ "Over_50",
    pct_over <= -50 ~ "Under_50",
    TRUE ~ "Within_50"))

# Requires 'dataModel' from section 2 (with spec = "Main")
dataModel_labeled <- dataModel %>%
  mutate(GEOID_chr = pad11(GEOID)) %>%
  left_join(bias_labels %>% mutate(GEOID = pad11(GEOID)),
            by = c("GEOID_chr" = "GEOID"))

dataModel_labeled <- dataModel_labeled %>%
  mutate(Bias3 = case_when(
    BiasLabel == "Over_50" ~ "Over_50",
    BiasLabel == "Under_50" ~ "Under_50",
    BiasLabel == "Within_50" ~ "WellPredicted",
    TRUE ~ NA_character_))

numeric_cols <- names(dataModel_labeled)[sapply(dataModel_labeled, is.numeric)]
cols_pred <- numeric_cols[!numeric_cols %in% c("GEOID", "YEAR", "cases", "pct_over", "Pred_total", "Obsv_total", "Enhanced_Inc") & !grepl("^MONTH_", numeric_cols)]

data_std <- dataModel_labeled %>%
  mutate(across(all_of(cols_pred), ~ as.numeric(scale(.x)), .names = "{.col}_z"))

# Compare a bias group to well-predicted tracts only
.get_group_vs_neutral <- function(dat_raw, dat_std, group_label, neutral_label = "WellPredicted") {
  dat_raw2 <- dat_raw %>% filter(Bias3 %in% c(group_label, neutral_label))
  dat_std2 <- dat_std %>% filter(Bias3 %in% c(group_label, neutral_label))

  raw_grp_neu <- dat_raw2 %>%
    group_by(Bias3) %>%
    summarise(across(all_of(cols_pred), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    arrange(match(Bias3, c(group_label, neutral_label))) %>%
    mutate(BiasLabel = Bias3) %>% select(-Bias3)

  std_grp_neu <- dat_std2 %>%
    group_by(Bias3) %>%
    summarise(across(ends_with("_z"), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    arrange(match(Bias3, c(group_label, neutral_label)))

  diff_vec <- as.numeric(std_grp_neu[1, -1]) - as.numeric(std_grp_neu[2, -1])
  names(diff_vec) <- gsub("_z$", "", names(std_grp_neu)[-1])

  diff_tbl <- tibble(predictor = names(diff_vec), diff_z = as.numeric(diff_vec)) %>%
    mutate(abs_diff = abs(diff_z)) %>%
    arrange(desc(abs_diff))

  list(raw_grp_neu = raw_grp_neu, diff_vec = diff_vec, diff_tbl = diff_tbl)}

over_obj <- .get_group_vs_neutral(dataModel_labeled, data_std, "Over_50")
under_obj <- .get_group_vs_neutral(dataModel_labeled, data_std, "Under_50")

top_over_vars <- over_obj$diff_tbl %>% head(10) %>% pull(predictor)
top_under_vars <- under_obj$diff_tbl %>% head(10) %>% pull(predictor)

plot_vars <- union(top_over_vars, top_under_vars)
z_cols <- paste0(plot_vars, "_z")

plot_df <- data_std %>%
  filter(Bias3 %in% c("Over_50", "Under_50", "WellPredicted")) %>%
  select(Bias3, all_of(z_cols)) %>%
  pivot_longer(cols = all_of(z_cols), names_to = "predictor_z", values_to = "z_value") %>%
  mutate(predictor = gsub("_z$", "", predictor_z)) %>%
  group_by(Bias3, predictor) %>%
  summarise(z_mean = mean(z_value, na.rm = TRUE), .groups = "drop")

# Order predictors by largest absolute difference from well-predicted tracts
pred_order <- bind_rows(
  over_obj$diff_tbl %>% filter(predictor %in% plot_vars) %>% mutate(Comparison = "Over"),
  under_obj$diff_tbl %>% filter(predictor %in% plot_vars) %>% mutate(Comparison = "Under")) %>%
  group_by(predictor) %>%
  summarise(max_abs_diff = max(abs(diff_z), na.rm = TRUE), .groups = "drop") %>%
  arrange(max_abs_diff) %>%
  pull(predictor)

plot_df <- plot_df %>%
  mutate(predictor = factor(predictor, levels = pred_order),
         BiasGroup = recode(Bias3, WellPredicted = "AllOthers", Over_50 = "Over", Under_50 = "Under"),
         BiasGroup = factor(BiasGroup, levels = c("Over", "Under", "AllOthers")))

pretty_feat <- c(SkinTest_PercentReactors_Numeric = "Coccidioidin reactivity",
                 mean_annual_precip = "Mean annual precipitation",
                 mean_annual_temp = "Mean annual temperature",
                 SoilOrgC_30cm = "Soil organic carbon",
                 elev_m = "Elevation", pct_OpenWater = "% open water",
                 PopEstimate = "Population size", median_hh_income = "Median household income",
                 tmin_lag_12 = "Min. temperature (12-mo lag)",
                 tmin_lag_8 = "Min. temperature (8-mo lag)",
                 tmin_lag_13 = "Min. temperature (13-mo lag)",
                 pct_MixedForest = "% mixed forest", pct_Evergreen = "% Evergreen forest",
                 pct_Grassland = "% grassland", pct_HerbyWetlands = "% Herby wetlands",
                 pct_uninsured_u65 = "% uninsured <65 yrs", precip_lag_24 = "Precip. (24-mo lag)",
                 precip_lag_23 = "Precip. (23-mo lag)", pct_Barren = "% barren",
                 pct_ScrubShrub = "% Scrub/shrub",
                 EHRInc_wt_downscale = "EHR-based incidence",
                 modal_land_DevelopedMed = "Modal land cover: developed (mid)",
                 modal_land_Evergreen = "Modal land cover: evergreen",
                 modal_land_Grassland = "Modal land cover: grassland",
                 modal_land_DevelopedOpen = "Modal land cover: developed (open)",
                 pct_construction_extraction = "% Construction/extraction",
                 pct_NoLandData = "% Land cover missing",
                 pct_DevelopedHigh = "% Developed (high)",
                 pct_DevelopedMed = "% Developed (mid)",
                 SoilTexture_30cm_SaLo = "Sandy Loam soil",
                 SoilWaterPercent_30cm = "Soil water %",
                 modal_land_OpenWater = "Modal land cover: open water",
                 pct_age_65plus = "% 65+", SoilTexture_30cm_ClLo = "Clay loam soil",
                 SoilTexture_30cm_SiLo = "Silt loam soil",
                 Total_Pos_Dog_Tests_lag_0 = "# Positive canine tests",
                 mean_dust_ugm3_lag_3 = "Dust concs. (3-mo lag)",
                 mean_dust_ugm3_lag_2 = "Dust concs. (2-mo lag)",
                 mean_dust_ugm3_lag_1 = "Dust concs. (1-mo lag)",
                 Monthly_Dog_Inc_lag_0 = "Canine incidence",
                 SoilTexture_30cm_Lo = "Loam soil texture",
                 Total_Dog_Tests_lag_0 = "Total # canine tests", Diabetes_Per = "Diabetes prevalence",
                 pct_farming_fishing_forestry = "% farming/fishing/forestry")

plot_df_pretty <- plot_df %>%
  mutate(predictor_pretty = dplyr::recode(as.character(predictor), !!!pretty_feat),
         predictor_pretty = factor(predictor_pretty, levels = dplyr::recode(pred_order, !!!pretty_feat)))

ggplot(plot_df_pretty, aes(x = z_mean, y = predictor_pretty, fill = BiasGroup)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey60") +
  geom_col(position = position_dodge(width = 0.6), width = 0.55) +
  labs(x = "Mean z-score", y = NULL, fill = NULL,
       title = "Predictor profiles for under- and over-predicted tracts") +
  scale_fill_manual(values = c(Over = alpha("#b2182b", 0.8), Under = alpha("#2166ac", 0.8), AllOthers = "grey30")) +
  theme_minimal() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "right",
        axis.title.x = element_text(size = 12),
        axis.text.y = element_text(size = 11.5),
        axis.text.x = element_text(size = 12),
        legend.text = element_text(size = 11),
        legend.title = element_text(size = 12),
        plot.title = element_text(size = 14, hjust = 0.7))


##### 7. Supplementary Figure S5: monthly time series for Arizona and California ####

comparedf_tract_oos <- comparedf_tract_oos %>%
  mutate(GEOID_chr = pad11(GEOID),
         STATEFP = substr(GEOID_chr, 1, 2),
         Group = case_when(STATEFP == "04" ~ "Arizona",
                           STATEFP == "06" ~ "California",
                           TRUE ~ NA_character_))

ok <- stats::complete.cases(comparedf_tract_oos[, c("Obsv_cases", "Pred_cases_mean")]) & comparedf_tract_oos$n_runs >= 10

# Ribbon sums tract-level quantiles (a conservative bound); each bootstrap holds out only ~20% of tracts,
# so a within-bootstrap state total is not available here
df_time_state <- comparedf_tract_oos[ok, ] %>%
  filter(!is.na(Group), complete.cases(Obsv_cases, Pred_cases_mean, Pred_cases_lwr, Pred_cases_upr)) %>%
  group_by(YEAR, MONTH, Group) %>%
  summarise(
    Reported = sum(Obsv_cases, na.rm = TRUE),
    Predicted = sum(Pred_cases_mean, na.rm = TRUE),
    q05 = sum(Pred_cases_lwr, na.rm = TRUE),
    q95 = sum(Pred_cases_upr, na.rm = TRUE),
    .groups = "drop") %>%
  mutate(date = as.Date(paste0(YEAR, "-", MONTH, "-01")))

df_time_state_long <- df_time_state %>%
  select(date, Group, Reported, Predicted) %>%
  pivot_longer(cols = c("Reported", "Predicted"), names_to = "Type", values_to = "Cases")

ggplot() + geom_ribbon(data = df_time_state,
                       aes(x = date, ymin = q05, ymax = q95, group = Group),
                       inherit.aes = FALSE, fill = "firebrick3", alpha = 0.12) +
  geom_line(data = df_time_state_long,
            aes(x = date, y = Cases, color = Type, linetype = Type,
                group = interaction(Group, Type)),
            linewidth = 0.7) +
  scale_color_manual(values = c("Reported" = alpha("black", 0.9), "Predicted" = "firebrick3")) +
  scale_linetype_manual(values = c("Reported" = "solid", "Predicted" = "dashed")) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  labs(x = NULL, y = "Monthly cases (summed by state)",
       title = "Monthly cases for 2013 - 2023 (predicted and reported)") +
  facet_wrap(~ Group, ncol = 1, scales = "free_y") +
  theme_bw(base_size = 13) +
  theme(
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.text.x = element_text(hjust = 0.5, size = 14),
    axis.text.y = element_text(hjust = 0.5, size = 14),
    axis.title.y = element_text(size = 16),
    strip.text = element_text(size = 14),
    strip.background = element_rect(fill = "grey95"),
    legend.position = c(0.01, 0.98),
    legend.justification = c("left", "top"),
    legend.background = element_rect(fill = alpha("white", 0.7), color = NA),
    panel.grid.major = element_line(color = "grey96"),
    panel.grid.minor = element_line(color = "grey96")) +
  guides(color = guide_legend(keywidth = unit(1.3, "cm"), override.aes = list(linewidth = 1.2)),
         linetype = guide_legend(keywidth = unit(1.3, "cm")))

# Correlation between predicted and reported state time series
AZ_wide <- df_time_state %>% filter(Group == "Arizona") %>% arrange(YEAR, MONTH)
ob_AZ <- AZ_wide$Reported; pr_AZ <- AZ_wide$Predicted
r_AZ <- cor(ob_AZ, pr_AZ, use = "complete.obs")
R2_AZ <- 1 - sum((ob_AZ - pr_AZ)^2, na.rm = TRUE) / sum((ob_AZ - mean(ob_AZ, na.rm = TRUE))^2, na.rm = TRUE)

CA_wide <- df_time_state %>% filter(Group == "California") %>% arrange(YEAR, MONTH)
ob_CA <- CA_wide$Reported; pr_CA <- CA_wide$Predicted
r_CA <- cor(ob_CA, pr_CA, use = "complete.obs")
R2_CA <- 1 - sum((ob_CA - pr_CA)^2, na.rm = TRUE) / sum((ob_CA - mean(ob_CA, na.rm = TRUE))^2, na.rm = TRUE)

r_AZ; R2_AZ
r_CA; R2_CA


##### 8. Model performance under alternative specifications (Supplementary Table S2) ##############
########### Alt Spec No Skin Test Predictor ##############################################

pred_summary <- fread(file.path(out_dir, "test_predictions_summary_by_geoid_year_month_AltSpec_NoSkinTest.csv"))
dataCompare  <- fread(file.path(out_dir, "FullTrainingAndTestingData_ForComparison_AltSpec_NoSkinTest.csv"))

obsv_all <- as.data.table(dataCompare)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                           YEAR = as.integer(YEAR),
                                           MONTH = as.integer(MONTH),
                                           Obsv = as.numeric(Enhanced_Inc) * PopEstimate / 100000)]

pred_oos <- as.data.table(pred_summary)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                            YEAR = as.integer(YEAR),
                                            MONTH = as.integer(MONTH),
                                            mean_pred = as.numeric(mean_pred),
                                            n_runs = as.integer(n_runs))]

setkey(pred_oos, GEOID, YEAR, MONTH); setkey(obsv_all, GEOID, YEAR, MONTH)
comparedf_alt <- pred_oos[obsv_all]

ok_alt <- stats::complete.cases(comparedf_alt[, c("Obsv", "mean_pred")]) & comparedf_alt$n_runs >= 2

r2 <- cor(comparedf_alt$Obsv[ok_alt], comparedf_alt$mean_pred[ok_alt])^2
rmse <- sqrt(mean((comparedf_alt$Obsv[ok_alt] - comparedf_alt$mean_pred[ok_alt])^2))
cat(sprintf("Tracts OOS (monthly cases):  R^2 = %.3f   RMSE = %.3f", r2, rmse))

mean(tweedie_deviance(y = comparedf_alt$Obsv[ok_alt], mu = comparedf_alt$mean_pred[ok_alt], p = 1.2))
nrow(comparedf_alt[ok_alt, ])
(mean(comparedf_alt$mean_pred[ok_alt]) - mean(comparedf_alt$Obsv[ok_alt])) / mean(comparedf_alt$Obsv[ok_alt])

# County-month
comparedf_alt_county <- comparedf_alt[ok_alt, .(mean_pred = sum(mean_pred, na.rm = TRUE),
                                                Obsv_county = sum(Obsv, na.rm = TRUE)),
                                      by = .(FIPS = as.integer(substr(GEOID, 1, 5)), YEAR, MONTH)]

r2c <- cor(comparedf_alt_county$Obsv_county, comparedf_alt_county$mean_pred)^2
rmsec <- sqrt(mean((comparedf_alt_county$Obsv_county - comparedf_alt_county$mean_pred)^2))
cat(sprintf("Counties OOS (monthly cases):  R^2 = %.3f   RMSE = %.3f", r2c, rmsec))

mean(tweedie_deviance(y = comparedf_alt_county$Obsv_county, mu = comparedf_alt_county$mean_pred, p = 1.2))
nrow(comparedf_alt_county)

results_noskin <- boot_metrics(file.path(out_dir, "all_test_predictions_long_AltSpec_NoSkinTest.csv"),
                               file.path(out_dir, "FullTrainingAndTestingData_ForComparison_AltSpec_NoSkinTest.csv"))
print(results_noskin[metric == "bias_pct"])

########### Alt Spec No Vet/Epi Predictors ##############################################

pred_summary <- fread(file.path(out_dir, "test_predictions_summary_by_geoid_year_month_AltSpec_NoVetEpi.csv"))
dataCompare  <- fread(file.path(out_dir, "FullTrainingAndTestingData_ForComparison_AltSpec_NoVetEpi.csv"))

obsv_all <- as.data.table(dataCompare)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                           YEAR = as.integer(YEAR),
                                           MONTH = as.integer(MONTH),
                                           Obsv = as.numeric(Enhanced_Inc) * PopEstimate / 100000)]

pred_oos <- as.data.table(pred_summary)[, .(GEOID = sprintf("%011s", as.character(GEOID)),
                                            YEAR = as.integer(YEAR),
                                            MONTH = as.integer(MONTH),
                                            mean_pred = as.numeric(mean_pred),
                                            n_runs = as.integer(n_runs))]

setkey(pred_oos, GEOID, YEAR, MONTH); setkey(obsv_all, GEOID, YEAR, MONTH)
comparedf_alt <- pred_oos[obsv_all]

ok_alt <- stats::complete.cases(comparedf_alt[, c("Obsv", "mean_pred")]) & comparedf_alt$n_runs >= 2

r2 <- cor(comparedf_alt$Obsv[ok_alt], comparedf_alt$mean_pred[ok_alt])^2
rmse <- sqrt(mean((comparedf_alt$Obsv[ok_alt] - comparedf_alt$mean_pred[ok_alt])^2))
cat(sprintf("Tracts OOS (monthly cases):  R^2 = %.3f   RMSE = %.3f", r2, rmse))
(mean(comparedf_alt$mean_pred[ok_alt]) - mean(comparedf_alt$Obsv[ok_alt])) / mean(comparedf_alt$Obsv[ok_alt])

comparedf_alt_county <- comparedf_alt[ok_alt, .(mean_pred = sum(mean_pred, na.rm = TRUE),
                                                Obsv_county = sum(Obsv, na.rm = TRUE)),
                                      by = .(FIPS = as.integer(substr(GEOID, 1, 5)), YEAR, MONTH)]

r2c <- cor(comparedf_alt_county$Obsv_county, comparedf_alt_county$mean_pred)^2
rmsec <- sqrt(mean((comparedf_alt_county$Obsv_county - comparedf_alt_county$mean_pred)^2))
cat(sprintf("Counties OOS (monthly cases):  R^2 = %.3f   RMSE = %.3f", r2c, rmsec))

results_novetepi <- boot_metrics(file.path(out_dir, "all_test_predictions_long_AltSpec_NoVetEpi.csv"),
                                 file.path(out_dir, "FullTrainingAndTestingData_ForComparison_AltSpec_NoVetEpi.csv"))
print(results_novetepi[metric == "bias_pct"])

################################################################################
#################### Feature importance ########################################
################################################################################
#### 9. Figure 3a: SHAP feature importance ######

shap_summary <- fread(file.path(out_dir, "shap_importance_summary_Main.csv"))

top_n <- 12
shap_top <- shap_summary %>%
  arrange(desc(mean_mean_abs_shap)) %>%
  slice_head(n = top_n) %>%
  mutate(Feature = fct_reorder(Feature, mean_mean_abs_shap))

shap_top <- shap_top %>% mutate(Feature =
                                  {recode_map <- c("Coccidioidin reactivity"="SkinTest_PercentReactors_Numeric",
                                                   "Mean annual precip."="mean_annual_precip","Year"="YEAR",
                                                   "Canine incidence"="Monthly_Dog_Inc_lag_0",
                                                   "# positive canine tests"="Total_Pos_Dog_Tests_lag_0",
                                                   "# canine tests"="Total_Dog_Tests_lag_0",
                                                   "EHR incidence"="EHRInc_wt_downscale","Elevation"="elev_m",
                                                   "% 65+"="pct_age_65plus","Mean annual temp."="mean_annual_temp",
                                                   "Soil organic carbon %"="SoilOrgC_10cm","Diabetes prevalence"="Diabetes_Per",
                                                   "% farm/fish/forestry"="pct_farming_fishing_forestry");
                                  do.call(forcats::fct_recode, c(list(Feature), as.list(recode_map[recode_map %in% levels(Feature)])))})

xmax <- max(shap_top$q975_mean_abs_shap, na.rm = TRUE)

ggplot(shap_top, aes(x = mean_mean_abs_shap, y = Feature)) +
  geom_errorbarh(aes(xmin = q025_mean_abs_shap, xmax = q975_mean_abs_shap), height = 0) +
  geom_point(size = 2.6) +
  labs(x = "Mean |SHAP| (mean ± 95% CI)", y = NULL) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
    axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 12),
        axis.title.x = element_text(size = 14),
        plot.title = element_text(face = "bold", hjust = 1, size = 16),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(t = 5.5, r = 18, b = 5.5, l = 18)) +
  scale_x_continuous(limits = c(0, xmax * 1.05))

#### 9b. Figure 3b: partial dependence plots #######

vars <- c("mean_annual_precip", "Monthly_Dog_Inc_lag_0", "SkinTest_PercentReactors_Numeric",
          "EHRInc_wt_downscale", "elev_m", "mean_annual_temp")

pretty_map <- c("mean_annual_precip" = "Mean annual precip. (mm)",
                "Monthly_Dog_Inc_lag_0" = "Canine inc. (cases/10K hh)",
                "SkinTest_PercentReactors_Numeric" = "Coccidioidin reactivity (%)",
                "EHRInc_wt_downscale" = "EHR inc. (cases/100K)",
                "elev_m" = "Elevation (m)",
                "mean_annual_temp" = "Mean annual temp. (°C)")

# Requires 'dataModel' from section 2 (with spec = "Main")
X_all <- dataModel %>% select(-c(Enhanced_Inc, GEOID, PopEstimate))
pop_vec <- pmax(dataModel$PopEstimate, 1)

# PDP for one model, returned as mean predicted cases per tract-month
compute_pdp_one_model <- function(model_path, feat_name, grid_vals, X_all, pop_all, n_bg = 5000) {
  n <- nrow(X_all); bg_idx <- if (n_bg >= n) seq_len(n) else sample.int(n, n_bg)
  X_bg <- X_all[bg_idx, , drop = FALSE]; pop_bg <- pop_all[bg_idx]
  feat_col <- which(colnames(X_bg) == feat_name); if (length(feat_col) != 1) stop("Feature not found (or duplicated): ", feat_name)
  mdl <- xgb.load(model_path)
  pd_vals <- numeric(length(grid_vals))
  for (i in seq_along(grid_vals)) {
    X_tmp <- X_bg; X_tmp[, feat_col] <- grid_vals[i]
    dtemp <- xgb.DMatrix(as.matrix(X_tmp))
    pred_rate <- predict(mdl, dtemp)
    pred_cases <- pred_rate * (pop_bg / 100000)
    pd_vals[i] <- mean(pred_cases, na.rm = TRUE)}
  data.frame(Feature = feat_name, feat_value = grid_vals, pd_cases = pd_vals)}

boot_ids <- 1:100

pdp_all <- bind_rows(lapply(vars, function(v) {
  fvals <- X_all[[v]]
  grid_vals <- seq(quantile(fvals, 0.05, na.rm = TRUE), quantile(fvals, 0.95, na.rm = TRUE), length.out = 30)
  pdp_list <- lapply(boot_ids, function(b) {
    model_path <- file.path(out_dir, sprintf("final_xgb_model_bootstrap_%d_Main.json", b))
    df <- compute_pdp_one_model(model_path, v, grid_vals, X_all = as.matrix(X_all), pop_all = pop_vec, n_bg = 2000)
    df$bootstrap_id <- b; df})
  bind_rows(pdp_list)}))

pdp_mean <- pdp_all %>% group_by(Feature, feat_value) %>%
  summarise(pd_cases_mean = mean(pd_cases, na.rm = TRUE), .groups = "drop")

pdp_all <- pdp_all %>% mutate(Pretty = ifelse(Feature %in% names(pretty_map), unname(pretty_map[Feature]), Feature))
pdp_mean <- pdp_mean %>% mutate(Pretty = ifelse(Feature %in% names(pretty_map), unname(pretty_map[Feature]), Feature))

panel_order <- c("Canine inc. (cases/10K hh)","Mean annual precip. (mm)","Coccidioidin reactivity (%)", "Elevation (m)","EHR inc. (cases/100K)","Mean annual temp. (°C)")

pdp_all$Pretty  <- factor(pdp_all$Pretty,  levels = panel_order)
pdp_mean$Pretty <- factor(pdp_mean$Pretty, levels = panel_order)

p_core <- ggplot() +
  geom_line(data = pdp_all, aes(x = feat_value, y = pd_cases, group = interaction(Pretty, bootstrap_id)), color = "grey85", alpha = 0.8, linewidth = 0.3) +
  geom_line(data = pdp_mean, aes(x = feat_value, y = pd_cases_mean, group = Pretty), color = "orange3", linewidth = 1.6) +
  facet_wrap(~Pretty, ncol = 2, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(strip.text = element_text(size = 16),
        axis.title.x = element_text(size = 18),
        axis.text.x  = element_text(size = 14),
        axis.text.y  = element_text(size = 14),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(t = 5, r = 5, b = 5, l = 35))

ggdraw(p_core) + draw_label("Predicted cases", x = 0.02, y = 0.5, angle = 90, vjust = 0.5, size = 16)


#### 9c. Supplementary Figure S14: partial dependence plots for additional predictors #######

vars <- c("pct_construction_extraction", "pct_farming_fishing_forestry",
          "pct_uninsured_u65","pct_age_65plus","pct_black","pct_adults_below_poverty","Diabetes_Per",
          "median_hh_income", "mean_dust_ugm3_lag_1",
          "pdsi", "SoilWaterPercent_30cm", "SoilOrgC_30cm")

pretty_map <- c("PopEstimate" = "Population size",
                "pct_construction_extraction" = "% construction/exctraction",
                "pct_farming_fishing_forestry" = "% farm/fish/forestry",
                "pct_uninsured_u65" = "% uninsured <65 yrs",
                "pct_age_65plus" = "% >65 years age",
                "pct_black" = "% population Black race/eth.",
                "pct_adults_below_poverty" = "% adults below poverty line",
                "Diabetes_Per" = "Diabetes prevalence",
                "median_hh_income" = "Median household income",
                "mean_dust_ugm3_lag_1" = "Mean dust conc. (1-mo lag)",
                "pdsi" = "PDSI (drought)",
                "SoilWaterPercent_30cm" = "Soil water %",
                "SoilOrgC_30cm" = "Soil organic carbon")

pdp_all <- bind_rows(lapply(vars, function(v) {
  fvals <- X_all[[v]]
  grid_vals <- seq(quantile(fvals, 0.05, na.rm = TRUE), quantile(fvals, 0.95, na.rm = TRUE), length.out = 30)
  pdp_list <- lapply(boot_ids, function(b) {
    model_path <- file.path(out_dir, sprintf("final_xgb_model_bootstrap_%d_Main.json", b))
    df <- compute_pdp_one_model(model_path, v, grid_vals, X_all = as.matrix(X_all), pop_all = pop_vec, n_bg = 2000)
    df$bootstrap_id <- b; df})
  bind_rows(pdp_list)}))

pdp_mean <- pdp_all %>% group_by(Feature, feat_value) %>%
  summarise(pd_cases_mean = mean(pd_cases, na.rm = TRUE), .groups = "drop")

pdp_all <- pdp_all %>% mutate(Pretty = ifelse(Feature %in% names(pretty_map), unname(pretty_map[Feature]), Feature))
pdp_mean <- pdp_mean %>% mutate(Pretty = ifelse(Feature %in% names(pretty_map), unname(pretty_map[Feature]), Feature))
pdp_all$Pretty <- factor(pdp_all$Pretty, levels = unname(pretty_map[vars]))
pdp_mean$Pretty <- factor(pdp_mean$Pretty, levels = unname(pretty_map[vars]))

p_core <- ggplot() +
  geom_line(data = pdp_all, aes(x = feat_value, y = pd_cases, group = interaction(Pretty, bootstrap_id)), color = "grey85", alpha = 0.8, linewidth = 0.3) +
  geom_line(data = pdp_mean, aes(x = feat_value, y = pd_cases_mean, group = Pretty), color = "orange3", linewidth = 1.6) +
  facet_wrap(~Pretty, ncol = 3, scales = "free") +
  labs(x = NULL, y = NULL) +
  theme_minimal() +
  theme(strip.text = element_text(size = 14),
        axis.text.x  = element_text(size = 12),
        axis.text.y  = element_text(size = 12),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(t = 5, r = 5, b = 5, l = 35))

ggdraw(p_core) + draw_label("Predicted cases", x = 0.02, y = 0.5, angle = 90, vjust = 0.5, size = 16)


#### 9d. Supplementary Figure S15: SHAP importance under alternative specifications ##############

top_n <- 12

shap_summary_alt1 <- fread(file.path(out_dir, "shap_importance_summary_AltSpec_NoSkinTest.csv"))
shap_top_alt1 <- shap_summary_alt1 %>%
  arrange(desc(mean_mean_abs_shap)) %>%
  slice_head(n = top_n) %>%
  mutate(Feature = fct_reorder(Feature, mean_mean_abs_shap))

shap_top_alt1 <- shap_top_alt1 %>% mutate(Feature =
                                            {recode_map <- c("Coccidioidin reactivity"="SkinTest_PercentReactors_Numeric",
                                                             "Mean annual precip."="mean_annual_precip","Year"="YEAR",
                                                             "Canine incidence"="Monthly_Dog_Inc_lag_0",
                                                             "# positive canine tests"="Total_Pos_Dog_Tests_lag_0",
                                                             "# canine tests"="Total_Dog_Tests_lag_0",
                                                             "EHR incidence"="EHRInc_wt_downscale","Elevation"="elev_m",
                                                             "% 65+"="pct_age_65plus","Mean annual temp."="mean_annual_temp",
                                                             "Soil organic carbon %"="SoilOrgC_10cm","Diabetes prevalence"="Diabetes_Per",
                                                             "% farm/fish/forestry"="pct_farming_fishing_forestry",
                                                             "Mean dust conc. (1-mo lag)" = "mean_dust_ugm3_lag_1",
                                                             "Max temp. (15-mo lag)" = "tmax_lag_15",
                                                             "Soil organic carbon"="SoilOrgC_30cm",
                                                             "% construction/extraction"="pct_construction_extraction",
                                                             "Soil water %"="SoilWaterPercent_30cm");
                                            do.call(forcats::fct_recode, c(list(Feature), as.list(recode_map[recode_map %in% levels(Feature)])))})

shap_summary_alt2 <- fread(file.path(out_dir, "shap_importance_summary_AltSpec_NoVetEpi.csv"))
shap_top_alt2 <- shap_summary_alt2 %>%
  arrange(desc(mean_mean_abs_shap)) %>%
  slice_head(n = top_n) %>%
  mutate(Feature = fct_reorder(Feature, mean_mean_abs_shap))

shap_top_alt2 <- shap_top_alt2 %>% mutate(Feature =
                                            {recode_map <- c("Coccidioidin reactivity"="SkinTest_PercentReactors_Numeric",
                                                             "Mean annual precip."="mean_annual_precip","Year"="YEAR",
                                                             "Elevation"="elev_m", "Soil organic carbon"="SoilOrgC_30cm",
                                                             "% construction/extraction"="pct_construction_extraction",
                                                             "Soil water %"="SoilWaterPercent_30cm",
                                                             "% 65+"="pct_age_65plus","Mean annual temp."="mean_annual_temp",
                                                             "Soil organic carbon %"="SoilOrgC_10cm","Diabetes prevalence"="Diabetes_Per",
                                                             "% farm/fish/forestry"="pct_farming_fishing_forestry",
                                                             "Max temp. (15-mo lag)"="tmax_lag_15",
                                                             "Mean dust conc. (1-mo lag)" = "mean_dust_ugm3_lag_1");
                                            do.call(forcats::fct_recode, c(list(Feature), as.list(recode_map[recode_map %in% levels(Feature)])))})

# Shared x-axis limit across both specs
xmax <- max(shap_top_alt1$q975_mean_abs_shap, shap_top_alt2$q975_mean_abs_shap, na.rm = TRUE)

p1 <- ggplot(shap_top_alt1, aes(x = mean_mean_abs_shap, y = Feature)) +
  geom_errorbarh(aes(xmin = q025_mean_abs_shap, xmax = q975_mean_abs_shap), height = 0) +
  geom_point(size = 2.6) +
  labs(title = "No coccidioidin reactivity", x = "Mean |SHAP| (mean ± 95% CI)", y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 15),
        plot.title = element_text(hjust = 0.5, size = 16),        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(t = 5.5, r = 18, b = 5.5, l = 18)) +
  scale_x_continuous(limits = c(0, xmax * 1.05))

p2 <- ggplot(shap_top_alt2, aes(x = mean_mean_abs_shap, y = Feature)) +
  geom_errorbarh(aes(xmin = q025_mean_abs_shap, xmax = q975_mean_abs_shap), height = 0) +
  geom_point(size = 2.6) +
  labs(title = "No EHR/veterinary predictors", x = "Mean |SHAP| (mean ± 95% CI)", y = NULL) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 14),
        axis.text.x = element_text(size = 14),
        axis.title.x = element_text(size = 15),
        plot.title = element_text(hjust = 0.7, size = 16),
        panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
        plot.margin = margin(t = 5.5, r = 18, b = 5.5, l = 18)) +
  scale_x_continuous(limits = c(0, xmax * 1.05))

title_grob <- ggdraw() + draw_label("Feature contribution to predictions",
                                    fontface = "bold", hjust = 0.25, size = 17)
panel_row <- plot_grid(p1, p2, ncol = 2, align = "h")

plot_grid(title_grob, panel_row, ncol = 1, rel_heights = c(0.08, 1))


################################################################################
#################### Apply the model (main) ####################################
################################################################################
#### 10. Predict monthly cases for all U.S. counties #########################

countiesraw <- fread(county_file)
countiesraw <- countiesraw %>% select(-c("Enhanced_Inc"))

# Retain a single (30cm) depth for each soil variable
countiesraw <- countiesraw %>% select(-c("SoilOrgC_0cm", "SoilOrgC_10cm", "SoilWaterPercent_0cm", "SoilWaterPercent_10cm",
                                         "SoilTexture_0cm", "SoilTexture_10cm"))
countiesraw <- countiesraw %>% filter(!is.na(PopEstimate))

countiesraw$SoilTexture_30cm <- as.factor(countiesraw$SoilTexture_30cm)
countiesraw$modal_land <- as.factor(countiesraw$modal_land)
countiesraw$MONTH <- as.factor(countiesraw$MONTH)

countiesraw$SkinTest_PercentReactors_Numeric <- str_extract_all(countiesraw$SkinTest_PercentReactors, "\\d+") %>%
  lapply(function(x) as.numeric(x)) %>%
  sapply(function(x) mean(x))
countiesraw <- countiesraw %>% select(-"SkinTest_PercentReactors")

counties_model_All <- mltools::one_hot(as.data.table(countiesraw)) %>% as.data.frame()
meta_counties <- countiesraw[, c("FIPS","YEAR","MONTH","PopEstimate", "cases")]
counties_model <- counties_model_All %>% select(-c("FIPS","cases","PopEstimate"))

# Align county columns to the training features
align_to_train <- function(df, feat_names){
  miss <- setdiff(feat_names, names(df)); if(length(miss)) df[miss] <- 0
  extra <- setdiff(names(df), feat_names); if(length(extra)) df <- df[setdiff(names(df), extra)]
  df <- df[feat_names]
  m <- as.matrix(df); storage.mode(m) <- "double"; m}

train_feat_names <- fread(file.path(out_dir, "TrainingFeatureNames_Main.csv"))[-1, V2]

counties_model_x <- align_to_train(counties_model, train_feat_names)
dcounties_model <- xgb.DMatrix(as.matrix(counties_model_x))

model_files <- list.files(out_dir, pattern = "final_xgb_model_bootstrap_\\d+_Main\\.json$", full.names = TRUE)

# Predicted incidence -> cases, one column per bootstrap model
pred_mat <- sapply(model_files, function(f) {
  bst <- xgb.load(f)
  pred_inc <- predict(bst, dcounties_model)
  pred_inc * (countiesraw$PopEstimate / 100000)
}, simplify = "matrix")

counties_pred_mean <- rowMeans(pred_mat, na.rm = TRUE)
counties_pred_lwr  <- apply(pred_mat, 1, quantile, probs = 0.025, na.rm = TRUE)
counties_pred_upr  <- apply(pred_mat, 1, quantile, probs = 0.975, na.rm = TRUE)

# Larger aggregations are summed within each bootstrap model, then quantiled across models
key_cy <- as.data.table(meta_counties)[, .(FIPS, YEAR)]

annual_by_boot <- rbindlist(lapply(seq_len(ncol(pred_mat)), function(b) {
  data.table(key_cy, cases = pred_mat[, b])[, .(cases = sum(cases, na.rm = TRUE)),
                                            by = .(FIPS, YEAR)][, boot := b][]
}))

countyYear_ci <- annual_by_boot[, .(Annual_Pred_Cases     = mean(cases),
                                    Annual_Pred_Cases_Lwr = quantile(cases, 0.025),
                                    Annual_Pred_Cases_Upr = quantile(cases, 0.975)),
                                by = .(FIPS, YEAR)]

fwrite(countyYear_ci, file.path(pred_out, "CountyYearPredictions_BootstrapCI_Main.csv"))
fwrite(annual_by_boot, file.path(pred_out, "CountyYearByBootstrap_Main.csv"))

key_sm <- as.data.table(meta_counties)[, .(STATEFP = substr(sprintf("%05d", FIPS), 1, 2),
                                           YEAR, MONTH = as.integer(as.character(MONTH)))]
stateMonth_by_boot <- rbindlist(lapply(seq_len(ncol(pred_mat)), function(b) {
  data.table(key_sm, cases = pred_mat[, b])[, .(cases = sum(cases, na.rm = TRUE)),
                                            by = .(STATEFP, YEAR, MONTH)][, boot := b][]
}))
fwrite(stateMonth_by_boot, file.path(pred_out, "StateMonthByBootstrap_Main.csv"))

counties_preds_df <- cbind.data.frame(
  meta_counties,
  counties_pred_cases = counties_pred_mean,
  pred_cases_lwr = counties_pred_lwr,
  pred_cases_upr = counties_pred_upr)

# Drop non-continental U.S.
counties_preds_df <- counties_preds_df %>%
  mutate(STATEFP = substr(sprintf("%05d", FIPS), 1, 2)) %>%
  filter(!STATEFP %in% c("02","15","60","66","69","72","78")) %>%
  select(-STATEFP)

fwrite(counties_preds_df, file.path(pred_out, "CountyPredictionsCompiled_Main.csv"))


#### 10b. Alternative specification 1: no coccidioidin reactivity #########################

countiesraw_alt1 <- fread(county_file) %>%
  select(-c("Enhanced_Inc", "SkinTest_PercentReactors", "SoilOrgC_0cm", "SoilOrgC_10cm",
            "SoilWaterPercent_0cm", "SoilWaterPercent_10cm", "SoilTexture_0cm", "SoilTexture_10cm")) %>%
  filter(!is.na(PopEstimate))

countiesraw_alt1$SoilTexture_30cm <- as.factor(countiesraw_alt1$SoilTexture_30cm)
countiesraw_alt1$modal_land <- as.factor(countiesraw_alt1$modal_land)
countiesraw_alt1$MONTH <- as.factor(countiesraw_alt1$MONTH)

counties_model_alt1 <- mltools::one_hot(as.data.table(countiesraw_alt1)) %>% as.data.frame() %>%
  select(-c("FIPS","cases","PopEstimate"))

train_feat_names_alt1 <- fread(file.path(out_dir, "TrainingFeatureNames_AltSpec_NoSkinTest.csv"))[-1, V2]
dcounties_model_alt1 <- xgb.DMatrix(align_to_train(counties_model_alt1, train_feat_names_alt1))

model_files_alt1 <- list.files(out_dir, pattern = "final_xgb_model_bootstrap_\\d+_AltSpec_NoSkinTest\\.json$", full.names = TRUE)

pred_mat_alt1 <- sapply(model_files_alt1, function(f) {
  bst <- xgb.load(f)
  predict(bst, dcounties_model_alt1) * (countiesraw_alt1$PopEstimate / 100000)
}, simplify = "matrix")

counties_preds_df_alt1 <- cbind.data.frame(
  countiesraw_alt1[, c("FIPS","YEAR","MONTH","PopEstimate", "cases")],
  counties_pred_cases = rowMeans(pred_mat_alt1, na.rm = TRUE)) %>%
  mutate(STATEFP = substr(sprintf("%05d", FIPS), 1, 2)) %>%
  filter(!STATEFP %in% c("02","15","60","66","69","72","78")) %>%
  select(-STATEFP)

fwrite(counties_preds_df_alt1, file.path(pred_out, "CountyPredictionsCompiled_AltSpec_NoSkinTest.csv"))


#### 10c. Alternative specification 2: no EHR or canine serology ###########################

countiesraw_alt2 <- fread(county_file) %>%
  select(-c("Enhanced_Inc", "SoilOrgC_0cm", "SoilOrgC_10cm", "SoilWaterPercent_0cm", "SoilWaterPercent_10cm",
            "SoilTexture_0cm", "SoilTexture_10cm",
            "Monthly_Dog_Inc_lag_0", "Total_Pos_Dog_Tests_lag_0", "Total_Dog_Tests_lag_0", "EHRInc_wt_downscale")) %>%
  filter(!is.na(PopEstimate))

countiesraw_alt2$SoilTexture_30cm <- as.factor(countiesraw_alt2$SoilTexture_30cm)
countiesraw_alt2$modal_land <- as.factor(countiesraw_alt2$modal_land)
countiesraw_alt2$MONTH <- as.factor(countiesraw_alt2$MONTH)

countiesraw_alt2$SkinTest_PercentReactors_Numeric <- str_extract_all(countiesraw_alt2$SkinTest_PercentReactors, "\\d+") %>%
  lapply(function(x) as.numeric(x)) %>% sapply(function(x) mean(x))
countiesraw_alt2 <- countiesraw_alt2 %>% select(-"SkinTest_PercentReactors")

counties_model_alt2 <- mltools::one_hot(as.data.table(countiesraw_alt2)) %>% as.data.frame() %>%
  select(-c("FIPS","cases","PopEstimate"))

train_feat_names_alt2 <- fread(file.path(out_dir, "TrainingFeatureNames_AltSpec_NoVetEpi.csv"))[-1, V2]
dcounties_model_alt2 <- xgb.DMatrix(align_to_train(counties_model_alt2, train_feat_names_alt2))

model_files_alt2 <- list.files(out_dir, pattern = "final_xgb_model_bootstrap_\\d+_AltSpec_NoVetEpi\\.json$", full.names = TRUE)

pred_mat_alt2 <- sapply(model_files_alt2, function(f) {
  bst <- xgb.load(f)
  predict(bst, dcounties_model_alt2) * (countiesraw_alt2$PopEstimate / 100000)
}, simplify = "matrix")

counties_preds_df_alt2 <- cbind.data.frame(
  countiesraw_alt2[, c("FIPS","YEAR","MONTH","PopEstimate", "cases")],
  counties_pred_cases = rowMeans(pred_mat_alt2, na.rm = TRUE)) %>%
  mutate(STATEFP = substr(sprintf("%05d", FIPS), 1, 2)) %>%
  filter(!STATEFP %in% c("02","15","60","66","69","72","78")) %>%
  select(-STATEFP)

fwrite(counties_preds_df_alt2, file.path(pred_out, "CountyPredictionsCompiled_AltSpec_NoVetEpi.csv"))

################################################################################
############# Plotting and downstream analysis #################################
################################################################################
##### 11. Figure 1a: county-level mean annual incidence #####

county_month_preds_all <- fread(file.path(pred_out, "CountyPredictionsCompiled_Main.csv"))
county_month_preds_all <- county_month_preds_all %>% mutate(FIPS = as.integer(FIPS))

countiesAll_annual <- county_month_preds_all %>%
  group_by(FIPS, YEAR) %>%
  summarise(Annual_Pop = first(PopEstimate), .groups = "drop")

countyYear_ci <- fread(file.path(pred_out, "CountyYearPredictions_BootstrapCI_Main.csv"))

countiesAll_annual <- countiesAll_annual %>%
  left_join(countyYear_ci %>% mutate(FIPS = as.integer(FIPS)), by = c("FIPS","YEAR")) %>%
  mutate(
    Annual_Pred_Inc_per100k = ifelse(Annual_Pop > 0, 1e5 * Annual_Pred_Cases / Annual_Pop, NA_real_),
    Annual_Lwr_Inc_per100k  = ifelse(Annual_Pop > 0, 1e5 * Annual_Pred_Cases_Lwr  / Annual_Pop, NA_real_),
    Annual_Upr_Inc_per100k  = ifelse(Annual_Pop > 0, 1e5 * Annual_Pred_Cases_Upr  / Annual_Pop, NA_real_))

countiesAll_annual <- countiesAll_annual %>%
  mutate(STATEFP = sprintf("%02d", as.integer(FIPS %/% 1000))) %>%
  relocate(STATEFP, .before = FIPS)

# Multi-year mean per county
AllModelMeans <- countiesAll_annual %>%
  group_by(FIPS) %>%
  summarise(cases_mean = mean(Annual_Pred_Cases, na.rm = TRUE),
            cases_lwr  = mean(Annual_Pred_Cases_Lwr,  na.rm = TRUE),
            cases_upr  = mean(Annual_Pred_Cases_Upr,  na.rm = TRUE),
            inc_mean = mean(Annual_Pred_Inc_per100k, na.rm = TRUE),
            inc_lwr  = mean(Annual_Lwr_Inc_per100k,  na.rm = TRUE),
            inc_upr  = mean(Annual_Upr_Inc_per100k,  na.rm = TRUE),
            n_years  = n(), .groups = "drop")

counties <- tigris::counties(cb=TRUE, year=2020) %>%
  filter(!STATEFP %in% c("02","15","60","66","69","72","78")) # drop AK, HI, territories
states  <- tigris::states(cb=TRUE, year=2020) %>%
  filter(!STUSPS %in% c("AK","HI","PR","GU","VI","MP","AS"))
states  <- st_transform(states, st_crs(counties))

counties <- counties %>% mutate(FIPS = sprintf("%02s%03s", STATEFP, COUNTYFP))
AllModelMeans <- AllModelMeans %>%  mutate(FIPS = sprintf("%05s", as.character(FIPS)))

AllModelMeansPlotdf <- counties %>% left_join(AllModelMeans, by="FIPS")

# Cap incidence at 30 for visualization
AllModelMeansPlotdf <- AllModelMeansPlotdf %>%
  mutate(inc_cap = pmin(inc_mean, 30))

ggplot() +
  geom_sf(data = AllModelMeansPlotdf, aes(fill = inc_cap), color = NA) +
  geom_sf(data = states, fill = NA, color = "gray20", linewidth = 0.16) +
  scale_fill_gradientn(colors = c("white", "#fff7ec", "#fdd49e", "#fdbb84", "#d7301f", "#7f0000"),
                       values = scales::rescale(c(0, 0.4, 5, 10, 20, 30)),
                       limits = c(0, 30), breaks = c(0, 5, 10, 20, 30),
                       labels = c("0", "5", "10", "20", "≥30"),
                       oob = scales::squish, na.value = "gray95",
                       name = "Mean annual incidence\nper 100,000") +
  guides(fill = guide_colorbar(barheight = unit(55, "mm"), barwidth = unit(4, "mm"),
                               title.position = "top", title.hjust = 0)) +
  coord_sf(crs = 5070, datum = NA) +
  theme_void() +
  theme(legend.position = "right",
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 9),
        plot.margin = margin(4, 4, 4, 4))


##### 11b. Supplementary Figure S6: relative uncertainty in predicted annual incidence ######

AllModelMeansUnc <- AllModelMeans %>%
  mutate(rel_width = ifelse(inc_mean > 0, (inc_upr - inc_lwr) / inc_mean, NA_real_))

# Cap at the 99th percentile to avoid blowups in very-low-incidence counties
cap_val <- quantile(AllModelMeansUnc$rel_width, 0.99, na.rm = TRUE)
AllModelMeansUnc <- AllModelMeansUnc %>% mutate(rel_width_cap = pmin(rel_width, cap_val))
AllModelMeansUncPlotdf <- counties %>% left_join(AllModelMeansUnc, by = "FIPS")

ggplot(AllModelMeansUncPlotdf) +
  geom_sf(aes(fill = rel_width_cap), color = NA) +
  geom_sf(data = states, fill = NA, color = "black", size = 0.01) +
  scale_fill_gradientn(colors = c("#fff5f5", "#f2e9e4", "#da9f93", "#b6465f", alpha("#890620", 0.7)),
                       na.value = "white") +
  theme_void() +
  coord_sf(crs = 5070, datum = NA) +
  theme(panel.grid = element_blank(),
        plot.title = element_text(hjust = 0.5, size = 16),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 12)) +
  labs(title = "Relative uncertainty in predicted annual incidence",
       fill = "CI width /\nmean prediction")


##### 11c. Figure 1b: high, moderate, low incidence counties (CDPH definition) ######

# Low (<2), moderate (2-9), high (10+) cases per 100,000 on average (MMWR 69(42))
countiesAll_annual_cats <- countiesAll_annual %>%
  arrange(FIPS, YEAR) %>%
  group_by(FIPS) %>%
  mutate(mean_inc = mean(Annual_Pred_Inc_per100k, na.rm = TRUE),
         HighInc_CDPH = mean_inc >= 10,
         ModInc_CDPH  = !HighInc_CDPH & mean_inc >= 2 & mean_inc < 10,
         CDPHInc_Category = case_when(HighInc_CDPH ~ "high",
                                      ModInc_CDPH  ~ "moderate",
                                      TRUE        ~ "low")) %>% ungroup()

CDPH_flag <- countiesAll_annual_cats %>%
  select(FIPS, CDPHInc_Category) %>%
  distinct() %>%
  mutate(FIPS = sprintf("%05d", FIPS))

non_reporting_statefp <- c("01","02", "09","12","13","15","16","17","19","23","25","28",
                           "34","36","37","40","42","45","47","48","50","51","54")

highinc <- countiesAll_annual_cats %>% filter(CDPHInc_Category == "high") %>% distinct(FIPS, .keep_all = TRUE)
table(highinc$CDPHInc_Category, highinc$STATEFP)

highinc_nonreps <- highinc %>% filter(STATEFP %in% non_reporting_statefp)
sum(table(highinc_nonreps$CDPHInc_Category, highinc_nonreps$STATEFP))
table(highinc_nonreps$CDPHInc_Category, highinc_nonreps$STATEFP)

medinc <- countiesAll_annual_cats %>% filter(CDPHInc_Category == "moderate") %>% distinct(FIPS, .keep_all = TRUE)
table(medinc$CDPHInc_Category, medinc$STATEFP)

medinc_nonreps <- medinc %>% filter(STATEFP %in% non_reporting_statefp)
sum(table(medinc_nonreps$CDPHInc_Category, medinc_nonreps$STATEFP))
table(medinc_nonreps$CDPHInc_Category, medinc_nonreps$STATEFP)

reporting_states <- c("AR","AZ", "CA", "CO","DE","IN","KS","KY","LA","MD","MI","MN",
  "MO","MT","NE","NV","NH","NM","ND","OH","OR","RI","SD","UT",
  "WA","WI","WY","DC")

states <- states(cb = TRUE, year = 2022, class = "sf") %>%
  filter(!STUSPS %in% c("PR","VI","GU","MP","AS","AK","HI")) %>%
  mutate(reporting = STUSPS %in% reporting_states) %>%
  st_transform(5070)

state_pts <- states %>% filter(!reporting) %>% st_point_on_surface()

ggplot(AllModelMeansPlotdf %>% left_join(CDPH_flag, by = "FIPS")) +
  geom_sf(aes(fill = CDPHInc_Category), color = NA) +
  geom_sf(data = states, fill = NA, color = "black", size = 0.01) +
  geom_sf(data = state_pts, color = "black", size = 3, shape = 16) +
  scale_fill_manual(values = c("low" = "#fdfbfc",
    "moderate" = alpha("#d5bdaf", 0.5),"high" = "#890620"),
  breaks = c("high", "moderate", "low"),
  na.value = "white", name = "Incidence\ncategory") +
  labs(title = "‘High’ and ‘moderate’ incidence counties based on model predictions") +
  theme_minimal() +
  coord_sf(crs = 5070, datum = NA) +
  theme(panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        legend.position = c(0.9, 0.2),
        legend.background = element_rect(fill = alpha("white", 0.7), color = NA),
        legend.key.height = unit(0.7, "cm"),
        legend.key.width = unit(0.7, "cm"),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.title = element_text(size = 14, hjust = 0.5),
        plot.title.position = "plot")


##### 11d. Supplementary Figure S8: high-incidence states ######

# CDC: high-incidence states have >=10 cases per 100,000 for 3+ consecutive years
state_annual <- countiesAll_annual %>%
  mutate(FIPS_chr = str_pad(as.character(FIPS), 5, pad = "0"),
         STATEFP = substr(FIPS_chr, 1, 2)) %>%
  group_by(STATEFP, YEAR) %>%
  summarise(State_Pred_Cases = sum(Annual_Pred_Cases, na.rm = TRUE),
            State_Pop        = sum(Annual_Pop,        na.rm = TRUE),
            State_Inc_per100k     = (State_Pred_Cases / State_Pop) * 100000,
            .groups = "drop") %>%
  arrange(STATEFP, YEAR) %>%
  group_by(STATEFP) %>%
  mutate(high_year = State_Inc_per100k >= 10,
         run_id = cumsum(high_year != lag(high_year, default = FALSE))) %>%
  group_by(STATEFP, run_id) %>%
  mutate(run_length = sum(high_year)) %>%
  group_by(STATEFP) %>%
  summarise(HighInc_CDC             = any(high_year & run_length >= 3),
            Mean_State_Inc_per100k  = mean(State_Inc_per100k,    na.rm = TRUE),
            Mean_State_Pred_Cases   = mean(State_Pred_Cases,      na.rm = TRUE),
            Total_State_Pred_Cases  = sum(State_Pred_Cases,       na.rm = TRUE),
            .groups = "drop")

state_annual$STATEFP[which(state_annual$HighInc_CDC == TRUE)]

states_sf <- tigris::states(cb = TRUE, year = 2020) %>%
  st_as_sf() %>%
  filter(!STUSPS %in% c("AK", "HI", "PR", "VI", "GU", "MP", "AS"))

state_map_df <- states_sf %>%
  left_join(state_annual, by = c("GEOID" = "STATEFP"))

states_highlight <- state_map_df %>% filter(STUSPS %in% c("CA", "AZ", "NV", "NM"))
states_other <- state_map_df %>% filter(!STUSPS %in% states_highlight)

states_highlight_5070 <- states_highlight %>% st_transform(5070)
states_other_5070 <- states_other %>% st_transform(5070)

ggplot() +
  geom_sf(data = states_other_5070,aes(fill = Mean_State_Inc_per100k),color = "black",linewidth = 0.16) +
  geom_sf(data = states_highlight_5070,aes(fill = Mean_State_Inc_per100k),color = "#df2935",linewidth = 0.6) +
  scale_fill_gradientn(colors = c("white", alpha("#eae2b7", 0.9),"#fee08b", "#f46d43", "#d53e4f", "#9e0142"),
                       oob = scales::squish,na.value = "white", limits = c(0,30), name = "Incidence\n(capped at 30)") +
  coord_sf() + theme_minimal() +
  labs(title = "State-level mean annual incidence (cases/100K) averaged across 2013-2023") +
  theme(panel.grid = element_blank(), axis.text = element_blank(),axis.ticks = element_blank(),
        axis.title = element_blank(), legend.position = c(0.93, 0.25),
        legend.background = element_rect(fill = alpha("white", 0.1), color = NA),
        legend.key.height = unit(0.6, "cm"), legend.key.width  = unit(0.6, "cm"),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 11),
        plot.title = element_text(size = 15, hjust = 0.5),
        plot.title.position = "plot")

##### 11e. Supplementary Figure S9: western Texas ########

# West Texas and Rio Grande regions (https://en.wikipedia.org/wiki/Geography_of_Texas)
west_tx_counties <- c(
  "Midland","Ector","Tom Green","Howard","Gaines","Andrews","Pecos","Reeves",
  "Dawson","Ward","McCulloch","Winkler","Martin","Crane","Kimble","Mason",
  "Coke","Concho","Sutton","Reagan","Upton","Crockett","Schleicher","Menard",
  "Irion","Sterling","Glasscock","Terrell","Borden","Loving","El Paso","Brewster",
  "Presidio","Hudspeth","Culberson","Jeff Davis")

tx_county_lookup <- tigris::counties(state = "TX", cb = TRUE, year = 2020) %>%
  st_as_sf() %>% st_drop_geometry() %>% transmute(COUNTY_NAME = NAME, FIPS = as.integer(GEOID))

west_tx_fips <- tx_county_lookup %>% filter(COUNTY_NAME %in% west_tx_counties) %>% pull(FIPS)

westTexpop <- countiesAll_annual %>% filter(FIPS %in% west_tx_fips) %>%
  group_by(YEAR) %>% summarise(Region_Pop = sum(Annual_Pop, na.rm = TRUE), .groups = "drop")
mean(westTexpop$Region_Pop)

# Sum across the 36 counties within each bootstrap, then take quantiles
cyb_wtx <- fread(file.path(pred_out, "CountyYearByBootstrap_Main.csv"))
cyb_wtx <- cyb_wtx[FIPS %in% west_tx_fips]

region_by_boot <- cyb_wtx[, .(Region_Pred_Cases = sum(cases, na.rm = TRUE)), by = .(YEAR, boot)]
region_ci <- region_by_boot[, .(Region_Pred_Cases     = mean(Region_Pred_Cases),
                                Region_Pred_Cases_Lwr = quantile(Region_Pred_Cases, 0.025),
                                Region_Pred_Cases_Upr = quantile(Region_Pred_Cases, 0.975)),
                            by = YEAR]

west_tx_yearly <- westTexpop %>%
  left_join(region_ci, by = "YEAR") %>%
  mutate(
    Region_Inc_per100k     = (Region_Pred_Cases     / Region_Pop) * 100000,
    Region_Inc_Lwr_per100k = (Region_Pred_Cases_Lwr / Region_Pop) * 100000,
    Region_Inc_Upr_per100k = (Region_Pred_Cases_Upr / Region_Pop) * 100000) %>%
  arrange(YEAR) %>% mutate(
    high_year = Region_Inc_per100k >= 10,
    run_id = cumsum(high_year != lag(high_year, default = FALSE))) %>%
  group_by(run_id) %>% mutate(run_length = sum(high_year)) %>% ungroup()

tx_state_sf <- tigris::states(cb = TRUE, year = 2020) %>%  st_as_sf() %>% filter(STUSPS == "TX")
tx_counties_sf <- tigris::counties(state = "TX", cb = TRUE, year = 2020) %>%
  st_as_sf() %>% mutate(FIPS = as.integer(GEOID),is_west = FIPS %in% west_tx_fips)

ggplot() +
  geom_sf(data = tx_counties_sf %>% filter(!is_west), fill = NA, color = "grey90", linewidth = 0.25) +
  geom_sf(data = tx_counties_sf %>% filter(is_west),
          fill = alpha("#b23a48", 0.7), color = "black", linewidth = 0.5) +
  geom_sf(data = tx_state_sf, fill = NA, color = "black", linewidth = 0.6) +
  coord_sf() +theme_void()

ggplot(west_tx_yearly, aes(x = YEAR, y = Region_Inc_per100k)) +
  geom_ribbon(aes(ymin = Region_Inc_Lwr_per100k, ymax = Region_Inc_Upr_per100k),
              fill = "#b23a48", alpha = 0.08) +
  geom_line(color = "#b23a48", linewidth = 0.9) +
  geom_point(color = "#b23a48", size = 2.2) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(yintercept = 10, linetype = "dashed", color = alpha("black", 0.7)) +
  scale_x_continuous(breaks = 2013:2023, expand = expansion(mult = c(0, 0.02))) +
  scale_y_continuous(limits = c(0, NA), expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "cases per 100,000", title = "Annual incidence in West Texas region") +
  theme_bw(base_size = 13) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 17),
    panel.grid.major = element_line(linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.ticks = element_blank(),
    axis.line.y = element_line(color = "black"),
    strip.background = element_blank(),
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 17))

################################################################################
############# Alternative model specifications #################################
################################################################################
##### 12. Supplementary Figure S7: incidence under alternative specifications ####################
######## Alt spec 1: No coccidioidin reactivity #########

county_month_preds_all_alt1 <- fread(file.path(pred_out, "CountyPredictionsCompiled_AltSpec_NoSkinTest.csv"))
county_month_preds_all_alt1 <- county_month_preds_all_alt1 %>% mutate(FIPS = as.integer(FIPS))

countiesAll_annual_alt1 <- county_month_preds_all_alt1 %>%
  group_by(FIPS, YEAR) %>%
  summarise(Annual_Pred_Cases = sum(counties_pred_cases, na.rm = TRUE),
            Annual_Pop  = first(PopEstimate), .groups = "drop") %>%
  mutate(Annual_Pred_Inc_per100k = ifelse(Annual_Pop > 0, 1e5 * Annual_Pred_Cases / Annual_Pop, NA_real_),
         STATEFP = sprintf("%02d", as.integer(FIPS %/% 1000))) %>%
  relocate(STATEFP, .before = FIPS)

AllModelMeans_alt1 <- countiesAll_annual_alt1 %>%
  group_by(FIPS) %>%
  summarise(inc_mean = mean(Annual_Pred_Inc_per100k, na.rm = TRUE), .groups = "drop") %>%
  mutate(FIPS = sprintf("%05s", as.character(FIPS)))

counties <- tigris::counties(cb=TRUE, year=2020) %>%
  filter(!STATEFP %in% c("02","15","60","66","69","72","78")) # drop AK, HI, territories
states  <- tigris::states(cb=TRUE, year=2020) %>%
  filter(!STUSPS %in% c("AK","HI","PR","GU","VI","MP","AS"))

states  <- st_transform(states, st_crs(counties))
counties <- counties %>% mutate(FIPS = sprintf("%02s%03s", STATEFP, COUNTYFP))

AllModelMeansPlotdf_alt1 <- counties %>% left_join(AllModelMeans_alt1, by="FIPS") %>%
  mutate(inc_cap = pmin(inc_mean, 30))

AllModelMeansPlotdf_alt1_5070 <- AllModelMeansPlotdf_alt1 %>% st_transform(5070)
states_5070 <- states %>% st_transform(5070)

ggplot(AllModelMeansPlotdf_alt1_5070) +
  geom_sf(aes(fill = inc_cap), color = NA) +
  geom_sf(data = states_5070, fill = NA, color = "black", size = 0.04) +
  scale_fill_gradientn(colors = c("white", "#fff7ec", "#fdd49e", "#fdbb84", "#d7301f", "#7f0000"),
                       values = scales::rescale(c(0, 0.4, 5, 10, 20, 30)),
                       limits = c(0, 30), breaks = c(0, 5, 10, 20, 30),
                       labels = c("0", "5", "10", "20", "≥30"),
                       oob = scales::squish, na.value = "gray95") +
  theme_void() +
  theme(panel.grid = element_blank()) +
  labs(title = "Model without coccidioidin reactivity",
       fill = "Incidence\n(capped at 30)" )

######### Alt spec 2: No EHR or canine serology predictors ########

county_month_preds_all_alt2 <- fread(file.path(pred_out, "CountyPredictionsCompiled_AltSpec_NoVetEpi.csv"))
county_month_preds_all_alt2 <- county_month_preds_all_alt2 %>% mutate(FIPS = as.integer(FIPS))

countiesAll_annual_alt2 <- county_month_preds_all_alt2 %>%
  group_by(FIPS, YEAR) %>%
  summarise(Annual_Pred_Cases = sum(counties_pred_cases, na.rm = TRUE),
            Annual_Pop  = first(PopEstimate), .groups = "drop") %>%
  mutate(Annual_Pred_Inc_per100k = ifelse(Annual_Pop > 0, 1e5 * Annual_Pred_Cases / Annual_Pop, NA_real_),
         STATEFP = sprintf("%02d", as.integer(FIPS %/% 1000))) %>%
  relocate(STATEFP, .before = FIPS)

AllModelMeans_alt2 <- countiesAll_annual_alt2 %>%
  group_by(FIPS) %>%
  summarise(inc_mean = mean(Annual_Pred_Inc_per100k, na.rm = TRUE), .groups = "drop") %>%
  mutate(FIPS = sprintf("%05s", as.character(FIPS)))

AllModelMeansPlotdf_alt2 <- counties %>% left_join(AllModelMeans_alt2, by="FIPS") %>%
  mutate(inc_cap = pmin(inc_mean, 30))

AllModelMeansPlotdf_alt2_5070 <- AllModelMeansPlotdf_alt2 %>% st_transform(5070)

ggplot(AllModelMeansPlotdf_alt2_5070) +
  geom_sf(aes(fill = inc_cap), color = NA) +
  geom_sf(data = states_5070, fill = NA, color = "black", size = 0.04) +
  scale_fill_gradientn(colors = c("white", "#fff7ec", "#fdd49e", "#fdbb84", "#d7301f", "#7f0000"),
                       values = scales::rescale(c(0, 0.4, 5, 10, 20, 30)),
                       limits = c(0, 30), breaks = c(0, 5, 10, 20, 30),
                       labels = c("0", "5", "10", "20", "≥30"),
                       oob = scales::squish, na.value = "gray95") +
  theme_void() +
  theme(panel.grid = element_blank()) +
  labs(title = "Model without canine serology or EHR",
       fill = "Incidence\n(capped at 30)" )


##### 12b. Supplementary Table S3: high/moderate incidence counties and states under alt. specs ########

### Alt Spec 1: No skin test predictor
countiesAll_annual_cats_alt1 <- countiesAll_annual_alt1 %>%
  arrange(FIPS, YEAR) %>%  group_by(FIPS) %>%
  mutate(mean_inc = mean(Annual_Pred_Inc_per100k, na.rm = TRUE),
         HighInc_CDPH = mean_inc >= 10,
         ModInc_CDPH  = !HighInc_CDPH & mean_inc >= 2 & mean_inc < 10,
         CDPHInc_Category = case_when(HighInc_CDPH ~ "high", ModInc_CDPH  ~ "moderate",
                                      TRUE  ~ "low")) %>% ungroup()

highinc_alt1 <- countiesAll_annual_cats_alt1 %>% filter(CDPHInc_Category == "high") %>% distinct(FIPS, .keep_all = TRUE)
table(highinc_alt1$CDPHInc_Category, highinc_alt1$STATEFP)

medinc_alt1 <- countiesAll_annual_cats_alt1 %>% filter(CDPHInc_Category == "moderate") %>% distinct(FIPS, .keep_all = TRUE)
table(medinc_alt1$CDPHInc_Category, medinc_alt1$STATEFP)

#### Alt spec 2: No EHR/canine serology predictors
countiesAll_annual_cats_alt2 <- countiesAll_annual_alt2 %>%
  arrange(FIPS, YEAR) %>%  group_by(FIPS) %>%
  mutate(mean_inc = mean(Annual_Pred_Inc_per100k, na.rm = TRUE),
         HighInc_CDPH = mean_inc >= 10,
         ModInc_CDPH  = !HighInc_CDPH & mean_inc >= 2 & mean_inc < 10,
         CDPHInc_Category = case_when(HighInc_CDPH ~ "high", ModInc_CDPH  ~ "moderate",
                                      TRUE  ~ "low")) %>% ungroup()

highinc_alt2 <- countiesAll_annual_cats_alt2 %>% filter(CDPHInc_Category == "high") %>% distinct(FIPS, .keep_all = TRUE)
table(highinc_alt2$CDPHInc_Category, highinc_alt2$STATEFP)

medinc_alt2 <- countiesAll_annual_cats_alt2 %>% filter(CDPHInc_Category == "moderate") %>% distinct(FIPS, .keep_all = TRUE)
table(medinc_alt2$CDPHInc_Category, medinc_alt2$STATEFP)

#### High-incidence states, alt spec 1
state_annual_alt1 <- countiesAll_annual_alt1 %>%
  group_by(STATEFP, YEAR) %>%
  summarise(State_Pred_Cases = sum(Annual_Pred_Cases, na.rm = TRUE),
            State_Pop        = sum(Annual_Pop,        na.rm = TRUE),
            State_Inc_per100k     = (State_Pred_Cases / State_Pop) * 100000,
            .groups = "drop") %>%
  arrange(STATEFP, YEAR) %>%
  group_by(STATEFP) %>%
  mutate(high_year = State_Inc_per100k >= 10,
         run_id = cumsum(high_year != lag(high_year, default = FALSE))) %>%
  group_by(STATEFP, run_id) %>%
  mutate(run_length = sum(high_year)) %>%
  group_by(STATEFP) %>%
  summarise(HighInc_CDC = any(high_year & run_length >= 3), .groups = "drop")

state_annual_alt1$STATEFP[which(state_annual_alt1$HighInc_CDC == TRUE)]

#### High-incidence states, alt spec 2
state_annual_alt2 <- countiesAll_annual_alt2 %>%
  group_by(STATEFP, YEAR) %>%
  summarise(State_Pred_Cases = sum(Annual_Pred_Cases, na.rm = TRUE),
            State_Pop        = sum(Annual_Pop,        na.rm = TRUE),
            State_Inc_per100k     = (State_Pred_Cases / State_Pop) * 100000,
            .groups = "drop") %>%
  arrange(STATEFP, YEAR) %>%
  group_by(STATEFP) %>%
  mutate(high_year = State_Inc_per100k >= 10,
         run_id = cumsum(high_year != lag(high_year, default = FALSE))) %>%
  group_by(STATEFP, run_id) %>%
  mutate(run_length = sum(high_year)) %>%
  group_by(STATEFP) %>%
  summarise(HighInc_CDC = any(high_year & run_length >= 3), .groups = "drop")

state_annual_alt2$STATEFP[which(state_annual_alt2$HighInc_CDC == TRUE)]

################################################################################
############# Compare with NNDSS data ##########################################
################################################################################
########## 13. Set up #######

# Restrict to 2013-2022 to match NNDSS
counties_preds_monthly22 <- county_month_preds_all %>% filter(YEAR < 2023) %>%
  select(-cases) %>%
  mutate(Monthly_Pred_Inc_per100k = ifelse(PopEstimate > 0, 1e5 * counties_pred_cases / PopEstimate, NA_real_))

nndss <- fread(nndss_file)
nndss <- nndss %>% filter(YEAR >= 2013)
nndss <- nndss %>% mutate(FIPS = stringr::str_pad(as.character(FIPS), 5, pad = "0"))

non_reporting_statefp <- c("01","02","09","12","13","15","16","17","19","23","25","28",
                           "34","36","37","40","42","45","47","48","50","51","54")

# Counties missing from NNDSS: 0 in reporting states, NA in non-reporting states
combined_monthly <- counties_preds_monthly22 %>%
  mutate(FIPS = stringr::str_pad(as.character(FIPS), 5, pad = "0")) %>%
  left_join(nndss, by = c("FIPS", "YEAR", "MONTH")) %>%
  rename(NNDSS_cases = CaseCount,
         NNDSS_inc   = Inc,
         Predicted_cases      = counties_pred_cases,
         Predicted_cases_lwr  = pred_cases_lwr,
         Predicted_cases_upr  = pred_cases_upr,
         Predicted_inc        = Monthly_Pred_Inc_per100k) %>%
  mutate(STATEFP = substr(FIPS, 1, 2),
         NNDSS_cases = dplyr::case_when(
           STATEFP %in% non_reporting_statefp ~ NA_real_,
           is.na(NNDSS_cases) ~ 0, TRUE ~ as.numeric(NNDSS_cases)),
         NNDSS_inc = dplyr::case_when(
           STATEFP %in% non_reporting_statefp ~ NA_real_,
           is.na(NNDSS_inc) ~ 0, TRUE ~ as.numeric(NNDSS_inc))) %>%
  select(-"CountyPop")

annual_cases_df <- combined_monthly %>%
  group_by(FIPS, YEAR) %>%
  summarise(Annual_Predicted_cases = sum(Predicted_cases, na.rm = TRUE),
            Annual_NNDSS_cases = ifelse(all(is.na(NNDSS_cases)), NA_real_, sum(NNDSS_cases, na.rm = TRUE)),
            Annual_PopEstimate = max(PopEstimate, na.rm = TRUE),
            Annual_Pred_Inc_per100k = 1e5 * Annual_Predicted_cases / Annual_PopEstimate,
            Annual_NNDSS_Inc_per100k = ifelse(is.na(Annual_NNDSS_cases), NA_real_, 1e5 * Annual_NNDSS_cases / Annual_PopEstimate),
            .groups = "drop")

##### 13b. Supplementary Table S4: state-level cases and incidence #####

# State totals are summed within each bootstrap model, then quantiled
cyb <- fread(file.path(pred_out, "CountyYearByBootstrap_Main.csv"))
cyb <- cyb[YEAR >= 2013 & YEAR <= 2022]
cyb[, FIPS := stringr::str_pad(as.character(FIPS), 5, pad = "0")]

county_year_obs <- combined_monthly %>%
  filter(YEAR >= 2013, YEAR <= 2022) %>%
  group_by(STATEFP, FIPS, YEAR) %>%
  summarise(Pop = first(PopEstimate),
            NNDSS_cases = sum(NNDSS_cases, na.rm = TRUE),
            .groups = "drop") %>%
  as.data.table()

cy_boot <- merge(county_year_obs, cyb, by = c("FIPS","YEAR"), allow.cartesian = TRUE)

state_year_boot <- cy_boot[, .(State_Pop         = sum(Pop, na.rm = TRUE),
                               State_Pred_Cases  = sum(cases, na.rm = TRUE),
                               State_NNDSS_Cases = sum(NNDSS_cases, na.rm = TRUE)),
                           by = .(STATEFP, YEAR, boot)]
state_year_boot[, `:=`(State_Pred_Inc  = 1e5 * State_Pred_Cases  / State_Pop,
                       State_NNDSS_Inc = 1e5 * State_NNDSS_Cases / State_Pop)]

state_boot <- state_year_boot[, .(Pred_Cases  = mean(State_Pred_Cases),
                                  Pred_Inc    = mean(State_Pred_Inc),
                                  NNDSS_Cases = mean(State_NNDSS_Cases),
                                  NNDSS_Inc   = mean(State_NNDSS_Inc)),
                              by = .(STATEFP, boot)]

state_summary <- state_boot[, .(
  Mean_Annual_Pred_Cases     = mean(Pred_Cases),
  Mean_Annual_Pred_Cases_lwr = quantile(Pred_Cases, 0.025),
  Mean_Annual_Pred_Cases_upr = quantile(Pred_Cases, 0.975),
  Mean_Annual_Pred_Inc       = mean(Pred_Inc),
  Mean_Annual_Pred_Inc_lwr   = quantile(Pred_Inc, 0.025),
  Mean_Annual_Pred_Inc_upr   = quantile(Pred_Inc, 0.975),
  Mean_Annual_NNDSS_Cases    = mean(NNDSS_Cases),
  Mean_Annual_NNDSS_Inc      = mean(NNDSS_Inc)), by = STATEFP]

states_lookup <- tigris::states(cb = TRUE, year = 2020) %>% st_as_sf() %>%
  st_drop_geometry() %>% transmute(STATEFP = GEOID, State = NAME)

# * marks states without mandatory reporting
state_table <- state_summary %>%
  left_join(states_lookup, by = "STATEFP") %>%
  mutate(
    Pred_Mean_Annual_Cases_fmt = sprintf("%.1f [%.1f, %.1f]", Mean_Annual_Pred_Cases, Mean_Annual_Pred_Cases_lwr, Mean_Annual_Pred_Cases_upr),
    Pred_Mean_Annual_Inc_fmt   = sprintf("%.2f [%.2f, %.2f]", Mean_Annual_Pred_Inc,   Mean_Annual_Pred_Inc_lwr,   Mean_Annual_Pred_Inc_upr),
    NNDSS_Mean_Annual_Inc_fmt  = sprintf("%.2f", Mean_Annual_NNDSS_Inc),
    State = if_else(STATEFP %in% non_reporting_statefp, paste0(State, "*"), State)) %>%
  filter(!STATEFP %in% c("02","15")) %>% transmute(
    State,
    Pred_Mean_Annual_Cases        = Pred_Mean_Annual_Cases_fmt,
    NNDSS_Mean_Annual_Cases       = if_else(STATEFP %in% non_reporting_statefp, "NA", sprintf("%.1f", Mean_Annual_NNDSS_Cases)),
    Pred_Mean_Annual_Inc_per100k  = Pred_Mean_Annual_Inc_fmt,
    NNDSS_Mean_Annual_Inc_per100k = if_else(STATEFP %in% non_reporting_statefp, "NA", NNDSS_Mean_Annual_Inc_fmt)) %>%
  arrange(State)

fwrite(state_table, file.path(out_dir, "Burdens_PredictedAndReported_ByState.csv"))


###### 13c. Supplementary Figure S10: predicted vs NNDSS cases #######

ok <- stats::complete.cases(combined_monthly[, c("Predicted_cases", "NNDSS_cases")]) & !is.na(combined_monthly$STATEFP)
is_caz <- combined_monthly$STATEFP %in% c("06", "04")

r2_other <- cor(combined_monthly$Predicted_cases[ok & !is_caz],
                combined_monthly$NNDSS_cases[ok & !is_caz])^2
n_other  <- nrow(combined_monthly[ok & !is_caz, ])

r2_caz <- cor(combined_monthly$Predicted_cases[ok & is_caz],
              combined_monthly$NNDSS_cases[ok & is_caz])^2
n_caz  <- nrow(combined_monthly[ok & is_caz, ])

lab_caz <- bquote("AZ/CA  (" * R^2 == .(format(round(r2_caz, 3), nsmall = 3)) *
                    ", N = " * .(format(n_caz, big.mark = ",")) * ")")
lab_other <- bquote("Other states  (" * R^2 == .(format(round(r2_other, 3), nsmall = 3)) *
                      ", N = " * .(format(n_other, big.mark = ",")) * ")")

op <- par(no.readonly = TRUE)
par(mgp = c(2.5, 0.7, 0))
plot(Predicted_cases ~ NNDSS_cases, data = combined_monthly[ok, ],
     type = "n", ylab = "Predicted cases (county-month)",
     xlab = "Reported cases (county-month)", main = "Predicted vs NNDSS reported cases",
     cex.lab = 1.3, cex.main = 1.4, cex.axis = 1.1)
points(Predicted_cases ~ NNDSS_cases,
       data = combined_monthly[ok & is_caz, ],
       pch = 16, col = adjustcolor("#b23a48", alpha.f = 0.5))
points(Predicted_cases ~ NNDSS_cases,
       data = combined_monthly[ok & !is_caz, ], pch = 16, col = adjustcolor("black", alpha.f = 0.25))
abline(0, 1, lty = 2, col = "#2B76CC", lwd = 3)
abline(lm(Predicted_cases ~ NNDSS_cases, data = combined_monthly, subset = ok & !is_caz),
       col = alpha("black", 0.7),  lwd = 3)
abline(lm(Predicted_cases ~ NNDSS_cases, data = combined_monthly, subset = ok &  is_caz),
       col = alpha("#b23a48", 0.8), lwd = 3)
legend("topleft",legend = c(lab_caz, lab_other),inset = c(0.01, 0.01),pch = 16,
       col = c("#b23a48", adjustcolor("black", alpha.f = 0.8)),y.intersp = 1.2,
       bty = "o",bg = adjustcolor("white", alpha.f = 0.4))
par(op)


##### 13d. Under-ascertainment in reporting states (excluding AZ/CA) ####

statefp <- substr(annual_cases_df$FIPS, 1, 2)
ok_yr <- stats::complete.cases(annual_cases_df[, c("Annual_Predicted_cases", "Annual_NNDSS_cases")])
ok_other_yr <- ok_yr & !statefp %in% c("06", "04")
ok_caz_yr   <- ok_yr &  statefp %in% c("06", "04")

# % higher (mean of county-level differences) and additional cases per year
mean(annual_cases_df$Annual_Pred_Inc_per100k[ok_other_yr] - annual_cases_df$Annual_NNDSS_Inc_per100k[ok_other_yr], na.rm = TRUE) /
  mean(annual_cases_df$Annual_NNDSS_Inc_per100k[ok_other_yr], na.rm = TRUE)
sum(annual_cases_df$Annual_Predicted_cases[ok_other_yr] - annual_cases_df$Annual_NNDSS_cases[ok_other_yr]) / n_distinct(annual_cases_df$YEAR)
nrow(annual_cases_df[ok_other_yr, ])

# Bootstrap intervals: recompute both quantities within each bootstrap model
other_cy <- as.data.table(annual_cases_df[ok_other_yr, ])[, .(FIPS, YEAR, NNDSS = Annual_NNDSS_cases, Pop = Annual_PopEstimate)]
n_years <- uniqueN(other_cy$YEAR)
other_boot <- merge(other_cy, cyb, by = c("FIPS","YEAR"), allow.cartesian = TRUE)

under_by_boot <- other_boot[, .(extra_cases = sum(cases - NNDSS) / n_years,
                                pct_higher  = mean(1e5 * (cases - NNDSS) / Pop) / mean(1e5 * NNDSS / Pop)), by = boot]
under_by_boot[, .(extra_cases = mean(extra_cases), extra_cases_lwr = quantile(extra_cases, 0.025), extra_cases_upr = quantile(extra_cases, 0.975),
                  pct_higher = mean(pct_higher), pct_higher_lwr = quantile(pct_higher, 0.025), pct_higher_upr = quantile(pct_higher, 0.975))]

# CA / AZ
mean(annual_cases_df$Annual_Predicted_cases[ok_caz_yr] - annual_cases_df$Annual_NNDSS_cases[ok_caz_yr], na.rm = TRUE)
mean(annual_cases_df$Annual_Pred_Inc_per100k[ok_caz_yr] - annual_cases_df$Annual_NNDSS_Inc_per100k[ok_caz_yr], na.rm = TRUE)
nrow(annual_cases_df[ok_caz_yr, ])


####### 13e. Supplementary Figure S11: % over/underprediction by state ######

state_lookup <- tibble(
  STATEFP = c("01", "04", "05", "06", "08", "09", "10", "11", "12", "13",
              "16", "17", "18", "19", "20", "21", "22", "23", "24", "25",
              "26", "27", "28", "29", "30", "31", "32", "33", "34", "35",
              "36", "37", "38", "39", "40", "41", "42", "44", "45", "46",
              "47", "48", "49", "50", "51", "53", "54", "55", "56"),
  State = c("Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
            "Delaware", "District of Columbia", "Florida", "Georgia", "Idaho",
            "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
            "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
            "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
            "New Hampshire", "New Jersey", "New Mexico", "New York",
            "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
            "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
            "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
            "West Virginia", "Wisconsin", "Wyoming"))

state_bias_calc <- annual_cases_df %>%
  mutate(STATEFP = substr(FIPS, 1, 2)) %>%
  filter(!STATEFP %in% c("04", "06", non_reporting_statefp),
         !is.na(Annual_NNDSS_Inc_per100k),
         !is.na(Annual_Pred_Inc_per100k)) %>%
  group_by(STATEFP) %>%
  summarise(n_counties = n(),
            mean_pred_inc = mean(Annual_Pred_Inc_per100k, na.rm = TRUE),
            mean_nndss_inc = mean(Annual_NNDSS_Inc_per100k, na.rm = TRUE),
            mean_diff_inc = mean(Annual_Pred_Inc_per100k - Annual_NNDSS_Inc_per100k, na.rm = TRUE),
            rel_bias_inc = mean_diff_inc / mean_nndss_inc,
            rel_bias_inc_pct = 100 * rel_bias_inc,
            .groups = "drop")

state_bias_inc_other <- state_lookup %>%
  left_join(state_bias_calc, by = "STATEFP") %>%
  select(State, STATEFP, everything()) %>%
  arrange(State)

states_map <- tigris::states(cb = TRUE, year = 2020) %>%
  st_as_sf() %>%
  filter(!STUSPS %in% c("AK", "HI", "PR", "VI", "GU", "MP", "AS")) %>%
  transmute(STATEFP = GEOID, geometry) %>%
  left_join(state_bias_inc_other %>% select(STATEFP, rel_bias_inc_pct), by = "STATEFP") %>%
  mutate(rel_bias_inc_pct = if_else(!is.finite(rel_bias_inc_pct), NA_real_, rel_bias_inc_pct),
         hash_flag = STATEFP %in% c("04", "06", non_reporting_statefp) | is.na(rel_bias_inc_pct))

hash_states_5070 <- states_map %>% filter(hash_flag) %>% st_transform(5070)
fill_states_5070 <- states_map %>% filter(!hash_flag) %>% st_transform(5070)

ggplot() +
  geom_sf(data = fill_states_5070, aes(fill = rel_bias_inc_pct), color = "black", linewidth = 0.2) +
  ggpattern::geom_sf_pattern(data = hash_states_5070,
                             fill = "gray90", color = "black", linewidth = 0.2,
                             pattern = "stripe", pattern_fill = "gray90", pattern_colour = "gray70",
                             pattern_angle = 45, pattern_density = 0.35, pattern_spacing = 0.02,
                             show.legend = FALSE) +
  scale_fill_gradient2(name = "Over or under\nprediction(%)",
                       low = "#3B7FB6", mid = "white", high = "#C65B4B", midpoint = 0,
                       limits = c(-100, 350), oob = squish, na.value = "gray90") +
  theme_void() +
  labs(title = "State-level over vs underprediction in incidence relative to NNDSS") +
  theme(plot.title = element_text(size = 15, face = "bold", hjust = 0.8),
        legend.title = element_text(size = 11),
        legend.text = element_text(size = 10))


###### 13f. Figure 2: monthly time series for select states #######

states_to_plot <- c("04","06", "32", "35", "48")
state_name_map <- c("04" = "Arizona", "06" = "California",
                    "32" = "Nevada", "35" = "New Mexico", "48" = "Texas*")

# State-month interval from per-bootstrap state totals
smb_ci <- fread(file.path(pred_out, "StateMonthByBootstrap_Main.csv"))[
  , .(Predicted_cases_lwr = quantile(cases, 0.025), Predicted_cases_upr = quantile(cases, 0.975)),
  by = .(STATEFP = sprintf("%02d", as.integer(STATEFP)), YEAR, MONTH)]

state_monthly <- combined_monthly %>%
  filter(STATEFP %in% states_to_plot) %>%
  mutate(State = factor(state_name_map[STATEFP]),
         date = as.Date(paste(YEAR, MONTH, "01", sep = "-")),
         date = if_else(date == as.Date("2022-12-01"), as.Date("2023-01-01"), date)) %>%
  group_by(State, STATEFP, YEAR, MONTH, date) %>%
  summarise(PopEstimate = sum(PopEstimate, na.rm = TRUE),
            Predicted_cases = sum(Predicted_cases, na.rm = TRUE),
            NNDSS_cases = sum(NNDSS_cases, na.rm = TRUE),
            .groups = "drop") %>%
  left_join(smb_ci, by = c("STATEFP", "YEAR", "MONTH")) %>%
  mutate(Predicted_inc = Predicted_cases / PopEstimate * 100000,
         Predicted_inc_lwr = Predicted_cases_lwr / PopEstimate * 100000,
         Predicted_inc_upr = Predicted_cases_upr / PopEstimate * 100000,
         NNDSS_inc = NNDSS_cases / PopEstimate * 100000)

state_monthly_long <- state_monthly %>%
  transmute(State, STATEFP, date,
            Predicted_cases, Predicted_cases_lwr, Predicted_cases_upr,
            Predicted_inc, Predicted_inc_lwr, Predicted_inc_upr,
            NNDSS_cases, NNDSS_inc) %>%
  pivot_longer(cols = -c(State, STATEFP, date),
               names_to = c("Series", ".value"),
               names_pattern = "^(Predicted|NNDSS)_(cases_lwr|cases_upr|inc_lwr|inc_upr|cases|inc)$") %>%
  filter(!(Series == "NNDSS" & STATEFP %in% non_reporting_statefp))

start_date <- as.Date("2018-01-01")
end_date <- as.Date("2023-01-01")

state_monthly_long_short <- state_monthly_long %>%
  filter(date >= start_date, date <= end_date)

# Metric to plot: "cases" or "inc"
plot_metric <- "cases"

y_col <- plot_metric
lwr_col <- paste0(plot_metric, "_lwr")
upr_col <- paste0(plot_metric, "_upr")

y_axis_lab <- ifelse(plot_metric == "cases", "Monthly cases (state totals)", "Monthly incidence per 100,000")

ggplot() +
  geom_ribbon(data = filter(state_monthly_long_short, Series == "Predicted"),
              aes(x = date, ymin = .data[[lwr_col]], ymax = .data[[upr_col]], group = State),
              inherit.aes = FALSE, fill = "#b23a48", alpha = 0.08) +
  geom_line(data = state_monthly_long_short,
            aes(x = date, y = .data[[y_col]], color = Series),
            linewidth = 0.8, na.rm = TRUE) +
 facet_wrap(~ State, ncol = 1, scales = "free_y", strip.position = "top") +
  ggh4x::facetted_pos_scales(y = list(State %in% c("Arizona", "California") ~ scale_y_continuous(limits = c(0, 2500)))) +
   scale_x_date(limits = c(start_date, end_date),
               expand = expansion(mult = c(0, 0.02)),
               date_breaks = "1 year",
               date_labels = "%Y") +
  geom_hline(yintercept = 0, color = "black") +
  labs(x = NULL, y = y_axis_lab, color = NULL, title = NULL) +
  scale_color_manual(values = c("Predicted" = "#b23a48", "NNDSS" = alpha("black", 0.8)),
                     labels = c("Predicted" = "Predicted", "NNDSS" = "Reported (NNDSS)"),
                     guide = guide_legend(override.aes = list(linewidth = 1.3), keywidth = 1.8)) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom",
        plot.title = element_text(hjust = 0.5, size = 12),
        panel.grid.major = element_line(linewidth = 0.2),
        panel.grid.minor = element_blank(),
        panel.border = element_blank(),
        axis.ticks = element_blank(),
        axis.line.y = element_line(color = "black"),
        strip.background = element_blank(),
        axis.text = element_text(size = 12),
        axis.title = element_text(size = 14),
        legend.text = element_text(size = 13),
        panel.spacing.y = grid::unit(1.4, "lines"),
        strip.text = element_text(face = "plain", size = 12, margin = margin(b = 1.4)))


###### 13g. Supplementary Figure S12: high-incidence counties, model vs NNDSS (excluding AZ/CA) #####

annual_cases_df_fips <- annual_cases_df %>%
  mutate(STATEFP = substr(FIPS, 1, 2)) %>%
  filter(!STATEFP %in% c("04", "06")) %>%
  group_by(FIPS, STATEFP) %>%
  summarise(Mean_Annual_Pred_Inc = na_if(mean(Annual_Pred_Inc_per100k, na.rm = TRUE), NaN),
            Mean_Annual_NNDSS_Inc = na_if(mean(Annual_NNDSS_Inc_per100k, na.rm = TRUE), NaN),
            HighInc_Pred = Mean_Annual_Pred_Inc >= 10,
            HighInc_NNDSS = Mean_Annual_NNDSS_Inc >= 10,
            .groups = "drop")

pred_high_obs_low <- annual_cases_df_fips %>%
  filter(HighInc_Pred == TRUE, HighInc_NNDSS == FALSE)
unique(pred_high_obs_low$STATEFP)

pred_low_obs_high <- annual_cases_df_fips %>%
  filter(HighInc_Pred == FALSE, HighInc_NNDSS == TRUE)

bias_flags <- annual_cases_df_fips %>%
  mutate(bias_cat = case_when(
    HighInc_Pred & HighInc_NNDSS ~ "Predicted and reported high",
    HighInc_Pred & !HighInc_NNDSS ~ "Predicted high, reported low",
    !HighInc_Pred & HighInc_NNDSS ~ "Predicted low, reported high",
    TRUE ~ "Other")) %>%  select(FIPS, bias_cat)

bias_map <- counties %>% left_join(bias_flags, by = "FIPS")
bias_map_5070 <- bias_map %>% st_transform(5070)

ggplot() + geom_sf(data = bias_map_5070 %>%
            mutate(fill_cat = case_when(
              STATEFP %in% c("04", "06", non_reporting_statefp) ~ "Excluded from analysis",
              TRUE ~ bias_cat)),
          aes(fill = fill_cat), color = NA, lwd = 0.01) +
  geom_sf(data = states, fill = NA, color = "black", size = 0.1) +
  geom_sf(data = bias_map %>%
            filter(bias_cat %in% c("Predicted high, reported low",
                                   "Predicted low, reported high",
                                   "Predicted and reported high")),
          color = "black", fill = NA, size = 0.35) +
  scale_fill_manual(values = c(
    "Excluded from analysis" = "grey82",
    "Predicted high, reported low" = "#ad2831",
    "Predicted low, reported high" = "#0077b6",
    "Predicted and reported high" = "#69306d"),
    breaks = c("Predicted high, reported low",
               "Predicted low, reported high",
               "Predicted and reported high",
               "Excluded from analysis"),
    na.value = NA, na.translate = FALSE, name = NULL) +
  labs(title = "Discrepancies in counties considered high-incidence") +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank(),
        axis.title = element_blank(),
        legend.position = c(0.15, 0.16),
        legend.background = element_rect(fill = alpha("white", 0.01), color = NA),
        legend.key.height = unit(0.6, "cm"),
        legend.key.width = unit(0.6, "cm"),
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 12),
        legend.spacing.y = unit(0.5, "cm"),
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.title.position = "plot")


###### 13h. Supplementary Figure S13: predictors in counties predicted high but reported low ######

# Uses counties_model and meta_counties from section 10 (main specification)
monthly_vars <- c("mean_dust_ugm3_lag_1", "mean_dust_ugm3_lag_2", "mean_dust_ugm3_lag_3",
                  "Monthly_Dog_Inc_lag_0", "Total_Pos_Dog_Tests_lag_0", "Total_Dog_Tests_lag_0",
                  "EHRInc_wt_downscale", "pdsi",
                  paste0("tmin_lag_", 2:26),
                  paste0("tmax_lag_", 2:26),
                  paste0("precip_lag_", 2:26))

static_vars <- c("elev_m", "mean_annual_temp", "mean_annual_precip", "SkinTest_PercentReactors_Numeric",
                 "SoilOrgC_30cm", "SoilWaterPercent_30cm",
                 "SoilTexture_30cm_", "SoilTexture_30cm_Cl", "SoilTexture_30cm_ClLo", "SoilTexture_30cm_Lo",
                 "SoilTexture_30cm_LoSa", "SoilTexture_30cm_Sa", "SoilTexture_30cm_SaClLo", "SoilTexture_30cm_SaLo",
                 "SoilTexture_30cm_Si", "SoilTexture_30cm_SiClLo", "SoilTexture_30cm_SiLo")

counties_model_nometadata <- counties_model %>% select(-YEAR, -starts_with("MONTH_"))
yearly_vars <- setdiff(names(counties_model_nometadata), c(monthly_vars, static_vars))

# Exclude AZ/CA, AK, and HI
counties_model2 <- bind_cols(meta_counties, counties_model_nometadata) %>%
  mutate(FIPS = sprintf("%05s", as.character(FIPS)),
         STATEFP = substr(FIPS, 1, 2)) %>%
  filter(!STATEFP %in% c("04", "06", "02", "15"))

county_features <- counties_model2 %>%
  group_by(FIPS) %>%
  summarise(across(all_of(monthly_vars), ~ mean(.x, na.rm = TRUE), .names = "{.col}"),
            across(all_of(yearly_vars), ~ mean(.x, na.rm = TRUE), .names = "{.col}"),
            across(all_of(static_vars), ~ dplyr::first(.x[!is.na(.x)]), .names = "{.col}"),
            .groups = "drop")

over_ids <- pred_high_obs_low %>%
  mutate(FIPS = sprintf("%05s", as.character(FIPS))) %>%
  pull(FIPS)

analysis_df <- county_features %>%
  mutate(FIPS = sprintf("%05s", as.character(FIPS)),
         BiasGroup = case_when(FIPS %in% over_ids ~ "Over",
                               TRUE ~ "AllOthers"))
table(analysis_df$BiasGroup)

num_cols <- names(analysis_df)[sapply(analysis_df, is.numeric)]

analysis_scaled <- analysis_df %>%
  mutate(across(all_of(num_cols), ~ as.numeric(scale(.x)), .names = "z_{.col}"))

z_wide <- analysis_scaled %>%
  select(BiasGroup, starts_with("z_")) %>%
  pivot_longer(-BiasGroup, names_to = "feature_z", values_to = "z_value") %>%
  group_by(BiasGroup, feature_z) %>%
  summarise(z_mean = mean(z_value, na.rm = TRUE), .groups = "drop") %>%
  mutate(feature = sub("^z_", "", feature_z)) %>%
  select(BiasGroup, feature, z_mean) %>%
  pivot_wider(names_from = BiasGroup, values_from = z_mean)

if(!"Over" %in% names(z_wide)) z_wide$Over <- 0
if(!"AllOthers" %in% names(z_wide)) z_wide$AllOthers <- 0

# Rank features by standardized difference between groups
feature_rank <- z_wide %>%
  mutate(diff_over = Over - AllOthers) %>%
  arrange(desc(abs(diff_over)))
feature_rank %>% slice_head(n = 10)

top8_features <- c("EHRInc_wt_downscale", "SkinTest_PercentReactors_Numeric",
                   "SoilTexture_30cm_SiClLo", "Total_Dog_Tests_lag_0",
                   "Total_Pos_Dog_Tests_lag_0", "mean_dust_ugm3_lag_1",
                   "modal_land_ScrubShrub", "pct_ScrubShrub")

pretty_feat <- c(pct_ScrubShrub = "% Scrub/shrub",
                 SkinTest_PercentReactors_Numeric = "Coccidioidin reactivity",
                 EHRInc_wt_downscale = "EHR-based incidence",
                 mean_dust_ugm3_lag_1 = "Mean dust (lag 1-mo)",
                 modal_land_ScrubShrub = "Modal land: scrub/shrub",
                 Total_Pos_Dog_Tests_lag_0 = "# Positive canine tests",
                 Total_Dog_Tests_lag_0 = "# Total canine tests",
                 SoilTexture_30cm_SiClLo = "Silt clay loam soil texture")

facet_order <- feature_rank %>%
  filter(feature %in% top8_features) %>% arrange(desc(abs(diff_over))) %>% pull(feature)

plot_raw <- analysis_df %>%
  select(BiasGroup, all_of(top8_features)) %>%
  pivot_longer(-BiasGroup, names_to = "feature", values_to = "value") %>%
  group_by(BiasGroup, feature) %>%
  summarise(mean_value = mean(value, na.rm = TRUE),
            se_value = sd(value, na.rm = TRUE) / sqrt(sum(!is.na(value))),
            .groups = "drop") %>%
  mutate(value_lwr = mean_value - se_value,
         value_upr = mean_value + se_value,
         BiasGroup = recode(BiasGroup, Over = "Model only", AllOthers = "All Others"),
         BiasGroup = factor(BiasGroup, levels = c("Model only", "All Others")),
         feature = factor(feature, levels = facet_order),
         feature_lab = ifelse(as.character(feature) %in% names(pretty_feat), pretty_feat[as.character(feature)], as.character(feature)),
         feature_lab = factor(feature_lab, levels = ifelse(facet_order %in% names(pretty_feat), pretty_feat[facet_order], facet_order)))

ggplot(plot_raw, aes(x = BiasGroup, y = mean_value, fill = BiasGroup)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = value_lwr, ymax = value_upr), width = 0.2, linewidth = 0.5) +
  facet_wrap(~feature_lab, ncol = 2, scales = "free_y") +
  labs(x = NULL, y = NULL, title = "Predictor comparisons") +
  scale_fill_manual(values = c("Model only" = "#b2182b", "All Others" = "grey30"), name = NULL) +
  scale_x_discrete(labels = c("Model only" = "Predicted High,\nReported Low",
                              "All Others" = "All Others")) +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank(),
        legend.position = "none",
        axis.text.x = element_text(size = 11),
        axis.text.y = element_text(size = 11),
        strip.text = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))


################################################################################
############# Additional supplementary figures #################################
################################################################################
###### 14. Supplementary Figure S2: correlation between predictors (tract-month) ######

dataRaw <- fread(tract_file)
dataRaw$GEOID <- as.numeric(dataRaw$GEOID)

dataRaw <- dataRaw %>% filter(YEAR >= 2013) %>%
  select(-c("FIPS", "cases")) %>% filter(PopEstimate > 0) %>% filter(!is.na(Enhanced_Inc)) %>%
  mutate(across(contains("Dog"),~ ifelse(is.na(.) | is.nan(.), 0, .))) %>%
  filter(PopEstimate >= 100) %>% select(-c("SoilpH_30cm", "SoilpH_10cm", "SoilpH_0cm",
  "SoilOrgC_0cm", "SoilOrgC_10cm", "SoilWaterPercent_0cm", "SoilWaterPercent_10cm",
   "SoilTexture_0cm", "SoilTexture_10cm", "Enhanced_Inc"))

dataRaw$SkinTest_PercentReactors_Numeric <- str_extract_all(dataRaw$SkinTest_PercentReactors, "\\d+") %>%  lapply(\(x) as.numeric(x)) %>%  sapply(\(x) mean(x))
dataRaw <- dataRaw %>% select(-"SkinTest_PercentReactors")

data_dt <- as.data.table(dataRaw)
num_cols <- names(Filter(is.numeric, data_dt))

# Show only 2- to 6-month climate lags
all_lag_vars <- grep("^(tmin|tmax|precip)_lag_", num_cols, value = TRUE)
keep_lag_vars <- grep("^(tmin|tmax|precip)_lag_[2-6]$", num_cols, value = TRUE)
drop_lag_vars <- setdiff(all_lag_vars, keep_lag_vars)

corr_vars <- setdiff(num_cols, c("GEOID","YEAR","MONTH", drop_lag_vars))

time_grid <- data_dt[YEAR >= 2013 & YEAR <= 2023, .(YEAR, MONTH)]
time_grid <- unique(time_grid)[order(YEAR, MONTH)]

p <- length(corr_vars)
corr_sum   <- matrix(0, nrow = p, ncol = p, dimnames = list(corr_vars, corr_vars))
corr_count <- matrix(0, nrow = p, ncol = p, dimnames = list(corr_vars, corr_vars))

# Correlation matrix for each month, then averaged across months
for (i in seq_len(nrow(time_grid))) {
  yr <- time_grid$YEAR[i]
  mo <- time_grid$MONTH[i]
  sub_dt <- data_dt[YEAR == yr & MONTH == mo, ..corr_vars]
  if (nrow(sub_dt) < 2) next
  cm <- suppressWarnings(cor(sub_dt, use = "pairwise.complete.obs"))
  ok_cm <- !is.na(cm)
  corr_sum[ok_cm]   <- corr_sum[ok_cm]   + cm[ok_cm]
  corr_count[ok_cm] <- corr_count[ok_cm] + 1}

corr_mat <- corr_sum
corr_mat[corr_count > 0] <- corr_sum[corr_count > 0] / corr_count[corr_count > 0]
corr_mat[corr_count == 0] <- NA_real_

pretty_feat <- c(SkinTest_PercentReactors_Numeric = "Coccidioidin reactivity",
  Enhanced_Inc = "Enhanced surveillance incidence",
  EHRInc_wt_downscale = "EHR-based incidence",
  PopEstimate = "Population size",
  Monthly_Dog_Inc_lag_0 = "Canine incidence",
  Total_Pos_Dog_Tests_lag_0 = "# positive canine tests",
  Total_Dog_Tests_lag_0 = "Total # canine tests",
  pct_construction_extraction = "% construct/extract",
  pct_farming_fishing_forestry = "% farm/fish/forestry",
  pct_uninsured_u65 = "% uninsured <65 yrs",
  pct_age_65plus = "% 65+",
  pct_black = "% Black",
  pct_adults_below_poverty = "% adults below poverty",
  Diabetes_Per = "Diabetes prevalence",
  median_hh_income = "Median hh income",
  mean_dust_ugm3_lag_1 = "Dust conc. (1-mo lag)",
  mean_dust_ugm3_lag_2 = "Dust con. (2-mo lag)",
  mean_dust_ugm3_lag_3 = "Dust conc. (3-mo lag)",
  SoilOrgC_30cm = "Soil organic carbon",
  SoilWaterPercent_30cm = "Soil water %",
  pct_OpenWater = "% open water",
  pct_IceSnow = "% ice/snow",
  pct_DevelopedOpen = "% developed (open)",
  pct_DevelopedLow = "% developed (low)",
  pct_DevelopedMed = "% developed (mid)",
  pct_DevelopedHigh = "% developed (high)",
  pct_Barren = "% barren",
  pct_Deciduous = "% deciduous forest",
  pct_Evergreen = "% evergreen forest",
  pct_MixedForest = "% mixed forest",
  pct_ScrubShrub = "% scrub/shrub",
  pct_Grassland = "% grassland",
  pct_Pasture = "% pasture/hay",
  pct_Crops = "% cultivated crops",
  pct_WoodyWet = "% woody wetlands",
  pct_HerbyWetlands = "% herby wetlands",
  pct_NoLandData = "% land cover missing",
  elev_m = "Elevation", pdsi = "PDSI",
  mean_annual_temp = "Mean annual temp.",
  mean_annual_precip = "Mean annual precip.",
  SoilTexture_30cm_SaLo = "Sandy loam soil",
  SoilTexture_30cm_ClLo = "Clay loam soil",
  SoilTexture_30cm_SiLo = "Silt loam soil",
  modal_land_DevelopedMed = "Modal land: developed (mid)",
  modal_land_Evergreen = "Modal land: evergreen",
  modal_land_Grassland = "Modal land: grassland",
  modal_land_DevelopedOpen = "Modal land: developed (open)",
  modal_land_OpenWater = "Modal land: open water")

for (k in 2:14) {pretty_feat[paste0("tmin_lag_", k)] <- paste0("Min. temp. (", k, "-mo lag)")}
for (k in 2:14) {pretty_feat[paste0("tmax_lag_", k)] <- paste0("Max. temp. (", k, "-mo lag)")}
for (k in 2:14) {pretty_feat[paste0("precip_lag_", k)] <- paste0("Precip. (", k, "-mo lag)")}

var_names <- colnames(corr_mat)

feat_order <- rev(c(
  c("EHRInc_wt_downscale","SkinTest_PercentReactors_Numeric","PopEstimate"),
  grep("^Monthly_Dog_Inc_lag_", var_names, value = TRUE),
  grep("^Total_Pos_Dog_Tests_lag_", var_names, value = TRUE),
  grep("^Total_Dog_Tests_lag_", var_names, value = TRUE),
  c("pct_construction_extraction","pct_farming_fishing_forestry"),
  c("pct_uninsured_u65","pct_age_65plus","pct_black",
    "pct_adults_below_poverty","Diabetes_Per","median_hh_income"),
  grep("^mean_dust_ugm3_lag_", var_names, value = TRUE),
  grep("^SoilOrgC_", var_names, value = TRUE),
  grep("^SoilWaterPercent_", var_names, value = TRUE),
  grep("^SoilTexture_", var_names, value = TRUE),
  grep("^pct_", var_names, value = TRUE),
  c("elev_m","pdsi","mean_annual_temp","mean_annual_precip"),
  grep("^(tmin|tmax|precip)_lag_\\d+$", var_names, value = TRUE)))

feat_order <- unique(feat_order[feat_order %in% var_names])
axis_labels <- ifelse(feat_order %in% names(pretty_feat), pretty_feat[feat_order], feat_order)

corr_dt <- as.data.table(as.table(corr_mat))
setnames(corr_dt, c("Var1","Var2","corr"))
corr_dt[, Var1 := factor(Var1, levels = feat_order, labels = axis_labels)]
corr_dt[, Var2 := factor(Var2, levels = feat_order, labels = axis_labels)]

ggplot(corr_dt, aes(x = Var1, y = Var2, fill = corr)) +
  geom_rect(data = data.frame(xmin = 0.5,xmax = length(levels(corr_dt$Var1)) + 0.5,
      ymin = 0.5,ymax = length(levels(corr_dt$Var2)) + 0.5),
      aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
      inherit.aes = FALSE,color = "black",fill = NA,linewidth = 0.6) +
    geom_tile() +
  scale_fill_gradient2(low = "#457b9d", mid = "white", high = "#e63946", midpoint = 0) +
  labs(fill = "r", x = NULL, y = NULL,
       title = "Predictor correlation matrix based on tract-month averages") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(hjust = 0.5, size = 16),
    legend.title = element_text(size = 12,  hjust = 0.2))


#### 14b. Supplementary Figure S16: distribution of tract-level incidence #####

data <- fread(file.path(out_dir, "FullTrainingAndTestingData_ForComparison_Main.csv"))

data_zero <- data %>% filter(Enhanced_Inc == 0)
data_pos  <- data %>% filter(Enhanced_Inc > 0)

p_zero <- ggplot(data.frame(x = 0, count = nrow(data_zero)), aes(x = x, y = count)) +
  geom_col(fill = "#415a77", color = "white", alpha = 0.8, width = 0.5) +
  scale_x_continuous(limits = c(-0.5, 0.5), breaks = 0, labels = "0") +
  labs(x = "cases/100K", y = "Count", title = "Zero incidence") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 12))

p_pos <- ggplot(data_pos, aes(x = Enhanced_Inc)) +
  geom_histogram(binwidth = 5, boundary = 0,
                 fill = "#415a77", color = "white", alpha = 0.8) +
  coord_cartesian(xlim = c(0, 150)) +
  labs(x = "cases/100K", y = "Count", title = "Non-zero incidence") +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 12))

combined <- plot_grid(p_zero, p_pos, ncol = 2, labels = NULL, rel_widths = c(0.5, 1.5))

ggdraw() +
  draw_label("Histogram of tract-level coccidioidomycosis incidence (cases/100K)",
             x = 0.5, y = 0.98, hjust = 0.5, vjust = 1, size = 16) +
  draw_plot(combined, y = 0, height = 0.94)
