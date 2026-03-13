suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(patchwork)
})

ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

safe_read_csv <- function(path) {
  if (!file.exists(path)) stop(sprintf("Missing file: %s", path))
  read.csv(path, stringsAsFactors = FALSE)
}

resolve_existing_path <- function(candidates) {
  existing <- candidates[file.exists(candidates)]
  if (length(existing) == 0) {
    stop(sprintf("None of the candidate files exist:\n%s", paste(candidates, collapse = "\n")))
  }
  existing[[1]]
}

make_dataset_distributions <- function(data_dir = "data/processed", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  davis <- safe_read_csv(file.path(data_dir, "davis.csv"))
  kiba <- safe_read_csv(file.path(data_dir, "kiba.csv"))

  davis$affinity <- as.numeric(davis$affinity)
  kiba$affinity <- as.numeric(kiba$affinity)
  davis$smiles_len <- nchar(davis$smiles)
  kiba$smiles_len <- nchar(kiba$smiles)
  davis$target_len <- nchar(davis$fasta)
  kiba$target_len <- nchar(kiba$fasta)

  png(file.path(out_dir, "dataset_distributions.png"), width = 2200, height = 1600, res = 220)
  par(mfrow = c(2, 3), mar = c(4, 4, 3, 1))

  hist(davis$affinity, breaks = 30, col = "#4C78A8", border = "white", main = "(a) Davis Affinity", xlab = "Affinity")
  hist(kiba$affinity, breaks = 30, col = "#F58518", border = "white", main = "(b) KIBA Affinity", xlab = "Affinity")
  hist(davis$smiles_len, breaks = 30, col = "#54A24B", border = "white", main = "(c) Davis SMILES Length", xlab = "Length")
  hist(kiba$smiles_len, breaks = 30, col = "#E45756", border = "white", main = "(d) KIBA SMILES Length", xlab = "Length")
  hist(davis$target_len, breaks = 30, col = "#72B7B2", border = "white", main = "(e) Davis Target Length", xlab = "Length")
  hist(kiba$target_len, breaks = 30, col = "#B279A2", border = "white", main = "(f) KIBA Target Length", xlab = "Length")

  dev.off()
}

make_training_curves <- function(log_path = "results/logs.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  logs <- safe_read_csv(log_path)

  png(file.path(out_dir, "training_curves_publication.png"), width = 2200, height = 1000, res = 220)
  par(mfrow = c(1, 2), mar = c(5, 5, 3, 1))

  plot(logs$epoch, logs$train_loss, type = "o", pch = 16, col = "#4C78A8", lwd = 2,
       xlab = "Epoch", ylab = "Loss", main = "(a) Training vs Validation Loss")
  lines(logs$epoch, logs$val_loss, type = "o", pch = 17, col = "#E45756", lwd = 2)
  legend("topright", legend = c("Train", "Validation"), col = c("#4C78A8", "#E45756"), pch = c(16,17), bty = "n")

  plot(logs$epoch, logs$train_pred_std, type = "o", pch = 16, col = "#54A24B", lwd = 2,
       xlab = "Epoch", ylab = "Prediction Std / LR", main = "(b) Prediction Stability and LR")
  lines(logs$epoch, logs$val_pred_std, type = "o", pch = 17, col = "#F58518", lwd = 2)
  par(new = TRUE)
  plot(logs$epoch, logs$lr, type = "o", pch = 15, col = "#B279A2", axes = FALSE, xlab = "", ylab = "", log = "y", lwd = 2)
  axis(side = 4)
  mtext("Learning Rate", side = 4, line = 3)
  legend("topright", legend = c("Train pred std", "Val pred std", "LR"),
         col = c("#54A24B", "#F58518", "#B279A2"), pch = c(16,17,15), bty = "n")

  dev.off()
}

make_scatter_plot <- function(csv_path, out_path, title_text) {
  data <- safe_read_csv(csv_path)
  if (!all(c("y_true", "y_pred") %in% names(data))) stop("Scatter CSV must contain y_true and y_pred")

  rmse <- sqrt(mean((data$y_true - data$y_pred)^2, na.rm = TRUE))
  corr <- suppressWarnings(cor(data$y_true, data$y_pred, use = "complete.obs"))

  p <- ggplot(data, aes(x = y_true, y = y_pred)) +
    geom_point(color = "#4C78A8", alpha = 0.65, size = 1.8) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#E45756", linewidth = 0.8) +
    geom_smooth(method = "lm", se = FALSE, color = "#54A24B", linewidth = 0.9) +
    labs(title = title_text, x = "Measured Affinity", y = "Predicted Affinity",
         subtitle = sprintf("Pearson R = %.4f | RMSE = %.4f", corr, rmse)) +
    theme_bw(base_size = 14)

  ggsave(out_path, p, width = 7.5, height = 6.0, dpi = 300)
}

make_combined_scatter <- function(csv_path = "paper_assets/templates/scatter_template.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  data <- safe_read_csv(csv_path)
  if (!all(c("y_true", "y_pred", "dataset") %in% names(data))) stop("Scatter template requires y_true, y_pred, dataset")
  data$dataset <- factor(data$dataset, levels = c("Davis", "KIBA"))

  stat_df <- do.call(rbind, lapply(split(data, data$dataset), function(df) {
    data.frame(dataset = unique(df$dataset),
               rmse = sqrt(mean((df$y_true - df$y_pred)^2, na.rm = TRUE)),
               corr = suppressWarnings(cor(df$y_true, df$y_pred, use = "complete.obs")))
  }))
  data <- merge(data, stat_df, by = "dataset", all.x = TRUE)

  p <- ggplot(data, aes(x = y_true, y = y_pred)) +
    geom_point(color = "#4C78A8", alpha = 0.6, size = 1.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#E45756") +
    geom_smooth(method = "lm", se = FALSE, color = "#54A24B", linewidth = 0.8) +
    facet_wrap(~dataset, scales = "free") +
    labs(title = "Predicted vs Measured Affinity", x = "Measured Affinity", y = "Predicted Affinity") +
    theme_bw(base_size = 13)

  ggsave(file.path(out_dir, "scatter_davis_kiba.png"), p, width = 10, height = 5, dpi = 300)
}

make_combined_scatter_from_runs <- function(out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  davis_path <- resolve_existing_path(c(
    "results/base_final/test_predictions.csv",
    "results/base_final/validation_predictions.csv"
  ))
  kiba_path <- resolve_existing_path(c(
    "results/kiba_base_final/test_predictions.csv",
    "results/kiba_base_final/validation_predictions.csv"
  ))

  davis <- safe_read_csv(davis_path)
  kiba <- safe_read_csv(kiba_path)
  data <- rbind(davis[, c("y_true", "y_pred", "dataset")], kiba[, c("y_true", "y_pred", "dataset")])
  write.csv(data, "paper_assets/data/scatter_from_runs.csv", row.names = FALSE)

  stat_df <- do.call(rbind, lapply(split(data, data$dataset), function(df) {
    data.frame(dataset = unique(df$dataset),
               rmse = sqrt(mean((df$y_true - df$y_pred)^2, na.rm = TRUE)),
               corr = suppressWarnings(cor(df$y_true, df$y_pred, use = "complete.obs")))
  }))
  data <- merge(data, stat_df, by = "dataset", all.x = TRUE)

  p <- ggplot(data, aes(x = y_true, y = y_pred)) +
    geom_point(color = "#4C78A8", alpha = 0.6, size = 1.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#E45756") +
    geom_smooth(method = "lm", se = FALSE, color = "#54A24B", linewidth = 0.8) +
    facet_wrap(~dataset, scales = "free") +
    labs(title = "Predicted vs Measured Affinity", x = "Measured Affinity", y = "Predicted Affinity") +
    theme_bw(base_size = 13)

  ggsave(file.path(out_dir, "scatter_davis_kiba.png"), p, width = 10, height = 5, dpi = 300)
}

make_performance_bars <- function(csv_path = "paper_assets/templates/main_results_template.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  num_cols <- intersect(c("ci", "mse", "rmse", "pearson"), names(df))
  for (col in num_cols) df[[col]] <- as.numeric(df[[col]])

  for (metric in c("ci", "rmse")) {
    p <- ggplot(df, aes(x = model, y = .data[[metric]], fill = model)) +
      geom_col() +
      facet_wrap(~dataset, scales = "free_y") +
      coord_flip() +
      labs(title = sprintf("Performance Comparison by %s", toupper(metric)), x = "Model", y = toupper(metric)) +
      theme_bw(base_size = 13) +
      theme(legend.position = "none")
    ggsave(file.path(out_dir, paste0("performance_", metric, ".png")), p, width = 10, height = 6, dpi = 300)
  }
}

make_ablation_bars <- function(csv_path = "paper_assets/templates/ablation_results_template.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  for (metric in c("ci", "mse", "rmse")) df[[metric]] <- as.numeric(df[[metric]])

  p <- ggplot(df, aes(x = variant, y = rmse, fill = variant)) +
    geom_col() +
    facet_wrap(~dataset, scales = "free_y") +
    coord_flip() +
    labs(title = "Ablation Comparison (RMSE)", x = "Variant", y = "RMSE") +
    theme_bw(base_size = 13) +
    theme(legend.position = "none")
  ggsave(file.path(out_dir, "ablation_comparison.png"), p, width = 10, height = 6, dpi = 300)
}

make_attention_heatmap <- function(csv_path = "paper_assets/templates/attention_heatmap_template.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  p <- ggplot(df, aes(x = column_label, y = row_label, fill = attention_value)) +
    geom_tile() +
    scale_fill_gradient(low = "#F3F7FA", high = "#0B6E99") +
    labs(title = "Cross-Attention Heat Map", x = "Protein Residues", y = "Drug Atoms", fill = "Weight") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
  ggsave(file.path(out_dir, "attention_heatmap.png"), p, width = 9, height = 6, dpi = 300)
}

make_attention_heatmap_from_run <- function(csv_path = NULL, out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  if (is.null(csv_path)) {
    csv_path <- resolve_existing_path(c(
      "results/base_final/attention_matrix.csv",
      "results/path3_final/attention_matrix.csv",
      "paper_assets/templates/attention_heatmap_template.csv"
    ))
  }
  make_attention_heatmap(csv_path, out_dir)
}

make_metrics_heatmap <- function(csv_path = "paper_assets/data/final_runs_metrics.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  required_cols <- c("run", "split", "dataset")
  if (!all(required_cols %in% names(df))) stop("Metrics CSV must contain run, split, dataset")

  metric_cols <- intersect(c("ci", "mse", "rmse", "pearson", "mae"), names(df))
  if (length(metric_cols) == 0) stop("Metrics CSV must contain at least one metric column")

  for (col in metric_cols) df[[col]] <- as.numeric(df[[col]])
  df$run_label <- paste(df$dataset, df$run, df$split, sep = " | ")

  long_df <- tidyr::pivot_longer(df, cols = all_of(metric_cols), names_to = "metric", values_to = "value")

  p <- ggplot(long_df, aes(x = metric, y = run_label, fill = value)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.4f", value)), size = 3) +
    facet_wrap(~dataset, scales = "free_y") +
    scale_fill_gradient(low = "#F3F7FA", high = "#0B6E99") +
    labs(title = "Validation/Test Metrics Heatmap", x = "Metric", y = "Run | Split", fill = "Value") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  ggsave(file.path(out_dir, "metrics_heatmap.png"), p, width = 12, height = 8, dpi = 300)
}

make_kg_case_study <- function(csv_path = "paper_assets/templates/kg_case_study_template.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  df$neighbor_label <- factor(df$neighbor_label, levels = rev(df$neighbor_label))
  p <- ggplot(df, aes(x = neighbor_label, y = similarity)) +
    geom_col(fill = "#4C78A8") +
    coord_flip() +
    labs(title = "KG Retrieval Case Study", x = "Retrieved Neighbor", y = "Similarity") +
    theme_bw(base_size = 13)
  ggsave(file.path(out_dir, "kg_case_study.png"), p, width = 8, height = 5, dpi = 300)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  message("Usage: Rscript scripts/generate_paper_figures.R <task>")
  message("Tasks: distributions, training_curves, scatter, scatter_from_runs, performance, ablation, attention, attention_from_run, metrics_heatmap, kg_case")
  quit(status = 1)
}

task <- args[[1]]
if (task == "distributions") make_dataset_distributions()
if (task == "training_curves") make_training_curves()
if (task == "scatter") make_combined_scatter()
if (task == "scatter_from_runs") make_combined_scatter_from_runs()
if (task == "performance") make_performance_bars()
if (task == "ablation") make_ablation_bars()
if (task == "attention") make_attention_heatmap()
if (task == "attention_from_run") make_attention_heatmap_from_run()
if (task == "metrics_heatmap") make_metrics_heatmap()
if (task == "kg_case") make_kg_case_study()
