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

ref_palette <- c("#355f8d", "#4a90c2", "#6eb5c0", "#a7c957", "#f2cc8f", "#f28f3b", "#d1495b")
variant_short_levels <- c("Base", "P1-UQ", "P2-Pose", "P3-KG", "P4-SN")

compress_variant_labels <- function(x) {
  mapping <- c(
    "Base backbone" = "Base",
    "Path 1: Pocket uncertainty" = "P1-UQ",
    "Path 2: Multitask pose" = "P2-Pose",
    "Path 3: Knowledge graph" = "P3-KG",
    "Path 4: Structural negatives" = "P4-SN"
  )
  unname(mapping[x])
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

  make_hist <- function(df, column, title_text, xlab_text, fill_color) {
    values <- df[[column]]
    upper <- unname(stats::quantile(values, 0.995, na.rm = TRUE))
    lower <- min(values, na.rm = TRUE)
    plot_df <- df[values >= lower & values <= upper, , drop = FALSE]
    ggplot(plot_df, aes(x = .data[[column]])) +
      geom_histogram(bins = 24, fill = fill_color, color = "black", linewidth = 0.25) +
      labs(title = title_text, x = xlab_text, y = "Frequency") +
      theme_bw(base_size = 12) +
      theme(panel.grid.minor = element_blank())
  }

  p1 <- make_hist(davis, "affinity", "(a) Davis Affinity", "Binding Affinity", ref_palette[1])
  p2 <- make_hist(davis, "smiles_len", "(b) Davis SMILES Length", "Ligand SMILES Length", ref_palette[5])
  p3 <- make_hist(davis, "target_len", "(c) Davis Target Length", "Protein Sequence Length", ref_palette[4])
  p4 <- make_hist(kiba, "affinity", "(d) KIBA Affinity", "Binding Affinity", ref_palette[2])
  p5 <- make_hist(kiba, "smiles_len", "(e) KIBA SMILES Length", "Ligand SMILES Length", ref_palette[6])
  p6 <- make_hist(kiba, "target_len", "(f) KIBA Target Length", "Protein Sequence Length", ref_palette[7])

  combined <- (p1 + p2 + p3) / (p4 + p5 + p6)
  ggsave(file.path(out_dir, "dataset_distributions.png"), combined, width = 12, height = 8, dpi = 300)
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
  p <- build_attention_heatmap_plot(df, "Cross-Attention Heat Map")
  ggsave(file.path(out_dir, "attention_heatmap.png"), p, width = 9, height = 6, dpi = 300)
}

build_attention_heatmap_plot <- function(df, title_text) {
  ggplot(df, aes(x = column_label, y = row_label, fill = attention_value)) +
    geom_tile() +
    scale_fill_gradientn(colors = c("#355f8d", "#4a90c2", "#6eb5c0", "#a7c957", "#f2cc8f", "#f28f3b", "#d1495b")) +
    labs(title = title_text, x = "Protein Residues", y = "Drug Atoms", fill = "Weight") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
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

  if ("pearson_r" %in% names(df) && !("pearson" %in% names(df))) {
    df$pearson <- df$pearson_r
  }
  metric_cols <- intersect(c("ci", "mse", "rmse", "pearson", "mae"), names(df))
  if (length(metric_cols) == 0) stop("Metrics CSV must contain at least one metric column")

  for (col in metric_cols) df[[col]] <- as.numeric(df[[col]])
  shorten_run <- function(run_name) {
    map <- c(
      base_final = "Base",
      path1_final = "P1-UQ",
      path2_final = "P2-Pose",
      path3_final = "P3-KG",
      path4_final = "P4-SN",
      kiba_base_final = "Base",
      kiba_path1_final = "P1-UQ",
      kiba_path2_final = "P2-Pose",
      kiba_path3_final = "P3-KG",
      kiba_path4_final = "P4-SN"
    )
    ifelse(run_name %in% names(map), unname(map[run_name]), run_name)
  }
  df$run_label <- paste0(ifelse(df$dataset == "davis", "D-", "K-"), shorten_run(df$run), "-", ifelse(df$split == "validation", "Val", "Test"))

  long_df <- tidyr::pivot_longer(df, cols = all_of(metric_cols), names_to = "metric", values_to = "value")
  long_df$metric <- factor(long_df$metric, levels = c("ci", "pearson", "mse", "rmse", "mae"), labels = c("CI", "Pearson", "MSE", "RMSE", "MAE"))
  ordered_labels <- c(
    "D-Base-Val", "D-Base-Test", "D-P1-UQ-Val", "D-P1-UQ-Test", "D-P2-Pose-Val", "D-P2-Pose-Test",
    "D-P3-KG-Val", "D-P3-KG-Test", "D-P4-SN-Val", "D-P4-SN-Test",
    "K-Base-Val", "K-Base-Test", "K-P1-UQ-Val", "K-P1-UQ-Test", "K-P2-Pose-Val", "K-P2-Pose-Test",
    "K-P3-KG-Val", "K-P3-KG-Test", "K-P4-SN-Val", "K-P4-SN-Test"
  )
  long_df$run_label <- factor(long_df$run_label, levels = rev(ordered_labels))
  p <- ggplot(long_df, aes(x = metric, y = run_label, fill = value)) +
    geom_tile(color = "white", linewidth = 0.3) +
    scale_fill_gradientn(colors = c("#3b4cc0", "#70a5d8", "#d8ef9b", "#f9d057", "#f28f3b", "#b40426")) +
    labs(x = "Metric", y = "Run", fill = NULL) +
    theme_classic(base_size = 12) +
    theme(
      axis.text.x = element_text(angle = 30, hjust = 1)
    )
  ggsave(file.path(out_dir, "metrics_heatmap.png"), p, width = 9.2, height = 8.5, dpi = 300)
}

make_thesis_error_scatter <- function(csv_path = "paper_assets/data/thesis_scatter_path3_combined_test.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  for (col in c("y_true", "y_pred", "abs_error")) df[[col]] <- as.numeric(df[[col]])
  df$dataset <- factor(toupper(df$dataset), levels = c("DAVIS", "KIBA"), labels = c("Davis", "KIBA"))
  build_panel <- function(sub_df, title_text) {
    xr <- range(sub_df$y_pred, na.rm = TRUE)
    yr <- range(sub_df$y_true, na.rm = TRUE)
    xpad <- 0.03 * (xr[2] - xr[1])
    ypad <- 0.03 * (yr[2] - yr[1])
    ggplot(sub_df, aes(x = y_pred, y = y_true)) +
      geom_point(color = "#6c8ebf", alpha = 0.8, size = 0.6, stroke = 0) +
      geom_abline(intercept = 0, slope = 1, linetype = "dotted", color = "black", linewidth = 0.55) +
      geom_smooth(method = "lm", se = FALSE, color = "#D7191C", linewidth = 0.9) +
      coord_cartesian(xlim = c(xr[1] - xpad, xr[2] + xpad), ylim = c(yr[1] - ypad, yr[2] + ypad)) +
      labs(title = title_text, x = "Predicted Affinity", y = "Measured Affinity") +
      theme_classic(base_size = 12) +
      theme(plot.title = element_text(hjust = 0.5))
  }

  p1 <- build_panel(df[df$dataset == "Davis", ], "Davis")
  p2 <- build_panel(df[df$dataset == "KIBA", ], "KIBA")
  combined <- p1 / p2

  ggsave(file.path(out_dir, "thesis_scatter_path3_error.png"), combined, width = 7.3, height = 9.2, dpi = 300)
}

make_thesis_variant_comparison <- function(csv_path = "paper_assets/data/thesis_variant_comparison_test.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  for (col in c("ci", "mse", "rmse", "pearson_r")) df[[col]] <- as.numeric(df[[col]])
  long_df <- tidyr::pivot_longer(df, cols = c("ci", "mse", "rmse", "pearson_r"), names_to = "metric", values_to = "value")
  long_df$metric <- factor(long_df$metric, levels = c("ci", "mse", "rmse", "pearson_r"), labels = c("CI", "MSE", "RMSE", "Pearson"))
  long_df$variant <- factor(compress_variant_labels(long_df$variant), levels = variant_short_levels)

  p <- ggplot(long_df, aes(x = variant, y = value, fill = variant, group = dataset)) +
    geom_col(width = 0.72, color = "black", linewidth = 0.2) +
    geom_line(color = "black", linewidth = 0.45) +
    geom_point(color = "black", size = 1.15) +
    scale_fill_manual(values = c("#4C78A8", "#F28F3B", "#F2CC8F", "#A7C957", "#D1495B")) +
    facet_grid(dataset ~ metric, scales = "free_y") +
    labs(
      title = "Innovation-Path Comparison Across Davis and KIBA",
      x = "Variant",
      y = "Metric Value"
    ) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(out_dir, "thesis_variant_comparison.png"), p, width = 13, height = 7.5, dpi = 300)
}

make_thesis_training_curves <- function(
  davis_log_path = "results/path3_final/logs.csv",
  kiba_log_path = "results/kiba_path3_final/logs.csv",
  out_dir = "results/paper_figures"
) {
  ensure_dir(out_dir)
  davis <- safe_read_csv(davis_log_path)
  kiba <- safe_read_csv(kiba_log_path)

  davis <- davis[, c("epoch", "train_loss", "val_loss")]
  kiba <- kiba[, c("epoch", "train_loss", "val_loss")]
  davis$dataset <- "Davis"
  kiba$dataset <- "KIBA"
  df <- rbind(davis, kiba)
  long_df <- tidyr::pivot_longer(df, cols = c("train_loss", "val_loss"), names_to = "curve", values_to = "loss")
  long_df$curve <- factor(long_df$curve, levels = c("train_loss", "val_loss"), labels = c("Train", "Validation"))

  p <- ggplot(long_df, aes(x = epoch, y = loss, color = curve)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.2) +
    facet_wrap(~dataset, scales = "free_y") +
    scale_color_manual(values = c("Train" = "#4C78A8", "Validation" = "#E45756")) +
    labs(
      title = "Training and Validation Curves for the Knowledge-Graph Path",
      x = "Epoch",
      y = "Loss",
      color = "Curve"
    ) +
    theme_bw(base_size = 13) +
    theme(panel.grid.minor = element_blank())

  ggsave(file.path(out_dir, "thesis_training_curves_path3.png"), p, width = 10, height = 5.5, dpi = 300)
}

make_thesis_attention_panels <- function(
  davis_csv = "results/path3_final/attention_matrix.csv",
  kiba_csv = "results/kiba_path3_final/attention_matrix.csv",
  out_dir = "results/paper_figures"
) {
  ensure_dir(out_dir)
  davis <- safe_read_csv(davis_csv)
  kiba <- safe_read_csv(kiba_csv)
  select_top_attention <- function(df, top_rows = 18, top_cols = 24) {
    row_scores <- aggregate(attention_value ~ row_label, df, mean)
    col_scores <- aggregate(attention_value ~ column_label, df, mean)
    keep_rows <- head(row_scores[order(-row_scores$attention_value), "row_label"], top_rows)
    keep_cols <- head(col_scores[order(-col_scores$attention_value), "column_label"], top_cols)
    sub <- df[df$row_label %in% keep_rows & df$column_label %in% keep_cols, ]
    sub$row_label <- factor(sub$row_label, levels = rev(keep_rows))
    sub$column_label <- factor(sub$column_label, levels = keep_cols)
    sub
  }
  p1 <- build_attention_heatmap_plot(select_top_attention(davis), "(a) Davis Cross-Attention")
  p2 <- build_attention_heatmap_plot(select_top_attention(kiba), "(b) KIBA Cross-Attention")
  combined <- p1 + p2 + plot_layout(ncol = 2)
  ggsave(file.path(out_dir, "thesis_attention_heatmaps.png"), combined, width = 14, height = 6, dpi = 300)
}

make_thesis_variant_heatmap <- function(csv_path = "paper_assets/data/thesis_variant_comparison_test.csv", out_dir = "results/paper_figures") {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  for (col in c("ci", "mse", "rmse", "pearson_r")) df[[col]] <- as.numeric(df[[col]])
  long_df <- tidyr::pivot_longer(df, cols = c("ci", "mse", "rmse", "pearson_r"), names_to = "metric", values_to = "value")
  long_df$metric <- factor(long_df$metric, levels = c("ci", "mse", "rmse", "pearson_r"), labels = c("CI", "MSE", "RMSE", "Pearson"))
  long_df$variant <- factor(compress_variant_labels(long_df$variant), levels = rev(variant_short_levels))
  direction_map <- c("CI" = "high", "Pearson" = "high", "MSE" = "low", "RMSE" = "low")
  split_groups <- split(long_df, interaction(long_df$dataset, long_df$metric, drop = TRUE))
  long_df$score <- unlist(lapply(split_groups, function(sub) {
    vals <- sub$value
    rng <- max(vals, na.rm = TRUE) - min(vals, na.rm = TRUE)
    if (rng == 0) return(rep(0.5, nrow(sub)))
    metric_name <- as.character(sub$metric[[1]])
    if (direction_map[[metric_name]] == "high") {
      (vals - min(vals, na.rm = TRUE)) / rng
    } else {
      (max(vals, na.rm = TRUE) - vals) / rng
    }
  }))
  long_df$label <- sprintf("%.3f", long_df$value)

  build_panel <- function(ds) {
    sub <- long_df[long_df$dataset == ds, ]
    ggplot(sub, aes(x = metric, y = variant, fill = score)) +
      geom_tile(color = "white", linewidth = 0.35) +
      geom_text(aes(label = label), size = 3.0) +
      scale_fill_gradientn(colors = c("#3b4cc0", "#70a5d8", "#d8ef9b", "#f9d057", "#f28f3b", "#b40426"), limits = c(0, 1)) +
      labs(title = ds, x = NULL, y = NULL, fill = NULL) +
      theme_classic(base_size = 12) +
      theme(
        plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 0, hjust = 0.5)
      )
  }

  combined <- build_panel("Davis") + build_panel("KIBA") + plot_layout(ncol = 2, guides = "collect")
  ggsave(file.path(out_dir, "thesis_variant_heatmap.png"), combined, width = 10, height = 4.8, dpi = 300)
}

make_thesis_ablation_panels <- function(
  csv_path = "paper_assets/data/ablation_results_final.csv",
  out_dir = "results/paper_figures"
) {
  ensure_dir(out_dir)
  df <- safe_read_csv(csv_path)
  candidate_metrics <- intersect(c("mse", "ci", "rmse", "pearson_r", "rm2", "aupr"), names(df))
  if (length(candidate_metrics) == 0) stop("Ablation CSV must contain at least one supported metric column.")
  for (col in candidate_metrics) df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
  long_df <- tidyr::pivot_longer(df, cols = all_of(candidate_metrics), names_to = "metric", values_to = "value")
  long_df <- long_df[!is.na(long_df$value), ]
  if (nrow(long_df) == 0) stop("Ablation CSV contains no numeric values yet.")

  label_map <- c(
    mse = "MSE",
    ci = "CI",
    rmse = "RMSE",
    pearson_r = "Pearson",
    rm2 = "r_m^2",
    aupr = "AUPR"
  )
  long_df$metric <- factor(long_df$metric, levels = candidate_metrics, labels = unname(label_map[candidate_metrics]))
  long_df$variant <- factor(long_df$variant, levels = c(
    "Without cross-attention",
    "Without contrastive learning",
    "Without KG fusion",
    "Without physicochemical features",
    "Full model (Path 3)"
  ))

  p <- ggplot(long_df, aes(x = variant, y = value, fill = variant, group = dataset)) +
    geom_col(color = "black", linewidth = 0.2) +
    geom_line(color = "black", linewidth = 0.45) +
    geom_point(color = "black", size = 1.2) +
    facet_grid(dataset ~ metric, scales = "free_y") +
    scale_fill_manual(values = c("#4C78A8", "#F28F3B", "#F2CC8F", "#A7C957", "#D1495B")) +
    labs(
      title = "Ablation Comparison Across Davis and KIBA",
      x = "Variant",
      y = "Metric Value"
    ) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 30, hjust = 1),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )

  ggsave(file.path(out_dir, "thesis_ablation_panels.png"), p, width = 14, height = 8, dpi = 300)
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
  message("Tasks: distributions, training_curves, scatter, scatter_from_runs, performance, ablation, attention, attention_from_run, metrics_heatmap, kg_case, thesis_scatter_error, thesis_variant_comparison, thesis_variant_heatmap, thesis_training_curves, thesis_attention_panels, thesis_ablation_panels")
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
if (task == "thesis_scatter_error") make_thesis_error_scatter()
if (task == "thesis_variant_comparison") make_thesis_variant_comparison()
if (task == "thesis_variant_heatmap") make_thesis_variant_heatmap()
if (task == "thesis_training_curves") make_thesis_training_curves()
if (task == "thesis_attention_panels") make_thesis_attention_panels()
if (task == "thesis_ablation_panels") make_thesis_ablation_panels()
