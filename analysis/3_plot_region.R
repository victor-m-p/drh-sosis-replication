# VMP 2026-01-31 (draft)
# Plot AME by region for each marker and save one figure per marker.

library(pacman)
p_load(
  tidyverse,
  here,
  brms,
  tidybayes,
  ggdist,
  ggthemes
)

# -------------------------
# Config
# -------------------------
fits_dir  <- here("data", "mdl_fits")          # where saved brms fits live
input_dir <- here("data", "mdl_input")         # per-marker csv inputs
out_dir   <- here("figures", "region_ame")     # output plots

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot(dir.exists(out_dir))

# If you want nicer labels, fill this (optional)
pretty_marker <- c(
  circumcision = "Circumcision",
  dress = "Dress",
  extra_ritual_group_markers = "Extra-Ritual In-Group Markers",
  food_taboos = "Food Taboos",
  hair = "Hair",
  ornaments = "Ornaments",
  permanent_scarring = "Permanent Scarring",
  tattoos_scarification = "Tattoos or Scarification"
)

# -------------------------
# Discover markers from saved fits
# -------------------------
fit_files <- list.files(fits_dir, pattern = ".rds$", full.names = TRUE)
stopifnot(length(fit_files) > 0)
fit_files

markers <- fit_files |>
  basename() |>
  str_replace(".rds$", "") |>
  sort()

# -------------------------
# Helper: compute + plot AME by region
# -------------------------
plot_region_ame <- function(fit, dat, marker, year0 = 0) {
  regions <- unique(dat$world_region)
  
  ep_reg <- fit %>%
    epred_draws(
      newdata = tidyr::expand_grid(
        violent_external = c(0, 1),
        year_scaled = year0,
        world_region = regions
      ),
      re_formula = NULL
    ) %>%
    ungroup() %>% 
    select(.draw, world_region, violent_external, .epred) %>%
    mutate(violent_external = as.integer(violent_external)) %>%
    group_by(.draw, world_region, violent_external) %>%
    summarise(.epred = mean(.epred), .groups = "drop")
  
  ame_reg <- ep_reg %>%
    pivot_wider(
      names_from = violent_external,
      values_from = .epred,
      names_prefix = "vx_"
    ) %>%
    mutate(ame = vx_1 - vx_0)
  
  stopifnot(sum(is.na(ame_reg$ame)) == 0)

  # --- counts + ordering by n ---
  region_n <- dat %>%
    count(world_region, name = "n") %>%
    arrange(desc(n))
  
  # order regions by sample size (largest at top)
  region_order <- region_n$world_region
  
  # make labels like "Europe (n=64)"
  region_labels <- setNames(
    paste0(region_n$world_region, " (n=", region_n$n, ")"),
    region_n$world_region
  )
  
  # apply ordering + labels
  ame_reg <- ame_reg %>%
    mutate(world_region = factor(world_region, levels = region_order))
  
  title_label <- pretty_marker[[marker]]
  if (is.null(title_label) || is.na(title_label)) title_label <- marker
  
  lims <- quantile(ame_reg$ame, c(0.0005, 0.9995))   # 0.05%–99.95%
  pad  <- 0.02
  lims <- c(lims[1] - pad, lims[2] + pad)
  
  ggplot(ame_reg, aes(x = ame, y = world_region)) +
    stat_halfeye(alpha = 0.9, normalize = "groups", adjust = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3) +
    coord_cartesian(xlim = lims) +
    scale_y_discrete(labels = region_labels) +
    theme_clean() +
    labs(
      title = title_label,
      subtitle = paste0("AME by region (Yes − No), year_scaled = ", year0),
      x = "Average marginal effect (probability scale)",
      y = NULL
    )
}

# -------------------------
# Main loop
# -------------------------
for (marker in markers) {
  message("\n--- ", marker, " ---")
  
  fit_path <- file.path(fits_dir, paste0(marker, ".rds"))
  in_path  <- file.path(input_dir, paste0(marker, ".csv"))
  
  if (!file.exists(in_path)) {
    warning("Missing input csv for marker: ", marker, " (", in_path, "). Skipping.")
    next
  }
  
  fit <- readRDS(fit_path)
  dat <- readr::read_csv(in_path, show_col_types = FALSE) %>%
    mutate(world_region = factor(world_region))
  
  # basic sanity
  stopifnot(all(dat$violent_external %in% c(0, 1)))
  stopifnot(is.numeric(dat$year_scaled))
  stopifnot(!anyNA(dat$world_region))
  
  p <- plot_region_ame(fit, dat, marker, year0 = 0)
  
  out_pdf <- file.path(out_dir, paste0(marker, "__ame_by_region.pdf"))
  out_png <- file.path(out_dir, paste0(marker, "__ame_by_region.png"))
  
  ggsave(out_pdf, p, width = 8.5, height = 5.5, device = "pdf")
  ggsave(out_png, p, width = 8.5, height = 5.5, dpi = 300)
  
}

