# VMP 2026-02-01
# Summary plots across markers:
#   (A) Grand mean posterior predictions + AME (2 plots total)
#   (B) Empirical expectation over observed region x year rows + AME (2 plots total)

library(pacman)
p_load(
  tidyverse,
  here,
  brms,
  tidybayes,
  ggdist,
  ggthemes,
  ggokabeito
)

# configuration
fits_dir  <- here("data", "mdl_fits")
input_dir <- here("data", "mdl_input")
out_dir   <- here("figures", "grand_mean_summary")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stopifnot(dir.exists(out_dir))

YEAR0  <- 0
EMP_N  <- 300   # sample rows per marker for empirical expectation (speed)
SEED   <- 1

# pretty markers
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

marker_label <- function(marker) {
  x <- pretty_marker[[marker]]
  if (is.null(x) || is.na(x)) marker else x
}

# labels and sizes
theme_size <- function(base = 13) {
  theme(
    plot.title    = element_text(size = base + 4, face = "bold"),
    plot.subtitle = element_text(size = base + 1),
    
    axis.title.x  = element_text(size = base + 2),
    axis.title.y  = element_text(size = base + 2),
    
    axis.text.x   = element_text(size = base),
    axis.text.y   = element_text(size = base),
    
    legend.title  = element_text(size = base),
    legend.text   = element_text(size = base),
    
    plot.margin   = margin(10, 14, 10, 14)  # top, right, bottom, left
  )
}

# load data and fits into one object
discover_markers <- function(fits_dir) {
  fit_files <- list.files(fits_dir, pattern = "\\.rds$", full.names = TRUE)
  stopifnot(length(fit_files) > 0)
  fit_files |>
    basename() |>
    str_replace("\\.rds$", "") |>
    sort()
}

load_all_markers <- function(markers, fits_dir, input_dir) {
  out <- list()
  
  for (m in markers) {
    fit_path <- file.path(fits_dir, paste0(m, ".rds"))
    in_path  <- file.path(input_dir, paste0(m, ".csv"))
    
    if (!file.exists(in_path)) {
      warning("Missing input csv for marker: ", m, " (", in_path, "). Skipping.")
      next
    }
    
    fit <- readRDS(fit_path)
    dat <- readr::read_csv(in_path, show_col_types = FALSE)
    
    # minimal sanity
    stopifnot(all(dat$violent_external %in% c(0, 1)))
    stopifnot(is.numeric(dat$year_scaled))
    stopifnot(!anyNA(dat$world_region))
    
    out[[m]] <- list(
      marker = m,
      marker_label = marker_label(m),
      fit = fit,
      dat = dat,
      n = nrow(dat)
    )
  }
  
  stopifnot(length(out) > 0)
  out
}

markers <- discover_markers(fits_dir)
obj <- load_all_markers(markers, fits_dir, input_dir)

# let us order by the size of the effect here, if possible
# that would mean doing it after AME.
order_df <- tibble(
  marker = names(obj),
  marker_label = map_chr(obj, ~ .x$marker_label),
  n = map_int(obj, ~ .x$n)
) %>%
  arrange(desc(n), marker)

marker_levels <- order_df$marker

# display labels keyed by unique marker key
marker_labels_n <- setNames(
  paste0(order_df$marker_label, " (n=", order_df$n, ")"),
  order_df$marker
)

# calculate grand mean and AME
calc_grand_mean_draws <- function(fit, n_total, marker, marker_label, year0 = 0) {
  gm <- fit %>%
    epred_draws(
      newdata = tidyr::expand_grid(
        violent_external = c(0, 1),
        year_scaled = year0
      ),
      re_formula = NA
    ) %>%
    ungroup() %>%   # important: drop any lingering grouping from tidybayes
    mutate(
      marker = marker,
      marker_label = marker_label,
      n = n_total,
      violent_external = factor(
        violent_external,
        levels = c(0, 1),
        labels = c("No", "Yes")
      )
    ) %>%
    select(marker, marker_label, n, .draw, violent_external, .epred)
  
  ame <- gm %>%
    group_by(marker, marker_label, n, .draw) %>%
    summarise(
      ame = .epred[violent_external == "Yes"] - .epred[violent_external == "No"],
      .groups = "drop"
    )
  
  list(gm = gm, ame = ame)
}

grand <- purrr::imap(obj, ~ calc_grand_mean_draws(
  fit = .x$fit,
  n_total = .x$n,
  marker = .x$marker,
  marker_label = .x$marker_label,
  year0 = YEAR0
))

grand_mean_all <- purrr::map_dfr(grand, "gm") %>%
  mutate(marker = factor(marker, levels = marker_levels))

grand_ame_all <- purrr::map_dfr(grand, "ame") %>%
  mutate(marker = factor(marker, levels = marker_levels))

# plot grand mean and AME.

plot_grand_mean <- function(df, y_labels, year0 = 0) {
  
  ggplot(df, aes(x = .epred, y = marker, fill = violent_external)) +
    stat_halfeye(alpha = 0.9, normalize = "groups", adjust = 0.8) +
    scale_y_discrete(labels = y_labels) +
    scale_fill_okabe_ito() +
    theme_clean() +
    theme_size(base = 13) +
    labs(
      title = "Posterior mean probability by marker",
      subtitle = paste0("Population-level (no region RE) at average year"),
      x = "Expected probability (E[Y | X])",
      y = NULL,
      fill = "Violent external conflict"
    ) +
    theme(legend.position = "bottom")
}

plot_grand_ame <- function(df, y_labels, year0 = 0) {
  
  ggplot(df, aes(x = ame, y = marker)) +
    stat_halfeye(alpha = 0.95, adjust = 0.8, fill = palette_okabe_ito(order = 7)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3) +
    scale_y_discrete(labels = y_labels) +
    theme_clean() +
    theme_size(base=13) +
    labs(
      title = "Average marginal effect (Yes - No) by marker",
      subtitle = paste0("AME with no region RE and fixed year"),
      x = "AME in expected probability (E[Y|Yes] - E[Y|No])",
      y = NULL
    ) +
    theme(legend.position = "none")
}

p_grand_mean <- plot_grand_mean(grand_mean_all, marker_labels_n, YEAR0)
p_grand_ame  <- plot_grand_ame(grand_ame_all, marker_labels_n, YEAR0)

ggsave(file.path(out_dir, "ALL_markers__grand_mean_predictions.pdf"),
       p_grand_mean, width = 9.5, height = 6.5, device = "pdf")
ggsave(file.path(out_dir, "ALL_markers__grand_mean_predictions.png"),
       p_grand_mean, width = 9.5, height = 6.5, dpi = 300)

ggsave(file.path(out_dir, "ALL_markers__grand_AME.pdf"),
       p_grand_ame, width = 9.5, height = 6.5, device = "pdf")
ggsave(file.path(out_dir, "ALL_markers__grand_AME.png"),
       p_grand_ame, width = 9.5, height = 6.5, dpi = 300)


# empirical distribution
calc_empirical_draws <- function(fit, dat, n_total, marker, marker_label,
                                 emp_n = 300, seed = 1, re_form = NULL) {
  # observed covariate rows we want to average over
  nd_emp <- dat %>% transmute(world_region, year_scaled)
  
  set.seed(seed)
  nd_emp_s <- nd_emp %>% slice_sample(n = min(emp_n, nrow(nd_emp)))
  
  # posterior draws of conditional mean E[Y|X] at each observed row, for vx=0/1
  ep_emp <- fit %>%
    epred_draws(
      newdata = tidyr::expand_grid(
        violent_external = c(0, 1),
        nd_emp_s
      ),
      re_formula = NULL
    ) %>%
    ungroup() %>%
    mutate(
      marker = marker,
      marker_label = marker_label,
      n = n_total,
      violent_external = factor(
        violent_external,
        levels = c(0, 1),
        labels = c("No", "Yes")
      )
    )
  
  # average over sampled observed rows within each draw and condition
  ep_emp_bar <- ep_emp %>%
    group_by(marker, marker_label, n, .draw, violent_external) %>%
    summarise(.epred = mean(.epred), .groups = "drop") %>%
    select(marker, marker_label, n, .draw, violent_external, .epred)
  
  # AME per draw (no pivot_wider; safe)
  ame_emp <- ep_emp_bar %>%
    group_by(marker, marker_label, n, .draw) %>%
    summarise(
      ame = .epred[violent_external == "Yes"] - .epred[violent_external == "No"],
      .groups = "drop"
    )
  
  list(ep = ep_emp_bar, ame = ame_emp)
}

emp <- purrr::imap(obj, ~ calc_empirical_draws(
  fit = .x$fit,
  dat = .x$dat,
  n_total = .x$n,
  marker = .x$marker,
  marker_label = .x$marker_label,
  emp_n = EMP_N,
  seed = SEED,
  re_form = EMP_RE_FORMULA
))

emp_mean_all <- purrr::map_dfr(emp, "ep") %>%
  mutate(marker = factor(marker, levels = marker_levels))

emp_ame_all <- purrr::map_dfr(emp, "ame") %>%
  mutate(marker = factor(marker, levels = marker_levels))

# -------------------------
# 5) Plot empirical expectation + AME and save
# -------------------------
plot_emp_mean <- function(df, y_labels, re_form, emp_n, seed) {
  ggplot(df, aes(x = .epred, y = marker, fill = violent_external)) +
    stat_halfeye(alpha = 0.9, normalize = "groups", adjust = 0.8) +
    scale_y_discrete(labels = y_labels) +
    scale_fill_okabe_ito() +
    theme_clean() +
    theme_size(base=13) +
    labs(
      title = "Expected probability by marker",
      subtitle = "Average over empirical time x region observations (incl. RE)",
      x = "Expected probability (E[Y | X])",
      y = NULL,
      fill = "Violent external conflict"
    ) +
    theme(legend.position = "bottom")
}

plot_emp_ame <- function(df, y_labels, re_form, emp_n, seed) {
  ggplot(df, aes(x = ame, y = marker)) +
    stat_halfeye(alpha = 0.95, adjust = 0.8, fill = palette_okabe_ito(order = 7)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.3) +
    scale_y_discrete(labels = y_labels) +
    theme_clean() +
    theme_size(base=13) +
    labs(
      title = "Average marginal effect (Yes - No) by marker",
      subtitle = "Average over empirical time x region observations (incl. RE)",
      x = "AME in expected probability (E[Y|Yes] - E[Y|No])",
      y = NULL
    ) +
    theme(legend.position = "none")
}

p_emp_mean <- plot_emp_mean(emp_mean_all, marker_labels_n, EMP_RE_FORMULA, EMP_N, SEED)
p_emp_ame  <- plot_emp_ame(emp_ame_all, marker_labels_n, EMP_RE_FORMULA, EMP_N, SEED)

# filenames reflect whether RE included
re_tag <- "with_region_RE"
ggsave(file.path(out_dir, paste0("ALL_markers__empirical_mean__", re_tag, ".pdf")),
       p_emp_mean, width = 9.5, height = 6.5, device = "pdf")
ggsave(file.path(out_dir, paste0("ALL_markers__empirical_mean__", re_tag, ".png")),
       p_emp_mean, width = 9.5, height = 6.5, dpi = 300)

ggsave(file.path(out_dir, paste0("ALL_markers__empirical_AME__", re_tag, ".pdf")),
       p_emp_ame, width = 9.5, height = 6.5, device = "pdf")
ggsave(file.path(out_dir, paste0("ALL_markers__empirical_AME__", re_tag, ".png")),
       p_emp_ame, width = 9.5, height = 6.5, dpi = 300)

