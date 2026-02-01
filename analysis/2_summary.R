# analysis/R/02_extract_summaries.R
library(pacman)
p_load(brms, tidyverse, here, posterior)

fit_dir <- here("data", "mdl_fits")
out_dir <- here("data", "mdl_output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

fit_files <- list.files(fit_dir, pattern = "\\.rds$", full.names = TRUE)

all_fixed <- list()
all_diag  <- list()
all_hyp   <- list()

for (f in fit_files) {
  marker <- tools::file_path_sans_ext(basename(f))
  fit <- readRDS(f)
  
  # Fixed effects
  fe <- as.data.frame(fixef(fit)) |>
    rownames_to_column("term") |>
    mutate(marker = marker)
  
  # Diagnostics (summaries)
  summ <- posterior::summarise_draws(as_draws_df(fit))
  # filter to sampler diagnostics if you want; or do:
  # brms has:
  nh <- nuts_params(fit)
  div_rate <- mean(nh$Parameter == "divergent__" & nh$Value == 1)
  
  diag <- tibble(
    marker = marker,
    n_obs = nrow(fit$data),
    n_regions = nlevels(fit$data$world_region),
    divergent_rate = div_rate
  )
  
  # Posterior prob beta_conflict > 0 (direct, transparent)
  draws <- as_draws_df(fit)
  p_gt0 <- mean(draws$b_violent_external > 0)
  
  hyp <- tibble(
    marker = marker,
    p_beta_conflict_gt0 = p_gt0
  )
  
  all_fixed[[marker]] <- fe
  all_diag[[marker]]  <- diag
  all_hyp[[marker]]   <- hyp
}

bind_rows(all_fixed) |> write_csv(file.path(out_dir, "fixed_effects.csv"))
bind_rows(all_diag)  |> write_csv(file.path(out_dir, "diagnostics.csv"))
bind_rows(all_hyp)   |> write_csv(file.path(out_dir, "posterior_probs.csv"))
