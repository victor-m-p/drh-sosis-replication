# analysis/R/01_fit_models.R
library(pacman)
p_load(brms, tidyverse, here)

markers <- c(
  "permanent_scarring",
  "circumcision",
  "tattoos_scarification",
  "extra_ritual_group_markers",
  "food_taboos",
  "hair",
  "dress",
  "ornaments"
)

in_dir  <- here("data", "mdl_input")
fit_dir <- here("data", "mdl_fits")
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)

priors <- c(
  prior(normal(0, 5), class = b),
  prior(normal(0, 5), class = Intercept),
  prior(normal(0, 5), class = sd),
  prior(lkj_corr_cholesky(1), class = L)
)

for (marker in markers) {
  
  message("Fitting: ", marker)
  
  dat <- readr::read_csv(here("data", "mdl_input", paste0(marker, ".csv")),
                         show_col_types = FALSE)
  
  dat <- dat |>
    rename(dv = all_of(marker)) |>
    mutate(world_region = factor(world_region))
  
  fit <- brm(
    data = dat,
    family = bernoulli(link = "logit"),
    formula = dv ~ 1 + violent_external + year_scaled + (1 + violent_external | world_region),
    prior = priors,
    iter = 8000, warmup = 4000, chains = 4, cores = 4,
    seed = 1342,
    control = list(adapt_delta = 0.999, max_treedepth = 20),
    backend = "cmdstanr"
  )
  
  saveRDS(fit, file = file.path(fit_dir, paste0(marker, ".rds")))
}
