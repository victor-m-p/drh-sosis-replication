library(pacman)
p_load(brms, tidyverse, HDInterval, stringr)

# Set the directory containing the CSV files
directory_path <- "data/mdl_input"

# List all CSV files in the directory
file_paths <- list.files(path = directory_path, pattern = "\\.csv$", full.names = TRUE)

# Loop over the file paths and process each file
for (file_path in file_paths) {
    # Read the data
    data <- read_csv(file_path)
    colnames(data)
    predictor_name <- strsplit(basename(file_path), "\\.csv$", `[`, 1)[[1]]
    data <- data |> rename(dv = predictor_name)
    
    # Construct the formula dynamically
    #formula_str <- paste("violent_external ~ 1 +", predictor_name, "+ time_scaled + (1 +", predictor_name, "|world_region)")
    #formula <- as.formula(formula_str)
    
    # fit the model
    fit <- brm(
        data = data,
        family = bernoulli(link=logit),
        #formula = formula,
        dv ~ 1 + violent_external + year_scaled + (1+violent_external|world_region), 
        prior = c(
            prior(normal(0, 5), class=b),
            prior(normal(0, 5), class=Intercept),
            prior(lkj_corr_cholesky(1), class=L),
            prior(normal(0, 5), class=sd)
        ),
        iter = 8000,
        warmup = 4000,
        chains = 4,
        cores = 4,
        control = list(adapt_delta = .999, max_treedepth = 20),
        seed = 1342)
    # Extract the model summary
    model_summary <- summary(fit)
    coef_summary <- as.data.frame(model_summary$fixed)
    coef_summary$parameter <- rownames(coef_summary)
    write_csv(coef_summary, file = paste0("data/mdl_output/", predictor_name, "_summary.csv"))
    # Extract samples for conversion to natural scale
    alpha_samples <- as_draws_df(fit)$b_Intercept
    beta_samples <- as_draws_df(fit)$b_violent_external
    alpha_converted <- plogis(alpha_samples)
    beta_converted <- plogis(alpha_samples + beta_samples)
    effect <- beta_converted - alpha_converted
    # Collect the draws
    draws <- tibble(
        intercept = alpha_converted, 
        beta = beta_converted, 
        effect = effect
    )
    write_csv(draws, file = paste0("data/mdl_output/", predictor_name, "_draws.csv"))
    # Summarize the results
    results <- tibble(
        parameter = c("intercept", "beta", "effect"),
        Estimate = c(mean(alpha_converted), mean(beta_converted), mean(effect)),
        `l-95% CI` = c(quantile(alpha_converted, 0.025), quantile(beta_converted, 0.025), quantile(effect, 0.025)),
        `u-95% CI` = c(quantile(alpha_converted, 0.975), quantile(beta_converted, 0.975), quantile(effect, 0.975))
    )
    write_csv(results, file = paste0("data/mdl_output/", predictor_name, "_results.csv"))
}