def fit_region_logistic_model(data, outcome):

    with pm.Model() as model:
        # Hyperpriors
        mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=5)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=5)

        # Random intercepts for regions
        intercepts = pm.Normal(
            "intercepts",
            mu=mu_intercept,
            sigma=sigma_intercept,
            shape=len(data["world_region"].unique()),
        )

        # Predictors and other parameters
        b_external = pm.Normal("b_external", mu=0, sigma=10)
        region_idx = pm.Data("region_idx", data["world_region"].cat.codes)

        # Logit function
        logit_p = intercepts[region_idx] + b_external * data["violent external"]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome])

        trace = pm.sample(draws=2000, tune=2000, target_accept=0.99)
        return trace


def fit_region_logistic_model_nonc(data, outcome):
    with pm.Model() as model:
        # Hyperpriors
        mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=5)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=5)

        # Non-centered reparameterization of random intercepts
        offsets = pm.Normal(
            "offsets", mu=0, sigma=1, shape=len(data["world_region"].unique())
        )
        intercepts = pm.Deterministic(
            "intercepts", mu_intercept + sigma_intercept * offsets
        )

        # Predictors and other parameters
        b_external = pm.Normal("b_external", mu=0, sigma=5)
        region_idx = pm.Data("region_idx", data["world_region"].cat.codes)

        # Logit function
        logit_p = intercepts[region_idx] + b_external * data["violent external"].astype(
            float
        )
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome].astype(int))

        trace = pm.sample(draws=2000, tune=2000, target_accept=0.99)
        return trace


def fit_region_logistic_random(data, outcome):
    with pm.Model() as model:
        # Hyperpriors for intercepts
        mu_intercept = pm.Normal("mu_intercept", mu=0, sigma=5)
        sigma_intercept = pm.HalfCauchy("sigma_intercept", beta=5)

        # Hyperpriors for slopes
        mu_slope = pm.Normal("mu_slope", mu=0, sigma=5)
        sigma_slope = pm.HalfCauchy("sigma_slope", beta=5)

        # Random intercepts for regions
        intercepts = pm.Normal(
            "intercepts",
            mu=mu_intercept,
            sigma=sigma_intercept,
            shape=len(data["world_region"].unique()),
        )

        # Random slopes for regions
        slopes = pm.Normal(
            "slopes",
            mu=mu_slope,
            sigma=sigma_slope,
            shape=len(data["world_region"].unique()),
        )

        # Data indexing for regions
        region_idx = pm.Data("region_idx", data["world_region"].cat.codes)

        # Logit function incorporating random slopes
        logit_p = intercepts[region_idx] + slopes[region_idx] * data[
            "violent external"
        ].astype(float)
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome].astype(int))

        # Sampling
        trace = pm.sample(draws=2000, tune=2000, target_accept=0.99)
        return trace
