# now we do the same but control for time
entry_time = pd.read_csv("../data/preprocessed/entry_time.csv")
answers_wide_time = answers_wide.merge(entry_time, on="entry_id", how="inner")


def preprocess_data_time(data, id, predictor, outcome, time):
    data_subset = data[[id, predictor, outcome, time]]
    data_subset = data_subset.dropna()
    data_subset[predictor] = data_subset[predictor].astype(int)
    data_subset[outcome] = data_subset[outcome].astype(int)
    data_subset[time] = data_subset[time].astype(int)
    data_subset["time_scaled"] = (
        data_subset[time] - data_subset[time].mean()
    ) / data_subset[time].std()
    return data_subset


def fit_time_logistic_pymc(data, predictor, outcome, time):
    with pm.Model() as model:
        prior_external = pm.Normal("prior_external", mu=0, sigma=5)
        prior_time = pm.Normal("prior_time", mu=0, sigma=5)
        logit_p = prior_external * data[predictor] + prior_time * data[time]
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        observed = pm.Binomial("y", n=1, p=p, observed=data[outcome])
        trace = pm.sample(2000, return_inferencedata=True)
    return trace


# circumcision
circumcision = preprocess_data_time(
    answers_wide_time, "entry_id", "violent external", "circumcision", "year_from"
)
circumcision_trace = fit_time_logistic_pymc(
    circumcision, "violent external", "circumcision", "time_scaled"
)
circumcision_summary = summary_percent(
    circumcision_trace, var_names=["prior_external", "prior_time"]
)
circumcision_summary

logistic_to_percent()

trace = az.summary(trace, var_names=["prior_external", "prior_time"], hdi_prob=0.95)
