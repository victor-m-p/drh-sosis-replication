import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load
entry_time = pd.read_csv("../data/preprocessed/entry_time.csv")
answerset = pd.read_csv("../data/preprocessed/answers_conflict.csv")

# select and merge
answerset = answerset.merge(entry_time, on="entry_id", how="inner")


# heler function
def smooth_time_end(df, bin_width, step_size):
    df_smoothed = pd.DataFrame()
    min_year, max_year = df["year_from"].min(), df["year_to"].max()
    adjusted_max_year = max_year - bin_width + 1
    bins = range(min_year, adjusted_max_year + 1, step_size)

    for start in bins:
        end = start + bin_width
        # Adjusted condition for finding overlapping intervals
        mask = (df["year_from"] <= end) & (df["year_to"] >= start)
        temp_df = df.loc[mask].copy()
        temp_df["time_bin"] = np.mean([start, end])
        df_smoothed = pd.concat([df_smoothed, temp_df])

    return df_smoothed


bin_width = 300
step_size = 100
answerset_timebin = smooth_time_end(answerset, bin_width, step_size)

answerset_agg = (
    answerset_timebin.groupby(["time_bin", "question_short"])["answer_value"]
    .mean()
    .reset_index()
)
answerset_agg = answerset_agg.dropna()

# Take only from the time range
xmin = -2000
xmax = 2000

# Plot
plt.figure(figsize=(10, 4), dpi=300)
sns.lineplot(data=answerset_agg, x="time_bin", y="answer_value", hue="question_short")
plt.xticks(rotation=45)
plt.yticks()
plt.xlabel(f"Year", size=12)
plt.ylabel("Fraction Yes", size=12)
plt.xlim(xmin, xmax)
plt.ylim(0, 1)

plt.legend(
    fontsize=12,
    loc="upper center",
    bbox_to_anchor=(1.3, 1.05),
    ncol=1,
    frameon=False,
)

plt.tight_layout()

plt.savefig(
    f"../figures/supplementary/time_variables.pdf", dpi=300, bbox_inches="tight"
)

"""
# coverage
def smooth_time_bins(df, bins):
    df_smoothed = pd.DataFrame()
    bin_n = 0
    for start, end in bins:
        mask = (df["year_from"] <= end) & (df["year_to"] >= start)
        temp_df = df.loc[mask].copy()
        temp_df["time_bin"] = np.mean([start, end])
        temp_df["time_range"] = f"({start}, {end})"
        temp_df["bin_n"] = bin_n
        df_smoothed = pd.concat([df_smoothed, temp_df])
        bin_n += 1
    return df_smoothed


min_year_from = answerset["year_from"].min()
max_year_to = answerset["year_to"].max()
bins = [
    (min_year_from, -2000),
    (-2000, 0),
    (0, 1000),
    (1000, 2000),
    (2000, max_year_to),
]

answerset_bins = smooth_time_bins(answerset, bins)
answerset_bins = answerset_bins[["entry_id", "time_range", "bin_n"]].drop_duplicates()
count_bins = (
    answerset_bins.groupby(["time_range", "bin_n"]).size().reset_index(name="count")
)
count_bins = count_bins.sort_values("bin_n")

# Add x axis labels
bin_values_to_labels = count_bins[["bin_n", "time_range"]].drop_duplicates()
bin_values_to_labels = bin_values_to_labels.set_index("bin_n")["time_range"].to_dict()

# Plot
plt.figure(figsize=(8, 4), dpi=300)
sns.barplot(data=count_bins, x="bin_n", y="count")
plt.xticks(
    list(bin_values_to_labels.keys()), list(bin_values_to_labels.values()), rotation=45
)
plt.xlabel(f"Year", size=12)
plt.ylabel("Dimension weight", size=12)

plt.tight_layout()
answerset["year_from"].min()
answerset["year_to"].max()
"""
