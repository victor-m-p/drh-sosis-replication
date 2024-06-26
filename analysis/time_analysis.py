import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
answers = pd.read_csv("../data/preprocessed/answers_clean.csv")
entries = pd.read_csv("../data/preprocessed/entries_clean.csv")

# just take out one question
question = "permanent_scarring"
answers = answers[answers["question_short"].isin(["violent_external", question])]
answers = answers[["entry_id", "question_short", "answer_value"]]
answers = answers.pivot(
    index="entry_id", columns="question_short", values="answer_value"
).reset_index()
answers = answers.dropna()

# merge to get year and world region
entries = entries[["entry_id", "year_from", "year_to", "world_region"]]
answers = pd.merge(answers, entries, on="entry_id")


# plot this for all groups;
# smooth time intervals
def smooth_time_intervals(df, bin_width, step_size):
    df_smoothed = pd.DataFrame()
    min_year, max_year = (
        df["year_from"].min(),
        df["year_to"].max(),
    )
    adjusted_max_year = max_year - bin_width + 1
    bins = range(min_year, adjusted_max_year + 1, step_size)

    for start in bins:
        end = start + bin_width
        mask = (df["year_from"] <= end) & (df["year_to"] >= start)
        temp_df = df.loc[mask].copy()
        temp_df["time_bin"] = np.mean([start, end])
        df_smoothed = pd.concat([df_smoothed, temp_df])

    return df_smoothed


df_time_intervals = smooth_time_intervals(answers, 100, 20)

# plot the data
world_regions = ["Europe", "Africa", "South Asia", "Southwest Asia"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
fig.suptitle(
    "Violent External Conflict and Permanent Scarring Across Regions",
    fontsize=16,
)

for ax, wr in zip(axes.flatten(), world_regions):
    sns.lineplot(
        data=df_time_intervals[df_time_intervals["world_region"] == wr],
        x="time_bin",
        y="violent_external",
        color="tab:blue",
        label="Violent External Conflict",
        ax=ax,
    )
    sns.lineplot(
        data=df_time_intervals[df_time_intervals["world_region"] == wr],
        x="time_bin",
        y=question,
        color="tab:orange",
        label="Permanent Scarring",
        ax=ax,
    )
    ax.set_title(wr)
    ax.set_ylabel("Fraction of Cultures")
    ax.set_xlabel("Time (100 year bins)")
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect to fit the suptitle
plt.show()

# really looks like these are just different types of cultures
# then there are sometimes fewer and sometimes more of these
# not clear to me that we are observing cultures experiencing
# a decline in warfare and subsequently a decline in practices.
