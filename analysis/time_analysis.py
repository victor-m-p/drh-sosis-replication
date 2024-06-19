import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
answers = pd.read_csv("../data/preprocessed/answers_conflict.csv")
entries = pd.read_csv("../data/preprocessed/entry_data.csv")

# just take out one question
question = "food_taboos"  # "extra_ritual_group_markers"
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
wr = world_regions[0]
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    data=df_time_intervals[df_time_intervals["world_region"] == wr],
    x="time_bin",
    y="violent_external",
    color="tab:blue",
)
sns.lineplot(
    data=df_time_intervals[df_time_intervals["world_region"] == wr],
    x="time_bin",
    y=question,
    color="tab:orange",
)

# really looks like these are just different types of cultures
# then there are sometimes fewer and sometimes more of these
# not clear to me that we are observing cultures experiencing
# a decline in warfare and subsequently a decline in practices.
