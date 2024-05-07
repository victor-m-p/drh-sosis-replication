"""
Willis has the idea for this study.
Does not seem like there is an effect.
We might want to delete this and focus on replication.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/preprocessed/answers_cremation.csv")

# first focus on super-questions
df_subset = df[df["question_id"].isin([4776, 4795])]

# to wide
df_wide = df_subset.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)
df_wide = df_wide.dropna()

# recode 1-0 to Yes-No for visual clarity
df_wide["spirit-body distinction"] = np.where(
    df_wide["spirit-body distinction"] == 1, "Yes", "No"
)

# make a simple plot
fig, ax = plt.subplots(figsize=(8, 4))
sns.set_style("white")
sns.barplot(
    x="spirit-body distinction",
    y="Cremation",
    data=df_wide,
    ax=ax,
)

"""
Does not look like we have an effect above. 
Large error-bar for no spirit-body distinction because we have many fewer answers there.
"""

df_subset = df[df["question_id"].isin([4778, 4795])]
df_wide = df_subset.pivot(
    index="entry_id", columns="question_short", values="answer_value"
)
df_wide = df_wide.dropna()

# recode 1-0 to Yes-No for visual clarity
df_wide["ontologically distinct"] = np.where(
    df_wide["ontologically distinct"] == 1, "Yes", "No"
)

# make a simple plot
fig, ax = plt.subplots(figsize=(8, 4))
sns.set_style("white")
sns.barplot(
    x="ontologically distinct",
    y="Cremation",
    data=df_wide,
    ax=ax,
)
