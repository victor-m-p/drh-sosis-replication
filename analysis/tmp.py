import pandas as pd
import numpy as np

answers = pd.read_csv("../data/preprocessed/answers.csv")
answers.groupby(["question_id", "question_name"]).size()
