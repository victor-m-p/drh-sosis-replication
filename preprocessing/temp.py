import pandas as pd 
data = pd.read_csv("../data/preprocessed/answerset.csv")
data.groupby('world_region').size().reset_index(name='count').sort_values('count', ascending=False)
data['year_from'].mean()
data['year_from'].median()
data['year_from'].min()
data[data['year_from'] < -3000]

entries = pd.read_csv("../data/raw/entry_data.csv")
entries[entries['entry_id'] == 284]
entries[entries['entry_id'] == 2286]