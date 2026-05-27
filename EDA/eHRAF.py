import pandas as pd 

# load data for tattoos/scarification
df = pd.read_csv("../data/model/external/input/tattoos_scarification.csv")

# load data on source
entry_data = pd.read_csv("../data/raw/entry_data.csv")
entry_data = entry_data[['entry_id', 'entry_name', 'data_source']].drop_duplicates()

# inner join
df = df.merge(entry_data, on='entry_id', how='inner')

# look at only eHRAF
df = df[df['data_source'] == 'eHRAF']