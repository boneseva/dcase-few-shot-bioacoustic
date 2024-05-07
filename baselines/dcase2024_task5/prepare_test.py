import pandas as pd
import os

input_file = 'C:/Users/Eva/Documents/VibroScape/Annotated recordings nopeaks/all_annotations.nomulti.16k.csv'

# read csv
df = pd.read_csv(input_file)
# only copy 2000 random rows
df = df.sample(n=10)

# save to new file
# delete file if it already exists
if os.path.exists('all_annotations.nomulti.16k.testing.csv'):
    os.remove('all_annotations.nomulti.16k.testing.csv')
df.to_csv('all_annotations.nomulti.16k.testing.csv', index=False)