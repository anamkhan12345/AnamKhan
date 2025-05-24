import pandas as pd
import numpy as np
import pdb


# Explore Data
debug_explore = False
df = pd.read_csv("../../dataSets/airline-delay/Airline_Delay_Cause.csv")
sample = df[ (df['month'] == 2) & (df['year'] == 2016) ]
sample = sample.sort_values(['airport'])
american = df[ (df['carrier'] == 'AA')]

if debug_explore:
    print("View of American Arrivals to airpots")
    print(american.head())
    print("Looking at all arrivals in the 2/2016")
    print(sample.head())
    print("**********")

# Clean the data up - find any missing values or incomplete values
total_data = np.prod(df.shape)
missing = df[df.isnull().any(axis=1)]
missing_sum = df.isnull().sum()
missing = missing.fillna('missing')

# Find if all 240 missing values are for the same rows....
missing_sum = missing_sum[missing_sum == 240].index.tolist()
common_miss = df[missing_sum]
common_miss = common_miss[common_miss.isnull().any(axis=1)]
same_miss = (common_miss.nunique(axis=1) == 1)
all_rows_same = same_miss.nunique() == 1

# I want to know the worst airports to fly into on average over the entire data timeframe
worst_carrier_delays = df.groupby(['airport','airport_name'])['carrier_delay'].mean() # TODO: Probably need to take into account total delays to total arrivals
worst_carrier_delays = worst_carrier_delays.sort_values(ascending=False)
#print(worst_carrier_delays.head())

# I want to know for each month, which is the worst airport to fly into
#   - what are the top 3 worst carriers on those months

# I want to know the worst carriers on average

