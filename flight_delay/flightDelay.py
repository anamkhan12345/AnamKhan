import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Figure out the percentage of missing cells to total cells
missing = df[df.isnull().any(axis=1)]
missing_sum = df.isnull().sum()
total_missing = missing_sum.sum()
total_cells = np.prod(df.shape)
percent_cells_missing = (total_missing/total_cells) * 100
print("Percent Missing: ", percent_cells_missing)

# Figure out how many % of rows are removed
df_dropped = df.dropna()
percent_rows_dropped = ( df_dropped.shape[0] / df.shape[0] ) * 100
print(" Percent of original df after rows dropped: ", percent_rows_dropped)

# Does the dropped data effect specific carriers more than others?
old_carriers = df.carrier_name.value_counts()
new_carriers = df_dropped.carrier_name.value_counts()
percent_change_carriers =  ( (old_carriers - new_carriers) / old_carriers ) * 100
print("Carrier with most change: ", percent_change_carriers.idxmax())
print("Carrier change percentage: ", percent_change_carriers.max())

plt.bar(percent_change_carriers.index, percent_change_carriers)
plt.xlabel('Carriers')
plt.ylabel('Percent Change')
plt.title('Percent Change of Carriers')
plt.xticks(fontsize=6)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Does the dropped data effect specific airports more than others?
old_airports = df.airport_name.value_counts()
new_airports = df_dropped.airport_name.value_counts()
percent_change_airports =  ( (old_airports - new_airports) / old_airports ) * 100
percent_change_airports = percent_change_airports[percent_change_airports > 1]
print("Airport with most change: ", percent_change_airports.idxmax())
print("Airport change percentage: ", percent_change_airports.max())

plt.bar(percent_change_airports.index, percent_change_airports)
plt.xlabel('Airports')
plt.ylabel('Percent Change')
plt.title('Percent Change of Airports')
plt.xticks(fontsize=6)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Worst is for Mobile (AL), a 6.5% drop, but I think that's statistically insignificant...
# Replace NaN with missing
missing = missing.fillna('missing')


missing_carriers_list = missing.carrier_name.value_counts().index.tolist()
all_missing_carriers_data = df[df.carrier_name.isin(missing_carriers_list)]
breakpoint()

# Find if all 240 missing values are for the same rows....
missing_sum = missing_sum[missing_sum == 240].index.tolist()
common_miss = df[missing_sum]
common_miss = common_miss[common_miss.isnull().any(axis=1)]
same_miss = (common_miss.nunique(axis=1) == 1)
all_rows_same = same_miss.nunique() == 1

breakpoint()
# I want to know the worst airports to fly into on average over the entire data timeframe
worst_carrier_delays = df.groupby(['airport','airport_name'])['carrier_delay'].mean() # TODO: Probably need to take into account total delays to total arrivals
worst_carrier_delays = worst_carrier_delays.sort_values(ascending=False)
#print(worst_carrier_delays.head())

# I want to know for each month, which is the worst airport to fly into
#   - what are the top 3 worst carriers on those months

# I want to know the worst carriers on average

