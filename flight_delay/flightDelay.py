import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
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
print("Percent of original df after rows dropped: ", percent_rows_dropped)

# Does the dropped data effect specific carriers more than others?
old_carriers = df.carrier_name.value_counts()
new_carriers = df_dropped.carrier_name.value_counts()
percent_change_carriers =  ( (old_carriers - new_carriers) / old_carriers ) * 100
print("Carrier with most change: ", percent_change_carriers.idxmax())
print("Carrier change percentage: ", percent_change_carriers.max())
print("Carrier impact summary: ", percent_change_carriers.describe())

if debug_explore:
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

if debug_explore:
    plt.bar(percent_change_airports.index, percent_change_airports)
    plt.xlabel('Airports')
    plt.ylabel('Percent Change')
    plt.title('Percent Change of Airports')
    plt.xticks(fontsize=6)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Worst is for Mobile (AL), a 8.5% drop, I think it's ok
df = df_dropped

# More Data Cleaning - Duplicate rows?
dupes = df.duplicated().sum()
print("Duplicated rows: ", dupes) # None, so good to move on

# More Data Cleaning - set up dates in correct time format
df['combined_date'] = pd.to_datetime(df['month'].astype(str) + '-' + df['year'].astype(str), format='%m-%Y') 
df = df.drop(['month', 'year'], axis=1)
df = df[['combined_date'] + [col for col in df.columns if col != 'combined_date']]

# More Data Cleaning - scaling or normalizing data?
numeric_cols = df.select_dtypes(include='number').columns

if debug_explore:
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# After inspecting each features histogram, I don't think,
# any of them follow a normal distribution. Therefore, I 
# should use MinMaxScaler for my scaling type.
scaler = MinMaxScaler()
scaled_df = df.copy()
scaled_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Compute composit score per row
scaled_df['impact_score'] = scaled_df[numeric_cols].sum(axis=1)

# I want to know the worst airports to fly into on average over the entire data timeframe
scaled_df.rename(columns={'airport': 'iata_code'}, inplace=True)
carrier_delay_scores = scaled_df.groupby('carrier_name')['impact_score'].mean().sort_values(ascending=False)
airport_delay_scores = scaled_df.groupby(['iata_code'])['impact_score'].mean().sort_values(ascending=False)

print("Worst airport on average from 2013 - 2023: ", airport_delay_scores.idxmax())
print("Worst carrier on average from 2013 - 2023: ", carrier_delay_scores.idxmax())
plt_cols = ['airport_name', 'iata_code', 'latitude_deg', 'longitude_deg', 'impact_score']

# Combine lat long data
df_loc = pd.read_csv(r"C:\Users\anamk\projects\dataSets\airline-delay\airports_lat_long.csv")
merge_cols = ['iata_code', 'latitude_deg', 'longitude_deg']
df_delay = airport_delay_scores.to_frame()
df_merged = pd.merge(df_delay, df_loc[merge_cols], on='iata_code', how='left')

# Create bubble map
fig = px.scatter_geo(
    df_merged,
    lat='latitude_deg',
    lon='longitude_deg',
    text='iata_code',
    size='impact_score',  # Bubble size based on value
    projection='albers usa',
    title='Bubble Chart of U.S. Airport Average Delay times from 2013- 2023',
    scope='usa'
)

fig.update_traces(marker=dict(color='skyblue', line=dict(width=1, color='black')))
fig.show()

# I want to know for each month, which is the worst airport to fly into
#   - what are the top 3 worst carriers on those months

# I want to know the worst carriers on average

