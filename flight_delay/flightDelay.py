import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pdb
from sklearn.preprocessing import MinMaxScaler


def group_delay_scaled(feature, df):
    group = df.groupby(feature).agg(
        TOTAL_FLIGHTS_count = ('arr_flights', 'sum'),
        TOTAL_ARR_DELAY_count = ('arr_del15', 'sum'),
        AVG_ARR_DELAY_mins = ('arr_delay', 'mean'),
        NUM_CANCELLED_count = ('arr_cancelled', 'sum'),
        NUM_DIVERTED_count = ('arr_diverted', 'sum')
    )

    # Step 3: Calculate derived metrics
    group['CANCEL_RATE'] = group['NUM_CANCELLED_count'] / group['TOTAL_FLIGHTS_count']
    group['DIVERT_RATE'] = group['NUM_DIVERTED_count'] / group['TOTAL_FLIGHTS_count']
    group['DELAY_PER_FLIGHT'] = group['TOTAL_ARR_DELAY_count'] / group['TOTAL_FLIGHTS_count']


    # Scale all 4 bad-performance indicators
    scaler = MinMaxScaler()
    metrics = ['AVG_ARR_DELAY_mins', 'CANCEL_RATE', 'DIVERT_RATE', 'DELAY_PER_FLIGHT']
    scaled_vars = ['S_ARR_DELAY_15', 'S_CANCEL', 'S_DIVERT', 'S_DELAY_PER_FLIGHT']
    scaled_metrics = scaler.fit_transform(group[metrics])
    group[scaled_vars] = scaled_metrics

    group['COMPOSITE_SCORE'] = (
        .4 * group['S_ARR_DELAY_15'] +
        .3 * group['S_DELAY_PER_FLIGHT'] +
        .2 * group['S_CANCEL'] +
        .1 * group['S_DIVERT']
    )

    top5_worst = group.sort_values('COMPOSITE_SCORE', ascending=False).head(5)
    print(top5_worst)

    return group

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
df['Year'] = df['combined_date'].dt.year

# More Data Cleaning - scaling or normalizing data?
numeric_cols = df.select_dtypes(include='number').columns
numeric_cols = numeric_cols.drop(['Year'])

if debug_explore:
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=30, edgecolor='black')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# Focus on arrival dealys to track airline and airport performance
carrier_group = group_delay_scaled('carrier_name', df).reset_index()
airport_group = group_delay_scaled('airport', df).reset_index()
airport_group.rename(columns={'airport': 'iata_code'}, inplace=True)

plt_cols = ['airport_name', 'iata_code', 'latitude_deg', 'longitude_deg', 'COMPOSITE_SCORE']

# Combine lat long data
df_loc = pd.read_csv(r"C:\Users\anamk\projects\dataSets\airline-delay\airports_lat_long.csv")
merge_cols = ['iata_code', 'latitude_deg', 'longitude_deg']
df_merged = pd.merge(airport_group, df_loc[merge_cols], on='iata_code', how='left')

# Create bubble map
fig = px.scatter_geo(
        df_merged,
        lat='latitude_deg',
        lon='longitude_deg',
        text='iata_code',
        size='COMPOSITE_SCORE',  # Bubble size based on value
        color='COMPOSITE_SCORE',       # color gradient
        color_continuous_scale='Viridis',  # or 'Plasma', 'Inferno', 'Turbo', etc.
        projection='albers usa',
        title='Bubble Chart of U.S. Airport Average Delay times from 2013- 2023',
        scope='usa'
    )
fig.show()

#if debug_explore:
    # # ORD Bar Chart 
    # dfORD = scaled_df[scaled_df['iata_code'] == 'ORD']
    # avg_score_per_year = dfORD.groupby('Year')['impact_score'].mean().reset_index()
    # plt.figure()
    # plt.title('ORD delay over time')
    # sns.barplot(x=avg_score_per_year.Year, y = avg_score_per_year.impact_score)
    # plt.ylabel('Delay Impact Score (scaled)')
    # plt.show()

    # # Heatmap
    # pivot = scaled_df.pivot_table(index='carrier_name', columns='Year', values='impact_score', aggfunc='mean')
    # plt.figure()
    # plt.title('Average Delay Score per airline over time')
    # sns.heatmap(data=pivot, annot=True, cmap='Reds')
    # plt.show()