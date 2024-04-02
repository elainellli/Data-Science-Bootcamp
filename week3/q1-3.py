import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD'
df = pd.read_csv(url)

df.head()
df['hour_beginning'] = pd.to_datetime(df['hour_beginning'])

df.info
df['hour_beginning'].head(5)
df['hour'] = df['hour_beginning'].dt.hour
df['month'] = df['hour_beginning'].dt.month
df['year'] = df['hour_beginning'].dt.year
df['date'] = df['hour_beginning'].dt.date
df['day_name'] = df['hour_beginning'].dt.day_name()
df.head()

weekdays = df.loc[df['day_name'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]
weekdays.head()

pedestrians_each_day = weekdays.groupby('day_name')['Pedestrians'].sum()
pedestrians_each_day.head()

plt.figure(figsize=(12, 6))
pedestrians_each_day.plot(kind='line')
plt.title('Pedestrian Count each Day')
plt.xlabel('day_name')
plt.ylabel('Pedestrian Count')
plt.grid(True)
plt.tight_layout()
plt.show()

data2019 = df.loc[(df['year'] == 2019) & (df['location']=='Brooklyn Bridge')]
data2019.head()

weather = pd.get_dummies(data2019['weather_summary'])
weather.head()
pedestrians_weather = pd.concat([data2019['Pedestrians'], weather])
correlation_matrix = pedestrians_weather.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Pedestrians and Temperature')
plt.tight_layout()
plt.show()

def time_categorize(hour):
    if (6<=hour<12):
        return 'morning'
    elif (12<=hour<17):
        return 'afternoon'
    elif (17<=hour<20):
        return 'evening'
    else:
        return 'night'

df['time_of_day'] = df['hour'].apply(time_categorize)

pedestrians_over_time = df.groupby('time_of_day')['Pedestrians'].sum()

print(pedestrians_over_time)
