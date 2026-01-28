import pandas as pd

data = pd.read_csv("dataset/initial_data.csv")

data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%b')

data_sorted = data.sort_values('Date')

data_sorted = data_sorted[['Year', 'Month', 'Value']]


data_sorted.to_csv("dataset/date_sorted_tourists.csv", index=False)