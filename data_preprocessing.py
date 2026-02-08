import pandas as pd

# Load original dataset
file_path = "dataset/date_sorted_tourists.csv"
df = pd.read_csv(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Add Global_Event feature
df['Global_Event'] = 0
df.loc[(df['Date'] >= '2015-04-01') & (df['Date'] <= '2015-05-31'), 'Global_Event'] = 1
df.loc[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2021-12-31'), 'Global_Event'] = 2

# Fill missing values if any
df['Value'].interpolate(method='linear', inplace=True)

# Save the new dataset
output_file = "dataset/tourists_with_global_event.csv"
df.to_csv(output_file, index=False)

print(f"File saved successfully as {output_file}")
