import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset/initial_data.csv")

plt.scatter(x=dataset["Month"], y= dataset["Value"])
plt.plot()
plt.show()

print(dataset)