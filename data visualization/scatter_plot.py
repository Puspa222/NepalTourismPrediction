import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("dataset/date_sorted_tourists.csv")

plt.scatter(data["Date"], data["Value"])

plt.plot()
plt.show()