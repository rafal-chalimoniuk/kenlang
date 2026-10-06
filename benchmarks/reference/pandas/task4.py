import pandas as pd

df = pd.read_csv("orders.csv")
units = df[df.status == "returned"].groupby("customer").qty.sum()
for customer, n in units[units >= 5].items():
    print(f"{customer}: {n}")
