import pandas as pd

df = pd.read_csv("orders.csv")
print("Cancelled:")
print((df.status == "cancelled").sum())
top = df[df.status == "done"].groupby("product").qty.sum().nlargest(5)
print("Best sellers:")
for product, units in top.items():
    print(f"{product}: {units}")
