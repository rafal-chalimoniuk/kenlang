import pandas as pd

df = pd.read_csv("orders.csv")
returned = df[df.status == "returned"].sort_values("date")[["date", "customer", "product"]]
returned.to_csv("returns.csv", index=False)
print(len(returned))
