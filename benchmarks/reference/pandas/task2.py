import pandas as pd

df = pd.read_csv("orders.csv")
done = df[df.status == "done"]
revenue = (done.qty * done.price).groupby(done.date.str[:7]).sum().round(2)
for month, value in revenue.items():
    print(f"{month}: {value}")
