import pandas as pd

df = pd.read_csv("orders.csv")
share = ((df.status == "returned").groupby(df.category).mean() * 100).round(1)
for category, pct in share.sort_values(ascending=False).items():
    print(f"{category}: {pct}%")
