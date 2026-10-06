import pandas as pd

df = pd.read_csv("orders.csv")
sel = df[(df.status == "done") & df.category.isin(["clothing", "groceries"]) & df["product"].str.contains("e")]
print(sel.qty.sum())
