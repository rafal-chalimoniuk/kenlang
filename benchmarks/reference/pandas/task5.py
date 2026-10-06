import pandas as pd

df = pd.read_csv("orders.csv")
value = (df.qty * df.price)[df.status == "done"]
tier = value.map(lambda v: "large" if v >= 1000 else "medium" if v >= 200 else "small")
for name, n in tier.value_counts().items():
    print(f"{name}: {n}")
