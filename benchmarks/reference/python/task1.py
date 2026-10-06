import csv
from collections import Counter

with open("orders.csv", newline="") as f:
    orders = list(csv.DictReader(f))

print("Cancelled:")
print(sum(1 for o in orders if o["status"] == "cancelled"))

units = Counter()
for o in orders:
    if o["status"] == "done":
        units[o["product"]] += int(o["qty"])
print("Best sellers:")
for product, n in units.most_common(5):
    print(f"{product}: {n}")
