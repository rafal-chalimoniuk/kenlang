import csv
from collections import defaultdict

revenue = defaultdict(float)
with open("orders.csv", newline="") as f:
    for o in csv.DictReader(f):
        if o["status"] == "done":
            revenue[o["date"][:7]] += int(o["qty"]) * float(o["price"])
for month in sorted(revenue):
    print(f"{month}: {round(revenue[month], 2)}")
