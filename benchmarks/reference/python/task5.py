import csv
from collections import Counter


def tier(value):
    if value >= 1000:
        return "large"
    if value >= 200:
        return "medium"
    return "small"


counts = Counter()
with open("orders.csv", newline="") as f:
    for o in csv.DictReader(f):
        if o["status"] == "done":
            counts[tier(int(o["qty"]) * float(o["price"]))] += 1
for name, n in counts.most_common():
    print(f"{name}: {n}")
