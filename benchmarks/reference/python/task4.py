import csv
from collections import Counter

units = Counter()
with open("orders.csv", newline="") as f:
    for o in csv.DictReader(f):
        if o["status"] == "returned":
            units[o["customer"]] += int(o["qty"])
for customer in sorted(units):
    if units[customer] >= 5:
        print(f"{customer}: {units[customer]}")
