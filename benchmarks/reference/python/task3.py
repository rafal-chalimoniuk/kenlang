import csv
from collections import Counter

total, returned = Counter(), Counter()
with open("orders.csv", newline="") as f:
    for o in csv.DictReader(f):
        total[o["category"]] += 1
        if o["status"] == "returned":
            returned[o["category"]] += 1
share = {c: round(returned[c] / total[c] * 100, 1) for c in total}
for category, pct in sorted(share.items(), key=lambda kv: kv[1], reverse=True):
    print(f"{category}: {pct}%")
