import csv

with open("orders.csv", newline="") as f:
    rows = [o for o in csv.DictReader(f) if o["status"] == "returned"]
rows.sort(key=lambda o: o["date"])
with open("returns.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "customer", "product"])
    writer.writerows([o["date"], o["customer"], o["product"]] for o in rows)
print(len(rows))
