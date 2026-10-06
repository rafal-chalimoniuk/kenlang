import csv

total = 0
with open("orders.csv", newline="") as f:
    for o in csv.DictReader(f):
        if o["status"] == "done" and o["category"] in ("clothing", "groceries") and "e" in o["product"]:
            total += int(o["qty"])
print(total)
