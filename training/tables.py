"""The tables the training examples are about.

The first version of the model saw a single table (``orders.csv``), so it could not tell the language
from the data. This module describes four tables:

* ``orders``, ``bookings`` and ``shipments`` have the same *kind* of columns (a date, a few categories,
  a quantity, a price, a state) under different names, with different values. Training examples use them.
* ``grades`` has a different shape (no price) and is **never** used for training: it measures how the model
  copes with a table it has not seen.

``python -m training.tables`` (re)writes ``training/tables/*.csv``; the files are committed, and
``tests/test_training_data.py`` checks that they still match this module.
"""
import datetime
import pathlib
import random
from dataclasses import dataclass

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
TABLE_DIR = HERE / "tables"


_FRAMES = {}


@dataclass(frozen=True)
class Num:
    """A numeric column: ``noun`` is how task texts refer to it, ``steps`` are sensible thresholds."""
    name: str
    noun: str
    steps: tuple
    unit: str = ""
    additive: bool = True


@dataclass(frozen=True)
class Table:
    key: str
    csv: pathlib.Path
    rows: str                    # what one row is, in the plural: "orders"
    cols: tuple
    description: str
    date: str                    # name of the date column ("" if none)
    state: str                   # name of the column with three states
    states: tuple
    dims: dict                   # categorical column -> {"each": [phrases], "noun": ..., "plural": ...}
    name_col: str                # text column used for "contains the fragment"
    qty: Num = None              # additive whole numbers
    price: Num = None            # unit price: revenue is qty x price
    stat: Num = None             # one more numeric column
    money: dict = None           # {"noun", "unit"}: what qty x price is called

    @property
    def file(self):
        return self.csv.name

    def frame(self):
        """The table as a DataFrame, plus ``rev`` (qty x price) and ``month``. Cached: do not modify it."""
        if self.key not in _FRAMES:
            df = pd.read_csv(self.csv)
            if self.qty and self.price:
                df["rev"] = df[self.qty.name] * df[self.price.name]
            if self.date:
                df["month"] = df[self.date].str[:7]
            _FRAMES[self.key] = df
        return _FRAMES[self.key]


def _describe(file, lines):
    return f"The file `{file}` has a header row. Columns:\n\n" + "\n".join(f"- `{n}` – {t}" for n, t in lines) + "\n"


def _day(rnd):
    return (datetime.date(2026, 1, 1) + datetime.timedelta(days=rnd.randrange(181))).isoformat()


def make_bookings(rnd, n=400):
    rooms = {"single": 79.0, "double": 109.5, "twin": 99.0, "family": 159.0, "suite": 289.9, "studio": 129.0}
    guests = ["alice", "ben", "carla", "dmitri", "elena", "farid", "greta", "hugo", "ines", "jonas", "kira", "leo"]
    cities = ["paris", "rome", "lisbon", "vienna", "prague"]
    rows = []
    for _ in range(n):
        room = rnd.choice(list(rooms))
        rows.append(dict(booked=_day(rnd), guest=rnd.choice(guests), room=room, city=rnd.choice(cities),
                         nights=rnd.choice([1, 1, 2, 2, 2, 3, 3, 4, 5, 7]), rate=rooms[room],
                         rating=rnd.choice([1, 2, 3, 4, 4, 5, 5, 5]),
                         outcome=rnd.choices(["stayed", "cancelled", "no_show"], [70, 20, 10])[0]))
    return pd.DataFrame(rows).sort_values("booked", kind="stable")


def make_shipments(rnd, n=400):
    items = {"pallet": (18.5, 11.0), "crate": (12.0, 6.5), "drum": (34.9, 14.0), "parcel": (6.5, 1.2),
             "bundle": (9.9, 3.4), "roll": (22.0, 8.8), "tank": (140.0, 55.0), "carton": (4.25, 2.1)}
    clients = ["acme", "borealis", "cobalt", "delta", "everest", "falcon", "granite", "harbor", "iris", "juno"]
    rows = []
    for _ in range(n):
        item = rnd.choice(list(items))
        cost, kg = items[item]
        units = rnd.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        rows.append(dict(sent=_day(rnd), client=rnd.choice(clients), item=item,
                         region=rnd.choice(["north", "south", "east", "west", "central"]), units=units,
                         unit_cost=cost, weight=round(units * kg * rnd.uniform(0.9, 1.1), 1),
                         result=rnd.choices(["delivered", "returned", "lost"], [80, 12, 8])[0]))
    return pd.DataFrame(rows).sort_values("sent", kind="stable")


def make_grades(rnd, n=400):
    subjects = {"math": (4, 58), "physics": (3, 54), "history": (2, 66), "art": (1, 78),
                "biology": (3, 62), "chemistry": (3, 52)}
    students = ["mia", "noah", "olga", "paul", "quinn", "rosa", "sam", "tara", "ugo", "vera", "will", "xena",
                "yuri", "zoe"]
    classes = ["red", "blue", "green", "yellow"]
    rows = []
    for _ in range(n):
        s = rnd.randrange(len(students))
        subject = rnd.choice(list(subjects))
        credits, mu = subjects[subject]
        absent = rnd.random() < 0.05
        score = 0 if absent else min(100, max(1, round(rnd.gauss(mu, 18))))
        rows.append(dict(exam_date=_day(rnd), student=students[s], **{"class": classes[s % 4]}, subject=subject,
                         credits=credits, score=score,
                         result="absent" if absent else "passed" if score >= 50 else "failed"))
    return pd.DataFrame(rows).sort_values("exam_date", kind="stable")


def _dim(each, noun, plural):
    return {"each": each, "noun": noun, "plural": plural}


TABLES = {
    "orders": Table(
        key="orders", csv=ROOT / "examples" / "data" / "orders.csv", rows="orders",
        cols=("date", "customer", "product", "category", "qty", "price", "status"),
        description=(ROOT / "benchmarks" / "data_description.md").read_text(encoding="utf-8"),
        date="date", state="status", states=("done", "returned", "cancelled"),
        dims={"customer": _dim(["each customer"], "customer", "customers"),
              "product": _dim(["each product"], "product", "products"),
              "category": _dim(["each category", "each product category"], "category", "categories")},
        name_col="product",
        qty=Num("qty", "the number of items (`qty`)", (1, 2, 3, 4), unit=" pcs"),
        price=Num("price", "unit price (`price`)", (50, 100, 250, 300)),
        money={"noun": "revenue", "unit": " USD", "avg": "the average order value",
               "max": "the largest single order value"}),
    "bookings": Table(
        key="bookings", csv=TABLE_DIR / "bookings.csv", rows="bookings",
        cols=("booked", "guest", "room", "city", "nights", "rate", "rating", "outcome"),
        description=_describe("bookings.csv", [
            ("booked", "booking date in the format YYYY-MM-DD"), ("guest", "guest name"),
            ("room", "room type"), ("city", "city of the hotel"), ("nights", "number of nights (integer)"),
            ("rate", "price per night"), ("rating", "guest rating from 1 to 5 (integer)"),
            ("outcome", "what became of the booking: `stayed`, `cancelled` or `no_show`")]),
        date="booked", state="outcome", states=("stayed", "cancelled", "no_show"),
        dims={"guest": _dim(["each guest"], "guest", "guests"),
              "room": _dim(["each room type", "each room"], "room type", "room types"),
              "city": _dim(["each city"], "city", "cities")},
        name_col="room",
        qty=Num("nights", "the number of nights (`nights`)", (1, 2, 3, 4), unit=" nights"),
        price=Num("rate", "nightly rate (`rate`)", (80, 100, 120, 200)),
        stat=Num("rating", "guest rating (`rating`)", (2, 3, 4), additive=False),
        money={"noun": "booking value", "unit": " EUR", "avg": "the average booking value",
               "max": "the largest single booking value"}),
    "shipments": Table(
        key="shipments", csv=TABLE_DIR / "shipments.csv", rows="shipments",
        cols=("sent", "client", "item", "region", "units", "unit_cost", "weight", "result"),
        description=_describe("shipments.csv", [
            ("sent", "dispatch date in the format YYYY-MM-DD"), ("client", "client name"),
            ("item", "kind of goods"), ("region", "destination region"), ("units", "number of units (integer)"),
            ("unit_cost", "cost of one unit"), ("weight", "total weight in kilograms"),
            ("result", "what became of the shipment: `delivered`, `returned` or `lost`")]),
        date="sent", state="result", states=("delivered", "returned", "lost"),
        dims={"client": _dim(["each client"], "client", "clients"),
              "item": _dim(["each kind of goods", "each item"], "item", "items"),
              "region": _dim(["each region"], "region", "regions")},
        name_col="item",
        qty=Num("units", "the number of units (`units`)", (2, 5, 8, 12), unit=" units"),
        price=Num("unit_cost", "unit cost (`unit_cost`)", (10, 20, 30, 50)),
        stat=Num("weight", "weight in kilograms (`weight`)", (5, 10, 20, 40), unit=" kg"),
        money={"noun": "cost", "unit": " USD", "avg": "the average shipment cost",
               "max": "the largest single shipment cost"}),
    "grades": Table(
        key="grades", csv=TABLE_DIR / "grades.csv", rows="exams",
        cols=("exam_date", "student", "class", "subject", "credits", "score", "result"),
        description=_describe("grades.csv", [
            ("exam_date", "exam date in the format YYYY-MM-DD"), ("student", "student name"),
            ("class", "the student's class"), ("subject", "school subject"),
            ("credits", "credits the exam is worth (integer)"), ("score", "points scored, 0 to 100 (integer)"),
            ("result", "`passed`, `failed` or `absent`")]),
        date="exam_date", state="result", states=("passed", "failed", "absent"),
        dims={"student": _dim(["each student"], "student", "students"),
              "class": _dim(["each class"], "class", "classes"),
              "subject": _dim(["each subject"], "subject", "subjects")},
        name_col="subject",
        qty=Num("credits", "the number of credits (`credits`)", (1, 2, 3), unit=" credits"),
        stat=Num("score", "score (`score`)", (40, 50, 60, 75), additive=False)),
}
TRAIN_TABLES = ("orders", "bookings", "shipments")
HELD_OUT_TABLE = "grades"
_MAKERS = {"bookings": (make_bookings, 11), "shipments": (make_shipments, 12), "grades": (make_grades, 13)}


def write_all():
    TABLE_DIR.mkdir(exist_ok=True)
    _FRAMES.clear()
    for key, (maker, seed) in _MAKERS.items():
        maker(random.Random(seed)).to_csv(TABLES[key].csv, index=False, lineterminator="\n")


def expected_text(key):
    """The CSV text ``write_all`` would write for a generated table (used to detect drift)."""
    maker, seed = _MAKERS[key]
    return maker(random.Random(seed)).to_csv(index=False, lineterminator="\n")


if __name__ == "__main__":
    write_all()
    for key in TABLES:
        print(key, TABLES[key].frame().shape)
