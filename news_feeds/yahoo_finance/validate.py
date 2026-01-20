#!/usr/bin/env python3
import argparse
import csv
import os
import sys

SCHEMA = [
    "id",
    "title",
    "link",
    "published",
    "source",
    "summary",
    "query",
    "fetched_at",
]
REQUIRED_FIELDS = {"id", "title", "link", "fetched_at"}


def validate_csv(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty")

        if header != SCHEMA:
            raise ValueError(
                "CSV header mismatch. Expected: "
                f"{SCHEMA} but found: {header}"
            )

        ids = set()
        for row_num, row in enumerate(reader, start=2):
            if len(row) != len(SCHEMA):
                raise ValueError(f"Row {row_num} has wrong number of columns")
            row_data = dict(zip(SCHEMA, row))
            for field in REQUIRED_FIELDS:
                if not row_data.get(field):
                    raise ValueError(f"Row {row_num} missing required field {field}")
            if row_data["id"] in ids:
                raise ValueError(f"Duplicate id found at row {row_num}")
            ids.add(row_data["id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Yahoo Finance CSV output")
    parser.add_argument(
        "--csv",
        default="data/yahoo_finance_news.csv",
        help="Path to output CSV",
    )
    args = parser.parse_args()
    try:
        validate_csv(args.csv)
    except Exception as exc:  # noqa: BLE001
        print(f"Validation failed: {exc}")
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
