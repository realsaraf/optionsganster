"""
One-time download of QQQ minute-bar data from Massive Flat Files.

Saves filtered QQQ rows as lightweight CSV files under data/qqq/
so the app (mock mode) and backtest scripts can load them instantly
without re-downloading 20+ MB compressed files each time.

Usage:
    python download_flatfiles.py              # downloads all available days
    python download_flatfiles.py 2026-02-11   # downloads a single day
"""
import sys
import os
import io
import gzip
import csv
from pathlib import Path

import boto3
from botocore.config import Config

# ── S3 credentials ───────────────────────────────────────────
ACCESS_KEY = "f47f4bc7-1e8c-43b5-9f18-92bcf935fda8"
SECRET_KEY = "X6qu3ZNlEqQl1lOkUjj325DtmxH2nmTU"
ENDPOINT   = "https://files.massive.com"
BUCKET     = "flatfiles"
TICKER     = "QQQ"

DATA_DIR = Path(__file__).parent / "data" / "qqq"


def get_s3_client():
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    return session.client(
        "s3",
        endpoint_url=ENDPOINT,
        config=Config(signature_version="s3v4"),
    )


def list_available_days(s3) -> list[str]:
    """Return sorted list of YYYY-MM-DD dates available in S3."""
    # List all year/month combos
    years_resp = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix="us_stocks_sip/minute_aggs_v1/",
        Delimiter="/",
    )
    dates = []
    for yp in years_resp.get("CommonPrefixes", []):
        year_prefix = yp["Prefix"]
        months_resp = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=year_prefix,
            Delimiter="/",
        )
        for mp in months_resp.get("CommonPrefixes", []):
            month_prefix = mp["Prefix"]
            files_resp = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=month_prefix,
            )
            for obj in files_resp.get("Contents", []):
                key = obj["Key"]
                # key like us_stocks_sip/minute_aggs_v1/2026/02/2026-02-05.csv.gz
                fname = key.rsplit("/", 1)[-1]
                if fname.endswith(".csv.gz"):
                    dates.append(fname.replace(".csv.gz", ""))
    return sorted(dates)


def download_day(s3, date_str: str) -> int:
    """Download, filter for QQQ, and save locally. Returns row count."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{date_str}.csv"

    if out_path.exists():
        # Count existing rows
        with open(out_path) as f:
            count = sum(1 for _ in f) - 1  # minus header
        print(f"  {date_str}: already exists ({count} rows), skipping")
        return count

    # Build S3 key
    parts = date_str.split("-")
    year, month = parts[0], parts[1]
    object_key = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz"

    print(f"  Downloading {object_key} ...")
    buf = io.BytesIO()
    try:
        s3.download_fileobj(BUCKET, object_key, buf)
    except Exception as e:
        print(f"  ERROR: {e}")
        return 0

    size_mb = buf.getbuffer().nbytes / 1024 / 1024
    print(f"  Downloaded {size_mb:.1f} MB, filtering for {TICKER} ...")
    buf.seek(0)

    rows = []
    with gzip.open(buf, "rt") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["ticker"] == TICKER:
                rows.append(row)

    if not rows:
        print(f"  WARNING: No {TICKER} rows found for {date_str}")
        return 0

    # Write filtered CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved {len(rows)} {TICKER} rows → {out_path}")
    return len(rows)


def main():
    s3 = get_s3_client()

    if len(sys.argv) > 1:
        # Download specific date(s)
        for date_str in sys.argv[1:]:
            download_day(s3, date_str)
    else:
        # Download all available days
        print("Listing available days ...")
        days = list_available_days(s3)
        print(f"Found {len(days)} days: {days}")
        total = 0
        for d in days:
            total += download_day(s3, d)
        print(f"\nDone! Total {TICKER} rows saved: {total}")


if __name__ == "__main__":
    main()
