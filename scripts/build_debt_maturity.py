
from functools import reduce
from pathlib import Path

import polars as pl
import pycountry

from src.debt_maturity.ecb import SERIES, fetch_indicator


def country_name(code):
    # Special ECB / geographic codes
    special = {
        "GR": "Greece",
        "EL": "Greece",
        "XK": "Kosovo",
    }
    if code in special:
        return special[code]

    country = pycountry.countries.get(alpha_2=code)
    return country.name if country else None


def to_quarterly(df, variable):
    # Year and quarter
    df = df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("q"),
    )

    # Stocks and WAM: last monthly observation
    return (
        df.sort("date")
        .group_by("country_code", "year", "q")
        .agg(pl.col(variable).last())
    )


def build_panel():
    frames = []

    for name, key in SERIES.items():
        print(f"Downloading {name}...")

        try:
            monthly = fetch_indicator(name, key)
            quarterly = to_quarterly(monthly, name)
            frames.append(quarterly)
        except Exception as e:
            print(f"Failed: {name}: {e}")

    if not frames:
        raise RuntimeError("No ECB series downloaded.")

    panel = reduce(
        lambda a, b: a.join(
            b,
            on=["country_code", "year", "q"],
            how="full",
            coalesce=True,
        ),
        frames,
    )

    # Country names and removal of aggregates
    panel = (
        panel.with_columns(
            pl.col("country_code")
              .map_elements(country_name,
                            return_dtype=pl.String)
              .alias("country")
        )
        .filter(pl.col("country").is_not_null())
        .with_columns(
            pl.format(
                "{}Q{}",
                pl.col("year"),
                pl.col("q"),
            ).alias("quarter")
        )
        .sort("country_code", "year", "q")
    )

    Path("data/processed").mkdir(
        parents=True, exist_ok=True
    )

    panel.write_csv(
        "data/processed/debt_maturity_panel.csv"
    )
    panel.write_parquet(
        "data/processed/debt_maturity_panel.parquet"
    )

    print("Countries:", panel["country"].n_unique())
    print("Rows:", panel.height)
    print(panel.head())


if __name__ == "__main__":
    build_panel()
