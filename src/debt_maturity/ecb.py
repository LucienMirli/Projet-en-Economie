import io
import requests
import polars as pl


BASE_URL = "https://data-api.ecb.europa.eu/service/data/GFS"

SERIES = {
    "wam": "M.N.{geo}.W0.S13.S1.N.L.LE.F3.TT._Z.YR._T.F.V.A1._T",
    "total_debt": "M.N.{geo}.W0.S13.S1.N.L.LE.F3.T._Z.EUR._T.F.V.N._T",
    "long_debt_short_residual": "M.N.{geo}.W0.S13.S1.N.L.LE.F3.LS._Z.EUR._T.F.V.N._T",
    "short_term_debt": "M.N.{geo}.W0.S13.S1.N.L.LE.F3.S._Z.EUR._T.F.V.N._T",
    "floating_debt": "M.N.{geo}.W0.S13.S1.N.L.LE.F3B.T._Z.EUR._T.F.V.N._T",
    "redemptions": "M.N.{geo}.W0.S13.S1.N.LD.F.F3.T._Z.EUR._T.F.V.N._T",
    "issuance": "M.N.{geo}.W0.S13.S1.N.LI.F.F3.T._Z.EUR._T.F.V.N._T",
}


def download_series(country, name, key, start, end):
    """Download a monthly ECB series."""

    url = f"{BASE_URL}/{key.format(geo=country)}"

    response = requests.get(
        url,
        params={
            "startPeriod": start,
            "endPeriod": end,
            "format": "csvdata",
        },
        headers={"Accept": "text/csv"},
        timeout=60,
    )

    if response.status_code == 404:
        print(f"Missing: {country} / {name}")
        return None

    response.raise_for_status()

    df = (
        pl.read_csv(
            BytesIO(response.content),
            columns=["TIME_PERIOD", "OBS_VALUE"],
        )
        .rename({
            "TIME_PERIOD": "date",
            "OBS_VALUE": name,
        })
        .with_columns(
            pl.col("date").str.strptime(pl.Date, "%Y-%m"),
            pl.col(name).cast(pl.Float64, strict=False),
        )
        .unique(subset=["date"])
    )

    return df


def download_panel(countries, start="2010-01", end="2025-12"):
    """Build a monthly country panel from ECB data."""

    panels = []

    for country in countries:
        print(f"Downloading {country}...")

        country_df = None

        for name, key in SERIES.items():
            df = download_series(country, name, key, start, end)

            if df is None:
                continue

            country_df = (
                df if country_df is None
                else country_df.join(df, on="date", how="full", coalesce=True)
            )

        if country_df is not None:
            panels.append(
                country_df.with_columns(
                    pl.lit(country).alias("country")
                )
            )

    if not panels:
        raise ValueError("No data downloaded.")

    return (
        pl.concat(panels, how="diagonal")
        .sort(["country", "date"])
    )


def to_quarterly(df):
    """Convert monthly data to a quarterly panel."""

    df = df.with_columns(
        pl.col("date").dt.truncate("1q").alias("quarter")
    )

    stock_cols = [
        "wam",
        "total_debt",
        "long_debt_short_residual",
        "short_term_debt",
        "floating_debt",
    ]

    flow_cols = ["redemptions", "issuance"]

    # Stocks: December, March, June and September observations.
    stocks = (
        df.filter(
            pl.col("date").dt.month().is_in([3, 6, 9, 12])
        )
        .group_by(["country", "quarter"])
        .agg([
            pl.col(col).last().alias(col)
            for col in stock_cols
        ])
    )

    # Flows: sum the three monthly observations.
    flows = (
        df.group_by(["country", "quarter"])
        .agg([
            pl.when(pl.col(col).count() == 3)
            .then(pl.col(col).sum())
            .otherwise(None)
            .alias(col)
            for col in flow_cols
        ])
    )

    quarterly = (
        stocks.join(
            flows,
            on=["country", "quarter"],
            how="full",
            coalesce=True,
        )
        .with_columns(
            (
                (
                    pl.col("short_term_debt")
                    + pl.col("long_debt_short_residual")
                )
                / pl.col("total_debt")
            ).alias("share_short"),

            (
                pl.col("floating_debt")
                / pl.col("total_debt")
            ).alias("share_floating"),
        )
        .sort(["country", "quarter"])
    )

    return quarterly




BASE_URL = "https://data-api.ecb.europa.eu/service/data/GFS"

def fetch_indicator(name: str, key_template: str) -> pl.DataFrame:
    # Empty geographical dimension = all available areas
    key = key_template.format(geo="")

    response = requests.get(
        f"{BASE_URL}/{key}",
        params={
            "startPeriod": "1995-01",
            "format": "csvdata",
            "detail": "dataonly",
        },
        timeout=120,
    )
    response.raise_for_status()

    df = pl.read_csv(
        io.BytesIO(response.content),
        infer_schema_length=0,
    )

    return (
        df.select(
            pl.col("REF_AREA").alias("country_code"),
            pl.col("TIME_PERIOD").alias("date"),
            pl.col("OBS_VALUE")
              .cast(pl.Float64, strict=False)
              .alias(name),
        )
        .with_columns(
            (
                pl.col("date").str.slice(0, 7) + "-01"
            ).str.strptime(
                pl.Date, "%Y-%m-%d"
            ).alias("date")
        )
    )
