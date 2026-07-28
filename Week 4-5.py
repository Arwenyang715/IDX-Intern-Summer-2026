from pathlib import Path
import numpy as np
import pandas as pd


BASE_DIR = Path.home() / "Desktop" / "IDX Summer 2026"
INPUT_DIR = BASE_DIR / "Week 2"
OUTPUT_DIR = BASE_DIR / "Week 4-5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILES = {
    "Sold": INPUT_DIR / "Sold_Processed.csv",
    "Listed": INPUT_DIR / "Listed_Processed.csv",
}

DATE_COLUMNS = [
    "CloseDate",
    "PurchaseContractDate",
    "ListingContractDate",
    "ContractStatusChangeDate",
]

# Numeric columns specifically required by the assignment, plus common
# Bedrooms/Bathrooms variants that may appear in MLS exports.
NUMERIC_COLUMNS = [
    "ClosePrice",
    "LivingArea",
    "DaysOnMarket",
    "BedroomsTotal",
    "Bedrooms",
    "BathroomsTotalInteger",
    "BathroomsTotalDecimal",
    "BathroomsFull",
    "BathroomsHalf",
    "BathroomsThreeQuarter",
    "Latitude",
    "Longitude",
]

BEDROOM_COLUMNS = ["BedroomsTotal", "Bedrooms"]
BATHROOM_COLUMNS = [
    "BathroomsTotalInteger",
    "BathroomsTotalDecimal",
    "BathroomsFull",
    "BathroomsHalf",
    "BathroomsThreeQuarter",
]

# Approximate California bounding box used only as a quality-control flag.
CA_LAT_MIN = 32.0
CA_LAT_MAX = 42.1
CA_LON_MIN = -125.0
CA_LON_MAX = -114.0

summary_records = []
dtype_records = []


def count_true(series: pd.Series) -> int:
    """Safely count True values in a Boolean flag column."""
    return int(series.fillna(False).astype(bool).sum())


def add_summary(dataset: str, category: str, metric: str, value) -> None:
    """Add one result to the combined cleaning summary."""
    summary_records.append(
        {
            "dataset": dataset,
            "category": category,
            "metric": metric,
            "value": value,
        }
    )


def convert_dates(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Convert available assignment date fields to datetime."""
    for column in DATE_COLUMNS:
        if column in df.columns:
            original_non_null = int(df[column].notna().sum())
            df[column] = pd.to_datetime(df[column], errors="coerce")
            converted_non_null = int(df[column].notna().sum())
            invalid_date_count = original_non_null - converted_non_null

            add_summary(dataset, "date conversion", f"{column}_invalid_or_unparseable", invalid_date_count)

    return df


def convert_numeric_fields(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Convert available required numeric fields to numeric dtype."""
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            original_non_null = int(df[column].notna().sum())

            # Remove commas and dollar signs before conversion when the
            # source column was imported as text.
            cleaned = (
                df[column]
                .astype("string")
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )

            df[column] = pd.to_numeric(cleaned, errors="coerce")
            converted_non_null = int(df[column].notna().sum())
            conversion_failures = original_non_null - converted_non_null

            add_summary(dataset, "numeric conversion", f"{column}_conversion_failures", conversion_failures)

    return df


def create_numeric_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create flags for invalid numeric values required by the assignment."""
    df["invalid_close_price_flag"] = (
        df["ClosePrice"].le(0) if "ClosePrice" in df.columns else False
    )

    df["invalid_living_area_flag"] = (
        df["LivingArea"].le(0) if "LivingArea" in df.columns else False
    )

    df["invalid_days_on_market_flag"] = (
        df["DaysOnMarket"].lt(0) if "DaysOnMarket" in df.columns else False
    )

    available_bedroom_columns = [c for c in BEDROOM_COLUMNS if c in df.columns]
    if available_bedroom_columns:
        df["invalid_bedrooms_flag"] = df[available_bedroom_columns].lt(0).any(axis=1)
    else:
        df["invalid_bedrooms_flag"] = False

    available_bathroom_columns = [c for c in BATHROOM_COLUMNS if c in df.columns]
    if available_bathroom_columns:
        df["invalid_bathrooms_flag"] = df[available_bathroom_columns].lt(0).any(axis=1)
    else:
        df["invalid_bathrooms_flag"] = False

    numeric_flag_columns = [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
    ]
    df["invalid_numeric_flag"] = df[numeric_flag_columns].any(axis=1)

    return df


def create_date_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three date-consistency flags required by the assignment."""
    # Listing date should not occur after close date.
    if {"ListingContractDate", "CloseDate"}.issubset(df.columns):
        df["listing_after_close_flag"] = (
            df["ListingContractDate"].notna()
            & df["CloseDate"].notna()
            & (df["ListingContractDate"] > df["CloseDate"])
        )
    else:
        df["listing_after_close_flag"] = False

    # Purchase contract date should not occur after close date.
    if {"PurchaseContractDate", "CloseDate"}.issubset(df.columns):
        df["purchase_after_close_flag"] = (
            df["PurchaseContractDate"].notna()
            & df["CloseDate"].notna()
            & (df["PurchaseContractDate"] > df["CloseDate"])
        )
    else:
        df["purchase_after_close_flag"] = False

    # Negative timeline means ListingContractDate > PurchaseContractDate,
    # PurchaseContractDate > CloseDate, or ListingContractDate > CloseDate.
    listing_after_purchase = pd.Series(False, index=df.index)
    if {"ListingContractDate", "PurchaseContractDate"}.issubset(df.columns):
        listing_after_purchase = (
            df["ListingContractDate"].notna()
            & df["PurchaseContractDate"].notna()
            & (df["ListingContractDate"] > df["PurchaseContractDate"])
        )

    df["negative_timeline_flag"] = (
        listing_after_purchase
        | df["listing_after_close_flag"]
        | df["purchase_after_close_flag"]
    )

    return df


def create_geographic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing, sentinel, sign, and implausible coordinate flags."""
    if "Latitude" not in df.columns:
        df["Latitude"] = np.nan
    if "Longitude" not in df.columns:
        df["Longitude"] = np.nan

    df["missing_coordinates_flag"] = df["Latitude"].isna() | df["Longitude"].isna()

    df["zero_coordinates_flag"] = (
        df["Latitude"].eq(0) | df["Longitude"].eq(0)
    )

    # California longitude should be negative.
    df["positive_longitude_flag"] = df["Longitude"].gt(0)

    # Only evaluate the bounding box when both coordinates are present.
    coordinates_present = df["Latitude"].notna() & df["Longitude"].notna()
    outside_california_box = (
        ~df["Latitude"].between(CA_LAT_MIN, CA_LAT_MAX, inclusive="both")
        | ~df["Longitude"].between(CA_LON_MIN, CA_LON_MAX, inclusive="both")
    )

    df["implausible_coordinates_flag"] = coordinates_present & outside_california_box

    geographic_flags = [
        "missing_coordinates_flag",
        "zero_coordinates_flag",
        "positive_longitude_flag",
        "implausible_coordinates_flag",
    ]
    df["invalid_coordinate_flag"] = df[geographic_flags].any(axis=1)

    return df


def remove_redundant_data(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Remove only clearly redundant columns and exact duplicate rows."""
    rows_before = len(df)
    columns_before = len(df.columns)

    # Remove CSV index-export columns such as "Unnamed: 0".
    unnamed_columns = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    # Remove duplicated column names, keeping the first occurrence.
    duplicated_column_count = int(df.columns.duplicated().sum())
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Remove columns containing no usable values.
    all_empty_columns = [c for c in df.columns if df[c].isna().all()]
    if all_empty_columns:
        df = df.drop(columns=all_empty_columns)

    # Remove exact duplicate records.
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    add_summary(dataset, "removed data", "unnamed_columns_removed", len(unnamed_columns))
    add_summary(dataset, "removed data", "duplicate_columns_removed", duplicated_column_count)
    add_summary(dataset, "removed data", "all_empty_columns_removed", len(all_empty_columns))
    add_summary(dataset, "removed data", "exact_duplicate_rows_removed", duplicate_rows)
    add_summary(dataset, "shape", "rows_before", rows_before)
    add_summary(dataset, "shape", "columns_before", columns_before)
    add_summary(dataset, "shape", "rows_after", len(df))
    add_summary(dataset, "shape", "columns_after", len(df.columns))

    return df


def record_quality_summary(df: pd.DataFrame, dataset: str) -> None:
    """Record assignment-required flag counts and missing-value totals."""
    flag_columns = [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
        "invalid_numeric_flag",
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag",
        "missing_coordinates_flag",
        "zero_coordinates_flag",
        "positive_longitude_flag",
        "implausible_coordinates_flag",
        "invalid_coordinate_flag",
    ]

    for column in flag_columns:
        add_summary(dataset, "flag count", column, count_true(df[column]))

    add_summary(dataset, "missing values", "total_missing_cells", int(df.isna().sum().sum()))
    add_summary(dataset, "missing values", "rows_with_any_missing_value", int(df.isna().any(axis=1).sum()))


def record_data_types(df: pd.DataFrame, dataset: str) -> None:
    """Save final column data types for confirmation."""
    for column, dtype in df.dtypes.items():
        dtype_records.append(
            {
                "dataset": dataset,
                "column": column,
                "dtype": str(dtype),
            }
        )


def clean_dataset(dataset: str, input_path: Path) -> pd.DataFrame:
    """Run all Weeks 4–5 transformations for one MLS dataset."""
    print("\n" + "=" * 70)
    print(f"Processing {dataset}: {input_path}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find {input_path}. Check BASE_DIR and INPUT_DIR at the top of the script."
        )

    # low_memory=False reduces mixed-type inference warnings in wide MLS files.
    df = pd.read_csv(input_path, low_memory=False)

    print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    # Standardize empty and whitespace-only text as missing values.
    object_columns = df.select_dtypes(include=["object", "string"]).columns
    if len(object_columns) > 0:
        df[object_columns] = df[object_columns].replace(r"^\s*$", np.nan, regex=True)

    df = remove_redundant_data(df, dataset)
    df = convert_dates(df, dataset)
    df = convert_numeric_fields(df, dataset)
    df = create_numeric_flags(df)
    df = create_date_flags(df)
    df = create_geographic_flags(df)

    record_quality_summary(df, dataset)
    record_data_types(df, dataset)

    output_path = OUTPUT_DIR / f"{dataset}_Cleaned.csv"
    # ISO date format keeps CSV dates consistent and readable.
    df.to_csv(output_path, index=False, date_format="%Y-%m-%d")

    print(f"Final shape:    {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Saved cleaned data to: {output_path}")

    print("\nDate consistency flag counts:")
    for column in [
        "listing_after_close_flag",
        "purchase_after_close_flag",
        "negative_timeline_flag",
    ]:
        print(f"  {column}: {count_true(df[column]):,}")

    print("\nGeographic data quality summary:")
    for column in [
        "missing_coordinates_flag",
        "zero_coordinates_flag",
        "positive_longitude_flag",
        "implausible_coordinates_flag",
        "invalid_coordinate_flag",
    ]:
        print(f"  {column}: {count_true(df[column]):,}")

    print("\nNumeric validation summary:")
    for column in [
        "invalid_close_price_flag",
        "invalid_living_area_flag",
        "invalid_days_on_market_flag",
        "invalid_bedrooms_flag",
        "invalid_bathrooms_flag",
        "invalid_numeric_flag",
    ]:
        print(f"  {column}: {count_true(df[column]):,}")

    print("\nConfirmed data types for required fields:")
    confirmation_columns = [
        c for c in DATE_COLUMNS + NUMERIC_COLUMNS if c in df.columns
    ]
    print(df[confirmation_columns].dtypes.to_string())

    return df


def main() -> None:
    """Process all available input files and save combined documentation."""
    processed_count = 0

    for dataset, input_path in INPUT_FILES.items():
        if input_path.exists():
            clean_dataset(dataset, input_path)
            processed_count += 1
        else:
            print(f"\nSkipped {dataset}: file not found at {input_path}")

    if processed_count == 0:
        raise FileNotFoundError(
            "Neither Sold_Processed.csv nor Listed_Processed.csv was found. "
            "Check INPUT_DIR at the top of the script."
        )

    summary_df = pd.DataFrame(summary_records)
    summary_path = OUTPUT_DIR / "Weeks_4_5_Cleaning_Summary.csv"
    summary_df.to_csv(summary_path, index=False)

    dtypes_df = pd.DataFrame(dtype_records)
    dtypes_path = OUTPUT_DIR / "Weeks_4_5_Data_Types.csv"
    dtypes_df.to_csv(dtypes_path, index=False)

    print("\n" + "=" * 70)
    print("WEEKS 4–5 CLEANING COMPLETE")
    print("=" * 70)
    print(f"Summary saved to:    {summary_path}")
    print(f"Data types saved to: {dtypes_path}")


if __name__ == "__main__":
    main()
