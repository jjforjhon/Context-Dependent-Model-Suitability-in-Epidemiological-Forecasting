"""
Google Trends Data Collection for Measles Forecasting in Bangladesh.

This script collects Google Trends search interest data for measles-related
queries in Bangladesh, for use as a supplementary covariate in the forecasting
pipeline.

Scientific Rationale:
    People search for disease symptoms online before or during seeking medical
    care. Google Trends search interest can serve as a proxy for public awareness
    and early outbreak detection. This approach is established in the digital
    epidemiology literature (e.g., Ginsberg et al., 2009; Lazer et al., 2014).

Data Characteristics:
    - Google Trends returns RELATIVE search interest (0-100 scale), not absolute
      search volume. The value 100 represents the peak search interest for the
      given query within the specified time period and geography.
    - For periods > 5 years: monthly resolution
    - For periods 90 days - 5 years: weekly resolution
    - For periods < 90 days: daily resolution
    - Values can change retroactively as Google re-normalizes the index.

Search Terms:
    - "measles" (English)
    - "হাম" (Bengali for measles)
    - Combined index (sum or max of both queries)

Geographic Filter:
    - BD (Bangladesh, country-level)
    - Google Trends does NOT provide division-level data for Bangladesh.

Output Files:
    1. google_trends_monthly_2012_2025.csv (for Phase 1)
    2. google_trends_daily_2026.csv (for Phase 2)
    3. google_trends_weekly_2026.csv (intermediate file)
    4. google_trends_metadata.json (collection parameters and timestamp)

Author: Research Pipeline
Date: 2026-07-31
Python: 3.11+
Dependencies: pytrends>=4.9, pandas>=2.0
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from pytrends.request import TrendReq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Search terms: English and Bengali for "measles"
SEARCH_TERMS: list[str] = ["measles", "হাম"]

# Geographic filter: Bangladesh
GEO: str = "BD"

# Output directory
OUTPUT_DIR: Path = Path("data/google_trends")

# Time periods for data collection
# Phase 1: Monthly data (2012-01 to 2025-12)
PHASE1_TIMEFRAME: str = "2012-01-01 2025-12-31"

# Phase 2: Daily data (2026-04-01 to 2026-07-31)
# Note: For daily resolution, we need to request periods < 90 days
PHASE2_START: str = "2026-04-01"
PHASE2_END: str = "2026-07-31"

# Rate limiting: seconds to wait between API requests
# Google Trends aggressively rate-limits; be conservative
WAIT_BETWEEN_REQUESTS: int = 15

# Retry configuration
MAX_RETRIES: int = 3
RETRY_WAIT: int = 60  # seconds

# Logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def create_output_directory(output_dir: Path) -> None:
    """Create the output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")


def initialize_pytrends(
    hl: str = "en-US",
    tz: int = 360,  # Bangladesh is UTC+6 = 360 minutes
) -> TrendReq:
    """
    Initialize the pytrends API client.

    Parameters
    ----------
    hl : str
        Language for Google Trends interface.
    tz : int
        Timezone offset in minutes from UTC. Bangladesh = UTC+6 = 360.

    Returns
    -------
    TrendReq
        Initialized pytrends client.

    Note
    ----
    We set retries=0 in pytrends to avoid a compatibility issue between
    pytrends 4.9.x and urllib3 >=2.0 (deprecated 'method_whitelist'
    parameter in Retry). Retries are handled manually in our
    fetch_interest_over_time function instead.
    """
    logger.info("Initializing pytrends client...")
    pytrends = TrendReq(
        hl=hl,
        tz=tz,
        retries=0,  # Disable pytrends' built-in retries (compatibility fix)
    )
    logger.info("pytrends client initialized successfully.")
    return pytrends



def fetch_interest_over_time(
    pytrends: TrendReq,
    keywords: list[str],
    timeframe: str,
    geo: str = GEO,
    wait: int = WAIT_BETWEEN_REQUESTS,
) -> Optional[pd.DataFrame]:
    """
    Fetch Google Trends interest-over-time data with retry logic.

    Parameters
    ----------
    pytrends : TrendReq
        Initialized pytrends client.
    keywords : list[str]
        Search terms to query (max 5 per request).
    timeframe : str
        Time period string, e.g. "2012-01-01 2025-12-31".
    geo : str
        Geographic filter (ISO 3166-1 alpha-2 country code).
    wait : int
        Seconds to wait before the request (rate limiting).

    Returns
    -------
    pd.DataFrame or None
        DataFrame with date index and keyword columns, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                f"Attempt {attempt}/{MAX_RETRIES}: "
                f"Fetching {keywords} for {timeframe} in {geo}..."
            )

            # Rate limiting: wait before request
            if wait > 0:
                logger.info(f"  Waiting {wait}s for rate limiting...")
                time.sleep(wait)

            # Build payload and fetch
            pytrends.build_payload(
                kw_list=keywords,
                timeframe=timeframe,
                geo=geo,
                cat=0,  # All categories
            )
            df = pytrends.interest_over_time()

            if df.empty:
                logger.warning(
                    f"  Empty response for {keywords}. "
                    "This may indicate no search volume in the region."
                )
                return pd.DataFrame()

            # Drop the 'isPartial' column if present
            if "isPartial" in df.columns:
                df = df.drop(columns=["isPartial"])

            logger.info(
                f"  Success: {len(df)} rows, "
                f"date range: {df.index.min()} to {df.index.max()}"
            )
            return df

        except Exception as e:
            logger.error(f"  Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                retry_wait = RETRY_WAIT * attempt
                logger.info(f"  Retrying in {retry_wait}s...")
                time.sleep(retry_wait)
            else:
                logger.error(
                    f"  All {MAX_RETRIES} attempts failed for {keywords}."
                )
                return None

    return None


# ---------------------------------------------------------------------------
# Phase 1: Monthly Data (2012-2025)
# ---------------------------------------------------------------------------

def collect_phase1_monthly(pytrends: TrendReq) -> Optional[pd.DataFrame]:
    """
    Collect monthly Google Trends data for 2012-2025.

    Why monthly:
        When requesting a timeframe > 5 years, Google Trends automatically
        returns monthly resolution. This aligns with our Phase 1 monthly
        national forecasting dataset.

    Strategy:
        We request both search terms ("measles" + "হাম") in a single query
        so that their relative scales are comparable. Google Trends allows
        up to 5 keywords per request.

    Returns
    -------
    pd.DataFrame
        Monthly Google Trends data with columns:
        - measles: search interest for "measles"
        - হাম: search interest for "হাম"
        - trends_combined: max of both queries per month
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Collecting monthly data (2012-2025)")
    logger.info("=" * 60)

    df = fetch_interest_over_time(
        pytrends=pytrends,
        keywords=SEARCH_TERMS,
        timeframe=PHASE1_TIMEFRAME,
        geo=GEO,
    )

    if df is None or df.empty:
        logger.error("Phase 1 collection failed or returned empty data.")
        return None

    # Create combined index: take the maximum of both queries per period
    # Why max instead of sum: If someone searches in Bengali, they won't
    # also search in English. Max captures the dominant signal.
    df["trends_combined"] = df[SEARCH_TERMS].max(axis=1)

    # Reset index to get date as a column
    df = df.reset_index()
    df = df.rename(columns={"date": "Date"})

    # Create Year_Month column to match our other datasets
    df["Year_Month"] = df["Date"].dt.strftime("%Y-%m")

    logger.info(f"Phase 1 result: {len(df)} monthly observations")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"  Combined trends range: "
                f"{df['trends_combined'].min()} to {df['trends_combined'].max()}")

    return df


# ---------------------------------------------------------------------------
# Phase 2: Daily Data (2026 outbreak period)
# ---------------------------------------------------------------------------

def collect_phase2_daily(pytrends: TrendReq) -> Optional[pd.DataFrame]:
    """
    Collect daily Google Trends data for the 2026 outbreak period.

    Why daily:
        For timeframes < 90 days, Google Trends provides daily resolution.
        Our Phase 2 outbreak data runs Apr 3 - Jul 27, 2026 (~116 days).
        Since this exceeds 90 days, we split into two overlapping chunks
        and normalize them to a common scale.

    Overlap normalization:
        When fetching two separate time windows, Google Trends normalizes
        each independently (0-100). To combine them, we use an overlapping
        period and rescale the second chunk relative to the first.

    Returns
    -------
    pd.DataFrame
        Daily Google Trends data with columns matching Phase 1.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2: Collecting daily data (2026 outbreak)")
    logger.info("=" * 60)

    # Split into two chunks with ~14 days overlap for normalization
    # Chunk 1: Apr 1 - Jun 15 (76 days, < 90)
    # Chunk 2: Jun 1 - Jul 31 (61 days, < 90)
    # Overlap: Jun 1 - Jun 15 (15 days for normalization)

    chunk1_tf = "2026-04-01 2026-06-15"
    chunk2_tf = "2026-06-01 2026-07-31"

    logger.info("Fetching Chunk 1 (Apr 1 - Jun 15)...")
    df1 = fetch_interest_over_time(
        pytrends=pytrends,
        keywords=SEARCH_TERMS,
        timeframe=chunk1_tf,
        geo=GEO,
    )

    logger.info("Fetching Chunk 2 (Jun 1 - Jul 31)...")
    df2 = fetch_interest_over_time(
        pytrends=pytrends,
        keywords=SEARCH_TERMS,
        timeframe=chunk2_tf,
        geo=GEO,
    )

    if df1 is None or df2 is None:
        logger.error("Phase 2 chunk collection failed.")
        return None

    if df1.empty and df2.empty:
        logger.warning("Both chunks returned empty. No search data available.")
        return pd.DataFrame()

    if df1.empty or df2.empty:
        # If one chunk is empty, return the other
        df = df1 if not df1.empty else df2
        df["trends_combined"] = df[SEARCH_TERMS].max(axis=1)
        df = df.reset_index().rename(columns={"date": "Date"})
        return df

    # Normalize Chunk 2 relative to Chunk 1 using the overlap period
    # Overlap: Jun 1 - Jun 15
    overlap_start = pd.Timestamp("2026-06-01")
    overlap_end = pd.Timestamp("2026-06-15")

    overlap_1 = df1.loc[overlap_start:overlap_end]
    overlap_2 = df2.loc[overlap_start:overlap_end]

    if len(overlap_1) == 0 or len(overlap_2) == 0:
        logger.warning(
            "No overlap found for normalization. "
            "Using Chunk 1 only."
        )
        df = df1.copy()
    else:
        # Calculate scaling factor for each keyword
        for kw in SEARCH_TERMS:
            mean_1 = overlap_1[kw].mean()
            mean_2 = overlap_2[kw].mean()

            if mean_2 > 0:
                scale_factor = mean_1 / mean_2
                logger.info(
                    f"  Scaling factor for '{kw}': {scale_factor:.3f} "
                    f"(overlap means: chunk1={mean_1:.1f}, chunk2={mean_2:.1f})"
                )
                df2[kw] = (df2[kw] * scale_factor).round(1)
            else:
                logger.warning(
                    f"  Chunk 2 mean for '{kw}' is 0. Cannot normalize."
                )

        # Combine: use Chunk 1 up to Jun 15, then Chunk 2 from Jun 16
        cutoff = pd.Timestamp("2026-06-16")
        df = pd.concat([
            df1.loc[:pd.Timestamp("2026-06-15")],
            df2.loc[cutoff:],
        ])

    # Create combined index
    df["trends_combined"] = df[SEARCH_TERMS].max(axis=1)

    # Reset index
    df = df.reset_index()
    df = df.rename(columns={"date": "Date"})

    logger.info(f"Phase 2 result: {len(df)} daily observations")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    return df


# ---------------------------------------------------------------------------
# Metadata & Save
# ---------------------------------------------------------------------------

def save_metadata(output_dir: Path) -> None:
    """
    Save collection metadata for reproducibility.

    Why:
        Google Trends data can change retroactively as Google re-normalizes
        the index. Recording when data was collected is essential for
        reproducibility. Any reviewer or collaborator should know the exact
        collection parameters and timestamp.
    """
    metadata = {
        "collection_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "collection_timestamp_local": datetime.now().isoformat(),
        "search_terms": SEARCH_TERMS,
        "geographic_filter": GEO,
        "geographic_name": "Bangladesh",
        "phase1_timeframe": PHASE1_TIMEFRAME,
        "phase2_start": PHASE2_START,
        "phase2_end": PHASE2_END,
        "pytrends_version": "4.9.2",
        "python_version": sys.version,
        "notes": [
            "Google Trends returns RELATIVE search interest (0-100), "
            "not absolute search volume.",
            "Values may differ if re-downloaded at a later date due to "
            "Google's re-normalization.",
            "trends_combined = max(measles, হাম) per time period.",
            "Phase 2 daily data was collected in two overlapping chunks "
            "and normalized using overlap-period scaling.",
        ],
        "data_source": "Google Trends via pytrends (unofficial API)",
        "citation": (
            "Google Trends (https://trends.google.com). "
            "Accessed via pytrends Python library."
        ),
    }

    metadata_path = output_dir / "google_trends_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Metadata saved to {metadata_path}")


def save_dataframe(
    df: pd.DataFrame,
    filepath: Path,
    description: str,
) -> None:
    """Save DataFrame to CSV with logging."""
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info(f"Saved {description}: {filepath} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main execution function.

    Collects Google Trends data for both Phase 1 (monthly, 2012-2025)
    and Phase 2 (daily, 2026 outbreak period), saves to CSV files,
    and records collection metadata for reproducibility.
    """
    logger.info("=" * 60)
    logger.info("GOOGLE TRENDS DATA COLLECTION")
    logger.info("Measles Forecasting in Bangladesh")
    logger.info("=" * 60)

    # Step 1: Setup
    create_output_directory(OUTPUT_DIR)
    pytrends = initialize_pytrends()

    # Step 2: Collect Phase 1 (monthly)
    df_monthly = collect_phase1_monthly(pytrends)

    if df_monthly is not None and not df_monthly.empty:
        save_dataframe(
            df_monthly,
            OUTPUT_DIR / "google_trends_monthly_2012_2025.csv",
            "Phase 1 monthly Google Trends data",
        )

        # Print summary statistics
        logger.info("\nPhase 1 Summary:")
        logger.info(f"  Rows: {len(df_monthly)}")
        logger.info(f"  Date range: {df_monthly['Year_Month'].iloc[0]} "
                     f"to {df_monthly['Year_Month'].iloc[-1]}")
        for term in SEARCH_TERMS:
            if term in df_monthly.columns:
                logger.info(
                    f"  '{term}': "
                    f"mean={df_monthly[term].mean():.1f}, "
                    f"max={df_monthly[term].max()}, "
                    f"zeros={( df_monthly[term] == 0).sum()}"
                )
    else:
        logger.error("Phase 1 collection failed. Continuing to Phase 2...")

    # Step 3: Collect Phase 2 (daily)
    df_daily = collect_phase2_daily(pytrends)

    if df_daily is not None and not df_daily.empty:
        save_dataframe(
            df_daily,
            OUTPUT_DIR / "google_trends_daily_2026.csv",
            "Phase 2 daily Google Trends data",
        )

        logger.info("\nPhase 2 Summary:")
        logger.info(f"  Rows: {len(df_daily)}")
        logger.info(f"  Date range: {df_daily['Date'].iloc[0]} "
                     f"to {df_daily['Date'].iloc[-1]}")
    else:
        logger.error("Phase 2 collection failed.")

    # Step 4: Save metadata
    save_metadata(OUTPUT_DIR)

    # Step 5: Final report
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("Files created:")
    for f in OUTPUT_DIR.iterdir():
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
