"""
compute_housing_price.py
========================
Computes median housing price per sqm (SGD/sqm) per 200 m fishnet cell
by combining HDB resale and URA private residential transactions.

Data sources
------------
  HDB Resale (2017–present)
    data.gov.sg — free, no API key required
    Dataset ID : d_8b84c4ee58e3cfc0ece0d773c8ca6abc
    Fields     : month, block, street_name, floor_area_sqm, resale_price, ...
    Geocoding  : block + street_name → OneMap search API

  URA Private Residential Transactions
    URA DataMall — free registration at https://www.ura.gov.sg/maps/api/
    Set URA_ACCESS_KEY below after registering.
    Endpoint   : https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1
    Fields     : project, street, postal, unitPrice (SGD/sqm), transactionDate, ...
    Geocoding  : postal code → OneMap search API

Price metric
------------
  PRICE_PSM = median(resale_price / floor_area_sqm) over PRICE_YEARS
  for all transactions whose geocoded centroid falls within the cell.

  Cells with fewer than MIN_TRANSACTIONS valid transactions → NaN / 0.
  These are non-residential cells (parks, industrial, water bodies) or
  private enclaves with sparse transactions — noted as a limitation.

Outputs
-------
  geocoded_hdb.csv              HDB transactions with lat/lon (cached)
  geocoded_ura.csv              URA transactions with lat/lon (cached)
  housing_price.csv             per-node PRICE_PSM (OBJECTID key)
  fishnet_final_price.shp       input shapefile + PRICE_PSM column

Install
-------
  pip install geopandas pandas numpy requests
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import requests

warnings.filterwarnings("ignore")

# ── PROJ conflict fix ─────────────────────────────────────────────────────────
try:
    import pyproj as _pp
    _pd = os.path.join(os.path.dirname(_pp.__file__), "proj_dir", "share", "proj")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"]  = _pd
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

# ── !! CONFIGURE THESE !! ────────────────────────────────────────────────────
URA_ACCESS_KEY   = "5d7f0435-403d-4477-b3bd-a9cf69cc60dd"

# Token fetched manually via curl (valid for 24 hours).
# To refresh: run in Command Prompt:
#   curl "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1" -H "AccessKey: 5d7f0435-403d-4477-b3bd-a9cf69cc60dd"
# Then paste the "Result" value below.
URA_TOKEN_MANUAL = "dCd6AQ-k95Mz-z50tn2R++9+qw47dWQdr47bfd43tFc9-573EdnWcWa4f6Pd6c6dY764nRxeC0eyyd747+-6kav7f39daan06qK7"
# ─────────────────────────────────────────────────────────────────────────────

SHP_FILE       = "fishnet_final_albedo.shp"
OUT_SHP        = "fishnet_final_price.shp"
HDB_CACHE      = "geocoded_hdb.csv"
URA_CACHE      = "geocoded_ura.csv"

# Use last 3 years of transactions for a stable median
PRICE_YEARS    = [2022, 2023, 2024]
MIN_TRANS      = 3      # cells with fewer transactions → NaN

# data.gov.sg HDB resale dataset ID (Jan 2017 onwards)
HDB_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
HDB_API_URL    = "https://data.gov.sg/api/action/datastore_search"

# URA DataMall endpoint
URA_TOKEN_URL  = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"
URA_DATA_URL   = "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1"

# OneMap geocoding (no auth needed, 250 req/min)
ONEMAP_URL     = "https://www.onemap.gov.sg/api/common/elastic/search"

# SVY21 WKT — bypasses proj.db conflicts
SVY21_WKT = (
    'PROJCS["SVY21 / Singapore TM",'
    'GEOGCS["SVY21",DATUM["SVY21",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",1.36666666666667],'
    'PARAMETER["central_meridian",103.833333333333],'
    'PARAMETER["scale_factor",1],'
    'PARAMETER["false_easting",28001.642],'
    'PARAMETER["false_northing",38744.572],'
    'UNIT["metre",1]]'
)

SHP_COL_MAP = {
    'SG_200m_Fe': 'OBJECTID',
    'SG_200m__1': 'BCR',    'SG_200m__2': 'FAR',
    'SG_200m__3': 'AHI',    'SG_200m__4': 'ABV',
    'SG_200m__5': 'NB_BCR', 'SG_200m__6': 'NB_FAR',
    'SG_200m__7': 'NB_AHI', 'SG_200m__8': 'NB_ABV',
    'SG_200m__9': 'NDVI',   'SG_200m_10': 'NDWI',
    'SG_200m_11': 'DIS_MRT','SG_200m_12': 'DIS_CBD',
    'SG_200m_13': 'LST',    'SG_200m_14': 'NTL',
}

# Rate limiter: stays within 250 req/min safely
_rate_lock = threading.Semaphore(15)


# ── OneMap geocoding ──────────────────────────────────────────────────────────

def onemap_geocode(query, retries=3):
    """Geocode a query string via OneMap. Returns (lat, lon) or (None, None)."""
    for attempt in range(retries):
        try:
            r = requests.get(
                ONEMAP_URL,
                params={"searchVal": query, "returnGeom": "Y",
                        "getAddrDetails": "N"},
                timeout=10,
            )
            results = r.json().get("results", [])
            if results:
                return float(results[0]["LATITUDE"]), float(results[0]["LONGITUDE"])
        except Exception:
            time.sleep(1)
    return None, None


def _geocode_one(args):
    """Worker: geocode one query with rate limiting."""
    query, cache_file_lock, cached, cache_file = args
    with _rate_lock:
        lat, lon = onemap_geocode(query)
        time.sleep(0.05)
    return query, lat, lon


def batch_geocode(queries, cache_file, label="addresses"):
    """
    Geocode a list of unique queries with caching, using parallel threads.
    Returns dict {query: (lat, lon)}.
    """
    cached = {}
    if os.path.exists(cache_file):
        df_c = pd.read_csv(cache_file)
        for _, row in df_c.iterrows():
            cached[row["query"]] = (row["lat"], row["lon"])
        print(f"      Loaded {len(cached):,} cached geocodes.")

    to_do = [q for q in queries if q not in cached]
    print(f"      Geocoding {len(to_do):,} new {label} via OneMap (parallel)...")

    if not to_do:
        return cached

    lock = threading.Lock()
    args = [(q, lock, cached, cache_file) for q in to_do]
    completed = 0

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(_geocode_one, a): a[0] for a in args}
        for future in as_completed(futures):
            query, lat, lon = future.result()
            cached[query] = (lat, lon)
            completed += 1
            if completed % 200 == 0:
                print(f"        {completed:,} / {len(to_do):,} geocoded...")
                with lock:
                    pd.DataFrame(
                        [{"query": k, "lat": v[0], "lon": v[1]}
                         for k, v in cached.items()]
                    ).to_csv(cache_file, index=False)

    pd.DataFrame(
        [{"query": k, "lat": v[0], "lon": v[1]} for k, v in cached.items()]
    ).to_csv(cache_file, index=False)
    print(f"      Saved → {cache_file}")

    n_ok   = sum(1 for v in cached.values() if v[0] is not None)
    n_fail = len(cached) - n_ok
    print(f"      Geocoded: {n_ok:,} OK  |  {n_fail:,} failed")
    return cached


# ── Step 1: HDB resale transactions ──────────────────────────────────────────

def fetch_hdb():
    if os.path.exists(HDB_CACHE):
        cached_df = pd.read_csv(HDB_CACHE)
        if len(cached_df) > 1000:
            print(f"  Using cached HDB data: {HDB_CACHE} ({len(cached_df):,} rows)")
            return cached_df
        else:
            print(f"  Cache only has {len(cached_df)} rows — deleting and re-downloading...")
            os.remove(HDB_CACHE)

    print("  Downloading HDB resale transactions from data.gov.sg...")
    records, offset = [], 0
    while True:
        r = requests.get(
            HDB_API_URL,
            params={"resource_id": HDB_DATASET_ID,
                    "limit": 10000, "offset": offset},
            timeout=30,
        )
        batch = r.json()["result"]["records"]
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if offset % 50000 == 0:
            print(f"    {offset:,} records downloaded...")

    df = pd.DataFrame(records)
    print(f"  {len(df):,} total HDB transactions downloaded.")

    df["year"] = pd.to_datetime(df["month"]).dt.year
    df = df[df["year"].isin(PRICE_YEARS)].copy()
    print(f"  {len(df):,} transactions in {PRICE_YEARS}")

    df["resale_price"]   = pd.to_numeric(df["resale_price"],   errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df = df.dropna(subset=["resale_price", "floor_area_sqm"])
    df = df[df["floor_area_sqm"] > 0]
    df["price_psm"] = df["resale_price"] / df["floor_area_sqm"]

    # Build geo_query — try to match whatever format is already in cache
    # OneMap works best with just block + street (no BLK prefix, no SINGAPORE suffix)
    df["geo_query"] = (
        df["block"].str.strip() + " " +
        df["street_name"].str.strip()
    )

    # Show sample queries so we can verify format
    print(f"  Sample geo queries: {df['geo_query'].head(3).tolist()}")

    unique_queries = df["geo_query"].unique().tolist()
    geo_map = batch_geocode(unique_queries, "hdb_geo_cache.csv",
                            label="HDB block addresses")

    df["lat"] = df["geo_query"].map(lambda q: geo_map.get(q, (None, None))[0])
    df["lon"] = df["geo_query"].map(lambda q: geo_map.get(q, (None, None))[1])

    # Diagnose: how many cache entries actually have valid coords?
    valid_in_cache = {k: v for k, v in geo_map.items() if v[0] is not None}
    print(f"  Cache entries with valid lat/lon: {len(valid_in_cache):,} / {len(geo_map):,}")

    n_matched = df[["lat","lon"]].notna().all(axis=1).sum()
    print(f"  Transactions matched to geocode: {n_matched:,} / {len(df):,}")

    # Show sample of failed lookups
    failed = df[df["lat"].isna()]["geo_query"].head(5).tolist()
    print(f"  Sample failed queries: {failed}")

    df = df.dropna(subset=["lat", "lon"])

    out = df[["month", "year", "block", "street_name",
              "flat_type", "floor_area_sqm", "resale_price",
              "price_psm", "lat", "lon"]].copy()
    out.to_csv(HDB_CACHE, index=False)
    print(f"  Saved → {HDB_CACHE}  ({len(out):,} geocoded transactions)")
    return out


# ── Step 2: URA private residential transactions ──────────────────────────────

def _get_ura_token():
    """Return the manually fetched URA token (valid 24h)."""
    if not URA_TOKEN_MANUAL:
        raise RuntimeError(
            "URA token is empty!\n"
            "Run this in Command Prompt to get a fresh token:\n"
            '  curl "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"'
            ' -H "AccessKey: 5d7f0435-403d-4477-b3bd-a9cf69cc60dd"\n'
            "Then paste the Result value into URA_TOKEN_MANUAL at the top of this script."
        )
    return URA_TOKEN_MANUAL


def fetch_ura():
    if os.path.exists(URA_CACHE):
        print(f"  Using cached URA data: {URA_CACHE}")
        return pd.read_csv(URA_CACHE)

    print("  Downloading URA private residential transactions...")
    token   = _get_ura_token()
    headers = {
        "AccessKey":      URA_ACCESS_KEY,
        "Token":          token,
        "User-Agent":     "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept":         "application/json, text/plain, */*",
        "Accept-Language":"en-US,en;q=0.9",
        "Referer":        "https://eservice.ura.gov.sg/",
    }

    all_records = []
    for batch in range(1, 5):
        r = requests.get(
            URA_DATA_URL,
            params={"service": "PMI_Resi_Transaction", "batch": batch},
            headers=headers,
            timeout=30,
        )
        # Debug: print raw response if it looks wrong
        if r.text.strip().startswith("<!") or not r.text.strip():
            print(f"  !! Batch {batch} returned unexpected response (HTTP {r.status_code}):")
            print(f"     {r.text[:300]}")
            print("  Tip: Try fetching data via curl (see below) and save as ura_raw.json")
            print(f'  curl "{URA_DATA_URL}?service=PMI_Resi_Transaction&batch={batch}" '
                  f'-H "AccessKey: {URA_ACCESS_KEY}" -H "Token: {token}" -o ura_raw_b{batch}.json')
            continue
        data = r.json().get("Result", [])
        for project in data:
            project_name = project.get("project", "")
            street       = project.get("street", "")
            postal       = project.get("postal", "")
            for t in project.get("transaction", []):
                all_records.append({
                    "project":         project_name,
                    "street":          street,
                    "postal":          postal,
                    "transactionDate": t.get("contractDate", ""),
                    "typeOfSale":      t.get("typeOfSale", ""),
                    "price":           t.get("price", ""),
                    "area":            t.get("area", ""),
                    "unitPrice":       t.get("unitPrice", ""),
                    "propertyType":    t.get("propertyType", ""),
                    "tenure":          t.get("tenure", ""),
                    "floorRange":      t.get("floorRange", ""),
                })
        print(f"    Batch {batch}: {len(all_records):,} records so far")

    df = pd.DataFrame(all_records)
    print(f"  {len(df):,} total URA transactions downloaded.")

    # Debug: show sample date formats to confirm parsing
    print(f"  Sample contractDate values: {df['transactionDate'].dropna().head(5).tolist()}")

    # URA contractDate is "MMYY" (4 digits) e.g. "0522" = May 2022, "0724" = July 2024
    def parse_ura_year(s):
        try:
            s = str(s).strip()
            if len(s) == 4 and s.isdigit():
                yy = int(s[2:])          # last 2 digits = year
                return 2000 + yy
            # fallback: slash-separated "MM/YY"
            parts = s.split("/")
            if len(parts) == 2:
                yy = int(parts[1])
                return 2000 + yy if yy < 100 else yy
        except Exception:
            pass
        return None

    df["year"] = df["transactionDate"].apply(parse_ura_year)
    df = df[df["year"].isin(PRICE_YEARS)].copy()
    print(f"  {len(df):,} transactions in {PRICE_YEARS}")

    # Debug: check raw columns and unitPrice BEFORE filtering
    print(f"  Columns in URA data  : {list(df.columns)}")
    print(f"  Sample unitPrice raw : {df['unitPrice'].head(5).tolist()}")
    print(f"  Sample price raw     : {df['price'].head(5).tolist()}")
    print(f"  Sample area raw      : {df['area'].head(5).tolist()}")

    # unitPrice in URA is PSM already; if empty compute from price/area
    df["price_psm"] = pd.to_numeric(df["unitPrice"], errors="coerce")
    no_unit_price = df["price_psm"].isna()
    if no_unit_price.any():
        price_num = pd.to_numeric(df.loc[no_unit_price, "price"], errors="coerce")
        area_num  = pd.to_numeric(df.loc[no_unit_price, "area"],  errors="coerce")
        df.loc[no_unit_price, "price_psm"] = price_num / area_num

    print(f"  Valid price_psm after parsing: {df['price_psm'].notna().sum():,} / {len(df):,}")
    df = df.dropna(subset=["price_psm"])
    df = df[df["price_psm"] > 0]

    # Debug: check postal code field
    print(f"  Sample postal values : {df['postal'].head(5).tolist()}")
    print(f"  Sample unitPrice     : {df['unitPrice'].head(5).tolist()}")
    n_empty_postal = (df["postal"].str.strip() == "").sum()
    n_na_postal    = df["postal"].isna().sum()
    print(f"  Empty postal: {n_empty_postal:,}  |  NaN postal: {n_na_postal:,}  |  Total: {len(df):,}")

    # OneMap works best with just the 6-digit postal code (no "SINGAPORE" prefix)
    df["postal_clean"] = df["postal"].str.strip().str.zfill(6)
    has_postal = (df["postal_clean"] != "000000") & (df["postal_clean"] != "")
    df["geo_query"] = df["postal_clean"]   # just the 6-digit postal code
    no_postal = ~has_postal
    df.loc[no_postal, "geo_query"] = (
        df.loc[no_postal, "project"].str.strip() + " " +
        df.loc[no_postal, "street"].str.strip()
    )
    print(f"  Sample geo_query     : {df['geo_query'].head(5).tolist()}")

    unique_queries = df["geo_query"].unique().tolist()
    print(f"  Unique geo queries   : {len(unique_queries):,}")
    geo_map = batch_geocode(unique_queries, "ura_geo_cache.csv",
                            label="URA postal codes")

    df["lat"] = df["geo_query"].map(lambda q: geo_map.get(q, (None, None))[0])
    df["lon"] = df["geo_query"].map(lambda q: geo_map.get(q, (None, None))[1])
    df = df.dropna(subset=["lat", "lon"])

    out = df[["project", "street", "postal", "year", "typeOfSale",
              "propertyType", "price_psm", "lat", "lon"]].copy()
    out.to_csv(URA_CACHE, index=False)
    print(f"  Saved → {URA_CACHE}  ({len(out):,} geocoded transactions)")
    return out


# ── Step 3: Combine and spatial join ─────────────────────────────────────────

def build_price_gdf(hdb_df, ura_df):
    """Combine HDB and URA into a single GeoDataFrame with price_psm and source."""
    parts = []
    if not hdb_df.empty:
        hdb_df = hdb_df.copy()
        hdb_df["source"] = "HDB"
        parts.append(hdb_df[["price_psm", "lat", "lon", "source"]])
    if not ura_df.empty:
        ura_df = ura_df.copy()
        ura_df["source"] = "URA"
        parts.append(ura_df[["price_psm", "lat", "lon", "source"]])

    if not parts:
        raise RuntimeError("No transaction data available.")

    combined = pd.concat(parts, ignore_index=True)
    n_hdb = (combined["source"] == "HDB").sum()
    n_ura = (combined["source"] == "URA").sum()
    print(f"\n  Combined: {len(combined):,} transactions  (HDB: {n_hdb:,}  URA: {n_ura:,})")
    print(f"  HDB median PSM : SGD {combined.loc[combined.source=='HDB','price_psm'].median():,.0f}/sqm")
    if n_ura > 0:
        print(f"  URA median PSM : SGD {combined.loc[combined.source=='URA','price_psm'].median():,.0f}/sqm")

    gdf = gpd.GeoDataFrame(
        combined,
        geometry=[Point(lon, lat) for lat, lon
                  in zip(combined["lat"], combined["lon"])],
        crs="EPSG:4326",
    ).to_crs(SVY21_WKT)
    return gdf


def compute_price_per_cell(gdf_trans, gdf_fishnet):
    """
    Spatial join transactions to fishnet cells.
    PRICE_PSM = transaction-count weighted combination of HDB and URA medians:
        PRICE_PSM = (n_hdb * median_hdb + n_ura * median_ura) / (n_hdb + n_ura)
    This prevents a single high-value condo from dominating a cell with many HDB units.
    """
    print("  Spatial join: transactions → fishnet cells...")
    joined = gpd.sjoin(
        gdf_trans,
        gdf_fishnet[["OBJECTID", "geometry"]],
        how="left",
        predicate="within",
    )

    # Aggregate HDB and URA separately per cell
    def cell_stats(grp):
        hdb = grp[grp["source"] == "HDB"]["price_psm"]
        ura = grp[grp["source"] == "URA"]["price_psm"]
        n_hdb = len(hdb)
        n_ura = len(ura)
        med_hdb = hdb.median() if n_hdb > 0 else np.nan
        med_ura = ura.median() if n_ura > 0 else np.nan
        n_total = n_hdb + n_ura

        # Weighted combination
        if n_hdb > 0 and n_ura > 0:
            price = (n_hdb * med_hdb + n_ura * med_ura) / n_total
        elif n_hdb > 0:
            price = med_hdb
        elif n_ura > 0:
            price = med_ura
        else:
            price = np.nan

        return pd.Series({
            "PRICE_PSM": price,
            "n_trans":   n_total,
            "n_hdb":     n_hdb,
            "n_ura":     n_ura,
            "med_hdb":   med_hdb,
            "med_ura":   med_ura,
        })

    agg = joined.groupby("OBJECTID").apply(cell_stats).reset_index()

    all_ids = gdf_fishnet[["OBJECTID"]].copy()
    agg     = all_ids.merge(agg, on="OBJECTID", how="left")
    for col in ["n_trans", "n_hdb", "n_ura"]:
        agg[col] = agg[col].fillna(0).astype(int)

    agg.loc[agg["n_trans"] < MIN_TRANS, "PRICE_PSM"] = np.nan

    n_valid = agg["PRICE_PSM"].notna().sum()
    n_nan   = agg["PRICE_PSM"].isna().sum()
    n_mixed = ((agg["n_hdb"] > 0) & (agg["n_ura"] > 0)).sum()
    print(f"  Cells with valid price (≥{MIN_TRANS} trans): {n_valid:,}")
    print(f"  Cells with NaN / 0 (<{MIN_TRANS} trans)    : {n_nan:,}")
    print(f"  Cells with BOTH HDB and URA transactions   : {n_mixed:,}")
    print(f"  (NaN = parks, industrial, water, private enclaves — paper limitation)")
    return agg[["OBJECTID", "PRICE_PSM", "n_trans", "n_hdb", "n_ura", "med_hdb", "med_ura"]]


# ── Step 4: Load fishnet, join, save ─────────────────────────────────────────

def load_fishnet():
    gdf = gpd.read_file(SHP_FILE).rename(columns=SHP_COL_MAP)
    return gdf, gdf.to_crs(SVY21_WKT)


def save_results(gdf, price_df):
    print("[4/4] Saving...")
    gdf = gdf.copy()
    merge_cols = ["OBJECTID", "PRICE_PSM", "n_trans", "n_hdb", "n_ura", "med_hdb", "med_ura"]
    gdf = gdf.merge(price_df[merge_cols], on="OBJECTID", how="left")

    gdf["PRICE_PSM"] = gdf["PRICE_PSM"].fillna(0.0).round(0)
    gdf["n_trans"]   = gdf["n_trans"].fillna(0).astype(int)
    gdf["n_hdb"]     = gdf["n_hdb"].fillna(0).astype(int)
    gdf["n_ura"]     = gdf["n_ura"].fillna(0).astype(int)

    s_valid = gdf.loc[gdf["PRICE_PSM"] > 0, "PRICE_PSM"]
    n_hdb_only = ((gdf["n_hdb"] > 0) & (gdf["n_ura"] == 0) & (gdf["PRICE_PSM"] > 0)).sum()
    n_ura_only = ((gdf["n_ura"] > 0) & (gdf["n_hdb"] == 0) & (gdf["PRICE_PSM"] > 0)).sum()
    n_mixed    = ((gdf["n_hdb"] > 0) & (gdf["n_ura"] > 0)).sum()

    print(f"\n{'='*60}")
    print("  PRICE_PSM SUMMARY  (weighted HDB + URA, SGD/sqm)")
    print(f"{'='*60}")
    print(f"  Cells with data    : {len(s_valid):,}")
    print(f"    HDB-only cells   : {n_hdb_only:,}")
    print(f"    URA-only cells   : {n_ura_only:,}")
    print(f"    Mixed HDB+URA    : {n_mixed:,}")
    print(f"  Cells = 0 (no data): {(gdf['PRICE_PSM']==0).sum():,}")
    print(f"  Median  : SGD {s_valid.median():>8,.0f} /sqm")
    print(f"  Mean    : SGD {s_valid.mean():>8,.0f} /sqm")
    print(f"  Min     : SGD {s_valid.min():>8,.0f} /sqm")
    print(f"  Max     : SGD {s_valid.max():>8,.0f} /sqm")
    print(f"\n  Weighting formula: (n_hdb * median_hdb + n_ura * median_ura) / (n_hdb + n_ura)")
    print(f"  Note: PRICE_PSM = 0 for {(gdf['PRICE_PSM']==0).sum():,} cells")
    print(f"  (non-residential / private enclaves / sparse transactions)")

    # Save CSV with full breakdown
    gdf[["OBJECTID", "PRICE_PSM", "n_trans", "n_hdb", "n_ura", "med_hdb", "med_ura"]].to_csv(
        "housing_price.csv", index=False
    )
    print("\nSaved → housing_price.csv  (includes n_hdb, n_ura, med_hdb, med_ura columns)")
    gdf.drop(columns=["n_trans", "n_hdb", "n_ura", "med_hdb", "med_ura"]).to_file(OUT_SHP)
    print(f"Saved → {OUT_SHP}")
    print(f"Columns: {[c for c in gdf.columns if c != 'geometry']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HOUSING PRICE — HDB + URA COMBINED")
    print(f"  Period: {PRICE_YEARS}")
    print("=" * 60)

    print("\n[1/4] Fetching HDB resale transactions...")
    hdb_df = fetch_hdb()

    print("\n[2/4] Fetching URA private residential transactions...")
    ura_df = fetch_ura()

    print("\n[3/4] Combining and aggregating to fishnet cells...")
    gdf, gdf_proj = load_fishnet()
    gdf_trans     = build_price_gdf(hdb_df, ura_df)
    price_df      = compute_price_per_cell(gdf_trans, gdf_proj)

    save_results(gdf, price_df)
