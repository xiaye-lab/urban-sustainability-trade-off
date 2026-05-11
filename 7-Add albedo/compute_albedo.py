"""
compute_albedo.py
==================
Computes broadband surface Albedo per 200 m fishnet cell from the
Landsat 8/9 Collection 2 Level-2 Surface Reflectance raster exported
from Google Earth Engine (gee_albedo_gvi.js).

Why Albedo is an urban morphology indicator (not environmental)
---------------------------------------------------------------
  Albedo is a physical property of the built surface — determined by
  building materials, roof colour, road surface type, and pavement
  design. It is a morphological input that drives LST outcomes rather
  than an environmental outcome itself. Stewart & Oke (2012) explicitly
  include albedo as a defining surface property in the Local Climate
  Zone classification framework.

  Reference: Liang, S. (2001). Narrowband to broadband conversions of
  land surface albedo I. Remote Sensing of Environment, 76(2), 213–238.
  https://doi.org/10.1016/S0034-4257(00)00205-4

Data source
-----------
  Albedo_SG_30m.tif — exported from gee_albedo_gvi.js
  (Google Earth Engine → Google Drive → Landsat_8_SG folder)

  Formula (computed in GEE):
    α = 0.356·B2 + 0.130·B4 + 0.373·B5 + 0.085·B6 + 0.072·B7 − 0.0018
    Bands are Landsat 8/9 OLI surface reflectance scaled to [0, 1].

Outputs
-------
  albedo.csv                  per-node Albedo (OBJECTID key)
  fishnet_final_albedo.shp    input shapefile + Albedo column
"""

import os
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol

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

# ── Configuration ─────────────────────────────────────────────────────────────
ALBEDO_FILE = "Albedo_SG_30m.tif"         # exported from gee_albedo_gvi.js
SHP_FILE    = "fishnet_final_bhd.shp"   # latest input shapefile
OUT_SHP     = "fishnet_final_albedo.shp"  # output with Albedo column
# ─────────────────────────────────────────────────────────────────────────────

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


# ── Step 1: Load Albedo raster ────────────────────────────────────────────────

def load_albedo():
    if not os.path.exists(ALBEDO_FILE):
        raise FileNotFoundError(
            f"Albedo raster not found: {ALBEDO_FILE}\n"
            "Run gee_albedo_gvi.js in Google Earth Engine first:\n"
            "  https://code.earthengine.google.com\n"
            "Then download Albedo_SG_30m.tif from Google Drive."
        )

    print(f"[1/3] Loading {ALBEDO_FILE}...")
    with rasterio.open(ALBEDO_FILE) as src:
        data      = src.read(1).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        nodata    = src.nodata if src.nodata is not None else -9999

    data = np.where(data == nodata, np.nan, data)
    data = np.clip(data, 0, 1)   # albedo must be in [0, 1]

    valid = data[~np.isnan(data)]
    print(f"      Shape : {data.shape}  |  CRS: {crs}")
    print(f"      mean  = {valid.mean():.4f}  "
          f"std = {valid.std():.4f}  "
          f"min = {valid.min():.4f}  "
          f"max = {valid.max():.4f}")

    return data, transform, crs


# ── Step 2: Aggregate to 200 m fishnet cells ──────────────────────────────────

def extract_window(data, transform, bounds):
    """Slice raster array to the pixel window for one fishnet cell."""
    minx, miny, maxx, maxy = bounds
    r0, c0 = rowcol(transform, minx, maxy, op=int)
    r1, c1 = rowcol(transform, maxx, miny, op=int)
    nr, nc = data.shape
    r0, r1 = max(r0, 0), min(r1 + 1, nr)
    c0, c1 = max(c0, 0), min(c1 + 1, nc)
    if r0 >= r1 or c0 >= c1:
        return None
    return data[r0:r1, c0:c1]


def aggregate(gdf_proj, data, transform):
    """
    Compute mean Albedo per cell.
    Mean over all valid (non-NaN) 30 m pixels within each 200 m cell.
    At 30 m resolution each 200 m cell contains ~44 pixels.
    """
    print(f"[2/3] Aggregating Albedo to {len(gdf_proj):,} cells...")

    vals    = []
    n_empty = 0

    for row in gdf_proj.itertuples():
        window = extract_window(data, transform, row.geometry.bounds)
        if window is None or window.size == 0:
            vals.append(np.nan)
            n_empty += 1
            continue
        valid = window[~np.isnan(window)]
        vals.append(float(valid.mean()) if valid.size > 0 else np.nan)

    if n_empty:
        print(f"      {n_empty} cells outside raster extent → NaN")

    return np.array(vals)


# ── Step 3: Attach to shapefile and save ─────────────────────────────────────

def save_results(gdf, albedo_vals):
    print("[3/3] Saving...")
    gdf = gdf.copy()
    gdf["Albedo"] = albedo_vals.round(4)

    # Fill NaN (boundary / water cells) with column median
    n_nan = gdf["Albedo"].isna().sum()
    if n_nan:
        gdf["Albedo"] = gdf["Albedo"].fillna(gdf["Albedo"].median())
        print(f"      {n_nan:,} NaN → median fill (boundary / water cells)")

    s = gdf["Albedo"]
    print(f"\n{'='*55}")
    print("  ALBEDO SUMMARY  (urban morphology indicator)")
    print(f"{'='*55}")
    print(f"  mean  = {s.mean():.4f}")
    print(f"  std   = {s.std():.4f}")
    print(f"  min   = {s.min():.4f}")
    print(f"  max   = {s.max():.4f}")
    print(f"\n  Interpretation:")
    print(f"    > 0.25  high-reflectance (light roofs, bare concrete, sand)")
    print(f"    0.10–0.25  mixed urban surfaces")
    print(f"    < 0.10  dark surfaces (asphalt, dark rooftops, water)")

    gdf[["OBJECTID", "Albedo"]].to_csv("albedo.csv", index=False)
    print("\nSaved → albedo.csv")

    gdf.to_file(OUT_SHP)
    print(f"Saved → {OUT_SHP}")
    print(f"New column: Albedo  |  Total: {len(gdf.columns)} columns")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  ALBEDO — Urban Morphology Indicator")
    print("  Source: Landsat 8/9 C2 L2  |  Liang (2001) formula")
    print("=" * 55)

    # Load shapefile
    print(f"\nLoading {SHP_FILE}...")
    gdf           = gpd.read_file(SHP_FILE).rename(columns=SHP_COL_MAP)
    gdf_raster_crs = None   # set after loading raster
    print(f"  {len(gdf):,} nodes")

    # Load raster
    albedo_data, raster_transform, raster_crs = load_albedo()

    # Reproject fishnet to match raster CRS (WGS84)
    gdf_raster_crs = gdf.to_crs(str(raster_crs))

    # Aggregate
    albedo_vals = aggregate(gdf_raster_crs, albedo_data, raster_transform)

    # Save
    save_results(gdf, albedo_vals)
