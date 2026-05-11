"""
compute_landscape_metrics.py
=============================
Downloads URA Master Plan 2025 (data.gov.sg), rasterizes to 10 m, and computes
four FRAGSTATS-equivalent landscape metrics per 200 m fishnet cell.

Data source (free, CC-BY 4.0, citable)
---------------------------------------
  Urban Redevelopment Authority. (2026). Master Plan 2025 Land Use Layer
  [Dataset]. data.gov.sg.
  https://data.gov.sg/datasets/d_a8c3546b26712e35021f3a681d0353ae/view

Metrics (FRAGSTATS conventions, verified against McGarigal et al. 2012)
------------------------------------------------------------------------
  LSI   Landscape Shape Index         0.25 × E / √A  (PyLandStats, count_boundary=True)
  LPI   Largest Patch Index (%)       max patch area / total area × 100  (PyLandStats)
  MENN  Mean Euclidean Nearest        edge-to-edge via scipy EDT, area-weighted
        Neighbor Distance (m)         across classes; 0 = fully contiguous cell
  SHDI  Shannon's Diversity Index     −Σ pᵢ ln pᵢ  (PyLandStats); 0 = single-class cell

URA land use grouping (8 classes + nodata=0)
--------------------------------------------
  1  Residential          2  Commercial            3  Business/Industrial
  4  Civic/Institutional  5  Open Space/Recreation 6  Utilities/Transport
  7  Reserve/Special      8  Agriculture/Nature

Install
-------
  pip install pylandstats rasterio geopandas requests numpy pandas scipy

Outputs
-------
  ura_lulc_10m.tif         rasterized URA zoning at 10 m (cached for reuse)
  fishnet_final.shp        all original columns + LSI, LPI, MENN, SHDI
"""

import io
import os
import time
import warnings
import zipfile

# ── PROJ conflict fix (PostgreSQL PostGIS vs Python pyproj) ──────────────────
# Must run before any rasterio/pyproj import. Uses hardcoded WKT for SVY21
# as fallback so EPSG lookup never touches a conflicting proj.db.
try:
    import pyproj as _pp
    _pd = os.path.join(os.path.dirname(_pp.__file__), "proj_dir", "share", "proj")
    if os.path.isdir(_pd):
        os.environ["PROJ_DATA"] = _pd
        os.environ["PROJ_LIB"]  = _pd
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rasterio
from rasterio.transform import from_bounds, rowcol
from rasterio.features import rasterize
from scipy.ndimage import label as _cc, distance_transform_edt as _edt
import pylandstats as pls

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
SHP_FILE     = "fishnet_subset_netdist.shp"   # input shapefile
OUT_SHP      = "fishnet_final.shp"            # output: all columns + LSI/LPI/MENN/SHDI
LULC_CACHE   = "ura_lulc_10m.tif"            # cached raster (skips download on rerun)
PIXEL_SIZE_M = 10                             # rasterisation resolution (metres)

# data.gov.sg dataset ID — URA Master Plan 2025 Land Use Layer
DATASET_ID = "d_a8c3546b26712e35021f3a681d0353ae"
API_URL    = (f"https://api-open.data.gov.sg/v1/public/api/datasets/"
              f"{DATASET_ID}/poll-download")

# Singapore bounding box in SVY21 metres
SG_EXTENT = (2000, 14000, 52000, 56000)   # (minx, miny, maxx, maxy)

# SVY21 CRS as WKT — bypasses proj.db lookup entirely (avoids PROJ version conflicts)
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

NODATA = 0   # pixels outside URA zoning extent (sea, unclassified)

# ── Column rename: truncated DBF names → readable names ──────────────────────
SHP_COL_MAP = {
    'SG_200m_Fe': 'OBJECTID',  # SG_200m_FeatureID  →  OBJECTID   (Feature ID)
    'SG_200m__1': 'BCR',       # SG_200m_BCR        →  BCR        (Building Coverage Ratio)
    'SG_200m__2': 'FAR',       # SG_200m_FAR        →  FAR        (Floor Area Ratio)
    'SG_200m__3': 'AHI',       # SG_200m_AHI        →  AHI        (Average Height Index)
    'SG_200m__4': 'ABV',       # SG_200m_ABV        →  ABV        (Average Building Volume)
    'SG_200m__5': 'NB_BCR',    # SG_200m_NB_BCR     →  NB_BCR     (Neighbour BCR)
    'SG_200m__6': 'NB_FAR',    # SG_200m_NB_FAR     →  NB_FAR     (Neighbour FAR)
    'SG_200m__7': 'NB_AHI',    # SG_200m_NB_AHI     →  NB_AHI     (Neighbour AHI)
    'SG_200m__8': 'NB_ABV',    # SG_200m_NB_ABV     →  NB_ABV     (Neighbour ABV)
    'SG_200m__9': 'NDVI',      # SG_200m_NDVI       →  NDVI       (Normalised Difference Vegetation Index)
    'SG_200m_10': 'NDWI',      # SG_200m_NDWI       →  NDWI       (Normalised Difference Water Index)
    'SG_200m_11': 'DIS_MRT',   # SG_200m_DIS_MRT    →  DIS_MRT    (Distance to MRT — network distance)
    'SG_200m_12': 'DIS_CBD',   # SG_200m_DIS_CBD    →  DIS_CBD    (Distance to CBD)
    'SG_200m_13': 'LST',       # SG_200m_LST        →  LST        (Land Surface Temperature)
    'SG_200m_14': 'NTL',       # SG_200m_NTL        →  NTL        (Nighttime Light)
    # Already correctly named: POI, RND, Shape_Leng, Shape_Area, PM25, CO2, POPU
}

# ── URA land use → integer class code ────────────────────────────────────────
LU_CLASS_MAP = {
    "RESIDENTIAL":                            1,
    "RESIDENTIAL WITH COMMERCIAL AT 1ST STOREY": 1,
    "COMMERCIAL":                             2,
    "COMMERCIAL & RESIDENTIAL":               2,
    "HOTEL":                                  2,
    "BUSINESS 1":                             3,
    "BUSINESS 2":                             3,
    "BUSINESS 1 - WHITE":                     3,
    "BUSINESS 2 - WHITE":                     3,
    "BUSINESS PARK":                          3,
    "BUSINESS PARK - WHITE":                  3,
    "CIVIC & COMMUNITY INSTITUTION":          4,
    "EDUCATIONAL INSTITUTION":                4,
    "HEALTH & MEDICAL CARE":                  4,
    "PLACE OF WORSHIP":                       4,
    "PARK":                                   5,
    "OPEN SPACE":                             5,
    "SPORT & RECREATION":                     5,
    "TRANSPORT FACILITIES":                   6,
    "UTILITY":                                6,
    "ROAD":                                   6,
    "WHITE":                                  7,
    "RESERVE SITE":                           7,
    "SPECIAL USE":                            7,
    "AGRICULTURE":                            8,
    "CEMETERY":                               8,
}
CLASS_LABELS = {
    1: "Residential",        2: "Commercial",          3: "Business/Industrial",
    4: "Civic/Institutional",5: "Open Space/Recreation",6: "Utilities/Transport",
    7: "Reserve/Special",    8: "Agriculture/Nature",
}


# ── Step 1: Fetch and rasterize URA Master Plan ───────────────────────────────

def fetch_ura_geojson():
    print("  Requesting download URL from data.gov.sg...")
    r = requests.get(API_URL, timeout=30)
    r.raise_for_status()
    j = r.json()
    if j.get("code") != 0:
        raise RuntimeError(f"API error: {j.get('errMsg')}")
    dl_url = j["data"]["url"]
    print("  Downloading GeoJSON...")
    r2 = requests.get(dl_url, timeout=120)
    r2.raise_for_status()
    is_zip = (dl_url.endswith(".zip") or
              r2.headers.get("content-type", "").startswith("application/zip"))
    if is_zip:
        with zipfile.ZipFile(io.BytesIO(r2.content)) as z:
            name = [n for n in z.namelist() if n.endswith(".geojson")][0]
            return gpd.read_file(z.open(name))
    return gpd.read_file(io.BytesIO(r2.content))


def build_lulc_raster(gdf_ura):
    """Rasterize URA polygons to 10 m grid in SVY21. Returns (array, transform)."""
    gdf = gdf_ura.to_crs(SVY21_WKT).copy()

    lu_col = next(
        c for c in gdf.columns
        if c.upper() in ("LU_DESC", "LANDUSE", "LAND_USE", "LU_TYPE")
    )
    gdf["class_id"] = (
        gdf[lu_col].str.upper().str.strip()
                   .map({k.upper(): v for k, v in LU_CLASS_MAP.items()})
                   .fillna(NODATA).astype(int)
    )
    print(f"  Land use column  : '{lu_col}'")
    print(f"  Class counts     :")
    for cid, label in CLASS_LABELS.items():
        n = (gdf["class_id"] == cid).sum()
        if n:
            print(f"    {cid}  {label:<30}  {n:,} polygons")

    minx, miny, maxx, maxy = SG_EXTENT
    width     = int((maxx - minx) / PIXEL_SIZE_M)
    height    = int((maxy - miny) / PIXEL_SIZE_M)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    shapes = (
        (geom, val)
        for geom, val in zip(gdf.geometry, gdf["class_id"])
        if val != NODATA and geom is not None and not geom.is_empty
    )
    raster = rasterize(
        shapes, out_shape=(height, width),
        transform=transform, fill=NODATA, dtype=np.uint8,
    )

    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": width, "height": height,
        "count": 1, "crs": SVY21_WKT, "transform": transform,
        "nodata": NODATA, "compress": "lzw",
    }
    with rasterio.open(LULC_CACHE, "w", **profile) as dst:
        dst.write(raster, 1)

    print(f"  Rasterized       : {height} × {width} px at {PIXEL_SIZE_M} m")
    return raster, transform


def fetch_lulc():
    if os.path.exists(LULC_CACHE):
        print(f"[1/4] Loading cached raster: {LULC_CACHE}")
        with rasterio.open(LULC_CACHE) as src:
            data      = src.read(1)
            transform = src.transform
            pixel_res = src.res
        classes = np.unique(data[data != NODATA])
        print(f"      Shape {data.shape}  |  {len(classes)} classes present")
        return data, transform, pixel_res

    print("[1/4] Fetching URA Master Plan 2025 from data.gov.sg...")
    gdf_ura = fetch_ura_geojson()
    print(f"      {len(gdf_ura):,} zoning polygons downloaded.")
    data, transform = build_lulc_raster(gdf_ura)
    print(f"      Saved → {LULC_CACHE}  (reused on future runs)")
    return data, transform, (PIXEL_SIZE_M, PIXEL_SIZE_M)


# ── Step 2: Load shapefile ────────────────────────────────────────────────────

def load_fishnet():
    print("[2/4] Loading shapefile...")
    gdf      = gpd.read_file(SHP_FILE).rename(columns=SHP_COL_MAP)
    gdf_proj = gdf.to_crs(SVY21_WKT)
    print(f"      {len(gdf):,} nodes  |  {len(gdf.columns)} columns")
    return gdf, gdf_proj


# ── Step 3: Per-cell metric computation ──────────────────────────────────────

def _extract_window(raster_data, transform, bounds):
    """Slice pixel array for one cell bounding box via array indexing."""
    minx, miny, maxx, maxy = bounds
    r0, c0 = rowcol(transform, minx, maxy, op=int)
    r1, c1 = rowcol(transform, maxx, miny, op=int)
    nr, nc = raster_data.shape
    r0, r1 = max(r0, 0), min(r1 + 1, nr)
    c0, c1 = max(c0, 0), min(c1 + 1, nc)
    if r0 >= r1 or c0 >= c1:
        return None
    return raster_data[r0:r1, c0:c1]


def _menn_edgetoedge(arr, pixel_size_m):
    """
    Edge-to-edge MENN per FRAGSTATS (McGarigal et al. 2012).
    Uses scipy distance_transform_edt; subtracts one pixel for edge-to-edge.

    Returns:
      0.0  — all classes have exactly one patch (no fragmentation)
      NaN  — cell is empty / all nodata
    """
    valid_mask = arr != NODATA
    if valid_mask.sum() == 0:
        return np.nan
    struct = np.ones((3, 3), int)   # 8-connectivity (FRAGSTATS default)
    all_dists, all_areas = [], []
    for cls in np.unique(arr[valid_mask]):
        cls_mask = arr == cls
        labeled, n_patches = _cc(cls_mask, structure=struct)
        if n_patches < 2:
            continue
        for pid in range(1, n_patches + 1):
            this_patch    = labeled == pid
            other_patches = cls_mask & ~this_patch
            if not other_patches.any():
                continue
            dist_c2c = _edt(~other_patches) * pixel_size_m
            nn_e2e   = max(0.0, dist_c2c[this_patch].min() - pixel_size_m)
            all_dists.append(nn_e2e)
            all_areas.append(int(this_patch.sum()))
    if not all_dists:
        return 0.0
    return float(np.average(all_dists, weights=all_areas))


def _cell_metrics(arr, pixel_res):
    """
    Compute LSI, LPI, MENN, SHDI for one 200 m cell.
    Returns (nan, nan, nan, nan) for empty or sub-4-pixel cells.
    """
    if arr is None or (arr != NODATA).sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    try:
        ls = pls.Landscape(arr, res=pixel_res, nodata=NODATA)
        lm = ls.compute_landscape_metrics_df(metrics=[
            "landscape_shape_index",
            "largest_patch_index",
            "shannon_diversity_index",
        ])
        lsi  = float(lm["landscape_shape_index"].iloc[0])
        lpi  = float(lm["largest_patch_index"].iloc[0])
        shdi = float(lm["shannon_diversity_index"].iloc[0])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    menn = _menn_edgetoedge(arr, pixel_res[0])
    return lsi, lpi, menn, shdi


def compute_all_cells(gdf_proj, raster_data, transform, pixel_res):
    print(f"[3/4] Computing LSI, LPI, MENN, SHDI for {len(gdf_proj):,} cells...")
    t0 = time.time()
    lsi_v, lpi_v, menn_v, shdi_v = [], [], [], []
    for i, row in enumerate(gdf_proj.itertuples(), 1):
        arr = _extract_window(raster_data, transform, row.geometry.bounds)
        lsi, lpi, menn, shdi = _cell_metrics(arr, pixel_res)
        lsi_v.append(lsi);  lpi_v.append(lpi)
        menn_v.append(menn); shdi_v.append(shdi)
        if i % 1000 == 0:
            elapsed = time.time() - t0
            print(f"      {i:,} / {len(gdf_proj):,}  "
                  f"({elapsed:.0f}s elapsed  ~{elapsed/i*(len(gdf_proj)-i):.0f}s left)")
    print(f"      Done in {time.time()-t0:.0f}s")
    return {"LSI": lsi_v, "LPI": lpi_v, "MENN": menn_v, "SHDI": shdi_v}


# ── Step 4: Fill NaN, attach to shapefile, save ───────────────────────────────

def save_results(gdf, results):
    print("[4/4] Saving...")
    for col, vals in results.items():
        gdf[col] = vals

    # LSI, LPI, MENN: NaN = boundary / water cell → median fill
    for col in ["LSI", "LPI", "MENN"]:
        n = gdf[col].isna().sum()
        if n:
            gdf[col] = gdf[col].fillna(gdf[col].median())
            print(f"      {col}  : {n:,} boundary/water cells → median fill")

    # SHDI: NaN = single-class cell (mathematically 0, not missing)
    n_shdi = gdf["SHDI"].isna().sum()
    if n_shdi:
        gdf["SHDI"] = gdf["SHDI"].fillna(0.0)
        print(f"      SHDI : {n_shdi:,} single-class cells → 0.0 fill")

    # Round to 4 d.p. to keep shapefile attribute table readable
    for col in ["LSI", "LPI", "MENN", "SHDI"]:
        gdf[col] = gdf[col].round(4)

    print(f"\n{'='*55}")
    print("  LANDSCAPE METRICS SUMMARY  (URA Master Plan 2025)")
    print(f"{'='*55}")
    for col in ["LSI", "LPI", "MENN", "SHDI"]:
        s = gdf[col]
        print(f"  {col:<6}  mean={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")

    gdf.to_file(OUT_SHP)
    print(f"\nSaved → {OUT_SHP}")
    print(f"Columns : {[c for c in gdf.columns if c != 'geometry']}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raster_data, transform, pixel_res = fetch_lulc()
    gdf, gdf_proj                     = load_fishnet()
    results                           = compute_all_cells(
                                            gdf_proj, raster_data,
                                            transform, pixel_res
                                        )
    save_results(gdf, results)
